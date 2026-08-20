#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON_BIN:-python}"
TRAIN_SCRIPT="$ROOT/run_rewardmodel_three_stage_sft.py"
PREPARE_SCRIPT="$ROOT/prepare_rewardmodel_three_stage.py"
DEFAULT_OUTPUT_ROOT="/opt/dlami/nvme/cyl/autodl-tmp/JudgeStealer_outputs"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}"
LOG_ROOT="$OUTPUT_ROOT/rewardmodel_lora_auto_queue_logs"
STATUS_LOG="$LOG_ROOT/job_status.log"
POLL_SECONDS="${POLL_SECONDS:-30}"
SKIP_JOBS="${SKIP_JOBS:-}"
GPU_MEMORY_USED_LIMIT_MB=1024

DEFAULT_REWARDMODEL_SOURCE="$ROOT/data/reward-model"
REWARDMODEL_SOURCE="${REWARDMODEL_SOURCE:-$DEFAULT_REWARDMODEL_SOURCE}"
ALPACA_REWARDMODEL_SOURCE="${ALPACA_REWARDMODEL_SOURCE:-}"
GPT4ALL_REWARDMODEL_SOURCE="${GPT4ALL_REWARDMODEL_SOURCE:-}"

if [[ "$#" -eq 0 ]]; then
  echo "usage: $0 <gpu_id> [gpu_id ...]" >&2
  exit 2
fi
DATASET_MODE="single"
if [[ -n "$ALPACA_REWARDMODEL_SOURCE" || -n "$GPT4ALL_REWARDMODEL_SOURCE" ]]; then
  if [[ -z "$ALPACA_REWARDMODEL_SOURCE" || -z "$GPT4ALL_REWARDMODEL_SOURCE" ]]; then
    echo "set both ALPACA_REWARDMODEL_SOURCE and GPT4ALL_REWARDMODEL_SOURCE" >&2
    exit 2
  fi
  DATASET_MODE="matrix"
fi

ALLOWED_GPUS=("$@")
ALL_JOBS=(
  naive_alpaca_llama1b ours_alpaca_llama1b
  naive_gpt4all_llama1b ours_gpt4all_llama1b
  naive_alpaca_qwen1p7b ours_alpaca_qwen1p7b
  naive_gpt4all_qwen1p7b ours_gpt4all_qwen1p7b
)
if [[ "$DATASET_MODE" == "matrix" ]]; then
  JOBS=("${ALL_JOBS[@]}")
else
  JOBS=(
    naive_rewardmodel_llama1b ours_rewardmodel_llama1b
    naive_rewardmodel_qwen1p7b ours_rewardmodel_qwen1p7b
  )
fi
declare -A GPU_WORKER_PIDS=()
declare -A GPU_WORKER_JOBS=()

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
cd "$ROOT"

log_status() {
  { flock 9; echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "$STATUS_LOG"; } \
    9>"$LOG_ROOT/.status.lock"
}

check_output_storage() {
  local fs_type="" available="unknown"
  if command -v findmnt >/dev/null 2>&1; then
    fs_type="$(findmnt -n -o FSTYPE -T "$OUTPUT_ROOT" 2>/dev/null || true)"
  fi
  available="$(df -hP "$OUTPUT_ROOT" 2>/dev/null | awk 'NR == 2 {print $4}' || true)"
  [[ -n "$available" ]] || available="unknown"
  [[ -n "$fs_type" ]] || fs_type="unknown"
  log_status "STORAGE output_root=$OUTPUT_ROOT fstype=$fs_type available=$available"
  case "$fs_type" in
    nfs|nfs4)
      log_status "ERROR network filesystem output is not allowed output_root=$OUTPUT_ROOT"
      exit 1
      ;;
  esac
}

prepare_source() {
  local source="$1" kind path
  for kind in pointwise pairwise listwise; do
    path="$source/$kind.json"
    if [[ ! -f "$path" ]]; then
      log_status "ERROR missing reward-model source file=$path"
      return 1
    fi
  done
  local required=(
    "$source/split1500_500/pointwise_train1500.json"
    "$source/split1500_500/pairwise_train1500.json"
    "$source/split1500_500/listwise_train1500.json"
    "$source/mix200_eval300/pointwise_train200.json"
    "$source/mix200_eval300/pairwise_train200.json"
    "$source/mix200_eval300/listwise_train200.json"
    "$source/mix200_eval300/pointwise_eval300.json"
    "$source/mix200_eval300/pairwise_eval300.json"
    "$source/mix200_eval300/listwise_eval300.json"
  )
  for path in "${required[@]}"; do
    [[ -f "$path" ]] || break
  done
  if [[ -f "$path" ]]; then
    log_status "PREPARED source=$source"
    return 0
  fi
  log_status "PREPARE source=$source seed=42 train=1500 mix=200 eval=300"
  "$PY" "$PREPARE_SCRIPT" \
    --source "$source" --seed 42 --train-size 1500 --mix-size 200 --eval-size 300
}

should_skip_job() {
  local candidate="$1" skipped
  for skipped in $SKIP_JOBS; do
    [[ "$skipped" == "$candidate" ]] && return 0
  done
  return 1
}

gpu_uuid() {
  nvidia-smi -i "$1" --query-gpu=uuid --format=csv,noheader,nounits 2>/dev/null \
    | awk '{$1=$1; print}'
}

gpu_memory_used_mb() {
  nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk '{$1=$1; print}'
}

gpu_has_compute_process() {
  local uuid="$1" processes
  if ! processes="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits 2>/dev/null)"; then
    return 0
  fi
  awk '{$1=$1; print}' <<<"$processes" | grep -Fx -- "$uuid" >/dev/null 2>&1
}

gpu_is_idle() {
  local gpu="$1" uuid used
  uuid="$(gpu_uuid "$gpu")" || return 1
  used="$(gpu_memory_used_mb "$gpu")" || return 1
  [[ -n "$uuid" && "$used" =~ ^[0-9]+$ ]] || return 1
  (( used <= GPU_MEMORY_USED_LIMIT_MB )) || return 1
  ! gpu_has_compute_process "$uuid"
}

resolve_job() {
  local job="$1"
  IFS=_ read -r mode dataset model_key <<<"$job"
  case "$dataset" in
    alpaca) source="$ALPACA_REWARDMODEL_SOURCE" ;;
    gpt4all) source="$GPT4ALL_REWARDMODEL_SOURCE" ;;
    rewardmodel) source="$REWARDMODEL_SOURCE" ;;
    *) return 2 ;;
  esac
  case "$model_key" in
    llama1b)
      model_tag="llama3p2_1b"
      model="$ROOT/models/Llama-3.2-1b-instruct"
      ;;
    qwen1p7b)
      model_tag="qwen3_1p7b"
      model="$ROOT/models/Qwen3-1.7B"
      ;;
    *) return 2 ;;
  esac
  case "$mode" in
    naive)
      name="${model_tag}_${dataset}_rewardmodel_lora_naive_mix200x3_ep10_softties"
      train_root="$source/mix200_eval300"
      ;;
    ours)
      name="${model_tag}_${dataset}_rewardmodel_lora_ours_mainselector_lmheadreuse_b600_orderaug_softties"
      train_root="$source/split1500_500"
      ;;
    *) return 2 ;;
  esac
  eval_root="$source/mix200_eval300"
  out="$OUTPUT_ROOT/$name"
  log="$LOG_ROOT/$name.log"
}

common_args() {
  COMMON_ARGS=(
    --target-format converted
    --pointwise-eval "$eval_root/pointwise_eval300.json"
    --pairwise-eval "$eval_root/pairwise_eval300.json"
    --listwise-eval "$eval_root/listwise_eval300.json"
    --llama "$model"
    --out "$out"
    --seed 42
    --tie-policy soft
    --per-device-batch-size 1
    --gradient-accumulation-steps 16
    --learning-rate 1e-4
    --max-length 4096
    --eval-batch-size 1
    --use-lora
    --load-in-4bit
  )
}

run_naive() {
  common_args
  "$PY" -u "$TRAIN_SCRIPT" \
    --mode mix \
    --pointwise-train "$train_root/pointwise_train200.json" \
    --pairwise-train "$train_root/pairwise_train200.json" \
    --listwise-train "$train_root/listwise_train200.json" \
    --pointwise-train-samples 200 \
    --pairwise-train-samples 200 \
    --listwise-train-samples 200 \
    --pointwise-epochs 10 \
    --pairwise-epochs 10 \
    --listwise-epochs 10 \
    --smooth-alpha 0 \
    "${COMMON_ARGS[@]}"
}

run_ours() {
  common_args
  "$PY" -u "$TRAIN_SCRIPT" \
    --mode three_signal_selector \
    --pointwise-train "$train_root/pointwise_train1500.json" \
    --pairwise-train "$train_root/pairwise_train1500.json" \
    --listwise-train "$train_root/listwise_train1500.json" \
    --budget-units 600 \
    --pointwise-epochs 1 \
    --pairwise-epochs 1 \
    --listwise-epochs 1 \
    --selector-init-questions 80 \
    --selector-batch-size 20 \
    --selector-pool-size 100 \
    --selector-proxy-warmup-epochs 3 \
    --selector-proxy-update-epochs 1 \
    --pairwise-order-augmentation \
    --listwise-order-augmentation \
    --smooth-alpha 0.1 \
    "${COMMON_ARGS[@]}"
}

run_job() {
  local job="$1" gpu="$2"
  local mode="" dataset="" model_key="" source="" model_tag="" model=""
  local name="" train_root="" eval_root="" out="" log="" rc
  if ! resolve_job "$job"; then
    log_status "ERROR unknown job=$job"
    return 2
  fi
  for required in "$TRAIN_SCRIPT" "$model/config.json"; do
    if [[ ! -f "$required" ]]; then
      log_status "ERROR missing file=$required job=$job"
      return 1
    fi
  done
  if [[ -f "$out/metrics_compact.json" ]]; then
    log_status "SKIP completed job=$job out=$out"
    return 0
  fi
  if [[ -e "$out" ]]; then
    log_status "ERROR incomplete output exists job=$job out=$out"
    return 1
  fi
  log_status "START job=$job gpu=$gpu out=$out"
  if [[ "$mode" == "naive" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True run_naive >"$log" 2>&1
  else
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True run_ours >"$log" 2>&1
  fi
  rc=$?
  if [[ "$rc" -eq 0 && -f "$out/metrics_compact.json" ]]; then
    log_status "DONE job=$job gpu=$gpu out=$out"
    return 0
  fi
  [[ "$rc" -ne 0 ]] || rc=1
  log_status "ERROR job=$job gpu=$gpu rc=$rc out=$out log=$log"
  return "$rc"
}

if ! command -v "$PY" >/dev/null 2>&1; then
  echo "python not found: $PY" >&2
  exit 2
fi
for required in "$TRAIN_SCRIPT" "$PREPARE_SCRIPT"; do
  [[ -f "$required" ]] || { echo "missing file: $required" >&2; exit 2; }
done
for command_name in nvidia-smi flock; do
  command -v "$command_name" >/dev/null 2>&1 || { echo "$command_name is required" >&2; exit 2; }
done

check_output_storage
if [[ "$DATASET_MODE" == "matrix" ]]; then
  prepare_source "$ALPACA_REWARDMODEL_SOURCE" || exit 1
  prepare_source "$GPT4ALL_REWARDMODEL_SOURCE" || exit 1
else
  prepare_source "$REWARDMODEL_SOURCE" || exit 1
fi
for gpu in "${ALLOWED_GPUS[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ ]] || [[ -z "$(gpu_uuid "$gpu")" ]]; then
    echo "invalid GPU id: $gpu" >&2
    exit 2
  fi
done

next_job_index=0
active_workers=0
overall_rc=0
while (( next_job_index < ${#JOBS[@]} || active_workers > 0 )); do
  for gpu in "${!GPU_WORKER_PIDS[@]}"; do
    pid="${GPU_WORKER_PIDS[$gpu]}"
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" || overall_rc=1
      unset 'GPU_WORKER_PIDS[$gpu]'
      unset 'GPU_WORKER_JOBS[$gpu]'
      active_workers=$((active_workers - 1))
    fi
  done
  for gpu in "${ALLOWED_GPUS[@]}"; do
    [[ -z "${GPU_WORKER_PIDS[$gpu]+assigned}" ]] || continue
    while (( next_job_index < ${#JOBS[@]} )) && should_skip_job "${JOBS[$next_job_index]}"; do
      log_status "SKIP configured job=${JOBS[$next_job_index]}"
      next_job_index=$((next_job_index + 1))
    done
    (( next_job_index < ${#JOBS[@]} )) || break
    gpu_is_idle "$gpu" || continue
    job="${JOBS[$next_job_index]}"
    run_job "$job" "$gpu" &
    GPU_WORKER_PIDS[$gpu]=$!
    GPU_WORKER_JOBS[$gpu]="$job"
    active_workers=$((active_workers + 1))
    next_job_index=$((next_job_index + 1))
  done
  (( next_job_index >= ${#JOBS[@]} && active_workers == 0 )) && break
  sleep "$POLL_SECONDS"
done

exit "$overall_rc"
