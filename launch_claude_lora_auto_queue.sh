#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON_BIN:-python}"
SELECTOR_SCRIPT="$ROOT/run_pointwise5answers_three_stage_pairwise_listwise_sft_v1.py"
MIX_SCRIPT="$ROOT/run_newnew_one_answer_trueval_three_stage_sft.py"
DEFAULT_OUTPUT_ROOT="/opt/dlami/nvme/cyl/autodl-tmp/JudgeStealer_outputs"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}"
LOG_ROOT="$OUTPUT_ROOT/claude_lora_auto_queue_logs"
STATUS_LOG="$LOG_ROOT/job_status.log"
POLL_SECONDS="${POLL_SECONDS:-30}"
GPU_MEMORY_USED_LIMIT_MB=1024
SKIP_JOBS="${SKIP_JOBS:-}"

if [[ "$#" -eq 0 ]]; then
  echo "usage: $0 <gpu_id> [gpu_id ...]" >&2
  exit 2
fi

ALLOWED_GPUS=("$@")
JOBS=(
  selector_alpaca_llama1b selector_gpt4all_llama1b
  selector_alpaca_qwen1p7b selector_gpt4all_qwen1p7b
  mixep10_alpaca_llama1b mixep10_gpt4all_llama1b
  mixep10_alpaca_qwen1p7b mixep10_gpt4all_qwen1p7b
)
declare -A GPU_WORKER_PIDS=()
declare -A GPU_WORKER_JOBS=()

if ! mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"; then
  echo "failed to create output directory: $OUTPUT_ROOT" >&2
  exit 1
fi
cd "$ROOT"

log_status() {
  { flock 9; echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "$STATUS_LOG"; } \
    9>"$LOG_ROOT/.status.lock"
}

check_output_storage() {
  local fs_type="" available="unknown"

  if command -v findmnt >/dev/null 2>&1; then
    fs_type="$(findmnt -n -o FSTYPE -T "$OUTPUT_ROOT" 2>/dev/null || true)"
  else
    log_status "WARNING findmnt unavailable; output filesystem type not checked"
  fi
  available="$(df -hP "$OUTPUT_ROOT" 2>/dev/null | awk 'NR == 2 {print $4}' || true)"
  [[ -n "$available" ]] || available="unknown"
  [[ -n "$fs_type" ]] || fs_type="unknown"

  log_status "STORAGE output_root=$OUTPUT_ROOT fstype=$fs_type available=$available"
  case "$fs_type" in
    nfs|nfs4)
      log_status "ERROR network filesystem output is not allowed output_root=$OUTPUT_ROOT fstype=$fs_type"
      exit 1
      ;;
  esac
}

check_output_storage

should_skip_job() {
  local candidate="$1" skipped
  for skipped in $SKIP_JOBS; do
    [[ "$skipped" == "$candidate" ]] && return 0
  done
  return 1
}

skip_configured_jobs() {
  local job
  while (( next_job_index < ${#JOBS[@]} )) && should_skip_job "${JOBS[$next_job_index]}"; do
    job="${JOBS[$next_job_index]}"
    log_status "SKIP configured job=$job"
    next_job_index=$((next_job_index + 1))
  done
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
  if ! processes="$(
    nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits 2>/dev/null
  )"; then
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

  case "$mode" in
    selector|mixep10) ;;
    *) return 2 ;;
  esac
  case "$dataset" in
    alpaca)
      train="$ROOT/data/alpaca/claude/train.json"
      val="$ROOT/data/alpaca/claude/val.json"
      ;;
    gpt4all)
      train="$ROOT/data/gpt4all/claude/train.json"
      val="$ROOT/data/gpt4all/claude/val.json"
      ;;
    *) return 2 ;;
  esac
  case "$model_key" in
    llama1b)
      model_tag=llama3p2_1b
      model="$ROOT/models/Llama-3.2-1b-instruct"
      ;;
    qwen1p7b)
      model_tag=qwen3_1p7b
      model="$ROOT/models/Qwen3-1.7B"
      ;;
    *) return 2 ;;
  esac

  case "$mode" in
    selector)
      name="${model_tag}_${dataset}_claude_b600_lora_selector_smooth_a010_pool100_stage4stratfull"
      script="$SELECTOR_SCRIPT"
      ;;
    mixep10)
      name="${model_tag}_${dataset}_claude_b600_lora_trueval_mix200pw200pair200list_ep10_noreplay_nosmooth"
      script="$MIX_SCRIPT"
      ;;
  esac
  out="$OUTPUT_ROOT/$name"
  log="$LOG_ROOT/$name.log"
}

run_selector() {
  CUDA_VISIBLE_DEVICES="$1" \
  PYTHONUNBUFFERED=1 \
  TOKENIZERS_PARALLELISM=false \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" -u "$SELECTOR_SCRIPT" \
      --pointwise-5answers-dataset "$train" \
      --pairwise-eval-dataset "$val" \
      --listwise-eval-dataset "$val" \
      --llama "$model" \
      --seed 42 \
      --budget-units 600 \
      --stage2-pointwise-replay-ratio 0 \
      --stage3-pointwise-replay-ratio 0 \
      --stage3-pairwise-replay-ratio 0 \
      --stage4-replay-strategy stratified_triple \
      --stage4-replay-fraction 1 \
      --stage4-epochs 1 \
      --pointwise-epochs 1 \
      --pairwise-epochs 1 \
      --listwise-epochs 1 \
      --per-device-batch-size 1 \
      --gradient-accumulation-steps 16 \
      --learning-rate 1e-4 \
      --max-length 4096 \
      --eval-batch-size 1 \
      --eval-stages final \
      --use-lora \
      --load-in-4bit \
      --pointwise-global-smooth-alpha 0.1 \
      --pointwise-global-smooth-mode local_gaussian \
      --pointwise-global-smooth-gaussian-sigma 1.0 \
      --pointwise-global-smooth-stages all \
      --train-selection-mode candidate_triple_selector \
      --candidate-selector-kind bias_trap_pointwise \
      --candidate-selector-target-task pointwise \
      --candidate-selector-proxy-mode lm_head \
      --reuse-selection-proxy-for-stage1 \
      --candidate-selector-init-triples 80 \
      --candidate-selector-batch-size 20 \
      --candidate-selector-max-score-candidates 100 \
      --candidate-selector-one-per-question \
      --candidate-selector-proxy-warmup-epochs 3 \
      --candidate-selector-proxy-update-epochs 1 \
      --candidate-selector-exploration-ratio 0 \
      --candidate-selector-diversity-weight 1 \
      --candidate-selector-uncertainty-weight 0.25 \
      --candidate-selector-bias-weight 1 \
      --candidate-selector-pointwise-length-bias-weight 0.5 \
      --candidate-selector-pairwise-position-bias-weight 0.5 \
      --candidate-selector-pairwise-position-pairs 1 \
      --candidate-selector-pairwise-position-bias-scale 0.02 \
      --candidate-selector-signal-normalization none \
      --candidate-selector-uncertainty-view pointwise \
      --candidate-selector-density-k 10 \
      --candidate-selector-embedding-model BAAI/bge-small-en-v1.5 \
      --candidate-selector-embedding-max-length 512 \
      --candidate-selector-embedding-batch-size 64 \
      --candidate-selector-embedding-pooling cls \
      --candidate-selector-diversity-view pointwise \
      --proxy-lr 1e-4 \
      --proxy-max-length 768 \
      --out "$out"
}

run_mixep10() {
  CUDA_VISIBLE_DEVICES="$1" \
  PYTHONUNBUFFERED=1 \
  TOKENIZERS_PARALLELISM=false \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" -u "$MIX_SCRIPT" \
      --mode trueval_three_stage \
      --pointwise-5answers-dataset "$train" \
      --pairwise-val-dataset "$val" \
      --listwise-val-dataset "$val" \
      --llama "$model" \
      --out "$out" \
      --seed 42 \
      --budget 600 \
      --pointwise-train-samples 200 \
      --pairwise-train-pairs 200 \
      --listwise-train-examples 200 \
      --pointwise-epochs 10 \
      --pairwise-epochs 10 \
      --listwise-epochs 10 \
      --stage2-pointwise-replay-ratio 0 \
      --stage3-pointwise-replay-ratio 0 \
      --stage3-pairwise-replay-ratio 0 \
      --per-device-batch-size 1 \
      --gradient-accumulation-steps 16 \
      --learning-rate 1e-4 \
      --max-length 4096 \
      --eval-batch-size 1 \
      --eval-stages final \
      --use-lora \
      --load-in-4bit
}

run_job() {
  local job="$1" gpu="$2"
  local mode="" dataset="" model_key="" model_tag="" model="" train="" val="" script="" name="" out="" log=""
  local rc

  if ! resolve_job "$job"; then
    log_status "ERROR unknown job=$job"
    return 2
  fi
  if ! command -v "$PY" >/dev/null 2>&1; then
    log_status "ERROR python not found=$PY job=$job"
    return 1
  fi
  if [[ ! -d "$model" ]]; then
    log_status "ERROR missing directory=$model job=$job"
    return 1
  fi
  for required_file in "$script" "$model/config.json" "$train" "$val"; do
    if [[ ! -f "$required_file" ]]; then
      log_status "ERROR missing file=$required_file job=$job"
      return 1
    fi
  done
  if [[ -f "$out/metrics_compact.json" ]]; then
    log_status "SKIP completed job=$job out=$out"
    return 0
  fi
  if ps -eo args --cols 4096 | grep -F -- "--out $out" | grep -v grep >/dev/null 2>&1; then
    log_status "SKIP running job=$job out=$out"
    return 0
  fi
  if [[ -e "$out" ]]; then
    log_status "ERROR incomplete output exists job=$job out=$out"
    return 1
  fi

  log_status "START job=$job gpu=$gpu out=$out"
  case "$mode" in
    selector)
      run_selector "$gpu" >"$log" 2>&1
      ;;
    mixep10)
      run_mixep10 "$gpu" >"$log" 2>&1
      ;;
  esac
  rc=$?

  if [[ "$rc" -eq 0 && -f "$out/metrics_compact.json" ]]; then
    log_status "DONE job=$job gpu=$gpu out=$out"
    return 0
  fi
  if [[ "$rc" -eq 0 ]]; then
    rc=1
  fi
  log_status "ERROR job=$job gpu=$gpu rc=$rc out=$out log=$log"
  return "$rc"
}

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required" >&2
  exit 2
fi
if ! command -v flock >/dev/null 2>&1; then
  echo "flock is required" >&2
  exit 2
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
      if wait "$pid"; then
        rc=0
      else
        rc=$?
        overall_rc=1
      fi
      log_status "WORKER_EXIT job=${GPU_WORKER_JOBS[$gpu]} gpu=$gpu rc=$rc"
      unset 'GPU_WORKER_PIDS[$gpu]'
      unset 'GPU_WORKER_JOBS[$gpu]'
      active_workers=$((active_workers - 1))
    fi
  done

  for gpu in "${ALLOWED_GPUS[@]}"; do
    skip_configured_jobs
    (( next_job_index < ${#JOBS[@]} )) || break
    [[ -z "${GPU_WORKER_PIDS[$gpu]+assigned}" ]] || continue
    gpu_is_idle "$gpu" || continue
    job="${JOBS[$next_job_index]}"
    gpu_is_idle "$gpu" || continue
    run_job "$job" "$gpu" &
    GPU_WORKER_PIDS[$gpu]=$!
    GPU_WORKER_JOBS[$gpu]="$job"
    active_workers=$((active_workers + 1))
    next_job_index=$((next_job_index + 1))
  done

  if (( next_job_index < ${#JOBS[@]} || active_workers > 0 )); then
    sleep "$POLL_SECONDS"
  fi
done
exit "$overall_rc"
