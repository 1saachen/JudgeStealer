#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs}"
LOG_ROOT="$OUTPUT_ROOT/gpt5_cot_auto_queue_logs"
STATUS_LOG="$LOG_ROOT/job_status.log"
POLL_SECONDS="${POLL_SECONDS:-30}"
GPU_MEMORY_USED_LIMIT_MB=1024
SKIP_JOBS="${SKIP_JOBS:-}"

DATA_ROOT="${DATA_ROOT:-/data/model-extraction-attack/yaolin/JudgeStealer/data}"
ALPACA_COT_DATA_DIR="${ALPACA_COT_DATA_DIR:-$DATA_ROOT/Alpaca-cot-gpt}"
GPT4ALL_COT_DATA_DIR="${GPT4ALL_COT_DATA_DIR:-$DATA_ROOT/GPT4All-cot-gpt}"
ALPACA_COT_PREPARED_DIR="${ALPACA_COT_PREPARED_DIR:-$(dirname "$ALPACA_COT_DATA_DIR")/prepared_4066}"
GPT4ALL_COT_PREPARED_DIR="${GPT4ALL_COT_PREPARED_DIR:-$(dirname "$GPT4ALL_COT_DATA_DIR")/prepared_4066}"

if [[ "$#" -eq 0 ]]; then
  echo "usage: $0 <gpu_id> [gpu_id ...]" >&2
  exit 2
fi

ALLOWED_GPUS=("$@")
ALL_JOBS=(
  alpaca_llama_naive alpaca_llama_ours
  alpaca_qwen_naive alpaca_qwen_ours
  gpt4all_llama_naive gpt4all_llama_ours
  gpt4all_qwen_naive gpt4all_qwen_ours
)
JOBS=("${ALL_JOBS[@]}")
declare -A GPU_WORKER_PIDS=()
declare -A GPU_WORKER_JOBS=()

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT" || exit 1
cd "$ROOT"

log_status() {
  { flock 9; printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$STATUS_LOG"; } \
    9>"$LOG_ROOT/.status.lock"
}

check_output_storage() {
  local fs_type available
  fs_type="$(findmnt -n -o FSTYPE -T "$OUTPUT_ROOT" 2>/dev/null || true)"
  available="$(df -hP "$OUTPUT_ROOT" 2>/dev/null | awk 'NR == 2 {print $4}' || true)"
  [[ -n "$fs_type" ]] || fs_type=unknown
  [[ -n "$available" ]] || available=unknown
  log_status "STORAGE output_root=$OUTPUT_ROOT fstype=$fs_type available=$available"
  case "$fs_type" in
    nfs|nfs4)
      log_status "ERROR network filesystem output is not allowed output_root=$OUTPUT_ROOT fstype=$fs_type"
      exit 1
      ;;
  esac
}

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
  processes="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits 2>/dev/null || true)"
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

prepare_dataset() {
  local data_dir="$1" prepared_dir="$2" dataset
  local train="$data_dir/train_pointwise_8k.json"
  local validation="$data_dir/test_2k.json"
  dataset="$(basename "$data_dir")"

  if [[ -f "$prepared_dir/manifest.json" ]]; then
    return 0
  fi
  if [[ ! -f "$train" || ! -f "$validation" ]]; then
    log_status "ERROR missing CoT source dataset=$dataset train=$train validation=$validation"
    return 1
  fi
  if [[ -e "$prepared_dir" ]]; then
    log_status "ERROR incomplete prepared directory dataset=$dataset dir=$prepared_dir"
    return 1
  fi

  log_status "PREPARE dataset=$dataset out=$prepared_dir"
  "$PY" "$ROOT/prepare_alpaca_cot_4066.py" \
    --train "$train" \
    --validation "$validation" \
    --output-dir "$prepared_dir" \
    --seed 42 \
    --mix-questions 200
}

run_job() {
  local job="$1" gpu="$2"
  local dataset surrogate paper_method mode model prepared_dir name out log rc required_file

  case "$job" in
    alpaca_llama_naive) dataset=alpaca; surrogate=llama3_1b; paper_method=naive; model="$ROOT/llama/Llama-3.2-1B-Instruct" ;;
    alpaca_llama_ours) dataset=alpaca; surrogate=llama3_1b; paper_method=ours; model="$ROOT/llama/Llama-3.2-1B-Instruct" ;;
    alpaca_qwen_naive) dataset=alpaca; surrogate=qwen3_1p7b; paper_method=naive; model="$ROOT/qwen/Qwen3-1.7B" ;;
    alpaca_qwen_ours) dataset=alpaca; surrogate=qwen3_1p7b; paper_method=ours; model="$ROOT/qwen/Qwen3-1.7B" ;;
    gpt4all_llama_naive) dataset=gpt4all; surrogate=llama3_1b; paper_method=naive; model="$ROOT/llama/Llama-3.2-1B-Instruct" ;;
    gpt4all_llama_ours) dataset=gpt4all; surrogate=llama3_1b; paper_method=ours; model="$ROOT/llama/Llama-3.2-1B-Instruct" ;;
    gpt4all_qwen_naive) dataset=gpt4all; surrogate=qwen3_1p7b; paper_method=naive; model="$ROOT/qwen/Qwen3-1.7B" ;;
    gpt4all_qwen_ours) dataset=gpt4all; surrogate=qwen3_1p7b; paper_method=ours; model="$ROOT/qwen/Qwen3-1.7B" ;;
    *) log_status "ERROR unknown job=$job"; return 2 ;;
  esac

  case "$paper_method" in
    naive) mode=mix ;;
    ours) mode=stage4 ;;
    *) log_status "ERROR unknown paper method=$paper_method"; return 2 ;;
  esac

  if [[ "$dataset" == alpaca ]]; then
    prepared_dir="$ALPACA_COT_PREPARED_DIR"
  else
    prepared_dir="$GPT4ALL_COT_PREPARED_DIR"
  fi
  if [[ ! -f "$model/config.json" ]]; then
    log_status "ERROR missing model job=$job model=$model"
    return 1
  fi
  for required_file in \
    "$prepared_dir/train_questions_4066.json" \
    "$prepared_dir/eval_questions_1800.json" \
    "$prepared_dir/mix_pointwise_train_200.json" \
    "$prepared_dir/mix_pairwise_train_200.json" \
    "$prepared_dir/mix_listwise_train_200.json"; do
    if [[ ! -f "$required_file" ]]; then
      log_status "ERROR missing prepared file=$required_file job=$job"
      return 1
    fi
  done

  name="${surrogate}_${dataset}_gpt5_cot_${paper_method}_seed42"
  out="$OUTPUT_ROOT/$name"
  log="$LOG_ROOT/$name.log"
  if [[ -f "$out/metrics_compact.json" ]]; then
    log_status "SKIP completed job=$job out=$out"
    return 0
  fi
  if ps -eo args --cols 4096 2>/dev/null | grep -F -- "--out $out" | grep -v grep >/dev/null 2>&1; then
    log_status "SKIP running job=$job out=$out"
    return 0
  fi
  if [[ -e "$out" ]]; then
    log_status "ERROR incomplete output exists job=$job out=$out"
    return 1
  fi

  log_status "START job=$job gpu=$gpu mode=$mode out=$out"
  CUDA_VISIBLE_DEVICES="$gpu" \
  PYTHONUNBUFFERED=1 \
  TOKENIZERS_PARALLELISM=false \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" -u "$ROOT/run_alpaca_cot_stage4_mix.py" \
      --mode "$mode" \
      --llama "$model" \
      --out "$out" \
      --train-questions "$prepared_dir/train_questions_4066.json" \
      --eval-questions "$prepared_dir/eval_questions_1800.json" \
      --mix-pointwise "$prepared_dir/mix_pointwise_train_200.json" \
      --mix-pairwise "$prepared_dir/mix_pairwise_train_200.json" \
      --mix-listwise "$prepared_dir/mix_listwise_train_200.json" \
      --seed 42 \
      --per-device-batch-size 1 \
      --gradient-accumulation-steps 16 \
      --learning-rate 1e-4 \
      --max-length 4096 \
      --eval-batch-size 1 \
      --eval-max-new-tokens 192 \
      --use-lora \
      --load-in-4bit \
      >"$log" 2>&1
  rc=$?
  if [[ "$rc" -eq 0 && -f "$out/metrics_compact.json" ]]; then
    log_status "DONE job=$job gpu=$gpu out=$out"
    return 0
  fi
  [[ "$rc" -ne 0 ]] || rc=1
  log_status "ERROR job=$job gpu=$gpu rc=$rc out=$out log=$log"
  return "$rc"
}

check_output_storage
command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi is required" >&2; exit 2; }
command -v flock >/dev/null 2>&1 || { echo "flock is required" >&2; exit 2; }
command -v "$PY" >/dev/null 2>&1 || { echo "python not found: $PY" >&2; exit 2; }

for gpu in "${ALLOWED_GPUS[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ ]] || [[ -z "$(gpu_uuid "$gpu")" ]]; then
    echo "invalid GPU id: $gpu" >&2
    exit 2
  fi
done

prepare_dataset "$ALPACA_COT_DATA_DIR" "$ALPACA_COT_PREPARED_DIR" || exit 1
prepare_dataset "$GPT4ALL_COT_DATA_DIR" "$GPT4ALL_COT_PREPARED_DIR" || exit 1

next_job_index=0
active_workers=0
overall_rc=0
while (( next_job_index < ${#JOBS[@]} || active_workers > 0 )); do
  for gpu in "${!GPU_WORKER_PIDS[@]}"; do
    pid="${GPU_WORKER_PIDS[$gpu]}"
    if ! kill -0 "$pid" 2>/dev/null; then
      if wait "$pid"; then rc=0; else rc=$?; overall_rc=1; fi
      log_status "WORKER_EXIT job=${GPU_WORKER_JOBS[$gpu]} gpu=$gpu rc=$rc"
      unset 'GPU_WORKER_PIDS[$gpu]' 'GPU_WORKER_JOBS[$gpu]'
      active_workers=$((active_workers - 1))
    fi
  done
  for gpu in "${ALLOWED_GPUS[@]}"; do
    skip_configured_jobs
    (( next_job_index < ${#JOBS[@]} )) || break
    [[ -z "${GPU_WORKER_PIDS[$gpu]+assigned}" ]] || continue
    gpu_is_idle "$gpu" || continue
    job="${JOBS[$next_job_index]}"
    run_job "$job" "$gpu" &
    GPU_WORKER_PIDS[$gpu]=$!
    GPU_WORKER_JOBS[$gpu]="$job"
    active_workers=$((active_workers + 1))
    next_job_index=$((next_job_index + 1))
  done
  if (( next_job_index < ${#JOBS[@]} || active_workers > 0 )); then sleep "$POLL_SECONDS"; fi
done
exit "$overall_rc"
