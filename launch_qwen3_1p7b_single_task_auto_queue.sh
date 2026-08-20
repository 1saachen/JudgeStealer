#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON_BIN:-python}"
SCRIPT="$ROOT/run_newnew_one_answer_trueval_three_stage_sft.py"
MODEL="$ROOT/models/Qwen3-1.7B"
DEFAULT_OUTPUT_ROOT="/opt/dlami/nvme/cyl/autodl-tmp/JudgeStealer_outputs"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}"
RUN_ROOT="$OUTPUT_ROOT/qwen3_1p7b_single_task_seed42"
LOG_ROOT="$RUN_ROOT/logs"
STATUS_LOG="$LOG_ROOT/job_status.log"
POLL_SECONDS="${POLL_SECONDS:-30}"
GPU_MEMORY_USED_LIMIT_MB="${GPU_MEMORY_USED_LIMIT_MB:-1024}"
SKIP_JOBS="${SKIP_JOBS:-}"

if [[ "$#" -eq 0 ]]; then
  echo "usage: $0 <gpu_id> [gpu_id ...]" >&2
  exit 2
fi

ALLOWED_GPUS=("$@")
JOBS=(
  alpaca_pointwise_only
  alpaca_pairwise_only
  alpaca_listwise_only
  gpt4all_pointwise_only
  gpt4all_pairwise_only
  gpt4all_listwise_only
)
declare -A GPU_WORKER_PIDS=()
declare -A GPU_WORKER_JOBS=()

mkdir -p "$OUTPUT_ROOT" "$RUN_ROOT" "$LOG_ROOT" || exit 1
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
    nfs|nfs4) log_status "ERROR network filesystem output is not allowed output_root=$OUTPUT_ROOT fstype=$fs_type"; exit 1 ;;
  esac
}

should_skip_job() {
  local candidate="$1" skipped
  for skipped in $SKIP_JOBS; do [[ "$skipped" == "$candidate" ]] && return 0; done
  return 1
}

gpu_uuid() {
  nvidia-smi -i "$1" --query-gpu=uuid --format=csv,noheader,nounits 2>/dev/null | awk '{$1=$1; print}'
}

gpu_memory_used_mb() {
  nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{$1=$1; print}'
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

run_job() {
  local job="$1" gpu="$2" dataset single_task train pairwise_val listwise_val out log name rc required_file
  local job_lock_fd
  dataset="${job%%_*}"
  single_task="${job#${dataset}_}"
  single_task="${single_task%_only}"
  case "$dataset" in
    alpaca)
      train="$ROOT/data/alpaca/gpt5/train-20k.json"
      pairwise_val="$ROOT/data/alpaca/gpt5/val-2k-eval.json"
      listwise_val="$ROOT/data/alpaca/gpt5/val-2k-eval-listwise.json"
      ;;
    gpt4all)
      train="$ROOT/data/gpt4all/gpt5/train9k_pointwise_pairwise_no_val_overlap.json"
      pairwise_val="$ROOT/data/gpt4all/gpt5/val3k_pairwise_listwise.json"
      listwise_val="$ROOT/data/gpt4all/gpt5/val3k_pairwise_listwise.json"
      ;;
    *) log_status "ERROR unknown job=$job"; return 2 ;;
  esac
  name="qwen3_1p7b_${dataset}_gpt5_b600_lora_trueval_${single_task}_only_ep10_noreplay_nosmooth"
  out="$RUN_ROOT/$name"
  log="$LOG_ROOT/$name.log"
  for required_file in "$SCRIPT" "$MODEL/config.json" "$train" "$pairwise_val" "$listwise_val"; do
    [[ -f "$required_file" ]] || { log_status "ERROR missing file=$required_file job=$job"; return 1; }
  done
  exec {job_lock_fd}>"$LOG_ROOT/.job_${job}.lock"
  if ! flock -n "$job_lock_fd"; then log_status "SKIP locked job=$job out=$out"; exec {job_lock_fd}>&-; return 0; fi
  if [[ -f "$out/metrics_compact.json" ]]; then log_status "SKIP completed job=$job out=$out"; exec {job_lock_fd}>&-; return 0; fi
  if [[ -e "$out" ]]; then log_status "ERROR incomplete output exists job=$job out=$out"; exec {job_lock_fd}>&-; return 1; fi

  log_status "START job=$job gpu=$gpu task=$single_task out=$out"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" -u "$SCRIPT" \
      --mode single_task --single-task "$single_task" \
      --pointwise-5answers-dataset "$train" \
      --pairwise-val-dataset "$pairwise_val" \
      --listwise-val-dataset "$listwise_val" \
      --llama "$MODEL" --out "$out" --seed 42 --budget 600 \
      --pointwise-train-samples 600 --pairwise-train-pairs 600 --listwise-train-examples 600 \
      --pointwise-epochs 10 --pairwise-epochs 10 --listwise-epochs 10 \
      --per-device-batch-size 1 --gradient-accumulation-steps 16 --learning-rate 1e-4 \
      --max-length 4096 --eval-batch-size 1 --eval-stages final \
      --use-lora --load-in-4bit >"$log" 2>&1
  rc=$?
  if [[ "$rc" -eq 0 && -f "$out/metrics_compact.json" ]]; then log_status "DONE job=$job gpu=$gpu out=$out"; exec {job_lock_fd}>&-; return 0; fi
  [[ "$rc" -ne 0 ]] || rc=1
  log_status "ERROR job=$job gpu=$gpu rc=$rc out=$out log=$log"
  exec {job_lock_fd}>&-
  return "$rc"
}

check_output_storage
command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi is required" >&2; exit 2; }
command -v flock >/dev/null 2>&1 || { echo "flock is required" >&2; exit 2; }

for gpu in "${ALLOWED_GPUS[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ && -n "$(gpu_uuid "$gpu")" ]] || { echo "invalid GPU id: $gpu" >&2; exit 2; }
done

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
  while (( next_job_index < ${#JOBS[@]} )) && should_skip_job "${JOBS[$next_job_index]}"; do
    log_status "SKIP configured job=${JOBS[$next_job_index]}"
    next_job_index=$((next_job_index + 1))
  done
  for gpu in "${ALLOWED_GPUS[@]}"; do
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
