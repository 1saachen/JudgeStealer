#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON_BIN:-python}"
SCRIPT="$ROOT/run_newnew_one_answer_trueval_three_stage_sft.py"
MODEL="$ROOT/models/Qwen3-1.7B"
DEFAULT_OUTPUT_ROOT="/opt/dlami/nvme/cyl/autodl-tmp/JudgeStealer_outputs"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}"
RUN_ROOT="$OUTPUT_ROOT/qwen3_1p7b_mix_budget_seed42"
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
  alpaca_mix_b0p5
  alpaca_mix_b1
  alpaca_mix_b2
  alpaca_mix_b5
  alpaca_mix_b10
  gpt4all_mix_b0p5
  gpt4all_mix_b1
  gpt4all_mix_b2
  gpt4all_mix_b5
  gpt4all_mix_b10
)
declare -A GPU_WORKER_PIDS=()
declare -A GPU_WORKER_JOBS=()

if ! mkdir -p "$OUTPUT_ROOT" "$RUN_ROOT" "$LOG_ROOT"; then
  echo "failed to create output directory: $RUN_ROOT" >&2
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
  dataset="${job%%_*}"
  train=""
  pairwise_val=""
  listwise_val=""

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
    *) return 2 ;;
  esac

  case "$job" in
    alpaca_mix_b0p5) budget_percent=0.5; candidate_queries=18000; sample_count=90; budget_units=270 ;;
    alpaca_mix_b1) budget_percent=1; candidate_queries=18000; sample_count=180; budget_units=540 ;;
    alpaca_mix_b2) budget_percent=2; candidate_queries=18000; sample_count=360; budget_units=1080 ;;
    alpaca_mix_b5) budget_percent=5; candidate_queries=18000; sample_count=900; budget_units=2700 ;;
    alpaca_mix_b10) budget_percent=10; candidate_queries=18000; sample_count=1800; budget_units=5400 ;;
    gpt4all_mix_b0p5) budget_percent=0.5; candidate_queries=8100; sample_count=40; budget_units=120 ;;
    gpt4all_mix_b1) budget_percent=1; candidate_queries=8100; sample_count=80; budget_units=240 ;;
    gpt4all_mix_b2) budget_percent=2; candidate_queries=8100; sample_count=160; budget_units=480 ;;
    gpt4all_mix_b5) budget_percent=5; candidate_queries=8100; sample_count=410; budget_units=1230 ;;
    gpt4all_mix_b10) budget_percent=10; candidate_queries=8100; sample_count=810; budget_units=2430 ;;
    *) return 2 ;;
  esac

  variant="${job#*_}"
  name="qwen3_1p7b_lora_seed42_${dataset}_${variant}_trueval_ep10_noreplay_nosmooth"
  out="$RUN_ROOT/$name"
  log="$LOG_ROOT/$name.log"
}

run_job() {
  local job="$1" gpu="$2"
  local dataset="" train="" pairwise_val="" listwise_val="" variant="" name="" out="" log=""
  local budget_percent=0 candidate_queries=0 sample_count=0 budget_units=0 rc
  local job_lock_fd

  if ! resolve_job "$job"; then
    log_status "ERROR unknown job=$job"
    return 2
  fi
  if ! command -v "$PY" >/dev/null 2>&1; then
    log_status "ERROR python not found=$PY job=$job"
    return 1
  fi
  if [[ ! -d "$MODEL" ]]; then
    log_status "ERROR missing directory=$MODEL job=$job"
    return 1
  fi
  for required_file in "$SCRIPT" "$MODEL/config.json" "$train" "$pairwise_val" "$listwise_val"; do
    if [[ ! -f "$required_file" ]]; then
      log_status "ERROR missing file=$required_file job=$job"
      return 1
    fi
  done

  exec {job_lock_fd}>"$LOG_ROOT/.job_${job}.lock"
  if ! flock -n "$job_lock_fd"; then
    log_status "SKIP locked job=$job out=$out"
    exec {job_lock_fd}>&-
    return 0
  fi
  if [[ -f "$out/metrics_compact.json" ]]; then
    log_status "SKIP completed job=$job out=$out"
    exec {job_lock_fd}>&-
    return 0
  fi
  if [[ -e "$out" ]]; then
    log_status "ERROR incomplete output exists job=$job out=$out"
    exec {job_lock_fd}>&-
    return 1
  fi

  mkdir -p "$out"
  printf '{"dataset":"%s","budget_percent":%s,"candidate_queries":%s,"query_budget":%s,"budget_units":%s,"pointwise_train_samples":%s,"pairwise_train_pairs":%s,"listwise_train_examples":%s}\n' \
    "$dataset" "$budget_percent" "$candidate_queries" "$((sample_count))" "$budget_units" "$sample_count" "$sample_count" "$sample_count" \
    >"$out/mix_budget_resolution.json"

  log_status "START job=$job gpu=$gpu B=${budget_percent}% samples=$sample_count out=$out"
  CUDA_VISIBLE_DEVICES="$gpu" \
  PYTHONUNBUFFERED=1 \
  TOKENIZERS_PARALLELISM=false \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" -u "$SCRIPT" \
      --mode trueval_three_stage \
      --pointwise-5answers-dataset "$train" \
      --pairwise-val-dataset "$pairwise_val" \
      --listwise-val-dataset "$listwise_val" \
      --llama "$MODEL" \
      --out "$out" \
      --seed 42 \
      --budget "$budget_units" \
      --pointwise-train-samples "$sample_count" \
      --pairwise-train-pairs "$sample_count" \
      --listwise-train-examples "$sample_count" \
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
      --load-in-4bit \
      >"$log" 2>&1
  rc=$?

  if [[ "$rc" -eq 0 && -f "$out/metrics_compact.json" ]]; then
    log_status "DONE job=$job gpu=$gpu out=$out"
    exec {job_lock_fd}>&-
    return 0
  fi
  if [[ "$rc" -eq 0 ]]; then
    rc=1
  fi
  log_status "ERROR job=$job gpu=$gpu rc=$rc out=$out log=$log"
  exec {job_lock_fd}>&-
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
