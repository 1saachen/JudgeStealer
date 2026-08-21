#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON_BIN:-python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
SCRIPT="$ROOT/run_pointwise5answers_three_stage_pairwise_listwise_sft_v1.py"
MODEL_DIR="${MODEL_DIR:-$ROOT/models/Qwen3-32B}"
MODEL_TAG="${MODEL_TAG:-qwen3_32b}"
DEFAULT_OUTPUT_ROOT="/opt/dlami/nvme/cyl/autodl-tmp/JudgeStealer_outputs"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}"
LOG_ROOT="$OUTPUT_ROOT/${MODEL_TAG}_gpt5_fullft_fsdp_logs"
STATUS_LOG="$LOG_ROOT/job_status.log"
POLL_SECONDS="${POLL_SECONDS:-30}"
GPU_MEMORY_USED_LIMIT_MB=1024
SKIP_JOBS="${SKIP_JOBS:-}"

if [[ "$#" -ne 4 ]]; then
  echo "usage: $0 <gpu_id_a> <gpu_id_b> <gpu_id_c> <gpu_id_d>" >&2
  exit 2
fi
GPU_IDS=("$1" "$2" "$3" "$4")
JOBS=(alpaca gpt4all)

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT" || exit 1
cd "$ROOT"

log_status() {
  { flock 9; echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "$STATUS_LOG"; } \
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

resolve_job() {
  local job="$1"
  dataset="$job"
  case "$job" in
    alpaca|gpt4all) ;;
    *) return 2 ;;
  esac
  if [[ "$dataset" == alpaca ]]; then
    train="$ROOT/data/alpaca/gpt5/train-20k.json"
    eval="$ROOT/data/alpaca/gpt5/val-2k-eval-listwise.json"
  else
    train="$ROOT/data/gpt4all/gpt5/train9k_pointwise_pairwise_no_val_overlap.json"
    eval="$ROOT/data/gpt4all/gpt5/val3k_pairwise_listwise.json"
  fi
  name="${MODEL_TAG}_${dataset}_gpt5_b600_fullft_selector_smooth_a010_pool100_stage4stratfull"
  out="$OUTPUT_ROOT/$name"
  log="$LOG_ROOT/$name.log"
}

run_job() {
  local job="$1" dataset train eval name out log required_file rc
  resolve_job "$job" || { log_status "ERROR unknown job=$job"; return 2; }
  if ! command -v "$PY" >/dev/null 2>&1 || ! command -v "$TORCHRUN_BIN" >/dev/null 2>&1; then
    log_status "ERROR python_or_torchrun_missing job=$job python=$PY torchrun=$TORCHRUN_BIN"
    return 1
  fi
  for required_file in "$SCRIPT" "$MODEL_DIR/config.json" "$train" "$eval"; do
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

  log_status "START job=$job gpus=${GPU_IDS[*]} out=$out"
  CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${GPU_IDS[*]}")" PYTHONUNBUFFERED=1 \
  TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$TORCHRUN_BIN" --standalone --nproc_per_node=4 "$SCRIPT" \
      --pointwise-5answers-dataset "$train" \
      --listwise-eval-dataset "$eval" \
      --llama "$MODEL_DIR" \
      --seed 42 --budget-units 600 \
      --stage2-pointwise-replay-ratio 0 --stage3-pointwise-replay-ratio 0 \
      --stage3-pairwise-replay-ratio 0 \
      --stage4-replay-strategy stratified_triple --stage4-replay-fraction 1 \
      --stage4-epochs 1 --pointwise-epochs 1 --pairwise-epochs 1 --listwise-epochs 1 \
      --per-device-batch-size 1 --gradient-accumulation-steps 16 \
      --learning-rate 1e-5 --max-length 4096 --eval-batch-size 1 --eval-stages final \
      --fsdp "full_shard auto_wrap" \
      --fsdp-transformer-layer-cls-to-wrap Qwen3DecoderLayer \
      --fsdp-activation-checkpointing --fsdp-use-orig-params \
      --fsdp-state-dict-type FULL_STATE_DICT \
      --pointwise-global-smooth-alpha 0.1 --pointwise-global-smooth-mode local_gaussian \
      --pointwise-global-smooth-gaussian-sigma 1.0 --pointwise-global-smooth-stages all \
      --train-selection-mode candidate_triple_selector \
      --candidate-selector-kind bias_trap_pointwise --candidate-selector-target-task pointwise \
      --candidate-selector-proxy-mode lm_head --candidate-selector-finetune-mode lora \
      --candidate-selector-load-in-4bit \
      --candidate-selector-init-triples 80 \
      --candidate-selector-batch-size 20 --candidate-selector-max-score-candidates 100 \
      --candidate-selector-one-per-question --candidate-selector-proxy-warmup-epochs 3 \
      --candidate-selector-proxy-update-epochs 1 --candidate-selector-exploration-ratio 0 \
      --candidate-selector-diversity-weight 1 --candidate-selector-uncertainty-weight 0.25 \
      --candidate-selector-bias-weight 1 --candidate-selector-pointwise-length-bias-weight 0.5 \
      --candidate-selector-pairwise-position-bias-weight 0.5 --candidate-selector-pairwise-position-pairs 1 \
      --candidate-selector-pairwise-position-bias-scale 0.02 --candidate-selector-signal-normalization none \
      --candidate-selector-uncertainty-view pointwise --candidate-selector-density-k 10 \
      --candidate-selector-embedding-model BAAI/bge-small-en-v1.5 \
      --candidate-selector-embedding-max-length 512 --candidate-selector-embedding-batch-size 64 \
      --candidate-selector-embedding-pooling cls --candidate-selector-diversity-view pointwise \
      --proxy-lr 1e-5 --proxy-max-length 768 --out "$out" >"$log" 2>&1
  rc=$?
  if [[ "$rc" -eq 0 && -f "$out/metrics_compact.json" ]]; then
    log_status "DONE job=$job gpus=${GPU_IDS[*]} out=$out"
    return 0
  fi
  [[ "$rc" -ne 0 ]] || rc=1
  log_status "ERROR job=$job gpus=${GPU_IDS[*]} rc=$rc out=$out log=$log"
  return "$rc"
}

check_output_storage
command -v nvidia-smi >/dev/null 2>&1 || { echo "nvidia-smi is required" >&2; exit 2; }
command -v flock >/dev/null 2>&1 || { echo "flock is required" >&2; exit 2; }
for gpu in "${GPU_IDS[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ ]] || [[ -z "$(gpu_uuid "$gpu")" ]]; then
    echo "invalid GPU id: $gpu" >&2
    exit 2
  fi
done
declare -A SEEN_GPU_IDS=()
for gpu in "${GPU_IDS[@]}"; do
  if [[ -n "${SEEN_GPU_IDS[$gpu]+seen}" ]]; then
    echo "four distinct GPU ids are required" >&2
    exit 2
  fi
  SEEN_GPU_IDS[$gpu]=1
done

all_gpus_are_idle() {
  local gpu
  for gpu in "${GPU_IDS[@]}"; do
    gpu_is_idle "$gpu" || return 1
  done
}

overall_rc=0
for job in "${JOBS[@]}"; do
  should_skip_job "$job" && { log_status "SKIP configured job=$job"; continue; }
  resolve_job "$job" || exit 1
  while ! all_gpus_are_idle; do
    log_status "WAIT job=$job gpus=${GPU_IDS[*]} reason=not_idle"
    sleep "$POLL_SECONDS"
  done
  if ! run_job "$job"; then
    overall_rc=1
  fi
done
exit "$overall_rc"
