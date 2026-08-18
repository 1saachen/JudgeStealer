#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON_BIN:-python}"
SCRIPT="$ROOT/run_pointwise5answers_three_stage_pairwise_listwise_sft_v1.py"
MODEL="$ROOT/models/Qwen3-1.7B"
TRAIN_DATA="$ROOT/data/gpt4all/gpt5/train9k_pointwise_pairwise_no_val_overlap.json"
EVAL_DATA="$ROOT/data/gpt4all/gpt5/val3k_pairwise_listwise.json"
NAME="qwen3_1p7b_lora_seed42_gpt4all_ours_b600_selector_smooth_a010_pool100_stage4stratfull"
DEFAULT_OUTPUT_ROOT="/opt/dlami/nvme/cyl/autodl-tmp/JudgeStealer_outputs"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}"
RUN_ROOT="$OUTPUT_ROOT/qwen3_1p7b_ablation_seed42"
OUT="$RUN_ROOT/$NAME"
LOG_ROOT="$RUN_ROOT/logs"
STATUS_LOG="$LOG_ROOT/job_status.log"
LOG="$LOG_ROOT/$NAME.log"

if [[ "$#" -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 <gpu_id>" >&2
  exit 2
fi
GPU_ID="$1"

mkdir -p "$RUN_ROOT" "$LOG_ROOT"
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
      log_status "ERROR network filesystem output is not allowed output_root=$OUTPUT_ROOT fstype=$fs_type"
      exit 1
      ;;
  esac
}

check_output_storage

if ! command -v "$PY" >/dev/null 2>&1; then
  log_status "ERROR python not found=$PY"
  exit 1
fi
for required_file in "$SCRIPT" "$MODEL/config.json" "$TRAIN_DATA" "$EVAL_DATA"; do
  if [[ ! -f "$required_file" ]]; then
    log_status "ERROR missing file=$required_file"
    exit 1
  fi
done
if ! command -v nvidia-smi >/dev/null 2>&1; then
  log_status "ERROR nvidia-smi is required"
  exit 2
fi

if [[ -f "$OUT/metrics_compact.json" ]]; then
  log_status "SKIP completed out=$OUT"
  exit 0
fi
if [[ -e "$OUT" ]]; then
  log_status "ERROR incomplete output exists out=$OUT"
  exit 1
fi

exec {JOB_LOCK_FD}>"$LOG_ROOT/.job_gpt4all_ours.lock"
if ! flock -n "$JOB_LOCK_FD"; then
  log_status "SKIP locked out=$OUT"
  exec {JOB_LOCK_FD}>&-
  exit 0
fi

log_status "START gpu=$GPU_ID out=$OUT"
CUDA_VISIBLE_DEVICES="$GPU_ID" \
PYTHONUNBUFFERED=1 \
TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" -u "$SCRIPT" \
    --pointwise-5answers-dataset "$TRAIN_DATA" \
    --listwise-eval-dataset "$EVAL_DATA" \
    --llama "$MODEL" \
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
    --candidate-selector-finetune-mode lora \
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
    --out "$OUT" >"$LOG" 2>&1
rc=$?

if [[ "$rc" -eq 0 && -f "$OUT/metrics_compact.json" ]]; then
  log_status "DONE gpu=$GPU_ID out=$OUT"
  exec {JOB_LOCK_FD}>&-
  exit 0
fi
if [[ "$rc" -eq 0 ]]; then
  rc=1
fi
log_status "ERROR gpu=$GPU_ID rc=$rc out=$OUT log=$LOG"
exec {JOB_LOCK_FD}>&-
exit "$rc"
