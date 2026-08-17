#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON_BIN:-python}"
SCRIPT="$ROOT/run_pointwise5answers_three_stage_pairwise_listwise_sft_v1.py"
MODEL="$ROOT/models/Qwen3-8B"
TRAIN_DATA="$ROOT/data/gpt4all/gpt5/train9k_pointwise_pairwise_no_val_overlap.json"
EVAL_DATA="$ROOT/data/gpt4all/gpt5/val3k_pairwise_listwise.json"
NAME="qwen3_8b_gpt4all_gpt5_b600_selector_smooth_a010_pool100_stage4stratfull"
OUT="$ROOT/outputs/$NAME"
LOG_ROOT="$ROOT/outputs/qwen3_8b_gpt4all_gpt5_four_stage_logs"
STATUS_LOG="$LOG_ROOT/job_status.log"
LOG="$LOG_ROOT/$NAME.log"

GPU_ID="${1:?usage: $0 <gpu_id>}"
if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 <gpu_id>" >&2
  exit 2
fi

mkdir -p "$LOG_ROOT"
cd "$ROOT"

log_status() {
  { flock 9; echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "$STATUS_LOG"; } \
    9>"$LOG_ROOT/.status.lock"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    log_status "ERROR missing file=$1"
    exit 1
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    log_status "ERROR missing directory=$1"
    exit 1
  fi
}

if ! command -v "$PY" >/dev/null 2>&1; then
  log_status "ERROR python not found=$PY"
  exit 1
fi
require_file "$SCRIPT"
require_dir "$MODEL"
require_file "$MODEL/config.json"
require_file "$TRAIN_DATA"
require_file "$EVAL_DATA"

if [[ -f "$OUT/metrics_compact.json" ]]; then
  log_status "SKIP completed out=$OUT"
  exit 0
fi
if ps -eo args --cols 4096 | grep -F -- "--out $OUT" | grep -v grep >/dev/null 2>&1; then
  log_status "SKIP running out=$OUT"
  exit 0
fi
if [[ -e "$OUT" ]]; then
  log_status "ERROR incomplete output exists out=$OUT"
  exit 1
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
  exit 0
fi
if [[ "$rc" -eq 0 ]]; then
  rc=1
fi

log_status "ERROR gpu=$GPU_ID rc=$rc out=$OUT log=$LOG"
exit "$rc"
