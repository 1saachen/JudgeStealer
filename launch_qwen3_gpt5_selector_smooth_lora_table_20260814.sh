#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON_BIN:-python}"
SCRIPT="$ROOT/run_pointwise5answers_three_stage_pairwise_listwise_sft_v1.py"
DATE=20260814
RUN_TAG=table_lora_gpt5_v1
LOG_ROOT="$ROOT/outputs/qwen3_gpt5_selector_smooth_lora_logs_${DATE}_${RUN_TAG}"
STATUS_LOG="$LOG_ROOT/job_status.log"

GPU_ID="${1:?usage: $0 <gpu_id> <job> [job ...]}"
shift
if [[ "$#" -eq 0 ]]; then
  echo "at least one job is required" >&2
  exit 2
fi

mkdir -p "$LOG_ROOT"
cd "$ROOT"

log_status() {
  { flock 9; echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "$STATUS_LOG"; } \
    9>"$LOG_ROOT/.status.lock"
}

run_job() {
  local job="$1"
  local dataset model_tag model train eval out name log
  case "$job" in
    alpaca_0p6b)
      dataset=alpaca; model_tag=qwen3_0p6b; model="$ROOT/qwen/Qwen3-0.6B" ;;
    alpaca_4b)
      dataset=alpaca; model_tag=qwen3_4b; model="$ROOT/qwen/Qwen3-4B" ;;
    alpaca_8b)
      dataset=alpaca; model_tag=qwen3_8b; model="$ROOT/qwen/Qwen3-8B" ;;
    dolly_0p6b)
      dataset=dolly; model_tag=qwen3_0p6b; model="$ROOT/qwen/Qwen3-0.6B" ;;
    dolly_1p7b)
      dataset=dolly; model_tag=qwen3_1p7b; model="$ROOT/qwen/Qwen3-1.7B" ;;
    dolly_4b)
      dataset=dolly; model_tag=qwen3_4b; model="$ROOT/qwen/Qwen3-4B" ;;
    dolly_8b)
      dataset=dolly; model_tag=qwen3_8b; model="$ROOT/qwen/Qwen3-8B" ;;
    *)
      log_status "ERROR unknown job=$job"
      return 2
      ;;
  esac

  if [[ "$dataset" == alpaca ]]; then
    train="$ROOT/train_with_selector/train_with_selector/data/Alpaca/gpt5/train-20k.json"
    eval="$ROOT/train_with_selector/train_with_selector/data/Alpaca/gpt5/val-2k-eval-listwise.json"
  else
    train="$ROOT/train_with_selector/train_with_selector/data/Dolly/gpt5/train9k_pointwise_pairwise_no_val_overlap.json"
    eval="$ROOT/train_with_selector/train_with_selector/data/Dolly/gpt5/val3k_pairwise_listwise.json"
  fi

  name="${model_tag}_${dataset}_gpt5_b600_selector_smooth_a010_pool100_stage4stratfull_${DATE}_${RUN_TAG}"
  out="$ROOT/outputs/$name"
  log="$LOG_ROOT/$name.log"

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

  log_status "START job=$job gpu=$GPU_ID out=$out"
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  PYTHONUNBUFFERED=1 \
  TOKENIZERS_PARALLELISM=false \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" -u "$SCRIPT" \
      --pointwise-5answers-dataset "$train" \
      --listwise-eval-dataset "$eval" \
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
      --out "$out" >"$log" 2>&1
  local rc=$?
  if [[ "$rc" -eq 0 && -f "$out/metrics_compact.json" ]]; then
    log_status "DONE job=$job gpu=$GPU_ID out=$out"
  else
    log_status "ERROR job=$job gpu=$GPU_ID rc=$rc out=$out log=$log"
    return "$rc"
  fi
}

worker_rc=0
for job in "$@"; do
  run_job "$job" || worker_rc=1
done
exit "$worker_rc"
