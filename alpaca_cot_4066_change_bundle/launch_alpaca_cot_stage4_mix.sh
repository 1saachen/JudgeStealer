#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON_BIN:-python}"
GPU_ID="${1:?usage: $0 <gpu_id> <model_path> [both|stage4|mix] [run_tag]}"
MODEL="${2:?usage: $0 <gpu_id> <model_path> [both|stage4|mix] [run_tag]}"
MODE="${3:-both}"
RUN_TAG="${4:-cot_proxy_v1}"

case "$MODE" in
  both|stage4|mix) ;;
  *) echo "mode must be one of: both, stage4, mix" >&2; exit 2 ;;
esac
if [[ ! -d "$MODEL" ]]; then
  echo "model directory not found: $MODEL" >&2
  exit 2
fi

MODEL_TAG="$(basename "$MODEL" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9._-' '_')"
LOG_ROOT="$ROOT/outputs/alpaca_cot_4066_logs_${MODEL_TAG}_${RUN_TAG}"
PREPARED="$ROOT/train_with_selector/train_with_selector/data/Alpaca-cot-gpt/prepared_4066"
mkdir -p "$LOG_ROOT"

if [[ ! -f "$PREPARED/manifest.json" ]]; then
  "$PY" "$ROOT/prepare_alpaca_cot_4066.py"
fi

run_one() {
  local mode="$1"
  local out="$ROOT/outputs/${MODEL_TAG}_alpaca_cot_4066_${mode}_${RUN_TAG}"
  local log="$LOG_ROOT/${mode}.log"
  if [[ -f "$out/metrics_compact.json" ]]; then
    echo "SKIP completed: $out"
    return 0
  fi
  if [[ -e "$out" ]]; then
    echo "incomplete output already exists: $out" >&2
    return 1
  fi
  echo "START mode=$mode gpu=$GPU_ID model=$MODEL out=$out"
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  PYTHONUNBUFFERED=1 \
  TOKENIZERS_PARALLELISM=false \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" -u "$ROOT/run_alpaca_cot_stage4_mix.py" \
      --mode "$mode" \
      --llama "$MODEL" \
      --out "$out" \
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
  echo "DONE mode=$mode out=$out"
}

if [[ "$MODE" == both || "$MODE" == stage4 ]]; then
  run_one stage4
fi
if [[ "$MODE" == both || "$MODE" == mix ]]; then
  run_one mix
fi

