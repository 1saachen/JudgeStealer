#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON_BIN:-python}"
GPU_ID="${1:?usage: $0 <gpu_id> [job ...]}"
shift

OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/outputs}"
LOG_ROOT="$OUTPUT_ROOT/gpt5_cot_auto_queue_logs"
STATUS_LOG="$LOG_ROOT/job_status.log"

DATA_ROOT="${DATA_ROOT:-/data/model-extraction-attack/yaolin/JudgeStealer/data}"
ALPACA_COT_DATA_DIR="${ALPACA_COT_DATA_DIR:-$DATA_ROOT/Alpaca-cot-gpt}"
GPT4ALL_COT_DATA_DIR="${GPT4ALL_COT_DATA_DIR:-$DATA_ROOT/GPT4All-cot-gpt}"
ALPACA_COT_PREPARED_DIR="${ALPACA_COT_PREPARED_DIR:-$(dirname "$ALPACA_COT_DATA_DIR")/prepared_4066}"
GPT4ALL_COT_PREPARED_DIR="${GPT4ALL_COT_PREPARED_DIR:-$(dirname "$GPT4ALL_COT_DATA_DIR")/prepared_4066}"

ALL_JOBS=(
  alpaca_llama_naive alpaca_llama_ours
  alpaca_qwen_naive alpaca_qwen_ours
  gpt4all_llama_naive gpt4all_llama_ours
  gpt4all_qwen_naive gpt4all_qwen_ours
)
if [[ "$#" -gt 0 ]]; then
  JOBS=("$@")
else
  JOBS=("${ALL_JOBS[@]}")
fi

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
cd "$ROOT"

log_status() {
  local message="$*"
  printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$message" | tee -a "$STATUS_LOG"
}

prepare_dataset() {
  local data_dir="$1"
  local prepared_dir="$2"
  local dataset="$3"
  local train="$data_dir/train_pointwise_8k.json"
  local validation="$data_dir/test_2k.json"

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
  local job="$1"
  local dataset surrogate paper_method mode model data_dir prepared_dir name out log rc

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
    data_dir="$ALPACA_COT_DATA_DIR"
    prepared_dir="$ALPACA_COT_PREPARED_DIR"
  else
    data_dir="$GPT4ALL_COT_DATA_DIR"
    prepared_dir="$GPT4ALL_COT_PREPARED_DIR"
  fi
  if [[ ! -f "$model/config.json" ]]; then
    log_status "ERROR missing model job=$job model=$model"
    return 1
  fi
  prepare_dataset "$data_dir" "$prepared_dir" "$dataset" || return $?

  name="${surrogate}_${dataset}_gpt5_cot_${paper_method}_seed42"
  out="$OUTPUT_ROOT/$name"
  log="$LOG_ROOT/$name.log"
  if [[ -f "$out/metrics_compact.json" ]]; then
    log_status "SKIP completed job=$job out=$out"
    return 0
  fi
  if [[ -e "$out" ]]; then
    log_status "ERROR incomplete output exists job=$job out=$out"
    return 1
  fi

  log_status "START job=$job mode=$mode gpu=$GPU_ID out=$out"
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
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
    log_status "DONE job=$job gpu=$GPU_ID out=$out"
    return 0
  fi
  log_status "ERROR job=$job gpu=$GPU_ID rc=$rc out=$out log=$log"
  return "$rc"
}

worker_rc=0
for job in "${JOBS[@]}"; do
  run_job "$job" || worker_rc=1
done
exit "$worker_rc"
