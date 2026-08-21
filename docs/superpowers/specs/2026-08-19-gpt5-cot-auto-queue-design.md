# GPT-5 CoT Auto Queue Design

## Goal

Launch the GPT-5-scored CoT matrix for Alpaca and GPT4All across Llama-3.2-1B-Instruct and Qwen3-1.7B. `mix` is recorded as the paper's Naive method and `stage4` as Ours.

## Queue

The launcher runs eight jobs in a fixed, resumable order:

```text
alpaca_llama_naive  alpaca_llama_ours
alpaca_qwen_naive   alpaca_qwen_ours
gpt4all_llama_naive gpt4all_llama_ours
gpt4all_qwen_naive  gpt4all_qwen_ours
```

Each job uses a separate output directory and log. An existing `metrics_compact.json` marks a completed job and is skipped. Any other pre-existing output directory is treated as an interrupted run and fails without overwriting it.

## Data Interface

The launcher accepts `ALPACA_COT_DATA_DIR` and `GPT4ALL_COT_DATA_DIR`. Each directory must contain the two original CoT files `train_pointwise_8k.json` and `test_2k.json`, whose schema is validated by `prepare_alpaca_cot_4066.py`.

For each dataset, the launcher runs the preparation script into a dataset-specific `prepared_4066` directory when its `manifest.json` is absent. It then passes that prepared directory's train, eval, and Mix files explicitly to `run_alpaca_cot_stage4_mix.py`.

Defaults are `/data/model-extraction-attack/yaolin/JudgeStealer/data/Alpaca-cot-gpt` and `/data/model-extraction-attack/yaolin/JudgeStealer/data/GPT4All-cot-gpt`. Override either directory with `ALPACA_COT_DATA_DIR` or `GPT4ALL_COT_DATA_DIR`. Missing source data stops before any GPU work and identifies the missing path.

## Models And Runtime

The launcher accepts one or more allowed GPU IDs, matching existing queue scripts. For example, `bash launch_gpt5_cot_auto_queue.sh 5 6 7` schedules all eight jobs across GPUs 5, 6, and 7, waiting for an idle card before each launch. Use space-separated job IDs in `SKIP_JOBS` to omit selected jobs. Llama uses `llama/Llama-3.2-1B-Instruct`; Qwen uses `qwen/Qwen3-1.7B`. Both use LoRA, 4-bit loading, a 600-answer budget, seed 42, and the existing CoT runner defaults.

`OUTPUT_ROOT` defaults to `outputs/`. `PYTHON_BIN` can select an alternate interpreter. The script writes its logs below `outputs/gpt5_cot_auto_queue_logs/`.

## Verification

A shell-focused pytest verifies accepted job names, mode mapping, required source-file checks, prepared-data argument wiring, and that Naive and Ours outputs cannot collide. The existing CoT unit test suite remains the regression check for the Python runner.
