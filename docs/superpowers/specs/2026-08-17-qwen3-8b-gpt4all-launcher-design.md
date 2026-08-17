# Qwen3-8B GPT4All Four-Stage Launcher Design

## Goal

Add one single-purpose launcher for the existing Qwen3-8B, GPT4All GPT-5
experiment. The launcher must preserve the training configuration from
`launch_qwen3_gpt5_selector_smooth_lora_table_20260814.sh` while using the
portable model and dataset paths on the new server.

## Scope

Create `launch_qwen3_8b_gpt4all_gpt5_four_stage.sh` at the repository root.
Do not change the multi-job launcher or Python training code. The new script
will run exactly one experiment and will accept one positional argument: the
physical GPU index passed to `CUDA_VISIBLE_DEVICES`.

## Paths

- Training script:
  `run_pointwise5answers_three_stage_pairwise_listwise_sft_v1.py`
- Model: `models/Qwen3-8B`
- Training data:
  `data/gpt4all/gpt5/train9k_pointwise_pairwise_no_val_overlap.json`
- Evaluation data: `data/gpt4all/gpt5/val3k_pairwise_listwise.json`
- Outputs: `outputs/qwen3_8b_gpt4all_gpt5_b600_selector_smooth_a010_pool100_stage4stratfull`
- Logs: `outputs/qwen3_8b_gpt4all_gpt5_four_stage_logs/`

All paths are resolved relative to the launcher location so the repository can
be moved without editing absolute paths.

## Training Configuration

The launcher will preserve these settings from the reference table launcher:

- seed 42 and budget 600;
- Stage 1 reuses the LM-head bias-trap selection proxy;
- Stage 2 trains pairwise with no pointwise replay;
- Stage 3 trains listwise with no pointwise or pairwise replay;
- Stage 4 uses `stratified_triple`, replay fraction 1.0, and one epoch;
- one epoch each for pointwise, pairwise, and listwise training;
- LoRA with 4-bit loading;
- per-device batch size 1 and gradient accumulation 16;
- learning rate `1e-4`, maximum length 4096, and evaluation batch size 1;
- final-stage-only evaluation;
- local-Gaussian smoothing with alpha 0.1 and sigma 1.0 for all stages;
- bias-trap pointwise selection with LM-head proxy reuse, init 80, query batch
  20, candidate pool 100, no exploration, diversity 1.0, uncertainty 0.25,
  and bias 1.0;
- `BAAI/bge-small-en-v1.5` embeddings with the original embedding and proxy
  settings.

## Runtime Behavior

The script will:

1. Validate that the GPU argument is present.
2. Validate the training script, model directory, model `config.json`, and two
   dataset files before starting.
3. Use `PYTHON_BIN` when supplied, otherwise use `python` from the activated
   Conda environment.
4. Refuse to overwrite an incomplete output directory.
5. Skip the run when `metrics_compact.json` already exists.
6. Refuse to launch a duplicate process targeting the same output directory.
7. Record start, completion, and failure status in a status log and write the
   full training output to a dedicated log file.
8. Return a nonzero exit code when validation or training fails.

The launcher will not download models or datasets automatically.

## Verification

Add a focused static test that checks the launcher uses the required portable
paths, exact four-stage replay configuration, model size, selector settings,
and smoothing settings, and that it does not reference the legacy `qwen/` or
`Dolly/` paths. Run a Bash syntax check where Bash is available. Existing
Python training tests are outside this configuration-only change and will not
be run unless requested.

