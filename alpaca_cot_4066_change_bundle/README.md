# Three-stage SFT and selector experiments

This repository contains the training and evaluation code for the recent
continuous reward-model and selector experiments. Large local model copies,
datasets, checkpoints, logs, and generated outputs are intentionally excluded
from Git. See `.gitignore` for the complete list.

The maintained Python entry points and their dependency graph are documented
in `PYTHON_CODE_MAP.md`. Dated shell files only capture experiment-specific
arguments and are not the core implementation.

## Current experiment entry points

The recovered Alpaca CoT 4066-question experiment, including its synthetic
four-stage method and real-CoT Mix control, is documented in
`ALPACA_COT_4066_EXPERIMENT.md`.

The following are the maintained launchers at the moment:

- `launch_rewardmodel_native_selector_swap_20260815.sh`: current native-format
  reward-model mix/selector comparison.
- `launch_rewardmodel_mix_soft_20260814.sh`: continuous reward-model soft-tie
  mix control.
- `launch_rewardmodel_three_stage_20260813.sh`: compact mix and selector
  launcher for Llama-3.2-1B and Qwen3-1.7B.
- `launch_qwen3_gpt5_selector_smooth_lora_table_20260814.sh`: Qwen3 surrogate
  size table on Alpaca and Dolly GPT-5 data.
- `rerun_qwen3_8b_dolly_selector_smooth_20260815.sh`: fixed-selection Qwen3-8B
  Dolly rerun.
- `launch_new4_mix_selector_smooth_20260813.sh`: recent true-value mix and
  selector comparison.
- `launch_three_stage_sft_fsdp.sh`: generic multi-GPU FSDP wrapper.
- `download_qwen3_surrogates.sh`: optional model download helper.

Older dated launch and schedule files are kept only in the local ignored
directory `archive/legacy_launchers_20260815/`; they are not part of the
reproducible project surface.

## Core Python modules

The maintained training path is:

`run_rewardmodel_three_stage_sft.py` ->
`run_pointwise5answers_three_stage_pairwise_listwise_sft_v1.py` ->
`run_pointwise5answers_three_to_listwise_v1.py` ->
`run_pointwise5answers_two_to_pairwise_v1.py`.

`prepare_rewardmodel_three_stage.py` prepares aligned pointwise, pairwise, and
listwise JSON files. The package under `train_with_selector/` provides the
selector, proxy, dataset, and utility implementations.

## Running on another machine

Create a compatible Python environment with PyTorch, Transformers, PEFT,
Datasets, Accelerate, NumPy, and tqdm. Set `PYTHON_BIN` if `python` is not the
desired interpreter:

```bash
export PYTHON_BIN=/path/to/python
git clone <repository-url>
cd <repository>
pip install -r requirements.txt
bash launch_rewardmodel_three_stage_20260813.sh
```

The launchers resolve the repository root from their own location. Model and
data paths are still expected to be supplied locally; do not commit those
large files. Replace the local paths in a launcher, or add a small machine-
specific wrapper that supplies the model and dataset paths.

Before publishing, verify that no local data, model weights, API keys, or
output directories are staged:

```bash
git status --short
git diff --cached --stat
```
