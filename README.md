# Recent three-stage SFT project bundle

This is the minimal code bundle for the experiments active on 2026-08-13 to
2026-08-16. It contains source code, data-preparation utilities, tests, and
project notes. It intentionally contains no datasets, model weights, training
outputs, caches, or logs.

## What to run

There are two main entry points:

- `run_pointwise5answers_three_stage_pairwise_listwise_sft_v1.py` for the
  pointwise -> pairwise -> listwise -> Stage 4 selector experiments.
- `run_rewardmodel_three_stage_sft.py` for continuous reward-model mix,
  selector, and native-JSON experiments.

The other `run_*.py` files are imported by these entry points and must remain
beside them. Their roles are documented in `docs/PYTHON_CODE_MAP.md`.

## 1. Create the environment

Python 3.10 or 3.11 is recommended. Create an isolated environment and install
a PyTorch build compatible with the machine's NVIDIA driver/CUDA setup first.

```bash
conda create -n three-stage-sft python=3.10 -y
conda activate three-stage-sft

# Install a CUDA-compatible PyTorch build for this machine first.
pip install torch
pip install -r requirements.txt
```

For 4-bit loading, the GPU environment must support `bitsandbytes`. To do a
non-quantized run, pass `--no-load-in-4bit` where that option is available and
make sure the GPU has enough memory.

Verify the installation:

```bash
python -m pytest -q tests
python run_pointwise5answers_three_stage_pairwise_listwise_sft_v1.py --help
python run_rewardmodel_three_stage_sft.py --help
```

## 2. Place models

Model weights are not included. Either download them locally:

```text
models/
├── Qwen3-1.7B/
├── Qwen3-4B/
└── Llama-3.2-1b-instruct/
```

or pass a Hugging Face model ID to `--llama` if the machine has network access
and the model license/access requirements are satisfied.

Examples:

```bash
--llama models/Qwen3-1.7B
--llama Qwen/Qwen3-1.7B
```

## 3. Place datasets

Datasets are not included. The recommended local layout is:

```text
data/
├── rewardmodel/source/
│   ├── pointwise.json
│   ├── pairwise.json
│   └── listwise.json
├── alpaca/
│   ├── train.json
│   ├── pairwise_eval.json
│   └── listwise_eval.json
└── dolly/
    ├── train.json
    ├── pairwise_eval.json
    └── listwise_eval.json
```

See `data/README.md` for required fields and preprocessing commands.

## 4. Prepare continuous reward-model splits

```bash
python prepare_rewardmodel_three_stage.py \
  --source data/rewardmodel/source \
  --seed 42 \
  --train-size 1500 \
  --mix-size 200 \
  --eval-size 300
```

This creates `split1500_500/`, `mix200_eval300/`, and
`three_stage_split.json` below the source directory.

The reward-model three-stage entry point keeps continuous pointwise rewards
and source pairwise choices. Equal-score pairwise rows use a soft two-winner
target, while explicit `choice=C` remains a hard tie. Listwise SFT emits only
the source best choice; equal top scores use a soft target over the tied best
responses rather than a full ranking target.

The LoRA table launcher and its data contract are documented in
`docs/REWARDMODEL_LORA_EXPERIMENTS.md`.

## 5. Smoke-run the continuous mix experiment

```bash
python run_rewardmodel_three_stage_sft.py \
  --mode mix \
  --target-format converted \
  --pointwise-train data/rewardmodel/source/mix200_eval300/pointwise_train200.json \
  --pairwise-train data/rewardmodel/source/mix200_eval300/pairwise_train200.json \
  --listwise-train data/rewardmodel/source/mix200_eval300/listwise_train200.json \
  --pointwise-eval data/rewardmodel/source/mix200_eval300/pointwise_eval300.json \
  --pairwise-eval data/rewardmodel/source/mix200_eval300/pairwise_eval300.json \
  --listwise-eval data/rewardmodel/source/mix200_eval300/listwise_eval300.json \
  --llama models/Qwen3-1.7B \
  --out outputs/rewardmodel_mix \
  --smooth-alpha 0 \
  --per-device-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --max-length 4096
```

## 6. Smoke-run three-stage SFT

Start with random triple selection to verify the environment before enabling
the more expensive learned selector:

```bash
python run_pointwise5answers_three_stage_pairwise_listwise_sft_v1.py \
  --pointwise-5answers-dataset data/alpaca/train.json \
  --pairwise-eval-dataset data/alpaca/pairwise_eval.json \
  --listwise-eval-dataset data/alpaca/listwise_eval.json \
  --llama models/Qwen3-1.7B \
  --out outputs/alpaca_random_smoke \
  --train-selection-mode selected_triple \
  --triple-selection-strategy random \
  --budget-units 600 \
  --pointwise-epochs 1 \
  --pairwise-epochs 1 \
  --listwise-epochs 1 \
  --eval-stages final \
  --use-lora \
  --load-in-4bit
```

After the smoke run succeeds, use `--train-selection-mode
candidate_triple_selector` and the selector parameters recorded in
`docs/WORK_LOG.md` for the exact recent experiment configuration.

## Notes

- Run commands from the bundle root so local imports resolve consistently.
- Write all generated artifacts under `outputs/`; this directory is ignored.
- Keep machine-specific paths, API tokens, datasets, and weights out of Git.
- `docs/PROJECT_MEMORY.md` and `docs/WORK_LOG.md` preserve the project history
  and decisions that led to the current experiments.
