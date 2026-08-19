# Qwen3-1.7B GPT-5 Mix Budget Experiments

This queue runs true-value mixed controls for the two GPT-5 datasets. It is
intended to be compared with the fixed Ours runs already in the repository;
Ours remains at 200 selected queries, which is `B=1.1111%` on Alpaca and
`B=2.4691%` on GPT4All.

## Protocol

Each job uses LoRA + 4-bit Qwen3-1.7B, seed 42, learning rate `1e-4`, maximum
length `4096`, batch size `1`, gradient accumulation `16`, and final-only
evaluation. It uses the existing `trueval_three_stage` entry point with:

- equal pointwise, pairwise, and listwise training counts;
- ten epochs for each of the three stages;
- no pointwise replay, pairwise replay, Stage 4, or smoothing;
- pairwise/listwise training examples sampled from their validation files, with
  the remaining examples used for evaluation.

The measured train-split denominators and rounded query counts are:

| B | Alpaca denominator | Alpaca per-task count | GPT4All denominator | GPT4All per-task count |
|---:|---:|---:|---:|---:|
| 0.5% | 18,000 | 90 | 8,100 | 40 |
| 1% | 18,000 | 180 | 8,100 | 80 |
| 2% | 18,000 | 360 | 8,100 | 160 |
| 5% | 18,000 | 900 | 8,100 | 410 |
| 10% | 18,000 | 1,800 | 8,100 | 810 |

Each per-task count is used for all three task types. The queue passes the
corresponding `3 * Q` value as the run budget for metadata consistency.

## Run on the server

```bash
cd /data/model-extraction-attack/yaolin/JudgeStealer
git pull origin main
conda activate cyl

tmux new -s qwen17_mix_budget
./launch_qwen3_1p7b_mix_budget_auto_queue.sh 0 1 2 3
```

Detach with `Ctrl+B`, then `D`. The queue automatically dispatches jobs only to
idle GPUs and skips completed jobs. To skip selected jobs explicitly:

```bash
SKIP_JOBS="alpaca_mix_b0p5 gpt4all_mix_b0p5" \
  ./launch_qwen3_1p7b_mix_budget_auto_queue.sh 0 1 2 3
```

Outputs and logs are stored under:

```text
/opt/dlami/nvme/cyl/autodl-tmp/JudgeStealer_outputs/
qwen3_1p7b_mix_budget_seed42/
```

Each completed run contains `metrics_compact.json` and
`mix_budget_resolution.json`.
