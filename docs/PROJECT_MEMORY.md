# Project Memory

> Bundle note (2026-08-16): paths and environment commands in this document
> describe the original training machine. Use the portable setup in the
> bundle-root `README.md` on a new machine.

Last updated: 2026-07-31

## Work Log

Chronological work, experiment launches, completed results, and next actions
are maintained in `WORK_LOG.md`.

At the start of a new session, read this file first for stable project rules,
then read the latest entries in `WORK_LOG.md` for current experiment state.

## Critical Launch Rules

Last checked: 2026-07-09.

Do not launch training/eval experiments with bare `python`. On this machine bare `python` may resolve to:

- `/home/chenchen/anaconda3/bin/python`
- Python 3.13.x

That environment has a PyTorch/CUDA stack incompatible with the current NVIDIA driver and can fail with:

```text
RuntimeError: The NVIDIA driver on your system is too old
```

Always launch project experiments with the `cyl` environment Python:

```bash
PY=/opt/dlami/nvme/conda/envs/cyl/bin/python
CUDA_VISIBLE_DEVICES=<gpu_id> "$PY" -u <script>.py ...
```

Known-good environment check:

```bash
/opt/dlami/nvme/conda/envs/cyl/bin/python --version
/opt/dlami/nvme/conda/envs/cyl/bin/python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
PY
```

Expected as of 2026-07-09:

- Python 3.10.20
- torch 2.5.1+cu121
- CUDA available: `True`

Before claiming an experiment is running, verify:

```bash
ps -p <pid> -o pid,stat,etime,cmd
tail -n 40 <log>
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
```

## Llama-3.2-1B Checkpoint Rule

Last updated: 2026-07-31.

For all future 1B experiments, default to:

```bash
llama/Llama-3.2-1B-Instruct
```

Do not use `llama/Llama-3.2-1B` base unless the user explicitly requests a
base-checkpoint ablation or comparison. This applies especially to Alpaca
generative three-stage SFT, pointwise/pairwise/listwise replay experiments,
and LM-head proxy-reuse experiments.

Output names should make the checkpoint unambiguous:

- if using the base checkpoint, include `base` in the run name;
- otherwise, `llama3p2_1b` or `llama3p2_1b_instruct` means the Instruct
  checkpoint.

## LLM-as-Judge PDF Selector Rule

Last updated: 2026-08-01.

For the PDF-designed bias-trap selector experiments, "different parameter
settings" means changing the selector score weights under `S`, for example
diversity, density, existing-CE uncertainty, bias-trap, and coverage weights.
Do not interpret that request as changing Stage23 training mixture/replay
counts.

In the 600-budget three-answer setting, selected triples are transformed into
training data with 600 pointwise samples, 1200 pairwise samples, and 1200
listwise samples. Pairwise and listwise counts come from deterministic
conversion of selected triples and should stay fixed unless the user explicitly
asks for a training-mixture ablation.

## Reward-Model Pointwise Selection Standard

Last updated: 2026-07-24.

This standard applies only to Reward Model class datasets with grouped
continuous rewards, such as Skywork and ARMO. It does not replace the
selection/training standards for LLM-as-a-Judge, Alpaca/Dolly classifier-head,
or generative three-stage experiments.

Unless the user explicitly requests an ablation, Reward Model pointwise main
experiments must use the proxy itself as the active-learning selector. Do not
use the BERT/Longformer selector in the main experiment matrix.

Canonical proxy acquisition settings:

- acquisition score uses only MC-dropout predictive uncertainty
  (`--proxy-uncertainty-weight 1` and
  `--proxy-response-std-weight 0`);
- individual-answer predictive uncertainty is obtained from MC-dropout reward
  predictions for each answer, then aggregated across the question's three
  answers as `0.75 * mean(answer uncertainty) + 0.25 * max(answer uncertainty)`;
- for question-level selection, individual-answer uncertainty is aggregated
  as `0.75 * mean(answer uncertainty) + 0.25 * max(answer uncertainty)`;
- this continuous regression path has no categorical entropy distribution;
  MC predictive standard deviation is used as the pure uncertainty score and
  is monotonic with Gaussian predictive entropy;
- random exploration is disabled (`--proxy-exploration-ratio 0`);
- scoring pool: all remaining unlabeled candidates, with no random candidate
  pool cap (`--selector-max-score-candidates 0`);
- question-level experiments: run both the no-smoothing baseline
  (`--smooth-alpha 0`) and smoothing treatment (`--smooth-alpha 0.01`) for
  both random and proxy selection;
- answer-level experiments: no smoothing (`--smooth-alpha 0`);
- compare random and proxy under matching question-level smoothing settings.

For grouped A/B/C reward-model data, question-level budget 600 means 200
questions and 600 answers. Answer-level budget 600 means 600 independently
selected answers. A future answer-level active baseline should be
`proxy_answer`, not the existing BERT-based `selector_answer`.

These are main-experiment defaults, not immutable implementation defaults.
Only deviate for an explicitly named selector, exploration, pool-size, or
acquisition-score ablation, and label that output as an ablation.

For selector-based three-stage runs in `run_pointwise5answers_three_stage_pairwise_listwise_sft_v1.py`, include:

```bash
--llama-multitask-mode classifier_heads
```

Important: this flag is for the **active-learning selector proxy only**, not
for the final SFT model. The three-stage script still trains the final Llama
with generative causal-LM SFT to output scores, pairwise choices, and
rankings. The normal selector proxy path uses `classifier_heads`.

Exception: LM-head proxy-reuse experiments intentionally train the selector in
the same text/token output space as generative SFT. Those runs must use:

```bash
--candidate-selector-proxy-mode lm_head \
--reuse-selection-proxy-for-stage1
```

In that mode, the selected pointwise proxy is reused as Stage 1 and the
separate generative Stage-1 SFT is skipped. Treat this as a named ablation or
experiment variant, not as the default pointproxy path.

Without this selector-proxy flag, selector runs can fail with:

```text
ValueError: multitask_mode must be one of {'lm_head','classifier_heads'}, got shared_head
```

## 1-10 Judge Pointwise Selection Standard

Last updated: 2026-07-24.

For Alpaca/Dolly-style 1-10 score-judging main experiments, use the training
proxy directly and rank questions only by normalized categorical predictive
entropy. Do not mix in per-answer score-distribution standard deviation or
within-question predicted-score spread. Use no random exploration and score
the full remaining unlabeled pool every round.

Canonical settings:

- `candidate_selector_kind = pointwise_proxy`;
- `candidate_selector_entropy_weight = 1.0`;
- `candidate_selector_score_std_weight = 0.0`;
- `candidate_selector_exploration_ratio = 0.0`;
- `candidate_selector_max_score_candidates = 0` (full remaining pool);
- BERT/Longformer selectors are historical baselines/ablations, not the main
  selector.

This protocol is separate in model/output semantics from the Reward Model
continuous-regression standard above. Both now use pure predictive
uncertainty, zero exploration, and a full remaining pool, but the judge path
uses categorical predictive entropy while the Reward Model path uses
MC-dropout predictive standard deviation.

## Current Three-Stage Experiment Standard

The current research focus is the **600-budget generative three-stage SFT**
pipeline in `run_pointwise5answers_three_stage_pairwise_listwise_sft_v1.py`.

Unless the user explicitly requests a fixed-selection/control-variable
ablation, do **not** reuse prior `selected_triples.jsonl` files or launch with
`--fixed-selected-triples-path`. New random/proxy experiments should perform
their own selection under the stated seed, selector, budget, and acquisition
settings.

Use fixed selection only when the goal is specifically to hold selected data
constant while changing another factor such as smoothing, full-FT versus LoRA,
or stage structure. In those cases, record the source selection path and make
the run name explicit, preferably `fixed_random_*` or `fixed_pointproxy_*`
rather than ambiguous names like `fixedproxy_*`.

Unless the user explicitly requests a replay/stage-structure ablation, all new
three-stage experiments must use this training structure:

1. Stage 1: train pointwise.
2. Stage 2: train pairwise with no pointwise replay.
3. Stage 3: train listwise as the main task while replaying both pointwise and
   pairwise data at ratio 1.

Required arguments:

```bash
--budget-units 600 \
--stage2-pointwise-replay-ratio 0 \
--stage3-pointwise-replay-ratio 1 \
--stage3-pairwise-replay-ratio 1
```

Terminology: listwise is the main Stage-3 task, not a replay stream. Therefore
"replay both in the final stage" means replay **pointwise and pairwise** while
training listwise. Do not silently launch older structures that replay only
pointwise, omit pairwise replay, or replay pointwise during Stage 2.

Replay ratio semantics: `ratio=1` means one pass over the replay task's own
available training examples, not upsampling the smaller replay task to match
the main task. In the 600-budget three-answer setting this normally means:

- Stage 1 pointwise data: 600 examples;
- Stage 2 pairwise data with order augmentation: 1200 examples;
- Stage 3 listwise data with order augmentation: 1200 examples;
- Stage 3 pointwise replay ratio 1: replay 600 pointwise examples;
- Stage 3 pairwise replay ratio 1: replay 1200 pairwise examples.

Do not repeat the 600 pointwise examples into 1200 examples unless the user
explicitly asks for an oversampling ablation.

Before launching, verify the resolved command/config contains all three replay
ratios above. Only change this standard when the user explicitly asks for an
ablation or a new stage structure.

## 2026-07-31 Conversation Decisions: 1B Alpaca Generative SFT

This section records the main decisions from the 2026-07-31 discussion about
1B-Instruct proxy selection, replay/continual-learning, and merged Stage23
controls.

- Use `Llama-3.2-1B-Instruct` for 1B experiments by default. The base
  `Llama-3.2-1B` checkpoint is only for explicitly requested base ablations;
  existing base LM-head results are not the main 1B result.
- Avoid reusing `selected_triples.jsonl` unless the user explicitly requests a
  fixed-selection/control-variable ablation. New random/proxy experiments
  should rerun their own selection.
- Use names that separate selection from training reuse:
  `pointproxy_*` means the run performs proxy selection; `fixed_pointproxy_*`
  means the run reuses a prior pointproxy-selected file.
- For corrected replay semantics, pointwise replay ratio `1` means replay the
  available 600 pointwise examples once. Do not upsample pointwise replay to
  1200 merely to match listwise order augmentation unless running an explicit
  oversampling ablation.
- Current merged Stage23 exploration is allowed, but it is not yet established
  as the main experiment. Corrected merged exposure is:
  `Stage1 600 pointwise + Stage23 1200 pairwise + 1200 listwise + 600
  pointwise replay = 3600` training examples.
- Exposure-matched controls for corrected merged Stage23 should be
  `one_answer_ep6` and `trueval_mix_ep6`. For corrected standard three-stage
  (`Stage1 600 + Stage2 1200 + Stage3 1200 listwise + 600 pointwise replay +
  1200 pairwise replay = 4800`), exposure-matched controls would be ep8.
  Old ep7/ep9 controls correspond to old replay-count assumptions.
- Current read on replay/continual learning: replay helps prevent complete
  forgetting and can recover pointwise/pairwise after listwise training, but
  the observed gains are modest and not clearly a listwise win. Old
  1B-Instruct results showed random replay stronger on listwise than
  pointproxy replay, and Stage4 consolidation without Stage3 replay looked
  weak. Do not claim replay is a successful mechanism until corrected
  reruns/ep6 controls are compared.

## What This Project Is

This repository is an experiment workspace for active-learning-based judge/proxy training. The main research flow compares how to select limited training data under a budget, then trains/evaluates pointwise scoring and pairwise preference judging models.

The project is script-driven rather than a packaged application. Most experiments are launched from root-level Python or shell scripts, and outputs are written under `outputs/`.

## Current Main Script

`run_pointwise5answers_two_to_pairwise_v1.py` is the central experiment entry point currently being edited.

Its main pipeline:

1. Load a scored pointwise dataset where each question has 5 candidate answers.
2. Select exactly 2 answers per question.
3. Train stage 1 on the selected pointwise scored answers.
4. Convert the selected answer pair into a natural pairwise preference sample using the scores.
5. Train stage 2 on pairwise data, optionally with pointwise replay.
6. Evaluate pointwise and pairwise metrics and write summaries to the output directory.

Important modes/options:

- `--train-selection-mode selected_pair`: legacy flow; pick one pair per question first, then apply budget.
- `--train-selection-mode candidate_pair_selector`: build all candidate answer pairs, then select pairs using a selector or distribution objective.
- `--candidate-selector-kind`: includes `bert`, `shared_llama`, `random`, and `distribution`.
- `--candidate-selector-target-task`: selector target can be `pointwise` or `pairwise`.
- `--budget-units`: budget is counted in pointwise answer units; one selected pair costs 2 units.
- `--stage2-pointwise-replay-ratio`: controls pointwise replay during pairwise stage 2.
- `--internal-val-mode question_single_answer`: keeps pointwise validation at single-answer/question level.
- `--pairwise-order-augmentation`: adds reversed pairwise order samples.

## Data Concepts

Pointwise judge data:

- One candidate answer receives a score from 1 to 10.
- The score is mapped to a class label `0..9`.
- Prompt construction lives in `train_with_selector/train_with_selector/data/judge_dataset.py`.

Pairwise preference data:

- Two assistant answers are compared.
- Labels are:
  - `0`: Assistant 1 better
  - `1`: Assistant 2 better
  - `2`: Tie
- Prompt construction lives in `train_with_selector/train_with_selector/data/pairwise_dataset.py`.

The script also supports ABC-style pairwise eval/train records with model/output A/B/C fields and AB/BC choices.

## Core Code Map

- `run_pointwise5answers_two_to_pairwise_v1.py`: current two-stage pointwise-to-pairwise experiment.
- `train_with_selector/train_with_selector/config.py`: active learning configuration dataclass.
- `train_with_selector/train_with_selector/active_learning_pipeline.py`: original active learning runner.
- `train_with_selector/train_with_selector/active_learning_pipeline_v2.py`: improved runner with pred-error supervision, optional two-stage filtering, and optional selector head restart.
- `train_with_selector/train_with_selector/models/llama_proxy.py`: Llama proxy implementation.
- `train_with_selector/train_with_selector/models/llama_shared_proxy.py`: shared Llama proxy with feature extraction for selectors.
- `train_with_selector/train_with_selector/models/llama_shared_multitask_proxy.py`: multitask pointwise/pairwise proxy.
- `train_with_selector/train_with_selector/selector/`: random, BERT, binary, and shared-Llama selector implementations.
- `pairwise_common.py`: pairwise loading/normalization helpers.
- `pairwise_eval.py`: standalone pairwise generation-style evaluator.
- `aggregate_eval_metrics.py` and `summarize_aggregated_metrics.py`: metric aggregation utilities.

## Output Files To Expect

Typical output directories contain:

- `config.json`: resolved run config.
- `dataset_load_stats.json`: input loading stats.
- `selected_pair_stats.json` / `selected_pairs.jsonl`: legacy selected pairs.
- `candidate_pair_pool.jsonl`: full candidate pair pool.
- `candidate_pair_selection_stats.json` / `selected_candidate_pairs.jsonl`: selector-mode chosen pairs.
- `train_budget.json`: budget accounting.
- `split_by_question.json`, `split_pointwise.json`, `split_pairwise.json`: split metadata.
- `pointwise_train.jsonl`, `pointwise_val*.jsonl`: pointwise train/eval views.
- `pairwise_train.jsonl`, `pairwise_val.jsonl`: generated pairwise train/eval views.
- `metrics_*_before_stage1.json`, `metrics_*_after_stage1.json`, `metrics_*_after_stage2.json`: staged metrics.
- `summary.json` and `metrics_compact.json`: final run summaries.

## Current Working-State Notes

At the time this memory was written, the working tree had local modifications in:

- `run_pointwise5answers_two_to_pairwise_v1.py`
- `train_with_selector/train_with_selector/models/llama_proxy.py`
- `train_with_selector/train_with_selector/models/llama_shared_multitask_proxy.py`

There was also an untracked command note file:

- `pointwise5answers_experiment_commands.md`

Do not overwrite or revert these changes unless explicitly asked.

## 2026-06-10 Experiment Snapshot

### Newnew Data

Primary new dataset paths:

- Train: `train_with_selector/train_with_selector/data/newnew/train-20k.json`
- Pairwise eval: `train_with_selector/train_with_selector/data/newnew/val-2k-eval.json`
- Listwise eval: `train_with_selector/train_with_selector/data/newnew/val-2k-eval-listwise.json`

`train-20k.json` has 20,000 records. Each record is one question with exactly three scored answers in A/B/C fields:

- `instruction`, `input`
- `modelA/outputA/scoreA`
- `modelB/outputB/scoreB`
- `modelC/outputC/scoreC`

It is not three separate records per question; it is one grouped A/B/C record. The pairwise script can convert A/B/C into pointwise and pairwise views. The listwise script can directly use the three answers as one triple.

`val-2k-eval.json` has 2,000 records with A/B/C answers plus `choiceAB` and `choiceBC`. It yields 4,000 pairwise eval pairs: A=1860, B=1682, tie=458.

`val-2k-eval-listwise.json` has 2,000 A/B/C records with `ranking` labels. It is the correct validation file for `run_pointwise5answers_three_to_listwise_v1.py`; do not use `val-2k-eval.json` for listwise validation.

### Direct Pointwise Script

Added `run_pointwise_direct_train_eval.py` for direct pointwise-only train/eval from explicit files:

- `--pointwise-train-dataset`
- `--pointwise-eval-dataset`

It supports flattened pointwise samples such as `instruction/input/output/score` or `prompt/score`, and grouped A/B/C records. It does not split data, select pairs, or run stage 2.

`--pointwise-distance-weight` was intentionally removed from this direct script and from its command docs because direct pointwise CE runs do not use it.

### Pairwise Newnew Results

Main script: `run_pointwise5answers_two_to_pairwise_v1.py`.

Common settings for the newnew pairwise sweep:

- `--pointwise-training-mode proxy`
- `--pairwise-abc-eval-dataset train_with_selector/train_with_selector/data/newnew/val-2k-eval.json`
- `--pairwise-order-augmentation`
- `--stage2-pointwise-replay-ratio 3`
- `--pointwise-loss-type ce`
- `--pairwise-abc-train-records 0`
- `--pointwise-epochs 1`
- `--pairwise-epochs 1`
- `--pointwise-batch-size 32`
- `--pairwise-batch-size 32`

Completed pairwise outputs live under `outputs/newnew_budget_sweep/`.

Important finished results:

| method | budget | seed | final acc | final within1 | final MAE | pairwise acc | tie rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| random | 500 | 42 | 0.3865 | 0.6050 | 1.7870 | 0.5680 | 0.0013 |
| random | 500 | 43 | 0.4115 | 0.6200 | 1.6365 | 0.4645 | 0.0010 |
| BERT gap0 | 500 | 42 | 0.4210 | 0.6105 | 1.5525 | 0.6068 | 0.0013 |
| BERT gap0 | 500 | 43 | 0.3990 | 0.6105 | 1.5990 | 0.5350 | 0.0008 |
| BERT gap1 | 500 | 42 | 0.4075 | 0.5915 | 1.6020 | 0.5118 | 0.0027 |
| random | 1000 | 42 | 0.1915 | 0.4910 | 2.8645 | 0.4410 | 0.0000 |
| random | 1000 | 43 | 0.3600 | 0.5855 | 1.5875 | 0.6235 | 0.0027 |
| BERT gap0 | 1000 | 42 | 0.3770 | 0.6325 | 1.5225 | 0.5982 | 0.0020 |
| BERT gap0 | 1000 | 43 | 0.4385 | 0.6425 | 1.6045 | 0.6130 | 0.0003 |
| BERT gap1 | 1000 | 42 | 0.4265 | 0.6340 | 1.4845 | 0.5795 | 0.0005 |

Interpretation:

- Random has high seed variance. `random budget1000 seed42` collapsed after stage 2, predicting almost everything as score 9; `random budget1000 seed43` did not collapse and reached pairwise acc 0.6235.
- BERT gap0 is the most stable pairwise selector setting so far. For budget1000, pairwise acc was 0.5982 and 0.6130 across seeds 42/43.
- BERT gap1 may help pointwise MAE, especially at budget1000, but it was worse for pairwise at budget500 and caused a strong A-win prediction bias.
- Tie is still essentially not learned in pairwise runs. Real eval tie count is 458/4000, but model tie rates remain near zero.

Useful pairwise output names:

- Random 500 seed42: `outputs/newnew_budget_sweep/newnew_e0_random_ce_replay3_budget500_aug_abc0`
- Random 500 seed43: `outputs/newnew_budget_sweep/newnew_e0_random_ce_replay3_budget500_seed43_aug_abc0`
- Random 1000 seed42: `outputs/newnew_budget_sweep/newnew_e0_random_ce_replay3_budget1000_aug_abc0`
- Random 1000 seed43: `outputs/newnew_budget_sweep/newnew_e0_random_ce_replay3_budget1000_seed43_aug_abc0`
- BERT gap0 500 seed42: `outputs/newnew_budget_sweep/newnew_e11_selector_bert_pointwise_unc_gap0_init100_ce_replay3_budget500_aug_abc0`
- BERT gap0 500 seed43: `outputs/newnew_budget_sweep/newnew_e11_selector_bert_pointwise_unc_gap0_init100_ce_replay3_budget500_seed43_aug_abc0`
- BERT gap0 1000 seed42: `outputs/newnew_budget_sweep/newnew_e11_selector_bert_pointwise_unc_gap0_init200_ce_replay3_budget1000_aug_abc0`
- BERT gap0 1000 seed43: `outputs/newnew_budget_sweep/newnew_e11_selector_bert_pointwise_unc_gap0_init200_ce_replay3_budget1000_seed43_aug_abc0`
- BERT gap1 500 seed42: `outputs/newnew_budget_sweep/newnew_e11_selector_bert_pointwise_unc_gap1_init100_ce_replay3_budget500_aug_abc0`
- BERT gap1 1000 seed42: `outputs/newnew_budget_sweep/newnew_e11_selector_bert_pointwise_unc_gap1_init200_ce_replay3_budget1000_aug_abc0`

### Listwise Newnew Results

Main script: `run_pointwise5answers_three_to_listwise_v1.py`.

For listwise, budget is counted in pointwise answer units. One selected triple costs 3 units. With `--listwise-order-augmentation`, each triple becomes 6 listwise training examples.

Baseline legacy selected_triple runs completed under `outputs/newnew_listwise/`:

| method | budget | train triples | listwise train | pointwise acc | pointwise MAE | listwise acc | top-group acc | pairwise relation acc | best-in-top | rank MAE | tie rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| legacy selected_triple | 750 | 250 | 1500 | 0.3715 | 1.6775 | 0.2975 | 0.4985 | 0.6508 | 0.6095 | 0.5595 | 0.1100 |
| legacy selected_triple | 1500 | 500 | 3000 | 0.3945 | 1.5725 | 0.3800 | 0.5025 | 0.7442 | 0.9015 | 0.4383 | 0.5245 |

Interpretation:

- `budget1500` is much stronger than `budget750`, especially for listwise relation quality and best-answer-in-top.
- Listwise learns tie behavior much more than pairwise, but `budget1500` appears to overpredict tie or partial-tie labels.
- Baseline legacy selected_triple runs are not selector runs. For newnew, each question already has exactly three answers, so non-selector listwise baselines all use the same three answers per question; current script defaults to `triple_selection_strategy=random`.

Baseline output names:

- `outputs/newnew_listwise/newnew_maxspread_ce_replay3_budget750_seed42_aug`
- `outputs/newnew_listwise/newnew_maxspread_ce_replay3_budget1500_seed42_aug`

### Listwise BERT Selector

The listwise script supports selector-based training with:

- `--train-selection-mode candidate_triple_selector`
- `--candidate-selector-kind bert`
- `--candidate-selector-target-task pointwise`

Old listwise selector outputs from earlier data exist under names such as:

- `outputs/listwise_bert_pointwise_selector_init50_batch20_budget1500`
- `outputs/listwise_bert_pointwise_selector_unc_gap1_init50_batch20_budget750`
- `outputs/listwise_bert_selector_aug_budget1500`

For newnew, the current requested selector setting is BERT gap0, interpreted as:

- `--candidate-selector-target-task pointwise`
- `--candidate-selector-score-range-weight 0.0`
- `--candidate-selector-gap-sum-weight 0.0`
- `--candidate-selector-uncertainty-weight 1.0`
- `--candidate-selector-kl-weight 0.0`
- `--candidate-selector-score-bin-weight 0.0`
- `--candidate-selector-init-triples 50`
- `--candidate-selector-batch-size 20`
- `--candidate-selector-epochs 4`

On 2026-06-10, a sequential tmux run was started in `task2:lwgap0_750_1000` on GPU5:

1. `outputs/newnew_listwise/newnew_bert_pointwise_selector_gap0_init50_batch20_budget750_seed42_aug`
2. `outputs/newnew_listwise/newnew_bert_pointwise_selector_gap0_init50_batch20_budget1000_seed42_aug`

The first run had successfully entered selector mode:

- `Candidate triples: 18000`
- selected 70/250 after round 1
- selected 90/250 after round 2

Check these output directories before deciding whether to restart or analyze them.

## Current Experiment Defaults

As of 2026-06-13, future newnew experiments should default to the SFT-first
setup unless the user explicitly asks for the old proxy/CE baseline.

Default pairwise path:

- Use full two-stage SFT first: pointwise score SFT -> pairwise preference SFT.
- In `run_pointwise5answers_two_to_pairwise_v1.py`, this means:
  - `--pointwise-training-mode sft`
  - `--training-mode sft`
- Current low-budget SFT runs use LoRA + 4bit unless explicitly testing full
  parameter finetuning:
  - `--sft-use-lora`
  - `--sft-4bit`
- Pairwise SFT evaluation must use the strict invalid policy: parse failures
  are counted as invalid and wrong, not converted to tie.
- Output names for strict invalid SFT runs should include
  `_leftpad_strictinvalid`.

Old proxy/CE runs remain useful as baselines, but they should not be the first
default choice for new experiments because SFT substantially improved pairwise
accuracy and learned tie behavior on newnew.

### Tmux / Runtime Conventions

The user prefers long experiments to run in tmux session `task2`, under conda env `cyl`.

Use:

- `source /home/chenchen/anaconda3/etc/profile.d/conda.sh && conda activate cyl`
- Avoid relying on `/home/chenchen/.bashrc`; in some tmux windows it does not exist.
- Check GPUs with `nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits`.
- Do not overwrite existing output directories unless explicitly asked.

## How To Use This Memory

At the start of a future session, ask the assistant to read `PROJECT_MEMORY.md` first. Then it should inspect only the specific files relevant to the current task, because many experiment details may have changed after this snapshot.
