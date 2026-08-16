# Work Log / 工作日志

> Bundle note (2026-08-16): this is historical experiment context. Dated
> launcher names and absolute paths are records, not commands for a new host.

This file is the chronological project handoff log. Keep `PROJECT_MEMORY.md`
for stable rules and durable project knowledge; record experiments, code
changes, results, and next actions here.

## 2026-08-02 - Bias-Trap LM-Head Reuse Stage3/Stage4 Sweep

Clarified the requested "full" version: the new comparison should use full
Stage4 stratified replay, not DER/continual-distillation. The requested grid is
two selector-weight settings times three training flows:

- `s_default`: `S=(div=0.25, density=0.15, uncertainty=0.25, bias=0.25,
  coverage=0.10)`.
- `s_bias50`: `S=(0.10, 0.10, 0.20, 0.50, 0.10)`, the strongest completed
  non-reuse setting for Listwise Acc so far.
- Standard Stage3 flow: reuse the LM-head selector proxy as Stage1, then
  Stage2 pairwise with `stage2_pointwise_replay_ratio=0`, then Stage3 listwise
  with `stage3_pointwise_replay_ratio=1` and `stage3_pairwise_replay_ratio=1`.
- Stage4 half/full flow: reuse the LM-head selector proxy as Stage1, train
  Stage2 pairwise and Stage3 listwise without Stage3 replay, then Stage4
  `stratified_triple` replay with fraction `0.5` or `1.0`. No DER, no
  hard-loss replay, no teacher distillation.

Code change: allowed `--reuse-selection-proxy-for-stage1` for
`candidate_selector_kind=bias_trap_pointwise` when
`candidate_selector_proxy_mode=lm_head`. Previously the reuse path was
restricted to `pointwise_proxy`.

Scripts:

- Launcher:
  `launch_llama3p2_1b_instruct_biastrap_lmheadreuse_stage3_stage4_20260802.sh`
- Scheduler:
  `schedule_llama3p2_1b_instruct_biastrap_lmheadreuse_stage3_stage4_20260802.sh`
- Active tmux session:
  `l1b_biastrap_lmheadreuse_stage3_stage4_0802_v2`
- Queue log:
  `outputs/llama3p2_1b_instruct_biastrap_lmheadreuse_queue_logs_20260802_reusefull_v2/queue_status.log`
- Run logs:
  `outputs/llama3p2_1b_instruct_biastrap_lmheadreuse_logs_20260802_reusefull_v2/`

The first launch used GPU list `6 7 5`; GPU5 was still occupied by the older
`s_coverage30` full-pool job, so the partial new outputs/logs were stopped and
archived with suffix `aborted_gpu5overcommit_20260802T153252Z`. The clean
restart uses only GPUs `6 7`.

Initial clean launch status: `default_stage3` is running on GPU6 and
`bias_stage3` is running on GPU7. Both have config
`candidate_selector_kind=bias_trap_pointwise`,
`candidate_selector_proxy_mode=lm_head`,
`reuse_selection_proxy_for_stage1=true`, and
`candidate_selector_max_score_candidates=0`.

## 2026-08-01 - Bias-Trap Selector `S` Weight Sweep Launched

Corrected the sweep target after the replay-ratio confusion: the varying
parameters are now the PDF selector score weights under `S`, while the
training conversion and replay protocol are fixed to the previous separate
three-stage scheme.

Fixed training protocol for this sweep:

- Model: `llama/Llama-3.2-1B-Instruct`.
- Selection: `candidate_triple_selector` with
  `candidate_selector_kind=bias_trap_pointwise`.
- Budget: `600` pointwise units, i.e. `200` selected triples converted to
  `600` pointwise, `1200` pairwise, and `1200` listwise examples.
- Stage flow: Stage 1 pointwise, Stage 2 pairwise with
  `stage2_pointwise_replay_ratio=0`, then Stage 3 listwise with
  `stage3_pointwise_replay_ratio=1` and `stage3_pairwise_replay_ratio=1`.
- Selector scoring now uses the full remaining candidate pool each round:
  `candidate_selector_max_score_candidates=0`. Pairwise position-bias probing
  still uses `candidate_selector_pairwise_position_pairs=1` per triple.

Queued configs:

- `s_default`: `S=(div=0.25, density=0.15, uncertainty=0.25, bias=0.25,
  coverage=0.10)`.
- `s_uncertainty50`: `S=(0.10, 0.10, 0.55, 0.15, 0.10)`.
- `s_bias50`: `S=(0.10, 0.10, 0.20, 0.50, 0.10)`.
- `s_divdensity60`: `S=(0.35, 0.25, 0.20, 0.10, 0.10)`.
- `s_coverage30`: `S=(0.15, 0.10, 0.25, 0.20, 0.30)`.

Queue:

- tmux session: `l1b_bias_selector_stage3_0801`.
- Queue script: `schedule_llama3p2_1b_instruct_bias_selector_stage3_20260801.sh`.
- Per-job launcher: `launch_llama3p2_1b_instruct_bias_selector_stage3_20260801.sh`.
- Queue log:
  `outputs/llama3p2_1b_instruct_bias_selector_stage3_queue_logs_20260801_fullpool/queue_status.log`.
- Run logs:
  `outputs/llama3p2_1b_instruct_bias_selector_stage3_logs_20260801_fullpool/`.

The first `pool1000` queue was stopped and archived with suffix
`aborted_pool1000_20260801T071954Z`; it was only a speed approximation and is
not the run to analyze. The restarted full-pool queue uses tmux session
`l1b_bias_selector_stage3_fullpool_0801`.

Initial full-pool launch status: `s_default`, `s_uncertainty50`, and `s_bias50`
are running on GPUs 5/6/7 respectively with output names containing
`poolall`; `s_divdensity60` and `s_coverage30` are pending until a queue GPU is
free.

Partial results as of 2026-08-02: four full-pool configurations have completed
successfully; `s_coverage30` has completed selection and is still in Stage-1
evaluation/training, so it is excluded from final metric comparisons for now.

Final Stage-3 metrics for completed full-pool runs:

| Config | S weights `(div,den,unc,bias,cov)` | Point Acc | Within1 | MAE | Pair Acc | List Acc | PairRel | Rank MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `s_bias50` | `(0.10,0.10,0.20,0.50,0.10)` | `0.2635` | `0.4705` | `2.5850` | `0.6857` | `0.3685` | `0.5960` | `0.5775` |
| `s_default` | `(0.25,0.15,0.25,0.25,0.10)` | `0.2335` | `0.4975` | `2.7115` | `0.6872` | `0.3655` | `0.5922` | `0.5830` |
| `s_divdensity60` | `(0.35,0.25,0.20,0.10,0.10)` | `0.2605` | `0.4195` | `2.6980` | `0.6793` | `0.3435` | `0.5805` | `0.6040` |
| `s_uncertainty50` | `(0.10,0.10,0.55,0.15,0.10)` | `0.2690` | `0.4730` | `2.6720` | `0.6937` | `0.3170` | `0.5512` | `0.6265` |

Current read: `s_bias50` is the strongest completed selector-weight setting
for listwise ranking and also has the best MAE among completed runs.
`s_uncertainty50` improves pointwise/pairwise accuracy but hurts listwise
ranking substantially, suggesting high CE uncertainty alone selects examples
that are useful for local discrimination but less useful for final ordering.
`s_divdensity60` is not competitive in this sweep.

## 2026-08-01 - Llama-3.2-1B-Instruct Stage23 Grid Results

Correction: the first Stage23 parameter-grid analysis included training-mixture
ablations (`pw0/pw0.25/pw0.75/pw1/replay2`). These changed the Stage23
pointwise replay exposure rather than the PDF selector scoring weights `S`.
They should not be interpreted as the requested selector-weight experiment.

A follow-up `fixedpw600` queue was also launched in the wrong direction and
then stopped on 2026-08-01. The tmux session
`l1b_instruct_stage23_fixedpw600_0801` was killed, no matching processes
remained, and partial outputs/logs were archived with suffix
`aborted_wrong_target_20260801T070122Z`. The temporary training-script and
launcher changes for `--stage23-pointwise-replay-samples` were reverted.

Correct target: keep the transformed training data protocol unchanged
(`pairwise=1200`, `listwise=1200` for the 600-budget triple selection setting)
and tune/train the PDF selector score weights under `S` instead.

All fixed-selection merged Stage23 parameter-grid runs completed. The original
separate Stage-2 then Stage-3 replay flow also completed. Metrics below are
final-stage validation metrics.

Main fixed-selection baseline:

- merged Stage23 `pair/list/pointwise = 1/1/0.5`:
  Point Acc `0.2740`, Pair Acc `0.6610`, List Acc `0.3195`,
  List PairRel `0.5570`, Rank MAE `0.6283`.
- original separate Stage-2 then Stage-3 with Stage-3 pointwise/pairwise
  replay `1/1`: Point Acc `0.2675`, Pair Acc `0.6842`,
  List Acc `0.3570`, List PairRel `0.5857`, Rank MAE `0.5873`.

Best fixed-selection merged Stage23 variants:

- `list_weight=2.0`, `pair_weight=1.0`, `pointwise_weight=0.5`:
  Point Acc `0.2705`, Pair Acc `0.6887`, List Acc `0.3710`,
  List PairRel `0.5922`, Rank MAE `0.5705`.
- `stage23_epochs=2`, `pair/list/pointwise=1/1/0.5`:
  Point Acc `0.2945`, Pair Acc `0.6858`, List Acc `0.3675`,
  List PairRel `0.5920`, Rank MAE `0.5722`.
- `list_weight=1.5`, `pair_weight=1.0`, `pointwise_weight=0.5`:
  Point Acc `0.2645`, Pair Acc `0.6832`, List Acc `0.3640`,
  List PairRel `0.5860`, Rank MAE `0.5765`.

Observed patterns:

- The default merged Stage23 `1/1/0.5` underperforms the original separate
  flow on ranking metrics. The issue is not merge itself; it is the mixture
  and exposure setting.
- Increasing listwise weight is the cleanest ranking improvement. `list=2`
  beats the separate flow on List Acc, PairRel, Rank MAE, and Pair Acc.
- Increasing Stage23 exposure to 2 epochs is the best all-round fixed-selection
  variant: highest pointwise accuracy among fixed-selection runs and nearly
  tied best ranking metrics.
- Pairwise weight increases Pair Acc monotonically in this sweep
  (`pair=0.5` -> `0.5752`, `1.0` -> `0.6610`, `1.5` -> `0.6892`,
  `2.0` -> `0.6957`), but ranking improves less than with listwise weight.
- Pointwise replay is necessary. `pointwise_weight=0` damages pointwise and
  ranking; `0.75` is stronger than `0.5`/`1.0` for one-epoch ranking, but
  `stage23_epochs=2` is better overall.
- DER/LwF-style pointwise teacher distillation at weights `0.1` and `0.3`
  does not help materially here and worsens pointwise MAE.
- LM-head proxy-reuse merged Stage23 remains weak versus its original
  separate-flow counterpart: merge `List Acc 0.2125`, original separate
  `0.2705`. That branch likely needs the same list/exposure tuning before
  judging the selection method itself.

## 2026-07-31 - 1B Default Checkpoint Correction

Recorded a stable project rule: future 1B experiments default to
`llama/Llama-3.2-1B-Instruct`. The plain `llama/Llama-3.2-1B` base checkpoint
is only for explicitly requested base ablations/comparisons, and output names
must include `base` when the base checkpoint is used.

The completed 2026-07-30 LM-head proxy-reuse run with
`llama/Llama-3.2-1B` should be treated as a base-checkpoint comparison record,
not as the main 1B result. Relaunch the same no-smooth, Stage-3
pointwise/pairwise replay 1/1, LM-head proxy-reuse setting with
`llama/Llama-3.2-1B-Instruct`.

Relaunched the corrected Instruct LM-head proxy-reuse run on GPU 5.

Script: `launch_llama3p2_1b_instruct_lmhead_reuse_20260731.sh`

Tmux: `l1b_instruct_lmhead_reuse_g5`

Output:
`outputs/llama3p2_1b_instruct_alpaca_b600_lmheadproxy_reuse_skipstage1_stage3pw1pair1_nosmooth_20260731/`

Log:
`outputs/llama3p2_1b_instruct_lmheadproxy_reuse_logs_20260731/llama3p2_1b_instruct_alpaca_b600_lmheadproxy_reuse_skipstage1_stage3pw1pair1_nosmooth.log`

Config was verified after launch:
`llama=Llama-3.2-1B-Instruct`, `candidate_selector_proxy_mode=lm_head`,
`reuse_selection_proxy_for_stage1=true`, Stage-2 pointwise replay `0`,
Stage-3 pointwise/pairwise replay `1/1`, and
`pointwise_global_smooth_alpha=0`.

Added a stable selection-reuse rule: by default, new random/proxy three-stage
experiments should run their own selection and should not reuse
`selected_triples.jsonl` through `--fixed-selected-triples-path`. Fixed
selection is only for explicitly requested control-variable ablations, and run
names should say `fixed_random_*` or `fixed_pointproxy_*` with the source path
recorded.

Corrected replay-ratio semantics in
`run_pointwise5answers_three_stage_pairwise_listwise_sft_v1.py`: pointwise
replay ratio `1` now means replay one copy of the available pointwise training
examples. For the standard 600-budget three-answer setting, Stage 3 should
train on `1200 listwise + 600 pointwise replay + 1200 pairwise replay` rather
than repeating pointwise to `1200`. The same correction applies to merged
Stage23 ablations. Oversampling the 600 pointwise examples to 1200 is now only
for an explicit oversampling ablation.

The Instruct LM-head proxy-reuse run that had started at
`2026-07-31T05:53:27Z` used the old in-memory replay code and was interrupted
before completion. Its partial output/log were archived with suffix
`aborted_old_pointwise_replay_20260731T060352Z`. The same run was restarted at
`2026-07-31T06:04:00Z` in tmux `l1b_instruct_lmhead_reuse_g5` using the
corrected code; config was verified again with Instruct checkpoint,
`lm_head` proxy reuse, no smoothing, Stage-2 pointwise replay `0`, and Stage-3
pointwise/pairwise replay `1/1`.

Launched two Llama-3.2-1B-Instruct ep6 controls for the current merged
Stage23 exploration. These are exposure-matched to the corrected merged
Stage1+Stage23 total of 3600 training examples and are not fixed-selection
reuse runs.

- One-answer pointwise control: tmux `l1b_ep6_one_g0`, GPU 0,
  `outputs/llama3p2_1b_instruct_alpaca_one_answer_random600_ep6_merge23match_20260731/`.
  Config verified: `pointwise_only_one_answer`, 600 pointwise samples,
  6 pointwise epochs, LoRA/4bit, no replay, no smoothing.
- True-val mixed control: tmux `l1b_ep6_mix_g1`, GPU 1,
  `outputs/llama3p2_1b_instruct_alpaca_trueval_mix_200pw_200pair_200list_ep6_merge23match_20260731/`.
  Config verified: 200 pointwise, 200 pairwise, 200 listwise samples, 6 epochs
  per stage, LoRA/4bit, no replay, no smoothing.

Launched the missing clean merged-Stage23 pointwise-replay-600 ablation. This
is a fixed-selection control using the same pointproxy-selected triples as the
other merged-Stage23 variants, but without pointwise teacher distillation.

- Script: `launch_llama3p2_1b_instruct_merge23_clean_pw600_20260731.sh`
- Tmux: `l1b_merge23_clean_pw600_g0`, GPU 0
- Output:
  `outputs/llama3p2_1b_instruct_alpaca_b600_fixedpointproxy_entropy100_stage23merged_pair1_list1_pw0p5_nodistill_nosmooth_20260731/`
- Log:
  `outputs/llama3p2_1b_instruct_alpaca_continual_logs_20260731/llama3p2_1b_instruct_alpaca_b600_fixedpointproxy_entropy100_stage23merged_pair1_list1_pw0p5_nodistill_nosmooth.log`
- Verified config: `Llama-3.2-1B-Instruct`, `merge_stage2_stage3=true`,
  Stage23 weights pair/list/pointwise `1/1/0.5`,
  `pointwise_teacher_distill_weight=0`, no smoothing, LoRA/4bit.
  Expected Stage23 composition is `1200 pairwise + 1200 listwise + 600
  pointwise replay`.

Launched the matching LM-head proxy-reuse merged-Stage23 pointwise-replay-600
run. Unlike the fixed-selection control above, this reruns LM-head active
selection and reuses the LM-head selector proxy as Stage 1.

- Script: `launch_llama3p2_1b_instruct_lmhead_reuse_merge23_pw600_20260731.sh`
- Tmux: `l1b_lmhead_merge23_pw600_g1`, GPU 1
- Output:
  `outputs/llama3p2_1b_instruct_alpaca_b600_lmheadproxy_reuse_stage23merged_pair1_list1_pw0p5_nodistill_nosmooth_20260731/`
- Log:
  `outputs/llama3p2_1b_instruct_lmheadproxy_reuse_logs_20260731/llama3p2_1b_instruct_alpaca_b600_lmheadproxy_reuse_stage23merged_pair1_list1_pw0p5_nodistill_nosmooth.log`
- Verified config: `train_selection_mode=candidate_triple_selector`,
  `candidate_selector_kind=pointwise_proxy`,
  `candidate_selector_proxy_mode=lm_head`,
  `reuse_selection_proxy_for_stage1=true`, `merge_stage2_stage3=true`,
  Stage23 weights pair/list/pointwise `1/1/0.5`,
  `pointwise_teacher_distill_weight=0`, no smoothing, LoRA/4bit.
  Expected Stage23 composition after selection is `1200 pairwise + 1200
  listwise + 600 pointwise replay`, with Stage 1 coming from the reused
  LM-head selection proxy.

Launched a small Llama-3.2-1B-Instruct Stage23 config queue to compare the
merged Stage23 setup against the original separate Stage-2 then Stage-3 replay
flow and to probe Stage23 mixture weights. `nvidia-smi` is temporarily
unusable due to an NVML driver/library mismatch, so the queue uses
`/dev/nvidia*` memory mappings to avoid busy GPUs.

- Launcher: `launch_llama3p2_1b_instruct_stage23_job_20260731.sh`
- Scheduler: `schedule_llama3p2_1b_instruct_stage23_configs_20260731.sh`
- Tmux: `l1b_instruct_stage23_q_0731`
- Logs:
  `outputs/llama3p2_1b_instruct_stage23_config_queue_logs_20260731/`
- Already completed/skipped controls: fixed merged pair/list/pointwise
  `1/1/0.5`, LM-head original separate Stage-3 replay, one-answer ep6, and
  true-val mix ep6.
- Running at launch:
  - GPU 0:
    `outputs/llama3p2_1b_instruct_alpaca_b600_fixedpointproxy_entropy100_stage3pw1pair1_nosmooth_20260731/`
    (original separate Stage-2 then Stage-3 with pointwise/pairwise replay
    `1/1`).
  - GPU 2:
    `outputs/llama3p2_1b_instruct_alpaca_b600_fixedpointproxy_entropy100_stage23merged_pair1_list1_pw1_nodistill_nosmooth_20260731/`
    (merged Stage23 weights pair/list/pointwise `1/1/1`).
  - GPU 3:
    `outputs/llama3p2_1b_instruct_alpaca_b600_fixedpointproxy_entropy100_stage23merged_pair2_list1_pw0p5_nodistill_nosmooth_20260731/`
    (merged Stage23 weights pair/list/pointwise `2/1/0.5`).
  - Existing GPU 1 run continues:
    `outputs/llama3p2_1b_instruct_alpaca_b600_lmheadproxy_reuse_stage23merged_pair1_list1_pw0p5_nodistill_nosmooth_20260731/`.

Expanded the merged Stage23 parameter sweep after the initial queue proved too
narrow. The expanded grid is fixed-selection only so mixture changes are
isolated from selection noise. It covers pointwise replay weight, pairwise
weight, listwise weight, pointwise teacher distillation, pointwise replay
multiplier, and Stage23 exposure.

- Scheduler:
  `schedule_llama3p2_1b_instruct_stage23_param_grid_20260731.sh`
- Tmux: `l1b_instruct_stage23_grid_0731`
- Logs:
  `outputs/llama3p2_1b_instruct_stage23_param_grid_logs_20260731/`
- GPU pool corrected to `5 6 7` only. A first attempt briefly launched onto
  busy GPUs 0/2/3 because `/dev/nvidia*` probing was not conservative enough;
  those partial outputs were archived with `aborted_overcommit` or
  `aborted_scheduler_restart` suffixes and the queue was restarted on the
  free-card pool only.
- Grid jobs:
  - pointwise weight sweep: `pw0`, `pw0p25`, `pw0p75` in addition to existing
    completed/running `pw0p5` and `pw1`.
  - pairwise weight sweep at `pw0p5`: `pair0p5`, `pair1p5`, plus running
    `pair2`.
  - listwise weight sweep at `pw0p5`: `list0p5`, `list1p5`, `list2`.
  - distillation sweep at `pw0p5`: DER/LwF pointwise replay weights `0.1`
    and `0.3`, temperature `2`.
  - replay/exposure ablations: pointwise replay multiplier `2`, and
    Stage23 epochs `2`.
  Running immediately after restart: `pw0` on GPU 5, `pw0p25` on GPU 6,
  and `pw0p75` on GPU 7; the remaining jobs wait for those GPUs to free.

## 2026-07-28 - Llama-3.2-1B-Instruct Tuning and Full-FT Queue

Launched a dependency-aware 59-run Alpaca queue for
`llama/Llama-3.2-1B-Instruct`. All runs keep seed 42, budget 600, max length
1024, batch size 2, gradient accumulation 8, Stage-2 pointwise replay 0, and
Stage-3 pointwise/pairwise replay 1/1 unless the historical control definition
requires no replay.

The new LoRA runs contain:

- A fixed-selection smoothing grid for Random and pure-entropy Proxy at alpha
  `0.005, 0.01, 0.02, 0.04, 0.06, 0.10`. Existing alpha `0` and `0.03` runs
  complete the eight-point comparison for each selection method.
- Six smoothing mechanism checks: Stage-1 only, Stage-3 only, frozen Stage-1
  prior in Stage 3, 10% uniform prior shrinkage, entropy-adaptive alpha, and a
  trainable alpha.
- Fourteen no-smooth proxy-selector variations covering entropy versus ordinal
  score standard deviation, predicted score-bin coverage, a labeled 10%
  exploration ablation, initialization size, query batch size, proxy warmup
  and update epochs, and proxy learning rate. All variants retain full-pool
  scoring (`max_score_candidates=0`).
- Five fixed-selection selector+smoothing combinations at alpha `0.01`.

The full-finetuning branch contains all eight established controls: one-answer
ep1/ep9, mix ep1/ep9, Random, pure-entropy Proxy, and matched Random/Proxy
alpha `0.03`. It also adds six LR scans (`1e-5, 2e-5, 5e-5` for Random and
Proxy), four alpha checks (`0.01, 0.04` for each), and four tuned-selector
selections. Full FT disables both LoRA and 4-bit loading. Selection JSONL files
are shared with the LoRA comparisons so that the final training method is the
only changed variable.

Worker: `launch_llama3p2_1b_instruct_tuning_20260728.sh`

Scheduler: `schedule_llama3p2_1b_instruct_tuning_20260728.sh`

Logs: `outputs/llama3p2_1b_instruct_alpaca_tuning_logs_20260728/`

Runtime status: scheduler tmux `l1b_instruct_tune_0728`. The initial jobs use
GPUs 0, 1, 2, 3, and 7; GPUs 4, 5, and 6 remain reserved automatically because
other processes leave less than 36 GiB free. The first full-FT one-answer ep1
run is the gate for all remaining full-FT jobs. It successfully loaded the
model in full mode (`use_lora=false`, `load_in_4bit=false`) and entered
training; its first epoch finished without an OOM before evaluation began.

## 2026-07-24 - Reward-Model Proxy Selection Protocol / Reward Model 代理选样协议

The protocol applies specifically to Reward Model class datasets with grouped
continuous rewards, such as Skywork and ARMO. It does not apply to the
LLM-as-a-Judge, Alpaca/Dolly classifier-head, or generative three-stage
pipelines. The active selector is the training proxy itself; BERT/Longformer
selectors are excluded from this Reward Model main-experiment matrix.

Canonical settings (superseded later on 2026-07-24; see correction below):

- Proxy acquisition uses equal question-level weights: 50% individual-answer
  predictive uncertainty and 50% spread among the three answers' predicted
  mean rewards. Individual uncertainty is aggregated within a question as
  `0.75 * mean + 0.25 * max`; response spread is the standard deviation of the
  three predicted means. This is distinct from the older classifier-head
  entropy/score-distribution-std combination.
- Each query batch retains 10% random exploration.
- The proxy scores the full remaining unlabeled set each round; candidate-pool
  limiting is disabled with `--selector-max-score-candidates 0`.
- Question-level proxy and random experiments each run both alpha `0` and
  alpha `0.01` as a matched no-smooth/smooth comparison.
- Answer-level proxy and random experiments use no smoothing (`alpha=0`).
- The existing `selector_answer` is BERT-based and is not part of the new main
  protocol. If an answer-level active method is required, implement and use
  `proxy_answer`.

These rules apply to the main Reward Model pointwise matrix, including the
planned ARMO replication. Historical BERT runs remain valid audit/ablation
records. Future deviations must be explicitly labeled as ablations. No ARMO
training experiments were launched while recording this decision.

该协议只适用于 Skywork、ARMO 等 grouped continuous-reward 的 Reward Model 类
数据集，不覆盖 LLM-as-a-Judge、Alpaca/Dolly classifier-head 或生成式三阶段实验。
Reward Model pointwise 主实验默认直接使用训练中的 proxy 作为选择器，不再把
BERT/Longformer selector 纳入主实验矩阵。
Proxy acquisition 使用 50% MC 预测不确定性与 50% 预测回答分数标准差，每轮保留
10% 随机探索，并对全部剩余未标注样本打分，不限制候选池。Question-level 的
proxy 与 random 都分别运行 `alpha=0` 和 `alpha=0.01`；answer-level 两者只运行
`alpha=0`。
现有 `selector_answer` 是 BERT-based，不再使用；需要 answer-level 主动选样时应
实现 `proxy_answer`。本次只记录决策，未启动 ARMO 训练。

### ARMO launch / ARMO 启动

Implemented `proxy_answer` in `run_skywork_pointwise.py`. It combines each
answer's normalized MC-dropout uncertainty with the normalized prediction
spread among the remaining answers from the same question, using the frozen
50/50 weights. Selection uses 90% highest acquisition scores and 10% random
exploration. A budget-6 Qwen3-0.6B GPU smoke run completed selection, proxy
updates, final training, validation, and metrics output successfully.

Initially launched the 16-run ARMO subset with
`launch_armo_rewardmodel_main_20260724.sh` in tmux session
`armo_rm_main_0724`. Qwen3-0.6B/1.7B/4B/8B run on GPUs 2/3/5/6 respectively;
each GPU processes `random`, `random_answer`, `proxy`, and `proxy_answer`
serially. This first queue contains question-level alpha `0.01` and
answer-level alpha `0`. All proxy modes use full-pool scoring, 50/50
acquisition, and 10%
exploration. The first four processes were verified alive after model loading
with active GPU memory and compute utilization.

Logs: `outputs/armo_rewardmodel_main_logs_20260724/`.

已在 `run_skywork_pointwise.py` 中实现 `proxy_answer`，并用 Qwen3-0.6B、预算 6
完成端到端 GPU smoke。首批 16 项 ARMO 实验子集已通过
`launch_armo_rewardmodel_main_20260724.sh` 启动于 tmux
`armo_rm_main_0724`。Qwen3-0.6B/1.7B/4B/8B 分别使用 GPU 2/3/5/6；每张卡串行
运行 `random`、`random_answer`、`proxy`、`proxy_answer`。该队列包含
question-level `alpha=0.01` 与 answer-level `alpha=0`。已核实首批四个进程完成
模型加载并实际占用 GPU 计算资源。

Correction: question-level experiments require both alpha `0` and `0.01`, not
only alpha `0.01`. The four running random-alpha-0.01 jobs remain valid. Eight
missing no-smooth jobs (`random` and `proxy` across four Qwen3 sizes) are
dependency-queued after the current queue. The corrected full ARMO matrix is
24 runs: 16 question-level runs plus 8 answer-level no-smooth runs.

更正：question-level 必须同时运行 `alpha=0` 与 `0.01`，而不是只运行
`alpha=0.01`。当前四个 random-alpha-0.01 任务仍是有效实验并继续运行。四种
Qwen3 尺寸缺少的 `random/proxy × alpha=0` 共 8 项将在当前队列结束后依赖补跑。
修正后的 ARMO 完整矩阵共 24 项：16 项 question-level 加 8 项 answer-level。

### Acquisition correction: pure uncertainty / 选样更正：纯不确定性

The Reward Model proxy-selector standard was changed before any formal 50/50
proxy run began. Proxy selection now uses only MC-dropout predictive
uncertainty (`uncertainty_weight=1`, `response_std_weight=0`), with no random
exploration (`exploration_ratio=0`) and no candidate-pool cap. In this
continuous regression implementation there is no categorical probability
entropy; predictive standard deviation is the operational pure-uncertainty
score and gives the same ordering as Gaussian predictive entropy.

At correction time, only four random alpha-0.01 jobs were active; no 50/50
proxy job had started or completed. Stopping the old queue also stopped those
four incomplete random jobs. Their partial output directories were archived,
not used as results. The corrected 24-run matrix is restarted from scratch
with output names containing `fullpool_entropy_noexplore`.

在任何正式 50/50 proxy 实验启动前，Reward Model proxy 规则改为纯 MC-dropout
预测不确定性：`uncertainty_weight=1`、`response_std_weight=0`、无随机探索且全量
候选池。连续回归没有分类概率 entropy；实际使用 predictive standard deviation，
其排序与高斯预测熵单调等价。更正时仅有四个 random alpha-0.01 任务在运行，没有
任何 50/50 proxy 已启动或完成。旧队列停止时这四个未完成 random 也退出，其部分
输出已归档，不作为结果。修正后的 24 项矩阵以
`fullpool_entropy_noexplore` 命名并从头启动。

Corrected logs: `outputs/armo_rewardmodel_entropy_noexplore_logs_20260724/`.
Archived partial outputs:
`outputs/aborted_armo_balanced_explore10_20260724_1052/`.

## 2026-07-24 - Missing Alpaca Entropy-Only Proxy Runs / 补跑纯熵代理选样

Queued the missing Alpaca budget-600 entropy-only direct-proxy runs for
Qwen3-0.6B and Llama-3-8B-Instruct. The scheduler waits for genuinely idle
GPUs by checking active compute processes and does not use GPU 4.

The previous Llama run completed selection and Stage-1 training but failed in
padded batch evaluation. Root cause: the loader added a new `[PAD]` token to
the tokenizer before model loading and therefore never resized the model's
embedding table. The loader now adds missing tokens after model loading and
resizes tokenizer/model together. Qwen tokenizers with an existing pad token
are unaffected.

Scheduler: `schedule_alpaca_entropy_proxy_missing_20260724.sh`.
Worker: `launch_alpaca_entropy_proxy_missing_20260724.sh`.
Logs: `outputs/alpaca_entropy_proxy_missing_logs_20260724/`.

### Protocol revision / 协议修订

The initial Qwen3-0.6B and Llama-3-8B entropy-only jobs used a 1,000-question
scoring pool. They were stopped before completion after the 1-10 judge
protocol was frozen and remain only as incomplete audit outputs. Replacement
runs use pure categorical predictive entropy, zero exploration, and the full
remaining candidate pool (`candidate_selector_max_score_candidates=0`).

## 2026-07-23 - Standalone Pointwise SFT Train-Only Entry Point / 独立单条 Pointwise 训练入口

Added `run_pointwise_sft_train_only.py` to isolate the generative pointwise
training used by Stage 1. It accepts explicit JSON/JSONL training data,
supports flattened pointwise rows and grouped A/B/C or 1..5 answers, and
trains only the `Score: [1..10]` generation target. It has no evaluation,
selector, smoothing, pairwise stage, or listwise stage.

The entry point supports LoRA, 4-bit loading, full training, sample caps, and
a prepare-only data validation mode. A two-example Qwen3-0.6B LoRA+4-bit
smoke run completed one optimizer step and saved a reloadable adapter and
tokenizer under `/tmp/pointwise_train_only_smoke_768_20260723/`.

## 2026-07-22 - Alpaca/Dolly Direct-Proxy Matrix Fill / 补齐代理自主选样矩阵

Added an idle-GPU scheduler for the missing canonical balanced direct
pointwise-proxy runs. It checks real GPU compute processes before launch,
reserves GPU 4 for the existing process, skips completed outputs, and records
per-job completion/failure markers without preventing independent jobs from
continuing.

The fill matrix contains Alpaca Qwen3-0.6B/14B/30B-A3B and Llama-3-8B, plus
Dolly-4096 Qwen3-0.6B/1.7B/4B/8B/14B/30B-A3B. Existing completed Alpaca
Qwen3-1.7B/4B/8B and Dolly Llama-3-8B runs are reused rather than repeated.
All new runs use budget 600, seed 42, an 80-triple warmup, 20-triple query
batches, a 1,000-triple scoring pool, balanced entropy/score-std acquisition,
and no smoothing. Dolly keeps the established 10% exploration setting and
4,096-token proxy/SFT context; Alpaca keeps no exploration and the established
1,024-token SFT context.

Scheduler: `schedule_direct_proxy_fill_20260722.sh`.
Worker: `launch_direct_proxy_fill_worker_20260722.sh`.
Logs: `outputs/direct_proxy_fill_logs_20260722/`.

本文档是按时间顺序维护的项目交接日志。稳定规则和长期项目知识保存在
`PROJECT_MEMORY.md`；实验、代码变更、结果和后续行动记录在这里。

## 2026-08-09 - Repeat Result-Only Weight Cleanup / 再次清理权重，仅保留结果

### Rule / 规则

For every output directory with `summary.json` or `metrics_compact.json`,
remove model weight tensors, checkpoints, optimizer/scheduler states, and
empty model directories. Preserve top-level training parameter files such as
`config.json`, argument/command records, metrics, summaries, selected data,
and logs. Directories without a final result file remain untouched.

对于包含 `summary.json` 或 `metrics_compact.json` 的输出目录，删除模型权重张量、
checkpoint、optimizer/scheduler 状态和清空后的模型目录；保留顶层训练参数文件，
包括 `config.json`、参数/命令记录、指标、summary、选样数据和日志。没有最终结果
文件的目录完全不动。

### Result / 结果

- Processed 428 result directories and protected 205 directories without final results.
- Freed `27,698,696,192` bytes (about `27.70 GB`).
- Verification: 508 top-level `config.json` files remain.
- Verification: result directories with remaining model weights: `0`.
- Protected no-final-result directories still containing weights: `23`, about `1.21 GB`.
- `outputs/` now uses about `23 GB`.

- 处理了 428 个有结果的目录，保护了 205 个没有最终结果的目录。
- 释放 `27,698,696,192` 字节（约 `27.70 GB`）。
- 复核确认：仍保留 508 个顶层 `config.json` 参数文件。
- 复核确认：有结果目录中残留模型权重的数量为 `0`。
- 仍保留权重的无最终结果目录有 `23` 个，约占 `1.21 GB`。
- `outputs/` 当前占用约 `23 GB`。

## 2026-07-24 - Completed-Run Weight Cleanup / 清理已完成实验的权重

### Retention Rule / 保留规则

Delete model weights, checkpoints, and trainer-state tensors only from output
directories that already contain `summary.json` or `metrics_compact.json`.
Protect every output directory referenced by a live process through `--out`,
and also protect directories without a final result file.

仅从已经包含 `summary.json` 或 `metrics_compact.json` 的输出目录中删除模型
权重、checkpoint 和训练器状态张量。所有被活跃进程通过 `--out` 引用的目录，
以及尚无最终结果文件的目录，均予以保护。

### Result / 结果

- Removed weights from 105 completed experiment directories.
- Freed approximately `955,685,538,650` bytes (`955.7 GB` / `890.2 GiB`).
- Preserved all configs, metrics, selected data, statistics, and logs.
- Protected six actively training output directories identified from live
  command lines.
- Protected five non-final directories containing about `1.33 GB` of weights.
- Final verification found zero completed, inactive result directories with
  remaining model weights.
- `outputs/` now uses about `5.2 GiB`.

- 已清理 105 个已完成实验目录中的权重。
- 释放约 `955,685,538,650` 字节（`955.7 GB` / `890.2 GiB`）。
- 所有配置、指标、选样数据、统计信息和日志均已保留。
- 根据活跃进程命令行识别并保护了 6 个正在训练的输出目录。
- 另保护了 5 个尚无最终结果的目录，其中权重约 `1.33 GB`。
- 最终复核确认：已完成且非活跃的结果目录中，剩余模型权重数量为 0。
- `outputs/` 当前占用约 `5.2 GiB`。

## 2026-07-19 - Corrected Dolly 4096 Long-Context Selector / 修正 Dolly 4096 长上下文 Selector

The first Dolly 4096 selector run used `bert-base-uncased` with its native
512-token limit. Its Llama SFT context was 4096, but the selector itself was
not a 4096 experiment; its metrics are retained only as an audit record.

The selector now supports Longformer global attention and uses a local
`allenai/longformer-base-4096` checkpoint with `candidate_bert_selector_max_length=4096`.
Long-context selector inference uses batch size 2 to control memory. A smoke
test reached 4096-capable tokenization and completed a forward pass on GPU 6.

The incorrectly configured selector-smooth process was stopped before its
final metrics were produced. Random smooth was left running. A corrected
selector + selector-smooth queue was launched with:
`launch_dolly_4096_longformer_selector_20260719.sh`.

Current corrected selector output:
`outputs/dolly4096_b600_longformer4096_pointwise_init80_stage3pw1pair1_nosmooth_20260719/`

Current status: Longformer selector is selecting candidates on GPU 6; memory
use is approximately 15.2 GiB of 40 GiB and no CUDA OOM has occurred.

### 2026-07-19 Resume / 补跑

The original corrected run stopped at `Evaluating all metrics: after_stage1`
without a Python traceback, CUDA OOM, or kernel OOM record. It had been started
inside a foreground tool execution session, so the evidence is consistent
with external process cleanup rather than a training-code failure.

The completed 200-triple Longformer-4096 selection is being reused instead of
spending another hour selecting it. A fresh no-smooth run and its exact-selection
smooth counterpart are queued in tmux session `dolly4096_longformer_resume` on
GPU 6 using `launch_dolly_4096_longformer_fixed_selection_resume_20260719.sh`.
The no-smooth run entered Stage-1 training successfully at approximately
19.2 GiB of 40 GiB GPU memory.

## 2026-07-13 - Selective Model-Weight Cleanup / 选择性清理模型权重

### Retention Rule / 保留规则

Keep model weights only for experiments whose resolved `config.json` has all
of the following:

- `budget_units = 600`
- `stage3_pointwise_replay_ratio = 1`
- `stage3_pairwise_replay_ratio = 1`

仅为 `config.json` 同时满足以下条件的实验保留模型权重：

- `budget_units = 600`
- `stage3_pointwise_replay_ratio = 1`
- `stage3_pairwise_replay_ratio = 1`

### Result / 结果

- Fully retained model weights for 31 qualifying experiments.
- Removed model/checkpoint directories and weight files from every other
  result directory; configs, metrics, selected data, statistics, and logs were
  preserved.
- Verification found zero non-qualifying directories with remaining model
  weight files.
- `outputs/` was reduced from `2,393,329,643,520` bytes to
  `711,326,957,568` bytes, freeing `1,682,002,685,952` bytes (about `1.68 TB`
  / `1.53 TiB`).

- 完整保留了 31 个符合条件的实验模型权重。
- 其他结果目录只删除模型/checkpoint 目录及权重文件；配置、指标、选样数据、
  统计信息和日志均已保留。
- 复核确认，不符合条件的目录中剩余模型权重文件数量为 0。
- `outputs/` 从 `2,393,329,643,520` 字节降至 `711,326,957,568` 字节，
  共释放 `1,682,002,685,952` 字节（约 `1.68 TB` / `1.53 TiB`）。

## 2026-07-13 - Pre-July Output Cleanup / 清理七月前的实验输出

### Goal / 目标

Remove top-level experiment output directories whose contents were last
modified before `2026-07-01 00:00:00 UTC`, while preserving July and newer
results.

删除内容最后修改时间早于 `2026-07-01 00:00:00 UTC` 的一级实验输出目录，
保留七月及之后的结果。

### Result / 结果

- Deleted 64 top-level directories under `outputs/`.
- Freed `1,391,501,873,152` bytes (about `1.39 TB` / `1.27 TiB`).
- `outputs/` now uses `2,393,329,643,520` bytes (about `2.39 TB` / `2.18 TiB`).
- 132 top-level output directories remain.
- No July or newer output directory was included in this cleanup.

- 已删除 `outputs/` 下的 64 个一级目录。
- 释放 `1,391,501,873,152` 字节（约 `1.39 TB` / `1.27 TiB`）。
- `outputs/` 当前占用 `2,393,329,643,520` 字节（约 `2.39 TB` / `2.18 TiB`）。
- 还剩 132 个一级输出目录。
- 本次清理未包含七月及之后的输出目录。

## Logging Convention / 记录规范

Each meaningful work session should append a dated entry containing:

- Goal: what was requested.
- Changes: scripts or behavior changed.
- Runs: launch script, output directories, logs, and runtime status.
- Results: comparable metrics and the interpretation.
- Next: unresolved questions or the next experiment.

每次有实质内容的工作会话都应追加一条带日期的记录，包括：

- 目标（Goal）：用户提出的需求。
- 变更（Changes）：修改的脚本或行为。
- 运行（Runs）：启动脚本、输出目录、日志和运行状态。
- 结果（Results）：可比较的指标与解释。
- 后续（Next）：未解决的问题或下一项实验。

Never record secrets or large raw logs here. Link to output directories
instead.

不要在这里记录密钥或大段原始日志；应改为链接到输出目录。

## 2026-07-21 - Dolly 4096 Direct Pointwise Proxy Selector / Dolly 4096 代理直接选样

### Goal / 目标

Run the Dolly counterpart of the direct-proxy acquisition previously used for
the reward-model experiments. The pointwise Llama proxy selects examples
directly; no BERT/Longformer error-prediction selector is trained.

运行 reward-model 实验中代理直接选样方法的 Dolly 对应实验。由 pointwise
Llama proxy 直接选择样本，不再训练 BERT/Longformer 误差预测 selector。

### Controlled Setup / 受控设置

- Llama-3-8B-Instruct, LoRA + 4-bit, seed 42, budget 600.
- Both proxy selection and final SFT use a 4,096-token maximum length.
- Random initialization 80 triples, query batch 20, and a random scoring pool
  of 1,000 triples per round.
- Acquisition uses balanced pointwise predictive entropy and ordinal score
  distribution standard deviation (`0.5/0.5`), 10% random exploration, and no
  predicted-bin coverage bonus.
- Stage-2 pointwise replay 0; Stage-3 pointwise/pairwise replay 1/1; one epoch
  per stage.
- The queued smooth run reuses the exact proxy-selected triples and applies the
  established Stage-3 fixed-prior smoothing protocol with alpha `0.01`.

- 使用 Llama-3-8B-Instruct、LoRA + 4-bit、seed 42、预算 600。
- proxy 选样和最终 SFT 的最大长度均为 4,096 token。
- 随机初始化 80 个 triple，每轮查询 20 个，每轮随机打分池为 1,000 个 triple。
- 选样均衡使用 pointwise 预测熵与序数分数分布标准差（`0.5/0.5`），保留
  10% 随机探索，不使用预测分数档覆盖奖励。
- Stage 2 pointwise replay 为 0；Stage 3 pointwise/pairwise replay 为 1/1；
  每阶段 1 epoch。
- 队列中的 smooth 实验严格复用 proxy 选样，并使用既定 Stage-3 固定先验
  smoothing，alpha 为 `0.01`。

### Memory Guard And Run / 显存保护与运行

For proxy contexts longer than 2,048 tokens, classifier-head proxy inference
and training batches are reduced to one; shorter-context behavior is unchanged.
This avoids the historical fixed batches of 4/8 at 4,096 tokens on a 40 GiB
GPU.

当 proxy 上下文超过 2,048 token 时，分类头 proxy 的推理和训练 batch 均降为
1；短上下文行为不变，以避免旧的固定 4/8 batch 在 40 GiB GPU 上运行 4,096
token 时 OOM。

Launcher: `launch_dolly_4096_pointwise_proxy_20260721.sh`

Tmux: `dolly4096_pointproxy_0721`, GPU 6.

Outputs:

- `outputs/dolly4096_b600_pointproxy_balanced_init80_pool1000_explore10_nocover_nosmooth_20260721/`
- `outputs/dolly4096_b600_fixed_pointproxy_balanced_stage3_fixedprior_u10_a001_20260721/`

The no-smooth run loaded the 4-bit proxy successfully and reached about 12.3
GiB GPU memory without CUDA OOM. The smooth run is dependency-queued after the
no-smooth run.

## 2026-07-19 - Direct Pointwise Proxy Acquisition / 直接 Pointwise Proxy 选样

### Goal / 目标

Improve the selector while keeping its acquisition objective strictly
pointwise-only. The existing frozen-BERT selector was retained for comparison.

在保持选样目标严格为纯 pointwise 的前提下改进 selector；原冻结 BERT
selector 保留作为对照。

### Diagnosis / 诊断

- The previous BERT selector learned a second-order approximation of dynamic
  proxy error from only 80 initial triples and six 20-triple query rounds.
- Its target was batch-wise min-max-normalized `1 - p(true score)`, so target
  scales changed between rounds.
- Fully greedy acquisition had no exploration or predicted score coverage.
- Final Stage-3 metrics obscure whether pointwise acquisition helped Stage 1;
  Stage-1 pointwise metrics should be the primary selector endpoint.

- 旧 BERT selector 只根据 80 个初始三元组和六轮、每轮 20 个三元组的数据，
  二次拟合动态 proxy error。
- 监督目标是逐批 min-max 归一化的 `1 - p(true score)`，不同轮次标尺不一致。
- 完全贪心选择，没有探索和预测分数覆盖约束。
- 最终 Stage-3 指标会掩盖纯 pointwise 选样对 Stage 1 的作用；selector 的主要
  终点应为 Stage-1 pointwise 指标。

### Changes / 变更

Added selector kind `pointwise_proxy` to the generative three-stage runner.
It trains the Qwen/Llama pointwise proxy on a random warmup set and then scores
the unlabeled pool directly, without a frozen-BERT error-prediction layer.

新增 `pointwise_proxy` selector：先用随机 warmup 集训练 Qwen/Llama pointwise
proxy，再由 proxy 直接为未标注池打分，不再经过冻结 BERT 的误差预测层。

Acquisition combines only pointwise signals:

- normalized predictive entropy;
- ordinal score-distribution standard deviation;
- predicted score-bin coverage based on queried pointwise labels;
- configurable random exploration.

选样只组合 pointwise 信号：预测熵、序数分数分布标准差、基于已查询 pointwise
标签的预测分数覆盖，以及可配置的随机探索。

Default experimental settings are warmup 80 triples for 3 proxy passes, query
batch 20, entropy/std weights `0.5/0.5`, coverage weight `0.2`, exploration
ratio `0.1`, and a 4,096-triple random scoring pool per round.

默认实验设置：随机 warmup 80 个三元组并训练 proxy 3 遍；query batch 20；
熵/标准差权重 `0.5/0.5`；覆盖权重 `0.2`；随机探索比例 `0.1`；每轮随机打分
4,096 个候选三元组。

### Verification / 验证

- Both modified scripts pass `py_compile`.
- A synthetic acquisition check confirmed finite scores and higher utility for
  a deliberately uncertain triple.
- An end-to-end Qwen3-0.6B budget-9 smoke run completed selection and all three
  training/evaluation stages under
  `outputs/qwen3_pointwise_proxy_smoke_20260719_v2/`.
- No formal budget-600 comparison was launched in this session.

- 两个修改脚本均通过 `py_compile`。
- 合成选样检查确认分数有限，且人为构造的不确定三元组获得更高 utility。
- Qwen3-0.6B、budget 9 的端到端 smoke run 已完成选样和全部三阶段训练/评估，
  输出位于 `outputs/qwen3_pointwise_proxy_smoke_20260719_v2/`。
- 本次尚未启动正式 budget-600 对照实验。

### Qwen3 Pointwise Acquisition Sweep / Qwen3 Pointwise 选样扫描

Launched a formal budget-600 sweep on Qwen3-1.7B, 4B, and 8B. Random
exploration and predicted score coverage are both disabled to isolate the two
pointwise uncertainty terms.

已在 Qwen3-1.7B、4B、8B 上启动正式 budget-600 扫描。随机探索与预测分数
覆盖均禁用，以单独考察两个 pointwise uncertainty 信号。

Variants / 参数组：

| Variant | Entropy weight | Score-std weight |
|---|---:|---:|
| entropy-only | 1.00 | 0.00 |
| entropy-heavy | 0.75 | 0.25 |
| balanced | 0.50 | 0.50 |
| std-heavy | 0.25 | 0.75 |
| score-std-only | 0.00 | 1.00 |

Controlled settings: seed 42, budget 600, init 80 triples, query batch 20,
proxy warmup 3 passes, one proxy update pass per queried batch, random scoring
pool 1000 per round, exploration 0, coverage 0, and no smoothing. Final SFT
and replay settings match the July 17 Qwen scale experiments.

受控设置：seed 42、预算 600、init 80、query batch 20、proxy warmup 3 遍、
每个新查询 batch 更新 proxy 1 遍、每轮随机候选池 1000、探索 0、覆盖 0、无
smoothing；最终 SFT 与 replay 设置和 7 月 17 日 Qwen 规模实验一致。

Scheduler / 调度器：
`schedule_qwen3_pointwise_proxy_sweep_20260719.sh`

Worker / 工作脚本：
`launch_qwen3_pointwise_proxy_sweep_20260719.sh`

Scheduler tmux: `qwen_pointproxy_sweep_0719`. Workers use sessions beginning
with `qpps_gpu`. GPUs `0,1,2,3,5,7` are used dynamically; GPU 4 is reserved for
an unrelated persistent process and GPU 6 for the running Dolly experiment.
The scheduler checks both its worker sessions and all GPU compute processes
before assigning a job.

调度器 tmux 为 `qwen_pointproxy_sweep_0719`；worker 会话以 `qpps_gpu` 开头。
动态使用 GPU `0,1,2,3,5,7`；GPU 4 留给无关常驻进程，GPU 6 留给正在运行的
Dolly 实验。调度器在分配任务前同时检查 worker 会话与 GPU 上的全部计算进程。

The first six jobs, entropy-only and score-std-only for all three model sizes,
were launched successfully and loaded the correct Qwen checkpoints.

首批六项实验（三个模型规模的 entropy-only 与 score-std-only）已成功启动，且
加载了正确的 Qwen checkpoint。

## 2026-07-18 - Dolly 4096-Token Experiment Queue / Dolly 4096 Token 实验队列

### Goal / 目标

Run eight requested Dolly controls with a 4,096-token SFT context on the
deduplicated shuffled 9k train set and shuffled 3k validation set.

在去重后的打乱版 9k 训练集和打乱版 3k 验证集上，以 4,096 token 的 SFT
上下文运行八项指定实验。

### Controlled Setup / 受控设置

- Llama-3-8B-Instruct, LoRA + 4-bit, seed 42, learning rate `1e-4`.
- Maximum length 4096; per-device train batch 1; gradient accumulation 16;
  evaluation batch 1; expandable CUDA allocator enabled.
- Pointwise controls: 600 one-answer-per-question samples, epochs 1 and 9.
- Mixed controls: 200 pointwise + 200 true pairwise + 200 true listwise,
  epochs 1 and 9, no replay.
- Three-stage random and pointwise-only BERT selector: budget 600; Stage-2
  pointwise replay 0; Stage-3 pointwise/pairwise replay 1/1; one epoch/stage.
- Each smooth run reuses exactly the selected triples from its corresponding
  no-smooth run. Both use Stage-3-only pointwise smoothing with fixed Stage-1
  prior, alpha `0.01`, 10% uniform prior mixing, 200 examples before smoothing,
  and a 200-example warmup.

- 使用 Llama-3-8B-Instruct、LoRA + 4-bit、seed 42、学习率 `1e-4`。
- 最大长度 4096；训练 batch 1；梯度累积 16；评估 batch 1；启用 expandable
  CUDA allocator。
- Pointwise：600 个不同问题的单答案样本，分别训练 1/9 epoch。
- Mixed：200 pointwise + 200 真实 pairwise + 200 真实 listwise，分别训练
  1/9 epoch，不使用 replay。
- Random 与纯 pointwise BERT selector 三阶段实验：预算 600；阶段 2 不回放
  pointwise；阶段 3 以 1/1 回放 pointwise/pairwise；每阶段 1 epoch。
- 两项 smooth 严格复用对应 no-smooth 选样；均只平滑阶段 3 pointwise replay，
  固定阶段 1 先验，alpha `0.01`，先验混入 10% 均匀分布，前 200 条不平滑，
  随后 200 条预热。

Launch script / 启动脚本：
`launch_dolly_4096_b600_queue_20260718.sh`

The pointwise ep1 run is launched first as a 4096-token memory validation.
Remaining jobs launch only after it enters training without CUDA OOM.

首先启动 pointwise ep1 作为 4096 token 显存验证；确认进入训练且没有 CUDA
OOM 后再启动其余队列。

### Runtime Status / 运行状态

The memory validation completed multiple optimizer steps successfully. GPU 0
used approximately 14.5 GiB during 4096-token training, leaving substantial
headroom on the 40 GiB A100. All independent jobs were then launched; smooth
jobs remain dependency-queued behind their matching no-smooth run.

显存验证已成功完成多个 optimizer step；4096 token 训练时 GPU 0 使用约
14.5 GiB，在 40 GiB A100 上仍有较大余量。随后已启动全部独立任务；smooth
实验分别在对应 no-smooth 实验后依赖排队。

| Job / 实验 | GPU | tmux | Status / 状态 |
|---|---:|---|---|
| pointwise 600 ep1 | 0 | `dolly4096_memcheck_0718` | running |
| pointwise 600 ep9 | 1 | `dolly4096_pw9_0718` | running |
| mixed 200/200/200 ep1 | 2 | `dolly4096_mix1_0718` | running |
| mixed 200/200/200 ep9 | 3 | `dolly4096_mix9_0718` | running |
| random -> same-selection smooth a0.01 | 5 | `dolly4096_randomq_0718` | random running, smooth queued |
| selector -> same-selection smooth a0.01 | 6 | `dolly4096_selectorq_0718` | selector running, smooth queued |

GPU 4 remains occupied by an unrelated process; GPU 7 is intentionally kept
free as recovery capacity.

GPU 4 仍被无关进程占用；GPU 7 有意留空，用作故障恢复余量。

## 2026-07-18 - Dolly Shuffled Train/Val Deduplication / Dolly 打乱版训练验证去重

### Goal / 目标

Extract a leakage-free 9,000-record training set from the newly shuffled
12,000-record Dolly file by removing all content-equivalent records present in
the shuffled 3,000-record validation set.

从新打乱的 12,000 条 Dolly 文件中移除新验证集包含的全部 3,000 条内容等价
记录，提取无验证泄漏的 9,000 条训练集。

### Method And Result / 方法与结果

- Matched records by instruction, input, and the unordered set of three
  `(model, output, score)` answers, so differing A/B/C permutations still count
  as the same underlying example.
- Preserved the A/B/C ordering and all fields from the shuffled training rows
  that were retained.
- Input 12,000; removed 3,000; output 9,000; verified post-filter overlap 0.
- Output: `train_with_selector/train_with_selector/data/Dolly/train9k_pointwise_pairwise_no_val_overlap.json`.

- 根据 instruction、input 以及忽略 A/B/C 顺序的三组 `(model, output, score)`
  匹配内容，因此不同答案排列仍会识别为同一底层样本。
- 保留剩余训练记录已打乱的 A/B/C 顺序及全部字段。
- 输入 12,000 条，移除 3,000 条，输出 9,000 条；过滤后重叠为 0。
- 输出文件：`train_with_selector/train_with_selector/data/Dolly/train9k_pointwise_pairwise_no_val_overlap.json`。

## 2026-07-17 - Alpaca Qwen3 Surrogate-Scale Queue / Alpaca Qwen3 代理规模实验队列

### Goal / 目标

Run the requested Alpaca experiments sequentially for Qwen3-0.6B, 1.7B, 4B,
and 8B. Each model runs the same eight configurations before the queue moves
to the next model size.

按 Qwen3-0.6B、1.7B、4B、8B 的顺序运行 Alpaca 实验；每个模型依次完成相同
的八项配置后，再进入下一个模型规模。

### Controlled Setup / 受控设置

- Seed 42; LoRA + 4-bit; one epoch per active training stage; learning rate
  `1e-4`; max length 1024; batch size 2; gradient accumulation 8.
- One-answer baseline: 600 random one-answer-per-question pointwise samples.
- Mix baseline: 200 pointwise + 200 true pairwise + 200 true listwise, with no
  replay, matching the historical control.
- Three-stage runs: budget 600, Stage-2 pointwise replay 0, Stage-3
  pointwise/pairwise replay 1/1, with pairwise/listwise order augmentation.
- Selector: pointwise-only BERT selector, init 80 triples, query batch 20,
  four selector epochs, uncertainty weights `1/0/0`.
- Stage1+Stage3 smoothing uses start step 20 and warmup 20. Smooth runs reuse
  the exact triples selected by their corresponding no-smooth run.

- seed 为 42；LoRA + 4-bit；每个有效训练阶段 1 epoch；学习率 `1e-4`；
  最大长度 1024；batch size 2；梯度累积 8。
- One-answer 基线：从不同问题随机选择一个答案，共 600 条 pointwise 样本。
- Mix 基线：200 pointwise + 200 真实 pairwise + 200 真实 listwise，不使用
  replay，与历史控制实验一致。
- 三阶段实验：预算 600；阶段 2 pointwise replay 为 0；阶段 3 的
  pointwise/pairwise replay 为 1/1；启用 pairwise/listwise 顺序增强。
- Selector：纯 pointwise BERT selector，init 80，query batch 20，训练 4 个
  selector epoch，不确定性权重为 `1/0/0`。
- Stage1+Stage3 smoothing 从 step 20 开始并预热 20 step；smooth 实验严格复用
  对应 no-smooth 实验的选样。

Per-model order / 每个模型的实验顺序：

1. One-answer random 600.
2. Mix 200/200/200.
3. Random three-stage.
4. Selector three-stage.
5. Fixed-random smoothing `alpha=0.03`, Stage 1 + Stage 3.
6. Fixed-selector smoothing `alpha=0.03`, Stage 1 + Stage 3.
7. Fixed-selector smoothing `alpha=0.05`, Stage 1 + Stage 3.
8. Fixed-selector smoothing `alpha=0.03`, Stage 3 only.

### Compatibility Changes / 兼容性变更

- Updated global-prior smoothing to score complete token sequences. Qwen3
  tokenizes score `10` as `["1", "0"]`; it is now distinct from score `1`
  instead of sharing the first token in the soft target.
- Avoided a no-op embedding resize when no special token is added. This keeps
  Qwen PEFT outputs adapter-sized instead of saving full input/output embedding
  matrices at every stage.
- SFT target preprocessing now replaces the historical literal `</s>`
  placeholder with the active tokenizer's real EOS token. Qwen3 therefore
  trains on `<|im_end|>` instead of treating `</s>` as ordinary text.
- Completed end-to-end no-smooth, smoothing, and selector smoke tests on
  Qwen3-0.6B.

- Global-prior smoothing 现按完整 token 序列计算分数概率。Qwen3 会将分数
  `10` 切分为 `["1", "0"]`，现在不会再因共享首 token 而与分数 `1` 混淆。
- 当没有新增 special token 时不再执行无效 embedding resize，避免每个阶段保存
  完整输入/输出 embedding，使 Qwen PEFT 输出保持为 adapter 规模。
- SFT target 预处理现在会把历史字面量 `</s>` 占位符替换为当前 tokenizer 的真实
  EOS；Qwen3 因而学习 `<|im_end|>`，而不是把 `</s>` 当作普通文本。
- Qwen3-0.6B 的 no-smooth、smoothing 和 selector 端到端 smoke test 均已通过。

Launch script / 启动脚本：
`launch_qwen3_scale_alpaca_b600_queue_20260717.sh`

Log directory / 日志目录：
`outputs/qwen3_scale_alpaca_b600_logs_20260717/`

Runtime status / 运行状态：

- Launched in tmux session `qwen3_scale_alpaca_0717` on GPU 0.
- The first job, Qwen3-0.6B one-answer random 600, passed data/model loading
  and entered Stage-1 training successfully.
- The queue stops on the first failed or incomplete run; completed outputs are
  skipped safely when the launcher is rerun.

- 已在 GPU 0 的 tmux 会话 `qwen3_scale_alpaca_0717` 中启动。
- 首项 Qwen3-0.6B one-answer random 600 已通过数据和模型加载，并成功进入
  Stage-1 训练。
- 队列遇到首个失败或不完整实验即停止；重新运行启动脚本时会安全跳过已完成输出。

### Parallel Scheduling Update / 并行调度更新

The original strict serial interpretation was replaced after clarification:
model sizes may run concurrently. A dependency-aware scheduler now uses GPUs
`0,1,2,3,5,6,7`; GPU 4 remains reserved for an unrelated existing process.
Random/selector smooth jobs start only after their corresponding no-smooth
selection has completed.

用户澄清模型规模之间也不要求串行后，已改为依赖感知并行调度。当前使用 GPU
`0,1,2,3,5,6,7`，GPU 4 保留给已有的无关进程。Random/selector smooth 仅在
对应 no-smooth 选样完成后启动。

Parallel scheduler / 并行调度器：
`schedule_qwen3_scale_alpaca_parallel_20260717.sh`

Tmux sessions / Tmux 会话：

- Scheduler: `qwen3_scale_sched_0717`.
- Workers use names beginning with `qs_gpu`.

At the first parallel launch, three remaining Qwen3-0.6B selector-smoothing
runs and the four independent Qwen3-1.7B baselines were assigned across all
seven available GPUs. The previously running 0.6B `selector alpha=0.03`
Stage1+Stage3 run was interrupted while switching away from the serial parent
queue; its incomplete directory was archived with suffix
`_interrupted_at_stage3_step196`, and the formal run restarted cleanly.

首次并行启动时，0.6B 剩余的三个 selector smoothing 实验与 1.7B 的四个独立
基线已分配到七张可用 GPU。切换串行父队列时，原 0.6B `selector alpha=0.03`
Stage1+Stage3 在 Stage3 step 196 被中断；不完整目录已加后缀
`_interrupted_at_stage3_step196` 归档，正式实验已从干净输出重新启动。

## 2026-07-15 - Dolly Three-Stage Initial Baselines / Dolly 三阶段首轮基线

### Goal / 目标

Run the first controlled experiments on the newly added Dolly data without
correcting its fixed A/B/C score ordering, as explicitly requested.

按当前要求，暂不修正 Dolly 数据中 A/B/C 按分数排列的问题，先进行首轮受控实验。

### Setup / 设置

- Train: `data/Dolly/gpt4all_pointwise_pairwise_train9k.json` (9,000 triples).
- Eval: `data/Dolly/gpt4all_pointwise_pairwise_listwise_val3k.json` (3,000 triples).
- Budget 600 / 200 selected triples; seed 42; no smoothing.
- Stage 1 pointwise; Stage 2 pairwise with no pointwise replay; Stage 3
  listwise with pointwise/pairwise replay ratios 1/1.
- Compare random selection against the prior strongest BERT multi-target
  selector setting (pointwise/pairwise/listwise weights 0.6/0.2/0.2).

- 训练集为 9,000 个 Dolly 三元组；验证集为 3,000 个 Dolly 三元组。
- 预算 600（200 个三元组），seed 42，不使用平滑。
- 阶段 1 为 pointwise；阶段 2 为 pairwise 且不回放 pointwise；阶段 3 为
  listwise，并以 1/1 回放 pointwise/pairwise。
- 对比随机选样与此前综合表现最好的 BERT 多目标 selector 配置
  （pointwise/pairwise/listwise 权重 0.6/0.2/0.2）。

Launch script / 启动脚本：
`launch_dolly_three_stage_b600_baselines_20260715.sh`

Expected outputs / 预期输出：

- `outputs/dolly_three_stage_sft_b600_random_stage3pw1pair1_nosmooth_20260715/`
- `outputs/dolly_three_stage_sft_b600_bert_multitarget_p60_pair20_list20_init80_stage3pw1pair1_nosmooth_20260715/`

Runtime status / 运行状态：

- Random baseline: tmux `dolly_b600_random_0715`, GPU 0.
- BERT multi-target selector: tmux `dolly_b600_selector_0715`, GPU 1.
- Both processes passed dataset loading and entered model loading successfully.
- Random-run materialized sizes: 600 pointwise train, 1,200 pairwise train,
  1,200 listwise train; 900 pointwise eval, 9,000 pairwise eval, and 3,000
  listwise eval examples.

- 随机基线运行于 tmux `dolly_b600_random_0715`、GPU 0。
- BERT 多目标 selector 运行于 tmux `dolly_b600_selector_0715`、GPU 1。
- 两项进程均已通过数据加载并成功进入模型加载阶段。
- 随机实验实际生成 600 个 pointwise、1,200 个 pairwise、1,200 个 listwise
  训练样本；评估样本分别为 900、9,000 和 3,000 个。

### Extended Queue / 扩展实验队列

Additional requested controls use the definitions from the earlier Newnew
experiments:

- BERT pointwise-only selector (`1/0/0` target weights), budget 600, followed
  by a smooth run that reuses exactly the same selected triples.
- Pointwise-only: 600 one-answer-per-question samples, epochs 1 and 9.
- Mixed: 200 pointwise + 200 true pairwise + 200 true listwise samples, epochs
  1 and 9. As in the historical controls, these mixed runs do not use replay;
  sampled pairwise/listwise validation units are excluded from evaluation.
- Smooth setting: Stage-3 pointwise replay only, fixed Stage-1 prior, alpha
  `0.01`, 10% uniform prior mixing, first 200 replay examples unsmoothed and
  200-example warmup.

新增实验沿用此前 Newnew 对照实验的定义：

- BERT 单 pointwise selector（目标权重 `1/0/0`），预算 600；随后严格复用
  同一组选样运行 smooth 对照。
- Pointwise-only：从不同问题各取一个答案，共 600 条，分别训练 1/9 epoch。
- Mixed：200 pointwise + 200 真实 pairwise + 200 真实 listwise，分别训练
  1/9 epoch。与历史对照一致，mixed 不使用 replay；用于训练的验证样本会从
  对应评估集中排除。
- Smooth：仅平滑阶段 3 的 pointwise replay；固定阶段 1 先验，alpha `0.01`，
  先验混入 10% 均匀分布，前 200 条不平滑，随后用 200 条预热。

Compatibility changes / 兼容性变更：

- Added Dolly aliases `pairwise_ab_choice` and `pairwise_bc_choice` to the ABC
  pairwise loader. A data-only check produced 200 pairwise train / 5,800 eval
  examples and 200 listwise train / 2,800 eval examples.
- Updated the older true-validation control config builder for fields newly
  required by the current shared three-stage `RunConfig`.

Launch script / 启动脚本：
`launch_dolly_b600_extended_queue_20260715.sh`

Runtime queue / 运行队列：

| Job / 实验 | GPU | tmux | Status / 状态 |
|---|---:|---|---|
| pointwise selector -> same-selection smooth | 2 | `dolly_pointsel_smooth_q_0715` | running, smooth queued |
| pointwise 600 ep1 | 3 | `dolly_point_ep1_0715` | running |
| pointwise 600 ep9 | 5 | `dolly_point_ep9_0715` | running |
| mixed 200/200/200 ep1 | 6 | `dolly_mixed_ep1_0715` | running |
| mixed 200/200/200 ep9 | 7 | `dolly_mixed_ep9_0715` | running |

## 2026-07-12 - Standardized Final-Stage Replay Structure / 标准化最终阶段回放结构

### Decision / 决定

All future 600-budget generative three-stage experiments use the following
default structure unless an explicit ablation is requested:

- Stage 1: pointwise.
- Stage 2: pairwise, with pointwise replay ratio `0`.
- Stage 3: listwise main training plus pointwise replay ratio `1` and pairwise
  replay ratio `1`.

除非明确要求消融实验，今后所有预算为 600 的生成式三阶段实验均采用以下默认结构：

- 阶段 1：pointwise。
- 阶段 2：pairwise，pointwise 回放比例为 `0`。
- 阶段 3：以 listwise 为主任务，同时以比例 `1` 回放 pointwise、以比例 `1` 回放 pairwise。

Canonical arguments / 标准参数：

```bash
--budget-units 600 \
--stage2-pointwise-replay-ratio 0 \
--stage3-pointwise-replay-ratio 1 \
--stage3-pairwise-replay-ratio 1
```

This rule is now recorded in `PROJECT_MEMORY.md` under "Current Three-Stage
Experiment Standard" so it survives context compaction and session handoff.

该规则已写入 `PROJECT_MEMORY.md` 的“Current Three-Stage Experiment Standard”，
确保它在上下文压缩和会话交接后仍然保留。

## 2026-07-12 - Stable Smoothing Sweep on Fixed Pointwise Selector / 固定 Pointwise Selector 的稳定平滑扫描

### Goal / 目标

Test smoothing variants on exactly the same pointwise-only selector triples,
under the standardized final-stage pointwise+pairwise replay structure.

在标准化的最终阶段 pointwise+pairwise 回放结构下，使用完全相同的
pointwise-only selector 三元组测试不同平滑方案。

### Code Changes / 代码变更

- Added alpha scheduling by the number of pointwise replay examples seen,
  independent of listwise/pairwise batch ordering.
- Added optional uniform shrinkage of the fixed Stage1 score prior.
- Added optional prediction-entropy scaling, where the configured alpha is the
  per-sample maximum and confident samples receive less smoothing.

- 根据已处理的 pointwise 回放样本数调度 alpha，不受 listwise/pairwise batch 顺序影响。
- 增加固定阶段 1 分数先验的可选均匀收缩。
- 增加基于预测熵的可选缩放：配置的 alpha 是单样本上限，置信度高的样本使用更弱的平滑。

### Controlled Setup / 受控设置

- Fixed triples:
  `outputs/three_stage_sft_b600_selector_init80_stage3pw1pair1_nosmooth_20260707_152033/selected_triples.jsonl`
- Budget 600; Stage2 pointwise replay 0; Stage3 pointwise/pairwise replay 1/1.
- Smooth only Stage3 pointwise replay.
- Prior initialized from all 600 Stage1 pointwise labels and frozen in Stage3.
- First 200 pointwise replay examples use no smoothing; alpha warms up over the
  next 200 pointwise examples.

- 固定三元组见上述 `selected_triples.jsonl`。
- 预算 600；阶段 2 pointwise 回放为 0；阶段 3 pointwise/pairwise 回放为 1/1。
- 只对阶段 3 的 pointwise 回放进行平滑。
- 先验由阶段 1 的全部 600 个 pointwise 标签初始化，并在阶段 3 冻结。
- 前 200 个 pointwise 回放样本不平滑，随后 200 个样本逐步预热 alpha。

Runs / 运行方案：

- fixed prior, alpha `0.005`
- fixed prior, alpha `0.01`
- fixed prior + 10% uniform, alpha `0.01`
- fixed prior + entropy-adaptive alpha, maximum `0.01`

- 固定先验，alpha `0.005`
- 固定先验，alpha `0.01`
- 固定先验 + 10% 均匀分布，alpha `0.01`
- 固定先验 + 熵自适应 alpha，最大值 `0.01`

Launch script / 启动脚本：
`launch_three_stage_b600_fixed_pointselector_stable_smooth_20260712.sh`

Runtime status: all four runs launched successfully.

运行状态：四项实验均成功启动。

- `fixed_a0005`: tmux `b600_stablesm_a0005_0712`, GPU 0.
- `fixed_a001`: tmux `b600_stablesm_a001_0712`, GPU 1.
- `fixed_u10_a001`: tmux `b600_stablesm_u10_0712`, GPU 2.
- `fixed_entropy_a001`: tmux `b600_stablesm_entropy_0712`, GPU 3.

Expected output prefix / 预期输出前缀：
`outputs/three_stage_sft_b600_fixed_pointselector_stage3_fixedprior_*_stage3pw1pair1_20260712/`

### Final Results / 最终结果（归档于 2026-07-13）

All four runs completed. Final Stage-3 metrics:

四项实验均已完成。阶段 3 最终指标如下：

| Variant / 方案 | Point Acc | Within1 | MAE | Pair Acc | List Acc | List PairRel | List Rank MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed prior, alpha 0.005 | **0.4210** | 0.5955 | 1.6140 | 0.7550 | 0.4890 | 0.6710 | 0.4357 |
| fixed prior, alpha 0.01 | 0.4175 | **0.6015** | 1.6150 | **0.7620** | 0.4995 | **0.6835** | 0.4223 |
| fixed prior + 10% uniform, alpha 0.01 | 0.4185 | 0.5980 | **1.6035** | 0.7543 | **0.5060** | 0.6822 | **0.4217** |
| entropy-adaptive, max alpha 0.01 | 0.4130 | 0.5950 | 1.6260 | 0.7560 | 0.5000 | 0.6780 | 0.4282 |

Interpretation: uniform shrinkage produced the strongest listwise result and
lowest pointwise MAE, while fixed alpha `0.01` produced the strongest pairwise
accuracy. Entropy adaptation did not improve on the simpler fixed/uniform
variants in this run.

解释：均匀收缩取得了最好的 listwise 结果和最低的 pointwise MAE；固定
alpha `0.01` 的 pairwise 准确率最高。本次实验中，熵自适应方案没有超过
更简单的固定或均匀收缩方案。

Canonical result files are the `config.json`, `summary.json`,
`metrics_compact.json`, selection statistics, and `selected_triples.jsonl`
inside each output directory. The multi-GB stage model directories are not
needed to preserve this comparison table.

每个输出目录中的 `config.json`、`summary.json`、`metrics_compact.json`、
选择统计和 `selected_triples.jsonl` 是标准结果文件。保留本对比表并不需要
数 GB 大小的各阶段模型目录。

## 2026-07-11 - Current-Structure Multi-Target Selector Queue / 当前结构的多目标 Selector 队列

### Goal / 目标

Retest the selector that combines pointwise, pairwise, and listwise
uncertainty under the current 600-budget three-stage training structure.

在当前预算 600 的三阶段训练结构下，重新测试结合 pointwise、pairwise 和
listwise 不确定性的 selector。

### Controlled Setup / 受控设置

- Generative SFT: pointwise -> pairwise -> listwise.
- Budget: 600 answer units / 200 selected triples.
- Selector: frozen BERT, init 80 triples, batch 20, four selector epochs.
- Replay: none in stage2; stage3 pointwise ratio 1 and pairwise ratio 1.
- No label smoothing, so this isolates selection quality.
- Compared selector target weights:
  - pointwise/pairwise/listwise = `0.6/0.2/0.2`
  - pointwise/pairwise/listwise = `0.4/0.3/0.3`

- 生成式 SFT：pointwise -> pairwise -> listwise。
- 预算：600 个答案单位 / 200 个选中三元组。
- Selector：冻结 BERT，初始 80 个三元组，batch 20，训练四个 selector epoch。
- 回放：阶段 2 无回放；阶段 3 pointwise 比例 1、pairwise 比例 1。
- 不使用标签平滑，从而单独考察选择质量。
- 对比的 selector 目标权重为 `0.6/0.2/0.2` 和 `0.4/0.3/0.3`。

Launch script / 启动脚本：
`launch_three_stage_b600_multitarget_selector_queue_20260711.sh`

Expected outputs / 预期输出：

- `outputs/three_stage_sft_b600_selector_init80_multitarget_p60_pair20_list20_stage3pw1pair1_nosmooth_20260711/`
- `outputs/three_stage_sft_b600_selector_init80_multitarget_p40_pair30_list30_stage3pw1pair1_nosmooth_20260711/`

The required `--llama-multitask-mode classifier_heads` applies only to the
temporary selector proxy. Final model training remains generative causal-LM
SFT.

必需参数 `--llama-multitask-mode classifier_heads` 只作用于临时 selector
代理；最终模型仍使用生成式 causal-LM SFT 训练。

Runtime status: launched in tmux session `b600_multitarget_sel_0711` on GPU 1.
The `0.6/0.2/0.2` run started successfully; the `0.4/0.3/0.3` run is queued
behind it.

运行状态：已在 GPU 1 的 tmux 会话 `b600_multitarget_sel_0711` 中启动；
`0.6/0.2/0.2` 先运行，`0.4/0.3/0.3` 排队等待。

### Final Results / 最终结果（归档于 2026-07-13）

Both runs completed. Final Stage-3 metrics:

两项实验均已完成。阶段 3 最终指标如下：

| Selector weights / 权重 (point/pair/list) | Point Acc | Within1 | MAE | Pair Acc | List Acc | List PairRel | List Rank MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.6 / 0.2 / 0.2 | **0.4055** | **0.6135** | 1.6445 | **0.7717** | **0.5330** | **0.6968** | **0.3998** |
| 0.4 / 0.3 / 0.3 | 0.3985 | 0.5980 | **1.5970** | 0.7640 | 0.5180 | 0.6928 | 0.4105 |

The `0.6/0.2/0.2` selector is the stronger overall result; `0.4/0.3/0.3`
only leads on pointwise MAE.

`0.6/0.2/0.2` 是综合表现更强的 selector；`0.4/0.3/0.3` 仅在 pointwise
MAE 上领先。

## 2026-07-11 - Frozen-Llama Pointwise Selector Experiments / 冻结 Llama 的 Pointwise Selector 实验

### Goal / 目标

Compare a pointwise-only selector built on frozen Llama features against a
two-level BERT-prefilter plus frozen-Llama reranker.

比较基于冻结 Llama 特征的纯 pointwise selector，与“BERT 预筛选 + 冻结
Llama 重排器”的两级方案。

### Changes / 变更

- Added `shared_llama` and `shared_llama_two_stage` selector kinds to the
  generative three-stage entry point.
- The pointwise uncertainty proxy still learns with LoRA, but selector feature
  extraction disables that adapter. The selector therefore sees stable base
  Llama hidden features throughout active selection.
- A small MLP head predicts pointwise uncertainty from those frozen features.
  Its replay buffer retains all queried triples (maximum 1000).
- In the two-level version, frozen BERT and the Llama MLP are trained from the
  same queried uncertainty targets.

- 在生成式三阶段入口中增加 `shared_llama` 和 `shared_llama_two_stage` selector。
- Pointwise 不确定性代理仍通过 LoRA 学习，但提取 selector 特征时禁用该适配器，
  因此主动选择全程使用稳定的基础 Llama 隐藏特征。
- 使用小型 MLP head 根据冻结特征预测 pointwise 不确定性；回放缓冲区保留全部
  已查询三元组，最多 1000 个。
- 两级版本中，冻结 BERT 和 Llama MLP 使用同一组已查询不确定性目标训练。

### Controlled Setup / 受控设置

- Budget 600, selector init 80, query batch 20, no smoothing.
- Pointwise uncertainty weight 1; pairwise/listwise uncertainty weights 0.
- Stage2 replay 0; stage3 pointwise replay 1 and pairwise replay 1.
- Direct version: random candidate pool 1000 -> Frozen-Llama top 20.
- Two-level version: random pool 4096 -> BERT top 1000 -> Frozen-Llama top 20.

- 预算 600，selector 初始 80 个样本，query batch 20，不使用平滑。
- Pointwise 不确定性权重为 1；pairwise/listwise 不确定性权重为 0。
- 阶段 2 回放为 0；阶段 3 pointwise 和 pairwise 回放均为 1。
- 直接版本：随机候选池 1000 -> Frozen-Llama 选前 20。
- 两级版本：随机候选池 4096 -> BERT 选前 1000 -> Frozen-Llama 选前 20。

Launch script / 启动脚本：
`launch_three_stage_b600_frozen_llama_selector_20260711.sh`

Expected outputs / 预期输出：

- `outputs/three_stage_sft_b600_frozenllama_direct_pool1000_pointunc_init80_stage3pw1pair1_nosmooth_20260711/`
- `outputs/three_stage_sft_b600_bert4096_frozenllama_rerank1000_pointunc_init80_stage3pw1pair1_nosmooth_20260711/`

Runtime status / 运行状态：

- Direct version: running in tmux `b600_frozenllama_direct_0711` on GPU 2.
- Two-level version: running in tmux `b600_frozenllama_2stage_0711` on GPU 3.
- Both processes passed startup and began loading the selector proxy normally.

- 直接版本：在 GPU 2 的 tmux `b600_frozenllama_direct_0711` 中运行。
- 两级版本：在 GPU 3 的 tmux `b600_frozenllama_2stage_0711` 中运行。
- 两个进程均通过启动检查，并正常开始加载 selector 代理。

### Final Results / 最终结果（归档于 2026-07-13）

Both runs completed. Final Stage-3 metrics:

两项实验均已完成。阶段 3 最终指标如下：

| Selector / 选择器 | Point Acc | Within1 | MAE | Pair Acc | List Acc | List PairRel | List Rank MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen-Llama direct | **0.4030** | **0.6180** | **1.6150** | **0.7713** | **0.5205** | **0.6885** | **0.4110** |
| BERT prefilter + Frozen-Llama rerank | 0.4005 | 0.6160 | 1.6525 | 0.7473 | 0.4875 | 0.6743 | 0.4360 |

The direct Frozen-Llama selector won every listed final metric, so the BERT
prefilter/reranker did not justify its extra complexity in this comparison.

直接 Frozen-Llama selector 在列出的所有最终指标上均胜出，因此在本次对比中，
BERT 预筛选/重排方案增加的复杂度没有带来收益。

## 2026-07-11 - Fixed-Selector Stage1+Stage3 Smooth Sweep / 固定 Selector 的阶段 1+3 平滑扫描

### Goal / 目标

Measure the real effect of smoothing alpha in the 600-budget generative
three-stage experiment while holding selector output fixed.

固定 selector 输出，测量预算 600 的生成式三阶段实验中平滑 alpha 的真实影响。

### Controlled Setup / 受控设置

- Pipeline: pointwise -> pairwise -> listwise generative SFT.
- Budget: 600 pointwise answer units = 200 selected triples.
- Fixed triples:
  `outputs/three_stage_sft_b600_selector_init80_stage3pw1pair1_nosmooth_20260707_152033/selected_triples.jsonl`
- Training examples:
  - stage1: 600 pointwise
  - stage2: 1200 augmented pairwise
  - stage3: 1200 augmented listwise + 1200 pointwise replay + 1200 pairwise replay
- Replay ratios: stage2 pointwise `0`, stage3 pointwise `1`, stage3 pairwise `1`.
- Smooth stages: `stage1,stage3`.
- Smooth schedule: start step `20`, warmup `20`.
- Seed: `42`.
- SFT: Llama-3-8B-Instruct, LoRA, 4-bit, learning rate `1e-4`.

- 流水线：pointwise -> pairwise -> listwise 生成式 SFT。
- 预算：600 个 pointwise 答案单位，即 200 个选中三元组。
- 固定三元组见上述 `selected_triples.jsonl`。
- 训练样本：阶段 1 为 600 个 pointwise；阶段 2 为 1200 个增强 pairwise；
  阶段 3 为 1200 个增强 listwise + 1200 个 pointwise 回放 + 1200 个 pairwise 回放。
- 回放比例：阶段 2 pointwise 为 `0`；阶段 3 pointwise、pairwise 均为 `1`。
- 平滑阶段：`stage1,stage3`；从 step `20` 开始，预热 `20` step。
- 随机种子 `42`；SFT 使用 Llama-3-8B-Instruct、LoRA、4-bit，学习率 `1e-4`。

Launch script / 启动脚本：
`launch_three_stage_b600_fixedselector_stage13_smooth_sweep_20260710_114811.sh`

Log directory / 日志目录：
`outputs/three_stage_sft_b600_fixedselector_stage13_smooth_sweep_logs_20260710_114811/`

All five sweep runs completed successfully.

五项扫描实验均成功完成。

### Final Metrics / 最终指标

| Alpha | Point Acc | Within1 | MAE | Pair Acc | List Acc | List PairRel | List Rank MAE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.4175 | 0.5950 | 1.6195 | 0.7533 | 0.4900 | 0.6715 | 0.4340 |
| 0.01 | **0.4250** | **0.6030** | **1.5995** | **0.7588** | 0.4975 | 0.6780 | 0.4258 |
| 0.02 | 0.4180 | 0.5980 | 1.6220 | 0.7557 | 0.5000 | 0.6788 | 0.4258 |
| 0.03 | 0.4225 | 0.5960 | 1.6205 | 0.7528 | 0.4995 | 0.6753 | 0.4297 |
| 0.04 | 0.4170 | 0.5940 | 1.6370 | 0.7538 | **0.5025** | **0.6795** | **0.4252** |
| 0.05 | 0.4195 | 0.5975 | 1.6190 | 0.7537 | 0.4920 | 0.6747 | 0.4297 |
| 0.08 | 0.4220 | 0.5915 | 1.6310 | 0.7545 | 0.4935 | 0.6733 | 0.4305 |

The alpha `0.00` and `0.03` rows come from the earlier fixed-selector control
runs. The remaining rows come from the current sweep.

alpha `0.00` 和 `0.03` 两行来自较早的固定 selector 对照实验，其余行来自
本次扫描。

### Initial Interpretation / 初步解释

- Alpha `0.01` is the best pointwise-balanced result and also has the highest
  pairwise accuracy.
- Alpha `0.04` is the strongest listwise-oriented setting.
- Larger alpha values do not improve consistently; the useful range appears
  to be small, around `0.01-0.04`.
- Because all rows reuse the same selected triples, this comparison removes
  the earlier selector-data confound. Remaining small differences can still
  include GPU training nondeterminism.

- Alpha `0.01` 是 pointwise 综合表现最好的方案，同时 pairwise 准确率最高。
- Alpha `0.04` 是最偏向 listwise 排序表现的方案。
- 更大的 alpha 没有稳定提升；有效范围似乎较小，约为 `0.01-0.04`。
- 所有行复用相同的选中三元组，因此排除了先前 selector 数据差异造成的混杂；
  剩余的小幅差异仍可能包含 GPU 训练的非确定性。

Compared with no smoothing, alpha `0.01` changes the final metrics by:
Point Acc `+0.0075`, Within1 `+0.0080`, MAE `-0.0200`, Pair Acc `+0.0055`,
List Acc `+0.0075`, List PairRel `+0.0065`, and List Rank MAE `-0.0082`.
It is the only tested alpha that improves every primary final metric.

相对无平滑，alpha `0.01` 的最终指标变化为：Point Acc `+0.0075`、
Within1 `+0.0080`、MAE `-0.0200`、Pair Acc `+0.0055`、List Acc
`+0.0075`、List PairRel `+0.0065`、List Rank MAE `-0.0082`。它是唯一
在所有主要最终指标上都有改善的已测试 alpha。

Alpha `0.04` trades pointwise quality for ranking quality: versus no
smoothing, List Acc improves by `+0.0125`, PairRel by `+0.0080`, and Rank MAE
by `-0.0088`, while Point Acc is essentially unchanged (`-0.0005`) and
pointwise MAE worsens by `+0.0175`.

Alpha `0.04` 用 pointwise 质量换取排序质量：与无平滑相比，List Acc 提高
`+0.0125`、PairRel 提高 `+0.0080`、Rank MAE 降低 `-0.0088`；Point Acc
基本不变（`-0.0005`），但 pointwise MAE 变差 `+0.0175`。

The response is non-monotonic, so more smoothing is not generally better.
With one training seed, differences of only a few tenths of a percentage
point should not be treated as conclusive. Alpha `0.01` is the strongest
candidate for pointwise-focused follow-up; alpha `0.04` is the ranking-focused
candidate.

响应并非单调，因此更强的平滑通常不等于更好的结果。当前只有一个训练种子，
零点几个百分点的差异不应视为定论。Alpha `0.01` 是偏 pointwise 后续实验的
最佳候选，alpha `0.04` 是偏排序后续实验的候选。

### Next / 后续

- Decide whether the primary objective is pointwise (`alpha=0.01`) or
  listwise ranking (`alpha=0.04`).
- For a publishable conclusion, repeat no-smooth, `0.01`, and `0.04` with
  additional training seeds while keeping selected triples fixed.

- 决定主要目标是 pointwise（`alpha=0.01`）还是 listwise 排序（`alpha=0.04`）。
- 若要形成可发表的结论，应在固定选中三元组的前提下，用更多训练种子重复
  无平滑、`0.01` 和 `0.04`。

## 2026-07-11 - Random Alpha 0.01 Control / 随机选样的 Alpha 0.01 对照实验

### Goal / 目标

Test whether the Stage1+Stage3 alpha `0.01` improvement also appears with
randomly selected training triples.

测试阶段 1+3 使用 alpha `0.01` 的改善是否也会出现在随机选择的训练三元组上。

### Controlled Setup / 受控设置

The run reuses the exact random triples from:
`outputs/three_stage_sft_b600_random_stage3pw1pair1_nosmooth_20260707_152033/selected_triples.jsonl`

This makes the new run a direct comparison against the existing random
no-smooth baseline. It uses the same 600 budget, stage3 pointwise and pairwise
replay ratios of `1`, and smooth schedule used in the fixed-selector sweep.

该实验复用上述路径中的完全相同随机三元组，因此可与已有随机无平滑基线直接
对比。预算同为 600，阶段 3 pointwise 和 pairwise 回放比例均为 `1`，平滑
调度与固定 selector 扫描相同。

Launch script / 启动脚本：
`launch_three_stage_b600_fixedrandom_stage13_smooth_a001_20260711_020745.sh`

Expected output / 预期输出：
`outputs/three_stage_sft_b600_fixedrandom_smooth_a001_stage13_stage3pw1pair1_20260711_020745/`

Runtime status: running in tmux session `b600_random_a001_0711` on GPU 0.

运行状态：在 GPU 0 的 tmux 会话 `b600_random_a001_0711` 中运行。

The first launch exposed that fixed random rows can store a different A/B/C
order from the candidate-pool reconstruction. The fixed-triple loader was
updated to resolve an unordered model/score signature and then restore the
exact saved A/B/C order. The failed empty output was removed before restarting
the experiment.

首次启动发现，固定随机记录中的 A/B/C 顺序可能与候选池重建时不同。固定三元组
加载器已改为先按无序的模型/分数签名定位记录，再恢复保存时的准确 A/B/C 顺序。
重启实验前已删除失败产生的空输出目录。

### Final Result / 最终结果（归档于 2026-07-13）

The restarted run completed. Final Stage-3 metrics were: Point Acc `0.3840`,
Within1 `0.6020`, MAE `1.6615`, Pair Acc `0.7603`, List Acc `0.5160`, List
PairRel `0.6857`, and List Rank MAE `0.4125`.

重启后的实验已完成。阶段 3 最终指标为：Point Acc `0.3840`、Within1
`0.6020`、MAE `1.6615`、Pair Acc `0.7603`、List Acc `0.5160`、List
PairRel `0.6857`、List Rank MAE `0.4125`。
## 2026-08-01 - Stage4 continual consolidation runs

Implemented a narrow Stage4 extension in
`run_pointwise5answers_three_stage_pairwise_listwise_sft_v1.py`: Stage4
consolidation can now receive Stage1 pointwise teacher logits on pointwise
replay samples, so `--pointwise-teacher-distill-weight` applies to
`stage4_consolidation` as well as Stage23. Pairwise/listwise replay still use
ordinary SFT CE; this run is therefore explicitly pointwise-teacher distill,
not full three-task teacher distillation.

Launched four fixed-selection Instruct experiments in tmux session
`l1b_stage4_consolidation_0801` using fixed pointproxy triples from
`outputs/llama3p2_1b_alpaca_b600_pointproxy_entropy100_init80_fullpool_noexplore_nosmooth_20260726/selected_triples.jsonl`.
All runs use `Llama-3.2-1B-Instruct`, no smooth, Stage1 pointwise only,
Stage2 pairwise only, Stage3 listwise only, then Stage4 stratified replay:

- `llama3p2_1b_instruct_alpaca_b600_fixedpointproxy_seq_ppl_stage4_stratfull_nodistill_nosmooth_20260801`
  on GPU 0: Stage4 replay fraction 1.0, no distill.
- `llama3p2_1b_instruct_alpaca_b600_fixedpointproxy_seq_ppl_stage4_stratfull_ptdistillw01_t2_nosmooth_20260801`
  on GPU 1: Stage4 replay fraction 1.0, pointwise teacher distill weight 0.1,
  temperature 2.
- `llama3p2_1b_instruct_alpaca_b600_fixedpointproxy_seq_ppl_stage4_strathalf_nodistill_nosmooth_20260801`
  on GPU 2: Stage4 replay fraction 0.5, no distill.
- `llama3p2_1b_instruct_alpaca_b600_fixedpointproxy_seq_ppl_stage4_stratquarter_nodistill_nosmooth_20260801`
  on GPU 3: Stage4 replay fraction 0.25, no distill.

Launcher script:
`launch_llama3p2_1b_instruct_stage4_consolidation_20260801.sh`.

Note: plain `nohup` background processes are cleaned up in this execution
environment, so the stable launch is via tmux, not detached `nohup`.

## 2026-08-01 - Stage4 half replay algorithm ablations

Launched the first two continual-learning ablations requested after the Stage4
half/full/quarter replay discussion. Both runs use the same fixed pointproxy
selection as the Stage4 consolidation runs, `Llama-3.2-1B-Instruct`, no smooth,
Stage1 pointwise only, Stage2 pairwise only, Stage3 listwise only, then Stage4
with replay fraction `0.5`. Half replay keeps the training count fixed at 100
selected triples: pointwise 300, pairwise 600, listwise 600.

Launcher script:
`launch_llama3p2_1b_instruct_stage4_half_algos_20260801.sh`

tmux session:
`l1b_stage4_half_algos_0801`

- `llama3p2_1b_instruct_alpaca_b600_fixedpointproxy_seq_ppl_stage4_strathalf_pertaskder_w01_t2_nosmooth_20260801`
  on GPU 0: stratified half replay plus pointwise teacher distill weight `0.1`
  and pairwise/listwise decision-token teacher distill weight `0.1`, temperature
  `2`. The listwise teacher is a lightweight top-answer decision teacher, not a
  full ranking-sequence KL.
- `llama3p2_1b_instruct_alpaca_b600_fixedpointproxy_seq_ppl_stage4_losshalf_nodistill_nosmooth_20260801`
  on GPU 1: forgetting-aware half replay. After Stage3, select the highest-loss
  100 triples under the current model and replay their pointwise/pairwise/listwise
  samples. No distillation.

## 2026-08-01 - LM-head selector Stage4 half replay ablations

Clarified that the Stage4 half replay algorithm ablations above use fixed
triples from the older `pointwise_proxy` selector output:
`outputs/llama3p2_1b_alpaca_b600_pointproxy_entropy100_init80_fullpool_noexplore_nosmooth_20260726/selected_triples.jsonl`.
That selection contains 80 `random_init` triples and 120 `pointwise_proxy`
acquisitions. Its config predates explicit `candidate_selector_proxy_mode`, so
it should be treated as the legacy classifier-head pointwise proxy selection.

Launched the same two Stage4 half replay mechanisms with a fresh LM-head
pointwise proxy selector. These runs do not use
`--reuse-selection-proxy-for-stage1`; the LM-head proxy is used only for active
selection, then Stage1 pointwise SFT is trained normally. This isolates the
selector choice from the proxy-reuse mechanism.

Launcher script:
`launch_llama3p2_1b_instruct_lmheadselector_stage4_half_algos_20260801.sh`

tmux session:
`l1b_stage4_lmheadselector_half_algos_0801`

- `llama3p2_1b_instruct_alpaca_b600_lmheadselector_seq_ppl_stage4_strathalf_pertaskder_w01_t2_nosmooth_20260801`
  on GPU 2: fresh `pointwise_proxy` selection with
  `candidate_selector_proxy_mode=lm_head`, entropy-only acquisition, no
  exploration, then stratified half replay plus per-task DER.
- `llama3p2_1b_instruct_alpaca_b600_lmheadselector_seq_ppl_stage4_losshalf_nodistill_nosmooth_20260801`
  on GPU 3: same fresh LM-head selector setup, then loss-based hard-triple half
  replay with no distillation.

## 2026-08-01 - LM-head proxy reuse Stage4 half replay ablations

Launched the reuse version requested after comparing the fresh LM-head selector
results. These runs use a fresh `pointwise_proxy` selector with
`candidate_selector_proxy_mode=lm_head` and `--reuse-selection-proxy-for-stage1`,
so the LM-head selection proxy is kept as Stage1 instead of retraining
pointwise SFT from scratch. Stage2 pointwise replay and Stage3 pointwise/pairwise
replay remain disabled; Stage4 uses half replay.

Launcher script:
`launch_llama3p2_1b_instruct_lmheadreuse_stage4_half_algos_20260801.sh`

tmux session:
`l1b_stage4_lmheadreuse_half_algos_0801`

- `llama3p2_1b_instruct_alpaca_b600_lmheadproxy_reuse_seq_ppl_stage4_strathalf_pertaskder_w01_t2_nosmooth_20260801`
  on GPU 0: reuse LM-head proxy as Stage1, then stratified half replay plus
  pointwise and pair/list decision-token DER.
- `llama3p2_1b_instruct_alpaca_b600_lmheadproxy_reuse_seq_ppl_stage4_losshalf_nodistill_nosmooth_20260801`
  on GPU 1: reuse LM-head proxy as Stage1, then loss-based hard-triple half
  replay with no distillation.

## 2026-08-02 - LM-head proxy reuse plain Stage4 half/full

Launched the plain replay controls requested after the reuse Stage4 results.
These runs keep the fresh LM-head `pointwise_proxy` selector and
`--reuse-selection-proxy-for-stage1`, but remove the extra continual-learning
algorithms: no DER, no hard-loss replay. Stage2 pointwise replay and Stage3
pointwise/pairwise replay remain disabled, and Stage4 uses ordinary stratified
triple replay.

Launcher script:
`launch_llama3p2_1b_instruct_lmheadreuse_stage4_plain_20260802.sh`

tmux session:
`l1b_stage4_lmheadreuse_plain_0802`

- `llama3p2_1b_instruct_alpaca_b600_lmheadproxy_reuse_seq_ppl_stage4_strathalf_nodistill_nosmooth_20260802`
  on GPU 0: reuse LM-head proxy as Stage1, ordinary stratified Stage4 half replay.
- `llama3p2_1b_instruct_alpaca_b600_lmheadproxy_reuse_seq_ppl_stage4_stratfull_nodistill_nosmooth_20260802`
  on GPU 1: reuse LM-head proxy as Stage1, ordinary stratified Stage4 full replay.

## 2026-08-02 - No-selector random three-stage replay baseline

Launched a clean no-selector baseline requested as "pure three stages + replay".
This run does not use candidate selectors, pointproxy, or LM-head reuse. It uses
random triple selection through `train_selection_mode=selected_triple` and
`triple_selection_strategy=random`, then trains:
Stage1 pointwise 600, Stage2 pairwise 1200, Stage3 listwise 1200 plus pointwise
replay ratio `1` and pairwise replay ratio `1`. No smoothing.

Launcher script:
`launch_llama3p2_1b_instruct_random_three_stage_replay_20260802.sh`

tmux session:
`l1b_instruct_random_three_stage_replay_0802`

Output:
`outputs/llama3p2_1b_instruct_alpaca_b600_random_three_stage_stage3pw1pair1_nosmooth_20260802`

Final metrics for the clean no-selector baseline:
pointwise acc `0.2840`, within1 `0.4705`, MAE `2.6200`, pairwise acc
`0.6858`, listwise acc `0.3560`.

Final metrics for the LM-head reuse plain Stage4 controls:

- Half replay:
  pointwise acc `0.2765`, within1 `0.4620`, MAE `2.7335`, pairwise acc
  `0.7287`, listwise acc `0.3375`.
- Full replay:
  pointwise acc `0.2940`, within1 `0.4905`, MAE `2.5785`, pairwise acc
  `0.7173`, listwise acc `0.3325`.

Interpretation note: plain stratified replay is healthier for listwise than the
DER/hard-loss variants, but LM-head reuse still mainly helps pointwise and
pairwise. The clean random no-selector three-stage replay baseline matches the
fixed selector standard three-stage listwise result while improving pointwise
MAE, so selector improvements should be treated cautiously until replicated
against this no-selector baseline.

## 2026-08-03 - Requested Stage4 suite: no-selector random and LM-head selector full replay

Launched the requested four-stage suite in tmux session
`l1b_stage4_requested_suite_0803`.

Launcher script:
`launch_llama3p2_1b_instruct_stage4_requested_suite_20260803.sh`

To control selection noise, the no-selector random variants reuse the random
selected triples from:
`outputs/llama3p2_1b_instruct_alpaca_b600_random_three_stage_stage3pw1pair1_nosmooth_20260802/selected_triples.jsonl`.
This is still a no-selector condition; it fixes one random draw so Stage4
mechanisms are comparable.

The LM-head selector full-replay variants reuse the fresh LM-head selector
selected triples from:
`outputs/llama3p2_1b_instruct_alpaca_b600_lmheadselector_seq_ppl_stage4_strathalf_pertaskder_w01_t2_nosmooth_20260801/selected_triples.jsonl`.

First batch launched on GPUs 0-5:

- `llama3p2_1b_instruct_alpaca_b600_randomfixed_seq_ppl_stage4_strathalf_nodistill_nosmooth_20260803`
- `llama3p2_1b_instruct_alpaca_b600_randomfixed_seq_ppl_stage4_stratfull_nodistill_nosmooth_20260803`
- `llama3p2_1b_instruct_alpaca_b600_randomfixed_seq_ppl_stage4_strathalf_pertaskder_w01_t2_nosmooth_20260803`
- `llama3p2_1b_instruct_alpaca_b600_randomfixed_seq_ppl_stage4_stratfull_pertaskder_w01_t2_nosmooth_20260803`
- `llama3p2_1b_instruct_alpaca_b600_randomfixed_seq_ppl_stage4_losshalf_nodistill_nosmooth_20260803`
- `llama3p2_1b_instruct_alpaca_b600_fixedlmheadselector_seq_ppl_stage4_stratfull_nodistill_nosmooth_20260803`

Second batch is queued in the same script after first-batch success:

- `llama3p2_1b_instruct_alpaca_b600_fixedlmheadselector_seq_ppl_stage4_stratfull_pertaskder_w01_t2_nosmooth_20260803`
- `llama3p2_1b_instruct_alpaca_b600_fixedlmheadselector_seq_ppl_stage4_lossfull_nodistill_nosmooth_20260803`

Note: `lossfull` replays all triples after loss scoring, so it mostly tests
whether loss-scored ordering/full replay differs from ordinary full replay.
Hard-loss selection is only non-degenerate when the replay fraction is below 1.

Follow-up: the first launcher evaluated after every stage and was too slow for
the requested "final Stage4 only" comparison. Added `--eval-stages final` to
`run_pointwise5answers_three_stage_pairwise_listwise_sft_v1.py` so training is
unchanged but intermediate Stage1/2/3 evaluations are skipped.

Final-only launcher:
`launch_llama3p2_1b_instruct_stage4_requested_suite_finaleval_20260803.sh`

tmux session:
`l1b_stage4_requested_finaleval_0803`

The final-only suite also includes no-selector hard-loss full replay:
`llama3p2_1b_instruct_alpaca_b600_randomfixed_seq_ppl_stage4_lossfull_nodistill_nosmooth_finaleval_20260803`

Final `after_stage4` metrics:

| Run | Point Acc | Within1 | MAE | Pairwise Acc | Listwise Acc |
|---|---:|---:|---:|---:|---:|
| No selector half | 0.2435 | 0.4420 | 2.4505 | 0.6812 | 0.3845 |
| No selector full | 0.2790 | 0.4855 | 2.4175 | 0.6933 | 0.3830 |
| No selector half + DER | 0.2535 | 0.4480 | 2.5205 | 0.6743 | 0.3880 |
| No selector full + DER | 0.2795 | 0.4980 | 2.4440 | 0.6677 | 0.3725 |
| No selector hard-loss half | 0.2315 | 0.4195 | 2.9900 | 0.6113 | 0.2925 |
| No selector hard-loss full | 0.2800 | 0.4850 | 2.4025 | 0.6922 | 0.3810 |
| LM-head selector full | 0.2250 | 0.4715 | 2.7630 | 0.6215 | 0.3460 |
| LM-head selector full + DER | 0.2095 | 0.4790 | 2.8390 | 0.6052 | 0.3345 |
| LM-head selector hard-loss full | 0.2255 | 0.4675 | 2.7680 | 0.6240 | 0.3450 |

Interpretation: no-selector fixed-random Stage4 is strongest here, especially
plain full / half+DER for listwise. DER does not help full replay and hurts the
LM-head selector condition. Hard-loss half is harmful; hard-loss full is mostly
a full replay order-control and stays close to ordinary full replay.

## 2026-08-03 - Random Stage4 local-Gaussian smoothing grid

Launched a fixed-random Stage4 full-replay smoothing grid using the same random
selected triples as the no-selector Stage4 suite:

`outputs/llama3p2_1b_instruct_alpaca_b600_random_three_stage_stage3pw1pair1_nosmooth_20260802/selected_triples.jsonl`

The grid tests local Gaussian score-token smoothing in Stage4 only, with
`--stage4-replay-strategy random_triple`, `--stage4-replay-fraction 1.0`,
`--pointwise-global-smooth-mode local_gaussian`, sigma `1.0`, no distillation,
and final-only evaluation.

Grid:

- alpha `0.01`, start-pointwise-seen `0`
- alpha `0.01`, start-pointwise-seen `200`
- alpha `0.03`, start-pointwise-seen `0`
- alpha `0.03`, start-pointwise-seen `200`
- alpha `0.05`, start-pointwise-seen `0`
- alpha `0.05`, start-pointwise-seen `200`

Launcher:
`schedule_llama3p2_1b_instruct_random_stage4_localgauss_smooth_grid_20260803.sh`

Logs:
`outputs/llama3p2_1b_instruct_alpaca_random_stage4_localgauss_smooth_grid_logs_20260803/`

Tmux:

- Main grid session: `l1b_random_stage4_localgauss_smooth_grid_0803`
- Tail queue for alpha `0.05`: `l1b_random_stage4_localgauss_smooth_a005_tail_0803`

Runtime note: `nvidia-smi` is unavailable on this host due an NVML
driver/library mismatch, so the launcher uses `fuser -v /dev/nvidia*` memory
mapping checks. An initial pipefail interaction made two alpha `0.05` jobs
briefly launch on busy GPUs 5/6, then they were stopped and their incomplete
outputs were preserved with `.aborted_gpu_collision*` suffixes. The scheduler
now captures `fuser` output before grepping, defaults to GPUs `0 1 2 3`, and
the alpha `0.05` tail queue waits for one of those GPUs to become free.

Follow-up: all six final-only runs completed successfully. Final
`after_stage4` metrics:

| Run | Alpha | Start seen | Point Acc | Within1 | MAE | Pairwise Acc | Listwise Acc | List PairRel | Rank MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| No-smooth stratified full baseline | - | - | 0.2790 | 0.4855 | 2.4175 | 0.6933 | 0.3830 | 0.6048 | 0.5303 |
| Local Gaussian | 0.01 | 0 | 0.2800 | 0.4850 | 2.4160 | 0.6922 | 0.3785 | 0.6025 | 0.5327 |
| Local Gaussian | 0.01 | 200 | 0.2805 | 0.4845 | 2.4075 | 0.6950 | 0.3830 | 0.6062 | 0.5303 |
| Local Gaussian | 0.03 | 0 | 0.2815 | 0.4860 | 2.4050 | 0.6948 | 0.3865 | 0.6067 | 0.5285 |
| Local Gaussian | 0.03 | 200 | 0.2800 | 0.4855 | 2.4145 | 0.6947 | 0.3830 | 0.6060 | 0.5315 |
| Local Gaussian | 0.05 | 0 | 0.2795 | 0.4880 | 2.4060 | 0.6958 | 0.3860 | 0.6080 | 0.5288 |
| Local Gaussian | 0.05 | 200 | 0.2770 | 0.4840 | 2.4320 | 0.6952 | 0.3775 | 0.6025 | 0.5330 |

Best local-Gaussian pointwise accuracy and MAE are at alpha `0.03`,
start-pointwise-seen `0`: Point Acc `0.2815`, MAE `2.4050`. Best pairwise
accuracy and listwise relation accuracy are at alpha `0.05`,
start-pointwise-seen `0`: Pairwise Acc `0.6958`, PairRel `0.6080`.

Interpretation: local Gaussian smoothing is not a large win on this fixed
random Stage4 full-replay setup, but alpha `0.03`/`0.05` with start seen `0`
is slightly healthier than no smoothing across most final metrics. Delaying
smoothing to start seen `200` does not help here and alpha `0.05` with start
seen `200` is clearly worse, especially for pointwise MAE and listwise acc.
The comparison is slightly imperfect because the old no-smooth baseline used
`stratified_triple` full replay while this grid used `random_triple` full
replay; with replay fraction `1.0` both see all triples, so this is mainly an
ordering difference. The no-smooth hard-loss full ordering control was close
to the no-smooth stratified full baseline, which suggests the small gains here
are plausibly from smoothing rather than only replay ordering.

## 2026-08-04 - Random Stage4 local-Gaussian larger-alpha follow-up

Follow-up decision: use `--pointwise-global-smooth-start-pointwise-seen 0` as
the default for subsequent local-Gaussian smoothing sweeps. The previous grid
showed no benefit from delaying smoothing to start seen `200`.

Launched a stricter Stage4 random/full no-smooth baseline plus larger-alpha
local-Gaussian sweep in tmux session
`l1b_random_stage4_localgauss_largeralpha_0804`.

Launcher:
`schedule_llama3p2_1b_instruct_random_stage4_localgauss_largeralpha_20260804.sh`

Logs:
`outputs/llama3p2_1b_instruct_alpaca_random_stage4_localgauss_largeralpha_logs_20260804/`

All runs use the same fixed random triples as the prior Stage4 smoothing grid:
`outputs/llama3p2_1b_instruct_alpaca_b600_random_three_stage_stage3pw1pair1_nosmooth_20260802/selected_triples.jsonl`

Common setup: Stage4 only, `--stage4-replay-strategy random_triple`,
`--stage4-replay-fraction 1.0`, no distillation, final-only evaluation,
local-Gaussian sigma `1.0`, start seen `0`, warmup seen `200`.

Grid:

- strict no-smooth random/full baseline
- alpha `0.07`, start seen `0`
- alpha `0.10`, start seen `0`
- alpha `0.15`, start seen `0`

Follow-up: all four runs completed successfully. Final `after_stage4` metrics:

| Run | Alpha | Point Acc | Within1 | MAE | Pairwise Acc | Listwise Acc | List PairRel | Rank MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Strict random/full no-smooth | - | 0.2825 | 0.4875 | 2.4095 | 0.6928 | 0.3810 | 0.6047 | 0.5307 |
| Local Gaussian | 0.07 | 0.2860 | 0.4945 | 2.4060 | 0.6952 | 0.3870 | 0.6078 | 0.5280 |
| Local Gaussian | 0.10 | 0.2835 | 0.4900 | 2.4180 | 0.6952 | 0.3840 | 0.6058 | 0.5305 |
| Local Gaussian | 0.15 | 0.2815 | 0.4880 | 2.4315 | 0.6935 | 0.3860 | 0.6078 | 0.5270 |

Interpretation: alpha `0.07` is the cleanest result so far in this
random/full setting: vs the strict random/full no-smooth baseline it improves
Point Acc by `+0.0035`, Within1 by `+0.0070`, MAE by `-0.0035`, Pairwise Acc
by `+0.0023`, Listwise Acc by `+0.0060`, PairRel by `+0.0032`, and Rank MAE
by `-0.0027`. Pushing to alpha `0.10` weakens pointwise MAE and listwise, and
alpha `0.15` starts to hurt pointwise accuracy/MAE even though listwise rank
MAE remains good.

## 2026-08-04 - Random Stage4 local-Gaussian smoothing in Stage1 and Stage4

Launched a focused comparison to test whether applying local-Gaussian
score-token smoothing earlier helps. This keeps the same fixed-random
Stage4 random/full setup and changes only
`--pointwise-global-smooth-stages` from `stage4` to `stage1,stage4`.

Launcher:
`schedule_llama3p2_1b_instruct_random_stage14_localgauss_20260804.sh`

Logs:
`outputs/llama3p2_1b_instruct_alpaca_random_stage14_localgauss_logs_20260804/`

tmux session:
`l1b_random_stage14_localgauss_0804`

Common setup: same fixed random triples, Stage4 only, `random_triple` full
replay, no distillation, final-only eval, local-Gaussian sigma `1.0`, start
seen `0`, warmup seen `200`.

Grid:

- alpha `0.07`, stages `stage1,stage4`
- alpha `0.10`, stages `stage1,stage4`

Startup check: both logs confirmed smoothing is enabled for
`stage1_pointwise` with `stages=stage1,stage4`.

Follow-up: both runs completed successfully. Final `after_stage4` metrics:

| Run | Alpha | Smooth stages | Point Acc | Within1 | MAE | Pairwise Acc | Listwise Acc | List PairRel | Rank MAE |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Strict random/full no-smooth | - | - | 0.2825 | 0.4875 | 2.4095 | 0.6928 | 0.3810 | 0.6047 | 0.5307 |
| Local Gaussian | 0.07 | stage4 | 0.2860 | 0.4945 | 2.4060 | 0.6952 | 0.3870 | 0.6078 | 0.5280 |
| Local Gaussian | 0.07 | stage1,stage4 | 0.2780 | 0.4900 | 2.4220 | 0.6947 | 0.3895 | 0.6113 | 0.5255 |
| Local Gaussian | 0.10 | stage4 | 0.2835 | 0.4900 | 2.4180 | 0.6952 | 0.3840 | 0.6058 | 0.5305 |
| Local Gaussian | 0.10 | stage1,stage4 | 0.2800 | 0.4890 | 2.4295 | 0.6948 | 0.3840 | 0.6072 | 0.5300 |

Interpretation: applying local-Gaussian smoothing in Stage1 as well as Stage4
helps listwise/ranking but hurts pointwise calibration. At alpha `0.07`,
`stage1,stage4` beats `stage4` on List Acc (`+0.0025`), PairRel (`+0.0035`),
and Rank MAE (`-0.0025`), but loses substantial Point Acc (`-0.0080`) and MAE
(`+0.0160`). At alpha `0.10`, adding Stage1 gives no List Acc gain and still
hurts pointwise. Current recommendation: keep the main/default smoothing stage
as `stage4`; consider `stage1,stage4` only if the objective prioritizes
listwise relation/ranking over exact pointwise score accuracy.

## 2026-08-04 - Ep10 one-answer and mix controls for Stage4 full replay comparisons

For Stage4 full-replay comparisons, use ep10 controls rather than the earlier
ep6 controls: the four-stage full-replay setup has 6000 total training
exposures (`P600 + Pair1200 + List1200 + replay P600/Pair1200/List1200`),
while both controls have 600 examples per epoch.

Launched tmux session `l1b_ep10_stage4full_controls_0804`:

- one-answer control:
  `outputs/llama3p2_1b_instruct_alpaca_one_answer_random600_ep10_stage4fullmatch_20260804`
- trueval mix control:
  `outputs/llama3p2_1b_instruct_alpaca_trueval_mix_200pw_200pair_200list_ep10_stage4fullmatch_20260804`

Launcher:
`launch_llama3p2_1b_instruct_ep10_stage4full_controls_20260804.sh`

Logs:
`outputs/llama3p2_1b_instruct_alpaca_ep10_stage4full_controls_logs_20260804/`

Compatibility note: `run_newnew_one_answer_trueval_three_stage_sft.py` was
updated to pass default `resume_stage1_model_dir=""` and `eval_stages="all"`
to the shared `RunConfig`, preserving the old control-evaluation behavior.

## 2026-08-04 - LM-head reuse full + DER noexpand resume

Retried `LM-head reuse full + DER` from the saved Stage1 adapter and fixed
selected triples, using a fresh output directory and explicitly unsetting
`PYTORCH_CUDA_ALLOC_CONF`. The earlier original and resume attempts both
failed during Stage1 teacher-logit caching with a PyTorch/NVML allocator
assert while `expandable_segments:True` was set.

Launcher:
`launch_llama3p2_1b_instruct_lmheadreuse_stage4_full_der_resume_noexpand_20260804.sh`

Tmux:
`l1b_lmheadreuse_full_der_noexpand_0804`

Output:
`outputs/llama3p2_1b_instruct_alpaca_b600_lmheadproxy_reuse_seq_ppl_stage4_stratfull_pertaskder_w01_t2_nosmooth_resume_noexpand_20260804`

Log:
`outputs/llama3p2_1b_instruct_alpaca_lmheadreuse_stage4_full_der_resume_noexpand_logs_20260804/llama3p2_1b_instruct_alpaca_b600_lmheadproxy_reuse_seq_ppl_stage4_stratfull_pertaskder_w01_t2_nosmooth_resume_noexpand_20260804.log`

Initial check: the run passed the previous failure point and entered
`Training stage2_pairwise`, so removing `expandable_segments:True` appears to
avoid the immediate NVML allocator crash.

## 2026-08-04 - Random Stage4 local-Gaussian sigma sweep

Launched a focused sigma sweep around the current best comprehensive setting:
alpha `0.07`, smoothing stages `stage4`, start seen `0`, warmup seen `200`.
The existing alpha `0.07`, sigma `1.0`, stage4-only run is reused as the
middle point; this sweep adds narrower and wider local-Gaussian distributions.

Launcher:
`schedule_llama3p2_1b_instruct_random_stage4_localgauss_sigma_20260804.sh`

Logs:
`outputs/llama3p2_1b_instruct_alpaca_random_stage4_localgauss_sigma_logs_20260804/`

tmux session:
`l1b_random_stage4_localgauss_sigma_0804`

Common setup: same fixed random triples, Stage4 only, `random_triple` full
replay, no distillation, final-only eval, local-Gaussian alpha `0.07`, stages
`stage4`, start seen `0`, warmup seen `200`.

Grid:

- sigma `0.7`
- sigma `0.8`
- sigma `1.2`
- sigma `1.3`

Reference already completed from prior sweep:

- sigma `1.0`: `llama3p2_1b_instruct_alpaca_b600_fixedrandom_stage4_randfull_localgauss_a007_s0_finaleval_20260804`
## 2026-08-13 - Continuous Reward-Model Three-Stage SFT

- Added `prepare_rewardmodel_three_stage.py` to align the pointwise, pairwise,
  and listwise files by question ID, shuffle with seed 42, preserve float
  pointwise rewards, add local pairwise `choice_code`, and derive full
  listwise rankings from the independent listwise scores.
- Prepared `split1500_500/` and `mix200_eval300/` under
  `data/rewardmodel/reward-model/`.
- Added `run_rewardmodel_three_stage_sft.py`. The evaluated model is a causal
  LM trained sequentially with decimal pointwise SFT, pairwise SFT, and
  listwise SFT. Selector mode uses a temporary continuous proxy only for
  acquisition and then trains/evaluates the causal-LM SFT model.
- Compact final metrics are continuous pointwise MAE, PairAcc, Listwise Acc,
  and listwise rank MAE. Continuous smoothing uses alpha 0.01 shrinkage toward
  the selected pointwise mean; Mix is the no-smooth ep10 control.
- Added `launch_rewardmodel_three_stage_20260813.sh` for Llama-3.2-1B-Instruct
  and Qwen3-1.7B Mix/Selector runs. It was not launched because the current
  shuffled Alpaca/Dolly batch is still occupying GPUs 0-5 and GPUs 6-7 remain
  reserved.
## 2026-08-14 - Qwen3 GPT-5 Selector+Smooth LoRA Table

Launched the seven missing single-seed LoRA cells for the Qwen3 surrogate-size
table on Alpaca and Dolly. The completed Qwen3-1.7B Alpaca result is reused, so
the full LoRA matrix contains eight cells total.

- Common setup: seed 42, budget 600, candidate-triple selector, LM-head proxy,
  pool 100, init 80, acquisition batch 20, one triple per question, diversity
  1.0, uncertainty 0.25, bias 1.0, no exploration, and Stage4 full stratified
  replay.
- Training: LoRA + 4-bit, pointwise/pairwise/listwise/Stage4 one epoch each,
  batch size 1, gradient accumulation 16, learning rate 1e-4, max length 4096.
- Smoothing: local Gaussian, alpha 0.1, sigma 1.0, all stages.
- Alpaca GPT-5 data: `Alpaca/gpt5/train-20k.json` and
  `Alpaca/gpt5/val-2k-eval-listwise.json`.
- Dolly GPT-5 data: the leakage-free
  `Dolly/gpt5/train9k_pointwise_pairwise_no_val_overlap.json` and
  `Dolly/gpt5/val3k_pairwise_listwise.json`.
- Launcher: `launch_qwen3_gpt5_selector_smooth_lora_table_20260814.sh`.
- Logs/status:
  `outputs/qwen3_gpt5_selector_smooth_lora_logs_20260814_table_lora_gpt5_v1/`.
- Workers: GPU1 runs Alpaca 0.6B then Dolly 0.6B and 1.7B; GPU2 runs Alpaca
  4B; GPU3 runs Dolly 4B; GPU5 runs Alpaca 8B then Dolly 8B. GPUs 6 and 7 are
  intentionally reserved. The unrelated Mix job remains on GPU0 and the
  external process on GPU4 is untouched.

All four first-wave jobs passed argument/data validation and model loading.
