# Qwen3-1.7B Single-Task Controls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add six Qwen3-1.7B LoRA + 4-bit GPT-5 controls that train on exactly 600 examples of only pointwise, pairwise, or listwise supervision and evaluate all three tasks.

**Architecture:** Extend the existing true-value runner with an explicit `single_task` mode so data splitting, prompts, training, evaluation, and compact metrics remain aligned with Naive Mix. Add a separate NVMe-backed idle-GPU queue for the Alpaca/GPT4All by pointwise/pairwise/listwise matrix, without changing the existing Mix or selector launchers.

**Tech Stack:** Python 3.10, PyTorch/Transformers/PEFT, Bash, pytest.

---

## File Structure

- Modify `run_newnew_one_answer_trueval_three_stage_sft.py`: parse and execute one selected training task while preserving all three evaluation paths.
- Create `launch_qwen3_1p7b_single_task_auto_queue.sh`: resolve the six jobs, enforce the fixed protocol, and dispatch them to idle GPUs.
- Create `tests/test_qwen3_1p7b_single_task.py`: unit and source-level regression tests for runner mode, sample routing, single Trainer execution, and final metrics.
- Create `tests/test_qwen3_1p7b_single_task_queue.py`: launcher matrix, fixed arguments, path, output, and scheduling checks.
- Create `docs/QWEN3_1P7B_SINGLE_TASK_EXPERIMENTS.md`: server usage, paths, job names, logs, and metrics commands.

### Task 1: Add Explicit Single-Task Training Mode

**Files:**
- Modify: `run_newnew_one_answer_trueval_three_stage_sft.py:407-449`
- Modify: `run_newnew_one_answer_trueval_three_stage_sft.py:476-740`
- Create: `tests/test_qwen3_1p7b_single_task.py`

- [ ] **Step 1: Write failing routing and parser tests**

Create tests that require an explicit task choice and exact budget routing:

```python
import run_newnew_one_answer_trueval_three_stage_sft as runner


def test_single_task_counts_put_the_entire_budget_in_one_task():
    assert runner._single_task_counts("pointwise", 600) == (600, 0, 0)
    assert runner._single_task_counts("pairwise", 600) == (0, 600, 0)
    assert runner._single_task_counts("listwise", 600) == (0, 0, 600)


def test_single_task_training_spec_uses_existing_epoch_stage_names():
    point = [("pointwise", "p", "t", 0)]
    pair = [("pairwise", "p", "t", 0)]
    listing = [("listwise", "p", "t", 0)]
    assert runner._single_task_training_spec("pointwise", point, pair, listing) == (
        point,
        "stage1_pointwise",
    )
    assert runner._single_task_training_spec("pairwise", point, pair, listing) == (
        pair,
        "stage2_pairwise",
    )
    assert runner._single_task_training_spec("listwise", point, pair, listing) == (
        listing,
        "stage3_listwise",
    )
```

Add source assertions that `parse_args()` accepts `single_task`, defines
`--single-task` with exactly `pointwise`, `pairwise`, and `listwise`, and rejects an
empty task whenever mode is `single_task`.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
python -m pytest -q tests/test_qwen3_1p7b_single_task.py
```

Expected: FAIL because `_single_task_counts`, `_single_task_training_spec`, and the
new CLI mode do not exist.

- [ ] **Step 3: Implement task routing helpers and CLI validation**

Add constants and helpers near the existing JSON helpers:

```python
SINGLE_TASK_CHOICES = ("pointwise", "pairwise", "listwise")


def _single_task_counts(single_task: str, budget: int) -> Tuple[int, int, int]:
    task = str(single_task)
    if task not in SINGLE_TASK_CHOICES:
        raise ValueError(f"unknown single task: {task}")
    if int(budget) <= 0:
        raise ValueError("single-task budget must be positive")
    return tuple(int(budget) if name == task else 0 for name in SINGLE_TASK_CHOICES)


def _single_task_training_spec(
    single_task: str,
    point_items: Sequence[Tuple[str, str, str, int]],
    pair_items: Sequence[Tuple[str, str, str, int]],
    list_items: Sequence[Tuple[str, str, str, int]],
) -> Tuple[Sequence[Tuple[str, str, str, int]], str]:
    mapping = {
        "pointwise": (point_items, "stage1_pointwise"),
        "pairwise": (pair_items, "stage2_pairwise"),
        "listwise": (list_items, "stage3_listwise"),
    }
    if str(single_task) not in mapping:
        raise ValueError(f"unknown single task: {single_task}")
    return mapping[str(single_task)]
```

Extend mode choices with `single_task`, add:

```python
parser.add_argument("--single-task", choices=SINGLE_TASK_CHOICES, default="")
```

After parsing, require `--single-task` exactly when `--mode single_task` is used.

- [ ] **Step 4: Route data construction by task without evaluation leakage**

For `single_task`, derive the three requested train counts using
`_single_task_counts(args.single_task, args.budget)`. Always build the pointwise
evaluation split. Build pairwise/listwise training splits only for the selected task;
for an unselected task load the entire corresponding validation file as evaluation
and record `split: not_trained_full_eval`.

Preserve the existing `_split_pairwise_trueval` ABC-record exclusion and
`_split_listwise_trueval` index exclusion for the selected pairwise/listwise task.
Write the resolved task and exact counts into `config.json`, `summary.json`, and all
existing train/eval JSONL and stats artifacts.

- [ ] **Step 5: Train exactly once and evaluate all tasks once**

Before the current Stage-1/2/3 branch, add a `single_task` branch that:

```python
single_items, single_stage_name = _single_task_training_spec(
    args.single_task, point_items, pair_items, list_items
)
single_model_dir = out / f"single_task_{args.single_task}_sft_model"
train_stats[f"single_task_{args.single_task}"], model, tokenizer = three._train_sft_on_items(
    model_name_or_path=str(args.llama),
    model=None,
    tokenizer=None,
    items=single_items,
    output_dir=single_model_dir,
    cfg=cfg,
    stage_name=single_stage_name,
)
metrics = _eval_all(
    model=model,
    tokenizer=tokenizer,
    cfg=cfg,
    pointwise_eval=pointwise_eval,
    pairwise_eval=pairwise_eval,
    listwise_eval=listwise_eval,
)
```

Store all three metric dictionaries under `after_single_task`; skip the existing
three-stage branch for this mode. Set summary mode to `single_task`, include the chosen
task, and let the existing `_compact_metrics` produce `metrics_compact.json`.

- [ ] **Step 6: Run focused tests and commit**

Run:

```bash
python -m pytest -q \
  tests/test_qwen3_1p7b_single_task.py \
  tests/test_qwen3_1p7b_mix_budget_queue.py
python -m py_compile run_newnew_one_answer_trueval_three_stage_sft.py
```

Expected: all tests PASS and compilation exits 0.

Commit only the runner and focused test:

```bash
git add run_newnew_one_answer_trueval_three_stage_sft.py tests/test_qwen3_1p7b_single_task.py
git commit -m "feat: add single-task true-value controls"
```

### Task 2: Add the Six-Job Idle-GPU Queue

**Files:**
- Create: `launch_qwen3_1p7b_single_task_auto_queue.sh`
- Create: `tests/test_qwen3_1p7b_single_task_queue.py`

- [ ] **Step 1: Write the failing launcher contract tests**

Test that the launcher contains exactly:

```python
expected = {
    f"{dataset}_{task}_only"
    for dataset in ("alpaca", "gpt4all")
    for task in ("pointwise", "pairwise", "listwise")
}
```

Require the current model/data paths and these fixed arguments:

```text
--mode single_task
--single-task "$single_task"
--budget 600
--pointwise-epochs 10
--pairwise-epochs 10
--listwise-epochs 10
--per-device-batch-size 1
--gradient-accumulation-steps 16
--learning-rate 1e-4
--max-length 4096
--eval-batch-size 1
--eval-stages final
--use-lora
--load-in-4bit
```

Also require NVMe output, NFS rejection, idle GPU detection, job locks,
`SKIP_JOBS`, completed-job skip, and incomplete-output refusal.

- [ ] **Step 2: Run the launcher tests and verify RED**

Run:

```bash
python -m pytest -q tests/test_qwen3_1p7b_single_task_queue.py
```

Expected: FAIL because the launcher does not exist.

- [ ] **Step 3: Implement the launcher using the established queue pattern**

Use `launch_qwen3_1p7b_mix_budget_auto_queue.sh` as the complete scheduling reference.
Define the six explicit jobs and resolve:

```bash
dataset="${job%%_*}"
single_task="${job#${dataset}_}"
single_task="${single_task%_only}"
```

Map Alpaca and GPT4All to the same pointwise, pairwise, and listwise files as the
Mix budget queue. Name outputs:

```bash
name="qwen3_1p7b_${dataset}_gpt5_b600_lora_trueval_${single_task}_only_ep10_noreplay_nosmooth"
```

Invoke the runner with the fixed arguments from Step 1. Write outputs beneath:

```text
/opt/dlami/nvme/cyl/autodl-tmp/JudgeStealer_outputs/qwen3_1p7b_single_task_seed42
```

- [ ] **Step 4: Verify Bash and queue tests**

Run:

```bash
bash -n launch_qwen3_1p7b_single_task_auto_queue.sh
python -m pytest -q \
  tests/test_qwen3_1p7b_single_task_queue.py \
  tests/test_qwen3_1p7b_single_task.py \
  tests/test_qwen3_1p7b_mix_budget_queue.py
```

Expected: Bash exits 0 and all tests PASS.

- [ ] **Step 5: Commit the queue**

```bash
git add launch_qwen3_1p7b_single_task_auto_queue.sh tests/test_qwen3_1p7b_single_task_queue.py
git commit -m "feat: add Qwen3 1.7B single-task queue"
```

### Task 3: Document, Verify, and Publish

**Files:**
- Create: `docs/QWEN3_1P7B_SINGLE_TASK_EXPERIMENTS.md`
- Modify: `docs/LAUNCHER_RUNBOOK.md`

- [ ] **Step 1: Write the operator documentation**

Document the six jobs, exact single-task definition, data paths, fixed protocol, NVMe
output root, and these commands:

```bash
tmux new -s qwen17_single_task
./launch_qwen3_1p7b_single_task_auto_queue.sh 0 1 2 3 4 5

tail -f /opt/dlami/nvme/cyl/autodl-tmp/JudgeStealer_outputs/\
qwen3_1p7b_single_task_seed42/logs/job_status.log

find /opt/dlami/nvme/cyl/autodl-tmp/JudgeStealer_outputs/\
qwen3_1p7b_single_task_seed42 -name metrics_compact.json -print -exec cat {} \;
```

Add a short reference to the new guide in `docs/LAUNCHER_RUNBOOK.md`.

- [ ] **Step 2: Run full focused verification**

Run:

```bash
python -m pytest -q \
  tests/test_qwen3_1p7b_single_task.py \
  tests/test_qwen3_1p7b_single_task_queue.py \
  tests/test_qwen3_1p7b_mix_budget_queue.py
python -m py_compile run_newnew_one_answer_trueval_three_stage_sft.py
bash -n launch_qwen3_1p7b_single_task_auto_queue.sh
git diff --check
```

Expected: all pytest tests PASS; compilation, Bash syntax, and diff checks exit 0.

- [ ] **Step 3: Commit documentation and push only feature commits**

```bash
git add docs/QWEN3_1P7B_SINGLE_TASK_EXPERIMENTS.md docs/LAUNCHER_RUNBOOK.md
git commit -m "docs: add single-task experiment runbook"
git log --oneline --max-count=4
git status --short
```

Use an isolated worktree based on `origin/main` to cherry-pick the design, runner,
queue, and documentation commits. Rerun Step 2 there, then push that isolated HEAD to
`origin/main`. Do not include unrelated staged or unstaged user files.
