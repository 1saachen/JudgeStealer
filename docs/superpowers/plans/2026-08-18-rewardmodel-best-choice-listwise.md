# Reward-Model Best-Choice Listwise Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and evaluate reward-model pairwise/listwise examples with source choice labels and soft targets only for score ties.

**Architecture:** Add canonical best-choice targets and tie-aware metadata to the reward-model entry point. Pointwise remains continuous; pairwise keeps source choices and softens only equal-score non-explicit ties; listwise emits best-choice targets and softens equal top groups. Add a LoRA queue for the Naive/Ours table.

**Tech Stack:** Python, PyTorch SFT helpers, pytest, JSON reward-model records.

---

### Task 1: Add failing focused tests

**Files:**
- Modify: `tests/test_rewardmodel_pointwise.py`
- Test: `run_rewardmodel_three_stage_sft.py`

- [x] **Step 1: Test direct source-choice target construction**

Add a test that passes a listwise record whose `listwise_choice` is `C` while its numeric scores would imply a different winner. Assert the generated target is the canonical best-choice target for `C` and contains no score fields or ranking text.

- [x] **Step 2: Test best-choice evaluation helper**

Add a test covering `Response2` as a valid prediction and malformed output as invalid, with the expected accuracy and invalid count.

- [x] **Step 3: Run the focused tests and verify failure**

Run:

```powershell
pytest -q tests/test_rewardmodel_pointwise.py
```

Expected: FAIL because the best-choice target and evaluation helpers do not yet exist.

### Task 2: Implement best-choice-only listwise training and evaluation

**Files:**
- Modify: `run_rewardmodel_three_stage_sft.py`

- [x] **Step 1: Add canonical best-choice formatting/parsing helpers**

Implement helpers that format `A/B/C` source choices as `Best: [Response1]`-style targets and parse the same response IDs from generated text. Reject missing or out-of-range choices as invalid.

- [x] **Step 2: Route both listwise target modes through source choice**

Replace the current native listwise JSON target and converted ranking target with best-choice targets derived from `record["listwise_choice"]`. Keep pointwise and pairwise target construction unchanged.

- [x] **Step 3: Evaluate best-choice accuracy**

Evaluate listwise generations by comparing parsed best response IDs against the raw `listwise_choice` field. Preserve detailed diagnostics, but set compact `listwise_acc` to best-choice accuracy.

- [x] **Step 4: Run the focused tests and verify success**

Run:

```powershell
pytest -q tests/test_rewardmodel_pointwise.py
```

Expected: PASS.

### Task 3: Match the Naive 200/200/200 control

**Files:**
- Modify: `run_rewardmodel_three_stage_sft.py`
- Modify: `tests/test_rewardmodel_pointwise.py`

- [x] **Step 1: Add deterministic item sampling**

Add a helper that samples an exact number of final SFT items while preserving
the alignment of choice distributions and candidate targets.

- [x] **Step 2: Apply final-item limits only to Naive mix mode**

Keep `pointwise_train_samples=200` and add explicit pairwise/listwise final
sample limits of 200. Do not truncate selector-mode Ours data.

- [x] **Step 3: Verify the focused tests**

Run:

```powershell
pytest -q tests/test_rewardmodel_pointwise.py
```

Expected: PASS with exactly aligned sampled items and metadata.

### Task 4: Regression verification and documentation consistency

**Files:**
- Modify: `README.md` or reward-model documentation only if the existing target-format description becomes inaccurate.

- [x] **Step 1: Run all tests**

Run:

```powershell
pytest -q
```

Expected: all tests pass.

Observed: 71 tests passed and one pre-existing unrelated failure remains in
`tests/test_judge_data_loaders.py::test_trueval_control_builds_lora_run_config`
because `RunConfig` currently requires `budget_percent`.

- [x] **Step 2: Check the diff and whitespace**

Run:

```powershell
git diff --check
git diff --stat
```

- [x] **Step 3: Confirm the data contract**

Verify `data/reward-model/reward-model/listwise.json` contains `listwise_choice`,
that unique listwise labels remain source-driven, and that scores are used only
to identify equal top groups for soft targets.

### Task 5: Add the reward-model LoRA experiment queue

**Files:**
- Create: `launch_rewardmodel_lora_auto_queue.sh`
- Create: `docs/REWARDMODEL_LORA_EXPERIMENTS.md`
- Test: `tests/test_rewardmodel_lora_queue.py`

- [x] **Step 1: Add eight explicit Naive/Ours jobs**

The queue covers both models and both reward-model source directories, checks
the model/data contract, prepares the fixed `1500/200/300` split, and refuses
to overwrite incomplete output directories.

- [x] **Step 2: Encode the training protocols**

Naive uses 200 pointwise, pairwise, and listwise final items for 10 epochs
without smoothing. Ours uses selector budget 600, `80/20/100` acquisition
settings, one epoch per stage, LoRA + 4-bit, and the shared 300-example eval.

- [x] **Step 3: Verify launcher and focused tests**

Run `python -m pytest -q tests/test_rewardmodel_pointwise.py
tests/test_rewardmodel_lora_queue.py` and `python -m py_compile
run_rewardmodel_three_stage_sft.py prepare_rewardmodel_three_stage.py`.
