# Reward-Model Best-Choice Listwise Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and evaluate reward-model listwise examples using only the source `listwise_choice` best-answer label.

**Architecture:** Add a small listwise best-choice target/parser path in the reward-model entry point. Both `native_json` and `converted` modes will use the source choice for listwise training, while pointwise and pairwise paths remain unchanged. The compact metric will expose best-choice accuracy as `listwise_acc`.

**Tech Stack:** Python, PyTorch SFT helpers, pytest, JSON reward-model records.

---

### Task 1: Add failing focused tests

**Files:**
- Modify: `tests/test_rewardmodel_pointwise.py`
- Test: `run_rewardmodel_three_stage_sft.py`

- [ ] **Step 1: Test direct source-choice target construction**

Add a test that passes a listwise record whose `listwise_choice` is `C` while its numeric scores would imply a different winner. Assert the generated target is the canonical best-choice target for `C` and contains no score fields or ranking text.

- [ ] **Step 2: Test best-choice evaluation helper**

Add a test covering `Response2` as a valid prediction and malformed output as invalid, with the expected accuracy and invalid count.

- [ ] **Step 3: Run the focused tests and verify failure**

Run:

```powershell
pytest -q tests/test_rewardmodel_pointwise.py
```

Expected: FAIL because the best-choice target and evaluation helpers do not yet exist.

### Task 2: Implement best-choice-only listwise training and evaluation

**Files:**
- Modify: `run_rewardmodel_three_stage_sft.py`

- [ ] **Step 1: Add canonical best-choice formatting/parsing helpers**

Implement helpers that format `A/B/C` source choices as `Best: [Response1]`-style targets and parse the same response IDs from generated text. Reject missing or out-of-range choices as invalid.

- [ ] **Step 2: Route both listwise target modes through source choice**

Replace the current native listwise JSON target and converted ranking target with best-choice targets derived from `record["listwise_choice"]`. Keep pointwise and pairwise target construction unchanged.

- [ ] **Step 3: Evaluate best-choice accuracy**

Evaluate listwise generations by comparing parsed best response IDs against the raw `listwise_choice` field. Preserve detailed diagnostics, but set compact `listwise_acc` to best-choice accuracy.

- [ ] **Step 4: Run the focused tests and verify success**

Run:

```powershell
pytest -q tests/test_rewardmodel_pointwise.py
```

Expected: PASS.

### Task 3: Regression verification and documentation consistency

**Files:**
- Modify: `README.md` or reward-model documentation only if the existing target-format description becomes inaccurate.

- [ ] **Step 1: Run all tests**

Run:

```powershell
pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Check the diff and whitespace**

Run:

```powershell
git diff --check
git diff --stat
```

- [ ] **Step 3: Confirm the data contract**

Verify `data/reward-model/reward-model/listwise.json` contains `listwise_choice` and that the implementation does not derive listwise labels from pointwise scores.
