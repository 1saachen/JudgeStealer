# GPT-5 CoT Auto Queue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible eight-job launcher for GPT-5-scored Alpaca and GPT4All CoT experiments.

**Architecture:** A Bash queue maps paper names to the existing CoT runner's `mix` and `stage4` modes. It validates raw CoT sources, creates one prepared-data directory per dataset, and invokes the existing runner with explicit prepared-data paths.

**Tech Stack:** Bash, Python pytest, existing `prepare_alpaca_cot_4066.py`, existing `run_alpaca_cot_stage4_mix.py`.

---

### Task 1: Define Launcher Contract Test

**Files:**
- Create: `tests/test_gpt5_cot_auto_queue.py`

- [ ] **Step 1: Write failing tests**

```python
def test_launcher_maps_naive_and_ours_to_distinct_runner_modes():
    text = Path("launch_gpt5_cot_auto_queue.sh").read_text(encoding="utf-8")
    assert 'naive) mode=mix' in text
    assert 'ours) mode=stage4' in text

def test_launcher_accepts_dataset_roots_from_environment():
    text = Path("launch_gpt5_cot_auto_queue.sh").read_text(encoding="utf-8")
    assert 'ALPACA_COT_DATA_DIR' in text
    assert 'GPT4ALL_COT_DATA_DIR' in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_gpt5_cot_auto_queue.py`

Expected: FAIL because `launch_gpt5_cot_auto_queue.sh` does not exist.

### Task 2: Add GPT-5 CoT Queue Launcher

**Files:**
- Create: `launch_gpt5_cot_auto_queue.sh`

- [ ] **Step 1: Implement the launcher**

Add a `run_job` function that maps the eight job names to dataset, surrogate, and paper method; validates the model and raw CoT input files; prepares each dataset only when `manifest.json` is absent; and calls `run_alpaca_cot_stage4_mix.py` with explicit `--train-questions`, `--eval-questions`, and Mix inputs.

- [ ] **Step 2: Run launcher contract tests**

Run: `pytest -q tests/test_gpt5_cot_auto_queue.py`

Expected: PASS.

### Task 3: Run Regression Checks

**Files:**
- Test: `tests/test_alpaca_cot_stage4_mix.py`
- Test: `tests/test_gpt5_cot_auto_queue.py`

- [ ] **Step 1: Validate Bash syntax**

Run: `bash -n launch_gpt5_cot_auto_queue.sh`

Expected: exit code 0.

- [ ] **Step 2: Run focused tests**

Run: `PYTHONPATH="$PWD" pytest -q tests/test_gpt5_cot_auto_queue.py tests/test_alpaca_cot_stage4_mix.py`

Expected: PASS.
