# Claude LoRA Selector And MixEp10 Queue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a single NVMe-backed GPU queue for eight Claude-labelled experiments: four end-to-end selector runs and four LoRA + 4-bit MixEp10 controls.

**Architecture:** `launch_claude_lora_auto_queue.sh` owns task naming, data/model resolution, idle-GPU dispatch, output protection and logs. A task mode branch invokes the existing generative three-stage entry point for selector experiments or the historical true-value control entry point for MixEp10. Both use Claude's explicit pairwise and listwise labels without changing source JSON.

**Tech Stack:** Bash, tmux, NVIDIA CLI, Python, Transformers, PEFT, bitsandbytes, existing project training entry points.

---

### Task 1: Define Regression Expectations

**Files:**
- Create: `tests/test_claude_lora_queue.py`

- [ ] **Step 1: Write the failing launcher-content test**

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "launch_claude_lora_auto_queue.sh"


def test_claude_queue_contains_eight_lora_jobs_with_explicit_validation():
    text = LAUNCHER.read_text(encoding="utf-8")
    for job in (
        "selector_alpaca_llama1b",
        "selector_gpt4all_llama1b",
        "selector_alpaca_qwen1p7b",
        "selector_gpt4all_qwen1p7b",
        "mixep10_alpaca_llama1b",
        "mixep10_gpt4all_llama1b",
        "mixep10_alpaca_qwen1p7b",
        "mixep10_gpt4all_qwen1p7b",
    ):
        assert job in text
    for path in (
        "$ROOT/data/alpaca/claude/train.json",
        "$ROOT/data/alpaca/claude/val.json",
        "$ROOT/data/gpt4all/claude/train.json",
        "$ROOT/data/gpt4all/claude/val.json",
    ):
        assert path in text
    assert "--pairwise-eval-dataset" in text
    assert "--listwise-eval-dataset" in text
    assert "--use-lora" in text
    assert "--load-in-4bit" in text


def test_claude_queue_keeps_selector_and_mixep10_protocols_distinct():
    text = LAUNCHER.read_text(encoding="utf-8")
    for selector_argument in (
        "--train-selection-mode candidate_triple_selector",
        "--candidate-selector-proxy-mode lm_head",
        "--candidate-selector-init-triples 80",
        "--candidate-selector-batch-size 20",
        "--candidate-selector-max-score-candidates 100",
        "--stage4-replay-strategy stratified_triple",
        "--pointwise-global-smooth-alpha 0.1",
    ):
        assert selector_argument in text
    for mix_argument in (
        "--mode trueval_three_stage",
        "--pointwise-train-samples 200",
        "--pairwise-train-pairs 200",
        "--listwise-train-examples 200",
        "--pointwise-epochs 10",
        "--pairwise-epochs 10",
        "--listwise-epochs 10",
    ):
        assert mix_argument in text
```

- [ ] **Step 2: Run the test before implementation**

Run: `python -m pytest -q tests/test_claude_lora_queue.py`

Expected: failure because the launcher does not yet exist.

### Task 2: Create The Claude Queue

**Files:**
- Create: `launch_claude_lora_auto_queue.sh`

- [ ] **Step 1: Add environment and job definitions**

Set `OUTPUT_ROOT` to `/opt/dlami/nvme/cyl/autodl-tmp/JudgeStealer_outputs`, use a `claude_lora_auto_queue_logs` subdirectory, and define all eight selector/MixEp10 job IDs. Accept one or more allowed GPU IDs and support `SKIP_JOBS`.

- [ ] **Step 2: Resolve models, data and unique names**

Map `llama1b` to `$ROOT/models/Llama-3.2-1b-instruct` and `qwen1p7b` to `$ROOT/models/Qwen3-1.7B`. Map `alpaca` and `gpt4all` to their lower-case `data/*/claude/train.json` and `val.json` paths. Name outputs with the model, dataset, `claude`, `lora`, and either `selector` or `trueval_mix200pw200pair200list_ep10`.

- [ ] **Step 3: Implement selector task execution**

Call `run_pointwise5answers_three_stage_pairwise_listwise_sft_v1.py` with explicit pointwise, pairwise and listwise paths; `--use-lora --load-in-4bit`; budget 600; selector settings `80/20/100`; stage-four full stratified replay; one epoch per stage; learning rate `1e-4`; local-Gaussian smoothing `alpha=0.1`, `sigma=1.0`, all stages; and final-only evaluation.

- [ ] **Step 4: Implement MixEp10 task execution**

Call `run_newnew_one_answer_trueval_three_stage_sft.py` with `--mode trueval_three_stage`; the Claude train file plus the same Claude validation file for pairwise and listwise inputs; 200 pointwise, pairwise and listwise training examples; ten epochs for each stage; replay ratios of zero; `--use-lora --load-in-4bit`; learning rate `1e-4`; and no smoothing arguments.

- [ ] **Step 5: Add safe dispatch and output handling**

Reuse the existing queue's output filesystem check, GPU UUID/memory/compute-process checks, `metrics_compact.json` completion marker, running-process guard, incomplete-output refusal, per-GPU worker tracking, and per-dispatch `SKIP_JOBS` processing.

### Task 3: Verify The New Queue

**Files:**
- Test: `tests/test_claude_lora_queue.py`
- Test: `tests/test_judge_data_loaders.py`
- Test: `launch_claude_lora_auto_queue.sh`

- [ ] **Step 1: Run the new launcher test**

Run: `python -m pytest -q tests/test_claude_lora_queue.py`

Expected: all tests pass.

- [ ] **Step 2: Run Claude loader regression coverage**

Run: `python -m pytest -q tests/test_judge_data_loaders.py`

Expected: all Claude answer, pairwise and listwise field tests pass.

- [ ] **Step 3: Validate Bash syntax**

Run: `bash -n launch_claude_lora_auto_queue.sh`

Expected: exit status 0 with no output.

- [ ] **Step 4: Run the related queue test suite**

Run: `python -m pytest -q tests/test_claude_lora_queue.py tests/test_judge_data_loaders.py tests/test_qwen3_gpt5_fullft_queue.py`

Expected: all tests pass.

### Task 4: Document Operation And Publish

**Files:**
- Modify: `docs/LAUNCHER_RUNBOOK.md`
- Create: `docs/CLAUDE_LORA_EXPERIMENTS.md`

- [ ] **Step 1: Record the eight-job matrix and protocol**

Document the model/data matrix, the distinction between Selector and MixEp10, exact Claude paths, required Llama model path, NVMe output convention, and status/metrics commands.

- [ ] **Step 2: Add start commands**

Document `tmux new -s claude_lora`, model download verification, and `./launch_claude_lora_auto_queue.sh <gpu_id> [gpu_id ...]`.

- [ ] **Step 3: Commit and push only task-owned files**

Run:

```bash
git add launch_claude_lora_auto_queue.sh tests/test_claude_lora_queue.py \
  docs/LAUNCHER_RUNBOOK.md docs/CLAUDE_LORA_EXPERIMENTS.md
git commit -m "feat: add Claude LoRA experiment queue"
git push origin main
```

Do not stage unrelated local notes or user-owned scripts.
