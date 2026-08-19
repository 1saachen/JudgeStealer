# Qwen3-1.7B GPT-5 Mix Budget Queue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add ten LoRA true-value Mix jobs for Alpaca and GPT4All at B=0.5%, 1%, 2%, 5%, and 10% using the existing Ours comparison protocol.

**Architecture:** Reuse `run_newnew_one_answer_trueval_three_stage_sft.py` and add only a launcher plus tests. The launcher maps each dataset's measured train-split denominator to equal pointwise/pairwise/listwise sample counts, trains each stage for 10 epochs with no replay or smoothing, and writes to a dedicated local-output root.

**Tech Stack:** Bash, Python, pytest, existing Transformers/LoRA SFT entry point.

### Task 1: Lock the job matrix and budget mapping

**Files:** Create `tests/test_qwen3_1p7b_mix_budget_queue.py` and `launch_qwen3_1p7b_mix_budget_auto_queue.sh`.

- [ ] Test exactly ten unique jobs, two datasets, five budgets, and the measured mapping `alpaca: 18000`, `gpt4all: 8100`.
- [ ] Test the launcher uses true-value mode, equal samples per task, 10 epochs, no replay, no smoothing, LoRA+4bit, final-only evaluation, and dedicated local output.
- [ ] Implement idle-GPU scheduling, NFS rejection, completion/incomplete-output protection, per-job locking, and `SKIP_JOBS`.

### Task 2: Verify and publish

- [ ] Run focused and existing queue tests.
- [ ] Run Python compilation and diff checks.
- [ ] Commit the launcher and tests, merge to `main`, rerun focused tests, and push `origin/main` without staging unrelated user changes.
