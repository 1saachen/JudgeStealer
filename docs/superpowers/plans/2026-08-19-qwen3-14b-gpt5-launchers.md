# Qwen3-14B GPT-5 Launchers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reproducible launchers for Qwen3-14B GPT-5 Alpaca/GPT4All LoRA and Full-FT experiments.

**Architecture:** Keep experiment arguments in the existing three-stage Python entry point. Add one single-GPU LoRA queue and one two-GPU FSDP Full-FT queue, both using NVMe outputs, explicit job names, completion checks, and logs. Add focused static tests for task mappings and incompatible training flags.

**Tech Stack:** Bash, `nvidia-smi`, `flock`, `torchrun`, Python/pytest, existing Transformers/PEFT/FSDP runner.

---

### Task 1: Add static launcher contract tests

**Files:**
- Create: `tests/test_qwen3_14b_launchers.py`

- [x] **Step 1: Write tests for both launchers**

Assert that each launcher contains the four task/data mappings, the Qwen3-14B model path, the expected runner, output tags, and common protocol flags. Assert the LoRA launcher contains `--use-lora` and `--load-in-4bit` but no FSDP/torchrun, while the Full-FT launcher contains `torchrun`, `--fsdp`, and no LoRA/4-bit flags.

- [x] **Step 2: Run the focused tests and confirm they fail because the launchers do not exist**

Run: `pytest -q tests/test_qwen3_14b_launchers.py`

Expected: collection or file/path failure until the two launchers are created.

### Task 2: Implement the single-GPU LoRA queue

**Files:**
- Create: `launch_qwen3_14b_gpt5_lora_auto_queue.sh`

- [x] **Step 1: Add task resolution and data validation**

Define `alpaca` and `gpt4all` jobs, map both to `models/Qwen3-14B`, use the GPT-5 train/eval files already used by the full-FT queue, and reject missing model/config/data/runner files before starting.

- [x] **Step 2: Add idle-GPU scheduling and restart guards**

Accept allowed GPU IDs, use `nvidia-smi` UUID/memory/process checks, schedule at most one worker per GPU, skip `metrics_compact.json`, and report incomplete output directories without overwriting them. Use `SKIP_JOBS` and NVMe `OUTPUT_ROOT`/log paths.

- [x] **Step 3: Add the LoRA training invocation**

Invoke `run_pointwise5answers_three_stage_pairwise_listwise_sft_v1.py` with budget 600, one epoch for stages 1--4, max length 4096, per-device batch 1, final-only evaluation, local Gaussian smoothing, selector settings matching the existing table, `--use-lora`, `--load-in-4bit`, `--learning-rate 1e-4`, and `--proxy-lr 1e-4`.

- [x] **Step 4: Run shell syntax and focused tests**

Run: `bash -n launch_qwen3_14b_gpt5_lora_auto_queue.sh` and `pytest -q tests/test_qwen3_14b_launchers.py`.

Expected: no shell syntax error and the LoRA assertions pass.

### Task 3: Implement the two-GPU FSDP Full-FT launcher

**Files:**
- Create: `launch_qwen3_14b_gpt5_fullft_fsdp.sh`

- [x] **Step 1: Add paired-GPU validation and job scheduling**

Accept exactly two GPU IDs, validate both with `nvidia-smi`, wait until both are idle, and run Alpaca then GPT4All serially so a pair is never shared by another Full-FT job. Preserve completed/incomplete output guards and NVMe logs.

- [x] **Step 2: Add the Full-FT torchrun invocation**

Set `CUDA_VISIBLE_DEVICES` to the two selected physical IDs and invoke `torchrun --standalone --nproc_per_node=2` with `--fsdp "full_shard auto_wrap"`, `--fsdp-transformer-layer-cls-to-wrap Qwen3DecoderLayer`, activation checkpointing, full state dict, no LoRA or 4-bit flags, selector full mode, learning rate `1e-5`, and proxy learning rate `1e-5`.

- [x] **Step 3: Run shell syntax and focused tests**

Run: `bash -n launch_qwen3_14b_gpt5_fullft_fsdp.sh` and `pytest -q tests/test_qwen3_14b_launchers.py`.

Expected: no shell syntax error and the Full-FT assertions pass.

### Task 4: Document and verify the handoff

**Files:**
- Modify: `docs/LAUNCHER_RUNBOOK.md`

- [x] **Step 1: Document model path, commands, skips, logs, and metrics**

Add the Qwen3-14B model directory, the LoRA and Full-FT commands, `SKIP_JOBS` usage for LoRA, the paired-GPU requirement for Full-FT, and the two log roots.

- [x] **Step 2: Run final static verification**

Run: `bash -n launch_qwen3_14b_gpt5_lora_auto_queue.sh; bash -n launch_qwen3_14b_gpt5_fullft_fsdp.sh; pytest -q tests/test_qwen3_14b_launchers.py; git diff --check`.

Expected: both syntax checks, all focused tests, and whitespace checks pass. No training process is started.

- [x] **Step 3: Review the diff without touching unrelated worktree changes**

Run: `git status --short; git diff --stat -- launch_qwen3_14b_gpt5_lora_auto_queue.sh launch_qwen3_14b_gpt5_fullft_fsdp.sh docs/LAUNCHER_RUNBOOK.md tests/test_qwen3_14b_launchers.py`.
