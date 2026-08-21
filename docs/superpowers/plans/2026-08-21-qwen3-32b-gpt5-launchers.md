# Qwen3-32B GPT-5 Launchers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended; inline execution is used here). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add robust LoRA and Full-FT GPT-5 launchers for Qwen3-32B across Alpaca and GPT4All.

**Architecture:** Clone the proven Qwen14B queue behavior into two Qwen3-32B shell launchers. Make the model path and output tag environment-overridable, and use `Qwen3DecoderLayer` for the FSDP auto-wrap policy.

**Tech Stack:** Bash, `nvidia-smi`, `flock`, `python`, `torchrun`, existing three-stage SFT runner.

---

### Task 1: Add launcher contract tests

**Files:**
- Create: `tests/test_llama32_gpt5_launchers.py`

- [ ] **Step 1: Write tests** for both launcher files covering model override, Alpaca/GPT4All paths, LoRA/full-FT flags, Qwen3 FSDP layer, and output protection.
- [ ] **Step 2: Run tests** and confirm they fail because the launchers do not exist.

### Task 2: Implement LoRA launcher

**Files:**
- Create: `launch_qwen3_32b_gpt5_lora_auto_queue.sh`

- [ ] **Step 1: Implement** single-GPU idle scheduling, `MODEL_DIR`/`MODEL_TAG` overrides, dataset resolution, output guards, and the established LoRA command.
- [ ] **Step 2: Run launcher syntax and contract tests.**

### Task 3: Implement Full-FT launcher

**Files:**
- Create: `launch_qwen3_32b_gpt5_fullft_fsdp.sh`

- [ ] **Step 1: Implement** four-GPU idle scheduling, FSDP torchrun invocation, `Qwen3DecoderLayer`, model overrides, and output guards.
- [ ] **Step 2: Run syntax, contract, and diff checks.**

### Task 4: Document usage

**Files:**
- Modify: `docs/LAUNCHER_RUNBOOK.md`

- [ ] **Step 1: Add commands** for LoRA, Full-FT, model override, tmux, logs, skip jobs, and result inspection.
- [ ] **Step 2: Run the focused test suite and commit the launcher change.**
