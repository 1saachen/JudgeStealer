# Qwen3-8B Local Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Qwen3-8B GPT4All launcher write outputs, logs, and checkpoints to local storage by default and reject NFS output paths.

**Architecture:** Keep code, model, and dataset paths rooted in the repository. Introduce an overridable absolute `OUTPUT_ROOT`, derive both experiment and log directories from it, and perform a filesystem preflight before launching Python.

**Tech Stack:** Bash, `findmnt`, `df`, Python/pytest static launcher tests

---

### Task 1: Specify local output behavior with failing tests

**Files:**
- Modify: `tests/test_qwen3_8b_gpt4all_launcher.py`

- [ ] **Step 1: Add tests for the default path and NFS guard**

```python
def test_launcher_defaults_outputs_and_logs_to_local_storage():
    text = launcher_text()
    assert (
        'DEFAULT_OUTPUT_ROOT="/opt/dlami/nvme/cyl/autodl-tmp/'
        'JudgeStealer_outputs"' in text
    )
    assert 'OUTPUT_ROOT="${OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}"' in text
    assert 'OUT="$OUTPUT_ROOT/$NAME"' in text
    assert 'LOG_ROOT="$OUTPUT_ROOT/qwen3_8b_gpt4all_gpt5_four_stage_logs"' in text


def test_launcher_reports_storage_and_rejects_nfs_outputs():
    text = launcher_text()
    assert 'findmnt -n -o FSTYPE -T "$OUTPUT_ROOT"' in text
    assert 'df -hP "$OUTPUT_ROOT"' in text
    assert 'nfs|nfs4)' in text
    assert 'ERROR network filesystem output is not allowed' in text
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
python -m pytest -q tests/test_qwen3_8b_gpt4all_launcher.py
```

Expected: the two new tests fail because the launcher still derives `OUT` and `LOG_ROOT` from `$ROOT/outputs` and has no filesystem preflight.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/test_qwen3_8b_gpt4all_launcher.py
git commit -m "Test local output guard for Qwen3-8B launcher"
```

### Task 2: Route writes to local storage

**Files:**
- Modify: `launch_qwen3_8b_gpt4all_gpt5_four_stage.sh`
- Test: `tests/test_qwen3_8b_gpt4all_launcher.py`

- [ ] **Step 1: Derive all writable paths from `OUTPUT_ROOT`**

Replace the existing `OUT` and `LOG_ROOT` definitions with:

```bash
DEFAULT_OUTPUT_ROOT="/opt/dlami/nvme/cyl/autodl-tmp/JudgeStealer_outputs"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}"
OUT="$OUTPUT_ROOT/$NAME"
LOG_ROOT="$OUTPUT_ROOT/qwen3_8b_gpt4all_gpt5_four_stage_logs"
STATUS_LOG="$LOG_ROOT/job_status.log"
LOG="$LOG_ROOT/$NAME.log"
```

- [ ] **Step 2: Add the filesystem preflight after `log_status`**

```bash
check_output_storage() {
  local fs_type=""
  local available="unknown"

  if command -v findmnt >/dev/null 2>&1; then
    fs_type="$(findmnt -n -o FSTYPE -T "$OUTPUT_ROOT" 2>/dev/null || true)"
  else
    log_status "WARNING findmnt unavailable; output filesystem type not checked"
  fi
  available="$(df -hP "$OUTPUT_ROOT" 2>/dev/null | awk 'NR == 2 {print $4}' || true)"
  [[ -n "$available" ]] || available="unknown"
  [[ -n "$fs_type" ]] || fs_type="unknown"

  log_status "STORAGE output_root=$OUTPUT_ROOT fstype=$fs_type available=$available"
  case "$fs_type" in
    nfs|nfs4)
      log_status "ERROR network filesystem output is not allowed output_root=$OUTPUT_ROOT fstype=$fs_type"
      exit 1
      ;;
  esac
}

check_output_storage
```

Keep `mkdir -p "$LOG_ROOT"` before this function is called so the default directory exists before `findmnt` and `df` inspect it.

- [ ] **Step 3: Run the launcher tests and verify GREEN**

Run:

```bash
python -m pytest -q tests/test_qwen3_8b_gpt4all_launcher.py
```

Expected: `6 passed`.

- [ ] **Step 4: Verify Bash syntax**

Run:

```bash
bash -n launch_qwen3_8b_gpt4all_gpt5_four_stage.sh
```

Expected: exit status 0 with no output.

- [ ] **Step 5: Inspect the focused diff**

Run:

```bash
git diff --check
git diff -- launch_qwen3_8b_gpt4all_gpt5_four_stage.sh tests/test_qwen3_8b_gpt4all_launcher.py
```

Expected: only local-output routing, storage preflight, and their tests are changed.

- [ ] **Step 6: Commit the implementation**

```bash
git add launch_qwen3_8b_gpt4all_gpt5_four_stage.sh tests/test_qwen3_8b_gpt4all_launcher.py
git commit -m "Write Qwen3-8B experiment outputs to local storage"
```

### Task 3: Final verification and publication

**Files:**
- Verify: `launch_qwen3_8b_gpt4all_gpt5_four_stage.sh`
- Verify: `tests/test_qwen3_8b_gpt4all_launcher.py`

- [ ] **Step 1: Run focused regression tests**

```bash
python -m pytest -q tests/test_qwen3_8b_gpt4all_launcher.py tests/test_skywork_dataset.py
```

Expected: all focused tests pass.

- [ ] **Step 2: Confirm unrelated user files remain untouched**

```bash
git status --short
```

Expected: only the pre-existing untracked `PROJECT_MEMORY.md`, `WORK_LOG.md`, and `launch_qwen3_gpt5_selector_smooth_lora_table_20260814.sh` remain.

- [ ] **Step 3: Push the commits**

```bash
git push origin main
```

Expected: `main` is updated on GitHub without modifying the three unrelated untracked files.
