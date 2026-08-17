# Qwen3 GPT-5 Full-FT Auto Queue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add end-to-end Full-FT candidate selection and a dynamic single-GPU scheduler for the eight Qwen3 × GPT-5 dataset experiments.

**Architecture:** Preserve legacy LoRA behavior by adding an explicit selector finetune-mode option whose default is `lora`, then forward `full` into the existing proxy implementation. Add one Bash launcher that owns the eight-job matrix, detects idle GPUs only inside a user-provided allowlist, and runs one independent Full-FT job per idle GPU.

**Tech Stack:** Python 3.10, PyTorch/Transformers, pytest, Bash 4+, `nvidia-smi`, `flock`.

---

### Task 1: Expose Full-FT selector mode

**Files:**
- Create: `tests/test_fullft_selector_mode.py`
- Modify: `run_pointwise5answers_three_stage_pairwise_listwise_sft_v1.py:90-95`
- Modify: `run_pointwise5answers_three_stage_pairwise_listwise_sft_v1.py:1995-2010`
- Modify: `run_pointwise5answers_three_stage_pairwise_listwise_sft_v1.py:2208-2220`
- Modify: `run_pointwise5answers_three_stage_pairwise_listwise_sft_v1.py:2355-2370`
- Modify: `run_pointwise5answers_three_to_listwise_v1.py:1847-1978`

- [ ] **Step 1: Write failing selector-mode tests**

Create `tests/test_fullft_selector_mode.py`:

```python
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

import run_pointwise5answers_three_to_listwise_v1 as selector


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "run_pointwise5answers_three_stage_pairwise_listwise_sft_v1.py"


def test_legacy_selector_mode_defaults_to_lora():
    cfg = SimpleNamespace(load_in_4bit=False)
    assert selector._resolve_candidate_selector_finetune_mode(cfg) == "lora"


def test_full_selector_mode_is_supported_without_quantization():
    cfg = SimpleNamespace(
        candidate_selector_finetune_mode="full",
        load_in_4bit=False,
    )
    assert selector._resolve_candidate_selector_finetune_mode(cfg) == "full"


def test_full_selector_mode_rejects_4bit():
    cfg = SimpleNamespace(
        candidate_selector_finetune_mode="full",
        load_in_4bit=True,
    )
    with pytest.raises(ValueError, match="full.*4-bit"):
        selector._resolve_candidate_selector_finetune_mode(cfg)


def test_candidate_selector_forwards_resolved_finetune_mode():
    source = inspect.getsource(selector._select_candidate_triples_with_selector)
    assert "finetune_mode=_resolve_candidate_selector_finetune_mode(cfg)" in source


def test_main_cli_exposes_and_records_selector_finetune_mode():
    source = MAIN.read_text(encoding="utf-8")
    assert '"--candidate-selector-finetune-mode"' in source
    assert 'choices=["lora", "full"]' in source
    assert 'default="lora"' in source
    assert (
        "candidate_selector_finetune_mode="
        "str(args.candidate_selector_finetune_mode)" in source
    )
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
python -m pytest -q tests/test_fullft_selector_mode.py
```

Expected: failures report that `_resolve_candidate_selector_finetune_mode` and the new CLI parameter do not exist.

- [ ] **Step 3: Add the resolver and forward it into the proxy**

Add this helper immediately before `_select_candidate_triples_with_selector` in
`run_pointwise5answers_three_to_listwise_v1.py`:

```python
def _resolve_candidate_selector_finetune_mode(cfg: Any) -> str:
    mode = str(getattr(cfg, "candidate_selector_finetune_mode", "lora"))
    if mode not in {"lora", "full"}:
        raise ValueError(
            "candidate selector finetune mode must be one of {'lora','full'}"
        )
    if mode == "full" and bool(getattr(cfg, "load_in_4bit", False)):
        raise ValueError(
            "candidate selector full finetuning is incompatible with 4-bit loading"
        )
    return mode
```

Replace the hard-coded constructor argument:

```python
finetune_mode=_resolve_candidate_selector_finetune_mode(cfg),
```

Record the resolved mode in selector statistics as:

```python
"proxy_finetune_mode": _resolve_candidate_selector_finetune_mode(cfg),
```

- [ ] **Step 4: Add the main-entry CLI and RunConfig field**

Add `candidate_selector_finetune_mode: str` beside
`candidate_selector_proxy_mode` in `RunConfig`. Add the parser argument:

```python
parser.add_argument(
    "--candidate-selector-finetune-mode",
    choices=["lora", "full"],
    default="lora",
    help="Train the active-selection proxy with LoRA adapters or all model parameters.",
)
```

Populate `RunConfig` with:

```python
candidate_selector_finetune_mode=str(args.candidate_selector_finetune_mode),
```

Before selector-specific reuse validation, reject the incompatible combination:

```python
if (
    str(cfg.candidate_selector_finetune_mode) == "full"
    and bool(cfg.load_in_4bit)
):
    raise ValueError(
        "--candidate-selector-finetune-mode full cannot be combined with --load-in-4bit"
    )
```

- [ ] **Step 5: Run selector tests and existing focused tests**

Run:

```bash
python -m pytest -q \
  tests/test_fullft_selector_mode.py \
  tests/test_rewardmodel_pointwise.py \
  tests/test_skywork_dataset.py
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit selector support**

```bash
git add \
  tests/test_fullft_selector_mode.py \
  run_pointwise5answers_three_stage_pairwise_listwise_sft_v1.py \
  run_pointwise5answers_three_to_listwise_v1.py
git commit -m "Add full finetuning selector mode"
```

### Task 2: Specify the eight-job Full-FT launcher

**Files:**
- Create: `tests/test_qwen3_gpt5_fullft_queue.py`
- Create: `launch_qwen3_gpt5_fullft_auto_queue.sh`

- [ ] **Step 1: Write failing launcher tests**

Create `tests/test_qwen3_gpt5_fullft_queue.py`:

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "launch_qwen3_gpt5_fullft_auto_queue.sh"


def launcher_text() -> str:
    return LAUNCHER.read_text(encoding="utf-8")


def test_queue_contains_all_eight_model_dataset_jobs():
    text = launcher_text()
    for size in ("0p6b", "1p7b", "4b", "8b"):
        assert f"alpaca_{size}" in text
        assert f"gpt4all_{size}" in text
    for model_dir in (
        "Qwen3-0.6B",
        "Qwen3-1.7B",
        "Qwen3-4B",
        "Qwen3-8B",
    ):
        assert f'$ROOT/models/{model_dir}' in text


def test_queue_uses_current_gpt5_data_paths():
    text = launcher_text()
    assert '$ROOT/data/alpaca/gpt5/train-20k.json' in text
    assert '$ROOT/data/alpaca/gpt5/val-2k-eval-listwise.json' in text
    assert (
        '$ROOT/data/gpt4all/gpt5/'
        'train9k_pointwise_pairwise_no_val_overlap.json' in text
    )
    assert '$ROOT/data/gpt4all/gpt5/val3k_pairwise_listwise.json' in text


def test_queue_writes_outputs_and_logs_to_overridable_nvme_storage():
    text = launcher_text()
    assert (
        'DEFAULT_OUTPUT_ROOT="/opt/dlami/nvme/cyl/autodl-tmp/'
        'JudgeStealer_outputs"' in text
    )
    assert 'OUTPUT_ROOT="${OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}"' in text
    assert 'out="$OUTPUT_ROOT/$name"' in text
    assert 'LOG_ROOT="$OUTPUT_ROOT/qwen3_gpt5_fullft_auto_queue_logs"' in text
    assert "check_output_storage" in text
    assert 'findmnt -n -o FSTYPE -T "$OUTPUT_ROOT"' in text
    assert "nfs|nfs4" in text


def test_queue_runs_fullft_selector_with_exact_training_configuration():
    text = launcher_text()
    required = [
        "--candidate-selector-finetune-mode full",
        "--candidate-selector-proxy-mode lm_head",
        "--reuse-selection-proxy-for-stage1",
        "--proxy-lr 1e-5",
        "--learning-rate 1e-5",
        "--budget-units 600",
        "--candidate-selector-init-triples 80",
        "--candidate-selector-batch-size 20",
        "--candidate-selector-max-score-candidates 100",
        "--stage4-replay-strategy stratified_triple",
        "--stage4-replay-fraction 1",
        "--stage4-epochs 1",
        "--max-length 4096",
        "--per-device-batch-size 1",
        "--gradient-accumulation-steps 16",
        "--eval-stages final",
    ]
    for argument in required:
        assert argument in text
    for forbidden in (
        "--use-lora",
        "--load-in-4bit",
        "--fixed-selected-triples-path",
        "torchrun",
        "--fsdp",
    ):
        assert forbidden not in text


def test_queue_preserves_smoothing_and_bias_trap_selector():
    text = launcher_text()
    for argument in (
        "--pointwise-global-smooth-alpha 0.1",
        "--pointwise-global-smooth-mode local_gaussian",
        "--pointwise-global-smooth-gaussian-sigma 1.0",
        "--pointwise-global-smooth-stages all",
        "--candidate-selector-kind bias_trap_pointwise",
        "--candidate-selector-diversity-weight 1",
        "--candidate-selector-uncertainty-weight 0.25",
        "--candidate-selector-bias-weight 1",
        "--candidate-selector-exploration-ratio 0",
    ):
        assert argument in text


def test_queue_requires_allowlist_and_checks_real_gpu_idleness():
    text = launcher_text()
    assert 'usage: $0 <gpu_id> [gpu_id ...]' in text
    assert "GPU_MEMORY_USED_LIMIT_MB=1024" in text
    assert "--query-compute-apps=gpu_uuid" in text
    assert "--query-gpu=memory.used" in text
    assert 'sleep "$POLL_SECONDS"' in text
    assert "declare -A GPU_WORKER_PIDS" in text


def test_queue_protects_existing_outputs_and_continues_failures():
    text = launcher_text()
    assert 'if [[ -f "$out/metrics_compact.json" ]]' in text
    assert 'if [[ -e "$out" ]]' in text
    assert 'grep -F -- "--out $out"' in text
    assert "overall_rc=1" in text
    assert "job_status.log" in text
```

- [ ] **Step 2: Run launcher tests and verify RED**

Run:

```bash
python -m pytest -q tests/test_qwen3_gpt5_fullft_queue.py
```

Expected: all tests fail because `launch_qwen3_gpt5_fullft_auto_queue.sh` does not exist.

- [ ] **Step 3: Create the launcher foundation and GPU checks**

Create `launch_qwen3_gpt5_fullft_auto_queue.sh` with `set -uo pipefail`, repository-relative
paths, the allowlist usage check, and these scheduler constants:

```bash
DEFAULT_OUTPUT_ROOT="/opt/dlami/nvme/cyl/autodl-tmp/JudgeStealer_outputs"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}"
LOG_ROOT="$OUTPUT_ROOT/qwen3_gpt5_fullft_auto_queue_logs"
POLL_SECONDS="${POLL_SECONDS:-30}"
GPU_MEMORY_USED_LIMIT_MB=1024
JOBS=(
  alpaca_8b gpt4all_8b
  alpaca_4b gpt4all_4b
  alpaca_1p7b gpt4all_1p7b
  alpaca_0p6b gpt4all_0p6b
)
declare -A GPU_WORKER_PIDS=()
declare -A GPU_WORKER_JOBS=()
```

Implement GPU UUID, memory, and compute-process checks using:

```bash
gpu_uuid() {
  nvidia-smi -i "$1" --query-gpu=uuid --format=csv,noheader,nounits 2>/dev/null \
    | awk '{$1=$1; print}'
}

gpu_memory_used_mb() {
  nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk '{$1=$1; print}'
}

gpu_has_compute_process() {
  local uuid="$1"
  nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits 2>/dev/null \
    | awk '{$1=$1; print}' \
    | grep -Fx -- "$uuid" >/dev/null 2>&1
}

gpu_is_idle() {
  local gpu="$1" uuid used
  uuid="$(gpu_uuid "$gpu")" || return 1
  used="$(gpu_memory_used_mb "$gpu")" || return 1
  [[ -n "$uuid" && "$used" =~ ^[0-9]+$ ]] || return 1
  (( used <= GPU_MEMORY_USED_LIMIT_MB )) || return 1
  ! gpu_has_compute_process "$uuid"
}
```

- [ ] **Step 4: Implement job resolution and the exact training command**

Implement job resolution with these exact mappings:

```bash
resolve_job() {
  local job="$1"
  dataset="${job%%_*}"
  case "$job" in
    alpaca_0p6b|gpt4all_0p6b)
      model_tag=qwen3_0p6b; model="$ROOT/models/Qwen3-0.6B" ;;
    alpaca_1p7b|gpt4all_1p7b)
      model_tag=qwen3_1p7b; model="$ROOT/models/Qwen3-1.7B" ;;
    alpaca_4b|gpt4all_4b)
      model_tag=qwen3_4b; model="$ROOT/models/Qwen3-4B" ;;
    alpaca_8b|gpt4all_8b)
      model_tag=qwen3_8b; model="$ROOT/models/Qwen3-8B" ;;
    *) return 2 ;;
  esac
  if [[ "$dataset" == "alpaca" ]]; then
    train="$ROOT/data/alpaca/gpt5/train-20k.json"
    eval="$ROOT/data/alpaca/gpt5/val-2k-eval-listwise.json"
  else
    train="$ROOT/data/gpt4all/gpt5/train9k_pointwise_pairwise_no_val_overlap.json"
    eval="$ROOT/data/gpt4all/gpt5/val3k_pairwise_listwise.json"
  fi
  name="${model_tag}_${dataset}_gpt5_b600_fullft_selector_smooth_a010_pool100_stage4stratfull"
  out="$OUTPUT_ROOT/$name"
  log="$LOG_ROOT/$name.log"
}
```

After the standard file, completed-output, duplicate-process and incomplete-output guards, use
this exact Python argument list inside `run_job`:

```bash
CUDA_VISIBLE_DEVICES="$gpu" \
PYTHONUNBUFFERED=1 \
TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" -u "$SCRIPT" \
    --pointwise-5answers-dataset "$train" \
    --listwise-eval-dataset "$eval" \
    --llama "$model" \
    --seed 42 \
    --budget-units 600 \
    --stage2-pointwise-replay-ratio 0 \
    --stage3-pointwise-replay-ratio 0 \
    --stage3-pairwise-replay-ratio 0 \
    --stage4-replay-strategy stratified_triple \
    --stage4-replay-fraction 1 \
    --stage4-epochs 1 \
    --pointwise-epochs 1 \
    --pairwise-epochs 1 \
    --listwise-epochs 1 \
    --per-device-batch-size 1 \
    --gradient-accumulation-steps 16 \
    --learning-rate 1e-5 \
    --max-length 4096 \
    --eval-batch-size 1 \
    --eval-stages final \
    --pointwise-global-smooth-alpha 0.1 \
    --pointwise-global-smooth-mode local_gaussian \
    --pointwise-global-smooth-gaussian-sigma 1.0 \
    --pointwise-global-smooth-stages all \
    --train-selection-mode candidate_triple_selector \
    --candidate-selector-kind bias_trap_pointwise \
    --candidate-selector-target-task pointwise \
    --candidate-selector-proxy-mode lm_head \
    --candidate-selector-finetune-mode full \
    --reuse-selection-proxy-for-stage1 \
    --candidate-selector-init-triples 80 \
    --candidate-selector-batch-size 20 \
    --candidate-selector-max-score-candidates 100 \
    --candidate-selector-one-per-question \
    --candidate-selector-proxy-warmup-epochs 3 \
    --candidate-selector-proxy-update-epochs 1 \
    --candidate-selector-exploration-ratio 0 \
    --candidate-selector-diversity-weight 1 \
    --candidate-selector-uncertainty-weight 0.25 \
    --candidate-selector-bias-weight 1 \
    --candidate-selector-pointwise-length-bias-weight 0.5 \
    --candidate-selector-pairwise-position-bias-weight 0.5 \
    --candidate-selector-pairwise-position-pairs 1 \
    --candidate-selector-pairwise-position-bias-scale 0.02 \
    --candidate-selector-signal-normalization none \
    --candidate-selector-uncertainty-view pointwise \
    --candidate-selector-density-k 10 \
    --candidate-selector-embedding-model BAAI/bge-small-en-v1.5 \
    --candidate-selector-embedding-max-length 512 \
    --candidate-selector-embedding-batch-size 64 \
    --candidate-selector-embedding-pooling cls \
    --candidate-selector-diversity-view pointwise \
    --proxy-lr 1e-5 \
    --proxy-max-length 768 \
    --out "$out" >"$log" 2>&1
```

- [ ] **Step 5: Implement dynamic dispatch and failure collection**

Use this dispatch loop after defining `run_job`:

```bash
next_job_index=0
overall_rc=0
while (( next_job_index < ${#JOBS[@]} || ${#GPU_WORKER_PIDS[@]} > 0 )); do
  for gpu in "${!GPU_WORKER_PIDS[@]}"; do
    pid="${GPU_WORKER_PIDS[$gpu]}"
    if ! kill -0 "$pid" 2>/dev/null; then
      if wait "$pid"; then
        rc=0
      else
        rc=$?
        overall_rc=1
      fi
      log_status "WORKER_EXIT job=${GPU_WORKER_JOBS[$gpu]} gpu=$gpu rc=$rc"
      unset 'GPU_WORKER_PIDS[$gpu]'
      unset 'GPU_WORKER_JOBS[$gpu]'
    fi
  done

  for gpu in "${ALLOWED_GPUS[@]}"; do
    (( next_job_index < ${#JOBS[@]} )) || break
    [[ -z "${GPU_WORKER_PIDS[$gpu]+assigned}" ]] || continue
    gpu_is_idle "$gpu" || continue
    job="${JOBS[$next_job_index]}"
    gpu_is_idle "$gpu" || continue
    run_job "$job" "$gpu" &
    GPU_WORKER_PIDS[$gpu]=$!
    GPU_WORKER_JOBS[$gpu]="$job"
    next_job_index=$((next_job_index + 1))
  done

  if (( next_job_index < ${#JOBS[@]} || ${#GPU_WORKER_PIDS[@]} > 0 )); then
    sleep "$POLL_SECONDS"
  fi
done
exit "$overall_rc"
```

Use a `log_status` implementation guarded by `flock`:

```bash
log_status() {
  { flock 9; echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "$STATUS_LOG"; } \
    9>"$LOG_ROOT/.status.lock"
}
```

- [ ] **Step 6: Run launcher tests and Bash syntax validation**

Run:

```bash
python -m pytest -q tests/test_qwen3_gpt5_fullft_queue.py
bash -n launch_qwen3_gpt5_fullft_auto_queue.sh
```

Expected: all launcher tests pass and Bash exits with status 0 without output.

- [ ] **Step 7: Mark the launcher executable and commit**

```bash
git update-index --add --chmod=+x launch_qwen3_gpt5_fullft_auto_queue.sh
git add tests/test_qwen3_gpt5_fullft_queue.py
git commit -m "Add Qwen3 Full-FT automatic GPU queue"
```

### Task 3: Final verification and operator handoff

**Files:**
- Verify: `run_pointwise5answers_three_stage_pairwise_listwise_sft_v1.py`
- Verify: `run_pointwise5answers_three_to_listwise_v1.py`
- Verify: `launch_qwen3_gpt5_fullft_auto_queue.sh`
- Verify: `tests/test_fullft_selector_mode.py`
- Verify: `tests/test_qwen3_gpt5_fullft_queue.py`

- [ ] **Step 1: Run the full focused test set**

```bash
python -m pytest -q \
  tests/test_fullft_selector_mode.py \
  tests/test_qwen3_gpt5_fullft_queue.py \
  tests/test_qwen3_8b_gpt4all_launcher.py \
  tests/test_rewardmodel_pointwise.py \
  tests/test_skywork_dataset.py
```

Expected: all tests pass.

- [ ] **Step 2: Run syntax checks**

```bash
python -m py_compile \
  run_pointwise5answers_three_stage_pairwise_listwise_sft_v1.py \
  run_pointwise5answers_three_to_listwise_v1.py
bash -n launch_qwen3_gpt5_fullft_auto_queue.sh
```

Expected: all commands exit with status 0 without output.

- [ ] **Step 3: Check the final diff and repository state**

```bash
git diff --check
git status --short
git log -3 --oneline
```

Expected: no whitespace errors; only the user's pre-existing untracked files remain; the two
implementation commits appear after the design and plan commits.

- [ ] **Step 4: Provide the server launch command**

Use a tmux session and pass only GPUs the user permits the scheduler to occupy:

```bash
tmux new -s qwen3_fullft
conda activate cyl
cd /data/model-extraction-attack/yaolin/JudgeStealer
./launch_qwen3_gpt5_fullft_auto_queue.sh 1 2 3 5
```

Detach with `Ctrl-b d`. Follow queue state with:

```bash
tail -f /opt/dlami/nvme/cyl/autodl-tmp/JudgeStealer_outputs/\
qwen3_gpt5_fullft_auto_queue_logs/job_status.log
```
