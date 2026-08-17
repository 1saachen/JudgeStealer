# Qwen3-8B GPT4All 四阶段启动脚本实现计划

> **供智能体执行：** 必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`，逐项执行本计划。所有步骤使用复选框跟踪。

**目标：** 新增一个只运行 Qwen3-8B、GPT4All GPT-5 四阶段实验的可靠启动脚本。

**架构：** 根目录 Bash 脚本负责路径解析、启动前检查、重复运行保护、日志记录和训练命令组装；一个聚焦的 Python 静态测试负责锁定模型、数据、四阶段回放、selector 和平滑配置。现有多任务脚本和 Python 训练代码保持不变。

**技术栈：** Bash、Python 标准库、pytest、现有四阶段训练入口。

---

### 任务 1：用失败测试锁定单用途启动配置

**文件：**
- 新建：`tests/test_qwen3_8b_gpt4all_launcher.py`
- 待新建：`launch_qwen3_8b_gpt4all_gpt5_four_stage.sh`

- [ ] **步骤 1：编写失败测试**

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "launch_qwen3_8b_gpt4all_gpt5_four_stage.sh"


def launcher_text() -> str:
    return LAUNCHER.read_text(encoding="utf-8")


def test_launcher_uses_portable_qwen3_8b_and_gpt4all_paths():
    text = launcher_text()
    assert 'MODEL="$ROOT/models/Qwen3-8B"' in text
    assert 'TRAIN_DATA="$ROOT/data/gpt4all/gpt5/train9k_pointwise_pairwise_no_val_overlap.json"' in text
    assert 'EVAL_DATA="$ROOT/data/gpt4all/gpt5/val3k_pairwise_listwise.json"' in text
    assert "$ROOT/qwen/" not in text
    assert "$ROOT/train_with_selector/train_with_selector/data/Dolly/" not in text


def test_launcher_enables_exact_four_stage_review_configuration():
    text = launcher_text()
    required = [
        "--budget-units 600",
        "--stage2-pointwise-replay-ratio 0",
        "--stage3-pointwise-replay-ratio 0",
        "--stage3-pairwise-replay-ratio 0",
        "--stage4-replay-strategy stratified_triple",
        "--stage4-replay-fraction 1",
        "--stage4-epochs 1",
        "--eval-stages final",
        "--use-lora",
        "--load-in-4bit",
    ]
    for argument in required:
        assert argument in text


def test_launcher_preserves_selector_and_smoothing_configuration():
    text = launcher_text()
    required = [
        "--pointwise-global-smooth-alpha 0.1",
        "--pointwise-global-smooth-mode local_gaussian",
        "--pointwise-global-smooth-gaussian-sigma 1.0",
        "--pointwise-global-smooth-stages all",
        "--candidate-selector-kind bias_trap_pointwise",
        "--candidate-selector-proxy-mode lm_head",
        "--reuse-selection-proxy-for-stage1",
        "--candidate-selector-init-triples 80",
        "--candidate-selector-batch-size 20",
        "--candidate-selector-max-score-candidates 100",
        "--candidate-selector-exploration-ratio 0",
        "--candidate-selector-diversity-weight 1",
        "--candidate-selector-uncertainty-weight 0.25",
        "--candidate-selector-bias-weight 1",
        "--candidate-selector-embedding-model BAAI/bge-small-en-v1.5",
    ]
    for argument in required:
        assert argument in text


def test_launcher_has_preflight_and_duplicate_run_guards():
    text = launcher_text()
    assert 'require_file "$SCRIPT"' in text
    assert 'require_dir "$MODEL"' in text
    assert 'require_file "$MODEL/config.json"' in text
    assert 'require_file "$TRAIN_DATA"' in text
    assert 'require_file "$EVAL_DATA"' in text
    assert 'if [[ -f "$OUT/metrics_compact.json" ]]' in text
    assert 'if [[ -e "$OUT" ]]' in text
    assert 'grep -F -- "--out $OUT"' in text
```

- [ ] **步骤 2：运行测试，确认因脚本尚不存在而失败**

运行：

```bash
python -m pytest -q tests/test_qwen3_8b_gpt4all_launcher.py
```

预期：4 个测试均因 `launch_qwen3_8b_gpt4all_gpt5_four_stage.sh` 不存在而失败。

### 任务 2：创建单用途启动脚本

**文件：**
- 新建：`launch_qwen3_8b_gpt4all_gpt5_four_stage.sh`
- 测试：`tests/test_qwen3_8b_gpt4all_launcher.py`

- [ ] **步骤 1：实现最小启动脚本**

脚本必须包含：

```bash
#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON_BIN:-python}"
SCRIPT="$ROOT/run_pointwise5answers_three_stage_pairwise_listwise_sft_v1.py"
MODEL="$ROOT/models/Qwen3-8B"
TRAIN_DATA="$ROOT/data/gpt4all/gpt5/train9k_pointwise_pairwise_no_val_overlap.json"
EVAL_DATA="$ROOT/data/gpt4all/gpt5/val3k_pairwise_listwise.json"
NAME="qwen3_8b_gpt4all_gpt5_b600_selector_smooth_a010_pool100_stage4stratfull"
OUT="$ROOT/outputs/$NAME"
LOG_ROOT="$ROOT/outputs/qwen3_8b_gpt4all_gpt5_four_stage_logs"
STATUS_LOG="$LOG_ROOT/job_status.log"
LOG="$LOG_ROOT/$NAME.log"

GPU_ID="${1:?usage: $0 <gpu_id>}"
if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 <gpu_id>" >&2
  exit 2
fi

mkdir -p "$LOG_ROOT"
cd "$ROOT"

log_status() {
  { flock 9; echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "$STATUS_LOG"; } \
    9>"$LOG_ROOT/.status.lock"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    log_status "ERROR missing file=$1"
    exit 1
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    log_status "ERROR missing directory=$1"
    exit 1
  fi
}

if ! command -v "$PY" >/dev/null 2>&1; then
  log_status "ERROR python not found=$PY"
  exit 1
fi
require_file "$SCRIPT"
require_dir "$MODEL"
require_file "$MODEL/config.json"
require_file "$TRAIN_DATA"
require_file "$EVAL_DATA"

if [[ -f "$OUT/metrics_compact.json" ]]; then
  log_status "SKIP completed out=$OUT"
  exit 0
fi
if ps -eo args --cols 4096 | grep -F -- "--out $OUT" | grep -v grep >/dev/null 2>&1; then
  log_status "SKIP running out=$OUT"
  exit 0
fi
if [[ -e "$OUT" ]]; then
  log_status "ERROR incomplete output exists out=$OUT"
  exit 1
fi

log_status "START gpu=$GPU_ID out=$OUT"
CUDA_VISIBLE_DEVICES="$GPU_ID" \
PYTHONUNBUFFERED=1 \
TOKENIZERS_PARALLELISM=false \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" -u "$SCRIPT" \
    --pointwise-5answers-dataset "$TRAIN_DATA" \
    --listwise-eval-dataset "$EVAL_DATA" \
    --llama "$MODEL" \
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
    --learning-rate 1e-4 \
    --max-length 4096 \
    --eval-batch-size 1 \
    --eval-stages final \
    --use-lora \
    --load-in-4bit \
    --pointwise-global-smooth-alpha 0.1 \
    --pointwise-global-smooth-mode local_gaussian \
    --pointwise-global-smooth-gaussian-sigma 1.0 \
    --pointwise-global-smooth-stages all \
    --train-selection-mode candidate_triple_selector \
    --candidate-selector-kind bias_trap_pointwise \
    --candidate-selector-target-task pointwise \
    --candidate-selector-proxy-mode lm_head \
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
    --proxy-lr 1e-4 \
    --proxy-max-length 768 \
    --out "$OUT" >"$LOG" 2>&1
rc=$?

if [[ "$rc" -eq 0 && -f "$OUT/metrics_compact.json" ]]; then
  log_status "DONE gpu=$GPU_ID out=$OUT"
  exit 0
fi

log_status "ERROR gpu=$GPU_ID rc=$rc out=$OUT log=$LOG"
exit "$rc"
```

- [ ] **步骤 2：运行聚焦测试，确认全部通过**

运行：

```bash
python -m pytest -q tests/test_qwen3_8b_gpt4all_launcher.py
```

预期：`4 passed`。

- [ ] **步骤 3：在 Bash 可用环境执行语法检查**

运行：

```bash
bash -n launch_qwen3_8b_gpt4all_gpt5_four_stage.sh
```

预期：退出码 0，无输出。如果当前 Windows 主机没有可用 Bash，则记录限制，并在 Linux 服务器拉取后执行同一命令。

### 任务 3：复核、提交并推送

**文件：**
- 新建：`launch_qwen3_8b_gpt4all_gpt5_four_stage.sh`
- 新建：`tests/test_qwen3_8b_gpt4all_launcher.py`
- 已有：`docs/superpowers/specs/2026-08-17-qwen3-8b-gpt4all-launcher-design.md`
- 新建：`docs/superpowers/plans/2026-08-17-qwen3-8b-gpt4all-launcher.md`

- [ ] **步骤 1：检查差异和未跟踪文件，确保不提交用户的无关文件**

运行：

```bash
git status --short
git diff -- launch_qwen3_8b_gpt4all_gpt5_four_stage.sh tests/test_qwen3_8b_gpt4all_launcher.py
```

预期：只处理计划、单用途脚本及其测试；根目录 `PROJECT_MEMORY.md`、`WORK_LOG.md` 和原多任务脚本保持未跟踪状态。

- [ ] **步骤 2：提交实现**

```bash
git add docs/superpowers/plans/2026-08-17-qwen3-8b-gpt4all-launcher.md \
  launch_qwen3_8b_gpt4all_gpt5_four_stage.sh \
  tests/test_qwen3_8b_gpt4all_launcher.py
git commit -m "Add Qwen3-8B GPT4All four-stage launcher"
```

- [ ] **步骤 3：推送并核对远程提交**

```bash
git push origin main
git rev-parse HEAD
git ls-remote --heads origin main
```

预期：本地 `HEAD` 与远程 `main` 哈希完全一致。

