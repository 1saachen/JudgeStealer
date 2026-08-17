# Qwen3-1.7B LoRA 消融去重队列实现计划

> **供执行代理使用：** 必须使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`，按任务逐项执行本计划，并用复选框跟踪进度。

**目标：** 增加 percentage budget 解析能力，并建立不重复运行 Ours 的 36-job Qwen3-1.7B LoRA 自动消融队列。

**架构：** 将 percentage budget 数学逻辑放入独立的轻量 Python 模块，由三阶段主程序在数据划分后解析并记录。新增单机多 GPU 自动队列，沿用现有 GPU 空闲检测、本地输出和失败保护模式，通过 job case 只覆盖对应消融参数。

**技术栈：** Python 3.10、pytest、Bash、`nvidia-smi`、`findmnt`、Hugging Face Trainer

---

### 任务 1：实现 percentage budget 解析

**文件：**
- 新建：`ablation_budget.py`
- 新建：`tests/test_ablation_budget.py`
- 修改：`run_pointwise5answers_three_stage_pairwise_listwise_sft_v1.py`

- [ ] **步骤 1：先写预算数学的失败测试**

```python
import pytest

from ablation_budget import resolve_percentage_budget


@pytest.mark.parametrize(
    ("candidate_queries", "percent", "query_budget"),
    [
        (18_000, 0.5, 90),
        (18_000, 1.0, 180),
        (18_000, 2.0, 360),
        (18_000, 5.0, 900),
        (18_000, 10.0, 1_800),
        (8_120, 0.5, 40),
        (8_120, 1.0, 80),
        (8_120, 2.0, 160),
        (8_120, 5.0, 410),
        (8_120, 10.0, 810),
    ],
)
def test_resolve_percentage_budget(candidate_queries, percent, query_budget):
    resolved = resolve_percentage_budget(candidate_queries, percent)
    assert resolved.query_budget == query_budget
    assert resolved.budget_units == query_budget * 3
    assert resolved.init_triples == query_budget * 4 // 10
    assert resolved.selection_batch_size == query_budget // 10
    assert resolved.max_score_candidates == query_budget // 2


@pytest.mark.parametrize(
    ("candidate_queries", "percent"),
    [(0, 1.0), (9, 1.0), (100, 0.0), (100, -1.0), (100, 100.1)],
)
def test_resolve_percentage_budget_rejects_invalid_inputs(candidate_queries, percent):
    with pytest.raises(ValueError):
        resolve_percentage_budget(candidate_queries, percent)
```

- [ ] **步骤 2：运行测试，确认因模块不存在而失败**

```bash
python -m pytest -q tests/test_ablation_budget.py
```

预期：collection 失败并报告 `ModuleNotFoundError: ablation_budget`。

- [ ] **步骤 3：实现轻量预算模块**

```python
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PercentageBudget:
    candidate_queries: int
    percent: float
    raw_query_budget: float
    query_budget: int
    budget_units: int
    init_triples: int
    selection_batch_size: int
    max_score_candidates: int


def resolve_percentage_budget(candidate_queries: int, percent: float) -> PercentageBudget:
    candidate_count = int(candidate_queries)
    percentage = float(percent)
    if candidate_count < 10:
        raise ValueError("candidate_queries must be at least 10")
    if not (0.0 < percentage <= 100.0):
        raise ValueError("percent must be in (0, 100]")

    raw_query_budget = candidate_count * percentage / 100.0
    query_budget = int(math.floor(raw_query_budget / 10.0 + 0.5)) * 10
    query_budget = max(10, min(query_budget, candidate_count))
    if query_budget % 10 != 0:
        query_budget = candidate_count - candidate_count % 10
    if query_budget < 10:
        raise ValueError("resolved query budget must be at least 10")

    selection_batch_size = query_budget // 10
    return PercentageBudget(
        candidate_queries=candidate_count,
        percent=percentage,
        raw_query_budget=float(raw_query_budget),
        query_budget=query_budget,
        budget_units=query_budget * 3,
        init_triples=query_budget * 4 // 10,
        selection_batch_size=selection_batch_size,
        max_score_candidates=selection_batch_size * 5,
    )
```

- [ ] **步骤 4：运行预算测试，确认通过**

```bash
python -m pytest -q tests/test_ablation_budget.py
```

预期：`15 passed`。

- [ ] **步骤 5：先增加主程序集成约束测试**

在 `tests/test_ablation_budget.py` 追加：

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "run_pointwise5answers_three_stage_pairwise_listwise_sft_v1.py"


def test_three_stage_script_resolves_and_records_percentage_budget():
    text = TRAIN_SCRIPT.read_text(encoding="utf-8")
    assert 'parser.add_argument("--budget-percent", type=float, default=0.0)' in text
    assert "resolve_percentage_budget(len(train_questions), cfg.budget_percent)" in text
    assert "cfg.budget_units = resolved_budget.budget_units" in text
    assert "cfg.candidate_selector_init_triples = resolved_budget.init_triples" in text
    assert "cfg.candidate_selector_batch_size = resolved_budget.selection_batch_size" in text
    assert "cfg.candidate_selector_max_score_candidates = resolved_budget.max_score_candidates" in text
    assert 'out / "budget_percent_resolution.json"' in text
```

- [ ] **步骤 6：运行集成约束测试，确认失败**

```bash
python -m pytest -q tests/test_ablation_budget.py
```

预期：新增测试因 `--budget-percent` 和解析调用不存在而失败。

- [ ] **步骤 7：接入三阶段主程序**

在 imports 中加入：

```python
from ablation_budget import resolve_percentage_budget
```

在 `RunConfig` 的 `budget_units` 前增加：

```python
    budget_percent: float
```

在 parser 的 `--budget-units` 附近增加：

```python
    parser.add_argument("--budget-percent", type=float, default=0.0)
```

构建 `RunConfig` 时增加：

```python
        budget_percent=float(args.budget_percent),
```

参数校验中增加：

```python
    if not (0.0 <= float(cfg.budget_percent) <= 100.0):
        raise ValueError("budget-percent must be in [0, 100]")
```

将启动信息和 `config.json` 写入移动到 train/validation split 之后，并在其前面加入：

```python
    percentage_budget_stats: Dict[str, Any] = {}
    if float(cfg.budget_percent) > 0.0:
        resolved_budget = resolve_percentage_budget(
            len(train_questions), cfg.budget_percent
        )
        cfg.budget_units = resolved_budget.budget_units
        cfg.candidate_selector_init_triples = resolved_budget.init_triples
        cfg.candidate_selector_batch_size = resolved_budget.selection_batch_size
        cfg.candidate_selector_max_score_candidates = resolved_budget.max_score_candidates
        percentage_budget_stats = asdict(resolved_budget)
        split_info["percentage_budget"] = percentage_budget_stats
        _write_json(out / "budget_percent_resolution.json", percentage_budget_stats)
```

在 `summary["train_budget"]` 中增加：

```python
            "budget_percent": float(cfg.budget_percent),
            "percentage_resolution": percentage_budget_stats,
```

- [ ] **步骤 8：运行预算测试与主程序现有单元测试**

```bash
python -m pytest -q tests/test_ablation_budget.py tests/test_rewardmodel_pointwise.py
```

预期：全部通过。

- [ ] **步骤 9：提交预算支持**

```bash
git add ablation_budget.py tests/test_ablation_budget.py run_pointwise5answers_three_stage_pairwise_listwise_sft_v1.py
git commit -m "Add percentage budgets for ablation runs"
```

### 任务 2：实现 36-job 自动消融队列

**文件：**
- 新建：`launch_qwen3_1p7b_gpt5_ablation_auto_queue.sh`
- 新建：`tests/test_qwen3_1p7b_ablation_queue.py`

- [ ] **步骤 1：先写完整 job 矩阵的失败测试**

```python
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "launch_qwen3_1p7b_gpt5_ablation_auto_queue.sh"

BLOCK_SETTINGS = {
    "selector": (
        "random", "no_uncertainty", "no_diversity", "no_bias",
        "uncertainty_only", "diversity_only", "bias_only",
    ),
    "smoothing": ("a000", "a001", "a005", "a020", "adaptive"),
    "reviewing": ("none",),
    "budget": ("b0p5", "b1", "b2", "b5", "b10"),
}


def launcher_text():
    return LAUNCHER.read_text(encoding="utf-8")


def expected_jobs():
    return {
        f"{dataset}_{block}_{setting}"
        for dataset in ("alpaca", "gpt4all")
        for block, settings in BLOCK_SETTINGS.items()
        for setting in settings
    }


def test_queue_contains_exactly_36_unique_non_ours_jobs():
    text = launcher_text()
    jobs_block = re.search(r"JOBS=\((.*?)\)\n", text, re.S).group(1)
    jobs = re.findall(r"[a-z0-9_]+", jobs_block)
    assert len(jobs) == 36
    assert len(set(jobs)) == 36
    assert set(jobs) == expected_jobs()
    assert "selector_hybrid" not in text
    assert "smoothing_a010" not in text
    assert "reviewing_joint" not in text
```

- [ ] **步骤 2：运行矩阵测试，确认因启动器不存在而失败**

```bash
python -m pytest -q tests/test_qwen3_1p7b_ablation_queue.py
```

预期：失败并报告 launcher 文件不存在。

- [ ] **步骤 3：增加配置、路径和调度测试**

在同一测试文件增加：

```python
def test_queue_uses_qwen3_1p7b_lora_and_current_data_paths():
    text = launcher_text()
    for required in (
        '$ROOT/models/Qwen3-1.7B',
        '$ROOT/data/alpaca/gpt5/train-20k.json',
        '$ROOT/data/alpaca/gpt5/val-2k-eval-listwise.json',
        '$ROOT/data/gpt4all/gpt5/train9k_pointwise_pairwise_no_val_overlap.json',
        '$ROOT/data/gpt4all/gpt5/val3k_pairwise_listwise.json',
        "--use-lora", "--load-in-4bit", "--learning-rate 1e-4",
        "--max-length 4096", "--gradient-accumulation-steps 16",
        "--eval-stages final",
    ):
        assert required in text


def test_queue_encodes_each_ablation_control():
    text = launcher_text()
    for fragment in (
        "selector_random)", "selector_no_uncertainty)",
        "selector_no_diversity)", "selector_no_bias)",
        "selector_uncertainty_only)", "selector_diversity_only)",
        "selector_bias_only)", "smoothing_adaptive)",
        "reviewing_none)", "budget_b0p5)", "budget_b10)",
        '--pointwise-global-smooth-adaptive-entropy',
        '--budget-percent "$budget_percent"',
        '--stage4-replay-strategy "$stage4_strategy"',
    ):
        assert fragment in text


def test_queue_uses_local_output_guards_and_auto_gpu_dispatch():
    text = launcher_text()
    for fragment in (
        'OUTPUT_ROOT="${OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}"',
        'findmnt -n -o FSTYPE -T "$OUTPUT_ROOT"', "nfs|nfs4",
        '--query-compute-apps=gpu_uuid', '--query-gpu=memory.used',
        'if [[ -f "$out/metrics_compact.json" ]]',
        'if [[ -e "$out" ]]', 'sleep "$POLL_SECONDS"',
    ):
        assert fragment in text
```

- [ ] **步骤 4：实现自动队列**

以 `launch_qwen3_gpt5_fullft_auto_queue.sh` 的空闲 GPU 调度、`SKIP_JOBS`、
本地输出检查和 worker 生命周期为参考，新建 launcher。`JOBS` 必须显式列出
两个数据集的 18 个 control。`resolve_job` 使用以下状态变量：

```bash
selector_kind=bias_trap_pointwise
reuse_proxy=1
diversity_weight=1
uncertainty_weight=0.25
bias_weight=1
smooth_alpha=0.1
adaptive_smoothing=0
stage4_strategy=stratified_triple
budget_percent=0
```

setting case 必须精确覆盖：

```bash
case "$variant" in
  selector_random) selector_kind=random; reuse_proxy=0 ;;
  selector_no_uncertainty) uncertainty_weight=0 ;;
  selector_no_diversity) diversity_weight=0 ;;
  selector_no_bias) bias_weight=0 ;;
  selector_uncertainty_only) diversity_weight=0; bias_weight=0 ;;
  selector_diversity_only) uncertainty_weight=0; bias_weight=0 ;;
  selector_bias_only) diversity_weight=0; uncertainty_weight=0 ;;
  smoothing_a000) smooth_alpha=0 ;;
  smoothing_a001) smooth_alpha=0.01 ;;
  smoothing_a005) smooth_alpha=0.05 ;;
  smoothing_a020) smooth_alpha=0.20 ;;
  smoothing_adaptive) adaptive_smoothing=1 ;;
  reviewing_none) stage4_strategy=none ;;
  budget_b0p5) budget_percent=0.5 ;;
  budget_b1) budget_percent=1 ;;
  budget_b2) budget_percent=2 ;;
  budget_b5) budget_percent=5 ;;
  budget_b10) budget_percent=10 ;;
  *) return 2 ;;
esac
```

可选参数用 Bash 数组构造：

```bash
extra_args=()
(( reuse_proxy == 1 )) && extra_args+=(--reuse-selection-proxy-for-stage1)
(( adaptive_smoothing == 1 )) && extra_args+=(--pointwise-global-smooth-adaptive-entropy)
if [[ "$budget_percent" != "0" ]]; then
  extra_args+=(--budget-percent "$budget_percent")
fi
```

公共命令固定传入设计文档中的 LoRA、selector、smoothing、四阶段和 final-only
参数，并在末尾展开 `"${extra_args[@]}"`。输出根目录为：

```bash
RUN_ROOT="$OUTPUT_ROOT/qwen3_1p7b_ablation_seed42"
```

- [ ] **步骤 5：运行队列测试，修正到全部通过**

```bash
python -m pytest -q tests/test_qwen3_1p7b_ablation_queue.py
```

预期：全部通过。

- [ ] **步骤 6：验证 Bash 语法**

```bash
bash -n launch_qwen3_1p7b_gpt5_ablation_auto_queue.sh
```

预期：退出状态 0，无输出。

- [ ] **步骤 7：提交队列**

```bash
git add launch_qwen3_1p7b_gpt5_ablation_auto_queue.sh tests/test_qwen3_1p7b_ablation_queue.py
git commit -m "Add Qwen3-1.7B LoRA ablation queue"
```

### 任务 3：聚焦回归、审查和发布

**文件：**
- 验证：`ablation_budget.py`
- 验证：`run_pointwise5answers_three_stage_pairwise_listwise_sft_v1.py`
- 验证：`launch_qwen3_1p7b_gpt5_ablation_auto_queue.sh`
- 验证：`tests/test_ablation_budget.py`
- 验证：`tests/test_qwen3_1p7b_ablation_queue.py`

- [ ] **步骤 1：运行聚焦回归测试**

```bash
python -m pytest -q \
  tests/test_ablation_budget.py \
  tests/test_qwen3_1p7b_ablation_queue.py \
  tests/test_qwen3_8b_gpt4all_launcher.py \
  tests/test_qwen3_gpt5_fullft_queue.py \
  tests/test_skywork_dataset.py
```

预期：全部通过。

- [ ] **步骤 2：运行语法与差异检查**

```bash
bash -n launch_qwen3_1p7b_gpt5_ablation_auto_queue.sh
git diff --check
git status --short
```

预期：Bash 语法和 diff 检查通过，当前任务文件已提交；用户原有未提交文件保持不变。

- [ ] **步骤 3：审查需求覆盖**

确认以下事实：36 个唯一新任务、没有 Ours、两个数据集各 18 个、seed 42、
Qwen3-1.7B LoRA+4bit、五个 percentage budgets、Reviewing none、Adaptive entropy、
本地输出与 NFS 防护全部由测试覆盖。

- [ ] **步骤 4：合并并推送**

```bash
git checkout main
git merge codex/qwen3-1p7b-lora-ablation
python -m pytest -q tests/test_ablation_budget.py tests/test_qwen3_1p7b_ablation_queue.py
git push origin main
```

预期：GitHub `main` 包含预算支持、36-job 启动器及其测试。
