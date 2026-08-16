# 当前 Python 代码说明

本文只描述当前实验真正需要的 Python 代码。Shell 文件只是具体实验的
参数组合，不属于核心实现。

## 训练入口

| 文件 | 作用 | 何时直接运行 |
|---|---|---|
| `run_rewardmodel_three_stage_sft.py` | 连续 reward-model 的 pointwise -> pairwise -> listwise SFT，并负责 mix、selector、native JSON 等模式 | 当前 reward-model 实验 |
| `run_pointwise5answers_three_stage_pairwise_listwise_sft_v1.py` | 三阶段 SFT、candidate-triple selector、Stage 4 replay、平滑和最终评估 | Alpaca/Dolly selector 实验 |
| `run_newnew_one_answer_trueval_three_stage_sft.py` | one-answer 和 true-value mix 对照 | Mix/control 实验 |

## 核心依赖

这些文件通常不需要直接运行，但上面的训练入口会 import 它们：

| 文件 | 作用 |
|---|---|
| `run_pointwise5answers_two_to_pairwise_v1.py` | 基础数据结构、prompt、pairwise 训练/评估、模型加载和 selector 公共实现 |
| `run_pointwise5answers_three_to_listwise_v1.py` | listwise 数据、candidate-triple selector 和 listwise 评估 |
| `run_skywork_pointwise.py` | continuous reward 样本选择和 pointwise proxy |

当前依赖关系：

```text
run_rewardmodel_three_stage_sft.py
├── run_skywork_pointwise.py
├── run_newnew_one_answer_trueval_three_stage_sft.py
└── run_pointwise5answers_three_stage_pairwise_listwise_sft_v1.py
    ├── run_pointwise5answers_three_to_listwise_v1.py
    └── run_pointwise5answers_two_to_pairwise_v1.py
```

## 数据准备

| 文件 | 作用 |
|---|---|
| `prepare_rewardmodel_three_stage.py` | 对齐 continuous pointwise、pairwise、listwise 数据并生成 train/mix/eval split |
| `prepare_fair_split.py` | 为近期多 judge 实验生成公平的 1500/200/300 划分 |
| `prepare_dolly_train9k_no_val_overlap.py` | 生成无验证集泄漏的 Dolly 训练数据 |

这些脚本生成的数据仍应放在 GitHub 之外。

## 内部包

当前训练链只需要 `train_with_selector/train_with_selector/` 下的以下实现：

- `data/judge_dataset.py`
- `data/pairwise_dataset.py`
- `data/skywork_dataset.py`
- `models/base.py`
- `models/llama_proxy.py`
- `models/llama_shared_proxy.py`
- `models/llama_shared_multitask_proxy.py`
- `selector/base.py`
- `selector/binary_selector.py`
- `selector/shared_llama_selector_v2.py`

项目统一使用完整包路径 `train_with_selector.train_with_selector...`，不再在
运行时修改 `sys.path`。这也避免了之前测试顺序不同导致导入失败的问题。

## 已归档代码

旧的 standalone 训练、早期 selector comparison、旧 pairwise evaluator、
profiling 和汇总脚本已放到本地忽略目录：

```text
archive/legacy_python_20260816/
```

该目录包含 53 个 Python 文件，不会上传到 GitHub；确认以后完全不再需要时
可以在本地删除。
