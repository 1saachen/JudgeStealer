# Reward-Model LoRA 实验

本实验脚本只使用三文件 reward-model 数据：`pointwise.json`、
`pairwise.json` 和 `listwise.json`。Claude 数据不在本实验范围内。

## 实验矩阵

| 方法 | Llama-3.2-1B-Instruct | Qwen3-1.7B |
| --- | --- | --- |
| Naive | Alpaca、GPT4All | Alpaca、GPT4All |
| Ours | Alpaca、GPT4All | Alpaca、GPT4All |

启动器：`launch_rewardmodel_lora_auto_queue.sh`。

## 数据切分

启动器默认读取：

```text
/data/model-extraction-attack/yaolin/JudgeStealer/data/reward-model/
```

并使用 `prepare_rewardmodel_three_stage.py` 做一次固定 seed=42 的切分：
1500 条 selector 训练池、200 条 Naive 训练集、300 条共同验证集。三种
监督信号按 question ID 对齐，验证集不参与训练。

单一 reward-model 数据源可直接使用默认路径，也可以覆盖：

```bash
export REWARDMODEL_SOURCE=/data/model-extraction-attack/yaolin/JudgeStealer/data/reward-model
```

默认单源模式会运行 4 项：两个模型 × Naive/Ours。只有在确实有两套
独立数据时，才指定两个目录并运行 Alpaca/GPT4All 的 8 项矩阵：

```bash
export ALPACA_REWARDMODEL_SOURCE=/path/to/alpaca/reward-model
export GPT4ALL_REWARDMODEL_SOURCE=/path/to/gpt4all/reward-model
```

每个目录必须包含 `pointwise.json`、`pairwise.json` 和 `listwise.json`。

## 训练协议

Naive 不使用 selector，三阶段最终各取 200 个训练例，分别训练 10 个
epoch，关闭平滑。

Ours 在 1500 条 question 训练池中使用 proxy selector，预算为 600 个
answer units，即 200 个三回答问题；selector 使用 `80 / 20 / 100` 的
初始化、批次和候选池配置，三阶段各训练 1 个 epoch，并使用 LoRA + 4-bit。

Pointwise 保留连续分数。Pairwise 的唯一胜者遵从原始 `choice`；分数相同
时，两个候选赢家使用均匀软目标；数据明确标记的 `choice=C` 保持为平局。
Listwise 只输出 `Best: [ResponseN]`，不输出 ranking 或分数；最高分并列
时，在并列最佳之间使用均匀软目标。验证时，预测任一并列最佳都计为正确。

## 启动

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cyl
cd /data/model-extraction-attack/yaolin/JudgeStealer

REWARDMODEL_SOURCE=/data/model-extraction-attack/yaolin/JudgeStealer/data/reward-model \
  ./launch_rewardmodel_lora_auto_queue.sh 1 2 3 4 5
```

可用 `SKIP_JOBS` 跳过已完成任务。结果根目录由 `OUTPUT_ROOT` 控制，完成
标记为每个实验目录下的 `metrics_compact.json`。
