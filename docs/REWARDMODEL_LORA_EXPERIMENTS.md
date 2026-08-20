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

Ours 在 1500 条 question 训练池中使用与主实验一致的
`candidate_triple_selector + bias_trap_pointwise`，预算为 600 个 answer
units，即 200 个三回答问题。selector 使用 LM-head proxy、`80 / 20 / 100`
的初始化/批次/候选池配置，以及 diversity/uncertainty/bias 权重
`1.0 / 0.25 / 1.0`；选样 proxy 直接复用为 Stage 1。连续的 1--5 reward
只为 selector 临时量化到 1--10，最终 pointwise SFT 和评估仍使用原始连续分数。
Ours 的 pairwise 和 listwise 训练同时开启顺序增强，Naive 不开启。后续阶段
各训练 1 个 epoch，并使用 LoRA + 4-bit。

Converted 分支保留 UniRRM 的三阶段评价协议和 `<User_Input>` /
`<ResponseN>` 输入结构，只把最终输出转换成任务标签：pointwise 输出
`Score: [X]`，pairwise 输出 `[[1]]` / `[[2]]` / `[[3]]`，listwise 只输出裸
数字 `1` / `2` / `3`。不输出 JSON、ranking、`Best:` 或 `ResponseN` 文本。
Pointwise 保留连续分数。Pairwise 的唯一胜者遵从原始 `choice`；分数相同
时，两个候选赢家使用均匀软目标；数据明确标记的 `choice=C` 保持为平局。
Listwise 最高分并列时，在并列最佳之间使用均匀软目标。验证时，预测任一
并列最佳都计为正确。

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
