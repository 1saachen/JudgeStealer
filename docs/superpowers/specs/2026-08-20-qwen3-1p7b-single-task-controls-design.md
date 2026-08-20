# Qwen3-1.7B 单任务监督对照实验设计

## 1. 目标与范围

在现有 Qwen3-1.7B GPT-5 消融表中新增三种单任务监督对照：

- Pointwise-only
- Pairwise-only
- Listwise-only

Alpaca 和 GPT4All 均运行这三项，共六个任务。每项总训练预算固定为 600，
仅使用一种监督类型，不把 600 条数据拆成三个训练阶段。

## 2. 对照定义

现有 Naive Mix 使用 200 条 pointwise、200 条 pairwise 和 200 条 listwise，
三个阶段各训练 10 epoch。新增对照保留其模型适配、优化和评估配置，只改变训练
监督的组成：

| 对照 | Pointwise | Pairwise | Listwise | 训练阶段 |
|---|---:|---:|---:|---:|
| Naive Mix | 200 | 200 | 200 | 3 |
| Pointwise-only | 600 | 0 | 0 | 1 |
| Pairwise-only | 0 | 600 | 0 | 1 |
| Listwise-only | 0 | 0 | 600 | 1 |

每个 single-task 任务只创建一次 Trainer，并对选定的 600 条样本训练 10 epoch。
这样避免重复重启学习率调度，也使 `only` 明确表示仅使用一种监督。

## 3. 固定配置

六项任务统一使用：

```text
model = models/Qwen3-1.7B
adaptation = LoRA + 4-bit
seed = 42
budget = 600
epochs = 10
learning_rate = 1e-4
max_length = 4096
per_device_batch_size = 1
gradient_accumulation_steps = 16
eval_batch_size = 1
evaluation = final only
selector = disabled
replay = disabled
stage4 = disabled
smoothing = disabled
```

除训练任务类型和对应样本来源外，不改变 Naive Mix 的配置。

## 4. 数据与评估

数据路径沿用现有 Qwen3-1.7B GPT-5 Naive Mix：

```text
Alpaca pointwise source = data/alpaca/gpt5/train-20k.json
Alpaca pairwise source  = data/alpaca/gpt5/val-2k-eval.json
Alpaca listwise source  = data/alpaca/gpt5/val-2k-eval-listwise.json

GPT4All pointwise source = data/gpt4all/gpt5/train9k_pointwise_pairwise_no_val_overlap.json
GPT4All pair/list source = data/gpt4all/gpt5/val3k_pairwise_listwise.json
```

Pointwise-only 从 pointwise 原训练集的训练划分中抽取 600 个不同问题的一条回答。
Pairwise-only 和 listwise-only 沿用 true-value 控制组的固定 seed 划分逻辑，从对应
验证来源划出 600 条训练样本，其余样本用于评估，避免训练和评估重叠。

所有任务训练结束后统一评估 pointwise、pairwise 和 listwise。指标 stage key 使用
`after_single_task`，保证三个对照可以直接汇总到同一张表中。

## 5. 程序结构

扩展 `run_newnew_one_answer_trueval_three_stage_sft.py`，增加明确的 single-task
运行模式及 `--single-task pointwise|pairwise|listwise` 参数。该模式负责：

1. 加载三类评估数据；
2. 只构建指定任务的 600 条训练数据；
3. 从原始 Qwen checkpoint 创建一次 LoRA Trainer；
4. 使用对应任务名称训练 10 epoch；
5. 在三类验证集上执行一次最终评估；
6. 写入配置、抽样记录、训练统计、summary 和 `metrics_compact.json`。

不使用“把另外两类样本数量设为 0 后继续执行三阶段”的隐式方案，避免空训练集
报错及不准确的阶段统计。

## 6. 自动队列

新增一个独立启动器，包含以下六个任务：

```text
alpaca_pointwise_only
alpaca_pairwise_only
alpaca_listwise_only
gpt4all_pointwise_only
gpt4all_pairwise_only
gpt4all_listwise_only
```

启动器沿用已有队列约定：

- 接受一个或多个允许使用的 GPU ID；
- 只向空闲 GPU 分配任务；
- 默认输出到 `/opt/dlami/nvme/cyl/autodl-tmp/JudgeStealer_outputs`；
- 拒绝 NFS/NFS4 输出目录；
- 已存在 `metrics_compact.json` 时跳过；
- 同名残缺输出目录存在时拒绝覆盖；
- 使用 job lock 防止重复启动；
- 支持 `SKIP_JOBS`；
- 保存独立任务日志和统一 `job_status.log`。

输出目录名必须包含模型、数据集、任务类型、budget 600、LoRA、true-value、
10 epoch、无 replay 和无 smoothing，避免与 Naive Mix 或 selector 结果冲突。

## 7. 验证要求

实现验证覆盖：

- single-task 参数只能接受三种任务类型；
- 每种模式只训练对应的 600 条数据；
- pairwise/listwise 训练与评估不重叠；
- 仅创建一次训练阶段并使用 10 epoch；
- 三种验证指标均写入 `after_single_task`；
- 六个 job 名和输出目录唯一；
- LoRA、4-bit、优化参数和 Naive Mix 一致；
- selector、replay、Stage 4 和 smoothing 均未启用；
- 自动队列的存储、GPU、锁和残缺输出保护有效；
- Python focused tests、Bash 语法检查和 diff 检查通过。
