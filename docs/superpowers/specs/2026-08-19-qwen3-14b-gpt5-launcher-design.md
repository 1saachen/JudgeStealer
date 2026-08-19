# Qwen3-14B GPT-5 补跑启动脚本设计

## 目标

为 GPT-5 victim 的模型规模对比补齐 Qwen3-14B 四项结果：Alpaca 与 GPT4All
各一项 LoRA 和 Full-FT。所有任务沿用当前 0.6B--8B 表格实验的预算、选样、
训练阶段、平滑和最终评测协议。

## 任务矩阵

| 任务 | 训练方式 | 并行方式 | 输出名称前缀 |
| --- | --- | --- | --- |
| Alpaca | LoRA + 4-bit | 单卡 | `qwen3_14b_alpaca_gpt5_b600_lora_` |
| GPT4All | LoRA + 4-bit | 单卡 | `qwen3_14b_gpt4all_gpt5_b600_lora_` |
| Alpaca | Full-FT | 四卡 FSDP | `qwen3_14b_alpaca_gpt5_b600_fullft_` |
| GPT4All | Full-FT | 四卡 FSDP | `qwen3_14b_gpt4all_gpt5_b600_fullft_` |

模型目录固定为 `models/Qwen3-14B`。输出默认放在
`/opt/dlami/nvme/cyl/autodl-tmp/JudgeStealer_outputs`，可以由 `OUTPUT_ROOT`
覆盖。

## 共同实验协议

- seed 42，budget 600；每个 surrogate 端到端重新选样；
- Stage 1/2/3 各训练 1 epoch，Stage 4 使用全部分层 triple replay 训练 1 epoch；
- `max_length=4096`，训练 batch size 1，梯度累积 16，只做最终评测；
- local-Gaussian smoothing：alpha 0.1、sigma 1.0，覆盖全部阶段；
- selector 维持 candidate triple selector、LM head proxy、init 80、每轮 20、候选池
  100，以及当前表格使用的零探索与 bias/diversity 设置。LoRA 复用 Stage-1 proxy；
  FSDP Full-FT 不复用该 proxy，因为当前训练入口明确禁止 FSDP 与
  `--reuse-selection-proxy-for-stage1` 组合，Stage 1 从原始 Qwen checkpoint 开始。
- Alpaca 使用 `data/alpaca/gpt5/train-20k.json` 和
  `data/alpaca/gpt5/val-2k-eval-listwise.json`；GPT4All 使用
  `data/gpt4all/gpt5/train9k_pointwise_pairwise_no_val_overlap.json` 和
  `data/gpt4all/gpt5/val3k_pairwise_listwise.json`。

## LoRA 队列

新建单卡自动队列脚本。它接受一组允许使用的 GPU 编号，检测未被计算进程占用且
显存占用不超过 1 GB 的 GPU；两个数据集任务可并行。每项使用 `--use-lora` 与
`--load-in-4bit`，selector proxy 采用 LoRA，学习率和 proxy 学习率均为 `1e-4`。

脚本在开始前检查模型、数据和 Python 入口；完成标志为 `metrics_compact.json`。
已完成任务跳过，残缺输出保留并明确报错，避免覆盖。状态和每项日志都写入 NVMe
输出目录。`SKIP_JOBS` 可显式跳过某项任务。

## Full-FT 启动器

新建四卡 FSDP 启动器。它接受四个 GPU 编号，在四张卡都空闲时，用 `torchrun`
以四个进程启动单个任务；两个数据集依序完成。FSDP 使用 `full_shard auto_wrap`、
`Qwen3DecoderLayer`、activation checkpointing 和 `FULL_STATE_DICT`，只保存最后
阶段。主模型不传递 LoRA 或 4-bit 参数；由于 selector proxy 在 FSDP 主训练前初始化，
selector proxy 单独使用 LoRA + 4-bit（通过 `--candidate-selector-load-in-4bit`），
主模型学习率和 proxy 学习率均为 `1e-5`。

FSDP 仅改变分布式内存布局，仍然是完整参数训练。由于两张卡组成一个任务，Full-FT
启动器不会在同一 GPU 集合上并行第二项实验。该入口会重新完成候选选样，但 Stage 1
从原始 checkpoint 开始，不加载单进程 selector proxy。结果应标注为“Full-FT
surrogate，LoRA + 4-bit candidate selector”。

## 验证与使用

静态验证检查脚本的 shell 语法、任务矩阵、数据路径、模型路径、互斥的训练参数和
FSDP 启动参数。不会实际下载模型或启动训练。

LoRA 预期使用方式：

```bash
./launch_qwen3_14b_gpt5_lora_auto_queue.sh 2 3
```

Full-FT 预期使用方式：

```bash
./launch_qwen3_14b_gpt5_fullft_fsdp.sh 2 3 4 5
```

两种脚本均可在 tmux 中运行；状态日志用于查看进度，`metrics_compact.json` 是任务
完成与汇总结果的唯一判据。
