# Qwen3-1.7B 单任务监督对照实验

这组实验用于 Qwen3-1.7B GPT-5 消融表，覆盖 Alpaca 和 GPT4All 两个数据集，
共六个任务：

```text
alpaca_pointwise_only
alpaca_pairwise_only
alpaca_listwise_only
gpt4all_pointwise_only
gpt4all_pairwise_only
gpt4all_listwise_only
```

## 实验定义

每项任务只使用一种监督，训练 600 条样本、10 epoch：

| 任务 | 训练数据 | 训练样本 |
|---|---|---:|
| pointwise-only | pointwise | 600 |
| pairwise-only | pairwise | 600 |
| listwise-only | listwise | 600 |

配置与 Naive Mix 对齐：Qwen3-1.7B、LoRA + 4-bit、seed 42、学习率 `1e-4`、
`max_length=4096`、per-device batch size 1、gradient accumulation 16、final-only
评估。selector、replay、Stage 4 和 smoothing 均关闭。

即使只训练一种监督，结束后仍会在 pointwise、pairwise、listwise 三套验证集上评估，
指标阶段名称为 `after_single_task`。

## 服务器运行

```bash
cd /data/model-extraction-attack/yaolin/JudgeStealer
conda activate cyl
git pull --ff-only origin main

tmux new -s qwen17_single_task
./launch_qwen3_1p7b_single_task_auto_queue.sh 0 1 2 3 4 5
```

退出 tmux：按 `Ctrl+B`，再按 `D`。队列会自动使用空闲 GPU，任务完成后自动跳过。

只运行指定任务时：

```bash
SKIP_JOBS="alpaca_pairwise_only gpt4all_pairwise_only" \
  ./launch_qwen3_1p7b_single_task_auto_queue.sh 0 1 2 3 4 5
```

默认输出在：

```text
/opt/dlami/nvme/cyl/autodl-tmp/JudgeStealer_outputs/qwen3_1p7b_single_task_seed42/
```

## 查看状态、日志和指标

```bash
ROOT=/opt/dlami/nvme/cyl/autodl-tmp/JudgeStealer_outputs/qwen3_1p7b_single_task_seed42

tail -f "$ROOT/logs/job_status.log"
tail -f "$ROOT/logs/qwen3_1p7b_alpaca_gpt5_b600_lora_trueval_pointwise_only_ep10_noreplay_nosmooth.log"
```

只查看所有完成任务的 compact metrics，不复制模型权重：

```bash
find "$ROOT" -name metrics_compact.json -print -exec sh -c \
  'echo "===== $1 ====="; cat "$1"' _ {} \;
```

单个结果的完整统计位于对应输出目录的 `summary.json`、`config.json` 和
`train_stats_single_task_<task>.json`。

若某个任务日志显示 `ERROR incomplete output exists`，先检查日志确认失败原因；
确认可以重跑后，将该任务目录改名保留，再重新启动队列，避免覆盖实验痕迹。
