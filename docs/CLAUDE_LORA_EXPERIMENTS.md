# Claude LoRA 实验说明

本实验使用 Claude 已标注的离线 JSON，比较两种 surrogate：Llama-3.2-1B-Instruct 与 Qwen3-1.7B，并在 Alpaca 和 GPT4All 上分别运行主动选样主方法和 MixEp10 对照。

启动脚本：[`launch_claude_lora_auto_queue.sh`](../launch_claude_lora_auto_queue.sh)。

## 实验矩阵

| 数据集 | Llama-3.2-1B-Instruct | Qwen3-1.7B |
| --- | --- | --- |
| Alpaca Claude | Selector、MixEp10 | Selector、MixEp10 |
| GPT4All Claude | Selector、MixEp10 | Selector、MixEp10 |

共八项任务。任务名如下：

```text
selector_alpaca_llama1b       selector_gpt4all_llama1b
selector_alpaca_qwen1p7b      selector_gpt4all_qwen1p7b
mixep10_alpaca_llama1b        mixep10_gpt4all_llama1b
mixep10_alpaca_qwen1p7b       mixep10_gpt4all_qwen1p7b
```

## 数据与模型路径

```text
data/alpaca/claude/train.json
data/alpaca/claude/val.json
data/gpt4all/claude/train.json
data/gpt4all/claude/val.json

models/Llama-3.2-1b-instruct
models/Qwen3-1.7B
```

Claude 的 `val.json` 同时用作显式 pairwise 和 listwise 输入。数据 loader 会读取 `answerA/B/C`、`scoreA/B/C`、`pairwise_ab_choice`、`pairwise_bc_choice` 和 `listwise_ranking`；缺失的 AC 偏好不会被解释为 tie。

Llama 模型需要先在 Hugging Face 接受 Meta 的访问协议并完成登录，然后下载：

```bash
hf download meta-llama/Llama-3.2-1B-Instruct \
  --local-dir models/Llama-3.2-1b-instruct
```

下载后确认：

```bash
ls -lh models/Llama-3.2-1b-instruct/config.json
ls -lh models/Qwen3-1.7B/config.json
```

## Selector 主实验

每个 surrogate 和数据集独立进行端到端主动选样：

- LoRA + 4-bit；学习率 `1e-4`；最大长度 `4096`；batch size `1`，梯度累积 `16`；
- `candidate_triple_selector`，LM-head proxy；初始化 `80` triples；每轮 `20`；候选池 `100`；
- budget `600`，即 `200` 个 triples；
- Stage 1/2/3/4 各一 epoch；Stage 4 使用 full stratified replay；
- local-Gaussian smoothing：alpha=`0.1`、sigma=`1.0`、所有 stage；
- 仅做最终评估；
- 显式传入 Claude pairwise 和 listwise 验证文件。

Selector 的输出名称包含 `claude_b600_lora_selector_smooth_a010_pool100_stage4stratfull`。

## MixEp10 对照

MixEp10 是 exposure-matched 的 true-value mixed control，而不是四阶段 replay：

- LoRA + 4-bit；学习率 `1e-4`；最大长度 `4096`；
- 200 pointwise + 200 真实 pairwise + 200 真实 listwise 训练例；
- Stage 1/2/3 各训练 `10` epoch；没有 Stage 4；
- 无 selector、无 replay、无 smoothing；
- 从 pairwise/listwise 验证文件中抽走用于训练的样本后，剩余样本继续用于评估；
- 总训练 exposure 为 `600 × 10 = 6000`，与四阶段 full-replay Selector 主方法匹配。

MixEp10 的输出名称包含 `claude_b600_lora_trueval_mix200pw200pair200list_ep10_noreplay_nosmooth`。

## 启动

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cyl
cd /data/model-extraction-attack/yaolin/JudgeStealer

tmux new -s claude_lora
./launch_claude_lora_auto_queue.sh 1 2 3 4 5
```

脚本仅在指定 GPU 空闲时启动任务，结果和日志默认写到：

```text
/opt/dlami/nvme/cyl/autodl-tmp/JudgeStealer_outputs
```

退出 tmux 而不断开训练：按 `Ctrl+B`，松开后按 `D`。重新进入：

```bash
tmux attach -t claude_lora
```

若已有某些结果，可按任务名跳过：

```bash
SKIP_JOBS="selector_alpaca_llama1b mixep10_alpaca_llama1b" \
  ./launch_claude_lora_auto_queue.sh 1 2 3 4 5
```

## 查看状态和指标

实时状态：

```bash
tail -F /opt/dlami/nvme/cyl/autodl-tmp/JudgeStealer_outputs/claude_lora_auto_queue_logs/job_status.log
```

列出已完成任务：

```bash
find /opt/dlami/nvme/cyl/autodl-tmp/JudgeStealer_outputs \
  -path '*claude*' -name metrics_compact.json -printf '%h\n' | sort
```

查看一项指标：

```bash
python -m json.tool /opt/dlami/nvme/cyl/autodl-tmp/JudgeStealer_outputs/EXPERIMENT_NAME/metrics_compact.json
```

若任务失败，先读取同名 `.log` 文件。输出目录存在但缺少 `metrics_compact.json` 时，队列会拒绝覆盖；先保留失败目录定位错误，再移动或删除该单个目录后重跑。
