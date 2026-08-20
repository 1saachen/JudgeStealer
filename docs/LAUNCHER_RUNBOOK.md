# 训练启动脚本运行手册

本手册记录本项目训练启动脚本的约定、当前 Full-FT 队列配置，以及排查和查看结果的方法。后续新增实验脚本时，应优先沿用这些规则。

当前对应脚本：[`launch_qwen3_gpt5_fullft_auto_queue.sh`](../launch_qwen3_gpt5_fullft_auto_queue.sh)。

Qwen3-1.7B 单任务监督对照实验见
[`QWEN3_1P7B_SINGLE_TASK_EXPERIMENTS.md`](QWEN3_1P7B_SINGLE_TASK_EXPERIMENTS.md)，
启动器为 `launch_qwen3_1p7b_single_task_auto_queue.sh`。

## 1. 核心原则

1. **配置必须显式写入脚本。** 模型、数据、训练阶段、学习率、输出目录和 GPU 规则都不能依赖隐含默认值。
2. **训练结果写入本机 NVMe。** Checkpoint 大、写入频繁，不能放在网络文件系统。当前默认目录为 `/opt/dlami/nvme/cyl/autodl-tmp/JudgeStealer_outputs`。
3. **输出目录名是实验配置的一部分。** 同一实验必须有稳定、可读且唯一的名称；不要在不同配置之间复用同一个输出目录。
4. **`metrics_compact.json` 是完成标记。** 只有该文件存在，队列才把任务认定为完成并自动跳过。
5. **绝不覆盖不完整输出。** 同名目录存在但缺少 `metrics_compact.json` 时，脚本报错退出该任务，防止把失败运行与新运行混在一起。
6. **只使用明确允许的 GPU。** 启动参数给出 GPU 编号，脚本只会在显存占用不高且无计算进程的允许 GPU 上启动任务。
7. **先读日志，再清理失败输出。** 失败目录用于定位原因；确认原因后再移动到备份目录或删除。

## 2. 当前服务器路径

| 角色 | 路径 |
| --- | --- |
| 项目根目录 | `/data/model-extraction-attack/yaolin/JudgeStealer` |
| Conda 环境 | `cyl` |
| 模型目录 | `models/Qwen3-0.6B`、`models/Qwen3-1.7B`、`models/Qwen3-4B`、`models/Qwen3-8B`、`models/Qwen3-14B` |
| Alpaca GPT-5 训练集 | `data/alpaca/gpt5/train-20k.json` |
| Alpaca GPT-5 验证集 | `data/alpaca/gpt5/val-2k-eval-listwise.json` |
| GPT4All GPT-5 训练集 | `data/gpt4all/gpt5/train9k_pointwise_pairwise_no_val_overlap.json` |
| GPT4All GPT-5 验证集 | `data/gpt4all/gpt5/val3k_pairwise_listwise.json` |
| NVMe 输出根目录 | `/opt/dlami/nvme/cyl/autodl-tmp/JudgeStealer_outputs` |
| 状态日志 | `/opt/dlami/nvme/cyl/autodl-tmp/JudgeStealer_outputs/qwen3_gpt5_fullft_auto_queue_logs/job_status.log` |

项目根目录中的 `outputs/` 可以保留旧结果，但不应作为新的大型 checkpoint 写入位置。若只查看结果，直接读取 NVMe 的 `metrics_compact.json` 即可，无需同步权重。

## 3. 当前 Full-FT 队列

脚本按以下任务名调度。`SKIP_JOBS` 必须使用这些任务名，而不是输出目录名。

| 模型 | Alpaca 任务名 | GPT4All 任务名 |
| --- | --- | --- |
| Qwen3-0.6B | `alpaca_0p6b` | `gpt4all_0p6b` |
| Qwen3-1.7B | `alpaca_1p7b` | `gpt4all_1p7b` |
| Qwen3-4B | `alpaca_4b` | `gpt4all_4b` |
| Qwen3-8B | `alpaca_8b` | `gpt4all_8b` |

这套实验的关键配置如下：

| 类别 | 当前配置 |
| --- | --- |
| 训练方式 | Full fine-tuning；不使用 LoRA、4-bit 或 FSDP |
| 选择模式 | `candidate_triple_selector`，代理模式 `lm_head` |
| 选择器训练 | Full-FT；复用 Stage 1 selection proxy |
| 初始/每轮/候选池 | `80 / 20 / 100` |
| 预算 | `600`，即 `200` 个 query |
| Stage 1/2/3 | 各 `1` epoch |
| Stage 4 | `stratified_triple` replay，比例 `1`，`1` epoch |
| 学习率 | 主模型与 proxy 都为 `1e-5` |
| 平滑 | local Gaussian，alpha=`0.1`、sigma=`1.0`、所有 stage |
| 最大长度 | `4096` |
| 每卡 batch size | `1`，梯度累积 `16` |
| 评估 | 仅 final stage |

## 4. 启动前检查

每次在新的 shell 或 tmux 会话中先执行：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cyl
cd /data/model-extraction-attack/yaolin/JudgeStealer

command -v python
command -v nvidia-smi
nvidia-smi -L
```

若脚本报 `nvidia-smi is required`，说明当前 shell 的 `PATH` 找不到该命令。先检查：

```bash
export PATH="/usr/local/nvidia/bin:/usr/bin:$PATH"
hash -r
command -v nvidia-smi
nvidia-smi -L
```

启动前也应确认模型和数据存在。例如：

```bash
ls -lh models/Qwen3-0.6B/config.json models/Qwen3-0.6B/model.safetensors
ls -lh data/alpaca/gpt5/train-20k.json
ls -lh data/gpt4all/gpt5/train9k_pointwise_pairwise_no_val_overlap.json
```

下载缺失的 0.6B 模型：

```bash
hf download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B
```

对分片模型，确认 `model.safetensors.index.json` 以及它列出的所有 `model-*.safetensors` 都已下载；不要只看到配置文件就认为模型完整。

## 5. 用 tmux 启动队列

新建会话：

```bash
tmux new -s qwen_fullft
```

进入 tmux 后，执行环境初始化和启动命令。下面示例只允许使用 GPU `1 2 3 4 5`：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cyl
cd /data/model-extraction-attack/yaolin/JudgeStealer
./launch_qwen3_gpt5_fullft_auto_queue.sh 1 2 3 4 5
```

按 `Ctrl+B`，松开后按 `D` 可退出 tmux 而不中止训练。重新进入：

```bash
tmux attach -t qwen_fullft
```

查看已有会话：

```bash
tmux ls
```

## 6. 可选环境变量

| 变量 | 用途 | 示例 |
| --- | --- | --- |
| `OUTPUT_ROOT` | 覆盖默认 NVMe 输出根目录 | `OUTPUT_ROOT=/new/nvme/path` |
| `PYTHON_BIN` | 覆盖 Python 可执行文件 | `PYTHON_BIN=/root/anaconda3/envs/cyl/bin/python` |
| `POLL_SECONDS` | 空闲 GPU 轮询间隔（秒） | `POLL_SECONDS=60` |
| `SKIP_JOBS` | 显式跳过任务名，以空格分隔 | `SKIP_JOBS="alpaca_1p7b gpt4all_1p7b"` |

例如，保留已有 1.7B 结果，只运行其他任务：

```bash
SKIP_JOBS="alpaca_1p7b gpt4all_1p7b" ./launch_qwen3_gpt5_fullft_auto_queue.sh 1 2 3 4 5
```

环境变量与命令写在同一行最不容易出错。若使用续行反斜杠 `\`，其后不能有任何空格；否则 Bash 会把空格解析成命令。

脚本会在**每一张 GPU 分配任务前**重新处理 `SKIP_JOBS`。因此多卡同时启动时，显式跳过的后排任务也不会被分配。

## 7. 日志、状态和结果

实时查看队列状态：

```bash
tail -f /opt/dlami/nvme/cyl/autodl-tmp/JudgeStealer_outputs/qwen3_gpt5_fullft_auto_queue_logs/job_status.log
```

状态日志的含义：

| 日志 | 含义 |
| --- | --- |
| `STORAGE` | 已确认输出文件系统与可用空间 |
| `START` | 任务已经在指定 GPU 上启动 |
| `DONE` | 训练结束且写入 `metrics_compact.json` |
| `SKIP completed` | 输出目录已有 `metrics_compact.json`，不会重跑 |
| `SKIP configured` | 任务在 `SKIP_JOBS` 中，主动跳过 |
| `ERROR` | 任务未完成；应继续查看该任务的独立日志 |

列出所有 Full-FT 已完成实验：

```bash
find /opt/dlami/nvme/cyl/autodl-tmp/JudgeStealer_outputs \
  -path '*fullft*' -name metrics_compact.json -printf '%h\n' | sort
```

查看单个实验的完整指标：

```bash
python -m json.tool /opt/dlami/nvme/cyl/autodl-tmp/JudgeStealer_outputs/EXPERIMENT_NAME/metrics_compact.json
```

一次打印全部 Full-FT 指标：

```bash
find /opt/dlami/nvme/cyl/autodl-tmp/JudgeStealer_outputs \
  -path '*fullft*' -name metrics_compact.json -print \
  -exec python -m json.tool {} \;
```

仅查看结果时不要用 `rsync` 同步整个输出目录，因为它会复制 checkpoint 权重。直接读取 NVMe 的 JSON 指标文件即可。

## 8. 失败恢复流程

1. 先读取独立任务日志，而不是只看 `job_status.log`：

   ```bash
   tail -n 200 /opt/dlami/nvme/cyl/autodl-tmp/JudgeStealer_outputs/qwen3_gpt5_fullft_auto_queue_logs/EXPERIMENT_NAME.log
   ```

2. 修复根因，例如补全模型下载、修正路径或恢复 CUDA 环境。
3. 检查失败输出目录是否存在。它存在但没有 `metrics_compact.json` 时，队列会报 `ERROR incomplete output exists`。
4. 优先将失败目录移动到备份位置，而不是立即删除：

   ```bash
   mkdir -p /opt/dlami/nvme/cyl/autodl-tmp/JudgeStealer_outputs/failed_runs
   mv /opt/dlami/nvme/cyl/autodl-tmp/JudgeStealer_outputs/EXPERIMENT_NAME \
     /opt/dlami/nvme/cyl/autodl-tmp/JudgeStealer_outputs/failed_runs/
   ```

5. 确认移动的是正确的单个实验目录后，再重新启动队列。只有在不需要保留排查材料时才删除该备份。

## 9. 编写新启动脚本的检查清单

新脚本至少应做到：

1. 把路径、模型、数据和实验超参数集中放在脚本顶部或 `resolve_job` 函数中。
2. 训练前检查 Python、GPU 工具、模型配置、训练脚本和数据文件是否存在。
3. 输出目录包含模型、数据集、打分来源、预算、训练方式和关键策略标签。
4. 将主日志和状态日志写在同一个 NVMe 输出根目录下。
5. 用 `metrics_compact.json` 作为完成标记，并拒绝覆盖不完整输出。
6. GPU 必须由启动命令显式允许，且应检查 GPU UUID、显存占用和计算进程。
7. 每个后台任务都必须记录退出码；任务失败不能悄悄被当成成功。
8. 对可选跳过项使用明确、稳定的任务名，并在每次 GPU 分配前处理跳过规则。
9. 在 README 或本手册中记录一条可直接复制的启动命令、日志命令、结果命令和失败恢复步骤。

这个结构可以直接复用于后续 LoRA、Full-FT、消融实验和不同模型规模的启动脚本；只需替换具体数据、模型、参数和输出名称。

## 10. Claude LoRA 队列

Claude 的八项 LoRA + 4-bit 实验使用：

```bash
./launch_claude_lora_auto_queue.sh 1 2 3 4 5
```

它统一调度 Llama-3.2-1B-Instruct 和 Qwen3-1.7B 在 Alpaca/GPT4All Claude 数据上的四项 Selector 主实验和四项 MixEp10 对照。与 GPT-5 队列不同，Claude 必须显式使用同一份 `val.json` 作为 pairwise 和 listwise 验证输入。完整协议、模型下载和结果命令见 [`CLAUDE_LORA_EXPERIMENTS.md`](CLAUDE_LORA_EXPERIMENTS.md)。

## 11. Qwen3-14B GPT-5 补跑

14B 的 GPT-5 补跑分成两个启动器：LoRA 队列使用单卡，Full-FT 使用四张卡的
FSDP。两者都使用模型目录 `models/Qwen3-14B`，并将结果写到 NVMe。Full-FT 的
主模型 Stage 1/2/3/4 仍然是未量化的全参数训练；由于选样 proxy 在 FSDP 主训练
之前初始化，14B Full-FT 启动器单独使用 `candidate-selector-finetune-mode lora`
和 `candidate-selector-load-in-4bit`，避免每个 rank 各自加载一份完整 proxy 导致
Adam 状态 OOM。这个设置应在结果表或实验记录中标注为“Full-FT surrogate，LoRA
+ 4-bit candidate selector”。多卡运行时，selector proxy 和 bias-trap embedding
模型会按 `LOCAL_RANK` 绑定到各自的可见 GPU；例如传入 `1 2 3 4` 时，四个 rank
分别使用物理 GPU `1/2/3/4`，不应全部落到 GPU 1。

LoRA 允许多个任务在不同空闲 GPU 上并行：

```bash
./launch_qwen3_14b_gpt5_lora_auto_queue.sh 2 3 4 5
```

LoRA 的日志位于：

```bash
tail -f /opt/dlami/nvme/cyl/autodl-tmp/JudgeStealer_outputs/qwen3_14b_gpt5_lora_auto_queue_logs/job_status.log
```

Full-FT 必须提供四张不同且都空闲的 GPU。脚本会等待四张卡同时空闲，然后依次运行
Alpaca 和 GPT4All，避免同一 GPU 对被两个 FSDP 作业复用：

FSDP Full-FT 会在 Stage 1、Stage 2、Stage 3 和最终 Stage 4 都保存阶段 checkpoint，
并在进入下一阶段前重新加载上一个 checkpoint。这是为了避免 Accelerate/FSDP2 把
上一阶段的 FSDP wrapper 当成下一阶段的原始模型；因此单项实验会比只保存最终模型
多占一些 NVMe 空间。

```bash
./launch_qwen3_14b_gpt5_fullft_fsdp.sh 2 3 4 5
```

Full-FT 的日志位于：

```bash
tail -f /opt/dlami/nvme/cyl/autodl-tmp/JudgeStealer_outputs/qwen3_14b_gpt5_fullft_fsdp_logs/job_status.log
```

LoRA 可按任务名跳过：

```bash
SKIP_JOBS="alpaca" ./launch_qwen3_14b_gpt5_lora_auto_queue.sh 2 3 4 5
```

四个结果的完成标志仍是 `metrics_compact.json`。只查看指标而不复制模型权重：

```bash
find /opt/dlami/nvme/cyl/autodl-tmp/JudgeStealer_outputs \
  -maxdepth 1 -type d -name 'qwen3_14b_*_gpt5_*' \
  -exec sh -c 'test -f "$1/metrics_compact.json" && printf "\n== %s ==\n" "$1" && python -m json.tool "$1/metrics_compact.json"' _ {} \;
```
