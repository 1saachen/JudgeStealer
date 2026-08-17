# Qwen3 GPT-5 Full-FT 自动队列设计

## 目标

为 Qwen3-0.6B、1.7B、4B、8B 在 Alpaca GPT-5 与 GPT4All GPT-5 上建立
八项单卡全量微调实验。每项实验由全量微调 proxy 独立完成主动选样，并把选样结束
后的模型直接复用为 Stage 1 结果；Stage 2、Stage 3 和 Stage 4 在同一张 GPU 上
继续训练。

同时新增一个动态 Bash 调度器。用户显式提供允许使用的 GPU 编号，调度器只在这些
GPU 中选择真正空闲的卡，一张卡同时最多运行一项实验，并在任务完成后继续领取队列
中的下一项。

## 实验矩阵

队列包含以下八项，按预计耗时从长到短排列，以缩短并行队列的总完成时间：

1. Qwen3-8B + Alpaca GPT-5
2. Qwen3-8B + GPT4All GPT-5
3. Qwen3-4B + Alpaca GPT-5
4. Qwen3-4B + GPT4All GPT-5
5. Qwen3-1.7B + Alpaca GPT-5
6. Qwen3-1.7B + GPT4All GPT-5
7. Qwen3-0.6B + Alpaca GPT-5
8. Qwen3-0.6B + GPT4All GPT-5

模型目录：

- `models/Qwen3-0.6B`
- `models/Qwen3-1.7B`
- `models/Qwen3-4B`
- `models/Qwen3-8B`

数据目录：

- Alpaca train：`data/alpaca/gpt5/train-20k.json`
- Alpaca listwise eval：`data/alpaca/gpt5/val-2k-eval-listwise.json`
- GPT4All train：
  `data/gpt4all/gpt5/train9k_pointwise_pairwise_no_val_overlap.json`
- GPT4All listwise eval：`data/gpt4all/gpt5/val3k_pairwise_listwise.json`

每项实验使用不同输出目录，名称包含模型规模、数据集、`gpt5`、`b600`、
`fullft`、`selector`、平滑配置和 Stage 4 配置。Full-FT 结果不得覆盖已有 LoRA
结果。默认输出根目录统一使用本地 NVMe：

```text
/opt/dlami/nvme/cyl/autodl-tmp/JudgeStealer_outputs
```

可通过环境变量 `OUTPUT_ROOT` 覆盖，以适配其他服务器。

## Full-FT Selector

在三阶段主入口增加参数：

```text
--candidate-selector-finetune-mode {lora,full}
```

默认值为 `lora`，从而保持所有旧启动脚本和旧实验的行为不变。该值经
`RunConfig` 传入 `_select_candidate_triples_with_selector`，再传给
`LlamaSharedMultiTaskProxyModel(finetune_mode=...)`。底层 proxy 已支持 `lora` 和
`full` 两种模式，不新增第二套模型实现。

当模式为 `full` 时禁止 `--load-in-4bit`。Full-FT 队列不传 `--use-lora` 和
`--load-in-4bit`，并同时设置：

```text
--candidate-selector-finetune-mode full
--candidate-selector-proxy-mode lm_head
--reuse-selection-proxy-for-stage1
--proxy-lr 1e-5
--learning-rate 1e-5
```

选样使用 init 80，之后每轮 select 20，候选池上限 100。预算为 600 个 answer
units，即 200 个 query，因此 init 后恰好完成六轮更新。每个模型和数据集组合均
从对应的原始 Qwen checkpoint 独立选样，不读取 LoRA 实验的
`selected_triples.jsonl`。

## Stage 1 语义

启用 `--reuse-selection-proxy-for-stage1` 后，Full-FT proxy 在 init warmup 和六轮
增量更新中完成的训练就是 Stage 1。选样结束后的完整 causal LM 直接交给后续阶段，
不再额外执行 `--pointwise-epochs 1`。

因此本实验的阶段定义为：

- Stage 1：init 80 训练 3 epochs，六轮新增样本各更新 1 epoch；
- Stage 2：pairwise 1 epoch；
- Stage 3：listwise 1 epoch；
- Stage 4：对 200 个已选 triple 做 full stratified replay，1 epoch。

启动参数仍保留 `--pointwise-epochs 1`，用于记录统一配置并保持未复用分支的默认值，
但在 proxy 复用分支中不会触发第二次 Stage 1 训练。

## 公共训练配置

八项实验统一使用：

```text
seed = 42
budget_units = 600
stage2_pointwise_replay_ratio = 0
stage3_pointwise_replay_ratio = 0
stage3_pairwise_replay_ratio = 0
stage4_replay_strategy = stratified_triple
stage4_replay_fraction = 1
stage4_epochs = 1
pointwise_epochs = 1
pairwise_epochs = 1
listwise_epochs = 1
per_device_batch_size = 1
gradient_accumulation_steps = 16
proxy_lr = 1e-5
learning_rate = 1e-5
max_length = 4096
proxy_max_length = 768
eval_batch_size = 1
eval_stages = final
pointwise_global_smooth_mode = local_gaussian
pointwise_global_smooth_alpha = 0.1
pointwise_global_smooth_gaussian_sigma = 1.0
pointwise_global_smooth_stages = all
```

Selector 的其余 bias-trap、diversity、uncertainty、embedding 和 no-exploration 参数
与现有 GPT-5 LoRA 八项队列保持一致。

## 自动调度器

在仓库根目录新增：

```text
launch_qwen3_gpt5_fullft_auto_queue.sh
```

调用方式：

```bash
./launch_qwen3_gpt5_fullft_auto_queue.sh 1 2 3 5
```

位置参数是允许使用的物理 GPU 编号。脚本至少要求一个 GPU 编号，不在白名单中的
GPU 永远不会被使用。脚本不调用 `torchrun`，也不传入任何 FSDP 参数；每项实验是
单进程、单 GPU Full-FT。

GPU 只有同时满足以下条件才视为空闲：

1. `nvidia-smi` 报告该 GPU 上没有活跃 compute process；
2. `memory.used` 不超过 1024 MiB；
3. 该 GPU 没有本调度器仍在运行的 worker。

调度器在真正启动 worker 前重新检查一次 GPU 状态，以缩小与其他任务竞争 GPU 的
时间窗口。没有空闲 GPU 时每 30 秒检查一次。每张空闲卡最多启动一个实验；worker
退出后，调度器记录返回码并继续调度后续实验。

## 输出保护与日志

每项实验启动前检查：

- Python 入口存在；
- 模型目录及 `config.json` 存在；
- train 与 eval 文件存在；
- 同一输出目录没有正在运行的进程。

若输出目录包含 `metrics_compact.json`，该项标记为完成并跳过。若输出目录存在但不
完整，记录错误并跳过该项，不删除、不覆盖现有文件。某一项缺少资源或训练失败时，
队列继续运行其他独立实验，最终调度器以非零状态退出。

日志目录统一位于：

```text
$OUTPUT_ROOT/qwen3_gpt5_fullft_auto_queue_logs/
```

其中包含共享 `job_status.log` 和每项实验各自的训练日志。状态日志至少记录
`START`、`DONE`、`SKIP` 和 `ERROR`，并包含 job、GPU、输出目录和错误码。
启动器使用 `findmnt` 记录输出文件系统类型，并拒绝把完整模型 checkpoint 写入
`nfs` 或 `nfs4`；同时使用 `df` 把 NVMe 剩余容量写入状态日志。

## 测试与验证

新增针对性测试验证：

- 新 CLI 参数默认是 `lora`，并接受 `full`；
- selector 构造时实际收到配置中的 finetune mode；
- `full + load_in_4bit` 被拒绝；
- 旧 LoRA 配置仍保持原行为；
- 启动器包含完整的 4 × 2 实验矩阵和正确路径；
- Full-FT 启动命令不包含 LoRA、4-bit、FSDP 或固定 LoRA 选样；
- 两个学习率均为 `1e-5`，其余公共实验参数与设计一致；
- GPU 白名单、空闲检测、完成跳过、不完整输出保护和失败后继续队列的逻辑存在。

验证只执行静态或 CPU 级单元测试、Python 语法检查和 Bash 语法检查，不启动真实
模型训练。
