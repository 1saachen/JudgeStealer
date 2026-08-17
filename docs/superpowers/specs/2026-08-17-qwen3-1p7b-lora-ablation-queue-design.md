# Qwen3-1.7B LoRA 消融实验去重队列设计

## 1. 目标与范围

为论文消融表建立 Qwen3-1.7B LoRA 单 seed 去重队列，覆盖 Alpaca GPT-5 与
Dolly/GPT4All GPT-5。表格包含 Selector、Smoothing、Reviewing 和 Query Budget
四个区块，共 42 个表格位置。

两个数据集的标准 Ours 实验已经完成，不进入新队列。Selector 的 Hybrid、
Smoothing 的 fixed local-Gaussian `alpha=0.10` 和 Reviewing 的 Joint 三行分别
复用同一个已有 Ours 结果。因此新队列每个数据集运行 18 个任务，合计 36 个。

本设计不修改论文中 `Impact of Pointwise/Pairwise Reviewing` 的标题。

## 2. 统一实验配置

除被消融因素外，所有任务固定为：

```text
surrogate_model = Qwen3-1.7B
training = LoRA + 4-bit
seed = 42
learning_rate = 1e-4
max_length = 4096
per_device_batch_size = 1
gradient_accumulation_steps = 16
pointwise_epochs = 1
pairwise_epochs = 1
listwise_epochs = 1
stage4_epochs = 1
evaluation = final only
```

默认方法配置为：

```text
selector = bias_trap_pointwise
selector_proxy_mode = lm_head
reuse_selection_proxy_for_stage1 = true
diversity_weight = 1.0
uncertainty_weight = 0.25
bias_weight = 1.0
pointwise_length_bias_weight = 0.5
pairwise_position_bias_weight = 0.5
pairwise_position_bias_scale = 0.02
exploration_ratio = 0
smoothing_mode = local_gaussian
smoothing_alpha = 0.10
smoothing_sigma = 1.0
smoothing_stages = all
stage4_replay_strategy = stratified_triple
stage4_replay_fraction = 1
```

数据路径使用仓库当前布局：

```text
Alpaca train = data/alpaca/gpt5/train-20k.json
Alpaca eval  = data/alpaca/gpt5/val-2k-eval-listwise.json
Dolly train  = data/gpt4all/gpt5/train9k_pointwise_pairwise_no_val_overlap.json
Dolly eval   = data/gpt4all/gpt5/val3k_pairwise_listwise.json
Model        = models/Qwen3-1.7B
```

## 3. 消融矩阵

### 3.1 Selector

每个数据集新增 7 个任务：

| Setting | diversity | uncertainty | bias | 说明 |
|---|---:|---:|---:|---|
| Random selection | 当前 random 规则 | 当前 random 规则 | 当前 random 规则 | 一次随机选满，不复用 selector proxy |
| Without uncertainty | 1.0 | 0 | 1.0 | 其余默认 |
| Without diversity | 0 | 0.25 | 1.0 | 其余默认 |
| Without bias | 1.0 | 0.25 | 0 | 其余默认 |
| Uncertainty only | 0 | 0.25 | 0 | 保留默认 uncertainty 尺度 |
| Diversity only | 1.0 | 0 | 0 | 保留默认 diversity 尺度 |
| Bias only | 0 | 0 | 1.0 | 保留默认 bias 尺度 |

Hybrid selector 直接复用已有 Ours，不运行。

Random 按用户确认的现有代码规则执行。它与 Hybrid 的 Stage-1 proxy 复用方式不同；
结果解释中应把它视为当前完整 Random pipeline 对照，而不是严格只改变 acquisition
score 的单因素对照。

### 3.2 Smoothing

每个数据集新增 5 个任务：

```text
Without smoothing: alpha=0
Fixed local-Gaussian: alpha=0.01
Fixed local-Gaussian: alpha=0.05
Fixed local-Gaussian: alpha=0.20
Adaptive entropy smoothing: base alpha=0.10 + adaptive entropy
```

Fixed local-Gaussian `alpha=0.10` 复用已有 Ours，不运行。所有新 smoothing 任务从头
执行当前完整 pipeline。Adaptive entropy 使用
`--pointwise-global-smooth-adaptive-entropy`，不启用 trainable alpha。

### 3.3 Reviewing

每个数据集新增 1 个任务：

```text
Without reviewing: --stage4-replay-strategy none
```

Joint reviewing 复用已有 Ours，其定义为 Stage 4 full stratified replay，fraction 1，
1 epoch，同时使用从 selected triples 派生的 pointwise、pairwise 和 listwise 数据。

### 3.4 Query Budget

每个数据集新增 5 个任务：

```text
B = 0.5%, 1%, 2%, 5%, 10%
```

百分比分母为数据加载、过滤和 train/validation split 后实际可选的 query 数 `N`。
每个预算按以下规则在运行时解析：

```text
raw_query_budget = B * N
query_budget = 四舍五入到最近的 10 的倍数
budget_units = 3 * query_budget
init_triples = 0.4 * query_budget
selection_batch_size = 0.1 * query_budget
max_score_candidates = 5 * selection_batch_size
```

上述比例保证 `init 40% + 6 rounds * 10% = 100%`，同时维持当前
`pool:select = 5:1`。预算任务使用完整 Ours selector、`alpha=0.10` smoothing 和
Joint reviewing。

为避免使用数据文件名中的近似规模推算，主程序需要支持 percentage budget，并在
构建实际候选池后解析上述整数参数。解析后的 `N`、百分比、query budget、
`budget_units`、init、batch 和 pool 必须写入配置与统计文件。

## 4. 去重与任务数量

表格位置数量：

```text
Selector: 8 * 2 = 16
Smoothing: 6 * 2 = 12
Reviewing: 2 * 2 = 4
Budget: 5 * 2 = 10
Total: 42
```

每个数据集的 Hybrid、`alpha=0.10` 和 Joint 是同一个已有 Ours 结果，均不排队。
新增任务数量为：

```text
Selector controls: 7 * 2 = 14
Smoothing controls: 5 * 2 = 10
Reviewing controls: 1 * 2 = 2
Budget sweep: 5 * 2 = 10
New jobs: 36
```

队列不得生成 Ours job，也不得覆盖用户已有 Ours 输出。

## 5. 启动器与输出管理

新增一个 Qwen3-1.7B LoRA 消融 worker，接口沿用现有模式：

```bash
./launch_qwen3_1p7b_gpt5_ablation_queue.sh <gpu_id> <job> [job ...]
```

job 名必须包含 dataset、block 和 setting。输出目录名还必须包含 Qwen3-1.7B、
LoRA、seed 42 和解析后的预算，保证所有任务互不覆盖。

输出和日志默认写入本地磁盘：

```text
/opt/dlami/nvme/cyl/autodl-tmp/JudgeStealer_outputs/qwen3_1p7b_ablation_seed42
```

允许通过 `OUTPUT_ROOT` 覆盖，但检测到 `nfs` 或 `nfs4` 时拒绝启动。模型、数据和
代码继续从 `/data` 中的仓库读取。训练完成后由用户统一复制到 NFS。

worker 保留以下保护：

- 存在 `metrics_compact.json` 时跳过完成任务；
- 检测到相同 `--out` 进程时跳过正在运行任务；
- 输出目录存在但未完成时拒绝覆盖；
- 每个任务单独保存日志，并记录 START、DONE、SKIP、ERROR 状态。

## 6. 表格修正

LaTeX 表格保留现有标题和指标列，只修正行数与 Ours 标记：

```latex
\multirow{8}{*}{Selector}
\multirow{6}{*}{Smoothing}
\multirow{2}{*}{Reviewing strategy}
\multirow{5}{*}{Query budget}
```

Smoothing 行使用：

```latex
Fixed local-Gaussian, $\alpha=0.10$ (\textbf{Ours})
Adaptive entropy smoothing
```

第一轮所有结果均为 `seed=42` 的单次结果，表格暂不写 mean ± standard deviation。

## 7. 验证要求

不运行完整训练测试。实现需要验证：

- percentage budget 的舍入和比例计算；
- 五个预算在 Alpaca、Dolly 的实际候选池上解析正确；
- 36 个 job 名唯一，且不包含 Ours；
- 每个区块只覆盖声明的消融因素；
- 数据、模型、LoRA、四阶段和 final-only 参数保持一致；
- Reviewing control 确实关闭 Stage 4；
- Adaptive control 启用 adaptive entropy 且不启用 trainable alpha；
- 本地输出与 NFS 防护有效；
- Bash 语法检查通过。
