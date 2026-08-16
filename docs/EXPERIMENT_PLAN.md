# Experiment Plan / 实验计划

Last updated / 最后更新：2026-07-24

## 1. Research Objective / 研究目标

系统评估在有限查询预算下，使用 surrogate model 主动选择评审样本并学习
victim judge/reward model 的效果。实验覆盖 LLM-as-a-Judge 与 Reward Model 两类
victim，并分析 surrogate 架构、selector、smoothing、评审形式和预算的影响。

核心问题包括：

- 主动选样是否优于随机选样，并提高 surrogate 对 victim 偏好的拟合能力。
- CoT 与 non-CoT judge 反馈对攻击效果有何影响。
- 方法能否迁移到不同数据集、victim model 和 surrogate model。
- selector、smoothing、pointwise/pairwise reviewing 分别贡献多少性能。
- 攻击成本、潜在防御方法及自适应防御下的效果如何。

## 2. Experimental Setup / 实验设置

### 2.1 Datasets / 数据集

| Task | Dataset | Status / 状态 |
|---|---|---|
| LLM-as-a-Judge | Alpaca | Primary / 主数据集 |
| LLM-as-a-Judge | Dolly | In progress / 进行中 |
| Reward Model | Alpaca | Planned / 待实验 |
| Reward Model | Dolly | TBD |

Dolly 当前数据：

- Train: `train_with_selector/train_with_selector/data/Dolly/gpt4all_pointwise_pairwise_train9k.json`
- Validation: `train_with_selector/train_with_selector/data/Dolly/gpt4all_pointwise_pairwise_listwise_val3k.json`

数据集间应尽量采用相同的数据规模、预算定义、划分策略和评价口径。若某数据集
无法提供完全相同的标注形式，需要在结果表中明确说明差异。

### 2.2 Victim Models / 目标模型

#### LLM-as-a-Judge

Proprietary models / 闭源模型：

- GPT-5
- Claude

Open-source models / 开源模型：

- Qwen3-235B-A22B-Instruct
- DeepSeek-V4-Pro

#### Reward Models

- ArmoRM-Llama3-8B-v0.1
- Skywork-Reward-V2 (27B)

所有 victim 实验应记录准确的模型版本、checkpoint/API 版本、调用日期、解码
参数和评审 prompt。闭源模型还应保存请求量、token 用量和费用。

### 2.3 Surrogate Models / 代理模型

主实验 surrogate：

- Llama-3.1-8B
- Qwen3-8B

多样化 surrogate 实验：

| Model | Full/standard training | LoRA |
|---|:---:|:---:|
| Qwen3-0.6B | Planned | Planned |
| Qwen3-1.7B | Planned | Planned |
| Qwen3-4B | Planned | Planned |
| Qwen3-8B | Planned | Planned |
| Qwen3-14B | Planned | Planned |
| Qwen3-30B-A3B | Planned | Planned |

“Full/standard training”在启动实验前需要进一步冻结定义：全参数训练或项目默认的
非 LoRA 训练方式。两种设置必须使用一致的数据、预算和评测集。

## 3. Main Experiments / 主实验

### 3.1 LLM-as-a-Judge

在 Alpaca 上完成下列完整组合，并在 Dolly 上复现实验：

- Victim model：GPT-5、Claude、Qwen3-235B-A22B-Instruct、DeepSeek-V4-Pro。
- Surrogate model：Llama-3.1-8B、Qwen3-8B。
- Feedback mode：non-CoT、CoT。
- Selection method：proposed selector、random baseline。

non-CoT 仅要求 victim 输出最终分数、偏好或排序。CoT 要求先产生评审推理，再
输出可严格解析的最终标签。两种模式应采用相同样本、预算和评价指标。

### 3.2 Reward Model

在 Alpaca 上比较：

- ArmoRM-Llama3-8B-v0.1。
- Skywork-Reward-V2 (27B)。
- Llama-3.1-8B 与 Qwen3-8B surrogate。
- Proposed selector 与 random baseline。
- non-CoT / scalar reward feedback。

以下规则仅适用于 Skywork、ARMO 等具有 grouped continuous rewards 的 Reward
Model 类数据集，不覆盖 LLM-as-a-Judge、Alpaca/Dolly classifier-head 或生成式
三阶段实验。Reward Model pointwise 主实验直接使用训练中的 proxy 进行主动
选样，不再使用 BERT/Longformer selector。Proxy acquisition 只使用 MC-dropout
predictive uncertainty（`uncertainty_weight=1`、`response_std_weight=0`），不做
随机探索（`exploration_ratio=0`），并对全部剩余未标注样本打分，不限制候选池
大小（`--selector-max-score-candidates 0`）。连续 reward 回归没有分类概率熵，
这里的纯 entropy 对应 MC predictive standard deviation；它与高斯预测熵的排序
单调等价。

Smoothing 按选样粒度设置：question-level 的 proxy 与 random 都必须分别运行
`alpha=0` 和 `alpha=0.01`，形成 matched no-smooth/smooth 对比；answer-level
的 proxy 与 random 只运行 `alpha=0`。Answer-level 主动方法应实现为
`proxy_answer`，不使用现有的 BERT-based `selector_answer`。除明确标记的消融
实验外，后续 Reward Model pointwise 实验均遵循此协议。

Dolly Reward Model 实验暂标记为 TBD，在 Alpaca 协议稳定后决定是否加入。

### 3.3 Diverse Surrogate Models on Alpaca

固定数据集、victim、预算、selector 和评测集，仅改变 Qwen3 surrogate 的模型规模
与训练方式。至少报告：

- 任务性能随参数规模的变化。
- LoRA 与非 LoRA 设置的性能差异。
- 训练时间、峰值显存和存储开销。
- 单位性能提升对应的计算成本。

## 4. Ablation Studies / 消融实验

### 4.1 Impact of Selector / Selector 的影响

比较 proposed selector 与 random selection，并根据论文需要加入已有的 BERT、
multi-target、Frozen-Llama 或 two-stage selector。所有对比必须固定 surrogate、
victim、预算、训练样本数、训练阶段和 seed。

当前 Reward Model pointwise 主实验中的 proposed selector 特指 direct proxy
selector；BERT/Longformer 仅在明确的历史对照或 selector 消融中使用，不进入
默认主实验矩阵。Direct proxy 默认使用纯 MC predictive uncertainty、不做随机
探索，并使用全量未标注候选池。

### 4.2 Impact of Smoothing / Smoothing 的影响

固定完全相同的 selected samples，对比 no smoothing 与不同 smoothing alpha。
当前重点候选为：

- `alpha=0.00`：无 smoothing 基线。
- `alpha=0.01`：当前 pointwise/pairwise 综合候选。
- `alpha=0.04`：当前 listwise-oriented 候选。

正式结论应至少使用多个训练 seed，并报告均值和标准差。

Reward Model pointwise 主实验的默认规则是：question-level 的 random/proxy
均运行 `alpha=0` 与 `alpha=0.01` 对比；answer-level 的 random/proxy 仅运行
`alpha=0`。报告 smooth 效果时必须使用相同选择方式、预算、seed 和其他超参数，
并清楚说明 smoothing 是否影响主动选样阶段。

### 4.3 Pointwise vs. Pairwise Reviewing / 评审形式

在相同 answer-unit budget 下比较：

- Pointwise reviewing。
- Pairwise reviewing。
- Pointwise + pairwise reviewing。
- 如作为论文任务保留，再加入 listwise reviewing。

需要同时报告原始 review 数量和 answer-unit 消耗，避免不同评审形式的单次成本
差异造成不公平比较。

### 4.4 Performance under Different Budgets / 不同预算

以 answer units 作为统一预算单位。项目当前标准预算为 `600`，预算曲线应在协议
冻结后选取低、中、高多个点，并在每个预算下保持 selector 初始化比例和 query
batch 比例一致。

## 5. Discussion / 讨论与扩展实验

### 5.1 Cost Analysis / 成本分析

报告 victim 查询次数、输入/输出 token、API 费用、训练 GPU hours、峰值显存、
模型存储和端到端耗时。闭源与开源 victim 应分别统计。

### 5.2 Defense Methods / 防御方法

负责人：Lin Juan。需要明确 threat model、防御可访问的信息、效用损失和攻击成功率
之间的权衡。

### 5.3 Adaptive Defense / 自适应防御

在攻击者了解防御机制并可调整 selector/训练策略的条件下重新评估。自适应攻击
必须沿用与非自适应实验相同的预算核算方式。

### 5.4 Theoretical Analysis / 理论分析

围绕主动选样目标、有限预算下的样本效率、反馈噪声和 smoothing 的作用建立分析。
理论假设应与实际 selector 和训练目标对应。

### 5.5 Visualization / 可视化

计划生成：

- 性能随查询预算变化的曲线。
- 性能随 surrogate 参数规模变化的曲线。
- Selector 与 random 的样本分布对比。
- Smoothing alpha 的性能曲线。
- Pointwise/pairwise/listwise 预测混淆矩阵与错误案例。
- 成本—性能 Pareto 图。

### 5.6 Limitations / 局限性

重点讨论 API 模型版本漂移、judge bias、数据集覆盖范围、单一 prompt 模板、计算
资源限制、开闭源 victim 可比性以及防御实验的适用范围。

## 6. Unified Protocol / 统一实验协议

除明确消融外，当前三阶段生成式实验采用：

1. Stage 1：pointwise training。
2. Stage 2：pairwise training，pointwise replay ratio 为 `0`。
3. Stage 3：listwise training，pointwise/pairwise replay ratio 均为 `1`。
4. Budget：`600` answer units，即选择 200 个 answer triples。
5. 默认 seed：`42`；论文核心结果需要补充多个 seed。
6. 默认对照：proposed selector 与 random selection。
7. Eval parse failure 作为 invalid 且计错，不自动视为 tie。

标准三阶段参数：

```bash
--budget-units 600 \
--stage2-pointwise-replay-ratio 0 \
--stage3-pointwise-replay-ratio 1 \
--stage3-pairwise-replay-ratio 1
```

所有实验至少记录：dataset/version、victim、surrogate、feedback mode、selection
method、budget、seed、训练方式、checkpoint、prompt、超参数、输出目录和代码版本。

## 7. Evaluation Metrics / 评价指标

Pointwise：

- Exact accuracy。
- Within-1 accuracy。
- Mean absolute error (MAE)。
- Invalid rate。

Pairwise：

- Accuracy。
- Macro-F1，尤其关注 tie 类别。
- Tie precision/recall/F1 与预测 tie rate。
- Invalid rate。

Listwise：

- Exact ranking accuracy。
- Pairwise-relation accuracy。
- Ranking MAE 或等价的排名距离。
- Invalid rate。

主表应报告多 seed 的 mean ± standard deviation，并在适用时进行显著性检验。

## 8. Execution Order / 执行顺序

1. 冻结数据划分、prompt、预算、解析规则、指标和模型版本。
2. 完成 Alpaca 上 LLM-as-a-Judge non-CoT 的最小主表。
3. 补齐 CoT 主实验。
4. 完成 Alpaca Reward Model 主实验。
5. 在 Dolly 上复现核心方法与基线。
6. 完成 selector、smoothing、reviewing 和 budget 消融。
7. 完成 Qwen3 diverse-surrogate 矩阵。
8. 汇总 cost、defense、adaptive defense、theory、visualization 和 limitations。

## 9. Result Management / 结果管理

- 实验启动与结果继续按日期写入 `WORK_LOG.md`。
- 稳定规则和长期结论写入 `PROJECT_MEMORY.md`。
- 本文档只维护总体实验矩阵、统一协议和完成状态。
- 每个实验输出目录应保留 `config.json`、指标、selected samples、统计文件和日志。
- 不得用新实验覆盖旧输出；目录名应包含 dataset、victim、surrogate、method、
  budget、seed 和关键消融设置。

## 10. Open Decisions / 待确认事项

- Listwise 是主实验任务、辅助训练目标，还是仅作为消融。
- “非 LoRA”Qwen3 surrogate 的确切训练定义。
- GPT-5、Claude 与 DeepSeek 的具体版本/API endpoint。
- Alpaca 数据集的正式版本及其与当前 `newnew` 数据的对应关系。
- Dolly 是否进入 Reward Model 主实验。
- 主实验预算点与重复 seed 集合。
