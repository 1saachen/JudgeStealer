# Alpaca CoT 4066：四阶段合成 CoT 与 Mix 对照实验

本文档集中说明本次针对 `Alpaca-cot-gpt` 数据完成的数据恢复、固定切分、prompt、训练目标、四阶段实验、Mix 对照、评测口径和运行方法。实现入口为：

- `prepare_alpaca_cot_4066.py`：恢复数据并建立固定训练/评测切分；
- `run_alpaca_cot_stage4_mix.py`：四阶段与 Mix 的统一训练、评测入口；
- `launch_alpaca_cot_stage4_mix.sh`：单卡顺序启动四阶段和 Mix；
- `tests/test_alpaca_cot_stage4_mix.py`：本实验新增的关键行为测试。

## 1. 实验问题

现有训练集只有 pointwise 分数与 pointwise CoT，不能直接获得真实 pairwise/listwise CoT。本实验验证以下问题：

> 在只有 pointwise 查询结果的条件下，能否让 pointwise 代理依据回答内容和查询到的 pointwise 分数生成近似 pairwise/listwise CoT，再用这些合成监督训练四阶段 judge？

核心原则是：

1. 代理只负责生成 CoT 理由，不决定最终标签；
2. pairwise/listwise 最终标签始终由 pointwise 分数确定；
3. 合成时使用的 private 分数不进入后续训练 prompt，避免显式分数泄漏；
4. Mix 使用验证集中真实的 pointwise/pairwise/listwise CoT，作为真实 CoT 对照。

## 2. 数据恢复

原始训练文件：

```text
train_with_selector/train_with_selector/data/Alpaca-cot-gpt/
  Alpaca-cot-gpt/train_pointwise_8k.json
```

该文件在第 4067 条记录中间截断，不是合法的完整 JSON 数组。准备脚本不修改原文件，而是用流式 JSON decoder 读取其合法前缀。

恢复结果：

| 项目 | 数量 |
|---|---:|
| 完整训练问题 | 4066 |
| 每题回答数 | 3 |
| pointwise 训练回答 | 12198 |
| 非空 pointwise 理由 | 12198 |
| 重复训练 ID | 0 |
| 与原验证集 ID 重叠 | 0 |

恢复与切分使用固定 seed `42`。生成文件位于：

```text
train_with_selector/train_with_selector/data/Alpaca-cot-gpt/prepared_4066/
```

此目录属于本地派生数据，已加入 `.gitignore`。

## 3. Mix 与固定评测切分

原验证集包含 2000 个问题，每题具有：

- 3 条 pointwise 分数和理由；
- AB、AC、BC 三组 pairwise 标签和理由；
- 1 条 listwise 排序和理由。

用 seed `42` 固定抽取同一批 200 个源问题用于 Mix。对每个 Mix 问题：

- 随机选择 A/B/C 中一条，得到 1 条真实 pointwise CoT；
- 随机选择 AB/AC/BC 中一组，得到 1 条真实 pairwise CoT；
- 使用完整三回答排序，得到 1 条真实 listwise CoT。

因此 Mix 数据为：

| 任务 | 训练样本 |
|---|---:|
| Pointwise real CoT | 200 |
| Pairwise real CoT | 200 |
| Listwise real CoT | 200 |

这 200 个问题从所有评测任务中同时移除。剩余统一评测集为：

| 任务 | 评测样本 |
|---|---:|
| Pointwise | 5400（1800 × 3） |
| Pairwise | 5400（1800 × 3） |
| Listwise | 1800（1800 × 1） |

Mix 与评测集 source ID 交集为 0。

### 3.1 回答位置随机化

原始 2000 条验证数据存在明显位置偏差：

- pairwise 左侧获胜率为 80.83%；
- listwise `A>B>C` 比例为 67.45%。

因此 Mix 和评测数据都对 A/B/C 位置进行固定随机排列，并同步转换：

- pointwise 分数与理由；
- pairwise 左右标签与理由；
- listwise 排序与理由。

并列组会规范为现有 evaluator 接受的 13 种 canonical ranking，例如 `C>B=A` 会写成等价的 `C>A=B`。

## 4. 历史 CoT 代码与本次新增部分

### 4.1 历史 pointwise CoT 保持不变

项目原来已经有 pointwise CoT SFT：

```text
archive/legacy_python_20260816/run_pointwise_cot_sft_train_eval.py
```

旧代码使用 `JUDGE_SYSTEM_PROMPT_WITH_EXPLANATION`，训练目标为：

```text
{judge_reason}
Score: [{judge_score}]
```

本次 Stage 1 复用同一个 system prompt、同一个 `build_judge_prompt` 和同一个 target 格式。自动对照结果为：

```text
pointwise_prompt_identical = True
pointwise_target_identical = True
```

历史无 CoT 的 `JUDGE_SYSTEM_PROMPT_SCORE_ONLY` 也保持原样，避免改变以前 baseline 的实验定义。

### 4.2 Pairwise/listwise CoT prompt

旧 pairwise/listwise prompt 是无 CoT 模板，只要求输出最终标签。本次 CoT 版本从旧模板派生，保留以下内容：

- helpfulness、relevance、accuracy、depth、creativity、level of detail；
- position bias 约束；
- length bias 约束；
- tie 规则；
- 原有最终标签格式。

唯一的任务级变化是：从“只输出标签”改为“先给简短解释，最后输出标签”。

Pairwise target：

```text
{reason}
[[1]]                 # 或 [[2]] / [[3]]
```

Listwise target：

```text
{reason}
Ranking:[A>B=C]       # 13 种 canonical ranking 之一
```

## 5. 根据 pointwise 分数生成 CoT

### 5.1 默认合成输入

代理生成 pairwise/listwise CoT 时会看到：

- 用户 instruction/input；
- 两个或三个候选回答；
- private pointwise 分数。

默认不把原始 pointwise 理由直接提供给合成器，因此生成的 CoT 是代理依据回答内容和分数自己写出的。可选参数：

```text
--include-pointwise-assessments-in-synthesis
```

可用于“分数 + 原 pointwise 理由”的增强消融；默认关闭。

### 5.2 合成 prompt 的约束

Pairwise 合成 prompt 明确规定：

- 高分回答必须被描述为更强；
- 分数相等时应描述为质量相近；
- 不允许代理推翻分数定义的偏好；
- 理由应根据回答内容和旧 prompt 的完整评价维度生成；
- 不得提及 private 分数；
- 不得输出 `[[1]]/[[2]]/[[3]]`。

Listwise 同理：pointwise 分数定义完整排序和 tie，代理只解释该排序。

### 5.3 强制标签

即使代理违反指令并输出了 provisional verdict/ranking，代码也会先删除它，再追加由 pointwise 分数计算的标签。

Pairwise：

```text
score_left > score_right  -> [[1]]
score_left < score_right  -> [[2]]
score_left = score_right  -> [[3]]
```

Listwise 按分数从高到低排列，相同分数使用 `=`。并列组内部按 A/B/C 规范化。

合成 prompt 会写入审计 JSONL，但 private 分数不会进入 Stage 2/3/4 的训练 prompt。

### 5.4 转换标签的已知噪声

在原始 2000 问题验证集上：

- 分数转换的 pairwise 标签与显式 pairwise 标签一致率为 96.50%；
- 分数转换的 listwise 排序与显式 listwise 排序完全一致率为 90.85%。

因此该方法是有数据支持的弱监督转换，但不等价于真实 pairwise/listwise 标注。Mix 保留验证集显式标签，不用 pointwise 分数覆盖真实标签。

## 6. 四阶段实验

预算按 pointwise answer query 计：

```text
budget = 600 answers = 200 answer triples
```

默认 selector 为 `bias_trap_pointwise`：

- 初始化 80 个 triple；
- 每轮选择 20 个 triple；
- 每轮候选池 100；
- pointwise proxy warmup 3 epochs；
- 每轮增量更新 1 epoch；
- 最终选择 200 个不同问题的 triple。

默认复用选择阶段的 pointwise proxy，继续进行 Stage 1 CoT SFT，而不是重新加载一个无关模型。

### Stage 1：真实 pointwise CoT

200 个 triple × 3 个回答：

```text
600 real pointwise CoT examples × 1 epoch
```

Stage 1 使用训练集原始 pointwise reason 和 score。

### CoT 合成

Stage 1 后的模型依据回答与 pointwise 分数生成：

- 每个 triple 的 6 个有向 pairwise 顺序；
- 每个 triple 的 6 个 listwise 排列。

### Stage 2：合成 pairwise CoT

```text
200 × 6 = 1200 synthetic pairwise CoT examples × 1 epoch
```

### Stage 3：合成 listwise CoT

```text
200 × 6 = 1200 synthetic listwise CoT examples × 1 epoch
```

### Stage 4：完整 consolidation replay

```text
600 pointwise + 1200 pairwise + 1200 listwise
= 3000 examples × 1 epoch
```

四阶段总 SFT exposure：

```text
600 + 1200 + 1200 + 3000 = 6000
```

主动选择期间临时 proxy 的 warmup/update 计算不计入最终 SFT exposure；这与项目现有 selector 实验的记账方式一致。

## 7. Mix 对照

当前实现使用上述固定 200 个验证问题的真实 CoT，按以下顺序训练：

1. 200 pointwise real CoT × 10 epochs；
2. 200 pairwise real CoT × 10 epochs；
3. 200 listwise real CoT × 10 epochs。

总 exposure：

```text
(200 + 200 + 200) × 10 = 6000
```

因此当前代码中的 Mix 是“相同 200 源问题、三类真实 CoT、三阶段顺序训练”的对照，并非把 600 条样本打乱后进行单阶段 multitask 混合。如果后续需要标准的单阶段 simultaneous Mix，应作为独立实验模式实现，不能与当前结果混用。

## 8. CoT 的训练损失

当前不是“CoT 只作上下文、只训练最终标签”，而是真正的 CoT SFT。

对每个训练样本：

- prompt token：label 为 `-100`，不计算 loss；
- CoT reason token：计算 causal-LM cross entropy；
- 最终 score/verdict/ranking token：计算训练 loss；
- EOS：保留在 target 中。

Pointwise 启用 `alpha=0.1` 的 local-Gaussian score smoothing 时：

- CoT reason 仍使用 hard CE；
- 只有最终 `Score: [X]` 中实际分数 token 使用平滑目标；
- 数据集会定位 target 末尾的真实 score token，而不是把 CoT 的第一个 token 或理由中出现的数字误当作分数。

Pairwise/listwise 的 CoT 与最终标签当前都使用 hard CE。

## 9. 运行方法

### 9.1 重新生成固定数据

```bash
python prepare_alpaca_cot_4066.py
```

### 9.2 单独运行四阶段

```bash
CUDA_VISIBLE_DEVICES=0 python -u run_alpaca_cot_stage4_mix.py \
  --mode stage4 \
  --llama llama/Llama-3.2-1B-Instruct \
  --out outputs/llama3p2_1b_alpaca_cot_4066_stage4_cot_proxy_v1
```

### 9.3 单独运行 Mix

```bash
CUDA_VISIBLE_DEVICES=0 python -u run_alpaca_cot_stage4_mix.py \
  --mode mix \
  --llama llama/Llama-3.2-1B-Instruct \
  --out outputs/llama3p2_1b_alpaca_cot_4066_mix_cot_proxy_v1
```

### 9.4 顺序运行二者

```bash
bash launch_alpaca_cot_stage4_mix.sh \
  0 llama/Llama-3.2-1B-Instruct both cot_proxy_v1
```

启动脚本也可接受 `stage4` 或 `mix`，只运行其中一个：

```bash
bash launch_alpaca_cot_stage4_mix.sh 0 qwen/Qwen3-1.7B stage4 qwen_cot_proxy_v1
```

## 10. 输出与审计文件

四阶段输出目录包括：

```text
config.json
dataset_load_stats.json
candidate_stats.json
candidate_triples.jsonl
selection_stats.json
selected_triples.jsonl
pointwise_cot_train.jsonl
synthetic_cot_stats.json
synthetic_pairwise_cot.jsonl
synthetic_listwise_cot.jsonl
train_stats_stage1.json
train_stats_stage2.json
train_stats_stage3.json
train_stats_stage4.json
metrics_pointwise_after_stage4.json
metrics_pairwise_after_stage4.json
metrics_listwise_after_stage4.json
metrics_compact.json
summary.json
```

合成 JSONL 同时保存 raw generation、private synthesis prompt、公开 training prompt 和强制后的 target，便于检查 CoT 是否与标签矛盾。

Mix 输出对应的三个阶段训练统计、三任务评测指标、`metrics_compact.json` 和 `summary.json`。

## 11. 验证

运行：

```bash
PYTHONPATH="$PWD" pytest -q \
  tests/test_alpaca_cot_stage4_mix.py \
  tests/test_rewardmodel_pointwise.py \
  tests/test_skywork_dataset.py
```

当前结果：

```text
13 passed
```

覆盖内容包括：

- tie ranking 在位置变换后的 canonicalization；
- 删除代理自行输出的最终标签；
- CoT 训练时定位 target 末尾分数 token；
- private 分数不进入公开训练 prompt；
- 默认 synthesis 不读取原 pointwise reason；
- CoT pairwise/listwise prompt 保留旧评价规则。

此外已完成数据级 dry-run 检查：旧 pointwise CoT prompt/target 与新 Stage 1
一致；四阶段样本数为 `600/1200/1200/3000`；Mix 为 `200/200/200`；
评测为 `5400/5400/1800`；Mix/eval source ID 交集为 0。

## 12. 解释结果时的限制

1. 1200 pairwise 和 1200 listwise 包含同一批 200 问题的顺序增强，不是完全独立的新标注；
2. 四阶段与 Mix 虽然都是 6000 exposure，但数据多样性、阶段数和学习率调度重启次数不同；
3. synthetic CoT 可能出现理由与强制标签不一致，应结合审计 JSONL 统计冲突率；
4. 分数转换本身存在约 3.5% pairwise 和 9.15% listwise 显式标签不一致；
5. 当前实验适合支持“pointwise 监督能否扩展为结构化弱监督”的结论，不能直接把 synthetic CoT 当作真实 pairwise/listwise CoT。

建议至少保留以下消融：

- score-derived label only，不训练 CoT；
- score-derived label + proxy synthetic CoT；
- score + 原 pointwise assessment 的 synthetic CoT；
- validation real-CoT Mix；
- 可选的 simultaneous 600-example Mix。
