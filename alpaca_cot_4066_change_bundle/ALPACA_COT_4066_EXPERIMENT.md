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

## 13. 逐文件代码改动说明（不含新启动脚本）

本节只说明本次为 Alpaca CoT 4066 实验新增或修改的 Python、测试和文档代码，`launch_alpaca_cot_stage4_mix.sh` 不在本节范围内。

### 13.1 新增 `prepare_alpaca_cot_4066.py`

这是本次新增的数据准备入口。原项目没有处理“顶层 JSON 数组在一条记录中间截断”的逻辑，也没有针对这份 CoT 验证集建立共享的 Mix/eval 切分。

#### `_recover_json_array_prefix`

原始文件不能直接执行 `json.load`。新函数使用 `json.JSONDecoder.raw_decode` 从 `[` 后逐对象解码：

1. 跳过空白和逗号；
2. 每次只解码一个完整对象；
3. 遇到第一个不完整对象时停止；
4. 保留此前所有完整对象；
5. 记录停止字符位置和 `JSONDecodeError`，但不修改源文件。

因此恢复的是 4066 条可完整解码的记录，而不是尝试修补第 4067 条不完整记录。

#### `_validate_pointwise_record` 与 `_score`

增加了准备阶段的强校验：

- instruction 必须非空；
- A/B/C 三个回答必须存在且非空；
- `pointwise.A/B/C` 必须存在；
- score 必须可以转换为整数且位于 1--10；
- pointwise reason 必须非空。

这样错误会在生成训练切分前暴露，而不是在 GPU 训练过程中才失败。

#### `_normalize_question`

把原数据中的：

```text
answerA / answerB / answerC
pointwise.A.score / pointwise.A.reason
```

规范为现有 triple loader 可读取的扁平格式：

```text
outputA / outputB / outputC
scoreA / scoreB / scoreC
reasonA / reasonB / reasonC
```

同时保留 `id`、`source_id`、dataset、instruction、input 和 model 字段。

#### `_permuted_validation_record`

新增验证集回答位置随机化。函数接收 `new position -> old position` 的 permutation，并同时转换所有依赖位置的字段。

Pointwise 直接随回答搬移 score/reason。Pairwise 会通过 `_pair_value` 判断新 pair 对应旧 AB/AC/BC 中哪一组；如果左右顺序反转，则：

```text
choice 1 <-> choice 2
choice 3 保持不变
Assistant 1 <-> Assistant 2
```

Listwise 通过 `old_to_new` 映射替换 A/B/C，并用 `_map_ranking` 规范 tie group。例如排列后产生的 `C>B=A` 会规范为 evaluator 支持的 `C>A=B`。

#### `_select_mix_examples`

对固定选出的每个 Mix 源问题：

- 随机取一个 pointwise position；
- 随机取一个 pair（AB/AC/BC）；
- 保留一个完整 listwise 样本。

三类样本使用同一组 200 source IDs。剩余 1800 IDs 用于所有评测任务。

#### `manifest.json`

新增可复现信息：

- seed；
- 原训练/验证文件 SHA-256；
- 截断恢复位置与异常；
- 所有输出样本数；
- train/validation 和 Mix/eval 泄漏检查；
- Mix 标签直方图；
- 位置随机化和标签同步变换设置。

### 13.2 修改 `run_pointwise5answers_two_to_pairwise_v1.py`

这个文件是现有通用 pointwise/pairwise SFT 和 loss 实现。本次没有重写 trainer，而是在保持旧无 CoT 路径兼容的基础上补齐 CoT 所需元数据。

#### `AnswerWithScore` 增加 `reason`

原结构只有：

```python
model, output, score
```

现在增加：

```python
reason: str = ""
```

使用空字符串默认值，因此以前只构造三个字段的代码不需要修改。

#### `PointwiseScoredExample` 增加 `reason`

同样增加：

```python
reason: str = ""
```

用于把 pointwise CoT 从 loader 传递到 SFT target builder，同时保持旧调用兼容。

#### `_load_scored_questions` 传递 reason

loader 原来主要读取 model/output/score。现在在以下输入格式中同时读取 reason：

- `models` 列表中的 `reason`；
- `reason1` ... `reason5`；
- `reasonA` ... `reasonE`。

最后构造：

```python
AnswerWithScore(
    model=model_name,
    output=output,
    score=score,
    reason=reason,
)
```

这是 `train_questions_4066.json` 中 pointwise CoT 能进入 Stage 1 的数据通路。

#### `_pointwise_sft_target` 增加 `cot_feedback`

旧 target 只生成 score：

```text
Score: [X]</s>
```

现在增加可选参数 `cot_feedback=False`。启用且 reason 非空时生成：

```text
{reason}
Score: [X]</s>
```

默认仍为 `False`，所以旧实验不会自动变成 CoT 实验。本次独立 runner 也直接使用与历史 pointwise CoT 脚本相同的 target 格式。

#### `SFTPairwiseDataset` 增加 score token 定位

旧 score smoothing 默认把 target 的第一个非 `-100` token 当作 score token。对无 CoT target 这通常成立，但对：

```text
{reason tokens ...}
Score: [7]
```

第一个 target token 是 reason，不是分数。如果继续使用旧位置，平滑 loss 会错误作用在 CoT 第一个 token 上。

现在构造 dataset 时可传入：

```python
pointwise_score_token_ids: Sequence[Sequence[int]]
```

对每条有效 pointwise 样本，dataset 会：

1. 根据 `pointwise_score_label` 取得真实 score 的 tokenizer 序列；
2. 只在 target labels 中搜索该 token 序列；
3. 保存最后一次匹配位置，避免 reason 前面偶然出现相同数字时定位错误；
4. 找不到时立即抛错，避免静默训练错误；
5. 对 pairwise/listwise 样本保留 `IGNORE_INDEX`。

之所以支持 token 序列而不是单 token，是为了兼容某些 tokenizer 将 `10` 编码为多个 token 的情况。

#### `__getitem__` 与 `_data_collator_sft`

dataset item 新增：

```python
pointwise_score_position
```

collator 将其收集为：

```python
pointwise_score_positions: LongTensor[batch]
```

padding 发生在序列右侧，因此已记录的位置不需要额外偏移。

#### `OnlineGlobalPriorSFTTrainer.compute_loss`

trainer 现在从 batch 中取出 `pointwise_score_positions`。有该字段时使用真实 score 位置；没有时退回旧的 `first_label_pos`，保持其他调用者兼容。

改动只改变 smoothing 的锚点，不改变已有 local-Gaussian/global-prior 公式：

- reason token 继续使用 hard token CE；
- 最终 score token 的 hard CE 被 pointwise soft target 按 alpha 混合；
- pairwise/listwise 不带有效 pointwise label，不执行 score smoothing。

### 13.3 修改 `run_pointwise5answers_three_stage_pairwise_listwise_sft_v1.py`

通用 `_train_sft_on_items` 原来只把 `pointwise_score_labels` 传给 `SFTPairwiseDataset`。本次增加：

```python
pointwise_score_token_ids = base._score_token_ids_for_sft(
    tokenizer,
    score_min=cfg.score_min,
    score_max=cfg.score_max,
)
```

并把同一组 token IDs 同时传给：

1. `SFTPairwiseDataset`：用于定位真实 score token；
2. `OnlineGlobalPriorSFTTrainer`：用于构造 1--10 的候选 score soft loss。

这样 Stage 1 和 Stage 4 的 pointwise CoT 都使用正确位置；Stage 2/3 仍走原 pairwise/listwise hard CE。

### 13.4 修改 `train_with_selector/.../data/pairwise_dataset.py`

`build_pairwise_prompt` 原来无条件在 `### Judge` 后追加：

```text
Please output exactly one of: [[1]] / [[2]] / [[3]].
```

这适合旧 label-only prompt，但如果 CoT system prompt 要求“先解释再给 verdict”，这句可能被模型理解成“只能输出一个标签”，形成冲突。

现在增加向后兼容参数：

```python
include_verdict_instruction: bool = True
```

- 旧调用不传参数，行为完全不变；
- 新 CoT training/synthesis prompt 传 `False`；
- 最终输出格式由 CoT system prompt 完整描述。

### 13.5 新增 `run_alpaca_cot_stage4_mix.py`

这是本次实验的主 Python 入口，不包含 shell 启动逻辑。它复用现有 selector、SFT trainer 和 evaluator，只负责组织 CoT 数据流。

#### Prompt 定义

`PAIRWISE_COT_SYSTEM_PROMPT` 从旧 `DEFAULT_PAIRWISE_SYSTEM_PROMPT` 派生，`LISTWISE_COT_SYSTEM_PROMPT` 从旧 `LISTWISE_SYSTEM_PROMPT` 派生。实现保留旧评价规则，只替换 label-only 输出要求。

新增的 `PAIRWISE_SYNTH_SYSTEM_PROMPT` 和 `LISTWISE_SYNTH_SYSTEM_PROMPT` 专门用于根据分数写理由，明确规定：

- 分数定义必须解释的偏好/排序；
- 不允许重新决定标签；
- 使用旧评价维度解释回答内容；
- 不得提及 private score；
- 只输出 reason，不输出最终标签。

#### `_make_cfg`

把独立 CLI 参数转换为现有 `three.RunConfig`，固定本实验核心口径：

- budget 600；
- Stage 1/2/3/4 各 1 epoch；
- pairwise/listwise 全排列增强；
- Stage 4 full replay；
- pointwise local-Gaussian smoothing alpha 0.1；
- 默认 `bias_trap_pointwise` selector；
- 默认复用 lm-head selection proxy 继续 Stage 1 CoT SFT；
- Mix 关闭 smoothing，并把三任务 epoch 设置为 10。

随机 selector 不具有可复用 pointwise proxy，因此选择 `--selector-kind random` 时会自动关闭 proxy reuse。

#### `_cot_pointwise_prompt` 与 `_cot_pointwise_target`

这里没有重新发明 pointwise CoT 格式，而是复用历史：

```python
base.JUDGE_SYSTEM_PROMPT_WITH_EXPLANATION
base.build_judge_prompt(..., fix_score_prefix=False)
```

target 为：

```text
reason\nScore: [score]</s>
```

已用旧 `run_pointwise_cot_sft_train_eval.py` 做函数级对照，prompt 和 target 均一致。

#### `_private_pair_prompt` 与 `_private_list_prompt`

这两个函数构造只用于 inference 的 private synthesis prompt：

```text
公开问题与回答
+ private pointwise scores
+ 可选 pointwise assessments
```

默认 `include_pointwise_assessments=False`，因此默认只使用回答与分数。private prompt 不会被直接放入 Stage 2/3 training item。

#### `_generate_texts`

使用 Stage 1 模型贪心生成 reason：

- `do_sample=False`，保证固定模型/seed 下更稳定；
- 为输出预留 `synthetic_max_new_tokens`；
- 批量生成并只 decode 新生成部分；
- 生成结束后恢复模型原来的 `use_cache` 设置。

#### `_clean_synthetic_reason`

删除代理可能自行生成的：

- `[[1]]/[[2]]/[[3]]`；
- `Ranking:[...]`；
- 开头冗余的 `Explanation:` 或 `Rationale:`。

如果清理后 reason 为空，使用 `_fallback_pair_reason` 或 `_fallback_list_reason`，保证每条合成样本都有可训练理由。

#### `_synthesize_pairwise_listwise_items`

对每个选中 triple：

1. 枚举 6 个有向 pair；
2. 枚举 6 个 listwise permutation；
3. 构造 private synthesis prompt；
4. 调用 Stage 1 模型生成 reason；
5. 清除生成模型自己的标签；
6. 从 pointwise score 计算 canonical label；
7. 构造不含 private score 的公开 training prompt；
8. 拼接 `reason + forced label + EOS`。

同时保存两类数据：

- trainer 使用的 `(task, public_prompt, target, pointwise_label)` tuple；
- 审计 JSONL 使用的 raw generation、private prompt、public prompt、scores 和 forced target。

#### `_pointwise_items_from_mix`、`_mix_pair_items`、`_mix_list_items`

Mix 不生成 synthetic reason：

- pointwise 使用验证集 `judge_reason + judge_score`；
- pairwise 使用验证集显式 reason 和 choice；
- listwise 使用验证集显式 reason 和 ranking。

pair/list reason 中已有的最终标签会先移除，再按准备后数据中的显式 gold label 统一追加，避免 target 出现两个 final label。

#### `_build_eval_sets`

把固定 1800 问题展开为三任务 evaluator 对象：

- 3 个 `PointwiseScoredExample`；
- AB/AC/BC 三个 `PairwiseExample`；
- 1 个 `ListwiseExample`。

评测 pairwise/listwise 使用验证集显式 gold label，不使用 pointwise score 重新推导。CLI 的三个 `--max-*-eval-samples` 可用于小规模 smoke test，0 表示完整评测。

#### `_run_stage4`

组织完整链路：

```text
加载 4066 -> 建候选 -> selector 选 200
-> Stage 1 real pointwise CoT
-> 生成 pair/list synthetic CoT
-> Stage 2 pairwise
-> Stage 3 listwise
-> Stage 4 full replay
-> 三任务统一评测
```

当 selector 返回可复用 proxy 时，取出其 `.model` 和 `.tokenizer` 继续 Stage 1；否则从 `--llama` 加载基础模型。

#### `_run_mix`

强制检查 point/pair/list 三个训练文件各为 200 条，然后按 10 epochs 顺序训练三个真实 CoT 任务，最后在同一 1800 问题评测集上评测。

#### 输出防覆盖

`main` 在 output directory 已存在时抛出 `FileExistsError`，避免意外覆盖未完成或已有实验。配置、prompt policy、训练统计、合成审计和最终指标都会写入输出目录。

### 13.6 新增 `tests/test_alpaca_cot_stage4_mix.py`

新增测试没有加载大模型，使用轻量 character tokenizer 和最小对象验证关键逻辑：

1. 位置变换后的 tie ranking 会 canonicalize；
2. synthetic reason 中模型自行生成的 final label 会被删除；
3. reason 里提前出现同一个数字时，score smoothing 仍定位最后的真正 score；
4. private synthesis prompt 包含分数，但 public training prompt 不包含；
5. 默认不向 synthesizer 提供 pointwise assessment；
6. 打开消融开关后 assessment 会进入 private prompt；
7. pairwise/listwise CoT prompt 保留旧 position/length bias 规则；
8. pairwise CoT prompt 不再附加与“先解释”冲突的 label-only 指令。

测试与原有 reward-model/skywork 测试一起执行，当前为 `13 passed`。

### 13.7 `.gitignore`、`README.md` 与本文档

`.gitignore` 增加本地 `Alpaca-cot-gpt` 数据目录，避免把原始数据、恢复数据和固定切分提交到 Git。代码和文档仍正常跟踪。

`README.md` 增加本文档入口；本文档是本次实验的单一说明来源。

### 13.8 明确没有改变的历史行为

为保证以前实验仍可复现，本次最终版本没有改变以下语义：

- `JUDGE_SYSTEM_PROMPT_WITH_EXPLANATION` 的历史 pointwise CoT 内容；
- `JUDGE_SYSTEM_PROMPT_SCORE_ONLY` 的历史无 CoT 内容；
- 旧 pointwise CoT target：`reason + Score`；
- 旧 pairwise builder 的默认 label-only 行为；
- 旧 listwise label 集合及 evaluator 的 13 种 canonical ranking；
- 现有 smoothing 分布公式。

所有新行为都由独立 runner、显式 CoT prompt 或向后兼容的可选参数触发。
