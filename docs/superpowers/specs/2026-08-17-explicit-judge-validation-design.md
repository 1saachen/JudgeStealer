# 显式 Judge 验证标签与字段兼容设计

## 目标

让现有三阶段 SFT 训练入口直接读取已经解压的 GPT-5、Claude Alpaca 和
Claude GPT4ALL JSON，并在显式传入 pairwise 验证文件时只使用数据中已有的
judge 金标，不把缺失的 AC 选择解释为 tie。

## 范围

本次修改覆盖：

- GPT-5 的 `outputA/outputB/outputC` 与 Claude 的
  `answerA/answerB/answerC` 字段兼容；
- 外部 pairwise 验证的显式标签读取；
- 外部 listwise 验证的回答字段兼容；
- 聚焦的数据 loader 回归测试与修改后的数据一致性报告。

训练脚本仍只读取普通 JSON，不直接读取 ZIP，也不改写原始 judge 导出文件。
训练阶段从 pointwise score 构造 pairwise/listwise 训练标签的现有逻辑保持不变。
三阶段入口和现有启动脚本不在修改范围内；新的启动脚本由用户单独编写。

## 规范化边界

字段规范化发生在 loader 入口。内部继续使用现有 `output`/`model` 数据结构：

- 回答文本按顺序查找 `outputA`、`output_a` 等现有别名，最后兼容
  `answerA`；B、C 同理；
- Claude 文件缺少模型名时使用稳定的位置默认值 `A`、`B`、`C`；
- score 仍使用现有 `scoreA/scoreB/scoreC` 及下划线别名；
- 不生成第二份规范化数据，也不修改 `data/` 下的源 JSON。

该兼容同时应用于训练问题、显式 pairwise 验证和显式 listwise 验证，避免三个
入口各自形成不同的字段规则。

## 外部验证口径

### Pointwise

pointwise 验证继续使用训练数据内部划分记录上已有的 score。它没有从其他任务
推导标签，因此本次不改变其语义。

### Pairwise

本次运行的新启动脚本必须同时传入 `--pairwise-eval-dataset` 和
`--listwise-eval-dataset`。三阶段入口已有的可选 fallback 保持不变，以免改变
旧实验命令；只要显式传入 pairwise 路径，就会走显式 pairwise loader。

显式 pairwise loader 支持现有别名：

- AB：`choice_AB`、`choiceAB`、`pairwise_ab_choice`；
- BC：`choice_BC`、`choiceBC`、`pairwise_bc_choice`；
- AC：`choice_AC`、`choiceAC`、`pairwise_ac_choice`。

只有存在显式选择的 pair 才生成样本。缺少 AC 的记录只生成 AB 和 BC，不把
缺失值解释为 tie。空白选择计入 `skipped_missing_choice`；无法识别的非空选择
抛出 `ValueError`。选择 `3` 或明确的 `tie` 仍表示真实平局。

因此 2,000 条仅含 AB/BC 标注的 Alpaca 验证记录应精确生成 4,000 条 pairwise
样本，不再生成 2,000 条伪 AC tie。

### Listwise

现有 listwise loader 已经优先读取 `ranking`、`listwise_ranking`、
`raw_ranking` 或 `label_ranking`，Claude 和 GPT-5 的验证文件都会走这个显式
ranking 分支。本次只给回答文本增加 `answerA/answerB/answerC` 别名，不改变
ranking 的优先级、fallback 或错误处理。

## 启动参数约定

`run_pointwise5answers_three_stage_pairwise_listwise_sft_v1.py` 不做修改。新启动
脚本负责显式传入正确的数据参数，从而避开旧的 pairwise fallback。

GPT-5 Alpaca 运行分别传入：

- train：`data/gpt5/train-20k.json`；
- pairwise val：`data/gpt5/val-2k-eval.json`；
- listwise val：`data/gpt5/val-2k-eval-listwise.json`。

Claude Alpaca 运行将 `data/Alpaca-claude/val.json` 同时传给 pairwise 和
listwise 参数，因为该文件同时包含两类显式标签。Claude GPT4ALL 同样使用
`data/GPT4ALL-claude/val.json`。

现有 launcher 保持原样。用户新写的 Claude launcher 必须同时传入 pairwise 和
listwise 验证路径；本次实现不创建或修改 launcher。

## 数据一致性报告

实现后运行一次只读报告，分别核对：

- 记录数、ID 唯一性、train/val 重叠；
- GPT-5/Claude Alpaca 的 instruction、input、A/B/C 回答对齐；
- pointwise score 的逐项一致率、行级一致率和平均绝对差；
- 显式 AB/BC 标签的一致率；
- 显式 listwise ranking 的一致率；
- 每个 loader 的生成数、跳过数和标签分布。

报告不得用推导标签填补缺失值。当前只对拥有共同底层样本的 GPT-5/Claude
Alpaca 做 judge 间逐条比较；现有 `gpt5.zip` 不包含 GPT4ALL/Dolly，因此 Claude
GPT4ALL 只做自身结构与标签完整性检查。

## 错误处理

- 输入仍必须是 JSON 数组；
- 训练记录少于三个有效 scored answers 时沿用现有失败行为；
- 显式 pairwise 非空但未知的 choice 立即报错，防止静默制造 tie；
- 显式 pairwise 文件零有效样本时报错；
- 显式 listwise 的现有错误行为保持不变；
- pairwise 统计中区分缺失输出、缺失 choice 和非法 choice。

## 测试

新增聚焦测试覆盖：

1. `answerA/B/C` 训练记录能加载为三个 scored answers；
2. Claude 风格的显式 AB/BC 验证只生成两对，AC 缺失不会生成 tie；
3. 显式 choice `3` 仍生成真实 tie；
4. 非空未知 choice 抛出异常；
5. listwise loader 接受 `answerA/B/C` 和 `listwise_ranking`；
6. GPT-5 风格的 `outputA/B/C` 和现有 ranking 行为不回归。

测试先以当前行为运行并确认失败，再实施最小修改使其通过，最后运行相关测试和
完整测试集。
