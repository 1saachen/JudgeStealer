# Qwen3-8B GPT4All 四阶段启动脚本设计

## 目标

为现有的 Qwen3-8B、GPT4All GPT-5 实验新增一个单用途启动脚本。脚本需要
完整保留 `launch_qwen3_gpt5_selector_smooth_lora_table_20260814.sh` 中对应
实验的训练配置，同时改用新服务器上的可移植模型和数据路径。

## 范围

在仓库根目录创建 `launch_qwen3_8b_gpt4all_gpt5_four_stage.sh`。不修改原有
多任务启动脚本，也不修改 Python 训练代码。新脚本只运行一个实验，并且只
接收一个位置参数：传给 `CUDA_VISIBLE_DEVICES` 的物理 GPU 编号。

## 路径

- 训练脚本：
  `run_pointwise5answers_three_stage_pairwise_listwise_sft_v1.py`
- 模型：`models/Qwen3-8B`
- 训练数据：
  `data/gpt4all/gpt5/train9k_pointwise_pairwise_no_val_overlap.json`
- 验证数据：`data/gpt4all/gpt5/val3k_pairwise_listwise.json`
- 结果目录：
  `outputs/qwen3_8b_gpt4all_gpt5_b600_selector_smooth_a010_pool100_stage4stratfull`
- 日志目录：`outputs/qwen3_8b_gpt4all_gpt5_four_stage_logs/`

所有路径都相对于启动脚本所在位置解析，因此仓库移动后不需要修改绝对路径。

## 训练配置

启动脚本将保留原表格实验启动脚本中的以下设置：

- 随机种子 42，预算 600；
- Stage 1 复用 LM-head bias-trap 选样代理模型；
- Stage 2 训练 pairwise，不回放 pointwise；
- Stage 3 训练 listwise，不回放 pointwise 或 pairwise；
- Stage 4 使用 `stratified_triple`，回放比例 1.0，训练 1 个 epoch；
- pointwise、pairwise 和 listwise 各训练 1 个 epoch；
- 使用 LoRA 和 4-bit 加载；
- 单卡 batch size 为 1，梯度累积步数为 16；
- 学习率 `1e-4`，最大长度 4096，评估 batch size 为 1；
- 只在最终阶段完成后评估；
- 所有阶段使用 local-Gaussian 平滑，alpha 为 0.1，sigma 为 1.0；
- 使用 bias-trap pointwise selector，并复用 LM-head 代理模型：初始样本 80、
  每轮查询 20、候选池 100、无随机探索、diversity 权重 1.0、uncertainty
  权重 0.25、bias 权重 1.0；
- 使用 `BAAI/bge-small-en-v1.5` embedding，并保留原脚本中的 embedding
  和代理模型参数。

## 运行行为

启动脚本将执行以下操作：

1. 检查是否提供 GPU 编号。
2. 启动前检查训练脚本、模型目录、模型 `config.json` 和两个数据文件是否存在。
3. 如果设置了 `PYTHON_BIN`，则使用该解释器；否则使用当前已激活 Conda
   环境中的 `python`。
4. 如果存在未完成的结果目录，拒绝覆盖。
5. 如果结果目录中已经存在 `metrics_compact.json`，跳过本次运行。
6. 如果已有进程正在写入同一个结果目录，拒绝重复启动。
7. 在状态日志中记录开始、完成和失败状态，并把完整训练输出写入独立日志文件。
8. 参数检查或训练失败时返回非零退出码。

启动脚本不会自动下载模型或数据集。

## 验证方式

新增一个聚焦的静态测试，用来检查启动脚本是否包含正确的可移植路径、准确的
四阶段回放配置、模型尺寸、selector 参数和平滑参数，同时确保脚本不再引用
旧的 `qwen/` 或 `Dolly/` 路径。在能够使用 Bash 的环境中额外执行 Bash 语法
检查。现有 Python 训练测试不属于本次纯配置改动的范围，除非用户另行要求，
否则不运行。

