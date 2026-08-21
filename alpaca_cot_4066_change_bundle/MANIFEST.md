# Alpaca CoT 4066 改动文件清单

本目录是本次 Alpaca CoT 4066 实验相关代码的快照。文件保留项目内的原相对路径，便于对照或复制回项目根目录。

## 修改的文件

| 文件 | 修改内容 |
|---|---|
| `.gitignore` | 忽略本地 Alpaca CoT 原始数据与派生切分 |
| `README.md` | 增加本实验统一文档入口 |
| `run_pointwise5answers_two_to_pairwise_v1.py` | reason 字段传递、CoT target、最终 score token 定位及 smoothing 元数据 |
| `run_pointwise5answers_three_stage_pairwise_listwise_sft_v1.py` | 将 tokenizer 的 1--10 score token 序列传给 dataset 和 trainer |
| `train_with_selector/train_with_selector/data/pairwise_dataset.py` | pairwise prompt builder 增加可选的 label-only 指令开关，旧行为默认不变 |

## 新增的文件

| 文件 | 用途 |
|---|---|
| `prepare_alpaca_cot_4066.py` | 恢复 4066 条完整训练数据，生成固定 Mix/eval 切分 |
| `run_alpaca_cot_stage4_mix.py` | 四阶段 synthetic-CoT 与 real-CoT Mix 的统一训练/评测入口 |
| `launch_alpaca_cot_stage4_mix.sh` | 单卡运行 stage4、mix 或二者 |
| `tests/test_alpaca_cot_stage4_mix.py` | prompt、标签清理、tie canonicalization 和 score-token 定位测试 |
| `ALPACA_COT_4066_EXPERIMENT.md` | 完整实验与逐文件代码改动文档 |

## 未包含内容

- 原始或派生数据集；
- 模型权重与 checkpoint；
- `outputs/`、日志和评测生成结果；
- 与本实验无关的工作区改动。

## 目录使用方式

本目录是代码快照，不建议直接在目录内启动训练。需要恢复到相同项目结构时，从本目录把文件按相对路径复制到项目根目录，再在项目根目录运行测试或训练。
