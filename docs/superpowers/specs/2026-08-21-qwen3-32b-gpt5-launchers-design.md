# Qwen3-32B GPT-5 Launcher Design

## Goal

补充 Qwen3-32B 在 Alpaca 和 GPT4All GPT-5 数据上的 LoRA 与 Full-FT 主实验启动器。

## Scope

- 新增 LoRA 单卡自动队列脚本。
- 新增 Full-FT 动态卡数 FSDP 顺序队列脚本，至少 4 卡，Qwen3-32B 推荐 8 卡。
- 两个脚本都支持 `SKIP_JOBS`、空闲 GPU 检测、完成标记保护、NVMe 输出和独立日志。
- 默认模型路径为 `models/Qwen3-32B`。
- 通过 `MODEL_DIR` 覆盖模型路径，通过 `MODEL_TAG` 覆盖输出名称中的模型标签。

## Experiment Configuration

- Dataset jobs: `alpaca`, `gpt4all`.
- Budget: 600 answer queries.
- Stage 1/2/3/4 epochs: 1 each.
- Max length: 4096; per-device batch size: 1; gradient accumulation: 16.
- Final-only evaluation.
- Pointwise local-Gaussian smoothing: alpha 0.1, sigma 1.0, all stages.
- Candidate selector: candidate triple selector, LM-head proxy, 80 initial triples, pool 100, batch 20.
- LoRA: main model LoRA + 4-bit, learning rate `1e-4`.
- Full-FT: main model FSDP full parameter training, learning rate `1e-5`; selector proxy remains LoRA + 4-bit.

## Output

Outputs default to `/opt/dlami/nvme/cyl/autodl-tmp/JudgeStealer_outputs` with names:

```text
${MODEL_TAG}_${dataset}_gpt5_b600_lora_selector_smooth_a010_pool100_stage4stratfull
${MODEL_TAG}_${dataset}_gpt5_b600_fullft_selector_smooth_a010_pool100_stage4stratfull
```

`metrics_compact.json` is the completion marker. Existing completed outputs are skipped; incomplete output directories are never overwritten.

## Model Compatibility

The Full-FT launcher wraps `Qwen3DecoderLayer` for FSDP. The model path is validated through `config.json` before a job starts.
