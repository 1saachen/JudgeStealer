from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LORA = ROOT / "launch_qwen3_32b_gpt5_lora_auto_queue.sh"
FULLFT = ROOT / "launch_qwen3_32b_gpt5_fullft_fsdp.sh"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_both_launchers_use_overridable_qwen_model_and_both_datasets():
    for path in (LORA, FULLFT):
        text = read(path)
        assert 'MODEL_DIR="${MODEL_DIR:-$ROOT/models/Qwen3-32B}"' in text
        assert 'MODEL_TAG="${MODEL_TAG:-qwen3_32b}"' in text
        assert 'JOBS=(alpaca gpt4all)' in text
        assert "$ROOT/data/alpaca/gpt5/train-20k.json" in text
        assert "$ROOT/data/alpaca/gpt5/val-2k-eval-listwise.json" in text
        assert "$ROOT/data/gpt4all/gpt5/train9k_pointwise_pairwise_no_val_overlap.json" in text
        assert "$ROOT/data/gpt4all/gpt5/val3k_pairwise_listwise.json" in text
        assert "metrics_compact.json" in text
        assert 'SKIP_JOBS="${SKIP_JOBS:-}"' in text


def test_lora_launcher_is_single_gpu_and_uses_lora_protocol():
    text = read(LORA)
    assert "--use-lora" in text
    assert "--load-in-4bit" in text
    assert "--learning-rate 1e-4" in text
    assert "--proxy-lr 1e-4" in text
    assert "torchrun" not in text
    assert "--fsdp" not in text
    assert 'name="${MODEL_TAG}_${dataset}_gpt5_b600_lora_selector_smooth_a010_pool100_stage4stratfull"' in text


def test_fullft_launcher_uses_four_gpu_fsdp_and_qwen_layer():
    text = read(FULLFT)
    for argument in (
        '"$TORCHRUN_BIN" --standalone --nproc_per_node="$NPROC_PER_NODE" "$SCRIPT"',
        "--fsdp-transformer-layer-cls-to-wrap Qwen3DecoderLayer",
        "--fsdp-activation-checkpointing",
        "--learning-rate 1e-5",
        "--proxy-lr 1e-5",
        "--candidate-selector-load-in-4bit",
        'name="${MODEL_TAG}_${dataset}_gpt5_b600_fullft_selector_smooth_a010_pool100_stage4stratfull"',
    ):
        assert argument in text
    assert "--use-lora" not in text
    assert "--load-in-4bit" not in text
    assert 'if [[ "$#" -lt 4 ]]' in text


def test_fullft_launcher_supports_eight_gpu_runs_from_argument_count():
    text = read(FULLFT)
    assert 'if [[ "$#" -lt 4 ]]' in text
    assert 'NPROC_PER_NODE="${NPROC_PER_NODE:-${#GPU_IDS[@]}}"' in text
    assert '--nproc_per_node="$NPROC_PER_NODE"' in text
    assert 'gpus=${GPU_IDS[*]}' in text


def test_both_launchers_use_common_experiment_controls():
    for path in (LORA, FULLFT):
        text = read(path)
        for argument in (
            "--budget-units 600",
            "--stage4-epochs 1",
            "--pointwise-epochs 1",
            "--pairwise-epochs 1",
            "--listwise-epochs 1",
            "--max-length 4096",
            "--eval-stages final",
            "--pointwise-global-smooth-alpha 0.1",
            "--pointwise-global-smooth-gaussian-sigma 1.0",
            "--candidate-selector-init-triples 80",
            "--candidate-selector-max-score-candidates 100",
        ):
            assert argument in text
