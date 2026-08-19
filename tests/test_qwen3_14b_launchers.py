from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LORA = ROOT / "launch_qwen3_14b_gpt5_lora_auto_queue.sh"
FULLFT = ROOT / "launch_qwen3_14b_gpt5_fullft_fsdp.sh"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_both_launchers_have_qwen14b_task_matrix_and_data_paths():
    for path in (LORA, FULLFT):
        text = read(path)
        assert "Qwen3-14B" in text
        assert "alpaca" in text and "gpt4all" in text
        assert "$ROOT/data/alpaca/gpt5/train-20k.json" in text
        assert "$ROOT/data/alpaca/gpt5/val-2k-eval-listwise.json" in text
        assert "$ROOT/data/gpt4all/gpt5/train9k_pointwise_pairwise_no_val_overlap.json" in text
        assert "$ROOT/data/gpt4all/gpt5/val3k_pairwise_listwise.json" in text
        assert 'JOBS=(alpaca gpt4all)' in text
        assert 'qwen3_14b_${dataset}_gpt5_b600' in text


def test_lora_launcher_uses_single_gpu_lora_protocol():
    text = read(LORA)
    for argument in (
        "--use-lora",
        "--load-in-4bit",
        "--candidate-selector-finetune-mode lora",
        "--learning-rate 1e-4",
        "--proxy-lr 1e-4",
        "--budget-units 600",
        "--max-length 4096",
        "--eval-stages final",
        "--stage4-replay-strategy stratified_triple",
    ):
        assert argument in text
    assert "torchrun" not in text
    assert "--fsdp" not in text
    assert 'SKIP_JOBS="${SKIP_JOBS:-}"' in text


def test_fullft_launcher_uses_two_process_fsdp_without_lora_or_4bit():
    text = read(FULLFT)
    for argument in (
        "torchrun",
        "--nproc_per_node=2",
        "--fsdp",
        "full_shard auto_wrap",
        "--fsdp-transformer-layer-cls-to-wrap Qwen3DecoderLayer",
        "--fsdp-activation-checkpointing",
        "--candidate-selector-finetune-mode full",
        "--learning-rate 1e-5",
        "--proxy-lr 1e-5",
        "--budget-units 600",
        "--max-length 4096",
        "--eval-stages final",
    ):
        assert argument in text
    assert "--use-lora" not in text
    assert "--load-in-4bit" not in text
    assert "--reuse-selection-proxy-for-stage1" not in text
    assert "GPU_IDS" in text


def test_fullft_torchrun_receives_python_script_not_python_executable():
    text = read(FULLFT)
    assert '"$TORCHRUN_BIN" --standalone --nproc_per_node=2 "$SCRIPT"' in text
    assert '"$TORCHRUN_BIN" --standalone --nproc_per_node=2 "$PY"' not in text


def test_launchers_protect_completed_and_incomplete_outputs():
    for path in (LORA, FULLFT):
        text = read(path)
        assert 'metrics_compact.json' in text
        assert 'if [[ -e "$out" ]]' in text
        assert "job_status.log" in text
