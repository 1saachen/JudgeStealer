from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "launch_qwen3_gpt5_fullft_auto_queue.sh"


def launcher_text() -> str:
    return LAUNCHER.read_text(encoding="utf-8")


def test_queue_contains_all_eight_model_dataset_jobs():
    text = launcher_text()
    for size in ("0p6b", "1p7b", "4b", "8b"):
        assert f"alpaca_{size}" in text
        assert f"gpt4all_{size}" in text
    for model_dir in (
        "Qwen3-0.6B",
        "Qwen3-1.7B",
        "Qwen3-4B",
        "Qwen3-8B",
    ):
        assert f'$ROOT/models/{model_dir}' in text


def test_queue_uses_current_gpt5_data_paths():
    text = launcher_text()
    assert '$ROOT/data/alpaca/gpt5/train-20k.json' in text
    assert '$ROOT/data/alpaca/gpt5/val-2k-eval-listwise.json' in text
    assert (
        '$ROOT/data/gpt4all/gpt5/'
        'train9k_pointwise_pairwise_no_val_overlap.json' in text
    )
    assert '$ROOT/data/gpt4all/gpt5/val3k_pairwise_listwise.json' in text


def test_queue_writes_outputs_and_logs_to_overridable_nvme_storage():
    text = launcher_text()
    assert (
        'DEFAULT_OUTPUT_ROOT="/opt/dlami/nvme/cyl/autodl-tmp/'
        'JudgeStealer_outputs"' in text
    )
    assert 'OUTPUT_ROOT="${OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}"' in text
    assert 'out="$OUTPUT_ROOT/$name"' in text
    assert 'LOG_ROOT="$OUTPUT_ROOT/qwen3_gpt5_fullft_auto_queue_logs"' in text
    assert "check_output_storage" in text
    assert 'findmnt -n -o FSTYPE -T "$OUTPUT_ROOT"' in text
    assert "nfs|nfs4" in text


def test_queue_runs_fullft_selector_with_exact_training_configuration():
    text = launcher_text()
    required = [
        "--candidate-selector-finetune-mode full",
        "--candidate-selector-proxy-mode lm_head",
        "--reuse-selection-proxy-for-stage1",
        "--proxy-lr 1e-5",
        "--learning-rate 1e-5",
        "--budget-units 600",
        "--candidate-selector-init-triples 80",
        "--candidate-selector-batch-size 20",
        "--candidate-selector-max-score-candidates 100",
        "--stage4-replay-strategy stratified_triple",
        "--stage4-replay-fraction 1",
        "--stage4-epochs 1",
        "--max-length 4096",
        "--per-device-batch-size 1",
        "--gradient-accumulation-steps 16",
        "--eval-stages final",
    ]
    for argument in required:
        assert argument in text
    for forbidden in (
        "--use-lora",
        "--load-in-4bit",
        "--fixed-selected-triples-path",
        "torchrun",
        "--fsdp",
    ):
        assert forbidden not in text


def test_queue_preserves_smoothing_and_bias_trap_selector():
    text = launcher_text()
    for argument in (
        "--pointwise-global-smooth-alpha 0.1",
        "--pointwise-global-smooth-mode local_gaussian",
        "--pointwise-global-smooth-gaussian-sigma 1.0",
        "--pointwise-global-smooth-stages all",
        "--candidate-selector-kind bias_trap_pointwise",
        "--candidate-selector-diversity-weight 1",
        "--candidate-selector-uncertainty-weight 0.25",
        "--candidate-selector-bias-weight 1",
        "--candidate-selector-exploration-ratio 0",
    ):
        assert argument in text


def test_queue_requires_allowlist_and_checks_real_gpu_idleness():
    text = launcher_text()
    assert 'usage: $0 <gpu_id> [gpu_id ...]' in text
    assert "GPU_MEMORY_USED_LIMIT_MB=1024" in text
    assert "--query-compute-apps=gpu_uuid" in text
    assert "--query-gpu=memory.used" in text
    assert 'sleep "$POLL_SECONDS"' in text
    assert "declare -A GPU_WORKER_PIDS" in text


def test_queue_protects_existing_outputs_and_continues_failures():
    text = launcher_text()
    assert '"$model/config.json"' in text
    assert "$MODEL_CONFIG" not in text
    assert 'if [[ -f "$out/metrics_compact.json" ]]' in text
    assert 'if [[ -e "$out" ]]' in text
    assert 'grep -F -- "--out $out"' in text
    assert "overall_rc=1" in text
    assert "job_status.log" in text


def test_queue_supports_explicit_job_skips_before_gpu_assignment():
    text = launcher_text()
    assert 'SKIP_JOBS="${SKIP_JOBS:-}"' in text
    assert "should_skip_job()" in text
    assert 'SKIP configured job=$job' in text
