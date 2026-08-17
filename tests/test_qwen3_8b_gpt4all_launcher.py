from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "launch_qwen3_8b_gpt4all_gpt5_four_stage.sh"


def launcher_text() -> str:
    return LAUNCHER.read_text(encoding="utf-8")


def test_launcher_uses_portable_qwen3_8b_and_gpt4all_paths():
    text = launcher_text()
    assert 'MODEL="$ROOT/models/Qwen3-8B"' in text
    assert (
        'TRAIN_DATA="$ROOT/data/gpt4all/gpt5/'
        'train9k_pointwise_pairwise_no_val_overlap.json"' in text
    )
    assert (
        'EVAL_DATA="$ROOT/data/gpt4all/gpt5/val3k_pairwise_listwise.json"'
        in text
    )
    assert "$ROOT/qwen/" not in text
    assert "$ROOT/train_with_selector/train_with_selector/data/Dolly/" not in text


def test_launcher_enables_exact_four_stage_review_configuration():
    text = launcher_text()
    required = [
        "--budget-units 600",
        "--stage2-pointwise-replay-ratio 0",
        "--stage3-pointwise-replay-ratio 0",
        "--stage3-pairwise-replay-ratio 0",
        "--stage4-replay-strategy stratified_triple",
        "--stage4-replay-fraction 1",
        "--stage4-epochs 1",
        "--eval-stages final",
        "--use-lora",
        "--load-in-4bit",
    ]
    for argument in required:
        assert argument in text


def test_launcher_preserves_selector_and_smoothing_configuration():
    text = launcher_text()
    required = [
        "--pointwise-global-smooth-alpha 0.1",
        "--pointwise-global-smooth-mode local_gaussian",
        "--pointwise-global-smooth-gaussian-sigma 1.0",
        "--pointwise-global-smooth-stages all",
        "--candidate-selector-kind bias_trap_pointwise",
        "--candidate-selector-proxy-mode lm_head",
        "--reuse-selection-proxy-for-stage1",
        "--candidate-selector-init-triples 80",
        "--candidate-selector-batch-size 20",
        "--candidate-selector-max-score-candidates 100",
        "--candidate-selector-exploration-ratio 0",
        "--candidate-selector-diversity-weight 1",
        "--candidate-selector-uncertainty-weight 0.25",
        "--candidate-selector-bias-weight 1",
        "--candidate-selector-embedding-model BAAI/bge-small-en-v1.5",
    ]
    for argument in required:
        assert argument in text


def test_launcher_has_preflight_and_duplicate_run_guards():
    text = launcher_text()
    assert 'require_file "$SCRIPT"' in text
    assert 'require_dir "$MODEL"' in text
    assert 'require_file "$MODEL/config.json"' in text
    assert 'require_file "$TRAIN_DATA"' in text
    assert 'require_file "$EVAL_DATA"' in text
    assert 'if [[ -f "$OUT/metrics_compact.json" ]]' in text
    assert 'if [[ -e "$OUT" ]]' in text
    assert 'grep -F -- "--out $OUT"' in text


def test_launcher_defaults_outputs_and_logs_to_local_storage():
    text = launcher_text()
    assert (
        'DEFAULT_OUTPUT_ROOT="/opt/dlami/nvme/cyl/autodl-tmp/'
        'JudgeStealer_outputs"' in text
    )
    assert 'OUTPUT_ROOT="${OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}"' in text
    assert 'OUT="$OUTPUT_ROOT/$NAME"' in text
    assert 'LOG_ROOT="$OUTPUT_ROOT/qwen3_8b_gpt4all_gpt5_four_stage_logs"' in text


def test_launcher_reports_storage_and_rejects_nfs_outputs():
    text = launcher_text()
    assert 'findmnt -n -o FSTYPE -T "$OUTPUT_ROOT"' in text
    assert 'df -hP "$OUTPUT_ROOT"' in text
    assert 'nfs|nfs4)' in text
    assert 'ERROR network filesystem output is not allowed' in text
