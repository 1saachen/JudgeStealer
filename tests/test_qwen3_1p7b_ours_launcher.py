from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "launch_qwen3_1p7b_gpt4all_ours.sh"


def launcher_text():
    return LAUNCHER.read_text(encoding="utf-8")


def test_ours_launcher_uses_dolly_paths_and_dedicated_output():
    text = launcher_text()
    for required in (
        "$ROOT/models/Qwen3-1.7B",
        "$ROOT/data/gpt4all/gpt5/train9k_pointwise_pairwise_no_val_overlap.json",
        "$ROOT/data/gpt4all/gpt5/val3k_pairwise_listwise.json",
        'NAME="qwen3_1p7b_lora_seed42_gpt4all_ours_b600_selector_smooth_a010_pool100_stage4stratfull"',
        'DEFAULT_OUTPUT_ROOT="/opt/dlami/nvme/cyl/autodl-tmp/JudgeStealer_outputs"',
    ):
        assert required in text


def test_ours_launcher_encodes_standard_pipeline_without_ablation_override():
    text = launcher_text()
    for required in (
        "--seed 42",
        "--budget-units 600",
        "--stage4-replay-strategy stratified_triple",
        "--stage4-replay-fraction 1",
        "--stage4-epochs 1",
        "--pointwise-epochs 1",
        "--pairwise-epochs 1",
        "--listwise-epochs 1",
        "--per-device-batch-size 1",
        "--gradient-accumulation-steps 16",
        "--learning-rate 1e-4",
        "--max-length 4096",
        "--eval-stages final",
        "--use-lora",
        "--load-in-4bit",
        "--candidate-selector-kind bias_trap_pointwise",
        "--candidate-selector-proxy-mode lm_head",
        "--reuse-selection-proxy-for-stage1",
        "--pointwise-global-smooth-alpha 0.1",
        "--pointwise-global-smooth-mode local_gaussian",
        "--pointwise-global-smooth-gaussian-sigma 1.0",
        "--pointwise-global-smooth-stages all",
    ):
        assert required in text
    assert "--budget-percent" not in text
    assert "--pointwise-global-smooth-adaptive-entropy" not in text


def test_ours_launcher_has_local_storage_guard_and_completion_checks():
    text = launcher_text()
    for required in (
        'OUTPUT_ROOT="${OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}"',
        'findmnt -n -o FSTYPE -T "$OUTPUT_ROOT"',
        "nfs|nfs4",
        'if [[ -f "$OUT/metrics_compact.json" ]]',
        'if [[ -e "$OUT" ]]',
        'usage: $0 <gpu_id>',
    ):
        assert required in text
