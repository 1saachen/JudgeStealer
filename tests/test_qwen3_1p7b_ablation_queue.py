import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "launch_qwen3_1p7b_gpt5_ablation_auto_queue.sh"

BLOCK_SETTINGS = {
    "selector": (
        "random",
        "no_uncertainty",
        "no_diversity",
        "no_bias",
        "uncertainty_only",
        "diversity_only",
        "bias_only",
    ),
    "smoothing": ("a000", "a001", "a005", "a020", "adaptive"),
    "reviewing": ("none",),
    "budget": ("b0p5", "b1", "b2", "b5", "b10"),
}


def launcher_text():
    return LAUNCHER.read_text(encoding="utf-8")


def expected_jobs():
    return {
        f"{dataset}_{block}_{setting}"
        for dataset in ("alpaca", "gpt4all")
        for block, settings in BLOCK_SETTINGS.items()
        for setting in settings
    }


def test_queue_contains_exactly_36_unique_non_ours_jobs():
    text = launcher_text()
    jobs_block = re.search(r"JOBS=\((.*?)\)\n", text, re.S).group(1)
    jobs = re.findall(r"[a-z0-9_]+", jobs_block)
    assert len(jobs) == 36
    assert len(set(jobs)) == 36
    assert set(jobs) == expected_jobs()
    assert "selector_hybrid" not in text
    assert "smoothing_a010" not in text
    assert "reviewing_joint" not in text


def test_queue_uses_qwen3_1p7b_lora_and_current_data_paths():
    text = launcher_text()
    for required in (
        "$ROOT/models/Qwen3-1.7B",
        "$ROOT/data/alpaca/gpt5/train-20k.json",
        "$ROOT/data/alpaca/gpt5/val-2k-eval-listwise.json",
        "$ROOT/data/gpt4all/gpt5/train9k_pointwise_pairwise_no_val_overlap.json",
        "$ROOT/data/gpt4all/gpt5/val3k_pairwise_listwise.json",
        "--use-lora",
        "--load-in-4bit",
        "--learning-rate 1e-4",
        "--max-length 4096",
        "--gradient-accumulation-steps 16",
        "--eval-stages final",
    ):
        assert required in text


def test_queue_encodes_each_ablation_control():
    text = launcher_text()
    for fragment in (
        "selector_random)",
        "selector_no_uncertainty)",
        "selector_no_diversity)",
        "selector_no_bias)",
        "selector_uncertainty_only)",
        "selector_diversity_only)",
        "selector_bias_only)",
        "smoothing_adaptive)",
        "reviewing_none)",
        "budget_b0p5)",
        "budget_b10)",
        "--pointwise-global-smooth-adaptive-entropy",
        '--budget-percent "$budget_percent"',
        '--stage4-replay-strategy "$stage4_strategy"',
    ):
        assert fragment in text


def test_each_variant_changes_exactly_the_planned_controls():
    text = launcher_text()
    case_block = re.search(r'case "\$variant" in(.*?)\n  esac', text, re.S).group(1)
    actual = {}
    for variant, body in re.findall(r"\n    ([a-z0-9_]+)\) (.*?) ;;", case_block):
        actual[variant] = dict(re.findall(r"([a-z0-9_]+)=([a-z0-9_.]+)", body))

    expected = {
        "selector_random": {"selector_kind": "random", "reuse_proxy": "0"},
        "selector_no_uncertainty": {"uncertainty_weight": "0"},
        "selector_no_diversity": {"diversity_weight": "0"},
        "selector_no_bias": {"bias_weight": "0"},
        "selector_uncertainty_only": {"diversity_weight": "0", "bias_weight": "0"},
        "selector_diversity_only": {"uncertainty_weight": "0", "bias_weight": "0"},
        "selector_bias_only": {"diversity_weight": "0", "uncertainty_weight": "0"},
        "smoothing_a000": {"smooth_alpha": "0"},
        "smoothing_a001": {"smooth_alpha": "0.01"},
        "smoothing_a005": {"smooth_alpha": "0.05"},
        "smoothing_a020": {"smooth_alpha": "0.20"},
        "smoothing_adaptive": {"adaptive_smoothing": "1"},
        "reviewing_none": {"stage4_strategy": "none"},
        "budget_b0p5": {"budget_percent": "0.5"},
        "budget_b1": {"budget_percent": "1"},
        "budget_b2": {"budget_percent": "2"},
        "budget_b5": {"budget_percent": "5"},
        "budget_b10": {"budget_percent": "10"},
    }
    assert actual == expected


def test_queue_uses_local_output_guards_and_auto_gpu_dispatch():
    text = launcher_text()
    for fragment in (
        'OUTPUT_ROOT="${OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}"',
        'findmnt -n -o FSTYPE -T "$OUTPUT_ROOT"',
        "nfs|nfs4",
        "--query-compute-apps=gpu_uuid",
        "--query-gpu=memory.used",
        'exec {job_lock_fd}>"$LOG_ROOT/.job_${job}.lock"',
        'flock -n "$job_lock_fd"',
        'if [[ -f "$out/metrics_compact.json" ]]',
        'if [[ -e "$out" ]]',
        'sleep "$POLL_SECONDS"',
    ):
        assert fragment in text
    assert 'grep -F -- "--out $out"' not in text
