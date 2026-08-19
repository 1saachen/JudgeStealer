import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "launch_qwen3_1p7b_mix_budget_auto_queue.sh"


def launcher_text():
    return LAUNCHER.read_text(encoding="utf-8")


def test_queue_contains_ten_unique_budget_matched_mix_jobs():
    text = launcher_text()
    jobs_block = re.search(r"JOBS=\((.*?)\)\n", text, re.S).group(1)
    jobs = re.findall(r"[a-z0-9_]+", jobs_block)
    expected = {
        f"{dataset}_mix_{budget}"
        for dataset in ("alpaca", "gpt4all")
        for budget in ("b0p5", "b1", "b2", "b5", "b10")
    }
    assert len(jobs) == 10
    assert len(set(jobs)) == 10
    assert set(jobs) == expected


def test_queue_maps_measured_budgets_to_equal_mix_sample_counts():
    text = launcher_text()
    for fragment in (
        "alpaca_mix_b0p5) budget_percent=0.5; candidate_queries=18000; sample_count=90; budget_units=270 ;;",
        "alpaca_mix_b1) budget_percent=1; candidate_queries=18000; sample_count=180; budget_units=540 ;;",
        "alpaca_mix_b2) budget_percent=2; candidate_queries=18000; sample_count=360; budget_units=1080 ;;",
        "alpaca_mix_b5) budget_percent=5; candidate_queries=18000; sample_count=900; budget_units=2700 ;;",
        "alpaca_mix_b10) budget_percent=10; candidate_queries=18000; sample_count=1800; budget_units=5400 ;;",
        "gpt4all_mix_b0p5) budget_percent=0.5; candidate_queries=8100; sample_count=40; budget_units=120 ;;",
        "gpt4all_mix_b1) budget_percent=1; candidate_queries=8100; sample_count=80; budget_units=240 ;;",
        "gpt4all_mix_b2) budget_percent=2; candidate_queries=8100; sample_count=160; budget_units=480 ;;",
        "gpt4all_mix_b5) budget_percent=5; candidate_queries=8100; sample_count=410; budget_units=1230 ;;",
        "gpt4all_mix_b10) budget_percent=10; candidate_queries=8100; sample_count=810; budget_units=2430 ;;",
    ):
        assert fragment in text


def test_queue_uses_true_value_mix_protocol_and_current_paths():
    text = launcher_text()
    for required in (
        "$ROOT/models/Qwen3-1.7B",
        "$ROOT/data/alpaca/gpt5/train-20k.json",
        "$ROOT/data/alpaca/gpt5/val-2k-eval.json",
        "$ROOT/data/alpaca/gpt5/val-2k-eval-listwise.json",
        "$ROOT/data/gpt4all/gpt5/train9k_pointwise_pairwise_no_val_overlap.json",
        "$ROOT/data/gpt4all/gpt5/val3k_pairwise_listwise.json",
        "--mode trueval_three_stage",
        "--pointwise-train-samples \"$sample_count\"",
        "--pairwise-train-pairs \"$sample_count\"",
        "--listwise-train-examples \"$sample_count\"",
        "--pointwise-epochs 10",
        "--pairwise-epochs 10",
        "--listwise-epochs 10",
        "--stage2-pointwise-replay-ratio 0",
        "--stage3-pointwise-replay-ratio 0",
        "--stage3-pairwise-replay-ratio 0",
        "--learning-rate 1e-4",
        "--max-length 4096",
        "--eval-stages final",
        "--use-lora",
        "--load-in-4bit",
    ):
        assert required in text
    assert "--pointwise-global-smooth" not in text
    assert "--stage4-replay" not in text


def test_queue_uses_local_output_guards_and_job_locks():
    text = launcher_text()
    for required in (
        'OUTPUT_ROOT="${OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}"',
        'findmnt -n -o FSTYPE -T "$OUTPUT_ROOT"',
        "nfs|nfs4",
        'if [[ -f "$out/metrics_compact.json" ]]',
        'if [[ -e "$out" ]]',
        'exec {job_lock_fd}>"$LOG_ROOT/.job_${job}.lock"',
        'flock -n "$job_lock_fd"',
        'SKIP_JOBS="${SKIP_JOBS:-}"',
        'sleep "$POLL_SECONDS"',
    ):
        assert required in text
    assert 'grep -F -- "--out $out"' not in text
