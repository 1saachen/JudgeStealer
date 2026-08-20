import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "launch_qwen3_1p7b_single_task_auto_queue.sh"


def launcher_text():
    return LAUNCHER.read_text(encoding="utf-8")


def test_queue_has_exact_six_single_task_jobs():
    text = launcher_text()
    jobs_block = re.search(r"JOBS=\((.*?)\)\n", text, re.S).group(1)
    jobs = re.findall(r"[a-z0-9_]+", jobs_block)
    expected = {
        f"{dataset}_{task}_only"
        for dataset in ("alpaca", "gpt4all")
        for task in ("pointwise", "pairwise", "listwise")
    }
    assert len(jobs) == 6
    assert len(set(jobs)) == 6
    assert set(jobs) == expected


def test_queue_uses_single_task_protocol_and_current_paths():
    text = launcher_text()
    for required in (
        "$ROOT/models/Qwen3-1.7B",
        "$ROOT/data/alpaca/gpt5/train-20k.json",
        "$ROOT/data/alpaca/gpt5/val-2k-eval.json",
        "$ROOT/data/alpaca/gpt5/val-2k-eval-listwise.json",
        "$ROOT/data/gpt4all/gpt5/train9k_pointwise_pairwise_no_val_overlap.json",
        "$ROOT/data/gpt4all/gpt5/val3k_pairwise_listwise.json",
        "--mode single_task",
        '--single-task "$single_task"',
        "--budget 600",
        "--pointwise-epochs 10",
        "--pairwise-epochs 10",
        "--listwise-epochs 10",
        "--per-device-batch-size 1",
        "--gradient-accumulation-steps 16",
        "--learning-rate 1e-4",
        "--max-length 4096",
        "--eval-batch-size 1",
        "--eval-stages final",
        "--use-lora",
        "--load-in-4bit",
        "qwen3_1p7b_single_task_seed42",
    ):
        assert required in text
    assert "--stage4-replay" not in text
    assert "--pointwise-global-smooth" not in text


def test_queue_has_storage_and_process_guards():
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
        "nvidia-smi",
    ):
        assert required in text
