from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "launch_claude_lora_auto_queue.sh"


def launcher_text() -> str:
    return LAUNCHER.read_text(encoding="utf-8")


def test_claude_queue_contains_eight_lora_jobs_with_explicit_validation():
    text = launcher_text()
    for job in (
        "selector_alpaca_llama1b",
        "selector_gpt4all_llama1b",
        "selector_alpaca_qwen1p7b",
        "selector_gpt4all_qwen1p7b",
        "mixep10_alpaca_llama1b",
        "mixep10_gpt4all_llama1b",
        "mixep10_alpaca_qwen1p7b",
        "mixep10_gpt4all_qwen1p7b",
    ):
        assert job in text
    for path in (
        "$ROOT/data/alpaca/claude/train.json",
        "$ROOT/data/alpaca/claude/val.json",
        "$ROOT/data/gpt4all/claude/train.json",
        "$ROOT/data/gpt4all/claude/val.json",
    ):
        assert path in text
    assert "--pairwise-eval-dataset" in text
    assert "--listwise-eval-dataset" in text
    assert "--use-lora" in text
    assert "--load-in-4bit" in text


def test_claude_queue_keeps_selector_and_mixep10_protocols_distinct():
    text = launcher_text()
    for selector_argument in (
        "--train-selection-mode candidate_triple_selector",
        "--candidate-selector-proxy-mode lm_head",
        "--candidate-selector-init-triples 80",
        "--candidate-selector-batch-size 20",
        "--candidate-selector-max-score-candidates 100",
        "--stage4-replay-strategy stratified_triple",
        "--pointwise-global-smooth-alpha 0.1",
    ):
        assert selector_argument in text
    for mix_argument in (
        "--mode trueval_three_stage",
        "--pointwise-train-samples 200",
        "--pairwise-train-pairs 200",
        "--listwise-train-examples 200",
        "--pointwise-epochs 10",
        "--pairwise-epochs 10",
        "--listwise-epochs 10",
    ):
        assert mix_argument in text
