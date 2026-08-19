from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "launch_rewardmodel_lora_auto_queue.sh"


def test_rewardmodel_queue_defines_the_table_jobs_and_data_contract():
    text = LAUNCHER.read_text(encoding="utf-8")

    for job in (
        "naive_alpaca_llama1b",
        "ours_alpaca_llama1b",
        "naive_gpt4all_llama1b",
        "ours_gpt4all_llama1b",
        "naive_alpaca_qwen1p7b",
        "ours_alpaca_qwen1p7b",
        "naive_gpt4all_qwen1p7b",
        "ours_gpt4all_qwen1p7b",
    ):
        assert job in text

    for required in (
        "REWARDMODEL_SOURCE",
        "ALPACA_REWARDMODEL_SOURCE",
        "GPT4ALL_REWARDMODEL_SOURCE",
        "prepare_rewardmodel_three_stage.py",
        "run_rewardmodel_three_stage_sft.py",
        "models/Llama-3.2-1b-instruct",
        "models/Qwen3-1.7B",
    ):
        assert required in text

    assert '"$ROOT/data/reward-model"' in text
    assert "naive_rewardmodel_llama1b" in text
    assert "ours_rewardmodel_qwen1p7b" in text


def test_rewardmodel_queue_keeps_naive_and_ours_protocols_distinct():
    text = LAUNCHER.read_text(encoding="utf-8")

    for naive_argument in (
        "--mode mix",
        "--pointwise-train-samples 200",
        "--pairwise-train-samples 200",
        "--listwise-train-samples 200",
        "--pointwise-epochs 10",
        "--pairwise-epochs 10",
        "--listwise-epochs 10",
        "--smooth-alpha 0",
    ):
        assert naive_argument in text
    for ours_argument in (
        "--mode selector",
        "--budget-units 600",
        "--selector-init-questions 80",
        "--selector-batch-size 20",
        "--selector-pool-size 100",
        "--smooth-alpha 0.1",
    ):
        assert ours_argument in text


def test_rewardmodel_ours_enables_pairwise_and_listwise_order_augmentation():
    text = LAUNCHER.read_text(encoding="utf-8")
    ours = text.split("run_ours() {", 1)[1].split("\nrun_job() {", 1)[0]
    naive = text.split("run_naive() {", 1)[1].split("\nrun_ours() {", 1)[0]

    assert "--pairwise-order-augmentation" in ours
    assert "--listwise-order-augmentation" in ours
    assert "--pairwise-order-augmentation" not in naive
    assert "--listwise-order-augmentation" not in naive
