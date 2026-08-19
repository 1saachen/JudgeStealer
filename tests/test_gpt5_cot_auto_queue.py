from pathlib import Path


LAUNCHER = Path(__file__).resolve().parents[1] / "launch_gpt5_cot_auto_queue.sh"


def test_launcher_maps_paper_methods_to_cot_modes():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "naive) mode=mix" in text
    assert "ours) mode=stage4" in text
    assert "alpaca_llama_naive" in text
    assert "gpt4all_qwen_ours" in text


def test_launcher_uses_configurable_cot_roots_and_unique_outputs():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert 'DATA_ROOT="${DATA_ROOT:-/data/model-extraction-attack/yaolin/JudgeStealer/data}"' in text
    assert 'ALPACA_COT_DATA_DIR="${ALPACA_COT_DATA_DIR:-$DATA_ROOT/Alpaca-cot-gpt}"' in text
    assert "ALPACA_COT_DATA_DIR" in text
    assert "GPT4ALL_COT_DATA_DIR" in text
    assert 'name="${surrogate}_${dataset}_gpt5_cot_${paper_method}_seed42"' in text
    assert '"$PY" "$ROOT/prepare_alpaca_cot_4066.py"' in text
