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


def test_launcher_treats_all_positionals_as_allowed_gpus():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert 'usage: $0 <gpu_id> [gpu_id ...]' in text
    assert 'ALLOWED_GPUS=("$@")' in text
    assert 'JOBS=("${ALL_JOBS[@]}")' in text
    assert 'GPU_WORKER_PIDS' in text
    assert 'SKIP_JOBS="${SKIP_JOBS:-}"' in text


def test_launcher_defaults_outputs_to_local_nvme():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert 'DEFAULT_OUTPUT_ROOT="/opt/dlami/nvme/cyl/autodl-tmp/JudgeStealer_outputs"' in text
    assert 'OUTPUT_ROOT="${OUTPUT_ROOT:-$DEFAULT_OUTPUT_ROOT}"' in text


def test_launcher_uses_server_model_directories():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert 'LLAMA_MODEL="${LLAMA_MODEL:-$ROOT/models/Llama-3.2-1b-instruct}"' in text
    assert 'QWEN_MODEL="${QWEN_MODEL:-$ROOT/models/Qwen3-1.7B}"' in text
    assert 'model="$LLAMA_MODEL"' in text
    assert 'model="$QWEN_MODEL"' in text
