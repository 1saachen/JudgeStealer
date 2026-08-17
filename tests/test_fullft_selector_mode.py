import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "run_pointwise5answers_three_stage_pairwise_listwise_sft_v1.py"
SELECTOR = ROOT / "run_pointwise5answers_three_to_listwise_v1.py"


def selector_source() -> str:
    return SELECTOR.read_text(encoding="utf-8")


def load_mode_resolver():
    tree = ast.parse(selector_source())
    resolver = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_resolve_candidate_selector_finetune_mode"
    )
    namespace = {"Any": object}
    exec(compile(ast.Module(body=[resolver], type_ignores=[]), str(SELECTOR), "exec"), namespace)
    return namespace[resolver.name]


def test_legacy_selector_mode_defaults_to_lora():
    resolve = load_mode_resolver()
    assert resolve(SimpleNamespace(load_in_4bit=False)) == "lora"


def test_full_selector_mode_is_supported_without_quantization():
    resolve = load_mode_resolver()
    cfg = SimpleNamespace(
        candidate_selector_finetune_mode="full",
        load_in_4bit=False,
    )
    assert resolve(cfg) == "full"


def test_full_selector_mode_rejects_4bit():
    resolve = load_mode_resolver()
    cfg = SimpleNamespace(
        candidate_selector_finetune_mode="full",
        load_in_4bit=True,
    )
    with pytest.raises(ValueError, match="full.*4-bit"):
        resolve(cfg)


def test_candidate_selector_forwards_resolved_finetune_mode():
    source = selector_source()
    assert "finetune_mode=_resolve_candidate_selector_finetune_mode(cfg)" in source


def test_main_cli_exposes_and_records_selector_finetune_mode():
    source = MAIN.read_text(encoding="utf-8")
    assert '"--candidate-selector-finetune-mode"' in source
    assert 'choices=["lora", "full"]' in source
    assert 'default="lora"' in source
    assert (
        "candidate_selector_finetune_mode="
        "str(args.candidate_selector_finetune_mode)" in source
    )
