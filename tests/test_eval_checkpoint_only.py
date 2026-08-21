from pathlib import Path

import pytest


pytest.importorskip("numpy")

import eval_checkpoint_only as evaluator


def test_empty_pairwise_path_uses_listwise_pair_expansion(monkeypatch):
    expected = (["pair"], [{"row": 1}], {"format": "expanded"})

    def fake_loader(path):
        assert path == "listwise.json"
        return expected

    monkeypatch.setattr(evaluator.three, "_load_pairwise_eval_from_listwise_dataset", fake_loader)

    assert evaluator._load_pairwise_eval_dataset("", "listwise.json") == expected


def test_explicit_pairwise_path_keeps_abc_loader(monkeypatch):
    expected = (["pair"], [{"row": 2}], {"format": "abc"})

    def fake_loader(path, pairwise_system_prompt):
        assert path == "pairwise.json"
        return expected

    monkeypatch.setattr(evaluator.base, "_load_pairwise_abc_eval_dataset", fake_loader)

    assert evaluator._load_pairwise_eval_dataset("pairwise.json", "listwise.json") == expected
