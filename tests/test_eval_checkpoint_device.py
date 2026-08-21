import pytest


pytest.importorskip("numpy")

import eval_checkpoint_only as evaluator


class FakeModel:
    def __init__(self):
        self.calls = []

    def to(self, **kwargs):
        self.calls.append(kwargs)
        return self


def test_move_model_to_cuda_uses_current_device_and_bfloat16(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(evaluator.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(evaluator.torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(evaluator.torch.cuda, "is_bf16_supported", lambda: True)

    assert evaluator._move_model_to_eval_device(model) is model
    assert model.calls == [{"device": evaluator.torch.device("cuda", 2), "dtype": evaluator.torch.bfloat16}]


def test_move_model_to_cuda_fails_without_gpu(monkeypatch):
    monkeypatch.setattr(evaluator.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA"):
        evaluator._move_model_to_eval_device(FakeModel())
