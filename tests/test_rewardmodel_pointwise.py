import numpy as np

from run_rewardmodel_three_stage_sft import (
    _apply_tie_policy,
    _listwise_soft_choice_metadata,
    _pairwise_soft_choice_metadata,
)
from run_skywork_pointwise import _answer_proxy_scores_from_predictions
from train_with_selector.train_with_selector.data.skywork_dataset import SkyworkAnswer


def _answer(question_id: int, key: str) -> SkyworkAnswer:
    return SkyworkAnswer(
        question_id=question_id,
        source_id=question_id,
        dataset="test",
        instruction="question",
        input_text="",
        answer_key=key,
        model="model",
        output="answer",
        reward=0.0,
    )


def test_answer_proxy_scores_use_equal_uncertainty_and_question_spread():
    answers = [_answer(1, "A"), _answer(1, "B"), _answer(2, "A"), _answer(2, "B")]
    means = np.asarray([0.0, 2.0, 1.0, 1.0], dtype=np.float32)
    uncertainty = np.asarray([0.0, 0.0, 2.0, 2.0], dtype=np.float32)

    scores, diagnostics = _answer_proxy_scores_from_predictions(
        answers, means, uncertainty, uncertainty_weight=0.5, response_std_weight=0.5
    )

    np.testing.assert_allclose(scores, np.full((4,), 0.5, dtype=np.float32))
    assert diagnostics["pool_mc_uncertainty_mean"] == 1.0
    assert diagnostics["pool_predicted_response_std_mean"] == 0.5


def test_answer_proxy_scores_handle_one_remaining_answer_per_question():
    answers = [_answer(1, "A"), _answer(2, "B")]
    scores, _ = _answer_proxy_scores_from_predictions(
        answers,
        np.asarray([0.0, 3.0], dtype=np.float32),
        np.asarray([0.25, 0.75], dtype=np.float32),
        uncertainty_weight=0.5,
        response_std_weight=0.5,
    )

    np.testing.assert_allclose(scores, np.asarray([0.25, 0.75], dtype=np.float32))


def test_pairwise_soft_tie_target_is_uniform_over_two_winners():
    rows = [{"record_id": 7, "pair_name": "AB"}]
    records = [{"id": 7, "pairwise": {"AB": {"scoreA": 4.5, "scoreB": 4.5}}}]
    distributions, candidates, stats = _pairwise_soft_choice_metadata(rows, records, eos="</s>")
    assert distributions == [{"left": 0.5, "right": 0.5}]
    assert candidates == [["[[1]]</s>", "[[2]]</s>"]]
    assert stats["tied_winner"] == 1


def test_listwise_soft_tie_target_keeps_all_top_winners():
    rows = [{"listwise_scoreA": 5.0, "listwise_scoreB": 5.0, "listwise_scoreC": 2.0}]
    distributions, candidates, stats = _listwise_soft_choice_metadata(rows, eos="</s>")
    assert distributions[0] == {"0": 0.5, "1": 0.5}
    assert candidates[0] == ["Ranking:[A>B>C]</s>", "Ranking:[B>A>C]</s>"]
    assert stats["tied_winner"] == 1


def test_unique_only_drops_only_tied_rows():
    items = [("pairwise", "prompt", "target", -100), ("pairwise", "prompt2", "target2", -100)]
    distributions = [{"a": 0.5, "b": 0.5}, None]
    candidates = [["a", "b"], None]
    kept, kept_dist, kept_candidates, stats = _apply_tie_policy(
        items, distributions, candidates, policy="unique_only"
    )
    assert len(kept) == 1
    assert kept_dist == [None]
    assert kept_candidates == [None]
    assert stats["dropped_tied_rows"] == 1
