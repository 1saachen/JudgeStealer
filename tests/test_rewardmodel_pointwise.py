from pathlib import Path
from types import SimpleNamespace

import numpy as np

from run_rewardmodel_three_stage_sft import (
    _apply_tie_policy,
    _best_choice_listwise_items,
    _best_choice_prompt,
    _best_choice_target,
    _best_choice_metrics,
    _listwise_best_choice_metadata,
    _native_pairwise_items,
    _parse_best_choice,
    _pointwise_prompt,
    _listwise_soft_choice_metadata,
    _pairwise_soft_choice_metadata,
    _sample_training_items,
    _selection_args,
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


def test_pairwise_explicit_tie_stays_hard_instead_of_softening():
    rows = [{"record_id": 7, "pair_name": "AB", "pairwise_label": 2}]
    records = [{"id": 7, "pairwise": {"AB": {"scoreA": 4.5, "scoreB": 4.5}}}]
    distributions, candidates, stats = _pairwise_soft_choice_metadata(rows, records, eos="</s>")
    assert distributions == [None]
    assert candidates == [None]
    assert stats["explicit_tie"] == 1


def test_pairwise_native_target_uses_choice_without_emitting_scores():
    records = [
        {
            "id": 7,
            "instruction": "Choose.",
            "input": "",
            "outputA": "answer A",
            "outputB": "answer B",
            "pairwise": {
                "AB": {"scoreA": 5.0, "scoreB": 1.0, "choice": "B"},
            },
        }
    ]

    items = _native_pairwise_items(records, decimals=4)

    assert len(items) == 1
    assert items[0][2].startswith("[[2]]")
    assert "5.0" not in items[0][2]


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


def test_listwise_target_uses_source_best_choice_only():
    records = [
        {
            "id": 7,
            "instruction": "Choose the best answer.",
            "input": "",
            "outputA": "answer A",
            "outputB": "answer B",
            "outputC": "answer C",
            "listwise_scoreA": 5.0,
            "listwise_scoreB": 1.0,
            "listwise_scoreC": 4.0,
            "listwise_choice": "C",
        }
    ]

    items = _best_choice_listwise_items(records)

    assert len(items) == 1
    target = items[0][2]
    assert target.startswith("3")
    assert "Best:" not in target
    assert "Response" not in target
    assert "5.0" not in target
    assert "Ranking" not in target


def test_converted_prompts_keep_unirrm_evaluation_protocol():
    answer = _answer(7, "A")
    answer = SkyworkAnswer(
        question_id=7,
        source_id=7,
        dataset="test",
        instruction="Explain recursion.",
        input_text="",
        answer_key="A",
        model="model",
        output="A recursive answer.",
        reward=4.5,
    )
    point_prompt = _pointwise_prompt(answer)
    list_prompt = _best_choice_prompt(
        {
            "instruction": "Choose the best answer.",
            "input": "",
            "outputA": "answer A",
            "outputB": "answer B",
            "outputC": "answer C",
        }
    )
    assert "### Phase 1: Deep Analysis" in point_prompt
    assert "### Phase 2: Dynamic Rubric Generation" in point_prompt
    assert "<User_Input>" in point_prompt
    assert "<Response1>" in point_prompt
    assert point_prompt.endswith("Score: [")
    assert "### Phase 1: Deep Analysis" in list_prompt
    assert "Return only one integer: 1, 2, or 3." in list_prompt
    assert "Best: [Response" not in list_prompt


def test_listwise_target_and_parser_use_bare_response_numbers():
    assert _best_choice_target("B") == "2</s>"
    assert _parse_best_choice("2</s>") == "Response2"
    assert _parse_best_choice(" 3<|im_end|>") == "Response3"


def test_listwise_equal_top_scores_use_soft_best_choice_targets():
    records = [
        {
            "id": 7,
            "instruction": "Choose the best answer.",
            "input": "",
            "outputA": "answer A",
            "outputB": "answer B",
            "outputC": "answer C",
            "listwise_scoreA": 5.0,
            "listwise_scoreB": 5.0,
            "listwise_scoreC": 2.0,
            "listwise_choice": "A",
        }
    ]

    distributions, candidates, truth_groups, stats = _listwise_best_choice_metadata(
        records, eos="</s>"
    )

    assert distributions == [{"0": 0.5, "1": 0.5}]
    assert candidates == [["1</s>", "2</s>"]]
    assert truth_groups == [["Response1", "Response2"]]
    assert stats["tied_winner"] == 1


def test_best_choice_parser_and_accuracy():
    assert _parse_best_choice("Best: [Response2]</s>") == "Response2"
    assert _parse_best_choice("best choice: C") == "Response3"
    assert _parse_best_choice("not a valid choice") is None

    metrics = _best_choice_metrics(
        ["Response1", "Response2", "Response3"],
        ["Response1", "Response2", None],
    )
    assert metrics["sft_acc"] == 2 / 3
    assert metrics["sft_top_group_acc"] == 2 / 3
    assert metrics["sft_invalid_pred"] == 1


def test_mix_sampling_keeps_items_and_metadata_aligned():
    items = [("pairwise", f"prompt-{index}", f"target-{index}", -100) for index in range(6)]
    distributions = [f"distribution-{index}" for index in range(6)]
    candidates = [f"candidate-{index}" for index in range(6)]

    sampled_items, sampled_distributions, sampled_candidates = _sample_training_items(
        items,
        distributions,
        candidates,
        samples=2,
        seed=42,
    )

    assert len(sampled_items) == 2
    for item, distribution, candidate in zip(
        sampled_items, sampled_distributions, sampled_candidates
    ):
        index = int(item[1].rsplit("-", 1)[1])
        assert distribution == f"distribution-{index}"
        assert candidate == f"candidate-{index}"


def test_selector_budget_units_is_forwarded_to_the_selector():
    args = SimpleNamespace(
        budget_units=603,
        seed=42,
        selector_init_questions=80,
        selector_batch_size=20,
        selector_pool_size=100,
        llama="models/Llama-3.2-1b-instruct",
        proxy_lr=1e-4,
        proxy_max_length=768,
        load_in_4bit=True,
        use_lora=True,
        gradient_accumulation_steps=16,
        selector_proxy_warmup_epochs=3,
        selector_proxy_update_epochs=1,
        smooth_alpha=0.1,
        proxy_mc_samples=4,
    )

    selection_args = _selection_args(args, Path("outputs/test"))

    assert selection_args.budget_units == 603
