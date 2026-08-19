from types import SimpleNamespace

import prepare_alpaca_cot_4066 as prepare
import run_alpaca_cot_stage4_mix as runner
import run_pointwise5answers_two_to_pairwise_v1 as base


class _CharacterTokenizer:
    model_max_length = 512
    eos_token = "<eos>"

    def __call__(self, text, add_special_tokens=False, truncation=False):
        del truncation
        ids = [ord(char) + 1 for char in str(text)]
        if add_special_tokens:
            ids.insert(0, 1)
        return SimpleNamespace(input_ids=ids)


def test_permuted_tie_ranking_is_canonical():
    assert prepare._map_ranking("A>B=C", {"A": "C", "B": "B", "C": "A"}) == "C>A=B"
    assert prepare._map_ranking("A=B>C", {"A": "C", "B": "B", "C": "A"}) == "B=C>A"


def test_synthetic_reason_discards_model_label():
    assert runner._clean_synthetic_reason("Useful comparison. [[2]]", task="pairwise") == "Useful comparison."
    assert (
        runner._clean_synthetic_reason("Useful ranking. Ranking:[C>B>A]", task="listwise")
        == "Useful ranking."
    )


def test_cot_smoothing_finds_final_score_instead_of_reason_digit():
    tokenizer = _CharacterTokenizer()
    score_ids = base._score_token_ids_for_sft(tokenizer, score_min=1, score_max=10)
    target = runner._cot_pointwise_target("The reason mentions score 2 first.", 2)
    dataset = base.SFTPairwiseDataset(
        ["prompt"],
        [target],
        tokenizer,
        pointwise_score_labels=[1],
        pointwise_score_token_ids=score_ids,
    )
    position = dataset.pointwise_score_positions[0]
    labels = dataset.labels[0].tolist()
    assert labels[position : position + len(score_ids[1])] == score_ids[1]
    assert position > next(index for index, token in enumerate(labels) if token == score_ids[1][0])


def test_private_pair_prompt_does_not_request_a_verdict():
    answer_a = base.AnswerWithScore("a", "answer a", 8, "strong")
    answer_b = base.AnswerWithScore("b", "answer b", 5, "weaker")
    triple = SimpleNamespace(instruction="Do it", input_text="", answer_a=answer_a, answer_b=answer_b)
    prompt = runner._private_pair_prompt(triple, answer_a, answer_b)
    assert "Private pointwise evidence" in prompt
    assert "Please output exactly one of" not in prompt
    assert "assessment=" not in prompt
    public_train_prompt = runner._pairwise_train_prompt("Do it", "", answer_a.output, answer_b.output)
    assert "Private pointwise evidence" not in public_train_prompt
    assert "score=" not in public_train_prompt
    enriched = runner._private_pair_prompt(
        triple, answer_a, answer_b, include_pointwise_assessments=True
    )
    assert "assessment=strong" in enriched


def test_cot_prompts_preserve_legacy_judging_rules_without_label_only_conflict():
    assert "Avoid any position bias" in runner.PAIRWISE_COT_SYSTEM_PROMPT
    assert "Do not favor longer responses" in runner.PAIRWISE_COT_SYSTEM_PROMPT
    assert "provide a brief explanation" in runner.PAIRWISE_COT_SYSTEM_PROMPT
    assert "Avoid position bias" in runner.LISTWISE_COT_SYSTEM_PROMPT
    assert "Do not favor longer responses" in runner.LISTWISE_COT_SYSTEM_PROMPT
    assert "First provide a brief explanation" in runner.LISTWISE_COT_SYSTEM_PROMPT
    prompt = runner._pairwise_train_prompt("Do it", "", "answer 1", "answer 2")
    assert "Please output exactly one of" not in prompt
