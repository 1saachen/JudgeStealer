import run_newnew_one_answer_trueval_three_stage_sft as runner


def test_single_task_counts_put_the_entire_budget_in_one_task():
    assert runner._single_task_counts("pointwise", 600) == (600, 0, 0)
    assert runner._single_task_counts("pairwise", 600) == (0, 600, 0)
    assert runner._single_task_counts("listwise", 600) == (0, 0, 600)


def test_single_task_training_spec_uses_existing_stage_names():
    point = [("pointwise", "p", "t", 0)]
    pair = [("pairwise", "p", "t", 0)]
    listing = [("listwise", "p", "t", 0)]
    assert runner._single_task_training_spec("pointwise", point, pair, listing) == (
        point,
        "stage1_pointwise",
    )
    assert runner._single_task_training_spec("pairwise", point, pair, listing) == (
        pair,
        "stage2_pairwise",
    )
    assert runner._single_task_training_spec("listwise", point, pair, listing) == (
        listing,
        "stage3_listwise",
    )


def test_single_task_mode_is_explicit_in_runner_source():
    source = runner.__file__
    text = open(source, encoding="utf-8").read()
    assert '"single_task"' in text
    assert '--single-task' in text
    assert '"after_single_task"' in text
