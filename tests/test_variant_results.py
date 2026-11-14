import pytest

from eval.summarize_variants import summarize


@pytest.mark.parametrize("response", [
    "There is insufficient information", "E", "The answer is X", "", None,
])
def test_egolife_summary_counts_invalid_choices_as_failures(response):
    row = {"ID": "q1", "completed": True, "response": response,
           "choices": {"A": "one", "B": "two", "C": "three", "D": "four"},
           "evaluate": False}
    result = summarize(["q1"], [row])
    assert result["completed"] == 0
    assert result["failed"] == 1
    assert result["failures"][0]["ID"] == "q1"


def test_egolife_completion_requires_a_choice_present_in_the_question():
    row = {"ID": "q1", "completed": True, "response": "Final answer: D.",
           "choices": {"A": "one", "B": "two"}, "evaluate": False}
    assert summarize(["q1"], [row])["failed"] == 1


def test_egolife_wrong_but_valid_answer_is_completed():
    row = {"ID": "q1", "completed": True, "response": "Final answer: B.",
           "choices": {"A": "one", "B": "two"}, "evaluate": False}
    result = summarize(["q1", "q2"], [row])
    assert (result["completed"], result["correct"], result["failed"], result["missing"]) == (1, 0, 0, 1)


def test_explicit_error_status_does_not_become_success_from_completed_flag():
    row = {"ID": "q1", "status": "error", "completed": True,
           "response": "A", "choices": {"A": "one"}, "evaluate": True}
    result = summarize(["q1"], [row])
    assert (result["completed"], result["correct"], result["failed"]) == (0, 0, 1)
