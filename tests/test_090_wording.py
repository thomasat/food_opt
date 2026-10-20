"""0.8.0 BO views, Task 0: every line the wave changes or adds is spelled
once in wording.py, in 03-wording-audit.md's style (spec 2026-09-30-bo-views
§4, §8 rev. 2; review I7, M3, M6): named statistics with their basis, one
term per quantity, no hedges, and every line passes the vocabulary guard."""
import re

import pytest

import insights
import wording
from tests.test_food_bo import _BANNED

W = wording


def _changed_lines():
    """One call of every builder, and every constant, 03 §5 rewrites."""
    return [
        W.pairs_text(1),
        W.FOLD_METHOD,
        W.waiting_line(1, "Cook loss"),
        W.runner_up_line(15, "90.5", "0.2"),
        W.tie_equal(22, 15, "90.45"),
        W.OVERALL_SCORE_HELP,
    ]


_STYLE = [re.compile(p, re.I) for p in (
    r"\blikely\b", r"\btreat\b", r"\bthough\b", r"stands out", r"Nothing is predicted",
    r"\bworth\b", r"keep in mind", r"\brough\b", r"some luck", r"\bbarely\b", r"On track",
    r"too few to be sure", r"\ba little\b", r"\boptimiser\b", r"\bheld\b")]


@pytest.mark.parametrize("group", [_changed_lines])
def test_every_line_passes_the_vocabulary_guard(group):
    offenders = [t for t in group() if any(p.search(t) for p in _BANNED)]
    assert offenders == []


@pytest.mark.parametrize("group", [_changed_lines])
def test_no_line_uses_a_word_the_style_guide_retired(group):
    """03 §3 rules 3, 5, 7: one term per quantity, no hedges, no advice
    words; "likely range" is gone everywhere."""
    offenders = [t for t in group() if any(p.search(t) for p in _STYLE)]
    assert offenders == []


@pytest.mark.parametrize("group", [_changed_lines])
def test_at_most_one_dash_per_line(group):
    """03 §3 rule 9: no em-dash chains."""
    assert [t for t in group() if t.count("—") > 1] == []


def test_the_process_only_forms_say_trial_through_for_project():
    """A process-only study's lines are made from these through for_project;
    no line here types the word (review M3)."""
    for text in _changed_lines():
        assert not re.search(r"\btrials?\b", text, re.I), text


@pytest.mark.parametrize("value,text", [(-1.23, "−1.2"), (0.36, "+0.4"), (0.0, "0.0"),
                                        (-0.04, "0.0")])
def test_signed(value, text):
    assert W.signed(value) == text


def test_dead_lines_are_gone():
    """03 §5.7: unused lines that still carried the old words."""
    for name in ("score_record_line", "CHECKING", "TRACK_INSIDE"):
        assert not hasattr(W, name), name


