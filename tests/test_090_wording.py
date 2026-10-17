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
        W.TRUST_BEFOREHAND, W.FOLD_MEASURED_CHART, W.MVP_CAPTION,
        W.Q_STAND,
        W.stand_best("90.45", 15, 4), 
        W.pairs_text(1), 
        W.stand_limits(["the Highest of Pea"]), W.best_luck(11, "84", "62", "100", "90.09"),
        W.FOLD_PROGRESS_CHART, W.progress_caption_2(), 
        W.NEXT_RISING, 
        W.try_one_fold(True), W.try_one_fold(False), W.try_one_fold(True, False),
        W.near_line("Coconut oil", "1.4", "Best so far"),
        W.near_line("Coconut oil", "1.4", "Best so far", only=True),
        W.near_two("A", "B", "8.7", "8.2", "Best so far"), W.near_bigger(), W.near_none(),
        W.moved_together("Pea", "Soy", "0.86"), W.change_caption(), W.CHANGE_NEAR,
        W.CHANGE_WHOLE, W.CHANGE_AXIS, W.path_key(15, grey=True, targets=True), W.Q_PAIR,
        W.PAIR_CAPTION, W.pair_caption(["Firmness"]), W.pair_caption([]), W.Q_MIXTURE,
        W.mixture_caption(15), W.FOLD_METHOD, W.HOW_MODELS, W.how_checks(), W.how_checks(values=True), 
        W.waiting_line(1, "Cook loss"), W.tie_line(22, "90.65", "0.20", 15, "2.3"),
        W.tie_line(22, "90.65", "0.20", 15, "4.0"), W.tie_equal(22, 15, "90.45"),
        W.OVERALL_SCORE_HELP,
    ]


def _new_lines():
    """One call of every builder, and every constant, §§1-5 add."""
    return [
        W.start_suggested(23, edited=True),
        W.why_chart_caption("3,000", line=False),
        W.mixture_within_noise("4.1", "5.0"), W.GUARD_FLOOR, W.GUARD_NOISE,
        W.EXPECTED_IMPROVEMENT, W.NEAREST_TESTED,
        W.ROLE_REFINES, W.ROLE_BALANCED, W.ROLE_EXPLORES,
        W.EXPECTED_IMPROVEMENT_HELP, W.EXPECTED_IMPROVEMENT_EI_HELP,
        W.WHY_PENDING, 
        W.why_chart_caption(3000),
        W.why_no_roles(120), W.GAIN_HEADING,
        W.WHY_POOL, W.WHY_TESTED, W.WHY_SUGGESTED, W.WHY_RECOMPUTED,
        W.GAIN_BEST_MEASURED, W.GAIN_HIGHEST_PREDICTED, 
        W.GAIN_AXIS, W.GAIN_EXACT, W.GAIN_RECOMPUTED, W.GAIN_STATUS, W.GAIN_CHART_CAPTION,
        W.LAYER_PICK, W.LAYER_SD_LEGEND,
        W.LAYER_GAIN_LEGEND, W.OVERLAY_NAME, W.overlay_caption("88"), W.start_suggested(22),
    ]


_STYLE = [re.compile(p, re.I) for p in (
    r"\blikely\b", r"\btreat\b", r"\bthough\b", r"stands out", r"Nothing is predicted",
    r"\bworth\b", r"keep in mind", r"\brough\b", r"some luck", r"\bbarely\b", r"On track",
    r"too few to be sure", r"\ba little\b", r"\boptimiser\b", r"\bheld\b")]


@pytest.mark.parametrize("group", [_changed_lines, _new_lines])
def test_every_line_passes_the_vocabulary_guard(group):
    offenders = [t for t in group() if any(p.search(t) for p in _BANNED)]
    assert offenders == []


@pytest.mark.parametrize("group", [_changed_lines, _new_lines])
def test_no_line_uses_a_word_the_style_guide_retired(group):
    """03 §3 rules 3, 5, 7: one term per quantity, no hedges, no advice
    words; "likely range" is gone everywhere."""
    offenders = [t for t in group() if any(p.search(t) for p in _STYLE)]
    assert offenders == []


@pytest.mark.parametrize("group", [_changed_lines, _new_lines])
def test_at_most_one_dash_per_line(group):
    """03 §3 rule 9: no em-dash chains."""
    assert [t for t in group() if t.count("—") > 1] == []


def test_the_process_only_forms_say_trial_through_for_project():
    """A process-only study's lines are made from these through for_project;
    no line here types the word (review M3)."""
    for text in _new_lines() + _changed_lines():
        assert not re.search(r"\btrials?\b", text, re.I), text


@pytest.mark.parametrize("value,text", [(-1.23, "−1.2"), (0.36, "+0.4"), (0.0, "0.0"),
                                        (-0.04, "0.0")])
def test_signed(value, text):
    assert W.signed(value) == text


def test_dead_lines_are_gone():
    """03 §5.7: unused lines that still carried the old words."""
    for name in ("score_record_line", "CHECKING", "TRACK_INSIDE"):
        assert not hasattr(W, name), name


