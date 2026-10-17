"""0.8.0 e2e fixes, Task 0: every line W1, W2 and W3 need is spelled once
in wording.py (spec 2026-10-01-e2e-fixes-design.md rev. 2; every "Proposed
ruling" of research/2026-10-01-e2e/spec-review.md), in 03-wording-audit.md's
style: it passes _BANNED and the retired-word list, has at most one dash,
says "trial" only through for_project, and the frozen strings stay."""
import re

import pytest

import wording as W
from tests.test_080_wording import _FROZEN
from tests.test_090_wording import _STYLE
from tests.test_food_bo import _BANNED


class _SettingsOnly:
    """A study of process settings alone, as for_project asks it."""

    def has_ingredients(self):
        return False

    def _process_settings(self):
        return ["Baking temperature"]


def _changed_lines():
    """One call of every builder, and every constant, the rulings edit in
    place."""
    return [
        W.SUGGESTION_CLOSE, W.SUGGESTION_BALANCED, W.SUGGESTION_DIFFERENT,
        W.compared_with_cell(W.SUGGESTION_DIFFERENT, "Water +12.00 g"), W.HOW_IT_WORKS[-1],
        W.how_models(), W.how_models("matern52"), W.how_models("rbf", "fixed_tiny"),
        W.how_models("linear", "low"), W.HOW_MODELS, W.how_checks(), W.how_checks(values=True),
        W.how_checks(total=False), W.beat_by(), W.beat_by("1.3"), W.beat_by(assumed=True),
        W.beat_by(assumed=True, pairs=1), W.beat_by(assumed=True, pairs=2), W.score_gap_line(30, 32, -4.3, rounds=[2, 3, 4, 5, 6, 7]),
        W.score_gap_line(11, 11, 0.0, rounds=[3]), W.repeat_disagrees(21, 11, "10.5",
                                                                     half_text="7.4"),
        W.repeat_disagrees(20, 14, "1.6", lower=False), W.stand_repeats("9.6", 8, 5, 3),
        W.stand_repeats("2.7", 1, 1, 1), W.stand_repeats("1.3", 4), W.CHECK_IF_CHANGED,
        W.next_bias("4.1"), W.next_bias("3.3", below=False, rounds=[3, 4, 5, 6]),
        W.next_misses([5, 6, 7]), W.next_misses([5, 6, 7], W.next_bias("4.1", rounds=[3, 7])),
        W.NEXT_MISSES, W.why_guard_line(["x"]), W.why_guard_line(skill_only=True),
        W.why_guard_line(verdict=True), W.WHY_CHECKING, W.ROLE_REFINES, W.ROLE_BALANCED,
        W.ROLE_EXPLORES, W.EXPECTED_IMPROVEMENT_HELP, W.EXPECTED_IMPROVEMENT_EI_HELP,
        W.WHY_PENDING, W.gain_line("2.66", None, W.beat_by()),
        W.gain_line("0.30", 6, W.beat_by(assumed=True), measured_text="91.69"),
        W.gain_line("1.08", 13, W.beat_by(), above=True, predicted_none=True),
        W.gain_line("0.12", 5, W.beat_by("1.3"), recomputed=True), W.LAYER_GAIN,
        W.LAYER_GAIN_LEGEND, W.open_saved_copy_warning("Pea burger patty", ["24 formulations"],
                                                       "Pea burger patty", "28 formulations",
                                                       W.open_round_lost(7, 4)),
    ]


def _new_lines():
    """One call of every builder, and every constant, the new section adds."""
    return [
        W.bias_item("Overall score", "4.3"), W.bias_item("Firmness", "3.0", "N"),
        W.bias_item("Juiciness", "0.4", ""), W.bias_item("Overall score", "1.0"),
        W.steady_bias_line([W.bias_item("Overall score", "4.3")], rounds=[3, 4, 5, 6, 7]),
        W.steady_bias_line(low=[W.bias_item("Liking", "3.2")], rounds=[3, 4, 5]),
        W.steady_bias_line([W.bias_item("Overall score", "4.3")],
                           [W.bias_item("Juiciness", "0.4", "")], [3, 4]),
        W.effects_wait_line(6), W.effects_wait_line(1, settings_only=True),
        W.effects_unclear_line("0.09", 24), W.effects_unclear_line("0.12", 12, True),
        W.gain_unreliable(W.gain_reason_too_few(6)),
        W.gain_unreliable(W.gain_reason_unclear("0.09", 24)), W.NEXT_REPEAT_TO_MEASURE, W.repeat_score("67.30"),
        W.repeat_score("75.61", disagrees=True),
        W.best_repeats_line(14, ["67.30", "66.34", "75.61"], "75.17"),
        W.best_repeats_line(14, ["67.30", W.repeat_score("75.61", True)], "78.12"),
        W.best_repeats_line(26, ["77.97"], "79.90"),
        W.setup_callout("Textured pea protein", "30 g"),
        W.setup_callout("Salt", "0.5 g", highest=False),
        W.setup_callout_limit("Fat per 100 g: at most 20 g"), W.rest_needs_batch_size("Water"),
        W.import_limits_not_numbers("Date paste"),
        W.import_limits_not_numbers("Date paste", (W.HIGHEST_LABEL,)),
        W.import_spreadsheet_formula("Date paste", W.LOWEST_LABEL, "=B2*2"),
        W.open_round_lost(6, 4, 2), W.open_round_lost(7, 1, 1), W.open_round_lost(7, 4),
        W.YES_MOVE_TO_NOT_SCORED, W.gain_pending(), W.gain_pending(7), W.INSIGHTS_IN_PROCESS,
        W.role_help(), W.TIP_NEAREST_TESTED, W.LARGEST_DIFFERENCES_HELP,
        W.LAYER_SCORE_LEGEND, W.LAYER_GAIN_FORMER, W.KERNEL_NAMES["matern32"],
        # T0′ at integration: the interval chart's caption in both themes,
        # the progress line with every Round checked, the Why fold's A3
        # reasons in Progress's words.
        W.MVP_CAPTION, W.insights_progress(12, 12), W.insights_progress(1, 1),
        W.insights_progress(3, 12),
        W.why_guard_line(gate_reason=W.gain_reason_too_few(6)),
        W.why_guard_line(gate_reason=W.gain_reason_unclear("0.09", 24)),
        W.why_guard_line(gate_reason=W.GAIN_REASON_SKILL),
    ]


_GROUPS = [_changed_lines, _new_lines]


@pytest.mark.parametrize("group", _GROUPS)
def test_every_line_passes_the_vocabulary_guard(group):
    offenders = [t for t in group() if any(p.search(t) for p in _BANNED)]
    assert offenders == []


@pytest.mark.parametrize("group", _GROUPS)
def test_no_line_uses_a_word_the_style_guide_retired(group):
    """03 §3 rules 3, 5, 7. The I2 ruling's "treat new predictions as rough"
    is said without the retired words (next_misses)."""
    assert [t for t in group() if any(p.search(t) for p in _STYLE)] == []


@pytest.mark.parametrize("group", _GROUPS)
def test_at_most_one_dash_per_line(group):
    assert [t for t in group() if t.count("—") > 1] == []


@pytest.mark.parametrize("group", _GROUPS)
def test_trial_only_through_for_project(group):
    """No line types "trial"; on a study of process settings alone,
    for_project leaves no "formulation" in any of them."""
    opt = _SettingsOnly()
    for text in group():
        assert not re.search(r"\btrials?\b", text, re.I), text
        assert not re.search(r"\bformulations?\b", W.for_project(opt, text), re.I), text


def test_the_settings_only_forms_name_no_ingredient():
    """A line that names ingredients has a form for process settings alone."""
    for text in (W.effects_wait_line(6, settings_only=True),
                 W.effects_unclear_line("0.09", 24, settings_only=True),
                 W.how_checks(values=True, total=False)):
        assert "ingredient" not in text.lower(), text


def test_process_only_round_heading_says_trials():
    """B4 (04 M-3): the heading reaches the screen through for_project."""
    opt = _SettingsOnly()
    assert W.for_project(opt, W.make_these(4, 3)) == "**Round 4 · Make 3 trials**"
    assert W.for_project(opt, W.make_these(4, 1)) == "**Round 4 · Make 1 trial**"


@pytest.mark.parametrize("name,value", sorted(_FROZEN.items()))
def test_the_frozen_strings_did_not_move(name, value):
    assert getattr(W, name) == value


def test_import_aliases_are_frozen():
    """B1: a file written for these must keep loading."""
    assert W.FILL_TO_TOTAL_IMPORT_ALIASES == ("fill to total", "= rest", "=rest")
    assert W.LAYER_GAIN_FORMER == "Expected improvement"   # charts proto: one name, Expected gain


def test_one_vocabulary_for_role_and_compared_with():
    """A2 (02 I1): the Why fold's Role and the Compared with cell say the
    same three words."""
    assert (W.ROLE_REFINES, W.ROLE_BALANCED, W.ROLE_EXPLORES) == (
        W.SUGGESTION_CLOSE, W.SUGGESTION_BALANCED, W.SUGGESTION_DIFFERENT) == (
        "close to the best", "balanced", "explores")
    assert W.compared_with_cell(W.SUGGESTION_BALANCED, "Salt +0.56 g") == (
        "Balanced · Salt +0.56 g")


def test_no_line_advises_a_check_without_repeat_evidence():
    """Review B1, I1: only a disagreeing Repeat advises a check, and
    conditionally."""
    for text in (W.next_bias("4.1"), W.next_misses([5, 6, 7], W.next_bias("4.1")),
                 W.steady_bias_line([W.bias_item("Overall score", "4.3")])):
        assert "Check" not in text and "check it" not in text, text
    assert W.repeat_disagrees(21, 11, "10.5").endswith(W.CHECK_IF_CHANGED)
    assert W.CHECK_IF_CHANGED.startswith("If something changed")


