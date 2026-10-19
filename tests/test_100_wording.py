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
        W.open_saved_copy_warning("Pea burger patty", ["24 formulations"],
                                  "Pea burger patty", "28 formulations",
                                  W.open_round_lost(7, 4)),
    ]


def _new_lines():
    """One call of every builder, and every constant, the new section adds."""
    return [
        W.best_repeats_line(14, ["67.30", "66.34", "75.61"], "75.17"),
        W.best_repeats_line(26, ["77.97"], "79.90"),
        W.rest_needs_batch_size("Water"),
        W.import_limits_not_numbers("Date paste"),
        W.import_limits_not_numbers("Date paste", (W.HIGHEST_LABEL,)),
        W.import_spreadsheet_formula("Date paste", W.LOWEST_LABEL, "=B2*2"),
        W.open_round_lost(6, 4, 2), W.open_round_lost(7, 1, 1), W.open_round_lost(7, 4),
        W.YES_MOVE_TO_NOT_SCORED,
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


def test_one_vocabulary_for_role_and_compared_with():
    """A2 (02 I1): the Compared with cell says the three suggestion words."""
    assert (W.SUGGESTION_CLOSE, W.SUGGESTION_BALANCED, W.SUGGESTION_DIFFERENT) == (
        "close to the best", "balanced", "explores")
    assert W.compared_with_cell(W.SUGGESTION_BALANCED, "Salt +0.56 g") == (
        "Balanced · Salt +0.56 g")


