"""Insights for a first-time user, T0 (spec 2026-10-01 rev. 2): the lines
that survive the v3 Review stay in wording.py in 03-wording-audit.md's
style: they pass _BANNED, say none of the style guide's retired words, have
at most one dash and have a Trial form. The Model and method summary, the
status-rule table, the strips and map lines and their tests left with the
Insights tab."""
import re

import pytest

import wording as W
from tests.test_080_wording import _FROZEN
from tests.test_090_wording import _STYLE
from tests.test_food_bo import _ALLOWED_EXACT, _BANNED


class _SettingsOnly:
    """A study of process settings alone, as for_project asks it."""

    def has_ingredients(self):
        return False

    def _process_settings(self):
        return ["Bake temperature"]


def _fold_lines():
    return [W.FOLD_METHOD]


_GROUPS = [_fold_lines]


@pytest.mark.parametrize("group", _GROUPS)
def test_every_line_passes_the_vocabulary_guard(group):
    """The glossary is drawn from its own constants (below); nothing here
    leans on the allowance except Model and method's one lead sentence."""
    offenders = [t for t in group() if t not in _ALLOWED_EXACT
                 and any(p.search(t) for p in _BANNED)]
    assert offenders == []


@pytest.mark.parametrize("group", _GROUPS)
def test_no_line_uses_a_word_the_style_guide_retired(group):
    assert [t for t in group() if any(p.search(t) for p in _STYLE)] == []


@pytest.mark.parametrize("group", _GROUPS)
def test_at_most_one_dash_per_line(group):
    assert [t for t in group() if t.count("—") > 1] == []


@pytest.mark.parametrize("group", _GROUPS)
def test_trial_only_through_for_project(group):
    """No line types "trial"; on a study of process settings alone,
    for_project leaves no "formulation"."""
    opt = _SettingsOnly()
    for text in group():
        assert not re.search(r"\btrials?\b", text, re.I), text
        assert not re.search(r"\bformulations?\b", W.for_project(opt, text), re.I), text


@pytest.mark.parametrize("name,value", sorted(_FROZEN.items()))
def test_the_frozen_strings_did_not_move(name, value):
    assert getattr(W, name) == value
