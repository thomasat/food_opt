"""Demo build (owner, 2026-10-02): review-diag, review-guide and
review-plots merged, then the owner's rulings:

a. Make a round's status line reads "Review:", not "Last Round:" (the
   verdict covers several recent Rounds).
b. Fewer charts: no second "How the app chooses" in Explored, no Expected
   gain by Round, no Review copy of the mean-sd map, no Predictions
   sparkline; the dead strips chart is gone.
c. One action, one card: the Predictions card does not repeat the Next
   card's Repeat request (tests/test_review_diag.py pins the demos).
d. Try one at a limit: the answer line says the limit fact, in the chart
   heading's words."""
import os

import pytest

import bo_views
import ui_insights
import wording
from food_bo import FoodOptimizer
from tests import demo_projects as D
from tests.test_food_bo import _BANNED
from tests.test_review_cards import _install, _leaves, _open


def _words(text):
    return len(text.split())


def test_status_line_says_review_not_last_round():
    line = wording.review_status_line("Off", "Rising")
    assert line.startswith("Review: ") and "Last" not in line
    assert wording.REVIEW_STATUS_CHECKING.startswith("Review: ")
    plain = line.replace("*", "")
    assert _words(plain) <= 10, plain


def test_limit_fact_is_short_and_agrees_with_the_chart_heading():
    for end in ("Highest", "Lowest"):
        fact = wording.at_limit_fact("Tap water", end)
        assert fact == f"Tap water: already at its {end}"
        assert _words(fact) <= 8
        assert f"at its {end}" in wording.path_head_at_limit("Tap water", end, "42 g")
        assert not any(p.search(fact) for p in _BANNED)


def test_limit_facts_name_only_choices_at_a_limit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _install(D.LINE_A)
    opt = FoodOptimizer(D.LINE_A)
    opt.load()
    names = ui_insights.insights.path_choices(opt)
    lo, hi = (float(b) for b in ui_insights.insights._var(opt, names[0])["bounds"])
    mid = (lo + hi) / 2
    recipe = {n: mid for n in names}
    assert ui_insights._limit_facts(opt, recipe, names[:1]) == ""
    recipe[names[0]] = hi
    assert ui_insights._limit_facts(opt, recipe, names[:1]) == (
        f" {names[0]}: already at its Highest.")
    recipe[names[0]] = lo
    assert ui_insights._limit_facts(opt, recipe, names[:1]).endswith("at its Lowest.")
    assert ui_insights._limit_facts(opt, None, names[:1]) == ""


def test_removed_charts_are_gone():
    for name in ("strips_chart", "_chooses", "_explored_map", "_gain_chart_body"):
        assert not hasattr(ui_insights, name), name


@pytest.fixture(scope="module")
def line_a(tmp_path_factory):
    here = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("demo-build"))
    try:
        with bo_views.fresh():
            _install(D.LINE_A)
            yield _open(D.LINE_A)
    finally:
        os.chdir(here)


def _charts_by_card(at):
    out = {}
    for el, folds, col in _leaves(at.tabs[3]):
        if col is not None and not folds and "chart" in el.type:
            out[col] = out.get(col, 0) + 1
    return out


def test_only_the_progress_card_has_a_sparkline(line_a):
    by_card = _charts_by_card(line_a)
    assert 0 not in by_card, by_card          # Predictions
    assert by_card.get(1, 0) == 1, by_card    # Progress
    assert 2 not in by_card, by_card          # Next
