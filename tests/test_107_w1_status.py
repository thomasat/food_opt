"""Insights for a first-time user, W1 (spec 2026-10-01 rev. 2; review
R-I9): the tie rule. insights.tie_partner is the next highest different
formulation, never a Repeat, and Results asks it."""
import pytest

import insights
import ui_results
from tests.test_complete_results import _patty


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


# ------------------------------------------------------------------ #
#  Ties (R-I9): equal scores only, at two decimals; a Repeat is never one
# ------------------------------------------------------------------ #

def test_tie_partner_is_the_next_highest_different_formulation():
    opt = _patty(rows=6)
    order = sorted(opt.best_candidates(), key=lambda i: float(opt.Y_history[i]), reverse=True)
    best = order[0]
    assert insights.tie_partner(opt, best) == order[1]


def test_a_repeat_is_never_the_tie_partner():
    opt = _patty(rows=6)
    order = sorted(opt.best_candidates(), key=lambda i: float(opt.Y_history[i]), reverse=True)
    best, second = order[0], order[1]
    opt.recipe_history[second] = dict(opt.recipe_history[best])
    opt.X_history[second] = list(opt.X_history[best])
    assert insights.tie_partner(opt, best) == order[2]


def test_results_asks_insights_for_the_tie_partner(monkeypatch):
    opt = _patty(rows=6)
    monkeypatch.setattr(insights, "tie_partner", lambda opt, best: "asked")
    assert ui_results._tie_partner(opt, 0) == "asked"
