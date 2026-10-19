"""Final fix wave, item 6: the Round-ahead checks are kept per Round, so a
rerun after one new result refits only the Rounds whose inputs changed, and
the answer is exactly the one check_rounds gives without the keep."""
import torch
import pytest
import streamlit as st

import insights
import ui_insights
import wording
from tests.test_insights import _burger


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _count_fits(monkeypatch):
    """Every insights.fit call, as the number of rows it was offered: a
    Round's check fits on the rows before that Round's first row, so the
    count names the Round it was for."""
    seen = []
    real = insights.fit
    def spy(opt, name, rows):
        seen.append(len(rows))
        return real(opt, name, rows)
    monkeypatch.setattr(insights, "fit", spy)
    return seen


def _firsts(opt):
    return {r: first for r, first in insights.round_order(opt)}


def _projects():
    a = _burger(rounds=4, per_round=3, noise=0.4, seed=1)
    b = _burger(rounds=5, per_round=2, noise=0.3, seed=2, fill=False)
    c = _burger(rounds=4, per_round=3, noise=0.4, seed=3)
    # a partly measured row in the last Round and a formulation with no Round
    c.tell(c.fill_formulas({"Pea": 6.0, "Oil": 4.0}), {"Juiciness": 5.0},
           formulation_no=40, batch_no=4)
    c.tell(c.fill_formulas({"Pea": 5.0, "Oil": 3.0}), {"Firmness": 6.0, "Juiciness": 4.0},
           formulation_no=41)
    d = _burger(rounds=4, per_round=3, noise=0.4, seed=4)
    d.bo_config = {"noise": "fixed_tiny"}
    return [a, b, c, d]


@pytest.mark.parametrize("k", range(4))
def test_kept_checks_are_exactly_the_fresh_ones(k):
    opt = _projects()[k]
    fresh = insights.check_rounds(opt)
    cache = {}
    cold = insights.check_rounds(opt, cache=cache)
    warm = insights.check_rounds(opt, cache=cache)
    assert fresh and cold == fresh and warm == fresh
    # One entry per Round, and a warm cache is left as it was.
    assert len(cache) == len(insights.round_order(opt))


def test_a_warm_cache_fits_nothing(monkeypatch):
    opt = _burger(rounds=4, per_round=3, noise=0.4)
    cache = {}
    insights.check_rounds(opt, cache=cache)
    seen = _count_fits(monkeypatch)
    insights.check_rounds(opt, cache=cache)
    assert seen == []


def test_adding_one_round_refits_only_that_round(monkeypatch):
    opt = _burger(rounds=4, per_round=3, noise=0.4)
    cache = {}
    insights.check_rounds(opt, cache=cache)
    g = torch.Generator().manual_seed(11)
    for no in (13, 14, 15):
        u = torch.rand(2, generator=g).tolist()
        pea, oil = 4 + 4 * u[0], 2 + 4 * u[1]
        opt.tell(opt.fill_formulas({"Pea": pea, "Oil": oil}),
                 {"Firmness": 2 + 0.9 * pea, "Juiciness": 1 + 1.2 * oil},
                 formulation_no=no, batch_no=5)
    seen = _count_fits(monkeypatch)
    kept = insights.check_rounds(opt, cache=cache)
    outputs = len(insights.outputs(opt))
    assert seen == [_firsts(opt)[5]] * outputs            # Round 5's fits alone
    monkeypatch.undo()
    assert kept == insights.check_rounds(opt)
    assert len(cache) == 5


def test_editing_an_old_value_refits_that_round_and_every_later_one(monkeypatch):
    opt = _burger(rounds=5, per_round=3, noise=0.4)
    cache = {}
    insights.check_rounds(opt, cache=cache)
    firsts = _firsts(opt)
    i = firsts[3] + 1                                       # a row of Round 3
    edited = dict(opt.results_history[i], Firmness=opt.results_history[i]["Firmness"] + 1.0)
    opt.edit_result(i, edited)
    seen = _count_fits(monkeypatch)
    kept = insights.check_rounds(opt, cache=cache)
    outputs = len(insights.outputs(opt))
    assert sorted(seen) == sorted([firsts[3]] * outputs + [firsts[4]] * outputs
                                  + [firsts[5]] * outputs)
    monkeypatch.undo()
    assert kept == insights.check_rounds(opt)
    # The old Round 3, 4 and 5 entries are gone: one entry per Round.
    assert len(cache) == 5


@pytest.mark.parametrize("change", ["objectives", "bo_config", "bounds"])
def test_a_model_setting_refits_every_round(monkeypatch, change):
    opt = _burger(rounds=4, per_round=3, noise=0.4)
    cache = {}
    insights.check_rounds(opt, cache=cache)
    if change == "objectives":
        opt.objectives[0]['max_val'] = 12.0
    elif change == "bo_config":
        opt.bo_config = {"noise": "fixed_tiny"}
    else:
        opt._by_name()["Oil"]['bounds'] = [1.0, 6.0]
    seen = _count_fits(monkeypatch)
    kept = insights.check_rounds(opt, cache=cache)
    assert set(seen) == {first for r, first in insights.round_order(opt)}
    monkeypatch.undo()
    assert kept == insights.check_rounds(opt)


def test_kept_checks_leave_the_project_rng_and_next_suggestion_untouched():
    a, b = _burger(rounds=3, noise=0.4), _burger(rounds=3, noise=0.4)
    before = b.export_json()
    rng = torch.get_rng_state().clone()
    cache = {}
    insights.check_rounds(b, cache=cache)
    insights.check_rounds(b, cache=cache)
    assert torch.equal(torch.get_rng_state(), rng)
    assert b.export_json() == before
    assert a.ask(2) == b.ask(2)


def test_each_round_fitted_is_named_while_it_is_checked():
    opt = _burger(rounds=4, per_round=3, noise=0.4)        # Rounds 3 and 4 checkable
    named = []
    class _Line:
        def __init__(self, r, last):
            self.r, self.last = r, last
        def __enter__(self):
            named.append((self.r, self.last))
        def __exit__(self, *exc):
            return False
    cache = {}
    insights.check_rounds(opt, cache=cache, checking=_Line)
    assert named == [(3, 4), (4, 4)]
    named.clear()
    insights.check_rounds(opt, cache=cache, checking=_Line)
    assert named == []                                      # nothing fitted, nothing named


def test_checking_wording():
    """The one progress line (CHECKING was dead code; 03 §5.7 deleted it)."""
    assert wording.insights_progress(0, 3).startswith("Checking predictions against your results")


# ------------------------------------------------------------------ #
#  In the app
# ------------------------------------------------------------------ #

from tests.test_insights_ui import _app, _open_insights, insights_scored  # noqa: E402,F401
from food_bo import FoodOptimizer  # noqa: E402
from tests.test_app_ui import _submit_button, burger  # noqa: E402,F401
from tests.test_insights_wave_c_ui import ferment_scored  # noqa: E402,F401


def _progress_steps(monkeypatch):
    """Every (value, text) the one progress bar is set to, and every
    spinner line, across the runs that follow."""
    from streamlit.delta_generator import DeltaGenerator
    steps, spins = [], []
    real_progress, real_spinner = DeltaGenerator.progress, st.spinner

    def progress(self, value, text=None, **k):
        steps.append((value, text))
        return real_progress(self, value, text=text, **k)

    def spinner(text="", *a, **k):
        spins.append(text)
        return real_spinner(text, *a, **k)
    monkeypatch.setattr(DeltaGenerator, "progress", progress)
    monkeypatch.setattr(st, "spinner", spinner)
    return steps, spins


def test_the_review_fits_nothing_and_draws_no_progress_bar(insights_scored, monkeypatch):
    """Review v3 draws from the measurements alone: no model is fitted, so
    no progress bar or spinner is shown while the tab opens (the bar of the
    old Insights tab is gone)."""
    steps, spins = _progress_steps(monkeypatch)
    fits = []
    real = insights.fit
    monkeypatch.setattr(insights, "fit", lambda *a, **k: fits.append(1) or real(*a, **k))
    at = _open_insights(_app())
    assert not at.exception
    assert steps == [] and fits == []
    assert not [t for t in spins if t.startswith("Checking")]
    assert spins.count(wording.INSIGHTS_WORKING) == 0
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    assert steps == [] and fits == []


def test_a_process_only_project_shows_no_bar_either(ferment_scored, monkeypatch):
    steps, _ = _progress_steps(monkeypatch)
    at = _open_insights(_app())
    assert not at.exception
    assert steps == []


def test_the_round_checks_are_dropped_on_project_switch(insights_scored):
    at = _open_insights(_app())
    # Review v3 fits nothing: the key as another screen's work leaves it.
    at.session_state[ui_insights.ROUNDS_KEY] = {"stale": True}
    FoodOptimizer("second").add_ingredient("Flour", 0, 100)
    at.run()
    at.selectbox(key="project_select").set_value("second")
    at.run()
    _submit_button(at.sidebar, "Open").click()
    at.run()
    assert not at.exception
    assert ui_insights.ROUNDS_KEY not in at.session_state
