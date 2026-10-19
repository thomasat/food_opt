"""Final fix wave, item 6, as the v3 Review keeps it: opening the Review
fits no model, draws no progress bar and drops stale round checks on a
project switch. (The per-Round check cache itself left with the old engine.)"""
import pytest
import streamlit as st

import food_bo
import ui_insights
import wording


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


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
    real = food_bo.fit_gpytorch_mll
    monkeypatch.setattr(food_bo, "fit_gpytorch_mll", lambda *a, **k: fits.append(1) or real(*a, **k))
    at = _open_insights(_app())
    assert not at.exception
    assert steps == [] and fits == []
    assert not [t for t in spins if t.startswith("Checking")]
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
