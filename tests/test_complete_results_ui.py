"""Complete results only (0.8.0) on screen: Results, Make a round, Insights.
Spec: docs/superpowers/specs/2026-09-28-complete-results-design.md (rev. 2)."""
import pytest
from streamlit.testing.v1 import AppTest

import wording
from food_bo import FoodOptimizer
from tests.test_app_ui import (  # noqa: F401  (fixtures)
    APP_PATH, _submit_button, burger, open_batch, scored)


def _app(tab=None):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    if tab is not None:
        at.session_state["main_tab"] = tab
        at.run()
    return at


def _firmness_only(opt, numbers, round_no=1):
    """Formulations with Firmness measured and Juiciness still to come."""
    for no in numbers:
        opt.tell({"Pea protein": 8.0 + no, "Methylcellulose": 0.5 + 0.2 * no},
                 {"Firmness": 3.0 + 0.3 * no}, formulation_no=no,
                 batch_no=round_no)


@pytest.fixture
def all_waiting(burger):
    """Two formulations, neither with Juiciness yet: no best at all."""
    _firmness_only(burger, (1, 2))
    return burger


@pytest.fixture
def one_waiting(scored):
    """`scored` (1 and 2 complete, 3 Not scored) plus 4, waiting for
    Juiciness, in Round 2."""
    _firmness_only(scored, (4,), round_no=2)
    return scored


@pytest.fixture
def six_waiting(burger):
    """Six formulations over two Rounds, all waiting: Insights opens."""
    _firmness_only(burger, (1, 2, 3), 1)
    _firmness_only(burger, (4, 5, 6), 2)
    return burger


# ---------------------------------------------------------------- Task 4


def test_results_with_no_complete_formulation_names_no_best(all_waiting):
    at = _app(wording.TAB_RESULTS)
    assert not at.exception
    assert not any(h.value.startswith("Best so far:") for h in at.subheader), \
        [h.value for h in at.subheader]


def test_make_a_round_with_no_complete_formulation_offers_no_best(all_waiting):
    at = _app(wording.TAB_BATCH)
    assert not at.exception
    assert wording.START_FROM_BEST not in [b.label for b in at.button]


def test_insights_with_no_complete_formulation(six_waiting):
    at = _app(wording.TAB_INSIGHTS)
    assert not at.exception
    start = at.tabs[3].selectbox(key="insights_start")
    assert start.options[0] == wording.start_formulation(6)
