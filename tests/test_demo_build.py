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

import ui_insights
from tests import demo_projects as D
from tests.review_screens import _install, _leaves, _open


def test_removed_charts_are_gone():
    for name in ("strips_chart", "_chooses", "_explored_map", "_gain_chart_body"):
        assert not hasattr(ui_insights, name), name


@pytest.fixture(scope="module")
def line_a(tmp_path_factory):
    here = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("demo-build"))
    try:
        _install(D.LINE_A)
        yield _open(D.LINE_A)
    finally:
        os.chdir(here)


def test_the_strip_has_no_sparkline(line_a):
    """Review story 2: the Progress sparkline went with the cards; the
    improving chart sits right under the strip. No chart above the first
    section."""
    for el, folds, _ in _leaves(line_a.tabs[3]):
        if el.type == "markdown" and el.value.startswith("#### "):
            break
        assert "chart" not in el.type, el.type
