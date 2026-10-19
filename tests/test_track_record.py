"""0.8.0 review fixes, Task 5: the Insights header and "Did the predictions
come true?" — scored formulations counted apart from waiting ones, a
past-tense verdict about the likely range, a wide-range warning, the
Overall score's own record always in view, and the typical miss compared
with the spread between Repeats instead of R² (spec 2026-09-29 §3.1-3.5,
§9 B1, I7, I8, I9)."""
import pytest
from streamlit.testing.v1 import AppTest

import insights
from food_bo import FoodOptimizer
import ui_insights
import wording
from tests.test_app_ui import APP_PATH, burger  # noqa: F401
from tests.test_complete_results import _patty, _tell
from tests.test_insights_ui import _open_insights, insights_scored  # noqa: F401


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


# ------------------------------------------------------------------ #
#  §3.1 / §3.5 the header
# ------------------------------------------------------------------ #

def test_headline_text_leaves_out_waiting_and_set_aside_rows():
    opt = _patty(rows=6)
    want = ui_insights.headline_text(opt, ui_insights.round_view(opt))
    _tell(opt, 7, 3, drop=("Cook loss",))
    _tell(opt, 8, 3, drop=("Cook loss",))
    opt.set_aside_formulation(8)
    view = ui_insights.round_view(opt)
    assert view.scored == [0, 1, 2, 3, 4, 5]
    assert ui_insights.headline_text(opt, view) == want


def _recorded(opt, complete, waiting):
    """`complete` formulations with both results in Round 1, then `waiting`
    with Firmness only (Juiciness to come) in Round 2."""
    for no in range(1, complete + waiting + 1):
        values = {"Firmness": 3.0 + 0.3 * no}
        if no <= complete:
            values["Juiciness"] = 4.0 + 0.2 * no
        opt.tell({"Pea protein": 8.0 + no, "Methylcellulose": 0.5 + 0.2 * no}, values,
                 formulation_no=no, batch_no=1 if no <= complete else 2)


@pytest.mark.parametrize("complete", [3, 4])
def test_the_headline_counts_only_the_scored_formulations(burger, complete):
    """Review I-1: with one of 4 (or 5) recorded formulations waiting, the
    headline is about the scored ones (the waiting one is not Best so far
    and is not named), and a waiting one does not open the checks early."""
    _recorded(burger, complete, 1)
    at = _open_insights(AppTest.from_file(APP_PATH, default_timeout=180))
    assert not at.exception
    opt = FoodOptimizer("burger")                    # as the app loaded it
    view = ui_insights.round_view(opt)
    assert view.scored == list(range(complete)) and len(view.tried) == complete + 1
    markdown = [m.value for m in at.tabs[3].markdown]
    headline = ui_insights.headline_text(opt, view)
    assert f"**{headline}**" in markdown, markdown
    assert f"F{complete + 1}" not in headline and "Round 2" not in headline
    gates = [m for m in markdown if "once 5 formulations have results" in m]
    assert gates == [], gates


# ------------------------------------------------------------------ #
#  §3.2 / §9 B1, I8 the verdict and the wide-range line
# ------------------------------------------------------------------ #


# ------------------------------------------------------------------ #
#  §3.3 / §9 I7 the Overall score's own record
# ------------------------------------------------------------------ #


# ------------------------------------------------------------------ #
#  §3.4 / §9 I9 typical miss against the spread between Repeats
# ------------------------------------------------------------------ #

def _with_repeats():
    """Formulation 1 made three times (Firmness 5.0, 5.4, 5.2) and
    Formulation 4 twice (6.0, 6.6): two groups, four pairs."""
    opt = _patty(rows=6)
    for src, dst in ((0, 1), (0, 2), (3, 4)):
        opt.recipe_history[dst] = dict(opt.recipe_history[src])
        opt.X_history[dst] = list(opt.X_history[src])
    for i, v in zip((0, 1, 2, 3, 4), (5.0, 5.4, 5.2, 6.0, 6.6)):
        opt.results_history[i]["Firmness"] = v
    opt._recompute_utilities()
    return opt


def test_a_chain_of_near_matches_is_one_group():
    opt = _patty(rows=3)
    base = dict(opt.recipe_history[0])
    span = 4.0                                  # Pea 4-8
    for k, step in enumerate((0.0, 0.004, 0.008)):
        opt.recipe_history[k] = dict(base, Pea=base["Pea"] + step * span)
        opt.X_history[k] = opt._encode(opt.recipe_history[k])
    groups = insights.repeat_groups(opt, [0, 1, 2])
    assert groups == [[0, 1, 2]]


# ------------------------------------------------------------------ #
#  On screen
# ------------------------------------------------------------------ #

def test_the_tab_shows_the_headline(insights_scored):
    at = _open_insights(AppTest.from_file(APP_PATH, default_timeout=180))
    assert not at.exception
    tab = at.tabs[3]
    opt = FoodOptimizer("burger")
    headline = ui_insights.headline_text(opt, ui_insights.round_view(opt))
    assert f"**{headline}**" in [m.value for m in tab.markdown]
