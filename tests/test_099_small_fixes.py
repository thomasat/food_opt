"""0.8.0 BO views, small fixes after the answer-key refresh (answer-key
refresh, "BO-views refresh", items 2, 6, 7 and 8): the Why map's dashed
line is Best so far's predicted Repeat, the Insights outline's reference,
for every acquisition; "regression to the mean" only in Method and
details; the Why fold never shows roles it withdraws once the checks are
in; and the next Round's expected gain never holds up the first
answers right after launch."""
import concurrent.futures as cf
import json
import os

import pytest
from tests.demo_build_reason import REVIEW_FOUR
from streamlit.testing.v1 import AppTest

import bo_views
import insights
import insights_jobs
import ui_insights
import wording
# Review four, ruling 2: the Why fold's "?" says the gain is greyed, not hidden.
from food_bo import FoodOptimizer
from tests import demo_projects as demo
from tests.test_app_ui import APP_PATH
from tests.test_insights import _burger

README = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "README.md")


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(insights_jobs, "_JOBS", {})
    monkeypatch.setattr(insights_jobs, "_INFLIGHT", {})
    with bo_views.fresh():
        yield


def _app(name, tab=wording.TAB_BATCH, why_open=True):
    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.session_state["_loaded_project"] = name
    at.run()
    at.session_state["main_tab"] = tab
    if why_open:
        at.session_state[ui_insights.WHY_FOLD_KEY] = True
    at.run()
    assert not at.exception
    return at


def _rerun(at, tab=wording.TAB_BATCH):
    at.session_state["main_tab"] = tab
    at.run()
    assert not at.exception


def _install(name):
    """A demo project saved in the working folder, as the app opens it."""
    state = json.loads((demo.HERE / f"{name}.json").read_text())
    opt = FoodOptimizer(state["project_name"])
    opt.import_json(state)
    opt.save()
    return opt


# ------------------------------------------------------------------ #
#  1. The Why map's reference line (refresh item 6; spec-review I6/I7)
# ------------------------------------------------------------------ #

def test_the_map_caption_names_the_line_only_when_it_is_drawn():
    with_line = wording.why_chart_caption("3,000")
    assert with_line.endswith("; dashed line, Best so far (predicted Repeat).")
    without = wording.why_chart_caption("3,000", line=False)
    assert without.endswith("labelled, this Round's suggestions.")
    for line in (with_line, without):
        assert line.count(". ") == 0 and "grey rings, tested formulations" in line


# ------------------------------------------------------------------ #
#  2. "Regression to the mean" only in Method and details (item 7)
# ------------------------------------------------------------------ #

def test_the_luck_line_says_it_in_plain_words():
    line = wording.best_luck(11, "84", "62", "100", "90.09")
    assert line == ("A Repeat of Formulation 11 is predicted at 84 (95 % interval 62–100), "
                    "below its measured 90.09. Best so far's measured score is partly chance, "
                    "so its Repeat is predicted lower.")
    assert "regression" not in line.lower()


def test_the_new_lines_pass_the_guards():
    from tests.test_090_wording import _STYLE
    from tests.test_food_bo import _BANNED
    lines = [wording.best_luck(11, "84", "62", "100", "90.09"),
             wording.WHY_REFERENCE, wording.why_chart_caption("3,000"), wording.WHY_CHECKING]
    assert [t for t in lines if any(p.search(t) for p in _BANNED)] == []
    assert [t for t in lines if any(p.search(t) for p in _STYLE)] == []
    assert [t for t in lines if t.count("—") > 1] == []


def test_method_and_details_says_the_term():
    opt = _burger(rounds=5, noise=0.6)
    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.session_state["_loaded_project"] = opt.project_name
    at.run()
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.session_state[ui_insights.HOW_KEY] = True
    at.run()
    assert not at.exception
    how = next(e for e in at.tabs[3].expander if e.label == wording.FOLD_METHOD)
    # One-glance Review (owner, 2026-10-14): the fold is one line per term
    # and the luck line's statistics term is off the screen; the
    # caveat itself lives in the README's "How the Review works", Limits.
    # Review v3: the five terms, worked out for this project; no README line.
    assert [m.value for m in how.markdown] == [f"- {line}" for line in
                                               ui_insights.definitions(opt)]
    section = open(README, encoding="utf-8").read().split("## How the Review works", 1)[1]
    assert "Best so far is one noisy maximum and tends to read high." in \
        section.split("\n## ", 1)[0]


# ------------------------------------------------------------------ #
#  3. The Why fold never shows roles it later withdraws (item 8)
# ------------------------------------------------------------------ #

class _Stuck:
    """A pool whose workers never answer: the screens wait for them."""

    broken = False

    def submit(self, priority, request):
        return cf.Future()

    def idle(self):
        return False


def _workers(monkeypatch, pool):
    monkeypatch.setattr(insights_jobs, "workers_on", lambda: pool is not None)
    monkeypatch.setattr(insights_jobs, "pool", lambda: pool)


def _seed(at, opt, stage):
    """This session's memo holding exactly what `stage` needs."""
    memo = {}
    specs = insights_jobs.stage_specs(opt, insights_jobs.plan(opt), stage)
    insights_jobs.compute_inline(opt, memo, specs)
    at.session_state[ui_insights.MEMO_KEY] = (opt.project_name, memo)


# ------------------------------------------------------------------ #
#  4. The next Round's figure never holds up the first answers (item 2)
# ------------------------------------------------------------------ #

@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_verdict_is_drawn_before_the_figure_is_worked_out(monkeypatch):
    """Insights draws the Prediction check and Best so far's lines before
    the next Round's expected gain is calculated (about 1.5 s at
    60 x 4), so the first answers never wait for it."""
    opt = _burger(rounds=5, noise=0.6)
    # A model with skill, so the figure is worked out at all (A3 gate, 0.8.0
    # e2e fixes, review I4), and a record that does not withhold it (ftue
    # fix A's folds read this project as More misses than expected).
    monkeypatch.setattr(insights, "skill_gate", lambda opt, checks, fact=None: None)
    monkeypatch.setattr(ui_insights, "withheld", lambda opt, score, checks: None)
    order = []
    real_trust, real_gain = ui_insights._trust, ui_insights.latest_gain
    monkeypatch.setattr(ui_insights, "_trust",
                        lambda *a, **k: order.append("verdict") or real_trust(*a, **k))
    monkeypatch.setattr(ui_insights, "latest_gain",
                        lambda *a, **k: order.append("gain") or real_gain(*a, **k))
    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.session_state["_loaded_project"] = opt.project_name
    at.run()
    _rerun(at, wording.TAB_INSIGHTS)
    assert "verdict" in order and "gain" in order
    assert order.index("verdict") < order.index("gain")


def test_while_the_workers_fit_the_figure_is_worked_out_after_the_page(monkeypatch):
    """Right after launch the workers are still fitting the Overall score's
    models: the app's process would only wait, so keep_up works out the
    next Round's figure then, after the page is drawn, and the answers
    find it ready when the models land."""
    _workers(monkeypatch, _Stuck())
    opt = _burger(rounds=5, noise=0.6)
    calls = []
    monkeypatch.setattr(ui_insights, "warm_gain", lambda o: calls.append(o.project_name))
    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.session_state["_loaded_project"] = opt.project_name
    at.run()
    assert not at.exception
    assert opt.project_name in calls


def test_a_search_left_running_by_a_rerun_still_serves_the_next_run(monkeypatch):
    """A click reruns the page while the first run is still in the search
    (Streamlit lets it finish in its own thread, its session-state write
    dropped): the next run takes its answer from the process's cache, and a
    run that only warms the figure never queues behind it."""
    opt = demo.load(demo.TIMING)
    searches = []
    real = bo_views._next_optimum_here
    monkeypatch.setattr(bo_views, "_next_optimum_here",
                        lambda *a: searches.append(1) or real(*a))
    model = bo_views.next_model(opt)
    first = bo_views.next_optimum(opt, model)
    again = bo_views.next_optimum(opt, bo_views.next_model(opt))
    assert searches == [1] and again == first

    from types import SimpleNamespace
    state = {}
    monkeypatch.setattr(ui_insights, "st", SimpleNamespace(session_state=state))
    monkeypatch.setattr(ui_insights, "memo", lambda o: {})
    key = ui_insights._gain_key(opt)
    import threading
    monkeypatch.setitem(ui_insights._GAIN_RUNS, key, threading.Event())   # another run's
    assert ui_insights.latest_gain(opt, wait=False) is None
    assert ui_insights.GAIN_KEY not in state and searches == [1]
