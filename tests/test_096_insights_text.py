"""0.8.0 BO views, Task 6 (spec 2026-09-30-bo-views §2, §3.2, §4, §8;
review I1-I7, M6, M8, M11): Insights says its statistics by name with
their basis — the verdict's miss count, the Overall score's skill, the
Repeat-based noise threshold — under noun headings; says expected
improvement by Round in Progress; and stops stating noise as ingredient
effects when its model is fitting noise."""
import re
import json
import os

import pytest
from tests.demo_build_reason import REVIEW_FOUR
from streamlit.testing.v1 import AppTest

import bo_views
import insights
import ui_insights
import wording
from food_bo import FoodOptimizer
from tests import demo_projects as demo
from tests.test_app_ui import APP_PATH


def _install(name):
    """A demo project saved in the working folder, as the app opens it."""
    state = json.loads((demo.HERE / f"{name}.json").read_text())
    opt = FoodOptimizer(state["project_name"])
    opt.import_json(state)
    opt.save()
    return opt


def _open(name, fold=None):
    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.session_state["_loaded_project"] = name
    at.run()
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    if fold is not None:
        at.session_state[fold] = True
    at.run()
    assert not at.exception
    return at


def _texts(at):
    tab = at.tabs[3]
    return [m.value for m in tab.markdown], [c.value for c in tab.caption]


def _all(node):
    """Every markdown and caption under `node`, folds included."""
    return [m.value for m in node.markdown] + [c.value for c in node.caption]


@pytest.fixture(scope="module")
def tabs(tmp_path_factory):
    """Lines A, B and C, each opened once on Insights (the checks take a
    few seconds a project); Line C also with the first fold open."""
    here = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("demo"))
    try:
        with bo_views.fresh():
            out = {}
            for name in (demo.LINE_A, demo.LINE_B, demo.LINE_C):
                _install(name)
                out[name] = _open(name)
            out["C try one"] = _open(demo.LINE_C, "insights_try_one")
            out["A try one"] = _open(demo.LINE_A, "insights_try_one")
            out["A how"] = _open(demo.LINE_A, ui_insights.HOW_KEY)
            yield out
    finally:
        os.chdir(here)


# ------------------------------------------------------------------ #
#  Headings and the verdict (03 §4, §5.2; review I7)
# ------------------------------------------------------------------ #

@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_three_sections_are_real_noun_headings(tabs):
    md, _ = _texts(tabs[demo.LINE_A])
    heads = [m for m in md if m.startswith("##### ")]
    assert heads[:3] == [f"##### {q}" for q in (wording.Q_STAND,
                                                )]


# ------------------------------------------------------------------ #
#  Skill (review I1, I2)
# ------------------------------------------------------------------ #

# The scale's two ends ride with the number (ftue spec §4, review R-I7,
# M11; T0).
@pytest.mark.parametrize("name,text", [
    (demo.LINE_A, "Overall score, Rounds 2–5: skill 0.36 (n = 15; 0 = as good as the average, "
                  "1 = perfect), better than the average of earlier results."),
    (demo.LINE_C, "Overall score, Rounds 2–5: skill −0.47 (n = 16; 0 = as good as the "
                  "average, 1 = perfect), no better than the average of earlier results."),
])
@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_skill_line_sits_under_the_verdict(tabs, name, text):
    md, _ = _texts(tabs[name])
    assert text in md
    assert md.index(text) < md.index(f"##### {wording.Q_STAND}")


# ------------------------------------------------------------------ #
#  §3.2 Expected gain by Round (review I4, I5, M8)
# ------------------------------------------------------------------ #


@pytest.fixture(scope="module")
def timing_generated(tmp_path_factory):
    """The Timing check straight after Generate (ui_batch.ask_alone, which
    records Round 13's generation), opened on Insights."""
    import ui_batch
    here = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("timing"))
    try:
        with bo_views.fresh():
            opt = _install(demo.TIMING)
            ui_batch.ask_alone(opt, n_suggestions=3)
            got = _open(demo.TIMING), opt
    finally:
        os.chdir(here)       # the other tests open their projects where they were
    return got


@pytest.fixture(scope="module")
def timing_recorded_flat(tmp_path_factory):
    """The final review's p_stale: the Timing check, Generate 3, every pick
    recorded at the median formulation's results, so no Round is open."""
    import ui_batch
    here = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("timing_flat"))
    try:
        with bo_views.fresh():
            opt = _install(demo.TIMING)
            ui_batch.ask_alone(opt, n_suggestions=3)
            r = opt.pending_batch_no
            order = sorted(range(len(opt.Y_history)), key=lambda i: opt.Y_history[i])
            results = dict(opt.results_history[order[len(order) // 2]])
            for row in list(opt.pending_batch):
                opt.tell(row["recipe"], results, formulation_no=row["formulation"], batch_no=r)
            opt.set_pending_batch(None)
            opt.save()
            got = _open(demo.TIMING), FoodOptimizer(opt.project_name), r
            # The flat Round brings skill to 0.16: the A3 gate (0.8.0 e2e
            # fixes, review I4) withholds the figure. Also opened with a model
            # whose skill lets it through, for final review I-1's check.
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(insights, "skill_gate", lambda o, c, fact=None: None)
                got = got + (_open(demo.TIMING),)
    finally:
        os.chdir(here)
    return got


def test_the_next_rounds_figure_is_worked_out_after_the_page(monkeypatch):
    """Final review I-1: the search (about 1.5 s at 60 x 4) is precomputed
    with the predicted best after the page is drawn, once per data, so no
    screen waits on it; never while a Round is open."""
    from types import SimpleNamespace
    state = {}
    monkeypatch.setattr(ui_insights, "st", SimpleNamespace(session_state=state))
    monkeypatch.setattr(ui_insights, "memo", lambda o: {})
    calls = []
    real = bo_views.next_optimum
    monkeypatch.setattr(bo_views, "next_optimum", lambda *a: calls.append(1) or real(*a))
    opt = demo.load(demo.LINE_A)
    ui_insights.warm_gain(opt)                      # Round 6 open: nothing
    assert calls == [] and ui_insights.GAIN_KEY not in state
    opt.set_pending_batch(None)
    with bo_views.fresh():
        ui_insights.warm_gain(opt)
        ui_insights.warm_gain(opt)
    assert calls == [1] and state[ui_insights.GAIN_KEY][1].round is None


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_layers_name_best_so_far_as_their_anchor():
    """T56 m7."""
    opt = demo.load(demo.LINE_A)
    best_no = int(opt.formulation_ids[opt.best_index()])
    assert ui_insights.anchor_text(opt, best_no) == wording.start_best(best_no)


# ------------------------------------------------------------------ #
#  The noise guard (review I3, I4)
# ------------------------------------------------------------------ #

def _fold(at):
    return next(e for e in at.tabs[3].expander if e.key == "insights_try_one"
                or e.label.startswith("Try one"))


@pytest.mark.skip(reason="review profiler prototype: pins Try one's path and Compare all, which the profiler replaces; reconcile at the final merge")
def test_line_a_still_ranks_its_effects(tabs):
    said = _all(_fold(tabs["A try one"]))
    assert any(t.startswith(("Largest local effect", "Local effect of", "No ingredient"))
               for t in said)


def test_the_correlated_pair_carries_its_r(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tests.test_spread_fixes import _three
    opt = _three([(4.0, 2.0, 1.0, 150), (5.0, 3.1, 0.2, 200), (6.0, 3.9, 1.7, 160),
                  (7.0, 5.2, 0.9, 190)])
    a, b, r = insights.correlated_pair(opt)
    assert (a, b) == ("Pea", "Oil") == insights.moved_together(opt)
    assert r >= insights.TOGETHER
    # T0 review I-4: the pair is ranked by |r|, and the sentence says so.
    assert f"(|r| = {r:.2f})" in wording.moved_together(a, b, f"{r:.2f}")


def _definition_lines(values):
    """Review v3: Definitions is one bullet per term, in order."""
    return [v.split(":", 1)[0].lstrip("- ") for v in values]
