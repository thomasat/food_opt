"""0.8.0 BO views, small fixes after the answer-key refresh (answer-key
refresh, "BO-views refresh", items 2, 6, 7 and 8): the Why map's dashed
line is Best so far's predicted Repeat, the Insights outline's reference,
for every acquisition; "regression to the mean" only in Method and
details; the Why fold never shows roles it withdraws once the checks are
in; and the next Round's expected improvement never holds up the first
answers right after launch."""
import concurrent.futures as cf
import json
import re

import pytest
from streamlit.testing.v1 import AppTest

import bo_views
import insights
import insights_jobs
import ui_batch
import ui_insights
import wording
from food_bo import FoodOptimizer
from tests import demo_projects as demo
from tests.test_app_ui import APP_PATH
from tests.test_insights import _burger


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
        at.session_state[ui_batch.WHY_KEY] = True
    at.run()
    assert not at.exception
    return at


def _rerun(at, tab=wording.TAB_BATCH):
    at.session_state["main_tab"] = tab
    at.run()
    assert not at.exception


def _fold(at):
    return next(e for e in at.tabs[1].expander if e.label == wording.WHY_THESE)


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

def _map_spec(monkeypatch):
    """A list that receives the map's Vega-Lite spec each time it is drawn."""
    got = []
    real = ui_batch.why_chart

    def why_chart(*a, **k):
        chart = real(*a, **k)
        if chart is not None:
            got.append(chart.to_dict())
        return chart
    monkeypatch.setattr(ui_batch, "why_chart", why_chart)
    return got


def _layer_values(spec, mark):
    """The data rows of every layer drawn with `mark`."""
    out = []
    for layer in spec["layer"]:
        m = layer["mark"]["type"] if isinstance(layer["mark"], dict) else layer["mark"]
        if m == mark:
            out.append(spec["datasets"][layer["data"]["name"]])
    return out


def _predicted_repeat(opt):
    """Best so far's predicted Repeat as Insights' outline takes it: the
    Overall-score display model's mean at Best so far's amounts."""
    with insights.memo_in_use({}):
        score = insights.fit(opt, insights.SCORE, list(range(len(opt.Y_history))))
    [p] = insights.predict(opt, score, [opt._encode(insights.start_recipe(opt, opt.best_index()))])
    return p.mean


@pytest.mark.parametrize("kind", ["qlognei", "qlogei", "qucb"])
def test_the_map_line_is_best_so_far_s_predicted_repeat(kind, monkeypatch):
    """Like with like on a predicted axis (binding speed-spec I7), the
    reference the Insights outline uses, for every acquisition: qUCB too,
    and qLogNEI no longer at the model's best tested mean."""
    opt = _burger(rounds=4)
    if kind != "qlognei":
        opt.set_bo_config({"acquisition": kind})
    ui_batch.ask_alone(opt, n_suggestions=3)
    specs = _map_spec(monkeypatch)
    at = _app(opt.project_name)
    spec = specs[-1]
    [rule] = _layer_values(spec, "rule")
    assert rule[0]["ref"] == pytest.approx(_predicted_repeat(opt), abs=1e-6)
    labels = [row.get("label") for rows in _layer_values(spec, "text") for row in rows]
    assert wording.WHY_REFERENCE in labels
    assert wording.WHY_REFERENCE == "Best so far (predicted Repeat)"
    captions = [c.value for c in _fold(at).caption]
    assert wording.why_chart_caption(f"{bo_views.POOL_SIZE:,}") in captions


def test_the_map_and_the_outline_read_one_reference(monkeypatch):
    """The map draws what ui_insights.repeat_of_best gives, the function
    the Predicted above Best so far outline compares with."""
    opt = _burger(rounds=4)
    ui_batch.ask_alone(opt, n_suggestions=3)
    monkeypatch.setattr(ui_insights, "repeat_of_best", lambda o, fitted: 42.5)
    specs = _map_spec(monkeypatch)
    _app(opt.project_name)
    [rule] = _layer_values(specs[-1], "rule")
    assert rule[0]["ref"] == 42.5
    import inspect
    assert "repeat_of_best(opt, fitted)" in inspect.getsource(ui_insights._mixture_extra)


def test_the_map_caption_names_the_line_only_when_it_is_drawn():
    with_line = wording.why_chart_caption("3,000")
    assert with_line.endswith("; dashed line, Best so far (predicted Repeat).")
    without = wording.why_chart_caption("3,000", line=False)
    assert without.endswith("labelled, this Round's suggestions.")
    for line in (with_line, without):
        assert line.count(". ") == 0 and "grey rings, tested formulations" in line
