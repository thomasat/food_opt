"""Charts proto (2026-10-01): the Explored and unexplored fold's four charts
— What you've tried (parallel coordinates), How the app chooses (the
classic figure along one row through Best so far), the triangle, the map.
Data and spec checks; the fold's order and privacy are in
test_107_w3_explored."""
import ast
import re
import inspect
import pathlib

import pytest
from tests.demo_build_reason import REVIEW_FOUR
import torch

import bo_views
import insights
import insights_jobs
import ui_insights as U
import wording
from tests import demo_projects as D
from tests.test_104_speed_robust_charts import PAGES, _contrast_on
from tests.test_107_w3_explored import _all_marks, _mark, _rows, _with_a_choice
from tests.demo_build_reason import REMOVED_CHART


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(insights_jobs, "_JOBS", {})
    monkeypatch.setattr(insights_jobs, "_INFLIGHT", {})
    monkeypatch.setattr(U.st, "session_state", {})
    with bo_views.fresh():
        yield


# Review critique (fix 5): "Show as lines" and its parallel coordinates were cut; their
# tests went with them.


# ------------------------------------------------------------------ #
#  How the app chooses
# ------------------------------------------------------------------ #

def _computed(name, monkeypatch):
    opt = D.load(name)
    computed = U._computed(opt, quiet=True)
    monkeypatch.setattr(U, "checks_for", lambda o, stage: computed["checks"])
    return opt, computed


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_with_no_round_open_the_next_model(monkeypatch):
    opt, computed = _computed(D.TIMING, monkeypatch)
    assert not opt.pending_batch
    fitted = computed["fitted"]
    data = U.chooses_data(opt, fitted, insights.path_choices(opt)[0])
    model = bo_views.next_model(opt)
    want = bo_views.evaluate(opt, model, bo_views.points(opt, data.path.recipes))
    assert torch.allclose(data.values.value, want.value)
    assert data.pick is None


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_markers_and_tested_dots(monkeypatch):
    opt, computed = _computed(D.LINE_C, monkeypatch)
    name = "Methylcellulose"
    data = U.chooses_data(opt, computed["fitted"], name)
    best = insights.start_recipe(opt, opt.best_index())
    assert data.path.start_amount == pytest.approx(best[name])
    # The tested-near rule: within CLOSE_TO_THE_BEST of a point of the path.
    for i, amount, score in data.tested:
        t = insights.start_recipe(opt, i)
        assert min(insights.distance(opt, r, t) for r in data.path.recipes) <= 0.15 + 1e-9
        assert score == pytest.approx(float(opt.Y_history[i]))
    assert opt.best_index() in [i for i, _, _ in data.tested]
    # Line C's first suggestion lies near this path: an orange rule there.
    no, at = data.pick
    spec = U.chooses_chart(opt, data).to_dict()
    rules = [m for m in _all_marks(spec) if _mark(m)["type"] == "rule"
             and _mark(m).get("color") in (U.outline_colour(), U.mark("bar_latest")[0])]
    xs = {_mark(m)["color"]: _rows(spec, m)[0]["x"] for m in rules}
    assert xs[U.mark("bar_latest")[0]] == pytest.approx(best[name])
    assert xs[U.outline_colour()] == pytest.approx(at)


@pytest.mark.skip(reason=REMOVED_CHART)
def test_cold_says_so(monkeypatch):
    opt = D.load(D.SNACK)
    said = []
    monkeypatch.setattr(U.st, "caption", lambda text, **k: said.append(text))
    monkeypatch.setattr(U.st, "markdown", lambda *a, **k: None)
    U._chooses(opt, None)
    assert said == [wording.CHOOSES_COLD]


# ------------------------------------------------------------------ #
#  The triangle, the map, Trade-offs, privacy, wording
# ------------------------------------------------------------------ #


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_trade_offs_holds_the_two_measurement_chart_alone():
    opt = D.load(D.LINE_A)
    assert U.trade_offs_label(opt, {"fitted": {}}) == wording.PAIR_FOLD
    src = inspect.getsource(U._more_charts)
    assert "_mixture" not in src and "_pair" in src
    assert "_triangle" in inspect.getsource(U._explored_body)


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_every_new_chart_goes_through_show_chart():
    for fn in (U._tried, U._hero_body, U._triangle, U._mixture):   # demo build: _chooses, _explored_map gone
        src = inspect.getsource(fn)
        assert "altair_chart" not in src, fn.__name__
    tree = ast.parse(pathlib.Path(U.__file__).read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == "altair_chart"]
    assert len(calls) == 1                                   # show_chart's own


def test_the_new_lines_stay_short():
    """The owner's concision rule: captions about 8 words, titles 5."""
    for text in (wording.tried_caption(), wording.CHOOSES_CAPTION, wording.CHOOSES_WITHHELD,
                 wording.CHOOSES_OFF_LINE, wording.CHOOSES_COLD, wording.basis_charts()):
        assert len(text.split()) <= 9, text
    for text in (wording.TRIED_TITLE, wording.CHOOSES_TITLE, wording.PAIR_FOLD):
        assert len(text.split()) <= 5, text
    assert wording.CHOOSES_CAPTION == "Suggestions go where expected gain peaks."


# ------------------------------------------------------------------ #
#  Follow-up: the straight line Best so far → Suggestion 1, one name for
#  the gain, the band's width said
# ------------------------------------------------------------------ #

def _line(name, monkeypatch):
    opt, computed = _computed(name, monkeypatch)
    data = U.chooses_data(opt, computed["fitted"], None, mode=U.CHOOSES_LINE)
    return opt, computed, data


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_line_extends_both_ways_where_allowed(monkeypatch):
    opt, computed, data = _line(D.LINE_C, monkeypatch)
    assert data.path.amounts[0] == pytest.approx(insights.LINE_FROM)
    assert data.path.amounts[-1] == pytest.approx(insights.LINE_TO)


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_line_chart_reads_best_so_far_to_suggestion_1(monkeypatch):
    opt, computed, data = _line(D.LINE_A, monkeypatch)
    spec = U.chooses_chart(opt, data).to_dict()
    top, bottom = spec["vconcat"]
    line = next(l for l in top["layer"] if _mark(l)["type"] == "line")
    assert line["encoding"]["x"]["title"] == "From Best so far → Suggestion 1"
    assert line["encoding"]["x"]["axis"]["values"] == [0, 1]
    assert [t["title"] for t in line["encoding"]["tooltip"]][0] == wording.CHOOSES_AMOUNTS_HERE
    rows = [r for r in _rows(spec, line) if r.get("mark") == "curve"]
    assert "Textured pea protein" in rows[0]["amount"] and "Water" in rows[0]["amount"]
    rules = {_mark(m)["color"]: _rows(spec, m)[0]["x"] for m in _all_marks(spec)
             if _mark(m)["type"] == "rule" and "x" in _rows(spec, m)[0]
             and _mark(m).get("color") in (U.outline_colour(), U.mark("bar_latest")[0])}
    assert rules[U.mark("bar_latest")[0]] == 0.0 and rules[U.outline_colour()] == 1.0
    texts = [r["label"] for m in _all_marks(spec) if _mark(m)["type"] == "text"
             for r in _rows(spec, m) if "label" in r]


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_with_no_round_open_the_line_ends_at_the_predicted_best(monkeypatch):
    opt, computed, data = _line(D.TIMING, monkeypatch)
    assert not opt.pending_batch
    assert data.end_label == wording.CHOOSES_PREDICTED_BEST and not data.end_is_pick
    best = insights.predicted_best(opt, computed["fitted"], computed["checks"])
    end = insights.at_total(opt, best.recipe)
    at1 = next(r for t, r in zip(data.path.amounts, data.path.recipes) if abs(t - 1) < 1e-9)
    for v in opt.varying_variables():
        if v.get('type') == 'continuous':
            assert at1[v['name']] == pytest.approx(end[v['name']])
    spec = U.chooses_chart(opt, data).to_dict()
    assert not [m for m in _all_marks(spec) if _mark(m).get("color") == U.outline_colour()]
    line = next(l for l in spec["vconcat"][0]["layer"] if _mark(l)["type"] == "line")
    assert line["encoding"]["x"]["title"] == "From Best so far → Predicted best"


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_band_says_it_is_narrower_than_try_one(monkeypatch):
    assert len(wording.CHOOSES_BAND_LEGEND.split()) <= 8
    assert "narrower" in wording.CHOOSES_BAND_LEGEND and "Try one" in wording.CHOOSES_BAND_LEGEND
    assert "narrower" in wording.TIP_CHOOSES_BAND and "Repeat noise" in wording.TIP_CHOOSES_BAND
    opt, computed, data = _line(D.LINE_A, monkeypatch)
    spec = U.chooses_chart(opt, data).to_dict()
    band = next(l for l in spec["vconcat"][0]["layer"] if _mark(l)["type"] == "area")
    assert band["encoding"]["fill"]["field"] == "band"
    rows = _rows(spec, band)
    # Review plots: the legend says what the band is; its tooltip, that it
    # is narrower than Try one's.
    assert {r["band"] for r in rows} == {wording.PLOT_BAND}
    assert {r["band_tip"] for r in rows} == {wording.TIP_CHOOSES_BAND}
    assert band["encoding"]["tooltip"][0]["field"] == "band_tip"
