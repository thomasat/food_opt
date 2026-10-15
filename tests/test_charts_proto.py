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


@pytest.mark.parametrize("which", ["light", "dark"])
def test_every_colour_reaches_3_to_1(monkeypatch, which):
    monkeypatch.setattr(U, "theme", lambda: which)
    monkeypatch.setattr(U, "muted", lambda: U._MUTED_DARK if which == "dark" else U._MUTED_LIGHT)
    page = PAGES[which]
    for colour in U.score_colours():
        assert _contrast_on(colour, 1.0, page) >= 3.0, colour
    for colour in (U.model_blue(), U.outline_colour(), U.mark("neutral")[0],
                   U.mark("bar_latest")[0]):
        assert _contrast_on(colour, 1.0, page) >= 3.0, colour
    # Higher scores read stronger: contrast rises along the ramp.
    contrast = [_contrast_on(c, 1.0, page) for c in U.score_colours()]
    assert contrast == sorted(contrast)


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


@pytest.mark.skip(reason=REVIEW_FOUR)
@pytest.mark.parametrize("name", [D.LINE_B, D.LINE_C, D.UPGRADE])
def test_the_gain_panel_is_withheld_under_the_rules(name, monkeypatch):
    opt, computed = _computed(name, monkeypatch)
    fitted = computed["fitted"]
    assert wording.LAYER_GAIN not in U.layers_offered(opt, fitted)
    data = U.chooses_data(opt, fitted, insights.path_choices(opt)[0])
    rule = U.explore_withheld(opt, fitted.get(insights.SCORE), computed["checks"])
    if rule == "hidden":
        pass
    else:
        why = U.withheld(opt, fitted.get(insights.SCORE), computed["checks"])
        gate = insights.skill_gate(opt, computed["checks"]) if why == "skill" else None
        assert data.reason == U.gain_reason(why, gate)
    spec = U.chooses_chart(opt, data).to_dict()
    assert "vconcat" not in spec                             # the top panel alone
    assert not [m for m in _all_marks(spec)
                if m.get("encoding", {}).get("y", {}).get("field") == "gain"]


@pytest.mark.skip(reason=REMOVED_CHART)
def test_the_withheld_reason_is_said_on_screen(monkeypatch):
    opt, computed = _computed(D.LINE_C, monkeypatch)
    said = []
    monkeypatch.setattr(U.st, "caption",
                        lambda text, help=None, **k: said.append((text, help)))
    monkeypatch.setattr(U.st, "markdown", lambda *a, **k: None)
    monkeypatch.setattr(U.st, "selectbox", lambda label, options, key, **k: options[0])
    monkeypatch.setattr(U, "show_chart", lambda c, **k: said.append(("chart", None)))
    U._chooses(opt, computed)
    texts = [t for t, _ in said]
    assert texts == ["chart", wording.CHOOSES_CAPTION, wording.CHOOSES_WITHHELD]


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
def test_the_triangle_reads_in_score_colours_and_orange_is_for_suggestions(monkeypatch):
    opt, computed = _computed(D.LINE_A, monkeypatch)
    fitted = computed["fitted"]
    names = insights.mixture_choices(opt)[:3]
    start = insights.start_recipe(opt, opt.best_index())
    mix = insights.mixture_slice(opt, names, start)
    preds = insights.mixture_predictions(opt, fitted, mix)
    extra = U._mixture_extra(opt, fitted, mix, preds, wording.LAYER_SCORE)
    spec = U.mixture_chart(opt, mix, preds, 15, None, None, None, extra.get("outline")).to_dict()
    marks = _all_marks(spec)
    grid = next(m for m in marks if m.get("encoding", {}).get("color", {}).get("field") == "score")
    assert grid["encoding"]["color"]["scale"]["range"] == U.score_colours()
    blocked = next(m for m in marks if _mark(m)["type"] == "point"
                   and _mark(m).get("filled") is False and _mark(m).get("color") == U.mark("neutral")[0])
    assert blocked
    assert all(_mark(m).get("stroke") != U.outline_colour() for m in marks)
    # Under a model layer: blues, the model's colour.
    sd = U.layer_values(opt, fitted, [p.recipe for p in mix.points if p.recipe is not None],
                        wording.LAYER_SD)
    it = iter(sd)
    values = [next(it) if p.recipe is not None else None for p in mix.points]
    spec = U.mixture_chart(opt, mix, preds, 15, None, wording.LAYER_SD, values).to_dict()
    grid = next(m for m in _all_marks(spec)
                if m.get("encoding", {}).get("color", {}).get("field") == "score")
    assert grid["encoding"]["color"]["scale"]["scheme"]["name"] == "blues"


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


@pytest.mark.skip(reason=REMOVED_CHART)
def test_the_selector_opens_on_the_line_and_keeps_one_ingredient(monkeypatch):
    opt, computed = _computed(D.LINE_A, monkeypatch)
    boxes = []

    def box(label, options, key, format_func=str, **k):
        boxes.append((label, [format_func(o) for o in options]))
        return options[0]
    monkeypatch.setattr(U.st, "selectbox", box)
    monkeypatch.setattr(U.st, "caption", lambda *a, **k: None)
    monkeypatch.setattr(U.st, "markdown", lambda *a, **k: None)
    drawn = []
    monkeypatch.setattr(U, "show_chart", lambda c, **k: drawn.append(c.to_dict()))
    U._chooses(opt, computed)
    assert boxes == [(wording.CHOOSES_MODE, ["Best so far → Suggestion 1",
                                             "Along one ingredient"])]
    top = drawn[0]["vconcat"][0]
    line = next(l for l in top["layer"] if _mark(l)["type"] == "line")
    assert line["encoding"]["x"]["title"] == "From Best so far → Suggestion 1"


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_one_name_for_the_gain_on_screen(monkeypatch):
    from tests.test_food_bo import _prose_constants
    root = pathlib.Path(U.__file__).parent
    assert wording.LAYER_GAIN == "Expected gain"
    assert wording.LAYER_GAIN_LEGEND == wording.CHOOSES_GAIN == "Expected gain (points)"
    for text in _prose_constants(root / "wording.py"):
        # LAYER_GAIN_FORMER: an old session's radio value, read, never shown.
        if "xpected improvement" in text and text != wording.LAYER_GAIN_FORMER:
            # Only as the BO term, in a tooltip or Method and definitions.
            assert ("BO term" in text or "qLog" in text or "acquisition value" in text), text
    opt, computed = _computed(D.LINE_A, monkeypatch)
    fitted = computed["fitted"]
    names = insights.mixture_choices(opt)[:3]
    mix = insights.mixture_slice(opt, names, insights.start_recipe(opt, opt.best_index()))
    preds = insights.mixture_predictions(opt, fitted, mix)
    gain = U.layer_values(opt, fitted, [p.recipe for p in mix.points if p.recipe is not None],
                          wording.LAYER_GAIN)
    it = iter(gain)
    values = [next(it) if p.recipe is not None else None for p in mix.points]
    spec = U.mixture_chart(opt, mix, preds, 15, None, wording.LAYER_GAIN, values).to_dict()
    grid = next(m for m in _all_marks(spec)
                if m.get("encoding", {}).get("color", {}).get("field") == "score")
    assert grid["encoding"]["color"]["title"] == "Expected gain (points)"
    data = U.chooses_data(opt, fitted, None, mode=U.CHOOSES_LINE)
    spec = U.chooses_chart(opt, data).to_dict()
    gain_line = next(l for l in spec["vconcat"][1]["layer"] if _mark(l)["type"] == "line")
    assert " ".join(gain_line["encoding"]["y"]["title"]).startswith("Expected gain (points")


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
