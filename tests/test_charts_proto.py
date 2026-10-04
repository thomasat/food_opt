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
def test_the_gain_is_bo_views_evaluate_along_the_path(monkeypatch):
    opt, computed = _computed(D.LINE_A, monkeypatch)
    fitted = computed["fitted"]
    name = U.chooses_default(opt)
    data = U.chooses_data(opt, fitted, name)
    assert data.reason is None
    # The open Round's exact rebuild, as the triangle's layers and the map.
    model = U.layer_model(opt, fitted)
    assert model.gen.round == opt.pending_batch_no
    best = insights.start_recipe(opt, opt.best_index())
    path = insights.build_path(opt, name, best)
    want = bo_views.evaluate(opt, model, bo_views.points(opt, path.recipes))
    assert torch.allclose(data.values.value, want.value)
    assert torch.allclose(data.values.mean, want.mean)
    spec = U.chooses_chart(opt, data).to_dict()
    assert len(spec["vconcat"]) == 2                      # prediction above, gain below
    bottom = spec["vconcat"][1]
    gain_line = next(l for l in bottom["layer"] if _mark(l)["type"] == "line")
    rows = [r for r in _rows(spec, gain_line) if r.get("mark") == "curve"]
    assert [r["x"] for r in rows] == pytest.approx(path.amounts)
    assert [r["gain"] for r in rows] == pytest.approx(want.value.tolist())
    # Review plots: two lines, "zoomed" added when under a point.
    assert " ".join(gain_line["encoding"]["y"]["title"]).startswith("Expected gain (points")
    top = spec["vconcat"][0]
    band = next(l for l in top["layer"] if _mark(l)["type"] == "area")
    for r, m, s in zip(rows, want.mean.tolist(), want.sd.tolist()):
        assert r["low"] == pytest.approx(m - insights.Z95 * s)
        assert r["high"] == pytest.approx(m + insights.Z95 * s)
    assert band["encoding"]["y2"]["field"] == "high"


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
def test_the_default_row_is_the_one_the_next_round_changes_most(monkeypatch):
    opt = D.load(D.LINE_A)
    best = insights.start_recipe(opt, opt.best_index())
    picks = [U.anchor_recipe(opt, n) for n in U.suggested_numbers(opt)]
    span = lambda n: [float(b) for b in opt._by_name()[n]['bounds']]
    share = lambda n, p: abs(p[n] - best[n]) / (span(n)[1] - span(n)[0])
    moved = {n: (max(share(n, p) for p in picks), sum(share(n, p) for p in picks))
             for n in insights.path_choices(opt)}
    # Beet juice powder and Mixing time after fat both reach a limit (1.0);
    # Mixing time moves more over the four.
    assert U.chooses_default(opt) == max(moved, key=moved.get) == "Mixing time after fat"


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
    assert no == U.suggested_numbers(opt)[0] and at is not None
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
        assert data.reason == wording.CHOOSES_REASON_NOISE
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
    assert said[2][1] == wording.tip_chooses_withheld(wording.CHOOSES_REASON_NOISE)


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


def test_the_map_s_suggestions_are_orange(monkeypatch):
    opt, computed = _computed(D.LINE_A, monkeypatch)
    values = bo_views.map_values(opt, U.layer_model(opt, computed["fitted"]))
    picks = [{"mean": m, "sd": s, "no": str(n)} for n, m, s in values.picks]
    spec = U.explore_map_chart(values.pool_mean, values.pool_sd, [], picks, None, opt).to_dict()
    # Review plots: the marks' kinds are encoded, with one legend.
    shape = next(l["encoding"]["shape"]["scale"] for l in spec["layer"]
                 if "shape" in l.get("encoding", {}))
    fill = next(l["encoding"]["fill"]["scale"] for l in spec["layer"]
                if "fill" in l.get("encoding", {}))
    k = shape["range"].index("diamond")
    assert shape["domain"][k] == wording.PLOT_SUGGESTION
    assert fill["range"][k] == U.outline_colour()


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
                 wording.CHOOSES_OFF_LINE, wording.CHOOSES_COLD, wording.basis_charts(),
                 wording.TRIED_NOTHING):
        assert len(text.split()) <= 9, text
    for text in (wording.TRIED_TITLE, wording.CHOOSES_TITLE, wording.PAIR_FOLD,
                 wording.INGREDIENT_OR_SETTING):
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
def test_the_line_runs_from_best_so_far_to_suggestion_1(monkeypatch):
    opt, computed, data = _line(D.LINE_A, monkeypatch)
    path = data.path
    best = insights.start_recipe(opt, opt.best_index())
    first = U.suggested_numbers(opt)[0]
    pick = U.anchor_recipe(opt, first)
    assert data.line and data.end_is_pick and data.end_label == wording.CHOOSES_SUGGESTION_ONE
    assert data.pick == (first, 1.0)
    moving = [v['name'] for v in opt.varying_variables() if v.get('type') == 'continuous']
    at = {round(t, 9): r for t, r in zip(path.amounts, path.recipes)}
    for n in moving:
        assert at[0.0][n] == pytest.approx(best[n]) and at[1.0][n] == pytest.approx(pick[n])
    # Every row moves together, in proportion.
    for t, r in zip(path.amounts, path.recipes):
        for n in moving:
            assert r[n] == pytest.approx(best[n] + t * (pick[n] - best[n]))
    # Clipped to the allowed region: every point keeps the limits and the
    # total, the Fill to total row written in again.
    fill = opt.balance_row_name()
    ingredients = [v['name'] for v in opt.variables if v.get('type') == 'continuous'
                   and v.get('category', 'ingredient') == 'ingredient'
                   and (v['name'] == fill or not opt.has_formula(v))]
    total = sum(float(best[n]) for n in ingredients)
    assert total == pytest.approx(float(opt.formulation_total), rel=1e-6)
    for r in path.recipes:
        assert insights._limits_broken(opt, r, moving, best) is None
        assert sum(float(r[n]) for n in ingredients) == pytest.approx(total, rel=1e-6)
    assert path.recipes[-1][fill] == pytest.approx(opt.fill_formulas(path.recipes[-1])[fill])
    # Line A: Best so far's Mixing time is at its Lowest, so the line stops
    # at t = 0 on that side, and runs on to 1.5 on the other.
    assert path.amounts[0] == pytest.approx(0.0, abs=1e-4)
    assert path.low_end.reason == wording.path_past_lowest("Mixing time after fat")
    assert path.amounts[-1] == pytest.approx(insights.LINE_TO)


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_line_extends_both_ways_where_allowed(monkeypatch):
    opt, computed, data = _line(D.LINE_C, monkeypatch)
    assert data.path.amounts[0] == pytest.approx(insights.LINE_FROM)
    assert data.path.amounts[-1] == pytest.approx(insights.LINE_TO)


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_gain_peaks_at_suggestion_1_on_line_a(monkeypatch):
    opt, computed, data = _line(D.LINE_A, monkeypatch)
    model = U.layer_model(opt, computed["fitted"])
    want = bo_views.evaluate(opt, model, bo_views.points(opt, data.path.recipes))
    assert torch.allclose(data.values.value, want.value)
    t, g = data.path.amounts, data.values.value.tolist()
    peak = t[max(range(len(g)), key=g.__getitem__)]
    assert abs(peak - 1.0) <= 0.1
    at = dict(zip([round(x, 9) for x in t], g))
    assert at[1.0] > 100 * max(at[0.0], 1e-6) or at[0.0] < 1e-3


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
    assert wording.CHOOSES_SUGGESTION_ONE in texts and wording.TRIED_BEST in texts


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
    assert "expected improvement (acquisition value)" in wording.GAIN_TOOLTIP_TITLE
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
    assert wording.GAIN_TOOLTIP_TITLE in [t["title"] for t in grid["encoding"]["tooltip"]]
    data = U.chooses_data(opt, fitted, None, mode=U.CHOOSES_LINE)
    spec = U.chooses_chart(opt, data).to_dict()
    gain_line = next(l for l in spec["vconcat"][1]["layer"] if _mark(l)["type"] == "line")
    assert " ".join(gain_line["encoding"]["y"]["title"]).startswith("Expected gain (points")
    assert gain_line["encoding"]["tooltip"][1]["title"] == wording.GAIN_TOOLTIP_TITLE


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
