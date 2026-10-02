"""Charts proto (2026-10-01): the Explored and unexplored fold's four charts
— What you've tried (parallel coordinates), How the app chooses (the
classic figure along one row through Best so far), the triangle, the map.
Data and spec checks; the fold's order and privacy are in
test_107_w3_explored."""
import ast
import inspect
import pathlib

import pytest
import torch

import bo_views
import insights
import insights_jobs
import ui_insights as U
import wording
from tests import demo_projects as D
from tests.test_104_speed_robust_charts import PAGES, _contrast_on
from tests.test_107_w3_explored import _all_marks, _mark, _rows, _with_a_choice


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(insights_jobs, "_JOBS", {})
    monkeypatch.setattr(insights_jobs, "_INFLIGHT", {})
    monkeypatch.setattr(U.st, "session_state", {})
    with bo_views.fresh():
        yield


def _tried(opt):
    return U.tried_chart(opt, insights.tried_axes(opt), insights.tried_lines(opt)).to_dict()


def _line_layers(spec):
    return [m for m in _all_marks(spec) if _mark(m)["type"] == "line"]


# ------------------------------------------------------------------ #
#  What you've tried
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("name", [D.LINE_A, D.LINE_C, D.SNACK, D.PROCESS_ONLY])
def test_the_lines_are_the_formulations(name):
    opt = D.load(name)
    lines = insights.tried_lines(opt)
    recorded = [int(n) for n in opt.formulation_ids]
    skipped = [int(r['formulation']) for r in opt.skipped if r.get('recipe')]
    pending = [n for n in U.suggested_numbers(opt) if n not in recorded]
    assert sorted(l.formulation for l in lines) == sorted(recorded + skipped + pending)
    df, _, labels = U.tried_frame(opt, insights.tried_axes(opt), lines)
    assert sorted(set(df["no"])) == sorted(l.formulation for l in lines)
    score = U._say(opt, wording.OVERALL_SCORE_COLUMN)
    for line in lines:
        mine = df[df["no"] == line.formulation]
        # Every axis is crossed; the score axis only with a score.
        assert list(mine["k"]) == list(range(len(labels) - (line.score is None)))
        if line.score is not None:
            [t] = mine[mine["axis"] == score]["t"]
            assert t == pytest.approx(line.score / opt.utility_ceiling())


@pytest.mark.parametrize("name", [D.LINE_A, D.SNACK, D.PROCESS_ONLY])
def test_each_axis_spans_lowest_to_highest(name):
    opt = D.load(name)
    axes = insights.tried_axes(opt)
    # One per varying ingredient and process setting; the Fill to total row
    # follows the others and is left out.
    assert [a.name for a in axes] == [n for n in (s.name for s in insights.tested_strips(opt))
                                      if n != opt.balance_row_name()]
    for a in axes:
        v = opt._by_name()[a.name]
        assert (a.low, a.high) == tuple(float(b) for b in v['bounds'])
        assert insights.axis_share(a, a.low) == 0.0 and insights.axis_share(a, a.high) == 1.0
    lines = insights.tried_lines(opt)
    df, ends, labels = U.tried_frame(opt, axes, lines)
    for k, a in enumerate(axes):
        here = ends[ends["axis"] == labels[k]]
        low = here[here["where"] == "low"]["text"].item()
        high = here[here["where"] == "high"]["text"].item()
        assert low.startswith(U._num(a.low).split(".")[0]) and high.startswith(
            U._num(a.high).split(".")[0])
        if a.unit:
            assert low.endswith(a.unit) and high.endswith(a.unit)
            assert f"({a.unit})" in labels[k].replace("\n", " ")
        for line in lines:
            [t] = df[(df["no"] == line.formulation) & (df["k"] == k)]["t"]
            assert t == pytest.approx((line.values[a.name] - a.low) / (a.high - a.low))
    score = ends[ends["axis"] == labels[-1]]
    assert list(score["text"]) == ["0", "100"]


def test_a_choice_gets_its_options_spaced_evenly():
    opt = _with_a_choice()
    axes = insights.tried_axes(opt)
    package = next(a for a in axes if a.name == "Package")
    assert package.options == ["Tray", "Pouch", "Box"]
    assert [insights.axis_share(package, o) for o in package.options] == [0.0, 0.5, 1.0]
    spec = _tried(opt)
    texts = [m for m in _all_marks(spec) if _mark(m)["type"] == "text"]
    options = [r["text"] for m in texts for r in _rows(spec, m) if r.get("where") == "option"]
    assert options == ["Tray", "Pouch", "Box"]


def test_suggestions_are_bold_orange_waiting_grey_dashed_best_thick():
    opt = D.load(D.LINE_A)
    spec = _tried(opt)
    by_kind = {}
    for m in _line_layers(spec):
        kinds = {r["role"] for r in _rows(spec, m)}
        flt = m.get("transform", [{}])[0].get("filter", {})
        want = flt.get("equal") or (flt.get("oneOf") or [None])[0]
        by_kind.setdefault(want, []).append(m)
        assert kinds  # the layer's frame is the one long frame
    rows = _rows(spec, _line_layers(spec)[0])
    kind_of = {r["no"]: r["role"] for r in rows}
    assert {n for n, k in kind_of.items() if k == wording.TRIED_SUGGESTED} == {22, 23, 24, 25}
    assert kind_of[19] == wording.TRIED_WAITING
    best_no = int(opt.formulation_ids[opt.best_index()])
    assert kind_of[best_no] == wording.TRIED_BEST
    # Suggestions carry no score: they stop before the score axis.
    assert all(r["score"] is None for r in rows if r["role"] == wording.TRIED_SUGGESTED)
    stroke = next(m for m in by_kind[wording.TRIED_SUGGESTED])["encoding"]["stroke"]["scale"]
    look = dict(zip(stroke["domain"], stroke["range"]))
    assert look[wording.TRIED_SUGGESTED] == U.outline_colour()
    assert look[wording.TRIED_WAITING] == U.mark("neutral")[0]
    dash = by_kind[wording.TRIED_WAITING][0]["encoding"]["strokeDash"]["scale"]
    assert dict(zip(dash["domain"], dash["range"]))[wording.TRIED_WAITING] == [4, 3]
    width = lambda kind: max(_mark(m)["strokeWidth"] for m in by_kind[kind])
    assert width(wording.TRIED_SUGGESTED) >= 3
    assert width(wording.TRIED_BEST) > width(wording.TRIED_SCORED)
    # Scored lines are coloured by the Overall score on the score colours.
    [plain] = by_kind[wording.TRIED_SCORED]
    assert plain["encoding"]["color"]["field"] == "score"
    assert plain["encoding"]["color"]["scale"]["range"] == U.score_colours()


def test_the_line_tooltip_reads_formulation_round_score_and_amounts():
    opt = D.load(D.LINE_A)
    spec = _tried(opt)
    for m in _line_layers(spec):
        titles = [t["title"] for t in m["encoding"]["tooltip"]]
        assert titles == [wording.FORMULATION_CAP, wording.ROUND_CAP,
                          wording.OVERALL_SCORE_COLUMN, wording.tried_amounts_title()]
    rows = _rows(spec, _line_layers(spec)[0])
    best = next(r for r in rows if r["no"] == 15)
    assert best["round"] == "4" and best["score_text"] == f"{opt.Y_history[14]:.2f}"
    assert "Coconut oil" in best["amounts"] and " g" in best["amounts"]
    assert next(r for r in rows if r["no"] == 19)["score_text"].startswith("waiting for")


def test_many_lines_are_thin_and_semi_transparent():
    opt = D.load(D.TIMING)
    assert len(insights.tried_lines(opt)) > U._MANY_LINES
    plain = next(m for m in _line_layers(_tried(opt)) if "color" in m["encoding"]
                 and m["encoding"]["color"].get("legend") is not None)
    assert _mark(plain)["strokeWidth"] <= 1 and _mark(plain)["opacity"] < 1


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
    spec = _tried(D.load(D.LINE_A))
    for m in _all_marks(spec):
        mk = _mark(m)
        if mk["type"] == "text":
            assert _contrast_on(mk["color"], 1.0, page) >= 4.5


# ------------------------------------------------------------------ #
#  How the app chooses
# ------------------------------------------------------------------ #

def _computed(name, monkeypatch):
    opt = D.load(name)
    computed = U._computed(opt, quiet=True)
    monkeypatch.setattr(U, "checks_for", lambda o, stage: computed["checks"])
    return opt, computed


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
    assert gain_line["encoding"]["y"]["title"] == wording.CHOOSES_GAIN
    top = spec["vconcat"][0]
    band = next(l for l in top["layer"] if _mark(l)["type"] == "area")
    for r, m, s in zip(rows, want.mean.tolist(), want.sd.tolist()):
        assert r["low"] == pytest.approx(m - insights.Z95 * s)
        assert r["high"] == pytest.approx(m + insights.Z95 * s)
    assert band["encoding"]["y2"]["field"] == "high"


def test_with_no_round_open_the_next_model(monkeypatch):
    opt, computed = _computed(D.TIMING, monkeypatch)
    assert not opt.pending_batch
    fitted = computed["fitted"]
    data = U.chooses_data(opt, fitted, insights.path_choices(opt)[0])
    model = bo_views.next_model(opt)
    want = bo_views.evaluate(opt, model, bo_views.points(opt, data.path.recipes))
    assert torch.allclose(data.values.value, want.value)
    assert data.pick is None


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
             and _mark(m).get("color") in (U.outline_colour(), U.mark("reference")[0])]
    xs = {_mark(m)["color"]: _rows(spec, m)[0]["x"] for m in rules}
    assert xs[U.mark("reference")[0]] == pytest.approx(best[name])
    assert xs[U.outline_colour()] == pytest.approx(at)


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
    diamond = next(l for l in spec["layer"] if _mark(l).get("shape") == "diamond")
    assert _mark(diamond)["color"] == U.outline_colour()


def test_trade_offs_holds_the_two_measurement_chart_alone():
    opt = D.load(D.LINE_A)
    assert U.trade_offs_label(opt, {"fitted": {}}) == wording.PAIR_FOLD
    src = inspect.getsource(U._more_charts)
    assert "_mixture" not in src and "_pair" in src
    assert "_triangle" in inspect.getsource(U._explored_body)


def test_every_new_chart_goes_through_show_chart():
    for fn in (U._tried, U._chooses, U._triangle, U._mixture, U._explored_map):
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
