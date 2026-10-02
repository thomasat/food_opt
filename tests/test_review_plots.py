"""Plot review (2026-10-02, owner: "I want to make sure the plots are all
interpretable"). Pinned on the chart specs of the Plant patty Line A demo:
every chart has a title (the hero's is its heading), labelled axes, a
legend for every encoded colour, tooltips with a "BO term" row, one
colour per meaning across charts, short captions, and a withheld chart's
reason. One model fit for the module."""
import re
import types

import pytest

import bo_views
import insights
import insights_jobs
import ui_insights as U
import wording
from tests import demo_projects as D

_COLOUR_CHANNELS = ("color", "fill", "stroke", "strokeDash", "shape", "opacity")


@pytest.fixture(scope="module")
def line_a(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.chdir(tmp_path_factory.mktemp("review-plots"))
    mp.setattr(insights_jobs, "_JOBS", {})
    mp.setattr(insights_jobs, "_INFLIGHT", {})
    mp.setattr(U.st, "session_state", {})
    with bo_views.fresh():
        opt = D.load(D.LINE_A)
        computed = U._computed(opt, quiet=True)
        mp.setattr(U, "checks_for", lambda o, stage: computed["checks"])
        yield opt, computed
    mp.undo()


def _specs(opt, computed):
    """{name: spec} for every chart the Review and Make a round tabs draw."""
    fitted, checks = computed["fitted"], computed["checks"]
    out = {}
    facts = types.SimpleNamespace(checks=checks)
    out["spark_predictions"] = U.spark_chart(opt, facts, "predictions")
    out["spark_progress"] = U.spark_chart(opt, facts, "progress")
    hero = U.chooses_data(opt, fitted, None, mode=U.CHOOSES_LINE)
    out["hero"] = U.chooses_chart(opt, hero, heights=U.HERO_HEIGHTS, compact=True)
    out["chooses"] = U.chooses_chart(opt, hero)
    out["tried"] = U.tried_chart(opt, insights.tried_axes(opt), insights.tried_lines(opt))
    out["progress"] = U.progress_chart(opt)
    out["gain"] = U.gain_chart(opt, bo_views.round_history(opt))
    out["mvp"] = U.measured_vs_predicted_chart(opt, checks)
    name = insights.path_choices(opt)[0]
    path = insights.build_path(opt, name, insights.start_recipe(opt, opt.best_index()))
    out["path"] = U.path_chart(opt, path, insights.path_predictions(opt, fitted, path))
    found = insights.tested_near_changes(opt, fitted, insights.start_recipe(opt, opt.best_index()))
    out["change"] = U.change_chart(found.moved)
    a, b = [o['name'] for o in opt.objectives][:2]
    out["pair"] = U.pair_chart(opt, a, b, insights.pair_points(opt, a, b))
    values = bo_views.map_values(opt, U.suggestion_model(opt, fitted))
    picks = [{"mean": m, "sd": s, "no": str(n)} for n, m, s in values.picks]
    tested = [{"mean": float(m), "sd": float(s), "no": n}
              for m, s, n in zip(values.tested_mean.tolist(), values.tested_sd.tolist(),
                                 values.tested)]
    out["map"] = U.explore_map_chart(values.pool_mean, values.pool_sd, tested, picks,
                                     U.repeat_of_best(opt, fitted), opt)
    start = U._start_number(opt)
    names = U._three_default(opt, fitted, insights.mixture_choices(opt), start)
    mix = insights.mixture_slice(opt, names, U.anchor_recipe(opt, start))
    preds = insights.mixture_predictions(opt, fitted, mix)
    out["triangle"] = U.mixture_chart(opt, mix, preds, start)
    missing = [k for k, v in out.items() if v is None]
    assert not missing, missing
    return {k: v.to_dict() for k, v in out.items()}


@pytest.fixture(scope="module")
def specs(line_a):
    return _specs(*line_a)


def _units(spec):
    """Every view with a mark, through layers, concats and facets."""
    if "mark" in spec:
        yield spec
    for key in ("layer", "concat", "vconcat", "hconcat"):
        for child in spec.get(key, []):
            yield from _units(child)
    if "spec" in spec:
        yield from _units(spec["spec"])


def _title(spec):
    t = spec.get("title")
    return t.get("text") if isinstance(t, dict) else t


def _words(text):
    return len([w for w in re.sub(r"[^\w%' ]", " ", text).split() if w])


TITLED = ["spark_predictions", "spark_progress", "tried", "progress", "gain", "mvp", "path",
          "change", "pair", "map", "triangle"]


@pytest.mark.parametrize("name", TITLED)
def test_each_chart_has_a_short_title(specs, name):
    title = _title(specs[name])
    assert title, name
    assert _words(title) <= 6, title


def test_the_hero_is_titled_by_its_heading():
    assert _words(wording.hero_title(6, True)) <= 6
    assert _words(wording.hero_title(None, False)) <= 6


# Charts whose axes are drawn by design without titles: the sparklines
# (Round labels only), the parallel coordinates (each axis named on top),
# the triangle (corner labels), the change bars' names, the path's
# facets (panel titles).
_NO_Y_TITLE = {"spark_predictions", "spark_progress", "tried", "triangle", "change", "path"}
_NO_X_TITLE = {"spark_predictions", "spark_progress", "tried", "triangle", "hero"}


@pytest.mark.parametrize("name", sorted(set(TITLED) | {"hero", "chooses"}))
def test_axes_are_titled(specs, name):
    """Each drawn x and y axis has a title in at least one layer (a rule or
    label layer sharing the axis takes the title Vega-Lite merges in)."""
    for ch, skip in (("x", _NO_X_TITLE), ("y", _NO_Y_TITLE)):
        if name in skip:
            continue
        encs = [u["encoding"][ch] for u in _units(specs[name])
                if ch in u.get("encoding", {}) and "field" in u["encoding"][ch]
                and u["encoding"][ch].get("axis", {}) is not None]
        if not encs:
            continue
        assert any(e.get("title") for e in encs), (name, ch)


@pytest.mark.parametrize("name", sorted(set(TITLED) | {"hero", "chooses"}))
def test_every_encoded_colour_has_a_legend(specs, name):
    """A field on a colour channel is explained by a legend somewhere in the
    chart (a layer may repeat it with legend None, as the Best so far line's
    inner colour does)."""
    shown, used = set(), set()
    for unit in _units(specs[name]):
        for ch in _COLOUR_CHANNELS:
            e = unit.get("encoding", {}).get(ch)
            if not isinstance(e, dict):
                continue
            for part in [e] + ([e["condition"]] if isinstance(e.get("condition"), dict) else []):
                if "field" in part:
                    used.add(part["field"])
                    if part.get("legend", {}) is not None:
                        shown.add(part["field"])
    assert used <= shown, (name, used - shown)


@pytest.mark.parametrize("name", sorted(set(TITLED) | {"hero", "chooses"}))
def test_marks_have_tooltips_with_a_bo_term(specs, name):
    tips = []
    for unit in _units(specs[name]):
        t = unit.get("encoding", {}).get("tooltip")
        if t:
            tips += t if isinstance(t, list) else [t]
    assert tips, name
    if name != "tried":       # model-free, its rows are the formulation's own
        assert any(t.get("title") == wording.PLOT_BO for t in tips), name


def _colours(spec):
    found = set()

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in ("color", "stroke", "fill") and isinstance(v, str):
                    found.add(v.lower())
                elif k == "value" and isinstance(v, str) and v.startswith("#"):
                    found.add(v.lower())
                elif k == "range" and isinstance(v, list):
                    found.update(c.lower() for c in v if isinstance(c, str))
                else:
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(spec)
    return found


def test_one_colour_per_meaning_across_charts(specs):
    ink = U.mark("bar_latest")[0].lower()
    orange = U.outline_colour().lower()
    blue = U.model_blue().lower()
    # Best so far: ink, wherever it is drawn.
    for name in ("hero", "tried", "progress", "pair", "map", "gain"):
        assert ink in _colours(specs[name]), name
    # This Round's suggestions: orange.
    for name in ("hero", "tried", "map"):
        assert orange in _colours(specs[name]), name
    # The model: blue.
    for name in ("hero", "path", "change", "gain"):
        assert blue in _colours(specs[name]), name
    # An Overall score: the one ramp, never Vega's blues (the model's).
    ramp = [c.lower() for c in U.score_colours()]
    for name in ("tried", "pair", "triangle"):
        assert set(ramp) <= _colours(specs[name]), name
        assert '"blues"' not in str(specs[name]).replace("'", '"'), name
    # No default palette: every colour channel with a field has its range.
    for name, spec in specs.items():
        for unit in _units(spec):
            for ch in ("color", "fill", "stroke"):
                e = unit.get("encoding", {}).get(ch)
                if isinstance(e, dict) and e.get("type") == "nominal" and "field" in e:
                    assert "range" in e.get("scale", {}), (name, ch, e)


def test_the_band_says_what_it_is(specs):
    assert wording.PLOT_BAND in str(specs["hero"])
    assert "95 %" in wording.PLOT_BAND and "95 %" in wording.PLOT_INTERVAL
    assert wording.PLOT_INTERVAL in str(specs["path"])


def test_a_zoomed_score_axis_says_so(specs):
    """Line A's hero spans about 4 points of 100: its axis says zoomed."""
    titles = [u["encoding"]["y"].get("title") for u in _units(specs["hero"])
              if "y" in u.get("encoding", {})]
    assert any(isinstance(t, list) and wording.PLOT_ZOOMED in t[-1] for t in titles), titles
    assert U.zoomed("Score (of 100)", 0, 100, 100) == "Score (of 100)"
    assert U.zoomed("Score (of 100)", 88, 91, 100) == f"Score (of 100, {wording.PLOT_ZOOMED})"


def test_the_hero_band_is_cut_at_the_ceiling(specs):
    rows = [r for d in specs["hero"]["datasets"].values() for r in d if r.get("mark") == "curve"]
    assert rows and all(0 <= r["low"] <= r["high"] <= 100 for r in rows)


def test_short_captions_and_withheld_reasons():
    for text in (wording.PLOT_CAPTION_PROGRESS, wording.PLOT_CAPTION_GAIN,
                 wording.PLOT_CAPTION_MVP, wording.PLOT_CAPTION_CHANGE,
                 wording.PLOT_CAPTION_PATH, wording.PLOT_CAPTION_MAP,
                 wording.PLOT_CAPTION_PAIR, wording.PLOT_CAPTION_PAIR_NONE,
                 wording.plot_caption_triangle(), wording.plot_caption_triangle(wording.LAYER_SD),
                 wording.plot_caption_triangle(wording.LAYER_GAIN), wording.HERO_CAPTION,
                 wording.tried_caption()):
        assert _words(text) <= 8, text
    for text in (wording.PLOT_SPARK_WAIT, wording.PLOT_GAIN_WITHHELD,
                 *wording.HERO_WITHHELD.values()):
        assert _words(text) <= 10, text


def test_a_sparkline_under_two_rounds_is_withheld_with_its_reason():
    opt = D.load(D.SNACK)
    assert U.spark_chart(opt, None, "progress") is None
    src = open(U.__file__).read()
    body = src[src.index("def _card_body"):src.index("def predictions_spark_frame")]
    assert "wording.PLOT_SPARK_WAIT" in body


def test_sparklines_name_their_rounds_and_the_95_aim(specs):
    s = specs["spark_predictions"]
    assert wording.PLOT_SPARK_TARGET in str(s)
    xs = [u["encoding"]["x"] for u in _units(s) if "x" in u.get("encoding", {})
          and u["encoding"]["x"].get("field") == "round"]
    assert xs and all(x["axis"]["values"] for x in xs)
    # The predictions line runs over the whole 0-100 %.
    ys = [u["encoding"]["y"]["scale"]["domain"] for u in _units(s)
          if u.get("encoding", {}).get("y", {}).get("field") == "inrange"]
    assert ys and all(d == [0.0, 1.0] for d in ys)


def test_try_one_says_when_the_start_is_at_a_limit(line_a):
    opt, _ = line_a
    name = insights.path_choices(opt)[0]
    lo, hi = insights._var(opt, name)['bounds']
    path = types.SimpleNamespace(name=name, start_amount=float(hi), unit="g", mode="fill",
                                 fill_name="Water")
    assert U.path_at_limit(opt, path) == "Highest"
    assert U._path_head(opt, path).startswith(f"{name} is at its Highest")
    path.start_amount = (lo + hi) / 2
    assert U.path_at_limit(opt, path) is None
    assert U._path_head(opt, path).startswith("More ")


def test_tried_end_label_counts_a_bundle_at_a_limit(line_a):
    opt, _ = line_a
    axes, lines = insights.tried_axes(opt), insights.tried_lines(opt)
    _, ends, _ = U.tried_frame(opt, axes, lines)
    counted = [t for t in ends["text"] if re.search(r"\(\d+ of \d+\)$", t)]
    # Line A: Textured pea protein and Beet juice powder sit at a limit in
    # most formulations.
    assert counted, list(ends["text"])
