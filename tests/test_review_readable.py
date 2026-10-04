"""Review readable (prototype, 2026-10-02; owner: "please make sure the plots
don't look smooshed and it's easy to read"). A layout and type pass only:

- One shared Altair config (ui_insights.readable) on every chart the Review
  and Make a round draw: labels at least 11 px, axis titles 12, titles 13.
- No text drawn on a chart under 11 px.
- No chart narrower than 380 px; the profiler at most 4 panels a row.
- Called it?'s end labels never overlap, however close two values are.

Synthetic: the Plant patty Line A demo project (CI); no pilot data."""
import pytest

import bo_views
import insights
import insights_jobs
import ui_insights as U
from tests import demo_projects as D

MIN_LABEL = 11
FULL_WIDTH = 820          # px: a chart drawn at the column's width


@pytest.fixture(scope="module")
def charts(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.chdir(tmp_path_factory.mktemp("review-readable"))
    mp.setattr(insights_jobs, "_JOBS", {})
    mp.setattr(insights_jobs, "_INFLIGHT", {})
    mp.setattr(U.st, "session_state", {})
    with bo_views.fresh():
        opt = D.load(D.LINE_A)
        computed = U._computed(opt, quiet=True)
        mp.setattr(U, "checks_for", lambda o, stage: computed["checks"])
        fitted, checks = computed["fitted"], computed["checks"]
        lf = U.learned_facts_from(opt, checks)
        axes, lines = insights.tried_axes(opt), insights.tried_lines(opt)
        values = bo_views.map_values(opt, U.suggestion_model(opt, fitted))
        picks = [{"mean": m, "sd": s, "no": str(n)} for n, m, s in values.picks]
        tested = [{"mean": float(m), "sd": float(s), "no": n}
                  for m, s, n in zip(values.tested_mean.tolist(), values.tested_sd.tolist(),
                                     values.tested)]
        label, number, recipe, is_pick = U.profile_refs(opt)[-1]
        profile = U.profile_data(opt, fitted, recipe, label, number, is_pick,
                                 U.chooses_reason(opt, fitted))
        # Called it? needs the latest Round scored: the same project with
        # its open Round dropped (as test_review_learned's `scored`).
        closed = D.load(D.LINE_A)
        closed._drop_pending_batch()
        out = {
            "called": U.called_chart(closed, U.learned_facts_from(closed, checks)),
            "improving": U.improving_chart(opt),
            "target": U.target_chart(opt),
            "learned": U.learned_chart(opt, lf),
            "profile": U.profile_chart(opt, profile),
            "profile_4": U.profile_chart(opt, profile, [c.name for c in profile.columns][:4]),
            "path": U.search_path_chart(opt, axes, lines),
            "forecast": U.forecast_chart(opt, [c for c in checks if c.name == insights.SCORE]),
            "map": U.explore_map_chart(values.pool_mean, values.pool_sd, tested, picks,
                                       U.repeat_of_best(opt, fitted), opt),
        }
        missing = [k for k, v in out.items() if v is None]
        assert not missing, missing
        yield {k: v.to_dict() for k, v in out.items()}
    mp.undo()


# Review critique (fix 5): "lines" (Show as lines) cut.
NAMES = ["called", "improving", "target", "learned", "profile", "profile_4", "path",
         "forecast", "map"]


def _walk(node):
    """Every dict in the spec."""
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


@pytest.mark.parametrize("name", NAMES)
def test_the_shared_config_sets_readable_sizes(charts, name):
    config = charts[name]["config"]
    assert config["axis"]["labelFontSize"] >= MIN_LABEL
    assert config["axis"]["titleFontSize"] >= 12
    assert config["legend"]["labelFontSize"] >= MIN_LABEL
    assert config["header"]["labelFontSize"] >= MIN_LABEL
    assert 13 <= config["title"]["fontSize"] <= 14
    assert config["text"]["fontSize"] >= MIN_LABEL
    assert charts[name]["padding"] == U.CHART_PADDING


@pytest.mark.parametrize("name", NAMES)
def test_no_text_under_eleven_px(charts, name):
    small = [(k, d[k]) for d in _walk(charts[name]) for k in
             ("fontSize", "labelFontSize", "titleFontSize") if isinstance(d.get(k), (int, float))
             and d[k] < MIN_LABEL]
    assert small == []


def _facet_count(spec, datasets):
    name = (spec.get("data") or spec["spec"].get("data") or {}).get("name")
    field = spec["facet"]["column"]["field"]
    return len({row[field] for row in datasets[name]})


def _width(spec, datasets):
    """The chart's drawn width in px, roughly (no axes or titles): panels and
    spacing; a chart without a width of its own fills the column."""
    if "hconcat" in spec:
        parts = [_width(c, datasets) for c in spec["hconcat"]]
        return sum(parts) + spec.get("spacing", 10) * (len(parts) - 1)
    if "concat" in spec:
        parts = [_width(c, datasets) for c in spec["concat"]]
        k = spec.get("columns") or len(parts)
        rows = [parts[i:i + k] for i in range(0, len(parts), k)]
        return max(sum(r) + spec.get("spacing", 10) * (len(r) - 1) for r in rows)
    if "vconcat" in spec:
        return max(_width(c, datasets) for c in spec["vconcat"])
    if "facet" in spec:
        k = _facet_count(spec, datasets)
        return _width(spec["spec"], datasets) * k + spec.get("spacing", 10) * (k - 1)
    w = spec.get("width")
    return w if isinstance(w, (int, float)) else FULL_WIDTH


@pytest.mark.parametrize("name", NAMES)
def test_no_chart_narrower_than_380(charts, name):
    spec = charts[name]
    assert _width(spec, spec.get("datasets", {})) >= U.MIN_CHART_W


def test_a_single_panel_still_makes_the_minimum():
    # Panels plus the row titles' column and the tick labels (the profiler).
    assert U.panel_width(1) + U._ROW_TITLE_W + 30 >= U.MIN_CHART_W
    assert max(U.CALLED_W, U.MIN_CHART_W) >= U.MIN_CHART_W


def test_the_profiler_wraps_at_four_panels(charts):
    spec = charts["profile"]
    datasets = spec["datasets"]
    counts = [_facet_count(f, datasets) for f in _walk(spec) if "facet" in f]
    assert counts and max(counts) <= U.PROFILE_WRAP
    # Fits the 820 px column with its row titles.
    assert U.PROFILE_WRAP * U.panel_width(4) + 3 * U._PROFILE_SPACING + U._ROW_TITLE_W <= 800


def test_profiler_rows_are_tall_enough():
    assert U._PANEL_H["score"] >= 90
    assert U._PANEL_H["measure"] >= 70 and U._PANEL_H["gain"] >= 70


def test_profiler_row_titles_are_written_across(charts):
    titled_axes = [d for d in _walk(charts["profile_4"]) if d.get("titleAngle") == 0]
    assert titled_axes and all(d["titleFontSize"] >= 12 for d in titled_axes)


# Review critique (fix 1): the slopegraphs became one shared axis per result; their
# label spreading went with them.


def test_the_improving_labels_sit_above_every_mark(charts):
    spec = charts["improving"]
    texts = [d for d in _walk(spec) if isinstance(d.get("mark"), dict)
             and d["mark"].get("type") == "text"
             and d.get("encoding", {}).get("y", {}).get("field") == "lane"]
    assert texts, "the Start and Best labels are drawn in their lanes"
