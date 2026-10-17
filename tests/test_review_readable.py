"""Review readable (prototype, 2026-10-02; owner: "please make sure the plots
don't look smooshed and it's easy to read"). A layout and type pass only:

- One shared Altair config (ui_insights.readable) on every chart the Review
  and Make a round draw: labels at least 11 px, axis titles 12, titles 13.
- No text drawn on a chart under 11 px.
- No chart narrower than 380 px.

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
        view = U.round_view(opt)
        out = {
            "improving": U.improving_chart(opt),
            "targets": U.targets_chart(opt, view),
            "amounts": U.amounts_chart(opt, view),
        }
        missing = [k for k, v in out.items() if v is None]
        assert not missing, missing
        yield {k: v.to_dict() for k, v in out.items()}
    mp.undo()


# One-glance Review (owner, 2026-10-14): the Called it? panels, the ordering
# chart, the profiler and the search path are gone.
# Review v3: Best so far, Best so far vs targets, Amounts in the top formulations.
NAMES = ["improving", "targets", "amounts"]
MIN_CHART_W = 380         # px: no chart narrower than this


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
    assert _width(spec, spec.get("datasets", {})) >= MIN_CHART_W


# Review critique (fix 1): the slopegraphs became one shared axis per result; their
# label spreading went with them.


