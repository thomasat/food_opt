"""Minor 14 (privacy): no chart drawn by the app may offer to send its data
off the machine. Every Altair chart's top-level spec carries
`usermeta.embedOptions.actions = False` (Vega Editor, view source, export
all turned off), and the Progress chart — once an `st.line_chart` with no
such hook — is now drawn with Altair so it can carry it too.
Spec: docs/superpowers/specs/2026-09-29-review-fixes-design.md §2, §9 Minor 14."""
import pytest
from tests.demo_build_reason import REVIEW_FOUR
import glob
import json
import os
import re

import insights
import ui_insights
import wording
from tests.test_app_ui import APP_PATH, burger, scored  # noqa: F401
from tests.test_insights_ui import _open_insights, insights_scored  # noqa: F401
from tests.test_insights_wave_c_ui import mixture_scored  # noqa: F401

_NO_ACTIONS = {"embedOptions": {"actions": False}}

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Every module the running app can import; tests, notebooks, experiments and
# the desktop packaging scripts are not part of the served app.
_APP_FILES = [f for f in glob.glob(os.path.join(_ROOT, "*.py"))]


# ------------------------------------------------------------------ #
#  Grep-style: no native Streamlit chart, and one door for altair_chart
# ------------------------------------------------------------------ #

def test_no_native_streamlit_chart_remains_in_app_code():
    """st.line_chart / st.bar_chart / st.area_chart have no actions hook at
    all (Minor 14): none of them may remain anywhere in the app's own code."""
    offenders = {}
    pattern = re.compile(r"st\.(line_chart|bar_chart|area_chart)\(")
    for path in _APP_FILES:
        text = open(path, encoding="utf-8").read()
        hits = pattern.findall(text)
        if hits:
            offenders[os.path.basename(path)] = hits
    assert offenders == {}


def test_every_altair_chart_call_goes_through_the_privacy_helper():
    """No bare `st.altair_chart(` call is left in the app (v3: the Review
    draws kept specs); the one chart call is show_spec's
    `st.vega_lite_chart(`, which sets the action menu off, so no chart can
    slip through with it on."""
    calls = {}
    for name, pattern in (("altair", r"st\.altair_chart\("), ("vega", r"st\.vega_lite_chart\(")):
        for path in _APP_FILES:
            n = len(re.findall(pattern, open(path, encoding="utf-8").read()))
            if n:
                calls[(name, os.path.basename(path))] = n
    assert calls == {("vega", "ui_insights.py"): 1}, calls


# ------------------------------------------------------------------ #
#  The helper itself
# ------------------------------------------------------------------ #

def test_private_sets_actions_off_at_the_top_level():
    import altair as alt
    import pandas as pd
    chart = alt.Chart(pd.DataFrame({"x": [1], "y": [2]})).mark_point().encode(x="x", y="y")
    spec = ui_insights._private(chart).to_dict()
    assert spec["usermeta"] == _NO_ACTIONS


def test_private_survives_layering_and_faceting():
    """The probed Streamlit-1.55 approach: usermeta must be set on the
    TOP-LEVEL spec, after any layering or faceting — set earlier, it would
    sit on an inner view Streamlit's conversion does not read from."""
    import altair as alt
    import pandas as pd
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4], "p": ["a", "b"]})
    base = alt.Chart(df).mark_point().encode(x="x", y="y")
    layered = alt.layer(base, base).properties(width=10, height=10).configure_view(stroke=None)
    faceted = alt.layer(base).facet(facet=alt.Facet("p:N"), columns=2)
    for chart in (layered, faceted):
        spec = ui_insights._private(chart).to_dict()
        assert spec["usermeta"] == _NO_ACTIONS


# ------------------------------------------------------------------ #
#  Every real chart builder in the file, fed real data
# ------------------------------------------------------------------ #

def _checks(opt):
    return insights.check_rounds(opt)


def _every_chart(opt):
    """One instance of every chart the app can draw for `opt`, built the
    same way ui_insights.render does it."""
    fitted = insights.fit_all(opt)
    checks = _checks(opt)
    ids = [int(x) for x in opt.formulation_ids]
    best = opt.best_index()
    start = insights.start_recipe(opt, best if best is not None else 0)

    charts = [ui_insights.improving_chart(opt)]

    track = ui_insights.measured_vs_predicted_chart(opt, checks)
    if track is not None:
        charts.append(track)

    name = insights.path_choices(opt)[0]
    path = insights.build_path(opt, name, start)
    path_chart = ui_insights.path_chart(opt, path, insights.path_predictions(opt, fitted, path))
    if path_chart is not None:
        charts.append(path_chart)

    changes = insights.tested_near_changes(opt, fitted, start)
    if changes.moved:
        charts.append(ui_insights.change_chart(changes.moved))

    pair_names = insights.pair_choices(opt)
    if len(pair_names) >= 2:
        points = insights.pair_points(opt, *pair_names[:2])
        pair = ui_insights.pair_chart(opt, *pair_names[:2], points)
        if pair is not None:
            charts.append(pair)

    mixture_names = insights.mixture_choices(opt)
    if len(mixture_names) >= 3:
        mix = insights.mixture_slice(opt, mixture_names[:3], start)
        if mix.refusal is None and not mix.start_breaks:
            preds = insights.mixture_predictions(opt, fitted, mix)
            charts.append(ui_insights.mixture_chart(opt, mix, preds, ids[best]))

    return charts


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_every_chart_builder_in_the_file_gets_the_privacy_wrapper(mixture_scored):
    charts = _every_chart(mixture_scored)
    # Guards against every builder failing silently and the loop testing
    # nothing: this project's docstring promises "enough for all three
    # charts" (two since the misses chart went) plus the Progress,
    # predictions, path and Compare all charts.
    assert len(charts) >= 6, len(charts)
    for chart in charts:
        spec = ui_insights._private(chart).to_dict()
        assert spec["usermeta"] == _NO_ACTIONS


# ------------------------------------------------------------------ #
#  The Progress chart, redrawn with Altair (was st.line_chart)
# ------------------------------------------------------------------ #


def test_progress_chart_keeps_the_axis_titles_and_a_gap(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tests.test_complete_results import _patty, _tell
    opt = _patty(rows=2)
    _tell(opt, 3, 1, drop=("Cook loss",))
    _tell(opt, 4, 2)
    spec = ui_insights.improving_chart(opt).to_dict()
    # One panel, Formulation on x, Overall score on y, and a waiting
    # formulation leaves a gap in the Overall-score dots (it never appears
    # in Best so far's own series: opt.best_so_far() repeats the value
    # before it).
    text = json.dumps(spec)
    assert wording.FORMULATION_CAP in text
    assert wording.OVERALL_SCORE_COLUMN in text
    assert wording.BEST_SO_FAR_COLUMN in text
    datasets = list(spec["datasets"].values())
    dots = [row[wording.FORMULATION_CAP] for rows in datasets for row in rows
            if wording.OVERALL_SCORE_COLUMN in row]
    steps = [row["x"] for rows in datasets for row in rows
             if "v" in row and "text" in row and "avg" not in row]
    assert dots and sorted(dots) == [1, 2, 4]      # Formulation 3 is waiting: no dot
    assert steps == [1, 2, 3, 4]                   # Best so far repeats through the wait


def _app():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(APP_PATH, default_timeout=180)


# ------------------------------------------------------------------ #
#  Every chart actually drawn by the running app, on real tabs
# ------------------------------------------------------------------ #

def _vega_specs(at):
    """Every `vega_lite_chart` element anywhere on the page: `at.get` walks
    the whole tree, tab bar included, so this covers Results (rendered on
    every run, whichever tab is selected) and Insights (rendered only while
    open) together."""
    return [json.loads(e.proto.spec) for e in at.get("vega_lite_chart")]


def test_every_chart_on_a_second_project_shape_disables_its_menu(insights_scored):
    """A different project shape (no triangle, no misses yet): whatever
    charts it does draw must still all carry the same usermeta."""
    at = _open_insights(_app())
    assert not at.exception
    specs = _vega_specs(at)
    assert specs  # the Progress chart alone is always there once scored
    bad = [s for s in specs if s.get("usermeta") != _NO_ACTIONS]
    assert bad == []
