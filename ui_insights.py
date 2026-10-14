"""The Insights tab. Builders here return Altair charts from computed facts
and never call Streamlit; render() at the bottom is the only Streamlit code."""
import collections
import contextlib
import dataclasses
import math
import os
import re
import threading
import time
import types

import altair as alt
import pandas as pd
import streamlit as st

import bo_views
import insights
import insights_cache
import insights_jobs
import learned
import sample_projects
import wording
from food_bo import fmt_setting, join_unit, number_list
from ui_helpers import (SCROLL_KEY, SETUP_CALLOUT_KEY, SETUP_CALLOUT_SLOT,
                        TAB_BATCH, app_rerun, TAB_SETUP, ask_tab, confirmation_open,
                        go_to_tab, link, open_rows, readiness, table_height)
# Ftue fix C: where the links land (each fold in a keyed place).
from ui_helpers import (ADD_OWN_KEY, ADD_OWN_PLACE, EXPLORED_PLACE, REPEAT_REQUEST_KEY,
                        WHY_PLACE)

_BAND = 0.25
# Text drawn on a chart (labels, the band's name): at least 4.5:1 against
# the page in each theme. No single grey reaches that on both white and
# Streamlit's dark background, so the theme picks (0.8.0 review fixes §6).
_MUTED_LIGHT = "#5c6370"      # 6.0:1 on white
_MUTED_DARK = "#a3a8b4"       # 7.9:1 on #0e1117


def muted():
    """The chart text colour for the theme the reader has on."""
    try:
        dark = st.context.theme.type == "dark"
    except Exception:
        dark = False
    return _MUTED_DARK if dark else _MUTED_LIGHT


def theme():
    """"dark" or "light": the theme the reader has on (light when unknown)."""
    return "dark" if muted() == _MUTED_DARK else "light"


# D1 (spec 2026-10-01; 01 F1, O16; review M7): every mark that carries
# meaning, as (colour, opacity) together per theme, at least 3:1 against
# that theme's page once blended (WCAG 1.4.11, graphics). The interval bars
# were drawn black at opacity 0.25/0.6 — invisible on #0e1117 — and at
# opacity 0.25 no colour reaches 3:1 there, so each bar is opaque and the
# latest Round's is the stronger one in either theme.
_MARKS = {
    "light": {"bar_latest": ("#3d434d", 1.0),      # 10.0:1 on white
              "bar": ("#8b919c", 1.0),             # 3.2:1
              "reference": (_MUTED_LIGHT, 1.0),    # the path's dotted start, 6.0:1
              "untested_edge": ("#8b919c", 1.0),   # the grey band's edges, 3.2:1
              # "Can you have both?": a waiting formulation's hollow dot, Best
              # so far's ring, a target line (integration of the e2e fixes).
              "neutral": ("#8b919c", 1.0)},        # 3.2:1
    "dark": {"bar_latest": ("#c8ccd4", 1.0),       # 11.7:1 on #0e1117
             "bar": ("#8b919c", 1.0),              # 6.0:1
             "reference": (_MUTED_DARK, 1.0),      # 7.9:1
             "untested_edge": ("#8b919c", 1.0),    # 6.0:1
             "neutral": ("#8b919c", 1.0)},         # 6.0:1
}


def mark(name):
    """(colour, opacity) of the mark `name` in the reader's theme."""
    return _MARKS[theme()][name]


# Fills under the marks: visible as a region, kept light enough that the
# line and its 95 % interval drawn over them still read, so not held to the
# marks' 3:1 (the band's edge rules carry that). 01 O16: grey at 0.12 was
# 1.14:1 on the dark page; #8b919c at 0.35 is 1.8:1 there.
_FILLS = {
    "light": {"untested": ("#808080", 0.12)},       # CSS gray, 1.14:1 on white, as before
    "dark": {"untested": ("#8b919c", 0.35)},        # 1.8:1 on #0e1117
}


# The colour scales' range in Vega's "blues", per theme, darker for higher.
# On the dark page the top of the scheme, rgb(10,74,144), is about 2:1 (O16):
# it stops at 0.8 there (3.6:1); its low end, 0.25, is 10.6:1. On white the
# low end was 0.25, 1.8:1 (final review M-10): it starts at 0.55 there
# (#509aca, 3.1:1), so every stop reaches 3:1 in both themes.
_BLUES_EXTENT = {"light": (0.55, 1.0), "dark": (0.25, 0.8)}


def decimals(opt, name):
    """One number of decimals for everything this measurement shows — a
    prediction, its likely range, a tooltip — from the width of its Range:
    0 from 100, 1 from 1, 2 below. The Overall score (Range 0 to the
    ceiling) gets 0: a predicted score is a whole number."""
    lo, hi = insights._range_of(opt, name)
    width = abs(hi - lo)
    return 0 if width >= 100 else 1 if width >= 1 else 2

# Minor 14 (privacy): no chart this tab draws may offer to send its data
# off the machine. `usermeta` on the TOP-LEVEL spec — set after any layering
# or faceting — survives Streamlit 1.55's conversion to Vega-Lite and
# overrides Streamlit's own embed options; set on an inner view instead, it
# would not reach the frontend at all.
_NO_CHART_ACTIONS = {"embedOptions": {"actions": False}}


def _private(chart):
    """`chart`, its action menu ("…" → Open in Vega Editor, view source,
    export) turned off. Pure: no Streamlit call, so it can be tested on its
    own against whatever a builder below actually returns."""
    return chart.properties(usermeta=_NO_CHART_ACTIONS)


def show_chart(chart, **kwargs):
    """The one door this file draws an Altair chart through: every chart on
    this tab goes here instead of calling st.altair_chart directly, so none
    of a project's data can leave the machine through its action menu. The
    shared type scale (readable) is set here too, for any chart a builder
    did not already set it on."""
    st.altair_chart(_private(readable(chart)), **kwargs)


# Slim Review: a Round step redraws two charts. Building an
# Altair chart and converting it is most of a step's time (about 0.5 s on
# the yogurt pilot), so the section's charts are converted once — through
# this file's one door (_private, readable) and Streamlit's own conversion —
# and the Vega-Lite spec kept; a step sends the kept spec.
_SPEC_CACHE = collections.OrderedDict()
_SPEC_KEEP = 64
_SPEC_LOCK = threading.Lock()


def _vega_spec(chart):
    """The spec st.altair_chart would send for `chart` (privacy and type
    scale applied, as show_chart)."""
    chart = _private(readable(chart))
    try:
        from streamlit.elements.vega_charts import _convert_altair_to_vega_lite_spec
    except Exception:       # a Streamlit without it: Altair's own
        return chart.to_dict()
    return _convert_altair_to_vega_lite_spec(chart)


def chart_spec(key, build):
    """build()'s chart as a kept Vega-Lite spec (None when it draws
    nothing), worked out once per `key`."""
    with _SPEC_LOCK:
        if key in _SPEC_CACHE:
            _SPEC_CACHE.move_to_end(key)
            return _SPEC_CACHE[key]
    chart = build()
    spec = None if chart is None else _vega_spec(chart)
    with _SPEC_LOCK:
        _SPEC_CACHE[key] = spec
        while len(_SPEC_CACHE) > _SPEC_KEEP:
            _SPEC_CACHE.popitem(last=False)
    return spec


def _spec_copy(spec):
    """A copy Streamlit may change: the datasets shared, the rest copied."""
    import copy
    out = {k: copy.deepcopy(v) for k, v in spec.items() if k != "datasets"}
    if "datasets" in spec:
        out["datasets"] = dict(spec["datasets"])
    return out


def show_spec(spec, clear=False, **kwargs):
    """show_chart for a kept spec (chart_spec): the action menu stays off
    (usermeta at the top level, as _private sets it). `clear`: no
    background of its own (the Replay panel's tint shows through)."""
    out = _spec_copy(spec)
    out["usermeta"] = _NO_CHART_ACTIONS
    if clear:
        out["background"] = "transparent"
    st.vega_lite_chart(spec=out, **kwargs)


# Review readable (2026-10-02; owner: "please make sure the plots don't look
# smooshed and it's easy to read"): one type scale and padding for every
# chart the Review and Make a round draw, set on the top-level spec, so
# Streamlit's theme (14 px legends and titles, 16 px label padding) and the
# old 9–10 px marks no longer mix. No text on a chart is under FONT_SMALL.
FONT_TITLE = 13          # a chart's own title
FONT_AXIS_TITLE = 12     # axis, legend and header titles
FONT_LABEL = 12          # axis labels, legend labels, facet headers, end labels
FONT_SMALL = 11          # the least: tick labels in small multiples, notes
CHART_PADDING = {"left": 4, "top": 6, "right": 14, "bottom": 4}
MIN_CHART_W = 380        # px: no chart narrower than this


def readable(chart):
    """`chart` with the shared type scale, legend and padding (one Altair
    config for the Review's charts). Idempotent; top-level charts only. A
    single or layered view fits the column's width only: its height is the
    plot's own (Streamlit's "fit" squeezed the plot by its legend and axes,
    so rows and lanes sized in px came out short)."""
    if isinstance(chart, (alt.Chart, alt.LayerChart)) and chart.autosize is alt.Undefined:
        # A chart that sets its own (the square sensory profile) keeps it.
        chart = chart.properties(autosize=alt.AutoSizeParams(type="fit-x", contains="padding"))
    return (chart
            .configure_axis(labelFontSize=FONT_LABEL, titleFontSize=FONT_AXIS_TITLE,
                            titleFontWeight="normal", labelPadding=4, titlePadding=8)
            .configure_legend(labelFontSize=FONT_LABEL, titleFontSize=FONT_AXIS_TITLE,
                              labelLimit=0, padding=4, rowPadding=4, columnPadding=16)
            .configure_header(labelFontSize=FONT_LABEL, titleFontSize=FONT_AXIS_TITLE)
            .configure_title(fontSize=FONT_TITLE, anchor="start", offset=8)
            .configure_text(fontSize=FONT_SMALL)
            .properties(padding=CHART_PADDING))


# --- review plots (2026-10-02) ---
# One meaning, one colour, on every chart (Okabe-Ito orange and the ink and
# blue tokens above, all at least 3:1 in both themes):
#   Best so far          ink (mark("bar_latest")), a thick line, a ring or a dashed rule;
#                        another start (Try one) the dotted reference token
#   this Round's picks   orange (outline_colour())
#   tested formulations  neutral grey rings (mark("neutral"))
#   the model            blue (model_blue(); its 95 % band in _BAND_FILL)
#   an Overall score     the score ramp (score_scale), measured or predicted,
#                        the legend says which
# Every chart has a title (wording.PLOT_*), every colour a legend entry, and
# a "BO term" row in its tooltip naming the exact quantity.
_TITLE_SIZE = 13


def titled(chart, text):
    """`chart` with its plain title, left-aligned above it."""
    return chart.properties(title=alt.TitleParams(text, anchor="start", fontSize=_TITLE_SIZE,
                                                  fontWeight="bold", offset=8))


def zoomed(title, lo, hi, full):
    """An axis title, with "zoomed" added inside its unit's brackets when
    the axis shows under a quarter of its whole Range (`full`), so a small
    difference never reads as a large one unsaid."""
    if full and full > 0 and (hi - lo) < 0.25 * full:
        if title.endswith(")"):
            return f"{title[:-1]}, {wording.PLOT_ZOOMED})"
        return f"{title} ({wording.PLOT_ZOOMED})"
    return title


def bo_tip(field="bo"):
    """The tooltip's expert row."""
    return alt.Tooltip(f"{field}:N", title=wording.PLOT_BO)


def panel_label(opt, name):
    if name == insights.SCORE:
        return wording.OVERALL_SCORE_COLUMN
    obj = next(o for o in opt.objectives if o['name'] == name)
    return opt._measurement_column(obj)


def track_panels(opt):
    """The measurements in the order Results lists them (largest Share of
    score first), then the Overall score."""
    return [o['name'] for o in opt.measurements_by_importance()] + [insights.SCORE]


def _only(mark):
    """Each layer draws only its own rows of the chart's one long frame. A
    layer given a frame of its own is drawn whole in EVERY facet panel, so a
    target, a grey interval or a start marker would show up everywhere."""
    return alt.FieldEqualPredicate(field="mark", equal=mark)


def _padded(values):
    """One domain for both axes of a measured-against-predicted panel: the
    values' extent padded 5 % (review M4), never zero-width."""
    lo, hi = min(values), max(values)
    pad = 0.05 * (hi - lo) if hi > lo else max(0.5, 0.05 * abs(hi))
    return [lo - pad, hi + pad]


def measured_vs_predicted_chart(opt, checks, height=None):
    """Can you trust the predictions? — one panel per measurement, then the
    Overall score (spec 2026-09-30 §2.2, review M4). Each check is a dot at
    (predicted before it was made, measured) with its likely range as a thin
    vertical bar; the dashed line is "exactly as predicted". Filled = inside
    its likely range; the latest Round is drawn darker. Both axes of a panel
    share one domain, padded 5 %, so the line is at 45 degrees."""
    if not checks:
        return None
    names = [n for n in track_panels(opt) if any(c.name == n for c in checks)]
    latest = max(c.round for c in checks)
    formulation = _say(opt, wording.FORMULATION_CAP)
    predicted, measured = wording.TRACK_PREDICTED, wording.TRACK_MEASURED
    low, high = wording.TRACK_LOW, wording.TRACK_HIGH
    inside_name, outside_name = _say(opt, wording.PLOT_INSIDE), _say(opt, wording.PLOT_OUTSIDE)
    latest_name, earlier_name = _say(opt, wording.PLOT_LATEST), _say(opt, wording.PLOT_EARLIER)
    ink, _ = mark("bar_latest")
    # Review plots: one legend for the whole figure — filled inside its 95 %
    # range, hollow outside; the latest Round dark, earlier ones lighter.
    none_name = _say(opt, wording.NO_FORECAST)
    # Fix round 2: a whole-Range range is "No forecast" (neutral), keyed
    # only when one is drawn.
    nf = any(no_forecast(opt, c) for c in checks if c.name in names)
    fills = alt.Scale(domain=[inside_name, outside_name] + ([none_name] if nf else []),
                      range=[ink, "transparent"] + ([mark("neutral")[0]] if nf else []))
    whens = alt.Scale(domain=[latest_name, earlier_name], range=[1.0, 0.45])
    legend = alt.Legend(title=None, orient="top", labelLimit=0)
    panels = []
    for name in names:
        mine = [c for c in checks if c.name == name]
        d = decimals(opt, name)
        domain = _padded([v for c in mine for v in (c.low, c.high, c.measured, c.predicted)])
        scale = alt.Scale(domain=domain, nice=False, zero=False)
        df = pd.DataFrame([{
            formulation: c.formulation, wording.ROUND_CAP: c.round,
            predicted: round(c.predicted, d), low: round(c.low, d), high: round(c.high, d),
            measured: round(c.measured, max(1, d)), "inside": c.inside,
            "latest": c.round == latest,
            "state": (none_name if no_forecast(opt, c) else
                      inside_name if c.inside else outside_name),
            "when": latest_name if c.round == latest else earlier_name,
            "bo": wording.PLOT_BO_ROUND_AHEAD} for c in mine])
        x = alt.X(f"{predicted}:Q", title=predicted, scale=scale)
        y = alt.Y(f"{measured}:Q", title=measured, scale=scale)
        fmt = f".{d}f"
        tip = [alt.Tooltip(f"{f}:{t}", title=f, **({"format": v} if v else {}))
               for f, t, v in ((formulation, "Q", "d"), (wording.ROUND_CAP, "N", None),
                               (predicted, "Q", fmt), (low, "Q", fmt), (high, "Q", fmt),
                               (measured, "Q", f".{max(1, d)}f"))] + [bo_tip()]
        latest_first = alt.FieldEqualPredicate(field="latest", equal=True)
        line = (alt.Chart(pd.DataFrame({"a": domain, "b": domain}))
                .mark_line(color="gray", strokeDash=[4, 3], strokeWidth=1, tooltip=False)
                .encode(x=alt.X("a:Q", scale=scale, title=predicted),
                        y=alt.Y("b:Q", scale=scale, title=measured)))
        base = alt.Chart(df)
        (new_c, new_o), (old_c, old_o) = mark("bar_latest"), mark("bar")
        bars = base.mark_rule(strokeWidth=2).encode(
            x=x, y=alt.Y(f"{low}:Q", scale=scale, title=measured), y2=f"{high}:Q",
            color=alt.condition(latest_first, alt.value(new_c), alt.value(old_c)),
            opacity=alt.condition(latest_first, alt.value(new_o), alt.value(old_o)), tooltip=tip)
        dots = base.mark_point(size=50, strokeWidth=1.5, color=ink).encode(
            x=x, y=y, fill=alt.Fill("state:N", scale=fills, legend=legend),
            opacity=alt.Opacity("when:N", scale=whens, legend=legend), tooltip=tip)
        panels.append(alt.layer(line, bars, dots)
                      .properties(width=220, height=220, title=_say(opt, panel_label(opt, name))))
    if not panels:
        # Checks only for outputs no longer in the project: nothing to draw.
        return None
    if height is not None and len(panels) == 1:
        # Review four: Q1's one panel, the width of its column.
        # The answer line names the Overall score: no panel title.
        return panels[0].properties(width="container", height=height, title="")
    return titled(alt.concat(*panels, columns=3), _say(opt, wording.PLOT_MVP))


# ------------------------------------------------------------------ #
#  render: the only Streamlit code in this module
# ------------------------------------------------------------------ #

# The fitted models, keyed by insights.data_key, shared by Make a round (its
# predictions) and this tab, so neither fits them twice. None: the fit
# failed, cached like a success so it is not tried again on every rerun.
FITTED_KEY = "_insights_fitted"
# The Round-by-Round checks, under the same key. They take the models from
# FITTED_KEY and add only the checks.
CACHE_KEY = "_insights_cache"
# Each Round's checks on their own, keyed by insights.round_keys: after a
# new result only the Rounds whose inputs changed are fitted again.
ROUNDS_KEY = "_insights_rounds"
# The formulation predicted to score highest, keyed by insights.setup_key: a
# limit edit on Set up changes it without changing a fitted model.
BEST_KEY = "_insights_best"
# All of the above are a project's own: app.py's _reset_project_session drops
# them with the two section-2 controls below.
CHOICE_KEY = "insights_choice"
START_KEY = "insights_start"
# What "Compare all" ranks by: the Overall score or one measurement (§4.1).
SPREAD_KEY = "insights_spread_on"
# Where a link lands on this tab (spec 2026-09-30-bo-views §5; 04 §5): the
# Prediction check (the top), Progress with Next step under it, and the
# Measured against predicted fold, which a link opens by setting its key.
TRUST_SECTION = "insights_trust"
STAND_SECTION = "insights_stand"
EACH_KEY = "insights_each"
# BO views §3.3 (review I6, I8, M14): the layer each chart shows, the start
# the Uncertainty and Expected gain layers use (a suggestion may be one),
# their worked-out values, and whether the Progress chart fold is open.
LAYER_KEY = "insights_layer"
PATH_LAYER_KEY = "insights_path_layer"
LAYER_START_KEY = "insights_layer_start"
PATH_LAYER_CACHE = "_insights_path_layer"
PROGRESS_KEY = "insights_progress_fold"
HOW_KEY = "insights_how_fold"
# Insights for a first-time user (spec 2026-10-01 rev. 2, review M3, R-I14):
# the two new folds' keys, here so W1's link and W3's fold agree.
EXPLORED_KEY = "insights_explored"
MORE_KEY = "insights_more"
_LAYERS = (wording.LAYER_SCORE, wording.LAYER_SD, wording.LAYER_GAIN)
# The "Predicted above Best so far" outline: Okabe-Ito orange, which holds
# its contrast on white and on Streamlit's dark background (review M14).
_OUTLINE_LIGHT = "#B35C00"     # about 4.6:1 on white (T56 m4)
_OUTLINE_DARK = "#E69F00"      # on Streamlit's dark background


def outline_colour():
    """The Predicted above Best so far outline for the reader's theme."""
    return _OUTLINE_DARK if muted() == _MUTED_DARK else _OUTLINE_LIGHT


def fitted_models(opt, working=wording.INSIGHTS_WORKING, bar=None):
    """insights.fit_all for this project's data, fitted once per data_key.
    None when the fit failed. With `bar` (Insights' one progress bar) the
    fit is its first step instead of a spinner of its own."""
    key = insights.data_key(opt)
    hit = st.session_state.get(FITTED_KEY)
    if hit and hit[0] == key:
        return hit[1]
    if bar is not None:
        order = _checked(opt)
        done = len(_done_already(opt))
        bar.progress(done / (len(order) + 1),
                     text=_say(opt, wording.insights_progress(done, len(order))))
    quiet = bar is not None or working is None
    with (contextlib.nullcontext() if quiet else st.spinner(working)), \
            insights.memo_in_use(memo(opt)):
        try:
            # Through the module attribute, so a test can stand in for the fit.
            value = insights.fit_all(opt)
        except Exception:
            value = None
    st.session_state[FITTED_KEY] = (key, value)
    return value


def _checked(opt):
    """The Rounds that get a check, in order: the count on the one line."""
    return [r for r, first in insights.round_order(opt) if first >= insights.MIN_ROWS]


def _done_already(opt):
    """The checked Rounds counted done before anything is fitted here
    (spec 2026-10-01 C1, review I11; 04 M-1): those whose checks are kept,
    and those whose models are all in the memo — the workers' own count
    (insights_jobs.rounds_done) — so the line never starts at 0 for a
    project already checked, nor goes back when the workers fail."""
    keys = insights.round_keys(opt)
    kept = st.session_state.get(ROUNDS_KEY, {})
    mem = memo(opt)
    models = {}
    for s in _plan(opt):
        if s.stage != insights_jobs.NOW:
            models.setdefault(s.stage, []).append(s.key in mem)
    return {r for r in _checked(opt)
            if keys.get(r) in kept or (r in models and all(models[r]))}


def _nothing_to_fit(opt):
    """Every model this data needs is in the memo: what is left is
    conditioning, with no progress line (04 I-1)."""
    return insights_jobs.ready(_plan(opt), memo(opt))


def _stepper(opt, bar):
    """check_rounds' `checking`: moves the one progress bar on as each Round
    is checked (0.8.0 review fixes §6, §9 I11), with the count the workers'
    line shows (Plan A review C I-1): "3 of 9 Rounds checked…". A Round
    already done counts from the start (C1), so the count never goes back."""
    order = _checked(opt)
    done = _done_already(opt)

    @contextlib.contextmanager
    def checking(r, last):
        bar.progress(min(1.0, (len(done) + 1) / (len(order) + 1)),
                     text=_say(opt, wording.insights_progress(len(done), len(order))))
        yield
        if r in order:
            done.add(r)
    return checking


def _cached(opt):
    """_computed's value for this project's data when it is already worked
    out, else None."""
    hit = st.session_state.get(CACHE_KEY)
    return hit[1] if hit and hit[0] == insights.data_key(opt) else None


def cached_checks(opt):
    """The Round-by-Round checks for this project's data when they are
    already worked out — by this tab, or from the workers' models (at
    least every Round's Overall score and the verdict's Rounds) — else ().
    Results and Make a round read them here and never fit for them."""
    return checks_for(opt, "trust") or ()


def _computed(opt, bar=None, quiet=False):
    """The fitted models and the Round-by-Round checks, once per data_key.
    `bar`, a slot under the header, shows one progress bar while they are
    worked out; without it (a test, or a caller with no slot) a slot of
    its own; `quiet`: none at all (the models are in: nothing is fitted)."""
    hit = _cached(opt)
    if hit is not None:
        return hit
    key = insights.data_key(opt)
    bar = None if quiet else (bar if bar is not None else st.empty())
    fitted = fitted_models(opt, working=None) if quiet else fitted_models(opt, bar=bar)
    if fitted is None:
        value = _FAILED
    else:
        rounds = st.session_state.setdefault(ROUNDS_KEY, {})
        checking = None if quiet else _stepper(opt, bar)
        try:
            with insights.memo_in_use(memo(opt)):
                checks = insights.check_rounds(opt, cache=rounds, checking=checking)
            value = {"fitted": fitted, "checks": checks}
        except Exception:
            # Cached like a success, under the same key: one line on the
            # tab instead of a traceback, and not tried again every rerun.
            value = _FAILED
    st.session_state[CACHE_KEY] = (key, value)
    return value


_FAILED = {"failed": True}


# ------------------------------------------------------------------ #
#  0.8.0 speed (spec 2026-09-30 §10; review B2, I1, I15): the models
#  are fitted in worker processes as soon as a project opens or its data
#  changes, kept here and beside the project, and each screen draws what
#  is ready. Without workers everything below reads as ready, and the
#  screens fit in this process as before.
# ------------------------------------------------------------------ #

# (project name, {insights.model_key: fitted model entry}): read from
# beside the project on first use, filled by the workers and by every fit
# made here; insights.fit() takes a model from it instead of fitting.
MEMO_KEY = "_insights_memo"
# (data_key, [insights_jobs.Spec]): what this project's data needs fitted.
PLAN_KEY = "_insights_plan"
# {(data_key, stage): checks}: a part of the checks worked out while the
# rest are still being fitted.
PARTIAL_KEY = "_insights_partial"
# What was last written beside the project, so it is written once per state.
SAVED_KEY = "_insights_saved"
# What the screen drew last, for the poller to compare with.
SEEN_KEY = "_insights_seen"
# How often a screen waiting for the workers looks again (seconds).
POLL_SECS = 0.5
# (data_key, Rounds done) on the progress line Insights drew last while
# filling in: the poller reruns the page when a Round lands (C1).
PROGRESS_SEEN_KEY = "_insights_progress_seen"
# The parts of the Insights tab (app._insights_stages): the only screen
# with the progress line.
_INSIGHTS_STAGES = ("stand", "all")


def memo(opt):
    """This project's fitted models, read from beside the project (with
    its Round checks, predicted best and next Round's expected
    improvement) the first time it is asked for."""
    hit = st.session_state.get(MEMO_KEY)
    if hit is None or hit[0] != opt.project_name:
        saved = insights_cache.load(opt.project_name)
        st.session_state[MEMO_KEY] = (opt.project_name, saved.memo)
        rounds = st.session_state.setdefault(ROUNDS_KEY, {})
        for key, checks in saved.rounds.items():
            rounds.setdefault(key, checks)
        if saved.best is not None and BEST_KEY not in st.session_state:
            st.session_state[BEST_KEY] = saved.best
        # The next Round's expected improvement (C1): latest_gain takes it
        # when its key is still this setup's, instead of searching again.
        if saved.gain is not None:
            st.session_state[NEXT_GAIN_KEY] = saved.gain
    return st.session_state[MEMO_KEY][1]


def _plan(opt):
    key = insights.data_key(opt)
    hit = st.session_state.get(PLAN_KEY)
    if hit is None or hit[0] != key:
        hit = (key, insights_jobs.plan(opt))
        st.session_state[PLAN_KEY] = hit
    return hit[1]


def start(opt):
    """Called by app.py on every run with a project open: the background
    fits for this project's data — started after it opens and after
    anything that changes its data (Save results, Move to Not scored, a
    waiting value entered), and nothing when they are running or done.
    Returns the Job while fits are in flight, else None (no workers, too
    few formulations, or everything is in)."""
    if opt is None or opt.complete_count() < insights.MIN_ROWS or not insights_jobs.workers_on():
        return None
    mem, specs = memo(opt), _plan(opt)
    if insights_jobs.ready(specs, mem):
        save(opt)
        return None
    pool = insights_jobs.pool()        # started only when there is something to fit
    if pool is None:
        return None
    job = insights_jobs.job_for(opt, pool, mem, specs)
    if job.poll(mem):
        save(opt)
    return None if job.failed else job


def waiting(opt, stage):
    """True while the workers are still fitting what `stage` needs
    (insights_jobs.stage_specs); False once it is in, and always without
    workers — the screen then fits it here, as before."""
    if start(opt) is None:
        return False
    specs = insights_jobs.stage_specs(opt, _plan(opt), stage)
    return not insights_jobs.ready(specs, memo(opt))


def _partial_checks(opt, stage):
    """The checks `stage` needs, from the fitted models alone; None until
    they are all in. When the verdict's Rounds came out short of plan (a fit
    failed), the verdict could read other Rounds: then every check, once
    everything is in."""
    specs = insights_jobs.stage_specs(opt, _plan(opt), stage)
    mem = memo(opt)
    if not insights_jobs.ready(specs, mem):
        return None
    data = insights.data_key(opt)
    parts = st.session_state.setdefault(PARTIAL_KEY, {})
    if (data, stage) not in parts:
        for gone in [k for k in parts if k[0] != data]:
            del parts[gone]
        rounds = st.session_state.setdefault(ROUNDS_KEY, {})
        with insights.memo_in_use(mem):
            checks = insights.check_rounds(opt, cache=rounds, only=insights_jobs.only_pairs(specs))
            if stage != "column" and not insights_jobs.settled(opt, checks):
                if not insights_jobs.ready(_plan(opt), mem):
                    return None
                checks = insights.check_rounds(opt, cache=rounds)
        parts[(data, stage)] = checks
    return parts[(data, stage)]


def checks_for(opt, stage):
    """This tab's checks when worked out, else the part `stage` needs from
    the fitted models, else None. Never fits anything."""
    hit = _cached(opt)
    if hit is not None and not hit.get("failed"):
        return hit["checks"]
    return _partial_checks(opt, stage)


def checks_here(opt):
    """The checks "trust" needs, fitted in this process: for a screen that
    must have them now when there are no workers (or they failed) — Make a
    round's Why fold, opened before Insights (answer-key refresh item 8).
    Every check when the verdict's Rounds come out short of plan."""
    mem, specs = memo(opt), _plan(opt)
    insights_jobs.compute_inline(opt, mem, insights_jobs.stage_specs(opt, specs, "trust"))
    checks = _partial_checks(opt, "trust")
    if checks is None:
        insights_jobs.compute_inline(opt, mem, specs)
        checks = _partial_checks(opt, "trust")
    return checks or ()


def save(opt):
    """Write what is worked out for this project's data beside the project:
    its fitted models, its Round checks, its predicted best and the next
    Round's expected improvement (C1), pruned to
    what this data uses. Once per state; a failed write only means the next
    open fits again."""
    mem = memo(opt)
    specs = _plan(opt)
    keys = set(insights.round_keys(opt).values())
    rounds = {k: v for k, v in st.session_state.get(ROUNDS_KEY, {}).items() if k in keys}
    best = st.session_state.get(BEST_KEY)
    best = best if best and best[0] == insights.setup_key(opt) else None
    gain = st.session_state.get(NEXT_GAIN_KEY)
    gain = gain if gain and gain[0] == bo_views.next_gain_key(opt) else None
    kept = {s.key: mem[s.key] for s in specs if s.key in mem}
    mark = (insights.data_key(opt), len(kept), tuple(sorted(rounds)), best is not None,
            None if gain is None else gain[0])
    if st.session_state.get(SAVED_KEY) == mark:
        return
    try:
        insights_cache.save(opt.project_name, insights_cache.Saved(kept, rounds, best, gain))
    except Exception:
        pass
    st.session_state[SAVED_KEY] = mark


def warm_best(opt):
    """The predicted best (a 0.7 s search in this process), as soon as the
    Overall score's models are in from the workers, so answer 2 need not
    wait for it. In a script run of this session, never beside ask()
    (review B2); once per setup_key. Nothing without workers."""
    if opt is None or opt.complete_count() < insights.MIN_ROWS or not insights_jobs.workers_on():
        return
    hit = st.session_state.get(BEST_KEY)
    if (hit and hit[0] == insights.setup_key(opt)) or not _in_memo(opt, "column"):
        return
    checks = _partial_checks(opt, "column")
    with insights.memo_in_use(memo(opt)):
        score = insights.fit(opt, insights.SCORE, list(range(len(opt.Y_history))))
    if checks is None or score.model is None:
        return
    _predicted_best(opt, {insights.SCORE: score}, checks, spinner=False)
    save(opt)


def warm_gain(opt):
    """§3.2's next-Round figure (ask()'s own first-pick search, about 1.5 s
    at 60 x 4) worked out after the page is drawn, so neither Insights nor
    Make a round's E3 waits on it (final review I-1). Once per _gain_key;
    only with no Round open; nothing while another run of this app is
    already working it out (it never waits for that run)."""
    if opt is None or opt.pending_batch or opt.complete_count() < insights.MIN_ROWS:
        return
    hit = st.session_state.get(GAIN_KEY)
    if hit and hit[0] == _gain_key(opt):
        return
    latest_gain(opt, wait=False)


def fits_here(opt):
    """Whether Make a round's predictions would be fitted in this process
    now: no workers (or they failed) and its models not all in."""
    job = insights_jobs.current_job(opt)
    if insights_jobs.workers_on() and not (job is not None and job.failed):
        return False
    specs = [s for s in _plan(opt) if s.name == insights.SCORE or s.stage == insights_jobs.NOW]
    return not insights_jobs.ready(specs, memo(opt))


def column_models(opt):
    """(models, factor) for Make a round's predicted columns (review I1):
    the current models that are in — the Overall score's at least; None
    when its fit failed — and insights.column_factor of the Rounds'
    Overall-score checks, so the range shown is the one that Round's check
    will judge.
    None while the workers are still fitting those: the column shows "…".
    Without workers they are fitted here, with no spinner of their own
    (the "…" is the one indicator)."""
    if waiting(opt, "column"):
        return None
    mem = memo(opt)
    checks = checks_for(opt, "column")
    if checks is None:
        insights_jobs.compute_inline(
            opt, mem, insights_jobs.stage_specs(opt, _plan(opt), "column"))
        checks = _partial_checks(opt, "column")
    if waiting(opt, "now"):
        rows = list(range(len(opt.Y_history)))
        ready = {s.name for s in _plan(opt) if s.stage == insights_jobs.NOW and s.key in mem}
        with insights.memo_in_use(mem):
            models = {n: insights.fit(opt, n, rows) for n, _, _ in insights.outputs(opt)
                      if n in ready}
    else:
        models = fitted_models(opt, working=None)
    return models, insights.column_factor(opt, checks)


def track_short(t):
    """track_text, short (concision round): the record and each warning in
    a few words; track_text is its tooltip."""
    m = getattr(t, "misses", None)
    line = wording.round_track_short(
        t.rounds or None, t.inside if t.record else None, t.total, rough=bool(t.rough),
        gap_text=None if t.gap is None else f"{abs(t.gap):.1f}", below=(t.gap or 0) < 0,
        see_insights=bool(t.see_insights), no_skill=t.no_skill is not None)
    if t.see_insights and m is not None and m.total:
        # Specific wording: the measurements' count against the 1 in 20.
        line = line.replace(wording.TRACK_VALUES_MISS,
                            wording.values_missed_short(m.misses, m.total, m.allowed))
    return line


def track_text(t):
    """The line for insights.round_track's answer: rough only when too few
    landed inside; a bias in Insights' own words; holding otherwise — each
    naming its Rounds (03 §5.8). A skill at or below the no-skill cut is
    said after it, or alone where the coverage part is not said (spec
    2026-09-30-bo-views §2.1, review I1)."""
    skill = ""
    if t.no_skill is not None:
        skill = wording.round_track_no_skill(skill_value(t.no_skill.value), t.no_skill.n,
                                             t.no_skill.rounds)
    if not t.record:
        return skill.strip()
    gap = None if t.gap is None else f"{abs(t.gap):.1f}"
    below = (t.gap or 0) < 0
    rounds = t.rounds or None
    if t.rough:
        line = wording.round_track_rough(t.inside, t.total, gap, below=below,
                                         see_insights=t.see_insights, rounds=rounds,
                                         allowed=insights.allowed_misses(t.total))
    elif gap is not None:
        line = wording.round_track_bias(t.inside, t.total, gap, below, t.see_insights,
                                        rounds=rounds)
    else:
        line = wording.round_track_holding(t.inside, t.total, t.see_insights, rounds=rounds)
    m = getattr(t, "misses", None)
    if t.see_insights and m is not None and m.total:
        # Specific wording: the measurements' misses counted, not "many more".
        counted = wording.track_measured_counted(m.misses, m.total, m.allowed)
        line = line.replace(wording._MEASURED_MORE, counted).replace(
            "; many more in the measured values", counted)
        line = wording.with_left_out(line, getattr(m, "no_forecast", 0))
    return line + skill


def _seen(opt, stages):
    return tuple(not waiting(opt, stage) for stage in stages)


def _in_memo(opt, stage):
    """Whether every model `stage` needs is in the memo — fitted, not merely
    no longer waited for (a failed pool stops the waiting too)."""
    return insights_jobs.ready(insights_jobs.stage_specs(opt, _plan(opt), stage), memo(opt))


def _catch_up(opt):
    """Work out here, from the workers' models, what the tab would otherwise
    work out when it is opened: the predicted best once the Overall score's
    models are in the memo, then every check once all of them are —
    conditioning only, about a second in all — and write it beside the
    project. Never a fit: when the workers fail, what is missing is fitted
    by the screen that shows it, behind its own indicator (Plan A review C
    I-1). Only with workers; in a script run of this session.

    The next Round's expected improvement (warm_gain, about 1.5 s at
    60 x 4) needs no display model: it is worked out at once, while the
    workers are still fitting the Overall score's models — time this
    process would only wait — so the predicted best and the first answers
    never queue behind it (answer-key refresh item 2: 5.0 s to the first
    answer right after launch at 60 x 4)."""
    if not insights_jobs.workers_on():
        return
    if _in_memo(opt, "column"):
        warm_best(opt)
    warm_gain(opt)
    if _in_memo(opt, "all") and _cached(opt) is None:
        _computed(opt, quiet=True)
        save(opt)


def _in_flight(opt):
    """Whether the workers are still fitting this project's data."""
    return start(opt) is not None


def keep_up(opt, stages=()):
    """app.py, after the page is drawn, on every run with a project open:
    catch up with the workers now, and while they are still fitting (or
    something is left to catch up), look again every POLL_SECS in a
    fragment of its own — which reruns only itself, so no other tab is
    disturbed — and rerun the page the moment one more part of the open
    screen (`stages`, the parts it draws) is in: each part is drawn once,
    final. Nothing without workers."""
    if opt is None or opt.complete_count() < insights.MIN_ROWS:
        return
    drawn = st.session_state.pop(SEEN_KEY, None)
    if not insights_jobs.workers_on():
        # The workers were given up on while the page was drawn with a part
        # still waiting for them: draw it again, fitted here behind the
        # screen's own indicator (Plan A review C I-1). Once: drawing()
        # marks nothing without workers.
        if drawn is not None and not all(drawn):
            app_rerun()
        return
    _catch_up(opt)
    now = _seen(opt, stages)
    st.session_state[SEEN_KEY] = now
    if drawn is not None and len(drawn) == len(now) and drawn != now:
        # A part landed while the page was being drawn (drawing() marked
        # what it was drawn from), or the workers failed and it is now
        # fitted here: one more run shows it instead of keeping the older
        # state until the next click.
        app_rerun()
    if _in_flight(opt):
        _poller(opt, tuple(stages))


def drawing(opt, stages=()):
    """app.py, before the page is drawn: which parts of the open screen
    (`stages`) are in, so keep_up can tell when one landed while the page
    was drawn — after Save results the last fits often land in that
    moment, and nothing else would rerun the page."""
    if opt is None or opt.complete_count() < insights.MIN_ROWS or not insights_jobs.workers_on():
        return
    st.session_state[SEEN_KEY] = _seen(opt, stages)


def _poll_step(opt, stages):
    """One look by the poller: catch up, then whether to rerun the page —
    when a part of the open screen landed, or when nothing is in flight any
    more, so the fragment (which only a full run stops calling) stops and
    no screen polls once everything is in (Plan A review C I-3)."""
    _catch_up(opt)
    return (not _in_flight(opt) or _seen(opt, stages) != st.session_state.get(SEEN_KEY)
            or _count_moved(opt, stages))


def _count_moved(opt, stages):
    """Whether Insights' progress line, drawn while filling in, is behind
    the Rounds the workers have returned (C1: it moves per Round returned,
    not only when a part of the screen lands; 01 F2)."""
    drawn = st.session_state.get(PROGRESS_SEEN_KEY)
    if drawn is None or not all(s in stages for s in _INSIGHTS_STAGES):
        return False
    if drawn[0] != insights.data_key(opt):
        return False
    return insights_jobs.rounds_done(opt, _plan(opt), memo(opt))[0] != drawn[1]


@st.fragment(run_every=POLL_SECS)
def _poller(opt, stages):
    # The app's own rerun, not the user's: a sticky flash stays (review I10).
    # Layout study: never while a Replay steps (a full run would stop it);
    # the next look, POLL_SECS on, reruns the page instead.
    if _poll_step(opt, stages) and not replaying():
        app_rerun()


def _say(opt, text):
    """Every line on the tab in the project's own words, as the rest of the
    app writes them: trials, not formulations, in a project of process
    settings alone."""
    return wording.for_project(opt, text)


# Ftue spec §2, review R-I8, M3: each detail section is its noun heading and
# one figures fold holding today's exact lines; the folds that were there
# keep their keys (EACH_KEY, PROGRESS_KEY) and now sit inside these.
PREDICTION_FIGURES_KEY = "insights_prediction_figures"
PROGRESS_FIGURES_KEY = "insights_progress_figures"
NEXT_REASONING_KEY = "insights_next_reasoning"
# Where the first screen's links go (R-I4, R-I14): Make a round's Why fold
# (ui_batch.WHY_KEY: ui_batch imports this module, so the key is named here
# and a test ties the two).
WHY_FOLD_KEY = "why_these"


def render(opt):
    """The Review (review four, owner 2026-10-02: "clean up the Review tab
    to cleanly (and visually) answer these 4 questions"): the header, the
    three cards (each with a "See chart ↓" link to its question), the one
    "How the app knows" link, then the four questions, always open — Q1
    "Can we trust the predictions?" beside Q2 "Are we improving?", then Q3
    "What drives each result?" and Q4 "What have we tried?" full width —
    each a small heading, one answer line opening with its card's word, one
    chart and its caption; Method and definitions, closed, at the foot.

    Drawn at once, before anything is fitted (review I15, R-I8): the
    header, the cards' places, the buttons and links, the headings, and the
    model-free charts (Q2, Q4). The rest fills as its facts are ready. Only
    while the tab is open (app.py): nothing here is worked out before.

    Review story 2: the cards are one status strip under the header line
    (the method link at its right), the charts right under it, and the
    Next card's content the closing section, What next?.

    Slim Review (owner, 2026-10-08; layout study 2026-10-09): strip → Are
    we improving? (one panel: its Round control over the score chart, What
    changed and the profile, all in one viewport) → What
    next? → Forecast check (its verdict line; the ordering chart in a closed
    fold) → What have we tried? (closed) → the profiler (closed; left out on
    a stub project) → Method and definitions."""
    # A formulation moved to Not scored counts for nothing here (review M-3).
    if len(opt.Y_history) == sum(map(opt.is_set_aside, range(len(opt.Y_history)))):
        st.caption(_say(opt, wording.INSIGHTS_NONE))
        _foot(opt)
        return
    n = opt.complete_count()
    _header_line(opt)
    if insights_jobs.stopped():
        st.caption(_say(opt, wording.INSIGHTS_IN_PROCESS))
    _land()
    if n < insights.MIN_ROWS:
        _gate(opt)
        return
    filling = waiting(opt, "all")
    places = _places(opt)
    if filling:
        _fill_in(opt, places)
        _how(opt, None)
        return
    st.session_state.pop(PROGRESS_SEEN_KEY, None)
    cold = _cached(opt) is None
    quiet = cold and _nothing_to_fit(opt)
    computed = _computed(opt, places.bar, quiet=quiet)
    if computed.get("failed"):
        places.bar.empty()
        facts = insights.status_facts(opt, ())
        with places.item["predictions"].container(gap=None):
            st.caption(_say(opt, _CARD_LABELS["predictions"]))
            st.error(_say(opt, wording.INSIGHTS_FAILED))
        for which in ("progress", "next"):
            _strip_item(opt, places, facts, which)
        _next_body(opt, places, facts)
        _answer(opt, places.q["progress"], facts, "progress",
                text=progress_answer_text(opt, facts))
        for which in _model_parts(places):
            with places.q[which].body:
                st.caption(_say(opt, wording.CHART_FAILED))
        _how(opt, None)
        return
    checks = computed["checks"]
    yard = insights.noise_yardstick(opt, checks)
    standing = _standing(opt, computed, yard, spinner=not cold or quiet)
    places.bar.empty()
    score = computed["fitted"].get(insights.SCORE)
    gain = shown_gain(opt, score, checks)
    with insights.memo_in_use(memo(opt)):
        facts = screen_facts(opt, checks, standing, gain,
                             withheld=claims_withheld_reason(opt, score, checks))
    # Review learned: worked out before the strip, which reads its verdicts.
    lf = learned_facts(opt, computed)
    _finish(opt, places, facts)
    _called_top(opt, places.called, lf)
    _learned_section(opt, places.q["predictions"], computed, facts)
    _drives_section(opt, places.q["drives"], computed, facts)
    _how(opt, computed)
    save(opt)
    # Review learned: the next suggestions, previewed last (a few seconds
    # once per setup), under a spinner in What next?.
    _next_preview(opt, places, computed, facts)


class _Places:
    """The places render draws before anything is worked out, in screen
    order (R-I8), for _fill_in and the rest of render to fill."""


def _model_parts(places):
    """The model sections drawn (the profiler's left out when hidden)."""
    return [w for w in ("predictions", "drives") if not getattr(places.q[w], "hidden", False)]


class _Slot:
    """One question's places: its answer line and its body."""

    def __init__(self, answer, body):
        self.answer, self.body = answer, body


# Review four: where each question sits (a link lands on its key). Q1 and
# Q2 keep the keys the links from Results and Make a round land on; Q4 the
# Explored link's.
DRIVES_SECTION = "insights_drives"
NEXT_SECTION = "insights_next"      # review story 2: What next?, the closing section
SECTION_KEYS = {"predictions": TRUST_SECTION, "progress": STAND_SECTION,
                "drives": DRIVES_SECTION, "tried": EXPLORED_PLACE, "next": NEXT_SECTION}
_QUESTIONS = {"predictions": wording.FORECAST_CHECK_TITLE, "progress": wording.Q2_TITLE,
              "drives": wording.Q3_TITLE, "tried": wording.Q4_TITLE, "next": wording.Q5_TITLE}
STRIP_KEY = "insights_strip"
SEE_CHART_KEY = "insights_see_chart"
Q_HEIGHT = 300          # Q1 and Q2's chart height, side by side (Streamlit fits legend and
                        # axis titles inside it)


STRIP_STYLE = """<style>
.st-key-insights_strip { padding: 0.45rem 1rem !important; }
.st-key-insights_strip button[kind="tertiary"] { min-height: 1.6rem; padding: 0; text-align: left;
  justify-content: flex-start; }
.st-key-insights_strip button[kind="tertiary"] p { text-align: left; font-size: 0.94rem; }
[data-testid="stLayoutWrapper"]:has(> .st-key-insights_strip_progress),
[data-testid="stLayoutWrapper"]:has(> .st-key-insights_strip_predictions),
[data-testid="stLayoutWrapper"]:has(> .st-key-insights_strip_next) {
  flex: 1 1 auto !important; min-width: 0; }
.st-key-insights_stop_callout { background: rgba(214, 158, 46, 0.13);
  border-left: 4px solid #b7791f; border-radius: 0.5rem; padding: 0.7rem 1rem !important; }
.st-key-insights_stop_callout [data-testid="stMarkdownContainer"] p { margin-bottom: 0.1rem; }
.st-key-insights_stop_head [data-testid="stMarkdownContainer"] p { font-size: 1.2rem; }
.st-key-insights_tasting_noise_row [data-testid="stSelectbox"] { min-width: 9.5rem; }
</style>"""


def _header_line(opt):
    """The header line — 'Review after Round 10 · 30 formulations', its
    full text in the "?" — with "How the app knows" at its right end
    (review story 2: a small link, off the way of the charts)."""
    with st.container(horizontal=True, horizontal_alignment="distribute",
                      vertical_alignment="center", gap="small"):
        st.caption(_say(opt, cards_header_text(opt)), width="content",
                   help=_say(opt, f"{wording.REVIEW_ABOUT}\n\n{header_detail(opt)}"))
        method_link(opt)


def _places(opt, gate=False):
    """Review story 2 (owner: "Should the plots be more central and higher
    up?"): one status strip — Progress · Predictions · Next, each a small
    label with its "?", then its bold word and one short line, a click
    landing on its section; the one primary button at the strip's right
    end — then the progress bar's place and the five sections, the
    model-free charts drawn at once, What next? last (its word, lines, the
    Set up secondary and the one link)."""
    p = _Places()
    p.item = {}
    st.html(STRIP_STYLE)       # review story 2: a compact strip (60–80 px)
    with st.container(border=True, key=STRIP_KEY, horizontal=True, vertical_alignment="center",
                      gap="small"):
        for which in _STATUS_ORDER:
            # Each item its own width plus an equal share of the room left
            # (STRIP_STYLE: flex 1 1 auto), so a line like "No new best
            # 87.8 → 90.5 of 100" keeps to one row at 1280 px.
            with st.container(key=f"{STRIP_KEY}_{which}", gap=None, width="content"):
                p.item[which] = st.empty()
                with p.item[which].container(gap=None):
                    _strip_item(opt, None, None, which)
        with st.container(key=f"{STRIP_KEY}_go", width="content"):
            p.go = st.empty()
            with p.go.container():
                _foot(opt)
    p.bar = st.empty()
    p.q = {}
    # Slim Review: Are we improving? (one panel for its three parts), What
    # next? (the stop signal), the Forecast check (the model's verdicts in
    # one line, the ordering chart in a closed fold), What have we tried?
    # and the profiler in closed folds, then Method and definitions.
    for which in SECTION_ORDER:
        p.q[which] = _section(opt, which)
    p.called = p.q["predictions"].called
    with p.q["next"].answer.container(gap="xsmall"):
        _next_body(opt, None, None)
    with p.q["next"].body:
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            with st.container(width="content"):
                p.change = st.empty()
            with st.container(width="content"):
                p.link = st.empty()
            with p.link.container():
                p.why = _why_link(opt)
        # Final demo (B5): the project's tasting noise, beside the stop.
        _noise_control(opt)
        p.preview = st.empty()
        _what_happened_next(opt)
    # The model-free charts, at once (they need no fit).
    _progress_body(opt, p.q["progress"])
    _tried_section(opt, p.q["tried"])
    if gate:
        for which in _model_parts(p):
            with p.q[which].body:
                st.caption(_say(opt, wording.Q_WAIT))
    return p


def _what_happened_next(opt):
    """The Round 6 sample, Round 6 recorded: the pilot's Rounds 7-10, at the
    end of What next? (below the stop and the Tasting noise), secondary.
    app.py opens the all-10 sample on the run the key asks for it."""
    if sample_projects.offers_what_next(opt):
        if st.button(wording.SEE_WHAT_HAPPENED_NEXT, key="sample_what_next"):
            st.session_state[sample_projects.OPEN_NEXT_KEY] = True
            st.rerun()
        st.caption(wording.SEE_WHAT_HAPPENED_NEXT_CAPTION)


def strip_line(opt, facts, which, card):
    """(line, warn) for a strip item: the first warning line when the card
    has one (warnings are never folded), else its short line — Progress
    'start → best of 100', Predictions its count without 'Overall score:',
    Next nothing (its word is the line; What next? holds the rest)."""
    for text, warn in card.lines:
        if warn and not (which == "next" and card.tone == "off"):
            return wording.strip_short(text), True
    if which == "progress":
        f = facts.best
        if f is None:
            return (card.lines[0][0] if card.lines else ""), False
        # Persona fixes: 'Rising · best 98.3 (+3.3)'; the start stays on the
        # chart ("Start: 91.7 (Round 1)").
        a = facts.improvement
        if facts.progress in ("rising", "within_noise") and a is not None \
                and a.rise is not None:
            rise, _ = rise_and_need(a)
            return _say(opt, wording.strip_progress_tail(one_decimal(f.score), rise)), False
        best = wording.score_of(one_decimal(f.score), opt.utility_ceiling())
        return _say(opt, wording.strip_progress_tail(best)), False
    if which == "predictions":
        return (wording.strip_short(card.lines[0][0]) if card.lines else ""), False
    return "", False


def strip_stop_label(opt, facts):
    """'**:orange[Stop]** 1.7 below top; 2.3 needed'."""
    need = tie_need(facts.yardstick)
    gap = float(opt.utility_ceiling()) - float(facts.best.score)
    return (f":orange[**{wording.STOP_WORD}**] "
            + _say(opt, wording.stop_strip_line(one_decimal(gap), f"{need:.1f}")))


def strip_label(card, line, warn):
    """A strip item's button: '**Flat** 88.4 → 96.4 of 100'; a red word or
    line ':red[**Check panel first**]'."""
    word = f":red[**{card.word}**]" if card.tone == "off" else f"**{card.word}**"
    if line:
        word += " " + (f":red[**{line}**]" if warn else line)
    return word


def _strip_item(opt, places, facts, which):
    """One strip item, in its place: the label with the card's "?" (the
    plain sentence, the Evidence line and the rule), then its word and
    line as one tertiary button landing on its section (the "See chart ↓"
    mechanism); a placeholder while its facts are worked out. The button
    is drawn once per run, with its facts (its key would repeat)."""
    card = card_for(opt, facts, which)
    lf = learned_cached(opt) if which == "predictions" else None
    if lf is not None and card is not None:
        # Review learned: the Model item reads the section's verdicts.
        card = learned_card(opt, lf, card, facts)
    slot = places.item[which].container(gap=None) if places is not None else None
    with slot if slot is not None else st.container(gap=None):
        label = _say(opt, wording.LEARNED_STRIP_LABEL if lf is not None
                     else _CARD_LABELS[which])
        if card is None:
            st.caption(label)
            st.caption(_say(opt, _PLACEHOLDERS[which]))
            return
        st.caption(label, help=card.help)
        line, warn = strip_line(opt, facts, which, card)
        question = _say(opt, _QUESTIONS[which])
        text = strip_label(card, line, warn)
        if which == "next" and card.tone != "off" and ceiling_stop(opt, facts):
            # Final demo (B1): the strip says Stop, with its numbers.
            text = strip_stop_label(opt, facts)
        st.button(text, type="tertiary", key=f"{SEE_CHART_KEY}_{which}",
                  help=_say(opt, wording.strip_jump_help(question)),
                  disabled=confirmation_open(), on_click=ask_tab,
                  args=(wording.TAB_INSIGHTS, SECTION_KEYS[which]))


def _next_body(opt, places, facts):
    """What next?'s word and lines (today's Next card): the big word with
    its "?", each line, warnings in red. The Set up secondary and the one
    link sit under it (places.change, places.link); the primary stays in
    the strip only."""
    card = card_for(opt, facts, "next")
    slot = places.q["next"].answer.container(gap="xsmall") if places is not None else None
    with slot if slot is not None else st.container(gap="xsmall"):
        if card is None:
            st.caption(_say(opt, _PLACEHOLDERS["next"]))
            return
        step = getattr(facts, "next", None) if facts is not None else None
        if card.tone != "off" and ceiling_stop(opt, facts):
            # Final demo (B1): the stop, the loudest thing here — a tinted
            # callout; the Repeat request a quiet line under it.
            _stop_callout(opt, facts)
            lines = card.lines + repeat_asks(opt, facts, card)
            # The Repeat request the card does not say itself (the all-10
            # sample's), in this same place, so it lines up with the one
            # the card says (the Round 6 sample's).
            quiet = quiet_repeat_text(opt, facts)
            if quiet and not any(quiet in t for t, _ in lines):
                lines.append((quiet, False))
            for text, warn in lines:
                if warn:
                    st.markdown(f":red[**{text}**]")
                else:
                    st.caption(text)
            return
        elif step is not None and step.kind == "close" and card.tone != "off":
            # Persona fixes: the stop signal is the headline.
            y = getattr(facts, "yardstick", None)
            tip = (wording.stop_help(wording.points_text(f"{y.value:.1f}"))
                   if y is not None and y.value is not None else "")
            st.markdown(stop_headline(opt, facts),
                        help="\n\n".join(t for t in (_say(opt, tip), card.help) if t))
        else:
            # Persona fixes 2: the same small style as the other answer
            # lines (it was a subheader, oversized).
            word = (f":red[**{card.word}.**]" if card.tone == "off"
                    else wording.answer_line(card.word, ""))
            st.markdown(word, help=card.help)
        for text, warn in card.lines + repeat_asks(opt, facts, card):
            st.markdown(f":red[**{text}**]" if warn else text)


def _stop_callout(opt, facts):
    """Final demo (B1), sceptic review: 'Stop: score within noise of 100',
    then '1.7 below the top; 2.3 needed to tell two results apart.' (its
    "?" the stop's numbers) and 'On target 3 of 4; Tangy 1.0 under, beyond
    noise.' (its "?" on target as rated and each miss against its noise
    band), in a warm tinted box."""
    with st.container(key="insights_stop_callout", gap="xsmall"):
        with st.container(key="insights_stop_head", gap=None):
            st.markdown(f"**{_say(opt, stop_head_text(opt))}**")
        st.markdown(_say(opt, stop_line_text(opt, facts)),
                    help=wording.brief(ceiling_tip(opt, facts)))
        reason = stop_reason_text(opt)
        if reason:
            best = opt.best_index()
            res = opt.results_history[best] if best is not None else {}
            st.markdown(reason, help=on_target_help_text(opt, res) or None)


def stop_head_text(opt):
    """Sceptic review: 'Stop: score within noise of 100' — the top score,
    from the project."""
    return wording.stop_callout_head(f"{float(opt.utility_ceiling()):g}")


def noise_half(opt, name, noise=None):
    """The ± band of `name` around its best value where 95 % of single
    tastings land: Z95 × its noise (its Repeats' SD, else derived from the
    project's tasting noise, learned.pair_noise) — within_noise_of_target's
    band; None without a noise."""
    noise = learned.pair_noise(opt) if noise is None else noise
    sd = noise.get(name)
    return insights.Z95 * float(sd) if sd else None


def beyond_noise(opt, offs):
    """[True / False / None] for ideal_gaps' `offs`: whether each gap is
    beyond its own band (noise_half, the "?"'s ±); None for a measurement
    with no noise (fix round 2: nothing is said about its noise)."""
    noise = learned.pair_noise(opt)
    flags = []
    for o, _amount, gap in offs:
        half = noise_half(opt, o['name'], noise)
        flags.append(None if half is None else abs(float(gap)) > half + 1e-9)
    return flags


def stop_line_text(opt, facts):
    """'1.7 below the top; 2.3 needed to tell two results apart.'"""
    need = tie_need(facts.yardstick)
    gap = float(opt.utility_ceiling()) - float(facts.best.score)
    return wording.stop_callout_line(one_decimal(gap), f"{need:.1f}")


def stop_reason_text(opt):
    """'On target 3 of 4; Tangy 1.0 under.' — ceiling_reason_text's facts
    in the callout's short form; '' with nothing to read."""
    best = opt.best_index()
    if best is None or best >= len(opt.results_history or []):
        return ""
    gaps = ideal_gaps(opt, opt.results_history[best] or {})
    if gaps is None:
        return ""
    on, n, offs, targets = gaps
    texts = [wording.spider_off(o['name'], amount, "short" if o.get('goal') != 'target'
                                else "under" if gap < 0 else "over") for o, amount, gap in offs]
    # Sceptic review: each off-target measurement against its own noise band.
    try:
        beyond = beyond_noise(opt, offs)
    except Exception:
        beyond = None
    return _say(opt, wording.stop_on_target(on, n, texts, targets, beyond))


NOISE_KEY = "insights_tasting_noise"


def _on_noise(opt):
    value = st.session_state.get(NOISE_KEY)
    try:
        insights.set_tasting_noise(opt, value)
    except Exception:
        pass


def _noise_control(opt):
    """Final demo (B5): the Overall score's tasting noise, in points (1 SD):
    a choice of TASTING_NOISE_CHOICES, the assumed 1 point the default;
    saved with the project. From YARDSTICK_DF Repeat df the measured value
    instead, read-only. Review-side only: the rise and tie thresholds, the
    stop, the derived measurement noise, too-close pairs, the band and the
    "?"s follow it; suggestions never read it."""
    try:
        yard = insights.noise_yardstick(opt)
    except Exception:
        return
    with st.container(horizontal=True, vertical_alignment="center", gap="small",
                      key="insights_tasting_noise_row"):
        if getattr(yard, "from_repeats", False) and yard.value is not None:
            st.caption(_say(opt, wording.tasting_noise_measured(f"{yard.value:.1f}", yard.df)),
                       help=_say(opt, wording.TASTING_NOISE_MEASURED_HELP), width="content")
            return
        st.markdown(_say(opt, wording.TASTING_NOISE_LABEL), width="content",
                    help=_say(opt, tasting_noise_help_text(opt, yard)))
        want = insights.assumed_noise(opt)
        choices = list(insights.TASTING_NOISE_CHOICES)
        if all(abs(want - c) > 1e-9 for c in choices):
            choices = sorted(choices + [want])
        if st.session_state.get(NOISE_KEY) != want:
            st.session_state[NOISE_KEY] = want
        st.selectbox(_say(opt, wording.TASTING_NOISE_LABEL), choices, key=NOISE_KEY,
                     format_func=lambda v: wording.tasting_noise_option(
                         v, insights.ASSUMED_NOISE_SD),
                     label_visibility="collapsed", width=200, on_change=_on_noise,
                     args=(opt,), disabled=confirmation_open())


def noise_unit_text(opt):
    """Sceptic review: what 1 point of the Overall score is in each scored
    measurement's own units, the assumption the Review's measurement noise
    is derived by (learned.derived_noise at 1 point): 'Here 1 point = 0.3
    per measurement, in its own units (Range 0–15).'; '' with none."""
    cells = []
    for o in learned.scored_measurements(opt):
        v = learned.derived_noise(opt, o['name'], 1.0)
        if not v:
            continue
        text = _num(v)
        exact = abs(float(text) - v) < 1e-9
        rng = f"{float(o.get('min_val', 0.0)):g}–{float(o.get('max_val', 10.0)):g}"
        # A unit like '/15' says the Range again; one like 'N' joins both.
        unit = str(o.get('unit') or "")
        if unit and not unit.startswith("/"):
            text, rng = join_unit(text, unit), join_unit(rng, unit)
        cells.append((o['name'], text if exact else f"≈{text}", rng))
    return wording.noise_per_measurement(cells)


def stop_noise_floor(opt, yard):
    """Sceptic review: the tasting noise (points) over which the stop shows
    — at_ceiling's rule, round(top − best, 2) < round(rise_threshold(σ, 1),
    2), solved for σ by bisection (the rule rises with σ). None where the
    stop does not rest on that rule alone: noise measured from Repeats, a
    formulation made more than once (a disagreeing Repeat replaces the
    stop), a Round open, fewer than MIN_ROWS scored, no Best so far."""
    if yard is None or getattr(yard, "from_repeats", False) or getattr(yard, "groups", 0):
        return None
    if opt.complete_count() < insights.MIN_ROWS:
        return None
    if getattr(opt, "pending_batch", None) and open_rows(opt):
        return None
    best = insights.best_facts(opt)
    if best is None:
        return None
    gap = round(float(opt.utility_ceiling()) - float(best.score), 2)

    def holds(sigma):
        return gap < round(insights.rise_threshold(sigma, 1), 2)
    lo, hi = 0.0, 1.0
    while not holds(hi):
        lo, hi = hi, hi * 2
        if hi > 1e6:
            return None
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if holds(mid) else (mid, hi)
    return hi


def stop_floor_text(floor):
    """The floor as the "?" says it (fix round 2): rounded DOWN to the
    hundredth, so it never reads above the true one; under 0.01 (Best so
    far at the top): None, 'at any noise'."""
    if floor is None:
        return None
    down = math.floor(floor * 100 + 1e-9) / 100
    return None if down < 0.01 else f"{down:.2f}"


def tasting_noise_help_text(opt, yard=None):
    """The Tasting noise "?" (sceptic review): its unit, what 1 point is per
    measurement here, and the noise over which the stop shows."""
    try:
        per = noise_unit_text(opt)
    except Exception:
        per = ""
    try:
        floor = stop_noise_floor(opt, yard)
    except Exception:
        floor = None
    stop = wording.stop_holds_above(stop_floor_text(floor)) if floor is not None else ""
    return wording.tasting_noise_help(per, stop, f"{float(opt.utility_ceiling()):g}")


def ceiling_tip(opt, facts):
    """The stop signal's "?" at the top score (round-2 fixes, P1 #5): the
    gap to the top and the threshold, in numbers; the score a formulation
    exactly on every target would get with tasting noise; what can still
    move; the limits that bind (said here only)."""
    need = tie_need(facts.yardstick)
    try:
        limits = binding_texts(opt, getattr(facts.next, "bindings", None) or [])
    except Exception:
        limits = []
    top = float(opt.utility_ceiling())
    best = float(facts.best.score)
    try:
        perfect = perfect_noisy_score(opt)
    except Exception:
        perfect = None
    assumed = getattr(facts.yardstick, "assumed", True)
    return _say(opt, wording.ceiling_help(
        one_decimal(best), f"{top:g}", one_decimal(top - best), f"{need:.1f}",
        None if perfect is None else one_decimal(perfect), assumed, limits,
        sd_text=noise_sd_text(opt)))


def ceiling_answer(opt, facts):
    """Specific wording: the stop's answer line with the gap to the top and
    the rise one new result needs, the "?"'s own numbers: '**Within noise
    of the top: 1.7 below, 2.3 needed.** Stop, or change the target.'"""
    try:
        need = tie_need(facts.yardstick)
        gap = float(opt.utility_ceiling()) - float(facts.best.score)
        head = wording.ceiling_head(one_decimal(gap), f"{need:.1f}")
    except Exception:
        head = wording.CEILING_HEAD
    return wording.answer_line(head, wording.CEILING_TAIL)


def _ideal(obj):
    """A measurement's best value: its target, else the end of its Range
    the score rewards."""
    if obj.get('goal') == 'target':
        return float(obj.get('target', (float(obj['max_val']) + float(obj['min_val'])) / 2))
    return float(obj['max_val'] if obj.get('goal') == 'max' else obj['min_val'])


def perfect_noisy_score(opt):
    """Round-2 fixes (P1 #5): the Overall score a formulation exactly at
    every measurement's best would get on average, each measurement tasted
    with its noise (learned.pair_noise: the Repeats', else the assumed 1
    point's share): 100 − Σ points-per-unit × σ × √(2/π) for targets. Noise
    only pulls a score on its target down. None without a scored
    measurement."""
    objs = list(getattr(opt, "objectives", None) or ())
    if not objs:
        return None
    noise = learned.pair_noise(opt)
    per = {}
    for o in objs:
        v = _ideal(o)
        per[o['name']] = insights.Prediction(v, v, v, v, noise.get(o['name']) or 0.0)
    return insights.expected_score(opt, per)


def ceiling_reason_text(opt):
    """The concrete line under the stop (round-2 fixes, P1 #5): how Best so
    far's measurements sit against their targets — 'On target: 3 of 4.
    Tangy 1.0 under target.' — largest miss first; '' when Best so far has
    no results to read."""
    best = opt.best_index()
    if best is None or best >= len(opt.results_history or []):
        return ""
    gaps = ideal_gaps(opt, opt.results_history[best] or {})
    if gaps is None:
        return ""
    on, n, offs, targets = gaps
    texts = [wording.off_target(o['name'], amount, gap < 0) if o.get('goal') == 'target'
             else wording.off_best(o['name'], amount) for o, amount, gap in offs]
    return _say(opt, wording.ceiling_reason(on, n, texts, targets))


def within_noise_of_target(opt, res):
    """Final demo (A2): (within, n, half-width text) — how many of the
    measured values in `res` lie within 95 % of single tastings of their
    best value (± Z95 × each measurement's noise: its Repeats' SD, else
    derived from the project's tasting noise, learned.pair_noise); the
    half-width as text when every measurement shares it, else '' . None
    with nothing measured or no noise known."""
    objs = [o for o in opt.objectives if res.get(o['name']) is not None]
    if not objs:
        return None
    noise = learned.pair_noise(opt)
    halves, within = [], 0
    for o in objs:
        sd = noise.get(o['name'])
        if not sd:
            return None
        half = insights.Z95 * float(sd)
        halves.append(half)
        within += abs(float(res[o['name']]) - _ideal(o)) <= half + 1e-9
    texts = {f"{h:.1f}" for h in halves}
    return within, len(objs), (texts.pop() if len(texts) == 1 else "")


def on_target_help_text(opt, res):
    """Final demo (A2): the "?" line on "on target": as rated (the rating
    equals the target, ideal_gaps' rule), then within noise of target."""
    try:
        res = {k: v for k, v in (res or {}).items() if v is not None}
        gaps = ideal_gaps(opt, res)
        if gaps is None:
            return ""
        on, n, offs, targets = gaps
        w = within_noise_of_target(opt, res)
        # Sceptic review: each off-target one named with its band, beyond
        # it or within it; one with no noise is named in neither.
        noise = learned.pair_noise(opt)
        beyond, inside = [], []
        for o, amount, gap in offs:
            h = noise_half(opt, o['name'], noise)
            if h is None:
                continue
            kind = ("short" if o.get('goal') != 'target' else "under" if gap < 0 else "over")
            item = (wording.spider_off(o['name'], amount, kind), f"{h:.1f}")
            (beyond if abs(float(gap)) > h + 1e-9 else inside).append(item)
    except Exception:
        return ""
    within, half = (w[0], w[2]) if w is not None else (None, "")
    return _say(opt, wording.on_target_help(on, n, within, half or None, targets,
                                            beyond, inside))


def on_ideal(opt, name, gap):
    """Whether a measured value `gap` from its best value (its target, or
    its Range's better end) reads as on it: under half a step at the
    decimals the measurement is shown with. The one rule the Stop line and
    the sensory profile both count by."""
    return abs(gap) < ideal_tol(opt, name)


def ideal_tol(opt, name):
    """on_ideal's tolerance in the measurement's units."""
    return 0.5 * 10 ** -max(1, decimals(opt, name))


def ideal_gaps(opt, res):
    """(on, n, offs, targets) for one formulation's results `res`: how many
    measurements sit on their best value (on_ideal), out of the n measured;
    offs [(measurement, amount text, gap)] largest in score points first;
    targets: every one measured has a target. None with nothing measured."""
    objs = [o for o in opt.objectives if res.get(o['name']) is not None]
    if not objs:
        return None
    on, offs = 0, []
    for o in objs:
        d = decimals(opt, o['name'])
        gap = float(res[o['name']]) - _ideal(o)
        if on_ideal(opt, o['name'], gap):
            on += 1
            continue
        offs.append((abs(gap) * learned.points_per_unit(o), (o, f"{abs(gap):.{max(1, d)}f}", gap)))
    offs.sort(key=lambda t: -t[0])
    targets = all(o.get('goal') == 'target' for o in objs)
    return on, len(objs), [t for _, t in offs], targets


def stop_headline(opt, facts):
    """'**Little left to gain inside your limits.** Widen a limit or stop.'
    (no limit binds: 'Add an ingredient or stop.')."""
    b = getattr(facts.next, "binding", None)
    return _say(opt, wording.answer_line(wording.STOP_HEAD,
                                         wording.STOP_WIDEN if b is not None else wording.STOP_ADD))


def repeat_asks(opt, facts, next_card):
    """Review learned 2: the Model card's Repeat request ('Repeat
    Formulation 18 to check noise', 'Repeats vary a lot'), once no longer
    under What has the model learned?, joins What next? — unless What
    next? asks it already (one action, one card)."""
    if facts is None:
        return []
    try:
        pred = card_for(opt, facts, "predictions")
    except Exception:
        return []
    if pred is None:
        return []
    suffix = wording.card_noise_repeat(None).split("Repeat", 1)[1]
    have = [t for t, _ in next_card.lines]
    return [(t, w) for t, w in pred.lines
            if (t == wording.CARD_REPEATS_VARY or t.endswith(suffix))
            and not any(t.split(suffix)[0] in h for h in have)]


def _section(opt, which):
    """A question: its heading (one style for the four), its answer line's
    place and its body, in a keyed place a link lands on. Slim Review
    (owner, 2026-10-08): the Forecast check heads the model's section (its
    lines, then the verdict line, then the ordering chart in a closed
    fold); What have we tried? and What drives each result? are closed
    folds (their answer first inside), worked out only while open; the
    profiler is left out where most rows have too little room to move."""
    if which == "drives" and profiler_hidden(opt):
        slot = _Slot(None, None)
        slot.open, slot.hidden = False, True
        return slot
    with st.container(key=SECTION_KEYS[which], gap="small"):
        if which in FOLDS:
            with st.expander(_say(opt, _QUESTIONS[which]), expanded=False,
                             key=FOLDS[which], on_change="rerun") as fold:
                answer = st.empty()
                body = st.container(gap="small")
            slot = _Slot(answer, body)
            # Worked out only while open (the profiler is the slowest part).
            slot.open = bool(getattr(fold, "open", True))
            return slot
        st.markdown(f"#### {_say(opt, _QUESTIONS[which])}")
        called = st.empty() if which == "predictions" else None
        answer = st.empty()
        body = st.container(gap="small")
    slot = _Slot(answer, body)
    slot.called = called
    return slot


DRIVES_FOLD_KEY = "insights_drives_fold"
TRIED_FOLD_KEY = "insights_tried_fold"
FOLDS = {"tried": TRIED_FOLD_KEY, "drives": DRIVES_FOLD_KEY}


def _answer(opt, slot, facts, which, text=None, help=None):
    """A question's answer line: its card's bold word, then `text` (the
    card's first line when None); the card's evidence and full sentence in
    its "?"."""
    card = card_for(opt, facts, which)
    if card is None:
        return
    if text is None:
        text = card.lines[0][0] if card.lines else ""
    plain = re.sub(r":\w+\[(.*?)\]", r"\1", card.word).replace("**", "")
    text = re.sub(r":\w+\[(.*?)\]", r"\1", text).replace("**", "")
    line = wording.answer_line(plain, text)
    if card.tone == "off":
        line = f":red[{line}]"
    with slot.answer.container(gap="xsmall"):
        st.markdown(line, help=help if help is not None else card.help)
        # Review story 2: the card's further lines (a Repeat request, a
        # lacking measurement), warnings in red, under the answer.
        for extra, warn in card.lines[1:]:
            st.markdown(f":red[**{extra}**]" if warn else extra)


def _land():
    """A link that lands on Method and definitions opens it (review polish:
    the method link), and the Explored link opens What have we tried?.
    Only on arrival (ui_helpers.SCROLL_KEY, drained by app.py after the tab
    is drawn), so a fold the reader closed stays closed."""
    if st.session_state.get(SCROLL_KEY) == HOW_PLACE:
        st.session_state[HOW_KEY] = True
    if st.session_state.get(SCROLL_KEY) == EXPLORED_PLACE:
        # Slim Review: the Explored link lands on What have we tried?, open.
        st.session_state[TRIED_FOLD_KEY] = True


def _gate(opt):
    """Fewer than MIN_ROWS scored (review M8, R-B1 row 3): the cards with
    no model behind them, the button, the model-free questions (Q2, Q4)
    and Method and definitions."""
    with insights.memo_in_use(memo(opt)):
        facts = insights.status_facts(opt, ())
    places = _places(opt, gate=True)
    for which in _STATUS_ORDER:
        _strip_item(opt, places, facts, which)
    _next_body(opt, places, facts)
    _answer(opt, places.q["progress"], facts, "progress", text=progress_answer_text(opt, facts))
    _how(opt, None)


def _finish(opt, places, facts):
    """The cards, the Set up secondary, the one link when the Why link was
    not drawn, and Q2's answer: everything that waits for the predicted
    best and the expected improvement."""
    for which in _STATUS_ORDER:
        _strip_item(opt, places, facts, which)
    _next_body(opt, places, facts)
    change = None
    if facts.next is not None:
        checks = checks_for(opt, "trust")
        rounds = insights.headline(opt, checks).rounds if checks else None
        _, change = next_step_text(opt, facts.next, rounds, amounts_shown(facts))
        if ceiling_stop(opt, facts):
            change = None       # review critique: no widening advice at the top score
    with places.change.container():
        _setup_button(change)
    if ceiling_stop(opt, facts) and getattr(places, "go", None) is not None:
        # Review critique, follow-up: at the top score the strip's one
        # primary leads to Set up (a target or a Share of score), not to
        # another Round.
        with places.go.container():
            _ceiling_foot(opt)
    if not places.why:
        # R-I4's one link, by priority: Why (drawn with the places), then
        # the Repeat, then the Explored link (ftue fix C).
        with places.link.container():
            # Final demo (B1): at the stop the Repeat is a quiet line under
            # the callout (_next_body), not here beside the Set up button.
            if not ceiling_stop(opt, facts) and not _repeat_link(opt, facts):
                explored_link(opt, facts.next)
    _answer(opt, places.q["progress"], facts, "progress", text=progress_answer_text(opt, facts))


def _standing(opt, computed, yard, spinner=True):
    """Progress's Standing (the predicted best against Best so far), worked
    out without drawing; None when it could not be."""
    fitted = computed["fitted"]
    score = fitted.get(insights.SCORE)
    if score is None or score.model is None:
        return None
    best = _predicted_best(opt, fitted, computed["checks"],
                           spinner=spinner and not computed.get("partial"))
    if isinstance(best, str) or best is _FAILED:
        return None
    return standing_for(opt, best, computed["checks"], yard)


def _fill_in(opt, places):
    """The tab while the workers are still fitting (spec 2026-09-30 §10.3,
    review I15, ftue R-I8): one progress line with a count, the model
    questions' placeholders, then the cards once the checks and the
    current Overall-score model are in. keep_up's poller reruns the page as
    parts land."""
    specs, mem = _plan(opt), memo(opt)
    done, total = insights_jobs.rounds_done(opt, specs, mem)
    fraction = sum(s.key in mem for s in specs) / max(1, len(specs))
    places.bar.progress(fraction, text=_say(opt, wording.insights_progress(done, total)))
    # The poller reruns the page when this count moves (C1; 01 F2).
    st.session_state[PROGRESS_SEEN_KEY] = (insights.data_key(opt), done)
    for which in _model_parts(places):
        with places.q[which].body:
            st.caption(_say(opt, wording.CHECKING_PREDICTIONS))
    checks = None if waiting(opt, "trust") else _partial_checks(opt, "trust")
    if checks is None or waiting(opt, "stand"):
        return
    yard = insights.noise_yardstick(opt, checks)
    with insights.memo_in_use(mem):
        score = insights.fit(opt, insights.SCORE, list(range(len(opt.Y_history))))
    computed = {"fitted": {insights.SCORE: score}, "checks": checks, "partial": True}
    standing = _standing(opt, computed, yard)
    gain = shown_gain(opt, score, checks)
    with insights.memo_in_use(mem):
        facts = screen_facts(opt, checks, standing, gain,
                             withheld=claims_withheld_reason(opt, score, checks), partial=True)
    _finish(opt, places, facts)


def _foot(opt):
    """The foot Results uses, in its words and by its rule, with a key of its
    own (spec 2026-09-30 §2.4, review M13): back to the open Round, else
    the next one. Needs no model: drawn before anything is worked out,
    under the three status lines' places (ftue §1: Next's primary)."""
    if opt.pending_batch:
        label = wording.back_to_batch_label(opt.pending_batch_no, len(open_rows(opt)))
    else:
        label = wording.START_NEXT_BATCH
    ready, _ = readiness(opt)
    lit = ready and not confirmation_open()
    if st.button(label, type="primary" if lit else "secondary", disabled=not lit,
                 key="insights_foot") and lit:
        go_to_tab(TAB_BATCH)


def _setup_button(change):
    """The secondary Set up button "What to do next" asked for (its label),
    under the foot; nothing for None. It lands on Set up's callout above
    the ingredient grid, which names the row or limit and its value (0.8.0
    e2e fixes A7, review I8; 01 O3: it landed at the top with nothing
    pointing at the row)."""
    # Review polish: the Next card already names the change; the button,
    # at most 5 words, keeps the full label in its "?".
    if change and st.button(wording.SETUP_BUTTON_SHORT, key="insights_change_setup",
                            help=change, disabled=confirmation_open()):
        st.session_state[SETUP_CALLOUT_KEY] = change
        go_to_tab(TAB_SETUP, section=SETUP_CALLOUT_SLOT)


# ------------------------------------------------------------------ #
#  Insights for a first-time user (spec 2026-10-01 rev. 2), W1: the three
#  status lines (review R-B1, R-I5, R-I9, R-I10, M1, M7, M10) and the one
#  tertiary link (R-I3, R-I4, R-I14).
# ------------------------------------------------------------------ #

_PLACEHOLDERS = {"predictions": wording.CHECKING_PREDICTIONS,
                 "progress": wording.CHECKING_PROGRESS, "next": wording.PREPARING_NEXT_STEP}
# Review story: Progress first (the payoff), then Predictions; Next last.
_STATUS_ORDER = ("progress", "predictions", "next")
SECTION_ORDER = ("progress", "next", "predictions", "tried", "drives")


def _wide_clause_names(names):
    """The wide clause's names: measurements as said, the Overall score as
    "the Overall score"."""
    return [f"the {wording.OVERALL_SCORE_COLUMN}" if n == insights.SCORE else n for n in names]


# Fix round B (five-second re-test, Line C Q4): the wide clause names its
# cause when every wide output's current model was widened by at least this
# calibration factor (its cross-validated errors this much larger than its
# own uncertainty: results vary more than the model explains), and the
# output it is largest for when that factor is at least MOST_FACTOR times
# the next one's.
CAUSE_FACTOR = 1.5
MOST_FACTOR = 1.5


def wide_cause(names, factors):
    """None: no cause is said (an output under CAUSE_FACTOR, or a factor
    not known yet); else the output the cause is clearly largest for, ''
    when none is (or there is one output)."""
    got = [factors.get(n) for n in names]
    if not names or any(f is None or f < CAUSE_FACTOR for f in got):
        return None
    if len(names) < 2:
        return ""
    ranked = sorted(zip(got, names), reverse=True)
    return ranked[0][1] if ranked[0][0] >= MOST_FACTOR * ranked[1][0] else ""


def current_factors(opt, names):
    """{output: its current model's calibration factor} for `names`, from
    the fitted models kept for this data (the memo), never fitted here; an
    output whose model is not in yet is left out."""
    mem, out = memo(opt), {}
    for spec in _plan(opt):
        if spec.stage == insights_jobs.NOW and spec.name in names:
            hit = mem.get(spec.key)
            if hit is not None and hit.get("widen") is not None:
                out[spec.name] = float(hit["widen"])
    return out


def little_more(standing, gain, yard, withheld, settings=()):
    """Fix round B (walkthrough ruling): the "little more" clause's items
    (wording.binding_plain, empty when nothing binds), or None when it is
    not said: it needs the predicted best and the expected improvement
    both claimed (`withheld` None) and both at or under the noise
    threshold. `settings`: the names that are process settings (said by
    value, not amount)."""
    if (standing is None or withheld or gain is None or yard is None or yard.value is None
            or getattr(standing, "higher", True) or gain > yard.value):
        return None
    out = _Little(wording.binding_plain(b.kind, b.name, b.text, values=b.name in settings)
                  for b in (getattr(standing, "binding", None) or ()))
    # Specific wording: the gain and the threshold it fell under.
    out.gain_text, out.noise_text = f"{gain:.1f}", f"{yard.value:.1f}"
    return out


class _Little(tuple):
    """little_more's items, carrying the numbers the clause says."""
    gain_text = noise_text = None


def screen_facts(opt, checks, standing=None, gain=None, withheld=None, partial=False):
    """insights.status_facts, with what the first screen's sentences need
    beyond it (fix round B), set on the facts: `wide_factors` (the wide
    outputs' calibration factors, current_factors) and `little`
    (little_more). `partial` (the workers are still fitting): the
    Predictions line waits (facts.pending) while a wide output's factor is
    not in yet, so it is drawn once, final (R-I8)."""
    facts = insights.status_facts(opt, checks, standing, gain)
    if facts is None:
        return None
    facts.checks = checks       # review cards: the Predictions sparkline
    wide = next((c[1] for c in facts.clauses if c[0] == "wide"), None)
    if wide:
        facts.wide_factors = current_factors(opt, wide)
        if partial and len(facts.wide_factors) < len(wide):
            facts.pending = tuple(facts.pending) + ("predictions",)
    settings = [n for n in insights.path_choices(opt) if insights.is_setting(opt, n)]
    facts.little = little_more(standing, gain, facts.yardstick, withheld, settings)
    return facts


def prediction_sentence(opt, facts):
    """The Predictions line's sentence for facts.prediction_row (R-B1 rows
    1-9), its clauses in the ruling's order. The Overall score's steady bias
    joins rows 7-8's sentence (", and scores have come in …") and row 2's,
    and is a clause on the others. Rows 7-8 give the skill with its
    meaning; a wide clause names its cause when wide_cause finds one, and
    the sentence then ends with what to check (fix round B)."""
    row = facts.prediction_row
    bias = next((c[1] for c in facts.clauses if c[0] == "bias"), None)
    gap = None if bias is None else (f"{abs(bias):.1f}", bias < 0)
    fact = getattr(facts, "skill", None)
    skill = None if fact is None else skill_value(fact.value)
    tail, check = [], False
    for c in facts.clauses:
        if c[0] == "wide":
            most = wide_cause(c[1], getattr(facts, "wide_factors", None) or {})
            check = most is not None
            names = _wide_clause_names(c[1])
            tail.append(wording.clause_wide(
                names, cause=check,
                most=_wide_clause_names([most])[0] if most else None))
        elif c[0] == "over_wide":
            tail.append(wording.CLAUSE_OVER_WIDE)
        elif c[0] == "bias" and row not in ("no_skill", "unclear"):
            tail.append(wording.clause_bias(*gap))
        elif c[0] == "guard":
            tail.append(wording.clause_guard(settings_only=not opt.has_ingredients()))
    if row == "off_repeat":
        return wording.PRED_OFF_REPEAT
    if row == "gate":
        return wording.pred_gate(facts.scored, insights.MIN_ROWS)
    then = scatter_tail(opt, facts) if check else ""
    if row == "too_few":
        return wording.pred_too_few(facts.checked, insights.SKILL_MIN, tail) + then
    if row == "wide":
        return wording.pred_wide(tail)
    if row == "no_skill":
        return wording.pred_no_skill(gap, tail, skill_text=skill) + then
    if row == "unclear":
        return wording.pred_unclear(gap, tail, skill_text=skill) + then
    # Review four, ruling 1: the counts are the Overall score's; the
    # measurements' own record follows, and where their misses fell.
    inside, total, rounds = score_counts(facts)
    allowed = insights.allowed_misses(total)
    d = getattr(facts, "diagnosis", None)
    # The measurements' record when they miss more than expected (their
    # counts are in the evidence line either way), then the check advice,
    # then where their misses fell.
    said = measures_text(facts) if getattr(facts, "measures_miss", False) else ""
    where = f" {diag_plain_text(d)}" if d is not None else ""
    if row == "off_coverage":
        return _on_score(facts, wording.pred_off_coverage(
            rounds, total - inside, total, allowed,
            None if gap is None else wording.bias_fact(*gap))) + said + then + where
    if row == "caution":
        return (_on_score(facts, wording.pred_caution(rounds, total - inside, total, allowed,
                                                      tail)) + said + then + where)
    return (_on_score(facts, wording.pred_reliable(rounds, inside, total, allowed, tail))
            + said + then + where)


def score_counts(facts):
    """(inside, total, Rounds) of the Overall score's Round-ahead record,
    the one the Predictions word reads (review four, ruling 1)."""
    h = getattr(facts, "headline", None)
    r = getattr(h, "score", None)
    if r is None or not r.total:
        # No Overall-score record yet: the headline's own counts.
        return ((h.inside, h.total, list(h.rounds or [])) if h is not None
                else (0, 0, []))
    return r.inside, r.total, list(r.rounds or [])


def _on_score(facts, text):
    """wording.on_score when the counts are the Overall score's record;
    unchanged when score_counts fell back to the headline's."""
    r = getattr(getattr(facts, "headline", None), "score", None)
    return wording.on_score(text) if r is not None and r.total else text


def measures_text(facts):
    """The measurements' own record, for the full sentence: '' without one."""
    h = getattr(facts, "headline", None)
    if h is None or not h.total:
        return ""
    return wording.measures_sentence(h.inside, h.total, insights.allowed_misses(h.total),
                                     miss=getattr(facts, "measures_miss", False))


def one_round(opt):
    """Whether every formulation with all its results is from one Round
    (ftue §4: "It came from the latest Round" and "One Round so far" say
    nothing else). Counted over every scored formulation, not only Best so
    far's candidates: a project whose earlier Rounds are scored without a
    measurement has had more than one Round."""
    rounds = {opt.batch_history[i] for i in range(len(opt.Y_history)) if opt.fully_measured(i)}
    return len(rounds) == 1


def _one_round_cold(opt, r):
    """Whether Round `r` was laid out to cover the allowed amounts (the cold
    start), not chosen by the model."""
    try:
        return r is not None and bo_views.generation(opt, r).regime == "cold"
    except Exception:
        return False


def progress_sentence(opt, facts):
    """The Progress line's sentence (R-B1, R-I9, M1, M10): the status reason
    and Best so far to two decimals of the score's ceiling; an equal-score
    tie named; the luck clause only when facts.luck (Reliable)."""
    f = facts.best
    if f is None:
        return wording.NO_COMPLETE_FORMULATION
    ceiling = opt.utility_ceiling()
    score = f"{f.score:.1f}"     # round-2 fixes (P3 #19): 1 decimal on the plain layer
    tie = None if facts.tie is None else int(opt.formulation_ids[facts.tie])
    # Fix round B (walkthrough ruling): a runner-up within the noise
    # threshold is named too ("tied with Formulation 18"); one at an equal
    # score keeps its own sentence.
    near = None if tie is not None else near_tie(opt, facts.yardstick)
    little = getattr(facts, "little", None)
    a = facts.improvement
    if facts.progress in ("rising", "within_noise"):
        rise, need = rise_and_need(a)
        build = (wording.progress_rising if facts.progress == "rising"
                 else wording.progress_within_noise)
        return build(f.round, rise, score, ceiling, f.formulation, need, near_no=near)
    if facts.progress == "flat":
        luck = None if facts.luck_repeat is None else _whole(facts.luck_repeat)
        return wording.progress_flat(a.rounds, score, ceiling, f.formulation, f.round,
                                     luck_text=luck, tie_no=tie, near_no=near, little=little)
    order = insights.scored_rounds(opt)
    if facts.progress == "no_new_best":
        return wording.progress_no_new_best(order[-1][0], score, ceiling, f.formulation,
                                            f.round, tie_no=tie, near_no=near, little=little)
    one = one_round(opt)
    return wording.progress_too_early(score, ceiling, f.formulation, f.round, one_round=one,
                                      values=not opt.has_ingredients(),
                                      cold=one and _one_round_cold(opt, f.round), tie_no=tie,
                                      near_no=near,
                                      without=None if one else scored_without(opt, f.round))


def rise_and_need(answer):
    """The rise and what it needed (Improvement.threshold; fix A follow-up),
    as the screen prints them: one decimal, unless the rise would read 0.0
    or the same as its threshold; then hundredths, so the words never
    contradict the numbers (Task 6 review M-1)."""
    threshold = answer.threshold if answer.threshold is not None else (
        insights.rise_threshold(insights.ASSUMED_NOISE_SD if answer.sd is None else answer.sd,
                                max(1, answer.q),
                                answer.df or None))
    d = 2 if f"{answer.rise:.1f}" in ("0.0", f"{threshold:.1f}") else 1
    return f"{answer.rise:.{d}f}", f"{threshold:.{d}f}"


def rise_factor_text(q, df=None):
    """rise_factor's words for the max-of-q factor (z, or t on `df`)."""
    return wording.rise_factor(f"{insights.rise_quantile(max(1, q), df or None):.2f}",
                               max(1, q), df or None)


def tie_need(yard):
    """What a lead must beat to be more than a tie: rise_threshold(yard, 1),
    the rule Rising uses for one new result (fix A follow-up, ruling on
    ties), so a tie and Rising never disagree. None without a yardstick."""
    if yard is None or yard.value is None:
        return None
    return insights.rise_threshold(yard, 1)


def near_tie(opt, yard):
    """Best so far's runner-up within the noise threshold (tie_text's rule:
    insights.tie_partner, a Repeat never one; the lead to the hundredth
    under the threshold), as its Formulation number; None without one."""
    top = opt.best_index()
    other = None if top is None else insights.tie_partner(opt, top)
    need = tie_need(yard)
    if other is None or need is None:
        return None
    lead = float(opt.Y_history[top]) - float(opt.Y_history[other])
    if round(lead, 2) >= round(need, 2):
        return None
    return int(opt.formulation_ids[other])


def scored_without(opt, best_round):
    """Fix round B (five-second re-test, Upgrade check Q2): when no earlier
    formulation is compared with Best so far because every scored one
    outside its Round lacks a measurement (FoodOptimizer.best_candidates),
    (their Rounds, the measurements they lack, said); None otherwise."""
    candidates = set(opt.best_candidates())
    if any(opt.batch_history[i] != best_round for i in candidates):
        return None
    left = [i for i in range(len(opt.Y_history))
            if opt.fully_measured(i) and not opt.is_set_aside(i) and i not in candidates
            and opt.batch_history[i] != best_round and opt.batch_history[i] is not None]
    if not left:
        return None
    rounds = sorted({int(opt.batch_history[i]) for i in left})
    names = []
    for i in left:
        names += [n for n in opt.not_expected(i) if n not in names]
    return (rounds, number_list(names)) if names else None


def widen_reason_text(opt, b):
    """Persona fixes 2: the widen advice with its reason — 'The predicted
    best sits at Pea protein's Lowest (20 g). To test below it, widen
    the limit.' ("Your best" when Best so far sits there too); None for a
    limit that is not a row's Lowest or Highest."""
    if b is None or b.kind not in ("lowest", "highest"):
        return None
    var = opt._by_name().get(b.name)
    if var is None or var.get('type') != 'continuous':
        return None
    lo, hi = var['bounds']
    value = float(lo if b.kind == "lowest" else hi)
    unit = opt.unit_of(b.name)
    amount = (fmt_setting(value, unit) if insights.is_setting(opt, b.name)
              else join_unit(f"{value:.0f}" if value == round(value) else _num(value), unit))
    best = opt._best_recipe_index()
    ours = False
    if best is not None:
        ours = any(x.kind == b.kind and x.name == b.name
                   for x in insights.binding_limits(opt, insights.start_recipe(opt, best)))
    return _say(opt, wording.widen_reason(b.kind, b.name, amount, predicted=not ours))


def at_ceiling(opt, facts):
    """Review critique (fix 2): whether Best so far is within the tie
    threshold of the top score (best ≥ top − rise_threshold(noise, 1)): no
    higher score could be told apart from tasting noise. False without a
    best or a noise threshold."""
    best = getattr(facts, "best", None) if facts is not None else None
    need = tie_need(getattr(facts, "yardstick", None)) if facts is not None else None
    if best is None or need is None:
        return False
    top = float(opt.utility_ceiling())
    return round(top - float(best.score), 2) < round(need, 2)


# The Next-step rows the stop signal at the top score replaces: every row
# that would make another Round — the ceiling is model-free (Best so far,
# the top, the noise) — except a disagreeing Repeat (check the panel
# first). The misses row keeps its Repeat request under the stop.
CEILING_KINDS = ("close", "explore", "rising", "see", "misses", "no_skill", "untested", "early")


def ceiling_stop(opt, facts):
    """Review critique (owner, follow-up): the stop signal shows when Best
    so far is within the tie threshold of the top score (at_ceiling) and
    What next? would otherwise make another Round — after a rise too (the
    Progress word stays Rising); never while a Round is open."""
    step = getattr(facts, "next", None) if facts is not None else None
    if step is None or step.kind not in CEILING_KINDS:
        return False
    if getattr(opt, "pending_batch", None) and open_rows(opt):
        return False
    return at_ceiling(opt, facts)


def ceiling_setup_text(opt):
    """Sceptic review: the stop's Set up "?", checking first — 'To check
    the stop, repeat Formulation 16: …' (Best so far, the formulation the
    stop reads), then what a new target or Share of score does."""
    try:
        best = opt.best_index()
        no = int(opt.formulation_ids[best]) if best is not None else None
        assumed = not insights.noise_yardstick(opt).from_repeats
        sd_text = noise_sd_text(opt)
    except Exception:
        return wording.CEILING_SETUP_HELP
    return wording.ceiling_setup_help(no, sd_text, assumed)


def _ceiling_foot(opt):
    """The strip's one primary at the top score: "Change on Set up", landing
    on Set up's callout (a target or a Share of score)."""
    lit = not confirmation_open()
    # Round-2 fixes (P2 #16): secondary — stopping needs no button, and a
    # red Set up read as the next step.
    # Final demo (B1): tertiary — no primary, nor a button, at the stop.
    tip = _say(opt, ceiling_setup_text(opt))
    if st.button(wording.SETUP_BUTTON_SHORT, type="tertiary",
                 disabled=not lit, key="insights_foot_setup", help=tip) and lit:
        st.session_state[SETUP_CALLOUT_KEY] = tip
        go_to_tab(TAB_SETUP, section=SETUP_CALLOUT_SLOT)


def binding_texts(opt, bindings):
    """['Tap water's Highest (42 g)', 'the Batch total (at most 98.7 g)']:
    every limit in `bindings`, with its amount."""
    out = []
    for b in bindings or ():
        if b.kind in ("lowest", "highest"):
            var = opt._by_name().get(b.name)
            amount = ""
            if var is not None and var.get('type') == 'continuous':
                value = float(var['bounds'][0 if b.kind == "lowest" else 1])
                unit = opt.unit_of(b.name)
                amount = (fmt_setting(value, unit) if insights.is_setting(opt, b.name)
                          else join_unit(f"{value:.0f}" if value == round(value)
                                         else _num(value), unit))
            out.append(wording.binding_item(b.kind, b.name, amount))
        else:
            name, _, rest = str(b.text).partition(":")
            name = re.sub(r"\s*\(.*?\)", "", name).strip() or b.text
            out.append(wording.binding_item("limit", name, rest.strip().replace(
                "at least ", "≥ ").replace(" and at most ", ", ≤ ").replace("at most ", "≤ ")))
    return out


def widen_all_text(opt, step):
    """The close row's widen advice naming EVERY limit the predicted best
    sits at (review critique, fix 2), and any ingredient among them that
    moved together with another (insights.moved_pairs); None with none."""
    bindings = list(getattr(step, "bindings", None) or
                    ([step.binding] if step.binding is not None else []))
    if not bindings:
        return None
    items = binding_texts(opt, bindings)
    named = {b.name for b in bindings if b.kind in ("lowest", "highest")}
    try:
        pairs = insights.moved_pairs(opt)
    except Exception:
        pairs = []
    moved = []
    for a, b, _r in pairs:
        if a in named:
            moved.append((a, b))
        elif b in named:
            moved.append((b, a))
    best = opt._best_recipe_index()
    ours = False
    if best is not None:
        here = {(x.kind, x.name) for x in insights.binding_limits(opt, insights.start_recipe(opt, best))}
        ours = all((x.kind, x.name) in here for x in bindings)
    return _say(opt, wording.widen_all(items, predicted=not ours, moved=moved))


def _close_change(b):
    """The close row's change in its words: raise a Highest, lower a Lowest,
    widen a limit; '' when nothing binds."""
    if b is None:
        return ""
    if b.kind == "limit":
        return wording.widen_limit(b.text)
    return wording.raise_highest(b.name) if b.kind == "highest" else wording.lower_lowest(b.name)


def next_sentence(opt, facts):
    """The Next line's sentence (R-I5): one plain sentence per next_step
    row, the Repeat of Best so far asked for while the noise threshold is
    assumed; at the gate, the record-or-make tail alone. Today's sentence
    follows it verbatim in Next-step reasoning."""
    no = opt.pending_batch_no
    left = len(open_rows(opt)) if opt.pending_batch else 0
    values = not opt.has_ingredients()
    step = facts.next
    if step is None:
        return wording.next_plain_tail(no, left)
    ask = (facts.best.formulation if getattr(step, "assumed", False) and facts.best is not None
           else None)
    if step.kind == "repeat":
        d = step.disagreement
        return wording.next_plain_repeat(d.later, d.earlier, f"{abs(d.diff):.1f}",
                                         lower=d.diff < 0,
                                         half_text=None if d.limit is None else f"{d.limit:.1f}",
                                         values=values,
                                         round_no=no if opt.pending_batch else opt.next_batch_no())
    if step.kind == "misses":
        if getattr(step, "diagnosis", None) is not None and facts.best is not None:
            # Review diagnosis: the Repeat that would measure the noise.
            return (f"{wording.next_plain_misses(no, left)} "
                    f"{wording.next_plain_repeat_ask(facts.best.formulation, bool(left))}")
        return wording.next_plain_misses(no, left)
    if step.kind == "early":
        return wording.next_plain_tail(no, left)
    if step.kind == "no_skill":
        return wording.next_plain_no_skill(no, left)
    if step.kind == "untested":
        return wording.next_plain_untested(no, open_round=bool(opt.pending_batch), values=values,
                                           shown=amounts_shown(facts))
    if step.kind == "explore":
        return wording.next_plain_explore(f"{step.gain:.1f}", assumed=step.assumed, no=no,
                                          left=left, repeat_no=ask, values=values)
    if step.kind == "close":
        return wording.next_plain_close(_close_change(step.binding),
                                        settings_only=not insights.has_ingredient_choice(opt),
                                        repeat_no=ask, values=values)
    if step.kind == "rising":
        return wording.next_plain_rising(no, left, repeat_no=ask, values=values)
    return wording.next_plain_see(no, left, repeat_no=ask, values=values)


def status_full_texts(opt, facts):
    """{line: its full sentence, through for_project}: 'Predictions: **Not
    yet.** …', 'Progress: **Flat.** …', 'Next: …' (M7: no word). Shown
    first in the line's tooltip since the concision round. The lines
    facts.pending names are left out."""
    out = {}
    if "predictions" not in facts.pending:
        out["predictions"] = wording.status_line(
            wording.STATUS_PREDICTIONS, prediction_sentence(opt, facts),
            wording.PREDICTIONS_WORDS[facts.predictions])
    if "progress" not in facts.pending:
        word = progress_word(opt, facts)
        out["progress"] = wording.status_line(wording.STATUS_PROGRESS,
                                              progress_sentence(opt, facts), word)
    if "next" not in facts.pending:
        out["next"] = wording.status_line(wording.STATUS_NEXT, next_sentence(opt, facts))
    return {k: _say(opt, v) for k, v in out.items()}


def status_texts(opt, facts):
    """{line: its on-screen text, through for_project} (concision round):
    'Predictions: **Reliable.** 33 of 35 results in range.', its bold word
    and one short fact; the full sentence is in its tooltip
    (status_full_texts). The lines facts.pending names are left out."""
    out = {}
    if "predictions" not in facts.pending:
        out["predictions"] = wording.status_line(
            wording.STATUS_PREDICTIONS, prediction_short(opt, facts),
            wording.PREDICTIONS_WORDS[facts.predictions])
    if "progress" not in facts.pending:
        word = progress_word(opt, facts)
        out["progress"] = wording.status_line(wording.STATUS_PROGRESS,
                                              progress_short(opt, facts), word)
    if "next" not in facts.pending:
        out["next"] = wording.status_line(wording.STATUS_NEXT, next_short(opt, facts))
    return {k: _say(opt, v) for k, v in out.items()}


def misses_plain(opt, facts):
    """The misses row's sentence, scoped (persona fixes): the measurements
    that missed most (2 misses or more, from the diagnosis), the Overall
    score's own record, then the row's tail and its Repeat."""
    step = facts.next
    d = getattr(step, "diagnosis", None)
    names = [i.name for i in (d.items if d is not None else []) if i.misses >= 2][:3]
    score = getattr(facts.headline, "score", None) if facts.headline is not None else None
    inside, total = (score.inside, score.total) if score is not None else (0, 0)
    # Fix round 4: the Overall score's checks with no forecast, named.
    nf = getattr(score, "no_forecast", 0) if score is not None else 0
    no = opt.pending_batch_no
    left = len(open_rows(opt)) if opt.pending_batch else 0
    scoped = wording.misses_scoped(names, inside, total, nf)
    said = misses_detail(facts)
    if said:
        # Specific wording: which measurements missed and how often.
        landed = wording.score_landed(inside, total, nf)
        scoped = f"{said} {landed}" if landed else said
    line = f"{scoped} {wording.next_plain_tail(no, left)}"
    if d is not None and facts.best is not None:
        line += " " + wording.next_plain_repeat_ask(facts.best.formulation, bool(left))
    return _say(opt, line)


def progress_answer_text(opt, facts):
    """Are we improving?'s answer after its word, after a rise (persona
    fixes): 'Best 95.0 → 98.3 (+3.3; needs 3.0)'; None otherwise (the
    card's line)."""
    f, a = facts.best, getattr(facts, "improvement", None)
    if f is None or a is None or getattr(a, "rise", None) is None \
            or facts.progress not in ("rising", "within_noise"):
        return None
    rise, need = rise_and_need(a)
    # Review critique (owner's decision on noise): the assumption shown
    # wherever the verdict rests on it.
    assumed = bool(getattr(getattr(facts, "yardstick", None), "assumed", True))
    return wording.progress_rise_answer(one_decimal(f.score - a.rise), one_decimal(f.score),
                                        rise, need, assumed=assumed, sd_text=noise_sd_text(opt))


def score_built_text(opt):
    """How the Overall score is built, one plain sentence from the
    project's own shares, goals and Ranges (FoodOptimizer._compute_utility)."""
    rows = []
    for o in opt.objectives:
        width = float(o.get('max_val', 10.0)) - float(o.get('min_val', 0.0))
        rows.append((o['name'], f"{round(float(o['weight']), 1):g}", o.get('goal'),
                     f"{round(width, 2):g}"))
    return _say(opt, wording.score_built(rows, f"{round(opt.utility_ceiling(), 1):g}"))


def score_recipe_text(opt):
    """Persona fixes 2: the score recipe in one caption under Closer to
    target — 'Score: 25 per measurement, minus 25 × distance ÷ 15.' — from
    the project's own shares and Ranges, when every measurement has a
    target with one share and one Range width; None otherwise (the "?"
    has the full sentence)."""
    objs = [o for o in opt.objectives]
    if not objs or any(o.get('goal') != 'target' for o in objs):
        return None
    shares = {f"{round(float(o['weight']), 1):g}" for o in objs}
    widths = {f"{round(float(o.get('max_val', 10.0)) - float(o.get('min_val', 0.0)), 2):g}"
              for o in objs}
    if len(shares) != 1 or len(widths) != 1:
        return None
    per = "measurement" if len(objs) > 1 else objs[0]['name']
    return _say(opt, wording.score_recipe(next(iter(shares)), next(iter(widths)), per))


def progress_plain(opt, facts):
    """The Progress "?"'s plain lines (persona fixes): what the rise needed
    is, then how the Overall score is built."""
    out = []
    a, y = getattr(facts, "improvement", None), getattr(facts, "yardstick", None)
    if facts.progress in ("rising", "within_noise") and a is not None \
            and getattr(a, "rise", None) is not None:
        _, need = rise_and_need(a)
        assumed = y is None or getattr(y, "assumed", True)
        sd = noise_sd_text(opt) if (a.sd is None or assumed) else f"{a.sd:.1f}"
        out.append(wording.progress_need_plain(need, sd, assumed, max(1, a.q or 1)))
    if plateau_round(opt, facts) is not None:
        out.append(wording.plateau_rule(plateau_round(opt, facts)))
    try:
        out.append(score_built_text(opt))
    except Exception:
        pass
    return [t for t in out if t]


def _full_sentence(line):
    """A full status line's sentence, its label and bold word taken off."""
    return re.sub(r"^\w+: (\*\*[^*]+\*\* )?", "", line)


def status_help(opt, facts, which):
    """The line's tooltip: its full sentence, then (review polish) one
    compact evidence line — n, method, rule — then W2's exact rule."""
    full = status_full_texts(opt, facts).get(which)
    tip = tooltip_for(opt, which, facts)
    try:
        ev = _say(opt, evidence_line(opt, facts, which))
    except Exception:       # the evidence line must never take a card down
        ev = None
    plain = []
    if which == "progress" and facts is not None and facts.best is not None:
        plain = progress_plain(opt, facts)
    elif which == "next" and facts is not None and facts.next is not None \
            and facts.next.kind == "misses":
        # Persona fixes: the misses row scoped to what missed, so it agrees
        # with the Model item's verdicts (its full sentence said "recent
        # results fell outside").
        full = None
        plain = [misses_plain(opt, facts)]
    # Persona fixes: the plain additions join the plain paragraph, so the
    # "?" stays plain, then evidence, then rule.
    lead = " ".join(t for t in (full and _full_sentence(full), *plain) if t)
    parts = [p for p in (lead, ev, tip) if p]
    return "\n\n".join(parts) or None


def evidence_line(opt, facts, which):
    """Review polish: the "?"'s second paragraph, the evidence in one line
    — how many, by what method, judged by which rule — read off the same
    facts the card's words come from (so every number agrees with the card
    and More detail); None when there is nothing yet."""
    if which == "predictions":
        h, row = facts.headline, facts.prediction_row
        if row == "gate":
            return wording.evidence([wording.ev_scored(facts.scored, insights.MIN_ROWS)])
        # Review four, ruling 1: the Overall score's record first (the
        # word's), then the measurements'; both Round-ahead.
        parts = []
        score = getattr(h, "score", None) if h is not None else None
        sk = facts.skill
        if score is not None and score.total:
            parts.append(wording.ev_predictions(
                score.inside, score.total, None if sk is None else skill_value(sk.value),
                None if sk is None else sk.n, h.inside if h.total else None, h.total or None))
            parts.append(wording.ev_misses(score.total - score.inside,
                                           insights.allowed_misses(score.total)))
        elif row == "too_few":
            parts.append(wording.ev_checks(facts.checked, insights.SKILL_MIN))
            if h is not None and h.total:
                parts.append(wording.ev_values(h.total, h.rounds))
        parts += [wording.EV_ROUND_AHEAD, wording.EV_METHOD]
        return wording.evidence(parts)
    if which == "progress":
        if facts.best is None:
            return None
        parts = [wording.EV_MODEL_FREE]
        a, y = facts.improvement, facts.yardstick
        if facts.progress in ("rising", "within_noise") and a is not None \
                and a.rise is not None and a.threshold is not None:
            parts.append(wording.ev_rise(f"{a.rise:.1f}", f"{a.threshold:.1f}"))
        elif facts.progress == "flat" and a is not None:
            parts.append(wording.ev_flat(a.rounds))
        if y is not None and y.value is not None:
            parts.append(wording.ev_noise(f"{y.value:g}", y.assumed, y.df))
        return wording.evidence(parts)
    step = facts.next
    if step is None:
        return None
    cfg = getattr(opt, "bo_config", None) or {}
    # The row's name as the rule table and the rule paragraph name it.
    # Round-2 fixes (P3 #17): the rule itself in plain words, its own
    # sentence, not the rule table's name.
    rule = insights.status_rule(opt, ("next", step.kind))
    return (wording.evidence([wording.ev_suggestions(cfg.get("acquisition", "qlognei")),
                              wording.EV_METHOD]) + " " + wording.rule_line(rule))


def _status_line(opt, facts, which):
    """One status line, short (concision round), its full sentence and
    W2's exact rule in the tooltip; its caption while its facts are still
    being worked out (R-I8)."""
    text = None if facts is None else status_texts(opt, facts).get(which)
    if text is None:
        st.caption(_say(opt, _PLACEHOLDERS[which]))
    else:
        st.markdown(text, help=status_help(opt, facts, which))


def prediction_short(opt, facts):
    """The Predictions line on screen (concision round): one fact, the
    numbers kept; the check advice kept when the full sentence gives it."""
    row = facts.prediction_row
    check = False
    for c in facts.clauses:
        if c[0] == "wide":
            check = wide_cause(c[1], getattr(facts, "wide_factors", None) or {}) is not None
    then = scatter_tail(opt, facts, short=True) if check else ""
    if row == "off_repeat":
        return wording.SHORT_OFF_REPEAT
    if row == "gate":
        return wording.short_gate(facts.scored, insights.MIN_ROWS)
    if row == "too_few":
        return wording.short_too_few(facts.checked, insights.SKILL_MIN) + then
    if row == "wide":
        return wording.SHORT_WIDE
    if row == "no_skill":
        return wording.SHORT_NO_SKILL + then
    if row == "unclear":
        return wording.SHORT_UNCLEAR + then
    inside, total, _ = score_counts(facts)      # review four, ruling 1
    allowed = insights.allowed_misses(total)
    if row in ("off_coverage", "caution"):
        line = _on_score(facts, wording.short_out_of_range(total - inside, total, allowed))
        return line + (then if row == "caution" else "")
    return _on_score(facts, wording.short_in_range(inside, total)) + then


def progress_short(opt, facts):
    """The Progress line on screen (concision round): one fact and Best so
    far; ties within noise, luck and "little more" are in the tooltip."""
    f = facts.best
    if f is None:
        return wording.NO_COMPLETE_FORMULATION
    ceiling = opt.utility_ceiling()
    tie = None if facts.tie is None else int(opt.formulation_ids[facts.tie])
    best = wording.short_best(f"{f.score:.1f}", ceiling, f.formulation, tie)
    a = facts.improvement
    if facts.progress in ("rising", "within_noise"):
        rise, need = rise_and_need(a)
        short = wording.short_rise_best(f"{f.score:.1f}", f.formulation, tie)
        return f"{wording.short_rise(rise, need, rising=facts.progress == 'rising')} {short}"
    if facts.progress == "flat":
        return f"{wording.short_flat(a.rounds)} {best}"
    if facts.progress == "no_new_best" or one_round(opt):
        return best
    without = scored_without(opt, f.round)
    return f"{best} {wording.short_lack(*without)}" if without else best


def next_short(opt, facts):
    """The Next line on screen (concision round): the action, the Repeat
    request and the check advice kept; the reason is in the tooltip."""
    no = opt.pending_batch_no
    left = len(open_rows(opt)) if opt.pending_batch else 0
    step = facts.next
    if step is None:
        return wording.short_tail(no, left)
    ask = (facts.best.formulation if getattr(step, "assumed", False) and facts.best is not None
           else None)
    if step.kind == "repeat":
        d = step.disagreement
        return wording.short_check_repeat(
            d.later, d.earlier, f"{abs(d.diff):.1f}", lower=d.diff < 0,
            round_no=no if opt.pending_batch else opt.next_batch_no())
    if step.kind == "misses":
        return f"{wording.short_tail(no, left)} {wording.SHORT_MISSES}"
    if step.kind == "early":
        return wording.short_tail(no, left)
    if step.kind == "no_skill":
        return f"{wording.short_tail(no, left)} {wording.SHORT_NO_SKILL_NEXT}"
    if step.kind == "untested":
        return wording.short_untested(no, open_round=bool(opt.pending_batch))
    if step.kind == "close":
        line = wording.short_close(_close_change(step.binding),
                                   settings_only=not insights.has_ingredient_choice(opt))
        return f"{line} {wording.short_add_repeat(ask)}" if ask is not None else line
    if ask is not None:
        return wording.short_repeat(ask, no, left, opt.next_batch_no())
    return wording.short_tail(no, left)


def _status_lines(opt, facts):
    """W1 (§1, review R-B1, R-I8): the three status lines from
    insights.StatusFacts, each with W2's tooltip_for help, in their order;
    for None (nothing worked out yet), the three captions that hold their
    places."""
    for which in _STATUS_ORDER:
        _status_line(opt, facts, which)


def _why_link(opt):
    """W1 (review R-I4): "Why the model suggested Round N", only with a warm
    Round open holding at least one generated pick (_why_link_offered); the
    one tertiary link above the fold. Lands on Make a round's Why fold,
    opened and scrolled into view (its keyed place, ftue fix C: the fold
    itself carries no class to scroll to). Returns whether it was drawn."""
    if not _why_link_offered(opt):
        return False
    link(_say(opt, wording.why_link(opt.pending_batch_no)), TAB_BATCH, section=WHY_PLACE,
         fold=WHY_FOLD_KEY, key="insights_why_link")
    return True


REPEAT_LINK_KEY = "insights_repeat_link"


def quiet_repeat_text(opt, facts):
    """At the stop: the Repeat request _repeat_link would draw, as the text
    of a quiet line (no button), or None. _next_body draws it under the
    callout unless the card says it already."""
    step = getattr(facts, "next", None)
    best = getattr(facts, "best", None)
    if (step is None or step.kind not in _ASK_REPEAT or not getattr(step, "assumed", False)
            or best is None or _why_link_offered(opt)):
        return None
    return _say(opt, wording.card_noise_repeat(best.formulation))


def _repeat_link(opt, facts):
    """Review R-I3, ruled in by the early walkthrough (ftue fix C): "Make a
    Repeat of Formulation N" while Next asks for a Repeat of Best so far
    (next_sentence's rule: an asking row while the noise threshold is
    assumed), the one tertiary link when the Why link is not drawn
    (priority 2). It leaves Make a round a one-shot request
    (REPEAT_REQUEST_KEY) to open "Add a formulation to this round" with
    Start from Best so far applied; nothing is saved until Add.

    With a Round open it lands on that fold, opened. With none it lands at
    the top of Make a round: adding first would open a Round of the user's
    own alone, with no Generate after it, so the fold opens filled in once
    Generate has started the Round (ui_batch._take_repeat_request).
    Returns whether it was drawn."""
    step = getattr(facts, "next", None)
    best = getattr(facts, "best", None)
    if (step is None or step.kind not in _ASK_REPEAT or not getattr(step, "assumed", False)
            or best is None or _why_link_offered(opt)):
        return False
    open_no = opt.pending_batch_no if opt.pending_batch else None
    request = {REPEAT_REQUEST_KEY: {"project": opt.project_name, "no": int(best.formulation),
                                    "round": open_no}}
    if open_no is None:
        section, fold = "top", None
    else:
        section, fold = ADD_OWN_PLACE, ADD_OWN_KEY
    link(_say(opt, wording.repeat_link(best.formulation)), TAB_BATCH, section=section, fold=fold,
         key=REPEAT_LINK_KEY, state=request)
    return True


def score_record_help(opt):
    """The Overall score's own record for Make a round's column help, or
    '' — only from checks already worked out for the data on screen, by
    this tab or from the workers' models (§9 I6): Make a round never fits
    for it. A steady bias follows it, in Insights' words (0.8.0 e2e fixes,
    review B1): one wording for one fact."""
    checks = checks_for(opt, "trust")
    if checks is None:
        return ""
    line = score_line(opt, checks)
    bias = steady_bias_text(opt, checks) if line else ""
    return f"{line} {bias}" if bias else line


def score_line(opt, checks, h=None, yard=None):
    """The Overall score's line, in one wording for Insights' answer 1 and
    Make a round's column help (spec §0.3): its 95 % intervals and mean
    error over every Round checked, naming them (0.8.0 e2e fixes, review
    I1). A steady bias is said once, by steady_bias_text, so this line
    never says it. '' without a record. `yard` is not read (kept for
    callers)."""
    h = insights.headline(opt, checks) if h is None else h
    if h.score is None:
        return ""
    nf = getattr(h.score, "no_forecast", 0)
    # Fix round 4: the mean error is over every check, the no-forecast
    # ones too; they are left out of the inside count only.
    return wording.with_left_out(
        wording.score_gap_line(h.score.inside, h.score.total, h.score.mean_miss, "",
                               n=h.score.total + nf, rounds=h.score.rounds),
        nf, _say(opt, wording.OVERALL_SCORE_COLUMN), wording.LEFT_OUT_OF_INSIDE)


def _bias_size(opt, b):
    """A steady bias's size as the line says it: points to one decimal for
    the Overall score; a measurement's own decimals, at least one."""
    d = 1 if b.name == insights.SCORE else max(1, decimals(opt, b.name))
    return f"{abs(b.mean):.{d}f}"


def _unit(opt, name):
    obj = next((o for o in opt.objectives if o['name'] == name), {})
    return obj.get('unit') or ""


def steady_bias_text(opt, checks, flagged=None):
    """Prediction check's one steady-bias line (0.8.0 e2e fixes, review B1):
    every output whose predictions ran steadily high or low over every Round
    checked, as a fact with no advice; '' when none did. Results below
    prediction (a negative error) mean predictions ran high."""
    found = insights.steady_bias(opt, checks, flagged)
    if not found:
        return ""
    high, low, rounds = [], [], []
    for b in found:
        name = wording.OVERALL_SCORE_COLUMN if b.name == insights.SCORE else b.name
        unit = "points" if b.name == insights.SCORE else _unit(opt, b.name)
        (high if b.mean < 0 else low).append(wording.bias_item(name, _bias_size(opt, b), unit))
        rounds.extend(r for r in b.rounds if r not in rounds)
    return wording.steady_bias_line(high, low, sorted(rounds))


def threshold_text(yard, assumed=True):
    """The noise threshold's one name (03 §3 rule 3): the Repeat SD from 3
    pairs, else the fixed 1 point, said to be assumed while fewer than 3
    pairs exist (0.8.0 e2e fixes, review B2: beat_by's third form).
    `assumed` False: the plain form, where the screen has said it above."""
    if yard.from_repeats:
        return wording.beat_by(f"{yard.value:.1f}")
    return wording.beat_by(None, assumed=assumed and getattr(yard, "assumed", False),
                           pairs=yard.pairs, left_out=getattr(yard, "left_out", 0),
                           df=getattr(yard, "df", 0) or None)


def skill_value(value):
    """A skill, two decimals, a true minus sign: '0.36', '−0.47'."""
    return wording.signed(value, 2).lstrip("+")


def skill_text(opt, checks):
    """The Overall score's skill line (spec 2026-09-30-bo-views §2.1;
    review I1, I2), or '' below 10 checks."""
    fact = insights.skill(opt, checks)
    if fact is None:
        return ""
    return wording.skill_line(skill_value(fact.value), fact.n, fact.rounds, fact.state)


def header_text(opt, h=None):
    """The tab's first line (§2.1): scored formulations (all their results
    in), up to the latest scored Round, then the waiting ones. A formulation
    moved to Not scored is neither scored nor waiting. The likely range is
    defined where it is first used, in the verdict. `h` is not read (kept
    for callers of 0.8.0's two forms)."""
    order = insights.scored_rounds(opt)
    waiting = opt.awaited_rows()
    sets = {tuple(opt.waiting_on(i)) for i in waiting}
    names = number_list(next(iter(sets))) if len(sets) == 1 else ""
    # Fix round B (five-second re-test): the waiting formulations by number.
    return wording.insights_header_short(opt.complete_count(),
                                         order[-1][0] if order else None,
                                         len(waiting), names,
                                         numbers=[int(opt.formulation_ids[i]) for i in waiting])


def header_detail(opt):
    """header_text's full line, for its tooltip: '… (all results in) …
    and is not used yet.'"""
    order = insights.scored_rounds(opt)
    waiting = opt.awaited_rows()
    sets = {tuple(opt.waiting_on(i)) for i in waiting}
    names = number_list(next(iter(sets))) if len(sets) == 1 else ""
    return wording.insights_header_line(opt.complete_count(),
                                        order[-1][0] if order else None,
                                        len(waiting), names,
                                        numbers=[int(opt.formulation_ids[i]) for i in waiting])


def _wide(opt, h):
    """The wide-range line under the verdict, or ''."""
    if not h.wide:
        return ""
    measured_names = [o['name'] for o in opt.objectives]
    if all(n in h.wide for n in measured_names + [insights.SCORE]):
        return wording.WIDE_EVERY
    return wording.wide_line(_wide_names(h.wide))


def _wide_names(names):
    """'Firmness, Juiciness and the Overall score' for outputs whose likely
    range is wide; '' for none."""
    return number_list([f"the {wording.OVERALL_SCORE_COLUMN}" if n == insights.SCORE else n
                        for n in names]) if names else ""


def verdict_text(opt, h, state, bias=None, checks=()):
    """Prediction check in one line per state (§2.2, review I6, M9; 03
    §5.2): the measurements' values inside their 95 % intervals with the
    miss count the rule allows; some outputs wide adds the wide line. The
    verdict names the measurements, and the Overall score's line under it
    says its mean error, so nothing is added for a bias (`bias` is kept for
    callers). `checks`: for the over-wide rule's basis."""
    if state == "none":
        return wording.TRUST_NONE
    left = getattr(h, "no_forecast", 0)
    if state == "early":
        return wording.trust_early(h.total, left)
    if state == "wide":
        return wording.TRUST_WIDE
    allowed = insights.allowed_misses(h.total)
    d = diag_for(opt, checks, h) if state in ("caution", "unreliable") else None
    if d is not None:
        # Review diagnosis: one story, the count then where the misses fell;
        # no "too narrow" beside a "wide" line.
        line = wording.trust_misses_diag(h.rounds, h.inside, h.total,
                                         many=state == "unreliable", allowed=allowed)
        return wording.with_left_out(" ".join([
            line, wording.diag_detail(diag_items(opt, d), d.rounds, d.far, d.misses,
                                      d.far_checked, d.checked),
            wording.DIAG_READING[d.kind]]), left)
    if state in ("caution", "unreliable"):
        line = wording.trust_misses(len(h.rounds), h.rounds, h.inside, h.total,
                                    many=state == "unreliable", allowed=allowed)
    elif state == "over_wide":
        line = wording.trust_over_wide(h.total, h.rounds,
                                       insights.median_z(h, checks) if checks else None)
    else:
        line = wording.trust_within(h.inside, h.total, h.rounds, allowed=allowed)
    line = wording.with_left_out(line, left)
    wide = _wide(opt, h)
    return f"{line} {wide}" if wide else line


def _beyond(gap, yard):
    """What the Overall score's average gap is more than, when it is (review
    I5): "Repeats differ" from 3 pairs, else "1 point"; '' within. The
    screens now say a bias only by insights.bias_gap (score_line); this
    is kept for the answer-key scripts that call it."""
    if yard.value is None or abs(gap) <= yard.value:
        return ""
    return "Repeats differ" if yard.from_repeats else "1 point"


def disagreement_text(d):
    return wording.repeat_disagrees(d.later, d.earlier, f"{abs(d.diff):.1f}", lower=d.diff < 0,
                                    half_text=None if d.limit is None else f"{d.limit:.1f}")


def _keep_if_offered(key, options):
    """A choice left from before a deletion or a Set up change is dropped
    before its box is drawn, so the box falls back to its first entry."""
    if key in st.session_state and st.session_state[key] not in options:
        del st.session_state[key]


# Review profiler (prototype): Try one's body below (_change, its answer
# line, Compare all, the path layer) is no longer drawn; the profiler at the
# end of this file replaces it. Kept for the pins that still call it
# (test_104_speed_robust_charts and the skip-marked ones) and for path_chart,
# which test_review_plots pins; remove at the final merge.
# The whole path's change must beat the tested-near one by more than this
# (a tenth of a point, the precision shown) before near_bigger is said.
_FLAT = 0.1


_GUARD_WORDS = {"floor": wording.GUARD_FLOOR, "noise": wording.GUARD_NOISE}


def _guard(opt, fitted, name=insights.SCORE):
    """insights.noise_guard for the display model of `name`, with the
    checks already worked out for this data (none: skill is not judged)."""
    fitted = fitted or {}
    return insights.noise_guard(opt, fitted.get(name), checks_for(opt, "trust") or (), name)


def guard_text(guard, settings_only=False):
    """The guard's one line, naming its trigger (review I3): "fitting
    noise" for the correlation length or the fitted noise; trigger (c)
    alone reads "not reported until…", by its A3 gate (0.8.0 e2e fixes,
    review I4): too few checks yet, skill not clearly better than the
    average, or no skill."""
    if guard.skill_only:
        gate = getattr(guard, "gate", None)
        if gate is not None and gate.state == "wait":
            return wording.effects_wait_line(gate.n, settings_only)
        if gate is not None and gate.state == "unclear":
            return wording.effects_unclear_line(skill_value(gate.fact.value), gate.fact.n,
                                                settings_only)
        return wording.guard_skill_line(settings_only)
    return wording.guard_line([_GUARD_WORDS[r] for r in guard.reasons if r != "skill"],
                              settings_only)


def path_guard_line(guard):
    """The path's caption while the guard fires (review I3, I4), or '':
    the curve still draws; its bends are not to be read as effects."""
    if not guard.fires:
        return ""
    if not guard.skill_only:
        return wording.PATH_GUARD
    gate = getattr(guard, "gate", None)
    if gate is not None and gate.state == "wait":
        return wording.path_guard_wait(gate.n)
    if gate is not None and gate.state == "unclear":
        return wording.path_guard_unclear(skill_value(gate.fact.value), gate.fact.n)
    return wording.PATH_GUARD_SKILL


def _whole(value):
    return f"{value:.0f}"


def range_text(low, high, fmt=None):
    """"70–85"; "-1.25 to 0.4" when a bound is negative, so a minus sign is
    never read as part of the dash."""
    fmt = fmt or _num
    sep = " to " if low < 0 or high < 0 else "–"
    return f"{fmt(low)}{sep}{fmt(high)}"


def score_text(p, top=100.0):
    """An Overall-score prediction: whole numbers, "78 (70–85)"; never 100
    (review learned): "≥ 99 (95–100)"."""
    return capped_range(p, top, 0)


def value_text(p, d=None):
    """A measurement prediction: its mean and likely range with one number
    of decimals, `d` (decimals(opt, name)); without one, the tab's own
    `_num` rule for each number."""
    if d is None:
        return f"{_num(p.mean)} ({range_text(p.low, p.high)})"
    fmt = lambda v: f"{v:.{d}f}"
    return f"{fmt(p.mean)} ({range_text(p.low, p.high, fmt)})"


def _num(value):
    """A number for the tab, never in scientific notation: 0 decimals from
    100, 1 from 10, otherwise up to 2 significant decimals with no trailing
    zeros. Only a true zero reads 0: a non-zero value gets enough decimals
    (up to 6) to show, and one too small even for that says so."""
    a = abs(value)
    if a >= 100:
        return f"{value:.0f}"
    if a >= 10:
        return f"{value:.1f}"
    if a == 0:
        return "0"
    decimals = 2 if a >= 1 else min(6, 1 - math.floor(math.log10(a)))
    text = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    if text.strip("-") in ("0", ""):
        return "< 0.000001" if value > 0 else "> -0.000001"
    return text


def _amount(value, unit):
    return join_unit(_num(value), unit)


def _shared_decimals(values):
    """How many decimals to show every one of `values` with: the most any
    of them needs on its own (by `_num`'s own rule), capped at 2 — so three
    numbers shown together never mix "10.0" and "9"."""
    decimals = 0
    for v in values:
        text = _num(v)
        if "." in text:
            decimals = max(decimals, len(text.split(".", 1)[1]))
    return min(2, decimals)


def limit_end(opt, name, amount):
    """"Highest" or "Lowest" when `amount` sits within 1 % of the span from
    that limit of row `name`, else None (review plots)."""
    try:
        lo, hi = (float(b) for b in insights._var(opt, name)['bounds'])
        amount = float(amount)
    except Exception:
        return None
    span = hi - lo
    if span <= 0:
        return None
    if amount >= hi - 0.01 * span:
        return "Highest"
    if amount <= lo + 0.01 * span:
        return "Lowest"
    return None


def stand_text(answer, yard=None):
    """The Yes / Barely / Not lately answer (§2.3, review I5, M3). One
    decimal, unless the rise would read 0.0 or the same as what it is
    judged against: then hundredths, so the words never contradict the
    numbers (Task 6 review M-1). `yard` (noise_yardstick): when its 1 point
    is assumed, the line says so (0.8.0 e2e fixes, review B2); this is the
    first line on the tab that names the threshold."""
    if answer.state == "not_lately":
        return wording.stand_not_lately(answer.rounds)
    rise, need = rise_and_need(answer)
    sd = None if answer.sd is None else f"{answer.sd:.1f}"
    assumed = (sd is None and yard is not None and getattr(yard, "assumed", False))
    basis = wording.beat_by(
        sd, assumed=assumed, pairs=yard.pairs if assumed else 0,
        left_out=getattr(yard, "left_out", 0) if assumed else 0,
        df=getattr(yard, "df", 0) or None if assumed else None)
    # Fix A follow-up (decision 4): the threshold itself, Improvement's,
    # and its make-up: noise SD × √2 × the max-of-q factor.
    threshold = wording.rise_needs(need, rise_factor_text(answer.q, answer.df), basis)
    if answer.state == "yes":
        return wording.stand_yes(rise, threshold)
    return wording.stand_barely(rise, threshold_text=threshold)


# (id of the project, Repeat pairs, whether Progress's line 1 named the
# noise threshold), set by _stand_lines, which every run draws before line 2.
_LINE1_KEY = "_insights_line1_threshold"


def _line1_named(opt, yard):
    """Whether line 1 named the noise threshold: as _stand_lines recorded it
    in this run, else worked out (a caller outside the tab)."""
    hit = st.session_state.get(_LINE1_KEY)
    if hit is not None and hit[0] == id(opt) and hit[1] == yard.pairs:
        return hit[2]
    return says_threshold(insights.improvement(opt, yardstick=yard))


def says_threshold(answer):
    """Whether Progress's line 1 names the noise threshold: Rising and
    Within noise do."""
    return answer is not None and answer.state in ("yes", "barely")


def repeats_text(yard):
    """The Repeat SD line under a rise, with its degrees of freedom and the
    formulations made more than once (0.8.0 e2e fixes, review I13)."""
    return wording.stand_repeats(f"{yard.sd:.1f}", yard.pairs,
                                 df=getattr(yard, "df", 0) or None,
                                 groups=getattr(yard, "groups", 0) or None)


def tie_text(opt, yard):
    """R-I9's tie line for Progress figures: Best so far's runner-up
    (insights.tie_partner, a Repeat never one) when it is within the noise
    threshold, equal scores included; '' otherwise."""
    top = opt.best_index()
    other = None if top is None else insights.tie_partner(opt, top)
    need = tie_need(yard)
    if other is None or need is None:
        return ""
    lead = float(opt.Y_history[top]) - float(opt.Y_history[other])
    if round(lead, 2) >= round(need, 2):
        return ""
    # Fix A follow-up (ruling on ties): rise_threshold(yard, 1), Rising's
    # rule for one new result, named with its make-up.
    df = yard.df if yard.from_repeats else None
    return wording.tie_runner_up(int(opt.formulation_ids[other]),
                                 f"{float(opt.Y_history[other]):.2f}",
                                 wording.rise_needs(f"{need:.1f}", rise_factor_text(1, df),
                                                    threshold_text(yard, assumed=False),
                                                    what="needed to differ"))


def progress_frame(opt):
    """The Progress chart's data, one row per scored formulation. A
    formulation still waiting for results is a gap in the Overall score line
    and repeats the best so far before it."""
    formulation = _say(opt, wording.FORMULATION_CAP)
    return pd.DataFrame({
        formulation: [int(n) for n in opt.formulation_ids],
        wording.OVERALL_SCORE_COLUMN: [float(y) if opt.fully_measured(i) else None
                                       for i, y in enumerate(opt.Y_history)],
        wording.BEST_SO_FAR_COLUMN: opt.best_so_far(),
    }).set_index(formulation)


def progress_chart(opt):
    """The Progress chart: Overall score per Formulation (a line with a dot
    at each measured one — a gap where a formulation is still waiting for
    its results) and Best so far (a second, neutral grey dashed line), both
    against Formulation number on one shared axis. Drawn with Altair, not
    st.line_chart (Minor 14): a built-in chart has no hook to turn its
    action menu off, so it is redrawn here through the same door — show_chart
    — as every other chart on this tab."""
    frame = progress_frame(opt).reset_index()
    formulation = frame.columns[0]
    score, best = wording.OVERALL_SCORE_COLUMN, wording.BEST_SO_FAR_COLUMN
    # Spec 2026-09-29 §9 I4: a formulation scored without a measurement
    # (recorded before it existed) is never Best so far, and its dot is a
    # ring rather than a filled dot. "series" names the line both marks
    # share, so the ring's outline and the filled dot are the line's colour.
    pool = set(opt.best_candidates())
    frame["full"] = [i in pool for i in range(len(frame))]
    each, best_name = _say(opt, wording.PLOT_SCORE_EACH), _say(opt, wording.BEST_SO_FAR_COLUMN)
    frame["series"] = each
    frame["best_series"] = best_name
    frame["bo"] = wording.PLOT_BO_MODEL_FREE
    frame["round"] = [_round_label(r) for r in
                      (list(opt.batch_history) + [None] * len(frame))[:len(frame)]]
    # Both axes named. Unlabelled, the x axis read 1.0 / 1.5 / 2.0 with one
    # round of data and nothing on the chart said what either number was.
    # Whole-number ticks, each once (spec 2026-09-30 §2.3, expert #8), and a
    # y axis that fits the scores rather than starting at 0.
    # Review four polish: Q2 asks about Rounds, so the axis says the order
    # and each Round is marked (round_marks).
    x = alt.X(f"{formulation}:Q", title=_say(opt, wording.Q2_X_TITLE),
              axis=alt.Axis(format="d", values=formulation_ticks(frame[formulation])))
    values = [v for v in list(frame[score]) + list(frame[best]) if v is not None and v == v]
    y_title = (zoomed(wording.PLOT_MEASURED_SCORE, min(values), max(values),
                      float(opt.utility_ceiling())) if values else wording.PLOT_MEASURED_SCORE)
    y = alt.Y(f"{score}:Q", title=y_title, scale=alt.Scale(zero=False))
    # Review plots: each score in neutral grey, Best so far in ink, both in
    # the legend (was a default blue, unexplained but for the caption).
    grey, _ = mark("neutral")
    ink, _ = mark("bar_latest")
    kinds = alt.Scale(domain=[each, best_name], range=[grey, ink])
    marks = round_marks(opt, list(frame[formulation]))
    legend = alt.Legend(title=None, orient="top", labelLimit=0)
    colour = alt.Color("series:N", scale=kinds, legend=legend)
    # Titled and rounded (0.8.0 review fixes §6): a measured Overall score
    # reads one decimal, never 73.283749….
    no_tip = alt.Tooltip(f"{formulation}:Q", title=formulation, format="d")
    score_line = (alt.Chart(frame).mark_line(tooltip=False)
                  .encode(x=x, y=y, color=colour))
    dots = (alt.Chart(frame).mark_point(size=40, strokeWidth=1.5)
            .encode(x=x, y=y, color=colour,
                    fill=alt.condition(alt.FieldEqualPredicate(field="full", equal=True),
                                       alt.Fill("series:N", scale=kinds, legend=None),
                                       alt.value("transparent")),
                    tooltip=[no_tip, alt.Tooltip("round:N", title=_say(opt, wording.ROUND_CAP)),
                             alt.Tooltip(f"{score}:Q", title=score, format=".1f"), bo_tip()])
            .transform_filter(alt.FieldValidPredicate(field=score, valid=True)))
    best_line = (alt.Chart(frame).mark_line(strokeWidth=2.5, interpolate="step-after")
                 .encode(x=x, y=alt.Y(f"{best}:Q", title=y_title, scale=alt.Scale(zero=False)),
                         color=alt.Color("best_series:N", scale=kinds, legend=legend),
                         tooltip=[no_tip, alt.Tooltip(f"{best}:Q", title=best, format=".1f")]))
    layers = [score_line, dots, best_line]
    if marks:
        rules, labels = marks
        light, _ = _TRIED_GREYS[theme()]
        # The rules on the Formulation field itself (one scale); their own
        # axis along the top names the Rounds (marks above the plot as text
        # pushed the bottom axis out of Streamlit's fitted height).
        mids = [m for m, _ in labels]
        expr = " : ".join(f"abs(datum.value - {m}) < 0.01 ? '{t}'" for m, t in labels) + " : ''"
        top = alt.X(f"{formulation}:Q", title=None,
                    axis=alt.Axis(orient="top", values=mids, labelExpr=expr, ticks=False,
                                  domain=False, grid=False, labelFontSize=9, labelPadding=2,
                                  labelColor=muted()))
        frame_rules = (alt.Chart(pd.DataFrame({formulation: rules}))
                       .mark_rule(color=light, strokeWidth=1, tooltip=False).encode(x=top))
        return titled(alt.layer(frame_rules, alt.layer(*layers)).resolve_axis(x="independent")
                      .properties(height=220), _say(opt, wording.PLOT_PROGRESS))
    return titled(alt.layer(*layers).properties(height=220),
                  _say(opt, wording.PLOT_PROGRESS))


def round_marks(opt, numbers):
    """(rule positions, [(x, label)]) for Q2's Rounds: a light rule between
    two Rounds (halfway between their Formulations), 'R3' over each Round's
    middle — every other label past 12 Rounds. None with one Round or none."""
    rounds = (list(opt.batch_history) + [None] * len(numbers))[:len(numbers)]
    groups = []          # [round, first x, last x]
    for r, x in zip(rounds, numbers):
        if groups and groups[-1][0] == r:
            groups[-1][2] = x
        else:
            groups.append([r, x, x])
    named = [g for g in groups if g[0] is not None]
    if len(named) < 2:
        return None
    rules = [(a[2] + b[1]) / 2 for a, b in zip(groups, groups[1:])]
    step = 2 if len(named) > 12 else 1
    labels = [((g[1] + g[2]) / 2, wording.round_tick(g[0])) for g in named[::step]]
    return rules, labels


# At most this many ticks on the Progress chart's Formulation axis.
_MAX_TICKS = 12


def formulation_ticks(numbers):
    """Whole numbers from the first formulation to the last, every one while
    they fit in _MAX_TICKS, else every k-th: never a repeated or a
    fractional tick."""
    numbers = [int(n) for n in numbers]
    if not numbers:
        return []
    lo, hi = min(numbers), max(numbers)
    step = max(1, math.ceil((hi - lo + 1) / _MAX_TICKS))
    return list(range(lo, hi + 1, step))


def _predicted_best(opt, fitted, checks=(), spinner=True):
    """insights.predicted_best, once per setup_key (the checks follow from
    the data it covers). A failure is cached as such and shows one line.
    `spinner` False while the tab's one progress line is showing (V28)."""
    key = insights.setup_key(opt)
    hit = st.session_state.get(BEST_KEY)
    if hit and hit[0] == key:
        return hit[1]
    with (st.spinner(wording.INSIGHTS_WORKING) if spinner else contextlib.nullcontext()):
        try:
            value = insights.predicted_best(opt, fitted, checks)
        except Exception:
            value = _FAILED
    st.session_state[BEST_KEY] = (key, value)
    return value


def binding_text(b):
    """One item of Line 3: 'the Highest of Pea', 'the limit Fat: at most 20 g'."""
    return (wording.at_highest(b.name) if b.kind == "highest"
            else wording.at_lowest(b.name) if b.kind == "lowest" else wording.at_limit(b.text))


def standing_text(opt, standing, yard, withheld=None):
    """Line 2 (§2.3, review I7): the predicted best against Best so far's
    predicted Repeat, in whole points, in one of four texts. `withheld`
    (claims_withheld_reason): the gain is not claimed, and the line says
    why (polish round)."""
    p = standing.best.score
    top = float(opt.utility_ceiling())
    # Review critique (fix 12): never a top below the point.
    mean = capped(min(p.mean, top), top, 0)
    low = f"{min(p.low, top):.0f}"
    high = f"{top:g}" if p.high >= top - 0.5 else f"{p.high:.0f}"
    if withheld:
        return wording.stand_withheld(mean, low, high, withheld, standing.close_no,
                                      values=not opt.has_ingredients())
    # An assumed threshold is said once on Progress (review B2): here only
    # when line 1 did not name it.
    by = threshold_text(yard, assumed=getattr(yard, "assumed", False)
                        and not _line1_named(opt, yard))
    values = not opt.has_ingredients()
    if standing.higher:
        close = standing.best.close
        best = opt.best_index()
        ref = (standing.best.repeat.mean if standing.best.repeat is not None
               else None if best is None else float(opt.Y_history[best]))
        # The predicted gain the rule used (03 §5.3): over a Repeat of Best
        # so far as predicted.
        gain = None if ref is None else wording.signed(p.mean - ref)
        return wording.stand_higher(mean, low, high, standing.close_no,
                                    "" if close is None else f"{opt.Y_history[close]:.2f}",
                                    gain_text=gain, by_text=by, values=values)
    return wording.stand_not_higher(by, mean, low, high, standing.close_no, values=values)


def claims_withheld_reason(opt, score, checks):
    """Why the model's gains are not claimed, in Progress's words, or None:
    the expected improvement's own rule (withheld, gain_reason). The
    predicted gain over Best so far, its amounts and the triangle's
    Predicted above Best so far outline follow it too (polish round: the
    upgrade file withheld the one and claimed +9.3 points beside it)."""
    why = withheld(opt, score, checks)
    if why is None:
        return None
    return gain_reason(why, insights.skill_gate(opt, checks) if why == "skill" else None)


def standing_for(opt, best, checks, yard):
    """Progress's Standing. The luck line is read against the Overall
    score's steady bias, worked out only when there is a luck line to
    read (final review I-1)."""
    standing = insights.best_standing(opt, best, yard.value)
    if standing.luck:
        standing = insights.best_standing(
            opt, best, yard.value, score_bias=insights.score_steady_bias(opt, checks))
    return standing


# (key, bo_views.RoundGain or None): §3.2's figure for this data, open Round
# and generation record, worked out once (fix round 1, C1: Progress, Next
# step and Make a round's E3 read the same number).
GAIN_KEY = "_insights_gain"
# (bo_views.next_gain_key, bo_views.RoundGain): the next Round's figure as
# it is written beside the project (spec 2026-10-01 C1, review I11), read
# back by memo() on a reopen so it is neither fitted nor searched for again.
NEXT_GAIN_KEY = "_insights_next_gain"


def _gain_key(opt):
    """What the figure on screen depends on: the setup, the acquisition,
    the open Round and its record, and (no Round open) the seed the next
    Generate uses and robust, which move without the data (review I11)."""
    record = (getattr(opt, "round_generation", None) or {}).get(opt.pending_batch_no)
    return insights._digest({"setup": insights.setup_key(opt),
                             "acquisition": bo_views.acquisition(opt),
                             "open": opt.pending_batch, "record": record,
                             "seed": int(opt.next_formulation_no),
                             "robust": bool(getattr(opt, "robust", False))})


def _gain_needs_no_fit(opt, mem):
    """Whether the open Round's figure needs no display-model fit: a cold
    Round, an exact one (one default fit of the suggestion model, as the
    Why fold makes), or a recomputed one whose display model is in the
    memo. With no Round open the figure needs the pool: never here."""
    r = opt.pending_batch_no if opt.pending_batch else None
    if r is None:
        return False
    gen = bo_views.generation(opt, r)
    if gen.regime != "warm" or gen.exact:
        return True
    use = [i for i in gen.rows if insights.measured(opt, insights.SCORE, i) is not None]
    return insights.model_key(opt, insights.SCORE, use) in mem


# _gain_key -> threading.Event, set once that figure is stored: a page
# rerun while an earlier run of the session still works the figure out
# (Streamlit lets the earlier run finish in its own thread) waits for it,
# or skips it, instead of a second search queued behind the first on
# RNG_LOCK (answer-key refresh item 2: 3.8 s in the second run).
_GAIN_RUNS = {}
_GAIN_RUNS_LOCK = threading.Lock()


def _gain_pending(opt, computed):
    """Whether Progress's expected-improvement line is about to be worked
    out here (the next Round's search, about 1.6 s at 60 x 4): no Round
    open, the figure neither in this session nor saved, and not withheld."""
    if opt.pending_batch or bo_views.acquisition(opt) == "qucb":
        return False
    hit = st.session_state.get(GAIN_KEY)
    if hit and hit[0] == _gain_key(opt):
        return False
    if bo_views.saved_next_gain(opt, st.session_state.get(NEXT_GAIN_KEY)) is not None:
        return False
    return withheld(opt, computed["fitted"].get(insights.SCORE),
                    computed.get("checks") or ()) is None


def latest_gain(opt, fit=True, wait=True):
    """§3.2's figure once per _gain_key (the data, the open Round and its
    record): the open Round's own (bo_views.latest_gain); with no Round
    open, the next Round's from today's rows (bo_views.next_gain under
    bo_views.next_model, the default fit ask() makes, final review I-1),
    worked out here under the lock, unless it was saved beside the project
    or worked out by another session (C1). `fit`
    False (Make a round's E3): only what is already worked out; None, not
    stored, otherwise. While another run works the same figure out:
    `wait` (a screen that shows it) waits for that run's answer; else None."""
    key = _gain_key(opt)
    hit = st.session_state.get(GAIN_KEY)
    if hit and hit[0] == key:
        return hit[1]
    mem = memo(opt)
    nxt = not (opt.pending_batch and opt.pending_batch_no is not None)
    if nxt:
        # Saved beside the project, or worked out by another session (C1):
        # neither a fit nor a search, nor a wait for another run.
        known = bo_views.saved_next_gain(opt, st.session_state.get(NEXT_GAIN_KEY))
        if known is not None:
            _keep_next_gain(opt, key, known)
            return known
    if not fit and not _gain_needs_no_fit(opt, mem):
        return None
    with _GAIN_RUNS_LOCK:
        running = _GAIN_RUNS.get(key)
        if running is None:
            _GAIN_RUNS[key] = done = threading.Event()
    if running is not None:
        if not wait:
            return None
        running.wait()
        hit = st.session_state.get(GAIN_KEY)
        if hit and hit[0] == key:
            return hit[1]
        # Its write did not land in this session (another session's run, or
        # one Streamlit had stopped): again here, its search from the cache.
        return latest_gain(opt, fit, wait)
    try:
        try:
            with insights.memo_in_use(mem):
                if not nxt:
                    g = bo_views.latest_gain(opt)
                else:
                    g = bo_views.next_gain_for(opt, st.session_state.get(NEXT_GAIN_KEY))
        except Exception:
            g = None
        if nxt and g is not None:
            _keep_next_gain(opt, key, g)
        else:
            st.session_state[GAIN_KEY] = (key, g)
    finally:
        # Stored first, then the runs waiting for it are let go.
        with _GAIN_RUNS_LOCK:
            _GAIN_RUNS.pop(key, None)
        done.set()
    return g


def _keep_next_gain(opt, key, g):
    """The next Round's figure for this session (GAIN_KEY) and for the
    file beside the project (NEXT_GAIN_KEY, written by save())."""
    st.session_state[GAIN_KEY] = (key, g)
    st.session_state[NEXT_GAIN_KEY] = (bo_views.next_gain_key(opt), g)


def gain_state(opt, score, checks, fit=True):
    """(why it is withheld or None, bo_views.RoundGain or None) for §3.2's
    line (review I4): "ucb" (no expected improvement), "verdict" (a
    caution or unreliable verdict), "skill" (no skill), "guard" (the noise
    guard); else the figure itself. `score`: the Overall score's display
    model (insights.Fitted) or None."""
    if bo_views.acquisition(opt) == "qucb":
        return "ucb", None
    why = withheld(opt, score, checks)
    if why is not None:
        return why, None
    return None, latest_gain(opt, fit)


def withheld(opt, score, checks):
    """Why the model's gains are not reported, or None (review I4): "verdict"
    (a caution or unreliable verdict), "skill" (the A3 gate: no skill, skill
    under 0.20, or fewer than 10 checks; 0.8.0 e2e fixes), "guard" (the
    noise guard). Progress's expected improvement and the Why fold's
    expected improvement and roles follow this one rule (fix round 1)."""
    h = insights.headline(opt, checks)
    if insights.verdict_state(opt, h, checks) in ("caution", "unreliable"):
        return "verdict"
    guard = insights.noise_guard(opt, score, checks)
    if "skill" in guard.reasons:
        return "skill"
    if guard.fires:
        return "guard"
    return None


def shown_gain(opt, score, checks, fit=True):
    """The expected improvement Progress shows, in points, or None where
    it shows none: what Next step weighs against the noise threshold (fix
    round 1, ledger ruling C1)."""
    why, g = gain_state(opt, score, checks, fit)
    if why is not None or g is None or g.why_none is not None or g.gain is None:
        return None
    return float(g.gain)


def gain_text(opt, computed, yard, standing=None):
    """§3.2's line (review I5): the expected improvement the open Round's
    first suggestion was chosen for, else the last Round's, named with its
    Round and compared with the noise threshold; withheld, saying why,
    under a caution or unreliable verdict, a no-skill record or the noise
    guard (review I4). Above the threshold while the predicted best is not
    higher (`standing`), it says the number comes from the model's
    uncertainty (C1). '' for a Round of the user's own formulations."""
    checks = computed.get("checks") or ()
    why, g = gain_state(opt, computed["fitted"].get(insights.SCORE), checks)
    if why == "ucb":
        return wording.GAIN_UCB
    if why is not None:
        gate = insights.skill_gate(opt, checks) if why == "skill" else None
        return wording.gain_unreliable(gain_reason(why, gate))
    if g is None or g.why_none == "own":
        return ""
    if g.why_none == "cold":
        return wording.gain_cold(g.round, values=not opt.has_ingredients())
    best = opt.best_index()
    measured = (f"{opt.Y_history[best]:.2f}" if bo_views.acquisition(opt) == "qlogei"
                and best is not None else None)
    # The next Round's figure is today's, never "recomputed" (final review I-1).
    recomputed = g.round is not None and not g.exact
    # Lines 1 and 2 above it have said whether the threshold is assumed.
    by = threshold_text(yard, assumed=False)
    if g.gain < 0.005:
        # Not "0.00 points … beats … by about 0.00 points" (T56 m3).
        return wording.gain_line_tiny(g.round, by, recomputed=recomputed)
    return wording.gain_line(f"{g.gain:.2f}", g.round, by,
                             above=g.gain > yard.value, recomputed=recomputed,
                             measured_text=measured,
                             predicted_none=standing is not None and not standing.higher)


# Next step's rows that end by asking for a Repeat of Best so far while the
# noise threshold is assumed (0.8.0 e2e fixes, review B2, I5).
_ASK_REPEAT = ("close", "explore", "see", "rising")


def gain_reason(why, gate=None):
    """withheld()'s reason in words for Progress's expected improvement:
    "skill" by its A3 gate (0.8.0 e2e fixes, review I4): too few checks,
    skill not clearly better than the average, or no skill."""
    if why == "skill" and gate is not None and gate.state == "wait":
        return wording.gain_reason_too_few(gate.n)
    if why == "skill" and gate is not None and gate.state == "unclear":
        return wording.gain_reason_unclear(skill_value(gate.fact.value), gate.fact.n)
    return {"verdict": wording.GAIN_REASON_VERDICT, "skill": wording.GAIN_REASON_SKILL,
            "guard": wording.GAIN_REASON_GUARD}[why]


def next_step_text(opt, step, rounds=None, amounts_shown=True):
    """(sentence, Set up button label or None): _next_step_text's, with
    the Repeat sentence appended to the close, explore, see and rising
    rows while the noise threshold is assumed (review B2, I5). A sentence,
    not a button. `amounts_shown`: "Show its amounts" is drawn under
    Progress figures (amounts_shown()); the untested row points at it only
    then."""
    text, change = _next_step_text(opt, step, rounds, amounts_shown)
    if step.kind in _ASK_REPEAT and getattr(step, "assumed", False):
        text = f"{text} {wording.NEXT_REPEAT_TO_MEASURE}"
    return text, change


def _next_step_text(opt, step, rounds=None, amounts_shown=True):
    """(sentence, Set up button label or None) for insights.next_step's
    answer (§2.4, review I2, I3, I14, M17; BO views I1; 0.8.0 e2e fixes,
    review I2). `rounds`: the verdict's Rounds, named in the misses row (03
    §5.4); its steady bias names its own Rounds."""
    values = not opt.has_ingredients()
    settings_only = not insights.has_ingredient_choice(opt)
    # A Round still open: record it first, as the foot says (final review M-1).
    left = len(open_rows(opt)) if opt.pending_batch else 0
    no = opt.pending_batch_no
    if step.kind == "misses":
        # Review I2: the intervals are too narrow; a steady bias of the
        # Overall score is added as a fact over its own Rounds, no advice.
        gap = getattr(step, "gap", None)
        bias = ("" if gap is None else
                wording.next_bias(f"{abs(gap):.1f}", below=gap < 0,
                                  rounds=getattr(step, "bias_rounds", None) or None))
        d = getattr(step, "diagnosis", None)
        if d is not None:
            # Review diagnosis: where the misses fell, and a Repeat to
            # measure the noise, in place of "the intervals are too narrow".
            best = opt.best_index()
            no_best = None if best is None else int(opt.formulation_ids[best])
            # Specific wording: the misses counted against the 1 in 20.
            return wording.next_misses_counted(
                rounds, d.kind, d.misses, d.checked, insights.allowed_misses(d.checked), no_best,
                bias_text=bias), None
        return wording.next_misses(rounds, bias_text=bias), None
    if step.kind == "repeat":
        return disagreement_text(step.disagreement), None
    if step.kind == "early":
        return (wording.next_too_early_open(no, left) if left else wording.NEXT_TOO_EARLY), None
    if step.kind == "no_skill":
        return wording.next_no_skill(skill_value(step.skill.value), step.skill.n, no,
                                     left, rounds=step.skill.rounds), None
    if step.kind == "untested":
        return wording.next_untested(values, open_round=bool(opt.pending_batch),
                                     round_no=no, shown=amounts_shown), None
    if step.kind == "explore":
        return wording.next_explore(no, left), None
    if step.kind == "close":
        text, setup = _close_text(opt, step, settings_only)
        if step.gain is not None and step.noise is not None:
            # Specific wording: the largest expected gain against the noise
            # threshold it fell under, in place of "No gain beyond noise".
            text = text.replace(wording.NO_GAIN_HEAD, wording.no_gain_head(
                f"{step.gain:.1f}", f"{step.noise:.1f}"), 1)
        return text, setup
    if step.kind == "rising":
        return wording.next_rising(_wide_names(getattr(step, "wide", None) or []), no, left), None
    return (wording.next_see_open(no, left) if left else wording.NEXT_SEE), None


def _close_text(opt, step, settings_only):
    """The close step's (sentence, Set up label): every binding limit named."""
    b = step.binding
    if b is None:
        return wording.next_close_to_top(settings_only=settings_only), None
    setup = (wording.CHANGE_LIMITS_ON_SETUP if b.kind == "limit"
             else wording.change_on_setup(b.name, highest=b.kind == "highest"))
    # Round-2 fixes (P3 #18): every binding limit named, as What next?
    # names them, so Make a round's "?" and the Review agree.
    try:
        every = widen_all_text(opt, step)
    except Exception:
        every = None
    if every:
        return f"{wording.NO_GAIN_HEAD} {every}", setup
    if b.kind == "limit":
        return (wording.next_close_to_top(wording.widen_limit(b.text), settings_only),
                setup)
    change = (wording.raise_highest(b.name) if b.kind == "highest"
              else wording.lower_lowest(b.name))
    return wording.next_close_to_top(change, settings_only), setup


def amounts_shown(facts):
    """Whether Progress figures draws "Show its amounts" for the untested
    row (_stand: not while the claims are withheld). The untested row
    fires only past the skill gate with no caution or unreliable verdict,
    so what can still hide them is the noise guard's floor or noise
    trigger, which status_facts reads under Reliable."""
    reasons = getattr(getattr(facts, "guard", None), "reasons", None) or ()
    return not any(r in reasons for r in ("floor", "noise"))


# ------------------------------------------------------------------ #
#  Wave C: More charts
# ------------------------------------------------------------------ #

# The two multiselects (a project's own rows and measurements) and the
# triangle's cached grid, keyed by setup_key, the three names and the start.
PAIR_KEY = "insights_pair"
THREE_KEY = "insights_three"
MIXTURE_KEY = "_insights_mixture"

_TRI_H = math.sqrt(3) / 2
_MIX_SIZE = 380          # px, the longer side of the drawn region plus its pad: it keeps its shape
_MIX_PAD = 0.12          # of the region's longer side, around it for the labels


def _keep_all_offered(key, options):
    """A multiselect's picks, minus any no longer offered (a deletion or a
    Set up change), before the box is drawn."""
    if key in st.session_state:
        kept = [v for v in st.session_state[key] if v in options]
        if kept != list(st.session_state[key]):
            st.session_state[key] = kept


GRID_BASE = insights.GRID_STEPS


def suggested_numbers(opt):
    """The open Round's generated rows still to make, in order (review I8:
    a start for the layers only)."""
    return [int(r['formulation']) for r in open_rows(opt)
            if not r.get('note') or r.get('original_recipe')]


def anchor_recipe(opt, number):
    """A start's make-up at the Formulation total: a recorded formulation's
    (start_recipe) or a suggestion's still to make."""
    ids = [int(x) for x in opt.formulation_ids]
    if number in ids:
        return insights.start_recipe(opt, ids.index(number))
    row = next(r for r in opt.pending_batch or [] if int(r['formulation']) == number)
    return insights.at_total(opt, row['recipe'])


def layers_offered(opt, fitted):
    """The three layers (review I6), or fewer: the Predicted score layer
    alone without a model yet, and under the noise guard's (a) or (b);
    Predicted score and Uncertainty while the gains are otherwise withheld
    — the A3 gate, or a caution or unreliable verdict, as Progress withholds
    its expected improvement (fix round 1). The Uncertainty layer follows
    the explored–unexplored map's rule (explore_withheld; review R-I16,
    which overrides e2e I4 for this layer only); Expected improvement stays
    withheld."""
    rows = [i for i in range(len(opt.Y_history)) if opt.fully_measured(i)]
    if len(rows) < insights.MIN_ROWS:
        return (wording.LAYER_SCORE,)
    rule = explore_withheld(opt, (fitted or {}).get(insights.SCORE),
                            checks_for(opt, "trust") or ())
    if rule == "hidden":
        return (wording.LAYER_SCORE,)
    if opt.pending_batch and opt.pending_batch_no is not None:
        if bo_views.generation(opt, opt.pending_batch_no).regime != "warm":
            return (wording.LAYER_SCORE,)
    if rule == "not_yet":
        return (wording.LAYER_SCORE, wording.LAYER_SD)
    if bo_views.acquisition(opt) == "qucb":
        # qUCB has no expected improvement: its value is an optimistic
        # score, not a gain (T56 I2).
        return (wording.LAYER_SCORE, wording.LAYER_SD)
    return _LAYERS


def layer_model(opt, fitted):
    """The suggestion model the layers show (spec §3.3; ftue R-I16): the
    open Round's, rebuilt (bo_views.generation; the display model of the
    same rows stands in when the generation is not exact); with no Round
    open, bo_views.next_model, the one the next Generate fits. Try one's
    and the triangle's Uncertainty layers and the Explored map read this
    one model (integration, W3's note 3)."""
    with insights.memo_in_use(memo(opt)):
        if not (opt.pending_batch and opt.pending_batch_no is not None):
            return bo_views.next_model(opt)
        rows = [i for i in range(len(opt.Y_history)) if opt.fully_measured(i)]
        gen = bo_views.generation(opt, opt.pending_batch_no)
        score = (fitted or {}).get(insights.SCORE)
        return bo_views.rebuild(opt, gen, score if gen.rows == rows else None)


# ------------------------------------------------------------------ #
#  Insights for a first-time user (spec 2026-10-01 rev. 2), W3: the
#  "Explored and unexplored" fold (§8, review R-I14 to R-I16). T0 stubs with
#  their final signatures; W3 fills them in.
# ------------------------------------------------------------------ #

# The fold's own keys: Next step's link to it (R-I14), the pointer (§8.3)
# and its one-shot request that the layer radios open on Uncertainty.
EXPLORED_LINK_KEY = "insights_explored_link"
POINTER_KEY = "insights_explored_pointer"
LAYER_PRESELECT_KEY = "_insights_layer_preselect"
TRY_ONE_KEY = "insights_try_one"
# The Next-step rows whose sentence points at untried amounts (R-I14).
_EXPLORED_ROWS = ("explore", "untested")
# At most this many limit-hit lines (R-I15 counts them; the screen stays sparse).
_LIMIT_LINES = 2


def _round_label(r):
    return wording.IMPORTED_NOTE if r is None else str(r)


def _strip_number(opt, strip, v, unit=True, d=None):
    """A strip's value as the screens write it: a process setting by
    fmt_setting, an amount to `d` decimals (else `_num`'s rule)."""
    if strip.setting:
        return fmt_setting(v, strip.unit if unit else "")
    text = _num(v) if d is None else f"{v:.{d}f}"
    return join_unit(text, strip.unit) if unit and strip.unit else text


def _gap_said(gap):
    """The values a gap's words name: the nearest tested value of an end
    gap, both ends of an interior one."""
    return ([gap.low] if gap.kind == "above" else [gap.high] if gap.kind == "below"
            else [gap.low, gap.high])


def _gap_text(opt, gap, strip, d=None):
    """'Coconut oil above 8.1 g', 'Salt below 0.6 g', 'Salt between 0.6 and
    1.0 g': an end gap names the nearest tested value. `d`: the decimals
    the whole line shares for this unit (strip_gap_lines); else the pair's
    own."""
    if d is None and not strip.setting:
        d = _shared_decimals(_gap_said(gap))
    if strip.setting:
        d = None
    if gap.kind == "above":
        return wording.gap_above(gap.name, _strip_number(opt, strip, gap.low, True, d))
    if gap.kind == "below":
        return wording.gap_below(gap.name, _strip_number(opt, strip, gap.high, True, d))
    return wording.gap_between(gap.name, _strip_number(opt, strip, gap.low, False, d),
                               _strip_number(opt, strip, gap.high, True, d))


def strip_gap_lines(opt, strips):
    """The plain lines above the strips (review R-I15): the largest untested
    stretches (at most 2), or that there is none; then each option never
    tested. Nothing while no formulation has all its results."""
    if not any(t.state == "tested" for s in strips for t in s.ticks) and not any(
            n for s in strips if s.options for n in s.options.values()):
        return []
    by_name = {s.name: s for s in strips}
    gaps = insights.strip_gaps(opt, strips)
    named = [g for g in gaps if g.kind != "option"]
    # One line, one decimal rule (integration, T0′): amounts in one unit
    # share their decimals across the line, not only within a pair.
    said = {}
    for g in named:
        strip = by_name[g.name]
        if not strip.setting:
            said.setdefault(strip.unit, []).extend(_gap_said(g))
    decimals = {unit: _shared_decimals(values) for unit, values in said.items()}
    stretches = [_gap_text(opt, g, by_name[g.name], decimals.get(by_name[g.name].unit))
                 for g in named]
    out = []
    if stretches:
        out.append(wording.strip_gaps_line(stretches))
    elif any(not s.fill and s.options is None for s in strips):
        out.append(wording.strips_no_gap(
            settings_only=not opt.has_ingredients(),
            has_settings=any(s.setting and not s.fill and s.options is None for s in strips)))
    out += [wording.option_never_tested(g.name, g.option) for g in gaps if g.kind == "option"]
    return out


def limit_hit_lines(opt, hits=None):
    """'3 of the 4 suggestions in Round 6 sit at Textured pea protein's
    Highest.' (review R-I15), most hits first, at most _LIMIT_LINES."""
    hits = insights.limit_hits(opt) if hits is None else hits
    out = []
    for h in hits[:_LIMIT_LINES]:
        b = h.binding
        text = (wording.limit_highest(b.name) if b.kind == "highest"
                else wording.limit_lowest(b.name) if b.kind == "lowest"
                else wording.limit_named(b.text))
        out.append(wording.limit_hit_line(h.hits, h.picks, h.round, text))
    return out


def explore_map_chart(pool_mean, pool_sd, tested, picks, ref=None, opt=None, gain=None,
                      muted_gain=False):
    """W3 (review R-I16): the mean × model-sd map both folds draw (Insights'
    Explored and unexplored, Make a round's Why these suggestions, whose
    ui_batch.why_chart is a thin caller): every sampled allowed formulation
    as a grey dot, the suggestion model's mean across and its sd up;
    `tested` ({"mean", "sd"}, with "no" for a tooltip) as rings; `picks`
    ({"mean", "sd", "no"}) as labelled diamonds; a dashed line, labelled,
    at `ref` (Best so far's predicted Repeat). `opt`: for the project's own
    word in the tooltip. None without a pool."""
    if pool_mean is None:
        return None
    pool = pd.DataFrame({"mean": list(map(float, pool_mean.tolist())),
                         "sd": list(map(float, pool_sd.tolist()))})
    if gain is not None:
        pool["gain"] = list(map(float, gain.tolist()))
    tested = pd.DataFrame(list(tested), columns=None if tested else ["mean", "sd"])
    mine = pd.DataFrame(list(picks), columns=None if picks else ["mean", "sd", "no"])
    say = (lambda t: t) if opt is None else (lambda t: _say(opt, t))
    # Review critique (fix 9): x zoomed to the scores that matter — the
    # upper three quarters of the cloud, the tested, the picks and the
    # reference — the rest clipped at the left edge.
    keep = [float(pool["mean"].quantile(0.75))]
    keep += list(mine["mean"]) if len(mine) else []
    if len(tested):
        keep.append(float(tested["mean"].max()))
    keep += [ref] if ref is not None else []
    x_lo = min(keep)
    x_hi = max([float(pool["mean"].max())] + keep)
    pad = 0.04 * max(x_hi - x_lo, 1.0)
    x = alt.X("mean:Q", title=say(wording.MAP_X_TITLE),
              scale=alt.Scale(zero=False, domain=[x_lo - pad, x_hi + pad], nice=False),
              axis=alt.Axis(tickCount=8))
    # From 0: the model's sd is a size, and a zoomed sd axis made a small
    # difference in certainty look large (review plots).
    y = alt.Y("sd:Q", title=say(wording.MAP_Y_TITLE), scale=alt.Scale(zero=True))
    word = say(wording.FORMULATION_CAP)
    ring_c, ring_o = mark("neutral")
    ink, _ = mark("bar_latest")
    # Review plots: one legend for the three kinds of mark (the caption
    # named them before), the ring and diamond drawn as on the chart.
    kinds = [say(wording.PLOT_ALLOWED), say(wording.PLOT_TESTED), say(wording.PLOT_SUGGESTION)]
    kind_colour = alt.Scale(domain=kinds, range=["#b8bcc4", ring_c, outline_colour()])
    kind_fill = alt.Scale(domain=kinds, range=["#b8bcc4", "transparent", outline_colour()])
    kind_shape = alt.Scale(domain=kinds, range=["circle", "circle", "diamond"])
    # Round-2 fixes (P2 #14): the marks' legend in the top-right corner.
    legend = alt.Legend(title=None, orient="top-right", labelLimit=0, fillColor=_page(),
                        padding=6, cornerRadius=3)
    pool = pool.assign(what=kinds[0])
    tested = tested.assign(what=kinds[1])
    mine = mine.assign(what=kinds[2])
    if len(mine):
        mine["bo"] = [wording.plot_bo_map(m, d) for m, d in zip(mine["mean"], mine["sd"])]
    if len(tested):
        tested["bo"] = [wording.plot_bo_map(m, d) for m, d in zip(tested["mean"], tested["sd"])]
    line = pd.DataFrame({"ref": [ref], "label": [wording.WHY_REFERENCE]})
    rule = [] if ref is None else [
        alt.Chart(line).mark_rule(strokeDash=[4, 3], color=ink, tooltip=False)
        .encode(x="ref:Q"),
        alt.Chart(line).mark_text(align="left", baseline="top", dx=5, color=muted(),
                                  fontSize=FONT_SMALL, tooltip=False)
        .encode(x="ref:Q", y=alt.value(4), text="label:N")]
    numbered = "no" in tested.columns
    ring_tip = ({"tooltip": [alt.Tooltip("no:Q", title=word, format="d"),
                             alt.Tooltip("mean:Q", title=wording.WHY_AXIS_MEAN, format=".1f"),
                             alt.Tooltip("sd:Q", title=wording.MODEL_SD, format=".2f"),
                             bo_tip()]}
                if numbered else {})
    if "gain" in pool.columns:
        # Review critique (fix 9): the cloud coloured by expected gain, a
        # blue ramp with its own legend; light where there is little to gain.
        lo_b, hi_b = _BLUES_EXTENT[theme()]
        cloud = (alt.Chart(pool.sort_values("gain")).mark_circle(size=16, opacity=0.75,
                                                                  tooltip=False, clip=True)
                 .encode(x=x, y=y, color=alt.Color(
                     "gain:Q", title=say(wording.MAP_GAIN_LEGEND),
                     # Muted when the gain is greyed (P2 #14), as its column is.
                     scale=alt.Scale(scheme=alt.SchemeParams(
                         "greys" if muted_gain else "blues",
                         extent=[0.1, 0.55] if muted_gain else [0.15, hi_b])),
                     legend=alt.Legend(orient="bottom", direction="horizontal",
                                       titleFontSize=FONT_AXIS_TITLE, gradientLength=160,
                                       format=".2f"))))
    else:
        cloud = (alt.Chart(pool).mark_point(size=12, opacity=0.6, tooltip=False, clip=True)
                 .encode(x=x, y=y, stroke=alt.Stroke("what:N", scale=kind_colour, legend=legend),
                         fill=alt.Fill("what:N", scale=kind_fill, legend=legend),
                         shape=alt.Shape("what:N", scale=kind_shape, legend=legend)))
    if "gain" in pool.columns:
        kinds = kinds[1:]
        kind_colour = alt.Scale(domain=kinds, range=[ring_c, outline_colour()])
        kind_fill = alt.Scale(domain=kinds, range=["transparent", outline_colour()])
        kind_shape = alt.Scale(domain=kinds, range=["circle", "diamond"])
    layers = [
        cloud,
        # The neutral token (polish round): gray at Vega's default 0.7 was
        # 2.94:1 on the dark page.
        alt.Chart(tested).mark_point(size=40, opacity=ring_o, strokeWidth=1.5, clip=True,
                                     **({} if numbered else {"tooltip": False}))
        .encode(x=x, y=y, stroke=alt.Stroke("what:N", scale=kind_colour, legend=legend),
                fill=alt.Fill("what:N", scale=kind_fill, legend=legend),
                shape=alt.Shape("what:N", scale=kind_shape, legend=legend), **ring_tip),
        *rule,
        # Charts proto: orange for suggestions, as on every chart here.
        alt.Chart(mine).mark_point(size=90, opacity=1)
        .encode(x=x, y=y, stroke=alt.Stroke("what:N", scale=kind_colour, legend=legend),
                fill=alt.Fill("what:N", scale=kind_fill, legend=legend),
                shape=alt.Shape("what:N", scale=kind_shape, legend=legend),
                tooltip=[alt.Tooltip("no:N", title=word),
                         alt.Tooltip("mean:Q", title=wording.WHY_AXIS_MEAN, format=".1f"),
                         alt.Tooltip("sd:Q", title=wording.MODEL_SD, format=".2f"),
                         *([bo_tip()] if len(mine) else [])]),
        *_pick_labels(mine, pool, tested, ref, x, y),
    ]
    return readable(titled(alt.layer(*layers).properties(height=300), say(wording.PLOT_MAP)))


_MAP_W, _MAP_H = 730, 300      # px: the map's plot, about (the column less its axis)


def _nice(lo, hi, count=10):
    """(lo, hi) widened to whole ticks, as Vega's nice scale does (d3's
    tick increment for about `count` ticks)."""
    span = hi - lo
    if span <= 0:
        return lo - 0.5, hi + 0.5
    raw = span / count
    power = 10 ** math.floor(math.log10(raw))
    err = raw / power
    step = power * (10 if err >= 50 ** 0.5 else 5 if err >= 10 ** 0.5 else 2 if err >= 2 ** 0.5
                    else 1)
    return math.floor(lo / step) * step, math.ceil(hi / step) * step
_LABEL_SPOTS = (("above", 0, -13, "center", "bottom"), ("left", -9, 0, "right", "middle"),
                ("right", 9, 0, "left", "middle"), ("below", 0, 13, "center", "top"))


def place_labels(points, others, xr, yr, line=None):
    """Review readable: for each labelled point ({"x", "y", "text"}, data
    units) the spot — above, left, right or below its mark — whose label
    box overlaps least the labels already placed, the marks in `points`
    and `others` and the vertical `line` (an x), in px on a _MAP_W ×
    _MAP_H plot over the ranges `xr`, `yr`. Returns the spot names."""
    def px(x, y):
        return ((x - xr[0]) / ((xr[1] - xr[0]) or 1) * _MAP_W,
                (1 - (y - yr[0]) / ((yr[1] - yr[0]) or 1)) * _MAP_H)

    def box(cx, cy, w, h, spot):
        _, dx, dy, align, base = spot
        x0 = cx + dx - (w / 2 if align == "center" else w if align == "right" else 0)
        y0 = cy + dy - (h / 2 if base == "middle" else h if base == "bottom" else 0)
        return x0, y0, x0 + w, y0 + h

    def overlap(a, b):
        return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))

    marks = [px(p["x"], p["y"]) for p in list(points) + list(others)]
    marks = [(mx - 6, my - 6, mx + 6, my + 6) for mx, my in marks]
    lx = None if line is None else px(line, yr[0])[0]
    placed, out = [], []
    for p in points:
        cx, cy = px(p["x"], p["y"])
        w, h = 7 * len(str(p["text"])) + 2, 13
        best = None
        for k, spot in enumerate(_LABEL_SPOTS):
            b = box(cx, cy, w, h, spot)
            cost = sum(overlap(b, o) for o in placed + marks) + k * 0.5
            if lx is not None and b[0] - 2 <= lx <= b[2] + 2:
                cost += 40
            if b[0] < 0 or b[2] > _MAP_W or b[1] < 0 or b[3] > _MAP_H:
                cost += 200
            if best is None or cost < best[0]:
                best = (cost, spot, b)
        placed.append(best[2])
        out.append(best[1][0])
    return out


def _pick_labels(mine, pool, tested, ref, x, y):
    """The suggestions' numbers, each at its place_labels spot (one text
    layer per spot)."""
    if not len(mine):
        return []
    xs = list(pool["mean"]) + list(tested.get("mean", [])) + list(mine["mean"])
    ys = list(pool["sd"]) + list(tested.get("sd", [])) + list(mine["sd"]) + [0.0]
    pts = [{"x": m, "y": d, "text": n} for m, d, n in zip(mine["mean"], mine["sd"], mine["no"])]
    near = [{"x": m, "y": d} for m, d in zip(tested.get("mean", []), tested.get("sd", []))]
    spots = place_labels(pts, near, _nice(min(xs), max(xs)), _nice(min(ys), max(ys)), ref)
    frame = mine.assign(spot=spots)
    out = []
    for name, dx, dy, align, base in _LABEL_SPOTS:
        part = frame[frame["spot"] == name]
        if len(part):
            out.append(alt.Chart(part).mark_text(dx=dx, dy=dy, align=align, baseline=base,
                                                 color=muted(), fontSize=FONT_LABEL,
                                                 tooltip=False)
                       .encode(x=x, y=y, text="no:N"))
    return out


def explore_withheld(opt, score, checks):
    """The map's and the Uncertainty layer's one rule (review R-I16):
    "hidden" under the noise guard's (a) correlation length at its floor
    or (b) fitted noise well under the Repeat SD — the model sd is high
    everywhere but at the data, so the map would show everything as
    unexplored; "not_yet" under the A3 gate or a caution or unreliable
    verdict (shown, with MAP_NOT_YET); None when nothing is withheld."""
    guard = insights.noise_guard(opt, score, checks)
    if "floor" in guard.reasons or "noise" in guard.reasons:
        return "hidden"
    return None if withheld(opt, score, checks) is None else "not_yet"


def _map_model(opt, fitted):
    """The suggestion model the map shows (review R-I16): layer_model's,
    the one the layers show; with a Round open whose model cannot be
    rebuilt (a cold Round), the one the next Generate fits. None while
    cold."""
    model = layer_model(opt, fitted)
    if model is None and opt.pending_batch:
        return bo_views.next_model(opt)
    return model


def _why_link_offered(opt):
    """Review R-I4's condition for the Why link, which outranks this fold's
    link: a warm Round open still holding at least one generated pick (one
    rule, which _why_link draws by)."""
    if not opt.pending_batch or opt.pending_batch_no is None:
        return False
    try:
        gen = bo_views.generation(opt, opt.pending_batch_no)
        own, _ = bo_views._picks(opt, gen, list(opt.pending_batch))
    except Exception:
        return False
    return gen.regime == "warm" and not all(own)


def explored_link(opt, step):
    """R-I14: Next step's link to this fold on its explore and untested rows,
    the one tertiary link (R-I4: only when the Why link is not shown)."""
    if step is None or step.kind not in _EXPLORED_ROWS or _why_link_offered(opt):
        return False
    # Lands on the fold's keyed place (ftue fix C), so the opened fold's
    # strips are in view, not under the fold line.
    link(_say(opt, wording.explored_link(not opt.has_ingredients())), wording.TAB_INSIGHTS,
         section=EXPLORED_PLACE, fold=TRIED_KEY, key=EXPLORED_LINK_KEY)
    return True


# ------------------------------------------------------------------ #
#  Charts proto (2026-10-01): What you've tried (parallel coordinates),
#  How the app chooses (the classic figure along one row through Best so
#  far), the triangle, then the map — all in Explored and unexplored.
# ------------------------------------------------------------------ #

# Score colours, low to high, per theme: stops of viridis (a perceptual,
# colour-blind-safe sequential scheme whose hue moves as well as its
# lightness, so neighbouring scores separate better than one blue did).
# Each end is cut so every stop reaches 3:1 on its theme's page: light runs
# viridis 0.5 → 0 (teal #21918c, 3.8:1 on white, to purple #440154,
# 15.2:1); dark runs viridis 0.4 → 0.9 (#2a788e, 3.8:1 on #0e1117, to
# lime #bddf26, 12.4:1). Higher reads stronger in each theme. The top is
# cut before viridis' yellow, which sits too near the suggestions' orange.
_SCORE_RAMP = {
    "light": ["#21918c", "#2a788e", "#355f8d", "#414487", "#482475", "#440154"],
    "dark": ["#2a788e", "#21918c", "#22a884", "#44bf70", "#7ad151", "#bddf26"],
}
# The model's 95 % band (a region: not held to 3:1, as _FILLS) and the
# expected gain's line and area. Blue means "the model's uncertainty or
# gain" on every chart here (the triangle's Uncertainty and Expected
# improvement layers use blues()). Light #2f6db5 is 5.2:1 on white; dark
# #7fb0e6 is 8.3:1 on #0e1117.
_MODEL_BLUE = {"light": "#2f6db5", "dark": "#7fb0e6"}
_BAND_FILL = {"light": ("#2f6db5", 0.18), "dark": ("#7fb0e6", 0.22)}
_ZONE_FILL = {"light": "#c9ced6", "dark": "#4a515c"}   # within noise of the top
_MANY_LINES = 30       # past this many lines, thinner and semi-transparent
CHOOSES_KEY = "insights_chooses_axis"
_PROTO_MODEL = "_insights_proto_model"


def score_colours(which=None):
    """The Overall score's colour stops, low to high, in the reader's theme
    or `which`."""
    return list(_SCORE_RAMP[which or theme()])


def score_scale(domain):
    return alt.Scale(range=score_colours(), domain=list(domain), interpolate="rgb")


def model_blue(which=None):
    return _MODEL_BLUE[which or theme()]


def _axis_label(axis):
    return f"{axis.name} ({axis.unit})" if axis.unit else axis.name


def _axis_end(opt, axis, value, d=None):
    """An axis value as the screens write it; `d`: decimals shared by the
    axis' two ends."""
    if axis.setting:
        return fmt_setting(value, axis.unit)
    return join_unit(_num(value) if d is None else f"{value:.{d}f}", axis.unit)


def _wrap(text, width=14):
    """A label broken into lines of about `width` characters, joined by a
    newline the axis splits on."""
    out, line = [], ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    out.append(line)
    return "\n".join(out)


# Review four polish (owner: the score colours "all look very similar"):
# Q4 coloured by role. Every other tested line thin and light grey (context,
# so an untested stretch reads as empty space); the next best few mid-grey;
# Best so far thick ink; this Round's suggestions orange. (light grey, mid-
# grey) per theme: light #cfd3d9 / #7d838e on white; dark #444a54 / #9aa0aa
# on #0e1117 — the mid-grey and the ink reach 3:1, the context lines need not.
_TRIED_GREYS = {"light": ("#cfd3d9", "#7d838e"), "dark": ("#444a54", "#9aa0aa")}


# ---- How the app chooses ------------------------------------------ #

CHOOSES_MODE_KEY = "insights_chooses_mode"
CHOOSES_LINE, CHOOSES_ONE = "line", "one"


def suggestion_model(opt, fitted):
    """_map_model's model, kept once per data and Round in the session, so
    the chart, the triangle's layers and the map fit it once a rerun."""
    key = (insights.data_key(opt), insights.setup_key(opt), opt.pending_batch_no,
           len(opt.pending_batch or []))
    hit = st.session_state.get(_PROTO_MODEL)
    if hit and hit[0] == key:
        return hit[1]
    model = _map_model(opt, fitted)
    st.session_state[_PROTO_MODEL] = (key, model)
    return model


def chooses_reason(opt, fitted):
    """Why the profiler's Expected gain row is greyed or not drawn, in
    words, or None. Not drawn: a cold Round, an upper-bound acquisition.
    Greyed (review four polish): gain_grey's reasons — the Predictions
    card's word short of Reliable, or the noise guard."""
    if opt.pending_batch and opt.pending_batch_no is not None and \
            bo_views.generation(opt, opt.pending_batch_no).regime != "warm":
        return wording.CHOOSES_REASON_COLD
    if bo_views.acquisition(opt) == "qucb":
        return wording.CHOOSES_REASON_UCB
    why = gain_grey(opt, (fitted or {}).get(insights.SCORE), checks_for(opt, "trust") or ())
    return None if why is None else _GREY_REASON[why]


def gain_grey(opt, score, checks):
    """Review four polish: why Expected gain is greyed, or None — the
    Predictions card's word when it is not Reliable ("not_yet" or "off",
    insights.score_predictions: the Overall score's record, ruling 1), else
    "noise" when the noise guard fires (a skill gate the card's word has
    passed reads "not_yet"). The profiler's gain row and Make a round's Why
    these both read this one rule; the suggestions are made the same way
    regardless."""
    word = insights.score_predictions(opt, checks)
    if word != "reliable":
        return word
    guard = insights.noise_guard(opt, score, checks)
    if any(r in guard.reasons for r in insights._NOISE_TRIGGERS):
        return "noise"
    return "not_yet" if guard.fires else None


def gain_greyed_text(opt, checks, reason):
    """Specific wording: Make a round's greyed-gain line with the fact
    behind the Predictions word — the Overall score's checks against the 10
    it needs, its misses, or its skill against 0.20 — else the word's own
    line (GAIN_GREYED)."""
    fallback = wording.why_gain_greyed(reason)
    if reason not in ("not_yet", "off") or not checks:
        return fallback
    checks = list(checks)
    h = insights.headline(opt, checks)
    verdict = insights.score_verdict_state(opt, h, checks)
    fact = insights.skill(opt, checks)
    checked = sum(1 for c in checks if c.name == insights.SCORE and c.fully_measured)
    row = insights.score_prediction_row(opt.complete_count(), verdict, fact, checked)
    r = h.score
    if row in ("off_coverage", "caution") and r is not None and r.total:
        return wording.gain_greyed_missed(r.total - r.inside, r.total)
    if row == "too_few" and checked < insights.SKILL_MIN:
        return wording.gain_greyed_checks(checked, insights.SKILL_MIN)
    if row in ("unclear", "no_skill") and fact is not None:
        return wording.gain_greyed_skill(skill_value(fact.value),
                                         f"{insights.SKILL_SMALL:.2f}")
    return fallback


_GREY_REASON = {"noise": wording.GAIN_REASON_GUARD, "not_yet": wording.GAIN_REASON_NOT_YET,
                "off": wording.GAIN_REASON_OFF}


def _first_pick(opt):
    """(number, make-up at the total) of the open Round's first suggestion
    still to make, or None."""
    numbers = suggested_numbers(opt)
    if not numbers:
        return None
    return numbers[0], anchor_recipe(opt, numbers[0])


# Review polish: the hero's two charts' y axes (ticks and labels) one width.
_HERO_Y_EXTENT = 36


# How many combinations _three_default tries before it settles for the
# first: each try is a three-point slice, a few milliseconds.
_THREE_TRIES = 20


def repeat_of_best(opt, fitted):
    """Best so far's predicted Repeat (binding speed-spec I7): the predicted
    best's when it is worked out, else the display model's at its amounts.
    The one reference for the Predicted above Best so far outline and the
    Why map's dashed line (answer-key refresh item 6)."""
    hit = st.session_state.get(BEST_KEY)
    if hit and hit[0] == insights.setup_key(opt) and isinstance(hit[1], insights.PredictedBest) \
            and hit[1].repeat is not None:
        return hit[1].repeat.mean
    best = opt.best_index()
    if best is None:
        return None
    [p] = insights.predict(opt, fitted[insights.SCORE],
                           [opt._encode(insights.start_recipe(opt, best))])
    return p.mean


def _how(opt, computed):
    """Method and definitions (spec 2026-10-01 rev. 2 §2, §3, §7; review
    M3, M12): Model and method read off the project's own model, with its
    known limits; How each status is decided; the plain paragraphs, the
    details table with its window named and the lines that read it; the
    glossary last. Everything a tooltip says is here too (M12)."""
    with st.container(key=HOW_PLACE), \
            st.expander(_say(opt, wording.FOLD_METHOD), key=HOW_KEY,
                        on_change="rerun") as fold:
        # Keyed and worked out only while open (T56 I5; binding I15).
        if not fold.open:
            return
        _model_and_method(opt, computed)
        _how_learned(opt, computed)
        _status_table(opt, computed)
        _how_details(opt, computed)
        _glossary(opt)


def _how_details(opt, computed):
    """Method and definitions' paragraphs and details table (0.8.0 §2.5;
    review M5, M6, M10): plain paragraphs, the two method lines that were
    under the verdict (audit V23), then the details table with its window
    named and a "Likely range widened by" column, and the lines that read
    it."""
    for text in how_paragraphs(opt, computed):
        st.markdown(_say(opt, text))
    if computed is None:
        return
    checks = computed["checks"]
    skipped = insights.rounds_not_checked(opt)
    if skipped and checks:
        st.caption(_say(opt, wording.rounds_not_checked(skipped)))
    frame, config = details_frame(opt, computed)
    if frame is None:
        st.caption(_say(opt, wording.DETAILS_NONE))
        return
    h = insights.headline(opt, checks)
    st.caption(_say(opt, wording.details_window(insights._ordered_rounds(checks),
                                                len(h.rounds), h.rounds)))
    st.dataframe(frame, hide_index=True, height=table_height(len(frame)),
                 column_config=config)
    for text in how_notes(opt, computed):
        st.caption(_say(opt, text))


def how_paragraphs(opt, computed):
    """Method and details' paragraphs above the table: the models with the
    kernel in use (0.8.0 e2e fixes D5), the interval, the checks, the
    verdict rule with this project's own example (review M11), the noise
    threshold and the rest. `computed` None: nothing is worked out yet."""
    cfg = getattr(opt, "bo_config", None) or {}
    tiny = cfg.get("noise") == "fixed_tiny"
    verdict = wording.HOW_VERDICT
    checks = (computed or {}).get("checks")
    if checks:
        h = insights.headline(opt, checks)
        if h.total:
            verdict = wording.with_left_out(
                wording.how_verdict(h.total, insights.allowed_misses(h.total)),
                getattr(h, "no_forecast", 0))
    return [wording.how_models(cfg.get("kernel"), cfg.get("noise")),
            wording.HOW_RANGE_FIXED_TINY if tiny else wording.HOW_RANGE,
            wording.how_checks(values=not opt.has_ingredients(),
                               total=opt.has_formulation_total()),
            verdict, wording.HOW_NOISE_THRESHOLD, wording.HOW_REGRESSION,
            wording.how_same_process(values=not opt.has_ingredients()),
            wording.HOW_NO_P, wording.HOW_FORECAST_WORDS]


def _details_rows(opt, computed):
    checks = computed["checks"]
    return insights.details(opt, checks, computed["fitted"],
                            insights.disagreeing_repeat(opt, checks))


# The details table's columns, in order (0.8.0 e2e fixes, review B1, B2,
# M8): the mean error beside the median absolute error, the fitted noise SD
# last, as context, so Error ÷ Repeat noise and Skill stay in view.
DETAILS_ORDER = (*wording.DETAILS_COLUMNS_2[:5], wording.MEAN_ERROR_COLUMN,
                 *wording.DETAILS_COLUMNS_2[5:], wording.SKILL_COLUMN,
                 wording.FITTED_NOISE_COLUMN)


def details_frame(opt, computed, rows=None):
    """(the details table, its column_config) or (None, {}) with no row.
    The mean error and fitted noise SD columns (review B1, B2) sit after
    Skill's neighbours so Error ÷ Repeat noise and Skill stay in view at
    1280 px; every column is narrow, with its full meaning in its tooltip
    (D2, review M8: wording.DETAILS_HELP)."""
    checks = computed["checks"]
    rows = _details_rows(opt, computed) if rows is None else rows
    if not rows:
        return None, {}
    columns = wording.DETAILS_COLUMNS_2
    frame = pd.DataFrame([{
        columns[0]: panel_label(opt, r.name),
        columns[1]: r.checked,
        columns[2]: r.inside,
        columns[3]: f"± {_num(r.half_width)}",
        columns[4]: _num(r.typical_miss),
        wording.MEAN_ERROR_COLUMN: mean_error_text(opt, r),
        columns[5]: _say(opt, widen_cell(opt, r, checks)),
        columns[6]: wording.repeat_compare(r.ratio, r.repeat_sd, r.pairs, r.left_out),
        # Each output's own skill (spec 2026-09-30-bo-views §2.1, I2).
        wording.SKILL_COLUMN: skill_cell(opt, checks, r.name),
        wording.FITTED_NOISE_COLUMN: fitted_noise_text(opt, r),
    } for r in rows], columns=list(DETAILS_ORDER))
    config = {c: st.column_config.Column(_say(opt, c), help=details_help(opt, c), width="small")
              for c in frame.columns if c in wording.DETAILS_HELP}
    return frame, config


# The details columns that name a statistic, with its standard term after
# the plain help (§3, R-I6).
_DETAILS_TERMS = {
    wording.DETAILS_COLUMNS_2[1]: wording.TIP_ROUND_AHEAD,
    wording.DETAILS_COLUMNS_2[2]: wording.TIP_INTERVAL,
    wording.MEAN_ERROR_COLUMN: wording.TIP_MEAN_ERROR,
    wording.SKILL_COLUMN: wording.TIP_SKILL,
}


def details_help(opt, column):
    """A details column's "?": its plain help (wording.DETAILS_HELP), then
    the BO term of the statistic it shows, where it shows one."""
    term = bo_term(_DETAILS_TERMS[column]) if column in _DETAILS_TERMS else ""
    return _say(opt, " ".join(t for t in (wording.DETAILS_HELP[column], term) if t))


def how_notes(opt, computed, rows=None):
    """The captions under the details table: how each column is read, the
    steady bias both ways and what the Round-ahead check cannot see (review
    B1, M2), the fitted noise as context (B2), skill, the expected
    improvement and the role rule."""
    checks = computed["checks"]
    rows = _details_rows(opt, computed) if rows is None else rows
    # The Overall-score row carries a second factor when its cell is
    # more than its own fit's factor (the sentence's words are wording's).
    top = any(r.name == insights.SCORE and r.widen is not None
              and widen_cell(opt, r, checks) != wording.widen_text(r.widen)
              for r in rows)
    texts = [wording.HOW_WIDENED, wording.HOW_WIDENED_SCORE if top else "",
             wording.HOW_REPEATS_COMPARED, wording.HOW_THRESHOLDS, wording.HOW_BIAS,
             wording.HOW_BIAS_DIRECTION, wording.HOW_CHANGE_LIMIT, wording.HOW_FITTED_NOISE,
             how_skill_text(settings_only=not insights.has_ingredient_choice(opt)),
             wording.HOW_EXPECTED_IMPROVEMENT,
             wording.why_role_rule(values=not opt.has_ingredients(),
                                   total=bool(opt.formulation_total))]
    return [t for t in texts if t]


def how_texts(opt, computed):
    """Every paragraph and caption Method and details says, in order, for
    a check of its words (the table's window line aside)."""
    out = how_paragraphs(opt, computed)
    if computed is not None and computed.get("checks"):
        out += how_notes(opt, computed)
    return out


def mean_error_text(opt, r):
    """A details row's Mean error cell (review B1): signed, in the Overall
    score's tenths of a point or the measurement's own decimals (at least
    one), marked "(bias)" when it is a steady bias; '' without one."""
    if r.mean_error is None:
        return ""
    d = 1 if r.name == insights.SCORE else max(1, decimals(opt, r.name))
    return wording.mean_error_cell(wording.signed(r.mean_error, d), bias=r.bias)


def fitted_noise_text(opt, r):
    """A details row's Fitted noise SD cell (review B2): the model's own
    estimate in the output's units, context only; "fixed near zero" under
    the fixed_tiny model setting; '' without a model."""
    if r.noise_fixed:
        return wording.fitted_noise_cell(None)
    if r.fitted_noise is None:
        return ""
    d = 1 if r.name == insights.SCORE else max(1, decimals(opt, r.name))
    return wording.fitted_noise_cell(f"{r.fitted_noise:.{d}f}")


def how_skill_text(settings_only=False):
    """Method and details' skill paragraph, with the no-skill cut stated
    plainly (fix round 1, T2 I-1; 0.8.0 e2e fixes D4: no percentile table)."""
    return wording.how_skill(skill_value(insights.skill_cut(insights.SKILL_MIN)), settings_only)


def skill_cell(opt, checks, name):
    """The details table's Skill cell: '0.36 (n = 15)', or 'under 10 values'."""
    fact = insights.skill(opt, checks, name)
    return (wording.SKILL_UNDER_TEN if fact is None
            else wording.skill_cell(skill_value(fact.value), fact.n))


def widen_cell(opt, r, checks):
    """A details row's "Likely range widened by" cell: the current fit's
    widening, and for the Overall score also the widening every range on
    screen carries on top of it (insights.column_factor; ruling M5, final
    review I-2) when that is more than ×1.0."""
    if r.widen is None:
        return ""
    if r.name == insights.SCORE:
        top = insights.column_factor(opt, checks)
        if wording.widen_text(top) != wording.widen_text(1.0):
            return wording.score_widen_text(r.widen, top)
    return wording.widen_text(r.widen)


# ------------------------------------------------------------------ #
#  Insights for a first-time user (spec 2026-10-01 rev. 2), W2: the expert
#  layer (§3, §7, review R-B2, R-I6, R-I12, R-I13). T0 stubs with their
#  final signatures; W2 fills them in.
# ------------------------------------------------------------------ #

# The lines a tooltip is asked for by (W1's and W3's draw helpers pass one).
TIP_LINES = (
    "predictions", "progress", "next",                        # the status lines
    "basis_predictions", "verdict", "score_line", "steady_bias", "skill", "disagreement",
    "basis_progress", "stand_line", "repeats", "tie", "standing", "limits", "luck", "gain",
    "basis_next", "next_reasoning",
    "basis_explored", "strips", "gaps", "limit_hits", "map",
)


def bo_term(tip):
    """A tooltip's standard term alone, from "BO term: " on; '' without one."""
    at = tip.find("BO term: ")
    return tip[at:] if at >= 0 else ""


def _joined(opt, texts):
    """One help text in the project's own words, or None for nothing."""
    text = " ".join(t for t in texts if t)
    return _say(opt, text) if text else None


def _verdict_names(opt, h):
    """The measurements the verdict's Rounds checked, in Results' order:
    from the checks already worked out, else every measurement."""
    names = [o['name'] for o in opt.measurements_by_importance()]
    try:
        checks = cached_checks(opt)
    except Exception:   # outside a session: every measurement
        checks = ()
    window = {c.name for c in checks if h is not None and c.round in h.rounds}
    return [n for n in names if n in window] if window else names


def _coverage_tip(opt, h):
    """The measurements' coverage with its window and numbers (tip_coverage
    and its early and none forms)."""
    if h is None or h.state == "none":
        return wording.TIP_COVERAGE_NONE
    names = _verdict_names(opt, h)
    left = getattr(h, "no_forecast", 0)
    if h.state == "early":
        return wording.tip_coverage_early(names, h.rounds, h.total, left)
    return wording.with_left_out(wording.tip_coverage(names, h.rounds, h.inside, h.total,
                                                      insights.allowed_misses(h.total)), left)


def _skill_tip(opt, facts):
    """The Overall score's skill with its window and n, or the count it
    still needs."""
    fact = facts.skill
    if fact is not None:
        return wording.tip_skill(fact.rounds, skill_value(fact.value), fact.n)
    gate = getattr(facts.guard, "gate", None)
    if gate is not None and gate.state == "wait":
        n = gate.n
    elif getattr(facts, "checked", None) is not None:
        # Fix round 4: skill()'s own count, every fully measured check
        # (status_facts' checked), the no-forecast ones too.
        n = facts.checked
    else:
        h = facts.headline
        n = _score_checks(h.score) if h is not None and h.score is not None else 0
    return wording.tip_skill_too_few(n)


def _score_checks(record):
    """Every fully measured check behind an Overall-score record: its
    inside count's plus the ones with no forecast (the mean error's n)."""
    return record.total + getattr(record, "no_forecast", 0)


def _repeat_tip(d):
    return wording.tip_repeat_off(d.later, d.earlier, f"{abs(d.diff):.1f}",
                                  "" if d.limit is None else f"{d.limit:.1f}")


def _bias_tip(facts):
    h = facts.headline
    if facts.bias is None or h is None or h.score is None:
        return ""
    # Fix round 4: n is the mean's own, every check (no-forecast ones too).
    return wording.tip_bias(wording.signed(facts.bias, 1), _score_checks(h.score), h.score.rounds)


def _wide_label(name):
    return wording.OVERALL_SCORE_COLUMN if name == insights.SCORE else name


# --- review diagnosis (display only, 2026-10-02) ---
def diag_items(opt, d):
    """The misses by measurement as wording.diag_item says them, most first;
    the mean error to each measurement's own decimals, at least one."""
    return [wording.diag_item(m.name, m.misses, m.checked, m.above, m.below,
                              wording.signed(m.mean_error, max(1, decimals(opt, m.name))))
            for m in d.items]


def _diag_score(d):
    """(inside, total) of the Overall score when it is as expected, else None."""
    return (d.score.inside, d.score.total) if d is not None and d.score_ok else None


def diag_plain_text(d):
    return wording.diag_plain(d.kind, d.far, d.misses, _diag_score(d), d.far_checked,
                              d.checked, d.one_way, d.side)


def diag_tip(opt, d):
    """The expert line: n misses / n checked by measurement, side and mean
    error, how many on new ground, and the rule by name."""
    from food_bo import FAR_FROM_TESTED
    return wording.tip_diag(diag_items(opt, d), d.rounds, d.far, d.misses, d.far_checked,
                            d.checked, f"{100 * FAR_FROM_TESTED:.0f}", d.kind,
                            score=None if d.score is None else (d.score.inside, d.score.total),
                            score_rounds=None if d.score is None else d.score.rounds)


def scatter_check(opt, facts):
    """Ruling 2: the CV-widening check (wide_cause) as the card may say it.
    None when the rule does not fire; else (RepeatsVary, None) when the
    Repeats show large noise (the panel advice is backed), or (None, N)
    with N the Formulation to repeat to check noise (Best so far)."""
    fire = any(c[0] == "wide" and wide_cause(c[1], getattr(facts, "wide_factors", None) or {})
               is not None for c in facts.clauses)
    if not fire:
        return None
    vary = insights.repeats_vary(opt, getattr(facts, "yardstick", None))
    if vary is not None:
        return vary, None
    best = getattr(facts, "best", None)
    return None, (best.formulation if best is not None else None)


def scatter_tail(opt, facts, short=False):
    """The sentence the wide clause ends with: the consistency advice and
    its Repeat evidence when backed; else the Repeat that would measure the
    noise; '' when the rule does not fire."""
    found = scatter_check(opt, facts)
    if found is None:
        return ""
    vary, no = found
    if vary is not None:
        if short:
            return f" {wording.SHORT_CHECK_SCATTER}"
        return (f" {wording.CHECK_SCATTER} "
                f"{wording.repeats_vary_line(f'{vary.sd:.1f}', vary.df, f'{vary.spread:.1f}')}")
    if no is None:
        return ""
    if short:
        return f" {wording.short_add_repeat(no)}"
    ask = wording.next_plain_repeat_ask(no, open_round=bool(opt.pending_batch),
                                        values=not opt.has_ingredients())
    return f" {ask}"


def diag_for(opt, checks, h):
    """The diagnosis More detail says under a miss verdict: None with a
    disagreeing Repeat (Line B keeps its check) or no checks."""
    if not checks or insights.disagreeing_repeat(opt, checks) is not None:
        return None
    return insights.miss_diagnosis(opt, checks, h)


def _clause_tips(opt, facts):
    """The clauses' rules, in the order the line says them."""
    out = []
    for clause in facts.clauses or ():
        kind = clause[0]
        if kind == "wide":
            out.append(wording.tip_wide([_wide_label(n) for n in clause[1]]))
        elif kind == "over_wide":
            out.append(insights.status_rule(opt, ("predictions", "reliable")))
    out.append(_bias_tip(facts))
    reasons = [_GUARD_WORDS[r] for r in getattr(facts.guard, "reasons", None) or ()
               if r in _GUARD_WORDS]
    if reasons and any(c[0] == "guard" for c in facts.clauses or ()):
        out.append(wording.tip_guard(reasons))
    return out


def _rule_text(opt, key):
    """'Not yet (gate): Fewer than 5 formulations with all results in.'"""
    return f"{insights.status_label(opt, key)}: {insights.status_rule(opt, key)}"


def _threshold_tip(yard):
    if yard is not None and yard.from_repeats:
        return wording.tip_threshold_measured(f"{yard.value:.1f}", yard.pairs, yard.df or None)
    value = getattr(yard, "value", None) if yard is not None else None
    return wording.tip_threshold_assumed(f"{value:g}" if value else "1")


def noise_sd_text(opt):
    """Final demo (B5): the Overall score's assumed noise SD as the screen
    writes it: '1', '2', '0.5' (the project's tasting noise, else 1)."""
    return f"{insights.assumed_noise(opt):g}"


def _predictions_tip(opt, facts):
    """The Predictions line's exact rule (§7, R-B1, I13): the numbers it
    rests on, each output and window named on its own, then the row that
    holds and its condition; Reliable's own limits last (R-I11)."""
    row = facts.prediction_row
    if row is None:
        return None
    parts = []
    if row == "off_repeat" and facts.disagreement is not None:
        parts.append(_repeat_tip(facts.disagreement))
    parts += [_coverage_tip(opt, facts.headline if row != "gate" else None),
              _skill_tip(opt, facts)]
    if row != "gate":
        parts.append(wording.TIP_REFITTED)
    parts += _clause_tips(opt, facts)
    parts.append(_rule_text(opt, ("predictions", row)))
    if getattr(facts, "diagnosis", None) is not None:
        parts.append(diag_tip(opt, facts.diagnosis))
    if row == "reliable":
        parts.append(wording.TIP_RELIABLE_LIMITS)
    return _joined(opt, parts)


def _rise_tip(facts, yard):
    """tip_progress for this project: the rule and the noise SD's basis,
    then, after a rise, the latest Round's threshold and its make-up (fix A
    follow-up, decision 4)."""
    a = getattr(facts, "improvement", None)
    need = None
    if a is not None and a.state in ("yes", "barely") and a.threshold is not None:
        base = getattr(yard, "value", None) or insights.ASSUMED_NOISE_SD
        sd = base if a.sd is None else a.sd
        need = (f"{a.threshold:.2f}", rise_factor_text(a.q, a.df),
                f"{sd:g}" if a.sd is None else f"{sd:.2f}")
    return wording.tip_progress(_threshold_tip(yard), need)


def _progress_tip(opt, facts):
    """The Progress line's rule (§7): the word that holds and its condition,
    then the rise rule against the noise threshold and its basis."""
    if facts.progress is None:
        return None
    yard = facts.yardstick if facts.yardstick is not None else insights.noise_yardstick(opt)
    return _joined(opt, [_rule_text(opt, ("progress", facts.progress)), _rise_tip(facts, yard)])


def _next_tip(opt, facts):
    """Which Next-step row fired and its condition (§7, I13), with the
    numbers it rests on; at the gate, the gate."""
    step = facts.next
    if step is None:
        key = ("predictions", "gate")
        return _joined(opt, [wording.tip_next(insights.status_label(opt, key),
                                              insights.status_rule(opt, key))])
    key = ("next", step.kind)
    parts = [wording.tip_next(insights.status_label(opt, key), insights.status_rule(opt, key))]
    yard = facts.yardstick
    if step.kind == "repeat":
        d = step.disagreement or facts.disagreement
        if d is not None:
            parts.append(_repeat_tip(d))
    elif step.kind in ("misses", "early"):
        parts.append(_coverage_tip(opt, facts.headline))
        if step.kind == "misses":
            parts.append(_bias_tip(facts))
            if getattr(step, "diagnosis", None) is not None:
                parts.append(diag_tip(opt, step.diagnosis))
        h = facts.headline
        if step.kind == "early" and facts.verdict == "wide" and h is not None and h.wide:
            # Every output very uncertain: the width rule, its outputs named.
            parts.append(wording.tip_wide([_wide_label(n) for n in h.wide]))
    elif step.kind == "no_skill":
        fact = step.skill or facts.skill
        if fact is not None:
            parts.append(wording.tip_skill(fact.rounds, skill_value(fact.value), fact.n))
    elif step.kind == "explore" and step.gain is not None:
        # T0′: the expected improvement the row weighed, and its threshold.
        parts.append(wording.tip_next_gain(f"{step.gain:.1f}", _threshold_tip(yard)))
    elif step.kind in ("untested", "explore", "close"):
        parts.append(_rise_tip(facts, yard))
    if step.kind in ("rising", "see"):
        # T0′: Best so far, the number Progress weighs, then its rule.
        best = facts.best
        if best is not None:
            parts.append(wording.tip_next_best(f"{best.score:.2f}", opt.utility_ceiling(),
                                               best.formulation, best.round))
        parts.append(_rise_tip(facts, yard))
    return _joined(opt, parts)


def _next_terms(opt, facts):
    """The terms Next-step reasoning's sentence names, by the row it says."""
    if facts is None or facts.next is None:
        return []
    values = not opt.has_ingredients()
    return {
        "repeat": [wording.tip_repeats(values), wording.TIP_INTERVAL],
        "misses": [wording.TIP_VERDICT, wording.TIP_MEAN_ERROR],
        "early": [wording.TIP_NOT_JUDGED, wording.TIP_ROUND_AHEAD],
        "no_skill": [wording.TIP_SKILL],
        "untested": [wording.TIP_PREDICTED_BEST, wording.tip_untested(values)],
        "explore": [wording.TIP_EXPECTED_IMPROVEMENT, wording.TIP_NOISE_THRESHOLD],
        "close": [wording.TIP_PREDICTED_BEST, wording.TIP_ACTIVE_BOUNDS,
                  wording.TIP_NOISE_THRESHOLD],
        "rising": [wording.TIP_PROGRESS_WORDS, wording.TIP_NOISE_THRESHOLD],
        "see": [wording.TIP_PROGRESS_WORDS, wording.TIP_NOISE_THRESHOLD],
    }.get(facts.next.kind, [])


def _term_tips(opt, line_id, facts):
    """The TIP_* texts of every statistic the line names, in screen order,
    with the line's own numbers where `facts` has them."""
    values = not opt.has_ingredients()
    total = opt.has_formulation_total()
    judged = facts is not None and facts.verdict not in (None, "none", "early", "wide")
    with_facts = facts is not None
    space = wording.tip_design_space(f"{bo_views.POOL_SIZE:,}")
    tips = {
        "basis_predictions": [wording.TIP_ROUND_AHEAD, wording.TIP_INTERVAL,
                              wording.TIP_MEAN_ERROR, wording.TIP_SKILL],
        "verdict": [wording.TIP_INTERVAL, wording.TIP_VERDICT,
                    "" if judged else wording.TIP_NOT_JUDGED,
                    _coverage_tip(opt, facts.headline) if with_facts else ""],
        "score_line": [wording.TIP_ROUND_AHEAD, wording.TIP_INTERVAL, wording.TIP_MEAN_ERROR],
        "steady_bias": [wording.TIP_MEAN_ERROR, _bias_tip(facts) if with_facts else ""],
        "skill": [wording.TIP_SKILL, _skill_tip(opt, facts) if with_facts else ""],
        "disagreement": [wording.tip_repeats(values), wording.TIP_INTERVAL,
                         _repeat_tip(facts.disagreement)
                         if with_facts and facts.disagreement is not None else ""],
        "basis_progress": [wording.TIP_NOISE_THRESHOLD, wording.tip_repeats(values),
                           wording.TIP_PROGRESS_WORDS, wording.TIP_PREDICTED_BEST,
                           wording.TIP_EXPECTED_IMPROVEMENT],
        "stand_line": [wording.TIP_PROGRESS_WORDS, wording.TIP_NOISE_THRESHOLD],
        "repeats": [wording.tip_repeats(values), wording.TIP_NOISE_THRESHOLD],
        "tie": [wording.TIP_NOISE_THRESHOLD],
        "standing": [wording.TIP_PREDICTED_BEST, wording.TIP_INTERVAL,
                     wording.tip_untested(values)],
        "limits": [wording.TIP_ACTIVE_BOUNDS],
        "luck": [wording.TIP_LUCK],
        "gain": [wording.TIP_EXPECTED_IMPROVEMENT],
        "next_reasoning": _next_terms(opt, facts),
        # W3's lines (§8 expert layer): the design space and the feasible
        # region get_polytope_samples draws, the L∞ rule "untested" uses,
        # the latent posterior sd the map plots.
        "basis_explored": [wording.tip_strips(values, total), wording.tip_untested(values),
                           wording.TIP_MAP, wording.tip_predicted_score(values),
                           wording.TIP_MODEL_SD, space],
        "strips": [wording.tip_strips(values, total), wording.tip_untested(values), space],
        "gaps": [wording.tip_strips(values, total), wording.tip_untested(values), space],
        "limit_hits": [wording.TIP_LIMIT_HITS, space],
        "map": [wording.tip_predicted_score(values), wording.TIP_MODEL_SD, wording.TIP_MAP,
                space],
    }
    return tips.get(line_id)


def tooltip_for(opt, line_id, facts=None):
    """W2 (review R-I6): the one help text of the line `line_id` (a
    TIP_LINES entry), plain reading first, then the BO term, joining the
    wording.TIP_* texts of every statistic the line names, in screen order;
    the status lines' exact rules from `facts` (insights.StatusFacts), with
    this project's numbers. None for a line with no help, and for a status
    line without its facts. In the project's own words (for_project)."""
    if line_id in ("predictions", "progress", "next", "basis_next"):
        if facts is None:
            return None
        draw = {"predictions": _predictions_tip, "progress": _progress_tip,
                "next": _next_tip, "basis_next": _next_tip}[line_id]
        return draw(opt, facts)
    tips = _term_tips(opt, line_id, facts)
    return None if tips is None else _joined(opt, tips)


def basis_line(opt, which, facts=None):
    """(text, help) of a figures fold's first line (§7, R-I6): "predictions",
    "progress" or "next". The fold label carries no tooltip, so the help
    sits here. W1 draws it first inside each figures fold:
    st.caption(text, help=help)."""
    full = {"predictions": wording.BASIS_PREDICTIONS, "progress": wording.BASIS_PROGRESS,
            "next": wording.BASIS_NEXT}[which]
    # Concision round: the short basis on screen, the full one first in its help.
    tip = tooltip_for(opt, f"basis_{which}", facts)
    return (_say(opt, wording.BASIS_SHORT[which]),
            "\n\n".join(t for t in (_say(opt, full), tip) if t))


def _model_and_method(opt, computed):
    """W2 (§7, R-I12, R-I13): Method and definitions' first part,
    wording.model_summary(insights.model_summary(opt)) read off a model
    built for this project, then the known limits (at the gate, M9, the
    cold start alone). The only drawer, with _glossary, of the allowlisted
    Model-and-method text (tests/test_food_bo._FTUE_ALLOWED)."""
    st.markdown(f"**{_say(opt, wording.MODEL_AND_METHOD)}**")
    try:
        facts = insights.model_summary(opt)
    except Exception:   # a model that cannot be built must not take the tab down
        st.caption(_say(opt, wording.WHY_FAILED))
        return
    # One line each, as written (the allowlisted lead stays an exact string).
    for line in wording.model_summary(facts):
        st.markdown(_say(opt, line))
    if facts.cold:
        return
    st.markdown(f"**{_say(opt, wording.MODEL_LIMITS_HEADING)}**")
    for line in wording.MODEL_LIMITS:
        st.markdown(_say(opt, line))


def _status_table(opt, computed):
    """W2 (I13, R-I13): "How each status is decided", insights.status_rules,
    one row per state and per Next row with its exact condition."""
    checks = (computed or {}).get("checks")
    rows = insights.status_rules(opt, checks)
    st.markdown(f"**{_say(opt, wording.STATUS_TABLE)}**")
    columns = [_say(opt, c) for c in wording.STATUS_TABLE_COLUMNS]
    frame = pd.DataFrame([[_say(opt, c) for c in row] for row in rows], columns=columns)
    st.table(frame.set_index(columns[0]))
    # Fix round B (statistician I8): which row wins when several hold.
    st.caption(_say(opt, wording.STATUS_TABLE_ORDER))


def _glossary(opt):
    """W2 (§3, R-B2, M4, M8): the glossary in the project's own form,
    wording.glossary_rows(not opt.has_ingredients()), never through
    for_project, in its one canonical order."""
    st.markdown(f"**{wording.GLOSSARY}**")
    columns = list(wording.GLOSSARY_COLUMNS)
    frame = pd.DataFrame(list(wording.glossary_rows(not opt.has_ingredients())),
                         columns=columns)
    st.table(frame.set_index(columns[0]))


# ------------------------------------------------------------------ #
#  Review cards (prototype, 2026-10-01; owner: "a clearer, less wordy way
#  to present the review"): three bordered cards side by side, each a
#  small label, one big status word, one number line, any warning on a
#  line of its own, and a 60 px sparkline; the full sentence and the exact
#  rule in the card's "?" (status_help). The hero chart under them is the
#  BO figure along Best so far -> Suggestion 1.
# ------------------------------------------------------------------ #

CARD_KEY = "insights_card"
MORE_DETAIL_KEY = "insights_more_detail"
TRIED_KEY = "insights_tried"
_CARD_LABELS = {"predictions": wording.CARD_PREDICTIONS, "progress": wording.CARD_PROGRESS,
                "next": wording.CARD_NEXT}
SPARK_HEIGHT = 40          # review polish: the plot alone (fit-x), title and labels added
HERO_HEIGHTS = (130, 80)      # review plots: room for a two-line axis title


def cards_header_text(opt):
    """'Review after Round 5 · 20 formulations · Round 6 open · Formulation
    19 waiting · 1 Not scored' (review polish: the header says what the tab
    is, and after which Round; the scored count as before)."""
    waiting = opt.awaited_rows()
    not_scored = sum(map(opt.is_set_aside, range(len(opt.Y_history))))
    order = insights.scored_rounds(opt)
    return wording.review_header(
        opt.complete_count(), order[-1][0] if order else None,
        opt.pending_batch_no if opt.pending_batch else None,
        len(waiting), numbers=[int(opt.formulation_ids[i]) for i in waiting],
        not_scored=not_scored)


class Card:
    """What one card says: its big word, its lines [(text, warn)], its
    word's tone ("off" draws it red) and its "?"."""

    def __init__(self, word, lines, tone=None, help=None):
        self.word, self.lines, self.tone, self.help = word, lines, tone, help


def misses_line(facts):
    """Specific wording: the measurements' miss line, which ones, how
    many and against the 1 in 20 a correct range misses ('Creaminess +3: 9
    of 36 outside range (about 2 expected).'); the headline's own counts
    when the facts carry no breakdown (an older cache)."""
    m = getattr(facts, "misses", None)
    if m is not None and m.total:
        return wording.values_missed(m.items, m.misses, m.total, m.allowed)
    h = getattr(facts, "headline", None)
    total = getattr(h, "total", 0) or 0
    if not total:
        return wording.TRACK_VALUES_MISS
    return wording.values_missed_short(total - h.inside, total, insights.allowed_misses(total))


def misses_detail(facts):
    """The miss line's "?" sentences (values_missed_detail), or ''."""
    m = getattr(facts, "misses", None)
    if m is None or not m.total:
        return ""
    return wording.with_left_out(
        wording.values_missed_detail(m.rounds, m.items, m.misses, m.total, m.allowed),
        getattr(m, "no_forecast", 0))


def _check_round(opt):
    return opt.pending_batch_no if opt.pending_batch else opt.next_batch_no()


def predictions_card(opt, facts):
    """Review four, ruling 1 (owner, 2026-10-02): the word reads the Overall
    score's Round-ahead record (the suggestions are made from its model);
    the measurements' misses are a secondary, non-red line that still asks
    for the neutral Repeat; a disagreeing Repeat is the Next card's, in red;
    "not explained yet" stays in the "?"."""
    row = facts.prediction_row
    word = wording.PREDICTIONS_WORDS[facts.predictions]
    check = scatter_check(opt, facts)
    inside, total, _ = score_counts(facts)
    lines = []
    if row == "gate":
        lines.append((wording.card_gate(facts.scored, insights.MIN_ROWS), False))
    elif row == "too_few":
        # Review story: the record so far, then how many it needs (it read
        # "9 / 10 formulations checked", which hid that all 9 landed).
        # Only when every check counts towards the 10 (a score recorded
        # without a measurement does not): else the count it needs.
        lines.append((wording.card_checked_so_far(inside, total, insights.SKILL_MIN)
                      if total and total == facts.checked
                      else wording.card_checked(facts.checked, insights.SKILL_MIN), False))
    elif row == "wide":
        lines.append((wording.CARD_WIDE, False))
    elif row == "no_skill":
        lines.append((wording.CARD_NO_SKILL, False))
    elif row == "unclear":
        lines.append((wording.CARD_UNCLEAR, False))
    elif row in ("off_coverage", "caution"):
        lines.append((wording.card_score_missed(total - inside, total), True))
    else:
        lines.append((wording.card_score_in_range(inside, total), False))
    asked = False
    if getattr(facts, "measures_miss", False):
        lines.append((misses_line(facts), False))
    if check is not None:
        # Ruling 2: the panel is named only with Repeat evidence; without
        # it, the neutral data request.
        vary, no = check
        if vary is not None:
            lines.append((wording.CARD_REPEATS_VARY, True))
            asked = True
        elif no is None or no != next_repeat_no(opt, facts):
            lines.append((wording.card_noise_repeat(no), False))
            asked = True
    if (not asked and getattr(facts, "measures_miss", False)
            and facts.disagreement is None and next_repeat_no(opt, facts) is None):
        # The measurements' misses still ask for the neutral Repeat, once
        # (one action, one card: not when the Next card asks already).
        best = facts.best.formulation if facts.best is not None else None
        lines.append((wording.card_noise_repeat(best), False))
    return Card(word, lines, "off" if facts.predictions == "off" else None)


def plateau_round(opt, facts):
    """Persona fixes 2: the Round of the last new best when the status is
    Flat but Best so far rose since the first Round (the display word reads
    "Plateau since Round N"; the status rule is unchanged); else None."""
    if facts is None or facts.progress != "flat" or facts.best is None:
        return None
    try:
        nums = story_numbers(opt)
    except Exception:
        return None
    if nums is None or nums[3] is None or not nums[2] > nums[0] + 1e-9:
        return None
    return nums[3]


def progress_word(opt, facts):
    """The Progress word as shown everywhere (strip, section, Make a round,
    tooltips): 'Plateau since Round 7' after a rise, else the status's."""
    if facts is None or facts.progress is None:
        return None
    r = plateau_round(opt, facts)
    return wording.plateau_word(r) if r is not None else wording.PROGRESS_WORDS[facts.progress]


def progress_card(opt, facts):
    f = facts.best
    word = progress_word(opt, facts)
    if f is None:
        return Card(word or "—", [(wording.NO_COMPLETE_FORMULATION, False)])
    tie = None if facts.tie is None else int(opt.formulation_ids[facts.tie])
    a = facts.improvement
    lines = []
    if facts.progress in ("rising", "within_noise"):
        rise, need = rise_and_need(a)
        tail = (wording.card_rising(rise, need) if facts.progress == "rising"
                else wording.card_rise(rise, need))
    elif facts.progress == "flat":
        # Persona fixes 2: "Plateau since Round 7" already says how long.
        # Round-2 fixes (P3 #18): the plateau line names the best.
        tail = (wording.card_best_who(f.formulation, tie) if plateau_round(opt, facts) is not None
                else wording.card_flat(a.rounds))
    else:
        tail = wording.card_best_who(f.formulation, tie)
    # Review story: the line goes from the start (the first scored Round's
    # best) to the best, one decimal each, as the improving chart names them.
    # Review story 2: "of 100" on a 0–100 score, so Flat at 96.4 reads as
    # near the ceiling, not stuck.
    nums = story_numbers(opt)
    best = wording.score_of(one_decimal(f.score), opt.utility_ceiling())
    if nums is not None and nums[0] != nums[2]:
        lines.append((wording.card_progress(one_decimal(nums[0]), best, tail), False))
    else:
        lines.append((wording.card_progress(None, best, tail), False))
    if facts.progress not in ("rising", "within_noise", "flat", "no_new_best") \
            and not one_round(opt):
        without = scored_without(opt, f.round)
        if without:
            lines.append((wording.short_lack(*without).rstrip("."), True))
    return Card(word, lines)


def next_repeat_no(opt, facts):
    """The Formulation the Next card asks to repeat (its misses row's noise
    check, or the assumed-gap Repeat), else None."""
    step = facts.next
    kind = None if step is None else step.kind
    best = facts.best.formulation if facts.best is not None else None
    if kind == "misses":
        return best
    if kind in _ASK_REPEAT and getattr(step, "assumed", False):
        return best
    return None


def next_card(opt, facts):
    no = opt.pending_batch_no
    left = len(open_rows(opt)) if opt.pending_batch else 0
    step = facts.next
    word = wording.card_record(no) if left else wording.card_make(opt.next_batch_no())
    lines = [(wording.card_open(left) if left else wording.CARD_NONE_OPEN, False)]
    if left and getattr(facts, "measures_miss", False):
        # Round-2 fixes (P3 #17): the measurements' miss warning Make a round
        # shows stays with "Record Round 6" here too.
        lines.append((misses_line(facts), False))
    tone = None
    ask = (facts.best.formulation if step is not None and getattr(step, "assumed", False)
           and facts.best is not None else None)
    kind = None if step is None else step.kind
    if kind == "repeat":
        d = step.disagreement
        word, tone = wording.CARD_CHECK_FIRST, "off"
        lines = [(wording.card_before(_check_round(opt)), True),
                 (wording.card_repeat_off(d.earlier, f"{abs(d.diff):.1f}", lower=d.diff < 0),
                  True)]
    elif kind == "misses" and ceiling_stop(opt, facts):
        # Review critique, follow-up: the stop, its Repeat request kept.
        word = wording.CARD_CEILING
        lines = [(wording.card_noise_repeat(
            facts.best.formulation if facts.best is not None else None), False)]
    elif kind == "misses":
        # Review diagnosis: a neutral data request, never "check panel".
        lines.append((wording.card_noise_repeat(
            facts.best.formulation if facts.best is not None else None), False))
    elif kind == "no_skill":
        lines.append((wording.CARD_GUIDED, False))
    elif kind == "untested":
        word = wording.CARD_TRY_BEST
        lines = [(wording.card_add_to(no if opt.pending_batch else None), False)]
    elif ceiling_stop(opt, facts):
        # Review critique (fix 2): within noise of the top score, the stop
        # signal leads; no widening advice (the limits are in its "?").
        word = wording.CARD_CEILING
        lines = []
    elif kind == "close":
        word = wording.CARD_CLOSE if step.binding is not None else wording.CARD_CLOSE_TOP
        reason = None
        try:
            reason = widen_all_text(opt, step)
        except Exception:
            reason = None
        lines = [(reason or wording.card_close_line(
            _close_change(step.binding),
            settings_only=not insights.has_ingredient_choice(opt)), False)]
    if ask is not None and kind in _ASK_REPEAT and (left or _why_link_offered(opt)):
        # Review polish, one action said once: with no Round open the
        # Repeat link under the buttons ("Repeat Formulation 21") is the
        # request, so the card does not say it again above them.
        lines.append((wording.card_repeat_ask(ask, opt.next_batch_no() if left else None),
                      False))
    return Card(word, lines, tone)


_CARD_BUILDERS = {"predictions": predictions_card, "progress": progress_card,
                  "next": next_card}


def card_for(opt, facts, which):
    """The Card for `which`, its "?" the full sentence then the exact rule
    (status_help), in the project's words; None while its facts are still
    being worked out."""
    if facts is None or which in facts.pending:
        return None
    card = _CARD_BUILDERS[which](opt, facts)
    card.word = _say(opt, card.word)
    card.lines = [(_say(opt, t), w) for t, w in card.lines]
    help_ = status_help(opt, facts, which)
    y = getattr(facts, "yardstick", None)
    if (which == "progress" and help_ and getattr(y, "assumed", False)
            and facts.progress in ("rising", "within_noise", "flat")):
        # Review critique: the assumed noise, and what would confirm it.
        head, _, rest = help_.partition("\n\n")
        help_ = f"{head} {wording.REPEAT_CONFIRMS}" + (f"\n\n{rest}" if rest else "")
    if which == "predictions" and help_:
        # Fix round 3: the Overall score's checks with no forecast, named
        # beside its counts (the card's 'x / N missed').
        nf = getattr(getattr(getattr(facts, "headline", None), "score", None), "no_forecast", 0)
        head, _, rest = help_.partition("\n\n")
        help_ = wording.with_left_out(head, nf, _say(opt, wording.OVERALL_SCORE_COLUMN),
                                      wording.LEFT_OUT_OF_INSIDE) + (
            f"\n\n{rest}" if rest else "")
    card.help = wording.brief(help_)
    step = getattr(facts, "next", None)
    if which == "next" and card.tone != "off" and ceiling_stop(opt, facts):
        # Review critique (fix 2): the strip's "?" says the stop signal too.
        card.help = wording.brief(ceiling_tip(opt, facts))
    return card


REVIEW_CHECKING = "checking"


def review_says_stop(opt):
    """Round-2 fixes (P2 #16): whether the Review's Next is the stop at the
    top score (ceiling_stop), read from what Review has worked out — never
    fitting here; False while it is not known."""
    try:
        if opt.complete_count() < insights.MIN_ROWS:
            return False
        checks = checks_for(opt, "stand")
        if checks is None:
            return False
        with insights.memo_in_use(memo(opt)):
            facts = screen_facts(opt, checks)
        return bool(ceiling_stop(opt, facts))
    except Exception:
        return False


def review_status(opt):
    """Review guide: Make a round's one line, the cards' own words —
    (text, warn), warn when the Predictions or the Next card is drawn red
    (Off, Check panel first); REVIEW_CHECKING while the workers are still
    fitting what the cards read; None with nothing scored, or when the
    checks are not worked out and there are no workers to work them out
    (the line never fits a model: it is read from what Review has)."""
    if opt.complete_count() == 0:
        return None
    checks = None
    with insights.memo_in_use(memo(opt)):
        if opt.complete_count() < insights.MIN_ROWS:
            facts = insights.status_facts(opt, ())      # Review's gate, model-free
        else:
            checks = checks_for(opt, "stand")
            if checks is None:
                return REVIEW_CHECKING if waiting(opt, "stand") else None
            facts = screen_facts(opt, checks)
    if facts is None or facts.progress is None:
        return None
    pred, nxt = predictions_card(opt, facts), next_card(opt, facts)
    prog = progress_card(opt, facts)
    warn = pred.tone == "off" or nxt.tone == "off"
    text = wording.review_status_line(pred.word, prog.word,
                                      nxt.word if nxt.tone == "off" else None)
    lf = learned_cached(opt)
    every = None if lf is not None else checks_for(opt, "trust")
    if opt.complete_count() >= insights.MIN_ROWS and (lf is not None or every):
        # Review learned: the Model word, as the Review's strip says it
        # (from every Round's checks, never a partial set); red as before.
        ords = lf.orderings if lf is not None else learned.orderings(opt, every)
        pick = lambda v: model_names(opt, ords, v)
        word, line = wording.learned_strip_counted(pick(learned.PREDICTABLE),
                                                   pick(learned.PARTLY), pick(learned.NOT_YET),
                                                   pair_counts(opt, ords))
        # Round-2 fixes (P3 #18): the strip's own three items, in its words:
        # 'Review: Rising · Predicts: Overall score · Stop, or change the target'.
        text = wording.review_status_strip(prog.word, f"{word} {line}".strip(), nxt.word)
    return _say(opt, text), warn


def predictions_spark_frame(opt, checks):
    """Per checked Round: the share of its Round-ahead results inside their
    95 % intervals (the verdict's values: the measurements)."""
    rows = []
    values = [c for c in insights.ranged(checks) if c.name != insights.SCORE]
    for r in insights._ordered_rounds(values):
        mine = [c for c in values if c.round == r]
        if mine:
            inside = sum(bool(c.inside) for c in mine)
            rows.append({"round": int(r), "inrange": inside / len(mine),
                         "text": f"{inside} / {len(mine)}"})
    return pd.DataFrame(rows, columns=["round", "inrange", "text"])


def progress_spark_frame(opt):
    """Per Round: Best so far at its end."""
    best = opt.best_so_far()
    out = {}
    for i, r in enumerate(opt.batch_history):
        if r is not None and best[i] is not None:
            out[int(r)] = max(out.get(int(r), float("-inf")), float(best[i]))
    rows = [{"round": r, "best": v, "text": f"{v:.2f}"} for r, v in sorted(out.items())]
    return pd.DataFrame(rows, columns=["round", "best", "text"])


def spark_chart(opt, facts, which):
    """A small trend line under a card, titled, its first and last Round
    named on the x axis (review plots): Predictions, the share of
    Round-ahead results in range per Round, 0-100 %, against the 95 % a
    calibrated model aims for (dashed, labelled); Progress, Best so far by
    Round, on at least a 10-point span so a small rise never fills the
    card. None for Next, or with under two Rounds to draw (the card then
    says wording.PLOT_SPARK_WAIT)."""
    rnd = _say(opt, wording.ROUND_CAP)
    if which == "predictions":
        df, field = predictions_spark_frame(opt, getattr(facts, "checks", None)), "inrange"
        title = _say(opt, wording.CARD_SPARK_IN_RANGE)
        heading = _say(opt, wording.PLOT_SPARK_IN_RANGE)
        bo = wording.PLOT_BO_COVERAGE
    elif which == "progress":
        df, field = progress_spark_frame(opt), "best"
        title = _say(opt, wording.BEST_SO_FAR_COLUMN)
        heading = _say(opt, wording.PLOT_SPARK_BEST)
        bo = wording.PLOT_BO_BEST
    else:
        return None
    if len(df) < 2:
        return None
    df = df.assign(bo=bo)
    ink, _ = mark("bar_latest")
    if which == "predictions":
        # The whole 0-100 %: a drop from 95 % to 60 % reads as the fall it is.
        domain = [0.0, 1.0]
    else:
        lo, hi = float(df[field].min()), float(df[field].max())
        span = max(hi - lo, 10.0)
        mid = (lo + hi) / 2
        domain = [mid - span / 2 - 0.5, mid + span / 2 + 0.5]
    y = alt.Y(f"{field}:Q", axis=None, scale=alt.Scale(domain=domain, nice=False))
    first, last = int(df["round"].min()), int(df["round"].max())
    x = alt.X("round:Q", title=None, scale=alt.Scale(nice=False, padding=8),
              axis=alt.Axis(values=[first, last], grid=False, ticks=False, domain=False,
                            labelFontSize=10, labelColor=muted(),
                            labelExpr=f"{rnd + ' '!r} + datum.value",
                            labelFlush=True))
    tip = [alt.Tooltip("round:Q", title=rnd, format="d"),
           alt.Tooltip("text:N", title=title), bo_tip()]
    base = alt.Chart(df)
    layers = [base.mark_line(color=ink, strokeWidth=2, tooltip=False).encode(x=x, y=y),
              base.mark_point(filled=True, size=28, color=ink, opacity=1).encode(
                  x=x, y=y, tooltip=tip)]
    if which == "predictions":
        ref, _ = mark("reference")
        aim = pd.DataFrame({"inrange": [0.95], "label": [_say(opt, wording.PLOT_SPARK_TARGET)]})
        # The dashed aim, labelled at the right end.
        layers[:0] = [alt.Chart(aim).mark_rule(color=ref, strokeDash=[3, 3], tooltip=False)
                      .encode(y=y),
                      alt.Chart(aim).mark_text(color=ref, align="left", baseline="bottom", dy=-2,
                                               fontSize=9, tooltip=False)
                      .encode(y=y, x=alt.value(0), text="label:N")]
    # Review polish: fit the width only, so the title is not squeezed into
    # the line (the theme fit the title and labels into the 60 px).
    return (alt.layer(*layers).properties(
                height=SPARK_HEIGHT,
                autosize=alt.AutoSizeParams(type="fit-x", contains="padding"),
                title=alt.TitleParams(heading, anchor="start", fontSize=11, fontWeight="normal",
                                      color=muted(), offset=4))
            .configure_view(strokeWidth=0))


def hero_reason_key(reason):
    """The withheld reason's short key (wording.HERO_WITHHELD)."""
    if reason in (wording.CHOOSES_REASON_NOISE, wording.GAIN_REASON_GUARD):
        return "noise"
    if reason == wording.CHOOSES_REASON_COLD:
        return "cold"
    if reason == wording.CHOOSES_REASON_UCB:
        return "ucb"
    if reason == wording.GAIN_REASON_OFF:
        return "off"
    if reason == wording.GAIN_REASON_VERDICT:
        return "verdict"
    if reason == wording.GAIN_REASON_NOT_YET:
        return "not_yet"
    return "skill"


HOW_PLACE = "insights_how_place"
METHOD_LINK_KEY = "insights_method_link"


def method_link(opt):
    """Review polish: "How the app knows" opens More detail and Method and
    definitions and lands on it — the model read from this project, the
    acquisition, the checks and the known limits, one click from the
    Review."""
    st.button(_say(opt, wording.METHOD_LINK), type="tertiary", key=METHOD_LINK_KEY,
              help=_say(opt, wording.METHOD_LINK_HELP), icon=":material/science:",
              disabled=confirmation_open(), on_click=ask_tab,
              args=(wording.TAB_INSIGHTS, HOW_PLACE, None,
                    {MORE_DETAIL_KEY: True, HOW_KEY: True}))


# ------------------------------------------------------------------ #
#  Review profiler (prototype, 2026-10-02): a JMP-style prediction
#  profiler, adapted for BO. It replaces Try one's path and Compare all in
#  the same fold. Columns: each ingredient or process setting the path
#  choices offer (fixed rows, choices and the Fill to total row have none),
#  each moved alone from its Lowest to its Highest through the reference
#  formulation along Try one's own path (insights.build_path: the Fill to
#  total row follows, as it does there). Rows: the Overall score, each
#  measurement (largest Share of score first) and, when the hero draws it,
#  the expected gain. All points of all columns go through each display
#  model in one posterior call, and through the suggestion model once.
# ------------------------------------------------------------------ #

PROFILE_POINTS = 25         # per column, plus the reference's own amount and the path's ends
PROFILE_COLUMNS = 4         # review story: the 4 that move the score most; the rest under "+N more"
PROFILE_ROWS = 4            # measurement rows shown at once
PROFILE_CACHE = "_insights_profile"
PROFILE_CHART_CACHE = "_insights_profile_chart"
PROFILE_REF_KEY = "insights_profile_ref"
PROFILE_MORE_KEY = "insights_profile_more"
PROFILE_MORE_ROWS_KEY = "insights_profile_more_rows"
_PANEL_W = 150


def panel_width(n):
    """Review story: wider panels when fewer columns are drawn. Review
    readable: at most PROFILE_WRAP panels a row (more wrap to a second
    block), so a panel is never under 140 px; one or two panels still make
    MIN_CHART_W with the row titles."""
    n = min(max(1, n), PROFILE_WRAP)
    return max(_PANEL_W, min(260, 600 // n))


# Review readable: taller rows (were 84 / 60 / 60 px), the row titles
# written across in a column at the left (were rotated, 10 px).
_PANEL_H = {"score": 100, "measure": 76, "gain": 76}
PROFILE_WRAP = 4            # panels a row at 820 px
_ROW_TITLE_W = 104          # px for a row's title, left of its tick labels
_PROFILE_SPACING = 14


class ProfileColumn:
    """One ingredient or process setting: its path through the reference,
    the display models' predictions along it, the expected gain (None when
    withheld), which points are tested near, the tested formulations near
    it, and its effect (the Overall score's predicted rise and fall)."""

    def __init__(self, path, preds, gain, near, rings, lo, hi, end):
        self.path, self.preds, self.gain, self.near = path, preds, gain, near
        self.rings, self.lo, self.hi, self.end = rings, lo, hi, end
        score = [p.mean for p in preds.get(insights.SCORE, [])]
        self.effect = (max(score) - min(score)) if score else 0.0
        # The rise and fall near what has been tested (Compare all's dark
        # bar): what the columns are ranked by, so a large swing that is
        # all guesswork does not lead. Review story 3: within each tested
        # stretch on its own (the largest), never across an untested gap.
        self.near_effect = max([max(m) - min(m) for t, m in _runs(near, score) if t] or [0.0])

    @property
    def name(self):
        return self.path.name


class Profile:
    def __init__(self, columns, rows, ref_label, ref_no, is_pick, reason, fill, fixed,
                 failed, gain_drawn=None):
        self.columns, self.rows = columns, rows
        self.ref_label, self.ref_no, self.is_pick = ref_label, ref_no, is_pick
        self.reason, self.fill, self.fixed, self.failed = reason, fill, fixed, failed
        # Review four, ruling 2: the gain row is drawn whenever there is a
        # gain; greyed when a rule would have withheld it.
        self.gain_drawn = (reason is None) if gain_drawn is None else gain_drawn

    @property
    def greyed(self):
        return self.gain_drawn and self.reason is not None


def _runs(flags, values):
    """[(flag, [values])]: `values` cut into runs of equal `flags`."""
    out = []
    for f, v in zip(flags, values):
        if out and out[-1][0] == f:
            out[-1][1].append(v)
        else:
            out.append((f, [v]))
    return out


def split_tested(amounts, near, points):
    """Review story 3: the line through `amounts` (each point a dict of
    numbers, its x added) cut into runs, [(tested, [points])], at the
    halfway amounts insights.untested_runs shades from, the values there
    halfway too, so the solid and the faded pieces meet where the grey
    begins."""
    out = []
    for a, ok, p in zip(amounts, near, points):
        pt = dict(p, x=a)
        if out and out[-1][0] != ok:
            prev = out[-1][1][-1]
            mid = {f: (prev[f] + pt[f]) / 2 for f in pt}
            out[-1][1].append(mid)
            out.append((ok, [dict(mid)]))
        if not out:
            out.append((ok, []))
        out[-1][1].append(pt)
    return out


def _near_flags(opt, recipes, tested):
    """For each recipe, whether a tested make-up lies within
    CLOSE_TO_THE_BEST of it (insights.distance's rule, vectorised: the
    largest change of a varying row as a share of its span; a categorical
    row must match), and for each tested one whether it is near any."""
    import numpy as np
    cont, cats = [], []
    for v in opt.varying_variables():
        if v.get('type') == 'continuous':
            lo, hi = (float(t) for t in v['bounds'])
            cont.append((v['name'], hi - lo))
        else:
            cats.append(v['name'])
    if not recipes or not tested:
        return [False] * len(recipes), [False] * len(tested)

    def mat(rows):
        return np.array([[float(r.get(n, 0.0) or 0.0) for n, _ in cont] for r in rows])
    spans = np.array([s for _, s in cont]) if cont else np.ones(0)
    a, b = mat(recipes), mat(tested)
    if cont:
        d = (np.abs(a[:, None, :] - b[None, :, :]) / spans).max(axis=2)
    else:
        d = np.zeros((len(recipes), len(tested)))
    ok = d <= insights.CLOSE_TO_THE_BEST + insights._EPS
    for n in cats:
        same = np.array([[r.get(n) == t.get(n) for t in tested] for r in recipes])
        ok &= same
    return ok.any(axis=1).tolist(), ok.any(axis=0).tolist()


def profile_data(opt, fitted, recipe, ref_label, ref_no, is_pick=False, reason=None,
                 points=PROFILE_POINTS):
    """The profiler's numbers through `recipe` (at the Formulation total).
    `reason`: why the expected gain is withheld (chooses_reason), None to
    draw it."""
    fitted = fitted or {}
    paths, failed = [], []
    for name in insights.path_choices(opt):
        try:
            path = insights.build_path(opt, name, recipe, points=points)
        except Exception:
            failed.append(name)
            continue
        if path.start_breaks or not path.recipes:
            failed.append(name)
            continue
        paths.append(_refined(opt, path, recipe, points))
    recipes = [r for p in paths for r in p.recipes]
    X = [opt._encode(r) for r in recipes]
    # One posterior call per display model over every column's points.
    allp = {n: insights.predict(opt, f, X) for n, f in fitted.items()
            if f is not None and f.model is not None}
    gain = None
    # Ruling 2: worked out under a greyable reason too (drawn grey).
    if (reason is None or gain_greyable(reason)) and recipes:
        model = suggestion_model(opt, fitted)
        if model is not None:
            gain = bo_views.evaluate(opt, model, bo_views.points(opt, recipes)).value.tolist()
    rows_tested = [i for i in range(len(opt.Y_history)) if opt.fully_measured(i)]
    tested = [insights.start_recipe(opt, i) for i in rows_tested]
    columns, k = [], 0
    for path in paths:
        n = len(path.recipes)
        near, hit = _near_flags(opt, path.recipes, tested)
        rings = [(i, float(t.get(path.name, 0.0) or 0.0), float(opt.Y_history[i]))
                 for i, t, h in zip(rows_tested, tested, hit) if h]
        var = insights._var(opt, path.name)
        lo, hi = (float(b) for b in var['bounds'])
        lo, hi = min(lo, path.start_amount), max(hi, path.start_amount)
        columns.append(ProfileColumn(
            path, {name: p[k:k + n] for name, p in allp.items()},
            None if gain is None else gain[k:k + n], near, rings, lo, hi,
            limit_end(opt, path.name, path.start_amount)))
        k += n
    # Review critique (fix 4): Set up's order (ranking by the swing over
    # each column's own stretch favoured the columns with the most room).
    rows = [insights.SCORE] + [o['name'] for o in opt.measurements_by_importance()
                               if o['name'] in allp]
    if insights.SCORE not in allp:
        rows = rows[1:]
    return Profile(columns, rows, ref_label, ref_no, is_pick,
                   reason if gain is not None else (reason or wording.CHOOSES_REASON_COLD),
                   opt.balance_row_name(), [v['name'] for v in opt.fixed_variables()], failed,
                   gain_drawn=gain is not None)


def _refined(opt, path, start, points=PROFILE_POINTS):
    """Review critique (fix 4): `path` with `points` evenly spaced amounts
    over its reachable stretch (low end to high end) added, so a column
    zoomed to that stretch draws a curve, not two points."""
    lo, hi = path.low_end.amount, path.high_end.amount
    if hi - lo <= 1e-12 or len([t for t in path.amounts if lo <= t <= hi]) >= points:
        return path
    extra = [lo + (hi - lo) * k / (points - 1) for k in range(points)]
    have = dict(zip(path.amounts, path.recipes))
    amounts, recipes = [], []
    for t in sorted(set(path.amounts) | set(extra)):
        r = have.get(t)
        if r is None:
            r = insights._recipe_at(opt, path.name, start, path.mode, t)
        if r is None:
            continue
        amounts.append(t)
        recipes.append(r)
    return dataclasses.replace(path, amounts=amounts, recipes=recipes)


def gain_greyable(reason):
    """Ruling 2: a withheld reason under which the gain is still drawn,
    in grey (the noise guard, the verdict, the skill gate); a cold Round
    or an upper-bound acquisition has no gain to draw."""
    return reason is not None and hero_reason_key(reason) not in ("cold", "ucb")


def _row_title(opt, name):
    """A row's title, short lines: the result and its aim (written across,
    in the column left of the panels)."""
    if name == insights.SCORE:
        return ["Overall score", "(of 100)"]
    # (The Overall row's "zoomed" is added by profile_chart, from its domain.)
    obj = next(o for o in opt.objectives if o['name'] == name)
    unit = str(obj.get('unit') or "")
    if obj.get('goal') == 'target' and obj.get('target') is not None:
        aim = wording.profiler_row_target(join_unit(f"{float(obj['target']):g}", unit))
    elif obj.get('goal') == 'min':
        aim = wording.PROFILER_ROW_LOWER
    else:
        aim = wording.PROFILER_ROW_HIGHER
    # Beside a short panel: the name on at most two lines.
    return _wrap(obj['name'], 14).split("\n")[:2] + [aim]


def _row_domain(opt, profile, columns, name):
    vals = []
    for c in columns:
        for p in c.preds.get(name, []):
            vals += [p.low, p.high]
    if name == insights.SCORE:
        vals += [s for c in columns for _, _, s in c.rings]
        lo_cap, hi_cap = 0.0, float(opt.utility_ceiling())
    else:
        obj = next(o for o in opt.objectives if o['name'] == name)
        if obj.get('goal') == 'target' and obj.get('target') is not None:
            vals.append(float(obj['target']))
        lo_cap, hi_cap = float(obj['min_val']), float(obj['max_val'])
    if not vals:
        return [lo_cap, hi_cap]
    lo, hi = min(vals), max(vals)
    pad = 0.06 * max(hi - lo, 1e-6 + 0.02 * (hi_cap - lo_cap))
    return [max(lo_cap, lo - pad), min(hi_cap, hi + pad)]


def _x_title(path):
    return f"{path.name} ({path.unit})" if path.unit else path.name


def _x_text(opt, path, t):
    if insights.is_setting(opt, path.name):
        return fmt_setting(float(t), path.unit)
    return _amount(t, path.unit)


def _profile_row_frame(opt, cols, kind, name, labels, texts):
    """One row's long frame: for each column (`labels`, its header) the
    curve, the grey stretches, the reference, two invisible points at its
    Lowest and Highest (so each column's x runs over exactly that), the
    target, the at-a-limit note and the rings."""
    ref_name, target_name, untested_name, tested_name, band_name = texts
    d = 2 if kind == "gain" else decimals(opt, name)
    fmt = lambda v: f"{v:.{d}f}"
    target = None
    if kind == "measure":
        obj = next(o for o in opt.objectives if o['name'] == name)
        if obj.get('goal') == 'target' and obj.get('target') is not None:
            target = float(obj['target'])
    recs = []
    for c in cols:
        col, path = labels[c.name], c.path
        for idx, t in enumerate(path.amounts):
            rec = {"col": col, "mark": "curve", "x": t, "amount": _x_text(opt, path, t),
                   "factor": path.name, "band": band_name}
            if kind == "gain":
                g = c.gain[idx]
                rec.update(v=g, low=0.0, high=g, text=f"{g:.2f}", bo=wording.profiler_bo_gain(g))
            else:
                p = c.preds[name][idx]
                rec.update(v=p.mean, low=p.low, high=p.high,
                           text=f"{fmt(p.mean)} ({range_text(p.low, p.high, fmt)})",
                           bo=wording.profiler_bo(p.mean, p.low, p.high))
            recs.append(rec)
        # Review story 3: the drawn line and band, solid over tested
        # amounts and faded over untested ones (the same rule as the grey).
        points = [{"v": r["v"], "low": r["low"], "high": r["high"]}
                  for r in recs[-len(path.amounts):]]
        for k, (ok, run) in enumerate(split_tested(path.amounts, c.near, points)):
            recs += [dict(p, col=col, mark="seg_t" if ok else "seg_u", run=f"{c.name}#{k}",
                          band=band_name) for p in run]
        for lo, hi in insights.untested_runs(path, c.near):
            recs.append({"col": col, "mark": "grey", "x": lo, "x2": hi, "region": untested_name})
        recs.append({"col": col, "mark": "ref", "x": path.start_amount, "role": ref_name})
        # Review critique (fix 4): each column's x zoomed to its reachable
        # stretch (where its line runs), not the row's whole Lowest–Highest.
        recs += [{"col": col, "mark": "ends", "x": e}
                 for e in (min(path.amounts), max(path.amounts))]
        if target is not None:
            recs.append({"col": col, "mark": "target", "target": target, "role": target_name})
        if kind == "score" and c.end is not None:
            recs.append({"col": col, "mark": "note_" + c.end, "x": path.start_amount,
                         "note": wording.profiler_at_end(c.end)})
        # A line that stops short of Lowest or Highest: a limit holds it
        # (the Fill to total row, a property limit); said where it stops.
        if kind == "score":
            for e, side in ((path.low_end, "Lowest"), (path.high_end, "Highest")):
                if e.reason:
                    recs.append({"col": col, "mark": "stop", "x": e.amount,
                                 "note": wording.PROFILER_STOPPED, "why": e.reason,
                                 "side": side})
        if kind == "score":
            recs += [{"col": col, "mark": "ring", "x": a, "score": round(sc, 2),
                      "no": int(opt.formulation_ids[i]), "what": tested_name,
                      "bo": wording.PLOT_BO_MODEL_FREE} for i, a, sc in c.rings]
    return pd.DataFrame(recs)


# Review story 3: over untested amounts the line is thin, dashed and faint
# and the band a quarter as strong, so a model curve there reads as a guess.
_UNTESTED_LINE = {"strokeWidth": 1.2, "strokeDash": [3, 3], "opacity": 0.45}
_UNTESTED_BAND = 0.25


def profile_chart(opt, profile, columns=None, rows=None, gain=True):
    """The prediction profiler: a grid of small panels, one row per result
    (shared y within a row), one column per ingredient or process setting
    (its own Lowest-Highest on x, named under the bottom row). In each: the
    display model's mean (blue) and its 95 % likely range of a result; the
    measurement's target dashed; grey where nothing was tested near; the
    reference's own amount as a vertical line (ink for Best so far, orange
    for a suggestion); on the Overall row, grey rings at the tested
    formulations near the line and a note where the reference sits at a
    limit. The bottom row, when the gain is shown: the expected gain along
    each line. Each row is one faceted layer (fast to build and draw).
    `columns`, `rows`: names to draw, in order (all when None)."""
    cols = [c for c in profile.columns if columns is None or c.name in columns]
    if columns is not None:
        cols.sort(key=lambda c: list(columns).index(c.name))
    names = [n for n in profile.rows if rows is None or n in rows or n == insights.SCORE]
    if not cols or not names:
        return None
    # Review readable: at most PROFILE_WRAP panels a row; more columns
    # (added under "+N more") wrap into a second block below, with its own
    # headers and row titles.
    blocks = [_profile_block(opt, profile, cols[i:i + PROFILE_WRAP], names, gain, len(cols))
              for i in range(0, len(cols), PROFILE_WRAP)]
    chart = blocks[0] if len(blocks) == 1 else alt.vconcat(*blocks, spacing=22)
    chart = chart.resolve_scale(fill="shared", stroke="shared", strokeDash="shared")
    return readable(titled(chart, _say(opt, wording.PLOT_PROFILER)))


def _profile_block(opt, profile, cols, names, gain, n_all):
    """One block of the profiler: a row per result over `cols` (at most
    PROFILE_WRAP), each a faceted layer."""
    with_gain = gain and profile.gain_drawn and all(c.gain is not None for c in cols)
    blue = model_blue()
    gain_c = mark("neutral")[0] if profile.greyed else blue
    band_c, band_o = _BAND_FILL[theme()]
    grey_c, grey_o = _FILLS[theme()]["untested"]
    ink, _ = mark("bar_latest")
    ref_c = outline_colour() if profile.is_pick else ink
    tgt_c, tgt_o = mark("neutral")
    ring_c, ring_o = mark("neutral")
    texts = (_say(opt, profile.ref_label), _say(opt, wording.PLOT_TARGET),
             _say(opt, wording.PROFILER_UNTESTED_DASHED), _say(opt, wording.PROFILER_TESTED),
             _say(opt, wording.PLOT_INTERVAL))
    ref_name, target_name, untested_name, tested_name, band_name = texts
    # The header above each column: its name and unit, wrapped to the panel.
    pw = panel_width(n_all)
    labels = {c.name: _wrap(_x_title(c.path), max(14, pw // 7)) for c in cols}
    order = [labels[c.name] for c in cols]
    any_grey = any(not all(c.near) for c in cols)
    any_target = any(o.get('goal') == 'target' and o.get('target') is not None
                     for o in opt.objectives if o['name'] in names)
    regions = alt.Scale(domain=[band_name] + ([untested_name] if any_grey else []),
                        range=[band_c] + ([grey_c] if any_grey else []))
    role_dom = [ref_name] + ([target_name] if any_target else [])
    roles = alt.Scale(domain=role_dom, range=[ref_c, tgt_c][:len(role_dom)])
    dashes = alt.Scale(domain=role_dom, range=[[1, 0], [4, 3]][:len(role_dom)])
    rings_s = alt.Scale(domain=[tested_name], range=["circle"])
    legend = alt.Legend(title=None, orient="top", labelLimit=0, symbolOpacity=0.7,
                        offset=10)
    predicted = _say(opt, wording.PREDICTED)
    row_kinds = [("score" if n == insights.SCORE else "measure", n) for n in names]
    if with_gain:
        row_kinds.append(("gain", None))
    gain_top = max(max(max(c.gain) for c in cols), 0.05) if with_gain else None
    out_rows = []
    for r, (kind, name) in enumerate(row_kinds):
        last_row = r == len(row_kinds) - 1
        df = _profile_row_frame(opt, cols, kind, name, labels, texts)
        if kind == "gain":
            ys = alt.Scale(domain=[0.0, gain_top * 1.05], nice=False)
            y_title = [wording.PROFILER_GAIN_ROW, "(points)"]
        else:
            domain = _row_domain(opt, profile, cols, name)
            if kind == "score" and df["mark"].astype(str).str.startswith("note_").any():
                # Review readable: room above the line for "at its Highest",
                # so the note never sits on the band or the rings.
                domain = [domain[0], domain[1] + (domain[1] - domain[0]) * 22
                          / (_PANEL_H[kind] - 22)]
            ys = alt.Scale(domain=domain, nice=False, zero=False)
            y_title = _row_title(opt, name)
            if kind == "score":
                ceiling = float(opt.utility_ceiling())
                y_title = [y_title[0], zoomed("(of 100)", domain[0], domain[1], ceiling)
                           .replace("((", "(").replace("))", ")")]
        x = alt.X("x:Q", title=None, scale=alt.Scale(nice=False, zero=False),
                  axis=alt.Axis(labelFontSize=FONT_SMALL, tickCount=3, labelFlush=True,
                                grid=False, labels=last_row, ticks=last_row))
        # Review readable: the row's title written across, right-aligned in
        # a column of its own left of the tick labels (was rotated, 10 px).
        y_axis = alt.Axis(title=y_title, labelFontSize=FONT_SMALL, tickCount=3,
                          titleFontSize=FONT_AXIS_TITLE, titleFontWeight="normal",
                          titleAngle=0, titleAlign="right", titleBaseline="middle",
                          titleAnchor="middle", titleX=-30,
                          titleY=_PANEL_H[kind] / 2 - 7 * (len(y_title) - 1),
                          titleLimit=_ROW_TITLE_W, titleLineHeight=14)
        y = alt.Y("v:Q", scale=ys, axis=y_axis)
        base = alt.Chart()
        tip_title = (wording.GAIN_TOOLTIP_TITLE if kind == "gain"
                     else f"{predicted} ({wording.CHOOSES_BAND})")
        tip = [alt.Tooltip("amount:N", title=_say(opt, wording.INGREDIENT_OR_SETTING)),
               alt.Tooltip("text:N", title=tip_title), bo_tip()]
        layers = [base.mark_point(opacity=0, tooltip=False).encode(x=x)
                  .transform_filter(_only("ends"))]
        if any_grey and (df["mark"] == "grey").any():
            layers.append(base.mark_rect(opacity=grey_o, tooltip=False)
                          .encode(x=x, x2="x2:Q",
                                  fill=alt.Fill("region:N", scale=regions, legend=legend))
                          .transform_filter(_only("grey")))
        # Review story 3: each piece of line and band drawn by its run
        # (detail), solid where tested, faded where not.
        run = alt.Detail("run:N")
        line_c = gain_c if kind == "gain" else blue
        for seg, fade in (("seg_t", 1.0), ("seg_u", _UNTESTED_BAND)):
            if kind == "gain":
                layers.append(base.mark_area(color=gain_c, opacity=0.25 * fade, tooltip=False)
                              .encode(x=x, y=y, detail=run).transform_filter(_only(seg)))
            else:
                layers.append(base.mark_area(opacity=band_o * fade, tooltip=False)
                              .encode(x=x, y=alt.Y("low:Q", scale=ys, axis=y_axis), y2="high:Q",
                                      fill=alt.Fill("band:N", scale=regions, legend=legend),
                                      detail=run)
                              .transform_filter(_only(seg)))
        solid = {"strokeDash": [4, 3]} if kind == "gain" and profile.greyed else {}
        layers += [base.mark_line(color=line_c, strokeWidth=2, tooltip=False, **solid)
                   .encode(x=x, y=y, detail=run).transform_filter(_only("seg_t")),
                   base.mark_line(color=line_c, tooltip=False, **_UNTESTED_LINE)
                   .encode(x=x, y=y, detail=run).transform_filter(_only("seg_u")),
                   # Wide invisible points: the hover reads anywhere along the line.
                   base.mark_point(size=80, opacity=0, filled=True)
                   .encode(x=x, y=y, tooltip=tip).transform_filter(_only("curve"))]
        if (df["mark"] == "target").any():
            layers.append(base.mark_rule(strokeWidth=1.2, opacity=tgt_o, tooltip=False)
                          .encode(y=alt.Y("target:Q", scale=ys, axis=y_axis),
                                  stroke=alt.Stroke("role:N", scale=roles, legend=legend),
                                  strokeDash=alt.StrokeDash("role:N", scale=dashes, legend=legend))
                          .transform_filter(_only("target")))
        layers.append(base.mark_rule(strokeWidth=1.5, tooltip=False)
                      .encode(x=x, stroke=alt.Stroke("role:N", scale=roles, legend=legend),
                              strokeDash=alt.StrokeDash("role:N", scale=dashes, legend=legend))
                      .transform_filter(_only("ref")))
        if (df["mark"] == "ring").any():
            layers.append(base.mark_point(filled=False, size=34, strokeWidth=1.3,
                                          opacity=ring_o, color=ring_c)
                          .encode(x=x, y=alt.Y("score:Q", scale=ys, axis=y_axis),
                                  shape=alt.Shape("what:N", scale=rings_s, legend=legend),
                                  tooltip=[alt.Tooltip("no:Q", format="d", title=_say(
                                               opt, wording.FORMULATION_CAP)),
                                           alt.Tooltip("score:Q", format=".1f", title=_say(
                                               opt, wording.OVERALL_SCORE_COLUMN)),
                                           bo_tip()])
                          .transform_filter(_only("ring")))
        for end, align, dx in (("Highest", "right", -3), ("Lowest", "left", 3)):
            if (df["mark"] == "note_" + end).any():
                layers.append(base.mark_text(fontSize=FONT_SMALL, color=muted(), baseline="top",
                                             dy=2, dx=dx, align=align, tooltip=False)
                              .encode(x=x, y=alt.value(0), text="note:N")
                              .transform_filter(_only("note_" + end)))
        if (df["mark"] == "stop").any():
            stop_c, stop_o = mark("neutral")
            stop_tip = [alt.Tooltip("why:N", title=_say(opt, wording.PROFILER_STOPPED))]
            # Review critique (fix 4): each "limit" written inside its own
            # edge (the Lowest's to the right of it, the Highest's to the
            # left), so the two never overlap.
            layers.append(base.mark_rule(strokeDash=[2, 2], color=stop_c, opacity=stop_o)
                          .encode(x=x, tooltip=stop_tip).transform_filter(_only("stop")))
            # Round-2 fixes (P2 #13): under the plot, below the tick
            # labels, clear of every line.
            for side, align, dx in (("Highest", "right", -3), ("Lowest", "left", 3)):
                layers.append(base.mark_text(fontSize=FONT_SMALL, color=muted(), baseline="top",
                                             dy=20, dx=dx, align=align)
                              .encode(x=x, y=alt.value(_PANEL_H[kind]), text="note:N",
                                      tooltip=stop_tip)
                              .transform_filter(_only("stop"))
                              .transform_filter(alt.FieldEqualPredicate(field="side", equal=side)))
        # Each column named once, above the top row (a header under the
        # bottom row's tick labels overlapped them in Streamlit's renderer).
        header = (alt.Header(labelOrient="top", labelFontSize=FONT_LABEL, labelLimit=pw + 8,
                             labelExpr="split(datum.label, '\\n')", labelPadding=6,
                             labelBaseline="bottom", labelLineHeight=14)
                  if r == 0 else alt.Header(labels=False))
        out_rows.append(
            alt.layer(*layers, data=df).properties(width=pw, height=_PANEL_H[kind])
            .facet(column=alt.Column("col:N", sort=order, title=None, header=header),
                   spacing=_PROFILE_SPACING, bounds="flush")   # review four: tick labels clear
            .resolve_scale(x="independent"))
    return alt.vconcat(*out_rows, spacing=26).resolve_scale(
        fill="shared", stroke="shared", strokeDash="shared")


ROOM_MIN = 0.05     # a profiler column needs this share of its row's span to move in


def _too_little_room(opt, name, path):
    """Whether a row's reachable stretch (the path's two ends) is under
    ROOM_MIN of its Lowest–Highest span; None when it cannot be read."""
    try:
        lo, hi = (float(b) for b in insights._var(opt, name)['bounds'])
        reach = float(path.high_end.amount) - float(path.low_end.amount)
    except Exception:
        return None
    return hi > lo and reach < ROOM_MIN * (hi - lo)


def cramped_columns(opt, profile):
    """The columns whose reachable stretch (the path's two ends) is under
    ROOM_MIN of the row's Lowest–Highest span, in profile order."""
    return [c.name for c in getattr(profile, "columns", None) or ()
            if _too_little_room(opt, c.name, getattr(c, "path", None))]


_ROOM_CACHE = {}


def room_counts(opt):
    """(rows with too little room, rows) over the profiler's rows (every
    varied ingredient and process setting) through its default reference
    (profile_refs' last: Suggestion 1, else Best so far) — the same
    paths and the same rule (ROOM_MIN) as the profiler's columns, worked
    out with no model. A row whose path cannot leave the reference has no
    room. Cached per data, setup and reference."""
    names = insights.path_choices(opt)
    refs = profile_refs(opt) if names else []
    if not refs:
        return 0, len(names)
    _label, _no, recipe, _pick = refs[-1]
    key = (insights.data_key(opt), insights.setup_key(opt), insights._digest(recipe))
    if key in _ROOM_CACHE:
        return _ROOM_CACHE[key]
    tight = 0
    for name in names:
        try:
            path = insights.build_path(opt, name, recipe, points=PROFILE_POINTS)
        except Exception:
            path = None
        if path is None or path.start_breaks or not path.recipes:
            tight += 1
        elif _too_little_room(opt, name, path):
            tight += 1
    if len(_ROOM_CACHE) >= 32:
        _ROOM_CACHE.clear()
    _ROOM_CACHE[key] = (tight, len(names))
    return tight, len(names)


def profiler_hidden(opt):
    """Slim Review (owner, 2026-10-08): What drives each result? is left
    out when more than half of the varied rows have too little room to
    move (room_counts) — its lines would be stubs."""
    try:
        tight, n = room_counts(opt)
    except Exception:
        return False
    return n > 0 and tight * 2 > n


def profile_shown(profile, order=None):
    """(columns shown, columns under "+N more", measurement rows shown,
    rows under "+N more"): the PROFILE_COLUMNS columns that move the
    predicted Overall score most, the PROFILE_ROWS measurements with the
    largest Share of score; the rest offered. `order`: the columns in this
    order instead (Set up's, while the noise guard withholds any ranking)."""
    names = [c.name for c in profile.columns]
    if order is not None:
        names = [n for n in order if n in names]
    measures = [n for n in profile.rows if n != insights.SCORE]
    return (names[:PROFILE_COLUMNS], names[PROFILE_COLUMNS:],
            measures[:PROFILE_ROWS], measures[PROFILE_ROWS:])


def profile_refs(opt):
    """[(label, Formulation number, make-up at the total, is a suggestion)]:
    Best so far, then the open Round's Suggestion 1 when there is one."""
    best = opt.best_index()
    out = []
    if best is not None:
        out.append((wording.PROFILER_REF_BEST, int(opt.formulation_ids[best]),
                    insights.start_recipe(opt, best), False))
    first = _first_pick(opt)
    if first is not None:
        out.append((wording.PROFILER_REF_PICK, first[0], first[1], True))
    return out


def profile_cached(opt, fitted, ref):
    """profile_data once per data, setup, reference and its make-up (the
    Try one pattern: the session keeps the latest)."""
    label, number, recipe, is_pick = ref
    reason = chooses_reason(opt, fitted)
    key = (insights.data_key(opt), insights.setup_key(opt), label, number,
           insights._digest(recipe), opt.pending_batch_no, len(opt.pending_batch or []),
           reason)
    hit = st.session_state.get(PROFILE_CACHE)
    if hit and hit[0] == key:
        return hit[1]
    # Review four: no spinner of its own (V28, one indicator per wait): Q3
    # is drawn on the tab's first screen, under the progress bar's place.
    value = profile_data(opt, fitted, recipe, label, number, is_pick, reason)
    st.session_state[PROFILE_CACHE] = (key, value)
    return value


def _more_kind(opt, names):
    """The "+N more" columns' kind: "ingredients", "settings" or "both"."""
    settings = [insights.is_setting(opt, n) for n in names]
    return "settings" if all(settings) else "both" if any(settings) else "ingredients"


def _profiler(opt, computed):
    """The fold's body: the reference selector (with a suggestion to draw
    through), the profiler, its caption, the lines it needs, and "+N more"
    for the columns and rows left out."""
    if not insights.path_choices(opt):
        st.caption(_say(opt, wording.change_none(settings_only=not opt.has_ingredients())))
        return None
    fitted = computed["fitted"]
    refs = profile_refs(opt)
    if not refs:
        return None
    if len(refs) > 1:
        labels = [r[0] for r in refs]
        _keep_if_offered(PROFILE_REF_KEY, labels)
        # Review four: Suggestion 1 first when there is one (a Round open,
        # or suggestions made), else Best so far.
        picked = st.segmented_control(_say(opt, wording.PROFILER_THROUGH), labels,
                                      key=PROFILE_REF_KEY, default=labels[-1],
                                      help=_say(opt, wording.PROFILER_REF_HELP))
        ref = next((r for r in refs if r[0] == picked), refs[-1])
    else:
        ref = refs[0]
    try:
        profile = profile_cached(opt, fitted, ref)
    except Exception:   # the profiler failing must not take the tab down
        st.caption(_say(opt, wording.CHART_FAILED))
        return None
    # The noise guard (review I3): no ranking while the Overall-score model
    # is likely fitting noise; the columns keep Set up's order.
    guard = _guard(opt, fitted)
    shown, extra_cols, rows_shown, extra_rows = profile_shown(
        profile, insights.path_choices(opt) if guard.fires else None)
    # Round-2 fixes (P2 #13): a column whose line can barely move (under
    # ROOM_MIN of its row's span) is not drawn; the caption names it.
    cramped = cramped_columns(opt, profile)
    shown = [n for n in shown if n not in cramped]
    extra_cols = [n for n in extra_cols if n not in cramped]
    # Review learned: rows only for the measurements the model predicts
    # (Predictable or Partly); the rest named in one line, still addable.
    hidden = []
    lf = learned_cached(opt)
    if lf is not None:
        every = rows_shown + extra_rows
        hidden = [n for n in every if n in lf.orderings
                  and lf.orderings[n].verdict == learned.NOT_YET]
        rows_shown = [n for n in every if n not in hidden][:PROFILE_ROWS]
        extra_rows = [n for n in every if n not in rows_shown]
    chart_place = st.container()
    if extra_cols or extra_rows:
        left, right = st.columns(2)
        if extra_cols:
            _keep_all_offered(PROFILE_MORE_KEY, extra_cols)
            # Review four polish: a label and a placeholder of its own
            # (Streamlit's read "Choose options").
            kind = _more_kind(opt, extra_cols)
            shown += left.multiselect(wording.profiler_more_label(kind), extra_cols,
                                      key=PROFILE_MORE_KEY,
                                      placeholder=wording.profiler_more_placeholder(kind),
                                      help=_say(opt, wording.profiler_more_help(
                                          not opt.has_ingredients())))
        if extra_rows:
            _keep_all_offered(PROFILE_MORE_ROWS_KEY, extra_rows)
            rows_shown += right.multiselect(wording.profiler_more_label("rows"), extra_rows,
                                            key=PROFILE_MORE_ROWS_KEY,
                                            placeholder=wording.profiler_more_placeholder(
                                                "rows"),
                                            help=_say(opt, wording.profiler_more_help(
                                                rows=True)))
    # The chart too, once per profile, picks and theme: building it is most
    # of the profiler's time (about 0.3 s of Altair on the yogurt pilot).
    chart_key = ((st.session_state.get(PROFILE_CACHE) or (None,))[0], tuple(shown),
                 tuple(rows_shown), theme())
    hit = st.session_state.get(PROFILE_CHART_CACHE)
    if hit and hit[0] == chart_key:
        chart = hit[1]
    else:
        chart = profile_chart(opt, profile, shown, rows_shown, gain=False)
        st.session_state[PROFILE_CHART_CACHE] = (chart_key, chart)
    with chart_place:
        if chart is None:
            st.caption(_say(opt, wording.CHART_FAILED))
            return None
        show_chart(_untitled(chart), width="content")
        # Review four: with Suggestion 1 as the reference, the caption says
        # it sits at the gain's peaks (the acquisition's maximiser).
        caption = (wording.Q3_CAPTION_PICK if profile.is_pick and profile.gain_drawn
                   else wording.PROFILER_CAPTION)
        how = wording.profiler_help(profile.ref_label, profile.ref_no, profile.fill,
                                    profile.fixed, gain=False,
                                    settings_only=not opt.has_ingredients())
        if guard.fires:
            # Review four: the guard's long line goes in the "?" (captions
            # at most 8 words); the columns keep Set up's order.
            how = f"{path_guard_line(guard)}\n\n{how}"
        st.caption(_say(opt, caption), help=_say(opt, wording.brief(how)))
        if profile.fill:
            st.caption(_say(opt, wording.profiler_fill_note(profile.fill)))
        # Round-2 fixes (P2 #13): no Expected gain row here (Make a round's
        # map has it), so no line on why it is greyed.
        if cramped:
            # Specific wording: the share that counts as too little room.
            room = f"{ROOM_MIN:.0%}".replace("%", " %")
            st.caption(_say(opt, wording.cramped_counted(cramped, room)),
                       help=_say(opt, wording.PROFILER_CRAMPED_HELP))
        note = moved_note(opt, shown)
        if note is not None:
            st.caption(_say(opt, note[0]), help=_say(opt, note[1]))
        hidden = [n for n in hidden if n not in rows_shown]
        if hidden:
            st.caption(_say(opt, wording.not_yet_counted(hidden, pair_counts(opt, lf.orderings))
                            if lf is not None else wording.learned_not_predictable(hidden)),
                       help=_say(opt, wording.LEARNED_RULE))
    return profile, shown, guard


def moved_note(opt, shown):
    """Review story 3: (note, its "?") for the strongest pair of rows
    that moved together (|r| >= insights.MOVED_R) with at least one of
    them among the `shown` columns, the other pairs in the "?"; or None."""
    pairs = insights.moved_pairs(opt)
    on_screen = [p for p in pairs if p[0] in shown or p[1] in shown]
    if not on_screen:
        return None
    a, b, r = on_screen[0]
    # The column further left (or the one shown) named first.
    a, b = sorted((a, b), key=lambda n: list(shown).index(n) if n in shown else len(shown))
    n = sum(1 for i in range(len(opt.Y_history)) if opt.fully_measured(i))
    others = [p for p in pairs if p is not on_screen[0]]
    return wording.moved_note(a, b), wording.moved_help(a, b, r, n, others)


# ------------------------------------------------------------------ #
#  Review four (prototype, 2026-10-02; owner: "clean up the Review tab to
#  cleanly (and visually) answer these 4 questions"): each question's body
#  — its answer line (the card's bold word first), one chart, its caption
#  of at most 8 words; the evidence in the answer line's "?".
# ------------------------------------------------------------------ #

def _untitled(chart):
    """`chart` without its own title: the section's heading asks the
    question."""
    out = chart.copy()
    out.title = alt.Undefined
    return out


def _evidence_help(opt, facts, which):
    """An answer line's "?": the evidence first, then the card's full
    sentence and rule."""
    card = card_for(opt, facts, which)
    return None if card is None else card.help


# ------------------------------------------------------------------ #
#  Review learned (prototype, 2026-10-02; owner: the Review "doesn't
#  inspire confidence in the tool"): "What has the model learned?" leads
#  with checkable evidence — each result's within-Round order forecast
#  before tasting, against chance; its verdict; the effects that held;
#  "Called it?" for the latest Round. The rules are learned.py's.
#  Review learned 2 (owner: concise, lead with the aha): Called it? at the
#  top, as charts; the section one line, one chart, one caption; an effect
#  named only by the strict rule.
# ------------------------------------------------------------------ #

LEARNED_KEY = "_insights_learned"
PREVIEW_KEY = "_insights_preview"
LEARNED_ROW_H = 34          # px per row of the ordering chart (review readable: 12 px labels)
CALLED_KEY = "insights_called"
CALLED_W = 600              # px per Forecast check panel (round-2 fixes: one per row)
CALLED_ROW_H = 40           # px per formulation row
CALLED_H = 160              # px, a Called it? slopegraph's height (review readable: was 120)


class LearnedFacts:
    """What the section, the strip, the profiler and Make a round read:
    orderings {name: learned.Ordering}, effects {name: [learned.Effect]},
    called (Round, [(name, learned.RoundCall)]), score_call (the Overall
    score's RoundCall for that Round), landing (inside, above, below), the
    expert skill line, and Called it?'s panels: items [(name, RoundCall,
    [Check])], the Overall score first, when that Round is the project's
    latest (else [])."""

    def __init__(self, orderings, effects, called, score_call, landing, skill_line,
                 items=()):
        self.orderings, self.effects = orderings, effects
        self.called, self.score_call, self.landing = called, score_call, landing
        self.skill_line = skill_line
        self.items = list(items)
        self.noise = {}         # persona fixes 2: {name: the SD an order claim must reach}
        self.checks = None      # the Round-ahead checks (the bias line reads every result)

    def names_by(self, verdict):
        return [n for n, o in self.orderings.items()
                if n != insights.SCORE and o.verdict == verdict]

    def model_names(self, opt, verdict):
        """The Model item's names (persona fixes 2): the Overall score too,
        first, as the screen says it."""
        return model_names(opt, self.orderings, verdict)


def pair_counts(opt, orderings):
    """Specific wording: {label: (right, called pairs, all pairs)} — the
    evidence a verdict word carries wherever it shows, called of all
    (sceptic review: 'Partly: Tangy (5 of 9 called)'; it read '(5 of 5
    pairs)', the called ones only). All pairs: every pair within a checked
    Round, ties and too-close ones too (the Forecast check's own total)."""
    return {learned_label(opt, n): (o.right, o.pairs, o.pairs + getattr(o, "ties", 0)
                                    + getattr(o, "close", 0))
            for n, o in orderings.items()}


def model_names(opt, orderings, verdict):
    """[label] of the results with `verdict`, the Overall score first
    (persona fixes 2: the strip, the section and Make a round name it, so
    they agree with the Forecast check's record)."""
    return [learned_label(opt, n) for n, o in orderings.items() if o.verdict == verdict]


def _short(opt, text):
    """'F18' as the project says it: 'T18' in a study of trials."""
    letter = _say(opt, wording.FORMULATION_CAP)[:1] or "F"
    return re.sub(r"\bF(\d+)\b", rf"{letter}\1", _say(opt, text))


def learned_label(opt, name):
    return _say(opt, wording.OVERALL_SCORE_COLUMN) if name == insights.SCORE else name


def learned_facts_from(opt, checks):
    """LearnedFacts from the Round-ahead checks; the effects from the models
    already fitted for them (memo)."""
    # Persona fixes 2: an order claim only when the forecasts were one
    # noise SD of the result apart (learned.pair_noise), everywhere.
    noise = learned.pair_noise(opt, checks)
    orderings = learned.orderings(opt, checks, noise)
    # Review learned 2b: no effect is claimed on screen; learned.effects
    # stays (tested) for when Repeats measure the noise.
    effects = {}
    r, calls = learned.called_it(opt, checks, None, noise)
    score_call = (None if r is None else
                  learned.round_call(checks, insights.SCORE, r, noise.get(insights.SCORE) or 0.0))
    mine = [c for c in insights.ranged(checks)
            if c.name == insights.SCORE and c.round == r and c.fully_measured]
    landing = (sum(c.inside for c in mine), sum(c.measured > c.high for c in mine),
               sum(c.measured < c.low for c in mine))
    inside, total = learned.coverage(checks)
    fact = insights.skill(opt, checks)
    best = learned.skill_vs_best(opt, checks)
    skill_line = wording.learned_skill_line(
        inside, total, None if fact is None else fact.value,
        None if best is None else best[0], None if fact is None else fact.n)
    items = []
    if r is not None and learned.is_latest(opt, r):
        # Every Predictable measurement: learned.shown leaves out the ones
        # measured all the same, then keeps 1 + CALLED_PANELS.
        named = ([(insights.SCORE, score_call)] if score_call is not None else []) + calls
        items = [(n, call, learned.round_checks(checks, n, r)) for n, call in named
                 if call.pairs or learned.round_checks(checks, n, r)]
    lf = LearnedFacts(orderings, effects, (r, calls), score_call, landing, skill_line, items)
    lf.noise = noise
    lf.checks = list(checks)
    # Persona fixes: the same record in plain words, the terms of art last.
    lf.plain_line = wording.skill_plain(inside, total, None if fact is None else fact.value,
                                        None if best is None else best[0])
    # Fix round 3: the Overall score's checks with no forecast, named.
    left = sum(1 for c in checks if c.name == insights.SCORE and c.fully_measured
               and getattr(c, "no_forecast", False))
    lf.plain_line = wording.with_left_out(lf.plain_line, left,
                                          _say(opt, wording.OVERALL_SCORE_COLUMN),
                                          wording.LEFT_OUT_OF_INSIDE)
    return lf


def learned_facts(opt, computed):
    """learned_facts_from, once per data_key and tasting noise (cached in
    the session; the noise decides which pairs are too close to call)."""
    key = _learned_key(opt)
    hit = st.session_state.get(LEARNED_KEY)
    if hit and hit[0] == key:
        return hit[1]
    try:
        value = learned_facts_from(opt, computed["checks"] or ())
    except Exception:
        value = None
    st.session_state[LEARNED_KEY] = (key, value)
    return value


def learned_cached(opt):
    """This data's LearnedFacts when already worked out, else None."""
    hit = st.session_state.get(LEARNED_KEY)
    return hit[1] if hit and hit[0] == _learned_key(opt) else None


def _learned_key(opt):
    return (insights.data_key(opt), insights.assumed_noise(opt))


def capped(value, top, d=1):
    """A prediction, never at or past the top of its Range: '≥ 15', and on
    the Overall score's 0–100 '≥ 99' (review learned, owner: no 100s)."""
    text = f"{value:.{d}f}"
    if float(text) >= top - 1e-9:
        shown = top - 1 if abs(top - 100.0) < 1e-9 else top
        return f"≥ {shown:g}"
    return text


def capped_range(p, top, d=1):
    """Review critique (fix 12): a forecast and its 95 % range that never
    reads a top below its point: '13.5 (11.1–15)'; at the top, the point
    capped as before and the range ending at the highest possible value:
    '≥ 99 (93.4–100)' (it read '99.5 (93.4–≥ 99)'). `p` a Prediction or a
    Check. ' to ' when a bound is negative."""
    mean = getattr(p, "mean", None)
    mean = getattr(p, "predicted", 0.0) if mean is None else mean
    low, high = float(p.low), float(p.high)
    sep = " to " if low < 0 or high < 0 else "–"
    # Round-2 fixes (P3 #19): the range's top with the point's decimals.
    top_text = f"{top:.{d}f}"
    hi_text = top_text if high >= top - 1e-9 or float(f"{high:.{d}f}") >= top else f"{high:.{d}f}"
    lo_text = f"{min(low, top):.{d}f}"
    return f"{capped(min(mean, top), top, d)} ({lo_text}{sep}{hi_text})"


def wilson(right, n, z=insights.Z95):
    """The 95 % Wilson interval of right / n, in percent."""
    if not n:
        return None, None
    p = right / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * max(0.0, centre - half), 100 * min(1.0, centre + half)


_PARTLY_BLUE = {"light": "#8fb3dc", "dark": "#4f7fb4"}
_CHANCE_GREY = {"light": "#dfe2e7", "dark": "#3a404a"}


def learned_rows(opt, lf):
    """The ordering chart's rows: per result its share right, its chance
    range from the same permutation null its verdict uses (learned.Ordering
    .chance_share), its verdict; a result with fewer than learned.MIN_PAIRS
    counted pairs is 'too few pairs', with no dot and no range."""
    rows = []
    for name, o in lf.orderings.items():
        if not o.pairs and not o.ties and not o.close:
            continue
        # Fix round 2: the one form of a verdict's pairs, on its own line.
        counted = wording.called_of(o.right, o.pairs, o.pairs + o.ties + o.close)
        label = f"{learned_label(opt, name)} ({counted})"
        few = o.pairs < learned.MIN_PAIRS
        lo, hi = (None, None) if few else o.chance_share()
        if hi is not None and hi >= 1.0 - 1e-9 and o.right >= o.pairs:
            # Round-2 fixes (P2 #11): a bar to 100 % says why, on a second
            # line of its label.
            label += f"|({_say(opt, wording.learned_full_chance(len(o.rounds)))})"
        rows.append({"name": label, "pct": None if few else 100 * o.right / o.pairs,
                     "result": learned_label(opt, name),
                     "lo": None if lo is None else 100 * lo,
                     "hi": None if hi is None else 100 * hi,
                     "verdict": wording.LEARNED_WORDS[o.verdict], "few": few,
                     "n": counted, "p": o.p,
                     "chance": _say(opt, wording.LEARNED_KEY_CHANCE)})
    return rows


def learned_chart(opt, lf):
    """Review critique (fix 3): per result, the share of within-Round pairs
    ordered right (dot) against the range chance gives (grey bar: the exact
    within-Round permutation null, the very test the verdict uses, so the
    dot sits past the bar exactly when the result is Predictable). Fewer
    than MIN_PAIRS pairs: grey 'too few pairs', no dot. A legend names the
    colours."""
    rows = learned_rows(opt, lf)
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    order = list(frame["name"])
    blue, grey = model_blue(), muted()
    words = [wording.LEARNED_WORDS[k] for k in (learned.PREDICTABLE, learned.PARTLY,
                                                learned.NOT_YET)]
    chance = _say(opt, wording.LEARNED_KEY_CHANCE)
    colour = alt.Scale(domain=words + [chance],
                       range=[blue, _PARTLY_BLUE[theme()], mark("neutral")[0],
                              _CHANCE_GREY[theme()]])
    legend = alt.Legend(title=None, orient="bottom", labelLimit=0, symbolSize=90)
    # Round-2 fixes (P2 #11): the chance swatch is a bar, as drawn.
    bar_legend = alt.Legend(title=None, orient="bottom", labelLimit=0, symbolSize=260,
                            symbolType="M-1,-0.35L1,-0.35L1,0.35L-1,0.35Z", symbolStrokeWidth=0)
    y = alt.Y("name:N", sort=order, title=None,
              axis=alt.Axis(labelLimit=300, ticks=False, domain=False,
                            labelExpr="split(datum.label, '|')"))
    x = alt.X("pct:Q", title=_say(opt, wording.LEARNED_CHART_X),
              scale=alt.Scale(domain=[0, 100], padding=8),
              axis=alt.Axis(values=[0, 25, 50, 75, 100]))
    tip = [alt.Tooltip("result:N", title=_say(opt, wording.LEARNED_COL_RESULT)),
           alt.Tooltip("verdict:N", title=_say(opt, wording.LEARNED_COL_VERDICT)),
           alt.Tooltip("n:N", title=_say(opt, wording.LEARNED_COL_PAIRS)),
           alt.Tooltip("pct:Q", format=".0f", title="%")]
    counted = frame[~frame["few"]]
    few = frame[frame["few"]].assign(x=50.0, t=_say(opt, wording.learned_too_few(
        learned.MIN_PAIRS)))
    layers = []
    if not counted.empty:
        base = alt.Chart(counted)
        chance_scale = alt.Scale(domain=[chance], range=[_CHANCE_GREY[theme()]])
        dot_scale = alt.Scale(domain=words, range=[blue, _PARTLY_BLUE[theme()],
                                                   mark("neutral")[0]])
        dark = "#e6e9ef" if theme() == "dark" else "#1f2937"
        layers += [base.mark_bar(height=14, cornerRadius=3).encode(
                       y=y, x=alt.X("lo:Q", scale=alt.Scale(domain=[0, 100], padding=8)),
                       x2="hi:Q", fill=alt.Fill("chance:N", scale=chance_scale,
                                                legend=bar_legend)),
                   # Partly dots carry a dark outline (P2 #11), so the pale
                   # blue never reads as Predicts.
                   base.mark_point(filled=True, size=120, opacity=1, strokeWidth=1.5).encode(
                       y=y, x=x, color=alt.Color("verdict:N", scale=dot_scale, legend=legend),
                       stroke=alt.condition(
                           alt.FieldEqualPredicate(field="verdict", equal=words[1]),
                           alt.value(dark), alt.value("transparent")),
                       tooltip=tip)]
    if not few.empty:
        layers.append(alt.Chart(few).mark_text(color=grey, fontSize=FONT_LABEL,
                                               fontStyle="italic").encode(
            y=y, x=alt.X("x:Q", scale=alt.Scale(domain=[0, 100], padding=8),
                         title=_say(opt, wording.LEARNED_CHART_X),
                         axis=alt.Axis(values=[0, 25, 50, 75, 100])),
            text="t:N"))
    extra = max(str(n).count("|") for n in frame["name"])
    return readable(alt.layer(*layers).properties(
        width=_LEARNED_W, height=alt.Step(LEARNED_ROW_H + 12 * extra)))


_LEARNED_W = 640


def measured_groups(cks):
    """The Round's formulations by measured value, highest first, equal
    values together: [[16, 17], [18]]."""
    levels = sorted({c.measured for c in cks}, reverse=True)
    return [sorted(int(c.formulation) for c in cks if c.measured == v) for v in levels]


def _is_target(opt, name):
    if name == insights.SCORE:
        return False
    return any(o['name'] == name and o.get('goal') == 'target' for o in opt.objectives)


def check_bias_line(opt, lf):
    """Persona fixes 2: 'Forecasts ran about 2 high on Tangy.' — every
    result of the Round whose measured values all sit on one side of their
    forecasts by more than its noise SD (learned.round_bias), the largest
    first; '' with none."""
    r = lf.called[0]
    if r is None or not lf.items:
        return ""
    names = [insights.SCORE] + [n for n in lf.orderings if n != insights.SCORE]
    cells = []
    for name in names:
        cks = (learned.round_checks(lf.checks, name, r) if lf.checks is not None
               else next((it[2] for it in lf.items if it[0] == name), []))
        g = learned.round_bias(cks, lf.noise.get(name) or 0.0)
        if g is not None:
            cells.append((learned_label(opt, name), g))
    cells.sort(key=lambda c: -abs(c[1]))
    return _say(opt, wording.check_bias(cells))


def called_line(opt, lf):
    """The Forecast check's headline from the facts (persona fixes 2):
    where the Overall scores landed against their 95 % ranges (a miss on
    either side), then the order of the pairs that count — '**Round 6:
    scored higher than forecast.** 2 of 3 above their range; order too
    close to call.' A measurement leads when the Overall score is not
    there. None with nothing to check."""
    if not lf.items:
        return None
    r = lf.called[0]
    lead = next((it for it in lf.items if it[0] == insights.SCORE), lf.items[0])
    name, call, cks = lead
    # Sceptic review: a range over the whole Range is no forecast, counted
    # neither in nor outside.
    ranged = [c for c in cks if not no_forecast(opt, c)]
    inside = sum(1 for c in ranged if c.low <= c.measured <= c.high)
    above = sum(1 for c in ranged if c.measured > c.high)
    below = sum(1 for c in ranged if c.measured < c.low)
    # Review critique: measured ties are counted aloud too ("1 tie").
    order = wording.check_order(call.right, call.counted, call.close,
                                tied=bool(cks) and learned.all_tied(cks) and len(cks) > 1,
                                ties=call.ties)
    subject = None if name == insights.SCORE else learned_label(opt, name)
    return _short(opt, wording.check_line(r, inside, above, below, order, subject,
                                          none=len(cks) - len(ranged)))


def no_forecast(opt, c):
    """Sceptic review: whether the 95 % range of check (or prediction) `c`
    covers its result's whole Range — its low within NO_FORECAST_EDGE (1 %)
    of the Range's width of the Range's low, and its high as close to the
    Range's high: a range of the whole Range forecasts nothing (Creaminess
    0.0–15.0 on 0–15). False for a name with no Range."""
    return bool(getattr(c, "no_forecast", False)) or insights.covers_range(
        opt, c.name, c.low, c.high)


def round_predicted_frame(opt, checks, r):
    """Results' Round view: per formulation of Round `r`, the Overall score
    and (review critique, fix 11) every measurement, Measured beside
    Predicted — the Round-ahead forecast and its 95 % range — and where it
    landed: In, Above or Below that range. None when nothing was checked."""
    if not checks or r not in insights._ordered_rounds(checks):
        return None
    rows = []
    for name in learned.names(opt):
        _, hi = insights._range_of(opt, name)
        d = 1 if name == insights.SCORE else max(1, decimals(opt, name))
        for c in learned.round_checks(checks, name, r):
            side = ("above" if c.measured > c.high + 1e-9 else
                    "below" if c.measured < c.low - 1e-9 else "in")
            # Sceptic review: a range over the whole Range forecasts nothing.
            none = no_forecast(opt, c)
            rows.append({
                _say(opt, wording.RESULTS_RESULT_COLUMN): learned_label(opt, name),
                # Round-2 fixes (P3 #18): "F16" in tables and charts.
                _say(opt, wording.FORMULATION_CAP): _short(opt, f"F{int(c.formulation)}"),
                _say(opt, wording.MEASURED_COLUMN): f"{c.measured:.{d}f}",
                _say(opt, wording.RESULTS_PREDICTED_COLUMN): (
                    _say(opt, wording.NO_FORECAST) if none else capped_range(c, hi, d)),
                _say(opt, wording.RESULTS_LANDED_COLUMN): (
                    wording.TOO_UNCERTAIN if none else _say(opt, wording.RESULTS_LANDED[side]))})
    return pd.DataFrame(rows) if rows else None


WIDEN_SAID = 1.05      # a range's widening is said from this factor


def no_forecast_items(opt, checks, r):
    """Fix round 2: [(name, Range text, widen text or None)] for each result
    of Round `r` with a no-forecast 95 % range, for the Results "?" — the
    fact (its Range) and, when the range was widened after earlier misses,
    by how much."""
    out = []
    for name in learned.names(opt):
        cks = [c for c in learned.round_checks(checks, name, r) if no_forecast(opt, c)]
        if not cks:
            continue
        lo, hi = insights._range_of(opt, name)
        f = max(float(getattr(c, "widen", 1.0) or 1.0) for c in cks)
        # Fix round 3: widening named from ×1.05 (one decimal), never ×1.0.
        out.append((learned_label(opt, name), f"{lo:g}–{hi:g}",
                    f"{f:.1f}" if f >= WIDEN_SAID else None))
    return out


def called_so_far(opt, lf):
    """The cumulative record of the results drawn, a second short line:
    'So far: Overall 10 of 11 pairs, Consistency 8 of 8.'"""
    cells = []
    for name in learned.shown(lf.items):
        o = lf.orderings.get(name[0])
        if o is None:
            continue
        label = wording.CALLED_OVERALL if name[0] == insights.SCORE else learned_label(opt, name[0])
        cells.append((label, o.right, o.pairs))
    if cells and not any(p for _, _, p in cells) and any(
            getattr(lf.orderings.get(it[0]), "close", 0) for it in learned.shown(lf.items)):
        # Persona fixes 2: nothing far enough apart to count, said.
        return _say(opt, wording.CHECK_SO_FAR_NONE)
    return _say(opt, wording.called_so_far(cells))


def called_detail(opt, lf):
    """The Forecast check's "?" (review critique: about 120 words, plain
    first): the chart's rule, each result's order line, the tasting noise
    each claim had to reach (assumed, said), the experts' pairs last."""
    lines, pairs = [], []
    for name, call, cks in lf.items:
        label = learned_label(opt, name)
        if learned.all_tied(cks):
            lines.append(wording.called_tied(label))
            continue
        line = wording.check_measure_line(label, call.right, call.counted, call.close,
                                          learned.order_held(call))
        if name == insights.SCORE and sum(lf.landing):
            line = line.rstrip(".") + "; " + wording.called_range(*lf.landing)
        lines.append(line)
        bits = [wording.check_pair_close(p.high, p.low) if p.close
                else wording.called_pair(p.high, p.low, p.right) for p in call.pairs]
        top = ("" if _is_target(opt, name)
               else wording.called_top(call.top_predicted, call.top_measured, call.top_right))
        if top:
            bits.append(top)
        if bits:
            pairs.append(f"{label}: " + " · ".join(bits))
    noise = [(learned_label(opt, n), f"{lf.noise[n]:.1f}") for n, _, _ in lf.items
             if lf.noise.get(n)]
    try:
        assumed = learned.overall_noise(opt)[1]
    except Exception:
        assumed = True
    k = len(learned.scored_measurements(opt)) if any(n != insights.SCORE for n, _, _ in lf.items) \
        else None
    # Final demo (A3): the pairs called (raw and adjusted chance) and the
    # values just inside their range, after the noise they had to reach.
    out = [wording.CALLED_RULE, " ".join(lines),
           wording.check_noise(noise, assumed, k, noise_sd_text(opt)), near_edge_text(opt, lf)]
    if pairs:
        out.append(wording.for_experts("; ".join(pairs) + "."))
    return _short(opt, wording.brief("\n\n".join(t for t in out if t)))


def _chance_words(p):
    """A p-value in plain words: 'about 6 in 100 (p = 0.06)'."""
    n = max(1, int(round(p * 100))) if p > 0 else 0
    return f"{n} in 100" if n < 100 else "certain"


def pairs_called_text(opt, lf):
    """Final demo (A3): the Overall score's pairs, called of all, and the
    chance of as many right by luck, raw and adjusted ('' without pairs)."""
    o = lf.orderings.get(insights.SCORE) if lf is not None else None
    if o is None:
        return ""
    total = o.pairs + o.ties + o.close
    if not total:
        return ""
    return _say(opt, wording.pairs_called_help(
        learned_label(opt, insights.SCORE), o.right, o.pairs, total, o.close, o.ties,
        _chance_words(o.p) if o.pairs else "",
        _chance_words(o.p_holm) if o.p_holm is not None else None, o.tested))


NEAR_EDGE = 0.2         # a value in its range by less than this: its margin in the "?"


def near_edge_text(opt, lf):
    """Final demo (A3): 'Close to the range edge: F18 in by 0.07.' — the
    drawn values inside their 95 % range by under NEAR_EDGE; '' with none."""
    items = []
    for name, _call, cks in learned.shown(lf.items):
        for c in sorted(cks, key=lambda c: c.formulation):
            if not (c.low <= c.measured <= c.high):
                continue
            margin = min(c.measured - c.low, c.high - c.measured)
            _lo, top = insights._range_of(opt, name)
            if margin < NEAR_EDGE and not (c.high >= top - 1e-9 and c.high - c.measured
                                           == margin):
                item = wording.near_edge_item(int(c.formulation), f"{margin:.2f}")
                items.append(item if name == insights.SCORE
                             else f"{learned_label(opt, name)} {item}")
    return _short(opt, wording.near_edge(items)) if items else ""


def check_frame(opt, name, cks):
    """Review critique (fix 1): one Forecast check panel's rows, one per
    formulation, highest forecast first (row order is the forecast order):
    its forecast, 95 % range and measured value on the result's own values
    (one shared axis), and where the measured value landed — "in", "above"
    or "below" its range."""
    _, top = insights._range_of(opt, name)
    d = 1 if name == insights.SCORE else max(1, decimals(opt, name))
    ranked = sorted(cks, key=lambda c: (-learned._rank_value(c), c.formulation))
    rows = []
    for k, c in enumerate(ranked):
        # Fix round 2: a whole-Range range is "No forecast", neither side.
        side = ("none" if no_forecast(opt, c) else
                "above" if c.measured > c.high + 1e-9 else
                "below" if c.measured < c.low - 1e-9 else "in")
        rows.append({"f": _short(opt, f"F{int(c.formulation)}"), "k": k,
                     "predicted": float(min(c.predicted, top)), "low": float(c.low),
                     "high": float(c.high), "measured": float(c.measured), "side": side,
                     # Round-2 fixes (P2 #9): a miss prints its value: "98.3 above".
                     "miss": "" if side in ("in", "none") else
                     f"{c.measured:.{d}f} {_say(opt, wording.CHECK_SIDE[side])}",
                     "range": capped_range(c, top, d),
                     "value": f"{c.measured:.{d}f}"})
    return pd.DataFrame(rows)


def _check_domain(frame, lo_bound, hi_bound):
    """A panel's x domain: its ranges and measured values, padded, room
    for an "above" / "below" label on the side a miss sits, within the
    result's own possible values."""
    vals = list(frame["low"]) + list(frame["high"]) + list(frame["measured"])
    lo, hi = min(vals), max(vals)
    span = max(hi - lo, 1e-6)
    left = lo - 0.1 * span - (0.36 * span if (frame["side"] == "below").any() else 0)
    right = hi + 0.06 * span + (0.36 * span if (frame["side"] == "above").any() else 0)
    return [max(left, lo_bound - 0.02 * span), min(right, hi_bound + 0.3 * span)]


_CHECK_BAND = {"light": "#c9dbef", "dark": "#2c4a6e"}   # the forecast range, light


def called_chart(opt, lf):
    """The Forecast check (review critique, fix 1: the own-scale
    slopegraphs read as ranks, not values): per result drawn — the Overall
    score, then the measurements Predictable before the Round — one row per
    formulation, highest forecast first, on one shared axis: the forecast's
    95 % range a light bar, the forecast a tick, the measured value a dot,
    a miss (outside its range, either side) in the one accent colour and
    labelled "above" or "below". None with nothing to draw."""
    drawn = learned.shown(lf.items)
    if not drawn:
        return None
    ink, _ = mark("bar_latest")
    blue, band = model_blue(), _CHECK_BAND[theme()]
    accent = outline_colour()
    names = [_say(opt, t) for t in (wording.CHECK_KEY_RANGE, wording.CHECK_KEY_FORECAST,
                                     wording.CHECK_KEY_IN, wording.CHECK_KEY_OUT,
                                     wording.NO_FORECAST)]
    # Round-2 fixes (P2 #9): only the keys drawn, at full strength.
    frames = [check_frame(opt, name, cks) for name, _, cks in drawn]
    sides = {sd for f in frames if not f.empty for sd in f["side"]}
    used = names[:2] + ([names[2]] if "in" in sides else []) + (
        [names[3]] if sides - {"in", "none"} else []) + ([names[4]] if "none" in sides else [])
    colour = alt.Scale(domain=used, range=[dict(zip(names, (band, blue, ink, accent,
                                                            mark("neutral")[0])))[u]
                                           for u in used])
    legend = alt.Legend(title=None, orient="bottom", labelLimit=0, symbolSize=90,
                        symbolOpacity=1)
    width = CALLED_W
    panels = []
    for (name, call, cks), frame in zip(drawn, frames):
        if frame.empty:
            continue
        frame["k_range"], frame["k_fc"] = names[0], names[1]
        frame["k_dot"] = [names[2] if s == "in" else names[4] if s == "none" else names[3]
                          for s in frame["side"]]
        lo_b, hi_b = insights._range_of(opt, name)
        xs = alt.Scale(domain=_check_domain(frame, lo_b, hi_b), nice=False, zero=False)
        rows = list(frame["f"])
        y = alt.Y("f:N", sort=rows, title=None,
                  axis=alt.Axis(ticks=False, domain=False, labelFontSize=FONT_LABEL,
                                labelPadding=6))
        x = alt.X("measured:Q", scale=xs, title=None,
                  axis=alt.Axis(tickCount=4, grid=True, labelFontSize=FONT_SMALL,
                                labelColor=muted()))
        tip = [alt.Tooltip("f:N", title=_say(opt, wording.FORMULATION_CAP)),
               alt.Tooltip("range:N", title=_say(opt, wording.CHECK_TIP_FORECAST)),
               alt.Tooltip("value:N", title=_say(opt, wording.CALLED_TIP_MEASURED))]
        base = alt.Chart(frame)
        bars = base.mark_bar(height=12, cornerRadius=3).encode(
            x=alt.X("low:Q", scale=xs, title=None), x2="high:Q", y=y,
            color=alt.Color("k_range:N", scale=colour, legend=legend), tooltip=tip)
        ticks = base.mark_tick(thickness=2.5, size=18, orient="vertical").encode(
            x=alt.X("predicted:Q", scale=xs, title=None), y=y,
            color=alt.Color("k_fc:N", scale=colour, legend=legend), tooltip=tip)
        dots = base.mark_point(filled=True, size=90, opacity=1, stroke="white",
                               strokeWidth=1).encode(
            x=x, y=y, color=alt.Color("k_dot:N", scale=colour, legend=legend), tooltip=tip)
        above = base.transform_filter("datum.side == 'above'").mark_text(
            align="left", dx=9, fontSize=FONT_LABEL, fontWeight="bold", color=accent).encode(
            x=alt.X("measured:Q", scale=xs, title=None), y=y, text="miss:N")
        below = base.transform_filter("datum.side == 'below'").mark_text(
            align="right", dx=-9, fontSize=FONT_LABEL, fontWeight="bold", color=accent).encode(
            x=alt.X("measured:Q", scale=xs, title=None), y=y, text="miss:N")
        title = {"text": _say(opt, learned_label(opt, name)), "fontSize": FONT_TITLE,
                 "anchor": "start", "fontWeight": "bold", "offset": 6}
        if not call.counted:
            title.update(subtitle=_say(opt, wording.CHECK_TOO_CLOSE), subtitleFontSize=FONT_SMALL,
                         subtitleColor=muted())
        panels.append(alt.layer(bars, ticks, dots, above, below).properties(
            width=width, height=alt.Step(CALLED_ROW_H), title=alt.TitleParams(**title)))
    if not panels:
        return None
    # Round-2 fixes (P2 #9): one panel under another, each the full width.
    return readable(alt.vconcat(*panels, spacing=_CALLED_SPACING + 8)
                    .resolve_scale(color="shared", x="independent")
                    .configure_view(stroke=None))


_CALLED_SPACING = 16


def _called_top(opt, slot, lf):
    """Called it?, under the status strip: its answer line (the facts'
    words, the pairs in its "?") and the panels; nothing unless the
    project's latest Round was scored and forecast before tasting."""
    if lf is None:
        return
    line = called_line(opt, lf)
    if line is None:
        return
    with slot.container(key=CALLED_KEY, gap="small"):
        # Round-2 fixes (P2 #9): its own section, right after What next?;
        # slim Review: the heading is the section's (_section).
        # Rebuilt framing: worked out again today, never kept from the time.
        # Fix round 1: the "?" says exactly what the forecasts are.
        st.caption(_say(opt, wording.REBUILT_LABEL), help=_say(opt, wording.rebuilt_help(
            sample_projects.is_sample(getattr(opt, "project_name", None)))))
        st.markdown(line, help=called_detail(opt, lf))
        bias = check_bias_line(opt, lf)
        if bias:
            st.markdown(bias)
        # Slim Review: the "So far:" count and the model's verdicts, one line.
        verdicts = learned_verdict_line(opt, lf)
        if verdicts:
            st.markdown(verdicts[0], help=verdicts[1])
        try:
            chart = called_chart(opt, lf)
        except Exception:
            chart = None
        if chart is not None:
            show_chart(chart, width="content")
            st.caption(_say(opt, wording.CALLED_CAPTION))
            absent = called_absent(opt, lf)
            if absent:
                st.caption(_say(opt, wording.called_shown(lf.called[0])),
                           help=_say(opt, wording.called_shown_help(absent, lf.called[0])))


def called_absent(opt, lf):
    """Round-2 fixes (P3 #18): the results the Model verdicts name today
    (Predicts or Partly) that the Forecast check does not draw — it shows
    those predictable before the Round — as labels; [] with none."""
    drawn = {it[0] for it in learned.shown(lf.items)}
    named = [n for n, o in lf.orderings.items()
             if o.verdict in (learned.PREDICTABLE, learned.PARTLY) and n != insights.SCORE]
    return [learned_label(opt, n) for n in named if n not in drawn]


def _has_verdicts(lf):
    return lf is not None and any(o.pairs or o.ties for o in lf.orderings.values())


def _tested(lf):
    """How many results the verdicts were adjusted over."""
    return max((o.tested for o in lf.orderings.values() if o.pairs), default=0)


def _line_words(text):
    """Every word of a line, numbers too (the owner's limits count them)."""
    plain = re.sub(r":\w+\[(.*?)\]", r"\1", str(text)).replace("**", "")
    return len([w for w in plain.split() if re.search(r"\w", w)])


def learned_verdict_line(opt, lf):
    """(line, its "?") — the Forecast check's one verdict line (slim
    Review, owner 2026-10-08), merged with its old "So far:" count: 'So
    far: Overall score 9 of 24 pairs called, 7 right. Predicts:
    Consistency (11 of 24 called, 11 right).'
    — the Overall score's within-Round pairs, then the verdict groups
    (Predicts, Partly, Not yet; the Overall score named in its own) as
    many as keep the line to LEARNED_LINE_WORDS words, every word counted;
    when not even the first group fits beside the count, the groups alone.
    The details fold names every result. None with no verdict."""
    if not _has_verdicts(lf):
        return None
    o = lf.orderings.get(insights.SCORE)
    # Fewer counted pairs than a verdict needs: said in the line.
    # Specific wording: under the pairs a verdict needs, the count it needs.
    score = learned_label(opt, insights.SCORE)
    count = None
    total = None if o is None else o.pairs + o.ties + o.close
    if o is not None and o.pairs:
        # Final demo (A3): the pairs that could be called, of all.
        count = (wording.learned_count_needed(score, o.right, o.pairs, learned.MIN_PAIRS,
                                              total=total)
                 if o.pairs < learned.MIN_PAIRS else
                 wording.learned_count(score, o.right, o.pairs, total=total))
    counts = pair_counts(opt, lf.orderings)
    groups = [(wording.LEARNED_WORDS[v], lf.model_names(opt, v))
              for v in (learned.PREDICTABLE, learned.PARTLY, learned.NOT_YET)]
    groups = [(w, names) for w, names in groups if names]
    # Specific wording: each name with its pairs. Beside the count (which
    # says "pairs" and gives the Overall score's), the details chart's
    # form: 'Tangy (5 of 12 called, 5 right)'.
    beside = [(w, [n if n == score else wording.named_pairs_short(n, counts.get(n))
                   for n in names]) for w, names in groups]
    # Alone, the first says the unit: 'Overall score (5 of 12 pairs called, 5
    # right), Tangy (5 of 12 called, 5 right)'.
    counted = [n for w, names in groups for n in names if counts.get(n, (0, 0))[1]]
    said = {n: (wording.named_pairs([n], counts) if i == 0 else
                wording.named_pairs_short(n, counts.get(n))) for i, n in enumerate(counted)}
    alone = [(w, [said.get(n) or wording.named_pairs_short(n, counts.get(n)) for n in names])
             for w, names in groups]
    called, said_c = [], said
    if o is not None and o.pairs and score in said and total and total > o.pairs:
        # Final demo (A3): 'Overall score (5 of 12 pairs called, 5 right)',
        # when it fits; else the form before.
        said_c = dict(said, **{score: wording.score_pairs_called(score, o.right, o.pairs,
                                                                 total)})
        called = [(w, [said_c.get(n) or wording.named_pairs_short(n, counts.get(n))
                       for n in names]) for w, names in groups]
    # The count and as many groups as fit; else the groups alone with their
    # pairs, else bare (a verdict is never dropped for the count); else the
    # count alone.
    # Sceptic review: the called-of-all forms run longer; before every count
    # is dropped, the first name's count alone ('Partly: Overall score (15
    # of 24 pairs called, 10 right), Consistency, Tangy.').
    first = counted[0] if counted else None
    long_ = said_c if called else said
    lead = [(w, [long_.get(n) if n == first else n for n in names]) for w, names in groups]
    # More verdict groups first: at each number of groups, every count, then
    # the first count alone (named_pairs and score_pairs_called now read the
    # same, so called and alone differ only in the Overall score's place).
    tries = [(count, beside[:k]) for k in range(len(groups), 0, -1) if count]
    for k in range(len(groups), 0, -1):
        tries += [(None, called[:k])] if k <= len(called) else []
        tries += [(None, alone[:k])]
        tries += [(None, lead[:k])] if first else []
    tries += [(None, groups[:k]) for k in range(len(groups), 0, -1)]
    tries += [(count, [])] if count else []
    line = next((t for t in (wording.learned_line(c, g) for c, g in tries)
                 if t and _line_words(t) <= wording.LEARNED_LINE_WORDS), None)
    if line is None:
        # Long names: the first group alone, whatever its length.
        line = wording.learned_line(None, alone[:1]) or wording.learned_line(count, [])
    # Final demo (A3): the "?" opens on the pairs called and the chance by luck.
    return _short(opt, line), _say(opt, wording.brief("\n\n".join(t for t in (
        pairs_called_text(opt, lf), wording.learned_verdicts_help(_tested(lf))) if t)))


def _learned_section(opt, slot, computed, facts):
    """Slim Review (owner, 2026-10-08): "What has the model learned?" is no
    section of its own. Its one line sits in the Forecast check
    (_called_top, learned_verdict_line); here, when the Forecast check has
    nothing to draw, that line alone; then the ordering chart against
    chance in a closed fold, "Details: ordering by result". A disagreeing
    Repeat's warning stays in the strip and in What next?."""
    lf = learned_facts(opt, computed)
    if not _has_verdicts(lf):
        with slot.answer.container():
            st.markdown(_say(opt, wording.LEARNED_NONE_YET))
        return
    if called_line(opt, lf) is None:
        verdicts = learned_verdict_line(opt, lf)
        if verdicts:
            with slot.answer.container():
                st.markdown(verdicts[0], help=verdicts[1])
    # Review critique (fix 3, 13): plain first — the rule and the
    # multiplicity — then the terms of art; the counts are on the chart.
    tested = _tested(lf)
    help_ = wording.brief("\n\n".join(t for t in (
        wording.LEARNED_RULE, wording.learned_multiplicity(tested) if tested > 1 else "",
        wording.for_experts(wording.LEARNED_EXPERT)) if t))
    with slot.body:
        try:
            chart = learned_chart(opt, lf)
        except Exception:
            chart = None
        if chart is None:
            return
        with st.expander(_say(opt, wording.LEARNED_DETAILS_TITLE), expanded=False,
                         key=LEARNED_FOLD_KEY):
            show_chart(chart, width="content")
            st.caption(_say(opt, wording.learned_caption(tested) if tested > 1
                            else wording.LEARNED_CAPTION), help=_say(opt, help_))


LEARNED_FOLD_KEY = "insights_learned_fold"


def _expert_lines(lf):
    """'For experts: …': the skill figures and the method's terms of art."""
    return wording.for_experts(" ".join(t for t in (lf.skill_line, wording.EXPERT_TERMS) if t))


def learned_card(opt, lf, card, facts=None):
    """The strip's Model item from the section's verdicts; the old card's
    warning lines (red) kept. Its "?" (persona fixes): a plain sentence
    first, the records in plain words, the rule, the terms of art last;
    the old Predictions card's sentence only when it warns (its "still very
    uncertain" contradicted Predicts)."""
    counts = pair_counts(opt, lf.orderings)
    word, line = wording.learned_strip_counted(lf.model_names(opt, learned.PREDICTABLE),
                                               lf.model_names(opt, learned.PARTLY),
                                               lf.model_names(opt, learned.NOT_YET), counts)
    old = card.lines if card is not None else []
    warns = [(t, w) for t, w in old if w]
    # The old card's further lines (a Repeat request, the measurements'
    # misses) stay, under the names: the strip shows a warning or the names.
    rest = [(t, w) for t, w in old[1:] if not w]
    plain = wording.model_plain_counted(lf.model_names(opt, learned.PREDICTABLE),
                                        lf.model_names(opt, learned.PARTLY),
                                        lf.model_names(opt, learned.NOT_YET), counts)
    # Specific wording: the opening sentence gives the Predicts and Partly
    # names' pairs; the evidence line the rest, so no count is said twice.
    named = set(lf.model_names(opt, learned.PREDICTABLE) + lf.model_names(opt, learned.PARTLY))
    rows = evidence_rows(opt, lf, named)
    ev = wording.learned_evidence(rows) if rows else wording.LEARNED_EVIDENCE_ABOVE
    record = getattr(lf, "plain_line", "")
    h = getattr(facts, "headline", None) if facts is not None else None
    if h is not None and getattr(h, "total", 0):
        record = (record + " " + wording.with_left_out(wording.single_record(
            h.inside, h.total, len(getattr(h, "rounds", None) or ())),
            getattr(h, "no_forecast", 0))).strip()
    if rest and facts is not None and getattr(facts, "best", None) is not None:
        # The card's Repeat request, said plainly (it sits in What next?).
        # Round-2 fixes (P3 #18): the one form of the Repeat request.
        record += " " + wording.card_noise_repeat(facts.best.formulation) + "."
    # As every strip "?": plain first (one paragraph), the evidence, the rule.
    help_ = "\n\n".join(t for t in (f"{plain} {record}".strip(), ev, wording.LEARNED_RULE) if t)
    if card is not None and card.help and warns:
        # A warning stays explained, as before.
        help_ += f"\n\n{wording.LEARNED_RANGE_RECORD}\n\n{card.help}"
    expert = _expert_lines(lf)
    if record:
        # Round-2 fixes (P3 #19): no experts' line repeating the plain one.
        expert = re.sub(r" +", " ", re.sub(
            r"[^.:]*: \d+ of \d+ inside the 95\u00a0% predicted range\.", " ", expert))
    if card is not None and card.help and not warns and facts is not None:
        # The old card's exact rule and diagnosis stay, in the experts'
        # line; its plain sentence (it read "still very uncertain" beside
        # Predicts) does not.
        try:
            tip = tooltip_for(opt, "predictions", facts)
        except Exception:
            tip = None
        if tip:
            expert += " " + " ".join(tip.split("\n\n"))
    help_ += "\n\n" + expert
    # Review critique (fix 13): about 120 words, plain first, experts last
    # (it ran to about 450); the rest is in Method and definitions.
    return Card(_say(opt, word), warns + [(_say(opt, line), False)] + rest, None,
                _say(opt, wording.brief(help_)))


def evidence_rows(opt, lf, skip=()):
    """The Model item's evidence rows (label, right, called, all), one per
    ordering, leaving out the labels in `skip` that have pairs called (the
    "?"'s opening sentence says theirs)."""
    return [(learned_label(opt, n), o.right, o.pairs, o.pairs + getattr(o, "ties", 0)
             + getattr(o, "close", 0)) for n, o in lf.orderings.items()
            if learned_label(opt, n) not in skip or not o.pairs]


def how_learned_texts(opt, lf):
    """Method and definitions' paragraphs on the orderings: the rules, the
    evidence for every result (fix round 4: the Model "?" can drop its
    evidence paragraph for length; the numbers stay here), the skill."""
    rows = evidence_rows(opt, lf) if lf is not None else []
    evidence = wording.learned_evidence(rows) if any(r[2] for r in rows) else ""
    texts = (wording.LEARNED_RULE, evidence, wording.LEARNED_TOP_RULE,
             wording.PROFILER_NO_CLAIM,
             wording.profiler_hidden_rule(settings_only=not opt.has_ingredients()),
             lf.skill_line if lf is not None else "")
    return [t for t in texts if t]


def _how_learned(opt, computed):
    """Method and definitions: the ordering accuracy and verdict rules, the
    evidence, the stable-effect rule, the two skill baselines, and the
    Overall score's forecast against measured chart that the section
    replaced."""
    st.markdown(f"**{_say(opt, wording.METHOD_LEARNED_TITLE)}**")
    for text in how_learned_texts(opt, learned_cached(opt)):
        st.markdown(_say(opt, text))
    checks = [c for c in ((computed or {}).get("checks") or ()) if c.name == insights.SCORE]
    if checks:
        try:
            chart = forecast_chart(opt, checks)
        except Exception:
            chart = None
        if chart is not None:
            st.markdown(f"**{_say(opt, wording.METHOD_FORECAST_CHART)}**")
            show_chart(chart)
            st.caption(_say(opt, wording.STORY_TRUST_CAPTION))


def round_forecast_lines(opt, checks, rows=None):
    """(lines, help) for Make a round's forecast panel: one line
    per measurement Predictable on the checks so far, from the models fitted
    on the rows before the open Round; ([], None) with none."""
    rows = open_rows(opt) if rows is None else rows
    if not rows or not checks:
        return [], None
    ords = learned.orderings(opt, checks)
    # Persona fixes 2: the Overall score's row too when it is Predictable
    # (first, as the Model item names it), even if no measurement is.
    wanted = [n for n, o in ords.items() if o.verdict == learned.PREDICTABLE]
    if not wanted:
        return [], None
    with insights.memo_in_use(memo(opt)):
        preds = learned.forecast(opt, rows, wanted)
        score_fit = (insights.fit(opt, insights.SCORE, list(range(learned.round_start(opt))))
                     if insights.SCORE in wanted else None)
    factor = insights.column_factor(opt, checks) if score_fit is not None else 1.0
    lines = []
    for name in wanted:
        _, top = insights._range_of(opt, name)
        cells, means = [], {}
        for no in sorted(preds):
            p = preds[no].get(name)
            if p is None:
                continue
            if name == insights.SCORE:
                # Widened as that Round's check will be (the Round table's range).
                p = insights.widened(score_fit, p, factor)
            else:
                # Review critique (fix 10): by the measurement's own record.
                lo_r, hi_r = insights._range_of(opt, name)
                p = insights.widened(types.SimpleNamespace(low=lo_r, high=hi_r), p,
                                     insights.measure_widen(checks, name, opt.pending_batch_no))
            cells.append(f"F{no} {capped_range(p, top)}")
            means[no] = p.raw_mean if p.raw_mean is not None else p.mean
        label = learned_label(opt, name)
        if cells and _is_target(opt, name) and len(means) > 1:
            # Persona fixes: a measurement with a target has no "best"
            # highest; its forecast order, high to low, said neutrally.
            order = sorted(means, key=means.get, reverse=True)
            lines.append(wording.forecast_line(label, cells, None)
                         + wording.forecast_order_tail(order))
        elif cells:
            lines.append(wording.forecast_line(label, cells, learned.top_of(means)
                                               if len(means) > 1 else None))
    return lines, wording.forecast_help(learned.round_start(opt), opt.pending_batch_no)


def forecast_none_text(opt, checks):
    """Specific wording: with nothing Predictable, the result closest to
    it (Partly first, then the lowest chance p) and its pairs, from the
    orderings round_forecast_lines reads."""
    try:
        ords = learned.orderings(opt, checks)
    except Exception:
        return wording.LEARNED_FORECAST_NONE
    near = [o for o in ords.values() if o.pairs]
    if not near:
        return wording.forecast_none()
    o = min(near, key=lambda o: (o.verdict != learned.PARTLY, o.p, -o.right))
    return wording.forecast_none(learned_label(opt, o.name), o.right, o.pairs)


def forecast_panel(opt):
    """Make a round, above the results grid: the forecast before tasting,
    for Predictable measurements only (never fits for the verdict: it reads
    the checks the Review has)."""
    checks = cached_checks(opt)
    if not checks:
        return
    try:
        lines, help_ = round_forecast_lines(opt, checks)
    except Exception:
        return
    with st.container(border=True, key="round_forecast", gap="xsmall"):
        if not lines:
            st.caption(_say(opt, forecast_none_text(opt, checks)))
            return
        st.markdown(f"**{_say(opt, wording.LEARNED_FORECAST_TITLE)}**", help=_say(opt, help_))
        for line in lines:
            st.markdown(_short(opt, line))


def _preview_count(opt):
    records = opt.generation_records()
    try:
        n = int(records[max(records)].get("n", 3)) if records else 3
    except (TypeError, ValueError):
        n = 3
    return min(10, max(1, n))


class PreviewRow(tuple):
    """(k, role, score text, changes) as before, plus (persona fixes) the
    plain line that sets it apart, the predicted mean and the recipe."""

    def __new__(cls, k, role, text, changes, apart="", mean=None, recipe=None):
        row = super().__new__(cls, (k, role, text, changes))
        row.apart, row.mean, row.recipe = apart, mean, recipe
        return row


def apart_text(opt, recipe, role, binding=None):
    """What sets a suggestion apart (persona fixes), from its role and its
    two largest changes against Best so far (FoodOptimizer._variable_deltas):
    'Explores: less Tapioca starch and Nutritional yeast than anything
    tested' when they go below or above every formulation recorded; else
    against Best so far."""
    best = opt._best_recipe_index()
    if best is None:
        return wording.apart_line(role, [])
    here, there = insights.at_total(opt, recipe), insights.start_recipe(opt, best)
    tested = opt._recorded_before(None)
    variables = opt._by_name()
    out = []
    for name, delta in opt._variable_deltas(here, there)[:2]:
        setting = variables[name].get('category', 'ingredient') != 'ingredient'
        word = ("higher" if delta > 0 else "lower") if setting else (
            "more" if delta > 0 else "less")
        absent = variables[name].get('_absent_value', 0.0)
        vals = [float(r.get(name, absent)) for r in tested]
        v = float(here.get(name, absent))
        tol = 1e-6 * max(1.0, abs(v))
        beyond = bool(vals) and ((delta < 0 and v < min(vals) - tol)
                                 or (delta > 0 and v > max(vals) + tol))
        out.append((word, name, beyond))
    # Persona fixes 2: Next advises widening X's Lowest (Highest) while this
    # suggestion moves X up (down): said, so the two never read as a clash.
    other = None
    kind = binding.kind if binding is not None else None
    if kind in ("lowest", "highest"):
        away = ("more", "higher") if binding.kind == "lowest" else ("less", "lower")
        hit = next((it for it in out if it[1] == binding.name and it[0] in away), None)
        if hit is not None:
            out.remove(hit)
            other = wording.other_side(hit[0], hit[1])
    far = [(w, n) for w, n, b in out if b]
    if far:
        line = wording.apart_line(role, far, beyond=True)
    else:
        line = wording.apart_line(role, [(w, n) for w, n, _ in out])
    if other:
        line += f"; {other}" if out else f": {other}"
    return line


def preview_rows(opt, computed, binding=None):
    """The next suggestions as Generate would make them now: [PreviewRow
    (k, role, score text, top-2 changes against Best so far; .apart,
    .mean, .recipe)], cached per setup and next Formulation number; None
    when the preview failed."""
    key = preview_key(opt, binding)
    if key in _PREVIEW_CACHE:
        _PREVIEW_CACHE.move_to_end(key)
        return _PREVIEW_CACHE[key]
    n = _preview_count(opt)
    try:
        recipes = learned.preview_next(opt, n)
    except Exception:
        recipes = None
    out = None
    if recipes:
        score = (computed.get("fitted") or {}).get(insights.SCORE)
        factor = insights.column_factor(opt, computed.get("checks") or ())
        rows = [{"formulation": k, "recipe": r} for k, r in enumerate(recipes, 1)]
        preds = {}
        if score is not None and score.model is not None:
            try:
                preds = insights.round_predictions(opt, {insights.SCORE: score}, rows)
            except Exception:
                preds = {}
        top = float(opt.utility_ceiling())
        out = []
        for row in rows:
            p = preds.get(row["formulation"], {}).get(insights.SCORE)
            text = "—" if p is None else capped_range(insights.widened(score, p, factor), top, 0)
            role = opt.suggestion_role(row["recipe"])
            try:
                apart = apart_text(opt, row["recipe"], role, binding)
            except Exception:
                apart = wording.apart_line(role, [])
            out.append(PreviewRow(row["formulation"], role, text,
                                  opt.vs_best_text(row["recipe"], n=2, max_changes=2),
                                  apart, None if p is None else float(p.mean), row["recipe"]))
    if out is not None:
        _PREVIEW_CACHE[key] = out
        while len(_PREVIEW_CACHE) > _PREVIEW_KEEP:
            _PREVIEW_CACHE.popitem(last=False)
    return out


# Round-2 fixes (P2 #8): the preview kept per project state for the life of
# the app, not per session and one at a time, so a revisit (another tab,
# another project and back) draws it at once — no spinner, no jump.
_PREVIEW_CACHE = collections.OrderedDict()
_PREVIEW_KEEP = 16
PREVIEW_HEIGHT = 300        # px kept for the preview while it is worked out


def preview_key(opt, binding=None):
    """What the preview depends on: the project, its setup and data, the
    next Formulation number, the Round size and the binding limit named."""
    return (opt.project_name, insights.setup_key(opt), opt.next_formulation_no,
            _preview_count(opt), None if binding is None else (binding.kind, binding.name))


def preview_cached(opt, binding=None):
    """Whether the preview is already worked out for this project state."""
    return preview_key(opt, binding) in _PREVIEW_CACHE


def learning_round(opt, rows):
    """Every suggestion's predicted Overall score at or below Best so far
    (to the tenth): a learning round (persona fixes)."""
    best = opt.best_index()
    means = [r.mean for r in rows or () if getattr(r, "mean", None) is not None]
    if best is None or not means or len(means) != len(rows):
        return False
    return all(round(m, 1) <= round(float(opt.Y_history[best]), 1) for m in means)


def widen_beyond(opt, step, rows):
    """Next advises widening a Lowest or Highest of X while a suggestion
    moves X away from that limit (persona fixes): the one line saying the
    suggestions stay inside the limits."""
    b = getattr(step, "binding", None) if step is not None and step.kind == "close" else None
    best = opt._best_recipe_index()
    if b is None or b.kind not in ("lowest", "highest") or best is None:
        return False
    there = insights.start_recipe(opt, best)
    for r in rows or ():
        if getattr(r, "recipe", None) is None:
            continue
        here = insights.at_total(opt, r.recipe)
        d = float(here.get(b.name, 0.0)) - float(there.get(b.name, 0.0))
        if (b.kind == "lowest" and d > 1e-9) or (b.kind == "highest" and d < -1e-9):
            return True
    return False


def _next_preview(opt, places, computed, facts=None):
    """What next?'s preview of the next suggestions (no Round open): drawn
    last, under a spinner in its own place, once per setup."""
    if opt.pending_batch or computed is None or computed.get("failed"):
        return
    ready, _ = readiness(opt)
    if not ready or opt.complete_count() < insights.MIN_ROWS:
        return
    step = getattr(facts, "next", None)
    binding = (getattr(step, "binding", None)
               if step is not None and step.kind == "close" else None)
    with places.preview.container():
        if ceiling_stop(opt, facts):
            # Round-2 fixes (P2 #8): under Stop, the preview is a closed
            # fold, worked out only once opened.
            with st.expander(_say(opt, wording.LEARNED_PREVIEW_TITLE), expanded=False,
                             key=PREVIEW_FOLD_KEY, on_change="rerun") as fold:
                if getattr(fold, "open", True):
                    _preview_body(opt, computed, facts, binding, title=False)
            return
        _preview_body(opt, computed, facts, binding)


PREVIEW_FOLD_KEY = "insights_preview_fold"


def _preview_body(opt, computed, facts, binding, title=True):
    """The preview's title, table and lines; while it is worked out, its
    place is held at PREVIEW_HEIGHT so the page below does not jump."""
    if preview_cached(opt, binding):
        rows = preview_rows(opt, computed, binding)
    else:
        hold = st.empty()
        with hold.container(height=PREVIEW_HEIGHT, border=False):
            with st.spinner(_say(opt, wording.LEARNED_PREVIEW_SPINNER)):
                rows = preview_rows(opt, computed, binding)
        hold.empty()
    if not rows:
        st.caption(_say(opt, wording.LEARNED_PREVIEW_FAILED))
        return
    best = opt.best_formulation_no()
    from food_bo import CLOSE_TO_THE_BEST, FAR_FROM_TESTED
    help_ = (wording.preview_help(len(rows), best) + "\n\n" + wording.roles_help(
        f"{100 * CLOSE_TO_THE_BEST:.0f}", f"{100 * FAR_FROM_TESTED:.0f}"))
    if title:
        st.markdown(f"**{_say(opt, wording.LEARNED_PREVIEW_TITLE)}**",
                    help=_say(opt, help_))
    else:
        st.caption(_say(opt, wording.PREVIEW_FOLD_CAPTION), help=_say(opt, help_))
    # Persona fixes: a learning round said as one, above the table.
    if learning_round(opt, rows):
        # Specific wording: the highest prediction against the best.
        means = [r.mean for r in rows if getattr(r, "mean", None) is not None]
        best_i = opt.best_index()
        st.caption(_say(opt, wording.learning_round_counted(
            f"{max(means):.1f}", f"{float(opt.Y_history[best_i]):.1f}")
            if means and best_i is not None else wording.LEARNING_ROUND_WHY))
    cols = [_say(opt, c) for c in (wording.PREVIEW_COLUMNS[0], wording.PREVIEW_APART_COLUMN,
                                   *wording.PREVIEW_COLUMNS[2:])]
    frame = pd.DataFrame([dict(zip(cols, (r[0], _say(opt, getattr(r, "apart", "") or r[1]),
                                          r[2], r[3]))) for r in rows])
    # Persona fixes: a static table that wraps its words, so the line
    # that sets each apart and its changes read whole at 820 px.
    st.table(frame.set_index(cols[0]))
    if widen_beyond(opt, getattr(facts, "next", None), rows) and not ceiling_stop(opt, facts):
        # (Review critique: no widening advice at the top score.)
        st.markdown(_say(opt, wording.WIDEN_BEYOND))
    st.caption(_say(opt, wording.LEARNED_PREVIEW_CAPTION))


def _trust_section(opt, slot, computed, facts):
    """"Can we trust the predictions?" (review story): forecast against
    actual over time — each checked formulation's Round-ahead 95 % range
    for its Overall score, and the score measured; the answer counts how
    many landed in range. The scatter's numbers stay in the "?"."""
    checks = [c for c in computed["checks"] or () if c.name == insights.SCORE]
    inside, total, above, below = trust_counts(checks)
    text, help_ = None, None
    if total and facts is not None and facts.prediction_row in ("reliable", "too_few"):
        if facts.prediction_row == "reliable":
            text = wording.story_trust_answer(inside, total)
        elif total == facts.checked:
            text = wording.story_trust_so_far(inside, total, insights.SKILL_MIN)
        card = card_for(opt, facts, "predictions")
        help_ = "\n\n".join(t for t in (
            wording.story_trust_help(inside, total, wording.story_misses(above, below)),
            card.help if card is not None else None) if t)
    _answer(opt, slot, facts, "predictions", text=text, help=help_)
    with slot.body:
        try:
            chart = forecast_chart(opt, checks) if checks else None
        except Exception:   # one chart failing must not take the tab down
            chart = None
        if chart is None:
            st.caption(_say(opt, wording.Q_WAIT if not checks else wording.CHART_FAILED))
            return
        show_chart(chart)
        st.caption(_say(opt, wording.STORY_TRUST_CAPTION),
                   help=_say(opt, wording.story_trust_help(
                       inside, total, wording.story_misses(above, below))))


PROGRESS_ROUND_PLACE = "insights_progress_round"   # the section's Replay panel and captions

# Layout study (owner, 2026-10-09: "Doesn't the layout require us to scroll
# up and down while using the slider, making it confusing?"): one Replay
# panel holds everything a Round step changes — the Round control at its
# top (◀ Round 4 of 10 ▶, Replay Rounds, Back to latest), the score chart
# over What changed on the left, the sensory profile on the right — and fits
# a 1152 × 720 viewport (1440 × 900 at 125 %), so a step never needs a
# scroll. An earlier Round is framed and tinted, labelled "Earlier Round",
# with Back to latest; the section's answer line above and What next? below
# stay about today. Leaving the tab, or another project, is today again.
REPLAY_STATE = "insights_replay"        # {"project", "round"}: an earlier Round on view
# The project while Play steps. Final demo (B4): its own key, not the Play
# button's ("insights_replay_play"): Streamlit drops an undrawn widget's
# state after a click's run, so a second click (Stop) while Play stepped
# ended the run a Round later.
REPLAY_PLAY = "_insights_replay_playing"
REPLAY_STEP_SECS = 2.0                  # Play: one Round every 2 s (final demo B2), then stops
REPLAY_T = "_insights_replay_t"         # {"play", "stop", "end"}: when each last happened
STOP_GUARD_SECS = 0.8                   # a Stop this soon after Play is ignored (B4)
PLAY_GUARD_SECS = 1.0                   # a Play this soon after a Stop or a run's end, too
REPLAY_SLICE_SECS = 0.05                # the wait's slices: a click is seen within this
REPLAY_PANEL = "insights_replay_panel"  # + "_now" (the latest) or "_past" (framed, tinted)
REPLAY_BAR = "insights_replay_bar"
PANEL_CHART_H = 150                     # px: the score chart's plot in the panel
PANEL_PLOT_W = 320                      # px: its narrowest plot (1152 px), for label clearances
PANEL_SPIDER = 340                      # px: the profile's square beside the chart
PANEL_SPLIT = [0.44, 0.56]              # the columns: chart over What changed | profile

REPLAY_STYLE = """<style>
.st-key-insights_replay_panel_now, .st-key-insights_replay_panel_past { padding: 0.6rem 0.9rem !important; }
.st-key-insights_replay_panel_past { background: rgba(120, 128, 140, 0.09);
  border: 1px solid rgba(90, 98, 110, 0.55) !important;
  box-shadow: inset 0 0 0 1px rgba(90, 98, 110, 0.55); }
.st-key-insights_replay_bar button { min-height: 2rem; padding: 0.15rem 0.7rem; }
.st-key-insights_replay_bar p { margin: 0; }
[class*="st-key-insights_replay_panel"] [data-testid="stElementToolbar"] { display: none; }
[class*="st-key-insights_replay_panel"] [data-testid="stMarkdownContainer"] p { font-size: 0.94rem; }
.st-key-insights_changes table { font-size: 0.82rem; }
.st-key-insights_changes th, .st-key-insights_changes td {
  padding: 0.08rem 0.4rem !important; line-height: 1.35; }
.st-key-insights_changes td { white-space: nowrap; }
.st-key-insights_changes p { margin-bottom: 0.2rem; }
</style>"""


def _replay_project(opt):
    return getattr(opt, "project_name", None)


def shown_round(opt, rounds):
    """The Round on view: an earlier one chosen on this project, else the
    latest (another project, or none chosen, is always today's)."""
    s = st.session_state.get(REPLAY_STATE)
    if (rounds and isinstance(s, dict) and s.get("project") == _replay_project(opt)
            and s.get("round") in rounds[:-1]):
        return s["round"]
    return rounds[-1] if rounds else None


def show_round(opt, rounds, r):
    """Put Round `r` on view (the latest, or None: back to today)."""
    if r is None or not rounds or r not in rounds[:-1]:
        st.session_state.pop(REPLAY_STATE, None)
    else:
        st.session_state[REPLAY_STATE] = {"project": _replay_project(opt), "round": r}


def leave_review():
    """The Review tab closed: back to the latest Round, any Replay stopped
    (app.py, on every run the Review is not open)."""
    for k in (REPLAY_STATE, REPLAY_PLAY):
        st.session_state.pop(k, None)


def replay_next(rounds, r):
    """(the next Round, still playing): one step on; stops on the latest."""
    i = rounds.index(r) if r in rounds else len(rounds) - 1
    nxt = rounds[min(i + 1, len(rounds) - 1)]
    return nxt, nxt != rounds[-1]


def replaying(opt=None):
    """Whether a Replay is stepping (on this project)."""
    who = st.session_state.get(REPLAY_PLAY)
    return who is not None and (opt is None or who == _replay_project(opt))


def _on_step(opt, rounds, by):
    st.session_state.pop(REPLAY_PLAY, None)
    r = shown_round(opt, rounds)
    i = max(0, min(len(rounds) - 1, rounds.index(r) + by))
    show_round(opt, rounds, rounds[i])


def _on_latest(opt, rounds):
    st.session_state.pop(REPLAY_PLAY, None)
    show_round(opt, rounds, None)


def _replay_times():
    t = st.session_state.get(REPLAY_T)
    if not isinstance(t, dict):
        t = {}
        st.session_state[REPLAY_T] = t
    return t


def _since(t, *names):
    """Seconds since the latest of `names` in `t` (inf when none)."""
    last = [t[n] for n in names if n in t]
    return time.monotonic() - max(last) if last else math.inf


def _on_play(opt, rounds):
    """Play, from the first Round. Final demo (B4): ignored within
    PLAY_GUARD_SECS of a Stop or of a run's end (a second click of Stop
    lands on Play)."""
    t = _replay_times()
    if _since(t, "stop", "end") < PLAY_GUARD_SECS:
        return
    show_round(opt, rounds, rounds[0])
    st.session_state[REPLAY_PLAY] = _replay_project(opt)
    t["play"] = t["step"] = time.monotonic()


def _on_stop(opt=None, rounds=None, r=None):
    """Stop: hold on the Round that was on screen when it was clicked
    (`r`, the Round its button was drawn with), even if a step began.
    Final demo (B4): ignored within STOP_GUARD_SECS of Play (a slow
    double-click on Play lands its second click on Stop)."""
    t = _replay_times()
    if _since(t, "play") < STOP_GUARD_SECS and replaying(opt):
        return
    st.session_state.pop(REPLAY_PLAY, None)
    t["stop"] = time.monotonic()
    if opt is not None and rounds and r in rounds:
        show_round(opt, rounds, r)


def _fragment_rerun():
    """True in a fragment's own rerun (st.rerun(scope="fragment") is
    allowed only there), False in a full run."""
    try:
        from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx
        ctx = get_script_run_ctx(suppress_warning=True)
        return bool(ctx is not None and ctx.fragment_ids_this_run)
    except Exception:
        return False


def _click_pending():
    """Whether a click in this fragment is waiting for the run to end (a
    rerun request naming this fragment, or the whole page). Streamlit
    internals, guarded: False when they cannot be read (Stop still lands on
    the Round shown, _on_stop's own argument, only a step later)."""
    try:
        from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx
        ctx = get_script_run_ctx(suppress_warning=True)
        req = getattr(ctx, "script_requests", None)
        if req is None or getattr(getattr(req, "_state", None), "name", None) != "RERUN":
            return False
        queue = list(getattr(getattr(req, "_rerun_data", None), "fragment_id_queue", None) or [])
        return not queue or ctx.current_fragment_id in queue
    except Exception:
        return False


def _replay_tick(opt, rounds, r):
    """After a Replay step is drawn: wait, then the next Round — a rerun of
    this fragment only, the step's values exactly (nothing drawn between
    Rounds). A full run (a click elsewhere) stops it where it is."""
    if not replaying(opt):
        return
    if not _fragment_rerun():
        st.session_state.pop(REPLAY_PLAY, None)
        return
    # Wait in short slices: a click in this panel (Stop, ◀ ▶, Back to
    # latest) waits for this run to end — Streamlit never lets a fragment's
    # own click cut its run short — so the wait ends early and the step is
    # not taken; the click's callback then decides the Round.
    # Final demo (B2): the pace is REPLAY_STEP_SECS from the step before
    # (its rerun asked for), the drawing between included, so a Round
    # shows about 2 s, not 2 s plus the drawing.
    t = _replay_times()
    since = time.monotonic() - t["step"] if "step" in t else 0.0
    step = max(0.0, REPLAY_STEP_SECS - since) if 0.0 <= since < REPLAY_STEP_SECS else \
        REPLAY_STEP_SECS
    waited = 0.0
    while waited < step - 1e-9:
        if _click_pending():
            return
        time.sleep(min(REPLAY_SLICE_SECS, step - waited))
        waited += REPLAY_SLICE_SECS
    if _click_pending():
        return
    nxt, still = replay_next(rounds, r)
    show_round(opt, rounds, nxt)
    if not still:
        st.session_state.pop(REPLAY_PLAY, None)
        t["end"] = time.monotonic()
        t.pop("step", None)
    else:
        t["step"] = time.monotonic()
    st.rerun(scope="fragment")


def _hold(n):
    """`n` empty places: a Replay step draws as many elements as any other
    Round, so nothing of the Round before lingers while the step waits
    (Streamlit clears an element not drawn again only when the run ends)."""
    for _ in range(n):
        st.empty()


def _replay_bar(opt, rounds, r):
    """◀ Round 4 of 10 ▶ · Replay Rounds (Stop while it plays) · then, at
    the right, Latest, or Earlier Round and Back to latest."""
    last, i = rounds[-1], rounds.index(r)
    playing = replaying(opt)
    with st.container(horizontal=True, vertical_alignment="center", gap="small",
                      key=REPLAY_BAR):
        st.button(wording.REPLAY_PREV, key="insights_replay_prev",
                  disabled=i == 0 or playing, on_click=_on_step, args=(opt, rounds, -1),
                  help=_say(opt, wording.replay_to(rounds[max(0, i - 1)])))
        st.markdown(_say(opt, wording.replay_label(r, last)), width="content",
                    help=_say(opt, wording.REPLAY_HELP))
        st.button(wording.REPLAY_NEXT, key="insights_replay_next",
                  disabled=r == last or playing, on_click=_on_step, args=(opt, rounds, 1),
                  help=_say(opt, wording.replay_to(rounds[min(len(rounds) - 1, i + 1)])))
        if playing:
            st.button(wording.REPLAY_STOP, key="insights_replay_stop", icon=":material/stop:",
                      on_click=_on_stop, args=(opt, rounds, r))
        else:
            st.button(_say(opt, wording.REPLAY_PLAY), key="insights_replay_play",
                      icon=":material/play_arrow:", on_click=_on_play, args=(opt, rounds),
                      help=_say(opt, wording.replay_play_help(rounds[0], last)))
        st.space("stretch")
        if r != last:
            st.markdown(f":orange-badge[{_say(opt, wording.REPLAY_EARLIER)}]", width="content")
            st.button(wording.REPLAY_BACK, key="insights_replay_latest", icon=":material/undo:",
                      on_click=_on_latest, args=(opt, rounds))
        else:
            st.markdown(f":gray-badge[{wording.REPLAY_LATEST}]", width="content")
            _hold(1)


def _progress_body(opt, slot):
    """"Are we improving?" (review story; slim Review; layout study
    2026-10-09): the Replay panel — its Round control, the score chart (the
    Round shaded, a marker on Best so far there), What changed and the
    sensory profile, all at one Round and all in one viewport — then the
    parts' captions, which never change with the Round. One fragment: a
    step reruns only this section. The section's answer line (drawn with
    the cards, above) stays about the latest Round."""
    if not _fragment_rerun():
        st.session_state.pop(REPLAY_PLAY, None)    # a full run stops a Replay
    with slot.body:
        st.html(REPLAY_STYLE)
        with st.container(key=PROGRESS_ROUND_PLACE, gap="small"):
            _progress_fragment(opt)


def _section_rounds(opt):
    try:
        rounds, _at = round_bests(opt)
    except Exception:
        rounds = []
    return rounds


@st.fragment
def _progress_fragment(opt):
    """The Replay panel at the Round on view (shown_round: the latest
    unless an earlier one is chosen), then the captions. Only this reruns
    on a step; each Round's best is found once. One scored Round: the same
    panel without its control."""
    rounds = _section_rounds(opt)
    r = shown_round(opt, rounds)
    control = len(rounds) > 1
    past = control and r != rounds[-1]
    with st.container(border=True, gap="small",
                      key=f"{REPLAY_PANEL}_{'past' if past else 'now'}"):
        if control:
            _replay_bar(opt, rounds, r)
        left, right = st.columns(PANEL_SPLIT, gap="medium")
        with left:
            _improving_part(opt, r if control else None)
            _changes_block(opt, r if control else None)
        with right:
            _profile_part(opt, r)
    _panel_captions(opt)
    if control:
        _replay_tick(opt, rounds, r)


def _panel_captions(opt):
    """The three parts' captions, one row under the panel (they never
    change with the Round)."""
    avgs = round_averages(opt)
    first, last = (avgs[0], avgs[-1]) if avgs else (None, None)
    with st.container(horizontal=True, gap="medium"):
        st.caption(_say(opt, wording.STORY_IMPROVING_CAPTION), width="content",
                   help=_say(opt, wording.story_improving_help(
                       one_decimal(first[3]) if first else None,
                       one_decimal(last[3]) if last else None,
                       first[0] if first else None, last[0] if last else None,
                       assumed=getattr(insights.noise_yardstick(opt), "assumed", True),
                       need_text=_top_zone_need(opt), sd_text=noise_sd_text(opt))))
        try:
            spider = spider_data(opt)
        except Exception:
            spider = None
        if spider is not None:
            s_i, s_no, s_round, _vals = spider.start
            yard = insights.noise_yardstick(opt)
            st.caption(_say(opt, wording.SPIDER_CAPTION), width="content",
                       help=_say(opt, wording.brief(wording.spider_help(
                           s_no, s_round, repeats=not getattr(yard, "assumed", True)))))
        ch = what_changed(opt, None)
        if ch is not None and not ch.same:
            st.caption(_say(opt, wording.CHANGES_CAPTION), width="content",
                       help=_say(opt, wording.brief(wording.changes_help(
                           ch.start_no, ch.best_no, ch.start_round, fill=ch.fill,
                           fixed=ch.fixed, settings=any(x.setting for x in ch.rows),
                           near_zero=any(x.near_zero for x in ch.rows)))))
    try:
        recipe = score_recipe_text(opt)
    except Exception:
        recipe = None
    # Final demo (A5): what the score is, the recipe in its "?".
    try:
        built = score_built_text(opt)
    except Exception:
        built = None
    st.caption(_say(opt, wording.SCORE_IS_MATCH),
               help="\n\n".join(t for t in (recipe, built) if t) or None)


def _improving_part(opt, r):
    """The score chart in the Replay panel: slim (PANEL_CHART_H), Round `r`
    shaded with a marker on Best so far there (None: no highlight). Its
    caption sits under the panel (_panel_captions)."""
    try:
        spec = improving_spec(opt, r, height=PANEL_CHART_H)
    except Exception:
        spec = None
    if spec is None:
        st.caption(_say(opt, wording.CHART_FAILED))
        return
    show_spec(spec, clear=True)


def _profile_part(opt, r):
    """Sensory profile against target at Round `r` (the spider from
    SPIDER_MIN measurements, else the Start vs best dot chart with its
    caption). The score recipe sits under the panel."""
    try:
        spider = spider_data(opt)
    except Exception:
        spider = None
    if spider is not None:
        _spider_block(opt, spider, r if r in spider.bests else spider.rounds[-1])
        return
    try:
        _rounds, at = round_bests(opt)
        best = at.get(r) if r is not None else None
        chart = target_chart(opt, best=best)
    except Exception:
        chart, best = None, None
    if chart is None:
        return
    st.markdown(f"**{_say(opt, wording.STORY_TARGET_TITLE)}**")
    answer = target_answer(opt, rows=target_rows(opt, best=best))
    if answer:
        st.markdown(answer)
    show_chart(chart)
    s = story_start(opt)
    b = opt.best_index() if best is None else best
    st.caption(_say(opt, wording.STORY_TARGET_CAPTION),
               help=_say(opt, wording.story_target_help(
                   int(opt.formulation_ids[s[0]]), int(opt.formulation_ids[b]),
                   s[1])) + "\n\n" + score_built_text(opt))


def _drives_section(opt, slot, computed, facts):
    """"What drives each result?": the profiler (Overall score, each
    measurement, Expected gain), through Suggestion 1 when there is one.
    Review learned 2b (coordinator: 7 ingredients, at most 30 batches and
    no Repeats are too thin for any effect claim): the headline is a
    reading guide, never a finding; its "?" says why no effect is named.
    driver_finding and learned.effects stay, tested, unused here."""
    if not getattr(slot, "open", True):
        return      # worked out only while its fold is open
    with slot.body:
        try:
            got = _profiler(opt, computed)
        except Exception:
            st.caption(_say(opt, wording.CHART_FAILED))
            got = None
    profile = got[0] if got is not None else None
    kind = ("ingredients" if not getattr(profile, "columns", None)
            else _more_kind(opt, [c.name for c in profile.columns]))
    line = wording.profiler_guide(kind)
    limit = limiting_name(profile)
    if limit:
        # Review critique (fix 4): the fold's answer says what bounds the lines.
        line = wording.answer_line(wording.drives_limited(limit), "")
    ev = wording.PROFILER_NO_CLAIM
    if got is not None:
        cfg = getattr(opt, "bo_config", None) or {}
        ev += "\n\n" + wording.evidence([wording.ev_profiler(
            opt.complete_count(), profile.ref_label, profile.ref_no,
            cfg.get("acquisition", "qlognei"))])
    with slot.answer.container():
        st.markdown(_say(opt, line), help=_say(opt, wording.brief(ev)))


def limiting_name(profile):
    """The limit on several rows that stops most columns' lines (their
    paths' end reasons), its name as Set up gives it, brackets dropped:
    'Batch total'; None when no one limit stops more than half of them."""
    cols = getattr(profile, "columns", None) or []
    if not cols:
        return None
    counts = {}
    for c in cols:
        for why in {c.path.low_end.reason, c.path.high_end.reason} - {None}:
            name = re.sub(r"\s*\(.*?\)", "", str(why).partition(":")[0]).strip()
            if name and ":" in str(why):
                counts[name] = counts.get(name, 0) + 1
    if not counts:
        return None
    name, n = max(counts.items(), key=lambda kv: kv[1])
    return name if n * 2 > len(cols) else None


def _gap_kind(rows):
    """"ingredient", "setting" or "mixed": what the gap rows are."""
    settings = {bool(s.setting) for s in rows}
    return "mixed" if len(settings) > 1 else "setting" if settings == {True} else "ingredient"


def _row_decimals(strip, gaps):
    """One decimal rule for a row's gaps (a process setting: fmt_setting's)."""
    return None if strip.setting else _shared_decimals(
        [v for g in gaps for v in (g.low, g.high)])


def _span_text(opt, strip, gap, d, unit=True):
    """'0.6–1.0 g': a gap's two ends, the unit after the second."""
    return (f"{_strip_number(opt, strip, gap.low, False, d)}–"
            f"{_strip_number(opt, strip, gap.high, unit, d)}")


def _found_gaps(strips):
    """(the gap rows, [(strip, its gaps)] for the rows with a gap)."""
    rows = insights.gap_rows(strips)
    found = [(s, insights._stretches(s)) for s in rows]
    return rows, [(s, gs) for s, gs in found if gs]


_ANSWER_WORDS = 14            # a section answer's headline, at most


def tried_answer(opt, strips):
    """Q4's answer: '**30 tested.** Gaps in 6 of 7 ingredients; largest:
    Salt 0.6–1.0 g.' (owner, 2026-10-05: the count, so the
    other rows are never read as covered)."""
    rows, found = _found_gaps(strips)
    kind = _gap_kind(rows)
    word = wording.q4_word(opt.complete_count())
    if not found:
        return wording.answer_line(word, wording.tried_no_gap(kind) if rows else "")
    strip, gs = max(found, key=lambda f: max(g.size for g in f[1]))
    gap, d = max(gs, key=lambda g: g.size), _row_decimals(strip, gs)
    largest = wording.gap_span(strip.name, _strip_number(opt, strip, gap.low, False, d),
                               _strip_number(opt, strip, gap.high, True, d))
    line = wording.answer_line(word, wording.tried_gaps(len(found), len(rows), kind, largest))
    if len(line.replace("*", "").split()) > _ANSWER_WORDS:
        # A long name: the count alone; the "?" lists every gap.
        line = wording.answer_line(word, wording.tried_gaps(len(found), len(rows), kind))
    return line


def tried_gap_lines(opt, strips):
    """The answer's "?" first: every gap, one line per row, low to high
    ('- Oil: 2.0–4.0, 6.5–9.0 g'), or that there is none; then
    the one-row-at-a-time caveat. Over the "?"'s word budget, each row's
    largest and '+N more'; then as many rows as fit and '+N more'."""
    rows, found = _found_gaps(strips)
    kind = _gap_kind(rows)
    caveat = wording.tried_caveat(kind)
    if not rows:
        return []
    if not found:
        return [wording.strips_no_gap(settings_only=kind == "setting",
                                      has_settings=kind != "ingredient"), caveat]
    words = lambda lines: sum(len(x.split()) for x in lines)  # noqa: E731
    room = wording.TOOLTIP_WORDS - words([caveat])
    full, short = [], []
    for s, gs in found:
        d = _row_decimals(s, gs)
        full.append(wording.gap_row_line(s.name, ", ".join(
            _span_text(opt, s, g, d, k == len(gs) - 1) for k, g in enumerate(gs))))
        top = max(gs, key=lambda g: g.size)
        short.append(wording.gap_row_line(s.name, _span_text(opt, s, top, d), len(gs) - 1))
    lines = full if words(full) <= room else short
    if words(lines) > room:
        kept = []
        for k, line in enumerate(lines):
            left = len(lines) - k - 1
            tail = [wording.gap_rows_more(left, kind)] if left else []
            if words(kept + [line] + tail) > room:
                kept.append(wording.gap_rows_more(len(lines) - k, kind))
                break
            kept.append(line)
        lines = kept
    return ["\n".join(lines), caveat]


def _tried_section(opt, slot):
    """Q4, "What have we tried?": the parallel coordinates, model-free; the
    answer from the gap lines, every gap and limit-hit line in its "?".
    Slim Review: a closed fold, worked out only while open."""
    if not getattr(slot, "open", True):
        return
    try:
        pool = bo_views.feasible_pool(opt)
        strips = insights.tested_strips(opt, pool)
        answer = tried_answer(opt, strips)
        options = [wording.option_never_tested(g.name, g.option)
                   for g in insights.strip_gaps(opt, strips) if g.kind == "option"]
        lines = tried_gap_lines(opt, strips) + options + limit_hit_lines(opt)
        # The tried gaps: every gap the answer counts is shaded.
        gaps = insights.all_stretches(strips)
    except Exception:
        answer, lines, gaps = None, [], []
    if answer is not None and opt.complete_count():
        ev = wording.evidence([wording.ev_tried(opt.complete_count())])
        with slot.answer.container():
            st.markdown(_say(opt, answer),
                        help=_say(opt, wording.brief("\n\n".join(
                            [*lines, ev, tooltip_for(opt, "gaps")]))))
    with slot.body:
        settings_only = not opt.has_ingredients()
        # Review critique (fix 5): the search path only ("Show as lines"
        # and its parallel coordinates cut).
        axes, lines = insights.tried_axes(opt), insights.tried_lines(opt)
        try:
            chart = search_path_chart(opt, axes, lines, gaps)
        except Exception:
            chart = None
        if chart is None:
            st.caption(_say(opt, wording.TRIED_NOTHING))
            return
        show_chart(chart, width="content")
        st.caption(_say(opt, wording.STORY_PATH_CAPTION),
                   help=_say(opt, wording.brief(wording.story_path_help(settings_only))))


# ------------------------------------------------------------------ #
#  Review story (prototype, 2026-10-02): the Review read in the order a
#  food scientist asks — Are we improving? Can we trust the predictions?
#  What drives each result? What have we tried? Builders only here (no
#  Streamlit); the sections above draw them. Every number is read from the
#  project: the start is the first scored Round's best, Best so far is
#  best_index's, the forecasts are the Round-ahead checks.
# ------------------------------------------------------------------ #

STORY_Q_HEIGHT = 290         # the forecast chart's plot height (less 30)
# Review story 2: the improving pair fits the first screen at 1280×800,
# under the strip; Closer to target is drawn as tall, so the two end level.
# Review readable: the two are stacked, each the column's full width (side
# by side at 1280 px each was under 420 px and its labels collided).
IMPROVING_HEIGHT = 240
_IMPROVING_LANES = 48        # px above the data for the Best and Start labels
TARGET_ROW_H = 68            # px per Closer to target row (round 2: values never stack)
PATH_COLUMNS = 4             # search path panels per row
_PATH_PANEL_W = 160          # review readable: was 150
_PATH_ROW_H = 22             # px per Round in a search path panel (was 16: rows touched)
_PATH_SPACING = 24


def story_start(opt):
    """(position, Round) of the first scored Round's best formulation —
    the start the Review measures progress from — or None with no Round
    scored. Only Best-so-far candidates compete, as for Best so far."""
    pool = opt.best_candidates()
    rounds = [opt.batch_history[i] if i < len(opt.batch_history) else None for i in pool]
    named = [r for r in rounds if r is not None]
    if not named:
        return None
    first = min(named)
    mine = [i for i, r in zip(pool, rounds) if r == first]
    best = max(mine, key=lambda i: opt.Y_history[i])
    return best, first


def story_numbers(opt):
    """(start score, start Round, best score, best Round, best number) or
    None: the Progress card's and the improving chart's two named points."""
    s = story_start(opt)
    b = opt.best_index()
    if s is None or b is None:
        return None
    i, r0 = s
    rb = opt.batch_history[b] if b < len(opt.batch_history) else None
    return (float(opt.Y_history[i]), r0, float(opt.Y_history[b]), rb,
            int(opt.formulation_ids[b]))


def one_decimal(x):
    return f"{float(x):.1f}"


def round_averages(opt):
    """[(Round, first number, last number, mean Overall score, n)] per
    Round, over its formulations with all results (Best-so-far
    candidates), in Round order."""
    pool = set(opt.best_candidates())
    groups = {}
    for i, r in enumerate(opt.batch_history[:len(opt.Y_history)]):
        if r is None or i not in pool:
            continue
        no = int(opt.formulation_ids[i])
        g = groups.setdefault(int(r), [no, no, []])
        g[0], g[1] = min(g[0], no), max(g[1], no)
        g[2].append(float(opt.Y_history[i]))
    return [(r, a, b, sum(v) / len(v), len(v)) for r, (a, b, v) in sorted(groups.items())]


def _x_domain(opt):
    numbers = [int(n) for n in opt.formulation_ids]
    return [min(numbers) - 0.7, max(numbers) + 0.7] if numbers else [0, 1]


def _round_layer(opt, frame_x, numbers, scale):
    """Light rules between Rounds and 'R3' labels along the top, on the
    Formulation field (one scale), or None with under two Rounds."""
    marks = round_marks(opt, numbers)
    if not marks:
        return None
    rules, labels = marks
    light, _ = _TRIED_GREYS[theme()]
    mids = [m for m, _ in labels]
    expr = " : ".join(f"abs(datum.value - {m}) < 0.01 ? '{t}'" for m, t in labels) + " : ''"
    top = alt.X(f"{frame_x}:Q", title=None, scale=scale,
                axis=alt.Axis(orient="top", values=mids, labelExpr=expr, ticks=False,
                              domain=False, grid=False, labelFontSize=FONT_SMALL,
                              labelPadding=3, labelColor=muted()))
    return (alt.Chart(pd.DataFrame({frame_x: rules}))
            .mark_rule(color=light, strokeWidth=1, tooltip=False).encode(x=top))


def _page():
    """The page colour behind a chart, for a legend drawn over the plot."""
    return "#0e1117" if theme() == "dark" else "#ffffff"


def _halo_text(data, x, y, text, color, **kw):
    """A text mark with a page-coloured halo under it, so a label over a
    line or a dot still reads."""
    page = "#0e1117" if theme() == "dark" else "#ffffff"
    under = alt.Chart(data).mark_text(color=page, stroke=page, strokeWidth=4, tooltip=False,
                                      **kw).encode(x=x, y=y, text=text)
    over = alt.Chart(data).mark_text(color=color, tooltip=False, **kw).encode(
        x=x, y=y, text=text)
    return [under, over]


def best_steps(opt):
    """Review critique (fix 7): Best so far as a step that moves only at
    Round boundaries: [(x from, x to, best through that Round, Round)], each
    Round spanning its Formulations ± 0.5, from round_averages' Rounds."""
    pool = set(opt.best_candidates())
    out, best = [], None
    for r, a, b, _m, _n in round_averages(opt):
        mine = [float(opt.Y_history[i]) for i in pool
                if i < len(opt.batch_history) and opt.batch_history[i] == r]
        if mine:
            best = max(mine) if best is None else max(best, max(mine))
        if best is not None:
            out.append((a - 0.5, b + 0.5, best, r))
    return out


def _top_zone_need(opt):
    """The top zone's depth as text ('2.3'), or None when the chart draws
    no zone (the top score far from the data)."""
    try:
        need = tie_need(insights.noise_yardstick(opt))
    except Exception:
        return None
    return None if not need else f"{need:.1f}"


_AVG_GAP = 4                 # px between a Round's average bar and its label
_AVG_CLEAR = 2               # px a label keeps clear of a dot, a ring or the best line
_AVG_MAX_SHIFT = 16          # px a label may move past _AVG_GAP to get clear
_DOT_R = 4                   # px: a formulation's dot (size 30) and its stroke
_RING_R = 8                  # px: the Start and Best rings (size 150)
_MIN_PLOT_W = 560            # px: the narrowest plot the x clearances assume


def average_label_y(avgs, dots, steps, rings, y_domain, x_domain,
                    height=IMPROVING_HEIGHT, width=_MIN_PLOT_W, font=FONT_SMALL):
    """Where each Round's average label sits (presentation zoom check): the
    centre of its text, in score units, under its bar unless a mark is
    there, then over it, then whichever side needs the smaller nudge — so
    no label lands on a dot, a ring or the Best so far line.

    avgs: round_averages rows; dots: (x, score) per formulation; steps:
    best_steps rows; rings: (x, score) of the Start and Best rings."""
    y0, y1 = y_domain
    x0, x1 = x_domain
    ypx = height / (y1 - y0) if y1 > y0 else 1.0
    xpx = width / (x1 - x0) if x1 > x0 else 1.0
    out = []
    for _r, a, b, m, _n in avgs:
        mid, text = (a + b) / 2, one_decimal(m)
        half_w = 0.31 * font * len(text)
        # Every mark near the label's column, as (x px from the centre,
        # y px above the bar, radius): dots and rings round, the best line
        # a run of points along its step.
        marks = [((x - mid) * xpx, (v - m) * ypx, _DOT_R) for x, v in dots]
        marks += [((x - mid) * xpx, (v - m) * ypx, _RING_R) for x, v in rings]
        for sa, sb, v, _sr in steps:
            lo, hi = max(sa, mid - half_w / xpx - 1), min(sb, mid + half_w / xpx + 1)
            k = lo
            while k <= hi:
                marks.append(((k - mid) * xpx, (v - m) * ypx, 1.5))
                k += 2 / xpx
        marks = [(dx, dy, r) for dx, dy, r in marks if abs(dx) < half_w + r + _AVG_CLEAR]

        def clear(sign, gap):
            """Whether the label's box, `gap` px over (sign 1) or under (-1)
            the bar, keeps clear of every mark and inside the plot."""
            near, far = gap, gap + font
            top_px = (y1 - m) * ypx
            bottom_px = (m - y0) * ypx
            if (sign > 0 and far > top_px) or (sign < 0 and far > bottom_px):
                return False
            for _dx, dy, r in marks:
                d = dy * sign
                if near - r - _AVG_CLEAR < d < far + r + _AVG_CLEAR:
                    return False
            return True

        best = None
        for gap in range(_AVG_GAP, _AVG_GAP + _AVG_MAX_SHIFT + 1):
            # Under first (where the label always was), then over.
            for sign in (-1, 1):
                if clear(sign, gap):
                    best = (sign, gap)
                    break
            if best:
                break
        sign, gap = best or (-1, _AVG_GAP)
        out.append(m + sign * (gap + font / 2) / ypx)
    return out


def round_highlight(opt, r):
    """(band from, band to, marker x, Best so far at Round r, its
    Formulation number) for the score chart's chosen Round: the band over
    Round r's Formulations (± 0.5); the marker on Best so far by the end of
    Round r (round_bests' best) itself, at its Formulation and score — it
    may come from an earlier Round. None when Round r has no formulation."""
    nos = [int(opt.formulation_ids[i]) for i in range(len(opt.formulation_ids))
           if _round_of(opt, i) == r]
    _rounds, at = round_bests(opt)
    if not nos or r not in at:
        return None
    i = at[r]
    no = int(opt.formulation_ids[i])
    return min(nos) - 0.5, max(nos) + 0.5, float(no), float(opt.Y_history[i]), no


_HIGHLIGHT_FILL = {"light": 0.07, "dark": 0.12}     # the chosen Round's band, opacity


def highlight_band(lit, x_scale=None):
    """The chosen Round's band: a light rect over its Formulations, the
    plot's full height (no y), under every mark."""
    hdf = pd.DataFrame([{"x": lit[0], "x2": lit[1]}])
    x = alt.X("x:Q", scale=x_scale) if x_scale is not None else alt.X("x:Q")
    return alt.Chart(hdf).mark_rect(color=model_blue(), opacity=_HIGHLIGHT_FILL[theme()],
                                    tooltip=False).encode(x=x, x2="x2:Q")


def highlight_marker(opt, lit, r, x_scale=None, y_scale=None, y_title=None):
    """The marker: Best so far at the chosen Round, a blue diamond on that
    formulation's own dot, its Round, score and Formulation in the tooltip."""
    best = wording.BEST_SO_FAR_COLUMN
    mdf = pd.DataFrame([{"x": lit[2], "v": lit[3], "round": wording.round_tick(r),
                         "text": one_decimal(lit[3]), "f": _short(opt, f"F{lit[4]}")}])
    x = alt.X("x:Q", scale=x_scale) if x_scale is not None else alt.X("x:Q")
    y = (alt.Y("v:Q", scale=y_scale, title=y_title) if y_scale is not None
         else alt.Y("v:Q"))
    return alt.Chart(mdf).mark_point(
        shape="diamond", filled=True, size=150, opacity=1, color=model_blue(),
        stroke=_page(), strokeWidth=1.5).encode(
        x=x, y=y,
        tooltip=[alt.Tooltip("round:N", title=_say(opt, wording.ROUND_CAP)),
                 alt.Tooltip("text:N", title=_say(opt, best)),
                 alt.Tooltip("f:N", title=_say(opt, wording.FORMULATION_CAP))])


def _layer_dict(chart):
    """One small layer as a Vega-Lite dict, its data inline (it is added to
    a kept spec, inside the score chart's shared scales)."""
    d = chart.to_dict(validate=False)
    for k in ("$schema", "config"):
        d.pop(k, None)
    name = (d.get("data") or {}).get("name")
    if name and name in d.get("datasets", {}):
        d["data"] = {"values": d["datasets"][name]}
    d.pop("datasets", None)
    return d


def _plot_layers(spec):
    """The score chart's plot layers (under the Rounds' top axis layer)."""
    top = spec.get("layer") or []
    if len(top) == 2 and isinstance(top[1], dict) and "layer" in top[1]:
        return top[1]["layer"]
    return top


def improving_spec(opt, r=None, height=None):
    """The score chart as a kept spec, Round `r` highlighted: the chart
    without a highlight is converted once per project state and theme;
    the band (first, under every mark) and the marker (last) are added per
    Round. None when there is nothing to draw."""
    # Final demo (B2): an earlier Round draws only the Rounds up to it.
    rounds = _section_rounds(opt) if r is not None else []
    upto = r if r is not None and rounds and r in rounds[:-1] else None
    key = ("improving", insights.data_key(opt), spider_key(opt), insights.setup_key(opt),
           theme(), _say(opt, wording.ROUND_CAP), float(opt.utility_ceiling()),
           getattr(insights.noise_yardstick(opt), "assumed", True), height or IMPROVING_HEIGHT,
           insights.assumed_noise(opt), upto)
    base = chart_spec(key, lambda: improving_chart(opt, height=height, upto=upto))
    if base is None or r is None:
        return base
    lit = round_highlight(opt, r)
    if lit is None:
        return base
    out = _spec_copy(base)
    plot = _plot_layers(out)
    step = next((l for l in plot if (l.get("mark") or {}).get("type") == "line"), None)
    band = _layer_dict(highlight_band(lit))
    marker = _layer_dict(highlight_marker(opt, lit, r))
    if step is not None:
        # The step's own x and y (its scales and axis title), shared.
        band["encoding"]["x"] = dict(step["encoding"]["x"], field="x")
        for k in ("x", "y"):
            marker["encoding"][k] = dict(step["encoding"][k],
                                         field="x" if k == "x" else "v")
    plot.insert(0, band)
    plot.append(marker)
    return out


def stop_signal(opt):
    """Final demo (B3): the first Round after which the stop rule held —
    Best so far by then within the tie threshold of the top score
    (at_ceiling's rule, at the project's tasting noise) — as (Round, best
    then, best now, gain since, threshold); None when it never held or
    the noise is not known. Model-free except the noise."""
    try:
        rounds, at = round_bests(opt)
        need = tie_need(insights.noise_yardstick(opt))
    except Exception:
        return None
    if not rounds or not need:
        return None
    top = float(opt.utility_ceiling())
    now = float(opt.Y_history[at[rounds[-1]]])
    for r in rounds:
        best = float(opt.Y_history[at[r]])
        if round(top - best, 2) < round(need, 2):
            return r, best, now, now - best, need
    return None


def stop_marker_x(opt, r):
    """The x of the rule after Round `r`: half a step past its last
    Formulation; None when it has none."""
    nos = [int(opt.formulation_ids[i]) for i in range(len(opt.formulation_ids))
           if _round_of(opt, i) == r]
    return max(nos) + 0.5 if nos else None


def _drawn_stop(opt):
    """The stop marker the Latest chart draws: stop_signal's, only when a
    scored Round came after it (at the latest Round the What next? callout
    says it). None otherwise."""
    sig = stop_signal(opt)
    if sig is None:
        return None
    rounds = _section_rounds(opt)
    if not rounds or sig[0] >= rounds[-1]:
        return None
    x = stop_marker_x(opt, sig[0])
    return None if x is None else (sig, x)


_STOP_LANE = 16              # px under the data for the stop marker's labels
_STOP_FILL = {"light": "#b7791f", "dark": "#d69e2e"}    # warm, never error-red


def improving_chart(opt, highlight=None, height=None, upto=None):
    """Are we improving? (a): each formulation's measured Overall score
    (grey dots), Best so far (an ink step), each Round's average (a short
    mid-grey bar under its Formulations) and the start and the best named
    on the chart. Model-free. x: Formulations in the order made, the
    Rounds marked along the top — the same axis as the forecast chart.
    `highlight`: a Round to mark (the section's Round control) — a light band over
    its Formulations and a marker on Best so far there. `upto` (final demo,
    B2): an earlier Round on view — the formulations, Round averages, band
    and Best so far drawn only up to it (the axes as at the latest, so a
    step never moves them); None draws everything, with the stop marker
    (B3) when the stop rule held before the latest Round."""
    plot_h = height or IMPROVING_HEIGHT
    narrow = height is not None and height < IMPROVING_HEIGHT
    frame = progress_frame(opt).reset_index()
    if frame.empty:
        return None
    formulation = frame.columns[0]
    score, best = wording.OVERALL_SCORE_COLUMN, wording.BEST_SO_FAR_COLUMN
    pool = set(opt.best_candidates())
    frame["full"] = [i in pool for i in range(len(frame))]
    each, best_name = _say(opt, wording.STORY_EACH), _say(opt, wording.BEST_SO_FAR_COLUMN)
    avg_name = _say(opt, wording.STORY_ROUND_AVERAGE)
    frame["series"], frame["best_series"] = each, best_name
    frame["round"] = [_round_label(r) for r in
                      (list(opt.batch_history) + [None] * len(frame))[:len(frame)]]
    frame["bo"] = wording.PLOT_BO_MODEL_FREE
    avgs = round_averages(opt)
    values = [v for v in list(frame[score]) + list(frame[best]) if v is not None and v == v]
    if not values:
        return None
    nums = story_numbers(opt)
    lo, hi = min(values), max(values)
    ceiling = float(opt.utility_ceiling())
    steps = best_steps(opt)
    stop = _drawn_stop(opt)
    if upto is not None:
        # Final demo (B2): nothing after Round `upto` (the scales stay).
        later = [r is not None and r > upto for r in
                 (list(opt.batch_history) + [None] * len(frame))[:len(frame)]]
        frame.loc[later, score] = None
        avgs = [a for a in avgs if a[0] <= upto]
        steps = [st_ for st_ in steps if st_[3] <= upto]
        _rounds, at = round_bests(opt)
        if nums is not None and upto in at:
            bi = at[upto]
            nums = (nums[0], nums[1], float(opt.Y_history[bi]), _round_of(opt, bi),
                    int(opt.formulation_ids[bi]))
    noise = leader_noise(opt)
    # Review critique (fix 7): the top of the score drawn (a rule and its
    # tick) when it is near the data.
    show_top = ceiling >= hi and ceiling - hi <= max(hi - lo, 4.0)
    # Review readable: the Start and Best labels sit in two lanes above
    # every mark (_IMPROVING_LANES px), each joined to its point by a thin
    # leader, so neither label lands on a line or a dot.
    bottom = lo - 0.04 * max(hi - lo, 4.0)
    data_top = (ceiling if show_top else hi) + 0.03 * max(hi - lo, 4.0)
    lanes = _IMPROVING_LANES if nums is not None else 0
    # Final demo (B3): a lane under the data for the stop marker's labels,
    # kept at every Round so a step never moves the scale.
    low_lane = _STOP_LANE if stop is not None else 0
    data_bottom = bottom
    bottom = bottom - (data_top - bottom) * low_lane / max(1, plot_h - lanes - low_lane)
    top = data_top + (data_top - bottom) * lanes / (plot_h - lanes)
    y_scale = alt.Scale(domain=[bottom, top], nice=False, zero=False)

    def at_px(px):
        """The score drawn `px` below the plot's top."""
        return top - (top - bottom) * px / plot_h
    y_title = zoomed(wording.PLOT_MEASURED_SCORE, lo, hi, float(opt.utility_ceiling()))
    x_scale = alt.Scale(domain=_x_domain(opt), nice=False)
    x = alt.X(f"{formulation}:Q", title=_say(opt, wording.Q2_X_TITLE), scale=x_scale,
              axis=alt.Axis(format="d", values=formulation_ticks(frame[formulation]),
                            grid=False))
    grey, _ = mark("neutral")
    ink, _ = mark("bar_latest")
    light, mid = _TRIED_GREYS[theme()]
    # The legend names the two lines; the grey dots are each formulation
    # (their tooltip and the caption's "?" say so).
    # In the Replay panel (a narrow column) each legend is one entry a row.
    legend = alt.Legend(title=None, orient="top", labelLimit=0, symbolType="stroke",
                        symbolStrokeWidth=3, symbolOpacity=1,
                        **({"columns": 1} if narrow else {}))
    no_tip = alt.Tooltip(f"{formulation}:Q", title=_say(opt, wording.FORMULATION_CAP),
                         format="d")
    # Review readable: ticks only over the data, never in the label lanes.
    span = data_top - bottom
    step = 5 if span <= 30 else 10 if span <= 60 else 20
    first = math.ceil(data_bottom / step) * step
    ticks = [first + step * k for k in range(int((data_top - first) // step) + 1)]
    if show_top and all(abs(t - ceiling) > 1e-9 for t in ticks):
        ticks.append(ceiling)
    # Round-2 fixes (P2 #7): the dotted line at the top is named on its
    # own tick, "Top score 100", where no mark can cover it.
    top_tick = (f"abs(datum.value - {ceiling}) < 1e-9 ? "
                f"'{_say(opt, wording.top_score_label(f'{ceiling:g}'))}' : datum.label")
    y = alt.Y(f"{score}:Q", title=y_title, scale=y_scale,
              axis=alt.Axis(values=ticks, labelExpr=top_tick if show_top else "datum.label"))
    layers = []
    band_name = _say(opt, wording.noise_band(noise_sd_text(opt)) if getattr(
        insights.noise_yardstick(opt), "assumed", True) else wording.NOISE_BAND_MEASURED)
    zone_name = _say(opt, wording.TOP_ZONE)
    kinds = alt.Scale(domain=[best_name, avg_name], range=[ink, mid])
    # The two shaded areas have their own legend, a pale square each.
    zone_fill = _ZONE_FILL[theme()]
    area_names = [band_name] + ([zone_name] if show_top else [])
    areas = alt.Scale(domain=area_names,
                      range=[_BAND_FILL[theme()][0]] + ([zone_fill] if show_top else []))
    area_legend = alt.Legend(title=None, orient="top", labelLimit=0, symbolType="square",
                             symbolOpacity=0.35, symbolStrokeWidth=0,
                             **({"columns": 1} if narrow else {}))
    need = tie_need(insights.noise_yardstick(opt)) if show_top else None
    lit = round_highlight(opt, highlight) if highlight is not None else None
    if lit is not None:
        # Slim Review: the chosen Round, a light band under every mark.
        layers.append(highlight_band(lit, x_scale))
    if show_top and need:
        # Round-2 fixes (P2 #7): the zone within noise of the top (top − the
        # tie threshold to top): a best in it can't be told from the top.
        lo_x, hi_x = _x_domain(opt)
        zdf = pd.DataFrame([{"x": lo_x, "x2": hi_x, "lo": max(bottom, ceiling - need),
                             "hi": ceiling, "what": zone_name}])
        layers.append(alt.Chart(zdf).mark_rect(opacity=0.5, tooltip=False).encode(
            x=alt.X("x:Q", scale=x_scale), x2="x2:Q",
            y=alt.Y("lo:Q", scale=y_scale, title=y_title), y2="hi:Q",
            fill=alt.Fill("what:N", scale=areas, legend=area_legend)))
    if steps:
        # The ±noise band around the best step, light, under everything;
        # clipped at the top score (round-2 fixes, P1 #6).
        sdf = pd.DataFrame([{"x": x, "lo": v - noise, "hi": min(ceiling, v + noise), "v": v,
                             "what": band_name, best_name: v, "round": wording.round_tick(r)}
                            for a, b, v, r in steps for x in (a, b)])
        layers.append(alt.Chart(sdf).mark_area(opacity=0.25, tooltip=False).encode(
            x=alt.X("x:Q", scale=x_scale), y=alt.Y("lo:Q", scale=y_scale, title=y_title),
            y2="hi:Q", fill=alt.Fill("what:N", scale=areas, legend=area_legend)))
    if stop is not None and upto is None:
        layers += _stop_layers(opt, stop, x_scale, y_scale, y_title, bottom, data_bottom,
                               data_top)
    if show_top:
        tdf = pd.DataFrame([{"y": ceiling}])
        layers.append(alt.Chart(tdf).mark_rule(color=grey, strokeDash=[2, 3], strokeWidth=1,
                                               tooltip=False)
                      .encode(y=alt.Y("y:Q", scale=y_scale, title=y_title)))
    if avgs:
        adf = pd.DataFrame([{"x0": a - 0.4, "x1": b + 0.4, "mid": (a + b) / 2, "avg": m,
                             "what": avg_name, "round": wording.round_tick(r), "n": n,
                             "text": one_decimal(m)} for r, a, b, m, n in avgs])
        layers.append(alt.Chart(adf).mark_rule(strokeWidth=3).encode(
            x=alt.X("x0:Q", scale=x_scale), x2="x1:Q",
            y=alt.Y("avg:Q", scale=y_scale, title=y_title),
            color=alt.Color("what:N", scale=kinds, legend=legend),
            tooltip=[alt.Tooltip("round:N", title=_say(opt, wording.ROUND_CAP)),
                     alt.Tooltip("text:N", title=avg_name)]))
        # Round-2 fixes (P2 #7): each Round's average printed by its bar,
        # no hover needed; under it unless a mark is there (zoom check).
        rings = []
        if nums is not None:
            rings = [(int(opt.formulation_ids[story_start(opt)[0]]), nums[0]),
                     (nums[4], nums[2])]
        dots = [(float(n), float(v)) for n, v in zip(frame[formulation], frame[score])
                if v is not None and v == v]
        adf["label_y"] = average_label_y(avgs, dots, steps, rings, (bottom, top),
                                         tuple(_x_domain(opt)), height=plot_h,
                                         width=PANEL_PLOT_W if narrow else _MIN_PLOT_W)
        layers += _halo_text(adf, alt.X("mid:Q", scale=x_scale),
                             alt.Y("label_y:Q", scale=y_scale, title=y_title), "text:N", mid,
                             fontSize=FONT_SMALL, baseline="middle")
    layers.append(alt.Chart(frame).mark_point(size=30, strokeWidth=1.3, opacity=0.9,
                                              color=grey).encode(
        x=x, y=y,
        fill=alt.condition(alt.FieldEqualPredicate(field="full", equal=True),
                           alt.value(grey), alt.value("transparent")),
        tooltip=[no_tip, alt.Tooltip("round:N", title=_say(opt, wording.ROUND_CAP)),
                 alt.Tooltip(f"{score}:Q", title=score, format=".1f"), bo_tip()])
        .transform_filter(alt.FieldValidPredicate(field=score, valid=True)))
    if steps:
        # Review critique (fix 7): the best moves only at Round boundaries.
        bdf = pd.DataFrame([{"x": x, "v": v, "best_series": best_name,
                             "round": wording.round_tick(r), "text": one_decimal(v)}
                            for a, b, v, r in steps for x in (a, b)])
        layers.append(alt.Chart(bdf).mark_line(strokeWidth=2.5).encode(
            x=alt.X("x:Q", scale=x_scale), y=alt.Y("v:Q", scale=y_scale, title=y_title),
            color=alt.Color("best_series:N", scale=kinds, legend=legend),
            tooltip=[alt.Tooltip("round:N", title=_say(opt, wording.ROUND_CAP)),
                     alt.Tooltip("text:N", title=best)]))
    if lit is not None:
        # The marker: Best so far at the chosen Round, on its own dot.
        layers.append(highlight_marker(opt, lit, highlight, x_scale, y_scale, y_title))
    if nums is not None:
        s, r0, b, rb, bno = nums
        sno = int(opt.formulation_ids[story_start(opt)[0]])
        lo_x, hi_x = _x_domain(opt)
        span = hi_x - lo_x
        pts = []
        # Best in the top lane, the start in the one under it.
        if (sno, s) != (bno, b):
            pts.append({"x": sno, "y": s, "text": wording.story_point(
                wording.STORY_START, one_decimal(s), r0), "who": "start",
                "lane": at_px(_IMPROVING_LANES - 16)})
        if rb is not None:
            pts.append({"x": bno, "y": b, "text": wording.story_point(
                wording.STORY_BEST, one_decimal(b), rb), "who": "best",
                "lane": at_px(12)})
        for p in pts:
            share = (p["x"] - lo_x) / span if span else 0.5
            align = "left" if share < 0.3 else "right" if share > 0.7 else "center"
            dx = {"left": -6, "right": 6, "center": 0}[align]
            pdf = pd.DataFrame([p])
            ring_c = ink if p["who"] == "best" else grey
            # The leader: from just above the ring up to its label.
            px = (top - bottom) / plot_h
            layers.append(alt.Chart(pdf.assign(foot=p["y"] + 8 * px, head=p["lane"] - 9 * px))
                          .mark_rule(color=grey, strokeWidth=1, strokeDash=[2, 2], tooltip=False)
                          .encode(x=alt.X("x:Q", scale=x_scale),
                                  y=alt.Y("foot:Q", scale=y_scale, title=y_title), y2="head:Q"))
            layers.append(alt.Chart(pdf).mark_point(size=150, strokeWidth=1.8, color=ring_c,
                                                    filled=False, tooltip=False)
                          .encode(x=alt.X("x:Q", scale=x_scale),
                                  y=alt.Y("y:Q", scale=y_scale, title=y_title)))
            layers += _halo_text(pdf, alt.X("x:Q", scale=x_scale),
                                 alt.Y("lane:Q", scale=y_scale, title=y_title), "text:N",
                                 ink if p["who"] == "best" else muted(), fontSize=FONT_LABEL,
                                 fontWeight="bold" if p["who"] == "best" else "normal",
                                 align=align, dx=dx, baseline="middle")
    rounds = _round_layer(opt, formulation, list(frame[formulation]), x_scale)
    body = alt.layer(*layers)
    chart = alt.layer(rounds, body).resolve_axis(x="independent") if rounds else body
    return readable(chart.properties(height=plot_h))


def _stop_layers(opt, stop, x_scale, y_scale, y_title, bottom, data_bottom, data_top):
    """Final demo (B3): the stop marker — a light warm shade over the
    Rounds after the stop, a dashed rule after its Round, and in the lane
    under the data 'Stop signal (Round 6)' left of the rule and 'after the
    stop: +0.8' right of it."""
    (r, best_then, now, gain, need), x = stop
    hi_x = _x_domain(opt)[1]
    warm = _STOP_FILL[theme()]
    tip = _say(opt, wording.stop_marker_help(r, one_decimal(best_then), one_decimal(now),
                                             one_decimal(max(0.0, gain)), f"{need:.1f}"))
    out = []
    shade = pd.DataFrame([{"x": x, "x2": hi_x, "lo": bottom, "hi": data_top, "tip": tip}])
    out.append(alt.Chart(shade).mark_rect(color=warm, opacity=0.08).encode(
        x=alt.X("x:Q", scale=x_scale), x2="x2:Q",
        y=alt.Y("lo:Q", scale=y_scale, title=y_title), y2="hi:Q",
        tooltip=alt.Tooltip("tip:N", title=None)))
    rule = pd.DataFrame([{"x": x, "lo": bottom, "hi": data_top, "tip": tip}])
    out.append(alt.Chart(rule).mark_rule(color=warm, strokeWidth=1.5, strokeDash=[4, 3]).encode(
        x=alt.X("x:Q", scale=x_scale), y=alt.Y("lo:Q", scale=y_scale, title=y_title),
        y2="hi:Q", tooltip=alt.Tooltip("tip:N", title=None)))
    lane = (bottom + data_bottom) / 2
    labels = pd.DataFrame([
        {"x": x, "y": lane, "text": _say(opt, wording.stop_marker(r)), "side": "left"},
        {"x": x, "y": lane, "text": wording.after_stop(one_decimal(max(0.0, gain))),
         "side": "right"}])
    for side, align, dx in (("left", "right", -4), ("right", "left", 4)):
        part = labels[labels["side"] == side]
        out += _halo_text(part, alt.X("x:Q", scale=x_scale),
                          alt.Y("y:Q", scale=y_scale, title=y_title), "text:N", warm,
                          fontSize=FONT_SMALL, fontWeight="bold", align=align, dx=dx,
                          baseline="middle")
    return out


def target_rows(opt, best=None):
    """Closer to target: one row per measurement (largest Share of score
    first) — name, goal, target, Range, the start's value and Best so
    far's (at position `best` when given: Best so far at a chosen Round)
    — or [] with no start or no best."""
    s = story_start(opt)
    b = opt.best_index() if best is None else best
    if s is None or b is None:
        return []
    start, best = opt._results_at(s[0]), opt._results_at(b)
    out = []
    for obj in opt.measurements_by_importance():
        lo, hi = float(obj.get('min_val', 0.0)), float(obj.get('max_val', 10.0))
        v0, v1 = start.get(obj['name']), best.get(obj['name'])
        out.append({"name": obj['name'], "goal": obj.get('goal', 'max'),
                    "target": (float(obj['target']) if obj.get('goal') == 'target'
                               and obj.get('target') is not None else None),
                    "low": lo, "high": hi,
                    "start": None if v0 is None else float(v0),
                    "best": None if v1 is None else float(v1),
                    "unit": str(obj.get('unit') or "")})
    return out


def _share(row, v):
    span = row["high"] - row["low"]
    return 0.5 if span <= 0 else (float(v) - row["low"]) / span


ON_TARGET = 0.01        # within this share of its Range: on target


def target_move(r, tol=None):
    """Review critique (fix 6): how one measurement moved from the start to
    Best so far against its aim — ("onto" | "stayed" | "closer" | "away" |
    "level", distance moved toward (+) or away (−) from the target, in the
    measurement's units); None without both values. `tol`: the on-target
    and unchanged tolerance in the measurement's units (the sensory profile
    passes the Stop line's, on_ideal); None: ON_TARGET of its Range."""
    if r["start"] is None or r["best"] is None:
        return None
    span = max(r["high"] - r["low"], 1e-9)
    near = (lambda d: d <= ON_TARGET * span) if tol is None else (lambda d: d < tol)
    if r["target"] is not None:
        d0, d1 = abs(r["start"] - r["target"]), abs(r["best"] - r["target"])
        on0, on1 = near(d0), near(d1)
        if on0 and on1:
            return "stayed", 0.0
        if on1:
            return "onto", d0 - d1
        gain = d0 - d1
    else:
        sign = -1.0 if r["goal"] == "min" else 1.0
        gain = sign * (r["best"] - r["start"])
    if near(abs(gain)):
        return "level", 0.0
    return ("closer" if gain > 0 else "away"), gain


def _move_text(v, span):
    d = 0 if span >= 100 else 1
    text = f"{abs(v):.{d}f}"
    return text[:-2] if text.endswith(".0") else text


def target_answer(opt, rows=None, tols=None):
    """The chart's answer line, from the data: '2 moved onto target, 1
    stayed on target, 1 moved 0.5 away.'; '' with nothing to say. `rows`:
    target_rows' (the default) or the sensory profile's for one Round;
    `tols`: {measurement: tolerance} for target_move."""
    onto = stayed = level = 0
    closer, away = [], []
    for r in target_rows(opt) if rows is None else rows:
        m = target_move(r, (tols or {}).get(r["name"]))
        if m is None:
            continue
        kind, v = m
        span = r["high"] - r["low"]
        if kind == "onto":
            onto += 1
        elif kind == "stayed":
            stayed += 1
        elif kind == "closer":
            closer.append(_move_text(v, span))
        elif kind == "away":
            away.append(_move_text(v, span))
        else:
            level += 1
    return _say(opt, wording.target_moves(onto, stayed, closer, away, level))


def target_chart(opt, best=None):
    """Start vs best, against target (review critique, fix 6): one row per
    measurement on its own Range (0 to 1 across, its ends written): the
    target the strongest mark, a large ink diamond (an arrow at the better
    end for a 'higher' or 'lower' measurement); the first Round's best a
    hollow grey ring, never hidden under the best; Best so far a filled ink
    dot; the segment between them ink when the best is closer to the aim,
    grey and labelled "moved away" when farther. Model-free. None with
    nothing to draw. `best`: Best so far's position at a chosen Round."""
    rows = target_rows(opt, best=best)
    if not rows:
        return None
    s = story_start(opt)
    start_name = _say(opt, wording.story_target_start(s[1]))
    best_name = _say(opt, wording.STORY_TARGET_BEST)
    target_name = _say(opt, wording.TARGET_KEY)
    closer_name, away_name = (_say(opt, wording.TARGET_KEY_CLOSER),
                              _say(opt, wording.TARGET_KEY_AWAY))
    grey, _ = mark("neutral")
    ink, _ = mark("bar_latest")
    light, _ = _TRIED_GREYS[theme()]
    labels = [wording.story_target_row(r["name"], r["goal"], r["target"]) for r in rows]
    track, points, joins, ends, arrows, texts, targets = [], [], [], [], [], [], []
    for k, r in enumerate(rows):
        track.append({"k": k, "t0": 0.0, "t1": 1.0})
        d = 0 if (r["high"] - r["low"]) >= 10 else 1
        fmt = lambda v: f"{v:g}" if float(v).is_integer() else f"{v:.{max(1, d)}f}"   # noqa: E731
        ends += [{"k": k, "t": 0.0, "text": fmt(r["low"]), "align": "right"},
                 {"k": k, "t": 1.0, "text": fmt(r["high"]), "align": "left"}]
        tip = f"{r['name']}: " + " → ".join(
            fmt(v) for v in (r["start"], r["best"]) if v is not None)
        if r["target"] is not None:
            targets.append({"k": k, "t": _share(r, r["target"]), "what": target_name,
                            "value": fmt(r["target"]), "tip": tip})
        elif r["goal"] in ("max", "min"):
            arrows.append({"k": k, "t": 1.0 if r["goal"] == "max" else 0.0,
                           "shape": "triangle-right" if r["goal"] == "max" else "triangle-left"})
        move = target_move(r)
        if move is not None and move[0] in ("onto", "closer", "away"):
            joins.append({"k": k, "t0": _share(r, r["start"]), "t1": _share(r, r["best"]),
                          "what": away_name if move[0] == "away" else closer_name,
                          "mid": (_share(r, r["start"]) + _share(r, r["best"])) / 2,
                          "note": _say(opt, wording.TARGET_AWAY_LABEL) if move[0] == "away"
                          else ""})
        for v, kind in ((r["start"], start_name), (r["best"], best_name)):
            if v is not None:
                points.append({"k": k, "t": _share(r, v), "what": kind, "value": fmt(v),
                               "tip": tip})
        if r["best"] is not None:
            texts.append({"k": k, "t": _share(r, r["best"]), "text": fmt(r["best"]),
                          "who": "best"})
        if r["start"] is not None and (r["best"] is None
                                       or abs(_share(r, r["start"]) - _share(r, r["best"]))
                                       > 0.005):
            texts.append({"k": k, "t": _share(r, r["start"]), "text": fmt(r["start"]),
                          "who": "start"})
    n = len(rows)
    # Round-2 fixes (P2 #10): one Range for every row draws one shared axis
    # under the chart (no 0 and 15 on each row); mixed Ranges keep the ends.
    shared = len({(r["low"], r["high"]) for r in rows}) == 1
    y_scale = alt.Scale(domain=[-0.7, n - 0.3 + (0.6 if shared else 0)], reverse=True,
                        nice=False, zero=False)
    expr = " : ".join(f"datum.value == {k} ? '{_esc(l)}'" for k, l in enumerate(labels)) + " : ''"
    pad = 8 if shared else 14 + 7 * max(len(e["text"]) for e in ends)
    y_axis = alt.Axis(values=list(range(n)), labelExpr=expr, title=None, ticks=False,
                      domain=False, grid=False, labelFontSize=FONT_LABEL, labelLimit=260,
                      labelPadding=pad)
    y = alt.Y("k:Q", scale=y_scale, axis=y_axis)
    x_scale = alt.Scale(domain=[-0.02, 1.02], nice=False)
    x = alt.X("t:Q", scale=x_scale, axis=None)
    kinds = [target_name, start_name, best_name, closer_name, away_name]
    colour = alt.Scale(domain=kinds, range=[ink, grey, ink, ink, grey])
    legend = alt.Legend(title=None, orient="bottom", labelLimit=0, symbolStrokeWidth=2)
    # One legend: each entry drawn as on the chart (diamond, ring, dot,
    # line); the start's ring and the target hollow, as drawn (P2 #10).
    shapes = alt.Scale(domain=kinds, range=["diamond", "circle", "circle", "stroke", "stroke"])
    shape = alt.Shape("what:N", scale=shapes, legend=legend)
    fills = alt.Scale(domain=kinds, range=["transparent", "transparent", ink, "transparent",
                                           "transparent"])
    fill = alt.Fill("what:N", scale=fills, legend=legend)
    layers = [
        alt.Chart(pd.DataFrame(track)).mark_rule(color=light, strokeWidth=4, tooltip=False)
        .encode(x=alt.X("t0:Q", scale=x_scale, axis=None), x2="t1:Q", y=y)]
    if shared:
        # The shared axis drawn as text under the last row (a layer's axis
        # would be merged away by the others' axis=None).
        lo_v, hi_v = rows[0]["low"], rows[0]["high"]
        span_v = (hi_v - lo_v) or 1.0
        step_v = 5 if span_v <= 30 else 10 if span_v <= 60 else 20
        vals = sorted({lo_v, hi_v, *_frange(math.ceil(lo_v / step_v) * step_v, hi_v, step_v)})
        adf = pd.DataFrame([{"t": (v - lo_v) / span_v, "k": n - 1, "text": f"{v:g}"}
                            for v in vals])
        layers.append(alt.Chart(adf).mark_text(fontSize=FONT_SMALL, color=muted(),
                                               tooltip=False, baseline="top", dy=30)
                      .encode(x=x, y=y, text="text:N"))
        layers.append(alt.Chart(adf).mark_tick(color=muted(), orient="vertical", size=5,
                                               thickness=1, tooltip=False, yOffset=26)
                      .encode(x=x, y=y))
    if not shared:
        layers += [
            alt.Chart(pd.DataFrame(ends)).mark_text(fontSize=FONT_SMALL, color=muted(),
                                                    tooltip=False, dx=-7, align="right",
                                                    baseline="middle")
            .encode(x=x, y=y, text="text:N").transform_filter("datum.align == 'right'"),
            alt.Chart(pd.DataFrame(ends)).mark_text(fontSize=FONT_SMALL, color=muted(),
                                                    tooltip=False, dx=7, align="left",
                                                    baseline="middle")
            .encode(x=x, y=y, text="text:N").transform_filter("datum.align == 'left'")]
    if joins:
        jdf = pd.DataFrame(joins)
        layers.append(alt.Chart(jdf).mark_rule(strokeWidth=3, tooltip=False)
                      .encode(x=alt.X("t0:Q", scale=x_scale, axis=None), x2="t1:Q", y=y,
                              color=alt.Color("what:N", scale=colour, legend=legend)))
        away = jdf[jdf["note"] != ""]
        # Round-2 fixes (P2 #10): "moved away" on the side the dot moved to,
        # past the best's dot, clear of the values above and below it.
        for side, sel in (("left", away["t1"] < away["t0"]), ("right", away["t1"] >= away["t0"])):
            part = away[sel]
            if part.empty:
                continue
            layers += _halo_text(part, alt.X("t1:Q", scale=x_scale, axis=None), y, "note:N",
                                 muted(), fontSize=FONT_SMALL, dx=-14 if side == "left" else 14,
                                 align="right" if side == "left" else "left",
                                 baseline="middle", fontStyle="italic")
    if targets:
        # The strongest mark: a large ink diamond, hollow, so a dot on the
        # target still shows inside it.
        layers.append(alt.Chart(pd.DataFrame(targets)).mark_point(
            shape="diamond", size=420, strokeWidth=2.4, filled=False, opacity=1)
            .encode(x=x, y=y, color=alt.Color("what:N", scale=colour, legend=legend),
                    shape=shape, fill=fill,
                    tooltip=[alt.Tooltip("value:N", title=target_name)]))
    if arrows:
        layers.append(alt.Chart(pd.DataFrame(arrows)).mark_point(
            filled=True, size=90, color=ink, tooltip=False)
            .encode(x=x, y=y, shape=alt.Shape("shape:N", scale=None)))
    pdf = pd.DataFrame(points)
    tips = [alt.Tooltip("what:N", title=" "), alt.Tooltip("value:N", title=wording.STORY_VALUE),
            alt.Tooltip("tip:N", title=_say(opt, wording.STORY_TARGET_TITLE))]
    # The start a ring wider than the best's dot: never hidden under it.
    layers.append(alt.Chart(pdf[pdf["what"] == start_name]).mark_point(
        size=190, strokeWidth=2, filled=False, opacity=1)
        .encode(x=x, y=y, color=alt.Color("what:N", scale=colour, legend=legend), shape=shape,
                fill=fill, tooltip=tips))
    layers.append(alt.Chart(pdf[pdf["what"] == best_name]).mark_point(
        size=70, filled=True, opacity=1)
        .encode(x=x, y=y, color=alt.Color("what:N", scale=colour, legend=legend), shape=shape,
                fill=fill, tooltip=tips))
    if texts:
        # Best above its dot, the start below its ring: each row's values
        # on its own side, the rows far enough apart (TARGET_ROW_H) that no
        # value reads as the next row's.
        tdf = pd.DataFrame(texts)
        layers += _halo_text(tdf[tdf["who"] == "best"], x, y, "text:N", ink,
                             fontSize=FONT_LABEL, fontWeight="bold", dy=-11, baseline="bottom")
        if (tdf["who"] == "start").any():
            layers += _halo_text(tdf[tdf["who"] == "start"], x, y, "text:N", muted(),
                                 fontSize=FONT_LABEL, dy=11, baseline="top")
    return readable(alt.layer(*layers).properties(
        height=int(TARGET_ROW_H * (n + (0.6 if shared else 0)))))


# --- Sensory profile against target (review, 2026-10-08) ---
# From SPIDER_MIN measurements, Start vs best's dot chart becomes a spider:
# one spoke per measurement (most important first, clockwise from the top),
# each spoke its own Range from the centre (low) to the rim (high). The
# targets in bold ink, Best so far at the chosen Round filled blue, the Start
# (story_start, as What changed) dashed grey, the earlier Rounds' bests
# faint. Measured values only: no model, no forecast, nothing drawn between
# Rounds.

SPIDER_MIN = 3              # measurements: from this many a spider, else the dot chart
SPIDER_SIZE = 400           # px: the plot's square (labels and legend sit outside it)
SPOKE_SEP = " · "           # a spoke label's name · target (wrapped at it beside What changed)
_SPIDER_PAD = 1.1           # the plot's half-width in rims: room for the marks on the rim
_SPIDER_LABEL_AT = 1.08     # spoke labels: just past the rim
_SPIDER_NUDGE = 0.07        # a value label's distance from its vertex, in rims
_SPIDER_INWARD_FROM = 0.86  # a value this near the rim is labelled inside it
_SPIDER_RING_SEGMENTS = 96
# Earlier Rounds' bests: a faint grey, between the gridlines' and the Start's.
_SPIDER_EARLIER = {"light": "#b8bdc5", "dark": "#5d636d"}
_SPIDER_CACHE = {}          # spider_key -> Spider: the per-Round bests, worked once
_SPIDER_CACHE_MAX = 16


@dataclasses.dataclass
class Spider:
    spokes: list      # [{"name", "goal", "target", "band", "low", "high"}]
    rounds: list      # the control's Rounds: the first scored to the latest scored
    bests: dict       # Round -> (position, number, {name: value}, Overall score)
    start: tuple      # (position, number, Round, {name: value})
    shared: tuple     # (low, high) when every spoke has one Range, else None


def spider_angle(k, n):
    """Spoke k of n: the first straight up, the rest clockwise (radians)."""
    return math.pi / 2 - 2 * math.pi * k / n


def spider_xy(k, n, frac):
    """(x, y) of a point `frac` rims out along spoke k of n (y up)."""
    a = spider_angle(k, n)
    return frac * math.cos(a), frac * math.sin(a)


def spider_frac(spoke, v):
    """A value's distance from the centre in rims: linear from the Range's
    low (0) to its high (1); a value past either end drawn at that end (its
    label still says the value)."""
    span = float(spoke["high"]) - float(spoke["low"])
    if span <= 0:
        return 0.5
    return min(1.0, max(0.0, (float(v) - float(spoke["low"])) / span))


def spider_spokes(opt):
    """One spoke per measurement, the largest Share of score first: name,
    goal, target (a point target; None for higher or lower is better),
    band (a target range (low, high); no measurement has one today) and
    Range."""
    out = []
    for obj in opt.measurements_by_importance():
        goal = obj.get('goal', 'max')
        out.append({"name": obj['name'], "goal": goal,
                    "target": (float(obj['target']) if goal == 'target'
                               and obj.get('target') is not None else None),
                    "band": None,
                    "low": float(obj.get('min_val', 0.0)),
                    "high": float(obj.get('max_val', 10.0))})
    return out


def spider_key(opt):
    """Everything the per-Round bests depend on."""
    return insights._digest({
        "name": getattr(opt, "project_name", ""), "Y": opt.Y_history,
        "R": opt.results_history, "ids": opt.formulation_ids, "rounds": opt.batch_history,
        "aside": [opt.is_set_aside(i) for i in range(len(opt.Y_history))],
        "m": [(o['name'], o.get('goal'), o.get('target'), o.get('min_val'), o.get('max_val'),
               o.get('weight'), o.get('not_for')) for o in opt.objectives]})


def spider_data(opt):
    """The spider's data (Spider), worked once per project state; None
    under SPIDER_MIN measurements or without a start and a Best so far."""
    key = spider_key(opt)
    if key in _SPIDER_CACHE:
        return _SPIDER_CACHE[key]
    data = _spider_data(opt)
    if len(_SPIDER_CACHE) >= _SPIDER_CACHE_MAX:
        _SPIDER_CACHE.clear()
    _SPIDER_CACHE[key] = data
    return data


def _round_of(opt, i):
    r = opt.batch_history[i] if i < len(opt.batch_history) else None
    return None if r is None else int(r)


def round_bests(opt):
    """(rounds, {Round: position}): the Round control's Rounds — the first
    scored Round to the latest — and Best so far by the end of each: the
    highest Overall score among best_candidates over the Rounds up to it,
    the first made on a tie (best_index's rule); the latest is the strip's
    own Best so far. ([], {}) with no scored Round. Model-free."""
    b = opt.best_index()
    pool = opt.best_candidates()
    named = sorted({_round_of(opt, i) for i in pool if _round_of(opt, i) is not None})
    if b is None or not named:
        return [], {}
    rounds = list(range(named[0], named[-1] + 1))
    bests = {}
    for r in rounds:
        mine = [i for i in pool if _round_of(opt, i) is not None and _round_of(opt, i) <= r]
        i = max(mine, key=lambda j: opt.Y_history[j])
        if r == rounds[-1] and i != b:
            i = b       # a Best so far with no Round: the strip's own
        bests[r] = i
    return rounds, bests


def _spider_data(opt):
    spokes = spider_spokes(opt)
    s, b = story_start(opt), opt.best_index()
    if len(spokes) < SPIDER_MIN or s is None or b is None:
        return None
    names = [sp["name"] for sp in spokes]

    def values(i):
        res = opt._results_at(i)
        return {n: None if res.get(n) is None else float(res[n]) for n in names}

    rounds, at = round_bests(opt)
    if not rounds:
        return None
    # Best so far by the end of each Round (round_bests: the app's rule).
    bests = {r: (i, int(opt.formulation_ids[i]), values(i), float(opt.Y_history[i]))
             for r, i in at.items()}
    i0, r0 = s
    ranges = {(sp["low"], sp["high"]) for sp in spokes}
    return Spider(spokes, rounds, bests, (i0, int(opt.formulation_ids[i0]), r0, values(i0)),
                  next(iter(ranges)) if len(ranges) == 1 else None)


def spider_ring_values(low, high):
    """The gridline rings: round values of the Range, three or four of
    them, the high end always (0–15: 5, 10, 15)."""
    span = high - low
    if span <= 0:
        return []
    mag = 10 ** math.floor(math.log10(span))
    step = next(m * mag for m in (0.1, 0.2, 0.25, 0.5, 1, 2, 2.5, 5, 10) if span / (m * mag) <= 4)
    vals = [v for v in _frange(math.ceil(low / step + 1e-9) * step, high, step) if v > low + 1e-9]
    if not vals or abs(vals[-1] - high) > 1e-6:
        vals.append(high)
    return vals


def _spider_num(spoke, v):
    """A value as the spider writes it: '12', '12.5' (the dot chart's rule)."""
    d = 0 if (spoke["high"] - spoke["low"]) >= 10 else 1
    v = float(v)
    return f"{v:g}" if v.is_integer() else f"{v:.{max(1, d)}f}"


def _spider_anchor(dx, dy):
    """(align, baseline) for text set off from a point in direction
    (dx, dy): the text grows away from the point."""
    align = "left" if dx > 0.3 else "right" if dx < -0.3 else "center"
    baseline = "bottom" if dy > 0.3 else "top" if dy < -0.3 else "middle"
    return align, baseline


def spider_round_answer(opt, data, r):
    """The answer line for Round r: 'Round 6: on target 3 of 4; Tangy 1.0
    under.' — ideal_gaps' count and order, the Stop line's own; at most
    SPIDER_ANSWER_WORDS (the largest miss alone when two run long)."""
    i, no, vals, _score = data.bests[r]
    gaps = ideal_gaps(opt, {k: v for k, v in vals.items() if v is not None})
    if gaps is None:
        return ""
    on, n, offs, targets = gaps
    texts = [wording.spider_off(o['name'], amount, "short" if o.get('goal') != 'target'
                                else "under" if gap < 0 else "over") for o, amount, gap in offs]
    # Final demo (A1): the best so far by then, and the Round it was made
    # in; a long name keeps its miss with the short head ('Best F21 …').
    for k, short in ((2, False), (1, False), (1, True), (0, False)):
        line = wording.spider_answer(r, on, n, texts[:k], targets, no=no,
                                     made_round=_round_of(opt, i), short=short)
        if limit_words(line) <= SPIDER_ANSWER_WORDS or k == 0:
            return _short(opt, line)


SPIDER_ANSWER_WORDS = 12


def spider_round_help(opt, data, r):
    """The answer line's "?": how each measurement moved from the Start to
    Round r's best, by the same tolerance as the answer line."""
    i, no, vals, _score = data.bests[r]
    s_i, s_no, s_round, s_vals = data.start
    rated = on_target_help_text(opt, vals)
    if i == s_i:
        return "\n\n".join(t for t in (rated, _say(opt, wording.spider_same_help(r, s_no))) if t)
    items = []
    for sp in data.spokes:
        row = dict(sp, start=s_vals.get(sp["name"]), best=vals.get(sp["name"]))
        m = target_move(row, ideal_tol(opt, sp["name"]))
        if m is not None:
            items.append(wording.spider_move(sp["name"], m[0], _move_text(
                m[1], sp["high"] - sp["low"])))
    if not items:
        return rated or None
    return "\n\n".join(t for t in (rated, _short(opt, wording.spider_moves_help(s_no, no, items)))
                       if t)


def spider_frames(opt, data, r):
    """The spider's rows at Round r, one list per layer (pure: what the
    chart draws, for the chart and its tests). Every polygon closes on its
    first vertex (order j)."""
    n = len(data.spokes)
    i, no, vals, _score = data.bests[r]
    s_i, s_no, s_round, s_vals = data.start
    target_name = _say(opt, wording.SPIDER_KEY_TARGET)
    best_name = _short(opt, wording.spider_key_best(r, no, _round_of(opt, i)))
    start_name = _short(opt, wording.spider_key_start(s_no))
    earlier_name = _say(opt, wording.SPIDER_KEY_EARLIER)
    rlabel = wording.round_tick

    def polygon(what, series, values, number, round_no, closed=True):
        pts = []
        for k, sp in enumerate(data.spokes):
            v = values.get(sp["name"])
            if v is None:
                continue
            x, y = spider_xy(k, n, spider_frac(sp, v))
            pts.append({"what": what, "series": series, "x": x, "y": y, "k": k,
                        "name": sp["name"], "value": _spider_num(sp, v), "raw": float(v),
                        "no": number, "round": rlabel(round_no) if round_no else ""})
        if closed and len(pts) > 2:
            pts.append(dict(pts[0]))
        for j, p in enumerate(pts):
            p["j"] = j
        return pts

    out = {"names": {"target": target_name, "best": best_name, "start": start_name,
                     "earlier": earlier_name}}
    # Earlier Rounds: each distinct best between the Start and Round r, once.
    seen, earlier = {s_i, i}, []
    for q in data.rounds:
        if q <= s_round or q >= r:
            continue
        qi, qno, qvals, _ = data.bests[q]
        if qi in seen:
            continue
        seen.add(qi)
        earlier += polygon(earlier_name, f"e{qno}", qvals, qno, q)
    out["earlier"] = earlier
    out["best"] = polygon(best_name, "best", vals, no, r)
    out["start"] = polygon(start_name, "start", s_vals, s_no, s_round)
    tvals = {sp["name"]: sp["target"] for sp in data.spokes if sp["target"] is not None}
    tpts = polygon(target_name, "target", tvals, None, None, closed=len(tvals) > 2)
    for p in tpts:
        p["no"] = ""
    out["target"] = tpts
    out["diamonds"] = [p for p in tpts if p["j"] < len(tvals)]
    arrows, bands = [], []
    for k, sp in enumerate(data.spokes):
        bearing = 360.0 * k / n
        if sp["band"] is not None:
            (x0, y0), (x1, y1) = (spider_xy(k, n, spider_frac(sp, v)) for v in sp["band"])
            bands.append({"x": x0, "y": y0, "x2": x1, "y2": y1, "what": target_name,
                          "name": sp["name"],
                          "value": f"{_spider_num(sp, sp['band'][0])}–{_spider_num(sp, sp['band'][1])}"})
        elif sp["target"] is None and sp["goal"] in ("max", "min"):
            up = sp["goal"] == "max"
            x, y = spider_xy(k, n, 1.0 if up else 0.07)
            arrows.append({"x": x, "y": y, "angle": bearing if up else (bearing + 180) % 360,
                           "name": sp["name"]})
    out["arrows"], out["bands"] = arrows, bands
    # Best so far's value at each vertex: past it, away from the centre, or
    # inside it when the target lies beyond it or the rim is near — never
    # on the target's diamond.
    labels = []
    for p in out["best"][:-1] if len(out["best"]) > 2 else out["best"]:
        sp = data.spokes[p["k"]]
        frac = spider_frac(sp, p["raw"])
        short_of = sp["target"] is not None and p["raw"] < sp["target"] - 1e-9
        inward = short_of or frac > _SPIDER_INWARD_FROM
        at = frac - _SPIDER_NUDGE if inward else frac + _SPIDER_NUDGE
        x, y = spider_xy(p["k"], n, max(at, 0.0))
        dx, dy = spider_xy(p["k"], n, -1.0 if inward else 1.0)
        align, baseline = _spider_anchor(dx, dy)
        labels.append({"x": x, "y": y, "text": p["value"], "align": align,
                       "baseline": baseline})
    out["labels"] = labels
    spokes = []
    for k, sp in enumerate(data.spokes):
        x, y = spider_xy(k, n, _SPIDER_LABEL_AT)
        align, baseline = _spider_anchor(*spider_xy(k, n, 1.0))
        spokes.append({"x": x, "y": y, "x0": 0.0, "y0": 0.0,
                       "x1": spider_xy(k, n, 1.0)[0], "y1": spider_xy(k, n, 1.0)[1],
                       "text": wording.story_target_row(sp["name"], sp["goal"], sp["target"]),
                       "align": align, "baseline": baseline})
    out["spokes"] = spokes
    rings, ring_labels = [], []
    if data.shared is not None:
        lo, hi = data.shared
        values = spider_ring_values(lo, hi)
        fracs = [(v - lo) / (hi - lo) for v in values]
        # Labelled once, between the last spoke and the first: clear of
        # every spoke's own labels.
        a = spider_angle(-0.5, n)
        for v, f in zip(values, fracs):
            ring_labels.append({"x": f * math.cos(a), "y": f * math.sin(a), "text": f"{v:g}"})
    else:
        fracs = [0.25, 0.5, 0.75, 1.0]
    for f in fracs:
        for j in range(_SPIDER_RING_SEGMENTS + 1):
            t = 2 * math.pi * j / _SPIDER_RING_SEGMENTS
            rings.append({"ring": f, "j": j, "x": f * math.cos(t), "y": f * math.sin(t)})
    out["rings"], out["ring_labels"] = rings, ring_labels
    return out


def spider_chart(opt, data, r, size=SPIDER_SIZE, wrap=False):
    """Sensory profile against target at Round r (spider_frames' layers):
    a square `size` plot, the legend under it. `wrap`: a side spoke's
    label on two lines (name, then target), so the chart stays narrow
    beside What changed."""
    f = spider_frames(opt, data, r)
    names = f["names"]
    ink, _ = mark("bar_latest")
    grey, _ = mark("neutral")
    light, _ = _TRIED_GREYS[theme()]
    blue = model_blue()
    kinds = [names["target"], names["best"], names["start"]] + (
        [names["earlier"]] if f["earlier"] else [])
    colour = alt.Scale(domain=kinds, range=[ink, blue, grey, _SPIDER_EARLIER[theme()]][:len(kinds)])
    dash = alt.Scale(domain=kinds, range=[[1, 0], [1, 0], [6, 4], [1, 0]][:len(kinds)])
    legend = alt.Legend(title=None, orient="bottom", labelLimit=0, symbolType="stroke",
                        symbolStrokeWidth=2.5, symbolSize=260, columns=2, columnPadding=24)
    domain = [-_SPIDER_PAD, _SPIDER_PAD]
    xs = alt.Scale(domain=domain, nice=False, zero=False)
    ys = alt.Scale(domain=domain, nice=False, zero=False)
    x, y = alt.X("x:Q", scale=xs, axis=None), alt.Y("y:Q", scale=ys, axis=None)
    word = _say(opt, wording.FORMULATION_CAP)

    def line(rows, **kw):
        return (alt.Chart(pd.DataFrame(rows)).mark_line(interpolate="linear", **kw)
                .encode(x=x, y=y, order="j:Q", detail="series:N",
                        color=alt.Color("what:N", scale=colour, legend=legend),
                        strokeDash=alt.StrokeDash("what:N", scale=dash, legend=legend)))

    layers = [
        alt.Chart(pd.DataFrame(f["rings"])).mark_line(color=light, strokeWidth=1, tooltip=False)
        .encode(x=x, y=y, order="j:Q", detail="ring:N"),
        alt.Chart(pd.DataFrame(f["spokes"])).mark_rule(color=light, strokeWidth=1, tooltip=False)
        .encode(x=alt.X("x0:Q", scale=xs, axis=None),
                y=alt.Y("y0:Q", scale=ys, axis=None), x2="x1:Q", y2="y1:Q")]
    if f["earlier"]:
        layers.append(line(f["earlier"], strokeWidth=1.3, tooltip=False))
    layers.append(line(f["best"], strokeWidth=2.5, fill=blue, fillOpacity=0.25,
                       strokeJoin="round", tooltip=False))
    layers.append(line(f["start"], strokeWidth=2, tooltip=False))
    if len(f["target"]) > 1:
        layers.append(line(f["target"], strokeWidth=3, strokeJoin="round", tooltip=False))
    if f["bands"]:
        layers.append(alt.Chart(pd.DataFrame(f["bands"])).mark_rule(
            color=ink, strokeWidth=8, opacity=0.5, strokeCap="round")
            .encode(x=x, y=y, x2="x2:Q", y2="y2:Q",
                    tooltip=[alt.Tooltip("name:N", title=" "),
                             alt.Tooltip("value:N", title=names["target"])]))
    if f["diamonds"]:
        layers.append(alt.Chart(pd.DataFrame(f["diamonds"])).mark_point(
            shape="diamond", filled=True, size=80, color=ink, opacity=1)
            .encode(x=x, y=y, tooltip=[alt.Tooltip("name:N", title=" "),
                                       alt.Tooltip("value:N", title=names["target"])]))
    if f["arrows"]:
        layers.append(alt.Chart(pd.DataFrame(f["arrows"])).mark_point(
            shape="triangle-up", filled=True, size=120, color=ink, opacity=1, tooltip=False)
            .encode(x=x, y=y, angle=alt.Angle("angle:Q", scale=None)))
    if f["ring_labels"]:
        # Over the outlines, on a halo: the Range's numbers stay readable.
        layers += _halo_text(pd.DataFrame(f["ring_labels"]), x, y, "text:N", muted(),
                             fontSize=FONT_SMALL, baseline="middle", align="center")
    point_tips = [alt.Tooltip("what:N", title=" "), alt.Tooltip("name:N", title=" "),
                  alt.Tooltip("value:N", title=wording.STORY_VALUE),
                  alt.Tooltip("no:N", title=word), alt.Tooltip("round:N", title=_say(
                      opt, wording.ROUND_CAP))]
    start_pts = [p for p in f["start"] if p["j"] < len(f["start"]) - 1]
    layers.append(alt.Chart(pd.DataFrame(start_pts)).mark_point(
        filled=False, size=34, strokeWidth=1.5, color=grey, opacity=1)
        .encode(x=x, y=y, tooltip=point_tips))
    best_pts = [p for p in f["best"] if p["j"] < len(f["best"]) - 1]
    layers.append(alt.Chart(pd.DataFrame(best_pts)).mark_point(
        filled=True, size=40, color=blue, opacity=1).encode(x=x, y=y, tooltip=point_tips))
    lab = pd.DataFrame(f["labels"])
    for (align, baseline), part in lab.groupby(["align", "baseline"]):
        layers += _halo_text(part, x, y, "text:N", blue, fontSize=FONT_LABEL,
                             fontWeight="bold", align=align, baseline=baseline)
    sp = pd.DataFrame(f["spokes"])
    if wrap:
        sp["text"] = [t.replace(SPOKE_SEP, "\n", 1) if a != "center" else t
                      for t, a in zip(sp["text"], sp["align"])]
    for (align, baseline), part in sp.groupby(["align", "baseline"]):
        layers.append(alt.Chart(part).mark_text(color=ink, fontSize=FONT_LABEL, align=align,
                                                baseline=baseline, tooltip=False,
                                                lineBreak="\n")
                      .encode(x=x, y=y, text="text:N"))
    chart = alt.layer(*layers).properties(width=size, height=size)
    return readable(chart.properties(autosize=alt.AutoSizeParams(type="pad"))) \
        .configure_view(strokeWidth=0)


def _spider_block(opt, data, r):
    """Sensory profile against target at Round `r`, in the Replay panel:
    the title, the answer line and the chart — a PANEL_SPIDER square, its
    side labels on two lines. Its caption and the score recipe sit under
    the panel (_panel_captions)."""
    st.markdown(f"**{_say(opt, wording.SPIDER_TITLE)}**")
    with st.container(key="insights_spider", gap="small"):
        st.markdown(spider_round_answer(opt, data, r), help=spider_round_help(opt, data, r))
        try:
            spec = chart_spec(("spider", spider_key(opt), theme(), r,
                               _say(opt, wording.ROUND_CAP), PANEL_SPIDER),
                              lambda: spider_chart(opt, data, r, size=PANEL_SPIDER, wrap=True))
        except Exception:
            spec = None
        if spec is None:
            st.caption(_say(opt, wording.CHART_FAILED))
        else:
            show_spec(spec, width="content", clear=True)


# --- What changed in the formulation (review, 2026-10-08) ---
# Start (story_start: the first scored Round's best, as Start vs best) against
# Best so far, one row per varied ingredient or process setting. Model-free:
# the recorded make-ups only, at the Formulation total (insights.start_recipe).

_CHANGE_WORDS = 12        # the answer line, at most
_LIMIT_WORDS = 16         # the limits line, not counting the amounts in brackets
_PERCENT_FROM = 0.10      # a % only from a Start of this share of the row's range


@dataclasses.dataclass
class ChangeRow:
    name: str
    start: str
    best: str
    change: str
    percent: str            # '(+62 %)', '' with none
    share: float            # |change| as a share of Highest − Lowest (a choice: 0 or 1)
    item: str               # the answer line's words: 'Coconut cream −6.4 g'
    setting: bool = False
    changed: bool = True    # False when the change reads as nothing
    near_zero: bool = False  # no %: Start under _PERCENT_FROM of the range


@dataclasses.dataclass
class Changes:
    start_no: int
    best_no: int
    start_round: int
    rows: list              # [ChangeRow], largest share first
    same: bool              # Best so far is the start
    fill: str | None        # the Fill to total row, left out
    fixed: bool             # a fixed row was left out
    limits: str | None      # the limits line, None when Best so far sits at none
    limits_help: str | None
    round_no: int | None = None   # the Round on view (Best so far then); None: today


def limit_words(text):
    """Words in a line, the amounts in brackets not counted: '(42 g)' is
    the number the name already announces."""
    plain = re.sub(r"\([^)]*\)", "", str(text)).replace("**", "")
    return len([w for w in plain.split() if re.search(r"\w", w)])


def _change_row(opt, var, a, b):
    """One row: Start, Best so far and the change in the row's own unit and
    the Review's decimal rule (_shared_decimals over the pair; a setting by
    fmt_setting). The change is the shown Best minus the shown Start, so
    the row's arithmetic reads right (57.6 → 51.3 is −6.3). The % against
    Start in grey only from a Start of _PERCENT_FROM of the row's range: a
    near-zero Start gives a % nobody can use."""
    name = var['name']
    if var.get('type', 'continuous') != 'continuous':
        moved = a != b
        return ChangeRow(name, str(a), str(b),
                         wording.CHANGES_CHOICE_CHANGED if moved else wording.CHANGES_CHOICE_SAME,
                         "", 1.0 if moved else 0.0,
                         wording.changes_item(name, wording.changes_choice(a, b)),
                         setting=not insights._is_ingredient(var), changed=moved)
    a, b = float(a), float(b)
    delta = b - a
    unit = opt.unit_of(name)
    setting = not insights._is_ingredient(var)
    lo, hi = (float(x) for x in var['bounds'])
    share = abs(delta) / (hi - lo) if hi > lo else 0.0
    near_zero = False
    if setting:
        start, best = fmt_setting(a, unit), fmt_setting(b, unit)
        shown = round(round(b, 2) - round(a, 2), 2)
        size = fmt_setting(abs(shown), "")
        changed = size != "0"
        change = join_unit(("+" if shown > 0 else "−") + size if changed else "0", unit)
        percent = ""
    else:
        d = _shared_decimals([a, b])
        a_shown, b_shown = float(f"{a:.{d}f}"), float(f"{b:.{d}f}")
        start, best = join_unit(f"{a_shown:.{d}f}", unit), join_unit(f"{b_shown:.{d}f}", unit)
        shown = round(b_shown - a_shown, d)
        signed = wording.signed(shown, d)
        changed = float(signed.replace("−", "-")) != 0
        change = join_unit(signed, unit)
        percent = ""
        if changed:
            if a >= _PERCENT_FROM * (hi - lo) and a_shown != 0:
                percent = f"({wording.signed(100.0 * shown / a_shown, 0)} %)"
            else:
                near_zero = True
    return ChangeRow(name, start, best, change, percent, share,
                     wording.changes_item(name, change), setting=setting, changed=changed,
                     near_zero=near_zero)


def _limit_amount(opt, name, value):
    """A row's limit as binding_texts writes it: '42 g', '2.6 g', '180 °C'."""
    unit = opt.unit_of(name)
    if insights.is_setting(opt, name):
        return fmt_setting(value, unit)
    return join_unit(f"{value:.0f}" if value == round(value) else _num(value), unit)


def _made_recipes(opt, upto=None):
    """(formulation number, make-up at the Formulation total) of every
    formulation recorded, scored or not, and of the open Round's rows.
    `upto`: only those recorded by Round `upto` (a formulation of a later
    Round, and the open Round's rows, left out; one with no Round kept, so
    'nothing tested beyond' is never said of it)."""
    out = []
    for i in range(len(opt.X_history)):
        r = _round_of(opt, i)
        if upto is not None and r is not None and r > upto:
            continue
        no = int(opt.formulation_ids[i]) if i < len(opt.formulation_ids) else None
        out.append((no, insights.start_recipe(opt, i)))
    if upto is not None:
        return out
    for row in getattr(opt, 'pending_batch', None) or []:
        try:
            out.append((int(row['formulation']), insights.at_total(opt, row['recipe'])))
        except Exception:
            continue
    return out


def _qc_total(qc, recipe):
    return sum(float(recipe.get(n, 0.0) or 0.0) for n in qc['ingredients'])


def best_limits(opt, recipe, upto=None):
    """[(item, beyond numbers)] for every limit `recipe` sits at — the
    app's own rule, insights.binding_limits (BINDING) — each with the
    recorded formulations that lie beyond it (by Round `upto` when given).
    Rows first ('Highest', then 'Lowest'), then the limits on several
    rows."""
    made = _made_recipes(opt, upto)
    ends = {"highest": [], "lowest": []}
    others = []
    bindings = insights.binding_limits(opt, recipe)
    for b in bindings:
        if b.kind in ("highest", "lowest"):
            lo, hi = (float(x) for x in insights._var(opt, b.name)['bounds'])
            tol = 1e-6 * max(hi - lo, 1e-9)
            value = hi if b.kind == "highest" else lo
            beyond = [no for no, r in made if r.get(b.name) is not None
                      and not isinstance(r.get(b.name), str)
                      and (float(r[b.name]) > hi + tol if b.kind == "highest"
                           else float(r[b.name]) < lo - tol)]
            ends[b.kind].append((b.name, _limit_amount(opt, b.name, value), beyond))
    for qc in getattr(opt, 'quantity_constraints', []):
        if qc.get('source') == 'formulation_total' or qc.get('exactly') is not None:
            continue
        total = _qc_total(qc, recipe)
        for side, bound in (("min", qc.get('min')), ("max", qc.get('max'))):
            if bound is None:
                continue
            bound = float(bound)
            if abs(total - bound) > insights.BINDING * max(abs(bound), insights._EPS):
                continue
            tol = 1e-6 * max(abs(bound), 1.0)
            beyond = [no for no, r in made
                      if (_qc_total(qc, insights._numeric(r)) > bound + tol if side == "max"
                          else _qc_total(qc, insights._numeric(r)) < bound - tol)]
            one = dict(qc, min=bound if side == "min" else None,
                       max=bound if side == "max" else None)
            text = binding_texts(opt, [insights.Binding("limit", "", opt.limit_text(one))])[0]
            others.append((text, beyond))
    amount = max(opt.ingredient_total(recipe), insights._EPS)
    numeric = insights._numeric(recipe)
    for c in getattr(opt, 'constraints', []):
        if opt.ingredients_without_property(c['metric']):
            continue
        for side, bound in (("min", c.get('min')), ("max", c.get('max'))):
            if bound is None:
                continue
            if abs(opt._property_residual(numeric, c['metric'], bound)) > \
                    insights.BINDING * max(abs(float(bound)), insights._EPS) * amount:
                continue
            beyond = []
            for no, r in made:
                res = opt._property_residual(insights._numeric(r), c['metric'], bound)
                tol = 1e-6 * max(opt.ingredient_total(insights._numeric(r)), 1.0)
                if (res > tol) if side == "max" else (res < -tol):
                    beyond.append(no)
            one = dict(c, min=bound if side == "min" else None,
                       max=bound if side == "max" else None)
            text = binding_texts(opt, [insights.Binding(
                "limit", c['metric'], opt.property_limit_text(one))])[0]
            others.append((text, beyond))
    return ends, others


def limits_line(opt, recipe, upto=None):
    """(the limits line, its "?") — None, None when Best so far sits at no
    limit. The line names the limits on one row (an ingredient or a
    setting); a limit on several rows (the Batch total) is named on the
    line only when Best so far sits at no row's limit, and always in the
    "?" with what lies beyond it. 'nothing tested beyond' only of the
    limits no recorded formulation lies beyond (limits can change after a
    formulation is made). `upto`: judged on the formulations recorded by
    that Round (the section's Round control)."""
    ends, others = best_limits(opt, recipe, upto)
    groups = []         # (text, every item's beyond list, [(item, beyond)])
    for kind, end in (("highest", "Highest"), ("lowest", "Lowest")):
        rows = ends[kind]
        for clear in (True, False):
            part = [r for r in rows if (not r[2]) == clear]
            if part:
                groups.append((wording.changes_end_group(
                    end, [f"{n} ({a})" for n, a, _ in part]), clear,
                    [(f"{n}'s {end} ({a})", bey) for n, a, bey in part]))
    multi = []
    for text, bey in others:
        m = re.match(r"^(.*) (\(.*\))$", text)
        said = wording.changes_multi_item(m.group(1), m.group(2)) if m else text
        multi.append((said, not bey, [(text, bey)]))
    if not groups and not multi:
        return None, None
    shown = groups or multi
    untested = [g[0] for g in shown if g[1]]
    tested = [g[0] for g in shown if not g[1]]
    line = wording.changes_limits(untested, tested)
    items = [it for g in groups + multi for it in g[2]]
    if limit_words(line) > _LIMIT_WORDS:
        on_line = [it for g in shown for it in g[2]]
        line = wording.changes_limits_count(len(on_line),
                                            sum(1 for _, b in on_line if not b))
    beyond = [wording.changes_beyond(number_list(sorted(set(b))), len(set(b)) > 1, it)
              for it, b in items if b]
    return (_say(opt, line),
            _say(opt, wording.brief(wording.changes_limits_help([i for i, _ in items], beyond))))


def what_changed(opt, at_round=None):
    """Changes from Start to Best so far, or None with no start, no best or
    no varied row. `at_round`: Best so far by the end of that Round (the
    section's Round control), its limits line judged on the formulations recorded
    by then; the latest Round is today's Best so far, judged on everything
    recorded."""
    s, b = story_start(opt), opt.best_index()
    upto = None
    if at_round is not None:
        rounds, at = round_bests(opt)
        if at_round in at:
            b = at[at_round]
            upto = at_round if at_round != rounds[-1] else None
    if s is None or b is None or b >= len(opt.X_history):
        return None
    i, r0 = s
    start_no, best_no = int(opt.formulation_ids[i]), int(opt.formulation_ids[b])
    fill = opt.balance_row_name()
    fixed = bool(opt.fixed_variables())
    if i == b:
        return Changes(start_no, best_no, r0, [], True, fill, fixed, None, None, at_round)
    a, z = insights.start_recipe(opt, i), insights.start_recipe(opt, b)
    rows = []
    for at, var in enumerate(opt.varying_variables()):
        name = var['name']
        if name == fill:
            continue
        absent = var.get('_absent_value', 0.0)
        row = _change_row(opt, var, a.get(name, absent), z.get(name, absent))
        rows.append(((-row.share, row.setting, at), row))
    if not rows:
        return None
    rows = [r for _, r in sorted(rows, key=lambda t: t[0])]
    limits, help_ = limits_line(opt, z, upto)
    return Changes(start_no, best_no, r0, rows, False, fill, fixed, limits, help_, at_round)


def changes_answer(opt, ch):
    """The answer line: the two largest changes by share of range (one
    when two run past _CHANGE_WORDS)."""
    if ch.same:
        return _say(opt, wording.changes_same(ch.start_round))
    moved = [r.item for r in ch.rows if r.changed]
    line = wording.changes_answer(moved[:2])
    if limit_words(line) > _CHANGE_WORDS and len(moved) > 1:
        line = wording.changes_answer(moved[:1])
    return _say(opt, line)


def _made_round(opt, no):
    """The Round Formulation `no` was made in, or None."""
    for i, f in enumerate(opt.formulation_ids):
        if int(f) == int(no):
            return _round_of(opt, i)
    return None


def _md_cell(text):
    return re.sub(r"([\\|*_`\[\]<>])", r"\\\1", str(text))


def _keep_together(text):
    """'Best so far (F21, Round 10)' wrapping only before its bracket: the
    bracket's spaces made non-breaking (a narrow column's header)."""
    return re.sub(r"\(([^)]*)\)", lambda m: "(" + m.group(1).replace(" ", "\u00a0") + ")", text)


def changes_table(opt, ch, compact=False):
    """The table as Markdown: four columns, the numbers right-aligned, the
    % in grey after the change (`compact`: a header wraps only before its
    bracket)."""
    kinds = {r.setting for r in ch.rows}
    head = (wording.CHANGES_NAME_BOTH if len(kinds) > 1 else
            wording.CHANGES_NAME_SETTING if kinds == {True} else wording.CHANGES_NAME_INGREDIENT)
    cols = [_say(opt, head), _short(opt, wording.changes_start_column(ch.start_no)),
            _short(opt, wording.changes_best_column(ch.best_no, ch.round_no,
                                                    _made_round(opt, ch.best_no))),
            wording.CHANGES_CHANGE]
    if compact:
        cols = [_keep_together(c) for c in cols]
    lines = ["| " + " | ".join(_md_cell(c) for c in cols) + " |",
             "| :-- | --: | --: | --: |"]
    for r in ch.rows:
        change = _md_cell(r.change) + (f" :gray[{_md_cell(r.percent)}]" if r.percent else "")
        lines.append(f"| {_md_cell(r.name)} | {_md_cell(r.start)} | {_md_cell(r.best)} | {change} |")
    return "\n".join(lines)


def _changes_block(opt, r=None):
    """'What changed by Round 4' in the Replay panel, under the score
    chart: the answer, the table (compact), the limits line — at the
    panel's Round `r` (None: today). Always as many elements, whatever the
    Round (_hold). Its caption sits under the panel."""
    try:
        ch = what_changed(opt, r)
    except Exception:
        ch = None
    if ch is None:
        return
    # Final demo (B2): the block keeps one height at every Round (a Round
    # with no table reserves the table's room), so a step never moves the
    # panel: the title, the answer, the table and a two-line limits line.
    st.html(f"<style>.st-key-insights_changes {{ min-height: {changes_height(opt)}px; }}</style>")
    with st.container(key="insights_changes"):
        st.markdown(f"**{_say(opt, wording.changes_title(ch.round_no))}**")
        st.markdown(changes_answer(opt, ch))
        if ch.same:
            _hold(2)
            return
        st.markdown(changes_table(opt, ch, compact=True))
        if ch.limits:
            st.markdown(ch.limits, help=ch.limits_help)
        else:
            _hold(1)


CHANGES_BASE_H = 148          # px: What changed's title, answer, header and a 2-line limits line
CHANGES_ROW_H = 21            # px per table row (0.82rem, 1152-1280 px)


def changes_height(opt):
    """What changed's reserved height: its rows (the varying rows but the
    one filling to the total) at CHANGES_ROW_H, plus CHANGES_BASE_H."""
    fill = opt.balance_row_name()
    n = sum(1 for v in opt.varying_variables() if v['name'] != fill)
    return CHANGES_BASE_H + CHANGES_ROW_H * n


def _frange(a, b, step):
    """a, a + step, … up to b (inclusive, to a millionth)."""
    out, v = [], a
    while v <= b + 1e-6:
        out.append(v)
        v += step
    return out


def _esc(text):
    return str(text).replace("\\", "\\\\").replace("'", "\\'")


def forecast_chart(opt, checks):
    """Can we trust the predictions? — forecast against actual, over time:
    for each checked formulation, the 95 % range the model predicted for
    its Overall score before it was made (a light blue bar, from the
    earlier Rounds only) and the score measured (a dot: filled ink inside,
    hollow ink outside). x: Formulations in the order made, as the
    improving chart. None without a check."""
    mine = [c for c in checks or () if c.name == insights.SCORE]
    if not mine:
        return None
    formulation = _say(opt, wording.FORMULATION_CAP)
    inside_name = _say(opt, wording.STORY_LANDED_IN)
    outside_name = _say(opt, wording.STORY_LANDED_OUT)
    none_name = _say(opt, wording.NO_FORECAST)
    band_name = wording.STORY_FORECAST
    blue = model_blue()
    band_c, _ = _BAND_FILL[theme()]
    ink, _ = mark("bar_latest")
    df = pd.DataFrame([{
        formulation: int(c.formulation), "round": wording.round_tick(c.round),
        "predicted": round(c.predicted, 1), "low": round(c.low, 1), "high": round(c.high, 1),
        "measured": round(c.measured, 1),
        "state": (none_name if no_forecast(opt, c) else
                  inside_name if c.inside else outside_name), "band": band_name,
        "bo": wording.PLOT_BO_ROUND_AHEAD} for c in mine])
    vals = [v for c in mine for v in (c.low, c.high, c.measured)]
    lo, hi = min(vals), max(vals)
    pad = 0.05 * max(hi - lo, 4.0)
    y_scale = alt.Scale(domain=[lo - pad, hi + pad], nice=False, zero=False)
    x_scale = alt.Scale(domain=_x_domain(opt), nice=False)
    numbers = [int(n) for n in opt.formulation_ids]
    x = alt.X(f"{formulation}:Q", title=_say(opt, wording.Q2_X_TITLE), scale=x_scale,
              axis=alt.Axis(format="d", values=formulation_ticks(numbers), grid=False))
    y_title = zoomed(wording.PLOT_MEASURED_SCORE, lo, hi, float(opt.utility_ceiling()))
    y = alt.Y("measured:Q", title=y_title, scale=y_scale)
    tip = [alt.Tooltip(f"{formulation}:Q", title=formulation, format="d"),
           alt.Tooltip("round:N", title=_say(opt, wording.ROUND_CAP)),
           alt.Tooltip("predicted:Q", title=wording.TRACK_PREDICTED, format=".1f"),
           alt.Tooltip("low:Q", title=wording.TRACK_LOW, format=".1f"),
           alt.Tooltip("high:Q", title=wording.TRACK_HIGH, format=".1f"),
           alt.Tooltip("measured:Q", title=wording.TRACK_MEASURED, format=".1f"), bo_tip()]
    legend = alt.Legend(title=None, orient="top", labelLimit=0)
    bars = alt.Chart(df).mark_rule(strokeWidth=7, opacity=0.45, strokeCap="round").encode(
        x=x, y=alt.Y("low:Q", scale=y_scale, title=y_title), y2="high:Q",
        color=alt.Color("band:N", scale=alt.Scale(domain=[band_name], range=[blue]),
                        legend=legend), tooltip=tip)
    # Round-2 fixes (P3 #18): a miss in orange, as on the Forecast check.
    # Fix round 2: a whole-Range range is "No forecast" (neutral), keyed only
    # when one is drawn.
    nf = any(no_forecast(opt, c) for c in mine)
    fills = alt.Scale(domain=[inside_name, outside_name] + ([none_name] if nf else []),
                      range=[ink, outline_colour()] + ([mark("neutral")[0]] if nf else []))
    dots = alt.Chart(df).mark_point(size=46, strokeWidth=1.6, filled=True, opacity=1).encode(
        x=x, y=y, color=alt.Color("state:N", scale=fills, legend=legend), tooltip=tip)
    # The predicted value itself: a short blue tick across its bar.
    ticks = alt.Chart(df).mark_tick(color=blue, thickness=2, size=11, opacity=1, orient="horizontal").encode(
        x=x, y=alt.Y("predicted:Q", scale=y_scale, title=y_title), tooltip=tip)
    layers = [bars, ticks, dots]
    first_checked = min(int(c.formulation) for c in mine)
    before = [n for n in numbers if n < first_checked]
    if len(before) >= 3:
        note = pd.DataFrame([{"x": (min(before) + max(before)) / 2, "y": (lo + hi) / 2,
                              "text": wording.STORY_NO_FORECAST}])
        layers.append(alt.Chart(note).mark_text(fontSize=FONT_SMALL, color=muted(), tooltip=False)
                      .encode(x=alt.X("x:Q", scale=x_scale), y=alt.Y("y:Q", scale=y_scale),
                              text="text:N"))
    rounds = _round_layer(opt, formulation, numbers, x_scale)
    body = alt.layer(*layers)
    chart = alt.layer(rounds, body).resolve_axis(x="independent") if rounds else body
    return readable(chart.properties(height=STORY_Q_HEIGHT - 30))


def trust_counts(checks):
    """(inside, total, above, below) of the Overall score's Round-ahead
    checks: misses above and below their range counted apart."""
    mine = [c for c in insights.ranged(checks) if c.name == insights.SCORE]
    inside = sum(bool(c.inside) for c in mine)
    above = sum(1 for c in mine if not c.inside and c.measured > c.high)
    below = sum(1 for c in mine if not c.inside and c.measured < c.low)
    return inside, len(mine), above, below


def driver_finding(opt, profile, guard):
    """Q3's answer, a finding: '**Methylcellulose matters most.** Best so
    far uses its Highest.' — the column that moves the predicted Overall score
    most near tested formulations (the profiler's own ranking), and where
    the reference sits. Review story 3: from tested stretches only
    (near_effect); with no clear leader there (clear_leader), under the
    noise guard or with no effect, '**No single ingredient clearly leads
    yet.**'."""
    if (profile is None or guard is None or guard.fires or not profile.columns
            or not clear_leader(profile, leader_noise(opt))):
        kind = ("ingredients" if opt is None or not getattr(profile, "columns", None)
                else _more_kind(opt, [c.name for c in profile.columns]))
        return wording.answer_line(wording.story_no_leader(kind), "")
    lead = profile.columns[0]
    rest = ""
    if lead.end:
        rest = wording.story_driver_end(profile.ref_label, lead.end)
    elif len(profile.columns) > 1 and profile.columns[1].near_effect > 0:
        rest = wording.story_driver_then(profile.columns[1].name)
    return wording.answer_line(wording.story_driver(lead.name), rest)


def leader_noise(opt):
    """The Overall score's noise SD a lead must beat (noise_yardstick's:
    Repeats' from 3 df, else the assumed 1 point)."""
    if opt is None:
        return insights.ASSUMED_NOISE_SD
    try:
        value = insights.noise_yardstick(opt).value
    except Exception:
        value = None
    return insights.ASSUMED_NOISE_SD if value is None else float(value)


def clear_leader(profile, noise):
    """Review story 3: whether the first column clearly leads over its
    tested stretch: it moves the predicted Overall score by at least
    `noise`, and the next column by under NEAR_TIE of that (the trust
    audit's near tie: the named top was often wrong within 20 %)."""
    cols = profile.columns
    lead = cols[0].near_effect
    second = cols[1].near_effect if len(cols) > 1 else 0.0
    return lead >= noise - 1e-12 and second < insights.NEAR_TIE * lead


def path_frame(opt, axes, lines):
    """The search path's long frame: one row per formulation per axis —
    its amount, its Round's row, its role — for every tested, waiting and
    Not scored formulation and the open Round's suggestions."""
    rows = []
    # Honesty guard: only the rows the app generated are suggestions; the
    # team's own formulations in the open Round (a note, no generated
    # make-up) wait for results like any other.
    generated = set(suggested_numbers(opt))
    for line in lines:
        r = line.round if line.round is not None else (
            opt.pending_batch_no if line.state == "suggested" else None)
        if r is None:
            continue
        suggested = line.state == "suggested" and line.formulation in generated
        role = (wording.TRIED_SUGGESTED if suggested
                else wording.TRIED_WAITING if line.state != "tested"
                else wording.TRIED_BEST if line.best else wording.STORY_PATH_TESTED)
        for a in axes:
            v = line.values.get(a.name)
            if v is None:
                continue
            rows.append({"axis": a.name, "no": line.formulation, "round": int(r),
                         "rlabel": wording.round_tick(r), "value": v if a.options is not None
                         else float(v), "role": role,
                         "score": "" if line.score is None else f"{line.score:.1f}",
                         "text": (str(v) if a.options is not None
                                  else _axis_end(opt, a, float(v)))})
    return pd.DataFrame(rows, columns=["axis", "no", "round", "rlabel", "value", "role",
                                       "score", "text"])


def search_path_chart(opt, axes, lines, gaps=()):
    """What have we tried? — the search path: one small panel per
    ingredient or process setting; across, its allowed amounts (Lowest to
    Highest); down, the Rounds, the first at the top. One grey dot per
    formulation, a light bar from each Round's lowest to highest amount,
    Best so far ringed in ink, this Round's suggestions orange in their
    Round's row. Empty stretches were not tried; bars that shorten show the
    search settling. None without an axis or a formulation."""
    if not axes or not lines:
        return None
    df = path_frame(opt, axes, lines)
    if df.empty:
        return None
    rounds = sorted(set(df["round"]))
    labels = [wording.round_tick(r) for r in rounds]
    grey, _ = mark("neutral")
    ink, _ = mark("bar_latest")
    light, mid = _TRIED_GREYS[theme()]
    orange = outline_colour()
    order = [wording.STORY_PATH_TESTED, wording.TRIED_BEST, wording.TRIED_SUGGESTED,
             wording.TRIED_WAITING]
    kinds = [k for k in order if k in set(df["role"])]
    # Review critique (fix 8): an amount at its row's Lowest or Highest
    # marked (a square), and the named untested gap shaded.
    at_limit, gap_name = _say(opt, wording.PATH_AT_LIMIT), _say(opt, wording.PATH_GAP)
    spans = {a.name: (a.low, a.high) for a in axes if a.options is None}
    df["limit"] = [a in spans and (abs(v - spans[a][0]) <= insights.BINDING * (spans[a][1] - spans[a][0])
                                   or abs(v - spans[a][1]) <= insights.BINDING * (spans[a][1] - spans[a][0]))
                   if not isinstance(v, str) else False
                   for a, v in zip(df["axis"], df["value"])]
    gap_by = {}
    for g in gaps or ():
        gap_by.setdefault(g.name, []).append(g)
    look = {wording.STORY_PATH_TESTED: mid, wording.TRIED_BEST: ink,
            wording.TRIED_SUGGESTED: orange, wording.TRIED_WAITING: grey}
    # Round-2 fixes (P2 #12): each legend's symbols the size of the marks
    # they name; Waiting a hollow ring in the legend too (its own stroke
    # legend), the gap a pale square.
    filled_kinds = [k for k in kinds if k != wording.TRIED_WAITING]
    colour = alt.Scale(domain=filled_kinds, range=[look[k] for k in filled_kinds])
    waiting = alt.Scale(domain=[wording.TRIED_WAITING], range=[grey])
    any_waiting = wording.TRIED_WAITING in kinds
    legend = alt.Legend(title=None, orient="bottom", labelLimit=0, symbolSize=44)
    limit_legend = alt.Legend(title=None, orient="bottom", labelLimit=0, symbolSize=70,
                              symbolStrokeWidth=1.6)
    gap_legend = alt.Legend(title=None, orient="bottom", labelLimit=0, symbolType="square",
                            symbolSize=120, symbolOpacity=0.55, symbolStrokeWidth=0)
    dot_r = 5      # px: a dot's radius and a pixel, the inset of every gap
    word = _say(opt, wording.FORMULATION_CAP)
    height = max(80, _PATH_ROW_H * len(rounds))
    y_scale = alt.Scale(domain=labels, paddingInner=0.2, paddingOuter=0.3)
    panels = []
    # Review readable: a single panel or two still make MIN_CHART_W.
    shown_cols = min(len(axes), PATH_COLUMNS)
    width = max(_PATH_PANEL_W,
                (MIN_CHART_W - 60 - _PATH_SPACING * (shown_cols - 1)) // shown_cols)
    for k, a in enumerate(axes):
        mine = df[df["axis"] == a.name]
        first = k % PATH_COLUMNS == 0
        y = alt.Y("rlabel:O", scale=y_scale, sort=labels,
                  title=_say(opt, wording.ROUND_CAP) if first else None,
                  axis=alt.Axis(labels=first, ticks=False, domain=False, grid=False,
                                labelFontSize=FONT_SMALL, labelColor=muted(),
                                titleFontSize=FONT_AXIS_TITLE)
                  if first else None)
        if a.options is not None:
            x_scale = alt.Scale(domain=list(a.options))
            x_axis = alt.Axis(labelAngle=0, labelFontSize=FONT_SMALL, labelLimit=80, ticks=False,
                              domain=False)
            x = alt.X("value:N", scale=x_scale, title=None, axis=x_axis)
        else:
            # The ends with as many decimals as the gaps' own text (P2 #12).
            d = max(_shared_decimals([a.low, a.high]), _shared_decimals(
                [v for g in gap_by.get(a.name, ()) for v in (g.low, g.high)]))
            x_scale = alt.Scale(domain=[a.low, a.high], nice=False, zero=False, padding=9)
            x_axis = alt.Axis(values=[a.low, a.high], grid=False, ticks=False, domain=False,
                              labelFontSize=FONT_SMALL, labelColor=muted(),
                              labelExpr=_ends_expr(opt, a, d), labelFlush=True)
            x = alt.X("value:Q", scale=x_scale, title=None, axis=x_axis)
        tip = [alt.Tooltip("no:Q", title=word, format="d"),
               alt.Tooltip("rlabel:N", title=_say(opt, wording.ROUND_CAP)),
               alt.Tooltip("text:N", title=a.name),
               alt.Tooltip("score:N", title=_say(opt, wording.OVERALL_SCORE_COLUMN))]
        base = alt.Chart(mine)
        layers = []
        if a.options is None and a.name in gap_by:
            # Inset by a dot's radius each side, so a tested dot never sits
            # inside the shading and two gaps that meet stay apart.
            inset = dot_r * (a.high - a.low) / max(1.0, width - 18)
            gdf = pd.DataFrame([{"lo": g.low + inset, "hi": g.high - inset, "what": gap_name}
                                for g in gap_by[a.name] if g.high - g.low > 2 * inset])
            if not gdf.empty:
                layers.append(alt.Chart(gdf).mark_rect(opacity=0.55, tooltip=False).encode(
                    x=alt.X("lo:Q", scale=x_scale, title=None, axis=x_axis), x2="hi:Q",
                    fill=alt.Fill("what:N", scale=alt.Scale(domain=[gap_name], range=[light]),
                                  legend=gap_legend)))
        layers.append(base.mark_point(size=40, filled=True, opacity=0.95).encode(
            x=x, y=y, color=alt.Color("role:N", scale=colour, legend=legend), tooltip=tip)
            .transform_filter(f"datum.role != '{_esc(wording.TRIED_BEST)}' && "
                              f"datum.role != '{_esc(wording.TRIED_WAITING)}'"))
        # Waiting for results (or Not scored): a hollow grey ring.
        layers.append(base.mark_point(size=44, filled=False, strokeWidth=1.4).encode(
            x=x, y=y, stroke=alt.Stroke("role:N", scale=waiting,
                                        legend=legend if any_waiting else None), tooltip=tip)
            .transform_filter(f"datum.role == '{_esc(wording.TRIED_WAITING)}'"))
        layers.append(base.mark_point(size=130, strokeWidth=2, filled=False, color=ink)
                      .encode(x=x, y=y, tooltip=tip)
                      .transform_filter(f"datum.role == '{_esc(wording.TRIED_BEST)}'"))
        layers.append(base.mark_point(size=40, filled=True).encode(
            x=x, y=y, color=alt.Color("role:N", scale=colour, legend=legend), tooltip=tip)
            .transform_filter(f"datum.role == '{_esc(wording.TRIED_BEST)}'"))
        if a.options is None and mine["limit"].any():
            layers.append(base.mark_point(size=70, strokeWidth=1.6, filled=False, color=ink)
                          .encode(x=x, y=y, tooltip=tip,
                                  shape=alt.Shape("lim:N", scale=alt.Scale(
                                      domain=[at_limit], range=["square"]), legend=limit_legend))
                          .transform_filter("datum.limit")
                          .transform_calculate(lim=f"'{_esc(at_limit)}'"))
        title = alt.TitleParams(_axis_label(a), fontSize=FONT_LABEL, fontWeight="normal",
                                anchor="start", color=None, limit=width + 20, offset=6)
        panels.append(alt.layer(*layers).properties(width=width, height=height,
                                                     title=title))
    return readable(alt.concat(*panels, columns=PATH_COLUMNS, spacing=_PATH_SPACING)
                    .resolve_scale(color="shared", x="independent", shape="shared",
                                   fill="shared", stroke="shared")
                    .configure_view(strokeWidth=0))


def _ends_expr(opt, a, d):
    """The x axis' two labels: Lowest and Highest with the unit."""
    lo, hi = _axis_end(opt, a, a.low, d), _axis_end(opt, a, a.high, d)
    return (f"abs(datum.value - {a.low}) < 1e-9 ? '{_esc(lo)}' : "
            f"abs(datum.value - {a.high}) < 1e-9 ? '{_esc(hi)}' : ''")
