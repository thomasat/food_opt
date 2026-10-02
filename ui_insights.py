"""The Insights tab. Builders here return Altair charts from computed facts
and never call Streamlit; render() at the bottom is the only Streamlit code."""
import contextlib
import itertools
import math
import re
import threading

import altair as alt
import pandas as pd
import streamlit as st

import bo_views
import insights
import insights_cache
import insights_jobs
import wording
from food_bo import fmt_setting, join_unit, number_list
from ui_helpers import (SCROLL_KEY, SETUP_CALLOUT_KEY, SETUP_CALLOUT_SLOT,
                        TAB_BATCH, app_rerun, TAB_SETUP, amount_format, ask_tab, confirmation_open,
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
# line and its 95 % interval drawn over them still read, so not held to the
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


def blues(which=None):
    """The colour scheme of the triangle's layers for a theme (the reader's
    when not given)."""
    return alt.SchemeParams(name="blues", extent=list(_BLUES_EXTENT[which or theme()]))


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
    of a project's data can leave the machine through its action menu."""
    st.altair_chart(_private(chart), **kwargs)


def panel_label(opt, name):
    if name == insights.SCORE:
        return wording.OVERALL_SCORE_COLUMN
    obj = next(o for o in opt.objectives if o['name'] == name)
    return opt._measurement_column(obj)


def track_panels(opt):
    """The measurements in the order Results lists them (largest Share of
    score first), then the Overall score."""
    return [o['name'] for o in opt.measurements_by_importance()] + [insights.SCORE]


def path_panels(opt):
    return [insights.SCORE] + [o['name'] for o in opt.measurements_by_importance()]


def _target_rows(opt, names):
    return [{"panel": panel_label(opt, o['name']), "mark": "target",
             "target": float(o['target'])}
            for o in opt.objectives
            if o['name'] in names and o.get('goal') == 'target' and o.get('target') is not None]


def _only(mark):
    """Each layer draws only its own rows of the chart's one long frame. A
    layer given a frame of its own is drawn whole in EVERY facet panel, so a
    target, a grey interval or a start marker would show up everywhere."""
    return alt.FieldEqualPredicate(field="mark", equal=mark)


def _untested_layers(base):
    """"Grey: not tested near here": a light band, kept light so the line
    and the interval band over it still read, with a rule at each edge in
    the theme's token, which carries the 3:1 (D1; O16: the band alone was
    1.13:1)."""
    edge_c, edge_o = mark("untested_edge")
    fill_c, fill_o = _FILLS[theme()]["untested"]
    grey = _only("grey")
    return [base.mark_rect(opacity=fill_o, color=fill_c, tooltip=False)
            .encode(x="x:Q", x2="x2:Q").transform_filter(grey),
            base.mark_rule(color=edge_c, opacity=edge_o, strokeWidth=1, tooltip=False)
            .encode(x="x:Q").transform_filter(grey),
            base.mark_rule(color=edge_c, opacity=edge_o, strokeWidth=1, tooltip=False)
            .encode(x="x2:Q").transform_filter(grey)]


def _target_layer():
    # Dashed, thin and grey: never mistaken for a likely-range bar. The
    # neutral token, as on "Can you have both?": gray at 0.6 was 2.5:1 on
    # the dark page and 2.1:1 on white (last fix wave).
    colour, opacity = mark("neutral")
    return (alt.Chart().mark_rule(strokeDash=[4, 3], strokeWidth=1, color=colour,
                                  opacity=opacity, tooltip=False)
            .encode(y="target:Q").transform_filter(_only("target")))


def _faceted(layers, df, order):
    # Fixed panel size, three to a row: fits beside the sidebar however many
    # formulations there are (the x axis squeezes, the page does not widen).
    return (alt.layer(*layers, data=df).properties(width=220, height=160)
            .facet(facet=alt.Facet("panel:N", sort=order, title=None), columns=3)
            .resolve_scale(y="independent"))


def _padded(values):
    """One domain for both axes of a measured-against-predicted panel: the
    values' extent padded 5 % (review M4), never zero-width."""
    lo, hi = min(values), max(values)
    pad = 0.05 * (hi - lo) if hi > lo else max(0.5, 0.05 * abs(hi))
    return [lo - pad, hi + pad]


def measured_vs_predicted_chart(opt, checks):
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
            "latest": c.round == latest} for c in mine])
        x = alt.X(f"{predicted}:Q", title=predicted, scale=scale)
        y = alt.Y(f"{measured}:Q", title=measured, scale=scale)
        fmt = f".{d}f"
        tip = [alt.Tooltip(f"{f}:{t}", title=f, **({"format": v} if v else {}))
               for f, t, v in ((formulation, "Q", "d"), (wording.ROUND_CAP, "N", None),
                               (predicted, "Q", fmt), (low, "Q", fmt), (high, "Q", fmt),
                               (measured, "Q", f".{max(1, d)}f"))]
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
        rings = base.mark_point(size=50, strokeWidth=1.5).encode(
            x=x, y=y, opacity=alt.condition(latest_first, alt.value(1.0), alt.value(0.45)),
            tooltip=tip)
        filled = (base.mark_point(size=50, filled=True).encode(
            x=x, y=y, opacity=alt.condition(latest_first, alt.value(1.0), alt.value(0.45)),
            tooltip=tip).transform_filter(alt.FieldEqualPredicate(field="inside", equal=True)))
        panels.append(alt.layer(line, bars, rings, filled)
                      .properties(width=220, height=220, title=_say(opt, panel_label(opt, name))))
    if not panels:
        # Checks only for outputs no longer in the project: nothing to draw.
        return None
    return alt.concat(*panels, columns=3)


def path_chart(opt, path, preds, near=None):
    """What happens if you change one ingredient? — one panel per output: a
    neutral predicted line, a light 95 % likely-range band, grey shading
    where nothing was tested near here (`near`, insights.tested_near, worked
    out when not given), a dotted start marker."""
    if not path.amounts or not preds:
        return None
    near = insights.tested_near(opt, path) if near is None else near
    names = [n for n in path_panels(opt) if n in preds]
    x_title = f"{path.name} ({path.unit})" if path.unit else path.name
    rows = []
    for n in names:
        label = panel_label(opt, n)
        d = decimals(opt, n)
        for t, p in zip(path.amounts, preds[n]):
            rows.append({"panel": label, "mark": "curve", "x": t, "mean": p.mean,
                         "low": p.low, "high": p.high,
                         "amount": _amount(t, path.unit),
                         "predicted": f"{p.mean:.{d}f} ({range_text(p.low, p.high, lambda v: f'{v:.{d}f}')})"})
        rows.append({"panel": label, "mark": "start", "x": path.start_amount})
        for lo, hi in insights.untested_runs(path, near):
            rows.append({"panel": label, "mark": "grey", "x": lo, "x2": hi})
    targets = _target_rows(opt, names)
    df = pd.DataFrame(rows + targets)
    # Not from 0: the path runs over this row's own Lowest-Highest, and a
    # zero-based axis squeezed a 4-6 g path into the right third of each panel.
    x = alt.X("x:Q", title=x_title, scale=alt.Scale(zero=False, nice=False))
    # Not from 0 either: the effect the question asks about filled a tenth
    # of a 0-100 panel (0.8.0 review fixes §6).
    y = alt.Y("low:Q", title=None, scale=alt.Scale(zero=False))
    # The current model's prediction, not a Round-ahead one (03 §5.2).
    tip = [alt.Tooltip("amount:N", title=path.name),
           alt.Tooltip("predicted:N", title=_say(opt, wording.PREDICTED))]
    band = (alt.Chart().mark_area(opacity=_BAND, tooltip=False)
            .encode(x=x, y=y, y2="high:Q")
            .transform_filter(_only("curve")))
    line = (alt.Chart().mark_line().encode(x=x, y="mean:Q", tooltip=tip)
            .transform_filter(_only("curve")))
    ref_c, ref_o = mark("reference")
    start = (alt.Chart().mark_rule(strokeDash=[2, 2], tooltip=False, color=ref_c, opacity=ref_o)
             .encode(x="x:Q").transform_filter(_only("start")))
    layers = [band, line, start]
    if any(r["mark"] == "grey" for r in rows):
        layers[:0] = _untested_layers(alt.Chart())
    if targets:
        layers.append(_target_layer())
    return _faceted(layers, df, [panel_label(opt, n) for n in names])


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
    return wording.round_track_short(
        t.rounds or None, t.inside if t.record else None, t.total, rough=bool(t.rough),
        gap_text=None if t.gap is None else f"{abs(t.gap):.1f}", below=(t.gap or 0) < 0,
        see_insights=bool(t.see_insights), no_skill=t.no_skill is not None)


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
    if _poll_step(opt, stages):
        app_rerun()


def _say(opt, text):
    """Every line on the tab in the project's own words, as the rest of the
    app writes them: trials, not formulations, in a project of process
    settings alone."""
    return wording.for_project(opt, text)


def _question(opt, text):
    """A section's noun heading, drawn as a real heading (03 §4): readers
    of JMP and Design-Expert scan headings for the analysis they know, and
    a screen reader lists them."""
    st.markdown(f"##### {_say(opt, text)}")


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
    """The tab (ftue spec §1-§2, rev. 2; review R-B1, R-I4, R-I8): the
    header, the three status lines (Predictions, Progress, Next), Next's
    primary button (the foot) with its Set up secondary when there is one,
    at most one tertiary link, the Explored and unexplored fold, then the
    detail sections — each its noun heading and one figures fold with
    today's exact lines — and the folds below. At the gate (fewer than
    MIN_ROWS scored) the same order with the model-free parts; with nothing
    scored, one line and the foot; after a failed fit, one line and the
    model-free Progress line.

    Drawn at once, before anything is fitted (review I15, R-I8): the header,
    each status line's place with its caption, the button under them, the
    link, and Progress's figures. The places fill as their facts are ready —
    here in this run when the tab fits in this process, part by part while
    the workers are still fitting — so the button moves by at most a line's
    difference per line and never jumps over a filled screen."""
    # A formulation moved to Not scored counts for nothing here (review M-3).
    if len(opt.Y_history) == sum(map(opt.is_set_aside, range(len(opt.Y_history)))):
        st.caption(_say(opt, wording.INSIGHTS_NONE))
        _foot(opt)
        return
    # The one "scored" count on the tab, the header's: formulations with all
    # their results. A waiting one, or one moved to Not scored, is not
    # scored, so it counts towards neither MIN_ROWS nor "So far: N".
    n = opt.complete_count()
    st.caption(_say(opt, header_text(opt)), help=_say(opt, header_detail(opt)))
    if insights_jobs.stopped():
        # C2 (review M5): the workers broke too often; everything is now
        # worked out here, behind each screen's own indicator.
        st.caption(_say(opt, wording.INSIGHTS_IN_PROCESS))
    _land()
    if n < insights.MIN_ROWS:
        _gate(opt)
        return
    filling = waiting(opt, "all")
    # The page in screen order, before any fit (R-I8): the three places,
    # the button, its secondary's place and the link's, the progress line's
    # place (under the button, so it never moves it), the Explored fold's,
    # then the sections.
    places = _places(opt)
    if filling:
        _fill_in(opt, places)
        with places.explored:
            explored_fold(opt, None)    # W3 (§8, R-I14): under the button
        _how(opt, None)
        return
    st.session_state.pop(PROGRESS_SEEN_KEY, None)
    cold = _cached(opt) is None
    # Nothing to fit (a reopen, everything beside the project): no progress
    # line, which said "0 of 23 Rounds checked…" while none was (04 I-1).
    quiet = cold and _nothing_to_fit(opt)
    computed = _computed(opt, places.bar, quiet=quiet)
    if computed.get("failed"):
        # The model lines come off; one line says so, in their place, with
        # the model-free Progress line under it (review B1's table).
        places.bar.empty()
        with places.lines.container():
            st.caption(_say(opt, wording.INSIGHTS_FAILED))
            _status_line(opt, insights.status_facts(opt, ()), "progress")
        with places.explored:
            explored_fold(opt, computed)    # W3 (§8, R-I14): under the button
        _how(opt, None)
        return
    checks = computed["checks"]
    h = insights.headline(opt, checks)
    yard = insights.noise_yardstick(opt, checks)
    _restate(opt, places.line1, places.first_yard, yard)
    with places.trust:
        # The disagreeing Repeat is said once: in Prediction figures, unless
        # Next step already says it (its row does not depend on the
        # predicted best or the expected improvement: item 2).
        _prediction_section(opt, computed, h, yard,
                            repeat=insights.next_step(opt, checks, None).kind != "repeat")
    with places.rest:
        # The bar, when it showed, stays up through the predicted best's
        # search: one indicator per wait (V28, Plan A review C M-2).
        standing = _stand(opt, computed, yard, spinner=not cold or quiet)
    places.bar.empty()
    score = computed["fitted"].get(insights.SCORE)
    gain = shown_gain(opt, score, checks)
    with insights.memo_in_use(memo(opt)):
        facts = screen_facts(opt, checks, standing, gain,
                             withheld=claims_withheld_reason(opt, score, checks))
    _finish(opt, places, facts)
    with places.explored:
        explored_fold(opt, computed)
    # The folds, collapsed (§2.5), each with a key of its own.
    with st.expander(_say(opt, _try_one_label(opt)), key="insights_try_one",
                     on_change="rerun"):        # W3: the §8.3 pointer opens it
        _change(opt, computed, yard)
    _more_charts(opt, computed)
    _how(opt, computed)
    save(opt)


class _Places:
    """The places render draws before anything is worked out, in screen
    order (R-I8), for _fill_in and the rest of render to fill."""


def _places(opt):
    p = _Places()
    p.lines = st.empty()
    with p.lines.container():
        _status_lines(opt, None)
    _foot(opt)
    p.change = st.empty()
    p.link = st.empty()
    with p.link.container():
        p.why = _why_link(opt)
    p.bar = st.empty()
    p.explored = st.container()
    # Keyed, so a link from another tab can land on them (04 §5).
    p.trust = st.container(key=TRUST_SECTION)
    stand = st.container(key=STAND_SECTION)
    with stand:
        _question(opt, wording.Q_STAND)
        with _figures(opt, wording.FOLD_PROGRESS_FIGURES, PROGRESS_FIGURES_KEY):
            _basis(opt, "progress")
            p.first_yard = insights.noise_yardstick(opt)
            p.line1 = st.empty()
            with p.line1.container():
                best = _stand_lines(opt, p.first_yard)
            p.rest = st.container()
            if best is not None:
                _progress_fold(opt)
    p.nxt = st.container()
    return p


def _land():
    """A link that lands on a section opens the figures fold the lines it
    points at now sit in (ftue M2): E1 at Prediction check with Measured
    against predicted open; E3's Next step at Progress. Only on arrival
    (ui_helpers.SCROLL_KEY, drained by app.py after the tab is drawn), so
    a fold the reader closed stays closed."""
    where = st.session_state.get(SCROLL_KEY)
    if where == TRUST_SECTION and st.session_state.get(EACH_KEY):
        st.session_state[PREDICTION_FIGURES_KEY] = True
    elif where == STAND_SECTION:
        st.session_state[PROGRESS_FIGURES_KEY] = True


def _gate(opt):
    """Fewer than MIN_ROWS scored (review M8, R-B1 row 3): the three status
    lines with no model behind them, the button, then Progress's figures
    and Method and definitions."""
    with insights.memo_in_use(memo(opt)):
        facts = insights.status_facts(opt, ())
    _status_lines(opt, facts)
    _foot(opt)
    _why_link(opt)
    explored_fold(opt, None)
    with st.container(key=STAND_SECTION):
        _question(opt, wording.Q_STAND)
        with _figures(opt, wording.FOLD_PROGRESS_FIGURES, PROGRESS_FIGURES_KEY):
            _basis(opt, "progress")
            _stand_first(opt, insights.noise_yardstick(opt))
    _how(opt, None)


def _finish(opt, places, facts):
    """The status lines, Next step's section, the Set up secondary and, when
    no Why link was drawn, the Explored link: everything that waits for the
    predicted best and the expected improvement."""
    with places.lines.container():
        _status_lines(opt, facts)
    with places.nxt:
        _question(opt, wording.Q_NEXT)
        with _figures(opt, wording.FOLD_NEXT_REASONING, NEXT_REASONING_KEY):
            _basis(opt, "next", facts)
            change = _next_step(opt, facts)
    with places.change.container():
        _setup_button(change)
    if not places.why:
        # R-I4's one link, by priority: Why (drawn above, with the places),
        # then the Repeat, then the Explored link (ftue fix C).
        with places.link.container():
            if not _repeat_link(opt, facts):
                explored_link(opt, facts.next)


def _figures(opt, label, key):
    """A figures fold (§2, R-I8): keyed, so a link can open it; never lazy,
    so its lines are worked out with the status lines they detail."""
    return st.expander(_say(opt, label), key=key, on_change="rerun")


def _basis(opt, which, facts=None):
    """A figures fold's first line, its technical basis, as W2's basis_line
    gives it with its help (R-I6: a fold label carries no tooltip): "next"
    needs the facts, for the row that fired."""
    text, help_text = basis_line(opt, which, facts)
    st.caption(text, help=help_text)


def _line(opt, text, line_id, slot=None, facts=None):
    """One of today's exact lines in its figures fold, with W2's help for
    the statistics it names (R-I6: st.markdown, which renders as st.write
    did and takes help=)."""
    (slot or st).markdown(_say(opt, text), help=tooltip_for(opt, line_id, facts))


def _restate(opt, lines, first_yard, yard):
    """Line 1 again, in its place, when the checks changed the yardstick it
    was drawn with (a disagreeing Repeat is left out of the Repeat spread)."""
    if yard != first_yard:
        with lines.container():
            _stand_lines(opt, yard)


def _fill_in(opt, places):
    """The tab while the workers are still fitting (spec 2026-09-30 §10.3,
    review I15, ftue R-I8), into the places render drew: one progress line
    with a count under the button, then Prediction figures once every
    Round's Overall score and the verdict's Rounds are in, then the status
    lines, Progress's figures and Next step together once the current
    Overall-score model is in too (each status line needs it: the guard
    under Reliable, the luck clause, the Next row). Each is drawn once,
    final; the folds wait for everything. keep_up's poller reruns the page
    as parts land."""
    specs, mem = _plan(opt), memo(opt)
    done, total = insights_jobs.rounds_done(opt, specs, mem)
    fraction = sum(s.key in mem for s in specs) / max(1, len(specs))
    places.bar.progress(fraction, text=_say(opt, wording.insights_progress(done, total)))
    # The poller reruns the page when this count moves (C1; 01 F2).
    st.session_state[PROGRESS_SEEN_KEY] = (insights.data_key(opt), done)
    checks = None if waiting(opt, "trust") else _partial_checks(opt, "trust")
    if checks is None:
        return
    yard = insights.noise_yardstick(opt, checks)
    _restate(opt, places.line1, places.first_yard, yard)
    step = insights.next_step(opt, checks, None)
    # Prediction figures first (item 2): the Repeat row does not depend on
    # the predicted best.
    with places.trust:
        _prediction_section(opt, {"checks": checks, "partial": True},
                            insights.headline(opt, checks), yard,
                            repeat=step.kind != "repeat")
    if waiting(opt, "stand"):
        return
    with insights.memo_in_use(mem):
        score = insights.fit(opt, insights.SCORE, list(range(len(opt.Y_history))))
    computed = {"fitted": {insights.SCORE: score}, "checks": checks, "partial": True}
    with places.rest:
        standing = _stand(opt, computed, yard)
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
    if change and st.button(change, key="insights_change_setup",
                            disabled=confirmation_open()):
        st.session_state[SETUP_CALLOUT_KEY] = change
        go_to_tab(TAB_SETUP, section=SETUP_CALLOUT_SLOT)



# ------------------------------------------------------------------ #
#  Insights for a first-time user (spec 2026-10-01 rev. 2), W1: the three
#  status lines (review R-B1, R-I5, R-I9, R-I10, M1, M7, M10) and the one
#  tertiary link (R-I3, R-I4, R-I14).
# ------------------------------------------------------------------ #

_PLACEHOLDERS = {"predictions": wording.CHECKING_PREDICTIONS,
                 "progress": wording.CHECKING_PROGRESS, "next": wording.PREPARING_NEXT_STEP}
_STATUS_ORDER = ("predictions", "progress", "next")


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
    return tuple(wording.binding_plain(b.kind, b.name, b.text, values=b.name in settings)
                 for b in (getattr(standing, "binding", None) or ()))


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
    row, h = facts.prediction_row, facts.headline
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
    then = f" {wording.CHECK_SCATTER}" if check else ""
    if row == "too_few":
        return wording.pred_too_few(facts.checked, insights.SKILL_MIN, tail) + then
    if row == "wide":
        return wording.pred_wide(tail)
    if row == "no_skill":
        return wording.pred_no_skill(gap, tail, skill_text=skill) + then
    if row == "unclear":
        return wording.pred_unclear(gap, tail, skill_text=skill) + then
    allowed = insights.allowed_misses(h.total)
    if row == "off_coverage":
        return wording.pred_off_coverage(h.rounds, h.total - h.inside, h.total, allowed,
                                         None if gap is None else wording.bias_fact(*gap))
    if row == "caution":
        return wording.pred_caution(h.rounds, h.total - h.inside, h.total, allowed, tail) + then
    return wording.pred_reliable(h.rounds, h.inside, h.total, allowed, tail) + then


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
    score = f"{f.score:.2f}"
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
        insights.rise_threshold(1.0 if answer.sd is None else answer.sd, max(1, answer.q),
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
        word = None if facts.progress is None else wording.PROGRESS_WORDS[facts.progress]
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
        word = None if facts.progress is None else wording.PROGRESS_WORDS[facts.progress]
        out["progress"] = wording.status_line(wording.STATUS_PROGRESS,
                                              progress_short(opt, facts), word)
    if "next" not in facts.pending:
        out["next"] = wording.status_line(wording.STATUS_NEXT, next_short(opt, facts))
    return {k: _say(opt, v) for k, v in out.items()}


def _full_sentence(line):
    """A full status line's sentence, its label and bold word taken off."""
    return re.sub(r"^\w+: (\*\*[^*]+\*\* )?", "", line)


def status_help(opt, facts, which):
    """The line's tooltip: its full sentence, then W2's exact rule."""
    full = status_full_texts(opt, facts).get(which)
    tip = tooltip_for(opt, which, facts)
    parts = [p for p in (full and _full_sentence(full), tip) if p]
    return "\n\n".join(parts) or None


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
    row, h = facts.prediction_row, facts.headline
    check = False
    for c in facts.clauses:
        if c[0] == "wide":
            check = wide_cause(c[1], getattr(facts, "wide_factors", None) or {}) is not None
    then = f" {wording.SHORT_CHECK_SCATTER}" if check else ""
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
    allowed = insights.allowed_misses(h.total)
    if row in ("off_coverage", "caution"):
        line = wording.short_out_of_range(h.total - h.inside, h.total, allowed)
        return line + (then if row == "caution" else "")
    return wording.short_in_range(h.inside, h.total) + then


def progress_short(opt, facts):
    """The Progress line on screen (concision round): one fact and Best so
    far; ties within noise, luck and "little more" are in the tooltip."""
    f = facts.best
    if f is None:
        return wording.NO_COMPLETE_FORMULATION
    ceiling = opt.utility_ceiling()
    tie = None if facts.tie is None else int(opt.formulation_ids[facts.tie])
    best = wording.short_best(f"{f.score:.2f}", ceiling, f.formulation, tie)
    a = facts.improvement
    if facts.progress in ("rising", "within_noise"):
        rise, need = rise_and_need(a)
        return f"{wording.short_rise(rise, need, rising=facts.progress == 'rising')} {best}"
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


def _try_one_label(opt):
    """The first fold's name ("Try one …", ftue §2; review I14): names
    process settings only when the project has one, and ingredients only
    when it has one."""
    choices = insights.path_choices(opt)
    return wording.try_one_fold(insights.has_setting_choice(opt),
                                insights.has_ingredient_choice(opt)
                                or (not choices and opt.has_ingredients()))


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
    Make a round's column help (spec §0.3): its 95 % intervals and mean
    error over every Round checked, naming them (0.8.0 e2e fixes, review
    I1). A steady bias is said once, by steady_bias_text, so this line
    never says it. '' without a record. `yard` is not read (kept for
    callers)."""
    h = insights.headline(opt, checks) if h is None else h
    if h.score is None:
        return ""
    return wording.score_gap_line(h.score.inside, h.score.total, h.score.mean_miss, "",
                                  n=h.score.total, rounds=h.score.rounds)


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
    §5.2): the measurements' values inside their 95 % intervals with the
    miss count the rule allows; some outputs wide adds the wide line. The
    verdict names the measurements, and the Overall score's line under it
    says its mean error, so nothing is added for a bias (`bias` is kept for
    callers). `checks`: for the over-wide rule's basis."""
    if state == "none":
        return wording.TRUST_NONE
    if state == "early":
        return wording.trust_early(h.total)
    if state == "wide":
        return wording.TRUST_WIDE
    allowed = insights.allowed_misses(h.total)
    if state in ("caution", "unreliable"):
        line = wording.trust_misses(len(h.rounds), h.rounds, h.inside, h.total,
                                    many=state == "unreliable", allowed=allowed)
    elif state == "over_wide":
        line = wording.trust_over_wide(h.total, h.rounds,
                                       insights.median_z(h, checks) if checks else None)
    else:
        line = wording.trust_within(h.inside, h.total, h.rounds, allowed=allowed)
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


def _prediction_section(opt, computed, h, yard, repeat=True):
    """Prediction check (ftue §2, R-I8): its noun heading and one fold,
    Prediction figures, opening with its basis, then today's exact lines."""
    _question(opt, wording.Q_TRUST)
    with _figures(opt, wording.FOLD_PREDICTION_FIGURES, PREDICTION_FIGURES_KEY):
        _basis(opt, "predictions")
        _trust(opt, computed, h, yard, repeat)


def _trust(opt, computed, h, yard, repeat=True):
    """Prediction figures' lines (§2.2, verbatim since ftue R-I8): one
    verdict, the Overall score's own line, a disagreeing Repeat
    (`repeat`), one line on how, and Measured against predicted in its
    fold. Each line with W2's help (R-I6)."""
    checks = computed["checks"]
    state = insights.verdict_state(opt, h, checks)
    _line(opt, verdict_text(opt, h, state, checks=checks), "verdict")
    if h.score is not None:
        _line(opt, score_line(opt, checks, h, yard), "score_line")
    # Every output's steady bias, one line, a fact (review B1).
    bias = steady_bias_text(opt, checks) if state != "none" else ""
    if bias:
        _line(opt, bias, "steady_bias")
    # Skill, the Overall score's own fact under the verdict (review I1): in
    # every state past the gate but "none".
    skill = skill_text(opt, checks) if state != "none" else ""
    if skill:
        _line(opt, skill, "skill")
    flagged = insights.disagreeing_repeat(opt, checks) if repeat else None
    if flagged is not None:
        _line(opt, disagreement_text(flagged), "disagreement")
    st.caption(_say(opt, wording.TRUST_BEFOREHAND))
    # While the rest are still being fitted the chart would gain dots: it
    # waits for them (a part of the tab is drawn once, final). Keyed, so a
    # link can open it (E1, EACH_KEY); drawn only while open (04 §5).
    if computed.get("partial") or not checks:
        return
    with st.expander(_say(opt, wording.FOLD_MEASURED_CHART), key=EACH_KEY,
                     on_change="rerun") as each:
        if not each.open:
            return
        chart = measured_vs_predicted_chart(opt, checks)
        if chart is not None:
            show_chart(chart)
            st.caption(_say(opt, wording.MVP_CAPTION))


def _keep_if_offered(key, options):
    """A choice left from before a deletion or a Set up change is dropped
    before its box is drawn, so the box falls back to its first entry."""
    if key in st.session_state and st.session_state[key] not in options:
        del st.session_state[key]


def _change(opt, computed, yard):
    """"Try one ingredient or process setting" (§2.5): its first line ranks
    the choices near what has been tested (review I8), then the two boxes,
    the path and its two captions."""
    choices = insights.path_choices(opt)
    if not choices:
        st.caption(_say(opt, wording.change_none(settings_only=not opt.has_ingredients())))
        return
    # render asks this only once MIN_ROWS formulations are scored, so there
    # is always a Best so far to start from (Task 5: its None fallback went).
    best = opt.best_index()
    ids = [int(x) for x in opt.formulation_ids]
    best_no = ids[best]
    fitted = computed["fitted"]
    # BO views §3.3 (review I6, I8): the path's layer; the Uncertainty and
    # Expected gain layers may start from a suggestion still to make, and
    # open on the open Round's first one.
    layer = _layer_pick(opt, fitted, PATH_LAYER_KEY)
    layered = layer != wording.LAYER_SCORE
    # By Formulation number, not list position: a deletion on Results shifts
    # positions, and the box must keep naming the formulation it named.
    starts = [best_no] + [ids[i] for i in reversed(range(len(ids))) if i != best]
    if layered:
        starts = suggested_numbers(opt) + starts
    start_key = LAYER_START_KEY if layered else START_KEY
    _keep_if_offered(CHOICE_KEY, choices)
    _keep_if_offered(start_key, starts)
    answer = st.container()
    left, right = st.columns(2)
    number = right.selectbox(
        _say(opt, wording.START_FROM), starts, key=start_key,
        format_func=lambda no: _say(opt, wording.start_best(no) if no == best_no
                                    else anchor_text(opt, no)))
    try:
        recipe = anchor_recipe(opt, number)
    except Exception:   # no start: the path says it could not be drawn, below
        recipe = None
    # One anchor per headline (01 O10, over T56 I3): the answer line,
    # Compare all and the choice's default rank from the start the chart
    # draws from, under every layer. Under Uncertainty or Expected
    # improvement that may be a suggestion still to make, named as the
    # chart's caption names it.
    ranked_no = number
    try:
        found = insights.tested_near_changes(opt, fitted, recipe)
    except Exception:   # the answer line is left out; the path still draws
        found = insights.NearChanges([], [], False)
    # The box opens on the largest change near what has been tested; after
    # that it keeps the reader's own choice (_keep_if_offered drops it only
    # when it is no longer offered).
    if found.moved and CHOICE_KEY not in st.session_state:
        st.session_state[CHOICE_KEY] = found.moved[0].name
    with answer:
        _near_answer(opt, found, ranked_no == best_no, ranked_no, fitted, recipe, yard,
                     start_text=None if ranked_no in ids else anchor_text(opt, ranked_no))
    name = left.selectbox(_say(opt, wording.INGREDIENT_OR_SETTING), choices, key=CHOICE_KEY)
    try:
        if recipe is None:
            raise ValueError("no start")
        path = insights.build_path(opt, name, recipe)
        if path.start_breaks:
            st.caption(_say(opt, wording.path_start_breaks(number, path.start_breaks)))
            return
        near = insights.tested_near(opt, path)
        lines = _path_captions(opt, path, number, fitted, near)
        if layered:
            chart = path_layer_chart(opt, path, _path_layer(opt, fitted, path, name, number,
                                                            layer), near, layer)
            lines = lines[:1] + [wording.path_layer_caption(layer, anchor_text(opt, number))]
        else:
            chart = path_chart(opt, path, insights.path_predictions(opt, fitted, path), near)
    except Exception:   # one path failing must not take the tab down
        st.caption(_say(opt, wording.panel_failed(name)))
        return
    if chart is not None:
        show_chart(chart)
    for line in lines:
        st.caption(_say(opt, line))


def _path_layer(opt, fitted, path, name, number, layer):
    """A layer's values along the path, once per data, path and layer; the
    anchor's own amounts in the key, so an edited suggestion is not served
    stale (T56 m11)."""
    key = (insights.setup_key(opt), name, number, layer,
           insights._digest(path.recipes[:1] if path.recipes else []))
    hit = st.session_state.get(PATH_LAYER_CACHE)
    if hit and hit[0] == key:
        return hit[1]
    with st.spinner(_say(opt, wording.INSIGHTS_WORKING)):
        value = layer_values(opt, fitted, path.recipes, layer)
    st.session_state[PATH_LAYER_CACHE] = (key, value)
    return value


def path_layer_chart(opt, path, values, near, layer):
    """The path under the Uncertainty or Expected gain layer (spec §3.3):
    one panel, the suggestion model's sd or its expected improvement in
    points along the path, grey where nothing was tested near, the start
    dotted."""
    if not values or not path.amounts:
        return None
    title = wording.LAYER_SD_LEGEND if layer == wording.LAYER_SD else wording.LAYER_GAIN_LEGEND
    x_title = f"{path.name} ({path.unit})" if path.unit else path.name
    rows = [{"mark": "curve", "x": t, "v": v, "amount": _amount(t, path.unit)}
            for t, v in zip(path.amounts, values)]
    rows.append({"mark": "start", "x": path.start_amount})
    rows += [{"mark": "grey", "x": lo, "x2": hi} for lo, hi in insights.untested_runs(path, near)]
    df = pd.DataFrame(rows)
    x = alt.X("x:Q", title=x_title, scale=alt.Scale(zero=False, nice=False))
    layers = [alt.Chart(df).mark_line().encode(
                  x=x, y=alt.Y("v:Q", title=_say(opt, title)),
                  tooltip=[alt.Tooltip("amount:N", title=path.name),
                           alt.Tooltip("v:Q", title=_say(opt, title), format=".2f")])
              .transform_filter(_only("curve")),
              alt.Chart(df).mark_rule(strokeDash=[2, 2], tooltip=False, color=mark("reference")[0],
                                      opacity=mark("reference")[1]).encode(x="x:Q")
              .transform_filter(_only("start"))]
    if any(r["mark"] == "grey" for r in rows):
        layers[:0] = _untested_layers(alt.Chart(df))
    return alt.layer(*layers).properties(height=200)


def change_chart(changes, axis=wording.CHANGE_AXIS, d=1):
    """Compare all: one bar per choice, ranked by how much it changes the
    prediction near what has been tested (dark), extended in a lighter
    shade to its change along the whole path (spec 2026-09-30 §2.5, review
    I8). A choice whose start is not tested near has only the light bar,
    and its tooltip says so rather than reading 0.0 (Task 5 review I-1).
    `changes`: insights.NearChange, in the order to draw them."""
    if not changes:
        return None
    df = pd.DataFrame([{"name": c.name, "near": round(c.near or 0.0, d),
                        "near_text": (wording.MIXTURE_UNTESTED if c.near is None
                                      else f"{c.near:.{d}f}"),
                        "whole": round(c.whole, d)} for c in changes])
    order = [c.name for c in changes]
    y = alt.Y("name:N", sort=order, title=None, axis=alt.Axis(labelLimit=0, titleLimit=0))
    tip = [alt.Tooltip("name:N", title=wording.INGREDIENT_OR_SETTING),
           alt.Tooltip("near_text:N", title=wording.CHANGE_NEAR),
           alt.Tooltip("whole:Q", title=wording.CHANGE_WHOLE, format=f".{d}f")]
    base = alt.Chart(df)
    light = base.mark_bar(opacity=0.35).encode(x=alt.X("whole:Q", title=axis), y=y, tooltip=tip)
    dark = base.mark_bar().encode(x=alt.X("near:Q", title=axis), y=y, tooltip=tip)
    return alt.layer(light, dark).properties(height=28 * len(changes) + 20)


# The whole path's change must beat the tested-near one by more than this
# (a tenth of a point, the precision shown) before near_bigger is said.
_FLAT = 0.1


def _near_answer(opt, found, from_best, number, fitted, recipe, yard, start_text=None):
    """The fold's first line (§2.5, review I8): the choice that changes the
    predicted Overall score most near what has been tested — two when they
    are nearly tied — only when that change is more than Repeats differ
    (the yardstick); a line naming any choice the limits hold in place, one
    naming two ingredients moved together, and the Compare all fold."""
    start = start_text or wording.spread_from(from_best, number)
    # With no ingredient to change, every choice is a process setting: it
    # has values, not amounts, and the lines name process settings alone.
    settings_only = not insights.has_ingredient_choice(opt)
    moved = found.moved
    if not moved:
        if found.held:
            st.markdown(_say(opt, wording.held_all(start, settings_only)))
        return
    top = moved[0]
    standing = [c for c in moved if c.near is not None and c.near > yard.value]
    # The noise guard (spec 2026-09-30-bo-views §2.3, review I3, I4): no
    # effect is ranked, nor the Compare all bars drawn, while the Overall-
    # score model is likely fitting noise. The lines that need no model
    # (held in place, correlated) stay.
    guard = _guard(opt, fitted)
    if guard.fires:
        st.markdown(_say(opt, guard_text(guard, settings_only)))
    elif not found.start_tested:
        st.markdown(_say(opt, untested_start_text(opt, recipe, start)))
    elif not standing:
        st.markdown(_say(opt, wording.near_none(settings_only, insights.has_setting_choice(opt))))
    else:
        second = moved[1] if len(moved) > 1 else None
        if (second is not None and second.near is not None
                and second.near >= insights.NEAR_TIE * top.near):
            st.markdown(_say(opt, wording.near_two(top.name, second.name, f"{top.near:.1f}",
                                                   f"{second.near:.1f}", start)),
                        help=_say(opt, wording.tip_local_effect(settings_only)))
        else:
            st.markdown(_say(opt, wording.near_line(top.name, f"{top.near:.1f}", start,
                                                    only=len(moved) == 1)),
                        help=_say(opt, wording.tip_local_effect(settings_only)))
        if top.whole > top.near + _FLAT:
            st.markdown(_say(opt, wording.near_bigger(values=insights.is_setting(opt, top.name))))
    if found.held:
        st.caption(_say(opt, wording.held_line(start, found.held)))
    together = insights.correlated_pair(opt)
    if together:
        a, b, r = together
        st.caption(_say(opt, wording.moved_together(a, b, f"{r:.2f}")))
    if len(moved) > 1 and not guard.fires:
        with st.expander(_say(opt, wording.compare_all(insights.has_setting_choice(opt),
                                                       not settings_only))):
            _compare_all(opt, moved, settings_only, fitted, recipe)


def untested_start_text(opt, recipe, start_text):
    """The effect fold's line when its start is not tested near: with the
    nearest tested formulation and how far it is, by insights.nearest, the
    rule the Why fold's Nearest tested column uses (polish round)."""
    try:
        index, d = insights.nearest(opt, recipe)
    except Exception:
        index, d = None, None
    if index is None or d is None or d <= insights.CLOSE_TO_THE_BEST:
        # None to name, or the start is itself a tested formulation.
        return wording.near_start_untested(start_text)
    # Tenths, rounded up: 15.49 % must not read "15 %" beside a 15 % rule.
    share = f"{math.ceil(1000 * d - 1e-9) / 10:.1f}".removesuffix(".0")
    return wording.near_start_untested(start_text, int(opt.formulation_ids[index]), share)


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


def _compare_all(opt, moved, settings_only, fitted, recipe):
    """The ranking, by the Overall score (the answer's own changes) or by
    the measurement chosen beside it: its changes come from that
    measurement's own predictions, so the ranking agrees with its panel
    (0.8.0 review fixes §4.1). Each bar is the change near what has been
    tested, extended in a lighter shade to the whole path (review I8)."""
    fitted = fitted or {}
    options = [insights.SCORE] + ([] if recipe is None else [
        o['name'] for o in opt.measurements_by_importance()
        if getattr(fitted.get(o['name']), "model", None) is not None])
    _keep_if_offered(SPREAD_KEY, options)
    on = st.selectbox(_say(opt, wording.SPREAD_ON), options, key=SPREAD_KEY,
                      format_func=lambda n: _say(opt, panel_label(opt, n)))
    ranked, axis, d = moved, wording.CHANGE_AXIS, 1
    label = wording.OVERALL_SCORE_COLUMN
    if on != insights.SCORE:
        try:
            ranked = insights.tested_near_changes(opt, fitted, recipe, on).moved
        except Exception:
            ranked = []
        obj = next(o for o in opt.objectives if o['name'] == on)
        label = on
        axis, d = wording.change_axis(on, obj.get('unit') or ""), decimals(opt, on)
        # A measurement's own model, its own guard (review I3 scope).
        guard = _guard(opt, fitted, on)
        if guard.fires:
            st.caption(_say(opt, guard_text(guard, settings_only)))
            return
    chart = change_chart(ranked, axis, d)
    if chart is None:
        # No bars, so nothing for the caption to explain (Task 8 review,
        # Minor 4).
        return
    show_chart(chart)
    st.caption(_say(opt, wording.change_caption(values=settings_only, measure=label)))


def _whole(value):
    return f"{value:.0f}"


def range_text(low, high, fmt=None):
    """"70–85"; "-1.25 to 0.4" when a bound is negative, so a minus sign is
    never read as part of the dash."""
    fmt = fmt or _num
    sep = " to " if low < 0 or high < 0 else "–"
    return f"{fmt(low)}{sep}{fmt(high)}"


def score_text(p):
    """An Overall-score prediction: whole numbers, "78 (70–85)"."""
    return f"{_whole(p.mean)} ({range_text(p.low, p.high, _whole)})"


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


def _path_head(opt, path):
    if path.mode == "fill":
        return wording.path_head_fill(path.name, path.fill_name)
    if path.mode == "proportional":
        return wording.path_head_proportional(path.name,
                                              opt.batch_total_text(opt.formulation_total))
    if path.mode == "process":
        return wording.path_head_process(path.name)
    return wording.path_head_free(path.name)


def _path_captions(opt, path, number, fitted, near=None):
    """The path's two captions (§2.5, expert "Merge"): what moves, where it
    stops and what is fixed; then the key. The rarer lines (outside
    today's Lowest-Highest, calculations, unchecked limits, a missing
    panel) come between them."""
    stops = [(_amount(end.amount, path.unit), end.reason)
             for end in (path.low_end, path.high_end) if end.reason]
    fixed = [(n, _amount(a, u)) for n, a, u in path.fixed]
    lines = [wording.path_what(_path_head(opt, path), stops, fixed)]
    if path.outside:
        lines.append(wording.path_outside(number, _amount(path.start_amount, path.unit)))
    lines += [wording.path_follows(n) for n in path.follows]
    lines += [wording.path_not_checked(t, prop) for t, prop in path.skipped]
    for name in path_panels(opt):
        f = fitted.get(name)
        if f is None:
            continue
        if f.error:
            lines.append(wording.panel_failed(panel_label(opt, name)))
        elif f.model is None:
            lines.append(wording.panel_missing(panel_label(opt, name)))
    near = insights.tested_near(opt, path) if near is None else near
    lines.append(wording.path_key(number, grey=not all(near),
                                  targets=bool(_target_rows(opt, path_panels(opt)))))
    # The noise guard (review I3): the path still draws; its bends are not
    # to be read as effects.
    guard = _guard(opt, fitted)
    if guard.fires:
        lines.append(path_guard_line(guard))
    return lines


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


def _stand_first(opt, yard):
    """Progress figures' line 1 (§2.3) and the Progress chart's fold.
    Needs no model: drawn at the gate too."""
    if _stand_lines(opt, yard) is not None:
        _progress_fold(opt)


def _progress_fold(opt):
    """The Progress chart fold (review M8): Best so far by formulation, then
    Expected improvement by Round — worked out only once the fold is open
    (binding I15), and only from what the tab has worked out already."""
    with st.expander(_say(opt, wording.FOLD_PROGRESS_CHART), key=PROGRESS_KEY,
                     on_change="rerun") as fold:
        if not fold.open:
            return
        _progress_chart_body(opt)
        _gain_chart_body(opt)


def gain_frame(history):
    """One row per Round with an expected improvement or a Best so far."""
    return pd.DataFrame([{
        "round": g.round, "best": g.best_measured, "highest": g.highest, "gain": g.gain,
        "how": wording.GAIN_EXACT if g.exact else wording.GAIN_RECOMPUTED}
        for g in history])


def gain_chart(opt, history, model_free=False):
    """§3.2's chart (02 §4.4, review M11): two panels on one Round axis.
    Top, Best so far (measured) and the highest predicted mean over the
    allowed formulations; below, a bar for the expected improvement each
    Round's first suggestion was chosen for (none for a cold, own-only or
    imported Round). `model_free` (the guard or an unreliable verdict,
    review I4): Best so far alone."""
    df = gain_frame(history)
    if df.empty:
        return None
    rnd = _say(opt, wording.ROUND_CAP)
    x = alt.X("round:O", title=rnd)
    lines = df.melt(id_vars=["round"], value_vars=["best"] if model_free else ["best", "highest"],
                    var_name="series", value_name="value").dropna()
    names = {"best": wording.GAIN_BEST_MEASURED, "highest": wording.GAIN_HIGHEST_PREDICTED}
    lines["series"] = lines["series"].map(names)
    top = (alt.Chart(lines).mark_line(point=True)
           .encode(x=x, y=alt.Y("value:Q", title=wording.OVERALL_SCORE_COLUMN,
                                scale=alt.Scale(zero=False)),
                   color=alt.Color("series:N", title=None, legend=alt.Legend(labelLimit=0)),
                   tooltip=[alt.Tooltip("round:O", title=rnd),
                            alt.Tooltip("series:N", title=wording.OVERALL_SCORE_COLUMN),
                            alt.Tooltip("value:Q", title=wording.OVERALL_SCORE_COLUMN,
                                        format=".1f")])
           .properties(height=150))
    if model_free:
        return top
    bars = (alt.Chart(df.dropna(subset=["gain"])).mark_bar()
            .encode(x=x, y=alt.Y("gain:Q", title=wording.GAIN_AXIS),
                    tooltip=[alt.Tooltip("round:O", title=rnd),
                             alt.Tooltip("gain:Q", title=wording.GAIN_AXIS, format=".2f"),
                             alt.Tooltip("how:N", title=wording.GAIN_STATUS)])
            .properties(height=120))
    # One Round axis for both panels (T56 m1): a Round with no bar keeps
    # its place.
    return alt.vconcat(top, bars).resolve_scale(x="shared")


def _gain_chart_body(opt):
    """Expected improvement by Round in the Progress chart fold: from the
    checks and models already worked out (the memo's Round-ahead models, so
    a recomputed Round costs no fit, review I10); nothing before they are."""
    computed = _cached(opt)
    if computed is None or computed.get("failed") or bo_views.acquisition(opt) == "qucb":
        return
    checks = computed["checks"]
    h = insights.headline(opt, checks)
    trusted = (insights.verdict_state(opt, h, checks) not in ("caution", "unreliable")
               and not _model_guard(opt, computed["fitted"]).fires)
    if not trusted:
        # Nothing to add under a caution or unreliable verdict, a no-skill
        # record or the noise guard: Best so far is charted just above, and
        # Progress says why the figure is withheld (T56 I4).
        return
    try:
        with insights.memo_in_use(memo(opt)):
            history = bo_views.round_history(opt)
    except Exception:
        return
    chart = gain_chart(opt, history)
    if chart is None:
        return
    _question(opt, wording.GAIN_HEADING)
    show_chart(chart)
    st.caption(_say(opt, wording.GAIN_CHART_CAPTION))


def _stand_lines(opt, yard):
    """Progress figures' first lines (§2.3, verbatim since ftue R-I8): the
    answer, Best so far to two decimals as Results shows it (review M1),
    the Repeats line only after a rise (audit V24), then the runner-up
    within the noise threshold (R-I9: the tie said here; level 0 names
    equal scores only). "It came from the latest Round" is not said when
    there is only one Round (ftue §4). Returns insights.best_facts (None:
    nothing is scored)."""
    f = insights.best_facts(opt)
    if f is None:
        # Everything recorded still waits, in the words Results uses.
        _line(opt, wording.NO_COMPLETE_FORMULATION, "stand_line")
        return None
    answer = insights.improvement(opt, f, yardstick=yard)
    second = ("" if answer is not None
              else wording.STAND_LATEST if f.latest and f.rounds_since == 0 and not one_round(opt)
              else wording.stand_since(f.rounds_since) if f.rounds_since else "")
    line = wording.stand_best(f"{f.score:.2f}", f.formulation, f.round, number_list(f.without))
    head = stand_text(answer, yard) + " " if answer is not None else ""
    # Whether this line named the threshold, for line 2 in this same run
    # (standing_text): an assumed threshold is said once (review B2).
    st.session_state[_LINE1_KEY] = (id(opt), yard.pairs, says_threshold(answer))
    _line(opt, head + line + (" " + second if second else ""), "stand_line")
    if (says_threshold(answer)
            and yard.sd is not None and yard.sd >= insights.REPEAT_SD_SHOWN):
        _line(opt, repeats_text(yard), "repeats")
    tie = tie_text(opt, yard)
    if tie:
        _line(opt, tie, "tie")
    return f


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
    frame["series"] = score
    # Both axes named. Unlabelled, the x axis read 1.0 / 1.5 / 2.0 with one
    # round of data and nothing on the chart said what either number was.
    # Whole-number ticks, each once (spec 2026-09-30 §2.3, expert #8), and a
    # y axis that fits the scores rather than starting at 0.
    x = alt.X(f"{formulation}:Q", title=formulation,
              axis=alt.Axis(format="d", values=formulation_ticks(frame[formulation])))
    y = alt.Y(f"{score}:Q", title=score, scale=alt.Scale(zero=False))
    colour = alt.Color("series:N", legend=None)
    # Titled and rounded (0.8.0 review fixes §6): a measured Overall score
    # reads one decimal, never 73.283749….
    no_tip = alt.Tooltip(f"{formulation}:Q", title=formulation, format="d")
    score_line = (alt.Chart(frame).mark_line(tooltip=False)
                  .encode(x=x, y=y, color=colour))
    dots = (alt.Chart(frame).mark_point(size=40, strokeWidth=1.5)
            .encode(x=x, y=y, color=colour,
                    fill=alt.condition(alt.FieldEqualPredicate(field="full", equal=True),
                                       colour, alt.value("transparent")),
                    tooltip=[no_tip, alt.Tooltip(f"{score}:Q", title=score, format=".1f")])
            .transform_filter(alt.FieldValidPredicate(field=score, valid=True)))
    best_line = (alt.Chart(frame).mark_line(color="gray", strokeDash=[4, 3])
                 .encode(x=x, y=alt.Y(f"{best}:Q", title=score, scale=alt.Scale(zero=False)),
                         tooltip=[no_tip, alt.Tooltip(f"{best}:Q", title=best, format=".1f")]))
    return alt.layer(score_line, dots, best_line).properties(height=220)


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


def _progress_chart_body(opt):
    # Moved unchanged from Results (0.7.2), minus its own expander.
    show_chart(progress_chart(opt))
    # A ring is a scored (plotted) formulation outside best_candidates(); a
    # formulation still waiting is a gap, not a dot, so it does not count.
    pool = set(opt.best_candidates())
    rings = any(opt.fully_measured(i) and i not in pool for i in range(len(opt.Y_history)))
    st.caption(_say(opt, wording.progress_caption_2(rings)))


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


def best_frame(opt, best, fitted):
    """The predicted best and the formulation it is compared with, in the
    Round table's columns and units, each with its own predicted Overall
    score as on Make a round: the compared formulation's measured score can
    be above the predicted best, its prediction never is."""
    ids = [int(x) for x in opt.formulation_ids]
    no = ids[best.compare]
    label = (wording.start_formulation(no) if best.close is not None
             else wording.start_best(no))
    other = insights.start_recipe(opt, best.compare)
    frame = opt.batch_frame([{"formulation": 0, "recipe": best.recipe},
                             {"formulation": no, "recipe": other}])
    trying = opt.compared_with_column()
    frame = frame[[c for c in frame.columns if c not in (wording.NOTE, trying)]].copy()
    frame[wording.FORMULATION_CAP] = [wording.BEST_ROW, _say(opt, label)]
    try:
        [own] = insights.predict(opt, fitted[insights.SCORE], [opt._encode(other)])
        own = insights.widened(fitted[insights.SCORE], own, best.factor)
    except Exception:   # the table stands without the column
        own = None
    if own is not None:
        # Straight after Formulation: at 1440 px the last column was off
        # screen, and it is the one number the question is about (§6).
        frame.insert(1, wording.PREDICTED_SCORE_COLUMN,
                     [score_text(best.score), score_text(own)])
    return frame.rename(columns={wording.FORMULATION_CAP: _say(opt, wording.FORMULATION_CAP)})


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
    mean, low, high = _whole(p.mean), _whole(p.low), _whole(p.high)
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


def _stand(opt, computed, yard, spinner=True):
    """Answer 2 after line 1 (render draws that first, §2.3, review I15):
    the predicted best against Best so far (line 2), what it sits at (line
    3), the luck line, and its amounts in a fold only when it is clearly
    higher and not tested near. Returns the insights.Standing, or None when
    the predicted best could not be worked out (one line says why)."""
    fitted = computed["fitted"]
    score = fitted.get(insights.SCORE)
    if score is None or score.model is None:
        st.caption(_say(opt, wording.BEST_FAILED))
        return None
    best = _predicted_best(opt, fitted, computed["checks"],
                           spinner=spinner and not computed.get("partial"))
    if isinstance(best, str):
        st.caption(_say(opt, best))
        return None
    if best is _FAILED:
        st.caption(_say(opt, wording.BEST_FAILED))
        return None
    standing = standing_for(opt, best, computed["checks"], yard)
    held_back = claims_withheld_reason(opt, score, computed["checks"])
    _line(opt, standing_text(opt, standing, yard, withheld=held_back), "standing")
    if standing.binding:
        _line(opt, wording.stand_limits([binding_text(b) for b in standing.binding]), "limits")
    top = opt.best_index()
    if standing.luck and best.repeat is not None and top is not None:
        r = best.repeat
        _line(opt, wording.best_luck(int(opt.formulation_ids[top]), _whole(r.mean),
                                     _whole(r.low), _whole(r.high),
                                     f"{opt.Y_history[top]:.2f}"), "luck")
    # Expected improvement by Round (spec 2026-09-30-bo-views §3.2; review
    # I5, M8): its line here, its chart in the Progress chart fold.
    slot = st.empty()
    if _gain_pending(opt, computed):
        # Its place says it is coming while ask()'s search runs (C1): the
        # lines above are already drawn and never wait for it.
        _line(opt, wording.gain_pending(), "gain", slot)
    line = gain_text(opt, computed, yard, standing)
    if line:
        _line(opt, line, "gain", slot)
    else:
        slot.empty()
    if standing.higher and not standing.tested_near and not held_back:
        with st.expander(_say(opt, wording.show_its_amounts(not opt.has_ingredients()))):
            try:
                frame = best_frame(opt, best, fitted)
            except Exception:   # one line, never a traceback: the tab still stands
                st.caption(_say(opt, wording.BEST_FAILED))
            else:
                st.dataframe(frame.style.format(amount_format(opt, frame)), hide_index=True,
                             column_config={wording.PREDICTED_SCORE_COLUMN: st.column_config.Column(
                                 wording.PREDICTED_SCORE_COLUMN,
                                 help=_say(opt, wording.PREDICTED_SCORE_HELP))},
                             height=table_height(len(frame)))
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
        b = step.binding
        if b is None:
            return wording.next_close_to_top(settings_only=settings_only), None
        if b.kind == "limit":
            return (wording.next_close_to_top(wording.widen_limit(b.text), settings_only),
                    wording.CHANGE_LIMITS_ON_SETUP)
        change = (wording.raise_highest(b.name) if b.kind == "highest"
                  else wording.lower_lowest(b.name))
        return (wording.next_close_to_top(change, settings_only),
                wording.change_on_setup(b.name, highest=b.kind == "highest"))
    if step.kind == "rising":
        return wording.next_rising(_wide_names(getattr(step, "wide", None) or []), no, left), None
    return (wording.next_see_open(no, left) if left else wording.NEXT_SEE), None


def amounts_shown(facts):
    """Whether Progress figures draws "Show its amounts" for the untested
    row (_stand: not while the claims are withheld). The untested row
    fires only past the skill gate with no caution or unreliable verdict,
    so what can still hide them is the noise guard's floor or noise
    trigger, which status_facts reads under Reliable."""
    reasons = getattr(getattr(facts, "guard", None), "reasons", None) or ()
    return not any(r in reasons for r in ("floor", "noise"))


def _next_step(opt, facts):
    """Next-step reasoning's sentence (verbatim since ftue R-I8) for
    facts.next, naming the verdict's Rounds where it says a bias or misses
    (from the checks already worked out), with W2's help for the row that
    fired. Returns the Set up button's label, or None."""
    checks = checks_for(opt, "trust")
    rounds = insights.headline(opt, checks).rounds if checks else None
    text, change = next_step_text(opt, facts.next, rounds, amounts_shown(facts))
    _line(opt, text, "next_reasoning", facts=facts)
    return change


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


def _labels(chart, x, y, text, **mark):
    """A single grey text label, moved clear of the mark it names by a `dy`
    offset (and `dx`, from the caller, when the mark also needs to move
    sideways) rather than a white halo: a halo glows in dark mode and
    duplicates the text for screen readers. Carries no tooltip."""
    mark.setdefault("dy", -14)
    return [chart.mark_text(color=muted(), tooltip=False, **mark).encode(x=x, y=y, text=text)]


def _xy(x):
    """Barycentric (x1, x2, x3) to the page: corner 1 bottom left, corner 2
    bottom right, corner 3 at the top."""
    return x[1] + x[2] / 2, x[2] * _TRI_H


def pair_chart(opt, a, b, points):
    """Can you have both? Measured values only: one dot per formulation with
    both values, coloured by its Overall score on 0 to the ceiling; one still
    waiting for results is a hollow grey dot; Best so far is ringed and
    named; a target is a dashed line."""
    if not points:
        return None
    objs = {o['name']: o for o in opt.objectives}
    oa, ob = objs[a], objs[b]
    xt = _say(opt, wording.pair_axis(opt._measurement_column(oa), oa.get('goal'),
                                     oa.get('target')))
    yt = _say(opt, wording.pair_axis(opt._measurement_column(ob), ob.get('goal'),
                                     ob.get('target')))
    overall = _say(opt, wording.OVERALL_SCORE_COLUMN)

    def domain(o, values):
        return [min(float(o['min_val']), *values), max(float(o['max_val']), *values)]

    waiting = _say(opt, wording.PAIR_WAITING)
    def score_text(p):
        if p.score is not None:
            return f"{p.score:.1f}"
        i = opt.index_of_formulation(p.formulation)
        return wording.waiting_for(number_list(opt.waiting_on(i) if i is not None else []))
    df = pd.DataFrame([{"x": p.a, "y": p.b, "no": p.formulation,
                        "score": None if p.score is None else round(p.score, 1),
                        "score_text": score_text(p),
                        "waiting": p.score is None, "state": waiting, "best": p.best}
                       for p in points])
    x = alt.X("x:Q", title=xt, scale=alt.Scale(domain=domain(oa, list(df["x"])), nice=False))
    y = alt.Y("y:Q", title=yt, scale=alt.Scale(domain=domain(ob, list(df["y"])), nice=False))
    tip = [alt.Tooltip("no:Q", title=_say(opt, wording.FORMULATION_CAP)),
           alt.Tooltip("x:Q", title=_say(opt, opt._measurement_column(oa)),
                       format=f".{decimals(opt, a)}f"),
           alt.Tooltip("y:Q", title=_say(opt, opt._measurement_column(ob)),
                       format=f".{decimals(opt, b)}f"),
           # Words, never "null": a formulation still waiting says what for.
           alt.Tooltip("score_text:N", title=overall)]
    base = alt.Chart(df)
    layers = []
    # W3's D1 tokens and scale (integration of the e2e fixes): the old
    # blues top was about 2.2:1 on the dark page.
    grey, grey_opacity = mark("neutral")
    for o, enc in ((oa, "x"), (ob, "y")):
        if _target_line(o):
            layers.append(alt.Chart(pd.DataFrame({"t": [float(o['target'])]}))
                          .mark_rule(strokeDash=[4, 3], strokeWidth=1, color=grey,
                                     opacity=grey_opacity, tooltip=False)
                          .encode(**{enc: alt.X("t:Q") if enc == "x" else alt.Y("t:Q")}))
    # A hollow grey dot is named in the legend, only when there is one: a
    # one-entry stroke legend under the Overall score colour bar.
    hollow = {}
    if df["waiting"].any():
        hollow["stroke"] = alt.Stroke("state:N",
                                      scale=alt.Scale(domain=[waiting], range=[grey]),
                                      legend=alt.Legend(title=None, labelLimit=0))
    layers.append(base.mark_point(filled=False, color=grey, opacity=grey_opacity, size=60)
                  .encode(x=x, y=y, tooltip=tip, **hollow)
                  .transform_filter(alt.FieldEqualPredicate(field="waiting", equal=True)))
    layers.append(base.mark_circle(size=70, opacity=1)
                  .encode(x=x, y=y, tooltip=tip,
                          color=alt.Color("score:Q", title=overall,
                                          scale=alt.Scale(scheme=blues(),
                                                          domain=[0, opt.utility_ceiling()])))
                  .transform_filter(alt.FieldEqualPredicate(field="waiting", equal=False)))
    best = base.transform_filter(alt.FieldEqualPredicate(field="best", equal=True))
    layers.append(best.mark_point(filled=False, size=260, strokeWidth=2, color=grey,
                                  opacity=grey_opacity)
                  .encode(x=x, y=y, tooltip=tip))
    layers += _labels(best, x, y, alt.value(_say(opt, wording.PAIR_BEST)), align="left", dx=12)
    return alt.layer(*layers).properties(height=320)


def _target_line(o):
    """Whether "Can you have both?" draws a target line for this measurement."""
    return o.get('goal') == 'target' and o.get('target') is not None


def pair_caption(opt, a, b):
    """The chart's one caption line, by how many of the two have a target
    line drawn (two, one or none)."""
    objs = {o['name']: o for o in opt.objectives}
    return wording.pair_caption([n for n in (a, b) if _target_line(objs[n])])


def _mixture_kinds(opt):
    """What a grid point is, in the legend and the tooltip: tested near
    here, not tested near here, not allowed by your limits."""
    return [_say(opt, k) for k in (wording.MIXTURE_TESTED, wording.MIXTURE_UNTESTED,
                                   wording.MIXTURE_BLOCKED)]


def mixture_frame(opt, mix, preds, values=None, outline=None):
    """One row per grid point: its place on the page, what it is (tested
    near here, not tested near here, not allowed) and its tooltip.
    `values` (a layer's number at each point, None where blocked) replaces
    the predicted score as the colour; `outline` marks the points Predicted
    above Best so far (BO views §3.3)."""
    unit = mix.units[0] if len(set(mix.units)) == 1 else ""
    cols = [opt._amount_column(n) for n in mix.names]
    tested, untested, blocked = _mixture_kinds(opt)
    rows = []
    values = values or [None] * len(mix.points)
    outline = outline or [False] * len(mix.points)
    for p, pred, v, ring in zip(mix.points, preds or [None] * len(mix.points), values, outline):
        X, Y = _xy(p.x)
        state = blocked if p.broken else tested if p.tested else untested
        score = None if pred is None else pred.mean
        text = "" if pred is None else score_text(pred)
        if v is not None:
            score, text = v, f"{v:.2f}"
        row = {"X": X, "Y": Y, "state": state, "score": score,
               "a0": round(p.amounts[0], 2), "a1": round(p.amounts[1], 2),
               "a2": round(p.amounts[2], 2), "sum": round(sum(p.amounts), 2),
               "pred": text, "why": p.broken or "", "outline": bool(ring)}
        rows.append(row)
    return pd.DataFrame(rows), cols, unit


def mixture_chart(opt, mix, preds, number, start_pred=None, layer=None, values=None,
                  outline=None):
    """What happens when three ingredients trade places? Predicted Overall
    scores over the part of the triangle the limits allow, zoomed to it
    (0.8.0 review fixes §5.1): one hue, darker is higher; a smaller mark
    where nothing was tested near; light grey where a limit is broken.
    Tested formulations are white dots and the start is a grey ring named
    by its number. No axes and no whole-triangle outline: each label names
    an ingredient and its largest amount in the drawn region, written where
    the region reaches it. `start_pred`, the start's own predicted Overall
    score, goes in the start ring's tooltip."""
    if not mix.points or not preds:
        return None
    df, cols, unit = mixture_frame(opt, mix, preds, values, outline)
    kinds = _mixture_kinds(opt)
    formulation = _say(opt, wording.FORMULATION_CAP)
    # The Predicted score layer's colour is the model mean, not an interval
    # (02 M8, 03 M5; review I13).
    predicted = _say(opt, {wording.LAYER_SD: wording.LAYER_SD_LEGEND,
                           wording.LAYER_GAIN: wording.LAYER_GAIN_LEGEND}.get(
                               layer, wording.LAYER_SCORE_LEGEND))
    # A tooltip under Predicted score reads "78 (70–85)", with its interval:
    # it keeps the column's name; the legend names the colour alone.
    tip_title = (wording.GAIN_TOOLTIP_TITLE if layer == wording.LAYER_GAIN
                 else predicted if layer == wording.LAYER_SD
                 else _say(opt, wording.PREDICTED_SCORE_COLUMN))
    allowed = df[df["state"] != kinds[2]]["score"]
    lo, hi = (float(allowed.min()), float(allowed.max())) if len(allowed) else (0.0, 1.0)
    # Zoomed to the drawn region (0.8.0 review fixes §5.1): the domains are
    # its extent plus a margin for the labels, and the chart's width and
    # height follow them, so a unit is the same length both ways.
    (x0, x1), (y0, y1) = mixture_extent(df)
    side = max(x1 - x0, y1 - y0, 1e-9)
    pad = _MIX_PAD * side
    x_dom, y_dom = [x0 - pad, x1 + pad], [y0 - pad, y1 + pad]
    px = _MIX_SIZE / (side + 2 * pad)
    width, height = round((x_dom[1] - x_dom[0]) * px), round((y_dom[1] - y_dom[0]) * px)
    x = alt.X("X:Q", axis=None, scale=alt.Scale(domain=x_dom, nice=False, zero=False))
    y = alt.Y("Y:Q", axis=None, scale=alt.Scale(domain=y_dom, nice=False, zero=False))
    # A grid point's mark fills about the same share of the gap to its
    # neighbour however far the region is zoomed: 70 px² for a step of
    # _MIX_SIZE / GRID_BASE px. The pad takes part of _MIX_SIZE, so the
    # whole triangle is drawn about 306 px wide and its marks are about
    # 46 px²; a zoomed region's are larger in proportion.
    step_px = px / mix.steps
    big = max(8, round(70 * (step_px / (_MIX_SIZE / GRID_BASE)) ** 2))
    small = max(3, round(big * 16 / 70))
    # An explicit ordinal type: without it Vega-Lite compiles a "point"
    # scale for a size range, which only honours the range's start and
    # stop and interpolates the middle entry instead of using its own
    # value (16 draws as ~43).
    size_scale = alt.Scale(domain=kinds, range=[big, small, small], type="ordinal")
    # A size-only channel, shared by both layers: the smaller mark says
    # "not tested near here" without fading a colour that already means a
    # score. Its legend swatches get their fill from state_legend below —
    # Vega-Lite merges a size legend and a colour legend that share the
    # same field into one, so the entries read larger blue, small blue,
    # small grey instead of one flat grey circle.
    size = alt.Size("state:N", title=None, sort=kinds, scale=size_scale)
    amounts = ([alt.Tooltip(f"a{k}:Q", title=cols[k]) for k in range(3)]
               + [alt.Tooltip("sum:Q", title=_say(opt, wording.mixture_together(unit)))])
    blocked = alt.FieldEqualPredicate(field="state", equal=kinds[2])
    # Charts proto: the not-allowed points are hollow rings in the neutral
    # token (3.2:1 on white, 6.0:1 on the dark page); lightgray was 1.5:1
    # on white.
    no_c, no_o = mark("neutral")
    grey = (alt.Chart(df).mark_point(opacity=no_o, color=no_c, filled=False, strokeWidth=1)
            .encode(x=x, y=y, size=size,
                    tooltip=amounts + [alt.Tooltip("why:N", title=kinds[2])])
            .transform_filter(blocked))
    grid = (alt.Chart(df).mark_circle(opacity=1)
            .encode(x=x, y=y, size=size,
                    # Charts proto: the score layer in the score colours
                    # (as What you've tried); Uncertainty and Expected
                    # improvement in blues, the model's colour.
                    color=alt.Color("score:Q", title=predicted,
                                    scale=(alt.Scale(scheme=blues(), domain=[lo, hi])
                                           if layer in (wording.LAYER_SD, wording.LAYER_GAIN)
                                           else score_scale([lo, hi])),
                                    legend=alt.Legend(titleLimit=0)),
                    tooltip=amounts + [alt.Tooltip("pred:N", title=tip_title)])
            .transform_filter(alt.LogicalNotPredicate(**{"not": blocked})))
    # An invisible layer, off the drawn area, whose only job is the legend:
    # sharing the "state" field with `size` above merges the two into one
    # legend, so each entry shows both its size and its true colour — blue
    # for tested and untested, light grey for not allowed. `fill`/`stroke`
    # rather than `color`, so the merge is not confused by the grid's own
    # colour scale (the continuous predicted score) sharing the same view.
    # Its rows sit outside the X/Y domains (so they can never be mistaken
    # for real data); `clip=True` keeps Vega from sizing the canvas around
    # them anyway, which "pad" autosize would otherwise do even at opacity 0.
    swatch = (model_blue() if layer in (wording.LAYER_SD, wording.LAYER_GAIN)
              else score_colours()[len(score_colours()) // 2])
    state_colors = alt.Scale(domain=kinds, range=[swatch, swatch, no_c])
    legend_rows = pd.DataFrame([{"X": -1.0, "Y": -1.0, "state": k} for k in kinds])
    state_legend = (alt.Chart(legend_rows).mark_circle(opacity=0, tooltip=False, clip=True)
                    .encode(x=x, y=y,
                            size=alt.Size("state:N", sort=kinds, scale=size_scale),
                            fill=alt.Fill("state:N", sort=kinds, scale=state_colors,
                                         legend=alt.Legend(title=None, labelLimit=0)),
                            stroke=alt.Stroke("state:N", sort=kinds, scale=state_colors)))
    # Each one's largest amount inside the region, U*ᵢ where it can reach
    # it — never an amount past its Highest — written where the region
    # reaches it. The three share one number of decimals, the most any of
    # them needs, up to 2, so the labels never mix "10.0 g" and "9 g".
    corner_decimals = _shared_decimals(mix.top)

    def corner(k):
        return _say(opt, wording.mixture_corner(
            mix.names[k], join_unit(f"{mix.top[k]:.{corner_decimals}f}", mix.units[k])))
    places = _corner_places(mix, pad, corner)
    text = [alt.Chart(pd.DataFrame([row]))
            .mark_text(align=row["align"], baseline="middle", color=muted(), tooltip=False)
            .encode(x=x, y=y, text="t:N") for row in places]
    layers = [grey, grid, state_legend]
    if df["outline"].any():
        # Predicted above Best so far (review I6, M14): an outline, not a
        # fill, so the colours under it still read.
        # Charts proto: orange is for suggestions alone; the ring is the
        # ink token, drawn a size larger than the dot so it sits on the page.
        layers.append(alt.Chart(df).mark_circle(filled=False, stroke=mark("bar_latest")[0],
                                                strokeWidth=1.5,
                                                size=round(big * 1.9), opacity=1, tooltip=False)
                      .encode(x=x, y=y)
                      .transform_filter(alt.FieldEqualPredicate(field="outline", equal=True)))
    if mix.dots:
        dots = pd.DataFrame([{**dict(zip(("X", "Y"), _xy(d.x))), "no": d.formulation}
                             for d in mix.dots])
        # The neutral token (polish round: 3:1 or better in both themes).
        layers.append(alt.Chart(dots).mark_point(filled=True, fill="white",
                                                 stroke=mark("neutral")[0],
                                                 strokeWidth=1.5, size=45, opacity=1)
                      .encode(x=x, y=y, tooltip=[alt.Tooltip("no:Q", title=formulation)]))
    s_amounts = [float(mix.start.get(n, 0.0)) for n in mix.names]
    s = [(s_amounts[k] - mix.lower[k]) / mix.span for k in range(3)]
    sx, sy = _xy(s)
    start = pd.DataFrame([{
        "X": sx, "Y": sy, "t": _say(opt, wording.start_formulation(number)), "no": number,
        "a0": round(s_amounts[0], 2), "a1": round(s_amounts[1], 2), "a2": round(s_amounts[2], 2),
        "sum": round(sum(s_amounts), 2),
        "pred": "" if start_pred is None else score_text(start_pred)}])
    own = [] if start_pred is None else [alt.Tooltip("pred:N", title=tip_title)]
    ring_c, ring_o = mark("neutral")
    layers.append(alt.Chart(start).mark_point(filled=False, size=260, strokeWidth=2.5,
                                              color=ring_c, opacity=ring_o)
                  .encode(x=x, y=y,
                          tooltip=[alt.Tooltip("no:Q", title=formulation)] + amounts + own))
    layers += _labels(alt.Chart(start), x, y, "t:N", align="left", dx=12)
    return (alt.layer(*layers, *text).properties(
                width=width, height=height,
                autosize=alt.AutoSizeParams(type="pad", contains="padding"))
            .configure_view(stroke=None))


GRID_BASE = insights.GRID_STEPS


def mixture_extent(df):
    """((x0, x1), (y0, y1)) of the grid points on the page."""
    return (float(df["X"].min()), float(df["X"].max())), (float(df["Y"].min()), float(df["Y"].max()))


def _corner_places(mix, pad, text):
    """Where each label goes: beside the grid point where that one is
    largest (ties: the middle of that edge), pushed outwards from the
    region's centre by `pad`, aligned away from it."""
    pts = [(p.x, _xy(p.x)) for p in mix.points]
    cx = sum(xy[0] for _, xy in pts) / len(pts)
    cy = sum(xy[1] for _, xy in pts) / len(pts)
    rows = []
    for k in range(3):
        most = max(x[k] for x, _ in pts)
        at = [xy for x, xy in pts if x[k] >= most - 1e-9]
        X = sum(a for a, _ in at) / len(at)
        Y = sum(b for _, b in at) / len(at)
        dx, dy = X - cx, Y - cy
        norm = math.hypot(dx, dy) or 1.0
        align = "left" if dx > 0.25 * norm else "right" if dx < -0.25 * norm else "center"
        rows.append({"X": X + 0.5 * pad * dx / norm, "Y": Y + 0.5 * pad * dy / norm,
                     "t": text(k), "align": align})
    return rows


def _pair(opt):
    names = insights.pair_choices(opt)
    if len(names) < 2:
        return False
    _question(opt, wording.Q_PAIR)
    _keep_all_offered(PAIR_KEY, names)
    if PAIR_KEY not in st.session_state:
        st.session_state[PAIR_KEY] = names[:2]
    picked = st.multiselect(_say(opt, wording.PAIR_PICK), names, key=PAIR_KEY,
                            max_selections=2)
    if len(picked) != 2:
        st.caption(_say(opt, wording.PICK_TWO))
        return True
    points = insights.pair_points(opt, *picked)
    if not points:
        st.caption(_say(opt, wording.PAIR_NONE))
        return True
    show_chart(pair_chart(opt, *picked, points))
    st.caption(_say(opt, pair_caption(opt, *picked)))
    return True


def _start_number(opt, layered=False):
    """Section 2's Start from, else Best so far, else the newest. For the
    Uncertainty and Expected gain layers (review I8): the layers' own
    Start from, else the open Round's first suggestion still to make."""
    ids = [int(x) for x in opt.formulation_ids]
    if layered:
        offered = suggested_numbers(opt)
        chosen = st.session_state.get(LAYER_START_KEY)
        if chosen in ids or chosen in offered:
            return chosen
        if offered:
            return offered[0]
    chosen = st.session_state.get(START_KEY)
    if chosen in ids:
        return chosen
    best = opt.best_index()
    return ids[best] if best is not None else ids[-1]


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


def anchor_text(opt, number):
    """The anchor as the caption names it: Best so far by that name (T56
    m7), a recorded formulation, or a suggestion still to make — "edited"
    when the user changed it (T56 m8)."""
    ids = [int(x) for x in opt.formulation_ids]
    best = opt.best_index()
    if best is not None and number == ids[best]:
        return wording.start_best(number)
    if number in ids:
        return wording.start_formulation(number)
    row = next((r for r in opt.pending_batch or [] if int(r['formulation']) == number), {})
    return wording.start_suggested(number, edited=bool(row.get('original_recipe')))


def _model_guard(opt, fitted):
    """The noise guard of the Overall-score display model (review I3), from
    the checks already worked out: what the charts' model layers obey."""
    score = (fitted or {}).get(insights.SCORE)
    return insights.noise_guard(opt, score, checks_for(opt, "trust") or ())


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


def _layer_pick(opt, fitted, key):
    """The one radio (review I6): Predicted score, Uncertainty, Expected
    improvement. Not drawn when only the first is offered."""
    offered = layers_offered(opt, fitted)
    if len(offered) == 1:
        return wording.LAYER_SCORE
    if st.session_state.get(key) == wording.LAYER_GAIN_FORMER:
        # A session from before the rename (review I6): the same layer.
        st.session_state[key] = wording.LAYER_GAIN
    # Explored and unexplored's pointer (§8.3) asked for Uncertainty: once
    # per radio, set before the radio is drawn.
    asked = st.session_state.get(LAYER_PRESELECT_KEY) or ()
    if key in asked:
        st.session_state[LAYER_PRESELECT_KEY] = tuple(k for k in asked if k != key)
        if wording.LAYER_SD in offered:
            st.session_state[key] = wording.LAYER_SD
    _keep_if_offered(key, offered)
    picked = st.radio(_say(opt, wording.LAYER_PICK), offered, key=key, horizontal=True)
    if picked == wording.LAYER_SD and wording.LAYER_GAIN not in offered and explore_withheld(
            opt, (fitted or {}).get(insights.SCORE), checks_for(opt, "trust") or ()):
        # R-I16: shown under the A3 gate or the verdict, with the map's caption.
        st.caption(_say(opt, wording.MAP_NOT_YET))
    return picked


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

# The strips' Round colours (§8.1): a ramp from the first Round to the
# latest, each stop at least 3:1 on its theme's page (WCAG 1.4.11), the
# latest the strongest. Light: #5b8ec9 is 3.4:1 on white, #0b2e59 13.6:1.
# Dark: #3d6fa8 is 3.6:1 on #0e1117, #dbe9f8 15.3:1.
_ROUND_RAMP = {"light": ("#5b8ec9", "#0b2e59"), "dark": ("#3d6fa8", "#dbe9f8")}
# The reachable part's shading: a region under the ticks, as _FILLS.
_REACH_FILL = {"light": ("#c6d4e6", 1.0), "dark": ("#2c3a4f", 1.0)}
# The marks' shapes: a tick (a vertical line), a taller one for the latest
# Round, a ring for waiting or Not scored, a triangle for a suggestion.
_TICK = "M0,-1L0,1"
_SHAPES = {wording.STRIP_TESTED: _TICK, wording.STRIP_LATEST: _TICK,
           wording.STRIP_WAITING: "circle", wording.STRIP_SUGGESTED: "triangle-down"}
_SIZES = {wording.STRIP_TESTED: 160, wording.STRIP_LATEST: 420,
          wording.STRIP_WAITING: 50, wording.STRIP_SUGGESTED: 70}


def _mix(a, b, t):
    """The colour a share `t` of the way from hex `a` to hex `b`."""
    ca = [int(a[k:k + 2], 16) for k in (1, 3, 5)]
    cb = [int(b[k:k + 2], 16) for k in (1, 3, 5)]
    return "#" + "".join(f"{round(x + t * (y - x)):02x}" for x, y in zip(ca, cb))


def round_colours(n, which=None):
    """`n` Round colours, the first Round's to the latest's (the latest the
    strongest), in the reader's theme or `which`."""
    a, b = _ROUND_RAMP[which or theme()]
    if n <= 1:
        return [b] * max(n, 0)
    return [_mix(a, b, k / (n - 1)) for k in range(n)]


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


def _strip_kind(tick):
    if tick.state == "suggested":
        return wording.STRIP_SUGGESTED
    if tick.state in ("waiting", "not_scored"):
        return wording.STRIP_WAITING
    return wording.STRIP_LATEST if tick.latest else wording.STRIP_TESTED


def _tick_score(opt, tick):
    """The tooltip's Overall score: two decimals, or what the formulation
    is waiting for, Not scored, or "This Round's suggestion"."""
    if tick.state == "tested":
        return f"{tick.score:.2f}"
    if tick.state == "suggested":
        return wording.STRIP_SUGGESTED
    if tick.state == "not_scored":
        return wording.NOT_SCORED
    i = opt.index_of_formulation(tick.formulation)
    return wording.waiting_for(number_list(opt.waiting_on(i) if i is not None else []))


def _strip_frame(opt, strip):
    return pd.DataFrame([{"value": t.value, "value_text": _strip_number(opt, strip, t.value),
                          "no": t.formulation, "round": _round_label(t.round),
                          "score_text": _tick_score(opt, t), "mark": _strip_kind(t)}
                         for t in strip.ticks],
                        columns=["value", "value_text", "no", "round", "score_text", "mark"])


def _strip_title(strip):
    return f"{strip.name} ({strip.unit})" if strip.unit else strip.name


def _option_chart(opt, strips, colour):
    """The choice rows: one bar per option, the times it was tested."""
    df = pd.DataFrame([{"row": s.name, "option": o, "n": n, "which": f"{s.name}: {o}"}
                       for s in strips for o, n in s.options.items()])
    tip = [alt.Tooltip("which:N", title=_say(opt, wording.INGREDIENT_OR_SETTING)),
           alt.Tooltip("n:Q", title=_say(opt, wording.STRIP_COUNT), format="d")]
    return (alt.Chart(df).mark_bar(color=colour, opacity=1)
            .encode(x=alt.X("n:Q", title=_say(opt, wording.STRIP_COUNT),
                            axis=alt.Axis(tickMinStep=1, format="d")),
                    y=alt.Y("option:N", title=None, sort=None),
                    row=alt.Row("row:N", title=None, sort=[s.name for s in strips],
                                header=alt.Header(labelAngle=0, labelAlign="left",
                                                  labelAnchor="start")),
                    tooltip=tip)
            .properties(height=alt.Step(18)))


def _share(strip, v):
    """Where `v` sits along the strip, 0 at its Lowest, 1 at its Highest."""
    span = strip.high - strip.low
    return 0.5 if span <= 0 else min(1.0, max(0.0, (float(v) - strip.low) / span))


def _numeric_strips(opt, strips, labels, colours):
    """The numeric strips as one chart (so it takes the column's width):
    one band per row, its Lowest–Highest across (written at its ends),
    the reachable part shaded, the formulations as marks."""
    rows = [_strip_title(s) for s in strips]
    present = {_strip_kind(t) for s in strips for t in s.ticks}
    kinds = [k for k in _SHAPES if k in present]
    shape = alt.Scale(domain=kinds or [wording.STRIP_TESTED],
                      range=[_SHAPES[k] for k in kinds] or [_TICK])
    size = alt.Scale(domain=kinds or [wording.STRIP_TESTED],
                     range=[_SIZES[k] for k in kinds] or [_SIZES[wording.STRIP_TESTED]])
    stroke = alt.Scale(domain=labels or [""], range=colours or [mark("neutral")[0]])
    grey, grey_o = mark("neutral")
    fill, fill_o = _REACH_FILL[theme()]
    value_title = _say(opt, wording.strip_value_title(not opt.has_ingredients()))
    tips = [alt.Tooltip("value_text:N", title=value_title),
            alt.Tooltip("no:Q", title=_say(opt, wording.FORMULATION_CAP), format="d"),
            alt.Tooltip("round:N", title=wording.ROUND_CAP),
            alt.Tooltip("score_text:N", title=_say(opt, wording.OVERALL_SCORE_COLUMN))]
    y = alt.Y("row:N", sort=rows, title=None,
              scale=alt.Scale(paddingInner=0.25, paddingOuter=0.1),
              axis=alt.Axis(ticks=False, domain=False, labelLimit=260, labelFontSize=12))
    x = alt.X("t:Q", scale=alt.Scale(domain=[-0.01, 1.01], nice=False, zero=False), axis=None)
    bands, ends, made, picks = [], [], [], []
    for s, row in zip(strips, rows):
        lo, hi = s.reach if s.reach is not None else (s.low, s.high)
        number = lambda v, _s=s: _strip_number(opt, _s, v)
        bands.append({"row": row, "t": _share(s, lo), "t2": _share(s, hi),
                      "reach": range_text(lo, hi, lambda v, _s=s: _strip_number(opt, _s, v, False))
                      + (f" {s.unit}" if s.unit else "")})
        ends += [{"row": row, "t": 0.0, "label": number(s.low), "end": "low"},
                 {"row": row, "t": 1.0, "label": number(s.high), "end": "high"}]
        for _, r in _strip_frame(opt, s).iterrows():
            item = dict(r, row=row, t=_share(s, r["value"]))
            (picks if r["mark"] == wording.STRIP_SUGGESTED else made).append(item)
    bands = pd.DataFrame(bands)
    ends = pd.DataFrame(ends)
    muted_c = _MUTED_DARK if theme() == "dark" else _MUTED_LIGHT
    layers = [
        alt.Chart(bands).mark_bar(height=12, color=fill, opacity=fill_o)
        .encode(x=x, x2="t2:Q", y=y,
                tooltip=[alt.Tooltip("reach:N", title=_say(opt, wording.STRIP_REACHABLE))]),
        alt.Chart(bands.assign(t=0.0, t2=1.0)).mark_bar(height=2, color=grey, opacity=grey_o,
                                                        tooltip=False)
        .encode(x=x, x2="t2:Q", y=y),
        alt.Chart(ends[ends["end"] == "low"]).mark_text(
            align="left", baseline="top", dy=12, fontSize=10, color=muted_c, tooltip=False)
        .encode(x=x, y=y, text="label:N"),
        alt.Chart(ends[ends["end"] == "high"]).mark_text(
            align="right", baseline="top", dy=12, fontSize=10, color=muted_c, tooltip=False)
        .encode(x=x, y=y, text="label:N"),
    ]
    if made:
        layers.append(
            alt.Chart(pd.DataFrame(made)).mark_point(filled=False, strokeWidth=2, opacity=1)
            .encode(x=x, y=y,
                    shape=alt.Shape("mark:N", scale=shape,
                                    legend=alt.Legend(title=None, orient="bottom",
                                                      labelLimit=0, symbolStrokeWidth=2,
                                                      symbolStrokeColor=muted_c)),
                    size=alt.Size("mark:N", scale=size, legend=None),
                    stroke=alt.Stroke("round:N", scale=stroke,
                                      legend=alt.Legend(title=wording.ROUND_CAP, orient="bottom",
                                                        direction="horizontal",
                                                        columns=min(12, max(1, len(labels))))),
                    tooltip=tips))
    if picks:
        layers.append(
            alt.Chart(pd.DataFrame(picks)).mark_point(filled=True, color=outline_colour(),
                                                      opacity=1, yOffset=-11)
            .encode(x=x, y=y, shape=alt.Shape("mark:N", scale=shape),
                    size=alt.Size("mark:N", scale=size, legend=None), tooltip=tips))
    return alt.layer(*layers).properties(height=alt.Step(46))


def strips_chart(opt, strips):
    """W3 (§8.1, review R-I15): one horizontal strip per insights.Strip, its
    Lowest–Highest across, the reachable part shaded; a tick per recorded
    formulation at its amount, coloured by Round (the latest the strongest
    and tallest), a ring while it waits or is Not scored, a triangle above
    for each of the open Round's suggestions; the choice rows as count
    bars. Every colour at least 3:1 on its theme's page. Drawn through
    show_chart by the caller. None without a strip."""
    if not strips:
        return None
    rounds = sorted({t.round for s in strips for t in s.ticks if t.state != "suggested"},
                    key=lambda r: (r is not None, r or 0))
    labels = [_round_label(r) for r in rounds]
    colours = round_colours(len(labels))
    numeric = [s for s in strips if s.options is None]
    choices = [s for s in strips if s.options is not None]
    charts = []
    if numeric:
        charts.append(_numeric_strips(opt, numeric, labels, colours))
    if choices:
        charts.append(_option_chart(opt, choices, round_colours(1)[0]))
    if len(charts) == 1:
        return charts[0]
    return alt.vconcat(*charts, spacing=16).resolve_scale(x="independent", y="independent")


def explore_map_chart(pool_mean, pool_sd, tested, picks, ref=None, opt=None):
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
    tested = pd.DataFrame(list(tested), columns=None if tested else ["mean", "sd"])
    mine = pd.DataFrame(list(picks), columns=None if picks else ["mean", "sd", "no"])
    x = alt.X("mean:Q", title=wording.WHY_AXIS_MEAN, scale=alt.Scale(zero=False))
    y = alt.Y("sd:Q", title=wording.MODEL_SD, scale=alt.Scale(zero=False))
    word = wording.FORMULATION_CAP if opt is None else _say(opt, wording.FORMULATION_CAP)
    ring_c, ring_o = mark("neutral")
    line = pd.DataFrame({"ref": [ref], "label": [wording.WHY_REFERENCE]})
    rule = [] if ref is None else [
        alt.Chart(line).mark_rule(strokeDash=[4, 3], color="gray", tooltip=False)
        .encode(x="ref:Q"),
        alt.Chart(line).mark_text(align="left", baseline="top", dx=4, color=muted(),
                                  tooltip=False)
        .encode(x="ref:Q", y=alt.value(4), text="label:N")]
    numbered = "no" in tested.columns
    ring_tip = ({"tooltip": [alt.Tooltip("no:Q", title=word, format="d"),
                             alt.Tooltip("mean:Q", title=wording.WHY_AXIS_MEAN, format=".1f"),
                             alt.Tooltip("sd:Q", title=wording.MODEL_SD, format=".2f")]}
                if numbered else {})
    layers = [
        alt.Chart(pool).mark_circle(size=12, color="lightgray", opacity=0.6, tooltip=False)
        .encode(x=x, y=y),
        # The neutral token (polish round): gray at Vega's default 0.7 was
        # 2.94:1 on the dark page.
        alt.Chart(tested).mark_point(size=40, color=ring_c, opacity=ring_o, filled=False,
                                     **({} if numbered else {"tooltip": False}))
        .encode(x=x, y=y, **ring_tip),
        *rule,
        # Charts proto: orange for suggestions, as on every chart here.
        alt.Chart(mine).mark_point(shape="diamond", size=90, filled=True,
                                   color=outline_colour(), opacity=1)
        .encode(x=x, y=y, tooltip=[alt.Tooltip("no:N", title=word),
                                   alt.Tooltip("mean:Q", title=wording.WHY_AXIS_MEAN,
                                               format=".1f"),
                                   alt.Tooltip("sd:Q", title=wording.MODEL_SD, format=".2f")]),
        alt.Chart(mine).mark_text(dy=-12, color=muted(), tooltip=False)
        .encode(x=x, y=y, text="no:N"),
    ]
    return alt.layer(*layers).properties(height=280)


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


def _outline_model(opt, fitted):
    """The model under the triangle's Predicted score outline, drawn on
    every render: the open Round's (layer_model); with no Round open, the
    display model of today's rows standing in for the next Generate's, as
    before integration, so a reopen fits nothing (04 I-1; tests/
    test_104_speed_robust_charts). The Uncertainty and Expected gain layers
    and the map, drawn on request, read layer_model's next_model."""
    if opt.pending_batch and opt.pending_batch_no is not None:
        return layer_model(opt, fitted)
    rows = [i for i in range(len(opt.Y_history)) if opt.fully_measured(i)]
    gen = bo_views.Generation(None, int(opt.next_formulation_no), rows,
                              "warm" if len(rows) >= insights.MIN_ROWS else "cold", False)
    with insights.memo_in_use(memo(opt)):
        return bo_views.rebuild(opt, gen, (fitted or {}).get(insights.SCORE))


def _map_model(opt, fitted):
    """The suggestion model the map shows (review R-I16): layer_model's,
    the one the layers show; with a Round open whose model cannot be
    rebuilt (a cold Round), the one the next Generate fits. None while
    cold."""
    model = layer_model(opt, fitted)
    if model is None and opt.pending_batch:
        return bo_views.next_model(opt)
    return model


def _explored_map(opt, computed):
    """§8.2: the map, its plain title and reading, under the withhold rule
    of R-I16."""
    if opt.complete_count() < insights.MIN_ROWS:
        st.caption(_say(opt, wording.MAP_COLD))
        return
    if computed is None:
        # The Prediction check is still being calculated: the rule waits for it.
        st.caption(_say(opt, wording.CHECKING_PREDICTIONS))
        return
    if computed.get("failed"):
        st.caption(_say(opt, wording.CHART_FAILED))
        return
    fitted = computed["fitted"]
    score = fitted.get(insights.SCORE)
    rule = explore_withheld(opt, score, computed["checks"])
    if rule == "hidden":
        st.caption(_say(opt, wording.MAP_HIDDEN), help=tooltip_for(opt, "map"))
        return
    values = bo_views.map_values(opt, suggestion_model(opt, fitted))
    if values is None or values.pool_mean is None:
        st.caption(_say(opt, wording.CHART_FAILED))
        return
    try:
        ref = repeat_of_best(opt, fitted)
    except Exception:
        ref = None
    tested = [{"mean": float(m), "sd": float(s), "no": n}
              for m, s, n in zip(values.tested_mean.tolist(), values.tested_sd.tolist(),
                                 values.tested)]
    picks = [{"mean": m, "sd": s, "no": str(n)} for n, m, s in values.picks]
    help_text = tooltip_for(opt, "map")
    st.markdown(f"**{_say(opt, wording.MAP_TITLE)}**")
    st.caption(_say(opt, wording.MAP_READING), help=help_text)
    if rule == "not_yet":
        st.caption(_say(opt, wording.MAP_NOT_YET))
    show_chart(explore_map_chart(values.pool_mean, values.pool_sd, tested, picks, ref, opt))
    if picks:
        st.caption(_say(opt, wording.why_chart_caption(f"{len(values.pool_mean):,}",
                                                       line=ref is not None)),
                   help=_say(opt, wording.why_chart_caption_full(
                       f"{len(values.pool_mean):,}", line=ref is not None)))


def _point_to_uncertainty():
    """The pointer's on_click: Try one opens, both layer radios open on
    Uncertainty the next time each is drawn (_layer_pick), the page lands
    on Try one (ui_helpers.ask_tab)."""
    st.session_state[LAYER_PRESELECT_KEY] = (PATH_LAYER_KEY, LAYER_KEY)
    ask_tab(wording.TAB_INSIGHTS, section=TRY_ONE_KEY, fold=TRY_ONE_KEY)


def _explored_pointer(opt, computed):
    """§8.3: one line to the in-context views, a link, shown only when the
    Uncertainty layer is offered (review R-I16)."""
    if computed is None or computed.get("failed") or not insights.path_choices(opt):
        return
    if wording.LAYER_SD not in layers_offered(opt, computed["fitted"]):
        return
    # Charts proto: the triangle is in this fold now, above the map.
    text = wording.explored_pointer(_try_one_label(opt), settings_only=not opt.has_ingredients(),
                                    triangle=False)
    tip = wording.explored_pointer_help(_try_one_label(opt),
                                        settings_only=not opt.has_ingredients(), triangle=False)
    st.button(_say(opt, text), type="tertiary", key=POINTER_KEY, help=_say(opt, tip),
              disabled=confirmation_open(), on_click=_point_to_uncertainty)


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
         section=EXPLORED_PLACE, fold=EXPLORED_KEY, key=EXPLORED_LINK_KEY)
    return True


def _explored_body(opt, computed):
    values = not opt.has_ingredients()
    total = opt.has_formulation_total()
    tip = tooltip_for(opt, "basis_explored")
    st.caption(_say(opt, wording.basis_charts(values, total)), help=tip)
    # Charts proto: 1. What you've tried (parallel coordinates; model-free,
    # drawn whatever the checks say), 2. How the app chooses, 3. the
    # triangle, 4. the map, then the pointer.
    _tried(opt)
    _chooses(opt, computed)
    _triangle(opt, computed)
    _explored_map(opt, computed)
    _explored_pointer(opt, computed)


def explored_fold(opt, computed):
    """W3 (§8, review R-I14 to R-I16): "Explored and unexplored", one fold
    row under the button and its one link, above the detail sections, keyed
    EXPLORED_KEY, collapsed, and nothing in it worked out until it is
    opened (as the Why fold). It holds the strips (model-free), the map and
    the pointer. `computed`: render's, or None while the checks are still
    being calculated (and at the gate)."""
    # In a keyed place, the Explored link's landing (ftue fix C).
    with st.container(key=EXPLORED_PLACE), \
            st.expander(_say(opt, wording.FOLD_EXPLORED), key=EXPLORED_KEY,
                        on_change="rerun") as fold:
        if not fold.open:
            return
        try:
            _explored_body(opt, computed)
        except Exception:   # one fold failing must not take the tab down
            st.caption(_say(opt, wording.CHART_FAILED))


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
# The model's 95 % band (a region: not held to 3:1, as _FILLS) and the
# expected gain's line and area. Blue means "the model's uncertainty or
# gain" on every chart here (the triangle's Uncertainty and Expected
# improvement layers use blues()). Light #2f6db5 is 5.2:1 on white; dark
# #7fb0e6 is 8.3:1 on #0e1117.
_MODEL_BLUE = {"light": "#2f6db5", "dark": "#7fb0e6"}
_BAND_FILL = {"light": ("#2f6db5", 0.18), "dark": ("#7fb0e6", 0.22)}
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


def _line_kind(line):
    if line.state == "suggested":
        return wording.TRIED_SUGGESTED
    if line.state != "tested":
        return wording.TRIED_WAITING
    return wording.TRIED_BEST if line.best else wording.TRIED_SCORED


def _line_score_text(opt, line):
    if line.state == "tested":
        return f"{line.score:.2f}"
    if line.state == "suggested":
        return wording.TRIED_SUGGESTED
    if line.state == "not_scored":
        return wording.NOT_SCORED
    i = opt.index_of_formulation(line.formulation)
    return wording.waiting_for(number_list(opt.waiting_on(i) if i is not None else []))


def tried_frame(opt, axes, lines):
    """(long frame, axis frame): one row per line per axis it crosses, at
    its share of that axis (0 Lowest, 1 Highest; the Overall score's axis
    0 to the ceiling) — a line without a score stops at the last row's
    axis; one row per axis for its rule and its end labels."""
    ceiling = float(opt.utility_ceiling())
    score_label = _say(opt, wording.OVERALL_SCORE_COLUMN)
    labels = [_wrap(_axis_label(a)) for a in axes] + [score_label]
    rows = []
    for order, line in enumerate(lines):
        amounts = "; ".join(
            f"{a.name} {line.values[a.name]}" if a.options is not None
            else f"{a.name} {_axis_end(opt, a, line.values[a.name])}" for a in axes)
        common = {"no": line.formulation, "round": _round_label(line.round),
                  "role": _line_kind(line), "best": line.best,
                  "score": None if line.score is None else round(line.score, 2),
                  "score_text": _line_score_text(opt, line), "amounts": amounts,
                  "order": order}
        for k, a in enumerate(axes):
            t = insights.axis_share(a, line.values[a.name])
            if t is not None:
                rows.append({**common, "axis": labels[k], "k": k, "t": t})
        if line.score is not None:
            rows.append({**common, "axis": score_label, "k": len(axes),
                         "t": line.score / ceiling if ceiling else 0.0})
    ends = []
    for k, a in enumerate(axes):
        if a.options is not None:
            for o in a.options:
                ends.append({"axis": labels[k], "t": insights.axis_share(a, o), "text": o,
                             "where": "option"})
            continue
        d = _shared_decimals([a.low, a.high])
        ends += [{"axis": labels[k], "t": 0.0, "text": _axis_end(opt, a, a.low, d),
                  "where": "low"},
                 {"axis": labels[k], "t": 1.0, "text": _axis_end(opt, a, a.high, d),
                  "where": "high"}]
    ends += [{"axis": score_label, "t": 0.0, "text": "0", "where": "low"},
             {"axis": score_label, "t": 1.0, "text": _whole(ceiling), "where": "high"}]
    cols = ["no", "round", "role", "best", "score", "score_text", "amounts", "order", "axis",
            "k", "t"]
    return pd.DataFrame(rows, columns=cols), pd.DataFrame(ends), labels


def tried_chart(opt, axes, lines):
    """What you've tried: parallel coordinates. One vertical axis per
    varying ingredient and process setting, its Lowest at the bottom and
    its Highest at the top (written at the ends, with the unit), a choice's
    options spaced evenly; the last axis the Overall score, 0 to the
    ceiling. Each scored formulation is a line coloured by its Overall score
    (score_colours); waiting or Not scored, grey dashed; this Round's
    suggestions, bold orange, ending before the score; Best so far thick.
    None without an axis or a line."""
    if not axes or not lines:
        return None
    df, ends, labels = tried_frame(opt, axes, lines)
    many = len(lines) > _MANY_LINES
    x = alt.X("axis:N", sort=labels, title=None,
              scale=alt.Scale(type="point", padding=0.25),
              axis=alt.Axis(orient="top", labelAngle=0, labelPadding=22, ticks=False,
                            domain=False, labelLimit=0, labelFontSize=11,
                            labelExpr="split(datum.label, '\\n')"))
    lo = min([0.0] + list(df["t"])) if len(df) else 0.0
    hi = max([1.0] + list(df["t"])) if len(df) else 1.0
    t_scale = alt.Scale(domain=[lo, hi], nice=False, zero=False)
    y = alt.Y("t:Q", axis=None, scale=t_scale)
    word = _say(opt, wording.FORMULATION_CAP)
    tips = [alt.Tooltip("no:Q", title=word, format="d"),
            alt.Tooltip("round:N", title=wording.ROUND_CAP),
            alt.Tooltip("score_text:N", title=_say(opt, wording.OVERALL_SCORE_COLUMN)),
            alt.Tooltip("amounts:N", title=_say(opt, wording.tried_amounts_title(
                not opt.has_ingredients())))]
    grey, grey_o = mark("neutral")
    ink, _ = mark("bar_latest")
    text_c = muted()
    scored = df[df["score"].notna()]
    dom = ([float(scored["score"].min()), float(scored["score"].max())] if len(scored)
           else [0.0, float(opt.utility_ceiling())])
    if dom[1] - dom[0] < 1e-9:
        dom = [dom[0] - 1, dom[1] + 1]
    kinds = [k for k in (wording.TRIED_BEST, wording.TRIED_WAITING, wording.TRIED_SUGGESTED)
             if k in set(df["role"])]
    look = {wording.TRIED_BEST: (ink, [1, 0]), wording.TRIED_WAITING: (grey, [4, 3]),
            wording.TRIED_SUGGESTED: (outline_colour(), [1, 0])}
    stroke = alt.Scale(domain=kinds or [""], range=[look[k][0] for k in kinds] or [grey])
    dash = alt.Scale(domain=kinds or [""], range=[look[k][1] for k in kinds] or [[1, 0]])
    kind_legend = alt.Legend(title=None, orient="bottom", labelLimit=0, symbolStrokeWidth=3,
                             symbolType="stroke")
    axes_df = pd.DataFrame({"axis": labels, "t0": 0.0, "t1": 1.0})
    layers = [
        alt.Chart(axes_df).mark_rule(color=grey, opacity=grey_o, strokeWidth=1.5, tooltip=False)
        .encode(x=x, y=alt.Y("t0:Q", scale=t_scale, axis=None), y2="t1:Q"),
    ]

    def lines_of(kind_filter, **mark_kw):
        return (alt.Chart(df).mark_line(**mark_kw)
                .encode(x=x, y=y, detail="no:N", order=alt.Order("k:Q"), tooltip=tips)
                .transform_filter(kind_filter))

    plain = alt.FieldOneOfPredicate(field="role", oneOf=[wording.TRIED_SCORED])
    layers.append(lines_of(plain, strokeWidth=1 if many else 1.6,
                           opacity=0.55 if many else 0.9)
                  .encode(color=alt.Color("score:Q", scale=score_scale(dom),
                                          title=_say(opt, wording.OVERALL_SCORE_COLUMN),
                                          legend=alt.Legend(orient="right",
                                                            gradientLength=160))))
    waiting = alt.FieldEqualPredicate(field="role", equal=wording.TRIED_WAITING)
    if wording.TRIED_WAITING in kinds:
        layers.append(lines_of(waiting, strokeWidth=1.5, opacity=1)
                      .encode(stroke=alt.Stroke("role:N", scale=stroke, legend=kind_legend),
                              strokeDash=alt.StrokeDash("role:N", scale=dash,
                                                        legend=kind_legend)))
    best = alt.FieldEqualPredicate(field="role", equal=wording.TRIED_BEST)
    if wording.TRIED_BEST in kinds:
        # A casing in the ink colour, the score colour inside it: thick, and
        # readable on either page whatever its score colour.
        layers.append(lines_of(best, strokeWidth=6, opacity=1)
                      .encode(stroke=alt.Stroke("role:N", scale=stroke, legend=kind_legend),
                              strokeDash=alt.StrokeDash("role:N", scale=dash,
                                                        legend=kind_legend)))
        layers.append(lines_of(best, strokeWidth=3, opacity=1)
                      .encode(color=alt.Color("score:Q", scale=score_scale(dom), legend=None)))
    picks = alt.FieldEqualPredicate(field="role", equal=wording.TRIED_SUGGESTED)
    if wording.TRIED_SUGGESTED in kinds:
        layers.append(lines_of(picks, strokeWidth=3, opacity=1)
                      .encode(stroke=alt.Stroke("role:N", scale=stroke, legend=kind_legend),
                              strokeDash=alt.StrokeDash("role:N", scale=dash,
                                                        legend=kind_legend)))
    text = lambda where, **kw: (alt.Chart(ends[ends["where"] == where])
                                .mark_text(color=text_c, fontSize=10, tooltip=False, **kw)
                                .encode(x=x, y=alt.Y("t:Q", scale=t_scale, axis=None),
                                        text="text:N"))
    layers += [text("low", baseline="top", dy=6), text("high", baseline="bottom", dy=-6)]
    if (ends["where"] == "option").any():
        layers.append(text("option", align="left", dx=5, baseline="middle"))
    return (alt.layer(*layers).resolve_scale(color="independent", stroke="shared")
            .properties(height=300))


def _tried(opt):
    """§1 of the fold: the title, the gap and limit-hit lines (as the
    strips had them), the parallel-coordinates chart, its caption."""
    st.markdown(f"**{_say(opt, wording.TRIED_TITLE)}**")
    pool = bo_views.feasible_pool(opt)
    strips = insights.tested_strips(opt, pool)
    for k, line in enumerate(strip_gap_lines(opt, strips)):
        st.markdown(_say(opt, line), help=tooltip_for(opt, "gaps") if k == 0 else None)
    for k, line in enumerate(limit_hit_lines(opt)):
        st.markdown(_say(opt, line),
                    help=tooltip_for(opt, "limit_hits") if k == 0 else None)
    chart = tried_chart(opt, insights.tried_axes(opt), insights.tried_lines(opt))
    if chart is None:
        st.caption(_say(opt, wording.TRIED_NOTHING))
        return
    show_chart(chart)
    settings_only = not opt.has_ingredients()
    st.caption(_say(opt, wording.tried_caption(settings_only)),
               help=_say(opt, wording.tip_tried(settings_only)))


# ---- How the app chooses ------------------------------------------ #

class Chooses:
    """What "How the app chooses" draws along one row through Best so far:
    the path, the suggestion model's mean and sd and its acquisition along
    it (bo_views.evaluate), the formulations tested near it, where Best so
    far and this Round's first suggestion sit, and why the gain is
    withheld (None when it is drawn)."""

    def __init__(self, path, values, tested, best_no, pick, reason, line=False,
                 end_label=None, end_is_pick=False):
        self.path, self.values, self.tested = path, values, tested
        self.best_no, self.pick, self.reason = best_no, pick, reason
        # The straight-line form (charts proto follow-up): x is t, 0 at Best
        # so far, 1 at `end_label` (Suggestion 1, else Predicted best).
        self.line, self.end_label, self.end_is_pick = line, end_label, end_is_pick


CHOOSES_MODE_KEY = "insights_chooses_mode"
CHOOSES_LINE, CHOOSES_ONE = "line", "one"


def chooses_end(opt, fitted):
    """(label, make-up at the total, is a suggestion) of the line's far
    end: the open Round's first suggestion still to make; with no Round
    open, the predicted best (the Best so far section's own, cached once
    per setup). None when there is neither."""
    first = _first_pick(opt)
    if first is not None:
        return wording.CHOOSES_SUGGESTION_ONE, first[1], True
    best = _predicted_best(opt, fitted, checks_for(opt, "trust") or (), spinner=False)
    if isinstance(best, insights.PredictedBest) and best.recipe:
        return wording.CHOOSES_PREDICTED_BEST, insights.at_total(opt, best.recipe), False
    return None


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
    """Why the bottom panel is withheld, in words, or None — the Expected
    improvement layer's rule (layers_offered), reason by reason."""
    if wording.LAYER_GAIN in layers_offered(opt, fitted):
        return None
    score = (fitted or {}).get(insights.SCORE)
    checks = checks_for(opt, "trust") or ()
    rule = explore_withheld(opt, score, checks)
    if rule == "hidden":
        return wording.CHOOSES_REASON_NOISE
    if opt.pending_batch and opt.pending_batch_no is not None and \
            bo_views.generation(opt, opt.pending_batch_no).regime != "warm":
        return wording.CHOOSES_REASON_COLD
    why = withheld(opt, score, checks)
    if why is not None:
        gate = insights.skill_gate(opt, checks) if why == "skill" else None
        return gain_reason(why, gate)
    return wording.CHOOSES_REASON_UCB


def _first_pick(opt):
    """(number, make-up at the total) of the open Round's first suggestion
    still to make, or None."""
    numbers = suggested_numbers(opt)
    if not numbers:
        return None
    return numbers[0], anchor_recipe(opt, numbers[0])


def chooses_default(opt):
    """The row the next Round changes most from Best so far: the open
    Round's suggestions still to make, else the latest Round's
    formulations; else the first choice."""
    choices = insights.path_choices(opt)
    best = opt.best_index()
    if not choices or best is None:
        return choices[0] if choices else None
    start = insights.start_recipe(opt, best)
    recipes = [anchor_recipe(opt, n) for n in suggested_numbers(opt)]
    if not recipes:
        rounds = [r for r in opt.batch_history if r is not None]
        if rounds:
            last = max(rounds)
            recipes = [insights.start_recipe(opt, i) for i, r in enumerate(opt.batch_history)
                       if r == last]
    got = insights.most_changed(opt, start, recipes) if recipes else None
    return got or choices[0]


def chooses_data(opt, fitted, name, model=None, mode=CHOOSES_ONE):
    """Chooses along row `name` through Best so far (mode CHOOSES_ONE), or
    along the straight line from Best so far to chooses_end (CHOOSES_LINE,
    `name` unused); None when the path or the model cannot be had.
    `model`: the suggestion model (suggestion_model's when None)."""
    best = opt.best_index()
    if best is None:
        return None
    start = insights.start_recipe(opt, best)
    end = None
    if mode == CHOOSES_LINE:
        end = chooses_end(opt, fitted)
        if end is None:
            return None
        path = insights.line_path(opt, start, end[1])
    else:
        path = insights.build_path(opt, name, start)
    if path.start_breaks or not path.recipes:
        return None
    model = suggestion_model(opt, fitted) if model is None else model
    if model is None:
        return None
    values = bo_views.evaluate(opt, model, bo_views.points(opt, path.recipes))
    first = _first_pick(opt)
    reason = chooses_reason(opt, fitted)
    best_no = int(opt.formulation_ids[best])
    if end is not None:
        pick = (first[0], 1.0) if end[2] else None
        return Chooses(path, values, insights.tested_near_line(opt, path), best_no, pick,
                       reason, line=True, end_label=end[0], end_is_pick=end[2])
    pick = None if first is None else (first[0], insights.near_path(opt, path, first[1]))
    return Chooses(path, values, insights.tested_near_path(opt, path), best_no, pick, reason)


def _recipe_text(opt, recipe):
    """A make-up as the tooltip lists it: each varying row and the Fill to
    total row, in Set up order."""
    names = [a.name for a in insights.tried_axes(opt)]
    fill = opt.balance_row_name()
    if fill:
        names.append(fill)
    out = []
    for n in names:
        v = recipe.get(n)
        if isinstance(v, str):
            out.append(f"{n} {v}")
        elif v is not None:
            unit = opt.unit_of(n)
            setting = n != fill and insights.is_setting(opt, n)
            out.append(f"{n} " + (fmt_setting(float(v), unit) if setting
                                  else join_unit(_num(float(v)), unit)))
    return "; ".join(out)


def chooses_chart(opt, data):
    """How the app chooses: the classic figure along one row through Best
    so far. Top: the suggestion model's mean (its posterior mean) with its
    95 % band, grey where nothing was tested near, the formulations tested
    near the line as dots coloured by their Overall score. Bottom: the
    expected gain along the same line (the acquisition the suggestions are
    chosen by), unless withheld (data.reason). A dashed rule at Best so
    far; an orange one at this Round's first suggestion when it is near
    the line."""
    path, v = data.path, data.values
    if not path.amounts:
        return None
    mean, sd = v.mean.tolist(), v.sd.tolist()
    gain = v.value.tolist()
    if data.line:
        x_title = _say(opt, wording.chooses_line_axis(data.end_label))
        end = _say(opt, data.end_label)
        axis = alt.Axis(values=[0, 1], labelExpr=(
            f"datum.value == 0 ? {_say(opt, wording.TRIED_BEST)!r} : {end!r}"))
        along = _say(opt, wording.CHOOSES_AMOUNTS_HERE)
    else:
        x_title = f"{path.name} ({path.unit})" if path.unit else path.name
        axis = alt.Axis()
        along = path.name
    xs = alt.Scale(domain=[min(path.amounts), max(path.amounts)], zero=False, nice=False)
    x = alt.X("x:Q", title=x_title, scale=xs, axis=axis)
    d = decimals(opt, insights.SCORE)
    rows = [{"mark": "curve", "x": t, "mean": m, "low": m - insights.Z95 * s,
             "high": m + insights.Z95 * s, "gain": g,
             "amount": _recipe_text(opt, r) if data.line else _amount(t, path.unit),
             "pred": f"{m:.{d}f} ({range_text(m - insights.Z95 * s, m + insights.Z95 * s, lambda q: f'{q:.{d}f}')})",
             "gain_text": f"{g:.2f}"}
            for t, m, s, g, r in zip(path.amounts, mean, sd, gain, path.recipes)]
    near = insights.tested_near(opt, path)
    rows += [{"mark": "grey", "x": lo, "x2": hi} for lo, hi in insights.untested_runs(path, near)]
    df = pd.DataFrame(rows)
    ids = [int(n) for n in opt.formulation_ids]
    dots = pd.DataFrame([{"x": a, "score": round(s, 2), "no": ids[i]}
                         for i, a, s in data.tested], columns=["x", "score", "no"])
    marks = [{"x": path.start_amount, "label": _say(opt, wording.TRIED_BEST), "role": "best"}]
    if data.line and not data.end_is_pick:
        # The predicted best: not a suggestion, so ink, not orange.
        marks.append({"x": 1.0, "label": _say(opt, data.end_label), "role": "end"})
    elif data.pick is not None and data.pick[1] is not None:
        marks.append({"x": data.pick[1], "label": _say(opt, data.end_label if data.line
                                                       else wording.CHOOSES_SUGGESTION),
                      "role": "pick"})
    marks = pd.DataFrame(marks)
    ref_c, ref_o = mark("reference")
    blue = model_blue()
    band_c, band_o = _BAND_FILL[theme()]
    ink, _ = mark("bar_latest")
    word = _say(opt, wording.FORMULATION_CAP)
    curve = _only("curve")
    predicted = _say(opt, wording.CHOOSES_PREDICTED)
    tip = [alt.Tooltip("amount:N", title=along),
           alt.Tooltip("pred:N", title=f"{predicted} ({wording.CHOOSES_BAND})")]
    lo_all = min([r["low"] for r in rows if r["mark"] == "curve"]
                 + (list(dots["score"]) if len(dots) else []))
    hi_all = max([r["high"] for r in rows if r["mark"] == "curve"]
                 + (list(dots["score"]) if len(dots) else []))
    pad = 0.05 * max(hi_all - lo_all, 1.0)
    ys = alt.Scale(domain=[lo_all - pad, hi_all + pad], zero=False, nice=False)

    def rules(height_title=True):
        out = []
        for kind, colour, dash, opacity in (("best", ref_c, [4, 3], ref_o),
                                            ("pick", outline_colour(), [1, 0], 1.0),
                                            ("end", ink, [1, 0], 1.0)):
            sel = marks[marks["role"] == kind]
            if not len(sel):
                continue
            base = alt.Chart(sel)
            out.append(base.mark_rule(color=colour, opacity=opacity, strokeDash=dash,
                                      strokeWidth=2, tooltip=False).encode(x=alt.X("x:Q", scale=xs)))
            if height_title:
                out.append(base.mark_text(color=muted(), align="left", dx=4, baseline="top",
                                          fontSize=10, tooltip=False)
                           .encode(x=alt.X("x:Q", scale=xs),
                                   y=alt.value(4 if kind == "best" else 18), text="label:N"))
        return out

    top_layers = []
    if (df["mark"] == "grey").any():
        top_layers += _untested_layers(alt.Chart(df))
    scored_dom = ([float(dots["score"].min()), float(dots["score"].max())] if len(dots) > 1
                  else [0.0, float(opt.utility_ceiling())])
    top_layers += [
        # The band is the suggestion model's own uncertainty (no Repeat
        # noise, no widening): narrower than Try one's 95 % interval. Its
        # legend says so in a few words; its tooltip, in full.
        alt.Chart(df.assign(band=_say(opt, wording.CHOOSES_BAND_LEGEND),
                            band_tip=_say(opt, wording.TIP_CHOOSES_BAND)))
        .mark_area(opacity=band_o)
        .encode(x=x, y=alt.Y("low:Q", scale=ys, title=predicted), y2="high:Q",
                fill=alt.Fill("band:N", scale=alt.Scale(range=[band_c]),
                              legend=alt.Legend(title=None, orient="top", labelLimit=0)),
                tooltip=[alt.Tooltip("band_tip:N", title=_say(opt, wording.CHOOSES_BAND_LEGEND))])
        .transform_filter(curve),
        alt.Chart(df).mark_line(color=ink, strokeWidth=2)
        .encode(x=x, y=alt.Y("mean:Q", scale=ys, title=predicted), tooltip=tip)
        .transform_filter(curve),
        *rules(),
    ]
    if len(dots):
        top_layers.append(
            alt.Chart(dots).mark_point(filled=True, size=70, opacity=1, stroke=ink,
                                       strokeWidth=1)
            .encode(x=x, y=alt.Y("score:Q", scale=ys, title=predicted),
                    color=alt.Color("score:Q", scale=score_scale(scored_dom),
                                    title=_say(opt, wording.CHOOSES_TESTED),
                                    legend=alt.Legend(orient="right", gradientLength=90)),
                    tooltip=[alt.Tooltip("no:Q", title=word, format="d"),
                             alt.Tooltip("x:Q", title=("t" if data.line else path.name),
                                         format=".2f"),
                             alt.Tooltip("score:Q", title=_say(opt, wording.OVERALL_SCORE_COLUMN),
                                         format=".2f")]))
    top = alt.layer(*top_layers).properties(height=220)
    if data.reason is not None:
        return top
    gain_title = _say(opt, wording.CHOOSES_GAIN)
    bottom = alt.layer(
        alt.Chart(df).mark_area(color=blue, opacity=0.25, tooltip=False)
        .encode(x=x, y=alt.Y("gain:Q", title=gain_title)).transform_filter(curve),
        alt.Chart(df).mark_line(color=blue, strokeWidth=2)
        .encode(x=x, y=alt.Y("gain:Q", title=gain_title),
                tooltip=[alt.Tooltip("amount:N", title=along),
                         alt.Tooltip("gain_text:N", title=wording.GAIN_TOOLTIP_TITLE)])
        .transform_filter(curve),
        *rules(height_title=False),
    ).properties(height=130)
    return alt.vconcat(top, bottom, spacing=8).resolve_scale(x="shared", color="independent")


def _chooses(opt, computed):
    """§2 of the fold: the title, its lead, the selector, the figure, the
    one-line caption; the withheld reason under it."""
    st.markdown(f"**{_say(opt, wording.CHOOSES_TITLE)}**")
    if opt.complete_count() < insights.MIN_ROWS:
        st.caption(_say(opt, wording.CHOOSES_COLD))
        return
    if computed is None:
        st.caption(_say(opt, wording.CHECKING_PREDICTIONS))
        return
    if computed.get("failed"):
        st.caption(_say(opt, wording.CHART_FAILED))
        return
    choices = insights.path_choices(opt)
    best = opt.best_index()
    if not choices or best is None:
        st.caption(_say(opt, wording.change_none(settings_only=not opt.has_ingredients())))
        return
    fitted = computed["fitted"]
    # Charts proto follow-up: the straight line from Best so far to this
    # Round's first suggestion (else the predicted best) first; one row at
    # a time second.
    has_pick = _first_pick(opt) is not None
    modes = [CHOOSES_LINE, CHOOSES_ONE]
    _keep_if_offered(CHOOSES_MODE_KEY, modes)
    mode = st.selectbox(_say(opt, wording.CHOOSES_MODE), modes, key=CHOOSES_MODE_KEY,
                        format_func=lambda m: _say(opt, wording.chooses_mode_option(
                            m == CHOOSES_LINE, has_pick, not opt.has_ingredients())))
    name = None
    if mode == CHOOSES_ONE:
        _keep_if_offered(CHOOSES_KEY, choices)
        if CHOOSES_KEY not in st.session_state:
            st.session_state[CHOOSES_KEY] = chooses_default(opt)
        name = st.selectbox(_say(opt, wording.INGREDIENT_OR_SETTING), choices, key=CHOOSES_KEY)
    what = name or _say(opt, wording.CHOOSES_MODE)
    try:
        with st.spinner(_say(opt, wording.INSIGHTS_WORKING)):
            data = chooses_data(opt, fitted, name, mode=mode)
    except Exception:
        data = None
    if data is None:
        st.caption(_say(opt, wording.panel_failed(what)))
        return
    chart = chooses_chart(opt, data)
    if chart is None:
        st.caption(_say(opt, wording.panel_failed(what)))
        return
    show_chart(chart)
    st.caption(_say(opt, wording.CHOOSES_CAPTION),
               help=_say(opt, wording.tip_chooses(data.best_no,
                                                  settings_only=not opt.has_ingredients(),
                                                  line=data.line, end=data.end_label)))
    if data.reason is not None:
        st.caption(_say(opt, wording.CHOOSES_WITHHELD),
                   help=_say(opt, wording.tip_chooses_withheld(data.reason)))
    if data.pick is not None and data.pick[1] is None:
        st.caption(_say(opt, wording.CHOOSES_OFF_LINE),
                   help=_say(opt, wording.tip_chooses_off_line(data.pick[0])))


def _triangle(opt, computed):
    """§3 of the fold: the triangle (was under Trade-offs), under its own
    gate: the models fitted and three ingredients to trade."""
    if computed is None or computed.get("failed") or opt.complete_count() < insights.MIN_ROWS:
        return
    if not opt.has_ingredients():
        return
    try:
        _mixture(opt, computed["fitted"])
    except Exception:   # one chart failing must not take the fold down
        st.caption(_say(opt, wording.CHART_FAILED))


def layer_values(opt, fitted, recipes, layer):
    """The layer's number at each recipe (points at the Formulation total):
    the suggestion model's sd, or its acquisition in points; None without
    a model."""
    model = layer_model(opt, fitted)
    if model is None or not recipes:
        return None
    got = bo_views.evaluate(opt, model, bo_views.points(opt, recipes))
    return (got.sd if layer == wording.LAYER_SD else got.value).tolist()


# How many combinations _three_default tries before it settles for the
# first: each try is a three-point slice, a few milliseconds.
_THREE_TRIES = 20


def _three_default(opt, fitted, offered, start):
    """Of the first _THREE_TRIES combinations of three — the offered choices
    ranked by their spread from question 2 (a choice the limits keep in
    place has no spread and is not ranked), then the rest in Set up order —
    the one with the largest allowed share of its triangle
    (insights.allowed_share; ties go to the earlier). With none that can
    trade places, the first combination, as before; any failure of the
    spreads falls back to Set up order (0.8.0 review fixes §5.2)."""
    try:
        ranked = [s.name for s in insights.spreads(opt, fitted, start).moved
                  if s.name in offered]
    except Exception:
        ranked = []
    order = ranked + [n for n in offered if n not in ranked]
    best, best_share = None, 0.0
    for names in itertools.islice(itertools.combinations(order, 3), _THREE_TRIES):
        try:
            share = insights.allowed_share(opt, list(names), start)
        except Exception:
            share = 0.0
        if share > best_share + 1e-12:
            best, best_share = list(names), share
    return best if best is not None else order[:3]


def _mixture_cached(opt, fitted, names, number, start, layer=wording.LAYER_SCORE):
    """(mix, predictions, the start's own prediction, extra) once per setup,
    three, start and layer. `extra`: the layer's values; for the Predicted
    score layer, the Predicted above Best so far outline and the colour
    span against the model's own uncertainty (review I6, M7)."""
    checks = checks_for(opt, "trust") or ()
    # Whether the outline may be drawn is part of the answer (polish round).
    held_back = (layer == wording.LAYER_SCORE
                 and withheld(opt, fitted.get(insights.SCORE), checks) is not None)
    key = (insights.setup_key(opt), tuple(names), number, layer, held_back)
    hit = st.session_state.get(MIXTURE_KEY)
    if hit and hit[0] == key:
        return hit[1]
    with st.spinner(_say(opt, wording.INSIGHTS_WORKING)):
        try:
            mix = insights.mixture_slice(opt, names, start)
            preds = insights.mixture_predictions(opt, fitted, mix)
            value = (mix, preds, _start_prediction(opt, fitted, mix, start),
                     _mixture_extra(opt, fitted, mix, preds, layer, checks))
        except Exception:
            value = _FAILED
    st.session_state[MIXTURE_KEY] = (key, value)
    return value


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


def _mixture_extra(opt, fitted, mix, preds, layer, checks=None):
    """What the chart needs beyond the predictions, per point (None where
    blocked): a layer's values; or, on the Predicted score layer, the
    outline (the model's mean − 1.96 × its own sd above Best so far's
    predicted Repeat) and (colour span, median 1.96 × model sd) for the
    §2.2 caption. {} when the suggestion model cannot be rebuilt."""
    allowed = [p.recipe for p in mix.points if p.recipe is not None]
    model = (layer_model(opt, fitted) if layer != wording.LAYER_SCORE
             else _outline_model(opt, fitted))
    if model is None or not allowed:
        return {}
    got = bo_views.evaluate(opt, model, bo_views.points(opt, allowed))
    spread = lambda seq: iter(seq)
    if layer != wording.LAYER_SCORE:
        it = spread((got.sd if layer == wording.LAYER_SD else got.value).tolist())
        return {"values": [next(it) if p.recipe is not None else None for p in mix.points]}
    ref = repeat_of_best(opt, fitted)
    low = (got.mean - insights.Z95 * got.sd).tolist()
    it = spread(low)
    top = opt.best_index()
    measured_top = None if top is None else float(opt.Y_history[top])
    checks = (checks_for(opt, "trust") or ()) if checks is None else checks
    yard = insights.noise_yardstick(opt, checks).value
    # The outline is a gain claim: none while the model's gains are withheld
    # (polish round; the expected improvement's rule).
    held_back = withheld(opt, fitted.get(insights.SCORE), checks) is not None
    ring = []
    for p, pred in zip(mix.points, preds or [None] * len(mix.points)):
        lo = next(it) if p.recipe is not None else None
        ring.append(not held_back and p.recipe is not None and insights.above_best(
            lo, None if pred is None else pred.mean, ref, measured_top, yard))
    means = [p.mean for p in preds if p is not None]
    half = sorted((insights.Z95 * got.sd).tolist())[len(allowed) // 2]
    return {"outline": ring, "ref": ref,
            "span": (max(means) - min(means)) if means else None, "half": half}


def within_noise_line(extra):
    """Review M7: when the colours span less than the median 1.96 × model
    sd, one caption says so (the colours are not rescaled); else ''."""
    span, half = extra.get("span"), extra.get("half")
    if span is not None and half is not None and span < half:
        return wording.mixture_within_noise(f"{span:.1f}", f"{half:.1f}")
    return ""


def _start_prediction(opt, fitted, mix, start):
    """The start's own predicted Overall score, for its ring's tooltip; None
    when there is no chart or it cannot be predicted (the ring stands)."""
    if not mix.points:
        return None
    try:
        [own] = insights.predict(opt, fitted[insights.SCORE], [opt._encode(start)])
    except Exception:
        return None
    return own


def _mixture(opt, fitted):
    offered = insights.mixture_choices(opt)
    score = fitted.get(insights.SCORE)
    if len(offered) < 3 or score is None or score.model is None:
        return False
    _question(opt, wording.Q_MIXTURE)
    layer = _layer_pick(opt, fitted, LAYER_KEY)
    layered = layer != wording.LAYER_SCORE
    number = _start_number(opt, layered)
    start = anchor_recipe(opt, number)
    _keep_all_offered(THREE_KEY, offered)
    if THREE_KEY not in st.session_state:
        st.session_state[THREE_KEY] = _three_default(opt, fitted, offered, start)
    picked = st.multiselect(_say(opt, wording.MIXTURE_PICK), offered, key=THREE_KEY,
                            max_selections=3)
    if len(picked) != 3:
        st.caption(_say(opt, wording.PICK_THREE))
        return True
    got = _mixture_cached(opt, fitted, picked, number, start, layer)
    if got is _FAILED:
        st.caption(_say(opt, wording.panel_failed(wording.OVERALL_SCORE_COLUMN)))
        return True
    mix, preds, own, extra = got
    if mix.refusal == insights.NO_ROOM:
        st.caption(_say(opt, wording.mixture_no_room(number)))
        return True
    if mix.refusal == insights.FILL_FOLLOWS:
        st.caption(_say(opt, wording.mixture_fill_follows(mix.follows[0],
                                                          opt.balance_row_name())))
        return True
    if mix.start_breaks:
        st.caption(_say(opt, wording.path_start_breaks(number, mix.start_breaks)))
        return True
    chart = mixture_chart(opt, mix, preds, number, own, layer if layered else None,
                          extra.get("values"), extra.get("outline"))
    if chart is not None:
        show_chart(chart, width="content")
    if layered:
        st.caption(_say(opt, wording.layer_caption(layer, anchor_text(opt, number))))
    else:
        st.caption(_say(opt, wording.mixture_caption(number)))
        if extra.get("outline") and any(extra["outline"]):
            st.caption(_say(opt, wording.overlay_caption(_whole(extra["ref"]))))
        line = within_noise_line(extra)
        if line:
            st.caption(_say(opt, line))
    for line in ([wording.path_follows(n) for n in mix.follows]
                 + ([wording.mixture_fills(mix.fill_moves)] if mix.fill_moves else [])
                 + [wording.path_not_checked(t, prop) for t, prop in mix.skipped]):
        st.caption(_say(opt, line))
    return True


def more_charts_offered(opt, computed):
    """Whether "More charts" has anything to show: 5 formulations with all
    their results, the models fitted, and at least one of its charts past
    its own gate."""
    if computed is None or opt.complete_count() < insights.MIN_ROWS:
        return False
    fitted = computed["fitted"]
    score = fitted.get(insights.SCORE)
    # The misses chart is gone (spec 2026-09-30 §2.5): measured against
    # predicted, under "Show each prediction", says the same in the units.
    # Charts proto: the triangle moved to Explored and unexplored; the
    # fold holds the two-measurement chart alone.
    return len(opt.objectives) >= 2


def trade_offs_label(opt, computed):
    """The Trade-offs fold's name (ftue §2; was More charts): it names the
    three ingredients only when the triangle can be drawn."""
    return wording.PAIR_FOLD


def _more_charts(opt, computed):
    if not more_charts_offered(opt, computed):
        return
    # Keyed (M3), so its open state is the reader's across reruns.
    with st.expander(_say(opt, trade_offs_label(opt, computed)), key=MORE_KEY):
        for draw in (lambda: _pair(opt),):
            try:
                draw()
            except Exception:   # one chart failing must not take the tab down
                st.caption(_say(opt, wording.CHART_FAILED))


def _how(opt, computed):
    """Method and definitions (spec 2026-10-01 rev. 2 §2, §3, §7; review
    M3, M12): Model and method read off the project's own model, with its
    known limits; How each status is decided; the plain paragraphs, the
    details table with its window named and the lines that read it; the
    glossary last. Everything a tooltip says is here too (M12)."""
    with st.expander(_say(opt, wording.FOLD_METHOD), key=HOW_KEY,
                     on_change="rerun") as fold:
        # Keyed and worked out only while open (T56 I5; binding I15).
        if not fold.open:
            return
        _model_and_method(opt, computed)
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
            verdict = wording.how_verdict(h.total, insights.allowed_misses(h.total))
    return [wording.how_models(cfg.get("kernel"), cfg.get("noise")),
            wording.HOW_RANGE_FIXED_TINY if tiny else wording.HOW_RANGE,
            wording.how_checks(values=not opt.has_ingredients(),
                               total=opt.has_formulation_total()),
            verdict, wording.HOW_NOISE_THRESHOLD, wording.HOW_REGRESSION,
            wording.how_same_process(values=not opt.has_ingredients()),
            wording.HOW_NO_P]


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
    if h.state == "early":
        return wording.tip_coverage_early(names, h.rounds, h.total)
    return wording.tip_coverage(names, h.rounds, h.inside, h.total,
                                insights.allowed_misses(h.total))


def _skill_tip(opt, facts):
    """The Overall score's skill with its window and n, or the count it
    still needs."""
    fact = facts.skill
    if fact is not None:
        return wording.tip_skill(fact.rounds, skill_value(fact.value), fact.n)
    gate = getattr(facts.guard, "gate", None)
    if gate is not None and gate.state == "wait":
        n = gate.n
    else:
        h = facts.headline
        n = h.score.total if h is not None and h.score is not None else 0
    return wording.tip_skill_too_few(n)


def _repeat_tip(d):
    return wording.tip_repeat_off(d.later, d.earlier, f"{abs(d.diff):.1f}",
                                  "" if d.limit is None else f"{d.limit:.1f}")


def _bias_tip(facts):
    h = facts.headline
    if facts.bias is None or h is None or h.score is None:
        return ""
    return wording.tip_bias(wording.signed(facts.bias, 1), h.score.total, h.score.rounds)


def _wide_label(name):
    return wording.OVERALL_SCORE_COLUMN if name == insights.SCORE else name


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
    return wording.TIP_THRESHOLD_ASSUMED


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
        sd = 1.0 if a.sd is None else a.sd
        need = (f"{a.threshold:.2f}", rise_factor_text(a.q, a.df),
                "1" if a.sd is None else f"{sd:.2f}")
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
