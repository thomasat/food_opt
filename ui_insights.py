"""The Insights tab. Builders here return Altair charts from computed facts
and never call Streamlit; render() at the bottom is the only Streamlit code."""
import contextlib
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


# --- review plots (2026-10-02) ---
# One meaning, one colour, on every chart (Okabe-Ito orange and the ink and
# blue tokens above, all at least 3:1 in both themes):
#   Best so far          ink (mark("bar_latest")), a thick line, a ring or a dashed rule;
#                        another start (Try one) the dotted reference token
#   this Round's picks   orange (outline_colour())
#   tested formulations  neutral grey rings (mark("neutral"))
#   the model            blue (model_blue(); its 95 % band in _BAND_FILL)
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
    # Review plots: one legend for the whole figure — filled inside its 95 %
    # range, hollow outside; the latest Round dark, earlier ones lighter.
    fills = alt.Scale(domain=[inside_name, outside_name], range=[ink, "transparent"])
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
            "state": inside_name if c.inside else outside_name,
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
    Next card's content the closing section, What next?."""
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
        _answer(opt, places.q["progress"], facts, "progress")
        for which in ("predictions", "drives"):
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
    _finish(opt, places, facts)
    _trust_section(opt, places.q["predictions"], computed, facts)
    _drives_section(opt, places.q["drives"], computed, facts)
    _how(opt, computed)
    save(opt)


class _Places:
    """The places render draws before anything is worked out, in screen
    order (R-I8), for _fill_in and the rest of render to fill."""


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
_QUESTIONS = {"predictions": wording.Q1_TITLE, "progress": wording.Q2_TITLE,
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
            _foot(opt)
    p.bar = st.empty()
    p.q = {}
    # Review story: progress first (the payoff), then trust (the profiler
    # leans on the model), then drivers, then what was tried; review story
    # 2: then What next?, so the story ends on the action.
    for which in SECTION_ORDER:
        p.q[which] = _section(opt, which)
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
    # The model-free charts, at once (they need no fit).
    _progress_body(opt, p.q["progress"])
    _tried_section(opt, p.q["tried"])
    if gate:
        for which in ("predictions", "drives"):
            with p.q[which].body:
                st.caption(_say(opt, wording.Q_WAIT))
    return p


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
        nums = story_numbers(opt)
        best = wording.score_of(one_decimal(f.score), opt.utility_ceiling())
        start = one_decimal(nums[0]) if nums is not None and nums[0] != nums[2] else None
        return _say(opt, wording.strip_progress(start, best)), False
    if which == "predictions":
        return (wording.strip_short(card.lines[0][0]) if card.lines else ""), False
    return "", False


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
    slot = places.item[which].container(gap=None) if places is not None else None
    with slot if slot is not None else st.container(gap=None):
        label = _say(opt, _CARD_LABELS[which])
        if card is None:
            st.caption(label)
            st.caption(_say(opt, _PLACEHOLDERS[which]))
            return
        st.caption(label, help=card.help)
        line, warn = strip_line(opt, facts, which, card)
        question = _say(opt, _QUESTIONS[which])
        st.button(strip_label(card, line, warn), type="tertiary", key=f"{SEE_CHART_KEY}_{which}",
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
        word = f":red[{card.word}]" if card.tone == "off" else card.word
        st.subheader(word, anchor=False, help=card.help)
        for text, warn in card.lines:
            st.markdown(f":red[**{text}**]" if warn else text)


def _section(opt, which):
    """A question: its heading (one style for the four), its answer line's
    place and its body, in a keyed place a link lands on."""
    with st.container(key=SECTION_KEYS[which], gap="small"):
        st.markdown(f"#### {_say(opt, _QUESTIONS[which])}")
        answer = st.empty()
        body = st.container(gap="small")
    return _Slot(answer, body)


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
    the method link). Only on arrival (ui_helpers.SCROLL_KEY, drained by
    app.py after the tab is drawn), so a fold the reader closed stays
    closed. The questions are always open: nothing else to open."""
    if st.session_state.get(SCROLL_KEY) == HOW_PLACE:
        st.session_state[HOW_KEY] = True


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
    _answer(opt, places.q["progress"], facts, "progress")
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
    with places.change.container():
        _setup_button(change)
    if not places.why:
        # R-I4's one link, by priority: Why (drawn with the places), then
        # the Repeat, then the Explored link (ftue fix C).
        with places.link.container():
            if not _repeat_link(opt, facts):
                explored_link(opt, facts.next)
    _answer(opt, places.q["progress"], facts, "progress")


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
    for which in ("predictions", "drives"):
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
SECTION_ORDER = ("progress", "predictions", "drives", "tried", "next")


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
    """The line's tooltip: its full sentence, then (review polish) one
    compact evidence line — n, method, rule — then W2's exact rule."""
    full = status_full_texts(opt, facts).get(which)
    tip = tooltip_for(opt, which, facts)
    try:
        ev = _say(opt, evidence_line(opt, facts, which))
    except Exception:       # the evidence line must never take a card down
        ev = None
    parts = [p for p in (full and _full_sentence(full), ev, tip) if p]
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
    name = insights.status_label(opt, ("next", step.kind))
    return wording.evidence([wording.ev_rule(name),
                             wording.ev_suggestions(cfg.get("acquisition", "qlognei")),
                             wording.EV_METHOD])


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
    best = wording.short_best(f"{f.score:.2f}", ceiling, f.formulation, tie)
    a = facts.improvement
    if facts.progress in ("rising", "within_noise"):
        rise, need = rise_and_need(a)
        short = wording.short_rise_best(f"{f.score:.2f}", f.formulation, tie)
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
    d = diag_for(opt, checks, h) if state in ("caution", "unreliable") else None
    if d is not None:
        # Review diagnosis: one story, the count then where the misses fell;
        # no "too narrow" beside a "wide" line.
        line = wording.trust_misses_diag(h.rounds, h.inside, h.total,
                                         many=state == "unreliable", allowed=allowed)
        return " ".join([line, wording.diag_detail(diag_items(opt, d), d.rounds, d.far,
                                                   d.misses, d.far_checked, d.checked),
                         wording.DIAG_READING[d.kind]])
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
            return wording.next_misses_diag(rounds, d.kind, no_best, bias_text=bias), None
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
    say = (lambda t: t) if opt is None else (lambda t: _say(opt, t))
    x = alt.X("mean:Q", title=wording.WHY_AXIS_MEAN, scale=alt.Scale(zero=False))
    # From 0: the model's sd is a size, and a zoomed sd axis made a small
    # difference in certainty look large (review plots).
    y = alt.Y("sd:Q", title=wording.MODEL_SD, scale=alt.Scale(zero=True))
    word = say(wording.FORMULATION_CAP)
    ring_c, ring_o = mark("neutral")
    ink, _ = mark("bar_latest")
    # Review plots: one legend for the three kinds of mark (the caption
    # named them before), the ring and diamond drawn as on the chart.
    kinds = [say(wording.PLOT_ALLOWED), say(wording.PLOT_TESTED), say(wording.PLOT_SUGGESTION)]
    kind_colour = alt.Scale(domain=kinds, range=["#b8bcc4", ring_c, outline_colour()])
    kind_fill = alt.Scale(domain=kinds, range=["#b8bcc4", "transparent", outline_colour()])
    kind_shape = alt.Scale(domain=kinds, range=["circle", "circle", "diamond"])
    legend = alt.Legend(title=None, orient="top", labelLimit=0)
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
        alt.Chart(line).mark_text(align="left", baseline="top", dx=4, color=muted(),
                                  tooltip=False)
        .encode(x="ref:Q", y=alt.value(4), text="label:N")]
    numbered = "no" in tested.columns
    ring_tip = ({"tooltip": [alt.Tooltip("no:Q", title=word, format="d"),
                             alt.Tooltip("mean:Q", title=wording.WHY_AXIS_MEAN, format=".1f"),
                             alt.Tooltip("sd:Q", title=wording.MODEL_SD, format=".2f"),
                             bo_tip()]}
                if numbered else {})
    layers = [
        alt.Chart(pool).mark_point(size=12, opacity=0.6, tooltip=False)
        .encode(x=x, y=y, stroke=alt.Stroke("what:N", scale=kind_colour, legend=legend),
                fill=alt.Fill("what:N", scale=kind_fill, legend=legend),
                shape=alt.Shape("what:N", scale=kind_shape, legend=legend)),
        # The neutral token (polish round): gray at Vega's default 0.7 was
        # 2.94:1 on the dark page.
        alt.Chart(tested).mark_point(size=40, opacity=ring_o, strokeWidth=1.5,
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
        alt.Chart(mine).mark_text(dy=-12, color=muted(), tooltip=False)
        .encode(x=x, y=y, text="no:N"),
    ]
    return titled(alt.layer(*layers).properties(height=280), say(wording.PLOT_MAP))


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


# Review four polish (owner: the score colours "all look very similar"):
# Q4 coloured by role. Every other tested line thin and light grey (context,
# so an untested stretch reads as empty space); the next best few mid-grey;
# Best so far thick ink; this Round's suggestions orange. (light grey, mid-
# grey) per theme: light #cfd3d9 / #7d838e on white; dark #444a54 / #9aa0aa
# on #0e1117 — the mid-grey and the ink reach 3:1, the context lines need not.
_TRIED_GREYS = {"light": ("#cfd3d9", "#7d838e"), "dark": ("#444a54", "#9aa0aa")}


def next_best_count(scored):
    """How many tested lines after Best so far are drawn mid-grey: 3 to 5,
    about one in six of the tested ones."""
    return max(3, min(5, scored // 6))


def _next_best(lines):
    """The formulation numbers of the next best few tested lines, by score
    (Best so far itself left out)."""
    tested = sorted((l for l in lines if l.state == "tested" and not l.best
                     and l.score is not None), key=lambda l: -l.score)
    return {l.formulation for l in tested[:next_best_count(len(tested) + 1)]}


def _line_kind(line, nexts=()):
    if line.state == "suggested":
        return wording.TRIED_SUGGESTED
    if line.state != "tested":
        return wording.TRIED_WAITING
    if line.best:
        return wording.TRIED_BEST
    return wording.TRIED_NEXT_BEST if line.formulation in nexts else wording.TRIED_SCORED


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
    nexts = _next_best(lines)
    # Review story, honesty guard: the team's own formulations in the open
    # Round are not the app's suggestions (drawn as waiting, grey dashed).
    generated = set(suggested_numbers(opt))
    for order, line in enumerate(lines):
        amounts = "; ".join(
            f"{a.name} {line.values[a.name]}" if a.options is not None
            else f"{a.name} {_axis_end(opt, a, line.values[a.name])}" for a in axes)
        role = _line_kind(line, nexts)
        if line.state == "suggested" and line.formulation not in generated:
            role = wording.TRIED_WAITING
        common = {"no": line.formulation, "round": _round_label(line.round),
                  "role": role, "best": line.best,
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
    # Review plots (the yogurt pilot: water at its 42 g Highest, tapioca at
    # 2.6 g): an axis whose lines mostly sit at one end collapses into a
    # bundle there; its end label says how many, so the bundle reads.
    total = len(lines)
    for e in ends:
        if e["where"] not in ("low", "high") or e["axis"] == score_label or total < 4:
            continue
        at = sum(1 for r in rows if r["axis"] == e["axis"] and abs(r["t"] - e["t"]) <= 0.01)
        if at * 2 >= total:
            e["text"] = wording.plot_at_end(e["text"], at, total)
    cols = ["no", "round", "role", "best", "score", "score_text", "amounts", "order", "axis",
            "k", "t"]
    return pd.DataFrame(rows, columns=cols), pd.DataFrame(ends), labels


def tried_chart(opt, axes, lines):
    """What you've tried: parallel coordinates. One vertical axis per
    varying ingredient and process setting, its Lowest at the bottom and
    its Highest at the top (written at the ends, with the unit), a choice's
    options spaced evenly; the last axis the Overall score, 0 to the
    ceiling. Coloured by role (review four polish): every tested line thin
    light grey, the next best few mid-grey, Best so far thick ink; waiting or
    Not scored, grey dashed; this Round's suggestions, orange, ending before
    the score. None without an axis or a line."""
    if not axes or not lines:
        return None
    df, ends, labels = tried_frame(opt, axes, lines)
    many = len(lines) > _MANY_LINES
    x = alt.X("axis:N", sort=labels, title=None,
              scale=alt.Scale(type="point", padding=0.25),
              # Review plots: a two-line name stacks upward from its
              # baseline, clear of the Highest written under it (the second
              # line used to sit on that number).
              axis=alt.Axis(orient="top", labelAngle=0, ticks=False,
                            labelPadding=22 + 13 * (max(l.count("\n") for l in labels)),
                            domain=False, labelLimit=0, labelFontSize=11, labelLineHeight=13,
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
    light, mid = _TRIED_GREYS[theme()]
    text_c = muted()
    # Review four polish: one legend, by role — ink, greys and orange; no
    # score colours (the last axis carries the score).
    order = (wording.TRIED_BEST, wording.TRIED_NEXT_BEST, wording.TRIED_SCORED,
             wording.TRIED_SUGGESTED, wording.TRIED_WAITING)
    look = {wording.TRIED_BEST: (ink, [1, 0], 4, 1.0),
            wording.TRIED_NEXT_BEST: (mid, [1, 0], 2, 1.0),
            wording.TRIED_SCORED: (light, [1, 0], 1 if many else 1.3, 1.0),
            wording.TRIED_SUGGESTED: (outline_colour(), [1, 0], 3, 1.0),
            wording.TRIED_WAITING: (grey, [4, 3], 1.5, 1.0)}
    kinds = [k for k in order if k in set(df["role"])]
    stroke = alt.Scale(domain=kinds or [""], range=[look[k][0] for k in kinds] or [grey])
    dash = alt.Scale(domain=kinds or [""], range=[look[k][1] for k in kinds] or [[1, 0]])
    kind_legend = alt.Legend(title=None, orient="bottom", labelLimit=0, symbolStrokeWidth=3,
                             symbolType="stroke")
    axes_df = pd.DataFrame({"axis": labels, "t0": 0.0, "t1": 1.0})
    layers = [
        alt.Chart(axes_df).mark_rule(color=grey, opacity=grey_o, strokeWidth=1.5, tooltip=False)
        .encode(x=x, y=alt.Y("t0:Q", scale=t_scale, axis=None), y2="t1:Q"),
    ]
    # Drawn light to strong, Best so far on top.
    for k in (wording.TRIED_SCORED, wording.TRIED_WAITING, wording.TRIED_NEXT_BEST,
              wording.TRIED_SUGGESTED, wording.TRIED_BEST):
        if k not in kinds:
            continue
        _, _, width, opacity = look[k]
        layers.append(
            alt.Chart(df).mark_line(strokeWidth=width, opacity=opacity)
            .encode(x=x, y=y, detail="no:N", order=alt.Order("k:Q"), tooltip=tips,
                    stroke=alt.Stroke("role:N", scale=stroke, legend=kind_legend),
                    strokeDash=alt.StrokeDash("role:N", scale=dash, legend=kind_legend))
            .transform_filter(alt.FieldEqualPredicate(field="role", equal=k)))
    text = lambda where, **kw: (alt.Chart(ends[ends["where"] == where])
                                .mark_text(color=text_c, fontSize=10, tooltip=False, **kw)
                                .encode(x=x, y=alt.Y("t:Q", scale=t_scale, axis=None),
                                        text="text:N"))
    layers += [text("low", baseline="top", dy=6), text("high", baseline="bottom", dy=-6)]
    if (ends["where"] == "option").any():
        layers.append(text("option", align="left", dx=5, baseline="middle"))
    heading = wording.PLOT_TRIED_SETTINGS if not opt.has_ingredients() else wording.PLOT_TRIED
    return titled(alt.layer(*layers).properties(height=300), _say(opt, heading))


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
        lines.append((wording.CARD_MEASURES_MISS, False))
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


def progress_card(opt, facts):
    f = facts.best
    word = None if facts.progress is None else wording.PROGRESS_WORDS[facts.progress]
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
        tail = wording.card_flat(a.rounds)
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
    elif kind == "misses":
        # Review diagnosis: a neutral data request, never "check panel".
        lines.append((wording.card_noise_repeat(
            facts.best.formulation if facts.best is not None else None), False))
    elif kind == "no_skill":
        lines.append((wording.CARD_GUIDED, False))
    elif kind == "untested":
        word = wording.CARD_TRY_BEST
        lines = [(wording.card_add_to(no if opt.pending_batch else None), False)]
    elif kind == "close":
        word = wording.CARD_CLOSE if step.binding is not None else wording.CARD_CLOSE_TOP
        lines = [(wording.card_close_line(
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
    card.help = status_help(opt, facts, which)
    return card


REVIEW_CHECKING = "checking"


def review_status(opt):
    """Review guide: Make a round's one line, the cards' own words —
    (text, warn), warn when the Predictions or the Next card is drawn red
    (Off, Check panel first); REVIEW_CHECKING while the workers are still
    fitting what the cards read; None with nothing scored, or when the
    checks are not worked out and there are no workers to work them out
    (the line never fits a model: it is read from what Review has)."""
    if opt.complete_count() == 0:
        return None
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
    return _say(opt, text), warn


def predictions_spark_frame(opt, checks):
    """Per checked Round: the share of its Round-ahead results inside their
    95 % intervals (the verdict's values: the measurements)."""
    rows = []
    values = [c for c in (checks or ()) if c.name != insights.SCORE]
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
    Round-ahead results in range per Round, 0-100 %, against the 95 % a
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
        # The whole 0-100 %: a drop from 95 % to 60 % reads as the fall it is.
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
_PANEL_W = 104


def panel_width(n):
    """Review story: wider panels when fewer columns are drawn (4 by
    default), so the profiler keeps about the same overall width."""
    return max(_PANEL_W, min(160, 640 // max(1, n)))
_PANEL_H = {"score": 84, "measure": 60, "gain": 60}     # review four: compact rows


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
        paths.append(path)
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
    columns.sort(key=lambda c: (-c.near_effect, -c.effect))
    rows = [insights.SCORE] + [o['name'] for o in opt.measurements_by_importance()
                               if o['name'] in allp]
    if insights.SCORE not in allp:
        rows = rows[1:]
    return Profile(columns, rows, ref_label, ref_no, is_pick,
                   reason if gain is not None else (reason or wording.CHOOSES_REASON_COLD),
                   opt.balance_row_name(), [v['name'] for v in opt.fixed_variables()], failed,
                   gain_drawn=gain is not None)


def gain_greyable(reason):
    """Ruling 2: a withheld reason under which the gain is still drawn,
    in grey (the noise guard, the verdict, the skill gate); a cold Round
    or an upper-bound acquisition has no gain to draw."""
    return reason is not None and hero_reason_key(reason) not in ("cold", "ucb")


def _row_title(opt, name):
    """A row's y title, two short lines: the result and its aim."""
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
    # Rotated beside a short panel: the name on at most two lines.
    return _wrap(obj['name'], 13).split("\n")[:2] + [aim]


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
        recs += [{"col": col, "mark": "ends", "x": e} for e in (c.lo, c.hi)]
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
                                 "align": "left" if side == "Highest" else "right"})
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
    display model's mean (blue) and its 95 % likely range of a result; the
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
    # The header under each column: its name and unit, wrapped to the panel.
    labels = {c.name: _wrap(_x_title(c.path), 14) for c in cols}
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
    legend = alt.Legend(title=None, orient="top", labelLimit=0, symbolOpacity=0.7)
    predicted = _say(opt, wording.PREDICTED)
    row_kinds = [("score" if n == insights.SCORE else "measure", n) for n in names]
    if with_gain:
        row_kinds.append(("gain", None))
    gain_top = max(max(max(c.gain) for c in cols), 0.05) if with_gain else None
    out_rows = []
    for r, (kind, name) in enumerate(row_kinds):
        last_row = r == len(row_kinds) - 1
        if kind == "gain":
            ys = alt.Scale(domain=[0.0, gain_top * 1.05], nice=False)
            y_title = [wording.PROFILER_GAIN_ROW, "(points)"]
        else:
            domain = _row_domain(opt, profile, cols, name)
            ys = alt.Scale(domain=domain, nice=False, zero=False)
            y_title = _row_title(opt, name)
            if kind == "score":
                ceiling = float(opt.utility_ceiling())
                y_title = [y_title[0], zoomed("(of 100)", domain[0], domain[1], ceiling)
                           .replace("((", "(").replace("))", ")")]
        df = _profile_row_frame(opt, cols, kind, name, labels, texts)
        x = alt.X("x:Q", title=None, scale=alt.Scale(nice=False, zero=False),
                  axis=alt.Axis(labelFontSize=9, tickCount=3, labelFlush=True, grid=False,
                                labels=last_row, ticks=last_row))
        y_axis = alt.Axis(title=y_title, labelFontSize=9, tickCount=3, titleFontSize=10,
                          titleFontWeight="normal", titleLimit=_PANEL_H[kind])
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
            layers.append(base.mark_point(filled=False, size=26, strokeWidth=1.2,
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
                layers.append(base.mark_text(fontSize=9, color=muted(), baseline="top", dy=2,
                                             dx=dx, align=align, tooltip=False)
                              .encode(x=x, y=alt.value(0), text="note:N")
                              .transform_filter(_only("note_" + end)))
        if (df["mark"] == "stop").any():
            stop_c, stop_o = mark("neutral")
            stop_tip = [alt.Tooltip("why:N", title=_say(opt, wording.PROFILER_STOPPED))]
            layers += [base.mark_rule(strokeDash=[2, 2], color=stop_c, opacity=stop_o)
                       .encode(x=x, tooltip=stop_tip).transform_filter(_only("stop")),
                       base.mark_text(fontSize=9, color=muted(), baseline="bottom", dy=-2,
                                      dx=-3, align="right")
                       .encode(x=x, y=alt.value(_PANEL_H[kind]), text="note:N", tooltip=stop_tip)
                       .transform_filter(_only("stop"))]
        # Each column named once, above the top row (a header under the
        # bottom row's tick labels overlapped them in Streamlit's renderer).
        header = (alt.Header(labelOrient="top", labelFontSize=10, labelLimit=panel_width(len(cols)) + 8,
                             labelExpr="split(datum.label, '\\n')", labelPadding=4,
                             labelBaseline="bottom")
                  if r == 0 else alt.Header(labels=False))
        out_rows.append(
            alt.layer(*layers, data=df).properties(width=panel_width(len(cols)),
                                                  height=_PANEL_H[kind])
            .facet(column=alt.Column("col:N", sort=order, title=None, header=header),
                   spacing=14, bounds="flush")   # review four: tick labels clear
            .resolve_scale(x="independent"))
    chart = alt.vconcat(*out_rows, spacing=8).resolve_scale(
        fill="shared", stroke="shared", strokeDash="shared")
    return titled(chart, _say(opt, wording.PLOT_PROFILER))


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
        chart = profile_chart(opt, profile, shown, rows_shown)
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
                                    profile.fixed, gain=profile.gain_drawn,
                                    settings_only=not opt.has_ingredients())
        if guard.fires:
            # Review four: the guard's long line goes in the "?" (captions
            # at most 8 words); the columns keep Set up's order.
            how = f"{path_guard_line(guard)}\n\n{how}"
        st.caption(_say(opt, caption), help=_say(opt, how))
        if profile.fill:
            st.caption(_say(opt, wording.profiler_fill_note(profile.fill)))
        if profile.greyed:
            # Ruling 2: drawn in grey, never hidden; one line says why.
            st.caption(_say(opt, wording.GAIN_GREYED[hero_reason_key(profile.reason)]),
                       help=_say(opt, wording.tip_gain_greyed(profile.reason)))
        elif profile.reason is not None:
            st.caption(_say(opt, wording.HERO_WITHHELD[hero_reason_key(profile.reason)]),
                       help=_say(opt, wording.tip_hero_hidden(profile.reason)))
        note = moved_note(opt, shown)
        if note is not None:
            st.caption(_say(opt, note[0]), help=_say(opt, note[1]))
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


def _trust_section(opt, slot, computed, facts):
    """"Can we trust the predictions?" (review story): forecast against
    actual over time — each checked formulation's Round-ahead 95 % range
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


def _progress_body(opt, slot):
    """"Are we improving?" (review story): two model-free charts side by
    side — the best so far with each Round's average, the start and the
    best named; and Closer to target, one row per measurement. Drawn at
    once; the answer line is drawn with the cards."""
    with slot.body:
        left, right = st.columns(IMPROVING_SPLIT, gap="medium")
        with left:
            try:
                chart = improving_chart(opt)
            except Exception:
                chart = None
            if chart is None:
                st.caption(_say(opt, wording.CHART_FAILED))
            else:
                show_chart(chart)
                avgs = round_averages(opt)
                first, last = (avgs[0], avgs[-1]) if avgs else (None, None)
                st.caption(_say(opt, wording.STORY_IMPROVING_CAPTION),
                           help=_say(opt, wording.story_improving_help(
                               one_decimal(first[3]) if first else None,
                               one_decimal(last[3]) if last else None,
                               first[0] if first else None, last[0] if last else None)))
        with right:
            try:
                chart = target_chart(opt)
            except Exception:
                chart = None
            if chart is not None:
                st.markdown(f"**{_say(opt, wording.STORY_TARGET_TITLE)}**")
                show_chart(chart)
                s, b = story_start(opt), opt.best_index()
                st.caption(_say(opt, wording.STORY_TARGET_CAPTION),
                           help=_say(opt, wording.story_target_help(
                               int(opt.formulation_ids[s[0]]), int(opt.formulation_ids[b]),
                               s[1])))


def _drives_section(opt, slot, computed, facts):
    """"What drives each result?": the profiler (Overall score, each
    measurement, Expected gain), through Suggestion 1 when there is one;
    the answer a finding (review story): the row that moves the predicted
    score most, and where the reference sits on it."""
    with slot.body:
        try:
            got = _profiler(opt, computed)
        except Exception:
            st.caption(_say(opt, wording.CHART_FAILED))
            got = None
    profile, guard = (got[0], got[2]) if got is not None else (None, None)
    line = driver_finding(opt, profile, guard)
    ev = None
    if got is not None:
        cfg = getattr(opt, "bo_config", None) or {}
        ev = wording.evidence([wording.ev_profiler(opt.complete_count(), profile.ref_label,
                                                   profile.ref_no,
                                                   cfg.get("acquisition", "qlognei"))])
        rule = wording.story_leader_rule(f"{leader_noise(opt):g}")
        ev = f"{wording.story_driver_help(profile.ref_label)} {rule}\n\n{ev}"
    with slot.answer.container():
        st.markdown(_say(opt, line),
                    help=_say(opt, ev) if ev else _evidence_help(opt, facts, "next"))


def tried_answer(opt, strips):
    """Q4's answer: '**30 tested.** Tap water below 31 g untested.'"""
    gaps = [g for g in insights.strip_gaps(opt, strips) if g.kind != "option"]
    by_name = {s.name: s for s in strips}
    if gaps:
        g = gaps[0]
        strip = by_name[g.name]
        if g.kind == "between":
            # Compact: 'Textured soy protein 8.66–10.66 g' keeps the line short.
            d = None if strip.setting else _shared_decimals(_gap_said(g))
            stretch = wording.gap_span(g.name, _strip_number(opt, strip, g.low, False, d),
                                       _strip_number(opt, strip, g.high, True, d))
        else:
            stretch = _gap_text(opt, g, strip)
        rest = wording.q4_gap(stretch)
    else:
        rest = wording.Q4_NO_GAP
    return wording.answer_line(wording.q4_word(opt.complete_count()), rest)


def _tried_section(opt, slot):
    """Q4, "What have we tried?": the parallel coordinates, model-free; the
    answer from the gap lines, every gap and limit-hit line in its "?"."""
    try:
        pool = bo_views.feasible_pool(opt)
        strips = insights.tested_strips(opt, pool)
        answer = tried_answer(opt, strips)
        lines = strip_gap_lines(opt, strips) + limit_hit_lines(opt)
    except Exception:
        answer, lines = None, []
    if answer is not None and opt.complete_count():
        ev = wording.evidence([wording.ev_tried(opt.complete_count())])
        with slot.answer.container():
            st.markdown(_say(opt, answer),
                        help=_say(opt, "\n\n".join([*lines, ev, tooltip_for(opt, "gaps")])))
    with slot.body:
        settings_only = not opt.has_ingredients()
        # Review story: the search path by default; the parallel
        # coordinates on a small toggle, for seeing combinations.
        as_lines = st.toggle(_say(opt, wording.STORY_SHOW_LINES), key=PATH_SHOW_LINES_KEY,
                             help=_say(opt, wording.STORY_SHOW_LINES_HELP),
                             disabled=confirmation_open())
        axes, lines = insights.tried_axes(opt), insights.tried_lines(opt)
        try:
            chart = (tried_chart(opt, axes, lines) if as_lines
                     else search_path_chart(opt, axes, lines))
        except Exception:
            chart = None
        if chart is None:
            st.caption(_say(opt, wording.TRIED_NOTHING))
            return
        if as_lines:
            show_chart(_untitled(chart))
            st.caption(_say(opt, wording.tried_caption_roles()),
                       help=_say(opt, wording.tip_tried_roles(settings_only)))
        else:
            show_chart(chart, width="content")
            st.caption(_say(opt, wording.STORY_PATH_CAPTION),
                       help=_say(opt, wording.story_path_help(settings_only)))


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
IMPROVING_HEIGHT = 250
IMPROVING_SPLIT = (1.1, 1)   # the best-so-far chart a little wider; Closer to target keeps its labels
PATH_SHOW_LINES_KEY = "insights_path_lines"
PATH_COLUMNS = 4             # search path panels per row
_PATH_PANEL_W = 150
_PATH_ROW_H = 16             # px per Round in a search path panel


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
                              domain=False, grid=False, labelFontSize=9, labelPadding=2,
                              labelColor=muted()))
    return (alt.Chart(pd.DataFrame({frame_x: rules}))
            .mark_rule(color=light, strokeWidth=1, tooltip=False).encode(x=top))


def _halo_text(data, x, y, text, color, **kw):
    """A text mark with a page-coloured halo under it, so a label over a
    line or a dot still reads."""
    page = "#0e1117" if theme() == "dark" else "#ffffff"
    under = alt.Chart(data).mark_text(color=page, stroke=page, strokeWidth=4, tooltip=False,
                                      **kw).encode(x=x, y=y, text=text)
    over = alt.Chart(data).mark_text(color=color, tooltip=False, **kw).encode(
        x=x, y=y, text=text)
    return [under, over]


def improving_chart(opt):
    """Are we improving? (a): each formulation's measured Overall score
    (grey dots), Best so far (an ink step), each Round's average (a short
    mid-grey bar under its Formulations) and the start and the best named
    on the chart. Model-free. x: Formulations in the order made, the
    Rounds marked along the top — the same axis as the forecast chart."""
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
    top_room = 0.34 * max(hi - lo, 4.0)    # review story 2: room for the Best label at 230 px
    y_scale = alt.Scale(domain=[lo - 0.04 * max(hi - lo, 4.0), hi + top_room], nice=False,
                        zero=False)
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
    kinds = alt.Scale(domain=[best_name, avg_name], range=[ink, mid])
    legend = alt.Legend(title=None, orient="top", labelLimit=0, symbolType="stroke",
                        symbolStrokeWidth=3)
    no_tip = alt.Tooltip(f"{formulation}:Q", title=_say(opt, wording.FORMULATION_CAP),
                         format="d")
    y = alt.Y(f"{score}:Q", title=y_title, scale=y_scale)
    layers = []
    if avgs:
        adf = pd.DataFrame([{"x0": a - 0.4, "x1": b + 0.4, "avg": m, "what": avg_name,
                             "round": wording.round_tick(r), "n": n,
                             "text": one_decimal(m)} for r, a, b, m, n in avgs])
        layers.append(alt.Chart(adf).mark_rule(strokeWidth=3).encode(
            x=alt.X("x0:Q", scale=x_scale), x2="x1:Q",
            y=alt.Y("avg:Q", scale=y_scale, title=y_title),
            color=alt.Color("what:N", scale=kinds, legend=legend),
            tooltip=[alt.Tooltip("round:N", title=_say(opt, wording.ROUND_CAP)),
                     alt.Tooltip("text:N", title=avg_name)]))
    layers.append(alt.Chart(frame).mark_point(size=30, strokeWidth=1.3, opacity=0.9,
                                              color=grey).encode(
        x=x, y=y,
        fill=alt.condition(alt.FieldEqualPredicate(field="full", equal=True),
                           alt.value(grey), alt.value("transparent")),
        tooltip=[no_tip, alt.Tooltip("round:N", title=_say(opt, wording.ROUND_CAP)),
                 alt.Tooltip(f"{score}:Q", title=score, format=".1f"), bo_tip()])
        .transform_filter(alt.FieldValidPredicate(field=score, valid=True)))
    layers.append(alt.Chart(frame).mark_line(strokeWidth=2.5, interpolate="step-after").encode(
        x=x, y=alt.Y(f"{best}:Q", scale=y_scale, title=y_title),
        color=alt.Color("best_series:N", scale=kinds, legend=legend),
        tooltip=[no_tip, alt.Tooltip(f"{best}:Q", title=best, format=".1f")]))
    if nums is not None:
        s, r0, b, rb, bno = nums
        sno = int(opt.formulation_ids[story_start(opt)[0]])
        lo_x, hi_x = _x_domain(opt)
        span = hi_x - lo_x
        pts = []
        if (sno, s) != (bno, b):
            pts.append({"x": sno, "y": s, "text": wording.story_point(
                wording.STORY_START, one_decimal(s), r0), "who": "start"})
        if rb is not None:
            pts.append({"x": bno, "y": b, "text": wording.story_point(
                wording.STORY_BEST, one_decimal(b), rb), "who": "best"})
        for p in pts:
            share = (p["x"] - lo_x) / span if span else 0.5
            align = "left" if share < 0.3 else "right" if share > 0.7 else "center"
            dx = {"left": -4, "right": 4, "center": 0}[align]
            pdf = pd.DataFrame([p])
            ring_c = ink if p["who"] == "best" else grey
            layers.append(alt.Chart(pdf).mark_point(size=150, strokeWidth=1.8, color=ring_c,
                                                    filled=False, tooltip=False)
                          .encode(x=alt.X("x:Q", scale=x_scale),
                                  y=alt.Y("y:Q", scale=y_scale, title=y_title)))
            layers += _halo_text(pdf, alt.X("x:Q", scale=x_scale),
                                 alt.Y("y:Q", scale=y_scale, title=y_title), "text:N",
                                 ink if p["who"] == "best" else muted(), fontSize=11,
                                 fontWeight="bold" if p["who"] == "best" else "normal",
                                 align=align, dx=dx, dy=-15, baseline="bottom")
    rounds = _round_layer(opt, formulation, list(frame[formulation]), x_scale)
    body = alt.layer(*layers)
    chart = alt.layer(rounds, body).resolve_axis(x="independent") if rounds else body
    return chart.properties(height=IMPROVING_HEIGHT)


def target_rows(opt):
    """Closer to target: one row per measurement (largest Share of score
    first) — name, goal, target, Range, the start's value and Best so
    far's — or [] with no start or no best."""
    s = story_start(opt)
    b = opt.best_index()
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


def target_chart(opt):
    """Are we improving? (b), "Closer to target": one row per measurement
    on its own Range (drawn 0 to 1 across, its ends written): the first
    Round's best (hollow grey), Best so far (ink) joined by a grey line, the
    target as a dashed tick, or an arrow at the better end for a 'higher' or
    'lower' measurement. Model-free. None with nothing to draw."""
    rows = target_rows(opt)
    if not rows:
        return None
    s, b = story_start(opt), opt.best_index()
    start_name = _say(opt, wording.story_target_start(s[1]))
    best_name = _say(opt, wording.STORY_TARGET_BEST)
    grey, _ = mark("neutral")
    ink, _ = mark("bar_latest")
    ref, _ = mark("reference")
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
            # The dashed tick only: its row's label names the target.
            targets.append({"t": _share(r, r["target"]), "y0": k - 0.34, "y1": k + 0.34})
        elif r["goal"] in ("max", "min"):
            arrows.append({"k": k, "t": 1.0 if r["goal"] == "max" else 0.0,
                           "shape": "triangle-right" if r["goal"] == "max" else "triangle-left"})
        if r["start"] is not None and r["best"] is not None:
            joins.append({"k": k, "t0": _share(r, r["start"]), "t1": _share(r, r["best"])})
        for v, kind in ((r["start"], start_name), (r["best"], best_name)):
            if v is not None:
                points.append({"k": k, "t": _share(r, v), "what": kind, "value": fmt(v),
                               "tip": tip})
        if r["best"] is not None:
            texts.append({"k": k, "t": _share(r, r["best"]), "text": fmt(r["best"]),
                          "who": "best"})
        # The start's value under its dot, the best's above: both always
        # written, so a start that already sat on the target says so.
        if r["start"] is not None and (r["best"] is None
                                       or abs(_share(r, r["start"]) - _share(r, r["best"]))
                                       > 0.005):
            texts.append({"k": k, "t": _share(r, r["start"]), "text": fmt(r["start"]),
                          "who": "start"})
    n = len(rows)
    y_scale = alt.Scale(domain=[-0.7, n - 0.3], reverse=True, nice=False, zero=False)
    expr = " : ".join(f"datum.value == {k} ? '{_esc(l)}'" for k, l in enumerate(labels)) + " : ''"
    y_axis = alt.Axis(values=list(range(n)), labelExpr=expr, title=None, ticks=False,
                      domain=False, grid=False, labelFontSize=11, labelLimit=180, labelPadding=26)
    y = alt.Y("k:Q", scale=y_scale, axis=y_axis)
    x_scale = alt.Scale(domain=[-0.02, 1.02], nice=False)
    x = alt.X("t:Q", scale=x_scale, axis=None)
    kinds = [start_name, best_name]
    colour = alt.Scale(domain=kinds, range=[grey, ink])
    fill = alt.Scale(domain=kinds, range=["transparent", ink])
    legend = alt.Legend(title=None, orient="bottom", labelLimit=0, symbolStrokeWidth=2)
    layers = [
        alt.Chart(pd.DataFrame(track)).mark_rule(color=light, strokeWidth=4, tooltip=False)
        .encode(x=alt.X("t0:Q", scale=x_scale, axis=None), x2="t1:Q", y=y),
        alt.Chart(pd.DataFrame(ends)).mark_text(fontSize=9, color=muted(), tooltip=False,
                                                dx=-5, align="right", baseline="middle")
        .encode(x=x, y=y, text="text:N").transform_filter("datum.align == 'right'"),
        alt.Chart(pd.DataFrame(ends)).mark_text(fontSize=9, color=muted(), tooltip=False,
                                                dx=5, align="left", baseline="middle")
        .encode(x=x, y=y, text="text:N").transform_filter("datum.align == 'left'"),
    ]
    if joins:
        layers.append(alt.Chart(pd.DataFrame(joins)).mark_rule(color=grey, strokeWidth=1.5,
                                                               tooltip=False)
                      .encode(x=alt.X("t0:Q", scale=x_scale, axis=None), x2="t1:Q", y=y))
    if targets:
        # Taller than the dots, so a target hit exactly still shows above
        # and below the ink dot.
        layers.append(alt.Chart(pd.DataFrame(targets)).mark_rule(
            color=ref, strokeWidth=1.6, strokeDash=[3, 2], tooltip=False)
            .encode(x=x, y=alt.Y("y0:Q", scale=y_scale, axis=y_axis), y2="y1:Q"))
    if arrows:
        layers.append(alt.Chart(pd.DataFrame(arrows)).mark_point(
            filled=True, size=70, color=ref, tooltip=False)
            .encode(x=x, y=y, shape=alt.Shape("shape:N", scale=None)))
    pdf = pd.DataFrame(points)
    tips = [alt.Tooltip("what:N", title=" "), alt.Tooltip("value:N", title=wording.STORY_VALUE),
            alt.Tooltip("tip:N", title=_say(opt, wording.STORY_TARGET_TITLE))]
    layers.append(alt.Chart(pdf).mark_point(size=110, strokeWidth=2, opacity=1)
                  .encode(x=x, y=y, color=alt.Color("what:N", scale=colour, legend=legend),
                          fill=alt.Fill("what:N", scale=fill, legend=legend),
                          tooltip=tips))
    if texts:
        tdf = pd.DataFrame(texts)
        layers += _halo_text(tdf[tdf["who"] == "best"], x, y, "text:N", ink, fontSize=10,
                             fontWeight="bold", dy=-11, baseline="bottom")
        if (tdf["who"] == "start").any():
            layers += _halo_text(tdf[tdf["who"] == "start"], x, y, "text:N", muted(),
                                 fontSize=10, dy=10, baseline="top")
    return alt.layer(*layers).properties(height=max(IMPROVING_HEIGHT - 70, 45 * n))


def _esc(text):
    return str(text).replace("\\", "\\\\").replace("'", "\\'")


def forecast_chart(opt, checks):
    """Can we trust the predictions? — forecast against actual, over time:
    for each checked formulation, the 95 % range the model predicted for
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
    band_name = wording.STORY_FORECAST
    blue = model_blue()
    band_c, _ = _BAND_FILL[theme()]
    ink, _ = mark("bar_latest")
    df = pd.DataFrame([{
        formulation: int(c.formulation), "round": wording.round_tick(c.round),
        "predicted": round(c.predicted, 1), "low": round(c.low, 1), "high": round(c.high, 1),
        "measured": round(c.measured, 1),
        "state": inside_name if c.inside else outside_name, "band": band_name,
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
    fills = alt.Scale(domain=[inside_name, outside_name], range=[ink, "transparent"])
    dots = alt.Chart(df).mark_point(size=46, strokeWidth=1.6, color=ink, opacity=1).encode(
        x=x, y=y, fill=alt.Fill("state:N", scale=fills, legend=legend), tooltip=tip)
    # The predicted value itself: a short blue tick across its bar.
    ticks = alt.Chart(df).mark_tick(color=blue, thickness=2, size=11, opacity=1, orient="horizontal").encode(
        x=x, y=alt.Y("predicted:Q", scale=y_scale, title=y_title), tooltip=tip)
    layers = [bars, ticks, dots]
    first_checked = min(int(c.formulation) for c in mine)
    before = [n for n in numbers if n < first_checked]
    if len(before) >= 3:
        note = pd.DataFrame([{"x": (min(before) + max(before)) / 2, "y": (lo + hi) / 2,
                              "text": wording.STORY_NO_FORECAST}])
        layers.append(alt.Chart(note).mark_text(fontSize=10, color=muted(), tooltip=False)
                      .encode(x=alt.X("x:Q", scale=x_scale), y=alt.Y("y:Q", scale=y_scale),
                              text="text:N"))
    rounds = _round_layer(opt, formulation, numbers, x_scale)
    body = alt.layer(*layers)
    chart = alt.layer(rounds, body).resolve_axis(x="independent") if rounds else body
    return chart.properties(height=STORY_Q_HEIGHT - 30)


def trust_counts(checks):
    """(inside, total, above, below) of the Overall score's Round-ahead
    checks: misses above and below their range counted apart."""
    mine = [c for c in checks or () if c.name == insights.SCORE]
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


def search_path_chart(opt, axes, lines):
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
    look = {wording.STORY_PATH_TESTED: mid, wording.TRIED_BEST: ink,
            wording.TRIED_SUGGESTED: orange, wording.TRIED_WAITING: grey}
    colour = alt.Scale(domain=kinds, range=[look[k] for k in kinds])
    legend = alt.Legend(title=None, orient="bottom", labelLimit=0)
    word = _say(opt, wording.FORMULATION_CAP)
    height = max(70, _PATH_ROW_H * len(rounds))
    y_scale = alt.Scale(domain=labels, paddingInner=0.2, paddingOuter=0.3)
    panels = []
    for k, a in enumerate(axes):
        mine = df[df["axis"] == a.name]
        first = k % PATH_COLUMNS == 0
        y = alt.Y("rlabel:O", scale=y_scale, sort=labels,
                  title=_say(opt, wording.ROUND_CAP) if first else None,
                  axis=alt.Axis(labels=first, ticks=False, domain=False, grid=False,
                                labelFontSize=9, labelColor=muted(), titleFontSize=10)
                  if first else None)
        if a.options is not None:
            x_scale = alt.Scale(domain=list(a.options))
            x_axis = alt.Axis(labelAngle=0, labelFontSize=9, labelLimit=60, ticks=False,
                              domain=False)
            x = alt.X("value:N", scale=x_scale, title=None, axis=x_axis)
        else:
            d = _shared_decimals([a.low, a.high])
            x_scale = alt.Scale(domain=[a.low, a.high], nice=False, zero=False)
            x_axis = alt.Axis(values=[a.low, a.high], grid=False, ticks=False, domain=False,
                              labelFontSize=9, labelColor=muted(),
                              labelExpr=_ends_expr(opt, a, d), labelFlush=True)
            x = alt.X("value:Q", scale=x_scale, title=None, axis=x_axis)
        tip = [alt.Tooltip("no:Q", title=word, format="d"),
               alt.Tooltip("rlabel:N", title=_say(opt, wording.ROUND_CAP)),
               alt.Tooltip("text:N", title=a.name),
               alt.Tooltip("score:N", title=_say(opt, wording.OVERALL_SCORE_COLUMN))]
        base = alt.Chart(mine)
        layers = []
        if a.options is None:
            layers.append(
                base.transform_filter(f"datum.role != '{_esc(wording.TRIED_SUGGESTED)}'")
                .transform_aggregate(lo="min(value)", hi="max(value)", groupby=["rlabel"])
                .mark_rule(color=light, strokeWidth=5, strokeCap="round", tooltip=False)
                .encode(x=alt.X("lo:Q", scale=x_scale, title=None, axis=x_axis), x2="hi:Q",
                        y=y))
        layers.append(base.mark_point(size=26, filled=True, opacity=0.95).encode(
            x=x, y=y, color=alt.Color("role:N", scale=colour, legend=legend), tooltip=tip)
            .transform_filter(f"datum.role != '{_esc(wording.TRIED_BEST)}' && "
                              f"datum.role != '{_esc(wording.TRIED_WAITING)}'"))
        # Waiting for results (or Not scored): a hollow grey ring.
        layers.append(base.mark_point(size=30, filled=False, strokeWidth=1.4).encode(
            x=x, y=y, color=alt.Color("role:N", scale=colour, legend=legend), tooltip=tip)
            .transform_filter(f"datum.role == '{_esc(wording.TRIED_WAITING)}'"))
        layers.append(base.mark_point(size=110, strokeWidth=2, filled=False, color=ink)
                      .encode(x=x, y=y, tooltip=tip)
                      .transform_filter(f"datum.role == '{_esc(wording.TRIED_BEST)}'"))
        layers.append(base.mark_point(size=26, filled=True).encode(
            x=x, y=y, color=alt.Color("role:N", scale=colour, legend=legend), tooltip=tip)
            .transform_filter(f"datum.role == '{_esc(wording.TRIED_BEST)}'"))
        title = alt.TitleParams(_axis_label(a), fontSize=10, fontWeight="normal",
                                anchor="start", color=None, limit=_PATH_PANEL_W + 20)
        panels.append(alt.layer(*layers).properties(width=_PATH_PANEL_W, height=height,
                                                     title=title))
    return (alt.concat(*panels, columns=PATH_COLUMNS, spacing=18)
            .resolve_scale(color="shared", x="independent")
            .configure_view(strokeWidth=0))


def _ends_expr(opt, a, d):
    """The x axis' two labels: Lowest and Highest with the unit."""
    lo, hi = _axis_end(opt, a, a.low, d), _axis_end(opt, a, a.high, d)
    return (f"abs(datum.value - {a.low}) < 1e-9 ? '{_esc(lo)}' : "
            f"abs(datum.value - {a.high}) < 1e-9 ? '{_esc(hi)}' : ''")
