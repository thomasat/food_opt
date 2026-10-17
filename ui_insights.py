"""The Insights tab. Builders here return Altair charts from computed facts
and never call Streamlit; render() at the bottom is the only Streamlit code."""
import collections
import contextlib
import dataclasses
import html
import math
import re
import threading
import time

import altair as alt
import pandas as pd
import streamlit as st

import bo_views
import insights
import insights_cache
import insights_jobs
import sample_projects
import wording
from food_bo import fmt_setting, join_unit
from ui_helpers import (SCROLL_KEY, TAB_BATCH, app_rerun, confirmation_open, go_to_tab,
                        open_rows, readiness)

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


# Slim Review: a Round step redraws two charts. Building an
# Altair chart and converting it is most of a step's time (about 0.5 s on
# the yogurt pilot), so the section's charts are converted once — through
# this file's one door (_private, readable) and Streamlit's own conversion —
# and the Vega-Lite spec kept; a step sends the kept spec.
_SPEC_CACHE = collections.OrderedDict()
_SPEC_KEEP = 64
_SPEC_LOCK = threading.Lock()


def _vega_spec(chart):
    """The spec st.altair_chart would send for `chart`, privacy and type
    scale applied: every chart the app draws goes through here and
    show_spec, so none of a project's data can leave the machine through
    its action menu."""
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
    """Draw a kept spec (chart_spec), the app's one chart call: the action menu stays off
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


def bo_tip(field="bo"):
    """The tooltip's expert row."""
    return alt.Tooltip(f"{field}:N", title=wording.PLOT_BO)


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
STAND_SECTION = "insights_stand"
HOW_KEY = "insights_how_fold"
# The "Predicted above Best so far" outline: Okabe-Ito orange, which holds
# its contrast on white and on Streamlit's dark background (review M14).
_OUTLINE_LIGHT = "#B35C00"     # about 4.6:1 on white (T56 m4)
_OUTLINE_DARK = "#E69F00"      # on Streamlit's dark background


def outline_colour():
    """The Predicted above Best so far outline for the reader's theme."""
    return _OUTLINE_DARK if muted() == _MUTED_DARK else _OUTLINE_LIGHT


# Direction against a target (Results, the score fold, Review's strip) or
# against Best so far (Make a round): above / more orange, below / less
# purple, per theme (owner, v3 final cleanup; it replaced one orange for
# "off target"). Each at least 4.5:1 on its theme's page.
_DIRECTION = {"light": {"above": "#c2410c", "below": "#7e3fbf"},
              "dark": {"above": "#ff9f5a", "below": "#c79bff"}}


def direction_colour(kind, which=None):
    """The colour of `kind` ('above' | 'below') in theme `which` (the
    reader's when None)."""
    return _DIRECTION[which or theme()][kind]


def direction_key_html(head, above, below, tail="", which=None):
    """A caption's key: `head`, then `above` in the above colour and
    `below` in the below colour, bold, then `tail` — for st.caption with
    unsafe_allow_html (st.markdown's own colours are named, not tokens)."""
    def span(text, kind):
        return (f'<span style="color: {direction_colour(kind, which)}; font-weight: 600">'
                f'{html.escape(text)}</span>')
    parts = [span(above, "above"), span(below, "below")] + ([html.escape(tail)] if tail else [])
    key = " · ".join(parts)
    return f"{html.escape(head)} {key}" if head else key


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


def render(opt):
    """The Review, v3 (owner-approved 2026-10-16; final-v3/wording.md §4):
    Are we improving? — a factual headline, then one panel at the Round on
    view (◀ ▶ / Play): Best so far, Best so far vs targets and Amounts in
    the top formulations — then What next? (buttons only) and Definitions.
    Measurements only: nothing here waits for a model, and nothing says a
    verdict, a forecast or a stop."""
    # A formulation moved to Not scored counts for nothing here (review M-3).
    if len(opt.Y_history) == sum(map(opt.is_set_aside, range(len(opt.Y_history)))):
        if _example_setup(opt):
            return
        st.caption(_say(opt, wording.INSIGHTS_NONE))
        _foot(opt)
        return
    if round_view(opt) is None:
        st.caption(_say(opt, wording.INSIGHTS_NONE))
        _next_section(opt)
        _how(opt)
        return
    _land()
    with st.container(key=STAND_SECTION, gap="small"):
        st.markdown(f"#### {_say(opt, wording.Q2_TITLE)}")
        _progress_body(opt)
    _next_section(opt)
    _how(opt)


# Where links land: Are we improving? (Make a round's link) and What next?.
NEXT_SECTION = "insights_next"
HEADLINE_KEY = "insights_headline"
HEADLINE_STYLE = """<style>
.st-key-insights_headline [data-testid="stMarkdownContainer"] p { font-size: 1.75rem;
  line-height: 1.3; margin: 0; }
</style>"""


def _next_section(opt):
    """What next? (final-v3 §4): buttons only — the primary to the next
    Round (or back to the open one), and, on the Round 6 sample once Round 6
    is recorded, the pilot's Rounds 7–10."""
    with st.container(key=NEXT_SECTION, gap="small"):
        st.markdown(f"#### {_say(opt, wording.Q5_TITLE)}")
        with st.container(horizontal=True, gap="small", vertical_alignment="center"):
            _foot(opt)
            _what_happened_next(opt)


def _example_setup(opt):
    """The example setup's Review, which has no results by design (spec
    2026-10-13 §9): says so, and offers the yogurt pilot, whose Review has
    real ones (app.py opens it on the run the key asks for it). Whether
    it drew anything. Only while the example is as shipped — no result,
    nothing set aside, no Round open; after that the normal Review (and
    its foot back to the open Round) is the true one."""
    if not sample_projects.untouched_example(opt):
        return False
    st.caption(wording.BURGER_NO_RESULTS)
    if st.button(wording.OPEN_YOGURT_LABEL, key="sample_open_yogurt",
                 help=wording.OPEN_YOGURT_HELP):
        st.session_state[sample_projects.OPEN_NEXT_KEY] = sample_projects.ROUND6
        st.rerun()
    return True


def _what_happened_next(opt):
    """The Round 6 sample, Round 6 recorded: "See the pilot's Rounds 7–10",
    secondary, beside the primary. app.py opens the all-10 sample on the
    run the key asks for it."""
    if sample_projects.offers_what_next(opt):
        if st.button(wording.SEE_WHAT_HAPPENED_NEXT, key="sample_what_next",
                     disabled=confirmation_open()):
            st.session_state[sample_projects.OPEN_NEXT_KEY] = True
            st.rerun()


def _foot(opt):
    """What next?'s primary, by Results' rule (spec 2026-09-30 §2.4, review
    M13): back to the open Round, else 'Start Round 7' (the next Round's
    own number)."""
    if opt.pending_batch:
        label = wording.back_to_batch_label(opt.pending_batch_no, len(open_rows(opt)))
    else:
        label = wording.start_round_label(opt.next_batch_no())
    ready, _ = readiness(opt)
    lit = ready and not confirmation_open()
    if st.button(label, type="primary" if lit else "secondary", disabled=not lit,
                 key="insights_foot") and lit:
        go_to_tab(TAB_BATCH)


def _land():
    """A link that lands on Definitions opens it (review polish: the method
    link). Only on arrival (ui_helpers.SCROLL_KEY, drained by app.py after
    the tab is drawn), so a fold the reader closed stays closed."""
    if st.session_state.get(SCROLL_KEY) == HOW_PLACE:
        st.session_state[HOW_KEY] = True


# ------------------------------------------------------------------ #
#  Review v3: what the Review shows at one Round, read from the
#  measurements alone (no model). Ties are named, never picked from;
#  score comparisons use SCORE_TOL.
# ------------------------------------------------------------------ #

TOP_COUNT = 3             # the top formulations: the 3 best so far, ties at the 3rd included
SCORE_TOL = 1e-6          # points: two scores this close are equal (ties; the top's edge)
# One marker per tied best, the same on every chart (F16 a dot, F17 a diamond).
BEST_SHAPES = ("circle", "diamond", "square", "triangle-up", "triangle-down", "cross")


@dataclasses.dataclass
class RoundView:
    """The Review at one Round (round_view)."""
    round: int | None       # the Round on view (the latest when none was chosen)
    best: float             # Best so far by then
    bests: list             # positions at it (ties), in the order made
    rounds: list            # the Rounds those were made in
    new: bool               # Best so far first reached in the Round on view
    top: list               # the TOP_COUNT best (ties at the last included), highest first
    tried: list             # every formulation recorded by then, scored or not
    scored: list            # every Best-so-far candidate by then
    was: float | None = None                                  # the best before
    was_bests: list = dataclasses.field(default_factory=list)
    was_rounds: list = dataclasses.field(default_factory=list)


def _made_by(opt, i, r):
    """Whether position `i` was recorded by the end of Round `r` (None: by
    now; a formulation with no Round always counts)."""
    rr = _round_of(opt, i)
    return r is None or rr is None or rr <= r


def round_view(opt, r=None):
    """The Review's facts by the end of Round `r` (None: today), or None
    with nothing scored by then. Best so far and its ties (within
    SCORE_TOL), the Round it was first reached in and the best before that
    Round, the top formulations (the TOP_COUNT highest scores, with any
    tied with the last of them within SCORE_TOL: Round 6 gives F16, F17,
    F10; the pilot's 10 Rounds F21 and five at 98.33), and every
    formulation tried."""
    pool = [i for i in opt.best_candidates() if _made_by(opt, i, r)]
    if not pool:
        return None

    def y(i):
        return float(opt.Y_history[i])
    best = max(y(i) for i in pool)
    bests = [i for i in pool if y(i) >= best - SCORE_TOL]
    ranked = sorted(pool, key=lambda i: (-round(y(i), 6), int(opt.formulation_ids[i])))
    if len(ranked) > TOP_COUNT:
        last = y(ranked[TOP_COUNT - 1])
        top = [i for i in ranked if y(i) >= last - SCORE_TOL]
    else:
        top = ranked
    rounds = sorted({_round_of(opt, i) for i in bests if _round_of(opt, i) is not None})
    named = [_round_of(opt, i) for i in pool if _round_of(opt, i) is not None]
    on_view = r if r is not None else (max(named) if named else None)
    first = rounds[0] if rounds else None
    view = RoundView(on_view, best, bests, rounds, first is None or first == on_view, top,
                     [i for i in range(len(opt.X_history)) if _made_by(opt, i, r)], pool)
    if first is not None:
        before = [i for i in pool if _round_of(opt, i) is not None and _round_of(opt, i) < first]
        if before:
            view.was = max(y(i) for i in before)
            view.was_bests = [i for i in before if y(i) >= view.was - SCORE_TOL]
            view.was_rounds = sorted({_round_of(opt, i) for i in view.was_bests})
    return view


def _names(opt, positions):
    """'F16', 'F17' … for positions, as the project says them."""
    return [_short(opt, f"F{int(opt.formulation_ids[i])}") for i in positions]


def headline_text(opt, view):
    """'Best 98.3: F16, F17 tied (Round 6). Was 95.0 (F10, Round 4).'"""
    was = None if view.was is None else one_decimal(view.was)
    return _say(opt, wording.review_headline(
        one_decimal(view.best), _names(opt, view.bests), view.rounds, new=view.new, was=was,
        was_names=_names(opt, view.was_bests), was_rounds=view.was_rounds))


def _measure_text(v):
    """A measured value as the panel gave it: '4', '12', '4.5'."""
    v = float(v)
    return str(int(round(v))) if abs(v - round(v)) < 1e-9 else _num(v)


def ideal_value(obj):
    """The value that earns an attribute its whole share: its target, or
    the top (max) or bottom (min) of its Range."""
    lo, hi = float(obj.get('min_val', 0.0)), float(obj.get('max_val', 10.0))
    if obj.get('goal') == 'target':
        t = obj.get('target')
        return float(t) if t is not None else (lo + hi) / 2
    return hi if obj.get('goal', 'max') == 'max' else lo


def point_cost(obj, value):
    """The points `value` loses against the attribute's ideal: its share ×
    the distance as a share of its Range (food_bo._compute_utility's rule:
    each point off costs 1/Range of the share)."""
    lo, hi = float(obj.get('min_val', 0.0)), float(obj.get('max_val', 10.0))
    rng = (hi - lo) or 1.0
    return float(obj['weight']) * min(1.0, abs(float(value) - ideal_value(obj)) / rng)


def cost_example(opt, view):
    """The score caption's example, from the best so far's costliest miss:
    'Example: Tangy 4 instead of 5 costs 1.7 points.'; None when every
    attribute of every tied best is on its ideal."""
    worst = None
    for i in view.bests:
        res = opt._results_at(i)
        for obj in opt.measurements_by_importance():
            v = res.get(obj['name'])
            if v is None:
                continue
            cost = point_cost(obj, v)
            if cost > SCORE_TOL and (worst is None or cost > worst[0] + SCORE_TOL):
                worst = (cost, obj, v)
    if worst is None:
        return None
    cost, obj, v = worst
    return _say(opt, wording.review_cost_example(obj['name'], _measure_text(v),
                                                 _measure_text(ideal_value(obj)), f"{cost:.1f}"))


def made_once(opt, view=None):
    """Whether every scored formulation was made once (no Repeats on file)."""
    rows = list(view.scored) if view is not None else list(opt.best_candidates())
    try:
        return not insights.repeat_groups(opt, rows)
    except Exception:
        return True


def score_caption(opt, view):
    """(caption, its "?") under the score chart."""
    caption = wording.review_score_caption(sample_projects.panel_size(opt),
                                           once=made_once(opt, view))
    return _say(opt, caption), _say(opt, wording.review_score_help(cost_example(opt, view)))


def scale_span_text(opt):
    """The span every attribute's Range shares ('15'), or None when they
    differ: each point off costs 1/span of an attribute's share."""
    spans = {float(o.get('max_val', 10.0)) - float(o.get('min_val', 0.0))
             for o in opt.objectives}
    if len(spans) != 1:
        return None
    span = spans.pop()
    return f"{span:g}" if span > 0 else None


def definitions(opt):
    """Definitions' lines for this project (wording.method_definitions)."""
    try:
        view = round_view(opt)
        tags = {r["tag"] for r in amounts_rows(opt, view)} if view is not None else set()
    except Exception:
        tags = set()
    return [_say(opt, line) for line in wording.method_definitions(
        top=f"{float(opt.utility_ceiling()):g}", span=scale_span_text(opt),
        count=f"{TOP_COUNT}", once=made_once(opt), limits=tuple(tags - {None}))]


def _progress_body(opt):
    """Are we improving?'s panel (one fragment: a Round step reruns only
    it). A full run stops a Play."""
    if not _fragment_rerun():
        st.session_state.pop(REPLAY_PLAY, None)    # a full run stops a Replay
    st.html(REPLAY_STYLE + HEADLINE_STYLE)
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
    """The panel at the Round on view (shown_round: the latest unless an
    earlier one is chosen): the headline, the Round control, Best so far
    beside Best so far vs targets, then Amounts in the top formulations.
    Every part draws as many elements at every Round, so a step never
    leaves one behind. One scored Round: the same panel without its
    control."""
    rounds = _section_rounds(opt)
    r = shown_round(opt, rounds)
    control = len(rounds) > 1
    past = control and r != rounds[-1]
    view = round_view(opt, r if control else None)
    if view is None:
        return
    with st.container(gap="small", key=f"{REPLAY_PANEL}_{'past' if past else 'now'}"):
        with st.container(key=HEADLINE_KEY):
            st.markdown(f"**{headline_text(opt, view)}**")
        if control:
            _replay_bar(opt, rounds, r)
        with st.container(horizontal=True, gap="medium", key=PANEL_ROW):
            with st.container(gap="xsmall", key=PANEL_LEFT):
                _improving_part(opt, r if control else None, view)
            with st.container(gap="xsmall", key=PANEL_RIGHT):
                _targets_part(opt, view)
        _amounts_part(opt, view)
    if control:
        _replay_tick(opt, rounds, r)


def _improving_part(opt, r, view):
    """Best so far (its title, the chart, its caption and "?")."""
    st.markdown(f"**{_say(opt, wording.PANEL_SCORE_TITLE)}**")
    try:
        spec = improving_spec(opt, r, height=PANEL_CHART_H)
    except Exception:
        spec = None
    if spec is None:
        st.caption(_say(opt, wording.CHART_FAILED))
    else:
        show_spec(spec, clear=True)
    caption, help_ = score_caption(opt, view)
    st.caption(caption, help=help_)


def _targets_part(opt, view):
    """Best so far vs targets (its title and the strip)."""
    st.markdown(f"**{_say(opt, wording.review_targets_title(_names(opt, view.bests)))}**")
    try:
        spec = chart_spec(("targets", insights.data_key(opt), insights.setup_key(opt), theme(),
                           view.round, _say(opt, wording.ROUND_CAP),
                           sample_projects.panel_size(opt)),
                          lambda: targets_chart(opt, view))
    except Exception:
        spec = None
    if spec is None:
        st.caption(_say(opt, wording.CHART_FAILED))
    else:
        show_spec(spec, clear=True)


def _amounts_part(opt, view):
    """Amounts in the top N formulations: its title (Top's rule in its
    "?"), the linked-total line (held empty without a total), the chart."""
    st.markdown(f"**{_say(opt, wording.review_amounts_title(len(view.top)))}**",
                help=_say(opt, wording.review_amounts_help(f"{TOP_COUNT}")))
    line = amounts_linked_text(opt)
    if line:
        st.caption(line)
    else:
        _hold(1)
    try:
        spec = chart_spec(("amounts", insights.data_key(opt), insights.setup_key(opt), theme(),
                           view.round, _say(opt, wording.ROUND_CAP)),
                          lambda: amounts_chart(opt, view))
    except Exception:
        spec = None
    if spec is None:
        st.caption(_say(opt, wording.CHART_FAILED))
    else:
        show_spec(spec, clear=True)


# ---- Best so far (the score chart) ----------------------------------- #

PANEL_CHART_H = 250          # px: the score chart's plot in the panel
_TOP_LANE = 18               # px above the top score for its line's label
# px over that for the Best label: its own lane, so no Round line, dot or
# other label ever runs through it (owner, v3 final fixes).
_BEST_LANE = 20


def improving_spec(opt, r=None, height=None):
    """The score chart as a kept spec at Round `r` (None: today): an
    earlier Round draws only what was measured by its end, on the latest's
    axes, so a step never moves them."""
    rounds = _section_rounds(opt) if r is not None else []
    upto = r if r is not None and rounds and r in rounds[:-1] else None
    key = ("improving", insights.data_key(opt), insights.setup_key(opt), theme(),
           _say(opt, wording.ROUND_CAP), float(opt.utility_ceiling()), height, upto)
    return chart_spec(key, lambda: improving_chart(opt, height=height, upto=upto))


def improving_chart(opt, height=None, upto=None):
    """Best so far (final-v3): each formulation's measured Overall score (a
    grey dot; the best so far's own dots blue, ties alike at their one
    value), Best so far (an ink step, moving at Round boundaries), each
    Round's average (a light bar), the Rounds named along the top, the top
    score's dotted line labelled '100 = every attribute on target' and the
    best labelled 'Best: 98.3 (F16, F17 tied)' in its own lane over the
    plot (the Round lines stop at the 100 line, under both labels, so
    nothing runs through it). Model-free; no stop mark, no
    zone. `upto`: an earlier Round on view — only what was measured by its
    end (the axes as at the latest). None with nothing to draw."""
    plot_h = height or PANEL_CHART_H
    frame = progress_frame(opt).reset_index()
    if frame.empty:
        return None
    formulation = frame.columns[0]
    score = wording.OVERALL_SCORE_COLUMN
    values = [float(v) for v in frame[score] if v is not None and v == v]
    view = round_view(opt, upto)
    if not values or view is None:
        return None
    n = len(frame)
    rounds_of = [_round_of(opt, i) if i < len(opt.batch_history) else None for i in range(n)]
    pool, bests = set(opt.best_candidates()), set(view.bests)
    frame["full"] = [i in pool for i in range(n)]
    frame["best"] = [i in bests for i in range(n)]
    frame["round"] = [_round_label(r) for r in rounds_of]
    frame["bo"] = wording.PLOT_BO_MODEL_FREE
    avgs = round_averages(opt)
    best_col = wording.BEST_SO_FAR_COLUMN
    if upto is not None:
        later = [r is not None and r > upto for r in rounds_of]
        frame.loc[later, score] = None
        frame.loc[later, best_col] = None
        avgs = [a for a in avgs if a[0] <= upto]
    shown = frame[frame[score].notna()]
    lo, hi = min(values), max(values)
    ceiling = float(opt.utility_ceiling())
    show_top = ceiling >= hi - SCORE_TOL and ceiling - hi <= max(hi - lo, 4.0)
    pad = 0.04 * max(hi - lo, 4.0)
    bottom = lo - pad
    data_top = (ceiling if show_top else hi) + pad / 2
    lane = (_TOP_LANE if show_top else 0) + _BEST_LANE
    per_px = (data_top - bottom) / max(1, plot_h - lane)
    top = data_top + per_px * lane
    # The Best label's lane is the topmost; the Round lines stop at the
    # 100 line (or the data's top without it), below both labels.
    best_y = data_top + per_px * (lane - _BEST_LANE / 2)
    rules_top = ceiling if show_top else data_top
    y_scale = alt.Scale(domain=[bottom, top], nice=False, zero=False)
    x_scale = alt.Scale(domain=_x_domain(opt), nice=False)
    span = data_top - bottom
    step = 5 if span <= 30 else 10 if span <= 60 else 20
    first = math.ceil(bottom / step) * step
    ticks = [first + step * k for k in range(int((data_top - first) // step) + 1)]
    y_title = _say(opt, score)
    y = alt.Y(f"{score}:Q", title=y_title, scale=y_scale, axis=alt.Axis(values=ticks))
    x = alt.X(f"{formulation}:Q", title=_say(opt, wording.Q2_X_TITLE), scale=x_scale,
              axis=alt.Axis(format="d", values=formulation_ticks(frame[formulation]),
                            grid=False))
    grey, _ = mark("neutral")
    ink, _ = mark("bar_latest")
    light, _ = _TRIED_GREYS[theme()]
    blue = model_blue()
    each = _say(opt, wording.review_each(len(shown)))
    best_name = _say(opt, wording.BEST_SO_FAR_COLUMN)
    avg_name = _say(opt, wording.STORY_ROUND_AVERAGE)
    no_tip = alt.Tooltip(f"{formulation}:Q", title=_say(opt, wording.FORMULATION_CAP),
                         format="d")
    tips = [no_tip, alt.Tooltip("round:N", title=_say(opt, wording.ROUND_CAP)),
            alt.Tooltip(f"{score}:Q", title=y_title, format=".1f"), bo_tip()]
    layers = []
    if show_top:
        tdf = pd.DataFrame([{"y": ceiling, "x": _x_domain(opt)[1],
                             "text": _say(opt, wording.review_top_line(f"{ceiling:g}"))}])
        layers.append(alt.Chart(tdf).mark_rule(color=grey, strokeDash=[2, 3], strokeWidth=1,
                                               tooltip=False)
                      .encode(y=alt.Y("y:Q", scale=y_scale, title=y_title)))
        layers.append(alt.Chart(tdf).mark_text(color=muted(), fontSize=FONT_SMALL, align="right",
                                               baseline="bottom", dy=-3, dx=-2, tooltip=False)
                      .encode(x=alt.X("x:Q", scale=x_scale), y=alt.Y("y:Q", scale=y_scale,
                                                                       title=y_title),
                              text="text:N"))
    if avgs:
        adf = pd.DataFrame([{"x0": a - 0.4, "x1": b + 0.4, "avg": m, "what": avg_name,
                             "round": wording.round_tick(r), "text": one_decimal(m)}
                            for r, a, b, m, _n in avgs])
        layers.append(alt.Chart(adf).mark_rule(strokeWidth=4, color=light).encode(
            x=alt.X("x0:Q", scale=x_scale), x2="x1:Q",
            y=alt.Y("avg:Q", scale=y_scale, title=y_title),
            tooltip=[alt.Tooltip("round:N", title=_say(opt, wording.ROUND_CAP)),
                     alt.Tooltip("text:N", title=avg_name)]))
    # Best so far: the running highest, stepping at the formulation that
    # raised it (final-v3 chart).
    steps = frame[frame[best_col].notna()]
    if not steps.empty:
        bdf = pd.DataFrame({"x": steps[formulation].astype(float), "v": steps[best_col],
                            "text": [one_decimal(v) for v in steps[best_col]]})
        layers.append(alt.Chart(bdf).mark_line(strokeWidth=2.5, color=ink,
                                               interpolate="step-after").encode(
            x=alt.X("x:Q", scale=x_scale), y=alt.Y("v:Q", scale=y_scale, title=y_title),
            tooltip=[alt.Tooltip("x:Q", title=_say(opt, wording.FORMULATION_CAP), format="d"),
                     alt.Tooltip("text:N", title=best_name)]))
    dots = shown.assign(what=each)
    layers.append(alt.Chart(dots[~dots["best"]]).mark_point(size=40, strokeWidth=1.3, opacity=1,
                                                            color=grey)
                  .encode(x=x, y=y,
                          fill=alt.condition(alt.FieldEqualPredicate(field="full", equal=True),
                                             alt.value(grey), alt.value("transparent")),
                          tooltip=tips))
    top_dots = dots[dots["best"]]
    if not top_dots.empty:
        layers.append(alt.Chart(top_dots).mark_point(size=60, filled=True, opacity=1, color=blue,
                                                     stroke=_page(), strokeWidth=1)
                      .encode(x=x, y=y, tooltip=tips))
        x0 = float(top_dots[formulation].iloc[0])
        lo_x, hi_x = _x_domain(opt)
        right = (x0 - lo_x) / ((hi_x - lo_x) or 1.0) > 0.45
        ldf = pd.DataFrame([{"x": x0, "y": best_y, "text": _say(opt, wording.review_best_label(
            one_decimal(view.best), _names(opt, view.bests)))}])
        layers += _halo_text(ldf, alt.X("x:Q", scale=x_scale),
                             alt.Y("y:Q", scale=y_scale, title=y_title), "text:N", blue,
                             fontSize=FONT_LABEL, align="right" if right else "left",
                             dx=4 if right else -4, baseline="middle")
    layers.append(_key_layer([(each, grey, "circle"), (best_name, ink, "stroke"),
                              (avg_name, light, "stroke")]))
    rounds = _round_layer(opt, formulation, list(frame[formulation]), x_scale,
                          y=(alt.Y("y0:Q", scale=y_scale, title=y_title), bottom, rules_top))
    body = alt.layer(*layers)
    chart = alt.layer(rounds, body).resolve_axis(x="independent") if rounds else body
    return readable(chart.properties(height=plot_h))


# ---- Best so far vs targets (the strip that replaced the radar) ------ #

TARGET_STRIP_ROW_H = 44      # px per attribute row
_STRIP_ROOM_PX = 130         # px kept right of each track for its value text
_STRIP_GAP_PX = 8            # px between the names and a track's start
# Allowed, not tried (a track's own scale), per theme: the plain page tint
# with an outline, so it reads as empty room in both.
_TRACK_COLOURS = {"light": ("#f1f3f6", "#c3c8d0"), "dark": ("#161a22", "#4a515c")}


def target_strip_rows(opt, view):
    """One row per measurement with a value for the best so far, largest
    Share of score first: its Range, goal and target, one dot for the best
    so far (one per tied best where they differ, each named), whether each
    is off target and which way ('above' | 'below' | ''), and the value
    text ('4 (target 5, −1)')."""
    out = []
    names = _names(opt, view.bests)
    for obj in opt.measurements_by_importance():
        name, goal = obj['name'], obj.get('goal', 'max')
        lo, hi = float(obj.get('min_val', 0.0)), float(obj.get('max_val', 10.0))
        target = (float(obj['target']) if goal == 'target' and obj.get('target') is not None
                  else None)
        vals = [(k, float(opt._results_at(i)[name])) for k, i in enumerate(view.bests)
                if opt._results_at(i).get(name) is not None]
        if not vals:
            continue
        same = all(abs(v - vals[0][1]) <= 1e-9 for _k, v in vals)
        dots = []
        for k, v in (vals[:1] if same else vals):
            off = target is not None and abs(v - target) > 1e-9 * max(hi - lo, 1.0)
            dots.append({"k": k, "who": names[k], "value": v, "off": off,
                         "direction": ("" if not off else "above" if v > target else "below"),
                         "shape": "circle" if same else BEST_SHAPES[k % len(BEST_SHAPES)]})
        t_text = None if target is None else _measure_text(target)
        if same:
            v = vals[0][1]
            gap = None
            if target is not None and dots[0]["off"]:
                d = max(len(t.split(".", 1)[1]) if "." in t else 0
                        for t in (_measure_text(v), t_text))
                gap = wording.signed(v - target, d)
            text = wording.review_target_value(_measure_text(v), t_text, gap, goal)
        else:
            text = wording.review_target_value(
                ", ".join(f"{names[k]} {_measure_text(v)}" for k, v in vals), t_text, None, goal)
        out.append({"name": name, "goal": goal, "low": lo, "high": hi, "target": target,
                    "dots": dots, "text": text, "off": any(d["off"] for d in dots),
                    "named": not same})
    return out


def _axis_label(opt, rows):
    """The strip's axis title when every row shares one Range: 'Panel
    median (0–15)' for a project whose panel is on record, else 'Measured
    value (0–15)'; None when the Ranges differ (each row then says its
    own)."""
    ranges = {(r["low"], r["high"]) for r in rows}
    if len(ranges) != 1:
        return None
    lo, hi = ranges.pop()
    label = (wording.REVIEW_PANEL_MEDIAN if sample_projects.panel_size(opt)
             else wording.REVIEW_MEASURED)
    return wording.review_targets_axis(label, f"{lo:g}", f"{hi:g}")


def _room_scale(room, gap=_STRIP_GAP_PX):
    """An x scale whose 0 sits `gap` px into the plot and whose 1 sits
    `room` px before its right edge, at any width (the space after is for
    end text)."""
    unit = f"max(40, width - {room + gap})"
    return alt.Scale(domain=[alt.ExprRef(f"-{gap} / {unit}"),
                             alt.ExprRef(f"(width - {gap}) / {unit}")], nice=False, zero=False)


def targets_chart(opt, view):
    """Best so far vs targets (final-v3; replaces the radar): one row per
    measured attribute on its own Range — a light track, the target a
    tall ink tick, the best so far's dot blue on target; off target, its
    dot, the gap to the target and the value at the row's end ('4 (target
    5, −1)') orange above the target and purple below it (owner, v3 final
    cleanup). A max or min goal has no target
    tick: its dot grey, its text 'higher is better'. Tied bests that
    differ: one dot each, its own marker, named in the text. None with no
    row."""
    rows = target_strip_rows(opt, view)
    if not rows:
        return None
    names = [r["name"] for r in rows]
    ink, _ = mark("bar_latest")
    grey, _ = mark("neutral")
    blue = model_blue()
    fill, edge = _TRACK_COLOURS[theme()]
    axis_title = _axis_label(opt, rows)
    xs = _room_scale(_STRIP_ROOM_PX)
    y = alt.Y("name:N", sort=names, title=None,
              axis=alt.Axis(ticks=False, domain=False, labelLimit=220, labelPadding=6,
                            labelFontSize=FONT_LABEL + 1, labelColor=ink))

    def t(r, v):
        span = r["high"] - r["low"]
        return 0.5 if span <= 0 else min(1.0, max(0.0, (float(v) - r["low"]) / span))
    track = pd.DataFrame([{"name": r["name"], "x0": 0.0, "x1": 1.0,
                           "tip": f"{_measure_text(r['low'])}–{_measure_text(r['high'])}"}
                          for r in rows])
    if axis_title:
        lo, hi = rows[0]["low"], rows[0]["high"]
        span = (hi - lo) or 1.0
        step = 5 if span <= 30 else 10 if span <= 60 else 20
        vals = sorted({lo, hi, *_frange(math.ceil(lo / step) * step, hi, step)})
        axis = alt.Axis(values=[(v - lo) / span for v in vals], title=_say(opt, axis_title),
                        labelExpr=" : ".join(f"abs(datum.value - {(v - lo) / span}) < 1e-6 ? "
                                             f"'{v:g}'" for v in vals) + " : ''",
                        grid=False, domain=False, tickColor=muted(), labelColor=muted(),
                        titleColor=muted())
    else:
        axis = None

    def xa(field):
        """Every layer's x: one scale, one axis (a layer that said none
        would take the shared axis away)."""
        return alt.X(f"{field}:Q", scale=xs, axis=axis)
    x = xa("x0")
    layers = [alt.Chart(track).mark_bar(height=16, color=fill, stroke=edge, strokeWidth=1)
              .encode(x=x, x2="x1:Q", y=y, tooltip=alt.Tooltip("tip:N", title=" "))]
    if axis is None:
        ends = pd.DataFrame([{"name": r["name"], "x": xx, "text": _measure_text(v)}
                             for r in rows for xx, v in ((0.0, r["low"]), (1.0, r["high"]))])
        for xx, align in ((0.0, "left"), (1.0, "right")):
            layers.append(alt.Chart(ends[ends["x"] == xx]).mark_text(
                fontSize=FONT_SMALL, color=muted(), align=align, baseline="top", dy=10,
                tooltip=False).encode(x=xa("x"), y=y, text="text:N"))
    targets = pd.DataFrame([{"name": r["name"], "x": t(r, r["target"]),
                             "tip": _measure_text(r["target"])}
                            for r in rows if r["target"] is not None])
    if not targets.empty:
        layers.append(alt.Chart(targets).mark_tick(color=ink, thickness=3, size=32)
                      .encode(x=xa("x"), y=y,
                              tooltip=alt.Tooltip("tip:N", title=wording.REVIEW_KEY_TARGET)))
    dots = []
    for r in rows:
        for d in r["dots"]:
            colour = ((direction_colour(d["direction"]) if d["off"] else blue)
                      if r["target"] is not None else grey)
            dots.append({"name": r["name"], "x": t(r, d["value"]),
                         "tx": t(r, r["target"]) if r["target"] is not None else None,
                         "off": d["off"], "colour": colour, "shape": d["shape"],
                         "who": d["who"] if r["named"] else "", "value": _measure_text(d["value"]),
                         "label": f"{d['who']}" if r["named"] else ""})
    ddf = pd.DataFrame(dots)
    gaps = ddf[ddf["off"]]
    for colour, part in (gaps.groupby("colour") if not gaps.empty else ()):
        layers.append(alt.Chart(part).mark_rule(color=colour, strokeWidth=5, tooltip=False)
                      .encode(x=xa("x"), x2="tx:Q", y=y))
    tip = [alt.Tooltip("value:N", title=_say(opt, wording.BEST_SO_FAR_COLUMN))]
    if (ddf["who"] != "").any():
        tip = [alt.Tooltip("who:N", title=_say(opt, wording.FORMULATION_CAP))] + tip
    for (shape, colour), part in ddf.groupby(["shape", "colour"]):
        layers.append(alt.Chart(part).mark_point(shape=shape, filled=True, size=150, opacity=1,
                                                 color=colour, stroke=_page(), strokeWidth=1.5)
                      .encode(x=xa("x"), y=y, tooltip=tip))
    named = ddf[ddf["label"] != ""]
    if not named.empty:
        layers += _halo_text(named, xa("x"), y, "label:N", ink,
                             fontSize=FONT_SMALL, dy=-14, baseline="bottom")
    def text_colour(r):
        ways = [d["direction"] for d in r["dots"] if d["direction"]]
        return direction_colour(ways[0]) if ways else ink
    texts = pd.DataFrame([{"name": r["name"], "x": 1.0, "text": _say(opt, r["text"]),
                           "colour": text_colour(r)} for r in rows])
    for colour, part in texts.groupby("colour"):
        layers.append(alt.Chart(part).mark_text(fontSize=FONT_LABEL, align="left", dx=12,
                                                baseline="middle", color=colour, tooltip=False)
                      .encode(x=xa("x"), y=y, text="text:N"))
    # The key: drawn by a layer with no data (its explicit domain lists every
    # entry), so the marks above keep their own colours and shapes.
    entries = []
    if not targets.empty:
        # Only the entries drawn (owner, v3 final cleanup: keep it short).
        ways = {d["direction"] for r in rows if r["target"] is not None for d in r["dots"]}
        entries.append((wording.REVIEW_KEY_TARGET, ink, "stroke"))
        if "" in ways:
            entries.append((wording.REVIEW_KEY_ON, blue, "circle"))
        if "above" in ways:
            entries.append((wording.ABOVE_TARGET, direction_colour("above"), "circle"))
        if "below" in ways:
            entries.append((wording.BELOW_TARGET, direction_colour("below"), "circle"))
    if any(r["target"] is None for r in rows):
        entries.append((wording.REVIEW_KEY_NO_TARGET, grey, "circle"))
    layers.append(_key_layer(entries, width=2))
    height = TARGET_STRIP_ROW_H * len(rows) + (0 if axis is not None else 14)
    return readable(alt.layer(*layers).properties(height=height))


def _key_layer(entries, width=3, size=110):
    """A legend alone: [(label, fill, shape[, stroke])] — a "stroke" shape
    is a line in its colour, any other shape filled (outlined in `stroke`
    when given) — drawn from an empty layer with explicit domains: no mark
    on the plot and no position, so it leaves the chart's axes and its
    marks' own colours alone."""
    labels = [e[0] for e in entries]
    fills = ["transparent" if e[2] == "stroke" else e[1] for e in entries]
    strokes = [e[3] if len(e) > 3 and e[3] else e[1] for e in entries]
    legend = alt.Legend(title=None, orient="bottom", direction="horizontal", labelLimit=0,
                        symbolSize=size, symbolStrokeWidth=width, columnPadding=18, offset=12)

    def scale(rng):
        return alt.Scale(domain=labels, range=rng)
    return (alt.Chart(pd.DataFrame({"what": pd.Series([], dtype=str)}))
            .mark_point(filled=False, opacity=1)
            .encode(fill=alt.Fill("what:N", legend=legend, scale=scale(fills)),
                    stroke=alt.Stroke("what:N", legend=legend, scale=scale(strokes)),
                    shape=alt.Shape("what:N", legend=legend, scale=scale([e[2] for e in entries]))))


# ---- Amounts in the top formulations (replaced What changed) --------- #

AMOUNT_ROW_H = 62            # px per ingredient or setting
_AMOUNT_ROOM_PX = 150        # px kept right of each track for its limit tag
_TAG_FROM = 0.012            # bests closer than this share of a track are offset
_OFFSET_PX = 8               # px between two markers at one amount
# Tried (grey), the top formulations' range (blue) and the best so far's
# markers, per theme: blue only for the top and the best (owner's colour
# language); the allowed track is _TRACK_COLOURS'.
_AMOUNT_COLOURS = {"light": {"tried": "#cfd3d9", "top": "#a9c4f2", "best": "#123f91"},
                   "dark": {"tried": "#3a404a", "top": "#2b4f9e", "best": "#dbe7fb"}}


def _amount_value(recipe, var):
    v = recipe.get(var['name'], var.get('_absent_value', 0.0))
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def amounts_rows(opt, view):
    """One row per varying ingredient or setting with a numeric amount, in
    Set up's order: its limits, the range tried and the top formulations'
    range (each as shares of the limits, clipped to them), each tied best's
    amount, and the limit tag ('upper' | 'lower' | None: a best so far
    within insights.BINDING of the limit, the app's own rule; never on the
    Fill to total row) with the top formulations at that limit."""
    recipes = {i: insights.start_recipe(opt, i) for i in set(view.tried) | set(view.top)
               | set(view.bests)}
    fill = opt.balance_row_name()
    best_names = _names(opt, view.bests)
    out = []
    for var in opt.varying_variables():
        if var.get('type', 'continuous') != 'continuous':
            continue
        lo, hi = (float(b) for b in var['bounds'])
        span = hi - lo
        if span <= 0:
            continue

        def share(v):
            return min(1.0, max(0.0, (v - lo) / span))

        def values(idx):
            return [v for v in (_amount_value(recipes[i], var) for i in idx) if v is not None]
        tried, top = values(view.tried), values(view.top)
        bests = [(k, _amount_value(recipes[i], var)) for k, i in enumerate(view.bests)]
        bests = [(k, v) for k, v in bests if v is not None]
        if not tried:
            continue

        def at(v, upper):
            return v >= hi - insights.BINDING * span if upper else v <= lo + insights.BINDING * span
        tag, at_names = None, []
        if var['name'] != fill:
            for upper, kind in ((True, "upper"), (False, "lower")):
                if any(at(v, upper) for _k, v in bests):
                    tag = kind
                    at_names = [n for n, i in zip(_names(opt, view.top), view.top)
                                if _amount_value(recipes[i], var) is not None
                                and at(_amount_value(recipes[i], var), upper)]
                    break
        unit = opt.unit_of(var['name'])
        out.append({"name": var['name'], "low": lo, "high": hi, "unit": unit,
                    "tried": (share(min(tried)), share(max(tried))),
                    "tried_text": (min(tried), max(tried)),
                    "top": (share(min(top)), share(max(top))) if top else None,
                    "top_text": (min(top), max(top)) if top else None,
                    "bests": [{"k": k, "who": best_names[k], "x": share(v), "value": v}
                              for k, v in bests],
                    "tag": tag, "at": at_names})
    return out


def best_offsets(xs, near=_TAG_FROM, px=_OFFSET_PX):
    """Vertical offsets (px) for markers at shares `xs` on one row: markers
    closer than `near` are spread around the row's centre, so no marker
    hides another; a marker alone stays on it."""
    order = sorted(range(len(xs)), key=lambda k: xs[k])
    out = [0.0] * len(xs)
    group = []
    for k in order + [None]:
        if k is not None and group and abs(xs[k] - xs[group[-1]]) < near:
            group.append(k)
            continue
        m = len(group)
        for j, g in enumerate(group):
            out[g] = (j - (m - 1) / 2) * px
        group = [k] if k is not None else []
    return out


def _total_text(opt, lo, hi):
    """'97.7–98.7 g', '100 g': the total in one decimal (two where one
    would make its ends meet), in the amounts' unit."""
    unit = opt.one_amount_unit() or ""
    d = 1
    while d < 3 and hi is not None and lo is not None and f"{lo:.{d}f}" == f"{hi:.{d}f}" \
            and abs(hi - lo) > 1e-9:
        d += 1

    def fmt(v):
        return f"{v:g}" if float(v).is_integer() else f"{v:.{d}f}"
    if lo is None or hi is None or abs(hi - lo) <= 1e-9:
        return join_unit(fmt(hi if lo is None else lo), unit)
    return join_unit(f"{fmt(lo)}–{fmt(hi)}", unit)


def amounts_linked_text(opt):
    """'The 7 add to 97.7–98.7 g (the allowed total), so amounts are
    linked.' — from the project's own limit on the total of every varying
    ingredient (an Ingredient total or the Formulation total); None
    without one."""
    names = {v['name'] for v in opt.varying_variables() if insights._is_ingredient(v)
             and v.get('type', 'continuous') == 'continuous'}
    if len(names) < 2:
        return None
    for qc in getattr(opt, 'quantity_constraints', []) or []:
        if not names <= set(qc.get('ingredients') or ()):
            continue
        if qc.get('source') == 'formulation_total' and opt.has_formulation_total():
            lo = hi = float(opt.formulation_total)
        elif qc.get('exactly') is not None:
            lo = hi = float(qc['exactly'])
        elif qc.get('min') is not None and qc.get('max') is not None:
            lo, hi = float(qc['min']), float(qc['max'])
        else:
            continue
        n = len(qc.get('ingredients') or ())
        return _say(opt, wording.review_amounts_linked(n, _total_text(opt, lo, hi)))
    return None


def amounts_chart(opt, view):
    """Amounts in the top N formulations (final-v3; replaces What changed):
    one track per ingredient or setting between its limits (written under
    it) — outlined where allowed but not tried, grey over the range tried,
    blue over the range the top formulations use — each tied best its own
    marker (spread a little when two share an amount, so none hides
    another), and 'at upper limit' / 'at lower limit' in muted text where
    the best so far sits at a limit (Definitions says what it means; its
    hover names the top formulations there). The key counts what is drawn
    and names every top formulation inline, 'Top 3: F16, F17, F10' — with
    its rule on a tie at 3rd, 'Top 6 (3 best, with ties): …'. No "(?)" is
    drawn inside the chart (owner, v3 final cleanup). None with no row."""
    rows = amounts_rows(opt, view)
    if not rows:
        return None
    names = [r["name"] for r in rows]
    c = _AMOUNT_COLOURS[theme()]
    fill, edge = _TRACK_COLOURS[theme()]
    ink, _ = mark("bar_latest")
    page = _page()
    room = _AMOUNT_ROOM_PX if any(r["tag"] for r in rows) else 24
    xs = _room_scale(room, gap=4)
    y = alt.Y("name:N", sort=names, title=None,
              axis=alt.Axis(ticks=False, domain=False, labelLimit=220, labelPadding=10,
                            labelFontSize=FONT_LABEL + 2, labelColor=ink))
    X = lambda f: alt.X(f"{f}:Q", scale=xs, axis=None)    # noqa: E731

    def amt(r, v):
        return join_unit(_num(v), r["unit"])
    top_names = _names(opt, view.top)
    top_label = _say(opt, wording.review_top_key(top_names, f"{TOP_COUNT}"))
    tried_label = _say(opt, wording.review_tried(len(view.tried)))
    allowed = pd.DataFrame([{"name": r["name"], "x0": 0.0, "x1": 1.0,
                             "tip": f"{_limit_amount(opt, r['name'], r['low'])}–"
                                    f"{_limit_amount(opt, r['name'], r['high'])}"}
                            for r in rows])
    layers = [alt.Chart(allowed).mark_bar(height=24, color=fill, stroke=edge, strokeWidth=1)
              .encode(x=X("x0"), x2="x1:Q", y=y,
                      tooltip=alt.Tooltip("tip:N", title=_say(opt, wording.REVIEW_ALLOWED)))]
    tried = pd.DataFrame([{"name": r["name"], "x0": r["tried"][0],
                           "x1": max(r["tried"][1], r["tried"][0] + 0.004),
                           "tip": f"{amt(r, r['tried_text'][0])}–{amt(r, r['tried_text'][1])}"}
                          for r in rows])
    layers.append(alt.Chart(tried).mark_bar(height=24, color=c["tried"])
                  .encode(x=X("x0"), x2="x1:Q", y=y,
                          tooltip=alt.Tooltip("tip:N", title=tried_label)))
    top = pd.DataFrame([{"name": r["name"], "x0": max(0.0, r["top"][0] - 0.003),
                         "x1": min(1.0, max(r["top"][1], r["top"][0] + 0.006)),
                         "tip": f"{amt(r, r['top_text'][0])}–{amt(r, r['top_text'][1])}"}
                        for r in rows if r["top"] is not None])
    if not top.empty:
        layers.append(alt.Chart(top).mark_bar(height=24, color=c["top"])
                      .encode(x=X("x0"), x2="x1:Q", y=y,
                              tooltip=alt.Tooltip("tip:N", title=top_label)))
    ends = pd.DataFrame([{"name": r["name"], "x": xx,
                          "text": (_limit_amount(opt, r["name"], v) if xx == 1.0
                                   else _num(v) if not float(v).is_integer() else f"{v:g}")}
                         for r in rows for xx, v in ((0.0, r["low"]), (1.0, r["high"]))])
    for xx, align in ((0.0, "left"), (1.0, "right")):
        layers.append(alt.Chart(ends[ends["x"] == xx]).mark_text(
            fontSize=FONT_SMALL, color=muted(), align=align, baseline="top", dy=17,
            tooltip=False).encode(x=X("x"), y=y, text="text:N"))
    marks = []
    for r in rows:
        offs = best_offsets([b["x"] for b in r["bests"]])
        for b, off in zip(r["bests"], offs):
            marks.append({"name": r["name"], "x": b["x"], "off": off,
                          "shape": BEST_SHAPES[b["k"] % len(BEST_SHAPES)],
                          "who": b["who"], "tip": amt(r, b["value"])})
    mdf = pd.DataFrame(marks)
    if not mdf.empty:
        for (shape, off), part in mdf.groupby(["shape", "off"]):
            layers.append(alt.Chart(part).mark_point(
                shape=shape, filled=True, size=140, opacity=1, color=c["best"], stroke=page,
                strokeWidth=1.5, yOffset=float(off))
                .encode(x=X("x"), y=y, tooltip=[alt.Tooltip("who:N", title=" "),
                                                 alt.Tooltip("tip:N", title=" ")]))
    tags = pd.DataFrame([{"name": r["name"], "x": 1.0,
                          "text": (wording.REVIEW_AT_UPPER if r['tag'] == 'upper'
                                   else wording.REVIEW_AT_LOWER),
                          "help": _say(opt, wording.review_limit_help(r["at"], r["tag"] == "upper"))}
                         for r in rows if r["tag"]])
    if not tags.empty:
        layers.append(alt.Chart(tags).mark_text(fontSize=FONT_LABEL, color=muted(), align="left",
                                                dx=10, baseline="middle")
                      .encode(x=X("x"), y=y, text="text:N", tooltip="help:N"))
    entries = [(_say(opt, wording.REVIEW_ALLOWED), fill, "square", edge, None),
               (tried_label, c["tried"], "square", c["tried"], None)]
    if not top.empty:
        entries.append((top_label, c["top"], "square", c["top"], None))
    best_names = _names(opt, view.bests)
    for k, who in enumerate(best_names):
        entries.append((_say(opt, wording.review_best_key(who, tied=k > 0)), c["best"],
                        BEST_SHAPES[k % len(BEST_SHAPES)], page, None))
    height = AMOUNT_ROW_H * len(rows)
    layers += _mark_key(entries, height + _KEY_OFFSET_PX)
    return readable(alt.layer(*layers).properties(height=height))


# The amounts chart's key, drawn as marks under the tracks (a Vega-Lite
# legend cannot carry a "(?)" tooltip): one line, or two when the plot is
# narrower than one line's width.
_KEY_OFFSET_PX = 26          # px from the plot's bottom to the key's first line
_KEY_LINE_PX = 22            # px between the key's two lines
_KEY_SYMBOL_PX = 16          # px from an entry's start to its text
_KEY_GAP_PX = 20             # px between two entries


def _text_px(text, size=FONT_LABEL):
    """A generous width of `text` in px at `size` (the app's sans): narrow
    marks, wide letters, capitals and digits, the rest."""
    em = 0.0
    for ch in text:
        em += (0.3 if ch in " .,:;'|!()[]ilIjtf·" else 0.85 if ch in "mwMW—…"
               else 0.62 if ch.isupper() or ch.isdigit() else 0.52)
    return em * size


def _mark_key(entries, y_px, size=150):
    """Key layers for [(label, fill, shape, stroke, help)]: a symbol and
    its label per entry from the plot's left edge, on one line when the
    plot is wide enough, else on two (split where the longer line is
    shortest). An entry with `help` carries it as its label's tooltip."""
    spans = [_KEY_SYMBOL_PX + _text_px(e[0]) for e in entries]

    def starts(part):
        out, x = [], 0.0
        for w in part:
            out.append(x)
            x += w + _KEY_GAP_PX
        return out, x - _KEY_GAP_PX
    one, total = starts(spans)
    split = min(range(1, len(spans)), default=len(spans),
                key=lambda k: max(starts(spans[:k])[1], starts(spans[k:])[1]))
    two = starts(spans[:split])[0] + starts(spans[split:])[0]
    layers = []
    for k, (label, fill, shape, stroke, help_) in enumerate(entries):
        line2 = k >= split
        x = alt.ExprRef(f"width >= {total:.0f} ? {one[k]:.1f} : {two[k]:.1f}")
        y = alt.ExprRef(f"width >= {total:.0f} ? {y_px} : {y_px + (_KEY_LINE_PX if line2 else 0)}")
        tx = alt.ExprRef(f"width >= {total:.0f} ? {one[k] + _KEY_SYMBOL_PX:.1f} : "
                         f"{two[k] + _KEY_SYMBOL_PX:.1f}")
        dot = pd.DataFrame([{"k": k}])
        layers.append(alt.Chart(dot).mark_point(
            x=alt.ExprRef(f"({x.expr}) + 6"), y=y, shape=shape, filled=True, fill=fill,
            stroke=stroke or fill, strokeWidth=1, size=size, opacity=1, tooltip=False))
        text = dict(x=tx, y=y, text=label, align="left", baseline="middle",
                    fontSize=FONT_LABEL, color=muted())
        layers.append(alt.Chart(dot).mark_text(
            **text, **({"tooltip": help_} if help_ else {"tooltip": False})))
    return layers


                        # axis titles inside it)


# ------------------------------------------------------------------ #
#  Insights for a first-time user (spec 2026-10-01 rev. 2), W1: the three
#  status lines (review R-B1, R-I5, R-I9, R-I10, M1, M7, M10) and the one
#  tertiary link (R-I3, R-I4, R-I14).
# ------------------------------------------------------------------ #


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


# ------------------------------------------------------------------ #
#  Wave C: More charts
# ------------------------------------------------------------------ #

# The two multiselects (a project's own rows and measurements) and the
# triangle's cached grid, keyed by setup_key, the three names and the start.
PAIR_KEY = "insights_pair"
THREE_KEY = "insights_three"
MIXTURE_KEY = "_insights_mixture"


# ------------------------------------------------------------------ #
#  Insights for a first-time user (spec 2026-10-01 rev. 2), W3: the
#  "Explored and unexplored" fold (§8, review R-I14 to R-I16). T0 stubs with
#  their final signatures; W3 fills them in.
# ------------------------------------------------------------------ #

def _round_label(r):
    return wording.IMPORTED_NOTE if r is None else str(r)


_MAP_W, _MAP_H = 730, 300      # px: the map's plot, about (the column less its axis)


# ------------------------------------------------------------------ #
#  Charts proto (2026-10-01): What you've tried (parallel coordinates),
#  How the app chooses (the classic figure along one row through Best so
#  far), the triangle, then the map — all in Explored and unexplored.
# ------------------------------------------------------------------ #

# The model's 95 % band (a region: not held to 3:1) and the
# expected gain's line and area. Blue means "the model's uncertainty or
# gain" on every chart here (the triangle's Uncertainty and Expected
# improvement layers use blues()). Light #2f6db5 is 5.2:1 on white; dark
# #7fb0e6 is 8.3:1 on #0e1117.
_MODEL_BLUE = {"light": "#2f6db5", "dark": "#7fb0e6"}


def model_blue(which=None):
    return _MODEL_BLUE[which or theme()]


# Review four polish (owner: the score colours "all look very similar"):
# Q4 coloured by role. Every other tested line thin and light grey (context,
# so an untested stretch reads as empty space); the next best few mid-grey;
# Best so far thick ink; this Round's suggestions orange. (light grey, mid-
# grey) per theme: light #cfd3d9 / #7d838e on white; dark #444a54 / #9aa0aa
# on #0e1117 — the mid-grey and the ink reach 3:1, the context lines need not.
_TRIED_GREYS = {"light": ("#cfd3d9", "#7d838e"), "dark": ("#444a54", "#9aa0aa")}


# ---- How the app chooses ------------------------------------------ #

CHOOSES_LINE, CHOOSES_ONE = "line", "one"


def _how(opt):
    """Definitions (final-v3 §4): one plain line per term the Review and
    Make a round use, from this project's own numbers (definitions)."""
    with st.container(key=HOW_PLACE), \
            st.expander(_say(opt, wording.FOLD_METHOD), key=HOW_KEY, on_change="rerun") as fold:
        # Keyed and drawn only while open (T56 I5; binding I15).
        if not fold.open:
            return
        for line in definitions(opt):
            st.markdown(f"- {line}")


# ------------------------------------------------------------------ #
#  Review cards (prototype, 2026-10-01; owner: "a clearer, less wordy way
#  to present the review"): three bordered cards side by side, each a
#  small label, one big status word, one number line, any warning on a
#  line of its own, and a 60 px sparkline; the full sentence and the exact
#  rule in the card's "?" (status_help). The hero chart under them is the
#  BO figure along Best so far -> Suggestion 1.
# ------------------------------------------------------------------ #


def _check_round(opt):
    return opt.pending_batch_no if opt.pending_batch else opt.next_batch_no()


HOW_PLACE = "insights_how_place"


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


def _short(opt, text):
    """'F18' as the project says it: 'T18' in a study of trials."""
    letter = _say(opt, wording.FORMULATION_CAP)[:1] or "F"
    return re.sub(r"\bF(\d+)\b", rf"{letter}\1", _say(opt, text))


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


def no_forecast(opt, c):
    """Sceptic review: whether the 95 % range of check (or prediction) `c`
    covers its result's whole Range — its low within NO_FORECAST_EDGE (1 %)
    of the Range's width of the Range's low, and its high as close to the
    Range's high: a range of the whole Range forecasts nothing (Creaminess
    0.0–15.0 on 0–15). False for a name with no Range."""
    return bool(getattr(c, "no_forecast", False)) or insights.covers_range(
        opt, c.name, c.low, c.high)


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
# Zoom-proof panel (owner-approved 2026-10-14, spec §10): the two columns
# share the width as PANEL_GROW, each from its least width; below both
# (zoomed in) the profile wraps under the score chart and What changed, so
# the score chart is never narrower than at 100 % (534 px at 1440 × 900).
PANEL_ROW = "insights_panel_row"
PANEL_LEFT = "insights_panel_score"     # Best so far over What changed
PANEL_RIGHT = "insights_panel_profile"  # the sensory profile
PANEL_MIN_W = (440, 340)                # px: each column's least width
PANEL_GROW = (52, 48)                   # how the width past those is shared
SPIDER_ANSWER_LINES_EM = 3.1            # the profile's answer line: two lines' room (0.94rem)

# The panel row's selectors rely on Streamlit 1.55's DOM (requirements.txt
# pins it): a keyed st.container is a div with the class "st-key-" + key, and
# each child of a horizontal container sits in its own flex-item wrapper,
# hence ":has(> .st-key-…)". Check them when Streamlit is upgraded
# (tests/test_review_round_panel; the panel's screenshots). No comment goes
# inside the <style>: st.html would read "<key>" there as a tag.
REPLAY_STYLE = """<style>
.st-key-insights_replay_panel_now, .st-key-insights_replay_panel_past { padding: 0.6rem 0.9rem !important; }
.st-key-insights_replay_panel_past { background: rgba(120, 128, 140, 0.09);
  border: 1px solid rgba(90, 98, 110, 0.55) !important;
  box-shadow: inset 0 0 0 1px rgba(90, 98, 110, 0.55); }
.st-key-insights_replay_bar button { min-height: 2rem; padding: 0.15rem 0.7rem; }
.st-key-insights_replay_bar p { margin: 0; }
[class*="st-key-insights_replay_panel"] [data-testid="stElementToolbar"] { display: none; }
[class*="st-key-insights_replay_panel"] [data-testid="stMarkdownContainer"] p { font-size: 0.94rem; }
.st-key-insights_changes p { margin-bottom: 0.2rem; }
.st-key-insights_spider_answer { min-height: %.1fem; }
.st-key-insights_panel_row { flex-wrap: wrap !important; align-items: flex-start !important; }
.st-key-insights_panel_row > :has(> .st-key-insights_panel_score) {
  flex: %d 1 %dpx !important; min-width: min(%dpx, 100%%) !important; }
.st-key-insights_panel_row > :has(> .st-key-insights_panel_profile) {
  flex: %d 1 %dpx !important; min-width: min(%dpx, 100%%) !important; }
</style>""" % (SPIDER_ANSWER_LINES_EM, PANEL_GROW[0], PANEL_MIN_W[0], PANEL_MIN_W[0],
                PANEL_GROW[1], PANEL_MIN_W[1], PANEL_MIN_W[1])


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
                    help=_say(opt, wording.REVIEW_REPLAY_HELP))
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


# ------------------------------------------------------------------ #
#  Review story (prototype, 2026-10-02): the Review read in the order a
#  food scientist asks — Are we improving? Can we trust the predictions?
#  What drives each result? What have we tried? Builders only here (no
#  Streamlit); the sections above draw them. Every number is read from the
#  project: the start is the first scored Round's best, Best so far is
#  best_index's, the forecasts are the Round-ahead checks.
# ------------------------------------------------------------------ #


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


def _round_layer(opt, frame_x, numbers, scale, y=None):
    """Light rules between Rounds and 'R3' labels along the top, on the
    Formulation field (one scale), or None with under two Rounds. `y`:
    (the y encoding, from, to) — the rules span only that, so they stop
    short of labels above it; the full height without it."""
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
    if y is None:
        return (alt.Chart(pd.DataFrame({frame_x: rules}))
                .mark_rule(color=light, strokeWidth=1, tooltip=False).encode(x=top))
    enc, lo, hi = y
    rows = pd.DataFrame({frame_x: rules, "y0": float(lo), "y1": float(hi)})
    return (alt.Chart(rows).mark_rule(color=light, strokeWidth=1, tooltip=False)
            .encode(x=top, y=enc, y2="y1:Q"))


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


# --- Sensory profile against target (review, 2026-10-08) ---
# From SPIDER_MIN measurements, Start vs best's dot chart becomes a spider:
# one spoke per measurement (most important first, clockwise from the top),
# each spoke its own Range from the centre (low) to the rim (high). The
# targets in bold ink, Best so far at the chosen Round filled blue, the Start
# (story_start, as What changed) dashed grey, the earlier Rounds' bests
# faint. Measured values only: no model, no forecast, nothing drawn between
# Rounds.


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


# --- What changed in the formulation (review, 2026-10-08) ---
# Start (story_start: the first scored Round's best, as Start vs best) against
# Best so far, one row per varied ingredient or process setting. Model-free:
# the recorded make-ups only, at the Formulation total (insights.start_recipe).


def _limit_amount(opt, name, value):
    """A row's limit as binding_texts writes it: '42 g', '2.6 g', '180 °C'."""
    unit = opt.unit_of(name)
    if insights.is_setting(opt, name):
        return fmt_setting(value, unit)
    return join_unit(f"{value:.0f}" if value == round(value) else _num(value), unit)


# --- What changed as a picture (owner-approved 2026-10-14, spec §10) ---
# The table read slowly. One row per changed ingredient or setting on its
# own Lowest–Highest bar: a grey dot at Start (Round 1's best), a blue dot
# at Best so far at the Round on view (the profile's colours beside it), a
# line between, the change at the right ('−6.3 g', ', at Highest' when Best
# so far sits there). Largest move first, as a share of the bar; unchanged
# rows on one line under it. Exact amounts in each row's tooltip.


def _frange(a, b, step):
    """a, a + step, … up to b (inclusive, to a millionth)."""
    out, v = [], a
    while v <= b + 1e-6:
        out.append(v)
        v += step
    return out


