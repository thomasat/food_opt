"""The Insights tab. Builders here return Altair charts from computed facts
and never call Streamlit; render() at the bottom is the only Streamlit code."""
import contextlib
import itertools
import math

import altair as alt
import pandas as pd
import streamlit as st

import bo_views
import insights
import insights_cache
import insights_jobs
import wording
from food_bo import join_unit, number_list
from ui_helpers import (TAB_BATCH, TAB_SETUP, amount_format, confirmation_open, go_to_tab,
                        open_rows, readiness, table_height)

_BAND = 0.25
_GREY = 0.12
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


def _target_layer():
    # Dashed, thin and grey: never mistaken for a likely-range bar.
    return (alt.Chart().mark_rule(strokeDash=[4, 3], strokeWidth=1, color="gray", opacity=0.6,
                                  tooltip=False)
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
        bars = base.mark_rule(strokeWidth=2).encode(
            x=x, y=alt.Y(f"{low}:Q", scale=scale, title=measured), y2=f"{high}:Q",
            opacity=alt.condition(latest_first, alt.value(0.6), alt.value(0.25)), tooltip=tip)
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
    start = (alt.Chart().mark_rule(strokeDash=[2, 2], tooltip=False).encode(x="x:Q")
             .transform_filter(_only("start")))
    layers = [band, line, start]
    if any(r["mark"] == "grey" for r in rows):
        layers.insert(0, alt.Chart().mark_rect(opacity=_GREY, color="gray", tooltip=False)
                      .encode(x="x:Q", x2="x2:Q").transform_filter(_only("grey")))
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
        bar.progress(0.0, text=_say(opt, wording.insights_progress(0, len(_checked(opt)))))
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


def _stepper(opt, bar):
    """check_rounds' `checking`: moves the one progress bar on as each Round
    is checked (0.8.0 review fixes §6, §9 I11), with the count the workers'
    line shows (Plan A review C I-1): "3 of 9 Rounds checked…"."""
    order = _checked(opt)

    @contextlib.contextmanager
    def checking(r, last):
        done = order.index(r) if r in order else 0
        bar.progress(min(1.0, (done + 1) / (len(order) + 1)),
                     text=_say(opt, wording.insights_progress(done, len(order))))
        yield
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


def memo(opt):
    """This project's fitted models, read from beside the project (with
    its Round checks and predicted best) the first time it is asked for."""
    hit = st.session_state.get(MEMO_KEY)
    if hit is None or hit[0] != opt.project_name:
        saved = insights_cache.load(opt.project_name)
        st.session_state[MEMO_KEY] = (opt.project_name, saved.memo)
        rounds = st.session_state.setdefault(ROUNDS_KEY, {})
        for key, checks in saved.rounds.items():
            rounds.setdefault(key, checks)
        if saved.best is not None and BEST_KEY not in st.session_state:
            st.session_state[BEST_KEY] = saved.best
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
    its fitted models, its Round checks and its predicted best, pruned to
    what this data uses. Once per state; a failed write only means the next
    open fits again."""
    mem = memo(opt)
    specs = _plan(opt)
    keys = set(insights.round_keys(opt).values())
    rounds = {k: v for k, v in st.session_state.get(ROUNDS_KEY, {}).items() if k in keys}
    best = st.session_state.get(BEST_KEY)
    best = best if best and best[0] == insights.setup_key(opt) else None
    kept = {s.key: mem[s.key] for s in specs if s.key in mem}
    mark = (insights.data_key(opt), len(kept), tuple(sorted(rounds)), best is not None)
    if st.session_state.get(SAVED_KEY) == mark:
        return
    try:
        insights_cache.save(opt.project_name, insights_cache.Saved(kept, rounds, best))
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
    I-1). Only with workers; in a script run of this session."""
    if not insights_jobs.workers_on():
        return
    if _in_memo(opt, "column"):
        warm_best(opt)
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
            st.rerun()
        return
    _catch_up(opt)
    now = _seen(opt, stages)
    st.session_state[SEEN_KEY] = now
    if drawn is not None and len(drawn) == len(now) and drawn != now:
        # A part landed while the page was being drawn (drawing() marked
        # what it was drawn from), or the workers failed and it is now
        # fitted here: one more run shows it instead of keeping the older
        # state until the next click.
        st.rerun()
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
    return not _in_flight(opt) or _seen(opt, stages) != st.session_state.get(SEEN_KEY)


@st.fragment(run_every=POLL_SECS)
def _poller(opt, stages):
    if _poll_step(opt, stages):
        st.rerun()


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


def render(opt):
    """The tab (spec 2026-09-30 §2): the header, three answers — "Can you
    trust the predictions?", "Where do you stand?", "What to do next" — the
    foot button, then three folds. At the gate (fewer than MIN_ROWS
    scored), after a failed fit and with nothing scored, the header (or
    one line), Best so far and the foot (review M8).

    Drawn at once, before anything is fitted or worked out (review I15):
    the header, "Where do you stand?" line 1 and the foot. The verdict,
    lines 2-3 and "What to do next" fill their places as they are ready —
    here in this run when the tab fits in this process, part by part
    behind one progress line while the workers are still fitting."""
    # A formulation moved to Not scored counts for nothing here (review M-3).
    if len(opt.Y_history) == sum(map(opt.is_set_aside, range(len(opt.Y_history)))):
        st.caption(_say(opt, wording.INSIGHTS_NONE))
        _foot(opt)
        return
    # The one "scored" count on the tab, the header's: formulations with all
    # their results. A waiting one, or one moved to Not scored, is not
    # scored, so it counts towards neither MIN_ROWS nor "So far: N".
    n = opt.complete_count()
    st.caption(_say(opt, header_text(opt)))
    if n < insights.MIN_ROWS:
        _question(opt, wording.Q_STAND)
        _stand_first(opt, insights.noise_yardstick(opt))
        st.caption(_say(opt, wording.insights_gate(n)))
        _foot(opt)
        _how(opt, None)
        return
    filling = waiting(opt, "all")
    # The page in screen order, before any fit (review I15, Plan B Task 6
    # review I-1): the progress line's place, answer 1's, line 1, the
    # places of lines 2-3 and of answer 3, the foot, the Set up button's.
    bar = st.empty()
    # Keyed, so a link from another tab can land on them (04 §5).
    trust = st.container(key=TRUST_SECTION)
    stand = st.container(key=STAND_SECTION)
    with stand:
        _question(opt, wording.Q_STAND)
        first_yard = insights.noise_yardstick(opt)
        lines = st.empty()
        with lines.container():
            facts = _stand_lines(opt, first_yard)
        if facts is not None:
            _progress_fold(opt)
        rest = st.container()
    nxt = st.container()
    _foot(opt)
    below = st.container()
    if filling:
        _fill_in(opt, bar, trust, lines, first_yard, rest, nxt, below)
        _how(opt, None)
        return
    cold = _cached(opt) is None
    computed = _computed(opt, bar)
    if computed.get("failed"):
        # The model answers come off; one line says so, where the bar was.
        bar.empty()
        bar.caption(_say(opt, wording.INSIGHTS_FAILED))
        _how(opt, None)
        return
    checks = computed["checks"]
    h = insights.headline(opt, checks)
    yard = insights.noise_yardstick(opt, checks)
    _restate(opt, lines, first_yard, yard)
    with rest:
        # The bar, when it showed, stays up through the predicted best's
        # search: one indicator per wait (V28, Plan A review C M-2).
        standing = _stand(opt, computed, yard, spinner=not cold)
    bar.empty()
    with nxt:
        _question(opt, wording.Q_NEXT)
        step = insights.next_step(opt, checks, standing, gain=shown_gain(
            opt, computed["fitted"].get(insights.SCORE), checks))
        change = _next_step(opt, step)
    with trust:
        _question(opt, wording.Q_TRUST)
        # The disagreeing Repeat is said once: here, unless "What to do
        # next" already says it.
        _trust(opt, computed, h, yard, repeat=step.kind != "repeat")
    with below:
        _setup_button(change)
    # The folds, collapsed (§2.5), each with a key of its own.
    with st.expander(_say(opt, _try_one_label(opt)), key="insights_try_one"):
        _change(opt, computed, yard)
    _more_charts(opt, computed)
    _how(opt, computed)
    save(opt)


def _restate(opt, lines, first_yard, yard):
    """Line 1 again, in its place, when the checks changed the yardstick it
    was drawn with (a disagreeing Repeat is left out of the Repeat spread)."""
    if yard != first_yard:
        with lines.container():
            _stand_lines(opt, yard)


def _fill_in(opt, bar, trust, lines, first_yard, rest, nxt, below):
    """The tab while the workers are still fitting (spec 2026-09-30 §10.3,
    review I15), into the places render drew: one progress line with a
    count, then answer 1 once every Round's Overall score and the verdict's
    Rounds are in, answers 2 and 3 once the current Overall-score model is
    in too. Each is drawn once, final; the folds wait for everything.
    keep_up's poller reruns the page as parts land."""
    specs, mem = _plan(opt), memo(opt)
    done, total = insights_jobs.rounds_done(opt, specs, mem)
    fraction = sum(s.key in mem for s in specs) / max(1, len(specs))
    bar.progress(fraction, text=_say(opt, wording.insights_progress(done, total)))
    checks = None if waiting(opt, "trust") else _partial_checks(opt, "trust")
    if checks is None:
        return
    yard = insights.noise_yardstick(opt, checks)
    _restate(opt, lines, first_yard, yard)
    step = insights.next_step(opt, checks, None)
    if not waiting(opt, "stand"):
        with insights.memo_in_use(mem):
            score = insights.fit(opt, insights.SCORE, list(range(len(opt.Y_history))))
        computed = {"fitted": {insights.SCORE: score}, "checks": checks, "partial": True}
        with rest:
            standing = _stand(opt, computed, yard)
        with nxt:
            _question(opt, wording.Q_NEXT)
            step = insights.next_step(opt, checks, standing,
                                      gain=shown_gain(opt, score, checks))
            change = _next_step(opt, step)
        with below:
            _setup_button(change)
    with trust:
        _question(opt, wording.Q_TRUST)
        _trust(opt, {"checks": checks, "partial": True}, insights.headline(opt, checks),
               yard, repeat=step.kind != "repeat")


def _foot(opt):
    """The foot Results uses, in its words and by its rule, with a key of its
    own (spec 2026-09-30 §2.4, review M13): back to the open Round, else
    the next one. Needs no model: drawn before anything is worked out."""
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
    under the foot; nothing for None."""
    if change and st.button(change, key="insights_change_setup",
                            disabled=confirmation_open()):
        go_to_tab(TAB_SETUP)


def _try_one_label(opt):
    """The first fold's name (§2.5, review I14): names process settings only
    when the project has one, and ingredients only when it has one."""
    choices = insights.path_choices(opt)
    return wording.try_one(insights.has_setting_choice(opt),
                           insights.has_ingredient_choice(opt)
                           or (not choices and opt.has_ingredients()))


def score_record_help(opt):
    """The Overall score's own record for Make a round's column help, or
    '' — only from checks already worked out for the data on screen, by
    this tab or from the workers' models (§9 I6): Make a round never fits
    for it."""
    checks = checks_for(opt, "trust")
    if checks is None:
        return ""
    return score_line(opt, checks)


def score_line(opt, checks, h=None, yard=None):
    """The Overall score's line, in one wording for Insights' answer 1 and
    Make a round's column help (spec §0.3): what to do only when the gap is
    a bias by the shared rule (insights.bias_gap). '' without a record."""
    h = insights.headline(opt, checks) if h is None else h
    if h.score is None:
        return ""
    yard = insights.noise_yardstick(opt, checks) if yard is None else yard
    bias = insights.score_bias(opt, checks, h, yard)
    beyond = "" if bias is None else threshold_text(yard)
    return wording.score_gap_line(h.score.inside, h.score.total, h.score.mean_miss, beyond,
                                  n=h.score.total)


def threshold_text(yard):
    """The noise threshold's one name (03 §3 rule 3): the Repeat SD from 3
    pairs, else the fixed 1 point."""
    return wording.beat_by(f"{yard.value:.1f}" if yard.from_repeats else None)


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
    return wording.insights_header_line(opt.complete_count(),
                                        order[-1][0] if order else None,
                                        len(waiting), names)


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


def _trust(opt, computed, h, yard, repeat=True):
    """Answer 1 (§2.2): one verdict, the Overall score's own line, a
    disagreeing Repeat (`repeat`), one line on how, and measured against
    predicted in its fold."""
    checks = computed["checks"]
    state = insights.verdict_state(opt, h, checks)
    st.write(_say(opt, verdict_text(opt, h, state, checks=checks)))
    if h.score is not None:
        st.write(_say(opt, score_line(opt, checks, h, yard)))
    # Skill, the Overall score's own fact under the verdict (review I1): in
    # every state past the gate but "none".
    skill = skill_text(opt, checks) if state != "none" else ""
    if skill:
        st.write(_say(opt, skill))
    flagged = insights.disagreeing_repeat(opt, checks) if repeat else None
    if flagged is not None:
        st.write(_say(opt, disagreement_text(flagged)))
    st.caption(_say(opt, wording.TRUST_BEFOREHAND))
    # While the rest are still being fitted the chart would gain dots: it
    # waits for them (a part of the tab is drawn once, final). Keyed, so a
    # link can open it (E1, EACH_KEY); drawn only while open (04 §5).
    if computed.get("partial") or not checks:
        return
    with st.expander(_say(opt, wording.SHOW_PREDICTIONS), key=EACH_KEY,
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
    # The answer line, Compare all and the choice's default stay on Best so
    # far (or Section 2's own Start from) whatever the layer: only the path
    # and its caption follow the layers' anchor (T56 I3).
    ranked_no = _start_number(opt) if layered else number
    try:
        ranked_recipe = recipe if ranked_no == number else anchor_recipe(opt, ranked_no)
    except Exception:
        ranked_recipe = None
    try:
        found = insights.tested_near_changes(opt, fitted, ranked_recipe)
    except Exception:   # the answer line is left out; the path still draws
        found = insights.NearChanges([], [], False)
    # The box opens on the largest change near what has been tested; after
    # that it keeps the reader's own choice (_keep_if_offered drops it only
    # when it is no longer offered).
    if found.moved and CHOICE_KEY not in st.session_state:
        st.session_state[CHOICE_KEY] = found.moved[0].name
    with answer:
        _near_answer(opt, found, ranked_no == best_no, ranked_no, fitted, ranked_recipe, yard)
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
              alt.Chart(df).mark_rule(strokeDash=[2, 2], tooltip=False).encode(x="x:Q")
              .transform_filter(_only("start"))]
    if any(r["mark"] == "grey" for r in rows):
        layers.insert(0, alt.Chart(df).mark_rect(opacity=_GREY, color="gray", tooltip=False)
                      .encode(x="x:Q", x2="x2:Q").transform_filter(_only("grey")))
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


def _near_answer(opt, found, from_best, number, fitted, recipe, yard):
    """The fold's first line (§2.5, review I8): the choice that changes the
    predicted Overall score most near what has been tested — two when they
    are nearly tied — only when that change is more than Repeats differ
    (the yardstick); a line naming any choice the limits hold in place, one
    naming two ingredients moved together, and the Compare all fold."""
    start = wording.spread_from(from_best, number)
    # With no ingredient to change, every choice is a process setting: it
    # has values, not amounts, and the lines name process settings alone.
    settings_only = not insights.has_ingredient_choice(opt)
    moved = found.moved
    if not moved:
        if found.held:
            st.write(_say(opt, wording.held_all(start, settings_only)))
        return
    top = moved[0]
    standing = [c for c in moved if c.near is not None and c.near > yard.value]
    # The noise guard (spec 2026-09-30-bo-views §2.3, review I3, I4): no
    # effect is ranked, nor the Compare all bars drawn, while the Overall-
    # score model is likely fitting noise. The lines that need no model
    # (held in place, correlated) stay.
    guard = _guard(opt, fitted)
    if guard.fires:
        st.write(_say(opt, guard_text(guard, settings_only)))
    elif not found.start_tested:
        st.write(_say(opt, wording.near_start_untested(start)))
    elif not standing:
        st.write(_say(opt, wording.near_none(settings_only, insights.has_setting_choice(opt))))
    else:
        second = moved[1] if len(moved) > 1 else None
        if (second is not None and second.near is not None
                and second.near >= insights.NEAR_TIE * top.near):
            st.write(_say(opt, wording.near_two(top.name, second.name, f"{top.near:.1f}",
                                                f"{second.near:.1f}", start)))
        else:
            st.write(_say(opt, wording.near_line(top.name, f"{top.near:.1f}", start,
                                                 only=len(moved) == 1)))
        if top.whole > top.near + _FLAT:
            st.write(_say(opt, wording.near_bigger(values=insights.is_setting(opt, top.name))))
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


_GUARD_WORDS = {"floor": wording.GUARD_FLOOR, "noise": wording.GUARD_NOISE}


def _guard(opt, fitted, name=insights.SCORE):
    """insights.noise_guard for the display model of `name`, with the
    checks already worked out for this data (none: skill is not judged)."""
    fitted = fitted or {}
    return insights.noise_guard(opt, fitted.get(name), checks_for(opt, "trust") or (), name)


def guard_text(guard, settings_only=False):
    """The guard's one line, naming its trigger (review I3): "fitting
    noise" for the correlation length or the fitted noise; trigger (c)
    alone reads "not reported until…"."""
    if guard.skill_only:
        return wording.guard_skill_line(settings_only)
    return wording.guard_line([_GUARD_WORDS[r] for r in guard.reasons if r != "skill"],
                              settings_only)


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
        lines.append(wording.PATH_GUARD_SKILL if guard.skill_only else wording.PATH_GUARD)
    return lines


def stand_text(answer):
    """The Yes / Barely / Not lately answer (§2.3, review I5, M3). One
    decimal, unless the rise would read 0.0 or the same as what it is
    judged against: then hundredths, so the words never contradict the
    numbers (Task 6 review M-1)."""
    if answer.state == "not_lately":
        return wording.stand_not_lately(answer.rounds)
    threshold = 1.0 if answer.sd is None else answer.sd
    d = 2 if f"{answer.rise:.1f}" in ("0.0", f"{threshold:.1f}") else 1
    rise = f"{answer.rise:.{d}f}"
    sd = None if answer.sd is None else f"{answer.sd:.{d}f}"
    if answer.state == "yes":
        return wording.stand_yes(rise, wording.beat_by(sd))
    return wording.stand_barely(rise, sd)


def _stand_first(opt, yard):
    """"Where do you stand?" line 1 (§2.3) and the Progress chart's fold.
    Needs no model: drawn at the gate too."""
    if _stand_lines(opt, yard) is not None:
        _progress_fold(opt)


def _progress_fold(opt):
    """The Progress chart fold (review M8): Best so far by formulation, then
    Expected improvement by Round — worked out only once the fold is open
    (binding I15), and only from what the tab has worked out already."""
    with st.expander(_say(opt, wording.SHOW_PROGRESS), key=PROGRESS_KEY,
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
    """Line 1's text (§2.3): the answer, Best so far to two decimals as
    Results shows it (review M1), and the Repeats line only after a rise
    (audit V24). Returns insights.best_facts (None: nothing is scored)."""
    f = insights.best_facts(opt)
    if f is None:
        # Everything recorded still waits, in the words Results uses.
        st.write(_say(opt, wording.NO_COMPLETE_FORMULATION))
        return None
    answer = insights.improvement(opt, f, yardstick=yard)
    second = ("" if answer is not None
              else wording.STAND_LATEST if f.latest and f.rounds_since == 0
              else wording.stand_since(f.rounds_since) if f.rounds_since else "")
    line = wording.stand_best(f"{f.score:.2f}", f.formulation, f.round, number_list(f.without))
    head = stand_text(answer) + " " if answer is not None else ""
    st.write(_say(opt, head + line + (" " + second if second else "")))
    if (answer is not None and answer.state in ("yes", "barely")
            and yard.sd is not None and yard.sd >= insights.REPEAT_SD_SHOWN):
        st.write(_say(opt, wording.stand_repeats(f"{yard.sd:.1f}", yard.pairs)))
    return f


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


def standing_text(opt, standing, yard):
    """Line 2 (§2.3, review I7): the predicted best against Best so far's
    predicted Repeat, in whole points, in one of four texts."""
    p = standing.best.score
    mean, low, high = _whole(p.mean), _whole(p.low), _whole(p.high)
    by = threshold_text(yard)
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
    standing = insights.best_standing(opt, best, yard.value)
    st.write(_say(opt, standing_text(opt, standing, yard)))
    if standing.binding:
        st.write(_say(opt, wording.stand_limits([binding_text(b) for b in standing.binding])))
    top = opt.best_index()
    if standing.luck and best.repeat is not None and top is not None:
        r = best.repeat
        st.write(_say(opt, wording.best_luck(int(opt.formulation_ids[top]), _whole(r.mean),
                                             _whole(r.low), _whole(r.high),
                                             f"{opt.Y_history[top]:.2f}")))
    # Expected improvement by Round (spec 2026-09-30-bo-views §3.2; review
    # I5, M8): its line here, its chart in the Progress chart fold.
    line = gain_text(opt, computed, yard, standing)
    if line:
        st.write(_say(opt, line))
    if standing.higher and not standing.tested_near:
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


def _gain_key(opt):
    record = (getattr(opt, "round_generation", None) or {}).get(opt.pending_batch_no)
    return insights._digest({"setup": insights.setup_key(opt),
                             "acquisition": bo_views.acquisition(opt),
                             "open": opt.pending_batch, "record": record})


def _gain_needs_no_fit(opt, mem):
    """Whether bo_views.latest_gain needs no display-model fit: a cold
    Round, an exact one (one default fit of the suggestion model, as the
    Why fold makes), or a recomputed one whose display model is in the
    memo."""
    r = opt.pending_batch_no if opt.pending_batch else None
    if r is None:
        order = insights.round_order(opt)
        r = order[-1][0] if order else None
    if r is None:
        return True
    gen = bo_views.generation(opt, r)
    if gen.regime != "warm" or gen.exact:
        return True
    use = [i for i in gen.rows if insights.measured(opt, insights.SCORE, i) is not None]
    return insights.model_key(opt, insights.SCORE, use) in mem


def latest_gain(opt, fit=True):
    """bo_views.latest_gain once per _gain_key. `fit` False (Make a round):
    None, not stored, when it would need a display-model fit."""
    key = _gain_key(opt)
    hit = st.session_state.get(GAIN_KEY)
    if hit and hit[0] == key:
        return hit[1]
    mem = memo(opt)
    if not fit and not _gain_needs_no_fit(opt, mem):
        return None
    try:
        with insights.memo_in_use(mem):
            g = bo_views.latest_gain(opt)
    except Exception:
        g = None
    st.session_state[GAIN_KEY] = (key, g)
    return g


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
    (a caution or unreliable verdict), "skill" (no skill), "guard" (the
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
        return wording.gain_unreliable({"verdict": wording.GAIN_REASON_VERDICT,
                                        "skill": wording.GAIN_REASON_SKILL,
                                        "guard": wording.GAIN_REASON_GUARD}[why])
    if g is None or g.why_none == "own":
        return ""
    if g.why_none == "cold":
        return wording.gain_cold(g.round, values=not opt.has_ingredients())
    best = opt.best_index()
    measured = (f"{opt.Y_history[best]:.2f}" if bo_views.acquisition(opt) == "qlogei"
                and best is not None else None)
    if g.gain < 0.005:
        # Not "0.00 points … beats … by about 0.00 points" (T56 m3).
        return wording.gain_line_tiny(g.round, threshold_text(yard), recomputed=not g.exact)
    return wording.gain_line(f"{g.gain:.2f}", g.round, threshold_text(yard),
                             above=g.gain > yard.value, recomputed=not g.exact,
                             measured_text=measured,
                             predicted_none=standing is not None and not standing.higher)


def next_step_text(opt, step, rounds=None):
    """(sentence, Set up button label or None) for insights.next_step's
    answer (§2.4, review I2, I3, I14, M17; BO views I1). `rounds`: the
    verdict's Rounds, named in the bias and misses rows (03 §5.4)."""
    values = not opt.has_ingredients()
    settings_only = not insights.has_ingredient_choice(opt)
    # A Round still open: record it first, as the foot says (final review M-1).
    left = len(open_rows(opt)) if opt.pending_batch else 0
    no = opt.pending_batch_no
    if step.kind == "bias":
        return wording.next_bias(f"{abs(step.gap):.1f}", below=step.gap < 0,
                                 rounds=rounds), None
    if step.kind == "misses":
        return wording.next_misses(rounds), None
    if step.kind == "repeat":
        return disagreement_text(step.disagreement), None
    if step.kind == "early":
        return (wording.next_too_early_open(no, left) if left else wording.NEXT_TOO_EARLY), None
    if step.kind == "no_skill":
        return wording.next_no_skill(skill_value(step.skill.value), step.skill.n, no,
                                     left, rounds=step.skill.rounds), None
    if step.kind == "untested":
        return wording.next_untested(values, open_round=bool(opt.pending_batch),
                                     round_no=no), None
    if step.kind == "explore":
        return wording.next_explore(f"{step.gain:.2f}", no, left), None
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


def _next_step(opt, step):
    """Answer 3: one sentence, naming the verdict's Rounds where it says a
    bias or misses (from the checks already worked out). Returns the Set up
    button's label, or None."""
    checks = checks_for(opt, "trust")
    rounds = insights.headline(opt, checks).rounds if checks else None
    text, change = next_step_text(opt, step, rounds)
    st.write(_say(opt, text))
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

# One hue, light to dark, starting a quarter in: the palest blues read as
# the light grey of a point that breaks a limit.
_BLUES = alt.SchemeParams(name="blues", extent=[0.25, 1])


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
    for o, enc in ((oa, "x"), (ob, "y")):
        if _target_line(o):
            layers.append(alt.Chart(pd.DataFrame({"t": [float(o['target'])]}))
                          .mark_rule(strokeDash=[4, 3], strokeWidth=1, color="gray",
                                     tooltip=False)
                          .encode(**{enc: alt.X("t:Q") if enc == "x" else alt.Y("t:Q")}))
    # A hollow grey dot is named in the legend, only when there is one: a
    # one-entry stroke legend under the Overall score colour bar.
    hollow = {}
    if df["waiting"].any():
        hollow["stroke"] = alt.Stroke("state:N",
                                      scale=alt.Scale(domain=[waiting], range=["gray"]),
                                      legend=alt.Legend(title=None, labelLimit=0))
    layers.append(base.mark_point(filled=False, color="gray", size=60)
                  .encode(x=x, y=y, tooltip=tip, **hollow)
                  .transform_filter(alt.FieldEqualPredicate(field="waiting", equal=True)))
    layers.append(base.mark_circle(size=70, opacity=1)
                  .encode(x=x, y=y, tooltip=tip,
                          color=alt.Color("score:Q", title=overall,
                                          scale=alt.Scale(scheme=_BLUES,
                                                          domain=[0, opt.utility_ceiling()])))
                  .transform_filter(alt.FieldEqualPredicate(field="waiting", equal=False)))
    best = base.transform_filter(alt.FieldEqualPredicate(field="best", equal=True))
    layers.append(best.mark_point(filled=False, size=260, strokeWidth=2, color="gray")
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
    predicted = _say(opt, {wording.LAYER_SD: wording.LAYER_SD_LEGEND,
                           wording.LAYER_GAIN: wording.LAYER_GAIN_LEGEND}.get(
                               layer, wording.PREDICTED_SCORE_COLUMN))
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
    grey = (alt.Chart(df).mark_circle(opacity=1, color="lightgray")
            .encode(x=x, y=y, size=size,
                    tooltip=amounts + [alt.Tooltip("why:N", title=kinds[2])])
            .transform_filter(blocked))
    grid = (alt.Chart(df).mark_circle(opacity=1)
            .encode(x=x, y=y, size=size,
                    color=alt.Color("score:Q", title=predicted,
                                    scale=alt.Scale(scheme=_BLUES,
                                                    domain=[lo, hi]),
                                    legend=alt.Legend(titleLimit=0)),
                    tooltip=amounts + [alt.Tooltip("pred:N", title=predicted)])
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
    state_colors = alt.Scale(domain=kinds, range=["steelblue", "steelblue", "lightgray"])
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
        layers.append(alt.Chart(df).mark_circle(filled=False, stroke=outline_colour(),
                                                strokeWidth=1.5,
                                                size=big, opacity=1, tooltip=False)
                      .encode(x=x, y=y)
                      .transform_filter(alt.FieldEqualPredicate(field="outline", equal=True)))
    if mix.dots:
        dots = pd.DataFrame([{**dict(zip(("X", "Y"), _xy(d.x))), "no": d.formulation}
                             for d in mix.dots])
        layers.append(alt.Chart(dots).mark_point(filled=True, fill="white", stroke="gray",
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
    own = [] if start_pred is None else [alt.Tooltip("pred:N", title=predicted)]
    layers.append(alt.Chart(start).mark_point(filled=False, size=260, strokeWidth=2.5,
                                              color="gray")
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
    """The three layers (review I6), or the Predicted score layer alone:
    while the noise guard fires (review I4), and without a model yet."""
    rows = [i for i in range(len(opt.Y_history)) if opt.fully_measured(i)]
    if len(rows) < insights.MIN_ROWS or _model_guard(opt, fitted).fires:
        return (wording.LAYER_SCORE,)
    if opt.pending_batch and opt.pending_batch_no is not None:
        if bo_views.generation(opt, opt.pending_batch_no).regime != "warm":
            return (wording.LAYER_SCORE,)
    if bo_views.acquisition(opt) == "qucb":
        # qUCB has no expected improvement: its value is an optimistic
        # score, not a gain (T56 I2).
        return (wording.LAYER_SCORE, wording.LAYER_SD)
    return _LAYERS


def _layer_pick(opt, fitted, key):
    """The one radio (review I6): Predicted score, Uncertainty, Expected
    gain. Not drawn when only the first is offered."""
    offered = layers_offered(opt, fitted)
    if len(offered) == 1:
        return wording.LAYER_SCORE
    _keep_if_offered(key, offered)
    return st.radio(_say(opt, wording.LAYER_PICK), offered, key=key, horizontal=True)


def layer_model(opt, fitted):
    """The suggestion model the layers show (spec §3.3): the open Round's,
    rebuilt (bo_views.generation); with no Round open, the one the next
    Generate would fit (today's rows, the next seed). The display model of
    the same rows stands in when the generation is not exact."""
    rows = [i for i in range(len(opt.Y_history)) if opt.fully_measured(i)]
    if opt.pending_batch and opt.pending_batch_no is not None:
        gen = bo_views.generation(opt, opt.pending_batch_no)
    else:
        gen = bo_views.Generation(None, int(opt.next_formulation_no), rows,
                                  "warm" if len(rows) >= insights.MIN_ROWS else "cold", False)
    score = (fitted or {}).get(insights.SCORE)
    with insights.memo_in_use(memo(opt)):
        return bo_views.rebuild(opt, gen, score if gen.rows == rows else None)


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
    key = (insights.setup_key(opt), tuple(names), number, layer)
    hit = st.session_state.get(MIXTURE_KEY)
    if hit and hit[0] == key:
        return hit[1]
    with st.spinner(_say(opt, wording.INSIGHTS_WORKING)):
        try:
            mix = insights.mixture_slice(opt, names, start)
            preds = insights.mixture_predictions(opt, fitted, mix)
            value = (mix, preds, _start_prediction(opt, fitted, mix, start),
                     _mixture_extra(opt, fitted, mix, preds, layer))
        except Exception:
            value = _FAILED
    st.session_state[MIXTURE_KEY] = (key, value)
    return value


def _repeat_of_best(opt, fitted):
    """Best so far's predicted Repeat (binding speed-spec I7): the predicted
    best's when it is worked out, else the display model's at its amounts."""
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


def _mixture_extra(opt, fitted, mix, preds, layer):
    """What the chart needs beyond the predictions, per point (None where
    blocked): a layer's values; or, on the Predicted score layer, the
    outline (the model's mean − 1.96 × its own sd above Best so far's
    predicted Repeat) and (colour span, median 1.96 × model sd) for the
    §2.2 caption. {} when the suggestion model cannot be rebuilt."""
    allowed = [p.recipe for p in mix.points if p.recipe is not None]
    model = layer_model(opt, fitted)
    if model is None or not allowed:
        return {}
    got = bo_views.evaluate(opt, model, bo_views.points(opt, allowed))
    spread = lambda seq: iter(seq)
    if layer != wording.LAYER_SCORE:
        it = spread((got.sd if layer == wording.LAYER_SD else got.value).tolist())
        return {"values": [next(it) if p.recipe is not None else None for p in mix.points]}
    ref = _repeat_of_best(opt, fitted)
    low = (got.mean - insights.Z95 * got.sd).tolist()
    it = spread(low)
    top = opt.best_index()
    measured_top = None if top is None else float(opt.Y_history[top])
    yard = insights.noise_yardstick(opt, checks_for(opt, "trust") or ()).value
    ring = []
    for p, pred in zip(mix.points, preds or [None] * len(mix.points)):
        lo = next(it) if p.recipe is not None else None
        ring.append(p.recipe is not None and insights.above_best(
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
    return (len(opt.objectives) >= 2
            or (len(insights.mixture_choices(opt)) >= 3 and score is not None
                and score.model is not None))


def _more_charts(opt, computed):
    if not more_charts_offered(opt, computed):
        return
    with st.expander(_say(opt, wording.MORE_CHARTS)):
        for draw in (lambda: _pair(opt),
                     lambda: _mixture(opt, computed["fitted"])):
            try:
                draw()
            except Exception:   # one chart failing must not take the tab down
                st.caption(_say(opt, wording.CHART_FAILED))


def _how(opt, computed):
    """"How these insights are made" (§2.5; review M5, M6, M10): plain
    paragraphs, the two method lines that were under the verdict (audit
    V23), then the details table with its window named and a "Likely range
    widened by" column, and the lines that read it."""
    with st.expander(_say(opt, wording.HOW_WORKED_OUT), key=HOW_KEY,
                     on_change="rerun") as fold:
        # Keyed and worked out only while open (T56 I5; binding I15).
        if not fold.open:
            return
        tiny = (getattr(opt, "bo_config", None) or {}).get("noise") == "fixed_tiny"
        for text in (wording.HOW_MODELS,
                     wording.HOW_RANGE_FIXED_TINY if tiny else wording.HOW_RANGE,
                     wording.how_checks(values=not opt.has_ingredients(),
                                        total=opt.has_formulation_total()),
                     wording.HOW_VERDICT, wording.HOW_NOISE_THRESHOLD,
                     wording.how_same_process(values=not opt.has_ingredients()),
                     wording.HOW_NO_P):
            st.markdown(_say(opt, text))
        if computed is None:
            return
        checks = computed["checks"]
        skipped = insights.rounds_not_checked(opt)
        if skipped and checks:
            st.caption(_say(opt, wording.rounds_not_checked(skipped)))
        flagged = insights.disagreeing_repeat(opt, checks)
        rows = insights.details(opt, checks, computed["fitted"], flagged)
        if not rows:
            st.caption(_say(opt, wording.DETAILS_NONE))
            return
        h = insights.headline(opt, checks)
        st.caption(_say(opt, wording.details_window(insights._ordered_rounds(checks),
                                                    len(h.rounds), h.rounds)))
        columns = wording.DETAILS_COLUMNS_2
        st.dataframe(pd.DataFrame([{
            columns[0]: panel_label(opt, r.name),
            columns[1]: r.checked,
            columns[2]: r.inside,
            columns[3]: f"± {_num(r.half_width)}",
            columns[4]: _num(r.typical_miss),
            columns[5]: _say(opt, widen_cell(opt, r, checks)),
            columns[6]: wording.repeat_compare(r.ratio, r.repeat_sd, r.pairs, r.left_out),
            # Each output's own skill (spec 2026-09-30-bo-views §2.1, I2).
            wording.SKILL_COLUMN: skill_cell(opt, checks, r.name),
        } for r in rows]), hide_index=True, height=table_height(len(rows)))
        # The Overall-score row carries a second factor when its cell is
        # more than its own fit's factor (the sentence's words are wording's).
        top = any(r.name == insights.SCORE and r.widen is not None
                  and widen_cell(opt, r, checks) != wording.widen_text(r.widen)
                  for r in rows)
        for text in (wording.HOW_WIDENED, wording.HOW_WIDENED_SCORE if top else "",
                     wording.HOW_REPEATS_COMPARED, wording.HOW_THRESHOLDS, wording.HOW_BIAS,
                     how_skill_text(),
                     wording.HOW_EXPECTED_IMPROVEMENT,
                     wording.how_roles(f"{bo_views.POOL_SIZE:,}")):
            if text:
                st.caption(_say(opt, text))


def how_skill_text():
    """Method and details' skill paragraph, with the no-skill cut and the
    simulation gate's 10th percentiles by n (fix round 1, T2 I-1)."""
    gate = ", ".join(f"{n}: {skill_value(v)}" for n, v in sorted(insights.SKILL_GATE_Q10.items()))
    return wording.how_skill(skill_value(insights.skill_cut(insights.SKILL_MIN)), gate)


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
