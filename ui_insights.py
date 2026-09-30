"""The Insights tab. Builders here return Altair charts from computed facts
and never call Streamlit; render() at the bottom is the only Streamlit code."""
import contextlib
import itertools
import math

import altair as alt
import pandas as pd
import streamlit as st

import insights
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
    tip = [alt.Tooltip("amount:N", title=path.name),
           alt.Tooltip("predicted:N", title=_say(opt, wording.TRACK_PREDICTED))]
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


def fitted_models(opt, working=wording.INSIGHTS_WORKING, bar=None):
    """insights.fit_all for this project's data, fitted once per data_key.
    None when the fit failed. With `bar` (Insights' one progress bar) the
    fit is its first step instead of a spinner of its own."""
    key = insights.data_key(opt)
    hit = st.session_state.get(FITTED_KEY)
    if hit and hit[0] == key:
        return hit[1]
    if bar is not None:
        bar.progress(0.0, text=_say(opt, wording.CHECKING))
    with (contextlib.nullcontext() if bar is not None else st.spinner(working)):
        try:
            # Through the module attribute, so a test can stand in for the fit.
            value = insights.fit_all(opt)
        except Exception:
            value = None
    st.session_state[FITTED_KEY] = (key, value)
    return value


def _stepper(opt, bar):
    """check_rounds' `checking`: moves the one progress bar on as each Round
    is checked (0.8.0 review fixes §6, §9 I11)."""
    order = [r for r, _ in insights.round_order(opt)]

    @contextlib.contextmanager
    def checking(r, last):
        bar.progress(min(1.0, (order.index(r) + 1) / (len(order) + 1)),
                     text=_say(opt, wording.CHECKING))
        yield
    return checking


def _cached(opt):
    """_computed's value for this project's data when it is already worked
    out, else None."""
    hit = st.session_state.get(CACHE_KEY)
    return hit[1] if hit and hit[0] == insights.data_key(opt) else None


def cached_checks(opt):
    """The Round-by-Round checks for this project's data when Insights has
    already worked them out, else (). Results and Make a round read them
    here and never check Rounds themselves (spec 2026-09-30 Plan B; Plan A
    swaps in its precomputed cache behind this one door)."""
    hit = _cached(opt)
    return () if hit is None or hit.get("failed") else hit["checks"]


def _computed(opt, bar=None):
    """The fitted models and the Round-by-Round checks, once per data_key.
    `bar`, a slot under the header, shows one progress bar while they are
    worked out; without it (a test, or a caller with no slot) a slot of
    its own."""
    hit = _cached(opt)
    if hit is not None:
        return hit
    key = insights.data_key(opt)
    bar = bar if bar is not None else st.empty()
    fitted = fitted_models(opt, bar=bar)
    if fitted is None:
        value = _FAILED
    else:
        rounds = st.session_state.setdefault(ROUNDS_KEY, {})
        checking = _stepper(opt, bar)
        try:
            value = {"fitted": fitted,
                     "checks": insights.check_rounds(opt, cache=rounds, checking=checking)}
        except Exception:
            # Cached like a success, under the same key: one line on the
            # tab instead of a traceback, and not tried again every rerun.
            value = _FAILED
    st.session_state[CACHE_KEY] = (key, value)
    return value


_FAILED = {"failed": True}


def _say(opt, text):
    """Every line on the tab in the project's own words, as the rest of the
    app writes them: trials, not formulations, in a project of process
    settings alone."""
    return wording.for_project(opt, text)


def _question(opt, text):
    st.markdown(f"**{_say(opt, text)}**")


def render(opt):
    """The tab (spec 2026-09-30 §2): the header, three answers — "Can you
    trust the predictions?", "Where do you stand?", "What to do next" — the
    foot button, then three folds. At the gate (fewer than MIN_ROWS
    scored), after a failed fit and with nothing scored, the header (or
    one line), Best so far and the foot (review M8)."""
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
    bar = st.empty()
    computed = _computed(opt, bar)
    bar.empty()
    if computed.get("failed"):
        # The model answers come off; one line says so, where the bar was.
        bar.caption(_say(opt, wording.INSIGHTS_FAILED))
        _question(opt, wording.Q_STAND)
        _stand_first(opt, insights.noise_yardstick(opt))
        _foot(opt)
        _how(opt, None)
        return
    checks = computed["checks"]
    h = insights.headline(opt, checks)
    yard = insights.noise_yardstick(opt, checks)
    trust = st.container()
    _question(opt, wording.Q_STAND)
    standing = _stand(opt, computed, yard)
    _question(opt, wording.Q_NEXT)
    step = insights.next_step(opt, checks, standing)
    change = _next_step(opt, step)
    with trust:
        _question(opt, wording.Q_TRUST)
        # The disagreeing Repeat is said once: here, unless "What to do
        # next" already says it.
        _trust(opt, computed, h, yard, repeat=step.kind != "repeat")
    _foot(opt, change)
    # The folds, collapsed (§2.5), each with a key of its own.
    with st.expander(_say(opt, _try_one_label(opt)), key="insights_try_one"):
        _change(opt, computed, yard)
    _more_charts(opt, computed)
    _how(opt, computed)


def _foot(opt, change=None):
    """The foot Results uses, in its words and by its rule, with a key of its
    own (spec 2026-09-30 §2.4, review M13): back to the open Round, else
    the next one. `change`: (label, ...) of the secondary Set up button
    "What to do next" asked for, drawn under the foot, or None."""
    if opt.pending_batch:
        label = wording.back_to_batch_label(opt.pending_batch_no, len(open_rows(opt)))
    else:
        label = wording.START_NEXT_BATCH
    ready, _ = readiness(opt)
    lit = ready and not confirmation_open()
    if st.button(label, type="primary" if lit else "secondary", disabled=not lit,
                 key="insights_foot") and lit:
        go_to_tab(TAB_BATCH)
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
    '' — only when this tab's checks are already worked out for the data on
    screen (§9 I6): Make a round never checks Rounds itself."""
    hit = _cached(opt)
    if hit is None or hit.get("failed"):
        return ""
    rec = insights.headline(opt, hit["checks"]).score
    if rec is None:
        return ""
    return wording.score_record_line(rec.inside, rec.total, rec.mean_miss)


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
    names = [f"the {wording.OVERALL_SCORE_COLUMN}" if n == insights.SCORE else n
             for n in h.wide]
    return wording.wide_line(number_list(names))


def verdict_text(opt, h, state):
    """"Can you trust the predictions?" in one line per state (§2.2,
    review I6, M9); some outputs wide adds the wide line."""
    if state == "none":
        return wording.TRUST_NONE
    if state == "early":
        return wording.trust_early(h.total)
    if state == "wide":
        return wording.TRUST_WIDE
    if state in ("caution", "unreliable"):
        line = wording.trust_misses(len(h.rounds), h.rounds, h.inside, h.total,
                                    many=state == "unreliable")
    elif state == "over_wide":
        line = wording.trust_over_wide(h.total, h.rounds)
    else:
        line = wording.trust_within(h.inside, h.total, h.rounds)
    wide = _wide(opt, h)
    return f"{line} {wide}" if wide else line


def _beyond(gap, yard):
    """What the Overall score's average gap is more than, when it is (review
    I5): "Repeats differ" from 3 pairs, else "1 point"; '' within."""
    if yard.value is None or abs(gap) <= yard.value:
        return ""
    return "Repeats differ" if yard.from_repeats else "1 point"


def disagreement_text(d):
    return wording.repeat_disagrees(d.later, d.earlier, f"{abs(d.diff):.1f}", lower=d.diff < 0)


def _trust(opt, computed, h, yard, repeat=True):
    """Answer 1 (§2.2): one verdict, the Overall score's own line, a
    disagreeing Repeat (`repeat`), one line on how, and measured against
    predicted in its fold."""
    checks = computed["checks"]
    st.write(_say(opt, verdict_text(opt, h, insights.verdict_state(opt, h, checks))))
    if h.score is not None:
        st.write(_say(opt, wording.score_gap_line(h.score.inside, h.score.total,
                                                  h.score.mean_miss,
                                                  _beyond(h.score.mean_miss, yard))))
    flagged = insights.disagreeing_repeat(opt, checks) if repeat else None
    if flagged is not None:
        st.write(_say(opt, disagreement_text(flagged)))
    st.caption(_say(opt, wording.TRUST_BEFOREHAND))
    chart = measured_vs_predicted_chart(opt, checks)
    if chart is not None:
        with st.expander(_say(opt, wording.SHOW_PREDICTIONS)):
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
    # By Formulation number, not list position: a deletion on Results shifts
    # positions, and the box must keep naming the formulation it named.
    starts = [best_no] + [ids[i] for i in reversed(range(len(ids))) if i != best]
    _keep_if_offered(CHOICE_KEY, choices)
    _keep_if_offered(START_KEY, starts)
    answer = st.container()
    left, right = st.columns(2)
    number = right.selectbox(
        _say(opt, wording.START_FROM), starts, key=START_KEY,
        format_func=lambda no: _say(opt, wording.start_best(no) if no == best_no
                                    else wording.start_formulation(no)))
    fitted = computed["fitted"]
    try:
        recipe = insights.start_recipe(opt, ids.index(number))
        found = insights.tested_near_changes(opt, fitted, recipe)
    except Exception:   # the answer line is left out; the path still draws
        recipe, found = None, insights.NearChanges([], [], False)
    # The box opens on the largest change near what has been tested; after
    # that it keeps the reader's own choice (_keep_if_offered drops it only
    # when it is no longer offered).
    if found.moved and CHOICE_KEY not in st.session_state:
        st.session_state[CHOICE_KEY] = found.moved[0].name
    with answer:
        _near_answer(opt, found, number == best_no, number, fitted, recipe, yard)
    name = left.selectbox(_say(opt, wording.INGREDIENT_OR_SETTING), choices, key=CHOICE_KEY)
    try:
        path = insights.build_path(opt, name, insights.start_recipe(opt, ids.index(number)))
        if path.start_breaks:
            st.caption(_say(opt, wording.path_start_breaks(number, path.start_breaks)))
            return
        near = insights.tested_near(opt, path)
        chart = path_chart(opt, path, insights.path_predictions(opt, fitted, path), near)
        lines = _path_captions(opt, path, number, fitted, near)
    except Exception:   # one path failing must not take the tab down
        st.caption(_say(opt, wording.panel_failed(name)))
        return
    if chart is not None:
        show_chart(chart)
    for line in lines:
        st.caption(_say(opt, line))


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
    if not found.start_tested:
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
    together = insights.moved_together(opt)
    if together:
        st.caption(_say(opt, wording.moved_together(*together)))
    if len(moved) > 1:
        with st.expander(_say(opt, wording.compare_all(insights.has_setting_choice(opt),
                                                       not settings_only))):
            _compare_all(opt, moved, settings_only, fitted, recipe)


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
    if answer.state == "yes":
        return wording.stand_yes(rise)
    return wording.stand_barely(rise, None if answer.sd is None else f"{answer.sd:.{d}f}")


def _stand_first(opt, yard):
    """"Where do you stand?" line 1 (§2.3): the answer, Best so far to two
    decimals as Results shows it (review M1), and the Repeats line only
    after a rise (audit V24). Needs no model: drawn at the gate too."""
    f = insights.best_facts(opt)
    answer = insights.improvement(opt, f, yardstick=yard)
    if f is None:
        # Everything recorded still waits, in the words Results uses.
        st.write(_say(opt, wording.NO_COMPLETE_FORMULATION))
        return answer
    second = ("" if answer is not None
              else wording.STAND_LATEST if f.latest and f.rounds_since == 0
              else wording.stand_since(f.rounds_since) if f.rounds_since else "")
    line = wording.stand_best(f"{f.score:.2f}", f.formulation, f.round, number_list(f.without))
    head = stand_text(answer) + " " if answer is not None else ""
    st.write(_say(opt, head + line + (" " + second if second else "")))
    if (answer is not None and answer.state in ("yes", "barely")
            and yard.sd is not None and yard.sd >= insights.REPEAT_SD_SHOWN):
        st.write(_say(opt, wording.stand_repeats(f"{yard.sd:.1f}", yard.pairs)))
    with st.expander(_say(opt, wording.SHOW_PROGRESS)):
        _progress_chart_body(opt)
    return answer


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


def _predicted_best(opt, fitted, checks=()):
    """insights.predicted_best, once per setup_key (the checks follow from
    the data it covers). A failure is cached as such and shows one line."""
    key = insights.setup_key(opt)
    hit = st.session_state.get(BEST_KEY)
    if hit and hit[0] == key:
        return hit[1]
    with st.spinner(wording.INSIGHTS_WORKING):
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
    if standing.higher:
        close = standing.best.close
        return wording.stand_higher(mean, low, high, standing.close_no,
                                    "" if close is None else f"{opt.Y_history[close]:.2f}")
    by = wording.beat_by(f"{yard.value:.1f}" if yard.from_repeats else None)
    return wording.stand_not_higher(by, mean, low, high, standing.close_no)


def _stand(opt, computed, yard):
    """Answer 2 (§2.3): line 1, then the predicted best against Best so
    far (line 2), what it sits at (line 3), the luck line, and its amounts
    in a fold only when it is clearly higher and not tested near. Returns
    the insights.Standing, or None when the predicted best could not be
    worked out (one line says why)."""
    _stand_first(opt, yard)
    fitted = computed["fitted"]
    score = fitted.get(insights.SCORE)
    if score is None or score.model is None:
        st.caption(_say(opt, wording.BEST_FAILED))
        return None
    best = _predicted_best(opt, fitted, computed["checks"])
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
                                             _whole(r.low), _whole(r.high))))
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


def next_step_text(opt, step):
    """(sentence, Set up button label or None) for insights.next_step's
    answer (§2.4, review I2, I3, I14, M17)."""
    values = not opt.has_ingredients()
    settings_only = not insights.has_ingredient_choice(opt)
    if step.kind == "bias":
        return wording.next_bias(f"{abs(step.gap):.1f}", below=step.gap < 0), None
    if step.kind == "misses":
        return wording.NEXT_MISSES, None
    if step.kind == "repeat":
        return disagreement_text(step.disagreement), None
    if step.kind == "early":
        return wording.NEXT_TOO_EARLY, None
    if step.kind == "untested":
        return wording.next_untested(values), None
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
        return wording.NEXT_RISING, None
    return wording.NEXT_SEE, None


def _next_step(opt, step):
    """Answer 3: one sentence. Returns the Set up button's label, or None."""
    text, change = next_step_text(opt, step)
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


def mixture_frame(opt, mix, preds):
    """One row per grid point: its place on the page, what it is (tested
    near here, not tested near here, not allowed) and its tooltip."""
    unit = mix.units[0] if len(set(mix.units)) == 1 else ""
    cols = [opt._amount_column(n) for n in mix.names]
    tested, untested, blocked = _mixture_kinds(opt)
    rows = []
    for p, pred in zip(mix.points, preds or [None] * len(mix.points)):
        X, Y = _xy(p.x)
        state = blocked if p.broken else tested if p.tested else untested
        row = {"X": X, "Y": Y, "state": state,
               "score": None if pred is None else pred.mean,
               "a0": round(p.amounts[0], 2), "a1": round(p.amounts[1], 2),
               "a2": round(p.amounts[2], 2), "sum": round(sum(p.amounts), 2),
               "pred": "" if pred is None else score_text(pred),
               "why": p.broken or ""}
        rows.append(row)
    return pd.DataFrame(rows), cols, unit


def mixture_chart(opt, mix, preds, number, start_pred=None):
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
    df, cols, unit = mixture_frame(opt, mix, preds)
    kinds = _mixture_kinds(opt)
    formulation = _say(opt, wording.FORMULATION_CAP)
    predicted = _say(opt, wording.PREDICTED_SCORE_COLUMN)
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


def _start_number(opt):
    """Section 2's Start from, else Best so far, else the newest."""
    ids = [int(x) for x in opt.formulation_ids]
    chosen = st.session_state.get(START_KEY)
    if chosen in ids:
        return chosen
    best = opt.best_index()
    return ids[best] if best is not None else ids[-1]


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


def _mixture_cached(opt, fitted, names, number, start):
    key = (insights.setup_key(opt), tuple(names), number)
    hit = st.session_state.get(MIXTURE_KEY)
    if hit and hit[0] == key:
        return hit[1]
    with st.spinner(_say(opt, wording.INSIGHTS_WORKING)):
        try:
            mix = insights.mixture_slice(opt, names, start)
            value = (mix, insights.mixture_predictions(opt, fitted, mix),
                     _start_prediction(opt, fitted, mix, start))
        except Exception:
            value = _FAILED
    st.session_state[MIXTURE_KEY] = (key, value)
    return value


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
    number = _start_number(opt)
    ids = [int(x) for x in opt.formulation_ids]
    start = insights.start_recipe(opt, ids.index(number))
    _keep_all_offered(THREE_KEY, offered)
    if THREE_KEY not in st.session_state:
        st.session_state[THREE_KEY] = _three_default(opt, fitted, offered, start)
    picked = st.multiselect(_say(opt, wording.MIXTURE_PICK), offered, key=THREE_KEY,
                            max_selections=3)
    if len(picked) != 3:
        st.caption(_say(opt, wording.PICK_THREE))
        return True
    got = _mixture_cached(opt, fitted, picked, number, start)
    if got is _FAILED:
        st.caption(_say(opt, wording.panel_failed(wording.OVERALL_SCORE_COLUMN)))
        return True
    mix, preds, own = got
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
    chart = mixture_chart(opt, mix, preds, number, own)
    if chart is not None:
        show_chart(chart, width="content")
    st.caption(_say(opt, wording.mixture_caption(number)))
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
    with st.expander(_say(opt, wording.HOW_WORKED_OUT)):
        tiny = (getattr(opt, "bo_config", None) or {}).get("noise") == "fixed_tiny"
        for text in (wording.HOW_MODELS,
                     wording.HOW_RANGE_FIXED_TINY if tiny else wording.HOW_RANGE,
                     wording.how_checks(values=not opt.has_ingredients()),
                     wording.how_same_process(values=not opt.has_ingredients()),
                     wording.HOW_NO_P):
            st.markdown(_say(opt, text))
        if computed is None:
            return
        checks = computed["checks"]
        st.caption(_say(opt, wording.PREDICTIONS_CAPTION))
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
                                                    len(h.rounds))))
        columns = wording.DETAILS_COLUMNS_2
        st.dataframe(pd.DataFrame([{
            columns[0]: panel_label(opt, r.name),
            columns[1]: r.checked,
            columns[2]: r.inside,
            columns[3]: f"± {_num(r.half_width)}",
            columns[4]: _num(r.typical_miss),
            columns[5]: "" if r.widen is None else wording.widen_text(r.widen),
            columns[6]: wording.repeat_compare(r.ratio, r.repeat_sd, r.pairs),
        } for r in rows]), hide_index=True, height=table_height(len(rows)))
        for text in (wording.HOW_WIDENED, wording.HOW_REPEATS_COMPARED, wording.HOW_THRESHOLDS):
            st.caption(_say(opt, text))
