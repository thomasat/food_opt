"""The Insights tab. Builders here return Altair charts from computed facts
and never call Streamlit; render() at the bottom is the only Streamlit code."""
import math

import altair as alt
import pandas as pd
import streamlit as st

import insights
import wording
from food_bo import join_unit
from ui_helpers import table_height

_BAND = 0.25
_GREY = 0.12


def panel_label(opt, name):
    if name == insights.SCORE:
        return wording.OVERALL_SCORE_COLUMN
    obj = next(o for o in opt.objectives if o['name'] == name)
    return opt._measurement_column(obj)


def track_panels(opt):
    return [o['name'] for o in opt.objectives] + [insights.SCORE]


def path_panels(opt):
    return [insights.SCORE] + [o['name'] for o in opt.objectives]


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
    return (alt.Chart().mark_rule(strokeDash=[4, 3], strokeWidth=1, color="gray", opacity=0.6)
            .encode(y="target:Q").transform_filter(_only("target")))


def _faceted(layers, df, order):
    # Fixed panel size, three to a row: fits beside the sidebar however many
    # formulations there are (the x axis squeezes, the page does not widen).
    return (alt.layer(*layers, data=df).properties(width=220, height=160)
            .facet(facet=alt.Facet("panel:N", sort=order, title=None), columns=3)
            .resolve_scale(y="independent"))


def track_chart(opt, checks):
    """Did the predictions come true? — one panel per measurement plus the
    Overall score: a thin likely-range bar (no end caps) and a dot for the
    measured value, filled when inside, hollow when outside."""
    if not checks:
        return None
    names = track_panels(opt)
    order = [panel_label(opt, n) for n in names]
    rows = [{
        "panel": panel_label(opt, c.name), "mark": "check",
        wording.FORMULATION_CAP: c.formulation,
        wording.ROUND_CAP: c.round,
        wording.TRACK_PREDICTED: round(c.predicted, 2),
        wording.TRACK_LOW: round(c.low, 2),
        wording.TRACK_HIGH: round(c.high, 2),
        wording.TRACK_MEASURED: round(c.measured, 2),
        wording.TRACK_INSIDE: c.inside,
    } for c in checks]
    rows += _target_rows(opt, {c.name for c in checks})   # spec §3.1
    df = pd.DataFrame(rows)
    # Quantitative, not one band per formulation: 100 formulations fit the
    # same 220 px as 10.
    x = alt.X(f"{wording.FORMULATION_CAP}:Q", title=wording.FORMULATION_CAP,
              scale=alt.Scale(zero=False, nice=False),
              axis=alt.Axis(format="d", tickMinStep=1))
    tip = [wording.FORMULATION_CAP, wording.ROUND_CAP, wording.TRACK_PREDICTED,
           wording.TRACK_LOW, wording.TRACK_HIGH, wording.TRACK_MEASURED]
    bars = (alt.Chart().mark_rule(strokeWidth=3, opacity=0.35).encode(
        x=x, y=alt.Y(f"{wording.TRACK_LOW}:Q", title=None), y2=f"{wording.TRACK_HIGH}:Q",
        tooltip=tip).transform_filter(_only("check")))
    # Every measured value gets a ring in the theme's own mark colour; the
    # ones inside their likely range get a filled dot of the same colour on
    # top. A single layer filled with "currentColor" painted the inside dots
    # in the TEXT colour (near-black / near-white) inside a blue ring.
    y = alt.Y(f"{wording.TRACK_MEASURED}:Q")
    rings = (alt.Chart().mark_point(size=60, strokeWidth=2).encode(x=x, y=y, tooltip=tip)
             .transform_filter(_only("check")))
    filled = (alt.Chart().mark_point(size=60, filled=True, opacity=1)
              .encode(x=x, y=y, tooltip=tip)
              .transform_filter(_only("check"))
              .transform_filter(alt.FieldEqualPredicate(field=wording.TRACK_INSIDE, equal=True)))
    layers = [bars, rings, filled]
    if any(r["mark"] == "target" for r in rows):
        layers.insert(0, _target_layer())
    return _faceted(layers, df, order)


def path_chart(opt, path, preds):
    """What happens if you change one ingredient? — one panel per output: a
    neutral predicted line, a light 95 % likely-range band, grey shading
    where the amount was not tested, a dashed start marker."""
    if not path.amounts or not preds:
        return None
    names = [n for n in path_panels(opt) if n in preds]
    x_title = f"{path.name} ({path.unit})" if path.unit else path.name
    rows = []
    for n in names:
        label = panel_label(opt, n)
        for t, p in zip(path.amounts, preds[n]):
            rows.append({"panel": label, "mark": "curve", "x": t, "mean": p.mean,
                         "low": p.low, "high": p.high})
        rows.append({"panel": label, "mark": "start", "x": path.start_amount})
        if path.tested:
            lo, hi = path.amounts[0], path.amounts[-1]
            if lo < path.tested[0]:
                rows.append({"panel": label, "mark": "grey", "x": lo, "x2": path.tested[0]})
            if hi > path.tested[1]:
                rows.append({"panel": label, "mark": "grey", "x": path.tested[1], "x2": hi})
    targets = _target_rows(opt, names)
    df = pd.DataFrame(rows + targets)
    # Not from 0: the path runs over this row's own Lowest-Highest, and a
    # zero-based axis squeezed a 4-6 g path into the right third of each panel.
    x = alt.X("x:Q", title=x_title, scale=alt.Scale(zero=False, nice=False))
    band = (alt.Chart().mark_area(opacity=_BAND)
            .encode(x=x, y=alt.Y("low:Q", title=None), y2="high:Q")
            .transform_filter(_only("curve")))
    line = alt.Chart().mark_line().encode(x=x, y="mean:Q").transform_filter(_only("curve"))
    start = (alt.Chart().mark_rule(strokeDash=[2, 2]).encode(x="x:Q")
             .transform_filter(_only("start")))
    layers = [band, line, start]
    if any(r["mark"] == "grey" for r in rows):
        layers.insert(0, alt.Chart().mark_rect(opacity=_GREY, color="gray")
                      .encode(x="x:Q", x2="x2:Q").transform_filter(_only("grey")))
    if targets:
        layers.append(_target_layer())
    return _faceted(layers, df, [panel_label(opt, n) for n in names])


# ------------------------------------------------------------------ #
#  render: the only Streamlit code in this module
# ------------------------------------------------------------------ #

# The fitted models and the Round-by-Round checks, keyed by insights.data_key.
# A project's own: app.py's _reset_project_session drops it with the two
# section-2 controls below.
CACHE_KEY = "_insights_cache"
CHOICE_KEY = "insights_choice"
START_KEY = "insights_start"


def _computed(opt):
    key = insights.data_key(opt)
    hit = st.session_state.get(CACHE_KEY)
    if hit and hit[0] == key:
        return hit[1]
    with st.spinner(wording.INSIGHTS_WORKING):
        try:
            # Through the module attribute, so a test can stand in for the fit.
            value = {"fitted": insights.fit_all(opt), "checks": insights.check_rounds(opt)}
        except Exception:
            # Cached like a success, under the same key: one line on the tab
            # instead of a traceback, and not tried again on every rerun.
            value = _FAILED
    st.session_state[CACHE_KEY] = (key, value)
    return value


_FAILED = {"failed": True}


def _question(text):
    st.markdown(f"**{text}**")


def render(opt):
    n = len(opt.Y_history)
    if n == 0:
        st.caption(wording.INSIGHTS_NONE)
        return
    order = insights.round_order(opt)
    st.caption(wording.insights_header(n, order[-1][0] if order else None))
    computed = _computed(opt) if n >= insights.MIN_ROWS else None
    if computed is not None and computed.get("failed"):
        st.caption(wording.INSIGHTS_FAILED)
        computed = None
    else:
        _predictions(opt, computed, n)
        _change(opt, computed, n)
    _improving(opt)
    _how(opt, computed)


def _predictions(opt, computed, n):
    _question(wording.Q_PREDICTIONS)
    if computed is None:
        st.caption(wording.insights_gate(n))
        return
    checks = computed["checks"]
    h = insights.headline(opt, checks)
    if h.state == "none":
        st.write(wording.PREDICTIONS_NONE)
    elif h.state == "early":
        st.write(wording.predictions_early(h.total))
    else:
        st.write(wording.predictions_count(len(h.rounds), h.rounds, h.inside, h.total)
                 + " " + wording.VERDICT[h.verdict])
    # Only beside a check that happened: with no checkable Round there are no
    # Overall scores to have come in below anything.
    if h.score_below and h.state in ("early", "verdict"):
        st.write(wording.SCORE_BELOW)
    st.caption(wording.PREDICTIONS_CAPTION)
    # Only once something has been checked: before that, PREDICTIONS_NONE
    # already says when the first check comes.
    skipped = insights.rounds_not_checked(opt)
    if skipped and h.state != "none":
        st.caption(wording.rounds_not_checked(skipped))
    chart = track_chart(opt, checks)
    if chart is not None:
        with st.expander(wording.SHOW_PREDICTIONS):
            st.altair_chart(chart)


def _keep_if_offered(key, options):
    """A choice left from before a deletion or a Set up change is dropped
    before its box is drawn, so the box falls back to its first entry."""
    if key in st.session_state and st.session_state[key] not in options:
        del st.session_state[key]


def _change(opt, computed, n):
    _question(wording.q_change(insights.has_setting_choice(opt)))
    if computed is None:
        st.caption(wording.insights_gate(n))
        return
    choices = insights.path_choices(opt)
    if not choices:
        st.caption(wording.CHANGE_NONE)
        return
    # None while no formulation has all its results: then there is no best
    # to start from, and the box offers the newest first.
    best = opt.best_index()
    ids = [int(x) for x in opt.formulation_ids]
    best_no = None if best is None else ids[best]
    # By Formulation number, not list position: a deletion on Results shifts
    # positions, and the box must keep naming the formulation it named.
    starts = (([] if best is None else [best_no])
              + [ids[i] for i in reversed(range(len(ids))) if i != best])
    _keep_if_offered(CHOICE_KEY, choices)
    _keep_if_offered(START_KEY, starts)
    left, right = st.columns(2)
    name = left.selectbox(wording.INGREDIENT_OR_SETTING, choices, key=CHOICE_KEY)
    number = right.selectbox(
        wording.START_FROM, starts, key=START_KEY,
        format_func=lambda no: (wording.start_best(no) if no == best_no
                                else wording.start_formulation(no)))
    fitted = computed["fitted"]
    try:
        path = insights.build_path(opt, name, insights.start_recipe(opt, ids.index(number)))
        if path.start_breaks:
            st.caption(wording.path_start_breaks(number, path.start_breaks))
            return
        chart = path_chart(opt, path, insights.path_predictions(opt, fitted, path))
        lines = _path_captions(opt, path, number, fitted)
    except Exception:   # one path failing must not take the tab down
        st.caption(wording.panel_failed(name))
        return
    if chart is not None:
        st.altair_chart(chart)
    for line in lines:
        st.caption(line)


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


def _path_captions(opt, path, number, fitted):
    lines = []
    if path.mode == "fill":
        lines.append(wording.path_fill(path.name, path.fill_name, number))
    elif path.mode == "proportional":
        lines.append(wording.path_proportional(
            path.name, opt.batch_total_text(opt.formulation_total)))
    elif path.mode == "process":
        lines.append(wording.path_process(path.name, number))
    else:
        lines.append(wording.path_free(path.name, number))
    for end in (path.low_end, path.high_end):
        if end.reason:
            lines.append(wording.path_stops(_amount(end.amount, path.unit), end.reason))
    if path.outside:
        lines.append(wording.path_outside(number, _amount(path.start_amount, path.unit)))
    if path.fixed:
        lines.append(wording.path_fixed(", ".join(
            f"{n} {_amount(a, u)}" for n, a, u in path.fixed)))
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
    if path.tested and (path.amounts[0] < path.tested[0] or path.amounts[-1] > path.tested[1]):
        lines.append(wording.PATH_GREY)
    return lines


def _improving(opt):
    _question(wording.Q_IMPROVING)
    f = insights.best_facts(opt)
    if f is not None:
        second = (wording.BEST_LATEST if f.latest and f.rounds_since == 0
                  else wording.best_since(f.rounds_since) if f.rounds_since else "")
        st.write(wording.best_line(f"{f.score:.1f}", f.formulation, f.round)
                 + (" " + second if second else ""))
    spread = insights.repeat_spread(opt)
    # A spread that would read 0 says nothing about chance.
    if spread is not None and spread >= 0.05:
        st.write(wording.repeats_line(_num(spread)))
    with st.expander(wording.SHOW_PROGRESS):
        _progress_chart_body(opt)


def _progress_chart_body(opt):
    # Moved unchanged from Results (0.7.2), minus its own expander.
    st.line_chart(pd.DataFrame({
        wording.FORMULATION_CAP: [int(n) for n in opt.formulation_ids],
        wording.OVERALL_SCORE_COLUMN: [float(y) for y in opt.Y_history],
        wording.BEST_SO_FAR_COLUMN: opt.best_so_far(),
    }).set_index(wording.FORMULATION_CAP), height=220,
        # Both axes named. Unlabelled, the x axis read 1.0 / 1.5 / 2.0
        # with one round of data and nothing on the chart said what
        # either number was.
        x_label=wording.FORMULATION_CAP,
        y_label=wording.OVERALL_SCORE_COLUMN)
    st.caption(wording.PROGRESS_CHART_CAPTION)


def _how(opt, computed):
    with st.expander(wording.HOW_WORKED_OUT):
        tiny = (getattr(opt, "bo_config", None) or {}).get("noise") == "fixed_tiny"
        for text in (wording.HOW_MODELS,
                     wording.HOW_RANGE_FIXED_TINY if tiny else wording.HOW_RANGE,
                     wording.HOW_CHECKS, wording.HOW_NO_P):
            st.markdown(text)
        if computed is None:
            return
        rows = insights.details(opt, computed["checks"])
        if not rows:
            st.caption(wording.DETAILS_NONE)
        else:
            st.dataframe(pd.DataFrame([{
                wording.DETAILS_COLUMNS[0]: panel_label(opt, r.name),
                wording.DETAILS_COLUMNS[1]: r.checked,
                wording.DETAILS_COLUMNS[2]: r.inside,
                wording.DETAILS_COLUMNS[3]: f"± {_num(r.half_width)}",
                wording.DETAILS_COLUMNS[4]: _num(r.typical_miss),
                wording.DETAILS_COLUMNS[5]: "—" if r.r2 is None else f"{r.r2:.2f}",
            } for r in rows]), hide_index=True, height=table_height(len(rows)))
