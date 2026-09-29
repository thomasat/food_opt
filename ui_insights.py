"""The Insights tab. Builders here return Altair charts from computed facts
and never call Streamlit; render() at the bottom is the only Streamlit code."""
import math

import altair as alt
import pandas as pd
import streamlit as st

import insights
import wording
from food_bo import join_unit
from ui_helpers import amount_format, table_height

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
    formulation = _say(opt, wording.FORMULATION_CAP)
    rows = [{
        "panel": panel_label(opt, c.name), "mark": "check",
        formulation: c.formulation,
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
    x = alt.X(f"{formulation}:Q", title=formulation,
              scale=alt.Scale(zero=False, nice=False),
              axis=alt.Axis(format="d", tickMinStep=1))
    tip = [formulation, wording.ROUND_CAP, wording.TRACK_PREDICTED,
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

# The fitted models, keyed by insights.data_key, shared by Make a round (its
# predictions) and this tab, so neither fits them twice. None: the fit
# failed, cached like a success so it is not tried again on every rerun.
FITTED_KEY = "_insights_fitted"
# The Round-by-Round checks, under the same key. They take the models from
# FITTED_KEY and add only the checks.
CACHE_KEY = "_insights_cache"
# The formulation predicted to score highest, keyed by insights.setup_key: a
# limit edit on Set up changes it without changing a fitted model.
BEST_KEY = "_insights_best"
# All of the above are a project's own: app.py's _reset_project_session drops
# them with the two section-2 controls below.
CHOICE_KEY = "insights_choice"
START_KEY = "insights_start"


def fitted_models(opt, working=wording.INSIGHTS_WORKING):
    """insights.fit_all for this project's data, fitted once per data_key.
    None when the fit failed."""
    key = insights.data_key(opt)
    hit = st.session_state.get(FITTED_KEY)
    if hit and hit[0] == key:
        return hit[1]
    with st.spinner(working):
        try:
            # Through the module attribute, so a test can stand in for the fit.
            value = insights.fit_all(opt)
        except Exception:
            value = None
    st.session_state[FITTED_KEY] = (key, value)
    return value


def _computed(opt):
    key = insights.data_key(opt)
    hit = st.session_state.get(CACHE_KEY)
    if hit and hit[0] == key:
        return hit[1]
    fitted = fitted_models(opt)
    if fitted is None:
        value = _FAILED
    else:
        with st.spinner(wording.INSIGHTS_WORKING):
            try:
                value = {"fitted": fitted, "checks": insights.check_rounds(opt)}
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
    n = len(opt.Y_history)
    if n == 0:
        st.caption(_say(opt, wording.INSIGHTS_NONE))
        return
    order = insights.round_order(opt)
    st.caption(_say(opt, wording.insights_header(n, order[-1][0] if order else None,
                                                 len(opt.waiting_rows()))))
    computed = _computed(opt) if n >= insights.MIN_ROWS else None
    failed = computed is not None and computed.get("failed")
    if failed:
        st.caption(_say(opt, wording.INSIGHTS_FAILED))
        computed = None
    else:
        _predictions(opt, computed, n)
        _change(opt, computed, n)
    _improving(opt)
    # After "Is it still improving?", before "How these insights are made".
    # With the models failed, INSIGHTS_FAILED above already said so once.
    if not failed:
        _best(opt, computed)
        # Wave C: after question 4, before "How these insights are made".
        _more_charts(opt, computed)
    _how(opt, computed)


def _predictions(opt, computed, n):
    _question(opt, wording.Q_PREDICTIONS)
    if computed is None:
        st.caption(_say(opt, wording.insights_gate(n)))
        return
    checks = computed["checks"]
    h = insights.headline(opt, checks)
    if h.state == "none":
        st.write(_say(opt, wording.PREDICTIONS_NONE))
    elif h.state == "early":
        st.write(_say(opt, wording.predictions_early(h.total)))
    else:
        st.write(_say(opt, wording.predictions_count(len(h.rounds), h.rounds, h.inside, h.total)
                      + " " + wording.VERDICT[h.verdict]))
    # Only beside a check that happened: with no checkable Round there are no
    # Overall scores to have come in below anything.
    if h.score_below and h.state in ("early", "verdict"):
        st.write(_say(opt, wording.SCORE_BELOW))
    st.caption(_say(opt, wording.PREDICTIONS_CAPTION))
    # Only once something has been checked: before that, PREDICTIONS_NONE
    # already says when the first check comes.
    skipped = insights.rounds_not_checked(opt)
    if skipped and h.state != "none":
        st.caption(_say(opt, wording.rounds_not_checked(skipped)))
    chart = track_chart(opt, checks)
    if chart is not None:
        with st.expander(_say(opt, wording.SHOW_PREDICTIONS)):
            st.altair_chart(chart)


def _keep_if_offered(key, options):
    """A choice left from before a deletion or a Set up change is dropped
    before its box is drawn, so the box falls back to its first entry."""
    if key in st.session_state and st.session_state[key] not in options:
        del st.session_state[key]


def _change(opt, computed, n):
    _question(opt, wording.q_change(insights.has_setting_choice(opt),
                                    insights.has_ingredient_choice(opt)))
    if computed is None:
        st.caption(_say(opt, wording.insights_gate(n)))
        return
    choices = insights.path_choices(opt)
    if not choices:
        st.caption(_say(opt, wording.CHANGE_NONE))
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
    answer = st.container()
    left, right = st.columns(2)
    number = right.selectbox(
        _say(opt, wording.START_FROM), starts, key=START_KEY,
        format_func=lambda no: _say(opt, wording.start_best(no) if no == best_no
                                    else wording.start_formulation(no)))
    fitted = computed["fitted"]
    try:
        found = insights.spreads(opt, fitted, insights.start_recipe(opt, ids.index(number)))
    except Exception:   # the answer line is left out; the path still draws
        found = insights.Spreads([], [])
    # The box opens on the largest spread; after that it keeps the reader's
    # own choice (_keep_if_offered drops it only when it is no longer offered).
    if found.moved and CHOICE_KEY not in st.session_state:
        st.session_state[CHOICE_KEY] = found.moved[0].name
    with answer:
        _spread_answer(opt, found, number == best_no, number)
    name = left.selectbox(_say(opt, wording.INGREDIENT_OR_SETTING), choices, key=CHOICE_KEY)
    try:
        path = insights.build_path(opt, name, insights.start_recipe(opt, ids.index(number)))
        if path.start_breaks:
            st.caption(_say(opt, wording.path_start_breaks(number, path.start_breaks)))
            return
        chart = path_chart(opt, path, insights.path_predictions(opt, fitted, path))
        lines = _path_captions(opt, path, number, fitted)
    except Exception:   # one path failing must not take the tab down
        st.caption(_say(opt, wording.panel_failed(name)))
        return
    if chart is not None:
        st.altair_chart(chart)
    for line in lines:
        st.caption(_say(opt, line))


def spread_chart(spreads):
    """Compare all ingredients and settings: one horizontal bar per choice,
    largest spread first, in points of Overall score. No error bars: a
    spread is worked from the predicted means, and the caption says so."""
    if not spreads:
        return None
    df = pd.DataFrame({"name": [s.name for s in spreads],
                       wording.SPREAD_AXIS: [round(s.spread, 1) for s in spreads]})
    order = [s.name for s in spreads]
    return (alt.Chart(df).mark_bar()
            .encode(x=alt.X(f"{wording.SPREAD_AXIS}:Q", title=wording.SPREAD_AXIS),
                    y=alt.Y("name:N", sort=order, title=None),
                    tooltip=["name:N", f"{wording.SPREAD_AXIS}:Q"])
            .properties(height=28 * len(spreads) + 20))


# Under a tenth of a point every spread reads "0.0" or "0.1": no choice
# moves the prediction by more than SPREAD_FLAT's 0.1 points.
_FLAT = 0.1


def _spread_answer(opt, found, from_best, number):
    """The one line above the boxes, a caption naming any choice the limits
    hold in place, and the compare expander. Nothing when no path could be
    drawn or there is no Overall-score model yet."""
    start = wording.spread_from(from_best, number)
    # With no ingredient to change, every choice is a process setting: it
    # has values, not amounts, and the lines name process settings alone.
    settings_only = not insights.has_ingredient_choice(opt)
    moved = found.moved
    if not moved:
        if found.held:
            st.write(_say(opt, wording.held_all(start, settings_only)))
        return
    flat = all(s.spread < _FLAT for s in moved)
    if flat:
        st.write(_say(opt, wording.spread_flat(settings_only)))
    else:
        top = moved[0]
        st.write(_say(opt, wording.spread_line(top.name, f"{top.spread:.1f}", start,
                                               only=len(moved) == 1,
                                               values=insights.is_setting(opt, top.name))))
    if found.held:
        st.caption(_say(opt, wording.held_line(start, found.held)))
    if not flat and len(moved) > 1:
        with st.expander(_say(opt, wording.compare_all(insights.has_setting_choice(opt),
                                                       not settings_only))):
            st.altair_chart(spread_chart(moved))
            st.caption(_say(opt, wording.spread_caption(values=settings_only)))


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


def value_text(p):
    """A measurement prediction, formatted as the rest of the tab."""
    return f"{_num(p.mean)} ({range_text(p.low, p.high)})"


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
        lines.append(wording.path_process(path.name, number, opt.has_ingredients()))
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
        lines.append(wording.path_grey(values=path.mode == "process"))
    return lines


def _improving(opt):
    _question(opt, wording.Q_IMPROVING)
    f = insights.best_facts(opt)
    if f is not None:
        second = (wording.BEST_LATEST if f.latest and f.rounds_since == 0
                  else wording.best_since(f.rounds_since) if f.rounds_since else "")
        st.write(_say(opt, wording.best_line(f"{f.score:.1f}", f.formulation, f.round)
                      + (" " + second if second else "")))
    spread = insights.repeat_spread(opt)
    # A spread that would read 0 says nothing about chance.
    if spread is not None and spread >= 0.05:
        st.write(_say(opt, wording.repeats_line(_num(spread))))
    with st.expander(_say(opt, wording.SHOW_PROGRESS)):
        _progress_chart_body(opt)


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


def _progress_chart_body(opt):
    # Moved unchanged from Results (0.7.2), minus its own expander.
    st.line_chart(progress_frame(opt), height=220,
        # Both axes named. Unlabelled, the x axis read 1.0 / 1.5 / 2.0
        # with one round of data and nothing on the chart said what
        # either number was.
        x_label=_say(opt, wording.FORMULATION_CAP),
        y_label=wording.OVERALL_SCORE_COLUMN)
    st.caption(_say(opt, wording.PROGRESS_CHART_CAPTION))


def _predicted_best(opt, fitted):
    """insights.predicted_best, once per setup_key. A failure is cached as
    such and shows one line."""
    key = insights.setup_key(opt)
    hit = st.session_state.get(BEST_KEY)
    if hit and hit[0] == key:
        return hit[1]
    with st.spinner(wording.INSIGHTS_WORKING):
        try:
            value = insights.predicted_best(opt, fitted)
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
    [own] = insights.predict(opt, fitted[insights.SCORE], [opt._encode(other)])
    frame[wording.PREDICTED_SCORE_COLUMN] = [score_text(best.score), score_text(own)]
    return frame.rename(columns={wording.FORMULATION_CAP: _say(opt, wording.FORMULATION_CAP)})


def _best(opt, computed):
    _question(opt, wording.Q_BEST)
    fitted = None if computed is None else computed["fitted"]
    score = None if fitted is None else fitted.get(insights.SCORE)
    if (opt.complete_count() < insights.MIN_ROWS or score is None
            or score.model is None):
        st.caption(_say(opt, wording.insights_gate(opt.complete_count())))
        return
    best = _predicted_best(opt, fitted)
    if isinstance(best, str):
        st.caption(_say(opt, best))
        return
    if best is _FAILED:
        st.caption(_say(opt, wording.BEST_FAILED))
        return
    p = best.score
    line = wording.best_answer(_whole(p.mean), _whole(p.low), _whole(p.high))
    if best.close is not None:
        line += " " + wording.best_close(int(opt.formulation_ids[best.close]),
                                         f"{opt.Y_history[best.close]:.1f}")
    else:
        line += " " + wording.BEST_UNTESTED
    st.write(_say(opt, line))
    frame = best_frame(opt, best, fitted)
    st.dataframe(frame.style.format(amount_format(opt, frame)), hide_index=True,
                 column_config={wording.PREDICTED_SCORE_COLUMN: st.column_config.Column(
                     wording.PREDICTED_SCORE_COLUMN,
                     help=_say(opt, wording.PREDICTED_SCORE_HELP))},
                 height=table_height(len(frame)))
    st.caption(_say(opt, wording.BEST_CAPTION))


# ------------------------------------------------------------------ #
#  Wave C: More charts
# ------------------------------------------------------------------ #

# The two multiselects (a project's own rows and measurements) and the
# triangle's cached grid, keyed by setup_key, the three names and the start.
PAIR_KEY = "insights_pair"
THREE_KEY = "insights_three"
MIXTURE_KEY = "_insights_mixture"

_TRI_H = math.sqrt(3) / 2
_MIX_SIZE = 380          # px, width and height: the triangle keeps its shape

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
    xt = wording.pair_axis(opt._measurement_column(oa), oa.get('goal'), oa.get('target'))
    yt = wording.pair_axis(opt._measurement_column(ob), ob.get('goal'), ob.get('target'))

    def domain(o, values):
        return [min(float(o['min_val']), *values), max(float(o['max_val']), *values)]

    df = pd.DataFrame([{"x": p.a, "y": p.b, "no": p.formulation,
                        "score": None if p.score is None else round(p.score, 1),
                        "waiting": p.score is None, "best": p.best} for p in points])
    x = alt.X("x:Q", title=xt, scale=alt.Scale(domain=domain(oa, list(df["x"])), nice=False))
    y = alt.Y("y:Q", title=yt, scale=alt.Scale(domain=domain(ob, list(df["y"])), nice=False))
    tip = [alt.Tooltip("no:Q", title=wording.FORMULATION_CAP),
           alt.Tooltip("x:Q", title=opt._measurement_column(oa)),
           alt.Tooltip("y:Q", title=opt._measurement_column(ob)),
           alt.Tooltip("score:Q", title=wording.OVERALL_SCORE_COLUMN)]
    base = alt.Chart(df)
    layers = []
    for o, enc in ((oa, "x"), (ob, "y")):
        if o.get('goal') == 'target' and o.get('target') is not None:
            layers.append(alt.Chart(pd.DataFrame({"t": [float(o['target'])]}))
                          .mark_rule(strokeDash=[4, 3], strokeWidth=1, color="gray")
                          .encode(**{enc: alt.X("t:Q") if enc == "x" else alt.Y("t:Q")}))
    layers.append(base.mark_point(filled=False, color="gray", size=60)
                  .encode(x=x, y=y, tooltip=tip)
                  .transform_filter(alt.FieldEqualPredicate(field="waiting", equal=True)))
    layers.append(base.mark_circle(size=70, opacity=1)
                  .encode(x=x, y=y, tooltip=tip,
                          color=alt.Color("score:Q", title=wording.OVERALL_SCORE_COLUMN,
                                          scale=alt.Scale(scheme=_BLUES,
                                                          domain=[0, opt.utility_ceiling()])))
                  .transform_filter(alt.FieldEqualPredicate(field="waiting", equal=False)))
    best = base.transform_filter(alt.FieldEqualPredicate(field="best", equal=True))
    layers.append(best.mark_point(filled=False, size=260, strokeWidth=2, color="gray")
                  .encode(x=x, y=y, tooltip=tip))
    layers.append(best.mark_text(align="left", dx=12, color="gray")
                  .encode(x=x, y=y, text=alt.value(wording.PAIR_BEST)))
    return alt.layer(*layers).properties(height=320)


def mixture_frame(opt, mix, preds):
    """One row per grid point: its place on the page, what it is (tested
    near here, not tested near here, not allowed) and its tooltip."""
    unit = mix.units[0] if len(set(mix.units)) == 1 else ""
    cols = [opt._amount_column(n) for n in mix.names]
    rows = []
    for p, pred in zip(mix.points, preds or [None] * len(mix.points)):
        X, Y = _xy(p.x)
        state = (wording.MIXTURE_BLOCKED if p.broken else
                wording.MIXTURE_TESTED if p.tested else wording.MIXTURE_UNTESTED)
        row = {"X": X, "Y": Y, "state": state,
               "score": None if pred is None else pred.mean,
               "a0": round(p.amounts[0], 2), "a1": round(p.amounts[1], 2),
               "a2": round(p.amounts[2], 2), "sum": round(sum(p.amounts), 2),
               "pred": "" if pred is None else score_text(pred),
               "why": p.broken or ""}
        rows.append(row)
    return pd.DataFrame(rows), cols, unit


def mixture_chart(opt, mix, preds, number):
    """What happens when three ingredients trade places? A triangle of
    predicted Overall scores: one hue, darker is higher, over the allowed
    points only; a smaller mark where nothing was tested near; light grey
    where a limit is broken. Tested formulations are white dots and the
    start is a grey ring named by its number. No axes: the corners carry
    the names and the most of each ingredient allowed."""
    if not mix.points or not preds:
        return None
    df, cols, unit = mixture_frame(opt, mix, preds)
    allowed = df[df["state"] != wording.MIXTURE_BLOCKED]["score"]
    lo, hi = (float(allowed.min()), float(allowed.max())) if len(allowed) else (0.0, 1.0)
    pad = 0.12
    x = alt.X("X:Q", axis=None, scale=alt.Scale(domain=[-pad, 1 + pad], nice=False, zero=False))
    y = alt.Y("Y:Q", axis=None, scale=alt.Scale(domain=[-pad - 0.07, 1 + pad - 0.07],
                                                nice=False, zero=False))
    kinds = [wording.MIXTURE_TESTED, wording.MIXTURE_UNTESTED, wording.MIXTURE_BLOCKED]
    # An explicit ordinal type: without it Vega-Lite compiles a "point"
    # scale for a size range, which only honours the range's start and
    # stop and interpolates the middle entry instead of using its own
    # value (16 draws as ~43).
    size_scale = alt.Scale(domain=kinds, range=[70, 16, 16], type="ordinal")
    # A size-only channel, shared by both layers: the smaller mark says
    # "not tested near here" without fading a colour that already means a
    # score. Its legend swatches get their fill from state_legend below —
    # Vega-Lite merges a size legend and a colour legend that share the
    # same field into one, so the entries read larger blue, small blue,
    # small grey instead of one flat grey circle.
    size = alt.Size("state:N", title=None, sort=kinds, scale=size_scale)
    amounts = ([alt.Tooltip(f"a{k}:Q", title=cols[k]) for k in range(3)]
               + [alt.Tooltip("sum:Q", title=wording.mixture_together(unit))])
    blocked = alt.FieldEqualPredicate(field="state", equal=wording.MIXTURE_BLOCKED)
    grey = (alt.Chart(df).mark_circle(opacity=1, color="lightgray")
            .encode(x=x, y=y, size=size,
                    tooltip=amounts + [alt.Tooltip("why:N", title=wording.MIXTURE_BLOCKED)])
            .transform_filter(blocked))
    grid = (alt.Chart(df).mark_circle(opacity=1)
            .encode(x=x, y=y, size=size,
                    color=alt.Color("score:Q", title=wording.PREDICTED_SCORE_COLUMN,
                                    scale=alt.Scale(scheme=_BLUES,
                                                    domain=[lo, hi])),
                    tooltip=amounts + [alt.Tooltip("pred:N", title=wording.PREDICTED_SCORE_COLUMN)])
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
    corners = [(0.0, 0.0), (1.0, 0.0), (0.5, _TRI_H)]
    outline = (alt.Chart(pd.DataFrame([{"X": cx, "Y": cy, "k": k}
                                       for k, (cx, cy) in enumerate(corners + corners[:1])]))
               .mark_line(color="gray", strokeWidth=1, tooltip=False)
               .encode(x=x, y=y, order="k:Q"))
    labels = pd.DataFrame([
        {"X": 0.0, "Y": -0.06, "t": wording.mixture_corner(
            mix.names[0], _amount(mix.upper[0], mix.units[0])), "align": "left"},
        {"X": 1.0, "Y": -0.06, "t": wording.mixture_corner(
            mix.names[1], _amount(mix.upper[1], mix.units[1])), "align": "right"},
        {"X": 0.5, "Y": _TRI_H + 0.05, "t": wording.mixture_corner(
            mix.names[2], _amount(mix.upper[2], mix.units[2])), "align": "center"}])
    text = [alt.Chart(labels[labels["align"] == al])
            .mark_text(align=al, baseline="middle", color="gray", tooltip=False)
            .encode(x=x, y=y, text="t:N") for al in ("left", "right", "center")]
    layers = [outline, grey, grid, state_legend]
    if mix.dots:
        dots = pd.DataFrame([{**dict(zip(("X", "Y"), _xy(d.x))), "no": d.formulation}
                             for d in mix.dots])
        layers.append(alt.Chart(dots).mark_point(filled=True, fill="white", stroke="gray",
                                                 strokeWidth=1.5, size=45, opacity=1)
                      .encode(x=x, y=y, tooltip=[alt.Tooltip(
                          "no:Q", title=wording.FORMULATION_CAP)]))
    s_amounts = [float(mix.start.get(n, 0.0)) for n in mix.names]
    s = [(s_amounts[k] - mix.lower[k]) / mix.span for k in range(3)]
    sx, sy = _xy(s)
    start = pd.DataFrame([{
        "X": sx, "Y": sy, "t": wording.start_formulation(number), "no": number,
        "a0": round(s_amounts[0], 2), "a1": round(s_amounts[1], 2), "a2": round(s_amounts[2], 2),
        "sum": round(sum(s_amounts), 2)}])
    layers.append(alt.Chart(start).mark_point(filled=False, size=260, strokeWidth=2.5,
                                              color="gray")
                  .encode(x=x, y=y,
                          tooltip=[alt.Tooltip("no:Q", title=wording.FORMULATION_CAP)] + amounts))
    layers.append(alt.Chart(start).mark_text(align="left", dx=12, color="gray", tooltip=False)
                  .encode(x=x, y=y, text="t:N"))
    return (alt.layer(*layers, *text).properties(
                width=_MIX_SIZE, height=_MIX_SIZE,
                autosize=alt.AutoSizeParams(type="pad", contains="padding"))
            .configure_view(stroke=None))


def _pair(opt):
    names = insights.pair_choices(opt)
    if len(names) < 2:
        return False
    _question(opt, wording.Q_PAIR)
    _keep_all_offered(PAIR_KEY, names)
    if PAIR_KEY not in st.session_state:
        st.session_state[PAIR_KEY] = names[:2]
    picked = st.multiselect(wording.PAIR_PICK, names, key=PAIR_KEY, max_selections=2)
    if len(picked) != 2:
        st.caption(wording.PICK_TWO)
        return True
    points = insights.pair_points(opt, *picked)
    if not points:
        st.caption(wording.PAIR_NONE)
        return True
    st.altair_chart(pair_chart(opt, *picked, points))
    st.caption(wording.PAIR_CAPTION)
    return True


def _start_number(opt):
    """Section 2's Start from, else Best so far, else the newest."""
    ids = [int(x) for x in opt.formulation_ids]
    chosen = st.session_state.get(START_KEY)
    if chosen in ids:
        return chosen
    best = opt.best_index()
    return ids[best] if best is not None else ids[-1]


def _three_default(opt, fitted, offered, start):
    try:
        ranked = [s.name for s in insights.spreads(opt, fitted, start) if s.name in offered]
    except Exception:
        ranked = []
    out = ranked[:3]
    return out + [n for n in offered if n not in out][:3 - len(out)]


def _mixture_cached(opt, fitted, names, number, start):
    key = (insights.setup_key(opt), tuple(names), number)
    hit = st.session_state.get(MIXTURE_KEY)
    if hit and hit[0] == key:
        return hit[1]
    with st.spinner(wording.INSIGHTS_WORKING):
        try:
            mix = insights.mixture_slice(opt, names, start)
            value = (mix, insights.mixture_predictions(opt, fitted, mix))
        except Exception:
            value = _FAILED
    st.session_state[MIXTURE_KEY] = (key, value)
    return value


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
    picked = st.multiselect(wording.MIXTURE_PICK, offered, key=THREE_KEY, max_selections=3)
    if len(picked) != 3:
        st.caption(wording.PICK_THREE)
        return True
    got = _mixture_cached(opt, fitted, picked, number, start)
    if got is _FAILED:
        st.caption(wording.panel_failed(wording.OVERALL_SCORE_COLUMN))
        return True
    mix, preds = got
    if mix.refusal == insights.NO_ROOM:
        st.caption(wording.mixture_no_room(number))
        return True
    if mix.refusal == insights.FILL_FOLLOWS:
        st.caption(wording.mixture_fill_follows(mix.follows[0], opt.balance_row_name()))
        return True
    if mix.start_breaks:
        st.caption(wording.path_start_breaks(number, mix.start_breaks))
        return True
    chart = mixture_chart(opt, mix, preds, number)
    if chart is not None:
        st.altair_chart(chart, width="content")
    st.caption(wording.mixture_caption(number))
    for line in ([wording.path_follows(n) for n in mix.follows]
                 + ([wording.mixture_fills(mix.fill_moves)] if mix.fill_moves else [])
                 + [wording.path_not_checked(t, prop) for t, prop in mix.skipped]):
        st.caption(line)
    return True


def more_charts_offered(opt, computed):
    """Whether "More charts" has anything to show: 5 formulations with all
    their results, the models fitted, and at least one of its charts past
    its own gate."""
    if computed is None or opt.complete_count() < insights.MIN_ROWS:
        return False
    fitted = computed["fitted"]
    score = fitted.get(insights.SCORE)
    return (len(opt.objectives) >= 2
            or (len(insights.mixture_choices(opt)) >= 3 and score is not None
                and score.model is not None))


def _more_charts(opt, computed):
    if not more_charts_offered(opt, computed):
        return
    with st.expander(wording.MORE_CHARTS):
        for draw in (lambda: _pair(opt),
                     lambda: _mixture(opt, computed["fitted"])):
            try:
                draw()
            except Exception:   # one chart failing must not take the tab down
                st.caption(wording.CHART_FAILED)


def _how(opt, computed):
    with st.expander(_say(opt, wording.HOW_WORKED_OUT)):
        tiny = (getattr(opt, "bo_config", None) or {}).get("noise") == "fixed_tiny"
        for text in (wording.HOW_MODELS,
                     wording.HOW_RANGE_FIXED_TINY if tiny else wording.HOW_RANGE,
                     wording.HOW_CHECKS, wording.HOW_NO_P):
            st.markdown(_say(opt, text))
        if computed is None:
            return
        rows = insights.details(opt, computed["checks"])
        if not rows:
            st.caption(_say(opt, wording.DETAILS_NONE))
        else:
            st.dataframe(pd.DataFrame([{
                wording.DETAILS_COLUMNS[0]: panel_label(opt, r.name),
                wording.DETAILS_COLUMNS[1]: r.checked,
                wording.DETAILS_COLUMNS[2]: r.inside,
                wording.DETAILS_COLUMNS[3]: f"± {_num(r.half_width)}",
                wording.DETAILS_COLUMNS[4]: _num(r.typical_miss),
                wording.DETAILS_COLUMNS[5]: "—" if r.r2 is None else f"{r.r2:.2f}",
            } for r in rows]), hide_index=True, height=table_height(len(rows)))
