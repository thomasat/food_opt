"""The Insights tab. Builders here return Altair charts from computed facts
and never call Streamlit; render() at the bottom is the only Streamlit code."""
import altair as alt
import pandas as pd

import insights
import wording

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


def _targets(opt, names):
    rows = []
    for o in opt.objectives:
        if o['name'] in names and o.get('goal') == 'target' and o.get('target') is not None:
            rows.append({"panel": panel_label(opt, o['name']), "target": float(o['target'])})
    return pd.DataFrame(rows, columns=["panel", "target"])


def track_chart(opt, checks):
    """Did the predictions come true? — one panel per measurement plus the
    Overall score: a thin likely-range bar (no end caps) and a dot for the
    measured value, filled when inside, hollow when outside."""
    if not checks:
        return None
    order = [panel_label(opt, n) for n in track_panels(opt)]
    df = pd.DataFrame([{
        "panel": panel_label(opt, c.name),
        wording.FORMULATION_CAP: c.formulation,
        wording.ROUND_CAP: c.round,
        wording.TRACK_PREDICTED: round(c.predicted, 2),
        wording.TRACK_LOW: round(c.low, 2),
        wording.TRACK_HIGH: round(c.high, 2),
        wording.TRACK_MEASURED: round(c.measured, 2),
        wording.TRACK_INSIDE: c.inside,
    } for c in checks])
    x = alt.X(f"{wording.FORMULATION_CAP}:O", title=wording.FORMULATION_CAP)
    tip = [wording.FORMULATION_CAP, wording.ROUND_CAP, wording.TRACK_PREDICTED,
           wording.TRACK_LOW, wording.TRACK_HIGH, wording.TRACK_MEASURED]
    bars = alt.Chart().mark_rule(strokeWidth=3, opacity=0.35).encode(
        x=x, y=alt.Y(f"{wording.TRACK_LOW}:Q", title=None), y2=f"{wording.TRACK_HIGH}:Q",
        tooltip=tip)
    dots = alt.Chart().mark_point(size=40).encode(
        x=x, y=f"{wording.TRACK_MEASURED}:Q",
        fill=alt.condition(alt.datum[wording.TRACK_INSIDE], alt.value("currentColor"),
                           alt.value("transparent")),
        tooltip=tip)
    layers = [bars, dots]
    targets = _targets(opt, set(track_panels(opt)))
    if not targets.empty:   # spec §3.1: a light rule at the target
        layers.insert(0, alt.Chart(targets).mark_rule(opacity=0.4).encode(y="target:Q"))
    return (alt.layer(*layers, data=df)
            .facet(facet=alt.Facet("panel:N", sort=order, title=None), columns=3)
            .resolve_scale(y="independent"))


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
        for t, p in zip(path.amounts, preds[n]):
            rows.append({"panel": panel_label(opt, n), "x": t, "mean": p.mean,
                         "low": p.low, "high": p.high})
    df = pd.DataFrame(rows)
    grey = []
    if path.tested:
        lo, hi = path.amounts[0], path.amounts[-1]
        for n in names:
            label = panel_label(opt, n)
            if lo < path.tested[0]:
                grey.append({"panel": label, "x": lo, "x2": path.tested[0]})
            if hi > path.tested[1]:
                grey.append({"panel": label, "x": path.tested[1], "x2": hi})
    x = alt.X("x:Q", title=x_title)
    # `band`/`line` carry no data of their own so they inherit the top-level
    # `data=df` given to alt.layer() below; a faceted layer chart needs data
    # at that top level even when every other layer (start/grey/targets)
    # brings its own smaller frame — Altair only requires the combined
    # LayerChart itself to have a `data` attribute before `.facet()` lifts it.
    band = alt.Chart().mark_area(opacity=_BAND).encode(x=x, y=alt.Y("low:Q", title=None), y2="high:Q")
    line = alt.Chart().mark_line().encode(x=x, y="mean:Q")
    start = alt.Chart(pd.DataFrame([{"panel": panel_label(opt, n), "x": path.start_amount}
                                    for n in names])).mark_rule(strokeDash=[2, 2]).encode(x="x:Q")
    layers = [band, line, start]
    if grey:
        layers.insert(0, alt.Chart(pd.DataFrame(grey)).mark_rect(opacity=_GREY, color="gray")
                      .encode(x="x:Q", x2="x2:Q"))
    targets = _targets(opt, names)
    if not targets.empty:
        layers.append(alt.Chart(targets).mark_rule(opacity=0.5).encode(y="target:Q"))
    order = [panel_label(opt, n) for n in names]
    return (alt.layer(*layers, data=df)
            .facet(facet=alt.Facet("panel:N", sort=order, title=None), columns=3)
            .resolve_scale(y="independent"))
