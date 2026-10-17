"""Plot review (2026-10-02, owner: "I want to make sure the plots are all
interpretable"). Pinned on the chart specs of the Plant patty Line A demo:
every chart has a title (the hero's is its heading), labelled axes, a
legend for every encoded colour, tooltips with a "BO term" row, one
colour per meaning across charts, short captions, and a withheld chart's
reason. One model fit for the module."""
import json
import re
import types

import pytest

import bo_views
import insights
import insights_jobs
import ui_insights as U
import wording
from tests import demo_projects as D

_COLOUR_CHANNELS = ("color", "fill", "stroke", "strokeDash", "shape", "opacity")


@pytest.fixture(scope="module")
def line_a(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.chdir(tmp_path_factory.mktemp("review-plots"))
    mp.setattr(insights_jobs, "_JOBS", {})
    mp.setattr(insights_jobs, "_INFLIGHT", {})
    mp.setattr(U.st, "session_state", {})
    with bo_views.fresh():
        opt = D.load(D.LINE_A)
        computed = U._computed(opt, quiet=True)
        mp.setattr(U, "checks_for", lambda o, stage: computed["checks"])
        yield opt, computed
    mp.undo()


def _specs(opt, computed):
    """{name: spec} for every chart the Review draws (Make a round draws
    none since v3: its map went with the Why fold)."""
    out = {}
    # Review v3: Best so far, Best so far vs targets and Amounts in the top
    # formulations (the sparklines, the forecast and measured-vs-predicted
    # charts are gone).
    view = U.round_view(opt)
    out["improving"] = U.improving_chart(opt)
    out["targets"] = U.targets_chart(opt, view)
    out["amounts"] = U.amounts_chart(opt, view)
    missing = [k for k, v in out.items() if v is None]
    assert not missing, missing
    return {k: v.to_dict() for k, v in out.items()}


@pytest.fixture(scope="module")
def specs(line_a):
    return _specs(*line_a)


def _units(spec):
    """Every view with a mark, through layers, concats and facets."""
    if "mark" in spec:
        yield spec
    for key in ("layer", "concat", "vconcat", "hconcat"):
        for child in spec.get(key, []):
            yield from _units(child)
    if "spec" in spec:
        yield from _units(spec["spec"])


def _words(text):
    return len([w for w in re.sub(r"[^\w%' ]", " ", text).split() if w])


# The Review's charts are titled by the heading drawn above each (the panel's
# **title** lines).
REVIEW = ["improving", "targets", "amounts"]


# Charts whose axes are drawn by design without titles: the sparklines
# (Round labels only), the parallel coordinates (each axis named on top),
# the triangle (corner labels), the change bars' names.
# The strip and the amounts tracks name each row on the y axis and their
# ends in the plot; the amounts tracks have no x axis at all.
_NO_Y_TITLE = {"targets", "amounts"}
_NO_X_TITLE = {"amounts"}


@pytest.mark.parametrize("name", REVIEW)
def test_axes_are_titled(specs, name):
    """Each drawn x and y axis has a title in at least one layer (a rule or
    label layer sharing the axis takes the title Vega-Lite merges in)."""
    for ch, skip in (("x", _NO_X_TITLE), ("y", _NO_Y_TITLE)):
        if name in skip:
            continue
        encs = [u["encoding"][ch] for u in _units(specs[name])
                if ch in u.get("encoding", {}) and "field" in u["encoding"][ch]
                and u["encoding"][ch].get("axis", {}) is not None]
        if not encs:
            continue
        assert any(e.get("title") for e in encs), (name, ch)


@pytest.mark.parametrize("name", REVIEW)
def test_every_encoded_colour_has_a_legend(specs, name):
    """A field on a colour channel is explained by a legend somewhere in the
    chart (a layer may repeat it with legend None, as the Best so far line's
    inner colour does)."""
    shown, used = set(), set()
    for unit in _units(specs[name]):
        for ch in _COLOUR_CHANNELS:
            e = unit.get("encoding", {}).get(ch)
            if not isinstance(e, dict):
                continue
            for part in [e] + ([e["condition"]] if isinstance(e.get("condition"), dict) else []):
                if "field" in part:
                    used.add(part["field"])
                    if part.get("legend", {}) is not None:
                        shown.add(part["field"])
    assert used <= shown, (name, used - shown)


@pytest.mark.parametrize("name", REVIEW)
def test_marks_have_tooltips_with_a_bo_term(specs, name):
    tips = []
    for unit in _units(specs[name]):
        t = unit.get("encoding", {}).get("tooltip")
        if t:
            tips += t if isinstance(t, list) else [t]
    assert tips, name
    # Review v3: the score chart keeps the expert row; the strip and the
    # amounts are measurements and limits only, their numbers in the tooltip.
    if name == "improving":
        assert any(t.get("title") == wording.PLOT_BO for t in tips), name


def _colours(spec):
    found = set()

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in ("color", "stroke", "fill") and isinstance(v, str):
                    found.add(v.lower())
                elif k == "value" and isinstance(v, str) and v.startswith("#"):
                    found.add(v.lower())
                elif k == "range" and isinstance(v, list):
                    found.update(c.lower() for c in v if isinstance(c, str))
                else:
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(spec)
    return found


def test_one_colour_per_meaning_across_charts(specs):
    ink = U.mark("bar_latest")[0].lower()
    above, below = (U.direction_colour(k).lower() for k in ("above", "below"))
    blue = U.model_blue().lower()
    # Best so far: the ink line; its own dots and labels blue (v3).
    assert ink in _colours(specs["improving"]) and blue in _colours(specs["improving"])
    # Off target shows its direction on the strip (v3 final cleanup):
    # orange above its target, purple below; the old single orange is gone.
    assert {above, below} & _colours(specs["targets"])
    assert U.outline_colour().lower() not in _colours(specs["targets"])
    # The strip: blue only on target (this best is on target nowhere, and
    # its key lists only what is drawn); the amounts: no direction colour.
    assert above not in _colours(specs["amounts"]) and below not in _colours(specs["amounts"])
    # No default palette: every colour channel with a field has its range.
    for name, spec in specs.items():
        for unit in _units(spec):
            for ch in ("color", "fill", "stroke"):
                e = unit.get("encoding", {}).get(ch)
                if isinstance(e, dict) and e.get("type") == "nominal" and "field" in e:
                    assert "range" in e.get("scale", {}), (name, ch, e)


def test_short_captions_and_withheld_reasons():
    for text in (wording.PLOT_CAPTION_PROGRESS, wording.PLOT_CAPTION_GAIN,
                 wording.PLOT_CAPTION_MVP, wording.PLOT_CAPTION_CHANGE,
                 wording.PLOT_CAPTION_PATH, wording.PLOT_CAPTION_MAP,
                 wording.PLOT_CAPTION_PAIR, wording.PLOT_CAPTION_PAIR_NONE,
                 wording.HERO_CAPTION,
                 wording.tried_caption()):
        assert _words(text) <= 8, text
    for text in (wording.PLOT_SPARK_WAIT, wording.PLOT_GAIN_WITHHELD,
                 *wording.HERO_HIDDEN.values()):
        assert _words(text) <= 10, text


# Review critique (fix 5): the parallel coordinates' end labels went with the chart.


