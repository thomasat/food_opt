"""0.8.0 speed and simplicity, Plan B, Task 5: the Insights charts —
measured against predicted, Compare all on the tested-near change, the
Progress chart's axes and caption, and one door to the cached checks (spec
2026-09-30 §2.2, §2.3, §2.5; review I8, M4)."""
from types import SimpleNamespace

import pytest
from tests.demo_build_reason import REVIEW_FOUR
import streamlit as st

import insights
import ui_insights
import wording
from tests.test_complete_results import _patty


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _check(name, rnd, no, measured, predicted, half=1.0):
    low, high = predicted - half, predicted + half
    return insights.Check(no, rnd, name, predicted, low, high, measured,
                          low <= measured <= high, True, predicted, half / insights.Z95)


def _panels(chart):
    return chart.to_dict()["concat"]


# ------------------------------------------------------------------ #
#  Measured against predicted (review M4)
# ------------------------------------------------------------------ #


def _panel_rows(spec):
    """A one-panel chart's own rows (not the 1:1 line's)."""
    return next(rows for rows in spec["datasets"].values() if rows and "latest" in rows[0])


# ------------------------------------------------------------------ #
#  Compare all on the tested-near change (review I8)
# ------------------------------------------------------------------ #

@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_dark_bar_is_the_tested_near_change_and_the_light_one_the_whole_path():
    changes = [SimpleNamespace(name="Oil", near=1.4, whole=3.0),
               SimpleNamespace(name="Pea", near=None, whole=2.0)]
    spec = ui_insights.change_chart(changes).to_dict()
    light, dark = spec["layer"]
    assert light["encoding"]["x"]["field"] == "whole"
    # Review plots: two shades of the model's blue, named in a legend.
    assert light["encoding"]["fill"]["scale"]["domain"] == [wording.PLOT_NEAR, wording.PLOT_WHOLE]
    assert dark["encoding"]["x"]["field"] == "near"
    assert dark["encoding"]["x"]["title"] == wording.CHANGE_AXIS
    assert light["encoding"]["y"]["sort"] == ["Oil", "Pea"]
    rows = list(spec["datasets"].values())[0]
    assert [(r["near"], r["whole"]) for r in rows] == [(1.4, 3.0), (0.0, 2.0)]
    # Pea's start is not tested near: its tooltip says so, never "0.0"
    # (Task 5 review I-1).
    assert [r["near_text"] for r in rows] == ["1.4", wording.MIXTURE_UNTESTED]
    tips = {t["title"]: t["field"] for t in dark["encoding"]["tooltip"]}
    assert tips[wording.CHANGE_NEAR] == "near_text"
    assert ui_insights.change_chart([]) is None


# ------------------------------------------------------------------ #
#  The Progress chart (expert #8)
# ------------------------------------------------------------------ #

def test_formulation_ticks_are_whole_and_never_repeat():
    assert ui_insights.formulation_ticks([1, 2]) == [1, 2]
    assert ui_insights.formulation_ticks(range(1, 61)) == list(range(1, 61, 5))
    assert ui_insights.formulation_ticks([]) == []


def _leaves(spec):
    for layer in spec["layer"]:
        if "layer" in layer:
            yield from _leaves(layer)
        else:
            yield layer


def test_the_progress_chart_fits_the_scores():
    """The Best so far chart: the score axis is not zero-based and its domain
    holds every score; the Formulation axis lists whole numbers, one per
    formulation."""
    opt = _patty(rows=6)
    spec = ui_insights.improving_chart(opt).to_dict()
    leaves = list(_leaves(spec))
    scale = next(l for l in leaves if l["mark"]["type"] == "line")["encoding"]["y"]["scale"]
    assert scale["zero"] is False
    scores = [float(y) for y in opt.Y_history]
    assert scale["domain"][0] <= min(scores) and scale["domain"][1] >= max(scores)
    dots = [l for l in leaves if l["mark"]["type"] == "point" and "x" in l["encoding"]]
    assert dots[0]["encoding"]["x"]["axis"]["values"] == [1, 2, 3, 4, 5, 6]
    three = ui_insights.improving_chart(_patty(rows=3)).to_dict()
    dots = [l for l in _leaves(three) if l["mark"]["type"] == "point" and "x" in l["encoding"]]
    assert dots[0]["encoding"]["x"]["axis"]["values"] == [1, 2, 3]


def test_the_progress_caption_says_a_flat_stretch_proves_nothing():
    assert wording.progress_caption_2().startswith("Dashed line: Best so far.")
    assert wording.progress_caption_2(True).endswith(wording.PROGRESS_RING)


# ------------------------------------------------------------------ #
#  cached_checks: one door for Results and Make a round
# ------------------------------------------------------------------ #

def test_cached_checks_are_the_insights_cache_or_nothing(monkeypatch):
    opt = _patty(rows=6)
    state = {}
    monkeypatch.setattr(st, "session_state", state)
    assert ui_insights.cached_checks(opt) == ()
    checks = [_check("Firmness", 2, 4, 5.0, 5.0)]
    state[ui_insights.CACHE_KEY] = (insights.data_key(opt), {"fitted": {}, "checks": checks})
    assert ui_insights.cached_checks(opt) == checks
    state[ui_insights.CACHE_KEY] = (insights.data_key(opt), ui_insights._FAILED)
    assert ui_insights.cached_checks(opt) == ()
    state[ui_insights.CACHE_KEY] = ("stale", {"fitted": {}, "checks": checks})
    assert ui_insights.cached_checks(opt) == ()
