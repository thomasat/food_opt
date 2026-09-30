"""0.8.0 speed and simplicity, Plan B, Task 5: the Insights charts —
measured against predicted, Compare all on the tested-near change, the
Progress chart's axes and caption, and one door to the cached checks (spec
2026-09-30 §2.2, §2.3, §2.5; review I8, M4)."""
from types import SimpleNamespace

import pytest
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

def test_one_panel_per_measurement_then_the_overall_score():
    opt = _patty(rows=6)
    checks = [_check("Juiciness", 2, 4, 5.0, 5.5), _check("Firmness", 2, 4, 6.0, 6.2),
              _check(insights.SCORE, 2, 4, 70.0, 72.0, 5.0)]
    panels = _panels(ui_insights.measured_vs_predicted_chart(opt, checks))
    titles = [p["title"] for p in panels]
    assert titles == [ui_insights.panel_label(opt, n) for n in ("Firmness", "Juiciness")] + [
        wording.OVERALL_SCORE_COLUMN]


def test_both_axes_share_one_padded_domain():
    opt = _patty(rows=6)
    checks = [_check("Firmness", 2, 4, 3.0, 5.0), _check("Firmness", 3, 7, 8.0, 7.0)]
    [panel] = _panels(ui_insights.measured_vs_predicted_chart(opt, checks))
    domains = {tuple(layer["encoding"][axis]["scale"]["domain"])
               for layer in panel["layer"] for axis in ("x", "y")}
    assert len(domains) == 1
    [(lo, hi)] = domains
    # Low 3.0 (measured) to high 8.0 (the second's likely range), padded 5 %.
    assert (lo, hi) == pytest.approx((3.0 - 0.25, 8.0 + 0.25))


def test_the_bar_is_vertical_filled_means_inside_and_the_latest_round_is_darker():
    opt = _patty(rows=6)
    checks = [_check("Firmness", 2, 4, 3.0, 5.0), _check("Firmness", 3, 7, 7.2, 7.0)]
    [panel] = _panels(ui_insights.measured_vs_predicted_chart(opt, checks))
    line, bars, rings, filled = panel["layer"]
    assert line["mark"]["strokeDash"] == [4, 3]
    assert bars["mark"]["type"] == "rule" and "y2" in bars["encoding"]
    assert bars["encoding"]["x"]["field"] == wording.TRACK_PREDICTED
    assert filled["transform"] == [{"filter": {"field": "inside", "equal": True}}]
    assert rings["encoding"]["opacity"]["condition"]["value"] == 1.0
    assert rings["encoding"]["opacity"]["value"] == 0.45


def test_no_checks_no_chart():
    assert ui_insights.measured_vs_predicted_chart(_patty(rows=0), []) is None


# ------------------------------------------------------------------ #
#  Compare all on the tested-near change (review I8)
# ------------------------------------------------------------------ #

def test_the_dark_bar_is_the_tested_near_change_and_the_light_one_the_whole_path():
    changes = [SimpleNamespace(name="Oil", near=1.4, whole=3.0),
               SimpleNamespace(name="Pea", near=None, whole=2.0)]
    spec = ui_insights.change_chart(changes).to_dict()
    light, dark = spec["layer"]
    assert light["encoding"]["x"]["field"] == "whole" and light["mark"]["opacity"] == 0.35
    assert dark["encoding"]["x"]["field"] == "near"
    assert dark["encoding"]["x"]["title"] == wording.CHANGE_AXIS
    assert light["encoding"]["y"]["sort"] == ["Oil", "Pea"]
    rows = list(spec["datasets"].values())[0]
    assert [(r["near"], r["whole"]) for r in rows] == [(1.4, 3.0), (0.0, 2.0)]
    assert ui_insights.change_chart([]) is None


# ------------------------------------------------------------------ #
#  The Progress chart (expert #8)
# ------------------------------------------------------------------ #

def test_formulation_ticks_are_whole_and_never_repeat():
    assert ui_insights.formulation_ticks([1, 2]) == [1, 2]
    assert ui_insights.formulation_ticks(range(1, 61)) == list(range(1, 61, 5))
    assert ui_insights.formulation_ticks([]) == []


def test_the_progress_chart_fits_the_scores():
    opt = _patty(rows=6)
    spec = ui_insights.progress_chart(opt).to_dict()
    score_line, dots, best_line = spec["layer"]
    assert score_line["encoding"]["y"]["scale"] == {"zero": False}
    assert best_line["encoding"]["y"]["scale"] == {"zero": False}
    assert score_line["encoding"]["x"]["axis"]["values"] == [1, 2, 3, 4, 5, 6]


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
