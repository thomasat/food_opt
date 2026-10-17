"""0.8.0 review fixes, Task 4: every tooltip is titled and rounded, the two
charts without a key get one, the path chart's y axes are not forced to 0,
measurements come in Results' order, and chart text is readable in both
themes (spec 2026-09-29 §6). The action menu and the Altair Progress chart
are built separately ($PRIV)."""

import pytest
from tests.demo_build_reason import REVIEW_FOUR

import insights
import ui_insights
import wording
from tests.test_insights import _burger
from tests.test_insights_wave_c_ui import (  # noqa: F401
    _app, _mix_parts, _open_insights, mixture_scored)


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


# ------------------------------------------------------------------ #
#  Tooltips: titled, rounded, none on decoration
# ------------------------------------------------------------------ #

def _layers(spec):
    if "concat" in spec:            # measured against predicted: one panel per output
        return [layer for panel in spec["concat"] for layer in _layers(panel)]
    spec = spec.get("spec", spec)
    return spec.get("layer", [spec])


def _assert_tooltips_clean(spec):
    for layer in _layers(spec):
        mark = layer["mark"] if isinstance(layer["mark"], dict) else {"type": layer["mark"]}
        tip = layer.get("encoding", {}).get("tooltip")
        if mark.get("tooltip") is False:
            continue
        assert tip, f"{mark['type']} layer shows raw encoding fields"
        for t in tip if isinstance(tip, list) else [tip]:
            assert t.get("title"), t


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_every_layer_of_every_chart_has_a_titled_tooltip_or_none(mixture_scored):
    opt = mixture_scored
    fitted = insights.fit_all(opt)
    checks = insights.check_rounds(opt)
    start = insights.start_recipe(opt, opt.best_index())
    path = insights.build_path(opt, "Oil", start)
    mix, preds = _mix_parts(opt)
    points = insights.pair_points(opt, "Firmness", "Juiciness")
    for chart in (ui_insights.measured_vs_predicted_chart(opt, checks),
                  ui_insights.path_chart(opt, path, insights.path_predictions(opt, fitted, path)),
                  ui_insights.change_chart(insights.tested_near_changes(opt, fitted, start).moved),
                  ui_insights.pair_chart(opt, "Firmness", "Juiciness", points),
                  ui_insights.mixture_chart(opt, mix, preds, 7)):
        _assert_tooltips_clean(chart.to_dict())


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_a_waiting_formulation_reads_what_it_waits_for_not_null():
    opt = _burger(rounds=2)
    opt.add_objective("Cook loss", 20, goal="min", min_val=0, max_val=30, unit="%")
    opt.tell(opt.fill_formulas({"Pea": 6.0, "Oil": 4.0}), {"Firmness": 6.0, "Juiciness": 5.0},
             formulation_no=50, batch_no=3)
    spec = ui_insights.pair_chart(opt, "Firmness", "Juiciness",
                                  insights.pair_points(opt, "Firmness", "Juiciness")).to_dict()
    rows = [r for rows in spec["datasets"].values() for r in rows if r.get("no") == 50]
    assert rows and rows[0]["score_text"] == "waiting for Cook loss"


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_path_tooltip_reads_amount_and_prediction_with_its_range():
    opt = _burger(rounds=3)
    fitted = insights.fit_all(opt)
    path = insights.build_path(opt, "Oil", insights.start_recipe(opt, opt.best_index()))
    spec = ui_insights.path_chart(opt, path, insights.path_predictions(opt, fitted, path)).to_dict()
    row = next(r for rows in spec["datasets"].values() for r in rows if r.get("mark") == "curve")
    assert row["amount"].endswith(" g")
    assert " (" in row["predicted"] and "–" in row["predicted"]


# ------------------------------------------------------------------ #
#  Keys, axes, order, decimals, contrast
# ------------------------------------------------------------------ #

@pytest.mark.skip(reason="review profiler prototype: pins Try one's path and Compare all, which the profiler replaces; reconcile at the final merge")
def test_show_each_prediction_and_the_path_have_a_one_line_key(mixture_scored):
    at = _open_insights(_app())
    # Measured against predicted is drawn only while open (BO views, 04 §5).
    at.session_state[ui_insights.EACH_KEY] = True
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    tab = at.tabs[3]
    shown = next(e for e in tab.expander if e.label == wording.FOLD_MEASURED_CHART)
    # Review plots: a short reading under each chart; the key in its "?".
    assert [(c.value, c.proto.help) for c in shown.caption] == [
        (wording.PLOT_CAPTION_MVP, wording.MVP_CAPTION)]
    best = mixture_scored.best_formulation_no()
    assert any(c.value == wording.PLOT_CAPTION_PATH
               and c.proto.help.endswith(f"Dotted: Formulation {best}; dashed: target.")
               for c in tab.caption)


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_path_y_axes_are_not_forced_to_zero():
    opt = _burger(rounds=3)
    fitted = insights.fit_all(opt)
    path = insights.build_path(opt, "Oil", insights.start_recipe(opt, opt.best_index()))
    spec = ui_insights.path_chart(opt, path, insights.path_predictions(opt, fitted, path)).to_dict()
    band = next(l for l in spec["spec"]["layer"] if l["mark"]["type"] == "area")
    assert band["encoding"]["y"]["scale"]["zero"] is False


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_measurements_come_in_results_order():
    opt = _burger(rounds=3)                     # Firmness 60, Juiciness 40
    opt.set_shares({"Firmness": 30, "Juiciness": 70})
    assert ui_insights.track_panels(opt) == ["Juiciness", "Firmness", insights.SCORE]
    assert ui_insights.path_panels(opt) == [insights.SCORE, "Juiciness", "Firmness"]
    rows = insights.details(opt, insights.check_rounds(opt))
    assert [r.name for r in rows][:2] == ["Juiciness", "Firmness"]


def test_one_decimal_rule_per_measurement():
    opt = _burger(rounds=1)                      # 0-10 panels
    opt.add_objective("Viscosity", 10, goal="max", min_val=0, max_val=5000)
    opt.add_objective("pH drift", 10, goal="min", min_val=0, max_val=0.5)
    assert ui_insights.decimals(opt, "Firmness") == 1
    assert ui_insights.decimals(opt, "Viscosity") == 0
    assert ui_insights.decimals(opt, "pH drift") == 2
    assert ui_insights.decimals(opt, insights.SCORE) == 0


def _luminance(hex_colour):
    def lin(c):
        c /= 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (int(hex_colour[k:k + 2], 16) for k in (1, 3, 5))
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def _contrast(a, b):
    la, lb = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def test_chart_text_reaches_4_5_to_1_in_both_themes(monkeypatch):
    assert _contrast(ui_insights._MUTED_LIGHT, "#ffffff") >= 4.5
    assert _contrast(ui_insights._MUTED_DARK, "#0e1117") >= 4.5
    assert ui_insights.muted() == ui_insights._MUTED_LIGHT      # no theme known: light
    import streamlit as st

    class Dark:
        type = "dark"
    monkeypatch.setattr(type(st.context), "theme", property(lambda self: Dark()))
    assert ui_insights.muted() == ui_insights._MUTED_DARK


def test_the_progress_chart_tooltips_are_titled_and_read_one_decimal():
    """Carried from the privacy review: the Altair Progress chart's tooltips
    had no title or format and read 73.283749…. Every layer is titled or
    carries none; each score reads one decimal, each Formulation a whole
    number."""
    from tests.test_best_candidates import _upgrade_like
    spec = ui_insights.progress_chart(_upgrade_like()).to_dict()
    _assert_tooltips_clean(spec)
    tips = [t for layer in _layers(spec)
            for t in layer.get("encoding", {}).get("tooltip", [])]
    assert tips
    for t in tips:
        if t["field"] in (wording.OVERALL_SCORE_COLUMN, wording.BEST_SO_FAR_COLUMN):
            assert t["format"] == ".1f", t
        elif t["field"] in ("round", "bo"):          # review plots: Round, BO term
            assert t["title"] in (wording.ROUND_CAP, wording.PLOT_BO), t
        else:
            assert t["title"] == wording.FORMULATION_CAP and t["format"] == "d", t


