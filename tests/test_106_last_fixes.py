"""The last fix wave of the 0.8.0 e2e campaign: final review I-1, M-2,
M-3, M-4, M-6, M-7 and M-10; re-check 05 N1, and the rows it left "not
done" (02 I8, M1, M6, M10; 01 O9, O10, O13, O16; 04 M-7).

- N1: a measurement saved on Set up keeps every result typed on the open
  Round. Only a measurement deleted takes its typed results with it, and
  the flash says so.
- I-1: no luck line when the Overall score's steady low bias covers the
  gap between Best so far and its predicted Repeat.
- M-7: an unchanged measurements grid leaves the stored shares, and so
  every Overall score, exactly as they were.
"""
import pytest
from streamlit.testing.v1 import AppTest

import insights
import ui_insights
import wording
from food_bo import FoodOptimizer
from tests import demo_projects as D
from tests import e2e_fixtures as E
from tests.test_app_ui import (APP_PATH, ING_GRID, MEAS_GRID, _grid_frame,  # noqa: F401
                               _ingredient_row, _save_grid, _submit_button, burger,
                               mid_run, open_batch, scored)

# ------------------------------------------------------------------ #
#  N1: Set up's measurement save keeps the typed results
# ------------------------------------------------------------------ #


def test_a_measurement_edit_on_set_up_keeps_the_typed_results(open_batch):
    """05 N1: Firmness typed on Round 1, then Firmness' Highest widened on
    Set up (what the out-of-range caption tells the reader to do)."""
    at = AppTest.from_file(APP_PATH, default_timeout=180).run()
    at.number_input(key='f1_Firmness').set_value(6).run()
    at.number_input(key='f2_Juiciness').set_value(7).run()
    at.text_input(key='f2_note').set_value('Dry edge').run()
    _save_grid(at, MEAS_GRID,
               edited={0: {wording.HIGHEST_MEASURABLE_LABEL: 11.0}})
    assert not at.exception
    assert FoodOptimizer('burger').objectives[1]['max_val'] == 11.0
    assert at.number_input(key='f1_Firmness').value == 6
    assert at.number_input(key='f2_Juiciness').value == 7
    assert at.text_input(key='f2_note').value == 'Dry edge'
    at.run()
    drafts = FoodOptimizer('burger').result_drafts
    assert drafts[1]['results'] == {'Firmness': 6}
    assert drafts[2]['results'] == {'Juiciness': 7}
    assert drafts[2]['note'] == 'Dry edge'


def test_a_share_edit_that_reorders_the_boxes_keeps_the_typed_results(open_batch):
    at = AppTest.from_file(APP_PATH, default_timeout=180).run()
    at.number_input(key='f1_Firmness').set_value(6).run()
    at.number_input(key='f1_Juiciness').set_value(7).run()
    # Juiciness (row 1) to 70: it now comes first.
    _save_grid(at, MEAS_GRID, edited={1: {wording.SHARE_COLUMN: 70.0}})
    assert not at.exception
    assert at.number_input(key='f1_Firmness').value == 6
    assert at.number_input(key='f1_Juiciness').value == 7
    assert FoodOptimizer('burger').result_drafts[1]['results'] == {
        'Firmness': 6, 'Juiciness': 7}


def test_deleting_a_measurement_drops_only_its_typed_results_and_says_so(open_batch):
    at = AppTest.from_file(APP_PATH, default_timeout=180).run()
    at.number_input(key='f1_Firmness').set_value(6).run()
    at.number_input(key='f1_Juiciness').set_value(7).run()
    at.number_input(key='f2_Juiciness').set_value(5).run()
    _save_grid(at, MEAS_GRID, confirm=True, deleted=[1])   # Juiciness
    assert not at.exception
    assert [o['name'] for o in FoodOptimizer('burger').objectives] == ['Firmness']
    assert at.number_input(key='f1_Firmness').value == 6
    drafts = FoodOptimizer('burger').result_drafts
    assert drafts[1]['results'] == {'Firmness': 6}
    assert 2 not in drafts or drafts[2]['results'] == {}
    said = [s.value for s in at.success] + [i.value for i in at.info]
    line = wording.typed_results_deleted("Juiciness", 2, 1)
    assert any(line in s for s in said), said
    assert "2 results typed for Juiciness on Round 1" in line


def test_deleting_a_measurement_nobody_typed_says_nothing_more(open_batch):
    at = AppTest.from_file(APP_PATH, default_timeout=180).run()
    at.number_input(key='f1_Firmness').set_value(6).run()
    _save_grid(at, MEAS_GRID, confirm=True, deleted=[1])
    said = [s.value for s in at.success] + [i.value for i in at.info]
    assert not any("typed for" in s for s in said), said
    assert at.number_input(key='f1_Firmness').value == 6


# ------------------------------------------------------------------ #
#  I-1: no luck line when the Overall score's steady bias explains it
# ------------------------------------------------------------------ #


NOISY_RECOMMENDED = "03_bar_noisy (Recommended)"


def _load(name):
    if name == NOISY_RECOMMENDED:
        return E.load(E.BAR_NOISY, recommended=True)
    if name in E.NAMES:
        return E.load(name)
    return D.load(name)


@pytest.fixture(scope="module")
def standing_of():
    """name -> (plain luck, the screen's Standing, the Overall score's
    steady bias), fitted once per project."""
    seen = {}

    def get(name):
        if name not in seen:
            opt = _load(name)
            checks = insights.check_rounds(opt)
            yard = insights.noise_yardstick(opt, checks)
            score = insights.fit(opt, insights.SCORE, list(range(len(opt.Y_history))))
            best = insights.predicted_best(opt, {insights.SCORE: score}, checks)
            plain = insights.best_standing(opt, best, yard.value)
            seen[name] = (plain.luck, ui_insights.standing_for(opt, best, checks, yard),
                          insights.score_steady_bias(opt, checks), opt, best)
        return seen[name]
    return get


def test_03_clean_has_no_luck_line_its_steady_bias_explains_the_gap(standing_of):
    """Final review I-1: predictions of the Overall score run 2.1 points low
    (Rounds 3–8), and F23's predicted Repeat is 1.6 below its measured
    91.69. The gap is the model's bias, not chance."""
    plain, standing, bias, opt, best = standing_of(E.BAR_CLEAN)
    assert plain                      # what the screen said before
    assert bias == pytest.approx(2.13, abs=0.01)
    top = float(opt.Y_history[opt.best_index()])
    assert 0 < top - best.repeat.mean <= bias
    assert not standing.luck


@pytest.mark.parametrize("name", [D.LINE_B, D.TIMING, NOISY_RECOMMENDED])
def test_the_luck_line_stays_where_no_bias_explains_it(standing_of, name):
    plain, standing, bias, opt, best = standing_of(name)
    assert plain and standing.luck, (name, bias)


def test_a_low_bias_smaller_than_the_gap_keeps_the_luck_line(standing_of):
    plain, standing, bias, opt, best = standing_of(E.BAR_CLEAN)
    top = float(opt.Y_history[opt.best_index()])
    gap = top - best.repeat.mean
    yard = insights.noise_yardstick(opt, insights.check_rounds(opt)).value
    assert insights.best_standing(opt, best, yard, score_bias=gap - 0.01).luck
    assert not insights.best_standing(opt, best, yard, score_bias=gap).luck
    # A bias the other way (predictions run high) explains nothing here.
    assert insights.best_standing(opt, best, yard, score_bias=-5.0).luck


def test_method_and_details_says_the_bias_clause():
    assert wording.HOW_REGRESSION.endswith(
        "by more than the noise threshold, and no steady bias of the Overall score "
        "explains the gap.")


# ------------------------------------------------------------------ #
#  M-7: an unchanged measurements grid leaves the shares alone
# ------------------------------------------------------------------ #

def _equal_shares_line_a(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = D.load(D.LINE_A)
    opt.set_shares({o['name']: 1.0 for o in opt.objectives})   # 33.333… each
    return opt


def test_saving_the_grid_unchanged_keeps_every_weight_and_score(tmp_path, monkeypatch):
    """Final review M-7: the save wrote the shown tenths back (33.4 / 33.3 /
    33.3), every Overall score moved by up to 0.031, and those scores feed
    the suggestions."""
    opt = _equal_shares_line_a(tmp_path, monkeypatch)
    weights = [o['weight'] for o in opt.objectives]
    scores = list(opt.Y_history)
    copies = []
    errors, messages = opt.apply_measurement_grid(opt.measurement_grid_frame(),
                                                  archive=lambda: copies.append(1))
    assert errors == [] and messages == [] and copies == []
    assert [o['weight'] for o in opt.objectives] == weights
    assert list(opt.Y_history) == scores


def test_a_unit_corrected_on_the_grid_keeps_every_weight_and_score(tmp_path, monkeypatch):
    opt = _equal_shares_line_a(tmp_path, monkeypatch)
    weights = [o['weight'] for o in opt.objectives]
    scores = list(opt.Y_history)
    frame = opt.measurement_grid_frame()
    frame.loc[frame.index[0], wording.UNIT_LABEL] = "kN"
    errors, messages = opt.apply_measurement_grid(frame)
    assert errors == []
    assert [o['weight'] for o in opt.objectives] == weights
    assert list(opt.Y_history) == scores


def test_a_typed_share_still_rebalances(tmp_path, monkeypatch):
    opt = _equal_shares_line_a(tmp_path, monkeypatch)
    frame = opt.measurement_grid_frame()
    frame.loc[frame.index[0], wording.SHARE_COLUMN] = 50.0
    errors, _ = opt.apply_measurement_grid(frame)
    assert errors == []
    shares = opt.share_percents()
    assert shares[frame.loc[frame.index[0], wording.MEASUREMENT_COLUMN]] == 50.0
    assert sum(shares.values()) == pytest.approx(100.0)


def test_a_share_typed_back_to_what_was_shown_is_not_a_change(tmp_path, monkeypatch):
    """Compared with a tolerance: the grid hands back 33.3 as a float that
    need not be the exact float it was given."""
    opt = _equal_shares_line_a(tmp_path, monkeypatch)
    weights = [o['weight'] for o in opt.objectives]
    frame = opt.measurement_grid_frame()
    frame.loc[frame.index[1], wording.SHARE_COLUMN] = float(frame.loc[frame.index[1], wording.SHARE_COLUMN]) + 1e-7
    opt.apply_measurement_grid(frame)
    assert [o['weight'] for o in opt.objectives] == weights


def test_the_app_saves_a_unit_without_moving_any_overall_score(burger):
    burger.set_shares({'Juiciness': 1.0, 'Firmness': 2.0})     # 33.333 / 66.667
    burger.tell({"Pea protein": 10.0, "Methylcellulose": 1.0},
                {"Juiciness": 6.2, "Firmness": 4.1}, formulation_no=1, batch_no=1)
    weights = [o['weight'] for o in FoodOptimizer('burger').objectives]
    scores = list(FoodOptimizer('burger').Y_history)
    at = AppTest.from_file(APP_PATH, default_timeout=180).run()
    _save_grid(at, MEAS_GRID, edited={0: {wording.UNIT_LABEL: "kN"}})
    assert not at.exception
    saved = FoodOptimizer('burger')
    assert next(o['unit'] for o in saved.objectives if o['name'] == 'Firmness') == "kN"
    assert [o['weight'] for o in saved.objectives] == weights
    assert list(saved.Y_history) == scores


# ------------------------------------------------------------------ #
#  M-2: the A3 lines name the 0.20 rule
# ------------------------------------------------------------------ #

def test_the_a3_wait_line_names_the_rule():
    assert wording.effects_wait_line(4) == (
        "Ingredient effects are reported once Round-ahead checks show the Overall score's "
        "predictions clearly better than the average (skill 0.20 over 10 checked "
        "formulations; 4 so far).")
    assert wording.effects_wait_line(4, settings_only=True).startswith("Setting effects")


def test_the_a3_no_skill_line_names_the_rule():
    assert wording.guard_skill_line() == (
        "Ingredient effects are not reported until Round-ahead predictions are clearly "
        "better than the average of earlier results (skill 0.20 needed).")
    for line in (wording.effects_wait_line(4), wording.guard_skill_line()):
        assert "beat the average" not in line


# ------------------------------------------------------------------ #
#  M-3: a measurement's unit reads naturally after its number
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("unit,said", [
    ("score", "Crumbliness by about 0.4 points"),
    ("Score", "Crumbliness by about 0.4 points"),
    ("points", "Crumbliness by about 0.4 points"),
    ("/10", "Crumbliness by about 0.4"),
    ("", "Crumbliness by about 0.4"),
    ("N", "Crumbliness by about 0.4 N"),
])
def test_bias_item_units(unit, said):
    assert wording.bias_item("Crumbliness", "0.4", unit) == said


def test_03_clean_bias_line_reads_points_not_score(standing_of):
    *_, opt, _best = standing_of(E.BAR_CLEAN)
    line = ui_insights.steady_bias_text(opt, insights.check_rounds(opt))
    assert "Crumbliness by about 0.4 points" in line and " score;" not in line


# ------------------------------------------------------------------ #
#  M-4: Line B's track line says each thing once
# ------------------------------------------------------------------ #

def test_the_rough_track_line_says_more_misses_once():
    line = wording.round_track_rough(13, 16, "5.5", see_insights=True, rounds=[2, 3, 4, 5])
    assert line == ("Overall score, Rounds 2–5: 13 of 16 inside their 95 % interval; mean "
                    "error −5.5 points. More misses than expected; many more in the measured "
                    "values.")
    assert line.lower().count("misses than expected") == 1
    # Without the measurements' verdict, as before.
    assert wording.round_track_rough(13, 16, "5.5", rounds=[2, 3, 4, 5]).endswith(
        "mean error −5.5 points. More misses than expected.")


def test_line_b_track_line(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = D.load(D.LINE_B)
    line = ui_insights.track_text(insights.round_track(opt, insights.check_rounds(opt)))
    assert "More misses than expected; many more in the measured values." in line
    assert line.lower().count("misses than expected") == 1


# ------------------------------------------------------------------ #
#  M-6: the callout's limit form never lands on an empty slot
# ------------------------------------------------------------------ #

def test_the_limit_callout_falls_back_to_the_limits_fold(burger):
    import streamlit as st

    import ui_setup
    st.session_state.pop(ui_insights.BEST_KEY, None)      # no predicted best worked out
    text = ui_setup.callout_text(burger, wording.CHANGE_LIMITS_ON_SETUP)
    assert text == wording.SETUP_CALLOUT_LIMITS
    assert wording.INGREDIENT_LIMITS_EXPANDER in text


# ------------------------------------------------------------------ #
#  M-10: both ends of the blues scale reach 3:1 in both themes
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("which", ["light", "dark"])
def test_both_ends_of_the_blues_scale_reach_3_to_1(which):
    from tests.test_104_speed_robust_charts import PAGES, _contrast_on, _vega_blues
    low, high = ui_insights.blues(which).extent
    for end in (low, high):
        assert _contrast_on(_vega_blues(end), 1.0, PAGES[which]) >= 3.0, (which, end)
    assert high - low >= 0.4          # still a scale, not one colour


# ------------------------------------------------------------------ #
#  05 I8: "Change a measurement or an ingredient" while correcting a
#  recorded formulation goes to the correction, not to Set up
# ------------------------------------------------------------------ #

def test_change_while_correcting_goes_to_the_correction(scored):
    import ui_results
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["main_tab"] = wording.TAB_RESULTS
    at.run()
    at.selectbox(key="correct_formulation").set_value(2).run()
    at.number_input(key="correct_2_Firmness").set_value(8.5).run()
    _submit_button(at, wording.CHANGE_SETUP_FROM_RESULTS_BUTTON).click().run()
    assert not at.exception
    assert at.session_state["main_tab"] == wording.TAB_RESULTS
    assert at.session_state["correct_formulation"] == 2
    assert at.number_input(key="correct_2_Firmness").value == 8.5
    # The page was asked to bring the correction into view.
    scripts = [h.proto.body for h in at.get("html")]
    assert any(f".st-key-{ui_results.CORRECTION_SLOT}" in b for b in scripts), scripts
    assert FoodOptimizer("burger").results_history[1]["Firmness"] == 8.0   # nothing saved


def test_change_without_a_correction_still_goes_to_set_up(scored):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["main_tab"] = wording.TAB_RESULTS
    at.run()
    _submit_button(at, wording.CHANGE_SETUP_FROM_RESULTS_BUTTON).click().run()
    assert at.session_state["main_tab"] == wording.TAB_SETUP


# ------------------------------------------------------------------ #
#  02 M1: Generate opens on the count the project last used
# ------------------------------------------------------------------ #

def test_generate_opens_on_the_last_count_used(burger):
    import ui_batch
    assert ui_batch.last_generate_count(burger) == 3        # nothing generated yet
    burger.record_generation(1, {"seed": 1, "rows": [], "n": 4, "regime": "cold", "key": ""})
    burger.record_generation(2, {"seed": 5, "rows": [], "n": 5, "regime": "cold", "key": ""})
    assert ui_batch.last_generate_count(burger) == 5
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["main_tab"] = wording.TAB_BATCH
    at.run()
    assert at.number_input(key="how_many").value == 5
    # A count typed on this screen is the reader's until the next Round.
    at.number_input(key="how_many").set_value(2).run()
    at.run()
    assert at.number_input(key="how_many").value == 2


def test_generate_after_a_round_keeps_its_count(open_batch):
    """The box is not drawn while a Round is open, so Streamlit forgot it
    and it came back at 3 after every Round."""
    open_batch.record_generation(1, {"seed": 1, "rows": [], "n": 4, "regime": "cold",
                                     "key": ""})
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["main_tab"] = wording.TAB_BATCH
    at.run()
    for no in (1, 2):
        at.number_input(key=f"f{no}_Firmness").set_value(6).run()
        at.number_input(key=f"f{no}_Juiciness").set_value(7).run()
    _submit_button(at, wording.SAVE_RESULTS).click().run()
    assert FoodOptimizer("burger").pending_batch is None
    assert at.number_input(key="how_many").value == 4


# ------------------------------------------------------------------ #
#  02 M6: the line beside Save results cannot be read as saving
# ------------------------------------------------------------------ #

def test_the_draft_line_is_not_a_second_save():
    line = wording.RESULT_DRAFT_SAVED
    assert "saved" not in line.lower()
    assert wording.SAVE_RESULTS in line and "not recorded" in line


def test_the_draft_line_shows_under_typed_values(open_batch):
    at = AppTest.from_file(APP_PATH, default_timeout=180).run()
    at.number_input(key='f1_Firmness').set_value(6).run()
    assert wording.RESULT_DRAFT_SAVED in [c.value for c in at.caption]


# ------------------------------------------------------------------ #
#  02 M10: no empty Baseline column
# ------------------------------------------------------------------ #

def _shown_columns(at, which=0):
    return list(at.dataframe[which].proto.column_order)


def test_an_empty_baseline_column_is_not_shown(burger):
    burger.tell({"Pea protein": 10.0, "Methylcellulose": 1.0},
                {"Juiciness": 7.0, "Firmness": 6.0}, formulation_no=1, batch_no=1)
    at = AppTest.from_file(APP_PATH, default_timeout=180).run()
    assert wording.HIGHEST_LABEL in _shown_columns(at)
    assert wording.BASELINE_LABEL not in _shown_columns(at)


def test_a_baseline_a_setting_has_is_shown(mid_run):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "mid_run"
    at.run()
    assert wording.BASELINE_LABEL in _shown_columns(at)


def test_a_new_setting_without_a_baseline_brings_the_column_back(burger):
    """After results exist a new process setting needs its baseline: the
    refusal says so, and the column it is typed in appears, with the row
    the reader typed still there."""
    burger.tell({"Pea protein": 10.0, "Methylcellulose": 1.0},
                {"Juiciness": 7.0, "Firmness": 6.0}, formulation_no=1, batch_no=1)
    at = AppTest.from_file(APP_PATH, default_timeout=180).run()
    row = _ingredient_row("Cook temperature", kind=wording.KIND_SETTING,
                          low=150.0, high=200.0, unit="°C")
    _save_grid(at, ING_GRID, added=[row])
    assert [e.value for e in at.error] == [
        wording.named_row_error("Cook temperature", wording.ADD_BASELINE_ERROR)]
    assert wording.BASELINE_LABEL in _shown_columns(at)
    assert "Cook temperature" in list(_grid_frame(at, 0)["Name"])


# ------------------------------------------------------------------ #
#  01 O9: a withheld expected improvement has no "Recomputed" line
# ------------------------------------------------------------------ #

def _why_fold_captions(monkeypatch, gate, reasons=("skill",)):
    import dataclasses

    import bo_views
    from tests.test_105_integration import _why_caption
    real = bo_views.why
    monkeypatch.setattr(bo_views, "why",
                        lambda *a, **k: dataclasses.replace(real(*a, **k), exact=False))
    return _why_caption(monkeypatch, gate, reasons)


def test_a_withheld_expected_improvement_is_not_said_to_be_recomputed(monkeypatch):
    gate = insights.SkillGate("wait", 6)
    captions = _why_fold_captions(monkeypatch, gate)
    assert wording.why_guard_line(gate_reason=wording.gain_reason_too_few(6)) in captions
    assert wording.WHY_RECOMPUTED not in captions


def test_a_shown_expected_improvement_still_says_it_was_recomputed(monkeypatch):
    captions = _why_fold_captions(monkeypatch, None, reasons=())
    if any(c.startswith("Expected improvement is not shown") for c in captions):
        pytest.skip("the fixture's own checks withhold it")
    assert wording.WHY_RECOMPUTED in captions


# ------------------------------------------------------------------ #
#  01 O16: the "not tested near here" band shows in dark mode
# ------------------------------------------------------------------ #

def test_the_untested_band_is_visible_on_the_dark_page(monkeypatch):
    from tests.test_104_speed_robust_charts import PAGES, _all_layers, _contrast_on
    colour, opacity = ui_insights._FILLS["dark"]["untested"]
    assert _contrast_on(colour, opacity, PAGES["dark"]) >= 1.7      # was 1.14:1
    assert _contrast_on(*ui_insights._FILLS["light"]["untested"], PAGES["light"]) >= 1.1
    monkeypatch.setattr(ui_insights, "theme", lambda: "dark")
    import altair as alt
    [rect, *_] = ui_insights._untested_layers(alt.Chart())
    mark = rect.to_dict()["mark"]
    assert (mark["color"], mark["opacity"]) == (colour, opacity)


# ------------------------------------------------------------------ #
#  04 M-7: no deprecated use_container_width flooding the log
# ------------------------------------------------------------------ #

def test_no_screen_uses_the_deprecated_width_flag():
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent
    users = [p.name for p in sorted(root.glob("*.py"))
             if "use_container_width" in p.read_text()]
    assert users == []


def test_a_run_logs_no_width_deprecation(open_batch, caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        at = AppTest.from_file(APP_PATH, default_timeout=180).run()
        at.number_input(key='f1_Firmness').set_value(6).run()
    assert not at.exception
    assert not [r for r in caplog.records if "use_container_width" in r.getMessage()]
