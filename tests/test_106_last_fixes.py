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
import os

import pytest
from tests.demo_build_reason import REVIEW_FOUR
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


# ------------------------------------------------------------------ #
#  M-4: Line B's track line says each thing once
# ------------------------------------------------------------------ #

def test_the_rough_track_line_says_more_misses_once():
    line = wording.round_track_rough(13, 16, "5.5", see_insights=True, rounds=[2, 3, 4, 5])
    # The bias as one fact (polish round 8).
    assert line == ("Overall score, Rounds 2–5: 13 of 16 inside their 95 % interval; "
                    "Overall scores have come in about 5.5 points below the predictions. More "
                    "misses than expected; many more in the measured values.")
    assert line.lower().count("misses than expected") == 1
    # Without the measurements' verdict, as before.
    assert wording.round_track_rough(13, 16, "5.5", rounds=[2, 3, 4, 5]).endswith(
        "scores have come in about 5.5 points below the predictions. More misses than "
        "expected.")


def test_line_b_track_line(tmp_path, monkeypatch):
    # Line B's bias line shows under the earlier 2-standard-error rule only
    # (ftue fix A, cold read I5: t = 2.45 < t_{0.975, 3} = 3.18); the
    # wording with a bias is checked under that rule.
    monkeypatch.setattr(insights, "t_critical", lambda df, p=0.975: 2.0)
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

@pytest.mark.skip(reason=REVIEW_FOUR)
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


def _correction_fold(at):
    import ui_results
    return next(e for e in at.tabs[2].expander
                if e.label == wording.for_project(FoodOptimizer("burger"),
                                                  wording.EDIT_PAST_FOLD)), ui_results


def test_change_from_the_results_view_opens_the_correction_on_that_formulation(scored):
    """Polish round I8 (last-fix review R-0): from the plain Results view,
    with a recorded formulation on screen, the button opens the correction
    fold on that formulation, not Set up."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["main_tab"] = wording.TAB_RESULTS
    at.run()
    best = int(FoodOptimizer("burger").formulation_ids[FoodOptimizer("burger").best_index()])
    _submit_button(at, wording.CHANGE_SETUP_FROM_RESULTS_BUTTON).click().run()
    assert not at.exception
    assert at.session_state["main_tab"] == wording.TAB_RESULTS
    assert at.session_state["correct_formulation"] == best
    assert f"correct_{best}_Firmness" in [n.key for n in at.number_input]
    fold, ui_results = _correction_fold(at)
    assert at.session_state[ui_results.EDIT_PAST_KEY] is True
    scripts = [h.proto.body for h in at.get("html")]
    assert any(f".st-key-{ui_results.CORRECTION_SLOT}" in b for b in scripts), scripts


def test_change_opens_the_formulation_picked_in_the_view_box(scored):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["main_tab"] = wording.TAB_RESULTS
    at.run()
    at.selectbox(key="result_view_burger").set_value(1).run()
    _submit_button(at, wording.CHANGE_SETUP_FROM_RESULTS_BUTTON).click().run()
    assert at.session_state["correct_formulation"] == 1
    assert "correct_1_Firmness" in [n.key for n in at.number_input]


def test_the_correction_picker_reads_like_the_view_box(scored):
    """Number, Round and status, as the View formulation box says them,
    not bare numbers."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["main_tab"] = wording.TAB_RESULTS
    at.run()
    opt = FoodOptimizer("burger")
    best = int(opt.formulation_ids[opt.best_index()])
    want = [wording.result_view_option(n, 1, best=n == best, unscored=n == 3)
            for n in (1, 2, 3)]
    assert at.selectbox(key="correct_formulation").options == want
    view = at.selectbox(key="result_view_burger").options
    assert set(view) <= set(want), (view, want)


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
#  01 O16: the "not tested near here" band shows in dark mode
# ------------------------------------------------------------------ #

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


# ------------------------------------------------------------------ #
#  The Effect chart's target line takes the neutral token (3:1 or better)
# ------------------------------------------------------------------ #

@pytest.mark.skip(reason=REVIEW_FOUR)
@pytest.mark.parametrize("which", ["light", "dark"])
def test_the_target_line_reaches_3_to_1(monkeypatch, which):
    from tests.test_104_speed_robust_charts import PAGES, _contrast_on
    monkeypatch.setattr(ui_insights, "theme", lambda: which)
    mark = ui_insights._target_layer().to_dict()["mark"]
    assert (mark["color"], mark["opacity"]) == ui_insights.mark("neutral")
    assert mark["strokeDash"] == [4, 3]
    assert _contrast_on(mark["color"], mark["opacity"], PAGES[which]) >= 3.0


# ------------------------------------------------------------------ #
#  Last-fix review R-2: a delete-only save keeps equal shares equal
# ------------------------------------------------------------------ #

def test_deleting_a_row_leaves_the_equal_shares_at_50_50(tmp_path, monkeypatch):
    opt = _equal_shares_line_a(tmp_path, monkeypatch)
    frame = opt.measurement_grid_frame()
    gone = frame.loc[frame.index[2], wording.MEASUREMENT_COLUMN]
    errors, messages = opt.apply_measurement_grid(frame.drop(index=frame.index[2]))
    assert errors == []
    assert gone not in [o['name'] for o in opt.objectives]
    assert [o['weight'] for o in opt.objectives] == pytest.approx([50.0, 50.0], abs=1e-9)
    assert list(opt.share_percents().values()) == [50.0, 50.0]


def test_deleting_a_row_while_typing_a_share_still_rebalances(tmp_path, monkeypatch):
    opt = _equal_shares_line_a(tmp_path, monkeypatch)
    frame = opt.measurement_grid_frame().iloc[:2].copy()
    frame.loc[frame.index[0], wording.SHARE_COLUMN] = 70.0
    errors, _ = opt.apply_measurement_grid(frame)
    assert errors == []
    assert opt.share_percents()[frame.loc[frame.index[0], wording.MEASUREMENT_COLUMN]] == 70.0


# ------------------------------------------------------------------ #
#  Polish 4: the effect line names the nearest tested formulation, as
#  the Why fold does, by the same distance
# ------------------------------------------------------------------ #

def test_the_untested_start_line_without_a_nearest():
    assert wording.near_start_untested("Best so far") == (
        "Nothing has been tested within 15 % of each span of Best so far, so nothing "
        "is ranked.")


# ------------------------------------------------------------------ #
#  Polish 5: the A3 gate holds back the predicted gain, its amounts and
#  the triangle's outline too (the upgrade file, 2 checks)
# ------------------------------------------------------------------ #

@pytest.fixture(scope="module")
def upgrade():
    import tempfile
    here = os.getcwd()
    os.chdir(tempfile.mkdtemp())
    try:
        opt = D.load(D.UPGRADE)
        checks = insights.check_rounds(opt)
        fitted = insights.fit_all(opt)
    finally:
        os.chdir(here)
    return opt, checks, fitted


def test_the_upgrade_files_predicted_gain_is_withheld(upgrade):
    opt, checks, fitted = upgrade
    yard = insights.noise_yardstick(opt, checks)
    best = insights.predicted_best(opt, {insights.SCORE: fitted[insights.SCORE]}, checks)
    standing = ui_insights.standing_for(opt, best, checks, yard)
    assert standing.higher and not standing.tested_near        # what it claimed before
    reason = ui_insights.claims_withheld_reason(opt, fitted[insights.SCORE], checks)
    assert reason == wording.gain_reason_too_few(2)
    line = ui_insights.standing_text(opt, standing, yard, withheld=reason)
    assert "Predicted gain" not in line and "+9.3" not in line
    # 87–100 since ftue fix A's Round-grouped folds (cold read I9); 93–100
    # before. Review learned: no prediction reads 100 (the ceiling reads
    # "≥ 99").
    assert line.startswith("Predicted best: ≥ 99 (95 % interval 87–≥ 99), in untested amounts.")
    assert line.endswith("Its gain over Best so far is not reported: fewer than 10 "
                         "Round-ahead checks of the Overall score so far (2).")


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_upgrade_files_triangle_has_no_outline(upgrade, monkeypatch):
    opt, checks, fitted = upgrade
    names = insights.mixture_choices(opt)[:3]
    best = opt.best_index()
    start = insights.start_recipe(opt, best)
    mix = insights.mixture_slice(opt, names, start)
    preds = insights.mixture_predictions(opt, fitted, mix)
    gated = ui_insights._mixture_extra(opt, fitted, mix, preds, wording.LAYER_SCORE, checks)
    assert gated["outline"] and not any(gated["outline"])
    monkeypatch.setattr(ui_insights, "withheld", lambda *a, **k: None)
    open_ = ui_insights._mixture_extra(opt, fitted, mix, preds, wording.LAYER_SCORE, checks)
    assert any(open_["outline"])                       # the gate is what removes it


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_upgrade_files_progress_shows_no_amounts_fold(tmp_path, monkeypatch):
    import json
    monkeypatch.chdir(tmp_path)
    state = json.loads((D.HERE / f"{D.UPGRADE}.json").read_text())
    saved = FoodOptimizer(state["project_name"])
    saved.import_json(state)
    saved.save()
    at = AppTest.from_file(APP_PATH, default_timeout=600)
    at.session_state["_loaded_project"] = state["project_name"]
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    tab = at.tabs[3]
    said = [m.value for m in tab.markdown]
    assert not any("Predicted gain over" in m for m in said), said
    assert any("Its gain over Best so far is not reported" in m for m in said), said
    labels = [e.label for e in tab.expander]
    assert wording.show_its_amounts(False) not in labels, labels


# ------------------------------------------------------------------ #
#  Polish 6: "assumed" says why when the only Repeat disagrees (Line B)
# ------------------------------------------------------------------ #

def test_beat_by_says_the_only_repeat_disagrees():
    assert wording.beat_by(assumed=True, pairs=0, left_out=1) == (
        "the 1-point noise threshold (assumed: the only Repeat disagrees; a Repeat of Best "
        "so far would measure it)")
    assert wording.beat_by(assumed=True, pairs=1, left_out=1) == (
        "the 1-point noise threshold (assumed: 1 pair of Repeats, 3 needed, and 1 left out "
        "as it disagrees; a Repeat of Best so far would measure it)")
    assert wording.beat_by(assumed=True) == wording.beat_by(assumed=True, pairs=0, left_out=0)


def test_line_b_never_says_no_repeats_yet(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = D.load(D.LINE_B)
    checks = insights.check_rounds(opt)
    yard = insights.noise_yardstick(opt, checks)
    # Ftue fix A (cold read I7): the disagreeing Repeat stays in the Repeat
    # SD, so Line B counts its one pair (1 df, 3 needed). Re-pinned: fix
    # round B (statistician I11) says degrees of freedom, not pairs.
    assert (yard.pairs, yard.left_out, yard.assumed, yard.df) == (1, 0, True, 1)
    text = ui_insights.threshold_text(yard)
    assert "1 df of Repeats, 3 needed" in text and "no Repeats yet" not in text


# ------------------------------------------------------------------ #
#  Polish 7: one wording for the effect gate's reason, both lines name
#  the 0.20 rule
# ------------------------------------------------------------------ #

def _gates():
    from tests.test_102_w1_insights import _fact, _patty, _skill_checks
    opt = _patty(rows=6)
    return [insights.noise_guard(opt, None, _skill_checks(6)),
            insights.noise_guard(opt, None, [], fact=_fact(0.09, 24)),
            insights.noise_guard(opt, None, [], fact=_fact(-0.5, 24))]


def _reason(line):
    for cut in (" reported ", " read as effects "):
        if cut in line:
            return line.split(cut, 1)[1].split(" ", 1)[1]
    raise AssertionError(line)


# ------------------------------------------------------------------ #
#  Polish 8: the steady bias is one fact, with no reading clause, on
#  Insights and Make a round alike (Lines B and C)
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("name,gap", [(D.LINE_B, "5.5"), (D.LINE_C, "4.3")])
def test_the_steady_bias_reads_as_one_fact_on_both_screens(tmp_path, monkeypatch, name, gap):
    monkeypatch.chdir(tmp_path)
    opt = D.load(name)
    checks = insights.check_rounds(opt)
    # Ftue fix A (cold read I5): on t_{0.975, 3} = 3.18 standard errors
    # neither Line shows an Overall-score bias (t = 2.45 and 2.69; Line C's
    # Cook loss bias still shows) ...
    assert insights.score_steady_bias(opt, checks) is None
    assert "Overall score" not in ui_insights.steady_bias_text(opt, checks)
    # ... so the one-fact wording is checked under the earlier 2-SE rule.
    monkeypatch.setattr(insights, "t_critical", lambda df, p=0.975: 2.0)
    track = ui_insights.track_text(insights.round_track(opt, checks))
    insights_line = ui_insights.steady_bias_text(opt, checks)
    fact = wording.bias_fact(gap)
    assert fact == f"scores have come in about {gap} points below the predictions"
    assert f"; Overall {fact}" in track, track
    assert f"Overall score by about {gap} points" in insights_line
    for line in (track, insights_line):
        assert "read about" not in line and "so new predictions" not in line, line
    assert "mean error −" not in track, track


# ------------------------------------------------------------------ #
#  Polish 9: one Water caption once the batch size is set
# ------------------------------------------------------------------ #

def test_a_new_project_shows_one_water_caption_after_the_batch_size(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("new_water")
    opt.set_amount_unit("g")
    opt.add_ingredient("Pea protein", 15, 25)
    opt.add_ingredient("Oil", 5, 10)
    opt.add_ingredient("Water", 0, 0)
    opt.add_objective("Firmness", 1.0, goal="max", min_val=0, max_val=10)
    opt.set_formula("Water", "= rest")
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "new_water"
    at.run()
    at.number_input(key="formulation_total").set_value(100.0).run()
    assert not at.exception
    water = [c.value for c in at.caption if c.value.startswith("Water ")]
    assert len(water) == 1, water
    assert water[0].startswith("Water fills to the 100 g")
    said = [c.value for c in at.caption]
    assert wording.rest_row_keeps_the_others("100 g") in said, said
    assert wording.rest_row_keeps_the_others("100 g") == (
        "At 100 g, the other rows keep their amounts. To keep the same proportions, "
        "widen Lowest and Highest too.")


# ------------------------------------------------------------------ #
#  Polish 10: the grey rings of tested formulations reach 3:1 in both
#  themes (Why map, triangle)
# ------------------------------------------------------------------ #

def _point_layers(spec):
    from tests.test_104_speed_robust_charts import _all_layers
    out = []
    for layer in _all_layers(spec):
        mark = layer.get("mark")
        if isinstance(mark, dict) and mark.get("type") == "point":
            out.append(mark)
    return out


def _seen(mark):
    colour = mark.get("stroke") or mark.get("color")
    # Vega draws a point at opacity 0.7 unless told otherwise.
    return colour, mark.get("strokeOpacity", mark.get("opacity", 0.7))


@pytest.mark.skip(reason=REVIEW_FOUR)
@pytest.mark.parametrize("which", ["light", "dark"])
def test_the_triangles_rings_reach_3_to_1(tmp_path, monkeypatch, which):
    from tests.test_104_speed_robust_charts import PAGES, _contrast_on
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ui_insights, "theme", lambda: which)
    opt = D.load(D.LINE_A)
    fitted = insights.fit_all(opt)
    names = insights.mixture_choices(opt)[:3]
    best = opt.best_index()
    start = insights.start_recipe(opt, best)
    mix = insights.mixture_slice(opt, names, start)
    preds = insights.mixture_predictions(opt, fitted, mix)
    number = int(opt.formulation_ids[best])
    spec = ui_insights.mixture_chart(opt, mix, preds, number).to_dict()
    marks = _point_layers(spec)
    assert marks
    for mark in marks:
        colour, opacity = _seen(mark)
        assert _contrast_on(colour, opacity, PAGES[which]) >= 3.0, (which, mark)


# ------------------------------------------------------------------ #
#  Polish 11: the Role tooltip and the Method text say what the rule says
# ------------------------------------------------------------------ #

def test_the_role_tooltip_says_or_a_repeat_of_it():
    help_text = wording.role_help()
    assert ("Close to the best: every varying row within 15 % of its span of Best so far "
            "or a Repeat of it.") in help_text
    assert "recorded before its Round" in help_text


