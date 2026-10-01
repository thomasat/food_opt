"""The last fix wave of the 0.8.0 e2e campaign (final review I-1, M-2 … M-10;
re-check 05 N1 and the rows left "not done").

- N1: a measurement saved on Set up keeps every result typed on the open
  Round. Only a measurement deleted takes its typed results with it, and
  the flash says so.
"""
from streamlit.testing.v1 import AppTest

import wording
from food_bo import FoodOptimizer
from tests.test_app_ui import (APP_PATH, MEAS_GRID, _save_grid, burger,  # noqa: F401
                               open_batch)

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

import pytest  # noqa: E402

import insights  # noqa: E402
import ui_insights  # noqa: E402
from tests import demo_projects as D  # noqa: E402
from tests import e2e_fixtures as E  # noqa: E402

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
