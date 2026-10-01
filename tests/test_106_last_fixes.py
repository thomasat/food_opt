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
