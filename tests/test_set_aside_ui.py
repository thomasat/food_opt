"""0.8.0 review fixes, Task 3 on screen: the panel under a formulation that
waits for results (spec 2026-09-29 §1.1, §9 B2/I5, §9 I13 stale label)."""
import pytest
from streamlit.testing.v1 import AppTest

import ui_results
import wording
from food_bo import FoodOptimizer
from tests.test_app_ui import APP_PATH, _submit_button, burger, scored  # noqa: F401
from tests.test_complete_results_ui import one_waiting  # noqa: F401


def _results(select=None):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.session_state["main_tab"] = wording.TAB_RESULTS
    at.run()
    if select is not None:
        at.selectbox(key="result_view_burger").set_value(select)
        at.run()
    assert not at.exception
    return at


def test_a_waiting_formulation_offers_its_missing_value_first(one_waiting):
    at = _results(4)
    assert "Overall score waiting for Juiciness." in [c.value for c in at.caption]
    box = at.number_input(key="missing_4_Juiciness")
    assert box.value is None
    save = _submit_button(at, "Save Juiciness")
    assert save.proto.type == "primary"
    move = _submit_button(at, "Move to Not scored")
    assert move.proto.type == "secondary"
    # One lit button: the foot steps aside while the panel is on screen.
    assert _submit_button(at, wording.START_NEXT_BATCH).proto.type == "secondary"
    assert not [b for b in at.button if b.label.startswith("Score without")]


def test_saving_the_missing_value_scores_it_and_the_box_follows_the_new_best(one_waiting):
    at = _results(4)
    at.number_input(key="missing_4_Juiciness").set_value(7.0)
    at.run()
    _submit_button(at, "Save Juiciness").click()
    at.run()
    assert not at.exception
    flashes = [s.value for s in at.success]
    assert "Formulation 4 scored. " + wording.best_moved(2, 4) in flashes, flashes
    reopened = FoodOptimizer("burger")
    assert reopened.fully_measured(reopened.index_of_formulation(4))
    assert reopened.best_formulation_no() == 4
    # The stale label bug: the box is sent its new Best so far label.
    box = at.selectbox(key="result_view_burger")
    assert box.value == ui_results.BEST_CHOICE
    assert box.proto.set_value
    assert box.proto.options[0] == wording.result_view_option(4, 2, best=True)


def test_an_empty_save_is_refused(one_waiting):
    at = _results(4)
    _submit_button(at, "Save Juiciness").click()
    at.run()
    assert [e.value for e in at.error] == [wording.ENTER_A_MEASUREMENT]
    assert not FoodOptimizer("burger").fully_measured(3)


def test_move_to_not_scored_asks_first_and_keeps_the_row(one_waiting):
    at = _results(4)
    _submit_button(at, "Move to Not scored").click()
    at.run()
    assert [w.value for w in at.warning] == [wording.move_to_not_scored_warning(4)]
    assert FoodOptimizer("burger").set_aside == []
    _submit_button(at, wording.YES_CONTINUE).click()
    at.run()
    assert not at.exception
    assert "Formulation 4 moved to Not scored." in [s.value for s in at.success]
    reopened = FoodOptimizer("burger")
    assert reopened.set_aside == [4]
    assert reopened.results_history[reopened.index_of_formulation(4)] == {"Firmness": 4.2}
    at.selectbox(key="result_view_burger").set_value(4)
    at.run()
    assert "Not scored · Juiciness not measured." in [c.value for c in at.caption]
    assert _submit_button(at, "Save Juiciness").proto.type == "primary"
    assert _submit_button(at, wording.WAIT_AGAIN).proto.type == "secondary"
    assert not [b for b in at.button if b.label == "Move to Not scored"]


def test_wait_for_its_results_again_undoes_it(one_waiting):
    one_waiting.set_aside_formulation(4)
    at = _results(4)
    _submit_button(at, wording.WAIT_AGAIN).click()
    at.run()
    assert "Formulation 4 is waiting for Juiciness again." in [s.value for s in at.success]
    assert FoodOptimizer("burger").set_aside == []


def test_the_panel_steps_aside_while_another_confirmation_is_open(one_waiting):
    at = _results(4)
    at.multiselect(key="delete_formulations").set_value([1])
    at.run()
    _submit_button(at, "Delete Formulation 1").click()
    at.run()
    assert _submit_button(at, "Save Juiciness").disabled
    move = _submit_button(at, "Move to Not scored")
    assert move.disabled
    move.click()
    at.run()
    assert FoodOptimizer("burger").set_aside == []


def test_a_set_aside_formulation_is_not_counted_as_waiting_on_make_a_round(one_waiting):
    one_waiting.set_aside_formulation(4)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.session_state["main_tab"] = wording.TAB_BATCH
    at.run()
    assert not at.exception
    assert not any("waiting for" in c.value for c in at.caption), \
        [c.value for c in at.caption]


def test_the_view_box_follows_best_so_far_when_it_moves(one_waiting):
    """UX review minor 11: the box stood on Best so far's entry while Best
    so far moved, and kept the old label. Its value is sent again."""
    at = _results()
    assert at.selectbox(key="result_view_burger").value == ui_results.BEST_CHOICE
    opt = at.session_state["optimizer"]
    opt.edit_result(opt.index_of_formulation(4), {"Firmness": 4.2, "Juiciness": 7.0})
    at.session_state["main_tab"] = wording.TAB_RESULTS
    at.run()
    box = at.selectbox(key="result_view_burger")
    assert box.proto.set_value
    assert box.proto.options[0] == wording.result_view_option(4, 2, best=True)
    assert box.value == ui_results.BEST_CHOICE
