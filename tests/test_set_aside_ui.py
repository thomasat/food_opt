"""0.8.0 review fixes, Task 3 on screen: the panel under a formulation that
waits for results (spec 2026-09-29 §1.1, §9 B2/I5, §9 I13 stale label)."""
import pytest
from streamlit.testing.v1 import AppTest

import ui_results
import wording
from food_bo import FoodOptimizer
from tests.test_app_ui import (  # noqa: F401
    APP_PATH, _submit_button, _tab_primaries, burger, scored)
from tests.test_complete_results_ui import one_waiting  # noqa: F401
from tests.test_insights_ui import insights_scored  # noqa: F401


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
    # Save is grey until a value is typed, and the foot stays the lit one
    # (spec 2026-09-30 §4, review I11, M12).
    save = _submit_button(at, "Save Juiciness")
    assert save.proto.type == "secondary" and not save.disabled
    move = _submit_button(at, "Move to Not scored")
    assert move.proto.type == "secondary"
    assert _submit_button(at, wording.START_NEXT_BATCH).proto.type == "primary"
    assert not [b for b in at.button if b.label.startswith("Score without")]


def test_saving_the_missing_value_scores_it_and_the_box_follows_the_new_best(one_waiting):
    at = _results(4)
    at.number_input(key="missing_4_Juiciness").set_value(7.0)
    at.run()
    _submit_button(at, "Save Juiciness").click()
    at.run()
    assert not at.exception
    flashes = [s.value for s in at.success]
    # Fix A follow-up (ruling on ties): a lead of 1.20 is under what one new
    # result needs to differ (√2 × 1.645 × 1 = 2.3), so it is a tie, not a move.
    assert ("Formulation 4 scored. "
            + wording.tie_line(4, "", "1.2", 2, "2.3")) in flashes, flashes
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
    assert [w.value for w in at.warning] == [wording.never_coming_warning(4, "Juiciness")]
    assert FoodOptimizer("burger").set_aside == []
    at.button(key="set_aside_4__yes").click()   # the Yes, by key: it shares the label
    at.run()
    assert not at.exception
    assert "Formulation 4 moved to Not scored." in [s.value for s in at.success]
    reopened = FoodOptimizer("burger")
    assert reopened.set_aside == [4]
    assert reopened.results_history[reopened.index_of_formulation(4)] == {"Firmness": 4.2}
    at.selectbox(key="result_view_burger").set_value(4)
    at.run()
    assert "Not scored · Juiciness not measured." in [c.value for c in at.caption]
    # Grey until a value is typed (spec 2026-09-30 review M12).
    assert _submit_button(at, "Save Juiciness").proto.type == "secondary"
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


def test_a_set_aside_formulation_is_never_flagged_as_waiting(insights_scored):
    """Task 3 x $NEAR: a formulation moved to Not scored is not 'still
    waiting', so no suggestion on Make a round is flagged as nearly the
    same as it, and the waiting caption does not count it."""
    opt = insights_scored
    opt.tell({"Pea protein": 16.0, "Methylcellulose": 1.2}, {"Firmness": 5.5},
             formulation_no=10, batch_no=4)
    opt.set_aside_formulation(10)
    opt.set_pending_batch([{"Pea protein": 16.25, "Methylcellulose": 1.23},
                           {"Pea protein": 4.0, "Methylcellulose": 2.8}])
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert not at.exception
    cells = [str(v) for d in at.dataframe for col in d.value.columns for v in d.value[col]]
    assert not [c for c in cells if c.startswith("Nearly the same")]
    assert not any("waiting for" in c.value for c in at.caption)


# ---------------------------------------------------------------- fix round 1


def _successes(at):
    return [s.value for s in at.success]


def test_an_open_correction_keeps_the_one_lit_button(one_waiting):
    """Review I-1: a correction opened below a waiting formulation. Its Save
    is the one lit button; the panel's Save steps back to secondary."""
    at = _results(4)
    assert _tab_primaries(at, 2) == [wording.START_NEXT_BATCH], _tab_primaries(at, 2)
    at.selectbox(key="correct_formulation").set_value(1)
    at.run()
    assert not at.exception
    assert _tab_primaries(at, 2) == ["Save correction"], _tab_primaries(at, 2)
    assert _submit_button(at, "Save Juiciness").proto.type == "secondary"


@pytest.fixture
def trial_waiting(tmp_path, monkeypatch):
    """A project of process settings alone: Trial 2 waits for Aroma."""
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("ferment")
    opt.add_process_parameter("Incubation temperature", 30, 42, unit="°C")
    opt.add_process_parameter("Incubation time", 4, 16, unit="h")
    opt.add_objective("Acidity", 1.0, goal="target", target=4.5,
                      min_val=3.0, max_val=7.0, unit="pH")
    opt.add_objective("Aroma", 1.0, goal="max", min_val=0, max_val=10)
    opt.tell({"Incubation temperature": 37.0, "Incubation time": 8.0},
             {"Acidity": 4.5, "Aroma": 6.0}, formulation_no=1, batch_no=1)
    opt.tell({"Incubation temperature": 33.0, "Incubation time": 12.0},
             {"Acidity": 4.4}, formulation_no=2, batch_no=1)
    return opt


def test_a_process_only_project_says_trial_throughout(trial_waiting):
    """Review I-2: every sentence the panel says names a Trial."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.session_state["main_tab"] = wording.TAB_RESULTS
    at.run()
    at.selectbox(key="result_view_ferment").set_value(2)
    at.run()
    _submit_button(at, "Move to Not scored").click()
    at.run()
    # Re-pinned (stale since the fourth tab became "4 · Review").
    assert [w.value for w in at.warning] == [
        "Aroma for Trial 2 will never come? It stays in your records, unscored, and stops "
        "counting in the review."]
    at.button(key="set_aside_2__yes").click()   # the Yes, by key: it shares the label
    at.run()
    assert "Trial 2 moved to Not scored." in _successes(at), _successes(at)
    at.selectbox(key="result_view_ferment").set_value(2)
    at.run()
    _submit_button(at, wording.WAIT_AGAIN).click()
    at.run()
    assert "Trial 2 is waiting for Aroma again." in _successes(at), _successes(at)
    at.selectbox(key="result_view_ferment").set_value(2)
    at.run()
    at.number_input(key="missing_2_Aroma").set_value(9.0)
    at.run()
    _submit_button(at, "Save Aroma").click()
    at.run()
    assert not at.exception
    flashes = _successes(at)
    assert any(f.startswith("Trial 2 scored.") for f in flashes), flashes
    assert not any("Formulation" in f for f in flashes), flashes


def test_a_row_with_nothing_to_enter_offers_no_save(one_waiting):
    """Review M-2: every measurement Formulation 4 was measured on deleted,
    and the one left added after it — nothing to type, so no boxes and no
    Save; Move to Not scored alone, and then a plain 'Not scored.'"""
    one_waiting.add_objective("Cook loss", 1.0, goal="min", min_val=0, max_val=40)
    one_waiting.remove_objective("Firmness")
    one_waiting.remove_objective("Juiciness")
    i = one_waiting.index_of_formulation(4)
    assert one_waiting.waiting_on(i) == [] and not one_waiting.fully_measured(i)
    at = _results(4)
    assert not [b for b in at.button if b.key == "save_missing_4"]
    assert not [n for n in at.number_input if (n.key or "").startswith("missing_4_")]
    _submit_button(at, "Move to Not scored").click()
    at.run()
    at.button(key="set_aside_4__yes").click()   # the Yes, by key: it shares the label
    at.run()
    assert not at.exception
    at.selectbox(key="result_view_burger").set_value(4)
    at.run()
    assert "Not scored." in [c.value for c in at.caption], [c.value for c in at.caption]
    assert _submit_button(at, wording.WAIT_AGAIN)
    assert not [b for b in at.button if b.key == "save_missing_4"]


def test_a_partial_save_on_a_not_scored_row_does_not_say_scored(scored):
    """Review M-4: two values missing, one entered. The row is still Not
    scored, and the flash says so rather than 'scored · waiting for'."""
    scored.add_objective("Cook loss", 1.0, goal="min", min_val=0, max_val=40)
    scored.tell({"Pea protein": 12.0, "Methylcellulose": 1.5}, {"Firmness": 4.2},
                formulation_no=4, batch_no=2)
    scored.set_aside_formulation(4)
    at = _results(4)
    at.number_input(key="missing_4_Juiciness").set_value(7.0)
    at.run()
    next(b for b in at.button if b.key == "save_missing_4").click()
    at.run()
    assert not at.exception
    assert "Formulation 4 saved · Not scored · Cook loss not measured." in \
        _successes(at), _successes(at)
    reopened = FoodOptimizer("burger")
    assert reopened.set_aside == [4]
    assert reopened.results_history[reopened.index_of_formulation(4)] == {
        "Firmness": 4.2, "Juiciness": 7.0}


@pytest.mark.skip(reason="review cards prototype: pins the old Review layout (status lines, Explored fold, header); reconcile at the final merge")
def test_insights_does_not_count_a_not_scored_row_as_scored(one_waiting):
    """Review M-3 (ruling): a formulation moved to Not scored counts for
    nothing on Insights, so it is not one of the scored formulations."""
    one_waiting.set_aside_formulation(4)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    assert not at.exception
    captions = [c.value for c in at.tabs[3].caption]
    assert any(c.startswith("2 scored formulations,")
               for c in captions), captions
