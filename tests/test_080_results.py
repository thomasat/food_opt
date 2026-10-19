"""0.8.0 speed and simplicity, Plan B, Task 8: Results says Best so far
once — no heading under the box, the score first, one rule line, a tie
said as a tie — and a waiting result is told apart from one that will
never come (spec 2026-09-30 §4; review I9–I13, M7, M12, M18)."""
import pytest
from streamlit.testing.v1 import AppTest

import insights
import ui_results
import wording
from food_bo import FoodOptimizer
from tests.test_app_ui import APP_PATH, _submit_button, burger, scored  # noqa: F401
from tests.test_complete_results import _patty, _tell
from tests.test_complete_results_ui import one_waiting  # noqa: F401


def _results(select=None, project="burger"):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = project
    at.run()
    at.session_state["main_tab"] = wording.TAB_RESULTS
    at.run()
    if select is not None:
        at.selectbox(key=f"result_view_{project}").set_value(select)
        at.run()
    assert not at.exception
    return at


def _captions(at):
    return [c.value for c in at.tabs[2].caption]


# ------------------------------------------------------------------ #
#  Best so far said once (V7–V10, V30; review I10, I12)
# ------------------------------------------------------------------ #

def test_no_heading_repeats_the_box(one_waiting):
    # Best so far, another scored one, a Not scored one (its own details
    # path) and a waiting one.
    for select in (None, 1, 3, 4):
        at = _results(select)
        assert [h.value for h in at.tabs[2].subheader] == [], select


def test_the_score_is_the_first_line_under_the_box_with_its_help(scored):
    at = _results()
    opt = FoodOptimizer("burger")
    # v3: after Round N results' caption and its fold's line, the first
    # line under the View formulation box.
    captions = at.tabs[2].caption
    assert captions[0].value == ui_results.results_key_markup(ui_results.round_results(opt))
    first = captions[2]
    assert first.value == wording.overall_score_line(
        float(opt.Y_history[opt.best_index()]), opt.utility_ceiling())
    assert first.proto.help == wording.OVERALL_SCORE_HELP


def test_one_rule_line_and_the_other_two_are_gone(scored):
    said = _captions(_results())
    assert said.count(wording.BEST_RULE) == 1
    # The three old rule captions, by their text: Task 9 deletes the names.
    assert not any(c.startswith(("Best so far has the highest overall score",
                                 "Only formulations with all their results"))
                   for c in said)


def test_the_rule_line_says_when_none_has_every_measurement(scored):
    scored.add_objective("Aroma", 1.0, goal="max", min_val=0, max_val=10)
    said = _captions(_results())
    assert wording.BEST_RULE_WITHOUT in said and wording.BEST_RULE not in said


def test_the_dash_caption_only_with_a_dash_on_screen(one_waiting):
    said = _captions(_results())
    assert wording.complete_results_caption() not in said            # a waiting cell is no dash
    one_waiting.add_objective("Aroma", 1.0, goal="max", min_val=0, max_val=10)
    reloaded = FoodOptimizer("burger")
    assert any(reloaded.not_expected(i) for i in range(len(reloaded.Y_history)))
    said = _captions(_results())
    assert any("“—”" in c for c in said)


# ------------------------------------------------------------------ #
#  Ties (§4, review M18)
# ------------------------------------------------------------------ #

def test_a_small_lead_is_a_move_and_the_next_best_is_measured(tmp_path, monkeypatch):
    """v3 final cleanup: ties are equal scores only, with no tasting noise
    on screen; a 0.2 lead is a move, and the next best reads as measured."""
    monkeypatch.chdir(tmp_path)
    opt = _patty(rows=3)
    _tell(opt, 4, 2)
    old = max(range(3), key=lambda i: opt.Y_history[i])
    opt.Y_history[3] = opt.Y_history[old] + 0.2
    old_no = int(opt.formulation_ids[old])
    assert ui_results._progress_line(opt).startswith("Round 2 recorded · Best")
    assert ui_results._move_sentence(opt, old_no, 4) == wording.best_moved(old_no, 4)
    assert ui_results._runner_up_line(opt) == wording.runner_up_line(
        old_no, wording.score_shown(opt.Y_history[old]), "0.2")


def test_a_lead_beyond_the_yardstick_is_a_move(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = _patty(rows=3)
    _tell(opt, 4, 2)
    old = max(range(3), key=lambda i: opt.Y_history[i])
    opt.Y_history[3] = opt.Y_history[old] + 3.0
    assert ui_results._progress_line(opt).startswith("Round 2 recorded · Best")
    assert ui_results._move_sentence(opt, int(opt.formulation_ids[old]), 4) == (
        wording.best_moved(int(opt.formulation_ids[old]), 4))


def test_a_late_value_that_moves_best_so_far_is_said_under_it(tmp_path, monkeypatch):
    """Review M18 (Task 8 I-3): Formulation 1 scores 70.00 in Round 1,
    Formulation 2 in Round 1 waits for Juiciness, Formulation 3 in Round 2
    scores lower. Juiciness for 2 makes it Best so far at 70.50: the flash
    says Best so far moved, and the lasting line names Formulation 1 half a
    point lower (v3 final cleanup: no "within noise" tie)."""
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("late")
    opt.add_ingredient("Pea", 0, 10)
    opt.add_objective("Firmness", 1.0, goal="max", min_val=0, max_val=10)
    opt.add_objective("Juiciness", 1.0, goal="max", min_val=0, max_val=10)
    opt.tell({"Pea": 1.0}, {"Firmness": 7.0, "Juiciness": 7.0}, formulation_no=1, batch_no=1)
    opt.tell({"Pea": 2.0}, {"Firmness": 7.1}, formulation_no=2, batch_no=1)
    opt.tell({"Pea": 3.0}, {"Firmness": 5.0, "Juiciness": 5.0}, formulation_no=3, batch_no=2)
    at = _results(2, project="late")
    at.number_input(key="missing_2_Juiciness").set_value(7.0)
    at.run()
    _submit_button(at, "Save Juiciness").click()
    at.run()
    assert not at.exception
    flashes = [f.value for f in at.success]
    assert "Formulation 2 scored. " + wording.best_moved(1, 2) in flashes, flashes
    assert sum(f.count("Formulation 2 scored") for f in flashes) == 1, flashes
    # The box is back on Best so far, and the line under it names the next.
    assert FoodOptimizer("late").best_formulation_no() == 2
    line = wording.runner_up_line(1, "70.0", "0.5")
    said = _captions(at)
    assert line in said, said
    assert not any("noise" in c or "needed" in c for c in said), said
    # ...and still says it on the next visit, with the flash long gone.
    assert line in _captions(_results(project="late"))


@pytest.fixture
def trial_tie(tmp_path, monkeypatch):
    """Process settings alone: Trial 1 (Round 1) and Trial 2 (Round 2) half
    a point apart, Trial 3 Not scored."""
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("ferment")
    opt.add_process_parameter("Incubation temperature", 30, 42, unit="°C")
    opt.add_process_parameter("Incubation time", 4, 16, unit="h")
    opt.add_objective("Aroma", 1.0, goal="max", min_val=0, max_val=10)
    opt.tell({"Incubation temperature": 37.0, "Incubation time": 8.0},
             {"Aroma": 6.0}, formulation_no=1, batch_no=1)
    opt.tell({"Incubation temperature": 33.0, "Incubation time": 12.0},
             {"Aroma": 6.05}, formulation_no=2, batch_no=2)
    opt.record_skipped(3, 2, {"Incubation temperature": 40.0, "Incubation time": 6.0})
    return opt


def test_a_process_only_tie_says_trial(trial_tie):
    """Task 8 review I-1: the line under Best so far goes through
    for_project."""
    said = _captions(_results(project="ferment"))
    # v3 final cleanup: the next best as measured, no tasting noise.
    assert "Trial 1: 60.0 (0.5 lower than the best so far)." in said, said
    assert not any("Formulation" in c for c in said), said


def test_scoring_a_not_scored_trial_flashes_trial(trial_tie):
    """Task 8 review I-1: the flash after scoring a Not scored row from the
    fold goes through for_project too, tie and all."""
    at = _results(project="ferment")
    at.selectbox(key="correct_formulation").set_value(3)
    at.run()
    at.number_input(key="correct_3_Aroma").set_value(6.1)
    at.run()
    _submit_button(at, wording.SAVE_RESULT_BUTTON).click()
    at.run()
    assert not at.exception
    flashes = [f.value for f in at.success]
    # v3 final cleanup: a 0.5 lead is a move (ties are equal scores only).
    moved = wording.for_project(trial_tie, wording.best_moved(2, 3))
    assert ("Trial 3 scored. " + moved + " " + wording.COPY_KEPT) in flashes, flashes
    assert not any("Formulation" in f for f in flashes), flashes


# ------------------------------------------------------------------ #
#  Waiting against never coming (§4; review I9, I11, M12)
# ------------------------------------------------------------------ #

def test_a_waiting_cell_reads_waiting_and_a_set_aside_one_not_measured(one_waiting):
    opt = FoodOptimizer("burger")
    i = opt.index_of_formulation(4)
    cells = {d['name'].split(" (")[0]: d['measured'] for d in opt.closeness_details(i)}
    assert cells["Juiciness"] == wording.MEASURED_WAITING == "waiting"
    opt.set_aside_formulation(4)
    cells = {d['name'].split(" (")[0]: d['measured'] for d in opt.closeness_details(i)}
    assert cells["Juiciness"] == wording.NOT_MEASURED


def test_the_picker_and_the_best_line_name_what_waits(one_waiting):
    at = _results()
    box = at.selectbox(key="result_view_burger")
    assert "Formulation 4 (Round 2) · waiting for Juiciness" in box.proto.options
    assert wording.waiting_list("Juiciness", [4], 2) in _captions(at)


def test_a_waiting_formulation_keeps_the_foot_lit_until_a_value_is_typed(one_waiting):
    at = _results(4)
    tab = at.tabs[2]
    save = _submit_button(tab, "Save Juiciness")
    foot = next(b for b in tab.button if b.key == "foot_batch")
    assert (save.proto.type, save.disabled) == ("secondary", False)
    assert (foot.proto.type, foot.disabled) == ("primary", False)
    assert wording.CHANGE_SETUP_FROM_RESULTS_BUTTON not in [b.label for b in tab.button]
    at.number_input(key="missing_4_Juiciness").set_value(7.0)
    at.run()
    tab = at.tabs[2]
    save = _submit_button(tab, "Save Juiciness")
    foot = next(b for b in tab.button if b.key == "foot_batch")
    assert save.proto.type == "primary"
    assert (foot.proto.type, foot.disabled) == ("secondary", False)


def test_move_to_not_scored_says_what_it_does(one_waiting):
    at = _results(4)
    move = _submit_button(at, wording.MOVE_TO_NOT_SCORED)
    assert move.proto.help == wording.MOVE_TO_NOT_SCORED_HELP
    move.click()
    at.run()
    assert [w.value for w in at.warning] == [wording.never_coming_warning(4, "Juiciness")]
    assert "suggestions" not in at.warning[0].value              # review I9
    # The Yes names the act, as every other destructive confirmation does.
    yes = at.button(key="set_aside_4__yes")
    # 01 O13: not a second "Move to Not scored" beside the one that opened it.
    assert yes.label == wording.YES_MOVE_TO_NOT_SCORED == "Yes, move to Not scored"
    assert yes.proto.type == "primary"
    labels = [b.label for b in at.tabs[2].button]
    assert labels.count(wording.MOVE_TO_NOT_SCORED) == 1
    assert wording.YES_CONTINUE not in [b.label for b in at.tabs[2].button]


def test_one_fold_for_past_formulations_and_no_second_door(scored):
    at = _results()
    labels = [e.label for e in at.tabs[2].expander]
    assert wording.EDIT_PAST_FOLD in labels
    assert "Edit past formulations" not in labels
    assert "Correct a recorded formulation" not in [b.label for b in at.tabs[2].button]


# ------------------------------------------------------------------ #
#  Plan B parked: every Results flash in the project's own words
# ------------------------------------------------------------------ #

def _click(at, label):
    """The button reading `label`, in either form."""
    ours = [b for b in at.button if b.label == label]
    if not ours:
        ours = [b for b in at.button if b.label.replace("Trial", "Formulation")
                .replace("trial", "formulation") == label]
    ours[0].click()
    at.run()
    return at


def _no_formulation(at):
    said = [s.value for s in at.success] + [w.value for w in at.warning]
    assert not any("ormulation" in s for s in said), said
    return said


def test_an_unchanged_correction_says_trial(trial_tie):
    at = _results(project="ferment")
    at.selectbox(key="correct_formulation").set_value(1)
    at.run()
    _click(at, wording.SAVE_CORRECTION_BUTTON)
    assert not at.exception
    assert "Trial 1 is unchanged." in _no_formulation(at)


def test_a_deleted_trial_says_trial(trial_tie):
    at = _results(project="ferment")
    at.multiselect(key="delete_formulations").set_value([1])
    at.run()
    _click(at, wording.delete_formulation_button(1))
    _click(at, wording.YES_DELETE)
    assert not at.exception
    said = _no_formulation(at)
    assert any(s.startswith("Trial 1 deleted.") for s in said), said


def test_a_typed_past_trial_says_trial(trial_tie):
    at = _results(project="ferment")
    at.number_input(key="past_Incubation temperature").set_value(36.0)
    at.number_input(key="past_Incubation time").set_value(9.0)
    at.number_input(key="past_m_Aroma").set_value(5.0)
    at.run()
    _click(at, wording.ADD_THIS_FORMULATION)
    assert not at.exception
    assert "Trial 4 recorded." in _no_formulation(at)


def test_imported_trials_say_trials(trial_tie):
    import pandas as pd
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "ferment"
    at.session_state["main_tab"] = wording.TAB_RESULTS
    at.session_state["add_past_mode"] = wording.UPLOAD_A_FILE
    at.session_state["_import_rows"] = pd.DataFrame({
        "Incubation temperature": [36.0, 38.0], "Incubation time": [9.0, 10.0],
        "Aroma": [5.0, 5.5]})
    at.run()
    _click(at, wording.IMPORT_ALL_ROWS_BUTTON)
    assert not at.exception
    assert "Recorded 2 trials." in _no_formulation(at)


# ------------------------------------------------------------------ #
#  Answer-key read-through: Results' labels and words
# ------------------------------------------------------------------ #

def test_after_move_to_not_scored_the_box_is_sent_its_new_label(one_waiting):
    """The browser keeps the label it was shown: "· waiting for Juiciness"
    stayed above "Not scored". The value is sent again so the label is
    redrawn."""
    at = _results(4)
    _submit_button(at, wording.MOVE_TO_NOT_SCORED).click()
    at.run()
    at.button(key="set_aside_4__yes").click()
    at.run()
    assert not at.exception
    box = at.selectbox(key="result_view_burger")
    assert box.value == 4 and box.proto.set_value
    options = list(box.proto.options)
    assert "Formulation 4 (Round 2) · Not scored" in options
    assert not any("waiting" in o for o in options)
    at.run()                                                   # once, not every run
    assert not at.selectbox(key="result_view_burger").proto.set_value


def test_the_late_value_box_names_its_unit(one_waiting):
    one_waiting.objectives[[o["name"] for o in one_waiting.objectives].index("Juiciness")][
        "unit"] = "%"
    one_waiting.save()
    at = _results(4)
    box = at.number_input(key="missing_4_Juiciness")
    assert box.label.startswith("Juiciness (%) · ")


def test_the_add_fold_says_formulation_as_its_name_does(scored):
    at = _results()
    assert wording.ADD_PAST_FORMULATION_LABEL == "Add a past formulation"
    md = [m.value for m in at.tabs[2].markdown]
    assert "##### Add a past formulation" in md
    assert not any("Add a past result" in m for m in md)
