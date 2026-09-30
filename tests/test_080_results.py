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
    at = _results()
    assert [h.value for h in at.tabs[2].subheader] == []
    at = _results(1)
    assert [h.value for h in at.tabs[2].subheader] == []


def test_the_score_is_the_first_line_under_the_box_with_its_help(scored):
    at = _results()
    opt = FoodOptimizer("burger")
    first = at.tabs[2].caption[0]
    assert first.value == wording.overall_score_line(
        float(opt.Y_history[opt.best_index()]), opt.utility_ceiling())
    assert first.proto.help == wording.OVERALL_SCORE_HELP


def test_one_rule_line_and_the_other_two_are_gone(scored):
    said = _captions(_results())
    assert wording.BEST_RULE in said
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
    assert wording.COMPLETE_RESULTS_CAPTION not in said            # a waiting cell is no dash
    one_waiting.add_objective("Aroma", 1.0, goal="max", min_val=0, max_val=10)
    reloaded = FoodOptimizer("burger")
    assert any(reloaded.not_expected(i) for i in range(len(reloaded.Y_history)))
    said = _captions(_results())
    assert any("“—”" in c for c in said)


# ------------------------------------------------------------------ #
#  Ties (§4, review M18)
# ------------------------------------------------------------------ #

def test_a_lead_within_one_point_is_a_tie(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = _patty(rows=3)
    _tell(opt, 4, 2)
    old = max(range(3), key=lambda i: opt.Y_history[i])
    opt.Y_history[3] = opt.Y_history[old] + 0.2
    line = ui_results._progress_line(opt)
    assert line == wording.tie_line(4, f"{opt.Y_history[3]:.2f}", "0.20",
                                    int(opt.formulation_ids[old]))
    assert ui_results._move_sentence(opt, int(opt.formulation_ids[old]), 4) == line


def test_a_lead_beyond_the_yardstick_is_a_move(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = _patty(rows=3)
    _tell(opt, 4, 2)
    old = max(range(3), key=lambda i: opt.Y_history[i])
    opt.Y_history[3] = opt.Y_history[old] + 3.0
    assert ui_results._progress_line(opt).startswith("Round 2 recorded · Best")
    assert ui_results._move_sentence(opt, int(opt.formulation_ids[old]), 4) == (
        wording.best_moved(int(opt.formulation_ids[old]), 4))


def test_the_tie_uses_the_repeats_from_three_pairs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = _patty(rows=4)
    monkeypatch.setattr(insights, "noise_yardstick",
                        lambda o, checks=(), name=insights.SCORE:
                        insights.Yardstick(2.0, 2.0, 4, True))
    old = max(range(3), key=lambda i: opt.Y_history[i])
    opt.Y_history[3] = opt.Y_history[old] + 1.5
    opt.batch_history[3] = 2
    assert ui_results._tie(opt, old, 3).endswith(
        "within how much Repeats differ (2.0), so treat them as equal.")


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


def test_one_fold_for_past_formulations_and_no_second_door(scored):
    at = _results()
    labels = [e.label for e in at.tabs[2].expander]
    assert wording.EDIT_PAST_FOLD in labels
    assert "Edit past formulations" not in labels
    assert "Correct a recorded formulation" not in [b.label for b in at.tabs[2].button]
