"""0.8.0 review fixes, the final wave: the whole-branch review's items
(final-review.md I-1..I-3, M-1..M-5) and the parked minors."""
import pytest
from streamlit.testing.v1 import AppTest

import ui_insights
import wording
from food_bo import FoodOptimizer
from tests.test_app_ui import APP_PATH, _submit_button, burger, scored  # noqa: F401
from tests.test_complete_results_ui import one_waiting  # noqa: F401  (fixture)
from tests.test_complete_results import _patty, _tell
from tests.test_set_aside_ui import trial_waiting  # noqa: F401  (fixture)


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


# ------------------------------------------------------------------ #
#  I-1 the header's latest Round is the latest scored Round
# ------------------------------------------------------------------ #

def test_the_headline_names_the_latest_scored_round_while_a_later_one_waits():
    """Rounds 1-2 scored, Round 3 all waiting: the headline is about Round 2
    and counts only the scored formulations."""
    opt = _patty(rows=6)
    before = ui_insights.round_view(opt)
    for no in (7, 8, 9):
        _tell(opt, no, 3, drop=("Cook loss",))
    view = ui_insights.round_view(opt)
    assert view.round == 2 and view.scored == before.scored == [0, 1, 2, 3, 4, 5]
    assert ui_insights.headline_text(opt, view) == ui_insights.headline_text(opt, before)
    assert "Round 3" not in ui_insights.headline_text(opt, view)


def test_the_headline_leaves_out_a_round_moved_to_not_scored():
    opt = _patty(rows=6)
    before = ui_insights.headline_text(opt, ui_insights.round_view(opt))
    for no in (7, 8, 9):
        _tell(opt, no, 3, drop=("Cook loss",))
        opt.set_aside_formulation(no)
    view = ui_insights.round_view(opt)
    assert view.round == 2 and view.scored == [0, 1, 2, 3, 4, 5]
    assert ui_insights.headline_text(opt, view) == before


# ------------------------------------------------------------------ #
#  M-5 every formulation waiting: no "0 ... more", no empty question
# ------------------------------------------------------------------ #


# ------------------------------------------------------------------ #
#  I-2 Best so far's help in a process-only project
# ------------------------------------------------------------------ #

def test_best_so_far_help_names_trials_in_a_process_only_project(trial_waiting):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.session_state["main_tab"] = wording.TAB_RESULTS
    at.run()
    assert not at.exception
    captions = [c.value for c in at.tabs[2].caption]      # Results
    assert wording.for_project(trial_waiting, wording.BEST_RULE) in captions, captions
    assert not any("formulation" in c.lower() for c in captions), captions


def test_best_so_far_help_without_every_measurement_names_trials(trial_waiting):
    trial_waiting.add_objective("Colour", 1.0, goal="max", min_val=0, max_val=10)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.session_state["main_tab"] = wording.TAB_RESULTS
    at.run()
    assert not at.exception
    captions = [c.value for c in at.tabs[2].caption]      # Results
    assert wording.for_project(trial_waiting, wording.BEST_RULE_WITHOUT) in captions
    assert not any("formulation" in c.lower() for c in captions), captions


# ------------------------------------------------------------------ #
#  I-3 a typed, unsaved missing value stays in its own project
# ------------------------------------------------------------------ #

def test_an_unsaved_missing_value_does_not_follow_you_to_another_project(one_waiting):
    """Formulation 4 waits for Juiciness in both projects (a saved copy is
    the likely case). 12 typed but not saved in the first must not sit in
    the second's box behind a lit Save Juiciness (final review I-3)."""
    second = FoodOptimizer("second")
    second.set_amount_unit("g")
    second.add_ingredient("Pea protein", 0, 25)
    second.add_ingredient("Methylcellulose", 0, 3)
    second.add_objective("Juiciness", 1.0, goal="target", target=7,
                         min_val=0, max_val=10, unit="/10")
    second.add_objective("Firmness", 1.5, goal="target", target=6,
                         min_val=0, max_val=10, unit="N")
    # Its one formulation waits, so Results opens straight on it, in the
    # very run the project is opened: the box is drawn before Streamlit
    # could drop the first project's value as an unrendered widget's.
    second.tell({"Pea protein": 12.0, "Methylcellulose": 1.2},
                {"Firmness": 4.0}, formulation_no=4, batch_no=1)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "burger"
    at.session_state["main_tab"] = wording.TAB_RESULTS
    at.run()
    at.selectbox(key="result_view_burger").set_value(4)
    at.run()
    at.number_input(key="missing_4_Juiciness").set_value(12.0)
    at.run()
    at.sidebar.selectbox(key="project_select").select("second")
    at.run()
    _submit_button(at.sidebar, "Open").click()
    at.run()
    assert not at.exception
    box = at.number_input(key="missing_4_Juiciness")
    # Sent empty, and sent as a value: the browser does not keep the 12.
    assert box.value is None and box.proto.set_value
    # And nothing was written into the second project.
    reopened = FoodOptimizer("second")
    assert reopened.results_history[reopened.index_of_formulation(4)].get("Juiciness") is None


def _reset_prefixes():
    """app.py's _PER_NAME_BOX_PREFIXES, read without running the app."""
    import ast
    import pathlib
    tree = ast.parse((pathlib.Path(APP_PATH)).read_text())
    for node in tree.body:
        if (isinstance(node, ast.Assign)
                and getattr(node.targets[0], "id", None) == "_PER_NAME_BOX_PREFIXES"):
            return ast.literal_eval(node.value)
    raise AssertionError("_PER_NAME_BOX_PREFIXES not found in app.py")


def test_the_missing_value_boxes_are_parked_empty_on_a_project_switch():
    """The boxes are named after the project's own formulations and
    measurements, so the reset finds them by prefix, like past_ and
    correct_amount_ (final review I-3)."""
    import ui_results
    assert ui_results._missing_key(4, "Juiciness").startswith(_reset_prefixes())


# ------------------------------------------------------------------ #
#  M-1 Make a round's waiting caption: singular, and trials
# ------------------------------------------------------------------ #

def test_the_waiting_caption_says_trial_in_a_process_only_project(trial_waiting):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.session_state["main_tab"] = wording.TAB_BATCH
    at.run()
    assert not at.exception
    captions = [c.value for c in at.tabs[1].caption]      # Make a round
    assert ("1 trial is waiting for Aroma (enter it on Results). The model has not used it "
            "yet, so a suggestion may nearly repeat it.") in captions, captions


# ------------------------------------------------------------------ #
#  M-2 the luck line compares the numbers as they are shown
# ------------------------------------------------------------------ #

# ------------------------------------------------------------------ #
#  M-3 / M-4 a formulation moved to Not scored in Correct a formulation
# ------------------------------------------------------------------ #

def _results_tab():
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "burger"
    at.session_state["main_tab"] = wording.TAB_RESULTS
    at.run()
    assert not at.exception
    return at


def test_the_correction_picker_marks_a_set_aside_formulation_not_scored(one_waiting):
    """Formulation 3 was never recorded; 4 was, and was moved to Not
    scored. Both read "· Not scored", as the View box and the table say;
    4 still opens as the recorded row it is."""
    one_waiting.set_aside_formulation(4)
    at = _results_tab()
    box = at.selectbox(key="correct_formulation")
    # As the View formulation box reads them (polish round I8).
    assert [o.split(" (")[0] for o in box.options] == [
        "Formulation 1", "Best so far · Formulation 2", "Formulation 3",
        "Formulation 4"], box.options
    assert [o.endswith(" · Not scored") for o in box.options] == [
        False, False, True, True], box.options
    box.set_value(4)
    at.run()
    assert not at.exception
    keys = [n.key for n in at.number_input]
    assert "correct_amount_4_Pea protein" in keys and "correct_4_Firmness" in keys, keys


def test_correcting_a_set_aside_formulation_flashes_not_scored(one_waiting):
    one_waiting.set_aside_formulation(4)
    at = _results_tab()
    at.selectbox(key="correct_formulation").set_value(4)
    at.run()
    at.text_input(key="correct_note_4").set_value("second bowl")
    at.run()
    _submit_button(at, "Save correction").click()
    at.run()
    assert not at.exception
    note = next(s.value for s in at.success if "corrected" in s.value)
    assert note == ("Formulation 4 corrected · Not scored · Juiciness not measured. "
                    + wording.COPY_KEPT), note


# ------------------------------------------------------------------ #
#  Task 3 N-1 a stale set_aside number in a file is pruned on import
# ------------------------------------------------------------------ #

def test_import_prunes_a_stale_set_aside_number():
    """A hand-edited file names Formulation 3, which has all its results.
    Left in memory, blanking one of 3's values later flagged it Not scored
    instead of waiting."""
    import json
    opt = _patty(rows=6)
    _tell(opt, 7, 3, drop=("Cook loss",))
    opt.set_aside_formulation(7)
    state = json.loads(json.dumps(opt.export_json()))
    state['set_aside'] = [3, 7]
    other = FoodOptimizer("hand_edited")
    other.import_json(state)
    assert other.set_aside == [7]
    i = other.index_of_formulation(3)
    other.edit_result(i, {k: v for k, v in other.results_history[i].items()
                          if k != "Cook loss"})
    assert not other.is_set_aside(i)
    assert i in other.awaited_rows()


def test_the_changed_strings_pass_the_vocabulary_guard():
    from tests.test_food_bo import _BANNED
    for text in (wording.formulation_corrected_not_scored(7, "Cook loss"),
                 wording.formulation_corrected_not_scored(7, ""),
                 wording.NO_COMPLETE_FORMULATION):
        assert not any(p.search(text) for p in _BANNED), text
    assert wording.formulation_corrected_not_scored(7, "") == "Formulation 7 corrected · Not scored."
