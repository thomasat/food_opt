"""0.8.0 speed and simplicity, Plan B, Task 3: Make a round with fewer
boxes and fewer captions — "too uncertain yet", Compared with third with
two changes, one "More" per formulation, one workbook line, lot numbers in
a fold, the first-five caption only before five, one waiting line, "Start
from Best so far" and a lit Back to Set up (spec 2026-09-30 §3; review
M2, M10, M11, M14; audit V3, V11–V17, V31)."""
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import insights
import ui_batch
import wording
from food_bo import FoodOptimizer
from tests.test_app_ui import APP_PATH, burger, open_batch, scored  # noqa: F401
from tests.test_complete_results import _patty, _tell


def _app():
    return AppTest.from_file(APP_PATH, default_timeout=180)


def _on_batch(at):
    at.session_state["main_tab"] = wording.TAB_BATCH
    at.run()
    return at


# ------------------------------------------------------------------ #
#  Compared with: third, at most two changes, screen only (review M14)
# ------------------------------------------------------------------ #

@pytest.fixture
def four(tmp_path, monkeypatch):
    """Four ingredients and six scored formulations: past the first five, so
    the table has a Compared with column and three changes to name."""
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("four")
    opt.set_amount_unit("g")
    for name in ("Pea", "Oil", "Salt", "Fibre"):
        opt.add_ingredient(name, 0, 10)
    opt.add_objective("Firmness", 1.0, goal="max", min_val=0, max_val=10)
    for no in range(1, 7):
        opt.tell({"Pea": no, "Oil": 1.0, "Salt": 1.0, "Fibre": 1.0},
                 {"Firmness": float(no)}, formulation_no=no, batch_no=1)
    opt.set_pending_batch([{"Pea": 2.0, "Oil": 5.0, "Salt": 3.0, "Fibre": 9.0}])
    return opt


def test_the_screen_names_two_changes_and_the_sheet_keeps_its_list(four):
    recipe = four.pending_batch[0]['recipe']
    full = four.vs_best_text(recipe)
    assert full.count(",") == 2                              # three changes
    assert four.vs_best_text(recipe, max_changes=2) == ", ".join(full.split(", ")[:2])
    trying = four.compared_with_column()
    screen = four.batch_frame(four.pending_batch, max_changes=2)[trying].iloc[0]
    sheet = four.batch_frame(four.pending_batch)[trying].iloc[0]
    assert screen.count(",") == 1 and sheet.count(",") == 2
    assert four.compared_with_text(recipe) == sheet


def test_compared_with_comes_straight_after_formulation(four):
    frame = ui_batch._compared_second(four, four.batch_frame(four.pending_batch))
    assert list(frame.columns)[:2] == [wording.FORMULATION_CAP, four.compared_with_column()]
    assert list(four.batch_frame(four.pending_batch).columns)[-1] == four.compared_with_column()


def test_the_round_table_on_screen_has_compared_with_second(four):
    at = _on_batch(_app())
    at.run()
    assert not at.exception
    frame = at.tabs[1].dataframe[0].value
    assert list(frame.columns)[:3] == [wording.FORMULATION_CAP, wording.PREDICTED_SCORE_COLUMN,
                                       four.compared_with_column()]
    # The screen names two of the three changes (review M14).
    assert frame[four.compared_with_column()].iloc[0].count(",") == 1


# ------------------------------------------------------------------ #
#  "too uncertain yet" (§3, review M10)
# ------------------------------------------------------------------ #

def test_a_likely_range_over_three_quarters_of_the_range_is_too_uncertain():
    opt = _patty(rows=0)
    preds = {7: {"Firmness": insights.Prediction(5.0, 1.0, 9.0),        # 8 of 10
                 "Juiciness": insights.Prediction(5.0, 3.0, 7.0),       # 4 of 10
                 "Cook loss": insights.Prediction(20.0, 0.0, 30.0)}}    # 30 of 40: not over
    cells, tips = ui_batch._measurement_cells(opt, preds)
    assert cells["Firmness"].iloc[0] == wording.TOO_UNCERTAIN
    # Round 2: the cell reads "—"; its tip keeps the numbers and says why.
    assert tips["Firmness"].iloc[0] == "95\u00a0% range over ¾ of the Range: 5.0 (1.0–9.0)."
    assert cells["Juiciness"].iloc[0] == "5.0 (3.0–7.0)" and tips["Juiciness"].iloc[0] == ""
    assert cells["Cook loss"].iloc[0] != wording.TOO_UNCERTAIN
    assert cells.shape == tips.shape and list(cells.columns) == list(tips.columns)


def test_a_too_uncertain_cell_keeps_its_numbers_in_the_tooltip(four, monkeypatch):
    """On screen, not only in the frame: the cell says "too uncertain yet"
    and its tooltip carries the numbers (planning ruling 6)."""
    no = four.pending_batch[0]['formulation']
    wide = {no: {insights.SCORE: insights.Prediction(5.0, 3.0, 7.0),
                 "Firmness": insights.Prediction(5.0, 1.0, 9.0)}}     # 8 of 10
    monkeypatch.setattr(ui_batch, "_round_predictions", lambda opt: wide)
    at = _on_batch(_app())
    assert not at.exception
    [fold] = [e for e in at.tabs[1].expander if e.label == wording.PREDICTED_MEASUREMENTS]
    table = fold.table[0]
    column = four._measurement_column(four.objectives[0])
    assert table.value[column].iloc[0] == wording.TOO_UNCERTAIN
    assert '5.0 (1.0–9.0)."' in table.proto.arrow_data.styler.styles   # round 2: with its reason


# ------------------------------------------------------------------ #
#  The record form: one "More" per formulation (V11, review M11)
# ------------------------------------------------------------------ #

def test_note_and_not_scored_sit_behind_one_more_per_formulation(open_batch):
    at = _app()
    at.run()
    assert not at.exception
    labels = [e.label for e in at.tabs[1].expander]
    assert labels.count(wording.MORE) == 2
    more = next(e for e in at.tabs[1].expander if e.label == wording.MORE)
    assert [t.key for t in more.text_input] == ["f1_note"]
    assert [c.key for c in more.checkbox] == ["f1_leave_out"]
    assert more.checkbox[0].help == wording.NOT_SCORED_TICK_HELP


def test_the_more_label_says_not_scored_and_the_draft_is_kept(open_batch):
    at = _app()
    at.run()
    at.text_input(key="f1_note").input("burnt")
    at.checkbox(key="f1_leave_out").check()
    at.run()
    assert not at.exception
    labels = [e.label for e in at.tabs[1].expander]
    assert wording.more_label(True) in labels
    assert FoodOptimizer("burger").result_drafts[1]["not_scored"] is True
    assert at.text_input(key="f1_note").value == "burnt"


def test_the_more_fold_opens_on_the_not_scored_tick(open_batch):
    """Review M11: a tick with no note still opens the fold, so the greyed
    boxes above it have a visible reason."""
    at = _app()
    at.run()
    at.checkbox(key="f1_leave_out").check()
    at.run()
    assert not at.exception
    fold = next(e for e in at.tabs[1].expander if e.label == wording.more_label(True))
    assert fold.proto.expanded
    assert not next(e for e in at.tabs[1].expander if e.label == wording.MORE).proto.expanded


def test_a_note_alone_shows_in_the_label(open_batch):
    at = _app()
    at.run()
    at.text_input(key="f2_note").input("extra salt")
    at.run()
    assert wording.more_label(False, True) in [e.label for e in at.tabs[1].expander]


def test_the_record_caption_says_blank_means_not_in_yet(open_batch):
    at = _app()
    at.run()
    captions = [c.value for c in at.tabs[1].caption]
    assert wording.RECORD_RESULTS_LINE in captions


# ------------------------------------------------------------------ #
#  Captions (V12–V14, V16, V17)
# ------------------------------------------------------------------ #

def test_one_workbook_line_that_follows_the_box(open_batch):
    at = _app()
    at.run()
    captions = [c.value for c in at.tabs[1].caption]
    assert wording.workbook_line(False) in captions
    # The two old workbook captions, by their text: Task 9 deletes the names.
    assert not any(c.startswith(("Download → Enter measurements", "Three tabs: Round overview"))
                   for c in captions)
    at.checkbox(key="workbook_print_pack").check()
    at.run()
    assert wording.workbook_line(True) in [c.value for c in at.tabs[1].caption]


def test_the_download_help_follows_the_box(open_batch):
    """Review I-2 (Task 3): ticked, the file has no Round overview,
    Preparation or Results sheet, so the "?" must not say it has."""
    import io
    from openpyxl import load_workbook
    def download_help():
        return next(b for b in at.tabs[1].get("download_button")
                    if "download_batch_sheets" in b.proto.id).proto.help
    at = _app()
    at.run()
    assert download_help() == wording.WORKBOOK_TABS_HELP
    at.checkbox(key="workbook_print_pack").check()
    at.run()
    help_text = download_help()
    assert help_text == wording.workbook_tabs_help(True)
    assert not help_text.startswith("The workbook has three sheets")
    # What it says is what the file holds.
    book = load_workbook(io.BytesIO(open_batch.workbook_bytes(
        open_batch.pending_batch, print_pack=True)))
    sheets = [ws.title for ws in book.worksheets if ws.sheet_state == "visible"]
    assert wording.RESULTS_SHEET not in sheets and wording.OVERVIEW_SHEET not in sheets
    assert wording.batch_sheet_name(open_batch.pending_batch_no) in sheets


def test_the_edit_formulations_captions_are_gone(open_batch):
    open_batch.set_records("actual", True)
    at = _app()
    at.run()
    captions = [c.value for c in at.tabs[1].caption]
    assert not any(c.startswith(("To change amounts or process settings in this table",
                                 "Use Edit formulations before preparation"))
                   for c in captions)


def test_lot_numbers_are_in_a_fold(open_batch):
    at = _app()
    at.run()
    assert wording.LOT_ENTRY_HEADING in [e.label for e in at.tabs[1].expander]
    assert not next(e for e in at.tabs[1].expander
                    if e.label == wording.LOT_ENTRY_HEADING).proto.expanded     # V14
    assert f"**{wording.LOT_ENTRY_HEADING}**" not in [m.value for m in at.tabs[1].markdown]


def test_the_first_five_caption_shows_only_before_five_are_scored(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = _patty(rows=4)
    at = _app()
    at.session_state["_loaded_project"] = opt.project_name
    _on_batch(at)
    assert wording.HOW_CHOSEN_2 in [c.value for c in at.tabs[1].caption]
    _tell(opt, 5, 2)
    at = _app()
    at.session_state["_loaded_project"] = opt.project_name
    _on_batch(at)
    assert not at.exception
    captions = [c.value for c in at.tabs[1].caption]
    assert wording.HOW_CHOSEN_2 not in captions
    assert not any(c.startswith("Until five formulations have results") for c in captions)


def test_the_waiting_line_is_one_line(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = _patty(rows=6)
    _tell(opt, 7, 3, drop=("Cook loss",))
    _tell(opt, 8, 3, drop=("Cook loss",))
    assert ui_batch._waiting_caption(opt) == wording.waiting_line(2, "Cook loss")


# ------------------------------------------------------------------ #
#  Start from Best so far (V31); Back to Set up lit (V3)
# ------------------------------------------------------------------ #

def test_start_from_best_so_far_is_spelled_as_everywhere_else(scored):
    scored.set_pending_batch([{"Pea protein": 15.0, "Methylcellulose": 2.0}])
    at = _app()
    at.run()
    labels = [b.label for b in at.button]
    assert wording.START_FROM_BEST_SO_FAR in labels
    assert "Start from the best so far" not in labels


def test_a_new_project_lights_back_to_set_up(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    FoodOptimizer("empty").save()
    at = _app()
    at.session_state["_loaded_project"] = "empty"
    _on_batch(at)
    assert not at.exception
    back = next(b for b in at.tabs[1].button if b.label == wording.BACK_TO_SETUP)
    assert back.proto.type == "primary"
    assert next(b for b in at.tabs[1].button if b.key == "generate").disabled


def test_the_new_lines_pass_the_vocabulary_guard():
    from tests.test_food_bo import _BANNED
    for text in (wording.TOO_UNCERTAIN, wording.RECORD_RESULTS_LINE, wording.workbook_line(),
                 wording.HOW_CHOSEN_2, wording.waiting_line(2, "Cook loss")):
        assert not any(p.search(text) for p in _BANNED), text
