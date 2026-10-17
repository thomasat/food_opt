"""0.8.0 speed and simplicity, Plan B, Task 3: Make a round with fewer
boxes and fewer captions — "too uncertain yet", Compared with third with
two changes, one "More" per formulation, one workbook line, lot numbers in
a fold, the first-five caption only before five, one waiting line, "Start
from Best so far" and a lit Back to Set up (spec 2026-09-30 §3; review
M2, M10, M11, M14; audit V3, V11–V17, V31)."""
import pytest
from streamlit.testing.v1 import AppTest

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
    # v3: the line names the sheets, under the button.
    assert wording.workbook_sheets(False) in captions
    assert 'Upload the filled workbook, or type results below.' not in captions
    # The two old workbook captions, by their text: Task 9 deletes the names.
    assert not any(c.startswith(("Download → Enter measurements", "Three tabs: Round overview"))
                   for c in captions)
    at.checkbox(key="workbook_print_pack").check()
    at.run()
    assert wording.workbook_sheets(True) in [c.value for c in at.tabs[1].caption]


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
    for text in (wording.RECORD_RESULTS_LINE,
                 wording.HOW_CHOSEN_2, wording.waiting_line(2, "Cook loss")):
        assert not any(p.search(text) for p in _BANNED), text
