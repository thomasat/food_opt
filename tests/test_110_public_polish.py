"""Public-release polish (0.8.0 walk, S2–S14 and N1–N19): the behaviour
each fix changed, and the exact words the owner chose."""
from streamlit.testing.v1 import AppTest

import custom_records
import ui_setup
import wording
from food_bo import FoodOptimizer
from tests.test_app_ui import APP_PATH, burger, open_batch, scored  # noqa: F401
from ui_helpers import table_height


def _app(project=None, tab=None):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    if project:
        at.session_state["_loaded_project"] = project
    at.run()
    if tab:
        at.session_state["main_tab"] = tab
        at.run()
    return at


# ------------------------------------------------------------------ #
#  S2: a new project's grids open about six rows tall, and grow
# ------------------------------------------------------------------ #

def test_an_empty_grid_opens_six_rows_tall_and_grows_with_saved_rows():
    assert ui_setup.GRID_MIN_ROWS == 6
    assert ui_setup.grid_height(0) == table_height(6)
    assert ui_setup.grid_height(3) == table_height(6)
    # Room for the blank row being typed, and one more.
    assert ui_setup.grid_height(10) == table_height(12)
    assert ui_setup.grid_height(10) > ui_setup.grid_height(5)
    assert ui_setup.grid_height(50) == table_height(20, max_rows=20)


# ------------------------------------------------------------------ #
#  S13: Additional records' chips name the field only
# ------------------------------------------------------------------ #

def test_record_chips_name_the_field_only(burger):
    custom_records.add_field(burger, "Raw weight (g)", "formulation")
    custom_records.add_field(burger, "Operator", "formulation")
    custom_records.add_field(burger, "Operator", "ingredient")
    at = _app("burger", wording.TAB_SETUP)
    assert not at.exception
    box = at.multiselect(key="record_fields")
    shown = list(box.options)          # as the dropdown and chips write them
    assert "Lot numbers" in shown, shown
    assert "Vendor" in shown and "SKU" in shown, shown
    assert "Raw weight (g)" in shown, shown
    assert not any("Enter in" in o or "Ingredients table" in o for o in shown), shown
    # A name the project has for both scopes keeps its scope, or the two
    # chips would read the same.
    assert wording.custom_record_option("Operator", "formulation", burger) in shown, shown
    assert wording.custom_record_option("Operator", "ingredient", burger) in shown, shown
    assert "Lot numbers in Make a round" in wording.ALSO_RECORD_HELP


# ------------------------------------------------------------------ #
#  N19: the planned-amounts option says its sentence once
# ------------------------------------------------------------------ #

def test_record_changes_option_says_its_help_once(burger):
    at = _app("burger", wording.TAB_SETUP)
    assert burger.records("actual")
    assert wording.ACTUAL_RECORD_HELP not in [c.value for c in at.caption]
    assert at.checkbox(key="record_actual").help == wording.ACTUAL_RECORD_HELP


# ------------------------------------------------------------------ #
#  S6: the line under the download lists the sheets the file has
# ------------------------------------------------------------------ #

def _premix_round(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("premix round", robust=False)
    opt.set_amount_unit("g")
    opt.add_ingredient("Water", 0, 100)
    opt.add_objective("Taste", 1, goal="max", min_val=0, max_val=10)
    opt.add_premix("Dry blend", "portioned")
    opt.set_premix_parts("Dry blend", [{"name": "Flour", "share": 60, "unit": "g"},
                                       {"name": "Salt", "share": 40, "unit": "g"}])
    opt.add_ingredient("Dry blend", 10, 60)
    opt.set_pending_batch([{"Water": 60, "Dry blend": 30},
                           {"Water": 40, "Dry blend": 50}], batch_no=1)
    opt._snapshot_premixes(1)
    opt.save()
    return opt


def test_the_sheets_line_names_preparation_when_the_file_has_it(tmp_path, monkeypatch):
    import io

    import openpyxl
    opt = _premix_round(tmp_path, monkeypatch)
    names = openpyxl.load_workbook(io.BytesIO(
        opt.workbook_bytes(opt.pending_batch, None, print_pack=False))).sheetnames
    assert wording.PREPARATION_SHEET in names
    at = _app(opt.project_name, wording.TAB_BATCH)
    assert not at.exception
    captions = [c.value for c in at.tabs[1].caption]
    assert "Sheets: Round overview · Preparation · Results" in captions, captions
    assert wording.workbook_sheets() not in captions


def test_the_sheets_line_names_two_sheets_without_premixes(open_batch):
    at = _app(None, wording.TAB_BATCH)
    captions = [c.value for c in at.tabs[1].caption]
    assert wording.workbook_sheets() == "Sheets: Round overview · Results"
    assert wording.workbook_sheets() in captions, captions


# ------------------------------------------------------------------ #
#  S14, N10, N11, N17: Make a round's words
# ------------------------------------------------------------------ #

def test_make_a_round_words():
    assert wording.NO_BATCH_SIZE_OF_ITS_OWN == "Optional: scale every formulation to one batch size."
    assert wording.first_formulations_caption(5) == "First 5 formulations explore the allowed amounts."
    assert wording.ADD_OWN_NO_BATCH_CAPTION == "Generate first to see suggestions too."
    assert wording.SHEET_RETURN_PATH == (
        "Then upload this file in the app: 2 · Make a round → Or upload results from a file.")
    # N9: Round capitalised on every label that names one.
    for label in (wording.ADD_OWN_EXPANDER, wording.GENERATE_DIFFERENT_BATCH,
                  wording.START_NEXT_BATCH, wording.PREPARATION_SHEET_TITLE,
                  wording.MAKE_FOR_ROUND_HEADING, wording.ROUND_TOTAL_COLUMN,
                  wording.DOWNLOAD_BATCH_SHEETS, wording.WHOLE_BATCH_LABEL):
        assert " round" not in label, label
    assert wording.TAB_BATCH_NAME == "Make a round"     # the tab keeps its name
