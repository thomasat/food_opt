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
