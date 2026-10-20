"""Final public-release fixes, lane B-small: the Set up hint for a time or
temperature typed as an Ingredient (N16), the example's Start over label,
and the wording the other items changed."""
import pandas as pd
from streamlit.testing.v1 import AppTest

import ui_setup
import wording
from tests.test_app_ui import (APP_PATH, ING_GRID, _grid_edits, _ingredient_row,
                               burger)  # noqa: F401  (fixture)


def _frame(*rows):
    return pd.DataFrame([{wording.NAME_LABEL: n, wording.TYPE_LABEL: k,
                          wording.UNIT_LABEL: u} for n, k, u in rows])


def test_setting_unit_rows_finds_times_and_temperatures_typed_as_ingredients():
    frame = _frame(("Pea protein", wording.KIND_INGREDIENT, "g"),
                   ("Mixing time", wording.KIND_INGREDIENT, "min"),
                   ("Bake temp", wording.KIND_INGREDIENT, "°C"),
                   ("Rest", wording.KIND_SETTING, "h"),
                   ("", wording.KIND_INGREDIENT, "s"),
                   ("Hold", None, "Hours"))
    assert ui_setup.setting_unit_rows(frame) == [
        ("Mixing time", "min"), ("Bake temp", "°C"), ("Hold", "Hours")]
    assert ui_setup.setting_unit_rows(_frame(("Water", wording.KIND_INGREDIENT, "g"))) == []


def test_setting_unit_hint_names_the_row_in_ten_words_or_fewer():
    one = wording.setting_unit_hint([("Mixing time", "min")])
    assert one == "Mixing time is in min: set Type to Process setting?"
    assert len(one.split()) <= 10
    two = wording.setting_unit_hint([("Mixing time", "min"), ("Bake temp", "°C")])
    assert two.startswith("Mixing time, Bake temp:") and wording.KIND_SETTING in two


def test_set_up_asks_about_a_minute_row_typed_as_an_ingredient(burger):  # noqa: F811
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "burger"
    at.run()
    hint = wording.setting_unit_hint([("Mixing time", "min")])
    assert hint not in [c.value for c in at.caption]
    _grid_edits(at, ING_GRID, added=[_ingredient_row("Mixing time", low=1.0, high=10.0,
                                                     unit="min")])
    at.run()
    assert not at.exception
    assert hint in [c.value for c in at.caption]
    # Typed as a Process setting, it asks nothing; nothing is changed for
    # the reader either way.
    _grid_edits(at, ING_GRID, added=[_ingredient_row(
        "Mixing time", kind=wording.KIND_SETTING, low=1.0, high=10.0, unit="min")])
    at.run()
    assert hint not in [c.value for c in at.caption]


def test_start_over_names_the_example_not_a_sample():
    assert wording.START_SAMPLE_OVER_LABEL == "Start this example over"
    assert len(wording.START_SAMPLE_OVER_LABEL.split()) <= 5
    assert "sample" not in wording.SAFETY_COPY_REASONS['sample_reset'].lower()


def test_round_key_says_what_plain_means():
    assert wording.compare_key_plain(5) == "Plain: changed under 5 % of its range"


def test_empty_batch_size_help_and_empty_name_line():
    assert wording.BATCH_SIZE_EMPTY_HELP == "Leave empty to keep each formulation's own total."
    assert wording.NAME_EMPTY == "Type a project name first."
    assert wording.sample_opened("the yogurt pilot", 2, 1) == \
        "Opened the yogurt pilot (2 projects, 1 results file added)."
    assert len(wording.sample_opened("the yogurt pilot", 2, 1).split()) <= 10
    assert wording.sample_opened("the yogurt pilot", 0, 1) == \
        "Opened the yogurt pilot (1 results file added)."
