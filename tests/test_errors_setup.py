"""Set up's refusals name the row, the cell and the fix (0.8.0 error review).

The owner's example: an ingredient row saved with no Lowest or Highest said
"Butter: Enter a number." — which cell, and why? Every refusal on tab 1 now
names what the reader must look at (the row, the column, the file row, the
box) and says the fix as an action, in the app's own words. Each family is
pinned here where the reader meets it: on screen (AppTest) where a test can
reach the screen, at the model's door otherwise.
"""
import json
import re

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import custom_records
import wording
from food_bo import FoodOptimizer
from tests.test_app_ui import (APP_PATH, ING_GRID, MEAS_GRID, _ingredient_row,
                               _measurement_row, _save_grid, _submit_button,
                               burger)  # noqa: F401


def _errors(at):
    return [e.value for e in at.error]


def _app():
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    return at


def _butter(**cells):
    row = _ingredient_row("Butter")
    row.update(cells)
    return row


# --------------------------------------------------------------------- #
#  The ingredients grid, on screen
# --------------------------------------------------------------------- #

def test_the_owners_example_names_both_empty_cells(burger):
    at = _save_grid(_app(), ING_GRID, added=[_butter(**{
        wording.LOWEST_LABEL: None, wording.HIGHEST_LABEL: None})])
    assert not at.exception
    assert _errors(at) == ["Butter: Enter Lowest and Highest."]
    assert "Butter" not in [v["name"] for v in FoodOptimizer("burger").variables]


def test_one_empty_cell_is_named_alone(burger):
    at = _save_grid(_app(), ING_GRID, added=[_butter(**{
        wording.HIGHEST_LABEL: None})])
    assert _errors(at) == ["Butter: Enter Highest."]


def test_text_in_a_number_cell_is_said_back(burger):
    at = _save_grid(_app(), ING_GRID, added=[_butter(**{
        wording.LOWEST_LABEL: "lots"})])
    assert _errors(at) == ["Butter: Enter a number in Lowest, not “lots”."]


def test_lowest_above_highest_says_the_numbers_and_the_swap(burger):
    at = _save_grid(_app(), ING_GRID, added=[_butter(**{
        wording.LOWEST_LABEL: 90.0, wording.HIGHEST_LABEL: 10.0})])
    assert _errors(at) == ["Butter: Lowest 90 is above Highest 10; swap them."]


def test_a_blank_unit_names_the_cell_and_an_example(burger):
    at = _save_grid(_app(), ING_GRID, added=[_butter(**{
        wording.UNIT_LABEL: ""})])
    assert _errors(at) == ["Butter: Enter a Unit, such as g or ml."]


def test_a_saved_row_whose_name_was_rubbed_out_is_called_by_its_old_name(
        burger):
    at = _save_grid(_app(), ING_GRID,
                    edited={1: {wording.NAME_LABEL: ""}})
    assert _errors(at) == ["Methylcellulose: Enter a name."]


# --------------------------------------------------------------------- #
#  The measurements grid, on screen: named by measurement, not "Row 1"
# --------------------------------------------------------------------- #

def test_a_target_outside_the_scale_is_named_by_measurement(burger):
    at = _save_grid(_app(), MEAS_GRID,
                    edited={0: {wording.TARGET_LABEL: 50.0}})
    assert _errors(at) == [
        "Firmness: Target 50 must be between Scale minimum 0 and Scale "
        "maximum 10."]


def test_a_reversed_scale_says_the_numbers_and_the_swap(burger):
    at = _save_grid(_app(), MEAS_GRID,
                    edited={0: {wording.LOWEST_MEASURABLE_LABEL: 20.0}})
    assert _errors(at) == [
        "Firmness: Scale minimum 20 is above Scale maximum 10; swap them."]


def test_a_new_measurement_with_no_scale_names_both_cells(burger):
    row = _measurement_row("Crunch", low=None, high=None)
    at = _save_grid(_app(), MEAS_GRID, added=[row])
    assert _errors(at) == ["Crunch: Enter Scale minimum and Scale maximum."]


def test_a_share_of_nought_is_said_back(burger):
    at = _save_grid(_app(), MEAS_GRID,
                    edited={1: {wording.SHARE_COLUMN: 0.0}})
    assert _errors(at) == ["Juiciness: Share of score (%) is 0; enter more "
                           "than 0."]


def test_a_target_goal_with_no_target_names_the_cell(burger):
    row = _measurement_row("Crunch", goal="target")
    at = _save_grid(_app(), MEAS_GRID, added=[row])
    assert _errors(at) == ["Crunch: Enter a Target, or change Goal."]


# --------------------------------------------------------------------- #
#  The sidebar's project name
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("typed, said", [
    ("Burger #2", "Take # out of the name."),
    ("-burger", "Start the name with a letter or a number."),
    ("b" * 70, "Shorten the name to 64 characters; it has 70."),
])
def test_a_refused_project_name_says_what_to_change(tmp_path, monkeypatch,
                                                    typed, said):
    monkeypatch.chdir(tmp_path)
    at = _app()
    at.text_input(key="new_project_name").set_value(typed)
    _submit_button(at, "Create project").click()
    at.run()
    assert _errors(at) == [said]
    assert list(tmp_path.glob("*.pkl")) == []


# --------------------------------------------------------------------- #
#  The calculation editor names the row it refuses
# --------------------------------------------------------------------- #

def test_the_calculation_editor_names_the_refused_row(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("calc_named", robust=False)
    opt.add_ingredient("Flour", 10, 20, unit="g")
    opt.add_ingredient("Starch", 5, 10, unit="g")

    def render():
        from calculation_editor import open_editor
        from food_bo import FoodOptimizer
        import wording
        opt = FoodOptimizer("calc_named")
        frame = opt.ingredient_grid_frame()
        frame.loc[frame.index[1], wording.LOWEST_LABEL] = "lots"
        open_editor(opt, frame)

    at = AppTest.from_function(render, default_timeout=120).run()
    assert not at.exception
    assert _errors(at) == ["Starch: Enter a number in Lowest, not “lots”."]


# --------------------------------------------------------------------- #
#  At the model's door: the grids
# --------------------------------------------------------------------- #

@pytest.fixture
def opt(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("errors", robust=False)
    opt.set_amount_unit("g")
    opt.add_ingredient("Flour", 10, 50)
    opt.add_ingredient("Salt", 0, 2)
    opt.add_objective("Taste", 1.0, min_val=0, max_val=10)
    return opt


def _row(frame, position, **cells):
    frame = frame.copy()
    for column, value in cells.items():
        frame.loc[frame.index[position - 1], column] = value
    return frame


def test_a_type_change_after_results_names_the_type_it_keeps(opt):
    opt.tell({"Flour": 20.0, "Salt": 1.0}, {"Taste": 6.0})
    errors, _ = opt.apply_ingredient_grid(_row(
        opt.ingredient_grid_frame(), 2,
        **{wording.TYPE_LABEL: wording.KIND_SETTING}))
    assert errors == [(2, "Type stays Ingredient once results are recorded. "
                          "Delete the row and add it again.")]


def test_a_vendor_on_a_setting_names_both_fixes(opt):
    opt.add_process_parameter("Bake time", 10, 20, unit="min")
    frame = opt.ingredient_grid_frame()
    assert wording.only_an_ingredient_has(wording.VENDOR_LABEL) == (
        "Clear Vendor, or change Type to Ingredient.")
    assert wording.only_an_ingredient_has(wording.MADE_AS_LABEL) == (
        "Set Preparation to Single ingredient, or change Type to Ingredient.")
    if wording.VENDOR_LABEL in frame.columns:
        errors, _ = opt.apply_ingredient_grid(_row(
            frame, 3, **{wording.VENDOR_LABEL: "Acme"}))
        assert errors == [(3, "Clear Vendor, or change Type to Ingredient.")]


def test_a_baseline_outside_the_range_names_both_ends(opt):
    assert wording.baseline_outside_message(25, 10, 20) == (
        "Baseline 25 must be between Lowest 10 and Highest 20.")


def test_the_properties_grid_names_the_column(opt):
    opt.add_property("Cost")
    errors, _ = opt.apply_property_grid(_row(
        opt.property_grid_frame(), 1, Cost="cheap"))
    assert errors == [(1, "Enter a number in Cost, or leave it empty.")]


def test_a_measurement_typed_twice_is_said_once_by_row(opt):
    frame = opt.measurement_grid_frame()
    added = pd.concat([frame, frame.iloc[[0]].assign(
        **{"_id": None})], ignore_index=True)
    errors, _ = opt.apply_measurement_grid(added)
    assert errors == [(2, wording.NAME_ON_TWO_ROWS)]
    assert wording.NAME_ON_TWO_ROWS == ("Another row has this name; rename "
                                        "one of them.")


def test_a_named_refusal_that_starts_with_the_name_is_not_prefixed_again():
    assert wording.named_row_error(
        "Salt", "Salt is already the name of an ingredient.") == (
        "Salt is already the name of an ingredient.")
    assert wording.named_row_error("Salt", "Enter a name.") == (
        "Salt: Enter a name.")


def test_grid_refusals_are_named_by_position_and_old_name():
    from food_bo import GRID_ID
    from ui_setup import _row_names
    frame = pd.DataFrame({wording.NAME_LABEL: ["Flour", "", None],
                          GRID_ID: ["Flour", "Salt", None]},
                         index=[1, 3, 4])      # a row above was deleted
    assert _row_names(frame, wording.NAME_LABEL) == {
        1: "Flour", 2: "Salt", 3: ""}


# --------------------------------------------------------------------- #
#  The ingredients file
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("frame, said", [
    (pd.DataFrame({"Name": ["Flour"], "Lowest": [1]}),
     "The file has no Highest column. Add it, or start from the template."),
    (pd.DataFrame({"Name": ["Flour"]}),
     "The file has no Lowest, Highest columns. Add them, or start from the "
     "template."),
    (pd.DataFrame({"Name": ["Flour", "Salt"], "Lowest": [1, "lots"],
                   "Highest": [5, 2]}),
     "Salt: Enter a number in Lowest, not “lots”."),
    (pd.DataFrame({"Name": ["Flour", "Salt"], "Lowest": [1, None],
                   "Highest": [5, 2]}),
     "Salt: Enter Lowest."),
    (pd.DataFrame({"Name": ["Flour", "Salt"], "Lowest": [1, 5],
                   "Highest": [5, 2]}),
     "Salt: Lowest 5 is above Highest 2; swap them."),
    (pd.DataFrame({"Name": ["Flour", "Flour"], "Lowest": [1, 1],
                   "Highest": [5, 5]}),
     "Row 3: Flour is listed twice; delete one of the rows."),
    (pd.DataFrame({"Name": ["Flour", None], "Lowest": [1, 1],
                   "Highest": [5, 5]}),
     "Row 3: enter a Name, or delete the row."),
    (pd.DataFrame({"Name": ["Flour", "Salt"], "Lowest": [1, 0],
                   "Highest": [5, 2], "Part of": ["Mix", ""]}),
     "Row 2: add a row for Mix with its Preparation."),
    (pd.DataFrame({"Name": ["Flour", "Salt"], "Lowest": [1, 0],
                   "Highest": [5, 2], "Calculation": ["", "= 2 × Sugar"]}),
     "Salt: There is no ingredient called Sugar; check the spelling."),
])
def test_an_ingredient_file_is_refused_by_row_and_cell(opt, frame, said):
    with pytest.raises(ValueError) as refused:
        opt.load_ingredients_from_csv(frame)
    assert str(refused.value) == said
    assert [v["name"] for v in opt.variables] == ["Flour", "Salt"]


def test_an_unreadable_file_is_named():
    assert wording.ingredients_file_unreadable("burger.xlsx") == (
        "burger.xlsx could not be opened; upload the ingredients template "
        "as .xlsx or .csv.")


# --------------------------------------------------------------------- #
#  Limits and the default batch size
# --------------------------------------------------------------------- #

def test_limit_ends_are_said_in_their_numbers(opt):
    with pytest.raises(ValueError, match="^At least 30 is above At most 20; "
                                         "swap them.$"):
        opt.add_quantity_constraint(["Flour", "Salt"], min_val=30, max_val=20)
    with pytest.raises(ValueError, match="^At least and At most are both 5; "
                                         "make At most larger.$"):
        opt.add_quantity_constraint(["Flour", "Salt"], min_val=5, max_val=5)


def test_an_empty_limit_box_is_named(opt):
    assert wording.limit_number_needed(wording.LIMIT_KIND_EXACTLY) == (
        "Enter a number in Exactly.")
    assert wording.limit_number_needed(wording.LIMIT_KIND_BETWEEN) == (
        "Enter a number in At least, At most, or both.")
    assert wording.ENTER_LOWEST_HIGHEST_ERROR == (
        "Enter a number in At least, At most, or both.")


def test_exactly_on_one_ingredient_names_it(opt):
    with pytest.raises(ValueError) as refused:
        opt.add_quantity_constraint(["Salt"], exactly=1)
    assert str(refused.value) == ("To fix Salt at one amount, give it equal "
                                  "Lowest and Highest in the grid.")


def test_a_batch_size_out_of_reach_says_the_reach_and_the_fix(opt):
    with pytest.raises(ValueError) as high:
        opt.set_formulation_total(80)
    assert str(high.value) == ("These ingredients make at most 52 g; enter "
                               "52 g or less, or raise a Highest.")
    with pytest.raises(ValueError) as low:
        opt.set_formulation_total(5)
    assert str(low.value) == ("These ingredients make at least 10 g; enter "
                              "10 g or more, or lower a Lowest.")


def test_fixing_a_row_that_breaks_a_limit_is_two_short_sentences(opt):
    opt.set_formulation_total(30)
    with pytest.raises(ValueError) as refused:
        opt.add_ingredient("Flour", 10, 10)
    said = str(refused.value)
    assert said == ("Fixing these leaves no formulation at 30 g; change the "
                    "batch size or let more ingredients vary. Fixed at one "
                    "amount: Flour.")


# --------------------------------------------------------------------- #
#  Advanced, and Preparation and records
# --------------------------------------------------------------------- #

def test_pasted_settings_are_refused_where_they_go_wrong():
    with pytest.raises(json.JSONDecodeError) as bad:
        json.loads('{"kernel": "rbf", noise: 1}')
    assert wording.invalid_json(bad.value) == (
        "The settings stop making sense at line 1, character 19; check the "
        "quotes and commas there.")
    assert wording.invalid_json(AttributeError("'list' object")) == (
        "Paste the settings in braces, like the example in the box.")


def test_a_custom_field_already_there_is_named(opt):
    custom_records.add_field(opt, "Operator", "formulation")
    with pytest.raises(ValueError) as twice:
        custom_records.add_field(opt, "operator", "formulation")
    assert str(twice.value) == ("operator already exists for each "
                                "formulation; select it in Additional "
                                "records.")


# --------------------------------------------------------------------- #
#  The style rules themselves, over every rewritten sentence
# --------------------------------------------------------------------- #

_REWRITTEN = [
    wording.cells_need_numbers(["Lowest", "Highest"]),
    wording.cells_need_numbers([], [("Lowest", "lots")]),
    wording.lowest_above_highest(90, 10),
    wording.scale_ends_error(20, 10), wording.scale_ends_error(5, 5),
    wording.type_locked(wording.KIND_INGREDIENT),
    wording.share_not_above_zero(0), wording.SHARE_REQUIRED_ERROR,
    wording.UNIT_REQUIRED_ERROR, wording.ADD_BASELINE_ERROR,
    wording.TARGET_REQUIRED_ERROR, wording.NAME_REQUIRED_ERROR,
    wording.NAME_ON_TWO_ROWS,
    wording.only_a_setting_has(wording.BASELINE_LABEL),
    wording.only_an_ingredient_has(wording.MADE_AS_LABEL),
    wording.limit_bounds_order(30, 20), wording.limit_number_needed("At most"),
    wording.exactly_one_ingredient("Salt"),
    wording.premix_limit_on_its_own("Dry blend"),
    wording.premix_has_no_parts("Dry blend"),
    wording.total_not_reachable_at_most("80 g", "52 g"),
    wording.total_not_reachable_at_least("5 g", "10 g"),
    wording.total_not_reachable_at_all("0 g"),
    wording.balance_would_go_negative("Water", "50 g", "70 g", "g"),
    wording.FORMULA_UNREADABLE, wording.FORMULA_TWO_AMOUNTS,
    wording.FORMULA_DIVIDE_BY_AMOUNT, wording.FORMULA_DIVIDE_BY_ZERO,
    wording.FORMULA_REST_ALONE, wording.FORMULA_NEEDS_BATCH_SIZE,
    wording.RULE_USES_ITS_OWN_ROW, wording.formula_unknown_name("Sugar"),
    wording.formula_loop("Fat → Water → Fat"),
    wording.one_balance_only("Water and Salt"),
    wording.PARTS_ADD_TO_NOTHING,
    wording.premix_amount_is_its_parts("Dry blend"),
    wording.ingredients_file_missing_columns("Highest"),
    wording.file_row_name_blank(3), wording.file_row_duplicate_name(3, "Salt"),
    wording.file_row_premix_in_premix(3, "Mix"),
    wording.file_row_unknown_premix(3, "Mix"),
    wording.limit_needs_one_unit("enter Water in g instead of ml."),
    wording.limit_on_worked_out_rows("Water"),
    wording.fixing_breaks_the_total("30 g"),
    wording.limit_unreachable_above("Fat", "10 per 100 g"),
    wording.project_name_problem("Burger #2"),
    wording.name_collides_with_archive("Burger"),
    wording.custom_field_builtin("Lot"),
]


@pytest.mark.parametrize("said", _REWRITTEN)
def test_every_rewritten_refusal_keeps_the_owners_rules(said):
    words = said.split()
    assert len(words) <= 18, said
    assert said.count(". ") <= 1, said
    # A column's own "(%)" is its label, not a parenthetical.
    assert "(" not in said.replace("(%)", "") and "!" not in said, said
    for banned in ("invalid", "finite", "NaN", "None", "constraint",
                   "parse", "Error", "formula", "JSON"):
        assert not re.search(rf"\b{banned}\b", said), (banned, said)
