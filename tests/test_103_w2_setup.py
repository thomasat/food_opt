"""W2 of the 0.8.0 e2e fixes, Set up (spec 2026-10-01 rev. 2; review I9):
a blank project can set Water to Fill to total before it has a default
batch size, then take 100 g, then Generate (02 C2); the ingredient file
reads "Fill to total" and an Excel "= rest", and never loads a limit as
NaN (03 I8); the Fill-to-total row's caption says what to do while there
is no size (03 M4)."""
import io

import pandas as pd
import pytest
from openpyxl import Workbook

import calculation_editor
import ui_setup
import wording
from food_bo import FoodOptimizer

# 02's patty, as the template brought it in: no water yet.
PATTY = [("Textured pea protein", 15, 30), ("Coconut oil", 5, 20),
         ("Methylcellulose", 0.5, 4), ("Beet juice powder", 0.2, 1.5), ("Salt", 0.5, 2)]


def _blank(tmp_path, monkeypatch, name="blank"):
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer(name)
    opt.load_ingredients_from_csv(pd.DataFrame(
        [{"Name": n, "Lowest": lo, "Highest": hi} for n, lo, hi in PATTY]))
    return opt


def _with_water(opt, calculation=""):
    """The grid with a Water row typed on its empty line (02: "Add a Water
    row"), its Calculation cell as typed."""
    frame = opt.ingredient_grid_frame()
    row = {c: "" for c in frame.columns}
    row.update({wording.NAME_LABEL: "Water", wording.TYPE_LABEL: wording.KIND_INGREDIENT,
                wording.LOWEST_LABEL: "0", wording.HIGHEST_LABEL: "0",
                wording.UNIT_LABEL: "g", wording.FORMULA_LABEL: calculation})
    return pd.concat([frame, pd.DataFrame([row])], ignore_index=True)


def test_a_blank_project_fills_to_total_then_takes_100g_then_generates(tmp_path, monkeypatch):
    """02 C2's loop, end to end: "Use fill to total" needed a default batch
    size, and the batch size refused 100 g ("the most these ingredients can
    make is 57.5 g") until water filled to total."""
    opt = _blank(tmp_path, monkeypatch)
    errors, _ = opt.apply_ingredient_grid(calculation_editor.canonical_frame(
        _with_water(opt, wording.FILL_TO_TOTAL), opt.ingredient_grid_frame()))
    assert errors == []
    assert opt.balance_row_name() == "Water" and not opt.has_formulation_total()
    # Set up says what to do next, not a range worked out on a size of 0.
    assert opt.worked_out_captions() == [wording.rest_needs_batch_size("Water")]
    opt.set_formulation_total(100)
    assert float(opt.formulation_total) == 100.0
    opt.add_objective("Firmness", 1.0, goal="max", min_val=10, max_val=70)
    batch = opt.ask(n_suggestions=3)
    assert len(batch) == 3
    for recipe in batch:
        assert opt.ingredient_total(recipe) == pytest.approx(100.0, abs=1e-6)
        assert recipe["Water"] == pytest.approx(
            100.0 - sum(recipe[n] for n, _, _ in PATTY), abs=1e-6)


def test_the_dialog_accepts_fill_to_total_without_a_size(tmp_path, monkeypatch):
    """Edit calculation → Water → Fill to total → Use fill to total: the
    dialog validates through the grid's own plan."""
    opt = _blank(tmp_path, monkeypatch)
    errors, _ = opt.apply_ingredient_grid(_with_water(opt))
    assert errors == []
    frame = opt.ingredient_grid_frame()
    water = frame.index[frame[wording.NAME_LABEL] == "Water"][0]
    # The candidate the dialog's Fill to total mode builds and validates.
    candidate = frame.copy()
    candidate.at[water, wording.FORMULA_LABEL] = f"= {wording.REST_TOKEN}"
    errors, _ = calculation_editor._validate(opt, candidate)
    assert errors == []


def test_the_one_door_takes_rest_without_a_size(tmp_path, monkeypatch):
    opt = _blank(tmp_path, monkeypatch)
    opt.add_ingredient("Water", 0, 0)
    opt.set_formula("Water", "= rest")
    assert opt.balance_row_name() == "Water"
    opt.set_formulation_total(100)
    assert float(opt.formulation_total) == 100.0


# ------------------------------------------------------------------ #
#  The ingredient file
# ------------------------------------------------------------------ #

def _file(calculation, low="", high=""):
    return pd.DataFrame([{"Name": n, "Lowest": lo, "Highest": hi, "Calculation": ""}
                         for n, lo, hi in PATTY]
                        + [{"Name": "Water", "Lowest": low, "Highest": high,
                            "Calculation": calculation}])


@pytest.mark.parametrize("cell", ["Fill to total", "fill to total", "FILL TO TOTAL",
                                  " Fill to Total ", "= rest", "=rest", "= REST"])
def test_the_file_reads_fill_to_total(tmp_path, monkeypatch, cell):
    """02 C2: "Fill to total" in the template's Calculation column was
    refused as "Start a calculation with =."."""
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("file")
    opt.load_ingredients_from_csv(_file(cell))
    water = opt._var_by_name("Water")
    assert opt.balance_row_name() == "Water" and water["balance"]
    assert all(b == b for b in water["bounds"])          # never NaN
    opt.set_formulation_total(100)


def test_the_csv_door_reads_fill_to_total(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    text = _file("Fill to total").to_csv(index=False)
    opt = FoodOptimizer("csv")
    opt.load_ingredients_from_csv(pd.read_csv(io.StringIO(text)))
    assert opt.balance_row_name() == "Water"


class _Upload(io.BytesIO):
    name = "ingredients.xlsx"


def _xlsx(rows):
    book = Workbook()
    sheet = book.active
    sheet.append(["Name", "Lowest", "Highest", "Calculation", "Fat per 100 g"])
    for row in rows:
        sheet.append(row)
    out = io.BytesIO()
    book.save(out)
    return _Upload(out.getvalue())


def test_an_excel_rest_formula_is_read_as_written(tmp_path, monkeypatch):
    """03 I8: '= rest' typed in Excel is a formula; read for its (absent)
    saved value it came back blank, and Date paste loaded as an ordinary
    ingredient with Lowest/Highest NaN."""
    monkeypatch.chdir(tmp_path)
    upload = _xlsx([[n, lo, hi, None, 1.5] for n, lo, hi in PATTY]
                   + [["Water", None, None, "= rest", 0]])
    frame = ui_setup._read_ingredients_file(upload)
    assert list(frame["Calculation"])[-1] == "= rest"
    opt = FoodOptimizer("xlsx")
    opt.load_ingredients_from_csv(frame)
    assert opt.balance_row_name() == "Water"
    assert opt.ingredient_properties["Salt"]["Fat per 100 g"] == 1.5


def test_pandas_alone_read_the_excel_rest_as_blank():
    """Why the reader changed: the cell's saved value is all pandas sees."""
    upload = _xlsx([["Water", None, None, "= rest", 0]])
    frame = pd.read_excel(io.BytesIO(upload.getvalue()), sheet_name=0)
    assert frame["Calculation"].isna().all()


def test_a_limit_that_would_be_nan_is_refused_by_name(tmp_path, monkeypatch):
    """Review I9: refuse, naming the row, any row whose Lowest/Highest would
    be NaN — nothing is loaded."""
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("nan")
    with pytest.raises(ValueError) as caught:
        opt.load_ingredients_from_csv(_file("", low="0", high=float("nan")))
    assert str(caught.value) == wording.import_limits_not_numbers(
        "Water", (wording.HIGHEST_LABEL,))
    with pytest.raises(ValueError) as caught:
        opt.load_ingredients_from_csv(_file(""))
    assert str(caught.value) == wording.import_limits_not_numbers("Water")
    assert opt.variables == []


def test_an_excel_limit_formula_with_no_saved_value_is_named(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    upload = _xlsx([["Salt", "=B3*2", 2, None, 0], ["Water", 0, 80, None, 0]])
    frame = ui_setup._read_ingredients_file(upload)
    opt = FoodOptimizer("formula")
    with pytest.raises(ValueError) as caught:
        opt.load_ingredients_from_csv(frame)
    assert str(caught.value) == wording.import_spreadsheet_formula(
        "Salt", wording.LOWEST_LABEL, "=B3*2")
