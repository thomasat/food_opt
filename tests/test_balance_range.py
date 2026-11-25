"""The Fill to total row is held inside its own Lowest and Highest.

Ported from the web app's balance ingredient (e2752b7): the amount is the
batch size less every other ingredient, kept within a min and a max. In the
desktop that row is written '= rest'; its range is what was typed into its
own Lowest and Highest cells (or given it by an ingredient file), stored as
`fill_limits`, a blank cell no limit on that side. A row with none typed —
every file saved before 0.9.1 — has no range, only the floor of nothing
every calculated row has.
"""
import random

import pytest

import wording
from food_bo import FoodOptimizer


def _project(tmp_path, monkeypatch, water=(20, 40), name="fill_range"):
    """Flour 10–80 g and Sugar 5–60 g leave Water anywhere from 0 to 85 g of
    a 100 g formulation, so a 20–40 g range on Water binds at both ends."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer(name, robust=False)
    opt.set_amount_unit("g")
    opt.add_ingredient("Water", *water)
    opt.add_ingredient("Flour", 10, 80)
    opt.add_ingredient("Sugar", 5, 60)
    opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
    opt.set_formulation_total(100)
    opt.set_formula("Water", "= rest")
    _type_range(opt, *water)
    return opt


def _type_range(opt, low, high):
    """Type Water's range into its own two grid cells and Save, as the
    reader does: 0 and 0 is the two cells left blank."""
    blank = low == 0 and high == 0
    frame = opt.ingredient_grid_frame()
    at = int(frame.index[frame[wording.NAME_LABEL] == "Water"][0])
    frame.at[at, wording.LOWEST_LABEL] = "" if blank else str(low)
    frame.at[at, wording.HIGHEST_LABEL] = "" if blank else str(high)
    errors, _ = opt.apply_ingredient_grid(frame)
    assert errors == []


def _score(opt, recipes, seed=0):
    rng = random.Random(seed)
    for recipe in recipes:
        opt.tell(recipe, {"Taste": rng.uniform(0, 10)})


def _inside(recipe, low=20.0, high=40.0):
    return low - 1e-6 <= recipe["Water"] <= high + 1e-6


def test_held_range_reads_the_fill_rows_own_two_numbers(tmp_path, monkeypatch):
    opt = _project(tmp_path, monkeypatch)
    water = opt._var_by_name("Water")
    assert opt._held_range(water) == (20.0, 40.0)
    # A row with no range typed keeps only the floor of nothing, whatever
    # its kept bounds say; so does one typed 0 and 0.
    water.pop('fill_limits')
    assert water['bounds'] == (20.0, 40.0)
    assert opt._held_range(water) == (0.0, None)
    assert opt._fill_range() is None
    water['fill_limits'] = [0.0, 0.0]
    assert opt._fill_range() is None
    # Every other calculated row's range stays dormant.
    assert opt._held_range({'bounds': (3.0, 5.0), 'formula': '= 4'}) == (0.0, None)


def test_the_check_holds_the_fill_row_inside_its_range(tmp_path, monkeypatch):
    opt = _project(tmp_path, monkeypatch)
    assert opt._check_constraints({"Flour": 40, "Sugar": 30, "Water": 30})
    # Water above its Highest: 100 − 20 − 10 = 70 g.
    assert not opt._check_constraints({"Flour": 20, "Sugar": 10, "Water": 70})
    # ...and below its Lowest: 100 − 60 − 30 = 10 g.
    assert not opt._check_constraints({"Flour": 60, "Sugar": 30, "Water": 10})
    # The range ends themselves are inside.
    assert opt._check_constraints({"Flour": 40, "Sugar": 20, "Water": 40})
    assert opt._check_constraints({"Flour": 50, "Sugar": 30, "Water": 20})


def test_the_model_is_handed_both_ends_when_they_can_bind(tmp_path, monkeypatch):
    opt = _project(tmp_path, monkeypatch)
    rows = [(r[0].tolist(), [round(c, 9) for c in r[1].tolist()], round(float(r[2]), 9))
            for r in opt._get_botorch_constraints()]
    # Water = 100 − Flour − Sugar over columns Flour (10 + 70 z) and Sugar
    # (5 + 55 z): Water ≥ 20 and Water ≤ 40.
    assert ([0, 1], [-70.0, -55.0], -65.0) in rows
    assert ([0, 1], [70.0, 55.0], 45.0) in rows
    assert len(rows) == 2


def test_an_end_that_can_never_bind_changes_nothing_the_model_sees(
        tmp_path, monkeypatch):
    """A range the other rows always keep (0–100 g, the common case of a
    row that was 0–100 before it was made the filler) hands the model
    exactly the rows a row with no range of its own does."""
    wide = _project(tmp_path / "a", monkeypatch, water=(0, 100), name="a")
    none = _project(tmp_path / "b", monkeypatch, water=(0, 0), name="b")

    def rows(opt):
        return [(r[0].tolist(), r[1].tolist(), float(r[2]))
                for r in opt._get_botorch_constraints()]
    assert rows(wide) == rows(none)


def test_round_one_keeps_the_fill_row_inside_its_range(tmp_path, monkeypatch):
    opt = _project(tmp_path, monkeypatch)
    recipes = opt.ask(5)
    assert len(recipes) == 5
    for recipe in recipes:
        assert sum(recipe.values()) == pytest.approx(100.0, abs=1e-6)
        assert _inside(recipe), recipe


def test_model_suggestions_keep_the_fill_row_inside_its_range(tmp_path, monkeypatch):
    opt = _project(tmp_path, monkeypatch)
    _score(opt, opt.ask(5))
    recipes = opt.ask(3)
    assert len(recipes) == 3
    for recipe in recipes:
        assert sum(v for k, v in recipe.items()) == pytest.approx(100.0, abs=1e-4)
        assert _inside(recipe), recipe


def test_an_unreachable_range_is_refused_in_the_rows_name(tmp_path, monkeypatch):
    opt = _project(tmp_path, monkeypatch)
    # Flour and Sugar now need at least 110 g: Water is at most 0 g.
    opt._var_by_name("Flour")['bounds'] = (70.0, 80.0)
    opt._var_by_name("Sugar")['bounds'] = (40.0, 60.0)
    expected = wording.fill_outside_its_range(
        "Water", "20–40 g", wording.at_most_text("0 g"))
    with pytest.raises(ValueError) as refused:
        opt.ask(2)
    assert str(refused.value) == expected
    assert opt.worked_out_captions() == [expected]
    # Above its Highest instead: the others reach at most 25 g.
    opt._var_by_name("Flour")['bounds'] = (10.0, 15.0)
    opt._var_by_name("Sugar")['bounds'] = (5.0, 10.0)
    with pytest.raises(ValueError) as refused:
        opt.ask(2)
    assert str(refused.value) == wording.fill_outside_its_range(
        "Water", "20–40 g", wording.at_least_text("75 g"))


def test_the_default_batch_size_box_asks_the_fill_rows_range(tmp_path, monkeypatch):
    opt = _project(tmp_path, monkeypatch)
    # Flour and Sugar need at least 15 g, so Water's 20 g floor needs 35 g.
    with pytest.raises(ValueError) as refused:
        opt.set_formulation_total(30)
    assert str(refused.value) == wording.balance_would_go_negative(
        "Water", opt.batch_total_text(30), opt.batch_total_text(35),
        unit="g", floor_text="20 g")
    # The top is not the box's to refuse (a rest row has no largest batch
    # size): Flour and Sugar take up at most 140 g, so at 200 g Water is at
    # least 60 g, and Generate and the caption say so in Water's name.
    opt.set_formulation_total(200)
    assert opt.formulation_total == 200
    expected = wording.fill_outside_its_range(
        "Water", "20–40 g", wording.at_least_text("60 g"))
    assert opt.worked_out_captions() == [expected]
    with pytest.raises(ValueError) as refused:
        opt.ask(2)
    assert str(refused.value) == expected


def test_a_fill_row_with_no_range_keeps_the_old_batch_size_answers(
        tmp_path, monkeypatch):
    opt = _project(tmp_path, monkeypatch, water=(0, 0))
    with pytest.raises(ValueError) as refused:
        opt.set_formulation_total(10)
    assert str(refused.value) == wording.balance_would_go_negative(
        "Water", opt.batch_total_text(10), opt.batch_total_text(15), unit="g")
    opt.set_formulation_total(500)      # no top
    assert opt.formulation_total == 500


def test_the_caption_reads_inside_the_range_and_never_says_it_does_not_apply(
        tmp_path, monkeypatch):
    opt = _project(tmp_path, monkeypatch)
    opt.add_ingredient("Salt", 0, 3)
    opt.set_formula("Salt", "= 5")
    lines = opt.worked_out_captions()
    # Water: 100 − Flour − Sugar − 5 reaches 0 to 80 g, held to 20–40 g.
    assert lines[0] == wording.worked_out_caption(
        "Water", "= rest", "20.00", "40.00 g", opt.batch_total_text(100),
        rest=True)
    # Salt's own 0–3 g still does not apply while its calculation does.
    assert "do not apply" in lines[1]


def test_a_quantity_limit_already_holding_the_fill_row_moves_nothing(
        tmp_path, monkeypatch):
    """The demo burger's shape: the filler at least 10 g by a limit on the
    other rows AND by its own Lowest. The two say the same thing, so the
    suggestions are what the limit alone gave."""
    def build(path, water, name):
        opt = _project(path, monkeypatch, water=water, name=name)
        opt.add_quantity_constraint(["Flour", "Sugar"], max_val=90)
        return opt
    # One name: the space-filling design is seeded from it.
    ranged = build(tmp_path / "a", (10, 85), "burger")
    plain = build(tmp_path / "b", (0, 0), "burger")
    first = ranged.ask(5)
    assert first == plain.ask(5)
    _score(ranged, first)
    _score(plain, first)
    assert ranged.ask(2) == plain.ask(2)


def test_bounds_caution_follows_the_held_range(tmp_path, monkeypatch):
    opt = _project(tmp_path, monkeypatch)
    assert opt.bounds_caution("Water", 30) == ""
    assert opt.bounds_caution("Water", 45) != ""
    # A 200 g round of a 100 g formulation: 20–40 g is 40–80 g.
    assert opt.bounds_caution("Water", 70, 2.0) == ""
    opt._var_by_name("Water").pop('fill_limits')
    assert opt.bounds_caution("Water", 45) == ""


def test_a_project_without_a_fill_row_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("plain", robust=False)
    opt.add_ingredient("Flour", 10, 80)
    opt.add_ingredient("Sugar", 5, 60)
    opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
    assert opt._fill_range() is None
    assert opt._fill_range_refusal() == ""
    assert opt._get_botorch_constraints() == []


# ------------------------------------------------------------------ #
# The range on the grid: shown in the row's own Lowest and Highest cells,
# saved from them, a blank cell no limit on that side.
# ------------------------------------------------------------------ #

L, H, F = wording.LOWEST_LABEL, wording.HIGHEST_LABEL, wording.FORMULA_LABEL


def _water_row(opt):
    frame = opt.ingredient_grid_frame()
    return frame, int(frame.index[frame[wording.NAME_LABEL] == "Water"][0])


def test_the_grid_shows_and_saves_the_fill_rows_range(tmp_path, monkeypatch):
    opt = _project(tmp_path, monkeypatch)
    frame, at = _water_row(opt)
    assert (frame.at[at, L], frame.at[at, H], frame.at[at, F]) == (
        "20.00", "40.00", "= rest")
    # A blank Highest is no top: only the 30 g Lowest holds.
    frame.at[at, L], frame.at[at, H] = "30", ""
    errors, _ = opt.apply_ingredient_grid(frame)
    assert errors == []
    water = opt._var_by_name("Water")
    assert water['fill_limits'] == [30.0, None]
    assert water['bounds'] == (20.0, 40.0)     # what rubbing it out gives back
    assert opt._held_range(water) == (30.0, None)
    frame, at = _water_row(opt)
    assert (frame.at[at, L], frame.at[at, H]) == ("30.00", "")
    for recipe in opt.ask(4):
        assert recipe["Water"] >= 30.0 - 1e-6
    assert not opt._check_constraints({"Flour": 60, "Sugar": 20, "Water": 20})
    assert opt._check_constraints({"Flour": 10, "Sugar": 5, "Water": 85})
    # Both blank: no range at all, only the floor of nothing.
    frame.at[at, L] = ""
    assert opt.apply_ingredient_grid(frame)[0] == []
    assert opt._fill_range() is None


def test_the_grid_refuses_a_fill_range_it_cannot_read(tmp_path, monkeypatch):
    opt = _project(tmp_path, monkeypatch)
    frame, at = _water_row(opt)
    frame.at[at, L], frame.at[at, H] = "50", "40"
    errors, _ = opt.apply_ingredient_grid(frame)
    assert errors == [(at, wording.lowest_above_highest(50.0, 40.0))]
    frame.at[at, L], frame.at[at, H] = "a cup", ""
    errors, _ = opt.apply_ingredient_grid(frame)
    assert len(errors) == 1 and errors[0][0] == at
    assert opt._fill_limits(opt._var_by_name("Water")) == (20.0, 40.0)


def test_another_calculated_row_keeps_blank_ignored_cells(tmp_path, monkeypatch):
    opt = _project(tmp_path, monkeypatch)
    opt.add_ingredient("Salt", 0, 3)
    opt.set_formula("Salt", "= 0.02 * Flour")
    frame = opt.ingredient_grid_frame()
    at = int(frame.index[frame[wording.NAME_LABEL] == "Salt"][0])
    assert (frame.at[at, L], frame.at[at, H]) == ("", "")
    frame.at[at, L], frame.at[at, H] = "1", "2"
    assert opt.apply_ingredient_grid(frame)[0] == []
    salt = opt._var_by_name("Salt")
    assert salt['bounds'] == (0.0, 3.0) and 'fill_limits' not in salt


def test_a_new_fill_range_retires_the_open_round(tmp_path, monkeypatch):
    opt = _project(tmp_path, monkeypatch)
    opt.ask(2)
    assert opt.pending_batch_no is not None
    frame, at = _water_row(opt)
    assert opt.ingredient_grid_retires_round(frame) is None   # unchanged
    frame.at[at, H] = "35"
    assert opt.ingredient_grid_retires_round(frame) == opt.pending_batch_no
    opt.apply_ingredient_grid(frame)
    assert opt.pending_batch_no is None


def test_a_row_that_stops_filling_drops_its_fill_range(tmp_path, monkeypatch):
    opt = _project(tmp_path, monkeypatch)
    frame, at = _water_row(opt)
    frame.at[at, H] = ""
    opt.apply_ingredient_grid(frame)
    assert 'fill_limits' in opt._var_by_name("Water")
    opt.clear_formula("Water")
    water = opt._var_by_name("Water")
    assert 'fill_limits' not in water and water['bounds'] == (20.0, 40.0)


def test_an_older_project_has_no_fill_range(tmp_path, monkeypatch):
    """A file saved before fill_limits existed behaves as 0.9.0 did: no
    range, whatever its kept bounds say, blank cells on the grid, and a
    save that changes nothing writes no new key."""
    opt = _project(tmp_path, monkeypatch)
    state = opt.export_json()
    state['variables'][0].pop('fill_limits')
    assert tuple(state['variables'][0]['bounds']) == (20.0, 40.0)
    FoodOptimizer.validate_state(state)
    back = FoodOptimizer("older", robust=False)
    back.import_json(state)
    assert back._held_range(back._var_by_name("Water")) == (0.0, None)
    frame, at = _water_row(back)
    assert (frame.at[at, L], frame.at[at, H]) == ("", "")
    assert back.apply_ingredient_grid(frame)[0] == []
    assert 'fill_limits' not in back._var_by_name("Water")


def test_a_damaged_fill_range_is_refused_at_the_door(tmp_path, monkeypatch):
    opt = _project(tmp_path, monkeypatch)
    state = opt.export_json()
    state['variables'][0]['fill_limits'] = ["a", None]
    with pytest.raises(Exception):
        FoodOptimizer.validate_state(state)
    state['variables'][0]['fill_limits'] = [10.0, None]
    FoodOptimizer.validate_state(state)


def test_an_ingredient_file_gives_the_fill_row_a_one_sided_range(
        tmp_path, monkeypatch):
    import pandas as pd
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("file_fill", robust=False)
    opt.set_amount_unit("g")
    opt.load_ingredients_from_csv(pd.DataFrame({
        "Name": ["Flour", "Sugar", "Water"], "Lowest": [10, 5, 25],
        "Highest": [60, 20, None], "Unit": ["g", "g", "g"],
        "Rule": [None, None, "= rest"]}))
    water = opt._var_by_name("Water")
    assert water['fill_limits'] == [25.0, None]
    assert opt._held_range(water) == (25.0, None)
    opt2 = FoodOptimizer("file_fill2", robust=False)
    opt2.set_amount_unit("g")
    opt2.load_ingredients_from_csv(pd.DataFrame({
        "Name": ["Flour", "Water"], "Lowest": [10, 0], "Highest": [60, 100],
        "Unit": ["g", "g"], "Rule": [None, "= rest"]}))
    # Numbers a file gives the '= rest' row are its range; a Lowest of 0
    # says no more than the floor does.
    assert opt2._var_by_name("Water")['fill_limits'] == [None, 100.0]
    assert opt2._held_range(opt2._var_by_name("Water")) == (0.0, 100.0)


def test_limits_nothing_meets_are_said_as_limits(tmp_path, monkeypatch):
    """A total the allowed amounts reach, and a limit nothing meets at it:
    the sentence blames the limits, not the size."""
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("limits_case", robust=False)
    opt.set_amount_unit("g")
    opt.add_ingredient("Flour", 10, 80)
    opt.add_ingredient("Sugar", 5, 60)
    opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
    opt.set_formulation_total(100)
    opt.quantity_constraints.append({"ingredients": ["Flour"], "min": 79.9,
                                     "max": 80.0})
    opt.quantity_constraints.append({"ingredients": ["Sugar"], "min": None,
                                     "max": 5.0})
    with pytest.raises(ValueError) as refused:
        opt._ask_cold_start(3, opt._get_bounds(), 2)
    assert str(refused.value) == wording.no_formulation_meets_limits("100 g")


def test_the_set_up_sheet_carries_the_fill_range(tmp_path, monkeypatch):
    import io
    import openpyxl
    opt = _project(tmp_path, monkeypatch)
    frame, at = _water_row(opt)
    frame.at[at, H] = ""
    opt.apply_ingredient_grid(frame)
    _score(opt, opt.ask(2))
    book = openpyxl.load_workbook(io.BytesIO(opt.all_formulations_workbook()))
    rows = [row for sheet in book.worksheets
            for row in sheet.iter_rows(values_only=True)
            if row and row[0] == "Water" and len(row) > 4
            and row[1] == wording.KIND_INGREDIENT]
    assert rows and rows[0][2:4] == (20.0, None)


# ------------------------------------------------------------------ #
# Review fixes: a pinned filler, the message only a binding range earns,
# edits of the open Round, a no-op save, rows that are not the filler, and
# files with words in the filler's cells.
# ------------------------------------------------------------------ #

def test_a_pinned_fill_row_gets_full_rounds_at_its_one_amount(tmp_path, monkeypatch):
    """Lowest equal to Highest is an equality: round 1 and the model both
    give every formulation asked for, each with Water at exactly 30 g."""
    opt = _project(tmp_path, monkeypatch, water=(30, 30))
    assert opt._fill_pinned_at(opt._var_by_name("Water")) == 30.0
    assert len(opt._get_botorch_equalities()) == 1
    first = opt.ask(5)
    assert len(first) == 5
    assert len({round(r["Flour"], 6) for r in first}) == 5
    for recipe in first:
        assert recipe["Water"] == pytest.approx(30.0, abs=1e-6)
        assert opt._check_constraints(recipe)
    _score(opt, first)
    warm = opt.ask(3)
    assert len(warm) == 3
    for recipe in warm:
        assert recipe["Water"] == pytest.approx(30.0, abs=1e-4)
        assert sum(recipe.values()) == pytest.approx(100.0, abs=1e-4)


def test_a_pinned_fill_row_out_of_reach_is_refused_clearly(tmp_path, monkeypatch):
    opt = _project(tmp_path, monkeypatch, water=(30, 30))
    _type_range(opt, 90, 90)
    with pytest.raises(ValueError) as refused:
        opt.ask(3)
    assert str(refused.value) == wording.fill_outside_its_range(
        "Water", "90–90 g", wording.at_most_text("85 g"))
    assert not opt.pending_batch


def test_projects_without_a_pinned_filler_get_no_equality(tmp_path, monkeypatch):
    opt = _project(tmp_path, monkeypatch)
    assert opt._get_botorch_equalities() == []
    plain = _project(tmp_path / "b", monkeypatch, water=(0, 0), name="b")
    assert plain._get_botorch_equalities() == []


def test_a_range_that_cannot_bind_is_not_named_when_nothing_is_found(
        tmp_path, monkeypatch):
    """Generate finds nothing because of an amount limit; Water's 0–100 g
    never binds (the others leave it 0–85 g), so it is not named."""
    opt = _project(tmp_path, monkeypatch, water=(0, 100))
    opt.quantity_constraints.append({"ingredients": ["Flour"], "min": 79.9,
                                     "max": 80.0})
    opt.quantity_constraints.append({"ingredients": ["Sugar"], "min": None,
                                     "max": 5.0})
    monkeypatch.setattr(FoodOptimizer, "_sample_feasible",
                        lambda self, *a, **k: [])
    with pytest.raises(ValueError) as refused:
        opt._ask_cold_start(3, opt._get_bounds(), 2)
    assert str(refused.value) == wording.no_formulation_meets_limits("100 g")
    # A range that can bind is named.
    _type_range(opt, 20, 40)
    with pytest.raises(ValueError) as refused:
        opt._ask_cold_start(3, opt._get_bounds(), 2)
    assert str(refused.value) == wording.no_formulation_with_fill_range(
        "100 g", "Water", "20–40 g")


def test_an_edit_that_takes_the_fill_row_out_of_its_range_is_refused(
        tmp_path, monkeypatch):
    opt = _project(tmp_path, monkeypatch)
    opt.ask(2)
    row = opt.pending_batch[0]
    before = dict(row['recipe'])
    with pytest.raises(ValueError) as refused:
        opt.edit_pending_formulations({row['formulation']: {"Flour": 10,
                                                            "Sugar": 5}})
    assert str(refused.value) == wording.edit_formulation_error(
        row['formulation'], opt.bounds_caution("Water", 85.0))
    assert opt.pending_batch[0]['recipe'] == before
    # Inside the range it is accepted, and Water fills to the rest.
    assert opt.edit_pending_formulations({row['formulation']: {"Flour": 40,
                                                               "Sugar": 30}})
    assert opt.pending_batch[0]['recipe']["Water"] == pytest.approx(30.0)


def test_an_own_formulation_keeps_the_caution_rule(tmp_path, monkeypatch):
    """A formulation of the reader's own is never refused for an amount
    outside the allowed ones (ui_batch._add_own cautions instead); the Fill
    to total row's caution is the same sentence."""
    opt = _project(tmp_path, monkeypatch)
    number = opt.add_to_pending_batch({"Flour": 10, "Sugar": 5})
    added = next(r for r in opt.pending_batch if r['formulation'] == number)
    assert added['recipe']["Water"] == pytest.approx(85.0)
    assert opt.bounds_caution("Water", added['recipe']["Water"]) != ""


def test_typing_zero_in_lowest_of_a_no_range_filler_moves_nothing(
        tmp_path, monkeypatch):
    opt = _project(tmp_path, monkeypatch, water=(0, 0))
    opt.ask(2)
    frame, at = _water_row(opt)
    frame.at[at, L] = "0"
    assert opt.ingredient_grid_retires_round(frame) is None
    opt.apply_ingredient_grid(frame)
    assert opt.pending_batch_no is not None
    assert 'fill_limits' not in opt._var_by_name("Water")


def test_a_fill_range_on_another_row_is_dropped_on_load(tmp_path, monkeypatch):
    opt = _project(tmp_path, monkeypatch)
    state = opt.export_json()
    flour = next(v for v in state['variables'] if v['name'] == "Flour")
    flour['fill_limits'] = [1.0, 2.0]
    back = FoodOptimizer("stray", robust=False)
    back.import_json(state)
    assert 'fill_limits' not in back._var_by_name("Flour")
    assert back._var_by_name("Water")['fill_limits'] == [20.0, 40.0]


def test_a_file_with_words_in_the_fill_rows_cells_still_loads(
        tmp_path, monkeypatch):
    import pandas as pd
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("words", robust=False)
    opt.set_amount_unit("g")
    opt.load_ingredients_from_csv(pd.DataFrame({
        "Name": ["Flour", "Water"], "Lowest": [10, "n/a"],
        "Highest": [60, "calculated"], "Unit": ["g", "g"],
        "Rule": [None, "= rest"]}))
    water = opt._var_by_name("Water")
    assert water['balance'] and 'fill_limits' not in water
    assert opt._fill_range() is None


def test_the_burger_example_generates_at_200_and_500_g(tmp_path, monkeypatch):
    """As 0.9.0 does: the example's Remaining water has no range typed."""
    import sample_projects as sp
    import storage
    monkeypatch.chdir(tmp_path)
    store = storage.LocalStorage()
    sp.ensure(store, (sp.BURGER,))
    burger = FoodOptimizer(sp.BURGER, storage=store)
    assert burger._fill_range() is None
    frame = burger.ingredient_grid_frame()
    water = frame[frame[wording.NAME_LABEL] == "Remaining water"].iloc[0]
    assert (water[L], water[H]) == ("", "")
    for size in (200, 500):
        burger.set_formulation_total(size)
        assert len(burger.ask(3)) == 3
