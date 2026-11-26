"""Range edits that cannot stand (stable 940a0ce `set_variable_bounds`, A1).

An ingredient's Lowest below 0 is refused, and so is a change of a row's
Lowest or Highest that leaves a limit no formulation can meet. Only a limit
THIS edit breaks is refused: one already out of reach before it is not the
edit's doing, which is main's before/after rule, not stable's whole-state
check."""
import json

import pytest

import wording
from food_bo import FoodOptimizer


@pytest.fixture
def opt(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    o = FoodOptimizer(project_name="range_guard", robust=False)
    o.set_amount_unit("g")
    o.add_ingredient("A", 0, 10)
    o.add_ingredient("B", 0, 10)
    o.add_ingredient("C", 0, 10)
    o.add_objective("Taste", 1.0, goal="max")
    return o


def _state(o):
    return json.dumps(o.export_json(), sort_keys=True, default=str)


def _bounds(o, name):
    return o._var_by_name(name)['bounds']


def _row(frame, name):
    return frame.index[frame[wording.NAME_LABEL] == name][0]


# ---- a Lowest below 0 ---------------------------------------------------- #

def test_negative_lowest_refused(opt):
    before = _state(opt)
    with pytest.raises(ValueError) as caught:
        opt.add_ingredient("D", -1, 5)
    assert str(caught.value) == wording.lowest_below_zero(-1)
    assert _state(opt) == before


def test_negative_lowest_refused_on_an_existing_row(opt):
    before = _state(opt)
    with pytest.raises(ValueError):
        opt.add_ingredient("A", -1, 5)
    assert _bounds(opt, "A") == (0.0, 10.0)
    assert _state(opt) == before


def test_a_setting_may_go_below_zero(opt):
    """A freezer at -18 °C is a setting, not an amount."""
    opt.add_process_parameter("Freezer", -25, -10, unit="°C")
    assert _bounds(opt, "Freezer") == (-25.0, -10.0)


def test_a_negative_lowest_already_on_file_does_not_block_the_row(opt):
    """A project saved before this check may hold one. Editing its Highest
    is not this save putting the Lowest below 0."""
    opt._var_by_name("A")['bounds'] = (-1.0, 10.0)
    opt.add_ingredient("A", -1, 8)
    assert _bounds(opt, "A") == (-1.0, 8.0)


def test_grid_negative_lowest_refused_by_row(opt):
    before = _state(opt)
    frame = opt.ingredient_grid_frame()
    row = _row(frame, "A")
    frame.loc[row, wording.LOWEST_LABEL] = -1.0
    errors, messages = opt.apply_ingredient_grid(frame)
    assert errors == [(row, wording.lowest_below_zero(-1))]
    assert messages == []
    assert _state(opt) == before


# ---- a range edit that strands a limit ----------------------------------- #

def test_narrowing_that_strands_a_limit_refused(opt):
    opt.add_quantity_constraint(["A", "B"], min_val=15)
    before = _state(opt)
    with pytest.raises(ValueError) as caught:
        opt.add_ingredient("A", 0, 3)
    assert str(caught.value) == wording.limit_unreachable_above(
        "A + B", "13", lead=wording.changing_rows("A"))
    assert _bounds(opt, "A") == (0.0, 10.0)
    assert _state(opt) == before


def test_narrowing_that_keeps_the_limit_reachable_is_saved(opt):
    opt.add_quantity_constraint(["A", "B"], min_val=15)
    opt.add_ingredient("A", 0, 5)
    assert _bounds(opt, "A") == (0.0, 5.0)


def test_raising_a_lowest_past_an_at_most_refused(opt):
    opt.add_quantity_constraint(["A", "B"], max_val=6)
    with pytest.raises(ValueError) as caught:
        opt.add_ingredient("A", 8, 10)
    assert str(caught.value) == wording.limit_pinned_amounts_exceed(
        "A + B", "8", lead=wording.changing_rows("A"))
    assert "Lowest amounts" in str(caught.value)
    assert _bounds(opt, "A") == (0.0, 10.0)


def test_narrowing_past_the_batch_size_clears_it_rather_than_refusing(opt):
    """The batch size has its own answer to a range edit that puts it out
    of reach: it is cleared, and the screen says so. That stays."""
    opt.set_formulation_total(25.0)
    removed = opt.add_ingredient("A", 0, 2)
    assert _bounds(opt, "A") == (0.0, 2.0)
    assert opt.formulation_total is None
    assert [r['reason'] for r in removed
            if r.get('source') == 'formulation_total'] == ['unreachable']


def test_fixing_past_the_batch_size_is_still_refused(opt):
    opt.set_formulation_total(25.0)
    with pytest.raises(ValueError) as caught:
        opt.add_ingredient("A", 2, 2)
    assert str(caught.value) == (wording.fixing_breaks_the_total("25 g")
                                 + " " + wording.fixed_rows_tail("A"))
    assert _bounds(opt, "A") == (0.0, 10.0)


def test_grid_narrowing_past_the_batch_size_clears_it(opt):
    opt.set_formulation_total(25.0)
    frame = opt.ingredient_grid_frame()
    frame.loc[_row(frame, "A"), wording.HIGHEST_LABEL] = 2.0
    errors, _ = opt.apply_ingredient_grid(frame)
    assert errors == []
    assert opt.formulation_total is None


def test_edit_not_blamed_for_a_limit_already_broken(opt):
    """The limit was put out of reach before this edit, through the state
    rather than the guarded door; editing another row, or the same limit's
    own rows, must not be refused for it."""
    opt.add_quantity_constraint(["A", "B"], min_val=15)
    opt._var_by_name("A")['bounds'] = (0.0, 3.0)
    assert opt._limit_refusals()            # already out of reach
    opt.add_ingredient("C", 0, 5)
    assert _bounds(opt, "C") == (0.0, 5.0)
    opt.add_ingredient("B", 0, 9)
    assert _bounds(opt, "B") == (0.0, 9.0)


def test_grid_narrowing_that_strands_a_limit_refused(opt):
    opt.add_quantity_constraint(["A", "B"], min_val=15)
    before = _state(opt)
    frame = opt.ingredient_grid_frame()
    frame.loc[_row(frame, "A"), wording.HIGHEST_LABEL] = 3.0
    errors, messages = opt.apply_ingredient_grid(frame)
    assert errors == [(None, wording.limit_unreachable_above(
        "A + B", "13", lead=wording.changing_rows("A")))]
    assert messages == []
    assert _state(opt) == before


def test_grid_not_blamed_for_a_limit_already_broken(opt):
    opt.add_quantity_constraint(["A", "B"], min_val=15)
    opt._var_by_name("A")['bounds'] = (0.0, 3.0)
    frame = opt.ingredient_grid_frame()
    frame.loc[_row(frame, "C"), wording.HIGHEST_LABEL] = 5.0
    errors, _ = opt.apply_ingredient_grid(frame)
    assert errors == []
    assert _bounds(opt, "C") == (0.0, 5.0)


def test_grid_fixing_alone_keeps_its_own_sentence(opt):
    """Nothing narrowed, one row fixed: the refusal is the one the grid has
    always given, "Fixing these …" with the fixed rows named at the end."""
    opt.add_quantity_constraint(["A"], max_val=5)
    frame = opt.ingredient_grid_frame()
    row = _row(frame, "A")
    frame.loc[row, wording.LOWEST_LABEL] = 8.0
    frame.loc[row, wording.HIGHEST_LABEL] = 8.0
    errors, _ = opt.apply_ingredient_grid(frame)
    assert errors == [(None, wording.limit_pinned_amounts_exceed("A", "8")
                       + " " + wording.fixed_rows_tail("A"))]


def test_the_changing_lead_is_short_and_specific():
    assert wording.changing_rows("A and B") == "Changing A and B"
    assert len(wording.lowest_below_zero(-1).split()) <= 10


def test_ingredient_file_refuses_a_negative_lowest(opt):
    import pandas as pd
    before = _state(opt)
    with pytest.raises(ValueError) as caught:
        opt.load_ingredients_from_csv(pd.DataFrame({
            "Name": ["Oil", "Salt"], "Lowest": [0, -2], "Highest": [10, 5]}))
    assert str(caught.value) == wording.file_lowest_below_zero("Salt", -2)
    assert wording.lowest_below_zero(-2) in str(caught.value)
    assert _state(opt) == before
