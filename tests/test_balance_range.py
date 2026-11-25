"""The Fill to total row is held inside its own Lowest and Highest.

Ported from the web app's balance ingredient (e2752b7): the amount is the
batch size less every other ingredient, kept within a min and a max. In the
desktop that row is written '= rest'; its range is the two numbers it keeps
(var['bounds']), and (0, 0) — a row that arrived already worked out — is no
range of its own, only the floor of nothing every calculated row has.
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
    return opt


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
    # A row with no range of its own keeps only the floor of nothing.
    water['bounds'] = (0.0, 0.0)
    assert opt._held_range(water) == (0.0, None)
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
        "Water", "20.00 to 40.00 g", wording.at_most_text("0.00 g"))
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
        "Water", "20.00 to 40.00 g", wording.at_least_text("75.00 g"))


def test_the_default_batch_size_box_asks_the_fill_rows_range(tmp_path, monkeypatch):
    opt = _project(tmp_path, monkeypatch)
    # Flour and Sugar need at least 15 g, so Water's 20 g floor needs 35 g.
    with pytest.raises(ValueError) as refused:
        opt.set_formulation_total(30)
    assert str(refused.value) == wording.balance_would_go_negative(
        "Water", opt.batch_total_text(30), opt.batch_total_text(35),
        unit="g", floor_text="20.00 g")
    # The top is not the box's to refuse (a rest row has no largest batch
    # size): Flour and Sugar take up at most 140 g, so at 200 g Water is at
    # least 60 g, and Generate and the caption say so in Water's name.
    opt.set_formulation_total(200)
    assert opt.formulation_total == 200
    expected = wording.fill_outside_its_range(
        "Water", "20.00 to 40.00 g", wording.at_least_text("60.00 g"))
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
    opt._var_by_name("Water")['bounds'] = (0.0, 0.0)
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
