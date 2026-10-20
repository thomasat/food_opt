"""An amount limit may carry a name of its own (e.g. one inferred from a
project's past Rounds): it is listed, deleted and printed under that name,
and it changes nothing about what the limit enforces or about the
formulations already recorded."""
import pickle

import wording
from food_bo import FoodOptimizer


def _opt(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("named limit")
    opt.set_amount_unit("g")
    opt.add_ingredient("Water", 20.0, 60.0)
    opt.add_ingredient("Oil", 5.0, 40.0)
    opt.add_objective("Taste", 1.0, goal="max", min_val=0.0, max_val=10.0)
    return opt


def test_a_named_limit_is_listed_by_its_name(tmp_path, monkeypatch):
    opt = _opt(tmp_path, monkeypatch)
    opt.add_total_mass_constraint(60.0, 70.0, name="  Batch total (inferred)  ")
    qc = opt.quantity_constraints[-1]
    assert qc["name"] == "Batch total (inferred)"
    assert opt.limit_label(qc) == "Batch total (inferred)"
    assert opt.limit_text(qc) == "Batch total (inferred): at least 60 g and at most 70 g"
    assert qc["min"] == 60.0 and qc["max"] == 70.0 and "source" not in qc


def test_an_unnamed_limit_keeps_its_old_name(tmp_path, monkeypatch):
    opt = _opt(tmp_path, monkeypatch)
    opt.add_total_mass_constraint(60.0, 70.0)
    assert "name" not in opt.quantity_constraints[-1]
    assert opt.limit_label(opt.quantity_constraints[-1]) == wording.ALL_INGREDIENTS_LABEL
    opt.add_quantity_constraint(["Water"], max_val=50.0, name="  ")
    assert "name" not in opt.quantity_constraints[-1]
    assert opt.limit_label(opt.quantity_constraints[-1]) == "Water"


def test_a_named_limit_bounds_suggestions_and_leaves_records_alone(tmp_path, monkeypatch):
    opt = _opt(tmp_path, monkeypatch)
    made = {"Water": 55.0, "Oil": 30.0}            # 85 g: outside the limit below
    opt.tell(dict(made), {"Taste": 5.0})
    opt.add_total_mass_constraint(60.0, 70.0, name="Batch total (inferred)")
    assert opt.recipe_history == [made]
    assert not opt._check_constraints(made)
    assert opt._check_constraints({"Water": 40.0, "Oil": 25.0})
    assert not opt._check_constraints({"Water": 40.0, "Oil": 35.0})


def test_the_name_survives_save_and_export(tmp_path, monkeypatch):
    opt = _opt(tmp_path, monkeypatch)
    opt.add_total_mass_constraint(60.0, 70.0, name="Batch total (inferred)")
    again = FoodOptimizer("named limit")
    assert again.limit_label(again.quantity_constraints[-1]) == "Batch total (inferred)"
    exported = opt.export_json()
    assert exported["quantity_constraints"][-1]["name"] == "Batch total (inferred)"
