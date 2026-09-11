"""Tests for the FoodOptimizer core engine."""

import io
import json
import os
import pickle
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from food_bo import FoodOptimizer


@pytest.fixture
def opt(tmp_path, monkeypatch):
    """Create a FoodOptimizer that saves into a temp directory."""
    monkeypatch.chdir(tmp_path)
    return FoodOptimizer(project_name="test_project", robust=False)


@pytest.fixture
def opt_with_ingredients(opt):
    """Optimizer pre-loaded with three ingredients."""
    opt.add_ingredient("Water", 0, 100)
    opt.add_ingredient("Flour", 0, 50)
    opt.add_ingredient("Sugar", 0, 30)
    return opt


@pytest.fixture
def opt_configured(opt_with_ingredients):
    """Fully configured optimizer with ingredients + objective."""
    opt = opt_with_ingredients
    opt.add_objective("Taste", weight=1.0, goal="max", min_val=0, max_val=10)
    return opt


# ------------------------------------------------------------------ #
#  Initialization
# ------------------------------------------------------------------ #


class TestInit:
    def test_creates_pkl_file(self, opt):
        opt.add_ingredient("Water", 0, 100)
        assert os.path.exists(opt.filename)

    def test_default_state(self, opt):
        assert opt.variables == []
        assert opt.objectives == []
        assert opt.X_history == []
        assert opt.Y_history == []
        assert opt.constraints == []
        assert opt.quantity_constraints == []

    def test_robust_flag(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer(project_name="robust_test", robust=True)
        assert opt.robust is True


# ------------------------------------------------------------------ #
#  Variables (Ingredients & Process Parameters)
# ------------------------------------------------------------------ #


class TestVariables:
    def test_add_ingredient(self, opt):
        opt.add_ingredient("Water", 0, 100)
        assert len(opt.variables) == 1
        assert opt.variables[0]["name"] == "Water"
        assert opt.variables[0]["bounds"] == (0.0, 100.0)
        assert opt.variables[0]["category"] == "ingredient"

    def test_add_ingredient_idempotent(self, opt):
        opt.add_ingredient("Water", 0, 100)
        opt.add_ingredient("Water", 0, 200)  # same name, should not duplicate
        assert len(opt.variables) == 1

    def test_add_process_parameter(self, opt):
        opt.add_process_parameter("Temperature", 100, 250)
        assert opt.variables[0]["category"] == "process"
        assert opt.variables[0]["bounds"] == (100.0, 250.0)

    def test_remove_process_parameter(self, opt):
        opt.add_process_parameter("Temperature", 100, 250)
        opt.remove_process_parameter("Temperature")
        assert len(opt.variables) == 0

    def test_remove_process_parameter_reencodes_history(self, opt):
        """Removing a process parameter must re-encode X_history to the new
        dimension (mirrors remove_ingredient), otherwise a later ask() fails
        with a tensor size mismatch."""
        opt.add_ingredient("Water", 0, 100)
        opt.add_process_parameter("Temp", 100, 250, baseline=150)
        opt.add_objective("Taste", 1.0, goal="max")
        opt.tell({"Water": 50.0, "Temp": 150.0}, {"Taste": 7.0})
        opt.tell({"Water": 60.0, "Temp": 150.0}, {"Taste": 6.0})
        opt.remove_process_parameter("Temp")
        assert all(len(x) == len(opt.variables) for x in opt.X_history)
        assert opt.pending_batch is None

    def test_remove_process_does_not_remove_ingredient(self, opt):
        opt.add_ingredient("Water", 0, 100)
        opt.remove_process_parameter("Water")
        assert len(opt.variables) == 1  # ingredient should remain

    def test_load_ingredients_from_csv(self, opt):
        df = pd.DataFrame({
            "Name": ["Water", "Flour", "Sugar"],
            "Min": [0, 0, 0],
            "Max": [100, 50, 30],
            "Fat": [0.0, 1.0, 0.0],
        })
        opt.load_ingredients_from_csv(df)
        assert len(opt.variables) == 3
        assert "Water" in opt.ingredient_properties
        # The file's own capitalisation is kept: the picker and the limits
        # list show this name.
        assert opt.ingredient_properties["Flour"]["Fat"] == 1.0

    def test_load_ingredients_csv_lowercase_columns(self, opt):
        """User CSVs vary in header case; lowercase must work (the shipped
        example file uses name/min/max)."""
        df = pd.DataFrame({
            "name": ["Water", "Flour"],
            "min": [0, 0],
            "max": [100, 50],
            "fat_per_100g": [0.0, 1.8],
        })
        opt.load_ingredients_from_csv(df)
        assert len(opt.variables) == 2
        assert opt.variables[0]["name"] == "Water"
        assert opt.ingredient_properties["Flour"]["fat_per_100g"] == 1.8

    def test_load_ingredients_csv_missing_column_plain_error(self, opt):
        """A missing required column must raise a plain-language ValueError
        (which the app displays nicely), never a raw KeyError."""
        df = pd.DataFrame({"Name": ["Water"], "Min": [0]})
        with pytest.raises(ValueError, match="missing required column"):
            opt.load_ingredients_from_csv(df)

    def test_load_csv_rejects_after_experiments(self, opt_configured):
        opt = opt_configured
        recipe = {"Water": 50, "Flour": 25, "Sugar": 10}
        opt.tell(recipe, {"Taste": 7.0})

        df = pd.DataFrame({
            "Name": ["Water", "Flour"],
            "Min": [0, 0],
            "Max": [100, 50],
        })
        with pytest.raises(ValueError, match="Cannot reload ingredients"):
            opt.load_ingredients_from_csv(df)

    def test_load_csv_min_ge_max_raises(self, opt):
        df = pd.DataFrame({
            "Name": ["BadIngredient"],
            "Min": [50],
            "Max": [50],
        })
        with pytest.raises(ValueError, match="Lowest.*must be less than Highest"):
            opt.load_ingredients_from_csv(df)

    def test_csv_preserves_process_params(self, opt):
        opt.add_process_parameter("Temperature", 100, 250)
        df = pd.DataFrame({
            "Name": ["Water"],
            "Min": [0],
            "Max": [100],
        })
        opt.load_ingredients_from_csv(df)
        names = [v["name"] for v in opt.variables]
        assert "Water" in names
        assert "Temperature" in names


# ------------------------------------------------------------------ #
#  Objectives
# ------------------------------------------------------------------ #


class TestObjectives:
    def test_add_objective(self, opt):
        opt.add_objective("Taste", weight=0.7, goal="max", min_val=0, max_val=10)
        assert len(opt.objectives) == 1
        assert opt.objectives[0]["weight"] == 0.7

    def test_add_objective_replaces_same_name(self, opt):
        opt.add_objective("Taste", weight=0.5, goal="max")
        opt.add_objective("Taste", weight=0.9, goal="min")
        assert len(opt.objectives) == 1
        assert opt.objectives[0]["weight"] == 0.9
        assert opt.objectives[0]["goal"] == "min"

    def test_remove_objective(self, opt):
        opt.add_objective("Taste", weight=1.0, goal="max")
        opt.remove_objective("Taste")
        assert len(opt.objectives) == 0


class TestObjectiveValidation:
    def _opt(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("obj_val")
        opt.add_ingredient("Water", 0, 100)
        return opt

    def test_readding_objective_recomputes_history(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
        opt.tell({"Water": 5.0}, {"Taste": 5.0})
        assert opt.Y_history[0] == pytest.approx(0.5)
        replaced = opt.add_objective("Taste", 0.5, goal="max", min_val=0, max_val=10)
        assert replaced is True
        assert opt.Y_history[0] == pytest.approx(0.25)

    def test_weight_must_be_positive(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        with pytest.raises(ValueError, match="greater than 0"):
            opt.add_objective("Taste", 0.0)

    def test_target_must_lie_in_range(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        with pytest.raises(ValueError, match="must be between the range"):
            opt.add_objective("Taste", 1.0, goal="target", target=50, min_val=0, max_val=10)

    def test_blank_name_rejected(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        with pytest.raises(ValueError, match="cannot be empty"):
            opt.add_objective("   ", 1.0)

    def test_utility_ceiling_is_weight_sum(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_objective("Taste", 0.6)
        opt.add_objective("Cost", 0.4, goal="min")
        assert opt.utility_ceiling() == pytest.approx(1.0)

    def test_objective_name_collides_with_ingredient(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        with pytest.raises(ValueError, match="already the name of an ingredient"):
            opt.add_objective("water", 1.0)


def test_best_index_and_running_max(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("best")
    opt.add_ingredient("Water", 0, 100)
    opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
    assert opt.best_index() is None
    for water, taste in [(10.0, 3.0), (20.0, 8.0), (30.0, 5.0)]:
        opt.tell({"Water": water}, {"Taste": taste})
    assert opt.best_index() == 1
    assert opt.best_so_far() == pytest.approx([0.3, 0.8, 0.8])


def test_history_frame_names_formulations_not_experiments(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("hist")
    opt.add_ingredient("Water", 0, 100)
    opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
    opt.tell({"Water": 10.0}, {"Taste": 3.0})
    opt.tell({"Water": 20.0}, {"Taste": 8.0})
    df = opt.history_frame(order="Newest first")
    assert list(df.columns[:3]) == ["Best", "Batch", "Formulation"]
    assert list(df["Formulation"]) == [2, 1]
    assert "Taste" in df.columns
    assert len(df["Recorded"].iloc[0]) == 10


def test_reserved_column_name_is_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("reserved")
    with pytest.raises(ValueError, match="column name Food Optimizer uses"):
        opt.add_ingredient("Date", 0, 10)


def test_batch_frame_has_formulation_numbers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("bf")
    opt.add_ingredient("Water", 0, 100)
    opt.add_process_parameter("Temp", 100, 200)
    opt.set_pending_batch([{"Water": 10.0, "Temp": 180.0},
                           {"Water": 20.0, "Temp": 190.0}])
    df = opt.batch_frame(opt.pending_batch)
    assert list(df["Formulation"]) == [1, 2]
    # Ingredient columns carry the project unit (g is a new project's default);
    # a process setting is not an amount, so a cook temperature is never "(g)".
    # The total closes the amounts you weigh out, so it comes straight after
    # the ingredients and before the settings you dial in.
    assert list(df.columns) == ["Formulation", "Water (g)", "Total (g)", "Temp"]
    assert df["Total (g)"].iloc[0] == 10.0   # the process setting is not an amount


def test_recipe_lines_sorts_largest_first_and_omits_zeros(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("bf")
    recipe = {"Water": 1.0, "Salt": 0.0, "Sugar": 5.0}
    assert opt.recipe_lines(recipe) == [("Sugar", 5.0), ("Water", 1.0)]
    assert opt.recipe_lines(recipe, limit=1) == [("Sugar", 5.0)]


def test_batch_csv_rounds_to_two_decimals(tmp_path, monkeypatch):
    """The downloaded sheet must match the two-decimal table shown on screen,
    not the raw float precision the optimizer suggests."""
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("bf2")
    opt.add_ingredient("Water", 0, 100)
    opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
    opt.set_pending_batch([{"Water": 11.877679824829102}])
    df = pd.read_csv(io.StringIO(opt.batch_csv(opt.pending_batch)))
    # The sheet's amount columns carry the ingredient's unit, as the batch
    # table's do: a bare "Water" column left the lab guessing.
    assert df["Water (g)"].iloc[0] == 11.88
    assert list(df["Formulation"]) == [1]


def test_history_csv_roundtrips_through_importer_columns(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("csv")
    opt.add_ingredient("Water", 0, 100)
    opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
    opt.tell({"Water": 10.0}, {"Taste": 3.0})
    df = pd.read_csv(io.StringIO(opt.history_csv()))
    for col in ["Formulation", "Batch", "Recorded", "Overall score", "Water",
                "Taste", "Note"]:
        assert col in df.columns
    assert df["Taste"].iloc[0] == 3.0


def test_history_csv_backfills_variable_added_mid_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("csv_midrun")
    opt.add_ingredient("Water", 0, 100)
    opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
    opt.tell({"Water": 10.0}, {"Taste": 3.0})
    opt.tell({"Water": 20.0}, {"Taste": 5.0})
    opt.add_ingredient("Honey", 0, 30)
    import io
    df = pd.read_csv(io.StringIO(opt.history_csv()))
    assert "Honey" in df.columns
    assert len(df) == 2
    assert df["Honey"].notna().all()


# ------------------------------------------------------------------ #
#  Constraints
# ------------------------------------------------------------------ #


class TestConstraints:
    def test_add_property_constraint(self, opt):
        opt.add_constraint("fat", min_val=0, max_val=5)
        assert len(opt.constraints) == 1
        assert opt.constraints[0]["metric"] == "fat"

    def test_remove_property_constraint(self, opt):
        opt.add_constraint("fat", min_val=0, max_val=5)
        opt.remove_constraint(0)
        assert len(opt.constraints) == 0

    def test_remove_constraint_out_of_range(self, opt):
        opt.remove_constraint(99)  # should not raise
        assert len(opt.constraints) == 0

    def test_add_quantity_constraint(self, opt_with_ingredients):
        opt = opt_with_ingredients
        opt.add_quantity_constraint(["Water", "Flour"], min_val=50, max_val=100)
        assert len(opt.quantity_constraints) == 1
        assert opt.quantity_constraints[0]["ingredients"] == ["Water", "Flour"]

    def test_add_total_mass_constraint(self, opt_with_ingredients):
        opt = opt_with_ingredients
        opt.add_total_mass_constraint(min_val=80, max_val=120)
        assert len(opt.quantity_constraints) == 1
        all_names = [v["name"] for v in opt.variables]
        assert opt.quantity_constraints[0]["ingredients"] == all_names

    def test_remove_quantity_constraint(self, opt_with_ingredients):
        opt = opt_with_ingredients
        opt.add_quantity_constraint(["Water"], min_val=10)
        opt.remove_quantity_constraint(0)
        assert len(opt.quantity_constraints) == 0


# ------------------------------------------------------------------ #
#  Utility Scoring
# ------------------------------------------------------------------ #


class TestUtility:
    def test_maximize(self, opt):
        opt.add_objective("Score", weight=1.0, goal="max", min_val=0, max_val=10)
        assert opt._compute_utility({"Score": 10}) == pytest.approx(1.0)
        assert opt._compute_utility({"Score": 0}) == pytest.approx(0.0)
        assert opt._compute_utility({"Score": 5}) == pytest.approx(0.5)

    def test_minimize(self, opt):
        opt.add_objective("Cost", weight=1.0, goal="min", min_val=0, max_val=10)
        assert opt._compute_utility({"Cost": 0}) == pytest.approx(1.0)
        assert opt._compute_utility({"Cost": 10}) == pytest.approx(0.0)

    def test_target(self, opt):
        opt.add_objective("pH", weight=1.0, goal="target", target=5.0,
                          min_val=0, max_val=10)
        assert opt._compute_utility({"pH": 5.0}) == pytest.approx(1.0)
        assert opt._compute_utility({"pH": 0.0}) == pytest.approx(0.5)
        assert opt._compute_utility({"pH": 10.0}) == pytest.approx(0.5)

    def test_weighted_multi_objective(self, opt):
        opt.add_objective("Taste", weight=0.6, goal="max", min_val=0, max_val=10)
        opt.add_objective("Cost", weight=0.4, goal="min", min_val=0, max_val=10)
        # Taste=10 -> 1.0*0.6=0.6, Cost=0 -> 1.0*0.4=0.4 => total 1.0
        assert opt._compute_utility({"Taste": 10, "Cost": 0}) == pytest.approx(1.0)

    def test_missing_objective_ignored(self, opt):
        opt.add_objective("Taste", weight=1.0, goal="max", min_val=0, max_val=10)
        # Missing key is skipped gracefully
        assert opt._compute_utility({"Other": 5}) == pytest.approx(0.0)

    def test_clamps_to_0_1(self, opt):
        opt.add_objective("X", weight=1.0, goal="max", min_val=0, max_val=10)
        # Values outside range get clamped
        assert opt._compute_utility({"X": 20}) == pytest.approx(1.0)
        assert opt._compute_utility({"X": -5}) == pytest.approx(0.0)


# ------------------------------------------------------------------ #
#  Encode / Decode
# ------------------------------------------------------------------ #


class TestEncodeDecode:
    def test_roundtrip(self, opt_with_ingredients):
        opt = opt_with_ingredients
        recipe = {"Water": 50.0, "Flour": 25.0, "Sugar": 10.0}
        encoded = opt._encode(recipe)
        decoded = opt._decode(encoded)
        for key in recipe:
            assert decoded[key] == pytest.approx(recipe[key])

    def test_encode_length(self, opt_with_ingredients):
        opt = opt_with_ingredients
        recipe = {"Water": 50, "Flour": 25, "Sugar": 10}
        encoded = opt._encode(recipe)
        assert len(encoded) == 3

    def test_bounds_shape(self, opt_with_ingredients):
        opt = opt_with_ingredients
        bounds = opt._get_bounds()
        assert bounds.shape == (2, 3)
        assert bounds[0, 0].item() == 0.0   # Water min
        assert bounds[1, 0].item() == 100.0  # Water max


# ------------------------------------------------------------------ #
#  Constraint Checking
# ------------------------------------------------------------------ #


class TestConstraintChecking:
    def test_property_constraint_pass(self, opt):
        df = pd.DataFrame({
            "Name": ["Water", "Oil"],
            "Min": [0, 0],
            "Max": [100, 50],
            "Fat": [0.0, 0.8],
        })
        opt.load_ingredients_from_csv(df)
        opt.add_constraint("fat", max_val=20)
        assert opt._check_constraints({"Water": 50, "Oil": 10}) is True

    def test_property_constraint_fail(self, opt):
        """Per 100 g of the formulation: 50 g of water at 0 fat and 40 g of
        oil at 80 fat is 35.6 fat per 100 g, over a limit of 5."""
        df = pd.DataFrame({
            "Name": ["Water", "Oil"],
            "Min": [0, 0],
            "Max": [100, 50],
            "Fat": [0.0, 80.0],
        })
        opt.load_ingredients_from_csv(df)
        opt.add_constraint("fat", max_val=5)
        assert opt._check_constraints({"Water": 50, "Oil": 40}) is False

    def test_quantity_constraint_pass(self, opt_with_ingredients):
        opt = opt_with_ingredients
        opt.add_quantity_constraint(["Water", "Flour"], min_val=50, max_val=150)
        assert opt._check_constraints({"Water": 60, "Flour": 30, "Sugar": 10}) is True

    def test_quantity_constraint_fail(self, opt_with_ingredients):
        opt = opt_with_ingredients
        opt.add_quantity_constraint(["Water", "Flour"], max_val=50)
        assert opt._check_constraints({"Water": 60, "Flour": 30, "Sugar": 10}) is False


# ------------------------------------------------------------------ #
#  Property limits: per 100 g of the finished formulation
# ------------------------------------------------------------------ #


class TestPropertyLimitsPerHundred:
    """A property limit reads per 100 g of what you make, not as a total that
    grows with the batch. Doubling every amount leaves the fat per 100 g where
    it was, so the same limit means the same thing at 100 g and at 10 kg."""

    def _fatty(self, opt):
        opt.load_ingredients_from_csv(pd.DataFrame({
            "Name": ["Lean", "Fatty"],
            "Min": [0, 0],
            "Max": [100, 100],
            "Unit": ["g", "g"],
            "Fat per 100 g": [10.0, 30.0],
        }))
        return opt

    def test_a_limit_reads_the_mass_weighted_average(self, opt):
        """50 g at 10 fat and 50 g at 30 fat is 20 fat per 100 g."""
        opt = self._fatty(opt)
        opt.add_constraint("Fat per 100 g", max_val=25)
        assert opt._check_constraints({"Lean": 50.0, "Fatty": 50.0}) is True
        opt.add_constraint("Fat per 100 g", max_val=15)
        assert opt._check_constraints({"Lean": 50.0, "Fatty": 50.0}) is False

    def test_a_limit_does_not_move_with_the_batch_size(self, opt):
        """The old reading was a total: ten times the batch broke the limit
        without a formulation changing."""
        opt = self._fatty(opt)
        opt.add_constraint("Fat per 100 g", max_val=25)
        for scale in (0.5, 1.0, 10.0):
            assert opt._check_constraints({"Lean": 50.0 * scale,
                                           "Fatty": 50.0 * scale}) is True

    def test_a_minimum_reads_per_100_g_too(self, opt):
        opt = self._fatty(opt)
        opt.add_constraint("Fat per 100 g", min_val=25)
        assert opt._check_constraints({"Lean": 50.0, "Fatty": 50.0}) is False
        assert opt._check_constraints({"Lean": 10.0, "Fatty": 90.0}) is True

    def test_property_per_100_reports_the_same_number(self, opt):
        opt = self._fatty(opt)
        assert opt.property_per_100({"Lean": 50.0, "Fatty": 50.0},
                                    "Fat per 100 g") == pytest.approx(20.0)
        assert opt.property_per_100({"Lean": 0.0, "Fatty": 0.0},
                                    "Fat per 100 g") is None

    def test_ask_only_offers_formulations_inside_the_limit(self, opt):
        """The optimizer is handed the same limit in its linear form, so what
        it suggests is what the screen would accept."""
        opt = self._fatty(opt)
        opt.add_objective("Taste", weight=1.0, goal="max", min_val=0, max_val=10)
        opt.add_constraint("Fat per 100 g", max_val=15)
        batch = opt.ask(n_suggestions=3)
        assert len(batch) == 3
        for recipe in batch:
            assert opt.property_per_100(recipe, "Fat per 100 g") <= 15 + 1e-9

    def test_ask_respects_the_limit_once_the_model_is_fitted(self, opt):
        """The warm path hands BoTorch the constraint instead of filtering."""
        opt = self._fatty(opt)
        opt.add_objective("Taste", weight=1.0, goal="max", min_val=0, max_val=10)
        opt.add_constraint("Fat per 100 g", max_val=15)
        for i in range(6):
            opt.tell({"Lean": 80.0 - i, "Fatty": 10.0 + i}, {"Taste": float(i)})
        for recipe in opt.ask(n_suggestions=1):
            assert opt.property_per_100(recipe, "Fat per 100 g") <= 15 + 1e-6

    def test_a_property_limit_needs_one_unit(self, opt):
        opt = self._fatty(opt)
        opt.set_ingredient_unit("Fatty", "ml")
        with pytest.raises(ValueError, match="every ingredient needs a mass unit"):
            opt.add_constraint("Fat per 100 g", max_val=25)
        assert opt.constraints == []

    def test_a_limit_written_before_this_version_is_read_per_100_g(self, opt,
                                                                   tmp_path):
        """A 0.2.x file stored the limit without saying what it was a limit
        on. It opens, and it now means per 100 g — the file says nothing more,
        so the screen is what tells the user."""
        opt = self._fatty(opt)
        state = opt.export_json()
        state['constraints'] = [{'metric': "Fat per 100 g", 'min': None,
                                 'max': 25.0}]          # no 'basis' key
        clone = FoodOptimizer(project_name="old_project")
        clone.import_json(state)
        assert clone.constraints[0].get('basis') is None
        assert clone._check_constraints({"Lean": 50.0, "Fatty": 50.0}) is True
        assert clone._check_constraints({"Lean": 10.0, "Fatty": 90.0}) is False

    def test_a_new_limit_records_what_it_is_a_limit_on(self, opt):
        opt = self._fatty(opt)
        opt.add_constraint("Fat per 100 g", max_val=25)
        assert opt.constraints[0]['basis'] == 'per_100'

    def test_a_unit_set_on_one_ingredient_drops_a_property_limit(self, opt):
        """An average over the amounts needs one unit, exactly as a sum does,
        so the same three edits that prune an amount limit prune this one."""
        opt = self._fatty(opt)
        opt.add_constraint("Fat per 100 g", max_val=25)
        removed = opt.set_ingredient_unit("Fatty", "ml")
        assert opt.constraints == []
        assert [(r['metric'], r['reason']) for r in removed] == [
            ("Fat per 100 g", "unit")]

    def test_re_adding_an_ingredient_in_another_unit_drops_it(self, opt):
        opt = self._fatty(opt)
        opt.add_constraint("Fat per 100 g", max_val=25)
        removed = opt.add_ingredient("Fatty", 0, 100, unit="ml")
        assert opt.constraints == []
        assert [r['metric'] for r in removed] == ["Fat per 100 g"]

    def test_a_reloaded_file_in_two_units_drops_it(self, opt):
        opt = self._fatty(opt)
        opt.add_constraint("Fat per 100 g", max_val=25)
        removed = opt.load_ingredients_from_csv(pd.DataFrame({
            "Name": ["Lean", "Fatty"],
            "Min": [0, 0],
            "Max": [100, 100],
            "Unit": ["g", "ml"],
            "Fat per 100 g": [10.0, 30.0],
        }))
        assert opt.constraints == []
        assert [r['metric'] for r in removed] == ["Fat per 100 g"]

    def test_a_limit_that_still_means_something_is_left_alone(self, opt):
        """An edit that leaves every ingredient in one unit takes nothing."""
        opt = self._fatty(opt)
        opt.add_constraint("Fat per 100 g", max_val=25)
        assert opt.set_ingredient_unit("Fatty", "g") == []
        assert opt.add_ingredient("Fatty", 0, 90, unit="g") == []
        assert len(opt.constraints) == 1

    def test_a_settings_unit_is_set_without_touching_a_limit(self, opt):
        """A process setting is not part of any sum or any average, so its
        unit cannot break a limit."""
        opt = self._fatty(opt)
        opt.add_process_parameter("Cook temperature", 150, 200)
        opt.add_constraint("Fat per 100 g", max_val=25)
        opt.add_total_mass_constraint(max_val=150)
        assert opt.set_variable_unit("Cook temperature", "°C") == []
        assert opt._var_by_name("Cook temperature")['unit'] == "°C"
        assert len(opt.constraints) == 1
        assert len(opt.quantity_constraints) == 1

    def test_set_variable_unit_still_prunes_for_an_ingredient(self, opt):
        opt = self._fatty(opt)
        opt.add_constraint("Fat per 100 g", max_val=25)
        removed = opt.set_variable_unit("Fatty", "ml")
        assert [r['metric'] for r in removed] == ["Fat per 100 g"]

    def test_set_variable_unit_refuses_a_name_the_project_lacks(self, opt):
        opt = self._fatty(opt)
        with pytest.raises(ValueError, match="No ingredient or setting named"):
            opt.set_variable_unit("Nutmeg", "g")

    def test_the_stranded_limit_refusals_are_written_per_100_g(self, opt):
        """The number in the refusal is an average now, so it says so: a
        bare 5 read as five grams of fat in the whole batch."""
        opt = self._fatty(opt)                # Lean 10 fat, Fatty 30 fat
        opt.add_objective("Taste", weight=1.0, goal="max", min_val=0, max_val=10)
        opt.add_constraint("Fat per 100 g", min_val=25)
        with pytest.raises(ValueError) as caught:
            opt.deactivate_variable("Fatty")
        assert ("the remaining active ingredients can only reach 10 per 100 g "
                "at most. Loosen the limit first." in str(caught.value)), \
            str(caught.value)

        opt.remove_constraint(0)
        opt.add_constraint("Fat per 100 g", max_val=15)
        with pytest.raises(ValueError) as caught:
            opt.deactivate_variable("Lean")
        assert ("the remaining active ingredients cannot get below 30 per 100 "
                "g. Loosen the limit first." in str(caught.value)), \
            str(caught.value)

    def test_pausing_that_strands_a_limit_says_so_in_per_100_terms(self, opt):
        """Pausing the only ingredient that carries the fat leaves a minimum
        nothing can reach."""
        opt = self._fatty(opt)
        opt.add_objective("Taste", weight=1.0, goal="max", min_val=0, max_val=10)
        opt.add_constraint("Fat per 100 g", min_val=25)
        with pytest.raises(ValueError, match="impossible to meet"):
            opt.deactivate_variable("Fatty")
        assert opt.active_variables() == opt.variables


# ------------------------------------------------------------------ #
#  Ask / Tell Loop
# ------------------------------------------------------------------ #


class TestAskTell:
    def test_cold_start_returns_recipes(self, opt_configured):
        recipes = opt_configured.ask(n_suggestions=2)
        assert len(recipes) == 2
        for r in recipes:
            assert "Water" in r
            assert "Flour" in r
            assert "Sugar" in r

    def test_cold_start_respects_bounds(self, opt_configured):
        recipes = opt_configured.ask(n_suggestions=5)
        for r in recipes:
            assert 0 <= r["Water"] <= 100
            assert 0 <= r["Flour"] <= 50
            assert 0 <= r["Sugar"] <= 30

    def test_tell_stores_experiment(self, opt_configured):
        recipe = {"Water": 50, "Flour": 25, "Sugar": 10}
        opt_configured.tell(recipe, {"Taste": 7.0})
        assert len(opt_configured.X_history) == 1
        assert len(opt_configured.Y_history) == 1
        assert len(opt_configured.recipe_history) == 1
        assert len(opt_configured.results_history) == 1

    def test_tell_no_objectives_raises(self, opt_with_ingredients):
        with pytest.raises(ValueError, match="Add at least one measurement"):
            opt_with_ingredients.tell({"Water": 50, "Flour": 25, "Sugar": 10},
                                      {"Taste": 7.0})

    def test_tell_missing_result_raises(self, opt_configured):
        with pytest.raises(ValueError, match="Enter a value for"):
            opt_configured.tell({"Water": 50, "Flour": 25, "Sugar": 10}, {})

    def test_warm_start_after_enough_experiments(self, opt_configured):
        """After n_init_random experiments, ask() should use GP optimization."""
        for i in range(6):
            recipe = {"Water": 50 + i, "Flour": 25, "Sugar": 10}
            opt_configured.tell(recipe, {"Taste": float(i)})
        # This should trigger _ask_optimize (GP-based)
        recipes = opt_configured.ask(n_suggestions=1)
        assert len(recipes) == 1
        assert "Water" in recipes[0]


# ------------------------------------------------------------------ #
#  History Editing
# ------------------------------------------------------------------ #


class TestHistoryEditing:
    def _add_experiments(self, opt, n=3):
        for i in range(n):
            recipe = {"Water": 50 + i, "Flour": 25, "Sugar": 10}
            opt.tell(recipe, {"Taste": float(i + 1)})

    def test_edit_result(self, opt_configured):
        self._add_experiments(opt_configured)
        opt_configured.edit_result(0, {"Taste": 9.0})
        assert opt_configured.results_history[0]["Taste"] == 9.0

    def test_edit_result_out_of_range(self, opt_configured):
        self._add_experiments(opt_configured)
        with pytest.raises(IndexError):
            opt_configured.edit_result(99, {"Taste": 1.0})

    def test_delete_result(self, opt_configured):
        self._add_experiments(opt_configured, n=3)
        opt_configured.delete_result(1)
        assert len(opt_configured.X_history) == 2

    def test_delete_result_out_of_range(self, opt_configured):
        self._add_experiments(opt_configured)
        with pytest.raises(IndexError):
            opt_configured.delete_result(99)

    def test_rewind_to(self, opt_configured):
        self._add_experiments(opt_configured, n=5)
        opt_configured.rewind_to(2)
        assert len(opt_configured.X_history) == 3
        assert len(opt_configured.Y_history) == 3


# ------------------------------------------------------------------ #
#  Persistence: Save / Load / Export / Import
# ------------------------------------------------------------------ #


class TestPersistence:
    def test_save_and_reload(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer(project_name="persist_test")
        opt.add_ingredient("Water", 0, 100)
        opt.add_objective("Taste", weight=1.0, goal="max")
        opt.save()

        opt2 = FoodOptimizer(project_name="persist_test")
        assert len(opt2.variables) == 1
        assert opt2.variables[0]["name"] == "Water"

    def test_save_writes_json_not_pickle(self, tmp_path, monkeypatch):
        """Project files must be JSON (safe to open) — never executable pickle."""
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer(project_name="fmt_test")
        opt.add_ingredient("Water", 0, 100)
        opt.save()
        raw = (tmp_path / "fmt_test.pkl").read_bytes()
        state = json.loads(raw.decode("utf-8"))   # must parse as JSON
        assert state["project_name"] == "fmt_test"
        assert state["variables"][0]["name"] == "Water"

    def test_load_refuses_legacy_pickle(self, tmp_path, monkeypatch):
        """Pickle project files are no longer executed; the user gets a hint."""
        monkeypatch.chdir(tmp_path)
        legacy = {"project_name": "old", "variables": [], "objectives": []}
        with open("old.pkl", "wb") as f:
            pickle.dump(legacy, f)
        opt = FoodOptimizer("old")
        assert opt.load_error and "early version" in opt.load_error
        assert not (tmp_path / "old.pkl.tmp").exists()

    def test_load_corrupt_file_reports_error_not_silent_success(self, tmp_path, monkeypatch):
        """A damaged file must surface an error, never load as a blank project
        while claiming success."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "broken.pkl").write_bytes(b"\x80\x04 not json not pickle \xff\xfe")
        opt = FoodOptimizer(project_name="broken")
        assert opt.load_error is not None
        assert isinstance(opt.load_error, str) and opt.load_error

    def test_restore_backup_clears_load_error(self, tmp_path, monkeypatch):
        """Restoring a backup into a damaged project clears load_error and
        rewrites the file as valid JSON (the app's recovery path)."""
        monkeypatch.chdir(tmp_path)
        donor = FoodOptimizer(project_name="donor")
        donor.add_ingredient("Water", 0, 100)
        backup = donor.export_json()

        (tmp_path / "broken.pkl").write_bytes(b"\x80\x04 not json not pickle \xff\xfe")
        opt = FoodOptimizer(project_name="broken")
        assert opt.load_error is not None

        backup["project_name"] = "broken"  # what the app's restore flow does
        opt.import_json(backup)
        # import_json no longer autosaves (spec 2026-09-02): callers persist
        # explicitly, as the app's restore handler now does.
        opt.save()
        assert opt.load_error is None
        assert opt.variables[0]["name"] == "Water"
        json.loads((tmp_path / "broken.pkl").read_bytes().decode("utf-8"))

    def test_failed_save_preserves_existing_file(self, tmp_path, monkeypatch):
        """A crash mid-save must never corrupt the previously saved project."""
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer(project_name="atomic_test")
        opt.add_ingredient("Water", 0, 100)
        opt.save()
        good_bytes = (tmp_path / "atomic_test.pkl").read_bytes()

        def boom(*args, **kwargs):
            raise RuntimeError("simulated crash mid-write")

        with monkeypatch.context() as m:
            m.setattr(os, "replace", boom)   # crash at the atomic swap
            with pytest.raises(RuntimeError):
                opt.save()

        assert (tmp_path / "atomic_test.pkl").read_bytes() == good_bytes
        opt2 = FoodOptimizer(project_name="atomic_test")
        assert opt2.variables[0]["name"] == "Water"

    def test_load_ingredients_nonnumeric_bounds_plain_error(self, tmp_path, monkeypatch):
        """A text value in Min/Max must give a plain-language error, not a raw
        'could not convert string to float' traceback."""
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer(project_name="numtest")
        df = pd.DataFrame({"name": ["Water"], "min": ["abc"], "max": [100]})
        with pytest.raises(ValueError, match="must be numbers"):
            opt.load_ingredients_from_csv(df)

    def test_export_import_json(self, opt_configured):
        recipe = {"Water": 50, "Flour": 25, "Sugar": 10}
        opt_configured.tell(recipe, {"Taste": 7.0})

        exported = opt_configured.export_json()
        assert isinstance(exported, dict)
        assert exported["project_name"] == "test_project"
        assert len(exported["X_history"]) == 1

        # Import into a fresh optimizer
        opt_configured.import_json(exported)
        assert len(opt_configured.X_history) == 1
        assert len(opt_configured.variables) == 3

    def test_export_json_serializable(self, opt_configured):
        recipe = {"Water": 50, "Flour": 25, "Sugar": 10}
        opt_configured.tell(recipe, {"Taste": 7.0})
        exported = opt_configured.export_json()
        # Should be fully JSON serializable
        json_str = json.dumps(exported)
        assert isinstance(json_str, str)

    def test_import_converts_bounds_lists_to_tuples(self, opt_configured):
        exported = opt_configured.export_json()
        # JSON converts tuples to lists
        assert isinstance(exported["variables"][0]["bounds"], list)
        opt_configured.import_json(exported)
        # After import, bounds should be tuples again
        assert isinstance(opt_configured.variables[0]["bounds"], tuple)


# ------------------------------------------------------------------ #
#  Full journey
# ------------------------------------------------------------------ #


class TestFullJourney:
    def test_full_process_with_midrun_process_parameter(self, tmp_path, monkeypatch):
        """The full user journey: CSV ingredients -> objective -> recorded
        experiments -> process parameter added mid-run (needs a baseline) ->
        new suggestions include it -> everything survives a reload."""
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("journey")
        opt.load_ingredients_from_csv(pd.DataFrame({
            "Name": ["Water", "Flour", "Sugar"],
            "Min": [10, 5, 0], "Max": [80, 50, 30],
            "Cost": [0.0, 0.5, 0.8],
        }))
        opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
        for amounts, taste in [((40, 30, 10), 6.0), ((50, 20, 15), 7.0),
                               ((60, 10, 5), 5.5)]:
            recipe = dict(zip(["Water", "Flour", "Sugar"], amounts))
            opt.tell(recipe, {"Taste": taste})

        # Mid-run with no baseline must be a clear ValueError (the app shows
        # its text), never a crash — and must not half-add the variable.
        with pytest.raises(ValueError, match="baseline"):
            opt.add_process_parameter("Oven_Temp", 150, 220)
        assert all(v["name"] != "Oven_Temp" for v in opt.variables)

        # A baseline outside [min, max] is also a clear error.
        with pytest.raises(ValueError, match="must be between"):
            opt.add_process_parameter("Oven_Temp", 150, 220, baseline=100)

        opt.add_process_parameter("Oven_Temp", 150, 220, baseline=180)
        # History re-encoded: every past experiment ran at the baseline.
        assert all(len(x) == 4 for x in opt.X_history)
        assert all(x[3] == 180.0 for x in opt.X_history)

        batch = opt.ask(n_suggestions=1)
        assert 150 <= batch[0]["Oven_Temp"] <= 220
        opt.tell(batch[0], {"Taste": 8.0})

        reloaded = FoodOptimizer("journey")
        assert reloaded.load_error is None
        assert len(reloaded.X_history) == 4
        assert all(len(x) == 4 for x in reloaded.X_history)
        assert reloaded.X_history[0][3] == 180.0


# ------------------------------------------------------------------ #
#  Non-monotone active set (adaptive EGBO pruning)
# ------------------------------------------------------------------ #


class TestActiveSet:
    def test_no_pruning_is_a_noop(self, opt_configured):
        """The monotone path must be untouched when nothing is pruned."""
        assert opt_configured._get_fixed_features() == {}
        assert len(opt_configured.active_variables()) == 3
        assert opt_configured.inactive_variables() == []

    def test_deactivate_pins_the_column(self, opt_configured):
        opt_configured.deactivate_variable("Flour")
        assert opt_configured._get_fixed_features() == {1: 0.0}
        assert [v["name"] for v in opt_configured.inactive_variables()] == ["Flour"]

    def test_deactivate_with_explicit_value_normalizes(self, opt_configured):
        opt_configured.deactivate_variable("Flour", value=12.5)  # bounds (0, 50)
        assert opt_configured._get_fixed_features() == {1: 0.25}

    def test_deactivate_rejects_out_of_bounds_pin(self, opt_configured):
        with pytest.raises(ValueError, match="must lie within"):
            opt_configured.deactivate_variable("Sugar", value=99.0)

    def test_deactivate_is_idempotent(self, opt_configured):
        opt_configured.deactivate_variable("Flour")
        opt_configured.deactivate_variable("Flour")
        assert len(opt_configured.inactive_variables()) == 1

    def test_deactivate_blocks_last_active_variable(self, opt_configured):
        opt_configured.deactivate_variable("Flour")
        opt_configured.deactivate_variable("Sugar")
        with pytest.raises(ValueError, match="must stay active"):
            opt_configured.deactivate_variable("Water")

    def test_reactivate_restores_the_dimension(self, opt_configured):
        opt_configured.deactivate_variable("Flour", value=10.0)
        opt_configured.reactivate_variable("Flour")
        assert opt_configured._get_fixed_features() == {}
        assert "_frozen_at" not in opt_configured._var_by_name("Flour")

    def test_pruning_preserves_history_and_width(self, opt_configured):
        """Pruning restricts the search domain, it must not touch observations."""
        opt_configured.tell(
            {"Water": 50.0, "Flour": 20.0, "Sugar": 5.0}, {"Taste": 7.0}
        )
        before_x = [row[:] for row in opt_configured.X_history]
        before_y = list(opt_configured.Y_history)

        opt_configured.deactivate_variable("Flour")

        assert opt_configured.X_history == before_x
        assert opt_configured.Y_history == before_y
        assert len(opt_configured.X_history[0]) == 3

    def test_ask_respects_pins_in_cold_start(self, opt_configured):
        opt_configured.deactivate_variable("Flour")
        batch = opt_configured.ask(n_suggestions=3, n_init_random=5)
        assert all(r["Flour"] == 0.0 for r in batch)
        assert any(r["Water"] > 0.0 for r in batch)

    def test_ask_respects_pins_after_gp_fit(self, opt_configured):
        opt_configured.deactivate_variable("Flour")
        for _ in range(6):
            rec = opt_configured.ask(n_suggestions=1, n_init_random=5)[0]
            opt_configured.tell(rec, {"Taste": 5.0 + rec["Water"] / 100.0})
        batch = opt_configured.ask(n_suggestions=2, n_init_random=5)
        assert all(r["Flour"] == 0.0 for r in batch)

    def test_ask_guards_empty_active_set(self, opt_configured):
        opt_configured.deactivate_variable("Flour")
        opt_configured.deactivate_variable("Sugar")
        opt_configured._var_by_name("Water")["active"] = False
        with pytest.raises(ValueError, match="Everything is paused"):
            opt_configured.ask(n_suggestions=1)

    def test_deactivate_detects_stranded_quantity_constraint(self, opt_configured):
        opt_configured.add_quantity_constraint(["Water", "Flour"], min_val=120.0)
        with pytest.raises(ValueError, match="impossible to meet"):
            opt_configured.deactivate_variable("Flour")
        opt_configured.deactivate_variable("Sugar")  # unrelated, still fine
        assert [v["name"] for v in opt_configured.inactive_variables()] == ["Sugar"]

    def test_deactivate_detects_stranded_property_constraint(self, opt, monkeypatch):
        opt.load_ingredients_from_csv(pd.DataFrame([
            {"Name": "a", "Min": 0, "Max": 10, "Protein": 1.0},
            {"Name": "b", "Min": 0, "Max": 10, "Protein": 1.0},
        ]))
        opt.add_objective("Taste", weight=1.0, goal="max", min_val=0, max_val=10)
        opt.add_constraint("protein", min_val=15.0)
        with pytest.raises(ValueError, match="impossible to meet"):
            opt.deactivate_variable("a")

    def test_pruned_state_survives_pkl_reload(self, opt_configured, monkeypatch):
        opt_configured.deactivate_variable("Flour", value=2.5)
        reloaded = FoodOptimizer(project_name="test_project")
        assert [v["name"] for v in reloaded.inactive_variables()] == ["Flour"]
        assert reloaded._frozen_value(reloaded._var_by_name("Flour")) == 2.5

    def test_pruned_state_survives_json_roundtrip(self, opt_configured):
        opt_configured.deactivate_variable("Flour", value=12.5)
        state = opt_configured.export_json()
        clone = FoodOptimizer(project_name="clone_project")
        clone.import_json(state)
        assert [v["name"] for v in clone.inactive_variables()] == ["Flour"]
        assert clone._get_fixed_features() == {1: 0.25}

    def test_legacy_variables_default_to_active(self, opt_configured):
        """Projects saved before pruning existed must load as fully active."""
        for var in opt_configured.variables:
            var.pop("active", None)
        opt_configured.save()
        reloaded = FoodOptimizer(project_name="test_project")
        assert len(reloaded.active_variables()) == 3
        assert reloaded._get_fixed_features() == {}

    def test_export_trajectory_reports_pruned_and_past_usage(self, opt_configured):
        opt_configured.tell(
            {"Water": 50.0, "Flour": 20.0, "Sugar": 5.0}, {"Taste": 7.0}
        )
        opt_configured.deactivate_variable("Flour")
        text = opt_configured.export_trajectory()
        assert "Paused" in text
        assert "Flour=20" in text, "a pruned variable's past usage must stay visible"


class TestRemoveIngredient:
    def test_refuses_ingredient_used_at_nonzero(self, opt_configured):
        opt_configured.tell(
            {"Water": 50.0, "Flour": 20.0, "Sugar": 0.0}, {"Taste": 7.0}
        )
        with pytest.raises(ValueError, match="cannot be deleted"):
            opt_configured.remove_ingredient("Flour")

    def test_allows_never_used_ingredient(self, opt_configured):
        opt_configured.tell(
            {"Water": 50.0, "Flour": 20.0, "Sugar": 0.0}, {"Taste": 7.0}
        )
        opt_configured.remove_ingredient("Sugar")
        assert [v["name"] for v in opt_configured.variables] == ["Water", "Flour"]
        assert len(opt_configured.X_history[0]) == 2

    def test_force_deletes_and_drops_the_column(self, opt_configured):
        opt_configured.tell(
            {"Water": 50.0, "Flour": 20.0, "Sugar": 0.0}, {"Taste": 7.0}
        )
        opt_configured.remove_ingredient("Flour", force=True)
        assert [v["name"] for v in opt_configured.variables] == ["Water", "Sugar"]
        assert len(opt_configured.X_history[0]) == 2
        assert "Flour" not in opt_configured.recipe_history[0]

    def test_prunes_quantity_constraints(self, opt_configured):
        opt_configured.add_quantity_constraint(["Water", "Flour"], max_val=120.0)
        opt_configured.add_quantity_constraint(["Flour"], max_val=40.0)
        opt_configured.remove_ingredient("Flour")
        assert [
            qc["ingredients"] for qc in opt_configured.quantity_constraints
        ] == [["Water"]]

    def test_rejects_process_parameter(self, opt_configured):
        opt_configured.add_process_parameter("Temp", 100, 200)
        with pytest.raises(ValueError, match="process setting"):
            opt_configured.remove_ingredient("Temp")

    def test_blocks_removing_last_active_variable(self, opt_configured):
        opt_configured.deactivate_variable("Flour")
        opt_configured.deactivate_variable("Sugar")
        with pytest.raises(ValueError, match="last active variable"):
            opt_configured.remove_ingredient("Water")


class TestValidateState:
    def test_empty_dict_is_rejected(self):
        with pytest.raises(ValueError, match="not a Food Optimizer backup"):
            FoodOptimizer.validate_state({})

    def test_non_dict_is_rejected(self):
        with pytest.raises(ValueError, match="not a Food Optimizer backup"):
            FoodOptimizer.validate_state([1, 2, 3])

    def test_wrong_shape_is_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        state = FoodOptimizer("tmp_validate").export_json()
        state["variables"] = "garbage"
        with pytest.raises(ValueError, match="wrong shape"):
            FoodOptimizer.validate_state(state)

    def test_wrong_element_shape_is_rejected(self):
        with pytest.raises(ValueError, match="wrong shape"):
            FoodOptimizer.validate_state({
                "variables": ["Water"], "objectives": [],
                "recipe_history": [], "results_history": [],
                "CLASS_VERSION": 6,
            })

    def test_newer_version_is_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        state = FoodOptimizer("tmp_validate").export_json()
        state["CLASS_VERSION"] = FoodOptimizer.CLASS_VERSION + 1
        with pytest.raises(ValueError, match="newer version"):
            FoodOptimizer.validate_state(state)

    def test_malformed_formulation_ids_are_rejected(self, tmp_path, monkeypatch):
        """import_json assigns attributes one by one, so a bad identity list
        must be caught here — not halfway through the restore."""
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("tmp_ids")
        opt.add_ingredient("Water", 0, 100)
        opt.add_objective("Taste", 1.0, goal="max")
        opt.tell({"Water": 50.0}, {"Taste": 7.0})
        state = opt.export_json()
        state["formulation_ids"] = ["one"]
        with pytest.raises(ValueError, match="'formulation_ids' section has the wrong shape"):
            FoodOptimizer.validate_state(state)
        # The project the restore would have replaced is untouched: the refusal
        # comes before import_json assigns anything.
        target = FoodOptimizer("tmp_ids")
        assert target.formulation_ids == [1] and len(target.X_history) == 1

    def test_the_other_identity_lists_are_type_checked(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        base = FoodOptimizer("tmp_identity").export_json()
        for key, bad in (("formulation_ids", [1.5]),
                         ("batch_history", ["1"]),
                         ("notes_history", [3]),
                         ("skipped", ["not a dict"]),
                         ("next_formulation_no", "seven")):
            state = dict(base)
            state[key] = bad
            with pytest.raises(ValueError, match=f"'{key}' section has the wrong shape"):
                FoodOptimizer.validate_state(state)
        # Absent is fine: a 0.2.x file has none of these.
        for key in ("formulation_ids", "batch_history", "notes_history",
                    "skipped", "next_formulation_no"):
            state = dict(base)
            state.pop(key, None)
            FoodOptimizer.validate_state(state)
        # A well-formed batch_history may carry None for an imported row.
        state = dict(base)
        state["batch_history"] = [None, 3]
        FoodOptimizer.validate_state(state)

    def test_a_present_but_null_identity_list_is_refused(self, tmp_path,
                                                          monkeypatch):
        """import_json iterates these lists. Skipping the check for a null one
        let it raise a TypeError halfway through, after project_name,
        variables and objectives had already been overwritten."""
        monkeypatch.chdir(tmp_path)
        base = FoodOptimizer("tmp_null").export_json()
        for key in ("formulation_ids", "batch_history", "notes_history",
                    "skipped"):
            state = dict(base)
            state[key] = None
            with pytest.raises(ValueError,
                               match=f"'{key}' section has the wrong shape"):
                FoodOptimizer.validate_state(state)

    def test_a_left_out_formulation_needs_all_four_of_its_fields(
            self, tmp_path, monkeypatch):
        """history_frame reads formulation, batch, recipe and note on every
        render. A backup missing one loaded, saved, and then broke the All
        formulations table for good."""
        monkeypatch.chdir(tmp_path)
        base = FoodOptimizer("tmp_skipped").export_json()
        good = {"formulation": 1, "batch": 1, "recipe": {"Water": 5.0},
                "note": "Not made"}
        base["next_formulation_no"] = 2
        for bad in ({k: v for k, v in good.items() if k != "formulation"},
                    dict(good, formulation=0),
                    dict(good, formulation=-1),
                    dict(good, formulation="one"),
                    dict(good, batch="1"),
                    {k: v for k, v in good.items() if k != "recipe"},
                    dict(good, recipe="Water 5 g"),
                    dict(good, note=3)):
            state = dict(base)
            state["skipped"] = [bad]
            with pytest.raises(ValueError,
                               match="'skipped' section has the wrong shape"):
                FoodOptimizer.validate_state(state)
        FoodOptimizer.validate_state(dict(base, skipped=[good]))
        # A note is the one field that may be absent: record_skipped always
        # writes one, but a hand-edited file without it still renders.
        FoodOptimizer.validate_state(dict(
            base, skipped=[{k: v for k, v in good.items() if k != "note"}]))

    def test_a_backup_that_repeats_a_formulation_number_is_refused(
            self, tmp_path, monkeypatch):
        """A number is permanent: index_of_formulation finds only the first of
        two rows numbered 7, so deleting one leaves the others behind."""
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("tmp_dupes")
        opt.add_ingredient("Water", 0, 100)
        opt.add_objective("Taste", 1.0, goal="max")
        opt.tell({"Water": 50.0}, {"Taste": 7.0}, formulation_no=7)
        opt.tell({"Water": 60.0}, {"Taste": 6.0}, formulation_no=8)
        state = opt.export_json()
        FoodOptimizer.validate_state(state)              # as exported, fine
        with pytest.raises(ValueError, match="same number"):
            FoodOptimizer.validate_state(dict(state, formulation_ids=[7, 7]))
        # A left-out formulation and a scored one cannot share a number.
        clash = dict(state)
        clash["skipped"] = [{"formulation": 7, "batch": 1, "recipe": {},
                             "note": "Not made"}]
        with pytest.raises(ValueError, match="same number"):
            FoodOptimizer.validate_state(clash)
        # Neither can one batch, twice over.
        twice = dict(state)
        twice["pending_batch"] = [{"formulation": 9, "recipe": {"Water": 1.0}},
                                  {"formulation": 9, "recipe": {"Water": 2.0}}]
        twice["next_formulation_no"] = 10
        with pytest.raises(ValueError, match="same number"):
            FoodOptimizer.validate_state(twice)

    def test_a_batch_recorded_one_sheet_at_a_time_still_restores(
            self, tmp_path, monkeypatch):
        """A batch stays open while part of it is recorded, so a pending row
        legitimately carries a number the history already holds. Refusing
        that would refuse a backup of every half-recorded batch."""
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("tmp_partly")
        opt.add_ingredient("Water", 0, 100)
        opt.add_objective("Taste", 1.0, goal="max")
        opt.set_pending_batch([{"Water": 10.0}, {"Water": 20.0}])
        first = opt.pending_batch[0]
        opt.tell(first["recipe"], {"Taste": 7.0},
                 formulation_no=first["formulation"], batch_no=1)
        assert opt.pending_batch is not None            # the batch stays open
        FoodOptimizer.validate_state(opt.export_json())

    def test_a_formulation_number_the_counter_never_issued_is_refused(
            self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("tmp_counter")
        opt.add_ingredient("Water", 0, 100)
        opt.add_objective("Taste", 1.0, goal="max")
        opt.tell({"Water": 50.0}, {"Taste": 7.0})
        state = opt.export_json()
        with pytest.raises(ValueError, match="never issued"):
            FoodOptimizer.validate_state(dict(state, next_formulation_no=1))
        with pytest.raises(ValueError,
                           match="'formulation_ids' section has the wrong shape"):
            FoodOptimizer.validate_state(dict(state, formulation_ids=[0]))
        with pytest.raises(ValueError,
                           match="'formulation_ids' section has the wrong shape"):
            FoodOptimizer.validate_state(dict(state, formulation_ids=[-1]))

    def test_summary_of_valid_backup(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("src")
        opt.add_ingredient("Water", 0, 100)
        opt.add_objective("Taste", 1.0, goal="max")
        opt.tell({"Water": 50.0}, {"Taste": 7.0})
        opt.record_skipped(2, 1, {"Water": 60.0})
        opt.add_process_parameter("Cook temperature", 150, 200, baseline=170,
                                  unit="°C")
        summary = FoodOptimizer.validate_state(opt.export_json())
        # 'formulations' counts the left-out row too — a warning that offers
        # to replace a project must not undercount what it holds — and the
        # settings are named, so a settings-only project is not '0
        # ingredients' and nothing else.
        assert summary == {"name": "src", "experiments": 1, "formulations": 2,
                           "ingredients": 1, "settings": 1,
                           "version": FoodOptimizer.CLASS_VERSION}


# ------------------------------------------------------------------ #
#  Setup-form validation
# ------------------------------------------------------------------ #


class TestSetupValidation:
    def _opt(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        return FoodOptimizer("setup_val")

    def test_blank_process_parameter_name(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        with pytest.raises(ValueError, match="cannot be empty"):
            opt.add_process_parameter("   ", 0, 10)

    def test_inverted_bounds(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        with pytest.raises(ValueError, match="Lowest must be less than Highest"):
            opt.add_process_parameter("Temp", 200, 100)
        with pytest.raises(ValueError, match="Lowest must be less than Highest"):
            opt.add_ingredient("Water", 5, 5)

    def test_cross_category_name_collision(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_ingredient("Water", 0, 100)
        with pytest.raises(ValueError, match="already exists as an ingredient"):
            opt.add_process_parameter("water", 0, 10)

    def test_ingredient_name_collides_with_measurement(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
        with pytest.raises(ValueError, match="already the name of a measurement"):
            opt.add_ingredient("Taste", 0, 10)

    def test_csv_rejects_duplicate_and_blank_names(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        df = pd.DataFrame({"Name": ["Sugar", " Sugar", None], "Min": [0, 0, 0], "Max": [10, 10, 10]})
        with pytest.raises(ValueError, match="Row 3.*duplicate"):
            opt.load_ingredients_from_csv(df)

    def test_constraint_inverted_bounds(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_ingredient("Sugar", 0, 100)
        opt.add_ingredient("Honey", 0, 100)
        with pytest.raises(ValueError, match="Lowest must be less than Highest"):
            opt.add_quantity_constraint(["Sugar", "Honey"], min_val=50, max_val=10)

    def test_duplicate_quantity_constraint_replaces(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_ingredient("Sugar", 0, 100)
        opt.add_ingredient("Honey", 0, 100)
        opt.add_quantity_constraint(["Sugar", "Honey"], max_val=50)
        opt.add_quantity_constraint(["Honey", "Sugar"], max_val=40)
        assert len(opt.quantity_constraints) == 1
        assert opt.quantity_constraints[0]['max'] == 40.0


def test_sample_ingredients_csv_has_readable_names(tmp_path, monkeypatch):
    """The shipped sample CSV (the sample project and the CSV template
    download) must use plain, human-readable ingredient names, since they
    appear verbatim in every table, recipe card, and batch sheet."""
    monkeypatch.chdir(tmp_path)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(repo_root, "data", "ingredients.csv")
    df = pd.read_csv(csv_path)

    opt = FoodOptimizer(project_name="sample_check")
    opt.load_ingredients_from_csv(df)

    names = [v["name"] for v in opt.variables]
    assert len(names) == 20
    for name in names:
        assert "_" not in name, name

    # The shipped sample (sample project + CSV template) is a subset of the
    # same list, short enough to read at a glance.
    sample = pd.read_csv(os.path.join(repo_root, "data", "sample_ingredients.csv"))
    sample_opt = FoodOptimizer(project_name="sample_small")
    sample_opt.load_ingredients_from_csv(sample)
    sample_names = [v["name"] for v in sample_opt.variables]
    assert len(sample_names) == 8
    assert set(sample_names) <= set(names)
    assert list(sample.columns) == list(df.columns)

    # The repo's experiments example (used by experiments/, not shipped in
    # the disk image) must still import for the ingredient columns, the same
    # check app.py runs before accepting an import. Its objectives need not
    # exist in this bare project, so only the ingredient columns are asserted.
    example_path = os.path.join(repo_root, "data", "experiments_example.csv")
    import_df = pd.read_csv(example_path)
    for name in names:
        assert name in import_df.columns, name


def test_no_code_in_user_facing_errors(tmp_path, monkeypatch):
    """Backend errors reach the UI verbatim, so they must read as plain language."""
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("copy")
    opt.add_ingredient("Water", 0, 100)
    opt.add_objective("Taste", 1.0)
    opt.tell({"Water": 50.0}, {"Taste": 7.0})
    with pytest.raises(ValueError) as e:
        opt.remove_ingredient("Water")
    msg = str(e.value)
    for banned in ("force=True", "BO ", "GP", "pinned", "encode", "dimension"):
        assert banned not in msg, msg


def test_recipe_lines_ignores_nan_and_non_numeric(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("rl_nan")
    assert opt.recipe_lines({"Water": 5.0, "Salt": float("nan"), "Sugar": "2", "Oil": None}) == [("Water", 5.0), ("Sugar", 2.0)]


class TestARegeneratedBatchIsADifferentBatch:
    """`Generate a different batch` used to hand back the batch it had just
    discarded, byte for byte. Both regimes seeded on len(X_history), which a
    discard before any result leaves exactly where it was. They now seed on
    next_formulation_no, which advances on every generate."""

    def _opt(self, tmp_path, monkeypatch, name="reseed"):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer(name)
        opt.add_ingredient("Water", 0, 100)
        opt.add_ingredient("Pea protein", 0, 100)
        opt.add_objective("Firmness", 1.0, goal="target", target=6,
                          min_val=0, max_val=10)
        return opt

    @staticmethod
    def _amounts(opt):
        return [{k: round(float(v), 6) for k, v in r['recipe'].items()}
                for r in opt.pending_batch]

    def test_cold_start_regenerates_different_formulations(self, tmp_path,
                                                           monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.ask(n_suggestions=3)
        first = self._amounts(opt)
        opt.set_pending_batch(None)          # "Yes, discard"
        opt.ask(n_suggestions=3)
        assert self._amounts(opt) != first

    def test_the_warm_regime_regenerates_different_formulations(self, tmp_path,
                                                                monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        for k in range(5):
            opt.tell({"Water": 10.0 + k, "Pea protein": 20.0 - k},
                     {"Firmness": 4.0 + 0.3 * k}, formulation_no=k + 1,
                     batch_no=1)
        opt.ask(n_suggestions=2)
        first = self._amounts(opt)
        opt.set_pending_batch(None)
        opt.ask(n_suggestions=2)
        assert self._amounts(opt) != first

    def test_the_seed_source_survives_a_reload(self, tmp_path, monkeypatch):
        """next_formulation_no is persisted, so a project closed and reopened
        generates what the session that closed it would have. Seeding on
        anything held only in memory would make Generate depend on how long
        the window had been open."""
        opt = self._opt(tmp_path, monkeypatch, name="reseed_reload")
        opt.ask(n_suggestions=3)
        first = self._amounts(opt)
        opt.set_pending_batch(None)
        after_discard = int(opt.next_formulation_no)

        reloaded = FoodOptimizer("reseed_reload")
        assert int(reloaded.next_formulation_no) == after_discard
        reloaded.ask(n_suggestions=3)
        second = self._amounts(reloaded)
        assert second != first          # a different batch, as the label says

        # ...and a third session in that same state repeats the second's work
        # exactly, rather than wandering.
        reloaded.set_pending_batch(None)
        reloaded.next_formulation_no = after_discard
        reloaded.save()
        again = FoodOptimizer("reseed_reload")
        again.ask(n_suggestions=3)
        assert self._amounts(again) == second

    def test_the_same_state_still_generates_the_same_formulations(
            self, tmp_path, monkeypatch):
        """Determinism is the point of seeding at all: two sessions opening
        the same project and pressing Generate see the same formulations."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.ask(n_suggestions=3)
        first = self._amounts(opt)
        opt.set_pending_batch(None)
        opt.next_formulation_no = 1          # wind back: the same state again
        opt.ask(n_suggestions=3)
        assert self._amounts(opt) == first


class TestFormulationIdentity:
    def _opt(self, tmp_path, monkeypatch, name="ident"):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer(name)
        opt.add_ingredient("Water", 0, 100)
        opt.add_objective("Firmness", 1.0, goal="target", target=6, min_val=0, max_val=10)
        return opt

    def test_ask_issues_global_numbers_and_opens_a_batch(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.ask(n_suggestions=3)
        assert [r["formulation"] for r in opt.pending_batch] == [1, 2, 3]
        assert opt.pending_batch_no == 1
        assert opt.next_formulation_no == 4
        assert len(opt.pending_batch_created) == 10      # an ISO date

    def test_ask_issues_numbers_exactly_once_per_generate(self, tmp_path, monkeypatch):
        """A caller must not re-number a batch ask() already opened: ask()
        itself calls set_pending_batch, so a second call on the same rows
        (as app.py used to do) would burn numbers twice per generate."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.ask(n_suggestions=3)
        assert [r["formulation"] for r in opt.pending_batch] == [1, 2, 3]
        assert opt.next_formulation_no == 4
        opt.set_pending_batch(None)
        opt.ask(n_suggestions=3)
        assert [r["formulation"] for r in opt.pending_batch] == [4, 5, 6]

    def test_discarded_numbers_are_never_reissued(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.ask(n_suggestions=3)
        opt.set_pending_batch(None)          # "Generate a different batch"
        opt.ask(n_suggestions=2)
        assert [r["formulation"] for r in opt.pending_batch] == [4, 5]
        assert opt.pending_batch_no == 1     # the batch keeps its number

    def test_set_pending_batch_remembers_what_was_discarded(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.ask(n_suggestions=2)
        opt.set_pending_batch(None)
        opt.ask(n_suggestions=2)
        opt.set_pending_batch(opt.pending_batch, batch_no=1, discarded=[1, 2])
        assert opt.pending_batch_discarded == [1, 2]
        assert FoodOptimizer(opt.project_name).pending_batch_discarded == [1, 2]

    def test_add_to_pending_batch_draws_the_next_number(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.ask(n_suggestions=2)
        issued = opt.add_to_pending_batch({"Water": 42.0})
        assert issued == 3
        assert opt.pending_batch[-1] == {"formulation": 3, "recipe": {"Water": 42.0}}

    def test_add_to_pending_batch_opens_a_batch_when_none_is_open(
            self, tmp_path, monkeypatch):
        """A formulation of your own can be the first one in a batch: nothing
        is open, so it opens one, numbers it, and dates it for the sheets."""
        opt = self._opt(tmp_path, monkeypatch)
        issued = opt.add_to_pending_batch({"Water": 42.0}, note="Own formulation")
        assert issued == 1
        assert opt.pending_batch_no == 1
        assert len(opt.pending_batch_created) == 10      # an ISO date
        reloaded = FoodOptimizer(opt.project_name)
        assert reloaded.pending_batch == [{"formulation": 1,
                                           "recipe": {"Water": 42.0},
                                           "note": "Own formulation"}]
        assert reloaded.pending_batch_no == 1

    def test_tell_records_formulation_batch_and_note(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.ask(n_suggestions=2)
        row = opt.pending_batch[0]
        opt.tell(row["recipe"], {"Firmness": 6.0},
                 formulation_no=row["formulation"], batch_no=opt.pending_batch_no,
                 note="crumbly edges")
        assert opt.formulation_ids == [1]
        assert opt.batch_history == [1]
        assert opt.notes_history == ["crumbly edges"]

    def test_an_explicit_number_still_advances_the_counter(self, tmp_path, monkeypatch):
        """A number handed to tell() or record_skipped() must retire with it, or
        the next ask() would hand the same number to a different formulation."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.tell({"Water": 10.0}, {"Firmness": 6.0}, formulation_no=9)
        assert opt.next_formulation_no == 10
        opt.record_skipped(12, None, {"Water": 20.0})
        assert opt.next_formulation_no == 13
        opt.ask(n_suggestions=2)
        assert [r["formulation"] for r in opt.pending_batch] == [13, 14]

    def test_tell_stores_partial_results_and_refuses_an_all_blank_row(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_objective("Juiciness", 1.0, goal="target", target=7, min_val=0, max_val=10)
        opt.tell({"Water": 10.0}, {"Firmness": 6.0, "Juiciness": None})
        assert opt.results_history[0] == {"Firmness": 6.0}
        with pytest.raises(ValueError, match="Enter a value for"):
            opt.tell({"Water": 20.0}, {"Firmness": None, "Juiciness": None})

    def test_skipped_formulations_live_outside_the_scored_history(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.record_skipped(4, 2, {"Water": 30.0})
        assert opt.skipped == [
            {"formulation": 4, "batch": 2, "recipe": {"Water": 30.0}, "note": "Not made"}
        ]
        assert opt.X_history == []

    def test_delete_result_keeps_every_parallel_list_in_step(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        for k, v in enumerate([3.0, 6.0, 9.0]):
            opt.tell({"Water": v}, {"Firmness": v}, formulation_no=k + 1, batch_no=1, note=f"n{k}")
        opt.delete_result(1)
        assert opt.formulation_ids == [1, 3]
        assert opt.batch_history == [1, 1]
        assert opt.notes_history == ["n0", "n2"]
        assert len(opt.X_history) == len(opt.Y_history) == 2

    def test_rewind_keeps_every_parallel_list_in_step(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        for k, v in enumerate([3.0, 6.0, 9.0]):
            opt.tell({"Water": v}, {"Firmness": v}, formulation_no=k + 1, batch_no=1)
        opt.rewind_to(0)
        assert opt.formulation_ids == [1]
        assert opt.batch_history == [1]
        assert opt.notes_history == [""]

    def test_rewind_drops_left_out_rows_from_batches_past_the_cut(self, tmp_path, monkeypatch):
        """A left-out formulation belongs to its batch. Left behind, it would
        still be offered for deletion and would still make its batch look like
        the last one recorded."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.tell({"Water": 10.0}, {"Firmness": 5.0}, formulation_no=1, batch_no=1)
        opt.record_skipped(2, 1, {"Water": 20.0})
        opt.tell({"Water": 30.0}, {"Firmness": 6.0}, formulation_no=3, batch_no=2)
        opt.record_skipped(4, 2, {"Water": 40.0})
        opt.rewind_to(0)                     # keep batch 1's one scored row
        assert opt.formulation_ids == [1]
        assert [s['formulation'] for s in opt.skipped] == [2]
        assert opt.last_batch_no() == 1

    def test_undo_last_batch_removes_the_batch_and_retires_its_numbers(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.tell({"Water": 10.0}, {"Firmness": 5.0}, formulation_no=1, batch_no=1)
        opt.tell({"Water": 20.0}, {"Firmness": 6.0}, formulation_no=2, batch_no=2)
        opt.tell({"Water": 30.0}, {"Firmness": 4.0}, formulation_no=3, batch_no=2)
        opt.record_skipped(4, 2, {"Water": 40.0})
        assert opt.undo_last_batch() == (2, 3)
        assert opt.formulation_ids == [1]
        assert opt.batch_history == [1]
        assert opt.skipped == []
        assert opt.next_formulation_no == 5      # 2, 3 and 4 retire

    def test_undo_last_batch_finds_a_batch_that_was_entirely_left_out(self, tmp_path, monkeypatch):
        """A batch nobody made is still the last batch: undoing must take it,
        not the recorded batch before it."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.tell({"Water": 10.0}, {"Firmness": 5.0}, formulation_no=1, batch_no=1)
        opt.record_skipped(2, 2, {"Water": 20.0})
        opt.record_skipped(3, 2, {"Water": 30.0})
        assert opt.undo_last_batch() == (2, 2)
        assert opt.formulation_ids == [1]
        assert opt.batch_history == [1]
        assert opt.skipped == []

    def test_undo_last_batch_refuses_while_a_batch_is_open(self, tmp_path, monkeypatch):
        """Undo must never take an open batch down with it — its numbers would
        retire without the user asking."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.tell({"Water": 10.0}, {"Firmness": 5.0}, formulation_no=1, batch_no=1)
        opt.ask(n_suggestions=2)
        with pytest.raises(ValueError, match="Record or discard the open batch first."):
            opt.undo_last_batch()
        assert opt.pending_batch is not None
        assert opt.formulation_ids == [1]

    def test_undo_last_batch_refuses_while_an_open_batch_is_empty(self, tmp_path, monkeypatch):
        """An open batch with every row generated-and-removed is still open —
        `[]` is not the same as `None` — so undo must still refuse."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.tell({"Water": 10.0}, {"Firmness": 5.0}, formulation_no=1, batch_no=1)
        opt.set_pending_batch([])
        with pytest.raises(ValueError, match="Record or discard the open batch first."):
            opt.undo_last_batch()

    def test_undo_last_batch_returns_none_when_no_batch_is_numbered(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.tell({"Water": 10.0}, {"Firmness": 5.0})
        assert opt.undo_last_batch() is None

    def test_index_of_formulation(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.tell({"Water": 10.0}, {"Firmness": 5.0}, formulation_no=7)
        assert opt.index_of_formulation(7) == 0
        assert opt.index_of_formulation(8) is None

    def test_old_projects_backfill_numbers_on_load(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.tell({"Water": 10.0}, {"Firmness": 5.0})
        opt.tell({"Water": 20.0}, {"Firmness": 6.0})
        state = opt.export_json()
        for key in ("formulation_ids", "batch_history", "notes_history", "skipped",
                    "next_formulation_no", "pending_batch_no",
                    "pending_batch_created", "pending_batch_discarded"):
            state.pop(key, None)
        older = FoodOptimizer("older")
        older.import_json(state)
        assert older.formulation_ids == [1, 2]
        assert older.batch_history == [None, None]
        assert older.notes_history == ["", ""]
        assert older.skipped == []
        assert older.next_formulation_no == 3
        assert older.pending_batch_created is None
        assert older.pending_batch_discarded == []

    def test_a_legacy_pending_batch_of_bare_recipes_is_numbered_on_load(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.tell({"Water": 10.0}, {"Firmness": 5.0})
        state = opt.export_json()
        for key in ("formulation_ids", "next_formulation_no", "pending_batch_no"):
            state.pop(key, None)
        state["pending_batch"] = [{"Water": 30.0}, {"Water": 40.0}]
        older = FoodOptimizer("older_pending")
        older.import_json(state)
        assert [r["formulation"] for r in older.pending_batch] == [2, 3]
        assert older.next_formulation_no == 4

    def test_identity_survives_save_and_reload(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch, name="roundtrip")
        opt.ask(n_suggestions=2)
        row = opt.pending_batch[0]
        opt.tell(row["recipe"], {"Firmness": 6.0},
                 formulation_no=row["formulation"], batch_no=opt.pending_batch_no, note="ok")
        opt.record_skipped(2, 1, opt.pending_batch[1]["recipe"])
        reloaded = FoodOptimizer("roundtrip")
        assert reloaded.formulation_ids == [1]
        assert reloaded.batch_history == [1]
        assert reloaded.notes_history == ["ok"]
        assert reloaded.skipped[0]["formulation"] == 2
        assert reloaded.next_formulation_no == 3
        assert reloaded.pending_batch_no == 1


class TestUnitsAndImportance:
    def _opt(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("units")
        opt.set_amount_unit("g")
        opt.add_ingredient("Pea protein", 0, 25)
        opt.add_ingredient("Methylcellulose", 0, 3)
        opt.add_objective("Firmness", 1.5, goal="target", target=6,
                          min_val=0, max_val=10, unit="N")
        opt.add_objective("Juiciness", 1.0, goal="target", target=7,
                          min_val=0, max_val=10)
        return opt

    def test_join_unit_keeps_a_slash_unit_tight(self):
        from food_bo import join_unit
        assert join_unit("6", "N") == "6 N"
        assert join_unit("7", "/10") == "7/10"
        assert join_unit("7", "") == "7"

    def test_goal_line_reads_like_a_label(self):
        """A '/10' is shown once, on the measurement's own label, so it never
        follows a number: 'Firmness (/10) · target 7'."""
        from food_bo import goal_line
        assert goal_line({"goal": "target", "target": 6, "unit": "N"}) == "target 6 N"
        assert goal_line({"goal": "target", "target": 7, "unit": "/10"}) == "target 7"
        assert goal_line({"goal": "min", "unit": "N"}) == "lower is better"
        assert goal_line({"goal": "max", "unit": ""}) == "higher is better"

    def test_a_slash_unit_is_written_once_on_the_label(self):
        from food_bo import label_with_unit, unit_after_number
        assert label_with_unit("Firmness", "/10") == "Firmness (/10)"
        assert label_with_unit("Firmness", "N") == "Firmness"
        assert label_with_unit("Firmness", "") == "Firmness"
        assert unit_after_number("/10") == ""
        assert unit_after_number("N") == "N"
        assert unit_after_number(None) == ""

    def test_amount_unit_and_measurement_unit_persist(self, tmp_path, monkeypatch):
        self._opt(tmp_path, monkeypatch)
        reloaded = FoodOptimizer("units")
        assert reloaded.amount_unit == "g"
        assert reloaded.objectives[0]["unit"] == "N"

    def test_a_new_project_is_already_in_grams(self, tmp_path, monkeypatch):
        """A blank unit made every amount on every screen ambiguous."""
        monkeypatch.chdir(tmp_path)
        assert FoodOptimizer("fresh").amount_unit == "g"

    def test_a_project_saved_before_units_existed_opens_in_grams(
            self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("legacy")
        opt.add_ingredient("Water", 0, 100)
        state = opt.export_json()
        state.pop("amount_unit", None)          # a 0.2.x file has no unit
        opened = FoodOptimizer("legacy2")
        opened.import_json(state)
        assert opened.amount_unit == "g"
        # ...and the screen says the g was the app's guess, once.
        assert opened.amount_unit_backfilled is True

    def test_a_unit_cleared_on_purpose_stays_cleared(self, tmp_path, monkeypatch):
        """A blank unit is a decision — the amounts may be percentages. The
        default belonged to a MISSING key, not to a blank one, so reopening
        relabelled such a project grams without a word on screen."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_amount_unit("")
        assert opt.amount_unit == ""
        reopened = FoodOptimizer("units")
        assert reopened.amount_unit == ""
        assert reopened.amount_unit_backfilled is False
        # Round-tripping the export keeps the blank too.
        again = FoodOptimizer("units2")
        again.import_json(opt.export_json())
        assert again.amount_unit == ""

    def test_setting_the_unit_answers_the_backfill_notice(self, tmp_path,
                                                          monkeypatch):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("legacy3")
        opt.add_ingredient("Water", 0, 100)
        state = opt.export_json()
        state.pop("amount_unit")
        opt.import_json(state)
        assert opt.amount_unit_backfilled is True
        opt.set_amount_unit("%")
        assert opt.amount_unit_backfilled is False

    def test_a_process_setting_carries_its_own_unit(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_process_parameter("Cook temperature", 160, 200, unit="°C")
        setting = next(v for v in opt.variables if v.get('category') == 'process')
        assert setting["unit"] == "°C"
        # It rides the column header of every table the setting appears in,
        # and never the project's amount unit.
        assert opt._amount_column("Cook temperature") == "Cook temperature (°C)"
        assert opt._amount_column("Pea protein") == "Pea protein (g)"
        assert FoodOptimizer("units").variables[-1]["unit"] == "°C"

    def test_a_setting_saved_before_units_existed_backfills_to_blank(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_process_parameter("Cook temperature", 160, 200)
        state = opt.export_json()
        for var in state["variables"]:
            var.pop("unit", None)               # a 0.2.x setting had no unit
        opened = FoodOptimizer("units_legacy")
        opened.import_json(state)
        setting = next(v for v in opened.variables
                       if v.get('category') == 'process')
        assert setting["unit"] == ""
        assert opened._amount_column("Cook temperature") == "Cook temperature"

    def test_recorded_dates_are_the_users_own_date(self, tmp_path, monkeypatch):
        """Results are stamped in UTC. Slicing the stamp dated a batch
        recorded at 23:25 as tomorrow, on every row."""
        from food_bo import local_date
        assert local_date("2026-09-09T23:25:00+00:00") == (
            datetime.fromisoformat("2026-09-09T23:25:00+00:00")
            .astimezone().strftime("%Y-%m-%d"))
        assert local_date("") == ""
        assert local_date(None) == ""
        # Anything unparseable falls back to the first ten characters.
        assert local_date("not a date at all") == "not a date"
        opt = self._opt(tmp_path, monkeypatch)
        opt.tell({"Pea protein": 10.0, "Methylcellulose": 1.0},
                 {"Firmness": 6.0, "Juiciness": 7.0}, formulation_no=1,
                 batch_no=1)
        opt.timestamps_history[0] = "2026-09-09T23:25:00+00:00"
        expected = local_date("2026-09-09T23:25:00+00:00")
        assert opt.history_frame()["Recorded"].iloc[0] == expected
        assert expected in opt.history_csv()

    def test_the_target_refusal_names_the_range_in_plain_words(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        with pytest.raises(ValueError) as add:
            opt.add_objective("Chew", 1.0, goal="target", target=99,
                              min_val=0, max_val=10)
        assert str(add.value) == ("Target 99 must be between the range's "
                                  "lowest and highest (0 to 10).")
        with pytest.raises(ValueError) as edit:
            opt.update_objective("Firmness", target=99)
        assert str(edit.value) == ("Target 99 must be between the range's "
                                   "lowest and highest (0 to 10).")

    def test_a_backwards_range_is_refused_in_the_tab_s_words(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        with pytest.raises(ValueError) as e:
            opt.add_objective("Chew", 1.0, min_val=10, max_val=0)
        assert str(e.value) == "Range lowest must be less than range highest."

    def test_the_delete_refusal_names_the_formulations(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.tell({"Pea protein": 10.0, "Methylcellulose": 0.0},
                 {"Firmness": 6.0, "Juiciness": 7.0}, formulation_no=3, batch_no=1)
        with pytest.raises(ValueError) as one:
            opt.remove_ingredient("Pea protein")
        assert str(one.value) == (
            "'Pea protein' was used in Formulation 3, so it cannot be "
            "deleted. Tick 'Delete even if it was used' to discard that "
            "information."
        )
        opt.tell({"Pea protein": 12.0, "Methylcellulose": 0.0},
                 {"Firmness": 6.0, "Juiciness": 7.0}, formulation_no=5, batch_no=1)
        with pytest.raises(ValueError) as two:
            opt.remove_ingredient("Pea protein")
        assert "used in Formulations 3 and 5, so it cannot be deleted" in str(two.value)

    def test_measurements_are_ordered_by_importance(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        assert [o["name"] for o in opt.measurements_by_importance()] == ["Firmness", "Juiciness"]

    def test_score_line_carries_the_shares(self, tmp_path, monkeypatch):
        """No sentence about distance from a target: two of the three goals
        have none, and how closeness works lives in the expander below."""
        opt = self._opt(tmp_path, monkeypatch)
        assert opt.score_function_line() == (
            "Overall score = 1.5 (60 %) × Firmness closeness + 1 (40 %) × "
            "Juiciness closeness. Every measurement at its goal scores 2.50."
        )

    def test_share_of_score_sums_to_one_and_reads_as_whole_percent(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        assert opt.share_of_score("Firmness") == pytest.approx(0.6)
        assert opt.share_of_score("Juiciness") == pytest.approx(0.4)
        assert opt.share_text("Firmness") == "60 %"
        assert opt.share_text("Juiciness") == "40 %"

    def test_food_bo_join_unit_agrees_with_ui_helpers(self):
        import ui_helpers
        from food_bo import join_unit
        assert join_unit(60, "%") == ui_helpers.join_unit(60, "%") == "60 %"

    def test_update_objective_recomputes_scores_and_keeps_the_open_batch(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.tell({"Pea protein": 10.0, "Methylcellulose": 1.0},
                 {"Firmness": 6.0, "Juiciness": 3.0})
        opt.set_pending_batch([{"Pea protein": 5.0, "Methylcellulose": 0.5}])
        before = float(opt.Y_history[0])
        opt.update_objective("Firmness", weight=2.0)
        assert opt.objectives[0]["weight"] == 2.0
        assert float(opt.Y_history[0]) > before
        assert opt.utility_ceiling() == pytest.approx(3.0)
        assert opt.pending_batch is not None

    def test_adding_or_removing_a_measurement_keeps_the_open_batch(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_pending_batch([{"Pea protein": 5.0, "Methylcellulose": 0.5}])
        opt.add_objective("Chewiness", 0.5, goal="max", min_val=0, max_val=10)
        assert opt.pending_batch is not None
        opt.remove_objective("Chewiness")
        assert opt.pending_batch is not None

    def test_changing_an_ingredient_still_discards_the_open_batch(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_pending_batch([{"Pea protein": 5.0, "Methylcellulose": 0.5}])
        opt.add_ingredient("Beet juice powder", 0, 2)
        assert opt.pending_batch is None
        assert opt.pending_batch_no is None

    def test_update_objective_rejects_an_unknown_field(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        with pytest.raises(ValueError, match="Cannot change name"):
            opt.update_objective("Firmness", name="Hardness")

    def test_reserved_column_names_are_refused(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("reserved2")
        for bad in ("Formulation", "Batch", "Overall score", "Note", "Recorded", "Best"):
            with pytest.raises(ValueError, match="column name Food Optimizer uses"):
                opt.add_ingredient(bad, 0, 10)

    def test_closeness_details_only_measures_off_by_against_a_target(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.tell({"Pea protein": 10.0, "Methylcellulose": 1.0},
                 {"Firmness": 8.0, "Juiciness": None})
        rows = opt.closeness_details(0)
        assert [r["name"] for r in rows] == ["Firmness", "Juiciness"]
        assert rows[0] == {"name": "Firmness", "goal": "Target 6 N",
                           "measured": "8 N", "off_by": "2 N too high"}
        assert rows[1]["measured"] == "not scored"
        assert rows[1]["off_by"] == "not scored"

    def test_closeness_details_leaves_off_by_blank_for_higher_and_lower(self, tmp_path, monkeypatch):
        """A 'higher is better' measurement has no target, so an off-by number
        would read a pass as a failure."""
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("nogoal")
        opt.add_ingredient("Water", 0, 100)
        opt.add_objective("Juiciness", 1.0, goal="max", min_val=0, max_val=10, unit="/10")
        opt.add_objective("Grittiness", 0.5, goal="min", min_val=0, max_val=10, unit="/10")
        opt.tell({"Water": 10.0}, {"Juiciness": 8.0, "Grittiness": 2.0})
        rows = opt.closeness_details(0)
        # The /10 is on the name, once, and never after the number.
        assert rows[0] == {"name": "Juiciness (/10)", "goal": "Higher is better",
                           "measured": "8", "off_by": "—"}
        assert rows[1] == {"name": "Grittiness (/10)", "goal": "Lower is better",
                           "measured": "2", "off_by": "—"}

    def test_closeness_details_says_on_target(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.tell({"Pea protein": 10.0, "Methylcellulose": 1.0},
                 {"Firmness": 6.0, "Juiciness": 6.0})
        rows = opt.closeness_details(0)
        assert rows[0]["off_by"] == "On target"
        # Measured and Off by are written the same way: 6 and 1, not 6 and 1.0.
        assert rows[1]["off_by"] == "1 too low"
        assert rows[1]["measured"] == "6"

    def test_closeness_details_on_a_project_with_no_measurements(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.tell({"Pea protein": 10.0, "Methylcellulose": 1.0},
                 {"Firmness": 6.0, "Juiciness": 6.0})
        opt.remove_objective("Firmness")
        opt.remove_objective("Juiciness")
        assert opt.closeness_details(0) == []
        assert opt.score_function_line() == ""

    def test_closeness_details_guards_a_bad_index(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.tell({"Pea protein": 10.0, "Methylcellulose": 1.0},
                 {"Firmness": 6.0, "Juiciness": 6.0})
        assert opt.closeness_details(None) == []
        assert opt.closeness_details(-1) == []

    def test_biggest_changes_are_largest_first(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        changes = opt.biggest_changes({"Pea protein": 8.0, "Methylcellulose": 1.8},
                                      {"Pea protein": 10.1, "Methylcellulose": 1.0})
        assert changes[0][0] == "Pea protein"
        assert changes[0][1] == pytest.approx(-2.1)
        assert changes[1][0] == "Methylcellulose"
        assert changes[1][1] == pytest.approx(0.8)

    def test_biggest_changes_leaves_process_settings_out(self, tmp_path, monkeypatch):
        """A setting is not an amount: a cook temperature reported as
        '+180.00 g' priced an oven in grams, and against a formulation made
        before the setting existed the change was the whole baseline."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_process_parameter("Cook temperature", 160, 200, unit="°C")
        changes = opt.biggest_changes(
            {"Pea protein": 8.0, "Methylcellulose": 1.8, "Cook temperature": 180.0},
            {"Pea protein": 10.1, "Methylcellulose": 1.0, "Cook temperature": 0.0})
        assert [name for name, _ in changes] == ["Pea protein", "Methylcellulose"]

    def test_scaled_recipe_scales_amounts_and_leaves_settings_alone(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_process_parameter("Cook temperature", 100, 220)
        recipe = {"Pea protein": 10.0, "Methylcellulose": 1.0, "Cook temperature": 180.0}
        assert opt.ingredient_total(recipe) == pytest.approx(11.0)
        scaled = opt.scaled_recipe(recipe, 22.0)
        assert scaled["Pea protein"] == pytest.approx(20.0)
        assert scaled["Methylcellulose"] == pytest.approx(2.0)
        assert scaled["Cook temperature"] == pytest.approx(180.0)
        assert opt.scaled_recipe(recipe, None) == recipe
        # A stray key not tied to any current variable (e.g. left over from a
        # removed ingredient) must survive scaling, not just the unscaled copy.
        stray = dict(recipe, **{"Old ingredient": 3.0})
        assert set(opt.scaled_recipe(stray, 22.0)) == set(opt.scaled_recipe(stray, None))

    def test_history_frame_columns_and_star(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.tell({"Pea protein": 10.0, "Methylcellulose": 1.0},
                 {"Firmness": 3.0, "Juiciness": 3.0}, formulation_no=1, batch_no=1)
        opt.tell({"Pea protein": 12.0, "Methylcellulose": 1.0},
                 {"Firmness": 6.0, "Juiciness": 7.0}, formulation_no=2, batch_no=1,
                 note="best yet")
        opt.record_skipped(3, 1, {"Pea protein": 14.0, "Methylcellulose": 1.0})
        df = opt.history_frame()
        assert list(df.columns) == ["Best", "Batch", "Formulation", "Firmness (N)",
                                    "Juiciness", "Overall score", "Recorded", "Note"]
        assert list(df["Formulation"]) == [2, 1, 3]        # best first, skipped last
        # A row nobody made has no score to be best, and says so.
        assert list(df["Best"]) == ["★", "", "not made"]
        assert list(df["Batch"]) == ["1", "1", "1"]        # one type, always
        assert df["Note"].iloc[0] == "best yet"
        assert df["Note"].iloc[2] == "Not made"
        assert df["Overall score"].iloc[2] == ""

    def test_history_frame_trial_column_is_all_strings_on_a_mixed_project(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.tell({"Pea protein": 10.0, "Methylcellulose": 1.0},
                 {"Firmness": 6.0, "Juiciness": 7.0})            # no batch (imported)
        opt.tell({"Pea protein": 12.0, "Methylcellulose": 1.0},
                 {"Firmness": 6.0, "Juiciness": 7.0}, batch_no=1)
        assert set(opt.history_frame()["Batch"]) == {"", "1"}

    def test_history_frame_marks_a_partial_score(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.tell({"Pea protein": 10.0, "Methylcellulose": 1.0}, {"Firmness": 6.0})
        assert opt.history_frame()["Overall score"].iloc[0].endswith("· partial")

    def test_history_frame_orders(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.tell({"Pea protein": 10.0, "Methylcellulose": 1.0},
                 {"Firmness": 6.0, "Juiciness": 7.0}, formulation_no=1, batch_no=2)
        opt.tell({"Pea protein": 12.0, "Methylcellulose": 1.0},
                 {"Firmness": 1.0, "Juiciness": 1.0}, formulation_no=2, batch_no=1)
        assert list(opt.history_frame(order="Best first")["Formulation"]) == [1, 2]
        assert list(opt.history_frame(order="Newest first")["Formulation"]) == [2, 1]
        assert list(opt.history_frame(order="Batch order")["Formulation"]) == [2, 1]

    def test_history_frame_amounts_carry_the_unit(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.tell({"Pea protein": 10.0, "Methylcellulose": 1.0},
                 {"Firmness": 6.0, "Juiciness": 7.0})
        df = opt.history_frame(include_amounts=True)
        assert "Pea protein (g)" in df.columns
        assert df["Pea protein (g)"].iloc[0] == pytest.approx(10.0)

    def test_history_csv_uses_plain_names_and_carries_identity(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.tell({"Pea protein": 10.0, "Methylcellulose": 1.0},
                 {"Firmness": 6.0, "Juiciness": 7.0}, formulation_no=4, batch_no=2,
                 note="ok")
        df = pd.read_csv(io.StringIO(opt.history_csv()))
        for col in ("Formulation", "Batch", "Recorded", "Overall score",
                    "Pea protein", "Firmness", "Note"):
            assert col in df.columns
        assert df["Formulation"].iloc[0] == 4
        assert df["Batch"].iloc[0] == 2

    def test_history_csv_omits_not_made_rows_but_keeps_partial_rows(self, tmp_path, monkeypatch):
        """A Not-made row has no results at all, so exporting it would fail the
        importer's own 'these columns have blank cells' check for every
        measurement column in the sheet — it carries nothing to import, so it
        is left out. A partially scored row genuinely has one result and is
        kept, blank cell and all."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.tell({"Pea protein": 10.0, "Methylcellulose": 1.0},
                 {"Firmness": 6.0, "Juiciness": 7.0}, formulation_no=1, batch_no=1)
        opt.tell({"Pea protein": 12.0, "Methylcellulose": 1.0},
                 {"Firmness": 5.0}, formulation_no=2, batch_no=1)   # partial: no Juiciness
        opt.record_skipped(3, 1, {"Pea protein": 14.0, "Methylcellulose": 1.0})
        df = pd.read_csv(io.StringIO(opt.history_csv()))
        var_names = [v['name'] for v in opt.variables]
        obj_names = [o['name'] for o in opt.objectives]
        required = var_names + obj_names
        assert [c for c in required if c not in df.columns] == []   # the importer's own column check
        assert list(df["Formulation"]) == [1, 2]        # formulation 3 (Not made) is gone
        assert df["Firmness"].iloc[1] == 5.0
        assert pd.isna(df["Juiciness"].iloc[1])

    def test_batch_frame_is_the_make_these_table(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_pending_batch([{"Pea protein": 10.0, "Methylcellulose": 1.0}])
        df = opt.batch_frame(opt.pending_batch)
        assert list(df.columns) == ["Formulation", "Pea protein (g)",
                                    "Methylcellulose (g)", "Total (g)"]
        assert list(df["Formulation"]) == [1]
        assert df["Total (g)"].iloc[0] == pytest.approx(11.0)

    def test_batch_frame_and_csv_carry_the_scale_through(self, tmp_path, monkeypatch):
        """What is downloaded must equal what is on screen, or the lab weighs
        out the wrong amounts."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_pending_batch([{"Pea protein": 10.0, "Methylcellulose": 1.0}])
        df = opt.batch_frame(opt.pending_batch, scale_to=22.0)
        assert df["Pea protein (g)"].iloc[0] == pytest.approx(20.0)
        assert df["Total (g)"].iloc[0] == pytest.approx(22.0)
        sheet = pd.read_csv(io.StringIO(opt.batch_csv(opt.pending_batch, scale_to=22.0)))
        assert sheet["Pea protein (g)"].iloc[0] == 20.0
        assert sheet["Total (g)"].iloc[0] == pytest.approx(22.0)

    def test_batch_csv_has_formulation_numbers_and_blank_measurements(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_pending_batch([{"Pea protein": 11.877679824829102,
                                "Methylcellulose": 1.0}])
        df = pd.read_csv(io.StringIO(opt.batch_csv(opt.pending_batch)))
        assert list(df["Formulation"]) == [1]
        assert df["Pea protein (g)"].iloc[0] == 11.88
        assert df["Firmness"].isna().all()
        assert "Note" in df.columns

    def test_objectives_without_a_unit_key_backfill_to_blank_on_load(self, tmp_path, monkeypatch):
        """A 0.2.x project's objectives were saved before 'unit' existed;
        loading one must not raise a KeyError the first time a screen reads
        obj['unit']."""
        opt = self._opt(tmp_path, monkeypatch)
        state = opt.export_json()
        for o in state["objectives"]:
            o.pop("unit", None)
        older = FoodOptimizer("older_units")
        older.import_json(state)
        assert all(o["unit"] == "" for o in older.objectives)


class TestUnitPerIngredient:
    """Different ingredients are measured in different units: grams for the
    powders, millilitres for the water, and a process setting in °C or min.
    The project's own unit is only the DEFAULT a new ingredient starts with."""

    def _opt(self, tmp_path, monkeypatch, name="peruint"):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer(name)
        opt.add_ingredient("Pea protein", 0, 25)              # takes the default
        opt.add_ingredient("Water", 0, 60, unit="ml")
        opt.add_objective("Firmness", 1.0, goal="target", target=6,
                          min_val=0, max_val=10, unit="N")
        return opt

    def test_an_ingredient_keeps_the_unit_it_was_added_with(self, tmp_path,
                                                            monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        assert opt.unit_of("Water") == "ml"
        assert opt.unit_of("Pea protein") == "g"
        assert FoodOptimizer("peruint").unit_of("Water") == "ml"

    def test_an_ingredient_with_no_unit_of_its_own_follows_the_default(
            self, tmp_path, monkeypatch):
        """A 0.2.x project's ingredients were saved before an ingredient could
        carry a unit. They follow the project's default, so typing the right
        one into `Default unit for new ingredients` still fixes the whole
        project in one move, as the backfill notice promises."""
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("legacy_units")
        opt.add_ingredient("Water", 0, 100)
        state = opt.export_json()
        state.pop("amount_unit")                    # a 0.2.x file has no unit
        for var in state["variables"]:
            var.pop("unit", None)                   # nor did its ingredients
        opened = FoodOptimizer("legacy_units2")
        opened.import_json(state)
        assert opened.unit_of("Water") == "g"       # backfilled to the default
        opened.set_amount_unit("ml")
        assert opened.unit_of("Water") == "ml"

    def test_setting_one_ingredients_unit_leaves_the_others_alone(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        scores_before = list(opt.Y_history)
        opt.set_ingredient_unit("Pea protein", " kg ")
        assert opt.unit_of("Pea protein") == "kg"
        assert opt.unit_of("Water") == "ml"
        assert FoodOptimizer("peruint").unit_of("Pea protein") == "kg"
        assert list(opt.Y_history) == scores_before   # a unit rescores nothing

    def test_setting_the_unit_of_something_that_is_not_an_ingredient(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        with pytest.raises(ValueError, match="No ingredient named Salt."):
            opt.set_ingredient_unit("Salt", "g")

    def test_the_default_unit_only_reaches_new_ingredients(self, tmp_path,
                                                           monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_amount_unit("kg")
        assert opt.unit_of("Water") == "ml"           # its own unit stands
        opt.add_ingredient("Salt", 0, 3)
        assert opt.unit_of("Salt") == "kg"

    def test_the_ingredient_csv_carries_an_optional_unit_column(self, tmp_path,
                                                                monkeypatch):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("csv_units")
        df = pd.DataFrame({"name": ["Water", "Flour", "Salt"],
                           "min": [0, 0, 0], "max": [60, 100, 3],
                           "unit": ["ml", "", None],
                           "Fat per 100 g": [0.0, 1.0, 0.0]})
        opt.load_ingredients_from_csv(df)
        assert opt.unit_of("Water") == "ml"
        # A blank cell means "the project's default", not a blank unit.
        assert opt.unit_of("Flour") == "g"
        assert opt.unit_of("Salt") == "g"
        # ...and the unit column is not read as an ingredient property.
        assert set(opt.ingredient_properties["Water"]) == {"Fat per 100 g"}

    def test_the_shipped_csv_template_carries_the_unit_column(self, tmp_path,
                                                              monkeypatch):
        monkeypatch.chdir(tmp_path)
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        df = pd.read_csv(os.path.join(root, "data", "sample_ingredients.csv"))
        assert "unit" in df.columns
        opt = FoodOptimizer("template_units")
        opt.load_ingredients_from_csv(df)
        assert {opt.unit_of(v["name"]) for v in opt.variables} == {"g"}

    def test_every_table_header_carries_the_ingredients_own_unit(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_process_parameter("Cook temperature", 160, 200, unit="°C")
        assert opt._amount_column("Water") == "Water (ml)"
        assert opt._amount_column("Pea protein") == "Pea protein (g)"
        assert opt._amount_column("Cook temperature") == "Cook temperature (°C)"
        opt.tell({"Pea protein": 10.0, "Water": 40.0, "Cook temperature": 180.0},
                 {"Firmness": 6.0}, formulation_no=1, batch_no=1)
        frame = opt.history_frame(include_amounts=True)
        assert "Water (ml)" in frame.columns
        assert "Pea protein (g)" in frame.columns

    def test_the_batch_table_totals_per_unit(self, tmp_path, monkeypatch):
        """A total that added 25 g of powder to 40 ml of water was a number of
        nothing. Each unit gets its own total, in one cell."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_pending_batch([{"Pea protein": 10.0, "Water": 40.0}])
        df = opt.batch_frame(opt.pending_batch)
        assert list(df.columns) == ["Formulation", "Pea protein (g)",
                                    "Water (ml)", "Total"]
        assert df["Total"].iloc[0] == "10.00 g · 40.00 ml"

    def test_a_unit_that_adds_up_to_nothing_is_left_out_of_the_total(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_pending_batch([{"Pea protein": 10.0, "Water": 0.0}])
        assert opt.batch_frame(opt.pending_batch)["Total"].iloc[0] == "10.00 g"

    def test_one_unit_everywhere_keeps_the_total_a_number(self, tmp_path,
                                                          monkeypatch):
        """Nothing changes for a project whose ingredients are all in grams:
        one `Total (g)` column, holding a number the screen rounds itself."""
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("one_unit")
        opt.add_ingredient("Pea protein", 0, 25)
        opt.add_ingredient("Methylcellulose", 0, 3)
        opt.set_pending_batch([{"Pea protein": 10.0, "Methylcellulose": 1.0}])
        df = opt.batch_frame(opt.pending_batch)
        assert list(df.columns)[-1] == "Total (g)"
        assert df["Total (g)"].iloc[0] == pytest.approx(11.0)
        assert opt.one_amount_unit() == "g"

    def test_the_batch_sheet_carries_the_units(self, tmp_path, monkeypatch):
        """The sheet the lab fills in must say what the amounts are measured
        in, or 40 of water is 40 of nothing."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_pending_batch([{"Pea protein": 10.0, "Water": 40.0}])
        sheet = pd.read_csv(io.StringIO(opt.batch_csv(opt.pending_batch)))
        assert "Pea protein (g)" in sheet.columns
        assert sheet["Water (ml)"].iloc[0] == 40.0

    def test_ingredients_in_several_units_have_no_single_unit(self, tmp_path,
                                                              monkeypatch):
        """What the screen asks before offering to scale a batch to a total:
        scaling 25 g of powder and 40 ml of water to '400' means nothing."""
        opt = self._opt(tmp_path, monkeypatch)
        assert opt.one_amount_unit() is None
        opt.set_ingredient_unit("Water", "g")
        assert opt.one_amount_unit() == "g"

    def test_an_amount_limit_needs_one_unit(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        # The refusal names the fix: which ingredient to re-enter, and in
        # what.
        with pytest.raises(
                ValueError,
                match=r"A limit adds amounts, so these ingredients need "
                      r"one unit; enter Water in g instead of ml\."):
            opt.add_quantity_constraint(["Pea protein", "Water"], max_val=50)
        # All ingredients is the same limit over every one of them, refused
        # in the same words.
        with pytest.raises(
                ValueError,
                match=r"A limit adds amounts, so these ingredients need "
                      r"one unit; enter Water in g instead of ml\."):
            opt.add_total_mass_constraint(max_val=400)
        assert opt.quantity_constraints == []
        # One unit between them, and the same limit is accepted.
        opt.add_quantity_constraint(["Pea protein"], max_val=20)
        assert opt.quantity_constraints[0]["max"] == 20
        opt.set_ingredient_unit("Water", "g")
        opt.add_total_mass_constraint(max_val=400)
        assert len(opt.quantity_constraints) == 2

    def test_the_biggest_changes_are_reported_with_each_own_unit(
            self, tmp_path, monkeypatch):
        """biggest_changes hands back the names; the caller writes each one
        with that ingredient's unit."""
        opt = self._opt(tmp_path, monkeypatch)
        changes = opt.biggest_changes({"Pea protein": 12.0, "Water": 40.0},
                                      {"Pea protein": 10.0, "Water": 30.0})
        assert [n for n, _ in changes] == ["Water", "Pea protein"]
        assert [opt.unit_of(n) for n, _ in changes] == ["ml", "g"]

    def test_a_unit_change_removes_a_limit_it_breaks(self, tmp_path,
                                                     monkeypatch):
        """A limit is arithmetic, not a label: once its ingredients are in
        different units the sum it holds the next batch to is a sum of
        nothing. It is removed, and named back to the caller."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_ingredient_unit("Water", "g")
        opt.add_quantity_constraint(["Pea protein", "Water"], max_val=50)
        removed = opt.set_ingredient_unit("Water", "ml")
        assert [qc['ingredients'] for qc in removed] == [["Pea protein", "Water"]]
        assert opt.quantity_constraints == []
        assert FoodOptimizer("peruint").quantity_constraints == []

    def test_a_unit_change_leaves_a_limit_it_does_not_break(self, tmp_path,
                                                            monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_ingredient("Salt", 0, 3)
        opt.add_quantity_constraint(["Pea protein", "Salt"], max_val=20)
        assert opt.set_ingredient_unit("Water", "l") == []
        assert len(opt.quantity_constraints) == 1

    def test_changing_the_default_unit_removes_a_limit_it_breaks(
            self, tmp_path, monkeypatch):
        """The default is the unit of every ingredient without one of its
        own, so moving it can split a limit's ingredients apart too."""
        opt = self._opt(tmp_path, monkeypatch)          # Water is in ml
        opt.set_ingredient_unit("Water", "g")           # ...pinned to g
        opt.add_quantity_constraint(["Pea protein", "Water"], max_val=50)
        removed = opt.set_amount_unit("ml")             # Pea protein follows
        assert [r['ingredients'] for r in removed] == [["Pea protein", "Water"]]
        assert [r['reason'] for r in removed] == ["unit"]
        assert FoodOptimizer("peruint").quantity_constraints == []

    def test_changing_the_default_unit_leaves_a_limit_it_does_not_break(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_quantity_constraint(["Pea protein"], max_val=20)
        assert opt.set_amount_unit("kg") == []
        assert len(opt.quantity_constraints) == 1

    def test_reloading_the_ingredient_file_removes_limits_it_breaks(
            self, tmp_path, monkeypatch):
        """A reload can rewrite every unit and drop ingredients outright."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_ingredient_unit("Water", "g")
        opt.add_quantity_constraint(["Pea protein", "Water"], max_val=50)
        opt.add_quantity_constraint(["Pea protein"], max_val=20)
        removed = opt.load_ingredients_from_csv(pd.DataFrame({
            "name": ["Pea protein", "Water"], "min": [0, 0], "max": [25, 60],
            "unit": ["g", "ml"],
        }))
        assert [r['reason'] for r in removed] == ["unit"]
        assert [qc['ingredients'] for qc in opt.quantity_constraints] == \
            [["Pea protein"]]

    def test_reloading_without_an_ingredient_removes_the_limits_naming_it(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_ingredient_unit("Water", "g")
        opt.add_quantity_constraint(["Pea protein", "Water"], max_val=50)
        removed = opt.load_ingredients_from_csv(pd.DataFrame({
            "name": ["Pea protein"], "min": [0], "max": [25], "unit": ["g"],
        }))
        assert [r['reason'] for r in removed] == ["missing"]
        assert removed[0]['missing'] == ["Water"]
        assert opt.quantity_constraints == []

    def test_the_batch_sheet_totals_each_unit_in_its_own_column(
            self, tmp_path, monkeypatch):
        """A spreadsheet cannot add up '10.00 g · 40.00 ml', so the CSV gives
        each unit a column of numbers; the screen and the printable sheets
        keep the one cell."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_pending_batch([{"Pea protein": 10.0, "Water": 40.0}])
        sheet = pd.read_csv(io.StringIO(opt.batch_csv(opt.pending_batch)))
        assert list(sheet.columns) == ["Formulation", "Pea protein (g)",
                                       "Water (ml)", "Total (g)", "Total (ml)",
                                       "Firmness", "Note"]
        assert sheet["Total (g)"].iloc[0] == 10.0
        assert sheet["Total (ml)"].iloc[0] == 40.0
        # The screen is unchanged: one cell, both units.
        assert opt.batch_frame(opt.pending_batch)["Total"].iloc[0] == \
            "10.00 g · 40.00 ml"


class TestSettingsOnlyProject:
    """A fermentation project varies incubation temperature, time and culture
    dose. Nothing is weighed out, so nothing may claim a total."""

    def _opt(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("ferment_bo")
        opt.add_process_parameter("Incubation temperature", 30, 42, unit="°C")
        opt.add_process_parameter("Incubation time", 4, 16, unit="h")
        opt.add_objective("Acidity", 1.0, goal="target", target=4.5,
                          min_val=3.0, max_val=7.0, unit="pH")
        return opt

    def test_the_batch_table_and_sheet_carry_the_settings_and_no_total(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_pending_batch([{"Incubation temperature": 37.0,
                                "Incubation time": 8.0}])
        assert opt.has_ingredients() is False
        assert opt.total_column() is None
        df = opt.batch_frame(opt.pending_batch)
        assert list(df.columns) == ["Formulation",
                                    "Incubation temperature (°C)",
                                    "Incubation time (h)"]
        sheet = pd.read_csv(io.StringIO(opt.batch_csv(opt.pending_batch)))
        assert list(sheet.columns) == ["Formulation",
                                       "Incubation temperature (°C)",
                                       "Incubation time (h)", "Acidity",
                                       "Note"]

    def test_the_loop_runs_on_settings_alone(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.ask(n_suggestions=2)
        assert len(opt.pending_batch) == 2
        row = opt.pending_batch[0]
        opt.tell(row['recipe'], {"Acidity": 4.5},
                 formulation_no=row['formulation'],
                 batch_no=opt.pending_batch_no)
        assert opt.best_index() == 0
        assert opt.ingredient_total(row['recipe']) == 0.0
        # Nothing was weighed out, so no change can be reported as an amount.
        assert opt.biggest_changes(opt.pending_batch[1]['recipe'],
                                   row['recipe']) == []


class TestParseBatchResultsByFormulation:
    def _opt(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("pbr2")
        opt.add_ingredient("Water", 0, 100)
        opt.add_objective("Hardness", 1.0, goal="target", target=12, min_val=0, max_val=30)
        opt.add_objective("L*", 1.0, goal="max", min_val=0, max_val=100)
        opt.next_formulation_no = 7
        opt.set_pending_batch([{"Water": 10.0}, {"Water": 20.0}, {"Water": 30.0}],
                              batch_no=3)
        return opt

    def test_matches_global_numbers_case_insensitively(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        df = pd.DataFrame({"formulation": [7, 9], " hardness ": [11.0, 14.0],
                           "l*": [70.0, 65.0], "Note": ["", "soft"]})
        assert opt.parse_batch_results(df, opt.pending_batch) == [
            (7, {"Hardness": 11.0, "L*": 70.0}, ""),
            (9, {"Hardness": 14.0, "L*": 65.0}, "soft"),
        ]

    def test_legacy_recipe_and_experiment_columns_are_positions(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        legacy = pd.DataFrame({"Recipe": [1], "Hardness": [11.0], "L*": [70.0]})
        assert opt.parse_batch_results(legacy, opt.pending_batch)[0][0] == 7
        older = pd.DataFrame({"Experiment": [3], "Hardness": [11.0], "L*": [70.0]})
        assert opt.parse_batch_results(older, opt.pending_batch)[0][0] == 9

    def test_missing_key_column(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        df = pd.DataFrame({"Hardness": [1.0], "L*": [2.0]})
        with pytest.raises(ValueError, match="needs a Formulation column"):
            opt.parse_batch_results(df, opt.pending_batch)

    def test_missing_measurement_column_is_an_error(self, tmp_path, monkeypatch):
        """A measurement column absent from the sheet entirely is refused, not
        silently treated as unscored for every row — a typo'd header would
        otherwise drop that measurement from the whole batch without a word."""
        opt = self._opt(tmp_path, monkeypatch)
        df = pd.DataFrame({"Formulation": [7], "Hardness": [1.0]})
        with pytest.raises(ValueError, match=r"Missing columns: L\*"):
            opt.parse_batch_results(df, opt.pending_batch)

    def test_a_number_outside_the_trial_names_the_trial(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        df = pd.DataFrame({"Formulation": [11], "Hardness": [1.0], "L*": [5.0]})
        with pytest.raises(ValueError,
                           match=r"Formulation 11 is not in batch 3 \(it has 7, 8, 9\)\."):
            opt.parse_batch_results(df, opt.pending_batch)

    def test_a_blank_measurement_is_a_partial_result_not_an_error(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        df = pd.DataFrame({"Formulation": [7], "Hardness": [None], "L*": [70.0]})
        assert opt.parse_batch_results(df, opt.pending_batch) == [(7, {"L*": 70.0}, "")]

    def test_a_present_column_with_a_blank_cell_is_still_partial(self, tmp_path, monkeypatch):
        """The column existing is what matters; a blank cell within it is a
        per-row partial result, not the 'missing column' error."""
        opt = self._opt(tmp_path, monkeypatch)
        df = pd.DataFrame({"Formulation": [7], "Hardness": [11.0], "L*": [None]})
        assert opt.parse_batch_results(df, opt.pending_batch) == [(7, {"Hardness": 11.0}, "")]

    def test_a_row_with_nothing_filled_in_is_refused(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        df = pd.DataFrame({"Formulation": [7], "Hardness": [None], "L*": [None]})
        with pytest.raises(ValueError, match="Formulation 7 has no measurements filled in"):
            opt.parse_batch_results(df, opt.pending_batch)

    def test_out_of_range_value(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        df = pd.DataFrame({"Formulation": [7], "Hardness": [1.0], "L*": [140.0]})
        with pytest.raises(ValueError, match=r"outside your range of 0 to 100"):
            opt.parse_batch_results(df, opt.pending_batch)

    def test_the_out_of_range_refusal_follows_the_unit_rule(self, tmp_path,
                                                            monkeypatch):
        """A value read off an uploaded sheet is refused in exactly the words
        the results grid uses: a "/10" on the label, never after a number."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.update_objective("Hardness", unit="N", max_val=10, target=6)
        df = pd.DataFrame({"Formulation": [7], "Hardness": [12.0], "L*": [70.0]})
        with pytest.raises(ValueError) as with_unit:
            opt.parse_batch_results(df, opt.pending_batch)
        assert str(with_unit.value) == (
            "Formulation 7 Hardness 12 N is outside your range of 0 to 10 N. "
            "Widen the range in Set up, or check the value.")
        opt.update_objective("Hardness", unit="/10")
        with pytest.raises(ValueError) as slash:
            opt.parse_batch_results(df, opt.pending_batch)
        assert str(slash.value) == (
            "Formulation 7 Hardness 12 is outside your range of 0 to 10. "
            "Widen the range in Set up, or check the value.")

    def test_duplicate_row_is_rejected(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        df = pd.DataFrame({"Formulation": [7, 7], "Hardness": [11.0, 12.0],
                           "L*": [70.0, 71.0]})
        with pytest.raises(ValueError, match="appears more than once"):
            opt.parse_batch_results(df, opt.pending_batch)

    def test_non_integer_number(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        for bad in ("abc", 7.5):
            df = pd.DataFrame({"Formulation": [bad], "Hardness": [1.0], "L*": [5.0]})
            with pytest.raises(ValueError, match="is not a whole number"):
                opt.parse_batch_results(df, opt.pending_batch)

    def test_empty_sheet_is_rejected(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        df = pd.DataFrame({"Formulation": [], "Hardness": [], "L*": []})
        with pytest.raises(ValueError, match="no result rows"):
            opt.parse_batch_results(df, opt.pending_batch)


import ast
import pathlib
import re

_USER_FACING_SOURCES = ["app.py", "ui_helpers.py", "ui_setup.py", "ui_batch.py",
                        "ui_results.py", "food_bo.py", "storage.py", "wording.py"]
# The user-facing files that are not Python. They are scanned as plain text,
# except the Swift wrapper, where only its string literals are screen text.
_USER_FACING_TEXT = ["desktop/start_here.txt", "desktop/README.md", "README.md"]
_USER_FACING_SWIFT = "desktop/FoodOptimizerApp.swift"

_BANNED = [
    re.compile(r"\brecipes?\b", re.I),
    re.compile(r"\bexperiments?\b", re.I),
    re.compile(r"\bobjectives?\b", re.I),
    re.compile(r"\bweight(s|ed)?\b", re.I),
    # Range is the measurement's own word from 0.3.0 ("outside your range of
    # 0 to 10 N"); the plural only ever named an ingredient's allowed
    # amounts, which is what that is called.
    re.compile(r"\branges\b", re.I),
    re.compile(r"\branged\b", re.I),
    re.compile(r"\brewind(s|ing)?\b", re.I),
    re.compile(r"\balgorithm\b", re.I),
    # "Food Optimizer" is the product's name and stays; nothing else on screen
    # calls itself an optimizer or talks about optimization.
    re.compile(r"(?<!Food )\boptimi[sz](er|ation)\b", re.I),
    re.compile(r"Overall Score"),
    # The coherence wave (2026-09-10): one word per concept on every screen.
    # Lowest/Highest for the ends of a range (Min and Max survive only as
    # column headers an ingredient CSV may carry); Remove for anything taken
    # out of a project, with Delete kept for the project itself; Not made for
    # a formulation nobody made; Formulation total for the total a batch is written
    # to; and no Priority column beside the importance it was a rank of.
    re.compile(r"\bMin\b"),
    re.compile(r"\bMax\b"),
    re.compile(r"\bhard reset\b", re.I),
    re.compile(r"\bleave (it )?out\b", re.I),
    re.compile(r"\bscale each formulation\b", re.I),
    re.compile(r"\btotal amount\b", re.I),
    # The wording wave (2026-09-10): the words a formulation team uses, one
    # per concept. Delete replaces Remove everywhere (they were the same act
    # under two verbs, and Delete no longer belongs to the project alone);
    # Type replaces Kind; Range replaces Scale for a measurement; Repeat
    # replaces Remake; and Share was gone outright, back when it was a bare
    # column whose meaning was unclear. It now returns as "Share of score",
    # which says what it is a share of, so only a bare "Share" is banned.
    #
    # The batch wording wave (2026-09-10): the owner's team calls a round of
    # formulations a BATCH, so TRIAL — the old name for that same set — is
    # banned in its place. Stored field names keep their old spelling —
    # batch_history and pending_batch never reach a screen.
    re.compile(r"\btrials?\b", re.I),
    re.compile(r"\bscales?\b", re.I),
    re.compile(r"\bKind\b"),
    re.compile(r"\bRemove\b"),
    re.compile(r"\bRemake\b"),
    re.compile(r"\bShare\b(?! of score)"),
]

class TestRoundTwoFixes:
    """Verification round 2: a name the app owns, a unit edit that reaches the
    limits, a property that keeps its capitals, and a score written once."""

    def _opt(self, tmp_path, monkeypatch, name="round2"):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer(name)
        opt.set_amount_unit("g")
        opt.add_ingredient("Pea protein", 0, 100)
        opt.add_ingredient("Water", 0, 100)
        opt.add_objective("Firmness", 1.0, goal="target", target=6,
                          min_val=0, max_val=10, unit="N")
        return opt

    def test_total_is_reserved_like_every_other_column_the_app_owns(
            self, tmp_path, monkeypatch):
        """Two columns named 'Total (g)' break the batch table outright, and
        on the sheet the ingredient's own column carries the batch total."""
        opt = self._opt(tmp_path, monkeypatch)
        for name in ("Total", "total", "Total (g)", "TOTAL (ml)"):
            with pytest.raises(ValueError, match="column name Food Optimizer"):
                opt.add_ingredient(name, 0, 10)
        with pytest.raises(ValueError, match="column name Food Optimizer"):
            opt.add_process_parameter("Total", 0, 10)
        with pytest.raises(ValueError, match="column name Food Optimizer"):
            opt.add_objective("Total", 1.0, goal="max")
        assert [v['name'] for v in opt.variables] == ["Pea protein", "Water"]

    def test_total_is_refused_in_an_ingredient_csv_too(self, tmp_path,
                                                       monkeypatch):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("csv_total")
        df = pd.DataFrame({"Name": ["Flour", "Total (g)"], "Min": [0, 0],
                           "Max": [100, 10]})
        with pytest.raises(ValueError, match="Row 3: Total \\(g\\) is a column "
                                             "name Food Optimizer uses"):
            opt.load_ingredients_from_csv(df)
        # The refusal comes before the file is saved, so nothing on disk holds
        # a column the app owns.
        assert "Total (g)" not in [v['name'] for v in opt.variables]

    def test_re_adding_an_ingredient_prunes_the_limits_it_breaks(
            self, tmp_path, monkeypatch):
        """add_ingredient on a name the project already has is a unit edit,
        and it was the one way a project could hold a limit that adds grams
        to millilitres."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_total_mass_constraint(min_val=50, max_val=150)
        removed = opt.add_ingredient("Water", 0, 100, unit="ml")
        assert opt.quantity_constraints == []
        assert [r['reason'] for r in removed] == ["unit"]
        assert set(removed[0]['ingredients']) == {"Pea protein", "Water"}
        # A limit that still means something is left alone.
        opt.set_ingredient_unit("Water", "g")
        opt.add_total_mass_constraint(max_val=150)
        assert opt.add_ingredient("Water", 0, 90, unit="g") == []
        assert len(opt.quantity_constraints) == 1

    def test_a_property_keeps_the_capitals_the_file_gave_it(self, tmp_path,
                                                            monkeypatch):
        """The picker shows this name and the caption above it writes 'Cost or
        Sodium per 100 g', so lower-casing it read as a different column."""
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("props")
        opt.load_ingredients_from_csv(pd.DataFrame({
            "Name": ["Flour", "Oil"], "Min": [0, 0], "Max": [100, 50],
            "Fat per 100 g": [1.0, 90.0]}))
        assert set(opt.ingredient_properties["Oil"]) == {"Fat per 100 g"}
        # Matching ignores capitals, so a limit written by an older project
        # (which stored the name lower-cased) still finds the column.
        opt.add_constraint("fat per 100 g", max_val=10.0)
        assert opt._check_constraints({"Flour": 10.0, "Oil": 1.0}) is True
        assert opt._check_constraints({"Flour": 10.0, "Oil": 2.0}) is False
        # ...and a second limit on the same column replaces the first.
        opt.add_constraint("Fat per 100 g", max_val=20.0)
        assert len(opt.constraints) == 1

    def test_the_downloaded_scores_are_the_scores_on_screen(self, tmp_path,
                                                            monkeypatch):
        """The table shows 2.62; the file used to hold 2.625."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_objective("Juiciness", 0.75, goal="target", target=7,
                          min_val=0, max_val=10, unit="/10")
        opt.tell({"Pea protein": 10.0, "Water": 5.0},
                 {"Firmness": 5.5, "Juiciness": 6.5}, formulation_no=1,
                 batch_no=1)
        frame = pd.read_csv(io.StringIO(opt.history_csv()))
        on_screen = opt.history_frame()["Overall score"].iloc[0]
        assert f"{frame['Overall score'].iloc[0]:.2f}" == on_screen
        assert frame["Overall score"].iloc[0] == round(
            float(opt.Y_history[0]), 2)


# Sentences that are allowed to keep a banned word, each for a stated reason.
_ALLOWED_EXACT = {
    # The one legacy value that must stay spelled the old way: it is the
    # reserved column name a 0.2.x project could collide with.
    "Overall Score",
    # "Delete" is now the app's one verb for taking something out, so it is
    # no longer restricted; these entries stay because the sidebar's
    # sentences are assembled from fragments the scan sees one at a time.
    "Delete this project",
    "Yes, delete it",
    "Delete **",
    "**? It has no formulations yet, "
    "and it leaves this list. ",
    "** and its ",
}

# Fragments of the sidebar's delete-the-project sentences (they are f-strings,
# so each piece is scanned on its own).
_ALLOWED_PREFIXES = ("Delete **", "Deleted ")

# Single-word literals that are internal machinery, never screen text.
_ALLOWED_SINGLE_WORDS = {
    # Legacy CSV column headers an import still accepts, and the reserved
    # names a 0.2.x project could collide with (RESERVED_VARIABLE_NAMES,
    # which keeps Trial reserved too: a project stored before this wording
    # wave may carry that column).
    "Recipe", "Experiment", "Trial",
    # The ingredient file's own column headers. The boxes on screen say
    # Lowest and Highest; a sheet may head its columns either way, and the
    # loader reads both.
    "Min", "Max", "min", "max", "lowest", "highest",
    # Stored field names and JSON keys. The spec keeps the stored spelling of
    # importance ('weight') and of a formulation's amounts ('recipe').
    "recipe", "experiment", "experiments", "objectives", "weight",
}
# Fragments removed from the Swift wrapper before it is scanned: CSS property
# names inside the setup page's inline styles, not prose.
_SWIFT_NOT_PROSE = ("font-weight",)

_SINGLE_WORDS = re.compile(
    # 'ranges', not 'range': Range is the measurement's own column header.
    r"^(recipes?|experiments?|objectives?|weights?|ranges|rewind|pruned"
    r"|priority|trials?|kind|scales?|remove|remake)$", re.I)


def _how_it_works():
    """The one collapsed expander that maps the app's words to the
    optimization concepts. It is the single place variable, objective,
    weight and constraint are allowed to be said, so it is read from the
    source rather than copied here: a bullet reworded in the app cannot
    quietly fall out of the allowance."""
    from wording import HOW_IT_WORKS
    return set(HOW_IT_WORKS)


def _string_constants(path):
    """Every string literal in a Python file, minus docstrings — those are
    notes to the next engineer, not screen text."""
    tree = ast.parse(pathlib.Path(path).read_text())
    docstrings = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)) and body:
            first = body[0]
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                docstrings.add(id(first.value))
    return [node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
            and id(node) not in docstrings]


def _prose_constants(path):
    """The literals that read like a sentence (they contain a space)."""
    return [v for v in _string_constants(path) if " " in v]


def _single_word_constants(path):
    """The one-word literals. A screen label is often a single word — 'Recipe',
    'Weight' — so the prose pass alone would miss the words that matter most."""
    return [v for v in _string_constants(path) if " " not in v]


def test_no_old_vocabulary_reaches_the_user():
    """recipe → formulation, experiment → formulation, objective →
    measurement, weight → importance, ranges → allowed amounts (a
    measurement's own Range is the one exception), rewind → undo, trial →
    batch, Kind → Type, Scale → Range, Remove → Delete, Remake → Repeat,
    a bare Share → gone (it now returns only as "Share of score"), and
    nothing on screen mentions an algorithm or optimization. 'Overall
    Score' → 'Overall score'. The one exemption is the How it works
    expander, which exists to say these words once."""
    root = pathlib.Path(__file__).resolve().parent.parent
    allowed = _ALLOWED_EXACT | _how_it_works()
    offenders = []
    for name in _USER_FACING_SOURCES:
        for text in _prose_constants(root / name):
            if text in allowed or text.startswith(_ALLOWED_PREFIXES):
                continue
            if any(pattern.search(text) for pattern in _BANNED):
                offenders.append((name, text))
        for word in _single_word_constants(root / name):
            if word in _ALLOWED_SINGLE_WORDS:
                continue
            if _SINGLE_WORDS.fullmatch(word):
                offenders.append((name, word))
    assert offenders == [], offenders


def test_no_old_vocabulary_reaches_the_user_outside_python():
    """Start Here, the two READMEs and the app window's own copy are read by
    the same people, so they follow the same vocabulary."""
    root = pathlib.Path(__file__).resolve().parent.parent
    offenders = []
    for name in _USER_FACING_TEXT:
        for i, line in enumerate(( root / name).read_text().splitlines(), 1):
            if any(pattern.search(line) for pattern in _BANNED):
                offenders.append((name, i, line))
    swift = (root / _USER_FACING_SWIFT).read_text()
    for fragment in _SWIFT_NOT_PROSE:
        swift = swift.replace(fragment, "")
    for literal in re.findall(r'"((?:[^"\\\n]|\\.)*)"', swift):
        if any(pattern.search(literal) for pattern in _BANNED):
            offenders.append((_USER_FACING_SWIFT, literal))
    assert offenders == [], offenders


class TestPropertiesNamedInTheApp:
    """A property used to arrive only as an extra column in an ingredient CSV,
    so a project typed in by hand could not limit sodium at all. It can now be
    named in the app, given a value per ingredient, and removed."""

    def _opt(self, tmp_path, monkeypatch, name="props_in_app"):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer(name)
        opt.set_amount_unit("g")
        opt.add_ingredient("Salt", 0, 10)
        opt.add_ingredient("Water", 0, 100)
        opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
        return opt

    def test_a_property_can_be_named_and_survives_a_reload(self, tmp_path,
                                                           monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        assert opt.properties() == []
        assert opt.add_property("  Sodium per 100 g ") == "Sodium per 100 g"
        assert opt.properties() == ["Sodium per 100 g"]
        again = FoodOptimizer("props_in_app")
        assert again.properties() == ["Sodium per 100 g"]
        assert again.property_names == ["Sodium per 100 g"]

    def test_a_property_name_cannot_collide_with_anything_else(self, tmp_path,
                                                               monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_process_parameter("Cook temperature", 100, 200)
        with pytest.raises(ValueError, match="already the name of an ingredient"):
            opt.add_property("Salt")
        with pytest.raises(ValueError, match="already the name of a process setting"):
            opt.add_property("Cook temperature")
        with pytest.raises(ValueError, match="already the name of a measurement"):
            opt.add_property("Taste")
        with pytest.raises(ValueError, match="column name Food Optimizer"):
            opt.add_property("Total (g)")
        with pytest.raises(ValueError, match="Name cannot be empty"):
            opt.add_property("   ")
        opt.add_property("Cost")
        with pytest.raises(ValueError, match="already a property"):
            opt.add_property("cost")
        assert opt.properties() == ["Cost"]

    def test_a_value_is_set_cleared_and_told_apart_from_a_gap(self, tmp_path,
                                                              monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_property("Sodium per 100 g")
        opt.set_property_value("Salt", "Sodium per 100 g", 39000)
        opt.set_property_value("Water", "Sodium per 100 g", 0)
        assert opt.property_value("Salt", "Sodium per 100 g") == 39000.0
        # A 0 is a value; a blank is not, and the limit line tells them apart.
        assert opt.has_property_value("Water", "Sodium per 100 g") is True
        assert opt.ingredients_without_property("Sodium per 100 g") == []
        opt.set_property_value("Water", "Sodium per 100 g", None)
        assert opt.has_property_value("Water", "Sodium per 100 g") is False
        assert opt.property_value("Water", "Sodium per 100 g") == 0.0
        assert opt.ingredients_without_property("Sodium per 100 g") == ["Water"]
        # Capitals do not make a second property.
        opt.set_property_value("Salt", "sodium PER 100 G", 100)
        assert opt.ingredient_properties["Salt"] == {"Sodium per 100 g": 100.0}

    def test_a_value_needs_an_ingredient_and_a_property_that_exist(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_property("Cost")
        opt.add_process_parameter("Cook temperature", 100, 200)
        with pytest.raises(ValueError, match="No ingredient named Flour"):
            opt.set_property_value("Flour", "Cost", 1)
        # A process setting is weighed into nothing, so it carries no property.
        with pytest.raises(ValueError, match="No ingredient named Cook"):
            opt.set_property_value("Cook temperature", "Cost", 1)
        with pytest.raises(ValueError, match="No property named Fat"):
            opt.set_property_value("Salt", "Fat", 1)

    def test_a_hand_made_property_holds_a_batch_to_its_limit(self, tmp_path,
                                                             monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_property("Sodium per 100 g")
        opt.set_property_value("Salt", "Sodium per 100 g", 39000)
        opt.set_property_value("Water", "Sodium per 100 g", 0)
        opt.add_constraint("Sodium per 100 g", max_val=450)
        # 1 g of salt in 100 g of formulation averages 390 per 100 g.
        assert opt.property_per_100({"Salt": 1.0, "Water": 99.0},
                                    "Sodium per 100 g") == pytest.approx(390.0)
        assert opt._check_constraints({"Salt": 1.0, "Water": 99.0}) is True
        assert opt._check_constraints({"Salt": 2.0, "Water": 98.0}) is False

    def test_removing_a_property_takes_its_values_and_its_limits(self, tmp_path,
                                                                 monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_property("Sodium per 100 g")
        opt.add_property("Cost")
        opt.set_property_value("Salt", "Sodium per 100 g", 39000)
        opt.set_property_value("Salt", "Cost", 2)
        opt.add_constraint("Sodium per 100 g", max_val=450)
        opt.add_constraint("Cost", max_val=5)
        removed = opt.remove_property("sodium per 100 g")
        assert [c['metric'] for c in removed] == ["Sodium per 100 g"]
        assert opt.properties() == ["Cost"]
        assert opt.ingredient_properties["Salt"] == {"Cost": 2.0}
        assert [c['metric'] for c in opt.constraints] == ["Cost"]
        with pytest.raises(ValueError, match="No property named"):
            opt.remove_property("Sodium per 100 g")
        assert FoodOptimizer("props_in_app").properties() == ["Cost"]

    def test_a_csv_still_names_its_property_columns(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("csv_props")
        opt.add_property("Cost")
        opt.load_ingredients_from_csv(pd.DataFrame({
            "Name": ["Flour", "Oil"], "Min": [0, 0], "Max": [100, 50],
            "Fat per 100 g": [1.0, 90.0], "Sodium per 100 g": [1.0, 0.0]}))
        # The file's columns join the list, in the file's own order, and a
        # property named in the app earlier keeps its place.
        assert opt.properties() == ["Cost", "Fat per 100 g", "Sodium per 100 g"]
        assert opt.ingredient_properties["Oil"]["Fat per 100 g"] == 90.0
        assert opt.ingredients_without_property("Cost") == ["Flour", "Oil"]

    def test_a_property_survives_export_and_import(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_property("Sodium per 100 g")
        opt.set_property_value("Salt", "Sodium per 100 g", 39000)
        state = opt.export_json()
        assert state['property_names'] == ["Sodium per 100 g"]
        FoodOptimizer.validate_state(state)      # a good backup passes
        fresh = FoodOptimizer("restored_props")
        fresh.import_json(state)
        assert fresh.properties() == ["Sodium per 100 g"]
        assert fresh.property_value("Salt", "Sodium per 100 g") == 39000.0

    def test_a_malformed_property_list_is_refused_before_import(self, tmp_path,
                                                                monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        state = opt.export_json()
        state['property_names'] = [{"name": "Sodium"}]
        with pytest.raises(ValueError, match="'property_names' section"):
            FoodOptimizer.validate_state(state)
        state['property_names'] = None
        # An absent or null list is simply no properties, not a broken file.
        FoodOptimizer.validate_state(state)
