"""Tests for the FoodOptimizer core engine."""

import json
import os
import pickle
import tempfile

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
        assert opt.ingredient_properties["Flour"]["fat"] == 1.0

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
        with pytest.raises(ValueError, match="Min.*must be less than Max"):
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
        with pytest.raises(ValueError, match="within the range"):
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


def test_history_frame_is_1_based_and_chronological(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("hist")
    opt.add_ingredient("Water", 0, 100)
    opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
    opt.tell({"Water": 10.0}, {"Taste": 3.0})
    opt.tell({"Water": 20.0}, {"Taste": 8.0})
    df = opt.history_frame()
    assert list(df["Experiment"]) == [1, 2]
    assert list(df.columns[:3]) == ["Experiment", "Date", "Overall Score"]
    assert "Taste (result)" in df.columns and "Water" in df.columns
    assert df["Date"].iloc[0] and len(df["Date"].iloc[0]) == 10


def test_reserved_column_name_is_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("reserved")
    with pytest.raises(ValueError, match="column name Food Optimizer uses"):
        opt.add_ingredient("Date", 0, 10)


def test_history_frame_renames_variable_colliding_with_fixed_column(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("hist_collision")
    opt.add_ingredient("Water", 0, 100)
    opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
    # Bypass add_ingredient's validation to mirror a project that already has
    # a variable literally named "Date" (e.g. imported from an old file).
    opt.variables.append({
        'name': 'Date',
        'type': 'continuous',
        'bounds': (0.0, 10.0),
        'category': 'ingredient',
        'active': True,
    })
    opt.tell({"Water": 10.0, "Date": 5.0}, {"Taste": 3.0})
    df = opt.history_frame()
    assert "Date" in df.columns
    assert isinstance(df["Date"].iloc[0], str)
    assert "Date (ingredient)" in df.columns
    assert df["Date (ingredient)"].iloc[0] == 5.0


def test_batch_frame_has_recipe_labels(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("bf")
    df = opt.batch_frame([{"Water": 10.0, "Temp": 180.0}, {"Water": 20.0, "Temp": 190.0}])
    assert list(df["Recipe"]) == [1, 2]
    assert list(df.columns) == ["Recipe", "Water", "Temp"]


class TestParseBatchResults:
    def _opt(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("pbr")
        opt.add_ingredient("Water", 0, 100)
        opt.add_objective("Hardness", 1.0, goal="target", target=12, min_val=0, max_val=30)
        opt.add_objective("L*", 1.0, goal="max", min_val=0, max_val=100)
        return opt

    def test_parses_matching_rows_case_insensitively(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        batch = [{"Water": 10.0}, {"Water": 20.0}, {"Water": 30.0}]
        df = pd.DataFrame({"recipe": [1, 3], " hardness ": [11.0, 14.0], "l*": [70.0, 65.0]})
        parsed = opt.parse_batch_results(df, batch)
        assert parsed == [(0, {"Hardness": 11.0, "L*": 70.0}), (2, {"Hardness": 14.0, "L*": 65.0})]

    def test_missing_recipe_column(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        df = pd.DataFrame({"Hardness": [1.0], "L*": [2.0]})
        with pytest.raises(ValueError, match="needs a Recipe column"):
            opt.parse_batch_results(df, [{"Water": 10.0}])

    def test_missing_measurement_column(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        df = pd.DataFrame({"Recipe": [1], "Hardness": [1.0]})
        with pytest.raises(ValueError, match="Missing columns: L\\*"):
            opt.parse_batch_results(df, [{"Water": 10.0}])

    def test_blank_cell_and_unknown_recipe(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        batch = [{"Water": 10.0}]
        with pytest.raises(ValueError, match="Recipe 1 Hardness is blank"):
            opt.parse_batch_results(pd.DataFrame({"Recipe": [1], "Hardness": [None], "L*": [5.0]}), batch)
        with pytest.raises(ValueError, match="Recipe 7 is not in this batch"):
            opt.parse_batch_results(pd.DataFrame({"Recipe": [7], "Hardness": [1.0], "L*": [5.0]}), batch)

    def test_out_of_range_value(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        with pytest.raises(ValueError, match="Recipe 1 L\\* is 140.*0 to 100"):
            opt.parse_batch_results(pd.DataFrame({"Recipe": [1], "Hardness": [1.0], "L*": [140.0]}),
                                    [{"Water": 10.0}])

    def test_duplicate_recipe_row_is_rejected(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        batch = [{"Water": 10.0}, {"Water": 20.0}]
        df = pd.DataFrame({"Recipe": [1, 1], "Hardness": [11.0, 12.0], "L*": [70.0, 71.0]})
        with pytest.raises(ValueError, match="appears more than once"):
            opt.parse_batch_results(df, batch)

    def test_non_integer_recipe_number(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        batch = [{"Water": 10.0}]
        with pytest.raises(ValueError, match="is not a whole number"):
            opt.parse_batch_results(
                pd.DataFrame({"Recipe": ["abc"], "Hardness": [1.0], "L*": [5.0]}), batch)
        with pytest.raises(ValueError, match="is not a whole number"):
            opt.parse_batch_results(
                pd.DataFrame({"Recipe": [1.5], "Hardness": [1.0], "L*": [5.0]}), batch)

    def test_empty_sheet_is_rejected(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        df = pd.DataFrame({"Recipe": [], "Hardness": [], "L*": []})
        with pytest.raises(ValueError, match="no result rows"):
            opt.parse_batch_results(df, [{"Water": 10.0}])


def test_history_csv_roundtrips_through_importer_columns(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("csv")
    opt.add_ingredient("Water", 0, 100)
    opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
    opt.tell({"Water": 10.0}, {"Taste": 3.0})
    import io
    df = pd.read_csv(io.StringIO(opt.history_csv()))
    for col in ["Experiment", "Date", "Overall Score", "Water", "Taste"]:
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
        df = pd.DataFrame({
            "Name": ["Water", "Oil"],
            "Min": [0, 0],
            "Max": [100, 50],
            "Fat": [0.0, 0.8],
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
        with pytest.raises(ValueError, match="Add at least one objective"):
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
        with pytest.raises(ValueError, match="must lie within"):
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
        with pytest.raises(ValueError, match="must stay in play"):
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
        with pytest.raises(ValueError, match="Every variable is inactive"):
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
        with pytest.raises(ValueError, match="nonzero amount"):
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
        with pytest.raises(ValueError, match="process parameter"):
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

    def test_newer_version_is_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        state = FoodOptimizer("tmp_validate").export_json()
        state["CLASS_VERSION"] = FoodOptimizer.CLASS_VERSION + 1
        with pytest.raises(ValueError, match="newer version"):
            FoodOptimizer.validate_state(state)

    def test_summary_of_valid_backup(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("src")
        opt.add_ingredient("Water", 0, 100)
        opt.add_objective("Taste", 1.0, goal="max")
        opt.tell({"Water": 50.0}, {"Taste": 7.0})
        summary = FoodOptimizer.validate_state(opt.export_json())
        assert summary == {"name": "src", "experiments": 1, "ingredients": 1,
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
        with pytest.raises(ValueError, match="Min must be less than Max"):
            opt.add_process_parameter("Temp", 200, 100)
        with pytest.raises(ValueError, match="Min must be less than Max"):
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
        with pytest.raises(ValueError, match="Min must be less than Max"):
            opt.add_quantity_constraint(["Sugar", "Honey"], min_val=50, max_val=10)

    def test_duplicate_quantity_constraint_replaces(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_ingredient("Sugar", 0, 100)
        opt.add_ingredient("Honey", 0, 100)
        opt.add_quantity_constraint(["Sugar", "Honey"], max_val=50)
        opt.add_quantity_constraint(["Honey", "Sugar"], max_val=40)
        assert len(opt.quantity_constraints) == 1
        assert opt.quantity_constraints[0]['max'] == 40.0


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
