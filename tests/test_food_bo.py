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

import openpyxl

import wording
from food_bo import (
    FoodOptimizer, ingredients_template_workbook,
    _name_taken_message, FormulaError, LinearForm, parse_formula,
)


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

    def test_load_ingredients_file_lowercase_columns(self, opt):
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

    def test_load_ingredients_file_missing_column_plain_error(self, opt):
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
        with pytest.raises(ValueError, match="cannot be reloaded"):
            opt.load_ingredients_from_csv(df)

    def test_load_csv_min_above_max_raises(self, opt):
        """Equal is a FIXED row and is allowed; inverted is a range with
        nothing in it."""
        df = pd.DataFrame({
            "Name": ["BadIngredient"],
            "Min": [50],
            "Max": [10],
        })
        with pytest.raises(ValueError,
                           match="Lowest.*cannot be above Highest"):
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
        """What the measurement is worth is typed as its share of the score
        now, so the refusal for a nought says so in those words."""
        opt = self._opt(tmp_path, monkeypatch)
        with pytest.raises(ValueError, match="share above 0"):
            opt.add_objective("Taste", 0.0)

    def test_target_must_lie_in_range(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        with pytest.raises(ValueError, match="must be between Scale minimum"):
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
    assert list(df.columns[:3]) == ["Best so far", "Round", "Formulation"]
    assert list(df["Formulation"]) == [2, 1]
    assert "Taste" in df.columns
    assert len(df["Date recorded"].iloc[0]) == 10


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
    # No what-is-it-trying column during the cold start: every cell under it
    # repeated the header word for word.
    assert list(df.columns) == ["Formulation", "Water (g)", "Total (g)", "Temp"]
    assert df["Total (g)"].iloc[0] == 10.0   # the process setting is not an amount


def test_recipe_lines_sorts_largest_first_and_omits_zeros(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("bf")
    recipe = {"Water": 1.0, "Salt": 0.0, "Sugar": 5.0}
    assert opt.recipe_lines(recipe) == [("Sugar", 5.0), ("Water", 1.0)]
    assert opt.recipe_lines(recipe, limit=1) == [("Sugar", 5.0)]


def test_history_csv_carries_the_units_the_screen_shows(tmp_path, monkeypatch):
    """The amount columns are headed exactly as the All formulations table
    and the batch sheet head them. A bare "Water" column whose numbers were
    millilitres was the one place in the app an amount had no unit on it."""
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("csv")
    opt.add_ingredient("Water", 0, 100)
    opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
    opt.tell({"Water": 10.0}, {"Taste": 3.0})
    df = pd.read_csv(io.StringIO(opt.history_csv()))
    for col in ["Formulation", "Round", "Date recorded", "Overall score",
                "Water (g)", "Taste", "Note"]:
        assert col in df.columns, list(df.columns)
    assert "Water" not in df.columns
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
    assert "Honey (g)" in df.columns, list(df.columns)
    assert len(df) == 2
    assert df["Honey (g)"].notna().all()


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
    """The closeness arithmetic, on a project assembled in memory — so the
    importances are still on the scale the caller passed them on (see
    _shares_to_100: a series of adds has no last one the model can
    recognise). What a SAVED project reads them against is 100, and
    TestSharesAreTheWeights holds that half."""

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
        with pytest.raises(ValueError,
                           match="No ingredient or process setting named"):
            opt.set_variable_unit("Nutmeg", "g")

    def test_the_stranded_limit_refusals_are_written_per_100_g(self, opt):
        """The number in the refusal is an average now, so it says so: a
        bare 5 read as five grams of fat in the whole batch."""
        opt = self._fatty(opt)                # Lean 10 fat, Fatty 30 fat
        opt.add_objective("Taste", weight=1.0, goal="max", min_val=0, max_val=10)
        opt.add_constraint("Fat per 100 g", min_val=25)
        with pytest.raises(ValueError) as caught:
            opt.add_ingredient("Fatty", 0, 0)
        assert ("the ingredients that can still vary only reach 10 per 100 g "
                "at most. Loosen the limit first." in str(caught.value)), \
            str(caught.value)

        opt.remove_constraint(0)
        opt.add_constraint("Fat per 100 g", max_val=15)
        with pytest.raises(ValueError) as caught:
            opt.add_ingredient("Lean", 0, 0)
        assert ("the ingredients that can still vary cannot get below 30 per "
                "100 g. Loosen the limit first." in str(caught.value)), \
            str(caught.value)

    def test_fixing_that_strands_a_limit_says_so_in_per_100_terms(self, opt):
        """Fixing the only ingredient that carries the fat at nothing leaves
        a minimum nothing can reach."""
        opt = self._fatty(opt)
        opt.add_objective("Taste", weight=1.0, goal="max", min_val=0, max_val=10)
        opt.add_constraint("Fat per 100 g", min_val=25)
        with pytest.raises(ValueError, match="impossible to meet"):
            opt.add_ingredient("Fatty", 0, 0)
        assert opt.fixed_variables() == []


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
#  Non-monotone active set (adaptive EGBO pruning), now read off the
#  allowed amounts: a row whose Lowest is its Highest is out of the search.
# ------------------------------------------------------------------ #


class TestActiveSet:
    def test_no_pruning_is_a_noop(self, opt_configured):
        """The monotone path must be untouched when nothing is fixed."""
        assert opt_configured._get_fixed_features() == {}
        assert len(opt_configured.varying_variables()) == 3
        assert opt_configured.fixed_variables() == []

    def test_fixing_pins_the_column(self, opt_configured):
        opt_configured.add_ingredient("Flour", 0, 0)
        assert opt_configured._get_fixed_features() == {1: 0.0}
        assert [v["name"] for v in opt_configured.fixed_variables()] == ["Flour"]

    def test_a_fixed_column_is_pinned_inside_a_frame_that_has_a_width(
            self, opt_configured):
        """A point has no frame of its own: normalizing by a zero span is a
        division by zero, and it is the history that goes through it."""
        opt_configured.add_ingredient("Flour", 12.5, 12.5)  # was (0, 50)
        assert opt_configured._get_fixed_features() == {1: 0.0}
        bounds = opt_configured._get_bounds()
        assert bounds[0][1].item() == 12.5 and bounds[1][1].item() == 13.5

    def test_the_frame_stretches_to_cover_what_was_already_recorded(
            self, opt_configured):
        opt_configured.tell({"Water": 50.0, "Flour": 40.0, "Sugar": 5.0},
                            {"Taste": 7.0})
        opt_configured.add_ingredient("Flour", 10, 10)
        bounds = opt_configured._get_bounds()
        assert bounds[0][1].item() == 10.0 and bounds[1][1].item() == 40.0
        # ...and the fixed value still lands on its own amount.
        assert opt_configured._get_fixed_features() == {1: 0.0}

    def test_an_inverted_range_is_refused(self, opt_configured):
        with pytest.raises(ValueError, match="cannot be above Highest"):
            opt_configured.add_ingredient("Sugar", 99.0, 1.0)

    def test_fixing_twice_is_idempotent(self, opt_configured):
        opt_configured.add_ingredient("Flour", 0, 0)
        opt_configured.add_ingredient("Flour", 0, 0)
        assert len(opt_configured.fixed_variables()) == 1

    def test_pruning_preserves_history_and_width(self, opt_configured):
        """Fixing restricts the search domain, it must not touch observations."""
        opt_configured.tell(
            {"Water": 50.0, "Flour": 20.0, "Sugar": 5.0}, {"Taste": 7.0}
        )
        before_x = [row[:] for row in opt_configured.X_history]
        before_y = list(opt_configured.Y_history)

        opt_configured.add_ingredient("Flour", 0, 0)

        assert opt_configured.X_history == before_x
        assert opt_configured.Y_history == before_y
        assert len(opt_configured.X_history[0]) == 3

    def test_ask_respects_pins_in_cold_start(self, opt_configured):
        opt_configured.add_ingredient("Flour", 0, 0)
        batch = opt_configured.ask(n_suggestions=3, n_init_random=5)
        assert all(r["Flour"] == 0.0 for r in batch)
        assert any(r["Water"] > 0.0 for r in batch)

    def test_ask_respects_pins_after_gp_fit(self, opt_configured):
        opt_configured.add_ingredient("Flour", 0, 0)
        for _ in range(6):
            rec = opt_configured.ask(n_suggestions=1, n_init_random=5)[0]
            opt_configured.tell(rec, {"Taste": 5.0 + rec["Water"] / 100.0})
        batch = opt_configured.ask(n_suggestions=2, n_init_random=5)
        assert all(r["Flour"] == 0.0 for r in batch)

    def test_ask_guards_empty_active_set(self, opt_configured):
        for name in ("Flour", "Sugar", "Water"):
            opt_configured.add_ingredient(name, 1, 1)
        with pytest.raises(ValueError, match="fixed at one amount"):
            opt_configured.ask(n_suggestions=1)

    def test_fixing_detects_stranded_quantity_constraint(self, opt_configured):
        opt_configured.add_quantity_constraint(["Water", "Flour"], min_val=120.0)
        with pytest.raises(ValueError, match="impossible to meet"):
            opt_configured.add_ingredient("Flour", 0, 0)
        # Refused before anything was written.
        assert opt_configured._var_by_name("Flour")["bounds"] == (0.0, 50.0)
        opt_configured.add_ingredient("Sugar", 0, 0)  # unrelated, still fine
        assert [v["name"] for v in opt_configured.fixed_variables()] == ["Sugar"]

    def test_fixing_detects_stranded_property_constraint(self, opt, monkeypatch):
        """A limit that PASSES before the save and fails after it: a is the
        only ingredient carrying the protein, so pinning it at nothing takes
        the minimum out of reach."""
        opt.load_ingredients_from_csv(pd.DataFrame([
            {"Name": "a", "Min": 0, "Max": 10, "Protein": 20.0},
            {"Name": "b", "Min": 0, "Max": 10, "Protein": 0.0},
        ]))
        opt.add_objective("Taste", weight=1.0, goal="max", min_val=0, max_val=10)
        opt.add_constraint("protein", min_val=15.0)
        with pytest.raises(ValueError, match="impossible to meet"):
            opt.add_ingredient("a", 0, 0)
        assert opt._var_by_name("a")["bounds"] == (0.0, 10.0)

    def test_a_narrowed_range_that_strands_a_limit_is_still_allowed(
            self, opt_configured):
        """Only a save that FIXES a row is guarded. Narrowing one has always
        been the user's own business, and the guard must not quietly become
        a rule about every edit."""
        opt_configured.add_quantity_constraint(["Water", "Flour"], min_val=120.0)
        opt_configured.add_ingredient("Flour", 0, 1)
        assert opt_configured._var_by_name("Flour")["bounds"] == (0.0, 1.0)

    def test_a_fixed_row_survives_a_reload(self, opt_configured, monkeypatch):
        opt_configured.add_ingredient("Flour", 2.5, 2.5)
        reloaded = FoodOptimizer(project_name="test_project")
        assert [v["name"] for v in reloaded.fixed_variables()] == ["Flour"]
        assert reloaded._fixed_value(reloaded._var_by_name("Flour")) == 2.5

    def test_a_fixed_row_survives_a_json_roundtrip(self, opt_configured):
        opt_configured.add_ingredient("Flour", 12.5, 12.5)
        state = opt_configured.export_json()
        clone = FoodOptimizer(project_name="clone_project")
        clone.import_json(state)
        assert [v["name"] for v in clone.fixed_variables()] == ["Flour"]
        assert clone._get_fixed_features() == {1: 0.0}

    def test_legacy_variables_default_to_varying(self, opt_configured):
        """Projects saved before pruning existed must load fully in play."""
        for var in opt_configured.variables:
            var.pop("active", None)
        opt_configured.save()
        reloaded = FoodOptimizer(project_name="test_project")
        assert len(reloaded.varying_variables()) == 3
        assert reloaded._get_fixed_features() == {}

    def test_export_trajectory_reports_pruned_and_past_usage(self, opt_configured):
        opt_configured.tell(
            {"Water": 50.0, "Flour": 20.0, "Sugar": 5.0}, {"Taste": 7.0}
        )
        opt_configured.add_ingredient("Flour", 0, 0)
        text = opt_configured.export_trajectory()
        assert "Fixed" in text
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

    def test_blocks_removing_the_last_row_with_a_range(self, opt_configured):
        opt_configured.add_ingredient("Flour", 0, 0)
        opt_configured.add_ingredient("Sugar", 0, 0)
        with pytest.raises(ValueError,
                           match="last ingredient or setting that can still vary"):
            opt_configured.remove_ingredient("Water")


class TestValidateState:
    def test_empty_dict_is_rejected(self):
        with pytest.raises(ValueError, match="not a Food Optimizer copy"):
            FoodOptimizer.validate_state({})

    def test_non_dict_is_rejected(self):
        with pytest.raises(ValueError, match="not a Food Optimizer copy"):
            FoodOptimizer.validate_state([1, 2, 3])

    def test_wrong_shape_is_rejected(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        state = FoodOptimizer("tmp_validate").export_json()
        state["variables"] = "garbage"
        with pytest.raises(ValueError, match="damaged"):
            FoodOptimizer.validate_state(state)

    def test_wrong_element_shape_is_rejected(self):
        with pytest.raises(ValueError, match="damaged"):
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
        with pytest.raises(ValueError, match="damaged"):
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
            with pytest.raises(ValueError, match="damaged"):
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
                               match="damaged"):
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
                               match="damaged"):
                FoodOptimizer.validate_state(state)
        FoodOptimizer.validate_state(dict(base, skipped=[good]))
        # A note is the one field that may be absent: record_skipped always
        # writes one, but a hand-edited file without it still renders.
        FoodOptimizer.validate_state(dict(
            base, skipped=[{k: v for k, v in good.items() if k != "note"}]))

    def test_the_reason_a_copy_was_refused_goes_to_the_log_not_the_screen(
            self, tmp_path, monkeypatch, caplog):
        """'This copy\'s \'recipe_history\' section has the wrong shape.'
        was four of the words the app retired, programmer punctuation and a
        shape, shown to a food scientist whose saved copy will not open. One
        sentence reaches them; the field is what an engineer needs."""
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("tmp_logged")
        opt.add_ingredient("Water", 0, 100)
        opt.add_objective("Taste", 1.0, goal="max")
        state = opt.export_json()
        state["recipe_history"] = ["not a formulation"]
        with caplog.at_level("WARNING"):
            with pytest.raises(ValueError) as caught:
                FoodOptimizer.validate_state(state)
        assert str(caught.value) == wording.COPY_DAMAGED
        assert "recipe_history" not in str(caught.value)
        assert any("recipe_history" in record.getMessage()
                   for record in caplog.records), caplog.records

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
        with pytest.raises(ValueError, match="damaged"):
            FoodOptimizer.validate_state(dict(state, formulation_ids=[7, 7]))
        # A left-out formulation and a scored one cannot share a number.
        clash = dict(state)
        clash["skipped"] = [{"formulation": 7, "batch": 1, "recipe": {},
                             "note": "Not made"}]
        with pytest.raises(ValueError, match="damaged"):
            FoodOptimizer.validate_state(clash)
        # Neither can one batch, twice over.
        twice = dict(state)
        twice["pending_batch"] = [{"formulation": 9, "recipe": {"Water": 1.0}},
                                  {"formulation": 9, "recipe": {"Water": 2.0}}]
        twice["next_formulation_no"] = 10
        with pytest.raises(ValueError, match="damaged"):
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
        with pytest.raises(ValueError, match="damaged"):
            FoodOptimizer.validate_state(dict(state, next_formulation_no=1))
        with pytest.raises(ValueError,
                           match="damaged"):
            FoodOptimizer.validate_state(dict(state, formulation_ids=[0]))
        with pytest.raises(ValueError,
                           match="damaged"):
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
        """Equal is how a row is FIXED, so only Lowest ABOVE Highest is a
        refusal."""
        opt = self._opt(tmp_path, monkeypatch)
        with pytest.raises(ValueError, match="Lowest cannot be above Highest"):
            opt.add_process_parameter("Temp", 200, 100)
        with pytest.raises(ValueError, match="Lowest cannot be above Highest"):
            opt.add_ingredient("Water", 50, 5)
        opt.add_ingredient("Water", 5, 5)
        assert opt._var_by_name("Water")['bounds'] == (5.0, 5.0)

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
        # The limit form's own two boxes, named: it has no Lowest or Highest
        # on it — those belong to the ingredient above.
        with pytest.raises(ValueError,
                           match="At least must be less than At most"):
            opt.add_quantity_constraint(["Sugar", "Honey"], min_val=50, max_val=10)

    def test_duplicate_quantity_constraint_replaces(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_ingredient("Sugar", 0, 100)
        opt.add_ingredient("Honey", 0, 100)
        opt.add_quantity_constraint(["Sugar", "Honey"], max_val=50)
        opt.add_quantity_constraint(["Honey", "Sugar"], max_val=40)
        assert len(opt.quantity_constraints) == 1
        assert opt.quantity_constraints[0]['max'] == 40.0


def test_a_name_differing_only_by_case_is_refused(tmp_path, monkeypatch):
    """Two rows called "Oat flour" and "oat flour" are two rows with one name
    on every table in the app, and the CSV importer matches columns without
    regard to case, so the second could never be filled in."""
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("case")
    opt.add_ingredient("Oat flour", 0, 50)
    opt.add_objective("Firmness", 1.0, goal="max", min_val=0, max_val=10)
    with pytest.raises(ValueError, match="Oat flour already exists"):
        opt.add_ingredient("oat flour", 0, 60)
    with pytest.raises(ValueError, match="Oat flour already exists"):
        opt.add_process_parameter("OAT FLOUR", 0, 60)
    with pytest.raises(ValueError, match="That measurement already exists"):
        opt.add_objective("firmness", 1.0, goal="max", min_val=0, max_val=10)
    assert [v['name'] for v in opt.variables] == ["Oat flour"]
    assert [o['name'] for o in opt.objectives] == ["Firmness"]
    # The exact spelling is still an edit, not a refusal.
    opt.add_ingredient("Oat flour", 0, 60)
    assert opt.variables[0]['bounds'] == (0.0, 60.0)


def test_the_changes_leave_out_a_fixed_ingredient(tmp_path, monkeypatch):
    """A fixed ingredient is at one amount in every new formulation, so it
    cannot be a change this round made."""
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("fixed_changes")
    opt.add_ingredient("Water", 0, 100)
    opt.add_ingredient("Oil", 0, 100)
    opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
    opt.tell({"Water": 10.0, "Oil": 50.0}, {"Taste": 5.0})
    opt.add_ingredient("Oil", 50, 50)
    changes = opt._variable_deltas({"Water": 12.0, "Oil": 90.0},
                                   {"Water": 10.0, "Oil": 50.0}, 'ingredient')
    assert [name for name, _ in changes] == ["Water"]


def test_generating_prints_nothing_to_the_console(tmp_path, monkeypatch,
                                                 capsys):
    """Cold and warm alike: a desktop user has no console to read, and the
    launcher's log is for the launcher."""
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("quiet")
    opt.add_ingredient("Water", 0, 100)
    opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
    capsys.readouterr()
    opt.ask(n_suggestions=1)
    assert "DEBUG" not in capsys.readouterr().out
    for i in range(5):
        opt.tell({"Water": 10.0 * i}, {"Taste": 3.0 + i})
    opt.set_pending_batch(None)
    capsys.readouterr()
    opt.ask(n_suggestions=1)             # warm now: len(X_history) >= 5
    assert "DEBUG" not in capsys.readouterr().out


def test_sample_ingredients_file_has_readable_names(tmp_path, monkeypatch):
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
    assert len(sample_names) == 9
    assert list(sample_opt.ingredient_grid_frame()[wording.NAME_LABEL]) == [
        "Textured pea protein", "Textured soy protein", "Hydration water", "Remaining water",
        "Dry blend", "Wheat gluten", "Seasoning blend", "Fats and oils"]
    assert all("_" not in name for name in sample[wording.NAME_LABEL])
    # The template's headers are the add form's own words — Name, Lowest,
    # Highest, Unit — so a reader filling it in is answering the same four
    # questions the screen asks. data/ingredients.csv is the experiments'
    # list, not a template, and keeps the lowercase headers those scripts
    # read by name; the columns are the same columns either way.
    assert list(sample.columns) == ["Name", "Part of", "Preparation", "Lowest",
                                    "Highest", "Composition (%)", "Calculation", "Unit",
                                    "Fat per 100 g", "Sodium per 100 g",
                                    "Water per 100 g", "Cost per 100 g"]
    assert [c.lower() for c in df.columns] == [
        "name", "min", "max", "unit", "fat per 100 g", "sodium per 100 g"]

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

    def test_a_split_cold_start_is_the_same_five_points(self, tmp_path,
                                                        monkeypatch):
        """The caption promises five formulations spread across the allowed
        amounts. A fresh scramble per call gave 3 + 2 five points from two
        unrelated sequences — clustered exactly where the caption said they
        would not be. One sequence per project, fast-forwarded past what it
        has already issued, so a batch split makes no difference."""
        split = self._opt(tmp_path, monkeypatch, name="cold_split")
        split.ask(n_suggestions=3)
        first_three = self._amounts(split)
        split.tell({"Water": first_three[0]["Water"]}, {"Firmness": 6.0},
                   formulation_no=1, batch_no=1)
        split.tell({"Water": first_three[1]["Water"]}, {"Firmness": 6.0},
                   formulation_no=2, batch_no=1)
        split.tell({"Water": first_three[2]["Water"]}, {"Firmness": 6.0},
                   formulation_no=3, batch_no=1)
        split.set_pending_batch(None)
        split.ask(n_suggestions=2)
        split_five = first_three + self._amounts(split)

        whole = self._opt(tmp_path, monkeypatch, name="cold_split")
        # Same project name, so the same scramble; a clean counter, so the
        # sequence starts at its first point.
        whole.storage._seen.pop("cold_split", None)
        whole.X_history, whole.Y_history = [], []
        whole.recipe_history, whole.results_history = [], []
        whole.formulation_ids, whole.batch_history = [], []
        whole.notes_history, whole.timestamps_history = [], []
        whole.next_formulation_no = 1
        whole.pending_batch = None
        whole.pending_batch_no = None
        whole.ask(n_suggestions=5)
        assert self._amounts(whole) == split_five

    def test_a_regenerated_cold_start_still_differs(self, tmp_path,
                                                    monkeypatch):
        """One sequence per project, but the numbers have moved on, so
        `Generate a different batch` really is a different batch."""
        opt = self._opt(tmp_path, monkeypatch, name="cold_regen")
        opt.ask(n_suggestions=3)
        first = self._amounts(opt)
        opt.set_pending_batch(None)
        opt.ask(n_suggestions=3)
        assert self._amounts(opt) != first

    def test_the_sobol_seed_is_fixed_and_stored(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch, name="cold_seed")
        opt.ask(n_suggestions=1)
        seed = opt.sobol_seed
        assert isinstance(seed, int)
        assert FoodOptimizer("cold_seed").sobol_seed == seed
        # A file written before the seed was stored derives one from the
        # project's own name, so it is the same on every machine.
        state = opt.export_json()
        del state['sobol_seed']
        old = FoodOptimizer("cold_seed")
        old.import_json(state)
        assert old.sobol_seed is None
        assert old._sobol_seed() == seed

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
        """`Generate a different batch` as the screen does it: the old rows'
        numbers retire, and the batch keeps the number it was wearing — which
        is why ask() is told that number rather than being renumbered after
        it has already opened a batch of its own."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.ask(n_suggestions=3)
        opt.set_pending_batch(None)          # "Generate a different batch"
        opt.ask(n_suggestions=2, batch_no=1, discarded=[1, 2, 3])
        assert [r["formulation"] for r in opt.pending_batch] == [4, 5]
        assert opt.pending_batch_no == 1     # the batch keeps its number
        # And no batch number was spent behind the user's back: the batch
        # after this one is 2, not 3.
        assert opt.next_batch_no() == 2

    def test_a_batch_number_is_never_reissued_after_a_delete(self, tmp_path,
                                                              monkeypatch):
        """Two different sets of formulations must never wear one batch
        number in one project's records. The counter was read off the
        batches still on file, so deleting batch 2 handed its number
        straight back to the next batch generated."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.ask(n_suggestions=1)
        opt.tell({"Water": 10.0}, {"Firmness": 6.0}, formulation_no=1,
                 batch_no=1)
        opt.set_pending_batch(None)
        opt.ask(n_suggestions=1)
        assert opt.pending_batch_no == 2
        opt.tell({"Water": 20.0}, {"Firmness": 5.0}, formulation_no=2,
                 batch_no=2)
        opt.set_pending_batch(None)
        opt.delete_formulations([2])          # batch 2 leaves the project
        assert opt.last_batch_no() == 1       # nothing of batch 2 remains
        opt.ask(n_suggestions=1)
        assert opt.pending_batch_no == 3

    def test_a_batch_number_survives_a_reload(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.ask(n_suggestions=1)
        opt.set_pending_batch(None)
        again = FoodOptimizer(opt.project_name)
        again.ask(n_suggestions=1)
        assert again.pending_batch_no == 2

    def test_a_file_written_before_batch_numbers_were_stored_continues(
            self, tmp_path, monkeypatch):
        """A 60ed1d7-era file has no next_batch_number at all. It must open
        and go on from one past the highest batch it holds, exactly as a
        0.2.x file does for formulation numbers."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.tell({"Water": 10.0}, {"Firmness": 6.0}, formulation_no=1,
                 batch_no=1)
        opt.tell({"Water": 20.0}, {"Firmness": 5.0}, formulation_no=2,
                 batch_no=2)
        state = opt.export_json()
        del state['next_batch_number']
        assert 'next_batch_number' not in state
        old = FoodOptimizer(opt.project_name)
        old.import_json(state)
        assert old.next_batch_no() == 3
        old.ask(n_suggestions=1)
        assert old.pending_batch_no == 3

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
            {"formulation": 4, "batch": 2, "recipe": {"Water": 30.0},
             "note": "Not scored"}
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
        with pytest.raises(ValueError, match="Record or discard the open round first."):
            opt.undo_last_batch()
        assert opt.pending_batch is not None
        assert opt.formulation_ids == [1]

    def test_undo_last_batch_refuses_while_an_open_batch_is_empty(self, tmp_path, monkeypatch):
        """An open batch with every row generated-and-removed is still open —
        `[]` is not the same as `None` — so undo must still refuse."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.tell({"Water": 10.0}, {"Firmness": 5.0}, formulation_no=1, batch_no=1)
        opt.set_pending_batch([])
        with pytest.raises(ValueError, match="Record or discard the open round first."):
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

    def test_score_skipped_moves_the_row_into_history_keeping_number_and_batch(
            self, tmp_path, monkeypatch):
        """A formulation left not scored can be scored later. It keeps the
        number and the batch it was generated with, and nothing is issued:
        both numbers retired the day the row was recorded, and a second
        number for one formulation is two rows for one bowl."""
        opt = self._opt(tmp_path, monkeypatch, name="scoreskipped")
        opt.ask(n_suggestions=2)
        batch_no = opt.pending_batch_no
        first, second = opt.pending_batch[0], opt.pending_batch[1]
        opt.tell(first["recipe"], {"Firmness": 5.0},
                 formulation_no=first["formulation"], batch_no=batch_no)
        opt.record_skipped(second["formulation"], batch_no, second["recipe"],
                           note="Not scored · burner failed")
        opt.set_pending_batch(None)
        next_f, next_b = opt.next_formulation_no, opt.next_batch_number
        total_before = opt.batch_total(batch_no)
        opt.score_skipped(second["formulation"], {"Firmness": 7.0})
        assert opt.skipped == []
        assert opt.formulation_ids == [first["formulation"],
                                       second["formulation"]]
        assert opt.batch_history == [batch_no, batch_no]
        assert opt.recipe_history[-1] == second["recipe"]
        # The note said why it was not scored. It is scored now, so the row
        # carries no reason for a state it is no longer in.
        assert opt.notes_history[-1] == ""
        assert opt.next_formulation_no == next_f
        assert opt.next_batch_number == next_b
        assert opt.batch_total(batch_no) == total_before
        reloaded = FoodOptimizer("scoreskipped")
        assert reloaded.skipped == []
        assert reloaded.formulation_ids == [first["formulation"],
                                            second["formulation"]]
        assert reloaded.next_formulation_no == next_f

    def test_an_old_projects_not_made_note_still_loads_and_shows(
            self, tmp_path, monkeypatch):
        """A note is data. Projects saved before 0.4.0 carry "Not made · …"
        on their not-scored rows, and that is what those rows said when the
        technician wrote them: it is read back and shown, word for word.
        Only new notes say "Not scored"."""
        opt = self._opt(tmp_path, monkeypatch, name="oldnote")
        opt.tell({"Water": 10.0}, {"Firmness": 5.0}, formulation_no=1,
                 batch_no=1)
        state = opt.export_json()
        state["skipped"] = [{"formulation": 2, "batch": 1,
                             "recipe": {"Water": 20.0},
                             "note": "Not made · burner failed"}]
        older = FoodOptimizer("oldnote2")
        older.import_json(state)
        assert older.skipped[0]["note"] == "Not made · burner failed"
        assert older.history_frame()["Note"].iloc[1] == "Not made · burner failed"

    def test_score_skipped_keeps_the_row_when_nothing_was_measured(
            self, tmp_path, monkeypatch):
        """tell() refuses a row with nothing on it. The not-scored row must
        survive that refusal, or a slip of the hand deletes the formulation."""
        opt = self._opt(tmp_path, monkeypatch, name="scoreskipped2")
        opt.record_skipped(1, 1, {"Water": 10.0}, note="Not scored")
        with pytest.raises(ValueError, match="at least one measurement"):
            opt.score_skipped(1, {"Firmness": None})
        assert [s["formulation"] for s in opt.skipped] == [1]
        assert FoodOptimizer("scoreskipped2").skipped[0]["formulation"] == 1

    def test_score_skipped_stores_the_note_it_is_given(self, tmp_path,
                                                       monkeypatch):
        """The reason it was not scored is reopened on screen and saved back
        as the row's note; a row scored with nothing typed carries none."""
        opt = self._opt(tmp_path, monkeypatch, name="scorednote")
        opt.record_skipped(1, 1, {"Water": 10.0},
                           note="Not scored · burner failed")
        opt.score_skipped(1, {"Firmness": 6.0}, note="burner failed")
        assert opt.notes_history == ["burner failed"]
        assert FoodOptimizer("scorednote").notes_history == ["burner failed"]

    def test_score_skipped_refuses_a_number_it_does_not_hold(self, tmp_path,
                                                             monkeypatch):
        opt = self._opt(tmp_path, monkeypatch, name="scoreskipped3")
        opt.tell({"Water": 10.0}, {"Firmness": 5.0}, formulation_no=1)
        with pytest.raises(ValueError, match="not waiting to be scored"):
            opt.score_skipped(1, {"Firmness": 5.0})

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


    def test_edit_amounts_reencodes_the_row_and_saves(self, tmp_path, monkeypatch):
        """A correction to the amounts is not a note: the model reads
        X_history, so the row has to be encoded again or the project keeps
        scoring a formulation nobody made."""
        opt = self._opt(tmp_path, monkeypatch, name="editamounts")
        opt.add_ingredient("Pea protein", 0, 25)
        opt.tell({"Water": 20.0, "Pea protein": 10.0}, {"Firmness": 5.0},
                 formulation_no=1, batch_no=1)
        before = list(opt.X_history[0])
        opt.edit_amounts(0, {"Water": 20.0, "Pea protein": 12.0})
        assert opt.recipe_history[0] == {"Water": 20.0, "Pea protein": 12.0}
        assert opt.X_history[0] != before
        assert opt.X_history[0] == opt._encode({"Water": 20.0,
                                                "Pea protein": 12.0})
        reloaded = FoodOptimizer("editamounts")
        assert reloaded.recipe_history[0]["Pea protein"] == 12.0
        assert reloaded.X_history[0] == opt.X_history[0]

    def test_edit_amounts_refuses_a_position_that_is_not_there(self, tmp_path,
                                                               monkeypatch):
        opt = self._opt(tmp_path, monkeypatch, name="editamounts2")
        with pytest.raises(IndexError):
            opt.edit_amounts(0, {"Water": 1.0})

    def test_delete_formulations_takes_scored_and_not_scored_rows_in_one_save(
            self, tmp_path, monkeypatch):
        """Three numbers, one of them not scored, and one write to the disk:
        deleting row by row saved four times and, done in the wrong order,
        deleted the wrong rows as the positions shifted."""
        opt = self._opt(tmp_path, monkeypatch, name="delmany")
        for n in (1, 2, 3):
            opt.tell({"Water": 10.0 * n}, {"Firmness": 5.0},
                     formulation_no=n, batch_no=1, note=f"n{n}")
        opt.record_skipped(4, 1, {"Water": 40.0})
        opt.tell({"Water": 50.0}, {"Firmness": 6.0}, formulation_no=5, batch_no=2)
        saves = []
        real_save = opt.save
        opt.save = lambda *a, **k: saves.append(1) or real_save(*a, **k)
        gone = opt.delete_formulations([1, 4, 3])
        assert gone == 3
        assert len(saves) == 1, saves
        opt.save = real_save
        assert opt.formulation_ids == [2, 5]
        assert opt.notes_history == ["n2", ""]
        assert opt.skipped == []
        assert len(opt.X_history) == len(opt.Y_history) == 2
        reloaded = FoodOptimizer("delmany")
        assert reloaded.formulation_ids == [2, 5]

    def test_delete_formulations_ignores_a_number_it_does_not_hold(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch, name="delmany2")
        opt.tell({"Water": 10.0}, {"Firmness": 5.0}, formulation_no=1, batch_no=1)
        assert opt.delete_formulations([9]) == 0
        assert opt.formulation_ids == [1]


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

    def test_the_goal_reads_the_same_way_everywhere(self):
        """C3: one rendering of one cell. It was 'Aim for a target value' beside a
        Target of 6 on the grid, 'Target 6' in the Results table and
        'target 6' in lower case on the sheets the bench reads. A '/10' is
        shown once, on the measurement's own label, so it never follows a
        number: 'Firmness (/10) · Target 7'."""
        from food_bo import goal_line, goal_text
        assert goal_line is goal_text
        assert goal_text({"goal": "target", "target": 6, "unit": "N"}) == "Target 6 N"
        assert goal_text({"goal": "target", "target": 7, "unit": "/10"}) == "Target 7"
        assert goal_text({"goal": "min", "unit": "N"}) == "Prefer lower values"
        assert goal_text({"goal": "max", "unit": ""}) == "Prefer higher values"

    def test_a_slash_unit_is_written_once_on_the_label(self):
        from food_bo import label_with_unit, unit_after_number
        assert label_with_unit("Firmness", "/10") == "Firmness (/10)"
        assert label_with_unit("Firmness", "N") == "Firmness"
        assert label_with_unit("Firmness", "") == "Firmness"
        # ...and where the cell beside the label is EMPTY, the unit has
        # nowhere else to go: a bench handed a bare "Cook loss" writes 18,
        # 0.18 or 18.4 g.
        from food_bo import entry_label
        assert entry_label("Cook loss", "%") == "Cook loss (%)"
        assert entry_label("Firmness", "N") == "Firmness (N)"
        assert entry_label("Firmness", "/10") == "Firmness (/10)"
        assert entry_label("Firmness", "") == "Firmness"
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
        assert opt.history_frame()["Date recorded"].iloc[0] == expected
        assert expected in opt.history_csv()

    def test_the_target_refusal_names_the_range_in_plain_words(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        with pytest.raises(ValueError) as add:
            opt.add_objective("Chew", 1.0, goal="target", target=99,
                              min_val=0, max_val=10)
        assert str(add.value) == ("Target 99 must be between Scale minimum and Scale maximum "
                                  "(0 to 10).")
        with pytest.raises(ValueError) as edit:
            opt.update_objective("Firmness", target=99)
        assert str(edit.value) == ("Target 99 must be between Scale minimum and Scale maximum "
                                   "(0 to 10).")

    def test_a_backwards_range_is_refused_in_the_tab_s_words(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        with pytest.raises(ValueError) as e:
            opt.add_objective("Chew", 1.0, min_val=10, max_val=0)
        assert str(e.value) == ("Scale minimum must be less than "
                                "Scale maximum.")

    def test_the_delete_refusal_names_the_formulations(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.tell({"Pea protein": 10.0, "Methylcellulose": 0.0},
                 {"Firmness": 6.0, "Juiciness": 7.0}, formulation_no=3, batch_no=1)
        with pytest.raises(ValueError) as one:
            opt.remove_ingredient("Pea protein")
        # No programmer quotes round the name, and the tick is quoted back
        # as the checkbox on screen actually reads — it said "Delete even if
        # it was used", which is not what the box says.
        assert str(one.value) == (
            "Pea protein was used in Formulation 3, so it cannot be "
            "deleted. Tick Delete even though formulations used it to "
            "discard that information."
        )
        assert wording.DELETE_EVEN_IF_USED_CHECKBOX.startswith(
            wording.DELETE_EVEN_IF_USED)
        opt.tell({"Pea protein": 12.0, "Methylcellulose": 0.0},
                 {"Firmness": 6.0, "Juiciness": 7.0}, formulation_no=5, batch_no=1)
        with pytest.raises(ValueError) as two:
            opt.remove_ingredient("Pea protein")
        assert "used in Formulations 3 and 5, so it cannot be deleted" in str(two.value)

    def test_measurements_are_ordered_by_importance(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        assert [o["name"] for o in opt.measurements_by_importance()] == ["Firmness", "Juiciness"]

    def test_score_line_carries_the_shares(self, tmp_path, monkeypatch):
        """In the shares the reader typed and nothing else: since 0.5.0 the
        share IS the importance, so the line says the number once and the
        ceiling is 100. No sentence about distance from a target either —
        two of the three goals have none, and how closeness works lives in
        the expander below.

        Read back off the file, which is how every screen reads it."""
        opt = self._opt(tmp_path, monkeypatch)
        saved = FoodOptimizer(opt.project_name)
        assert saved.score_function_line() == (
            "Overall score = 60 % × Firmness closeness + 40 % × "
            "Juiciness closeness. A formulation that hits every goal "
            "scores 100."
        )
        assert saved.utility_ceiling() == 100.0

    def test_share_of_score_sums_to_one_and_reads_as_whole_percent(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        assert opt.share_of_score("Firmness") == pytest.approx(0.6)
        assert opt.share_of_score("Juiciness") == pytest.approx(0.4)
        assert opt.share_text("Firmness") == "60 %"
        assert opt.share_text("Juiciness") == "40 %"

    def test_the_shares_add_up_to_a_hundred(self, tmp_path, monkeypatch):
        """Share of score is a column the reader adds up. Rounding each
        share on its own put "33 %" against three equally important
        measurements, which is 99."""
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("thirds")
        opt.add_ingredient("Water", 0, 100)
        for name in ("Firmness", "Juiciness", "Colour"):
            opt.add_objective(name, 1.0, goal="max", min_val=0, max_val=10)
        shares = [opt.share_text(o['name'])
                  for o in opt.measurements_by_importance()]
        assert shares == ["34 %", "33 %", "33 %"], shares
        assert sum(opt.share_percents().values()) == 100
        # Seven equal measurements: 14.28 % apiece, so six get the point.
        opt2 = FoodOptimizer("sevenths")
        opt2.add_ingredient("Water", 0, 100)
        for i in range(7):
            opt2.add_objective(f"M{i}", 1.0, goal="max", min_val=0, max_val=10)
        assert sum(opt2.share_percents().values()) == 100

    def test_share_of_score_names_the_missing_measurement(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        with pytest.raises(ValueError, match="No measurement named Saltiness"):
            opt.share_of_score("Saltiness")

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
        # "measured", not "scored": the column is headed Measured, and a
        # panel rating is one kind of measurement among several.
        assert rows[1]["measured"] == "not measured"
        assert rows[1]["off_by"] == "not measured"

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
        assert rows[0] == {"name": "Juiciness (/10)", "goal": "Prefer higher values",
                           "measured": "8", "off_by": "—"}
        assert rows[1] == {"name": "Grittiness (/10)", "goal": "Prefer lower values",
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

    def test_the_changes_are_largest_first(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        changes = opt._variable_deltas(
            {"Pea protein": 8.0, "Methylcellulose": 1.8},
            {"Pea protein": 10.1, "Methylcellulose": 1.0}, 'ingredient')
        assert changes[0][0] == "Pea protein"
        assert changes[0][1] == pytest.approx(-2.1)
        assert changes[1][0] == "Methylcellulose"
        assert changes[1][1] == pytest.approx(0.8)

    def test_the_amounts_are_asked_for_without_the_settings(self, tmp_path,
                                                            monkeypatch):
        """A setting is not an amount: a cook temperature reported as
        '+180.00 g' priced an oven in grams, and against a formulation made
        before the setting existed the change was the whole baseline."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_process_parameter("Cook temperature", 160, 200, unit="°C")
        recipe = {"Pea protein": 8.0, "Methylcellulose": 1.8,
                  "Cook temperature": 180.0}
        ref = {"Pea protein": 10.1, "Methylcellulose": 1.0,
               "Cook temperature": 0.0}
        assert [name for name, _ in
                opt._variable_deltas(recipe, ref, 'ingredient')] == \
            ["Pea protein", "Methylcellulose"]
        assert [name for name, _ in
                opt._variable_deltas(recipe, ref, 'process')] == \
            ["Cook temperature"]

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
        assert list(df.columns) == ["Best so far", "Round", "Formulation", "Firmness (N)",
                                    "Juiciness", "Overall score", "Date recorded", "Note"]
        assert list(df["Formulation"]) == [2, 1, 3]        # best first, skipped last
        # Best is a star or nothing; the Note column carries "Not scored".
        assert list(df["Best so far"]) == ["★", "", ""]
        assert list(df["Round"]) == ["1", "1", "1"]        # one type, always
        assert df["Note"].iloc[0] == "best yet"
        assert df["Note"].iloc[2] == "Not scored"
        assert df["Overall score"].iloc[2] == ""

    def test_history_frame_trial_column_is_all_strings_on_a_mixed_project(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.tell({"Pea protein": 10.0, "Methylcellulose": 1.0},
                 {"Firmness": 6.0, "Juiciness": 7.0})            # no batch (imported)
        opt.tell({"Pea protein": 12.0, "Methylcellulose": 1.0},
                 {"Firmness": 6.0, "Juiciness": 7.0}, batch_no=1)
        assert set(opt.history_frame()["Round"]) == {"", "1"}

    def test_history_frame_marks_a_partial_score(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.tell({"Pea protein": 10.0, "Methylcellulose": 1.0}, {"Firmness": 6.0})
        # Named, not "partial": the reader should not have to work out which
        # measurement is missing from a row that has room to say.
        assert opt.history_frame()["Overall score"].iloc[0] == \
            "1.50 · Juiciness not measured"

    def test_history_frame_orders(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.tell({"Pea protein": 10.0, "Methylcellulose": 1.0},
                 {"Firmness": 6.0, "Juiciness": 7.0}, formulation_no=1, batch_no=2)
        opt.tell({"Pea protein": 12.0, "Methylcellulose": 1.0},
                 {"Firmness": 1.0, "Juiciness": 1.0}, formulation_no=2, batch_no=1)
        assert list(opt.history_frame(order="Best first")["Formulation"]) == [1, 2]
        assert list(opt.history_frame(order="Newest first")["Formulation"]) == [2, 1]
        assert list(opt.history_frame(order="Round order")["Formulation"]) == [2, 1]

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
        for col in ("Formulation", "Round", "Date recorded", "Overall score",
                    "Pea protein (g)", "Firmness (N)", "Note"):
            assert col in df.columns, list(df.columns)
        assert df["Formulation"].iloc[0] == 4
        assert df["Round"].iloc[0] == 2

    def test_history_csv_holds_every_formulation_the_project_holds(
            self, tmp_path, monkeypatch):
        """The file is downloaded from the All formulations table, so it says
        what that table says: a row nobody made has a number, its amounts and
        its note, and only its measurements are blank. Leaving it out made the
        file disagree with the screen it came from. Measurements run by
        importance, as they do everywhere else."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.tell({"Pea protein": 10.0, "Methylcellulose": 1.0},
                 {"Firmness": 6.0, "Juiciness": 7.0}, formulation_no=1, batch_no=1)
        opt.tell({"Pea protein": 12.0, "Methylcellulose": 1.0},
                 {"Firmness": 5.0}, formulation_no=2, batch_no=1)   # partial: no Juiciness
        opt.record_skipped(3, 1, {"Pea protein": 14.0, "Methylcellulose": 1.0},
                           note="Not scored · burner failed")
        df = pd.read_csv(io.StringIO(opt.history_csv()))
        assert list(df["Formulation"]) == [1, 2, 3]
        assert df["Firmness (N)"].iloc[1] == 5.0
        assert pd.isna(df["Juiciness"].iloc[1])
        # The left-out row: amounts and note, no measurements, no score.
        assert df["Pea protein (g)"].iloc[2] == 14.0
        assert df["Note"].iloc[2] == "Not scored · burner failed"
        assert pd.isna(df["Firmness (N)"].iloc[2])
        assert pd.isna(df["Juiciness"].iloc[2])
        # Firmness is the more important of the two, so it comes first.
        columns = list(df.columns)
        assert (columns.index("Firmness (N)")
                < columns.index("Juiciness"))

    def test_batch_frame_is_the_make_these_table(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_pending_batch([{"Pea protein": 10.0, "Methylcellulose": 1.0}])
        df = opt.batch_frame(opt.pending_batch)
        assert list(df.columns) == ["Formulation", "Pea protein (g)",
                                    "Methylcellulose (g)", "Total (g)"]
        assert list(df["Formulation"]) == [1]
        assert df["Total (g)"].iloc[0] == pytest.approx(11.0)

    def test_batch_frame_carries_the_scale_through(self, tmp_path, monkeypatch):
        """What the sheets carry must equal what is on screen, or the lab
        weighs out the wrong amounts."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_pending_batch([{"Pea protein": 10.0, "Methylcellulose": 1.0}])
        df = opt.batch_frame(opt.pending_batch, scale_to=22.0)
        assert df["Pea protein (g)"].iloc[0] == pytest.approx(20.0)
        assert df["Total (g)"].iloc[0] == pytest.approx(22.0)

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
        assert "Unit" in df.columns
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
        # The total closes the amounts, and during the cold start nothing
        # follows it.
        assert list(df.columns)[-1] == "Total (g)"
        assert df["Total (g)"].iloc[0] == pytest.approx(11.0)
        assert opt.one_amount_unit() == "g"

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

    def test_the_changes_are_reported_with_each_own_unit(
            self, tmp_path, monkeypatch):
        """_variable_deltas hands back the names; the caller writes each one
        with that ingredient's unit."""
        opt = self._opt(tmp_path, monkeypatch)
        changes = opt._variable_deltas({"Pea protein": 12.0, "Water": 40.0},
                                       {"Pea protein": 10.0, "Water": 30.0},
                                       'ingredient')
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

    def test_a_total_across_two_units_is_written_out(self, tmp_path,
                                                     monkeypatch):
        """Nothing can add up '10.00 g · 40.00 ml' to one number, so the
        screen and the sheets both carry it written out."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_pending_batch([{"Pea protein": 10.0, "Water": 40.0}])
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
        assert opt._variable_deltas(opt.pending_batch[1]['recipe'],
                                    row['recipe'], 'ingredient') == []


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
                           match=r"Formulation 11 is not in round 3 \(it has 7, 8, 9\)\."):
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
            "Check the value, or adjust Scale minimum or Scale maximum in Set up.")
        opt.update_objective("Hardness", unit="/10")
        with pytest.raises(ValueError) as slash:
            opt.parse_batch_results(df, opt.pending_batch)
        assert str(slash.value) == (
            "Formulation 7 Hardness 12 is outside your range of 0 to 10. "
            "Check the value, or adjust Scale minimum or Scale maximum in Set up.")

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



# ------------------------------------------------------------------ #
#  0.5.0 · the two editable grids, at the layer they are written at
#
#  AppTest cannot click a cell, so this is where the diff between an edited
#  frame and the project is pinned: every kind of change, every refusal, and
#  the rule that a refusal writes nothing at all.
# ------------------------------------------------------------------ #

def _edit(frame, row, **cells):
    """Type into one row of a grid, by the number the grid shows."""
    out = frame.copy()
    for column, value in cells.items():
        out.loc[row, column] = value
    return out


def _add(frame, **cells):
    """Type a row on the empty line at the bottom. The hidden identity comes
    back empty, which is what makes it an addition."""
    out = frame.copy()
    row = {c: None for c in out.columns}
    row.update(cells)
    out.loc[len(out) + 1] = row
    return out


def _drop(frame, row):
    return frame.drop(index=row)


def _ing_row(name, kind=None, low=0.0, high=100.0, unit="g", **extra):
    row = {wording.NAME_LABEL: name,
           wording.TYPE_LABEL: kind or wording.KIND_INGREDIENT,
           wording.LOWEST_LABEL: low, wording.HIGHEST_LABEL: high,
           wording.UNIT_LABEL: unit,
           wording.VENDOR_LABEL: "", wording.SKU_LABEL: ""}
    row.update(extra)
    return row


def _meas_row(name, goal="max", low=0.0, high=10.0, unit="", share=50.0,
              target=None):
    return {wording.MEASUREMENT_COLUMN: name,
            wording.GOAL_LABEL: wording.GOAL_LABELS[goal],
            wording.TARGET_LABEL: target,
            wording.LOWEST_MEASURABLE_LABEL: low,
            wording.HIGHEST_MEASURABLE_LABEL: high,
            wording.UNIT_LABEL: unit, wording.SHARE_COLUMN: share}


def _said(messages, kind=None):
    return [text for k, text in messages if kind is None or k == kind]


class TestTheIngredientsGrid:

    def _opt(self, tmp_path, monkeypatch, name="grid"):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer(name)
        opt.set_amount_unit("g")
        opt.add_ingredient("Water", 0, 100)
        opt.add_ingredient("Salt", 0, 10)
        opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
        return opt

    # ---- what the grid opens holding -------------------------------- #

    def test_the_frame_is_the_project_with_a_hidden_identity(self, tmp_path,
                                                             monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        frame = opt.ingredient_grid_frame()
        # Made as sits right after Type: it says what the row IS, which is
        # the question Type half answers, and a pre-mix's parts open in a
        # fold underneath rather than in a column of their own.
        assert list(frame.columns) == [
            "_id", wording.NAME_LABEL, wording.TYPE_LABEL,
            wording.MADE_AS_LABEL,
            wording.LOWEST_LABEL, wording.HIGHEST_LABEL, wording.UNIT_LABEL,
            wording.FORMULA_LABEL]
        # The ordinary case has a word of its own: a blank cell is not an
        # answer a reader can recognise as the one they want.
        assert list(frame[wording.MADE_AS_LABEL]) == \
            [wording.PREMIX_MADE_AS_BOUGHT_IN] * 2
        # Numbered from 1, so "Row 2" under the grid is the second row the
        # reader can see.
        assert list(frame.index) == [1, 2]
        assert list(frame["_id"]) == ["Water", "Salt"]

    def test_baseline_arrives_with_the_first_result(self, tmp_path,
                                                    monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        assert wording.BASELINE_LABEL not in opt.ingredient_grid_frame()
        opt.tell({"Water": 50.0, "Salt": 5.0}, {"Taste": 7.0})
        assert wording.BASELINE_LABEL in opt.ingredient_grid_frame()

    # ---- one diff kind at a time ------------------------------------ #

    def test_an_added_row(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        errors, messages = opt.apply_ingredient_grid(
            _add(opt.ingredient_grid_frame(), **_ing_row("Oil", high=20.0)))
        assert errors == []
        assert _said(messages, "success") == ["Oil added."]
        assert opt._var_by_name("Oil")["bounds"] == (0.0, 20.0)

    def test_a_deleted_row(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        frame = opt.ingredient_grid_frame()
        assert opt.ingredient_grid_deletions(_drop(frame, 2)) == ["Salt"]
        errors, messages = opt.apply_ingredient_grid(_drop(frame, 2))
        assert errors == []
        assert _said(messages, "success") == ["Salt deleted."]
        assert [v["name"] for v in opt.variables] == ["Water"]

    def test_a_renamed_row_keeps_everything_filed_under_it(self, tmp_path,
                                                           monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.tell({"Water": 50.0, "Salt": 5.0}, {"Taste": 7.0})
        opt.add_quantity_constraint(["Salt"], max_val=8)
        errors, messages = opt.apply_ingredient_grid(
            _edit(opt.ingredient_grid_frame(), 2,
                  **{wording.NAME_LABEL: "Sea salt"}))
        assert errors == []
        assert _said(messages, "success") == ["Sea salt saved."]
        assert opt.recipe_history[0] == {"Water": 50.0, "Sea salt": 5.0}
        assert opt.quantity_constraints[0]["ingredients"] == ["Sea salt"]

    def test_changed_bounds(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        errors, _ = opt.apply_ingredient_grid(
            _edit(opt.ingredient_grid_frame(), 1,
                  **{wording.HIGHEST_LABEL: 60.0}))
        assert errors == []
        assert opt._var_by_name("Water")["bounds"] == (0.0, 60.0)

    def test_a_changed_unit_says_nothing_was_converted(self, tmp_path,
                                                       monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        errors, messages = opt.apply_ingredient_grid(
            _edit(opt.ingredient_grid_frame(), 1,
                  **{wording.UNIT_LABEL: "ml"}))
        assert errors == []
        assert wording.unit_changed("Water", "ml", True) in _said(messages)
        assert opt.unit_of("Water") == "ml"

    def test_a_changed_type(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        errors, _ = opt.apply_ingredient_grid(
            _edit(opt.ingredient_grid_frame(), 2,
                  **{wording.TYPE_LABEL: wording.KIND_SETTING,
                     wording.UNIT_LABEL: "°C"}))
        assert errors == []
        assert opt._var_by_name("Salt")["category"] == "process"

    def test_a_type_change_is_refused_once_results_exist(self, tmp_path,
                                                         monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.tell({"Water": 50.0, "Salt": 5.0}, {"Taste": 7.0})
        errors, _ = opt.apply_ingredient_grid(
            _edit(opt.ingredient_grid_frame(), 2,
                  **{wording.TYPE_LABEL: wording.KIND_SETTING}))
        assert errors == [(2, wording.TYPE_LOCKED_ERROR)]
        assert opt._var_by_name("Salt")["category"] == "ingredient"

    def test_vendor_and_sku(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        errors, _ = opt.apply_ingredient_grid(
            _edit(opt.ingredient_grid_frame(), 1,
                  **{wording.VENDOR_LABEL: "Acme",
                     wording.SKU_LABEL: "H2O-1"}))
        assert errors == []
        var = opt._var_by_name("Water")
        assert (var["vendor"], var["sku"]) == ("Acme", "H2O-1")

    def test_a_fixed_row_is_lowest_equals_highest(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        errors, _ = opt.apply_ingredient_grid(
            _edit(opt.ingredient_grid_frame(), 2,
                  **{wording.LOWEST_LABEL: 4.0, wording.HIGHEST_LABEL: 4.0}))
        assert errors == []
        assert [v["name"] for v in opt.fixed_variables()] == ["Salt"]

    def test_four_changes_in_one_save(self, tmp_path, monkeypatch):
        """The point of a grid: one Save, and every consequence said once."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_ingredient("Oil", 0, 20)
        frame = opt.ingredient_grid_frame()
        frame = _edit(frame, 1, **{wording.HIGHEST_LABEL: 60.0})
        frame = _edit(frame, 2, **{wording.NAME_LABEL: "Sea salt"})
        frame = _add(frame, **_ing_row("Sugar", high=5.0))
        frame = _drop(frame, 3)
        errors, messages = opt.apply_ingredient_grid(frame)
        assert errors == []
        assert _said(messages, "success") == [
            "Sugar added.", "Water and Sea salt saved.", "Oil deleted."]
        assert [v["name"] for v in opt.variables] == ["Water", "Sea salt",
                                                      "Sugar"]

    # ---- every refusal, and nothing written -------------------------- #

    def _refused(self, opt, frame):
        before = json.dumps(opt.export_json(), sort_keys=True, default=str)
        errors, messages = opt.apply_ingredient_grid(frame)
        assert messages == []
        after = json.dumps(opt.export_json(), sort_keys=True, default=str)
        assert before == after, "a refused grid wrote something"
        return errors

    def test_a_taken_name_is_refused_by_row(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        assert self._refused(opt, _edit(opt.ingredient_grid_frame(), 2,
                                        **{wording.NAME_LABEL: "Water"})) == [
            (2, "Water is already the name of an ingredient. Choose another "
                "name.")]

    def test_a_name_that_differs_only_by_case_is_refused(self, tmp_path,
                                                         monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        assert self._refused(opt, _edit(opt.ingredient_grid_frame(), 2,
                                        **{wording.NAME_LABEL: "water"})) == [
            (2, wording.name_differs_only_by_case("Water"))]

    def test_a_measurements_name_is_refused(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        assert self._refused(opt, _edit(opt.ingredient_grid_frame(), 2,
                                        **{wording.NAME_LABEL: "Taste"})) == [
            (2, "Taste is already the name of a measurement. Choose another "
                "name.")]

    def test_a_reserved_name_is_refused(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        errors = self._refused(opt, _edit(opt.ingredient_grid_frame(), 2,
                                          **{wording.NAME_LABEL: "Total"}))
        assert errors[0][0] == 2 and "column name" in errors[0][1]

    def test_an_empty_name_is_refused(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        assert self._refused(opt, _edit(opt.ingredient_grid_frame(), 2,
                                        **{wording.NAME_LABEL: "  "})) == [
            (2, wording.NAME_REQUIRED_ERROR)]

    def test_lowest_above_highest_is_refused(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        assert self._refused(opt, _edit(opt.ingredient_grid_frame(), 1,
                                        **{wording.LOWEST_LABEL: 90.0,
                                           wording.HIGHEST_LABEL: 10.0})) == [
            (1, "Lowest cannot be above Highest.")]

    def test_a_number_that_is_not_a_number_is_refused(self, tmp_path,
                                                      monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        assert self._refused(opt, _edit(opt.ingredient_grid_frame(), 1,
                                        **{wording.HIGHEST_LABEL: "lots"})) == [
            (1, wording.NUMBER_REQUIRED_ERROR)]

    def test_a_blank_unit_is_refused_for_an_ingredient_only(self, tmp_path,
                                                            monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        assert self._refused(opt, _edit(opt.ingredient_grid_frame(), 1,
                                        **{wording.UNIT_LABEL: ""})) == [
            (1, wording.UNIT_REQUIRED_ERROR)]
        opt.add_process_parameter("Mixer speed", 1, 5, unit="rpm")
        errors, _ = opt.apply_ingredient_grid(
            _edit(opt.ingredient_grid_frame(), 3,
                  **{wording.UNIT_LABEL: ""}))
        assert errors == []
        assert opt.unit_of("Mixer speed") == ""

    def test_a_cell_that_does_not_belong_to_the_row_is_refused(self, tmp_path,
                                                               monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_process_parameter("Mixer speed", 1, 5, unit="rpm")
        assert self._refused(opt, _edit(opt.ingredient_grid_frame(), 3,
                                        **{wording.VENDOR_LABEL: "Acme"})) == [
            (3, wording.only_an_ingredient_has(wording.VENDOR_LABEL))]
        opt.tell({"Water": 50.0, "Salt": 5.0, "Mixer speed": 3.0},
                 {"Taste": 7.0})
        assert self._refused(opt, _edit(opt.ingredient_grid_frame(), 1,
                                        **{wording.BASELINE_LABEL: 2.0})) == [
            (1, wording.only_a_setting_has(wording.BASELINE_LABEL))]

    def test_every_row_is_checked_before_any_is_written(self, tmp_path,
                                                        monkeypatch):
        """Two bad rows come back as two lines, not one — the reader fixes
        the grid once rather than saving into a queue of refusals."""
        opt = self._opt(tmp_path, monkeypatch)
        frame = _edit(opt.ingredient_grid_frame(), 1,
                      **{wording.NAME_LABEL: ""})
        frame = _edit(frame, 2, **{wording.LOWEST_LABEL: 90.0,
                                   wording.HIGHEST_LABEL: 10.0})
        assert self._refused(opt, frame) == [
            (1, wording.NAME_REQUIRED_ERROR),
            (2, "Lowest cannot be above Highest.")]

    def test_a_deletion_of_a_used_ingredient_is_refused_before_it_writes(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.tell({"Water": 50.0, "Salt": 5.0}, {"Taste": 7.0})
        errors = self._refused(opt, _drop(opt.ingredient_grid_frame(), 2))
        assert errors[0][0] is None and "was used in Formulation 1" in errors[0][1]
        # ...and goes through once the reader says so.
        errors, messages = opt.apply_ingredient_grid(
            _drop(opt.ingredient_grid_frame(), 2), force={"Salt"})
        assert errors == []
        assert [v["name"] for v in opt.variables] == ["Water"]

    def test_a_blank_row_is_not_an_addition(self, tmp_path, monkeypatch):
        """The empty line at the bottom of a dynamic grid, clicked and left
        alone. Not a row, and not an error either."""
        opt = self._opt(tmp_path, monkeypatch)
        errors, messages = opt.apply_ingredient_grid(
            _add(opt.ingredient_grid_frame()))
        assert (errors, messages) == ([], [])
        assert len(opt.variables) == 2

    # ---- the fixed check, once, over the finished grid --------------- #

    def test_fixing_everything_below_the_batch_size_is_refused_over_the_grid(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_formulation_total(50.0)
        frame = _edit(opt.ingredient_grid_frame(), 1,
                      **{wording.LOWEST_LABEL: 10.0,
                         wording.HIGHEST_LABEL: 10.0})
        frame = _edit(frame, 2, **{wording.LOWEST_LABEL: 2.0,
                                   wording.HIGHEST_LABEL: 2.0})
        errors = self._refused(opt, frame)
        assert len(errors) == 1
        row, message = errors[0]
        assert row is None, "the refusal is about the grid, not a row"
        assert wording.fixing_breaks_the_total("50 g") in message
        assert wording.fixed_rows_tail("Water and Salt") in message

    def test_the_refusal_names_the_row_this_save_changed_and_no_other(
            self, tmp_path, monkeypatch):
        """"…let enough ingredients vary again. Fixed at one amount:
        Seasoning blend" was the answer to giving ANOTHER row a calculation.
        A reader sent to a cell they did not touch cannot act on it."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_formulation_total(50.0)
        # Salt was pinned last week, and it is not what changed today.
        frame = _edit(opt.ingredient_grid_frame(), 2,
                      **{wording.LOWEST_LABEL: 2.0,
                         wording.HIGHEST_LABEL: 2.0})
        errors, _ = opt.apply_ingredient_grid(frame)
        assert errors == []
        frame = _edit(opt.ingredient_grid_frame(), 1,
                      **{wording.LOWEST_LABEL: 10.0,
                         wording.HIGHEST_LABEL: 10.0})
        errors = self._refused(opt, frame)
        assert len(errors) == 1
        message = errors[0][1]
        assert wording.fixing_breaks_the_total("50 g") in message
        assert wording.fixed_rows_tail("Water") in message
        assert "Salt" not in message, message

    def test_one_row_fixed_mid_grid_is_never_asked_on_its_own(self, tmp_path,
                                                              monkeypatch):
        """Fixing Water at 48 g and widening Salt to 2 g in one save reaches
        50 g. Asked per row, the first half would have been refused for a
        state the project never sits in."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_formulation_total(50.0)
        frame = _edit(opt.ingredient_grid_frame(), 1,
                      **{wording.LOWEST_LABEL: 48.0,
                         wording.HIGHEST_LABEL: 48.0})
        frame = _edit(frame, 2, **{wording.LOWEST_LABEL: 2.0,
                                   wording.HIGHEST_LABEL: 2.0})
        errors, _ = opt.apply_ingredient_grid(frame)
        assert errors == []
        assert opt.formulation_total == 50.0

    def test_a_limit_already_impossible_is_not_blamed_on_this_save(
            self, tmp_path, monkeypatch):
        """A range narrowed before this edit can strand a limit, and always
        could. Refusing the next save for it would leave no way back out."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_quantity_constraint(["Water"], min_val=200)
        errors, _ = opt.apply_ingredient_grid(
            _edit(opt.ingredient_grid_frame(), 2,
                  **{wording.LOWEST_LABEL: 3.0, wording.HIGHEST_LABEL: 3.0}))
        assert errors == []
        assert opt._var_by_name("Salt")["bounds"] == (3.0, 3.0)

    def test_a_stranded_limit_does_not_wave_through_the_one_this_save_breaks(
            self, tmp_path, monkeypatch):
        """Fix round 1. The comparison is limit by limit, not "was anything
        broken before": one limit already stranded used to turn the whole
        check off, and the save that broke a second went through."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_quantity_constraint(["Water"], min_val=200)   # already stranded
        opt.add_quantity_constraint(["Salt"], max_val=10.0)   # this one is fine
        errors = self._refused(opt, _edit(opt.ingredient_grid_frame(), 2,
                                          **{wording.LOWEST_LABEL: 20.0,
                                             wording.HIGHEST_LABEL: 20.0}))
        assert len(errors) == 1
        row, message = errors[0]
        assert row is None
        assert "limit on Salt" in message and "Water" not in message
        assert message.endswith(wording.fixed_rows_tail("Salt"))

    # ---- consequences, once ----------------------------------------- #

    def test_the_open_round_is_discarded_once(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_pending_batch([{"Water": 50.0, "Salt": 5.0}], batch_no=1)
        frame = _edit(opt.ingredient_grid_frame(), 1,
                      **{wording.HIGHEST_LABEL: 60.0})
        frame = _edit(frame, 2, **{wording.HIGHEST_LABEL: 20.0})
        errors, messages = opt.apply_ingredient_grid(frame)
        assert errors == []
        assert _said(messages, "info") == [
            wording.batch_discarded_notice(1)]
        assert opt.pending_batch is None

    def test_a_pruned_limit_is_named_once(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_quantity_constraint(["Water", "Salt"], max_val=50)
        errors, messages = opt.apply_ingredient_grid(
            _edit(opt.ingredient_grid_frame(), 2,
                  **{wording.UNIT_LABEL: "ml"}))
        assert errors == []
        assert len(_said(messages, "warning")) == 1
        assert opt.quantity_constraints == []

    def test_the_default_batch_size_is_kept_and_said_once(self, tmp_path,
                                                          monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_formulation_total(50.0)
        errors, messages = opt.apply_ingredient_grid(
            _add(opt.ingredient_grid_frame(), **_ing_row("Oil", high=20.0)))
        assert errors == []
        assert _said(messages, "success") == [
            "Oil added. " + wording.total_still_holds("50 g")]

    def test_a_new_ingredient_mid_run_says_what_the_records_contain(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.tell({"Water": 50.0, "Salt": 5.0}, {"Taste": 7.0})
        errors, messages = opt.apply_ingredient_grid(
            _add(opt.ingredient_grid_frame(),
                 **_ing_row("Sugar", low=5.0, high=5.0)))
        assert errors == []
        assert _said(messages, "info") == [
            wording.formulations_contain_none_of("Sugar")]
        # 0.5.0: a new row may start above 0 — the old form forced it there.
        assert opt._var_by_name("Sugar")["bounds"] == (5.0, 5.0)

    def test_a_grid_that_says_what_the_project_says_writes_nothing(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_pending_batch([{"Water": 50.0, "Salt": 5.0}], batch_no=1)
        errors, messages = opt.apply_ingredient_grid(
            opt.ingredient_grid_frame())
        assert (errors, messages) == ([], [])
        assert opt.pending_batch is not None

    def test_a_rename_on_its_own_keeps_the_open_round(self, tmp_path,
                                                      monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_pending_batch([{"Water": 50.0, "Salt": 5.0}], batch_no=1)
        errors, messages = opt.apply_ingredient_grid(
            _edit(opt.ingredient_grid_frame(), 2,
                  **{wording.NAME_LABEL: "Sea salt"}))
        assert errors == [] and _said(messages, "info") == []
        assert opt.pending_batch[0]["recipe"] == {"Water": 50.0,
                                                  "Sea salt": 5.0}


class TestTheMeasurementsGrid:

    def _opt(self, tmp_path, monkeypatch, name="mgrid"):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer(name)
        opt.set_amount_unit("g")
        opt.add_ingredient("Water", 0, 100)
        opt.add_objective("Firmness", 1.5, goal="target", target=6,
                          min_val=0, max_val=10, unit="N")
        opt.add_objective("Juiciness", 1.0, goal="max", min_val=0, max_val=10)
        return opt

    def test_the_frame_is_the_measurements_with_their_shares(self, tmp_path,
                                                             monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        frame = opt.measurement_grid_frame()
        assert list(frame.columns) == [
            "_id", wording.MEASUREMENT_COLUMN, wording.GOAL_LABEL,
            wording.TARGET_LABEL, wording.LOWEST_MEASURABLE_LABEL,
            wording.HIGHEST_MEASURABLE_LABEL, wording.UNIT_LABEL,
            wording.SHARE_COLUMN]
        assert list(frame[wording.SHARE_COLUMN]) == [60.0, 40.0]

    def test_a_typed_share_is_kept_and_the_rest_give_way(self, tmp_path,
                                                         monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        errors, messages = opt.apply_measurement_grid(
            _edit(opt.measurement_grid_frame(), 1,
                  **{wording.SHARE_COLUMN: 80.0}))
        assert errors == []
        assert opt.share_percents() == {"Firmness": 80, "Juiciness": 20}
        # The row that gave way is named, with what it gave way TO: the
        # reader typed one number and watched another move.
        assert _said(messages, "info") == [
            "Juiciness adjusted to 20 % so the shares add up to 100 %."], \
            _said(messages, "info")

    def test_three_shares_give_way_proportionally(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_objective("Colour", 1.0, goal="max", min_val=0, max_val=10)
        # 43 / 28.5 / 28.5 before; Firmness moves to 50 and the other two
        # share the remaining 50 in the ratio they already had.
        errors, _ = opt.apply_measurement_grid(
            _edit(opt.measurement_grid_frame(), 1,
                  **{wording.SHARE_COLUMN: 50.0}))
        assert errors == []
        shares = opt.share_percents()
        assert shares["Firmness"] == 50
        assert shares["Juiciness"] == shares["Colour"] == 25

    def test_a_column_that_already_adds_up_says_nothing(self, tmp_path,
                                                        monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        errors, messages = opt.apply_measurement_grid(
            _edit(opt.measurement_grid_frame(), 1,
                  **{wording.HIGHEST_MEASURABLE_LABEL: 20.0}))
        assert errors == []
        assert _said(messages, "info") == []

    def test_a_share_that_is_not_a_number_is_refused_by_row(self, tmp_path,
                                                            monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        errors, messages = opt.apply_measurement_grid(
            _edit(opt.measurement_grid_frame(), 2,
                  **{wording.SHARE_COLUMN: "half"}))
        assert errors == [(2, wording.SHARE_REQUIRED_ERROR)]
        assert messages == []
        assert opt.share_percents() == {"Firmness": 60, "Juiciness": 40}

    def test_a_share_of_nought_is_refused(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        errors, _ = opt.apply_measurement_grid(
            _edit(opt.measurement_grid_frame(), 2,
                  **{wording.SHARE_COLUMN: 0.0}))
        assert errors == [(2, wording.SHARE_REQUIRED_ERROR)]

    def test_a_renamed_measurement_keeps_every_result_filed_under_it(
            self, tmp_path, monkeypatch):
        """The name is a KEY: every row of results_history is a dict filed
        under it. rename_objective moves the two together, so nothing is
        recalculated and no copy is kept."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.tell({"Water": 50.0}, {"Firmness": 6.0, "Juiciness": 7.0})
        opt.set_shares(opt.share_percents())     # as every screen sees it
        before = list(opt.Y_history)
        errors, messages = opt.apply_measurement_grid(
            _edit(opt.measurement_grid_frame(), 1,
                  **{wording.MEASUREMENT_COLUMN: "Bite"}))
        assert errors == []
        assert [o["name"] for o in opt.objectives] == ["Bite", "Juiciness"]
        assert opt.results_history[0] == {"Bite": 6.0, "Juiciness": 7.0}
        assert opt.Y_history == before
        assert _said(messages, "success") == ["Bite saved."]
        # Where the targets came from is a note about the project, not about
        # this measurement.
        assert opt.targets_source == ""

    def test_a_rename_onto_a_name_that_is_taken_is_refused(self, tmp_path,
                                                           monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        errors, messages = opt.apply_measurement_grid(
            _edit(opt.measurement_grid_frame(), 1,
                  **{wording.MEASUREMENT_COLUMN: "Water"}))
        assert errors == [
            (1, "Water is already the name of an ingredient. Choose another "
                "name.")]
        assert messages == []
        assert [o["name"] for o in opt.objectives] == ["Firmness", "Juiciness"]

    def test_a_target_outside_the_range_is_refused(self, tmp_path,
                                                   monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        errors, _ = opt.apply_measurement_grid(
            _edit(opt.measurement_grid_frame(), 1,
                  **{wording.TARGET_LABEL: 50.0}))
        assert errors[0][0] == 1
        assert "must be between Scale minimum" in errors[0][1]

    def test_an_added_measurement_takes_its_share_from_the_rest(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        errors, messages = opt.apply_measurement_grid(
            _add(opt.measurement_grid_frame(),
                 **_meas_row("Colour", share=20.0)))
        assert errors == []
        assert _said(messages, "success")[0] == "Colour added."
        assert opt.share_percents() == {"Firmness": 48, "Juiciness": 32,
                                        "Colour": 20}

    def test_an_added_measurement_leaves_an_even_split_even(
            self, tmp_path, monkeypatch):
        """Two rows at 50 % each and a new row at 20: the two that were
        equal stay equal. The rest give way against what the grid was
        SHOWING before the save, not against a column already carrying the
        new row's placeholder weight — which read 41 / 39 / 20, two
        identical rows separating under the reader's hands."""
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("even_split")
        opt.set_amount_unit("g")
        opt.add_ingredient("Water", 0, 100)
        opt.add_objective("A", 1.0, goal="max", min_val=0, max_val=10)
        opt.add_objective("B", 1.0, goal="max", min_val=0, max_val=10)
        assert opt.share_percents() == {"A": 50, "B": 50}
        errors, _ = opt.apply_measurement_grid(
            _add(opt.measurement_grid_frame(), **_meas_row("C", share=20.0)))
        assert errors == []
        assert opt.share_percents() == {"A": 40, "B": 40, "C": 20}

    def test_a_deleted_measurement_recalculates_and_rebalances(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.tell({"Water": 50.0}, {"Firmness": 6.0, "Juiciness": 7.0})
        errors, messages = opt.apply_measurement_grid(
            _drop(opt.measurement_grid_frame(), 2))
        assert errors == []
        assert [o["name"] for o in opt.objectives] == ["Firmness"]
        assert opt.share_percents() == {"Firmness": 100}
        assert any(wording.RECALCULATED_SUFFIX.strip() in line
                   for line in _said(messages, "success"))

    def test_a_unit_on_its_own_keeps_no_copy(self, tmp_path, monkeypatch):
        """A copy is kept before a save that takes something away. A unit
        corrected on one row takes nothing away, and a copy per keystroke is
        a copy of nothing."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.tell({"Water": 50.0}, {"Firmness": 6.0, "Juiciness": 7.0})
        frame = _edit(opt.measurement_grid_frame(), 1,
                      **{wording.UNIT_LABEL: "kPa"})
        copies = []
        errors, messages = opt.apply_measurement_grid(
            frame, archive=lambda: copies.append(True))
        assert errors == [] and copies == []
        assert _said(messages, "success") == ["Firmness saved."]

    def test_a_range_moved_keeps_a_copy_first(self, tmp_path, monkeypatch):
        """Asked once per save, and before the first write: a copy kept
        afterwards is a copy of nothing."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.tell({"Water": 50.0}, {"Firmness": 6.0, "Juiciness": 7.0})
        frame = _edit(opt.measurement_grid_frame(), 1,
                      **{wording.HIGHEST_MEASURABLE_LABEL: 20.0})
        seen = []
        opt.apply_measurement_grid(
            frame, archive=lambda: seen.append(opt.objectives[0]["max_val"]))
        assert seen == [10.0], seen

    def test_a_refused_grid_never_reaches_the_copy(self, tmp_path,
                                                   monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.tell({"Water": 50.0}, {"Firmness": 6.0, "Juiciness": 7.0})
        copies = []
        errors, _ = opt.apply_measurement_grid(
            _edit(opt.measurement_grid_frame(), 2,
                  **{wording.SHARE_COLUMN: 0.0}),
            archive=lambda: copies.append(True))
        assert errors and copies == []


class TestSharesAreTheWeights:
    """0.5.0 ruling: what a measurement is worth IS its share of the score.
    The shares add up to 100, so the ceiling is 100 and every score is read
    against it — one number on screen, and it is the one the reader typed."""

    def _opt(self, tmp_path, monkeypatch, name="shares"):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer(name)
        opt.set_amount_unit("g")
        opt.add_ingredient("Water", 0, 100)
        opt.add_objective("Firmness", 1.5, goal="target", target=6,
                          min_val=0, max_val=10, unit="N")
        opt.add_objective("Juiciness", 1.0, goal="max", min_val=0, max_val=10)
        return opt

    def test_an_older_project_is_rescaled_the_moment_it_opens(
            self, tmp_path, monkeypatch):
        """1.5 and 1 out of 2.50 become 60 and 40 out of 100. A change of
        units: the ORDER of the formulations does not move, and each score
        moves by the one factor."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.tell({"Water": 10.0}, {"Firmness": 6.0, "Juiciness": 2.0})
        opt.tell({"Water": 20.0}, {"Firmness": 3.0, "Juiciness": 9.0})
        was = list(opt.Y_history)
        assert opt.utility_ceiling() == pytest.approx(2.5)
        saved = FoodOptimizer(opt.project_name)
        assert saved.utility_ceiling() == 100.0
        assert saved.share_percents() == {"Firmness": 60, "Juiciness": 40}
        assert saved.Y_history == pytest.approx([y * 40.0 for y in was])
        assert saved.best_index() == opt.best_index()

    def test_a_project_already_at_a_hundred_is_left_alone(self, tmp_path,
                                                          monkeypatch):
        """Idempotent, so opening a file twice does not drift it — and its
        mtime is not bumped for nothing."""
        opt = self._opt(tmp_path, monkeypatch)
        once = FoodOptimizer(opt.project_name)
        assert once.utility_ceiling() == 100.0
        weights = [o['weight'] for o in once.objectives]
        once.import_json(once.export_json())
        assert [o['weight'] for o in once.objectives] == weights

    def test_the_score_is_read_against_a_hundred(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.tell({"Water": 10.0}, {"Firmness": 6.0, "Juiciness": 10.0})
        saved = FoodOptimizer(opt.project_name)
        assert saved.Y_history[0] == pytest.approx(100.0)
        assert wording.overall_score_caption(
            saved.Y_history[0], saved.utility_ceiling()).startswith(
            "Overall score 100.00 of 100")

    def test_a_measurement_deleted_leaves_the_rest_at_a_hundred(
            self, tmp_path, monkeypatch):
        opt = FoodOptimizer("shares_del")
        monkeypatch.chdir(tmp_path)
        opt = self._opt(tmp_path, monkeypatch, name="shares_del")
        opt.remove_objective("Juiciness")
        assert opt.share_percents() == {"Firmness": 100}
        assert opt.utility_ceiling() == 100.0

    def test_set_shares_says_whether_it_had_to_move_anything(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch, name="shares_ret")
        assert opt.set_shares({"Firmness": 70, "Juiciness": 30}) is False
        assert opt.share_percents() == {"Firmness": 70, "Juiciness": 30}
        # A column that does not add up is scaled, and says so.
        assert opt.set_shares({"Firmness": 30, "Juiciness": 30}) is True
        assert opt.share_percents() == {"Firmness": 50, "Juiciness": 50}


class TestRenamingAMeasurement:

    def _opt(self, tmp_path, monkeypatch, name="mrename"):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer(name)
        opt.set_amount_unit("g")
        opt.add_ingredient("Water", 0, 100)
        opt.add_objective("Firmness", 1.0, goal="max", min_val=0, max_val=10,
                          unit="N")
        opt.add_property("Cost")
        opt.set_targets_source("A trained panel")
        opt.tell({"Water": 50.0}, {"Firmness": 6.0})
        return opt

    def test_the_results_move_with_the_name(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        before = list(opt.Y_history)
        opt.rename_objective("Firmness", "Bite")
        assert [o['name'] for o in opt.objectives] == ["Bite"]
        assert opt.results_history[0] == {"Bite": 6.0}
        assert opt.Y_history == before          # nothing is rescored
        assert opt.targets_source == "A trained panel"

    def test_a_name_that_is_taken_is_refused_before_anything_moves(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch, name="mrename_taken")
        for taken, why in (("Water", "already the name of an ingredient"),
                           ("Cost", "already a property"),
                           ("Total", "column name")):
            with pytest.raises(ValueError, match=why):
                opt.rename_objective("Firmness", taken)
        with pytest.raises(ValueError, match="cannot be empty"):
            opt.rename_objective("Firmness", "   ")
        assert [o['name'] for o in opt.objectives] == ["Firmness"]
        assert opt.results_history[0] == {"Firmness": 6.0}

    def test_renaming_to_its_own_name_is_a_no_op(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch, name="mrename_same")
        stamp = opt.last_saved_at
        opt.rename_objective("Firmness", "Firmness")
        assert opt.last_saved_at == stamp


class TestTheGridFixesRoundOne:
    """The five findings of fix round 1, each at the layer it lives at."""

    def _opt(self, tmp_path, monkeypatch, name="round1"):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer(name)
        opt.set_amount_unit("g")
        opt.add_ingredient("Flour", 0, 100)
        opt.add_ingredient("Water", 0, 80)
        opt.add_ingredient("Salt", 0, 10)
        opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
        return opt

    # ---- the row that moved is the row that is blamed ---------------- #

    def test_a_clash_is_blamed_on_the_row_that_moved(self, tmp_path,
                                                     monkeypatch):
        """Row 2 is typed over to read Flour; row 1 has said Flour all
        along. Blaming row 1 asks the reader to fix a row they never
        touched."""
        opt = self._opt(tmp_path, monkeypatch)
        errors, _ = opt.apply_ingredient_grid(
            _edit(opt.ingredient_grid_frame(), 2,
                  **{wording.NAME_LABEL: "Flour"}))
        assert errors == [(2, _name_taken_message("Flour", 'ingredient'))]

    def test_a_clash_is_blamed_on_the_row_that_moved_whichever_way_it_reads(
            self, tmp_path, monkeypatch):
        """...and the same when the row that moved is the FIRST one: row 1
        is typed over to read Salt, row 3 has said Salt all along."""
        opt = self._opt(tmp_path, monkeypatch, name="round1_first")
        errors, _ = opt.apply_ingredient_grid(
            _edit(opt.ingredient_grid_frame(), 1,
                  **{wording.NAME_LABEL: "Salt"}))
        assert errors == [(1, _name_taken_message("Salt", 'ingredient'))]

    def test_a_case_only_clash_is_blamed_the_same_way(self, tmp_path,
                                                      monkeypatch):
        opt = self._opt(tmp_path, monkeypatch, name="round1_case")
        errors, _ = opt.apply_ingredient_grid(
            _edit(opt.ingredient_grid_frame(), 1,
                  **{wording.NAME_LABEL: "salt"}))
        assert errors == [(1, wording.name_differs_only_by_case("Salt"))]

    def test_two_new_rows_of_one_name_blame_the_later(self, tmp_path,
                                                      monkeypatch):
        opt = self._opt(tmp_path, monkeypatch, name="round1_new")
        frame = _add(opt.ingredient_grid_frame(), **_ing_row("Oil", high=5.0))
        frame = _add(frame, **_ing_row("Oil", high=6.0))
        errors, _ = opt.apply_ingredient_grid(frame)
        assert errors == [(5, _name_taken_message("Oil", 'ingredient'))]

    # ---- a name freed in the same save is free ----------------------- #

    def test_a_name_renamed_away_can_be_taken_in_the_same_save(
            self, tmp_path, monkeypatch):
        """Flour becomes Barley and Water becomes Flour. Both at once is one
        edit the reader can make in a grid, and the write order frees the
        name before it is taken."""
        opt = self._opt(tmp_path, monkeypatch, name="round1_free")
        opt.tell({"Flour": 40.0, "Water": 30.0, "Salt": 1.0}, {"Taste": 7.0})
        frame = _edit(opt.ingredient_grid_frame(), 1,
                      **{wording.NAME_LABEL: "Barley"})
        frame = _edit(frame, 2, **{wording.NAME_LABEL: "Flour"})
        errors, messages = opt.apply_ingredient_grid(frame)
        assert errors == []
        assert [v['name'] for v in opt.variables] == ["Barley", "Flour", "Salt"]
        # Everything filed under each name moved with it, and did not
        # collide on the way.
        assert opt.recipe_history[0] == {"Barley": 40.0, "Flour": 30.0,
                                         "Salt": 1.0}

    def test_a_name_deleted_in_the_same_save_can_be_taken(self, tmp_path,
                                                          monkeypatch):
        opt = self._opt(tmp_path, monkeypatch, name="round1_del")
        frame = _drop(opt.ingredient_grid_frame(), 3)       # Salt goes
        frame = _edit(frame, 2, **{wording.NAME_LABEL: "Salt"})
        errors, _ = opt.apply_ingredient_grid(frame)
        assert errors == []
        assert [v['name'] for v in opt.variables] == ["Flour", "Salt"]

    def test_a_true_swap_is_refused_rather_than_half_applied(self, tmp_path,
                                                             monkeypatch):
        """Flour to Water and Water to Flour cannot both go first. Refused
        by name, and nothing is written."""
        opt = self._opt(tmp_path, monkeypatch, name="round1_swap")
        frame = _edit(opt.ingredient_grid_frame(), 1,
                      **{wording.NAME_LABEL: "Water"})
        frame = _edit(frame, 2, **{wording.NAME_LABEL: "Flour"})
        errors, messages = opt.apply_ingredient_grid(frame)
        assert errors and messages == []
        assert [v['name'] for v in opt.variables] == ["Flour", "Water", "Salt"]

    # ---- the feasibility snapshot follows the renames ---------------- #

    def test_a_rename_and_a_fix_and_a_property_limit_is_a_legal_save(
            self, tmp_path, monkeypatch):
        """The limit reads a figure filed under the ingredient's name. Left
        behind in the snapshot, the renamed row read as having none and a
        perfectly legal save was refused."""
        opt = self._opt(tmp_path, monkeypatch, name="round1_props")
        opt.add_property("Fat per 100 g")
        opt.set_property_value("Flour", "Fat per 100 g", 2.0)
        opt.set_property_value("Water", "Fat per 100 g", 0.0)
        opt.set_property_value("Salt", "Fat per 100 g", 0.0)
        opt.add_constraint("Fat per 100 g", max_val=2.0)
        frame = _edit(opt.ingredient_grid_frame(), 1,
                      **{wording.NAME_LABEL: "Wheat flour"})
        frame = _edit(frame, 3, **{wording.LOWEST_LABEL: 1.0,
                                   wording.HIGHEST_LABEL: 1.0})
        errors, _ = opt.apply_ingredient_grid(frame)
        assert errors == [], errors
        assert opt.property_value("Wheat flour", "Fat per 100 g") == 2.0

    def test_an_amount_limit_follows_the_rename_in_the_snapshot_too(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch, name="round1_qc")
        opt.add_quantity_constraint(["Flour", "Water"], min_val=10)
        frame = _edit(opt.ingredient_grid_frame(), 1,
                      **{wording.NAME_LABEL: "Wheat flour"})
        frame = _edit(frame, 3, **{wording.LOWEST_LABEL: 1.0,
                                   wording.HIGHEST_LABEL: 1.0})
        errors, _ = opt.apply_ingredient_grid(frame)
        assert errors == [], errors
        assert opt.quantity_constraints[0]['ingredients'] == ["Wheat flour",
                                                              "Water"]

    # ---- an emptied row is not a deletion ---------------------------- #

    def test_a_row_whose_cells_are_rubbed_out_is_refused_not_deleted(
            self, tmp_path, monkeypatch):
        """It carries an identity, so it is a row of the project with its
        answers rubbed out — not the empty line at the bottom. Skipping it
        took it off the grid, which applied a deletion with no question
        asked and no copy kept."""
        opt = self._opt(tmp_path, monkeypatch, name="round1_blank")
        frame = _edit(opt.ingredient_grid_frame(), 2,
                      **{wording.NAME_LABEL: "", wording.TYPE_LABEL: "",
                         wording.LOWEST_LABEL: None,
                         wording.HIGHEST_LABEL: None,
                         wording.UNIT_LABEL: ""})
        assert opt.ingredient_grid_deletions(frame) == []
        errors, messages = opt.apply_ingredient_grid(frame)
        assert errors == [(2, wording.NAME_REQUIRED_ERROR)]
        assert messages == []
        assert [v['name'] for v in opt.variables] == ["Flour", "Water", "Salt"]

    def test_an_emptied_measurement_row_is_refused_too(self, tmp_path,
                                                       monkeypatch):
        opt = self._opt(tmp_path, monkeypatch, name="round1_mblank")
        opt.add_objective("Colour", 1.0, goal="max", min_val=0, max_val=10)
        frame = _edit(opt.measurement_grid_frame(), 1,
                      **{wording.MEASUREMENT_COLUMN: "",
                         wording.GOAL_LABEL: "",
                         wording.LOWEST_MEASURABLE_LABEL: None,
                         wording.HIGHEST_MEASURABLE_LABEL: None,
                         wording.SHARE_COLUMN: None})
        assert opt.measurement_grid_deletions(frame) == []
        errors, _ = opt.apply_measurement_grid(frame)
        assert errors == [(1, wording.NAME_REQUIRED_ERROR)]
        assert len(opt.objectives) == 2

    def test_the_line_at_the_bottom_is_still_not_a_row(self, tmp_path,
                                                       monkeypatch):
        """The one it must not catch: a blank row with no identity is the
        empty line of a dynamic grid, clicked and left alone."""
        opt = self._opt(tmp_path, monkeypatch, name="round1_trailing")
        errors, messages = opt.apply_ingredient_grid(
            _add(opt.ingredient_grid_frame()))
        assert (errors, messages) == ([], [])


class TestTheGridFixesRoundTwo:
    """The name-freeing half of round 2, at the layer it lives at. The
    disarming half is on the screen and is pinned in test_app_ui.py."""

    def _opt(self, tmp_path, monkeypatch, name="round2"):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer(name)
        opt.set_amount_unit("g")
        opt.add_ingredient("Water", 0, 100)
        opt.add_objective("Firmness", 1.0, goal="max", min_val=0, max_val=10,
                          unit="N")
        opt.add_objective("Juiciness", 1.0, goal="max", min_val=0, max_val=10)
        return opt

    def test_a_measurement_name_given_up_in_the_same_save_can_be_taken(
            self, tmp_path, monkeypatch):
        """Firmness becomes Bite and Juiciness takes the name it gave up.
        One edit, one Save, and the write order frees the name first."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.tell({"Water": 50.0}, {"Firmness": 6.0, "Juiciness": 7.0})
        frame = _edit(opt.measurement_grid_frame(), 1,
                      **{wording.MEASUREMENT_COLUMN: "Bite"})
        frame = _edit(frame, 2, **{wording.MEASUREMENT_COLUMN: "Firmness"})
        errors, _ = opt.apply_measurement_grid(frame)
        assert errors == [], errors
        assert sorted(o['name'] for o in opt.objectives) == ["Bite", "Firmness"]
        # Each result followed its own row, and neither landed on the other.
        assert opt.results_history[0] == {"Bite": 6.0, "Firmness": 7.0}

    def test_a_measurement_deleted_in_the_same_save_frees_its_name(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch, name="round2_del")
        frame = _drop(opt.measurement_grid_frame(), 1)      # Firmness goes
        frame = _edit(frame, 2, **{wording.MEASUREMENT_COLUMN: "Firmness"})
        errors, _ = opt.apply_measurement_grid(frame)
        assert errors == [], errors
        assert [o['name'] for o in opt.objectives] == ["Firmness"]

    def test_a_measurement_swap_is_refused_rather_than_half_applied(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch, name="round2_swap")
        frame = _edit(opt.measurement_grid_frame(), 1,
                      **{wording.MEASUREMENT_COLUMN: "Juiciness"})
        frame = _edit(frame, 2, **{wording.MEASUREMENT_COLUMN: "Firmness"})
        errors, messages = opt.apply_measurement_grid(frame)
        assert errors and messages == []
        assert sorted(o['name'] for o in opt.objectives) == ["Firmness",
                                                             "Juiciness"]

    def test_an_ingredients_name_is_still_refused_on_the_measurements_grid(
            self, tmp_path, monkeypatch):
        """The pass that is skipped is the one about this grid's OWN rows.
        An ingredient cannot move under it, so it still refuses."""
        opt = self._opt(tmp_path, monkeypatch, name="round2_ing")
        opt.add_property("Cost")
        for taken, why in (("Water", "already the name of an ingredient"),
                           ("Cost", "already a property")):
            errors, _ = opt.apply_measurement_grid(
                _add(opt.measurement_grid_frame(),
                     **_meas_row(taken, share=20.0)))
            assert errors and why in errors[0][1], errors

    def test_a_rename_onto_a_deleted_name_keeps_the_two_apart(
            self, tmp_path, monkeypatch):
        """Salt goes and Water takes its name. The figures and the limits
        filed under the old Salt go with it — they must not be handed to the
        row that has just taken the name."""
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("round2_props")
        opt.set_amount_unit("g")
        opt.add_ingredient("Water", 0, 100)
        opt.add_ingredient("Salt", 0, 10)
        opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
        opt.add_property("Fat per 100 g")
        opt.set_property_value("Water", "Fat per 100 g", 0.0)
        opt.set_property_value("Salt", "Fat per 100 g", 9.0)
        opt.add_quantity_constraint(["Salt"], max_val=5)
        frame = _drop(opt.ingredient_grid_frame(), 2)       # Salt goes
        frame = _edit(frame, 1, **{wording.NAME_LABEL: "Salt"})
        errors, _ = opt.apply_ingredient_grid(frame)
        assert errors == [], errors
        assert [v['name'] for v in opt.variables] == ["Salt"]
        # The new Salt is the old Water: its own figure, not the dead row's.
        assert opt.property_value("Salt", "Fat per 100 g") == 0.0
        # ...and the limit on the row that went is gone with it.
        assert opt.quantity_constraints == []

    def test_the_snapshot_keeps_them_apart_too(self, tmp_path, monkeypatch):
        """The same save with a fixed row in it, so the feasibility question
        is asked: the snapshot it is asked over must not read the dead row's
        figures under the living row's name."""
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("round2_snap")
        opt.set_amount_unit("g")
        opt.add_ingredient("Water", 0, 100)
        opt.add_ingredient("Salt", 0, 10)
        opt.add_ingredient("Flour", 0, 50)
        opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
        opt.add_property("Fat per 100 g")
        opt.set_property_value("Water", "Fat per 100 g", 0.0)
        opt.set_property_value("Salt", "Fat per 100 g", 90.0)
        opt.set_property_value("Flour", "Fat per 100 g", 0.0)
        opt.add_constraint("Fat per 100 g", max_val=1.0)
        frame = _drop(opt.ingredient_grid_frame(), 2)       # Salt goes
        frame = _edit(frame, 1, **{wording.NAME_LABEL: "Salt",
                                   wording.LOWEST_LABEL: 50.0,
                                   wording.HIGHEST_LABEL: 50.0})
        errors, _ = opt.apply_ingredient_grid(frame)
        assert errors == [], errors
        assert opt._var_by_name("Salt")['bounds'] == (50.0, 50.0)


def test_a_column_whose_absent_value_is_outside_its_range_keeps_its_frame(
        tmp_path, monkeypatch):
    """An ingredient added mid-run with a Lowest above 0 is encoded at 0 in
    every formulation made before it. The search frame has to reach that or
    the GP is handed training rows outside its own [0, 1] box."""
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("absent_frame")
    opt.set_amount_unit("g")
    opt.add_ingredient("Water", 0, 100)
    opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
    opt.tell({"Water": 50.0}, {"Taste": 7.0})
    opt.add_ingredient("Salt", 5, 5, keep_lowest=True)
    spans = opt._search_bounds()
    assert spans[1][0] <= 0.0 <= spans[1][1], spans
    lo, hi = spans[1]
    assert all(lo <= row[1] <= hi for row in opt.X_history), opt.X_history
    # An ingredient that has been there all along is untouched: widening its
    # frame down to 0 would change what the GP sees for every project.
    assert spans[0] == (0.0, 100.0)


def test_an_ingredient_that_is_in_every_formulation_keeps_its_own_frame(
        tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("absent_none")
    opt.set_amount_unit("g")
    opt.add_ingredient("Water", 20, 60)
    opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
    opt.tell({"Water": 30.0}, {"Taste": 7.0})
    assert opt._search_bounds() == [(20.0, 60.0)]

import ast
import pathlib
import re

_USER_FACING_SOURCES = ["app.py", "ui_helpers.py", "ui_setup.py", "ui_batch.py",
                        "ui_results.py", "food_bo.py", "storage.py", "wording.py",
                        # The three modules wave 3 added. Every one of them
                        # drew its own screen text where this sweep could
                        # not see it, which is how sixty sentences reached
                        # the user unread.
                        "calculation_editor.py", "workbook_flow.py",
                        "custom_records.py"]
# The user-facing files that are not Python. They are scanned as plain text,
# except the Swift wrapper, where only its string literals are screen text.
_USER_FACING_TEXT = ["desktop/start_here.txt", "desktop/README.md", "README.md"]
_USER_FACING_SWIFT = "desktop/FoodOptimizerApp.swift"
# The launcher writes the lines the first screen shows while it is setting
# itself up — `status "…"` and the WARM_MSG it publishes for the load step.
# It is a shell script, so only those strings are screen text; the rest is
# machinery. It was in none of these lists, which is how "Loading the model
# components…" stayed on the first screen for a whole wave after the word
# was retired from the window above it.
_USER_FACING_LAUNCHER = "desktop/launcher.sh"
_LAUNCHER_SCREEN_TEXT = re.compile(
    r'(?:status|[A-Z_]*MSG=)\s*"((?:[^"\\\n]|\\.)*)"')


def _launcher_lines(root):
    """Every sentence desktop/launcher.sh puts on the first screen, with the
    bar's own `|percent` tail and its shell variables taken off."""
    text = (root / _USER_FACING_LAUNCHER).read_text()
    out = []
    for literal in _LAUNCHER_SCREEN_TEXT.findall(text):
        line = literal.split("|")[0]
        line = re.sub(r"\$\{?[A-Za-z_][A-Za-z_0-9]*\}?", "", line)
        if line.strip():
            out.append(line)
    return out

# 0.4.0: a formulation with no result is NOT SCORED, whether or not it was
# made. One object, used by the sweep below and by the test that names this
# rename on its own, so the two can never disagree about what is banned.
_NOT_MADE = re.compile(r"\bnot made\b", re.I)

# 0.4.0: the sidebar's export/import pair is SAVED COPIES in plain words, not
# the engineering term. Same reuse as _NOT_MADE above.
_BACKUP = re.compile(r"\bbackups?\b", re.I)

# 0.4.0: a batch leaves the app as one workbook. "Bench sheet" was the name
# of the comma-separated file it used to be, and it named a thing that is
# no longer on screen.
_BENCH_SHEET = re.compile(r"\bbench sheets?\b", re.I)

# 0.4.1: a row left out of the search is HELD at one amount, and the button
# says which amount. Pause named a state and said nothing about the number
# the row would sit at, and "resume" was the only way back out of it.
_PAUSE = re.compile(r"\bpaus(e|ed|ing)\b", re.I)

# 0.4.0: a file is an Excel workbook or a comma-separated file, and the user
# chooses between them. CSV may therefore be SAID — but only inside the label
# that offers the choice, never as the name of the route out of the app.
# Upper case only: a file extension in a `type=` list is machinery.
_CSV_WORD = re.compile(r"\bCSV\b")
_CSV_ALLOWED = "(Excel or CSV)"

# 0.5.0: what a measurement is worth is typed as its SHARE OF SCORE, out of
# 100, and the importance behind it is derived. Importance named a number
# the reader no longer enters, on a scale they could not see the sum of.
_IMPORTANCE = re.compile(r"\bimportance\b", re.I)

_BANNED = [
    _IMPORTANCE,
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
    # out of a project, with Delete kept for the project itself; Not scored
    # for a formulation with no result; Formulation total for the total a
    # batch is written to; and no Priority column beside the importance it
    # was a rank of.
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
    # The batch wording wave (2026-09-10) made a round of formulations a
    # BATCH and banned TRIAL, the older name for the same set. 0.5.0 moved
    # again: BATCH SIZE is the weight of ONE formulation — the sense the
    # bench already uses the word in — so the set is a ROUND and "batch" is
    # banned everywhere except inside "batch size". TRIAL stays banned; it
    # was never either of them. Stored field names keep their old spelling
    # (batch_history, pending_batch, pending_batch_total, batch_totals):
    # they never reach a screen, and "batch" inside an identifier is not a
    # word boundary away from the underscores either side of it, so the
    # pattern below leaves them alone.
    re.compile(r"\btrials?\b", re.I),
    re.compile(r"\bbatch(es)?\b(?! size)", re.I),
    re.compile(r"\bscales?\b", re.I),
    re.compile(r"\bKind\b"),
    # Case-insensitive: "removed" survived in Start Here ("copy or remove
    # them") and in storage.py's load error ("moved, renamed or removed"),
    # because the pattern only ever matched a capitalised Remove.
    re.compile(r"\bremoved?\b", re.I),
    re.compile(r"\bRemake\b"),
    re.compile(r"\bShare\b(?! of score)"),
    # The screen said "Not made" of a bowl that was made and never measured,
    # and there was then no way to score it later.
    _NOT_MADE,
    # 0.4.0: the sidebar's export/import pair is named in plain words —
    # Saved copies, Save a copy of this project, Open a saved copy — not the
    # engineering term for what they are.
    _BACKUP,
    # 0.4.0: the batch leaves as a workbook, and nothing on screen calls a
    # sheet a bench sheet.
    _BENCH_SHEET,
    # 0.4.1: Hold replaces Pause, everywhere the user can read it.
    _PAUSE,
    # 0.5.0: and then Hold itself goes. A row pinned at one amount is one
    # whose Lowest IS its Highest, which the two boxes already say; there is
    # no control called Hold, no "held at" status and no "vary it again".
    # "holds" and "holding" are the ordinary English verb ("a copy holds
    # everything") and are not this word.
    re.compile(r"\b(hold|held)\b", re.I),
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

    def test_the_compared_with_column_is_reserved_too(self, tmp_path,
                                                      monkeypatch):
        """The batch table's last column is headed with the best
        formulation's number, or with the allowed amounts during the cold
        start; an ingredient of either name would collide with it exactly as
        one named 'Total (g)' collides with the total."""
        opt = self._opt(tmp_path, monkeypatch)
        for name in ("Compared with Formulation 3", "compared with the "
                     "allowed amounts", "Compared with Formulation3"):
            with pytest.raises(ValueError, match="column name Food Optimizer"):
                opt.add_ingredient(name, 0, 10)
        with pytest.raises(ValueError, match="column name Food Optimizer"):
            opt.add_process_parameter("Compared with Formulation 1", 0, 10)
        # An ingredient whose name merely starts the same way is fine: the
        # header is one of those two shapes and nothing else.
        opt.add_ingredient("Compared with last season", 0, 10)
        assert "Compared with last season" in [v['name'] for v in opt.variables]

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
    # Custom records apply to formulations and process-only trials.
    wording.CUSTOM_FIELD_HELP, wording.CUSTOM_SCOPE_NAMES["formulation"],
    wording.CUSTOM_SECTION_NAMES["formulation"],
    # Process-only studies use trial; weighed recipe describes the optional input method.
    wording.PROCESS_STUDY_INTRO, wording.METHOD_HELP, wording.COMPOSITION_ENTRY_LABEL, "run the trials",
    # Scientific teaching copy requested by the owner uses measurement scale,
    # ingredient weight and experimental results in their ordinary meanings.
    wording.MEASUREMENT_GRID_CAPTION, wording.MEASUREMENT_MIN_HELP,
    wording.MEASUREMENT_MAX_HELP, wording.RULE_GUIDE, wording.FORMULA_HELP,
    wording.MADE_AS_HELP, wording.SAMPLE_TAB1_DESCRIPTION,
    wording.LOWEST_MEASURABLE_LABEL, wording.HIGHEST_MEASURABLE_LABEL,
    wording.WIDEN_RANGE_HINT,
    wording.SAMPLE_TARGETS_SOURCE, wording.SAMPLE_METHOD, wording.LEGACY_SAMPLE_TARGETS_SOURCE, wording.SAMPLE_REFERENCE,

    # The one legacy value that must stay spelled the old way: it is the
    # reserved column name a 0.2.x project could collide with.
    "Overall Score",
    # "Delete" is now the app's one verb for taking something out, so it is
    # no longer restricted; these entries stay because the sidebar's
    # sentences are assembled from fragments the scan sees one at a time.
    "Delete this project",
    "Yes, delete it",
    "Delete **",
    "**? It has no formulations yet. ",
    "** and its ",
    # 0.5.0, the Batch size box's help. "Scale" is banned as the NAME of a
    # measurement's range (that is Range); here it is the plain verb for
    # what the sheets do when the size changes, and no other word says it as
    # shortly. The sentence is an f-string, so the scan sees this fragment.
    " adds up to this. Change it and the sheets scale with it.",
}

# Fragments of the sidebar's delete-the-project sentences (they are f-strings,
# so each piece is scanned on its own).
_ALLOWED_PREFIXES = ("Delete **", "Deleted ")

# Single-word literals that are internal machinery, never screen text.
_ALLOWED_SINGLE_WORDS = {
    # Legacy CSV column headers an import still accepts, and the reserved
    # names a 0.2.x project could collide with (RESERVED_VARIABLE_NAMES,
    # which keeps Trial reserved too: a project stored before that wording
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

# Allowances that belong to ONE file. A blanket entry would have waved the
# same word through everywhere, and "Batch" waved through in wording.py is a
# screen label: that module is nothing but screen text, so a bare "Batch"
# there is exactly the mistake this guard exists to catch. In food_bo.py the
# same two literals are machinery — the reserved column name a 0.2.x project
# could collide with, and the key each row of the record keeps its round
# number under — and nothing in that file is a label.
#
# 0.7.0 adds one more to the same scoped set, for the same reason: what per
# cent of a pre-mix each part is is stored under the key 'share', and that
# key is named in food_bo.py alone. The word itself stays banned — a bare
# "Share" column header is the mistake the guard exists to catch, and the
# prose pattern above cannot see a one-word literal — so the allowance is
# the exact lower-cased key, scoped to the one file that writes it.
# Wave 3's three modules keep the same shape of allowance, each for a
# stored key the user never reads: the round a metadata block issues its
# amounts under, and the round number a skipped row carries.
_ALLOWED_SINGLE_WORDS_BY_FILE = {
    "wording.py": {"trial", "trials", "Trials"},  # process-only studies
    "food_bo.py": {"Batch", "batch", "share"},
    "workbook_flow.py": {"recipes"},   # the issued-plan key in the metadata
    "custom_records.py": {"batch"},    # the round number a skipped row keeps
}

# Fragments removed from a file before it is scanned, by name and by file:
# CSS property names and DOM ids inside the calculation editor's own web
# component, which is machinery the reader never meets as words. The same
# shape as _SWIFT_NOT_PROSE, for the same reason.
_NOT_PROSE_BY_FILE = {
    "calculation_editor.py": ("font-weight", 'id="batch"', "#batch",
                              "'batch size'"),
}


def _allowed_single_words(name):
    """The one-word literals `name` may keep: the ones every file may keep,
    plus that file's own."""
    return _ALLOWED_SINGLE_WORDS | _ALLOWED_SINGLE_WORDS_BY_FILE.get(name, set())


def _single_word_offenders(name, words):
    """The one-word literals of `words` that the guard refuses in `name`.
    Split out from the sweep so a test can hand it words no file contains."""
    allowed = _allowed_single_words(name)
    return [(name, word) for word in words
            if word not in allowed and _SINGLE_WORDS.fullmatch(word)]
# Fragments removed from the Swift wrapper before it is scanned: CSS property
# names inside the setup page's inline styles, not prose.
_SWIFT_NOT_PROSE = ("font-weight",)

_SINGLE_WORDS = re.compile(
    # 'ranges', not 'range': Range is the measurement's own column header.
    # 'share', not the two-word "Share of score": a bare one-word "Share"
    # column header is still banned, and the prose pattern above only
    # catches Share when it is NOT followed by "of score" — a single-word
    # literal never has room for that trailing phrase, so it needs its own
    # ban here.
    r"^(recipes?|experiments?|objectives?|weights?|ranges|rewind|pruned"
    r"|priority|trials?|batch(es)?|kind|scales?|remove|remake|share|backups?"
    # 0.4.1: a one-word "Paused" column value is the shape this one took.
    # 0.5.0: and the one-word "Held" a Status cell could hold.
    # 0.5.0: and the one-word "Importance" a column header was.
    r"|pause[ds]?|pausing|hold|held|importance)$", re.I)


def _how_it_works():
    """The two collapsed folds that explain how the app works: How it works
    and, under it, How closeness is calculated.

    They no longer say a banned word between them — the specialist
    vocabulary they were written to carry is gone — so the blanket exemption
    is gone with it and they are scanned like every other sentence. The set
    is empty rather than removed so that the reason stays written down: a
    bullet that reaches for "objective" again is a bullet the guard should
    catch, not one it waves through.
    """
    return set()


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
    measurement's own Range is the one exception), rewind → undo, trial and
    batch → round (batch survives only in "batch size", the weight of one
    formulation), Kind → Type, Scale → Range, Remove → Delete, Remake →
    Repeat,
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
            scanned = text
            for fragment in _NOT_PROSE_BY_FILE.get(name, ()):
                scanned = scanned.replace(fragment, "")
            if any(pattern.search(scanned) for pattern in _BANNED):
                offenders.append((name, text))
        offenders += _single_word_offenders(
            name, _single_word_constants(root / name))
    assert offenders == [], offenders


@pytest.mark.parametrize("word", ["Batch", "batch", "BATCH"])
def test_a_bare_batch_is_refused_everywhere_but_food_bo(word):
    """The allowance is scoped, and this is the proof. Until it was, a
    "Batch" column header typed into wording.py — the module that is nothing
    but screen text — passed the guard on the strength of an entry that
    exists for food_bo's reserved names."""
    for name in _USER_FACING_SOURCES:
        refused = _single_word_offenders(name, [word])
        # food_bo keeps the reserved column name and the row key; custom
        # records keeps the round number a skipped row is filed under.
        if ((name == "food_bo.py" and word in {"Batch", "batch"})
                or (name == "custom_records.py" and word == "batch")):
            assert refused == [], (name, word)
        else:
            assert refused == [(name, word)], (name, word)


@pytest.mark.parametrize("word", ["trial", "trials", "Trials"])
def test_a_bare_trial_is_refused_everywhere_but_wording(word):
    """The allowance is scoped, and this is the proof. A process-only study
    is a run of trials and wording.py is where those sentences are built;
    the same bare word typed into any other module is the mistake the guard
    exists to catch."""
    for name in _USER_FACING_SOURCES:
        refused = _single_word_offenders(name, [word])
        if name == "wording.py":
            assert refused == [], (name, word)
        else:
            assert refused == [(name, word)], (name, word)


def test_the_measurement_range_headers_are_exempt_by_name_only():
    """The two column headers are the owner's choice and are allowed as
    exact sentences named in _ALLOWED_EXACT — not as a licence for the word
    itself. A bare "Scale" is still refused in every file, this one
    included."""
    for label in (wording.LOWEST_MEASURABLE_LABEL,
                  wording.HIGHEST_MEASURABLE_LABEL):
        assert label in _ALLOWED_EXACT, label
        assert any(pattern.search(label) for pattern in _BANNED), label
    for name in _USER_FACING_SOURCES:
        assert _single_word_offenders(name, ["Scale"]) == [(name, "Scale")]


def test_even_food_bo_may_not_say_batches():
    """The scoped allowance is two exact literals, not the word. A plural is
    not a stored key or a reserved column name, so nothing wants one."""
    assert _single_word_offenders("food_bo.py", ["batches", "Batches"]) == [
        ("food_bo.py", "batches"), ("food_bo.py", "Batches")]


# What a prose literal in food_bo.py may still be. Each is machinery the
# user never reads, and each is here by name rather than by shape so that a
# sentence can never hide behind the allowance.
_FOOD_BO_NOT_PROSE = {
    "see its page",  # legacy workbook pointer, accepted only when reading old files
    # Reserved column names and the frame keys that ARE those columns. The
    # words are wording.py's; these are the lookups into a dataframe.
    "Overall Score", "Overall score",
    # A namedtuple's field list, a MIME type, a file extension, two regexes
    # and the log line _damaged writes.
    "frame actual lots",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".pkl", r"^total(\s*\(.*\))?$", r"\s+(\d+)(\.\d+)?", r")\s+(\d+)(\.\d+)?",
    "saved copy refused: %s",
    # outside_message's own glue. It is the one sentence builder left in
    # this file: it needs join_unit and unit_after_number, and wording.py
    # imports nothing from food_bo. Its arguments — what the value is
    # outside of, and the hint after it — are wording's.
    " is outside ", " of ", " to ",
}
# export_trajectory writes a plain-text dump for an expert re-query. It
# reaches no screen in the app (nothing under ui_*.py or app.py calls it);
# it is left out by name, not by shape.
_FOOD_BO_NOT_SCREENS = ("export_trajectory",)


def _food_bo_prose_literals(source=None):
    """Every string literal in food_bo.py that reads like a sentence, with
    the docstrings, the named machinery and export_trajectory taken out.
    `source` lets a test sweep an altered copy of the file."""
    root = pathlib.Path(__file__).resolve().parent.parent
    if source is None:
        source = (root / "food_bo.py").read_text()
    tree = ast.parse(source)
    skip = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)) and body:
            first = body[0]
            if (isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                skip.add(id(first.value))
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name in _FOOD_BO_NOT_SCREENS):
            for inner in ast.walk(node):
                skip.add(id(inner))
        # A programming mistake, not a refusal: it reaches a traceback in
        # the log, never a sentence on a screen.
        if (isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call)
                and getattr(node.exc.func, "id", "") in ("IndexError",
                                                         "TypeError")):
            for inner in ast.walk(node):
                skip.add(id(inner))
        # _damaged(detail) IS the log line: one sentence reaches the screen
        # (wording.COPY_DAMAGED) and the field name goes to logging, which
        # is the whole point of R1.
        if (isinstance(node, ast.Call)
                and getattr(node.func, "id", "") == "_damaged"):
            for inner in ast.walk(node):
                skip.add(id(inner))
    out = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Constant)
                and isinstance(node.value, str)) or id(node) in skip:
            continue
        text = node.value
        if text in _FOOD_BO_NOT_PROSE:
            continue
        # Prose is what has words in it: two letters, and a space or a stop
        # between or after them. A stored key, a category or a separator is
        # not.
        if re.search(r"[A-Za-z].*[A-Za-z]", text) and (" " in text
                                                       or "." in text):
            out.append((node.lineno, text))
    return out


def test_no_sentence_in_food_bo_is_written_there():
    """wording.py's first line says it is every word the user reads. It was
    not true: seventy-odd `raise ValueError` sites in food_bo.py carried
    their own prose, and six more families were assembled in helpers and
    handed to a bare `raise ValueError(trouble)`, to a per-row error list or
    straight onto a screen through `load_error` — which the first version of
    this guard, reading only the literals AT a raise site, could not see.

    Every sentence the user can reach is a wording constant or a small
    wording function. food_bo may build one out of values — numbers, names,
    another message — and may keep the machinery named above; it may not
    write one.
    """
    assert _food_bo_prose_literals() == [], _food_bo_prose_literals()


def test_the_food_bo_sweep_reads_more_than_the_raise_sites():
    """The proof that the widening is real: a sentence assigned in a helper,
    nowhere near a `raise`, is what the sweep has to catch — so the sweep
    itself is run over a copy of the file with one planted."""
    root = pathlib.Path(__file__).resolve().parent.parent
    source = (root / "food_bo.py").read_text()
    sentence = "the ingredients that can still vary"
    planted = source.replace(
        "    def per_amount_text(self):",
        '    def _planted(self):\n'
        f'        return "{sentence}"\n\n'
        "    def per_amount_text(self):", 1)
    assert planted != source
    # The plant is not at a raise site, which is the whole point...
    tree = ast.parse(planted)
    at_raises = [inner.value for node in ast.walk(tree)
                 if isinstance(node, ast.Raise)
                 for inner in ast.walk(node)
                 if isinstance(inner, ast.Constant) and inner.value == sentence]
    assert at_raises == []
    # ...and the sweep still finds it, and finds nothing without it.
    assert [t for _, t in _food_bo_prose_literals(planted)] == [sentence]
    assert _food_bo_prose_literals(source) == []


def test_no_old_vocabulary_reaches_the_user_outside_python():
    """Start Here, the two READMEs and the app window's own copy are read by
    the same people, so they follow the same vocabulary."""
    root = pathlib.Path(__file__).resolve().parent.parent
    offenders = []
    for name in _USER_FACING_TEXT:
        for i, line in enumerate(( root / name).read_text().splitlines(), 1):
            # Both record scopes also apply to process-only trials.
            if (name, line) in {
                ("desktop/start_here.txt", "formulation/trial or each ingredient in the round. Enter values in the"),
                ("README.md", "record, then choose each formulation/trial or each ingredient in the round."),
            }:
                continue
            # Process-only studies deliberately use trials in the user guide.
            if name == "desktop/start_here.txt" and line == "and time. A project containing only process settings uses trials and does":
                continue
            if any(pattern.search(line) for pattern in _BANNED):
                offenders.append((name, i, line))
    swift = (root / _USER_FACING_SWIFT).read_text()
    for fragment in _SWIFT_NOT_PROSE:
        swift = swift.replace(fragment, "")
    for literal in re.findall(r'"((?:[^"\\\n]|\\.)*)"', swift):
        if any(pattern.search(literal) for pattern in _BANNED):
            offenders.append((_USER_FACING_SWIFT, literal))
    for line in _launcher_lines(root):
        if any(pattern.search(line) for pattern in _BANNED):
            offenders.append((_USER_FACING_LAUNCHER, line))
    assert offenders == [], offenders


def test_the_launcher_and_the_window_name_the_load_step_alike():
    """The window decides which step a status line is about by matching the
    load step's label as a PREFIX of it, so the label and the launcher's own
    line for it cannot drift apart: they drifted, and the warm-up — the slow
    part of a cold start — was drawn under "Opening your projects"."""
    root = pathlib.Path(__file__).resolve().parent.parent
    swift = (root / _USER_FACING_SWIFT).read_text()
    label = re.search(r'let loadStepLabel = "((?:[^"\\\n]|\\.)*)"',
                      swift).group(1)
    warm = re.search(r'WARM_MSG="((?:[^"\\\n]|\\.)*)"',
                     (root / _USER_FACING_LAUNCHER).read_text()).group(1)
    assert warm.startswith(label), (warm, label)
    assert "model" not in warm.lower() and "model" not in label.lower()


def test_no_screen_says_not_made():
    """0.4.0: a formulation with no result is "Not scored". "Not made" was
    wrong about the bowl that was made and never measured, and it said
    nothing could be done about it — a not-scored formulation can now be
    scored from the Results tab."""
    root = pathlib.Path(__file__).resolve().parent.parent
    offenders = []
    for name in _USER_FACING_SOURCES:
        offenders += [(name, text) for text in _string_constants(root / name)
                      if _NOT_MADE.search(text)]
    for name in _USER_FACING_TEXT:
        offenders += [(name, i) for i, line
                      in enumerate((root / name).read_text().splitlines(), 1)
                      if _NOT_MADE.search(line)]
    swift = (root / _USER_FACING_SWIFT).read_text()
    offenders += [(_USER_FACING_SWIFT, literal) for literal
                  in re.findall(r'"((?:[^"\\\n]|\\.)*)"', swift)
                  if _NOT_MADE.search(literal)]
    assert offenders == [], offenders


def test_no_screen_says_bench_sheet():
    """0.4.0: the batch leaves the app as one workbook — a summary sheet and
    one sheet per formulation. "Bench sheet" named the comma-separated file
    that used to be the only route out, and the screen no longer has one."""
    root = pathlib.Path(__file__).resolve().parent.parent
    offenders = []
    for name in _USER_FACING_SOURCES:
        offenders += [(name, text) for text in _string_constants(root / name)
                      if _BENCH_SHEET.search(text)]
    for name in _USER_FACING_TEXT:
        offenders += [(name, i) for i, line
                      in enumerate((root / name).read_text().splitlines(), 1)
                      if _BENCH_SHEET.search(line)]
    swift = (root / _USER_FACING_SWIFT).read_text()
    offenders += [(_USER_FACING_SWIFT, literal) for literal
                  in re.findall(r'"((?:[^"\\\n]|\\.)*)"', swift)
                  if _BENCH_SHEET.search(literal)]
    assert offenders == [], offenders


def test_csv_is_named_only_where_the_user_picks_a_file():
    """0.4.0: every route out of the app is a workbook, and a route in
    accepts either shape. So "CSV" belongs in exactly one place — the label
    that offers the choice, `(Excel or CSV)` — and nowhere else on screen: a
    caption that tells a formulator to Save As CSV is describing a workflow
    the app no longer has."""
    root = pathlib.Path(__file__).resolve().parent.parent
    offenders = []
    for name in _USER_FACING_SOURCES:
        for text in _string_constants(root / name):
            if _CSV_WORD.search(text.replace(_CSV_ALLOWED, "")):
                offenders.append((name, text))
    assert offenders == [], offenders


def test_no_screen_says_backup():
    """0.4.0: the sidebar's export/import pair is Saved copies — Save a copy
    of this project, Open a saved copy — in the words a formulation
    scientist already uses, not the engineering term for what they are."""
    root = pathlib.Path(__file__).resolve().parent.parent
    offenders = []
    for name in _USER_FACING_SOURCES:
        offenders += [(name, text) for text in _string_constants(root / name)
                      if _BACKUP.search(text)]
    for name in _USER_FACING_TEXT:
        offenders += [(name, i) for i, line
                      in enumerate((root / name).read_text().splitlines(), 1)
                      if _BACKUP.search(line)]
    swift = (root / _USER_FACING_SWIFT).read_text()
    offenders += [(_USER_FACING_SWIFT, literal) for literal
                  in re.findall(r'"((?:[^"\\\n]|\\.)*)"', swift)
                  if _BACKUP.search(literal)]
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
        with pytest.raises(ValueError, match="damaged"):
            FoodOptimizer.validate_state(state)
        state['property_names'] = None
        # An absent or null list is simply no properties, not a broken file.
        FoodOptimizer.validate_state(state)


class TestTheTotalABatchWasPrintedTo:
    """The amounts stored with a result are always as generated, so without
    this the number the bench actually weighed out was lost the moment the
    batch closed — and tab 3's `Amounts to make it` showed a formulation
    nobody had ever made."""

    def _opt(self, tmp_path, monkeypatch, name="batch_totals"):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer(name)
        opt.set_amount_unit("g")
        opt.add_ingredient("Pea protein", 0, 100)
        opt.add_ingredient("Water", 0, 100)
        opt.add_objective("Firmness", 1.0, goal="max", min_val=0, max_val=10)
        return opt

    def test_the_open_batchs_total_is_stored_and_read_back(
            self, tmp_path, monkeypatch):
        """The model's half of it. That the SCREEN opens at the stored number
        is tab 2's claim, and test_the_stored_total_opens_the_box_in_a_new_
        session in tests/test_app_ui.py makes it."""
        opt = self._opt(tmp_path, monkeypatch)
        assert opt.pending_batch_total is None
        opt.set_pending_batch([{"Pea protein": 10.0, "Water": 5.0}])
        opt.set_pending_batch_total(150.0)
        assert FoodOptimizer("batch_totals").pending_batch_total == 150.0
        # Clearing the box is a value of its own: as generated.
        opt.set_pending_batch_total(None)
        assert FoodOptimizer("batch_totals").pending_batch_total is None

    def test_setting_the_same_total_again_writes_nothing(self, tmp_path,
                                                         monkeypatch):
        """It is set on every render of the tab, and a write per rerun would
        bump the file's mtime and make another open window cry conflict."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_pending_batch([{"Pea protein": 10.0, "Water": 5.0}])
        opt.set_pending_batch_total(150.0)
        saved_at = opt.last_saved_at
        opt.set_pending_batch_total(150.0)
        assert opt.last_saved_at == saved_at

    def test_discarding_the_batch_clears_the_total(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_pending_batch([{"Pea protein": 10.0, "Water": 5.0}])
        opt.set_pending_batch_total(150.0)
        opt.set_pending_batch(None)
        assert opt.pending_batch_total is None
        assert FoodOptimizer("batch_totals").pending_batch_total is None

    def test_recording_the_batch_keeps_the_total_under_its_number(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_pending_batch([{"Pea protein": 10.0, "Water": 5.0}],
                              batch_no=4)
        opt.set_pending_batch_total(150.0)
        opt.tell({"Pea protein": 10.0, "Water": 5.0}, {"Firmness": 6.0},
                 formulation_no=1, batch_no=4)
        opt.set_pending_batch(None)
        again = FoodOptimizer("batch_totals")
        assert again.batch_total(4) == 150.0
        assert again.batch_total(3) is None
        assert again.batch_total(None) is None

    def test_a_batch_printed_as_generated_records_that_it_was(self, tmp_path,
                                                              monkeypatch):
        """"Made as generated" is an answer, and the batch keeps it: with no
        entry at all, a total typed on tab 1 months later was read back as
        the total this batch had been made to."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_pending_batch([{"Pea protein": 10.0, "Water": 5.0}],
                              batch_no=1)
        opt.tell({"Pea protein": 10.0, "Water": 5.0}, {"Firmness": 6.0},
                 formulation_no=1, batch_no=1)
        assert opt.batch_totals == {1: None}
        assert opt.batch_total(1) is None
        assert opt.recorded_total(1) is None

    def test_both_totals_round_trip_through_a_backup(self, tmp_path,
                                                     monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_pending_batch([{"Pea protein": 10.0, "Water": 5.0}],
                              batch_no=2)
        opt.set_pending_batch_total(150.0)
        opt.tell({"Pea protein": 10.0, "Water": 5.0}, {"Firmness": 6.0},
                 formulation_no=1, batch_no=2)
        state = opt.export_json()
        assert state['pending_batch_total'] == 150.0
        FoodOptimizer.validate_state(state)
        fresh = FoodOptimizer("restored_totals")
        fresh.import_json(state)
        assert fresh.pending_batch_total == 150.0
        assert fresh.batch_total(2) == 150.0

    def test_a_file_from_before_the_feature_opens_as_generated(self, tmp_path,
                                                               monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_pending_batch([{"Pea protein": 10.0, "Water": 5.0}])
        state = opt.export_json()
        state.pop('pending_batch_total', None)
        state.pop('batch_totals', None)
        FoodOptimizer.validate_state(state)
        fresh = FoodOptimizer("old_file_totals")
        fresh.import_json(state)
        assert fresh.pending_batch_total is None
        assert fresh.batch_totals == {}

    def test_a_malformed_total_is_refused_before_import(self, tmp_path,
                                                        monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        state = opt.export_json()
        state['pending_batch_total'] = "big"
        with pytest.raises(ValueError, match="damaged"):
            FoodOptimizer.validate_state(state)
        state['pending_batch_total'] = None
        state['batch_totals'] = [150.0]
        with pytest.raises(ValueError, match="damaged"):
            FoodOptimizer.validate_state(state)
        state['batch_totals'] = {"2": "big"}
        with pytest.raises(ValueError, match="damaged"):
            FoodOptimizer.validate_state(state)
        state['batch_totals'] = {"2": 150.0}
        FoodOptimizer.validate_state(state)

    def test_a_batch_with_no_rows_left_forgets_its_total(self, tmp_path,
                                                         monkeypatch):
        """A batch number never comes back, so a total left behind could only
        ever be read against a batch nobody can see."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_pending_batch([{"Pea protein": 10.0, "Water": 5.0},
                               {"Pea protein": 20.0, "Water": 10.0}],
                              batch_no=1)
        opt.set_pending_batch_total(150.0)
        opt.tell({"Pea protein": 10.0, "Water": 5.0}, {"Firmness": 6.0},
                 formulation_no=1, batch_no=1)
        opt.tell({"Pea protein": 20.0, "Water": 10.0}, {"Firmness": 7.0},
                 formulation_no=2, batch_no=1)
        opt.set_pending_batch(None)
        assert opt.batch_total(1) == 150.0
        # One row of the batch goes: the batch is still there, so is its total.
        opt.delete_formulations([1])
        assert opt.batch_total(1) == 150.0
        opt.delete_formulations([2])
        assert opt.batch_total(1) is None
        assert FoodOptimizer("batch_totals").batch_totals == {}

    def test_undoing_a_batch_forgets_its_total(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_pending_batch([{"Pea protein": 10.0, "Water": 5.0}],
                              batch_no=1)
        opt.set_pending_batch_total(150.0)
        opt.tell({"Pea protein": 10.0, "Water": 5.0}, {"Firmness": 6.0},
                 formulation_no=1, batch_no=1)
        opt.set_pending_batch(None)
        assert opt.undo_last_batch() == (1, 1)
        assert opt.batch_total(1) is None
        assert FoodOptimizer("batch_totals").batch_totals == {}

    def test_the_total_is_written_out_the_same_way_wherever_it_is_read(
            self, tmp_path, monkeypatch):
        """Tab 2's caption and tab 3's heading name the same number, so the
        number is written out in one place."""
        opt = self._opt(tmp_path, monkeypatch)
        assert opt.batch_total_text(150.0) == "150 g"
        assert opt.batch_total_text(12.5) == "12.5 g"
        assert opt.batch_total_text(None) == ""


class TestWhereTheTargetsComeFrom:
    """The optional free-text note on where the measurement targets came
    from -- a benchmark product, a published panel, a brief from marketing.
    Blank means nothing was ever recorded."""

    def _opt(self, tmp_path, monkeypatch, name="targets_source"):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer(name)
        opt.set_amount_unit("g")
        opt.add_ingredient("Pea protein", 0, 100)
        opt.add_objective("Firmness", 1.0, goal="max", min_val=0, max_val=10)
        return opt

    def test_blank_by_default(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        assert opt.targets_source == ""

    def test_set_and_read_back_after_reload(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_targets_source("  Benchmark burger, panel of 8.  ")
        assert opt.targets_source == "Benchmark burger, panel of 8."
        assert (FoodOptimizer("targets_source").targets_source
                == "Benchmark burger, panel of 8.")

    def test_clearing_it_is_a_value_of_its_own(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_targets_source("Benchmark burger, panel of 8.")
        opt.set_targets_source("")
        assert opt.targets_source == ""
        assert FoodOptimizer("targets_source").targets_source == ""

    def test_setting_the_same_text_again_writes_nothing(self, tmp_path,
                                                        monkeypatch):
        """It is set only from an explicit Save click, but the same guard
        every other per-project setter uses keeps a same-value click from
        bumping the file's mtime and making another open window cry
        conflict."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_targets_source("Benchmark burger, panel of 8.")
        saved_at = opt.last_saved_at
        opt.set_targets_source("Benchmark burger, panel of 8.")
        assert opt.last_saved_at == saved_at

    def test_round_trips_through_export_and_import(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_targets_source("Benchmark burger, panel of 8.")
        state = opt.export_json()
        assert state['targets_source'] == "Benchmark burger, panel of 8."
        FoodOptimizer.validate_state(state)
        fresh = FoodOptimizer("restored_targets_source")
        fresh.import_json(state)
        assert fresh.targets_source == "Benchmark burger, panel of 8."

    def test_a_file_from_before_the_feature_opens_blank(self, tmp_path,
                                                        monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        state = opt.export_json()
        state.pop('targets_source', None)
        FoodOptimizer.validate_state(state)
        fresh = FoodOptimizer("old_file_targets_source")
        fresh.import_json(state)
        assert fresh.targets_source == ""

    def test_a_malformed_value_is_refused_before_import(self, tmp_path,
                                                        monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        state = opt.export_json()
        state['targets_source'] = 123
        with pytest.raises(ValueError, match="damaged"):
            FoodOptimizer.validate_state(state)
        state['targets_source'] = ["a list"]
        with pytest.raises(ValueError, match="damaged"):
            FoodOptimizer.validate_state(state)
        state['targets_source'] = None
        FoodOptimizer.validate_state(state)
        state['targets_source'] = "Benchmark burger, panel of 8."
        FoodOptimizer.validate_state(state)


# ------------------------------------------------------------------ #
#  Default batch size (0.4.0 §C)
# ------------------------------------------------------------------ #

_SAMPLE_CSV = pathlib.Path(__file__).resolve().parent.parent / "data" / \
    "sample_ingredients.csv"


_FLAT_BURGER_CSV = os.path.join(os.path.dirname(__file__), "fixtures", "flat_burger.csv")


class TestFormulationTotal:
    """One number on tab 1 says how big a formulation is, and every
    suggestion adds up to it. It is stored as a number AND written as the
    limit over every ingredient, because the limit is what the space-filling
    opening and the model already obey."""

    def _sample(self, tmp_path, monkeypatch, name="sample"):
        """The sample project's own nine grid rows, with the Water typed
        by hand rather than calculated: they then add up to at least 20 g
        and at most 131 g, which is what makes 100 g reachable and 150 g
        not.

        The shipped sample gives Water `= rest` (0.5.0 wave 2), and a
        balance row makes every sum exactly the batch size — so a reach with
        two ends to name needs the row it had before. That sample is pinned
        on its own in TestTheFormulaColumn."""
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer(name, robust=False)
        opt.set_amount_unit("g")
        opt.load_ingredients_from_csv(pd.read_csv(_FLAT_BURGER_CSV))
        opt.clear_formula("Water")
        opt.add_ingredient("Water", 20, 60, unit="g")
        opt.add_objective("Juiciness", 1.0, goal="target", target=7,
                          min_val=0, max_val=10, unit="/10")
        opt.add_objective("Firmness", 1.5, goal="target", target=6,
                          min_val=0, max_val=10, unit="/10")
        return opt

    def test_setting_it_writes_the_limit_over_every_ingredient(
            self, tmp_path, monkeypatch):
        opt = self._sample(tmp_path, monkeypatch)
        opt.set_formulation_total(100)
        assert opt.formulation_total == 100.0
        qcs = opt.quantity_constraints
        assert len(qcs) == 1
        assert qcs[0]['source'] == 'formulation_total'
        assert set(qcs[0]['ingredients']) == {v['name'] for v in opt.variables}
        # Half a percent either way: an equality is not something a
        # continuous search can be held to exactly.
        assert qcs[0]['min'] == pytest.approx(99.5)
        assert qcs[0]['max'] == pytest.approx(100.5)

    def test_the_reachable_range_is_the_sum_of_the_allowed_amounts(
            self, tmp_path, monkeypatch):
        opt = self._sample(tmp_path, monkeypatch)
        assert opt.total_reach() == (20.0, 131.0)

    def test_a_total_the_amounts_cannot_reach_is_refused_in_numbers(
            self, tmp_path, monkeypatch):
        """Nothing about the search can rescue a sum with no solution, so the
        refusal names the number asked for and the number reachable."""
        opt = self._sample(tmp_path, monkeypatch)
        with pytest.raises(ValueError) as high:
            opt.set_formulation_total(150)
        assert str(high.value) == (
            "A default batch size of 150 g is not reachable: the most these "
            "ingredients can make is 131 g.")
        with pytest.raises(ValueError) as low:
            opt.set_formulation_total(10)
        assert str(low.value) == (
            "A default batch size of 10 g is not reachable: the least these "
            "ingredients can make is 20 g.")
        assert opt.formulation_total is None
        assert opt.quantity_constraints == []

    def test_every_total_the_amounts_reach_stays_feasible_with_the_band(
            self, tmp_path, monkeypatch):
        """The tolerance is centred on the total, so a total inside the
        reachable range always leaves the sum somewhere to land — including
        at both ends of it."""
        opt = self._sample(tmp_path, monkeypatch)
        lowest, highest = opt.total_reach()
        for total in (lowest, 40.0, 75.5, 131.0):
            opt.set_formulation_total(total)
            band = opt.quantity_constraints[0]
            assert band['min'] <= min(highest, total) <= band['max'] or \
                band['min'] <= max(lowest, total) <= band['max']
            # The total itself is a sum the allowed amounts can make, and it
            # is inside the band.
            assert band['min'] <= total <= band['max']
            assert lowest <= total <= highest
        assert highest == 131.0

    def test_a_cold_start_batch_adds_up_to_the_total(self, tmp_path,
                                                     monkeypatch):
        opt = self._sample(tmp_path, monkeypatch)
        opt.set_formulation_total(100)
        rows = opt.ask(n_suggestions=3)
        assert len(rows) == 3
        for row in rows:
            assert sum(row.values()) == pytest.approx(100.0, abs=0.5)

    def test_a_model_chosen_batch_adds_up_to_the_total(self, tmp_path,
                                                       monkeypatch):
        """The same promise once the model is driving: the limit is handed to
        the acquisition optimizer as an inequality pair, not re-applied by
        rejection."""
        opt = self._sample(tmp_path, monkeypatch)
        opt.set_formulation_total(100)
        for i, row in enumerate(opt.ask(n_suggestions=5)):
            opt.tell(row, {"Juiciness": 5.0 + 0.4 * i,
                           "Firmness": 6.5 - 0.3 * i})
        assert opt.X_history
        for row in opt.ask(n_suggestions=2):
            # The acquisition optimizer pushes right up against the band, so
            # the comparison carries a float's width, not a wider tolerance.
            assert sum(row.values()) == pytest.approx(100.0, abs=0.5 + 1e-6)

    def test_clearing_it_takes_the_limit_with_it(self, tmp_path, monkeypatch):
        opt = self._sample(tmp_path, monkeypatch)
        opt.set_formulation_total(100)
        opt.clear_formulation_total()
        assert opt.formulation_total is None
        assert opt.quantity_constraints == []
        # A second clear writes nothing: a rerun must not bump the mtime.
        before = opt.last_saved_at
        opt.clear_formulation_total()
        assert opt.last_saved_at == before

    def test_setting_the_same_total_again_writes_nothing(self, tmp_path,
                                                         monkeypatch):
        opt = self._sample(tmp_path, monkeypatch)
        opt.set_formulation_total(100)
        before = opt.last_saved_at
        opt.set_formulation_total(100)
        assert opt.last_saved_at == before

    def test_a_limit_the_user_types_on_every_ingredient_is_its_own(
            self, tmp_path, monkeypatch):
        """Replacement matches the source as well as the ingredients: a limit
        typed by hand over all eight must not silently take the total away."""
        opt = self._sample(tmp_path, monkeypatch)
        opt.set_formulation_total(100)
        names = [v['name'] for v in opt.variables]
        opt.add_quantity_constraint(names, min_val=None, max_val=120)
        assert len(opt.quantity_constraints) == 2
        assert opt.formulation_total == 100.0
        assert opt._formulation_total_index() is not None

    def test_deleting_the_limit_by_index_takes_the_total_with_it(
            self, tmp_path, monkeypatch):
        opt = self._sample(tmp_path, monkeypatch)
        opt.set_formulation_total(100)
        opt.remove_quantity_constraint(0)
        assert opt.formulation_total is None

    def test_deleting_the_limit_by_index_takes_the_percent_limits_too(
            self, tmp_path, monkeypatch):
        """The third door onto a blanked default batch size. Its two
        siblings take every % of batch size limit with them — there is
        nothing left for one to be a percent OF — and this one set the
        number to None and returned, leaving a percent limit naming a batch
        size the project no longer had and still enforced at yesterday's
        grams."""
        opt = self._sample(tmp_path, monkeypatch)
        opt.set_formulation_total(100)
        opt.add_quantity_constraint(["Salt", "Beet juice powder"],
                                    percent={'min': None, 'max': 3.0})
        assert any(qc.get('percent') for qc in opt.quantity_constraints)
        messages = opt.remove_quantity_constraint(
            opt._formulation_total_index())
        assert opt.formulation_total is None
        assert not any(qc.get('percent') for qc in opt.quantity_constraints)
        assert [kind for kind, _ in messages] == ["warning"]

    def test_an_added_ingredient_is_covered_by_the_total(self, tmp_path,
                                                         monkeypatch):
        opt = self._sample(tmp_path, monkeypatch)
        opt.set_formulation_total(100)
        opt.add_ingredient("Onion powder", 0, 5, unit="g")
        qc = opt.quantity_constraints[opt._formulation_total_index()]
        assert "Onion powder" in qc['ingredients']
        assert opt.formulation_total == 100.0
        assert opt.total_reach() == (20.0, 136.0)

    def test_a_deleted_ingredient_leaves_the_total_over_the_rest(
            self, tmp_path, monkeypatch):
        opt = self._sample(tmp_path, monkeypatch)
        opt.set_formulation_total(100)
        opt.remove_ingredient("Beet juice powder")
        qc = opt.quantity_constraints[opt._formulation_total_index()]
        assert "Beet juice powder" not in qc['ingredients']
        assert len(qc['ingredients']) == 7
        assert opt.formulation_total == 100.0

    def test_a_unit_change_that_splits_the_ingredients_clears_it(
            self, tmp_path, monkeypatch):
        """A sum across units is a number of nothing, so the total goes — and
        the caller is told which number went, in the unit it was written in."""
        opt = self._sample(tmp_path, monkeypatch)
        opt.set_formulation_total(100)
        removed = opt.set_ingredient_unit("Water", "ml")
        assert opt.formulation_total is None
        assert opt.quantity_constraints == []
        gone = [r for r in removed if r.get('source') == 'formulation_total']
        assert len(gone) == 1
        assert gone[0]['reason'] == 'unit'
        assert gone[0]['total'] == 100.0
        assert gone[0]['unit'] == "g"

    def test_amounts_that_can_no_longer_reach_it_clear_it(self, tmp_path,
                                                          monkeypatch):
        opt = self._sample(tmp_path, monkeypatch)
        opt.set_formulation_total(100)
        removed = opt.add_ingredient("Water", 0, 10, unit="g")   # 131 g -> 81 g
        assert opt.formulation_total is None
        gone = [r for r in removed if r.get('source') == 'formulation_total']
        assert [g['reason'] for g in gone] == ['unreachable']
        assert opt.quantity_constraints == []

    def test_it_survives_a_saved_copy(self, tmp_path, monkeypatch):
        opt = self._sample(tmp_path, monkeypatch)
        opt.set_formulation_total(100)
        state = opt.export_json()
        assert state['formulation_total'] == 100.0
        FoodOptimizer.validate_state(state)
        fresh = FoodOptimizer("copy_of_sample")
        fresh.import_json(state)
        assert fresh.formulation_total == 100.0
        assert fresh._formulation_total_index() is not None

    def test_a_file_from_before_the_feature_opens_without_a_total(
            self, tmp_path, monkeypatch):
        opt = self._sample(tmp_path, monkeypatch)
        state = opt.export_json()
        state.pop('formulation_total', None)
        FoodOptimizer.validate_state(state)
        fresh = FoodOptimizer("old_file_total")
        fresh.import_json(state)
        assert fresh.formulation_total is None

    def test_a_malformed_total_is_refused_before_import(self, tmp_path,
                                                        monkeypatch):
        opt = self._sample(tmp_path, monkeypatch)
        state = opt.export_json()
        for bad in ("100", 0, -5, True, ["100"]):
            state['formulation_total'] = bad
            with pytest.raises(ValueError, match="damaged"):
                FoodOptimizer.validate_state(state)
        state['formulation_total'] = None
        FoodOptimizer.validate_state(state)

    def test_the_bench_and_the_records_answer_from_opposite_ends(
            self, tmp_path, monkeypatch):
        """Two accessors, not one. `open_round_size` is what the round on the
        bench is being made to — its own size first, the project's default
        behind it — and `recorded_total` is what a round in the records was
        made to, which nothing typed later can reach back and change.

        `sheet_total` used to answer both from the project's end, which was
        right only while a project default hid tab 2's box altogether."""
        opt = self._sample(tmp_path, monkeypatch)
        assert opt.open_round_size() is None
        opt.set_pending_batch_total(400.0)
        assert opt.open_round_size() == 400.0     # with no default at all
        opt.set_pending_batch_total(None)
        opt.set_formulation_total(100)
        assert opt.open_round_size() == 100.0     # the default behind it
        opt.set_pending_batch_total(400.0)
        assert opt.open_round_size() == 400.0     # the round's own size wins
        # And what the records hold is a different question with a different
        # answer: this round has not been recorded, so it has none.
        assert opt.recorded_total(opt.pending_batch_no) is None
        assert not hasattr(opt, "sheet_total")


class TestWhatEachFormulationIsTrying:
    """0.4.0 §D: every suggested formulation says whether it stays close to
    the best or tries something different, and which amounts carry it."""

    def _opt(self, tmp_path, monkeypatch, name="trying"):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer(name)
        opt.add_ingredient("Water", 0, 100)
        opt.add_ingredient("Wheat gluten", 0, 100)
        opt.add_objective("Firmness", 1.0, goal="max", min_val=0, max_val=10)
        return opt

    def _warm(self, opt):
        """Five results, so the cold start is over, with Formulation 1 — 40 g
        of water — the best of them."""
        opt.tell({"Water": 40.0, "Wheat gluten": 20.0}, {"Firmness": 9.0},
                 formulation_no=1, batch_no=1)
        for i in range(2, 6):
            opt.tell({"Water": 40.0 + i, "Wheat gluten": 20.0},
                     {"Firmness": 3.0}, formulation_no=i, batch_no=1)
        assert opt.best_formulation_no() == 1
        return opt

    def test_fifteen_hundredths_of_the_range_is_still_close_to_the_best(
            self, tmp_path, monkeypatch):
        """The line is at 0.15 of a variable's own allowed range, and 0.15
        itself falls on the near side of it. Water is allowed 0 to 100 g, so
        the move that decides it is 15 g."""
        opt = self._warm(self._opt(tmp_path, monkeypatch))
        assert opt.suggestion_kind({"Water": 55.0, "Wheat gluten": 20.0}) == \
            "close to the best"
        assert opt.suggestion_kind({"Water": 55.1, "Wheat gluten": 20.0}) == \
            "trying something different"
        # Down is the same distance as up.
        assert opt.suggestion_kind({"Water": 24.9, "Wheat gluten": 20.0}) == \
            "trying something different"

    def test_the_kind_is_a_fraction_of_each_variables_own_range(
            self, tmp_path, monkeypatch):
        """Raw numbers are not comparable: 5 of an allowed 0 to 10 is a bold
        move, 5 of an allowed 100 to 200 is not."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_ingredient("Salt", 0, 10)
        self._warm(opt)
        # Salt moves 5 g of an allowed 10 — half its range — while water
        # moves nothing at all.
        assert opt.suggestion_kind(
            {"Water": 40.0, "Wheat gluten": 20.0, "Salt": 5.0}) == \
            "trying something different"
        assert opt.suggestion_kind(
            {"Water": 40.0, "Wheat gluten": 20.0, "Salt": 0.5}) == \
            "close to the best"

    def test_a_fixed_variable_does_not_decide_the_kind(self, tmp_path,
                                                        monkeypatch):
        """A fixed ingredient is at one amount in every new formulation, so
        it cannot be what this round is trying."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_ingredient("Salt", 0, 10)
        self._warm(opt)
        moved = {"Water": 40.0, "Wheat gluten": 20.0, "Salt": 5.0}
        assert opt.suggestion_kind(moved) == "trying something different"
        opt.add_ingredient("Salt", 0, 0)
        assert opt.suggestion_kind(moved) == "close to the best"

    def test_the_cold_start_is_spread_across_the_allowed_amounts(
            self, tmp_path, monkeypatch):
        """Until five formulations have results there is nothing to be close
        to, so no row claims a reason and none lists a change."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.tell({"Water": 40.0, "Wheat gluten": 20.0}, {"Firmness": 9.0},
                 formulation_no=1, batch_no=1)
        recipe = {"Water": 80.0, "Wheat gluten": 20.0}
        assert opt.suggestion_kind(recipe) == "spread across the allowed amounts"
        assert opt.compared_with_text(recipe) == \
            "Spread across the allowed amounts"
        assert opt.compared_with_column() == "Compared with the allowed amounts"

    def test_with_no_best_there_is_nothing_to_compare(self, tmp_path,
                                                      monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        assert opt.best_formulation_no() is None
        assert opt.vs_best_text({"Water": 40.0}) == ""
        assert opt.compared_with_text({"Water": 40.0}) == \
            "Spread across the allowed amounts"

    def test_the_changes_are_the_three_largest_in_their_own_units(
            self, tmp_path, monkeypatch):
        """Three amounts, largest first, each in its own ingredient's unit and
        written to the two decimals a balance works to; the settings follow
        the amounts, and a setting is not an amount, so it is not."""
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("units")
        opt.add_ingredient("Water", 0, 100, unit="ml")
        opt.add_ingredient("Wheat gluten", 0, 100, unit="g")
        opt.add_ingredient("Salt", 0, 10, unit="g")
        opt.add_ingredient("Oil", 0, 50, unit="g")
        opt.add_process_parameter("Cook temperature", 100, 200, unit="°C")
        opt.add_objective("Firmness", 1.0, goal="max", min_val=0, max_val=10)
        best = {"Water": 40.0, "Wheat gluten": 20.0, "Salt": 1.0,
                "Oil": 10.0, "Cook temperature": 180.0}
        opt.tell(best, {"Firmness": 9.0}, formulation_no=1, batch_no=1)
        text = opt.vs_best_text({"Water": 52.0, "Wheat gluten": 17.0,
                                 "Salt": 1.5, "Oil": 10.2,
                                 "Cook temperature": 185.0})
        assert text == ("Water +12.00 ml, Wheat gluten −3.00 g, Salt +0.50 g, "
                        "Cook temperature +5 °C")
        # The minus is the typographic one, so it sits beside a plus.
        assert "-3" not in text

    def test_two_equal_changes_keep_their_set_up_order(self, tmp_path,
                                                       monkeypatch):
        """Same size, same order every time the batch is looked at."""
        opt = self._opt(tmp_path, monkeypatch)
        self._warm(opt)
        assert opt.vs_best_text({"Water": 45.0, "Wheat gluten": 15.0}) == \
            "Water +5.00 g, Wheat gluten −5.00 g"

    def test_the_cell_leads_with_the_kind(self, tmp_path, monkeypatch):
        """'Trying something different · Water +12.00 g' — the kind first,
        capitalised as the first word of the cell, then what carries it."""
        opt = self._warm(self._opt(tmp_path, monkeypatch))
        assert opt.compared_with_text({"Water": 60.0, "Wheat gluten": 20.0}) \
            == "Trying something different · Water +20.00 g"
        assert opt.compared_with_text({"Water": 41.0, "Wheat gluten": 20.0}) \
            == "Close to the best · Water +1.00 g"

    def test_the_batch_table_carries_the_column_and_the_sheet_does_not(
            self, tmp_path, monkeypatch):
        """The table says what each formulation is trying; the CSV sheet is a
        grid to fill in, and a column of prose is not something to weigh."""
        opt = self._warm(self._opt(tmp_path, monkeypatch))
        opt.set_pending_batch([{"Water": 60.0, "Wheat gluten": 20.0},
                               {"Water": 41.0, "Wheat gluten": 20.0}])
        df = opt.batch_frame(opt.pending_batch)
        assert list(df.columns)[-1] == "Compared with Formulation 1"
        assert list(df["Compared with Formulation 1"]) == [
            "Trying something different · Water +20.00 g",
            "Close to the best · Water +1.00 g"]
        # ... and not on the sheets: a workbook's rows are the ingredients
        # to weigh out, and a column of prose is not one of them.
        summary = _book(opt.workbook_bytes(opt.pending_batch, None)).worksheets[0]
        assert not any(isinstance(row[0], str)
                       and row[0].startswith("Compared with")
                       for row in summary.iter_rows(values_only=True))

    def test_a_scaled_table_compares_the_amounts_it_shows(self, tmp_path,
                                                          monkeypatch):
        """A change read off a scaled table has to be the difference between
        two numbers the screen actually shows, so the best is rewritten to
        that same total before the two are compared."""
        opt = self._warm(self._opt(tmp_path, monkeypatch))
        opt.set_pending_batch([{"Water": 52.0, "Wheat gluten": 20.0}])
        df = opt.batch_frame(opt.pending_batch, scale_to=120.0)
        cell = df["Compared with Formulation 1"].iloc[0]
        # Both rewritten to 120 g: 86.67 g of water against 80.00 g, and the
        # gluten the same distance the other way. The KIND is read off the
        # amounts the project stores, because the allowed amounts it is a
        # fraction of are the project's own: 52 g against 40 g is still close.
        assert cell == ("Close to the best · Water +6.67 g, "
                        "Wheat gluten −6.67 g"), cell
        # The frame itself carries the two decimals the screen shows.
        assert df["Water (g)"].iloc[0] == 86.67

    def test_the_settings_are_counted_apart_from_the_amounts(self, tmp_path,
                                                             monkeypatch):
        """Three amounts and three settings, each counted on its own, so
        asking for more amounts does not also lengthen the list of dials."""
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("counts")
        best = {}
        for i in range(4):
            opt.add_ingredient(f"Powder {i}", 0, 100)
            opt.add_process_parameter(f"Dial {i}", 0, 100, unit="rpm")
            best[f"Powder {i}"] = 10.0
            best[f"Dial {i}"] = 10.0
        opt.add_objective("Firmness", 1.0, goal="max", min_val=0, max_val=10)
        opt.tell(best, {"Firmness": 9.0}, formulation_no=1, batch_no=1)
        moved = {k: v + 1.0 for k, v in best.items()}
        names = [part.split(" +")[0]
                 for part in opt.vs_best_text(moved).split(", ")]
        assert names == ["Powder 0", "Powder 1", "Powder 2",
                         "Dial 0", "Dial 1", "Dial 2"], names
        # A wider list of amounts leaves the settings' own count alone.
        names = [part.split(" +")[0]
                 for part in opt.vs_best_text(moved, n=4).split(", ")]
        assert names == ["Powder 0", "Powder 1", "Powder 2", "Powder 3",
                         "Dial 0", "Dial 1", "Dial 2"], names

    def test_the_sheet_line_says_what_it_is_compared_with(self, tmp_path,
                                                          monkeypatch):
        """A sheet carries one formulation and no column header, so the line
        on paper names the formulation the changes are measured from."""
        opt = self._warm(self._opt(tmp_path, monkeypatch))
        recipe = {"Water": 41.0, "Wheat gluten": 20.0}
        assert wording.compared_with_line(opt.compared_with_column(),
                                          opt.compared_with_text(recipe)) == \
            ("Compared with Formulation 1: Close to the best · "
             "Water +1.00 g")


class TestTheTotalIsAlwaysReachable:
    """Fix round 1: a total the box accepts must produce a batch. Rejection
    sampling cannot find an equality on a sum near the ends of what the
    allowed amounts reach, so the opening projects onto the total instead."""

    def _sample(self, tmp_path, monkeypatch, name="reach"):
        """The shipped sample with Water typed by hand: a balance row makes
        every sum the batch size exactly, and what this class is about is
        the ends of a reach the box has to land inside."""
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer(name, robust=False)
        opt.set_amount_unit("g")
        opt.load_ingredients_from_csv(pd.read_csv(_FLAT_BURGER_CSV))
        opt.clear_formula("Water")
        opt.add_ingredient("Water", 20, 60, unit="g")
        opt.add_objective("Juiciness", 1.0, goal="target", target=7,
                          min_val=0, max_val=10, unit="/10")
        opt.add_objective("Firmness", 1.5, goal="target", target=6,
                          min_val=0, max_val=10, unit="/10")
        return opt

    @pytest.mark.parametrize("total", list(range(20, 131, 5)))
    def test_every_total_the_box_accepts_fills_a_batch(self, total, tmp_path,
                                                       monkeypatch):
        opt = self._sample(tmp_path, monkeypatch, name=f"reach_{total}")
        opt.set_formulation_total(total)
        rows = opt.ask(n_suggestions=3)
        assert len(rows) == 3
        bounds = {v['name']: v['bounds'] for v in opt.variables}
        for row in rows:
            assert sum(row.values()) == pytest.approx(float(total), abs=0.5)
            for name, value in row.items():
                low, high = bounds[name]
                assert low - 1e-6 <= value <= high + 1e-6, (name, value)

    def test_a_row_worth_less_than_nothing_moves_the_other_way(
            self, tmp_path, monkeypatch):
        """A free row whose effective coefficient is negative was pinned
        wherever the space-filling point dropped it and never moved, even
        when moving it was the ONLY way onto the total. total_reach and
        set_formulation_total counted it, so the box accepted a size the
        projection then refused for every candidate and Generate raised.

        Filler = 50 − 2 × A makes A worth −1 to the total, so a 40 g
        formulation is A = 10, Filler = 30 and nothing else."""
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("neg_snap", robust=False)
        opt.set_amount_unit("g")
        opt.add_ingredient("A", 5, 20)
        opt.add_ingredient("Filler", 0, 100)
        opt.add_process_parameter("Temp", 100, 200, baseline=150)
        opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
        opt.set_formulation_total(40)
        opt.set_formula("Filler", "= 50 - 2 * A")
        assert opt.total_reach() == (30.0, 45.0)
        # Wherever the point lands, the projection walks A onto the total.
        for started_at in (5.0, 10.0, 12.0, 20.0):
            snapped = opt._snap_to_total({"A": started_at, "Temp": 150}, 40)
            assert snapped is not None
            assert snapped["A"] == pytest.approx(10.0)
            assert snapped["Filler"] == pytest.approx(30.0)
        rows = opt.ask(n_suggestions=3)
        assert len(rows) == 3
        for row in rows:
            assert row["A"] + row["Filler"] == pytest.approx(40.0, abs=0.5)

    def test_a_negative_row_only_takes_what_the_positive_rows_cannot(
            self, tmp_path, monkeypatch):
        """Mixed signs: the rows worth more than nothing go first, so a
        point a one-sided projection already handled comes out of this
        unchanged, and the negative row answers only for the rest."""
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("mixed_snap", robust=False)
        opt.set_amount_unit("g")
        opt.add_ingredient("A", 0, 20)
        opt.add_ingredient("B", 0, 20)
        opt.add_ingredient("Filler", 0, 100)
        opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
        opt.set_formulation_total(50)
        # Filler = 60 − 2A + B, so the sum is 60 − A + 2B: A is worth −1
        # to the total and B worth +2.
        opt.set_formula("Filler", "= 60 - 2 * A + B")
        # 70 g is within what B alone can add, so A stays exactly where the
        # space-filling point put it — this is the one-sided answer, kept.
        snapped = opt._snap_to_total({"A": 4.0, "B": 10.0}, 70)
        assert snapped["A"] == pytest.approx(4.0)
        assert snapped["B"] == pytest.approx(7.0)
        assert sum(snapped.values()) == pytest.approx(70.0)
        # 100 g is more than B can add on its own, so A comes down for the
        # rest of it rather than the projection giving up.
        snapped = opt._snap_to_total({"A": 4.0, "B": 10.0}, 100)
        assert snapped is not None
        assert snapped["B"] == pytest.approx(20.0)
        assert snapped["A"] == pytest.approx(0.0)
        assert sum(snapped.values()) == pytest.approx(100.0)

    def test_a_warm_batch_lands_exactly_on_the_total(self, tmp_path,
                                                      monkeypatch):
        """The limit the total is enforced as is a band of ±0.5 %, so the
        model could choose 99.5 g where the project says 100 — and the sheets
        then carried an amount the caution had to apologise for. A warm batch
        is projected onto the total, exactly as the opening is."""
        opt = self._sample(tmp_path, monkeypatch, name="warm_total")
        opt.set_formulation_total(100)
        for row in opt.ask(n_suggestions=5):
            opt.tell(row, {"Juiciness": 6.0, "Firmness": 6.0})
        assert len(opt.X_history) == 5          # the cold start is over
        rows = opt.ask(n_suggestions=3)
        bounds = {v['name']: v['bounds'] for v in opt.variables}
        for row in rows:
            assert sum(row.values()) == pytest.approx(100.0, abs=1e-6)
            for name, value in row.items():
                low, high = bounds[name]
                assert low - 1e-6 <= value <= high + 1e-6, (name, value)
        # Nothing to caution about: the sheets are the total the box says.
        assert opt.scaled_caution(rows, opt.open_round_size()) == ""

    def test_a_snapped_suggestion_still_keeps_every_limit(self, tmp_path,
                                                          monkeypatch):
        """Moving a warm suggestion onto the total moves amounts, and an
        amount limit the user wrote is exactly what that can break: the
        batch came back with 20.32 g of protein under a limit of 20, and
        nothing on screen said so. The projected row is used only while it
        still keeps every limit; otherwise the optimiser's own row stands,
        inside the ±0.5 % band."""
        opt = self._sample(tmp_path, monkeypatch, name="warm_limit")
        opt.set_formulation_total(100)
        opt.add_quantity_constraint(["Pea protein isolate", "Wheat gluten"],
                                    max_val=20)
        for row in opt.ask(n_suggestions=5):
            opt.tell(row, {"Juiciness": 6.0, "Firmness": 6.0})
        rows = opt.ask(n_suggestions=3)
        assert len(rows) == 3
        for row in rows:
            assert (row["Pea protein isolate"] + row["Wheat gluten"]
                    <= 20 + 1e-9), row
            assert opt._check_constraints(row), row
            # Still the total the box says, within the band the limit is
            # enforced as.
            assert sum(row.values()) == pytest.approx(100.0, rel=0.005)

    def test_a_projection_that_would_break_a_limit_is_not_taken(
            self, tmp_path, monkeypatch):
        """The branch itself, without a model in the way: projecting this
        formulation onto 100 g pushes Water past the 20 g it is allowed, so
        the row is left where the optimiser put it — inside the band, and
        the sheets carry the caution."""
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("snap_limit", robust=False)
        opt.set_amount_unit("g")
        opt.add_ingredient("Water", 0, 100)
        opt.add_ingredient("Flour", 0, 100)
        opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
        opt.set_formulation_total(100)
        opt.add_quantity_constraint(["Water"], max_val=20)
        kept = {"Water": 20.0, "Flour": 70.0}
        # Unguarded, the projection is what it would have used.
        snapped = opt._snap_to_total(kept, 100.0)
        assert sum(snapped.values()) == pytest.approx(100.0)
        assert snapped["Water"] > 20, snapped
        assert not opt._check_constraints(snapped)
        assert opt._snapped_if_it_still_fits(kept) == kept
        # A formulation the projection does not push over is still moved.
        fits = {"Water": 5.0, "Flour": 90.0}
        moved = opt._snapped_if_it_still_fits(fits)
        assert sum(moved.values()) == pytest.approx(100.0)
        assert moved["Water"] <= 20 + 1e-9

    def test_a_locked_recipe_that_adds_up_generates(self, tmp_path,
                                                    monkeypatch):
        """Every ingredient fixed, the fixed amounts adding up to the batch
        size, and one process setting still varying: a real project — the
        recipe is locked and the round varies the oven. The projection has
        nothing to move, so it hands back the one formulation the project
        describes rather than refusing it."""
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("locked", robust=False)
        opt.set_amount_unit("g")
        opt.add_ingredient("Water", 60, 60)
        opt.add_ingredient("Flour", 40, 40)
        opt.add_process_parameter("Oven", 150, 200)
        opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
        opt.set_formulation_total(100)
        snapped = opt._snap_to_total({"Water": 0.0, "Flour": 0.0}, 100.0)
        assert snapped["Water"] == 60.0 and snapped["Flour"] == 40.0
        rows = opt.ask(n_suggestions=2)
        assert len(rows) == 2
        for row in rows:
            assert row["Water"] == pytest.approx(60.0)
            assert row["Flour"] == pytest.approx(40.0)
        # The amounts are identical, which is the point; the oven is what
        # the round varies.
        assert len({round(row["Oven"], 3) for row in rows}) == 2, rows

    def test_a_locked_recipe_that_misses_the_size_names_both_numbers(
            self, tmp_path, monkeypatch):
        """Nothing to widen and nothing to move, so the refusal is the two
        numbers: what the fixed amounts make, and what was asked for."""
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("locked_short", robust=False)
        opt.set_amount_unit("g")
        opt.add_ingredient("Water", 0, 100)
        opt.add_ingredient("Flour", 0, 100)
        opt.add_process_parameter("Oven", 150, 200)
        opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
        opt.set_formulation_total(100)
        # Fixing both rows through the app is refused while the size is out
        # of their reach (fixing_breaks_the_total), so the state is written
        # the way a copy saved before that guard existed carries it.
        opt._var_by_name("Water")['bounds'] = (60.0, 60.0)
        opt._var_by_name("Flour")['bounds'] = (35.0, 35.0)
        assert opt.fixed_ingredient_total() == 95.0
        assert opt._snap_to_total({"Water": 60.0, "Flour": 35.0}, 100.0) is None
        with pytest.raises(ValueError) as caught:
            opt.ask(n_suggestions=1)
        assert str(caught.value) == ("The fixed amounts add up to 95 g, not "
                                     "the 100 g batch size.")

    def test_the_opening_is_still_spread_out(self, tmp_path, monkeypatch):
        """Projecting onto the total must not collapse the design: three rows
        at a total the box has room around are three different formulations."""
        opt = self._sample(tmp_path, monkeypatch)
        opt.set_formulation_total(100)
        rows = opt.ask(n_suggestions=3)
        waters = {round(row["Water"], 3) for row in rows}
        assert len(waters) == 3, waters

    def test_a_fixed_ingredient_keeps_its_one_amount(self, tmp_path,
                                                     monkeypatch):
        """A fixed ingredient takes no part in the projection: its amount
        comes off the target first."""
        opt = self._sample(tmp_path, monkeypatch)
        opt.set_formulation_total(100)
        opt.add_ingredient("Salt", 2.0, 2.0)
        for row in opt.ask(n_suggestions=2):
            assert row["Salt"] == pytest.approx(2.0)
            assert sum(row.values()) == pytest.approx(100.0, abs=0.5)

    def test_fixing_that_puts_the_total_out_of_reach_names_the_total(
            self, tmp_path, monkeypatch):
        """Not nine grid rows and a limit the user never wrote."""
        opt = self._sample(tmp_path, monkeypatch)
        opt.set_formulation_total(120)
        # Water is 20 to 60 g; fixed at 20, the rest reach 91 g at most.
        with pytest.raises(ValueError) as refused:
            opt.add_ingredient("Water", 20.0, 20.0)
        assert str(refused.value) == (
            "Fixing these would leave no formulation adding up to 120 g. "
            "Change the batch size, or let enough ingredients vary again. "
            "Fixed at one amount: Water.")

    def test_a_total_of_nothing_is_refused_in_the_reach_words(self, tmp_path,
                                                              monkeypatch):
        """A project whose every amount can be 0 reaches 0, so the reach
        sentence alone let a total of nothing through to the band, which
        refused it by naming two boxes the user never saw."""
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("zeroes", robust=False)
        opt.add_ingredient("Water", 0, 100)
        opt.add_ingredient("Flour", 0, 50)
        assert opt.total_reach() == (0.0, 150.0)
        with pytest.raises(ValueError) as refused:
            opt.set_formulation_total(0)
        assert str(refused.value) == (
            "A default batch size of 0 g is not reachable: every formulation has to add "
            "up to something.")
        assert opt.formulation_total is None
        assert opt.quantity_constraints == []

    def test_the_total_keeps_its_place_in_the_limits_list(self, tmp_path,
                                                          monkeypatch):
        """A limit that jumped to the bottom of the list every time the
        ingredient list was touched read as a new limit."""
        opt = self._sample(tmp_path, monkeypatch)
        opt.set_formulation_total(100)
        opt.add_quantity_constraint(["Salt"], max_val=2)
        assert [qc.get('source') for qc in opt.quantity_constraints] == [
            'formulation_total', None]
        opt.add_ingredient("Onion powder", 0, 5, unit="g")
        assert [qc.get('source') for qc in opt.quantity_constraints] == [
            'formulation_total', None]
        assert "Onion powder" in opt.quantity_constraints[0]['ingredients']

    def test_deleting_an_ingredient_hands_back_the_total_that_went(
            self, tmp_path, monkeypatch):
        """add_ingredient already did; the screen owes the same notice
        either way, and could not say it without this."""
        opt = self._sample(tmp_path, monkeypatch)
        opt.set_formulation_total(130)       # only 131 g of amounts allowed
        removed = opt.remove_ingredient("Water")    # 131 g -> 71 g
        assert opt.formulation_total is None
        gone = [r for r in removed if r.get('source') == 'formulation_total']
        assert [g['reason'] for g in gone] == ['unreachable']
        assert gone[0]['total'] == 130.0

    def test_deleting_an_ingredient_the_total_survives_returns_nothing(
            self, tmp_path, monkeypatch):
        opt = self._sample(tmp_path, monkeypatch)
        opt.set_formulation_total(100)
        assert opt.remove_ingredient("Beet juice powder") == []
        assert opt.formulation_total == 100.0

    def test_a_batch_records_the_total_its_sheets_were_printed_to(
            self, tmp_path, monkeypatch):
        """Tab 2 draws no box while the project has a total, so
        pending_batch_total is blank — and the bench still weighed out 100 g."""
        opt = self._sample(tmp_path, monkeypatch)
        opt.set_formulation_total(100)
        rows = opt.ask(n_suggestions=2)
        batch_no = opt.pending_batch_no
        assert opt.pending_batch_total is None
        opt.tell(rows[0], {"Juiciness": 7.0, "Firmness": 6.0},
                 formulation_no=1, batch_no=batch_no)
        assert opt.batch_total(batch_no) == 100.0

    def test_a_recorded_batch_keeps_the_total_it_was_made_to(self, tmp_path,
                                                             monkeypatch):
        """A total typed on tab 1 afterwards must not rewrite what the bench
        already weighed out."""
        opt = self._sample(tmp_path, monkeypatch)
        rows = opt.ask(n_suggestions=1)
        batch_no = opt.pending_batch_no
        opt.set_pending_batch_total(400.0)
        opt.tell(rows[0], {"Juiciness": 7.0, "Firmness": 6.0},
                 formulation_no=1, batch_no=batch_no)
        assert opt.recorded_total(batch_no) == 400.0
        opt.set_formulation_total(100)
        assert opt.recorded_total(batch_no) == 400.0     # still 400 g
        opt.set_pending_batch(None)
        assert opt.recorded_total(batch_no) == 400.0     # and still 400 g

    def test_a_batch_from_before_totals_stays_as_generated(
            self, tmp_path, monkeypatch):
        """A batch recorded before the total was stored was made to no total,
        and a number typed on tab 1 today cannot reach back and claim it."""
        opt = self._sample(tmp_path, monkeypatch)
        assert opt.recorded_total(7) is None
        opt.set_formulation_total(100)
        assert opt.recorded_total(7) is None


# ------------------------------------------------------------------ #
#  0.4.0 §G: the workbook. One file carries the batch to the bench and
#  the results back, and one file carries the whole project away.
# ------------------------------------------------------------------ #
def _book(data):
    """An .xlsx in memory, opened."""
    return openpyxl.load_workbook(io.BytesIO(data))


def _rows(sheet):
    """One sheet as a list of row tuples."""
    return list(sheet.iter_rows(values_only=True))


def _labelled(sheet):
    """A sheet as {first-column label: [the rest of the row]}."""
    return {row[0]: list(row[1:]) for row in _rows(sheet)
            if isinstance(row[0], str)}


class TestTheWorkbook:
    """The batch leaves the app as one Excel file: a summary sheet a whole
    batch is weighed out from and written back onto, and one sheet per
    formulation to print, carry and tick."""

    def _opt(self, tmp_path, monkeypatch, name="sheets"):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer(name)
        for field in ("vendor", "sku", "lot", "actual"):
            opt.set_records(field, True)
        opt.set_amount_unit("g")
        opt.add_ingredient("Pea protein", 0, 100)
        opt.add_ingredient("Water", 0, 100)
        opt.add_ingredient("Salt", 0, 10)
        opt.add_process_parameter("Cook temperature", 100, 220, unit="°C")
        opt.add_objective("Firmness", 1.5, goal="target", target=6,
                          min_val=0, max_val=10, unit="N")
        opt.add_objective("Juiciness", 1.0, goal="max", min_val=0, max_val=10,
                          unit="/10")
        opt.set_pending_batch([
            {"Pea protein": 20.0, "Water": 70.0, "Salt": 10.0,
             "Cook temperature": 180.0},
            {"Pea protein": 30.0, "Water": 65.0, "Salt": 5.0,
             "Cook temperature": 188.4936},
            {"Pea protein": 25.0, "Water": 70.0, "Salt": 5.0,
             "Cook temperature": 200.0},
        ], batch_no=2)
        return opt

    def test_the_file_opens_and_names_a_sheet_for_the_batch_and_for_each_row(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        book = _book(opt.workbook_bytes(opt.pending_batch, 100.0))
        assert [s.title for s in book if s.sheet_state == "visible"] == ["Round 2", "Formulation 1",
                                   "Formulation 2", "Formulation 3"], \
            book.sheetnames

    def test_the_summary_carries_the_amounts_their_share_and_the_total(
            self, tmp_path, monkeypatch):
        """One column per formulation, one row per ingredient, and a `%`
        beside every amount: the sheet a whole batch is weighed out from."""
        opt = self._opt(tmp_path, monkeypatch)
        sheet = _book(opt.workbook_bytes(opt.pending_batch, 100.0))["Round 2"]
        rows = _rows(sheet)
        # The title line says which batch of which project this is and when
        # it was asked for; the header the upload reads is under it.
        # The title line says which batch of which project this is, when it
        # was asked for and what size it is made to.
        assert rows[0][0] == wording.summary_title(2, "sheets",
                                                   opt._sheet_date(), "100 g")
        # Under the title, the one line that says which cells can be
        # written in — this sheet's own, the Lot among them, and not the
        # Actual cells that live on the pages; the header the upload reads
        # is under that.
        assert rows[1][0] == wording.SUMMARY_SHADED_NOTE
        assert wording.LOT_COLUMN in wording.SUMMARY_SHADED_NOTE
        assert wording.ACTUAL_COLUMN not in wording.SUMMARY_SHADED_NOTE
        # It names the column by the word the block's own header row uses.
        # It said "Measured", which is a header this sheet does not write —
        # the formulation pages do, and their line names it.
        head = next(r for r in rows
                    if r and r[0] == wording.MEASUREMENT_COLUMN)
        assert head[0] in wording.SUMMARY_SHADED_NOTE, head
        assert wording.MEASURED_COLUMN not in wording.SUMMARY_SHADED_NOTE
        assert wording.MEASURED_COLUMN in wording.sheet_write_in_note()
        pct = wording.PERCENT_COLUMN
        assert rows[2] == ("Ingredient or process setting", "Formulation 1",
                           pct, "Formulation 2", pct, "Formulation 3", pct,
                           "Lot"), rows[2]
        labelled = _labelled(sheet)
        # The last cell of an ingredient row is its Lot, left blank for the
        # bench to write in.
        assert labelled["Pea protein (g)"] == [20.0, 20.0, 30.0, 30.0,
                                               25.0, 25.0, None]
        assert labelled["Salt (g)"] == [10.0, 10.0, 5.0, 5.0, 5.0, 5.0, None]
        # The total is the number the sheets were written to, and every
        # column of shares adds up to it.
        assert labelled["Total (g)"] == [100.0, 100.0, 100.0, 100.0,
                                         100.0, 100.0, None]
        # A setting is dialled in, not weighed: no share, no lot, and
        # rounded as every screen rounds it.
        assert labelled["Cook temperature (°C)"] == [180.0, None, 188.49,
                                                     None, 200.0, None, None]

    def test_the_summary_has_a_row_to_write_in_for_every_measurement(
            self, tmp_path, monkeypatch):
        """Each measurement says what a good number looks like and leaves an
        empty cell under every formulation, then the tick and the note."""
        opt = self._opt(tmp_path, monkeypatch)
        sheet = _book(opt.workbook_bytes(opt.pending_batch, 100.0))["Round 2"]
        labels = list(_labelled(sheet))
        # The block is headed in the word both kinds of sheet use for it,
        # the formulation names are repeated directly above the cells they
        # head, the measurements are in importance order, and the last two
        # lines say which of the two ways of filling a column in wins and
        # who made it.
        assert labels[-9:] == ["Measurements", wording.SHEET_WRITE_IN_NOTE,
                               "Measurement",
                               "Firmness (N) · Target 6 N",
                               "Juiciness (/10) · Prefer higher values",
                               "Not scored", "Note",
                               wording.SUMMARY_TICK_NOTE,
                               wording.MADE_BY_FOOTER], labels
        assert wording.SUMMARY_TICK_NOTE == (
            "A ticked Not scored box wins over numbers typed in that column.")
        # The measurement rows open empty; the Not scored row opens holding
        # the box the instruction asks the reader to tick, in the cell the
        # pen can reach.
        for label in ("Firmness (N) · Target 6 N",
                      "Juiciness (/10) · Prefer higher values"):
            assert _labelled(sheet)[label] == [None] * 7, label
        assert _labelled(sheet)["Not scored"] == [
            wording.TICK_BOX, None, wording.TICK_BOX, None,
            wording.TICK_BOX, None, None]

    def test_a_note_the_app_wrote_is_already_on_the_sheet(self, tmp_path,
                                                          monkeypatch):
        """A repeat of the best formulation says so on the page, or it is two
        identical bowls with nothing to tell them apart."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_to_pending_batch({"Pea protein": 20.0, "Water": 70.0,
                                  "Salt": 10.0, "Cook temperature": 180.0},
                                 note=wording.repeat_of_formulation(1))
        book = _book(opt.workbook_bytes(opt.pending_batch, 100.0))
        # Formulation 4's own column: the amount columns come in pairs, and
        # the Lot column is past the end of them.
        assert _labelled(book["Round 2"])[wording.NOTE][(4 - 1) * 2] == \
            wording.repeat_of_formulation(1)
        assert wording.repeat_of_formulation(1) in \
            [v for row in _rows(book["Formulation 4"]) for v in row]

    def test_a_formulation_sheet_is_the_page_the_bench_carries(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        sheet = _book(opt.workbook_bytes(opt.pending_batch,
                                         100.0))["Formulation 1"]
        rows = _rows(sheet)
        assert rows[0][0] == "Formulation 1 · Round 2 · sheets"
        # The round's own search strategy is stated once, on the Round
        # sheet. On the page a bench carries it was a line to read and then
        # discard, above the amounts it has to weigh.
        assert rows[1][0] is None
        # Then the one line that says which cells the sheet will take,
        # naming the Actual column as this page's own header writes it.
        assert rows[2][0] == wording.sheet_write_in_note("Actual (g)")
        # A column to tick as each ingredient goes in, in set-up order, and
        # a column to write what the balance actually said.
        assert rows[4] == ("Tick", "Ingredient", "Amount (g)", "Actual (g)",
                           wording.PERCENT_COLUMN), rows[4]
        assert [row[1] for row in rows[5:8]] == ["Pea protein", "Water",
                                                 "Salt"]
        assert rows[5][2] == 20.0 and rows[5][4] == 20.0
        assert rows[5][3] is None        # Actual: blank means as printed
        assert (rows[8][1], rows[8][2], rows[8][4]) == ("Total", 100.0, 100.0)
        text = [v for row in rows for v in row if isinstance(v, str)]
        assert wording.SETTINGS_SHEET_HEADING in text, text
        assert "Cook temperature (°C)" in text, text
        assert wording.MEASURED_COLUMN in text, text
        assert "Target 6 N" in text, text
        assert wording.NOT_SCORED_CHECKBOX_SHEET in text, text
        # Whose work it was, on the page that comes back a week later.
        assert text[-1] == wording.MADE_BY_FOOTER, text

    def test_only_a_write_in_cell_is_shaded_on_any_sheet(self, tmp_path,
                                                         monkeypatch):
        """There used to be eight pastel bands down the ingredient rows, and
        an instruction line reading "fill in the shaded cells only" over
        twenty shaded cells that were locked. On the black-and-white
        printer a bench sheet actually goes through, the writable cream and
        the pastel bands came out the same grey."""
        opt = self._opt(tmp_path, monkeypatch)
        book = _book(opt.workbook_bytes(opt.pending_batch, 100.0))
        for name in book.sheetnames:
            sheet = book[name]
            # A merged box is ONE cell to the reader and to the pen: Excel
            # paints the whole of it in the anchor's shade, and openpyxl
            # keeps no style on the cells the merge swallowed. The anchor is
            # the cell this rule is about.
            swallowed = {c for rng in sheet.merged_cells.ranges
                         for row in sheet[rng.coord] for c in row[1:]}
            for row in sheet.iter_rows():
                for cell in row:
                    if cell in swallowed:
                        continue
                    shaded = cell.fill.fgColor.rgb not in (None, "00000000")
                    writable = cell.protection.locked is False
                    assert shaded == writable, (name, cell.coordinate,
                                                cell.fill.fgColor.rgb)
                    if writable:
                        assert cell.border.left.style == "thin", cell.coordinate

    def test_the_share_is_of_the_total_the_sheet_was_written_to(
            self, tmp_path, monkeypatch):
        """No total typed: each formulation's share is of its own sum, and
        the total row says what that sum is."""
        opt = self._opt(tmp_path, monkeypatch)
        sheet = _book(opt.workbook_bytes(opt.pending_batch, None))["Round 2"]
        labelled = _labelled(sheet)
        assert labelled["Total (g)"] == [100.0, 100.0, 100.0, 100.0,
                                         100.0, 100.0, None]
        # Scaled to 200 g, the amounts double and the shares do not move.
        sheet = _book(opt.workbook_bytes(opt.pending_batch, 200.0))["Round 2"]
        labelled = _labelled(sheet)
        assert labelled["Pea protein (g)"] == [40.0, 20.0, 60.0, 30.0,
                                               50.0, 25.0, None]
        assert labelled["Total (g)"] == [200.0, 100.0, 200.0, 100.0,
                                         200.0, 100.0, None]

    def test_the_sheets_are_set_up_to_print(self, tmp_path, monkeypatch):
        """A sheet that prints its last two ingredients on a second page is
        a sheet the bench weighs out wrong."""
        opt = self._opt(tmp_path, monkeypatch)
        sheet = _book(opt.workbook_bytes(opt.pending_batch,
                                         100.0))["Formulation 1"]
        assert sheet.page_setup.orientation == "portrait"
        assert sheet.sheet_properties.pageSetUpPr.fitToPage is True
        assert sheet.page_setup.fitToWidth == 1
        assert sheet.print_area == "'Formulation 1'!$A$1:$E$24"
        assert sheet.column_dimensions["B"].width == 34

    # -------------------------- and back again -------------------------- #

    def _filled_in(self, opt, total=100.0):
        """The workbook as it comes back from the bench: two formulations
        measured, the third ticked Not scored with a reason."""
        book = _book(opt.workbook_bytes(opt.pending_batch, total))
        sheet = book[wording.batch_sheet_name(opt.pending_batch_no)]
        labels = [sheet.cell(row=r, column=1).value
                  for r in range(1, sheet.max_row + 1)]
        firm = labels.index("Firmness (N) · Target 6 N") + 1
        juice = labels.index("Juiciness (/10) · Prefer higher values") + 1
        not_scored = labels.index(wording.NOT_SCORED_CHECKBOX_SHEET) + 1
        note = labels.index(wording.NOTE) + 1
        sheet.cell(row=firm, column=2, value=5.5)
        sheet.cell(row=juice, column=2, value=7)
        sheet.cell(row=firm, column=4, value=6.5)
        sheet.cell(row=juice, column=4, value=6)
        sheet.cell(row=not_scored, column=6, value="x")
        sheet.cell(row=note, column=6, value="burner failed")
        out = io.BytesIO()
        book.save(out)
        out.seek(0)
        return out

    def test_a_filled_in_workbook_comes_back_as_results_and_a_ticked_row(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        frame = opt.results_from_workbook(self._filled_in(opt)).frame
        assert list(frame["Formulation"]) == [1, 2, 3]
        parsed, skipped = opt.parse_batch_results(frame, opt.pending_batch,
                                                  with_skipped=True)
        assert parsed == [(1, {"Firmness": 5.5, "Juiciness": 7.0}, ""),
                          (2, {"Firmness": 6.5, "Juiciness": 6.0}, "")]
        # The tick is not a row missing its numbers: it is a formulation
        # nobody scored, and the reason typed beside it comes with it.
        assert skipped == [(3, "burner failed")]
        # The old shape still answers the old way.
        assert opt.parse_batch_results(frame, opt.pending_batch) == parsed

    def test_a_column_nobody_touched_is_not_a_row_to_refuse(self, tmp_path,
                                                            monkeypatch):
        """Half a batch measured today and the rest tomorrow is how a bench
        works; an empty column is not a sheet filled in wrongly."""
        opt = self._opt(tmp_path, monkeypatch)
        book = _book(opt.workbook_bytes(opt.pending_batch, 100.0))
        sheet = book["Round 2"]
        labels = [sheet.cell(row=r, column=1).value
                  for r in range(1, sheet.max_row + 1)]
        sheet.cell(row=labels.index("Firmness (N) · Target 6 N") + 1, column=2,
                   value=5.5)
        out = io.BytesIO()
        book.save(out)
        out.seek(0)
        frame = opt.results_from_workbook(out).frame
        assert list(frame["Formulation"]) == [1]

    def test_last_weeks_workbook_says_which_sheet_it_wanted(self, tmp_path,
                                                            monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        stale = opt.workbook_bytes(opt.pending_batch, 100.0)
        opt.set_pending_batch([{"Pea protein": 20.0, "Water": 70.0,
                                "Salt": 10.0, "Cook temperature": 180.0}],
                              batch_no=3)
        with pytest.raises(ValueError, match="different round"):
            opt.results_from_workbook(io.BytesIO(stale))

    def test_a_sheet_with_no_formulation_columns_is_refused(self, tmp_path,
                                                            monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        book = openpyxl.Workbook()
        book.active.title = "Round 2"
        book.active.cell(row=1, column=1, value="Ingredient")
        book.active.cell(row=1, column=2, value="Something else")
        book.active.cell(row=2, column=1, value="Pea protein (g)")
        out = io.BytesIO()
        book.save(out)
        out.seek(0)
        with pytest.raises(ValueError, match="no formulation columns"):
            opt.results_from_workbook(out)

    def test_a_workbook_with_nothing_written_on_it_says_so(self, tmp_path,
                                                           monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        with pytest.raises(ValueError, match="Nothing is filled in"):
            opt.results_from_workbook(
                io.BytesIO(opt.workbook_bytes(opt.pending_batch, 100.0)))

    def test_something_that_is_not_a_workbook_at_all(self, tmp_path,
                                                     monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        with pytest.raises(ValueError, match="could not be read as a workbook"):
            opt.results_from_workbook(io.BytesIO(b"not a workbook"))

    def _old_layout(self, opt, extra_rows=()):
        """A workbook as the app wrote them before the summary had a title
        line or a Measured heading: the header IS row 1. `extra_rows` are
        (label, values) pairs written among the ingredients — an ingredient
        named `Not scored` is what a project saved before that name was
        reserved can still hold."""
        book = openpyxl.Workbook()
        sheet = book.active
        sheet.title = wording.batch_sheet_name(opt.pending_batch_no)
        numbers = [row['formulation'] for row in opt.pending_batch]
        sheet.cell(row=1, column=1, value=wording.KIND_INGREDIENT)
        for j, number in enumerate(numbers):
            sheet.cell(row=1, column=2 + 2 * j,
                       value=wording.formulation_sheet_name(number))
            sheet.cell(row=1, column=3 + 2 * j, value=wording.PERCENT_COLUMN)
        r = 2
        for label, values in list(extra_rows) + [("Pea protein (g)", [20, 30, 25])]:
            sheet.cell(row=r, column=1, value=label)
            for j, value in enumerate(values):
                sheet.cell(row=r, column=2 + 2 * j, value=value)
            r += 1
        sheet.cell(row=r, column=1, value="Total (g)")
        r += 2
        for obj in opt.measurements_by_importance():
            sheet.cell(row=r, column=1,
                       value=opt._measurement_sheet_label(obj))
            r += 1
        sheet.cell(row=r, column=1, value=wording.NOT_SCORED)
        not_scored_row = r
        r += 1
        sheet.cell(row=r, column=1, value=wording.NOTE)
        return book, not_scored_row

    def test_a_workbook_from_before_the_title_line_still_imports(
            self, tmp_path, monkeypatch):
        """The summary opens with a title line now, so the header is row 2 —
        but a bench filling in the file it downloaded last month must not be
        told its work is unreadable. The header is found, not counted to."""
        opt = self._opt(tmp_path, monkeypatch)
        book, _ = self._old_layout(opt)
        sheet = book[wording.batch_sheet_name(2)]
        labels = [sheet.cell(row=r, column=1).value
                  for r in range(1, sheet.max_row + 1)]
        sheet.cell(row=labels.index("Firmness (N) · Target 6 N") + 1, column=2,
                   value=5.5)
        out = io.BytesIO()
        book.save(out)
        out.seek(0)
        frame = opt.results_from_workbook(out).frame
        assert list(frame["Formulation"]) == [1]
        assert frame["Firmness"].iloc[0] == 5.5
        # ... and the file the app writes today reads back the same way.
        today = opt.results_from_workbook(self._filled_in(opt)).frame
        assert list(today["Formulation"]) == [1, 2, 3]

    def test_an_ingredient_called_not_scored_cannot_hijack_the_tick(
            self, tmp_path, monkeypatch):
        """The tick and the note are read by position under the last
        measurement row, not by the first label that matches. An ingredient
        of that name put its amount on the tick row, and the whole column
        came back as a formulation nobody scored."""
        opt = self._opt(tmp_path, monkeypatch)
        # The name is reserved now...
        with pytest.raises(ValueError, match="column name Food Optimizer"):
            opt.add_ingredient(wording.NOT_SCORED, 0, 10)
        with pytest.raises(ValueError, match="column name Food Optimizer"):
            opt.add_ingredient(wording.MEASURED_COLUMN, 0, 10)
        # ... but a project saved before it was still has the row.
        book, _ = self._old_layout(opt, extra_rows=[(wording.NOT_SCORED,
                                                     [5, 5, 5])])
        sheet = book[wording.batch_sheet_name(2)]
        labels = [sheet.cell(row=r, column=1).value
                  for r in range(1, sheet.max_row + 1)]
        sheet.cell(row=labels.index("Firmness (N) · Target 6 N") + 1, column=2,
                   value=5.5)
        out = io.BytesIO()
        book.save(out)
        out.seek(0)
        frame = opt.results_from_workbook(out).frame
        assert list(frame["Formulation"]) == [1]
        assert frame["Firmness"].iloc[0] == 5.5
        assert frame[wording.NOT_SCORED].iloc[0] == "", frame.to_dict()

    def test_the_formulation_sheets_are_read_when_the_summary_is_empty(
            self, tmp_path, monkeypatch):
        """A bench that prints one sheet per bowl writes on the sheet in its
        hand. The summary is where the upload looks first; the sheets are
        where it looks when nothing was written there."""
        opt = self._opt(tmp_path, monkeypatch)
        book = _book(opt.workbook_bytes(opt.pending_batch, 100.0))
        sheet = book[wording.formulation_sheet_name(2)]
        for r in range(1, sheet.max_row + 1):
            label = sheet.cell(row=r, column=2).value
            if label == "Firmness (N)":
                sheet.cell(row=r, column=4, value=6.5)
            elif label == "Juiciness (/10)":
                sheet.cell(row=r, column=4, value=7.0)
        third = book[wording.formulation_sheet_name(3)]
        for r in range(1, third.max_row + 1):
            if third.cell(row=r, column=2).value == wording.NOT_SCORED_CHECKBOX_SHEET:
                third.cell(row=r, column=4, value="x")
            elif third.cell(row=r, column=2).value == wording.NOTE:
                third.cell(row=r, column=3, value="burner failed")
        out = io.BytesIO()
        book.save(out)
        out.seek(0)
        frame = opt.results_from_workbook(out).frame
        assert list(frame["Formulation"]) == [2, 3]
        parsed, skipped = opt.parse_batch_results(frame, opt.pending_batch,
                                                  with_skipped=True)
        assert parsed == [(2, {"Firmness": 6.5, "Juiciness": 7.0}, "")]
        assert skipped == [(3, "burner failed")]

    def test_a_row_inserted_in_the_block_does_not_shift_the_tick(
            self, tmp_path, monkeypatch):
        """A technician adds a line of their own between the measurements
        and the tick. Counting positions read the inserted row as the tick
        and the tick as the note; the labels are what the rows are found
        by."""
        opt = self._opt(tmp_path, monkeypatch)
        book = openpyxl.load_workbook(self._filled_in(opt))
        sheet = book[wording.batch_sheet_name(2)]
        labels = [sheet.cell(row=r, column=1).value
                  for r in range(1, sheet.max_row + 1)]
        at = labels.index(wording.NOT_SCORED_CHECKBOX_SHEET) + 1
        sheet.insert_rows(at)
        sheet.cell(row=at, column=1, value="Panel notes")
        out = io.BytesIO()
        book.save(out)
        out.seek(0)
        frame = opt.results_from_workbook(out).frame
        parsed, skipped = opt.parse_batch_results(frame, opt.pending_batch,
                                                  with_skipped=True)
        assert parsed == [(1, {"Firmness": 5.5, "Juiciness": 7.0}, ""),
                          (2, {"Firmness": 6.5, "Juiciness": 6.0}, "")]
        assert skipped == [(3, "burner failed")]

    def test_a_sheet_whose_measurement_rows_are_all_unreadable_is_refused(
            self, tmp_path, monkeypatch):
        """The tick and the note are read from under the last measurement
        row. With no measurement row found at all there is nothing to count
        from, and reading the first ingredient's amount as a ticked box would
        lose the whole batch to it."""
        opt = self._opt(tmp_path, monkeypatch)
        book, _ = self._old_layout(opt)
        sheet = book[wording.batch_sheet_name(2)]
        for r in range(1, sheet.max_row + 1):
            label = str(sheet.cell(row=r, column=1).value or "")
            if label.startswith(("Firmness", "Juiciness")):
                sheet.cell(row=r, column=1, value="Panel " + label)
                sheet.cell(row=r, column=2, value=5.5)
        out = io.BytesIO()
        book.save(out)
        out.seek(0)
        with pytest.raises(ValueError, match="Nothing is filled in"):
            opt.results_from_workbook(out)

    def test_conflicting_results_across_sheets_are_refused(
            self, tmp_path, monkeypatch):
        """One sheet has to win, and it is the one the whole batch is
        written on."""
        opt = self._opt(tmp_path, monkeypatch)
        book = openpyxl.load_workbook(self._filled_in(opt))
        sheet = book[wording.formulation_sheet_name(1)]
        for r in range(1, sheet.max_row + 1):
            if sheet.cell(row=r, column=2).value == "Firmness (N)":
                sheet.cell(row=r, column=4, value=9.9)
        out = io.BytesIO()
        book.save(out)
        out.seek(0)
        with pytest.raises(ValueError, match="conflicting Firmness"):
            opt.results_from_workbook(out)

    # ------------------------- the whole project ------------------------ #

    def test_all_formulations_holds_the_table_and_the_set_up(self, tmp_path,
                                                             monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        # Setting the total discards the open batch (its rows were built to
        # the old answer), so the rows are taken first.
        rows = list(opt.pending_batch)
        opt.set_formulation_total(100.0)
        opt.set_targets_source("A benchmark burger, panel of 8.")
        opt.tell(rows[0]['recipe'],
                 {"Firmness": 5.5, "Juiciness": 7.0}, formulation_no=1,
                 batch_no=2)
        opt.record_skipped(3, 2, rows[2]['recipe'],
                           note=wording.not_scored_with_note("burner failed"))
        book = _book(opt.all_formulations_workbook())
        assert book.sheetnames == ["All formulations", "Set up"]
        # The same table the download used to carry, to the cell.
        # A title row above the header says which amounts these are: the
        # RECORDED ones, which are not always the ones a batch sheet printed.
        written = pd.read_excel(io.BytesIO(opt.all_formulations_workbook()),
                                sheet_name="All formulations", header=1)
        downloaded = pd.read_csv(io.StringIO(opt.history_csv()))
        assert list(written.columns) == list(downloaded.columns)
        # check_dtype off: a spreadsheet reads 20.0 back as a whole number,
        # which is the same amount written the way a spreadsheet writes it.
        pd.testing.assert_frame_equal(written, downloaded, check_dtype=False)
        # ... and the set-up it was made under, which no table of numbers
        # can be read without six months later.
        setup = [v for row in _rows(book["Set up"]) for v in row
                 if v is not None]
        for expected in ("Ingredients and process settings", "Pea protein",
                         "Process setting", "Measurements and targets",
                         "Target 6 N", "0 to 10 N", "Limits",
                         "Default batch size · 100 g (set in Set up)",
                         "Where the targets come from",
                         "A benchmark burger, panel of 8."):
            assert expected in setup, expected

    def test_the_ingredients_template_opens_and_loads_back(self, tmp_path,
                                                           monkeypatch):
        """The file a project starts from is the same kind of file every
        other download is, and it comes back in through the same door — with
        the headers and ONE example row, so nobody has to work out which
        lines are theirs."""
        monkeypatch.chdir(tmp_path)
        template = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "data", "sample_ingredients.csv")
        book = _book(ingredients_template_workbook(template))
        assert book.sheetnames == ["Ingredients"]
        header = _rows(book["Ingredients"])[0]
        assert header[:4] == ("Name", "Lowest", "Highest", "Unit")
        # Every column the loader reserves is on the template, including the
        # three a pre-mix is written with: a reader who downloads it can
        # learn the shape no other way.
        for column in (wording.PART_OF_LABEL, wording.MADE_AS_LABEL,
                       wording.PREMIX_SHARE_LABEL, wording.FORMULA_LABEL):
            assert column in header
        frame = pd.read_excel(io.BytesIO(ingredients_template_workbook(template)))
        assert len(frame) == 1
        # The one example row is an ordinary row of the list: the three
        # pre-mix cells are empty, so nothing on the template says this
        # ingredient is part of something that is not there.
        for column in (wording.PART_OF_LABEL, wording.MADE_AS_LABEL,
                       wording.PREMIX_SHARE_LABEL):
            assert pd.isna(frame[column][0]) or not str(frame[column][0]).strip()
        opt = FoodOptimizer("from_template")
        opt.set_amount_unit("g")
        opt.load_ingredients_from_csv(frame)
        assert len(opt.variables) == 1 and opt.premixes == {}
        assert opt.unit_of(opt.variables[0]['name']) == "g"


# ------------------------------------------------------------------ #
#  The final fix wave (0.4.0): totals, batches and state
# ------------------------------------------------------------------ #

class TestTotalsAndBatchesFinalWave:
    """One total, one set of numbers: what the screen shows, what the sheet
    prints and what the record holds can never be three answers."""

    def _opt(self, tmp_path, monkeypatch, name="finalwave"):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer(name, robust=False)
        opt.set_amount_unit("g")
        opt.add_ingredient("Pea protein", 0, 60)
        opt.add_ingredient("Water", 0, 80)
        opt.add_objective("Firmness", 1.0, goal="target", target=6,
                          min_val=0, max_val=10, unit="/10")
        return opt

    # ---- F2: the reach counts a fixed ingredient at both ends -----

    def test_total_reach_counts_a_fixed_ingredient_at_both_ends(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        assert opt.total_reach() == (0.0, 140.0)
        opt.add_ingredient("Pea protein", 10.0, 10.0)
        # Pea protein can no longer move: it adds exactly 10 g at both ends.
        assert opt.total_reach() == (10.0, 90.0)

    def test_a_total_beyond_the_fixed_reach_is_refused_in_the_numbers(
            self, tmp_path, monkeypatch):
        """The reach is one pair of numbers now — a fixed row has no second
        range behind it — so the refusal is the reach sentence and nothing
        else."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_ingredient("Pea protein", 10.0, 10.0)
        with pytest.raises(ValueError) as excinfo:
            opt.set_formulation_total(120.0)
        assert "90 g" in str(excinfo.value)
        assert opt.formulation_total is None

    # ---- F1 / G-e2 / C4: nothing is rescaled under a project total ----

    def test_a_project_total_never_rescales_what_is_shown(self, tmp_path,
                                                          monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_formulation_total(100.0)
        row = {'formulation': 7, 'recipe': {"Pea protein": 20.0,
                                            "Water": 79.5}}
        shown, basis = opt.shown_recipe(row, opt.open_round_size())
        assert shown == row['recipe']      # the band edge stands
        assert basis is None               # its % is of its own sum

    def test_a_row_that_misses_the_total_says_so_in_one_line(self, tmp_path,
                                                             monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_formulation_total(100.0)
        line = opt.total_mismatch(7, {"Pea protein": 20.0, "Water": 79.5},
                                  100.0)
        assert line == ("Formulation 7 adds up to 99.50 g, not the 100 g "
                        "batch size.")
        assert opt.total_mismatch(8, {"Pea protein": 20.0, "Water": 80.0},
                                  100.0) == ""

    def test_a_formulation_of_your_own_is_never_rescaled(self, tmp_path,
                                                         monkeypatch):
        """Tab 2's typed total scales the suggestions; it never rewrites a
        formulation the user typed in."""
        opt = self._opt(tmp_path, monkeypatch)
        own = {'formulation': 4, 'recipe': {"Pea protein": 20.0,
                                            "Water": 77.0},
               'note': "Own formulation"}
        made = {'formulation': 5, 'recipe': {"Pea protein": 20.0,
                                             "Water": 77.0}}
        assert opt.shown_recipe(own, 150.0)[0] == own['recipe']
        assert opt.shown_recipe(made, 150.0)[0] != made['recipe']
        assert opt.total_mismatch(4, own['recipe'], 150.0) == (
            "Formulation 4 adds up to 97.00 g, not the 150 g batch size.")

    def test_the_batch_table_shows_each_row_at_its_own_sum_under_a_total(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_formulation_total(100.0)
        batch = [{'formulation': 1, 'recipe': {"Pea protein": 20.0,
                                               "Water": 79.5}},
                 {'formulation': 2, 'recipe': {"Pea protein": 25.0,
                                               "Water": 72.0},
                  'note': "Own formulation"}]
        frame = opt.batch_frame(batch, scale_to=opt.open_round_size())
        assert list(frame["Total (g)"]) == [99.5, 97.0]

    def test_the_scaled_caution_is_silent_under_a_project_total(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_formulation_total(100.0)
        rows = [{'formulation': 1, 'recipe': {"Pea protein": 20.0,
                                              "Water": 80.0}}]
        assert opt.scaled_cautions(rows, opt.open_round_size()) == []

    def test_a_typed_total_that_breaks_a_limit_names_the_limit(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_quantity_constraint(["Pea protein"], max_val=30.0)
        rows = [{'formulation': 1, 'recipe': {"Pea protein": 20.0,
                                              "Water": 80.0}}]
        lines = opt.scaled_cautions(rows, 200.0)
        assert any("Pea protein" in line and "is not met" in line
                   for line in lines), lines

    def test_an_own_row_takes_no_part_in_the_scaled_caution(self, tmp_path,
                                                            monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        rows = [{'formulation': 1, 'recipe': {"Pea protein": 20.0,
                                              "Water": 80.0},
                 'note': "Own formulation"}]
        assert opt.scaled_cautions(rows, 400.0) == []

    def test_an_own_row_says_it_is_the_users_own_rather_than_a_kind(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        text = opt.compared_with_text({"Pea protein": 20.0, "Water": 77.0},
                                      own=True)
        assert text.startswith("Your own formulation")
        assert "spread across" not in text.lower()

    # ---- G-d4 / C16: no Compared with column during the cold start ----

    def test_the_cold_start_batch_table_has_no_compared_with_column(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        batch = [{'formulation': 1, 'recipe': {"Pea protein": 20.0,
                                               "Water": 80.0}}]
        frame = opt.batch_frame(batch)
        assert not [c for c in frame.columns if str(c).startswith("Compared")]

    # ---- F4: a total that changes drops the open batch -----------------

    def test_setting_the_total_drops_the_open_batch(self, tmp_path,
                                                    monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_pending_batch([{"Pea protein": 20.0, "Water": 60.0}])
        assert opt.pending_batch
        opt.set_formulation_total(100.0)
        assert opt.pending_batch is None

    def test_clearing_the_total_drops_the_open_batch(self, tmp_path,
                                                     monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_formulation_total(100.0)
        opt.set_pending_batch([{"Pea protein": 20.0, "Water": 80.0}])
        opt.clear_formulation_total()
        assert opt.pending_batch is None

    def test_setting_the_same_total_again_leaves_the_batch_alone(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_formulation_total(100.0)
        opt.set_pending_batch([{"Pea protein": 20.0, "Water": 80.0}])
        opt.set_formulation_total(100.0)
        assert opt.pending_batch is not None

    # ---- F5: the recorded total is the one the sheets used -------------

    def test_a_batch_made_as_generated_is_recorded_as_generated(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_pending_batch([{"Pea protein": 20.0, "Water": 60.0}])
        no = opt.pending_batch_no
        opt.tell({"Pea protein": 20.0, "Water": 60.0}, {"Firmness": 6.0},
                 formulation_no=opt.pending_batch[0]['formulation'],
                 batch_no=no)
        assert opt.batch_total(no) is None
        # A total typed on tab 1 afterwards cannot rewrite what was made.
        opt.set_formulation_total(100.0)
        assert opt.recorded_total(no) is None

    def test_a_batch_from_before_totals_existed_shows_as_generated(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.tell({"Pea protein": 20.0, "Water": 60.0}, {"Firmness": 6.0},
                 formulation_no=1, batch_no=1)
        opt.batch_totals = {}          # a project saved before the key existed
        opt.set_formulation_total(100.0)
        assert opt.recorded_total(1) is None

    # ---- F9 / F11: imports and deletions leave no total behind ---------

    def test_an_imported_formulation_writes_no_batch_total(self, tmp_path,
                                                           monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_formulation_total(100.0)
        opt.set_pending_batch([{"Pea protein": 20.0, "Water": 80.0}])
        no = opt.pending_batch_no
        opt.import_formulation({"Pea protein": 20.0, "Water": 82.0},
                               {"Firmness": 6.0})
        assert opt.batch_history[-1] is None
        assert opt.batch_total(no) is None

    def test_deleting_the_last_row_of_a_batch_forgets_its_total(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_formulation_total(100.0)
        opt.set_pending_batch([{"Pea protein": 20.0, "Water": 80.0}])
        no = opt.pending_batch_no
        number = opt.pending_batch[0]['formulation']
        opt.tell({"Pea protein": 20.0, "Water": 80.0}, {"Firmness": 6.0},
                 formulation_no=number, batch_no=no)
        assert opt.batch_total(no) == 100.0
        opt._drop_pending_batch()      # the batch is closed once it is saved
        opt.delete_formulation(number)
        assert opt.batch_total(no) is None

    # ---- the queued float slack on the quantity check ------------------

    def test_a_quantity_limit_is_not_broken_by_the_last_bit_of_a_float(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_quantity_constraint(["Pea protein", "Water"], max_val=100.0)
        # 100.0 reached the long way round, as _snap_to_total reaches it.
        recipe = {"Pea protein": 33.333333333333336,
                  "Water": 66.66666666666669}
        assert sum(recipe.values()) > 100.0
        assert opt._check_constraints(recipe)


# ------------------------------------------------------------------ #
#  The final fix wave (0.4.0): the paper
# ------------------------------------------------------------------ #

class TestTheWorkbookFinalWave:
    """The sheet is the one screen the app cannot watch being used, so it
    says exactly what the screen says, and it refuses what it cannot read."""

    def _opt(self, tmp_path, monkeypatch, name="paper"):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer(name)
        opt.set_amount_unit("g")
        opt.add_ingredient("Pea protein", 0, 100)
        opt.add_ingredient("Water", 0, 100)
        opt.add_process_parameter("Cook temperature", 100, 220, unit="°C")
        opt.add_objective("Firmness", 1.5, goal="target", target=6,
                          min_val=0, max_val=10, unit="N")
        opt.add_objective("Juiciness", 1.0, goal="max", min_val=0, max_val=10,
                          unit="/10")
        opt.set_pending_batch([
            {"Pea protein": 20.0, "Water": 80.0, "Cook temperature": 180.0},
            {"Pea protein": 30.0, "Water": 70.0, "Cook temperature": 190.0},
        ], batch_no=2)
        return opt

    # ---- G-a1/b1: Goal for the words, Target for the number -----------

    def test_a_formulation_sheet_heads_the_words_goal(self, tmp_path,
                                                      monkeypatch):
        """'Target: higher is better' was printed on every sheet."""
        opt = self._opt(tmp_path, monkeypatch)
        sheet = _book(opt.workbook_bytes(opt.pending_batch))["Formulation 1"]
        rows = [r for r in _rows(sheet) if r[1] == wording.MEASUREMENT_COLUMN]
        assert rows and rows[0][2] == wording.GOAL_LABEL, rows

    def test_the_summary_reads_a_measurement_with_the_middle_dot(
            self, tmp_path, monkeypatch):
        """The screens say 'Firmness (N) · Target 6 N'; the sheet said it
        with a comma. One separator, and the upload matches on it."""
        opt = self._opt(tmp_path, monkeypatch)
        sheet = _book(opt.workbook_bytes(opt.pending_batch))["Round 2"]
        assert "Firmness (N) · Target 6 N" in _labelled(sheet)

    # ---- G-b2 / G-d5 / C24 / C25: what the paper says ------------------

    def test_the_summary_names_the_column_for_a_project_with_settings(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        sheet = _book(opt.workbook_bytes(opt.pending_batch))["Round 2"]
        assert _rows(sheet)[2][0] == wording.INGREDIENT_OR_SETTING_LABEL

    def test_the_summary_tick_row_carries_the_box(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        sheet = _book(opt.workbook_bytes(opt.pending_batch))["Round 2"]
        assert wording.NOT_SCORED_CHECKBOX_SHEET in _labelled(sheet)

    def test_the_title_line_says_what_the_sheets_were_made_to(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        title = _rows(_book(opt.workbook_bytes(opt.pending_batch,
                                               100.0))["Round 2"])[0][0]
        assert title.endswith(" · Batch size 100 g"), title
        plain = _rows(_book(opt.workbook_bytes(
            opt.pending_batch))["Round 2"])[0][0]
        assert "made to" not in plain

    def test_a_formulation_sheet_says_how_to_mark_what_it_asks_for(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        sheet = _book(opt.workbook_bytes(opt.pending_batch))["Formulation 1"]
        flat = [v for row in _rows(sheet) for v in row if v is not None]
        assert wording.SHEET_WRITE_IN_NOTE in flat, flat
        # The tick column is a column of boxes, not a header over nothing.
        head = next(i for i, r in enumerate(_rows(sheet))
                    if r[0] == wording.TICK_COLUMN)
        assert _rows(sheet)[head + 1][0] == wording.TICK_BOX

    # ---- C10 / F10: the All formulations sheet -------------------------

    def test_all_formulations_carries_the_total_and_the_tick(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        rows = list(opt.pending_batch)
        opt.tell(rows[0]['recipe'], {"Firmness": 6.0, "Juiciness": 7.0},
                 formulation_no=1, batch_no=2)
        opt.record_skipped(2, 2, rows[1]['recipe'],
                           note=wording.NOT_SCORED)
        sheet = _book(opt.all_formulations_workbook())["All formulations"]
        grid = _rows(sheet)
        assert grid[0][0] == wording.RECORDED_AMOUNTS
        header = list(grid[1])
        assert opt.total_column() in header
        assert wording.NOT_SCORED in header
        by_name = dict(zip(header, grid[2]))
        assert by_name[opt.total_column()] == 100.0
        assert by_name[wording.NOT_SCORED] is None   # blank: it was scored
        assert dict(zip(header, grid[3]))[wording.NOT_SCORED] == "☒"

    def test_all_formulations_writes_amounts_to_two_decimals(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.tell({"Pea protein": 33.333333, "Water": 66.666667,
                  "Cook temperature": 180.0},
                 {"Firmness": 6.0}, formulation_no=1, batch_no=2)
        sheet = _book(opt.all_formulations_workbook())["All formulations"]
        header = list(_rows(sheet)[1])
        cell = sheet.cell(row=3, column=header.index("Pea protein (g)") + 1)
        assert cell.number_format == "0.00"

    # ---- G-b5: the Set-up sheet says what the screen says --------------

    def test_the_set_up_sheet_separates_the_two_kinds_of_limit(
            self, tmp_path, monkeypatch):
        """On screen the finished-product limits have a heading of their own
        and a caption saying what they are per. On paper both kinds were
        rows of one Limits block, so "Fat per 100 g: at most 15" and
        "Lean: at most 40 g" read as one kind of rule — a weight in the bowl
        and an average of what you make."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_property("Fat per 100 g")
        for name in ("Pea protein", "Water"):
            opt.set_property_value(name, "Fat per 100 g", 10.0)
        opt.add_constraint("Fat per 100 g", max_val=15)
        opt.add_quantity_constraint(["Water"], max_val=80)
        rows = [[v for v in row if v is not None]
                for row in _rows(_book(opt.all_formulations_workbook())
                                 ["Set up"])]
        flat = [r[0] for r in rows if r]
        limits = flat.index(wording.LIMITS_SHEET_HEADING)
        props = flat.index(wording.PROPERTY_LIMITS_SHEET_HEADING)
        assert wording.PROPERTY_LIMITS_SHEET_HEADING == (
            "Finished-product limits")
        assert limits < props, flat
        assert flat[limits + 1] == "Water: at most 80 g", flat
        # The basis is said once, under the heading, as the screen says it.
        assert flat[props + 1] == opt.per_amount_text() == "per 100 g"
        assert flat[props + 2] == "Fat per 100 g: at most 15 g", flat

    def test_the_set_up_sheet_carries_the_share_of_score(self, tmp_path,
                                                         monkeypatch):
        """And nothing else: 0.5.0 makes the share the number the reader
        types, so the sheet no longer prints the importance behind it beside
        it — one fact, in one scale."""
        opt = self._opt(tmp_path, monkeypatch)
        rows = _rows(_book(opt.all_formulations_workbook())["Set up"])
        head = next(r for r in rows if r[0] == wording.MEASUREMENT_COLUMN)
        # Goal, Range, Share of score — and no Target column of its own:
        # 'Target 6 N' is the whole of what the Goal cell says.
        assert head[3] == wording.SHARE_COLUMN
        assert len([c for c in head if c]) == 4
        line = next(r for r in rows if r[0] == "Firmness")
        assert line[1] == "Target 6 N"
        assert line[3] == "60 %"

    # ---- F6 / F7 / F8: what an uploaded workbook is read as ------------

    def _filled(self, opt, edit):
        """The downloaded workbook, written on, handed back."""
        data = io.BytesIO(opt.workbook_bytes(opt.pending_batch))
        book = openpyxl.load_workbook(data)
        edit(book)
        out = io.BytesIO()
        book.save(out)
        out.seek(0)
        return out

    def _row_of(self, sheet, label):
        for r, row in enumerate(sheet.iter_rows(values_only=True), start=1):
            if row and str(row[0]).strip() == label:
                return r
        raise AssertionError(f"no row {label!r}")

    def test_a_note_with_no_numbers_and_no_tick_is_refused(self, tmp_path,
                                                           monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)

        def edit(book):
            sheet = book["Round 2"]
            sheet.cell(row=self._row_of(sheet, wording.NOTE), column=2).value \
                = "fell apart"

        with pytest.raises(ValueError) as excinfo:
            opt.results_from_workbook(self._filled(opt, edit))
        assert str(excinfo.value) == (
            "Formulation 1 has a note but no numbers. Tick Not scored to "
            "record it, or fill in the numbers.")

    def test_a_measurement_label_that_was_retyped_away_is_refused(
            self, tmp_path, monkeypatch):
        """The positional fallback reads the row the app wrote at that
        position. With the label gone AND that cell empty, the fallback is
        reading somebody else's row."""
        opt = self._opt(tmp_path, monkeypatch)

        def edit(book):
            sheet = book["Round 2"]
            row = self._row_of(sheet, "Firmness (N) · Target 6 N")
            sheet.cell(row=row, column=1).value = "Bite force"
            sheet.cell(row=row + 1, column=2).value = 7.0    # Juiciness

        with pytest.raises(ValueError) as excinfo:
            opt.results_from_workbook(self._filled(opt, edit))
        assert str(excinfo.value) == (
            "Firmness was not found on the Round 2 sheet. Keep the row "
            "labels the app wrote.")

    def test_the_written_in_note_wins_over_the_apps_own(self, tmp_path,
                                                        monkeypatch):
        """The app prints its own note on the sheet; what the bench writes
        beside it is what happened."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_pending_batch([{"Pea protein": 20.0, "Water": 80.0,
                                "Cook temperature": 180.0}], batch_no=3)
        number = opt.pending_batch[0]['formulation']
        opt.pending_batch[0]['note'] = "Repeat of Formulation 1"

        def edit(book):
            sheet = book[wording.formulation_sheet_name(number)]
            for r, row in enumerate(sheet.iter_rows(values_only=True), start=1):
                if row and len(row) > 1 and str(row[1]).strip() == "Firmness (N)":
                    sheet.cell(row=r, column=4).value = 6.0
                if row and len(row) > 1 and str(row[1]).strip() == wording.NOTE:
                    sheet.cell(row=r, column=4).value = "second try"

        frame = opt.results_from_workbook(self._filled(opt, edit)).frame
        assert list(frame[wording.NOTE]) == ["second try"]


class TestTheWriteInBlockIsFoundPastTheInstruction:
    """The instruction line sits between the `Measured` heading and the first
    measurement, so a fallback that counted from the heading read one row too
    high and handed back the row above the one it meant."""

    def _opt(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("firstrow")
        opt.set_amount_unit("g")
        opt.add_ingredient("Pea protein", 0, 100)
        opt.add_ingredient("Water", 0, 100)
        opt.add_objective("Firmness", 1.5, goal="target", target=6,
                          min_val=0, max_val=10, unit="N")
        opt.add_objective("Juiciness", 1.0, goal="max", min_val=0, max_val=10,
                          unit="/10")
        opt.set_pending_batch([{"Pea protein": 20.0, "Water": 80.0}],
                              batch_no=2)
        return opt

    def _handed_back(self, opt, edit):
        book = openpyxl.load_workbook(
            io.BytesIO(opt.workbook_bytes(opt.pending_batch)))
        edit(book["Round 2"])
        out = io.BytesIO()
        book.save(out)
        out.seek(0)
        return out

    @staticmethod
    def _row_of(sheet, label):
        for r, row in enumerate(sheet.iter_rows(values_only=True), start=1):
            if row and str(row[0]).strip() == label:
                return r
        raise AssertionError(f"no row {label!r}")

    def test_one_renamed_label_still_reads_both_measurements(self, tmp_path,
                                                             monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)

        def edit(sheet):
            firm = self._row_of(sheet, "Firmness (N) · Target 6 N")
            juice = self._row_of(sheet, "Juiciness (/10) · Prefer higher values")
            sheet.cell(row=juice, column=1).value = "Mouth juiciness"
            sheet.cell(row=firm, column=2).value = 5.5
            sheet.cell(row=juice, column=2).value = 8.0

        frame = opt.results_from_workbook(self._handed_back(opt, edit)).frame
        assert list(frame["Firmness"]) == [5.5]
        assert list(frame["Juiciness"]) == [8.0]


class TestRenameVariable:
    """A name is a KEY: the amounts of every formulation are filed under it,
    the open batch and the not-scored rows hold it, the limits list it and an
    ingredient's property values are filed under it. Renaming moves all of
    it, or it is not a rename — it is a deletion and an addition wearing one
    button."""

    def _opt(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("rename")
        opt.set_amount_unit("g")
        opt.add_ingredient("Pea protein", 0, 60)
        opt.add_ingredient("Water", 0, 60)
        opt.add_property("Fat per 100 g")
        opt.set_property_value("Pea protein", "Fat per 100 g", 8.0)
        opt.add_objective("Firmness", 1.0, goal="target", target=6,
                          min_val=0, max_val=10, unit="N")
        opt.add_quantity_constraint(["Pea protein"], max_val=40.0)
        opt.tell({"Pea protein": 20.0, "Water": 40.0}, {"Firmness": 5.5},
                 formulation_no=1, batch_no=1)
        opt.record_skipped(2, 1, {"Pea protein": 25.0, "Water": 35.0})
        opt.set_formulation_total(60.0)
        # Last: a set-up change retires the open batch, and this one has to
        # still be open when the rename reaches it.
        opt.set_pending_batch([{"Pea protein": 30.0, "Water": 30.0}],
                              batch_no=2)
        return opt

    def test_everything_filed_under_the_old_name_moves(self, tmp_path,
                                                       monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        before = list(opt.X_history)
        opt.rename_variable("Pea protein", "Pea protein isolate")
        reloaded = FoodOptimizer("rename")
        assert [v['name'] for v in reloaded.variables] == \
            ["Pea protein isolate", "Water"]
        # The formulation already recorded keeps its amount, under the new
        # name, and encodes to the same vector it always did.
        assert reloaded.recipe_history[0] == {"Pea protein isolate": 20.0,
                                              "Water": 40.0}
        assert reloaded.X_history == before
        # The open batch and the row nobody scored.
        assert reloaded.pending_batch[0]['recipe'] == \
            {"Pea protein isolate": 30.0, "Water": 30.0}
        assert reloaded.skipped[0]['recipe'] == \
            {"Pea protein isolate": 25.0, "Water": 35.0}
        # The limit that named it, and the total's own limit over every
        # ingredient.
        chosen = [qc for qc in reloaded.quantity_constraints
                  if qc.get('source') != 'formulation_total']
        assert chosen[0]['ingredients'] == ["Pea protein isolate"]
        total = next(qc for qc in reloaded.quantity_constraints
                     if qc.get('source') == 'formulation_total')
        assert sorted(total['ingredients']) == ["Pea protein isolate", "Water"]
        assert reloaded.formulation_total == 60.0
        # The property value follows the ingredient it was measured on.
        assert reloaded.property_value("Pea protein isolate",
                                       "Fat per 100 g") == 8.0
        assert "Pea protein" not in reloaded.ingredient_properties

    def test_a_fixed_row_stays_fixed(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_ingredient("Pea protein", 20.0, 20.0)
        opt.rename_variable("Pea protein", "Pea protein isolate")
        fixed = opt.fixed_variables()
        assert [v['name'] for v in fixed] == ["Pea protein isolate"]
        assert opt._fixed_value(fixed[0]) == 20.0

    def test_a_name_that_is_taken_is_refused_and_nothing_moves(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_process_parameter("Cook temperature", 150, 200, baseline=175,
                                  unit="°C")
        for taken, says in (("Water", "already the name of an ingredient"),
                            ("water", "already the name of an ingredient"),
                            ("Cook temperature",
                             "already the name of a process setting"),
                            ("Firmness", "already the name of a measurement"),
                            ("Total", "column name Food Optimizer"),
                            ("   ", "Name cannot be empty")):
            with pytest.raises(ValueError, match=says):
                opt.rename_variable("Pea protein", taken)
        assert [v['name'] for v in opt.variables][0] == "Pea protein"
        assert opt.recipe_history[0]["Pea protein"] == 20.0

    def test_its_own_name_is_not_a_rename(self, tmp_path, monkeypatch):
        """Saving the form without touching the Name box must not be a write
        that renames a row onto itself."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.rename_variable("Pea protein", "  Pea protein  ")
        assert [v['name'] for v in opt.variables] == ["Pea protein", "Water"]
        assert opt.recipe_history[0]["Pea protein"] == 20.0

    def test_only_the_capitals_change(self, tmp_path, monkeypatch):
        """A row is allowed to correct its own capitals: the clash rule is
        about OTHER rows."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.rename_variable("Pea protein", "Pea Protein")
        assert [v['name'] for v in opt.variables] == ["Pea Protein", "Water"]
        assert opt.recipe_history[0] == {"Pea Protein": 20.0, "Water": 40.0}


class TestEditingASetting:
    """Re-adding a name the project already has is how the screen saves an
    edit, so that path owes the setting everything the add did — including
    the baseline, which is what the formulations already made are read at."""

    def _opt(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("edit_setting")
        opt.add_ingredient("Water", 0, 100)
        opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
        opt.tell({"Water": 50.0}, {"Taste": 7.0})
        opt.add_process_parameter("Cook temperature", 150, 200, baseline=175,
                                  unit="°C")
        return opt

    def test_a_corrected_baseline_re_encodes_the_history(self, tmp_path,
                                                         monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        before = list(opt.X_history)
        opt.add_process_parameter("Cook temperature", 150, 200, baseline=160,
                                  unit="°C")
        var = opt._var_by_name("Cook temperature")
        assert var['_absent_value'] == 160.0
        # The formulation already made is read at the new baseline, so its
        # encoded vector moves with it.
        assert opt.X_history != before
        assert opt.X_history == [opt._encode(r) for r in opt.recipe_history]

    def test_a_baseline_outside_the_new_amounts_is_refused_whole(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        with pytest.raises(ValueError, match="must be between"):
            opt.add_process_parameter("Cook temperature", 150, 170,
                                      baseline=175, unit="°C")
        var = opt._var_by_name("Cook temperature")
        assert var['bounds'] == (150.0, 200.0) and var['_absent_value'] == 175.0

    def test_the_baseline_is_left_alone_when_none_is_passed(self, tmp_path,
                                                            monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_process_parameter("Cook temperature", 140, 210, unit="°C")
        var = opt._var_by_name("Cook temperature")
        assert var['bounds'] == (140.0, 210.0) and var['_absent_value'] == 175.0


class TestAFixedRowIsItsOwnTwoNumbers:
    """0.5.0: a row pinned at one amount IS its Lowest and its Highest, so
    there is nothing left to keep in step. Narrowing the range moves the
    amount because the range is the amount; the class this replaces existed
    only because a hold was stored beside the range and could disagree
    with it."""

    def _opt(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("clamped")
        opt.set_amount_unit("g")
        opt.add_ingredient("Water", 0, 60)
        opt.add_ingredient("Flour", 0, 60)
        opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
        return opt

    def test_re_fixing_a_row_moves_it_and_the_search_with_it(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_ingredient("Water", 20, 20)
        opt.add_ingredient("Water", 5, 5)
        var = opt._var_by_name("Water")
        assert var['bounds'] == (5.0, 5.0)
        assert opt._fixed_value(var) == 5.0
        # And the search is there too, not at the amount that went.
        assert all(row["Water"] == pytest.approx(5.0)
                   for row in opt.ask(n_suggestions=2))

    def test_giving_a_fixed_row_a_range_again_puts_it_back_in_play(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_ingredient("Water", 20, 20)
        assert [v['name'] for v in opt.fixed_variables()] == ["Water"]
        opt.add_ingredient("Water", 10, 60)
        assert opt.fixed_variables() == []
        assert opt._var_by_name("Water")['bounds'] == (10.0, 60.0)

    def test_nothing_is_stored_beside_the_range_to_disagree_with_it(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_ingredient("Water", 20, 20)
        var = opt._var_by_name("Water")
        assert '_frozen_at' not in var and 'active' not in var

    def test_a_setting_is_fixed_the_same_way(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_process_parameter("Cook temperature", 150, 200, unit="°C")
        opt.add_process_parameter("Cook temperature", 170, 170, unit="°C")
        var = opt._var_by_name("Cook temperature")
        assert var['bounds'] == (170.0, 170.0)
        assert all(row["Cook temperature"] == pytest.approx(170.0)
                   for row in opt.ask(n_suggestions=2))


class TestOneNameOneThing:
    """A variable, a measurement and a property all head columns of the same
    tables, so the three doors that name something refuse the same clashes."""

    def _opt(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("one_name")
        opt.add_ingredient("Water", 0, 100)
        opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
        opt.add_property("Cost per 100 g")
        return opt

    def test_a_rename_onto_a_property_is_refused(self, tmp_path, monkeypatch):
        """add_property already refuses an ingredient's name; this is the
        same rule read from the other end."""
        opt = self._opt(tmp_path, monkeypatch)
        with pytest.raises(ValueError, match="already a property"):
            opt.rename_variable("Water", "Cost per 100 g")
        assert [v['name'] for v in opt.variables] == ["Water"]

    def test_adding_a_variable_named_like_a_property_is_refused_too(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        with pytest.raises(ValueError, match="already a property"):
            opt.add_ingredient("Cost per 100 g", 0, 10)
        with pytest.raises(ValueError, match="already a property"):
            opt.add_process_parameter("cost per 100 g", 0, 10)

    def test_a_row_may_still_be_edited_and_may_keep_its_own_name(
            self, tmp_path, monkeypatch):
        """The one name a row is always free to wear is the one it has."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_ingredient("Water", 0, 50)          # an edit, not a clash
        assert opt._var_by_name("Water")['bounds'] == (0.0, 50.0)
        opt.rename_variable("Water", "Water")
        assert [v['name'] for v in opt.variables] == ["Water"]


class TestScaleRound:
    """0.5.0: the batch size lives on the round screen, and changing it
    rewrites the round rather than the picture of it.

    Until 0.5.0 the box scaled the DISPLAY: the stored rows kept their own
    amounts, the sheets were written at the typed size, and what was
    recorded on Save was the unscaled row — the bench weighed one thing and
    the model learned another. scale_round moves the amounts themselves, so
    the table, the sheets and the record are one set of numbers."""

    def _opt(self, tmp_path, monkeypatch, name="scale_round"):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer(name)
        opt.set_amount_unit("g")
        opt.add_ingredient("Pea protein", 0, 200)
        opt.add_ingredient("Water", 0, 200)
        opt.add_objective("Firmness", 1.0, goal="target", target=6,
                          min_val=0, max_val=10, unit="N")
        opt.set_pending_batch([{"Pea protein": 20.0, "Water": 30.0},
                               {"Pea protein": 10.0, "Water": 40.0}])
        return opt

    def test_every_row_is_rewritten_proportionally_to_the_size(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.scale_round(100.0)
        first, second = [r['recipe'] for r in opt.pending_batch]
        assert first == pytest.approx({"Pea protein": 40.0, "Water": 60.0})
        assert second == pytest.approx({"Pea protein": 20.0, "Water": 80.0})

    def test_a_formulation_of_the_users_own_is_scaled_too(self, tmp_path,
                                                          monkeypatch):
        """The user's own row was the one thing the display scaling never
        touched, which left it the odd number on a sheet everything else had
        been written to."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_to_pending_batch({"Pea protein": 5.0, "Water": 5.0},
                                 note="My own")
        opt.scale_round(100.0)
        own = opt.pending_batch[-1]
        assert own['note'] == "My own"
        assert own['recipe'] == pytest.approx({"Pea protein": 50.0,
                                               "Water": 50.0})

    def test_the_size_is_stored_and_the_round_stays_open(self, tmp_path,
                                                         monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        no = opt.pending_batch_no
        opt.scale_round(100.0)
        assert opt.pending_batch_total == 100.0
        assert opt.pending_batch_no == no
        assert len(opt.pending_batch) == 2
        reopened = FoodOptimizer("scale_round")
        assert reopened.pending_batch_total == 100.0
        assert [r['recipe'] for r in reopened.pending_batch] == pytest.approx(
            [r['recipe'] for r in opt.pending_batch])

    def test_scaling_twice_goes_from_the_amounts_on_the_table(
            self, tmp_path, monkeypatch):
        """Each scaling is from what the rows hold now, so 100 then 50 is 50
        and not 25."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.scale_round(100.0)
        opt.scale_round(50.0)
        assert opt.ingredient_total(opt.pending_batch[0]['recipe']) == \
            pytest.approx(50.0)
        assert opt.pending_batch[0]['recipe'] == pytest.approx(
            {"Pea protein": 20.0, "Water": 30.0})

    def test_a_process_setting_keeps_its_value(self, tmp_path, monkeypatch):
        """A cook temperature is not an amount: it does not scale with the
        size of what is being cooked."""
        opt = self._opt(tmp_path, monkeypatch, name="scale_round_setting")
        opt.add_process_parameter("Cook temperature", 60, 100, unit="°C")
        opt.set_pending_batch([{"Pea protein": 20.0, "Water": 30.0,
                                "Cook temperature": 80.0}])
        opt.scale_round(100.0)
        recipe = opt.pending_batch[0]['recipe']
        assert recipe["Cook temperature"] == 80.0
        assert opt.ingredient_total(recipe) == pytest.approx(100.0)

    def test_the_cautions_read_off_the_scaled_rows(self, tmp_path,
                                                   monkeypatch):
        """The amounts are past what the project allows at 1000 g, and the
        same one line says so."""
        opt = self._opt(tmp_path, monkeypatch, name="scale_round_caution")
        opt.scale_round(1000.0)
        cautions = opt.scaled_cautions(opt.pending_batch,
                                       opt.pending_batch_total)
        assert cautions and "1000" in cautions[0]

    def test_no_round_open_is_a_no_op(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch, name="scale_round_empty")
        opt.set_pending_batch(None)
        opt.scale_round(100.0)
        assert not opt.pending_batch
        assert opt.pending_batch_total is None

    def test_a_non_positive_size_is_not_a_size(self, tmp_path, monkeypatch):
        """Zero and below are the same answer as None: the round has no size
        of its own. Nothing is rewritten — the box moved the amounts once and
        there is no going back to what the model proposed — and nothing
        non-positive is stored, because a sheet cannot be printed to it."""
        opt = self._opt(tmp_path, monkeypatch, name="scale_round_zero_size")
        opt.scale_round(100.0)
        made = [dict(r['recipe']) for r in opt.pending_batch]
        for refused in (0.0, -5.0, None):
            opt.scale_round(refused)
            assert opt.pending_batch_total is None
            assert [r['recipe'] for r in opt.pending_batch] == made
            opt.scale_round(100.0)          # and a real size still lands
            assert opt.pending_batch_total == 100.0

    def test_a_row_that_adds_up_to_nothing_is_left_alone(self, tmp_path,
                                                        monkeypatch):
        """Nothing times anything is still nothing: there is no factor that
        takes a row of zeros to 100 g, and the size is still what the round
        is being made to."""
        opt = self._opt(tmp_path, monkeypatch, name="scale_round_zero")
        opt.set_pending_batch([{"Pea protein": 0.0, "Water": 0.0}])
        opt.scale_round(100.0)
        assert opt.pending_batch[0]['recipe'] == {"Pea protein": 0.0,
                                                  "Water": 0.0}
        assert opt.pending_batch_total == 100.0


class TestFixedIsLowestEqualsHighest:
    """0.5.0 §1.2: there is no Hold. A row whose Lowest is its Highest is
    FIXED — the model pins it at that amount everywhere the old inactive
    flag pinned it, and the two boxes the user already types into are the
    whole of the control."""

    def _opt(self, tmp_path, monkeypatch, name="fixed"):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer(name, robust=False)
        opt.set_amount_unit("g")
        opt.add_ingredient("Pea protein", 0, 60)
        opt.add_ingredient("Water", 0, 80)
        opt.add_ingredient("Salt", 20, 20)
        opt.add_objective("Firmness", 1.0, goal="target", target=6,
                          min_val=0, max_val=10, unit="/10")
        return opt

    def _warm(self, opt):
        """Five results, so the next ask goes through the GP rather than the
        space-filling opening."""
        for i in range(5):
            opt.tell({"Pea protein": 20.0 + i, "Water": 60.0 - i,
                      "Salt": 20.0}, {"Firmness": 5.0 + i * 0.1})

    # ---- the model pins it, cold and warm ----

    def test_a_fixed_row_is_read_off_its_two_boxes(self, tmp_path,
                                                   monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        assert [v['name'] for v in opt.fixed_variables()] == ["Salt"]
        assert [v['name'] for v in opt.varying_variables()] == \
            ["Pea protein", "Water"]
        assert opt._fixed_value(opt._var_by_name("Salt")) == 20.0

    def test_the_cold_start_pins_a_fixed_ingredient_and_still_lands_on_the_size(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_formulation_total(100.0)
        rows = opt.ask(n_suggestions=3)
        for recipe in rows:
            assert recipe["Salt"] == pytest.approx(20.0)
            assert sum(recipe.values()) == pytest.approx(100.0, abs=0.01)

    def test_the_warm_search_pins_a_fixed_ingredient_too(self, tmp_path,
                                                         monkeypatch):
        """The GP still sees Salt's column — every past formulation had
        some — but the acquisition is maximized with it held out."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_formulation_total(100.0)
        self._warm(opt)
        for recipe in opt.ask(n_suggestions=2):
            assert recipe["Salt"] == pytest.approx(20.0)
            assert sum(recipe.values()) == pytest.approx(100.0, abs=0.5)

    def test_the_reach_counts_a_fixed_ingredient_at_both_ends(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        assert opt.total_reach() == (20.0, 160.0)
        opt.add_ingredient("Pea protein", 10, 10)
        assert opt.total_reach() == (30.0, 110.0)

    def test_a_fixed_setting_is_dialled_to_its_one_number(self, tmp_path,
                                                         monkeypatch):
        opt = self._opt(tmp_path, monkeypatch, name="fixed_setting")
        opt.add_process_parameter("Cook temperature", 175, 175, unit="°C")
        assert "Cook temperature" in [v['name'] for v in opt.fixed_variables()]
        for recipe in opt.ask(n_suggestions=2):
            assert recipe["Cook temperature"] == pytest.approx(175.0)
        assert opt.fixed_at_text(opt._var_by_name("Cook temperature")) \
            == "175 °C"

    def test_a_fixed_row_is_not_one_of_the_changes_a_round_made(
            self, tmp_path, monkeypatch):
        """It is the same amount in every formulation, so it cannot be what
        this round is trying."""
        opt = self._opt(tmp_path, monkeypatch, name="fixed_changes")
        changes = opt._variable_deltas(
            {"Pea protein": 30.0, "Water": 50.0, "Salt": 20.0},
            {"Pea protein": 20.0, "Water": 60.0, "Salt": 20.0}, 'ingredient')
        assert [name for name, _ in changes] == ["Pea protein", "Water"]

    def test_generating_with_everything_fixed_says_what_to_do(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch, name="all_fixed")
        opt.add_ingredient("Pea protein", 10, 10)
        opt.add_ingredient("Water", 70, 70)
        with pytest.raises(ValueError) as refused:
            opt.ask(n_suggestions=1)
        assert "fixed at one amount" in str(refused.value)

    # ---- equal is a fix; inverted is a refusal ----

    def test_equal_amounts_are_accepted_and_inverted_ones_are_not(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch, name="equal_bounds")
        opt.add_ingredient("Oil", 5, 5)
        assert opt._var_by_name("Oil")['bounds'] == (5.0, 5.0)
        opt.add_process_parameter("Mixing time", 90, 90, unit="s")
        assert opt._var_by_name("Mixing time")['bounds'] == (90.0, 90.0)
        for lowest, highest in ((10, 5), (200, 100)):
            with pytest.raises(ValueError,
                               match="Lowest cannot be above Highest"):
                opt.add_ingredient("Oil", lowest, highest)
            with pytest.raises(ValueError,
                               match="Lowest cannot be above Highest"):
                opt.add_process_parameter("Mixing time", lowest, highest)
        assert opt._var_by_name("Oil")['bounds'] == (5.0, 5.0)

    def test_an_ingredient_file_may_fix_a_row_too(self, tmp_path,
                                                  monkeypatch):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("fixed_csv")
        opt.load_ingredients_from_csv(pd.DataFrame({
            "Name": ["Flour", "Salt"], "Min": [0, 2], "Max": [100, 2]}))
        assert [v['name'] for v in opt.fixed_variables()] == ["Salt"]
        with pytest.raises(ValueError, match="cannot be above Highest"):
            opt.load_ingredients_from_csv(pd.DataFrame({
                "Name": ["Flour"], "Min": [50], "Max": [10]}))

    # ---- the guard sentence ----

    def test_fixing_a_row_out_of_the_batch_size_is_refused_in_those_words(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch, name="fix_guard")
        opt.set_formulation_total(120.0)
        # Water is 0 to 80 g; fixed at 0, the rest reach 80 g at most.
        with pytest.raises(ValueError) as refused:
            opt.add_ingredient("Water", 0, 0)
        assert str(refused.value) == (
            "Fixing these would leave no formulation adding up to 120 g. "
            "Change the batch size, or let enough ingredients vary again. "
            "Fixed at one amount: Water.")
        # Refused before anything was written: Water still varies.
        assert opt._var_by_name("Water")['bounds'] == (0.0, 80.0)
        assert [v['name'] for v in opt.fixed_variables()] == ["Salt"]

    def test_a_fix_the_batch_size_still_reaches_is_allowed(self, tmp_path,
                                                           monkeypatch):
        opt = self._opt(tmp_path, monkeypatch, name="fix_ok")
        opt.set_formulation_total(100.0)
        opt.add_ingredient("Pea protein", 30, 30)
        assert opt._var_by_name("Pea protein")['bounds'] == (30.0, 30.0)
        for recipe in opt.ask(n_suggestions=2):
            assert recipe["Pea protein"] == pytest.approx(30.0)

    # ---- what the sheet says ----

    def test_the_set_up_sheet_says_a_row_is_fixed_and_at_what(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch, name="fixed_sheet")
        opt.tell({"Pea protein": 20.0, "Water": 60.0, "Salt": 20.0},
                 {"Firmness": 6.0})
        book = openpyxl.load_workbook(
            io.BytesIO(opt.all_formulations_workbook()))
        sheet = book[wording.SET_UP_SHEET]
        header = [c.value for c in sheet[2]]
        assert header[6] == wording.STATUS_LABEL
        said = {sheet.cell(row=r, column=1).value:
                sheet.cell(row=r, column=7).value for r in (3, 4, 5)}
        assert said == {"Pea protein": None, "Water": None,
                        "Salt": "Fixed at 20.00 g"}

    def test_the_sheet_has_no_status_column_when_nothing_is_fixed(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch, name="nothing_fixed")
        opt.add_ingredient("Salt", 0, 5)
        book = openpyxl.load_workbook(
            io.BytesIO(opt.all_formulations_workbook()))
        header = [c.value for c in book[wording.SET_UP_SHEET][2]]
        assert wording.STATUS_LABEL not in header

    # ---- a copy carries it ----

    def test_a_fixed_row_survives_a_saved_copy(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch, name="fixed_copy")
        state = opt.export_json()
        assert state['CLASS_VERSION'] == 15
        assert FoodOptimizer.validate_state(state)['ingredients'] == 3
        restored = FoodOptimizer("fixed_copy_restored")
        restored.import_json(state)
        assert [v['name'] for v in restored.fixed_variables()] == ["Salt"]
        assert restored._var_by_name("Salt")['bounds'] == (20.0, 20.0)
        assert restored.lots == {}

    def test_a_copy_from_a_newer_app_is_still_refused(self, tmp_path,
                                                      monkeypatch):
        opt = self._opt(tmp_path, monkeypatch, name="fixed_newer")
        state = opt.export_json()
        state['CLASS_VERSION'] = FoodOptimizer.CLASS_VERSION + 1
        with pytest.raises(ValueError, match="newer version"):
            FoodOptimizer.validate_state(state)

    def test_lot_numbers_ride_with_the_project(self, tmp_path, monkeypatch):
        """Task 3 fills these in from the sheets; the shape on disk is
        settled now so an older file needs no second migration."""
        opt = self._opt(tmp_path, monkeypatch, name="fixed_lots")
        opt.lots = {1: {"Salt": "LOT-7"}}
        opt.save()
        reloaded = FoodOptimizer("fixed_lots")
        assert reloaded.lots == {1: {"Salt": "LOT-7"}}


class TestMigratingAHeldProject:
    """A 0.4.1 file spelled a fixed row as `active: False` plus `_frozen_at`.
    It opens as a row whose Lowest is its Highest, and the two old keys are
    never read again."""

    def _state(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("pre_050")
        opt.set_amount_unit("g")
        opt.add_ingredient("Pea protein", 0, 60)
        opt.add_ingredient("Water", 0, 80)
        opt.add_process_parameter("Cook temperature", 150, 200, unit="°C")
        opt.add_objective("Firmness", 1.0, goal="target", target=6,
                          min_val=0, max_val=10, unit="/10")
        state = opt.export_json()
        state['CLASS_VERSION'] = 9
        for var in state['variables']:
            if var['name'] == "Water":
                var['active'] = False
                var['_frozen_at'] = 20.0
            if var['name'] == "Cook temperature":
                var['active'] = False
                var['_frozen_at'] = 175.0
        return state

    def test_a_held_row_and_a_held_setting_open_as_fixed_rows(
            self, tmp_path, monkeypatch):
        state = self._state(tmp_path, monkeypatch)
        opt = FoodOptimizer("migrated")
        opt.import_json(state)
        water = opt._var_by_name("Water")
        assert water['bounds'] == (20.0, 20.0)
        assert 'active' not in water and '_frozen_at' not in water
        setting = opt._var_by_name("Cook temperature")
        assert setting['bounds'] == (175.0, 175.0)
        assert [v['name'] for v in opt.fixed_variables()] == \
            ["Water", "Cook temperature"]
        assert [v['name'] for v in opt.varying_variables()] == ["Pea protein"]

    def test_the_migrated_amounts_are_what_the_next_round_uses(
            self, tmp_path, monkeypatch):
        state = self._state(tmp_path, monkeypatch)
        opt = FoodOptimizer("migrated_ask")
        opt.import_json(state)
        for recipe in opt.ask(n_suggestions=2):
            assert recipe["Water"] == pytest.approx(20.0)
            assert recipe["Cook temperature"] == pytest.approx(175.0)

    def test_a_row_held_without_a_number_of_its_own_keeps_the_old_answer(
            self, tmp_path, monkeypatch):
        """_frozen_value read an ingredient at nothing and a setting at its
        Lowest when no `_frozen_at` was stored."""
        state = self._state(tmp_path, monkeypatch)
        for var in state['variables']:
            var.pop('_frozen_at', None)
            if var['name'] in ("Water", "Cook temperature"):
                var['active'] = False
        opt = FoodOptimizer("migrated_bare")
        opt.import_json(state)
        assert opt._var_by_name("Water")['bounds'] == (0.0, 0.0)
        assert opt._var_by_name("Cook temperature")['bounds'] == (150.0, 150.0)

    def test_a_setting_held_at_its_baseline_keeps_it(self, tmp_path,
                                                     monkeypatch):
        state = self._state(tmp_path, monkeypatch)
        for var in state['variables']:
            if var['name'] == "Cook temperature":
                var.pop('_frozen_at')
                var['_absent_value'] = 190.0
        opt = FoodOptimizer("migrated_baseline")
        opt.import_json(state)
        assert opt._var_by_name("Cook temperature")['bounds'] == (190.0, 190.0)

    def test_an_active_project_is_left_exactly_as_it_was(self, tmp_path,
                                                         monkeypatch):
        state = self._state(tmp_path, monkeypatch)
        for var in state['variables']:
            var['active'] = True
            var.pop('_frozen_at', None)
        opt = FoodOptimizer("migrated_none")
        opt.import_json(state)
        assert opt._var_by_name("Water")['bounds'] == (0.0, 80.0)
        assert opt.fixed_variables() == []


def test_a_setting_with_a_baseline_can_still_be_fixed_elsewhere(tmp_path,
                                                                monkeypatch):
    """The baseline is what the bakes already done ran at; the range is what
    the next round will run at. Refusing "Baseline 175 must be between 190
    and 190" refused a thing the bench is entitled to ask for."""
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("fixed_baseline")
    opt.add_ingredient("Water", 0, 100)
    opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
    opt.tell({"Water": 50.0}, {"Taste": 5.0})
    opt.add_process_parameter("Cook temperature", 150, 200, baseline=175,
                              unit="°C")
    opt.add_process_parameter("Cook temperature", 190, 190, unit="°C")
    var = opt._var_by_name("Cook temperature")
    assert var['bounds'] == (190.0, 190.0)
    assert var['_absent_value'] == 175.0     # the past is not rewritten
    # The search frame stretches over both, so the history still normalizes.
    assert all(row["Cook temperature"] == pytest.approx(190.0)
               for row in opt.ask(n_suggestions=2))
    # A baseline actually typed outside a range is still refused.
    with pytest.raises(ValueError, match="must be between"):
        opt.add_process_parameter("Cook temperature", 150, 170, baseline=200,
                                  unit="°C")


class TestALimitOnlyFixedRowsFeed:
    """Fix round 1, finding 1. A limit every one of whose ingredients is
    fixed is a CONSTANT: there is no inequality to hand the solver. Dropping
    it silently let `ask` return formulations the app's own
    `_check_constraints` calls invalid — the base behaviour failed loudly, so
    this was a regression, not a gap."""

    def _opt(self, tmp_path, monkeypatch, name="constant_limit"):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer(name, robust=False)
        opt.set_amount_unit("g")
        opt.add_ingredient("Pea protein", 0, 60)
        opt.add_ingredient("Salt", 0, 10)
        opt.add_objective("Firmness", 1.0, goal="target", target=6,
                          min_val=0, max_val=10, unit="/10")
        return opt

    def _warm(self, opt):
        for i in range(5):
            opt.tell({"Pea protein": 20.0 + i, "Salt": 2.0},
                     {"Firmness": 5.0 + i * 0.1})

    # ---- the round refuses rather than answering with invalid rows ---- #

    def test_a_constant_amount_limit_the_fixed_rows_break_stops_the_round(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_quantity_constraint(["Salt"], max_val=5.0)
        self._warm(opt)
        # The limit is written around the model's back, the way a file
        # restored from an older copy could carry one.
        opt._var_by_name("Salt")['bounds'] = (8.0, 8.0)
        with pytest.raises(ValueError, match="impossible to meet"):
            opt.ask(n_suggestions=2)

    def test_a_constant_property_limit_the_fixed_rows_break_stops_it_too(
            self, tmp_path, monkeypatch):
        """Every INGREDIENT fixed, with a process setting still varying so
        there is a round to generate at all: the average is one number, so
        the property loop has no column to hand the solver either."""
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("constant_property", robust=False)
        opt.load_ingredients_from_csv(pd.DataFrame([
            {"Name": "Lean", "Min": 0, "Max": 100, "Fat per 100 g": 10.0},
            {"Name": "Fatty", "Min": 0, "Max": 100, "Fat per 100 g": 30.0},
        ]))
        opt.add_objective("Firmness", 1.0, goal="max", min_val=0, max_val=10)
        opt.add_process_parameter("Cook temperature", 150, 200, unit="°C")
        opt.add_constraint("Fat per 100 g", min_val=25.0)
        for i in range(5):
            opt.tell({"Lean": 10.0 + i, "Fatty": 60.0,
                      "Cook temperature": 170.0 + i},
                     {"Firmness": 5.0 + i * 0.1})
        for name in ("Lean", "Fatty"):
            opt._var_by_name(name)['bounds'] = (20.0, 20.0)
        with pytest.raises(ValueError, match="impossible to meet"):
            opt.ask(n_suggestions=2)

    def test_a_constant_limit_the_fixed_rows_MEET_is_no_trouble(
            self, tmp_path, monkeypatch):
        """There is still nothing to hand the solver — but a constant that
        holds is a limit already met, not a refusal."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_quantity_constraint(["Salt"], max_val=5.0)
        self._warm(opt)
        opt._var_by_name("Salt")['bounds'] = (2.0, 2.0)
        rows = opt.ask(n_suggestions=2)
        assert all(row["Salt"] == pytest.approx(2.0) for row in rows)
        assert all(opt._check_constraints(row) for row in rows)

    # ---- and the door the limit is written at refuses it first -------- #

    def test_a_limit_the_fixed_rows_already_break_is_refused_at_the_door(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_ingredient("Salt", 8.0, 8.0)
        with pytest.raises(ValueError) as refused:
            opt.add_quantity_constraint(["Salt"], max_val=5.0)
        assert str(refused.value) == (
            "No formulation can meet a limit on Salt while Salt is fixed at "
            "one amount. Change the limit, or give it a different Lowest "
            "and Highest.")
        assert opt.quantity_constraints == []

    def test_a_property_limit_the_fixed_rows_already_break_is_refused_too(
            self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("prop_door")
        # Lean pinned at 20 g dilutes every formulation, so the average can
        # never get above 26.67 per 100 g however much Fatty goes in.
        opt.load_ingredients_from_csv(pd.DataFrame([
            {"Name": "Lean", "Min": 20, "Max": 20, "Fat per 100 g": 10.0},
            {"Name": "Fatty", "Min": 0, "Max": 100, "Fat per 100 g": 30.0},
        ]))
        opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
        with pytest.raises(ValueError, match="No formulation can meet a limit "
                                             "on Fat per 100 g while Lean is "
                                             "fixed"):
            opt.add_constraint("Fat per 100 g", min_val=29.0)
        assert opt.constraints == []

    def test_a_limit_nothing_could_ever_meet_is_still_the_rounds_business(
            self, tmp_path, monkeypatch):
        """With no fixed row the door stays open: a minimum above every
        ingredient's figure may be half of an edit the reader is still
        making, and Generate has always been where that is answered."""
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("no_fixed_row")
        opt.load_ingredients_from_csv(pd.DataFrame([
            {"Name": "Lean", "Min": 0, "Max": 100, "Fat per 100 g": 1.0},
            {"Name": "Fatty", "Min": 0, "Max": 100, "Fat per 100 g": 1.0},
        ]))
        opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
        opt.add_constraint("Fat per 100 g", min_val=15.0)
        assert len(opt.constraints) == 1

    def test_the_batch_sizes_own_limit_keeps_its_own_refusal(self, tmp_path,
                                                             monkeypatch):
        """set_formulation_total asks the reach question first, in the two
        numbers the box was typed into. That answer is the better one, so the
        door stands aside for it."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_ingredient("Salt", 2.0, 2.0)
        with pytest.raises(ValueError) as refused:
            opt.set_formulation_total(500.0)
        assert "62 g" in str(refused.value)
        assert opt.formulation_total is None
        # ...and a size the amounts do reach is written as always.
        opt.set_formulation_total(50.0)
        assert opt.formulation_total == 50.0


class TestFeasibilityBlamesOnlyWhatThisSaveBroke:
    """Fix round 1, finding 3. A limit stranded last week — by a narrowing,
    which has always been allowed — is not this save's doing, and refusing
    for it leaves the reader with no way to edit their way back out."""

    def _opt(self, tmp_path, monkeypatch, name="blame"):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer(name)
        opt.set_amount_unit("g")
        opt.add_ingredient("Pea protein", 0, 60)
        opt.add_ingredient("Salt", 0, 10)
        opt.add_process_parameter("Cook temperature", 150, 200, unit="°C")
        opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
        return opt

    def test_an_already_stranded_limit_does_not_refuse_an_unrelated_fix(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_quantity_constraint(["Pea protein"], min_val=50.0)
        opt.add_ingredient("Pea protein", 0, 20)      # narrowing: allowed
        assert opt._limit_refusals()                  # the limit is stranded
        opt.add_process_parameter("Cook temperature", 175, 175, unit="°C")
        assert opt._var_by_name("Cook temperature")['bounds'] == (175.0, 175.0)

    def test_the_limit_this_save_does_break_is_still_refused_by_name(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_quantity_constraint(["Salt"], max_val=10.0)
        with pytest.raises(ValueError) as refused:
            opt.add_ingredient("Salt", 20.0, 20.0)
        message = str(refused.value)
        assert "limit on Salt" in message
        assert message.endswith("Fixed at one amount: Salt.")
        assert opt._var_by_name("Salt")['bounds'] == (0.0, 10.0)

    def test_both_at_once_blames_only_the_one_that_broke(self, tmp_path,
                                                         monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_quantity_constraint(["Pea protein"], min_val=50.0)
        opt.add_quantity_constraint(["Salt"], max_val=10.0)
        opt.add_ingredient("Pea protein", 0, 20)      # strands the first
        with pytest.raises(ValueError) as refused:
            opt.add_ingredient("Salt", 20.0, 20.0)
        assert "limit on Salt" in str(refused.value)
        assert "Pea protein" not in str(refused.value)


def test_the_grid_fixes_a_setting_that_carries_a_baseline(tmp_path,
                                                          monkeypatch):
    """Fix round 1, finding 2. The grid sends the STORED baseline straight
    back with every row, so "no baseline was passed" was never the test. What
    matters is whether the reader moved it."""
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("grid_baseline")
    opt.set_amount_unit("g")
    opt.add_ingredient("Water", 0, 100)
    opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
    opt.tell({"Water": 50.0}, {"Taste": 5.0})
    opt.add_process_parameter("Cook temperature", 150, 200, baseline=175,
                              unit="°C")
    frame = opt.ingredient_grid_frame()
    row = next(i for i in frame.index
               if frame.loc[i, wording.NAME_LABEL] == "Cook temperature")
    assert frame.loc[row, wording.BASELINE_LABEL] == 175.0   # sent back as is
    errors, _ = opt.apply_ingredient_grid(
        _edit(frame, row, **{wording.LOWEST_LABEL: 190.0,
                             wording.HIGHEST_LABEL: 190.0}))
    assert errors == []
    var = opt._var_by_name("Cook temperature")
    assert var['bounds'] == (190.0, 190.0)
    assert var['_absent_value'] == 175.0        # the past is not rewritten


def test_a_baseline_the_reader_moves_outside_a_fixed_range_is_still_refused(
        tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("moved_baseline")
    opt.add_ingredient("Water", 0, 100)
    opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
    opt.tell({"Water": 50.0}, {"Taste": 5.0})
    opt.add_process_parameter("Cook temperature", 150, 200, baseline=175,
                              unit="°C")
    with pytest.raises(ValueError, match="must be between"):
        opt.add_process_parameter("Cook temperature", 190, 190, baseline=160,
                                  unit="°C")
    assert opt._var_by_name("Cook temperature")['bounds'] == (150.0, 200.0)


def test_a_fixed_columns_frame_survives_a_history_that_barely_moved(
        tmp_path, monkeypatch):
    """Fix round 1, finding 4. A history spanning 1e-13 is a span the
    normalization would divide by."""
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("hairline", robust=False)
    opt.set_amount_unit("g")
    opt.add_ingredient("Water", 0, 100)
    opt.add_ingredient("Salt", 0, 10)
    opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
    for i in range(5):
        opt.tell({"Water": 40.0 + i, "Salt": 2.0 + i * 1e-13}, {"Taste": 5.0})
    opt.add_ingredient("Salt", 2.0, 2.0)
    low, high = opt._search_bounds()[1]
    assert high - low == pytest.approx(1.0)
    assert all(row["Salt"] == pytest.approx(2.0)
               for row in opt.ask(n_suggestions=2))


def test_a_categorical_row_keeps_what_the_migration_cannot_read(tmp_path,
                                                                monkeypatch):
    """Fix round 1, finding 4. A categorical has options rather than a range,
    so there is no pair of numbers to write a pin into — and stripping the
    keys it was stored with would lose the only record that it had them."""
    monkeypatch.chdir(tmp_path)
    var = {'name': "Starch", 'type': 'categorical', 'options': ["A", "B"],
           'active': False, '_frozen_at': 1.0, 'category': 'ingredient'}
    FoodOptimizer._migrate_fixed(var)
    assert var['active'] is False and var['_frozen_at'] == 1.0


# ------------------------------------------------------------------ #
#  0.5.0 · the properties grid (spec 1.5)
#
#  The third grid on tab 1 and the smallest: rows are the ingredients,
#  columns are the properties, and a cell is that ingredient's figure per
#  100 g. It has no rows of its own to add or delete — the ingredients grid
#  above owns the rows and `Add a property` owns the columns — so what is
#  pinned here is the figures, the two doors beside it, and the refusals.
# ------------------------------------------------------------------ #

class TestThePropertiesGrid:

    def _opt(self, tmp_path, monkeypatch, name="props"):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer(name)
        opt.set_amount_unit("g")
        opt.add_ingredient("Pea protein", 0, 40)
        opt.add_ingredient("Water", 0, 100)
        opt.add_process_parameter("Cook temperature", 150, 200, unit="°C")
        opt.add_property("Cost")
        opt.add_property("Sodium per 100 g")
        return opt

    def test_the_grid_is_ingredients_down_and_properties_across(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_property_value("Pea protein", "Cost", 3.5)
        frame = opt.property_grid_frame()
        assert list(frame.columns) == ["Ingredient", "Cost",
                                       "Sodium per 100 g"]
        # A process setting is weighed into nothing, so it has no row.
        assert list(frame["Ingredient"]) == ["Pea protein", "Water"]
        assert frame.loc[1, "Cost"] == 3.5
        # No figure is not a 0: the cell is empty, and every limit on the
        # property names the ingredients it is reading as zeroes.
        assert pd.isna(frame.loc[2, "Cost"])
        # Numbered from 1, as both grids above are, so "Row 2" under it
        # names the second row the reader can see.
        assert list(frame.index) == [1, 2]

    def test_a_figure_typed_into_a_cell_is_written(self, tmp_path,
                                                   monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        errors, messages = opt.apply_property_grid(
            _edit(opt.property_grid_frame(), 2, **{"Cost": 0.1}))
        assert errors == []
        assert messages == [("success", wording.PROPERTIES_SAVED)]
        assert FoodOptimizer(opt.project_name).property_value(
            "Water", "Cost") == 0.1

    def test_a_cell_emptied_clears_the_figure(self, tmp_path, monkeypatch):
        """A 0 is a figure; a blank is not, and the two must not read alike
        — an ingredient with no figure is named on the limit's own line."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_property_value("Pea protein", "Cost", 3.5)
        errors, _ = opt.apply_property_grid(
            _edit(opt.property_grid_frame(), 1, **{"Cost": None}))
        assert errors == []
        reloaded = FoodOptimizer(opt.project_name)
        assert not reloaded.has_property_value("Pea protein", "Cost")
        assert reloaded.ingredients_without_property("Cost") == ["Pea protein",
                                                                 "Water"]

    def test_a_grid_saved_unchanged_writes_nothing_and_says_nothing(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_property_value("Pea protein", "Cost", 3.5)
        before = json.dumps(opt.export_json(), sort_keys=True)
        errors, messages = opt.apply_property_grid(opt.property_grid_frame())
        assert (errors, messages) == ([], [])
        assert json.dumps(opt.export_json(), sort_keys=True) == before

    def test_a_cell_that_is_not_a_number_is_refused_by_row(self, tmp_path,
                                                           monkeypatch):
        """And nothing at all is written: the refusal names the row the
        reader can see and the column they typed into."""
        opt = self._opt(tmp_path, monkeypatch)
        before = json.dumps(opt.export_json(), sort_keys=True)
        errors, messages = opt.apply_property_grid(
            _edit(opt.property_grid_frame(), 2,
                  **{"Cost": "cheap", "Sodium per 100 g": 4.0}))
        assert errors == [(2, "Cost must be a number, or empty.")]
        assert messages == []
        assert json.dumps(opt.export_json(), sort_keys=True) == before

    def test_a_row_that_is_not_an_ingredient_is_refused(self, tmp_path,
                                                        monkeypatch):
        """The name column cannot be typed into on screen, so this is the
        refusal a grid arriving from anywhere else gets."""
        opt = self._opt(tmp_path, monkeypatch)
        errors, _ = opt.apply_property_grid(
            _edit(opt.property_grid_frame(), 2,
                  **{"Ingredient": "Semolina", "Cost": 2.0}))
        assert errors == [(2, "No ingredient named Semolina.")]

    def test_a_property_added_is_a_new_column_of_blanks(self, tmp_path,
                                                        monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_property("Fat per 100 g")
        frame = opt.property_grid_frame()
        assert list(frame.columns)[-1] == "Fat per 100 g"
        assert list(frame["Fat per 100 g"]) == [None, None]

    def test_a_property_deleted_takes_its_column_and_its_figures(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_property_value("Pea protein", "Cost", 3.5)
        opt.add_constraint("Cost", max_val=2.0)
        gone = opt.remove_property("Cost")
        assert len(gone) == 1
        frame = opt.property_grid_frame()
        assert list(frame.columns) == ["Ingredient", "Sodium per 100 g"]
        assert FoodOptimizer(opt.project_name).property_value(
            "Pea protein", "Cost") == 0.0

    def test_the_row_column_is_a_name_nothing_else_may_take(
            self, tmp_path, monkeypatch):
        """Two columns of one name is a frame nothing can read a cell out
        of, so the grid's own head is a reserved name — for a property and
        for an ingredient alike."""
        opt = self._opt(tmp_path, monkeypatch)
        with pytest.raises(ValueError, match="column name"):
            opt.add_property("Ingredient")
        with pytest.raises(ValueError, match="column name"):
            opt.add_ingredient("Ingredient", 0, 10)
        # ...and one that arrived as a column of an ingredient file is kept
        # in the project and left off the grid rather than breaking it.
        opt.ingredient_properties["Water"] = {"Ingredient": 1.0}
        assert "Ingredient" in opt.properties()
        assert list(opt.property_grid_frame().columns) == [
            "Ingredient", "Cost", "Sodium per 100 g"]

    def test_grid_properties_is_properties_minus_the_row_column(
            self, tmp_path, monkeypatch):
        """The one accessor `property_grid_frame`, `apply_property_grid` and
        the limit picker on screen all share, so a property that arrived as
        the row column's own name is invisible and uneditable in exactly
        the same places — and nowhere `properties()` itself is asked, since
        the delete picker still has to name it so it can be removed."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.ingredient_properties["Water"] = {"Ingredient": 1.0}
        assert opt.properties() == ["Cost", "Sodium per 100 g", "Ingredient"]
        assert opt.grid_properties() == ["Cost", "Sodium per 100 g"]

    def test_only_the_cells_that_moved_are_written(self, tmp_path,
                                                   monkeypatch):
        """Every figure goes through set_property_value, and that door saves
        the project; a grid of nine grid rows by six properties would
        save it forty-eight times to change one number."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_property_value("Pea protein", "Cost", 3.5)
        saves = []
        original = FoodOptimizer.save

        def counting_save(self, *args, **kwargs):
            saves.append(1)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(FoodOptimizer, "save", counting_save)
        errors, _ = opt.apply_property_grid(
            _edit(opt.property_grid_frame(), 2, **{"Cost": 0.1}))
        assert errors == []
        assert len(saves) == 1, saves


class TestTheDisplayRescaleIsOnlyForOlderRounds:
    """Task 4: whether shown_recipe's rescale is still doing anything.

    Until 0.5.0 the Batch size box scaled the PICTURE — the stored rows kept
    the amounts the model proposed and `shown_recipe` rewrote them on the
    way out — so the rescale was how the bench saw what it had weighed. Task
    1 moved the amounts themselves (`scale_round`), which makes the rescale
    a no-op for every round made since. It is not a no-op for a round
    recorded before that, which is why it stays.
    """

    def _opt(self, tmp_path, monkeypatch, name="rescale"):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer(name)
        opt.set_amount_unit("g")
        opt.add_ingredient("Pea protein", 0, 100)
        opt.add_ingredient("Water", 0, 100)
        opt.add_objective("Firmness", 1.0, goal="max", min_val=0, max_val=10)
        return opt

    def test_a_sized_round_already_sums_to_its_size_before_anything_is_shown(
            self, tmp_path, monkeypatch):
        """The stored rows ARE the numbers on the sheet, so rescaling them to
        the size they already weigh changes nothing."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_pending_batch([{"Pea protein": 20.0, "Water": 30.0},
                               {"Pea protein": 10.0, "Water": 30.0}])
        opt.scale_round(150.0)
        size = opt.open_round_size()
        assert size == 150.0
        for row in opt.pending_batch:
            assert opt.ingredient_total(row['recipe']) == pytest.approx(150.0)
            # ...and the rescale on the way to the screen is the identity.
            shown, _ = opt.shown_recipe(row, size)
            assert shown == pytest.approx(row['recipe'])

    def test_a_round_recorded_before_0_5_0_still_needs_the_rescale(
            self, tmp_path, monkeypatch):
        """Its stored amounts are as generated and its stored size is what
        the sheet was printed to. Tab 3 hands back what the bench weighed,
        so the rescale is the only thing that can produce those numbers."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.tell({"Pea protein": 20.0, "Water": 30.0}, {"Firmness": 6.0},
                 formulation_no=1, batch_no=1)
        # What an older version wrote: the picture's total, beside rows that
        # never moved.
        opt._batch_totals()[1] = 150.0
        opt.save()
        reloaded = FoodOptimizer(opt.project_name)
        size = reloaded.recorded_total(1)
        assert size == 150.0
        shown, basis = reloaded.shown_recipe(
            {'formulation': 1, 'recipe': reloaded.recipe_history[0]}, size)
        assert basis == 150.0
        assert reloaded.ingredient_total(shown) == pytest.approx(150.0)
        assert shown["Pea protein"] == pytest.approx(60.0)


# ------------------------------------------------------------------ #
#  0.5.0 §1.6: the workbook is locked where the app will not read it.
#  The cells a bench writes in are unlocked and shaded, two of them are
#  new — the Lot and the Actual weight — and both come back.
# ------------------------------------------------------------------ #

def _unlocked(sheet):
    """Every cell of a sheet a pen — or a keyboard — can still reach."""
    return {cell.coordinate for row in sheet.iter_rows() for cell in row
            if not cell.protection.locked}


class TestTheLockedWorkbook:
    """A sheet that can be overtyped anywhere comes back as results for a
    formulation the app never suggested, and nothing in the file says so."""

    def _opt(self, tmp_path, monkeypatch, name="locked"):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer(name)
        for field in ("vendor", "sku", "lot", "actual"):
            opt.set_records(field, True)
        opt.set_amount_unit("g")
        opt.add_ingredient("Pea protein", 0, 100)
        opt.add_ingredient("Water", 0, 100)
        opt.add_process_parameter("Cook temperature", 100, 220, unit="°C")
        opt.add_objective("Firmness", 1.5, goal="target", target=6,
                          min_val=0, max_val=10, unit="N")
        opt.add_objective("Juiciness", 1.0, goal="max", min_val=0, max_val=10,
                          unit="/10")
        opt.set_pending_batch([
            {"Pea protein": 20.0, "Water": 80.0, "Cook temperature": 180.0},
            {"Pea protein": 30.0, "Water": 70.0, "Cook temperature": 190.0},
        ], batch_no=2)
        return opt

    def _rows_of(self, sheet, column=1):
        """{label: its row number} down one column of a sheet."""
        return {sheet.cell(row=r, column=column).value: r
                for r in range(1, sheet.max_row + 1)
                if isinstance(sheet.cell(row=r, column=column).value, str)}

    # ---- protection ---------------------------------------------------- #

    def test_every_sheet_of_the_round_is_protected(self, tmp_path,
                                                   monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        book = _book(opt.workbook_bytes(opt.pending_batch, 100.0))
        assert [s.protection.sheet for s in book.worksheets] == \
            [True] * len(book.worksheets), book.sheetnames

    def test_the_summary_unlocks_exactly_the_cells_the_app_reads(
            self, tmp_path, monkeypatch):
        """Measured, Not scored, Note — one per formulation — and one Lot per
        ingredient. Nothing else: an amount overtyped on the way to the bench
        is a formulation the app never suggested."""
        opt = self._opt(tmp_path, monkeypatch)
        sheet = _book(opt.workbook_bytes(opt.pending_batch, 100.0))["Round 2"]
        at = self._rows_of(sheet)
        write_in = ("Firmness (N) · Target 6 N",
                    "Juiciness (/10) · Prefer higher values",
                    wording.NOT_SCORED_CHECKBOX_SHEET, wording.NOTE)
        expected = {f"{letter}{at[label]}" for label in write_in
                    for letter in ("B", "D")}          # one per formulation
        expected |= {f"F{at[name]}" for name in ("Pea protein (g)",
                                                 "Water (g)")}   # the Lot
        # ...and the signature line, which the app does not read but a pen
        # must reach: it was printed into a locked cell.
        expected |= {f"A{at[wording.MADE_BY_FOOTER]}"}
        assert _unlocked(sheet) == expected, sorted(_unlocked(sheet))
        # And the shading says the same thing, so nobody finds out by being
        # refused.
        assert all(sheet[c].fill.fgColor.rgb.endswith("FFF2CC")
                   for c in expected), expected

    def test_a_formulation_page_unlocks_the_actual_cells_and_no_amount(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        sheet = _book(opt.workbook_bytes(opt.pending_batch,
                                         100.0))["Formulation 1"]
        at = self._rows_of(sheet, column=2)
        expected = {f"D{at[label]}" for label in
                    ("Pea protein", "Water",            # what to weigh
                     "Cook temperature (°C)",           # what to dial in
                     "Firmness (N)", "Juiciness (/10)",  # what to measure
                     wording.NOT_SCORED_CHECKBOX_SHEET, wording.NOTE)}
        # Both note cells: the app reads the printed one back as its own, and
        # a technician who corrects it there is not writing to no effect.
        expected.add(f"C{at[wording.NOTE]}")
        # And the tick boxes: a sheet filled in on a screen has to be
        # tickable on the screen.
        expected |= {f"A{at[name]}" for name in ("Pea protein", "Water")}
        # And the signature line, which the app does not read but a pen
        # must reach: it was printed into a locked cell.
        expected.add(f"B{at[wording.MADE_BY_FOOTER]}")
        # The Total line's own Actual cell: a patty that came off the bench
        # at 97.4 g is a fact the model needs, and every row above it had
        # one while the line they add up to had none.
        expected.add(f"D{at[wording.TOTAL_LABEL]}")
        # And the tail of the merged Note box.
        expected.add(f"E{at[wording.NOTE]}")
        assert _unlocked(sheet) == expected, sorted(_unlocked(sheet))
        # The amount beside the Actual cell is locked.
        amount = sheet.cell(row=at["Water"], column=3)
        assert amount.value == 80.0 and amount.protection.locked
        # The tick cell carries its printed box and the write-in shade.
        tick = sheet.cell(row=at["Water"], column=1)
        assert tick.value == wording.TICK_BOX
        assert tick.fill.fgColor.rgb.endswith("FFF2CC")

    # ---- Lot, Actual, and the trip back -------------------------------- #

    def test_the_summary_carries_a_lot_cell_for_each_ingredient(
            self, tmp_path, monkeypatch):
        """One lot per ingredient for the whole round, not one per
        formulation: a round is weighed out of the sacks open that morning.
        A setting has none — nothing is poured out of a cook temperature."""
        opt = self._opt(tmp_path, monkeypatch)
        sheet = _book(opt.workbook_bytes(opt.pending_batch, 100.0))["Round 2"]
        header = self._rows_of(sheet)[wording.INGREDIENT_OR_SETTING_LABEL]
        assert sheet.cell(row=header, column=6).value == wording.LOT_COLUMN
        at = self._rows_of(sheet)
        assert sheet.cell(row=at["Cook temperature (°C)"],
                          column=6).value is None
        assert sheet.cell(row=at["Cook temperature (°C)"],
                          column=6).protection.locked

    def test_where_it_was_bought_is_printed_and_never_read(self, tmp_path,
                                                           monkeypatch):
        """Vendor and SKU are columns on the summary — beside the Lot, which
        is the same question asked of the same row — and a grey line under
        the name on the page the bench carries, which stays one page wide."""
        opt = self._opt(tmp_path, monkeypatch)
        opt._var_by_name("Water")['vendor'] = "Acme"
        opt._var_by_name("Water")['sku'] = "W-1"
        book = _book(opt.workbook_bytes(opt.pending_batch, 100.0))
        summary = book["Round 2"]
        at = self._rows_of(summary)
        header = at[wording.INGREDIENT_OR_SETTING_LABEL]
        assert [summary.cell(row=header, column=c).value for c in (7, 8)] == \
            [wording.VENDOR_LABEL, wording.SKU_LABEL]
        assert [summary.cell(row=at["Water (g)"], column=c).value
                for c in (7, 8)] == ["Acme", "W-1"]
        # Nothing for the ingredient the project says nothing about.
        assert summary.cell(row=at["Pea protein (g)"], column=7).value is None
        page = book["Formulation 1"]
        under = self._rows_of(page, column=2)["Water"] + 1
        assert page.cell(row=under, column=2).value == "Acme · W-1"

    def _filled(self, opt, total=100.0, actual=None, lot="L-7"):
        """The workbook back off the bench: both formulations measured, one
        ingredient weighed heavy on Formulation 1, and a lot written down."""
        book = _book(opt.workbook_bytes(opt.pending_batch, total))
        summary = book[wording.batch_sheet_name(opt.pending_batch_no)]
        at = {summary.cell(row=r, column=1).value: r
              for r in range(1, summary.max_row + 1)}
        for column in (2, 4):
            summary.cell(row=at["Firmness (N) · Target 6 N"], column=column,
                         value=6.0)
            summary.cell(row=at["Juiciness (/10) · Prefer higher values"],
                         column=column, value=7.0)
        if lot is not None:
            summary.cell(row=at["Water (g)"], column=6, value=lot)
        if actual is not None:
            page = book["Formulation 1"]
            rows = {page.cell(row=r, column=2).value: r
                    for r in range(1, page.max_row + 1)}
            page.cell(row=rows["Water"], column=4, value=actual)
        out = io.BytesIO()
        book.save(out)
        out.seek(0)
        return out

    def _save(self, opt, upload):
        """The upload path the round screen runs on Save: the parsed rows,
        the amounts as weighed, and the lots kept with the round."""
        parsed = opt.parse_batch_results(upload.frame, opt.pending_batch,
                                         weighed=upload.actual)
        by_number = {r['formulation']: r['recipe'] for r in opt.pending_batch}
        batch_no = opt.pending_batch_no
        for number, results, note in parsed:
            opt.tell(opt.amounts_as_weighed(by_number[number],
                                            upload.actual.get(number)),
                     results, formulation_no=number, batch_no=batch_no,
                     note=note)
        opt.store_lots(batch_no, upload.lots)
        return parsed

    def test_what_was_weighed_is_what_is_recorded_and_the_note_says_so(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        upload = opt.results_from_workbook(self._filled(opt, actual=82.5))
        # The three things the file carried, in the open: the rows, what was
        # weighed, and the lots.
        assert upload.actual == {1: {"Water": 82.5}}
        assert upload.lots == {"Water": "L-7"}
        self._save(opt, upload)
        # Formulation 1 came back with 82.5 g of water in it, and the row
        # says why its amounts are not the ones that were suggested.
        assert opt.recipe_history[0] == {"Pea protein": 20.0, "Water": 82.5,
                                         "Cook temperature": 180.0}
        assert opt.notes_history[0] == wording.AMOUNTS_AS_WEIGHED
        # A blank Actual cell means "as printed", on the same sheet.
        assert opt.recipe_history[1] == {"Pea protein": 30.0, "Water": 70.0,
                                         "Cook temperature": 190.0}
        assert opt.notes_history[1] == ""
        # The lot is kept with the round, never with the formulation.
        assert opt.lots == {2: {"Water": "L-7"}}
        assert FoodOptimizer(opt.project_name).lots == {2: {"Water": "L-7"}}

    def test_a_note_the_bench_wrote_keeps_it_behind_the_marker(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        book = openpyxl.load_workbook(self._filled(opt, actual=82.5))
        summary = book["Round 2"]
        row = next(r for r in range(1, summary.max_row + 1)
                   if summary.cell(row=r, column=1).value == wording.NOTE)
        summary.cell(row=row, column=2, value="lumpy")
        out = io.BytesIO()
        book.save(out)
        out.seek(0)
        self._save(opt, opt.results_from_workbook(out))
        assert opt.notes_history[0] == "Actual amounts · lumpy"

    def test_a_workbook_a_spreadsheet_re_saved_still_imports(self, tmp_path,
                                                             monkeypatch):
        """A technician opens the file, fills it in and presses save: openpyxl
        rewrites every sheet, and the app still finds its own cells."""
        opt = self._opt(tmp_path, monkeypatch)
        again = io.BytesIO()
        openpyxl.load_workbook(self._filled(opt, actual=82.5)).save(again)
        again.seek(0)
        upload = opt.results_from_workbook(again)
        assert list(upload.frame["Formulation"]) == [1, 2]
        self._save(opt, upload)
        assert opt.recipe_history[0]["Water"] == 82.5
        assert opt.lots == {2: {"Water": "L-7"}}

    def test_the_lots_sheet_lists_what_each_round_was_weighed_from(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        self._save(opt, opt.results_from_workbook(self._filled(opt)))
        book = _book(opt.all_formulations_workbook())
        assert wording.LOTS_SHEET in book.sheetnames, book.sheetnames
        rows = _rows(book[wording.LOTS_SHEET])
        assert rows[0] == (wording.ROUND_CAP, wording.KIND_INGREDIENT,
                           wording.LOT_COLUMN), rows[0]
        assert rows[1] == (2, "Water", "L-7"), rows[1]
        # The formulations table gains nothing: a lot belongs to a round.
        header = list(_rows(book[wording.ALL_FORMULATIONS_SHEET])[1])
        assert wording.LOT_COLUMN not in header, header

    def test_a_project_with_no_lots_has_no_lots_sheet(self, tmp_path,
                                                      monkeypatch):
        """A sheet of headers and nothing under them is a sheet nobody
        reads."""
        opt = self._opt(tmp_path, monkeypatch)
        self._save(opt, opt.results_from_workbook(self._filled(opt, lot=None)))
        assert wording.LOTS_SHEET not in \
            _book(opt.all_formulations_workbook()).sheetnames

    def test_the_instruction_line_fits_the_printed_page(self, tmp_path,
                                                        monkeypatch):
        """A line left in one column is cut off at the print edge, and an
        instruction the page ends halfway through is worse than none. It is
        merged across the sheet, wrapped, and given a row to wrap into —
        and it names that sheet's own cells, nobody else's."""
        opt = self._opt(tmp_path, monkeypatch)
        book = _book(opt.workbook_bytes(opt.pending_batch, 100.0))
        for sheet, row, note in ((book["Round 2"], 2,
                                  wording.SUMMARY_SHADED_NOTE),
                                 (book["Formulation 1"], 3,
                                  wording.sheet_write_in_note("Actual (g)"))):
            last = sheet.print_area.split(":")[-1].strip("'$0123456789")
            merged = [str(m) for m in sheet.merged_cells.ranges]
            assert f"A{row}:{last}{row}" in merged, (sheet.title, merged)
            cell = sheet.cell(row=row, column=1)
            assert cell.value == note
            assert cell.alignment.wrap_text
            assert sheet.row_dimensions[row].height >= 30
        # Lot is named where the Lot cells are, Actual where they are.
        assert wording.ACTUAL_COLUMN in wording.SHEET_SHADED_NOTE
        assert wording.LOT_COLUMN not in wording.SHEET_SHADED_NOTE
        assert wording.LOT_COLUMN in wording.SUMMARY_SHADED_NOTE
        # And the line the measurements block carries is merged too.
        page = book["Formulation 1"]
        row = self._rows_of(page, column=2)[wording.SHEET_WRITE_IN_NOTE]
        assert any(str(m).startswith(f"B{row}:")
                   for m in page.merged_cells.ranges), page.merged_cells

    def test_a_long_vendor_name_does_not_stretch_the_sheet(self, tmp_path,
                                                           monkeypatch):
        """Vendor and SKU are as wide as what they say — and no wider than a
        column a printed page can carry."""
        opt = self._opt(tmp_path, monkeypatch)
        opt._var_by_name("Water")['vendor'] = "A supplier with a very long name"
        opt._var_by_name("Water")['sku'] = "W-1"
        sheet = _book(opt.workbook_bytes(opt.pending_batch, 100.0))["Round 2"]
        # Two formulations and their shares, then Lot, Vendor and SKU.
        assert sheet.column_dimensions["G"].width == 30    # capped
        assert sheet.column_dimensions["H"].width == 12    # the short SKU

    def test_an_actual_cell_below_zero_is_refused_in_its_own_words(
            self, tmp_path, monkeypatch):
        """Nothing was ever weighed out of a bowl, and "is not a number" is
        the wrong thing to say about -80."""
        opt = self._opt(tmp_path, monkeypatch)
        with pytest.raises(ValueError, match="cannot be less than zero"):
            opt.results_from_workbook(self._filled(opt, actual=-80.0))

    def test_an_actual_cell_that_is_not_a_weight_is_refused(self, tmp_path,
                                                            monkeypatch):
        """Dropping it would file the printed amount under a formulation
        nobody made."""
        opt = self._opt(tmp_path, monkeypatch)
        with pytest.raises(ValueError, match="is not a number"):
            opt.results_from_workbook(self._filled(opt, actual="about 80"))

    def test_a_workbook_from_before_the_actual_column_reads_as_printed(
            self, tmp_path, monkeypatch):
        """The old layout: a summary sheet with no Lot column and formulation
        pages with no Actual column. Every amount is the one that was
        printed, and nothing about the round is lost."""
        opt = self._opt(tmp_path, monkeypatch)
        book = openpyxl.Workbook()
        summary = book.active
        summary.title = "Round 2"
        summary.cell(row=1, column=1, value="Ingredient or process setting")
        for c, number in ((2, 1), (3, 2)):
            summary.cell(row=1, column=c, value=f"Formulation {number}")
        summary.cell(row=2, column=1, value="Pea protein (g)")
        summary.cell(row=3, column=1, value=wording.MEASURED_COLUMN)
        summary.cell(row=4, column=1, value="Firmness (N) · Target 6 N")
        summary.cell(row=4, column=2, value=6.0)
        summary.cell(row=5, column=1, value="Juiciness (/10) · Prefer higher values")
        summary.cell(row=5, column=2, value=7.0)
        out = io.BytesIO()
        book.save(out)
        out.seek(0)
        upload = opt.results_from_workbook(out)
        assert list(upload.frame["Formulation"]) == [1]
        assert upload.actual == {} and upload.lots == {}
        self._save(opt, upload)
        assert opt.recipe_history[0] == {"Pea protein": 20.0, "Water": 80.0,
                                         "Cook temperature": 180.0}
        assert opt.notes_history[0] == ""       # nothing was weighed differently
        assert opt.lots == {}

    def test_a_copy_with_a_broken_lots_section_is_refused(self, tmp_path,
                                                          monkeypatch):
        """import_json walks the lots, so a bad one has to be caught before
        anything is assigned."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.lots = {2: {"Water": "L-7"}}
        state = opt.export_json()
        assert state['lots'] == {"2": {"Water": "L-7"}}
        FoodOptimizer.validate_state(state)      # the good one passes
        # A lot is text: a number written into that cell comes back as one,
        # and import_json would file 7 where the sheet said 007.
        for broken in ("L-7", {"two": {"Water": "L-7"}}, {"2": "L-7"},
                       {"2": {"Water": 7}}, {"2": {7: "L-7"}}):
            state['lots'] = broken
            with pytest.raises(ValueError, match="damaged"):
                FoodOptimizer.validate_state(state)

    def test_the_lots_follow_the_ingredient_list(self, tmp_path, monkeypatch):
        """A lot is filed against a round AND an ingredient, so a rename has
        to move it and a deletion has to take it: the Lots sheet printed
        'Round 1 · Water · L-1' for a project with no Water."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.lots = {1: {"Water": "L-1", "Pea protein": "L-2"}}
        opt.rename_variable("Water", "Cold water")
        assert opt.lots == {1: {"Cold water": "L-1", "Pea protein": "L-2"}}
        opt.remove_ingredient("Cold water", force=True)
        assert opt.lots == {1: {"Pea protein": "L-2"}}

    def test_a_round_that_is_undone_takes_its_lots_with_it(self, tmp_path,
                                                           monkeypatch):
        """A round number is never reissued, so a lot left behind could only
        ever be read against a round nobody can see."""
        opt = self._opt(tmp_path, monkeypatch)
        for row in list(opt.pending_batch):
            opt.tell(row['recipe'], {"Firmness": 6.0, "Juiciness": 7.0},
                     formulation_no=row['formulation'], batch_no=2)
        opt.store_lots(2, {"Water": "L-7"})
        opt.set_pending_batch(None)          # the round closes on Save
        assert opt.lots == {2: {"Water": "L-7"}}
        opt.undo_last_batch()
        assert opt.lots == {}

    def test_a_lot_on_an_ingredient_the_copy_does_not_have_is_refused(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        state = opt.export_json()
        state['lots'] = {"1": {"Beetroot": "L-9"}}
        with pytest.raises(ValueError, match="damaged"):
            FoodOptimizer.validate_state(state)


class TestFormulaGrammar:
    """Task 1 of the rules wave (2026-09-16): parse_formula reads a Formula
    cell's text as a LinearForm, the value type the Set up grid's balance
    and rules will be built on. No grid and no model here — just the
    grammar on its own."""

    NAMES = ["Water", "Salt", "Flour"]

    def test_a_formula_adds_and_subtracts_rows(self):
        form = parse_formula("= batch size - Water - Salt", self.NAMES,
                             has_batch_size=True)
        assert form == LinearForm(batch=1.0,
                                  terms={"Water": -1.0, "Salt": -1.0})
        assert form.names() == {"Water", "Salt"}
        assert not form.is_constant()

    def test_a_formula_multiplies_by_a_number_either_side(self):
        left = parse_formula("= 2 × Water", self.NAMES, has_batch_size=False)
        right = parse_formula("= Water × 2", self.NAMES, has_batch_size=False)
        assert left == LinearForm(terms={"Water": 2.0})
        assert right == LinearForm(terms={"Water": 2.0})

    def test_a_formula_divides_by_a_number(self):
        form = parse_formula("= Water ÷ 2", self.NAMES, has_batch_size=False)
        assert form == LinearForm(terms={"Water": 0.5})

    def test_brackets_and_a_leading_minus(self):
        form = parse_formula("= -(Water + Salt)", self.NAMES,
                             has_batch_size=False)
        assert form == LinearForm(terms={"Water": -1.0, "Salt": -1.0})

    def test_batch_size_is_one_token_not_two_names(self):
        form = parse_formula("= batch size", self.NAMES, has_batch_size=True)
        assert form == LinearForm(batch=1.0)
        assert form.names() == set()

    def test_a_name_that_contains_another_name_wins_longest_first(self):
        names = ["Cream", "Cream cheese"]
        form = parse_formula("= Cream cheese - Cream", names,
                             has_batch_size=False)
        assert form == LinearForm(terms={"Cream cheese": 1.0, "Cream": -1.0})

    @pytest.mark.parametrize("cell", ["= 1.5 % of batch size",
                                      "= 1.5% of batch size",
                                      "= 1.5 % OF batch size"])
    def test_a_percentage_of_the_batch_size_is_the_wording_on_screen(
            self, cell):
        """The Limits box above the grid writes this idea as
        "% of batch size", and the one control that can hold it for a single
        row refused the same words. 1.5 % is × 0.015."""
        form = parse_formula(cell, self.NAMES, has_batch_size=True)
        assert form == LinearForm(batch=0.015)

    def test_a_percentage_of_another_row_reads_that_row(self):
        form = parse_formula("= 50 % of Water", self.NAMES,
                             has_batch_size=False)
        assert form == LinearForm(terms={"Water": 0.5})

    def test_a_percentage_binds_like_a_multiplication(self):
        form = parse_formula("= 10 % of Water + Salt", self.NAMES,
                             has_batch_size=False)
        assert form == LinearForm(terms={"Water": 0.1, "Salt": 1.0})

    def test_a_percentage_of_an_amount_is_refused_like_any_other_product(
            self):
        with pytest.raises(FormulaError) as excinfo:
            parse_formula("= Water % of Salt", self.NAMES,
                          has_batch_size=False)
        assert str(excinfo.value) == wording.FORMULA_TWO_AMOUNTS

    def test_of_is_only_the_second_half_of_a_percentage(self):
        """A row really called 'of' is matched as a name, and the word on
        its own is not a formula."""
        with pytest.raises(FormulaError):
            parse_formula("= of", self.NAMES, has_batch_size=False)
        assert parse_formula("= of", ["of"], has_batch_size=False) == \
            LinearForm(terms={"of": 1.0})

    @pytest.mark.parametrize("dash", ["\u2013", "\u2014"])
    def test_a_dash_a_keyboard_makes_on_its_own_is_a_minus_sign(self, dash):
        """macOS smart dashes, Word and Excel all turn a typed '-' into an
        en or em dash. Refusing them printed a list of characters one of
        which looks identical to the one the reader had just typed."""
        form = parse_formula(f"= Water {dash} Salt", self.NAMES,
                             has_batch_size=False)
        assert form == LinearForm(terms={"Water": 1.0, "Salt": -1.0})

    @pytest.mark.parametrize("cell", ["= Water x 2", "= Water X 2",
                                      "= 2 x Water", "= 2x Water"])
    def test_a_lone_x_is_a_times_sign(self, cell):
        """A scientist types the letter x for ×. It was read as the start
        of an ingredient name and refused with "There is no ingredient
        called x 2." — a name they never typed."""
        assert parse_formula(cell, self.NAMES, has_batch_size=False) == \
            LinearForm(terms={"Water": 2.0})

    def test_a_name_that_contains_an_x_is_untouched(self):
        """Names are matched first and whole, so a lone x is only ever the
        one that is on its own."""
        names = ["Flax", "Xanthan gum"]
        assert parse_formula("= Flax x 2", names, has_batch_size=False) == \
            LinearForm(terms={"Flax": 2.0})
        assert parse_formula("= Xanthan gum x 3", names,
                             has_batch_size=False) == \
            LinearForm(terms={"Xanthan gum": 3.0})
        # A row really called x wins over the operator: the names are
        # matched before it, and this is what that is for.
        assert parse_formula("= x × 2", ["x"], has_batch_size=False) == \
            LinearForm(terms={"x": 2.0})

    def test_batch_size_typed_with_extra_space_is_still_batch_size(self):
        """The refusal printed the name back character for character as the
        one that works, and the reader could not see the difference."""
        assert parse_formula("= batch  size - Water", self.NAMES,
                             has_batch_size=True) == \
            LinearForm(batch=1.0, terms={"Water": -1.0})

    def test_a_rule_without_its_equals_says_which_character_is_missing(self):
        """A correct rule one character short was told only that it could
        not be read."""
        with pytest.raises(FormulaError) as excinfo:
            parse_formula("Water × 2", self.NAMES, has_batch_size=False)
        assert str(excinfo.value) == wording.RULE_NEEDS_EQUALS
        assert wording.RULE_NEEDS_EQUALS == "Start a calculation with =."

    def test_a_multiplication_dot_is_a_times_sign(self):
        form = parse_formula("= Water · 2", self.NAMES, has_batch_size=False)
        assert form == LinearForm(terms={"Water": 2.0})

    def test_two_amounts_multiplied_is_refused_in_the_spec_sentence(self):
        with pytest.raises(FormulaError) as excinfo:
            parse_formula("= Water × Salt", self.NAMES, has_batch_size=False)
        assert str(excinfo.value) == wording.FORMULA_TWO_AMOUNTS

    def test_dividing_by_an_amount_is_refused(self):
        with pytest.raises(FormulaError) as excinfo:
            parse_formula("= Water ÷ Salt", self.NAMES, has_batch_size=False)
        assert str(excinfo.value) == wording.FORMULA_DIVIDE_BY_AMOUNT

    def test_dividing_by_zero_is_refused(self):
        with pytest.raises(FormulaError) as excinfo:
            parse_formula("= Water ÷ 0", self.NAMES, has_batch_size=False)
        assert str(excinfo.value) == wording.FORMULA_DIVIDE_BY_ZERO

    def test_an_unknown_name_is_named_in_the_refusal(self):
        with pytest.raises(FormulaError) as excinfo:
            parse_formula("= Sodium citrate", self.NAMES,
                         has_batch_size=False)
        assert str(excinfo.value) == wording.formula_unknown_name(
            "Sodium citrate")

    def test_gibberish_is_refused_once(self):
        with pytest.raises(FormulaError) as excinfo:
            parse_formula("= @@@", self.NAMES, has_batch_size=False)
        assert str(excinfo.value) == wording.FORMULA_UNREADABLE

    def test_batch_size_with_no_default_is_refused_at_the_cell(self):
        with pytest.raises(FormulaError) as excinfo:
            parse_formula("= batch size - Water", self.NAMES,
                         has_batch_size=False)
        assert str(excinfo.value) == wording.FORMULA_NEEDS_BATCH_SIZE

    def test_rest_on_its_own_is_the_balance(self):
        assert parse_formula("= rest", self.NAMES,
                             has_batch_size=True) == LinearForm(rest=True)
        # Case-insensitive, and stray whitespace around the word is fine.
        assert parse_formula("=  REST  ", self.NAMES,
                             has_batch_size=True) == LinearForm(rest=True)

    def test_rest_with_anything_else_is_refused(self):
        for text in ("= rest + Water", "= Water + rest", "= rest × 2"):
            with pytest.raises(FormulaError) as excinfo:
                parse_formula(text, self.NAMES, has_batch_size=True)
            assert str(excinfo.value) == wording.FORMULA_REST_ALONE


class TestFormulaRows:
    """Task 2 of the rules wave (2026-09-16): a formula row leaves the search
    vector. Its amount is calculated from the rows it names, and its
    coefficients are folded into every limit — substitution, never an
    equality band, so the formula holds exactly at every point the app
    produces and the eliminated column carries no information the GP loses.
    """

    def _opt(self, tmp_path, monkeypatch, name="formulas"):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer(name)
        opt.set_amount_unit("g")
        opt.add_ingredient("Water", 0, 100)
        opt.add_ingredient("Flour", 10, 50)
        opt.add_ingredient("Sugar", 5, 30)
        opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
        return opt

    @staticmethod
    def _rows(opt):
        """Every inequality the model is handed, as plain numbers: the
        columns it reads, their coefficients, and the number the sum must
        stay at or above."""
        return [(idx.tolist(), co.tolist(), float(rhs))
                for idx, co, rhs in opt._get_botorch_constraints()]

    # -------------------------------------------------------------- #

    def test_a_formula_row_leaves_the_search_vector(self, tmp_path,
                                                    monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_formulation_total(100)
        opt.set_formula("Water", "= batch size - Flour - Sugar")
        assert opt.has_formula(opt._var_by_name("Water")) is True
        # Two columns, not three: Water is calculated, not searched.
        assert opt._encode({"Water": 60, "Flour": 30, "Sugar": 10}) == [
            30.0, 10.0]
        assert len(opt._search_bounds()) == 2
        assert [v['name'] for v in opt.varying_variables()] == ["Flour",
                                                               "Sugar"]
        # ...and decoding writes it back in.
        assert opt._decode([30.0, 10.0]) == {"Water": 60.0, "Flour": 30.0,
                                             "Sugar": 10.0}
        assert opt.fill_formulas({"Flour": 20.0, "Sugar": 5.0})["Water"] == 75.0

    def test_every_suggestion_obeys_its_formula_exactly_cold(self, tmp_path,
                                                             monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_formulation_total(100)
        opt.set_formula("Water", "= 0.5 × Flour")
        batch = opt.ask(3)
        assert len(batch) == 3
        for recipe in batch:
            assert recipe["Water"] == pytest.approx(0.5 * recipe["Flour"])
            assert sum(recipe.values()) == pytest.approx(100.0, abs=1e-6)

    def test_and_warm(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("warm_formulas")
        opt.set_amount_unit("g")
        opt.add_ingredient("Water", 0, 100)
        opt.add_ingredient("Flour", 10, 50)
        opt.add_ingredient("Sugar", 5, 80)
        opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
        opt.set_formulation_total(100)
        opt.set_formula("Water", "= 0.5 × Flour")
        for i, flour in enumerate([20.0, 25.0, 30.0, 35.0, 40.0]):
            sugar = 100 - 1.5 * flour
            opt.tell({"Water": 0.5 * flour, "Flour": flour, "Sugar": sugar},
                     {"Taste": 4.0 + i})
        assert len(opt.X_history) == 5
        assert len(opt.X_history[0]) == 2
        for recipe in opt.ask(2):
            assert recipe["Water"] == pytest.approx(0.5 * recipe["Flour"])

    def test_a_balance_row_makes_every_formulation_add_up_to_the_batch_size(
            self, tmp_path, monkeypatch):
        # Ranges the other two rows can overspend the batch size with:
        # 80 + 60 is 140 against 100, so the balance really can be asked
        # for less than none of the water and its floor is what stops it.
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("balance_rows")
        opt.set_amount_unit("g")
        opt.add_ingredient("Water", 0, 100)
        opt.add_ingredient("Flour", 10, 80)
        opt.add_ingredient("Sugar", 5, 60)
        opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
        opt.set_formulation_total(100)
        opt.set_formula("Water", "= rest")
        assert opt._var_by_name("Water")['balance'] is True
        assert opt._var_by_name("Water")['formula'] == "= rest"
        # Water is named, and named at what the balance makes it: the sum
        # is exactly 100, so the round's own total limit holds and the only
        # thing left to refuse it is the floor under the balance.
        assert opt._check_constraints({"Flour": 70.0, "Sugar": 50.0,
                                       "Water": -20.0}) is False
        for recipe in opt.ask(3):
            assert sum(recipe.values()) == pytest.approx(100.0, abs=1e-6)
            assert recipe["Water"] >= 0.0

    def test_a_balance_row_needs_no_snap(self, tmp_path, monkeypatch):
        """Every other amount is left exactly where the space-filling point
        put it: the balance makes the total hold identically, so there is
        no correction to share out."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_formulation_total(100)
        opt.set_formula("Water", "= rest")
        snapped = opt._snap_to_total({"Water": 0.0, "Flour": 22.0,
                                      "Sugar": 7.0}, 100)
        assert snapped["Flour"] == 22.0
        assert snapped["Sugar"] == 7.0
        assert snapped["Water"] == pytest.approx(71.0)

    def test_a_limit_naming_a_formula_row_is_substituted_not_dropped(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_formula("Water", "= 0.5 × Flour")
        # A limit every one of whose rows is calculated has nothing to act
        # on and is refused; one that also names a row the search moves is
        # substituted, which is what this is about.
        with pytest.raises(ValueError) as refused:
            opt.add_quantity_constraint(["Water"], max_val=15)
        assert str(refused.value) == wording.limit_on_worked_out_rows("Water")
        opt.add_quantity_constraint(["Water", "Sugar"], max_val=15)
        rows = self._rows(opt)
        # Flour is column 0 and Sugar column 1: Water + Sugar at most 15 is
        # 0.5 × Flour + Sugar ≤ 15, with Flour 10–50 and Sugar 5–30.
        assert ([0, 1], [-20.0, -25.0], -5.0) in rows
        # ...alongside the formula's own floor, 0.5 × Flour ≥ 0.
        assert ([0], [20.0], -5.0) in rows
        assert len(rows) == 2

    def test_a_limit_that_substitutes_to_a_constant_that_holds_is_dropped(
            self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("constant_holds")
        opt.set_amount_unit("g")
        opt.add_ingredient("Water", 0, 100)
        opt.add_ingredient("Flour", 10, 50)
        opt.set_formulation_total(100)
        opt.set_formula("Water", "= batch size - Flour")
        opt.add_quantity_constraint(["Water", "Flour"], max_val=150)
        # Both the limit just written and the batch size's own limit
        # substitute to the constant 100, which satisfies them: nothing is
        # left for the model to be told. Only the formula's floor remains.
        assert self._rows(opt) == [([0], [-40.0], -90.0)]

    def test_a_limit_that_substitutes_to_a_constant_that_fails_is_refused(
            self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("constant_fails")
        opt.set_amount_unit("g")
        opt.add_ingredient("Water", 0, 100)
        opt.add_ingredient("Flour", 10, 50)
        opt.set_formulation_total(100)
        opt.add_quantity_constraint(["Water", "Flour"], min_val=150)
        opt.set_formula("Water", "= batch size - Flour")
        with pytest.raises(ValueError) as caught:
            opt._get_botorch_constraints()
        assert str(caught.value) == wording.limit_unreachable_above(
            "Water + Flour", "100")

    def test_a_formula_row_never_goes_below_zero(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_formula("Sugar", "= 40 - Flour")
        assert opt._check_constraints({"Water": 10.0, "Flour": 30.0,
                                       "Sugar": 10.0}) is True
        assert opt._check_constraints({"Water": 10.0, "Flour": 45.0,
                                       "Sugar": -5.0}) is False
        for recipe in opt.ask(3):
            assert recipe["Sugar"] >= -1e-9
            assert recipe["Sugar"] == pytest.approx(40.0 - recipe["Flour"])

    def test_a_formula_that_is_always_below_zero_is_refused_at_save(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        with pytest.raises(ValueError) as caught:
            opt.set_formula("Sugar", "= 5 - Flour")
        assert str(caught.value) == wording.formula_below_zero("Sugar", "g")
        assert opt.has_formula(opt._var_by_name("Sugar")) is False

    def test_total_reach_folds_a_balance_row_in(self, tmp_path, monkeypatch):
        """A balance row takes whatever is left of the batch size, so there
        is no size these ingredients cannot make — only one too small for
        the other rows to fit inside. Reading the batch size at both ends
        said the same number was the floor and the ceiling, and the Batch
        size box then accepted nothing at all."""
        opt = self._opt(tmp_path, monkeypatch)
        assert opt.total_reach() == (15.0, 180.0)
        opt.set_formulation_total(100)
        opt.set_formula("Water", "= rest")
        assert opt.total_reach() == (15.0, float('inf'))
        # Flour 10-50 and Sugar 5-30 need 15 g between them, whatever the
        # size; anything at or above that is reachable.
        opt.set_formulation_total(250)
        assert opt.formulation_total == 250

    def test_a_rest_row_has_no_largest_batch_size(self, tmp_path,
                                                  monkeypatch):
        """The Batch size box on Make a round accepted no value at all: the
        same round was told its floor was 100 g and its ceiling was 100 g.
        A rest row takes whatever is left, so any size the other rows fit
        inside can be made."""
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("rest_reach", robust=False)
        opt.set_amount_unit("g")
        opt.load_ingredients_from_csv(pd.read_csv(_FLAT_BURGER_CSV))
        opt.add_objective("Juiciness", 1.0, goal="target", target=7,
                          min_val=0, max_val=10, unit="/10")
        opt.set_formulation_total(100.0)
        low, high = opt.total_reach()
        assert high == float('inf')
        assert low == 0.0        # every other row may be 0
        for size in (120.0, 250.0):
            opt.set_formulation_total(size)
            assert opt.formulation_total == size
        # And the floor, when there is one, is the least the OTHER rows
        # need — said in the balance's own words by every box that asks.
        opt.set_formulation_total(100.0)
        opt.apply_ingredient_grid(_edit(
            opt.ingredient_grid_frame(), 1,
            **{wording.LOWEST_LABEL: 20.0, wording.HIGHEST_LABEL: 25.0}))
        assert opt.total_reach() == (20.0, float('inf'))
        assert opt.balance_row_name() == "Water"
        with pytest.raises(ValueError) as caught:
            opt.set_formulation_total(15.0)
        assert str(caught.value) == wording.balance_would_go_negative(
            "Water", "15 g", "20 g", unit="g")

    def test_changing_the_default_says_what_the_rest_row_now_takes(
            self, tmp_path, monkeypatch):
        """Amounts written in grams do not follow the batch size and a rest
        row does, so one keystroke turned a burger into soup with nothing
        said about it but a deleted limit."""
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("rest_consequence", robust=False)
        opt.set_amount_unit("g")
        opt.add_ingredient("Water", 0, 100)
        opt.add_ingredient("Flour", 10, 50)
        opt.add_ingredient("Sugar", 20, 28)
        opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
        opt.set_formulation_total(100)
        opt.set_formula("Water", "= rest")
        opt.set_formulation_total(250)
        assert opt.batch_size_consequence() == (
            "Water takes up the difference: between 172.00 and 220.00 g in "
            "a 250 g formulation. To keep the same proportions, widen "
            "Lowest and Highest too.")
        # Nothing to say for a project with no rest row.
        opt.clear_formula("Water")
        assert opt.batch_size_consequence() == ""

    def test_a_batch_size_that_drives_the_balance_negative_is_refused(
            self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("balance_negative")
        opt.set_amount_unit("g")
        opt.add_ingredient("Water", 0, 100)
        opt.add_ingredient("Flour", 30, 50)
        opt.add_ingredient("Sugar", 40, 60)
        opt.set_formulation_total(100)
        opt.set_formula("Water", "= rest")
        with pytest.raises(ValueError) as caught:
            opt.set_formulation_total(60)
        assert str(caught.value) == wording.balance_would_go_negative(
            "Water", "60 g", "70 g", unit="g")
        assert opt.formulation_total == 100
        # A batch size the other amounts leave room in is written as ever.
        opt.set_formulation_total(120)
        assert opt.formulation_total == 120

    def test_a_constant_formula_is_a_formula_before_it_is_fixed(
            self, tmp_path, monkeypatch):
        """'= 0.15 × batch size' is one number at every point, so its
        Lowest is its Highest — and it is still a formula, not a fixed row
        pinned in the frame the search runs in."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_formulation_total(100)
        opt.add_ingredient("Starch", 15, 15)
        opt.set_formula("Starch", "= 0.15 × batch size")
        var = opt._var_by_name("Starch")
        assert opt.is_fixed(var) is True
        assert opt.has_formula(var) is True
        # It has no column at all, so it is not pinned inside one either.
        assert len(opt._search_bounds()) == 3
        assert opt._get_fixed_features() == {}
        assert "Starch" not in [v['name'] for v in opt.varying_variables()]
        assert opt.fill_formulas({"Water": 10.0, "Flour": 10.0,
                                  "Sugar": 5.0})["Starch"] == 15.0

    def test_a_circular_pair_is_refused_naming_the_loop(self, tmp_path,
                                                        monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_formula("Flour", "= Sugar × 2")
        with pytest.raises(FormulaError) as caught:
            opt.set_formula("Sugar", "= Flour ÷ 2")
        assert str(caught.value) == wording.formula_loop(
            "Flour → Sugar → Flour")
        # Refused means nothing was written: Flour's formula still reads.
        assert opt.has_formula(opt._var_by_name("Sugar")) is False
        assert opt._linear_form("Flour") == LinearForm(terms={"Sugar": 2.0})

    def test_a_formula_survives_export_and_import(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_formulation_total(100)
        opt.set_formula("Water", "= rest")
        opt.set_formula("Sugar", "= 0.1 × batch size")
        state = opt.export_json()
        assert FoodOptimizer.validate_state(state)['version'] == 15

        other = FoodOptimizer("copy_of_formulas")
        other.import_json(state)
        assert other._var_by_name("Water")['formula'] == "= rest"
        assert other._var_by_name("Water")['balance'] is True
        assert other._var_by_name("Sugar")['formula'] == "= 0.1 × batch size"
        assert other._var_by_name("Sugar")['balance'] is False
        assert other._linear_form("Sugar") == LinearForm(batch=0.1)
        assert [v['name'] for v in other.varying_variables()] == ["Flour"]

        # A copy that gives two rows the balance is refused: one row can
        # take it, and which one is not the app's guess to make.
        two = json.loads(json.dumps(state))
        two['variables'][2]['balance'] = True
        with pytest.raises(ValueError) as caught:
            FoodOptimizer.validate_state(two)
        assert str(caught.value) == wording.COPY_TWO_BALANCE_ROWS
        for key, value in (('formula', 12.0), ('balance', "yes")):
            bad = json.loads(json.dumps(state))
            bad['variables'][1][key] = value
            with pytest.raises(ValueError):
                FoodOptimizer.validate_state(bad)

    def test_a_copy_whose_formula_cannot_be_read_is_refused_at_the_door(
            self, tmp_path, monkeypatch):
        """'formula' was checked for TYPE and never for readability, so a
        copy naming a row it does not have imported cleanly and then raised
        under the grid on every render — above the Save button, so there
        was no way to edit out of it."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_formulation_total(100)
        opt.set_formula("Water", "= rest")
        state = opt.export_json()
        for text in ("= Nonexistent", "= Sugar ÷ 0", "= Sugar × Flour",
                     "not a formula"):
            bad = json.loads(json.dumps(state))
            bad['variables'][1]['formula'] = text
            with pytest.raises(ValueError):
                FoodOptimizer.validate_state(bad)

    def test_a_copy_whose_exactly_is_not_a_number_is_refused_at_the_door(
            self, tmp_path, monkeypatch):
        """limit_text prints Exactly with a 'g' format code, so a copy
        holding 'twelve' raised on every render of the Limits section and
        of the Set-up sheet."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_formulation_total(100)
        opt.add_quantity_constraint(["Flour", "Sugar"], exactly=12.0)
        state = opt.export_json()
        for value in ("twelve", True, [12]):
            bad = json.loads(json.dumps(state, default=str))
            for qc in bad['quantity_constraints']:
                if qc.get('source') != 'formulation_total':
                    qc['exactly'] = value
            with pytest.raises(ValueError):
                FoodOptimizer.validate_state(bad)

    def test_an_unreadable_formula_costs_its_caption_and_not_the_screen(
            self, tmp_path, monkeypatch):
        """Belt and braces behind validate_state: whatever route a project
        reached the grid by, the captions are drawn above the Save button
        and must leave the reader one."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_formulation_total(100)
        opt.set_formula("Water", "= rest")
        assert len(opt.worked_out_captions()) == 1
        opt._var_by_name("Sugar")['formula'] = "= Nonexistent"
        # No line, and no raise: the grid still draws, so there is still a
        # Save button under it to edit the cell out with.
        assert opt.worked_out_captions() == []
        assert sorted(opt.ingredient_grid_frame()[wording.NAME_LABEL]) == [
            "Flour", "Sugar", "Water"]

    def test_a_version_11_project_without_formulas_loads_at_version_12(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_formulation_total(100)
        state = opt.export_json()
        state['CLASS_VERSION'] = 11
        for var in state['variables']:
            var.pop('formula', None)
            var.pop('balance', None)
        assert FoodOptimizer.validate_state(state)['version'] == 11

        other = FoodOptimizer("wave_one_project")
        other.import_json(state)
        assert FoodOptimizer.CLASS_VERSION == 15
        assert not any(other.has_formula(v) for v in other.variables)
        assert other.export_json()['CLASS_VERSION'] == 15
        assert len(other._search_bounds()) == 3
        assert len(other.ask(1)) == 1

    def test_a_rename_moves_the_name_inside_every_formula(self, tmp_path,
                                                          monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_formulation_total(100)
        opt.set_formula("Water", "= batch size - Flour - Sugar")
        opt.rename_variable("Flour", "Rye flour")
        assert opt._var_by_name("Water")['formula'] == (
            "= batch size - Rye flour - Sugar")
        assert opt._linear_form("Water").names() == {"Rye flour", "Sugar"}

    def test_a_rename_leaves_a_longer_name_that_contains_it_alone(
            self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("longest_first")
        opt.set_amount_unit("g")
        opt.add_ingredient("Cream", 0, 100)
        opt.add_ingredient("Cream cheese", 0, 100)
        opt.add_ingredient("Salt", 0, 10)
        opt.set_formula("Salt", "= 0.1 × Cream + 0.2 × Cream cheese")
        opt.rename_variable("Cream", "Milk")
        assert opt._var_by_name("Salt")['formula'] == (
            "= 0.1 × Milk + 0.2 × Cream cheese")

    def test_deleting_a_row_a_formula_names_is_refused(self, tmp_path,
                                                       monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_formulation_total(100)
        opt.set_formula("Water", "= batch size - Flour - Sugar")
        with pytest.raises(ValueError) as caught:
            opt.remove_ingredient("Sugar")
        assert str(caught.value) == wording.formula_reads_this_row("Sugar",
                                                                  "Water")
        # Even with force: the formula would be left naming nothing.
        with pytest.raises(ValueError):
            opt.remove_ingredient("Sugar", force=True)
        assert "Sugar" in [v['name'] for v in opt.variables]
        # The formula row itself is nobody's dependency and goes as ever.
        opt.remove_ingredient("Water")
        assert "Water" not in [v['name'] for v in opt.variables]

    # ---------------------------------------------------------------- #
    #  Fix round 1
    # ---------------------------------------------------------------- #

    def test_sizing_a_round_works_its_formulas_out_again(self, tmp_path,
                                                         monkeypatch):
        """Scaling every amount by one factor is right for a formula that is
        a multiple and wrong for one with a number in it: the rows the
        search moves are scaled, and the rows that are calculated are worked
        out again at the new size.

        Which is why the factor is solved rather than taken as the ratio of
        the two sizes. `= 5 + 0.1 × Flour` keeps its 5 whatever the size,
        so doubling the other rows landed the round at 69 g when 74 g was
        asked for — and every row on the sheet then wore a caption
        apologising for it."""
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("sized_formulas")
        opt.set_amount_unit("g")
        opt.add_ingredient("Water", 0, 100)
        opt.add_ingredient("Flour", 0, 50)
        opt.add_ingredient("Sugar", 0, 50)
        opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
        opt.set_formula("Water", "= 5 + 0.1 × Flour")
        sized = opt.scaled_recipe({"Water": 7.0, "Flour": 20.0,
                                   "Sugar": 10.0}, 74)
        assert sum(sized.values()) == pytest.approx(74.0)
        # Flour and Sugar keep the proportion they came in, and Water is
        # calculated again rather than scaled.
        assert sized["Flour"] / sized["Sugar"] == pytest.approx(2.0)
        assert sized["Water"] == pytest.approx(5 + 0.1 * sized["Flour"])
        opt.set_pending_batch([{"Water": 7.0, "Flour": 20.0, "Sugar": 10.0}])
        opt.scale_round(74)
        sized = opt._batch_rows(opt.pending_batch)[0]['recipe']
        assert sum(sized.values()) == pytest.approx(74.0)
        assert opt._check_constraints(sized) is True
        # A purely proportional formula is the plain ratio, as it always was.
        opt.clear_formula("Water")
        opt.set_formula("Water", "= 0.5 × Flour")
        assert opt.scaled_recipe({"Water": 10.0, "Flour": 20.0,
                                  "Sugar": 10.0}, 80) == {
            "Water": 20.0, "Flour": 40.0, "Sugar": 20.0}

        # ...and the balance takes up whatever that leaves, so a round with
        # one lands exactly on the size that was asked for.
        opt.clear_formula("Water")
        opt.set_formulation_total(100)
        opt.set_formula("Water", "= rest")
        opt.set_pending_batch([{"Water": 70.0, "Flour": 20.0, "Sugar": 10.0}])
        opt.scale_round(150)
        balanced = opt._batch_rows(opt.pending_batch)[0]['recipe']
        assert balanced["Flour"] == pytest.approx(30.0)
        assert balanced["Sugar"] == pytest.approx(15.0)
        assert sum(balanced.values()) == pytest.approx(150.0)

    def test_the_grid_refuses_a_deletion_in_the_words_written_for_it(
            self, tmp_path, monkeypatch):
        """formula_reads_this_row names the row at fault and says what to
        change. It was reachable only by deleting a process setting: from
        the grid, the per-row parse got there first and said "there is no
        ingredient called Salt" — true of the name list it was asked
        against, and flatly untrue of the grid on screen. Asked now for
        every deleted row, before any row is read, whatever its category.
        """
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_formula("Water", "= Sugar × 2")
        errors, _ = opt.apply_ingredient_grid(
            _drop(opt.ingredient_grid_frame(), 3))
        assert errors == [(None, wording.formula_reads_this_row("Sugar",
                                                                "Water"))]
        assert "Sugar" in [v['name'] for v in opt.variables]

    def test_a_rule_may_not_name_a_process_setting(self, tmp_path,
                                                   monkeypatch):
        """Grams of salt calculated from minutes of cooking is arithmetic
        across two units that cannot be mixed. The app refused a setting a
        rule of its own and then allowed the reverse, and said "Salt is
        calculated as Cook time × 0.1: between 0.20 and 1.00 g" about it."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_process_parameter("Oven", 100, 200)
        with pytest.raises(FormulaError) as caught:
            opt.set_formula("Water", "= 0.1 × Oven")
        assert str(caught.value) == wording.rule_ingredients_only("Oven")
        assert opt.has_formula(opt._var_by_name("Water")) is False
        errors, _ = opt.apply_ingredient_grid(_edit(
            opt.ingredient_grid_frame(), 1,
            **{wording.FORMULA_LABEL: "= 0.1 × Oven"}))
        assert errors == [(1, wording.rule_ingredients_only("Oven"))]
        # ...and so the setting is free to go, with no rule reading it.
        opt.remove_process_parameter("Oven")
        assert "Oven" not in [v['name'] for v in opt.variables]
        # Cleared, it goes the way it always did.
        opt.clear_formula("Water")
        opt.remove_process_parameter("Oven")
        assert "Oven" not in [v['name'] for v in opt.variables]

    def test_the_amounts_recorded_are_what_the_tables_show(self, tmp_path,
                                                           monkeypatch):
        """A formulation was made at the amounts it was made at. A formula
        written afterwards works out what the NEXT one will be; it does not
        rewrite what the bench weighed out last week."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.tell({"Water": 70.0, "Flour": 20.0, "Sugar": 10.0},
                 {"Taste": 6.0})
        opt.set_formulation_total(120)
        opt.set_formula("Water", "= rest")
        column = opt._amount_column("Water")
        shown = opt.history_frame(include_amounts=True)
        assert shown[column].iloc[0] == pytest.approx(70.0)
        exported = opt.history_export_frame()
        assert exported[column].iloc[0] == pytest.approx(70.0)
        assert exported[opt.total_column()].iloc[0] == pytest.approx(100.0)

    def test_a_new_row_is_stored_the_way_a_saved_copy_restores_it(
            self, tmp_path, monkeypatch):
        """One shape on disk: a row added today and the same row restored
        from a copy hold the same keys, or every comparison between them
        reads as a change nobody made."""
        opt = self._opt(tmp_path, monkeypatch)
        state = opt.export_json()
        other = FoodOptimizer("copy_of_rows")
        other.import_json(state)
        assert other.variables == opt.variables


class TestTheFormulaColumn:
    """Task 3 of the rules wave (2026-09-16): the Formula column on the
    ingredients grid.

    One column for one idea: `= rest` is typed in the Formula cell, so there
    is no Balance column beside it. A row that carries one is calculated —
    its range cells read `calculated`, the numbers typed there are ignored,
    and the consequence is said under the grid in the amounts the other rows
    leave it.
    """

    def _opt(self, tmp_path, monkeypatch, name="formula_grid"):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer(name)
        opt.set_amount_unit("g")
        opt.add_ingredient("Water", 20, 60)
        opt.add_ingredient("Pea protein", 30, 50)
        opt.add_ingredient("Salt", 8, 10)
        opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
        opt.set_formulation_total(100)
        return opt

    @staticmethod
    def _formula(frame, row, text):
        return _edit(frame, row, **{wording.FORMULA_LABEL: text})

    # ---- the cell is saved on the row ------------------------------- #

    def test_a_formula_typed_into_the_grid_is_saved_on_the_row(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        errors, messages = opt.apply_ingredient_grid(self._formula(
            opt.ingredient_grid_frame(), 1,
            "= batch size − Pea protein − Salt"))
        assert errors == []
        var = opt._var_by_name("Water")
        assert var['formula'] == "= batch size − Pea protein − Salt"
        assert var['balance'] is False
        assert opt.has_formula(var) is True
        # ...and it is calculated from the rows it names, not searched.
        assert [v['name'] for v in opt.varying_variables()] == ["Pea protein",
                                                               "Salt"]
        assert opt.fill_formulas({"Pea protein": 40.0, "Salt": 9.0}) == {
            "Pea protein": 40.0, "Salt": 9.0, "Water": 51.0}

    def test_a_worked_out_row_says_the_word_in_one_cell_not_two(
            self, tmp_path, monkeypatch):
        """One word doing one job twice on two cells side by side. Lowest
        is left blank and Highest carries the mark, so the pair reads as
        one fact about the row."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.apply_ingredient_grid(self._formula(
            opt.ingredient_grid_frame(), 1, "= rest"))
        frame = opt.ingredient_grid_frame()
        assert list(frame.columns)[-1] == wording.FORMULA_LABEL
        row = frame.loc[1]
        assert row[wording.FORMULA_LABEL] == "= rest"
        assert row[wording.LOWEST_LABEL] == ""
        assert row[wording.HIGHEST_LABEL] == ""
        # Every other row carries the two-decimal text a number column used
        # to format for it.
        assert frame.loc[2][wording.LOWEST_LABEL] == "30.00"
        assert frame.loc[2][wording.HIGHEST_LABEL] == "50.00"
        assert frame.loc[2][wording.FORMULA_LABEL] == ""
        # A number typed into a worked-out row's range cell is ignored, and
        # the cell is rewritten to the word on the next render.
        errors, _ = opt.apply_ingredient_grid(_edit(
            frame, 1, **{wording.LOWEST_LABEL: "5.00",
                         wording.HIGHEST_LABEL: "9.00"}))
        assert errors == []
        assert opt._var_by_name("Water")['bounds'] == (20.0, 60.0)
        back = opt.ingredient_grid_frame().loc[1]
        assert (back[wording.LOWEST_LABEL],
                back[wording.HIGHEST_LABEL]) == ("", "")

    def test_clearing_a_formula_gives_the_row_its_range_back(self, tmp_path,
                                                             monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.apply_ingredient_grid(self._formula(
            opt.ingredient_grid_frame(), 1, "= rest"))
        errors, _ = opt.apply_ingredient_grid(self._formula(
            opt.ingredient_grid_frame(), 1, ""))
        assert errors == []
        var = opt._var_by_name("Water")
        assert opt.has_formula(var) is False
        assert var['bounds'] == (20.0, 60.0)
        frame = opt.ingredient_grid_frame()
        assert frame.loc[1][wording.LOWEST_LABEL] == "20.00"
        assert frame.loc[1][wording.HIGHEST_LABEL] == "60.00"

    # ---- the refusals ----------------------------------------------- #

    def test_two_rest_rows_are_refused_over_the_grid(self, tmp_path,
                                                     monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        frame = self._formula(self._formula(
            opt.ingredient_grid_frame(), 1, "= rest"), 3, "= rest")
        errors, messages = opt.apply_ingredient_grid(frame)
        assert errors == [(None, "Only one row can be = rest: "
                          "Water and Salt both are.")]
        assert messages == []
        # Nothing was written: a refusal leaves the project as it was.
        assert opt.has_formula(opt._var_by_name("Water")) is False
        assert opt.has_formula(opt._var_by_name("Salt")) is False

    def test_a_formula_on_a_process_setting_is_refused(self, tmp_path,
                                                       monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_process_parameter("Oven", 150, 200, unit="°C")
        frame = self._formula(opt.ingredient_grid_frame(), 4, "= 0.5 × Salt")
        errors, _ = opt.apply_ingredient_grid(frame)
        assert errors == [(4, wording.only_an_ingredient_has(
            wording.FORMULA_LABEL))]
        assert opt.has_formula(opt._var_by_name("Oven")) is False

    def test_a_loop_typed_across_two_rows_is_refused_once(self, tmp_path,
                                                          monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        frame = self._formula(self._formula(
            opt.ingredient_grid_frame(), 1, "= Salt × 2"), 3, "= Water ÷ 2")
        errors, _ = opt.apply_ingredient_grid(frame)
        assert errors == [(None, wording.formula_loop("Water → Salt → Water"))]
        assert opt.has_formula(opt._var_by_name("Water")) is False

    # ---- the consequence, in numbers -------------------------------- #

    def test_the_worked_out_caption_gives_the_range_in_numbers(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.apply_ingredient_grid(self._formula(
            opt.ingredient_grid_frame(), 1,
            "= batch size − Pea protein − Salt"))
        # The cell is quoted with its '=', so the line can be matched
        # against what was typed; and the row's own dormant Lowest and
        # Highest are named where the rule takes it past them.
        assert opt.worked_out_captions() == [
            "Water is calculated from = batch size − Pea protein − Salt: "
            "between 40.00 and 62.00 g in a 100 g formulation. "
            "Its own Lowest and Highest (20.00 to 60.00 g) do not apply "
            "while the calculation does."]
        # The balance says the same thing in the words it was written in.
        opt.apply_ingredient_grid(self._formula(
            opt.ingredient_grid_frame(), 1, "= rest"))
        # The head names the number it fills to AND where that number is
        # set, so the line is read the same on a round made to another size.
        assert opt.worked_out_captions() == [
            "Water is calculated to bring the total to the 100 g default "
            "batch size: between 40.00 and 62.00 g. "
            "Its own Lowest and Highest (20.00 to 60.00 g) do not apply "
            "while the calculation does."]

    def test_the_caption_never_offers_an_amount_below_nothing(
            self, tmp_path, monkeypatch):
        """Every rule shows its consequence in numbers, and a negative gram
        is not one of them: the app holds a worked-out row at or above 0,
        so the line says the range the app will allow."""
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("caption_floor")
        opt.set_amount_unit("g")
        opt.add_ingredient("Water", 0, 100)
        opt.add_ingredient("Flour", 0, 50)
        opt.add_ingredient("Sugar", 0, 50)
        opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
        opt.set_formulation_total(50)
        errors, _ = opt.apply_ingredient_grid(
            self._formula(opt.ingredient_grid_frame(), 1, "= 20 − Flour"))
        assert errors == []
        # The arithmetic reaches −30.00 g; the app never will.
        assert opt.worked_out_captions() == [
            "Water is calculated from = 20 − Flour: between 0.00 and 20.00 g "
            "in a 50 g formulation."]

    def test_no_allowed_amounts_caution_lands_on_a_worked_out_row(
            self, tmp_path, monkeypatch):
        """A worked-out row's Lowest and Highest are dormant — the rule
        decides the amount, and the grid shows the word where those two
        numbers were — so a caution telling the bench to widen them names
        two numbers that are not on screen and are not enforced."""
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("no_caution", robust=False)
        opt.set_amount_unit("g")
        opt.add_ingredient("Water", 30, 70)
        opt.add_ingredient("Flour", 10, 30)
        opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
        opt.set_formulation_total(100)
        opt.set_formula("Water", "= rest")
        rows = [{"Water": 80.0, "Flour": 20.0}]
        # Water is at 80 g over a dormant 30-70 g: no caution, because
        # those two numbers are not what holds it.
        assert opt.scaled_caution(rows, 100.0, sized=True) == ""
        # ...and the caution still lands on a row that HAS its own amounts.
        opt.clear_formula("Water")
        assert "Water" in opt.scaled_caution(rows, 100.0, sized=True)

    def test_a_rule_that_comes_to_one_number_says_that_number(
            self, tmp_path, monkeypatch):
        """"between 1.50 and 1.50 g" asked the reader to read a range where
        nothing can vary."""
        opt = self._opt(tmp_path, monkeypatch)
        errors, _ = opt.apply_ingredient_grid(self._formula(
            opt.ingredient_grid_frame(), 3, "= 1.5 % of batch size"))
        assert errors == []
        assert opt.worked_out_captions() == [
            "Salt is calculated from = 1.5 % of batch size: 1.50 g in a "
            "100 g formulation. Its own Lowest and Highest (8.00 to "
            "10.00 g) do not apply while the calculation does."]

    def test_the_caption_is_absent_without_a_formula(self, tmp_path,
                                                     monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        assert opt.worked_out_captions() == []

    # ---- what a save costs the open round --------------------------- #

    def test_a_formula_discards_the_open_round_and_a_vendor_does_not(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_to_pending_batch({"Water": 55.0, "Pea protein": 35.0,
                                  "Salt": 10.0}, note="mine")
        assert opt.pending_batch_no is not None
        frame = _edit(opt.ingredient_grid_frame(), 3,
                      **{wording.SKU_LABEL: "SA-1"})
        assert opt.ingredient_grid_retires_round(frame) is None
        errors, _ = opt.apply_ingredient_grid(frame)
        assert errors == []
        assert opt.pending_batch_no is not None
        # A formula is a change to the question the model is being asked.
        frame = self._formula(opt.ingredient_grid_frame(), 1, "= rest")
        assert opt.ingredient_grid_retires_round(frame) == 1
        errors, _ = opt.apply_ingredient_grid(frame)
        assert errors == []
        assert opt.pending_batch_no is None

    def test_the_formula_line_is_said_once_for_two_new_formula_rows(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.tell({"Water": 55.0, "Pea protein": 35.0, "Salt": 10.0},
                 {"Taste": 6.0})
        frame = self._formula(self._formula(
            opt.ingredient_grid_frame(), 1, "= rest"), 3,
            "= 0.1 × Pea protein")
        errors, messages = opt.apply_ingredient_grid(frame)
        assert errors == []
        said = [text for _, text in messages
                if text.startswith("Formulations already made keep")]
        assert said == [
            "Formulations already made keep their amounts. Water and Salt "
            "are calculated from their calculations from the next round on."]

    # ---- a formula is data a file can bring in ---------------------- #

    def test_a_csv_formula_column_round_trips(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("from_file")
        opt.set_amount_unit("g")
        opt.load_ingredients_from_csv(pd.DataFrame({
            "Name": ["Flour", "Sugar", "Water"],
            "Lowest": [10, 5, None], "Highest": [50, 30, None],
            "Unit": ["g", "g", "g"],
            "Formula": [None, "= 0.2 × Flour", "= rest"],
            "Fat per 100 g": [1.0, 0.0, 0.0]}))
        assert opt._var_by_name("Sugar")['formula'] == "= 0.2 × Flour"
        assert opt._var_by_name("Water")['balance'] is True
        assert opt._var_by_name("Flour")['formula'] == ""
        # The column is a formula, not a property of every ingredient.
        assert set(opt.ingredient_properties["Flour"]) == {"Fat per 100 g"}
        # ...and it comes back out of the grid the way it went in.
        frame = opt.ingredient_grid_frame()
        assert list(frame[wording.FORMULA_LABEL]) == ["", "= 0.2 × Flour",
                                                      "= rest"]
        opt.set_formulation_total(100)
        assert opt.fill_formulas({"Flour": 40.0}) == {
            "Flour": 40.0, "Sugar": 8.0, "Water": 52.0}

    def test_the_sample_water_is_rest_and_every_sample_round_adds_up(
            self, tmp_path, monkeypatch):
        """Show, don't teach: the sample project the reader opens first has
        a worked-out row in it, and every round it makes lands exactly on
        the batch size because of it."""
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("sample_rest", robust=False)
        opt.set_amount_unit("g")
        opt.load_ingredients_from_csv(pd.read_csv(_SAMPLE_CSV))
        opt.add_objective("Juiciness", 1.0, goal="target", target=7,
                          min_val=0, max_val=10, unit="/10")
        opt.set_formulation_total(100.0)
        assert opt._var_by_name("Remaining water")['balance'] is True
        assert opt._formula_text(opt._var_by_name("Remaining water")) == "= rest"
        # Textured pea protein, Dry blend, Wheat gluten and the two oils
        # the Fat phase is weighed out as; Seasoning blend is fixed at
        # 2.20 g and Water is calculated.
        assert len(opt.varying_variables()) == 6
        for row in opt.ask(n_suggestions=3):
            assert sum(row.values()) == pytest.approx(100.0, abs=1e-6)
            assert row["Remaining water"] >= -1e-9


# ------------------------------------------------------------------ #
#  0.5.0 wave 2, task 4 (2026-09-16): a worked-out row on the workbook the
#  bench prints, and the round table on tab 2 stays read only.
# ------------------------------------------------------------------ #
class TestFormulasOnTheSheets:
    """A formula row prints the amount it computed to, marked so a bench
    reading the printed page knows it was not chosen, only calculated."""

    def _opt(self, tmp_path, monkeypatch, name="sheet_formulas"):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer(name)
        for field in ("vendor", "sku", "lot", "actual"):
            opt.set_records(field, True)
        opt.set_amount_unit("g")
        opt.add_ingredient("Pea protein", 30, 50)
        opt.add_ingredient("Salt", 8, 10)
        opt.add_ingredient("Water", 20, 60)
        opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
        opt.set_formulation_total(100)
        opt.set_formula("Water", "= batch size − Pea protein − Salt")
        opt.set_pending_batch([
            {"Pea protein": 40.0, "Salt": 9.0, "Water": 51.0},
        ], batch_no=1)
        return opt

    def _no_formula_opt(self, tmp_path, monkeypatch, name="sheet_plain"):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer(name)
        for field in ("vendor", "sku", "lot", "actual"):
            opt.set_records(field, True)
        opt.set_amount_unit("g")
        opt.add_ingredient("Pea protein", 30, 50)
        opt.add_ingredient("Salt", 8, 10)
        opt.add_ingredient("Water", 20, 60)
        opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
        opt.set_pending_batch([
            {"Pea protein": 40.0, "Salt": 9.0, "Water": 51.0},
        ], batch_no=1)
        return opt

    def _balance_opt(self, tmp_path, monkeypatch, name="sheet_balance"):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer(name)
        for field in ("vendor", "sku", "lot", "actual"):
            opt.set_records(field, True)
        opt.set_amount_unit("g")
        opt.add_ingredient("Pea protein", 30, 50)
        opt.add_ingredient("Salt", 8, 10)
        opt.add_ingredient("Water", 20, 60)
        opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
        opt.set_formulation_total(100)
        opt.set_formula("Water", "= rest")
        return opt

    # ---- the mark on the amount ------------------------------------- #

    def test_the_summary_sheet_marks_a_worked_out_row(self, tmp_path,
                                                      monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        book = _book(opt.workbook_bytes(opt.pending_batch, 100.0))
        summary = book[wording.batch_sheet_name(opt.pending_batch_no)]
        names = {row[0] for row in _rows(summary) if isinstance(row[0], str)}
        assert f'{wording.worked_out_label("Water")} (g)' in names
        assert "Water (g)" not in names
        # A row with no formula wears no mark.
        assert "Salt (g)" in names

    def test_each_formulation_page_marks_it_too(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        book = _book(opt.workbook_bytes(opt.pending_batch, 100.0))
        page = book["Formulation 1"]
        names = {row[1] for row in _rows(page) if isinstance(row[1], str)}
        assert wording.worked_out_label("Water") in names
        assert "Salt" in names

    # ---- the sparse note ---------------------------------------------- #

    def test_the_note_only_appears_when_a_round_has_a_formula(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        book = _book(opt.workbook_bytes(opt.pending_batch, 100.0))
        summary = book[wording.batch_sheet_name(opt.pending_batch_no)]
        page = book["Formulation 1"]
        # The note names each worked-out row's own rule: the rule is on no
        # sheet of THIS workbook, so "filled in from its rule" sent a bench
        # looking for something that is not in the file.
        note = opt.worked_out_note()
        assert note.startswith("Water is calculated: ")
        assert note.endswith("Weigh the amount printed.")
        assert any(row[0] == note for row in _rows(summary))
        assert any(row[1] == note for row in _rows(page))

        plain = self._no_formula_opt(tmp_path, monkeypatch)
        plain_book = _book(plain.workbook_bytes(plain.pending_batch, 100.0))
        plain_summary = plain_book[wording.batch_sheet_name(
            plain.pending_batch_no)]
        plain_page = plain_book["Formulation 1"]
        assert plain.worked_out_note() == ""
        assert not any(row[0] and str(row[0]).endswith(
            "Weigh the amount printed.") for row in _rows(plain_summary))
        assert not any(row[1] and str(row[1]).endswith(
            "Weigh the amount printed.") for row in _rows(plain_page))

    # ---- the Set-up sheet's Formula column ----------------------------- #

    def test_the_set_up_sheet_prints_the_formula(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        book = _book(opt.all_formulations_workbook())
        sheet = book[wording.SET_UP_SHEET]
        header = [c.value for c in sheet[2]]
        assert header[6] == wording.FORMULA_LABEL
        at = {sheet.cell(row=r, column=1).value: r
             for r in range(1, sheet.max_row + 1)}
        assert sheet.cell(row=at["Water"], column=7).value == \
            "= batch size − Pea protein − Salt"
        assert sheet.cell(row=at["Salt"], column=7).value is None

    def test_the_sheet_has_no_formula_column_without_one(self, tmp_path,
                                                         monkeypatch):
        opt = self._no_formula_opt(tmp_path, monkeypatch)
        book = _book(opt.all_formulations_workbook())
        header = [c.value for c in book[wording.SET_UP_SHEET][2]]
        assert wording.FORMULA_LABEL not in header

    def test_the_balance_row_prints_its_long_form(self, tmp_path,
                                                  monkeypatch):
        opt = self._balance_opt(tmp_path, monkeypatch)
        book = _book(opt.all_formulations_workbook())
        sheet = book[wording.SET_UP_SHEET]
        at = {sheet.cell(row=r, column=1).value: r
             for r in range(1, sheet.max_row + 1)}
        assert sheet.cell(row=at["Water"], column=7).value == (
            "Fill to total (batch size − every other ingredient)")

    # ---- the Actual cell is unaffected --------------------------------- #

    def test_a_worked_out_row_still_has_an_actual_cell(self, tmp_path,
                                                       monkeypatch):
        """What was weighed is what was weighed: a worked-out row's Actual
        cell is unlocked exactly like any other row's."""
        opt = self._opt(tmp_path, monkeypatch)
        book = _book(opt.workbook_bytes(opt.pending_batch, 100.0))
        page = book["Formulation 1"]
        at = {page.cell(row=r, column=2).value: r
             for r in range(1, page.max_row + 1)}
        row = at[wording.worked_out_label("Water")]
        actual = page.cell(row=row, column=4)
        assert actual.value is None
        assert not actual.protection.locked
        # The amount beside it, what it computed to, is still locked.
        amount = page.cell(row=row, column=3)
        assert amount.value == 51.0
        assert amount.protection.locked


class TestFormulasOnTheSheetsFixes:
    """Fix round 1 on Task 4: a worked-out row's printed name carries
    `wording.worked_out_label`, and the upload's own label lookups —
    `_lots_from_summary` and `_actual_from_sheets` — had never been taught
    the mark, so a filled-in Actual or Lot for that row matched nothing and
    vanished with no error."""

    def _opt(self, tmp_path, monkeypatch, name="sheet_upload_formulas"):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer(name)
        for field in ("vendor", "sku", "lot", "actual"):
            opt.set_records(field, True)
        opt.set_amount_unit("g")
        opt.add_ingredient("Pea protein", 30, 50)
        opt.add_ingredient("Salt", 8, 10)
        opt.add_ingredient("Water", 20, 60)
        opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
        opt.set_formulation_total(100)
        opt.set_formula("Water", "= rest")
        opt.set_pending_batch([
            {"Pea protein": 40.0, "Salt": 9.2, "Water": 50.8},
        ], batch_no=1)
        return opt

    @staticmethod
    def _lot_column(sheet, header_row=3):
        for c in range(1, sheet.max_column + 1):
            if sheet.cell(row=header_row, column=c).value == \
                    wording.LOT_COLUMN:
                return c
        raise AssertionError("no Lot column on the sheet")

    def _filled(self, opt, water_actual=53.5, water_lot="LOT-WATER-9",
                salt_actual=9.2, salt_lot="LOT-SALT-1"):
        """The workbook back off the bench: both the worked-out Water row
        and the plain Salt row weighed and lotted."""
        book = openpyxl.load_workbook(io.BytesIO(
            opt.workbook_bytes(opt.pending_batch, 100.0)))
        summary = book[wording.batch_sheet_name(opt.pending_batch_no)]
        lot_col = self._lot_column(summary)
        at = {summary.cell(row=r, column=1).value: r
             for r in range(1, summary.max_row + 1)}
        summary.cell(row=at[f'{wording.worked_out_label("Water")} (g)'],
                     column=lot_col, value=water_lot)
        summary.cell(row=at["Salt (g)"], column=lot_col, value=salt_lot)
        # A row with nothing measured is a row nothing came back for at
        # all — the one thing this fixture needs beside the amounts.
        summary.cell(row=at["Taste · Prefer higher values"], column=2, value=6.0)
        page = book["Formulation 1"]
        rows = {page.cell(row=r, column=2).value: r
               for r in range(1, page.max_row + 1)}
        page.cell(row=rows[wording.worked_out_label("Water")], column=4,
                  value=water_actual)
        page.cell(row=rows["Salt"], column=4, value=salt_actual)
        out = io.BytesIO()
        book.save(out)
        out.seek(0)
        return out

    def test_a_worked_out_rows_actual_comes_back_off_the_upload(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        upload = opt.results_from_workbook(self._filled(opt))
        assert upload.actual == {1: {"Water": 53.5, "Salt": 9.2}}

    def test_a_worked_out_rows_lot_comes_back_off_the_upload(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        upload = opt.results_from_workbook(self._filled(opt))
        assert upload.lots == {"Water": "LOT-WATER-9", "Salt": "LOT-SALT-1"}

    def test_a_plain_rows_actual_and_lot_are_unaffected(self, tmp_path,
                                                        monkeypatch):
        """The fix adds an alternative label; it must not stop the ordinary
        one from matching."""
        opt = self._opt(tmp_path, monkeypatch)
        upload = opt.results_from_workbook(
            self._filled(opt, water_actual=None, water_lot=None))
        assert upload.actual == {1: {"Salt": 9.2}}
        assert upload.lots == {"Salt": "LOT-SALT-1"}

    def test_the_worked_out_actual_is_recorded_like_any_other(
            self, tmp_path, monkeypatch):
        """The `Amounts as weighed` path: `amounts_as_weighed` overlays
        `upload.actual` straight onto the recipe by name, so once the
        label lookup finds the row the rest of the path needs nothing
        else."""
        opt = self._opt(tmp_path, monkeypatch)
        upload = opt.results_from_workbook(self._filled(opt))
        by_number = {r['formulation']: r['recipe'] for r in opt.pending_batch}
        recorded = opt.amounts_as_weighed(by_number[1], upload.actual.get(1))
        assert recorded == {"Pea protein": 40.0, "Salt": 9.2, "Water": 53.5}
        opt.tell(recorded, {"Taste": 6.0}, formulation_no=1,
                 batch_no=opt.pending_batch_no,
                 note=wording.amounts_as_weighed_note(""))
        assert opt.recipe_history[0] == {"Pea protein": 40.0, "Salt": 9.2,
                                         "Water": 53.5}
        assert opt.notes_history[0] == wording.AMOUNTS_AS_WEIGHED


class TestTheFormulaColumnFixes:
    """Fix round 1 on the Formula column. Four of these are the same
    mistake wearing different clothes: a formula cell is TEXT, and text
    that nobody typed into is not text that changed."""

    def _opt(self, tmp_path, monkeypatch, name="formula_fixes"):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer(name)
        opt.set_amount_unit("g")
        opt.add_ingredient("Water", 20, 60)
        opt.add_ingredient("Pea protein", 10, 25)
        opt.add_ingredient("Salt", 0, 3)
        opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
        opt.set_formulation_total(50)
        return opt

    @staticmethod
    def _formula(frame, row, text):
        return _edit(frame, row, **{wording.FORMULA_LABEL: text})

    def test_a_rename_leaves_an_untouched_formula_alone_on_the_grid(
            self, tmp_path, monkeypatch):
        """The row the formula names is renamed; the formula cell is never
        touched. rename_variable already rewrites the name inside it — the
        grid was reading the old spelling against the new names and calling
        the reader's own row unknown."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_formula("Water", "= batch size − Pea protein − Salt")
        opt.add_to_pending_batch({"Pea protein": 20.0, "Salt": 2.0})
        frame = _edit(opt.ingredient_grid_frame(), 3,
                      **{wording.NAME_LABEL: "Sea salt"})
        # A rename is a rename, whoever names the row: the open round stays.
        assert opt.ingredient_grid_retires_round(frame) is None
        errors, _ = opt.apply_ingredient_grid(frame)
        assert errors == []
        assert opt._var_by_name("Water")['formula'] == \
            "= batch size − Pea protein − Sea salt"
        assert opt.pending_batch_no is not None
        assert opt.fill_formulas({"Pea protein": 20.0, "Sea salt": 2.0}) == {
            "Pea protein": 20.0, "Sea salt": 2.0, "Water": 28.0}

    def test_two_renames_in_one_save_are_read_against_the_old_spelling(
            self, tmp_path, monkeypatch):
        """A chain of renames in one Save — Salt→Butter, Pea protein→Salt —
        rewrote the formula one rename at a time, so the second pass read
        the Butter the first pass had only just written and the cell ended
        up naming one row twice. Every name is looked up once now, against
        the spelling the reader typed the cell in."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_formula("Water", "= Pea protein + Salt")
        frame = opt.ingredient_grid_frame()
        frame = _edit(frame, 3, **{wording.NAME_LABEL: "Butter"})
        frame = _edit(frame, 2, **{wording.NAME_LABEL: "Salt"})
        errors, _ = opt.apply_ingredient_grid(frame)
        assert errors == []
        assert opt._var_by_name("Water")['formula'] == "= Salt + Butter"
        assert opt._linear_form("Water").terms == {"Salt": 1.0,
                                                   "Butter": 1.0}

    def test_three_renames_in_one_chain_land_where_the_reader_typed_them(
            self, tmp_path, monkeypatch):
        """The same one pass over a longer chain: Salt→Sea salt,
        Pea protein→Salt, Water→Pea protein, with a formula naming two of
        the three."""
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("three_chain")
        opt.set_amount_unit("g")
        opt.add_ingredient("Water", 20, 60)
        opt.add_ingredient("Pea protein", 10, 25)
        opt.add_ingredient("Salt", 0, 3)
        opt.add_ingredient("Oil", 0, 10)
        opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
        opt.set_formula("Oil", "= Salt + Pea protein")
        frame = opt.ingredient_grid_frame()
        frame = _edit(frame, 3, **{wording.NAME_LABEL: "Sea salt"})
        frame = _edit(frame, 2, **{wording.NAME_LABEL: "Salt"})
        frame = _edit(frame, 1, **{wording.NAME_LABEL: "Pea protein"})
        errors, _ = opt.apply_ingredient_grid(frame)
        assert errors == []
        assert opt._var_by_name("Oil")['formula'] == "= Sea salt + Salt"

    def test_a_formula_that_puts_the_default_out_of_reach_says_so_at_the_save(
            self, tmp_path, monkeypatch):
        """A formula folds a whole row's coefficients into the sum, so it
        moves what the ingredients can add up to more sharply than a range
        edit does. Neither door that writes one re-asked the total, so the
        save came back clean and tab 2's one lit button refused afterwards,
        naming a fix that was not the problem."""
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("stale_total")
        opt.set_amount_unit("g")
        for name in ("A", "B"):
            opt.add_ingredient(name, 5, 20)
        opt.add_ingredient("Filler", 0, 100)
        opt.set_formulation_total(75)
        opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
        errors, messages = opt.apply_ingredient_grid(self._formula(
            opt.ingredient_grid_frame(), 3, "= batch size - 2 * A - 2 * B"))
        assert errors == []
        assert opt.formulation_total is None
        assert any(wording.formulation_total_gone_unreachable(
            opt.batch_total_text(75.0)) == text for _, text in messages)

    def test_rubbing_a_formula_out_re_asks_the_default_batch_size_too(
            self, tmp_path, monkeypatch):
        """The other half of the one door. Clearing = rest takes the row
        that absorbed any size back out of the sum, so a default the rest
        row made reachable stops being reachable."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_formula("Water", "= rest")
        opt.set_formulation_total(95)
        assert opt.formulation_total == 95
        errors, messages = opt.apply_ingredient_grid(
            self._formula(opt.ingredient_grid_frame(), 1, ""))
        assert errors == []
        assert opt.formulation_total is None
        assert any(wording.formulation_total_gone_unreachable(
            opt.batch_total_text(95.0)) == text for _, text in messages)

    def test_set_formula_hands_back_what_the_default_batch_size_cost(
            self, tmp_path, monkeypatch):
        """The setter is the door, so the setter owes the sentence."""
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("setter_total")
        opt.set_amount_unit("g")
        for name in ("A", "B"):
            opt.add_ingredient(name, 5, 20)
        opt.add_ingredient("Filler", 0, 100)
        opt.set_formulation_total(75)
        opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
        removed = opt.set_formula("Filler", "= batch size - 2 * A - 2 * B")
        assert [qc.get('reason') for qc in removed] == ['unreachable']
        assert opt.formulation_total is None

    def test_clearing_a_formula_needs_the_amounts_too(self, tmp_path,
                                                      monkeypatch):
        """Giving a row a formula rebuilds the history and taking one away
        rebuilds it just as hard. The gate asked only about the rows that
        HAVE one, so the last formula could be rubbed out over a project
        whose recorded amounts have gone — and the history came back
        narrower than the rows it is read against."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_formula("Water", "= rest")
        opt.tell({"Pea protein": 20.0, "Salt": 2.0, "Water": 28.0},
                 {"Taste": 6.0})
        opt.recipe_history = []
        errors, _ = opt.apply_ingredient_grid(
            self._formula(opt.ingredient_grid_frame(), 1, ""))
        assert errors == [(None, wording.AMOUNTS_MISSING_DELETE_ERROR)]
        assert opt.has_formula(opt._var_by_name("Water")) is True
        assert len(opt.X_history[0]) == len(opt.varying_variables())

    def test_adding_a_plain_row_without_the_amounts_is_refused_not_raised(
            self, tmp_path, monkeypatch):
        """A row that ARRIVES is one more column in the search vector, so
        the history has to be rebuilt to hold it — and a project whose
        recorded amounts have gone cannot be rebuilt. The plan passed the
        row and add_ingredient then raised out of the middle of the save,
        past the grid's promise that nothing is written until every row
        passes, and reached the browser as a traceback.

        Both calls, deliberately: the plan is where the refusal belongs and
        apply_ingredient_grid is the call that reached the bug."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_formula("Water", "= rest")
        opt.tell({"Pea protein": 20.0, "Salt": 2.0, "Water": 28.0},
                 {"Taste": 6.0})
        opt.recipe_history = []
        frame = _add(opt.ingredient_grid_frame(), **_ing_row("Oil", high=5.0))
        assert opt._plan_ingredient_grid(frame)[0] == [
            (4, wording.CANNOT_ADD_WITHOUT_AMOUNTS)]
        errors, messages = opt.apply_ingredient_grid(frame)
        assert errors == [(4, wording.CANNOT_ADD_WITHOUT_AMOUNTS)]
        assert messages == []
        assert [v['name'] for v in opt.variables] == [
            "Water", "Pea protein", "Salt"]

    def test_adding_a_plain_row_is_free_while_the_amounts_are_all_there(
            self, tmp_path, monkeypatch):
        """The gate is about the amounts, not about the formula: a project
        that kept them adds a row with a rest row standing."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_formula("Water", "= rest")
        opt.tell({"Pea protein": 20.0, "Salt": 2.0, "Water": 28.0},
                 {"Taste": 6.0})
        errors, _ = opt.apply_ingredient_grid(
            _add(opt.ingredient_grid_frame(), **_ing_row("Oil", high=5.0)))
        assert errors == []
        assert opt._var_by_name("Oil")['bounds'] == (0.0, 5.0)

    def test_a_worked_out_rows_range_cells_are_ignored_whatever_they_hold(
            self, tmp_path, monkeypatch):
        """Ignored means ignored: the row's amount is its formula, so a
        word left in the cell beside it is not a number the reader owes."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_formula("Water", "= rest")
        errors, _ = opt.apply_ingredient_grid(_edit(
            opt.ingredient_grid_frame(), 1,
            **{wording.LOWEST_LABEL: "about a cup",
               wording.HIGHEST_LABEL: wording.WORKED_OUT}))
        assert errors == []
        assert opt._var_by_name("Water")['bounds'] == (20.0, 60.0)

    def test_a_file_rule_naming_a_process_setting_is_refused(
            self, tmp_path, monkeypatch):
        """The settings are put back before the rules are read, so the file
        door asks the same question the grid does — and gives the same
        answer: a rule uses ingredients and the batch size, not a setting
        measured in something else."""
        opt = self._opt(tmp_path, monkeypatch, name="file_setting")
        opt.add_process_parameter("Oven", 150, 200, unit="°C")
        with pytest.raises(ValueError) as refused:
            opt.load_ingredients_from_csv(pd.DataFrame({
                "Name": ["Flour", "Glaze"], "Lowest": [10, 0],
                "Highest": [50, 0], "Unit": ["g", "g"],
                "Rule": [None, "= 0.1 × Oven"]}))
        assert str(refused.value) == wording.rule_ingredients_only("Oven")

    @pytest.mark.parametrize("cell", ["=rest", "= REST", "=  rest",
                                      "= rest "])
    def test_a_file_spells_the_rest_row_however_it_likes(
            self, cell, tmp_path, monkeypatch):
        """The loader compared the cell with the literal '= rest', where
        every other reader asks the parser. A file written '=rest' loaded
        with the flag unset, so the project behaved as the rest row to the
        arithmetic and as having none to the default batch size box — and
        could not be given a size at all."""
        opt = self._opt(tmp_path, monkeypatch, name="file_rest")
        opt.load_ingredients_from_csv(pd.DataFrame({
            "Name": ["Flour", "Sugar", "Water"], "Lowest": [10, 5, None],
            "Highest": [60, 20, None], "Unit": ["g", "g", "g"],
            "Formula": [None, None, cell]}))
        assert opt._var_by_name("Water")['balance'] is True
        assert opt._balance_row()['name'] == "Water"
        opt.set_formulation_total(100)
        assert opt.formulation_total == 100
        assert opt.fill_formulas({"Flour": 20.0, "Sugar": 10.0})["Water"] == \
            pytest.approx(70.0)

    def test_two_rest_rows_spelled_differently_are_still_two(
            self, tmp_path, monkeypatch):
        """The uniqueness check counted the flag the compare had just set,
        so a file could bring in two rest rows by spelling one of them
        '=rest'."""
        opt = self._opt(tmp_path, monkeypatch, name="file_two_rest")
        with pytest.raises(ValueError) as refused:
            opt.load_ingredients_from_csv(pd.DataFrame({
                "Name": ["Flour", "Water"], "Lowest": [10, None],
                "Highest": [50, None], "Unit": ["g", "g"],
                "Formula": ["=REST", "= rest"]}))
        assert str(refused.value) == wording.one_balance_only(
            "Flour and Water")

    def test_the_set_up_sheet_glosses_the_rest_row_however_it_was_spelled(
            self, tmp_path, monkeypatch):
        """The Set-up sheet's own compare was the second literal one."""
        opt = self._opt(tmp_path, monkeypatch, name="sheet_rest")
        opt.load_ingredients_from_csv(pd.DataFrame({
            "Name": ["Flour", "Water"], "Lowest": [10, None],
            "Highest": [60, None], "Unit": ["g", "g"],
            "Formula": [None, "=REST"]}))
        assert wording.setup_sheet_formula_text(
            opt._formula_text(opt._var_by_name("Water")),
            rest=bool(opt._var_by_name("Water").get('balance'))) == (
                "Fill to total (batch size − every other ingredient)")

    def test_a_refused_file_leaves_the_ingredients_alone(self, tmp_path,
                                                         monkeypatch):
        """A refusal writes nothing, here as everywhere. The list was
        emptied and refilled before the formulas were read, so a file
        refused for two balance rows left them standing."""
        opt = self._opt(tmp_path, monkeypatch, name="file_refused")
        before = [dict(v) for v in opt.variables]
        with pytest.raises(ValueError) as refused:
            opt.load_ingredients_from_csv(pd.DataFrame({
                "Name": ["Flour", "Water"], "Lowest": [10, None],
                "Highest": [50, None], "Unit": ["g", "g"],
                "Formula": ["= rest", "= rest"]}))
        assert str(refused.value) == wording.one_balance_only(
            "Flour and Water")
        assert opt.variables == before

    def test_the_samples_water_keeps_the_range_its_formula_covers(
            self, tmp_path, monkeypatch):
        """A worked-out row may carry a dormant range: the grid shows the
        word, and rubbing the formula out gives the row its amounts back
        rather than a row pinned at nothing."""
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("sample_range", robust=False)
        opt.set_amount_unit("g")
        opt.load_ingredients_from_csv(pd.read_csv(_SAMPLE_CSV))
        opt.add_objective("Juiciness", 1.0, goal="target", target=7,
                          min_val=0, max_val=10, unit="/10")
        opt.set_formulation_total(100.0)
        water = opt._var_by_name("Remaining water")
        assert water['balance'] is True
        assert water['bounds'] == (0.0, 100.0)
        frame = opt.ingredient_grid_frame()
        row = frame[frame[wording.NAME_LABEL] == "Remaining water"].iloc[0]
        assert row[wording.LOWEST_LABEL] == ""
        assert row[wording.HIGHEST_LABEL] == ""
        at = int(frame.index[frame[wording.NAME_LABEL] == "Remaining water"][0])
        errors, _ = opt.apply_ingredient_grid(_edit(
            frame, at, **{wording.FORMULA_LABEL: ""}))
        assert errors == []
        back = opt.ingredient_grid_frame()
        row = back[back[wording.NAME_LABEL] == "Remaining water"].iloc[0]
        assert (row[wording.LOWEST_LABEL], row[wording.HIGHEST_LABEL]) == \
            ("0.00", "100.00")


# ------------------------------------------------------------------ #
# 0.5.0 wave 2, "rules" (task 5): a limit can say Exactly, and be written
# in grams or as a % of the default batch size.
# ------------------------------------------------------------------ #

class TestExactlyAndPercentLimits:
    """Exactly stores the number typed, not the band it is enforced as — a
    continuous search cannot be held to a point, the same reason the total
    of each formulation is a band too. A percent limit stores the percent
    it was written as, alongside the grams it comes to today, so a later
    change of default batch size can rewrite it instead of stranding it."""

    def _opt(self, tmp_path, monkeypatch, name="exactly_percent"):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer(name, robust=False)
        opt.set_amount_unit("g")
        opt.add_ingredient("Water", 0, 200)
        opt.add_ingredient("Oil", 0, 200)
        opt.add_ingredient("Salt", 0, 20)
        opt.add_objective("Taste", 1.0, goal="max")
        return opt

    def test_exactly_writes_a_band_and_reads_back_the_typed_number(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_quantity_constraint(["Water", "Oil"], exactly=50)
        qc = opt.quantity_constraints[0]
        assert qc['exactly'] == 50.0
        # Half a percent either way: an equality is not something a
        # continuous search can be held to exactly.
        assert qc['min'] == pytest.approx(49.75)
        assert qc['max'] == pytest.approx(50.25)
        assert opt.limit_text(qc) == "Water + Oil: exactly 50 g"

    def test_exactly_wins_over_a_range_rather_than_refusing_it(
            self, tmp_path, monkeypatch):
        """The form has one control for one idea now — a "Limit is" picker
        showing only the boxes its answer needs — so the combination the
        old refusal existed for cannot be typed. A caller that sends both
        gets the well-defined answer instead of a sentence nobody can see."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_quantity_constraint(["Water", "Oil"], min_val=10, exactly=50)
        qc = opt.quantity_constraints[-1]
        assert qc['exactly'] == 50.0
        assert qc['min'] < 50.0 < qc['max']
        assert opt.limit_text(qc) == "Water + Oil: exactly 50 g"

    def test_exactly_on_one_ingredient_points_at_the_grid(self, tmp_path,
                                                          monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        with pytest.raises(ValueError) as refused:
            opt.add_quantity_constraint(["Water"], exactly=10)
        assert str(refused.value) == wording.EXACTLY_ONE_INGREDIENT
        assert opt.quantity_constraints == []

    def test_a_percent_limit_converts_to_grams_at_save(self, tmp_path,
                                                        monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_formulation_total(100)
        opt.add_quantity_constraint(["Water", "Oil"], max_val=30,
                                    percent=True)
        qc = [q for q in opt.quantity_constraints if q.get('percent')][0]
        assert qc['percent'] == {'min': None, 'max': 30.0, 'exactly': None}
        assert qc['min'] is None
        assert qc['max'] == pytest.approx(30.0)
        assert opt.limit_text(qc) == (
            "Water + Oil: at most 30 % of default batch size "
            "(30 g at the default 100 g)")

    def test_a_percent_limit_follows_a_new_default_batch_size(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_formulation_total(80)
        opt.add_quantity_constraint(["Water", "Oil"], max_val=30,
                                    percent=True)
        messages = opt.set_formulation_total(120)
        assert ("info", "Limits written as a % of the default batch size "
                        "are now calculated from 120 g.") in messages
        qc = [q for q in opt.quantity_constraints if q.get('percent')][0]
        # The percent itself is unchanged; the grams it comes to follow the
        # new default.
        assert qc['percent']['max'] == 30.0
        assert qc['max'] == pytest.approx(36.0)
        assert opt.limit_text(qc) == (
            "Water + Oil: at most 30 % of default batch size "
            "(36 g at the default 120 g)")

    def test_a_percent_limit_that_the_new_default_makes_unreachable_is_removed_and_named(
            self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("percent_unreachable", robust=False)
        opt.set_amount_unit("g")
        opt.add_ingredient("Water", 40, 60)
        opt.add_ingredient("Oil", 0, 40)
        opt.add_objective("Taste", 1.0, goal="max")
        opt.set_formulation_total(80)
        # 50 % of 80 g is 40 g, exactly Water's own Lowest: still reachable.
        opt.add_quantity_constraint(["Water"], max_val=50, percent=True)
        messages = opt.set_formulation_total(60)
        # 50 % of 60 g is 30 g, and Water alone can never be under 40 g.
        assert ("warning", "The limit on Water was deleted: a default "
                          "batch size of 60 g can no longer reach it.") \
            in messages
        assert [q for q in opt.quantity_constraints if q.get('percent')] == []

    def test_clearing_the_default_removes_every_percent_limit(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_formulation_total(100)
        opt.add_quantity_constraint(["Water", "Oil"], max_val=30,
                                    percent=True)
        messages = opt.clear_formulation_total()
        assert messages == [
            ("warning", "The limit on Water + Oil was deleted: it was a "
                       "% of the default batch size, and there is none "
                       "now.")]
        assert opt.quantity_constraints == []
        assert opt.formulation_total is None

    def test_a_limit_whose_every_row_is_worked_out_is_refused(
            self, tmp_path, monkeypatch):
        """A limit can only change what the search moves. One whose every
        row is filled in by a rule was accepted, listed, and then quietly
        contradicted by those rules."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_formulation_total(100)
        opt.set_formula("Water", "= rest")
        opt.set_formula("Oil", "= 0.1 × Salt")
        with pytest.raises(ValueError) as refused:
            opt.add_quantity_constraint(["Water", "Oil"], max_val=40)
        assert str(refused.value) == wording.limit_on_worked_out_rows(
            "Water and Oil", many=True)
        # One row the search still moves and the limit stands.
        opt.add_quantity_constraint(["Water", "Salt"], max_val=90)
        assert opt.quantity_constraints[-1]['ingredients'] == ["Water",
                                                               "Salt"]

    def test_a_percent_limit_survives_export_and_import(self, tmp_path,
                                                         monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_formulation_total(100)
        opt.add_quantity_constraint(["Water", "Oil"], max_val=30,
                                    percent=True)
        state = opt.export_json()
        FoodOptimizer.validate_state(state)
        fresh = FoodOptimizer("copy_of_exactly_percent")
        fresh.import_json(state)
        before = [q for q in opt.quantity_constraints if q.get('percent')][0]
        after = [q for q in fresh.quantity_constraints
                if q.get('percent')][0]
        assert after == before
        assert fresh.limit_text(after) == opt.limit_text(before)

    def test_validate_state_refuses_a_malformed_percent_block(self, tmp_path,
                                                              monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_formulation_total(100)
        opt.add_quantity_constraint(["Water", "Oil"], max_val=30,
                                    percent=True)
        base = opt.export_json()
        # Not a dict at all.
        bad_shape = json.loads(json.dumps(base))
        bad_shape['quantity_constraints'][-1]['percent'] = "30"
        with pytest.raises(ValueError, match="damaged"):
            FoodOptimizer.validate_state(bad_shape)
        # A field that is not a number or None.
        bad_field = json.loads(json.dumps(base))
        bad_field['quantity_constraints'][-1]['percent'] = {
            'min': None, 'max': "thirty", 'exactly': None}
        with pytest.raises(ValueError, match="damaged"):
            FoodOptimizer.validate_state(bad_field)
        # An unexpected key.
        bad_key = json.loads(json.dumps(base))
        bad_key['quantity_constraints'][-1]['percent'] = {
            'min': None, 'max': 30.0, 'exactly': None, 'source': 'x'}
        with pytest.raises(ValueError, match="damaged"):
            FoodOptimizer.validate_state(bad_key)
        # A well-formed one still opens.
        FoodOptimizer.validate_state(base)


class TestPercentLimitsFollowTheDefaultsRealFate:
    """Fix round 1, finding 1. _sync_formulation_total — the door a unit
    split or amounts that no longer reach it blanks the default through —
    left every % of batch size limit standing: stale, still enforced, and
    naming a batch size that no longer existed. It now owns the same
    consequence clear_formulation_total's own box does: the limit goes
    with the default, named in one sentence."""

    def _opt(self, tmp_path, monkeypatch, name="percent_sync"):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer(name, robust=False)
        opt.set_amount_unit("g")
        opt.add_ingredient("Water", 0, 200)
        opt.add_ingredient("Oil", 0, 200)
        opt.add_ingredient("Salt", 0, 20)
        opt.add_objective("Taste", 1.0, goal="max")
        opt.set_formulation_total(100)
        opt.add_quantity_constraint(["Water", "Oil"], max_val=30,
                                    percent=True)
        return opt

    def test_a_unit_split_takes_every_percent_limit_with_it(self, tmp_path,
                                                            monkeypatch):
        """The reviewer's repro: Salt's unit is split off through an
        ordinary grid Save, which blanks the default the same way it always
        has. The percent limit must go with it, named, rather than survive
        stale and still enforced at yesterday's grams."""
        opt = self._opt(tmp_path, monkeypatch)
        errors, messages = opt.apply_ingredient_grid(_edit(
            opt.ingredient_grid_frame(), 3, **{wording.UNIT_LABEL: "ml"}))
        assert errors == []
        assert opt.formulation_total is None
        assert opt.quantity_constraints == []
        assert ("The limit on Water + Oil was deleted: it was a % of the "
               "default batch size, and there is none now.") in _said(messages)
        # Nothing is enforcing the stale 30 g cap any more.
        assert opt._check_constraints(
            {"Water": 90.0, "Oil": 90.0, "Salt": 5.0})

    def test_a_grid_save_that_does_not_blank_the_default_leaves_it(
            self, tmp_path, monkeypatch):
        """The same door, on a save that keeps the ingredients in one unit
        and the default still reachable: rewritten in place, as always, and
        the percent limit is not this save's business at all."""
        opt = self._opt(tmp_path, monkeypatch)
        errors, messages = opt.apply_ingredient_grid(_edit(
            opt.ingredient_grid_frame(), 3, **{wording.HIGHEST_LABEL: 25.0}))
        assert errors == []
        assert opt.formulation_total == 100.0
        qcs = [q for q in opt.quantity_constraints if q.get('percent')]
        assert len(qcs) == 1
        assert qcs[0]['percent'] == {'min': None, 'max': 30.0, 'exactly': None}
        assert qcs[0]['max'] == pytest.approx(30.0)
        assert not any("% of batch size" in m for m in _said(messages))


# ------------------------------------------------------------------ #
#  0.7.0 wave 3 · pre-mixes: the data, and the two mappings
# ------------------------------------------------------------------ #

class TestPreMixes:
    """A pre-mix is an ingredient made from its own parts, and it reaches
    the flat list of amounts the model searches one of two ways: made once
    and portioned into every formulation (the pre-mix is one row and its
    parts are no rows at all), or weighed into each formulation (the parts
    are the rows and the pre-mix is not one). Never both, and never
    neither."""

    def _opt(self, tmp_path, monkeypatch, name="premixes"):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer(name, robust=False)
        opt.set_amount_unit("g")
        opt.add_ingredient("Sugar", 1, 5)
        opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
        return opt

    @staticmethod
    def _names(opt):
        return [v['name'] for v in opt.variables]

    @staticmethod
    def _row(opt, name):
        return opt._by_name().get(name)

    @staticmethod
    def _parts(*pairs):
        return [{'name': name, 'share': share} for name, share in pairs]

    def _dry_blend(self, opt, mode="portioned"):
        opt.add_premix("Dry blend", mode)
        opt.set_premix_parts("Dry blend", self._parts(("Flour", 70),
                                                      ("Salt", 30)))
        return opt

    # ---- the two mappings ---------------------------------------------- #

    def test_portioned_puts_the_premix_in_the_list_and_not_its_parts(
            self, tmp_path, monkeypatch):
        opt = self._dry_blend(self._opt(tmp_path, monkeypatch))
        assert self._names(opt) == ["Sugar", "Dry blend"]
        # One continuous row with its own Lowest and Highest, like any
        # other ingredient.
        opt.add_ingredient("Dry blend", 5, 15)
        assert self._row(opt, "Dry blend")['bounds'] == (5.0, 15.0)
        assert self._row(opt, "Flour") is None
        assert self._row(opt, "Salt") is None
        assert [p['name'] for p in opt.premix_parts("Dry blend")] == \
            ["Flour", "Salt"]
        # The parts are still names the project knows facts about.
        assert "Flour" in opt.ingredient_properties
        assert opt.premix_of("Flour") == ["Dry blend"]
        assert opt.premix_of("Dry blend") == []

    def test_weighed_puts_the_parts_in_the_list_and_not_the_premix(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_premix("Wet blend", "weighed")
        opt.set_premix_parts("Wet blend", self._parts(("Water", 60),
                                                      ("Oil", 40)))
        assert self._names(opt) == ["Sugar", "Water", "Oil"]
        assert self._row(opt, "Wet blend") is None
        opt.add_ingredient("Water", 20, 40)
        opt.add_ingredient("Oil", 2, 8)
        # A limit on the group is a limit over its parts, tagged with the
        # group so it is named by the group and not by Water + Oil.
        opt.add_quantity_constraint(["Water", "Oil"], max_val=45,
                                    source="premix:Wet blend")
        qc = opt.quantity_constraints[-1]
        assert opt.limit_label(qc) == "Wet blend"
        assert opt.limit_text(qc) == "Wet blend: at most 45 g"

    def test_no_mode_ever_counts_the_mass_twice(self, tmp_path, monkeypatch):
        """Flipping how a pre-mix is made never leaves the group AND its
        parts in the list: the total of a formulation would count the same
        flour twice. Nor may it leave neither."""
        opt = self._dry_blend(self._opt(tmp_path, monkeypatch))
        for mode in ("weighed", "portioned", "weighed", "portioned"):
            opt.set_premix_mode("Dry blend", mode)
            listed = set(self._names(opt))
            group = "Dry blend" in listed
            parts = listed & {"Flour", "Salt"}
            assert group != bool(parts)
            if mode == "portioned":
                assert group and not parts
            else:
                assert parts == {"Flour", "Salt"}
        # The parts themselves never change with the way it is made.
        assert [p['name'] for p in opt.premix_parts("Dry blend")] == \
            ["Flour", "Salt"]

    # ---- the parts ------------------------------------------------------ #

    def test_parts_are_rebalanced_to_a_hundred(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_premix("Dry blend", "portioned")
        adjusted = opt.set_premix_parts(
            "Dry blend", self._parts(("Flour", 30), ("Salt", 30)))
        assert adjusted is True
        assert [p['share'] for p in opt.premix_parts("Dry blend")] == \
            [50.0, 50.0]
        # The sentence names the column it moved and the numbers it wrote:
        # the reader typed 30 and 30 and the app wrote 50 and 50.
        assert wording.shares_adjusted_premix("Flour 50.00, Salt 50.00") == (
            "Composition (%) adjusted to add up to 100 %: Flour 50.00, "
            "Salt 50.00.")
        # Already adding up to 100: nothing moved, and nothing to say.
        assert opt.set_premix_parts(
            "Dry blend", self._parts(("Flour", 70), ("Salt", 30))) is False
        assert [p['share'] for p in opt.premix_parts("Dry blend")] == \
            [70.0, 30.0]

    def test_a_part_may_belong_to_two_premixes(self, tmp_path, monkeypatch):
        """Water is in the dry blend and in the wet one. One name, one set
        of properties, and its own facts on each entry."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_premix("Dry blend", "portioned")
        opt.set_premix_parts("Dry blend", [
            {'name': "Flour", 'share': 70, 'vendor': "Mill Co"},
            {'name': "Water", 'share': 30, 'unit': "ml"}])
        opt.add_premix("Wet blend", "portioned")
        opt.set_premix_parts("Wet blend", [
            {'name': "Oil", 'share': 60},
            {'name': "Water", 'share': 40, 'unit': "g"}])
        assert opt.premix_of("Water") == ["Dry blend", "Wet blend"]
        assert self._names(opt) == ["Sugar", "Dry blend", "Wet blend"]
        # The facts ride on the entry, so the same name can be bought two
        # ways; the properties are keyed by name and shared.
        assert [p['unit'] for p in opt.premix_parts("Dry blend")] == ["", "ml"]
        assert [p['unit'] for p in opt.premix_parts("Wet blend")] == ["", "g"]
        assert list(opt.ingredient_properties).count("Water") == 1

    def test_a_premix_cannot_contain_itself(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_premix("Dry blend", "portioned")
        with pytest.raises(ValueError) as caught:
            opt.set_premix_parts("Dry blend", self._parts(("Dry blend", 100)))
        assert str(caught.value) == wording.PART_IS_ITS_OWN_PREMIX
        assert wording.PART_IS_ITS_OWN_PREMIX == \
            "A pre-mix cannot be a part of another pre-mix. Type an ingredient's name in the Part cell."
        opt.add_premix("Wet blend", "weighed")
        with pytest.raises(ValueError) as caught:
            opt.set_premix_parts("Dry blend", self._parts(("Wet blend", 100)))
        assert str(caught.value) == wording.PREMIX_INSIDE_PREMIX
        assert wording.PREMIX_INSIDE_PREMIX == \
            "A pre-mix cannot be a part of another pre-mix. Type an ingredient's name in the Part cell."
        # Nothing was written by either refusal.
        assert opt.premix_parts("Dry blend") == []

    def test_removing_a_premix_takes_its_rows_with_it(self, tmp_path,
                                                      monkeypatch):
        """The rows go; the name stays where another pre-mix still holds
        it as a part, with the facts filed under it."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_premix("Wet blend", "weighed")
        opt.set_premix_parts("Wet blend", self._parts(("Water", 60),
                                                      ("Oil", 40)))
        opt.add_premix("Fry blend", "portioned")
        opt.set_premix_parts("Fry blend", self._parts(("Oil", 100)))
        opt.add_ingredient("Water", 20, 40)
        opt.add_ingredient("Oil", 2, 8)
        opt.add_quantity_constraint(["Water", "Oil"], max_val=45,
                                    source="premix:Wet blend")
        removed = opt.remove_premix("Wet blend")
        assert list(opt.premixes) == ["Fry blend"]
        # Both rows go: Fry blend is made in one bowl, so Oil being one of
        # its parts is not a row. Fry blend's own row is what is left.
        assert self._names(opt) == ["Sugar", "Fry blend"]
        assert opt.quantity_constraints == []
        assert opt.limit_removed_messages(removed) == [
            ("warning", wording.quantity_limit_removed_missing(
                "Water + Oil",
                wording.no_longer_ingredients("Water and Oil", True)))]
        # Oil is still a part, and still a name the project knows facts
        # about; Water went with the pre-mix that was its only home.
        assert opt.premix_of("Oil") == ["Fry blend"]
        assert "Oil" in opt.ingredient_properties
        assert "Water" not in opt.ingredient_properties
        with pytest.raises(ValueError) as caught:
            opt.premix_parts("Wet blend")
        assert str(caught.value) == wording.premix_unknown("Wet blend")

    def test_a_stored_make_up_reads_back_by_round(self, tmp_path,
                                                  monkeypatch):
        """What a round was actually made with is fixed for that round: the
        make-up on file wins over what the pre-mix says today."""
        opt = self._dry_blend(self._opt(tmp_path, monkeypatch))
        opt.premixes["Dry blend"]['versions'][1] = opt.premix_parts("Dry blend")
        opt.set_premix_parts("Dry blend", self._parts(("Flour", 50),
                                                      ("Salt", 50)))
        assert [p['share'] for p in opt.premix_parts("Dry blend")] == \
            [50.0, 50.0]
        assert [p['share'] for p in opt.premix_parts("Dry blend", 1)] == \
            [70.0, 30.0]
        # A round with no make-up on file reads today's.
        assert [p['share'] for p in opt.premix_parts("Dry blend", 2)] == \
            [50.0, 50.0]

    # ---- a copy ---------------------------------------------------------- #

    def test_premixes_survive_export_and_import(self, tmp_path, monkeypatch):
        opt = self._dry_blend(self._opt(tmp_path, monkeypatch))
        opt.add_premix("Wet blend", "weighed")
        opt.set_premix_parts("Wet blend", self._parts(("Water", 60),
                                                      ("Oil", 40)))
        # The make-up one round was made with, filed under that round.
        opt.premixes["Dry blend"]['versions'][1] = opt.premix_parts("Dry blend")
        state = opt.export_json()
        assert state['premixes']['Dry blend']['mode'] == "portioned"
        assert state['premixes']['Wet blend']['mode'] == "weighed"
        # A JSON object's keys are strings, so the round numbers go out as
        # text and come back whole.
        assert list(state['premixes']['Dry blend']['versions']) == ["1"]
        assert FoodOptimizer.validate_state(state)['version'] == 15

        monkeypatch.chdir(tmp_path)
        other = FoodOptimizer("premix_copy", robust=False)
        other.import_json(state)
        assert list(other.premixes) == ["Dry blend", "Wet blend"]
        assert other.premixes["Dry blend"]['parts'] == \
            opt.premixes["Dry blend"]['parts']
        assert list(other.premixes["Dry blend"]['versions']) == [1]
        assert self._names(other) == ["Sugar", "Dry blend", "Water", "Oil"]

    def test_validate_state_refuses_a_bad_mode(self, tmp_path, monkeypatch):
        opt = self._dry_blend(self._opt(tmp_path, monkeypatch))
        opt.add_objective("Firmness", 1.0, goal="max", min_val=0, max_val=10)

        def refused(change):
            state = opt.export_json()
            change(state['premixes'], state)
            with pytest.raises(ValueError) as caught:
                FoodOptimizer.validate_state(state)
            assert str(caught.value) == wording.COPY_DAMAGED

        refused(lambda p, s: p["Dry blend"].__setitem__('mode', "stirred"))
        refused(lambda p, s: p["Dry blend"].__setitem__('mode', None))
        refused(lambda p, s: p["Dry blend"]['parts'][0]
                .__setitem__('share', "seventy"))
        refused(lambda p, s: p["Dry blend"]['parts'][0].__setitem__('name', ""))
        refused(lambda p, s: p["Dry blend"]['versions'].__setitem__('two', []))
        # A pre-mix that also names a measurement, or a property.
        refused(lambda p, s: p.__setitem__(
            "Firmness", {'mode': "portioned", 'parts': [], 'versions': {}}))
        refused(lambda p, s: p.__setitem__(
            "Protein", {'mode': "portioned", 'parts': [], 'versions': {}})
            or s.__setitem__('property_names', ["Protein"]))
        # And the copy the app itself writes is accepted.
        assert FoodOptimizer.validate_state(opt.export_json())['version'] == 15

    def test_a_version_12_project_loads_with_no_premixes_at_version_14(
            self, tmp_path, monkeypatch):
        assert FoodOptimizer.CLASS_VERSION == 15
        opt = self._opt(tmp_path, monkeypatch)
        state = opt.export_json()
        del state['premixes']
        state['CLASS_VERSION'] = 12
        assert FoodOptimizer.validate_state(state)['version'] == 12
        monkeypatch.chdir(tmp_path)
        other = FoodOptimizer("held_project", robust=False)
        other.import_json(state)
        assert other.premixes == {}
        assert other.export_json()['CLASS_VERSION'] == 15

    # ---- a file ---------------------------------------------------------- #

    @staticmethod
    def _premix_csv():
        return pd.DataFrame([
            {'Name': "Dry blend", 'Lowest': 5, 'Highest': 15, 'Unit': "g",
             wording.MADE_AS_LABEL: wording.PREMIX_MADE_AS_PORTIONED},
            {'Name': "Flour", 'Unit': "g", wording.PREMIX_LABEL: "Dry blend",
             wording.PREMIX_SHARE_LABEL: 70, 'Protein': 10},
            {'Name': "Salt", 'Unit': "g", wording.PREMIX_LABEL: "Dry blend",
             wording.PREMIX_SHARE_LABEL: 30},
            {'Name': "Wet blend",
             wording.MADE_AS_LABEL: wording.PREMIX_MADE_AS_WEIGHED},
            {'Name': "Water", 'Lowest': 20, 'Highest': 40, 'Unit': "ml",
             wording.PREMIX_LABEL: "Wet blend"},
            {'Name': "Oil", 'Lowest': 2, 'Highest': 8, 'Unit': "g",
             wording.PREMIX_LABEL: "Wet blend"},
            {'Name': "Sugar", 'Lowest': 1, 'Highest': 5, 'Unit': "g"},
        ], columns=['Name', 'Lowest', 'Highest', 'Unit', wording.PREMIX_LABEL,
                    wording.MADE_AS_LABEL, wording.PREMIX_SHARE_LABEL,
                    'Protein'])

    def test_a_csv_with_premix_columns_builds_both_kinds(self, tmp_path,
                                                         monkeypatch):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("premix_file", robust=False)
        opt.load_ingredients_from_csv(self._premix_csv())
        assert list(opt.premixes) == ["Dry blend", "Wet blend"]
        assert opt.premixes["Dry blend"]['mode'] == "portioned"
        assert opt.premixes["Wet blend"]['mode'] == "weighed"
        # Portioned: the pre-mix's own row carries the amounts; its parts
        # are not in the list. Weighed: the other way round.
        assert self._names(opt) == ["Dry blend", "Water", "Oil", "Sugar"]
        assert self._row(opt, "Dry blend")['bounds'] == (5.0, 15.0)
        assert self._row(opt, "Water")['bounds'] == (20.0, 40.0)
        assert opt.unit_of("Water") == "ml"
        assert [(p['name'], p['share']) for p in opt.premix_parts("Dry blend")] \
            == [("Flour", 70.0), ("Salt", 30.0)]
        assert [p['name'] for p in opt.premix_parts("Wet blend")] == \
            ["Water", "Oil"]
        # A part's own facts, and the properties keyed by its name.
        assert opt.premix_parts("Dry blend")[0]['unit'] == "g"
        assert opt.ingredient_properties["Flour"] == {"Protein": 10.0}
        assert "Protein" in opt.properties()
        assert wording.PREMIX_LABEL not in opt.properties()

    def test_a_csv_without_premix_columns_loads_as_before(self, tmp_path,
                                                          monkeypatch):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("plain_file", robust=False)
        opt.load_ingredients_from_csv(pd.DataFrame({
            "Name": ["Water", "Flour"], "Lowest": [10, 20],
            "Highest": [60, 40], "Unit": ["g", "g"]}))
        assert opt.premixes == {}
        assert opt.variables
        assert all(opt.premix_of(v['name']) == [] for v in opt.variables)

    def test_the_setup_sheet_round_trips_premixes(self, tmp_path,
                                                  monkeypatch):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("premix_sheet", robust=False)
        opt.load_ingredients_from_csv(self._premix_csv())
        opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
        sheet = _book(opt.all_formulations_workbook())[wording.SET_UP_SHEET]
        header = [c.value for c in sheet[2]]
        # One table about the make-up, not two. The list of rows says which
        # pre-mix a part belongs to — the one fact the list cannot say
        # otherwise — and `Made as` and `% of pre-mix` are the Pre-mixes
        # block's own, six rows down the same page.
        assert wording.PART_OF_LABEL in header
        assert wording.PREMIX_LABEL not in header
        assert wording.MADE_AS_LABEL not in header
        assert wording.PREMIX_SHARE_LABEL not in header
        rows = []
        for r in range(3, sheet.max_row + 1):
            values = [sheet.cell(row=r, column=c + 1).value
                      for c in range(len(header))]
            if not values[0]:
                break
            rows.append(dict(zip(header, values)))
        by_name = {row[wording.NAME_LABEL]: row for row in rows}
        assert by_name["Flour"][wording.PART_OF_LABEL] == "Dry blend"
        assert by_name["Dry blend"][wording.PART_OF_LABEL] is None

        # The Pre-mixes block carries the make-up, and every number in it is
        # a number: a portioned share was a float and a weighed part's range
        # a string, because the block was lifted off the screen's own frame.
        cells = [[c.value for c in row] for row in sheet.iter_rows()]
        start = next(i for i, row in enumerate(cells)
                     if row[0] == wording.PREMIXES_HEADING)
        block = cells[start:start + 12]
        assert [row[0] for row in block if row[0] in opt.premixes] == \
            opt.premix_grid_order()
        made_up = {row[0]: row[1:4] for row in block}
        # A portioned share and a weighed part's two amounts are all
        # numbers here. The screen's frame holds the range as text, because
        # a grid column that may read `sum of its parts` cannot be a number
        # column; the sheet has no such trouble.
        assert isinstance(made_up["Flour"][0], (int, float))
        assert isinstance(made_up["Oil"][0], (int, float))
        assert isinstance(made_up["Oil"][1], (int, float))

    # ---- Made as: the choice, its sentence, and switching ---------------- #

    def test_the_portioned_sentence_is_said_once_when_the_choice_is_made(
            self, tmp_path, monkeypatch):
        """One sentence at the choice, saying both halves of what it means:
        what the suggestions will move, and what the bench will do. It
        names the pre-mix and never its parts — portioned, the parts are
        not what varies — and it is said when the choice is MADE, so
        re-choosing the way a pre-mix is already made says nothing and
        changes nothing."""
        opt = self._dry_blend(self._opt(tmp_path, monkeypatch))
        said = opt.premix_consequence("Dry blend")
        assert said.startswith(
            "The suggestions vary how much Dry blend goes in. Its make-up "
            "stays the same for the whole round, so you make it once.")
        assert said.startswith(wording.premix_portioned_consequence(
            "Dry blend"))
        assert "Flour" not in said and "Salt" not in said
        # The row arrived with no amounts of its own, and the line says so.
        assert said.endswith(wording.premix_parts_need_amounts("Dry blend"))
        opt.add_ingredient("Dry blend", 5, 15)
        assert opt.premix_consequence("Dry blend").endswith(
            wording.premix_row_band("Dry blend", "5.00", "15.00 g"))
        assert opt.set_premix_mode("Dry blend", "portioned") == []
        assert opt.premixes["Dry blend"]['mode'] == "portioned"

    def test_the_weighed_sentence_names_every_part(self, tmp_path,
                                                   monkeypatch):
        """The other half of the same choice. Weighed, it is the parts the
        suggestions move, so the sentence lists them — in the order they
        are typed, joined the way every other list of names in the app is."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_premix("Dry blend", "weighed")
        opt.set_premix_parts("Dry blend", self._parts(
            ("pea protein", 50), ("fibre", 30), ("salt", 20)))
        said = opt.premix_consequence("Dry blend")
        # Two sentences: what the suggestions will vary, and what happened
        # to the numbers. The parts arrived as rows with no amounts of
        # their own, and a round generated in that state has none of them
        # in it — so the line says so rather than leaving it to be found.
        assert said.startswith(
            "The suggestions vary pea protein, fibre and salt separately. "
            "Each formulation gets its own amounts of them.")
        assert said.endswith(wording.premix_parts_need_amounts(
            "pea protein, fibre and salt", many=True))
        opt.set_premix_parts("Dry blend", self._parts(("pea protein", 100)))
        assert opt.premix_consequence("Dry blend").startswith(
            wording.premix_weighed_consequence("pea protein"))
        # Nothing to name is nothing to say: a pre-mix with no parts yet
        # does not get a sentence with a hole in it.
        opt.add_premix("Wet blend", "weighed")
        assert opt.premix_consequence("Wet blend") == ""

    def test_switching_discards_the_open_round_with_the_usual_notice(
            self, tmp_path, monkeypatch):
        """How a pre-mix is made decides which rows the suggestions move,
        so a round generated before the switch is not a round this project
        would generate now. It goes through the same door every other
        set-up change sends it through — and the screen can ask first,
        because the model answers what the question needs (the round, how
        many formulations, and how many of them the reader added) without
        writing anything."""
        opt = self._dry_blend(self._opt(tmp_path, monkeypatch))
        opt.add_ingredient("Dry blend", 5, 15)
        opt.ask(2)
        opt.add_to_pending_batch({"Sugar": 2.0, "Dry blend": 10.0},
                                 note="my own")
        no = opt.pending_batch_no
        at_risk = opt.premix_mode_retires_round("Dry blend", "weighed")
        assert at_risk == {'round': no, 'formulations': 3, 'own': 1}
        assert opt.pending_batch_no == no
        assert opt.premixes["Dry blend"]['mode'] == "portioned"
        # The way it is already made is no change, so no round is at risk.
        assert opt.premix_mode_retires_round("Dry blend", "portioned") is None

        opt.set_premix_mode("Dry blend", "weighed")
        assert opt.pending_batch is None
        assert opt.pending_batch_no is None
        assert wording.batch_discarded_notice(no) == (
            f"Round {no} was discarded: your set-up changed after it was "
            "made. Generate a new one.")
        # And what the screen's question is built from, in wave 1's words.
        assert wording.saving_discards_round(no, "3 formulations", 1) == (
            f"Saving will discard Round {no}: 3 formulations, 1 of them "
            "added by you. A formulation you added goes with the round — a "
            "set-up change can make it invalid.")

    def test_portioned_to_weighed_back_fills_the_parts_from_the_round_version(
            self, tmp_path, monkeypatch):
        """The recorded formulations are not zeroed: 10 g of a blend that
        was 70/30 the day it was made IS 7 g of flour and 3 g of salt. The
        make-up on file for that round is what it is read with, not the
        make-up the pre-mix carries today; a row that belongs to no round
        reads today's."""
        opt = self._dry_blend(self._opt(tmp_path, monkeypatch))
        opt.add_ingredient("Dry blend", 5, 15)
        opt.premixes["Dry blend"]['versions'][1] = opt.premix_parts("Dry blend")
        opt.tell({"Sugar": 2.0, "Dry blend": 10.0}, {"Taste": 7.0},
                 formulation_no=1, batch_no=1)
        opt.set_premix_parts("Dry blend", self._parts(("Flour", 50),
                                                      ("Salt", 50)))
        opt.tell({"Sugar": 2.0, "Dry blend": 10.0}, {"Taste": 6.0},
                 formulation_no=2)
        assert opt.batch_history == [1, None]

        opt.set_premix_mode("Dry blend", "weighed")
        assert self._names(opt) == ["Sugar", "Flour", "Salt"]
        first, second = opt.recipe_history
        assert first["Flour"] == pytest.approx(7.0)
        assert first["Salt"] == pytest.approx(3.0)
        assert second["Flour"] == pytest.approx(5.0)
        assert second["Salt"] == pytest.approx(5.0)
        assert "Dry blend" not in first and "Dry blend" not in second
        # The searched history is rebuilt from the amounts, not left short.
        assert opt.X_history == [opt._encode(r) for r in opt.recipe_history]

    def test_weighed_to_portioned_sums_the_parts(self, tmp_path, monkeypatch):
        """The other direction. 6 g of flour and 2 g of salt were 8 g of
        blend, and what that round's blend WAS is the shares those amounts
        imply — which is what the round's make-up on file becomes, so the
        row can be read back either way afterwards."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_premix("Dry blend", "weighed")
        opt.set_premix_parts("Dry blend", self._parts(("Flour", 70),
                                                      ("Salt", 30)))
        opt.add_ingredient("Flour", 0, 20)
        opt.add_ingredient("Salt", 0, 5)
        opt.tell({"Sugar": 2.0, "Flour": 6.0, "Salt": 2.0}, {"Taste": 7.0},
                 formulation_no=1, batch_no=1)

        opt.set_premix_mode("Dry blend", "portioned")
        assert self._names(opt) == ["Sugar", "Dry blend"]
        assert opt.recipe_history[0]["Dry blend"] == pytest.approx(8.0)
        assert "Flour" not in opt.recipe_history[0]
        assert opt.X_history == [opt._encode(r) for r in opt.recipe_history]
        # The shares that round was really made to, filed under that round;
        # the make-up the pre-mix carries today is untouched.
        assert [(p['name'], p['share'])
                for p in opt.premix_parts("Dry blend", 1)] == \
            [("Flour", 75.0), ("Salt", 25.0)]
        assert [(p['name'], p['share'])
                for p in opt.premix_parts("Dry blend")] == \
            [("Flour", 70.0), ("Salt", 30.0)]

    def test_a_round_records_the_version_it_was_made_from(self, tmp_path,
                                                          monkeypatch):
        """A round is generated from one make-up, and what that make-up was
        is a fact about the round. It is written at generation, because the
        make-up can move the same afternoon."""
        opt = self._dry_blend(self._opt(tmp_path, monkeypatch))
        opt.add_ingredient("Dry blend", 5, 15)
        assert opt.premixes["Dry blend"]['versions'] == {}
        opt.ask(2)
        no = opt.pending_batch_no
        assert [(p['name'], p['share'])
                for p in opt.premix_parts("Dry blend", no)] == \
            [("Flour", 70.0), ("Salt", 30.0)]
        # Copies: editing the make-up cannot reach back into the round.
        opt.premixes["Dry blend"]['versions'][no][0]['share'] = 1.0
        assert [p['share'] for p in opt.premix_parts("Dry blend")] == \
            [70.0, 30.0]

    def test_a_recorded_formulation_remembers_its_version_after_the_parts_change(
            self, tmp_path, monkeypatch):
        """The point of filing it by round: a formulation recorded weeks
        ago still says what it was made of after the make-up moves."""
        opt = self._dry_blend(self._opt(tmp_path, monkeypatch))
        opt.add_ingredient("Dry blend", 5, 15)
        batch = opt.ask(1)
        no = opt.pending_batch_no
        opt.tell(batch[0], {"Taste": 7.0}, batch_no=no)
        opt.set_premix_parts("Dry blend", self._parts(("Flour", 50),
                                                      ("Salt", 50)))
        assert [(p['name'], p['share'])
                for p in opt.premix_version_of("Dry blend", 0)] == \
            [("Flour", 70.0), ("Salt", 30.0)]
        assert [p['share'] for p in opt.premix_parts("Dry blend")] == \
            [50.0, 50.0]
        # A row that belongs to no round has no version of its own to
        # remember, so it reads the make-up as it stands.
        opt.set_pending_batch(None)
        opt.tell({"Sugar": 2.0, "Dry blend": 10.0}, {"Taste": 5.0})
        assert opt.batch_history[1] is None
        assert [p['share'] for p in opt.premix_version_of("Dry blend", 1)] == \
            [50.0, 50.0]

    def test_a_switch_with_no_amounts_on_file_is_refused(self, tmp_path,
                                                         monkeypatch):
        """Changing which rows are searched rebuilds the history out of the
        recorded amounts, and a project that has lost them cannot have it
        rebuilt. The same refusal deleting an ingredient gives, and it is
        asked before anything is written."""
        opt = self._dry_blend(self._opt(tmp_path, monkeypatch))
        opt.add_ingredient("Dry blend", 5, 15)
        opt.tell({"Sugar": 2.0, "Dry blend": 10.0}, {"Taste": 7.0})
        opt.recipe_history = []
        with pytest.raises(ValueError) as caught:
            opt.set_premix_mode("Dry blend", "weighed")
        assert str(caught.value) == wording.AMOUNTS_MISSING_DELETE_ERROR
        assert opt.premixes["Dry blend"]['mode'] == "portioned"
        assert self._names(opt) == ["Sugar", "Dry blend"]

    def test_a_switch_that_would_leave_nothing_to_vary_is_refused(
            self, tmp_path, monkeypatch):
        """A pre-mix's own row arrives pinned at one amount, so making the
        only weighed pre-mix in a project portioned can leave a project
        with nothing for the suggestions to move. The same question
        deleting a pre-mix asks, and the same sentence."""
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("premix_last_row", robust=False)
        opt.set_amount_unit("g")
        opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
        opt.add_premix("Wet blend", "weighed")
        opt.set_premix_parts("Wet blend", self._parts(("Water", 60),
                                                      ("Oil", 40)))
        opt.add_ingredient("Water", 20, 40)
        opt.add_ingredient("Oil", 2, 8)
        opt.ask(1)
        # A switch that is going to be refused puts no round at risk: the
        # question a screen asks first is about a change that will happen.
        assert opt.premix_mode_retires_round("Wet blend", "portioned") is None
        with pytest.raises(ValueError) as caught:
            opt.set_premix_mode("Wet blend", "portioned")
        assert str(caught.value) == wording.LAST_VARYING_ROW_ERROR
        assert opt.pending_batch_no is not None
        assert self._names(opt) == ["Water", "Oil"]
        assert opt.premixes["Wet blend"]['mode'] == "weighed"
        # With one row of its own that can still move, it goes through.
        opt.add_ingredient("Sugar", 1, 5)
        opt.set_premix_mode("Wet blend", "portioned")
        # The pre-mix keeps the place its parts were reading in — Water and
        # Oil headed the list, and Sugar was typed underneath them.
        assert self._names(opt) == ["Wet blend", "Sugar"]

    def test_a_limit_the_switch_empties_is_said_in_a_sentence(
            self, tmp_path, monkeypatch):
        """A switch changes which rows are in the list and what they can
        add up to, so a limit written against the old list can stop meaning
        anything. What it emptied comes back from the switch itself, in the
        shape every other door onto the list hands it back, and the screen
        says the same line."""
        opt = self._dry_blend(self._opt(tmp_path, monkeypatch))
        opt.add_ingredient("Dry blend", 5, 15)
        opt.add_quantity_constraint(["Sugar", "Dry blend"], max_val=18)
        assert opt.total_reach() == (6.0, 20.0)
        opt.set_formulation_total(18)
        removed = opt.set_premix_mode("Dry blend", "weighed")
        assert [qc.get('reason') for qc in removed] == ["unreachable"]
        assert opt.limit_removed_messages(removed) == [
            ("warning", wording.formulation_total_gone_unreachable("18 g"))]
        assert opt.formulation_total is None
        # A limit that merely lost one of its rows is left naming the rows
        # it still has, exactly as it is everywhere else in the app.
        assert opt.quantity_constraints[0]['ingredients'] == ["Sugar"]

    # ---- the carried defect from wave 2's fix wave ----------------------- #

    def test_the_snap_never_leaves_a_rule_row_below_zero(self, tmp_path,
                                                         monkeypatch):
        """A rule with a negative low end — Salt = 20 − Pea protein, with
        Pea protein up to 25 — let the projection onto the batch size land
        Salt below nothing while total_reach called the size reachable. The
        size is reachable; the projection now reaches it, by pinning the
        rule at its floor and sharing what is left over the other rows."""
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("snap_floor", robust=False)
        opt.set_amount_unit("g")
        opt.add_ingredient("Water", 20, 60)
        opt.add_ingredient("Pea protein", 10, 25)
        opt.add_ingredient("Salt", 0, 3)
        opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
        opt.set_formula("Salt", "= 20 - Pea protein")
        assert opt.total_reach() == (40.0, 80.0)
        opt.set_formulation_total(40)

        snapped = opt._snap_to_total({"Water": 20.0, "Pea protein": 25.0}, 40)
        assert snapped is not None
        assert snapped["Salt"] >= 0.0
        assert snapped["Pea protein"] == pytest.approx(20.0)
        assert sum(snapped.values()) == pytest.approx(40.0)
        assert opt._check_constraints(snapped) is True

        for recipe in opt.ask(3):
            assert recipe["Salt"] >= -1e-9
            assert sum(recipe.values()) == pytest.approx(40.0, abs=1e-6)
            assert opt._check_constraints(recipe) is True


@pytest.mark.parametrize("word", ["share", "Share", "SHARE"])
def test_a_bare_share_is_refused_everywhere_but_food_bo(word):
    """The scoped allowance is the proof, as it was for "Batch". 0.7.0
    stores what per cent of a pre-mix each part is under the key 'share',
    and food_bo.py is the only file that names that key; a bare "Share"
    typed into wording.py is still the column header this guard exists to
    catch. The allowance is that exact lower-cased key and nothing else:
    a capitalised "Share" is refused in food_bo.py too."""
    for name in _USER_FACING_SOURCES:
        refused = _single_word_offenders(name, [word])
        if name == "food_bo.py" and word == "share":
            assert refused == [], (name, word)
        else:
            assert refused == [(name, word)], (name, word)


class TestPreMixesFixRoundOne:
    """Fix round 1 on the pre-mixes. Every one of these is the same
    invariant defended in one direction only: a name a pre-mix owns is a
    row of the searched list, or the thing a row is made of, and nothing
    else in the project may wear it or take it away behind the pre-mix's
    back."""

    def _opt(self, tmp_path, monkeypatch, name="premix_fixes"):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer(name, robust=False)
        opt.set_amount_unit("g")
        opt.add_ingredient("Sugar", 1, 5)
        opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
        opt.add_premix("Dry blend", "portioned")
        opt.set_premix_parts("Dry blend", [{'name': "Flour", 'share': 70},
                                           {'name': "Salt", 'share': 30}])
        opt.add_ingredient("Dry blend", 5, 15)
        return opt

    @staticmethod
    def _names(opt):
        return [v['name'] for v in opt.variables]

    @staticmethod
    def _row(opt, name):
        return opt._by_name().get(name)

    # ---- 1 · a name a pre-mix owns is not free -------------------------- #

    def test_a_weighed_premixs_part_name_is_not_free(
            self, tmp_path, monkeypatch):
        """Weighed, a part IS a row of the searched list. A second row of
        that name is the same oil in the bowl twice and nothing downstream
        can tell which mass is which."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_premix("Fat phase", "weighed")
        opt.set_premix_parts("Fat phase", [{'name': "Oil", 'share': 0,
                                            'unit': "g"}])
        # The row the pre-mix put in the list is the reader's to set
        # amounts on — that is an EDIT of the part's own row...
        opt.add_ingredient("Oil", 0, 10)
        assert self._row(opt, "Oil")['bounds'] == (0.0, 10.0)
        # ...but nothing else in the project may wear the name.
        with pytest.raises(ValueError) as caught:
            opt.add_process_parameter("Oil", 0, 10)
        assert str(caught.value) == wording.name_taken_by(
            "Oil", wording.AN_INGREDIENT)
        with pytest.raises(ValueError):
            opt.add_property("Oil")
        assert wording.name_taken_by_part("Oil", "Fat phase") == \
            "Oil is already a part of Fat phase."

    def test_a_portioned_premixs_part_may_also_be_a_row_of_its_own(
            self, tmp_path, monkeypatch):
        """Portioned, a part is no row at all — it is a quantity inside the
        pre-mix — so the same water may be part of the dry blend and a row
        beside it. Each mass is weighed once, and the round's totals add
        the two under the one name."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_ingredient("Flour", 0, 10)
        assert self._names(opt) == ["Sugar", "Dry blend", "Flour"]
        assert [p['name'] for p in opt.premix_parts("Dry blend")] == \
            ["Flour", "Salt"]
        # A measurement and a property are still one name, one thing.
        with pytest.raises(ValueError):
            opt.add_property("Flour")
    def test_a_premixs_own_name_is_taken(self, tmp_path, monkeypatch):
        """Said in the pre-mix's words when it is not also a row, and in the
        row's when it is."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_premix("Wet blend", "weighed")
        with pytest.raises(ValueError) as caught:
            opt.add_ingredient("Wet blend", 0, 10)
        assert str(caught.value) == wording.name_taken_by("Wet blend",
                                                          wording.A_PREMIX)
        with pytest.raises(ValueError) as caught:
            opt.add_process_parameter("Dry blend", 0, 10)
        assert str(caught.value) == wording.name_taken_by(
            "Dry blend", wording.AN_INGREDIENT)
        # ...but the row the pre-mix itself put in the list is still the
        # reader's to set amounts on.
        opt.add_ingredient("Dry blend", 4, 16)
        assert self._row(opt, "Dry blend")['bounds'] == (4.0, 16.0)

    def test_a_rename_cannot_take_a_name_a_premix_owns(self, tmp_path,
                                                       monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_premix("Fat phase", "weighed")
        opt.set_premix_parts("Fat phase", [{'name': "Oil", 'share': 0,
                                            'unit': "g"}])
        with pytest.raises(ValueError) as caught:
            opt.rename_variable("Sugar", "Oil")
        assert str(caught.value) == _name_taken_message("Oil", "ingredient")
        # A pre-mix that is not also a row — one weighed into each
        # formulation, before anything is in it — is caught by the pre-mix
        # pass and nothing else; a portioned one is its own row, and the
        # variables pass gets there first with the same answer.
        opt.add_premix("Wet blend", "weighed")
        with pytest.raises(ValueError) as caught:
            opt.rename_variable("Sugar", "Wet blend")
        assert str(caught.value) == wording.name_taken_by("Wet blend",
                                                          wording.A_PREMIX)
        with pytest.raises(ValueError) as caught:
            opt.rename_variable("Sugar", "Dry blend")
        assert str(caught.value) == _name_taken_message("Dry blend",
                                                        "ingredient")
        assert self._names(opt) == ["Sugar", "Dry blend", "Oil"]
        # And the grid is the same door: it reads every row's name against
        # what is not on the grid, and a weighed pre-mix's parts are.
        errors, _ = opt.apply_ingredient_grid(_edit(
            opt.ingredient_grid_frame(), 1, **{wording.NAME_LABEL: "Oil"}))
        assert errors == [(1, wording.name_taken_by_part("Oil",
                                                         "Fat phase"))]
        assert self._names(opt) == ["Sugar", "Dry blend", "Oil"]

    # ---- 2 · a pre-mix's row is not deleted as an ingredient ------------ #

    def test_a_premix_row_cannot_be_deleted_as_an_ingredient(
            self, tmp_path, monkeypatch):
        """Deleting a weighed pre-mix's part left it in `parts`, so the
        next save of the make-up resurrected it at 0-0. The pre-mix is
        where a part is taken out."""
        opt = self._opt(tmp_path, monkeypatch)
        with pytest.raises(ValueError) as caught:
            opt.remove_ingredient("Dry blend")
        assert str(caught.value) == wording.delete_the_premix_instead(
            "Dry blend", "Dry blend")
        opt.add_premix("Wet blend", "weighed")
        opt.set_premix_parts("Wet blend", [{'name': "Water", 'share': 60},
                                           {'name': "Oil", 'share': 40}])
        with pytest.raises(ValueError) as caught:
            opt.remove_ingredient("Water")
        assert str(caught.value) == wording.delete_the_premix_instead(
            "Water", "Wet blend")
        assert self._row(opt, "Water") is not None
        # The pre-mix's own door still works, and takes the rows with it.
        opt.remove_premix("Wet blend")
        assert self._names(opt) == ["Sugar", "Dry blend"]

    # ---- 3 · a rename moves the pre-mix with the row -------------------- #

    def test_renaming_a_premix_row_moves_the_premix_with_it(
            self, tmp_path, monkeypatch):
        """A rename left the pre-mix keyed to the old name, so the next
        sync saw an orphan row plus a name with no row and built a second
        one."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.premixes["Dry blend"]['versions'][1] = opt.premix_parts("Dry blend")
        opt.rename_variable("Dry blend", "Blend")
        assert list(opt.premixes) == ["Blend"]
        assert self._names(opt) == ["Sugar", "Blend"]
        assert [p['name'] for p in opt.premix_parts("Blend")] == \
            ["Flour", "Salt"]
        assert [p['name'] for p in opt.premix_parts("Blend", 1)] == \
            ["Flour", "Salt"]
        # A second sync builds nothing: the key and its row are one again.
        opt.set_premix_parts("Blend", [{'name': "Flour", 'share': 70},
                                       {'name': "Salt", 'share': 30}])
        assert self._names(opt) == ["Sugar", "Blend"]

        # And a weighed part's row carries its part entries with it, in
        # every pre-mix that names it and in every make-up on file.
        opt.add_premix("Wet blend", "weighed")
        opt.set_premix_parts("Wet blend", [{'name': "Water", 'share': 60},
                                           {'name': "Oil", 'share': 40}])
        opt.premixes["Wet blend"]['versions'][2] = opt.premix_parts("Wet blend")
        opt.rename_variable("Water", "Spring water")
        assert [p['name'] for p in opt.premix_parts("Wet blend")] == \
            ["Spring water", "Oil"]
        assert [p['name'] for p in opt.premix_parts("Wet blend", 2)] == \
            ["Spring water", "Oil"]
        assert self._names(opt) == ["Sugar", "Blend", "Spring water", "Oil"]

    # ---- 4 · a pre-mix is not named after a part ------------------------ #

    def test_a_premix_cannot_be_named_after_a_part(self, tmp_path,
                                                   monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        with pytest.raises(ValueError) as caught:
            opt.add_premix("Flour", "weighed")
        assert str(caught.value) == wording.PREMIX_INSIDE_PREMIX
        assert list(opt.premixes) == ["Dry blend"]

    # ---- 5 · edited part facts reach the row ---------------------------- #

    def test_edited_part_facts_reach_the_row(self, tmp_path, monkeypatch):
        """A vendor typed against a part of a weighed pre-mix never
        reached the row: the sync returned early whenever no row arrived
        or left."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_premix("Wet blend", "weighed")
        opt.set_premix_parts("Wet blend", [
            {'name': "Water", 'share': 60, 'unit': "ml"},
            {'name': "Oil", 'share': 40}])
        assert opt.unit_of("Water") == "ml"
        opt.set_premix_parts("Wet blend", [
            {'name': "Water", 'share': 60, 'unit': "ml",
             'vendor': "Spring Co", 'sku': "W-1"},
            {'name': "Oil", 'share': 40, 'vendor': "Press Co"}])
        assert self._row(opt, "Water")['vendor'] == "Spring Co"
        assert self._row(opt, "Water")['sku'] == "W-1"
        assert self._row(opt, "Oil")['vendor'] == "Press Co"
        # The unit is sticky: it is part of a limit's arithmetic and of the
        # default batch size, and blanking it from a part entry would move
        # both behind the reader's back. The unit is set where it is read.
        opt.set_premix_parts("Wet blend", [
            {'name': "Water", 'share': 60, 'vendor': "Spring Co"},
            {'name': "Oil", 'share': 40}])
        assert opt.unit_of("Water") == "ml"
        assert self._row(opt, "Oil")['vendor'] == ""

    # ---- 6 · a deleted pre-mix does not name a limit -------------------- #

    def test_a_limit_does_not_name_a_deleted_premix(self, tmp_path,
                                                    monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_premix("Wet blend", "weighed")
        opt.set_premix_parts("Wet blend", [{'name': "Water", 'share': 60},
                                           {'name': "Oil", 'share': 40}])
        opt.add_ingredient("Water", 20, 40)
        opt.add_ingredient("Oil", 2, 8)
        opt.add_quantity_constraint(["Water", "Oil"], max_val=45,
                                    source="premix:Wet blend")
        qc = opt.quantity_constraints[-1]
        assert opt.limit_label(qc) == "Wet blend"
        del opt.premixes["Wet blend"]
        assert opt.limit_label(qc) == "Water + Oil"

    # ---- 7 · the rulings from Task 2's concerns ------------------------- #

    def test_a_limit_the_premix_row_took_with_it_is_said(self, tmp_path,
                                                         monkeypatch):
        """A limit whose every row went with the pre-mix was dropped in
        silence, because it never reached prune_amount_limits. It is named
        the way every other dropped limit is."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_premix("Wet blend", "weighed")
        opt.set_premix_parts("Wet blend", [{'name': "Water", 'share': 60},
                                           {'name': "Oil", 'share': 40}])
        opt.add_ingredient("Water", 20, 40)
        opt.add_ingredient("Oil", 2, 8)
        opt.add_quantity_constraint(["Water", "Oil"], max_val=45,
                                    source="premix:Wet blend")
        removed = opt.set_premix_mode("Wet blend", "portioned")
        # Named for the pre-mix, because the pre-mix is still there — it is
        # the rows underneath it that went.
        assert opt.limit_removed_messages(removed) == [
            ("warning", wording.quantity_limit_removed_missing(
                "Wet blend",
                wording.no_longer_ingredients("Water and Oil", True)))]
        assert opt.quantity_constraints == []
        # remove_ingredient owes the same line, through the same tail.
        opt.add_quantity_constraint(["Sugar"], max_val=4)
        removed = opt.remove_ingredient("Sugar", force=True)
        assert opt.limit_removed_messages(removed) == [
            ("warning", wording.quantity_limit_removed_missing(
                "Sugar", wording.no_longer_ingredients("Sugar", False)))]

    def test_a_part_cannot_be_in_two_weighed_premixes(self, tmp_path,
                                                      monkeypatch):
        """Weighed, a part IS a row. In two weighed pre-mixes it is one row
        standing for two lots of mass, and every roll-up counts it twice.
        Portioned, the part is no row at all, so sharing it is fine."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_premix("Wet blend", "weighed")
        opt.set_premix_parts("Wet blend", [{'name': "Water", 'share': 60},
                                           {'name': "Oil", 'share': 40}])
        opt.add_premix("Fry blend", "weighed")
        with pytest.raises(ValueError) as caught:
            opt.set_premix_parts("Fry blend", [{'name': "Oil", 'share': 100}])
        assert str(caught.value) == wording.part_in_two_weighed_premixes(
            "Oil", "Wet blend", "Fry blend")
        assert opt.premix_parts("Fry blend") == []
        # Shared between two PORTIONED pre-mixes is still allowed, and so is
        # sharing with a portioned one from a weighed one.
        opt.set_premix_mode("Fry blend", "portioned")
        opt.set_premix_parts("Fry blend", [{'name': "Oil", 'share': 100}])
        assert opt.premix_of("Oil") == ["Wet blend", "Fry blend"]


# ------------------------------------------------------------------ #
#  0.7.0 wave 3, task 3 — the pre-mix on the grid
# ------------------------------------------------------------------ #

def _part_row(name, share=None, low=None, high=None, unit="g",
              vendor="", sku=""):
    row = {wording.PART_LABEL: name, wording.UNIT_LABEL: unit,
           wording.VENDOR_LABEL: vendor, wording.SKU_LABEL: sku}
    if share is not None:
        row[wording.PREMIX_SHARE_LABEL] = share
    if low is not None:
        row[wording.LOWEST_LABEL] = low
    if high is not None:
        row[wording.HIGHEST_LABEL] = high
    return row


class TestThePreMixGrid:
    """The choice sits on the ingredients grid — one `Made as` column right
    after Type — and a pre-mix's parts are typed in a fold of their own
    underneath it, in the columns the way it is made actually needs."""

    def _opt(self, tmp_path, monkeypatch, name="premix_grid"):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer(name, robust=False)
        opt.set_amount_unit("g")
        opt.add_ingredient("Water", 0, 100)
        opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
        return opt

    def _dry(self, opt, mode="portioned"):
        opt.add_premix("Dry blend", mode)
        opt.set_premix_parts("Dry blend", [
            {'name': "Flour", 'share': 70, 'unit': "g"},
            {'name': "Salt", 'share': 30, 'unit': "g"}])
        return opt

    @pytest.mark.parametrize("mode", ["", wording.PREMIX_MADE_AS_PORTIONED])
    def test_weighed_mode_can_be_cleared_or_switched_without_typing_amounts(
            self, tmp_path, monkeypatch, mode):
        opt = self._dry(self._opt(tmp_path, monkeypatch), mode="weighed")
        opt.add_ingredient("Flour", 5, 20)
        opt.add_ingredient("Salt", 1, 3)
        frame = _edit(opt.ingredient_grid_frame(), 2,
                      **{wording.MADE_AS_LABEL: mode})
        errors, messages = opt.apply_ingredient_grid(frame)
        assert errors == []
        assert opt._by_name()["Dry blend"]['bounds'] == (6.0, 23.0)
        assert set(opt._by_name()) == {"Water", "Dry blend"}
        assert bool(opt.premixes) == bool(mode)
        if mode:
            # Switched to portioned, the pre-mix goes on being the same
            # line: nothing was added.
            assert wording.added("Dry blend") not in _said(messages)
        else:
            # Blanked, it is a row of the list for the first time — weighed,
            # the rows were its PARTS — so the line says what happened: one
            # row arrived and two left. "Dry blend saved." was the one
            # sentence that could not be true of either.
            assert wording.added("Dry blend") in " ".join(_said(messages))
            assert wording.deleted("Flour and Salt") in _said(messages)
            assert wording.saved("Dry blend") not in _said(messages)

    def test_blanking_a_portioned_row_does_not_report_it_as_added(
            self, tmp_path, monkeypatch):
        opt = self._dry(self._opt(tmp_path, monkeypatch))
        errors, messages = opt.apply_ingredient_grid(_edit(
            opt.ingredient_grid_frame(), 2, **{wording.MADE_AS_LABEL: ""}))
        assert errors == []
        assert wording.added("Dry blend") not in _said(messages)

    def test_unchanged_parts_do_not_claim_to_have_been_saved(
            self, tmp_path, monkeypatch):
        opt = self._dry(self._opt(tmp_path, monkeypatch))
        errors, messages = opt.apply_premix_grid(
            "Dry blend", opt.premix_grid_frame("Dry blend"))
        assert errors == []
        assert messages == []

    def test_weighed_parts_count_when_deleting_the_last_visible_varying_row(
            self, tmp_path, monkeypatch):
        opt = self._dry(self._opt(tmp_path, monkeypatch), mode="weighed")
        opt.add_ingredient("Flour", 5, 20)
        errors, _ = opt.apply_ingredient_grid(_drop(opt.ingredient_grid_frame(), 1))
        assert errors == []
        assert "Water" not in opt._by_name()

    # ---- the parts grid, by mode ------------------------------------- #

    def test_a_portioned_grid_takes_shares_and_no_ranges(self, tmp_path,
                                                         monkeypatch):
        opt = self._dry(self._opt(tmp_path, monkeypatch))
        frame = opt.premix_grid_frame("Dry blend")
        assert list(frame.columns) == [
            "_id", wording.PART_LABEL, wording.PREMIX_SHARE_LABEL,
            wording.UNIT_LABEL]
        assert list(frame[wording.PART_LABEL]) == ["Flour", "Salt"]
        assert list(frame[wording.PREMIX_SHARE_LABEL]) == [70.0, 30.0]
        errors, messages = opt.apply_premix_grid("Dry blend", _edit(
            frame, 1, **{wording.PREMIX_SHARE_LABEL: 50.0,
                         wording.VENDOR_LABEL: "Acme"}))
        assert errors == []
        parts = opt.premix_parts("Dry blend")
        assert [p['name'] for p in parts] == ["Flour", "Salt"]
        assert round(parts[0]['share'], 6) == 62.5
        assert parts[0]['vendor'] == "Acme"
        # Portioned, the parts are not rows of the list at all.
        assert [v['name'] for v in opt.variables] == ["Water", "Dry blend"]

    def test_a_weighed_grid_takes_ranges_and_no_shares(self, tmp_path,
                                                       monkeypatch):
        opt = self._dry(self._opt(tmp_path, monkeypatch), mode="weighed")
        frame = opt.premix_grid_frame("Dry blend")
        assert list(frame.columns) == [
            "_id", wording.PART_LABEL, wording.LOWEST_LABEL,
            wording.HIGHEST_LABEL, wording.UNIT_LABEL]
        assert wording.PREMIX_SHARE_LABEL not in frame.columns
        errors, messages = opt.apply_premix_grid("Dry blend", _edit(
            frame, 1, **{wording.LOWEST_LABEL: "5",
                         wording.HIGHEST_LABEL: "15"}))
        assert errors == []
        assert opt._by_name()["Flour"]['bounds'] == (5.0, 15.0)
        # Lowest above Highest is refused at its own row, and nothing moves.
        errors, _ = opt.apply_premix_grid("Dry blend", _edit(
            opt.premix_grid_frame("Dry blend"), 2,
            **{wording.LOWEST_LABEL: "9", wording.HIGHEST_LABEL: "1"}))
        assert errors == [(2, wording.LOWEST_ABOVE_HIGHEST_ERROR)]
        assert opt._by_name()["Flour"]['bounds'] == (5.0, 15.0)

    def test_parts_are_rebalanced_and_the_caption_says_so(self, tmp_path,
                                                          monkeypatch):
        opt = self._dry(self._opt(tmp_path, monkeypatch))
        frame = _edit(opt.premix_grid_frame("Dry blend"), 1,
                      **{wording.PREMIX_SHARE_LABEL: 10.0})
        errors, messages = opt.apply_premix_grid("Dry blend", frame)
        assert errors == []
        assert any(m.startswith(wording.PREMIX_SHARE_LABEL)
                   for m in _said(messages))
        assert round(sum(p['share'] for p in opt.premix_parts("Dry blend")),
                     6) == 100.0
        # A column of nothing but zeros is one refusal for the grid.
        zeroed = opt.premix_grid_frame("Dry blend")
        for row in (1, 2):
            zeroed.loc[row, wording.PREMIX_SHARE_LABEL] = 0.0
        errors, _ = opt.apply_premix_grid("Dry blend", zeroed)
        assert errors == [(None, wording.PARTS_ADD_TO_NOTHING)]

    def test_a_weighed_premix_row_reads_sum_of_its_parts(self, tmp_path,
                                                         monkeypatch):
        opt = self._dry(self._opt(tmp_path, monkeypatch), mode="weighed")
        frame = opt.ingredient_grid_frame()
        assert list(frame.columns)[:4] == [
            "_id", wording.NAME_LABEL, wording.TYPE_LABEL,
            wording.MADE_AS_LABEL]
        # One line per pre-mix, never a line per part: collapsed, the grid
        # is the project as the bench talks about it.
        assert list(frame[wording.NAME_LABEL]) == ["Water", "Dry blend"]
        line = frame.loc[2]
        assert line[wording.MADE_AS_LABEL] == wording.PREMIX_MADE_AS_WEIGHED
        assert line[wording.LOWEST_LABEL] == ""
        assert line[wording.HIGHEST_LABEL] == ""
        # And the word in those cells is not a refusal: a save that touches
        # nothing else leaves the pre-mix exactly as it was.
        errors, _ = opt.apply_ingredient_grid(frame)
        assert errors == []
        assert opt.premixes["Dry blend"]['mode'] == "weighed"
        assert [v['name'] for v in opt.variables] == ["Water", "Flour", "Salt"]

    def test_a_taken_part_name_is_refused_at_its_row(self, tmp_path,
                                                     monkeypatch):
        opt = self._dry(self._opt(tmp_path, monkeypatch))
        # Portioned, a part may be an ingredient the project already has:
        # the loose Water row stays, and the part is a quantity inside the
        # pre-mix. It is the pre-mix's OWN name that is refused here.
        frame = _edit(opt.premix_grid_frame("Dry blend"), 2,
                      **{wording.PART_LABEL: "Water"})
        errors, _ = opt.apply_premix_grid("Dry blend", frame)
        assert errors == []
        assert "Water" in opt._by_name()
        assert [p['name'] for p in opt.premix_parts("Dry blend")] == [
            "Flour", "Water"]
        frame = _edit(opt.premix_grid_frame("Dry blend"), 2,
                      **{wording.PART_LABEL: "Dry blend"})
        errors, _ = opt.apply_premix_grid("Dry blend", frame)
        assert errors == [(2, wording.PART_IS_ITS_OWN_PREMIX)]
        # A portioned part inherits the preparation unit; percentages have no unit cell.
        errors, _ = opt.apply_premix_grid("Dry blend", _edit(
            opt.premix_grid_frame("Dry blend"), 1,
            **{wording.UNIT_LABEL: ""}))
        assert errors == []
        assert opt.premix_parts("Dry blend")[0]["unit"] == "g"
        assert [p['name'] for p in opt.premix_parts("Dry blend")] == [
            "Flour", "Water"]

    def test_the_last_part_cannot_be_deleted(self, tmp_path, monkeypatch):
        opt = self._dry(self._opt(tmp_path, monkeypatch))
        one = _drop(opt.premix_grid_frame("Dry blend"), 2)
        errors, _ = opt.apply_premix_grid("Dry blend", one)
        assert errors == []
        assert [p['name'] for p in opt.premix_parts("Dry blend")] == ["Flour"]
        empty = _drop(opt.premix_grid_frame("Dry blend"), 1)
        assert opt.premix_grid_deletions("Dry blend", empty) == ["Flour"]
        errors, _ = opt.apply_premix_grid("Dry blend", empty)
        assert errors == [(None, wording.PREMIX_NEEDS_A_PART)]
        assert [p['name'] for p in opt.premix_parts("Dry blend")] == ["Flour"]

    def test_deleting_a_used_part_confirms_first(self, tmp_path, monkeypatch):
        opt = self._dry(self._opt(tmp_path, monkeypatch), mode="weighed")
        opt.add_ingredient("Flour", 0, 50)
        opt.add_ingredient("Salt", 0, 5)
        opt.tell({"Water": 50.0, "Flour": 40.0, "Salt": 2.0}, {"Taste": 7.0})
        frame = _drop(opt.premix_grid_frame("Dry blend"), 1)
        assert opt.premix_grid_deletions("Dry blend", frame) == ["Flour"]
        errors, _ = opt.apply_premix_grid("Dry blend", frame)
        assert len(errors) == 1 and "Flour" in errors[0][1]
        assert [p['name'] for p in opt.premix_parts("Dry blend")] == [
            "Flour", "Salt"]
        errors, _ = opt.apply_premix_grid("Dry blend", frame,
                                          force={"Flour"})
        assert errors == []
        assert [p['name'] for p in opt.premix_parts("Dry blend")] == ["Salt"]
        assert "Flour" not in opt._by_name()

    # ---- Made as, on the row ----------------------------------------- #

    def test_made_as_on_a_new_row_makes_a_premix(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        frame = _add(opt.ingredient_grid_frame(), **_ing_row(
            "Dry blend", low="10", high="30",
            **{wording.MADE_AS_LABEL: wording.PREMIX_MADE_AS_PORTIONED}))
        errors, messages = opt.apply_ingredient_grid(frame)
        assert errors == []
        assert opt.premixes["Dry blend"]['mode'] == "portioned"
        # The row it arrives as is the row the reader typed, not a row
        # fixed at nothing.
        assert opt._by_name()["Dry blend"]['bounds'] == (10.0, 30.0)
        # The consequence is said once, at the choice.
        assert opt.premix_consequence("Dry blend") == ""
        assert not any("make-up stays" in line for line in _said(messages))
        # Both cells are still asked for, exactly as a new ingredient's are.
        opt2 = self._opt(tmp_path, monkeypatch, name="premix_grid_2")
        blank = _add(opt2.ingredient_grid_frame(), **{
            wording.NAME_LABEL: "Fat phase",
            wording.TYPE_LABEL: wording.KIND_INGREDIENT,
            wording.UNIT_LABEL: "g",
            wording.MADE_AS_LABEL: wording.PREMIX_MADE_AS_PORTIONED})
        errors, _ = opt2.apply_ingredient_grid(blank)
        assert errors == [(2, wording.NUMBER_REQUIRED_ERROR)]
        assert opt2.premixes == {}

    def test_blanking_made_as_takes_the_premix_away(self, tmp_path,
                                                    monkeypatch):
        opt = self._dry(self._opt(tmp_path, monkeypatch))
        frame = _edit(opt.ingredient_grid_frame(), 2,
                      **{wording.MADE_AS_LABEL: ""})
        errors, messages = opt.apply_ingredient_grid(frame)
        assert errors == []
        assert opt.premixes == {}
        # The row is handed back to the ordinary list, with the amounts the
        # grid was showing.
        assert [v['name'] for v in opt.variables] == ["Water", "Dry blend"]
        assert opt.premix_of("Flour") == []

    def test_switching_made_as_on_the_row_puts_the_round_at_risk(
            self, tmp_path, monkeypatch):
        opt = self._dry(self._opt(tmp_path, monkeypatch))
        opt.add_ingredient("Dry blend", 5, 20)
        opt.set_pending_batch([{"Water": 50.0, "Dry blend": 10.0}])
        frame = _edit(opt.ingredient_grid_frame(), 2,
                      **{wording.MADE_AS_LABEL: wording.PREMIX_MADE_AS_WEIGHED})
        assert opt.ingredient_grid_retires_round(frame) == opt.pending_batch_no
        errors, _ = opt.apply_ingredient_grid(frame)
        assert errors == []
        assert opt.premixes["Dry blend"]['mode'] == "weighed"
        assert opt.pending_batch_no is None

    def test_a_switch_that_would_leave_nothing_to_vary_is_refused_at_the_row(
            self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer("premix_grid_last", robust=False)
        opt.set_amount_unit("g")
        opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
        opt.add_premix("Dry blend", "portioned")
        opt.set_premix_parts("Dry blend", [{'name': "Flour", 'share': 100,
                                            'unit': "g"}])
        opt.add_ingredient("Dry blend", 5, 20)
        frame = _edit(opt.ingredient_grid_frame(), 1,
                      **{wording.MADE_AS_LABEL: wording.PREMIX_MADE_AS_WEIGHED})
        errors, _ = opt.apply_ingredient_grid(frame)
        assert errors == [(1, wording.LAST_VARYING_ROW_ERROR)]
        assert opt.premixes["Dry blend"]['mode'] == "portioned"


class TestPreMixRollUpsAndLimits:
    """Task 4: an ingredient property rolls up through a portioned pre-mix's
    parts by share, the round owes a shopping total across ingredients and
    parts alike, and a limit on a weighed group reads in the pre-mix's own
    name, in wave 2's unit words when it is a percent."""

    def _opt(self, tmp_path, monkeypatch, name="premix_rollups"):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer(name, robust=False)
        opt.set_amount_unit("g")
        opt.add_ingredient("Sugar", 1, 5)
        opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
        return opt

    @staticmethod
    def _parts(*pairs):
        return [{'name': name, 'share': share} for name, share in pairs]

    def _dry_blend(self, opt, mode="portioned"):
        opt.add_premix("Dry blend", mode)
        opt.set_premix_parts("Dry blend", self._parts(("Flour", 70),
                                                      ("Salt", 30)))
        return opt

    # ---- the roll-up ---------------------------------------------------- #

    def test_a_portioned_premix_rolls_its_parts_properties_up_by_share(
            self, tmp_path, monkeypatch):
        opt = self._dry_blend(self._opt(tmp_path, monkeypatch))
        opt.add_property("Fat per 100 g")
        opt.set_property_value("Flour", "Fat per 100 g", 10.0)
        opt.set_property_value("Salt", "Fat per 100 g", 2.0)
        # 0.7 * 10 + 0.3 * 2 = 7.6 — the pre-mix's own row holds no figure
        # of its own; this IS the figure.
        assert opt.property_value("Dry blend", "Fat per 100 g") == \
            pytest.approx(7.6)
        # property_per_100 needs no case of its own: the pre-mix's row is
        # one ingredient, at whatever amount the recipe gives it.
        assert opt.property_per_100({"Dry blend": 50.0}, "Fat per 100 g") == \
            pytest.approx(7.6)

    def test_a_part_with_no_figure_counts_as_zero_and_is_named(
            self, tmp_path, monkeypatch):
        opt = self._dry_blend(self._opt(tmp_path, monkeypatch))
        opt.add_property("Fat per 100 g")
        opt.set_property_value("Sugar", "Fat per 100 g", 0.0)
        opt.set_property_value("Flour", "Fat per 100 g", 10.0)
        # Salt has no figure at all — it counts as 0 in the average, and is
        # named itself, not the pre-mix row that has no figure to be
        # missing.
        assert opt.property_value("Dry blend", "Fat per 100 g") == \
            pytest.approx(7.0)
        assert opt.ingredients_without_property("Fat per 100 g") == ["Salt"]

    def test_the_roll_up_uses_the_rounds_version_not_todays_parts(
            self, tmp_path, monkeypatch):
        opt = self._dry_blend(self._opt(tmp_path, monkeypatch))
        opt.add_ingredient("Dry blend", 5, 15)
        opt.add_property("Fat per 100 g")
        opt.set_property_value("Flour", "Fat per 100 g", 10.0)
        opt.set_property_value("Salt", "Fat per 100 g", 0.0)
        opt.ask(1)
        assert opt.pending_batch_no is not None
        # The make-up moves the same afternoon the round is on the bench —
        # the round keeps what it was generated with (70 / 30 == 7.0), not
        # today's parts (50 / 50 == 5.0).
        opt.set_premix_parts("Dry blend", self._parts(("Flour", 50),
                                                      ("Salt", 50)))
        assert opt.property_value("Dry blend", "Fat per 100 g") == \
            pytest.approx(7.0)
        assert [p['share'] for p in opt.premix_parts("Dry blend")] == \
            [50.0, 50.0]
        # No round open: today's make-up answers.
        opt.set_pending_batch(None)
        assert opt.property_value("Dry blend", "Fat per 100 g") == \
            pytest.approx(5.0)

    # ---- the shopping total ---------------------------------------------- #

    def test_water_in_two_premixes_is_added_once_across_them(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_premix("Dry blend", "portioned")
        opt.set_premix_parts("Dry blend", self._parts(("Water", 50),
                                                      ("Flour", 50)))
        opt.add_premix("Wet blend", "portioned")
        opt.set_premix_parts("Wet blend", self._parts(("Water", 100)))
        opt.add_ingredient("Dry blend", 50, 50)
        opt.add_ingredient("Wet blend", 20, 20)
        opt.set_pending_batch([{"Sugar": 2.0, "Dry blend": 50.0,
                                "Wet blend": 20.0}])
        totals = dict(opt.round_shopping_totals())
        # Half of Dry blend's 50 g, plus all of Wet blend's 20 g: one number
        # for Water, not two.
        assert totals["Water"] == pytest.approx(45.0)
        assert list(totals) == ["Sugar", "Water", "Flour"]

    def test_the_shopping_total_covers_ingredients_and_parts(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_premix("Dry blend", "portioned")
        opt.set_premix_parts("Dry blend", self._parts(("Flour", 70),
                                                      ("Salt", 30)))
        opt.add_premix("Wet blend", "weighed")
        opt.set_premix_parts("Wet blend", self._parts(("Water", 60),
                                                      ("Oil", 40)))
        opt.add_ingredient("Dry blend", 40, 40)
        opt.add_ingredient("Water", 30, 30)
        opt.add_ingredient("Oil", 20, 20)
        opt.set_pending_batch([{"Sugar": 2.0, "Dry blend": 40.0,
                                "Water": 18.0, "Oil": 12.0}])
        totals = dict(opt.round_shopping_totals())
        assert totals["Sugar"] == pytest.approx(2.0)
        # Portioned: shared out by share, under the PARTS' names.
        assert totals["Flour"] == pytest.approx(28.0)
        assert totals["Salt"] == pytest.approx(12.0)
        # Weighed: the parts are already rows, added in as they are.
        assert totals["Water"] == pytest.approx(18.0)
        assert totals["Oil"] == pytest.approx(12.0)
        # Nobody weighs out "Dry blend" at the shop.
        assert "Dry blend" not in totals
        assert "Wet blend" not in totals

    # ---- the group limit --------------------------------------------------- #

    def test_a_limit_on_a_weighed_group_reads_in_the_premix_name(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_premix("Dry blend", "weighed")
        opt.set_premix_parts("Dry blend", self._parts(("Flour", 60),
                                                      ("Salt", 40)))
        opt.add_ingredient("Flour", 20, 80)
        opt.add_ingredient("Salt", 10, 40)
        opt.set_formulation_total(100)
        opt.add_quantity_constraint(["Flour", "Salt"], min_val=30, max_val=40,
                                    percent=True, source="premix:Dry blend")
        qc = opt.quantity_constraints[-1]
        assert opt.limit_label(qc) == "Dry blend"
        assert opt.limit_text(qc) == (
            "Dry blend is 30 to 40 % of the default batch size "
            "(30 to 40 g at the default 100 g)")

    def test_a_group_limit_follows_a_part_added_later(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_premix("Dry blend", "weighed")
        opt.set_premix_parts("Dry blend", self._parts(("Flour", 60),
                                                      ("Salt", 40)))
        opt.add_ingredient("Flour", 20, 40)
        opt.add_ingredient("Salt", 10, 20)
        opt.add_quantity_constraint(["Flour", "Salt"], max_val=45,
                                    source="premix:Dry blend")
        index = len(opt.quantity_constraints) - 1
        opt.set_premix_parts("Dry blend", self._parts(("Flour", 50),
                                                      ("Salt", 30),
                                                      ("Pepper", 20)))
        qc = opt.quantity_constraints[index]
        assert opt.limit_label(qc) == "Dry blend"
        assert set(qc['ingredients']) == {"Flour", "Salt", "Pepper"}

    # ---- the properties grid --------------------------------------------- #

    def test_the_properties_grid_lists_parts_not_portioned_premix_rows(
            self, tmp_path, monkeypatch):
        opt = self._dry_blend(self._opt(tmp_path, monkeypatch))
        opt.add_ingredient("Dry blend", 5, 15)
        opt.add_premix("Wet blend", "weighed")
        opt.set_premix_parts("Wet blend", self._parts(("Water", 60),
                                                      ("Oil", 40)))
        opt.add_ingredient("Water", 20, 40)
        opt.add_ingredient("Oil", 5, 15)
        names = opt.property_grid_names()
        assert names == ["Sugar", "Flour", "Salt", "Water", "Oil"]
        assert "Dry blend" not in names
        frame = opt.property_grid_frame()
        assert list(frame[wording.PROPERTIES_ROW_COLUMN]) == names
        # A figure is set against the PART, not the pre-mix row.
        opt.add_property("Fat per 100 g")
        opt.set_property_value("Flour", "Fat per 100 g", 10.0)
        with pytest.raises(ValueError, match="No ingredient named Dry blend"):
            opt.set_property_value("Dry blend", "Fat per 100 g", 1.0)


class TestPreMixReviewFixes:
    _opt = TestPreMixRollUpsAndLimits._opt
    _dry_blend = TestPreMixRollUpsAndLimits._dry_blend
    _parts = staticmethod(TestPreMixRollUpsAndLimits._parts)

    def test_regeneration_calculates_and_snapshots_the_same_makeup(self, tmp_path, monkeypatch):
        opt = self._dry_blend(self._opt(tmp_path, monkeypatch))
        opt.add_ingredient("Dry blend", 5, 15)
        opt.add_property("Fat")
        opt.set_property_value("Flour", "Fat", 10)
        opt.ask(1)
        opt.set_premix_parts("Dry blend", self._parts(("Flour", 50), ("Salt", 50)))
        during = []
        def generate(*args):
            during.append(opt.property_value("Dry blend", "Fat"))
            return [{"Sugar": 2, "Dry blend": 10}]
        monkeypatch.setattr(opt, '_ask_cold_start', generate)
        opt.ask(1)
        assert during == [5.0]
        assert opt.property_value("Dry blend", "Fat") == 5.0
        before = opt.export_json()
        def fail(*args):
            raise ValueError("generation failed")
        monkeypatch.setattr(opt, '_ask_cold_start', fail)
        with pytest.raises(ValueError, match="generation failed"):
            opt.ask(1)
        assert opt.export_json() == before
        assert not hasattr(opt, '_generation_premixes')

    def test_a_chosen_group_limit_follows_its_new_parts(self, tmp_path, monkeypatch):
        opt = self._dry_blend(self._opt(tmp_path, monkeypatch), mode="weighed")
        assert "Dry blend" in opt.quantity_limit_choices()
        opt.add_chosen_quantity_constraint(["Dry blend"], max_val=45)
        opt.set_premix_parts("Dry blend", self._parts(("Flour", 50), ("Salt", 30), ("Pepper", 20)))
        assert opt.quantity_constraints[-1]['ingredients'] == ["Flour", "Salt", "Pepper"]
        assert opt.limit_label(opt.quantity_constraints[-1]) == "Dry blend"
        with pytest.raises(ValueError, match="Choose a pre-mix weighed into each formulation on its own"):
            opt.add_chosen_quantity_constraint(["Dry blend", "Flour"], max_val=45)


class TestTheCodexFixWave:
    """The minors the whole-branch review left open, each one a sentence or
    a list the reader sees."""

    def _opt(self, tmp_path, monkeypatch, name="codexfix"):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer(name, robust=False)
        opt.set_amount_unit("g")
        return opt

    def _weighed(self, opt):
        opt.add_ingredient("Water", 0, 50)
        opt.add_premix("Fat phase", "weighed")
        opt.set_premix_parts("Fat phase", [
            {'name': "Coconut oil", 'share': 0, 'unit': "g",
             'low': 0, 'high': 15},
            {'name': "Sunflower oil", 'share': 0, 'unit': "g",
             'low': 0, 'high': 10}])
        return opt

    def test_a_weighed_premix_sits_above_its_own_parts_in_the_picker(
            self, tmp_path, monkeypatch):
        """M9: appended at the end, `Fat phase` sat BELOW Coconut oil and
        Sunflower oil, so the one list the reader picks from was neither the
        grid's rows nor anything else they had seen."""
        opt = self._weighed(self._opt(tmp_path, monkeypatch))
        assert opt.quantity_limit_choices() == [
            "Water", "Fat phase", "Coconut oil", "Sunflower oil"]

    def test_one_name_collision_is_refused_once(self, tmp_path, monkeypatch):
        """M6: a portioned pre-mix IS a row of the list, so a new row typed
        with its name was refused twice for one keystroke."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_ingredient("Water", 0, 50)
        opt.add_ingredient("Dry blend", 5, 15)
        opt.make_premix("Dry blend", "portioned")
        opt.set_premix_parts("Dry blend", [
            {'name': "Flour", 'share': 100, 'unit': "g"}])
        frame = opt.ingredient_grid_frame()
        row = {c: "" for c in frame.columns}
        row[wording.NAME_LABEL] = "Dry blend"
        row[wording.TYPE_LABEL] = wording.KIND_INGREDIENT
        row[wording.LOWEST_LABEL] = "1"
        row[wording.HIGHEST_LABEL] = "2"
        row[wording.UNIT_LABEL] = "g"
        frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
        errors, _ = opt.apply_ingredient_grid(frame)
        assert len(errors) == 1
        assert errors[0][1] == _name_taken_message("Dry blend", "ingredient")

    def test_a_formulation_page_groups_the_rounds_parts_not_todays(
            self, tmp_path, monkeypatch):
        """M10: a part added after the round was generated printed under the
        group on a sheet for a round it was never in."""
        opt = self._weighed(self._opt(tmp_path, monkeypatch))
        opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
        opt.set_pending_batch([{'Water': 40, 'Coconut oil': 8,
                                'Sunflower oil': 4}])
        # The round was generated with one oil in the group; the make-up
        # holds two today. Every other sheet writer reads the round's own
        # version, and this one read `self.premixes[group]['parts']`.
        opt._snapshot_premixes(opt.pending_batch_no, {"Fat phase": [
            {'name': "Coconut oil", 'share': 0, 'unit': "g",
             'low': 0, 'high': 15}]})
        lines = [(group, var if var in (None, 'total') else var['name'])
                 for group, var in opt._formulation_ingredient_lines()]
        assert ("Fat phase", "Sunflower oil") not in lines
        assert ("Fat phase", "Coconut oil") in lines

    def test_the_round_total_of_a_batch_handed_over_uses_the_rounds_size(
            self, tmp_path, monkeypatch):
        """M13: `total` was only defaulted when the BATCH was left out, so a
        caller handing over the rows and no size got the project's default
        instead of the round's."""
        opt = self._weighed(self._opt(tmp_path, monkeypatch))
        opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
        opt.set_formulation_total(50.0)
        opt.set_pending_batch([{'Water': 40, 'Coconut oil': 8,
                                'Sunflower oil': 4}])
        assert (opt.round_shopping_totals(opt.pending_batch)
                == opt.round_shopping_totals())


class TestTheColdRead:
    """The blockers and the serious findings a food scientist met on a
    first read, with no spec in front of them."""

    def _opt(self, tmp_path, monkeypatch, name="coldread"):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer(name, robust=False)
        opt.set_amount_unit("g")
        opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
        opt.add_ingredient("Water", 30, 70)
        opt.add_ingredient("Dry blend", 20, 40)
        opt.make_premix("Dry blend", "portioned")
        opt.set_premix_parts("Dry blend", [
            {'name': "Pea protein isolate", 'share': 55, 'unit': "g"},
            {'name': "Wheat gluten", 'share': 25, 'unit': "g"},
            {'name': "Potato starch", 'share': 12, 'unit': "g"},
            {'name': "Methylcellulose", 'share': 8, 'unit': "g"}])
        return opt

    # ---- B1 · a mode switch never leaves a row at 0-0 ------------------- #

    def test_breaking_a_premix_up_gives_every_part_its_own_amounts(
            self, tmp_path, monkeypatch):
        """Every part came out at 0.00 to 0.00 and the row's own 20-40 g
        went with it, so a round generated straight afterwards held no
        protein, no gluten and no starch at all — a patty of water."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_premix_mode("Dry blend", "weighed")
        rows = {v['name']: tuple(v['bounds']) for v in opt.variables}
        assert rows["Pea protein isolate"] == (11.0, 22.0)
        assert rows["Wheat gluten"] == (5.0, 10.0)
        assert rows["Potato starch"] == (2.4, 4.8)
        assert rows["Methylcellulose"] == (1.6, 3.2)
        # And the sentence says so in numbers.
        said = opt.premix_consequence("Dry blend")
        assert "Pea protein isolate 11.00 to 22.00 g" in said

    def test_gathering_a_premix_back_up_gives_the_row_its_own_amounts(
            self, tmp_path, monkeypatch):
        """Switching back left the row fixed at 0.00 / 0.00 g: the band
        never came back."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_premix_mode("Dry blend", "weighed")
        opt.set_premix_mode("Dry blend", "portioned")
        assert tuple(opt._by_name()["Dry blend"]['bounds']) == (20.0, 40.0)
        assert [round(p['share'], 2)
                for p in opt.premix_parts("Dry blend")] == [55, 25, 12, 8]

    def test_a_made_as_change_keeps_the_row_where_it_was(self, tmp_path,
                                                        monkeypatch):
        """The whole reading order of the list changed under the reader's
        hand, with nothing said, and stayed changed."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_ingredient("Salt", 1, 3)
        assert [v['name'] for v in opt.variables] == [
            "Water", "Dry blend", "Salt"]
        opt.set_premix_mode("Dry blend", "weighed")
        assert [v['name'] for v in opt.variables] == [
            "Water", "Pea protein isolate", "Wheat gluten", "Potato starch",
            "Methylcellulose", "Salt"]
        opt.set_premix_mode("Dry blend", "portioned")
        assert [v['name'] for v in opt.variables] == [
            "Water", "Dry blend", "Salt"]

    # ---- B2 · a part may be an ingredient you already have -------------- #

    def test_a_portioned_part_may_share_a_row_and_the_totals_add_them_once(
            self, tmp_path, monkeypatch):
        """A binder slurry IS methylcellulose plus part of the water. The
        app made you invent a second name for the same material, and then
        listed both in the round's totals."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_formulation_total(100.0)
        opt.add_ingredient("Binder slurry", 2, 8)
        opt.make_premix("Binder slurry", "portioned")
        errors = None
        opt.set_premix_parts("Binder slurry", [
            {'name': "Methylcellulose", 'share': 30, 'unit': "g"},
            {'name': "Water", 'share': 70, 'unit': "g"}])
        assert errors is None
        # The loose Water row stays, and the part is a quantity inside the
        # pre-mix: two masses, weighed once each.
        assert "Water" in opt._by_name()
        assert [p['name'] for p in opt.premix_parts("Binder slurry")] == \
            ["Methylcellulose", "Water"]
        opt.set_pending_batch([{'Dry blend': 30, 'Binder slurry': 10,
                                'Water': 60}])
        totals = dict(opt.round_shopping_totals())
        # 60 g of its own plus 7 g inside the slurry, under the one name.
        assert totals["Water"] == pytest.approx(67.0)
        # 2.40 g inside the dry blend plus 3.00 g inside the slurry.
        assert totals["Methylcellulose"] == pytest.approx(5.4)

    def test_letting_go_of_an_adopted_part_leaves_its_row_behind(
            self, tmp_path, monkeypatch):
        """The pre-mix did not put that row there, so it is not the
        pre-mix's to take away."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.add_premix("Fat phase", "weighed")
        opt.set_premix_parts("Fat phase", [
            {'name': "Water", 'share': 0, 'unit': "g"}])
        assert "Water" in opt._by_name()
        opt.remove_premix("Fat phase")
        assert tuple(opt._by_name()["Water"]['bounds']) == (30.0, 70.0)

    # ---- S1 / S2 · the allowed amounts scale with the round ------------- #

    def test_a_bigger_round_of_the_same_formula_is_not_a_mistake(
            self, tmp_path, monkeypatch):
        """`At 250 g, 5 of 5 ingredients go past the amounts you allowed`
        was printed on screen, on every formulation page and on the Round
        sheet for ordinary bench work."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_formulation_total(100.0)
        opt.set_pending_batch([{'Water': 60, 'Dry blend': 40}])
        opt.scale_round(250.0)
        assert opt.open_round_size() == 250.0
        # The one line it owes is the plain statement that the box moved
        # the numbers — not five ingredients called mistakes.
        assert opt.scaled_cautions(opt.pending_batch, 250.0, sized=True) == [
            opt.scaled_amounts_note(opt.pending_batch, 250.0, sized=True)]
        assert opt.scaled_caution(opt.pending_batch, 250.0, sized=True) == ""
        assert opt.amount_scale(250.0) == 2.5
        # An amount that is genuinely out of proportion still says so.
        assert opt.bounds_caution("Dry blend", 120.0, 2.5)
        assert opt.bounds_caution("Dry blend", 75.0, 2.5) == ""
        # ...and the last bit of a float is not one. 60 g and 40 g both
        # scale to exact floats, so the case that matters is a row fixed
        # where the arithmetic does not land: 2.2 g per 100 g is 5.5 g at
        # 250 g, and the way there is not.
        opt.add_ingredient("Seasoning", 2.2, 2.2)
        scaled = 5.499999999999999      # what scale_round(250) reaches
        assert scaled != 2.2 * 2.5
        assert opt.bounds_caution("Seasoning", scaled, 2.5) == ""

    # ---- S3 · the cell that cannot be typed in -------------------------- #

    def test_typing_over_sum_of_its_parts_is_answered(self, tmp_path,
                                                      monkeypatch):
        """It lit Save, saved nothing, said nothing and the cell reverted."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_premix_mode("Dry blend", "weighed")
        frame = opt.ingredient_grid_frame()
        row = frame.index[frame[wording.NAME_LABEL] == "Dry blend"][0]
        frame.loc[row, wording.HIGHEST_LABEL] = "5"
        errors, _ = opt.apply_ingredient_grid(frame)
        assert errors == [(int(row),
                           wording.premix_amount_is_its_parts("Dry blend"))]
        assert wording.premix_amount_is_its_parts("Dry blend") == (
            "Dry blend's amount is the sum of its parts. Change the parts "
            "in its fold.")

    # ---- S8 · the rebalance names the numbers --------------------------- #

    def test_the_rebalance_says_what_it_wrote(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        _, messages = opt.apply_premix_grid("Dry blend", _edit(
            opt.premix_grid_frame("Dry blend"), 1,
            **{wording.PREMIX_SHARE_LABEL: 20.0}))
        said = [m for _, m in messages
                if m.startswith(wording.PREMIX_SHARE_LABEL)]
        assert said and "Pea protein isolate " in said[0]

    # ---- S11 · the third Made as word ----------------------------------- #

    def test_the_ordinary_row_has_a_word_of_its_own(self, tmp_path,
                                                    monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        frame = opt.ingredient_grid_frame()
        assert frame.loc[1, wording.MADE_AS_LABEL] == \
            wording.PREMIX_MADE_AS_BOUGHT_IN
        # Saving it back changes nothing: `bought in` is not a pre-mix.
        errors, messages = opt.apply_ingredient_grid(frame)
        assert (errors, messages) == ([], [])
        assert "Water" not in opt.premixes


class TestTheMethod:
    """How the formulation is made, in the order the bench does it: one
    project-level text, printed on the Round sheet under the amounts."""

    def _opt(self, tmp_path, monkeypatch, name="method"):
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer(name, robust=False)
        opt.set_amount_unit("g")
        opt.add_ingredient("Water", 0, 100)
        opt.add_ingredient("Flour", 0, 100)
        opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
        opt.set_pending_batch([{'Water': 60, 'Flour': 40}])
        return opt

    def test_the_round_sheet_prints_it_one_line_to_a_row(self, tmp_path,
                                                         monkeypatch):
        """Three formulations that must be made identically except for the
        amounts went to the bench with nothing on the page saying how."""
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_method("Mix 60 s.\n\nForm to 100 g.\nGriddle 3 min a side.")
        assert opt.method_lines() == ["Mix 60 s.", "Form to 100 g.",
                                      "Griddle 3 min a side."]
        sheet = _book(opt.workbook_bytes(opt.pending_batch, 100.0))["Round 1"]
        column = [sheet.cell(r, 1).value for r in range(1, sheet.max_row + 1)]
        at = column.index(wording.METHOD_SHEET_HEADING)
        assert column[at + 1:at + 4] == opt.method_lines()

    def test_a_project_with_no_method_prints_no_heading(self, tmp_path,
                                                        monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        assert opt.method_lines() == []
        sheet = _book(opt.workbook_bytes(opt.pending_batch, 100.0))["Round 1"]
        assert wording.METHOD_SHEET_HEADING not in [
            sheet.cell(r, 1).value for r in range(1, sheet.max_row + 1)]

    def test_it_survives_a_save_and_a_file_without_one_still_loads(
            self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_method("  Mix 60 s.  ")
        assert FoodOptimizer("method").method == "Mix 60 s."
        state = opt.export_json()
        assert state['method'] == "Mix 60 s."
        assert FoodOptimizer.validate_state(state)['version'] == \
            FoodOptimizer.CLASS_VERSION
        # A file written before the box existed says nothing about it.
        del state['method']
        assert FoodOptimizer.validate_state(state)['version'] == \
            FoodOptimizer.CLASS_VERSION
        monkeypatch.chdir(tmp_path)
        other = FoodOptimizer("method_back", robust=False)
        other.import_json(state)
        assert other.method == ""
        # ...and anything that is not text is a damaged file.
        state['method'] = 7
        with pytest.raises(ValueError):
            FoodOptimizer.validate_state(state)

    def test_the_make_quantity_is_makeable(self, tmp_path, monkeypatch):
        """`make 7.50 g` of five powders cannot be blended to any
        homogeneity nor portioned out of without the salt segregating to
        the bottom, and `make 85.71 g` asks for a blend dispensed with
        nothing left in the bowl."""
        opt = self._opt(tmp_path, monkeypatch)
        assert opt.premix_make_quantity(6.60) == 100.0
        assert opt.premix_make_quantity(85.71) == 100.0
        assert opt.premix_make_quantity(160.0) == 180.0
        assert opt.premix_make_quantity(253.25) == 280.0
