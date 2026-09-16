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
    _name_taken_message,
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
        with pytest.raises(ValueError, match="Cannot reload ingredients"):
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
    assert list(df.columns[:3]) == ["Best", "Round", "Formulation"]
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
    for col in ["Formulation", "Round", "Recorded", "Overall score",
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
        with pytest.raises(ValueError, match="No ingredient or setting named"):
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
                           match="last ingredient or setting with a range"):
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
    assert len(sample_names) == 8
    assert set(sample_names) <= set(names)
    # The template's headers are the add form's own words — Name, Lowest,
    # Highest, Unit — so a reader filling it in is answering the same four
    # questions the screen asks. data/ingredients.csv is the experiments'
    # list, not a template, and keeps the lowercase headers those scripts
    # read by name; the columns are the same columns either way.
    assert list(sample.columns) == ["Name", "Lowest", "Highest", "Unit",
                                    "Fat per 100 g", "Sodium per 100 g"]
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
        with pytest.raises(ValueError, match="not-scored"):
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
        assert list(df.columns) == ["Best", "Round", "Formulation", "Firmness (N)",
                                    "Juiciness", "Overall score", "Recorded", "Note"]
        assert list(df["Formulation"]) == [2, 1, 3]        # best first, skipped last
        # Best is a star or nothing; the Note column carries "Not scored".
        assert list(df["Best"]) == ["★", "", ""]
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
        for col in ("Formulation", "Round", "Recorded", "Overall score",
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
        assert list(frame.columns) == [
            "_id", wording.NAME_LABEL, wording.TYPE_LABEL,
            wording.LOWEST_LABEL, wording.HIGHEST_LABEL, wording.UNIT_LABEL,
            wording.VENDOR_LABEL, wording.SKU_LABEL]
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
        assert wording.SHARES_REBALANCED_CAPTION in _said(messages, "info")

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
        assert "must be between the range" in errors[0][1]

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
            "Overall score 100.00 of 100.00")

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
                        "ui_results.py", "food_bo.py", "storage.py", "wording.py"]
# The user-facing files that are not Python. They are scanned as plain text,
# except the Swift wrapper, where only its string literals are screen text.
_USER_FACING_TEXT = ["desktop/start_here.txt", "desktop/README.md", "README.md"]
_USER_FACING_SWIFT = "desktop/FoodOptimizerApp.swift"

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
_ALLOWED_SINGLE_WORDS_BY_FILE = {
    "food_bo.py": {"Batch", "batch"},
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
            if any(pattern.search(text) for pattern in _BANNED):
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
        if name == "food_bo.py" and word in {"Batch", "batch"}:
            assert refused == [], (name, word)
        else:
            assert refused == [(name, word)], (name, word)


def test_even_food_bo_may_not_say_batches():
    """The scoped allowance is two exact literals, not the word. A plural is
    not a stored key or a reserved column name, so nothing wants one."""
    assert _single_word_offenders("food_bo.py", ["batches", "Batches"]) == [
        ("food_bo.py", "batches"), ("food_bo.py", "Batches")]


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
        with pytest.raises(ValueError, match="'property_names' section"):
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
        with pytest.raises(ValueError, match="'pending_batch_total' section"):
            FoodOptimizer.validate_state(state)
        state['pending_batch_total'] = None
        state['batch_totals'] = [150.0]
        with pytest.raises(ValueError, match="'batch_totals' section"):
            FoodOptimizer.validate_state(state)
        state['batch_totals'] = {"2": "big"}
        with pytest.raises(ValueError, match="'batch_totals' section"):
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
        with pytest.raises(ValueError, match="'targets_source' section"):
            FoodOptimizer.validate_state(state)
        state['targets_source'] = ["a list"]
        with pytest.raises(ValueError, match="'targets_source' section"):
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


class TestFormulationTotal:
    """One number on tab 1 says how big a formulation is, and every
    suggestion adds up to it. It is stored as a number AND written as the
    limit over every ingredient, because the limit is what the space-filling
    opening and the model already obey."""

    def _sample(self, tmp_path, monkeypatch, name="sample"):
        """The sample project's own eight ingredients: they add up to at
        least 20 g and at most 131 g, which is what makes 100 g reachable and
        150 g not."""
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer(name, robust=False)
        opt.set_amount_unit("g")
        opt.load_ingredients_from_csv(pd.read_csv(_SAMPLE_CSV))
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
            with pytest.raises(ValueError, match="'formulation_total' section"):
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
        monkeypatch.chdir(tmp_path)
        opt = FoodOptimizer(name, robust=False)
        opt.set_amount_unit("g")
        opt.load_ingredients_from_csv(pd.read_csv(_SAMPLE_CSV))
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
        """Not eight ingredients and a limit the user never wrote."""
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
        assert book.sheetnames == ["Round 2", "Formulation 1",
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
        assert rows[2] == ("Ingredient or process setting", "Formulation 1", "%",
                           "Formulation 2", "%", "Formulation 3", "%",
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
                               "Firmness · target 6 N",
                               "Juiciness (/10) · higher is better",
                               "Not scored", "Note",
                               wording.SUMMARY_TICK_NOTE,
                               wording.MADE_BY_FOOTER], labels
        assert wording.SUMMARY_TICK_NOTE == (
            "A ticked Not scored box wins over numbers typed in that column.")
        # The measurement rows open empty; the Not scored row opens holding
        # the box the instruction asks the reader to tick, in the cell the
        # pen can reach.
        for label in ("Firmness · target 6 N",
                      "Juiciness (/10) · higher is better"):
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
        # What this formulation is trying. Nothing to compare it with yet,
        # so the line is the cell alone: "Compared with the allowed amounts:
        # Spread across the allowed amounts" is that sentence twice.
        assert rows[1][0] == wording.SUGGESTION_SPREAD.capitalize()
        # Then the one line that says which cells the sheet will take,
        # naming the Actual column as this page's own header writes it.
        assert rows[2][0] == wording.sheet_write_in_note("Actual (g)")
        # A column to tick as each ingredient goes in, in set-up order, and
        # a column to write what the balance actually said.
        assert rows[4] == ("Tick", "Ingredient", "Amount (g)", "Actual (g)",
                           "%"), rows[4]
        assert [row[1] for row in rows[5:8]] == ["Pea protein", "Water",
                                                 "Salt"]
        assert rows[5][2] == 20.0 and rows[5][4] == 20.0
        assert rows[5][3] is None        # Actual: blank means as printed
        assert (rows[8][1], rows[8][2], rows[8][4]) == ("Total", 100.0, 100.0)
        text = [v for row in rows for v in row if isinstance(v, str)]
        assert wording.SETTINGS_SHEET_HEADING in text, text
        assert "Cook temperature (°C)" in text, text
        assert wording.MEASURED_COLUMN in text, text
        assert "target 6 N" in text, text
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
            for row in sheet.iter_rows():
                for cell in row:
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
        firm = labels.index("Firmness · target 6 N") + 1
        juice = labels.index("Juiciness (/10) · higher is better") + 1
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
        sheet.cell(row=labels.index("Firmness · target 6 N") + 1, column=2,
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
        with pytest.raises(ValueError, match="no sheet called Round 3"):
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
        sheet.cell(row=labels.index("Firmness · target 6 N") + 1, column=2,
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
        sheet.cell(row=labels.index("Firmness · target 6 N") + 1, column=2,
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
            if label == "Firmness":
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

    def test_the_summary_is_read_first_when_both_are_filled_in(
            self, tmp_path, monkeypatch):
        """One sheet has to win, and it is the one the whole batch is
        written on."""
        opt = self._opt(tmp_path, monkeypatch)
        book = openpyxl.load_workbook(self._filled_in(opt))
        sheet = book[wording.formulation_sheet_name(1)]
        for r in range(1, sheet.max_row + 1):
            if sheet.cell(row=r, column=2).value == "Firmness":
                sheet.cell(row=r, column=4, value=9.9)
        out = io.BytesIO()
        book.save(out)
        out.seek(0)
        frame = opt.results_from_workbook(out).frame
        assert frame["Firmness"].iloc[0] == 5.5, frame.to_dict()

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
                         "Hit a target", "0 to 10 N", "Limits",
                         "Default batch size · 100 g",
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
        assert _rows(book["Ingredients"])[0][:4] == ("Name", "Lowest",
                                                     "Highest", "Unit")
        frame = pd.read_excel(io.BytesIO(ingredients_template_workbook(template)))
        opt = FoodOptimizer("from_template")
        opt.set_amount_unit("g")
        opt.load_ingredients_from_csv(frame)
        assert [v['name'] for v in opt.variables] == \
            list(pd.read_csv(template)["Name"])[:1]
        assert opt.unit_of("Pea protein isolate") == "g"


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
        """The screens say 'Firmness (N) · target 6 N'; the sheet said it
        with a comma. One separator, and the upload matches on it."""
        opt = self._opt(tmp_path, monkeypatch)
        sheet = _book(opt.workbook_bytes(opt.pending_batch))["Round 2"]
        assert "Firmness · target 6 N" in _labelled(sheet)

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
        assert title.endswith(" · made to 100 g"), title
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

    def test_the_set_up_sheet_carries_the_share_of_score(self, tmp_path,
                                                         monkeypatch):
        """And nothing else: 0.5.0 makes the share the number the reader
        types, so the sheet no longer prints the importance behind it beside
        it — one fact, in one scale."""
        opt = self._opt(tmp_path, monkeypatch)
        rows = _rows(_book(opt.all_formulations_workbook())["Set up"])
        head = next(r for r in rows if r[0] == wording.MEASUREMENT_COLUMN)
        assert head[4] == wording.SHARE_COLUMN
        assert len([c for c in head if c]) == 5
        line = next(r for r in rows if r[0] == "Firmness")
        assert line[4] == "60 %"

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
            row = self._row_of(sheet, "Firmness · target 6 N")
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
                if row and len(row) > 1 and str(row[1]).strip() == "Firmness":
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
            firm = self._row_of(sheet, "Firmness · target 6 N")
            juice = self._row_of(sheet, "Juiciness (/10) · higher is better")
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
                        "Salt": "fixed at 20.00 g"}

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
        assert state['CLASS_VERSION'] == 11
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
            "one amount. Change the limit, or give it a range.")
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
        the project; a grid of eight ingredients by six properties would
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
        write_in = ("Firmness · target 6 N",
                    "Juiciness (/10) · higher is better",
                    wording.NOT_SCORED_CHECKBOX_SHEET, wording.NOTE)
        expected = {f"{letter}{at[label]}" for label in write_in
                    for letter in ("B", "D")}          # one per formulation
        expected |= {f"F{at[name]}" for name in ("Pea protein (g)",
                                                 "Water (g)")}   # the Lot
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
                     "Firmness", "Juiciness (/10)",     # what to measure
                     wording.NOT_SCORED_CHECKBOX_SHEET, wording.NOTE)}
        # Both note cells: the app reads the printed one back as its own, and
        # a technician who corrects it there is not writing to no effect.
        expected.add(f"C{at[wording.NOTE]}")
        # And the tick boxes: a sheet filled in on a screen has to be
        # tickable on the screen.
        expected |= {f"A{at[name]}" for name in ("Pea protein", "Water")}
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
            summary.cell(row=at["Firmness · target 6 N"], column=column,
                         value=6.0)
            summary.cell(row=at["Juiciness (/10) · higher is better"],
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
        summary.cell(row=4, column=1, value="Firmness · target 6 N")
        summary.cell(row=4, column=2, value=6.0)
        summary.cell(row=5, column=1, value="Juiciness (/10) · higher is better")
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
            with pytest.raises(ValueError, match="'lots' section"):
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
        with pytest.raises(ValueError, match="'lots' section"):
            FoodOptimizer.validate_state(state)
