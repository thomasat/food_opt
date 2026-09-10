"""Tests for the FoodOptimizer core engine."""

import io
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
    assert list(df.columns) == ["Formulation", "Water", "Temp", "Total"]
    assert df["Total"].iloc[0] == 10.0   # the process setting is not an amount


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
    assert df["Water"].iloc[0] == 11.88
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
        from food_bo import goal_line
        assert goal_line({"goal": "target", "target": 6, "unit": "N"}) == "target 6 N"
        assert goal_line({"goal": "target", "target": 7, "unit": "/10"}) == "target 7/10"
        assert goal_line({"goal": "min", "unit": "N"}) == "lower is better"
        assert goal_line({"goal": "max", "unit": ""}) == "higher is better"

    def test_amount_unit_and_measurement_unit_persist(self, tmp_path, monkeypatch):
        self._opt(tmp_path, monkeypatch)
        reloaded = FoodOptimizer("units")
        assert reloaded.amount_unit == "g"
        assert reloaded.objectives[0]["unit"] == "N"

    def test_measurements_are_ordered_by_importance(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        assert [o["name"] for o in opt.measurements_by_importance()] == ["Firmness", "Juiciness"]
        assert opt.importance_share("Firmness") == pytest.approx(0.6)

    def test_score_function_line_is_the_spec_sentence(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        assert opt.score_function_line() == (
            "Overall score = 1.5 × Firmness closeness + 1.0 × Juiciness closeness. "
            "Closeness is 1 on target and falls evenly with distance from it; a "
            "full scale width away scores 0. Every measurement on target scores 2.50."
        )

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
                           "measured": "8 N", "off_by": "2.0 N too high"}
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
        assert rows[0] == {"name": "Juiciness", "goal": "Higher is better",
                           "measured": "8/10", "off_by": "—"}
        assert rows[1] == {"name": "Grittiness", "goal": "Lower is better",
                           "measured": "2/10", "off_by": "—"}

    def test_closeness_details_says_on_target(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.tell({"Pea protein": 10.0, "Methylcellulose": 1.0},
                 {"Firmness": 6.0, "Juiciness": 6.0})
        rows = opt.closeness_details(0)
        assert rows[0]["off_by"] == "On target"
        assert rows[1]["off_by"] == "1.0 too low"

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
        assert list(df["Best"]) == ["★", "", ""]
        assert list(df["Batch"]) == ["1", "1", "1"]        # one type, always
        assert df["Note"].iloc[0] == "best yet"
        assert df["Note"].iloc[2] == "Not made"
        assert df["Overall score"].iloc[2] == ""

    def test_history_frame_batch_column_is_all_strings_on_a_mixed_project(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.tell({"Pea protein": 10.0, "Methylcellulose": 1.0},
                 {"Firmness": 6.0, "Juiciness": 7.0})            # no batch (imported)
        opt.tell({"Pea protein": 12.0, "Methylcellulose": 1.0},
                 {"Firmness": 6.0, "Juiciness": 7.0}, batch_no=1)
        assert set(opt.history_frame()["Batch"]) == {"", "1"}

    def test_history_frame_marks_a_partial_score(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.tell({"Pea protein": 10.0, "Methylcellulose": 1.0}, {"Firmness": 6.0})
        assert opt.history_frame()["Overall score"].iloc[0].endswith("(partial)")

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
        assert sheet["Pea protein"].iloc[0] == 20.0

    def test_batch_csv_has_formulation_numbers_and_blank_measurements(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        opt.set_pending_batch([{"Pea protein": 11.877679824829102,
                                "Methylcellulose": 1.0}])
        df = pd.read_csv(io.StringIO(opt.batch_csv(opt.pending_batch)))
        assert list(df["Formulation"]) == [1]
        assert df["Pea protein"].iloc[0] == 11.88
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

    def test_a_number_outside_the_batch_names_the_batch(self, tmp_path, monkeypatch):
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

    def test_out_of_scale_value(self, tmp_path, monkeypatch):
        opt = self._opt(tmp_path, monkeypatch)
        df = pd.DataFrame({"Formulation": [7], "Hardness": [1.0], "L*": [140.0]})
        with pytest.raises(ValueError, match=r"outside your scale of 0 to 100"):
            opt.parse_batch_results(df, opt.pending_batch)

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
