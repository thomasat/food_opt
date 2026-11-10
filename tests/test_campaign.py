"""Tests for the campaign-2 adaptive EGBO features: rounds and checks, range
editing, reference samples, the expert report, finalists, and backups."""

import json
import os

import numpy as np
import pytest

from food_bo import DEFAULT_BO_CONFIG, FoodOptimizer

ATTRS = ("Stretch", "Flavor", "Melt")


@pytest.fixture
def opt(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    o = FoodOptimizer(project_name="mozz", robust=False)
    o.add_ingredient("Starch", 0, 20)
    o.add_ingredient("Oil", 0, 25)
    o.add_ingredient("Protein", 0, 15)
    o.add_process_parameter("CookTemp", 70, 95)
    o.add_objective("Stretch", 0.5, goal="max", min_val=0, max_val=10)
    o.add_objective("Flavor", 0.3, goal="max", min_val=0, max_val=10)
    o.add_objective("Melt", 0.2, goal="target", target=6, min_val=0, max_val=10)
    o.set_bo_config(DEFAULT_BO_CONFIG)
    return o


def score(recipe, rng):
    """A smooth synthetic panel with noise."""
    s = recipe.get("Starch", 0) / 20
    p = recipe.get("Protein", 0) / 15
    t = (recipe.get("CookTemp", 80) - 70) / 25
    return {
        "Stretch": float(np.clip(10 * (0.6 * p + 0.4 * t) + rng.normal(0, 0.3), 0, 10)),
        "Flavor": float(np.clip(8 - 6 * (s - 0.4) ** 2 + rng.normal(0, 0.3), 0, 10)),
        "Melt": float(np.clip(4 + 4 * s + rng.normal(0, 0.3), 0, 10)),
    }


def run_round(o, rng, q=5, reference=True):
    recipes = o.ask(n_suggestions=q)
    assert o.pending_batch and o.pending_batch["recipes"] == recipes
    for r in recipes:
        o.tell(r, score(r, rng))
    if reference and o.reference_recipe:
        o.tell(o.reference_recipe, score(o.reference_recipe, rng), role="reference")
    o.clear_pending_batch()
    return recipes


class TestRoundsAndChecks:
    def test_rounds_count_bo_batches(self, opt):
        rng = np.random.default_rng(0)
        assert opt.completed_rounds() == 0
        run_round(opt, rng)
        assert opt.completed_rounds() == 1
        assert {m["round"] for m in opt.meta_history} == {1}

    def test_check_due_after_every_third_round(self, opt):
        rng = np.random.default_rng(1)
        for r in range(1, 4):
            assert not opt.check_due()
            run_round(opt, rng)
        assert opt.completed_rounds() == 3 and opt.check_due()
        text, snap = opt.expert_report()
        opt.record_check("no change", "gaps are noise-level", by="FSA", report=snap)
        assert not opt.check_due()
        assert opt.design_log[-1]["type"] == "check"
        assert opt.design_log[-1]["report"]["after_round"] == 3

    def test_changes_logged_on_and_off_schedule(self, opt):
        rng = np.random.default_rng(2)
        run_round(opt, rng)
        opt.set_variable_bounds("Oil", 0, 10)          # after R1: off schedule
        assert opt.design_log[-1]["on_schedule"] is False
        run_round(opt, rng)
        run_round(opt, rng)
        opt.deactivate_variable("Oil")                 # after R3: check due
        e = opt.design_log[-1]
        assert e["action"] == "deactivate" and e["on_schedule"] is True
        assert e["after_round"] == 3

    def test_design_change_drops_pending_batch(self, opt):
        rng = np.random.default_rng(3)
        run_round(opt, rng)
        opt.ask(n_suggestions=2)
        assert opt.pending_batch
        opt.set_variable_bounds("Starch", 0, 30)
        assert opt.pending_batch is None


class TestRanges:
    def test_widen_and_narrow_keep_history(self, opt):
        rng = np.random.default_rng(4)
        run_round(opt, rng)
        run_round(opt, rng)
        n = len(opt.X_history)
        opt.set_variable_bounds("Starch", 0, 40)       # widen
        opt.set_variable_bounds("Oil", 2, 8)           # narrow
        assert len(opt.X_history) == n
        log = [e for e in opt.design_log if e.get("action") == "set_range"]
        assert log[0]["widened"] and not log[0]["narrowed"]
        assert log[1]["narrowed"]
        recipes = opt.ask(n_suggestions=4)
        for r in recipes:
            assert 0 - 1e-6 <= r["Starch"] <= 40 + 1e-6
            assert 2 - 1e-6 <= r["Oil"] <= 8 + 1e-6

    def test_invalid_ranges_rejected(self, opt):
        with pytest.raises(ValueError):
            opt.set_variable_bounds("Starch", 5, 5)
        with pytest.raises(ValueError):
            opt.set_variable_bounds("Starch", -1, 5)

    def test_pruned_variable_pin_must_stay_in_range(self, opt):
        rng = np.random.default_rng(5)
        run_round(opt, rng)
        opt.deactivate_variable("Oil")                 # pinned at 0
        with pytest.raises(ValueError, match="pinned"):
            opt.set_variable_bounds("Oil", 1, 10)

    def test_range_change_respects_quantity_constraints(self, opt):
        opt.add_quantity_constraint(["Starch", "Oil"], min_val=30)
        with pytest.raises(ValueError, match="unsatisfiable"):
            opt.set_variable_bounds("Oil", 0, 5)
        assert opt._var_by_name("Oil")["bounds"] == (0.0, 25.0)


class TestReferenceAndRoles:
    def test_reference_logged_and_excluded_when_requested(self, opt):
        rng = np.random.default_rng(6)
        opt.set_reference_recipe({"Starch": 8, "Oil": 10, "Protein": 5, "CookTemp": 80})
        recipes = opt.ask(n_suggestions=5)
        for r in recipes:
            opt.tell(r, score(r, rng))
        opt.tell(opt.reference_recipe, score(opt.reference_recipe, rng),
                 role="reference", in_gp=False)
        opt.clear_pending_batch()
        assert opt.meta_history[-1]["role"] == "reference"
        assert opt.meta_history[-1]["round"] == 1
        assert len(opt._training_indices()) == 5
        # A reference sample doesn't count as a round of BO.
        assert opt.completed_rounds() == 1

    def test_finalists_exclude_reference(self, opt):
        rng = np.random.default_rng(7)
        opt.set_reference_recipe({"Starch": 8, "Oil": 10, "Protein": 15, "CookTemp": 95})
        for _ in range(3):
            run_round(opt, rng)
        fin = opt.finalists(3)
        assert len(fin) == 3
        assert all(f["role"] != "reference" for f in fin)
        assert all(f["predicted"] is not None for f in fin)
        preds = [f["predicted"] for f in fin]
        assert preds == sorted(preds, reverse=True)


class TestExpertReport:
    def test_report_contents(self, opt):
        rng = np.random.default_rng(8)
        opt.set_reference_recipe({"Starch": 8, "Oil": 10, "Protein": 5, "CookTemp": 80})
        opt.set_candidate_pool("Transglutaminase 0-1%\nCoconut oil 0-20%")
        for _ in range(3):
            run_round(opt, rng)
        text, snap = opt.expert_report()
        for needle in ("Objective", "Stretch", "target 6", "Where the remaining gap is",
                       "Current design space", "Transglutaminase", "Reference recipe by session",
                       "Round 3", "predicted"):
            assert needle in text, needle
        json.dumps(snap)  # serializable
        assert snap["after_round"] == 3

    def test_report_flags_variable_at_limit(self, opt):
        rng = np.random.default_rng(9)
        for _ in range(3):
            run_round(opt, rng, reference=False)
        # Force the best recipes to sit at Protein's upper bound.
        for i in range(len(opt.recipe_history)):
            opt.recipe_history[i]["Protein"] = 15.0
        opt._reencode_history()
        text, _ = opt.expert_report()
        line = next(l for l in text.splitlines() if l.startswith("| Protein"))
        assert "upper limit" in line


class TestObjectives:
    def test_add_objective_midrun_requires_scores(self, opt):
        rng = np.random.default_rng(10)
        run_round(opt, rng)
        with pytest.raises(ValueError, match="no 'Gumminess' score"):
            opt.add_objective("Gumminess", 0.2, goal="min")
        assert [o["name"] for o in opt.objectives] == list(ATTRS)

    def test_add_objective_recomputes_utilities(self, opt):
        rng = np.random.default_rng(11)
        run_round(opt, rng)
        for i, r in enumerate(opt.results_history):
            r["Gumminess"] = 5.0
        before = list(opt.Y_history)
        opt.add_objective("Gumminess", 0.2, goal="min", min_val=0, max_val=10)
        assert all(abs(a + 0.2 * 0.5 - b) < 1e-9 for a, b in zip(before, opt.Y_history))

    def test_bo_config_locked_after_data(self, opt):
        rng = np.random.default_rng(12)
        run_round(opt, rng)
        with pytest.raises(ValueError, match="locked"):
            opt.set_bo_config({"kernel": "rbf"})
        opt.set_bo_config(DEFAULT_BO_CONFIG)  # same config is fine


class TestProcessParameters:
    def test_remove_process_refused_midrun(self, opt):
        rng = np.random.default_rng(13)
        run_round(opt, rng)
        with pytest.raises(ValueError, match="Deactivate"):
            opt.remove_process_parameter("CookTemp")
        assert any(v["name"] == "CookTemp" for v in opt.variables)


class TestBackup:
    def _campaign(self, opt):
        rng = np.random.default_rng(14)
        opt.set_reference_recipe({"Starch": 8, "Oil": 10, "Protein": 5, "CookTemp": 80})
        opt.set_candidate_pool("Pea protein 0-10%")
        for _ in range(3):
            run_round(opt, rng)
        _, snap = opt.expert_report()
        opt.record_check("widened Starch", "Melt below target", by="FSA", report=snap)
        opt.set_variable_bounds("Starch", 0, 30)
        opt.deactivate_variable("Oil")
        opt.add_ingredient("PeaProtein", 0, 10)
        opt.ask(n_suggestions=5)  # pending, unscored batch
        return opt

    def test_round_trip_restores_everything(self, opt, tmp_path):
        self._campaign(opt)
        blob = json.loads(json.dumps(opt.export_json()))
        fresh = FoodOptimizer(project_name="restored")
        fresh.import_json(blob)
        for attr in ("variables", "objectives", "recipe_history", "results_history",
                     "meta_history", "design_log", "pending_batch", "check_every",
                     "candidate_pool", "reference_recipe", "bo_config"):
            a, b = getattr(opt, attr), getattr(fresh, attr)
            assert json.loads(json.dumps(a, default=list)) == json.loads(json.dumps(b, default=list)), attr
        assert np.allclose(opt.Y_history, fresh.Y_history)
        assert np.allclose(np.array(opt.X_history), np.array(fresh.X_history))
        assert fresh.completed_rounds() == 3
        # Same next suggestion from the same state.
        assert np.allclose(
            [list(r.values()) for r in opt.ask(n_suggestions=2)],
            [list(r.values()) for r in fresh.ask(n_suggestions=2)],
        )

    def test_restore_is_saved_to_disk(self, opt):
        self._campaign(opt)
        blob = opt.export_json()
        blob["project_name"] = "restored2"  # as the app does before restoring
        fresh = FoodOptimizer(project_name="restored2")
        fresh.import_json(blob)
        reopened = FoodOptimizer(project_name="restored2")
        assert len(reopened.X_history) == len(opt.X_history)
        assert reopened.pending_batch is not None

    def test_bad_backup_changes_nothing(self, opt):
        self._campaign(opt)
        before = json.dumps(opt.export_json(), sort_keys=True)
        bad = opt.export_json()
        bad["results_history"] = bad["results_history"][:-2]
        with pytest.raises(ValueError):
            opt.import_json(bad)
        with pytest.raises(ValueError):
            opt.import_json({"hello": 1})
        after = opt.export_json()
        before_d = json.loads(before)
        for k in ("exported_at",):
            before_d.pop(k)
            after.pop(k)
        assert before_d == json.loads(json.dumps(after, sort_keys=True))

    def test_unreadable_project_file_is_set_aside(self, opt):
        self._campaign(opt)
        with open(opt.filename, "wb") as f:
            f.write(b"not a pickle")
        reopened = FoodOptimizer(project_name="mozz")
        assert reopened.load_error and "backup" in reopened.load_error
        assert reopened.X_history == []
        aside = [f for f in os.listdir(".") if f.startswith("mozz_unreadable_")]
        assert len(aside) == 1
        with open(aside[0], "rb") as f:
            assert f.read() == b"not a pickle"

    def test_old_project_without_metadata_loads(self, opt):
        rng = np.random.default_rng(15)
        run_round(opt, rng)
        state = opt.export_json()
        for k in ("meta_history", "design_log", "pending_batch", "check_every",
                  "candidate_pool", "reference_recipe"):
            state.pop(k)
        fresh = FoodOptimizer(project_name="old")
        fresh.import_json(state)
        assert len(fresh.meta_history) == len(fresh.X_history)
        assert fresh.meta_history[0]["role"] == "bo"
        fresh.ask(n_suggestions=2)

    def test_backup_filename_has_round(self, opt):
        rng = np.random.default_rng(16)
        run_round(opt, rng)
        assert opt.backup_filename().startswith("mozz_R01_")


# ------------------------------------------------------------------ #
#  Proxy's plant-based mozzarella setup: grams, water as balance,
#  cook step, sodium and cost limits
# ------------------------------------------------------------------ #

import pandas as pd

MOZZ = {  # name: (min g, max g, start g, sodium mg/g, cost $/g)
    "Coconut oil": (150, 280, 220, 0.0, 0.004),
    "Tapioca starch": (70, 150, 110, 0.0, 0.002),
    "Soy protein isolate": (20, 60, 35, 10.0, 0.008),
    "Potato starch": (10, 40, 25, 0.0, 0.002),
    "Salt": (8, 16, 13, 393.0, 0.0005),
    "Kappa carrageenan": (4, 12, 8, 10.0, 0.02),
    "Soy lecithin": (2, 8, 5, 0.0, 0.01),
    "Guar gum": (2, 6, 4, 0.0, 0.01),
    "Lactic acid": (2, 5, 3.6, 0.0, 0.005),
    "Nutritional yeast": (1, 5, 3, 1.0, 0.02),
}


@pytest.fixture
def mozz(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    o = FoodOptimizer(project_name="proxy_mozz")
    rows = [{"Name": n, "Min": lo, "Max": hi, "Sodium": na, "Cost": c}
            for n, (lo, hi, _, na, c) in MOZZ.items()]
    rows.append({"Name": "Tap water", "Min": 0, "Max": 1002, "Sodium": 0.0, "Cost": 0.0})
    o.load_ingredients_from_csv(pd.DataFrame(rows))
    o.set_balance("Tap water", total=1002, min_amount=450, max_amount=700)
    o.add_process_parameter("Cook temp", 78, 92)
    o.add_process_parameter("Cook time", 6, 12)
    o.add_constraint("sodium", max_val=6000)   # mg per batch
    o.add_constraint("cost", max_val=3.2)      # $ per batch
    for a, goal in (("Stretchy", "target"), ("Rich", "max"), ("Artificial", "min")):
        o.add_objective(a, 1 / 3, goal=goal, target=10 if goal == "target" else None,
                        min_val=0, max_val=15)
    o.set_bo_config(DEFAULT_BO_CONFIG)
    return o


def mozz_score(r, rng):
    oil, starch = r["Coconut oil"] / 280, r["Tapioca starch"] / 150
    t = (r["Cook temp"] - 78) / 14
    return {
        "Stretchy": float(np.clip(15 * (0.5 * starch + 0.3 * t + 0.2 * r["Kappa carrageenan"] / 12)
                                  + rng.normal(0, 0.5), 0, 15)),
        "Rich": float(np.clip(15 * oil + rng.normal(0, 0.5), 0, 15)),
        "Artificial": float(np.clip(3 + 0.6 * r["Nutritional yeast"] + rng.normal(0, 0.5), 0, 15)),
    }


def feasible(o, r):
    water = r["Tap water"]
    total = sum(r[n] for n in MOZZ) + water
    sodium = sum(r[n] * MOZZ[n][3] for n in MOZZ)
    cost = sum(r[n] * MOZZ[n][4] for n in MOZZ)
    return (abs(total - 1002) < 1e-6 and 450 - 1e-6 <= water <= 700 + 1e-6
            and sodium <= 6000 + 1e-6 and cost <= 3.2 + 1e-6
            and all(MOZZ[n][0] - 1e-6 <= r[n] <= MOZZ[n][1] + 1e-6 for n in MOZZ))


class TestMozzSetup:
    def test_water_is_balance_not_variable(self, mozz):
        assert "Tap water" not in [v["name"] for v in mozz.variables]
        assert mozz.ingredient_properties["Tap water"] == {"sodium": 0.0, "cost": 0.0}

    def test_round1_recipes_feasible_and_distinct(self, mozz):
        recipes = mozz.ask(n_suggestions=5)
        assert len(recipes) == 5
        assert all(feasible(mozz, r) for r in recipes)
        oils = {round(r["Coconut oil"], 3) for r in recipes}
        assert len(oils) == 5

    def test_bo_rounds_stay_feasible(self, mozz):
        rng = np.random.default_rng(20)
        for _ in range(3):
            recipes = mozz.ask(n_suggestions=5)
            assert all(feasible(mozz, r) for r in recipes), recipes
            for r in recipes:
                mozz.tell(r, mozz_score(r, rng))
            mozz.clear_pending_batch()
        assert mozz.recipe_history[-1]["Tap water"] > 0
        text, _ = mozz.expert_report()
        assert "Tap water=" in text

    def test_hold_ingredient_at_amount(self, mozz):
        rng = np.random.default_rng(21)
        for r in mozz.ask(n_suggestions=5):
            mozz.tell(r, mozz_score(r, rng))
        mozz.clear_pending_batch()
        mozz.deactivate_variable("Soy lecithin", value=5.0)
        recipes = mozz.ask(n_suggestions=3)
        assert all(abs(r["Soy lecithin"] - 5.0) < 1e-6 for r in recipes)
        assert all(feasible(mozz, r) for r in recipes)

    def test_add_ingredient_with_baseline(self, mozz):
        rng = np.random.default_rng(22)
        for r in mozz.ask(n_suggestions=5):
            mozz.tell(r, mozz_score(r, rng))
        mozz.clear_pending_batch()
        # Potassium sorbate was in every recipe at 0.8 g but not varied.
        mozz.add_ingredient("Potassium sorbate", 0.5, 1.2, baseline=0.8)
        col = [v["name"] for v in mozz.variables].index("Potassium sorbate")
        assert all(abs(x[col] - 0.8) < 1e-9 for x in mozz.X_history)
        with pytest.raises(ValueError, match="must lie within"):
            mozz.add_ingredient("Citric acid", 1, 2, baseline=0.5)
        mozz.ask(n_suggestions=2)

    def test_balance_survives_backup(self, mozz):
        blob = json.loads(json.dumps(mozz.export_json()))
        blob["project_name"] = "proxy_mozz_restored"
        fresh = FoodOptimizer(project_name="proxy_mozz_restored")
        fresh.import_json(blob)
        assert fresh.balance == mozz.balance
        assert all(feasible(fresh, r) for r in fresh.ask(n_suggestions=3))

    def test_round1_with_tight_constraints_uses_feasible_sampling(self, mozz):
        # Oil + tapioca within a 5 g window: random draws almost never satisfy it.
        mozz.add_quantity_constraint(["Coconut oil", "Tapioca starch"], min_val=330, max_val=335)
        recipes = mozz.ask(n_suggestions=5)
        assert len(recipes) == 5
        for r in recipes:
            assert feasible(mozz, r)
            assert 330 - 1e-6 <= r["Coconut oil"] + r["Tapioca starch"] <= 335 + 1e-6
