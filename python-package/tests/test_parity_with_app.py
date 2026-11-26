"""Parity with the app: egbo.Optimizer and food_bo.FoodOptimizer, driven
through the same seeded rounds, must suggest the same formulations."""

import importlib
import os
import sys
from pathlib import Path

import pytest

from egbo import Optimizer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_app_module():
    """food_bo, or a skip; a failure when EGBO_REQUIRE_PARITY=1 (CI)."""
    if os.environ.get("EGBO_REQUIRE_PARITY") == "1":
        try:
            return importlib.import_module("food_bo")
        except ImportError as exc:
            raise RuntimeError(
                f"EGBO_REQUIRE_PARITY=1 but food_bo cannot be imported: {exc}"
            ) from exc
    return pytest.importorskip("food_bo")


food_bo = load_app_module()

SEED = 12345
PROTEIN = {"A": 20.0, "B": 2.0, "C": 0.0}


def toy(recipe):
    """One deterministic tasting result for any suggestion."""
    taste = 10 - (recipe["A"] - 30) ** 2 / 50 - (recipe["B"] - 20) ** 2 / 80
    return {"Taste": max(0.0, min(10.0, taste))}


@pytest.fixture
def pair(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = food_bo.FoodOptimizer("parity")
    app.sobol_seed = SEED
    return app, Optimizer(seed=SEED)


def build(pair, *, quantity_limit=False, total=False, protein=False):
    app, opt = pair
    for o in (app, opt):
        o.add_ingredient("A", 0, 60)
        o.add_ingredient("B", 0, 50)
        o.add_ingredient("C", 0, 40)
        o.add_objective("Taste", 1)
        if quantity_limit:
            o.add_quantity_constraint(["A", "B"], max_val=70)
        if total:
            o.set_formulation_total(100)
    if protein:
        app.add_property("Protein")
        for name, value in PROTEIN.items():
            app.set_property_value(name, "Protein", value)
        opt.set_property_values("Protein", PROTEIN)
        for o in (app, opt):
            o.add_constraint("Protein", min_val=8)
    return app, opt


def assert_same_rows(app_rows, pkg_rows):
    assert len(app_rows) == len(pkg_rows)
    for a, b in zip(app_rows, pkg_rows):
        assert set(a) == set(b)
        for key in a:
            assert b[key] == pytest.approx(a[key], abs=1e-8), key


def ask_both(app, opt, n):
    """Ask both for n; assert the rows and their numbers match."""
    app.set_pending_batch(None)
    app_rows = app.ask(n)
    pkg_rows = opt.ask(n)
    assert_same_rows(app_rows, pkg_rows)
    assert ([r["formulation"] for r in app.pending_batch]
            == [r["formulation"] for r in opt.pending()])
    return app_rows, pkg_rows


def tell_both(app, opt, app_rows, pkg_rows):
    for i, (a, b) in enumerate(zip(app_rows, pkg_rows)):
        app.tell(a, toy(a), formulation_no=app.pending_batch[i]["formulation"])
        opt.tell(b, toy(b))


def round_both(app, opt, n):
    app_rows, pkg_rows = ask_both(app, opt, n)
    tell_both(app, opt, app_rows, pkg_rows)


def six_results(app, opt):
    round_both(app, opt, 3)
    round_both(app, opt, 3)
    assert len(opt.history()) == 6


def test_parity_cold_start(pair):
    app, opt = build(pair)
    ask_both(app, opt, 3)
    ask_both(app, opt, 2)


def test_parity_warm(pair):
    app, opt = build(pair)
    six_results(app, opt)
    ask_both(app, opt, 3)


def test_parity_fixed_variable(pair):
    app, opt = build(pair)
    six_results(app, opt)
    app.add_ingredient("C", 10, 10)
    opt.fix("C", 10)
    ask_both(app, opt, 2)


def test_parity_quantity_limit_and_total(pair):
    app, opt = build(pair, quantity_limit=True, total=True)
    six_results(app, opt)
    ask_both(app, opt, 2)


def test_parity_property_limit(pair):
    app, opt = build(pair, protein=True)
    six_results(app, opt)
    ask_both(app, opt, 2)


def test_parity_regenerate_after_discard(pair):
    """A batch asked again, unrecorded, retires its rows in both."""
    app, opt = build(pair)
    ask_both(app, opt, 3)
    ask_both(app, opt, 3)
    round_both(app, opt, 3)
    round_both(app, opt, 3)
    ask_both(app, opt, 2)


def test_parity_mid_run_variables_and_own_formulation(pair):
    app, opt = build(pair)
    round_both(app, opt, 5)
    for o in (app, opt):
        o.add_ingredient("D", 0, 10)
        o.add_process_parameter("Bake", 8, 14, baseline=10, step=1)
    own = {"A": 31.23, "B": 18.5, "C": 3.0, "D": 1.0, "Bake": 11.0}
    app.tell(own, toy(own))
    opt.tell(own, toy(own))
    for o in (app, opt):
        o.add_objective("Taste", 2, goal="target", target=8)
    ask_both(app, opt, 3)


def build_ab(pair):
    app, opt = pair
    for o in (app, opt):
        o.add_ingredient("A", 0, 60)
        o.add_ingredient("B", 0, 50)
        o.add_objective("Taste", 1)
    return app, opt


def test_parity_mid_run_kept_lowest_with_a_limit_on_it(pair):
    """D added mid-run keeping its Lowest 4: the frame widens to 0 for the
    past rows, but suggestions stay in 4..10 and "D >= 8" means 8."""
    app, opt = build_ab(pair)
    round_both(app, opt, 5)
    for o in (app, opt):
        o.add_ingredient("D", 4, 10, keep_lowest=True)
        o.add_quantity_constraint(["D"], min_val=8)
    _, rows = ask_both(app, opt, 3)
    for r in rows:
        assert 8 - 1e-6 <= r["D"] <= 10 + 1e-6, r


def test_parity_mid_run_kept_lowest_opening_round(pair):
    """The same in an opening Round: Sobol points outside D's own 4..10
    are dropped, as the app drops them."""
    app, opt = build_ab(pair)
    round_both(app, opt, 3)
    for o in (app, opt):
        o.add_ingredient("D", 4, 10, keep_lowest=True)
    _, rows = ask_both(app, opt, 2)
    for r in rows:
        assert 4 - 1e-9 <= r["D"] <= 10 + 1e-9, r


@pytest.mark.parametrize("low, total", [(30.0, None), (70.0, 100)])
def test_parity_thin_limit_fills_the_opening_round(pair, low, total):
    """A + B within a hair (low to low + 0.001) is a slab the Sobol pool
    all but misses; both fill Round 1 from inside the limits instead."""
    app, opt = build(pair)
    for o in (app, opt):
        o.add_quantity_constraint(["A", "B"], min_val=low, max_val=low + 0.001)
        if total is not None:
            o.set_formulation_total(total)
    _, rows = ask_both(app, opt, 5)
    assert len(rows) == 5
    for r in rows:
        assert low - 1e-6 <= r["A"] + r["B"] <= low + 0.001 + 1e-6, r
    # A regenerate draws a different chain, in both.
    ask_both(app, opt, 5)


def test_parity_unreachable_property_limit_opening_round(pair):
    """Protein >= 25 when no ingredient has more than 20: the pool finds
    nothing, and both samplers land on the same point(s) (today the
    all-zero formulation, which meets the limit only vacuously)."""
    app, opt = build(pair)
    app.add_property("Protein")
    for name, value in PROTEIN.items():
        app.set_property_value(name, "Protein", value)
    opt.set_property_values("Protein", PROTEIN)
    for o in (app, opt):
        o.add_constraint("Protein", min_val=25)
    ask_both(app, opt, 2)


@pytest.mark.parametrize("total", [False, True])
def test_parity_ingredient_baseline(pair, total):
    """Sorbate held at 0.8 in every formulation made, then searched at
    0.5..1.2: the past rows are read at 0.8 in both."""
    app, opt = build(pair, total=total)
    round_both(app, opt, 5)
    for o in (app, opt):
        o.add_ingredient("Sorbate", 0.5, 1.2, baseline=0.8)
    ask_both(app, opt, 3)
