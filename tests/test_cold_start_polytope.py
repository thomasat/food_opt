"""An opening Round is filled from inside tight limits when the Sobol pool
falls short (stable e2752b7, `_sample_feasible`).

A limit as thin as A + B between 30 and 30.001 is a sliver of the space: of
an 8,192-point pool hardly any lands in it, so the cold start handed back a
short Round or refused with a message about a total the project does not
have. A hit-and-run sample over the limits themselves fills the Round. It
runs only where the Round would otherwise come back short or be refused,
so a project the pool already fills gets exactly the Round it got before.
"""
import numpy as np
import pytest
import torch

from food_bo import FoodOptimizer
from storage import LocalStorage


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _project(low=30.0, high=30.001, name="tight"):
    opt = FoodOptimizer(name, storage=LocalStorage())
    opt.add_ingredient("A", 0, 60)
    opt.add_ingredient("B", 0, 50)
    opt.add_ingredient("C", 0, 40)
    opt.add_objective("Taste", 1.0)
    opt.add_quantity_constraint(["A", "B"], min_val=low, max_val=high)
    return opt


def _vectors(opt, recipes):
    return [opt._as_issued(r) for r in recipes]


def _distinct(opt, recipes):
    vectors = _vectors(opt, recipes)
    for i, v in enumerate(vectors):
        if opt._repeats(v, vectors[:i]):
            return False
    return True


def test_tight_limit_fills_round_one():
    opt = _project()
    recipes = opt.ask(5)
    assert len(recipes) == 5
    assert all(opt._check_constraints(r) for r in recipes)
    assert all(30.0 - 1e-6 <= r["A"] + r["B"] <= 30.001 + 1e-6 for r in recipes)
    assert _distinct(opt, recipes)


def test_tight_limit_with_a_total_fills_round_one():
    """With a total, a point from inside the limits is moved onto it only
    where that keeps them: projecting first pushed every point out of the
    thin A + B limit, and the Round was refused over the total."""
    opt = _project(low=70.0, high=70.001, name="tight total")
    opt.set_formulation_total(100)
    recipes = opt.ask(5)
    assert len(recipes) == 5
    assert all(opt._check_constraints(r) for r in recipes)
    assert all(70.0 - 1e-6 <= r["A"] + r["B"] <= 70.001 + 1e-6 for r in recipes)
    assert _distinct(opt, recipes)


def test_sampler_not_called_when_pool_suffices(monkeypatch):
    opt = _project(low=20.0, high=60.0, name="loose")

    def _boom(*a, **k):
        raise AssertionError("the pool fills this Round on its own")

    monkeypatch.setattr(opt, "_sample_feasible", _boom)
    recipes = opt.ask(5)
    assert len(recipes) == 5


def test_regenerate_differs():
    opt = _project()
    first = [dict(r) for r in opt.ask(5)]
    opt.set_pending_batch(None)
    second = [dict(r) for r in opt.ask(5)]
    assert len(second) == 5
    assert all(opt._check_constraints(r) for r in second)
    issued = np.vstack(_vectors(opt, first))
    assert not any(opt._repeats(v, issued) for v in _vectors(opt, second))


def test_sampler_leaves_the_random_state_alone():
    opt = _project()
    bounds = opt._get_bounds()
    torch_before = torch.random.get_rng_state().clone()
    numpy_before = np.random.get_state()[1].copy()
    rows = opt._sample_feasible(3, bounds, bounds.shape[1])
    assert rows
    assert torch.equal(torch.random.get_rng_state(), torch_before)
    assert np.array_equal(np.random.get_state()[1], numpy_before)


def test_sampler_returns_nothing_when_it_fails(monkeypatch):
    opt = _project()
    bounds = opt._get_bounds()

    def _broken():
        raise RuntimeError("no constraints today")

    monkeypatch.setattr(opt, "_get_botorch_constraints", _broken)
    assert opt._sample_feasible(3, bounds, bounds.shape[1]) == []
