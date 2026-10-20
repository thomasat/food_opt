"""The cold start never hands back a formulation the project already has.

It fast-forwards one Sobol sequence by the formulations issued so far, but a
Round can pass over points its limits reject, so the next Round's slice used
to start inside points a Round had already handed out: the burger example's
Round 2 repeated two of Round 1's formulations to the last digit (public
walk, 2026-10-19)."""
import pytest

import sample_projects as sp
import storage
from food_bo import FoodOptimizer


def _key(recipe):
    return tuple(round(float(v), 9) for _, v in sorted(recipe.items())
                 if not isinstance(v, str))


def _record(opt, recipes, first_no, batch_no):
    for no, recipe in enumerate(recipes, start=first_no):
        opt.tell(recipe, {o['name']: 5.0 + no for o in opt.objectives},
                 formulation_no=no, batch_no=batch_no)


@pytest.fixture
def burger(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = storage.LocalStorage()
    sp.ensure(store, (sp.BURGER,))
    return FoodOptimizer(sp.BURGER, storage=store)


def test_the_burger_round_2_repeats_nothing_from_round_1(burger):
    r1 = burger.ask(n_suggestions=3)
    _record(burger, r1, 1, 1)
    r2 = burger.ask(n_suggestions=3)
    assert not {_key(r) for r in r1} & {_key(r) for r in r2}
    assert len({_key(r) for r in r2}) == 3
    assert all(burger._check_constraints(r) for r in r2)


def test_a_round_never_repeats_an_open_rounds_formulation(burger):
    r1 = burger.ask(n_suggestions=3)
    _record(burger, r1[:2], 1, 1)
    # A formulation of its own waits in the open Round: the next suggestions
    # are not allowed to be it either.
    burger.set_pending_batch([dict(r1[2])], batch_no=2)
    vectors = burger._issued_vectors()
    assert burger._already_issued(r1[2], vectors)
    assert burger._already_issued(r1[0], vectors)


def _plain():
    opt = FoodOptimizer("plain")
    opt.add_ingredient("Flour", 10.0, 60.0)
    opt.add_ingredient("Sugar", 0.0, 30.0)
    opt.add_ingredient("Butter", 5.0, 25.0)
    opt.add_objective("Crunch", 1.0, goal="max")
    return opt


def test_without_rejected_points_the_cold_start_is_unchanged(tmp_path, monkeypatch):
    """Where no limit rejects a point there is nothing to skip: the same
    project asks the same formulations with the check on and off."""
    # Each in its own folder: a project of the same name would otherwise
    # open the other's saved file.
    (tmp_path / "on").mkdir()
    (tmp_path / "off").mkdir()
    monkeypatch.chdir(tmp_path / "on")
    with_check = _plain()
    r1 = with_check.ask(n_suggestions=3)
    _record(with_check, r1, 1, 1)
    r2 = with_check.ask(n_suggestions=2)

    monkeypatch.setattr(FoodOptimizer, "_already_issued", lambda self, r, i: False)
    monkeypatch.chdir(tmp_path / "off")
    without = _plain()
    s1 = without.ask(n_suggestions=3)
    _record(without, s1, 1, 1)
    s2 = without.ask(n_suggestions=2)
    assert [_key(r) for r in r1 + r2] == [_key(r) for r in s1 + s2]
