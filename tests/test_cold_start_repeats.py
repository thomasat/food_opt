"""The cold start never hands back a formulation the project already has.

It fast-forwards one Sobol sequence by the formulations issued so far, but a
Round can pass over points its limits reject, so the next Round's slice used
to start inside points a Round had already handed out: the burger example's
Round 2 repeated two of Round 1's formulations to the last digit (public
walk, 2026-10-19), Generate a different batch handed back the batch it
discarded, and a tightly limited project could get one formulation five
times. Reviews: scratchpad coldstart-review (methodologist, code reviewer,
tester)."""
import time

import pytest

import sample_projects as sp
import storage
from food_bo import FoodOptimizer


def _key(recipe):
    return tuple(round(float(v), 9) for _, v in sorted(recipe.items())
                 if not isinstance(v, str))


def _keys(recipes):
    return {_key(r) for r in recipes}


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


def _round(opt, n):
    """Generate as 2 · Make a round does: the Round stays open."""
    return [dict(r) for r in opt.ask(n_suggestions=n)]


def _close(opt, recipes, first_no):
    """Save results for every row, closing the Round."""
    _record(opt, recipes, first_no, opt.pending_batch_no)
    opt.set_pending_batch(None)


def test_the_burger_round_2_repeats_nothing_from_round_1(burger):
    r1 = _round(burger, 3)
    _close(burger, r1, 1)
    r2 = _round(burger, 3)
    assert not _keys(r1) & _keys(r2)
    assert len(_keys(r2)) == 3
    assert all(burger._check_constraints(r) for r in r2)


def test_a_regenerate_never_hands_back_what_it_discarded(burger):
    """Generate a different batch clears the Round, then asks again
    (ui_batch._regenerate); three times over, nothing comes back."""
    seen = _round(burger, 3)
    for _ in range(3):
        burger.set_pending_batch(None)
        again = _round(burger, 3)
        assert not _keys(seen) & _keys(again)
        seen += again


def test_dropped_suggestions_survive_a_save_and_reopen(tmp_path, monkeypatch, burger):
    first = _round(burger, 3)
    burger.set_pending_batch(None)
    burger.save()
    reopened = FoodOptimizer(sp.BURGER, storage=burger.storage)
    assert len(reopened._retired()) == 3
    assert not _keys(first) & _keys(_round(reopened, 3))


def test_a_not_scored_formulation_does_not_come_back(burger):
    r1 = _round(burger, 3)
    _record(burger, r1[:2], 1, burger.pending_batch_no)
    burger.set_pending_batch(None)
    burger.skipped.append({'formulation': 3, 'batch': 1, 'recipe': dict(r1[2]),
                           'note': ""})
    assert not _keys(r1) & _keys(_round(burger, 3))


def test_a_round_never_repeats_an_open_rounds_formulation(burger):
    r1 = _round(burger, 3)
    issued = burger._issued_matrix()
    assert all(burger._repeats(burger._as_issued(r), issued) for r in r1)


def _grid(tmp_path, monkeypatch):
    """Two dial-stepped settings and nothing else: nine formulations exist."""
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("grid")
    opt.add_process_parameter("Temp", 160.0, 200.0, step=20.0)
    opt.add_process_parameter("Time", 10.0, 20.0, step=5.0)
    opt.add_objective("Crunch", 1.0, goal="max")
    return opt


def test_a_stepped_grid_has_no_repeat_inside_a_round_and_never_fails(tmp_path, monkeypatch):
    opt = _grid(tmp_path, monkeypatch)
    r1 = _round(opt, 5)
    assert len(_keys(r1)) == 5
    opt.set_pending_batch(None)
    # Only 4 cells are left of 9; asking for 5 still answers: the 4 new
    # cells, then one repeat that is not already in this Round.
    r2 = _round(opt, 5)
    assert len(r2) == 5
    assert len(_keys(r2)) == 5
    assert len(_keys(r2) - _keys(r1)) == 4


def _plain():
    opt = FoodOptimizer("plain")
    opt.add_ingredient("Flour", 10.0, 60.0)
    opt.add_ingredient("Sugar", 0.0, 30.0)
    opt.add_ingredient("Butter", 5.0, 25.0)
    opt.add_objective("Crunch", 1.0, goal="max")
    return opt


def test_without_rejected_points_the_cold_start_is_unchanged(tmp_path, monkeypatch):
    """Where nothing would repeat there is nothing to pass over: the same
    project asks the same formulations with the check on and off."""
    # Each in its own folder: a project of the same name would otherwise
    # open the other's saved file.
    (tmp_path / "on").mkdir()
    (tmp_path / "off").mkdir()
    monkeypatch.chdir(tmp_path / "on")
    with_check = _plain()
    r1 = _round(with_check, 3)
    _close(with_check, r1, 1)
    r2 = _round(with_check, 2)

    monkeypatch.setattr(FoodOptimizer, "_repeats", lambda self, v, rows: False)
    monkeypatch.chdir(tmp_path / "off")
    without = _plain()
    s1 = _round(without, 3)
    _close(without, s1, 1)
    s2 = _round(without, 2)
    assert [_key(r) for r in r1 + r2] == [_key(r) for r in s1 + s2]


def test_a_project_that_never_dropped_a_suggestion_saves_as_before(burger):
    assert 'retired_suggestions' not in burger.export_json()
    _round(burger, 3)
    burger.set_pending_batch(None)
    assert 'retired_suggestions' in burger.export_json()


def test_the_check_stays_fast_on_a_long_history(burger):
    """70 formulations waiting for results keep the cold start going; one
    vectorised comparison per candidate keeps each Round quick."""
    for _ in range(7):
        _round(burger, 10)
        burger.set_pending_batch(None)
    start = time.perf_counter()
    _round(burger, 10)
    assert time.perf_counter() - start < 1.0


def test_an_unreadable_row_never_stops_generate(tmp_path, monkeypatch):
    """A blank or a word where an amount belongs (an older or hand-edited
    file) can match nothing; it is left out of the check."""
    monkeypatch.chdir(tmp_path)
    opt = _plain()
    opt.skipped.append({'formulation': 9, 'batch': 1, 'note': "",
                        'recipe': {'Flour': None, 'Sugar': "", 'Butter': 5.0}})
    opt.retired_suggestions = [{'Flour': "x"}]
    assert len(_round(opt, 3)) == 3


def test_a_damaged_retired_section_is_refused_at_the_door(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    state = _plain().export_json()
    state['retired_suggestions'] = 5
    with pytest.raises(ValueError):
        FoodOptimizer.validate_state(state)
    state['retired_suggestions'] = [[1, 2]]
    with pytest.raises(ValueError):
        FoodOptimizer.validate_state(state)


def test_a_blank_amount_never_stops_a_drop_or_a_generate(burger):
    """A blank amount in a damaged file: dropping the open Round (a setup
    change, Generate a different batch) and the next Generate still work."""
    _round(burger, 3)
    burger.pending_batch[0]['recipe'][next(iter(burger.pending_batch[0]['recipe']))] = None
    burger.set_pending_batch(None)
    burger.skipped.append({'formulation': 90, 'batch': 1, 'note': "",
                           'recipe': {'Wheat gluten': None, 'Coconut oil': ""}})
    assert len(_round(burger, 3)) == 3
