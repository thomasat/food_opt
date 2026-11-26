"""The rest row's own range holds only where the other limits let it.

Final review, Important #1 (Ruling 4). The held range was decided from the
rows' allowed amounts alone, so a limit or a widening the app accepted could
leave a range no formulation meets together with the limits, and Generate
then refused a project that generated before. The range is now read against
every other row and limit (one small linear programme each way); a range the
limits rule out stays dormant, as one the amounts rule out always did, and
the caption names it.
"""
import logging

import pytest
from botorch.exceptions.errors import InfeasibilityError

import food_bo
import wording


def _project(tmp_path, monkeypatch, a=(0, 400), b=(0, 300), water=(450, 700)):
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    app = food_bo.FoodOptimizer("rest_limits", robust=False)
    app.sobol_seed = 11
    app.set_amount_unit("g")
    app.add_ingredient("A", *a)
    app.add_ingredient("B", *b)
    app.add_ingredient("Water", *water)
    app.add_objective("Taste", 1)
    app.set_formula("Water", "= rest")
    app.set_formulation_total(1000)
    return app


def _tell_rounds(app, rounds=2, n=3):
    for _ in range(rounds):
        app.set_pending_batch(None)
        rows = app.ask(n)
        for i, r in enumerate(rows):
            app.tell(r, {"Taste": r["A"] / 40.0},
                     formulation_no=app.pending_batch[i]["formulation"])


def _water(app):
    return app._var_by_name("Water")


# P4: A + B at least 600 leaves Water at most 400 g, under its own 450.
def test_limit_that_rules_out_the_range_still_generates(tmp_path, monkeypatch):
    app = _project(tmp_path, monkeypatch)
    app.add_quantity_constraint(["A", "B"], min_val=600)
    assert app._held_rest_range(_water(app)) is None
    app.set_pending_batch(None)
    rows = app.ask(3)
    assert len(rows) == 3
    for r in rows:
        assert app._check_constraints(r), r
        assert r["A"] + r["B"] >= 600 - 1e-6
        assert 300 - 1e-6 <= r["Water"] <= 400 + 1e-6


def test_caption_names_the_range_the_limits_rule_out(tmp_path, monkeypatch):
    app = _project(tmp_path, monkeypatch)
    app.add_quantity_constraint(["A", "B"], min_val=600)
    assert app.worked_out_captions() == [wording.worked_out_caption(
        "Water", "= rest", "300.00", "1000.00 g", "1000 g", rest=True,
        outside_text="450.00 to 700.00 g",
        limited_text="300.00 to 400.00 g")]
    caption = app.worked_out_captions()[0]
    assert "450.00 to 700.00 g" in caption
    assert "300.00 to 400.00 g" in caption


# probe_widen: A + B at most 200 leaves Water at least 800 g, over its own
# 700. Before the widening the amounts rule the range out; after it the
# amounts alone would let it hold, but the limit still does not.
def test_widening_past_a_limit_still_generates_opening(tmp_path, monkeypatch):
    app = _project(tmp_path, monkeypatch, a=(0, 200), b=(0, 50))
    app.add_quantity_constraint(["A", "B"], max_val=200)
    assert app._held_rest_range(_water(app)) is None
    app.add_ingredient("A", 0, 400)
    assert app._held_rest_range(_water(app)) is None
    app.set_pending_batch(None)
    rows = app.ask(3)
    assert len(rows) == 3
    for r in rows:
        assert app._check_constraints(r), r
        assert r["Water"] >= 800 - 1e-6
    caption = app.worked_out_captions()[0]
    assert "450.00 to 700.00 g" in caption
    assert "800.00 to 1000.00 g" in caption


def test_widening_past_a_limit_still_generates_model(tmp_path, monkeypatch):
    app = _project(tmp_path, monkeypatch, a=(0, 200), b=(0, 50))
    app.add_quantity_constraint(["A", "B"], max_val=200)
    _tell_rounds(app)
    assert app.complete_count() == 6
    app.add_ingredient("A", 0, 400)
    assert app._held_rest_range(_water(app)) is None
    app.set_pending_batch(None)
    rows = app.ask(3)
    assert len(rows) == 3
    for r in rows:
        assert app._check_constraints(r), r
        assert r["A"] + r["B"] <= 200 + 1e-6


def test_a_limit_the_range_can_meet_keeps_it_held(tmp_path, monkeypatch):
    # A + B at least 100 leaves Water up to 900 g, so 450..700 still holds,
    # and it is held exactly as the amounts alone hold it.
    app = _project(tmp_path, monkeypatch)
    before = app._held_rest_range(_water(app))
    app.add_quantity_constraint(["A", "B"], min_val=100)
    assert app._held_rest_range(_water(app)) == before == (
        (450.0, 700.0), (300.0, 1000.0))
    app.set_pending_batch(None)
    for r in app.ask(3):
        assert 450 - 1e-6 <= r["Water"] <= 700 + 1e-6


def test_a_range_the_amounts_never_bind_reads_no_limits(tmp_path,
                                                        monkeypatch):
    # Water 0..2000 is wider than anything A and B leave it: the limits are
    # never read for it, so nothing is solved.
    app = _project(tmp_path, monkeypatch, water=(0, 2000))
    app.add_quantity_constraint(["A", "B"], min_val=600)
    calls = []
    real = app._solve_rest_reach
    monkeypatch.setattr(app, "_solve_rest_reach",
                        lambda var: calls.append(1) or real(var))
    assert app._held_rest_range(_water(app)) is not None
    app.set_pending_batch(None)
    app.ask(3)
    assert calls == []


def test_the_reach_is_solved_once_per_project_state(tmp_path, monkeypatch):
    # _check_constraints runs per candidate of the opening's pool; the
    # linear programme must not.
    app = _project(tmp_path, monkeypatch)
    app.add_quantity_constraint(["A", "B"], min_val=100)
    calls = []
    real = app._solve_rest_reach
    monkeypatch.setattr(app, "_solve_rest_reach",
                        lambda var: calls.append(1) or real(var))
    app.set_pending_batch(None)
    app.ask(5)
    assert 1 <= len(calls) <= 2
    # A new limit is a new state, and is read again.
    app.add_quantity_constraint(["A"], max_val=350)
    app._held_rest_range(_water(app))
    assert len(calls) >= 2


def test_held_caption_names_what_clips_the_amounts(tmp_path, monkeypatch):
    high_only = _project(tmp_path / "high", monkeypatch, water=(0, 700))
    assert high_only.worked_out_captions() == [
        "Water fills to the 1000 g default batch size: "
        "between 300.00 and 700.00 g (its own Highest)."]
    low_only = _project(tmp_path / "low", monkeypatch, water=(450, 1000))
    assert low_only.worked_out_captions() == [
        "Water fills to the 1000 g default batch size: "
        "between 450.00 and 1000.00 g (its own Lowest)."]


# BoTorch's own refusal of an empty polytope, in the model's path.
def _conflicting(tmp_path, monkeypatch):
    app = _project(tmp_path, monkeypatch, water=(0, 0))
    _tell_rounds(app)
    # Written straight onto the list, as an older file could leave them:
    # each is reachable alone, and no formulation meets both.
    app.quantity_constraints.append(
        {"ingredients": ["A"], "min": 350.0, "max": None})
    app.quantity_constraints.append(
        {"ingredients": ["A", "B"], "min": None, "max": 300.0})
    return app


def test_model_path_names_limits_that_leave_no_formulation(tmp_path,
                                                           monkeypatch):
    app = _conflicting(tmp_path, monkeypatch)
    logging.disable(logging.WARNING)
    try:
        app.set_pending_batch(None)
        with pytest.raises(ValueError) as e:
            app.ask(3)
    finally:
        logging.disable(logging.NOTSET)
    assert not isinstance(e.value, InfeasibilityError)
    assert str(e.value) == wording.limits_leave_no_formulation(2)
    assert "polytope" not in str(e.value)


def test_model_path_maps_botorch_infeasibility(tmp_path, monkeypatch):
    app = _project(tmp_path, monkeypatch, water=(0, 0))
    _tell_rounds(app)
    app.add_quantity_constraint(["A", "B"], min_val=100)

    def empty(*args, **kwargs):
        raise InfeasibilityError(
            "No feasible point found. Constraint polytope appears empty. "
            "Check your constraints.")

    monkeypatch.setattr(food_bo, "optimize_acqf", empty)
    app.set_pending_batch(None)
    with pytest.raises(ValueError) as e:
        app.ask(3)
    assert str(e.value) == wording.limits_leave_no_formulation(1)


def test_no_formulation_wording():
    assert wording.limits_leave_no_formulation(3) == (
        "No formulation meets all 3 limits at once. Widen one in Set up.")
    assert wording.limits_leave_no_formulation(1) == wording.GENERATE_FAILED
    assert wording.limits_leave_no_formulation(0) == wording.GENERATE_FAILED


def test_new_sentences_pass_the_vocabulary(tmp_path, monkeypatch):
    from tests.test_food_bo import _BANNED
    app = _project(tmp_path / "p4", monkeypatch)
    app.add_quantity_constraint(["A", "B"], min_val=600)
    held = _project(tmp_path / "held", monkeypatch)
    for text in (app.worked_out_captions() + held.worked_out_captions()
                 + [wording.limits_leave_no_formulation(4)]):
        assert not any(p.search(text) for p in _BANNED), text
