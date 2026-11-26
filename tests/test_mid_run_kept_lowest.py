"""An ingredient added mid-run with its Lowest kept above 0: the search
frame is widened to 0 so the past rows fit, but suggestions must stay in
the ingredient's own range and limits on it must hold."""
import pytest

import food_bo


def _taste(r):
    return {"Taste": 10 - ((r["A"] - 30) / 10) ** 2
            - ((r["B"] - 20) / 10) ** 2 - 0.2 * r.get("D", 0)}


def _project(tmp_path, monkeypatch, rounds):
    monkeypatch.chdir(tmp_path)
    app = food_bo.FoodOptimizer("kept_lowest")
    app.sobol_seed = 11
    app.add_ingredient("A", 0, 60)
    app.add_ingredient("B", 0, 50)
    app.add_objective("Taste", 1)
    for _ in range(rounds):
        app.set_pending_batch(None)
        rows = app.ask(3)
        for i, r in enumerate(rows):
            app.tell(r, _taste(r),
                     formulation_no=app.pending_batch[i]["formulation"])
    app.add_ingredient("D", 4, 10, keep_lowest=True)
    return app


@pytest.mark.parametrize("rounds", [1, 2])   # opening rounds, model rounds
def test_suggestions_stay_in_the_kept_range(tmp_path, monkeypatch, rounds):
    app = _project(tmp_path, monkeypatch, rounds)
    app.set_pending_batch(None)
    for r in app.ask(4):
        assert 4 - 1e-9 <= r["D"] <= 10 + 1e-9


@pytest.mark.parametrize("rounds", [1, 2])
@pytest.mark.parametrize("limit", ["D", "A+D", "protein"])
def test_limits_on_it_hold(tmp_path, monkeypatch, rounds, limit):
    app = _project(tmp_path, monkeypatch, rounds)
    if limit == "D":
        app.add_quantity_constraint(["D"], min_val=8)
    elif limit == "A+D":
        app.add_quantity_constraint(["A", "D"], min_val=40)
    else:
        app.add_property("Protein")
        for name, value in {"A": 10, "B": 0, "D": 80}.items():
            app.set_property_value(name, "Protein", value)
        app.add_constraint("Protein", min_val=14)
    app.set_pending_batch(None)
    for r in app.ask(4):
        assert 4 - 1e-9 <= r["D"] <= 10 + 1e-9
        assert app._check_constraints(r)
