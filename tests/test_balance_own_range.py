"""The row written = rest is kept inside its own Lowest and Highest.

Ported from stable e2752b7 (set_balance's min/max became two linear
limits on the others). A 0–400 and B 0–300 in a 1000 g formulation leave
Water anywhere in 300..1000 g; its own 450..700 g must hold on both sides.
"""
import food_bo


def _taste(r):
    # Rewards a lot of A, which pushes Water toward its own Lowest.
    return {"Taste": r["A"] / 40.0 - ((r["B"] - 150) / 100) ** 2}


def _project(tmp_path, monkeypatch, water=(450, 700)):
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    app = food_bo.FoodOptimizer("rest_range", robust=False)
    app.sobol_seed = 11
    app.set_amount_unit("g")
    app.add_ingredient("A", 0, 400)
    app.add_ingredient("B", 0, 300)
    app.add_ingredient("Water", *water)
    app.add_objective("Taste", 1)
    app.set_formula("Water", "= rest")
    app.set_formulation_total(1000)
    return app


def _in_range(app, rows):
    for r in rows:
        assert 450 - 1e-6 <= r["Water"] <= 700 + 1e-6, r
        assert app._check_constraints(r), r


def test_rest_row_kept_inside_its_range_opening(tmp_path, monkeypatch):
    app = _project(tmp_path, monkeypatch)
    app.set_pending_batch(None)
    _in_range(app, app.ask(5))


def test_rest_row_kept_inside_its_range_model_rounds(tmp_path, monkeypatch):
    app = _project(tmp_path, monkeypatch)
    for _ in range(2):
        app.set_pending_batch(None)
        rows = app.ask(3)
        for i, r in enumerate(rows):
            app.tell(r, _taste(r),
                     formulation_no=app.pending_batch[i]["formulation"])
    assert app.complete_count() == 6
    app.set_pending_batch(None)
    _in_range(app, app.ask(4))


def test_rest_range_wider_than_reach_adds_no_rows(tmp_path, monkeypatch):
    wide = _project(tmp_path / "wide", monkeypatch, water=(0, 2000))
    none = _project(tmp_path / "none", monkeypatch, water=(0, 0))
    assert len(wide._get_botorch_constraints()) == \
        len(none._get_botorch_constraints())


def test_rest_row_with_no_range_adds_no_rows(tmp_path, monkeypatch):
    app = _project(tmp_path, monkeypatch, water=(0, 0))
    assert app._rest_row_range(app._var_by_name("Water")) is None
    rows = app._get_botorch_constraints()
    # Only the floor: Water = 1000 − A − B ≥ 0.
    assert len(rows) == 1
    idx, coeffs, rhs = rows[0]
    assert idx.tolist() == [0, 1]
    assert coeffs.tolist() == [-400.0, -300.0]
    assert rhs == -1000.0


def test_rest_row_range_is_reported(tmp_path, monkeypatch):
    app = _project(tmp_path, monkeypatch)
    assert app._rest_row_range(app._var_by_name("Water")) == (450.0, 700.0)
    assert app._rest_row_range(app._var_by_name("A")) is None


def test_check_constraints_rejects_rest_outside_range(tmp_path, monkeypatch):
    app = _project(tmp_path, monkeypatch)
    low = app.fill_formulas({"A": 360.0, "B": 200.0})
    assert abs(low["Water"] - 440.0) < 1e-9
    assert app._check_constraints(low) is False
    high = app.fill_formulas({"A": 100.0, "B": 150.0})
    assert abs(high["Water"] - 750.0) < 1e-9
    assert app._check_constraints(high) is False
    inside = app.fill_formulas({"A": 300.0, "B": 150.0})
    assert app._check_constraints(inside) is True


def test_caption_gives_the_enforced_range(tmp_path, monkeypatch):
    app = _project(tmp_path, monkeypatch)
    assert app.worked_out_captions() == [
        "Water fills to the 1000 g default batch size: "
        "between 450.00 and 700.00 g."]


def test_rest_range_out_of_reach_stays_dormant(tmp_path, monkeypatch):
    # Water 0..200 g, but A and B leave it 300..1000 g: no amount can meet
    # its range, so it adds no rows and nothing is rejected for it.
    app = _project(tmp_path / "out", monkeypatch, water=(0, 200))
    none = _project(tmp_path / "none", monkeypatch, water=(0, 0))
    assert len(app._get_botorch_constraints()) == \
        len(none._get_botorch_constraints())
    assert app._check_constraints(app.fill_formulas({"A": 100.0, "B": 100.0}))
    monkeypatch.chdir(tmp_path / "out")
    app.set_pending_batch(None)
    rows = app.ask(3)
    assert len(rows) == 3
    for r in rows:
        assert 300 - 1e-6 <= r["Water"] <= 1000 + 1e-6
    assert "do not apply" in app.worked_out_captions()[0]
