"""An ingredient added mid-run at the amount earlier formulations used.

Ported from stable e2752b7 (add_ingredient's `baseline`). Sorbate was in
the base formulation at 0.8 g all along, held there rather than varied, so
once it is added to the search the formulations already made are read at
0.8 g of it, not at none. The stored amounts are never rewritten: the
baseline lives on the row (`_absent_value`) and in the encoding only.
"""
import json

import pytest

import food_bo
import wording
from food_bo import FoodOptimizer


def _taste(r):
    return {"Taste": 10 - ((r["A"] - 30) / 10) ** 2
            - ((r["B"] - 20) / 10) ** 2}


def _run(app, n_rows):
    app.set_pending_batch(None)
    rows = app.ask(n_rows)
    for i, r in enumerate(rows):
        app.tell(r, _taste(r),
                 formulation_no=app.pending_batch[i]["formulation"])


def _project(tmp_path, monkeypatch, name="baseline"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    app = FoodOptimizer(name)
    app.sobol_seed = 11
    app.set_amount_unit("g")
    app.add_ingredient("A", 0, 60)
    app.add_ingredient("B", 0, 50)
    app.add_objective("Taste", 1)
    _run(app, 5)
    assert len(app.X_history) == 5
    return app


def _column(app, name):
    names = [v["name"] for v in app.variables if not app.has_formula(v)]
    return names.index(name)


def test_past_rows_encode_at_baseline(tmp_path, monkeypatch):
    app = _project(tmp_path, monkeypatch)
    before = [dict(r) for r in app.recipe_history]
    app.add_ingredient("Sorbate", 0.5, 1.2, baseline=0.8)
    var = app._var_by_name("Sorbate")
    assert var["bounds"] == (0.5, 1.2)
    assert var["_absent_value"] == 0.8
    col = _column(app, "Sorbate")
    assert [row[col] for row in app.X_history] == [0.8] * 5
    # The baseline sits inside the range: no widening down to 0.
    assert app._search_bounds()[col] == (0.5, 1.2)
    # Stored amounts are never rewritten.
    assert app.recipe_history == before
    assert all("Sorbate" not in r for r in app.recipe_history)


def test_baseline_outside_range_refused(tmp_path, monkeypatch):
    app = _project(tmp_path, monkeypatch)
    with pytest.raises(ValueError) as e:
        app.add_ingredient("Sorbate", 0.5, 1.2, baseline=0.3)
    assert str(e.value) == wording.baseline_outside_message(0.3, 0.5, 1.2)
    assert "Sorbate" not in [v["name"] for v in app.variables]


def test_baseline_before_any_result_is_not_kept(tmp_path, monkeypatch):
    """Nothing has been made yet, so there is nothing to read at it."""
    monkeypatch.chdir(tmp_path)
    app = FoodOptimizer("fresh")
    app.add_ingredient("Sorbate", 0.5, 1.2, baseline=0.8)
    assert "_absent_value" not in app._var_by_name("Sorbate")


def _rest_project(tmp_path, monkeypatch):
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    app = FoodOptimizer("baseline_rest", robust=False)
    app.sobol_seed = 11
    app.set_amount_unit("g")
    app.add_ingredient("A", 0, 40)
    app.add_ingredient("B", 0, 30)
    app.add_ingredient("Water", 0, 100)
    app.add_objective("Taste", 1)
    app.set_formula("Water", "= rest")
    app.set_formulation_total(100)
    _run(app, 5)
    return app


def test_baseline_counts_in_past_totals(tmp_path, monkeypatch):
    app = _rest_project(tmp_path, monkeypatch)
    app.add_ingredient("Sorbate", 0.5, 1.2, baseline=0.8)
    for i, recorded in enumerate(app.recipe_history):
        assert "Sorbate" not in recorded
        a, b, water = recorded["A"], recorded["B"], recorded["Water"]
        assert app.ingredient_total(recorded) == pytest.approx(
            a + b + water + 0.8)
        filled = app.fill_formulas(recorded)
        assert filled["Water"] == pytest.approx(100 - a - b - 0.8)
        # The reader that skips the decode agrees with the one that does.
        assert app._recorded_ingredient_total(i) == pytest.approx(
            app.ingredient_total(app._recorded_recipe(i)))
        # Brought to the total, the baseline is kept, not dropped to 0.
        scaled = app.at_model_total(recorded)
        assert scaled["Sorbate"] > 0.0


def test_suggestions_in_range_with_baseline(tmp_path, monkeypatch):
    app = _project(tmp_path, monkeypatch)
    app.add_ingredient("Sorbate", 0.5, 1.2, baseline=0.8)
    app.set_pending_batch(None)
    rows = app.ask(3)
    assert len(rows) == 3
    for r in rows:
        assert 0.5 - 1e-9 <= r["Sorbate"] <= 1.2 + 1e-9


def test_baseline_survives_a_saved_copy(tmp_path, monkeypatch):
    app = _project(tmp_path, monkeypatch)
    app.add_ingredient("Sorbate", 0.5, 1.2, baseline=0.8)
    state = json.loads(json.dumps(app.export_json()))
    food_bo.FoodOptimizer.validate_state(state)
    other = FoodOptimizer("baseline_copy")
    other.import_json(state)
    assert other._var_by_name("Sorbate")["_absent_value"] == 0.8
    assert other.X_history == app.X_history


def test_deleting_it_names_the_formulations_that_held_it(tmp_path,
                                                         monkeypatch):
    app = _project(tmp_path, monkeypatch)
    app.add_ingredient("Sorbate", 0.5, 1.2, baseline=0.8)
    assert app._ingredient_delete_refusal("Sorbate", False)


# ---- the grid ------------------------------------------------------ #

def _add_row(frame, **cells):
    out = frame.copy()
    row = {c: None for c in out.columns}
    row.update(cells)
    out.loc[len(out) + 1] = row
    return out


def _ing_row(name, low, high, **extra):
    row = {wording.NAME_LABEL: name,
           wording.TYPE_LABEL: wording.KIND_INGREDIENT,
           wording.LOWEST_LABEL: low, wording.HIGHEST_LABEL: high,
           wording.UNIT_LABEL: "g",
           wording.VENDOR_LABEL: "", wording.SKU_LABEL: ""}
    row.update(extra)
    return row


def test_grid_accepts_ingredient_baseline(tmp_path, monkeypatch):
    app = _project(tmp_path, monkeypatch)
    frame = app.ingredient_grid_frame()
    errors, messages = app.apply_ingredient_grid(_add_row(
        frame, **_ing_row("Sorbate", 0.5, 1.2,
                          **{wording.BASELINE_LABEL: 0.8})))
    assert errors == []
    var = app._var_by_name("Sorbate")
    assert var["bounds"] == (0.5, 1.2)
    assert var["_absent_value"] == 0.8
    col = _column(app, "Sorbate")
    assert [row[col] for row in app.X_history] == [0.8] * 5
    info = [text for kind, text in messages if kind == "info"]
    assert wording.formulations_contain_baseline("Sorbate", "0.8 g") in info
    assert wording.formulations_contain_none_of("Sorbate") not in info
    # The grid now shows that baseline, and saving it back unchanged is
    # not refused.
    frame = app.ingredient_grid_frame()
    row = list(frame[wording.NAME_LABEL]).index("Sorbate") + 1
    assert frame.loc[row, wording.BASELINE_LABEL] == 0.8
    errors, _ = app.apply_ingredient_grid(frame)
    assert errors == []


def test_grid_refuses_a_baseline_outside_the_new_rows_range(tmp_path,
                                                           monkeypatch):
    app = _project(tmp_path, monkeypatch)
    errors, _ = app.apply_ingredient_grid(_add_row(
        app.ingredient_grid_frame(),
        **_ing_row("Sorbate", 0.5, 1.2, **{wording.BASELINE_LABEL: 0.3})))
    assert [message for _, message in errors] == [
        wording.baseline_outside_message(0.3, 0.5, 1.2)]
    assert "Sorbate" not in [v["name"] for v in app.variables]


def test_grid_corrects_a_stored_ingredient_baseline(tmp_path, monkeypatch):
    app = _project(tmp_path, monkeypatch)
    app.add_ingredient("Sorbate", 0.5, 1.2, baseline=0.8)
    frame = app.ingredient_grid_frame()
    row = list(frame[wording.NAME_LABEL]).index("Sorbate") + 1
    frame.loc[row, wording.BASELINE_LABEL] = 1.0
    errors, _ = app.apply_ingredient_grid(frame)
    assert errors == []
    assert app._var_by_name("Sorbate")["_absent_value"] == 1.0
    col = _column(app, "Sorbate")
    assert [r[col] for r in app.X_history] == [1.0] * 5


def test_grid_still_refuses_a_baseline_on_an_ingredient_without_one(
        tmp_path, monkeypatch):
    """A or B was in every formulation at its own recorded amounts, so a
    baseline for it would read nothing."""
    app = _project(tmp_path, monkeypatch)
    frame = app.ingredient_grid_frame()
    frame.loc[1, wording.BASELINE_LABEL] = 2.0
    errors, _ = app.apply_ingredient_grid(frame)
    assert [message for _, message in errors] == [
        wording.only_a_setting_has(wording.BASELINE_LABEL)]
