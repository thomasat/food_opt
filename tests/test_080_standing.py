"""0.8.0 speed and simplicity, Plan B, Task 2: where the predicted best
stands against Best so far, what it sits at, and which ingredient changes
the prediction most near what has been tested (spec 2026-09-30 §2.3, §2.5;
review I7, I8)."""
import pytest

import insights
from tests.test_insights import _burger


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.fixture(scope="module")
def fitted_burger(tmp_path_factory):
    import os
    old = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("standing"))
    try:
        opt = _burger(rounds=4)
        yield opt, insights.fit_all(opt)
    finally:
        os.chdir(old)


# ------------------------------------------------------------------ #
#  tested_near_changes (review I8)
# ------------------------------------------------------------------ #

def test_the_run_around_the_start():
    near = [False, True, True, False, True]
    assert insights._run_around(near, 1) == [1, 2]
    assert insights._run_around(near, 2) == [1, 2]
    assert insights._run_around(near, 4) == [4]
    assert insights._run_around(near, 3) == []


def test_changes_near_what_was_tested_rank_the_choices(fitted_burger):
    opt, fitted = fitted_burger
    start = insights.start_recipe(opt, opt.best_index())
    found = insights.tested_near_changes(opt, fitted, start)
    assert found.start_tested
    assert [c.name for c in found.moved] == sorted(
        [c.name for c in found.moved], key=lambda n: -next(c.near for c in found.moved
                                                            if c.name == n))
    for c in found.moved:
        assert c.near is not None and 0.0 <= c.near <= c.whole + 1e-9
    whole = {s.name: s.spread for s in insights.spreads(opt, fitted, start).moved}
    assert {c.name: c.whole for c in found.moved} == pytest.approx(whole)


def test_a_start_not_tested_near_ranks_nothing(fitted_burger, monkeypatch):
    opt, fitted = fitted_burger
    start = insights.start_recipe(opt, opt.best_index())
    monkeypatch.setattr(insights, "tested_near", lambda o, path: [False] * len(path.amounts))
    found = insights.tested_near_changes(opt, fitted, start)
    assert not found.start_tested
    assert found.moved and all(c.near is None for c in found.moved)


def test_no_model_no_changes(fitted_burger):
    opt, fitted = fitted_burger
    start = insights.start_recipe(opt, opt.best_index())
    none = dict(fitted, **{insights.SCORE: insights.Fitted(insights.SCORE, None, None, 0, 0, 1)})
    assert insights.tested_near_changes(opt, none, start) == insights.NearChanges([], [], False)


# ------------------------------------------------------------------ #
#  binding_limits (review I7, Line 3)
# ------------------------------------------------------------------ #

def test_rows_at_their_lowest_or_highest_bind_and_the_fill_row_does_not():
    opt = _burger(rounds=1)
    recipe = opt.fill_formulas({"Pea": 8.0, "Oil": 2.03})       # Oil within 1 % of 4 g
    assert insights.binding_limits(opt, recipe) == [
        insights.Binding("highest", "Pea", "Pea"), insights.Binding("lowest", "Oil", "Oil")]
    inside = opt.fill_formulas({"Pea": 6.0, "Oil": 4.0})
    assert insights.binding_limits(opt, inside) == []


def test_an_ingredient_limit_at_its_bound_binds():
    opt = _burger(rounds=1)
    opt.add_quantity_constraint(["Pea", "Oil"], max_val=10.0)
    recipe = opt.fill_formulas({"Pea": 6.0, "Oil": 3.95})       # 9.95: within 1 % of 10
    [b] = insights.binding_limits(opt, recipe)
    assert (b.kind, b.text) == ("limit", opt.limit_text(opt.quantity_constraints[-1]))


def test_a_property_limit_at_its_bound_binds():
    opt = _burger(rounds=1)
    opt.add_property("Fat")
    for n, v in (("Pea", 2.0), ("Oil", 100.0), ("Water", 0.0)):
        opt.set_property_value(n, "Fat", v)
    opt.add_constraint("Fat", max_val=5.0)
    at = opt.fill_formulas({"Pea": 6.0, "Oil": 4.87})           # Fat 4.99 per 100 g
    [b] = insights.binding_limits(opt, at)
    assert (b.kind, b.name, b.text) == ("limit", "Fat", opt.property_limit_text(opt.constraints[0]))
    below = opt.fill_formulas({"Pea": 6.0, "Oil": 3.0})         # Fat 3.12
    assert insights.binding_limits(opt, below) == []


# ------------------------------------------------------------------ #
#  best_standing (review I7, Line 2)
# ------------------------------------------------------------------ #

def _best(opt, mean, repeat_mean, close=None):
    p = insights.Prediction(mean, mean - 4, mean + 4, mean, 2.0)
    r = insights.Prediction(repeat_mean, repeat_mean - 4, repeat_mean + 4, repeat_mean, 2.0)
    recipe = opt.fill_formulas({"Pea": 6.0, "Oil": 4.0})
    return insights.PredictedBest(recipe, p, close, opt.best_index(), 1.0, r)


def test_higher_compares_with_best_so_fars_predicted_repeat():
    opt = _burger(rounds=2)
    top = float(opt.Y_history[opt.best_index()])
    s = insights.best_standing(opt, _best(opt, top + 1.5, top), 1.0)
    assert s.higher and not s.tested_near and s.close_no is None
    s = insights.best_standing(opt, _best(opt, top + 0.5, top), 1.0)
    assert not s.higher


def test_the_measured_score_is_not_the_comparison():
    """Best so far's measured score carries luck; its predicted Repeat does
    not. A predicted best 1.5 above the Repeat but below the measured score
    is still clearly higher than what a Repeat would score."""
    opt = _burger(rounds=2)
    top = float(opt.Y_history[opt.best_index()])
    s = insights.best_standing(opt, _best(opt, top - 1.0, top - 2.5), 1.0)
    assert s.higher and s.luck


def test_close_to_a_tested_formulation_is_named():
    opt = _burger(rounds=2)
    top = float(opt.Y_history[opt.best_index()])
    s = insights.best_standing(opt, _best(opt, top, top, close=2), 1.0)
    assert s.tested_near and s.close_no == int(opt.formulation_ids[2])
    assert not s.luck
