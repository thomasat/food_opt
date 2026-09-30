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
    near = [c.near for c in found.moved]
    assert near == sorted(near, reverse=True)
    for c in found.moved:
        assert c.near is not None and 0.0 <= c.near <= c.whole + 1e-9
    whole = {s.name: s.spread for s in insights.spreads(opt, fitted, start).moved}
    assert {c.name: c.whole for c in found.moved} == pytest.approx(whole)


def test_only_the_tested_near_run_holding_the_start_counts(fitted_burger):
    """Oil's path is tested near in two separate runs: only the one
    holding the start is measured (review I8), worked out here on its own."""
    opt, fitted = fitted_burger
    start = insights.start_recipe(opt, opt.best_index())
    path = insights.build_path(opt, "Oil", start)
    near = insights.tested_near(opt, path)
    k = min(range(len(path.amounts)), key=lambda j: abs(path.amounts[j] - path.start_amount))
    run = insights._run_around(near, k)
    assert run and sum(near) > len(run)             # a second run the start is not in
    means = [p.mean for p in insights.predict(
        opt, fitted[insights.SCORE], [opt._encode(r) for r in path.recipes])]
    expected = max(means[j] for j in run) - min(means[j] for j in run)
    oil = next(c for c in insights.tested_near_changes(opt, fitted, start).moved
               if c.name == "Oil")
    assert oil.near == pytest.approx(expected)
    assert oil.near < oil.whole


def test_a_choice_not_tested_near_follows_the_ranked_ones(fitted_burger, monkeypatch):
    opt, fitted = fitted_burger
    start = insights.start_recipe(opt, opt.best_index())
    real = insights.tested_near
    monkeypatch.setattr(insights, "tested_near",
                        lambda o, path: [False] * len(path.amounts) if path.name == "Pea"
                        else real(o, path))
    found = insights.tested_near_changes(opt, fitted, start)
    assert [c.name for c in found.moved] == ["Oil", "Pea"]
    assert found.moved[1].near is None and found.start_tested


def test_a_start_not_tested_near_ranks_nothing(fitted_burger, monkeypatch):
    opt, fitted = fitted_burger
    start = insights.start_recipe(opt, opt.best_index())
    monkeypatch.setattr(insights, "tested_near", lambda o, path: [False] * len(path.amounts))
    found = insights.tested_near_changes(opt, fitted, start)
    assert not found.start_tested
    assert found.moved and all(c.near is None for c in found.moved)


def test_a_real_start_far_from_everything_tested_ranks_nothing(fitted_burger):
    opt, fitted = fitted_burger
    far = opt.fill_formulas({"Pea": 8.0, "Oil": 6.0})
    assert min(insights.distance(opt, far, insights.start_recipe(opt, i))
               for i in range(len(opt.Y_history))) > insights.CLOSE_TO_THE_BEST
    found = insights.tested_near_changes(opt, fitted, far)
    assert not found.start_tested
    assert found.moved and all(c.near is None and c.whole > 0 for c in found.moved)


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


def test_just_outside_one_percent_does_not_bind():
    opt = _burger(rounds=1)
    recipe = opt.fill_formulas({"Pea": 6.0, "Oil": 2.05})       # 1.25 % of Oil's 4 g span
    assert insights.binding_limits(opt, recipe) == []


def test_the_fill_to_total_row_at_its_highest_does_not_bind(monkeypatch):
    opt = _burger(rounds=1)
    opt._var_by_name("Water")['bounds'] = [0.0, 90.0]
    recipe = opt.fill_formulas({"Pea": 6.0, "Oil": 4.0})        # Water 90 of 0-90
    assert float(recipe["Water"]) == pytest.approx(90.0)
    assert insights.binding_limits(opt, recipe) == []
    # varying_variables leaves the Fill to total row out today; the check
    # of its own keeps it out should that ever change.
    monkeypatch.setattr(opt, "varying_variables", lambda: list(opt.variables))
    assert insights.binding_limits(opt, recipe) == []


def test_an_exactly_limit_never_binds_and_a_range_on_the_same_rows_does():
    """An Exactly limit is a fixed amount every formulation sits at (review
    Task 2 I-1): nothing to widen, so it is never listed."""
    opt = _burger(rounds=1)
    opt.add_quantity_constraint(["Pea", "Oil"], exactly=10.0)
    for pea, oil in ((6.0, 4.0), (5.0, 5.0)):
        assert insights.binding_limits(opt, opt.fill_formulas({"Pea": pea, "Oil": oil})) == []
    ranged = _burger(rounds=1)
    ranged.add_quantity_constraint(["Pea", "Oil"], max_val=10.0)
    [b] = insights.binding_limits(ranged, ranged.fill_formulas({"Pea": 6.0, "Oil": 4.0}))
    assert b.kind == "limit"


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


def test_higher_and_luck_are_judged_by_the_yardstick_handed_in():
    opt = _burger(rounds=2)
    top = float(opt.Y_history[opt.best_index()])
    # A 1.5-point gap over Best so far's predicted Repeat: higher only
    # against a 1-point yardstick.
    assert not insights.best_standing(opt, _best(opt, top + 1.5, top), 2.0).higher
    assert insights.best_standing(opt, _best(opt, top + 1.5, top), 1.0).higher
    # A Repeat 1.5 points under the measured score: luck only against 1 point.
    assert not insights.best_standing(opt, _best(opt, top - 1.5, top - 1.5), 2.0).luck
    assert insights.best_standing(opt, _best(opt, top - 1.5, top - 1.5), 1.0).luck


def test_without_a_predicted_repeat_the_measured_score_is_the_comparison():
    opt = _burger(rounds=2)
    top = float(opt.Y_history[opt.best_index()])
    best = _best(opt, top + 1.5, top)
    best.repeat = None
    s = insights.best_standing(opt, best, 1.0)
    assert s.higher and not s.luck
    best = _best(opt, top + 0.5, top)
    best.repeat = None
    assert not insights.best_standing(opt, best, 1.0).higher


def test_not_higher_and_not_tested_near():
    opt = _burger(rounds=2)
    top = float(opt.Y_history[opt.best_index()])
    s = insights.best_standing(opt, _best(opt, top, top), 1.0)
    assert (s.higher, s.tested_near, s.close_no) == (False, False, None)


def test_higher_and_tested_near():
    opt = _burger(rounds=2)
    top = float(opt.Y_history[opt.best_index()])
    s = insights.best_standing(opt, _best(opt, top + 3.0, top, close=1), 1.0)
    assert (s.higher, s.tested_near, s.close_no) == (True, True, int(opt.formulation_ids[1]))


def test_no_luck_when_the_repeat_reads_no_lower_than_the_score_shown():
    """A small yardstick (review M1): a Repeat of 90.52 reads "91", which
    is not below 90.85, so no luck line sits beside it."""
    opt = _burger(rounds=2)
    opt.Y_history[opt.best_index()] = 90.85
    assert not insights.best_standing(opt, _best(opt, 90.0, 90.52), 0.3).luck
    assert insights.best_standing(opt, _best(opt, 90.0, 90.2), 0.3).luck


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
