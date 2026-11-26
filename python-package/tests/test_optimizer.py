import pytest
from botorch.utils.transforms import unnormalize

import egbo.optimizer as optimizer_module
from egbo import Optimizer
from egbo._score import score
from egbo._space import bounds_tensor, search_bounds
from egbo._start import sobol_pool


def _two(seed=0, **kwargs):
    opt = Optimizer(seed=seed, **kwargs)
    opt.add_ingredient("Pea protein", 0, 5)
    opt.add_ingredient("Oil", 0, 5)
    opt.add_objective("Taste", weight=1)
    return opt


def _taste(recipe):
    # A smooth toy response, peaked at Pea protein 3, Oil 1.
    value = 10 - (recipe["Pea protein"] - 3) ** 2 - (recipe.get("Oil", 0) - 1) ** 2
    return {"Taste": max(0.0, min(10.0, value))}


def _record(opt, n):
    for recipe in opt.ask(n):
        opt.tell(recipe, _taste(recipe))


def _numbers(opt):
    return [row["formulation"] for row in opt.pending()]


def test_first_asks_are_sobol_and_numbered():
    opt = _two(seed=7)
    batch = opt.ask(3)
    assert _numbers(opt) == [1, 2, 3]
    spans = search_bounds(opt.variables, [], [])
    pool = sobol_pool(2048, 2, 7, 0, {})
    expected = [unnormalize(pool[i], bounds_tensor(spans)).tolist() for i in range(3)]
    assert [[r["Pea protein"], r["Oil"]] for r in batch] == expected
    assert opt.next_formulation_no == 4


def test_ask_twice_discards_and_differs():
    opt = _two()
    first = opt.ask(3)
    second = opt.ask(3)
    assert second != first
    assert _numbers(opt) == [4, 5, 6]
    assert opt.retired == first


def test_same_state_same_batch():
    a, b = _two(seed=3), _two(seed=3)
    assert a.ask(4) == b.ask(4)


def test_seed_none_draws_and_stores_a_seed():
    opt = Optimizer()
    assert isinstance(opt.sobol_seed, int)
    assert 0 <= opt.sobol_seed < 2 ** 32


def test_tell_matches_open_row():
    opt = _two()
    batch = opt.ask(3)
    assert opt.tell(batch[1], {"Taste": 5}) == 2
    assert _numbers(opt) == [1, 3]
    assert opt.history()[0]["formulation"] == 2


def test_tell_rounded_recipe_is_own_formulation():
    opt = _two()
    batch = opt.ask(3)
    rounded = {k: round(v, 2) for k, v in batch[0].items()}
    assert opt.tell(rounded, {"Taste": 5}) == 4
    assert 1 in _numbers(opt)
    assert opt.next_formulation_no == 5


def test_tell_unknown_names_refused():
    opt = _two()
    batch = opt.ask(1)
    with pytest.raises(ValueError, match="Zinc"):
        opt.tell({**batch[0], "Zinc": 1.0}, {"Taste": 5})
    with pytest.raises(ValueError, match="Smell"):
        opt.tell(batch[0], {"Taste": 5, "Smell": 3})
    assert opt.history() == []
    assert _numbers(opt) == [1]


def test_tell_missing_result_refused():
    opt = _two()
    batch = opt.ask(1)
    with pytest.raises(ValueError, match="Missing result for 'Taste'"):
        opt.tell(batch[0], {})
    with pytest.raises(ValueError, match="Taste"):
        opt.tell(batch[0], {"Taste": "good"})
    assert opt.history() == []


def test_tell_recipe_missing_a_variable_reads_absent():
    opt = _two()
    opt.tell({"Pea protein": 2.0}, {"Taste": 4})
    entry = opt.history()[0]
    assert entry["recipe"] == {"Pea protein": 2.0}
    assert entry["score"] == score(opt.objectives, {"Taste": 4})


def test_warm_round_after_n_init(monkeypatch):
    opt = _two()
    _record(opt, 5)
    calls = []
    real = optimizer_module.suggest

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(optimizer_module, "suggest", spy)
    batch = opt.ask(2)
    assert len(calls) == 1
    train_X, train_Y, _, q, seed = calls[0][0]
    assert train_X.shape == (5, 2) and train_Y.shape == (5, 1) and q == 2
    assert seed == 6                       # the next formulation number
    assert _numbers(opt) == [6, 7]
    for recipe in batch:
        assert 0 <= recipe["Pea protein"] <= 5 and 0 <= recipe["Oil"] <= 5


def test_fix_shrinks_active_set_and_keeps_history():
    opt = _two()
    _record(opt, 5)
    opt.fix("Oil", 0)
    assert opt.active_variables() == ["Pea protein"]
    batch = opt.ask(2)
    assert all(recipe["Oil"] == 0 for recipe in batch)
    assert len(opt.history()) == 5
    opt.set_range("Oil", 0, 5)
    assert opt.active_variables() == ["Pea protein", "Oil"]


def test_fix_drops_open_batch():
    opt = _two()
    batch = opt.ask(2)
    opt.fix("Oil", 1)
    assert opt.pending() == []
    assert opt.retired == batch


def test_mid_run_ingredient_absent_in_past():
    opt = _two()
    _record(opt, 2)
    opt.add_ingredient("Inulin", 1, 3)
    inulin = opt.variables[-1]
    assert (inulin.low, inulin.high) == (0.0, 3.0)    # Lowest forced to 0
    assert all("Inulin" not in h["recipe"] for h in opt.history())
    batch = opt.ask(2)
    assert all(0 <= r["Inulin"] <= 3 for r in batch)

    kept = _two()
    _record(kept, 2)
    kept.add_ingredient("Inulin", 1, 3, keep_lowest=True)
    assert kept.variables[-1].low == 1.0
    spans = search_bounds(kept.variables, kept.recipes, [])
    assert spans[-1] == (0.0, 3.0)                     # widened to the absent 0


def test_mid_run_setting_needs_baseline():
    opt = _two()
    _record(opt, 1)
    with pytest.raises(ValueError, match="Bake time"):
        opt.add_process_parameter("Bake time", 8, 14)
    with pytest.raises(ValueError, match="20"):
        opt.add_process_parameter("Bake time", 8, 14, baseline=20)
    opt.add_process_parameter("Bake time", 8, 14, baseline=10)
    assert opt.variables[-1].absent == 10.0


def test_formulation_total_respected():
    opt = Optimizer(seed=1)
    for name in ("A", "B", "C"):
        opt.add_ingredient(name, 0, 60)
    opt.add_objective("Taste", weight=1)
    opt.set_formulation_total(100)
    total_limit = opt.quantity_limits[-1]
    assert total_limit.source == "formulation_total"
    assert total_limit.ingredients == ["A", "B", "C"]
    cold = opt.ask(5)
    for recipe in cold:
        assert sum(recipe.values()) == pytest.approx(100, abs=1e-9)
        opt.tell(recipe, {"Taste": recipe["A"] / 10})
    for recipe in opt.ask(2):
        assert 99.5 - 1e-6 <= sum(recipe.values()) <= 100.5 + 1e-6


def test_total_follows_a_new_ingredient():
    opt = _two()
    opt.set_formulation_total(8)
    opt.add_ingredient("Salt", 0, 1)
    assert opt.quantity_limits[-1].ingredients == ["Pea protein", "Oil", "Salt"]
    opt.clear_formulation_total()
    assert opt.formulation_total is None and opt.quantity_limits == []


def test_total_out_of_reach_refused():
    opt = Optimizer(seed=0)
    opt.add_ingredient("A", 1, 5)
    opt.add_ingredient("B", 1, 8)
    with pytest.raises(ValueError, match="13"):
        opt.set_formulation_total(20)
    with pytest.raises(ValueError, match="2"):
        opt.set_formulation_total(1)
    opt.add_quantity_constraint(["A", "B"], max_val=10)
    with pytest.raises(ValueError, match="10"):
        opt.set_formulation_total(12)
    assert opt.formulation_total is None


def test_total_of_nothing_refused():
    opt = Optimizer(seed=0)
    opt.add_ingredient("A", 0, 5)
    with pytest.raises(ValueError, match="0"):
        opt.set_formulation_total(0)


def test_ask_everything_fixed_refused():
    opt = _two()
    opt.fix("Pea protein", 2)
    opt.fix("Oil", 1)
    with pytest.raises(ValueError, match="Every variable is fixed"):
        opt.ask(1)


def test_ask_infeasible_limits_refused():
    opt = _two()
    opt.add_quantity_constraint(["Pea protein", "Oil"], min_val=20)
    with pytest.raises(ValueError, match="No formulation meets all the limits"):
        opt.ask(2)


def test_ask_infeasible_limits_refused_warm():
    opt = _two()
    _record(opt, 5)
    opt.add_quantity_constraint(["Pea protein", "Oil"], min_val=20)
    with pytest.raises(ValueError, match="No formulation meets all the limits"):
        opt.ask(1)


def test_ask_refused_while_property_values_missing():
    opt = _two()
    opt.set_property_values("Protein", {"Pea protein": 80})
    opt.add_constraint("Protein", min_val=5)
    with pytest.raises(ValueError, match="Oil"):
        opt.ask(1)
    opt.set_property_values("Protein", {"Oil": 0})
    for recipe in opt.ask(3):
        total = recipe["Pea protein"] + recipe["Oil"]
        assert 80 * recipe["Pea protein"] >= 5 * total - 1e-6


def test_fixing_that_breaks_a_limit_refused():
    opt = _two()
    opt.add_quantity_constraint(["Pea protein", "Oil"], max_val=6)
    opt.fix("Pea protein", 4)
    with pytest.raises(ValueError, match="Oil"):
        opt.fix("Oil", 3)
    assert opt.variables[1].high == 5.0


def test_bad_ranges_refused():
    opt = Optimizer(seed=0)
    with pytest.raises(ValueError, match="Coconut oil: min 6 is above max 5"):
        opt.add_ingredient("Coconut oil", 6, 5)
    opt.add_ingredient("Coconut oil", 0, 5)
    with pytest.raises(ValueError, match="coconut oil"):
        opt.add_ingredient("coconut oil", 0, 5)
    with pytest.raises(ValueError, match="Coconut oil"):
        opt.add_process_parameter("Coconut oil", 0, 5)
    with pytest.raises(ValueError, match="Pea"):
        opt.set_range("Pea", 0, 1)


def test_new_objective_after_results_refused():
    opt = _two()
    _record(opt, 1)
    with pytest.raises(ValueError,
                       match="Cannot add measurement 'Smell' after results are recorded"):
        opt.add_objective("Smell", weight=1)
    assert [o.name for o in opt.objectives] == ["Taste"]


def test_reweight_recomputes_scores():
    opt = _two()
    opt.add_objective("Texture", weight=1, goal="target", target=6)
    batch = opt.ask(2)
    opt.tell(batch[0], {"Taste": 8, "Texture": 6})
    opt.tell(batch[1], {"Taste": 2, "Texture": 10})
    before = [h["score"] for h in opt.history()]
    assert opt.add_objective("Taste", weight=3) is True
    after = [h["score"] for h in opt.history()]
    assert after != before
    assert after == [score(opt.objectives, h["results"]) for h in opt.history()]


def test_remove_objective_recomputes_scores_on_shares_of_100():
    opt = _two()
    opt.add_objective("Texture", weight=3)
    batch = opt.ask(1)
    opt.tell(batch[0], {"Taste": 8, "Texture": 6})
    opt.remove_objective("Texture")
    assert [o.weight for o in opt.objectives] == [100.0]
    assert opt.scores == [100.0 * 0.8]


def test_settings_snapped_to_step():
    opt = _two()
    opt.add_process_parameter("Bake time", 8, 14, step=1)
    for recipe in opt.ask(4):
        assert recipe["Bake time"] == round(recipe["Bake time"])
        assert 8 <= recipe["Bake time"] <= 14


def _thin(low=3.0, high=3.0005):
    opt = _two()
    opt.add_ingredient("Salt", 0, 4)
    opt.add_quantity_constraint(["Pea protein", "Oil"], min_val=low,
                                max_val=high)
    return opt


def test_thin_limit_fills_the_opening_round():
    opt = _thin()
    rows = opt.ask(5)
    assert len(rows) == 5
    for r in rows:
        assert 3.0 - 1e-6 <= r["Pea protein"] + r["Oil"] <= 3.0005 + 1e-6
    vectors = {tuple(round(r[k], 9) for k in sorted(r)) for r in rows}
    assert len(vectors) == 5


def test_feasible_sampler_only_when_the_pool_falls_short(monkeypatch):
    opt = _thin(1.0, 6.0)

    def _boom(*args, **kwargs):
        raise AssertionError("the pool fills this Round on its own")

    monkeypatch.setattr(opt, "_sample_feasible", _boom)
    assert len(opt.ask(5)) == 5


def test_feasible_sampler_leaves_the_torch_rng_alone():
    import torch
    opt = _thin()
    spans = search_bounds(opt.variables, [], [])
    before = torch.random.get_rng_state().clone()
    assert opt._sample_feasible(3, spans, bounds_tensor(spans), [])
    assert torch.equal(torch.random.get_rng_state(), before)


def test_feasible_sampler_failure_is_an_empty_list(monkeypatch):
    opt = _thin()
    spans = search_bounds(opt.variables, [], [])

    def _broken(spans):
        raise RuntimeError("no constraints today")

    monkeypatch.setattr(opt, "_linear_constraints", _broken)
    with pytest.warns(RuntimeWarning, match="feasible sampler failed: "
                                            "RuntimeError"):
        assert opt._sample_feasible(3, spans, bounds_tensor(spans), []) == []
