import itertools
import math
import copy
import torch
import pytest
from food_bo import FoodOptimizer
import insights
import wording


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    # FoodOptimizer.tell()/set_amount_unit() call self.save(), and the default
    # LocalStorage backend writes "<project_name>.pkl" into the current
    # working directory (see tests/test_food_bo.py's own `opt` fixture for the
    # same pattern). Isolate every test here into its own tmp_path so this
    # file does not litter the worktree root with test .pkl files.
    monkeypatch.chdir(tmp_path)


_burger_no = itertools.count()


def _burger(rounds=4, per_round=3, noise=0.0, seed=0, fill=True, shift=0.0):
    """A small mixture project with a known smooth truth.
    Ingredients Pea 4–8 g, Oil 2–6 g, Water fills to 100 g (fill=True) or is
    an ordinary ingredient 80–94 g (fill=False, total 100 g, Piepel paths).
    Firmness (target 6 on 0–10) rises with Pea; Juiciness (max, 0–10) rises
    with Oil. `shift` is added to Firmness in the LAST round only (a truth
    that moves, for miss tests). Rounds are numbered 1..rounds."""
    g = torch.Generator().manual_seed(seed)
    # A fresh, distinct project name per call: FoodOptimizer.__init__ loads
    # any existing file of the same name from storage (LocalStorage writes
    # "<project_name>.pkl" into the cwd on every tell()/set_amount_unit()),
    # so two calls sharing one name within a test would have the second
    # silently resume the first's saved history instead of starting empty.
    opt = FoodOptimizer(f"insights_test_{next(_burger_no)}")
    opt.set_amount_unit("g")
    opt.add_ingredient("Pea", 4, 8)
    opt.add_ingredient("Oil", 2, 6)
    if fill:
        opt.add_ingredient("Water", 0, 100)
        opt.set_formulation_total(100)
        opt.set_balance("Water")
    else:
        opt.add_ingredient("Water", 80, 94)
        opt.set_formulation_total(100)
    opt.add_objective("Firmness", 60, goal="target", target=6, min_val=0, max_val=10)
    opt.add_objective("Juiciness", 40, goal="max", min_val=0, max_val=10)
    no = 0
    for r in range(1, rounds + 1):
        for k in range(per_round):
            u = torch.rand(2, generator=g).tolist()
            pea, oil = 4 + 4 * u[0], 2 + 4 * u[1]
            recipe = {"Pea": pea, "Oil": oil}
            if not fill:
                recipe["Water"] = 100 - pea - oil
            eps = (torch.randn(2, generator=g) * noise).tolist()
            firm = 2 + 0.9 * pea + eps[0] + (shift if r == rounds else 0.0)
            juicy = 1 + 1.2 * oil + eps[1]
            no += 1
            opt.tell(opt.fill_formulas(recipe) if fill else recipe,
                     {"Firmness": min(10, max(0, firm)),
                      "Juiciness": min(10, max(0, juicy))},
                     formulation_no=no, batch_no=r)
    return opt


def test_measurement_models_use_only_rows_with_that_measurement():
    opt = _burger(rounds=3)
    # a partial row: Juiciness only
    opt.tell(opt.fill_formulas({"Pea": 6.0, "Oil": 4.0}), {"Juiciness": 5.0},
             formulation_no=99, batch_no=4)
    rows = list(range(len(opt.Y_history)))
    firm = insights.fit(opt, "Firmness", rows)
    juicy = insights.fit(opt, "Juiciness", rows)
    assert firm.n_rows == len(rows) - 1
    assert juicy.n_rows == len(rows)
    assert firm.model is not None and firm.error is None


def test_fewer_than_five_rows_gives_no_model():
    opt = _burger(rounds=1, per_round=4)
    f = insights.fit(opt, "Firmness", list(range(4)))
    assert f.model is None and f.error is None and f.n_rows == 4


def test_prediction_band_is_clipped_to_the_range():
    opt = _burger(rounds=3, noise=0.5)
    f = insights.fit(opt, "Firmness", list(range(len(opt.Y_history))))
    preds = insights.predict(opt, f, [opt.X_history[0], opt.X_history[1]])
    for p in preds:
        assert 0.0 <= p.low <= p.mean <= p.high <= 10.0


def test_score_output_uses_utility_ceiling():
    opt = _burger(rounds=2)
    names = [n for n, _, _ in insights.outputs(opt)]
    assert names == ["Firmness", "Juiciness", insights.SCORE]
    assert insights.outputs(opt)[-1][2] == pytest.approx(opt.utility_ceiling())


def test_orphan_measurement_keys_are_ignored():
    opt = _burger(rounds=2)
    opt.results_history[0]["Old taste"] = 3.0   # a deleted measurement's key
    assert [n for n, _, _ in insights.outputs(opt)] == ["Firmness", "Juiciness", insights.SCORE]
    assert set(insights.fit_all(opt)) == {"Firmness", "Juiciness", insights.SCORE}


def test_expert_settings_still_fit():
    opt = _burger(rounds=3)
    opt.bo_config = dict(opt.DEFAULT_BO_CONFIG) if hasattr(opt, "DEFAULT_BO_CONFIG") else {}
    opt.bo_config.update({"noise": "fixed_tiny"})
    f = insights.fit(opt, "Firmness", list(range(len(opt.Y_history))))
    assert f.model is not None, f.error


def test_a_failed_fit_is_isolated(monkeypatch):
    opt = _burger(rounds=3)
    real = opt._build_gp
    def boom(X, Y, dim, tf):
        if Y.shape[0] and float(Y.max()) <= 10.0:   # measurement fits only
            raise RuntimeError("no fit")
        return real(X, Y, dim, tf)
    monkeypatch.setattr(opt, "_build_gp", boom)
    fitted = insights.fit_all(opt)
    assert fitted["Firmness"].model is None and "no fit" in fitted["Firmness"].error
    assert fitted[insights.SCORE].model is not None


def test_insights_leave_the_project_and_rng_untouched():
    opt = _burger(rounds=3)
    before = opt.export_json()
    rng = torch.get_rng_state().clone()
    fitted = insights.fit_all(opt)
    insights.predict(opt, fitted["Firmness"], [opt.X_history[0]])
    assert opt.export_json() == before
    assert torch.equal(torch.get_rng_state(), rng)


def test_insights_do_not_change_the_next_suggestion():
    a, b = _burger(rounds=3), _burger(rounds=3)
    insights.fit_all(b)
    assert a.ask(2) == b.ask(2)


def test_data_key_changes_with_results_and_settings():
    opt = _burger(rounds=2)
    k1 = insights.data_key(opt)
    opt.results_history[0]["Firmness"] = 1.0
    assert insights.data_key(opt) != k1


def test_round_order_skips_imports_and_orders_by_first_row():
    opt = _burger(rounds=2)
    opt.import_formulation(opt.fill_formulas({"Pea": 5.0, "Oil": 3.0}),
                           {"Firmness": 6.0, "Juiciness": 4.0})
    assert [r for r, _ in insights.round_order(opt)] == [1, 2]


def test_first_rounds_without_five_earlier_rows_are_not_checked():
    opt = _burger(rounds=3, per_round=3)          # rows 0-2 R1, 3-5 R2, 6-8 R3
    checks = insights.check_rounds(opt)
    assert {c.round for c in checks} == {3}       # R2 has only 3 earlier rows


def test_imports_train_but_are_never_checked():
    opt = _burger(rounds=2, per_round=3)
    for k in range(3):
        opt.import_formulation(opt.fill_formulas({"Pea": 5.0 + k, "Oil": 3.0}),
                               {"Firmness": 6.0, "Juiciness": 4.0})
    # third round after the imports: 9 earlier rows
    opt.tell(opt.fill_formulas({"Pea": 6.0, "Oil": 4.0}),
             {"Firmness": 7.4, "Juiciness": 5.8}, formulation_no=50, batch_no=3)
    checks = insights.check_rounds(opt)
    assert all(c.round is not None for c in checks)
    assert {c.formulation for c in checks if c.round == 3} == {50}


def test_smooth_truth_holds_up():
    opt = _burger(rounds=6, per_round=4, noise=0.05)
    h = insights.headline(opt, insights.check_rounds(opt))
    assert h.state == "verdict" and h.verdict == "holding", h


def test_a_truth_that_moves_misses():
    opt = _burger(rounds=6, per_round=5, noise=0.05, shift=3.0)
    h = insights.headline(opt, insights.check_rounds(opt))
    assert h.state == "verdict" and h.verdict in ("caution", "unreliable"), h


def test_early_when_fewer_than_ten_values():
    opt = _burger(rounds=3, per_round=3)          # only R3 checked: 3 rows x 2 = 6 values
    h = insights.headline(opt, insights.check_rounds(opt))
    assert h.state == "early" and h.total == 6


def test_none_when_nothing_is_checkable():
    opt = _burger(rounds=1, per_round=5)
    assert insights.headline(opt, insights.check_rounds(opt)).state == "none"


def test_binomial_tail_boundaries():
    assert insights.binomial_tail(0, 20) == pytest.approx(1.0)
    assert insights.binomial_tail(1, 20) == pytest.approx(1 - 0.95 ** 20)
    assert insights.binomial_tail(3, 20) < 0.10 <= insights.binomial_tail(2, 20)


def test_partial_rows_are_checked_per_measurement():
    opt = _burger(rounds=2, per_round=3)
    opt.tell(opt.fill_formulas({"Pea": 6.0, "Oil": 4.0}), {"Juiciness": 5.8},
             formulation_no=40, batch_no=3)
    checks = [c for c in insights.check_rounds(opt) if c.formulation == 40]
    assert {c.name for c in checks} == {"Juiciness", insights.SCORE}
    assert all(not c.fully_measured for c in checks)


def test_score_line_needs_ten_fully_measured_and_eighty_percent_below():
    mk = lambda below: insights.Check(1, 3, insights.SCORE, 50.0, 40.0, 60.0,
                                      45.0 if below else 55.0, True, True)
    opt = _burger(rounds=1)
    nine = [mk(True)] * 9
    assert not insights.headline(opt, nine)._score_rule_for_tests()
    ten = [mk(True)] * 8 + [mk(False)] * 2
    assert insights.headline(opt, ten)._score_rule_for_tests()
    ten_bad = [mk(True)] * 7 + [mk(False)] * 3
    assert not insights.headline(opt, ten_bad)._score_rule_for_tests()


def test_details_r2_negative_and_constant_cases():
    c = lambda meas, pred: insights.Check(1, 3, "Firmness", pred, pred - 1, pred + 1,
                                          meas, True, True)
    opt = _burger(rounds=1)
    rows = insights.details(opt, [c(5.0, 5.0)] * 5)
    assert rows[0].r2 is None                       # every measured value the same
    bad = [c(m, 10 - m) for m in (1.0, 2.0, 3.0, 4.0, 5.0)]
    assert insights.details(opt, bad)[0].r2 < 0


def _start(opt):
    return insights.start_recipe(opt, opt.best_index())


def test_fill_path_moves_only_the_choice_and_the_fill_row():
    opt = _burger(rounds=2)
    s = _start(opt)
    p = insights.build_path(opt, "Oil", s)
    assert p.mode == "fill" and p.fill_name == "Water"
    for r in p.recipes:
        assert r["Pea"] == pytest.approx(s["Pea"])
        assert r["Pea"] + r["Oil"] + r["Water"] == pytest.approx(100.0)
    assert p.amounts == sorted(p.amounts)
    assert p.low_end.amount == pytest.approx(2.0) and p.low_end.reason is None
    assert p.high_end.amount == pytest.approx(6.0) and p.high_end.reason is None


def test_proportional_path_keeps_the_total_and_ends_when_room_is_used():
    opt = _burger(rounds=2, fill=False)
    s = _start(opt)
    p = insights.build_path(opt, "Pea", s)
    assert p.mode == "proportional"
    for r in p.recipes:
        assert r["Pea"] + r["Oil"] + r["Water"] == pytest.approx(100.0, abs=1e-6)
        assert 2.0 - 1e-6 <= r["Oil"] <= 6.0 + 1e-6
        assert 80.0 - 1e-6 <= r["Water"] <= 94.0 + 1e-6


def test_property_limit_stops_the_path_exactly_and_is_named():
    opt = _burger(rounds=2)
    opt.add_property("Fat")     # set_property_value requires a known property
    for n, v in (("Pea", 2.0), ("Oil", 100.0), ("Water", 0.0)):
        opt.set_property_value(n, "Fat", v)
    opt.add_constraint("Fat", max_val=5.0)       # Oil ≤ ~4.9 g with Pea ≈ 4–8
    s = _start(opt)
    s = opt.fill_formulas({**s, "Oil": 2.5})
    p = insights.build_path(opt, "Oil", s)
    assert p.high_end.reason == opt.property_limit_text(opt.constraints[0])
    fat = lambda r: sum(r[n] * {"Pea": 2.0, "Oil": 100.0, "Water": 0.0}[n]
                        for n in ("Pea", "Oil", "Water")) / 100.0
    end = opt.fill_formulas({**s, "Oil": p.high_end.amount})
    assert fat(end) == pytest.approx(5.0, abs=1e-3)


def test_quantity_limit_is_named():
    opt = _burger(rounds=2)
    opt.add_quantity_constraint(["Pea", "Oil"], max_val=10.0)
    s = opt.fill_formulas({"Pea": 5.0, "Oil": 3.0})
    p = insights.build_path(opt, "Oil", s)
    assert p.high_end.reason == opt.limit_text(opt.quantity_constraints[-1])
    assert p.high_end.amount == pytest.approx(5.0, abs=1e-3)


def test_fill_row_below_zero_is_named():
    opt = _burger(rounds=2)
    opt.set_formulation_total(12)                # Pea 4–8 + Oil 2–6 can exceed 12
    s = opt.fill_formulas({"Pea": 7.0, "Oil": 3.0})
    p = insights.build_path(opt, "Oil", s)
    assert p.high_end.reason == wording.path_below_zero("Water")
    assert p.high_end.amount == pytest.approx(5.0, abs=1e-3)


def test_property_limit_with_missing_values_is_skipped_and_noted():
    opt = _burger(rounds=2)
    opt.add_property("Fat")     # set_property_value requires a known property
    opt.set_property_value("Oil", "Fat", 100.0)  # Pea and Water have no Fat value
    opt.add_constraint("Fat", max_val=5.0)
    p = insights.build_path(opt, "Oil", _start(opt))
    assert p.skipped and p.high_end.reason is None


def test_start_breaking_a_limit_draws_nothing():
    opt = _burger(rounds=2)
    opt.add_quantity_constraint(["Pea", "Oil"], max_val=8.0)
    s = opt.fill_formulas({"Pea": 7.0, "Oil": 3.0})
    p = insights.build_path(opt, "Oil", s)
    assert p.start_breaks == opt.limit_text(opt.quantity_constraints[-1])
    assert p.amounts == [] and p.recipes == []


def test_start_outside_lowest_highest_widens_the_path():
    opt = _burger(rounds=2)
    s = opt.fill_formulas({"Pea": 6.0, "Oil": 7.0})   # Oil Highest is 6
    p = insights.build_path(opt, "Oil", s)
    assert p.outside and max(p.amounts) >= 7.0


def test_resized_round_is_brought_back_to_the_total():
    opt = _burger(rounds=2)
    big = opt.fill_formulas({"Pea": 6.0, "Oil": 4.0})
    big = {k: v * 5 for k, v in big.items()}          # made at 500 g
    opt.tell(big, {"Firmness": 6.0, "Juiciness": 5.0}, formulation_no=70, batch_no=3)
    s = insights.start_recipe(opt, len(opt.Y_history) - 1)
    assert s["Pea"] == pytest.approx(6.0) and s["Oil"] == pytest.approx(4.0)


def test_path_choices_premixes_and_process():
    opt = _burger(rounds=2)
    opt.add_process_parameter("Mixing time", 45, 150, baseline=90, unit="s")
    # Fixing a brand-new ingredient straight at one amount is refused mid-run:
    # _check_fixed_feasible compares the "Total of each formulation" limit's
    # reach before/after, and that limit's own ingredient list (Pea, Oil,
    # Water) predates Salt, so fixing Salt at 1 g reads as shrinking the
    # reachable total by 1 g -- past the 0.5 % tolerance. Adding Salt with
    # room first, then re-running set_formulation_total (a real change, so
    # it rebuilds the limit over every current ingredient, Salt included)
    # folds Salt into that limit; the balance row (Water) then absorbs it,
    # so fixing Salt afterwards changes nothing the limit can see.
    opt.add_ingredient("Salt", 0, 5)
    opt.set_formulation_total(101)
    opt.set_formulation_total(100)
    opt.add_ingredient("Salt", 1, 1)                  # fixed: not offered
    assert insights.path_choices(opt) == ["Pea", "Oil", "Mixing time"]


def test_process_only_project_path():
    opt = FoodOptimizer("process_only")
    opt.add_process_parameter("Proof time", 30, 90, unit="min")
    opt.add_objective("Volume", 100, goal="max", min_val=0, max_val=10)
    for k, t in enumerate((30, 45, 60, 75, 90, 50)):
        opt.tell({"Proof time": float(t)}, {"Volume": 2 + t / 20},
                 formulation_no=k + 1, batch_no=1 + k // 3)
    p = insights.build_path(opt, "Proof time", insights.start_recipe(opt, 0))
    assert p.mode == "process" and p.fill_name is None
    assert p.low_end.amount == pytest.approx(30) and p.high_end.amount == pytest.approx(90)


def test_fixed_rows_never_move_and_are_listed():
    opt = _burger(rounds=2)
    # See test_path_choices_premixes_and_process for why Salt is added with
    # room first and folded into the total's own limit before being fixed.
    opt.add_ingredient("Salt", 0, 5)
    opt.set_formulation_total(101)
    opt.set_formulation_total(100)
    opt.add_ingredient("Salt", 1, 1)
    s = opt.fill_formulas({**_start(opt), "Salt": 1.0})
    p = insights.build_path(opt, "Oil", s)
    assert all(r["Salt"] == pytest.approx(1.0) for r in p.recipes)
    assert ("Salt", 1.0, "g") in p.fixed


def test_calculations_follow_and_are_listed():
    opt = _burger(rounds=2)
    opt.add_ingredient("Hydration", 0, 50)
    opt.set_formula("Hydration", "= 2 * Pea")
    s = _start(opt)
    p = insights.build_path(opt, "Pea", s)
    assert "Hydration" in p.follows
    assert all(r["Hydration"] == pytest.approx(2 * r["Pea"]) for r in p.recipes)


def test_path_predictions_cover_every_output():
    opt = _burger(rounds=3)
    fitted = insights.fit_all(opt)
    p = insights.build_path(opt, "Oil", _start(opt))
    preds = insights.path_predictions(opt, fitted, p)
    assert set(preds) == {"Firmness", "Juiciness", insights.SCORE}
    assert all(len(v) == len(p.amounts) for v in preds.values())


def test_build_path_writes_nothing_to_the_optimizer():
    opt = _burger(rounds=2)
    opt.add_property("Fat")
    opt.set_property_value("Pea", "Fat", 2.0)   # exercise the property-limit
    opt.add_constraint("Fat", max_val=5.0)      # and quantity-limit branches
    opt.add_quantity_constraint(["Pea", "Oil"], max_val=10.0)
    s = _start(opt)
    before = opt.export_json()
    insights.build_path(opt, "Oil", s)
    insights.build_path(opt, "Pea", s)
    after = opt.export_json()
    assert before == after
