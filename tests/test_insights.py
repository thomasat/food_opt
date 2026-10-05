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
    # Waiting for Firmness: no Overall score to check it against yet.
    assert {c.name for c in checks} == {"Juiciness"}
    assert all(not c.fully_measured for c in checks)


def test_details_without_repeats_have_no_ratio():
    c = lambda meas, pred: insights.Check(1, 3, "Firmness", pred, pred - 1, pred + 1,
                                          meas, True, True)
    opt = _burger(rounds=1)
    rows = insights.details(opt, [c(5.0, 5.0)] * 5)
    assert rows[0].repeat_sd is None and rows[0].ratio is None


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


def _room_project():
    """A minimal three-ingredient proportional project where the OTHER
    ingredients' combined room (not the chosen row's own Lowest/Highest) is
    the true binding constraint in both directions: B and C only span 3-5 g
    each, so redistributing more than 4 g total between them runs out long
    before A's own declared 0-100 g range does."""
    opt = FoodOptimizer("insights_room_test")
    opt.set_amount_unit("g")
    opt.add_ingredient("A", 0, 100)
    opt.add_ingredient("B", 3, 5)
    opt.add_ingredient("C", 3, 5)
    opt.set_formulation_total(100)
    opt.add_objective("Score", 100, goal="max", min_val=0, max_val=10)
    return opt


def test_proportional_path_stops_when_the_others_run_out_of_room():
    # Ruling 3: when a proportional path ends because the OTHER ingredients
    # have used up their combined room, the end names that collectively
    # ("the other ingredients are all at their Highest"), not whichever one
    # of them happened to be checked first.
    opt = _room_project()
    start = {"A": 94.0, "B": 3.0, "C": 3.0}   # B, C already at their Lowest
    p = insights.build_path(opt, "A", start)
    assert p.mode == "proportional"
    # Decreasing A asks B and C to rise toward their Highest (5 each); their
    # combined room is (5-3)+(5-3) = 4, so A cannot go below 94-4 = 90 even
    # though A's own declared Lowest is 0.
    assert p.low_end.reason == wording.PATH_NO_ROOM_UP
    assert p.low_end.amount == pytest.approx(90.0, abs=1e-3)
    for r in p.recipes:
        assert r["A"] + r["B"] + r["C"] == pytest.approx(100.0, abs=1e-6)
        assert 3.0 - 1e-6 <= r["B"] <= 5.0 + 1e-6
        assert 3.0 - 1e-6 <= r["C"] <= 5.0 + 1e-6


def test_proportional_path_with_zero_room_at_the_start():
    # Ruling 3, the other direction of the same project: B and C are
    # already exactly at their Lowest, so increasing A has zero room from
    # the very first step -- the high end sits right at the start amount.
    opt = _room_project()
    start = {"A": 94.0, "B": 3.0, "C": 3.0}
    p = insights.build_path(opt, "A", start)
    assert p.high_end.reason == wording.PATH_NO_ROOM_DOWN
    assert p.high_end.amount == pytest.approx(94.0, abs=1e-3)


def test_a_moving_ingredient_recorded_outside_its_now_narrower_range_still_draws_a_path():
    # Ruling 4: C was recorded at 6 g before its allowed range was narrowed
    # to 3-5 g. Without widening C's own bounds to include its start (the
    # same rule the chosen row already gets), the bounds check in
    # _limits_broken would see C=6 > today's Highest=5 on every single
    # point of the path, including the start itself, and the path would
    # draw nothing at all.
    opt = _room_project()
    start = {"A": 90.0, "B": 4.0, "C": 6.0}
    p = insights.build_path(opt, "A", start)
    assert p.start_breaks is None
    assert len(p.amounts) > 1
    for r in p.recipes:
        assert r["A"] + r["B"] + r["C"] == pytest.approx(100.0, abs=1e-6)
    # path_past_lowest/path_past_highest for a MOVING other ingredient are
    # not reachable as an end reason here, or anywhere in proportional mode:
    # once ruling 3 makes _recipe_at report the aggregate "no room" wording
    # the instant a redistribution would ask more of the others than they
    # collectively have, no individual other ingredient's own bound can ever
    # be the first thing crossed (the algebra: redistributing delta <=
    # total_room never sends any single other ingredient past its own
    # share of that room). The two tests above exercise the wording this
    # collapses to; see the fix report for the fuller argument.
    assert p.low_end.reason in (None, wording.PATH_NO_ROOM_UP, wording.PATH_NO_ROOM_DOWN)
    assert p.high_end.reason in (None, wording.PATH_NO_ROOM_UP, wording.PATH_NO_ROOM_DOWN)


def test_property_min_limit_stops_the_path_and_is_named():
    opt = _burger(rounds=2)
    opt.add_property("Fat")
    for n, v in (("Pea", 2.0), ("Oil", 100.0), ("Water", 0.0)):
        opt.set_property_value(n, "Fat", v)
    opt.add_constraint("Fat", min_val=4.0)   # decreasing Oil drops Fat below 4
    s = opt.fill_formulas({"Pea": 6.0, "Oil": 4.0})
    p = insights.build_path(opt, "Oil", s)
    assert p.low_end.reason == opt.property_limit_text(opt.constraints[0])
    fat = lambda r: sum(r[n] * {"Pea": 2.0, "Oil": 100.0, "Water": 0.0}[n]
                        for n in ("Pea", "Oil", "Water")) / 100.0
    end = opt.fill_formulas({**s, "Oil": p.low_end.amount})
    assert fat(end) == pytest.approx(4.0, abs=1e-3)


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
    # A WEIGHED pre-mix: the parts (not the group) are rows, so each part is
    # its own choice on the path list, just like any other ingredient.
    opt.add_premix("Dry blend", "weighed")
    opt.set_premix_parts("Dry blend", [{"name": "Bran", "share": 60},
                                       {"name": "Fibre", "share": 40}])
    opt.add_ingredient("Bran", 1, 5)
    opt.add_ingredient("Fibre", 1, 5)
    # A PORTIONED pre-mix: the group is one row and its own choice; the
    # parts (Pepper, Cumin) are not rows at all, so they never appear.
    opt.add_premix("Spice mix", "portioned")
    opt.set_premix_parts("Spice mix", [{"name": "Pepper", "share": 50},
                                       {"name": "Cumin", "share": 50}])
    opt.add_ingredient("Spice mix", 1, 5)
    # A categorical variable is never offered. There is no public API left
    # to create one (the type is kept only for old files to migrate; see
    # test_food_bo.py's test_a_categorical_row_keeps_what_the_migration_
    # cannot_read for the same shape), so it is appended directly, the way
    # that test does.
    opt.variables.append({'name': "Package type", 'type': 'categorical',
                          'options': ["Tray", "Pouch"], 'category': 'process'})
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
    assert insights.path_choices(opt) == [
        "Pea", "Oil", "Bran", "Fibre", "Spice mix", "Mixing time"]


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
    # Every ingredient needs a value or ingredients_without_property("Fat")
    # is non-empty and the property-limit loop in _limits_broken skips
    # straight past _property_residual without calling it (see
    # test_property_limit_with_missing_values_is_skipped_and_noted) -- this
    # test means to actually exercise that call, not skip it.
    for n, v in (("Pea", 2.0), ("Oil", 100.0), ("Water", 0.0)):
        opt.set_property_value(n, "Fat", v)
    opt.add_constraint("Fat", max_val=5.0)      # exercise the property-limit
    opt.add_quantity_constraint(["Pea", "Oil"], max_val=10.0)  # and quantity-limit branches
    s = _start(opt)
    before = opt.export_json()
    insights.build_path(opt, "Oil", s)
    insights.build_path(opt, "Pea", s)
    after = opt.export_json()
    assert before == after


def test_best_facts_counts_rounds_since():
    opt = _burger(rounds=3, per_round=2)
    i = opt.best_index()
    f = insights.best_facts(opt)
    assert f.formulation == int(opt.formulation_ids[i])
    assert f.round == opt.batch_history[i]
    assert f.rounds_since == 3 - f.round
    assert f.latest == (f.round == 3)


def test_best_facts_for_an_import():
    opt = FoodOptimizer("imp")
    opt.add_ingredient("Pea", 0, 10)
    opt.add_objective("Firmness", 100, goal="max", min_val=0, max_val=10)
    opt.import_formulation({"Pea": 5.0}, {"Firmness": 9.5})
    opt.tell({"Pea": 1.0}, {"Firmness": 2.0}, formulation_no=10, batch_no=1)
    f = insights.best_facts(opt)
    assert f.round is None and f.rounds_since == 1 and not f.latest


def test_repeat_sd_of_one_pair():
    opt = _burger(rounds=2, per_round=3)
    assert insights.repeat_sd(opt, insights.SCORE) == insights.RepeatSpread(None, 0)
    same = opt._recorded_recipe(0)
    opt.tell(same, {"Firmness": 3.0, "Juiciness": 1.0}, formulation_no=80, batch_no=3)
    spread = insights.repeat_sd(opt, insights.SCORE)
    assert spread.sd == pytest.approx(abs(opt.Y_history[0] - opt.Y_history[-1]) / 2 ** 0.5)
    assert spread.pairs == 1


def test_repeat_tolerance_is_half_a_percent_of_the_span():
    opt = _burger(rounds=1, per_round=1)
    r = opt._recorded_recipe(0)
    near = opt.fill_formulas({"Pea": r["Pea"] + 0.004 * 4, "Oil": r["Oil"]})   # 0.4 % of 4 g
    far = opt.fill_formulas({"Pea": r["Pea"] + 0.02 * 4, "Oil": r["Oil"]})     # 2 %
    opt.tell(near, {"Firmness": 1.0, "Juiciness": 1.0}, formulation_no=81, batch_no=2)
    assert insights.repeat_sd(opt, insights.SCORE).pairs == 1
    opt2 = _burger(rounds=1, per_round=1)
    opt2.tell(far, {"Firmness": 1.0, "Juiciness": 1.0}, formulation_no=82, batch_no=2)
    assert insights.repeat_sd(opt2, insights.SCORE).pairs == 0


def test_repeat_groups_join_a_chain_of_pairs():
    # RULING (fix round 1, finding 1), kept for repeat_groups: every PAIR is
    # compared independently, not each new row against a group's first
    # member only. Pea's span is 4-8 g (4 g), tolerance = 0.005 * 4 = 0.02 g.
    # Row 0 (the recorded start) and row 1 (+0.015 g) match (diff 0.015 g).
    # Row 1 and row 2 (+0.030 g from row 0, i.e. +0.015 g from row 1) also
    # match (diff 0.015 g). Row 0 and row 2 do NOT match (diff 0.030 g).
    # An anchor-group scheme puts row 1 in row 0's group on that first match,
    # then tests row 2 only against the group's anchor (row 0), sees no
    # match, and starts a new group for row 2 alone -- so it never learns
    # that (1, 2) match at all. Row 1 and row 0 are told the SAME measured
    # result (so their Overall scores are equal, spread 0); row 2 is told a
    # very different result, so the only real spread -- and the one an
    # anchor-group scheme would miss entirely -- is between rows 1 and 2.
    opt = _burger(rounds=1, per_round=1)
    r = opt._recorded_recipe(0)
    row1 = opt.fill_formulas({"Pea": r["Pea"] + 0.015, "Oil": r["Oil"]})
    row2 = opt.fill_formulas({"Pea": r["Pea"] + 0.030, "Oil": r["Oil"]})
    row0_results = dict(opt.results_history[0])
    opt.tell(row1, row0_results, formulation_no=90, batch_no=2)
    opt.tell(row2, {"Firmness": 10.0, "Juiciness": 10.0}, formulation_no=91, batch_no=2)
    y0, y1, y2 = opt.Y_history[0], opt.Y_history[1], opt.Y_history[2]
    assert y0 == pytest.approx(y1)                 # rows 0 and 1: same result, no spread
    assert abs(y1 - y2) > abs(y0 - y1)              # the missed pair has the real spread
    assert insights.repeat_groups(opt, [0, 1, 2]) == [[0, 1, 2]]
    ys = [y0, y1, y2]
    mean = sum(ys) / 3
    assert insights.repeat_sd(opt, insights.SCORE).sd == pytest.approx(
        (sum((y - mean) ** 2 for y in ys) / 2) ** 0.5)


def test_same_fixed_row_requires_exact_match():
    # RULING (fix round 1, finding 3): a fixed row's tolerance is 0 (its
    # Lowest equals its Highest, so the usual "0.5 % of the span" rule would
    # divide by zero) -- decided via opt.is_fixed rather than re-deriving
    # "hi == lo" locally. `insights._same` is exercised directly because
    # `repeat_groups`' own pipeline (via `start_recipe` -> `_recorded_recipe`
    # -> decode) always snaps a fixed row back to its one allowed amount, so
    # two real formulations could never actually disagree on it -- there is
    # no way to reach this branch with a "slightly different" value through
    # the public entry point.
    opt = _burger(rounds=2)
    # See test_path_choices_premixes_and_process for why Salt is added with
    # room first and folded into the total's own limit before being fixed.
    opt.add_ingredient("Salt", 0, 5)
    opt.set_formulation_total(101)
    opt.set_formulation_total(100)
    opt.add_ingredient("Salt", 1, 1)               # fixed: Lowest == Highest
    a = {**opt.fill_formulas({"Pea": 6.0, "Oil": 4.0}), "Salt": 1.0}
    b = {**opt.fill_formulas({"Pea": 6.0, "Oil": 4.0}), "Salt": 1.0001}
    assert not insights._same(opt, a, b)


class _FakePosterior:
    def __init__(self, mean, var):
        self.mean = torch.tensor([[mean]], dtype=torch.double)
        self.variance = torch.tensor([[var]], dtype=torch.double)


class _FakeModel:
    def __init__(self, mean, var):
        self._p = _FakePosterior(mean, var)

    def posterior(self, X, observation_noise=False):
        return self._p


def test_band_is_worked_from_the_raw_mean_then_clipped():
    # Final fix G7: a mean past the Range's top keeps its band where the
    # model put it; only then is each end (and the shown mean) clipped.
    opt = _burger(rounds=1, per_round=1)
    f = insights.Fitted("Firmness", _FakeModel(11.0, 1.0), None, 5, 0.0, 10.0)
    p = insights.predict(opt, f, [opt.X_history[0]])[0]
    assert p.mean == 10.0 and p.high == 10.0
    assert p.low == pytest.approx(11.0 - insights.Z95)


# ------------------------------------------------------------------ #
#  Final fix wave D, E, F
# ------------------------------------------------------------------ #

def test_a_partly_measured_repeat_gives_no_spread():
    # D: a missing measurement counts as 0 in the Overall score, so a repeat
    # nobody finished measuring would read as a huge spread.
    opt = _burger(rounds=2, per_round=3)
    opt.tell(opt._recorded_recipe(0), {"Juiciness": 1.0}, formulation_no=80, batch_no=3)
    assert not opt.fully_measured(len(opt.Y_history) - 1)
    assert insights.repeat_sd(opt, insights.SCORE).sd is None


def test_a_fully_measured_repeat_beside_a_partial_one():
    opt = _burger(rounds=2, per_round=3)
    same = opt._recorded_recipe(0)
    opt.tell(same, {"Juiciness": 1.0}, formulation_no=80, batch_no=3)
    opt.tell(same, {"Firmness": 3.0, "Juiciness": 1.0}, formulation_no=81, batch_no=3)
    assert insights.repeat_sd(opt, insights.SCORE).sd == pytest.approx(
        abs(opt.Y_history[0] - opt.Y_history[-1]) / 2 ** 0.5)


def test_a_failed_prediction_is_treated_like_a_failed_fit(monkeypatch):
    # E: one (Round, output) that cannot be predicted gives no checks for
    # that pair; every other pair is still checked.
    opt = _burger(rounds=4, per_round=3)
    real = insights.predict
    def boom(o, fitted, rows):
        if fitted.name == "Juiciness":
            raise RuntimeError("no predict")
        return real(o, fitted, rows)
    monkeypatch.setattr(insights, "predict", boom)
    checks = insights.check_rounds(opt)
    assert {c.name for c in checks} == {"Firmness", insights.SCORE}


def test_details_has_a_row_for_every_output_checked_at_least_once():
    # F1: never silently empty; no ratio without Repeats.
    c = lambda name, meas: insights.Check(1, 3, name, 5.0, 4.0, 6.0, meas, True, True)
    opt = _burger(rounds=1)
    rows = insights.details(opt, [c("Firmness", 5.5)] * 3 + [c("Juiciness", m) for m in
                                                            (1.0, 2.0, 3.0, 4.0, 5.0)])
    assert [(r.name, r.checked) for r in rows] == [("Firmness", 3), ("Juiciness", 5)]
    assert rows[0].ratio is None and rows[1].ratio is None      # no Repeats in _burger
    assert insights.details(opt, []) == []


def test_rounds_not_checked_are_the_ones_without_five_rows_before_them():
    # F2
    opt = _burger(rounds=4, per_round=2)          # R1 0 before, R2 2, R3 4, R4 6
    assert insights.rounds_not_checked(opt) == [1, 2, 3]
    assert {c.round for c in insights.check_rounds(opt)} == {4}
    opt = _burger(rounds=2, per_round=5)          # R2 has 5 before it
    assert insights.rounds_not_checked(opt) == [1]


def test_setting_choice_is_known():
    # G2
    opt = _burger(rounds=1)
    assert not insights.has_setting_choice(opt)
    opt.add_process_parameter("Mixing time", 45, 150, baseline=90, unit="s")
    assert insights.has_setting_choice(opt)


# ------------------------------------------------------------------ #
#  Final fix wave A (part 2): the likely range is calibrated by 5-fold
#  cross-validation, refitting the model on each fold
# ------------------------------------------------------------------ #

def test_likely_range_holds_about_19_in_20_at_realistic_noise():
    """Noise sd 0.4 on a 0-10 Range (4 % of it), 4 per Round: the point-fitted
    noise alone left Round-ahead coverage near 78 % and called good models
    "not reliable". Widening by the cross-validated misses brings it to ~95 %."""
    rates, unreliable = [], 0
    for seed in range(8):
        opt = _burger(rounds=6 if seed % 2 else 5, per_round=4, noise=0.4, seed=seed)
        checks = insights.check_rounds(opt)
        values = [c for c in checks if c.name != insights.SCORE]
        rates.append(sum(c.inside for c in values) / len(values))
        unreliable += insights.headline(opt, checks).verdict == "unreliable"
    assert sum(rates) / len(rates) >= 0.88, rates
    assert unreliable <= 1, rates


def test_widening_only_ever_widens():
    for noise in (0.0, 0.05, 0.4):
        opt = _burger(rounds=4, per_round=4, noise=noise)
        for f in insights.fit_all(opt).values():
            assert f.widen >= 1.0 and math.isfinite(f.widen)


def test_widening_is_one_when_held_out_misses_are_small(monkeypatch):
    opt = _burger(rounds=3, noise=0.4)
    monkeypatch.setattr(insights, "_cv_z", lambda opt, name, use: [0.1, -0.2, 0.3])
    assert insights.fit(opt, "Firmness", list(range(len(opt.Y_history)))).widen == 1.0


def test_widening_is_the_rms_of_the_held_out_misses(monkeypatch):
    opt = _burger(rounds=3, noise=0.4)
    monkeypatch.setattr(insights, "_cv_z", lambda opt, name, use: [2.0, -2.0, 2.0, -2.0])
    f = insights.fit(opt, "Firmness", list(range(len(opt.Y_history))))
    assert f.widen == pytest.approx(2.0)
    plain = insights.Fitted(f.name, f.model, None, f.n_rows, f.low, f.high)
    X = [opt.X_history[0]]
    wide, narrow = insights.predict(opt, f, X)[0], insights.predict(opt, plain, X)[0]
    assert wide.mean == pytest.approx(narrow.mean)
    if 0.0 < wide.low and wide.high < 10.0:
        assert wide.high - wide.low == pytest.approx(2 * (narrow.high - narrow.low))


def test_folds_are_by_row_order_and_each_row_is_held_out_once(monkeypatch):
    """Ftue fix A (cold read I9): a Round at a time, each row held out once
    (before: row k in fold k % 5)."""
    opt = _burger(rounds=3, noise=0.4)                     # 9 rows, 3 Rounds
    seen = []
    real = insights._fit_gp
    def spy(opt_, name, rows):
        seen.append(list(rows))
        return real(opt_, name, rows)
    monkeypatch.setattr(insights, "_fit_gp", spy)
    use = list(range(9))
    z = insights._cv_z(opt, "Firmness", use)
    assert len(z) == 9
    rounds = [opt.batch_history[i] for i in use]
    assert seen == [[i for i in use if rounds[i] != r] for r in dict.fromkeys(rounds)]
    assert insights._cv_z(opt, "Firmness", use) == pytest.approx(z)   # repeatable


def test_held_out_misses_match_a_hand_refit():
    opt = _burger(rounds=3, noise=0.4)
    use = list(range(9))
    z = insights._cv_z(opt, "Juiciness", use)
    held = [0, 1]                       # fold 0: Round 1 (ftue fix A, I9)
    assert insights.cv_folds(opt, use)[0][1][:2] == held
    first = opt.batch_history[0]
    gp = insights._fit_gp(opt, "Juiciness", [i for i in use if opt.batch_history[i] != first])
    f = insights.Fitted("Juiciness", gp, None, 7, -1e9, 1e9)
    from botorch.utils.transforms import normalize
    with torch.no_grad():
        post = gp.posterior(normalize(torch.tensor([opt.X_history[i] for i in held],
                                                   dtype=torch.double), opt._get_bounds()),
                            observation_noise=True)
    y = torch.tensor([insights.measured(opt, "Juiciness", i) for i in held], dtype=torch.double)
    want = ((y - post.mean.reshape(-1)) / post.variance.sqrt().reshape(-1)).tolist()
    assert z[:2] == pytest.approx(want, rel=1e-6)


def test_a_failed_fold_is_skipped_and_no_folds_means_no_widening(monkeypatch):
    opt = _burger(rounds=3, noise=0.4)
    n = len(opt.Y_history)
    real = opt._build_gp
    def only_full(X, Y, dim, tf):
        if Y.shape[0] < n:
            raise RuntimeError("fold fit")
        return real(X, Y, dim, tf)
    monkeypatch.setattr(opt, "_build_gp", only_full)
    assert insights._cv_z(opt, "Firmness", list(range(n))) == []
    f = insights.fit(opt, "Firmness", list(range(n)))
    assert f.model is not None and f.error is None and f.widen == 1.0


def test_widening_with_fixed_tiny_noise():
    opt = _burger(rounds=3, noise=0.4)
    opt.bo_config = {"noise": "fixed_tiny"}
    f = insights.fit(opt, "Firmness", list(range(len(opt.Y_history))))
    assert f.model is not None, f.error
    assert f.widen >= 1.0 and math.isfinite(f.widen)


def test_checks_leave_the_project_rng_and_next_suggestion_untouched():
    a, b = _burger(rounds=3, noise=0.4), _burger(rounds=3, noise=0.4)
    before = b.export_json()
    rng = torch.get_rng_state().clone()
    insights.fit_all(b)
    insights.check_rounds(b)
    assert torch.equal(torch.get_rng_state(), rng)
    assert b.export_json() == before
    assert a.ask(2) == b.ask(2)
