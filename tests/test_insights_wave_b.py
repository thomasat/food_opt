"""Insights wave B, computation: predictions for the open Round, the spread
of each ingredient or process setting, and the formulation predicted to
score highest. Nothing here may write to the project or move the RNG."""
import itertools
import pickle

import pytest
import torch

import insights
import wording
from food_bo import CLOSE_TO_THE_BEST, FoodOptimizer
from ui_helpers import open_rows
from tests.test_insights import _burger


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    # tell() saves "<project_name>.pkl" into the cwd.
    monkeypatch.chdir(tmp_path)


_linear_no = itertools.count()


def _linear(n=8, seed=1):
    """Two free ingredients, Pea 0–10 g and Oil 0–10 g, no total, and one
    measurement whose truth is linear: Taste = 2·Pea + 0.5·Oil on 0–30. So
    Pea moves the Overall score four times as far as Oil does."""
    g = torch.Generator().manual_seed(seed)
    opt = FoodOptimizer(f"linear_{next(_linear_no)}")
    opt.set_amount_unit("g")
    opt.add_ingredient("Pea", 0, 10)
    opt.add_ingredient("Oil", 0, 10)
    opt.add_objective("Taste", 100, goal="max", min_val=0, max_val=30)
    for no in range(1, n + 1):
        pea, oil = (10 * torch.rand(2, generator=g)).tolist()
        opt.tell({"Pea": pea, "Oil": oil}, {"Taste": 2 * pea + 0.5 * oil},
                 formulation_no=no, batch_no=1 + (no - 1) // 4)
    return opt


def _close(p, q):
    """Two predictions equal up to float noise: one row predicted alone and
    the same row predicted beside others differ in the last bits."""
    return (p.mean, p.low, p.high) == pytest.approx((q.mean, q.low, q.high), rel=1e-9, abs=1e-9)


def _state(opt):
    # _forms_cache is a memo the calculations fill on first use, not
    # project data: reading a formulation may fill it.
    return pickle.dumps({k: v for k, v in opt.__dict__.items() if k != "_forms_cache"})


def _snapshot(opt):
    return _state(opt), torch.get_rng_state().clone()


def _unchanged(opt, snap):
    state, rng = snap
    return _state(opt) == state and torch.equal(torch.get_rng_state(), rng)


# ------------------------------------------------------------------ #
#  Task 1: predictions for the open Round
# ------------------------------------------------------------------ #

def _open_round(opt, recipes):
    opt.set_pending_batch([opt.fill_formulas(r) for r in recipes])
    return opt


def test_at_total_leaves_a_formulation_at_the_total_alone():
    opt = _burger(rounds=2)
    r = opt.fill_formulas({"Pea": 6.0, "Oil": 4.0})
    assert insights.at_total(opt, r) == pytest.approx(r)


def test_at_total_brings_a_resized_formulation_back():
    opt = _burger(rounds=2)
    big = {k: v * 1.5 for k, v in opt.fill_formulas({"Pea": 6.0, "Oil": 4.0}).items()}
    back = insights.at_total(opt, big)
    assert back["Pea"] == pytest.approx(6.0) and back["Oil"] == pytest.approx(4.0)
    assert back["Water"] == pytest.approx(90.0)


def test_round_predictions_equal_predict_for_each_row():
    opt = _open_round(_burger(rounds=3), [{"Pea": 5.0, "Oil": 3.0}, {"Pea": 7.0, "Oil": 5.0}])
    fitted = insights.fit_all(opt)
    got = insights.round_predictions(opt, fitted, open_rows(opt))
    assert sorted(got) == [r['formulation'] for r in opt.pending_batch]
    for row in opt.pending_batch:
        X = [opt._encode(row['recipe'])]
        for name, f in fitted.items():
            assert _close(got[row['formulation']][name], insights.predict(opt, f, X)[0])


def test_a_resized_round_is_predicted_at_the_formulation_total():
    opt = _open_round(_burger(rounds=3), [{"Pea": 5.0, "Oil": 3.0}, {"Pea": 7.0, "Oil": 5.0}])
    fitted = insights.fit_all(opt)
    before = insights.round_predictions(opt, fitted, open_rows(opt))
    opt.scale_round(150)
    assert opt.pending_batch[0]['recipe']["Pea"] == pytest.approx(7.5)
    after = insights.round_predictions(opt, fitted, open_rows(opt))
    for no in before:
        s0, s1 = before[no][insights.SCORE], after[no][insights.SCORE]
        assert _close(s0, s1)


def test_own_and_edited_rows_are_predicted():
    opt = _open_round(_burger(rounds=3), [{"Pea": 5.0, "Oil": 3.0}])
    rows = opt.pending_batch + [{"formulation": 99, "note": "our standard",
                                 "recipe": opt.fill_formulas({"Pea": 6.0, "Oil": 2.5})}]
    opt.set_pending_batch(rows)
    no = opt.pending_batch[0]['formulation']
    opt.edit_pending_formulations({no: {"Pea": 7.5}})
    fitted = insights.fit_all(opt)
    got = insights.round_predictions(opt, fitted, open_rows(opt))
    assert set(got) == {no, 99}
    edited = opt.pending_batch[0]['recipe']
    assert edited["Pea"] == pytest.approx(7.5)
    assert _close(got[no][insights.SCORE], insights.predict(
        opt, fitted[insights.SCORE], [opt._encode(edited)])[0])


def test_only_rows_not_yet_recorded_are_predicted():
    opt = _open_round(_burger(rounds=3),
                      [{"Pea": 5.0, "Oil": 3.0}, {"Pea": 6.0, "Oil": 4.0}, {"Pea": 7.0, "Oil": 5.0}])
    a, b, c = (r['formulation'] for r in opt.pending_batch)
    opt.tell(opt.pending_batch[0]['recipe'], {"Firmness": 6.0, "Juiciness": 5.0},
             formulation_no=a, batch_no=opt.pending_batch_no)
    opt.record_skipped(b, opt.pending_batch_no, opt.pending_batch[1]['recipe'])
    got = insights.round_predictions(opt, insights.fit_all(opt), open_rows(opt))
    assert set(got) == {c}


def test_a_measurement_without_a_model_is_left_out_of_the_row():
    opt = _open_round(_burger(rounds=3), [{"Pea": 5.0, "Oil": 3.0}])
    fitted = insights.fit_all(opt)
    fitted["Juiciness"] = insights.Fitted("Juiciness", None, None, 2, 0.0, 10.0)
    got = insights.round_predictions(opt, fitted, open_rows(opt))
    [row] = got.values()
    assert "Juiciness" not in row and "Firmness" in row and insights.SCORE in row


def test_no_open_rows_no_predictions():
    opt = _burger(rounds=3)
    assert insights.round_predictions(opt, insights.fit_all(opt), []) == {}


def test_round_predictions_leave_the_project_and_rng_untouched():
    opt = _open_round(_burger(rounds=3), [{"Pea": 5.0, "Oil": 3.0}])
    fitted = insights.fit_all(opt)
    snap = _snapshot(opt)
    insights.round_predictions(opt, fitted, open_rows(opt))
    assert _unchanged(opt, snap)


# ------------------------------------------------------------------ #
#  Task 3: which ingredients matter most
# ------------------------------------------------------------------ #

def test_the_steeper_ingredient_has_the_larger_spread():
    opt = _linear()
    fitted = insights.fit_all(opt)
    found = insights.spreads(opt, fitted, insights.start_recipe(opt, opt.best_index()))
    assert [s.name for s in found] == ["Pea", "Oil"]
    pea, oil = found[0].spread, found[1].spread
    assert pea > 2 * oil > 0


def test_a_spread_is_highest_minus_lowest_mean_along_the_path():
    opt = _linear()
    fitted = insights.fit_all(opt)
    start = insights.start_recipe(opt, opt.best_index())
    path = insights.build_path(opt, "Oil", start)
    means = [p.mean for p in insights.predict(
        opt, fitted[insights.SCORE], [opt._encode(r) for r in path.recipes])]
    oil = next(s for s in insights.spreads(opt, fitted, start) if s.name == "Oil")
    assert oil.spread == pytest.approx(max(means) - min(means))


def test_a_start_that_breaks_a_limit_is_left_out():
    opt = _burger(rounds=3)
    fitted = insights.fit_all(opt)
    opt.add_quantity_constraint(["Pea", "Oil"], max_val=8.0)
    start = opt.fill_formulas({"Pea": 7.0, "Oil": 3.0})
    assert insights.spreads(opt, fitted, start) == []


def test_no_overall_score_model_no_spreads():
    opt = _linear()
    fitted = insights.fit_all(opt)
    fitted[insights.SCORE] = insights.Fitted(insights.SCORE, None, None, 4, 0.0, 100.0)
    assert insights.spreads(opt, fitted, insights.start_recipe(opt, 0)) == []


def test_one_choice_gives_one_spread():
    opt = _linear()
    opt.add_ingredient("Oil", 5, 5)                     # fixed: not a choice
    fitted = insights.fit_all(opt)
    found = insights.spreads(opt, fitted, insights.start_recipe(opt, opt.best_index()))
    assert [s.name for s in found] == ["Pea"]


def test_equal_spreads_keep_the_path_choice_order(monkeypatch):
    opt = _linear()
    fitted = insights.fit_all(opt)
    monkeypatch.setattr(insights, "predict",
                        lambda o, f, X: [insights.Prediction(50.0, 40.0, 60.0) for _ in X])
    found = insights.spreads(opt, fitted, insights.start_recipe(opt, 0))
    assert [s.name for s in found] == ["Pea", "Oil"]
    assert all(s.spread == 0.0 for s in found)


def test_spreads_leave_the_project_and_rng_untouched():
    opt = _linear()
    fitted = insights.fit_all(opt)
    snap = _snapshot(opt)
    insights.spreads(opt, fitted, insights.start_recipe(opt, opt.best_index()))
    assert _unchanged(opt, snap)


#  Task 4: the formulation predicted to score highest
# ------------------------------------------------------------------ #

def test_predicted_best_lands_near_the_true_best():
    """Firmness hits its target 6 at Pea = 4.44 g; Juiciness rises with Oil,
    so the true best is Pea near 4.44 g and Oil at its Highest, 6 g."""
    opt = _burger(rounds=4)
    best = insights.predicted_best(opt, insights.fit_all(opt))
    assert isinstance(best, insights.PredictedBest)
    assert best.recipe["Oil"] >= 5.4
    assert abs(best.recipe["Pea"] - 4.44) <= CLOSE_TO_THE_BEST * 4
    assert opt._check_constraints(best.recipe)


def _limited(seed):
    """_burger with a property limit, a limit on two amounts and a process
    setting that moves in 10 °C steps, added after Round 3 at 170 °C."""
    opt = _burger(rounds=3, seed=seed, noise=0.2)
    opt.add_property("Fat")
    for name, fat in (("Pea", 2.0), ("Oil", 100.0), ("Water", 0.0)):
        opt.set_property_value(name, "Fat", fat)
    opt.add_constraint("Fat", max_val=5.0)
    opt.add_quantity_constraint(["Pea", "Oil"], max_val=11.0)
    opt.add_process_parameter("Temp", 150, 200, baseline=170, unit="°C", step=10)
    return opt


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_predicted_best_keeps_every_limit_the_total_and_the_steps(seed):
    opt = _limited(seed)
    best = insights.predicted_best(opt, insights.fit_all(opt))
    r = best.recipe
    assert opt._check_constraints(r)
    assert opt.ingredient_total(r) == pytest.approx(100.0, rel=opt.FORMULATION_TOTAL_TOLERANCE)
    assert r["Temp"] % 10 == pytest.approx(0.0, abs=1e-9) and 150 <= r["Temp"] <= 200
    assert 4 <= r["Pea"] <= 8 and 2 <= r["Oil"] <= 6


def test_predicted_best_keeps_a_fixed_row():
    opt = _burger(rounds=3)
    opt.add_ingredient("Oil", 3, 3)
    best = insights.predicted_best(opt, insights.fit_all(opt))
    assert best.recipe["Oil"] == pytest.approx(3.0)


def test_the_shown_score_is_predicted_again_and_clipped_to_the_ceiling():
    opt = _burger(rounds=4)
    fitted = insights.fit_all(opt)
    best = insights.predicted_best(opt, fitted)
    again = insights.predict(opt, fitted[insights.SCORE], [opt._encode(best.recipe)])[0]
    assert _close(best.score, again)
    assert 0.0 <= best.score.low <= best.score.mean <= best.score.high <= opt.utility_ceiling()


def test_a_mean_above_the_ceiling_is_shown_at_the_ceiling(monkeypatch):
    opt = _burger(rounds=4)
    fitted = insights.fit_all(opt)
    real = insights.predict

    def high(o, f, X):
        out = real(o, f, X)
        clip = lambda v: min(f.high, max(f.low, v))
        return [insights.Prediction(clip(p.mean + 50), clip(p.low + 50), clip(p.high + 50))
                for p in out]
    monkeypatch.setattr(insights, "predict", high)
    best = insights.predicted_best(opt, fitted)
    assert best.score.mean == opt.utility_ceiling() == best.score.high


def test_predicted_best_refuses_what_ask_refuses():
    opt = _burger(rounds=3)
    fitted = insights.fit_all(opt)
    opt.add_constraint("Fat", max_val=5.0)            # no ingredient has a Fat value
    assert insights.predicted_best(opt, fitted) == wording.missing_property_values(
        "Fat", "Pea, Oil and Water")
    opt = _burger(rounds=3)
    opt.add_premix("Spice mix", "portioned")
    assert insights.predicted_best(opt, fitted) == opt.premix_readiness_error()
    opt = _burger(rounds=3)
    opt.add_ingredient("Pea", 6, 6)
    opt.add_ingredient("Oil", 4, 4)
    assert insights.predicted_best(opt, fitted) == wording.EVERYTHING_IS_FIXED


def test_distance_is_the_largest_share_of_each_span():
    opt = _burger(rounds=2)                          # Pea 4–8, Oil 2–6: spans 4
    a = opt.fill_formulas({"Pea": 6.0, "Oil": 4.0})
    assert insights.distance(opt, a, opt.fill_formulas({"Pea": 6.6, "Oil": 4.2})) == pytest.approx(0.15)
    assert insights.distance(opt, a, opt.fill_formulas({"Pea": 6.0, "Oil": 3.0})) == pytest.approx(0.25)


def test_close_to_is_at_most_fifteen_hundredths(monkeypatch):
    opt = _burger(rounds=4)
    fitted = insights.fit_all(opt)
    monkeypatch.setattr(insights, "nearest", lambda o, r: (2, CLOSE_TO_THE_BEST))
    best = insights.predicted_best(opt, fitted)
    assert best.close == 2 and best.compare == 2
    monkeypatch.setattr(insights, "nearest", lambda o, r: (2, CLOSE_TO_THE_BEST + 0.001))
    best = insights.predicted_best(opt, fitted)
    assert best.close is None and best.compare == opt.best_index()


def test_nearest_uses_complete_rows_at_the_total_and_ties_go_to_the_lower_number():
    opt = _burger(rounds=0)
    r = opt.fill_formulas({"Pea": 6.0, "Oil": 4.0})
    opt.tell(opt.fill_formulas({"Pea": 6.4, "Oil": 4.0}), {"Firmness": 6.0, "Juiciness": 5.0},
             formulation_no=7, batch_no=1)
    opt.tell(opt.fill_formulas({"Pea": 5.6, "Oil": 4.0}), {"Firmness": 6.0, "Juiciness": 5.0},
             formulation_no=3, batch_no=1)
    opt.tell(r, {"Juiciness": 5.0}, formulation_no=9, batch_no=1)   # waiting: not a candidate
    big = {k: v * 2 for k, v in opt.fill_formulas({"Pea": 6.1, "Oil": 4.0}).items()}
    i, d = insights.nearest(opt, r)
    assert int(opt.formulation_ids[i]) == 3 and d == pytest.approx(0.1)
    opt.tell(big, {"Firmness": 6.0, "Juiciness": 5.0}, formulation_no=11, batch_no=2)
    i, d = insights.nearest(opt, r)                  # made at 200 g, compared at 100 g
    assert int(opt.formulation_ids[i]) == 11 and d == pytest.approx(0.025)


def _categorical(opt):
    # No public API makes a categorical row any more (old files only), so it
    # is appended the way tests/test_insights.py's path-choice test does.
    opt.variables.append({'name': "Package", 'type': 'categorical',
                          'options': ["Tray", "Pouch"], 'category': 'process'})
    return opt


def test_a_choice_that_differs_is_never_close():
    opt = _categorical(_burger(rounds=0))
    a = dict(opt.fill_formulas({"Pea": 6.0, "Oil": 4.0}), Package="Tray")
    assert insights.distance(opt, a, dict(a, Package="Pouch")) is None
    assert insights.distance(opt, a, dict(a)) == 0.0


def test_predicted_best_in_a_categorical_project():
    """An old file's categorical row: relaxed in the search, decoded to one
    of its options, never passed to a limit check as a number."""
    opt = _categorical(_linear(n=0))
    g = torch.Generator().manual_seed(3)
    for no in range(1, 9):
        pea, oil = (10 * torch.rand(2, generator=g)).tolist()
        pack = "Pouch" if no % 2 else "Tray"
        opt.tell({"Pea": pea, "Oil": oil, "Package": pack},
                 {"Taste": 2 * pea + 0.5 * oil + (3 if pack == "Pouch" else 0)},
                 formulation_no=no, batch_no=1 + (no - 1) // 4)
    best = insights.predicted_best(opt, insights.fit_all(opt))
    assert best.recipe["Package"] in ("Tray", "Pouch")
    assert 0 <= best.recipe["Pea"] <= 10 and 0 <= best.recipe["Oil"] <= 10


def test_predicted_best_leaves_the_project_rng_and_next_suggestion_untouched():
    a, b = _limited(0), _limited(0)
    fitted = insights.fit_all(b)
    snap = _snapshot(b)
    insights.predicted_best(b, fitted)
    assert _unchanged(b, snap)
    assert not hasattr(b, "_generation_premixes")
    assert a.ask(2) == b.ask(2)


def test_setup_key_follows_every_limit_but_not_a_rerun():
    opt = _burger(rounds=2)
    k = insights.setup_key(opt)
    assert insights.setup_key(opt) == k
    opt.add_quantity_constraint(["Pea", "Oil"], max_val=11.0)
    k2 = insights.setup_key(opt)
    assert k2 != k
    opt.add_property("Fat")
    opt.set_property_value("Oil", "Fat", 100.0)
    k3 = insights.setup_key(opt)
    assert k3 != k2
    opt.set_formulation_total(120)
    assert insights.setup_key(opt) != k3
