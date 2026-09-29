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
