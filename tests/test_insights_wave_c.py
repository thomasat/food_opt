"""Insights wave C, computation for two of the three "More charts": two
measurements against each other (measured values only) and how far each
prediction missed, in standard deviations. Nothing here may write to the
project or move the RNG."""
import pickle

import pytest
import torch

import insights
from tests.test_insights import _burger


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    # tell() saves "<project_name>.pkl" into the cwd.
    monkeypatch.chdir(tmp_path)


def _state(opt):
    # _forms_cache is a memo the calculations fill on first use, not
    # project data: reading a formulation may fill it.
    return pickle.dumps({k: v for k, v in opt.__dict__.items() if k != "_forms_cache"})


# ------------------------------------------------------------------ #
#  Task 1a: the likely range keeps its unclipped mean and sd
# ------------------------------------------------------------------ #

def test_a_prediction_carries_its_unclipped_mean_and_sd():
    opt = _burger(rounds=3)
    fitted = insights.fit_all(opt)
    X = [opt._encode(opt.fill_formulas({"Pea": 6.0, "Oil": 4.0}))]
    for f in fitted.values():
        [p] = insights.predict(opt, f, X)
        clip = lambda v: min(f.high, max(f.low, v))
        assert p.sd > 0
        assert p.mean == pytest.approx(clip(p.raw_mean))
        assert p.low == pytest.approx(clip(p.raw_mean - insights.Z95 * p.sd))
        assert p.high == pytest.approx(clip(p.raw_mean + insights.Z95 * p.sd))


def test_callers_that_make_their_own_prediction_or_check_are_unaffected():
    p = insights.Prediction(78.4, 70.2, 85.49)
    assert (p.raw_mean, p.sd) == (None, None)
    c = insights.Check(1, 3, "Firmness", 5.0, 4.0, 6.0, 5.2, True, True)
    assert (c.raw_mean, c.sd) == (None, None)


def test_every_check_carries_the_prediction_it_was_made_from():
    opt = _burger(rounds=4, noise=0.3)
    checks = insights.check_rounds(opt)
    assert checks
    for c in checks:
        assert c.sd > 0
        assert c.low <= c.predicted <= c.high
        lo, hi = insights._range_of(opt, c.name)
        assert c.predicted == pytest.approx(min(hi, max(lo, c.raw_mean)))


# ------------------------------------------------------------------ #
#  Task 1b: how far each prediction missed
# ------------------------------------------------------------------ #

def _check(name="Firmness", measured=5.2, raw_mean=5.0, sd=0.5, inside=True, rnd=3, no=1):
    return insights.Check(no, rnd, name, 5.0, 4.0, 6.0, measured, inside, True, raw_mean, sd)


# ------------------------------------------------------------------ #
#  Task 1c: two measurements against each other
# ------------------------------------------------------------------ #

def test_pair_choices_lead_with_the_largest_share_of_score():
    opt = _burger(rounds=1)                  # Firmness 60, Juiciness 40
    assert insights.pair_choices(opt) == ["Firmness", "Juiciness"]
    opt.add_objective("Colour", 40, goal="max", min_val=0, max_val=10)
    shares = {o['name']: o['weight'] for o in opt.objectives}
    names = insights.pair_choices(opt)
    assert [shares[n] for n in names] == sorted(shares.values(), reverse=True)
    # Equal shares keep the project's order.
    ties = [n for n in names if shares[n] == shares["Juiciness"]]
    assert ties == [o['name'] for o in opt.objectives if o['name'] in ties]


def test_pair_points_are_every_formulation_with_both_values():
    opt = _burger(rounds=2)
    opt.add_objective("Cook loss", 20, goal="min", min_val=0, max_val=30, unit="%")
    # Recorded after Cook loss was added, without it: waiting for results.
    opt.tell(opt.fill_formulas({"Pea": 6.0, "Oil": 4.0}), {"Firmness": 6.0, "Juiciness": 5.0},
             formulation_no=50, batch_no=3)
    both = insights.pair_points(opt, "Firmness", "Juiciness")
    assert [p.formulation for p in both] == [int(n) for n in opt.formulation_ids]
    waiting = [p for p in both if p.formulation == 50]
    assert waiting[0].score is None
    assert all(p.score == pytest.approx(opt.Y_history[i])
               for i, p in enumerate(both) if p.formulation != 50)
    assert insights.pair_points(opt, "Firmness", "Cook loss") == []
    assert [p.best for p in both].count(True) == 1
    assert both[opt.best_index()].best


def test_a_value_outside_the_range_is_kept():
    opt = _burger(rounds=2)
    opt.tell(opt.fill_formulas({"Pea": 6.0, "Oil": 4.0}), {"Firmness": 12.5, "Juiciness": 5.0},
             formulation_no=70, batch_no=3)
    [p] = [p for p in insights.pair_points(opt, "Firmness", "Juiciness") if p.formulation == 70]
    assert p.a == 12.5


def test_the_pair_leaves_the_project_and_rng_untouched():
    opt = _burger(rounds=4, noise=0.3)
    state, rng = _state(opt), torch.get_rng_state().clone()
    insights.pair_points(opt, "Firmness", "Juiciness")
    assert _state(opt) == state and torch.equal(torch.get_rng_state(), rng)
