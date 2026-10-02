"""Insights ftue, fix A (statistician's cold read 06, ledger rulings): the
statistical rules behind the status lines. Display only: no suggestion
changes.

- I5: the bias test uses t_{0.975, df} standard errors, not 2.
- I6: Rising needs the rise to beat noise SD x sqrt(2) x z_q.
- I7: a Repeat disagrees beyond sqrt(2) x 1.96 x sigma, sigma the larger of
  the model's fitted noise SD and the Repeat SD (3 df or more) without the
  pair being tested; the flagged pair stays in the reported Repeat SD.
- I9: cross-validation folds keep each Round, and each Repeat group, whole.
- I11: the Repeat SD counts from 3 degrees of freedom, not 3 pairs."""
import math
from statistics import NormalDist
from types import SimpleNamespace

import pytest

import insights


def _check(no, rnd, measured, predicted, name=insights.SCORE, sd=2.5):
    return insights.Check(no, rnd, name, predicted, predicted - 5, predicted + 5, measured,
                          True, True, predicted, sd)


def _record(gaps_by_round):
    checks, no = [], 0
    for r, gaps in gaps_by_round.items():
        for g in gaps:
            no += 1
            checks.append(_check(no, r, 50.0 + g, 50.0))
    return insights.error_record(checks, insights.SCORE)


# ------------------------------------------------------------------ #
#  I5: the bias test on t, not 2 standard errors
# ------------------------------------------------------------------ #

def test_t_critical_values():
    assert insights.t_critical(3) == pytest.approx(3.182, abs=1e-3)
    assert insights.t_critical(5) == pytest.approx(2.571, abs=1e-3)
    assert insights.t_critical(100000) == pytest.approx(1.96, abs=1e-3)


def test_four_rounds_use_t_on_three_df_line_c_case():
    """Line C's Round means (−8.0, −5.4, −3.4, −0.6): t = 2.78, over 2 but
    under t_{0.975, 3} = 3.18, so no steady bias."""
    record = _record({2: [-8.0] * 4, 3: [-5.4] * 4, 4: [-3.4] * 4, 5: [-0.6] * 4})
    assert record.df == 3
    assert abs(record.mean_miss) / record.se == pytest.approx(2.78, abs=0.01)
    assert insights.bias_gap(record, 1.0) is None


def test_four_rounds_a_clear_bias_still_shows():
    record = _record({2: [-6.0] * 4, 3: [-5.0] * 4, 4: [-5.5] * 4, 5: [-4.5] * 4})
    assert insights.bias_gap(record, 1.0) == pytest.approx(-5.25)


def test_under_four_rounds_the_values_count_with_t_on_n_minus_1():
    """6 values in 2 Rounds: SE from the values, t_{0.975, 5} = 2.571.
    mean −3, SE 1.29: t = 2.32, over 2, under 2.571."""
    record = _record({2: [-7.0, 1.0, -3.0], 3: [-6.0, 0.0, -3.0]})
    assert record.df == 5
    assert record.se == pytest.approx(math.sqrt(10) / math.sqrt(6))
    assert insights.bias_gap(record, 1.0) is None
    record = _record({2: [-4.0, -2.0, -3.0], 3: [-3.5, -2.5, -3.0]})
    assert insights.bias_gap(record, 1.0) == pytest.approx(-3.0)


def test_the_noise_floor_still_applies():
    record = _record({2: [-0.9] * 4, 3: [-0.8] * 4, 4: [-0.9] * 4, 5: [-0.85] * 4})
    assert insights.bias_gap(record, 1.0) is None


# ------------------------------------------------------------------ #
#  I11: the Repeat SD counts from 3 degrees of freedom, not 3 pairs
# ------------------------------------------------------------------ #

def _repeat_of(opt, src, dst):
    """Row `dst` made as a Repeat of row `src`."""
    opt.recipe_history[dst] = dict(opt.recipe_history[src])
    opt.X_history[dst] = list(opt.X_history[src])


@pytest.fixture
def _cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _three_of_one():
    """Formulation 1 made three times: 3 pairs but 2 degrees of freedom."""
    from tests.test_complete_results import _patty
    opt = _patty(rows=6)
    for dst in (1, 2):
        _repeat_of(opt, 0, dst)
    for i, v in zip((0, 1, 2), (5.0, 5.8, 5.4)):
        opt.results_history[i]["Firmness"] = v
    opt._recompute_utilities()
    return opt


def test_three_pairs_on_two_df_is_not_yet_the_yardstick(_cwd):
    opt = _three_of_one()
    spread = insights.repeat_sd(opt, insights.SCORE)
    assert (spread.pairs, spread.df) == (3, 2)
    y = insights.noise_yardstick(opt)
    assert (y.value, y.from_repeats, y.assumed, y.df, y.pairs) == (1.0, False, True, 2, 3)
    firmness = insights.noise_yardstick(opt, name="Firmness")
    assert (firmness.value, firmness.from_repeats, firmness.df) == (None, False, 2)
    assert insights._noise_floor(opt, "Firmness") == 0.0
    assert insights._noise_floor(opt, insights.SCORE) == 1.0


def test_three_df_is_the_yardstick(_cwd):
    opt = _three_of_one()
    _repeat_of(opt, 3, 4)
    opt.results_history[4]["Firmness"] = opt.results_history[3]["Firmness"] + 0.6
    opt._recompute_utilities()
    y = insights.noise_yardstick(opt)
    assert (y.df, y.from_repeats, y.assumed) == (3, True, False)
    assert y.value == pytest.approx(insights.repeat_sd(opt, insights.SCORE).sd)
    assert insights.YARDSTICK_DF == 3


def test_the_details_row_carries_the_df_and_uses_the_df_gate(_cwd):
    opt = _three_of_one()
    checks = [insights.Check(k, 2, "Firmness", 5.0, 4.0, 6.0, 5.2, True, True, 5.0, 0.5)
              for k in range(10, 16)]
    [row] = insights.details(opt, checks)
    assert (row.pairs, row.df) == (3, 2)
    # 2 df: no floor for a measurement, so a 0.2 mean error beyond 2.57 SE
    # (every gap 0.2: SE 0) is a bias; from 3 df the Repeat SD (≥ 0.2)
    # would be the floor.
    assert row.bias is True


def test_the_noise_guard_judges_against_the_repeat_sd_from_three_df(_cwd, monkeypatch):
    """Trigger (b), fitted noise under half the Repeat SD, waits for 3 df."""
    monkeypatch.setattr(insights, "_lengthscale_at_floor", lambda gp: False)
    monkeypatch.setattr(insights, "fitted_noise_sd", lambda gp: 0.01)
    fitted = SimpleNamespace(model=object())
    opt = _three_of_one()
    guard = insights.noise_guard(opt, fitted, name="Firmness")
    assert "noise" not in guard.reasons
    _repeat_of(opt, 3, 4)
    opt.results_history[4]["Firmness"] = opt.results_history[3]["Firmness"] + 0.6
    opt._recompute_utilities()
    assert "noise" in insights.noise_guard(opt, fitted, name="Firmness").reasons
