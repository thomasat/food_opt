"""Insights ftue, fix A (statistician's cold read 06, ledger rulings): the
statistical rules behind the status lines. Display only: no suggestion
changes.

- I5: the bias test uses t_{0.975, df} standard errors, not 2.
- I6: Rising needs the rise to beat noise SD x sqrt(2) x z_q.
- I7: a Repeat disagrees beyond sqrt(2) x 1.96 x sigma, sigma the larger of
  the model's fitted noise SD and the Repeat SD (3 df or more) without the
  pair being tested; the flagged pair stays in the reported Repeat SD.
- I9: cross-validation folds keep each Round, and each Repeat group, whole.
- I11: the Repeat SD counts from 3 degrees of freedom, not 3 pairs.

Fix A follow-up (owner's rulings on fix A's open decisions):
- 2: until the leave-one-out Repeat SD has 3 df, a Repeat disagrees only
  beyond BOTH sqrt(2) x 1.96 x max(model noise, the assumed 1 point) and
  sqrt(2) x its own Round-ahead half-width; from 3 df, sqrt(2) x 1.96 x
  that Repeat SD alone.
- 3: a noise SD from Repeats takes t_q = t_df^-1(0.95^(1/q)) for z_q.
- 4: Within noise and Rising name the threshold; ties use
  rise_threshold(yard, 1).
- Fix A's re-measured coverage and skill-gate figures on screen."""
import math
import re
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


# ------------------------------------------------------------------ #
#  I6: Rising beats noise SD × √2 × z_q, z_q = Φ⁻¹(0.95^(1/q))
# ------------------------------------------------------------------ #

def _rounds(scores_by_round):
    from tests.test_complete_results import _patty, _tell
    opt = _patty(rows=0)
    no = 0
    for r, scores in enumerate(scores_by_round, start=1):
        for y in scores:
            no += 1
            _tell(opt, no, r)
            opt.Y_history[-1] = y
    return opt


def test_the_rise_threshold():
    assert insights.rise_z(1) == pytest.approx(1.6449, abs=1e-4)
    assert insights.rise_threshold(1.0, 1) == pytest.approx(math.sqrt(2) * 1.6449, abs=1e-4)
    z4 = NormalDist().inv_cdf(0.95 ** 0.25)
    assert insights.rise_threshold(2.0, 4) == pytest.approx(2.0 * math.sqrt(2) * z4)
    assert insights.rise_threshold(1.0, 4) == pytest.approx(3.159, abs=1e-3)


def test_one_new_formulation_rises_past_root_two_times_1_645(_cwd):
    opt = _rounds([[70.0, 72.0], [74.3]])           # rise 2.3 < 2.33
    a = insights.improvement(opt)
    assert (a.state, a.q) == ("barely", 1)
    assert a.threshold == pytest.approx(2.326, abs=1e-3)
    a = insights.improvement(_rounds([[70.0, 72.0], [74.4]]))
    assert a.state == "yes"


def test_the_threshold_grows_with_the_round_size(_cwd):
    """Line C's case: a rise of 2.5 from the top of a Round of 4, against
    the assumed 1 point: under √2 × z_4 = 3.16, so Within noise."""
    opt = _rounds([[70.0, 72.0], [74.5, 60.0, 61.0, 62.0]])
    a = insights.improvement(opt, yardstick=insights.noise_yardstick(opt))
    assert (a.state, a.q, a.sd) == ("barely", 4, None)
    assert a.threshold == pytest.approx(3.159, abs=1e-3)
    opt = _rounds([[70.0, 72.0], [75.2, 60.0, 61.0, 62.0]])
    assert insights.improvement(opt).state == "yes"


def test_the_threshold_scales_with_the_repeat_sd(_cwd, monkeypatch):
    opt = _rounds([[86.0], [87.0]])
    monkeypatch.setattr(insights, "repeat_sd",
                        lambda o, name: insights.RepeatSpread(0.4, 3, 3, 3))
    a = insights.improvement(opt)
    # Fix A follow-up (owner, decision 3): an SD from Repeats on 3 df takes
    # t_{3}⁻¹(0.95) = 2.353 for z = 1.645: 0.93 -> 1.33, so 1.0 is now under.
    assert a.threshold == pytest.approx(0.4 * math.sqrt(2) * 2.353, abs=1e-3)
    assert a.state == "barely"                      # 1.0 < 1.33


def test_rising_fires_about_one_time_in_twenty_by_chance():
    """With q new formulations exactly as good as the old Best so far, the
    rise beats the threshold about 5 % of the time (a little under: the
    old best's own error is shared). The old 1-SD rule fired about half
    the time at q = 4 (cold read I6)."""
    import numpy as np
    rng = np.random.default_rng(1)
    n = 200_000
    for q in (1, 2, 4, 8):
        rise = rng.standard_normal((n, q)).max(1) - rng.standard_normal(n)
        rate = float(np.mean(rise >= insights.rise_threshold(1.0, q)))
        assert 0.035 < rate < 0.055, (q, rate)
    rise = rng.standard_normal((n, 4)).max(1) - rng.standard_normal(n)
    assert 0.48 < float(np.mean(rise >= 1.0)) < 0.53


# ------------------------------------------------------------------ #
#  I7: a Repeat disagrees beyond √2 × 1.96 × σ, σ = max(the model's
#  fitted noise SD, the Repeat SD from 3 df without the pair tested, else
#  the assumed 1 point)
# ------------------------------------------------------------------ #

def _score_check(no, rnd, value, sd=2.0, noise=None):
    low, high = value - insights.Z95 * sd, value + insights.Z95 * sd
    return insights.Check(no, rnd, insights.SCORE, value, low, high, value, True, True,
                          value, sd, noise=noise)


def _values(rnd, n=10, no=100):
    return [insights.Check(no + k, rnd, "Firmness", 5.1, 4.0, 6.0, 5.1, True, True, 5.0, 0.5)
            for k in range(n)]


def _pair(gap, noise, others=()):
    """Formulation 5 (row 4, Round 2) a Repeat of Formulation 1 (row 0),
    scoring `gap` against it; its Round-ahead check (sd 2, so the earlier
    rule's limit was √2 × 1.96 × 2 = 5.54) has fitted noise `noise`.
    `others`: (src, dst, gap) Repeats with no check of their own."""
    from tests.test_complete_results import _patty
    opt = _patty(rows=12)
    _repeat_of(opt, 0, 4)
    opt.Y_history[4] = opt.Y_history[0] + gap
    for src, dst, g in others:
        _repeat_of(opt, src, dst)
        opt.Y_history[dst] = opt.Y_history[src] + g
    checks = _values(2) + [_score_check(5, 2, opt.Y_history[4], noise=noise)]
    return opt, checks


LIMIT = math.sqrt(2) * insights.Z95


def test_the_limit_is_root_two_1_96_times_the_model_noise(_cwd):
    opt, checks = _pair(-6.0, noise=2.0)              # limit 5.54
    d = insights.disagreeing_repeat(opt, checks)
    assert (d.later, d.earlier, d.row) == (5, 1, 4)
    assert d.limit == pytest.approx(LIMIT * 2.0)
    assert (d.sigma, d.basis) == (pytest.approx(2.0), "model")
    opt, checks = _pair(-6.0, noise=2.5)              # limit 6.93
    assert insights.disagreeing_repeat(opt, checks) is None


def test_a_low_model_noise_yields_to_the_repeat_sd(_cwd):
    """Line A-expert's case: a low-noise prior no longer makes a Repeat
    disagree when other Repeats (3 df) show more spread."""
    others = [(1, 6, 3.0), (2, 7, -3.0), (3, 8, 3.0)]   # Repeat SD 2.12 on 3 df
    opt, checks = _pair(-4.0, noise=0.3, others=others)
    # 4.0 < √2 × t_{0.975, 3} × 2.12 = 9.54 (coordinator's ruling on 03
    # drift: t on the Repeats' df; 5.88 on 1.96).
    assert insights.disagreeing_repeat(opt, checks) is None
    # 2 df: the assumed 1 point, not the model's 0.3 (limit 2.77, not 0.83).
    # Fix A follow-up (decision 2): under 3 df the pair must also beat its
    # own predictive limit, here √2 × 1.96 × 1.0 = 2.77 too (check sd 1).
    opt, checks = _pair_sd(-4.0, noise=0.3, sd=1.0, others=others[:2])
    d = insights.disagreeing_repeat(opt, checks)
    assert d is not None and (d.basis, d.df) == ("assumed", 2)
    assert d.limit == pytest.approx(LIMIT * 1.0)


def test_lines_a_and_c_case_a_low_model_noise_alone_flags_nothing(_cwd):
    """Line A: F21 1.7 points under F15, the default model's fitted noise
    0.33, no other Repeat: against the assumed 1 point (limit 2.77) it does
    not disagree; against the model's noise alone (0.90) it would."""
    opt, checks = _pair(-1.74, noise=0.325)
    assert insights.disagreeing_repeat(opt, checks) is None


def test_the_pair_tested_is_left_out_of_sigma(_cwd):
    """Three quiet Repeats (3 df, SD 1.2) and one 10 points off: with the
    pair in, the Repeat SD (3.69 on 4 df) would set a limit of 10.2 and
    hide it; left out, σ = 1.2 and the limit √2 × t_{0.975, 3} × 1.2 =
    5.40 (t on the Repeats' df; 3.33 on 1.96)."""
    g = 1.2 * math.sqrt(2)
    others = [(1, 6, g), (2, 7, -g), (3, 8, g)]
    opt, checks = _pair(-10.0, noise=0.2, others=others)
    d = insights.disagreeing_repeat(opt, checks)
    assert d is not None and d.basis == "repeats" and d.df == 3
    assert d.sigma == pytest.approx(1.2) and d.repeat_sd == pytest.approx(1.2)
    assert d.limit == pytest.approx(math.sqrt(2) * insights.t_critical(3) * 1.2)


def test_the_flagged_pair_stays_in_the_reported_repeat_sd(_cwd):
    others = [(1, 6, 1.0), (2, 7, -1.0), (3, 8, 1.0)]
    opt, checks = _pair(-10.0, noise=0.5, others=others)
    assert insights.disagreeing_repeat(opt, checks) is not None
    whole = insights.repeat_sd(opt, insights.SCORE)
    y = insights.noise_yardstick(opt, checks)
    assert (y.pairs, y.df, y.left_out) == (4, 4, 0)
    assert y.sd == pytest.approx(whole.sd) and y.value == pytest.approx(whole.sd)
    [row] = [r for r in insights.details(opt, checks, flagged=insights.disagreeing_repeat(
        opt, checks)) if r.name == "Firmness"]
    assert row.left_out == 0


def test_without_a_fitted_noise_the_assumed_point_is_used(_cwd):
    """Fixed noise (Custom settings) fits no noise to judge by: σ is the
    assumed 1 point while Repeats are short (limit 2.77), whatever the
    check's own interval. Fix A follow-up (decision 2): under 3 df the pair
    must beat its own predictive limit as well, so the check here has sd 1
    (limit 2.77 both ways); at sd 2 (5.54) a 3-point pair is not flagged
    (test_under_three_df_a_repeat_must_beat_both_limits)."""
    opt, checks = _pair_sd(-3.0, noise=None, sd=1.0)
    d = insights.disagreeing_repeat(opt, checks)
    assert d is not None and (d.basis, d.noise) == ("assumed", None)
    assert d.limit == pytest.approx(LIMIT * 1.0)
    opt, checks = _pair_sd(-2.7, noise=None, sd=1.0)
    assert insights.disagreeing_repeat(opt, checks) is None


def test_each_check_carries_its_model_s_fitted_noise(_cwd):
    from tests.test_complete_results import _patty
    opt = _patty(rows=9)
    checks = insights.check_rounds(opt)
    score = [c for c in checks if c.name == insights.SCORE]
    assert score and all(c.noise is not None and c.noise > 0 for c in score)
    first = min(r for r, f in insights.round_order(opt) if f >= insights.MIN_ROWS)
    fitted = insights.fit(opt, insights.SCORE,
                          list(range(dict(insights.round_order(opt))[first])))
    assert [c.noise for c in score if c.round == first][0] == pytest.approx(
        insights.fitted_noise_sd(fitted.model))


# ------------------------------------------------------------------ #
#  I9: cross-validation folds hold out a Round, and a Repeat group, whole
# ------------------------------------------------------------------ #

def _held_sets(folds):
    return [sorted(held) for _, held in folds]


def test_units_are_dealt_to_folds_whole():
    units = ["a", "a", "b", "b", "b", "c", "d", "e", "f", "f"]
    got = insights.folds(units)
    # Six units over 5 folds, in turn: a and f share fold 0.
    assert _held_sets(got) == [[0, 1, 8, 9], [2, 3, 4], [5], [6], [7]]
    for kept, held in got:
        assert sorted(kept + held) == list(range(10))
        assert not {units[p] for p in held} & {units[p] for p in kept}


def test_few_units_make_few_folds():
    assert _held_sets(insights.folds(["x", "x", "y"])) == [[0, 1], [2]]
    assert insights.folds(["x", "x", "x"]) == []


def test_an_int_is_one_row_per_unit():
    assert _held_sets(insights.folds(7)) == [[0, 5], [1, 6], [2], [3], [4]]


def test_rounds_are_units_and_repeats_join_them(_cwd):
    from tests.test_complete_results import _patty
    opt = _patty(rows=9)                          # Rounds 1, 2, 3 of three
    units = insights.fold_units(opt, list(range(9)))
    assert len(set(units)) == 3
    assert units[0] == units[1] == units[2] and units[3] == units[4] == units[5]
    _repeat_of(opt, 0, 7)                         # a Round 3 Repeat of a Round 1 row
    units = insights.fold_units(opt, list(range(9)))
    assert len(set(units)) == 2 and units[0] == units[7] == units[8]
    assert _held_sets(insights.cv_folds(opt, range(9))) == [[0, 1, 2, 6, 7, 8], [3, 4, 5]]


def test_one_round_falls_back_to_rows_with_repeats_kept_together(_cwd):
    from tests.test_complete_results import _patty
    opt = _patty(rows=3)
    for i in range(3):
        opt.batch_history[i] = 1
    assert len(set(insights.fold_units(opt, [0, 1, 2]))) == 3
    _repeat_of(opt, 0, 2)
    units = insights.fold_units(opt, [0, 1, 2])
    assert units[0] == units[2] != units[1]


def test_the_widening_and_the_workers_use_the_round_folds(_cwd, monkeypatch):
    import insights_jobs
    from tests.test_complete_results import _patty
    opt = _patty(rows=9)
    seen = []
    real = insights._fit_gp
    monkeypatch.setattr(insights, "_fit_gp",
                        lambda o, name, rows: seen.append(list(rows)) or real(o, name, rows))
    insights._cv_z(opt, insights.SCORE, list(range(9)))
    assert seen == [[3, 4, 5, 6, 7, 8], [0, 1, 2, 6, 7, 8], [0, 1, 2, 3, 4, 5]]
    spec = insights_jobs.plan(opt)[0]
    folds = insights_jobs._requests(opt, spec)[1:]
    assert [len(f["Xt"]) for f in folds] == [3, 3, 3]


def test_the_model_key_follows_the_rounds_and_the_salt_moved(_cwd):
    from tests.test_complete_results import _patty
    opt = _patty(rows=6)
    key = insights.model_key(opt, insights.SCORE, list(range(6)))
    opt.batch_history[5] = 3
    assert insights.model_key(opt, insights.SCORE, list(range(6))) != key
    assert insights.MODEL_SALT != "0.8.0-speed-1"


# ------------------------------------------------------------------ #
#  Fix A follow-up (owner's rulings on fix A's open decisions 2-4)
# ------------------------------------------------------------------ #

def _pair_sd(gap, noise, sd, others=()):
    """_pair with the later one's Round-ahead check at predictive sd `sd`:
    its own predictive limit is √2 × 1.96 × sd."""
    opt, checks = _pair(gap, noise, others)
    c = checks[-1]
    checks[-1] = _score_check(c.formulation, c.round, c.measured, sd=sd, noise=noise)
    return opt, checks


def test_under_three_df_a_repeat_must_beat_both_limits(_cwd):
    """Ruling (decision 2): until the Repeat SD (pair left out) has 3 df,
    a pair is flagged only beyond BOTH √2 × 1.96 × max(model noise, the
    assumed 1 point) and the old limit from its own predictive half-width
    (√2 × 1.96 × 2 = 5.54 here). The limit it broke is the larger."""
    opt, checks = _pair(-3.0, noise=None)             # 3.0 > 2.77 but < 5.54
    assert insights.disagreeing_repeat(opt, checks) is None
    opt, checks = _pair(-6.0, noise=None)
    d = insights.disagreeing_repeat(opt, checks)
    assert d is not None and (d.basis, d.sigma) == ("assumed", pytest.approx(1.0))
    assert d.limit == pytest.approx(LIMIT * 2.0)
    assert d.predictive == pytest.approx(LIMIT * 2.0)
    # A narrow predictive interval does not lower the floor: 2.0 < 2.77.
    opt, checks = _pair_sd(-2.0, noise=None, sd=0.5)
    assert insights.disagreeing_repeat(opt, checks) is None
    opt, checks = _pair_sd(-3.0, noise=0.3, sd=0.5)
    d = insights.disagreeing_repeat(opt, checks)
    assert d is not None and d.limit == pytest.approx(LIMIT * 1.0)


def test_from_three_df_the_repeat_sd_alone_sets_the_limit(_cwd):
    """From 3 df: √2 × t_{0.975, df} × the leave-one-out Repeat SD alone
    (coordinator's ruling on 03 drift: t whenever the SD comes from
    Repeats); neither the model's noise (2.5) nor the predictive
    half-width (sd 2) counts. Limit √2 × 3.182 × 1.2 = 5.40."""
    g = 1.2 * math.sqrt(2)
    others = [(1, 6, g), (2, 7, -g), (3, 8, g)]
    opt, checks = _pair(-6.0, noise=2.5, others=others)
    d = insights.disagreeing_repeat(opt, checks)
    assert d is not None and (d.basis, d.df) == ("repeats", 3)
    assert d.limit == pytest.approx(math.sqrt(2) * insights.t_critical(3) * 1.2)
    assert d.limit == pytest.approx(5.40, abs=0.01)
    assert d.predictive is None
    # 5.0 beat √2 × 1.96 × 1.2 = 3.33, but not the t limit.
    opt, checks = _pair(-5.0, noise=2.5, others=others)
    assert insights.disagreeing_repeat(opt, checks) is None


def test_rise_threshold_uses_t_on_the_repeats_df():
    """Ruling (decision 3): a noise SD from Repeats with df degrees of
    freedom takes t_q = t_df⁻¹(0.95^(1/q)) for z_q; the assumed floor
    keeps z_q."""
    from scipy.special import stdtrit
    assert insights.rise_quantile(4) == pytest.approx(insights.rise_z(4))
    assert insights.rise_quantile(4, 3) == pytest.approx(float(stdtrit(3, 0.95 ** 0.25)))
    assert insights.rise_quantile(4, 3) == pytest.approx(4.146, abs=1e-3)
    assert insights.rise_threshold(1.2, 4, df=3) == pytest.approx(1.2 * math.sqrt(2) * 4.146,
                                                                  abs=1e-3)
    assert insights.rise_threshold(1.0, 4) == pytest.approx(3.159, abs=1e-3)
    yard = insights.Yardstick(1.2, 1.2, 3, True, df=3)
    assert insights.rise_threshold(yard, 1) == pytest.approx(
        1.2 * math.sqrt(2) * float(stdtrit(3, 0.95)))
    assumed = insights.Yardstick(1.0, None, 0, False, assumed=True)
    assert insights.rise_threshold(assumed, 1) == pytest.approx(math.sqrt(2) * 1.6449, abs=1e-4)


def test_improvement_uses_t_when_the_sd_is_from_repeats(_cwd, monkeypatch):
    c = insights.improvement(_rounds([[70.0, 72.0], [74.3]]))
    assert c.df == 0 and c.threshold == pytest.approx(2.326, abs=1e-3)
    opt = _rounds([[86.0], [87.0]])
    monkeypatch.setattr(insights, "repeat_sd",
                        lambda o, name: insights.RepeatSpread(0.4, 3, 3, 3))
    a = insights.improvement(opt)
    assert (a.q, a.df) == (1, 3)
    assert a.threshold == pytest.approx(0.4 * math.sqrt(2) * 2.353, abs=1e-3)
    assert a.state == "barely"                      # 1.0 < 1.33
    yard = insights.Yardstick(0.4, 0.4, 3, True, df=3)
    b = insights.improvement(opt, yardstick=yard)
    assert (b.df, b.threshold) == (3, pytest.approx(a.threshold))


def test_with_sd_from_three_df_rising_fires_at_most_one_time_in_twenty():
    """Fix A's I6 simulation with σ estimated from 3 df: with z it read
    Rising 13.7 % of the time at q = 4; with t_q it is about 3.3 % (Šidák
    on t is conservative: the q comparisons share one estimated SD)."""
    import numpy as np
    rng = np.random.default_rng(2)
    n = 200_000
    for q in (1, 4):
        rise = rng.standard_normal((n, q)).max(1) - rng.standard_normal(n)
        s = np.sqrt(rng.chisquare(3, n) / 3)
        rate = float(np.mean(rise >= insights.rise_threshold(s, q, df=3)))
        assert 0.025 < rate < 0.055, (q, rate)
        z_rate = float(np.mean(rise >= insights.rise_threshold(s, q)))
        assert z_rate > 0.09, (q, z_rate)


# Fix A follow-up (owner, decision 4): Within noise and Rising name the
# threshold the rise is judged against (Improvement.threshold, q), on
# screen and in the tooltip; ties use rise_threshold(yard, 1).


def _banned():
    from tests.test_food_bo import _BANNED
    return _BANNED


def test_ties_use_the_q1_rise_threshold(_cwd):
    """A lead of 2.0 is under rise_threshold(1 point, 1) = 2.33: a tie on
    Results and in Progress figures, as Rising would not read on it."""
    import ui_insights
    import ui_results
    opt = _rounds([[70.0, 72.0]])
    yard = insights.noise_yardstick(opt)
    assert insights.rise_threshold(yard, 1) == pytest.approx(2.326, abs=1e-3)
    assert ui_results._tie(opt, 0, 1) == (
        "Tied with Formulation 1: 2.00 apart, under the 2.3 points needed to differ.")
    assert ui_insights.tie_text(opt, yard).startswith(
        "Formulation 1 (70.00) is within the 2.3 points needed to differ")
    assert ui_insights.near_tie(opt, yard) == 1
    opt = _rounds([[70.0, 72.4]])
    assert ui_results._tie(opt, 0, 1) == ""
    assert ui_insights.tie_text(opt, insights.noise_yardstick(opt)) == ""


