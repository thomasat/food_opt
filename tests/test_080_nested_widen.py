"""0.8.0 speed, Plan A Task 2 (review I1): each Round's Overall-score check
is widened by score_widen of the checks of the Rounds before it, so the
record counts the range Make a round showed and stays before the fact."""
import dataclasses
import math
from types import SimpleNamespace

import pytest

import insights
from tests.test_080_trust import _check, _repeat_of, _values
from tests.test_complete_results import _patty
from tests.test_insights import _burger


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


TOP = SimpleNamespace(utility_ceiling=lambda: 100.0)


def _score(rnd, no, z, sd=1.0, mean=50.0):
    """An Overall-score check whose miss is `z` sd."""
    return _check(insights.SCORE, rnd, no, mean + z * sd, mean, sd)


def _round(rnd, z, n=5, no=0):
    return [_score(rnd, no + k, z) for k in range(n)]


def test_the_first_rounds_are_left_as_fitted():
    checks = _round(1, 2.0) + _round(2, 2.0, no=10)
    out = insights.nest_score_widen(TOP, checks)
    assert [c.widen for c in out if c.round == 1] == [1.0] * 5


def test_a_round_is_widened_by_the_rms_miss_of_the_rounds_before_it():
    checks = _round(1, 2.0) + _round(2, 0.5, no=10) + _round(3, 0.0, no=20)
    out = insights.nest_score_widen(TOP, checks)
    assert [c.widen for c in out if c.round == 2] == pytest.approx([2.0] * 5)
    # Round 3: the rms of Rounds 1 and 2 (z 2.0 five times, 0.5 five times).
    assert [c.widen for c in out if c.round == 3] == pytest.approx(
        [(0.5 * (4 + 0.25)) ** 0.5] * 5)
    c = next(c for c in out if c.round == 2)
    assert (c.low, c.high) == pytest.approx((50 - insights.Z95 * 2.0, 50 + insights.Z95 * 2.0))


def test_a_rounds_own_results_never_widen_it():
    before = _round(1, 2.0) + _round(2, 0.5, no=10)
    after = _round(1, 2.0) + _round(2, 3.0, no=10)
    a = insights.nest_score_widen(TOP, before)
    b = insights.nest_score_widen(TOP, after)
    assert [c.widen for c in a if c.round == 2] == [c.widen for c in b if c.round == 2]


def test_fewer_than_five_earlier_checks_leave_it_as_fitted():
    checks = _round(1, 3.0, n=4) + _round(2, 0.0, no=10)
    out = insights.nest_score_widen(TOP, checks)
    assert {c.widen for c in out} == {1.0}


def test_only_fully_measured_checks_count():
    earlier = [dataclasses.replace(c, fully_measured=False) for c in _round(1, 3.0)]
    out = insights.nest_score_widen(TOP, earlier + _round(2, 0.0, no=10))
    assert {c.widen for c in out} == {1.0}


def test_measurement_checks_are_never_widened():
    checks = _round(1, 3.0) + _values(2, n=4) + _round(2, 0.0, no=50)
    out = insights.nest_score_widen(TOP, checks)
    assert all(c.widen == 1.0 and c == d for c, d in zip(out, checks) if c.name != insights.SCORE)


def test_inside_follows_the_widened_range_and_it_stays_within_0_and_the_ceiling():
    checks = _round(1, 2.5) + [_score(2, 99, 2.2, sd=10.0, mean=60.0)]   # measured 82
    [late] = [c for c in insights.nest_score_widen(TOP, checks) if c.round == 2]
    assert not checks[-1].inside and late.inside
    assert late.high == 100.0 and late.low == pytest.approx(60 - insights.Z95 * 25.0)


def test_a_miss_is_still_measured_against_the_models_own_sd():
    checks = _round(1, 2.0) + _round(2, 2.0, no=10)
    out = insights.nest_score_widen(TOP, checks)
    assert insights.score_widen(out) == pytest.approx(insights.score_widen(checks))
    assert [c.sd for c in out] == [c.sd for c in checks]


def test_check_rounds_widens_each_round_from_the_ones_before():
    opt = _burger(rounds=5, noise=0.6)
    cache = {}
    nested = insights.check_rounds(opt, cache=cache)
    raw = [c for v in cache.values() for c in v]
    assert {c.widen for c in raw} == {1.0}                       # the cache keeps them raw
    for r, _ in insights.round_order(opt):
        earlier = [c for c in raw if c.round < r]
        mine = [c for c in nested if c.round == r and c.name == insights.SCORE]
        assert [c.widen for c in mine] == pytest.approx(
            [insights.score_widen(earlier)] * len(mine))
    # Round 2 (P1 #1): each measurement is widened the same way, by its own
    # record from the Rounds before (nest_measure_widen), so Make a round,
    # the Forecast check and Results read one range.
    for r, _ in insights.round_order(opt):
        for name in {c.name for c in raw} - {insights.SCORE}:
            earlier = [c for c in raw if c.round < r and c.name == name]
            mine = [c for c in nested if c.round == r and c.name == name]
            assert [c.widen for c in mine] == pytest.approx(
                [insights.measure_widen(earlier, name)] * len(mine))


def test_only_works_out_the_pairs_asked_for_and_keeps_nothing():
    opt = _burger(rounds=4, noise=0.6)
    full = insights.check_rounds(opt)
    pairs = {(r, insights.SCORE) for r, _ in insights.round_order(opt)} | {(4, "Firmness")}
    cache = {}
    part = insights.check_rounds(opt, cache=cache, only=pairs)
    assert cache == {}
    # With the checks their widening reads (2026-10-07): Round 4's
    # Firmness is widened by Firmness in the Rounds before it.
    need = insights.widen_pairs(opt, pairs)
    assert need - pairs == {(r, "Firmness") for r, _ in insights.round_order(opt) if r < 4}
    assert part == [c for c in full if (c.round, c.name) in need]


def test_a_disagreeing_repeat_is_judged_on_the_range_shown():
    """Ftue fix A (cold read I7): the limit is √2 × 1.96 × σ, σ a noise SD
    (here the assumed 1 point: 2.77), no longer the range shown, so the
    range's widening leaves it alone."""
    opt = _patty(rows=6)
    _repeat_of(opt, 0, 4)
    opt.Y_history[4] = opt.Y_history[0] - 10.0
    shown = _check(insights.SCORE, 2, 5, opt.Y_history[4], opt.Y_history[4], 2.0)
    checks = _values(2) + [shown]
    d = insights.disagreeing_repeat(opt, checks)
    assert d is not None and d.limit == pytest.approx(math.sqrt(2) * insights.Z95)
    wide = dataclasses.replace(shown, widen=2.0)
    assert insights.disagreeing_repeat(opt, _values(2) + [wide]) == d


def test_make_a_rounds_factor_leaves_out_the_open_rounds_own_checks():
    """Review A M-3: with the open Round partly recorded, its recorded rows
    are checked already; the column for the rest is widened as that Round's
    check will be — by the Rounds before it only."""
    opt = _burger(rounds=5, noise=0.6)
    checks = insights.check_rounds(opt)
    opt.pending_batch_no = 5                 # Round 5 still open, its rows recorded
    earlier = [c for c in checks if c.round != 5]
    assert insights.column_factor(opt, checks) == insights.score_widen(earlier)
    assert insights.column_factor(opt, checks) != insights.score_widen(checks)
    opt.pending_batch_no = 6                 # a new Round: every check counts
    assert insights.column_factor(opt, checks) == insights.score_widen(checks)
