"""One 95 % range per forecast, whatever is cached (2026-10-07).

A measurement's Round-ahead check is widened by that measurement's checks
in every Round before it (insights.nest_measure_widen); Make a round widens
its Predicted measurements by every Round's but the open one's
(insights.measure_widen). The screens read the checks from the part the
workers had fitted ("trust": the verdict's Rounds) or from every check,
whichever was in first, so the ranges on Results, Make a round and the
Forecast check changed with what the saved .insights cache already held.
Every part now holds the checks its widening reads.

Synthetic projects only; no pilot data."""
from types import SimpleNamespace as NS

import pytest

import insights
import insights_jobs
import ui_insights as U
from tests.test_insights import _burger


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _ranges(checks, names=None):
    return {(c.round, c.name, c.formulation): (round(c.low, 9), round(c.high, 9), c.inside,
                                               round(c.widen, 9))
            for c in checks if names is None or c.name in names}


def _measurements(opt):
    return [n for n, _, _ in insights.outputs(opt) if n != insights.SCORE]


def _trust_only(opt):
    return insights_jobs.only_pairs(
        insights_jobs.stage_specs(opt, insights_jobs.plan(opt), "trust"))


def test_the_trust_checks_are_widened_as_every_check_is():
    # Six Rounds: the verdict reads Rounds 4-6, whose widening reads 2-3 too.
    opt = _burger(rounds=6, noise=0.4)
    memo = {}
    with insights.memo_in_use(memo):
        cold = insights.check_rounds(opt, cache={}, only=_trust_only(opt))
        warm = insights.check_rounds(opt, cache={})
    names = _measurements(opt)
    got = _ranges(cold, names)
    want = {k: v for k, v in _ranges(warm, names).items() if k in got}
    assert got == want
    # The window's own checks are all there, and some were widened.
    window = set(insights_jobs.window_rounds(opt))
    assert {k[0] for k in got} >= window
    assert any(v[3] > 1.0 for k, v in got.items() if k[0] in window)
    # Make a round's widening for the next Round reads the same record.
    for name in names:
        assert (insights.measure_widen(cold, name, 7)
                == insights.measure_widen(warm, name, 7))


def test_each_stage_holds_every_check_its_widening_reads():
    opt = _burger(rounds=6, noise=0.4)
    specs = insights_jobs.plan(opt)
    for stage in ("column", "trust", "stand", "now", "all"):
        pairs = {(s.stage, s.name) for s in insights_jobs.stage_specs(opt, specs, stage)
                 if s.stage != insights_jobs.NOW}
        have = {(s.stage, s.name) for s in specs}
        assert insights.widen_pairs(opt, pairs) & have == pairs, stage


def _session(monkeypatch, memo):
    state = {}
    monkeypatch.setattr(U, "st", NS(session_state=state))
    monkeypatch.setattr(U, "memo", lambda o: memo)
    monkeypatch.setattr(U, "start", lambda o: None)      # no workers: nothing in flight
    return state


def test_results_shows_one_range_cold_or_warm(monkeypatch):
    """Results' Measured against predicted table: cold (only the verdict's
    models fitted, so the part "trust" needs is read) and warm (every model
    in the saved cache, every check worked out) show the same ranges."""
    opt = _burger(rounds=6, noise=0.4)
    last = 6
    cold_memo = {}
    insights_jobs.compute_inline(opt, cold_memo, insights_jobs.stage_specs(
        opt, insights_jobs.plan(opt), "trust"))
    _session(monkeypatch, cold_memo)
    assert U._cached(opt) is None
    cold = U.cached_checks(opt)
    cold_frame = U.round_predicted_frame(opt, cold, last)

    warm_memo = {}
    insights_jobs.compute_inline(opt, warm_memo, insights_jobs.plan(opt))
    _session(monkeypatch, warm_memo)
    U._computed(opt, quiet=True)
    warm = U.cached_checks(opt)
    assert U._cached(opt) is not None
    warm_frame = U.round_predicted_frame(opt, warm, last)

    assert cold_frame.equals(warm_frame)
    names = _measurements(opt)
    got = _ranges(cold, names)
    assert got == {k: v for k, v in _ranges(warm, names).items() if k in got}

