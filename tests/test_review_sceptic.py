"""Sceptic review (owner-approved, 2026-10-14): two expert reviewers found
every number right and some words claiming more than the data shows. The
stop says what is within noise (the Overall score, of 100) and whether each
off-target measurement is beyond its own noise band; the tasting noise
says its unit and what 1 point is per measurement; the Forecast check says
its count and its reference; a 95 % range over a measurement's whole Range
is "No forecast", counted nowhere as in or out; a verdict's pairs are
called of all with how many right; the stop's Set up "?" checks first; the
rebuilt forecasts' "?" says what they are (never "today").

Synthetic projects only (test_final_demo's yogurt-shaped panel)."""
import dataclasses
import re
from types import SimpleNamespace as NS

import pytest

import insights
import learned
import ui_insights as U
import wording
from tests.test_final_demo import ROWS, _FakeSt, _clean, _facts, _opt, _words


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


# ------------------------------------------------------------------ #
#  1 · the stop box: the score within noise of 100; each miss vs its band
# ------------------------------------------------------------------ #


# ------------------------------------------------------------------ #
#  2 · the tasting noise: the stop's "?" says the noise in use
# ------------------------------------------------------------------ #


# ------------------------------------------------------------------ #
#  3 · the Forecast check: the count and the reference
# ------------------------------------------------------------------ #

# ------------------------------------------------------------------ #
#  4 · a 95 % range over the whole Range: No forecast
# ------------------------------------------------------------------ #

def _check(name, low, high, measured=5.0, predicted=6.0, r=6, no=16):
    return insights.Check(no, r, name, predicted, low, high, measured, low <= measured <= high,
                          True, raw_mean=predicted, sd=1.0)


@pytest.mark.parametrize("low, high, none", [
    (0.0, 15.0, True),          # the whole Range
    (0.15, 14.85, True),        # each end exactly 1 % of 15 in
    (0.1, 14.9, True),
    (0.16, 15.0, False),        # the low end more than 1 % in
    (0.0, 14.84, False),        # the high end more than 1 % in
    (2.1, 10.3, False),         # a real range
])
def test_whole_range_is_each_end_within_one_percent(low, high, none):
    opt = _opt()
    assert U.no_forecast(opt, _check("Creaminess", low, high)) is none
    assert wording.NO_FORECAST_EDGE == 0.01


# ------------------------------------------------------------------ #
#  5 · a verdict's pairs: called of all
# ------------------------------------------------------------------ #


# ------------------------------------------------------------------ #
#  6 · the stop's advice: check first
# ------------------------------------------------------------------ #


# ------------------------------------------------------------------ #
#  Fix round 1 · no forecast is counted nowhere as inside or outside
# ------------------------------------------------------------------ #

def _round_of(r, whole, missed=0, name="Creaminess", start=1):
    """Three checks of `name` in Round `r`: whole-Range ranges when
    `whole`, else 4–6 with `missed` of them measured outside."""
    out = []
    for k in range(3):
        low, high = (0.0, 15.0) if whole else (4.0, 6.0)
        measured = 7.0 if k < missed else 5.0
        out.append(_check(name, low, high, measured, 5.0, r=r, no=start + k))
    return out


def test_check_rounds_marks_each_whole_range_check():
    opt = _opt()
    marked = insights.mark_no_forecast(opt, [_check("Tangy", 0.0, 15.0), _check("Tangy", 0.1, 14.9),
                                             _check("Tangy", 2.1, 10.3),
                                             _check(insights.SCORE, 0.0, 100.0, 90.0, 90.0)])
    assert [c.no_forecast for c in marked] == [True, True, False, True]
    assert insights.NO_FORECAST_EDGE == wording.NO_FORECAST_EDGE == 0.01
    assert insights.ranged(marked) == [marked[2]]


def test_check_rounds_returns_checks_marked_on_their_final_range(monkeypatch):
    """Behavioural (fix round 2): check_rounds marks after its widening, so
    a range widened to the whole Range is no forecast."""
    opt = _opt()
    fake = {r: [_check("Tangy", 2.0, 8.0, 5.0, 5.0, r=r, no=3 * r + k) for k in range(3)]
            for r in (2, 3, 4, 5, 6)}
    fake[6][0] = _check("Tangy", 0.0, 15.0, 5.0, 5.0, r=6, no=18)
    monkeypatch.setattr(insights, "_check_round", lambda o, r, first, names: list(fake.get(r, [])))
    got = insights.check_rounds(opt)
    whole = [(c.round, c.formulation) for c in got if c.no_forecast]
    assert (6, 18) in whole
    assert all(c.no_forecast == U.no_forecast(opt, c) for c in got)


# ------------------------------------------------------------------ #
#  Fix round 2
# ------------------------------------------------------------------ #

def test_error_record_counts_change_but_its_numbers_do_not():
    """The no-forecast rule changes the in/outside COUNT only: the mean
    miss, its sd, se and df stay those of every fully measured check."""
    opt = _opt()
    plain = _round_of(3, False, 1, start=1) + _round_of(4, True, 0, start=4) + \
        _round_of(5, False, 0, start=7)
    marked = insights.mark_no_forecast(opt, plain)
    a = insights.error_record(plain, "Creaminess")
    b = insights.error_record(marked, "Creaminess")
    assert (a.inside, a.total) == (8, 9) and (b.inside, b.total) == (5, 6)
    assert (a.mean_miss, a.gap_sd, a.se, a.df, a.rounds) == \
        (b.mean_miss, b.gap_sd, b.se, b.df, b.rounds)


def test_the_all_10_samples_bias_numbers_are_as_at_a4e1420(tmp_path):
    """The all-10 sample's records, pinned against the base a4e1420
    (before the sceptic review): only inside and total change. Re-pinned
    when the sample took the pilot's own limits (the model reads amounts
    as shares of each Lowest-Highest): Tangy's predictions now run high
    over Rounds 3-10, beyond its noise; the Overall score's do not."""
    import json
    import sample_projects as sp
    from food_bo import FoodOptimizer
    (tmp_path / f"{sp.ALL10}.pkl").write_text(json.dumps(sp.shipped_state(sp.ALL10)))
    opt = FoodOptimizer(sp.ALL10)
    checks = insights.check_rounds(opt)
    base = {insights.SCORE: (-0.6989407206590066, 3.2753873686595236, 0.7908622282691944, 7),
            "Consistency": (0.9367823847804867, 1.7650620976165172, 0.5993300420438497, 7),
            "Creaminess": (0.6095510888328993, 1.2235456484352254, 0.36900302965143467, 7),
            "Tangy": (-0.7730958575129686, 1.3654880880458824, 0.24230436210362377, 7),
            "Thickness uniformity": (-0.04256151085781501, 1.2345902897386172,
                                     0.4018420682708456, 7)}
    for name, (mean, sd, se, df) in base.items():
        r = insights.error_record(checks, name)
        assert r.mean_miss == pytest.approx(mean, abs=1e-9), name
        assert r.gap_sd == pytest.approx(sd, abs=1e-9) and r.se == pytest.approx(se, abs=1e-9)
        assert r.df == df
    h = insights.headline(opt, checks)
    assert insights.score_bias(opt, checks, h) is None
    assert [d.name for d in insights.details(opt, checks) if d.bias] == ["Tangy"]


# ------------------------------------------------------------------ #
#  Fix round 3
# ------------------------------------------------------------------ #


def test_the_gates_count_values_checked_not_values_forecast():
    """Fix round 3: the 10-value gate (headline) and the Overall score's
    SKILL_MIN gate count the checks made, as before the no-forecast rule;
    with no forecast at all, nothing is judged."""
    opt = _opt()
    checks = insights.mark_no_forecast(opt, _round_of(3, False, 0, start=1)
                                       + _round_of(4, True, 0, start=4)
                                       + _round_of(5, False, 0, start=7)
                                       + _round_of(5, False, 0, "Tangy", start=7))
    h = insights.headline(opt, checks)
    assert h.state == "verdict" and (h.total, h.no_forecast) == (9, 3)
    every = insights.mark_no_forecast(opt, _round_of(3, True, 0, start=1)
                                      + _round_of(4, True, 0, start=4)
                                      + _round_of(5, True, 0, start=7)
                                      + _round_of(6, True, 0, start=10))
    assert insights.headline(opt, every).state == "early"
    rec = insights.ScoreRecord(4, 4, 0.0, None, [1, 2], no_forecast=6)
    assert insights.score_verdict_state(opt, NS(state="verdict", score=rec, wide=[]), []) \
        != "early"
    rec0 = insights.ScoreRecord(0, 0, 0.0, None, [1, 2], no_forecast=12)
    assert insights.score_verdict_state(opt, NS(state="verdict", score=rec0, wide=[]), []) \
        == "early"


# ------------------------------------------------------------------ #
#  Fix round 4
# ------------------------------------------------------------------ #

def _early_window_checks(opt):
    """Rounds 1–2 forecast and inside; Rounds 3–6 every range over the
    whole Range: the window (Rounds 3–6, 12 values checked) has no value
    with a forecast, though earlier Rounds had."""
    return insights.mark_no_forecast(opt, _round_of(1, False, 0, start=1)
                                     + _round_of(2, False, 0, start=4)
                                     + _round_of(3, True, 0, start=7)
                                     + _round_of(4, True, 0, start=10)
                                     + _round_of(5, True, 0, start=13)
                                     + _round_of(6, True, 0, start=16))


