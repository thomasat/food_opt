"""0.8.0 speed, Plan A Task 10 (review I16): Make a round's track-record
line reads the Overall score's own record in the verdict's Rounds, and
Plan A's strings pass the vocabulary guard."""
import pytest

import insights
import wording
from tests.test_080_trust import _check, _values
from tests.test_complete_results import _patty
from tests.test_food_bo import _BANNED


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _scores(rnd, n, miss, sd=1.0, no=500):
    """`n` Overall-score checks in Round `rnd`, each missing by `miss`."""
    return [_check(insights.SCORE, rnd, no + k, 50.0 + miss, 50.0, sd) for k in range(n)]


def test_no_record_no_line():
    opt = _patty(rows=6)
    assert insights.round_track(opt, _values(2) + _scores(2, 4, 0.0)) is None


def test_an_early_verdict_shows_nothing():
    opt = _patty(rows=6)
    assert insights.round_track(opt, _values(2, n=6) + _scores(2, 6, 0.0)) is None


def test_a_record_that_holds():
    opt = _patty(rows=6)
    t = insights.round_track(opt, _values(2) + _scores(2, 11, 0.2))
    assert (t.rough, t.inside, t.total, t.gap, t.see_insights) == (False, 11, 11, None, False)


def test_an_average_miss_beyond_one_point_is_rough_and_says_how_much():
    opt = _patty(rows=6)
    t = insights.round_track(opt, _values(2) + _scores(2, 12, -1.5))   # all inside, 1.5 below
    assert t.rough and t.gap == pytest.approx(-1.5) and t.inside == 12


def test_many_outside_is_rough_even_without_a_gap():
    opt = _patty(rows=6)
    checks = _values(2) + _scores(2, 6, 3.0, no=500) + _scores(2, 6, -3.0, no=600)
    t = insights.round_track(opt, checks)
    assert t.rough and t.gap is None and (t.inside, t.total) == (0, 12)


def test_an_unreliable_verdict_adds_see_insights():
    opt = _patty(rows=6)
    t = insights.round_track(opt, _values(2, miss=3.0) + _scores(2, 11, 0.0))
    assert t.see_insights and not t.rough


def test_the_lines_read_as_the_spec_says():
    assert wording.round_track_holding(11, 11) == (
        "Recent Rounds: 11 of 11 Overall scores landed inside their likely range.")
    assert wording.round_track_rough(6, 12, "5.0") == (
        "Recent Rounds ran about 5.0 points below prediction (6 of 12 inside) — treat these "
        "predictions as rough.")
    assert wording.round_track_rough(6, 12, "1.0", below=False).startswith(
        "Recent Rounds ran about 1 point above prediction")
    assert wording.round_track_rough(6, 12, see_insights=True).endswith(
        "treat these predictions as rough. See Insights.")
    assert wording.insights_progress(4, 6) == (
        "Checking predictions against your results: 4 of 6 Rounds checked…")
    assert wording.insights_progress(0, 1).endswith("0 of 1 Round checked…")


def test_plan_a_strings_pass_the_vocabulary_guard():
    lines = [wording.insights_progress(4, 6), wording.insights_progress(0, 1),
             wording.PREDICTION_PENDING, wording.round_track_holding(11, 11),
             wording.round_track_holding(11, 11, True), wording.round_track_rough(6, 12, "5.0"),
             wording.round_track_rough(6, 12, "1.0", False, True),
             wording.round_track_rough(6, 12)]
    assert [t for t in lines if any(p.search(t) for p in _BANNED)] == []


@pytest.mark.parametrize("state", ["wide", "over_wide"])
def test_a_wide_or_over_wide_verdict_shows_nothing(state, monkeypatch):
    """Review C I-5: Insights says "Too early to judge" (wide) or that the
    range was wider than it needed to be (over-wide); Make a round must not
    offer "11 of 11 landed inside" beside it."""
    opt = _patty(rows=6)
    checks = _values(2) + _scores(2, 11, 0.2)
    assert insights.round_track(opt, checks) is not None            # a record that holds
    monkeypatch.setattr(insights, "verdict_state", lambda *a: state)
    assert insights.round_track(opt, checks) is None


def test_the_holding_line_adds_see_insights_after_an_unreliable_verdict():
    """Review C M-1."""
    assert wording.round_track_holding(11, 11, True).endswith(" See Insights.")
    assert not wording.round_track_holding(11, 11).endswith("See Insights.")
