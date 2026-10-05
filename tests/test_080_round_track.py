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


def test_an_average_miss_beyond_one_point_says_how_much_and_is_not_rough():
    """Answer-key read-through: "rough" only when too few land inside; a
    steady gap beyond the yardstick is said as Insights says it."""
    opt = _patty(rows=6)
    t = insights.round_track(opt, _values(2) + _scores(2, 12, -1.5))   # all inside, 1.5 below
    assert not t.rough and t.gap == pytest.approx(-1.5) and t.inside == 12


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
    """Rewritten for the BO-views wave (03 §5.8, review I7)."""
    assert wording.round_track_holding(11, 11, rounds=[3, 4, 5]) == (
        "Overall score, Rounds 3–5: 11 of 11 inside their 95 % interval.")
    assert wording.round_track_rough(6, 12, "5.0", rounds=[3, 4, 5]) == (
        "Overall score, Rounds 3–5: 6 of 12 inside their 95 % interval; Overall scores have "
        "come in about 5.0 points below the predictions. More misses than expected.")
    assert wording.round_track_rough(6, 12, "1.0", below=False).endswith(
        "scores have come in about 1 point above the predictions. More misses than expected.")
    assert wording.round_track_rough(6, 12, see_insights=True, allowed=2).endswith(
        "95 % intervals allow up to 2 misses in 12; measured values: many more misses than "
        "expected.")
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
    """Review C M-1; T0 review I-3: the unreliable verdict's own name on
    Insights, "Many more misses than expected"."""
    assert wording.round_track_holding(11, 11, True).endswith(
        "; measured values: many more misses than expected.")
    assert "measured values" not in wording.round_track_holding(11, 11)


def test_no_line_when_the_overall_scores_own_range_is_wide(monkeypatch):
    """Final review I-1: the verdict holds ("within") but Insights says the
    likely range for the Overall score is wide, so says little yet; Make a
    round must not report "12 of 12 landed inside" beside it. good-5 at
    Round 7 is one of the 13 such states in the coverage fixture."""
    from tests import coverage_sim as sim
    [entry] = [e for e in sim.fixture()["projects"] if e["project"] == "good-5"]
    opt = sim.stub(entry)
    checks = insights.nest_score_widen(opt, [c for c in sim.checks_of(entry) if c.round <= 7])
    h = insights.headline(opt, checks)
    assert insights.SCORE in h.wide and h.score is not None
    assert insights.verdict_state(opt, h, checks) == "within"
    # The fixture's stand-in project has no Repeats: the 1-point yardstick.
    monkeypatch.setattr(insights, "noise_yardstick",
                        lambda *a, **k: insights.Yardstick(1.0, None, 0, False))
    # The stand-in has no rows to take a skill baseline from (BO views:
    # skill is pinned in test_092_skill.py).
    monkeypatch.setattr(insights, "skill", lambda *a, **k: None)
    assert insights.round_track(opt, checks) is None
