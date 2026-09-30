"""0.8.0 answer-key read-through fixes (docs/superpowers/research/
2026-09-29-speed-ux/answer-key-refresh.md §7): advice that must hold on the
numbers shown, one bias rule for Insights and Make a round, Repeats left out
said as left out, the width caution beside "still rising", and Results' and
Set up's words."""
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

import insights
import ui_insights
import wording
from tests.test_app_ui import APP_PATH
from tests.test_complete_results import _patty
from tests.test_food_bo import _BANNED


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _clean(*texts):
    for t in texts:
        assert not any(p.search(t) for p in _BANNED), t


# ------------------------------------------------------------------ #
#  1. "Higher" must hold on the numbers shown (Timing check: the
#     predicted best 92 below Best so far's 92.50)
# ------------------------------------------------------------------ #

def _pred(mean):
    return insights.Prediction(mean, mean - 5, mean + 5, mean, 2.5)


def _best(opt, mean, repeat):
    return insights.PredictedBest(dict(opt.recipe_history[0]), _pred(mean), None,
                                  opt.best_index(), 1.0, _pred(repeat))


def _top(opt, score):
    top = opt.best_index()
    opt.Y_history[top] = score
    return opt


def test_the_predicted_best_is_not_higher_when_it_reads_below_best_so_far():
    """Timing check: predicted best 92.15 (shown 92), Best so far's
    predicted Repeat 89.99, Best so far measured 92.50."""
    opt = _top(_patty(rows=6), 92.50)
    s = insights.best_standing(opt, _best(opt, 92.15, 89.99), 1.0)
    assert not s.higher
    assert not ui_insights.standing_text(opt, s, insights.Yardstick(1.0, None, 0, False)
                                         ).startswith("The predicted best is 92 (likely")


def test_higher_needs_a_shown_lead_beyond_the_yardstick_too():
    opt = _top(_patty(rows=6), 92.50)
    assert insights.best_standing(opt, _best(opt, 94.2, 89.99), 1.0).higher     # 94 vs 92.50
    assert not insights.best_standing(opt, _best(opt, 93.4, 89.99), 1.0).higher  # 93 vs 92.50


def test_untested_without_an_open_round_says_the_next_round():
    opt = _patty(rows=6)
    step = SimpleNamespace(kind="untested", gap=None, disagreement=None, binding=None)
    text, _ = ui_insights.next_step_text(opt, step)
    assert text.endswith("you can include it in the next Round.")
    assert "add it to the Round" not in text
    opt.set_pending_batch([dict(opt.recipe_history[0])])
    text, _ = ui_insights.next_step_text(opt, step)
    assert text.endswith("you can also add it to the Round yourself.")
    _clean(text, wording.next_untested(False, open_round=False))


# ------------------------------------------------------------------ #
#  2, 6. One bias rule for Insights and Make a round: beyond the
#        yardstick AND clear of noise (|gap| > 2 sd / sqrt(n))
# ------------------------------------------------------------------ #

from tests.test_080_round_track import _scores  # noqa: E402
from tests.test_080_trust import _check, _values  # noqa: E402


def _noisy(rnd, misses, no=700):
    return [_check(insights.SCORE, rnd, no + k, 50.0 + m, 50.0, 3.0)
            for k, m in enumerate(misses)]


def test_the_bias_rule_needs_the_yardstick_and_a_gap_clear_of_noise():
    rec = insights.ScoreRecord(11, 11, -1.1, 1.36)            # Line A after Round 6
    assert insights.bias_gap(rec, 1.0) == pytest.approx(-1.1)   # 2 x 1.36 / sqrt(11) = 0.82
    assert insights.bias_gap(insights.ScoreRecord(11, 11, -1.1, 3.0), 1.0) is None   # 1.81
    assert insights.bias_gap(insights.ScoreRecord(11, 11, -0.9, 0.1), 1.0) is None   # yardstick
    assert insights.bias_gap(None, 1.0) is None


def test_a_steady_gap_is_said_the_same_way_on_both_screens():
    """Line A after Round 6 / Line C: every Overall score inside, a steady
    gap beyond the yardstick. Insights qualifies "On track"; Make a round
    says the gap, not "rough"."""
    opt = _patty(rows=6)
    checks = _values(2) + _scores(2, 11, -1.5)
    h = insights.headline(opt, checks)
    state = insights.verdict_state(opt, h, checks)
    bias = insights.score_bias(opt, checks)
    assert state == "within" and bias == pytest.approx(-1.5)
    verdict = ui_insights.verdict_text(opt, h, state, bias)
    assert verdict.startswith("Measurements on track: ")
    assert verdict.endswith(" The Overall score, though, ran consistently below prediction.")
    line = ui_insights.score_line(opt, checks)
    assert line == ("Overall score: 11 of 11 inside; results ran about 1.5 points below "
                    "prediction — more than 1 point, so read new predictions as about 1.5 "
                    "points too high.")
    t = insights.round_track(opt, checks)
    assert not t.rough and t.gap == pytest.approx(-1.5)
    assert ui_insights.track_text(t) == (
        "Recent Rounds: 11 of 11 inside; results ran about 1.5 points below prediction.")
    _clean(verdict, line, ui_insights.track_text(t))


def test_a_noisy_gap_is_not_a_bias_anywhere():
    opt = _patty(rows=6)
    checks = _values(2) + _noisy(2, [2.0, -5.0, 1.5, -4.5, 2.5, -5.5, 1.0, -4.0, 2.0, -3.0, 1.0])
    h = insights.headline(opt, checks)
    assert abs(h.score.mean_miss) > 1.0                   # beyond the yardstick...
    assert insights.score_bias(opt, checks) is None        # ...but not clear of noise
    state = insights.verdict_state(opt, h, checks)
    assert not ui_insights.verdict_text(opt, h, state).startswith("Measurements")
    assert "too high" not in ui_insights.score_line(opt, checks)
    t = insights.round_track(opt, checks)
    assert t is not None and t.gap is None


def test_rough_only_when_too_few_land_inside():
    opt = _patty(rows=6)
    checks = _values(2) + _scores(2, 6, 3.0, no=500) + _scores(2, 6, -3.0, no=600)
    t = insights.round_track(opt, checks)
    assert t.rough
    assert "treat these predictions as rough" in ui_insights.track_text(t)


def test_make_a_rounds_column_help_reads_as_insights_score_line():
    """6: one wording for one fact (spec §0.3)."""
    opt = _patty(rows=6)
    checks = _values(2) + _scores(2, 11, -0.4)
    assert ui_insights.score_line(opt, checks) == (
        "Overall score: 11 of 11 inside; results ran about 0.4 points below prediction.")


def test_how_says_the_bias_rule_in_plain_words():
    assert wording.HOW_BIAS == (
        "The Overall score is said to run above or below prediction only when its average "
        "gap is more than how much Repeats differ (1 point without Repeats) and too steady "
        "across formulations to be chance.")
    _clean(wording.HOW_BIAS)


# ------------------------------------------------------------------ #
#  3. Repeats left out because they disagreed are said as left out
#     (Line B: "no Repeats yet" under a named Repeat)
# ------------------------------------------------------------------ #

def test_a_pair_left_out_is_said_as_left_out():
    from tests.test_080_trust import _pair
    opt, checks = _pair(-10.0)
    flagged = insights.disagreeing_repeat(opt, checks)
    rows = {r.name: r for r in insights.details(opt, checks, flagged=flagged)}
    for r in rows.values():
        assert (r.pairs, r.left_out) == (0, 1)
        assert wording.repeat_compare(r.ratio, r.repeat_sd, r.pairs, r.left_out) == (
            "1 pair left out (it disagreed)")
    kept = {r.name: r for r in insights.details(opt, checks)}
    assert all(r.left_out == 0 for r in kept.values())
    assert wording.repeat_compare(None, None, 0) == "no Repeats yet"
    assert wording.repeat_compare(None, None, 0, 2) == "2 pairs left out (they disagreed)"
    _clean(wording.repeat_compare(None, None, 0, 1), wording.repeat_compare(None, None, 0, 2))


# ------------------------------------------------------------------ #
#  5. "Scores are still rising" carries the width caution when some
#     outputs' likely range is wide (Line C)
# ------------------------------------------------------------------ #

def test_rising_names_the_outputs_whose_range_is_still_wide(monkeypatch):
    opt = _patty(rows=6)
    checks = _values(2) + _scores(2, 11, 0.1)
    wide_firmness = [_check("Firmness", 2, 900 + k, 5.0, 5.0, 3.0) for k in range(10)]
    checks = [c for c in checks if c.name != "Firmness"] + wide_firmness
    h = insights.headline(opt, checks)
    assert h.wide == ["Firmness"]
    monkeypatch.setattr(insights, "improvement",
                        lambda *a, **k: SimpleNamespace(state="yes", rise=3.0, sd=None, rounds=0))
    step = insights.next_step(opt, checks, None)
    assert step.kind == "rising" and step.wide == ["Firmness"]
    text, _ = ui_insights.next_step_text(opt, step)
    assert text == ("Scores are still rising; the likely range for Firmness is still wide. "
                    "Make the next Round.")
    opt.set_pending_batch([dict(opt.recipe_history[0])])
    text, _ = ui_insights.next_step_text(opt, step)
    assert text == ("Scores are still rising; the likely range for Firmness is still wide. "
                    f"Record the formulation still open in Round {opt.pending_batch_no}, "
                    "then make the next Round.")
    three = SimpleNamespace(kind="rising", gap=None, disagreement=None, binding=None,
                            wide=["Firmness", "Juiciness", insights.SCORE])
    text, _ = ui_insights.next_step_text(opt, three)
    assert "the likely range for Firmness, Juiciness and the Overall score is still wide" in text
    assert "ranges" not in text
    _clean(text)


def test_rising_without_a_wide_output_is_unchanged():
    opt = _patty(rows=6)
    step = SimpleNamespace(kind="rising", gap=None, disagreement=None, binding=None, wide=[])
    assert ui_insights.next_step_text(opt, step)[0] == wording.NEXT_RISING
