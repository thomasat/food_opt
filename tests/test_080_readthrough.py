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


def test_higher_needs_a_shown_lead_beyond_the_yardstick_too():
    opt = _top(_patty(rows=6), 92.50)
    assert insights.best_standing(opt, _best(opt, 94.2, 89.99), 1.0).higher     # 94 vs 92.50
    assert not insights.best_standing(opt, _best(opt, 93.4, 89.99), 1.0).higher  # 93 vs 92.50


def test_untested_without_an_open_round_says_the_next_round():
    opt = _patty(rows=6)
    step = SimpleNamespace(kind="untested", gap=None, disagreement=None, binding=None)
    text, _ = ui_insights.next_step_text(opt, step)
    assert "add it to the next Round as your own formulation" in text
    opt.set_pending_batch([dict(opt.recipe_history[0])])
    text, _ = ui_insights.next_step_text(opt, step)
    assert f"add it to Round {opt.pending_batch_no} as your own formulation" in text
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


def test_rough_only_when_too_few_land_inside():
    opt = _patty(rows=6)
    checks = _values(2) + _scores(2, 6, 3.0, no=500) + _scores(2, 6, -3.0, no=600)
    t = insights.round_track(opt, checks)
    assert t.rough
    assert "95 % intervals allow up to 2 misses in 12" in ui_insights.track_text(t)


def test_make_a_rounds_column_help_reads_as_insights_score_line():
    """6: one wording for one fact (spec §0.3)."""
    opt = _patty(rows=6)
    checks = _values(2) + _scores(2, 11, -0.4)
    assert ui_insights.score_line(opt, checks) == (
        "Overall score, Round 2: 11 of 11 inside their 95 % interval; mean error −0.4 "
        "points (measured minus predicted).")


def test_a_custom_fields_own_name_is_kept_as_typed():
    """Re-review R6: only the scope is said in the project's words."""
    trials = SimpleNamespace(has_ingredients=lambda: False,
                             _process_settings=lambda: ["Incubation time"])
    assert wording.custom_record_option("Formulation lead", "formulation", trials) == (
        "Formulation lead · Each trial")
    assert wording.custom_record_option("Formulation lead", "formulation") == (
        "Formulation lead · Each formulation")


# ------------------------------------------------------------------ #
#  3. Repeats left out because they disagreed are said as left out
#     (Line B: "no Repeats yet" under a named Repeat)
# ------------------------------------------------------------------ #

def test_a_pair_left_out_is_said_as_left_out():
    from tests.test_080_trust import _pair
    opt, checks = _pair(-10.0)
    flagged = insights.disagreeing_repeat(opt, checks)
    rows = {r.name: r for r in insights.details(opt, checks, flagged=flagged)}
    # Ftue fix A (cold read I7): the disagreeing pair now stays in.
    for r in rows.values():
        assert (r.pairs, r.left_out) == (1, 0)
    kept = {r.name: r for r in insights.details(opt, checks)}
    assert all(r.left_out == 0 for r in kept.values())


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
    # 0.8.0 e2e fixes (review B2): no Repeats, so the 1 point is assumed and
    # the row asks for a Repeat of Best so far.
    assert text == ("Best so far is rising; 95 % intervals for Firmness are still wide. "
                    "Make the next Round. " + wording.NEXT_REPEAT_TO_MEASURE)
    opt.set_pending_batch([dict(opt.recipe_history[0])])
    text, _ = ui_insights.next_step_text(opt, step)
    assert text == ("Best so far is rising; 95 % intervals for Firmness are still wide. "
                    f"Record the formulation still open in Round {opt.pending_batch_no}, "
                    "then make the next Round. " + wording.NEXT_REPEAT_TO_MEASURE)
    three = SimpleNamespace(kind="rising", gap=None, disagreement=None, binding=None,
                            wide=["Firmness", "Juiciness", insights.SCORE])
    text, _ = ui_insights.next_step_text(opt, three)
    assert "95 % intervals for Firmness, Juiciness and the Overall score are still wide" in text
    assert "ranges" not in text
    _clean(text)


def test_rising_without_a_wide_output_is_unchanged():
    opt = _patty(rows=6)
    step = SimpleNamespace(kind="rising", gap=None, disagreement=None, binding=None, wide=[])
    assert ui_insights.next_step_text(opt, step)[0] == wording.NEXT_RISING


# ------------------------------------------------------------------ #
#  9. "trial" in an ingredient project's Preparation and records
# ------------------------------------------------------------------ #

def test_custom_records_say_formulation_and_trial_by_project():
    assert wording.CUSTOM_FIELD_HELP == ("For example, Operator for each formulation, or Expiry "
                                         "date for each ingredient. These records do not affect "
                                         "scores.")
    assert wording.CUSTOM_SCOPE_NAMES["formulation"] == "Each formulation"
    assert wording.CUSTOM_SECTION_NAMES["formulation"] == "Formulation records"
    trials = SimpleNamespace(has_ingredients=lambda: False,
                             _process_settings=lambda: ["Incubation time"])
    assert wording.for_project(trials, wording.CUSTOM_SCOPE_NAMES["formulation"]) == "Each trial"
    assert "Operator for each trial" in wording.for_project(trials, wording.CUSTOM_FIELD_HELP)


def test_set_up_never_says_trial_on_an_ingredient_project(tmp_path):
    opt = _patty(rows=3)
    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.session_state["_loaded_project"] = opt.project_name
    at.run()
    at.session_state["main_tab"] = wording.TAB_SETUP
    at.run()
    assert not at.exception
    said = ([c.value for c in at.tabs[0].caption] + [m.value for m in at.tabs[0].markdown]
            + [o for s in at.tabs[0].selectbox for o in s.proto.options])
    assert wording.CUSTOM_FIELD_HELP in said
    assert not [t for t in said if "trial" in t.lower()], [t for t in said if "trial" in t.lower()]
