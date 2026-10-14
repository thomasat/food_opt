"""0.8.0 final whole-branch fix wave (final review I-2, M-1, M-2; Plan B
parked items): what the "How" table says about the Overall score's
widening, "What to do next" while a Round is open, one factor for the
predicted best and Make a round's column."""
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

import insights
import ui_insights
import wording
from tests.test_app_ui import APP_PATH
from tests.test_food_bo import _BANNED
from tests.test_insights import _burger
from tests.test_insights_wave_c_ui import ferment_scored  # noqa: F401


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


# ------------------------------------------------------------------ #
#  Final review I-2 (ruling M5): the Overall-score row shows both
#  widenings, as the ranges on screen carry both
# ------------------------------------------------------------------ #

# ------------------------------------------------------------------ #
#  Final review M-2: mid-Round, answer 2's predicted best and Make a
#  round's column carry one factor
# ------------------------------------------------------------------ #

def test_the_predicted_best_is_widened_as_the_column_is_mid_round():
    opt = _burger(rounds=5, noise=0.6)
    checks = insights.check_rounds(opt)
    fitted = {insights.SCORE: insights.fit(opt, insights.SCORE, list(range(len(opt.Y_history))))}
    opt.pending_batch_no = 5                  # Round 5 open, its rows recorded
    best = insights.predicted_best(opt, fitted, checks)
    assert best.factor == insights.column_factor(opt, checks)
    assert best.factor != insights.score_widen(checks)


# ------------------------------------------------------------------ #
#  Final review M-1: "What to do next" while a Round is open says to
#  record it first, as the foot under it does
# ------------------------------------------------------------------ #

def _step(kind):
    return SimpleNamespace(kind=kind, gap=None, disagreement=None, binding=None)


def _open(opt, n):
    opt.set_pending_batch([{"Pea protein": 12.0 + k, "Methylcellulose": 1.0} for k in range(n)])
    return opt


@pytest.mark.parametrize("kind, start, end", [
    ("early", "", " The predictions cannot be judged yet."),
    ("rising", "Best so far is rising. ", ""), ("see", "", "")])
def test_with_a_round_open_the_next_step_is_to_record_it(kind, start, end):
    opt = _burger(rounds=3, noise=0.3)
    text, _ = ui_insights.next_step_text(opt, _step(kind))
    assert "Make the next Round" in text or "to see whether" in text
    _open(opt, 4)
    text, _ = ui_insights.next_step_text(opt, _step(kind))
    no = opt.pending_batch_no
    tail = (f"Record the 4 formulations still open in Round {no}, then make the next Round."
            if kind != "see" else
            f"Record the 4 formulations still open in Round {no} to see whether Best so far "
            "rises.")
    assert text == start + tail + end
    assert not any(p.search(text) for p in _BANNED), text


def test_one_open_formulation_is_said_in_the_singular():
    opt = _open(_burger(rounds=3, noise=0.3), 1)
    text, _ = ui_insights.next_step_text(opt, _step("rising"))
    assert text == ("Best so far is rising. Record the formulation still open in "
                    f"Round {opt.pending_batch_no}, then make the next Round.")


def test_a_process_only_project_says_trials(ferment_scored):
    """The Trial form (for_project), plural and singular."""
    opt = ferment_scored
    opt.set_pending_batch([{n: v for n, v in r.items()} for r in opt.recipe_history[:2]])
    text = ui_insights._say(opt, ui_insights.next_step_text(opt, _step("rising"))[0])
    assert f"Record the 2 trials still open in Round {opt.pending_batch_no}" in text
    assert "ormulation" not in text
    opt.set_pending_batch([dict(opt.recipe_history[0])])
    text = ui_insights._say(opt, ui_insights.next_step_text(opt, _step("see"))[0])
    assert text == (f"Record the trial still open in Round {opt.pending_batch_no} "
                    "to see whether Best so far rises.")


# ------------------------------------------------------------------ #
#  Final review M-3, M-7, M-8, M-9: words
# ------------------------------------------------------------------ #

def test_best_failed_says_predicted_best():
    """M-3: "Predicted best" everywhere; "worked out" is not the owner's word."""
    assert wording.BEST_FAILED == "The predicted best could not be found."


def test_one_point_is_said_in_the_singular():
    """M-7."""
    assert wording.stand_yes("1.0") == (
        "Rising: up 1 point in the last Round, above the 1-point noise threshold.")
    assert wording.stand_yes("3.1").startswith("Rising: up 3.1 points in the last Round")
    assert "up to 1 miss in 12" in wording.round_track_rough(11, 12, allowed=1)


def test_move_to_not_scored_help_counts_the_results():
    """M-7: two measurements missing are "these results"."""
    assert wording.move_to_not_scored_help(1) == wording.MOVE_TO_NOT_SCORED_HELP
    two = wording.move_to_not_scored_help(2)
    assert two.startswith("Use when these results will never come.")
    assert "this result" not in two


def test_the_holding_line_after_an_unreliable_verdict_is_not_good_news():
    """M-8."""
    assert wording.round_track_holding(11, 11, True, rounds=[3, 4, 5]) == (
        "Overall score, Rounds 3–5: 11 of 11 inside their 95 % interval; measured values: "
        "many more misses than expected.")


def test_the_results_setup_button_names_process_settings_on_a_process_only_project(
        ferment_scored):
    """M-9."""
    assert wording.change_setup_from_results() == wording.CHANGE_SETUP_FROM_RESULTS_BUTTON
    # Re-pinned to the concision round's short text (owner, 2026-10-01: too wordy).
    assert (wording.change_setup_from_results(settings_only=True)
            == "Change measurements or settings")
    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.run()
    at.session_state["main_tab"] = wording.TAB_RESULTS
    at.run()
    assert not at.exception
    labels = [b.label for b in at.tabs[2].button]
    assert "Change measurements or settings" in labels
    assert wording.CHANGE_SETUP_FROM_RESULTS_BUTTON not in labels


def test_the_new_words_pass_the_guard():
    for text in (wording.BEST_FAILED, wording.stand_yes("1.0"),
                 wording.round_track_rough(11, 12, allowed=1), wording.move_to_not_scored_help(2),
                 wording.round_track_holding(11, 11, True),
                 wording.change_setup_from_results(settings_only=True),
                 wording.next_rising_open(6, 4), wording.next_see_open(6, 1),
                 wording.next_too_early_open(6, 2)):
        assert not any(p.search(text) for p in _BANNED), text


# ------------------------------------------------------------------ #
#  Plan B parked: the tie line's partner is a different formulation;
#  an exact tie is said
# ------------------------------------------------------------------ #

def _three(scores):
    """Formulations 1-3 (Round 1) at `scores`."""
    from tests.test_complete_results import _patty
    opt = _patty(rows=3)
    for i, y in enumerate(scores):
        opt.Y_history[i] = y
    return opt


def _repeat_of(opt, no, score, rnd=2):
    """Formulation 4, a Repeat of `no` (the same make-up), scoring `score`."""
    from tests.test_complete_results import _recipe, _tell
    _tell(opt, 4, rnd)
    opt.recipe_history[3] = dict(opt.recipe_history[no - 1])
    opt.X_history[3] = list(opt.X_history[no - 1])
    opt.Y_history[3] = score
    return opt


def test_a_repeat_of_best_so_far_is_never_its_tie_partner():
    import ui_results
    opt = _repeat_of(_three([60.0, 70.0, 65.0]), 2, 70.3)
    line = ui_results._progress_line(opt)
    assert "treat them as equal" not in line, line          # 65 is 5.3 behind: no tie
    assert ui_results._move_sentence(opt, 2, 4) == wording.best_moved(2, 4)


def test_the_tie_partner_is_the_next_highest_different_formulation():
    import ui_results
    opt = _repeat_of(_three([60.0, 70.0, 69.8]), 2, 70.3)
    assert ui_results._progress_line(opt) == wording.tie_line(4, "70.30", "0.50", 3, "2.3")


def test_best_so_far_moving_to_its_own_repeat_flashes_what_the_line_says():
    """Re-review R-2: the flash uses the line's partner, the next highest
    different formulation — never the Repeat's own original."""
    import ui_results
    opt = _repeat_of(_three([60.0, 70.0, 69.8]), 2, 70.3)
    line = ui_results._progress_line(opt)
    assert ui_results._move_sentence(opt, 2, 4) == line == wording.tie_line(4, "70.30", "0.50", 3, "2.3")


def test_an_exact_tie_is_said_as_a_tie():
    import ui_results
    from tests.test_complete_results import _tell
    opt = _three([60.0, 70.0, 65.0])
    _tell(opt, 4, 2)                                     # a different make-up
    opt.Y_history[3] = 70.0
    line = ui_results._progress_line(opt)
    assert line == wording.tie_equal(2, 4, "70.00")
    assert line == "Formulations 2 and 4 are tied at 70.00."
    assert not any(p.search(line) for p in _BANNED)


def test_an_exact_trial_tie_says_trials():
    assert wording.for_project(_Trials(), wording.tie_equal(2, 4, "70.00")) == (
        "Trials 2 and 4 are tied at 70.00.")


class _Trials:
    def has_ingredients(self):
        return False

    def _process_settings(self):
        return ["Incubation time"]
