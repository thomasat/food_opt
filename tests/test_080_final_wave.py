"""0.8.0 final whole-branch fix wave (final review I-2, M-1, M-2; Plan B
parked items): what the "How" table says about the Overall score's
widening, "What to do next" while a Round is open, one factor for the
predicted best and Make a round's column."""
import pytest
from streamlit.testing.v1 import AppTest

import wording
from tests.test_app_ui import APP_PATH
from tests.test_food_bo import _BANNED
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

# ------------------------------------------------------------------ #
#  Final review M-1: "What to do next" while a Round is open says to
#  record it first, as the foot under it does
# ------------------------------------------------------------------ #

# ------------------------------------------------------------------ #
#  Final review M-3, M-7, M-8, M-9: words
# ------------------------------------------------------------------ #


def test_move_to_not_scored_help_counts_the_results():
    """M-7: two measurements missing are "these results"."""
    assert wording.move_to_not_scored_help(1) == wording.MOVE_TO_NOT_SCORED_HELP
    two = wording.move_to_not_scored_help(2)
    assert two.startswith("Use when these results will never come.")
    assert "this result" not in two


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
    for text in (wording.move_to_not_scored_help(2),
                 wording.change_setup_from_results(settings_only=True),
                 ):
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
    # v3 final cleanup: the next best as measured, never "within noise".
    assert ui_results._runner_up_line(opt) == wording.runner_up_line(3, "69.8", "0.5")


def test_best_so_far_moving_to_its_own_repeat_flashes_what_the_line_says():
    """Re-review R-2: the flash and the line use the same partner, the next
    highest different formulation — never the Repeat's own original. v3
    final cleanup: a 0.5 lead is a move, not a tie."""
    import ui_results
    opt = _repeat_of(_three([60.0, 70.0, 69.8]), 2, 70.3)
    assert ui_results._move_sentence(opt, 2, 4) == wording.best_moved(2, 4)
    assert ui_results._runner_up_line(opt).startswith("Next best: Formulation 3, 69.8")


def test_an_exact_tie_is_said_as_a_tie():
    import ui_results
    from tests.test_complete_results import _tell
    opt = _three([60.0, 70.0, 65.0])
    _tell(opt, 4, 2)                                     # a different make-up
    opt.Y_history[3] = 70.0
    line = ui_results._progress_line(opt)
    assert line == wording.tie_equal(2, 4, "70.0")
    assert line == "Formulations 2 and 4 are tied at 70.0."
    assert not any(p.search(line) for p in _BANNED)


def test_an_exact_trial_tie_says_trials():
    assert wording.for_project(_Trials(), wording.tie_equal(2, 4, "70.0")) == (
        "Trials 2 and 4 are tied at 70.0.")


class _Trials:
    def has_ingredients(self):
        return False

    def _process_settings(self):
        return ["Incubation time"]
