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

def test_the_overall_score_row_shows_both_widenings(monkeypatch):
    opt = _burger(rounds=5, noise=0.6)
    monkeypatch.setattr(insights, "column_factor", lambda o, c: 2.5)
    score = SimpleNamespace(name=insights.SCORE, widen=1.3)
    firm = SimpleNamespace(name="Firmness", widen=1.3)
    assert ui_insights.widen_cell(opt, score, []) == "×1.3 · ×2.5 at the top"
    assert ui_insights.widen_cell(opt, firm, []) == "×1.3"
    monkeypatch.setattr(insights, "column_factor", lambda o, c: 1.0)
    assert ui_insights.widen_cell(opt, score, []) == "×1.3"          # nothing on top
    assert ui_insights.widen_cell(opt, SimpleNamespace(name=insights.SCORE, widen=None),
                                  []) == ""


def test_the_how_table_carries_the_factor_the_screens_use():
    opt = _burger(rounds=5, noise=0.6)
    checks = insights.check_rounds(opt)
    top = insights.column_factor(opt, checks)
    assert top > 1.0
    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.session_state["_loaded_project"] = opt.project_name
    at.run()
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    assert not at.exception
    [table] = [d for d in at.tabs[3].dataframe
               if wording.DETAILS_COLUMNS_2[5] in d.value.columns]
    cells = dict(zip(table.value[wording.DETAILS_COLUMNS_2[0]],
                     table.value[wording.DETAILS_COLUMNS_2[5]]))
    [score_cell] = [v for k, v in cells.items() if k == wording.OVERALL_SCORE_COLUMN]
    assert score_cell.endswith(f"· {wording.widen_text(top)} at the top")
    captions = [c.value for c in at.tabs[3].caption]
    assert wording.HOW_WIDENED_SCORE in captions


def test_the_widening_words_pass_the_guard():
    for text in (wording.score_widen_text(1.3, 2.5), wording.HOW_WIDENED_SCORE):
        assert not any(p.search(text) for p in _BANNED), text


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


@pytest.mark.parametrize("kind, start", [("early", "Too early to tell where the top is. "),
                                         ("rising", "Scores are still rising. "),
                                         ("see", "")])
def test_with_a_round_open_the_next_step_is_to_record_it(kind, start):
    opt = _burger(rounds=3, noise=0.3)
    text, _ = ui_insights.next_step_text(opt, _step(kind))
    assert "Make the next Round" in text or "to see whether" in text
    _open(opt, 4)
    text, _ = ui_insights.next_step_text(opt, _step(kind))
    no = opt.pending_batch_no
    tail = (f"Record the 4 formulations still open in Round {no}, then make the next Round."
            if kind != "see" else
            f"Record the 4 formulations still open in Round {no} to see whether scores rise.")
    assert text == start + tail
    assert not any(p.search(text) for p in _BANNED), text


def test_one_open_formulation_is_said_in_the_singular():
    opt = _open(_burger(rounds=3, noise=0.3), 1)
    text, _ = ui_insights.next_step_text(opt, _step("rising"))
    assert text == ("Scores are still rising. Record the formulation still open in "
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
                    "to see whether scores rise.")


# ------------------------------------------------------------------ #
#  Final review M-3, M-7, M-8, M-9: words
# ------------------------------------------------------------------ #

def test_best_failed_says_predicted_best():
    """M-3: "Predicted best" everywhere; "worked out" is not the owner's word."""
    assert wording.BEST_FAILED == "The predicted best could not be found."


def test_one_point_is_said_in_the_singular():
    """M-7."""
    assert wording.stand_yes("1.0") == "Yes — up 1 point in the last Round."
    assert wording.stand_yes("3.1") == "Yes — up 3.1 points in the last Round."
    assert "on average 1 point below prediction" in wording.score_record_line(5, 5, -1.0)
    assert "on average 1.8 points below prediction" in wording.score_record_line(7, 8, -1.84)


def test_move_to_not_scored_help_counts_the_results():
    """M-7: two measurements missing are "these results"."""
    assert wording.move_to_not_scored_help(1) == wording.MOVE_TO_NOT_SCORED_HELP
    two = wording.move_to_not_scored_help(2)
    assert two.startswith("Use when these results will never come.")
    assert "this result" not in two


def test_the_holding_line_after_an_unreliable_verdict_is_not_good_news():
    """M-8."""
    assert wording.round_track_holding(11, 11, True) == (
        "Recent Rounds: 11 of 11 Overall scores landed inside their likely range; "
        "the measurements missed more often — see Insights.")


def test_the_results_setup_button_names_process_settings_on_a_process_only_project(
        ferment_scored):
    """M-9."""
    assert wording.change_setup_from_results() == wording.CHANGE_SETUP_FROM_RESULTS_BUTTON
    assert (wording.change_setup_from_results(settings_only=True)
            == "Change a measurement or a process setting")
    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.run()
    at.session_state["main_tab"] = wording.TAB_RESULTS
    at.run()
    assert not at.exception
    labels = [b.label for b in at.tabs[2].button]
    assert "Change a measurement or a process setting" in labels
    assert wording.CHANGE_SETUP_FROM_RESULTS_BUTTON not in labels


def test_the_new_words_pass_the_guard():
    for text in (wording.BEST_FAILED, wording.stand_yes("1.0"),
                 wording.score_record_line(5, 5, -1.0), wording.move_to_not_scored_help(2),
                 wording.round_track_holding(11, 11, True),
                 wording.change_setup_from_results(settings_only=True),
                 wording.next_rising_open(6, 4), wording.next_see_open(6, 1),
                 wording.next_too_early_open(6, 2)):
        assert not any(p.search(text) for p in _BANNED), text
