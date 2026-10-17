"""0.8.0 review fixes, Task 6: "Is it still improving?" answers in words
(Yes / Barely / Not lately), the Repeats line names how many pairs it rests
on, the four headings show at once with one progress bar, and the early
gate line is said once (spec 2026-09-29 §3.7, §6, §9 I9, I11)."""
import pytest
from tests.demo_build_reason import REVIEW_FOUR
from streamlit.testing.v1 import AppTest

import insights
import ui_insights
import wording
from tests.test_app_ui import APP_PATH, burger, scored  # noqa: F401
from tests.test_complete_results import _patty, _tell
from tests.test_insights_ui import _open_insights, insights_scored  # noqa: F401


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _rounds(scores_by_round):
    """One formulation per score, Rounds numbered from 1; each score is set
    straight onto the row (its measurements are not what is tested here)."""
    opt = _patty(rows=0)
    no = 0
    for r, scores in enumerate(scores_by_round, start=1):
        for y in scores:
            no += 1
            _tell(opt, no, r)
            opt.Y_history[-1] = y
    return opt


def test_no_answer_after_one_round_or_one_round_without_a_rise():
    assert insights.improvement(_rounds([[70.0, 72.0]])) is None
    assert insights.improvement(_rounds([[72.0], [70.0]])) is None
    assert insights.improvement(_patty(rows=0)) is None


def test_the_rise_is_measured_against_formulations_best_so_far_compares():
    """A score rescaled without a measurement is not the Best so far to beat."""
    opt = _patty(rows=2, cook_loss=False)
    opt.Y_history[0], opt.Y_history[1] = 97.0, 60.0          # before Cook loss existed
    opt.add_objective("Cook loss", 20, goal="min", min_val=0, max_val=40)
    _tell(opt, 3, 2)
    _tell(opt, 4, 3)
    opt.Y_history[2], opt.Y_history[3] = 70.0, 74.0
    a = insights.improvement(opt)
    assert a.state == "yes" and a.rise == pytest.approx(4.0)


def test_the_repeats_line_names_its_pairs():
    assert wording.stand_repeats("0.3", 1) == (
        "Repeat SD of the Overall score: 0.3 points from 1 pair. The noise threshold stays at "
        "1 point until 3 pairs.")
    assert "from 4 pairs; this is the noise threshold" in wording.stand_repeats("0.3", 4)


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_answer_comes_first_and_best_so_far_drops_what_it_repeats(insights_scored):
    at = _open_insights(AppTest.from_file(APP_PATH, default_timeout=180))
    md = [m.value for m in at.tabs[3].markdown]
    i = md.index(f"##### {wording.Q_STAND}")
    answer = insights.improvement(insights_scored,
                                  yardstick=insights.noise_yardstick(insights_scored))
    if answer is not None:
        # One line: the answer, then Best so far (spec 2026-09-30 §2.3).
        assert md[i + 1].startswith(ui_insights.stand_text(answer) + " Best so far:")
        assert wording.STAND_LATEST not in md[i + 1]
        assert "No higher score" not in md[i + 1]      # the fixture reads Not lately


@pytest.mark.skip(reason="review cards prototype: pins the old Review layout (status lines, Explored fold, header); reconcile at the final merge")
def test_the_gate_line_is_said_once_after_is_it_still_improving(scored):
    at = _open_insights(AppTest.from_file(APP_PATH, default_timeout=180))
    tab = at.tabs[3]
    # Since ftue W1 (R-B1 row 3) the gate is said once, by the Predictions
    # status line; the old gate line is gone from the tab.
    # Re-pinned to the concision round's short line (pred_gate's full
    # sentence is in its tooltip).
    said = [m.value for m in tab.markdown if wording.short_gate(2) in m.value]
    assert said == [wording.status_line(wording.STATUS_PREDICTIONS, wording.short_gate(2),
                                        wording.PREDICTIONS_WORDS["not_yet"])]
    assert wording.insights_gate(2) not in [c.value for c in tab.caption]


def test_the_new_strings_pass_the_vocabulary_guard():
    from tests.test_food_bo import _BANNED
    for text in (wording.insights_progress(0, 3), wording.insights_gate(3),
                 wording.stand_repeats("0.3", 2), wording.stand_yes("3.1"),
                 wording.stand_barely("0.2", "0.3"), wording.stand_barely("0.2"),
                 wording.stand_not_lately(3)):
        assert not any(p.search(text) for p in _BANNED), text
    assert "first visit" not in wording.insights_progress(0, 3)


# ---------------------------------------------------------------- fix round 1


def _waiting_round(opt, no, r):
    """Formulation `no` recorded in Round `r`, still waiting for Cook loss."""
    _tell(opt, no, r, drop=("Cook loss",))
    assert not opt.fully_measured(len(opt.Y_history) - 1)


def test_a_round_still_waiting_is_not_a_round_without_a_higher_score():
    """Review I-1: Round 3 is recorded but not scored, so Best so far in
    Round 1 has gone one scored Round without a higher score, not two."""
    opt = _rounds([[90.0], [80.0]])
    _waiting_round(opt, 3, 3)
    assert insights.improvement(opt) is None
    f = insights.best_facts(opt)
    assert f.rounds_since == 1 and not f.latest


def test_a_round_moved_to_not_scored_is_not_counted():
    opt = _rounds([[70.0], [75.0]])
    _waiting_round(opt, 3, 3)
    opt.set_aside_formulation(3)
    assert opt.is_set_aside(2)
    a = insights.improvement(opt)
    assert a.state == "yes" and a.rise == pytest.approx(5.0)
    opt2 = _rounds([[90.0], [80.0]])
    _waiting_round(opt2, 3, 3)
    opt2.set_aside_formulation(3)
    assert insights.improvement(opt2) is None
    assert insights.best_facts(opt2).rounds_since == 1


def test_improvement_takes_what_the_tab_already_worked_out(monkeypatch):
    """Review M-4: best_facts and repeat_sd run once per "Is it still
    improving?" section."""
    opt = _rounds([[70.0, 72.0], [75.1]])
    f, spread = insights.best_facts(opt), insights.repeat_sd(opt, insights.SCORE)
    monkeypatch.setattr(insights, "best_facts", lambda o: pytest.fail("computed again"))
    monkeypatch.setattr(insights, "repeat_sd", lambda o, n: pytest.fail("computed again"))
    assert insights.improvement(opt, f, spread).state == "yes"


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_section_works_out_best_so_far_once(insights_scored, monkeypatch):
    calls = []
    real = insights.best_facts

    def counted(opt):
        calls.append(1)
        return real(opt)
    monkeypatch.setattr(insights, "best_facts", counted)
    at = _open_insights(AppTest.from_file(APP_PATH, default_timeout=180))
    calls.clear()
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    assert not at.exception
    # Twice since ftue W1: Progress figures' line 1 and the Progress status
    # line (insights.status_facts) each read it once per run, no more.
    assert len(calls) == 2


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_question_4_without_a_score_model_says_it_could_not_be_worked_out(
        insights_scored, monkeypatch):
    """Review M-2: with 5 or more scored, never "Starts once 5 … So far: 9"."""
    import dataclasses
    real = insights.fit_all

    def no_score(opt):
        fitted = real(opt)
        fitted[insights.SCORE] = dataclasses.replace(fitted[insights.SCORE], model=None)
        return fitted
    monkeypatch.setattr(insights, "fit_all", no_score)
    at = _open_insights(AppTest.from_file(APP_PATH, default_timeout=180))
    assert not at.exception
    caps = [c.value for c in at.tabs[3].caption]
    assert wording.BEST_FAILED in caps
    assert not any("Starts once" in c for c in caps), caps
