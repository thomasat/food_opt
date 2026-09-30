"""0.8.0 review fixes, Task 6: "Is it still improving?" answers in words
(Yes / Barely / Not lately), the Repeats line names how many pairs it rests
on, the four headings show at once with one progress bar, and the early
gate line is said once (spec 2026-09-29 §3.7, §6, §9 I9, I11)."""
import pytest
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


def test_yes_when_the_latest_round_rose_by_more_than_a_point_without_repeats():
    opt = _rounds([[70.0, 72.0], [75.1]])
    a = insights.improvement(opt)
    assert (a.state, a.sd) == ("yes", None)
    assert a.rise == pytest.approx(3.1)
    assert ui_insights.improving_text(a) == "Yes — up 3.1 points in the last round."


def test_barely_when_the_rise_is_within_a_point_without_repeats():
    opt = _rounds([[70.0, 72.0], [72.4]])
    a = insights.improvement(opt)
    assert a.state == "barely"
    assert ui_insights.improving_text(a) == (
        "Barely — the last rise (0.4 points) is under 1 point.")


def test_barely_against_the_spread_between_repeats(monkeypatch):
    opt = _rounds([[86.46], [86.68]])
    monkeypatch.setattr(insights, "repeat_sd", lambda o, name: insights.RepeatSpread(0.29, 1))
    a = insights.improvement(opt)
    assert a.state == "barely"
    assert ui_insights.improving_text(a) == (
        "Barely — the last rise (0.2 points) is smaller than the spread between Repeats (0.3).")


def test_not_lately_after_two_or_more_rounds_without_a_higher_score():
    opt = _rounds([[90.0], [80.0], [85.0], [70.0]])
    a = insights.improvement(opt)
    assert (a.state, a.rounds) == ("not_lately", 3)
    assert ui_insights.improving_text(a) == "Not lately — no higher score in 3 rounds."


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
    assert wording.repeats_line("0.3", 1) == (
        "Repeats of the same formulation vary by about 0.3 points (1 pair), "
        "so a smaller rise may be chance.")
    assert "(4 pairs)" in wording.repeats_line("0.3", 4)


def test_the_answer_comes_first_and_best_so_far_drops_what_it_repeats(insights_scored):
    at = _open_insights(AppTest.from_file(APP_PATH, default_timeout=180))
    md = [m.value for m in at.tabs[3].markdown]
    i = md.index(f"**{wording.Q_IMPROVING}**")
    answer = insights.improvement(insights_scored)
    if answer is not None:
        assert md[i + 1] == ui_insights.improving_text(answer)
        assert md[i + 2].startswith("Best so far:")
        assert wording.BEST_LATEST not in md[i + 2]


def test_the_four_headings_come_in_order(insights_scored):
    at = _open_insights(AppTest.from_file(APP_PATH, default_timeout=180))
    md = [m.value for m in at.tabs[3].markdown]
    heads = [f"**{q}**" for q in (wording.Q_PREDICTIONS, wording.q_change(False),
                                  wording.Q_IMPROVING, wording.Q_BEST)]
    assert [md.index(h) for h in heads] == sorted(md.index(h) for h in heads)
    assert all(md.count(h) == 1 for h in heads)


def test_a_failed_fit_takes_the_three_model_questions_off(insights_scored, monkeypatch):
    monkeypatch.setattr(insights, "fit_all", lambda opt: (_ for _ in ()).throw(RuntimeError()))
    at = _open_insights(AppTest.from_file(APP_PATH, default_timeout=180))
    assert not at.exception
    md = " ".join(m.value for m in at.tabs[3].markdown)
    assert wording.Q_PREDICTIONS not in md and wording.Q_BEST not in md
    assert wording.Q_IMPROVING in md
    assert [c.value for c in at.tabs[3].caption].count(wording.INSIGHTS_FAILED) == 1


def test_the_gate_line_is_said_once_after_is_it_still_improving(scored):
    at = _open_insights(AppTest.from_file(APP_PATH, default_timeout=180))
    tab = at.tabs[3]
    assert [c.value for c in tab.caption].count(wording.insights_gate(2)) == 1
    assert wording.insights_gate(2) == "The other answers start once 5 formulations are scored. So far: 2."
    assert wording.best_gate(4) == "Starts once 5 formulations have all their results. So far: 4."


def test_the_new_strings_pass_the_vocabulary_guard():
    from tests.test_food_bo import _BANNED
    for text in (wording.CHECKING, wording.insights_gate(3), wording.best_gate(3),
                 wording.repeats_line("0.3", 2), wording.improving_yes("3.1"),
                 wording.improving_barely("0.2", "0.3"), wording.improving_barely("0.2"),
                 wording.improving_not_lately(3)):
        assert not any(p.search(text) for p in _BANNED), text
    assert "first visit" not in wording.CHECKING


def test_the_header_is_filled_before_the_checks_run(insights_scored, monkeypatch):
    """Task 5 review M-2: during the first compute the header is not blank.
    Its early form goes up before the checks, then the verdict form over it."""
    log = []
    real_header, real_checks = ui_insights.header_text, insights.check_rounds

    def header(opt, h):
        log.append(("header", h))
        return real_header(opt, h)

    def checks(*a, **k):
        log.append(("checks", None))
        return real_checks(*a, **k)
    monkeypatch.setattr(ui_insights, "header_text", header)
    monkeypatch.setattr(insights, "check_rounds", checks)
    at = _open_insights(AppTest.from_file(APP_PATH, default_timeout=180))
    assert not at.exception
    first = log.index(("checks", None))
    assert ("header", None) in log[:first]
    assert log[-1][0] == "header" and log[-1][1] is not None


def test_the_headline_is_worked_out_once_per_run(insights_scored, monkeypatch):
    """Task 5 review M-3: the header and question 1 share one headline."""
    at = _open_insights(AppTest.from_file(APP_PATH, default_timeout=180))
    calls = []
    real = insights.headline

    def counted(*a, **k):
        calls.append(1)
        return real(*a, **k)
    monkeypatch.setattr(insights, "headline", counted)
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    assert not at.exception
    assert len(calls) == 1
