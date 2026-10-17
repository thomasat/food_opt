"""Review four (prototype, 2026-10-02; owner: "clean up the Review tab to
cleanly (and visually) answer these 4 questions"). Under the header and
the three cards, four question sections, always open, in this order: Q1
"Can we trust the predictions?" beside Q2 "Are we improving?", then Q3
"What drives each result?" and Q4 "What have we tried?". Each is a small
heading, one answer line opening with its card's bold word, ONE chart and
its caption. Each card links to its question ("See chart ↓"). The hero,
"Can you have both?", the triangle and their folds are gone.

Ruling 1: the Predictions word reads the Overall score's Round-ahead record
(coverage + skill); the measurements' misses are a secondary, non-red line.
Ruling 2: Expected gain is never hidden: greyed, with its reason."""
import json
import os
import re
import shutil

import pytest
from streamlit.testing.v1 import AppTest

import bo_views
import insights
import insights_jobs
import ui_insights as U
import wording
from tests import demo_projects as D
from tests.test_app_ui import APP_PATH
from tests.test_food_bo import _BANNED
from tests.test_review_cards import _leaves, _text, cards, screens  # noqa: F401 (fixture)
from tests.test_review_diag import ALL10, R5, R6

# Review learned: "What has the model learned?" took Q1's place.
# Slim Review (owner, 2026-10-08, re-pinned): the model's verdicts folded into
# the Forecast check, which is now the model's section.
# One-glance Review (owner, 2026-10-14): What drives each result? and What
# have we tried? are gone.
QUESTIONS = [wording.FORECAST_CHECK_TITLE, wording.Q2_TITLE]
# Review story: the screen asks them as a story — improving first, then
# trust, drivers, what was tried.
# Review critique (fix 5, re-pinned): Are we improving?, What next? (the stop
# signal), What has the model learned? (the Forecast check just above it),
# What have we tried?, then What drives each result? as a closed fold.
# Slim Review (re-pinned): the Forecast check (with the model's verdict line)
# in the model's place; What have we tried? a closed fold too.
STORY = [wording.Q2_TITLE, wording.Q5_TITLE, wording.FORECAST_CHECK_TITLE]
CHART = ("arrow_vega_lite_chart", "vega_lite_chart")


def _words(text):
    plain = re.sub(r":\w+\[(.*?)\]", r"\1", text).replace("**", "")
    return len([w for w in plain.split() if re.search(r"\w", w)])


def sections(at):
    """{question: [(type, text)]} in screen order, from each heading to the
    next (the last one to Method and definitions)."""
    out, cur = {}, None
    for el, folds, _ in _leaves(at.tabs[3]):
        text = _text(el)
        if el.type == "markdown" and text.startswith("#### "):
            # Slim Review: the Forecast check is the model's section now.
            cur = text[5:]
            out[cur] = []
            continue
        if el.type == "expander":
            if el.label == wording.LEARNED_DETAILS_TITLE:
                continue    # slim Review: the Forecast check's own details fold
            # Review critique (fix 4): What drives each result? is a fold.
            # Round 2: Next suggestions folds under Stop, inside What next?.
            # Slim Review: What have we tried? is a fold too.
            cur = (el.label if not folds and el.label not in (
                wording.FOLD_METHOD, wording.LEARNED_PREVIEW_TITLE) else None)
            if cur is not None:
                out[cur] = []
            continue
        if el.type == "markdown" and text == f"**{wording.FORECAST_CHECK_TITLE}**":
            cur = None      # the Forecast check's own block, above the model's section
        if cur is not None and (text or el.type in CHART):
            out[cur].append((el.type, text))
    return out


# ------------------------------------------------------------------ #

def _changes_drawn(opt):
    """1 when What changed draws its picture (Best so far is not the start
    and a row changed), else 0."""
    ch = U.what_changed(opt)
    return int(ch is not None and not ch.same and any(r.changed for r in ch.rows))


@pytest.mark.parametrize("name", D.ALL)
def test_one_primary_on_the_screen(screens, name):  # noqa: F811
    at = screens[name]
    primaries = [el for el, _, _ in _leaves(at.tabs[3])
                 if el.type == "button" and el.proto.type == "primary"]
    assert len(primaries) == 1


REMOVED_TEXT = (wording.PLOT_PAIR, wording.PLOT_TRIANGLE, wording.PLOT_CHOOSES,
                "How the app chose", "Where the app looks next", wording.PAIR_FOLD,
                wording.FOLD_EXPLORED, wording.FOLD_MORE, wording.FOLD_TRIED,
                wording.PLOT_MAP, "What drives each result?", "What have we tried?",
                "Of pairs called, % right")


@pytest.mark.parametrize("name", D.ALL)
def test_the_removed_charts_are_gone(screens, name):  # noqa: F811
    at = screens[name]
    for el, _, _ in _leaves(at.tabs[3]):
        text = _text(el) or ""
        spec = el.proto.spec if el.type in CHART else ""
        for gone in REMOVED_TEXT:
            assert gone not in text and gone not in spec, (gone, text[:80])
    for fn in ("chooses_chart", "chooses_data", "pair_chart", "mixture_chart", "_hero",
               "_triangle", "_more_charts", "explored_fold", "tried_fold", "path_chart",
               "change_chart", "gain_chart", "search_path_chart", "profile_chart", "_profiler",
               "learned_chart", "called_chart", "_tried_section", "_drives_section"):
        assert not hasattr(U, fn), fn
    # The map stays on Make a round's Why these.
    assert hasattr(U, "explore_map_chart")


# ------------------------------------------------------------------ #
#  Ruling 1: the verdict on the Overall score
# ------------------------------------------------------------------ #

# (row, measurements' verdict) per demo project, from its own checks.
VERDICT = {
    D.SNACK: ("gate", "none"),
    D.LINE_A: ("reliable", "within"),        # score 13/15, skill 0.36
    D.LINE_B: ("caution", "unreliable"),     # score 13/16 (tail 0.04); measurements 28/36
    D.LINE_C: ("no_skill", "within"),        # score 14/16, skill -0.47
    D.UPGRADE: ("too_few", "early"),
    D.TIMING: ("reliable", "within"),        # score 54/55
    D.PROCESS_ONLY: ("off_coverage", "within"),   # score 12/16 (tail 0.007)
}


@pytest.mark.parametrize("name", D.ALL)
def test_the_word_reads_the_overall_score(name):
    opt = D.load(name)
    checks = insights.check_rounds(opt) if opt.complete_count() >= insights.MIN_ROWS else []
    f = insights.status_facts(opt, checks)
    assert (f.prediction_row, f.measure_verdict) == VERDICT[name]
    assert f.measures_miss == (f.measure_verdict in ("unreliable", "caution"))


def test_score_verdict_rule():
    h = insights.Headline("verdict", [3], 30, 36, "unreliable",
                          insights.ScoreRecord(14, 15, 0.0, 1.0, [2, 3]), [])
    assert insights.score_verdict_state(None, h, []) == "within"
    h.score = insights.ScoreRecord(9, 9, 0.0, 1.0, [2, 3])
    assert insights.score_verdict_state(None, h, []) == "early"    # under SKILL_MIN checks
    h.score = insights.ScoreRecord(12, 16, 0.0, 1.0, [2, 3])
    assert insights.score_verdict_state(None, h, []) == "unreliable"
    h.score = insights.ScoreRecord(13, 16, 0.0, 1.0, [2, 3])
    assert insights.score_verdict_state(None, h, []) == "caution"
    # A disagreeing Repeat is no Predictions row any more: the Next card's.
    assert insights.score_prediction_row(20, "within", None, 0) == "too_few"
    assert "off_repeat" not in {insights.score_prediction_row(20, v, None, 12)
                                for v in ("unreliable", "caution", "within", "wide")}


# Review story 2: the strip's line, without "Overall score:" (the label
# says Predictions; the answer line and the "?" still name the score).
# Review learned: the item is Model, its word the model's verdicts; the
# Overall score's red miss line is kept (never folded).
CARDS = {
    # Persona fixes 2: too-close pairs left out of the counts.
    # Review critique: the exact permutation test leaves them leaning right
    # (Partly); the strip's words now end in a colon.
    # Specific wording (2026-10-12), re-pinned: each name with its pairs.
    D.LINE_A: ("Partly:", ["Overall score (5 of 21 called, 5 right) +2"]),
    # Review learned 2: the section repeats no old-card line.
    D.LINE_B: ("Partly:", [":red[**3 / 16 missed**]"]),     # round 2: Holm
    D.PROCESS_ONLY: ("Predicts:", [":red[**4 / 10 missed**]"]),
    D.TIMING: ("Predicts:", ["Overall score (78 of 110 called, 64 right) +4"]),
}


def test_evidence_reports_both():
    assert wording.ev_predictions(10, 12, "0.69", 12, 27, 36) == (
        "Overall score 10 of 12 in range, skill 0.69 (n = 12); measurements 27 of 36 in "
        "range")   # fix round 4: never a bare a/b


# ------------------------------------------------------------------ #
#  Ruling 2: Expected gain greyed, never hidden
# ------------------------------------------------------------------ #

def test_the_greyed_lines_say_why_and_that_suggestions_are_the_same():
    for text in wording.GAIN_GREYED.values():
        assert _words(text) <= 10 and "greyed" in text
    assert "Suggestions are made the same way regardless" in wording.tip_gain_greyed("x")
    # Review four polish: Why these says the profiler's own line.
    for reason in ("noise", "not_yet", "off"):
        assert wording.why_gain_greyed(reason) == wording.GAIN_GREYED[reason]


# ------------------------------------------------------------------ #
#  The yogurt pilot (shipped samples)
# ------------------------------------------------------------------ #

@pytest.fixture(scope="module")
def pilot(tmp_path_factory, pilot_dir):
    here = os.getcwd()
    work = tmp_path_factory.mktemp("review-four-pilot")
    for n in (R5, R6, ALL10):
        shutil.copy(pilot_dir / f"{n}.pkl", work)
    os.chdir(work)
    try:
        with bo_views.fresh():
            yield work
    finally:
        os.chdir(here)


def _open(name):
    at = AppTest.from_file(APP_PATH, default_timeout=900)
    at.session_state["_loaded_project"] = name
    at.run()
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    return at


def test_new_strings_pass_the_vocabulary_guard():
    texts = [*QUESTIONS, wording.SEE_CHART, wording.SEE_CHART_HELP, wording.Q_WAIT,
             wording.answer_line("Rising", "Best 98.33"),
             wording.CARD_MEASURES_MISS, wording.card_score_missed(3, 16),
             wording.on_score("In Round 3, 9 of 10 results landed."),
             wording.measures_sentence(27, 36, 4, True), wording.measures_sentence(27, 36, 4),
             wording.ev_predictions(10, 12, "0.69", 12, 27, 36), *wording.GAIN_GREYED.values(),
             wording.tip_gain_greyed("x"), wording.why_gain_greyed("not_yet"),
             wording.why_greyed_line("off"), wording.why_greyed_line("not_yet"),
             wording.GAIN_REASON_NOT_YET, wording.GAIN_REASON_OFF,
             wording.Q2_X_TITLE, wording.round_tick(3)]
    for text in texts:
        assert not any(p.search(text) for p in _BANNED), text
        assert "spread" not in text.lower(), text
    # "Expected gain" is the gain's one name.
    for text in wording.GAIN_GREYED.values():
        assert "Expected gain" in text or "expected gain" in text
    # Process-only studies say Trial.
    opt = D.load(D.PROCESS_ONLY)
    assert "trials" in wording.for_project(opt, wording.Q_WAIT)


# ------------------------------------------------------------------ #
#  Review four polish (2026-10-02)
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("word, reasons, want", [
    ("reliable", [], None),
    # The measurements' own verdict no longer greys: only the card's word.
    ("not_yet", [], "not_yet"),
    ("off", [], "off"),
    ("reliable", ["noise"], "noise"),
    ("reliable", ["floor", "skill"], "noise"),
    ("reliable", ["skill"], "not_yet"),
    ("not_yet", ["noise"], "not_yet"),     # the card's word first: never contradicted
])
def test_gain_greying_follows_the_predictions_card(monkeypatch, word, reasons, want):
    monkeypatch.setattr(insights, "score_predictions", lambda opt, checks: word)
    monkeypatch.setattr(insights, "noise_guard",
                        lambda opt, score, checks=(), *a, **k: insights.Guard(list(reasons)))
    assert U.gain_grey(object(), None, ()) == want


@pytest.mark.parametrize("name", [D.LINE_A, D.LINE_B])
def test_score_predictions_is_the_cards_word(tmp_path, monkeypatch, name):
    """insights.score_predictions reads what status_facts reads (Line A
    Reliable, Line B Not yet: Line B's gain stays greyed, its line now the
    card's word, not the measurements')."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(insights_jobs, "_JOBS", {})
    monkeypatch.setattr(insights_jobs, "_INFLIGHT", {})
    monkeypatch.setattr(U.st, "session_state", {})
    with bo_views.fresh():
        opt = D.load(name)
        checks = U.checks_here(opt)
        facts = insights.status_facts(opt, checks)
        assert insights.score_predictions(opt, checks) == facts.predictions
        assert facts.predictions == ("reliable" if name == D.LINE_A else "not_yet")


# Review critique (fix 5): the parallel coordinates (Show as lines) were cut, and
# test_q4_is_coloured_by_role with them.


