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
import ui_insights as U
import wording
from tests import demo_projects as D
from tests.test_app_ui import APP_PATH
from tests.review_screens import _leaves, _text, screens  # noqa: F401 (fixture)
from tests.test_review_diag import ALL10, R5, R6

# Review learned: "What has the model learned?" took Q1's place.
# Slim Review (owner, 2026-10-08, re-pinned): the model's verdicts folded into
# the Forecast check, which is now the model's section.
# One-glance Review (owner, 2026-10-14): What drives each result? and What
# have we tried? are gone.
QUESTIONS = ['Forecast check', wording.Q2_TITLE]
# Review story: the screen asks them as a story — improving first, then
# trust, drivers, what was tried.
# Review critique (fix 5, re-pinned): Are we improving?, What next? (the stop
# signal), What has the model learned? (the Forecast check just above it),
# What have we tried?, then What drives each result? as a closed fold.
# Slim Review (re-pinned): the Forecast check (with the model's verdict line)
# in the model's place; What have we tried? a closed fold too.
STORY = [wording.Q2_TITLE, wording.Q5_TITLE, 'Forecast check']
CHART = ("arrow_vega_lite_chart", "vega_lite_chart")


def _words(text):
    plain = re.sub(r":\w+\[(.*?)\]", r"\1", text).replace("**", "")
    return len([w for w in plain.split() if re.search(r"\w", w)])


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
                'Which formulations are worth trying?', "What drives each result?", "What have we tried?",
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
    # v3: the map went with Make a round's Why fold.
    assert not hasattr(U, "explore_map_chart")


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


# ------------------------------------------------------------------ #
#  Ruling 2: Expected gain greyed, never hidden
# ------------------------------------------------------------------ #


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


# ------------------------------------------------------------------ #
#  Review four polish (2026-10-02)
# ------------------------------------------------------------------ #


# Review critique (fix 5): the parallel coordinates (Show as lines) were cut, and
# test_q4_is_coloured_by_role with them.


