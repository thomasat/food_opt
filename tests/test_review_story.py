"""Review story (prototype, 2026-10-02): the Review read as a story, in the
order a food scientist asks — Are we improving? Can we trust the
predictions? What drives each result? What have we tried? — the cards in
the same order (Progress · Predictions · Next).

- Are we improving?: two model-free charts side by side — the best so far
  with each Round's average and the start and the best named; Closer to
  target, one row per measurement (the first Round's best hollow, Best so
  far in ink, the target dashed or an arrow at the better end).
- Can we trust the predictions?: forecast against actual over time — each
  checked formulation's Round-ahead 95 % range (light blue) and the score
  measured (filled inside, hollow outside); the answer counts the landings.
- What drives each result?: a finding, not the Next card's word.
- What have we tried?: the search path (one dot plot per row, Rounds down),
  the parallel coordinates on a "Show as lines" toggle.

Review story 2: the cards became one status strip above the charts
(Progress · Predictions · Next, the one primary at its right end) and the
closing section What next? (the Next card's word, lines and secondaries).

A synthetic project (the demo projects) runs in CI; the yogurt pilot's
checks skip without the confidential files, and assert no pilot names."""
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
from food_bo import FoodOptimizer
from tests import demo_projects as D
from tests.test_app_ui import APP_PATH
from tests.test_food_bo import _BANNED
from tests.review_screens import _install
from tests.test_review_diag import ALL10, R6

# Review critique (fix 5, re-pinned): Are we improving?, What next?, What has
# the model learned? (the Forecast check above it), What have we tried?;
# What drives each result? is a closed fold under them (not a heading).
# Slim Review (owner, 2026-10-08, re-pinned): the Forecast check is the
# model's section (its verdict line in it); What have we tried? is a closed
# fold too, so the headings are three.
STORY = [wording.Q2_TITLE, wording.Q5_TITLE, 'Forecast check']
CHART = ("arrow_vega_lite_chart", "vega_lite_chart")


def _words(text):
    plain = re.sub(r":\w+\[(.*?)\]", r"\1", text).replace("**", "")
    return len([w for w in plain.split() if re.search(r"\w", w)])


def _open(name):
    at = AppTest.from_file(APP_PATH, default_timeout=900)
    at.session_state["_loaded_project"] = name
    at.run()
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    return at


@pytest.fixture(scope="module")
def screens(tmp_path_factory):
    here = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("review-story"))
    try:
        with bo_views.fresh():
            out = {}
            for name in (D.LINE_A, D.SNACK, D.PROCESS_ONLY):
                _install(name)
                out[name] = _open(name)
            yield out
    finally:
        os.chdir(here)


# ------------------------------------------------------------------ #
#  The order: a story
# ------------------------------------------------------------------ #


# ------------------------------------------------------------------ #
#  Are we improving?
# ------------------------------------------------------------------ #


def test_round_averages_are_the_rounds_means():
    opt = D.load(D.LINE_A)
    pool = set(opt.best_candidates())
    for r, a, b, mean, n in U.round_averages(opt):
        mine = [float(opt.Y_history[i]) for i, x in enumerate(opt.batch_history)
                if x == r and i in pool]
        assert n == len(mine) and mean == pytest.approx(sum(mine) / len(mine))


# ------------------------------------------------------------------ #
#  Can we trust the predictions?
# ------------------------------------------------------------------ #

def _checks():
    S = insights.SCORE
    return [insights.Check(7, 3, S, 85.0, 80.0, 90.0, 86.0, True, True),
            insights.Check(8, 3, S, 85.0, 80.0, 90.0, 93.0, False, True),
            insights.Check(9, 3, S, 85.0, 80.0, 90.0, 77.0, False, True),
            insights.Check(10, 4, S, 86.0, 82.0, 90.0, 88.0, True, True),
            insights.Check(10, 4, "Firmness", 6.0, 5.0, 7.0, 6.2, True, True)]


# ------------------------------------------------------------------ #
#  Vocabulary and word limits
# ------------------------------------------------------------------ #

def _story_strings():
    return [wording.STORY_ROUND_AVERAGE, wording.STORY_TARGET]


def test_story_strings_pass_the_vocabulary_guard():
    for text in _story_strings():
        assert not any(p.search(text) for p in _BANNED), text
        assert "spread" not in text.lower(), text
        # No claim of Rounds or time saved, nor any comparison with another method.
        assert not re.search(r"\bsaved?\b|faster|fewer rounds|than (DoE|random|manual)",
                             text, re.I), text
    # Labels at most 5 words.
    assert _words(wording.STORY_ROUND_AVERAGE) <= 5


# ------------------------------------------------------------------ #
#  The yogurt pilot (shipped samples)
# ------------------------------------------------------------------ #

@pytest.fixture(scope="module")
def pilot(tmp_path_factory, pilot_dir):
    here = os.getcwd()
    work = tmp_path_factory.mktemp("review-story-pilot")
    for n in (R6, ALL10):
        shutil.copy(pilot_dir / f"{n}.pkl", work)
    os.chdir(work)
    try:
        with bo_views.fresh():
            yield work
    finally:
        os.chdir(here)


# ------------------------------------------------------------------ #
#  Review story 2: the strip above the charts, What next? at the end
# ------------------------------------------------------------------ #


