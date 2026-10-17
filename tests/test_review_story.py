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
from tests.test_review_cards import _install, _leaves, _text, cards, section, strip
from tests.test_review_diag import ALL10, R6

# Review critique (fix 5, re-pinned): Are we improving?, What next?, What has
# the model learned? (the Forecast check above it), What have we tried?;
# What drives each result? is a closed fold under them (not a heading).
# Slim Review (owner, 2026-10-08, re-pinned): the Forecast check is the
# model's section (its verdict line in it); What have we tried? is a closed
# fold too, so the headings are three.
STORY = [wording.Q2_TITLE, wording.Q5_TITLE, wording.FORECAST_CHECK_TITLE]
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


def sections(at):
    """{question: [(type, text)]} in screen order."""
    out, cur = {}, None
    for el, folds, _ in _leaves(at.tabs[3]):
        text = _text(el)
        if el.type == "markdown" and text.startswith("#### "):
            cur = text[5:]
            out[cur] = []
            continue
        if el.type == "expander":
            # The Forecast check's details fold stays in its section.
            if el.label != wording.LEARNED_DETAILS_TITLE:
                cur = None
            continue
        if cur is not None and (text or el.type in CHART or el.type == "toggle"):
            out[cur].append((el.type, text or getattr(el, "label", "")))
    return out


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

def test_the_start_is_the_first_rounds_best():
    opt = D.load(D.LINE_A)
    i, r = U.story_start(opt)
    first = [k for k, b in enumerate(opt.batch_history) if b == r]
    assert r == min(b for b in opt.batch_history if b is not None)
    assert opt.Y_history[i] == max(opt.Y_history[k] for k in first if k in
                                   set(opt.best_candidates()))
    s, r0, b, rb, bno = U.story_numbers(opt)
    assert (s, r0) == (float(opt.Y_history[i]), r)
    assert b == float(opt.Y_history[opt.best_index()]) and bno == int(
        opt.formulation_ids[opt.best_index()])


def test_round_averages_are_the_rounds_means():
    opt = D.load(D.LINE_A)
    pool = set(opt.best_candidates())
    for r, a, b, mean, n in U.round_averages(opt):
        mine = [float(opt.Y_history[i]) for i, x in enumerate(opt.batch_history)
                if x == r and i in pool]
        assert n == len(mine) and mean == pytest.approx(sum(mine) / len(mine))


def test_the_progress_card_goes_from_start_to_best():
    assert wording.card_progress("87.8", "90.5", "same for 2 Rounds") == \
        "87.8 → 90.5 · same for 2 Rounds"
    assert wording.card_progress(None, "96.0", "Formulation 1") == "Best 96.0 · Formulation 1"
    # Review story 2: "of 100" on a 0–100 score only.
    assert wording.score_of("96.4", 100.0) == "96.4 of 100"
    assert wording.score_of("9.2", 10.0) == "9.2"
    assert wording.card_progress("88.4", wording.score_of("96.4", 100), "same for 3 Rounds") == \
        "88.4 → 96.4 of 100 · same for 3 Rounds"


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
    return [wording.card_progress("87.8", "90.5", "same for 2 Rounds"),
            wording.card_checked_so_far(7, 7, 10), wording.STORY_START, wording.STORY_BEST,
            wording.story_point("Start", "87.8", 1), wording.STORY_ROUND_AVERAGE,
            wording.STORY_EACH, wording.STORY_IMPROVING_CAPTION,
            wording.story_improving_help("83.8", "89.5", 1, 5),
            wording.story_target_start(1), wording.STORY_TARGET_BEST, wording.STORY_TARGET,
            wording.STORY_TARGET_TITLE, wording.STORY_TARGET_CAPTION,
            wording.story_target_row("Juiciness", "target", 7), wording.story_target_help(3, 21, 1),
            wording.STORY_FORECAST, wording.STORY_LANDED_IN, wording.STORY_LANDED_OUT,
            wording.STORY_NO_FORECAST, wording.STORY_TRUST_CAPTION,
            wording.story_trust_answer(13, 15), wording.story_trust_so_far(7, 7, 10),
            wording.story_misses(2, 0), wording.story_trust_help(13, 15, "x"),
            wording.STORY_VALUE]


def test_story_strings_pass_the_vocabulary_guard():
    for text in _story_strings():
        assert not any(p.search(text) for p in _BANNED), text
        assert "spread" not in text.lower(), text
        # No claim of Rounds or time saved, nor any comparison with another method.
        assert not re.search(r"\bsaved?\b|faster|fewer rounds|than (DoE|random|manual)",
                             text, re.I), text
    # Labels at most 5 words, captions at most 8, status lines at most 10.
    for label in (wording.STORY_ROUND_AVERAGE, wording.story_target_start(1),
                  wording.STORY_TARGET_TITLE, wording.STORY_FORECAST, wording.STORY_LANDED_IN,
                  wording.STORY_LANDED_OUT, wording.STORY_NO_FORECAST,
                  wording.story_target_row("Textured pea protein",
                                                                      "target", 13)):
        assert _words(label) <= 5, label
    for caption in (wording.STORY_IMPROVING_CAPTION, wording.STORY_TARGET_CAPTION,
                    wording.STORY_TRUST_CAPTION):
        assert _words(caption) <= 8, caption
    for status in (wording.card_progress("87.8", "90.5", "+2.5 (beat 2.0)"),
                   wording.card_checked_so_far(7, 7, 10), wording.story_trust_answer(13, 15),
                   wording.story_trust_so_far(7, 7, 10)):
        assert _words(status) <= 10, status
    # The section sits under its own heading in wording.py.
    src = open(wording.__file__).read()
    assert "# --- review story ---" in src
    assert src.index("# --- review story ---") < src.index("def card_progress(")


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


@pytest.mark.parametrize("name", [D.LINE_A, D.SNACK, D.PROCESS_ONLY])
def test_strip_word_limits(screens, name):
    items, _ = strip(screens[name])
    for i in items:
        assert _words(i["label"]) <= 5, i
        assert _words(f"{i['word']} {i['line']}") <= 10, i
    for el, text in section(screens[name], wording.Q5_TITLE):
        if el.type in ("markdown", "subheader"):
            assert _words(text) <= 10, text
        if el.type == "button":
            assert _words(text) <= 5, text


