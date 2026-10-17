"""Review guide (owner, 2026-10-02: "I'm not guided to see the review -- it
leads me through the 1-3"): Review is the loop's fourth step, Set up ->
Make a round -> Results -> Review -> Make a round.

- After Save results (whole Round or part) Results' one lit button is
  "Review Round N", and it opens 4 · Review.
- v3: Make a round carries no line in the Review cards' words (no
  verdicts there); only "Review Round N" while a Round is due its Review.
- One lit button per screen.
- The welcome lists Review as step 4."""
import json
import os
import re

import pytest
from streamlit.testing.v1 import AppTest

import bo_views
import wording
from food_bo import FoodOptimizer
from tests import demo_projects as D
from tests.test_app_ui import APP_PATH, _tab_primaries
from tests.test_review_cards import cards
from ui_helpers import REVIEW_DOT_STYLE, REVIEW_DUE_KEY, open_rows

# The cards' (Predictions, Progress) words and the Next card's word when it
# is red, per demo project (tests/test_review_cards.py EXPECTED).
STATUS = {
    D.SNACK: ("Not yet", "Too early", None),
    # Review learned: past the gate the word is the Model item's (the
    # model's verdicts); the line stays red where the Overall score's
    # record was Off (the process-only study, 12 of 16).
    # Persona fixes 2: too-close pairs left out; Flat after a rise reads
    # Plateau since Round N.
    # Review critique: the strip's words ("Predicts:", "Partly:", "Not
    # predictable yet"); the permutation test leaves Line A leaning right.
    D.LINE_A: ("Partly:", "No new best", None),
    # Round 2: Holm (Line B's Firmness Partly); "Not yet:" the one form.
    D.LINE_B: ("Partly:", "Plateau since Round 3", "Check panel first"),
    D.LINE_C: ("Not yet:", "Within noise", None),
    D.PROCESS_ONLY: ("Predicts:", "Within noise", None),
}
RED = {D.LINE_B, D.PROCESS_ONLY}
SETUP, BATCH, RESULTS, REVIEW = range(4)


def _install(name):
    state = json.loads((D.HERE / f"{name}.json").read_text())
    opt = FoodOptimizer(state["project_name"])
    opt.import_json(state)
    opt.save()


def _app(name, tab):
    at = AppTest.from_file(APP_PATH, default_timeout=600)
    at.session_state["_loaded_project"] = name
    at.run()
    _go(at, tab)
    return at


def _go(at, tab):
    at.session_state["main_tab"] = tab
    at.run()
    assert not at.exception, [e.message for e in at.exception]


def _plain(text):
    return re.sub(r":red\[(.*)\]$", r"\1", text)


def _status_line(at):
    lines = [c.value for c in at.tabs[BATCH].caption
             if _plain(c.value).startswith("Review:")]
    assert len(lines) <= 1, lines
    return lines[0] if lines else None


@pytest.fixture(scope="module")
def walked(tmp_path_factory):
    """Per project: the cards' big words on Review, then Make a round's
    line, as the loop reaches it (Review is opened first: AppTest has no
    workers, and the line never fits a model of its own)."""
    here = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("review-guide"))
    try:
        with bo_views.fresh():
            out = {}
            for name in STATUS:
                _install(name)
                at = _app(name, wording.TAB_INSIGHTS)
                # Review story: the screen shows Progress, Predictions, Next;
                # read in the (Predictions, Progress, Next) order pinned below.
                # Review story 2: the words are the status strip's.
                words = [c["word"] for c in cards(at)]
                review_lit = _tab_primaries(at, REVIEW)
                _go(at, wording.TAB_BATCH)
                links = [b for b in at.tabs[BATCH].button if b.key == "link_review_status"]
                out[name] = (words, review_lit, _status_line(at), links,
                             _tab_primaries(at, BATCH))
            yield out
    finally:
        os.chdir(here)


@pytest.mark.parametrize("name", list(STATUS))
def test_one_lit_button_on_review_and_make_a_round(walked, name):
    _, review_lit, _, _, batch_lit = walked[name]
    assert len(review_lit) == 1, review_lit
    assert len(batch_lit) == 1, batch_lit


def _fill(at, opt, numbers):
    by = {o['name']: o for o in opt.objectives}
    for no in numbers:
        for name, obj in by.items():
            key = f"f{no}_{name}"
            box = [n for n in at.number_input if n.key == key]
            if box:
                box[0].set_value((float(obj['min_val']) + float(obj['max_val'])) / 2)
    at.run()


def _button(at, tab, key):
    return next(b for b in at.tabs[tab].button if b.key == key)


def _tab_dot(at):
    """Whether the Review tab's dot is drawn: its own style block, matched
    whole (a bare "2022" also matched a scroll script's time_ns() comment,
    so the test failed at some times of day)."""
    return any(h.proto.body.strip() == REVIEW_DOT_STYLE.strip() for h in at.get("html"))


def test_save_results_then_review_round_n_opens_review(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with bo_views.fresh():
        _install(D.LINE_A)
        at = _app(D.LINE_A, wording.TAB_BATCH)
        opt = at.session_state["optimizer"]
        no = opt.pending_batch_no
        numbers = [r['formulation'] for r in open_rows(opt)]
        assert len(numbers) == 4
        # Part of the Round: it stays open, and Review is still due.
        _fill(at, opt, numbers[:1])
        _button(at, BATCH, "save_results").click().run()
        assert not at.exception, [e.message for e in at.exception]
        assert at.session_state[REVIEW_DUE_KEY] == no
        # The reader lands back here: "Review Round N" is its one lit button
        # (the sheets step aside), in Make a round's status line.
        _go(at, wording.TAB_BATCH)
        assert _tab_primaries(at, BATCH) == [wording.review_round_label(no)]
        _go(at, wording.TAB_RESULTS)
        assert _tab_primaries(at, RESULTS) == [wording.review_round_label(no)]
        foot = _button(at, RESULTS, "foot_batch")
        assert foot.label == wording.back_to_batch_label(no, 3) and not foot.disabled
        assert _tab_dot(at), "no tab dot"
        _button(at, RESULTS, "review_round").click().run()
        assert at.session_state["main_tab"] == wording.TAB_INSIGHTS
        assert REVIEW_DUE_KEY not in at.session_state
        assert not _tab_dot(at)
        # Reviewed: Results' foot is the lit one again.
        assert _tab_primaries(at, RESULTS) == [wording.back_to_batch_label(no, 3)]
        # The rest of the Round: Save results lands on Results, Review lit.
        _go(at, wording.TAB_BATCH)
        _fill(at, opt, numbers[1:])
        _button(at, BATCH, "save_results").click().run()
        assert not at.exception, [e.message for e in at.exception]
        assert at.session_state["main_tab"] == wording.TAB_RESULTS
        assert _tab_primaries(at, RESULTS) == [wording.review_round_label(no)]
        assert _button(at, RESULTS, "foot_batch").label == wording.START_NEXT_BATCH
        # AppTest cannot run again on a tree that still holds the closed
        # Round's boxes: the same session state, in a new AppTest.
        due = at.session_state[REVIEW_DUE_KEY]
        at = AppTest.from_file(APP_PATH, default_timeout=600)
        at.session_state["_loaded_project"] = D.LINE_A
        at.run()
        at.session_state[REVIEW_DUE_KEY] = due
        _go(at, wording.TAB_RESULTS)
        assert _tab_primaries(at, RESULTS) == [wording.review_round_label(no)]
        _button(at, RESULTS, "review_round").click().run()
        assert at.session_state["main_tab"] == wording.TAB_INSIGHTS
        # Review's Next card leads back to Make a round.
        # Review v3: What next?'s primary names the next Round.
        assert _tab_primaries(at, REVIEW) == [wording.start_round_label(
            at.session_state["optimizer"].next_batch_no())]
        _button(at, REVIEW, "insights_foot").click().run()
        assert at.session_state["main_tab"] == wording.TAB_BATCH
        assert _status_line(at) is None          # v3: no verdict line here


def test_the_welcome_lists_review_as_step_four(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    steps = next(m.value for m in at.markdown if m.value.startswith("1. "))
    assert steps.splitlines()[-1] == wording.WELCOME_REVIEW_STEP
    assert steps.count("\n") == 3
    words = re.sub(r"\*\*|^4\. ", "", wording.WELCOME_REVIEW_STEP).split()
    assert len(words) <= 8, words
