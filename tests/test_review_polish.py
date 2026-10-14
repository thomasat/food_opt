"""Review polish (2026-10-02, demo prep): the header says what the Review
is about; an Off card names which numbers missed; each card's "?" opens
with its plain sentence, then one evidence line whose numbers are the
card's; a hidden gain is a deliberate line; the hero's two panels are two
full-width charts that line up; the method is one click away; the three
Make a round lines over the owner's limits are trimmed. Wording and
layout only."""
import re

import pytest

import ui_insights as U
import wording
from tests import demo_projects as D
from tests.test_food_bo import _BANNED
from tests.test_review_cards import cards, screens  # noqa: F401  (module fixture)


def _words(text):
    plain = re.sub(r":\w+\[(.*?)\]", r"\1", text).replace("**", "")
    return len([w for w in plain.split() if re.search(r"\w", w)])


def test_the_header_says_what_the_review_is():
    assert wording.review_header(20, 5, 6, 1, numbers=[19]) == (
        "Review after Round 5 · 20 formulations · Round 6 open · Formulation 19 waiting")
    assert wording.review_header(3, 1) == "Review after Round 1 · 3 formulations"


@pytest.mark.parametrize("name", [D.LINE_A, D.LINE_B])
def test_each_tooltip_is_plain_then_evidence_then_rule(screens, name):  # noqa: F811
    for c in cards(screens[name]):
        # Review critique (fix 13): plain first, then the evidence, all in
        # about 120 words; the rule is in Method and definitions when it
        # does not fit.
        plain, evidence = c["help"].split("\n\n", 2)[:2]
        assert not plain.startswith("Evidence") and evidence.startswith("Evidence: ")
        assert ".." not in evidence
        assert len(c["help"].split()) <= wording.TOOLTIP_WORDS


def test_evidence_numbers_are_the_cards(screens):  # noqa: F811
    p, g, n = cards(screens[D.LINE_A])
    # Review four, ruling 1: the card and its evidence lead with the
    # Overall score's record; the measurements' count follows.
    # Review learned: the Model item; its evidence the ordering counts,
    # then the Overall score's coverage and skill against both baselines.
    # Persona fixes 2: too-close pairs left out; Cook loss joins Firmness.
    # Review critique: Partly under the permutation test (Cook loss too: +1).
    # Specific wording (2026-10-12), re-pinned: each name with its pairs.
    assert p["lines"] == ["Overall score (5 of 21 called, 5 right) +2"]
    ev = p["help"].split("\n\n")[1]
    # Persona fixes 2: a result with no pair far enough apart is left out
    # (Line A's Overall score).
    assert ev.startswith("Evidence: pairs ordered right from earlier Rounds — ")
    # Round 2 (P3 #19): the coverage is the plain line's; the experts' line
    # keeps the skill figures and never repeats it.
    # Sceptic review (2026-10-14), re-pinned: the plain sentence carries each
    # Partly name's pairs called of all (79 words with the evidence's 10), so
    # the experts' first sentence (36 words) no longer fits the ~120 and is
    # left to Method and definitions, as brief() does with any paragraph.
    plain_part = p["help"].split("For experts:")[0]
    assert "Overall score: 13 of 15 landed in their predicted range" in plain_part
    assert "Overall score (5 of 21 pairs called, 5 right), Firmness (16 of 24 called, " \
           "12 right)" in plain_part
    assert "For experts:" not in p["help"] and "Skill over" not in p["help"]
    assert len(p["help"].split()) <= wording.TOOLTIP_WORDS
    assert "qLogNEI" in n["help"].split("\n\n")[1]


def test_off_names_the_measurements(screens):  # noqa: F811
    p = cards(screens[D.LINE_B])[0]
    # Review four, ruling 1: the red line is the Overall score's; the
    # measurements' misses a plain line under it.
    # Review story 2: the strip's line drops "Overall score:" (its label
    # says Predictions); the plain line sits under the trust answer.
    # Review learned: the Model item keeps both lines; the section repeats
    # the red one in full under its answer.
    # Review learned 2 (owner: not wordy): the section repeats no old-card
    # line; "Measurements miss more than expected" contradicted the
    # per-measurement verdicts and is gone. The red line stays in the strip.
    assert p["lines"] == [":red[**3 / 16 missed**]"]


def test_the_method_is_one_click_away(screens):  # noqa: F811
    at = screens[D.LINE_A]
    [b] = [b for b in at.tabs[3].button if b.key == U.METHOD_LINK_KEY]
    assert b.label == wording.METHOD_LINK and b.proto.type == "tertiary"
    # One-glance Review: the fold is one line per term; the full method is
    # in the README.
    assert wording.FOLD_METHOD in wording.METHOD_LINK_HELP


def test_hidden_gain_is_one_deliberate_line():
    for text in wording.HERO_HIDDEN.values():
        assert _words(text) <= 10, text
    assert wording.HERO_HIDDEN_ICON.startswith(":material/")


def test_make_a_round_lines_within_the_limits():
    assert _words(wording.WHY_ROLE_KEY_SHORT) <= 8
    assert _words(wording.next_misses_short(16)) <= 10
    line = wording.round_track_short([3, 4, 5], 9, 9, see_insights=True)
    # Review critique (fix 9): "Measurements: more misses than expected."
    assert line == "Overall score: 9 of 9 in range. Measurements: more misses than expected."
    assert _words(line) <= 12
    # No warning: the Rounds stay on the line.
    assert wording.round_track_short([2, 3], 5, 6) == "Overall score, Rounds 2–3: 5 of 6 in range."


def test_new_strings_pass_the_vocabulary_guard():
    texts = [wording.review_header(3, 1), wording.REVIEW_ABOUT,
             wording.card_values_missed(9, 36, 4), wording.EV_ROUND_AHEAD, wording.EV_METHOD,
             wording.EV_MODEL_FREE, wording.METHOD_LINK, wording.METHOD_LINK_HELP,
             wording.hero_axis(), wording.hero_axis(True), *wording.HERO_HIDDEN.values(),
             wording.tip_hero_hidden("x"), wording.TIP_HERO_TOP, wording.WHY_ROLE_KEY_SHORT,
             wording.next_misses_short(16), wording.TRACK_VALUES_MISS,
             wording.SETUP_BUTTON_SHORT, wording.ev_suggestions(), wording.ev_rule("Misses")]
    for text in texts:
        assert not any(p.search(text) for p in _BANNED), text
        assert "spread" not in text.lower(), text
