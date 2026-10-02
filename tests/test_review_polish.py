"""Review polish (2026-10-02, demo prep): the header says what the Review
is about; an Off card names which numbers missed; each card's "?" opens
with its plain sentence, then one evidence line whose numbers are the
card's; a hidden gain is a deliberate line; the hero's two panels are two
full-width charts that line up; the method is one click away; the three
Make a round lines over the owner's limits are trimmed. Wording and
layout only."""
import re

import pytest

import insights
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
        plain, evidence, rule = c["help"].split("\n\n", 2)
        assert not plain.startswith("Evidence") and evidence.startswith("Evidence: ")
        assert ".." not in evidence and rule


def test_evidence_numbers_are_the_cards(screens):  # noqa: F811
    p, g, n = cards(screens[D.LINE_A])
    assert "33 / 35 in range" in p["lines"]
    ev = p["help"].split("\n\n")[1]
    assert "n = 35 measured values, Rounds 3–5" in ev and "2 outside, up to 3 expected" in ev
    # The Overall score's own Rounds and count, as More detail says them.
    assert "Overall score, Rounds 2–5: 13 of 15 in range, skill 0.36" in ev
    assert "posterior predictive" in ev
    assert "qLogNEI" in n["help"].split("\n\n")[1]


def test_off_names_the_measurements(screens):  # noqa: F811
    p = cards(screens[D.LINE_B])[0]
    assert p["lines"][0] == ":red[**Measurements: 8 / 36 missed (4 normal)**]"


def test_the_method_is_one_click_away(screens):  # noqa: F811
    at = screens[D.LINE_A]
    [b] = [b for b in at.tabs[3].button if b.key == U.METHOD_LINK_KEY]
    assert b.label == wording.METHOD_LINK and b.proto.type == "tertiary"
    for word in ("Gaussian-process", "qLogNEI", "Round-ahead", "noise threshold"):
        assert word in wording.METHOD_LINK_HELP


def test_hidden_gain_is_one_deliberate_line():
    for text in wording.HERO_HIDDEN.values():
        assert _words(text) <= 10, text
    assert wording.HERO_HIDDEN_ICON.startswith(":material/")


def test_the_hero_splits_into_two_aligned_charts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(U.st, "session_state", {})
    opt = D.load(D.LINE_A)
    computed = U._computed(opt, quiet=True)
    data = U.chooses_data(opt, computed["fitted"], None, mode=U.CHOOSES_LINE)
    top, bottom = U.chooses_chart(opt, data, heights=U.HERO_HEIGHTS, compact=True, split=True)
    for part in (top, bottom):
        spec = part.to_dict()
        assert spec["autosize"]["type"] == "fit-x"
        assert spec["config"]["axisY"]["minExtent"] == spec["config"]["axisY"]["maxExtent"]


def test_make_a_round_lines_within_the_limits():
    assert _words(wording.WHY_ROLE_KEY_SHORT) <= 8
    assert _words(wording.next_misses_short(16)) <= 10
    line = wording.round_track_short([3, 4, 5], 9, 9, see_insights=True)
    assert line == "Overall score: 9 of 9 in range. Measurements miss more."
    assert _words(line) <= 10
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
