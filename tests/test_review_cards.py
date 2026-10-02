"""Review cards prototype (owner, 2026-10-01: "a clearer, less wordy way to
present the review … understandable for someone that uses Bayesian
Optimization … and clear for a nontechnical food scientist").

The Review tab is a header line, three bordered cards side by side
(Predictions, Progress, Next: a small label, one big status word, one
number line, any warning on a line of its own, a sparkline), one hero
chart (the BO figure along Best so far -> Suggestion 1) and two folds,
What you've tried and More detail. Pinned here, per demo project as
AppTest sees it: each card's words, the one primary button (the Next
card's), warnings never folded, each card's "?" holding the full sentence
and the exact rule, the hero's lines and the two folds.

Each project is opened once."""
import json
import os
import re

import pytest
from streamlit.testing.v1 import AppTest

import bo_views
import ui_insights
import wording
from food_bo import FoodOptimizer
from tests import demo_projects as D
from tests.test_app_ui import APP_PATH

# (header, [(label, big word, [lines])], hero title, hero lines)
EXPECTED = {
    D.SNACK: (
        "Based on 3 formulations",
        [("Predictions", "Not yet", ["3 / 5 formulations scored"]),
         ("Progress", "Too early", ["Best 96.00 · Formulations 1 and 3"]),
         ("Next", "Make Round 2", ["No Round open"])],
        "Where the app looks next", ["Appears after 5 scored formulations."]),
    D.LINE_A: (
        "Based on 20 formulations · Round 6 open · Formulation 19 waiting",
        [("Predictions", "Reliable", ["33 / 35 in range"]),
         ("Progress", "No new best", ["Best 90.45 · Formulation 15"]),
         ("Next", "Record Round 6", ["4 formulations open", "Round 7: Repeat Formulation 15"])],
        "How the app chose Round 6", ["Suggestions go where expected gain peaks."]),
    D.LINE_B: (
        "Based on 21 formulations · Round 6 open",
        [("Predictions", ":red[Off]", [":red[**8 / 36 outside range (4 normal)**]",
                                       ":red[**Check panel before Round 6**]"]),
         ("Progress", "Flat", ["Best 90.09 · same for 2 Rounds"]),
         ("Next", ":red[Check panel first]", [":red[**Before Round 6**]",
                                              ":red[**Formulation 11's Repeat: 10.5 lower**]"])],
        "How the app chose Round 6", ["Suggestions go where expected gain peaks.",
                                      "Expected gain hidden: predictions miss too often."]),
    D.LINE_C: (
        "Based on 21 formulations · Round 6 open",
        # Review diagnosis ruling 2 (2026-10-02): the CV-widening rule no
        # longer says "Check your panel's consistency" without Repeat
        # evidence (Line C's one Repeat pair has 1 df); it asks for a Repeat.
        [("Predictions", "Not yet", ["No better than averaging",
                                     "Repeat Formulation 18 to check noise"]),
         ("Progress", "Within noise", ["Best 89.84 · +2.5 (needs 3.2)"]),
         ("Next", "Record Round 6", ["4 formulations open", "Guided search, not a forecast"])],
        "How the app chose Round 6", ["Suggestions go where expected gain peaks.",
                                      "Expected gain hidden: the model may be reading noise."]),
    D.UPGRADE: (
        "Based on 7 formulations · Formulation 8 waiting",
        [("Predictions", "Not yet", ["2 / 10 formulations checked"]),
         ("Progress", "Too early", ["Best 90.50 · Formulation 6",
                                    ":red[**Rounds 1–2 lack Cook loss**]"]),
         ("Next", "Make Round 4", ["No Round open"])],
        "Where the app looks next", ["Suggestions go where expected gain peaks.",
                                     "Expected gain hidden: predictions not yet reliable."]),
    D.TIMING: (
        "Based on 60 formulations",
        [("Predictions", "Reliable", ["58 / 60 in range"]),
         ("Progress", "Flat", ["Best 92.50 · same for 7 Rounds"]),
         ("Next", "Make Round 13", ["No Round open", "With a Repeat of Formulation 24"])],
        "Where the app looks next", ["Suggestions go where expected gain peaks."]),
    D.PROCESS_ONLY: (
        "Based on 21 trials · Round 6 open",
        # Review diagnosis ruling 2 asks Line C's Repeat; here the Next card
        # already asks for Trial 21, so Predictions does not (one action,
        # one card: demo build).
        [("Predictions", "Not yet", ["Not clearly better than averaging"]),
         ("Progress", "Within noise", ["Best 87.92 · +1.9 (needs 3.2)"]),
         ("Next", "Record Round 6", ["4 trials open", "Round 7: Repeat Trial 21"])],
        "How the app chose Round 6", ["Suggestions go where expected gain peaks.",
                                      "Expected gain hidden: predictions not yet reliable."]),
}


def _install(name):
    state = json.loads((D.HERE / f"{name}.json").read_text())
    opt = FoodOptimizer(state["project_name"])
    opt.import_json(state)
    opt.save()


def _open(name):
    at = AppTest.from_file(APP_PATH, default_timeout=600)
    at.session_state["_loaded_project"] = name
    at.run()
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    return at


@pytest.fixture(scope="module")
def screens(tmp_path_factory):
    here = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("review-cards"))
    try:
        with bo_views.fresh():
            out = {}
            for name in D.ALL:
                _install(name)
                out[name] = _open(name)
            yield out
    finally:
        os.chdir(here)


def _leaves(node, folds=()):
    """(element, fold labels it sits in, column index or None) in screen
    order; a fold's own row comes before its contents."""
    def walk(n, folds, col):
        k = -1
        for child in n.children.values():
            if child.type == "column":
                k += 1
                yield from walk(child, folds, k)
            elif child.type == "expander":
                yield child, folds, col
                yield from walk(child, folds + (child.label,), col)
            elif getattr(child, "children", None):
                yield from walk(child, folds, col)
            else:
                yield child, folds, col
    yield from walk(node, folds, None)


def _text(el):
    if el.type in ("markdown", "caption", "subheader", "heading"):
        return el.value
    if el.type in ("button", "expander"):
        return el.label
    return ""


def cards(at):
    """[(label, word, [lines], help)] per card, in column order."""
    out = {}
    for el, folds, col in _leaves(at.tabs[3]):
        if col is None or folds:
            continue
        card = out.setdefault(col, {"label": None, "word": None, "lines": [], "help": None,
                                    "buttons": []})
        if el.type == "caption" and card["label"] is None:
            card["label"] = el.value
        elif el.type in ("subheader", "heading") or (el.type == "markdown" and
                                                    card["word"] is None and el.proto.help):
            card["word"], card["help"] = el.value, el.proto.help
        elif el.type == "markdown":
            card["lines"].append(el.value)
        elif el.type == "button":
            card["buttons"].append((el.proto.type, el.label))
    return [out[k] for k in sorted(out)]


def outside(at):
    """[(type, text)] outside the cards and outside every fold."""
    return [(el.type, _text(el)) for el, folds, col in _leaves(at.tabs[3])
            if col is None and not folds and _text(el)]


def _words(text):
    plain = re.sub(r":\w+\[(.*?)\]", r"\1", text).replace("**", "")
    return len([w for w in plain.split() if re.search(r"\w", w)])   # as the DOM count


# ------------------------------------------------------------------ #

@pytest.mark.parametrize("name", D.ALL)
def test_each_card_says_its_word_and_lines(screens, name):
    header, expected, _, _ = EXPECTED[name]
    got = [(c["label"], c["word"], c["lines"]) for c in cards(screens[name])]
    assert got == expected


@pytest.mark.parametrize("name", D.ALL)
def test_the_header_is_one_short_line(screens, name):
    first = outside(screens[name])[0]
    assert first == ("caption", EXPECTED[name][0])


@pytest.mark.parametrize("name", D.ALL)
def test_one_primary_button_and_it_is_the_next_cards(screens, name):
    at = screens[name]
    primaries = [el.label for el, _, _ in _leaves(at.tabs[3])
                 if el.type == "button" and el.proto.type == "primary"]
    assert len(primaries) == 1
    nxt = cards(at)[2]
    assert ("primary", primaries[0]) in nxt["buttons"]


@pytest.mark.parametrize("name", D.ALL)
def test_word_limits(screens, name):
    """Owner's rule: a card line at most 6 words, a big word at most 5, the
    hero's caption at most 8 and its withheld reason at most 10."""
    for c in cards(screens[name]):
        assert _words(c["word"]) <= 5, c["word"]
        for line in c["lines"]:
            assert _words(line) <= 6, line
    _, _, _, hero = EXPECTED[name]
    assert _words(hero[0]) <= 8
    for line in hero[1:]:
        assert _words(line) <= 10


@pytest.mark.parametrize("name", D.ALL)
def test_each_card_tooltip_holds_the_full_sentence_then_the_exact_rule(screens, name):
    p, g, n = cards(screens[name])
    for c in (p, g, n):
        assert c["help"] and "\n\n" in c["help"], c
    # The full plain sentence first, then the BO / statistics statement.
    assert "Rise = Best so far minus the best before the latest scored Round" in g["help"]
    assert "Next-step rule that holds" in n["help"]


def test_predictions_tooltip_says_the_coverage_and_skill_exactly(screens):
    help_text = cards(screens[D.LINE_A])[0]["help"]
    plain, rule = help_text.split("\n\n", 1)
    assert "33 of 35 results landed inside the range the app predicted" in plain
    assert ("33 of 35 Round-ahead values inside their 95 % posterior predictive intervals"
            in rule)
    assert "skill 1 − SSE/SSE(mean of the rows before each Round) = 0.36 over 15" in rule


# Review diagnosis ruling 2: Line C and the process-only study have no
# warning left to show; their Repeat request is a neutral line, not red.
@pytest.mark.parametrize("name", [D.LINE_B, D.UPGRADE])
def test_warnings_are_never_folded(screens, name):
    """Off and every check-worthy line keep a red line of their own on the
    card, not in a fold or a tooltip only."""
    reds = [line for c in cards(screens[name]) for line in [c["word"], *c["lines"]]
            if line.startswith(":red[")]
    assert reds


def test_off_says_check_before_the_round(screens):
    p = cards(screens[D.LINE_B])[0]
    assert p["word"] == ":red[Off]"
    assert ":red[**Check panel before Round 6**]" in p["lines"]


def test_waiting_and_the_repeat_request_stay_visible(screens):
    assert "Formulation 19 waiting" in outside(screens[D.LINE_A])[0][1]
    assert "Round 7: Repeat Formulation 15" in cards(screens[D.LINE_A])[2]["lines"]
    timing = cards(screens[D.TIMING])[2]
    assert "With a Repeat of Formulation 24" in timing["lines"]
    assert ("tertiary", "Repeat Formulation 24") in timing["buttons"]


@pytest.mark.parametrize("name", D.ALL)
def test_the_hero_chart_and_its_lines(screens, name):
    _, _, title, lines = EXPECTED[name]
    rows = outside(screens[name])
    texts = [t for _, t in rows]
    at = texts.index(f"**{title}**")
    assert texts[at + 1:at + 1 + len(lines)] == lines


@pytest.mark.parametrize("name", D.ALL)
def test_two_folds_only_and_no_old_status_lines(screens, name):
    rows = outside(screens[name])
    folds = [t for k, t in rows if k == "expander"]
    assert folds == [wording.FOLD_TRIED, wording.FOLD_MORE]
    texts = [t for _, t in rows]
    assert not any(t.startswith(("Predictions:", "Progress:", "Next:")) for t in texts)
    assert wording.FOLD_EXPLORED not in folds


def test_more_detail_holds_everything_else(screens):
    inner = {el.label for el, folds, _ in _leaves(screens[D.LINE_A].tabs[3])
             if el.type == "expander" and folds[:1] == (wording.FOLD_MORE,)}
    for label in (wording.FOLD_PREDICTION_FIGURES, wording.FOLD_PROGRESS_FIGURES,
                  wording.FOLD_NEXT_REASONING, wording.FOLD_EXPLORED, wording.FOLD_METHOD,
                  wording.PAIR_FOLD,
                  wording.for_project(D.load(D.LINE_A),
                                      ui_insights._try_one_label(D.load(D.LINE_A)))):
        assert label in inner, (label, inner)


def test_sparklines_need_two_rounds():
    opt = D.load(D.SNACK)
    assert len(ui_insights.progress_spark_frame(opt)) == 1
    assert ui_insights.spark_chart(opt, None, "progress") is None
    line_a = D.load(D.LINE_A)
    frame = ui_insights.progress_spark_frame(line_a)
    assert list(frame["round"]) == sorted(frame["round"]) and len(frame) >= 2
    assert list(frame["best"]) == sorted(frame["best"])      # Best so far never falls
    assert ui_insights.spark_chart(line_a, None, "next") is None
