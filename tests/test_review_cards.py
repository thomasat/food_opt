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
# Review polish (2026-10-02): the header says "Review after Round N"; an Off
# card names the measurements that missed; "Check panel" is said once (the
# Next card's); a hidden gain is one line with its icon; with no Round open
# the Repeat request is the Next card's link, not a line too.
_HIDDEN = wording.HERO_HIDDEN_ICON + " "
# Review story: the Progress line goes from the start (the first scored
# Round's best) to the best, one decimal each; one number with one Round.
# Review story 2: the cards became the strip and What next?. Per status:
# the strip's word and line ("of 100" on a 0–100 score; Predictions
# without "Overall score:"; a warning line when there is one), then the
# lines under its section's answer; Next, What next?'s lines.
EXPECTED = {
    D.SNACK: (
        "Review after Round 1 · 3 formulations",
        [("Predictions", "Not yet", ["3 / 5 formulations scored"]),
         ("Progress", "Too early", ["· best 96.0 of 100"]),
         ("Next", "Make Round 2", ["No Round open"])],
        "Where the app looks next", ["Appears after 5 scored formulations."]),
    D.LINE_A: (
        "Review after Round 5 · 20 formulations · Round 6 open · Formulation 19 waiting",
        # Review four, ruling 1: the Overall score's record (was the
        # measurements' 33 / 35).
        # Review learned: the strip's Model item reads the model's verdicts
        # (Firmness's within-Round order beat chance).
        # Persona fixes 2: pairs under one tasting noise apart left out;
        # Firmness and Cook loss still clearly beat chance.
        # Persona fixes 2: the Overall score named when Predictable.
        [("Model", "Predicts", ["Overall score, Firmness"]),
         ("Progress", "No new best", ["· best 90.5 of 100"]),
         ("Next", "Record Round 6", ["4 formulations open", "Round 7: Repeat Formulation 15"])],
        "How the app chose Round 6", ["Suggestions go where expected gain peaks."]),
    D.LINE_B: (
        "Review after Round 5 · 21 formulations · Round 6 open",
        # Review four, ruling 1: the Overall score missed 3 of 16 (tail
        # 0.04: More misses than expected, Not yet); the measurements' 8 / 36
        # a plain line. The disagreeing Repeat stays the Next card's, red.
        # Review learned: Model's verdict word; the Overall score's red
        # coverage warning is kept (never folded).
        # The section repeats it in full under its answer.
        # Review learned 2: the section repeats no old-card line; the red
        # miss line stays in the strip.
        # Persona fixes 2: the too-close pairs left out, Partly; Flat after
        # a rise reads Plateau.
        [("Model", "Predicts", [":red[**3 / 16 missed**]"]),
         ("Progress", "Plateau since Round 3", ["· best 90.1 of 100"]),
         ("Next", ":red[Check panel first]", [":red[**Before Round 6**]",
                                              ":red[**Formulation 11's Repeat: 10.5 lower**]"])],
        "How the app chose Round 6", [_HIDDEN + wording.HERO_HIDDEN["verdict"]]),
    D.LINE_C: (
        "Review after Round 5 · 21 formulations · Round 6 open",
        # Review diagnosis ruling 2 (2026-10-02): the CV-widening rule no
        # longer says "Check your panel's consistency" without Repeat
        # evidence (Line C's one Repeat pair has 1 df); it asks for a Repeat.
        # Review learned 2: the Repeat request moved to What next?.
        [("Model", "Not yet", ["predictable"]),
         ("Progress", "Within noise", ["· best 89.8 (+2.5)"]),
         ("Next", "Record Round 6", ["4 formulations open", "Guided search, not a forecast",
                                     "Repeat Formulation 18 to check noise"])],
        "How the app chose Round 6", [_HIDDEN + wording.HERO_HIDDEN["noise"]]),
    D.UPGRADE: (
        "Review after Round 3 · 7 formulations · Formulation 8 waiting",
        [("Model", "Not yet", ["predictable"]),
         # A warning takes the strip's line (never folded).
         ("Progress", "Too early", [":red[**Rounds 1–2 lack Cook loss**]"]),
         # Review learned: What next? previews the next suggestions.
         ("Next", "Make Round 4", ["No Round open", f"**{wording.LEARNED_PREVIEW_TITLE}**"])],
        "Where the app looks next", [_HIDDEN + wording.HERO_HIDDEN["skill"]]),
    D.TIMING: (
        "Review after Round 12 · 60 formulations",
        [("Model", "Predicts", ["Overall score, Firmness +3"]),   # persona fixes 2
         ("Progress", "Plateau since Round 5", ["· best 92.5 of 100"]),   # persona fixes 2
         ("Next", "Make Round 13", ["No Round open", f"**{wording.LEARNED_PREVIEW_TITLE}**"])],
        "Where the app looks next", ["Suggestions go where expected gain peaks."]),
    D.PROCESS_ONLY: (
        "Review after Round 5 · 21 trials · Round 6 open",
        # Review diagnosis ruling 2 asks Line C's Repeat; here the Next card
        # already asks for Trial 21, so Predictions does not (one action,
        # one card: demo build).
        # Review four, ruling 1: the Overall score missed 4 of 16 (tail
        # 0.007): Off, on the score the suggestions use.
        # Review learned: Model's word; the red miss line kept.
        [("Model", "Predicts", [":red[**4 / 16 missed**]"]),
         ("Progress", "Within noise", ["· best 87.9 (+1.9)"]),
         ("Next", "Record Round 6", ["4 trials open", "Round 7: Repeat Trial 21"])],
        "How the app chose Round 6", [_HIDDEN + wording.HERO_HIDDEN["skill"]]),
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


_LABEL = re.compile(r"^(?::red\[\*\*(?P<red>.+?)\*\*\]|\*\*(?P<word>.+?)\*\*)\s*(?P<line>.*)$")


def strip(at):
    """Review story 2: the status strip, in screen order — [{label, help,
    word, line, key}] per item (word ':red[…]' when drawn red, line as
    drawn, ':red[**…**]' for a warning) — and the strip's buttons
    [(type, label)], read from the leaves above the first section."""
    items, buttons, cur = [], [], None
    # Review learned: the Predictions item reads "Model" once the model's
    # verdicts are in (the gate keeps "Predictions").
    labels = {wording.CARD_PROGRESS, wording.CARD_PREDICTIONS, wording.CARD_NEXT,
              wording.LEARNED_STRIP_LABEL}
    for el, folds, _ in _leaves(at.tabs[3]):
        text = _text(el)
        if el.type == "markdown" and text.startswith("#### "):
            break
        if el.type == "caption" and text in labels:
            cur = {"label": text, "help": el.proto.help or None, "word": None, "line": None,
                   "key": None}
            items.append(cur)
        elif el.type == "button" and el.key and el.key.startswith(ui_insights.SEE_CHART_KEY):
            m = _LABEL.match(el.label)
            cur["word"] = f":red[{m['red']}]" if m["red"] else m["word"]
            cur["line"], cur["key"] = m["line"], el.key
        elif el.type == "button" and items:     # the header's method link comes first
            buttons.append((el.proto.type, el.label))
    return items, buttons


def section(at, title):
    """[(element, text)] under one "#### " heading, in screen order."""
    out, on = [], False
    for el, folds, _ in _leaves(at.tabs[3]):
        text = _text(el)
        if el.type == "markdown" and text.startswith("#### "):
            on = text[5:] == title
            continue
        if el.type == "expander":
            on = False
        if on:
            out.append((el, text))
    return out


def _extras(at, title):
    """The card's further lines under a section's answer line."""
    rows = section(at, title)
    got, seen = [], False
    for el, text in rows:
        if el.type == "markdown" and el.proto.help and not seen:
            seen = True
            continue
        if seen and el.type == "markdown" and not el.proto.help and not text.startswith("**"):
            got.append(text)
            continue
        if seen:
            break
    return got


def cards(at, order="role"):
    """[{label, word, lines, help, buttons}] per status (review story 2:
    the cards became the strip and What next?): the strip's label, word
    and "?"; lines, the strip's line then the further lines under its
    section's answer (Progress, Predictions) or What next?'s lines (Next);
    buttons, What next?'s. In screen order with order="screen"; else
    Predictions, Progress, Next, the order the tests unpack them in."""
    items, _ = strip(at)
    found = []
    for it in items:
        card = {"label": it["label"], "word": it["word"], "help": it["help"], "lines": [],
                "buttons": []}
        if it["label"] == wording.CARD_NEXT:
            rows = _next_rows(at)
            # Persona fixes 2: What next?'s word is a small answer line
            # ('**Make Round 7.**'), no longer a subheader.
            word = lambda el, t: (el.type in ("subheader", "heading")
                                  or (el.type == "markdown" and re.fullmatch(r"(:red\[)?\*\*[^*]+\.\*\*\]?", t)))
            card["lines"] = [t for el, t in rows if el.type == "markdown" and not word(el, t)]
            card["buttons"] = [(el.proto.type, el.label) for el, _ in rows
                               if el.type == "button"]
            card["next_word"] = next((re.sub(r"^:red\[|\]$", "", t).strip("*").rstrip(".")
                                      for el, t in rows if word(el, t)), None)
            card["next_help"] = next((el.proto.help for el, t in rows if word(el, t)), None)
        else:
            if it["line"]:
                card["lines"].append(it["line"])
            title = _title_of(at, it["label"])
            card["lines"] += [t for t in _extras(at, title) if t != it["line"]]
        found.append(card)
    if order == "screen":
        return found
    rank = {wording.CARD_PREDICTIONS: 0, wording.LEARNED_STRIP_LABEL: 0,
            wording.CARD_PROGRESS: 1, wording.CARD_NEXT: 2}
    return sorted(found, key=lambda c: rank.get(c["label"], 3))


def _headings(at):
    return [_text(el)[5:] for el, _, _ in _leaves(at.tabs[3])
            if el.type == "markdown" and _text(el).startswith("#### ")]


def _title_of(at, label):
    """The section a strip item lands on, as the screen names it (Trial
    for a process-only study)."""
    heads = _headings(at)
    # Review learned: the Predictions/Model item lands on What has the model
    # learned?, which took Q1's place.
    want = {wording.CARD_PROGRESS: wording.Q2_TITLE,
            wording.CARD_PREDICTIONS: wording.LEARNED_TITLE,
            wording.LEARNED_STRIP_LABEL: wording.LEARNED_TITLE}[label]
    return next((h for h in heads if h == want), want)


def _next_rows(at):
    return section(at, wording.Q5_TITLE)


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
def test_one_primary_button_and_it_is_the_strips(screens, name):
    at = screens[name]
    primaries = [el.label for el, _, _ in _leaves(at.tabs[3])
                 if el.type == "button" and el.proto.type == "primary"]
    assert len(primaries) == 1
    # Review story 2: at the strip's right end; What next? has no second.
    _, buttons = strip(at)
    assert buttons == [("primary", primaries[0])]
    assert ("primary", primaries[0]) not in cards(at)[2]["buttons"]


@pytest.mark.parametrize("name", D.ALL)
def test_word_limits(screens, name):
    """Owner's rule: a big word at most 5 words; a line at most 10 (review
    story 2: a status line's limit; it was a card line's 6, and "of 100"
    takes two more: '88.4 → 96.4 of 100 · same for 3 Rounds')."""
    for c in cards(screens[name]):
        assert _words(c["word"]) <= 5, c["word"]
        for line in c["lines"]:
            assert _words(line) <= 10, line


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
    # Review learned: the Model item's "?" is the section's sentence, then
    # the ordering rule, then the Overall score's coverage and skill
    # against both baselines.
    # Persona fixes: a plain sentence first; the skill figures in plain
    # words, the terms of art in a last "For experts:" line.
    # Persona fixes 2: too-close pairs left out; Cook loss joins Firmness.
    assert plain.startswith("The model put the Overall score and Firmness in the right order "
                            "before tasting")
    assert "one-sided binomial test, p < 0.05, at least 5 pairs" in rule
    assert "Overall score: 13 of 15 landed in their predicted range" in plain
    expert = rule.split("\n\n")[-1]
    assert expert.startswith("For experts:")
    assert "Overall score: 13 of 15 inside the 95 % predicted range." in expert
    assert re.search(r"Skill over 15 checks: \+0\.36 against the average of earlier scores, "
                     r"[+−]\d\.\d\d against Best so far", expert), expert
    assert "Skill" not in rule.split("For experts:")[0]


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
    # Review polish, one action, one card: the Next card says it.
    p, _, n = cards(screens[D.LINE_B])
    # Review four, ruling 1: the check is the Next card's, in red. Review
    # learned: the strip's Model word reads the model's verdicts (Firmness).
    # Persona fixes 2: too-close pairs left out, Firmness Partly.
    assert p["word"] == "Predicts"     # the Overall score, named (persona fixes 2)
    assert ":red[**Check panel before Round 6**]" not in p["lines"]
    assert n["word"] == ":red[Check panel first]" and ":red[**Before Round 6**]" in n["lines"]


def test_waiting_and_the_repeat_request_stay_visible(screens):
    assert "Formulation 19 waiting" in outside(screens[D.LINE_A])[0][1]
    assert "Round 7: Repeat Formulation 15" in cards(screens[D.LINE_A])[2]["lines"]
    timing = cards(screens[D.TIMING])[2]
    # Review polish: with no Round open the link is the request, said once.
    assert "With a Repeat of Formulation 24" not in timing["lines"]
    assert ("tertiary", "Repeat Formulation 24") in timing["buttons"]


@pytest.mark.parametrize("name", D.ALL)
def test_one_fold_only_and_no_old_status_lines(screens, name):
    rows = outside(screens[name])
    folds = [t for k, t in rows if k == "expander"]
    # Review four: the questions are always open; Method and definitions
    # is the one fold left.
    assert folds == [wording.FOLD_METHOD]
    texts = [t for _, t in rows]
    assert not any(t.startswith(("Predictions:", "Progress:", "Next:")) for t in texts)
    assert wording.FOLD_EXPLORED not in folds


def test_sparklines_need_two_rounds():
    opt = D.load(D.SNACK)
    assert len(ui_insights.progress_spark_frame(opt)) == 1
    assert ui_insights.spark_chart(opt, None, "progress") is None
    line_a = D.load(D.LINE_A)
    frame = ui_insights.progress_spark_frame(line_a)
    assert list(frame["round"]) == sorted(frame["round"]) and len(frame) >= 2
    assert list(frame["best"]) == sorted(frame["best"])      # Best so far never falls
    assert ui_insights.spark_chart(line_a, None, "next") is None
