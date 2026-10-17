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
        # Review critique: the exact within-Round permutation test (the
        # binomial treated a Round's pairs as independent) and the noise
        # split over the measurements: the Overall score and Firmness now
        # lean right (Partly), not clearly; the strip reads "Partly:".
        # Specific wording (2026-10-12), re-pinned: each name with its pairs.
        [("Model", "Partly:", ["Overall score (5 of 21 called, 5 right) +2"]),
         ("Progress", "No new best", ["· best 90.5 of 100"]),
         # Round 2: one form of the Repeat request.
         ("Next", "Record Round 6", ["4 formulations open", "Repeat Formulation 15 to check noise"])],
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
        # Review critique: "Predicts:" (the strip's words, with a colon).
        # Round 2: Holm across the results checked: Firmness is Partly.
        [("Model", "Partly:", [":red[**3 / 16 missed**]"]),
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
        # Review critique: "Not predictable yet", the section's words.
        # Round 2: "Not yet:" with the names, the strip's one form.
        [("Model", "Not yet:", ["Overall score, Firmness +2"]),
         ("Progress", "Within noise", ["· best 89.8 (+2.5)"]),
         ("Next", "Record Round 6", ["4 formulations open", "Guided search, not a forecast",
                                     "Repeat Formulation 18 to check noise"])],
        "How the app chose Round 6", [_HIDDEN + wording.HERO_HIDDEN["noise"]]),
    D.UPGRADE: (
        "Review after Round 3 · 7 formulations · Formulation 8 waiting",
        [("Model", "Not yet:", ["Overall score, Firmness +2"]),   # round 2: one form
         # A warning takes the strip's line (never folded).
         ("Progress", "Too early", [":red[**Rounds 1–2 lack Cook loss**]"]),
         # Review learned: What next? previews the next suggestions.
         ("Next", "Make Round 4", ["No Round open", f"**{wording.LEARNED_PREVIEW_TITLE}**"])],
        "Where the app looks next", [_HIDDEN + wording.HERO_HIDDEN["skill"]]),
    D.TIMING: (
        "Review after Round 12 · 60 formulations",
        # Specific wording (2026-10-12), re-pinned: each name with its pairs.
        [("Model", "Predicts:", ["Overall score (78 of 110 called, 64 right) +4"]),   # review critique: "Predicts:"
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
        [("Model", "Predicts:", [":red[**4 / 10 missed**]"]),   # review critique: "Predicts:"
         ("Progress", "Within noise", ["· best 87.9 (+1.9)"]),
         ("Next", "Record Round 6", ["4 trials open", "Repeat Trial 21 to check noise"])],
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


# Final demo (B1): the strip's Stop is ':orange[**Stop**] 1.7 below top; …'.
_LABEL = re.compile(r"^(?::red\[\*\*(?P<red>.+?)\*\*\]|(?::orange\[)?\*\*(?P<word>.+?)\*\*\]?)"
                    r"\s*(?P<line>.*)$")


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
        # Review critique (fix 5): the Forecast check opens the model's
        # section, right under What next?: its own block, not What next?'s.
        if el.type == "markdown" and text == f"**{wording.FORECAST_CHECK_TITLE}**":
            on = False
        if on:
            out.append((el, text))
    return out


def _extras(at, title):
    """The card's further lines under a section's answer line."""
    rows = section(at, title)
    got, seen = [], False
    # Slim Review (re-pinned): in the Forecast check the model's answer is
    # its verdict line, after the check's own lines.
    verdict = title == wording.FORECAST_CHECK_TITLE
    for el, text in rows:
        if verdict and not re.match(r"(So far: |Predicts: |Partly: |Not yet: )", text):
            if not seen:
                continue
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
    # Slim Review (re-pinned): it lands on the Forecast check, which now
    # holds the model's verdict line.
    want = {wording.CARD_PROGRESS: wording.Q2_TITLE,
            wording.CARD_PREDICTIONS: wording.FORECAST_CHECK_TITLE,
            wording.LEARNED_STRIP_LABEL: wording.FORECAST_CHECK_TITLE}[label]
    return next((h for h in heads if h == want), want)


def _next_rows(at):
    # Spec §11: no Tasting noise control in What next? (it is in Set up).
    rows = section(at, wording.Q5_TITLE)
    assert wording.TASTING_NOISE_LABEL not in [t for _, t in rows]
    return rows


def outside(at):
    """[(type, text)] outside the cards and outside every fold."""
    return [(el.type, _text(el)) for el, folds, col in _leaves(at.tabs[3])
            if col is None and not folds and _text(el)]


def _words(text):
    plain = re.sub(r":\w+\[(.*?)\]", r"\1", text).replace("**", "")
    return len([w for w in plain.split() if re.search(r"\w", w)])   # as the DOM count


# ------------------------------------------------------------------ #


@pytest.mark.parametrize("name", D.ALL)
def test_word_limits(screens, name):
    """Owner's rule: a big word at most 5 words; a line at most 10 (review
    story 2: a status line's limit; it was a card line's 6, and "of 100"
    takes two more: '88.4 → 96.4 of 100 · same for 3 Rounds')."""
    for c in cards(screens[name]):
        assert _words(c["word"]) <= 5, c["word"]
        for line in c["lines"]:
            assert _words(line) <= 10, line


