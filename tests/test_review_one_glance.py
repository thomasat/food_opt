"""One-glance Review (owner, 2026-10-14; spec 2026-10-13 §10): the Review
below the Round panel keeps What next?, the Forecast check (its one Round
line, then a Details fold: one plain sentence per result with pairs and the
one forecast against measured chart) and Method and definitions (seven
one-line definitions and a link to the README). What have we tried? and
What drives each result? are gone, with everything that only they used.
Fix round 1 (r2-fix1-findings): the Round line back; each definition true
to the rule the app applies; measured-equal pairs never "too close"."""
import os
import re
import types

import pytest

import learned
import ui_insights as U
import wording

README = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "README.md")


def _words(text):
    plain = re.sub(r":\w+\[(.*?)\]", r"\1", text).replace("**", "")
    return len([w for w in plain.split() if re.search(r"\w", w)])


# ------------------------------------------------------------------ #
#  The Forecast check's details: one plain sentence per result
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("right, called, close, ties, want", [
    (5, 5, 6, 1, "Overall score: of 12 pairs, 5 called, all 5 right; 6 too close, 1 measured "
                 "equal."),
    (5, 9, 3, 0, "Overall score: of 12 pairs, 9 called, 5 right; 3 too close."),
    (0, 4, 0, 2, "Overall score: of 6 pairs, 4 called, none right; 2 measured equal."),
    (0, 0, 3, 0, "Overall score: of 3 pairs, none called; 3 too close."),
    (0, 0, 0, 3, "Overall score: of 3 pairs, none called; 3 measured equal."),
    (1, 1, 2, 0, "Overall score: of 3 pairs, 1 called, right; 2 too close."),
    (0, 1, 0, 0, "Overall score: of 1 pair, 1 called, wrong."),
    (2, 2, 0, 0, "Overall score: of 2 pairs, 2 called, both right."),
])
def test_one_sentence_per_result(right, called, close, ties, want):
    assert wording.pairs_sentence("Overall score", right, called, close, ties) == want


def test_measured_equal_pairs_are_never_said_too_close_to_call():
    """Finding 4: of 12 pairs, 5 called, 6 too close and 1 measured equal —
    the tie is named, never folded into 'not far enough apart to call';
    'all' only when every called pair was right."""
    text = wording.pairs_sentence("Overall score", 5, 5, 6, 1)
    assert "1 measured equal" in text and "6 too close" in text
    assert "far enough apart" not in text
    assert not re.search(r"\ball\b", wording.pairs_sentence("Overall score", 4, 5, 6, 1))


def _ordering(name, right, pairs, ties=0, close=0):
    return learned.Ordering(name, right=right, pairs=pairs, ties=ties, close=close)


# ------------------------------------------------------------------ #
#  What is gone
# ------------------------------------------------------------------ #


# ------------------------------------------------------------------ #
#  Method and definitions
# ------------------------------------------------------------------ #


def test_the_readme_says_the_v3_review_in_25_lines():
    """v3: four tabs; the Review shows best so far, best vs targets and the
    amounts in the top formulations; single tastings; no stop rule and no
    forecasts on screen."""
    text = open(README, encoding="utf-8").read()
    assert "## How the Review works" in text
    section = text.split("## How the Review works", 1)[1].split("\n## ", 1)[0]
    lines = [line for line in section.splitlines() if line.strip()]
    assert 3 <= len(lines) <= 25
    for term in ("four tabs", "Best so far", "Best vs targets",
                 "Amounts in the top formulations", "single tasting", "no stop rule",
                 "not shown on screen", "Gaussian process"):
        assert term in section, term
    for gone in ("Holm", "calibration factor", "No forecast", "Tasting noise", "**Stop.**",
                 "95 % range", "Round-ahead"):
        assert gone not in section, gone


