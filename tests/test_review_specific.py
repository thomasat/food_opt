"""Specific wording (owner, 2026-10-12: "I also don't like vague wording like
'Measurements: more misses than expected.'"). Each rewritten line names the
thing, the count and what it is measured against, from the app's own data;
within the owner's limits (status ~10 words, answer lines ~14, captions
8-10); and passes the vocabulary guard.

Synthetic checks and hand-made orderings only; no pilot data."""
import re
from types import SimpleNamespace as NS

import pytest

import insights
import learned
import wording
from insights import Check
from tests.test_food_bo import _BANNED

NAMES = ("Creaminess", "Tangy", "Thickness uniformity", "Consistency")


def _words(text):
    plain = re.sub(r":\w+\[(.*?)\]", r"\1", str(text)).replace("**", "")
    return len([w for w in plain.split() if re.search(r"\w", w)])


def _clean(text):
    assert not any(p.search(text) for p in _BANNED), text
    for word in ("panel", "replay", "recipe", "held", "scale", "importance", "interval"):
        assert word not in text.lower(), (word, text)


def _stub(names=NAMES):
    objectives = [{"name": n, "goal": "target", "target": 7, "weight": 100 / len(names),
                   "min_val": 0.0, "max_val": 15.0} for n in names]
    return NS(objectives=objectives, utility_ceiling=lambda: 100.0,
              measurements_by_importance=lambda: objectives)


def _check(no, r, name, inside, top=15.0):
    pred, sd = 7.0, 0.5
    low, high = pred - insights.Z95 * sd, pred + insights.Z95 * sd
    meas = pred if inside else high + 1.0
    return Check(no, r, name, pred, low, high, meas, low <= meas <= high, True,
                 raw_mean=pred, sd=sd)


def _checks(misses_by_round):
    """Rounds 1-4, three formulations each, every measurement and the
    Overall score checked; misses_by_round {round: {name: k}} puts k misses
    on that measurement in that Round (the Overall score never misses)."""
    out, no = [], 0
    for r in (1, 2, 3, 4):
        for k in range(3):
            no += 1
            for name in NAMES:
                miss = k < misses_by_round.get(r, {}).get(name, 0)
                out.append(_check(no, r, name, not miss))
            out.append(_check(no, r, insights.SCORE, True))
    return out


# ------------------------------------------------------------------ #
#  1 · The measurements' misses: counted, named, against 1 in 20
# ------------------------------------------------------------------ #

def test_value_misses_counts_the_verdicts_rounds_by_measurement():
    opt = _stub()
    # Round 1 is outside the verdict's window (the last 3 Rounds hold 36
    # values): its misses must not count.
    checks = _checks({1: {"Tangy": 3}, 2: {"Creaminess": 2}, 3: {"Creaminess": 2, "Tangy": 1},
                      4: {"Thickness uniformity": 2, "Consistency": 1}})
    h = insights.headline(opt, checks)
    assert h.rounds == [2, 3, 4] and h.total == 36
    m = insights.value_misses(opt, checks, h)
    assert (m.misses, m.total, m.rounds) == (8, 36, [2, 3, 4])
    assert m.misses == h.total - h.inside            # the verdict's own count
    # Most misses first; Results' order on a tie.
    assert m.items == [("Creaminess", 4, 9), ("Thickness uniformity", 2, 9), ("Tangy", 1, 9),
                       ("Consistency", 1, 9)]
    # One definition of "expected": the verdict's own allowance.
    assert m.allowed == insights.allowed_misses(36) == 4
    assert m.expected == pytest.approx(0.05 * 36)


def test_value_misses_none_without_values():
    opt = _stub()
    assert insights.value_misses(opt, [_check(1, 3, insights.SCORE, True)]) is None


# ------------------------------------------------------------------ #
#  2 · Predicts and Partly carry their pairs
# ------------------------------------------------------------------ #


P, H, N = learned.PREDICTABLE, learned.PARTLY, learned.NOT_YET
OPT = NS(objectives=[], study_kind=None)


# ------------------------------------------------------------------ #
#  3 · Greyed gain, ties, room to move
# ------------------------------------------------------------------ #


def test_the_runner_up_line_is_short_and_says_no_noise():
    """Replaced the tie line that said what a rise needs (owner, v3 final
    cleanup: no tasting noise on screen)."""
    line = wording.runner_up_line(13, "93.3", "1.7")
    assert line == "Formulation 13: 93.3 (1.7 lower than the best so far)."
    assert _words(line) <= 10


