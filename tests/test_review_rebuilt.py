"""Rebuilt framing (2026-10-03): every forecast shown beside a Round is
worked out again on screen from the formulations scored before it, with
today's model. The words say so — rebuilt, what the model expected — and
never claim a forecast kept from the time the Round was made."""
import re

import wording
from tests.test_food_bo import _BANNED


def _words(text):
    return len([w for w in text.replace("**", "").split() if re.search(r"\w", w)])


def _section():
    src = open(wording.__file__).read().split("# --- rebuilt framing ---", 1)[1]
    return [t for t in re.findall(r'"((?:[^"\\\n]|\\.)*)"', src) if t.strip()]


def _texts():
    return [*_section(), 
            wording.LEARNED_COL_PAIRS,
            wording.learned_counts([("Overall score", 10, 11, 5.5)]),
            wording.model_plain(["Overall score"], ["Tangy"], ["Creaminess"]),
            wording.model_plain([], ["Tangy"], ["Creaminess"])]


def test_rebuilt_strings_pass_banned():
    hits = [(t, p.pattern) for t in _texts() for p in _BANNED if p.search(t)]
    assert hits == []


def test_rebuilt_strings_never_claim_the_time():
    for t in _texts():
        assert not re.search(r"\b(locked|replay|recorded at the time|at the time)\b", t, re.I), t


