"""Review round 2 (2026-10-05) wording guards that still apply to the app:
label word budgets and the banned-word guard over the round-2 strings.
(The Holm, widening, forecast and stop tests went with the old Insights
engine.)"""
import ast
import pathlib
import re

import wording
from tests.test_food_bo import _BANNED


def test_captions_and_labels_keep_to_their_word_budgets():
    def words(t):
        return len(re.sub(r"\*\*", "", t).split())
    for text, limit in ((wording.back_to_batch_label(6, 3), 5), (wording.OFF_BY_TITLE, 5)):
        assert words(text) <= limit, text



def _round2_source():
    src = pathlib.Path(wording.__file__).read_text()
    return src[src.index("# --- round 2 fixes ---"):]


def test_no_round_two_string_says_a_banned_word():
    tree = ast.parse(_round2_source())
    # Docstrings say what a function returns, not what reaches the screen;
    # _GLOSSARY_GONE names rows taken off the screen.
    skip = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.body and isinstance(n.body[0], ast.Expr):
            skip.add(id(n.body[0].value))
        if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "_GLOSSARY_GONE"
                                             for t in n.targets):
            skip |= {id(c) for c in ast.walk(n.value)}
    shown = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
             and isinstance(n.value, str) and id(n) not in skip]
    bad = [t for t in shown if any(p.search(t) for p in _BANNED)]
    assert not bad, bad
