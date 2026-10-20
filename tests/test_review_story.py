"""Review story (2026-10-02), the parts the v3 Review still has: each
Round's average is the mean of that Round's formulations, and the story
label passes the vocabulary guard. (The Insights-tab story screens went
with the old engine.)"""
import re

import pytest

import ui_insights as U
import wording
from tests import demo_projects as D
from tests.test_food_bo import _BANNED


def _words(text):
    plain = re.sub(r":\w+\[(.*?)\]", r"\1", text).replace("**", "")
    return len([w for w in plain.split() if re.search(r"\w", w)])


def test_round_averages_are_the_rounds_means():
    opt = D.load(D.LINE_A)
    pool = set(opt.best_candidates())
    for r, a, b, mean, n in U.round_averages(opt):
        mine = [float(opt.Y_history[i]) for i, x in enumerate(opt.batch_history)
                if x == r and i in pool]
        assert n == len(mine) and mean == pytest.approx(sum(mine) / len(mine))


def test_story_strings_pass_the_vocabulary_guard():
    for text in [wording.STORY_ROUND_AVERAGE]:
        assert not any(p.search(text) for p in _BANNED), text
        assert "spread" not in text.lower(), text
        # No claim of Rounds or time saved, nor any comparison with another method.
        assert not re.search(r"\bsaved?\b|faster|fewer rounds|than (DoE|random|manual)",
                             text, re.I), text
    # Labels at most 5 words.
    assert _words(wording.STORY_ROUND_AVERAGE) <= 5
