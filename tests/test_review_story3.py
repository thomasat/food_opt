"""Review story 3 (prototype, 2026-10-02; owner: demo the yogurt pilot only,
and fade the profiler's untested stretches so a model curve there does not
read as a finding).

- The profiler draws each line solid and its band as before over tested
  amounts, and thin, dashed and faint, the band lighter, over the "not
  tested near here" stretches (the same rule as the grey).
- Q3's finding and the column order use each row's effect within its
  tested stretch; with no clear leader there, it says so.
- Two rows whose tested amounts moved together (|r| >= 0.7) are noted
  under the chart, the exact r in the note's "?".
- Method and definitions defines Expected gain once.

Synthetic: the Plant patty Line A demo project (CI); no pilot data."""
import re

import pytest

import bo_views
import insights
import insights_jobs
import ui_insights as U
import wording
from tests import demo_projects as D
from tests.test_food_bo import _BANNED


def _words(text):
    plain = re.sub(r":\w+\[(.*?)\]", r"\1", text).replace("**", "")
    return len([w for w in plain.split() if re.search(r"\w", w)])


# ------------------------------------------------------------------ #
#  Tested and untested stretches drawn differently
# ------------------------------------------------------------------ #

    # One piece when all tested.


def _blocks(spec):
    """Review readable: the profiler wraps at U.PROFILE_WRAP panels a row;
    more columns make a second block (a vconcat of its own rows). With one
    block, the rows are the top-level vconcat as before."""
    top = spec["vconcat"]
    return [b["vconcat"] for b in top] if "vconcat" in top[0] else [top]


def _rows(spec):
    """Every row facet of every block, with its data."""
    return [(r, spec["datasets"][r["data"]["name"]]) for b in _blocks(spec) for r in b]


def _filtered(layer):
    t = layer.get("transform") or []
    return next((x["filter"]["equal"] for x in t if "filter" in x), None)


    # The legend says what the dashes mean, in the grey's own entry.


# ------------------------------------------------------------------ #
#  The finding: tested stretches only
# ------------------------------------------------------------------ #

class _Col:
    def __init__(self, name, near_effect, effect=None, end=None):
        self.name, self.near_effect, self.end = name, near_effect, end
        self.effect = near_effect if effect is None else effect


class _Prof:
    def __init__(self, cols, ref="Best so far"):
        self.columns, self.ref_label = cols, ref


# ------------------------------------------------------------------ #
#  Amounts that moved together
# ------------------------------------------------------------------ #

def _together(monkeypatch, opt, pairs):
    """Rewrite the tested amounts: each (follower, leader, slope, wobble)
    makes `follower` move with `leader`."""
    real = insights.start_recipe

    def fake(o, i):
        r = dict(real(o, i))
        for follow, lead, slope, wobble in pairs:
            r[follow] = 2.0 + slope * float(r[lead]) + wobble * ((i * 7) % 5 - 2)
        return r
    monkeypatch.setattr(insights, "start_recipe", fake)


# ------------------------------------------------------------------ #
#  Method and definitions; wording
# ------------------------------------------------------------------ #

def _new_strings():
    return [wording.story_leader_rule("1"), wording.METHOD_GAIN]


def test_new_strings_pass_the_vocabulary_guard():
    for text in _new_strings():
        assert not any(p.search(text) for p in _BANNED), text
        assert "spread" not in text.lower(), text
    src = open(wording.__file__).read()
    assert "# --- review story 3 ---" in src
