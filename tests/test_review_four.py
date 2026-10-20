"""Review four (prototype, 2026-10-02; owner: "clean up the Review tab to
cleanly (and visually) answer these 4 questions"). Under the header and
the three cards, four question sections, always open, in this order: Q1
"Can we trust the predictions?" beside Q2 "Are we improving?", then Q3
"What drives each result?" and Q4 "What have we tried?". Each is a small
heading, one answer line opening with its card's bold word, ONE chart and
its caption. Each card links to its question ("See chart ↓"). The hero,
"Can you have both?", the triangle and their folds are gone.

Ruling 1: the Predictions word reads the Overall score's Round-ahead record
(coverage + skill); the measurements' misses are a secondary, non-red line.
Ruling 2: Expected gain is never hidden: greyed, with its reason."""
import pytest

import ui_insights as U
from tests import demo_projects as D
from tests.review_screens import _leaves, _text, screens  # noqa: F401 (fixture)

CHART = ("arrow_vega_lite_chart", "vega_lite_chart")


# ------------------------------------------------------------------ #


@pytest.mark.parametrize("name", D.ALL)
def test_one_primary_on_the_screen(screens, name):  # noqa: F811
    at = screens[name]
    primaries = [el for el, _, _ in _leaves(at.tabs[3])
                 if el.type == "button" and el.proto.type == "primary"]
    assert len(primaries) == 1


REMOVED_TEXT = ("How the app chose", "Where the app looks next",
                'Which formulations are worth trying?', "What drives each result?", "What have we tried?",
                "Of pairs called, % right")


@pytest.mark.parametrize("name", D.ALL)
def test_the_removed_charts_are_gone(screens, name):  # noqa: F811
    at = screens[name]
    for el, _, _ in _leaves(at.tabs[3]):
        text = _text(el) or ""
        spec = el.proto.spec if el.type in CHART else ""
        for gone in REMOVED_TEXT:
            assert gone not in text and gone not in spec, (gone, text[:80])
    for fn in ("chooses_chart", "chooses_data", "pair_chart", "mixture_chart", "_hero",
               "_triangle", "_more_charts", "explored_fold", "tried_fold", "path_chart",
               "change_chart", "gain_chart", "search_path_chart", "profile_chart", "_profiler",
               "learned_chart", "called_chart", "_tried_section", "_drives_section"):
        assert not hasattr(U, fn), fn
    # v3: the map went with Make a round's Why fold.
    assert not hasattr(U, "explore_map_chart")
