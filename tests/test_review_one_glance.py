"""One-glance Review (owner, 2026-10-14; spec 2026-10-13 §10): the Review
below the Round panel keeps What next?, the Forecast check (its one verdict
line, then a Details fold: one plain sentence per result with pairs and the
one forecast against measured chart) and Method and definitions (about six
one-line definitions and a pointer to the README). What have we tried? and
What drives each result? are gone, with everything that only they used."""
import os
import re
import types

import pytest

import learned
import ui_insights as U
import wording
from tests.test_090_wording import _STYLE
from tests.test_food_bo import _BANNED

README = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "README.md")


def _words(text):
    plain = re.sub(r":\w+\[(.*?)\]", r"\1", text).replace("**", "")
    return len([w for w in plain.split() if re.search(r"\w", w)])


# ------------------------------------------------------------------ #
#  The Forecast check's details: one plain sentence per result
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("right, called, total, want", [
    (5, 5, 12, "Overall score: of 12 pairs, 5 were far enough apart to call; all 5 called "
               "right."),
    (5, 9, 12, "Overall score: of 12 pairs, 9 were far enough apart to call; 5 of 9 called "
               "right."),
    (0, 4, 6, "Overall score: of 6 pairs, 4 were far enough apart to call; none of the 4 "
              "called right."),
    (0, 0, 3, "Overall score: of 3 pairs, none could be called."),
    (1, 1, 3, "Overall score: of 3 pairs, 1 was far enough apart to call; it was called "
              "right."),
    (0, 1, 1, "Overall score: of 1 pair, 1 was far enough apart to call; it was called "
              "wrong."),
])
def test_one_sentence_per_result(right, called, total, want):
    assert wording.pairs_sentence("Overall score", right, called, total) == want


def _ordering(name, right, pairs, ties=0, close=0):
    return learned.Ordering(name, right=right, pairs=pairs, ties=ties, close=close)


def test_pair_sentences_from_the_verdicts(monkeypatch):
    monkeypatch.setattr(U, "learned_label", lambda opt, name: name)
    monkeypatch.setattr(U, "_say", lambda opt, text: text)
    lf = types.SimpleNamespace(orderings={
        "Overall score": _ordering("Overall score", 5, 5, ties=1, close=6),
        "Creaminess": _ordering("Creaminess", 0, 0),
        "Tangy": _ordering("Tangy", 0, 0, close=12),
    })
    assert U.pair_sentences(None, lf) == [
        "Overall score: of 12 pairs, 5 were far enough apart to call; all 5 called right.",
        "Tangy: of 12 pairs, none could be called."]
    assert U.pair_sentences(None, None) == []


# ------------------------------------------------------------------ #
#  What is gone
# ------------------------------------------------------------------ #

def test_the_review_has_three_sections_and_no_tried_or_drives():
    assert U.SECTION_ORDER == ("progress", "next", "predictions")
    assert set(U.SECTION_KEYS) == {"progress", "next", "predictions"}
    for gone in ("_tried_section", "_drives_section", "_profiler", "profile_chart",
                 "search_path_chart", "learned_chart", "learned_rows", "called_chart",
                 "called_line", "check_bias_line", "explored_link", "_model_and_method",
                 "_status_table", "_glossary", "how_paragraphs", "details_frame"):
        assert not hasattr(U, gone), gone
    for gone in ("Q3_TITLE", "Q4_TITLE", "LEARNED_CHART_X", "LEARNED_KEY_CHANCE",
                 "PROFILER_CAPTION", "STORY_PATH_CAPTION", "MODEL_LIMITS", "HOW_VERDICT"):
        assert not hasattr(wording, gone), gone


def test_the_forecast_check_draws_one_chart_in_its_details():
    import inspect
    src = inspect.getsource(U._learned_section)
    assert "forecast_chart(" in src and src.count("show_chart(") == 1
    assert "LEARNED_DETAILS_TITLE" in src
    # The one verdict line, as it was.
    assert "learned_verdict_line(opt, lf)" in src


def test_no_forecast_points_stay_neutral_and_the_colours_do_not_merge():
    """The band's blue and the measured dots' ink and orange are two scales:
    a shared colour scale drew every dot in the band's blue."""
    import inspect
    src = inspect.getsource(U.forecast_chart)
    assert 'resolve_scale(color="independent")' in src
    assert "none_name" in src and 'mark("neutral")' in src


# ------------------------------------------------------------------ #
#  Method and definitions
# ------------------------------------------------------------------ #

def test_method_is_a_few_one_line_definitions_and_the_readme():
    lines = wording.METHOD_DEFINITIONS
    assert 6 <= len(lines) <= 7
    terms = [re.match(r"\*\*(.+?)\*\*: ", line).group(1) for line in lines]
    assert terms == ["Overall score", "95 % range", "Tasting noise", "Stop",
                     "Called · too close", "No forecast", "Predicts · Partly · Not yet"]
    for line in lines:
        assert _words(line) <= 20, line
    assert "README" in wording.METHOD_README and _words(wording.METHOD_README) <= 8
    # The verdict words the Forecast check shows are defined here.
    for word in wording.LEARNED_WORDS.values():
        assert word in lines[-1]


def test_the_readme_holds_the_full_method_in_25_lines():
    text = open(README, encoding="utf-8").read()
    assert "## How the Review works" in text
    section = text.split("## How the Review works", 1)[1].split("\n## ", 1)[0]
    lines = [line for line in section.splitlines() if line.strip()]
    assert 3 <= len(lines) <= 25
    for term in ("Gaussian process", "calibration factor", "Holm", "No forecast",
                 "Tasting noise", "Stop"):
        assert term in section, term


def test_the_new_strings_pass_the_guards():
    texts = [*wording.METHOD_DEFINITIONS, wording.METHOD_README, wording.METHOD_LINK_HELP,
             wording.LEARNED_DETAILS_TITLE, wording.pairs_sentence("Tangy", 5, 9, 12),
             wording.pairs_sentence("Tangy", 0, 0, 12), wording.pairs_sentence("Tangy", 1, 1, 1)]
    for text in texts:
        assert not any(p.search(text) for p in _BANNED), text
        assert not any(p.search(text) for p in _STYLE), text
        assert text.count("—") <= 1, text
    assert _words(wording.LEARNED_DETAILS_TITLE) <= 5
    assert _words(wording.METHOD_LINK_HELP) <= 14
