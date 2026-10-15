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
from tests.test_090_wording import _STYLE
from tests.test_food_bo import _BANNED

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


def test_pair_sentences_from_the_verdicts(monkeypatch):
    monkeypatch.setattr(U, "learned_label", lambda opt, name: name)
    monkeypatch.setattr(U, "_say", lambda opt, text: text)
    lf = types.SimpleNamespace(orderings={
        "Overall score": _ordering("Overall score", 5, 5, ties=1, close=6),
        "Creaminess": _ordering("Creaminess", 0, 0),
        "Tangy": _ordering("Tangy", 0, 0, close=12),
    })
    assert U.pair_sentences(None, lf) == [
        "Overall score: of 12 pairs, 5 called, all 5 right; 6 too close, 1 measured equal.",
        "Tangy: of 12 pairs, none called; 12 too close."]
    assert U.pair_sentences(None, None) == []


# ------------------------------------------------------------------ #
#  What is gone
# ------------------------------------------------------------------ #

def test_the_review_has_three_sections_and_no_tried_or_drives():
    assert U.SECTION_ORDER == ("progress", "next", "predictions")
    assert set(U.SECTION_KEYS) == {"progress", "next", "predictions"}
    for gone in ("_tried_section", "_drives_section", "_profiler", "profile_chart",
                 "search_path_chart", "learned_chart", "learned_rows", "called_chart",
                 "check_bias_line", "explored_link", "_model_and_method",
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
    # Fix round 1: the Round line is the one visible line; the verdict line
    # only when there is no Round to check.
    assert "called_line(opt, lf)" in src
    assert "None if line is not None else learned_verdict_line(opt, lf)" in src


def test_the_round_line_and_its_question_mark(monkeypatch):
    """Finding 1: the Round line, as at 5fb2b95, and its "?" opening on the
    Round's landing in plain words."""
    monkeypatch.setattr(U, "_short", lambda opt, text: text)
    monkeypatch.setattr(U, "_say", lambda opt, text: text)
    monkeypatch.setattr(U, "no_forecast", lambda opt, c: False)
    import insights
    ck = lambda no, low, high, measured, predicted: insights.Check(  # noqa: E731
        no, 6, insights.SCORE, predicted, low, high, measured, low <= measured <= high, True)
    cks = [ck(16, 80.0, 86.0, 90.0, 83.0), ck(17, 82.0, 88.0, 91.0, 85.0),
           ck(18, 70.0, 80.0, 75.0, 75.0)]
    call = learned.round_call(cks, insights.SCORE, 6, 1.0)
    lf = types.SimpleNamespace(items=[(insights.SCORE, call, cks)], called=(6, []),
                               orderings={})
    line = U.called_line(None, lf)
    # Fix round 2: the landing fact alone, at most 10 words.
    assert line == "**Round 6: 2 of 3 above their forecast range.**", line
    assert _words(line) <= 10
    # Its "?": the landing in plain words, then the Round's own pairs.
    assert U.called_help(None, lf) == (
        "Round 6, Overall score: 1 of 3 in range; 2 misses, both higher than forecast.\n\n"
        "Round 6 order, Overall score: of 3 pairs, 3 called, all 3 right.")
    assert U.called_line(None, types.SimpleNamespace(items=[], called=(None, []))) is None


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
    assert terms == ["Overall score", "95 % range", "Tasting noise", "Stop",
                     "Called · too close", "No forecast", "Predicts · Partly · Not yet"]
    # Glossary lines may run about 25 words; the verdict rule, in full, longer.
    limits = {"Called · too close": 30, "Predicts · Partly · Not yet": 40}
    for term, line in zip(terms, lines):
        assert _words(line) <= limits.get(term, 27), line
    # A link (the README is not in the app bundle), its caption short.
    assert "](https://github.com/thomasat/food_opt#how-the-review-works)" in \
        wording.METHOD_README
    assert _words(wording.METHOD_README) <= 8
    # The verdict words the Forecast check shows are defined here.
    for word in wording.LEARNED_WORDS.values():
        assert word in lines[-1]


def test_each_definition_is_true_to_the_rule_the_app_applies():
    """Findings 3, 5, 8: the verdicts as learned.judge / holm decide them,
    the 95 % range's coverage as the app's ruling says it (never '95 times
    in 100'), the called threshold as the result's own noise."""
    by = {re.match(r"\*\*(.+?)\*\*: ", line).group(1): line
          for line in wording.METHOD_DEFINITIONS}
    verdicts = by["Predicts · Partly · Not yet"]
    assert f"{learned.MIN_PAIRS}+ pairs called" in verdicts
    assert f"p < {learned.VERDICT_P:.2f}, adjusted" in verdicts
    assert f"p < {learned.PARTLY_P:.2f}" in verdicts and "over half right" in verdicts
    assert learned.PARTLY_MIN == learned.MIN_PAIRS
    assert learned.FLAT_TIES == 0.5 and "over half the pairs were measured equal" in verdicts
    rng = by["95 % range"]
    assert "95 times in 100" not in rng
    assert f"{wording.SIMULATED_COVERAGE['measurements']} in 100" in rng
    assert f"({wording.SIMULATED_COVERAGE['score']} for Overall scores)" in rng
    called = by["Called · too close"]
    assert "that result's noise" in called and "Overall score: points" in called
    assert "its own units" in called and "measured-equal pairs never are" in called


def test_the_readme_holds_the_full_method_in_25_lines():
    text = open(README, encoding="utf-8").read()
    assert "## How the Review works" in text
    section = text.split("## How the Review works", 1)[1].split("\n## ", 1)[0]
    lines = [line for line in section.splitlines() if line.strip()]
    assert 3 <= len(lines) <= 25
    for term in ("Gaussian process", "calibration factor", "Holm", "No forecast",
                 "Tasting noise", "Stop", "second factor", "1-in-20 threshold",
                 "No term p-values or coefficient table",
                 "until then the project's Tasting noise setting (Set up › More settings; "
                 "1 point unless changed), assumed"):
        assert term in section, term
    assert "a fixed rule, not a significance test" not in section
    assert "barely varies" not in section


def test_the_new_strings_pass_the_guards():
    texts = [*wording.METHOD_DEFINITIONS, wording.METHOD_README, wording.METHOD_LINK_HELP,
             wording.METHOD_LINK, wording.LEARNED_DETAILS_TITLE,
             wording.pairs_sentence("Tangy", 5, 9, 2, 1),
             wording.pairs_sentence("Tangy", 0, 0, 12), wording.pairs_sentence("Tangy", 1, 1),
             wording.round_landing(6, "Overall score", 1, 2, 0),
             wording.so_far_pairs("Overall score: 5 of 12 pairs called; 5 right."),
             wording.check_line(6, 1, 2, 0, "order too close to call")]
    for text in texts:
        assert not any(p.search(text) for p in _BANNED), text
        assert not any(p.search(text) for p in _STYLE), text
        assert text.count("—") <= 1, text
    assert _words(wording.LEARNED_DETAILS_TITLE) <= 5
    assert _words(wording.METHOD_LINK_HELP) <= 14
    # Labels at most 5 words; the link promises only what the fold holds.
    assert _words(wording.METHOD_LINK) <= 5 and "How the app knows" != wording.METHOD_LINK
    assert f"{len(wording.METHOD_DEFINITIONS)} terms" in wording.METHOD_LINK_HELP


def test_stale_pointers_are_gone():
    """Finding 10: no text points at what Method and definitions no longer
    holds; no shadowed constant is left."""
    import inspect
    assert "Method and definitions" not in wording.BASIS_NEXT
    assert "Method and definitions" not in wording.TIP_LUCK
    src = inspect.getsource(wording)
    assert src.count("\nLEARNED_DETAILS_TITLE = ") == 1
    assert src.count("\nMETHOD_LINK_HELP = ") == 1 and src.count("\nEV_METHOD = ") == 1
    doc = U.forecast_chart.__doc__
    assert "hollow" not in doc and "orange outside" in doc and "No" in doc
