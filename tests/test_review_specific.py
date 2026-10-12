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
import ui_insights as U
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
    assert m.expected == pytest.approx(0.05 * 36)
    assert insights.MISS_RATE == 0.05


def test_value_misses_none_without_values():
    opt = _stub()
    assert insights.value_misses(opt, [_check(1, 3, insights.SCORE, True)]) is None


@pytest.mark.parametrize("expected,said", [(1.8, "about 2"), (0.5, "about 1"),
                                           (0.45, "under 1"), (2.5, "about 3"),
                                           (0.0, "under 1")])
def test_expected_count(expected, said):
    assert wording.expected_count(expected) == said


def test_the_miss_line_names_the_worst_and_the_count():
    items = [("Creaminess", 4, 9), ("Tangy", 2, 9), ("Thickness uniformity", 2, 9),
             ("Consistency", 1, 9)]
    line = wording.values_missed(items, 9, 36, 1.8)
    # Tangy and Thickness uniformity tie for second: neither is named alone.
    assert line == "Creaminess +3: 9 of 36 outside range (about 2 expected)."
    two = wording.values_missed(items[:1] + items[3:], 5, 36, 1.8)
    assert two == "Creaminess, Consistency: 5 of 36 outside range (about 2 expected)."
    # Two long names would take 11 words: the worst and "+1".
    long = wording.values_missed(items[:1] + items[2:3], 6, 36, 1.8)
    assert long == "Creaminess +1: 6 of 36 outside range (about 2 expected)."
    clear = [("Creaminess", 4, 9), ("Tangy", 3, 9), ("Consistency", 1, 9)]
    # Two names and "+1" would take 11 words: the worst alone.
    assert wording.values_missed(clear, 8, 36, 1.8).startswith("Creaminess +2: 8 of 36")
    assert wording.values_missed(clear[:2], 7, 36, 1.8) == (
        "Creaminess, Tangy: 7 of 36 outside range (about 2 expected).")
    for text in (line, two, long):
        assert _words(text) <= 10          # a card line's limit
        _clean(text)
    assert "more misses" not in line.lower()


def test_the_miss_line_detail_lists_every_measurement():
    items = [("Creaminess", 4, 9), ("Tangy", 2, 9)]
    text = wording.values_missed_detail([3, 4, 5], items, 6, 36, 1.8)
    assert text.startswith("Single measurements, Rounds 3–5: 6 of 36 outside their 95 % "
                           "range; a correct range misses 1 in 20, about 2.")
    assert "Creaminess 4 of 9, Tangy 2 of 9." in text
    _clean(text)


def test_cards_say_the_miss_line_from_the_facts():
    m = insights.ValueMisses([3, 4, 5], 9, 36, [("Creaminess", 4, 9), ("Tangy", 2, 9),
                                                  ("Thickness uniformity", 2, 9),
                                                  ("Consistency", 1, 9)])
    facts = NS(misses=m, headline=NS(inside=27, total=36))
    assert U.misses_line(facts) == "Creaminess +3: 9 of 36 outside range (about 2 expected)."
    assert U.misses_detail(facts).startswith("Single measurements, Rounds 3–5: 9 of 36")
    # An older cache without the breakdown: the headline's own count.
    old = NS(misses=None, headline=NS(inside=27, total=36))
    assert U.misses_line(old) == "Measurements: 9 of 36 outside (about 2 expected)."
    assert U.misses_detail(old) == ""


def test_make_a_round_track_line_counts_the_measurements():
    m = insights.ValueMisses([3, 4, 5], 9, 36, [("Creaminess", 4, 9)])
    t = insights.RoundTrack(False, 9, 9, None, True, None, [3, 4, 5], misses=m)
    short = U.track_short(t)
    # Re-pinned (was "Overall score: 9 of 9 in range. Measurements: more
    # misses than expected."): the measurements' count against 1 in 20.
    assert short == ("Overall score: 9 of 9 in range. Measurements: 9 of 36 outside "
                     "(about 2 expected).")
    # Two statements of 7 and 8 words (was 12 words with the vague one).
    assert _words(short) <= 15
    full = U.track_text(t)
    assert "measurements: 9 of 36 outside their 95 % range (about 2 expected)" in full
    assert "many more" not in full
    _clean(short)


def test_make_a_round_misses_line_above_generate():
    line = wording.next_misses_said(10, 36, 1.8, 16)
    assert line == ("Measurements: 10 of 36 outside (about 2 expected). Repeat Formulation 16 "
                    "to check noise.")
    assert _words(line) <= 14
    _clean(line)
    full = wording.next_misses_counted([4, 5, 6], "unexplained", 10, 36, 1.8, 16)
    assert full.startswith("Measured values in Rounds 4–6: 10 of 36 outside their 95 % "
                           "range, about 2 expected (1 in 20)")
    assert "A Repeat of Formulation 16 would measure the noise." in full


# ------------------------------------------------------------------ #
#  2 · Predicts and Partly carry their pairs
# ------------------------------------------------------------------ #

def _lf(**verdicts):
    orderings = {}
    for name, (right, pairs, verdict) in verdicts.items():
        name = insights.SCORE if name == "score" else name.replace("_", " ")
        orderings[name] = learned.Ordering(name, right=right, pairs=pairs, verdict=verdict,
                                           tested=len(verdicts))
    return U.LearnedFacts(orderings, {}, (None, []), None, (0, 0, 0), "")


P, H, N = learned.PREDICTABLE, learned.PARTLY, learned.NOT_YET
OPT = NS(objectives=[], study_kind=None)


def test_the_strip_says_the_pairs_behind_its_word():
    assert wording.learned_strip_counted([], ["Tangy"], ["Creaminess"],
                                         {"Tangy": (7, 8)}) == ("Partly:", "Tangy (7 of 8 pairs)")
    word, line = wording.learned_strip_counted(
        [], ["Overall score", "Tangy", "Consistency"], [],
        {"Overall score": (5, 5), "Tangy": (5, 6), "Consistency": (3, 4)})
    # Two names: the details chart's form, so the item stays ~7 words.
    assert (word, line) == ("Partly:", "Overall score (5/5), Tangy (5/6) +1")
    assert _words(f"{word} {line}") <= 8
    # Not yet names stay bare: they are the ones without evidence.
    assert wording.learned_strip_counted([], [], ["Tangy"], {"Tangy": (1, 3)}) == (
        "Not yet:", "Tangy")


def test_the_forecast_check_line_carries_the_pairs(monkeypatch):
    monkeypatch.setattr(U, "_say", lambda opt, text: text)
    monkeypatch.setattr(U, "_short", lambda opt, text: text)
    ten = _lf(score=(7, 9, H), Consistency=(11, 11, P), Creaminess=(5, 12, N),
              Tangy=(11, 14, H), Thickness_uniformity=(5, 10, N))
    six = _lf(score=(5, 5, H), Consistency=(3, 3, N), Creaminess=(2, 4, N), Tangy=(5, 6, H),
              Thickness_uniformity=(2, 3, N))
    few = _lf(score=(4, 4, N), Consistency=(3, 3, N), Creaminess=(1, 5, N),
              Tangy=(5, 5, H), Thickness_uniformity=(2, 5, N))
    none = _lf(score=(0, 0, N), Tangy=(5, 6, H))
    long = _lf(score=(5, 5, H), Firmness=(12, 16, H), Cook_loss=(5, 5, H),
               Juiciness=(6, 16, N))
    lines = {k: U.learned_verdict_line(OPT, lf)[0]
             for k, lf in (("ten", ten), ("six", six), ("few", few), ("none", none),
                                       ("long", long))}
    assert lines["ten"] == ("So far: Overall score 7 of 9 pairs ordered right. "
                            "Predicts: Consistency (11/11).")
    # The count and the Partly names would take 15 words: the names alone,
    # each with its pairs.
    assert lines["six"] == "Partly: Overall score (5 of 5 pairs), Tangy (5/6)."

    # Under the pairs a verdict needs: the count it needs (was "too few to judge").
    assert lines["few"] == "So far: Overall score 4 of 4 pairs (needs 5). Partly: Tangy (5/5)."
    # No Overall-score pairs: the unit with the first count.
    assert lines["none"] == "Partly: Tangy (5 of 6 pairs). Not yet: Overall score."
    # Count and three names would run past 14 words: the names, with pairs.
    assert lines["long"] == ("Partly: Overall score (5 of 5 pairs), Firmness (12/16), "
                             "Cook loss (5/5).")
    for text in lines.values():
        assert _words(text) <= wording.LEARNED_LINE_WORDS == 14
        assert "too few" not in text
        _clean(text)


def test_the_model_question_mark_opens_with_the_pairs():
    text = wording.model_plain_counted([], ["Tangy"], ["Creaminess"], {"Tangy": (5, 5)})
    assert text.startswith("Forecasting each Round from earlier Rounds, the model leans right "
                           "on the order of Tangy (5 of 5 pairs), not yet clearly.")
    both = wording.model_plain_counted(["Consistency"], ["Overall score", "Tangy"], [],
                                       {"Consistency": (11, 11), "Overall score": (7, 9),
                                        "Tangy": (11, 14)})
    assert "Consistency (11 of 11 pairs)" in both
    assert "Partly: Overall score (7/9), Tangy (11/14)." in both


def test_the_ordering_chart_labels_count():
    assert wording.learned_too_few(learned.MIN_PAIRS) == "under 5 pairs"
    assert wording.learned_full_chance(3) == "chance up to 100 %: 3 Rounds"
    assert wording.learned_full_chance(1) == "chance up to 100 %: 1 Round"


def test_no_result_predictable_says_the_closest(monkeypatch):
    monkeypatch.setattr(U, "_say", lambda opt, text: text)
    ords = {insights.SCORE: learned.Ordering(insights.SCORE, right=4, pairs=4, verdict=N),
            "Tangy": learned.Ordering("Tangy", right=5, pairs=5, verdict=H),
            "Creaminess": learned.Ordering("Creaminess", right=1, pairs=5, verdict=N)}
    monkeypatch.setattr(learned, "orderings", lambda opt, checks: ords)
    line = U.forecast_none_text(OPT, ["x"])
    assert line == "No result predictable yet; closest: Tangy, 5 of 5 pairs."
    assert wording.forecast_none() == "No result predictable yet: no pairs ordered yet."
    assert _words(line) <= 10
    _clean(line)


# ------------------------------------------------------------------ #
#  3 · Greyed gain, ties, room to move
# ------------------------------------------------------------------ #

def test_greyed_gain_says_the_fact_behind_the_word():
    lines = [wording.gain_greyed_checks(9, 10), wording.gain_greyed_missed(4, 12),
             wording.gain_greyed_skill("0.12", "0.20")]
    assert lines == ["Expected gain greyed: 9 Overall score checks; needs 10.",
                     "Expected gain greyed: Overall score 4 of 12 outside range.",
                     "Expected gain greyed: skill 0.12; needs 0.20."]
    for text in lines:
        assert _words(text) <= 10
        _clean(text)


def test_greyed_gain_reads_the_overall_scores_checks(monkeypatch):
    opt = _stub()
    opt.complete_count = lambda: 12
    checks = [c for c in _checks({}) if c.round > 1]       # 9 Overall-score checks
    assert U.gain_greyed_text(opt, checks, "not_yet") == (
        "Expected gain greyed: 9 Overall score checks; needs 10.")
    assert U.gain_greyed_text(opt, checks, "noise") == wording.GAIN_GREYED["noise"]


def test_the_tie_line_says_what_a_rise_needs():
    line = wording.tie_line_needed(13, "1.67", "2.3")
    assert line == "Formulation 13 is within noise: 1.67 lower; 2.3 needed."
    assert _words(line) <= 10
    assert wording.cramped_counted(["Tap water"], "5 %") == "Under 5 % room to move: Tap water."


def test_the_tie_line_folds_the_need_in_and_never_reads_equal():
    # tie_line is the one Results uses; the need is said beside the lead.
    assert wording.tie_line(4, "70.30", "0.50", 3, "2.3") == (
        "Formulation 3 is within noise: 0.50 lower; 2.3 needed.")
    _clean(wording.LEARNED_EVIDENCE_ABOVE)
    assert wording.LEARNED_EVIDENCE_ABOVE.startswith(
        "Evidence: pairs ordered right from earlier Rounds")
