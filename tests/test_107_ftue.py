"""Insights for a first-time user (spec 2026-10-01 rev. 2; review I17,
R-I17): the pinned list of statistic names no level-0 sentence may say, and
the first screen's level-0 texts, built from wording's builders as the
review's R-B1 table gives them for the six demo projects and the states
they do not reach.

T0 writes the list and its check on the texts; W1 and the integration pins
(the 1280 × 800 browser test, the word budget) import STATISTIC_NAMES and
statistic_names_in from here, so there is one list. It is review 02 §2's
28 rows, less the plain words the first screen keeps (Repeat, Best so far,
Round, results, average, range in "the range the app predicted")."""
import re

import pytest

import wording as W

STATISTIC_NAMES = tuple(re.compile(p) for p in (
    r"95\s*%", r"\bintervals?\b", r"\bskill\b", r"noise threshold", r"\bS[Dd]\b", r"\bsd\b",
    r"\bdf\b", r"\bn = ", r"(?i)mean error", r"(?i)\bbias\b", r"(?i)round-ahead", r"(?i)\bmedian\b",
    r"(?i)half-width", r"(?i)standardi[sz]ed", r"(?i)calibrat", r"(?i)posterior",
    r"(?i)predictive", r"(?i)expected improvement", r"(?i)\bverdict\b", r"(?i)active bounds?",
    r"(?i)local effect", r"(?i)one-factor", r"(?i)correlation length", r"(?i)lengthscale",
    r"(?i)gaussian process", r"(?i)\bkernel\b", r"(?i)cross-validat", r"qLog", r"R²",
    r"(?i)p-value", r"(?i)regression to the mean", r"(?i)acquisition", r"As expected",
    r"(?i)misses than expected", r"Not judged yet", r"(?i)values checked",
    r"(?i)predicted best", r"(?i)\buntested\b", r"(?i)binomial", r"(?i)\bpairs?\b",
    r"(?i)model (mean|sd)", r"(?i)confidence", r"(?i)significan",
))


def statistic_names_in(text):
    """The statistic names `text` says, as their patterns."""
    return [p.pattern for p in STATISTIC_NAMES if p.search(str(text))]


# The R-B1 table's level-0 lines, project by project: (Predictions word,
# its sentence, Progress word, its sentence, Next sentence). Line C's Next
# says "the suggestions are a guided search" where the table says "treat
# the suggestions as": the style guide retires "treat … as" (t0-report §
# "Where the texts differ from the review").
# Ftue fix A (statistician's cold read I5, I6, I7, I9) moved Lines A-C:
# Line A 32 -> 33 of 35 inside (Round-grouped CV folds); Line B 10 -> 8 of
# 36 outside, no bias clause (t = 2.45 < t_{0.975, 3} = 3.18), the Repeat
# limit 7.4 -> 2.8 (√2 × 1.96 × the assumed 1 point); Line C no bias clause
# (t = 2.69) and Within noise, not Rising (2.5 < 1 × √2 × z_4 = 3.16). The
# Within-noise sentence still names the 1-point noise SD, not the 3.16 the
# rise is judged against: its wording is B's to update (Improvement.threshold).
FIRST_SCREENS = {
    "Snack bar": (
        "not_yet", W.pred_gate(3),
        "too_early", W.progress_too_early("96.00", 100, 1, one_round=True, tie_no=3),
        W.next_plain_tail()),
    "Line A": (
        "reliable", W.pred_reliable([3, 4, 5], 33, 35, 3),
        "no_new_best", W.progress_no_new_best(5, "90.45", 100, 15, 4),
        W.next_plain_see(6, 4, repeat_no=15)),
    "Line B": (
        "off", W.pred_off_coverage([3, 4, 5], 8, 36, 4),
        "flat", W.progress_flat(2, "90.09", 100, 11, 3),
        W.next_plain_repeat(21, 11, "10.5", half_text="2.8")),
    "Line C": (
        "not_yet", W.pred_no_skill(None,
                                   [W.clause_wide(["Firmness", "Juiciness", "Cook loss"])]),
        "within_noise", W.progress_within_noise(5, "2.5", "89.84", 100, 18),
        W.next_plain_no_skill(6, 4)),
    "Upgrade check": (
        "not_yet", W.pred_too_few(2),
        "too_early", W.progress_too_early("90.50", 100, 6, 3),
        W.next_plain_tail()),
    "Timing check": (
        "reliable", W.pred_reliable([10, 11, 12], 58, 60, 5),
        "flat", W.progress_flat(7, "92.50", 100, 24, 5, luck_text="90"),
        W.next_plain_explore("1.1", repeat_no=24)),
}

# The states the demo projects do not reach (R-B1), one sentence each.
UNREACHED = {
    "caution": W.pred_caution([3, 4, 5], 5, 35, 3),
    "over_wide": W.pred_reliable([3, 4, 5], 36, 36, 4, [W.CLAUSE_OVER_WIDE]),
    "wide": W.PRED_WIDE,
    "unclear": W.pred_unclear(),
    "off_repeat": W.PRED_OFF_REPEAT,
    "none": W.pred_too_few(0),
    "reliable_all_clauses": W.pred_reliable(
        [4, 5, 6], 33, 36, 4, [W.clause_wide(["Cook loss"]), W.clause_bias("1.2", False),
                               W.clause_guard()]),
    "within_noise": W.progress_within_noise(5, "0.4", "89.84", 100, 18),
    "within_noise_measured": W.progress_within_noise(5, "0.4", "89.84", 100, 18, "1.2"),
    "flat_tied": W.progress_flat(3, "90.09", 100, 11, 3, luck_text="88", tie_no=12),
    "no_new_best_tied": W.progress_no_new_best(5, "90.45", 100, 15, 4, tie_no=18),
    "misses": W.next_plain_misses(6, 4),
    "untested": W.next_plain_untested(6),
    "untested_closed": W.next_plain_untested(open_round=False),
    "close": W.next_plain_close(W.raise_highest("Textured pea protein")),
    "close_nothing_binds": W.next_plain_close(repeat_no=15),
    "rising": W.next_plain_rising(6, 4),
    "rising_repeat": W.next_plain_rising(repeat_no=18),
    "explore_open": W.next_plain_explore("1.1", assumed=False, no=6, left=4, repeat_no=24),
}


def _level0():
    for name, (_, pred, _, prog, nxt) in FIRST_SCREENS.items():
        yield name, pred
        yield name, prog
        yield name, nxt
    yield from UNREACHED.items()


@pytest.mark.parametrize("name,text", list(_level0()))
def test_no_level0_sentence_names_a_statistic(name, text):
    """§1: "0 statistic names in sentences" (numbers allowed, each with its
    plain meaning)."""
    assert statistic_names_in(text) == [], (name, text)


@pytest.mark.parametrize("name", sorted(FIRST_SCREENS))
def test_each_status_sentence_fits_two_body_lines(name):
    """R-I8: each status sentence at most 2 body lines, about 220
    characters at the 980 px main block."""
    _, pred, _, prog, nxt = FIRST_SCREENS[name]
    for text in (pred, prog):
        assert len(text) <= 220, (name, text)
    # Next carries the one action; the review's own Line B and Timing check
    # sentences run to three lines, which its fold estimate allows.
    assert len(nxt) <= 330, (name, nxt)


def test_the_list_catches_what_it_is_for():
    """The list is not empty in practice: today's verbatim lines name
    statistics, which is why they move into the figures folds."""
    assert statistic_names_in(W.trust_within(32, 35, [3, 4, 5], allowed=3))
    assert statistic_names_in(W.skill_line("0.36", 15, [2, 3, 4, 5], "skill"))
    assert statistic_names_in(W.next_explore())
    assert statistic_names_in(W.stand_repeats("0.8", 1, 1, 1))
