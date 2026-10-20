"""The pinned list of statistic names no first-screen sentence may say
(spec 2026-10-01 rev. 2; review I17, R-I17), and its check. Formerly
tests/test_107_ftue.py, whose sentence tables went with the v3 Review;
the later pins import STATISTIC_NAMES and statistic_names_in from here."""
import re

STATISTIC_NAMES = tuple(re.compile(p) for p in (
    r"95\s*%", r"\bintervals?\b",
    # Fix round B (five-second re-test, Line C Q5): the skill may stand with
    # its number and its meaning, never alone.
    r"\bskill\b(?! [−-]?\d+\.\d+, where 0 matches that average and 1 is exact\))",
    r"noise threshold", r"\bS[Dd]\b", r"\bsd\b",
    r"\bdf\b", r"\bn = ", r"(?i)mean error", r"(?i)\bbias\b", r"(?i)round-ahead", r"(?i)\bmedian\b",
    r"(?i)half-width", r"(?i)standardi[sz]ed", r"(?i)calibrat", r"(?i)posterior",
    r"(?i)predictive", r"(?i)expected gain", r"(?i)\bverdict\b", r"(?i)active bounds?",
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


# Fix round B: the sentences the ledger ruled in run past two lines —
# Line C's Predictions (its skill with its meaning, the cause and the
# check: about four lines) and Line A's Progress (the tie and the "little
# more" clause: three). The button stays above the fold
# (tests/browser/test_107_first_screen.py).
LONGER = {("Line C", "pred"): 460, ("Line A", "prog"): 280}


