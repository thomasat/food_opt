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
    return [*_section(), wording.REBUILT_LABEL, wording.RESULTS_PREDICTED_CAPTION,
            wording.RESULTS_PREDICTED_SPINNER, wording.LEARNED_FORECAST_TITLE,
            wording.CALLED_RULE, wording.forecast_help(15, 6),
            wording.results_predicted_help(6), wording.STORY_TRUST_CAPTION,
            wording.LEARNED_COL_PAIRS, wording.LEARNED_RULE,
            wording.learned_evidence([("Overall score", 10, 11)]),
            wording.learned_counts([("Overall score", 10, 11, 5.5)]),
            wording.model_plain(["Overall score"], ["Tangy"], ["Creaminess"]),
            wording.model_plain([], ["Tangy"], ["Creaminess"])]


def test_rebuilt_strings_pass_banned():
    hits = [(t, p.pattern) for t in _texts() for p in _BANNED if p.search(t)]
    assert hits == []


def test_rebuilt_strings_never_claim_the_time():
    for t in _texts():
        assert not re.search(r"\b(locked|replay|recorded at the time|at the time)\b", t, re.I), t


def test_rebuilt_wording_pinned():
    assert wording.REBUILT_LABEL == "Rebuilt from earlier Rounds"
    assert wording.RESULTS_PREDICTED_CAPTION == "Rebuilt from earlier Rounds: what the model expected."
    assert wording.LEARNED_FORECAST_TITLE == "Expected from earlier Rounds"
    assert wording.CALLED_RULE.startswith("Rebuilt today from the formulations scored before "
                                          "this Round only: what the model expected")
    assert "rebuilt today" in wording.results_predicted_help(6)
    assert "What the model expects" in wording.forecast_help(15, 6)


def test_rebuilt_wording_limits():
    assert _words(wording.RESULTS_PREDICTED_CAPTION) <= 8
    for label in (wording.REBUILT_LABEL, wording.LEARNED_FORECAST_TITLE,
                  wording.RESULTS_PREDICTED_SPINNER):
        assert _words(label) <= 5, label


def test_round_ahead_words_say_from_earlier_rounds_not_before_tasting():
    """A Round's forecast is rebuilt from the Rounds before it, so in a
    project replayed from records "before tasting" read as a claim about
    when it was made. The words say what is true live and replayed: the
    forecast had none of that Round's results."""
    for t in _texts():
        assert "before tasting" not in t, t
    assert wording.STORY_TRUST_CAPTION == "Blue: predicted from earlier Rounds. Dot: measured."
    assert wording.LEARNED_COL_PAIRS == "Ordered right from earlier Rounds"
    assert wording.LEARNED_RULE.startswith(
        "Each Round's formulations are forecast from the formulations scored before "
        "that Round only. ")
    assert wording.learned_evidence([("Overall score", 10, 11)]) == (
        "Evidence: pairs ordered right from earlier Rounds — Overall score 10/11.")
    assert wording.model_plain(["Overall score", "Consistency"], [], ["Tangy"]) == (
        "Forecasting each Round from earlier Rounds, the model put the Overall score and "
        "Consistency in the right order, clearly more often than chance. Not yet: Tangy.")
    assert wording.model_plain([], ["Tangy"], []) == (
        "Forecasting each Round from earlier Rounds, the model leans right on the order of "
        "Tangy, not yet clearly.")


def test_round_ahead_words_within_limits():
    assert _words(wording.STORY_TRUST_CAPTION) <= 8
    assert _words(wording.LEARNED_COL_PAIRS) <= 5
