"""Insights for a first-time user, T0 (spec 2026-10-01 rev. 2; every
"Proposed ruling" of research/2026-10-01-insights-ftue/spec-review.md):
every line W1, W2 and W3 need is spelled once in wording.py, in
03-wording-audit.md's style. It passes _BANNED (the glossary's few BO terms
by exact allowance only), says none of the style guide's retired words, has
at most one dash, has a Trial form, and the review's own texts come out
word for word where the style guide allows."""
import re
import types

import pytest

import wording as W
from tests.test_080_wording import _FROZEN
from tests.test_090_wording import _STYLE
from tests.test_107_ftue import FIRST_SCREENS, UNREACHED
from tests.test_food_bo import _ALLOWED_EXACT, _BANNED, _FTUE_ALLOWED


class _SettingsOnly:
    """A study of process settings alone, as for_project asks it."""

    def has_ingredients(self):
        return False

    def _process_settings(self):
        return ["Bake temperature"]


def _facts(**kw):
    """The facts W2 reads off Line A's built model (review I12), in the
    shape of insights.ModelFacts (test_107_t0_interfaces ties the two)."""
    base = dict(cold=False, scored=20, cold_start=5, values=False, total=True,
                recommended=True, inputs=6, kernel="rbf", ard=True,
                lengthscale_prior="LogNormal(2.31, 1.73)", lengthscale_floor=0.025,
                outputscale=False, outputscale_prior=None, noise="fitted",
                noise_prior="LogNormal(−4.00, 1.00)", noise_floor=1e-4, standardised=True,
                acquisition="qlognei", beta=None, q=4, restarts=20, raw_samples=1024,
                mc_samples=512, folds=5, fit_tolerance=1e-6, robust=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


def _status_lines():
    out = [W.STATUS_PREDICTIONS, W.STATUS_PROGRESS, W.STATUS_NEXT,
           *W.PREDICTIONS_WORDS.values(), *W.PROGRESS_WORDS.values(),
           W.status_line(W.STATUS_PREDICTIONS, W.pred_gate(3), "Not yet"),
           W.status_line(W.STATUS_NEXT, W.next_plain_tail()),
           W.CHECKING_PREDICTIONS, W.CHECKING_PROGRESS, W.PREPARING_NEXT_STEP,
           W.pred_gate(0), W.pred_gate(1), W.pred_too_few(1), W.pred_too_few(0),
           W.pred_off_coverage([4], 3, 12, 1), W.pred_off_coverage([3, 5], 9, 30, 0),
           W.pred_caution([3, 4, 5], 5, 35, 1), W.pred_no_skill(),
           W.pred_unclear(("1.2", False), [W.clause_wide(["the Overall score"])]),
           W.pred_reliable([4], 12, 12, 1), W.clause_guard(settings_only=True),
           W.progress_one_round(values=True), W.progress_one_round(cold=False),
           W.progress_too_early("96.00", 100, 1, one_round=True),
           W.progress_too_early("96.00", 100, 1, one_round=True, values=True),
           W.progress_tied(1, 3, "96.00", 100.0), W.progress_luck("90", 24),
           W.progress_rising(5, "2.5", "89.84", 100, 18, "1.2"), W.vary_threshold(),
           W.vary_threshold("1.2"), W.tie_runner_up(18, "90.12", W.beat_by()),
           W.tie_runner_up(18, "90.12", W.beat_by("1.2")),
           W.next_plain_repeat(21, 11, "10.5"), W.next_plain_repeat(21, 11, "4.0", False, "7.4",
                                                                    values=True),
           W.next_plain_repeat_ask(15), W.next_plain_repeat_ask(15, values=True),
           W.next_plain_repeat_ask(15, open_round=True, after_results=True),
           W.next_plain_misses(), W.next_plain_no_skill(),
           W.next_plain_untested(6, values=True), W.next_plain_explore("1.1", values=True),
           W.next_plain_close(settings_only=True, repeat_no=7, values=True),
           W.next_plain_close(W.widen_limit("Fat per 100 g: at most 10 g")),
           W.next_plain_see(), W.next_plain_see(repeat_no=15), W.next_plain_see(6, 4),
           W.next_plain_rising()]
    for _, pred, _, prog, nxt in FIRST_SCREENS.values():
        out += [pred, prog, nxt]
    return out + list(UNREACHED.values())


def _fold_lines():
    return [W.FOLD_PREDICTION_FIGURES, W.FOLD_PROGRESS_FIGURES, W.FOLD_NEXT_REASONING,
            W.FOLD_MEASURED_CHART, W.FOLD_PROGRESS_CHART, W.FOLD_METHOD, W.FOLD_EXPLORED,
            W.try_one_fold(True), W.try_one_fold(False), W.try_one_fold(True, False),
            W.trade_offs_fold(), W.trade_offs_fold(False), W.BASIS_PREDICTIONS,
            W.BASIS_PROGRESS, W.BASIS_NEXT, W.basis_explored(), W.basis_explored(values=True),
            W.basis_explored(total=False)]


def _tooltip_lines():
    return [W.TIP_INTERVAL, W.TIP_VERDICT, W.TIP_NOT_JUDGED, W.TIP_SKILL, W.TIP_ROUND_AHEAD,
            W.TIP_MEAN_ERROR, W.TIP_NOISE_THRESHOLD, W.tip_repeats(), W.tip_repeats(True),
            W.TIP_PROGRESS_WORDS, W.tip_predicted_score(), W.tip_predicted_score(True),
            W.TIP_MODEL_SD, W.TIP_PREDICTED_BEST, W.TIP_EXPECTED_IMPROVEMENT, W.tip_untested(),
            W.tip_untested(True), W.tip_local_effect(), W.tip_local_effect(True),
            W.TIP_ACTIVE_BOUNDS, W.TIP_LUCK, W.TIP_ROLES, W.TIP_NEAREST_TESTED,
            W.tip_coverage(["Firmness", "Juiciness", "Cook loss"], [3, 4, 5], 32, 35, 3),
            W.tip_coverage(["Firmness"], [6], 11, 12, 1),
            W.tip_coverage_early(["Firmness", "Cook loss"], [2, 3], 6),
            W.tip_coverage_early(["Firmness"], [], 1), W.TIP_COVERAGE_NONE,
            W.tip_skill([2, 3, 4, 5], "0.36", 15), W.tip_skill_too_few(2), W.TIP_REFITTED,
            W.TIP_RELIABLE_LIMITS, W.tip_repeat_off(21, 11, "10.5", "7.4"),
            W.tip_bias("−5.5", 16, [2, 3, 4, 5]), W.tip_wide(["Firmness", "Cook loss"]),
            W.tip_guard([W.GUARD_FLOOR, W.GUARD_NOISE]),
            W.tip_threshold_measured("1.2", 4, 3), W.tip_threshold_measured("1.2", 4),
            W.TIP_THRESHOLD_ASSUMED, W.tip_progress(W.TIP_THRESHOLD_ASSUMED),
            W.tip_next("Close", W.status_rule_rows()[-3][2])]


def _explored_lines():
    return [W.STRIPS_TITLE, W.strips_caption(), W.strips_caption(True),
            W.STRIP_TESTED, W.STRIP_WAITING, W.STRIP_SUGGESTED, W.STRIP_REACHABLE,
            W.STRIP_LATEST, W.STRIP_COUNT, W.strip_value_title(), W.strip_value_title(True),
            W.tip_strips(), W.tip_strips(values=True), W.tip_strips(total=False),
            W.TIP_LIMIT_HITS, W.tip_design_space("3,000"), W.MAP_TITLE, W.MAP_READING, W.MAP_NOT_YET, W.MAP_HIDDEN,
            W.MAP_COLD, W.TIP_MAP, W.explored_pointer(W.try_one_fold(True)),
            W.explored_pointer(W.try_one_fold(True, False), settings_only=True),
            W.explored_pointer(W.try_one_fold(False), triangle=False),
            W.why_link(6), W.repeat_link(24),
            W.SKILL_SCALE, W.skill_line("0.36", 15, [2, 3, 4, 5], "skill"),
            W.next_untested(round_no=6), W.next_untested(values=True, open_round=False)]


_GROUPS = [_status_lines, _fold_lines, _tooltip_lines, _explored_lines]


@pytest.mark.parametrize("group", _GROUPS)
def test_every_line_passes_the_vocabulary_guard(group):
    """The glossary is drawn from its own constants (below); nothing here
    leans on the allowance except Model and method's one lead sentence."""
    offenders = [t for t in group() if t not in _ALLOWED_EXACT
                 and any(p.search(t) for p in _BANNED)]
    assert offenders == []


@pytest.mark.parametrize("group", _GROUPS)
def test_no_line_uses_a_word_the_style_guide_retired(group):
    assert [t for t in group() if any(p.search(t) for p in _STYLE)] == []


@pytest.mark.parametrize("group", _GROUPS)
def test_at_most_one_dash_per_line(group):
    assert [t for t in group() if t.count("—") > 1] == []


@pytest.mark.parametrize("group", _GROUPS)
def test_trial_only_through_for_project(group):
    """No line types "trial"; on a study of process settings alone,
    for_project leaves no "formulation"."""
    opt = _SettingsOnly()
    for text in group():
        assert not re.search(r"\btrials?\b", text, re.I), text
        assert not re.search(r"\bformulations?\b", W.for_project(opt, text), re.I), text


# ------------------------------------------------------------------ #
#  The review's own texts (R-B1, R-I5), word for word
# ------------------------------------------------------------------ #

# Ftue fix A (statistician's cold read I5, I6, I7, I9) changed the facts
# of Lines A-C; the texts below are the review's words on the new facts.
# Fix A follow-up (decision 4): Line C's Within-noise sentence names the
# 3.2 points (1 × √2 × z_4 = 3.16) the rise is judged against (decision 4);
# Line B's Repeat limit is 8.2, the larger of 2.8 and its own predictive
# limit while Repeats have under 3 df (decision 2).
# Round 2: "tied" for equal values only; a close runner-up is "within noise".
_EXACT = {
    "Snack bar": (
        "The app starts checking its predictions once 5 formulations have results; 3 have so far.",
        "One Round so far, laid out to cover your allowed amounts. Formulations 1 and 3 are tied "
        "for best at 96.00 of 100.",
        "Make the next Round."),
    "Line A": (
        "In Rounds 3–5, 33 of 35 results landed inside the range the app predicted (up to 3 "
        "misses are normal), and score predictions beat the average of earlier results.",
        "Round 5 did not beat Best so far, 90.45 of 100 (Formulation 15, Round 4; within noise: "
        "Formulation 18). The app expects little more inside your current limits; the best it "
        "predicts sits at the most Textured pea protein and the least Beet juice powder allowed.",
        "Record the 4 formulations still open in Round 6. In the next Round, include a Repeat of "
        "Formulation 15 (the same amounts again) to measure how much results vary."),
    "Line B": (
        "In Rounds 3–5, 8 of 36 results fell outside the range the app predicted, where up to 4 "
        "are normal.",
        "No higher score in 2 Rounds. Best so far is 90.09 of 100 (Formulation 11, Round 3; within "
        "noise: Formulation 10).",
        "Check what changed before you make Round 6. Formulation 21, a Repeat of Formulation 11, "
        "scored 10.5 points lower, more than two makes of one formulation should differ (up to "
        "8.2 points). Look at the panel, suppliers and equipment."),
    "Line C": (
        "Score predictions do no better than simply using the average of earlier scores (skill "
        "−0.47, where 0 matches that average and 1 is exact); those for Firmness, Juiciness and "
        "Cook loss are still very uncertain, because results vary more from one formulation to "
        "the next than the app can explain, most for Cook loss. If the panel or method is new, "
        "check how consistently it measures.",
        "Round 5 raised Best so far by 2.5 points, to 89.84 of 100 (Formulation 18, within noise: "
        "Formulation 20), less than the 3.2 points needed to count as a rise.",
        # The review: "…treat the suggestions as a guided search…" (style
        # guide rule 7 retires "treat … as"; t0-report).
        "Record the 4 formulations still open in Round 6, then make the next Round. Until "
        "predictions beat the average, the suggestions are a guided search, not a forecast."),
    "Upgrade check": (
        "The app has checked its predictions against 2 formulations and needs 10 before it can "
        "say how far to trust them.",
        "Best so far is 90.50 of 100 (Formulation 6, Round 3). Rounds 1–2 were scored without "
        "Cook loss, so they are not compared with it.",
        "Make the next Round."),
    "Timing check": (
        "In Rounds 10–12, 58 of 60 results landed inside the range the app predicted (up to 5 "
        "misses are normal), and score predictions beat the average of earlier results.",
        "No higher score in 7 Rounds. Best so far is 92.50 of 100 (Formulation 24, Round 5; within "
        "noise: Formulation 3), probably helped by luck: a Repeat of it is predicted near 90.",
        "Make the next Round. The app still expects about 1.1 points of gain on average from "
        "amounts it has not tried, more than results are assumed to vary. Include a Repeat of "
        "Formulation 24 to measure how much they vary."),
}


@pytest.mark.parametrize("name", sorted(_EXACT))
def test_the_first_screens_of_the_review_word_for_word(name):
    pred_key, pred, prog_key, prog, nxt = FIRST_SCREENS[name]
    assert (pred, prog, nxt) == _EXACT[name]
    assert pred_key in W.PREDICTIONS_WORDS and prog_key in W.PROGRESS_WORDS


def test_the_status_words_and_labels():
    """The owner's words (2026-10-01), R-I10's Progress words, M7's label."""
    assert W.PREDICTIONS_WORDS == {"reliable": "Reliable", "not_yet": "Not yet", "off": "Off"}
    assert W.PROGRESS_WORDS == {"rising": "Rising", "within_noise": "Within noise",
                                "flat": "Flat", "too_early": "Too early",
                                "no_new_best": "No new best"}
    assert W.status_line("Predictions", "X.", "Reliable") == "Predictions: **Reliable.** X."
    assert W.status_line("Next", "Make the next Round.") == "Next: Make the next Round."


def test_the_templates_for_unreached_states():
    """R-B1: the states no demo project reaches, and its clause order."""
    assert W.pred_caution([3, 4, 5], 5, 35, 3) == (
        "In Rounds 3–5, more results than expected fell outside the range the app predicted "
        "(5 of 35, where up to 3 are normal).")
    assert W.PRED_WIDE == ("The app's predictions are still very uncertain, so results "
                           "landing inside them prove little yet.")
    assert W.pred_unclear() == ("Score predictions are not yet clearly better than simply "
                                "using the average of earlier scores.")
    assert W.PRED_OFF_REPEAT == (
        "A Repeat came out far from its original (see Next), so the conditions the predictions "
        "learned from may have changed.")
    assert W.pred_reliable([3, 4, 5], 36, 36, 4, [W.CLAUSE_OVER_WIDE]).endswith(
        "earlier results; the predicted range was wider than it needed to be.")
    assert W.pred_reliable([3], 11, 12, 1, [W.clause_wide(["Cook loss"]),
                                            W.clause_bias("1.2", False),
                                            W.clause_guard()]).endswith(
        "earlier results; those for Cook loss are still very uncertain; scores have come in "
        "about 1.2 points above the predictions; the app does not yet read what each ingredient "
        "does.")
    assert "(up to 1 miss is normal)" in W.pred_reliable([3], 11, 12, 1)
    # Fix A follow-up (decision 4): the threshold, not the noise SD.
    assert W.progress_within_noise(5, "0.4", "89.84", 100, 18, "2.3") == (
        "Round 5 raised Best so far by 0.4 points, to 89.84 of 100 (Formulation 18), less than "
        "the 2.3 points needed to count as a rise.")
    # R-I2: no line but Next's repeat row says "check".
    for text in (*_status_lines(),):
        if "check it before" in text:
            assert text.endswith(W.CHECK_IF_CHANGED), text
    assert "since Round" not in " ".join(_status_lines())


def test_the_next_sentences_one_per_row():
    """R-I5: the untested row points under Progress figures; the close row
    names what binds; a Repeat sentence says "in the next Round" while a
    Round is open."""
    assert W.next_plain_untested(6) == (
        "The model's highest prediction is in amounts you have not tried yet (its amounts are "
        "under Progress figures). To test it directly, add it to Round 6 as your own "
        "formulation.")
    assert W.next_plain_close(W.raise_highest("Textured pea protein")) == (
        "You are close to the top within these ingredients and limits: no formulation is "
        "predicted to beat Best so far by more than results vary. To look further, raise the "
        "Highest of Textured pea protein.")
    assert W.next_plain_misses() == (
        "Recent results fell outside the range the app predicted more often than they should, "
        "so new predictions can miss by more than their range shows. Make the next Round.")
    assert W.next_plain_rising(6, 4) == (
        "Best so far is rising. Record the 4 formulations still open in Round 6, then make the "
        "next Round.")
    assert W.next_plain_explore("1.1", no=6, left=4, repeat_no=24).endswith(
        "In the next Round, include a Repeat of Formulation 24 to measure how much they vary.")


def test_the_in_place_edits():
    """Review T0's three edits in place (R-I5, R-I7, R-I9) and their pins
    (M11)."""
    assert W.skill_line("−0.47", 16, [2, 3, 4, 5], "no_skill") == (
        "Overall score, Rounds 2–5: skill −0.47 (n = 16; 0 = as good as the average, 1 = "
        "perfect), no better than the average of earlier results.")
    assert "(Show its settings, under Progress figures)" in W.next_untested(values=True)
    # Round 2: "tied" for equal values only; close ones are within noise.
    assert W.tie_runner_up(18, "90.12", W.beat_by()) == (
        "Formulation 18, at 90.12, is within noise of Best so far: the 1-point noise "
        "threshold.")


def test_the_fold_labels_and_links():
    """§2, 02 §4, R-I4, R-I14."""
    # Concision round (2026-10-01): fold labels and links at most 5 words.
    assert W.FOLD_PREDICTION_FIGURES == "Prediction figures"
    assert W.FOLD_PROGRESS_FIGURES == "Progress figures"
    assert (W.FOLD_NEXT_REASONING, W.FOLD_MEASURED_CHART, W.FOLD_METHOD, W.FOLD_EXPLORED) == (
        "Next-step reasoning", "Measured against predicted", "Method and definitions",
        "Explored and unexplored")
    assert W.trade_offs_fold() == "Trade-offs"
    assert W.why_link(6) == "Why Round 6 was suggested"
    assert W.for_project(_SettingsOnly(), W.repeat_link(24)) == "Repeat Trial 24"
    # Every Basis line starts the same way (R-I6: the fold's help sits on it).
    for text in (W.BASIS_PREDICTIONS, W.BASIS_PROGRESS, W.BASIS_NEXT, W.basis_explored()):
        assert text.startswith("Basis: "), text


def test_the_status_tooltip_is_the_reviews_exact_rule():
    """R-B1: Line A's Predictions tooltip, and each window named on its own
    (I13)."""
    tip = " ".join([W.tip_coverage(["Firmness", "Juiciness", "Cook loss"], [3, 4, 5], 32, 35, 3),
                    W.tip_skill([2, 3, 4, 5], "0.36", 15), W.TIP_REFITTED])
    assert tip == (
        "Measurements (Firmness, Juiciness, Cook loss), Rounds 3–5: 32 of 35 Round-ahead values "
        "inside their 95 % posterior predictive intervals (5-fold cross-validation widening); a "
        "binomial count at 5 % per value allows up to 3 misses in 35 at the 1-in-10 level. "
        "Overall score, Rounds 2–5: out-of-sample skill 1 − SSE/SSE(mean of the rows before "
        "each Round) = 0.36 over 15 formulations, at least 0.20 over at least 10 needed. Each "
        "Round is predicted by models refitted today on earlier Rounds only.")
    # R-I11: Reliable's own limits, both figures.
    # Ftue fix A re-measured both (I9 folds): 12.5 % and 11.0 % (were 15 %
    # and 12 %); coverage 91 % and 90 % (was 89 %).
    assert "12.5 %" in W.TIP_RELIABLE_LIMITS and "11.0 %" in W.TIP_RELIABLE_LIMITS
    assert "91 %" in W.TIP_RELIABLE_LIMITS and "90 %" in W.TIP_RELIABLE_LIMITS


@pytest.mark.parametrize("tip", [
    W.TIP_INTERVAL, W.TIP_VERDICT, W.TIP_NOT_JUDGED, W.TIP_SKILL, W.TIP_ROUND_AHEAD,
    W.TIP_MEAN_ERROR, W.TIP_NOISE_THRESHOLD, W.tip_repeats(), W.TIP_PROGRESS_WORDS,
    W.tip_predicted_score(), W.TIP_MODEL_SD, W.TIP_PREDICTED_BEST, W.TIP_EXPECTED_IMPROVEMENT,
    W.tip_untested(), W.tip_local_effect(), W.TIP_ACTIVE_BOUNDS, W.TIP_ROLES,
    W.TIP_NEAREST_TESTED])
def test_each_term_tooltip_reads_plain_first_then_the_bo_term(tip):
    """§3: the plain reading first, then the standard term."""
    head, _, term = tip.partition(" BO term: ")
    assert head and term, tip
    assert ":" in head.split(".")[0], tip


# ------------------------------------------------------------------ #
#  Model and method (§7, R-I12, R-I13, M9)
# ------------------------------------------------------------------ #

def test_recommended_settings_read_as_rbf_with_ard_and_its_priors():
    text = " ".join(W.model_summary(_facts()))
    assert text.startswith(W.MODEL_SUMMARY_LEAD)
    assert ("Kernel (Recommended settings): squared exponential (RBF) with one lengthscale per "
            "input (ARD), lengthscale prior LogNormal(2.31, 1.73), at least 0.025; no "
            "outputscale.") in text
    assert "maximum a posteriori" in text and "not marginalised" in text
    assert "qLogNEI" in text and "q = 4 per Round" in text and "sequential greedy" in text
    assert "Cold start: the first 5 formulations" in text
    assert "rolling-origin (pseudo-prospective)" in text
    assert "Matérn" not in text and "Input warping" not in text


def test_each_setting_changes_the_text():
    """R-I13: kernel, noise and acquisition each change the summary; q
    follows the generation record."""
    base = W.model_summary(_facts())
    custom = W.model_summary(_facts(recommended=False, kernel="matern52", outputscale=True,
                                    outputscale_prior="Gamma(2.0, 0.15)", noise="fixed",
                                    acquisition="qucb", beta=0.2, q=None))
    text = " ".join(custom)
    assert "Kernel (your model settings): Matérn 5/2" in text
    assert "outputscale fitted, prior Gamma(2.0, 0.15)" in text
    assert "noise fixed near zero" in text
    assert "qUCB (upper confidence bound, β = 0.2)" in text
    assert "q = the number of formulations generated" in text
    assert base != custom
    assert "qLogEI (expected improvement over the best measured Overall score" in " ".join(
        W.model_summary(_facts(acquisition="qlogei")))
    assert "Input warping: on for the suggestion model" in " ".join(
        W.model_summary(_facts(robust=True)))


def test_the_gate_shows_only_what_applies():
    """M9: at the gate, only the cold start and the model to come."""
    lines = W.model_summary(_facts(cold=True))
    assert len(lines) == 2
    assert lines[0].startswith("Cold start:") and lines[1].startswith("The model to come:")
    assert W.MODEL_SUMMARY_LEAD not in lines


# ------------------------------------------------------------------ #
#  How each status is decided (I13, R-I13)
# ------------------------------------------------------------------ #

def test_one_rule_row_per_state_and_per_next_row_in_first_match_order():
    rows = W.status_rule_rows()
    assert all(len(r) == 3 for r in rows)
    lines = [r[0] for r in rows]
    assert lines == sorted(lines, key=["Predictions", "Progress", "Next"].index)
    predictions = [r[1] for r in rows if r[0] == "Predictions" and not r[1].startswith("Clause")]
    # B1's table, worst first: rows 1–9 of R-B1 (insights.PREDICTION_ROWS).
    assert [w.split(" (")[0] for w in predictions] == [
        "Off", "Off", "Not yet", "Not yet", "Not yet", "Not yet", "Not yet", "Not yet",
        "Reliable"]
    assert {r[1] for r in rows if r[0] == "Progress"} == set(W.PROGRESS_WORDS.values())
    # next_step's own first-match order.
    assert [r[1] for r in rows if r[0] == "Next"] == [
        "Repeat", "Misses", "Early", "No better than the average", "Untested", "Explore",
        "Close", "Rising", "See"]
    assert "−0.25" in " ".join(c for r in W.status_rule_rows("−0.25") for c in r)


# ------------------------------------------------------------------ #
#  The glossary (§3, R-B2, M4, M5, M8)
# ------------------------------------------------------------------ #

def test_the_glossary_keeps_one_order_in_both_forms():
    """M8: one canonical order, the status lines first; M5: the Round row."""
    assert len(W.GLOSSARY_ROWS) == len(W.GLOSSARY_ROWS_SETTINGS)
    assert W.GLOSSARY_ROWS[0][0] == "Reliable · Not yet · Off"
    assert W.GLOSSARY_ROUND_MEANING == (
        "the formulations made together; in BO texts a “batch” of q generated suggestions")
    shared = [a for a, b in zip(W.GLOSSARY_ROWS, W.GLOSSARY_ROWS_SETTINGS) if a == b]
    assert len(shared) >= len(W.GLOSSARY_ROWS) // 2


def test_the_glossary_is_allowed_only_where_it_must_be():
    """R-B2: of every glossary and Model-and-method string, only the
    allowlisted ones say a banned word, and every other line of this wave
    passes unaided."""
    every = set(W.glossary_cells()) | {W.MODEL_SUMMARY_LEAD}
    assert {t for t in every if any(p.search(t) for p in _BANNED)} == _FTUE_ALLOWED
    for group in _GROUPS:
        assert not set(group()) & (_FTUE_ALLOWED - {W.MODEL_SUMMARY_LEAD}), group.__name__


@pytest.mark.parametrize("name,value", sorted(_FROZEN.items()))
def test_the_frozen_strings_did_not_move(name, value):
    assert getattr(W, name) == value
