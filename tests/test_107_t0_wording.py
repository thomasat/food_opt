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


def _fold_lines():
    return [W.FOLD_PREDICTION_FIGURES, W.FOLD_NEXT_REASONING,
            W.FOLD_MEASURED_CHART, W.FOLD_PROGRESS_CHART, W.FOLD_METHOD, W.FOLD_EXPLORED,
            W.try_one_fold(True), W.try_one_fold(False), W.try_one_fold(True, False),
            W.trade_offs_fold(), W.trade_offs_fold(False),
            ]


def _tooltip_lines():
    return [
            W.tip_local_effect(), W.tip_local_effect(True),
            W.TIP_NEAREST_TESTED,
            ]


def _explored_lines():
    return [W.STRIPS_TITLE, W.strips_caption(), W.strips_caption(True),
            W.STRIP_TESTED, W.STRIP_WAITING, W.STRIP_SUGGESTED, W.STRIP_REACHABLE,
            W.STRIP_LATEST, W.STRIP_COUNT, W.strip_value_title(), W.strip_value_title(True),
            W.MAP_TITLE, W.MAP_NOT_YET, W.MAP_HIDDEN,
            W.MAP_COLD, W.explored_pointer(W.try_one_fold(True)),
            W.explored_pointer(W.try_one_fold(True, False), settings_only=True),
            W.explored_pointer(W.try_one_fold(False), triangle=False),
            ]


_GROUPS = [_fold_lines, _tooltip_lines, _explored_lines]


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


def test_the_settings_only_forms_name_no_ingredient_or_amount():
    """A line that names ingredients or amounts has a form for process
    settings alone (review I14, M4). Re-pinned (one-glance Review, fix round
    1) on the live forms: the removed What have we tried? forms
    (strips_no_gap, explored_link) and glossary rows are gone."""
    for text in (
                 W.try_one_fold(True, False),
                 W.tip_local_effect(True), W.strips_caption(True),
                 W.explored_pointer(W.try_one_fold(True, False), settings_only=True),
                 *W.model_summary(_facts(values=True, total=False)),
                 *[c for row in W.status_rule_rows(settings_only=True) for c in row]):
        assert "ingredient" not in text.lower(), text
        assert "amount" not in text.lower(), text
        # A study of process settings alone has no Formulation total, and
        # for_project would make it "Trial total".
        assert "Formulation total" not in text, text


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


@pytest.mark.parametrize("tip", [
    W.tip_local_effect(), 
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

def test_the_glossary_is_allowed_only_where_it_must_be():
    """R-B2: of every glossary and Model-and-method string, only the
    allowlisted ones say a banned word, and every other line of this wave
    passes unaided."""
    every = {W.MODEL_SUMMARY_LEAD}
    assert {t for t in every if any(p.search(t) for p in _BANNED)} == _FTUE_ALLOWED
    for group in _GROUPS:
        assert not set(group()) & (_FTUE_ALLOWED - {W.MODEL_SUMMARY_LEAD}), group.__name__


@pytest.mark.parametrize("name,value", sorted(_FROZEN.items()))
def test_the_frozen_strings_did_not_move(name, value):
    assert getattr(W, name) == value
