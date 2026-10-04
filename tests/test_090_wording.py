"""0.8.0 BO views, Task 0: every line the wave changes or adds is spelled
once in wording.py, in 03-wording-audit.md's style (spec 2026-09-30-bo-views
§4, §8 rev. 2; review I7, M3, M6): named statistics with their basis, one
term per quantity, no hedges, and every line passes the vocabulary guard."""
import re

import pytest

import insights
import wording
from tests.test_food_bo import _BANNED

W = wording


def _changed_lines():
    """One call of every builder, and every constant, 03 §5 rewrites."""
    return [
        W.insights_gate(3), W.insights_header_line(20, 5, 1, "Cook loss"),
        W.insights_header_line(20, 5, 2, "Cook loss"), W.Q_TRUST,
        W.trust_within(32, 35, [3, 4, 5], allowed=3), W.trust_within(32, 35, [3, 4, 5]),
        W.trust_misses(3, [3, 4, 5], 30, 35, allowed=3),
        W.trust_misses(3, [3, 4, 5], 26, 36, many=True, allowed=4),
        W.trust_over_wide(36, [3, 4, 5], 0.31), W.TRUST_WIDE, W.trust_early(6), W.trust_early(1),
        W.TRUST_NONE, W.wide_line("Firmness, Juiciness and Cook loss"), W.WIDE_EVERY,
        W.score_gap_line(11, 11, -0.4), W.score_gap_line(10, 10, -1.2, W.beat_by()),
        W.score_gap_line(11, 11, 0.0), W.repeat_disagrees(21, 11, "10.5", half_text="7.4"),
        W.TRUST_BEFOREHAND, W.FOLD_MEASURED_CHART, W.MVP_CAPTION, W.TRACK_PREDICTED,
        W.TRACK_LOW, W.TRACK_HIGH, W.PREDICTED, W.Q_STAND, W.stand_yes("2.5"),
        W.stand_barely("0.2"), W.stand_barely("0.2", "1.2"), W.stand_not_lately(7),
        W.stand_best("90.45", 15, 4), W.stand_repeats("1.2", 1), W.stand_repeats("1.2", 4),
        W.pairs_text(1), W.beat_by("1.2"), W.beat_by(),
        W.stand_not_higher(W.beat_by(), "90", "86", "94", 18),
        W.stand_not_higher(W.beat_by(), "88", "63", "100"),
        W.stand_higher("88", "63", "100", gain_text="+2.3"),
        W.stand_higher("90", "85", "95", 18, "90.12", "+2.3"),
        W.stand_limits(["the Highest of Pea"]), W.best_luck(11, "84", "62", "100", "90.09"),
        W.FOLD_PROGRESS_CHART, W.progress_caption_2(), W.Q_NEXT, W.next_bias("5.4", rounds=[3, 4, 5]),
        W.next_misses([3, 4, 5]), W.NEXT_MISSES, W.NEXT_TOO_EARLY, W.next_too_early_open(6, 4),
        W.next_untested(round_no=6), W.next_untested(open_round=False),
        W.next_close_to_top(W.raise_highest("Textured pea protein")), W.next_close_to_top(),
        W.NEXT_RISING, W.next_rising("Firmness", 6, 4), W.NEXT_SEE, W.next_see_open(6, 4),
        W.try_one_fold(True), W.try_one_fold(False), W.try_one_fold(True, False),
        W.near_line("Coconut oil", "1.4", "Best so far"),
        W.near_line("Coconut oil", "1.4", "Best so far", only=True),
        W.near_two("A", "B", "8.7", "8.2", "Best so far"), W.near_bigger(), W.near_none(),
        W.moved_together("Pea", "Soy", "0.86"), W.change_caption(), W.CHANGE_NEAR,
        W.CHANGE_WHOLE, W.CHANGE_AXIS, W.path_key(15, grey=True, targets=True), W.Q_PAIR,
        W.PAIR_CAPTION, W.pair_caption(["Firmness"]), W.pair_caption([]), W.Q_MIXTURE,
        W.mixture_caption(15), W.FOLD_METHOD, W.HOW_MODELS, W.HOW_RANGE,
        W.HOW_RANGE_FIXED_TINY, W.how_checks(), W.how_checks(values=True), W.HOW_NO_P,
        W.details_window([2, 3, 4, 5], 3, [3, 4, 5]), *W.DETAILS_COLUMNS_2, W.SKILL_COLUMN,
        W.HOW_WIDENED,
        W.score_widen_text(2.2, 1.6), W.HOW_WIDENED_SCORE, W.HOW_REPEATS_COMPARED,
        W.repeat_compare(1.34, 1.0, 4), W.repeat_compare(2.8, 1.0, 4),
        W.repeat_compare(None, None, 2), W.repeat_compare(None, 0, 3), W.HOW_THRESHOLDS,
        W.HOW_BIAS, W.PREDICTED_SCORE_COLUMN, W.PREDICTED_SCORE_HELP,
        W.predicted_score_help(20, "1.56"), W.predictions_round_caption(),
        W.PREDICTED_MEASUREMENTS, W.PREDICTED_MEASUREMENTS_CAPTION, W.TOO_UNCERTAIN,
        W.round_track_holding(11, 11, rounds=[3, 4, 5]),
        W.round_track_holding(11, 12, True, rounds=[3, 4, 5]),
        W.round_track_bias(10, 10, "1.2", rounds=[3, 4, 5]),
        W.round_track_rough(9, 12, "2.0", rounds=[3, 4, 5]),
        W.round_track_rough(7, 12, rounds=[3, 4, 5], allowed=2),
        W.waiting_line(1, "Cook loss"), W.tie_line(22, "90.65", "0.20", 15, "2.3"),
        W.tie_line(22, "90.65", "0.20", 15, "4.0"), W.tie_equal(22, 15, "90.45"),
        W.OVERALL_SCORE_HELP,
    ]


def _new_lines():
    """One call of every builder, and every constant, §§1-5 add."""
    return [
        W.skill_line("0.36", 15, [2, 3, 4, 5, 6], "skill"),
        W.skill_line("0.12", 12, [2, 3], "unclear"),
        W.skill_line("−0.47", 16, [2, 3, 4, 5, 6], "no_skill"), W.skill_cell("0.36", 15),
        W.SKILL_UNDER_TEN, W.how_skill(), W.how_skill("−0.25"), W.next_no_skill("−0.47", 16),
        W.next_no_skill("−0.47", 16, 6, 4), W.round_track_no_skill("−0.47", 16),
        W.next_no_skill("−0.47", 16, 6, 4, [2, 3, 4, 5]),
        W.round_track_no_skill("−0.47", 16, [2, 3, 4, 5]),
        W.why_guard_line([W.GUARD_FLOOR]), W.why_guard_line(skill_only=True),
        W.why_guard_line(verdict=True), W.gain_line_tiny(12, W.beat_by()),
        W.gain_line_tiny(12, W.beat_by(), recomputed=True), W.start_suggested(23, edited=True),
        W.why_cold(), W.why_cold(True), W.gain_cold(3, values=True),
        W.why_chart_caption("3,000", line=False), W.WHY_REFERENCE,
        W.mixture_within_noise("4.1", "5.0"), W.GUARD_FLOOR, W.GUARD_NOISE,
        W.guard_line([W.GUARD_FLOOR]), W.guard_line([W.GUARD_FLOOR, W.GUARD_NOISE], True),
        W.guard_skill_line(), W.guard_skill_line(True), W.PATH_GUARD, W.PATH_GUARD_SKILL,
        W.WHY_THESE, W.EXPECTED_IMPROVEMENT, W.OPTIMISTIC_SCORE, W.MODEL_SD, W.NEAREST_TESTED,
        W.LARGEST_DIFFERENCES, W.ROLE, W.ROLE_REFINES, W.ROLE_BALANCED, W.ROLE_EXPLORES,
        W.TIP_MODEL_SD, W.EXPECTED_IMPROVEMENT_HELP, W.EXPECTED_IMPROVEMENT_EI_HELP,
        W.OPTIMISTIC_SCORE_HELP, W.WHY_PENDING, W.WHY_COLD, W.WHY_FAILED, W.WHY_WORKING,
        W.why_row_label(24), W.why_row_label(25, own=True), W.why_row_label(26, edited=True),
        W.why_difference("Coconut oil", "+1.20 g"), W.why_chart_caption(3000),
        W.why_no_roles(120), W.why_role_rule(), W.HOW_EXPECTED_IMPROVEMENT, W.GAIN_HEADING,
        W.WHY_AXIS_MEAN, W.WHY_POOL, W.WHY_TESTED, W.WHY_SUGGESTED, W.WHY_RECOMPUTED,
        W.GAIN_BEST_MEASURED, W.GAIN_HIGHEST_PREDICTED, W.HOW_VERDICT, W.HOW_NOISE_THRESHOLD,
        W.gain_line("0.12", 6, W.beat_by()), W.gain_line("1.40", 6, W.beat_by("1.2"), above=True),
        W.gain_line("0.12", 5, W.beat_by(), recomputed=True),
        W.gain_line("1.08", 13, W.beat_by(), above=True, predicted_none=True),
        W.next_explore(13, 3), W.next_explore(), W.gain_line("0.37", None, W.beat_by()),
        W.gain_cold(None), W.gain_line_tiny(None, W.beat_by()),
        W.gain_line("0.30", 6, W.beat_by(), measured_text="90.45"),
        W.GAIN_UCB, W.gain_cold(2), W.gain_unreliable(W.GAIN_REASON_VERDICT),
        W.gain_unreliable(W.GAIN_REASON_SKILL), W.gain_unreliable(W.GAIN_REASON_GUARD),
        W.GAIN_AXIS, W.GAIN_EXACT, W.GAIN_RECOMPUTED, W.GAIN_STATUS, W.GAIN_CHART_CAPTION,
        W.LAYER_PICK, W.LAYER_SCORE, W.LAYER_SD, W.LAYER_GAIN, W.LAYER_SD_LEGEND,
        W.LAYER_GAIN_LEGEND, W.OVERLAY_NAME, W.overlay_caption("88"), W.start_suggested(22),
        W.layer_caption(W.LAYER_SD, "Formulation 22 (suggested, still to make)"),
        W.layer_caption(W.LAYER_GAIN, "Formulation 22 (suggested, still to make)"),
        W.path_layer_caption(W.LAYER_SD, "Formulation 22"),
        W.path_layer_caption(W.LAYER_GAIN, "Formulation 22"),
        W.round_measured_against_predicted(6), W.LINK_PREDICTION_CHECK, W.LINK_NEXT_STEP,
    ]


_STYLE = [re.compile(p, re.I) for p in (
    r"\blikely\b", r"\btreat\b", r"\bthough\b", r"stands out", r"Nothing is predicted",
    r"\bworth\b", r"keep in mind", r"\brough\b", r"some luck", r"\bbarely\b", r"On track",
    r"too few to be sure", r"\ba little\b", r"\boptimiser\b", r"\bheld\b")]


@pytest.mark.parametrize("group", [_changed_lines, _new_lines])
def test_every_line_passes_the_vocabulary_guard(group):
    offenders = [t for t in group() if any(p.search(t) for p in _BANNED)]
    assert offenders == []


@pytest.mark.parametrize("group", [_changed_lines, _new_lines])
def test_no_line_uses_a_word_the_style_guide_retired(group):
    """03 §3 rules 3, 5, 7: one term per quantity, no hedges, no advice
    words; "likely range" is gone everywhere."""
    offenders = [t for t in group() if any(p.search(t) for p in _STYLE)]
    assert offenders == []


@pytest.mark.parametrize("group", [_changed_lines, _new_lines])
def test_at_most_one_dash_per_line(group):
    """03 §3 rule 9: no em-dash chains."""
    assert [t for t in group() if t.count("—") > 1] == []


def test_the_process_only_forms_say_trial_through_for_project():
    """A process-only study's lines are made from these through for_project;
    no line here types the word (review M3)."""
    for text in _new_lines() + _changed_lines():
        assert not re.search(r"\btrials?\b", text, re.I), text


def test_headings_are_nouns():
    """03 §4, review I7: noun headings, and the link labels follow them."""
    assert (W.Q_TRUST, W.Q_STAND, W.Q_NEXT) == ("Prediction check", "Progress", "Next step")
    # Ftue §2 (integration): the folds' new names.
    assert W.FOLD_MEASURED_CHART == "Measured against predicted"
    assert W.FOLD_PROGRESS_CHART == "Progress chart"
    assert W.FOLD_METHOD == "Method and definitions"
    assert W.GAIN_HEADING == "Expected gain by Round"
    assert W.OVERLAY_NAME == "Predicted above Best so far"
    assert W.round_measured_against_predicted(6) == "Round 6: measured against predicted"
    assert W.LINK_PREDICTION_CHECK == "Prediction check"
    assert W.LINK_NEXT_STEP == "Next step"
    for heading in (W.Q_TRUST, W.Q_STAND, W.Q_NEXT, W.Q_PAIR, W.Q_MIXTURE, W.WHY_THESE,
                    W.GAIN_HEADING):
        assert not heading.endswith("?"), heading


def test_exact_texts_from_the_audit():
    """03 §5, as adopted by spec §4 (review I7 overrides the speed spec's §9
    B1, M9, I2 and I16 texts)."""
    # T0 review I-5: the two owner-mandated corrections, pinned whole, and
    # the 0.674 tied to the code's own constant.
    assert W.HOW_REPEATS_COMPARED == (
        "Error ÷ Repeat noise: the median absolute error divided by 0.674 × Repeat SD, which "
        "is the median absolute error Repeat noise alone would give. Near 1: the model misses "
        "by about as much as measurement noise. Above about 1.5: it misses by more than noise "
        "explains (lack of fit). Shown from 3 Repeat pairs.")
    assert f"{insights.MAD_OF_NORMAL:.3f}" in W.HOW_REPEATS_COMPARED
    assert W.HOW_WIDENED == (
        "Calibration factor: the root-mean-square of standardised errors (measured − "
        "predicted, ÷ predictive SD) in 5-fold cross-validation of the current model, the folds "
        "grouped by Round (a formulation and its Repeats in one fold), floored "
        "at 1. Every 95 % interval is multiplied by it. ×1.0: not widened; ×2.5: errors 2.5 "
        "times the model's own SD.")
    assert f"{insights.FOLDS}-fold" in W.HOW_WIDENED
    # T0 review I-1: the Formulation total is named only where there is one.
    assert W.how_checks().startswith(
        "Round-ahead check: each Round is predicted by models fitted only to the formulations "
        "recorded before it, at the amounts it was made at, brought to the Formulation total "
        "when made off it, and scored")
    assert "Formulation total" not in W.how_checks(total=False)
    assert "at the amounts it was made at, and scored" in W.how_checks(total=False)
    assert "at the values it was run at, and scored" in W.how_checks(values=True, total=False)
    # T0 review I-2: "calculated", never "worked out".
    assert (W.WHY_WORKING, W.WHY_FAILED) == ("Calculating why these were suggested…",
                                             "These values could not be calculated.")
    # T0 review I-3: Make a round names the unreliable verdict by its name.
    assert W.round_track_holding(11, 12, True, rounds=[3, 4, 5]).endswith(
        "; measured values: many more misses than expected.")
    assert W.round_track_rough(9, 12, "2.0", rounds=[3, 4, 5]).endswith(
        "; Overall scores have come in about 2.0 points below the predictions. More misses "
        "than expected.")
    # T0 review I-4: |r|, across the formulations with all results in.
    assert W.moved_together("Pea", "Soy", "0.86") == (
        "Pea and Soy are correlated across the formulations with all results in (|r| = 0.86), "
        "so their separate effects cannot be told apart.")
    assert W.trust_within(32, 35, [3, 4, 5], allowed=3) == (
        "Measurements, Rounds 3–5: 32 of 35 values inside their 95 % interval. As expected: "
        "95 % intervals allow up to 3 misses in 35.")
    assert W.trust_misses(3, [3, 4, 5], 30, 35, allowed=3) == (
        "Measurements, Rounds 3–5: 30 of 35 values inside their 95 % interval. 5 misses; "
        "95 % intervals allow up to 3. More misses than expected.")
    assert W.score_gap_line(11, 11, -0.4) == (
        "Overall score, same Rounds: 11 of 11 inside their 95 % interval; mean error −0.4 "
        "points (measured minus predicted).")
    assert W.score_gap_line(10, 10, -1.2, W.beat_by()) == (
        "Overall score, same Rounds: 10 of 10 inside their 95 % interval; mean error −1.2 "
        "points (n = 10). This exceeds the 1-point noise threshold and 2 standard errors: "
        "Overall scores have come in about 1.2 points below the predictions.")    # polish round 8
    assert W.repeat_disagrees(21, 11, "10.5", half_text="7.4") == (
        "Formulation 21, a Repeat of Formulation 11, scored 10.5 points lower. The 95 % "
        "interval for the difference of two results is ±7.4 points. If something changed "
        "(panel, supplier, equipment), check it before relying on the next suggestions.")
    assert W.stand_barely("0.2", "1.2") == (
        "Within noise: up 0.2 points in the last Round, under the noise threshold (Repeat SD "
        "1.2 points).")
    assert W.stand_not_higher(W.beat_by(), "90", "86", "94", 18) == (
        "Predicted best: 90 (95 % interval 86–94), near Formulation 18. Predicted gain over "
        "Best so far: none beyond the 1-point noise threshold.")
    assert W.near_line("Coconut oil", "1.4", "Best so far") == (
        "Largest local effect from Best so far: Coconut oil, 1.4 points.")
    assert W.PREDICTED_SCORE_COLUMN == "Predicted Overall score (95 % range)"
    assert W.predictions_round_caption() == "Exploring suggestions can predict below Best so far."
    assert W.predictions_round_help() == (
        "Suggestions favour formulations the model is unsure about as well as ones it predicts "
        "high, so one can be predicted below Best so far.")
    # Fix A follow-up (ruling on ties): the q = 1 rise threshold, √2 × 1.645.
    assert W.tie_line(22, "90.65", "0.20", 15, "2.3") == (
        "Tied with Formulation 15: 0.20 apart, under the 2.3 points needed to differ.")
    assert W.round_track_holding(11, 11, rounds=[3, 4, 5]) == (
        "Overall score, Rounds 3–5: 11 of 11 inside their 95 % interval.")
    assert W.HOW_NO_P.endswith("R² on formulations chosen this way is misleading, so skill "
                               "uses the Round-ahead baseline instead.")


def test_the_new_lines_from_the_rulings():
    """Review I2 (skill states), I3 (guard lines), I5 (expected gain)."""
    # Insights for a first-time user (ftue spec §4, review R-I7, M11): the
    # scale's two ends ride with the number, the one sanctioned edit of the
    # verbatim figures (T0).
    assert W.skill_line("0.36", 15, [2, 3, 4, 5, 6], "skill") == (
        "Overall score, Rounds 2–6: skill 0.36 (n = 15; 0 = as good as the average, 1 = "
        "perfect), better than the average of earlier results.")
    assert W.skill_line("−0.47", 16, [2, 3, 4, 5, 6], "no_skill").endswith(
        ", no better than the average of earlier results.")
    # Fix round 1 (ledger ruling on T2 I-1): between the cut and 0.2.
    assert W.skill_line("0.12", 12, [2, 3], "unclear").endswith(
        ", not clearly better than the average of earlier results.")
    # 0.8.0 e2e fixes D4 (review M10): the cut stated plainly, no table.
    # Ftue fix A re-measured the rates (I9 folds): 4.1 % and 12.5 % at n = 10.
    assert W.how_skill("−0.20").endswith(
        "At or below −0.20: no better than the average; below 0.20: not clearly better; "
        "0.20 or more: better. The cut is −0.20. In 24 simulated projects whose model's errors "
        "matched its own stated uncertainty (true skill 0.20 or more), the skill read “no "
        "better than the average” in 4.1 % of windows of 10 checks (n = 10), and below "
        "0.20 in 12.5 %; near a true skill of 0.20 it reads below 0.20 about half the "
        "time. Ingredient effects are reported only from 10 values and skill 0.20.")
    assert W.guard_line([W.GUARD_FLOOR]) == (
        "The model is fitting noise in these results (the model's correlation length is at its "
        "lower limit), so ingredient effects are not reported.")
    assert "setting effects" in W.guard_line([W.GUARD_NOISE], settings_only=True)
    # Final review M-2: the rule is named (tests/test_106_last_fixes.py).
    assert W.guard_skill_line() == (
        "Ingredient effects are not reported until Round-ahead predictions are clearly "
        "better than the average of earlier results (skill 0.20 needed).")
    assert W.EXPECTED_IMPROVEMENT == "Expected gain (model, one formulation)"
    # 0.8.0 e2e fixes A5 (review I6): what the statistic is.
    assert W.gain_line("0.12", 6, W.beat_by()) == (
        "Expected gain, Round 6: 0.12 points (model, one formulation): the model's "
        "average gain of one new formulation over the highest the model draws for the tested "
        "formulations (approximate), "
        "counting no gain as zero; under the 1-point noise threshold.")
    # Fix round 1 (ledger ruling C1): above the threshold with no predicted gain.
    # Final review M-2: precise, the number once, and not the same sentence twice.
    progress = W.gain_line("1.08", 13, W.beat_by(), above=True, predicted_none=True)
    assert progress == (
        "Expected gain, Round 13: 1.08 points (model, one formulation): the model's "
        "average gain of one new formulation over the highest the model draws for the tested "
        "formulations (approximate), "
        "counting no gain as zero; above the 1-point noise threshold. It comes from the "
        "model's uncertainty, not from a higher predicted score.")
    assert progress.count("1.08") == 1
    nxt = W.next_explore(13, 3)
    assert nxt == (
        "Record the 3 formulations still open in Round 13: the model's uncertainty, not a "
        "higher predicted score, puts their expected gain above the noise threshold.")
    assert nxt not in progress and "1.08" not in nxt
    assert W.next_explore() == (
        "Make the next Round: the model's uncertainty, not a higher predicted score, puts its "
        "expected gain above the noise threshold.")
    # Final review I-1: with no Round open the figure is the next Round's.
    assert W.gain_line("0.37", None, W.beat_by()).startswith(
        "Expected gain, next Round: 0.37 points (model, one formulation): ")
    assert W.gain_line("0.37", None, W.beat_by()).endswith("under the 1-point noise threshold.")
    assert W.MODEL_SD == "Model sd (points)"                        # final review M-3
    # T56 m3: a machine zero is said as under 0.01 points, with no reading.
    assert W.gain_line_tiny(12, W.beat_by()) == (
        "Expected gain, Round 12: under 0.01 points (model, one formulation), under "
        "the 1-point noise threshold.")
    assert W.start_suggested(23, edited=True) == "Formulation 23 (edited, still to make)"
    assert "Best so far's measured 90.45" in W.gain_line("0.30", 6, W.beat_by(),
                                                         measured_text="90.45")
    assert W.OPTIMISTIC_SCORE == "Optimistic score (mean + 0.45 × sd)"
    assert W.start_suggested(22) == "Formulation 22 (suggested, still to make)"


@pytest.mark.parametrize("value,text", [(-1.23, "−1.2"), (0.36, "+0.4"), (0.0, "0.0"),
                                        (-0.04, "0.0")])
def test_signed(value, text):
    assert W.signed(value) == text


def test_first_sentence_cuts_at_the_first_full_stop():
    assert W.first_sentence("Overall scores in Rounds 3–5 averaged 5.4 points below "
                            "prediction. Check the panel.") == (
        "Overall scores in Rounds 3–5 averaged 5.4 points below prediction.")
    assert W.first_sentence("One sentence only") == "One sentence only"


def test_dead_lines_are_gone():
    """03 §5.7: unused lines that still carried the old words."""
    for name in ("score_record_line", "CHECKING", "TRACK_INSIDE"):
        assert not hasattr(W, name), name


def test_the_t0_review_minors():
    """Fix round 1 (T0 review M-1, M-6, M-8)."""
    assert "median absolute standardised error of 0.31 (0.67 expected)" in W.trust_over_wide(
        36, [3, 4, 5], 0.31)
    many = W.trust_misses(3, [3, 4, 5], 26, 36, many=True, allowed=4)
    assert many.endswith("the intervals are too narrow for these results.")
    assert "recent" not in many
    assert "more often than 95 % intervals allow:" in W.next_misses([3, 4, 5])
    assert W.details_window([2, 3, 4, 5], 1, [5]).endswith("Verdict: the last Round (Round 5).")
    assert "±7.4 points." in W.repeat_disagrees(21, 11, "10.5", half_text="7.4")
    assert "3 degrees of freedom and is at least 0.05" in W.HOW_NOISE_THRESHOLD
    assert "average Overall score of the formulations" in W.how_skill()
