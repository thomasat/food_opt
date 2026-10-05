"""0.8.0 e2e fixes, Task 0: every line W1, W2 and W3 need is spelled once
in wording.py (spec 2026-10-01-e2e-fixes-design.md rev. 2; every "Proposed
ruling" of research/2026-10-01-e2e/spec-review.md), in 03-wording-audit.md's
style: it passes _BANNED and the retired-word list, has at most one dash,
says "trial" only through for_project, and the frozen strings stay."""
import re

import pytest

import wording as W
from tests.test_080_wording import _FROZEN
from tests.test_090_wording import _STYLE
from tests.test_food_bo import _BANNED


class _SettingsOnly:
    """A study of process settings alone, as for_project asks it."""

    def has_ingredients(self):
        return False

    def _process_settings(self):
        return ["Baking temperature"]


def _changed_lines():
    """One call of every builder, and every constant, the rulings edit in
    place."""
    return [
        W.SUGGESTION_CLOSE, W.SUGGESTION_BALANCED, W.SUGGESTION_DIFFERENT,
        W.compared_with_cell(W.SUGGESTION_DIFFERENT, "Water +12.00 g"), W.HOW_IT_WORKS[-1],
        W.how_models(), W.how_models("matern52"), W.how_models("rbf", "fixed_tiny"),
        W.how_models("linear", "low"), W.HOW_MODELS, W.how_verdict(), W.how_verdict(36, 4),
        W.how_verdict(12, 1), W.HOW_NOISE_THRESHOLD, W.how_checks(), W.how_checks(values=True),
        W.how_checks(total=False), W.beat_by(), W.beat_by("1.3"), W.beat_by(assumed=True),
        W.beat_by(assumed=True, pairs=1), W.beat_by(assumed=True, pairs=2), W.HOW_BIAS,
        W.score_gap_line(30, 32, -4.3, rounds=[2, 3, 4, 5, 6, 7]),
        W.score_gap_line(11, 11, 0.0, rounds=[3]), W.repeat_disagrees(21, 11, "10.5",
                                                                     half_text="7.4"),
        W.repeat_disagrees(20, 14, "1.6", lower=False), W.stand_repeats("9.6", 8, 5, 3),
        W.stand_repeats("2.7", 1, 1, 1), W.stand_repeats("1.3", 4), W.CHECK_IF_CHANGED,
        W.next_bias("4.1"), W.next_bias("3.3", below=False, rounds=[3, 4, 5, 6]),
        W.next_misses([5, 6, 7]), W.next_misses([5, 6, 7], W.next_bias("4.1", rounds=[3, 7])),
        W.NEXT_MISSES, W.how_skill(), W.how_skill("−0.25", settings_only=True),
        W.why_guard_line(["x"]), W.why_guard_line(skill_only=True),
        W.why_guard_line(verdict=True), W.WHY_CHECKING, W.ROLE_REFINES, W.ROLE_BALANCED,
        W.ROLE_EXPLORES, W.EXPECTED_IMPROVEMENT_HELP, W.EXPECTED_IMPROVEMENT_EI_HELP,
        W.WHY_PENDING, W.why_role_rule(), W.why_role_rule(values=True),
        W.why_role_rule(total=False), W.HOW_EXPECTED_IMPROVEMENT,
        W.gain_line("2.66", None, W.beat_by()),
        W.gain_line("0.30", 6, W.beat_by(assumed=True), measured_text="91.69"),
        W.gain_line("1.08", 13, W.beat_by(), above=True, predicted_none=True),
        W.gain_line("0.12", 5, W.beat_by("1.3"), recomputed=True), W.LAYER_GAIN,
        W.LAYER_GAIN_LEGEND, W.open_saved_copy_warning("Pea burger patty", ["24 formulations"],
                                                       "Pea burger patty", "28 formulations",
                                                       W.open_round_lost(7, 4)),
    ]


def _new_lines():
    """One call of every builder, and every constant, the new section adds."""
    return [
        W.bias_item("Overall score", "4.3"), W.bias_item("Firmness", "3.0", "N"),
        W.bias_item("Juiciness", "0.4", ""), W.bias_item("Overall score", "1.0"),
        W.steady_bias_line([W.bias_item("Overall score", "4.3")], rounds=[3, 4, 5, 6, 7]),
        W.steady_bias_line(low=[W.bias_item("Liking", "3.2")], rounds=[3, 4, 5]),
        W.steady_bias_line([W.bias_item("Overall score", "4.3")],
                           [W.bias_item("Juiciness", "0.4", "")], [3, 4]),
        W.HOW_BIAS_DIRECTION, W.HOW_CHANGE_LIMIT, W.MEAN_ERROR_COLUMN, W.FITTED_NOISE_COLUMN,
        W.mean_error_cell("−4.3"), W.mean_error_cell("−4.3", bias=True), W.fitted_noise_cell(),
        W.fitted_noise_cell("1.15"), W.HOW_FITTED_NOISE, *W.DETAILS_HELP.values(),
        W.effects_wait_line(6), W.effects_wait_line(1, settings_only=True),
        W.effects_unclear_line("0.09", 24), W.effects_unclear_line("0.12", 12, True),
        W.gain_unreliable(W.gain_reason_too_few(6)),
        W.gain_unreliable(W.gain_reason_unclear("0.09", 24)), W.path_guard_unclear("0.09", 24),
        W.path_guard_wait(6), W.NEXT_REPEAT_TO_MEASURE, W.repeat_score("67.30"),
        W.repeat_score("75.61", disagrees=True),
        W.best_repeats_line(14, ["67.30", "66.34", "75.61"], "75.17"),
        W.best_repeats_line(14, ["67.30", W.repeat_score("75.61", True)], "78.12"),
        W.best_repeats_line(26, ["77.97"], "79.90"),
        W.setup_callout("Textured pea protein", "30 g"),
        W.setup_callout("Salt", "0.5 g", highest=False),
        W.setup_callout_limit("Fat per 100 g: at most 20 g"), W.rest_needs_batch_size("Water"),
        W.import_limits_not_numbers("Date paste"),
        W.import_limits_not_numbers("Date paste", (W.HIGHEST_LABEL,)),
        W.import_spreadsheet_formula("Date paste", W.LOWEST_LABEL, "=B2*2"),
        W.open_round_lost(6, 4, 2), W.open_round_lost(7, 1, 1), W.open_round_lost(7, 4),
        W.YES_MOVE_TO_NOT_SCORED, W.gain_pending(), W.gain_pending(7), W.INSIGHTS_IN_PROCESS,
        W.role_help(), W.TIP_NEAREST_TESTED, W.LARGEST_DIFFERENCES_HELP,
        W.LAYER_SCORE_LEGEND, W.LAYER_GAIN_FORMER, W.KERNEL_NAMES["matern32"],
        # T0′ at integration: the interval chart's caption in both themes,
        # the progress line with every Round checked, the Why fold's A3
        # reasons in Progress's words.
        W.MVP_CAPTION, W.insights_progress(12, 12), W.insights_progress(1, 1),
        W.insights_progress(3, 12),
        W.why_guard_line(gate_reason=W.gain_reason_too_few(6)),
        W.why_guard_line(gate_reason=W.gain_reason_unclear("0.09", 24)),
        W.why_guard_line(gate_reason=W.GAIN_REASON_SKILL),
    ]


_GROUPS = [_changed_lines, _new_lines]


@pytest.mark.parametrize("group", _GROUPS)
def test_every_line_passes_the_vocabulary_guard(group):
    offenders = [t for t in group() if any(p.search(t) for p in _BANNED)]
    assert offenders == []


@pytest.mark.parametrize("group", _GROUPS)
def test_no_line_uses_a_word_the_style_guide_retired(group):
    """03 §3 rules 3, 5, 7. The I2 ruling's "treat new predictions as rough"
    is said without the retired words (next_misses)."""
    assert [t for t in group() if any(p.search(t) for p in _STYLE)] == []


@pytest.mark.parametrize("group", _GROUPS)
def test_at_most_one_dash_per_line(group):
    assert [t for t in group() if t.count("—") > 1] == []


@pytest.mark.parametrize("group", _GROUPS)
def test_trial_only_through_for_project(group):
    """No line types "trial"; on a study of process settings alone,
    for_project leaves no "formulation" in any of them."""
    opt = _SettingsOnly()
    for text in group():
        assert not re.search(r"\btrials?\b", text, re.I), text
        assert not re.search(r"\bformulations?\b", W.for_project(opt, text), re.I), text


def test_the_settings_only_forms_name_no_ingredient():
    """A line that names ingredients has a form for process settings alone."""
    for text in (W.why_role_rule(values=True), W.effects_wait_line(6, settings_only=True),
                 W.effects_unclear_line("0.09", 24, settings_only=True),
                 W.how_skill(settings_only=True), W.how_checks(values=True, total=False)):
        assert "ingredient" not in text.lower(), text
    assert "Amounts" not in W.why_role_rule(values=True)


def test_process_only_round_heading_says_trials():
    """B4 (04 M-3): the heading reaches the screen through for_project."""
    opt = _SettingsOnly()
    assert W.for_project(opt, W.make_these(4, 3)) == "**Round 4 · Make 3 trials**"
    assert W.for_project(opt, W.make_these(4, 1)) == "**Round 4 · Make 1 trial**"


@pytest.mark.parametrize("name,value", sorted(_FROZEN.items()))
def test_the_frozen_strings_did_not_move(name, value):
    assert getattr(W, name) == value


def test_import_aliases_are_frozen():
    """B1: a file written for these must keep loading."""
    assert W.FILL_TO_TOTAL_IMPORT_ALIASES == ("fill to total", "= rest", "=rest")
    assert W.LAYER_GAIN_FORMER == "Expected improvement"   # charts proto: one name, Expected gain


def test_one_vocabulary_for_role_and_compared_with():
    """A2 (02 I1): the Why fold's Role and the Compared with cell say the
    same three words."""
    assert (W.ROLE_REFINES, W.ROLE_BALANCED, W.ROLE_EXPLORES) == (
        W.SUGGESTION_CLOSE, W.SUGGESTION_BALANCED, W.SUGGESTION_DIFFERENT) == (
        "refines the best", "balanced", "explores")
    assert W.compared_with_cell(W.SUGGESTION_BALANCED, "Salt +0.56 g") == (
        "Balanced · Salt +0.56 g")


def test_no_line_advises_a_check_without_repeat_evidence():
    """Review B1, I1: only a disagreeing Repeat advises a check, and
    conditionally."""
    for text in (W.next_bias("4.1"), W.next_misses([5, 6, 7], W.next_bias("4.1")),
                 W.steady_bias_line([W.bias_item("Overall score", "4.3")]), W.HOW_BIAS):
        assert "Check" not in text and "check it" not in text, text
    assert W.repeat_disagrees(21, 11, "10.5").endswith(W.CHECK_IF_CHANGED)
    assert W.CHECK_IF_CHANGED.startswith("If something changed")


def test_exact_texts_from_the_rulings():
    """The spec-review rulings' own texts, adopted verbatim where the 03
    style allows (see the stage-1 report for the three that differ)."""
    # B1: one line naming every biased output.
    assert W.steady_bias_line([W.bias_item("Overall score", "4.3"),
                               W.bias_item("Firmness", "3", "N")], rounds=[3, 4, 5, 6, 7]) == (
        "Results have come in below the predictions: Overall score by about 4.3 points, "
        "Firmness by about 3 N (Rounds 3–7).")
    assert W.CHECK_IF_CHANGED == ("If something changed (panel, supplier, equipment), check it "
                                  "before relying on the next suggestions.")
    assert W.HOW_CHANGE_LIMIT == (
        "A change at the bench is learned from its first Round on, so later Round-ahead checks "
        "cannot show it; a Repeat of an earlier formulation in each Round would.")
    assert W.HOW_BIAS_DIRECTION == (
        "Suggestions sit where the model predicts high or is unsure, so results can come in "
        "below prediction (its optimism at its own maximum) or above it (its mean shrinks "
        "toward the average where it has seen little).")
    # B2: the assumed threshold, and the Repeat sentence (the new 02 R3 pin).
    assert W.stand_yes("6.3", W.beat_by(assumed=True)) == (
        "Rising: up 6.3 points in the last Round, above the 1-point noise threshold (assumed: "
        "no Repeats yet; a Repeat of Best so far would measure it).")
    assert W.NEXT_REPEAT_TO_MEASURE == (
        "Make a Repeat of Best so far in the next Round to measure the noise.")
    # I2: the unreliable row, with the bias appended.
    assert W.next_misses([5, 6, 7], W.next_bias("4.1")) == (
        "Measured values in Rounds 5–7 fell outside their 95 % interval more often than 95 % "
        "intervals allow: the intervals are too narrow here, so new predictions carry more "
        "error than their intervals show. Overall scores have come in about 4.1 points below "
        "the predictions.")
    # I4: effect claims withheld below n = 10 and below skill 0.20.
    assert W.effects_wait_line(6) == (
        "Ingredient effects are reported once Round-ahead checks show the Overall score's "
        "predictions clearly better than the average (skill 0.20 over 10 checked "
        "formulations; 6 so far).")
    assert W.effects_unclear_line("0.09", 24) == (
        "Ingredient effects are reported once Round-ahead predictions are clearly better than "
        "the average of earlier results (skill 0.09, n = 24; 0.20 needed).")
    # I6: the expected-improvement sentence, {ref} both ways.
    assert W.gain_line("2.66", None, W.beat_by()) == (
        "Expected gain, next Round: 2.66 points (model, one formulation): the model's "
        "average gain of one new formulation over the highest the model draws for the tested "
        "formulations (approximate), "
        "counting no gain as zero; under the 1-point noise threshold.")
    assert "over Best so far's measured 91.69, counting no gain as zero" in W.gain_line(
        "2.66", None, W.beat_by(), measured_text="91.69")
    for text in (W.gain_line("2.66", 6, W.beat_by()), W.HOW_EXPECTED_IMPROVEMENT,
                 W.EXPECTED_IMPROVEMENT_HELP):
        assert "beats" not in text and "best single" not in text, text
    assert (W.LAYER_GAIN, W.LAYER_GAIN_LEGEND) == ("Expected gain",
                                                   "Expected gain (points)")
    # I7: the Repeats line, two decimals.
    assert W.best_repeats_line(14, ["67.30", "66.34", "75.61"], "75.17") == (
        "Repeats of Formulation 14: 67.30, 66.34, 75.61. Mean of all: 75.17.")
    assert W.best_repeats_line(14, ["67.30", W.repeat_score("75.61", True)], "78.12") == (
        "Repeats of Formulation 14: 67.30, 75.61 (disagrees). Mean of all: 78.12.")
    # I8: the Set up callout.
    assert W.setup_callout("Textured pea protein", "30 g") == (
        "Textured pea protein: Highest 30 g. Next step suggests raising it; change the Highest "
        "cell in its row below.")
    # I9: the import refuses by row; the caption before a batch size exists.
    assert W.import_limits_not_numbers("Date paste") == (
        "Date paste: Lowest and Highest must be numbers.")
    assert W.rest_needs_batch_size("Date paste") == (
        "Date paste fills the remaining amount of the formulation. Set a default batch size in "
        "Preparation and records to calculate its amounts.")
    # I10: the restore warning.
    assert W.open_round_lost(6, 4, 2) == (
        "Round 6 is open with 4 formulations and 2 results typed; they will be lost.")
    assert W.open_saved_copy_warning("P", ["24 formulations"], "P", "28 formulations",
                                     W.open_round_lost(7, 4)).endswith(
        "which has 28 formulations? Round 7 is open with 4 formulations; they will be lost. "
        + W.COPY_KEPT)
    # I13: the triangle legend, the Method clause, the df.
    assert W.LAYER_SCORE_LEGEND == "Predicted Overall score (model mean)"
    assert "today's measurements, targets, Share of score and model settings" in (
        W.how_checks())
    assert W.stand_repeats("9.6", 8, df=5, groups=3) == (
        "Repeat SD of the Overall score: 9.6 points, 5 df (8 results of 3 formulations); this "
        "is the noise threshold.")
    # M10 / D4: the skill paragraph states its cut and its cost plainly.
    skill = W.how_skill()
    # Fix round B (statistician I4): the rates as a simulation's, with n;
    # ftue fix A re-measured them (I9 folds): 4.1 % and 12.5 % at n = 10.
    assert ("The cut is −0.20. In 24 simulated projects whose model's errors matched its own "
            "stated uncertainty (true skill 0.20 or more), the skill read “no better than the "
            "average” in 4.1 % of windows of 10 checks (n = 10), and below 0.20 in "
            "12.5 %") in skill
    assert "percentile" not in skill
    # M11 / D5: the verdict example from allowed_misses(n); the kernel named.
    assert "(for 36 values, up to 4 misses)" in W.how_verdict(36, 4)
    assert "kernel from your model settings: Matérn 5/2" in W.how_models("matern52")
    assert "default kernel: squared exponential" in W.HOW_MODELS
    # I3: the role rule's two cuts.
    rule = W.why_role_rule()
    assert "Refines the best: within 15 % of Best so far" in rule
    assert "Explores: more than 30 % from every formulation recorded before its Round" in rule
    assert "Balanced: in between." in rule
