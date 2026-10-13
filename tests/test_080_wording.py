"""0.8.0 speed and simplicity, Plan B, Task 0: every new line of the wave
is spelled once in wording.py, passes the vocabulary guard, says "spread"
nowhere, and the strings a workbook import matches stay exactly as they
were (spec 2026-09-30 §§2–6, §9 rev. 2; spec review B1, I14, I17, M3, M7)."""
import re

import pytest

import insights
import wording
from tests.test_food_bo import _BANNED

_SPREAD = re.compile(r"\bspread\b", re.I)


def _new_lines():
    """One call of every builder, and every constant, the wave adds."""
    w = wording
    return [
        w.pairs_text(1), w.pairs_text(2), w.pairs_text(4), w.points_text("1.0"),
        w.beat_by("1.2"), w.beat_by(),
        w.insights_header_line(24, 6, 1, "Cook loss"), w.insights_header_line(0, None, 3, ""),
        w.Q_TRUST, w.trust_within(32, 33, [4, 5, 6]), w.trust_over_wide(15, [4, 5]),
        w.TRUST_WIDE, w.trust_misses(3, [4, 5, 6], 20, 30), w.trust_misses(1, [6], 5, 12, True),
        w.trust_early(7), w.TRUST_NONE, w.TRUST_BEFOREHAND,
        w.score_gap_line(11, 11, -1.0), w.score_gap_line(6, 12, -5.2, "Repeats differ"),
        w.score_gap_line(6, 12, 0.02),
        w.repeat_disagrees(21, 11, "10.6"), w.repeat_disagrees(21, 11, "4.0", lower=False),
        w.MVP_CAPTION, w.Q_STAND, w.stand_yes("3.1"), w.stand_barely("0.2", "1.2"),
        w.stand_barely("0.4"), w.stand_not_lately(3), w.STAND_LATEST, w.stand_since(1),
        w.stand_since(2), w.stand_best("90.65", 15, 4), w.stand_best("90.65", 15, None, "Aroma"),
        w.stand_repeats("1.2", 1), w.stand_not_higher("Repeats differ (1.2 points)", "90", "86",
                                                      "94", 18),
        w.stand_not_higher("1 point", "90", "86", "94"), w.stand_higher("94", "89", "99"),
        w.stand_higher("94", "89", "99", 18, "90.12"),
        w.stand_limits([w.at_highest("Pea"), w.at_lowest("Salt"),
                        w.at_limit("Fat per 100 g: at most 20 g")]),
        w.show_its_amounts(), w.show_its_amounts(True), w.Q_NEXT, w.next_bias("5.0"),
        w.next_bias("2.0", below=False), w.NEXT_MISSES, w.NEXT_TOO_EARLY, w.next_untested(),
        w.next_untested(values=True),
        w.next_close_to_top(w.raise_highest("Coconut oil")),
        w.next_close_to_top(w.lower_lowest("Salt"), settings_only=True),
        w.next_close_to_top(w.widen_limit("Fat per 100 g: at most 20 g")),
        w.next_close_to_top(), w.next_close_to_top(settings_only=True),
        w.NEXT_RISING, w.NEXT_SEE, w.change_on_setup("Coconut oil"),
        w.change_on_setup("Salt", highest=False), w.CHANGE_LIMITS_ON_SETUP,
        w.try_one_fold(True), w.try_one_fold(False), w.try_one_fold(True, False),
        w.near_line("Coconut oil", "1.4", "Best so far"),
        w.near_line("Salt", "0.3", "Formulation 9", only=True),
        w.near_two("Coconut oil", "Salt", "1.4", "1.2", "Best so far"),
        w.near_bigger(), w.near_bigger(True), w.near_none(), w.near_none(True),
        w.near_none(False, False), w.near_start_untested("Formulation 9"),
        w.CHANGE_AXIS, w.change_axis("Firmness", "N"), w.change_caption(),
        w.change_caption(True, "Firmness"),
        w.path_what(w.path_head_fill("Coconut oil", "Water"), [("8.04 g", "Fat per 100 g")],
                    [("Salt", "1.5 g")]),
        w.path_what(w.path_head_proportional("Pea", "100 g")), w.path_what(w.path_head_free("Pea")),
        w.path_what(w.path_head_process("Cook temperature")),
        w.path_key(15, grey=True, targets=True), w.path_key(15),
        w.progress_caption_2(), w.progress_caption_2(True),
        w.details_window([2, 3, 4, 5], 3), w.details_window([2, 3], 2), w.details_window([4], 1),
        *w.DETAILS_COLUMNS_2, w.widen_text(2.1), w.HOW_WIDENED, w.HOW_REPEATS_COMPARED,
        w.HOW_THRESHOLDS, w.repeat_compare(None, None, 0), w.repeat_compare(2.0, 1.0, 2),
        w.repeat_compare(None, 0, 3), w.repeat_compare(2.34, 1.0, 4),
        w.TOO_UNCERTAIN, w.START_FROM_BEST_SO_FAR, w.HOW_CHOSEN_2, w.still_covering(5),
        w.waiting_line(1, "Cook loss"), w.waiting_line(2, "Cook loss and Aroma", True),
        w.waiting_line(3, ""), w.workbook_line(), w.workbook_line(True), w.WORKBOOK_TABS_HELP,
        w.workbook_tabs_help(True),
        w.RECORD_RESULTS_LINE, w.NOT_SCORED_TICK_HELP, w.more_label(), w.more_label(True),
        w.more_label(False, True),
        w.BEST_RULE, w.BEST_RULE_WITHOUT, w.overall_score_line(90.45, 100),
        w.overall_score_line(88.0, 100, "Aroma"), w.OVERALL_SCORE_HELP,
        w.tie_line(22, "90.65", "0.20", 15, "4.0"), w.tie_line(22, "90.65", "0.20", 15, "2.3"),
        w.MEASURED_WAITING, w.waiting_list("Cook loss", [22], 6),
        w.waiting_list("Cook loss", [21, 22]), w.MOVE_TO_NOT_SCORED_HELP,
        w.never_coming_warning(19, "Cook loss"), w.EDIT_PAST_FOLD,
        w.result_view_waiting("Formulation 22 (Round 6)", "Cook loss"),
        w.NEXT_MAKE_A_ROUND, w.SETUP_FIRST_LINE, w.setup_first_line(True),
        w.MEASUREMENT_GOAL_CAPTION,
        w.CHANGE_NEAR, w.CHANGE_WHOLE,
    ]


def test_every_new_line_passes_the_vocabulary_guard():
    offenders = [t for t in _new_lines() if any(p.search(t) for p in _BANNED)]
    assert offenders == []


def test_no_new_line_says_spread():
    """'spread' is kept for nothing: Repeats 'differ', an ingredient
    'changes … by up to' (spec §2, §6; review M3)."""
    assert [t for t in _new_lines() if _SPREAD.search(t)] == []


# The strings a workbook import compares cell text to, and the sheet names
# it looks up (spec review I17). A sweep that edits one breaks every
# workbook downloaded before it.
_FROZEN = {
    "MEASUREMENTS_SHEET_HEADING": "Measurements",
    "ACTUAL_SHEET_HEADING": "Actual amounts and settings — optional",
    "INGREDIENT_OR_SETTING_LABEL": "Ingredient or process setting",
    "PREMIX_LOT_ON_ITS_PAGE": "See its preparation sheet",
    "PREMIX_LOT_ON_PREPARATION": "See Preparation",
    "TICK_BOX": "☐",
    "WORKED_OUT": "calculated",
    "PART_OF_LABEL": "Part of",
    "MADE_AS_LABEL": "Preparation",
    "LOWEST_LABEL": "Lowest",
    "HIGHEST_LABEL": "Highest",
    "OVERVIEW_SHEET": "Round overview",
    "PREPARATION_SHEET": "Preparation",
    "RESULTS_SHEET": "Results",
    "NOT_SCORED": "Not scored",
    # Owner-confirmed 0.7.1 labels (spec §9 I17): not part of the sweep.
    "TAB_BATCH_NAME": "Make a round",
    "START_NEXT_BATCH": "Start the next round",
}


@pytest.mark.parametrize("name,value", sorted(_FROZEN.items()))
def test_import_matched_and_owner_confirmed_strings_are_frozen(name, value):
    assert getattr(wording, name) == value


def test_the_round_sheet_name_is_frozen():
    assert wording.batch_sheet_name(3) == "Round 3"


def test_exact_texts_from_the_spec():
    """Rewritten for the BO-views wave (spec 2026-09-30-bo-views §4, review
    I7): 03-wording-audit.md's texts override the speed spec's §9 B1, M9,
    I2 and I16 texts. Rewritten again for the 0.8.0 e2e fixes (spec
    2026-10-01 rev. 2, the Overrides clause of review I1): the Repeat row's
    check is conditional, the bias and misses rows advise no check, the
    score line names its own Rounds (every Round checked), the 1-point
    threshold says it is assumed, and the expected gain is said as
    the average gain it is. Rewritten again for Insights for a first-time
    user (spec 2026-10-01 rev. 2, review M11; T0): next_untested points
    under Progress figures; the first screen's own texts are pinned in
    tests/test_107_t0_wording.py."""
    w = wording
    assert w.insights_header_line(24, 6, 1, "Cook loss") == (
        "Based on 24 scored formulations (all results in), up to Round 6. 1 more is waiting "
        "for Cook loss and is not used yet.")
    assert w.TRUST_WIDE == (
        "Not judged yet: every 95 % interval is wide (median half-width over 25 % of the "
        "Range), so nearly any value falls inside.")
    # T0 review M-7: 4 misses are allowed from 36 values (33 allow 3).
    assert insights.allowed_misses(36) == 4
    assert w.trust_within(32, 36, [4, 5, 6], allowed=4) == (
        "Measurements, Rounds 4–6: 32 of 36 values inside their 95 % interval. As expected: "
        "95 % intervals allow up to 4 misses in 36.")
    assert w.score_gap_line(11, 11, -1.0) == (
        "Overall score, same Rounds: 11 of 11 inside their 95 % interval; mean error −1.0 "
        "points (measured minus predicted).")
    assert w.repeat_disagrees(21, 11, "10.6", half_text="7.4") == (
        "Formulation 21, a Repeat of Formulation 11, scored 10.6 points lower. The 95 % "
        "interval for the difference of two results is ±7.4 points. If something changed "
        "(panel, supplier, equipment), check it before relying on the next suggestions.")
    # Overrides (review I1 rows A1): the score line's window is every Round
    # checked, named; the bias and misses rows lose their "Check …".
    assert w.score_gap_line(30, 32, -4.3, rounds=[2, 3, 4, 5, 6, 7]) == (
        "Overall score, Rounds 2–7: 30 of 32 inside their 95 % interval; mean error −4.3 "
        "points (measured minus predicted).")
    assert w.next_bias("4.1", rounds=[3, 4, 5, 6, 7]) == (
        "Overall scores have come in about 4.1 points below the predictions (Rounds 3–7).")
    assert w.next_misses([5, 6, 7]) == (
        "Measured values in Rounds 5–7 fell outside their 95 % interval more often than 95 % "
        "intervals allow: the intervals are too narrow here, so new predictions carry more "
        "error than their intervals show.")
    # Overrides (review I1 row A4, ruled by B2): the threshold rule is
    # unchanged; below 3 pairs it says it is assumed.
    assert w.beat_by(assumed=True) == (
        "the 1-point noise threshold (assumed: no Repeats yet; a Repeat of Best so far would "
        "measure it)")
    assert w.beat_by(assumed=True, pairs=2) == (
        "the 1-point noise threshold (assumed: 2 pairs of Repeats, 3 needed; a Repeat of Best "
        "so far would measure it)")
    # Overrides (review I1 row A5): never "the best single new formulation
    # beats … by".
    assert w.gain_line("2.66", None, w.beat_by()) == (
        "Expected gain, next Round: 2.66 points (model, one formulation): the model's "
        "average gain of one new formulation over the highest the model draws for the tested "
        "formulations (approximate), "
        "counting no gain as zero; under the 1-point noise threshold.")
    assert w.stand_best("90.65", 15, 4) == (
        "Best so far: Overall score 90.65, Formulation 15 (Round 4).")
    assert w.stand_not_higher(w.beat_by("1.2"), "90", "86", "94", 18) == (
        "Predicted best: 90 (95 % interval 86–94), near Formulation 18. Predicted gain over "
        "Best so far: none beyond the noise threshold (Repeat SD 1.2 points).")
    assert w.stand_higher("94", "89", "99", 18, "90.12") == (
        "Predicted best: 94 (95 % interval 89–99), near Formulation 18 (measured 90.12).")
    # Insights for a first-time user (review R-I5, M11; T0): Show its
    # amounts moves below, into Progress figures.
    assert w.next_untested(round_no=6) == (
        "The predicted best is in untested amounts (Show its amounts, under Progress figures). "
        "To test it directly, add it to Round 6 as your own formulation; suggestions can also "
        "land near it.")
    assert w.path_what(w.path_head_fill("Coconut oil", "Water"),
                       [("8.04 g", "Fat per 100 g")], [("Salt", "1.5 g")]) == (
        "More Coconut oil in place of Water; stops at 8.04 g (Fat per 100 g). "
        "Salt fixed at 1.5 g.")
    assert w.path_key(15, grey=True, targets=True) == (
        "Band: 95 % interval. Grey: not tested near here. Dotted: Formulation 15; dashed: "
        "target.")
    assert w.details_window([2, 3, 4, 5], 3, [3, 4, 5]) == (
        "Table: all Rounds checked (Rounds 2–5). Verdict: the last 3 (Rounds 3–5).")
    assert w.overall_score_line(90.45, 100) == "Overall score 90.45 of 100."
    # Concision round (2026-10-01): short, the threshold kept. Fix A
    # follow-up (ruling on ties): the q = 1 rise threshold, 1.2 × √2 × 2.35.
    # round 2: "tied" for equal values only; specific wording (2026-10-12),
    # re-pinned: the rise needed, beside the lead.
    assert w.tie_line(22, "90.65", "0.20", 15, "4.0") == (
        "Formulation 15 is within noise: 0.20 lower; 4.0 needed.")
    # Re-pinned (stale since the fourth tab became "4 · Review"): "in the
    # review", not "on Insights".
    assert w.never_coming_warning(19, "Cook loss") == (
        "Cook loss for Formulation 19 will never come? It stays in your records, unscored, "
        "and stops counting in the review.")
    assert w.EDIT_PAST_FOLD == "Correct, delete or add past formulations"
    assert w.NEXT_MAKE_A_ROUND == "Next: Make a round"
    assert w.START_FROM_BEST_SO_FAR == "Start from Best so far"


@pytest.mark.parametrize("pairs,text", [(1, "1 pair"), (2, "2 pairs"), (3, "3 pairs"),
                                        (5, "5 pairs")])
def test_pairs_one_and_many(pairs, text):
    assert wording.pairs_text(pairs) == text


def test_singular_and_plural_forms():
    """Spec review M7: one helper per form, each right for 1 and 2."""
    w = wording
    assert w.waiting_line(1, "Cook loss") == (
        "1 formulation is waiting for Cook loss (enter it on Results). The model has not used "
        "it yet, so a suggestion may nearly repeat it.")
    assert w.waiting_line(2, "Cook loss and Aroma", many_measurements=True) == (
        "2 formulations are waiting for Cook loss and Aroma (enter them on Results). The model "
        "has not used them yet, so a suggestion may nearly repeat them.")
    assert w.waiting_line(2, "") == (
        "2 formulations are waiting for results (enter them on Results). The model has not "
        "used them yet, so a suggestion may nearly repeat them.")
    assert "lower" in w.repeat_disagrees(21, 11, "4.0")
    assert "higher" in w.repeat_disagrees(21, 11, "4.0", lower=False)
    # Polish round 8: the bias as one fact, no reading clause.
    assert w.score_gap_line(6, 12, -5.2, w.beat_by("1.3")).endswith(
        "This exceeds the noise threshold (Repeat SD 1.3 points) and 2 standard errors: "
        "Overall scores have come in about 5.2 points below the predictions.")
    assert w.score_gap_line(6, 12, 2.0, w.beat_by()).endswith(
        "This exceeds the 1-point noise threshold and 2 standard errors: Overall scores "
        "have come in about 2.0 points above the predictions.")
    assert w.insights_header_line(0, None, 1, "Aroma") == "1 formulation is waiting for Aroma."
    assert w.insights_header_line(1, 1) == (
        "Based on 1 scored formulation (all results in), up to Round 1.")
    assert w.stand_since(1) == "No higher score in the Round since."
    assert w.stand_since(2) == "No higher score in the 2 Rounds since."
    assert w.trust_early(1) == "Not judged yet: 1 value checked; the verdict needs 10."
    assert w.waiting_list("Cook loss", [22], 6) == "Waiting for Cook loss: Formulation 22, Round 6."
    assert w.waiting_list("Cook loss", [21, 22]) == "Waiting for Cook loss: Formulations 21–22."
    assert w.more_label(True, True) == "More · Not scored"


def test_process_only_forms_name_settings_not_ingredients():
    """Spec review I14: a sentence naming ingredients or amounts has a form
    for a study of process settings alone."""
    w = wording
    assert "ingredient" not in w.try_one_fold(True, has_ingredients=False).lower()
    assert "values" in w.next_untested(values=True)
    assert "ingredient" not in w.next_close_to_top(settings_only=True)
    assert "ingredient" not in w.near_none(settings_only=True)
    assert "values" in w.near_bigger(values=True)
    assert "values" in w.change_caption(values=True)
    assert "ingredient" not in w.setup_first_line(settings_only=True)
    assert "process setting" in w.setup_first_line(settings_only=True)
    assert w.setup_first_line() == w.SETUP_FIRST_LINE
