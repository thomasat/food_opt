"""0.8.0 speed and simplicity, Plan B, Task 0: every new line of the wave
is spelled once in wording.py, passes the vocabulary guard, says "spread"
nowhere, and the strings a workbook import matches stay exactly as they
were (spec 2026-09-30 §§2–6, §9 rev. 2; spec review B1, I14, I17, M3, M7)."""
import re

import pytest

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
        w.try_one(True), w.try_one(False), w.try_one(True, False),
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
        w.tie_line(22, "90.65", "0.20", 15, "1.2"), w.tie_line(22, "90.65", "0.20", 15),
        w.MEASURED_WAITING, w.waiting_list("Cook loss", [22], 6),
        w.waiting_list("Cook loss", [21, 22]), w.MOVE_TO_NOT_SCORED_HELP,
        w.never_coming_warning(19, "Cook loss"), w.EDIT_PAST_FOLD,
        w.result_view_waiting("Formulation 22 (Round 6)", "Cook loss"),
        w.NEXT_MAKE_A_ROUND, w.SETUP_FIRST_LINE, w.setup_first_line(True),
        w.MEASUREMENT_GOAL_CAPTION,
        w.SAMPLE_INTRO_LINE, w.CHANGE_NEAR, w.CHANGE_WHOLE,
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
    w = wording
    assert w.insights_header_line(24, 6, 1, "Cook loss") == (
        "Based on 24 scored formulations, up to Round 6. 1 is waiting for Cook loss.")
    assert w.TRUST_WIDE == (
        "Too early to judge: the likely range is still wide, so almost any result lands inside.")
    assert w.trust_within(32, 33, [4, 5, 6]) == (
        "On track: 32 of 33 results from Rounds 4–6 landed inside the likely range given "
        "beforehand (the app aims for about 19 in 20).")
    assert w.score_gap_line(11, 11, -1.0) == (
        "Overall score: 11 of 11 inside; results ran about 1 point below prediction.")
    assert w.repeat_disagrees(21, 11, "10.6") == (
        "Formulation 21, a Repeat of Formulation 11, scored 10.6 points lower — more than its "
        "likely range allows. Check whether something changed (panel, supplier, equipment).")
    assert w.stand_best("90.65", 15, 4) == "Best so far: 90.65, Formulation 15 (Round 4)."
    assert w.stand_not_higher("Repeats differ (1.2 points)", "90", "86", "94", 18) == (
        "Nothing is predicted to beat it by more than Repeats differ (1.2 points): the "
        "predicted best is 90 (likely 86–94), close to Formulation 18.")
    assert w.stand_higher("94", "89", "99", 18, "90.12") == (
        "The predicted best is 94 (likely 89–99), close to Formulation 18 (90.12).")
    assert w.next_untested() == (
        "The app predicts a higher score in amounts not tested yet (see “Show its amounts” "
        "above). Suggestions may try near there; you can also add it to the Round yourself.")
    assert w.path_what(w.path_head_fill("Coconut oil", "Water"),
                       [("8.04 g", "Fat per 100 g")], [("Salt", "1.5 g")]) == (
        "More Coconut oil in place of Water; stops at 8.04 g (Fat per 100 g). "
        "Salt fixed at 1.5 g.")
    assert w.path_key(15, grey=True, targets=True) == (
        "Grey: not tested near here. Dotted: Formulation 15; dashed: target.")
    assert w.details_window([2, 3, 4, 5], 3) == (
        "All Rounds checked (Rounds 2–5); the verdict uses the last 3.")
    assert w.overall_score_line(90.45, 100) == "Overall score 90.45 of 100."
    assert w.tie_line(22, "90.65", "0.20", 15, "1.2") == (
        "Formulation 22 scored 90.65, 0.20 above Formulation 15 — within how much Repeats "
        "differ (1.2), so treat them as equal.")
    assert w.never_coming_warning(19, "Cook loss") == (
        "Cook loss for Formulation 19 will never come? It stays in your records, unscored, "
        "and stops counting on Insights.")
    assert w.EDIT_PAST_FOLD == "Correct, delete or add past formulations"
    assert w.NEXT_MAKE_A_ROUND == "Next: Make a round"
    assert w.START_FROM_BEST_SO_FAR == "Start from Best so far"


@pytest.mark.parametrize("pairs,text", [(1, "1 pair, too few to be sure"),
                                        (2, "2 pairs, too few to be sure"),
                                        (3, "3 pairs"), (5, "5 pairs")])
def test_pairs_one_and_many(pairs, text):
    assert wording.pairs_text(pairs) == text


def test_singular_and_plural_forms():
    """Spec review M7: one helper per form, each right for 1 and 2."""
    w = wording
    assert w.waiting_line(1, "Cook loss") == (
        "1 formulation is waiting for Cook loss (enter it on Results); suggestions may land "
        "close to it.")
    assert w.waiting_line(2, "Cook loss and Aroma", many_measurements=True) == (
        "2 formulations are waiting for Cook loss and Aroma (enter them on Results); "
        "suggestions may land close to them.")
    assert w.waiting_line(2, "") == (
        "2 formulations are waiting for results (enter them on Results); suggestions may "
        "land close to them.")
    assert "lower" in w.repeat_disagrees(21, 11, "4.0")
    assert "higher" in w.repeat_disagrees(21, 11, "4.0", lower=False)
    assert w.score_gap_line(6, 12, -5.2, "Repeats differ").endswith(
        "below prediction — more than Repeats differ, so read new predictions as about "
        "5.2 points too high.")
    assert w.score_gap_line(6, 12, 2.0, "1 point").endswith(
        "above prediction — more than 1 point, so read new predictions as about 2.0 points "
        "too low.")
    assert w.insights_header_line(0, None, 1, "Aroma") == "1 formulation is waiting for Aroma."
    assert w.insights_header_line(1, 1) == "Based on 1 scored formulation, up to Round 1."
    assert w.stand_since(1) == "No higher score in the Round since."
    assert w.stand_since(2) == "No higher score in the 2 Rounds since."
    assert w.trust_early(1) == "Too early to judge: 1 result checked so far."
    assert w.waiting_list("Cook loss", [22], 6) == "Waiting for Cook loss: Formulation 22 (Round 6)."
    assert w.waiting_list("Cook loss", [21, 22]) == "Waiting for Cook loss: Formulations 21–22."
    assert w.more_label(True, True) == "More · Not scored"


def test_process_only_forms_name_settings_not_ingredients():
    """Spec review I14: a sentence naming ingredients or amounts has a form
    for a study of process settings alone."""
    w = wording
    assert "ingredient" not in w.try_one(True, has_ingredients=False).lower()
    assert "values" in w.next_untested(values=True)
    assert "ingredient" not in w.next_close_to_top(settings_only=True)
    assert "ingredient" not in w.near_none(settings_only=True)
    assert "values" in w.near_bigger(values=True)
    assert "values" in w.change_caption(values=True)
    assert "ingredient" not in w.setup_first_line(settings_only=True)
    assert "process setting" in w.setup_first_line(settings_only=True)
    assert w.setup_first_line() == w.SETUP_FIRST_LINE
