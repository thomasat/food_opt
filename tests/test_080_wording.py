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
        w.TRUST_BEFOREHAND,
        w.MVP_CAPTION, w.Q_STAND, 
        w.STAND_LATEST, w.stand_since(1),
        w.stand_since(2), w.stand_best("90.65", 15, 4), w.stand_best("90.65", 15, None, "Aroma"),
        w.stand_limits([
                        ]),
        w.NEXT_RISING,
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
        w.START_FROM_BEST_SO_FAR, w.HOW_CHOSEN_2, w.still_covering(5),
        w.waiting_line(1, "Cook loss"), w.waiting_line(2, "Cook loss and Aroma", True),
        w.waiting_line(3, ""), w.WORKBOOK_TABS_HELP,
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


@pytest.mark.parametrize("pairs,text", [(1, "1 pair"), (2, "2 pairs"), (3, "3 pairs"),
                                        (5, "5 pairs")])
def test_pairs_one_and_many(pairs, text):
    assert wording.pairs_text(pairs) == text


