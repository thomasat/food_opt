"""0.8.0 speed and simplicity, Plan B, Task 4: Set up's foot says where it
goes, a new project's first step is said and lit, one-line captions, and
the sidebar keeps Saved copies and the sample inside Manage project (spec
2026-09-30 §5; review M16; audit V4, V5, V18–V21)."""
import json

import pytest
from streamlit.testing.v1 import AppTest

import wording
from food_bo import FoodOptimizer
from tests.test_app_ui import APP_PATH, burger, open_batch, scored  # noqa: F401


def _app(project=None, tab=None):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    if project:
        at.session_state["_loaded_project"] = project
    at.run()
    if tab:
        at.session_state["main_tab"] = tab
        at.run()
    return at


def _setup_buttons(at):
    return [b for b in at.tabs[0].button]


# ------------------------------------------------------------------ #
#  The foot (V5, review M16)
# ------------------------------------------------------------------ #

def test_with_no_round_open_the_foot_says_next_make_a_round(burger):
    at = _app(tab=wording.TAB_SETUP)
    foot = next(b for b in at.tabs[0].button if b.key == "continue_to_batch")
    assert foot.label == wording.NEXT_MAKE_A_ROUND == "Next: Make a round"
    assert foot.proto.type == "primary"
    foot.click()
    at.run()
    assert at.session_state["main_tab"] == wording.TAB_BATCH


def test_with_a_round_open_the_foot_says_back_to_it(open_batch):
    at = _app(tab=wording.TAB_SETUP)
    foot = next(b for b in at.tabs[0].button if b.key == "continue_to_batch")
    # 9412f30: at most 5 words; the count to record is on Make a round.
    assert foot.label == wording.back_to_batch_label(1, 2) == "Back to Round 1"
    assert "Generate formulations" not in [b.label for b in at.tabs[0].button]


# ------------------------------------------------------------------ #
#  A new project (V4)
# ------------------------------------------------------------------ #

def test_a_new_project_says_its_first_step_and_lights_import(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    FoodOptimizer("empty").save()
    at = _app("empty", wording.TAB_SETUP)
    assert not at.exception
    assert at.tabs[0].caption[0].value == wording.SETUP_FIRST_LINE
    assert wording.SETUP_INTRO not in [c.value for c in at.tabs[0].caption]
    fold = next(p for p in at.get("popover")
                if p.proto.popover.label == wording.UPLOAD_INGREDIENTS_EXPANDER)
    assert fold.proto.popover.type == "primary"
    lit = [b.label for b in at.tabs[0].button if b.proto.type == "primary"]
    assert lit == []                             # the foot is grey: nothing to vary yet


def test_import_goes_grey_once_there_is_a_row(burger):
    at = _app(tab=wording.TAB_SETUP)
    fold = next(p for p in at.get("popover")
                if p.proto.popover.label == wording.UPLOAD_INGREDIENTS_EXPANDER)
    assert fold.proto.popover.type == "secondary"
    assert wording.SETUP_FIRST_LINE not in [c.value for c in at.tabs[0].caption]


# ------------------------------------------------------------------ #
#  One-line captions (V18–V20)
# ------------------------------------------------------------------ #

def test_the_measurements_caption_is_one_line_and_the_formula_moved(burger):
    at = _app(tab=wording.TAB_SETUP)
    captions = [c.value for c in at.tabs[0].caption]
    assert wording.MEASUREMENT_GOAL_CAPTION in captions
    # The old four-sentence caption, by its text: Task 9 deletes the name.
    assert not any(c.startswith("Set a goal for each measurement. Share of score")
                   for c in captions)
    formula = FoodOptimizer("burger").score_function_line()      # as the app loads it
    scoring = next(p for p in at.get("popover")
                   if p.proto.popover.label == wording.SCORING_DETAILS_CHECKBOX)
    assert formula in [c.value for c in scoring.caption]
    under_grid = [c.value for c in at.tabs[0].caption
                  if c not in list(scoring.caption)]
    assert formula not in under_grid


def test_the_calculation_line_is_said_once(burger):
    at = _app(tab=wording.TAB_SETUP)
    captions = [c.value for c in at.tabs[0].caption]
    assert not any(c.startswith("To calculate a row from the others") for c in captions)
    assert captions.count(wording.CALCULATION_EDIT_HINT) == 1


def test_the_sample_is_named_in_the_sidebar_and_setup_stays_quiet(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    at = _app()
    next(b for b in at.main.button if b.label == wording.OPEN_YOGURT_LABEL).click()
    at.run()
    at.session_state["main_tab"] = wording.TAB_SETUP
    at.run()
    assert not at.exception
    captions = [c.value for c in at.tabs[0].caption]
    assert wording.SAMPLE_PROVENANCE_CAPTION in [c.value for c in at.sidebar.caption]
    assert not any(c.startswith("Learn the workflow") for c in captions)
    assert not any(c.startswith("Teaching example") for c in captions)


# ------------------------------------------------------------------ #
#  The sidebar (V21)
# ------------------------------------------------------------------ #

def test_saved_copies_and_the_sample_are_inside_manage_project(scored):
    at = _app()
    assert not at.exception
    [manage] = [e for e in at.sidebar.expander if e.label == wording.MANAGE_PROJECT]
    assert wording.SAVED_COPIES_HEADING in [m.value for m in manage.markdown]
    assert wording.OPEN_YOGURT_LABEL in [b.label for b in manage.button]
    outside = [m.value for m in at.sidebar.markdown if m not in list(manage.markdown)]
    assert wording.SAVED_COPIES_HEADING not in outside


def test_a_checked_copy_asks_its_question_at_once(scored, tmp_path):
    copy = json.dumps(scored.export_json()).encode()
    at = _app()
    at.sidebar.get("file_uploader")  # the uploader is drawn inside the fold
    at.session_state["_restore_candidate"] = json.loads(copy)
    at.run()
    assert not at.exception
    assert wording.YES_REPLACE in [b.label for b in at.sidebar.button]
