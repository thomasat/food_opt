"""Insights for a first-time user, T0 (spec 2026-10-01 rev. 2; review I17,
R-I17): the fixtures the first-screen pins need. The Snack bar joins the
five demo projects (copied unchanged from docs/usability/0.8.0-demo), a
study of process settings alone is added (built once by
tests/process_only_fixture.py, never regenerated), and the browser pins
have their 1280 × 800 screen."""
import pathlib

import insights
import wording
from tests import demo_projects as D


def test_the_seven_projects_load_as_saved():
    assert len(D.SIX) == 6 and D.ALL[-1] == D.PROCESS_ONLY
    for name in D.ALL:
        opt = D.load(name)
        assert opt.project_name == name


def test_the_snack_bar_sits_at_the_gate():
    opt = D.load(D.SNACK)
    assert opt.complete_count() == 3 < insights.MIN_ROWS
    assert set(opt.batch_history) == {1}


def test_the_process_only_project_is_a_study_of_settings_alone():
    opt = D.load(D.PROCESS_ONLY)
    assert not opt.has_ingredients() and len(opt._process_settings()) == 3
    assert wording.for_project(opt, wording.why_link(6)) == "Why Round 6 was suggested"
    assert wording.for_project(opt, "Formulation 15") == "Trial 15"
    # Line A's shape: Rounds of 5, 4, 4, 4 and 3, a Repeat of Best so far in
    # Round 5, Round 6 open with 4 generated picks.
    rounds = list(opt.batch_history)
    assert [rounds.count(r) for r in range(1, 6)] == [5, 4, 4, 4, 4]
    assert opt.notes_history[-1].startswith("Repeat of Formulation ")
    assert (opt.pending_batch_no, len(opt.pending_batch)) == (6, 4)
    record = opt.generation_records()[6]
    assert record["regime"] == "warm"
    assert record["picks"] == [int(row["formulation"]) for row in opt.pending_batch]
    assert opt.complete_count() == 21


def test_the_process_only_fixture_is_frozen_json():
    text = (D.HERE / f"{D.PROCESS_ONLY}.json").read_text()
    assert "NaN" not in text and "Infinity" not in text


def test_the_first_screen_viewport():
    """R-I17: the new pins open at 1280 × 800, not the suite's 1280 × 860.
    Read from the file: the helper module imports Playwright."""
    text = (pathlib.Path(__file__).parent / "browser" / "grid_helpers.py").read_text()
    assert 'FIRST_SCREEN = {"width": 1280, "height": 800}' in text
    assert "def above_the_fold(locator, viewport=FIRST_SCREEN):" in text
