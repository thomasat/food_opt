"""The predicted best on the owner-bench project (plan-a-report open
question 2). The prototype's bench-60x4 file had lost its default batch
size while Water stayed "= rest": the row then fills the rest of nothing,
its floor (Water at least 0 g) reads 0 - every other amount >= 0, no
amounts meet the limits, and the search raised BoTorch's
InfeasibilityError — answer 2 read "could not be worked out". ask() stops
at the same place: the limits are the project's, not the search's. The
display now says why in one line, and never touches the limits or ask()."""
import pytest
from botorch.exceptions.errors import InfeasibilityError
from streamlit.testing.v1 import AppTest

import insights
import wording
from tests import speed_projects
from tests.test_app_ui import APP_PATH
from tests.test_food_bo import _BANNED


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _score(opt):
    return {insights.SCORE: insights.fit(opt, insights.SCORE, list(range(len(opt.Y_history))))}


def test_the_owner_bench_project_has_a_predicted_best():
    opt = speed_projects.bench(60, 4)
    best = insights.predicted_best(opt, _score(opt))
    assert isinstance(best, insights.PredictedBest)
    assert opt._check_constraints(best.recipe)


def test_a_rest_row_without_a_default_batch_size_is_said_not_failed():
    """The prototype's file, rebuilt the way the app gets there: Water is
    "= rest" and the default batch size box is cleared."""
    opt = speed_projects.bench(60, 4)
    opt.clear_formulation_total()
    assert not opt.has_formulation_total()
    assert opt._balance_row()['name'] == "Water"
    before = [(i.tolist(), c.tolist(), r) for i, c, r in opt._get_botorch_constraints()]
    assert insights.predicted_best(opt, _score(opt)) == wording.best_needs_batch_size("Water")
    # The limits are read, never changed.
    assert [(i.tolist(), c.tolist(), r) for i, c, r in opt._get_botorch_constraints()] == before


def test_limits_no_amounts_meet_together_are_said_not_failed(monkeypatch):
    opt = speed_projects.bench(30, 3)
    score = _score(opt)

    def empty(**kwargs):
        raise InfeasibilityError("No feasible point found. Constraint polytope appears empty.")
    monkeypatch.setattr(insights, "optimize_acqf", empty)
    assert insights.predicted_best(opt, score) == wording.BEST_NO_AMOUNTS


def test_on_the_tab_answer_2_says_why(tmp_path):
    opt = speed_projects.bench(18, 3, name="no-batch-size")
    opt.clear_formulation_total()
    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.session_state["_loaded_project"] = opt.project_name
    at.run()
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    assert not at.exception
    captions = [c.value for c in at.tabs[3].caption]
    assert wording.best_needs_batch_size("Water") in captions
    assert wording.BEST_FAILED not in captions


def test_the_new_lines_pass_the_vocabulary_guard():
    lines = [wording.best_needs_batch_size("Water"), wording.BEST_NO_AMOUNTS]
    assert [t for t in lines if any(p.search(t) for p in _BANNED)] == []


def test_the_batch_size_line_says_fill_to_total_and_where_to_set_it():
    """Review C I-2: "= rest" is the stored spelling, never said out loud;
    the line names the setting and the tab it is on."""
    line = wording.best_needs_batch_size("Water")
    assert line == ("There is no predicted best: Water is set to Fill to total, and there is "
                    "no default batch size for it to fill. Set the default batch size on "
                    "Set up, under Preparation and records.")
    assert "= rest" not in line and "rest" not in line.split()
    assert "\n" not in line
