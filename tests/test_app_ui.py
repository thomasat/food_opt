"""UI-level regression tests for app.py, using Streamlit's AppTest.

These drive the real Streamlit script headlessly, so they catch the class of
bug where the backend raises a clean ValueError but the UI fails to catch it
and shows the user a raw traceback.
"""

import os

import pytest
from streamlit.testing.v1 import AppTest

from food_bo import FoodOptimizer

APP_PATH = os.path.join(os.path.dirname(__file__), "..", "app.py")


def _submit_button(at, label):
    return next(b for b in at.button if b.label == label)


@pytest.fixture
def project_with_history(tmp_path, monkeypatch):
    """A project on disk (in a temp cwd) that already has one experiment,
    so mid-run rules apply. Named my_project: the app's default."""
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("my_project")
    opt.add_ingredient("Water", 0, 100)
    opt.add_objective("Taste", 1.0, goal="max")
    opt.tell({"Water": 50.0}, {"Taste": 7.0})
    return opt


def test_midrun_process_param_error_is_shown_not_raised(project_with_history):
    """Adding a process parameter mid-run with a bad baseline must surface as
    st.error text, never as an uncaught exception (raw traceback)."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert not at.exception

    at.text_input(key="pp_name").set_value("Oven_Temp")
    at.number_input(key="pp_min").set_value(150.0)
    at.number_input(key="pp_max").set_value(220.0)
    at.number_input(key="pp_base").set_value(100.0)  # outside [150, 220]
    _submit_button(at, "Add Process Parameter").click()
    at.run()

    assert not at.exception
    assert any("must lie within" in str(e.value) for e in at.error), \
        [str(e.value) for e in at.error]


def test_midrun_process_param_with_valid_baseline_succeeds(project_with_history):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert not at.exception

    at.text_input(key="pp_name").set_value("Oven_Temp")
    at.number_input(key="pp_min").set_value(150.0)
    at.number_input(key="pp_max").set_value(220.0)
    at.number_input(key="pp_base").set_value(180.0)
    _submit_button(at, "Add Process Parameter").click()
    at.run()

    assert not at.exception
    assert any("Added process parameter" in str(s.value) for s in at.success)
    # The variable really landed, with history re-encoded at the baseline.
    reloaded = FoodOptimizer("my_project")
    assert any(v["name"] == "Oven_Temp" for v in reloaded.variables)
    assert reloaded.X_history[0][1] == 180.0


def test_pending_batch_is_restored_in_a_new_session(project_with_history):
    """The recipes on the bench must survive closing the window."""
    project_with_history.set_pending_batch([{"Water": 10.0}, {"Water": 20.0}])
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert not at.exception
    assert "current_batch" in at.session_state
    assert len(at.session_state["current_batch"]) == 2
    assert any(b.label == "Save Results" for b in at.button), [b.label for b in at.button]


def test_stale_pending_batch_is_discarded_when_design_space_changes(project_with_history):
    project_with_history.set_pending_batch([{"Water": 10.0, "Ghost": 1.0}])
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert not at.exception
    assert "current_batch" not in at.session_state
    assert any("design space changed" in i.value for i in at.info), [i.value for i in at.info]


def test_delete_experiment_confirmation_is_visible_after_rerun(project_with_history):
    """The success message must survive the st.rerun() that follows a delete."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    _submit_button(at, "Delete Experiment #0").click()   # label becomes 1-based in Task 7
    at.run()
    assert not at.exception
    assert any("Deleted experiment" in s.value for s in at.success), \
        [s.value for s in at.success]


def test_hard_reset_targets_active_project_not_typed_name(project_with_history, tmp_path):
    """Typing another name in the sidebar without clicking Create must not
    redirect Hard Reset at that other project (audit: confirmed critical bug)."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.sidebar.text_input[0].set_value("other")   # the sidebar "Project Name" box (only text input in sidebar)
    _submit_button(at, "Hard Reset Project").click()
    at.run()
    _submit_button(at, "Yes, reset").click()
    at.run()
    assert not at.exception
    assert (tmp_path / "my_project_archived.pkl").exists()
    assert not (tmp_path / "other.pkl").exists()
    assert not (tmp_path / "other_archived.pkl").exists()
    assert any("my_project" in c.value and "0 experiments" in c.value for c in at.caption), \
        [c.value for c in at.caption]


def test_restore_requires_confirmation_and_archives_current(project_with_history, tmp_path):
    """Restore must preview, confirm, and archive the current project first.
    We can't drive st.file_uploader in AppTest, so we prime the parsed state
    the way the uploader handler does."""
    other = FoodOptimizer("donor")
    other.add_ingredient("Flour", 0, 100)
    other.add_objective("Crunch", 1.0, goal="max")
    for i in range(3):
        other.tell({"Flour": 10.0 * i}, {"Crunch": 5.0})
    donor_state = other.export_json()

    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_restore_candidate"] = donor_state
    at.run()
    assert any("3 experiments" in w.value for w in at.warning), [w.value for w in at.warning]
    assert len(FoodOptimizer("my_project").X_history) == 1   # nothing changed yet
    _submit_button(at, "Yes, replace").click()
    at.run()
    assert not at.exception
    assert (tmp_path / "my_project_pre_restore.pkl").exists()
    assert len(FoodOptimizer("my_project").X_history) == 3
    assert any("Restored 3 experiments" in s.value for s in at.success)


def test_restore_rejects_empty_json(project_with_history):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_restore_candidate"] = {}
    at.run()
    assert any("not a Food Optimizer backup" in e.value for e in at.error), \
        [e.value for e in at.error]
    assert len(FoodOptimizer("my_project").X_history) == 1


@pytest.fixture
def project_with_pending_batch(project_with_history):
    project_with_history.set_pending_batch([{"Water": 10.0}, {"Water": 20.0}])
    return project_with_history


def test_blank_result_is_refused_not_saved_as_zero(project_with_pending_batch):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.number_input(key="b0_r0o0").set_value(7.0)   # Recipe 1 only
    _submit_button(at, "Save Results").click()
    at.run()
    assert not at.exception
    assert any("Recipe 2" in e.value for e in at.error), [e.value for e in at.error]
    assert len(FoodOptimizer("my_project").X_history) == 1   # nothing saved


def test_skipped_recipe_is_left_out(project_with_pending_batch):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.number_input(key="b0_r0o0").set_value(7.0)
    at.checkbox(key="b0_skip1").check()
    _submit_button(at, "Save Results").click()
    at.run()
    assert not at.exception
    opt = FoodOptimizer("my_project")
    assert len(opt.X_history) == 2
    assert opt.pending_batch is None
    assert any("Saved 1 result" in s.value for s in at.success), [s.value for s in at.success]
