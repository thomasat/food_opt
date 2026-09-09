"""UI-level regression tests for app.py, using Streamlit's AppTest.

These drive the real Streamlit script headlessly, so they catch the class of
bug where the backend raises a clean ValueError but the UI fails to catch it
and shows the user a raw traceback.
"""

import os
import time

import pandas as pd
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
    assert any("must be between" in str(e.value) for e in at.error), \
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


def test_remove_process_parameter_confirms_and_archives(project_with_history, tmp_path):
    """Removing a process parameter mid-run must require confirmation and
    archive a copy first, then re-encode history (not just drop the column)."""
    project_with_history.add_process_parameter("Oven_Temp", 150, 220, baseline=180)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert not at.exception
    _submit_button(at, "Remove").click()
    at.run()
    assert not (tmp_path / "my_project_pre_delete.pkl").exists()   # not yet
    _submit_button(at, "Yes, remove").click()
    at.run()
    assert not at.exception
    assert (tmp_path / "my_project_pre_delete.pkl").exists()
    reloaded = FoodOptimizer("my_project")
    assert not any(v["name"] == "Oven_Temp" for v in reloaded.variables)
    assert all(len(x) == len(reloaded.variables) for x in reloaded.X_history)
    assert any("Removed Oven_Temp" in s.value for s in at.success)


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
    _submit_button(at, "Delete experiment 1").click()
    at.run()
    _submit_button(at, "Yes, delete").click()
    at.run()
    assert not at.exception
    assert any("Deleted experiment" in s.value for s in at.success), \
        [s.value for s in at.success]


def test_delete_experiment_confirms_and_archives(project_with_history, tmp_path):
    """Deleting an experiment must require confirmation and archive a copy first."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    _submit_button(at, "Delete experiment 1").click()
    at.run()
    assert len(FoodOptimizer("my_project").X_history) == 1   # not yet
    assert any("Delete experiment 1" in w.value for w in at.warning)
    _submit_button(at, "Yes, delete").click()
    at.run()
    assert not at.exception
    assert len(FoodOptimizer("my_project").X_history) == 0
    assert (tmp_path / "my_project_pre_delete.pkl").exists()
    assert any("Deleted experiment 1" in s.value for s in at.success)


def test_history_is_1_based(project_with_history):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert any("Experiment number to edit" == n.label for n in at.number_input)
    assert not any("0-based" in n.label for n in at.number_input)


def test_edit_form_keeps_old_value_and_refuses_blank_with_no_history(project_with_history):
    """A blank measurement in the edit form keeps the experiment's existing
    value; a measurement the experiment never had is refused, not saved as 0."""
    project_with_history.add_objective("Crunch", 1.0, goal="max", min_val=0, max_val=10)
    project_with_history.tell({"Water": 60.0}, {"Taste": 8.0, "Crunch": 5.0})

    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.number_input(key="edit_no").set_value(1)   # experiment 1, recorded before Crunch existed
    at.run()
    _submit_button(at, "Update Result").click()
    at.run()
    assert not at.exception
    assert any("Crunch" in e.value for e in at.error), [e.value for e in at.error]
    assert FoodOptimizer("my_project").results_history[0] == {"Taste": 7.0}

    at.number_input(key="edit_0_1").set_value(4.0)  # fill Crunch, leave Taste blank
    at.run()
    _submit_button(at, "Update Result").click()
    at.run()
    assert not at.exception
    assert any("Updated experiment 1" in s.value for s in at.success)
    assert FoodOptimizer("my_project").results_history[0] == {"Taste": 7.0, "Crunch": 4.0}


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
    at.session_state["_loaded_project"] = "my_project"  # pin: "donor" now exists on
    # disk with a later mtime, which would otherwise become the auto-loaded
    # "most recent" project (Task 8) and defeat this restore-flow test.
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


def test_restore_rejects_wrong_shaped_variables(project_with_history):
    """validate_state must catch a backup whose 'variables' list is the right
    type but contains elements of the wrong shape, before anything is
    applied to the live optimizer."""
    candidate = project_with_history.export_json()
    candidate["variables"] = [{"nope": 1}]
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "my_project"
    at.session_state["_restore_candidate"] = candidate
    at.run()
    assert not at.exception
    assert any("wrong shape" in e.value for e in at.error), [e.value for e in at.error]
    assert len(FoodOptimizer("my_project").X_history) == 1
    assert at.session_state["optimizer"].variables[0]["name"] == "Water"


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


def test_recipe_amounts_render_as_tables_not_long_lines(project_with_pending_batch):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert not at.exception
    captions = [c.value for c in at.caption]
    assert any("Water 10.00" in c for c in captions), captions
    assert not any(" · " in c and "Water" in c for c in captions), captions
    assert len(at.table) >= 1


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
    assert any("Results saved" in s.value for s in at.success), [s.value for s in at.success]


def test_results_card_and_best_panel_after_save(project_with_pending_batch):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.number_input(key="b0_r0o0").set_value(9.0)
    at.number_input(key="b0_r1o0").set_value(2.0)
    _submit_button(at, "Save Results").click()
    at.run()
    assert not at.exception
    texts = [s.value for s in at.success] + [m.value for m in at.markdown] + [c.value for c in at.caption]
    assert any("Recipe 1" in t and "0.900" in t for t in texts), texts
    assert any("new best" in t.lower() for t in texts), texts
    assert any(m.label == "Best Overall Score" for m in at.metric)


def test_no_backup_nag_after_successful_save(project_with_pending_batch):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.number_input(key="b0_r0o0").set_value(7.0)
    at.number_input(key="b0_r1o0").set_value(6.0)
    _submit_button(at, "Save Results").click()
    at.run()
    assert not any("Download a backup now" in w.value for w in at.warning)
    assert any(c.value.startswith("Saved ") for c in at.caption), [c.value for c in at.caption]


def test_upload_results_for_batch(project_with_pending_batch):
    """Prime the parsed sheet the way the uploader handler does (AppTest cannot
    drive st.file_uploader). Recipe 2 is absent, so it must stay pending."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_results_upload"] = pd.DataFrame({"Recipe": [1], "Taste": [8.0]})
    at.run()
    _submit_button(at, "Save uploaded results").click()
    at.run()
    assert not at.exception
    opt = FoodOptimizer("my_project")
    assert len(opt.X_history) == 2
    assert opt.pending_batch == [{"Water": 20.0}]
    assert any("Results saved" in s.value for s in at.success), [s.value for s in at.success]


def test_stale_results_upload_is_cleared_on_hard_reset(project_with_pending_batch, tmp_path):
    """A parsed sheet primed for one project/batch must not leak into the next
    project via a Hard Reset (session state is otherwise long-lived)."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_results_upload"] = pd.DataFrame({"Recipe": [1], "Taste": [8.0]})
    at.run()
    assert "_results_upload" in at.session_state
    _submit_button(at, "Hard Reset Project").click()
    at.run()
    _submit_button(at, "Yes, reset").click()
    at.run()
    assert not at.exception
    assert "_results_upload" not in at.session_state


def test_save_failure_shows_one_banner_with_backup_and_reload(project_with_history):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.session_state["optimizer"].save_error = "The server could not be reached — your last change was NOT saved."
    at.run()
    assert not at.exception
    assert sum(1 for w in at.warning if "not saved" in w.value) == 1, [w.value for w in at.warning]
    labels = [b.label for b in at.button]
    assert "Reload project" in labels, labels


def test_first_run_creates_no_project_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert not at.exception
    assert list(tmp_path.glob("*.pkl")) == []
    assert any("Create your first project" in m.value for m in at.markdown), \
        [m.value for m in at.markdown]
    # the welcome panel already offers the sample; the sidebar must not duplicate it on a first run
    assert not any(b.label == "Try the sample project" for b in at.sidebar.button), [b.label for b in at.sidebar.button]


def test_create_project_rejects_bad_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.text_input(key="new_project_name").set_value("bad/name")
    _submit_button(at, "Create project").click()
    at.run()
    assert any("letters, numbers" in e.value for e in at.error)
    assert list(tmp_path.glob("*.pkl")) == []


def test_create_project_writes_file_and_opens_it(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.text_input(key="new_project_name").set_value("Oat cookie v2")
    _submit_button(at, "Create project").click()
    at.run()
    assert (tmp_path / "Oat cookie v2.pkl").exists()
    assert any("Created Oat cookie v2" in s.value for s in at.success)


def test_returning_user_lands_in_most_recent_project(project_with_history):
    older = FoodOptimizer("older")
    older.add_ingredient("Sugar", 0, 50)
    time.sleep(0.01)
    project_with_history.save()   # touch my_project so it is newest on disk
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert any("my_project" in c.value and "1 experiments" in c.value for c in at.caption)
    assert not any("older" in c.value for c in at.caption)


def test_pending_confirm_is_cleared_on_project_switch(project_with_history, tmp_path):
    other = FoodOptimizer("second")
    other.add_ingredient("Flour", 0, 100)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "my_project"  # pin: "second" now exists on
    # disk with a later mtime, which would otherwise become the auto-loaded "most
    # recent" project (Task 8) before this test ever switches to it on purpose.
    at.run()
    _submit_button(at, "Hard Reset Project").click()   # arm the confirmation
    at.run()
    assert any(b.label == "Yes, reset" for b in at.button)
    # Choosing a project reruns the script and enables Open; only then can a
    # user click it (Open is greyed out while the box shows the open project).
    at.selectbox(key="project_select").set_value("second")
    at.run()
    _submit_button(at, "Open").click()
    at.run()
    assert not at.exception
    assert not any(b.label == "Yes, reset" for b in at.button), [b.label for b in at.button]
    assert any("second" in c.value for c in at.caption)


def test_open_button_lights_up_only_when_it_would_switch(project_with_history, tmp_path):
    """Button rule: the button that moves you forward is coloured, and a
    button that would do nothing is greyed out. Open must be disabled while
    the box shows the project already on screen, and primary once a
    different project is chosen."""
    FoodOptimizer("second").add_ingredient("Flour", 0, 100)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "my_project"
    at.run()
    open_btn = _submit_button(at.sidebar, "Open")
    assert open_btn.disabled and open_btn.proto.type == "secondary"

    at.selectbox(key="project_select").set_value("second")
    at.run()
    open_btn = _submit_button(at.sidebar, "Open")
    assert not open_btn.disabled and open_btn.proto.type == "primary"

    open_btn.click()
    at.run()
    assert any("second" in c.value for c in at.caption)
    open_btn = _submit_button(at.sidebar, "Open")
    assert open_btn.disabled


def test_delete_project_confirms_archives_and_leaves_the_list(project_with_history, tmp_path):
    """Delete project must ask first, rename the file to an archive copy
    rather than erase it, drop the project from the list, and land on the
    welcome panel when nothing is left to open."""
    from storage import LocalStorage
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    _submit_button(at.sidebar, "Delete project").click()
    at.run()
    assert (tmp_path / "my_project.pkl").exists()               # not yet
    assert any("my_project" in w.value for w in at.warning)
    _submit_button(at.sidebar, "Yes, delete project").click()
    at.run()
    assert not at.exception
    assert not (tmp_path / "my_project.pkl").exists()
    assert (tmp_path / "my_project_deleted.pkl").exists()
    assert LocalStorage().list_projects() == []
    assert "my_project_deleted" in LocalStorage().list_archives()
    assert any("Deleted my_project" in i.value for i in at.info), [i.value for i in at.info]
    assert any("Create your first project" in m.value for m in at.markdown)


def test_delete_project_opens_the_most_recent_remaining_project(project_with_history, tmp_path):
    other = FoodOptimizer("second")
    other.add_ingredient("Flour", 0, 100)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "my_project"
    at.run()
    _submit_button(at.sidebar, "Delete project").click()
    at.run()
    _submit_button(at.sidebar, "Yes, delete project").click()
    at.run()
    assert not at.exception
    assert any("second" in c.value for c in at.caption), [c.value for c in at.caption]
    assert at.selectbox(key="project_select").options == ["second"]


def test_lower_sidebar_heading_names_the_open_project_and_follows_switches(project_with_history):
    """Backup, restore, reset and delete act on the open project, so the
    heading above them carries its name and follows Open, not the box."""
    FoodOptimizer("second").add_ingredient("Flour", 0, 100)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "my_project"
    at.run()
    assert "my_project" in [h.value for h in at.sidebar.subheader]

    at.selectbox(key="project_select").set_value("second")   # chosen, not opened
    at.run()
    assert "my_project" in [h.value for h in at.sidebar.subheader]
    assert "second" not in [h.value for h in at.sidebar.subheader]

    _submit_button(at.sidebar, "Open").click()
    at.run()
    assert "second" in [h.value for h in at.sidebar.subheader]
    assert "my_project" not in [h.value for h in at.sidebar.subheader]


def test_csv_instructions_are_short_and_name_the_real_columns(project_with_history):
    """Standing instructions are captions, not info boxes; the import caption
    lists this project's own column names rather than a made-up example."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert not any("Upload a CSV" in i.value for i in at.info), [i.value for i in at.info]
    captions = [c.value for c in at.caption]
    assert any(c.startswith("One row per ingredient") for c in captions), captions
    imp = next(c for c in captions if c.startswith("One row per past experiment"))
    assert "Water" in imp and "Taste" in imp
    assert "flour, sugar, butter" not in imp


def test_forward_buttons_are_primary(project_with_history):
    """The step that advances the project is coloured; housekeeping is grey."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    types = {b.label: b.proto.type for b in at.button}
    for label in ("Create project", "Add ingredient", "Add or update objective"):
        assert types[label] == "primary", (label, types[label])
    for label in ("Remove Taste", "Add Quantity Constraint", "Add Total Mass Constraint"):
        assert types[label] == "secondary", (label, types[label])


def test_generate_is_disabled_with_reason_when_no_objectives(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("noobj")
    opt.add_ingredient("Water", 0, 100)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    gen = next(b for b in at.button if b.label.startswith("Generate"))
    assert gen.disabled
    assert any("Add at least one objective" in c.value for c in at.caption), [c.value for c in at.caption]


def test_exploration_phase_notice(project_with_history):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert any("Exploration phase" in i.value for i in at.info), [i.value for i in at.info]


def test_tab_and_section_labels_are_plain(project_with_history):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    labels = [t.label for t in at.tabs]
    assert labels == ["1. Set up your project", "2. Run experiments"], labels
    headers = [h.value for h in at.subheader]
    assert not any(h.startswith(("A.", "A2.", "B.", "C.", "D.", "E.")) for h in headers), headers
    assert not any("EGBO" in x for x in headers + [e.label for e in at.expander]), headers


def _render_order(at):
    """Flat list of (type, label-or-value) in render order, walking nested blocks.

    AppTest exposes a block's nested children as a dict (index -> element) in
    Streamlit 1.55.0, not a list — confirmed via `dir(at.main)` (has
    "children") and `at.main.children` (a dict). We iterate `.values()`.

    A handful of Element subclasses raise from their `.value` property
    instead of just not having one — e.g. UnknownElement (used for
    st.line_chart) raises KeyError when its widget id has no session_state
    entry yet, and Dataframe's `.value` is a DataFrame whose truthiness is
    ambiguous. We don't care about those values for this test, so we prefer
    `.label` (present on Expander, Button, etc.) and only fall back to a
    try/except'd `.value` otherwise, never testing its truthiness.
    """
    out = []

    def walk(node):
        children = getattr(node, "children", None) or {}
        if hasattr(children, "values"):
            children = children.values()
        for child in children:
            label = getattr(child, "label", None)
            if label is not None:
                v = label
            else:
                try:
                    v = child.value
                except Exception:
                    v = ""
            out.append((type(child).__name__, v))
            walk(child)

    walk(at.main)
    return out


def test_history_appears_before_advanced_expanders(project_with_history):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    kinds = _render_order(at)

    def pos(pred):
        return next(i for i, k in enumerate(kinds) if pred(k))

    # Some element values are DataFrames (st.dataframe) whose truthiness/
    # equality is ambiguous, so only compare when the value is a string.
    history = pos(lambda k: isinstance(k[1], str) and k[1] == "Experiment History")
    adaptive = pos(lambda k: isinstance(k[1], str) and "Change the ingredient list" in k[1])
    imp = pos(lambda k: isinstance(k[1], str) and "Import Historical" in k[1])
    assert history < imp < adaptive, kinds


def test_sample_project_button_creates_ready_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    # The sidebar now also offers "Try the sample project", so target the
    # welcome-panel button explicitly (scoped to at.main) to keep this test
    # about that specific first-run flow.
    _submit_button(at.main, "Try the sample project").click()
    at.run()
    assert not at.exception
    opt = FoodOptimizer("Sample project")
    # Eight ingredients: enough for a real recipe, few enough to read at a
    # glance in the results form and the recipe cards.
    assert len(opt.variables) == 8
    # The sample is the plant-based burger brief for a trained panel: two
    # intensity scores with targets (7 and 6 out of 10), firmness weighted
    # more than juiciness.
    assert [o["name"] for o in opt.objectives] == ["Juiciness", "Firmness"]
    assert all(o["goal"] == "target" for o in opt.objectives)
    targets = {o["name"]: o["target"] for o in opt.objectives}
    assert targets == {"Juiciness": 7, "Firmness": 6}
    weights = {o["name"]: o["weight"] for o in opt.objectives}
    assert weights["Firmness"] > weights["Juiciness"]


def test_manual_add_ingredient_in_setup(project_with_history):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.text_input(key="ing_name").set_value("Honey")
    at.number_input(key="ing_min").set_value(0.0)
    at.number_input(key="ing_max").set_value(30.0)
    _submit_button(at, "Add ingredient").click()
    at.run()
    assert any(v["name"] == "Honey" for v in FoodOptimizer("my_project").variables)


def test_manual_add_ingredient_shows_mid_run_min_notice(project_with_history):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert not at.exception
    assert at.number_input(key="ing_min").disabled is True
    assert any("fixed at 0" in c.value for c in at.caption), [c.value for c in at.caption]


def test_sample_project_button_reopens_existing_without_recreating(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Pre-create "Sample project" with an experiment, simulating a shared host
    # where a previous visitor already ran the sample project.
    pre = FoodOptimizer("Sample project")
    pre.add_ingredient("Water", 0, 100)
    pre.add_objective("Taste", 1.0, goal="max")
    pre.tell({"Water": 50.0}, {"Taste": 7.0})

    # app.py never auto-opens the most-recently-used project on a shared host
    # (os.path.isdir("/mount/src")); force that guard so the welcome panel
    # renders here even though "Sample project" already exists on disk, which
    # is exactly the scenario where clicking the sample button again must
    # reopen the existing project instead of re-creating it.
    real_isdir = os.path.isdir
    monkeypatch.setattr(
        os.path, "isdir",
        lambda p: True if p == "/mount/src" else real_isdir(p),
    )

    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert not at.exception
    assert any("Create your first project" in m.value for m in at.markdown)

    # Scope to the welcome-panel button explicitly: the sidebar now also has
    # a same-labelled button.
    _submit_button(at.main, "Try the sample project").click()
    at.run()
    assert not at.exception

    reloaded = FoodOptimizer("Sample project")
    assert len(reloaded.X_history) == 1
    assert any("Sample project" in c.value and "1 experiments" in c.value
                for c in at.caption), [c.value for c in at.caption]


def test_sidebar_sample_project_button_available_with_project_open(project_with_history):
    """The sample project must be reachable from the sidebar even once a
    project already exists (the welcome panel is gone at that point)."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert not at.exception

    _submit_button(at.sidebar, "Try the sample project").click()
    at.run()
    assert not at.exception

    opt = FoodOptimizer("Sample project")
    assert len(opt.variables) >= 3
    assert any("Sample project" in c.value for c in at.caption), \
        [c.value for c in at.caption]


def test_sidebar_sample_project_button_reopens_without_recreating(project_with_history):
    """Clicking the sidebar sample button when the sample already exists
    must reopen it, not recreate it (experiment count unchanged)."""
    pre = FoodOptimizer("Sample project")
    pre.add_ingredient("Water", 0, 100)
    pre.add_objective("Taste", 1.0, goal="max")
    pre.tell({"Water": 50.0}, {"Taste": 7.0})

    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert not at.exception

    _submit_button(at.sidebar, "Try the sample project").click()
    at.run()
    assert not at.exception

    assert len(FoodOptimizer("Sample project").X_history) == 1


def test_edit_form_guarded_when_no_objectives_remain(project_with_history):
    """Removing the last objective on a project with history must not crash
    st.columns(0) in the edit-results form; Delete Experiment and Rewind
    must still render."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert not at.exception
    _submit_button(at, "Remove Taste").click()
    at.run()
    _submit_button(at, "Yes, remove").click()
    at.run()
    assert not at.exception
    assert any("Add a measurement in the Set up tab before editing past results" in i.value
               for i in at.info), [i.value for i in at.info]
    assert any(b.label == "Delete experiment 1" for b in at.button), [b.label for b in at.button]
    assert any(b.label == "Rewind" for b in at.button), [b.label for b in at.button]


def test_conflicting_form_save_shows_banner_immediately(project_with_history, tmp_path):
    """A form submit does not rerun on its own, so when the save it triggers
    is refused (another window changed the file first), the conflict banner
    must appear on THIS run, not wait for the user's next click."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()

    # Another window edits and saves the project after this session loaded it.
    other = FoodOptimizer("my_project")
    other.add_objective("Crunch", 1.0)
    time.sleep(0.01)  # coarse-mtime filesystems

    obj_name_input = next(t for t in at.text_input if t.label.startswith("Measurement name"))
    obj_name_input.set_value("Zing")
    _submit_button(at, "Add or update objective").click()
    at.run()

    assert not at.exception
    assert any("another window" in e.value for e in at.error), [e.value for e in at.error]
    assert not any("Added Zing" in s.value for s in at.success), [s.value for s in at.success]
    reloaded = FoodOptimizer("my_project")
    assert [o["name"] for o in reloaded.objectives] == ["Taste", "Crunch"]


def test_restore_warns_when_another_window_edited_project(project_with_history, tmp_path):
    """Restore's 'Yes, replace' constructs a fresh FoodOptimizer against the
    current file, which would re-stamp the shared conflict tracker and hide a
    concurrent edit. The stale check must run against what THIS session
    loaded, before that reconstruction, and must leave the restore candidate
    in place so the user can Reload and retry."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "my_project"
    at.run()

    # Another window edits and saves the project after this session loaded it.
    other = FoodOptimizer("my_project")
    other.add_objective("Crunch", 1.0)
    time.sleep(0.01)  # coarse-mtime filesystems

    donor = FoodOptimizer("donor")
    donor.add_ingredient("Flour", 0, 100)
    donor.add_objective("Sweetness", 1.0)
    donor.tell({"Flour": 10.0}, {"Sweetness": 5.0})

    at.session_state["_restore_candidate"] = donor.export_json()
    at.run()
    _submit_button(at, "Yes, replace").click()
    at.run()

    assert not at.exception
    assert any("another window" in e.value for e in at.error), [e.value for e in at.error]
    reloaded = FoodOptimizer("my_project")
    assert len(reloaded.X_history) == 1   # original single experiment untouched
    assert [o["name"] for o in reloaded.objectives] == ["Taste", "Crunch"]
