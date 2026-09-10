"""UI-level regression tests for app.py, using Streamlit's AppTest.

These drive the real Streamlit script headlessly, so they catch the class of
bug where the backend raises a clean ValueError but the UI fails to catch it
and shows the user a raw traceback.
"""

import io
import os
import pathlib
import time

import pandas as pd
import pyarrow as pa
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


def test_hard_reset_targets_active_project_not_typed_name(project_with_history, tmp_path):
    """Typing another name in the sidebar without clicking Create must not
    redirect Hard Reset at that other project (audit: confirmed critical bug)."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.sidebar.text_input[0].set_value("other")   # the sidebar "Project Name" box (only text input in sidebar)
    _submit_button(at, "Hard reset").click()
    at.run()
    _submit_button(at, "Yes, reset").click()
    at.run()
    assert not at.exception
    assert (tmp_path / "my_project_archived.pkl").exists()
    assert not (tmp_path / "other.pkl").exists()
    assert not (tmp_path / "other_archived.pkl").exists()
    assert [h.value for h in at.sidebar.subheader] == ["Projects", "my_project"]
    assert FoodOptimizer("my_project").X_history == []


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
    assert any("3 formulations" in w.value for w in at.warning), [w.value for w in at.warning]
    assert len(FoodOptimizer("my_project").X_history) == 1   # nothing changed yet
    _submit_button(at, "Yes, replace").click()
    at.run()
    assert not at.exception
    assert (tmp_path / "my_project_pre_restore.pkl").exists()
    assert len(FoodOptimizer("my_project").X_history) == 3
    assert any("Restored 3 formulations" in s.value for s in at.success)


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
    assert "my_project" in [h.value for h in at.sidebar.subheader]
    assert "older" not in [h.value for h in at.sidebar.subheader]


def test_pending_confirm_is_cleared_on_project_switch(project_with_history, tmp_path):
    other = FoodOptimizer("second")
    other.add_ingredient("Flour", 0, 100)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "my_project"  # pin: "second" now exists on
    # disk with a later mtime, which would otherwise become the auto-loaded "most
    # recent" project (Task 8) before this test ever switches to it on purpose.
    at.run()
    _submit_button(at, "Hard reset").click()   # arm the confirmation
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
    assert "second" in [h.value for h in at.sidebar.subheader]


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
    assert "second" in [h.value for h in at.sidebar.subheader]
    open_btn = _submit_button(at.sidebar, "Open")
    assert open_btn.disabled


def test_delete_project_confirms_archives_and_leaves_the_list(project_with_history, tmp_path):
    """Delete project must ask first, rename the file to an archive copy
    rather than erase it, drop the project from the list, and land on the
    welcome panel when nothing is left to open."""
    from storage import LocalStorage
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    _submit_button(at.sidebar, "Delete").click()
    at.run()
    assert (tmp_path / "my_project.pkl").exists()               # not yet
    warn = next(w.value for w in at.warning if "my_project" in w.value)
    # Real grammar, never "1 formulation(s)".
    assert "its 1 formulation?" in warn and "(s)" not in warn, warn
    _submit_button(at.sidebar, "Yes, delete").click()
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
    _submit_button(at.sidebar, "Delete").click()
    at.run()
    _submit_button(at.sidebar, "Yes, delete").click()
    at.run()
    assert not at.exception
    assert "second" in [h.value for h in at.sidebar.subheader]
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

    assert len(FoodOptimizer("Sample project").X_history) == 1
    assert "Sample project" in [h.value for h in at.sidebar.subheader]


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
    assert "Sample project" in [h.value for h in at.sidebar.subheader]


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


def _unknown(node, kind, label):
    """AppTest has no accessor for st.download_button or st.file_uploader: both
    arrive as UnknownElement carrying the raw proto. Walk the element tree and
    return the first one of `kind` with this label. For a download button
    proto.type is 'primary'/'secondary'; for an uploader it is the list of
    accepted extensions."""
    def walk(n):
        children = getattr(n, "children", None) or {}
        if hasattr(children, "values"):
            children = children.values()
        for child in children:
            if (type(child).__name__ == "UnknownElement"
                    and getattr(child, "type", None) == kind
                    and getattr(child, "label", None) == label):
                yield child
            yield from walk(child)
    return next(walk(node))


def _tab_primaries(at, index):
    """The coloured buttons inside one tab. AppTest renders every tab's body on
    every run, so 'one primary per tab' can only be checked tab by tab:
    at.tabs[0] is Set up, [1] is Make a batch, [2] is Results."""
    return [b.label for b in at.tabs[index].button if b.proto.type == "primary"]


def test_tabs_are_the_loop_in_order(project_with_history):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["1 · Set up", "2 · Make a batch", "3 · Results"]


def test_opening_a_project_lands_on_set_up_when_it_is_incomplete(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    FoodOptimizer("bare")
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "bare"
    at.session_state["_land_on_open"] = True
    at.run()
    assert at.session_state["main_tab"] == "1 · Set up"


def test_opening_a_project_lands_on_the_batch_when_one_is_open(project_with_history):
    project_with_history.set_pending_batch([{"Water": 10.0}])
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "my_project"
    at.session_state["_land_on_open"] = True
    at.run()
    assert at.session_state["main_tab"] == "2 · Make a batch"


def test_opening_a_project_lands_on_results(project_with_history):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "my_project"
    at.session_state["_land_on_open"] = True
    at.run()
    assert at.session_state["main_tab"] == "3 · Results"


def test_a_plain_rerun_never_moves_the_tab(project_with_history):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "my_project"
    at.session_state["main_tab"] = "1 · Set up"
    at.run()
    at.run()
    assert at.session_state["main_tab"] == "1 · Set up"


def test_the_sidebar_carries_no_coloured_button_except_open(project_with_history, tmp_path):
    """The lit thing on screen must be the tab's next action, never a sidebar
    control that would do nothing."""
    FoodOptimizer("second").add_ingredient("Flour", 0, 100)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "my_project"
    at.run()
    lit = [b.label for b in at.sidebar.button if b.proto.type == "primary"]
    assert lit == [], lit
    open_button = _submit_button(at.sidebar, "Open")
    assert open_button.disabled and open_button.proto.type == "secondary"
    at.selectbox(key="project_select").set_value("second")
    at.run()
    open_button = _submit_button(at.sidebar, "Open")
    assert not open_button.disabled and open_button.proto.type == "primary"


def test_an_armed_confirmation_is_the_one_lit_sidebar_button(project_with_history):
    """A confirmation button is transient and IS the next action while it is
    armed, so exactly one thing is lit — and only until it is answered."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "my_project"
    at.run()
    assert [b.label for b in at.sidebar.button if b.proto.type == "primary"] == []
    _submit_button(at.sidebar, "Hard reset").click()
    at.run()
    lit = [b.label for b in at.sidebar.button if b.proto.type == "primary"]
    assert lit == ["Yes, reset"], lit
    _submit_button(at.sidebar, "Cancel").click()
    at.run()
    assert at.session_state["hard_reset__pending"] is False
    # Cancel ends in st.rerun(), and AppTest leaves the aborted pass's buttons
    # in the element tree alongside the settled ones; the next plain run shows
    # what the browser would show.
    at.run()
    assert [b.label for b in at.sidebar.button if b.proto.type == "primary"] == []


def test_the_restore_confirmation_is_lit_like_every_other_confirmation(project_with_history):
    donor = FoodOptimizer("donor")
    donor.add_ingredient("Flour", 0, 100)
    donor.add_objective("Crunch", 1.0, goal="max")
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "my_project"
    at.session_state["_restore_candidate"] = donor.export_json()
    at.run()
    lit = [b.label for b in at.sidebar.button if b.proto.type == "primary"]
    assert lit == ["Yes, replace"], lit


def test_removing_every_measurement_keeps_the_open_batch(project_with_history):
    """Measurements never discard a batch: a formulation is amounts, and it can
    still be made while the user reworks what they will score."""
    project_with_history.set_pending_batch([{"Water": 10.0}, {"Water": 20.0}])
    project_with_history.remove_objective("Taste")
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "my_project"
    at.run()
    assert not at.exception
    assert FoodOptimizer("my_project").pending_batch is not None
    assert not any("ingredient list" in i.value for i in at.info), [i.value for i in at.info]


def test_restore_says_nothing_about_a_copy_when_none_was_kept(project_with_history, monkeypatch):
    """'kept as None' is not a sentence. With no file to archive, the message
    simply stops after naming what was restored."""
    from storage import LocalStorage
    monkeypatch.setattr(LocalStorage, "archive", lambda self, *a, **k: None)
    donor = FoodOptimizer("donor")
    donor.add_ingredient("Water", 0, 100)
    donor.add_objective("Taste", 1.0, goal="max")
    for value in (10.0, 20.0, 30.0):
        donor.tell({"Water": value}, {"Taste": 5.0})
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "my_project"
    at.session_state["_restore_candidate"] = donor.export_json()
    at.run()
    _submit_button(at, "Yes, replace").click()
    at.run()
    assert not at.exception
    note = next(s.value for s in at.success if "Restored" in s.value)
    assert note == "Restored 3 formulations into my_project.", note
    assert "None" not in note


def test_the_saved_line_shows_for_a_project_opened_but_not_yet_saved(project_with_history):
    """last_saved_at is only set by a save in THIS session, so a returning user
    would otherwise see no saved line at all until their first edit."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "my_project"
    at.run()
    assert at.session_state["optimizer"].last_saved_at is None
    assert any("automatically, to this Mac" in c.value for c in at.sidebar.caption), \
        [c.value for c in at.sidebar.caption]


def test_manage_project_holds_hard_reset_and_delete(project_with_history):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert any(e.label == "Manage project" for e in at.sidebar.expander), \
        [e.label for e in at.sidebar.expander]
    labels = [b.label for b in at.sidebar.button]
    assert "Hard reset" in labels and "Delete" in labels, labels


def test_restore_accepts_any_file_name(project_with_history):
    """Contents decide, not the extension: the uploader must not filter."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    restore = _unknown(at.sidebar, "file_uploader", "Restore from backup")
    assert list(restore.proto.type) == []


def test_restoring_a_real_0_2_x_backup_numbers_it_on_the_way_in(project_with_history, tmp_path):
    """The whole compatibility promise, exercised through the sidebar the user
    actually clicks, not through import_json."""
    donor = FoodOptimizer("donor")
    donor.add_ingredient("Water", 0, 100)
    donor.add_objective("Taste", 1.0, goal="max")
    for value in (10.0, 20.0, 30.0):
        donor.tell({"Water": value}, {"Taste": 5.0})
    legacy = donor.export_json()
    for key in ("formulation_ids", "batch_history", "notes_history", "skipped",
                "next_formulation_no", "pending_batch_no",
                "pending_batch_created", "pending_batch_discarded"):
        legacy.pop(key, None)
    legacy["CLASS_VERSION"] = 6

    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "my_project"
    at.session_state["_restore_candidate"] = legacy
    at.run()
    _submit_button(at, "Yes, replace").click()
    at.run()
    assert not at.exception
    restored = FoodOptimizer("my_project")
    assert restored.formulation_ids == [1, 2, 3]
    assert restored.batch_history == [None, None, None]
    assert restored.next_formulation_no == 4


def test_a_batch_that_no_longer_fits_the_ingredients_is_discarded_with_a_notice(project_with_history):
    """Replaces the 0.2.x stale-batch test: a batch saved before the ingredient
    list changed cannot be made, and must not sit there looking makeable."""
    project_with_history.set_pending_batch([{"Water": 10.0, "Ghost": 1.0}])
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "my_project"
    at.run()
    assert not at.exception
    assert FoodOptimizer("my_project").pending_batch is None
    assert any("ingredient list" in i.value for i in at.info), [i.value for i in at.info]


def test_the_batch_line_counts_what_is_left_to_make(project_with_history):
    project_with_history.set_pending_batch([{"Water": 10.0}, {"Water": 20.0}])
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "my_project"
    at.run()
    assert any(c.value == "Batch 1 · 2 to make" for c in at.caption), \
        [c.value for c in at.caption]


def test_an_uploaded_import_sheet_does_not_survive_a_hard_reset(project_with_history):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_import_rows"] = pd.DataFrame({"Water": [1.0], "Taste": [5.0]})
    at.run()
    assert "_import_rows" in at.session_state
    _submit_button(at, "Hard reset").click()
    at.run()
    _submit_button(at, "Yes, reset").click()
    at.run()
    assert not at.exception
    assert "_import_rows" not in at.session_state


def test_an_uploaded_import_sheet_does_not_follow_a_project_switch(project_with_history, tmp_path):
    FoodOptimizer("second").add_ingredient("Flour", 0, 100)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "my_project"
    at.session_state["_import_rows"] = pd.DataFrame({"Water": [1.0], "Taste": [5.0]})
    at.run()
    at.selectbox(key="project_select").set_value("second")
    at.run()
    _submit_button(at.sidebar, "Open").click()
    at.run()
    assert not at.exception
    assert "_import_rows" not in at.session_state


@pytest.fixture
def burger(tmp_path, monkeypatch):
    """A project shaped like the sample: units everywhere, two targets,
    firmness the more important of the two."""
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("burger")
    opt.set_amount_unit("g")
    opt.add_ingredient("Pea protein", 0, 25)
    opt.add_ingredient("Methylcellulose", 0, 3)
    opt.add_objective("Juiciness", 1.0, goal="target", target=7,
                      min_val=0, max_val=10, unit="/10")
    opt.add_objective("Firmness", 1.5, goal="target", target=6,
                      min_val=0, max_val=10, unit="N")
    return opt


def _labels(at):
    return [b.label for b in at.button]


def test_setup_shows_the_amount_unit_and_uses_it_in_headers(burger):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert at.text_input(key="amount_unit").value == "g"
    assert any("Min (g)" in str(d.value.columns.tolist()) for d in at.dataframe), \
        [d.value.columns.tolist() for d in at.dataframe]


def test_a_unit_typed_with_a_trailing_space_is_saved_once(burger):
    """Comparing a stripped value with an unstripped one re-saved on every
    rerun, and a failing save then bounced the script into a rerun loop."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.text_input(key="amount_unit").set_value("kg ")
    at.run()
    assert FoodOptimizer("burger").amount_unit == "kg"
    saved_at = at.session_state["optimizer"].last_saved_at
    at.run()
    assert at.session_state["optimizer"].last_saved_at == saved_at   # no re-save
    assert not at.exception


def test_measurements_table_is_sorted_by_importance_with_shares(burger):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    table = next(d.value for d in at.dataframe if "Priority" in d.value.columns)
    assert list(table.columns) == ["Priority", "Measurement", "Goal", "Scale",
                                   "Importance", "Share"]
    assert list(table["Measurement"]) == ["Firmness", "Juiciness"]
    assert list(table["Share"]) == ["60%", "40%"]
    assert list(table["Goal"]) == ["Target 6 N", "Target 7/10"]


def test_the_score_function_is_written_out_under_the_table(burger):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert any(c.value == (
        "Overall score = 1.5 × Firmness closeness + 1.0 × Juiciness closeness. "
        "Closeness is 1 on target and falls evenly with distance from it; a "
        "full scale width away scores 0. Every measurement on target scores 2.50."
    ) for c in at.caption), [c.value for c in at.caption]


def test_the_collapsed_sections_are_the_spec_list(burger):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    labels = [e.label for e in at.expander]
    for label in ("Change the ingredient list", "Add a measurement",
                  "How closeness is worked out", "Process settings (optional)",
                  "Limits (optional)", "Advanced model settings"):
        assert label in labels, labels


def test_importance_is_a_number_from_a_tenth_to_a_hundred(burger):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    field = at.number_input(key="meas_new_importance")
    assert field.min == 0.1 and field.max == 100.0
    assert field.value == 1.0
    assert not at.slider, [s.label for s in at.slider]


def test_the_live_share_line_follows_what_is_typed(burger):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.text_input(key="meas_new_name").set_value("Chewiness")
    at.number_input(key="meas_new_importance").set_value(2.5)
    at.run()
    assert any(m.value == "Chewiness: 50% of the overall score" for m in at.markdown), \
        [m.value for m in at.markdown]


def test_continue_is_grey_and_names_what_is_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("bare2")
    opt.add_ingredient("Water", 0, 100)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    button = _submit_button(at, "Continue to make a batch")
    assert button.disabled
    assert button.proto.type == "secondary"
    assert any(c.value == "Add at least one measurement." for c in at.caption), \
        [c.value for c in at.caption]


def test_continue_is_lit_and_moves_to_the_batch_tab(burger):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    button = _submit_button(at, "Continue to make a batch")
    assert not button.disabled and button.proto.type == "primary"
    button.click()
    at.run()
    assert not at.exception
    assert at.session_state["main_tab"] == "2 · Make a batch"


def test_set_up_has_exactly_one_coloured_button(burger):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert _tab_primaries(at, 0) == ["Continue to make a batch"], _tab_primaries(at, 0)


def test_an_armed_confirmation_takes_the_colour_off_continue(burger):
    """Two lit buttons ask two questions at once. While a confirmation is on
    screen, answering it is the only coloured thing to do."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    _submit_button(at, "Remove Firmness").click()
    at.run()
    assert _tab_primaries(at, 0) == ["Yes, remove"], _tab_primaries(at, 0)
    assert _submit_button(at, "Continue to make a batch").disabled
    _submit_button(at, "Cancel").click()
    at.run()
    at.run()   # Cancel reruns from inside the handler; AppTest keeps the
               # aborted pass's elements until the next run
    assert _tab_primaries(at, 0) == ["Continue to make a batch"], _tab_primaries(at, 0)


def test_edit_reopens_the_measurement_with_the_name_locked(burger):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    _submit_button(at, "Edit Firmness").click()
    at.run()
    assert not at.exception
    assert at.text_input(key="meas_Firmness_name").disabled is True
    assert at.number_input(key="meas_Firmness_importance").value == 1.5
    assert "Cancel" in _labels(at)
    _submit_button(at, "Cancel").click()
    at.run()
    assert "_editing_measurement" not in at.session_state


def test_changing_importance_keeps_a_copy_and_reports_the_move(burger, tmp_path):
    # Chosen so Formulation 3 wins at importance 1.5 (1.75 vs 1.65) and
    # Formulation 7 wins at 2.0 (2.10 vs 2.00) — the move the sentence reports.
    burger.tell({"Pea protein": 10.0, "Methylcellulose": 1.0},
                {"Juiciness": 7.0, "Firmness": 1.0}, formulation_no=3, batch_no=1)
    burger.tell({"Pea protein": 20.0, "Methylcellulose": 2.0},
                {"Juiciness": 0.0, "Firmness": 7.0}, formulation_no=7, batch_no=1)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "burger"
    at.session_state["main_tab"] = "1 · Set up"
    at.run()
    _submit_button(at, "Edit Firmness").click()
    at.run()
    at.number_input(key="meas_Firmness_importance").set_value(2.0)
    _submit_button(at, "Save changes").click()
    at.run()
    assert not at.exception
    assert (tmp_path / "burger_pre_edit.pkl").exists()
    note = next(s.value for s in at.success if "importance changed" in s.value)
    assert note == ("Firmness importance changed to 2.0. Every overall score was "
                    "recalculated. Best moved from Formulation 3 to Formulation 7.")
    assert at.session_state["main_tab"] == "1 · Set up"   # a set-up edit never moves


def test_editing_a_measurement_leaves_the_open_batch_alone(burger):
    """A formulation is a set of amounts; changing what you measure cannot
    invalidate it, and silently retiring its numbers would break the promise
    that a number is never reissued."""
    burger.set_pending_batch([{"Pea protein": 10.0, "Methylcellulose": 1.0}])
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "burger"
    at.run()
    _submit_button(at, "Edit Firmness").click()
    at.run()
    at.number_input(key="meas_Firmness_importance").set_value(2.0)
    _submit_button(at, "Save changes").click()
    at.run()
    assert not at.exception
    reloaded = FoodOptimizer("burger")
    assert [r["formulation"] for r in reloaded.pending_batch] == [1]
    assert reloaded.pending_batch_no == 1


def test_adding_and_removing_a_measurement_leaves_the_open_batch_alone(burger):
    burger.set_pending_batch([{"Pea protein": 10.0, "Methylcellulose": 1.0}])
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "burger"
    at.run()
    at.text_input(key="meas_new_name").set_value("Chewiness")
    _submit_button(at, "Add measurement").click()
    at.run()
    assert FoodOptimizer("burger").pending_batch is not None
    _submit_button(at, "Remove Chewiness").click()
    at.run()
    _submit_button(at, "Yes, remove").click()
    at.run()
    assert not at.exception
    assert FoodOptimizer("burger").pending_batch is not None


def test_adding_an_ingredient_does_discard_the_open_batch(burger):
    burger.set_pending_batch([{"Pea protein": 10.0, "Methylcellulose": 1.0}])
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "burger"
    at.run()
    at.text_input(key="ing_name").set_value("Beet juice powder")
    at.number_input(key="ing_max").set_value(2.0)
    _submit_button(at, "Add ingredient").click()
    at.run()
    assert not at.exception
    assert FoodOptimizer("burger").pending_batch is None
    # food_bo drops the batch inside add_ingredient, so app.py's makeability
    # check never sees a mismatch: the handler has to say so itself.
    assert any("The open batch was discarded because the ingredient list or "
               "its ranges changed since it was generated." == i.value
               for i in at.info), [i.value for i in at.info]


def test_removing_a_measurement_keeps_a_copy_and_recalculates(burger, tmp_path):
    burger.tell({"Pea protein": 10.0, "Methylcellulose": 1.0},
                {"Juiciness": 7.0, "Firmness": 1.0}, formulation_no=1, batch_no=1)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    _submit_button(at, "Remove Juiciness").click()
    at.run()
    _submit_button(at, "Yes, remove").click()
    at.run()
    assert not at.exception
    assert (tmp_path / "burger_pre_edit.pkl").exists()
    assert [o["name"] for o in FoodOptimizer("burger").objectives] == ["Firmness"]
    assert any("Every overall score was recalculated" in s.value for s in at.success), \
        [s.value for s in at.success]


def test_adding_a_measurement_from_the_collapsed_form(burger):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.text_input(key="meas_new_name").set_value("Chewiness")
    at.text_input(key="meas_new_unit").set_value("N")
    at.number_input(key="meas_new_max").set_value(20.0)
    at.number_input(key="meas_new_importance").set_value(0.5)
    _submit_button(at, "Add measurement").click()
    at.run()
    assert not at.exception
    added = next(o for o in FoodOptimizer("burger").objectives
                 if o["name"] == "Chewiness")
    assert added["unit"] == "N" and added["weight"] == 0.5 and added["max_val"] == 20.0


def test_the_ingredient_uploader_help_names_the_real_columns(burger):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    uploader = _unknown(at.main, "file_uploader", "Upload ingredients CSV")
    assert "Name, Min, Max" in uploader.proto.help, uploader.proto.help
    assert any(c.value == "One row per ingredient: Name, Min, Max."
               for c in at.caption), [c.value for c in at.caption]


def test_change_the_ingredient_list_carries_the_unit(burger):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert at.number_input(key="ing_min").label == "Min (g)"
    assert at.number_input(key="ing_max").label == "Max (g)"


def test_manual_add_ingredient_and_the_mid_run_minimum_notice(burger):
    burger.tell({"Pea protein": 10.0, "Methylcellulose": 1.0},
                {"Juiciness": 7.0, "Firmness": 6.0})
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert at.number_input(key="ing_min").disabled is True
    at.text_input(key="ing_name").set_value("Beet juice powder")
    at.number_input(key="ing_max").set_value(2.0)
    _submit_button(at, "Add ingredient").click()
    at.run()
    assert not at.exception
    assert any(v["name"] == "Beet juice powder" for v in FoodOptimizer("burger").variables)


def test_process_setting_baseline_error_is_shown_not_raised(burger):
    burger.tell({"Pea protein": 10.0, "Methylcellulose": 1.0},
                {"Juiciness": 7.0, "Firmness": 6.0})
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.text_input(key="pp_name").set_value("Cook temperature")
    at.number_input(key="pp_min").set_value(150.0)
    at.number_input(key="pp_max").set_value(220.0)
    at.number_input(key="pp_base").set_value(100.0)   # outside [150, 220]
    _submit_button(at, "Add process setting").click()
    at.run()
    assert not at.exception
    assert any("must be between" in str(e.value) for e in at.error), \
        [str(e.value) for e in at.error]


def test_removing_a_process_setting_confirms_and_archives(burger, tmp_path):
    """0.2.x safety behaviour that must survive the rebuild: a removal that
    rewrites recorded formulations keeps a copy first."""
    burger.add_process_parameter("Cook temperature", 150, 220)
    burger.tell({"Pea protein": 10.0, "Methylcellulose": 1.0,
                 "Cook temperature": 180.0},
                {"Juiciness": 7.0, "Firmness": 6.0})
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    _submit_button(at, "Remove").click()
    at.run()
    assert any("Cook temperature" in w.value for w in at.warning), \
        [w.value for w in at.warning]
    assert any(v["name"] == "Cook temperature" for v in FoodOptimizer("burger").variables)
    _submit_button(at, "Yes, remove").click()
    at.run()
    assert not at.exception
    assert (tmp_path / "burger_pre_delete.pkl").exists()
    assert not any(v["name"] == "Cook temperature"
                   for v in FoodOptimizer("burger").variables)


def test_limit_fields_start_blank_and_there_is_no_maximum_tick_box(burger):
    """A pre-filled 100 g maximum on a 400 g burger made every future batch
    infeasible in one click; a tick box the user forgot then threw away the
    number they did type. Blank means no limit, everywhere."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    for key in ("tm_max", "tm_min", "qc_max", "qc_min"):
        assert at.number_input(key=key).value is None, key
        assert at.number_input(key=key).placeholder == "no limit", key
    keys = [c.key for c in at.checkbox]
    assert "qc_use_max" not in keys and "qc_use_min" not in keys, keys


def test_a_new_limit_says_past_formulations_are_kept(burger):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.number_input(key="tm_max").set_value(400.0)
    _submit_button(at, "Add a total amount limit").click()
    at.run()
    assert not at.exception
    assert any("Formulations already made are kept. The next batch will respect "
               "this limit." in s.value for s in at.success), [s.value for s in at.success]


def test_a_limit_that_excludes_everything_made_so_far_warns(burger):
    burger.tell({"Pea protein": 20.0, "Methylcellulose": 2.0},
                {"Juiciness": 7.0, "Firmness": 6.0})
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.number_input(key="tm_max").set_value(5.0)     # the one formulation is 22 g
    _submit_button(at, "Add a total amount limit").click()
    at.run()
    assert not at.exception
    assert any(w.value == "No formulation you have made fits this limit."
               for w in at.warning), [w.value for w in at.warning]
    assert FoodOptimizer("burger").quantity_constraints    # still added


def test_conflicting_save_shows_the_banner_immediately(burger, tmp_path):
    """Another window changed the file first: the conflict must appear on THIS
    run, not wait for the next click."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    other = FoodOptimizer("burger")
    other.add_objective("Crunch", 1.0)
    time.sleep(0.01)   # coarse-mtime filesystems
    at.text_input(key="meas_new_name").set_value("Zing")
    _submit_button(at, "Add measurement").click()
    at.run()
    assert not at.exception
    assert any("another window" in e.value for e in at.error), [e.value for e in at.error]


def test_adding_a_measurement_that_already_exists_is_refused(burger):
    """add_objective REPLACES by name: typing Firmness into the add form
    silently dropped its unit, target and scale and rescored every result
    with no copy kept."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.text_input(key="meas_new_name").set_value("firmness")   # case-insensitive
    _submit_button(at, "Add measurement").click()
    at.run()
    assert not at.exception
    assert any(e.value == "That measurement already exists. Use Edit on its "
               "row to change it." for e in at.error), [e.value for e in at.error]
    kept = next(o for o in FoodOptimizer("burger").objectives
                if o["name"] == "Firmness")
    assert kept["unit"] == "N" and kept["weight"] == 1.5
    assert kept["goal"] == "target" and kept["target"] == 6.0
    assert not any("Added" in m.value for m in at.success), [m.value for m in at.success]


def test_a_pause_that_discards_the_batch_says_so(burger):
    burger.set_pending_batch([{"Pea protein": 10.0, "Methylcellulose": 1.0}])
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "burger"
    at.run()
    at.multiselect(key="pause_pick").set_value(["Methylcellulose"])
    _submit_button(at, "Pause selected").click()
    at.run()
    assert not at.exception
    assert FoodOptimizer("burger").pending_batch is None
    assert any("The open batch was discarded" in i.value for i in at.info), \
        [i.value for i in at.info]


def test_changing_the_scale_also_keeps_a_copy_and_reports_it(burger, tmp_path):
    """Importance is not the only field that rescores history: the goal, the
    target and either end of the scale all feed closeness."""
    burger.tell({"Pea protein": 10.0, "Methylcellulose": 1.0},
                {"Juiciness": 7.0, "Firmness": 6.0}, formulation_no=1, batch_no=1)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    _submit_button(at, "Edit Firmness").click()
    at.run()
    at.number_input(key="meas_Firmness_max").set_value(20.0)
    _submit_button(at, "Save changes").click()
    at.run()
    assert not at.exception
    assert (tmp_path / "burger_pre_edit.pkl").exists()
    assert any(m.value == "Updated Firmness. Every overall score was "
               "recalculated." for m in at.success), [m.value for m in at.success]
    assert FoodOptimizer("burger").objectives[1]["max_val"] == 20.0


def test_only_the_unit_changed_means_no_copy_and_no_recalculation_claim(burger):
    burger.tell({"Pea protein": 10.0, "Methylcellulose": 1.0},
                {"Juiciness": 7.0, "Firmness": 6.0})
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    _submit_button(at, "Edit Firmness").click()
    at.run()
    at.text_input(key="meas_Firmness_unit").set_value("kPa")
    _submit_button(at, "Save changes").click()
    at.run()
    assert not at.exception
    assert any(m.value == "Updated Firmness." for m in at.success), \
        [m.value for m in at.success]


def test_removing_a_measurement_keeps_a_copy_even_with_no_results(burger, tmp_path):
    """A copy is kept before every removal on this tab, history or not."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    _submit_button(at, "Remove Juiciness").click()
    at.run()
    _submit_button(at, "Yes, remove").click()
    at.run()
    assert not at.exception
    assert (tmp_path / "burger_pre_edit.pkl").exists()
    assert [o["name"] for o in FoodOptimizer("burger").objectives] == ["Firmness"]


def test_removing_a_process_setting_confirms_even_with_no_results(burger, tmp_path):
    burger.add_process_parameter("Cook temperature", 150, 220)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    _submit_button(at, "Remove").click()
    at.run()
    assert any("Cook temperature" in w.value for w in at.warning), \
        [w.value for w in at.warning]
    assert any(v["name"] == "Cook temperature" for v in FoodOptimizer("burger").variables)
    _submit_button(at, "Yes, remove").click()
    at.run()
    assert not at.exception
    assert (tmp_path / "burger_pre_delete.pkl").exists()
    assert not any(v["name"] == "Cook temperature"
                   for v in FoodOptimizer("burger").variables)


def test_a_failed_save_shows_red_and_never_a_green_added(burger):
    """A green 'Added Beet juice powder.' over a change that never reached the
    file is the worst message this tab could show."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    other = FoodOptimizer("burger")          # another window writes first
    other.add_objective("Crunch", 1.0)
    time.sleep(0.01)                          # coarse-mtime filesystems
    at.text_input(key="ing_name").set_value("Beet juice powder")
    at.number_input(key="ing_max").set_value(2.0)
    _submit_button(at, "Add ingredient").click()
    at.run()
    assert not at.exception
    assert any("another window" in e.value for e in at.error), [e.value for e in at.error]
    assert not any("Beet juice powder" in m.value for m in at.success), \
        [m.value for m in at.success]


def test_the_live_share_line_stays_away_until_a_name_is_typed(burger):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert not any("% of the overall score" in m.value for m in at.markdown), \
        [m.value for m in at.markdown]


def test_the_scale_labels_are_sentence_case(burger):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    labels = [n.label for n in at.number_input]
    assert "Lowest possible" in labels and "Highest possible" in labels, labels


def test_the_ingredient_table_has_no_status_column_until_something_is_paused(burger):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    table = next(d.value for d in at.dataframe if "Min (g)" in d.value.columns)
    assert list(table.columns) == ["Name", "Min (g)", "Max (g)"], list(table.columns)
    at.multiselect(key="pause_pick").set_value(["Methylcellulose"])
    _submit_button(at, "Pause selected").click()
    at.run()
    table = next(d.value for d in at.dataframe if "Min (g)" in d.value.columns)
    assert list(table["Status"]) == ["active", "paused"], list(table["Status"])


@pytest.fixture
def open_batch(burger):
    """Batch 1 of two formulations, numbered 1 and 2, nothing recorded yet."""
    burger.set_pending_batch([{"Pea protein": 10.0, "Methylcellulose": 1.0},
                              {"Pea protein": 20.0, "Methylcellulose": 2.0}])
    return burger


def test_batch_tab_is_grey_and_names_what_is_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("bare3")
    opt.add_ingredient("Water", 0, 100)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    generate = _submit_button(at, "Generate formulations")
    back = _submit_button(at, "Back to set up")
    assert generate.disabled and generate.proto.type == "secondary"
    assert back.proto.type == "secondary"
    assert any(c.value == "Add at least one measurement." for c in at.caption)


def test_generate_opens_a_numbered_batch_and_stays_on_the_tab(burger):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    # Pin the project so the landing rule does not fire, and seed the tab:
    # st.tabs(key=) does not write session_state on render.
    at.session_state["_loaded_project"] = "burger"
    at.session_state["main_tab"] = "2 · Make a batch"
    at.run()
    _submit_button(at, "Generate 3 formulations").click()
    at.run()
    assert not at.exception
    reloaded = FoodOptimizer("burger")
    assert [r["formulation"] for r in reloaded.pending_batch] == [1, 2, 3]
    assert reloaded.pending_batch_no == 1
    assert at.session_state["main_tab"] == "2 · Make a batch"   # never auto-moves


def test_the_getting_started_sentence_changes_after_five_results(burger):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert any(c.value == ("The first few formulations spread across your "
                           "ingredient ranges; later batches aim closer to "
                           "your targets.") for c in at.caption), \
        [c.value for c in at.caption]
    for i in range(5):
        burger.tell({"Pea protein": 5.0 + i, "Methylcellulose": 1.0},
                    {"Juiciness": 6.0, "Firmness": 6.0})
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert any(c.value == "Each batch aims closer to your targets."
               for c in at.caption), [c.value for c in at.caption]


def test_only_one_batch_at_a_time_is_said_where_it_helps(open_batch):
    """The line belongs to the no-batch state, not under a batch that is
    plainly open."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert not any(c.value == "Only one batch is open at a time."
                   for c in at.caption), [c.value for c in at.caption]


def test_repeat_of_the_best_is_offered_and_adds_one_formulation(burger):
    burger.tell({"Pea protein": 10.0, "Methylcellulose": 1.0},
                {"Juiciness": 7.0, "Firmness": 6.0}, formulation_no=1, batch_no=1)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert any(c.label == "Include a repeat of Formulation 1" for c in at.checkbox), \
        [c.label for c in at.checkbox]
    at.checkbox(key="repeat_best").check()
    at.number_input(key="batch_size").set_value(2)
    at.run()
    _submit_button(at, "Generate 2 formulations").click()
    at.run()
    assert not at.exception
    reloaded = FoodOptimizer("burger")
    assert len(reloaded.pending_batch) == 3
    # Numbers 1 is taken, so the new ones start at 2 and never repeat it.
    assert [r["formulation"] for r in reloaded.pending_batch] == [2, 3, 4]
    assert reloaded.pending_batch[-1]["recipe"] == {"Pea protein": 10.0,
                                                    "Methylcellulose": 1.0}


def test_the_batch_table_carries_units_and_a_total(open_batch):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert any(m.value == "**Batch 1 · make these 2 formulations**"
               for m in at.markdown), [m.value for m in at.markdown]
    table = next(d.value for d in at.dataframe if "Formulation" in d.value.columns)
    assert list(table.columns) == ["Formulation", "Pea protein (g)",
                                   "Methylcellulose (g)", "Total (g)"]
    box = at.number_input(key="scale_total")
    assert box.value is None                          # empty: as generated
    assert box.proto.placeholder == "as generated"


def test_scaling_rescales_the_screen_the_sheet_and_nothing_else(open_batch):
    """What is downloaded must equal what is on screen: the whole point of the
    scale box is that the lab weighs those amounts out."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.number_input(key="scale_total").set_value(22.0)
    at.run()
    table = next(d.value for d in at.dataframe if "Formulation" in d.value.columns)
    assert table["Pea protein (g)"].iloc[0] == pytest.approx(20.0)
    assert any(c.value == "Sheets use the scaled amounts (total 22 g)."
               for c in at.caption), [c.value for c in at.caption]
    assert FoodOptimizer("burger").pending_batch[0]["recipe"]["Pea protein"] == 10.0


def test_biggest_changes_line_appears_from_batch_two(burger):
    burger.tell({"Pea protein": 10.1, "Methylcellulose": 1.0},
                {"Juiciness": 7.0, "Firmness": 6.0}, formulation_no=1, batch_no=1)
    burger.set_pending_batch([{"Pea protein": 8.0, "Methylcellulose": 1.8}],
                             batch_no=2)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert any(c.value == ("Biggest changes from Formulation 1: "
                           "Pea protein −2.1 g, Methylcellulose +0.8 g.")
               for c in at.caption), [c.value for c in at.caption]


def test_both_downloads_are_offered_and_only_the_batch_sheet_is_lit(open_batch):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert _unknown(at.main, "download_button",
                    "Download batch sheet").proto.type == "primary"
    assert _unknown(at.main, "download_button",
                    "Download printable sheets").proto.type == "secondary"
    # The download IS the coloured thing here, so no coloured button competes.
    assert _tab_primaries(at, 1) == [], _tab_primaries(at, 1)
    at.number_input(key="f1_Firmness").set_value(6.0)
    at.run()
    assert _unknown(at.main, "download_button",
                    "Download batch sheet").proto.type == "secondary"


def test_leaving_a_row_out_does_not_grey_the_batch_sheet(open_batch):
    """A disabled number_input still returns its stored value, so a naive
    'has anything been typed' check flipped the download grey on a tick."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.number_input(key="f1_Firmness").set_value(6.0)
    at.run()
    at.checkbox(key="f1_leave_out").check()
    at.run()
    assert _unknown(at.main, "download_button",
                    "Download batch sheet").proto.type == "primary"


def test_regenerating_names_the_numbers_it_discards(open_batch):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    _submit_button(at, "Generate a different batch").click()
    at.run()
    assert any(w.value == ("Batch 1 (Formulations 1, 2) will be discarded. "
                           "Those numbers will not be used again.")
               for w in at.warning), [w.value for w in at.warning]
    _submit_button(at, "Yes, discard").click()
    at.run()
    assert not at.exception
    reloaded = FoodOptimizer("burger")
    assert [r["formulation"] for r in reloaded.pending_batch] == [3, 4]
    assert reloaded.pending_batch_no == 1     # the batch keeps its number
    assert any(c.value == ("Formulations 1 and 2 were discarded and their "
                           "numbers will not be used again.")
               for c in at.caption), [c.value for c in at.caption]


def test_the_printable_sheet_names_the_formulation_and_the_batch(open_batch):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert any(e.label == "Printable formulation sheets" for e in at.expander)
    texts = [t.value for t in at.text]
    assert "Formulation 1 · batch 1" in texts, texts
    assert any(t.startswith("Not made") for t in texts), texts
    # Underscore rules go through st.text: st.markdown would italicise them and
    # would mangle a measurement named L_a_b.
    assert any("Measured Firmness · target 6 N:" in t for t in texts), texts


def test_result_inputs_carry_the_goal_and_unit_and_are_ordered_by_importance(open_batch):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert at.number_input(key="f1_Firmness").label == "Firmness · target 6 N"
    assert at.number_input(key="f1_Juiciness").label == "Juiciness · target 7/10"
    labels = [n.label for n in at.number_input]
    assert labels.index("Firmness · target 6 N") < labels.index("Juiciness · target 7/10")
    assert any(c.value == ("Enter your panel mean. Leave a measurement blank if "
                           "it could not be scored.") for c in at.caption)


def test_result_inputs_do_not_clamp_and_refuse_out_of_scale_on_save(open_batch):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    field = at.number_input(key="f1_Firmness")
    # An unbounded st.number_input reports ±DBL_MAX rather than None, so what
    # is checked is that neither end is the 0-to-10 scale: nothing is clamped.
    assert field.min < 0.0 and field.max > 10.0
    field.set_value(12.0)
    at.number_input(key="f2_Firmness").set_value(6.0)
    at.run()
    assert at.number_input(key="f1_Firmness").value == 12.0   # not clamped to 10
    _submit_button(at, "Save results").click()
    at.run()
    assert not at.exception
    assert any(e.value == ("Firmness 12 N is outside your scale of 0 to 10 N. "
                           "Widen the scale in Set up, or check the value.")
               for e in at.error), [e.value for e in at.error]
    assert FoodOptimizer("burger").X_history == []


def test_leave_out_comes_after_the_note_field(open_batch):
    """The first control in a row must be a box the user came to fill, not a
    checkbox for not making it."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()

    def positions(node, out):
        children = getattr(node, "children", None) or {}
        if hasattr(children, "values"):
            children = children.values()
        for child in children:
            key = getattr(child, "key", None)
            if key in ("f1_Firmness", "f1_note", "f1_leave_out"):
                out.append(key)
            positions(child, out)
        return out

    order = positions(at.main, [])
    assert order.index("f1_Firmness") < order.index("f1_note") < order.index("f1_leave_out"), order


def test_save_lights_only_when_every_kept_row_has_a_value(open_batch):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert _submit_button(at, "Save results").disabled
    at.number_input(key="f1_Firmness").set_value(6.0)
    at.run()
    assert _submit_button(at, "Save results").disabled
    assert any(c.value == "1 of 2 entered · saved when you press Save"
               for c in at.caption), [c.value for c in at.caption]
    at.number_input(key="f2_Juiciness").set_value(4.0)
    at.run()
    save = _submit_button(at, "Save results")
    assert not save.disabled and save.proto.type == "primary"
    assert _tab_primaries(at, 1) == ["Save results"], _tab_primaries(at, 1)


def test_the_counter_counts_kept_rows_only(open_batch):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.number_input(key="f1_Firmness").set_value(6.0)
    at.checkbox(key="f2_leave_out").check()
    at.run()
    assert any(c.value == "1 of 1 entered · 1 left out · saved when you press Save"
               for c in at.caption), [c.value for c in at.caption]
    assert not _submit_button(at, "Save results").disabled


def test_saving_records_partials_notes_and_moves_to_results(open_batch):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "burger"
    at.session_state["main_tab"] = "2 · Make a batch"
    at.run()
    at.number_input(key="f1_Firmness").set_value(6.0)
    at.number_input(key="f1_Juiciness").set_value(7.0)
    at.text_input(key="f1_note").set_value("held together")
    at.number_input(key="f2_Firmness").set_value(2.0)     # Juiciness left blank
    at.run()
    _submit_button(at, "Save results").click()
    at.run()
    assert not at.exception
    reloaded = FoodOptimizer("burger")
    assert reloaded.formulation_ids == [1, 2]
    assert reloaded.batch_history == [1, 1]
    assert reloaded.notes_history[0] == "held together"
    assert reloaded.results_history[1] == {"Firmness": 2.0}
    assert reloaded.pending_batch is None
    assert at.session_state["main_tab"] == "3 · Results"
    # A real save also puts the saved line in the sidebar and no backup nag.
    assert any("automatically, to this Mac" in c.value for c in at.sidebar.caption), \
        [c.value for c in at.sidebar.caption]
    assert not any("Download a backup now" in w.value for w in at.warning)


def test_a_failed_save_shows_the_banner_and_does_not_move_tabs(open_batch, monkeypatch):
    """go_to_tab reruns, which would outrun app.py's end-of-script save-error
    check and hide the fact that nothing was written."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "burger"
    at.session_state["main_tab"] = "2 · Make a batch"
    at.run()
    at.number_input(key="f1_Firmness").set_value(6.0)
    at.number_input(key="f2_Firmness").set_value(4.0)
    at.run()
    at.session_state["optimizer"].storage.save = lambda name, state: (_ for _ in ()).throw(
        __import__("storage").StorageError("The disk is full — your last change was NOT saved.")
    )
    _submit_button(at, "Save results").click()
    at.run()
    assert not at.exception
    assert at.session_state["main_tab"] == "2 · Make a batch"
    assert any("NOT saved" in e.value for e in at.error), [e.value for e in at.error]


def test_leave_out_disables_the_row_but_keeps_its_values(open_batch):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.number_input(key="f2_Firmness").set_value(3.0)
    at.checkbox(key="f2_leave_out").check()
    at.run()
    assert at.number_input(key="f2_Firmness").disabled is True
    assert at.number_input(key="f2_Firmness").value == 3.0
    at.number_input(key="f1_Firmness").set_value(6.0)
    at.run()
    _submit_button(at, "Save results").click()
    at.run()
    assert not at.exception
    reloaded = FoodOptimizer("burger")
    assert reloaded.formulation_ids == [1]
    assert reloaded.skipped == [{"formulation": 2, "batch": 1,
                                 "recipe": {"Pea protein": 20.0,
                                            "Methylcellulose": 2.0},
                                 "note": "Not made"}]


def test_every_row_left_out_says_there_is_nothing_to_save(open_batch):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.checkbox(key="f1_leave_out").check()
    at.checkbox(key="f2_leave_out").check()
    at.run()
    assert any(i.value == "Nothing to save — at least one formulation needs results."
               for i in at.info), [i.value for i in at.info]
    assert _submit_button(at, "Save results").disabled
    assert FoodOptimizer("burger").pending_batch is not None


def test_a_partly_recorded_batch_reopens_with_recorded_rows_read_only(open_batch):
    open_batch.tell({"Pea protein": 10.0, "Methylcellulose": 1.0},
                    {"Juiciness": 7.0, "Firmness": 6.0},
                    formulation_no=1, batch_no=1)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert not at.exception
    keys = [n.key for n in at.number_input]
    assert "f1_Firmness" not in keys
    assert "f2_Firmness" in keys
    assert any(c.value == "Firmness 6 N · Juiciness 7/10" for c in at.caption), \
        [c.value for c in at.caption]


def test_uploading_a_sheet_matches_global_formulation_numbers(open_batch):
    """AppTest cannot drive st.file_uploader, so prime the parsed sheet the way
    the uploader handler does."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "burger"
    at.session_state["main_tab"] = "2 · Make a batch"
    at.session_state["_results_upload"] = pd.DataFrame(
        {"Formulation": [2], "Firmness": [6.0], "Juiciness": [7.0],
         "Note": ["from the sheet"]})
    at.run()
    assert any("Formulation 2" in i.value for i in at.info), [i.value for i in at.info]
    _submit_button(at, "Save uploaded results").click()
    at.run()
    assert not at.exception
    reloaded = FoodOptimizer("burger")
    assert reloaded.formulation_ids == [2]
    assert reloaded.notes_history == ["from the sheet"]
    assert [r["formulation"] for r in reloaded.pending_batch] == [1, 2]
    assert at.session_state["main_tab"] == "2 · Make a batch"   # one row still open


def test_a_stale_upload_is_cleared_by_hard_reset(open_batch, tmp_path):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_results_upload"] = pd.DataFrame(
        {"Formulation": [1], "Firmness": [6.0], "Juiciness": [7.0]})
    at.run()
    assert "_results_upload" in at.session_state
    _submit_button(at, "Hard reset").click()
    at.run()
    _submit_button(at, "Yes, reset").click()
    at.run()
    assert not at.exception
    assert "_results_upload" not in at.session_state


def test_generate_is_the_one_lit_button_before_a_batch_exists(burger):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert _tab_primaries(at, 1) == ["Generate 3 formulations"], _tab_primaries(at, 1)


def test_a_confirmation_takes_the_colour_from_the_batch_sheet(open_batch):
    """While a confirmation is armed, answering it is the one thing to do: the
    lit download steps aside rather than competing with the Yes."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert _unknown(at.main, "download_button",
                    "Download batch sheet").proto.type == "primary"
    _submit_button(at, "Generate a different batch").click()
    at.run()
    assert _unknown(at.main, "download_button",
                    "Download batch sheet").proto.type == "secondary"
    assert _tab_primaries(at, 1) == ["Yes, discard"], _tab_primaries(at, 1)


def test_a_confirmation_greys_save_results(open_batch):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.number_input(key="f1_Firmness").set_value(6.0)
    at.number_input(key="f2_Firmness").set_value(4.0)
    at.run()
    assert _tab_primaries(at, 1) == ["Save results"], _tab_primaries(at, 1)
    _submit_button(at, "Generate a different batch").click()
    at.run()
    save = _submit_button(at, "Save results")
    assert save.disabled and save.proto.type == "secondary"
    assert _tab_primaries(at, 1) == ["Yes, discard"], _tab_primaries(at, 1)



def _displayed(element):
    """The strings a st.dataframe actually puts on screen. A Styler's number
    formatting rides along in the proto as display values, so this is what the
    user reads, not the raw floats behind it."""
    return pa.ipc.open_stream(
        io.BytesIO(element.proto.arrow_data.styler.display_values)
    ).read_pandas()


def test_amounts_stay_as_generated_until_a_total_is_typed(open_batch):
    """The two formulations weigh 11 g and 22 g. Nothing may quietly rewrite
    the second one to the first one's size just because the tab was opened."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    table = next(d.value for d in at.dataframe if "Formulation" in d.value.columns)
    assert list(table["Total (g)"]) == [11.0, 22.0], table.to_dict()
    assert any(c.value == "Shown as generated." for c in at.caption), \
        [c.value for c in at.caption]
    texts = [t.value for t in at.text]
    assert "Pea protein: 20 g" in texts, texts        # the sheet, as generated


def test_typing_the_first_rows_own_total_still_scales_the_others(open_batch):
    """11 g is what the first formulation already weighs, so a box that opened
    pre-filled with it made typing 11 a silent no-op. It must scale."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.number_input(key="scale_total").set_value(11.0)
    at.run()
    table = next(d.value for d in at.dataframe if "Formulation" in d.value.columns)
    assert list(table["Total (g)"]) == pytest.approx([11.0, 11.0])
    assert table["Pea protein (g)"].iloc[1] == pytest.approx(10.0)
    assert any(c.value == "Sheets use the scaled amounts (total 11 g)."
               for c in at.caption), [c.value for c in at.caption]
    # Both sheets now carry the same amounts, so the downloads followed.
    assert [t.value for t in at.text].count("Pea protein: 10 g") == 2
    assert FoodOptimizer("burger").pending_batch[1]["recipe"]["Pea protein"] == 20.0


def test_the_batch_table_never_dresses_a_setting_as_an_amount(burger):
    burger.add_process_parameter("Cook temperature", 100, 200)
    burger.set_pending_batch([{"Pea protein": 10.0, "Methylcellulose": 1.0,
                               "Cook temperature": 180.0}])
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    shown = _displayed(next(d for d in at.dataframe
                            if "Formulation" in d.value.columns))
    assert shown["Pea protein (g)"].iloc[0] == "10.00"
    assert shown["Cook temperature"].iloc[0] == "180", shown.to_dict()


def test_a_one_formulation_batch_reads_as_one(burger):
    burger.set_pending_batch([{"Pea protein": 10.0, "Methylcellulose": 1.0}])
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert any(m.value == "**Batch 1 · make this 1 formulation**"
               for m in at.markdown), [m.value for m in at.markdown]


def test_a_note_typed_on_a_row_that_is_left_out_is_kept(open_batch):
    """Why it was not made is often typed before the box is ticked, and it is
    the only record of what went wrong."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.number_input(key="f1_Firmness").set_value(6.0)
    at.text_input(key="f2_note").set_value("burner failed")
    at.checkbox(key="f2_leave_out").check()
    at.run()
    _submit_button(at, "Save results").click()
    at.run()
    assert not at.exception
    assert FoodOptimizer("burger").skipped[0]["note"] == "burner failed"


def test_a_recorded_row_shows_its_note(open_batch):
    open_batch.tell({"Pea protein": 10.0, "Methylcellulose": 1.0},
                    {"Juiciness": 7.0, "Firmness": 6.0},
                    formulation_no=1, batch_no=1, note="held together")
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert any(c.value == "Firmness 6 N · Juiciness 7/10 · Note: held together"
               for c in at.caption), [c.value for c in at.caption]


@pytest.fixture
def scored(burger):
    """Two recorded formulations in batch 1 and one left out, so the results
    screens have a best, a progress line and a left-out row to show."""
    burger.tell({"Pea protein": 10.0, "Methylcellulose": 1.0},
                {"Juiciness": 7.0, "Firmness": 1.0},
                formulation_no=1, batch_no=1, note="crumbly")
    burger.tell({"Pea protein": 20.0, "Methylcellulose": 2.0},
                {"Juiciness": 7.0, "Firmness": 8.0},
                formulation_no=2, batch_no=1)
    burger.record_skipped(3, 1, {"Pea protein": 25.0, "Methylcellulose": 3.0})
    return burger


def test_results_empty_state_offers_the_first_batch_and_the_import(burger):
    """A fresh project is exactly when someone imports past work, so the
    import must not hide behind the empty state."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "burger"
    at.session_state["main_tab"] = "3 · Results"
    at.run()
    assert any(m.value == "No results yet." for m in at.markdown), \
        [m.value for m in at.markdown]
    assert any(e.label == "Import past formulations from a CSV"
               for e in at.expander), [e.label for e in at.expander]
    button = _submit_button(at, "Make your first batch")
    assert button.proto.type == "primary"
    button.click()
    at.run()
    assert at.session_state["main_tab"] == "2 · Make a batch"


def test_best_heading_off_by_table_and_score_caption(scored):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert any(h.value == "Best so far: Formulation 2 (batch 1)"
               for h in at.subheader), [h.value for h in at.subheader]
    off_by = next(d.value for d in at.dataframe if "Off by" in d.value.columns)
    assert list(off_by["Measurement"]) == ["Firmness", "Juiciness"]
    assert off_by["Off by"].iloc[0] == "2.0 N too high"
    assert off_by["Off by"].iloc[1] == "On target"
    assert any(m.value == "**Amounts to make it**" for m in at.markdown)
    assert any(c.value == ("Overall score 2.20 of 2.50 · 2.50 is every "
                           "measurement on target · not comparable across "
                           "projects.") for c in at.caption), \
        [c.value for c in at.caption]


def test_not_used_lists_ingredients_only(scored):
    """A cook temperature legitimately set to 0 is not an unused ingredient."""
    # Added mid-run, a process setting needs the value all prior batches ran
    # at; zero is the point of this test.
    scored.add_process_parameter("Cook temperature", 0, 220, baseline=0)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert not any("Cook temperature" in c.value for c in at.caption
                   if c.value.startswith("Not used:")), [c.value for c in at.caption]


def test_the_progress_line_reports_the_first_batch_plainly(scored):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert any(c.value == "Batch 1 recorded." for c in at.caption), \
        [c.value for c in at.caption]


def test_the_progress_line_reports_an_improvement_and_a_flat_batch(scored):
    scored.tell({"Pea protein": 12.0, "Methylcellulose": 1.2},
                {"Juiciness": 7.0, "Firmness": 6.0}, formulation_no=4, batch_no=2)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert any(c.value == "Batch 2 recorded · best improved 2.20 → 2.50"
               for c in at.caption), [c.value for c in at.caption]
    scored.tell({"Pea protein": 30.0, "Methylcellulose": 0.5},
                {"Juiciness": 1.0, "Firmness": 1.0}, formulation_no=5, batch_no=3)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert any(c.value == "Batch 3 recorded · no improvement."
               for c in at.caption), [c.value for c in at.caption]


def test_all_formulations_table_stars_the_best_and_marks_the_left_out(scored):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    table = next(d.value for d in at.dataframe if "Best" in d.value.columns)
    assert list(table.columns) == ["Best", "Batch", "Formulation", "Firmness (N)",
                                   "Juiciness (/10)", "Overall score", "Recorded",
                                   "Note"]
    assert list(table["Formulation"]) == [2, 1, 3]
    assert list(table["Best"]) == ["★", "", ""]
    assert table["Note"].iloc[2] == "Not made"
    assert at.selectbox(key="results_order").options == ["Best first",
                                                         "Newest first",
                                                         "Batch order"]


def test_show_amounts_adds_the_amount_columns(scored):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.toggle(key="show_amounts").set_value(True)
    at.run()
    table = next(d.value for d in at.dataframe if "Best" in d.value.columns)
    assert "Pea protein (g)" in table.columns


def test_download_all_formulations_is_grey(scored):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    element = _unknown(at.main, "download_button",
                       "Download all formulations (CSV)")
    assert element.proto.type == "secondary"


def test_correcting_a_result_reports_the_change_and_the_move(scored):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "burger"
    at.session_state["main_tab"] = "3 · Results"
    at.run()
    at.selectbox(key="correct_formulation").set_value(1)
    at.run()
    at.number_input(key="correct_1_Firmness").set_value(6.0)
    at.run()
    _submit_button(at, "Save correction").click()
    at.run()
    assert not at.exception
    note = next(s.value for s in at.success if "corrected" in s.value)
    assert note == ("Formulation 1 Firmness corrected 1 → 6. Best moved from "
                    "Formulation 2 to Formulation 1.")
    assert at.session_state["main_tab"] == "3 · Results"    # no auto-move


def test_a_correction_can_be_closed_again(scored):
    """A selectbox opened with index=None cannot be cleared by the user, so
    the edit row would sit there for the rest of the session."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.selectbox(key="correct_formulation").set_value(1)
    at.run()
    assert "correct_1_Firmness" in [n.key for n in at.number_input]
    _submit_button(at, "Done").click()
    at.run()
    assert not at.exception
    assert "correct_1_Firmness" not in [n.key for n in at.number_input]


def test_a_correction_outside_the_scale_is_refused(scored):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.selectbox(key="correct_formulation").set_value(1)
    at.run()
    field = at.number_input(key="correct_1_Firmness")
    field.set_value(12.0)
    at.run()
    # No clamping: the box keeps the reading it was given (an unbounded
    # st.number_input reports its limits as +/-DBL_MAX, never None), and the
    # refusal comes on save.
    assert at.number_input(key="correct_1_Firmness").value == 12.0
    _submit_button(at, "Save correction").click()
    at.run()
    assert not at.exception
    assert any(e.value == ("Firmness 12 N is outside your scale of 0 to 10 N. "
                           "Widen the scale in Set up, or check the value.")
               for e in at.error), [e.value for e in at.error]
    assert FoodOptimizer("burger").results_history[0]["Firmness"] == 1.0


def test_a_correction_with_every_box_blank_is_refused(burger):
    """AppTest cannot type a box empty, so reach an all-blank edit row the way
    a user does: a partial result whose only measurement was then removed."""
    burger.tell({"Pea protein": 10.0, "Methylcellulose": 1.0},
                {"Firmness": 5.0}, formulation_no=1, batch_no=1)
    burger.remove_objective("Firmness")
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.selectbox(key="correct_formulation").set_value(1)
    at.run()
    assert at.number_input(key="correct_1_Juiciness").value is None
    _submit_button(at, "Save correction").click()
    at.run()
    assert not at.exception
    assert any(e.value == "Enter a value for at least one measurement."
               for e in at.error), [e.value for e in at.error]


def test_results_renders_when_the_last_measurement_is_gone(scored):
    """History outlives measurements: st.columns(0) and an empty closeness
    table must not take the tab down."""
    scored.remove_objective("Firmness")
    scored.remove_objective("Juiciness")
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert not at.exception
    assert any("Add a measurement in Set up" in i.value for i in at.info), \
        [i.value for i in at.info]


def test_results_foot_starts_the_next_batch(scored):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "burger"
    at.session_state["main_tab"] = "3 · Results"
    at.run()
    button = _submit_button(at, "Start the next batch")
    assert button.proto.type == "primary"
    button.click()
    at.run()
    assert at.session_state["main_tab"] == "2 · Make a batch"


def test_results_foot_points_back_at_an_unrecorded_batch(scored):
    scored.set_pending_batch([{"Pea protein": 5.0, "Methylcellulose": 0.5},
                              {"Pea protein": 6.0, "Methylcellulose": 0.6}],
                             batch_no=2)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert any(b.label == "Back to batch 2 · 2 to record" for b in at.button), \
        _labels(at)


def test_results_has_exactly_one_coloured_button(scored):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert _tab_primaries(at, 2) == ["Start the next batch"], _tab_primaries(at, 2)


def test_the_results_foot_steps_aside_for_a_confirmation(scored):
    """A tab never shows two coloured buttons: while a confirmation is armed
    its Yes is the one lit button and the foot is grey and unclickable."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    _submit_button(at, "Undo the last batch").click()
    at.run()
    assert _tab_primaries(at, 2) == ["Yes, undo"], _tab_primaries(at, 2)
    foot = _submit_button(at, "Start the next batch")
    assert foot.proto.type == "secondary" and foot.disabled


def test_the_empty_state_button_steps_aside_for_a_confirmation(burger):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "burger"
    at.session_state["main_tab"] = "3 · Results"
    at.run()
    _submit_button(at, "Hard reset").click()
    at.run()
    button = _submit_button(at, "Make your first batch")
    assert button.proto.type == "secondary" and button.disabled


def test_results_collapsed_sections_are_the_spec_list(scored):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    labels = [e.label for e in at.expander]
    for label in ("Progress chart", "Undo the last batch",
                  "Delete a formulation", "Import past formulations from a CSV"):
        assert label in labels, labels


def test_undo_the_last_batch_says_it_once_keeps_a_copy_and_removes_it(scored, tmp_path):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    sentence = "Removes batch 1 and its 3 results. A copy is kept first."
    # The confirmation carries the sentence; a caption above it would say the
    # same thing twice.
    assert not any(c.value == sentence for c in at.caption), \
        [c.value for c in at.caption]
    _submit_button(at, "Undo the last batch").click()
    at.run()
    assert any(w.value == sentence for w in at.warning), [w.value for w in at.warning]
    _submit_button(at, "Yes, undo").click()
    at.run()
    assert not at.exception
    assert (tmp_path / "burger_pre_undo.pkl").exists()
    reloaded = FoodOptimizer("burger")
    assert reloaded.X_history == [] and reloaded.skipped == []
    assert reloaded.next_formulation_no == 4      # 1, 2 and 3 retire


def test_undo_is_refused_while_a_batch_is_open(scored):
    """Undo must not take an open batch down as a side effect."""
    scored.set_pending_batch([{"Pea protein": 5.0, "Methylcellulose": 0.5}],
                             batch_no=2)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert any(c.value == "Record or discard the open batch first."
               for c in at.caption), [c.value for c in at.caption]
    assert _submit_button(at, "Undo the last batch").disabled
    assert FoodOptimizer("burger").pending_batch is not None


def test_deleting_a_formulation_keeps_later_numbers(scored, tmp_path):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.selectbox(key="delete_formulation").set_value(1)
    at.run()
    _submit_button(at, "Delete Formulation 1").click()
    at.run()
    assert any("Later formulations keep their numbers. A copy is kept first."
               in w.value for w in at.warning), [w.value for w in at.warning]
    # One Cancel on this tab, not the confirmation's and the select box's
    # side by side.
    tab_labels = [b.label for b in at.tabs[2].button]
    assert tab_labels.count("Cancel") == 1, tab_labels
    _submit_button(at, "Yes, delete").click()
    at.run()
    assert not at.exception
    assert (tmp_path / "burger_pre_delete.pkl").exists()
    assert FoodOptimizer("burger").formulation_ids == [2]


def test_the_import_caption_names_this_projects_columns(burger):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "burger"
    at.run()
    caption = next(c.value for c in at.caption
                   if c.value.startswith("One row per formulation"))
    assert "Pea protein" in caption and "Firmness" in caption, caption


def test_importing_past_formulations_marks_them_and_accepts_partials(burger):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "burger"
    at.session_state["_import_rows"] = pd.DataFrame({
        "Pea protein": [12.0, 13.0], "Methylcellulose": [1.2, 1.3],
        "Juiciness": [6.0, None], "Firmness": [5.0, 5.5],
    })
    at.run()
    _submit_button(at, "Import all rows").click()
    at.run()
    assert not at.exception
    reloaded = FoodOptimizer("burger")
    assert reloaded.formulation_ids == [1, 2]
    assert reloaded.batch_history == [None, None]
    assert reloaded.notes_history == ["Imported", "Imported"]
    assert reloaded.results_history[1] == {"Firmness": 5.5}
