"""UI-level regression tests for app.py, using Streamlit's AppTest.

These drive the real Streamlit script headlessly, so they catch the class of
bug where the backend raises a clean ValueError but the UI fails to catch it
and shows the user a raw traceback.
"""

import os
import pathlib
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
