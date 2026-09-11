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
    _submit_button(at, "Empty this project").click()
    at.run()
    _submit_button(at, "Yes, empty it").click()
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
    _submit_button(at, "Empty this project").click()   # arm the confirmation
    at.run()
    assert any(b.label == "Yes, empty it" for b in at.button)
    # Choosing a project reruns the script and enables Open; only then can a
    # user click it (Open is greyed out while the box shows the open project).
    at.selectbox(key="project_select").set_value("second")
    at.run()
    _submit_button(at, "Open").click()
    at.run()
    # Open is drawn at the foot of the sidebar, so the armed confirmation above
    # it is still in the aborted pass's element tree; the next plain run shows
    # what the browser would show.
    at.run()
    assert not at.exception
    assert not any(b.label == "Yes, empty it" for b in at.button), [b.label for b in at.button]
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
    _submit_button(at.sidebar, "Delete this project").click()
    at.run()
    assert (tmp_path / "my_project.pkl").exists()               # not yet
    warn = next(w.value for w in at.warning if "my_project" in w.value)
    # Real grammar, never "1 formulation(s)".
    assert "its 1 formulation?" in warn and "(s)" not in warn, warn
    _submit_button(at.sidebar, "Yes, delete it").click()
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
    _submit_button(at.sidebar, "Delete this project").click()
    at.run()
    _submit_button(at.sidebar, "Yes, delete it").click()
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
    # The sidebar also offers "Try the sample project", so target the
    # welcome-panel button explicitly (scoped to at.main).
    _submit_button(at.main, "Try the sample project").click()
    at.run()
    assert not at.exception
    opt = FoodOptimizer("Sample project")
    # Eight ingredients: enough for a real formulation, few enough to read at
    # a glance in the batch table and the printable sheets.
    assert len(opt.variables) == 8
    assert opt.amount_unit == "g"
    # The sample is the plant-based burger brief for a trained panel: two
    # intensity scores with targets (7 and 6 out of 10), firmness the more
    # important of the two.
    assert [o["name"] for o in opt.objectives] == ["Juiciness", "Firmness"]
    assert all(o["goal"] == "target" for o in opt.objectives)
    assert {o["name"]: o["target"] for o in opt.objectives} == {"Juiciness": 7,
                                                                "Firmness": 6}
    assert {o["name"]: o["unit"] for o in opt.objectives} == {"Juiciness": "/10",
                                                               "Firmness": "/10"}
    weights = {o["name"]: o["weight"] for o in opt.objectives}
    assert weights["Firmness"] > weights["Juiciness"]


def test_the_sample_lists_firmness_first(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    _submit_button(at.main, "Try the sample project").click()
    at.run()
    # Set up is complete, but nothing has been made yet: the sample opens on
    # its set-up so the user sees what they are about to make.
    assert at.session_state["main_tab"] == "1 · Set up"
    table = next(d.value for d in at.dataframe if "Importance" in d.value.columns)
    # A panel score's "/10" is written once, on the measurement's own row,
    # and never after a number.
    assert list(table["Measurement"]) == ["Firmness (/10)", "Juiciness (/10)"]
    assert list(table["Goal"]) == ["Target 6", "Target 7"]
    assert list(table["Scale"]) == ["0 to 10", "0 to 10"]


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


def _unknowns(node, kind):
    """Every UnknownElement of `kind` under `node`, in render order."""
    def walk(n):
        children = getattr(n, "children", None) or {}
        if hasattr(children, "values"):
            children = children.values()
        for child in children:
            if (type(child).__name__ == "UnknownElement"
                    and getattr(child, "type", None) == kind):
                yield child
            yield from walk(child)
    return list(walk(node))


def _tab_primaries(at, index):
    """The coloured ACTIONS inside one tab — st.button and st.download_button
    alike, because a lit download is as much the next action as a lit button.
    AppTest renders every tab's body on every run, so 'one primary per tab' can
    only be checked tab by tab: at.tabs[0] is Set up, [1] is Make a batch, [2]
    is Results."""
    tab = at.tabs[index]
    lit = [b.label for b in tab.button if b.proto.type == "primary"]
    lit += [d.label for d in _unknowns(tab, "download_button")
            if d.proto.type == "primary"]
    return lit


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


def test_opening_a_ready_project_with_no_results_lands_on_set_up(tmp_path,
                                                                 monkeypatch):
    """Results would be an empty tab. The set-up is what there is to look at,
    and confirming it is the step before making anything."""
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("ready")
    opt.add_ingredient("Water", 0, 100)
    opt.add_objective("Taste", 1.0, goal="max")
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "ready"
    at.session_state["_land_on_open"] = True
    at.run()
    assert at.session_state["main_tab"] == "1 · Set up"


def test_a_project_whose_only_formulation_was_left_out_lands_on_results(
        tmp_path, monkeypatch):
    """A left-out formulation is a result the project holds: it has a number,
    amounts and a note, and tab 3 lists it."""
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("left_out")
    opt.add_ingredient("Water", 0, 100)
    opt.add_objective("Taste", 1.0, goal="max")
    opt.record_skipped(1, 1, {"Water": 10.0})
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "left_out"
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
    # ...and it steps aside the moment a confirmation is armed: the Yes is the
    # one lit thing in the sidebar, even with a switch waiting to be made.
    _submit_button(at.sidebar, "Empty this project").click()
    at.run()
    lit = [b.label for b in at.sidebar.button if b.proto.type == "primary"]
    assert lit == ["Yes, empty it"], lit
    open_button = _submit_button(at.sidebar, "Open")
    assert not open_button.disabled and open_button.proto.type == "secondary"


def test_arming_one_manage_project_confirmation_greys_the_other(project_with_history):
    """Two armed confirmations would put two coloured Yes buttons in the
    sidebar, each quietly keeping its own copy of the project."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "my_project"
    at.run()
    assert not at.button(key="hard_reset__btn").disabled
    assert not at.button(key="delete_project__btn").disabled
    _submit_button(at.sidebar, "Empty this project").click()
    at.run()
    assert at.button(key="delete_project__btn").disabled
    assert not at.button(key="hard_reset__btn").disabled
    _submit_button(at.sidebar, "Cancel").click()
    at.run()
    at.run()
    _submit_button(at.sidebar, "Delete this project").click()
    at.run()
    # Delete renders below Hard reset, so it can only grey the button above it
    # on the next run; a click in between is ignored by confirm_action anyway.
    at.run()
    assert at.button(key="hard_reset__btn").disabled
    assert not at.button(key="delete_project__btn").disabled


def test_an_armed_confirmation_is_the_one_lit_sidebar_button(project_with_history):
    """A confirmation button is transient and IS the next action while it is
    armed, so exactly one thing is lit — and only until it is answered."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "my_project"
    at.run()
    assert [b.label for b in at.sidebar.button if b.proto.type == "primary"] == []
    _submit_button(at.sidebar, "Empty this project").click()
    at.run()
    lit = [b.label for b in at.sidebar.button if b.proto.type == "primary"]
    assert lit == ["Yes, empty it"], lit
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


def test_manage_project_holds_empty_and_delete(project_with_history):
    """Two names for two outcomes: one empties the project, the other takes it
    out of the list. 'Hard reset' named the mechanism and read as the worse of
    the two."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert any(e.label == "Manage project" for e in at.sidebar.expander), \
        [e.label for e in at.sidebar.expander]
    labels = [b.label for b in at.sidebar.button]
    assert "Empty this project" in labels and "Delete this project" in labels, \
        labels


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
    _submit_button(at, "Empty this project").click()
    at.run()
    _submit_button(at, "Yes, empty it").click()
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
def test_a_unit_typed_with_a_trailing_space_is_saved_once(burger):
    """The unit is stripped before it is stored, and a plain rerun after it
    writes nothing: a re-save on every rerun bounced the script into a loop."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.selectbox(key="var_pick").select("Methylcellulose")
    at.run()
    at.text_input(key="unit_value").set_value("kg ")
    at.run()
    _submit_button(at, "Set unit").click()
    at.run()
    assert FoodOptimizer("burger").unit_of("Methylcellulose") == "kg"
    saved_at = at.session_state["optimizer"].last_saved_at
    at.run()
    assert at.session_state["optimizer"].last_saved_at == saved_at   # no re-save
    assert not at.exception


def test_measurements_table_is_sorted_by_importance(burger):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    table = next(d.value for d in at.dataframe if "Importance" in d.value.columns)
    # No Priority column: it was the row's position in a table already sorted
    # by importance, which is the same fact written twice. And no Share: it
    # read w / Σw, which is not a measurement's influence on the score once
    # any goal is a target.
    assert list(table.columns) == ["Measurement", "Goal", "Scale",
                                   "Importance"]
    assert list(table["Measurement"]) == ["Firmness", "Juiciness (/10)"]
    assert list(table["Goal"]) == ["Target 6 N", "Target 7"]
    assert list(table["Scale"]) == ["0 to 10 N", "0 to 10"]


def test_the_score_function_is_written_out_under_the_table(burger):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert any(c.value == (
        "Overall score = 1.5 × Firmness closeness + 1.0 × Juiciness "
        "closeness. Every measurement at its goal scores 2.50."
    ) for c in at.caption), [c.value for c in at.caption]
    # How closeness works is said once, in the expander, per goal.
    assert not any("falls evenly with distance" in c.value for c in at.caption)
def test_importance_is_a_number_from_a_tenth_to_a_hundred(burger):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    field = at.number_input(key="meas_new_importance")
    assert field.min == 0.1 and field.max == 100.0
    assert field.value == 1.0
    assert not at.slider, [s.label for s in at.slider]


def test_no_share_line_follows_what_is_typed(burger):
    """w / Σw is not a measurement's influence on the overall score: a target
    goal's closeness never spans the full 0 to 1, so two equally weighted
    measurements are not equally influential. The line is gone."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.text_input(key="meas_new_name").set_value("Chewiness")
    at.number_input(key="meas_new_importance").set_value(2.5)
    at.run()
    assert not any("% of the overall score" in m.value for m in at.markdown), \
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
    # The copy is named in the sentence: this edit has no confirmation before
    # it, so nothing else tells the user there is a way back.
    assert note == ("Firmness importance changed to 2.0. Every overall score "
                    "was recalculated. Best moved from Formulation 3 to "
                    "Formulation 7. A copy is saved in your FoodOptimizer folder first.")
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
    at.text_input(key="var_name").set_value("Beet juice powder")
    at.number_input(key="var_high").set_value(2.0)
    _submit_button(at, "Add ingredient or setting").click()
    at.run()
    assert not at.exception
    assert FoodOptimizer("burger").pending_batch is None
    # food_bo drops the batch inside add_ingredient, so app.py's makeability
    # check never sees a mismatch: the handler has to say so itself.
    assert any("The open batch was discarded because the ingredient list or "
               "its allowed amounts changed since it was generated." == i.value
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
    # The caption lists the columns; the tooltip says the one thing it does
    # not — what a blank Unit cell means — and says it in this project's unit.
    assert uploader.proto.help == "A blank Unit cell is in g.", \
        uploader.proto.help
    assert any(c.value.startswith("A CSV with the columns Name, Lowest, Highest")
               for c in at.caption), [c.value for c in at.caption]
def test_manual_add_ingredient_and_the_mid_run_minimum_notice(burger):
    burger.tell({"Pea protein": 10.0, "Methylcellulose": 1.0},
                {"Juiciness": 7.0, "Firmness": 6.0})
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    box = at.number_input(key="var_low")
    assert box.disabled is True
    assert box.proto.help == ("A new ingredient starts at 0 in every "
                              "formulation already made, so its lowest is "
                              "fixed at 0 for now.")
    at.text_input(key="var_name").set_value("Beet juice powder")
    at.number_input(key="var_high").set_value(2.0)
    _submit_button(at, "Add ingredient or setting").click()
    at.run()
    assert not at.exception
    assert any(v["name"] == "Beet juice powder" for v in FoodOptimizer("burger").variables)


def test_process_setting_baseline_error_is_shown_not_raised(burger):
    burger.tell({"Pea protein": 10.0, "Methylcellulose": 1.0},
                {"Juiciness": 7.0, "Firmness": 6.0})
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.radio(key="var_kind").set_value("Process setting")
    at.run()
    at.text_input(key="var_name").set_value("Cook temperature")
    at.number_input(key="var_low").set_value(150.0)
    at.number_input(key="var_high").set_value(220.0)
    at.number_input(key="var_base").set_value(100.0)   # outside [150, 220]
    at.run()
    _submit_button(at, "Add ingredient or setting").click()
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
    at.selectbox(key="var_pick").select("Cook temperature")
    at.run()
    _submit_button(at, "Remove Cook temperature").click()
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
    for key in ("qc_max", "qc_min"):
        assert at.number_input(key=key).value is None, key
        assert at.number_input(key=key).placeholder == "no limit", key
    keys = [c.key for c in at.checkbox]
    assert "qc_use_max" not in keys and "qc_use_min" not in keys, keys


def test_a_new_limit_says_past_formulations_are_kept(burger):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.number_input(key="qc_max").set_value(400.0)
    _submit_button(at, "Add amount limit").click()
    at.run()
    assert not at.exception
    assert any("Formulations already made are kept. The next batch will respect "
               "this limit." in s.value for s in at.success), [s.value for s in at.success]


def test_a_limit_that_excludes_everything_made_so_far_warns(burger):
    burger.tell({"Pea protein": 20.0, "Methylcellulose": 2.0},
                {"Juiciness": 7.0, "Firmness": 6.0})
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.number_input(key="qc_max").set_value(5.0)     # the one formulation is 22 g
    _submit_button(at, "Add amount limit").click()
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
    at.selectbox(key="var_pick").select("Methylcellulose")
    at.run()
    _submit_button(at, "Pause").click()
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
               "recalculated. A copy is saved in your FoodOptimizer folder first."
               for m in at.success), [m.value for m in at.success]
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
    at.selectbox(key="var_pick").select("Cook temperature")
    at.run()
    _submit_button(at, "Remove Cook temperature").click()
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
    at.text_input(key="var_name").set_value("Beet juice powder")
    at.number_input(key="var_high").set_value(2.0)
    _submit_button(at, "Add ingredient or setting").click()
    at.run()
    assert not at.exception
    assert any("another window" in e.value for e in at.error), [e.value for e in at.error]
    assert not any("Beet juice powder" in m.value for m in at.success), \
        [m.value for m in at.success]


def test_the_scale_labels_are_sentence_case(burger):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    labels = [n.label for n in at.number_input]
    assert "Lowest possible" in labels and "Highest possible" in labels, labels


def test_the_ingredient_table_has_no_status_column_until_something_is_paused(burger):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    table = next(d.value for d in at.dataframe if "Kind" in d.value.columns)
    assert list(table.columns) == ["Kind", "Name", "Lowest", "Highest",
                                   "Unit"], list(table.columns)
    at.selectbox(key="var_pick").select("Methylcellulose")
    at.run()
    _submit_button(at, "Pause").click()
    at.run()
    table = next(d.value for d in at.dataframe if "Kind" in d.value.columns)
    assert list(table["Status"]) == ["active", "paused · held at 0.00 g"], \
        list(table["Status"])


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
    assert any(c.value == ("The first few formulations spread across the "
                           "amounts you allowed; later batches aim closer to "
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
    assert any(c.label == "Remake Formulation 1 (best so far) in this batch" for c in at.checkbox), \
        [c.label for c in at.checkbox]
    at.checkbox(key="repeat_best").check()
    at.number_input(key="batch_size").set_value(2)
    at.run()
    # The button counts the repeat: two new formulations plus the repeat is
    # three to make, and the box asks for NEW formulations.
    assert any(n.label == "New formulations in this batch"
               for n in at.number_input), [n.label for n in at.number_input]
    _submit_button(at, "Generate 3 formulations").click()
    at.run()
    assert not at.exception
    reloaded = FoodOptimizer("burger")
    assert len(reloaded.pending_batch) == 3
    # Numbers 1 is taken, so the new ones start at 2 and never repeat it.
    assert [r["formulation"] for r in reloaded.pending_batch] == [2, 3, 4]
    assert reloaded.pending_batch[-1]["recipe"] == {"Pea protein": 10.0,
                                                    "Methylcellulose": 1.0}
    # And the extra row says what it is, on the table and in its note box.
    assert reloaded.pending_batch[-1]["note"] == "Repeat of Formulation 1"
    table = next(d.value for d in at.dataframe if "Formulation" in d.value.columns)
    assert list(table["Note"]) == ["", "", "Repeat of Formulation 1"]
    assert at.session_state["f4_note"] == "Repeat of Formulation 1"


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
    assert any(c.value == "Amounts shown for this total."
               for c in at.caption), [c.value for c in at.caption]
    assert FoodOptimizer("burger").pending_batch[0]["recipe"]["Pea protein"] == 10.0


def test_biggest_changes_line_appears_from_batch_two(burger):
    burger.tell({"Pea protein": 10.1, "Methylcellulose": 1.0},
                {"Juiciness": 7.0, "Firmness": 6.0}, formulation_no=1, batch_no=1)
    burger.set_pending_batch([{"Pea protein": 8.0, "Methylcellulose": 1.8}],
                             batch_no=2)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert any(c.value == ("Biggest changes in Formulation 2 from "
                           "Formulation 1: "
                           "Pea protein −2.10 g, Methylcellulose +0.80 g.")
               for c in at.caption), [c.value for c in at.caption]


def test_both_downloads_are_offered_and_only_the_batch_sheet_is_lit(open_batch):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert _unknown(at.main, "download_button",
                    "Download batch sheet (CSV)").proto.type == "primary"
    assert _unknown(at.main, "download_button",
                    "Download formulation sheets (to print)").proto.type == "secondary"
    # The download IS the coloured thing here, and it is the only one.
    assert _tab_primaries(at, 1) == ["Download batch sheet (CSV)"], _tab_primaries(at, 1)
    at.number_input(key="f1_Firmness").set_value(6.0)
    at.run()
    assert _unknown(at.main, "download_button",
                    "Download batch sheet (CSV)").proto.type == "secondary"


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
                    "Download batch sheet (CSV)").proto.type == "primary"


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
    assert any(e.label == "Preview formulation sheets" for e in at.expander)
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
    # The panel unit is on the label, once, and never after the number.
    assert (at.number_input(key="f1_Juiciness").label
            == "Juiciness (/10) · target 7")
    labels = [n.label for n in at.number_input]
    assert (labels.index("Firmness · target 6 N")
            < labels.index("Juiciness (/10) · target 7"))
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
    assert any(c.value == "Results for 1 of 2 formulations"
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
    assert any(c.value == "Results for 1 of 1 formulation · 1 not made"
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


def test_an_uploaded_sheet_whose_final_write_fails_stays_on_the_batch(open_batch):
    """Twin of the typed path: closing the batch is a write like any other, and
    a green "Batch 1 recorded." over a batch still open on disk is a lie."""
    import storage as storage_backend

    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "burger"
    at.session_state["main_tab"] = "2 · Make a batch"
    at.session_state["_results_upload"] = pd.DataFrame(
        {"Formulation": [1, 2], "Firmness": [6.0, 4.0],
         "Juiciness": [7.0, 5.0]})
    at.run()
    real_save = at.session_state["optimizer"].storage.save
    calls = []

    def flaky(name, state):
        calls.append(name)
        if len(calls) == 3:      # the write that closes the batch
            raise storage_backend.StorageError(
                "The disk is full — your last change was NOT saved.")
        return real_save(name, state)

    at.session_state["optimizer"].storage.save = flaky
    _submit_button(at, "Save uploaded results").click()
    at.run()
    assert not at.exception
    assert any("NOT saved" in e.value for e in at.error), [e.value for e in at.error]
    assert not any("recorded" in s.value for s in at.success), \
        [s.value for s in at.success]
    assert at.session_state["main_tab"] == "2 · Make a batch"
    assert FoodOptimizer("burger").pending_batch is not None


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
    assert any(c.value == "Firmness 6 N · Juiciness (/10) 7" for c in at.caption), \
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
    _submit_button(at, "Empty this project").click()
    at.run()
    _submit_button(at, "Yes, empty it").click()
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
                    "Download batch sheet (CSV)").proto.type == "primary"
    _submit_button(at, "Generate a different batch").click()
    at.run()
    assert _unknown(at.main, "download_button",
                    "Download batch sheet (CSV)").proto.type == "secondary"
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
    assert at.number_input(key="scale_total").proto.placeholder == "as generated"
    texts = [t.value for t in at.text]
    # Two decimals on every line: a sheet is read down the column.
    assert "Pea protein: 20.00 g" in texts, texts     # the sheet, as generated
    assert "Methylcellulose: 1.00 g" in texts, texts


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
    assert any(c.value == "Amounts shown for this total."
               for c in at.caption), [c.value for c in at.caption]
    assert any(c.value == "Sheets use a batch size of 11 g."
               for c in at.caption), [c.value for c in at.caption]
    # Both sheets now carry the same amounts, so the downloads followed.
    assert [t.value for t in at.text].count("Pea protein: 10.00 g") == 2
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
    # "Not made" first: the row is otherwise blank, and nothing else on it
    # says the formulation was left out.
    assert FoodOptimizer("burger").skipped[0]["note"] == "Not made · burner failed"


def test_a_recorded_row_shows_its_note(open_batch):
    open_batch.tell({"Pea protein": 10.0, "Methylcellulose": 1.0},
                    {"Juiciness": 7.0, "Firmness": 6.0},
                    formulation_no=1, batch_no=1, note="held together")
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert any(c.value == "Firmness 6 N · Juiciness (/10) 7 · Note: held together"
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
    assert list(off_by["Measurement"]) == ["Firmness", "Juiciness (/10)"]
    assert off_by["Off by"].iloc[0] == "2 N too high"
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


def test_the_first_batch_is_not_announced_twice_on_one_screen(scored):
    """Saving flashes "Batch 1 recorded." above the tabs. A caption under the
    heading saying the same thing is the same sentence twice on one screen,
    and there is nothing to compare a first batch with anyway."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert not any(c.value == "Batch 1 recorded." for c in at.caption), \
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
    # A row nobody made has no score to be best, and says so in that column.
    assert list(table["Best"]) == ["★", "", "not made"]
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
    # Every number wears its unit: the flash is the only confirmation that
    # the right reading was typed.
    assert note == ("Formulation 1 Firmness corrected 1 N → 6 N. Best moved "
                    "from Formulation 2 to Formulation 1. A copy is saved "
                    "in your FoodOptimizer folder first.")
    assert at.session_state["main_tab"] == "3 · Results"    # no auto-move


def test_a_correction_can_be_closed_again(scored):
    """A selectbox opened with index=None cannot be cleared by the user, so
    the edit row would sit there for the rest of the session."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.selectbox(key="correct_formulation").set_value(1)
    at.run()
    assert "correct_1_Firmness" in [n.key for n in at.number_input]
    _submit_button(at, "Close").click()
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
    _submit_button(at, "Empty this project").click()
    at.run()
    button = _submit_button(at, "Make your first batch")
    assert button.proto.type == "secondary" and button.disabled


def test_results_collapsed_sections_are_the_spec_list(scored):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    labels = [e.label for e in at.expander]
    # One section holds both ways a formulation leaves the project.
    for label in ("Progress chart", "Remove a batch or a formulation",
                  "Import past formulations from a CSV"):
        assert label in labels, labels


def test_undo_the_last_batch_says_it_once_keeps_a_copy_and_removes_it(scored, tmp_path):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    # Formulations, not results: one of the three was left out and has none.
    sentence = "Removes batch 1 and its 3 formulations. A copy is saved in your FoodOptimizer folder first."
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
    _submit_button(at, "Remove Formulation 1").click()
    at.run()
    assert any("Later formulations keep their numbers. A copy is saved in your FoodOptimizer folder first."
               in w.value for w in at.warning), [w.value for w in at.warning]
    # One Cancel on this tab, not the confirmation's and the select box's
    # side by side.
    tab_labels = [b.label for b in at.tabs[2].button]
    assert tab_labels.count("Cancel") == 1, tab_labels
    _submit_button(at, "Yes, remove").click()
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


def test_the_best_amounts_table_carries_each_rows_own_unit(burger):
    """One Amount column formatted per row: a cook temperature is not grams,
    and settings are listed after the ingredients, however large they are."""
    burger.add_process_parameter("Cook temperature", 0, 220)
    burger.tell({"Pea protein": 20.0, "Methylcellulose": 2.0,
                 "Cook temperature": 200.0},
                {"Juiciness": 7.0, "Firmness": 6.0},
                formulation_no=1, batch_no=1)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    table = next(t.value for t in at.table
                 if "Ingredient or setting" in t.value.columns)
    assert list(table.columns) == ["Ingredient or setting", "Amount"]
    names = list(table["Ingredient or setting"])
    assert names == ["Pea protein", "Methylcellulose", "Cook temperature"], names
    amounts = dict(zip(names, table["Amount"]))
    # Two decimals on an amount, and a setting is not an amount.
    assert amounts["Pea protein"] == "20.00 g"
    assert amounts["Cook temperature"] == "200"


def test_only_one_confirmation_can_be_armed_at_a_time(scored):
    """Two armed confirmations would put two coloured Yes buttons on the tab
    and leave the user guessing which copy is kept."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.selectbox(key="delete_formulation").set_value(1)
    at.run()
    _submit_button(at, "Undo the last batch").click()
    at.run()
    assert _tab_primaries(at, 2) == ["Yes, undo"], _tab_primaries(at, 2)
    assert _submit_button(at, "Remove Formulation 1").disabled


def test_a_correction_box_says_a_blank_keeps_the_recorded_value(scored):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.selectbox(key="correct_formulation").set_value(1)
    at.run()
    assert (at.number_input(key="correct_1_Firmness").help
            == "Leave blank to keep the value already recorded.")


def test_an_import_outside_the_scale_is_refused_naming_the_row(burger):
    """A 99 typed into a 0-10 column would sail in as the best formulation."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "burger"
    at.session_state["_import_rows"] = pd.DataFrame({
        "Pea protein": [12.0, 13.0], "Methylcellulose": [1.2, 1.3],
        "Juiciness": [6.0, 6.5], "Firmness": [5.0, 99.0],
    })
    at.run()
    _submit_button(at, "Import all rows").click()
    at.run()
    assert not at.exception
    assert any(e.value == ("Row 2: Firmness 99 N is outside your scale of 0 to "
                           "10 N. Widen the scale in Set up, or check the "
                           "value.") for e in at.error), [e.value for e in at.error]
    assert FoodOptimizer("burger").X_history == []   # the whole file is refused


def test_an_import_outside_an_ingredient_range_warns_and_still_imports(burger):
    """An amount outside the range is a fact about work already done, not a
    mistake to refuse: the model needs it."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "burger"
    at.session_state["_import_rows"] = pd.DataFrame({
        "Pea protein": [12.0, 999.0], "Methylcellulose": [1.2, 1.3],
        "Juiciness": [6.0, 6.5], "Firmness": [5.0, 5.5],
    })
    at.run()
    _submit_button(at, "Import all rows").click()
    at.run()
    assert not at.exception
    assert any(w.value == ("Row 2: Pea protein 999 g is outside its allowed amounts of "
                           "0 to 25 g.") for w in at.warning), \
        [w.value for w in at.warning]
    assert len(FoodOptimizer("burger").X_history) == 2


def test_a_finished_import_does_not_offer_to_import_again(burger):
    """The parse used to run on every rerun, so a second click after a
    successful import recorded every row twice."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "burger"
    at.session_state["_import_rows"] = pd.DataFrame({
        "Pea protein": [12.0, 13.0], "Methylcellulose": [1.2, 1.3],
        "Juiciness": [6.0, 6.5], "Firmness": [5.0, 5.5],
    })
    at.run()
    _submit_button(at, "Import all rows").click()
    at.run()
    assert "_import_rows" not in at.session_state
    at.run()
    assert not any(b.label == "Import all rows" for b in at.button), _labels(at)
    assert len(FoodOptimizer("burger").X_history) == 2


def test_undo_takes_a_last_batch_that_was_entirely_left_out(burger):
    burger.tell({"Pea protein": 10.0, "Methylcellulose": 1.0},
                {"Juiciness": 7.0, "Firmness": 6.0},
                formulation_no=1, batch_no=1)
    burger.record_skipped(2, 2, {"Pea protein": 20.0, "Methylcellulose": 2.0})
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    _submit_button(at, "Undo the last batch").click()
    at.run()
    assert any(w.value == "Removes batch 2 and its 1 formulation. A copy is "
                          "saved in your FoodOptimizer folder first."
               for w in at.warning), \
        [w.value for w in at.warning]
    _submit_button(at, "Yes, undo").click()
    at.run()
    assert not at.exception
    reloaded = FoodOptimizer("burger")
    assert reloaded.skipped == [] and reloaded.formulation_ids == [1]


def test_the_desktop_bundle_ships_every_module():
    root = pathlib.Path(APP_PATH).resolve().parent
    build = (root / "desktop" / "build_dmg.sh").read_text()
    e2e = (root / "desktop" / "test_e2e.sh").read_text()
    for module in ("ui_setup.py", "ui_batch.py", "ui_results.py"):
        assert module in build, module
        assert module in e2e, module


_FIRST_RUN_SENTENCE = ("On a good connection this takes under a minute; "
                       "on a slow office network up to 15 minutes.")


def _flowed(text):
    """A file's prose as one line. A sentence is wrapped differently in each
    file — Swift splits it across concatenated literals, Markdown across
    blockquote lines, Start Here across indented ones — and the sentence, not
    the wrapping, is what has to be identical everywhere."""
    import re
    joined = re.sub(r'"\s*\+\s*"', "", text)          # Swift: "a " + "b"
    joined = re.sub(r"^[>\s]+", " ", joined, flags=re.M)  # quote / indent
    return re.sub(r"\s+", " ", joined)


def test_first_run_copy_gives_the_honest_timing():
    """One sentence, verbatim, in all four places a first-run user reads it."""
    root = pathlib.Path(APP_PATH).resolve().parent
    for name in ("desktop/FoodOptimizerApp.swift", "desktop/start_here.txt",
                 "desktop/README.md", "README.md"):
        text = (root / name).read_text()
        assert _FIRST_RUN_SENTENCE in _flowed(text), name
        assert "a few minutes, up to 15" not in text, name


def test_the_last_batch_line_counts_a_batch_that_was_entirely_left_out(burger):
    """Nobody managed to make batch 2, but it is still the last batch: the line
    under the title must not fall back to batch 1."""
    burger.tell({"Pea protein": 10.0, "Methylcellulose": 1.0},
                {"Juiciness": 7.0, "Firmness": 6.0},
                formulation_no=1, batch_no=1)
    burger.record_skipped(2, 2, {"Pea protein": 20.0, "Methylcellulose": 2.0})
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert not at.exception
    assert any(c.value == "Batch 2 · recorded" for c in at.caption), \
        [c.value for c in at.caption]


def test_a_stale_batch_is_reported_above_the_tabs_not_in_the_sidebar(open_batch,
                                                                     tmp_path):
    """The notice is about the batch the user is looking at, so it belongs in
    the fixed message container above the tabs."""
    import json
    notice = ("The open batch was discarded because the ingredient list or "
              "its allowed amounts changed since it was generated.")
    state = json.loads((tmp_path / "burger.pkl").read_text())
    for row in state["pending_batch"]:
        row["recipe"].pop("Methylcellulose")
    (tmp_path / "burger.pkl").write_text(json.dumps(state))
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "burger"
    at.run()
    assert not at.exception
    assert any(i.value == notice for i in at.info), [i.value for i in at.info]
    assert not any(i.value == notice for i in at.sidebar.info), \
        [i.value for i in at.sidebar.info]
    assert FoodOptimizer("burger").pending_batch is None


def test_advanced_and_limit_forms_do_not_follow_you_to_another_project(burger):
    """prop_ and bo_ keys belong to the project they were typed in, exactly as
    the measurement and ingredient forms do."""
    FoodOptimizer("second").save()
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "burger"
    at.run()
    at.session_state["bo_json"] = '{"kernel": "matern52"}'
    at.session_state["prop_min"] = 3.0
    at.run()
    at.sidebar.selectbox(key="project_select").select("second")
    at.run()
    _submit_button(at.sidebar, "Open").click()
    at.run()
    assert not at.exception
    assert "bo_json" not in at.session_state
    # A box that is still on screen in the new project is emptied rather than
    # popped: popping does not reach the browser, which posts the old value
    # straight back on the next click.
    assert at.session_state["prop_min"] is None


def test_a_correction_keeps_a_copy_first_and_says_so(scored, tmp_path):
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
    assert (tmp_path / "burger_pre_edit.pkl").exists(), \
        [p.name for p in tmp_path.glob("*.pkl")]
    assert any(s.value == ("Formulation 1 Firmness corrected 1 N → 6 N. "
                           "Best moved from Formulation 2 to Formulation 1. "
                           "A copy is saved in your FoodOptimizer folder first.")
               for s in at.success), [s.value for s in at.success]
    assert FoodOptimizer("burger").results_history[0]["Firmness"] == 6.0


def test_a_no_op_correction_says_so_and_keeps_no_copy(scored, tmp_path):
    """Saving a row nobody changed writes nothing, so it must not claim a copy
    was kept — or keep one."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "burger"
    at.session_state["main_tab"] = "3 · Results"
    at.run()
    at.selectbox(key="correct_formulation").set_value(1)
    at.run()
    _submit_button(at, "Save correction").click()      # the boxes hold what
    at.run()                                           # was already recorded
    assert not at.exception
    assert any(s.value == "Formulation 1 is unchanged." for s in at.success), \
        [s.value for s in at.success]
    assert not (tmp_path / "burger_pre_edit.pkl").exists(), \
        [p.name for p in tmp_path.glob("*.pkl")]


def test_a_confirmation_takes_the_colour_from_save_correction(scored):
    """Two coloured buttons on one tab, even for one frame: arming a
    confirmation below the correction row must take the colour from it on the
    same run, which is why the button row is drawn last."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "burger"
    at.session_state["main_tab"] = "3 · Results"
    at.run()
    at.selectbox(key="correct_formulation").set_value(1)
    at.run()
    assert _tab_primaries(at, 2) == ["Save correction"], _tab_primaries(at, 2)
    _submit_button(at, "Undo the last batch").click()
    at.run()
    assert _tab_primaries(at, 2) == ["Yes, undo"], _tab_primaries(at, 2)
    save = _submit_button(at, "Save correction")
    assert save.proto.type == "secondary" and save.disabled
    _submit_button(at, "Cancel").click()
    at.run()
    at.run()
    at.selectbox(key="delete_formulation").set_value(1)
    at.run()
    _submit_button(at, "Remove Formulation 1").click()
    at.run()
    assert _tab_primaries(at, 2) == ["Yes, remove"], _tab_primaries(at, 2)


def test_save_correction_is_the_one_lit_action_while_the_row_is_open(scored):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "burger"
    at.session_state["main_tab"] = "3 · Results"
    at.run()
    assert _tab_primaries(at, 2) == ["Start the next batch"], _tab_primaries(at, 2)
    at.selectbox(key="correct_formulation").set_value(1)
    at.run()
    assert _tab_primaries(at, 2) == ["Save correction"], _tab_primaries(at, 2)
    foot = _submit_button(at, "Start the next batch")
    assert foot.proto.type == "secondary" and foot.disabled


def test_a_half_typed_measurement_does_not_follow_you_to_another_project(burger):
    """Streamlit keeps a widget's value under its key for the whole session, so
    without a clean-up on switch, project B's form opens holding A's typing."""
    FoodOptimizer("second").save()              # somewhere to switch to
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "burger"
    at.run()
    at.text_input(key="meas_new_name").set_value("Chewiness")
    at.run()
    assert at.session_state["meas_new_name"] == "Chewiness"
    at.sidebar.selectbox(key="project_select").select("second")
    at.run()
    _submit_button(at.sidebar, "Open").click()
    at.run()
    assert not at.exception
    assert at.session_state["_loaded_project"] == "second"
    assert at.session_state["meas_new_name"] == ""
    assert at.text_input(key="meas_new_name").value == ""


def test_only_one_confirmation_can_be_armed_on_the_set_up_tab(burger):
    """Two armed confirmations would put two coloured Yes buttons on one tab,
    each quietly keeping its own copy of the project."""
    burger.add_process_parameter("Cook temp", 150, 200)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "burger"
    at.run()
    # Remove renders first, so arming it is the case that has to hold within
    # one render: confirmation_open() is read by every later site.
    arming = ["rm_var_Pea protein__btn", "rm_meas_Firmness__btn",
              "rm_meas_Juiciness__btn"]
    assert all(not at.button(key=k).disabled for k in arming)
    at.button(key="rm_var_Pea protein__btn").click()
    at.run()
    assert not at.exception
    assert at.button(key="rm_var_Pea protein__btn").disabled is False
    assert all(at.button(key=k).disabled for k in arming[1:]), \
        [(k, at.button(key=k).disabled) for k in arming]
    assert _tab_primaries(at, 0) == ["Yes, remove"], _tab_primaries(at, 0)


def test_an_uploaded_sheet_stops_at_the_first_row_that_did_not_save(open_batch,
                                                                    monkeypatch):
    """Half a sheet on disk under a green 'Batch 1 recorded.' is the worst
    outcome: the save loop must stop and say so at the first failure."""
    import storage as storage_backend

    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "burger"
    at.session_state["main_tab"] = "2 · Make a batch"
    at.session_state["_results_upload"] = pd.DataFrame(
        {"Formulation": [1, 2], "Firmness": [6.0, 4.0],
         "Juiciness": [7.0, 5.0]})
    at.run()
    real_save = at.session_state["optimizer"].storage.save
    calls = []

    def flaky(name, state):
        calls.append(name)
        if len(calls) == 2:
            raise storage_backend.StorageError(
                "The disk is full — your last change was NOT saved.")
        return real_save(name, state)

    at.session_state["optimizer"].storage.save = flaky
    _submit_button(at, "Save uploaded results").click()
    at.run()
    assert not at.exception
    assert any("NOT saved" in e.value for e in at.error), [e.value for e in at.error]
    assert not any("recorded" in s.value for s in at.success), \
        [s.value for s in at.success]
    # One row reached the disk; the second never did, and the batch stays open.
    reloaded = FoodOptimizer("burger")
    assert reloaded.formulation_ids == [1]
    assert at.session_state["main_tab"] == "2 · Make a batch"


# ------------------------------------------------------------------ #
#  Verification round: the fixes the four-agent pass asked for
# ------------------------------------------------------------------ #

def test_the_line_under_the_title_is_on_tab_one_only(open_batch):
    """Tab 2 carries the batch's own heading. The caption sat directly above
    `Batch 1 · make these 2 formulations`, saying the same thing twice."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert not at.exception
    assert any(c.value == "Batch 1 · 2 to make" for c in at.tabs[0].caption), \
        [c.value for c in at.tabs[0].caption]
    assert not any(c.value == "Batch 1 · 2 to make" for c in at.tabs[1].caption), \
        [c.value for c in at.tabs[1].caption]
    assert any(m.value == "**Batch 1 · make these 2 formulations**"
               for m in at.tabs[1].markdown)


def test_a_hard_reset_leaves_the_project_open_and_in_the_list(project_with_history,
                                                              tmp_path):
    """The archive is a copy of what was there; the project itself stays,
    empty, under its own name. It used to be renamed away, so it vanished
    from Open project while the sidebar still named it."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    _submit_button(at, "Empty this project").click()
    at.run()
    _submit_button(at, "Yes, empty it").click()
    at.run()
    assert not at.exception
    assert (tmp_path / "my_project_archived.pkl").exists()
    assert (tmp_path / "my_project.pkl").exists(), list(tmp_path.iterdir())
    assert FoodOptimizer("my_project").X_history == []
    assert at.session_state["_loaded_project"] == "my_project"
    assert at.sidebar.selectbox(key="project_select").value == "my_project"
    # One more run: AppTest keeps the answered confirmation's buttons in its
    # local element tree until the next full render.
    at.run()
    # Nothing in the sidebar is lit: the project on screen is the one open.
    assert [b.label for b in at.sidebar.button if b.proto.type == "primary"] == []
    assert _submit_button(at.sidebar, "Open").disabled


def test_every_confirmation_counts_the_left_out_formulations(scored):
    """Two scored and one left out is three formulations to archive."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    _submit_button(at, "Empty this project").click()
    at.run()
    assert any("Its 3 formulations and set-up go" in w.value
               for w in at.warning), [w.value for w in at.warning]
    _submit_button(at, "Cancel").click()
    at.run()
    _submit_button(at, "Delete this project").click()
    at.run()
    assert any("and its 3 formulations?" in w.value for w in at.warning), \
        [w.value for w in at.warning]


def test_save_changes_is_the_lit_action_while_a_measurement_is_open(burger):
    """Continue to make a batch would leave the tab and throw the edit away,
    exactly as tab 3's correction row already refuses to let happen."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    _submit_button(at, "Edit Firmness").click()
    at.run()
    assert _tab_primaries(at, 0) == ["Save changes"], _tab_primaries(at, 0)
    assert _submit_button(at, "Continue to make a batch").disabled


def test_the_empty_results_button_sends_an_unready_project_to_set_up(tmp_path,
                                                                     monkeypatch):
    """A project with no ingredients cannot make a batch: the lit button used
    to land on a tab holding a greyed Generate."""
    monkeypatch.chdir(tmp_path)
    FoodOptimizer("empty").save()
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "empty"
    at.session_state["main_tab"] = "3 · Results"
    at.run()
    assert not any(b.label == "Make your first batch" for b in at.button), \
        _labels(at)
    _submit_button(at, "Set up this project").click()
    at.run()
    assert not at.exception
    assert at.session_state["main_tab"] == "1 · Set up"


def test_a_project_saved_before_units_existed_says_the_g_is_a_guess(tmp_path,
                                                                    monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("legacy")
    opt.add_ingredient("Water", 0, 100)
    opt.add_objective("Taste", 1.0, goal="max")
    state = opt.export_json()
    state.pop("amount_unit")                 # a 0.2.x file has no unit at all
    opt.storage.save("legacy", state)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "legacy"
    at.run()
    assert not at.exception
    assert any(c.value == ("Made before units were recorded; amounts are in "
                           "g. Set each unit below if that is wrong.")
               for c in at.caption), [c.value for c in at.caption]
    # ...and it is a caption like every other on the tab: one line, said once.
    _assert_captions_read_once(at)


def test_undo_explains_a_history_that_belongs_to_no_batch(burger):
    """`No batch to undo yet.` under four recorded formulations reads as a
    contradiction; it points at the one way back instead."""
    burger.import_formulation({"Pea protein": 10.0, "Methylcellulose": 1.0},
                              {"Juiciness": 7.0, "Firmness": 6.0})
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "burger"
    at.run()
    assert any(c.value == ("These formulations were recorded before batches "
                           "existed, so there is no batch to undo. You can "
                           "remove one formulation at a time below.")
               for c in at.caption), [c.value for c in at.caption]


def test_a_process_setting_carries_its_own_unit_everywhere(burger):
    """Sheets printed `Cook temperature: 180` and the batch column was bare:
    a setting is not an amount, so it needs a unit of its own."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.radio(key="var_kind").set_value("Process setting")
    at.run()
    at.text_input(key="var_name").set_value("Cook temperature")
    at.text_input(key="var_unit").set_value("°C")
    at.number_input(key="var_low").set_value(160.0)
    at.number_input(key="var_high").set_value(200.0)
    at.run()
    _submit_button(at, "Add ingredient or setting").click()
    at.run()
    assert not at.exception
    reloaded = FoodOptimizer("burger")
    setting = next(v for v in reloaded.variables if v.get('category') == 'process')
    assert setting["unit"] == "°C"
    table = next(d.value for d in at.dataframe if "Kind" in d.value.columns)
    row = table[table["Name"] == "Cook temperature"].iloc[0]
    assert (row["Kind"], row["Lowest"], row["Highest"], row["Unit"]) == \
        ("Process setting", 160.0, 200.0, "°C")
    # ... on the batch table, the printable sheet and the amounts table.
    reloaded.set_pending_batch([{"Pea protein": 10.0, "Methylcellulose": 1.0,
                                 "Cook temperature": 180.0}])
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    table = next(d.value for d in at.dataframe if "Formulation" in d.value.columns)
    assert "Cook temperature (°C)" in table.columns, list(table.columns)
    assert any(t.value == "Cook temperature: 180 °C" for t in at.text), \
        [t.value for t in at.text]


def test_the_baseline_of_a_process_setting_is_shown_with_its_unit(burger):
    burger.tell({"Pea protein": 10.0, "Methylcellulose": 1.0},
                {"Juiciness": 7.0, "Firmness": 6.0}, formulation_no=1, batch_no=1)
    burger.add_process_parameter("Cook temperature", 160, 200, baseline=180,
                                 unit="°C")
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    table = next(d.value for d in at.dataframe if "Kind" in d.value.columns)
    assert dict(zip(table["Name"], table["Used so far"])) == {
        "Pea protein": "", "Methylcellulose": "",
        "Cook temperature": "180 °C"}


def test_the_limits_caption_covers_a_property_named_in_the_app(with_properties):
    """A property is named in the app as often as it arrives in a file, and
    the box that names one is two lines below this caption."""
    with_properties.add_property("Sodium mg per 100 g")
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert any(c.value == ("Limits hold every new formulation to an amount "
                           "you weigh out or a property of your ingredients. "
                           "Measurements are aimed at with targets, not "
                           "limited.") for c in at.caption), \
        [c.value for c in at.caption]


def test_a_project_with_no_properties_still_offers_to_name_one(burger):
    """It used to send the user off to build a CSV; a property can now be
    named here, so the section offers the box instead of an errand."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert not any("Upload an ingredient CSV with extra columns" in c.value
                   for c in at.caption), [c.value for c in at.caption]
    assert at.text_input(key="prop_new").label == \
        "Add a property, such as Sodium per 100 g"
    # Nothing to limit yet, so no property picker and no limit button.
    assert "prop_metric" not in [b.key for b in at.selectbox]
    assert "Add property limit" not in _labels(at)


def test_the_sample_project_is_created_not_opened(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    _submit_button(at.main, "Try the sample project").click()
    at.run()
    assert not at.exception
    assert any(s.value == "Created Sample project." for s in at.success), \
        [s.value for s in at.success]
    # And opening it again later is an opening.
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    _submit_button(at.sidebar, "Try the sample project").click()
    at.run()
    assert any(s.value == "Opened Sample project." for s in at.success), \
        [s.value for s in at.success]


def test_the_delete_picker_says_what_its_own_cancel_does(scored):
    """Two buttons called Cancel in one flow: one clears the picker, the
    other abandons the confirmation."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.selectbox(key="delete_formulation").set_value(1)
    at.run()
    assert "Close" in _labels(at), _labels(at)
    _submit_button(at, "Close").click()
    at.run()
    assert not at.exception
    assert at.selectbox(key="delete_formulation").value is None


def test_a_partly_uploaded_sheet_says_what_it_recorded(open_batch):
    """The whole-batch sentence over a batch that is still open claimed work
    nobody had done."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "burger"
    at.session_state["main_tab"] = "2 · Make a batch"
    at.session_state["_results_upload"] = pd.DataFrame(
        {"Formulation": [2], "Firmness": [6.0], "Juiciness": [7.0]})
    at.run()
    _submit_button(at, "Save uploaded results").click()
    at.run()
    assert not at.exception
    assert any(s.value == "Recorded 1 of 2 formulations in batch 1 · 1 to make."
               for s in at.success), [s.value for s in at.success]
    assert at.session_state["main_tab"] == "2 · Make a batch"
    assert FoodOptimizer("burger").pending_batch is not None


def test_an_edit_with_nothing_recorded_claims_no_recalculation(burger):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    _submit_button(at, "Edit Firmness").click()
    at.run()
    at.number_input(key="meas_Firmness_importance").set_value(2.0)
    _submit_button(at, "Save changes").click()
    at.run()
    assert not at.exception
    assert any(s.value == ("Firmness importance changed to 2.0. A copy is "
                           "saved in your FoodOptimizer folder first.")
               for s in at.success), \
        [s.value for s in at.success]


def test_a_half_typed_ingredient_does_not_follow_you_to_another_project(burger):
    """Popping the key does not reach the browser: the mounted box posts its
    old value back, and Add then added it to the other project."""
    FoodOptimizer("second").save()
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "burger"
    at.run()
    at.text_input(key="var_name").set_value("Half typed ingredient")
    at.number_input(key="var_high").set_value(7.0)
    at.run()
    at.sidebar.selectbox(key="project_select").select("second")
    at.run()
    _submit_button(at.sidebar, "Open").click()
    at.run()
    assert not at.exception
    assert at.session_state["var_name"] == ""
    assert at.session_state["var_high"] == 100.0
    assert at.text_input(key="var_name").value == ""


def test_the_result_grid_does_not_follow_you_to_another_project(open_batch):
    """f1_Firmness is a live key in every project that has a formulation 1."""
    FoodOptimizer("second").save()
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "burger"
    at.session_state["main_tab"] = "2 · Make a batch"
    at.run()
    at.number_input(key="f1_Firmness").set_value(9.0)
    at.text_input(key="f1_note").set_value("half typed")
    at.run()
    at.sidebar.selectbox(key="project_select").select("second")
    at.run()
    _submit_button(at.sidebar, "Open").click()
    at.run()
    assert not at.exception
    assert at.session_state["f1_Firmness"] is None
    assert at.session_state["f1_note"] == ""


def test_get_help_gives_an_address_to_write_to():
    """A user who threw away the disk image has no other route."""
    root = pathlib.Path(APP_PATH).resolve().parent
    swift = _flowed((root / "desktop" / "FoodOptimizerApp.swift").read_text())
    assert "https://github.com/thomasat/food_opt/issues" in swift
    assert ("Describe the problem in words, and do not attach project files, "
            "backups or formulations, because that page is public.") in swift


def test_emptying_the_forms_raises_no_widget_warning_on_screen(open_batch):
    """Streamlit puts a yellow box on the page when a widget carries both a
    `value=` and a session-state entry. Parking empty values across a project
    switch touches a dozen widgets, so none of them may declare a default."""
    FoodOptimizer("second").save()
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "burger"
    at.session_state["main_tab"] = "2 · Make a batch"
    at.run()
    at.number_input(key="f1_Firmness").set_value(9.0)
    at.number_input(key="scale_total").set_value(50.0)
    at.run()
    at.sidebar.selectbox(key="project_select").select("second")
    at.run()
    _submit_button(at.sidebar, "Open").click()
    at.run()
    assert not at.exception
    noisy = [w.value for w in at.warning if "Session State API" in w.value]
    assert noisy == [], noisy
    assert at.session_state["scale_total"] is None


def test_a_deleted_project_leaves_the_sidebar_box(project_with_history):
    """The box kept the deleted project's name — popping a widget's key does
    not reach the browser — so Open lit up over a project that was gone."""
    FoodOptimizer("keeper").save()
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "my_project"
    at.run()
    _submit_button(at, "Delete this project").click()
    at.run()
    _submit_button(at, "Yes, delete it").click()
    at.run()
    assert not at.exception
    assert at.session_state["_loaded_project"] == "keeper"
    assert at.sidebar.selectbox(key="project_select").value == "keeper"
    at.run()      # AppTest keeps the answered confirmation's buttons a run
    assert [b.label for b in at.sidebar.button if b.proto.type == "primary"] == []


def test_a_process_setting_is_rounded_wherever_it_is_shown(burger):
    """188.494 °C is a precision no oven dial has. The batch table, the
    printable sheet and the amounts table must round it the same way."""
    burger.add_process_parameter("Cook temperature", 100, 220, unit="°C")
    burger.set_pending_batch([{"Pea protein": 10.0, "Methylcellulose": 1.0,
                               "Cook temperature": 188.4936}])
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert not at.exception
    shown = _displayed(next(d for d in at.dataframe
                            if "Formulation" in d.value.columns))
    assert shown["Cook temperature (°C)"].iloc[0] == "188.49", shown.to_dict()
    assert any(t.value == "Cook temperature: 188.49 °C" for t in at.text), \
        [t.value for t in at.text]
    # ... and on tab 3, once it is recorded.
    burger.tell({"Pea protein": 10.0, "Methylcellulose": 1.0,
                 "Cook temperature": 188.4936},
                {"Juiciness": 7.0, "Firmness": 6.0},
                formulation_no=1, batch_no=1)
    burger.set_pending_batch(None)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    table = next(t.value for t in at.table
                 if "Ingredient or setting" in t.value.columns)
    amounts = dict(zip(table["Ingredient or setting"], table["Amount"]))
    assert amounts["Cook temperature"] == "188.49 °C", amounts


# ------------------------------------------------------------------ #
#  Units per ingredient (owner amendment, 2026-09-10)
# ------------------------------------------------------------------ #

@pytest.fixture
def mixed_units(tmp_path, monkeypatch):
    """A project whose ingredients are not all in one unit: the powder is
    weighed in grams, the water is measured in millilitres."""
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("mixed")
    opt.add_ingredient("Pea protein", 0, 25)
    opt.add_ingredient("Water", 0, 60, unit="ml")
    opt.add_objective("Firmness", 1.0, goal="target", target=6,
                      min_val=0, max_val=10, unit="N")
    return opt
def test_the_ingredient_table_has_plain_headers_and_a_unit_column(mixed_units):
    """Min (g) over a row measured in ml was a lie. The headers are plain and
    each row says what it is measured in."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    table = next(d.value for d in at.dataframe if "Name" in d.value.columns)
    assert list(table.columns) == ["Kind", "Name", "Lowest", "Highest", "Unit"]
    assert dict(zip(table["Name"], table["Unit"])) == {"Pea protein": "g",
                                                       "Water": "ml"}


def test_a_new_ingredient_is_added_with_its_own_unit(burger):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert at.text_input(key="var_unit").value == "g"      # ingredients are in g
    assert at.number_input(key="var_low").label == "Lowest"
    assert at.number_input(key="var_high").label == "Highest"
    at.text_input(key="var_name").set_value("Water")
    at.text_input(key="var_unit").set_value("ml")
    at.number_input(key="var_high").set_value(60.0)
    at.run()
    _submit_button(at, "Add ingredient or setting").click()
    at.run()
    assert not at.exception
    assert FoodOptimizer("burger").unit_of("Water") == "ml"


def test_set_unit_changes_one_ingredient_and_says_so(burger):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.selectbox(key="var_pick").select("Methylcellulose")
    at.text_input(key="unit_value").set_value("mg")
    button = _submit_button(at, "Set unit")
    assert button.proto.type == "secondary"     # the foot keeps the colour
    button.click()
    at.run()
    assert not at.exception
    assert any("Methylcellulose is now written in mg. The amounts were not converted." in s.value
               for s in at.success), [s.value for s in at.success]
    reloaded = FoodOptimizer("burger")
    assert reloaded.unit_of("Methylcellulose") == "mg"
    assert reloaded.unit_of("Pea protein") == "g"
    # The box empties: a unit left in it is one click away from being
    # applied to the next ingredient chosen.
    assert at.text_input(key="unit_value").value == ""


def test_the_uploader_says_the_unit_column_is_optional(burger):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    uploader = _unknown(at.main, "file_uploader", "Upload ingredients CSV")
    assert "Unit" in uploader.proto.help, uploader.proto.help


def test_the_batch_table_carries_each_unit_and_a_per_unit_total(mixed_units):
    mixed_units.set_pending_batch([{"Pea protein": 10.0, "Water": 40.0}])
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert not at.exception
    frame = next(d for d in at.dataframe if "Formulation" in d.value.columns)
    assert list(frame.value.columns) == ["Formulation", "Pea protein (g)",
                                         "Water (ml)", "Total"]
    assert _displayed(frame)["Total"].iloc[0] == "10.00 g · 40.00 ml"


def test_scaling_is_offered_only_when_the_ingredients_share_a_unit(mixed_units):
    """Scaling 10 g of powder and 40 ml of water to a total of 400 means
    nothing, so the box is not offered and one line says why."""
    mixed_units.set_pending_batch([{"Pea protein": 10.0, "Water": 40.0}])
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert [n.key for n in at.number_input if n.key == "scale_total"] == []
    assert any(c.value == "A batch size needs all ingredients in one unit."
               for c in at.caption), [c.value for c in at.caption]
    # ... and it comes back the moment they do share one.
    mixed_units.set_ingredient_unit("Water", "g")
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert at.number_input(key="scale_total").label == \
        "Batch size (g)"


def test_a_scale_left_behind_does_not_rewrite_a_mixed_unit_batch(mixed_units):
    """The box is gone, but Streamlit keeps a hidden widget's last value. It
    must not go on scaling amounts nobody can see it acting on."""
    mixed_units.set_pending_batch([{"Pea protein": 10.0, "Water": 40.0}])
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["scale_total"] = 400.0
    at.run()
    frame = next(d for d in at.dataframe if "Formulation" in d.value.columns)
    assert _displayed(frame)["Total"].iloc[0] == "10.00 g · 40.00 ml"


def test_the_printable_sheet_writes_every_amount_in_its_own_unit(mixed_units):
    mixed_units.set_pending_batch([{"Pea protein": 10.0, "Water": 40.0}])
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    lines = [t.value for t in at.text]
    assert "Pea protein: 10.00 g" in lines, lines
    assert "Water: 40.00 ml" in lines, lines
    assert "Total: 10.00 g · 40.00 ml" in lines, lines


def test_the_biggest_changes_line_uses_each_ingredients_unit(mixed_units):
    mixed_units.tell({"Pea protein": 10.0, "Water": 30.0}, {"Firmness": 6.0},
                     formulation_no=1, batch_no=1)
    mixed_units.set_pending_batch([{"Pea protein": 12.0, "Water": 40.0}],
                                  batch_no=2)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    line = next((c.value for c in at.caption
                 if c.value.startswith("Biggest changes")), "")
    assert line == ("Biggest changes in Formulation 2 from Formulation 1: "
                    "Water +10.00 ml, Pea protein +2.00 g."), line


def test_the_amounts_to_make_it_table_uses_each_ingredients_unit(mixed_units):
    mixed_units.tell({"Pea protein": 10.0, "Water": 40.0}, {"Firmness": 6.0},
                     formulation_no=1, batch_no=1)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    table = next(t.value for t in at.table
                 if "Ingredient or setting" in t.value.columns)
    amounts = dict(zip(table["Ingredient or setting"], table["Amount"]))
    assert amounts == {"Water": "40.00 ml", "Pea protein": "10.00 g"}


def test_an_amount_limit_across_units_is_refused_on_screen(mixed_units):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.multiselect(key="qty_pick").select("Pea protein")
    at.multiselect(key="qty_pick").select("Water")
    at.number_input(key="qc_max").set_value(50.0)
    _submit_button(at, "Add amount limit").click()
    at.run()
    assert not at.exception
    assert [e.value for e in at.error] == ["Amount limits add amounts, so these ingredients need one unit; enter Water in g instead of ml."]
    assert FoodOptimizer("mixed").quantity_constraints == []


def test_a_paused_ingredient_is_held_at_a_value_in_its_own_unit(mixed_units):
    mixed_units.add_process_parameter("Cook temperature", 160, 200, unit="°C")
    mixed_units.deactivate_variable("Water", value=30.0)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    table = next(d.value for d in at.dataframe if "Kind" in d.value.columns)
    assert dict(zip(table["Name"], table["Status"])) == {
        "Pea protein": "active", "Water": "paused · held at 30.00 ml",
        "Cook temperature": "active"}


def test_the_set_unit_box_survives_a_switch_to_a_project_without_that_row(
        mixed_units, tmp_path):
    """A select box cannot show a value that is not one of its options, and
    the other project has no ingredient called Water: it must fall back to
    that project's first ingredient rather than break the tab."""
    other = FoodOptimizer("other")
    other.set_amount_unit("ml")
    other.add_ingredient("Flour", 0, 100)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "mixed"
    at.run()
    at.selectbox(key="var_pick").select("Water")
    at.text_input(key="var_unit").set_value("kg")
    at.run()
    at.sidebar.selectbox(key="project_select").select("other")
    at.run()
    _submit_button(at.sidebar, "Open").click()
    at.run()
    assert not at.exception
    assert at.selectbox(key="var_pick").value == "Flour"
    # ...and the add form opens on the NEW project's default, not on what was
    # half-typed in the old one.
    assert at.text_input(key="var_unit").value == "ml"


def test_an_amount_limit_is_listed_in_the_unit_it_limits(mixed_units):
    """A limit is a sum of ingredients that share a unit, so the number it
    holds them to is written in that unit."""
    mixed_units.add_quantity_constraint(["Water"], max_val=45)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert any(t.value == "Water: at most 45 ml" for t in at.text), \
        [t.value for t in at.text]


# ------------------------------------------------------------------ #
#  Projects made of process settings alone (owner ruling, 2026-09-10)
# ------------------------------------------------------------------ #

@pytest.fixture
def ferment(tmp_path, monkeypatch):
    """A fermentation project: nothing is weighed out, two settings are
    dialled in, and one measurement is scored."""
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("ferment")
    opt.add_process_parameter("Incubation temperature", 30, 42, unit="°C")
    opt.add_process_parameter("Incubation time", 4, 16, unit="h")
    opt.add_objective("Acidity", 1.0, goal="target", target=4.5,
                      min_val=3.0, max_val=7.0, unit="pH")
    return opt


def test_a_project_of_settings_alone_is_complete_and_lights_continue(ferment):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert not at.exception
    assert _tab_primaries(at, 0) == ["Continue to make a batch"]
    assert not any(c.value.startswith("Add at least one") for c in at.caption), \
        [c.value for c in at.caption]


def test_the_foot_names_a_setting_as_well_as_an_ingredient(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    FoodOptimizer("empty_one")
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "empty_one"
    at.run()
    assert any(c.value == "Add at least one ingredient or process setting."
               for c in at.caption), [c.value for c in at.caption]


def test_nothing_that_belongs_to_ingredients_shows_without_any(ferment):
    """No amounts are weighed out, so there is no total to show, nothing to
    scale to a total, and no amount limits to set."""
    ferment.set_pending_batch([{"Incubation temperature": 37.0,
                                "Incubation time": 8.0},
                               {"Incubation temperature": 40.0,
                                "Incubation time": 12.0}])
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert not at.exception
    frame = next(d for d in at.dataframe if "Formulation" in d.value.columns)
    assert list(frame.value.columns) == ["Formulation",
                                         "Incubation temperature (°C)",
                                         "Incubation time (h)"]
    assert [n.key for n in at.number_input if n.key == "scale_total"] == []
    assert not any("Scaling needs" in c.value for c in at.caption), \
        [c.value for c in at.caption]
    # The printable sheet lists the settings and claims no total.
    lines = [t.value for t in at.text]
    assert "Incubation temperature: 37 °C" in lines, lines
    assert not any(l.startswith("Total") for l in lines), lines
    # Both kinds of limit are about what you weigh out, so the whole section
    # is gone: a setting's own Lowest and Highest are its bounds.
    assert [e.label for e in at.expander if e.label == "Limits (optional)"] == []
    assert [b.label for b in at.button if b.label == "Add amount limit"] == []
    assert [b.label for b in at.button if b.label == "Add property limit"] == []


def test_a_settings_only_project_goes_round_the_whole_loop(ferment):
    """Generate, record two formulations, save, and read the best back."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    _submit_button(at, "Continue to make a batch").click()
    at.run()
    at.number_input(key="batch_size").set_value(2)
    at.run()
    _submit_button(at, "Generate 2 formulations").click()
    at.run()
    assert not at.exception
    made = FoodOptimizer("ferment")
    assert len(made.pending_batch) == 2
    at.number_input(key="f1_Acidity").set_value(4.5)
    at.number_input(key="f2_Acidity").set_value(5.5)
    at.run()
    assert _tab_primaries(at, 1) == ["Save results"], _tab_primaries(at, 1)
    _submit_button(at, "Save results").click()
    at.run()
    assert not at.exception
    assert at.session_state["main_tab"] == "3 · Results"
    assert any(h.value == "Best so far: Formulation 1 (batch 1)"
               for h in at.subheader), [h.value for h in at.subheader]
    table = next(t.value for t in at.table
                 if "Ingredient or setting" in t.value.columns)
    shown = dict(zip(table["Ingredient or setting"], table["Amount"]))
    assert set(shown) == {"Incubation temperature", "Incubation time"}
    assert shown["Incubation temperature"].endswith(" °C"), shown
    assert not any(c.value.startswith("Not used:") for c in at.caption), \
        [c.value for c in at.caption]


def test_a_unit_change_that_breaks_an_amount_limit_removes_it_and_says_so(
        burger):
    """A limit is a sum. Once one of its ingredients is measured in another
    unit the sum means nothing, so the limit goes and the line says which."""
    burger.add_ingredient("Salt", 0, 3)
    burger.add_quantity_constraint(["Pea protein", "Methylcellulose"],
                                   max_val=20)
    burger.add_total_mass_constraint(max_val=100)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.selectbox(key="var_pick").select("Methylcellulose")
    at.text_input(key="unit_value").set_value("ml")
    _submit_button(at, "Set unit").click()
    at.run()
    assert not at.exception
    assert [w.value for w in at.warning] == [
        "The limit on Pea protein + Methylcellulose was removed because "
        "those ingredients no longer share a unit.",
        "The limit on All ingredients was removed because "
        "those ingredients no longer share a unit.",
    ], [w.value for w in at.warning]
    assert FoodOptimizer("burger").quantity_constraints == []
    # A limit whose ingredients still share a unit is left alone. (Read the
    # project back first: the fixture's own copy is now behind the screen's.)
    reloaded = FoodOptimizer("burger")
    reloaded.add_quantity_constraint(["Pea protein", "Salt"], max_val=20)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.selectbox(key="var_pick").select("Methylcellulose")   # not in it
    at.text_input(key="unit_value").set_value("mg")
    _submit_button(at, "Set unit").click()
    at.run()
    assert [w.value for w in at.warning] == [], [w.value for w in at.warning]
    assert len(FoodOptimizer("burger").quantity_constraints) == 1
def test_set_unit_refuses_an_empty_box(burger):
    """A blank would rewrite the ingredient to no unit at all — and could
    take an amount limit with it — on a click that looks like a no-op."""
    burger.add_quantity_constraint(["Pea protein", "Methylcellulose"],
                                   max_val=20)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.selectbox(key="var_pick").select("Methylcellulose")
    _submit_button(at, "Set unit").click()
    at.run()
    assert [e.value for e in at.error] == ["A unit is required; use g if the amount is a mass."]
    reloaded = FoodOptimizer("burger")
    assert reloaded.unit_of("Methylcellulose") == "g"
    assert len(reloaded.quantity_constraints) == 1
def test_the_formulations_download_says_why_it_carries_no_units(burger):
    burger.tell({"Pea protein": 10.0, "Methylcellulose": 1.0},
                {"Juiciness": 7.0, "Firmness": 6.0}, formulation_no=1,
                batch_no=1)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    download = _unknown(at.main, "download_button",
                        "Download all formulations (CSV)")
    assert download.proto.help == ("Amounts are unitless in this file so it "
                                   "can be imported back; units are shown on "
                                   "screen. Formulations that were not made "
                                   "are not included.")


# A reloaded ingredient file is the third edit that can empty a limit of
# meaning, and the only one AppTest cannot drive (st.file_uploader takes no
# file from a test). The backend path is covered in test_food_bo; this pins
# the lines the handler writes, including the one no other path can produce.
REMOVED_LIMIT_LINES = """
import streamlit as st
from ui_helpers import render_flash
from ui_setup import _flash_removed_limits


class _Opt:
    variables = [{"name": "Water", "category": "ingredient"},
                 {"name": "Oil", "category": "ingredient"},
                 {"name": "Salt", "category": "ingredient"}]


_flash_removed_limits(_Opt(), [
    {"ingredients": ["Water", "Oil"], "reason": "unit"},
    {"ingredients": ["Water", "Coconut oil"], "reason": "missing",
     "missing": ["Coconut oil"]},
    {"ingredients": ["Water", "Coconut oil", "Beet juice powder"],
     "reason": "missing", "missing": ["Coconut oil", "Beet juice powder"]},
    {"ingredients": ["Water", "Oil", "Salt"], "reason": "unit"},
])
render_flash()
"""


def test_every_removed_limit_is_named_in_one_line():
    at = AppTest.from_string(REMOVED_LIMIT_LINES)
    at.run()
    assert not at.exception
    assert [w.value for w in at.warning] == [
        "The limit on Water + Oil was removed because those ingredients no "
        "longer share a unit.",
        "The limit on Water + Coconut oil was removed because Coconut oil is "
        "no longer an ingredient.",
        "The limit on Water + Coconut oil + Beet juice powder was removed "
        "because Coconut oil and Beet juice powder are no longer ingredients.",
        # Every ingredient in it: named as the Limits list names it.
        "The limit on All ingredients was removed because "
        "those ingredients no longer share a unit.",
    ], [w.value for w in at.warning]


# ------------------------------------------------------------------ #
#  Verification round 2: fresh forms per project, an armed restore,
#  a unit on every number, and Total reserved
# ------------------------------------------------------------------ #
def test_a_unit_change_that_splits_the_units_says_the_batch_is_unscaled(burger):
    """Scaling needs one unit. A unit change that takes it away used to leave
    the table silently back at the generated amounts."""
    burger.set_pending_batch([{"Pea protein": 10.0, "Methylcellulose": 1.0}],
                             batch_no=2)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.number_input(key="scale_total").set_value(400.0)
    at.run()
    at.selectbox(key="var_pick").select("Methylcellulose")
    at.text_input(key="unit_value").set_value("ml")
    _submit_button(at, "Set unit").click()
    at.run()
    assert not at.exception
    assert any(s.value == ("Methylcellulose is now written in ml. The amounts "
                           "were not converted. Batch 2 is no longer shown at "
                           "a batch size of 400 g; a batch size needs all "
                           "ingredients in one unit.")
               for s in at.success), [s.value for s in at.success]
    # ...and the box that held the total is empty, not quietly meaning nothing.
    assert ("scale_total" not in at.session_state
            or at.session_state["scale_total"] is None)


def test_a_limit_on_all_ingredients_is_refused_in_words_the_screen_can_obey(
        mixed_units):
    """The picker left on All ingredients still writes an amount limit, and
    the refusal names the ingredient to re-enter and the unit to enter it
    in — not just that the units differ."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.number_input(key="qc_max").set_value(300.0)
    _submit_button(at, "Add amount limit").click()
    at.run()
    assert not at.exception
    assert [e.value for e in at.error] == [
        "Amount limits add amounts, so these ingredients need one unit; "
        "enter Water in g instead of ml."]
    assert FoodOptimizer("mixed").quantity_constraints == []


def test_leaving_a_formulation_out_keeps_the_note_box_open(open_batch):
    """The note is the only record of what went wrong, and it is typed after
    the tick as often as before it."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.checkbox(key="f1_leave_out").check()
    at.run()
    assert not at.exception
    note = next(t for t in at.text_input if t.key == "f1_note")
    assert not note.disabled
    tick = next(c for c in at.checkbox if c.key == "f1_leave_out")
    assert tick.label == "Not made"
    assert tick.help == "Type why in Note; it is kept with the formulation."
    # The measurement boxes still grey out: a left-out formulation has no
    # results, and only the note it leaves behind.
    assert next(n for n in at.number_input if n.key == "f1_Firmness").disabled


def test_a_left_out_formulation_keeps_the_note_typed_after_the_tick(open_batch):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.checkbox(key="f1_leave_out").check()
    at.run()
    at.text_input(key="f1_note").set_value("Mixer jammed")
    at.run()
    at.number_input(key="f2_Firmness").set_value(6.0)
    at.number_input(key="f2_Juiciness").set_value(7.0)
    at.run()
    _submit_button(at, "Save results").click()
    at.run()
    assert not at.exception
    assert FoodOptimizer("burger").skipped[0]["note"] == "Not made · Mixer jammed"


def test_a_newly_recorded_correction_carries_its_unit(burger):
    """66 could be °C or a panel score; the flash is the only confirmation
    that the right number was typed."""
    burger.add_objective("Serving temperature", 1.0, goal="target", target=65,
                         min_val=0, max_val=100, unit="°C")
    burger.tell({"Pea protein": 10.0, "Methylcellulose": 1.0},
                {"Juiciness": 7.0, "Firmness": 6.0}, formulation_no=1,
                batch_no=1)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["main_tab"] = "3 · Results"
    at.run()
    at.selectbox(key="correct_formulation").set_value(1)
    at.run()
    at.number_input(key="correct_1_Serving temperature").set_value(66.0)
    at.run()
    _submit_button(at, "Save correction").click()
    at.run()
    assert not at.exception
    assert any("Formulation 1 Serving temperature recorded as 66 °C."
               in s.value for s in at.success), [s.value for s in at.success]


def test_the_best_score_says_partial_when_a_measurement_was_not_scored(burger):
    """The All formulations row already says (partial); the caption above it
    said the same score as a full one."""
    burger.tell({"Pea protein": 10.0, "Methylcellulose": 1.0},
                {"Firmness": 6.0}, formulation_no=1, batch_no=1)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["main_tab"] = "3 · Results"
    at.run()
    assert not at.exception
    assert any(c.value.startswith("Overall score 1.50 · partial of 2.50 ·")
               for c in at.caption), [c.value for c in at.caption]
    # ...and one line under it says what the missing measurement costs, in
    # the same words the All formulations table uses. A dropped measurement
    # is arithmetically a zero closeness, and the row is fitted at that
    # depressed score, so the model is taught that region is bad.
    assert any(c.value == ("Partial scores are missing a measurement, which "
                           "counts as zero, so they are low and the model "
                           "treats them that way.")
               for c in at.caption), [c.value for c in at.caption]


def test_the_two_unit_boxes_on_the_tab_are_told_apart(burger):
    """The add form says what a new row is written in; the control row says
    what to rewrite the picked row to. Two boxes labelled Unit did not."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert at.text_input(key="var_unit").label == "Unit"
    box = at.text_input(key="unit_value")
    assert box.label == "New unit"
    assert box.proto.placeholder == "g"       # what the picked row is in today


def test_loading_over_a_list_says_it_replaces_it(burger, tmp_path,
                                                 monkeypatch):
    """The file does not add to the list, it replaces it."""
    from ui_setup import _load_label
    assert _load_label(burger) == "Replace ingredients"
    monkeypatch.chdir(tmp_path)
    assert _load_label(FoodOptimizer("empty_list")) == "Load ingredients"


def test_the_set_unit_picker_is_not_called_a_measurement(burger):
    """It picks the row the controls act on, not what to measure."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert at.selectbox(key="var_pick").label == "Ingredient or setting"


def test_pausing_speaks_for_settings_as_well_as_ingredients(ferment):
    """In a fermentation project the picker offers process settings only."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert at.selectbox(key="var_pick").options == ["Incubation temperature",
                                                    "Incubation time"]
    assert _submit_button(at, "Pause").proto.help == (
        "Held out of new formulations, with every result already recorded "
        "kept.")


def test_the_csv_template_has_one_name_on_both_screens(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()                                   # the welcome panel: no project
    assert _unknown(at.main, "download_button", "Download CSV template")
    FoodOptimizer("named").save()
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert _unknown(at.main, "download_button", "Download CSV template")


def test_the_tab_three_foot_never_offers_a_first_batch_over_an_open_one(burger):
    """A batch on the bench is not a first batch to make, and the foot of
    every other screen already knows the words."""
    burger.set_pending_batch([{"Pea protein": 10.0, "Methylcellulose": 1.0},
                              {"Pea protein": 20.0, "Methylcellulose": 2.0},
                              {"Pea protein": 5.0, "Methylcellulose": 0.5}])
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["main_tab"] = "3 · Results"
    at.run()
    assert not at.exception
    assert _tab_primaries(at, 2) == ["Back to batch 1 · 3 to record"], \
        _tab_primaries(at, 2)


def test_the_biggest_changes_line_reads_the_amounts_on_the_table(burger):
    """While the batch is scaled, a change read off the generated amounts is
    a number nothing on screen shows."""
    burger.tell({"Pea protein": 10.0, "Methylcellulose": 10.0},
                {"Juiciness": 7.0, "Firmness": 6.0}, formulation_no=1,
                batch_no=1)
    burger.set_pending_batch([{"Pea protein": 30.0, "Methylcellulose": 10.0}],
                             batch_no=2)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    # As generated, the only change is +20.00 g of protein.
    assert any(c.value == ("Biggest changes in Formulation 2 from "
                           "Formulation 1: Pea protein +20.00 g.")
               for c in at.caption), [c.value for c in at.caption]
    at.number_input(key="scale_total").set_value(20.0)
    at.run()
    # Scaled to 20 g the table reads 15.00 / 5.00 against 10.00 / 10.00, and
    # the line reads the table: the reference is scaled to the same total.
    line = next((c.value for c in at.caption
                 if c.value.startswith("Biggest changes")), "")
    assert line == ("Biggest changes in Formulation 2 from Formulation 1: "
                    "Pea protein +5.00 g, Methylcellulose −5.00 g."), line


def _uploader_keys(at):
    """The user key of every file uploader on the tabs, read off its widget
    id (AppTest gives no accessor for st.file_uploader)."""
    return sorted(u.proto.id.split("-")[-1]
                  for u in _unknowns(at.main, "file_uploader"))


def test_each_project_gets_its_own_file_uploaders(burger):
    """A file uploader cannot be emptied from session state — assigning None
    is refused and popping the key leaves the mounted widget holding the file
    — so each is keyed to its project and a new project renders an empty one."""
    burger.tell({"Pea protein": 10.0, "Methylcellulose": 1.0},
                {"Juiciness": 7.0, "Firmness": 6.0}, formulation_no=1,
                batch_no=1)
    burger.set_pending_batch([{"Pea protein": 20.0, "Methylcellulose": 2.0}],
                             batch_no=2)
    FoodOptimizer("second").save()
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "burger"
    at.run()
    assert _uploader_keys(at) == ["import_csv_burger", "ingredients_csv_burger",
                                  "results_csv_burger"], _uploader_keys(at)
    at.sidebar.selectbox(key="project_select").select("second")
    at.run()
    _submit_button(at.sidebar, "Open").click()
    at.run()
    assert not at.exception
    assert at.session_state["_loaded_project"] == "second"
    # Not one widget of the last project's is still on screen.
    assert all(k.endswith("_second") for k in _uploader_keys(at)), \
        _uploader_keys(at)


def test_opening_a_project_renames_the_sidebar_box_and_greys_open(burger):
    """The box kept the previous project's name and Open lit up over the
    project the user was typing into."""
    FoodOptimizer("second").save()
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "burger"
    at.run()
    at.sidebar.selectbox(key="project_select").select("second")
    at.run()
    _submit_button(at.sidebar, "Open").click()
    at.run()
    assert not at.exception
    assert at.sidebar.selectbox(key="project_select").value == "second"
    assert [b.label for b in at.sidebar.button if b.proto.type == "primary"] == []
    assert _submit_button(at.sidebar, "Open").disabled


def test_creating_a_project_renames_the_sidebar_box_and_greys_open(burger):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.sidebar.text_input[0].set_value("Probe proj")
    at.sidebar.button[0].click()          # Create project (a form submit)
    at.run()
    assert not at.exception
    assert at.session_state["_loaded_project"] == "Probe proj"
    assert at.sidebar.selectbox(key="project_select").value == "Probe proj"
    assert [b.label for b in at.sidebar.button if b.proto.type == "primary"] == []


def test_a_checked_backup_is_an_armed_confirmation(scored):
    """Yes, replace keeps a copy and replaces the project, so it is the one
    coloured button on screen until it is answered."""
    donor = FoodOptimizer("donor")
    donor.add_ingredient("Flour", 0, 100)
    donor.add_objective("Crunch", 1.0, goal="max")
    donor.tell({"Flour": 10.0}, {"Crunch": 5.0})
    FoodOptimizer("spare").save()
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "burger"
    at.session_state["_restore_candidate"] = donor.export_json()
    at.run()
    assert not at.exception
    # One lit button in the sidebar, and none on any tab.
    assert [b.label for b in at.sidebar.button
            if b.proto.type == "primary"] == ["Yes, replace"]
    assert _tab_primaries(at, 0) == [], _tab_primaries(at, 0)
    assert _tab_primaries(at, 2) == [], _tab_primaries(at, 2)
    assert _submit_button(at, "Continue to make a batch").disabled
    # Cancel puts everything back.
    _submit_button(at.sidebar, "Cancel").click()
    at.run()
    at.run()
    assert "_restore_candidate" not in at.session_state
    assert _tab_primaries(at, 0) == ["Continue to make a batch"]


def test_the_restore_warning_counts_every_formulation_and_names_settings(scored):
    """Both halves of the sentence count the same way, or replacing a project
    with its own backup reads as losing one."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "burger"
    at.session_state["_restore_candidate"] = scored.export_json()
    at.run()
    assert not at.exception
    assert any(w.value == ("This backup contains project **burger** with "
                           "3 formulations and 2 ingredients. Replace "
                           "**burger** (3 formulations)? A copy is saved "
                           "in your FoodOptimizer folder first.")
               for w in at.warning), [w.value for w in at.warning]


def test_the_restore_warning_names_a_settings_only_project(ferment):
    ferment.tell({"Incubation temperature": 38.0, "Incubation time": 6.0},
                 {"Acidity": 4.5}, formulation_no=1, batch_no=1)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_restore_candidate"] = ferment.export_json()
    at.run()
    assert not at.exception
    assert any(w.value.startswith("This backup contains project **ferment** "
                                  "with 1 formulation, 0 ingredients and "
                                  "2 process settings.")
               for w in at.warning), [w.value for w in at.warning]


def test_the_printed_sheet_says_a_repeat_is_a_repeat(burger):
    """Two sheets with identical amounts and nothing printed to say why is
    how a batch gets made twice."""
    burger.tell({"Pea protein": 10.0, "Methylcellulose": 1.0},
                {"Juiciness": 7.0, "Firmness": 6.0}, formulation_no=1,
                batch_no=1)
    burger.set_pending_batch([{"Pea protein": 20.0, "Methylcellulose": 2.0}],
                             batch_no=2)
    burger.add_to_pending_batch({"Pea protein": 10.0, "Methylcellulose": 1.0},
                                note="Repeat of Formulation 1")
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    lines = [t.value for t in at.text]
    assert "Note: Repeat of Formulation 1" in lines, lines
    # The rows with nothing to say still carry a line to write one on.
    assert "Note: ______________________________________________" in lines


def test_a_settings_only_project_reads_as_one_section(ferment):
    """A fermentation scientist weighs nothing out. One section covers both
    kinds, so there is nothing to reorder and no empty Ingredients fold."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert not at.exception
    labels = [e.label for e in at.expander]
    assert "Ingredients (optional)" not in labels, labels
    assert "Process settings" not in labels, labels
    table = next(d.value for d in at.dataframe if "Kind" in d.value.columns)
    assert list(table["Kind"]) == ["Process setting", "Process setting"]
    assert list(table["Unit"]) == ["°C", "h"]


def test_an_ingredient_list_puts_the_ingredients_first(burger):
    """One table, ingredients above settings, each in the order they were
    added."""
    burger.add_process_parameter("Cook temperature", 150, 200, unit="°C")
    burger.add_ingredient("Salt", 0, 3)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    table = next(d.value for d in at.dataframe if "Kind" in d.value.columns)
    assert list(table["Name"]) == ["Pea protein", "Methylcellulose", "Salt",
                                   "Cook temperature"]
    assert at.selectbox(key="var_pick").options == list(table["Name"])


def test_re_adding_an_ingredient_in_another_unit_names_the_limit_it_broke(burger):
    """Every other unit edit prunes the limits it breaks and says which. This
    one used to say only 'Added Water.'"""
    burger.add_quantity_constraint(["Pea protein", "Methylcellulose"],
                                   min_val=5, max_val=150)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.text_input(key="var_name").set_value("Methylcellulose")
    at.text_input(key="var_unit").set_value("ml")
    at.number_input(key="var_high").set_value(3.0)
    at.run()
    _submit_button(at, "Add ingredient or setting").click()
    at.run()
    assert not at.exception
    assert any(s.value == "Added Methylcellulose." for s in at.success), \
        [s.value for s in at.success]
    assert any(w.value == ("The limit on All ingredients was "
                           "removed because those ingredients no longer share "
                           "a unit.") for w in at.warning), \
        [w.value for w in at.warning]
    assert FoodOptimizer("burger").quantity_constraints == []


def test_an_ingredient_named_total_is_refused_on_the_form(burger):
    """Total is the batch table's own column: two of them break the table and
    put the batch total on the sheet where the ingredient's amount belongs."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.text_input(key="var_name").set_value("Total")
    at.run()
    _submit_button(at, "Add ingredient or setting").click()
    at.run()
    assert not at.exception
    assert any("Total is a column name Food Optimizer uses" in e.value
               for e in at.error), [e.value for e in at.error]
    assert [v["name"] for v in FoodOptimizer("burger").variables] == [
        "Pea protein", "Methylcellulose"]


# ------------------------------------------------------------------ #
#  A property limit is a limit on the finished formulation, per 100 g
# ------------------------------------------------------------------ #


@pytest.fixture
def with_properties(tmp_path, monkeypatch):
    """A project loaded from a file with a property column, so the limits
    section has something to limit."""
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("props")
    opt.load_ingredients_from_csv(pd.DataFrame({
        "Name": ["Coconut oil", "Water"],
        "Min": [0, 20],
        "Max": [15, 60],
        "Unit": ["g", "g"],
        "Fat per 100 g": [99.0, 0.0],
    }))
    opt.add_objective("Firmness", 1.0, goal="target", target=6,
                      min_val=0, max_val=10, unit="N")
    return opt


def test_a_property_limit_says_it_is_per_100_g_of_the_formulation(with_properties):
    """It used to be a total that grew with the batch, and nothing on screen
    said which it was."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert not at.exception
    assert any(m.value == "**Limit on the finished formulation**"
               for m in at.markdown), [m.value for m in at.markdown]
    assert at.selectbox(key="prop_metric").label == "Property"
    assert any(c.value == ("Per 100 g of formulation, worked out from each "
                           "ingredient's value for it.")
               for c in at.caption), [c.value for c in at.caption]


def test_the_caption_names_the_unit_the_ingredients_are_in(with_properties):
    with_properties.set_ingredient_unit("Coconut oil", "ml")
    with_properties.set_ingredient_unit("Water", "ml")
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert any(c.value.startswith("Per 100 ml of formulation")
               for c in at.caption), [c.value for c in at.caption]


def test_adding_a_property_limit_lists_it_and_keeps_what_was_made(with_properties):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.number_input(key="prop_max").set_value(20.0)
    at.run()
    _submit_button(at, "Add property limit").click()
    at.run()
    assert not at.exception
    assert any(s.value == ("Limit added on Fat per 100 g. Formulations "
                           "already made are kept. The next batch will "
                           "respect this limit.") for s in at.success), \
        [s.value for s in at.success]
    assert any(t.value == "Fat per 100 g: at most 20" for t in at.text), \
        [t.value for t in at.text]
    assert FoodOptimizer("props").constraints[0]['max'] == 20.0


def test_a_property_limit_is_refused_while_the_units_differ(with_properties):
    with_properties.set_ingredient_unit("Water", "ml")
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.number_input(key="prop_max").set_value(20.0)
    at.run()
    _submit_button(at, "Add property limit").click()
    at.run()
    assert not at.exception
    # The refusal names the fix, not just the fault.
    assert any(e.value == ("Property limits are per 100 g, so every ingredient "
                           "needs a mass unit; enter Water in g instead of "
                           "ml.") for e in at.error), \
        [e.value for e in at.error]
    assert FoodOptimizer("props").constraints == []


def test_a_limit_from_an_older_version_says_how_it_is_read_now(with_properties):
    """A 0.2.x file's limit was a total. It is now read per 100 g, which is a
    different number, so the screen says so once."""
    with_properties.constraints = [{'metric': "Fat per 100 g", 'min': None,
                                    'max': 25.0}]
    with_properties.save()
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert not at.exception
    assert sum(1 for c in at.caption
               if c.value == ("A limit set before this version is now read "
                              "per 100 g of formulation.")) == 1, \
        [c.value for c in at.caption]


def test_a_limit_set_now_says_nothing_extra(with_properties):
    with_properties.add_constraint("Fat per 100 g", max_val=25.0)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert not any("before this version" in c.value for c in at.caption), \
        [c.value for c in at.caption]


# ------------------------------------------------------------------ #
#  One section for everything you can vary
# ------------------------------------------------------------------ #


def _tab1(at):
    return at.tabs[0]


def _tab1_captions(at):
    """The captions a reader sees on tab 1 without opening anything: the
    contents of a collapsed expander are explanations, and they are allowed
    to be longer."""
    tab = at.tabs[0]
    folded = {id(c) for e in tab.expander for c in e.caption}
    return [c.value for c in tab.caption if id(c) not in folded]


def test_what_you_can_vary_is_the_one_section_for_both_kinds(burger):
    """Four places used to hold the ingredients and the settings: a subheader,
    an expander, another expander and a fold. They are one section now."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert not at.exception
    assert [h.value for h in _tab1(at).subheader] == [
        "What you can vary", "Measurements and targets"], \
        [h.value for h in _tab1(at).subheader]
    assert any(c == "Ingredients and process settings you will change between "
                    "formulations." for c in _tab1_captions(at)), \
        _tab1_captions(at)
    labels = [e.label for e in _tab1(at).expander]
    assert labels == ["Or upload an ingredients CSV", "Add a measurement",
                      "How closeness is worked out", "Limits (optional)",
                      "How formulations are chosen (advanced)"], labels


def test_the_add_form_is_the_first_thing_on_the_tab(burger):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert at.text_input(key="var_name").label == "Name"
    kind = at.radio(key="var_kind")
    assert kind.label == "Kind"
    assert kind.options == ["Ingredient", "Process setting"]
    assert at.number_input(key="var_low").label == "Lowest"
    assert at.number_input(key="var_high").label == "Highest"
    assert at.text_input(key="var_unit").value == "g"
    add = _submit_button(at, "Add ingredient or setting")
    assert add.proto.type == "secondary"
    # ...and it is not folded away inside anything.
    folded = {id(t) for e in _tab1(at).expander for t in e.text_input}
    assert id(at.text_input(key="var_name")) not in folded


def test_the_one_form_adds_an_ingredient(burger):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.text_input(key="var_name").set_value("Beet juice powder")
    at.number_input(key="var_high").set_value(2.0)
    at.run()
    _submit_button(at, "Add ingredient or setting").click()
    at.run()
    assert not at.exception
    assert any(s.value == "Added Beet juice powder." for s in at.success), \
        [s.value for s in at.success]
    saved = FoodOptimizer("burger")
    assert [v["name"] for v in saved.variables][-1] == "Beet juice powder"
    assert saved.unit_of("Beet juice powder") == "g"


def test_the_one_form_adds_a_process_setting(burger):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.radio(key="var_kind").set_value("Process setting")
    at.run()
    # A setting is not weighed out, so the unit box empties rather than
    # offering to cook at 175 g.
    assert at.text_input(key="var_unit").value == ""
    at.text_input(key="var_name").set_value("Cook temperature")
    at.text_input(key="var_unit").set_value("°C")
    at.number_input(key="var_low").set_value(150.0)
    at.number_input(key="var_high").set_value(200.0)
    at.run()
    _submit_button(at, "Add ingredient or setting").click()
    at.run()
    assert not at.exception
    assert any(s.value == "Added Cook temperature." for s in at.success), \
        [s.value for s in at.success]
    saved = FoodOptimizer("burger")
    setting = saved._var_by_name("Cook temperature")
    assert setting["category"] == "process"
    assert setting["unit"] == "°C"


def test_the_baseline_box_belongs_to_a_setting_added_mid_run(burger):
    burger.tell({"Pea protein": 10.0, "Methylcellulose": 1.0},
                {"Juiciness": 7.0, "Firmness": 6.0}, formulation_no=1,
                batch_no=1)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert not [n for n in at.number_input if n.key == "var_base"]
    at.radio(key="var_kind").set_value("Process setting")
    at.run()
    assert at.number_input(key="var_base").label == "Setting used so far"
    at.text_input(key="var_name").set_value("Cook temperature")
    at.run()
    _submit_button(at, "Add ingredient or setting").click()
    at.run()
    assert any("Enter the baseline" in e.value for e in at.error), \
        [e.value for e in at.error]


def test_the_table_names_the_kind_of_every_row(burger):
    burger.add_process_parameter("Cook temperature", 150, 200, unit="°C")
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    table = next(d.value for d in _tab1(at).dataframe if "Kind" in d.value.columns)
    assert list(table.columns) == ["Kind", "Name", "Lowest", "Highest", "Unit"]
    assert list(table["Name"]) == ["Pea protein", "Methylcellulose",
                                   "Cook temperature"]
    assert list(table["Kind"]) == ["Ingredient", "Ingredient",
                                   "Process setting"]
    assert list(table["Unit"]) == ["g", "g", "°C"]


def test_a_paused_row_is_the_only_reason_for_a_status_column(burger):
    burger.deactivate_variable("Methylcellulose", value=1.0)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    table = next(d.value for d in _tab1(at).dataframe if "Kind" in d.value.columns)
    assert "Status" in table.columns
    assert list(table["Status"]) == ["active", "paused · held at 1.00 g"]


def test_the_control_row_pauses_and_resumes_one_row(burger):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.selectbox(key="var_pick").select("Methylcellulose")
    at.run()
    _submit_button(at, "Pause").click()
    at.run()
    assert not at.exception
    assert any(s.value == "Paused Methylcellulose." for s in at.success), \
        [s.value for s in at.success]
    assert [v["name"] for v in FoodOptimizer("burger").inactive_variables()] \
        == ["Methylcellulose"]
    # The same button is now the way back.
    assert "Pause" not in _labels(at)
    _submit_button(at, "Resume").click()
    at.run()
    assert any(s.value == "Resumed Methylcellulose." for s in at.success), \
        [s.value for s in at.success]
    assert FoodOptimizer("burger").inactive_variables() == []


def test_pause_is_grey_and_says_why_when_only_one_is_left(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("one_only")
    opt.add_ingredient("Water", 0, 100)
    opt.add_objective("Taste", 1.0, goal="max", min_val=0, max_val=10)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    pause = _submit_button(at, "Pause")
    assert pause.disabled
    assert pause.proto.help == ("At least two ingredients or settings must "
                                "stay active before one can be paused.")


def test_the_control_row_sets_one_ingredient_unit(burger):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.selectbox(key="var_pick").select("Methylcellulose")
    at.run()
    at.text_input(key="unit_value").set_value("ml")
    at.run()
    _submit_button(at, "Set unit").click()
    at.run()
    assert not at.exception
    assert any(s.value == "Methylcellulose is now written in ml. The amounts were not converted."
               for s in at.success), [s.value for s in at.success]
    assert FoodOptimizer("burger").unit_of("Methylcellulose") == "ml"


def test_set_unit_changes_a_process_setting_too(burger):
    """A setting carries its own unit, so the same control sets it — and it
    is part of no sum and no average, so no limit is touched."""
    burger.add_process_parameter("Cook temperature", 150, 200, unit="°C")
    burger.add_quantity_constraint(["Pea protein", "Methylcellulose"],
                                   max_val=20)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.selectbox(key="var_pick").select("Cook temperature")
    at.run()
    assert not _submit_button(at, "Set unit").disabled
    at.text_input(key="unit_value").set_value("°F")
    at.run()
    _submit_button(at, "Set unit").click()
    at.run()
    assert not at.exception
    # A setting has one value, not amounts: the sentence says what it changed.
    assert any(s.value == ("Cook temperature is now written in °F. The value "
                           "was not converted.")
               for s in at.success), [s.value for s in at.success]
    reloaded = FoodOptimizer("burger")
    assert reloaded.unit_of("Cook temperature") == "°F"
    assert len(reloaded.quantity_constraints) == 1
    assert not at.warning, [w.value for w in at.warning]


def test_a_blank_unit_is_refused_for_a_setting_as_well(burger):
    burger.add_process_parameter("Cook temperature", 150, 200, unit="°C")
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.selectbox(key="var_pick").select("Cook temperature")
    at.run()
    _submit_button(at, "Set unit").click()
    at.run()
    assert [e.value for e in at.error] == ["A unit is required; use g if the amount is a mass."]
    assert FoodOptimizer("burger").unit_of("Cook temperature") == "°C"


def test_a_unit_change_that_breaks_a_property_limit_says_so(with_properties):
    """A property limit is an average over the amounts, so it needs one unit
    just as a sum does; when a unit change takes that away it is removed, in
    the same one line an amount limit gets."""
    with_properties.add_constraint("Fat per 100 g", max_val=25.0)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.selectbox(key="var_pick").select("Water")
    at.run()
    at.text_input(key="unit_value").set_value("ml")
    at.run()
    _submit_button(at, "Set unit").click()
    at.run()
    assert not at.exception
    assert [w.value for w in at.warning] == [
        "The limit on Fat per 100 g was removed because the ingredients no "
        "longer share a unit."], [w.value for w in at.warning]
    assert FoodOptimizer("props").constraints == []


def test_a_reloaded_file_in_two_units_says_which_limit_went(with_properties):
    with_properties.add_constraint("Fat per 100 g", max_val=25.0)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.selectbox(key="var_pick").select("Coconut oil")
    at.run()
    at.text_input(key="unit_value").set_value("ml")
    at.run()
    _submit_button(at, "Set unit").click()
    at.run()
    assert any("The limit on Fat per 100 g was removed" in w.value
               for w in at.warning), [w.value for w in at.warning]


def test_remove_from_the_control_row_confirms_and_keeps_a_copy(burger, tmp_path):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.selectbox(key="var_pick").select("Methylcellulose")
    at.run()
    _submit_button(at, "Remove Methylcellulose").click()
    at.run()
    assert any("Remove Methylcellulose from this project permanently?"
               in w.value for w in at.warning), [w.value for w in at.warning]
    _submit_button(at, "Yes, remove").click()
    at.run()
    assert not at.exception
    assert (tmp_path / "burger_pre_delete.pkl").exists()
    assert [v["name"] for v in FoodOptimizer("burger").variables] == \
        ["Pea protein"]


def test_remove_still_offers_the_used_ingredient_path(burger):
    """The permanent-deletion tick box lives inside the confirmation now."""
    burger.tell({"Pea protein": 10.0, "Methylcellulose": 1.0},
                {"Juiciness": 7.0, "Firmness": 6.0}, formulation_no=1,
                batch_no=1)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.selectbox(key="var_pick").select("Methylcellulose")
    at.run()
    assert not [c for c in at.checkbox if c.key == "delete_ing_force"]
    _submit_button(at, "Remove Methylcellulose").click()
    at.run()
    assert at.checkbox(key="delete_ing_force").label == (
        "Remove even if it was used (discards that information)")
    _submit_button(at, "Yes, remove").click()
    at.run()
    assert any("was used" in e.value for e in at.error), \
        [e.value for e in at.error]
    # The refusal leaves nothing armed, so the tick box is gone with it: the
    # deletion is asked for again, this time with the box ticked.
    assert not [c for c in at.checkbox if c.key == "delete_ing_force"]
    _submit_button(at, "Remove Methylcellulose").click()
    at.run()
    at.checkbox(key="delete_ing_force").set_value(True)
    at.run()
    _submit_button(at, "Yes, remove").click()
    at.run()
    assert not at.exception
    assert [v["name"] for v in FoodOptimizer("burger").variables] == \
        ["Pea protein"]


def test_remove_takes_a_process_setting_too(burger, tmp_path):
    burger.add_process_parameter("Cook temperature", 150, 200, unit="°C")
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.selectbox(key="var_pick").select("Cook temperature")
    at.run()
    _submit_button(at, "Remove Cook temperature").click()
    at.run()
    _submit_button(at, "Yes, remove").click()
    at.run()
    assert not at.exception
    assert any(s.value == "Removed Cook temperature." for s in at.success), \
        [s.value for s in at.success]
    assert (tmp_path / "burger_pre_delete.pkl").exists()
    assert [v["name"] for v in FoodOptimizer("burger").variables] == \
        ["Pea protein", "Methylcellulose"]


def test_changing_the_pick_disarms_a_removal(burger):
    """An armed confirmation for one row must not answer for another."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.selectbox(key="var_pick").select("Methylcellulose")
    at.run()
    _submit_button(at, "Remove Methylcellulose").click()
    at.run()
    at.selectbox(key="var_pick").select("Pea protein")
    at.run()
    assert not at.warning, [w.value for w in at.warning]
    assert "Yes, remove" not in _labels(at)
    assert _tab_primaries(at, 0) == ["Continue to make a batch"]


def test_the_upload_is_folded_away_beneath(burger):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    fold = next(e for e in _tab1(at).expander if e.label == "Or upload an ingredients CSV")
    assert any(c.value == ("A CSV with the columns Name, Lowest, Highest "
                           "and, optionally, Unit. Extra columns become "
                           "properties you can set limits on.")
               for c in fold.caption), \
        [c.value for c in fold.caption]
    assert [b.label for b in _unknowns(fold, "download_button")] == \
        ["Download CSV template"]


def test_the_tab_has_no_project_wide_unit_box(burger):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert not [t for t in at.text_input if t.key == "amount_unit"], \
        [t.key for t in at.text_input]


def test_the_target_bullet_does_not_claim_a_floor_of_zero(burger):
    """u = max(0, 1 - |norm - target|) with norm clamped to [0, 1], so the
    largest attainable distance is max(t, 1 - t): a mid-scale target bottoms
    out at 0.5 and only a target on a scale end ever reaches 0."""
    from food_bo import FoodOptimizer as _FO
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    fold = next(e for e in _tab1(at).expander
                if e.label == "How closeness is worked out")
    text = " ".join(m.value for m in fold.markdown)
    assert "a full scale width away scores 0" not in text, text
    assert ("by one point per full scale width; the lowest score depends on "
            "how far the target sits from the ends of your scale") in text, text
    # ...and the claim the bullet now makes is the one the code makes: a
    # target of 6 on a 0 to 10 scale scores 0.4 closeness at 0, never 0.
    burger.tell({"Pea protein": 10.0, "Methylcellulose": 1.0},
                {"Firmness": 0.0}, formulation_no=1, batch_no=1)
    reloaded = _FO(burger.project_name)
    assert float(reloaded.Y_history[0]) == pytest.approx(1.5 * 0.4)


def test_the_closeness_fold_says_what_closeness_is(burger):
    """The fold explains closeness; what importance is belongs to the
    Importance field's own tooltip, beside the box it is about."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    fold = next(e for e in _tab1(at).expander
                if e.label == "How closeness is worked out")
    assert any("Closeness is a measurement's normalised score between 0 and 1."
               in m.value for m in fold.markdown), \
        [m.value for m in fold.markdown]
    assert not any("importance is the weight" in m.value.lower()
                   for m in fold.markdown), [m.value for m in fold.markdown]
    assert at.number_input(key="meas_new_importance").help == (
        "Any positive number. 2 counts twice as much as 1. Importance is the "
        "weight of each measurement in the overall score.")


def _assert_captions_read_once(at):
    """Sparse: one line each, and never the same line twice. The score
    function is generated from the measurements, not written here."""
    captions = [c for c in _tab1_captions(at)
                if not c.startswith("Overall score = ")]
    assert len(captions) == len(set(captions)), captions
    assert all(len(c) <= 100 for c in captions), \
        [c for c in captions if len(c) > 100]


def test_no_caption_is_said_twice_on_the_tab(burger):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    _assert_captions_read_once(at)


def test_the_tab_reads_in_one_order(burger):
    """What you can vary, then the measurements, then what is optional, then
    the one coloured button."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    order = []
    for element in _tab1(at).children.values():
        name = element.__class__.__name__
        if name == "Subheader":
            order.append(element.value)
        elif name == "Expander" and element.label in (
                "Limits (optional)", "How formulations are chosen (advanced)"):
            order.append(element.label)
    assert order == ["What you can vary", "Measurements and targets",
                     "Limits (optional)", "How formulations are chosen (advanced)"], order
    assert _tab_primaries(at, 0) == ["Continue to make a batch"]


# ------------------------------------------------------------------ #
#  Properties named in the app (no ingredient CSV needed)
# ------------------------------------------------------------------ #

def _prop_name_box(at):
    return at.text_input(key="prop_new")


def test_a_property_is_named_on_the_limits_section_and_the_box_empties(burger):
    """A property used to need an ingredient CSV with an extra column, so a
    project typed in by hand could not limit sodium at all."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert _prop_name_box(at).label == "Add a property, such as Sodium per 100 g"
    _prop_name_box(at).set_value("Sodium per 100 g")
    at.run()
    next(b for b in at.button if b.key == "add_property").click()
    at.run()
    assert not at.exception
    assert any("Added Sodium per 100 g." in s.value for s in at.success), \
        [s.value for s in at.success]
    assert FoodOptimizer("burger").properties() == ["Sodium per 100 g"]
    # The box empties, or the next click adds the same name again.
    assert _prop_name_box(at).value == ""
    assert at.selectbox(key="prop_metric").options == ["Sodium per 100 g"]


def test_a_property_name_that_clashes_is_refused_on_screen(burger):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    _prop_name_box(at).set_value("Firmness")
    at.run()
    next(b for b in at.button if b.key == "add_property").click()
    at.run()
    assert not at.exception
    assert any("already the name of a measurement" in e.value for e in at.error), \
        [e.value for e in at.error]
    assert FoodOptimizer("burger").properties() == []


def test_the_add_form_and_the_table_carry_one_column_per_property(burger):
    """One number box per property on the add form, one column per property in
    the table, blank where an ingredient has no value."""
    burger.add_property("Sodium per 100 g")
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    box = at.number_input(key="var_prop_Sodium per 100 g")
    assert box.label == "Sodium per 100 g"
    at.text_input(key="var_name").set_value("Salt")
    at.number_input(key="var_high").set_value(5.0)
    box.set_value(39000.0)
    at.run()
    _submit_button(at, "Add ingredient or setting").click()
    at.run()
    assert not at.exception
    saved = FoodOptimizer("burger")
    assert saved.property_value("Salt", "Sodium per 100 g") == 39000.0
    table = next(d.value for d in at.dataframe if "Kind" in d.value.columns)
    assert list(table.columns)[-1] == "Sodium per 100 g"
    assert dict(zip(table["Name"], table["Sodium per 100 g"])) == {
        "Pea protein": "", "Methylcellulose": "", "Salt": "39000"}
    # The box empties, or the next ingredient inherits this one's value.
    assert at.number_input(key="var_prop_Sodium per 100 g").value is None


def test_a_process_setting_is_not_asked_for_a_property_value(burger):
    burger.add_property("Cost")
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert at.number_input(key="var_prop_Cost") is not None
    at.radio(key="var_kind").set_value("Process setting")
    at.run()
    assert "var_prop_Cost" not in [n.key for n in at.number_input]


def test_set_property_values_writes_the_picked_ingredients_values(burger):
    burger.add_property("Cost")
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.selectbox(key="var_pick").select("Methylcellulose")
    at.run()
    next(b for b in at.button if b.key == "set_props").click()
    at.run()
    at.number_input(key="setprop_Methylcellulose_Cost").set_value(42.0)
    at.run()
    next(b for b in at.button if b.key == "save_props").click()
    at.run()
    assert not at.exception
    assert any("Property values saved for Methylcellulose." in s.value
               for s in at.success), [s.value for s in at.success]
    assert FoodOptimizer("burger").property_value("Methylcellulose", "Cost") == 42.0
    # The editor closes; the button is back on its own.
    assert "setprop_Methylcellulose_Cost" not in [n.key for n in at.number_input]


def test_the_property_editor_opens_on_the_value_already_stored(burger):
    burger.add_property("Cost")
    burger.set_property_value("Pea protein", "Cost", 3.5)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    next(b for b in at.button if b.key == "set_props").click()
    at.run()
    assert at.number_input(key="setprop_Pea protein_Cost").value == 3.5
    # A second ingredient opens on its own value, not on this one's: the box
    # is keyed per ingredient and seeded when the editor opens.
    next(b for b in at.button if b.key == "close_props").click()
    at.run()
    at.selectbox(key="var_pick").select("Methylcellulose")
    at.run()
    next(b for b in at.button if b.key == "set_props").click()
    at.run()
    assert at.number_input(key="setprop_Methylcellulose_Cost").value is None
    # Emptying a box clears the value (no value is not 0); AppTest cannot type
    # an empty number box, so that half is covered in test_food_bo.py.


def test_a_limit_names_the_ingredients_that_have_no_value(burger):
    """An ingredient with no value counts as 0 in the average, and a limit
    that looks satisfied for that reason is the one way this lies."""
    burger.add_property("Sodium per 100 g")
    burger.set_property_value("Pea protein", "Sodium per 100 g", 20)
    burger.add_constraint("Sodium per 100 g", max_val=450)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    line = next(t.value for t in at.text if t.value.startswith("Sodium per 100 g:"))
    assert line == ("Sodium per 100 g: at most 450 · Methylcellulose has no "
                    "value and counts as 0.")
    at.session_state["optimizer"].set_property_value(
        "Methylcellulose", "Sodium per 100 g", 0)
    at.run()
    line = next(t.value for t in at.text if t.value.startswith("Sodium per 100 g:"))
    assert line == "Sodium per 100 g: at most 450"


def test_removing_a_property_asks_first_keeps_a_copy_and_takes_its_limit(
        burger, tmp_path):
    burger.add_property("Sodium per 100 g")
    burger.set_property_value("Pea protein", "Sodium per 100 g", 20)
    burger.add_constraint("Sodium per 100 g", max_val=450)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    next(b for b in at.button
         if b.key == "rm_prop_Sodium per 100 g__btn").click()
    at.run()
    assert any("Remove Sodium per 100 g and its 1 limit?" in w.value
               for w in at.warning), [w.value for w in at.warning]
    assert any("A copy is saved in your FoodOptimizer folder first." in w.value
               for w in at.warning), [w.value for w in at.warning]
    assert FoodOptimizer("burger").properties() == ["Sodium per 100 g"]
    next(b for b in at.button
         if b.key == "rm_prop_Sodium per 100 g__yes").click()
    at.run()
    assert not at.exception
    saved = FoodOptimizer("burger")
    assert saved.properties() == []
    assert saved.constraints == []
    assert saved.ingredient_properties["Pea protein"] == {}
    assert (tmp_path / "burger_pre_delete.pkl").exists()
    assert any("Removed Sodium per 100 g. Its 1 limit went with it." in s.value
               for s in at.success), [s.value for s in at.success]


def test_a_hand_made_property_limit_reaches_the_batch(burger):
    """The whole point: name a property, give it values, limit it, and the
    formulations the app chooses respect the limit."""
    burger.add_property("Sodium per 100 g")
    burger.set_property_value("Pea protein", "Sodium per 100 g", 1000)
    burger.set_property_value("Methylcellulose", "Sodium per 100 g", 0)
    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.run()
    at.selectbox(key="prop_metric").select("Sodium per 100 g")
    at.number_input(key="prop_max").set_value(200.0)
    at.run()
    _submit_button(at, "Add property limit").click()
    at.run()
    assert not at.exception
    assert FoodOptimizer("burger").constraints[0]['max'] == 200.0
    at.button(key="generate").click()
    at.run()
    opt = at.session_state["optimizer"]
    assert opt.pending_batch, [e.value for e in at.error]
    for row in opt.pending_batch:
        assert opt.property_per_100(row['recipe'], "Sodium per 100 g") <= 200.0 + 1e-6


# ------------------------------------------------------------------ #
#  The coherence wave: one word per concept, one control per idea
# ------------------------------------------------------------------ #

def test_one_amount_limit_control_covers_every_group_and_the_total(burger):
    """There were two controls for one idea — a group limit whose picker,
    with every ingredient ticked, wrote exactly what the total control wrote.
    The picker's empty state is now every ingredient."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    picker = at.multiselect(key="qty_pick")
    assert picker.value == []
    assert picker.proto.placeholder == "All ingredients"
    assert at.number_input(key="qc_min").label == "At least (g)"
    assert at.number_input(key="qc_max").label == "At most (g)"
    # One button, not three: there is no second amount-limit control.
    assert [b.label for b in at.button
            if "amount limit" in b.label] == ["Add amount limit"]
    # Left alone, the picker limits the whole formulation.
    at.number_input(key="qc_max").set_value(100.0)
    _submit_button(at, "Add amount limit").click()
    at.run()
    assert not at.exception
    saved = FoodOptimizer("burger")
    assert set(saved.quantity_constraints[0]['ingredients']) == {
        "Pea protein", "Methylcellulose"}
    assert any(s.value.startswith("Limit added on all ingredients.")
               for s in at.success), [s.value for s in at.success]
    # A group is picked the same way, and both read alike in the one list.
    at.multiselect(key="qty_pick").select("Pea protein")
    at.number_input(key="qc_max").set_value(50.0)
    at.run()
    _submit_button(at, "Add amount limit").click()
    at.run()
    assert not at.exception
    lines = [t.value for t in at.text]
    assert "All ingredients: at most 100 g" in lines, lines
    assert "Pea protein: at most 50 g" in lines, lines


def test_an_amount_limit_needs_a_number(burger):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    _submit_button(at, "Add amount limit").click()
    at.run()
    assert [e.value for e in at.error] == ["Enter a lowest, a highest, or both."]
    assert FoodOptimizer("burger").quantity_constraints == []


def test_the_kind_radio_says_what_the_two_kinds_are(burger):
    """It decides whether the row is weighed into the total, scaled, and
    eligible for an amount limit, and nothing on screen said so."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert at.radio(key="var_kind").help == (
        "Ingredients are weighed into the formulation and count towards its "
        "total. Process settings, such as temperature or time, are dialled "
        "in.")


def test_the_model_settings_in_use_line_shows_only_under_standard(burger):
    """With the four boxes on screen the line repeated them; it is what the
    project is using while the boxes are away."""
    burger.set_bo_config({"kernel": "matern52", "lengthscale_prior": "default",
                          "noise": "default", "acquisition": "qlognei"})
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert any(e.label == "How formulations are chosen (advanced)"
               for e in at.expander), [e.label for e in at.expander]
    assert at.radio(key="bo_cfg_mode").value == "Expert-selected"
    assert not any(c.value.startswith("In use:") for c in at.caption), \
        [c.value for c in at.caption]
    at.radio(key="bo_cfg_mode").set_value("Standard (default)")
    at.run()
    assert any(c.value.startswith("In use: kernel: matern52")
               for c in at.caption), [c.value for c in at.caption]


def test_off_by_appears_only_when_a_measurement_has_a_target(tmp_path,
                                                             monkeypatch):
    """Off by is a distance from a target; over Higher is better rows it was
    a column of dashes."""
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("no_target")
    opt.add_ingredient("Water", 0, 100)
    opt.add_objective("Crunch", 1.0, goal="max", min_val=0, max_val=10, unit="N")
    opt.tell({"Water": 50.0}, {"Crunch": 7.0}, formulation_no=1, batch_no=1)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    table = next(d.value for d in at.dataframe if "Measured" in d.value.columns)
    assert list(table.columns) == ["Measurement", "Goal", "Measured"]
    # ...and it comes back the moment a measurement has a target.
    opt.add_objective("Firmness", 1.0, goal="target", target=6, min_val=0,
                      max_val=10, unit="N")
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    table = next(d.value for d in at.dataframe if "Measured" in d.value.columns)
    assert list(table.columns) == ["Measurement", "Goal", "Measured", "Off by"]


def test_the_partial_sentence_is_said_once_on_the_tab(burger):
    """It belongs under the score or under the table, never twice on one
    screen."""
    burger.tell({"Pea protein": 10.0, "Methylcellulose": 1.0},
                {"Firmness": 6.0}, formulation_no=1, batch_no=1)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["main_tab"] = "3 · Results"
    at.run()
    sentence = ("Partial scores are missing a measurement, which counts as "
                "zero, so they are low and the model treats them that way.")
    assert sum(1 for c in at.caption if c.value == sentence) == 1, \
        [c.value for c in at.caption]
    # With a complete best above it, the table is the one that says it.
    burger.tell({"Pea protein": 12.0, "Methylcellulose": 1.5},
                {"Firmness": 6.0, "Juiciness": 7.0}, formulation_no=2,
                batch_no=1)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["main_tab"] = "3 · Results"
    at.run()
    assert sum(1 for c in at.caption if c.value == sentence) == 1, \
        [c.value for c in at.caption]


def test_the_correction_picker_says_why_a_formulation_is_missing(scored):
    """It offers fewer numbers than All formulations lists, and the reason is
    not visible from the box."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert at.selectbox(key="correct_formulation").options == ["1", "2"]
    assert any(c.value == ("Formulations that were not made have no result "
                           "to correct.") for c in at.caption), \
        [c.value for c in at.caption]


def test_the_two_downloads_on_the_batch_name_their_format(open_batch):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    names = [b.label for b in _unknowns(at.main, "download_button")]
    assert "Download batch sheet (CSV)" in names, names
    assert "Download formulation sheets (to print)" in names, names
    assert any(e.label == "Preview formulation sheets" for e in at.expander), \
        [e.label for e in at.expander]


# ------------------------------------------------------------------ #
#  Coherence follow-up: every Remove names its object, one shape for
#  the uploaders, one set of property boxes at a time
# ------------------------------------------------------------------ #

def test_every_remove_on_set_up_names_what_it_takes_out(burger):
    """A row of bare `Remove` buttons asks the reader which one is theirs."""
    burger.add_property("Fat per 100 g")
    burger.add_constraint("Fat per 100 g", max_val=20)
    burger.add_quantity_constraint(["Pea protein"], max_val=10)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    labels = _labels(at)
    assert "Remove Pea protein" in labels, labels        # the control row
    assert "Remove Fat per 100 g" in labels, labels      # the property list
    assert "Remove Firmness" in labels, labels           # the measurement
    # A limit line already names the limit beside it, so its button says only
    # what it takes out.
    assert labels.count("Remove limit") == 2, labels
    assert "Remove" not in labels, labels


def test_the_control_rows_remove_follows_the_pick(burger):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    at.selectbox(key="var_pick").select("Methylcellulose")
    at.run()
    assert "Remove Methylcellulose" in _labels(at), _labels(at)


def test_the_property_button_says_what_it_adds(burger):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert "Add property" in _labels(at), _labels(at)
    # The name carries the unit, and the placeholder shows one.
    assert at.text_input(key="prop_new").proto.placeholder == \
        "e.g. Sodium mg per 100 g"


def test_the_per_100_caption_waits_for_one_unit(mixed_units):
    """`Per 100 g` over a project of grams and millilitres named a hundred of
    nothing; the limit itself is refused until they agree."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert any(c.value == ("Per 100 g of formulation once every ingredient is "
                           "in one mass unit.") for c in at.caption), \
        [c.value for c in at.caption]
    mixed_units.set_ingredient_unit("Water", "g")
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    assert any(c.value == ("Per 100 g of formulation, worked out from each "
                           "ingredient's value for it.") for c in at.caption), \
        [c.value for c in at.caption]


def test_only_one_set_of_property_boxes_is_on_screen(burger):
    """The add form and Set property values both hold a box per property;
    two boxes for one property on one screen is two answers to one question."""
    burger.add_property("Cost")
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    keys = [n.key for n in at.number_input]
    assert "var_prop_Cost" in keys and "setprop_Pea protein_Cost" not in keys
    next(b for b in at.button if b.key == "set_props").click()
    at.run()
    keys = [n.key for n in at.number_input]
    assert "setprop_Pea protein_Cost" in keys, keys
    assert "var_prop_Cost" not in keys, keys
    assert "Save values" in _labels(at), _labels(at)
    next(b for b in at.button if b.key == "close_props").click()
    at.run()
    assert "var_prop_Cost" in [n.key for n in at.number_input]


def test_both_upload_doors_read_the_same_way(open_batch):
    """One shape for the three uploaders, and one verb for the button that
    reads whatever was put in them."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    labels = [u.label for u in _unknowns(at.main, "file_uploader")]
    assert "Upload ingredients CSV" in labels, labels
    assert "Upload results CSV" in labels, labels
    assert "Upload formulations CSV" in labels, labels
    assert not any(l == "Results sheet" for l in labels), labels
