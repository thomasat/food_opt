"""The sidebar's, the project file's and the Review's refusals (error
review, lane "app"): each names the thing to look at — the project, the
file, the copy, the character — and says the fix in the reader's words."""
import errno
import json
import os
import pathlib

import pytest
from streamlit.testing.v1 import AppTest

import sample_projects as sp
import storage as storage_backend
import wording
from food_bo import FoodOptimizer
from storage import LocalStorage, StorageError

APP_PATH = os.path.join(os.path.dirname(__file__), "..", "app.py")


def _button(at, label):
    return next(b for b in at.button if b.label == label)


def _errors(at):
    return [e.value for e in at.error]


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("Oat cookie")
    opt.add_ingredient("Oats", 0, 100)
    opt.add_objective("Taste", 1.0, goal="max")
    opt.tell({"Oats": 50.0}, {"Taste": 7.0})
    return opt


def _create(at, name):
    at.text_input(key="new_project_name").set_value(name)
    _button(at, wording.CREATE_PROJECT).click()
    at.run()


# ------------------------------------------------------------------ #
#  New project: the name
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("typed, expected", [
    ("bad/name", "Take “/” out of the name. Use letters, numbers, spaces, . - or _."),
    ("Crème brûlée", "Take “è” “û” “é” out of the name. Use letters, numbers, spaces, . - or _."),
    ("-oat", "Start the name with a letter or a number."),
    ("x" * 70, "Shorten the name to 64 characters; it has 70."),
])
def test_a_refused_name_says_which_part_and_the_fix(tmp_path, monkeypatch, typed, expected):
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    _create(at, typed)
    assert not at.exception
    assert _errors(at) == [expected]
    assert list(tmp_path.glob("*.pkl")) == []


def test_a_taken_name_points_to_open_project(project):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    _create(at, "Oat cookie")
    assert _errors(at) == [
        "A project named Oat cookie already exists. Open it from Open project, "
        "or choose another name."]


def test_a_name_in_other_capitals_names_the_project_it_matches(project):
    """The Mac's folder does not tell Oat cookie from OAT COOKIE: the old
    line blamed 'a project or an archived copy'."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    _create(at, "OAT COOKIE")
    assert _errors(at) == [wording.name_taken("Oat cookie")]


def test_a_saved_copy_ending_is_named(project):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    _create(at, "Granola_deleted")
    assert _errors(at) == [
        "Names ending in _deleted are kept for saved copies. Choose another name."]


def test_a_saved_copy_with_the_name_is_named(project):
    LocalStorage().archive("Oat cookie", "pre_edit", copy=True)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    _create(at, "Oat cookie_pre_edit")
    assert _errors(at) == [
        "A saved copy is already named Oat cookie_pre_edit. Choose another name."]


# ------------------------------------------------------------------ #
#  Open a saved copy: which file, that it will not open, what to do
# ------------------------------------------------------------------ #

def _offer(at, state, file="oat copy 2026-10-01.json"):
    at.session_state["_loaded_project"] = "Oat cookie"
    at.session_state["_restore_candidate"] = state
    at.session_state["_restore_candidate_file"] = file
    at.run()


def test_a_damaged_copy_names_its_file_and_where_to_look(project):
    state = project.export_json()
    state["variables"] = [{"nope": 1}]
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    _offer(at, state)
    assert not at.exception
    assert _errors(at) == [
        "**oat copy 2026-10-01.json** is damaged and can't be opened. Try "
        "another copy from your FoodOptimizer folder."]
    for retired in ("section", "shape", "variables"):
        assert retired not in at.error[0].value


def test_a_file_that_is_not_a_copy_says_so_by_name(project):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    _offer(at, {"hello": 1}, file="notes.json")
    assert _errors(at) == [
        "**notes.json** is not a Food Optimizer copy. Choose another file."]


def test_a_copy_from_a_newer_app_says_update(project):
    state = project.export_json()
    state["CLASS_VERSION"] = FoodOptimizer.CLASS_VERSION + 50
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    _offer(at, state)
    assert _errors(at) == [
        "**oat copy 2026-10-01.json** is from a newer Food Optimizer. Update "
        "the app, then open it again."]


def test_a_saved_copies_row_whose_file_is_gone_names_the_row(project, monkeypatch):
    name = LocalStorage().archive("Oat cookie", "pre_edit", copy=True)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    # Gone between the row being drawn and the click (load() is None).
    real = LocalStorage.load
    monkeypatch.setattr(LocalStorage, "load",
                        lambda self, n: None if n == name else real(self, n))
    next(b for b in at.button if b.key == f"open_copy_{name}").click()
    at.run()
    assert not at.exception
    (shown,) = _errors(at)
    assert shown.startswith("**" + wording.SAFETY_COPY_REASONS["pre_edit"])
    assert shown.endswith("is no longer in your FoodOptimizer folder. "
                          "Choose another copy.")


def test_a_copy_refused_on_replace_names_it_and_the_project(project, monkeypatch):
    donor = FoodOptimizer("donor")
    donor.add_ingredient("Flour", 0, 100)
    state = donor.export_json()
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    _offer(at, state, file="donor.json")

    def boom(self, st):
        raise TypeError("half way")
    monkeypatch.setattr(FoodOptimizer, "import_json", boom)
    _button(at, wording.YES_REPLACE).click()
    at.run()
    assert _errors(at) == [
        "**donor.json** could not be opened, and Oat cookie is unchanged. "
        "Try another copy."]


def test_replacing_over_another_windows_change_offers_reload(project):
    """'Click Reload project' named a button that was not on screen."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "Oat cookie"
    at.run()
    other = FoodOptimizer("Oat cookie")
    other.add_objective("Crunch", 1.0)
    donor = FoodOptimizer("donor")
    donor.add_ingredient("Flour", 0, 100)
    at.session_state["_restore_candidate"] = donor.export_json()
    at.run()
    _button(at, wording.YES_REPLACE).click()
    at.run()
    assert not at.exception
    assert wording.changed_elsewhere("Oat cookie") in _errors(at)
    assert any(w.value == wording.SAVE_ERROR_WARNING for w in at.warning)
    assert wording.RELOAD_PROJECT in [b.label for b in at.button]
    assert "NOT" not in " ".join(_errors(at))


# ------------------------------------------------------------------ #
#  The disk refuses a save or a copy
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("code, line", [
    (errno.ENOSPC, "Oat cookie was not saved: the disk is full. Free up space, "
                   "then try again."),
    (errno.EACCES, "Oat cookie was not saved: the FoodOptimizer folder is "
                   "read-only. Allow changes to it in Finder, then try again."),
    (errno.EBUSY, "Oat cookie was not saved: another program is using its "
                  "file. Close that program, then try again."),
    (errno.EIO, "Oat cookie was not saved: the file could not be written. "
                "Try again."),
])
def test_a_refused_save_is_the_banner_not_a_traceback(project, monkeypatch, code, line):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()

    def refuse(*a, **k):
        raise OSError(code, os.strerror(code))
    monkeypatch.setattr(os, "replace", refuse)
    at.session_state["optimizer"].save()
    at.run()
    assert not at.exception
    assert line in _errors(at)
    assert any(w.value == wording.SAVE_ERROR_WARNING for w in at.warning)
    assert not list(pathlib.Path().glob("*.tmp"))


def test_a_new_project_on_a_full_disk_says_so(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()

    def refuse(*a, **k):
        raise OSError(errno.ENOSPC, "No space left on device")
    monkeypatch.setattr(os, "replace", refuse)
    _create(at, "Granola")
    assert not at.exception
    assert _errors(at) == [wording.not_saved("Granola", "full")]


def test_a_new_project_refused_while_another_is_open_draws_no_traceback(project, monkeypatch):
    """The box and the fold were parked to clear before the write failed;
    the clears then landed on widgets already drawn (a traceback under the
    line) — only reachable once the line itself stopped being one."""
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()

    def refuse(*a, **k):
        raise PermissionError(errno.EACCES, "Permission denied")
    monkeypatch.setattr(os, "replace", refuse)
    _create(at, "Granola")
    assert not at.exception
    assert _errors(at) == [wording.not_saved("Granola", "read_only")]
    monkeypatch.undo()
    at.run()
    assert not at.exception


def test_a_refused_copy_before_delete_changes_nothing(project, monkeypatch):
    def refuse(*a, **k):
        raise PermissionError(errno.EACCES, "Permission denied")
    monkeypatch.setattr(os, "rename", refuse)
    with pytest.raises(StorageError) as caught:
        LocalStorage().archive("Oat cookie", "deleted", copy=False)
    assert str(caught.value) == (
        "Nothing changed in Oat cookie: the FoodOptimizer folder is read-only. "
        "Allow changes to it in Finder, then try again.")
    assert pathlib.Path("Oat cookie.pkl").exists()


def test_an_old_copy_that_will_not_delete_says_why(project, monkeypatch):
    name = LocalStorage().archive("Oat cookie", "pre_edit", copy=True)

    def refuse(*a, **k):
        raise OSError(errno.EBUSY, "Resource busy")
    monkeypatch.setattr(os, "remove", refuse)
    with pytest.raises(StorageError) as caught:
        LocalStorage().delete_archive(name)
    assert str(caught.value) == (
        "An old copy could not be deleted: another program is using it. "
        "Close that program, then try again.")


def test_a_crash_mid_save_still_propagates(project, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("simulated crash")
    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(RuntimeError):
        LocalStorage().save("Oat cookie", {"v": 1})


# ------------------------------------------------------------------ #
#  A project that will not open
# ------------------------------------------------------------------ #

def test_a_damaged_project_names_its_file_and_the_way_on(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "broken.pkl").write_bytes(b"\x80\x04 not json \xff\xfe")
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = "broken"
    at.run()
    assert not at.exception
    assert wording.project_file_unreadable("broken") in _errors(at)
    assert "**broken.pkl**" in wording.project_file_unreadable("broken")
    assert [i.value for i in at.info] == [
        "In the sidebar, use Manage project › Open a saved copy or Clear "
        "project contents. The damaged file is copied first."]
    assert not any(c.value.startswith("Saved") for c in at.sidebar.caption), \
        [c.value for c in at.sidebar.caption]


def test_a_damaged_sample_points_to_start_this_example_over(tmp_path, monkeypatch):
    """A sample has no Clear project contents: it is started over."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / f"{sp.BURGER}.pkl").write_bytes(b"not json")
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.session_state["_loaded_project"] = sp.BURGER
    at.run()
    assert not at.exception
    (info,) = [i.value for i in at.info]
    assert wording.START_SAMPLE_OVER_LABEL in info
    assert wording.START_OVER_LABEL not in info
    assert wording.START_SAMPLE_OVER_LABEL in [b.label for b in at.button]


def test_a_project_that_will_not_load_names_its_file(project, monkeypatch):
    monkeypatch.setattr(FoodOptimizer, "import_json",
                        lambda self, st: (_ for _ in ()).throw(KeyError("x")))
    assert FoodOptimizer("Oat cookie").load_error == (
        "**Oat cookie.pkl** is damaged and can't be opened.")


def test_a_locked_project_file_says_close_the_other_program(project, monkeypatch):
    real_open = open

    def locked(path, *a, **k):
        if str(path).endswith("Oat cookie.pkl"):
            raise PermissionError(errno.EACCES, "Permission denied")
        return real_open(path, *a, **k)
    monkeypatch.setattr("builtins.open", locked)
    with pytest.raises(StorageError) as caught:
        LocalStorage().load("Oat cookie")
    assert str(caught.value) == (
        "**Oat cookie.pkl** could not be read. Close any other program using "
        "it, then open the project again.")


# ------------------------------------------------------------------ #
#  A shipped sample that is missing or damaged
# ------------------------------------------------------------------ #

def test_a_damaged_sample_file_is_a_plain_line_not_a_traceback(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    broken = tmp_path / "shipped.json"
    broken.write_text("{ not json")
    real = sp._shipped_path
    monkeypatch.setattr(sp, "_shipped_path",
                        lambda n: broken if n == sp.BURGER else real(n))
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    _button(at, wording.SEE_EXAMPLE_SETUP_LABEL).click()
    at.run()
    assert not at.exception
    assert _errors(at) == [
        f"The app's file for {sp.BURGER} is missing or damaged. Download "
        "Food Optimizer again to replace it."]
    assert not (tmp_path / f"{sp.BURGER}.pkl").exists()


def test_a_sample_read_failure_is_still_an_oserror():
    with pytest.raises(OSError):
        raise sp.SampleUnreadable(wording.sample_unreadable(sp.BURGER))


# ------------------------------------------------------------------ #
#  4 · Review with nothing scored
# ------------------------------------------------------------------ #

def test_the_empty_review_names_the_tab_that_fills_it():
    assert wording.INSIGHTS_NONE == "Score a formulation in 3 · Results to see the Review."


def test_the_storage_messages_are_plain():
    """No retired or engineering words, no capitals, no 'Error'."""
    lines = [wording.not_saved("P", c) for c in ("full", "read_only", "in_use", None)]
    lines += [wording.nothing_changed("P", "full"), wording.copy_not_deleted(),
              wording.changed_elsewhere("P"), wording.project_file_unreadable("P"),
              wording.project_not_found("P"), wording.project_file_locked("P")]
    for line in lines:
        assert "NOT" not in line and "Error" not in line and "!" not in line, line
        assert ".tmp" not in line and "Errno" not in line, line
        assert len(line.split()) <= 20, line
    assert storage_backend.disk_cause(OSError(errno.EROFS, "x")) == "read_only"
    assert storage_backend.disk_cause(RuntimeError()) is None
