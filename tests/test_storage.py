import json
import os
import pickle
import sys
import time

import pytest

from storage import LocalStorage, StorageError


@pytest.fixture
def local(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return LocalStorage()


class TestLocalStorage:
    def test_flags(self, local):
        assert local.persist_empty_on_init is False
        assert local.persist_after_load is True

    def test_save_load_roundtrip(self, local):
        local.save("proj", {"a": 1})
        assert local.load("proj") == {"a": 1}
        assert json.loads(open("proj.pkl", encoding="utf-8").read()) == {"a": 1}  # JSON on disk

    def test_load_missing_returns_none(self, local):
        assert local.load("nope") is None

    def test_load_refuses_legacy_pickle_with_hint(self, local, tmp_path):
        (tmp_path / "old.pkl").write_bytes(pickle.dumps({"legacy": True}))
        with pytest.raises(StorageError, match="early version"):
            local.load("old")

    def test_load_damaged_raises_storage_error(self, local, tmp_path):
        (tmp_path / "bad.pkl").write_bytes(b"\x80\x04 not json not pickle \xff")
        with pytest.raises(StorageError):
            local.load("bad")

    def test_save_io_error_propagates_raw(self, local, monkeypatch):
        local.save("p", {"v": 1})

        def boom(*a, **k):
            raise RuntimeError("simulated crash")

        monkeypatch.setattr(os, "replace", boom)
        with pytest.raises(RuntimeError):   # NOT StorageError
            local.save("p", {"v": 2})
        assert local.load("p") == {"v": 1}  # old content survives

    def test_list_projects_excludes_archives_but_list_archives_has_them(self, local):
        local.save("a", {})
        local.save("a_archived", {})
        assert local.list_projects() == ["a"]
        assert local.list_archives() == ["a_archived"]

    def test_exists(self, local):
        assert not local.exists("x")
        local.save("x", {})
        assert local.exists("x")

    def test_archive_move(self, local):
        local.save("p", {"v": 1})
        name = local.archive("p", "archived", copy=False)
        assert name == "p_archived"
        assert not local.exists("p")
        assert local.load("p_archived") == {"v": 1}

    def test_archive_copy(self, local):
        local.save("p", {"v": 1})
        name = local.archive("p", "pre_rewind", copy=True)
        assert name == "p_pre_rewind"
        assert local.exists("p") and local.exists("p_pre_rewind")

    def test_archive_counter_suffix(self, local):
        local.save("p", {})
        local.save("p_archived", {})
        assert local.archive("p", "archived") == "p_archived_1"

    def test_archive_missing_is_noop(self, local):
        assert local.archive("ghost", "archived") is None


def test_local_storage_refuses_to_overwrite_a_newer_file(tmp_path, monkeypatch):
    """Two windows on one project: the stale one must get a conflict error,
    not silently clobber the other's work."""
    monkeypatch.chdir(tmp_path)
    window_a, window_b = LocalStorage(), LocalStorage()
    window_a.save("p", {"v": 1})
    window_b.load("p")
    time.sleep(0.01)  # coarse-mtime filesystems
    window_a.save("p", {"v": 2})
    with pytest.raises(StorageError, match="another window"):
        window_b.save("p", {"v": 3})
    assert json.loads((tmp_path / "p.pkl").read_text(encoding="utf-8"))["v"] == 2


def test_local_storage_save_after_own_save_is_fine(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = LocalStorage()
    s.save("p", {"v": 1})
    s.save("p", {"v": 2})
    assert json.loads((tmp_path / "p.pkl").read_text(encoding="utf-8"))["v"] == 2


def test_local_storage_reload_clears_conflict(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a, b = LocalStorage(), LocalStorage()
    a.save("p", {"v": 1})
    b.load("p")
    time.sleep(0.01)
    a.save("p", {"v": 2})
    b.load("p")              # user reloads
    b.save("p", {"v": 3})    # now allowed
    assert json.loads((tmp_path / "p.pkl").read_text(encoding="utf-8"))["v"] == 3


def test_is_stale_true_after_another_instance_saves_then_false_after_reload(tmp_path, monkeypatch):
    """is_stale must report a conflict without mutating state (unlike save()),
    so a caller can check before committing to an action such as Restore."""
    monkeypatch.chdir(tmp_path)
    a, b = LocalStorage(), LocalStorage()
    a.save("p", {"v": 1})
    b.load("p")
    assert not b.is_stale("p")
    time.sleep(0.01)  # coarse-mtime filesystems
    a.save("p", {"v": 2})
    assert b.is_stale("p")
    b.load("p")
    assert not b.is_stale("p")


def test_list_projects_hides_archives(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = LocalStorage()
    for name in ["Cookie", "Cookie_archived", "Cookie_archived_2", "Cookie_pre_rewind",
                 "Cookie_pre_restore_1", "Cookie_pre_delete", "Bread"]:
        s.save(name, {"v": 1})
    assert s.list_projects() == ["Bread", "Cookie"]
    assert "Cookie_archived_2" in s.list_archives()


def test_most_recent_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = LocalStorage()
    s.save("Old", {"v": 1})
    time.sleep(0.01)
    s.save("New", {"v": 1})
    assert s.most_recent_project() == "New"
    assert LocalStorage().most_recent_project() == "New"  # fresh instance, reads disk


def test_last_opened_is_the_project_opened_not_the_newest_file(tmp_path, monkeypatch):
    """S4: a fresh session reopens the project the reader opened."""
    monkeypatch.chdir(tmp_path)
    s = LocalStorage()
    assert s.last_opened() is None
    s.save("Opened", {"v": 1})
    s.remember_opened("Opened")
    time.sleep(0.01)
    s.save("Written later", {"v": 1})
    assert LocalStorage().last_opened() == "Opened"
    assert s.most_recent_project() == "Written later"
    # A project that left the list is no answer: the caller falls back.
    s.archive("Opened", "deleted")
    assert LocalStorage().last_opened() is None
    assert "Opened" not in s.list_projects()


@pytest.mark.skipif(sys.platform != "win32", reason="M9: only Windows' Explorer lists dot files")
def test_last_opened_is_hidden_on_windows(tmp_path, monkeypatch):
    """M9: Explorer lists dot files, so on Windows the reopen note is a
    hidden file beside the projects. Writing it again still works: Windows
    refuses to overwrite a hidden file opened for writing."""
    import ctypes
    from ctypes import wintypes
    get_attributes = ctypes.WinDLL("kernel32").GetFileAttributesW
    get_attributes.argtypes = [wintypes.LPCWSTR]
    get_attributes.restype = wintypes.DWORD          # INVALID_FILE_ATTRIBUTES reads as 0xFFFFFFFF
    monkeypatch.chdir(tmp_path)
    s = LocalStorage()
    s.save("First", {"v": 1})
    s.save("Second", {"v": 1})
    s.remember_opened("First")
    s.remember_opened("Second")
    assert (tmp_path / ".last_opened").read_text(encoding="utf-8") == "Second"
    attributes = get_attributes(str(tmp_path / ".last_opened"))
    assert attributes != 0xFFFFFFFF and attributes & 0x2  # FILE_ATTRIBUTE_HIDDEN
    assert LocalStorage().last_opened() == "Second"


def test_no_file_written_on_init(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from food_bo import FoodOptimizer
    FoodOptimizer("ghost")
    assert not (tmp_path / "ghost.pkl").exists()


def test_saved_at_reads_the_file_time_and_is_none_without_a_file(tmp_path, monkeypatch):
    """The sidebar's saved line must survive a fresh session: last_saved_at is
    only set by a save, so the file's own modification time is the fallback."""
    from datetime import datetime, timedelta
    monkeypatch.chdir(tmp_path)
    s = LocalStorage()
    assert s.saved_at("nope") is None
    s.save("proj", {"project_name": "proj"})
    stamp = s.saved_at("proj")
    assert isinstance(stamp, datetime)
    assert stamp.tzinfo is not None                     # aware, like last_saved_at
    assert abs(stamp - datetime.now().astimezone()) < timedelta(seconds=30)


@pytest.mark.parametrize("name", ["con", "CON", "Con.backup", "nul", "aux", "prn",
                                  "com1", "COM9", "lpt1 ", "LPT9.x"])
def test_windows_reserved_names_are_reserved(name):
    """Windows will not save CON.pkl (or con.backup.pkl): the part before the
    first dot names a device, whatever the case or trailing spaces."""
    from storage import is_reserved_name
    assert is_reserved_name(name)


@pytest.mark.parametrize("name", ["COM10", "console", "Oat cookie", "com", "lpt0"])
def test_ordinary_names_are_not_reserved(name):
    from storage import is_reserved_name
    assert not is_reserved_name(name)


def _sharing_violation():
    """What Windows raises when another program (antivirus, OneDrive,
    Excel) holds the file: PermissionError, errno EACCES, winerror 32."""
    error = PermissionError(13, "The process cannot access the file")
    error.winerror = 32
    return error


def _locked_twice(real, monkeypatch, name):
    calls = {"n": 0}

    def flaky(*args):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise _sharing_violation()
        return real(*args)
    monkeypatch.setattr(f"storage.os.{name}", flaky)
    monkeypatch.setattr("storage.time.sleep", lambda s: None)
    return calls


def test_save_retries_a_briefly_locked_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = _locked_twice(os.replace, monkeypatch, "replace")
    LocalStorage().save("p", {"v": 1})
    assert calls["n"] == 3
    assert json.loads((tmp_path / "p.pkl").read_text(encoding="utf-8")) == {"v": 1}
    assert list(tmp_path.glob("*.tmp")) == []


def test_save_reports_a_file_that_stays_locked(tmp_path, monkeypatch):
    import wording

    def always(*args):
        raise _sharing_violation()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("storage.os.replace", always)
    monkeypatch.setattr("storage.time.sleep", lambda s: None)
    with pytest.raises(StorageError) as caught:
        LocalStorage().save("p", {"v": 1})
    assert str(caught.value) == wording.not_saved("p", "in_use")


def test_disk_cause_reads_windows_sharing_errors():
    from storage import disk_cause
    assert disk_cause(_sharing_violation()) == "in_use"
    lock = PermissionError(13, "locked")
    lock.winerror = 33
    assert disk_cause(lock) == "in_use"


def test_errno_names_missing_on_some_platforms_do_not_break_import():
    import storage
    assert None not in storage._DISK_CAUSES


def test_archive_retries_a_briefly_locked_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = LocalStorage()
    s.save("p", {"v": 1})
    calls = _locked_twice(os.rename, monkeypatch, "rename")
    assert s.archive("p", "pre_delete") == "p_pre_delete"
    assert calls["n"] == 3
    assert (tmp_path / "p_pre_delete.pkl").exists()


def _held_by_another_program():
    """What Windows raises when os.replace lands on a file another program
    holds open without delete-sharing (OneDrive, antivirus, a second app
    window): PermissionError, winerror 5."""
    error = PermissionError(13, "Access is denied")
    error.winerror = 5
    return error


def test_save_retries_a_project_file_another_program_holds(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    real, calls = os.replace, {"n": 0}

    def held_twice(*args):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise _held_by_another_program()
        return real(*args)
    monkeypatch.setattr("storage.os.replace", held_twice)
    monkeypatch.setattr("storage.time.sleep", lambda s: None)
    LocalStorage().save("p", {"v": 1})
    assert calls["n"] == 3
    assert json.loads((tmp_path / "p.pkl").read_text(encoding="utf-8")) == {"v": 1}


def test_a_save_refused_at_the_replace_is_in_use_not_read_only(tmp_path, monkeypatch):
    """The .tmp beside it was just written, so the folder is not read-only:
    a refusal at the replace is another program holding the project file."""
    import wording

    def always(*args):
        raise _held_by_another_program()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("storage.os.replace", always)
    monkeypatch.setattr("storage.time.sleep", lambda s: None)
    with pytest.raises(StorageError) as caught:
        LocalStorage().save("p", {"v": 1})
    assert str(caught.value) == wording.not_saved("p", "in_use")


def test_saved_at_reads_a_file_dated_near_1970(tmp_path, monkeypatch):
    """Windows cannot turn a naive time from 1970's first day into local
    time (OSError 22), and files unpacked from some archives carry such
    dates: the saved line must still have a time to show."""
    monkeypatch.chdir(tmp_path)
    s = LocalStorage()
    s.save("p", {"v": 1})
    os.utime("p.pkl", (1000, 1000))
    when = s.saved_at("p")
    assert when is not None and when.tzinfo is not None
    assert abs(when.timestamp() - 1000) < 1
