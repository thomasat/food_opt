import json
import os
import pickle
import time

import pytest

from storage import LocalStorage, StorageError


@pytest.fixture
def local(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return LocalStorage()


class TestLocalStorage:
    def test_flags(self, local):
        assert local.persist_empty_on_init is True
        assert local.persist_after_load is True

    def test_save_load_roundtrip(self, local):
        local.save("proj", {"a": 1})
        assert local.load("proj") == {"a": 1}
        assert json.loads(open("proj.pkl").read()) == {"a": 1}  # JSON on disk

    def test_load_missing_returns_none(self, local):
        assert local.load("nope") is None

    def test_load_legacy_pickle(self, local, tmp_path):
        (tmp_path / "old.pkl").write_bytes(pickle.dumps({"legacy": True}))
        assert local.load("old") == {"legacy": True}

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

    def test_list_includes_archives(self, local):
        local.save("a", {})
        local.save("a_archived", {})
        assert local.list_projects() == ["a", "a_archived"]

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
    assert json.loads((tmp_path / "p.pkl").read_text())["v"] == 2


def test_local_storage_save_after_own_save_is_fine(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = LocalStorage()
    s.save("p", {"v": 1})
    s.save("p", {"v": 2})
    assert json.loads((tmp_path / "p.pkl").read_text())["v"] == 2


def test_local_storage_reload_clears_conflict(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    a, b = LocalStorage(), LocalStorage()
    a.save("p", {"v": 1})
    b.load("p")
    time.sleep(0.01)
    a.save("p", {"v": 2})
    b.load("p")              # user reloads
    b.save("p", {"v": 3})    # now allowed
    assert json.loads((tmp_path / "p.pkl").read_text())["v"] == 3
