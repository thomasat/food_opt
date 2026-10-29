"""The launcher's files, byte for byte as desktop/launcher.sh writes them."""
import hashlib
import re
import time

import pytest

from launcher import files


def test_status_line_bytes(tmp_path):
    assert files.write_status(tmp_path, "Starting the app", None) is True
    assert (tmp_path / "status.txt").read_bytes() == b"Starting the app|\n"
    files.write_status(tmp_path, "x", 38)
    assert (tmp_path / "status.txt").read_bytes() == b"x|38\n"
    assert not (tmp_path / "status.txt.tmp").exists()


def test_status_write_failure_never_raises(tmp_path):
    (tmp_path / "status.txt").mkdir()
    assert files.write_status(tmp_path, "Starting the app", None) is False


def test_clear_status_removes_both_files(tmp_path):
    (tmp_path / "status.txt").write_bytes(b"x|\n")
    (tmp_path / "status.txt.tmp").write_bytes(b"x|\n")
    files.clear_status(tmp_path)
    files.clear_status(tmp_path)
    assert not (tmp_path / "status.txt").exists() and not (tmp_path / "status.txt.tmp").exists()


def test_port_file_bytes_and_reading(tmp_path):
    files.write_port_file(tmp_path, 18501, 4242)
    assert (tmp_path / "server.port").read_bytes() == b"18501 4242\n"
    assert files.read_port_file(tmp_path) == (18501, 4242)
    assert files.port_file_is(tmp_path, 18501, 4242)
    assert not files.port_file_is(tmp_path, 1, 1)
    (tmp_path / "server.port").write_bytes(b"abc")
    assert files.read_port_file(tmp_path) is None
    (tmp_path / "server.port").write_bytes(b"18501\n")
    assert files.read_port_file(tmp_path) == (18501, None)
    (tmp_path / "server.port").unlink()
    assert files.read_port_file(tmp_path) is None


def test_base_file_and_removal(tmp_path):
    files.write_base_file(tmp_path, "abc")
    files.write_port_file(tmp_path, 1, 2)
    assert (tmp_path / "server.base").read_bytes() == b"abc\n"
    files.remove_port_files(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_atomic_write_retries_a_briefly_locked_file(tmp_path, monkeypatch):
    real = files.os.replace
    calls = []

    def flaky(a, b):
        calls.append(a)
        if len(calls) <= 2:
            raise PermissionError("held by the indexer")
        return real(a, b)

    monkeypatch.setattr(files.os, "replace", flaky)
    files.atomic_write(tmp_path / "f", b"data", delay=0.001)
    assert (tmp_path / "f").read_bytes() == b"data"
    assert len(calls) == 3


def test_atomic_write_gives_up_after_five_tries(tmp_path, monkeypatch):
    calls = []

    def locked(a, b):
        calls.append(a)
        raise PermissionError("held")

    monkeypatch.setattr(files.os, "replace", locked)
    with pytest.raises(PermissionError):
        files.atomic_write(tmp_path / "f", b"data", delay=0.001)
    assert len(calls) == 5


def test_lock_is_taken_when_free(tmp_path):
    lock = tmp_path / "launch.lock"
    assert files.acquire_lock(lock, 1234, alive=lambda p: True) == (True, None)
    assert (lock / "pid").read_text(encoding="utf-8") == "1234\n"
    assert files.lock_owner(lock) == 1234


def _held(tmp_path, content):
    lock = tmp_path / "launch.lock"
    lock.mkdir()
    (lock / "pid").write_text(content, encoding="utf-8")
    return lock


def test_lock_held_by_a_live_pid_is_refused(tmp_path):
    lock = _held(tmp_path, "77\n")
    assert files.acquire_lock(lock, 1234, alive=lambda p: True) == (False, 77)
    assert files.lock_owner(lock) == 77


def test_lock_held_by_a_dead_pid_is_taken(tmp_path):
    lock = _held(tmp_path, "77\n")
    assert files.acquire_lock(lock, 1234, alive=lambda p: False) == (True, None)
    assert (lock / "pid").read_text(encoding="utf-8") == "1234\n"


def test_empty_lock_is_taken_after_the_grace_window(tmp_path):
    lock = tmp_path / "launch.lock"
    lock.mkdir()
    slept = []
    assert files.acquire_lock(lock, 1234, alive=lambda p: True,
                              sleep=slept.append) == (True, None)
    assert slept == [1, 1, 1, 1, 1]
    assert files.lock_owner(lock) == 1234


def test_release_only_removes_our_own_lock(tmp_path):
    lock = _held(tmp_path, "77\n")
    assert files.release_lock(lock, 1234) is False
    assert lock.exists()
    assert files.release_lock(lock, 77) is True
    assert not lock.exists()
    assert files.release_lock(lock, 77) is False


NOW = time.strptime("2026-10-24 10:15:00", "%Y-%m-%d %H:%M:%S")


def _projects(tmp_path):
    d = tmp_path / "projects"
    d.mkdir()
    for name, data in (("A.pkl", b"aaa"), ("b.pkl", b"bbb"), ("Gamma 2.pkl", b"ggg")):
        (d / name).write_bytes(data)
    (d / "notes.txt").write_bytes(b"not a project")
    return d


def test_backup_copies_every_project_and_records_the_sum(tmp_path):
    d = _projects(tmp_path)
    made = files.backup_projects(d, now=NOW)
    assert made == d / "backups" / "2026-10-24_101500"
    assert sorted(p.name for p in made.iterdir()) == ["A.pkl", "Gamma 2.pkl", "b.pkl"]
    for p in made.iterdir():
        assert p.read_bytes() == (d / p.name).read_bytes()
    want = hashlib.sha256(b"aaa" + b"ggg" + b"bbb").hexdigest()
    assert (d / "backups" / ".last_sum").read_bytes() == want.encode("ascii")


def test_unchanged_projects_make_no_second_backup(tmp_path):
    d = _projects(tmp_path)
    files.backup_projects(d, now=NOW)
    later = time.strptime("2026-10-24 11:15:00", "%Y-%m-%d %H:%M:%S")
    assert files.backup_projects(d, now=later) is None
    assert len([p for p in (d / "backups").iterdir() if p.is_dir()]) == 1


def test_backups_keep_the_newest_ten(tmp_path):
    d = _projects(tmp_path)
    for i in range(10):
        (d / "backups" / f"2000-01-01_0000{i:02d}").mkdir(parents=True)
    assert files.backup_projects(d, now=NOW) is not None
    dirs = sorted(p.name for p in (d / "backups").iterdir() if p.is_dir())
    assert len(dirs) == 10
    assert "2000-01-01_000000" not in dirs and "2026-10-24_101500" in dirs


def test_no_projects_no_backup_folder(tmp_path):
    d = tmp_path / "projects"
    d.mkdir()
    assert files.backup_projects(d, now=NOW) is None
    assert not (d / "backups").exists()


def test_rotate_log_moves_the_log_when_no_launch_runs(tmp_path):
    log = tmp_path / "launcher.log"
    log.write_text("new\n", encoding="utf-8")
    (tmp_path / "launcher.log.1").write_text("older\n", encoding="utf-8")
    assert files.rotate_log(log, tmp_path / "launch.lock") == (True, None)
    assert not log.exists()
    assert (tmp_path / "launcher.log.1").read_text(encoding="utf-8") == "new\n"


def test_rotate_log_leaves_a_live_launchs_log(tmp_path):
    log = tmp_path / "launcher.log"
    log.write_text("live\n", encoding="utf-8")
    (tmp_path / "launch.lock").mkdir()
    assert files.rotate_log(log, tmp_path / "launch.lock") == (False, None)
    assert log.read_text(encoding="utf-8") == "live\n"
    assert files.rotate_log(tmp_path / "none.log", tmp_path / "x") == (False, None)


def test_rotate_log_skips_a_log_in_use(tmp_path, monkeypatch):
    log = tmp_path / "launcher.log"
    log.write_text("held\n", encoding="utf-8")

    def held(a, b):
        raise PermissionError("in use")

    monkeypatch.setattr(files.os, "replace", held)
    assert files.rotate_log(log, tmp_path / "launch.lock") == (False, "in use")
    assert log.read_text(encoding="utf-8") == "held\n"


def test_log_line_format(tmp_path):
    import io
    stream = io.StringIO()
    log = files.Log(stream)
    log.open(tmp_path / "launcher.log")
    log.say("hello")
    line = (tmp_path / "launcher.log").read_bytes().decode("utf-8")
    assert re.match(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d hello\n$", line), line
    assert stream.getvalue() == line
    with log.child_handle() as h:
        h.write(b"child\n")
    assert (tmp_path / "launcher.log").read_bytes().endswith(b"child\n")


def test_log_survives_a_closed_stream(tmp_path):
    import io
    stream = io.StringIO()
    stream.close()
    log = files.Log(stream)
    log.open(tmp_path / "launcher.log")
    log.say("still logged")
    assert "still logged" in (tmp_path / "launcher.log").read_text(encoding="utf-8")


def test_atomic_write_retries_a_tmp_file_it_cannot_open_yet(tmp_path, monkeypatch):
    real, calls = open, []

    def flaky(path, mode="r", *a, **k):
        if str(path).endswith(".tmp"):
            calls.append(path)
            if len(calls) <= 2:
                raise PermissionError("held by Defender")
        return real(path, mode, *a, **k)

    monkeypatch.setattr(files, "open", flaky, raising=False)
    files.atomic_write(tmp_path / "f", b"data", delay=0.001)
    assert (tmp_path / "f").read_bytes() == b"data" and len(calls) == 3


def test_release_reports_a_lock_it_could_not_remove(tmp_path, monkeypatch):
    lock = tmp_path / "launch.lock"
    assert files.acquire_lock(lock, 4242, alive=lambda p: False)[0]
    monkeypatch.setattr(files.shutil, "rmtree", lambda *a, **k: None)
    assert files.release_lock(lock, 4242) is False
