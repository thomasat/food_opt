"""One behaviour suite for both launchers: desktop/launcher.sh (the Mac's,
unchanged) and the shared Python launcher. Every test in the first part
runs against each; a test that fails for launcher.sh is wrong about it.
Everything happens under a home folder named "José Müller"."""
import hashlib
import os
import re
import socket
import subprocess
import sys
import time

import pytest

from tests.desktop_launcher.conftest import healthy_launch, slow
from tests.desktop_launcher.harness import wait_for

from launcher import procinfo

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")

LOAD = b"Loading the app's components|\n"
STARTING = b"Starting the app|\n"


def _read(path):
    try:
        return path.read_bytes()
    except OSError:
        return None


def _listing(path):
    return sorted(str(p.relative_to(path)) for p in path.rglob("*")) if path.exists() else None


def _logged(world, text, timeout=10):
    return wait_for(lambda: text in world.log_text(), timeout)


def _gone(path, timeout=5):
    return wait_for(lambda: not path.exists(), timeout)


def _dead(pid, timeout=10):
    return wait_for(lambda: not procinfo.pid_alive(pid), timeout)


def _serve_pids(world):
    return [r["pid"] for r in world.records("serve")]


def test_unsupported_machine_exits_2_and_creates_nothing(impl, world):
    launch = impl.start(world, env={"FOODOPT_TEST_ARCH": "x86_64"})
    before = _listing(world.state_dir)   # launcher.sh's bundle: setup done by the harness
    assert launch.wait(10) == 2
    assert "unsupported machine" in launch.output()
    assert _listing(world.state_dir) == before
    assert not world.projects_dir.exists()
    assert world.records() == []


def test_healthy_launch_writes_the_port_file_and_clears_status(impl, world):
    launch, port, pid = healthy_launch(impl, world)
    assert re.match(rb"^(\d+) (\d+)\n$", world.port_file.read_bytes())
    assert procinfo.pid_alive(pid)
    assert world.health() == 200
    assert _logged(world, f"server healthy on port {port} (pid {pid})")
    assert _gone(world.status_file) and _gone(world.lock_dir)


def test_status_lines_in_order(impl, world):
    launch = impl.start(world, env={"FOODOPT_FAKE_WARMUP_SECS": "1.5",
                                    "FOODOPT_FAKE_HEALTH_DELAY_SECS": "1"})
    seen = []
    end = time.monotonic() + 30
    while time.monotonic() < end and "server healthy" not in world.log_text():
        data = _read(world.status_file)
        if data is not None and (not seen or seen[-1] != data):
            seen.append(data)
        time.sleep(0.02)
    assert seen == [LOAD, STARTING], (seen, launch.output())


def test_port_base_skips_a_port_in_use(impl, world):
    from tests.desktop_launcher.harness import free_port_block
    base = free_port_block()
    s = socket.socket()
    s.bind(("127.0.0.1", base))
    s.listen()
    try:
        _, port, _ = healthy_launch(impl, world, env={"FOODOPT_PORT_BASE": str(base)})
        assert port == base + 1
    finally:
        s.close()


def test_a_second_launch_replaces_the_orphaned_server(impl, world):
    _, _, p1 = healthy_launch(impl, world)
    impl.start(world)
    assert _dead(p1, 15)
    assert wait_for(lambda: (world.port_pid() or (0, p1))[1] != p1 and world.health() == 200, 20)
    assert _logged(world, f"stopping orphaned server (pid {p1}) from a previous session")
    alive = [p for p in _serve_pids(world) if procinfo.pid_alive(p)]
    assert len(alive) == 1 and alive[0] != p1


def test_a_launch_in_progress_makes_the_next_exit_0(impl, world):
    world.lock_dir.mkdir(parents=True)
    (world.lock_dir / "pid").write_bytes(f"{os.getpid()}\n".encode())
    world.status_file.write_bytes(b"Finishing setup (step 3 of 3)|100\n")
    launch = impl.start(world)
    assert launch.wait(15) == 0
    assert (f"another launch is already in progress (pid {os.getpid()}) - exiting"
            in launch.output())
    assert (world.lock_dir / "pid").read_bytes() == f"{os.getpid()}\n".encode()
    assert world.status_file.read_bytes() == b"Finishing setup (step 3 of 3)|100\n"


def _dead_pid():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def test_a_stale_lock_with_a_dead_pid_is_taken(impl, world):
    world.lock_dir.mkdir(parents=True)
    (world.lock_dir / "pid").write_bytes(f"{_dead_pid()}\n".encode())
    healthy_launch(impl, world)
    assert _gone(world.lock_dir)


@slow
def test_an_empty_lock_is_taken_after_the_grace_window(impl, world):
    world.lock_dir.mkdir(parents=True)
    healthy_launch(impl, world, timeout=20)
    assert _gone(world.lock_dir)


def test_the_log_is_rotated_only_when_no_launch_runs(impl, world):
    world.state_dir.mkdir(parents=True, exist_ok=True)
    world.log.write_bytes(b"old\n")
    healthy_launch(impl, world)
    assert (world.state_dir / "launcher.log.1").read_bytes() == b"old\n"
    assert b"old\n" not in world.log.read_bytes()


def _pkls(world, names):
    world.projects_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        (world.projects_dir / name).write_bytes(f"project {name}".encode())


def _backup_dirs(world):
    if not world.backups.exists():
        return []
    return sorted(p for p in world.backups.iterdir() if p.is_dir())


def test_backups_on_launch(impl, world):
    _pkls(world, ["A.pkl", "b.pkl"])
    launch, _, _ = healthy_launch(impl, world)
    dirs = _backup_dirs(world)
    assert len(dirs) == 1 and re.fullmatch(r"\d{4}-\d\d-\d\d_\d{6}", dirs[0].name)
    for name in ("A.pkl", "b.pkl"):
        assert (dirs[0] / name).read_bytes() == (world.projects_dir / name).read_bytes()
    want = hashlib.sha256(b"project A.pkl" + b"project b.pkl").hexdigest().encode()
    assert (world.backups / ".last_sum").read_bytes() == want
    launch.stop()
    launch.wait(15)
    healthy_launch(impl, world)
    assert _backup_dirs(world) == dirs


def test_backups_keep_the_newest_ten(impl, world):
    _pkls(world, ["A.pkl"])
    for i in range(10):
        (world.backups / f"2000-01-01_0000{i:02d}").mkdir(parents=True)
    healthy_launch(impl, world)
    names = [p.name for p in _backup_dirs(world)]
    assert len(names) == 10 and "2000-01-01_000000" not in names


def test_a_failed_warmup_is_not_fatal(impl, world):
    healthy_launch(impl, world, env={"FOODOPT_FAKE_WARMUP_EXIT": "1"})
    assert _logged(world, "warm-up import did not finish - starting the server anyway")


def test_a_server_that_exits_before_healthy_exits_1(impl, world):
    launch = impl.start(world, env={"FOODOPT_FAKE_SERVER_EXIT": "3"})
    assert launch.wait(20) == 1
    assert "FATAL: server exited before becoming healthy" in world.log_text()
    for path in (world.port_file, world.status_file, world.lock_dir):
        assert not path.exists(), path


@posix_only
def test_sigterm_shuts_down_cleanly(impl, world):
    launch, _, pid = healthy_launch(impl, world)
    launch.stop()
    assert launch.wait(10) == 0
    assert "signal received - shut down" in world.log_text()
    assert not world.port_file.exists() and not world.lock_dir.exists()
    assert _dead(pid, 5)


@posix_only
def test_sigterm_during_warmup_exits_143(impl, world):
    launch = impl.start(world, env={"FOODOPT_FAKE_WARMUP_SECS": "30"})
    assert wait_for(lambda: _read(world.status_file) == LOAD and world.records("warmup"), 15)
    warm = world.records("warmup")[0]["pid"]
    launch.stop()
    assert launch.wait(10) == 143
    assert not world.status_file.exists() and not world.lock_dir.exists()
    assert _dead(warm, 5)


@posix_only
def test_late_cleanup_keeps_a_foreign_port_file(impl, world):
    launch, _, _ = healthy_launch(impl, world)
    world.port_file.write_bytes(b"1 1\n")
    launch.stop()
    launch.wait(10)
    assert world.port_file.read_bytes() == b"1 1\n"


@slow
def test_idle_shutdown(impl, world):
    launch, _, pid = healthy_launch(impl, world, env={"FOODOPT_IDLE_TIMEOUT_SECS": "10"})
    assert launch.wait(40) == 0
    log = world.log_text()
    assert "no window connected for 10s - shutting down" in log and "exiting" in log
    assert not world.port_file.exists()
    assert _dead(pid, 5)


@slow
def test_a_connected_window_keeps_it_running(impl, world):
    launch, port, _ = healthy_launch(impl, world, env={"FOODOPT_IDLE_TIMEOUT_SECS": "10"})
    held = socket.create_connection(("127.0.0.1", port))
    try:
        time.sleep(20)
        assert launch.proc.poll() is None
    finally:
        held.close()
    assert launch.wait(30) == 0


def test_paths_with_spaces_and_accents_survive_a_launch(impl, world):
    assert " " in str(world.projects_dir) and "é" in str(world.projects_dir)
    _pkls(world, ["Crème brûlée.pkl"])
    healthy_launch(impl, world)
    assert len(_backup_dirs(world)) == 1
    assert (_backup_dirs(world)[0] / "Crème brûlée.pkl").exists()
    assert world.log.exists() and "server healthy" in world.log_text()
    assert world.records("serve")[0]["cwd"] == str(world.projects_dir)
