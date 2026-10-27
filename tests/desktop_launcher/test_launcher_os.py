"""The launcher's OS layer: processes, connections and ports, without
psutil."""
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from launcher import connections, ports, procinfo

win32_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX only")


@pytest.fixture
def sleeper():
    procs = []

    def _start(code="import time; time.sleep(30)"):
        p = subprocess.Popen([sys.executable, "-c", code])
        procs.append(p)
        return p

    yield _start
    for p in procs:
        if p.poll() is None:
            p.kill()
        p.wait()


def _until(pred, timeout):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.05)
    return pred()


def test_pid_alive_for_self_and_a_finished_child():
    assert procinfo.pid_alive(os.getpid())
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    assert not procinfo.pid_alive(p.pid)


@win32_only
def test_pid_alive_never_signals_on_windows(sleeper):
    p = sleeper()
    for _ in range(3):
        assert procinfo.pid_alive(p.pid)
    time.sleep(0.5)
    assert p.poll() is None


def _same(a, b):
    return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))


def test_process_image_is_the_python_that_runs_it(sleeper):
    p = sleeper()
    assert _until(lambda: procinfo.process_image(p.pid) is not None, 5)
    assert _same(procinfo.process_image(p.pid), sys.executable)


def test_process_start_time_is_close_to_now(sleeper):
    t0 = time.time()
    p = sleeper()
    start = procinfo.process_start_time(p.pid)
    assert start is not None and t0 - 2 <= start <= t0 + 5


def test_is_our_server_checks_image_and_age(sleeper):
    p = sleeper()
    _until(lambda: procinfo.process_image(p.pid) is not None, 5)
    now = time.time()
    assert procinfo.is_our_server(p.pid, Path(sys.executable), now)
    assert not procinfo.is_our_server(p.pid, Path(sys.executable), now - 60)
    assert not procinfo.is_our_server(p.pid, Path("/bin/sh"), now)


def test_terminate_stops_a_process(sleeper):
    p = sleeper()
    procinfo.terminate(p.pid)
    assert _until(lambda: p.poll() is not None, 6)


@posix_only
def test_terminate_kills_one_that_ignores_sigterm(sleeper):
    p = sleeper("import signal, sys, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "sys.stdout.write('ready\\n'); sys.stdout.flush(); time.sleep(30)")
    time.sleep(0.5)
    t0 = time.monotonic()
    procinfo.terminate(p.pid, grace=1)
    assert _until(lambda: p.poll() is not None, 3)
    assert time.monotonic() - t0 < 3
    assert p.returncode == -signal.SIGKILL


def test_established_count_sees_a_loopback_connection():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen()
    port = srv.getsockname()[1]
    try:
        assert connections.established_count(port) == 0
        cli = socket.create_connection(("127.0.0.1", port))
        conn, _ = srv.accept()
        assert connections.established_count(port) >= 1
        cli.close()
        conn.close()
        assert _until(lambda: connections.established_count(port) == 0, 3)
    finally:
        srv.close()


def test_port_is_free():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen()
    port = s.getsockname()[1]
    assert not ports.port_is_free(port)
    s.close()
    assert ports.port_is_free(port)


def test_choose_port_with_a_base_skips_taken_ports():
    import random
    base = next(b for b in (random.randrange(30000, 39000, 10) for _ in range(50))
                if all(ports.port_is_free(b + i) for i in range(3)))
    s = socket.socket()
    s.bind(("127.0.0.1", base))
    s.listen()
    try:
        assert ports.choose_port(base) == base + 1
        assert ports.choose_port(base, exclude={base + 1}) == base + 2
    finally:
        s.close()


def test_choose_port_without_a_base_avoids_8501(monkeypatch):
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    free = s.getsockname()[1]
    s.close()
    offered = iter([8501, free])
    monkeypatch.setattr(ports, "_any_free_port", lambda: next(offered))
    assert ports.choose_port(None) == free


def test_a_connect_probe_that_times_out_still_counts_as_free(monkeypatch):
    """Windows answers a connect to a closed loopback port only after about
    2 s of retries, so the 0.2 s probe times out there; only a connect that
    is accepted means the port is taken."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()

    def slow_refusal(self, address):
        raise TimeoutError("timed out")

    monkeypatch.setattr(ports.socket.socket, "connect", slow_refusal)
    assert ports.port_is_free(port)


def _pid_of_another_user():
    """A live pid another account owns that is not python (so not a lock
    owner). Windows: SYSTEM processes, preferring one whose start time we
    cannot read (ACCESS_DENIED, which pid_alive counts as alive). POSIX:
    pid 1 (launchd / init)."""
    if sys.platform != "win32":
        return 1 if procinfo.pid_alive(1) else None
    out = subprocess.run(["tasklist", "/FI", "USERNAME eq SYSTEM", "/FO", "CSV", "/NH"],
                         capture_output=True, text=True).stdout
    found = []
    for line in out.splitlines():
        parts = line.strip().strip('"').split('","')
        if len(parts) > 1 and parts[1].isdigit() and int(parts[1]) > 4:
            pid = int(parts[1])
            image = procinfo.process_image(pid) or ""
            if procinfo.pid_alive(pid) and not image.replace("\\", "/").rsplit("/", 1)[-1].lower().startswith("python"):
                found.append(pid)
    unreadable = [p for p in found if procinfo.process_start_time(p) is None]
    return (unreadable or found or [None])[0]


def test_a_pid_owned_by_another_user_is_not_a_lock_owner(py, world):
    from tests.desktop_launcher.conftest import healthy_launch
    pid = _pid_of_another_user()
    if pid is None:
        if sys.platform == "win32":
            pytest.fail("tasklist found no SYSTEM-owned process")
        pytest.skip("pid 1 is not visible here")
    world.lock_dir.mkdir(parents=True)
    (world.lock_dir / "pid").write_bytes(f"{pid}\n".encode())
    healthy_launch(py, world)
    assert _until(lambda: not world.lock_dir.exists(), 5)
