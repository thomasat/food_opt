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
from tests.desktop_launcher.harness import end_world, make_world, wait_for

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


# ---------- what only the Python launcher does ----------

win32_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows only")


@pytest.fixture
def foreign():
    procs = []

    def _start(cmd):
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        procs.append(p)
        return p

    yield _start
    for p in procs:
        if p.poll() is None:
            p.kill()
        p.wait()


def _sleeper_cmd():
    if sys.platform == "win32":
        return [r"C:\Windows\System32\ping.exe", "-n", "60", "127.0.0.1"]
    return ["/bin/sleep", "60"]


def test_an_orphan_that_is_not_our_server_is_left_running(py, world, foreign):
    other = foreign(_sleeper_cmd())
    world.state_dir.mkdir(parents=True)
    world.port_file.write_bytes(f"18999 {other.pid}\n".encode())
    healthy_launch(py, world)
    assert other.poll() is None and procinfo.pid_alive(other.pid)
    assert "which is not our server - left running" in world.log_text()


def test_an_orphan_started_after_the_port_file_is_left_running(py, world, foreign):
    world.state_dir.mkdir(parents=True)
    world.port_file.write_bytes(b"18999 1\n")
    old = time.time() - 60
    os.utime(world.port_file, (old, old))
    reused = foreign([sys.executable, "-c", "import time; time.sleep(60)"])
    world.port_file.write_bytes(f"18999 {reused.pid}\n".encode())
    os.utime(world.port_file, (old, old))
    healthy_launch(py, world)
    assert reused.poll() is None
    assert "which is not our server - left running" in world.log_text()


def test_without_a_base_the_port_is_random_and_not_8501(py, world):
    _, port, _ = healthy_launch(py, world, env={"FOODOPT_PORT_BASE": None})
    assert port != 8501 and port >= 1024


def test_streamlit_port_error_moves_to_another_port(py, world):
    flag = world.root / "taken-once"
    healthy_launch(py, world, env={"FOODOPT_FAKE_PORT_TAKEN_ONCE": str(flag)})
    ports_tried = [next(a for a in r["argv"] if a.startswith("--server.port="))
                   for r in world.records("serve")]
    assert len(ports_tried) == 2 and ports_tried[0] != ports_tried[1]
    assert "was taken - trying another port" in world.log_text()


def test_the_base_path_file_and_the_server_agree(py, world):
    # With the lifeline, stop() is a clean shutdown on Windows too.
    launch = py.start(world, args=["--stdin-lifeline"], stdin=subprocess.PIPE)
    torn = []
    end = time.monotonic() + 20
    while time.monotonic() < end and world.health() != 200:
        if world.port_file.exists() and not world.base_file.exists():
            torn.append(time.monotonic())
        time.sleep(0.02)
    assert world.health() == 200
    assert torn == []
    assert re.match(rb"^[A-Za-z0-9_-]{32}\n$", world.base_file.read_bytes())
    base = world.base()
    assert f"--server.baseUrlPath={base}" in world.records("serve")[0]["argv"]
    assert world.health(f"/{base}/_stcore/health") == 200
    assert world.health("/_stcore/health") == 404
    launch.stop()
    launch.wait(15)
    assert not world.port_file.exists() and not world.base_file.exists()


def test_base_path_none_writes_no_base_file(py, world):
    healthy_launch(py, world, args=["--base-path", "none"])
    assert not world.base_file.exists()
    assert not any("baseUrlPath" in a for a in world.records("serve")[0]["argv"])


def _children(world):
    recs = world.records()
    assert {r["kind"] for r in recs} == {"warmup", "serve"}, recs
    return recs


def test_children_run_isolated(py, world):
    healthy_launch(py, world)
    for r in _children(world):
        assert r["flags"] == {"isolated": 1, "utf8_mode": 1, "dont_write_bytecode": 1}, r


def test_children_get_a_clean_environment(py, world):
    healthy_launch(py, world, env={"PYTHONPATH": str(world.root), "PYTHONSTARTUP": "x.py",
                                   "STREAMLIT_SERVER_PORT": "1", "HTTPS_PROXY": "http://p:1",
                                   "http_proxy": "http://p:1", "ALL_PROXY": "http://p:1"})
    for r in _children(world):
        assert set(r["env"]) == {"PYTHONUSERBASE", "GIT_PYTHON_REFRESH"}, r["env"]
        assert r["env"]["GIT_PYTHON_REFRESH"] == "quiet"
        assert r["userbase"] == str(world.state_dir / "no-user-base")
    assert not (world.state_dir / "no-user-base").exists()


def test_projects_folder_is_not_on_the_child_path(py, world):
    world.projects_dir.mkdir(parents=True)
    (world.projects_dir / "pandas.py").write_text("raise SystemExit('loaded')\n", encoding="utf-8")
    healthy_launch(py, world)
    serve = world.records("serve")[0]
    assert serve["cwd"] == str(world.projects_dir)
    for entry in ("", ".", str(world.projects_dir)):
        assert entry not in serve["sys_path"], entry


def test_streamlit_gets_the_full_flag_set(py, world):
    from launcher import childenv
    _, port, _ = healthy_launch(py, world)
    argv = world.records("serve")[0]["argv"]
    app = str(py.app(world))
    assert argv[argv.index(app) + 1:] == childenv.streamlit_flags(port, world.base())


@win32_only
def test_children_have_no_console_window(py, world):
    healthy_launch(py, world)
    for r in _children(world):
        assert r["console_window"] == 0, r


def test_stdin_eof_shuts_down(py, world):
    launch, _, pid = healthy_launch(py, world, args=["--stdin-lifeline"], stdin=subprocess.PIPE)
    launch.proc.stdin.close()
    assert launch.wait(10) == 0
    assert "the window closed - shut down" in world.log_text()
    assert _dead(pid, 5)
    assert not world.port_file.exists()


def test_stdin_eof_during_warmup(py, world):
    launch = py.start(world, env={"FOODOPT_FAKE_WARMUP_SECS": "30"}, args=["--stdin-lifeline"],
                      stdin=subprocess.PIPE)
    assert wait_for(lambda: _read(world.status_file) == LOAD and world.records("warmup"), 15)
    warm = world.records("warmup")[0]["pid"]
    launch.proc.stdin.close()
    assert launch.wait(10) == 0
    assert _dead(warm, 5)
    assert not world.status_file.exists()


@slow
def test_the_lifeline_turns_off_the_idle_watchdog(py, world):
    launch, _, _ = healthy_launch(py, world, env={"FOODOPT_IDLE_TIMEOUT_SECS": "5"},
                                  args=["--stdin-lifeline"], stdin=subprocess.PIPE)
    time.sleep(15)
    assert launch.proc.poll() is None and world.health() == 200


def test_stdin_at_eof_without_lifeline_keeps_running(py, world):
    launch, _, _ = healthy_launch(py, world, stdin=subprocess.DEVNULL)
    time.sleep(3)
    assert launch.proc.poll() is None and world.health() == 200


@posix_only
def test_lock_held_mode(py, world):
    from tests.desktop_launcher.harness import BOOT, _env, _popen
    world.state_dir.mkdir(parents=True)
    world.log.write_bytes(b"old\n")
    cmd = py.command(world, ["--lock-held"])
    script = 'mkdir "$1" && echo $$ > "$1/pid" && exec "$2" -I -X utf8 -B -c "$3" "${@:4}"'
    assert cmd[1:7] == ["-I", "-X", "utf8", "-B", "-c", BOOT]
    launch = _popen(world, ["/bin/bash", "-c", script, "_", str(world.lock_dir), cmd[0],
                            BOOT, *cmd[7:]], _env(world, None), subprocess.DEVNULL)
    if not wait_for(lambda: "server healthy" in world.log_text() and world.health() == 200, 20):
        pytest.fail(launch.output())
    assert _gone(world.lock_dir)
    assert not (world.state_dir / "launcher.log.1").exists()
    assert world.log.read_bytes().startswith(b"old\n")


def test_lock_held_without_the_lock_exits_1(py, world):
    launch = py.start(world, args=["--lock-held"])
    assert launch.wait(15) == 1
    assert "--lock-held but launch.lock is not ours" in world.log_text()


def test_warmup_timeout_is_an_argument(py, world):
    healthy_launch(py, world, env={"FOODOPT_FAKE_WARMUP_SECS": "60"},
                   args=["--warmup-timeout", "2"])
    assert "warm-up import still running after 2s - giving up on it" in world.log_text()
    assert _dead(world.records("warmup")[0]["pid"], 5)


def test_health_timeout_is_an_argument(py, world):
    launch = py.start(world, env={"FOODOPT_FAKE_HEALTH_DELAY_SECS": "60"},
                      args=["--health-timeout", "3"])
    assert launch.wait(20) == 1
    assert "FATAL: server did not become healthy within 3s" in world.log_text()
    assert _dead(world.records("serve")[0]["pid"], 5)


def test_a_status_file_that_cannot_be_written_is_not_fatal(py, world):
    world.status_file.mkdir(parents=True)
    healthy_launch(py, world)
    assert "could not write status.txt" in world.log_text()
    assert world.log_text().count("could not write status.txt") == 1


def test_a_log_in_use_is_appended_not_rotated(py, world):
    world.state_dir.mkdir(parents=True)
    world.log.write_bytes(b"old line\n")
    held = None
    if sys.platform == "win32":
        held = open(world.log, "ab")
    else:
        (world.state_dir / "launcher.log.1").mkdir()
        (world.state_dir / "launcher.log.1" / "x").write_bytes(b"x")
    try:
        healthy_launch(py, world)
    finally:
        if held is not None:
            held.close()
    text = world.log_text()
    assert text.startswith("old line\n") and "server healthy" in text
    assert "could not rotate launcher.log" in text


def test_a_chatty_server_never_overwrites_the_launchers_lines(py, world):
    """The children write to launcher.log through their own handle. Each
    write must land at the end of the file (as launcher.sh's tee -a does),
    never over the launcher's own lines."""
    launch, port, pid = healthy_launch(py, world, env={"FOODOPT_FAKE_CHATTER": "1"},
                                       args=["--stdin-lifeline"], stdin=subprocess.PIPE)
    time.sleep(1)
    launch.proc.stdin.close()
    assert launch.wait(15) == 0
    lines = world.log_text().splitlines()
    assert any(line.endswith(f"server healthy on port {port} (pid {pid})") for line in lines), lines
    assert any(line.endswith("the window closed - shut down") for line in lines), lines
    chatter = [int(line.rsplit(" ", 1)[1]) for line in lines if line.startswith("fake server chatter")]
    assert len(chatter) > 5 and chatter == sorted(chatter), chatter
    for line in lines:
        assert re.match(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d |fake server chatter \d+$)", line), line


def test_an_unexpected_error_is_logged_and_releases_the_lock(py, world):
    """Whatever goes wrong, the log the user attaches says so."""
    launch = py.start(world, env={"FOODOPT_PORT_BASE": "65536"})
    assert launch.wait(15) == 1
    log = world.log_text()
    assert "FATAL: unexpected error" in log and "Traceback" in log, log
    assert not world.lock_dir.exists() and not world.status_file.exists()


def test_a_stale_lock_whose_pid_was_reused_is_taken(py, world, foreign):
    """A launch that died holding the lock leaves its pid behind; on Windows
    another program soon has that pid. A process younger than the lock's pid
    file is not its owner."""
    world.lock_dir.mkdir(parents=True)
    pid_file = world.lock_dir / "pid"
    reused = foreign([sys.executable, "-c", "import time; time.sleep(60)"])
    pid_file.write_bytes(f"{reused.pid}\n".encode())
    old = time.time() - 60
    os.utime(pid_file, (old, old))
    healthy_launch(py, world)
    assert reused.poll() is None
    assert _gone(world.lock_dir)


def test_an_early_exit_with_the_lifeline_exits_cleanly(py, world):
    """The lifeline's stdin reader must not crash the interpreter's shutdown
    (Windows: "_enter_buffered_busy … daemon threads", 0xC0000005) when the
    launcher exits while the shell still holds stdin open."""
    world.lock_dir.mkdir(parents=True)
    (world.lock_dir / "pid").write_bytes(f"{os.getpid()}\n".encode())
    launch = py.start(world, args=["--stdin-lifeline"], stdin=subprocess.PIPE)
    assert launch.wait(15) == 0, launch.output()
    assert (f"another launch is already in progress (pid {os.getpid()}) - exiting"
            in launch.output())
    assert "Fatal Python error" not in launch.output(), launch.output()


def test_a_bad_argument_exits_1_not_2(py, world):
    launch = py.start(world, args=["--base-path", "sometimes"])
    assert launch.wait(30) == 1
    assert "error:" in launch.output()


def test_stdin_eof_while_waiting_for_the_lock_never_starts_the_warmup(py, world):
    world.lock_dir.mkdir(parents=True)          # no pid file: a 5 s grace wait
    launch = py.start(world, args=["--stdin-lifeline"], stdin=subprocess.PIPE)
    time.sleep(1)
    launch.proc.stdin.close()
    assert launch.wait(30) == 0
    assert world.records("warmup") == []
    assert "the window closed - shut down" in world.log_text()
    assert not world.lock_dir.exists()


def test_a_port_file_that_cannot_be_written_leaves_no_base_file(py, world):
    world.port_file.mkdir(parents=True)          # os.replace onto a folder fails
    launch = py.start(world)
    assert launch.wait(60) == 1
    assert "FATAL: could not write server.port" in world.log_text()
    assert not world.base_file.exists()


def test_world_teardown_ends_a_server_nothing_recorded(py, tmp_path):
    (tmp_path / "w").mkdir()
    w = make_world(tmp_path / "w")
    _, _, pid = healthy_launch(py, w, env={"FOODOPT_FAKE_RECORD": None})
    end_world(w)
    assert wait_for(lambda: not procinfo.pid_alive(pid), 10)


def test_a_projects_folder_that_cannot_be_made_is_logged(py, world, tmp_path):
    blocker = tmp_path / "a-regular-file"
    blocker.write_bytes(b"x")
    projects = blocker / "FoodOptimizer"
    world.projects_dir = projects
    launch = py.start(world)
    assert launch.wait(15) == 1
    log = world.log_text()
    assert "FATAL" in log and str(projects) in log, log
    assert not world.lock_dir.exists()
