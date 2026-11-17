"""The Python launcher's pieces, one at a time."""
import http.server
import re
import sys
import threading
import time
from pathlib import Path

import pytest

from launcher import childenv, cli


def _opts(environ=None, *extra):
    return cli.parse_args(["--state-dir", "s", "--projects-dir", "p", "--app", "a.py", *extra],
                          environ or {})


def test_parse_args_defaults():
    o = _opts()
    assert (o.warmup_timeout, o.health_timeout, o.idle_timeout) == (120, 180, 900)
    assert o.base_path == "auto" and o.port_base is None and o.test_arch is None
    assert o.python == Path(sys.executable)
    assert o.child_script.name == "child.py" and o.child_script.parent.name == "launcher"
    assert not o.stdin_lifeline and not o.lock_held
    assert _opts({"FOODOPT_IDLE_TIMEOUT_SECS": "10"}).idle_timeout == 10
    o = _opts({"FOODOPT_PORT_BASE": "30000", "FOODOPT_TEST_ARCH": "arm64",
               "FOODOPT_TEST_HOOKS": "1"},
              "--stdin-lifeline", "--lock-held", "--base-path", "none",
              "--warmup-timeout", "2", "--health-timeout", "3")
    assert (o.port_base, o.test_arch, o.base_path) == (30000, "arm64", "none")
    assert o.stdin_lifeline and o.lock_held and (o.warmup_timeout, o.health_timeout) == (2, 3)


class _Win:
    def __init__(self, build):
        self.build = build


def test_preflight(monkeypatch):
    monkeypatch.setattr(cli.platform, "mac_ver", lambda: ("13.0", ("", "", ""), ""))
    assert cli.preflight("darwin", "arm64") is None
    assert "unsupported machine" in cli.preflight("darwin", "x86_64")
    monkeypatch.setattr(cli.platform, "mac_ver", lambda: ("12.7", ("", "", ""), ""))
    assert "unsupported machine" in cli.preflight("darwin", "arm64")
    monkeypatch.setattr(cli.sys, "getwindowsversion", lambda: _Win(19045), raising=False)
    assert cli.preflight("win32", "AMD64") is None
    assert cli.preflight("win32", None) is None or struct_is_32_bit()
    monkeypatch.setattr(cli.sys, "getwindowsversion", lambda: _Win(19044), raising=False)
    assert "unsupported machine" in cli.preflight("win32", "AMD64")
    monkeypatch.setattr(cli.sys, "getwindowsversion", lambda: _Win(26100), raising=False)
    assert "unsupported machine" in cli.preflight("win32", "x86_64")
    assert cli.preflight("linux", None) == "unsupported machine: platform=linux"


def struct_is_32_bit():
    import struct
    return struct.calcsize("P") != 8


def test_child_env(tmp_path):
    environ = {"PYTHONPATH": "x", "PYTHONHOME": "y", "STREAMLIT_THEME_BASE": "dark",
               "HTTPS_PROXY": "p", "http_proxy": "p", "No_Proxy": "n", "PATH": "/bin",
               "FOODOPT_FAKE_RECORD": "r"}
    assert childenv.child_env(environ, tmp_path) == {
        "PATH": "/bin", "FOODOPT_FAKE_RECORD": "r", "GIT_PYTHON_REFRESH": "quiet",
        "PYTHONUSERBASE": str(tmp_path / "no-user-base")}
    assert not (tmp_path / "no-user-base").exists()


def test_child_env_carries_the_base_path_only_in_the_environment(tmp_path):
    environ = {"STREAMLIT_SERVER_BASE_URL_PATH": "evil", "PATH": "/bin"}
    assert childenv.child_env(environ, tmp_path, base="abc")[
        "STREAMLIT_SERVER_BASE_URL_PATH"] == "abc"
    assert "STREAMLIT_SERVER_BASE_URL_PATH" not in childenv.child_env(environ, tmp_path)
    assert "STREAMLIT_SERVER_BASE_URL_PATH" not in childenv.child_env(
        environ, tmp_path, base=None)


def test_test_arch_needs_the_hooks_opt_in():
    assert _opts({"FOODOPT_TEST_ARCH": "x86"}).test_arch is None
    assert _opts({"FOODOPT_TEST_ARCH": "x86", "FOODOPT_TEST_HOOKS": "1"}).test_arch == "x86"
    assert _opts({"FOODOPT_TEST_ARCH": "x86", "FOODOPT_TEST_HOOKS": "0"}).test_arch is None


def test_streamlit_flags_are_the_spec_list():
    assert childenv.streamlit_flags(18501, "abc") == [
        "--server.headless=true",
        "--server.address=127.0.0.1",
        "--server.port=18501",
        "--logger.hideWelcomeMessage=true",
        "--browser.gatherUsageStats=false",
        "--client.toolbarMode=minimal",
        "--client.showErrorDetails=none",
        "--server.fileWatcherType=none",
        "--server.enableXsrfProtection=true",
        "--global.developmentMode=false",
        "--server.runOnSave=false",
        "--server.enableCORS=true",
        "--server.enableStaticServing=false",
        "--server.useStarlette=false",
        "--server.enableWebsocketCompression=false",
        "--server.maxUploadSize=200",
        "--browser.serverAddress=127.0.0.1",
        "--browser.serverPort=18501",
        "--runner.magicEnabled=true",
    ]
    assert not any("baseUrlPath" in f for f in childenv.streamlit_flags(18501, None))


def test_child_command(tmp_path):
    py, child = Path("/py"), Path("/c.py")
    assert childenv.child_command(py, child, "serve", "app.py", "--x") == [
        str(py), "-I", "-X", "utf8", "-B", str(child), "serve", "app.py", "--x"]


def test_new_base_path_is_unguessable():
    bases = {childenv.new_base_path() for _ in range(100)}
    assert len(bases) == 100
    assert all(re.fullmatch(r"[A-Za-z0-9_-]{32}", b) for b in bases)


def test_popen_kwargs_hide_windows_consoles():
    assert childenv.popen_kwargs("win32")["creationflags"] & 0x08000000
    assert childenv.popen_kwargs("darwin") == {}


def _counter(values):
    calls = []

    def count():
        calls.append(1)
        return values[len(calls) - 1] if len(calls) <= len(values) else 0
    return count, calls


def test_watch_idle_counts_ticks_not_time(monkeypatch):
    count, calls = _counter([0, 0, 0])
    assert cli.watch_idle(lambda: True, count, 10, lambda t: False) == "idle"
    assert len(calls) == 2
    clock = [0.0]

    def jumpy():
        clock[0] += 1e6
        return clock[0]

    monkeypatch.setattr(time, "time", jumpy)
    monkeypatch.setattr(time, "monotonic", jumpy)
    count, calls = _counter([0, 0, 0])
    assert cli.watch_idle(lambda: True, count, 10, lambda t: False) == "idle"
    assert len(calls) == 2


def test_watch_idle_resets_on_a_connection():
    count, calls = _counter([0, 1, 0, 0])
    assert cli.watch_idle(lambda: True, count, 10, lambda t: False) == "idle"
    assert len(calls) == 4


def test_watch_idle_ends_when_the_server_does():
    alive = iter([True, True, False])
    count, _ = _counter([1, 1, 1])
    assert cli.watch_idle(lambda: next(alive), count, 10, lambda t: False) == "server_exited"


def test_watch_idle_stops_on_request():
    count, _ = _counter([1, 1, 1])
    assert cli.watch_idle(lambda: True, count, 10, lambda t: True) == "stopped"


class _Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/b/_stcore/health" else 404)
        self.end_headers()

    def log_message(self, *a):
        pass


def test_healthy_against_a_local_server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Health)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        port = srv.server_address[1]
        assert cli.healthy(port, "b") is True
        assert cli.healthy(port, None) is False
    finally:
        srv.shutdown()
        srv.server_close()
    assert cli.healthy(port, "b", timeout=0.5) is False


def test_a_lock_that_survives_release_is_logged(tmp_path, monkeypatch):
    import io
    import os
    from launcher import files
    stream = io.StringIO()
    opts = cli.Options(state_dir=tmp_path / "s", projects_dir=tmp_path / "p",
                       app=tmp_path / "a.py")
    run = cli._Run(opts, {}, files.Log(stream))
    opts.state_dir.mkdir()
    assert files.acquire_lock(run.lock, os.getpid(), alive=lambda p: False)[0]
    monkeypatch.setattr(files.shutil, "rmtree", lambda *a, **k: None)
    run.release()
    assert "could not remove launch.lock" in stream.getvalue()


def test_a_lock_owner_we_cannot_inspect_is_stale(tmp_path, monkeypatch):
    import io
    from launcher import files, procinfo
    opts = cli.Options(state_dir=tmp_path / "s", projects_dir=tmp_path / "p",
                       app=tmp_path / "a.py")
    run = cli._Run(opts, {}, files.Log(io.StringIO()))
    run.lock.mkdir(parents=True)
    (run.lock / "pid").write_bytes(b"4242\n")
    monkeypatch.setattr(procinfo, "pid_alive", lambda pid: True)

    def owner(start, image):
        monkeypatch.setattr(procinfo, "process_start_time", lambda pid: start)
        monkeypatch.setattr(procinfo, "process_image", lambda pid: image)
        return run.lock_owner_alive(4242)

    old = (run.lock / "pid").stat().st_mtime - 60
    assert owner(None, r"C:\Python312\python.exe") is False
    assert owner(old, None) is False
    assert owner(old, r"C:\Windows\System32\svchost.exe") is False
    assert owner(old, r"C:\Python312\python.exe") is True
    assert owner(old, "/usr/bin/Python3") is True
