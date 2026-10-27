"""Level 2: the installed shell, as the limited user on the desktop (spec
Testing, Level 2: shell smoke, stale-file freshness, warm relaunch, the
unsupported path). Each test has its own state and projects folders."""
import http.server
import os
import re
import socket
import statistics
import threading
import time

import l2_support as l2

NAVIGATE = r"app\.navigate http://127\.0\.0\.1:(\d+)/([A-Za-z0-9_-]{32})/"


def _close(job, timeout=5) -> bool:
    windows = l2.top_windows(job.pid)
    for hwnd in windows:
        l2.close_window(hwnd)
    return l2.wait_for(lambda: not job.alive(), timeout)


def _no_python(root, timeout=5) -> bool:
    return l2.wait_for(lambda: not l2.processes_under(root / "python"), timeout)


def _raw_get(port, path):
    with socket.create_connection(("127.0.0.1", port), timeout=10) as c:
        c.sendall(f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n".encode())
        data = b""
        while True:
            chunk = c.recv(65536)
            if not chunk:
                break
            data += chunk
    return int(data.split(b" ", 2)[1])


def _open(per_user, home, after=0, timeout=l2.FIRST_OPEN_LIMIT_S):
    started = time.monotonic()
    job = l2.start_shell(per_user.root, home)
    m = l2.wait_log(home.state / "shell.log", NAVIGATE, timeout, after=after)
    return job, m, time.monotonic() - started


def test_shell_smoke_no_console_health_close_no_python(per_user, l2_home):
    watch = l2.ConsoleWatch().start()
    job, m, seconds = _open(per_user, l2_home)
    try:
        assert m, (l2_home.state / "shell.log").read_text(encoding="utf-8", errors="replace")[-3000:]
        port, base = int(m.group(1)), m.group(2)
        assert port != 8501
        assert (l2_home.state / "server.port").read_text(encoding="ascii").split()[0] == str(port)
        assert _raw_get(port, f"/{base}/_stcore/health") == 200
        l2.report("first_open_s", round(seconds, 1))
        assert l2.wait_for(lambda: l2.top_windows(job.pid), 10)
        assert _close(job), "the shell did not exit within 5 s of WM_CLOSE"
        assert _no_python(per_user.root), l2.processes_under(per_user.root / "python")
    finally:
        consoles = watch.stop()
        if job.alive():
            l2.kill_under(per_user.root)
    assert consoles == []


def test_warm_relaunch_is_quick(per_user, l2_home):
    limit = l2.WARM_RELAUNCH_LIMIT_S["arm64" if l2.host_is_arm64() else "x64"]
    job, m, _seconds = _open(per_user, l2_home)
    assert m
    assert _close(job) and _no_python(per_user.root)
    after = l2.log_size(l2_home.state / "shell.log")
    job, m, seconds = _open(per_user, l2_home, after=after)
    try:
        assert m
        l2.report("warm_relaunch_runs", round(seconds, 1))
        assert limit is not None, f"measure first: warm relaunch took {seconds:.1f} s"
        assert seconds <= limit, f"warm relaunch took {seconds:.1f} s; limit {limit} s"
    finally:
        _close(job)
        _no_python(per_user.root)
        if job.alive():
            l2.kill_under(per_user.root)


class _Anything(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args):
        pass


def test_stale_server_files_are_not_loaded(per_user, l2_home):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Anything)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    dummy = server.server_address[1]
    try:
        l2_home.state.mkdir(parents=True, exist_ok=True)
        stale = "stalebase0000000000000000000000000"
        (l2_home.state / "server.port").write_bytes(f"{dummy} 4\n".encode("ascii"))
        (l2_home.state / "server.base").write_bytes(f"{stale}\n".encode("ascii"))
        old = time.time() - 3600
        for name in ("server.port", "server.base"):
            os.utime(l2_home.state / name, (old, old))
        job, m, _seconds = _open(per_user, l2_home)
        try:
            assert m, (l2_home.state / "shell.log").read_text(encoding="utf-8", errors="replace")[-3000:]
            assert int(m.group(1)) != dummy and m.group(2) != stale
        finally:
            _close(job)
            _no_python(per_user.root)
            if job.alive():
                l2.kill_under(per_user.root)
    finally:
        server.shutdown()


def test_unsupported_machine_shows_the_unsupported_page(per_user, l2_home):
    job = l2.start_shell(per_user.root, l2_home, env={"FOODOPT_TEST_ARCH": "x86", "FOODOPT_TEST_HOOKS": "1"})
    try:
        log = l2_home.state / "shell.log"
        assert l2.wait_log(log, r"launcher\.exit 2", 30), log.read_text(encoding="utf-8", errors="replace")
        assert l2.wait_log(log, r"page\.error unsupported", 30)
        assert l2.wait_for(lambda: l2.top_windows(job.pid), 10)
        assert _close(job)
        assert _no_python(per_user.root)
    finally:
        if job.alive():
            l2.kill_under(per_user.root)
