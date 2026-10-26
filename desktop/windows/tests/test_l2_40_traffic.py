"""Level 2: zero traffic (spec Testing, Level 2; Goal 2). While the
installed shell opens, runs and closes, and while the packaged checks run,
every connection Food Optimizer's own processes make stays on loopback,
by a Windows Filtering Platform audit. A control proves the capture sees
our Python reach a non-loopback address. Admin."""
import collections
import datetime
import os
import re
import subprocess
import time

import pytest

import l2_support as l2

NAVIGATE = r"app\.navigate http://127\.0\.0\.1:\d+/[A-Za-z0-9_-]{32}/"


@pytest.fixture(scope="module")
def capture():
    with l2.wfp_audit() as started:
        yield started


def test_the_audit_subcategory_is_filtering_platform_connection():
    listing = subprocess.run(["auditpol", "/list", "/subcategory:*", "/v"], capture_output=True, text=True,
                             errors="replace").stdout
    line = next(line for line in listing.splitlines() if l2.WFP_CONNECTION.lower() in line.lower())
    assert "Filtering Platform Connection" in line


def test_the_capture_sees_our_python(per_user, capture):
    started = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=2)
    r = subprocess.run([str(per_user.root / "python" / "python.exe"), "-I", "-c",
                        "import socket; s=socket.socket(); s.settimeout(1); s.connect_ex(('192.0.2.1', 9))"],
                       capture_output=True, timeout=60, creationflags=l2.CREATE_NO_WINDOW)
    assert r.returncode == 0, r.stderr
    found = []
    for _ in range(30):
        found = [e for e in l2.wfp_events(started) if l2.ours(e, per_user.root) and e["DestAddress"] == "192.0.2.1"]
        if found:
            break
        time.sleep(1)
    assert found, "the WFP audit saw no connection from our python to 192.0.2.1"


def test_our_processes_connect_only_to_loopback(per_user, capture, l2_home, tmp_path):
    t0 = datetime.datetime.now(datetime.timezone.utc)
    for keep_open in (60, 0):
        after = l2.log_size(l2_home.state / "shell.log")
        job = l2.start_shell(per_user.root, l2_home)
        try:
            assert l2.wait_log(l2_home.state / "shell.log", NAVIGATE, l2.FIRST_OPEN_LIMIT_S, after=after)
            time.sleep(keep_open)
        finally:
            for hwnd in l2.top_windows(job.pid):
                l2.close_window(hwnd)
            if not l2.wait_for(lambda: not job.alive(), 10):
                l2.kill_under(per_user.root)
            l2.wait_for(lambda: not l2.processes_under(per_user.root / "python"), 10)
        if keep_open:
            env = dict(os.environ)
            env["HOME"] = env["USERPROFILE"] = str(tmp_path)
            checks = subprocess.run(
                [str(per_user.root / "python" / "python.exe"), "-I", "-X", "utf8", "-B",
                 str(l2.REPO / "desktop" / "shared" / "packaged_app_checks.py"), "--resources",
                 str(per_user.root / "app"), "--projects", str(tmp_path / "FoodOptimizer")],
                env=env, capture_output=True, text=True, errors="replace", timeout=1800,
                creationflags=l2.CREATE_NO_WINDOW)
            assert checks.returncode == 0, checks.stdout[-3000:]
    time.sleep(5)
    events = [e for e in l2.wfp_events(t0) if l2.ours(e, per_user.root)]
    counts = collections.Counter(os.path.basename(e["Application"]) for e in events)
    l2.report("wfp_events_by_process", dict(counts))
    outside = [e for e in events
               if not ((e["DestAddress"] in l2.LOOPBACK | l2.LISTEN_ANY)
                       and (e["SourceAddress"] in l2.LOOPBACK | l2.LISTEN_ANY))]
    l2.report("wfp_outside", len(outside))
    assert events, "no events from our processes: the capture saw nothing"
    assert outside == [], outside[:10]
