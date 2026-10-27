"""Level 2: zero traffic (spec Testing, Level 2; Goal 2; I4). While the
installed shell opens, runs and closes, and while the packaged checks run,
every connection Food Optimizer's processes make stays on loopback, by a
Windows Filtering Platform audit. "Ours" is the shell's whole process tree
(WebView2's msedgewebview2.exe processes included) plus anything run from
the install folder. A control proves the capture sees our Python reach a
non-loopback address. Admin."""
import collections
import datetime
import os
import re
import subprocess
import time

import pytest

import l2_support as l2
from conftest import L2Home

NAVIGATE = l2.NAVIGATE
WEBVIEW2 = "msedgewebview2.exe"
DIRECTION = {"%%14592": "in", "%%14593": "out"}      # WFP's inbound / outbound


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


@pytest.fixture(scope="module")
def session(per_user, capture, tmp_path_factory):
    """Two sessions of the installed shell (open 60 s, then closed at
    once), the packaged checks between them, and every WFP event since:
    (events, {pid: {"image", "cmd", "created"}} of the shell's tree)."""
    home = L2Home("w40").make()
    tmp_path = tmp_path_factory.mktemp("traffic")
    t0 = datetime.datetime.now(datetime.timezone.utc)
    tree = {}
    for keep_open in (60, 0):
        after = l2.log_size(home.state / "shell.log")
        job = l2.start_shell(per_user.root, home)
        watch = l2.TreeWatch(job.pid).start()
        try:
            assert l2.wait_log(home.state / "shell.log", NAVIGATE, l2.FIRST_OPEN_LIMIT_S, after=after)
            time.sleep(keep_open)
        finally:
            for hwnd in l2.top_windows(job.pid):
                l2.close_window(hwnd)
            if not l2.wait_for(lambda: not job.alive(), 10):
                l2.kill_under(per_user.root)
            l2.wait_for(lambda: not l2.processes_under(per_user.root / "python"), 10)
            l2.wait_for(lambda: not watch.alive(), 20, 0.5)
            tree.update(watch.stop())
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
    events = [e for e in l2.wfp_events(t0) if l2.ours(e, per_user.root, tree)]
    return events, tree


def _image(path: str) -> str:
    return os.path.basename(path).lower()


def _browser_arguments() -> list:
    source = (l2.REPO / "desktop" / "windows" / "FoodOptimizer" / "FoodOptimizer.Shell"
              / "WebEnvironment.cs").read_text(encoding="utf-8")
    return re.search(r'BrowserArguments\s*=\s*"([^"]*)"', source).group(1).split()


def test_the_tree_includes_webview2(session):
    """The audit counts WebView2: its processes are in the shell's tree, and
    their own connections (to the app on 127.0.0.1) are among the events."""
    events, tree = session
    assert any(_image(p["image"]) == WEBVIEW2 for p in tree.values()), tree
    assert any(_image(e["Application"]) == WEBVIEW2 for e in events), collections.Counter(
        _image(e["Application"]) for e in events)


def test_webview2_runs_with_our_browser_arguments(session):
    """WebEnvironment.BrowserArguments reach WebView2's browser process (the
    msedgewebview2.exe without --type=) on its command line."""
    _events, tree = session
    browsers = [p["cmd"] for p in tree.values() if _image(p["image"]) == WEBVIEW2 and "--type=" not in p["cmd"]]
    assert browsers, tree
    l2.report("webview2_browser_command_line", browsers[0])
    for flag in _browser_arguments():
        assert all(flag in cmd.split() for cmd in browsers), (flag, browsers)


def test_our_processes_connect_only_to_loopback(session):
    events, tree = session
    counts = collections.Counter(_image(e["Application"]) for e in events)
    l2.report("wfp_events_by_process", dict(counts))
    l2.report("tree_processes", dict(collections.Counter(_image(p["image"]) for p in tree.values())))
    outside = [e for e in events
               if not ((e["DestAddress"] in l2.LOOPBACK | l2.LISTEN_ANY)
                       and (e["SourceAddress"] in l2.LOOPBACK | l2.LISTEN_ANY))]
    destinations = collections.defaultdict(collections.Counter)
    for e in outside:
        way = DIRECTION.get(e["Direction"], e["Direction"])
        destinations[_image(e["Application"])][f'{way} {e["DestAddress"]}:{e["DestPort"]}'] += 1
    l2.report("wfp_outside_by_process", {k: dict(v) for k, v in destinations.items()})
    l2.report("wfp_outside", len(outside))
    assert events, "no events from our processes: the capture saw nothing"
    assert outside == [], outside[:10]
