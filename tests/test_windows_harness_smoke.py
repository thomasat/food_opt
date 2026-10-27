"""Smoke checks on the Windows test harness: its scheduled tasks never fire
on schedule and are always deleted, and the Level 2 report names the
installer it tested and counts connections outside loopback. l2_support runs
on any OS with subprocess.run faked; the .ps1 and traffic files are read as
text."""
import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
WINDOWS = REPO / "desktop" / "windows"
LEVEL2 = WINDOWS / "tests" / "run_level2.ps1"
UI = WINDOWS / "FoodOptimizer" / "uitests" / "run_ui_tests.ps1"
TRAFFIC = WINDOWS / "tests" / "test_l2_40_traffic.py"


@pytest.fixture
def l2(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("l2_support_smoke", WINDOWS / "tests" / "l2_support.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "JOBS", tmp_path / "jobs")
    module.calls = calls = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        if args[:2] == ["schtasks", "/Run"]:   # the job runner reports its pid
            for f in (tmp_path / "jobs").glob("*.json"):
                if not f.name.endswith(".result.json"):
                    spec_ = json.loads(f.read_text(encoding="utf-8"))
                    pathlib.Path(spec_["result"]).write_text(json.dumps({"pid": 1, "elevated": False}),
                                                             encoding="utf-8")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    yield module
    module.end_all_jobs()
    sys.modules.pop(spec.name, None)


def _verbs(l2, verb):
    return [c for c in l2.calls if c[:2] == ["schtasks", verb]]


def test_harness_tasks_never_fire_on_schedule_and_are_deleted(l2):
    assert l2.NEVER_ON_SCHEDULE == ["/SC", "ONCE", "/SD", "01/01/2020", "/ST", "00:00"]
    job = l2.run_as("system", ["x"], wait=False)
    (create,) = _verbs(l2, "/Create")
    i = create.index("/SC")
    assert create[i:i + len(l2.NEVER_ON_SCHEDULE)] == l2.NEVER_ON_SCHEDULE
    assert not _verbs(l2, "/Delete")
    l2.end_all_jobs()                       # an unwaited job ends with /Delete
    (delete,) = _verbs(l2, "/Delete")
    assert delete[-1] == job.task

    l2.calls.clear()
    job = l2.run_as("user", ["x"], wait=False)
    task = job.task
    with pytest.raises(TimeoutError):       # a timed-out wait ends with /Delete too
        job.wait(timeout=0.3)
    (delete,) = _verbs(l2, "/Delete")
    assert delete[-1] == task

    for path in (LEVEL2, UI):               # create inside try, delete inside finally
        text = path.read_text(encoding="ascii")
        create_at, fin = text.index("/Create"), text.rindex("finally")
        assert text.rindex("try", 0, create_at) < create_at < fin, path.name
        assert "/Delete" in text[fin:], path.name
    assert "[int]$TimeoutMinutes = 240" in LEVEL2.read_text(encoding="ascii")


def test_level2_report_keeps_the_installer_hash_and_outside_count(l2, monkeypatch, tmp_path):
    exe = tmp_path / "setup.exe"
    data = b"installer bytes" * 1000
    exe.write_bytes(data)
    rep = tmp_path / "report.json"
    rep.write_text(json.dumps({"other": 1, "x_runs": [1]}), encoding="utf-8")
    monkeypatch.setenv("FOODOPT_L2_REPORT", str(rep))
    monkeypatch.setenv("FOODOPT_L2_INSTALLER", str(exe))
    l2.report_installer_sha256()
    got = json.loads(rep.read_text(encoding="utf-8"))
    assert got["installer_sha256"] == hashlib.sha256(data).hexdigest()
    assert got["other"] == 1 and got["x_runs"] == [1]

    text = TRAFFIC.read_text(encoding="utf-8")
    report = 'l2.report("wfp_outside", len(outside))'
    assert text.count(report) == 1
    assert text.index(report) < text.index("assert outside == []")


def _row(pid, ppid, created, image="x.exe"):
    return {"pid": pid, "ppid": ppid, "created": created, "image": image, "cmd": image}


def test_the_traffic_audit_counts_the_shells_whole_tree(l2):
    """I4: the shell's descendants by ParentProcessId (WebView2's included),
    not only images under the install folder. A process that started before
    its 'parent' (the parent's pid was reused) is not adopted; a child whose
    parent has exited stays in once it was seen."""
    rows = [_row(10, 1, 100, "FoodOptimizer.exe"), _row(11, 10, 101, "python.exe"),
            _row(12, 10, 102, "msedgewebview2.exe"), _row(13, 12, 103, "msedgewebview2.exe"),
            _row(14, 13, 90, "stranger.exe"), _row(15, 2, 104, "msedgewebview2.exe")]
    assert l2.process_tree(10, rows) == {10, 11, 12, 13}
    after_close = [r for r in rows if r["pid"] not in (10, 12)]
    assert l2.process_tree(10, after_close, {12: 102, 13: 103}) == {13}
    assert l2.process_tree(10, after_close, {13: 999}) == set()      # pid 13 reused since

    event = {"ProcessID": "13", "Application": r"\device\harddiskvolume3\edge\msedgewebview2.exe"}
    assert l2.ours(event, r"C:\FOL2\s\Food Optimizer", {10: {}, 13: {}})
    assert l2.ours(event, r"C:\FOL2\s\Food Optimizer", {13})

    text = TRAFFIC.read_text(encoding="utf-8")
    assert "l2.ours(e, per_user.root, tree)" in text
    assert "l2.TreeWatch(job.pid)" in text
    assert 'l2.report("wfp_outside_by_process"' in text
    for name in ("test_the_tree_includes_webview2", "test_webview2_runs_with_our_browser_arguments"):
        assert f"def {name}(" in text, name
