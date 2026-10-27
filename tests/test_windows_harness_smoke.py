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
