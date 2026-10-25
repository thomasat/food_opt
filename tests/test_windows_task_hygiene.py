"""The Windows test harnesses' scheduled tasks never fire on schedule and are
always deleted (Task 7b). l2_support is exercised on any OS with
subprocess.run recorded; the two .ps1 harnesses are read as text."""
import importlib.util
import json
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
WINDOWS = REPO / "desktop" / "windows"
OLD_TRIGGER = "23:" + "59"   # the leaked tasks' start time
LEVEL2 = WINDOWS / "tests" / "run_level2.ps1"
UI = WINDOWS / "FoodOptimizer" / "uitests" / "run_ui_tests.ps1"


@pytest.fixture
def l2(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("l2_support_hygiene", WINDOWS / "tests" / "l2_support.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "JOBS", tmp_path / "jobs")
    module.calls = calls = []
    module.fail_run = False

    def fake_run(args, **kwargs):
        calls.append(list(args))
        rc = 0
        if args[:2] == ["schtasks", "/Run"]:
            if module.fail_run:
                rc = 1
            else:   # the job runner reports its pid
                for f in (tmp_path / "jobs").glob("*.json"):
                    if not f.name.endswith(".result.json"):
                        spec_ = json.loads(f.read_text(encoding="utf-8"))
                        pathlib.Path(spec_["result"]).write_text(json.dumps({"pid": 1, "elevated": False}),
                                                                 encoding="utf-8")
        return subprocess.CompletedProcess(args, rc, "", "")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    yield module
    module.end_all_jobs()
    sys.modules.pop(spec.name, None)


def _verbs(l2, verb):
    return [c for c in l2.calls if c[:2] == ["schtasks", verb]]


def test_create_uses_a_trigger_that_never_fires(l2):
    l2.run_as("user", ["x"], wait=False)
    (create,) = _verbs(l2, "/Create")
    assert OLD_TRIGGER not in create
    n = len(l2.NEVER_ON_SCHEDULE)
    i = create.index("/SC")
    assert create[i:i + n] == l2.NEVER_ON_SCHEDULE == ["/SC", "ONCE", "/SD", "01/01/2020", "/ST", "00:00"]


def test_unwaited_job_is_deleted_by_end_all_jobs(l2):
    job = l2.run_as("system", ["x"], wait=False)
    assert not _verbs(l2, "/Delete")
    l2.end_all_jobs()
    (delete,) = _verbs(l2, "/Delete")
    assert delete[-1] == job.task and job.task.startswith("FoodOptL2-j")


def test_timed_out_wait_still_deletes_the_task(l2):
    job = l2.run_as("user", ["x"], wait=False)
    with pytest.raises(TimeoutError):
        job.wait(timeout=0.3)
    assert len(_verbs(l2, "/Delete")) == 1


def test_run_failure_deletes_the_task_and_raises(l2):
    l2.fail_run = True
    with pytest.raises(RuntimeError, match="/Run failed"):
        l2.run_as("user", ["x"], wait=False)
    assert len(_verbs(l2, "/Delete")) == 1
    assert len(_verbs(l2, "/Create")) == 1


def test_end_all_jobs_after_a_normal_wait_is_harmless(l2):
    job = l2.run_as("user", ["x"], wait=False)
    job.result.write_text(json.dumps({"pid": 1, "elevated": False, "rc": 0}), encoding="utf-8")
    assert job.wait(5) == 0
    before = list(l2.calls)
    l2.end_all_jobs()
    l2.end_all_jobs()
    job.close()
    assert l2.calls == before
    assert len(_verbs(l2, "/Create")) == 1 and len(_verbs(l2, "/Delete")) == 1


def test_conftest_ends_jobs_when_the_session_ends():
    text = (WINDOWS / "tests" / "conftest.py").read_text(encoding="utf-8")
    assert "end_all_jobs" in text
    assert "atexit" in (WINDOWS / "tests" / "l2_support.py").read_text(encoding="utf-8")


@pytest.mark.parametrize("path", [LEVEL2, UI], ids=["run_level2", "run_ui_tests"])
def test_ps1_trigger_and_finally(path):
    text = path.read_text(encoding="ascii")
    assert OLD_TRIGGER not in text
    assert "01/01/2020" in text and "/SD" in text and "00:00" in text
    create, fin = text.index("/Create"), text.rindex("finally")
    assert text.rindex("try", 0, create) < create < fin, "create/run/wait belong inside the try"
    assert "/Delete" in text[fin:], "the delete belongs inside the finally"


def test_level2_has_a_deadline():
    text = LEVEL2.read_text(encoding="ascii")
    assert "[int]$TimeoutMinutes = 240" in text
    assert 'Stop-Run "Level 2 did not finish in $TimeoutMinutes minutes"' in text
