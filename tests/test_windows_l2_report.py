"""The Level 2 report ties to the build (Task 7c): it names the installer it
tested by sha256 and counts the connections outside loopback. l2_support is
exercised on any OS; the traffic test is read as text."""
import hashlib
import importlib.util
import json
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
WINDOWS = REPO / "desktop" / "windows"
TRAFFIC = WINDOWS / "tests" / "test_l2_40_traffic.py"


@pytest.fixture
def l2():
    spec = importlib.util.spec_from_file_location("l2_support_report", WINDOWS / "tests" / "l2_support.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


def test_the_installer_hash_is_reported_and_other_keys_survive(l2, monkeypatch, tmp_path):
    exe = tmp_path / "setup.exe"
    data = b"installer bytes" * 100_000
    exe.write_bytes(data)
    rep = tmp_path / "report.json"
    rep.write_text(json.dumps({"other": 1, "x_runs": [1]}), encoding="utf-8")
    monkeypatch.setenv("FOODOPT_L2_REPORT", str(rep))
    monkeypatch.setenv("FOODOPT_L2_INSTALLER", str(exe))
    l2.report_installer_sha256()
    got = json.loads(rep.read_text(encoding="utf-8"))
    assert got["installer_sha256"] == hashlib.sha256(data).hexdigest()
    assert got["other"] == 1 and got["x_runs"] == [1]


def test_nothing_is_written_without_a_report_path(l2, monkeypatch, tmp_path):
    exe = tmp_path / "setup.exe"
    exe.write_bytes(b"x")
    monkeypatch.delenv("FOODOPT_L2_REPORT", raising=False)
    monkeypatch.setenv("FOODOPT_L2_INSTALLER", str(exe))
    monkeypatch.chdir(tmp_path)
    l2.report_installer_sha256()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["setup.exe"]


def test_nothing_is_written_without_an_installer(l2, monkeypatch, tmp_path):
    rep = tmp_path / "report.json"
    monkeypatch.setenv("FOODOPT_L2_REPORT", str(rep))
    monkeypatch.delenv("FOODOPT_L2_INSTALLER", raising=False)
    l2.report_installer_sha256()
    assert not rep.exists()


def test_report_merge_semantics_are_unchanged(l2, monkeypatch, tmp_path):
    rep = tmp_path / "report.json"
    monkeypatch.setenv("FOODOPT_L2_REPORT", str(rep))
    l2.report("a", 1)
    l2.report("a", 2)
    l2.report("t_runs", 1)
    l2.report("t_runs", 2)
    assert json.loads(rep.read_text(encoding="utf-8")) == {"a": 2, "t_runs": [1, 2]}


def test_the_traffic_test_reports_the_outside_count_before_asserting():
    text = TRAFFIC.read_text(encoding="utf-8")
    report = 'l2.report("wfp_outside", len(outside))'
    assert text.count(report) == 1
    assert text.index(report) < text.index("assert outside == []")
