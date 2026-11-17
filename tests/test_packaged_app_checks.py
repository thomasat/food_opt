"""desktop/shared/packaged_app_checks.py: the checks test_e2e.sh runs inside
the packaged app keep every check and sentinel, and run on any Python."""
import importlib.util
import inspect
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
CHECKS_FILE = REPO / "desktop" / "shared" / "packaged_app_checks.py"
E2E = REPO / "desktop" / "test_e2e.sh"

NAMES = ["smoke", "ui_errors", "wording_wave", "openpyxl", "kitchen_trust",
         "spreadsheet_feel", "rules_premix_sample_draft"]
ALL_SENTINELS = {"SMOKE_OK", "UI_OK", "WAVE_OK", "OPENPYXL_OK",
                 "KITCHEN_TRUST_OK", "GRID_OK", "RULES_OK", "PREMIX_OK",
                 "SAMPLE_OK", "DRAFT_OK"}


def _load():
    spec = importlib.util.spec_from_file_location("packaged_app_checks", CHECKS_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_check_names_only_grow():
    assert set(_load().CHECKS) >= set(NAMES)


def test_every_sentinel_is_kept():
    assert set().union(*_load().SENTINELS.values()) >= ALL_SENTINELS


def test_every_sentinel_is_printed_by_its_check():
    checks = _load()
    for name, sentinels in checks.SENTINELS.items():
        source = inspect.getsource(checks.CHECKS[name])
        for sentinel in sentinels:
            assert f'print("{sentinel}")' in source, (name, sentinel)


def test_e2e_runs_every_check_through_the_file():
    text = E2E.read_text(encoding="utf-8")
    assert "packaged_app_checks.py" in text
    for name in NAMES:
        assert f"run_check {name}" in text, name
    for sentinel in ALL_SENTINELS:
        assert f"grep -q {sentinel}" in text, sentinel
    assert not [line for line in text.splitlines() if "<<'PY'" in line]


def test_unknown_check_is_refused(tmp_path):
    cwd = os.getcwd()
    assert _load().main(["--resources", ".", "--projects", str(tmp_path),
                         "--only", "nope"]) == 2
    assert os.getcwd() == cwd


def test_packaged_checks_pass_against_the_source_tree(tmp_path):
    """The proof the file runs on any OS: the Windows job runs this too."""
    env = dict(os.environ, HOME=str(tmp_path), USERPROFILE=str(tmp_path),
               PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [sys.executable, str(CHECKS_FILE), "--resources", str(REPO),
         "--projects", str(tmp_path / "FoodOptimizer")],
        env=env, capture_output=True, text=True, timeout=1200)
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-4000:]
    missing = [s for s in sorted(ALL_SENTINELS) if s not in result.stdout]
    assert not missing, (missing, result.stdout[-4000:])
