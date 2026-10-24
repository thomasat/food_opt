"""Release guards: the bytes the Mac bundle and the launcher hash check out
the same on every OS, one file holds the version, and build_dmg.sh refuses a
release build it can't trust."""
import pathlib
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
MANIFEST = REPO / "desktop" / "shared" / "bundle_manifest.txt"


def _git(*args, cwd=REPO):
    """git's stdout; skips the test when this is not a git checkout."""
    probe = subprocess.run(["git", "rev-parse", "--git-dir"], cwd=cwd,
                           capture_output=True, text=True)
    if probe.returncode != 0:
        pytest.skip("not a git checkout")
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                          text=True, check=True).stdout


# ---------- line endings (Task 2) ----------

_HASHED_TEXT = ["desktop/requirements.lock.txt", "desktop/launcher.sh",
                "desktop/start_here.txt", "desktop/Info.plist",
                "desktop/build_dmg.sh", "desktop/VERSION", "app.py"]


def _attrs(path):
    out = {}
    for line in _git("check-attr", "text", "eol", "--", path).splitlines():
        _, attr, value = line.rsplit(": ", 2)
        out[attr] = value
    return out


def _check_lf(paths):
    for path in paths:
        attrs = _attrs(path)
        if path.endswith(".xlsx"):
            assert attrs["text"] == "unset", (path, attrs)
        else:
            assert attrs["eol"] == "lf", (path, attrs)


def test_shipped_text_checks_out_with_lf():
    """The bundle and the Mac lock are compared byte for byte (the launcher
    hashes the lock), so a Windows checkout must not turn LF into CRLF."""
    _check_lf(_HASHED_TEXT)
    if not MANIFEST.exists():
        pytest.skip("bundle_manifest.txt arrives in Task 5")
    _check_lf(MANIFEST.read_text(encoding="utf-8").splitlines())


def test_no_committed_crlf_but_the_usda_table():
    crlf = set()
    for line in _git("ls-files", "--eol").splitlines():
        meta, path = line.split("\t", 1)
        if meta.split()[0] == "i/crlf":
            crlf.add(path)
    assert crlf <= {"data/usda_nutrition_data.csv"}, sorted(crlf)
