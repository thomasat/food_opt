"""Release guards: the bytes the Mac bundle and the launcher hash check out
the same on every OS, one file holds the version, and build_dmg.sh refuses a
release build it can't trust."""
import pathlib
import re
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


# ---------- one version (Task 3) ----------

VERSION_FILE = REPO / "desktop" / "VERSION"
NOTES_DIR = REPO / "desktop" / "release_notes"


def read_version():
    return VERSION_FILE.read_text(encoding="utf-8").strip()


def _as_tuple(version):
    return tuple(int(part) for part in version.split("."))


def test_version_file_is_one_x_y_z_line():
    assert re.fullmatch(r"\d+\.\d+\.\d+\n",
                        VERSION_FILE.read_bytes().decode("utf-8")), \
        VERSION_FILE.read_bytes()


def test_readme_download_links_name_the_version():
    text = (REPO / "README.md").read_text(encoding="utf-8")
    dmgs = re.findall(r"FoodOptimizer-(\d+\.\d+\.\d+)\.dmg", text)
    tags = re.findall(r"releases/download/v(\d+\.\d+\.\d+)/", text)
    assert dmgs and tags, (dmgs, tags)
    assert set(dmgs) | set(tags) == {read_version()}, (dmgs, tags)


def test_release_notes_match_the_version():
    version = read_version()
    assert (NOTES_DIR / f"{version}.md").is_file()
    names = sorted(p.name for p in NOTES_DIR.iterdir())
    assert all(re.fullmatch(r"\d+\.\d+\.\d+\.md", n) for n in names), names
    newer = [n for n in names if _as_tuple(n[:-3]) > _as_tuple(version)]
    assert not newer, f"notes newer than desktop/VERSION (bump it?): {newer}"
    text = (NOTES_DIR / f"{version}.md").read_text(encoding="utf-8")
    named = (re.findall(r"FoodOptimizer-(\d+\.\d+\.\d+)", text)
             + re.findall(r"/v(\d+\.\d+\.\d+)/", text))
    assert set(named) <= {version}, named


def test_desktop_readme_names_no_stale_version():
    text = (REPO / "desktop" / "README.md").read_text(encoding="utf-8")
    allowed = {"X.Y.Z", "X.Y.Z-test", read_version()}
    named = re.findall(r'FoodOptimizer-([^ ")]+?)\.dmg', text)
    assert set(named) <= allowed, named
