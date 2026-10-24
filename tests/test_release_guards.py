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


# ---------- build_dmg.sh guards (Task 4) ----------

BUILD_SCRIPT = REPO / "desktop" / "build_dmg.sh"
_GIT_ID = ["-c", "user.name=t", "-c", "user.email=t@example.com",
           "-c", "commit.gpgsign=false", "-c", "tag.gpgSign=false"]
_REFUSE_HEAD = "refusing to build: HEAD is not tag v0.8.0 (tag first: git tag -a v0.8.0 "
_REFUSE_DIRTY = ("refusing to build: uncommitted changes "
                 "(build from a clean worktree at the tag)")
_REFUSE_VERSION = "refusing to build: desktop/VERSION must hold one X.Y.Z line"
_USAGE = "usage: build_dmg.sh [X.Y.Z] | --check [X.Y.Z] | --test-build"


def _repo_git(repo, *args):
    return subprocess.run(["git", *_GIT_ID, *args], cwd=repo, check=True,
                          capture_output=True, text=True).stdout


def _commit_all(repo, message):
    _repo_git(repo, "add", "-A")
    _repo_git(repo, "commit", "-q", "-m", message)


@pytest.fixture
def release_repo(tmp_path):
    """A throwaway checkout holding the real build script, tagged v0.8.0."""
    repo = tmp_path / "repo"
    (repo / "desktop").mkdir(parents=True)
    (repo / "desktop" / "build_dmg.sh").write_bytes(BUILD_SCRIPT.read_bytes())
    (repo / "desktop" / "VERSION").write_bytes(b"0.8.0\n")
    (repo / "desktop" / ".gitignore").write_bytes(b"dist/\n.cache/\n")
    (repo / "app.py").write_bytes(b"print('app')\n")
    _repo_git(repo, "init", "-q")
    _commit_all(repo, "release")
    _repo_git(repo, "tag", "v0.8.0")
    return repo


def run_build(repo, *args):
    return subprocess.run(["bash", str(repo / "desktop" / "build_dmg.sh"), *args],
                          capture_output=True, text=True, timeout=60)


def _refused(result, message):
    assert result.returncode == 2, (result.returncode, result.stdout, result.stderr)
    assert message in result.stderr, result.stderr


def test_check_passes_on_a_clean_tagged_tree(release_repo):
    result = run_build(release_repo, "--check")
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[0] == "ready to build 0.8.0 from v0.8.0"


def test_check_refuses_a_disagreeing_version(release_repo):
    for arg in ("0.9.0", "v0.8.0"):
        _refused(run_build(release_repo, "--check", arg),
                 f"refusing to build: version {arg} disagrees with "
                 "desktop/VERSION (0.8.0)")


def test_check_refuses_a_head_that_is_not_the_tag(release_repo):
    (release_repo / "app.py").write_bytes(b"print('later')\n")
    _commit_all(release_repo, "one past the tag")
    _refused(run_build(release_repo, "--check"), _REFUSE_HEAD)


def test_check_refuses_uncommitted_changes(release_repo):
    (release_repo / "app.py").write_bytes(b"print('edited')\n")
    _refused(run_build(release_repo, "--check"), _REFUSE_DIRTY)
    _repo_git(release_repo, "checkout", "--", "app.py")
    assert run_build(release_repo, "--check").returncode == 0
    (release_repo / "notes.py").write_bytes(b"")
    _refused(run_build(release_repo, "--check"), _REFUSE_DIRTY)


def test_ignored_build_output_is_not_a_change(release_repo):
    (release_repo / "desktop" / "dist").mkdir()
    (release_repo / "desktop" / "dist" / "old.dmg").write_bytes(b"x")
    (release_repo / "desktop" / ".cache").mkdir()
    (release_repo / "desktop" / ".cache" / "x").write_bytes(b"x")
    result = run_build(release_repo, "--check")
    assert result.returncode == 0, result.stderr


def test_version_file_must_be_x_y_z(release_repo):
    for raw in (b"0.8.0\r\n", b"0.8\n"):
        (release_repo / "desktop" / "VERSION").write_bytes(raw)
        _commit_all(release_repo, f"version {raw!r}")
        _repo_git(release_repo, "tag", "-f", "v0.8.0")
        _refused(run_build(release_repo, "--check"), _REFUSE_VERSION)


def test_a_release_build_refuses_before_building(release_repo):
    _refused(run_build(release_repo, "0.9.0"),
             "refusing to build: version 0.9.0 disagrees with desktop/VERSION (0.8.0)")
    assert not (release_repo / "desktop" / "dist").exists()
    assert not (release_repo / "desktop" / ".cache").exists()


def test_unknown_option_shows_usage(release_repo):
    _refused(run_build(release_repo, "--sign"), _USAGE)


def test_build_script_takes_its_version_from_the_file():
    text = BUILD_SCRIPT.read_text(encoding="utf-8")
    assert not re.findall(r"\b\d+\.\d+\.\d+\b", text)
    assert '"$DESKTOP_DIR/VERSION"' in text
