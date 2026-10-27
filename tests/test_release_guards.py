"""Release guards: the bytes the Mac bundle and the launcher hash check out
the same on every OS, one file holds the version, and build_dmg.sh refuses a
release build it can't trust."""
import os
import inspect
import pathlib
import re
import subprocess

import pytest

# The build scripts are Mac-only bash; their behaviour tests skip on Windows.
MAC_SCRIPT_CONDITION = "sys.platform == 'win32'"
mac_script = pytest.mark.skipif(MAC_SCRIPT_CONDITION, reason="Mac build script")

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
    _check_lf(_HASHED_TEXT + ["desktop/shared/bundle_manifest.txt"])
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
    exes = re.findall(r"FoodOptimizer-(\d+\.\d+\.\d+)-Windows\.exe", text)
    tags = re.findall(r"releases/download/v(\d+\.\d+\.\d+)/", text)
    assert dmgs and exes and tags, (dmgs, exes, tags)
    assert set(dmgs) | set(exes) | set(tags) == {read_version()}, (dmgs, exes, tags)
    links = re.findall(r"\]\((https://[^)]+)\)", text)
    assert any(link.endswith(".dmg") for link in links) and any(link.endswith("-Windows.exe") for link in links)


# The release's download links and the files a release publishes (the
# per-file list is attached, not advertised).
URL_BASE = "https://github.com/thomasat/food_opt/releases/download"
# A claim no test backs yet; one level of parentheses inside.
PENDING = re.compile(r"PENDING\(((?:[^()]|\([^()]*\))*)\)")
# The texts sent at a release: none may hold a PENDING marker at the tag.
SENT_TEXTS = ("README.md", "desktop/release_notes/{v}.md", "desktop/windows/download_email.md",
              "desktop/windows/README.md", "desktop/shared/start_here.src.md")


def release_assets(v):
    return {"dmg": f"FoodOptimizer-{v}.dmg", "exe": f"FoodOptimizer-{v}-Windows.exe",
            "files": f"FoodOptimizer-{v}-Windows.files.txt"}


def test_download_links_are_this_versions_release_assets():
    """Every release-download link in the README, the release page and the
    download email is one of this version's assets (desktop/VERSION)."""
    version = read_version()
    allowed = {f"{URL_BASE}/v{version}/{name}" for name in release_assets(version).values()}
    for rel in ("README.md", f"desktop/release_notes/{version}.md", "desktop/windows/download_email.md"):
        urls = re.findall(r"https://[^\s)\]>\"']*releases/download/[^\s)\]>\"']*",
                          (REPO / rel).read_text(encoding="utf-8"))
        assert urls, rel
        assert set(urls) <= allowed, (rel, set(urls) - allowed)
    email = (REPO / "desktop" / "windows" / "download_email.md").read_text(encoding="utf-8")
    assert set(re.findall(r"FoodOptimizer-(\d+\.\d+\.\d+)", email)) == {version}


def test_windows_readme_names_no_stale_installer():
    text = (REPO / "desktop" / "windows" / "README.md").read_text(encoding="utf-8")
    version = read_version()
    named = set(re.findall(r"FoodOptimizer-(\S+?)-Windows", text))
    assert named <= {version, "X.Y.Z", f"{version}-test", "0.0.1", "<ver>"}, named


def test_both_readmes_carry_the_same_windows_requirements():
    win = (REPO / "desktop" / "windows" / "README.md").read_text(encoding="utf-8")
    block = win[win.index("## System requirements (copy-paste"):]
    line = next(l for l in block.splitlines()[1:] if l.strip())
    assert line.startswith("Needs 64-bit Windows 10 (22H2) or Windows 11")
    assert line in (REPO / "README.md").read_text(encoding="utf-8")


def _at_release_tag():
    probe = subprocess.run(["git", "describe", "--exact-match", "--tags", "HEAD"], cwd=REPO,
                           capture_output=True, text=True)
    return probe.returncode == 0 and probe.stdout.strip() == f"v{read_version()}"


release_tag_only = pytest.mark.skipif(not _at_release_tag(), reason="release tag only")


@release_tag_only
def test_no_pending_marker_at_a_release_tag():
    """At the release tag, no sent text may still hold a PENDING(...)
    marker: the README, the release page, the download email and the
    Windows README."""
    version = read_version()
    for rel in SENT_TEXTS:
        rel = rel.format(v=version)
        assert not PENDING.search((REPO / rel).read_text(encoding="utf-8")), rel


def test_sent_texts_exist_and_their_markers_parse():
    """The tag check above reads these files; each exists, and every
    PENDING( in it is a whole marker the check can see."""
    for rel in SENT_TEXTS:
        text = (REPO / rel.format(v=read_version())).read_text(encoding="utf-8")
        assert text.count("PENDING(") == len(PENDING.findall(text)), rel


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
_REFUSE_ELSEWHERE = "refusing to build: HEAD is not tag v0.8.0 (v0.8.0 is "
_REFUSE_DIRTY = ("refusing to build: uncommitted changes "
                 "(build from a clean worktree at the tag)")
_REFUSE_VERSION = "refusing to build: desktop/VERSION must hold one X.Y.Z line"
_USAGE = "usage: build_dmg.sh [X.Y.Z] | --check [X.Y.Z] | --test-build"


def _own_git_env():
    """The environment without GIT_*: inside a git hook, GIT_DIR and
    GIT_INDEX_FILE would point every git call at the outer checkout."""
    return {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


def _repo_git(repo, *args):
    return subprocess.run(["git", *_GIT_ID, *args], cwd=repo, check=True,
                          capture_output=True, text=True, env=_own_git_env()).stdout


def _commit_all(repo, message):
    _repo_git(repo, "add", "-A")
    _repo_git(repo, "commit", "-q", "-m", message)


def _write_manifest(repo, raw):
    (repo / "desktop" / "shared" / "bundle_manifest.txt").write_bytes(raw)


def _retag(repo, message):
    _commit_all(repo, message)
    _repo_git(repo, "tag", "-f", "v0.8.0")


@pytest.fixture
def release_repo(tmp_path):
    return _make_release_repo(tmp_path)


def _make_release_repo(tmp_path):
    """A throwaway checkout holding the real build script, tagged v0.8.0."""
    repo = tmp_path / "repo"
    (repo / "desktop").mkdir(parents=True)
    (repo / "desktop" / "build_dmg.sh").write_bytes(BUILD_SCRIPT.read_bytes())
    (repo / "desktop" / "VERSION").write_bytes(b"0.8.0\n")
    (repo / "desktop" / ".gitignore").write_bytes(b"dist/\n.cache/\n")
    (repo / "app.py").write_bytes(b"print('app')\n")
    (repo / "desktop" / "shared").mkdir()
    (repo / "data" / "x").mkdir(parents=True)
    (repo / "data" / "x" / "Name with spaces.json").write_bytes(b"{}\n")
    _write_manifest(repo, b"app.py\ndata/x/Name with spaces.json\n")
    _repo_git(repo, "init", "-q")
    _commit_all(repo, "release")
    _repo_git(repo, "tag", "v0.8.0")
    return repo


def run_build(repo, *args):
    return subprocess.run(["bash", str(repo / "desktop" / "build_dmg.sh"), *args],
                          capture_output=True, text=True, timeout=60,
                          env=_own_git_env())


def _refused(result, message):
    assert result.returncode == 2, (result.returncode, result.stdout, result.stderr)
    assert message in result.stderr, result.stderr


@mac_script
def test_check_passes_on_a_clean_tagged_tree(release_repo):
    result = run_build(release_repo, "--check")
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[0] == "ready to build 0.8.0 from v0.8.0"


@mac_script
def test_check_refuses_a_disagreeing_version(release_repo):
    for arg in ("0.9.0", "v0.8.0"):
        _refused(run_build(release_repo, "--check", arg),
                 f"refusing to build: version {arg} disagrees with "
                 "desktop/VERSION (0.8.0)")


@mac_script
def test_check_refuses_a_head_when_the_tag_does_not_exist(release_repo):
    _repo_git(release_repo, "tag", "-d", "v0.8.0")
    r = run_build(release_repo, "--check")
    _refused(r, _REFUSE_HEAD)


@mac_script
def test_a_tag_on_another_commit_says_check_it_out(release_repo):
    (release_repo / "app.py").write_bytes(b"print('v2')\n")
    _commit_all(release_repo, "after the tag")  # HEAD moves past v0.8.0
    r = run_build(release_repo, "--check")
    _refused(r, _REFUSE_ELSEWHERE)
    assert "check out the tag or bump desktop/VERSION" in r.stdout + r.stderr
    assert "git tag -a" not in r.stdout + r.stderr


@mac_script
def test_check_refuses_uncommitted_changes(release_repo):
    (release_repo / "app.py").write_bytes(b"print('edited')\n")
    _refused(run_build(release_repo, "--check"), _REFUSE_DIRTY)
    _repo_git(release_repo, "checkout", "--", "app.py")
    assert run_build(release_repo, "--check").returncode == 0
    (release_repo / "notes.py").write_bytes(b"")
    _refused(run_build(release_repo, "--check"), _REFUSE_DIRTY)


@mac_script
def test_ignored_build_output_is_not_a_change(release_repo):
    (release_repo / "desktop" / "dist").mkdir()
    (release_repo / "desktop" / "dist" / "old.dmg").write_bytes(b"x")
    (release_repo / "desktop" / ".cache").mkdir()
    (release_repo / "desktop" / ".cache" / "x").write_bytes(b"x")
    result = run_build(release_repo, "--check")
    assert result.returncode == 0, result.stderr


@mac_script
def test_version_file_must_be_x_y_z(release_repo):
    for raw in (b"0.8.0\r\n", b"0.8\n"):
        (release_repo / "desktop" / "VERSION").write_bytes(raw)
        _commit_all(release_repo, f"version {raw!r}")
        _repo_git(release_repo, "tag", "-f", "v0.8.0")
        _refused(run_build(release_repo, "--check"), _REFUSE_VERSION)


@mac_script
def test_a_release_build_refuses_before_building(release_repo):
    _refused(run_build(release_repo, "0.9.0"),
             "refusing to build: version 0.9.0 disagrees with desktop/VERSION (0.8.0)")
    assert not (release_repo / "desktop" / "dist").exists()
    assert not (release_repo / "desktop" / ".cache").exists()


@mac_script
def test_unknown_option_shows_usage(release_repo):
    _refused(run_build(release_repo, "--sign"), _USAGE)


def test_build_script_takes_its_version_from_the_file():
    text = BUILD_SCRIPT.read_text(encoding="utf-8")
    assert not re.findall(r"\b\d+\.\d+\.\d+\b", text)
    assert '"$DESKTOP_DIR/VERSION"' in text


# ---------- the manifest build_dmg.sh copies (Task 5) ----------

@mac_script
def test_check_counts_the_manifest(release_repo):
    result = run_build(release_repo, "--check")
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[1] == "manifest: 2 files"


@mac_script
def test_manifest_without_final_newline_keeps_its_last_entry(release_repo):
    _write_manifest(release_repo, b"app.py\ndata/x/Name with spaces.json")
    _retag(release_repo, "manifest without a final newline")
    result = run_build(release_repo, "--check")
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[1] == "manifest: 2 files"


@mac_script
def test_manifest_with_crlf_is_refused(release_repo):
    _write_manifest(release_repo, b"app.py\r\ndata/x/Name with spaces.json\r\n")
    _retag(release_repo, "manifest saved on Windows")
    _refused(run_build(release_repo, "--check"),
             "refusing to build: bundle_manifest.txt has Windows line endings")


@mac_script
def test_manifest_entry_missing_from_the_repo_is_refused(release_repo):
    _write_manifest(release_repo, b"app.py\ndata/x/Name with spaces.json\nui_gone.py\n")
    _retag(release_repo, "manifest names a missing file")
    _refused(run_build(release_repo, "--check"),
             "refusing to build: listed in bundle_manifest.txt but missing "
             "from the repo: ui_gone.py")


@mac_script
def test_manifest_entry_outside_the_repo_is_refused(release_repo):
    (release_repo.parent / "x.py").write_bytes(b"")
    _write_manifest(release_repo, b"../x.py\napp.py\n")
    _retag(release_repo, "manifest leaves the repo")
    _refused(run_build(release_repo, "--check"),
             "refusing to build: bundle_manifest.txt entry must be a "
             "repo-relative path: ../x.py")


@mac_script
def test_manifest_missing_or_empty_is_refused(release_repo):
    _write_manifest(release_repo, b"")
    _retag(release_repo, "empty manifest")
    _refused(run_build(release_repo, "--check"),
             "refusing to build: bundle_manifest.txt is empty")
    (release_repo / "desktop" / "shared" / "bundle_manifest.txt").unlink()
    _retag(release_repo, "no manifest")
    _refused(run_build(release_repo, "--check"),
             "refusing to build: bundle_manifest.txt is missing")


@mac_script
def test_fixture_repos_ignore_an_outer_git_dir(tmp_path, monkeypatch):
    """Run from a git hook (the repo's pre-commit runs pytest), git exports
    GIT_DIR and GIT_INDEX_FILE; the throwaway repos must not commit or tag
    into the checkout running the tests."""
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep.txt").write_bytes(b"keep\n")
    _repo_git(victim, "init", "-q")
    _commit_all(victim, "victim")
    head = _repo_git(victim, "rev-parse", "HEAD")
    monkeypatch.setenv("GIT_DIR", str(victim / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(victim))
    monkeypatch.setenv("GIT_INDEX_FILE", str(victim / ".git" / "index"))
    repo = _make_release_repo(tmp_path / "inner")
    result = run_build(repo, "--check")
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(name)
    assert _repo_git(victim, "rev-parse", "HEAD") == head
    assert _repo_git(victim, "tag") == ""
    assert result.returncode == 0, result.stderr


def test_mac_script_tests_skip_on_windows():
    """The behaviour tests run Mac-only bash scripts; on Windows they would
    check nothing and fail the Windows pytest gate."""
    unmarked = []
    for name, fn in sorted(globals().items()):
        if not (name.startswith("test_") and callable(fn)):
            continue
        params = inspect.signature(fn).parameters
        if not ({"release_repo", "lock_sandbox"} & set(params)
                or name == "test_fixture_repos_ignore_an_outer_git_dir"):
            continue
        marks = getattr(fn, "pytestmark", [])
        if not any(m.name == "skipif" and m.args == (MAC_SCRIPT_CONDITION,)
                   for m in marks):
            unmarked.append(name)
    assert not unmarked, unmarked
