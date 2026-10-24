"""build_installer.ps1 refuses an untrusted release build exactly as 3a's
build_dmg.sh does (tests/test_release_guards.py), in a throwaway tagged
checkout on a path with a space."""
import re
import subprocess

import pytest

from conftest import WINDOWS, own_git_env, powershell

_GIT_ID = ["-c", "user.name=t", "-c", "user.email=t@example.com",
           "-c", "commit.gpgsign=false", "-c", "tag.gpgSign=false", "-c", "core.autocrlf=false"]
_REFUSE_HEAD = "refusing to build: HEAD is not tag v0.8.0 (tag first: git tag -a v0.8.0 "
_REFUSE_DIRTY = ("refusing to build: uncommitted changes "
                 "(build from a clean worktree at the tag)")
_REFUSE_VERSION = "refusing to build: desktop/VERSION must hold one X.Y.Z line"
_USAGE = "usage: build_installer.ps1 [-Version X.Y.Z] [-Check] [-TestBuild] [-OutDir DIR]"
_SCRIPTS = ("build_installer.ps1", "fetch_pinned.ps1", "pins.json", "build_env.py")


def _git(repo, *args):
    return subprocess.run(["git", *_GIT_ID, *args], cwd=repo, check=True,
                          capture_output=True, text=True, env=own_git_env()).stdout


def _commit_all(repo, message):
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)


def _write_manifest(repo, raw):
    (repo / "desktop" / "shared" / "bundle_manifest.txt").write_bytes(raw)


def _retag(repo, message):
    _commit_all(repo, message)
    _git(repo, "tag", "-f", "v0.8.0")


@pytest.fixture
def release_repo(short_tmp):
    """3a's throwaway checkout, tagged v0.8.0, holding the real Windows
    build scripts, under a folder whose name has a space."""
    repo = short_tmp / "a repo"
    windows = repo / "desktop" / "windows"
    windows.mkdir(parents=True)
    for name in _SCRIPTS:
        (windows / name).write_bytes((WINDOWS / name).read_bytes())
    (windows / ".gitignore").write_bytes(b"/build/\n")
    (repo / "desktop" / "VERSION").write_bytes(b"0.8.0\n")
    (repo / "desktop" / ".gitignore").write_bytes(b"dist/\n.cache/\n")
    (repo / "app.py").write_bytes(b"print('app')\n")
    (repo / "desktop" / "shared").mkdir()
    (repo / "data" / "x").mkdir(parents=True)
    (repo / "data" / "x" / "Name with spaces.json").write_bytes(b"{}\n")
    _write_manifest(repo, b"app.py\ndata/x/Name with spaces.json\n")
    _git(repo, "init", "-q")
    _commit_all(repo, "release")
    _git(repo, "tag", "v0.8.0")
    return repo


def run_build(repo, *args):
    return powershell("-File", repo / "desktop" / "windows" / "build_installer.ps1", *args,
                      env=own_git_env(), timeout=120)


def _refused(result, message):
    assert result.returncode == 2, (result.returncode, result.stdout, result.stderr)
    assert message in result.stderr, result.stderr


def test_check_passes_on_a_clean_tagged_tree(release_repo):
    result = run_build(release_repo, "-Check")
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines()[:2] == ["ready to build 0.8.0 from v0.8.0", "manifest: 2 files"]


def test_check_refuses_a_disagreeing_version(release_repo):
    for arg in ("0.9.0", "v0.8.0"):
        _refused(run_build(release_repo, "-Check", "-Version", arg),
                 f"refusing to build: version {arg} disagrees with desktop/VERSION (0.8.0)")


def test_check_refuses_a_head_that_is_not_the_tag(release_repo):
    (release_repo / "app.py").write_bytes(b"print('later')\n")
    _commit_all(release_repo, "one past the tag")
    short = _git(release_repo, "rev-parse", "--short", "HEAD").strip()
    _refused(run_build(release_repo, "-Check"), _REFUSE_HEAD + short + ")")


def test_check_refuses_uncommitted_changes(release_repo):
    (release_repo / "app.py").write_bytes(b"print('edited')\n")
    _refused(run_build(release_repo, "-Check"), _REFUSE_DIRTY)
    _git(release_repo, "checkout", "--", "app.py")
    assert run_build(release_repo, "-Check").returncode == 0
    (release_repo / "notes.py").write_bytes(b"")
    _refused(run_build(release_repo, "-Check"), _REFUSE_DIRTY)


def test_ignored_build_output_is_not_a_change(release_repo):
    for rel in ("desktop/windows/build/x", "desktop/windows/.cache/y"):
        path = release_repo / rel
        path.parent.mkdir(parents=True)
        path.write_bytes(b"x")
    result = run_build(release_repo, "-Check")
    assert result.returncode == 0, result.stdout + result.stderr


def test_version_file_must_be_x_y_z(release_repo):
    for raw in (b"0.8.0\r\n", b"0.8\n", b"0.8.0\n\n", b"\xd9\xa0.8.0\n"):
        (release_repo / "desktop" / "VERSION").write_bytes(raw)
        _retag(release_repo, f"version {raw!r}")
        _refused(run_build(release_repo, "-Check"), _REFUSE_VERSION)


def test_manifest_with_crlf_is_refused(release_repo):
    _write_manifest(release_repo, b"app.py\r\ndata/x/Name with spaces.json\r\n")
    _retag(release_repo, "manifest saved on Windows")
    _refused(run_build(release_repo, "-Check"),
             "refusing to build: bundle_manifest.txt has Windows line endings")


def test_manifest_without_final_newline_keeps_its_last_entry(release_repo):
    _write_manifest(release_repo, b"app.py\ndata/x/Name with spaces.json")
    _retag(release_repo, "manifest without a final newline")
    result = run_build(release_repo, "-Check")
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines()[1] == "manifest: 2 files"


def test_manifest_entries_are_checked(release_repo):
    cases = (
        (b"app.py\nui_gone.py\n",
         "refusing to build: listed in bundle_manifest.txt but missing from the repo: ui_gone.py"),
        (b"../x.py\napp.py\n",
         "refusing to build: bundle_manifest.txt entry must be a repo-relative path: ../x.py"),
        (b"C:/x.py\n",
         "refusing to build: bundle_manifest.txt entry must be a repo-relative path: C:/x.py"),
        (b"", "refusing to build: bundle_manifest.txt is empty"),
    )
    for raw, message in cases:
        _write_manifest(release_repo, raw)
        _retag(release_repo, f"manifest {raw!r}")
        _refused(run_build(release_repo, "-Check"), message)
    (release_repo / "desktop" / "shared" / "bundle_manifest.txt").unlink()
    _retag(release_repo, "no manifest")
    _refused(run_build(release_repo, "-Check"), "refusing to build: bundle_manifest.txt is missing")


def test_unknown_option_shows_usage(release_repo):
    for args in (("-Sign",), ("-Check", "-TestBuild"), ("extra",)):
        _refused(run_build(release_repo, *args), _USAGE)


def test_a_long_image_root_is_refused(release_repo, short_tmp):
    out = str(short_tmp) + "\\" + "o" * (90 - len(str(short_tmp)) - 1)
    assert len(out) == 90
    result = run_build(release_repo, "-Check", "-OutDir", out)
    _refused(result, f"refusing to build: image folder is {90 + len(chr(92) + 'FoodOptimizer' + chr(92))} "
                     "characters; at most 80 (clone to a shorter path)")


def test_a_refused_build_creates_nothing(release_repo):
    _refused(run_build(release_repo, "-Version", "0.9.0"),
             "refusing to build: version 0.9.0 disagrees with desktop/VERSION (0.8.0)")
    assert not (release_repo / "desktop" / "windows" / "build").exists()
    assert not (release_repo / "desktop" / "windows" / ".cache").exists()


def test_a_test_build_skips_the_release_guards_and_warns(release_repo):
    (release_repo / "app.py").write_bytes(b"print('edited')\n")
    result = run_build(release_repo, "-TestBuild", "-OutDir", release_repo / "nowhere")
    out = result.stdout
    assert "TEST BUILD - not for release" in out
    assert "UNSIGNED build - internal testing only." in out
    assert _REFUSE_DIRTY not in result.stderr
    assert re.search(r"build failed: ", result.stdout + result.stderr)
    assert result.returncode == 1
