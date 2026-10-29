"""A build that fails part-way stops with exit 1, says `build failed:`, and
leaves no build record (Review Focus 4: uv failing mid-build). A long clone
path still builds, and the image is the same from any build folder."""
import hashlib
import os
import shutil
import subprocess

from conftest import REPO, WINDOWS, own_git_env, powershell, slow

# Beside the clone (C:\work on the build VM), with the plan's names.
WORK = REPO.parent


def _offline_env(short_tmp):
    """uv with no network and an empty cache: the build stops at its uv step."""
    empty_cache = short_tmp / "uv cache"
    empty_cache.mkdir()
    return own_git_env({"UV_OFFLINE": "1", "UV_CACHE_DIR": str(empty_cache)})


@slow
def test_a_failed_install_stops_the_build(short_tmp):
    out = short_tmp / "out dir"
    r = powershell("-File", WINDOWS / "build_installer.ps1", "-TestBuild", "-OutDir", out,
                   env=_offline_env(short_tmp), timeout=1800)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "build failed: uv" in r.stdout + r.stderr
    assert not (out / "FoodOptimizer.build.txt").exists()
    assert not (out / "FoodOptimizer.files.txt").exists()
    assert "Built:" not in r.stdout


@slow
def test_an_out_dir_on_another_drive_builds(short_tmp):
    """The NuGet package is unpacked inside the image, so -OutDir on another
    drive (here a subst drive) gets past the Python step."""
    import string
    letter = next(c for c in reversed(string.ascii_uppercase[3:]) if not os.path.exists(f"{c}:\\"))
    subprocess.run(["subst", f"{letter}:", str(short_tmp)], check=True)
    try:
        r = powershell("-File", WINDOWS / "build_installer.ps1", "-TestBuild", "-OutDir",
                       f"{letter}:\\out", env=_offline_env(short_tmp), timeout=1800)
        assert r.returncode == 1, r.stdout + r.stderr
        assert "build failed: uv" in r.stdout + r.stderr, r.stdout + r.stderr
        assert (short_tmp / "out" / "FoodOptimizer" / "python" / "python.exe").is_file()
    finally:
        subprocess.run(["subst", f"{letter}:", "/d"])


@slow
def test_a_long_clone_path_gets_past_the_python_step(short_tmp):
    """The NuGet package is unpacked inside the image, which the 80-character
    budget covers, not under the clone's own .cache folder."""
    parent, out = WORK / "hd-long", WORK / "hd-lo"
    assert not parent.exists() and not out.exists(), "left from an earlier run: remove them"
    junction = parent / ("a" * 130)
    parent.mkdir()
    try:
        subprocess.run(["cmd", "/c", "mklink", "/J", str(junction), str(REPO)], check=True,
                       capture_output=True)
        r = powershell("-File", junction / "desktop" / "windows" / "build_installer.ps1",
                       "-TestBuild", "-OutDir", out, env=_offline_env(short_tmp), timeout=1800)
        assert r.returncode == 1, r.stdout + r.stderr
        assert "build failed: uv" in r.stdout + r.stderr, r.stdout + r.stderr
        assert (out / "FoodOptimizer" / "python" / "python.exe").is_file()
    finally:
        # rmdir without /s removes the junction only, never the clone it points at.
        if junction.exists():
            subprocess.run(["cmd", "/c", "rmdir", str(junction)], check=True)
        parent.rmdir()
        shutil.rmtree(out, ignore_errors=True)


def _rmtree(path):
    """shutil.rmtree that also deletes git's read-only object files."""
    import stat

    def writable(func, p, _exc):
        os.chmod(p, stat.S_IWRITE)
        func(p)
    if os.path.lexists(path):
        shutil.rmtree(path, onexc=writable)


def _clean_head():
    """The clone's commit, refusing a dirty tree (a second clone would not
    have its changes)."""
    status = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], env=own_git_env(),
                            capture_output=True, text=True, check=True).stdout
    assert status == "", "commit first: the second clone builds HEAD\n" + status
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], env=own_git_env(),
                          capture_output=True, text=True, check=True).stdout.strip()


def _second_clone(path, commit):
    """The same commit checked out in another folder, with LF files and the
    pinned downloads copied (fetch_pinned.ps1 checks their hashes)."""
    subprocess.run(["git", "-c", "core.autocrlf=false", "clone", "-q", "--no-checkout", str(REPO),
                    str(path)], env=own_git_env(), check=True)
    subprocess.run(["git", "-C", str(path), "config", "core.autocrlf", "false"], env=own_git_env(),
                   check=True)
    subprocess.run(["git", "-C", str(path), "checkout", "-q", commit], env=own_git_env(), check=True)
    cache = WINDOWS / ".cache"
    (path / "desktop" / "windows" / ".cache").mkdir(parents=True)
    for f in cache.iterdir() if cache.is_dir() else []:
        if f.is_file():
            shutil.copy2(f, path / "desktop" / "windows" / ".cache" / f.name)


@slow
def test_the_image_does_not_depend_on_the_build_folder():
    """The same commit built from two checkouts in different folders (one
    with a space), into image folders that differ (one with a space and an
    accent): the same per-file manifest and image hash (Review Focus 4), and
    the same installer bytes."""
    clones = [REPO, WORK / "hd-c 2"]
    outs = [WORK / "hd-o1", WORK / "hd-o 2\u00e9"]
    dist = WORK / "dist"          # where both builds put their installer
    assert not dist.exists() and not any(p.exists() for p in (*outs, clones[1])), "in the way: remove them"
    try:
        _second_clone(clones[1], _clean_head())
        manifests, records, installers = [], [], []
        for clone, out in zip(clones, outs):
            r = powershell("-File", clone / "desktop" / "windows" / "build_installer.ps1", "-TestBuild",
                           "-OutDir", out, env=own_git_env(), timeout=3600)
            assert r.returncode == 0, r.stdout[-4000:] + r.stderr
            manifests.append((out / "FoodOptimizer.files.txt").read_bytes())
            records.append((out / "FoodOptimizer.build.txt").read_text(encoding="utf-8"))
            [exe] = dist.glob("*-Windows-test.exe")
            installers.append(hashlib.sha256(exe.read_bytes()).hexdigest())
        if manifests[0] != manifests[1]:
            one, two = (set(m.decode("utf-8").splitlines()) for m in manifests)
            assert sorted(one ^ two) == []
        # Same lines in another order (or line ends) still differ: bytes count.
        assert manifests[0] == manifests[1]
        hashes = [next(line for line in rec.split("\n") if line.startswith("image_sha256="))
                  for rec in records]
        assert hashes[0] == hashes[1]
        assert installers[0] == installers[1]
    finally:
        for folder in (*outs, dist, clones[1]):
            _rmtree(folder)
