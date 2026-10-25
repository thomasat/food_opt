"""A build that fails part-way stops with exit 1, says `build failed:`, and
leaves no build record (Review Focus 4: uv failing mid-build). A long clone
path still builds, and the image is the same from any build folder."""
import os
import shutil
import subprocess

from conftest import REPO, WINDOWS, own_git_env, powershell, slow

# Beside the clone (C:\work on the build VM), with the plan's names.
WORK = REPO.parent


@slow
def test_a_failed_install_stops_the_build(short_tmp):
    out = short_tmp / "out dir"
    empty_cache = short_tmp / "uv cache"
    empty_cache.mkdir()
    env = own_git_env({"UV_OFFLINE": "1", "UV_CACHE_DIR": str(empty_cache)})
    r = powershell("-File", WINDOWS / "build_installer.ps1", "-TestBuild", "-OutDir", out,
                   env=env, timeout=1800)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "build failed: uv" in r.stdout + r.stderr
    assert not (out / "FoodOptimizer.build.txt").exists()
    assert not (out / "FoodOptimizer.files.txt").exists()
    assert "Built:" not in r.stdout


@slow
def test_an_out_dir_on_another_drive_builds(short_tmp):
    """The NuGet package is unpacked beside the image, so -OutDir on another
    drive (here a subst drive) gets past the Python step."""
    import string
    letter = next(c for c in reversed(string.ascii_uppercase[3:]) if not os.path.exists(f"{c}:\\"))
    subprocess.run(["subst", f"{letter}:", str(short_tmp)], check=True)
    try:
        empty_cache = short_tmp / "uv cache"
        empty_cache.mkdir()
        env = own_git_env({"UV_OFFLINE": "1", "UV_CACHE_DIR": str(empty_cache)})
        r = powershell("-File", WINDOWS / "build_installer.ps1", "-TestBuild", "-OutDir",
                       f"{letter}:\\out", env=env, timeout=1800)
        assert r.returncode == 1, r.stdout + r.stderr
        assert "build failed: uv" in r.stdout + r.stderr, r.stdout + r.stderr
        assert (short_tmp / "out" / "FoodOptimizer" / "python" / "python.exe").is_file()
    finally:
        subprocess.run(["subst", f"{letter}:", "/d"])


def _offline_env(short_tmp):
    """uv with no network and an empty cache: the build stops at its uv step."""
    empty_cache = short_tmp / "uv cache"
    empty_cache.mkdir()
    return own_git_env({"UV_OFFLINE": "1", "UV_CACHE_DIR": str(empty_cache)})


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


@slow
def test_the_image_does_not_depend_on_the_build_folder():
    """Two full test builds from one clone, into folders that differ (one with
    a space and an accent): the same per-file manifest (Review Focus 4)."""
    outs = [WORK / "hd-o1", WORK / "hd-o 2\u00e9"]
    dist = WORK / "dist"          # where both builds put their installer
    assert not dist.exists() and not any(o.exists() for o in outs), "in the way: remove them"
    try:
        manifests = []
        for out in outs:
            r = powershell("-File", WINDOWS / "build_installer.ps1", "-TestBuild", "-OutDir", out,
                           env=own_git_env(), timeout=3600)
            assert r.returncode == 0, r.stdout[-4000:] + r.stderr
            manifests.append((out / "FoodOptimizer.files.txt").read_bytes())
        if manifests[0] != manifests[1]:
            one, two = (set(m.decode("utf-8").splitlines()) for m in manifests)
            assert sorted(one ^ two) == []
    finally:
        for folder in (*outs, dist):
            shutil.rmtree(folder, ignore_errors=True)
