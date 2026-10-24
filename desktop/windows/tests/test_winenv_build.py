"""A build that fails part-way stops with exit 1, says `build failed:`, and
leaves no build record (Review Focus 4: uv failing mid-build)."""
import os

from conftest import WINDOWS, own_git_env, powershell, slow


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
    import subprocess
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
