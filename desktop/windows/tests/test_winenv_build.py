"""A build that fails part-way stops with exit 1, says `build failed:`, and
leaves no build record (Review Focus 4: uv failing mid-build)."""
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
