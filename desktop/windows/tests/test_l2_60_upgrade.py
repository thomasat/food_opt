"""Level 2: upgrades (spec section 5). An upgrade over an older install
whose lock differed leaves no stale package, module or shell DLL; a
reinstall of the same version repairs; a pre-warm that security software
blocks keeps the install, says so plainly and exits 9 (Review Focus 2)."""
import hashlib
import os
import pathlib
import subprocess
import sys

import pytest

import l2_support as l2

pytestmark = pytest.mark.usefixtures("clean_machine")

STALE = ("python/Lib/site-packages/stale_pkg/__init__.py",
         "python/Lib/site-packages/stale_pkg-1.0.dist-info/METADATA",
         "app/old_module.py", "shared/launcher/old_helper.py", "stale_shell.dll")


def _variant(name):
    path = os.environ.get(f"FOODOPT_L2_{name.upper()}_INSTALLER")
    if not path:
        pytest.skip(f"set FOODOPT_L2_{name.upper()}_INSTALLER: make_test_installer.ps1 -Variant {name}, "
                    f"then run_level2.ps1 -{name.capitalize()}Installer <exe>")
    return pathlib.Path(path)


@pytest.fixture
def cleanup(l2_home):
    yield
    entry = l2.uninstall_entry("user")
    if entry and entry.get("InstallLocation"):
        l2.kill_under(entry["InstallLocation"])
        l2.uninstall(entry["InstallLocation"], "user", env=l2_home.env)


def _image_sha(root) -> str:
    return hashlib.sha256(l2.manifest(root).encode("utf-8")).hexdigest()


def _run(root, *args, home, timeout=1800, **kw):
    env = dict(os.environ)
    env["HOME"] = env["USERPROFILE"] = str(home)
    return subprocess.run([str(root / "python" / "python.exe"), "-I", "-X", "utf8", "-B", *map(str, args)],
                          env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=timeout, creationflags=l2.CREATE_NO_WINDOW, **kw)


def test_upgrade_after_a_lock_change_leaves_nothing_stale(installer, dist_record, l2_home, cleanup, tmp_path):
    old = _variant("old")
    root = l2_home.root
    first = l2.install(old, "user", dir=root, env=l2_home.env)
    assert first.rc == 0, first.log[-3000:]
    assert l2.uninstall_entry("user")["DisplayVersion"] == "0.0.1"
    for rel in STALE:
        assert (root / rel).is_file(), rel
    second = l2.install(installer, "user", dir=root, env=l2_home.env)
    assert second.rc == 0, second.log[-3000:]
    for rel in STALE:
        assert not (root / rel).exists(), rel
    assert l2.uninstall_entry("user")["DisplayVersion"] == l2.VERSION
    assert _image_sha(root) == dist_record["image_sha256"]
    listed = _run(root, "-c", "import importlib.metadata as m; print(sorted(d.metadata['Name'].lower() "
                  "for d in m.distributions()))", home=tmp_path)
    assert listed.returncode == 0, listed.stderr
    assert "stale-pkg" not in listed.stdout
    checks = _run(root, l2.REPO / "desktop" / "shared" / "packaged_app_checks.py", "--resources", root / "app",
                  "--projects", tmp_path / "FoodOptimizer", home=tmp_path)
    assert checks.returncode == 0, (checks.stdout + checks.stderr)[-6000:]
    # A launcher start with the lifeline is healthy, then exits 0 when stdin closes.
    state, projects = tmp_path / "state", tmp_path / "projects"
    launcher = subprocess.Popen(
        [str(root / "python" / "python.exe"), "-I", "-X", "utf8", "-B", "-m", "launcher",
         "--state-dir", str(state), "--projects-dir", str(projects), "--app", str(root / "app" / "app.py"),
         "--python", str(root / "python" / "python.exe"), "--stdin-lifeline"],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env={**os.environ, "HOME": str(tmp_path), "USERPROFILE": str(tmp_path)},
        creationflags=l2.CREATE_NO_WINDOW)
    try:
        port_file = state / "server.port"
        assert l2.wait_for(lambda: port_file.exists() and _healthy(state), l2.FIRST_OPEN_LIMIT_S), \
            (state / "launcher.log").read_text(encoding="utf-8", errors="replace")[-3000:] \
            if (state / "launcher.log").exists() else "no launcher.log"
    finally:
        launcher.stdin.close()
    assert launcher.wait(60) == 0
    assert l2.wait_for(lambda: not l2.processes_under(root / "python"), 10)


def _healthy(state) -> bool:
    import http.client
    try:
        port = int((state / "server.port").read_text(encoding="ascii").split()[0])
        base = (state / "server.base").read_text(encoding="ascii").strip()
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        conn.request("GET", f"/{base}/_stcore/health", headers={"Host": f"127.0.0.1:{port}"})
        return conn.getresponse().status == 200
    except (OSError, ValueError, IndexError):
        return False


def test_reinstall_of_the_same_version_repairs(installer, dist_record, l2_home, cleanup):
    root = l2_home.root
    assert l2.install(installer, "user", dir=root, env=l2_home.env).rc == 0
    (root / "app" / "app.py").unlink()
    again = l2.install(installer, "user", dir=root, env=l2_home.env)
    assert again.rc == 0, again.log[-3000:]
    assert _image_sha(root) == dist_record["image_sha256"]


def test_a_failed_prewarm_says_so_and_exits_9(l2_home, cleanup):
    broken = _variant("broken")
    result = l2.install(broken, "user", dir=l2_home.root, env=l2_home.env)
    assert result.rc == l2.EXIT_CODES["prewarm_failed"], result.log[-3000:]
    assert "prewarm rc=1" in result.log
    assert "blocked by test" in result.log
    assert l2.uninstall_entry("user") is not None
