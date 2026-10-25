"""Level 2: the launcher behaviour suite and the packaged-app checks, run
against the installed app's own Python (spec Testing, Level 2: launcher end
to end, orphan replacement, idle and stdin-EOF shutdown, backups, packaged
checks)."""
import importlib.util
import os
import re
import subprocess
import sys

import l2_support as l2

REQUIRED_PASSES = (
    "test_unsupported_machine_exits_2_and_creates_nothing",
    "test_healthy_launch_writes_the_port_file_and_clears_status",
    "test_a_second_launch_replaces_the_orphaned_server",
    "test_an_orphan_that_is_not_our_server_is_left_running",
    "test_a_launch_in_progress_makes_the_next_exit_0",
    "test_backups_on_launch",
    "test_idle_shutdown",
    "test_stdin_eof_shuts_down",
    "test_without_a_base_the_port_is_random_and_not_8501",
    "test_the_base_path_file_and_the_server_agree",
    "test_real_streamlit_answers_only_on_its_base_path_and_host",
    "test_real_streamlit_shuts_down_on_stdin_eof",
)


def _sentinels():
    spec = importlib.util.spec_from_file_location(
        "packaged_app_checks", l2.REPO / "desktop" / "shared" / "packaged_app_checks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return [s for values in module.SENTINELS.values() for s in values]


def test_the_launcher_suite_passes_on_the_installed_app(per_user, tmp_path):
    env = dict(os.environ)
    env.update({"FOODOPT_INSTALLED_APP": str(per_user.root), "FOODOPT_SLOW_TESTS": "1"})
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", "tests/desktop_launcher/test_launcher_behaviour.py",
                            "tests/desktop_launcher/test_launcher_streamlit.py", "-rA", "-q", "-p",
                            "no:cacheprovider", "--basetemp", str(tmp_path / "bt")],
                           cwd=l2.REPO, env=env, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=3600)
    finally:
        # The suite's world fixture ends a launch it still holds with
        # TerminateProcess, which on Windows leaves that launch's real
        # Streamlit running (the shell's job object prevents this for
        # users). Count what it left, then end it, so the install stays
        # unused for the modules after this one.
        left = l2.processes_under(per_user.root / "python")
        l2.report("launcher_suite_leftover_pythons", len(left))
        l2.kill_under(per_user.root / "python")
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-6000:]
    passed = re.findall(r"^PASSED \S+::(\w+)", out, re.M)
    missing = [name for name in REQUIRED_PASSES if name not in passed]
    assert missing == [], out[-6000:]


def test_packaged_app_checks_pass_with_the_installed_python(per_user, tmp_path):
    root = per_user.root
    env = dict(os.environ)
    env["HOME"] = env["USERPROFILE"] = str(tmp_path)
    r = subprocess.run([str(root / "python" / "python.exe"), "-I", "-X", "utf8", "-B",
                        str(l2.REPO / "desktop" / "shared" / "packaged_app_checks.py"),
                        "--resources", str(root / "app"), "--projects", str(tmp_path / "FoodOptimizer")],
                       env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       timeout=1800, creationflags=l2.CREATE_NO_WINDOW)
    assert r.returncode == 0, (r.stdout + r.stderr)[-6000:]
    for sentinel in _sentinels():
        assert sentinel in r.stdout, sentinel
