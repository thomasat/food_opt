"""Level 2: a per-user install with no FOODOPT_* variables uses the user's
own default folders (spec section 5): projects in %USERPROFILE%\\FoodOptimizer,
made by the installer and pinned to Quick access by the app at its first
launch (D2); state in %LOCALAPPDATA%\\FoodOptimizer. A silent uninstall
deletes the state, keeps the projects and unpins them.

It runs in the signed-in user's own profile, so it needs both folders
absent (move them aside first) and removes what it made."""
import os
import pathlib
import shutil
import subprocess

import pytest

import l2_support as l2

pytestmark = pytest.mark.usefixtures("clean_machine")

NAVIGATE = l2.NAVIGATE
LOCAL = pathlib.Path(os.environ.get("LOCALAPPDATA", ""))
STATE = LOCAL / "FoodOptimizer"
PROJECTS = pathlib.Path(os.environ.get("USERPROFILE", "")) / "FoodOptimizer"
ROOT = LOCAL / "Programs" / "Food Optimizer"


class RealHome:
    """L2Home's shape for the signed-in user's own profile: no FOODOPT_*."""

    def __init__(self):
        self.root, self.state, self.projects = ROOT, STATE, PROJECTS
        self.env = {}


def _persistent_foodopt() -> list:
    """FOODOPT_* in the user's or the machine's stored environment, which
    the installer and the shell (run as scheduled tasks) would inherit."""
    import winreg
    found = []
    for root, sub in ((winreg.HKEY_CURRENT_USER, "Environment"),
                      (winreg.HKEY_LOCAL_MACHINE, r"System\CurrentControlSet\Control\Session Manager\Environment")):
        with winreg.OpenKey(root, sub) as key:
            for i in range(winreg.QueryInfoKey(key)[1]):
                name = winreg.EnumValue(key, i)[0]
                if name.upper().startswith("FOODOPT_"):
                    found.append(f"{sub}\\{name}")
    return found


def _unpin(folder):
    path = str(folder).replace("'", "''")
    subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                    f"$f = (New-Object -ComObject Shell.Application).Namespace('{path}'); "
                    "if ($f) { $f.Self.InvokeVerb('unpinfromhome') }"],
                   capture_output=True, timeout=120, creationflags=l2.CREATE_NO_WINDOW)


def _close(job, root):
    for hwnd in l2.top_windows(job.pid):
        l2.close_window(hwnd)
    if not l2.wait_for(lambda: not job.alive(), 10):
        l2.kill_under(root)
    l2.wait_for(lambda: not l2.processes_under(root), 10)


def test_a_default_install_uses_the_users_own_folders(installer):
    for folder in (STATE, PROJECTS):
        assert not folder.exists(), f"{folder} exists: move it aside to run this test"
    assert _persistent_foodopt() == []
    try:
        result = l2.install(installer, "user")
        assert result.rc == 0, result.log[-3000:]
        assert l2.uninstall_entry("user")["InstallLocation"] == str(ROOT) + "\\"
        assert PROJECTS.is_dir()
        assert not l2.in_quick_access(PROJECTS)
        home = RealHome()
        job = l2.start_shell(ROOT, home)
        try:
            assert l2.wait_log(STATE / "shell.log", NAVIGATE, l2.FIRST_OPEN_LIMIT_S), \
                (STATE / "shell.log").read_text(encoding="utf-8", errors="replace")[-3000:] \
                if (STATE / "shell.log").exists() else "no shell.log"
            assert l2.wait_log(STATE / "shell.log", r"quickaccess\.pinned", 30), \
                (STATE / "shell.log").read_text(encoding="utf-8", errors="replace")[-3000:]
            assert l2.in_quick_access(PROJECTS)
        finally:
            _close(job, ROOT)
        assert '"quickAccessPinned": true' in l2.settings_text(home)
        keep = PROJECTS / "Keep me.pkl"
        keep.write_bytes(b"\x80\x04keep")
        assert l2.uninstall(ROOT, "user") == 0
        assert not ROOT.exists()
        assert l2.wait_for(lambda: not STATE.exists(), 10), l2.last_uninstall_log()[-3000:]
        assert keep.read_bytes() == b"\x80\x04keep"
        assert not l2.in_quick_access(PROJECTS)
    finally:
        entry = l2.uninstall_entry("user")
        if entry and entry.get("InstallLocation"):
            l2.kill_under(entry["InstallLocation"])
            l2.uninstall(entry["InstallLocation"], "user")
        l2.kill_under(ROOT)
        if PROJECTS.exists() and l2.in_quick_access(PROJECTS):
            _unpin(PROJECTS)
        for folder in (STATE, PROJECTS):
            shutil.rmtree(folder, ignore_errors=True)
