"""Level 2: install scopes (spec section 5): per-user by default, all-users
for IT and Intune (SYSTEM), the two scopes never coexist, a running app
holds an install or uninstall (Review Focus 1 and 4), and an install folder
over the 80-character budget is refused (Review Focus 5). Every test starts
from a clean machine and uninstalls what it installed."""
import os
import pathlib

import pytest

import l2_support as l2

pytestmark = pytest.mark.usefixtures("clean_machine")

NAVIGATE = r"app\.navigate http://127\.0\.0\.1:\d+/[A-Za-z0-9_-]{32}/"
PROGRAM_FILES = pathlib.Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Food Optimizer"
DEFAULT_USER_ROOT = pathlib.Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Food Optimizer"


def _remove(scope):
    entry = l2.uninstall_entry(scope)
    if entry and entry.get("InstallLocation"):
        l2.kill_under(entry["InstallLocation"])
        l2.uninstall(entry["InstallLocation"], "user" if scope == "user" else "system")


@pytest.fixture
def cleanup():
    yield
    _remove("machine")
    _remove("user")


def _loaded_shell(root, home):
    job = l2.start_shell(root, home)
    assert l2.wait_log(home.state / "shell.log", NAVIGATE, l2.FIRST_OPEN_LIMIT_S), \
        (home.state / "shell.log").read_text(encoding="utf-8", errors="replace")[-3000:]
    return job


def _close(job, root):
    for hwnd in l2.top_windows(job.pid):
        l2.close_window(hwnd)
    if not l2.wait_for(lambda: not job.alive(), 10):
        l2.kill_under(root)
    l2.wait_for(lambda: not l2.processes_under(root), 10)


def test_default_per_user_folder(installer, l2_home, cleanup):
    result = l2.install(installer, "user", env=l2_home.env)
    assert result.rc == 0, result.log[-3000:]
    assert l2.uninstall_entry("user")["InstallLocation"] == str(DEFAULT_USER_ROOT) + "\\"
    assert (DEFAULT_USER_ROOT / "FoodOptimizer.exe").is_file()


def test_all_users_install_as_admin(installer, l2_home, cleanup):
    before = sorted(l2.quick_access())
    result = l2.install(installer, "admin", all_users=True, extra=["/NODESKTOPSHORTCUT"], env=l2_home.env)
    assert result.rc == 0, result.log[-3000:]
    assert l2.uninstall_entry("machine")["InstallLocation"] == str(PROGRAM_FILES) + "\\"
    assert l2.uninstall_entry("user") is None
    common = l2.shortcut_dirs("machine")
    assert (common["programs"] / "Food Optimizer.lnk").is_file()
    assert not (common["desktop"] / "Food Optimizer.lnk").exists()
    assert sorted(l2.quick_access()) == before
    assert not l2_home.projects.exists()


def test_system_without_allusers_is_refused(installer, cleanup):
    result = l2.install(installer, "system")
    assert result.rc == l2.EXIT_CODES["refused"], result.log[-3000:]
    assert "refused: system_needs_allusers" in result.log
    assert l2.uninstall_entry("machine") is None and l2.uninstall_entry("user") is None
    assert not pathlib.Path(r"C:\Windows\System32\config\systemprofile\AppData\Local\Programs\Food Optimizer").exists()


def test_intune_install_and_uninstall_as_system(installer, l2_home, cleanup):
    result = l2.install(installer, "system", all_users=True)
    assert result.rc == 0, result.log[-3000:]
    assert l2.uninstall_entry("machine")["DisplayVersion"] == l2.VERSION
    l2_home.state.mkdir(parents=True)
    (l2_home.state / "marker").write_bytes(b"user state")
    assert l2.uninstall(PROGRAM_FILES, "system") == 0
    assert l2.wait_for(lambda: not PROGRAM_FILES.exists(), 30)
    assert l2.uninstall_entry("machine") is None
    # SYSTEM cannot clean a user's state folder (the IT sheet says so).
    assert (l2_home.state / "marker").read_bytes() == b"user state"


def test_per_user_install_refuses_when_it_installed_it(installer, l2_home, cleanup):
    assert l2.install(installer, "system", all_users=True).rc == 0
    result = l2.install(installer, "user", dir=l2_home.root, env=l2_home.env)
    assert result.rc == l2.EXIT_CODES["refused"], result.log[-3000:]
    assert "refused: installed_by_it" in result.log
    assert l2.uninstall_entry("user") is None
    assert not l2_home.root.exists()


def test_all_users_install_replaces_a_per_user_copy(installer, l2_home, cleanup):
    assert l2.install(installer, "user", env=l2_home.env).rc == 0
    assert (DEFAULT_USER_ROOT / "FoodOptimizer.exe").is_file()
    l2_home.projects.mkdir(parents=True, exist_ok=True)
    keep = l2_home.projects / "Keep me.pkl"
    keep.write_bytes(b"\x80\x04keep")
    dirs = l2.shortcut_dirs("user")
    result = l2.install(installer, "system", all_users=True)
    assert result.rc == 0, result.log[-3000:]
    assert not DEFAULT_USER_ROOT.exists()
    assert l2.uninstall_entry("user") is None
    assert not (dirs["programs"] / "Food Optimizer.lnk").exists()
    assert not (dirs["programs"] / "Food Optimizer - Start Here.lnk").exists()
    assert not (dirs["desktop"] / "Food Optimizer.lnk").exists()
    assert l2.uninstall_entry("machine") is not None
    assert keep.read_bytes() == b"\x80\x04keep"


def _custom_dir_copy(installer, home):
    """A per-user copy installed with /DIR beside the user folder, and a
    project the IT install must keep."""
    folder = home.base / "Custom FO"
    result = l2.install(installer, "user", dir=folder, env=home.env)
    assert result.rc == 0, result.log[-3000:]
    assert l2.uninstall_entry("user")["InstallLocation"] == str(folder) + "\\"
    home.projects.mkdir(parents=True, exist_ok=True)
    keep = home.projects / "Keep me.pkl"
    keep.write_bytes(b"\x80\x04keep")
    return folder, keep


def test_all_users_install_replaces_a_per_user_dir_copy(installer, l2_home, cleanup):
    folder, keep = _custom_dir_copy(installer, l2_home)
    result = l2.install(installer, "system", all_users=True)
    assert result.rc == 0, result.log[-3000:]
    assert not folder.exists(), result.log[-3000:]
    assert f"replaced per-user copy: deleted {folder}" in result.log
    assert l2.uninstall_entry("user") is None
    assert keep.read_bytes() == b"\x80\x04keep"


def test_an_it_install_leaves_a_folder_that_is_no_longer_a_copy(installer, l2_home, cleanup):
    """Review Focus 3: the user's /DIR folder no longer holds Food Optimizer
    (its files were moved out and the user's own put in): the IT install
    leaves the folder and what is in it."""
    folder, keep = _custom_dir_copy(installer, l2_home)
    (folder / "FoodOptimizer.exe").unlink()
    (folder / "unins000.dat").unlink()
    mine = folder / "Mine.txt"
    mine.write_bytes(b"mine")
    result = l2.install(installer, "system", all_users=True)
    assert result.rc == 0, result.log[-3000:]
    assert mine.read_bytes() == b"mine", result.log[-3000:]
    assert f"replaced per-user copy: deleted {folder}" not in result.log
    assert keep.read_bytes() == b"\x80\x04keep"


_SHELL_FOLDERS = r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders"


def test_all_users_install_finds_a_redirected_start_menu(installer, l2_home, cleanup):
    """The user's Start menu lives elsewhere (folder redirection): the IT
    install deletes the per-user shortcuts where that user's Start menu is."""
    import winreg
    programs = l2_home.base / "Start Menu" / "Programs"
    programs.mkdir(parents=True)
    key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, _SHELL_FOLDERS, 0,
                         winreg.KEY_READ | winreg.KEY_SET_VALUE)
    with key:
        try:
            before = winreg.QueryValueEx(key, "Programs")
        except FileNotFoundError:
            before = None
        winreg.SetValueEx(key, "Programs", 0, winreg.REG_EXPAND_SZ, str(programs))
        try:
            assert l2.install(installer, "user", env=l2_home.env).rc == 0
            assert (programs / "Food Optimizer.lnk").is_file()
            result = l2.install(installer, "system", all_users=True)
            assert result.rc == 0, result.log[-3000:]
            for name in ("Food Optimizer.lnk", "Food Optimizer - Start Here.lnk"):
                assert not (programs / name).exists(), result.log[-3000:]
        finally:
            if before is None:
                winreg.DeleteValue(key, "Programs")
            else:
                winreg.SetValueEx(key, "Programs", 0, before[1], before[0])


def test_it_install_waits_for_a_running_copy(installer, l2_home, cleanup):
    assert l2.install(installer, "user", dir=l2_home.root, env=l2_home.env).rc == 0
    before = l2.manifest(l2_home.root)
    job = _loaded_shell(l2_home.root, l2_home)
    try:
        result = l2.install(installer, "system", all_users=True)
        assert job.alive()
        assert l2.manifest(l2_home.root) == before
        assert l2.uninstall_entry("machine") is None
        l2.report("app_running_rc_system", result.rc)
        assert result.rc == l2.EXIT_CODES["app_running"], (result.rc, result.log[-3000:])
    finally:
        _close(job, l2_home.root)


def test_silent_install_while_the_app_runs(installer, l2_home, cleanup):
    assert l2.install(installer, "user", dir=l2_home.root, env=l2_home.env).rc == 0
    before = l2.manifest(l2_home.root)
    job = _loaded_shell(l2_home.root, l2_home)
    try:
        result = l2.install(installer, "user", dir=l2_home.root, env=l2_home.env)
        assert job.alive()
        assert l2.manifest(l2_home.root) == before
        l2.report("app_running_rc_user", result.rc)
        assert result.rc == l2.EXIT_CODES["app_running"], (result.rc, result.log[-3000:])
    finally:
        _close(job, l2_home.root)
    again = l2.install(installer, "user", dir=l2_home.root, env=l2_home.env)
    assert again.rc == 0, again.log[-3000:]


def test_silent_uninstall_while_the_app_runs(installer, l2_home, cleanup):
    assert l2.install(installer, "user", dir=l2_home.root, env=l2_home.env).rc == 0
    before = l2.manifest(l2_home.root)
    job = _loaded_shell(l2_home.root, l2_home)
    try:
        rc = l2.uninstall(l2_home.root, "user", env=l2_home.env)
        l2.report("app_running_rc_uninstall", rc)
        assert rc != 0
        assert job.alive()
        assert l2.manifest(l2_home.root) == before
        assert l2.uninstall_entry("user") is not None
    finally:
        _close(job, l2_home.root)


def test_a_long_install_folder_is_refused(installer, cleanup):
    folder = l2.pathlib.Path(r"C:\FOL2") / ("x" * 75)
    assert len(str(folder) + "\\") > 80
    result = l2.install(installer, "user", dir=folder)
    assert result.rc == l2.EXIT_CODES["refused"], result.log[-3000:]
    assert "refused: folder_too_long" in result.log
    assert not folder.exists()


def test_a_folder_that_is_not_ours_is_refused(installer, l2_home, cleanup):
    """/DIR at a folder holding other files (final review): [InstallDelete]
    would delete its python, app, shared and *.dll, so it is refused and
    every file stays."""
    folder = l2_home.home / "Tools"
    planted = {folder / "python" / "keep.txt": b"keep", folder / "tool.dll": b"dll", folder / "notes.txt": b"n"}
    for path, data in planted.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    result = l2.install(installer, "user", dir=folder, env=l2_home.env)
    assert result.rc == l2.EXIT_CODES["refused"], result.log[-3000:]
    assert "refused: folder_not_ours" in result.log
    for path, data in planted.items():
        assert path.read_bytes() == data, path
    assert l2.uninstall_entry("user") is None
