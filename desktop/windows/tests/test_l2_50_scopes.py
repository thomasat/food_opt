"""Level 2: install scopes (spec section 5): per-user by default, all-users
for IT and Intune (SYSTEM). The IT install never changes a per-user copy;
that user's next launch of the IT copy runs the copy's own uninstaller
(D1). A running app holds an install or uninstall (Review Focus 1 and 4),
and an install folder over the 80-character budget is refused (Review
Focus 5). Every test starts from a clean machine and uninstalls what it
installed."""
import os
import pathlib
import subprocess

import pytest

import l2_support as l2

pytestmark = pytest.mark.usefixtures("clean_machine")

NAVIGATE = r"app\.navigate http://127\.0\.0\.1:\d+/[A-Za-z0-9_-]{32}/"
PROGRAM_FILES = pathlib.Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Food Optimizer"
DEFAULT_USER_ROOT = pathlib.Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Food Optimizer"


def _remove(scope, env=None):
    entry = l2.uninstall_entry(scope)
    if entry and entry.get("InstallLocation"):
        l2.kill_under(entry["InstallLocation"])
        if scope == "user":
            l2.uninstall(entry["InstallLocation"], "user", env=env)
        else:
            l2.uninstall(entry["InstallLocation"], "system")


@pytest.fixture
def cleanup(l2_home):
    """The per-user copy first, with the test's folders: its uninstaller
    then deletes (or keeps) the test's state folder and unpins the test's
    projects folder, never the signed-in user's own."""
    yield
    _remove("user", l2_home.env)
    _remove("machine")


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
    # SYSTEM cannot clean a user's state folder (desktop/windows/README.md, "For IT teams (reference)").
    assert (l2_home.state / "marker").read_bytes() == b"user state"


def test_per_user_install_refuses_when_it_installed_it(installer, l2_home, cleanup):
    assert l2.install(installer, "system", all_users=True).rc == 0
    result = l2.install(installer, "user", dir=l2_home.root, env=l2_home.env)
    assert result.rc == l2.EXIT_CODES["refused"], result.log[-3000:]
    assert "refused: installed_by_it" in result.log
    assert l2.uninstall_entry("user") is None
    assert not l2_home.root.exists()


# ---------- D1: the IT install leaves per-user copies alone ----------

PERCOPY = r"percopy\.(uninstall|none)( .*)?"


def _shell_log(home) -> str:
    return l2._text(home.state / "shell.log")[-3000:]


def _default_per_user_copy(installer, home):
    """A per-user copy in the default folder, its shortcuts, and a project
    the IT install and the cleanup must keep."""
    result = l2.install(installer, "user", env=home.env)
    assert result.rc == 0, result.log[-3000:]
    assert (DEFAULT_USER_ROOT / "FoodOptimizer.exe").is_file()
    home.projects.mkdir(parents=True, exist_ok=True)
    keep = home.projects / "Keep me.pkl"
    keep.write_bytes(b"\x80\x04keep")
    dirs = l2.shortcut_dirs("user")
    links = [dirs["programs"] / "Food Optimizer.lnk", dirs["programs"] / "Food Optimizer - Start Here.lnk",
             dirs["desktop"] / "Food Optimizer.lnk"]
    for lnk in links:
        assert lnk.is_file(), lnk
    return keep, links


def test_all_users_install_leaves_a_per_user_copy_untouched(installer, l2_home, cleanup):
    """D1 (owner): the elevated installer never deletes in a per-user
    location. The user's copy, its Apps entry and its shortcuts are
    exactly as they were."""
    keep, links = _default_per_user_copy(installer, l2_home)
    files = l2.manifest(DEFAULT_USER_ROOT, skip=())
    entry = l2.uninstall_entry("user")
    shortcuts = {lnk: lnk.read_bytes() for lnk in links}
    result = l2.install(installer, "system", all_users=True)
    assert result.rc == 0, result.log[-3000:]
    assert l2.uninstall_entry("machine") is not None
    assert l2.manifest(DEFAULT_USER_ROOT, skip=()) == files, result.log[-3000:]
    assert l2.uninstall_entry("user") == entry
    for lnk, data in shortcuts.items():
        assert lnk.read_bytes() == data, lnk
    assert keep.read_bytes() == b"\x80\x04keep"
    assert "per-user copy" not in result.log


def test_that_users_next_launch_removes_it(installer, l2_home, cleanup):
    """D1: when that user next opens the IT copy, it runs their per-user
    copy's own uninstaller, silently and as them. The copy, its Apps entry
    and its shortcuts go; their projects and the state folder the IT copy
    shares (settings.json) stay; the IT copy opens as usual."""
    keep, links = _default_per_user_copy(installer, l2_home)
    result = l2.install(installer, "system", all_users=True)
    assert result.rc == 0, result.log[-3000:]
    job = l2.start_shell(PROGRAM_FILES, l2_home)
    try:
        m = l2.wait_log(l2_home.state / "shell.log", PERCOPY, 60)
        assert m and m.group(1) == "uninstall", _shell_log(l2_home)
        assert m.group(2).strip().rstrip("\\").lower() == str(DEFAULT_USER_ROOT).lower(), m.group(0)
        assert l2.wait_for(lambda: not DEFAULT_USER_ROOT.exists() and l2.uninstall_entry("user") is None, 120), \
            _shell_log(l2_home)
        assert l2.wait_log(l2_home.state / "shell.log", r"percopy\.exit 0\b", 120), _shell_log(l2_home)
        for lnk in links:
            assert l2.wait_for(lambda: not lnk.exists(), 30), lnk
        assert l2.wait_log(l2_home.state / "shell.log", NAVIGATE, l2.FIRST_OPEN_LIMIT_S), _shell_log(l2_home)
        assert keep.read_bytes() == b"\x80\x04keep"
        assert (l2_home.state / "settings.json").is_file(), _shell_log(l2_home)
        assert (l2_home.state / "shell.log").is_file()
        assert l2.uninstall_entry("machine") is not None
        assert (PROGRAM_FILES / "FoodOptimizer.exe").is_file()
    finally:
        _close(job, PROGRAM_FILES)


def _canary(home):
    """A folder of the user's own that looks like a copy of ours (our
    files' bytes) beside a file of theirs."""
    canary = home.base / "Canary"
    canary.mkdir()
    (canary / "Mine.txt").write_bytes(b"mine")
    for name in ("FoodOptimizer.exe", "unins000.exe", "unins000.dat"):
        (canary / name).write_bytes((PROGRAM_FILES / name).read_bytes())
    return canary


def _planted_link(installer, home, name):
    """Once IT has installed, the user (no admin needed) makes a junction
    named `name` to a canary folder and writes their own Apps entry naming
    it as InstallLocation. The next IT install (a repair or an upgrade;
    /NOREDIRECTIONGUARD stands for a PC without Windows' RedirectionGuard)
    must leave the canary byte for byte, and the user's next launch of the
    IT copy must not run anything from it: a copy reached through a link,
    or through a name Win32 would trim, is never chosen."""
    import sys
    import winreg
    first = l2.install(installer, "system", all_users=True)
    assert first.rc == 0, first.log[-3000:]
    canary = _canary(home)
    before = l2.manifest(canary, skip=())
    junction = str(home.base) + "\\" + name
    raw = "\\\\?\\" + junction          # keeps a trailing space
    make = "import _winapi, sys; _winapi.CreateJunction(sys.argv[1], sys.argv[2])"
    assert l2.run_as("user", [sys.executable, "-I", "-c", make, str(canary), raw]) == 0
    assert os.path.lexists(raw) and name in os.listdir(home.base)
    job = None
    try:
        with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, l2.UNINSTALL_SUBKEY, 0, winreg.KEY_SET_VALUE) as key:
            winreg.SetValueEx(key, "InstallLocation", 0, winreg.REG_SZ, junction + "\\")
        result = l2.install(installer, "system", all_users=True, extra=["/NOREDIRECTIONGUARD"])
        assert result.rc == 0, result.log[-3000:]
        assert l2.manifest(canary, skip=()) == before, result.log[-3000:]
        assert (PROGRAM_FILES / "FoodOptimizer.exe").is_file() and (PROGRAM_FILES / "unins000.dat").is_file()
        assert "per-user copy" not in result.log
        job = l2.start_shell(PROGRAM_FILES, home)
        m = l2.wait_log(home.state / "shell.log", PERCOPY, 60)
        assert m and m.group(1) == "none", _shell_log(home)
        assert l2.manifest(canary, skip=()) == before
        assert l2.uninstall_entry("user") is not None
    finally:
        if job is not None:
            _close(job, PROGRAM_FILES)
        # rmdir without /s removes the junction only, never the folder it points at.
        if os.path.lexists(raw):
            subprocess.run(["cmd", "/c", "rmdir", raw], check=False)
        try:
            winreg.DeleteKey(winreg.HKEY_CURRENT_USER, l2.UNINSTALL_SUBKEY)
        except FileNotFoundError:
            pass


def test_an_elevated_launch_never_runs_a_per_user_uninstaller(installer, l2_home, cleanup):
    """Fix round 1: the IT copy opened with admin rights (here from the
    elevated runner) does not look for a per-user copy: that copy's
    unins000.exe is the user's to replace, and would run as admin."""
    keep, links = _default_per_user_copy(installer, l2_home)
    result = l2.install(installer, "system", all_users=True)
    assert result.rc == 0, result.log[-3000:]
    files = l2.manifest(DEFAULT_USER_ROOT, skip=())
    entry = l2.uninstall_entry("user")
    job = l2.run_as("admin", [str(PROGRAM_FILES / "FoodOptimizer.exe")], env=l2_home.env, wait=False)
    try:
        assert job.elevated
        m = l2.wait_log(l2_home.state / "shell.log", r"percopy\.(skipped|uninstall|none)( .*)?", 60)
        assert m and m.group(0).strip() == "percopy.skipped elevated", _shell_log(l2_home)
        l2.wait_for(lambda: False, 10)          # an uninstaller, had one started, would be at work by now
        assert l2.manifest(DEFAULT_USER_ROOT, skip=()) == files
        assert l2.uninstall_entry("user") == entry
        for lnk in links:
            assert lnk.is_file(), lnk
    finally:
        _close(job, PROGRAM_FILES)


def test_a_planted_junction_is_never_touched_by_the_it_install(installer, l2_home, cleanup):
    _planted_link(installer, l2_home, "Link")


def test_an_odd_named_copy_is_never_touched_by_the_it_install(installer, l2_home, cleanup):
    _planted_link(installer, l2_home, "Link ")


def test_it_install_waits_for_a_running_copy(installer, l2_home, cleanup):
    """An IT upgrade waits while the all-users app runs (its Global name,
    seen from SYSTEM's session). A per-user copy's app is seen only in its
    own session (test_l2_55_two_users)."""
    first = l2.install(installer, "system", all_users=True)
    assert first.rc == 0, first.log[-3000:]
    before = l2.manifest(PROGRAM_FILES)
    job = _loaded_shell(PROGRAM_FILES, l2_home)
    try:
        result = l2.install(installer, "system", all_users=True)
        assert job.alive()
        assert l2.manifest(PROGRAM_FILES) == before
        l2.report("app_running_rc_system", result.rc)
        assert result.rc == l2.EXIT_CODES["app_running"], (result.rc, result.log[-3000:])
        assert "refused: app_running" in result.log
    finally:
        _close(job, PROGRAM_FILES)


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
        assert "refused: app_running" in result.log
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
        assert "refused: app_running" in l2.last_uninstall_log(), l2.last_uninstall_log()[-3000:]
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
