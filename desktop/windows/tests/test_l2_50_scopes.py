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

NAVIGATE = l2.NAVIGATE
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


# Run as the limited user: each way of changing the IT folder, and what it
# raised; "regrant" and "reown" are icacls run by that user ("refused" when
# it exits non-zero).
_CHANGE_PROBE = r"""
import json, os, subprocess, sys
folder, out, which = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
def create():
    with open(os.path.join(folder, "x.txt"), "x") as f:
        f.write("x")
def create_in_app():
    with open(os.path.join(folder, "app", "x.py"), "x") as f:
        f.write("x")
def overwrite_exe():
    with open(os.path.join(folder, "FoodOptimizer.exe"), "r+b") as f:
        f.write(f.read(2))
def replace_exe():
    planted = os.path.join(os.path.dirname(out), "planted.exe")
    with open(planted, "wb") as f:
        f.write(b"MZ")
    os.replace(planted, os.path.join(folder, "FoodOptimizer.exe"))
def icacls(*args):
    r = subprocess.run(["icacls", folder, *args], capture_output=True, creationflags=0x08000000)
    if r.returncode:
        raise PermissionError(r.returncode)
def regrant():
    icacls("/grant", os.environ["USERNAME"] + ":(OI)(CI)F")
def reown():
    icacls("/setowner", os.environ["USERNAME"])
found = {}
for name in which:
    try:
        globals()[name]()
        found[name] = "changed"
    except OSError as e:
        found[name] = type(e).__name__
with open(out, "w", encoding="utf-8") as f:
    json.dump(found, f)
"""


def _changes(folder, home, which) -> dict:
    """What each change in `which` raised when the limited user tried it."""
    import json
    import sys
    out = home.home / "probe.json"
    out.unlink(missing_ok=True)
    assert l2.run_as("user", [sys.executable, "-I", "-c", _CHANGE_PROBE, str(folder), str(out), ",".join(which)]) == 0
    return json.loads(out.read_text(encoding="utf-8"))


def _owner(path) -> str:
    """The SID that owns path."""
    p = str(path).replace("'", "''")
    r = l2._powershell(f"(Get-Acl -LiteralPath '{p}').GetOwner([Security.Principal.SecurityIdentifier]).Value")
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def _sddl(path) -> str:
    """path's whole security descriptor (owner and rule), as SDDL."""
    p = str(path).replace("'", "''")
    r = l2._powershell(f"(Get-Acl -LiteralPath '{p}').Sddl")
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def _user_makes(folder, files=()):
    """The limited user makes `folder` (and `files` in it): theirs, as
    anything a user makes outside their profile is."""
    import sys
    make = ("import os, sys; d = sys.argv[1]; os.makedirs(d)\n"
            "for name in sys.argv[2:]:\n"
            "    open(os.path.join(d, name), 'wb').write(b'MZ planted')")
    assert l2.run_as("user", [sys.executable, "-I", "-c", make, str(folder), *files]) == 0
    assert folder.is_dir()


ADMINISTRATORS = "S-1-5-32-544"
ALL_CHANGES = ("create", "create_in_app", "overwrite_exe", "replace_exe", "regrant", "reown")


def test_an_it_dir_outside_program_files_is_read_only_for_users(installer, l2_home, cleanup):
    """M4: an IT install with /DIR outside Program Files gets Program Files'
    rule, not the drive's (C:\\ lets every signed-in user change a new
    folder): the folder is Administrators', and the limited user can
    neither add a file, replace the app's, grant themselves access nor
    take it; the app still opens for them. An IT upgrade into the same
    folder (its registered folder) keeps it so."""
    import shutil
    folder = l2_home.base / "IT FO"
    try:
        result = l2.install(installer, "system", all_users=True, dir=folder)
        assert result.rc == 0, result.log[-3000:]
        assert f"acl: locked {folder}" in result.log, result.log[-3000:]
        assert result.log.index(f"acl: locked {folder}") < result.log.index("Installing the file.")
        assert _owner(folder) == ADMINISTRATORS
        before = l2.manifest(folder, skip=())
        assert _changes(folder, l2_home, ALL_CHANGES) == {name: "PermissionError" for name in ALL_CHANGES}
        assert l2.manifest(folder, skip=()) == before
        assert _owner(folder) == ADMINISTRATORS
        _close(_loaded_shell(folder, l2_home), folder)
        again = l2.install(installer, "system", all_users=True, dir=folder)
        assert again.rc == 0, again.log[-3000:]
        assert f"acl: locked {folder}" in again.log, again.log[-3000:]
        assert _owner(folder) == ADMINISTRATORS
        assert _changes(folder, l2_home, ("create", "overwrite_exe")) == {
            "create": "PermissionError", "overwrite_exe": "PermissionError"}
    finally:
        # The folder is not under C:\FOL2's user grant any more: uninstall
        # it here, then delete what is left as admin.
        _remove("machine")
        shutil.rmtree(folder, ignore_errors=True)


@pytest.mark.parametrize("files", [(), ("FoodOptimizer.exe", "pyvenv.cfg")], ids=["empty", "filled"])
def test_a_folder_a_user_made_is_refused(installer, l2_home, cleanup, files):
    """Fix rounds 1-2 (M4): the user makes the /DIR folder first. Theirs,
    they could keep a handle that changes its rule after ours; filled, it
    could hold our exe's name (passing for an upgrade) and a pyvenv.cfg the
    pre-warm (python\\python.exe, as SYSTEM) would read. The IT install
    refuses before changing anything: no install, and the folder and its
    files keep their owner and rule."""
    import shutil
    folder = l2_home.base / "IT FO"
    try:
        _user_makes(folder, files)
        before = {p.name: _sddl(p) for p in (folder, *(folder / f for f in files))}
        result = l2.install(installer, "system", all_users=True, dir=folder)
        assert result.rc == l2.EXIT_CODES["refused"], result.log[-3000:]
        assert "refused: folder_not_ours" in result.log and "acl: locked" not in result.log, result.log[-3000:]
        assert ("acl: not empty" if files else "acl: not owned by Administrators or SYSTEM") in result.log
        assert l2.uninstall_entry("machine") is None
        assert sorted(p.name for p in folder.iterdir()) == sorted(files)
        assert {p.name: _sddl(p) for p in (folder, *(folder / f for f in files))} == before
        assert _owner(folder) != ADMINISTRATORS
    finally:
        _remove("machine")
        shutil.rmtree(folder, ignore_errors=True)


@pytest.mark.parametrize("where", ["folder", "above"])
def test_a_dir_through_a_link_is_refused(installer, l2_home, cleanup, where):
    """Fix round 2 (M4): a limited user can make a junction anywhere C:\\
    lets them make a folder. A /DIR that is one, or lies under one, is
    refused before anything is made or changed: the folder it points at
    keeps its files, owner and rule. (Linked itself, the target holds our
    exe's name, so the earlier not-ours check passes it for an upgrade.)"""
    import shutil
    import sys
    target = l2_home.base / "Target"
    mine = "FoodOptimizer.exe" if where == "folder" else "Mine.txt"
    _user_makes(target, (mine,))
    link = l2_home.base / "Link"
    make = "import _winapi, sys; _winapi.CreateJunction(sys.argv[1], sys.argv[2])"
    assert l2.run_as("user", [sys.executable, "-I", "-c", make, str(target), str(link)]) == 0
    assert os.path.lexists(link)
    folder = link if where == "folder" else link / "IT FO"
    before = (_sddl(target), _sddl(target / mine), sorted(p.name for p in target.iterdir()))
    try:
        result = l2.install(installer, "system", all_users=True, dir=folder)
        assert result.rc == l2.EXIT_CODES["refused"], result.log[-3000:]
        assert "refused: folder_not_ours" in result.log and f"acl: a link: {link}" in result.log, result.log[-3000:]
        assert "acl: locked" not in result.log
        assert l2.uninstall_entry("machine") is None
        assert (_sddl(target), _sddl(target / mine), sorted(p.name for p in target.iterdir())) == before
        assert (target / mine).read_bytes() == b"MZ planted"
    finally:
        _remove("machine")
        # rmdir without /s removes the junction only, never the folder it points at.
        if os.path.lexists(link):
            subprocess.run(["cmd", "/c", "rmdir", str(link)], check=False)
        shutil.rmtree(target, ignore_errors=True)


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


def _it_shells():
    return [pid for pid, path in l2.processes_under(PROGRAM_FILES)
            if pathlib.Path(path).name.lower() == "foodoptimizer.exe"]


def test_the_per_user_copy_opens_the_it_copy_instead(installer, l2_home, cleanup):
    """D1 hand-off (final review): the user's own shortcuts still open the
    per-user copy, which never cleans itself up. Opened once IT has
    installed, it starts the IT copy and exits 0 before it holds any name;
    the IT copy then runs the per-user copy's own uninstaller as at any
    launch. The copy, its Apps entry and its shortcuts go; projects stay;
    the IT window opens."""
    keep, links = _default_per_user_copy(installer, l2_home)
    result = l2.install(installer, "system", all_users=True)
    assert result.rc == 0, result.log[-3000:]
    log = l2_home.state / "shell.log"
    job = l2.start_shell(DEFAULT_USER_ROOT, l2_home)
    try:
        m = l2.wait_log(log, r"percopy\.handoff (.*)", 60)
        assert m, _shell_log(l2_home)
        assert m.group(1).strip().rstrip("\\").lower() == str(PROGRAM_FILES).lower(), m.group(0)
        assert l2.wait_for(lambda: job.poll() is not None, 30), _shell_log(l2_home)
        assert job.poll() == 0
        m = l2.wait_log(log, PERCOPY, 60)
        assert m and m.group(1) == "uninstall", _shell_log(l2_home)
        assert m.group(2).strip().rstrip("\\").lower() == str(DEFAULT_USER_ROOT).lower(), m.group(0)
        assert l2.wait_for(lambda: not DEFAULT_USER_ROOT.exists() and l2.uninstall_entry("user") is None, 120), \
            _shell_log(l2_home)
        for lnk in links:
            assert l2.wait_for(lambda: not lnk.exists(), 30), lnk
        assert l2.wait_log(log, NAVIGATE, l2.FIRST_OPEN_LIMIT_S), _shell_log(l2_home)
        assert _it_shells(), l2.processes()
        text = l2._text(log)
        # The per-user shell held nothing: the IT shell is the only one
        # that started (one shell.start, after the hand-off).
        assert "Running.PerUser" not in text, text[-3000:]
        assert text.count(" shell.start ") == 1 and text.index("percopy.handoff") < text.index(" shell.start "), \
            text[-3000:]
        assert keep.read_bytes() == b"\x80\x04keep"
        assert (PROGRAM_FILES / "FoodOptimizer.exe").is_file() and l2.uninstall_entry("machine") is not None
    finally:
        job.close()
        for pid in _it_shells():
            for hwnd in l2.top_windows(pid):
                l2.close_window(hwnd)
        if not l2.wait_for(lambda: not _it_shells(), 10):
            l2.kill_under(PROGRAM_FILES)
        l2.wait_for(lambda: not l2.processes_under(PROGRAM_FILES), 10)


def test_an_it_install_that_cannot_copy_its_marker_fails(installer, l2_home, cleanup):
    """Final review (ruling): the all-users marker is a [Files] entry, so
    an IT install that cannot copy it fails rather than leave an IT copy
    that would hold the per-user name and hand off. Here a folder named
    all-users (with a file in it) stands where the marker goes; Inno's file
    error is answered Abort under /SUPPRESSMSGBOXES. The same command, run
    again once the folder is gone, repairs the install (README: codes 4
    and 5)."""
    import shutil
    first = l2.install(installer, "system", all_users=True)
    assert first.rc == 0, first.log[-3000:]
    marker = PROGRAM_FILES / "all-users"
    assert marker.is_file()
    marker.unlink()
    marker.mkdir()
    (marker / "x.txt").write_bytes(b"x")
    try:
        result = l2.install(installer, "system", all_users=True)
        l2.report("marker_copy_failed_rc", result.rc)
        assert result.rc not in (0, 9), result.log[-3000:]
        assert "all-users" in result.log, result.log[-3000:]
        assert "prewarm rc=" not in result.log, result.log[-3000:]
        shutil.rmtree(marker)
        again = l2.install(installer, "system", all_users=True)
        assert again.rc == 0, again.log[-3000:]
        assert marker.is_file()
    finally:
        if marker.is_dir():
            shutil.rmtree(marker, ignore_errors=True)
        _remove("machine")
        shutil.rmtree(PROGRAM_FILES, ignore_errors=True)


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
