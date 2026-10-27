"""Level 2: two people on one PC (Review Focus 2). A second local account,
FoL2Other, runs in its own non-interactive session while the signed-in
user installs and runs Food Optimizer. Nobody else's app can block a
per-user install; the IT copy's running app is seen by every user's
installer and shell. Every test starts from a clean machine and
uninstalls what it installed; the account and its profile are removed at
the end of the module."""
import os
import pathlib
import re

import pytest

import l2_support as l2
from conftest import L2Home

pytestmark = pytest.mark.usefixtures("clean_machine")

PROGRAM_FILES = pathlib.Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Food Optimizer"
GUID = "3CC8CB77-0313-4A7D-9582-9896F0A3A8B3"
ALL_USERS = rf"Global\FoodOptimizer.Running.AllUsers.{GUID}"
# Every Global name an installer has ever checked: for the running app,
# the old one (AppMutex, before 0.9.0's per-scope check) and both of
# today's names in the Global namespace; for another installer, the
# Global SetupMutex name (all-users installers only, since 0.9.0).
EVERY_GLOBAL_NAME = (rf"Global\FoodOptimizer.Running.{GUID}",
                     rf"Global\FoodOptimizer.Running.PerUser.{GUID}",
                     ALL_USERS,
                     r"Global\FoodOptimizer.Setup.0E00CC2F-4E51-4EBF-A229-A910D5742A39")
HELD = r"mutex\.(held|global_failed|failed)( .*)?"

# Creates each name with an Everyone-full DACL, writes "<name>=<created>"
# lines, then holds them until the stop file appears (15 minutes at most).
HOLD_PS1 = r"""
param([string]$Names, [string]$Ready, [string]$Stop)
$ErrorActionPreference = 'Stop'
$everyone = New-Object System.Security.Principal.SecurityIdentifier('S-1-1-0')
$sec = New-Object System.Security.AccessControl.MutexSecurity
$sec.AddAccessRule((New-Object System.Security.AccessControl.MutexAccessRule($everyone, 'FullControl', 'Allow')))
$held = @()
$lines = @()
foreach ($name in ($Names -split ',')) {
    $created = $false
    $held += New-Object System.Threading.Mutex($false, $name, [ref]$created, $sec)
    $lines += ($name + '=' + $created)
}
[IO.File]::WriteAllLines($Ready, [string[]]$lines)
$deadline = (Get-Date).AddMinutes(15)
while (-not (Test-Path -LiteralPath $Stop) -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 200 }
"""


@pytest.fixture(scope="module")
def other_user(installer):
    sid = l2.add_other_user()
    try:
        # It must be able to run a password task (batch logon); if not,
        # stop here rather than change local security policy.
        probe = l2.OTHER_JOBS / "probe.txt"
        probe.unlink(missing_ok=True)
        rc = l2.run_as_other(["cmd.exe", "/c", f"echo ok>{probe}"], timeout=180)
        assert rc == 0 and probe.read_text().strip() == "ok", f"{l2.OTHER_USER} cannot run a batch task (rc {rc})"
        probe.unlink()
        yield sid
    finally:
        l2.remove_other_user()
        assert l2.other_user_sid() is None, f"{l2.OTHER_USER} was not deleted"
        if l2.other_profile_exists(sid):
            # Windows can keep a batch logon's profile loaded after the
            # logon ends, held by a process an admin cannot inspect
            # (VM_BUILD, 2026-10-27: a bare "cmd /c echo" password task's
            # profile was still loaded 20 minutes later). The account is
            # gone; the next run removes the profile once it is unloaded
            # (at the latest after a restart).
            l2.report("other_profile_left", sid)


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


def _log(home) -> str:
    path = home.state / "shell.log"
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else "(no shell.log)"


def _start_other_shell(tag, held_timeout):
    """FoL2Other starts the all-users shell (its folders under C:\\FOL2\\<tag>;
    WebView2 may not start in its non-interactive session, which is fine:
    the name is held before). Returns the job, its home and the first
    mutex line of its shell.log within held_timeout (None if none)."""
    home = L2Home(tag).make()
    l2.grant(home.base, l2.OTHER_USER)
    job = l2.run_as_other([str(PROGRAM_FILES / "FoodOptimizer.exe")], env=home.env, wait=False)
    assert l2.wait_log(home.state / "shell.log", r"shell\.start", 120), _log(home)[-3000:]
    return job, home, l2.wait_log(home.state / "shell.log", HELD, held_timeout)


def test_another_user_cannot_block_my_per_user_install(installer, other_user, l2_home, cleanup):
    work = l2_home.base / "hold"
    work.mkdir()
    l2.grant(work, l2.OTHER_USER)
    script, ready, stop = work / "hold.ps1", work / "ready.txt", work / "stop.txt"
    script.write_text(HOLD_PS1, encoding="ascii")
    holder = l2.run_as_other(["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                              "-File", str(script), "-Names", ",".join(EVERY_GLOBAL_NAME),
                              "-Ready", str(ready), "-Stop", str(stop)], wait=False)
    try:
        assert l2.wait_for(ready.exists, 120), f"{l2.OTHER_USER} never made the names"
        assert ready.read_text(encoding="ascii").split() == [f"{n}=True" for n in EVERY_GLOBAL_NAME]
        assert all(l2.mutex_exists(n) for n in EVERY_GLOBAL_NAME)
        result = l2.install(installer, "user", dir=l2_home.root, env=l2_home.env)
        l2.report("two_users_per_user_install_rc", result.rc)
        assert result.rc == 0, (result.rc, result.log[-3000:])
        assert "refused: app_running" not in result.log
        assert holder.alive()
    finally:
        stop.write_text("stop", encoding="ascii")
        if not l2.wait_for(lambda: not holder.alive(), 30):
            l2.end_job(holder)
        holder.close()


def test_an_it_upgrade_sees_another_users_app(installer, other_user, cleanup):
    first = l2.install(installer, "system", all_users=True)
    assert first.rc == 0, first.log[-3000:]
    job, home, m = _start_other_shell("o2", 120)
    try:
        again = l2.install(installer, "system", all_users=True, close_apps=False)
        l2.report("two_users_it_upgrade_rc", again.rc)
        assert again.rc == l2.EXIT_CODES["app_running"], (again.rc, again.log[-3000:])
        assert job.alive()
    finally:
        l2.end_job(job)
    assert l2.wait_for(lambda: not l2.processes_under(PROGRAM_FILES), 30), l2.processes_under(PROGRAM_FILES)
    after = l2.install(installer, "system", all_users=True, close_apps=False)
    assert after.rc == 0, after.log[-3000:]
    # How it was seen: the all-users installer marks its folder, and the
    # shell held the all-users name (checked last, so an older build's
    # run still shows the outcome above).
    assert "refused: app_running" in again.log
    assert (PROGRAM_FILES / "all-users").is_file()
    assert m and m.group(1) == "held", _log(home)[-3000:]
    assert f"mutex.held {ALL_USERS} created" in _log(home), _log(home)[-3000:]


def test_my_app_sees_the_global_name_another_user_made(installer, other_user, l2_home, cleanup):
    first = l2.install(installer, "system", all_users=True)
    assert first.rc == 0, first.log[-3000:]
    # FoL2Other's shell made the name before mine starts, so the roles can't swap.
    other, home, m = _start_other_shell("o3", 120)
    try:
        assert m and m.group(1) == "held", _log(home)[-3000:]
        assert f"mutex.held {ALL_USERS} created" in _log(home), _log(home)[-3000:]
        mine = l2.start_shell(PROGRAM_FILES, l2_home)
        try:
            assert l2.wait_log(l2_home.state / "shell.log", HELD, 120), _log(l2_home)[-3000:]
            text = _log(l2_home)
            assert not re.search(r"mutex\.(global_)?failed", text), text[-3000:]
            assert f"mutex.held {ALL_USERS} opened" in text, text[-3000:]
        finally:
            l2.end_job(mine)
    finally:
        l2.end_job(other)
    l2.wait_for(lambda: not l2.processes_under(PROGRAM_FILES), 30)
