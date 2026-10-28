"""Level 2 helpers: install, run and uninstall the packaged app as the right
Windows identity, and read what it left behind (spec Testing, Level 2).

The tests run under an elevated runner on the interactive desktop
(run_level2.ps1). run_as() drops each install or app start to the limited
user (a scheduled task with /IT), to SYSTEM (/RU SYSTEM), or runs it
directly as admin. Stdlib only (ctypes, winreg); importable on any OS so
the Mac can read the constants, usable only on Windows.
"""
import atexit
import contextlib
import ctypes
import datetime
import getpass
import hashlib
import importlib.util
import itertools
import json
import os
import pathlib
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional, Sequence

try:
    import winreg
    from ctypes import wintypes
except ImportError:          # the Mac reads the constants only
    winreg = None
    wintypes = None

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
VERSION = (REPO / "desktop" / "VERSION").read_text(encoding="utf-8").strip()
APP_ID = "{DE876ED9-927A-45AD-AD9B-D6E4B4C9A4A3}"
UNINSTALL_SUBKEY = rf"Software\Microsoft\Windows\CurrentVersion\Uninstall\{APP_ID}_is1"
SILENT = ("/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/CLOSEAPPLICATIONS")

# The installer's exit codes (Inno Setup's 0-8, plus our 9; the README's
# table). app_running is pinned by Task 8: what Inno's AppMutex check
# returned under /SUPPRESSMSGBOXES while the app ran; setup.iss's own
# per-scope check (AppClosed) keeps that code.
EXIT_CODES = {
    "ok": 0,
    "init_failed": 1,
    "cancelled": 2,
    "prepare_failed": 3,
    "install_failed": 4,
    "aborted": 5,
    "killed": 6,
    "refused": 7,
    "restart_needed": 8,
    "prewarm_failed": 9,
    # Inno's AppMutex check under /SUPPRESSMSGBOXES (VM_BUILD, 2026-10-24,
    # Inno 6.7.3): 1 for a per-user install, a SYSTEM all-users install and
    # a silent uninstall while the app runs. The same code as init_failed:
    # Intune treats code 1 as Retry, for both.
    "app_running": 1,
}
# Intune's return-code type for each (the README's table; a literal, so
# the Mac test reads it by AST).
INTUNE = {
    "ok": "Success",
    "init_failed": "Retry",         # shares code 1 with app_running
    "cancelled": "Failed",
    "prepare_failed": "Failed",
    "install_failed": "Failed",
    "aborted": "Failed",
    "killed": "Failed",
    "refused": "Failed",
    "restart_needed": "Failed",
    "prewarm_failed": "Failed",
    "app_running": "Retry",
}
REFUSALS = ("system_needs_allusers", "installed_by_it", "folder_too_long", "folder_not_ours", "needs_space")

# Seconds to the app's first page, and to it again on a warm relaunch. The
# x64 bound is the spec's 30 s (re-set from Azure in phase 6); arm64 is the
# Fusion VM under emulation, set in Task 7 from measurement.
FIRST_OPEN_LIMIT_S = 180
# arm64: VM_BUILD (Fusion, Windows 11 ARM, x64 emulation), 2026-10-24 at
# dd149ad: 5.1, 6.4 and 6.3 s; ceil(2 x median 6.3) = 13.
WARM_RELAUNCH_LIMIT_S = {"x64": 30, "arm64": 13}

# The shell's line once the app loads. The address's base path, its
# per-launch secret, is redacted (M3): shell.log never holds it; it is in
# the state folder's server.base. Group 1 is the port.
NAVIGATE = r"app\.navigate http://127\.0\.0\.1:(\d+)/\u2026/"

L2 = pathlib.Path(os.environ.get("FOODOPT_L2_ROOT") or r"C:\work\l2")
JOBS = L2 / "jobs"
OTHER_JOBS = JOBS / "other"       # OTHER_USER's job specs and results, apart from SYSTEM's
LOGS = L2 / "logs"
CREATE_NO_WINDOW = 0x08000000
_counter = itertools.count(1)
_run_stamp = time.strftime("%H%M%S")


def _name(prefix: str) -> str:
    return f"{prefix}{_run_stamp}-{os.getpid()}-{next(_counter)}"


def _load_build_env():
    spec = importlib.util.spec_from_file_location("build_env", REPO / "desktop" / "windows" / "build_env.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------- running as the right identity ----------

# The trigger of every scheduled task the harness creates: a start boundary
# in the past, so Windows never fires it on schedule; /Run still runs it.
# (schtasks warns "Task may not run because /ST is earlier than current
# time"; the exit code is what counts.)
NEVER_ON_SCHEDULE = ["/SC", "ONCE", "/SD", "01/01/2020", "/ST", "00:00"]

_live_tasks = set()          # tasks created and not yet deleted
_live_lock = threading.Lock()


def _delete_task(task: str):
    """Delete a task; one that is already gone is harmless."""
    with _live_lock:
        _live_tasks.discard(task)
    subprocess.run(["schtasks", "/Delete", "/F", "/TN", task], capture_output=True,
                   creationflags=CREATE_NO_WINDOW)


def end_all_jobs():
    """Delete every task still registered (session teardown, atexit)."""
    with _live_lock:
        tasks = sorted(_live_tasks)
    for task in tasks:
        _delete_task(task)


atexit.register(end_all_jobs)

class Job:
    """A process started by run_as. pid and elevated come from the job
    runner (l2_job.py) or, for admin, from the runner itself."""

    def __init__(self, name: str, kind: str, result: pathlib.Path, task: Optional[str] = None,
                 proc: Optional[subprocess.Popen] = None):
        self.name, self.kind, self.result, self.task, self.proc = name, kind, result, task, proc
        self._info = None
        if proc is not None:
            self._info = {"pid": proc.pid, "elevated": _token_elevated(), "started": time.time()}
        else:
            self._wait_info(120)

    def _read(self) -> Optional[dict]:
        try:
            return json.loads(self.result.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _wait_info(self, timeout):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            info = self._read()
            if info and "pid" in info:
                self._info = info
                return
            time.sleep(0.1)
        raise TimeoutError(f"job {self.name} ({self.kind}) never started; see {self.result}")

    @property
    def pid(self) -> int:
        return int(self._info["pid"])

    @property
    def elevated(self) -> bool:
        return bool(self._info["elevated"])

    def poll(self) -> Optional[int]:
        if self.proc is not None:
            return self.proc.poll()
        info = self._read()
        return None if not info or "rc" not in info else int(info["rc"])

    def wait(self, timeout: float = 1800) -> int:
        end = time.monotonic() + timeout
        try:
            while True:
                rc = self.poll()
                if rc is not None:
                    return rc
                if time.monotonic() >= end:
                    raise TimeoutError(f"job {self.name} ({self.kind}) still running after {timeout} s")
                time.sleep(0.2)
        finally:
            self.close()

    def alive(self) -> bool:
        return self.poll() is None

    def close(self):
        """Delete this job's scheduled task (idempotent)."""
        task, self.task = self.task, None
        if task:
            _delete_task(task)


def _runner_pythonw() -> str:
    """The job runner's interpreter: pythonw beside this Python, so a
    scheduled task opens no console window of its own."""
    exe = pathlib.Path(sys.executable)
    for name in ("pythonw.exe", exe.name):
        candidate = exe.with_name(name)
        if candidate.is_file():
            return str(candidate)
    return str(exe)


def run_as(kind: str, argv: Sequence[str], *, env=None, cwd=None, wait=True, timeout=1800):
    """Run argv as "user" (limited token, interactive desktop), "system",
    "admin" or "other" (OTHER_USER, see run_as_other). env holds extra
    variables only, laid over that identity's own environment. wait=True
    returns the exit code; wait=False returns the Job."""
    if kind not in ("user", "system", "admin", "other"):
        raise ValueError(kind)
    if kind == "admin":
        full_env = dict(os.environ)
        full_env.update(env or {})
        proc = subprocess.Popen([str(a) for a in argv], env=full_env, cwd=cwd)
        job = Job(_name("admin"), kind, pathlib.Path(), proc=proc)
        return job.wait(timeout) if wait else job
    folder = OTHER_JOBS if kind == "other" else JOBS
    folder.mkdir(parents=True, exist_ok=True)
    name = _name("j")
    spec = folder / f"{name}.json"
    result = folder / f"{name}.result.json"
    # Only the extras travel: the job runs in its identity's own environment
    # (the user's, or SYSTEM's as Intune would run it).
    spec.write_text(json.dumps({"argv": [str(a) for a in argv], "env": dict(env or {}),
                                "cwd": str(cwd) if cwd else None, "result": str(result)}),
                    encoding="utf-8")
    task = f"FoodOptL2-{name}"
    runner = L2 / "l2_job.py"
    command = f'"{_runner_pythonw()}" "{runner}" "{spec}"'
    if kind == "system":
        who = ["/RU", "SYSTEM"]
    elif kind == "other":
        if _other_password is None:
            raise RuntimeError(f"{OTHER_USER} does not exist: add_other_user() first")
        # A password task: OTHER_USER is not signed in, so it runs in its
        # own non-interactive session (batch logon).
        who = ["/RU", OTHER_USER, "/RP", _other_password]
    else:
        who = ["/RU", os.environ.get("USERNAME") or getpass.getuser(), "/IT"]
    created = subprocess.run(["schtasks", "/Create", "/F", "/TN", task, "/TR", command, *NEVER_ON_SCHEDULE,
                              *who], capture_output=True, text=True, creationflags=CREATE_NO_WINDOW)
    if created.returncode != 0:
        _delete_task(task)
        raise RuntimeError(f"schtasks /Create failed: {created.stdout} {created.stderr}")
    with _live_lock:
        _live_tasks.add(task)
    try:
        ran = subprocess.run(["schtasks", "/Run", "/TN", task], capture_output=True, text=True,
                             creationflags=CREATE_NO_WINDOW)
        if ran.returncode != 0:
            raise RuntimeError(f"schtasks /Run failed: {ran.stdout} {ran.stderr}")
        job = Job(name, kind, result, task=task)
    except BaseException:
        _delete_task(task)
        raise
    return job.wait(timeout) if wait else job


# ---------- a second user on the PC (Review Focus 2) ----------

# A local account that is not signed in: a member of Users and of
# Performance Log Users only. VM_BUILD's "Log on as a batch job" right
# (SeBatchLogonRight) is Administrators, Backup Operators and Performance
# Log Users (measured 2026-10-26), so that group lets it run a password
# task without admin rights or a policy change.
OTHER_USER = "FoL2Other"
PERFORMANCE_LOG_USERS = "S-1-5-32-559"
_other_password: Optional[str] = None


def _new_password() -> str:
    """24 random characters with every class Windows' complexity rule asks for."""
    import secrets
    import string
    classes = (string.ascii_uppercase, string.ascii_lowercase, string.digits, "!#%+-=_.")
    while True:
        chars = [secrets.choice("".join(classes)) for _ in range(24)]
        if all(any(c in cls for c in chars) for cls in classes):
            return "".join(chars)


def _powershell(script: str, timeout=300) -> subprocess.CompletedProcess:
    return subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                           "-Command", script], capture_output=True, text=True, errors="replace",
                          timeout=timeout, creationflags=CREATE_NO_WINDOW)


def other_user_sid() -> Optional[str]:
    r = _powershell(f"$u = Get-LocalUser -Name '{OTHER_USER}' -ErrorAction SilentlyContinue; "
                    "if ($u) { $u.SID.Value }")
    sid = r.stdout.strip()
    return sid or None


def add_other_user() -> str:
    """Creates OTHER_USER with a fresh random password (kept in this process
    only, never logged), in Users and Performance Log Users, and lets it
    write its own job folder only. Returns its SID."""
    global _other_password
    if other_user_sid() is not None:
        remove_other_user()
    remove_stale_other_profiles()
    password = _new_password()
    # /Y answers net's question about passwords over 14 characters (with no
    # console to ask, it exits -1).
    r = subprocess.run(["net", "user", OTHER_USER, password, "/add", "/passwordchg:no", "/expires:never", "/Y"],
                       capture_output=True, text=True, errors="replace", creationflags=CREATE_NO_WINDOW)
    if r.returncode != 0:
        raise RuntimeError(f"net user {OTHER_USER} /add exited {r.returncode}")
    _other_password = password
    r = _powershell(f"Add-LocalGroupMember -SID '{PERFORMANCE_LOG_USERS}' -Member '{OTHER_USER}'")
    if r.returncode != 0:
        raise RuntimeError(f"Add-LocalGroupMember: {r.stderr}")
    protect_jobs()
    OTHER_JOBS.mkdir(parents=True, exist_ok=True)
    grant(OTHER_JOBS, OTHER_USER)
    return other_user_sid()


def protect_jobs() -> None:
    """The job folder, whose specs SYSTEM's job runner reads, writable by
    Administrators, SYSTEM and the signed-in user only: under C:\\ every
    Authenticated User inherits Modify, OTHER_USER too."""
    JOBS.mkdir(parents=True, exist_ok=True)
    user = os.environ.get("USERNAME") or getpass.getuser()
    r = subprocess.run(["icacls", str(JOBS), "/inheritance:r", "/grant:r", "*S-1-5-32-544:(OI)(CI)F",
                        "*S-1-5-18:(OI)(CI)F", f"{user}:(OI)(CI)M"], capture_output=True, text=True,
                       errors="replace", creationflags=CREATE_NO_WINDOW)
    if r.returncode != 0:
        raise RuntimeError(f"icacls {JOBS} /inheritance:r: {r.stdout}")


_grants: list = []                # (folder, account) given by grant(), taken back by remove_other_user()


def grant(folder, account: str) -> None:
    """Modify on folder and below for account (Level 2's own folders only)."""
    r = subprocess.run(["icacls", str(folder), "/grant", f"{account}:(OI)(CI)M"], capture_output=True,
                       text=True, errors="replace", creationflags=CREATE_NO_WINDOW)
    if r.returncode != 0:
        raise RuntimeError(f"icacls {folder} /grant {account}: {r.stdout}")
    _grants.append((str(folder), account))


def kill_user_processes(account: str) -> None:
    """Ends every process the account runs (the tests' own: it is a test
    account that exists only for one module)."""
    _powershell(f"Get-Process -IncludeUserName | Where-Object {{ $_.UserName -like '*\\{account}' }} | "
                "Stop-Process -Force -ErrorAction SilentlyContinue")


def remove_other_user() -> None:
    """Deletes OTHER_USER, its profile and its access to the job folder."""
    global _other_password
    sid = other_user_sid()
    if sid is not None:
        kill_user_processes(OTHER_USER)
        # Windows may keep a batch logon's hive loaded after the logon ends
        # (other_profile_exists then stays true; see the module fixture).
        for _ in range(15):
            r = _powershell(f"Get-CimInstance Win32_UserProfile | Where-Object {{ $_.SID -eq '{sid}' }} | "
                            "Remove-CimInstance -ErrorAction Stop")
            if r.returncode == 0:
                break
            time.sleep(2)
        # Its ACEs go while the name still resolves (a folder already
        # deleted is harmless).
        for folder, account in [g for g in _grants if g[1] == OTHER_USER]:
            if os.path.exists(folder):
                subprocess.run(["icacls", folder, "/remove", account], capture_output=True,
                               creationflags=CREATE_NO_WINDOW)
        _grants[:] = [g for g in _grants if g[1] != OTHER_USER]
        subprocess.run(["net", "user", OTHER_USER, "/delete"], capture_output=True,
                       creationflags=CREATE_NO_WINDOW)
    _other_password = None


def remove_stale_other_profiles() -> None:
    """Profiles an earlier run's OTHER_USER left (their hive was still
    loaded then), once Windows has unloaded them."""
    _powershell(f"Get-CimInstance Win32_UserProfile | Where-Object {{ -not $_.Loaded -and "
                f"(Split-Path -Leaf $_.LocalPath) -like '{OTHER_USER}*' }} | "
                "Remove-CimInstance -ErrorAction SilentlyContinue")


def other_profile_exists(sid: str) -> bool:
    r = _powershell(f"@(Get-CimInstance Win32_UserProfile | Where-Object {{ $_.SID -eq '{sid}' }}).Count")
    return r.stdout.strip() not in ("", "0")


def run_as_other(argv: Sequence[str], env=None, *, cwd=None, wait=True, timeout=1800):
    """run_as for OTHER_USER: a password task with the never-on-schedule
    trigger, deleted when the job ends (or is closed)."""
    return run_as("other", argv, env=env, cwd=cwd, wait=wait, timeout=timeout)


def end_job(job: "Job") -> None:
    """Ends a job's process tree (the task's own /End can leave its
    children running) and deletes its task."""
    if job.proc is None and job.alive():
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(job.pid)], capture_output=True,
                       creationflags=CREATE_NO_WINDOW)
        wait_for(lambda: not job.alive(), 30, 0.2)
    job.close()


def mutex_exists(name: str) -> bool:
    """Whether a named mutex exists and this token may wait on it, as
    Inno's CheckForMutexes asks (OpenMutexW with SYNCHRONIZE)."""
    k32 = ctypes.windll.kernel32
    k32.OpenMutexW.restype = wintypes.HANDLE
    k32.OpenMutexW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = k32.OpenMutexW(0x00100000, False, name)
    if not handle:
        return False
    k32.CloseHandle(handle)
    return True


def _token_elevated() -> bool:
    if sys.platform != "win32":
        return False
    return bool(ctypes.windll.shell32.IsUserAnAdmin())


# ---------- install and uninstall ----------

@dataclass
class Install:
    rc: int
    log: str
    seconds: float
    app_dir: Optional[pathlib.Path]
    elevated: Optional[bool] = None


def install(installer, kind="user", *, dir=None, all_users=False, extra=(), env=None,
            close_apps=True) -> Install:
    """close_apps=False leaves out /CLOSEAPPLICATIONS, so a running app the
    check missed is not closed for it."""
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{_name('setup')}.log"
    silent = SILENT if close_apps else tuple(s for s in SILENT if s != "/CLOSEAPPLICATIONS")
    argv = [str(installer), *silent, f"/LOG={log}"]
    if all_users:
        argv.append("/ALLUSERS")
    if dir is not None:
        argv.append(f"/DIR={dir}")
    argv += list(extra)
    started = time.monotonic()
    job = run_as(kind, argv, env=env, wait=False)
    rc = job.wait(3600)
    seconds = time.monotonic() - started
    text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    entry = uninstall_entry("machine" if all_users else "user")
    app_dir = pathlib.Path(entry["InstallLocation"]) if entry and entry.get("InstallLocation") else None
    return Install(rc=rc, log=text, seconds=seconds, app_dir=app_dir, elevated=job.elevated)


def uninstall(app_dir, kind="user", *, env=None) -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{_name('unins')}.log"
    argv = [str(pathlib.Path(app_dir) / "unins000.exe"), "/VERYSILENT", "/SUPPRESSMSGBOXES",
            "/NORESTART", f"/LOG={log}"]
    rc = run_as(kind, argv, env=env)
    # unins000.exe hands the work to a copy of itself in %TEMP%, which may
    # still be deleting the folder when the first one returns: wait for the
    # copy's log to close (it closes last, after the state folder).
    if rc == 0:
        wait_for(lambda: "Log closed." in _text(log), 300, 0.25)
    return rc


def _text(path) -> str:
    try:
        return pathlib.Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def last_uninstall_log() -> str:
    logs = sorted(LOGS.glob("unins*.log"), key=lambda p: p.stat().st_mtime)
    return logs[-1].read_text(encoding="utf-8", errors="replace") if logs else ""


def uninstall_entry(scope: str) -> Optional[dict]:
    if scope == "user":
        root, access = winreg.HKEY_CURRENT_USER, winreg.KEY_READ
    elif scope == "machine":
        root, access = winreg.HKEY_LOCAL_MACHINE, winreg.KEY_READ | winreg.KEY_WOW64_64KEY
    else:
        raise ValueError(scope)
    try:
        key = winreg.OpenKey(root, UNINSTALL_SUBKEY, 0, access)
    except OSError:
        return None
    out = {}
    with key:
        for i in itertools.count():
            try:
                name, value, _type = winreg.EnumValue(key, i)
            except OSError:
                break
            out[name] = value
    return out


# ---------- shortcuts, Quick access, files ----------

_FOLDERS = {
    "Programs": "{A77F5D77-2E2B-44C3-A6A2-ABA601054A51}",
    "Desktop": "{B4BFCC3A-DB2C-424C-B029-7FE99A87C641}",
    "CommonPrograms": "{0139D44E-6AFE-49F2-8690-3DAFCAE6FFB8}",
    "PublicDesktop": "{C4AA340D-F20F-4863-AFEF-F87EF2E6BA25}",
}


def _known_folder(guid: str) -> pathlib.Path:
    class GUID(ctypes.Structure):
        _fields_ = [("a", ctypes.c_uint32), ("b", ctypes.c_uint16), ("c", ctypes.c_uint16),
                    ("d", ctypes.c_ubyte * 8)]
    g = GUID()
    ctypes.oledll.ole32.CLSIDFromString(ctypes.c_wchar_p(guid), ctypes.byref(g))
    out = ctypes.c_wchar_p()
    ctypes.windll.shell32.SHGetKnownFolderPath(ctypes.byref(g), 0, None, ctypes.byref(out))
    try:
        return pathlib.Path(out.value)
    finally:
        ctypes.windll.ole32.CoTaskMemFree(out)


def shortcut_dirs(scope: str) -> dict:
    if scope == "user":
        return {"programs": _known_folder(_FOLDERS["Programs"]), "desktop": _known_folder(_FOLDERS["Desktop"])}
    return {"programs": _known_folder(_FOLDERS["CommonPrograms"]),
            "desktop": _known_folder(_FOLDERS["PublicDesktop"])}


def _powershell_json(script: str):
    r = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                        "-Command", "[Console]::OutputEncoding = [Text.Encoding]::UTF8; " + script],
                       capture_output=True, timeout=120, creationflags=CREATE_NO_WINDOW)
    out = r.stdout.decode("utf-8", "replace").strip()
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode("utf-8", "replace"))
    return json.loads(out) if out else None


def shortcut_target(lnk) -> str:
    path = str(lnk).replace("'", "''")
    return _powershell_json(f"(New-Object -ComObject WScript.Shell).CreateShortcut('{path}').TargetPath"
                            " | ConvertTo-Json")


def quick_access() -> list:
    found = _powershell_json(
        "@((New-Object -ComObject Shell.Application).Namespace("
        "'shell:::{679f85cb-0220-4080-b29b-5540cc05aab6}').Items() | ForEach-Object { $_.Path })"
        " | ConvertTo-Json")
    if found is None:
        return []
    return found if isinstance(found, list) else [found]


def in_quick_access(path) -> bool:
    want = os.path.normcase(str(path))
    return any(os.path.normcase(p) == want for p in quick_access())


def _user_powershell(script: str) -> int:
    """PowerShell as the limited user on the desktop: Explorer's pin verbs
    act for the user who runs them, and an elevated runner's call (VM_BUILD,
    2026-10-27) left Quick access unchanged."""
    return run_as("user", ["powershell.exe", "-NoProfile", "-NonInteractive", "-WindowStyle", "Hidden",
                           "-ExecutionPolicy", "Bypass", "-Command", script], timeout=180)


def pin(path) -> bool:
    """Pins a folder to Quick access, as the user would by hand; True once
    Quick access lists it."""
    p = str(path).replace("'", "''")
    _user_powershell(f"$f = (New-Object -ComObject Shell.Application).Namespace('{p}'); "
                     "if ($f) { $f.Self.InvokeVerb('pintohome') }")
    return wait_for(lambda: in_quick_access(path), 15, 0.5)


def unpin_under(folder) -> list:
    """Unpins every Quick access entry at or under folder, gone or not: the
    app pins each test's projects folder at its first launch (D2), and
    Level 2's folders are its own. Returns the paths it found."""
    prefix = os.path.normcase(str(folder).rstrip("\\"))
    found = [q for q in quick_access()
             if os.path.normcase(q) == prefix or os.path.normcase(q).startswith(prefix + "\\")]
    if found:
        p = str(folder).rstrip("\\").replace("'", "''")
        _user_powershell(
            "$qa = (New-Object -ComObject Shell.Application).Namespace("
            "'shell:::{679f85cb-0220-4080-b29b-5540cc05aab6}'); "
            "foreach ($i in @($qa.Items())) { $q = [string]$i.Path; "
            f"if ($q -eq '{p}' -or $q.StartsWith('{p}\\', [StringComparison]::OrdinalIgnoreCase)) "
            "{ $i.InvokeVerb('unpinfromhome') } }")
        wait_for(lambda: not any(in_quick_access(q) for q in found), 15, 0.5)
    return found


def settings_text(home) -> str:
    """The shell's settings.json in home's state folder ('' if none)."""
    return _text(pathlib.Path(home.state) / "settings.json")


def manifest(dir, skip=("unins000.exe", "unins000.dat")) -> str:
    text, _files, _size = _load_build_env().file_manifest(dir)
    return "".join(line + "\n" for line in text.splitlines() if line[66:] not in skip)


def manifest_sha256(dir) -> str:
    return hashlib.sha256(manifest(dir).encode("utf-8")).hexdigest()


def version_info(path) -> dict:
    """The file's version strings, trimmed (Inno pads its setup's)."""
    p = str(path).replace("'", "''")
    info = _powershell_json(f"(Get-Item -LiteralPath '{p}').VersionInfo | Select-Object "
                            "ProductVersion, FileVersion, CompanyName, ProductName | ConvertTo-Json")
    return {k: (v or "").strip() for k, v in info.items()}


# ---------- processes ----------

def processes() -> list:
    """(pid, full image path) of every process this token may query."""
    if sys.platform != "win32":
        return []
    k32 = ctypes.windll.kernel32

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                    ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ctypes.c_void_p),
                    ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                    ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", ctypes.c_long),
                    ("dwFlags", wintypes.DWORD), ("szExeFile", ctypes.c_wchar * 260)]

    k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    k32.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
                                               ctypes.POINTER(wintypes.DWORD)]
    k32.Process32FirstW.argtypes = k32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
    snap = k32.CreateToolhelp32Snapshot(0x2, 0)
    out = []
    entry = PROCESSENTRY32W()
    entry.dwSize = ctypes.sizeof(entry)
    try:
        ok = k32.Process32FirstW(snap, ctypes.byref(entry))
        while ok:
            pid = entry.th32ProcessID
            handle = k32.OpenProcess(0x1000, False, pid)        # QUERY_LIMITED_INFORMATION
            if handle:
                try:
                    buf = ctypes.create_unicode_buffer(32768)
                    size = wintypes.DWORD(len(buf))
                    if k32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
                        out.append((pid, buf.value))
                finally:
                    k32.CloseHandle(handle)
            ok = k32.Process32NextW(snap, ctypes.byref(entry))
    finally:
        k32.CloseHandle(snap)
    return out


def processes_under(folder) -> list:
    prefix = os.path.normcase(os.path.abspath(str(folder))).rstrip("\\") + "\\"
    return [(pid, path) for pid, path in processes() if os.path.normcase(path).startswith(prefix)]


def pid_alive(pid: int) -> bool:
    k32 = ctypes.windll.kernel32
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = k32.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    try:
        code = wintypes.DWORD()
        k32.GetExitCodeProcess(handle, ctypes.byref(code))
        return code.value == 259                                # STILL_ACTIVE
    finally:
        k32.CloseHandle(handle)


def kill_under(folder) -> None:
    for pid, _path in processes_under(folder):
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True,
                       creationflags=CREATE_NO_WINDOW)


def host_is_arm64() -> bool:
    k32 = ctypes.windll.kernel32
    process, native = ctypes.c_ushort(), ctypes.c_ushort()
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    if not hasattr(k32, "IsWow64Process2"):
        return False
    k32.IsWow64Process2.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_ushort),
                                    ctypes.POINTER(ctypes.c_ushort)]
    k32.IsWow64Process2(k32.GetCurrentProcess(), ctypes.byref(process), ctypes.byref(native))
    return native.value == 0xAA64


def wait_for(predicate, timeout: float, interval: float = 0.1) -> bool:
    end = time.monotonic() + timeout
    while True:
        if predicate():
            return True
        if time.monotonic() >= end:
            return False
        time.sleep(interval)


# ---------- a small drive (diskpart) ----------

@contextlib.contextmanager
def vhd(size_mb: int):
    """A fresh NTFS drive of size_mb, attached at a free letter, that the
    limited user may write to; detached and deleted afterwards (admin)."""
    folder = L2 / "vhd"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{_name('d')}.vhdx"
    used = {p[0] for p in os.listdrives()} if hasattr(os, "listdrives") else set()
    letter = next(c for c in "RSTUVWXYQPONML" if c not in used and not os.path.exists(f"{c}:\\"))

    def diskpart(lines):
        script = folder / f"{_name('dp')}.txt"
        script.write_text("\n".join(lines) + "\n", encoding="ascii")
        r = subprocess.run(["diskpart", "/s", str(script)], capture_output=True, text=True,
                           errors="replace", creationflags=CREATE_NO_WINDOW)
        return r

    r = diskpart([f'create vdisk file="{path}" maximum={size_mb} type=expandable',
                  f'select vdisk file="{path}"', "attach vdisk", "create partition primary",
                  "format fs=ntfs quick label=FOL2", f"assign letter={letter}"])
    if r.returncode != 0 or not os.path.exists(f"{letter}:\\"):
        diskpart([f'select vdisk file="{path}"', "detach vdisk"])
        raise RuntimeError(f"diskpart failed ({r.returncode}): {r.stdout}")
    root = pathlib.Path(f"{letter}:\\")
    user = os.environ.get("USERNAME", "")
    subprocess.run(["icacls", str(root), "/grant", f"{user}:(OI)(CI)M"], capture_output=True,
                   creationflags=CREATE_NO_WINDOW)
    try:
        yield root
    finally:
        kill_under(root)
        diskpart([f'select vdisk file="{path}"', "detach vdisk"])
        with contextlib.suppress(OSError):
            path.unlink()


def diskpart_available() -> bool:
    return sys.platform == "win32" and os.path.exists(os.path.join(os.environ.get("SystemRoot", r"C:\Windows"),
                                                                    "System32", "diskpart.exe"))


# ---------- the installed shell ----------

def start_shell(app_dir, home, env=None) -> Job:
    """FoodOptimizer.exe from an install, as the limited user on the
    desktop, with home's state and projects folders."""
    full = dict(home.env)
    full.update(env or {})
    return run_as("user", [str(pathlib.Path(app_dir) / "FoodOptimizer.exe")], env=full, wait=False)


def log_size(path) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def wait_log(path, pattern, timeout, after: int = 0):
    """The first match of pattern in the file from byte `after` on,
    re-read every 100 ms (shared access), or None at the timeout."""
    rx = re.compile(pattern)
    end = time.monotonic() + timeout
    while True:
        try:
            with open(path, "rb") as f:
                f.seek(after)
                m = rx.search(f.read().decode("utf-8", "replace"))
            if m:
                return m
        except OSError:
            pass
        if time.monotonic() >= end:
            return None
        time.sleep(0.1)


_CONSOLE_CLASSES = ("ConsoleWindowClass", "CASCADIA_HOSTING_WINDOW_CLASS")


def _windows(predicate) -> list:
    user32 = ctypes.windll.user32
    found = []
    proto = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def each(hwnd, _lparam):
        if user32.IsWindowVisible(hwnd) and predicate(hwnd):
            found.append(hwnd)
        return True

    user32.EnumWindows(proto(each), 0)
    return found


def _class(hwnd) -> str:
    buf = ctypes.create_unicode_buffer(256)
    ctypes.windll.user32.GetClassNameW(hwnd, buf, 256)
    return buf.value


def _title(hwnd) -> str:
    buf = ctypes.create_unicode_buffer(512)
    ctypes.windll.user32.GetWindowTextW(hwnd, buf, 512)
    return buf.value


def window_pid(hwnd) -> int:
    pid = wintypes.DWORD()
    ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return pid.value


def top_windows(pid: int) -> list:
    """Visible top-level windows of one process."""
    return _windows(lambda h: window_pid(h) == pid)


def close_window(hwnd) -> None:
    ctypes.windll.user32.PostMessageW(hwnd, 0x0010, 0, 0)      # WM_CLOSE


class ConsoleWatch:
    """Every visible console window that appears between start() and
    stop(), checked every 200 ms (Review: no console window at any point)."""

    def __init__(self):
        self._before, self._new, self._stop = set(), {}, threading.Event()
        self._thread = None

    def _consoles(self):
        return _windows(lambda h: _class(h) in _CONSOLE_CLASSES)

    def start(self):
        self._before = set(self._consoles())
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            for h in self._consoles():
                if h not in self._before and h not in self._new:
                    self._new[h] = (window_pid(h), _title(h))
            self._stop.wait(0.2)

    def stop(self) -> list:
        self._stop.set()
        if self._thread:
            self._thread.join(5)
        return list(self._new.values())


# ---------- traffic: the Windows Filtering Platform audit ----------

WFP_CONNECTION = "{0CCE9226-69AE-11D9-BED3-505054503030}"     # "Filtering Platform Connection"
LOOPBACK = {"127.0.0.1", "::1"}
LISTEN_ANY = {"0.0.0.0", "::"}
_EVENT_NS = "{http://schemas.microsoft.com/win/2004/08/events/event}"


def _auditpol(*args) -> str:
    r = subprocess.run(["auditpol", *args], capture_output=True, text=True, errors="replace",
                       creationflags=CREATE_NO_WINDOW)
    if r.returncode != 0:
        raise RuntimeError(f"auditpol {' '.join(args)}: {r.stdout} {r.stderr}")
    return r.stdout


def wfp_setting() -> str:
    """The subcategory's current inclusion setting, e.g. "No Auditing"."""
    rows = [line for line in _auditpol("/get", f"/subcategory:{WFP_CONNECTION}", "/r").splitlines()
            if WFP_CONNECTION.lower() in line.lower()]
    return rows[0].split(",")[4].strip()


@contextlib.contextmanager
def wfp_audit():
    """Success and failure auditing of WFP connections for the block; the
    setting read first is restored exactly. Yields the UTC start (admin)."""
    before = wfp_setting()
    _auditpol("/set", f"/subcategory:{WFP_CONNECTION}", "/success:enable", "/failure:enable")
    started = datetime.datetime.now(datetime.timezone.utc)
    try:
        yield started
    finally:
        success = "enable" if "Success" in before else "disable"
        failure = "enable" if "Failure" in before else "disable"
        _auditpol("/set", f"/subcategory:{WFP_CONNECTION}", f"/success:{success}", f"/failure:{failure}")


def wfp_events(since: datetime.datetime) -> list:
    """5156 (allowed) and 5157 (blocked) connection events since `since`."""
    import xml.etree.ElementTree as ET
    iso = since.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    query = f"*[System[(EventID=5156 or EventID=5157) and TimeCreated[@SystemTime>='{iso}']]]"
    r = subprocess.run(["wevtutil", "qe", "Security", f"/q:{query}", "/f:xml", "/uni:true"],
                       capture_output=True, creationflags=CREATE_NO_WINDOW)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.decode("utf-16-le", "replace"))
    text = r.stdout.decode("utf-16-le", "replace").lstrip("\ufeff")
    root = ET.fromstring("<root>" + text + "</root>")
    events = []
    for event in root:
        data = {d.get("Name"): (d.text or "") for d in event.iter(f"{_EVENT_NS}Data")}
        events.append({"id": int(event.find(f"{_EVENT_NS}System/{_EVENT_NS}EventID").text),
                       "Application": data.get("Application", ""), "Direction": data.get("Direction", ""),
                       "SourceAddress": data.get("SourceAddress", ""), "DestAddress": data.get("DestAddress", ""),
                       "DestPort": data.get("DestPort", ""),
                       # the event's own field is "ProcessId" (System/Execution's
                       # ProcessID is the kernel's, 4)
                       "ProcessID": data.get("ProcessId", "")})
    return events


def _dos_device(drive: str) -> str:
    buf = ctypes.create_unicode_buffer(1024)
    ctypes.windll.kernel32.QueryDosDeviceW(drive, buf, 1024)
    return buf.value


def ours(event: dict, root, pids=()) -> bool:
    """Whether the event is Food Optimizer's: its process is in the shell's
    tree (pids, from TreeWatch; WebView2's msedgewebview2.exe included) or
    its image lies under the install root (the event names it by its
    \\device\\harddiskvolumeN path)."""
    try:
        if int(event.get("ProcessID") or -1) in pids:
            return True
    except ValueError:
        pass
    root = os.path.abspath(str(root))
    drive, rest = os.path.splitdrive(root)
    prefix = (_dos_device(drive) + rest).lower().rstrip("\\") + "\\"
    return event["Application"].lower().startswith(prefix)


# ---------- the shell's process tree ----------

def _process_rows_script(where: str = "") -> str:
    """PowerShell for Win32_Process rows as one JSON line: pid, parent pid,
    creation time (FILETIME, UTC), image and command line."""
    flt = f" -Filter '{where}'" if where else ""
    return ("@(Get-CimInstance Win32_Process" + flt + " | ForEach-Object { [pscustomobject]@{ "
            "pid = [int]$_.ProcessId; ppid = [int]$_.ParentProcessId; "
            "created = $(if ($null -ne $_.CreationDate) { $_.CreationDate.ToFileTimeUtc() } else { 0 }); "
            "image = [string]$_.ExecutablePath; cmd = [string]$_.CommandLine } }) | ConvertTo-Json -Compress")


def process_rows(where: str = "") -> list:
    """Every process (or those matching a WQL filter): Win32_Process rows,
    as _process_rows_script reads them (admin sees every user's)."""
    rows = _powershell_json(_process_rows_script(where)) or []
    return rows if isinstance(rows, list) else [rows]


def process_tree(root_pid: int, rows=None, known=None) -> set:
    """The pids of root_pid and its descendants, by ParentProcessId. A child
    counts only if it started after its parent (a reused parent pid does
    not adopt strangers). known: {pid: creation time} seen in the tree
    before, the root's included once its start is known; those still
    running (same pid, same start) seed the walk, so a child whose parent
    has exited (WebView2 outlives the shell by a moment) stays in, and a
    reused root pid is not taken for the shell."""
    rows = process_rows() if rows is None else rows
    created = {r["pid"]: r["created"] for r in rows}
    children = {}
    for r in rows:
        if r["pid"] != r["ppid"]:
            children.setdefault(r["ppid"], []).append(r["pid"])
    known = known or {}
    todo = [p for p in created
            if known.get(p) == created[p] or (p == root_pid and p not in known)]
    tree = set()
    while todo:
        pid = todo.pop()
        if pid in tree:
            continue
        tree.add(pid)
        todo.extend(c for c in children.get(pid, ()) if created[c] >= created[pid])
    return tree


class TreeWatch:
    """The shell's process tree, sampled every INTERVAL_S between start()
    and stop() and accumulated: {pid: {"image", "cmd", "created"}} of every
    process that was in it at any sample. One long-lived PowerShell writes
    a Win32_Process JSON line per sample (a query takes ~35-55 ms on
    VM_BUILD; a fresh powershell.exe per sample took ~0.5 s more), so the
    gap between samples stays small. polls() says how many samples parsed,
    how many failed and the largest gap between good ones (start and stop
    count as ends), so a watch that went blind shows."""

    INTERVAL_S = 0.25

    def __init__(self, root_pid: int):
        self.root_pid, self.seen = root_pid, {}
        self.ok = self.failed = 0
        self.max_gap_s = 0.0
        self._proc = self._thread = None
        self._last = None
        self._stopping = False
        self._lock = threading.Lock()

    def take(self, rows) -> set:
        """One sample: the tree in rows, added to seen."""
        with self._lock:
            tree = process_tree(self.root_pid, rows, {p: e["created"] for p, e in self.seen.items()})
            by_pid = {r["pid"]: r for r in rows}
            for pid in tree:
                entry = self.seen.setdefault(pid, {"image": "", "cmd": "", "created": by_pid[pid]["created"]})
                entry["image"] = by_pid[pid]["image"] or entry["image"]
                entry["cmd"] = by_pid[pid]["cmd"] or entry["cmd"]
            return tree

    def _mark(self, good: bool):
        now = time.monotonic()
        with self._lock:
            if good:
                self.ok += 1
                self.max_gap_s = max(self.max_gap_s, now - self._last)
                self._last = now
            else:
                self.failed += 1

    def _read(self):
        for raw in self._proc.stdout:
            if self._stopping and not raw.endswith("\n"):
                break                   # the sample stop()'s kill cut short
            line = raw.strip().lstrip("\ufeff")
            if not line:
                continue
            try:
                rows = json.loads(line)
                self.take(rows if isinstance(rows, list) else [rows])
            except (ValueError, KeyError, TypeError):
                self._mark(False)
                continue
            self._mark(True)

    def start(self):
        self._last = time.monotonic()       # the first gap counts from here
        # The root's start time first, while the shell is known to be up:
        # a later sample never takes a reused pid for it.
        root = [r for r in process_rows(f"ProcessId={int(self.root_pid)}") if r["pid"] == self.root_pid]
        if root:
            self.take(root)
        # The loop ends by itself once this process (pytest) is gone, so a
        # run that never reaches stop() (a deadline kill, Ctrl+C, a failed
        # teardown) leaves no sampler behind.
        loop = (f"[Console]::OutputEncoding = [Text.Encoding]::UTF8; $p = {os.getpid()}; while ($true) {{ "
                "if (-not (Get-Process -Id $p -ErrorAction SilentlyContinue)) { break }; "
                "[Console]::Out.WriteLine((" + _process_rows_script() + ")); [Console]::Out.Flush(); "
                f"Start-Sleep -Milliseconds {int(self.INTERVAL_S * 1000)} }}")
        self._proc = subprocess.Popen(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", loop],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, encoding="utf-8", errors="replace",
            creationflags=CREATE_NO_WINDOW)
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        return self

    def alive(self) -> bool:
        return any(pid_alive(pid) for pid in self.seen)

    def polls(self) -> dict:
        with self._lock:
            return {"ok": self.ok, "failed": self.failed, "max_gap_s": round(self.max_gap_s, 2),
                    "interval_s": self.INTERVAL_S}

    def stop(self) -> dict:
        self._stopping = True
        if self._proc is not None:
            ended_early = self._proc.poll() is not None
            self._proc.kill()
            self._proc.wait(30)
            if self._thread:
                self._thread.join(30)
            with self._lock:
                self.max_gap_s = max(self.max_gap_s, time.monotonic() - self._last)
                if ended_early:
                    self.failed += 1
        return self.seen


def browser_arguments(source: str) -> list:
    """The switches in WebEnvironment.cs's BrowserArguments: its string
    literals joined (they may be split with +), \\" unescaped, cut into
    --name or --name=value (a quoted value kept whole)."""
    expr = re.search(r"public const string BrowserArguments\s*=([^;]*);", source).group(1)
    text = "".join(m.replace('\\"', '"') for m in re.findall(r'"((?:[^"\\]|\\.)*)"', expr))
    return re.findall(r'--[a-z-]+(?:="[^"]*"|=\S+)?', text)


# ---------- the report ----------

def report(key: str, value) -> None:
    """Merge one measurement into $FOODOPT_L2_REPORT (JSON)."""
    path = os.environ.get("FOODOPT_L2_REPORT")
    if not path:
        return
    path = pathlib.Path(path)
    data = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            data = {}
    if isinstance(data.get(key), list) or key.endswith("_runs"):
        data.setdefault(key, []).append(value)
    else:
        data[key] = value
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path) -> str:
    """Lower-case hex sha256 of a file, read in chunks (installers are large)."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def report_installer_sha256() -> None:
    """Name the installer under test in the report, so the release tooling can
    tie the Level 2 run to the build. No-op unless both variables are set."""
    installer = os.environ.get("FOODOPT_L2_INSTALLER")
    if installer and os.environ.get("FOODOPT_L2_REPORT"):
        report("installer_sha256", sha256_file(installer))
