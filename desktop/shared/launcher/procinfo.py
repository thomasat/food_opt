"""Processes, without psutil: is a pid alive, which program is it, when did
it start, and stopping it. Windows uses kernel32 through ctypes and never
os.kill(pid, 0), which there sends CTRL_C_EVENT."""
import ctypes
import os
import signal
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path
from typing import Optional

PROCESS_TERMINATE = 0x0001
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
SYNCHRONIZE = 0x00100000
STILL_ACTIVE = 259
ERROR_ACCESS_DENIED = 5
_EPOCH_AS_FILETIME = 116444736000000000   # 1970-01-01 in 100 ns ticks since 1601

_kernel32 = None


def _k32():
    global _kernel32
    if _kernel32 is None:
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        k.OpenProcess.restype = wintypes.HANDLE
        k.CloseHandle.argtypes = [wintypes.HANDLE]
        k.CloseHandle.restype = wintypes.BOOL
        k.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        k.GetExitCodeProcess.restype = wintypes.BOOL
        k.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD,
                                                 wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
        k.QueryFullProcessImageNameW.restype = wintypes.BOOL
        k.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        k.GetProcessTimes.restype = wintypes.BOOL
        k.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        k.TerminateProcess.restype = wintypes.BOOL
        k.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k.WaitForSingleObject.restype = wintypes.DWORD
        _kernel32 = k
    return _kernel32


def _open(pid: int, access: int):
    if pid <= 0:
        return None
    return _k32().OpenProcess(access, False, pid) or None


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        h = _open(pid, PROCESS_QUERY_LIMITED_INFORMATION)
        if h is None:
            return ctypes.get_last_error() == ERROR_ACCESS_DENIED
        try:
            code = wintypes.DWORD()
            if not _k32().GetExitCodeProcess(h, ctypes.byref(code)):
                return False
            return code.value == STILL_ACTIVE
        finally:
            _k32().CloseHandle(h)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    # A child of ours that has exited stays a zombie until it is reaped; it
    # is not alive. WNOWAIT leaves it for its Popen to reap.
    try:
        if os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None:
            return False
    except (ChildProcessError, OSError, AttributeError):
        pass
    return True


def process_image(pid: int) -> Optional[str]:
    """The full path of the program `pid` runs, or None."""
    try:
        if sys.platform == "win32":
            h = _open(pid, PROCESS_QUERY_LIMITED_INFORMATION)
            if h is None:
                return None
            try:
                size = wintypes.DWORD(32768)
                buf = ctypes.create_unicode_buffer(size.value)
                if not _k32().QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
                    return None
                return buf.value or None
            finally:
                _k32().CloseHandle(h)
        if sys.platform == "darwin":
            libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
            libproc.proc_pidpath.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
            libproc.proc_pidpath.restype = ctypes.c_int
            buf = ctypes.create_string_buffer(4096)
            n = libproc.proc_pidpath(pid, buf, 4096)
            return os.fsdecode(buf.raw[:n]) if n > 0 else None
        return os.readlink(f"/proc/{pid}/exe")
    except (OSError, AttributeError, ValueError):
        return None


def process_start_time(pid: int) -> Optional[float]:
    """When `pid` started, in epoch seconds, or None."""
    try:
        if sys.platform == "win32":
            h = _open(pid, PROCESS_QUERY_LIMITED_INFORMATION)
            if h is None:
                return None
            try:
                times = [wintypes.FILETIME() for _ in range(4)]
                if not _k32().GetProcessTimes(h, *[ctypes.byref(t) for t in times]):
                    return None
                ft = (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
                return (ft - _EPOCH_AS_FILETIME) / 1e7
            finally:
                _k32().CloseHandle(h)
        if sys.platform == "darwin":
            out = subprocess.run(["/bin/ps", "-o", "lstart=", "-p", str(pid)],
                                 capture_output=True, text=True, timeout=5,
                                 env={**os.environ, "LC_ALL": "C"})
            text = out.stdout.strip()
            if out.returncode != 0 or not text:
                return None
            return time.mktime(time.strptime(text, "%a %b %d %H:%M:%S %Y"))
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return None


def terminate(pid: int, grace: float = 5.0) -> None:
    """Stop `pid`: SIGTERM, then SIGKILL after `grace` seconds (POSIX);
    TerminateProcess, then a wait of up to `grace` (Windows)."""
    if sys.platform == "win32":
        h = _open(pid, PROCESS_TERMINATE | SYNCHRONIZE)
        if h is None:
            return
        try:
            _k32().TerminateProcess(h, 1)
            _k32().WaitForSingleObject(h, int(grace * 1000))
        finally:
            _k32().CloseHandle(h)
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return
    end = time.monotonic() + grace
    while time.monotonic() < end:
        if not pid_alive(pid):
            return
        time.sleep(0.05)
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


def _norm(path) -> str:
    return os.path.normcase(os.path.realpath(path))


def is_our_server(pid: int, python: Path, port_file_mtime: float) -> bool:
    """True only for a process that runs our `python` and started no later
    than the port file was written (2 s tolerance): Windows reuses pids
    quickly, and a reused pid must never be stopped."""
    image = process_image(pid)
    if image is None or _norm(image) != _norm(python):
        return False
    start = process_start_time(pid)
    return start is not None and start <= port_file_mtime + 2
