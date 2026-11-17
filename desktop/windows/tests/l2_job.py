"""Level 2's job runner. run_as() starts it from a scheduled task, as the
limited user (/IT) or as SYSTEM:

    pythonw l2_job.py <job.json>

job.json holds {"argv", "env", "cwd", "result"}; env holds extras laid over
this process's own environment. It writes the result file
at once with {"pid", "elevated", "started"}, then adds {"rc", "ended"} when
the child exits. Stdlib only; it never imports the app.
"""
import ctypes
import json
import os
import subprocess
import sys
import time
from ctypes import wintypes


def _elevated() -> bool:
    """GetTokenInformation(TokenElevation) of this process's token."""
    k32, adv = ctypes.windll.kernel32, ctypes.windll.advapi32
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    adv.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
    adv.GetTokenInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
                                        ctypes.POINTER(wintypes.DWORD)]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    token = wintypes.HANDLE()
    if not adv.OpenProcessToken(k32.GetCurrentProcess(), 0x0008, ctypes.byref(token)):   # TOKEN_QUERY
        return False
    try:
        value = wintypes.DWORD()
        size = wintypes.DWORD()
        if not adv.GetTokenInformation(token, 20, ctypes.byref(value), ctypes.sizeof(value),
                                       ctypes.byref(size)):                             # TokenElevation
            return False
        return bool(value.value)
    finally:
        k32.CloseHandle(token)


def _write(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def main(spec_path: str) -> int:
    with open(spec_path, encoding="utf-8-sig") as f:
        spec = json.load(f)
    result = spec["result"]
    try:
        env = dict(os.environ)          # this identity's own environment
        env.update(spec.get("env") or {})
        proc = subprocess.Popen(spec["argv"], env=env, cwd=spec.get("cwd") or None)
    except OSError as e:
        _write(result, {"pid": 0, "elevated": _elevated(), "started": time.time(),
                        "rc": -1, "ended": time.time(), "error": str(e)})
        return 1
    info = {"pid": proc.pid, "elevated": _elevated(), "started": time.time()}
    _write(result, info)
    rc = proc.wait()
    info.update({"rc": rc, "ended": time.time()})
    _write(result, info)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
