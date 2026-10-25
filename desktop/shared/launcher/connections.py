"""How many TCP connections a port has: the idle watchdog's question. On
Windows, GetExtendedTcpTable (no admin, no psutil); elsewhere, lsof, as
launcher.sh asks it."""
import ctypes
import socket
import subprocess
import sys
from ctypes import wintypes

AF_INET = 2
TCP_TABLE_OWNER_PID_ALL = 5
MIB_TCP_STATE_ESTAB = 5
ERROR_INSUFFICIENT_BUFFER = 122
NO_ERROR = 0


def _windows_rows():
    iphlpapi = ctypes.WinDLL("iphlpapi", use_last_error=True)
    fn = iphlpapi.GetExtendedTcpTable
    fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD), wintypes.BOOL,
                   wintypes.ULONG, ctypes.c_int, wintypes.ULONG]
    fn.restype = wintypes.DWORD
    size = wintypes.DWORD(0)
    rc = fn(None, ctypes.byref(size), False, AF_INET, TCP_TABLE_OWNER_PID_ALL, 0)
    for _ in range(5):
        if rc not in (ERROR_INSUFFICIENT_BUFFER, NO_ERROR):
            raise OSError(rc, "GetExtendedTcpTable failed")
        buf = ctypes.create_string_buffer(max(size.value, 4))
        rc = fn(buf, ctypes.byref(size), False, AF_INET, TCP_TABLE_OWNER_PID_ALL, 0)
        if rc == NO_ERROR:
            break
    else:
        raise OSError(rc, "GetExtendedTcpTable failed")
    count = ctypes.c_uint32.from_buffer_copy(buf.raw[:4]).value
    dwords = (ctypes.c_uint32 * (1 + 6 * count)).from_buffer_copy(buf.raw[:4 * (1 + 6 * count)])
    for i in range(count):
        state, _laddr, lport, _raddr, rport, _pid = dwords[1 + 6 * i: 7 + 6 * i]
        yield state, socket.ntohs(lport & 0xFFFF), socket.ntohs(rport & 0xFFFF)


def established_count(port: int) -> int:
    """ESTABLISHED IPv4 connections whose local or remote port is `port`.
    When the count cannot be read it is 1: unknown counts as connected, so
    the server is never stopped under an open window."""
    if sys.platform == "win32":
        try:
            return sum(1 for state, lp, rp in _windows_rows()
                       if state == MIB_TCP_STATE_ESTAB and port in (lp, rp))
        except OSError:
            return 1
    try:
        out = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:ESTABLISHED"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return 1
    return sum(1 for line in out.stdout.splitlines() if "ESTABLISHED" in line)
