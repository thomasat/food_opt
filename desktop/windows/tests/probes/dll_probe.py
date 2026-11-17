"""Which C++ runtime DLLs does this process use after importing the given
modules in order? Prints one JSON object: DLL name -> full path, or null.
Spike B4's b4_dll.py, with the result as JSON. Run with the bundled Python:
    python.exe -I -X utf8 -B dll_probe.py torch,sklearn
"""
import ctypes
import json
import sys
from ctypes import wintypes

import sitecustomize

NAMES = tuple(sitecustomize.MSVCP_DLLS) + ("vcruntime140.dll", "vcruntime140_1.dll")


def loaded(name):
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetModuleHandleW.restype = wintypes.HMODULE
    k32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
    handle = k32.GetModuleHandleW(name)
    if not handle:
        return None
    buf = ctypes.create_unicode_buffer(32768)
    k32.GetModuleFileNameW(wintypes.HMODULE(handle), buf, len(buf))
    return buf.value


def main():
    for module in sys.argv[1].split(","):
        __import__(module)
    print(json.dumps({name: loaded(name) for name in NAMES}))


if __name__ == "__main__":
    main()
