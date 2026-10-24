"""Food Optimizer's network guard, the first thing every child imports.

  1. On Windows, load the bundled msvcp140*.dll by full path, before torch or
     scikit-learn can load another copy by name (spike B4).
  2. Turn urllib's proxies off: Windows' registry proxy otherwise catches
     http://127.0.0.1 URLs.
  3. Stub Streamlit's internal/external IP lookups, which reach 8.8.8.8 and
     checkip.amazonaws.com when a foreign origin opens a websocket.

Stdlib only and standalone (no package-relative imports): child.py loads it
by path, and the bundled Python also has it at the top of sys.path.
"""
import importlib.abc
import os
import sys

FOODOPT_GUARD = True
MSVCP_DLLS = ("msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll",
              "msvcp140_atomic_wait.dll", "msvcp140_codecvt_ids.dll")
_NET_UTIL = "streamlit.net_util"
_installed = False


def preload_cpp_runtime(platform: str, executable: str, loader) -> list:
    """Load each bundled C++ runtime DLL beside `executable`, in order, by
    full path. Only on win32; returns the paths loaded."""
    if platform != "win32":
        return []
    here = os.path.dirname(os.path.abspath(executable))
    loaded = []
    for name in MSVCP_DLLS:
        path = os.path.join(here, name)
        if os.path.isfile(path):
            loader(path)
            loaded.append(path)
    return loaded


def _no_ip():
    return None


def _stub_net_util(module) -> None:
    module.get_internal_ip = _no_ip
    module.get_external_ip = _no_ip


class _StubbingLoader(importlib.abc.Loader):
    def __init__(self, loader):
        self._loader = loader

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        self._loader.exec_module(module)
        _stub_net_util(module)


class _NetUtilFinder(importlib.abc.MetaPathFinder):
    """Stubs streamlit.net_util the moment it is imported."""

    def find_spec(self, name, path, target=None):
        if name != _NET_UTIL:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is not None and spec.loader is not None:
                spec.loader = _StubbingLoader(spec.loader)
                return spec
        return None


def install(platform=None, executable=None) -> None:
    """Install the guard once per process."""
    global _installed
    if _installed:
        return
    _installed = True
    platform = sys.platform if platform is None else platform
    executable = sys.executable if executable is None else executable
    if platform == "win32":
        import ctypes
        preload_cpp_runtime(platform, executable, ctypes.WinDLL)
    import urllib.request
    urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
    if _NET_UTIL in sys.modules:
        _stub_net_util(sys.modules[_NET_UTIL])
    elif not any(isinstance(f, _NetUtilFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _NetUtilFinder())


try:
    install()
except Exception as _err:   # never stop the app from starting
    try:
        sys.stderr.write(f"foodopt guard: {_err!r}\n")
    except Exception:
        pass
