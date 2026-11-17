"""A launcher child: `child.py warmup` or `child.py serve <app> <flags…>`.

Run as `<python> -I -X utf8 -B child.py …`. -I keeps the projects folder
(the cwd) and the user's site-packages off sys.path, so nothing here is
imported from the user's files: the guard and serve_guard.py are loaded by
path from this folder, and nothing is added to sys.path.
"""
import importlib
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WARMUP_MODULES = ("torch", "botorch", "gpytorch", "pandas", "openpyxl")


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def ensure_guard():
    current = sys.modules.get("sitecustomize")
    if current is not None and getattr(current, "FOODOPT_GUARD", False):
        return current
    return _load("sitecustomize", "sitecustomize.py")


def warmup() -> int:
    """Import what the app imports, so its files are in the OS cache."""
    for name in WARMUP_MODULES:
        importlib.import_module(name)
    return 0


def serve(app: str, flags: list) -> int:
    port = next((f.split("=", 1)[1] for f in flags if f.startswith("--server.port=")), None)
    if not port:
        sys.stderr.write("serve needs --server.port\n")
        return 1
    guard = _load("foodopt_serve_guard", "serve_guard.py")
    guard.install_host_check(f"127.0.0.1:{port}")
    guard.ignore_user_config_files()
    sys.argv = ["streamlit", "run", app, *flags]
    from streamlit.web import cli
    return cli.main(prog_name="streamlit")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ensure_guard()
    if argv[:1] == ["warmup"]:
        return warmup()
    if argv[:1] == ["serve"] and len(argv) >= 2:
        return serve(argv[1], argv[2:])
    sys.stderr.write("usage: child.py warmup | serve <app> <flags...>\n")
    return 2


if __name__ == "__main__":
    sys.exit(main())
