"""The bundled Python: python.org 3.13.7 x64 with its PSF signatures, exactly
the Windows lock, isolated from the working folder (-I keeps a stray
pandas.py off sys.path), and everything the app imports loads. Running it
never writes into the image (the program folder is read-only at run time)."""
import hashlib
import json
import pathlib
import sys

from conftest import (IMAGE, PY, REPO, SITE, WINDOWS, authenticode, build_env, needs_image,
                      record, run_image)

pytestmark = needs_image
sys.path.insert(0, str(REPO))
from tests.test_locks import pins  # noqa: E402

VC_FILES = {n.lower() for n in json.loads((WINDOWS / "pins.json").read_text(encoding="utf-8"))
            ["vc_runtime"]["files"]}
MANIFEST_MODULES = [e[:-3] for e in build_env.app_files(REPO) if e.endswith(".py") and "/" not in e]


def _norm(path):
    return str(pathlib.Path(path).resolve()).lower()


def test_bundled_python_is_python_org_3_13_7_x64(tmp_path):
    r = run_image("-c", "import struct,sys,sysconfig;"
                        "print(tuple(sys.version_info[:3]));print(sysconfig.get_platform());"
                        "print(struct.calcsize('P'))", home=tmp_path)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["(3,", "13,", "7)", "win-amd64", "8"]


def test_python_org_binaries_keep_their_signatures():
    folders = [IMAGE / "python", IMAGE / "python" / "DLLs"]
    files = [p for folder in folders for p in folder.iterdir()
             if p.is_file() and p.suffix.lower() in (".exe", ".dll", ".pyd")
             and p.name.lower() not in VC_FILES]
    assert any(p.name == "python.exe" for p in files)
    bad = {p: s for p, s in authenticode(files).items()
           if s != ("Valid", "Python Software Foundation")}
    assert bad == {}


def test_installed_packages_are_the_windows_lock(tmp_path):
    code = ("import importlib.metadata as m,json,re;"
            "print(json.dumps({re.sub(r'[-_.]+','-',d.metadata['Name'].lower()): d.version"
            " for d in m.distributions()}))")
    r = run_image("-c", code, home=tmp_path)
    assert r.returncode == 0, r.stderr
    installed = json.loads(r.stdout)
    assert installed == pins(REPO / "desktop" / "requirements-windows.lock.txt")


def test_isolated_start_finds_guard_app_and_launcher(tmp_path):
    projects = tmp_path / "FoodOptimizer"
    projects.mkdir()
    for name in ("pandas.py", "launcher.py", "food_bo.py"):
        (projects / name).write_text("raise SystemExit('hostile module imported')\n", encoding="utf-8")
    modules = ["launcher", "pandas"] + [m for m in MANIFEST_MODULES if m != "app"]
    code = ("import importlib.util,json,sitecustomize;"
            f"mods={modules!r};"
            "print(json.dumps({'sitecustomize':sitecustomize.__file__,"
            "'guard':getattr(sitecustomize,'FOODOPT_GUARD',None),"
            "**{m:importlib.util.find_spec(m).origin for m in mods}}))")
    r = run_image("-c", code, home=tmp_path, cwd=projects)
    assert r.returncode == 0, r.stderr
    found = json.loads(r.stdout)
    assert _norm(found["sitecustomize"]) == _norm(SITE / "sitecustomize.py")
    assert found["guard"] is True
    assert _norm(found["launcher"]) == _norm(IMAGE / "shared" / "launcher" / "__init__.py")
    assert _norm(found["pandas"]).startswith(_norm(SITE) + "\\")
    for module in modules[2:]:
        assert _norm(found[module]) == _norm(IMAGE / "app" / f"{module}.py"), module


def test_everything_imports(tmp_path):
    modules = ["torch", "botorch", "gpytorch", "pandas", "openpyxl", "streamlit", "sklearn",
               "scipy", "pyarrow", "numpy", "launcher", "launcher.cli"]
    modules += [m for m in MANIFEST_MODULES if m != "app"]
    code = "import importlib\n" + "".join(f"importlib.import_module({m!r})\n" for m in modules) \
        + "print('IMPORT_OK')\n"
    r = run_image("-c", code, home=tmp_path, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert "IMPORT_OK" in r.stdout
    assert "SyntaxWarning" not in r.stderr
    assert "Traceback" not in r.stderr


def test_launcher_module_runs_from_the_image(tmp_path):
    r = run_image("-m", "launcher", "--help", home=tmp_path, cwd=tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr


def test_the_image_is_not_written_by_running_it():
    text, _files, _size = build_env.file_manifest(IMAGE)
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == record()["image_sha256"]
