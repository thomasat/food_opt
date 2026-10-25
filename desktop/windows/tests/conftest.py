"""Windows build-environment tests (phase 3b). They run in the build VM only:
everything here skips unless sys.platform == "win32". Helpers are imported by
the test files with `from conftest import ...` (this folder has no
__init__.py, so pytest puts it on sys.path)."""
import json
import os
import pathlib
import subprocess
import sys
import tempfile

import pytest

REPO = pathlib.Path(__file__).resolve().parents[3]
WINDOWS = REPO / "desktop" / "windows"
CREATE_NO_WINDOW = 0x08000000

slow = pytest.mark.skipif(os.environ.get("FOODOPT_SLOW_TESTS") != "1",
                          reason="set FOODOPT_SLOW_TESTS=1")


def pytest_collection_modifyitems(config, items):
    if sys.platform == "win32":
        return
    here = pathlib.Path(__file__).resolve().parent
    skip = pytest.mark.skip(reason="Windows build VM only")
    for item in items:
        if here in pathlib.Path(str(item.fspath)).resolve().parents:
            item.add_marker(skip)


def powershell(*args, env=None, timeout=600, cwd=None) -> subprocess.CompletedProcess:
    """Run Windows PowerShell 5.1 with a clean, non-interactive profile."""
    cmd = ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
           *[str(a) for a in args]]
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", env=env, timeout=timeout, cwd=cwd,
                          creationflags=CREATE_NO_WINDOW)


_AUTHENTICODE_PS1 = r"""
param([string]$InPath, [string]$OutPath)
$ErrorActionPreference = 'Stop'
$paths = [IO.File]::ReadAllLines($InPath, [Text.Encoding]::UTF8)
$out = @{}
foreach ($p in $paths) {
    if ($p -eq '') { continue }
    $s = Get-AuthenticodeSignature -LiteralPath $p
    $cn = ''
    if ($s.SignerCertificate -ne $null) {
        if ($s.SignerCertificate.Subject -match '^CN=("[^"]*"|[^,]*)') { $cn = $Matches[1].Trim('"') }
    }
    $out[$p] = @([string]$s.Status, $cn)
}
[IO.File]::WriteAllText($OutPath, ($out | ConvertTo-Json -Depth 3 -Compress), (New-Object Text.UTF8Encoding $false))
"""


def authenticode(paths) -> dict:
    """{str(path): (Status, signer CN)} for each path, in one PowerShell call."""
    paths = [str(p) for p in paths]
    if not paths:
        return {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        script, inp, outp = tmp / "sig.ps1", tmp / "in.txt", tmp / "out.json"
        script.write_text(_AUTHENTICODE_PS1, encoding="utf-8")
        inp.write_text("\n".join(paths) + "\n", encoding="utf-8")
        r = powershell("-File", script, inp, outp, timeout=1800)
        assert r.returncode == 0, r.stdout + r.stderr
        data = json.loads(outp.read_text(encoding="utf-8"))
    return {k: tuple(v) for k, v in data.items()}


def own_git_env(extra=None) -> dict:
    """os.environ without GIT_*: inside a git hook, GIT_DIR and GIT_INDEX_FILE
    would point every git call at the outer checkout."""
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("GIT_")}
    env.update(extra or {})
    return env


@pytest.fixture
def short_tmp():
    """A fresh folder with a space in its name, on a short path: pytest's
    tmp_path is already longer than the 80-character image-root budget."""
    import shutil
    base = pathlib.Path(os.environ.get("SystemDrive", "C:") + "\\") / "fo-t"
    base.mkdir(exist_ok=True)
    path = pathlib.Path(tempfile.mkdtemp(prefix="p3b ", dir=base))
    yield path
    shutil.rmtree(path, ignore_errors=True)


# ---------- the built image (Task 5) ----------

def _load_build_env():
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_env", WINDOWS / "build_env.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_env = _load_build_env()
IMAGE = pathlib.Path(os.environ.get("FOODOPT_WIN_IMAGE")
                     or REPO / "desktop" / "windows" / "build" / build_env.IMAGE_NAME)
PY = IMAGE / "python" / "python.exe"
# Phase 4's published shell (dotnet publish FoodOptimizer.Shell -c Release),
# which build_installer.ps1 copies to the image root.
PUBLISH = REPO / "desktop" / ".cache" / "shell" / "publish" / "FoodOptimizer.Shell" / "release_win-x64"
SITE = IMAGE / build_env.SITE
RECORD = IMAGE.parent / f"{build_env.IMAGE_NAME}.build.txt"
UNSIGNED = IMAGE.parent / f"{build_env.IMAGE_NAME}.unsigned.txt"
RECORD_KEYS = ("version", "commit", "tree_clean", "test_build", "mac_lock_sha256",
               "windows_lock_sha256", "python_version", "python_nupkg_sha256",
               "vc_runtime_version", "vc_redist_sha256", "uv_version", "image_files",
               "image_bytes", "image_sha256", "longest_path_chars", "unsigned_binaries",
               "dotnet_sdk", "shell_files", "signed")
needs_image = pytest.mark.skipif(not PY.is_file(), reason="build first: build_installer.ps1 -TestBuild")
_PROXY_KEYS = {"http_proxy", "https_proxy", "all_proxy", "no_proxy", "ftp_proxy"}


def image_env(home) -> dict:
    """The environment the launcher gives its children: no PYTHON*,
    STREAMLIT_* or proxy variables; GIT_PYTHON_REFRESH=quiet; home = home."""
    env = {k: v for k, v in os.environ.items()
           if not k.upper().startswith(("PYTHON", "STREAMLIT_")) and k.lower() not in _PROXY_KEYS}
    env["GIT_PYTHON_REFRESH"] = "quiet"
    env["HOME"] = env["USERPROFILE"] = str(home)
    return env


def run_image(*args, home, cwd=None, extra_env=None, timeout=900) -> subprocess.CompletedProcess:
    """Run the bundled python.exe as the launcher runs it: -I -X utf8 -B."""
    env = image_env(home)
    env.update(extra_env or {})
    return subprocess.run([str(PY), "-I", "-X", "utf8", "-B", *[str(a) for a in args]],
                          capture_output=True, text=True, encoding="utf-8", errors="replace",
                          cwd=cwd, env=env, timeout=timeout, creationflags=CREATE_NO_WINDOW)


def record() -> dict:
    """FoodOptimizer.build.txt as {key: value}, in file order."""
    out = {}
    for line in RECORD.read_text(encoding="utf-8").split("\n"):
        if line:
            key, _, value = line.partition("=")
            out[key] = value
    return out


def binaries(under=IMAGE):
    """Every .exe, .dll and .pyd under a folder."""
    return sorted(p for p in pathlib.Path(under).rglob("*")
                  if p.suffix.lower() in (".exe", ".dll", ".pyd") and p.is_file())


# ---------- Level 2: the packaged app, installed (phase 5) ----------

L2_BASE = pathlib.Path(os.environ.get("SystemDrive", "C:") + "\\") / "FOL2"
L2_USER_FOLDER = "Jos\u00e9 M\u00fcller"          # spaces and accents (Review Focus 5)
_L2_SKIP = "set FOODOPT_L2_INSTALLER (run_level2.ps1)"
_home_ids = iter(range(1, 10 ** 6))


class L2Home:
    """A user folder for one test: state and projects as the shell and the
    installer see them through FOODOPT_STATE_DIR / FOODOPT_PROJECTS_DIR,
    and an install root beside them. Short ids keep the root well inside
    the 80-character budget."""

    def __init__(self, tag: str):
        self.base = L2_BASE / tag
        self.home = self.base / L2_USER_FOLDER
        self.state = self.home / "AppData" / "Local" / "FoodOptimizer"
        self.projects = self.home / "FoodOptimizer"
        self.root = self.home / "Food Optimizer"
        self.env = {"FOODOPT_STATE_DIR": str(self.state), "FOODOPT_PROJECTS_DIR": str(self.projects)}

    def make(self):
        import shutil
        L2_BASE.mkdir(exist_ok=True)
        user = os.environ.get("USERNAME", "")
        if user:
            subprocess.run(["icacls", str(L2_BASE), "/grant", f"{user}:(OI)(CI)M"], capture_output=True,
                           creationflags=CREATE_NO_WINDOW)
        if self.base.exists():
            shutil.rmtree(self.base, ignore_errors=True)
        self.home.mkdir(parents=True)
        return self


def _l2():
    import l2_support
    return l2_support


@pytest.fixture(scope="session")
def installer():
    path = os.environ.get("FOODOPT_L2_INSTALLER")
    if not path:
        pytest.skip(_L2_SKIP)
    return pathlib.Path(path)


@pytest.fixture(scope="session")
def dist_record(installer) -> dict:
    """The release record beside the installer (<name>.build.txt)."""
    out = {}
    for line in installer.with_suffix(".build.txt").read_text(encoding="utf-8").split("\n"):
        if line:
            key, _, value = line.partition("=")
            out[key] = value
    return out


@pytest.fixture
def l2_home(request):
    return L2Home(f"t{next(_home_ids):02d}").make()


def _ours(location: str) -> bool:
    return bool(location) and os.path.normcase(location).startswith(os.path.normcase(str(L2_BASE)) + "\\")


def ensure_clean_machine():
    """No Food Optimizer in either scope and none running. Level 2's own
    installs (under C:\\FOL2) are uninstalled; anything else only with
    FOODOPT_L2_CLEANUP=1 (run_level2.ps1 -Cleanup)."""
    l2 = _l2()
    for scope, kind in (("user", "user"), ("machine", "system")):
        entry = l2.uninstall_entry(scope)
        if entry is None:
            continue
        location = entry.get("InstallLocation", "")
        if not (_ours(location) or os.environ.get("FOODOPT_L2_CLEANUP") == "1"):
            pytest.fail("Food Optimizer is installed on this PC; rerun with -Cleanup")
        if location:
            l2.kill_under(location)
            if pathlib.Path(location, "unins000.exe").is_file():
                l2.uninstall(location, kind)
        if l2.uninstall_entry(scope) is not None:
            pytest.fail(f"could not uninstall the {scope} copy at {location}")
    for root in (L2_BASE, pathlib.Path(os.environ["LOCALAPPDATA"]) / "Programs" / "Food Optimizer",
                 pathlib.Path(os.environ["ProgramFiles"]) / "Food Optimizer"):
        running = l2.processes_under(root)
        if running and os.environ.get("FOODOPT_L2_CLEANUP") == "1":
            l2.kill_under(root)
            running = l2.processes_under(root)
        if running:
            pytest.fail(f"Food Optimizer is running from {root}: {running}; rerun with -Cleanup")


@pytest.fixture
def clean_machine(installer):
    ensure_clean_machine()
    yield


class PerUser:
    def __init__(self, home, result, manifest):
        self.home, self.result, self.manifest = home, result, manifest

    @property
    def root(self):
        return self.home.root


@pytest.fixture(scope="session")
def per_user(installer):
    """One silent per-user install, as the limited user, shared by modules
    10-45; its file manifest is taken right after the install."""
    l2 = _l2()
    ensure_clean_machine()
    home = L2Home("s").make()
    result = l2.install(installer, "user", dir=home.root, env=home.env)
    man = l2.manifest(home.root) if result.rc == 0 and home.root.is_dir() else ""
    yield PerUser(home, result, man)
    entry = l2.uninstall_entry("user")
    if entry and _ours(entry.get("InstallLocation", "")):
        l2.kill_under(home.root)
        l2.uninstall(home.root, "user", env=home.env)
