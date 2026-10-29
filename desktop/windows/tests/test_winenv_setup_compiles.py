"""Level 1 on the VM: setup.iss compiles, without a warning, against a fake
image (two copies of notepad and a Start Here), and the installer carries
desktop/VERSION and the publisher (spec Testing, Level 1)."""
import json
import os
import pathlib
import shutil
import subprocess

from conftest import REPO, WINDOWS, powershell


def find_iscc() -> pathlib.Path | None:
    """ISCC.exe in build_installer.ps1's lookup order."""
    candidates = [os.environ.get("FOODOPT_ISCC") or "", shutil.which("ISCC.exe") or ""]
    for base, rel in (("LOCALAPPDATA", r"Programs\Inno Setup 6"),
                      ("ProgramFiles(x86)", "Inno Setup 6"), ("ProgramFiles", "Inno Setup 6")):
        if os.environ.get(base):
            candidates.append(str(pathlib.Path(os.environ[base]) / rel / "ISCC.exe"))
    for c in candidates:
        if c and pathlib.Path(c).is_file():
            return pathlib.Path(c)
    return None


def version_info(path) -> dict:
    """The file's version strings, trimmed: Inno writes its setup's strings
    into fixed-size resource slots padded with spaces."""
    r = powershell("-Command", f"(Get-Item -LiteralPath '{path}').VersionInfo | "
                   "Select-Object ProductVersion, FileVersion, CompanyName, ProductName | ConvertTo-Json")
    assert r.returncode == 0, r.stderr
    return {k: (v or "").strip() for k, v in json.loads(r.stdout).items()}


def test_setup_compiles_without_warnings(short_tmp):
    iscc = find_iscc()
    assert iscc, "Inno Setup 6.3 or later (ISCC.exe) not found"
    image = short_tmp / "image"
    notepad = pathlib.Path(os.environ["SystemRoot"]) / "System32" / "notepad.exe"
    (image / "python").mkdir(parents=True)
    shutil.copyfile(notepad, image / "FoodOptimizer.exe")
    shutil.copyfile(notepad, image / "python" / "python.exe")
    (image / "Start Here.txt").write_bytes(b"Start here.\n")
    out = short_tmp / "out"
    r = subprocess.run([str(iscc), "/Qp", f"/DImageDir={image}", "/DDiskFreeGB=2", f"/O{out}", "/Ftest",
                        str(WINDOWS / "setup.iss")], capture_output=True, text=True, errors="replace",
                       timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "warning" not in (r.stdout + r.stderr).lower(), r.stdout + r.stderr
    exe = out / "test.exe"
    assert exe.is_file()
    info = version_info(exe)
    version = (REPO / "desktop" / "VERSION").read_text(encoding="utf-8").strip()
    assert info["ProductVersion"] == version
    assert info["FileVersion"] == version + ".0"
    assert info["CompanyName"] == "Food System Innovations"
    assert info["ProductName"] == "Food Optimizer"
