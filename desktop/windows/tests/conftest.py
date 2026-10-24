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
