"""desktop/windows/build_installer.ps1 and fetch_pinned.ps1 as text (Level 1,
any OS): Windows PowerShell 5.1-safe, strict, with every refusal of 3a's
build_dmg.sh, and no app file list of its own. Their behaviour is tested in
the VM (desktop/windows/tests)."""
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parent.parent
WINDOWS = REPO / "desktop" / "windows"
BUILD = WINDOWS / "build_installer.ps1"
FETCH = WINDOWS / "fetch_pinned.ps1"

REFUSALS = (
    "refusing to build: desktop/VERSION must hold one X.Y.Z line",
    "refusing to build: version $Version disagrees with desktop/VERSION ($ver)",
    "refusing to build: uncommitted changes (build from a clean worktree at the tag)",
    "refusing to build: HEAD is not tag v$ver (tag first: git tag -a v$ver $short)",
    "refusing to build: bundle_manifest.txt has Windows line endings",
    "refusing to build: bundle_manifest.txt is missing",
    "refusing to build: bundle_manifest.txt is empty",
    "refusing to build: bundle_manifest.txt entry must be a repo-relative path: $entry",
    "refusing to build: listed in bundle_manifest.txt but missing from the repo: $entry",
    "refusing to build: needs Windows PowerShell 5.1 or later",
    "refusing to build: $tool not found on PATH",
    "refusing to build: image folder is $n characters; at most 80 (clone to a shorter path)",
    "refusing to build: $Image is not an earlier build (it holds $held); choose another -OutDir",
    "usage: build_installer.ps1 [-Version X.Y.Z] [-Check] [-TestBuild] [-OutDir DIR]",
)


def _strict(path):
    raw = path.read_bytes()
    assert all(b < 0x80 for b in raw), f"{path.name} must be ASCII (5.1 reads it as ANSI)"
    text = raw.decode("ascii")
    for line in ("Set-StrictMode -Version 2", "$ErrorActionPreference = 'Stop'",
                 "$ProgressPreference = 'SilentlyContinue'", "Tls12"):
        assert line in text, line
    assert not [line for line in text.splitlines() if "2>&1" in line]
    return text


def _refusal_texts(text):
    """The script's refusal strings with PowerShell's quoting taken off:
    'a ' + $x + ' b' and "a $x b" both read as `a $x b`."""
    return re.sub(r"'\s*\+\s*|\s*\+\s*'", "", text)


def test_build_script_is_ascii_and_strict():
    _strict(BUILD)


def test_build_script_has_every_refusal():
    text = _refusal_texts(_strict(BUILD))
    for message in REFUSALS:
        assert message in text, message


def test_build_script_prints_the_unsigned_banner_and_record():
    text = _strict(BUILD)
    assert "UNSIGNED build - internal testing only." in text
    assert "TEST BUILD - not for release" in text
    assert "build failed: " in text


def test_build_script_lists_no_app_files():
    text = _strict(BUILD)
    allowed = {"build_env.py", "packaged_app_checks.py", "sitecustomize.py"}
    assert set(re.findall(r"\w+\.py\b", text)) <= allowed
    assert "bundle_manifest.txt" in text


def test_fetch_script_is_ascii_and_strict():
    _strict(FETCH)


def test_build_script_never_filters_a_literal_path_with_include():
    """Windows PowerShell 5.1 ignores -Include next to -LiteralPath, so the
    unsigned list once held every unsigned file (LICENSE.txt included)."""
    for line in _strict(BUILD).splitlines():
        assert not ("-LiteralPath" in line and "-Include" in line), line
