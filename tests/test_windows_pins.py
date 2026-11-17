"""The Windows build's pinned downloads (desktop/windows/pins.json): the same
Python patch as the Mac, https URLs with sha256 pins, and a VC runtime that
covers what sitecustomize preloads (phase 3b, spec §4)."""
import ast
import json
import pathlib
import re
import subprocess

REPO = pathlib.Path(__file__).resolve().parent.parent
WINDOWS = REPO / "desktop" / "windows"
PINS_PATH = WINDOWS / "pins.json"


def _pins():
    return json.loads(PINS_PATH.read_text(encoding="utf-8"))


def _msvcp_dlls():
    """MSVCP_DLLS from sitecustomize.py, read by AST: importing it would
    install the network guard in the test process."""
    src = (REPO / "desktop" / "shared" / "launcher" / "sitecustomize.py").read_text(encoding="utf-8")
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "MSVCP_DLLS" for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("MSVCP_DLLS not found in sitecustomize.py")


def test_python_pin_equals_the_mac_pin():
    text = (REPO / "desktop" / "launcher.sh").read_text(encoding="utf-8")
    mac = re.search(r'^PYTHON_VERSION="([^"]+)"', text, re.M).group(1)
    python = _pins()["python"]
    assert python["version"] == mac
    assert python["url"].endswith("/python/" + mac)


def test_pins_are_https_and_sha256():
    pins = _pins()
    for key in ("python", "vc_runtime"):
        entry = pins[key]
        assert entry["url"].startswith("https://"), key
        assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]), key
        for field in ("version", "url", "file", "sha256", "signer"):
            assert entry[field], (key, field)
    host = pins["vc_runtime"]["url"].split("/")[2]
    # A redirect (aka.ms) is not a pin: the URL must be the resolved one.
    assert host == "download.visualstudio.microsoft.com"
    assert pins["python"]["signer"] == "Python Software Foundation"
    assert pins["vc_runtime"]["signer"] == "Microsoft Corporation"
    assert pins["vc_runtime"]["file"] == f"VC_redist.x64-{pins['vc_runtime']['version']}.exe"
    assert pins["python"]["file"] == f"python.{pins['python']['version']}.nupkg"


def test_wix_pin_is_a_wixtoolset_release_with_dark():
    """The VC redistributable's Burn container is unpacked with WiX's
    dark.exe -x (owner, 2026-10-24: /layout no longer lays out the MSIs).
    dark.exe 3.14 is not Authenticode-signed, so the zip's sha256 is the pin."""
    wix = _pins()["wix"]
    assert wix["url"].startswith("https://github.com/wixtoolset/wix3/releases/download/")
    assert wix["url"].endswith("/wix314-binaries.zip")
    assert re.fullmatch(r"[0-9a-f]{64}", wix["sha256"])
    assert re.fullmatch(r"3\.14\.\d+\.\d+", wix["version"])
    assert wix["file"] == f"wix-{wix['version']}-binaries.zip"
    assert "signer" not in wix


def test_vc_dlls_are_pinned_to_microsofts_runtime_signer():
    """The redistributable and its MSI are signed CN=Microsoft Corporation;
    the runtime DLLs inside it CN=Microsoft Windows Software Compatibility
    Publisher (O=Microsoft Corporation), as seen on VC_redist 14.51.36247."""
    assert _pins()["vc_runtime"]["dll_signer"] == "Microsoft Windows Software Compatibility Publisher"


def test_vc_files_cover_what_sitecustomize_preloads():
    expected = set(_msvcp_dlls()) | {"vcruntime140.dll", "vcruntime140_1.dll"}
    assert set(_pins()["vc_runtime"]["files"]) == expected


def test_vc_version_is_at_least_sklearns():
    version = _pins()["vc_runtime"]["version"]
    assert re.fullmatch(r"\d+\.\d+\.\d+\.\d+", version)
    # Spike B4: sklearn's wheel ships msvcp140.dll 14.51.36231.
    assert tuple(map(int, version.split(".")))[:3] >= (14, 51, 36231)


def test_pins_file_is_lf_and_two_space_indented():
    raw = PINS_PATH.read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")
    assert raw.decode("utf-8") == json.dumps(_pins(), indent=2) + "\n"


def test_build_folder_is_ignored():
    lines = (WINDOWS / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "/build/" in lines
    for path in ("desktop/windows/.cache/x", "desktop/windows/build/x"):
        rc = subprocess.run(["git", "check-ignore", "-q", path], cwd=REPO).returncode
        assert rc == 0, path
