"""Microsoft's C++ runtime beside python.exe: one Microsoft-signed set of one
version, not older than any copy inside the wheels, and the one the process
loads whatever the import order, even with an older msvcp140.dll on PATH or
in the current folder (Review Focus 3; spike B4)."""
import ctypes
import json
import os
import pathlib
import shutil

import pytest

from conftest import IMAGE, PY, SITE, WINDOWS, authenticode, build_env, needs_image, record, run_image

pytestmark = needs_image
PINS = json.loads((WINDOWS / "pins.json").read_text(encoding="utf-8"))
VC_FILES = PINS["vc_runtime"]["files"]
PROBE = pathlib.Path(__file__).resolve().parent / "probes" / "dll_probe.py"


def file_version(path) -> str:
    """FileVersion from the fixed version resource, as "a.b.c.d"."""
    from ctypes import wintypes
    ver = ctypes.WinDLL("version")
    size = ver.GetFileVersionInfoSizeW(str(path), None)
    assert size, path
    buf = ctypes.create_string_buffer(size)
    assert ver.GetFileVersionInfoW(str(path), 0, size, buf)
    ptr, length = ctypes.c_void_p(), wintypes.UINT()
    assert ver.VerQueryValueW(buf, "\\", ctypes.byref(ptr), ctypes.byref(length))
    ms, ls = (ctypes.c_uint32 * 13).from_address(ptr.value)[2:4]
    return f"{ms >> 16}.{ms & 0xFFFF}.{ls >> 16}.{ls & 0xFFFF}"


def _key(version):
    return tuple(int(part) for part in version.split("."))


def test_vc_runtime_is_one_microsoft_signed_set():
    paths = [IMAGE / "python" / name for name in VC_FILES]
    for path in paths:
        assert path.is_file(), path
    signatures = authenticode(paths)
    # Microsoft signs the runtime DLLs as its Software Compatibility Publisher.
    assert {s for s in signatures.values()} == {("Valid", PINS["vc_runtime"]["dll_signer"])}
    assert PINS["vc_runtime"]["dll_signer"] == "Microsoft Windows Software Compatibility Publisher"
    versions = {file_version(p) for p in paths}
    assert versions == {PINS["vc_runtime"]["version"]} == {record()["vc_runtime_version"]}
    assert record()["vc_redist_sha256"] == PINS["vc_runtime"]["sha256"]


def test_vc_runtime_is_not_older_than_any_copy_in_the_wheels():
    bundled = _key(PINS["vc_runtime"]["version"])
    names = {n.lower() for n in VC_FILES}
    copies = [p for p in SITE.rglob("*.dll") if p.name.lower() in names]
    for path in copies:
        assert _key(file_version(path)) <= bundled, (path, file_version(path))


def _probe(order, home, cwd=None, extra_env=None):
    r = run_image(PROBE, order, home=home, cwd=cwd, extra_env=extra_env)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def _assert_ours(loaded):
    assert loaded["msvcp140.dll"] is not None
    for name, path in loaded.items():
        if name.startswith("msvcp140") and path is not None:
            assert os.path.normcase(path) == os.path.normcase(str(IMAGE / "python" / name)), (name, path)


@pytest.mark.parametrize("order", ["torch,sklearn", "sklearn,torch"])
def test_bundled_runtime_is_loaded_in_either_order(order, tmp_path):
    _assert_ours(_probe(order, tmp_path))


def test_a_decoy_msvcp140_on_path_or_cwd_is_not_loaded(tmp_path):
    decoy = tmp_path / "decoy dir"
    decoy.mkdir()
    shutil.copy(IMAGE / "python" / "vcruntime140.dll", decoy / "msvcp140.dll")
    path = str(decoy) + os.pathsep + os.environ.get("PATH", "")
    _assert_ours(_probe("sklearn,torch", tmp_path, cwd=decoy, extra_env={"PATH": path}))


def test_every_binary_finds_its_vc_runtime():
    assert build_env.vc_runtime_gaps(IMAGE) == []


def test_pe_imports_reads_python_exe():
    assert {"python313.dll", "vcruntime140.dll"} <= build_env.pe_imports(PY)
