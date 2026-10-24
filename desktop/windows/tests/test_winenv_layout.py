"""The built image (build_installer.ps1 -TestBuild): the golden layout, the
shared guard, nothing stripped left, unchecked-hash .pyc everywhere, the
long-path gate, vendor-signed .exe files, and a build record and unsigned
list that describe this image."""
import hashlib
import random
import subprocess

import pytest

from conftest import (IMAGE, PUBLISH, RECORD, RECORD_KEYS, REPO, SITE, UNSIGNED, WINDOWS,
                      authenticode, binaries, build_env, needs_image, own_git_env, record)

pytestmark = needs_image


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_image_matches_the_golden_listing():
    golden = (WINDOWS / "image_files.txt").read_text(encoding="utf-8").splitlines()
    assert build_env.image_listing(IMAGE, exclude=build_env.shell_listing(PUBLISH)) == golden


def test_sitecustomize_is_the_shared_guard():
    repo_copy = (REPO / "desktop" / "shared" / "launcher" / "sitecustomize.py").read_bytes()
    assert (SITE / "sitecustomize.py").read_bytes() == repo_copy
    assert (IMAGE / "shared" / "launcher" / "sitecustomize.py").read_bytes() == repo_copy
    assert (SITE / build_env.PTH_NAME).read_bytes() == build_env.PTH_TEXT.encode("utf-8")


def test_nothing_stripped_remains():
    assert build_env.strip_violations(IMAGE) == []


def test_every_py_has_an_unchecked_hash_pyc():
    assert build_env.pyc_gaps(IMAGE) == []


def test_long_path_gate_passes():
    assert build_env.long_path_violations(IMAGE) == []
    longest = build_env.longest_relative(IMAGE)[0]
    assert record()["longest_path_chars"] == str(longest)
    assert longest <= build_env.MAX_RELATIVE


def test_every_exe_is_vendor_signed():
    exes = [p for p in binaries() if p.suffix.lower() == ".exe"
            and p != IMAGE / build_env.SHELL_EXE]       # ours: phase 6 signs it
    assert exes
    assert (IMAGE / build_env.SHELL_EXE).is_file()
    signatures = authenticode(exes)
    bad = {p: s for p, s in signatures.items()
           if s[0] != "Valid" or s[1] not in ("Python Software Foundation", "Microsoft Corporation")}
    assert bad == {}


def test_build_record_matches_the_image():
    rec = record()
    assert tuple(rec) == RECORD_KEYS
    text, files, size = build_env.file_manifest(IMAGE)
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == rec["image_sha256"]
    assert (rec["image_files"], rec["image_bytes"]) == (str(files), str(size))
    assert rec["windows_lock_sha256"] == _sha256(REPO / "desktop" / "requirements-windows.lock.txt")
    assert rec["mac_lock_sha256"] == _sha256(REPO / "desktop" / "requirements.lock.txt")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True,
                          check=True, env=own_git_env()).stdout.strip()
    assert rec["commit"] == head
    assert rec["version"] == (REPO / "desktop" / "VERSION").read_text(encoding="utf-8").strip()
    assert rec["python_version"] == "3.13.7"
    assert rec["signed"] == "no"
    assert rec["shell_files"] == str(len(build_env.shell_listing(PUBLISH)))
    assert rec["dotnet_sdk"].startswith("10.")
    assert RECORD.read_bytes().endswith(b"\n") and b"\r" not in RECORD.read_bytes()


def test_unsigned_list_is_what_phase_6_signs():
    lines = UNSIGNED.read_text(encoding="utf-8").splitlines()
    assert lines and lines == sorted(lines)
    assert [p for p in lines if p.lower().endswith(".exe")] == [build_env.SHELL_EXE]
    for path in lines:
        folder = path.rpartition("/")[0]
        assert folder not in ("python", "python/DLLs"), path
        assert (IMAGE / path).is_file(), path
    sample = random.Random(0).sample(lines, min(5, len(lines)))
    statuses = authenticode([IMAGE / p for p in sample])
    assert not [p for p, s in statuses.items() if s[0] == "Valid"]
    assert len(lines) == int(record()["unsigned_binaries"])


def test_the_only_unsigned_shell_files_are_ours():
    shell = set(build_env.shell_listing(PUBLISH))
    assert shell
    lines = UNSIGNED.read_text(encoding="utf-8").splitlines()
    assert [p for p in lines if p in shell and not p.startswith("FoodOptimizer")] == []


@pytest.mark.parametrize("name", ["FoodOptimizer.files.txt"])
def test_the_file_manifest_is_written_beside_the_image(name):
    path = IMAGE.parent / name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record()["image_sha256"]
