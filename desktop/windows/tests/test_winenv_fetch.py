"""fetch_pinned.ps1: a pinned download is verified on every run, and a
corrupted or tampered cache file is removed and refused (Review Focus 5)."""
import hashlib
import json
import pathlib
import shutil

from conftest import WINDOWS, powershell, slow

SCRIPT = WINDOWS / "fetch_pinned.ps1"
PINS = json.loads((WINDOWS / "pins.json").read_text(encoding="utf-8"))
NAME = "python"   # the smaller of the two pinned files


def _fetch(cache, name=NAME, script=SCRIPT):
    return powershell("-File", script, "-Name", name, "-CacheDir", cache)


def _sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


@slow
def test_fetch_downloads_and_verifies(tmp_path):
    cache = tmp_path / "cache dir"
    first = _fetch(cache)
    assert first.returncode == 0, first.stderr
    path = pathlib.Path(first.stdout.strip())
    assert path == cache / PINS[NAME]["file"]
    assert _sha256(path) == PINS[NAME]["sha256"]
    second = _fetch(cache)
    assert second.returncode == 0, second.stderr
    assert pathlib.Path(second.stdout.strip()) == path
    assert not list(cache.glob("*.part"))


@slow
def test_a_tampered_cache_file_is_removed_and_refused(tmp_path):
    cache = tmp_path / "cache dir"
    assert _fetch(cache).returncode == 0
    cached = cache / PINS[NAME]["file"]
    cached.write_bytes(b"x")
    refused = _fetch(cache)
    assert refused.returncode == 1
    assert PINS[NAME]["file"] in refused.stderr
    assert "failed verification" in refused.stderr
    assert not cached.exists()
    again = _fetch(cache)
    assert again.returncode == 0, again.stderr
    assert _sha256(cached) == PINS[NAME]["sha256"]


@slow
def test_a_download_that_does_not_match_is_discarded(tmp_path):
    folder = tmp_path / "script copy"
    folder.mkdir()
    shutil.copy(SCRIPT, folder / SCRIPT.name)
    pins = json.loads(json.dumps(PINS))
    pins[NAME]["sha256"] = "0" * 64
    (folder / "pins.json").write_text(json.dumps(pins, indent=2) + "\n", encoding="utf-8")
    cache = tmp_path / "cache dir"
    r = _fetch(cache, script=folder / SCRIPT.name)
    assert r.returncode == 1
    assert f"download of {NAME} does not match pins.json (got {PINS[NAME]['sha256']})" in r.stderr
    assert not list(cache.iterdir())


def test_unknown_name_is_refused(tmp_path):
    r = _fetch(tmp_path / "cache", name="nope")
    assert r.returncode == 2
    assert "usage: fetch_pinned.ps1" in r.stderr
    assert not (tmp_path / "cache").exists()
