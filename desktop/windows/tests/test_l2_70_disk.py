"""Level 2: disk space (Review Focus 3). A fresh install onto a drive with
too little room is refused up front; an upgrade counts the space the copy
it replaces frees. Admin (diskpart)."""
import json

import pytest

import l2_support as l2

pytestmark = [pytest.mark.usefixtures("clean_machine"),
              pytest.mark.skipif(not l2.diskpart_available(), reason="diskpart is not available")]


def _disk_free_gb() -> int:
    return json.loads((l2.REPO / "desktop" / "windows" / "size_gate.json").read_text(encoding="utf-8"))["disk_free_gb"]


def test_too_little_free_space_is_refused(installer, l2_home):
    with l2.vhd(300) as drive:
        folder = drive / "FO"
        result = l2.install(installer, "user", dir=folder, env=l2_home.env)
        try:
            assert result.rc == l2.EXIT_CODES["refused"], result.log[-3000:]
            assert "refused: needs_space" in result.log
            assert not folder.exists() or not any(folder.iterdir())
        finally:
            if l2.uninstall_entry("user"):
                l2.uninstall(folder, "user", env=l2_home.env)


def test_an_upgrade_counts_the_space_it_frees(installer, l2_home, dist_record):
    with l2.vhd(_disk_free_gb() * 1024 + 256) as drive:
        folder = drive / "FO"
        try:
            first = l2.install(installer, "user", dir=folder, env=l2_home.env)
            assert first.rc == 0, first.log[-3000:]
            again = l2.install(installer, "user", dir=folder, env=l2_home.env)
            assert again.rc == 0, again.log[-3000:]
            assert "refused:" not in again.log
            import hashlib
            assert hashlib.sha256(l2.manifest(folder).encode("utf-8")).hexdigest() == dist_record["image_sha256"]
        finally:
            if l2.uninstall_entry("user"):
                l2.kill_under(folder)
                l2.uninstall(folder, "user", env=l2_home.env)
