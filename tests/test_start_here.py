"""One Start Here source (desktop/shared/start_here.src.md) renders the Mac
file byte for byte as it is today and the Windows file beside it."""
import hashlib
import importlib.util
import re

import pytest

import wording
from tests.shared_texts import REPO, flowed, read_lf

SRC = REPO / "desktop" / "shared" / "start_here.src.md"
MAC = REPO / "desktop" / "start_here.txt"
WIN = REPO / "desktop" / "windows" / "start_here.txt"
# 2026-11-24: plain-language pass (M1 chip, home folder, Go to Folder,
# named options, property limits, percentages, calculations).
# 2026-11-26: "Start Round 7" then Generate; Show Log File shows launcher.log.
MAC_SHA256 = "4cc66ee9e0e1c4aab4e4bdccd3a4828240c67a99b24d0637efa40a7d8440f2e7"


def _module():
    spec = importlib.util.spec_from_file_location(
        "start_here", REPO / "desktop" / "shared" / "start_here.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_mac_render_is_todays_start_here_byte_for_byte():
    out = _module().render(read_lf(SRC), "mac").encode("utf-8")
    assert out == read_lf(MAC).encode("utf-8")
    assert hashlib.sha256(out).hexdigest() == MAC_SHA256


def test_windows_render_is_committed():
    assert _module().render(read_lf(SRC), "windows") == read_lf(WIN)


def test_directives():
    render = _module().render
    src = ("<!-- a comment -->\nboth\n<!-- mac -->\nmac only\n<!-- end -->\n"
           "<!-- windows -->\nwin only\n<!-- end -->\nboth again\n")
    assert render(src, "mac") == "both\nmac only\nboth again\n"
    assert render(src, "windows") == "both\nwin only\nboth again\n"
    for bad in ("<!-- linux -->\nx\n<!-- end -->\n",
                "<!-- mac -->\n<!-- windows -->\nx\n<!-- end -->\n<!-- end -->\n",
                "<!-- mac -->\nx\n",
                "x\n<!-- end -->\n"):
        with pytest.raises(ValueError):
            render(bad, "mac")


def test_check_mode_reports_a_stale_render(tmp_path, monkeypatch):
    mod = _module()
    assert mod.main(["--check"]) == 0
    mac = tmp_path / "mac.txt"
    win = tmp_path / "win.txt"
    mac.write_text(read_lf(MAC), encoding="utf-8", newline="\n")
    win.write_text(read_lf(WIN).replace("INSTALL", "INSTAL", 1), encoding="utf-8", newline="\n")
    monkeypatch.setattr(mod, "OUTPUTS", {"mac": str(mac), "windows": str(win)})
    assert mod.main(["--check"]) == 1


def test_windows_start_here_says_what_the_spec_requires():
    text = flowed(read_lf(WIN))
    for needed in ("Projects are on this PC, not in OneDrive.",
                   f"Use “{wording.SAVE_A_COPY}” to back up.",
                   "Type Food Optimizer in Start.",
                   "Close the window to quit.",
                   "Run the new installer. Your projects are kept.",
                   "First opening takes longer. Keep the app's window open.",
                   "Quick access",
                   "Enable Editing",
                   "Download the Round workbook (Excel).",
                   # The three download warnings (tests/test_release_texts.py),
                   # lettered under INSTALL step 1.
                   "1. Open the Food Optimizer installer you downloaded. "
                   "Windows warns about new apps it doesn't know yet. This is expected. "
                   "To continue: "
                   "a. In Edge's download list: point at the file, click \u2026, then Keep. "
                   "b. In the box that opens: click the arrow beside Delete, then Keep anyway. "
                   "c. On the blue \"Windows protected your PC\" screen: click More info, "
                   "then Run anyway. "
                   "2. Click Install.",
                   wording.SUPPORT_EMAIL,
                   "paste %LOCALAPPDATA%\\FoodOptimizer into File Explorer's address bar "
                   "and attach launcher.log."):
        assert needed in text, needed
    assert "Setup time depends on your connection and computer." not in text
    for banned in ("Finder", "Applications", "⌘", "~/", r"\bMac\b", r"\bNumbers\b", "Apple"):
        assert not re.search(banned, text), banned


def test_no_render_ships_an_html_comment():
    """PENDING markers are comments in the source; a marker on a text line
    would leak into the render users read."""
    assert "<!--" not in read_lf(WIN)
    assert "<!--" not in read_lf(MAC)


def test_no_line_runs_past_80_characters():
    """Start Here opens in TextEdit and Notepad, which wrap a long line
    wherever the window ends; two lines ran to 102 and 121 characters."""
    for path in (MAC, WIN):
        long = [(n, len(line)) for n, line in enumerate(read_lf(path).splitlines(), 1)
                if len(line) > 80]
        assert long == [], (path.name, long)
