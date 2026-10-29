"""desktop/shared/messages.json holds every string the desktop shells and
the launcher show. These tests tie it to the Mac sources in both directions:
each Mac text is the Swift or launcher.sh literal it names, and every Mac
screen literal is in messages.json or on a Mac-only list below."""
import re

import pytest

from tests.mac_sources import (
    is_machinery, launcher_sh_literals, normalize, strip_placeholders,
    swift_literals, swift_scope, swift_screen_literals)
from tests.shared_texts import REPO, entries, messages_doc, text_for, texts
from tests.test_food_bo import _launcher_lines

KINDS = {"title", "body", "line", "label", "button", "menu"}

# Swift screen literals with no counterpart in messages.json.
MAC_ONLY_SWIFT = {
    # The copy-to-Applications flow: Windows installs, it has no disk image.
    "Copy Food Optimizer to your Applications folder?",
    "The app is running from the downloaded disk image. Copying it to "
    "Applications keeps it on your Mac so you can open it anytime.",
    "Copy to Applications",
    "Not Now",
    "The copy did not work",
    "Please drag Food Optimizer onto the Applications folder in the installer "
    "window, then open it from Applications.",
    # Native dialog buttons, labelled by the OS on Windows.
    "OK",
    "Cancel",
    # Machinery that reads like words: paths, keys and names.
    "/Applications/Food Optimizer.app",
    "CFBundleShortVersionString",
    "FoodOptimizer",
    "FoodOptimizerMain",
    "Glass",
    "Library/Application Support/FoodOptimizer",
    # The launcher-line prefixes the window recognises setup by; the shared
    # page mirrors them in status.js SETUP_STEP_PREFIXES.
    "Downloading Python",
    "Creating environment",
    "Updating components",
    "Installing components",
    "Finishing setup",
}

# launcher.sh lines that are logged, never shown.
MAC_ONLY_LAUNCHER = {"setup failed - most likely no internet connection"}

_ID = re.compile(r"^[a-z0-9_]+(\.[a-z0-9_]+)+$")
_SHARED = re.compile(r"^(pages|shell|mac-only|launcher:[A-Z_]+)$")


def test_every_entry_is_well_formed():
    ids = [e["id"] for e in entries()]
    assert len(ids) == len(set(ids)), [i for i in ids if ids.count(i) > 1]
    for e in entries():
        assert _ID.match(e["id"]), e["id"]
        assert e["kind"] in KINDS, e
        assert "mac" in e["text"] and set(e["text"]) <= {"mac", "windows"}, e
        assert (e["mac"] is None) == (e["text"]["mac"] is None), e
        assert _SHARED.match(e["shared"]), e


def _located():
    return [e for e in entries() if e["mac"]]


@pytest.mark.parametrize("entry", _located(), ids=lambda e: e["id"])
def test_mac_text_is_the_literal_it_names(entry):
    source, anchor = entry["mac"].split(":", 1)
    want = normalize(entry["text"]["mac"])
    if source == "FoodOptimizerApp.swift":
        found = {normalize(s) for s in swift_literals(swift_scope(anchor))}
    else:
        assert source == "launcher.sh", entry["mac"]
        found = {normalize(s) for s in launcher_sh_literals(anchor)}
    assert want in found, (entry["id"], sorted(found))


def _stripped_mac_texts():
    return {strip_placeholders(e["text"]["mac"]) for e in entries() if e["text"]["mac"]}


def test_every_swift_screen_literal_is_in_messages():
    assert swift_screen_literals() - MAC_ONLY_SWIFT - _stripped_mac_texts() == set()


def test_every_launcher_screen_line_is_in_messages():
    missing = set(_launcher_lines(REPO)) - MAC_ONLY_LAUNCHER - _stripped_mac_texts()
    assert missing == set()


def test_mac_only_allowlists_hold_only_real_literals():
    assert MAC_ONLY_SWIFT <= swift_screen_literals(), MAC_ONLY_SWIFT - swift_screen_literals()
    assert MAC_ONLY_LAUNCHER <= set(_launcher_lines(REPO))


def test_machinery_rules():
    """The literal filter keeps words and drops code."""
    assert is_machinery("pageZoom") and is_machinery("foodopt://retry")
    assert is_machinery("display:none") and is_machinery("q")
    assert not is_machinery("File") and not is_machinery("Try again")
    assert messages_doc()["_comment"]


# ---------- the Windows side ----------

def test_every_entry_names_its_windows_text():
    for e in entries():
        assert text_for(e, "windows") is None or isinstance(text_for(e, "windows"), str), e["id"]


def test_windows_text_is_written_only_when_it_differs():
    for e in entries():
        if "windows" in e["text"]:
            assert e["text"]["windows"] != e["text"]["mac"], e["id"]


def test_an_absent_windows_text_is_the_mac_text():
    assert text_for({"text": {"mac": "A"}}, "windows") == "A"
    assert text_for({"text": {"mac": "A", "windows": None}}, "windows") is None
    assert text_for({"text": {"mac": "A", "windows": "B"}}, "windows") == "B"
    assert text_for({"text": {"mac": "A", "windows": "B"}}, "mac") == "A"


def test_mac_only_entries_have_no_windows_text():
    by_id = {e["id"]: e for e in entries()}
    mac_only = ["steps.title.updating", "steps.download", "steps.note.first_run",
                "steps.note.upgrade", "menu.file.close_window"]
    mac_only += [i for i in by_id if i.startswith(("error.needs_internet.", "error.no_space.",
                                                   "menu.edit", "menu.window"))]
    mac_only += [i for i in by_id if i.startswith("launcher.status.")
                 and i not in ("launcher.status.load", "launcher.status.starting")]
    assert len(mac_only) > 20
    for i in mac_only:
        assert text_for(by_id[i], "windows") is None, i


_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def test_new_windows_strings_follow_the_copy_rules():
    for e in entries():
        win, mac = text_for(e, "windows"), e["text"]["mac"]
        if not isinstance(win, str) or win == mac:
            continue
        words = len(win.split())
        if e["kind"] in ("label", "button", "menu"):
            assert words <= 5, e["id"]
        elif e["kind"] == "title":
            assert words <= 8, e["id"]
        else:
            for sentence in _SENTENCE.split(win):
                assert len(sentence.split()) <= 10, (e["id"], sentence)


_MAC_TERMS = re.compile(r"Finder|this Mac|⌘|Cmd|Applications|~/|\bMac\b|\bdmg\b|\bNumbers\b")
_WINDOWS_TERMS = ("this PC", "File Explorer", "Ctrl", "Windows", "WebView2", "Reinstall")


def test_no_mac_terms_in_windows_texts():
    for i, t in texts("windows").items():
        assert not _MAC_TERMS.search(t), (i, t)


def test_no_windows_terms_in_mac_texts():
    for i, t in texts("mac").items():
        assert not any(term in t for term in _WINDOWS_TERMS), (i, t)


def test_mac_menu_keys_match_swift():
    from tests.mac_sources import swift_menu_items
    pairs = set(swift_menu_items())
    for e in messages_doc()["menus"]:
        for key in e["keys"]["mac"]:
            assert (e["text"]["mac"], key) in pairs, (e["id"], key)


def _windows_menu_ids():
    layout = messages_doc()["menu_layout"]["windows"]
    return [i for b in layout for i in [b["menu"], *b["items"]] if i != "-"]


def test_every_menu_id_is_used_by_the_windows_shell():
    cs = sorted((REPO / "desktop" / "windows" / "FoodOptimizer").glob("**/*.cs"))
    if not cs:
        pytest.skip("phase 4 adds the C# shell")
    from tests.shared_texts import read_lf
    source = "\n".join(read_lf(p) for p in cs)
    for i in _windows_menu_ids():
        assert i in source, i
