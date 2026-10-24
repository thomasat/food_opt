"""desktop/shared/messages.json holds every string the desktop shells and
the launcher show. These tests tie it to the Mac sources in both directions:
each Mac text is the Swift or launcher.sh literal it names, and every Mac
screen literal is in messages.json or on a Mac-only list below."""
import re

import pytest

import wording
from tests.mac_sources import (
    is_machinery, launcher_sh_literals, normalize, strip_placeholders,
    swift_literals, swift_scope, swift_screen_literals)
from tests.shared_texts import REPO, entries, messages_doc, texts
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
        assert "mac" in e["text"], e
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


def test_get_help_body_is_the_support_text():
    body = texts("mac")["help.get_help.body"]
    assert body == ("Email sohum.patnaik@fsi.org. Describe the problem and what you expected. "
                    "Attaching the app's log file helps — Help › Show Log File finds it "
                    "for you.")
    assert wording.SUPPORT_EMAIL in body


def test_machinery_rules():
    """The literal filter keeps words and drops code."""
    assert is_machinery("pageZoom") and is_machinery("foodopt://retry")
    assert is_machinery("display:none") and is_machinery("q")
    assert not is_machinery("File") and not is_machinery("Try again")
    assert messages_doc()["_comment"]
