"""The shared setup/error page, read as files: what it holds, what it
loads, and that every text it shows comes from messages.json. These always
run; test_page_browser.py drives the page in a browser."""
import re

from tests.mac_sources import swift_array
from tests.shared_texts import REPO, entries, read_lf

PAGES = REPO / "desktop" / "shared" / "pages"
STEPS_IDS = ["steps-view", "title", "steps", "barwrap", "bar", "detail", "note",
             "elapsed", "nextup", "footer"]
MESSAGE_IDS = ["message-view", "msg-title", "msg-spinner", "msg-step", "msg-barwrap",
               "msg-bar", "msg-slide", "msg-body", "retry", "msg-footer"]


def _html():
    return read_lf(PAGES / "status.html")


def _js():
    return read_lf(PAGES / "status.js")


def _js_code():
    """status.js without its comments."""
    code = re.sub(r"/\*.*?\*/", "", _js(), flags=re.S)
    return re.sub(r"(?m)^\s*//.*$|\s//[^\"'\n]*$", "", code)


def _js_literals():
    """status.js string literals, except SETUP_STEP_PREFIXES: those are the
    launcher's own line prefixes, matched, never shown (and pinned to the
    Swift list below)."""
    code = re.sub(r"var SETUP_STEP_PREFIXES = \[.*?\];", "", _js_code(), flags=re.S)
    return [a or b for a, b in re.findall(r'"((?:[^"\\\n]|\\.)*)"|\'((?:[^\'\\\n]|\\.)*)\'',
                                          code)]


def test_page_has_every_element_the_script_drives():
    html = _html()
    for i in STEPS_IDS + MESSAGE_IDS:
        assert f'id="{i}"' in html, i
    for prefix in ("row", "dot", "spin", "tick", "label"):
        assert f'"{prefix}"' in _js(), prefix


def test_page_carries_no_copy_of_its_own():
    body = re.sub(r"<script.*?</script>|<style.*?</style>", "", _html(), flags=re.S)
    for node in re.findall(r">([^<>]*)<", body):
        assert set(node) <= set(" \t\n•✓"), node
    for literal in _js_literals():
        assert not re.search(r"[A-Za-z]{2,} [A-Za-z]{2,}", literal), literal


_PAGE_ID = re.compile(r'"((?:steps|error|stalled|button|footer)\.[a-z0-9_.]+)"')


def test_page_message_ids_exist():
    known = {e["id"] for e in entries()}
    used = set(_PAGE_ID.findall(_js()))
    assert used and used <= known, used - known


def test_page_setup_prefixes_match_swift():
    m = re.search(r"var SETUP_STEP_PREFIXES = \[(.*?)\];", _js(), re.S)
    assert m
    assert re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1)) == swift_array("setupStepPrefixes")


def test_page_loads_nothing_from_outside():
    for path in sorted(PAGES.iterdir()):
        text = read_lf(path)
        for banned in ("http:", "https:", "//cdn", "@import", "foodopt://"):
            assert banned not in text, (path.name, banned)
    refs = re.findall(r'(?:href|src)="([^"]*)"', _html())
    assert sorted(refs) == ["status.css", "status.js"]


def test_font_list_names_segoe_ui():
    css = read_lf(PAGES / "status.css")
    body = re.search(r"(?m)^body\s*\{(.*?)\}", css, re.S)
    assert body and 'font-family: -apple-system, "Segoe UI", sans-serif' in body.group(1)


def test_every_page_message_is_used():
    used = set(_PAGE_ID.findall(_js()))
    for e in entries():
        if e["shared"] == "pages":
            assert e["id"] in used, e["id"]
