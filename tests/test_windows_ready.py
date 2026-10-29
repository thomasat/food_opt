"""The shared app, ready for Windows (Windows port phase 1).

The 0.8.0 wording snapshot is the Mac-safety guard: every module-level
string in wording.py stays exactly as 0.8.0 shipped it unless its name is in
ALLOWED_CHANGES, so a Windows fix cannot quietly change what a Mac user reads.
"""
import json
import pathlib

import wording

SNAPSHOT = pathlib.Path(__file__).parent / "fixtures" / "wording_snapshot_080.json"

# Names whose 0.8.0 value is meant to change (or that are new). Each entry
# says which fix changed it.
ALLOWED_CHANGES = {
    "SUPPORT_EMAIL",  # new: the support address (Windows port phase 1)
}


def _all_text(value):
    if isinstance(value, str):
        return True
    if isinstance(value, (list, tuple)):
        return all(_all_text(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _all_text(v) for k, v in value.items())
    return False


def _plain(value):
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


def collect_wording_constants():
    """Every module-level name in wording whose value is text, or a list,
    tuple or dict made only of text — as JSON-ready values."""
    return {name: _plain(value) for name, value in vars(wording).items()
            if not name.startswith("__") and _all_text(value)}


def test_wording_constants_match_080_snapshot():
    frozen = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    current = collect_wording_constants()
    changed = {k for k in frozen
               if k not in ALLOWED_CHANGES and current.get(k) != frozen[k]}
    assert changed == set()
    missing = {k for k in frozen if k not in current and k not in ALLOWED_CHANGES}
    assert missing == set()


# ------------------------------------------------------------------ #
# OS terms: the Mac keeps its words; Windows gets the PC's.
# ------------------------------------------------------------------ #
import re

import pytest

MAC_TERMS = ("Finder", "this Mac", "⌘", "Cmd", "Applications", ".dmg", "~/")
# "Numbers" the Mac app, not the word: these two use it as a table row
# naming numeric literals (`2`, `0.5`) in the calculation guide.
NUMBERS_NOT_THE_APP = {"CALCULATION_TERMS_TABLE", "RULE_GUIDE"}
# Constants that hold the Mac wording and reach a screen only through a
# platform helper (wording._disk_trouble), which the function checks below
# cover on Windows.
READ_THROUGH_A_HELPER = {"_DISK_TROUBLE"}


@pytest.fixture
def on_mac(monkeypatch):
    monkeypatch.setattr(wording.sys, "platform", "darwin")


@pytest.fixture
def on_windows(monkeypatch):
    monkeypatch.setattr(wording.sys, "platform", "win32")


def test_saved_line_names_this_mac(on_mac):
    assert wording.saved_line("14:32") == "Saved automatically at 14:32, on this Mac"


def test_saved_line_names_this_pc(on_windows):
    assert wording.saved_line("14:32") == "Saved automatically at 14:32, on this PC"


def test_read_only_fix_names_finder_on_a_mac(on_mac):
    assert wording.not_saved("Oat cookie", "read_only") == (
        "Oat cookie was not saved: the FoodOptimizer folder is read-only. "
        "Allow changes to it in Finder, then try again.")


def test_read_only_fix_names_file_explorer_on_windows(on_windows):
    assert wording.not_saved("Oat cookie", "read_only") == (
        "Oat cookie was not saved: the FoodOptimizer folder is read-only. "
        "Allow changes to it in File Explorer, then try again.")


def test_sheet_navigation_names_numbers_on_a_mac(on_mac):
    assert wording.sheet_navigation(premixes=False).endswith(
        "Tabs are at the bottom in Excel and at the top in Numbers. ")


def test_sheet_navigation_names_only_excel_on_windows(on_windows):
    assert wording.sheet_navigation(premixes=False).endswith(
        "Tabs are at the bottom in Excel. ")


def _windows_texts():
    texts = {name: str(value) for name, value in collect_wording_constants().items()
             if name not in READ_THROUGH_A_HELPER}
    for cause in ("full", "read_only", "in_use", None):
        texts[f"not_saved/{cause}"] = wording.not_saved("X", cause)
        texts[f"nothing_changed/{cause}"] = wording.nothing_changed("X", cause)
        texts[f"copy_not_deleted/{cause}"] = wording.copy_not_deleted(cause)
    texts["saved_line"] = wording.saved_line("14:32")
    texts["sheet_navigation"] = wording.sheet_navigation()
    texts["sheet_navigation/no premixes"] = wording.sheet_navigation(premixes=False)
    texts["copy_downloaded"] = wording.copy_downloaded("a.json")
    return texts


def test_no_mac_terms_reach_windows(on_windows):
    found = {}
    for name, text in _windows_texts().items():
        hits = [t for t in MAC_TERMS if t in text]
        if name not in NUMBERS_NOT_THE_APP and re.search(r"\bNumbers\b", text):
            hits.append("Numbers")
        if hits:
            found[name] = hits
    assert found == {}


def test_copy_downloaded_names_only_the_file(on_mac):
    """The window's save dialog lets the reader pick the folder, so the line
    names the file and leaves the folder to the reveal that follows."""
    assert (wording.copy_downloaded("burger copy 2026-09-15.json")
            == "Saved as 'burger copy 2026-09-15.json'.")


def test_copy_downloaded_is_the_same_on_windows(on_windows):
    assert wording.copy_downloaded("a.json") == "Saved as 'a.json'."


def test_support_email_is_the_one_address_everywhere():
    root = pathlib.Path(__file__).resolve().parent.parent
    for path in ("desktop/FoodOptimizerApp.swift", "desktop/start_here.txt", "README.md"):
        text = (root / path).read_text(encoding="utf-8")
        assert wording.SUPPORT_EMAIL in text, path
        assert "github.com/thomasat/food_opt/issues" not in text, path


# ------------------------------------------------------------------ #
# Text files name their encoding: Windows' default is cp1252, not UTF-8,
# and the app's files and tests hold ›, — and ✓.
# ------------------------------------------------------------------ #
import ast


def _python_files_to_lint():
    from tests.test_packaging_modules import REPO, _app_modules
    files = [REPO / f"{m}.py" for m in sorted(_app_modules())]
    files += sorted((REPO / "tests").rglob("*.py"))
    files += sorted((REPO / "desktop" / "shared").rglob("*.py"))
    return REPO, files


def _mode_is_binary(call):
    mode = call.args[1] if len(call.args) > 1 else next(
        (k.value for k in call.keywords if k.arg == "mode"), None)
    return isinstance(mode, ast.Constant) and isinstance(mode.value, str) and "b" in mode.value


def _unnamed_encodings(path):
    found = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call):
            continue
        named = any(k.arg == "encoding" for k in node.keywords)
        func = node.func
        if isinstance(func, ast.Name) and func.id == "open":
            if not named and not _mode_is_binary(node):
                found.append(node.lineno)
        elif isinstance(func, ast.Attribute) and func.attr in ("read_text", "write_text"):
            if not named:
                found.append(node.lineno)
    return found


def test_text_io_names_an_encoding():
    root, files = _python_files_to_lint()
    found = [f"{p.relative_to(root)}:{line}" for p in files for line in _unnamed_encodings(p)]
    assert found == []


def test_ui_fetches_nothing_from_the_internet():
    """st.map and pydeck draw from Mapbox and Carto, and a Material page icon
    is drawn from fonts.gstatic.com. Button icons (icon=":material/…") use
    the font Streamlit bundles and are fine."""
    root, files = _python_files_to_lint()
    app_files = [p for p in files if p.parent == root]
    found = []
    for path in app_files:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] + [getattr(node, "module", None) or ""]
                if any(n.split(".")[0] == "pydeck" for n in names):
                    found.append(f"{path.name}:{node.lineno} pydeck")
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if (node.func.attr == "map" and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "st"):
                found.append(f"{path.name}:{node.lineno} st.map")
            if node.func.attr == "set_page_config":
                for k in node.keywords:
                    if (k.arg == "page_icon" and isinstance(k.value, ast.Constant)
                            and str(k.value.value).startswith(":material/")):
                        found.append(f"{path.name}:{node.lineno} page_icon")
    assert found == []
