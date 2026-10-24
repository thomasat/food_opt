"""Repo-side guards for the C# Windows shell (desktop/windows/FoodOptimizer):
every screen string comes from desktop/shared/messages.json, every message
id it names exists with Windows text, and the page bridge calls only the
shared page's API. These run on any OS; the C# itself is tested by
FoodOptimizer.Tests and FoodOptimizer.UiTests."""
import re

from tests.shared_texts import REPO, entries, read_lf, texts

SHELL = REPO / "desktop" / "windows" / "FoodOptimizer"
SHELL_CS = sorted(
    p for d in ("FoodOptimizer.Core", "FoodOptimizer.Shell")
    for p in (SHELL / d).glob("**/*.cs"))

# Literals that read like words but are not screen text.
ALLOWED_CS_LITERALS = {
    "Start Here.txt",  # a file name; phase 5 installs it
}

_VERBATIM = re.compile(r'@"(?:[^"]|"")*"')
_RAW = re.compile(r'"""[\s\S]*?"""')
_REGULAR = re.compile(r'\$?"(?:[^"\\\n]|\\.)*"')
_WORDS = re.compile(r"[A-Za-z]{2,}\s+[A-Za-z]{2,}")
_LINE_COMMENT = re.compile(r"//[^\n]*")


def _strip_comments(source: str) -> str:
    """The source without // comments. A // inside a string literal (a URL)
    is kept: literals are blanked out while the comments are found."""
    out, i = [], 0
    pattern = re.compile(r'@"(?:[^"]|"")*"|"""[\s\S]*?"""|\$?"(?:[^"\\\n]|\\.)*"|//[^\n]*')
    for m in pattern.finditer(source):
        out.append(source[i:m.start()])
        if not m.group().startswith("//"):
            out.append(m.group())
        i = m.end()
    out.append(source[i:])
    return "".join(out)


def _literals(source: str) -> list:
    """Every string literal in a C# source: verbatim, raw and regular, with
    the quotes and the @ / $ prefix removed."""
    source = _strip_comments(source)
    found = []
    for pattern in (_RAW, _VERBATIM):
        for m in pattern.finditer(source):
            found.append(m.group())
        source = pattern.sub(" ", source)
    found += [m.group() for m in _REGULAR.finditer(source)]
    out = []
    for lit in found:
        if lit.startswith('"""'):
            out.append(lit[3:-3])
        elif lit.startswith('@"'):
            out.append(lit[2:-1].replace('""', '"'))
        else:
            out.append(lit.lstrip("$")[1:-1])
    return out


def _shell_sources() -> dict:
    return {p: read_lf(p) for p in SHELL_CS}


def test_the_shell_has_sources():
    assert SHELL_CS, "no C# under desktop/windows/FoodOptimizer"


def test_shell_code_has_no_screen_text():
    every_text = set(texts("windows").values()) | set(texts("mac").values())
    for path, source in _shell_sources().items():
        name = path.relative_to(REPO).as_posix()
        for lit in _literals(source):
            if lit in ALLOWED_CS_LITERALS:
                continue
            assert not _WORDS.search(lit), (name, lit)
            assert lit not in every_text, (name, lit)
        assert "MessageBox.Show" not in _strip_comments(source), name


def test_the_literal_finder_sees_every_kind():
    sample = ('var a = "two words"; var b = @"say ""hi"" now"; '
              'var c = $"x {y} z"; // "in a comment"\n'
              'var d = """raw text here"""; var e = "http://a//b";')
    assert _literals(sample) == ['raw text here', 'say "hi" now', "two words",
                                 "x {y} z", "http://a//b"]


def test_shell_message_ids_have_windows_text():
    windows = texts("windows")
    ids = set()
    for source in _shell_sources().values():
        code = _strip_comments(source)
        ids |= set(re.findall(r'\b(?:Text|Format)\(\s*"([a-z0-9_.]+)"', code))
        ids |= set(re.findall(r'"(menu\.[a-z_.]+)"', code))
    missing = sorted(i for i in ids if i not in windows)
    assert not missing, missing


PAGE_API = {"init", "showSteps", "statusLine", "serverStarted", "tick",
            "showStalled", "showError", "post"}


def test_page_bridge_calls_only_the_page_api():
    bridge = read_lf(SHELL / "FoodOptimizer.Core" / "PageBridge.cs")
    status_js = read_lf(REPO / "desktop" / "shared" / "pages" / "status.js")
    called = set(re.findall(r"FoodOpt\.(\w+)\(", bridge))
    assert called, "PageBridge.cs calls nothing on the page"
    assert called <= PAGE_API, called - PAGE_API
    for name in called:
        assert re.search(r"\b%s:\s*%s\b" % (name, name), status_js), name


NEW_SHELL_MESSAGES = {
    "dialog.ok": ("button", "OK"),
    "dialog.cancel": ("button", "Cancel"),
    "error.shell_crashed.body": ("body", "Your projects are saved. Open Food Optimizer again."),
}


def test_new_shell_messages():
    by_id = {e["id"]: e for e in entries()}
    for i, (kind, text) in NEW_SHELL_MESSAGES.items():
        e = by_id[i]
        assert e["kind"] == kind, i
        assert e["text"] == {"mac": None, "windows": text}, i
        assert e["mac"] is None and e["shared"] == "shell", i


def test_powershell_scripts_are_ascii():
    scripts = sorted((REPO / "desktop" / "windows").glob("**/*.ps1"))
    assert scripts
    for p in scripts:
        p.read_bytes().decode("ascii")


WEBVIEW_SETTINGS = {
    "AreBrowserAcceleratorKeysEnabled": "false",
    "AreDevToolsEnabled": "false",
    "IsStatusBarEnabled": "false",
    "IsSwipeNavigationEnabled": "false",
    "IsGeneralAutofillEnabled": "false",
    "IsPasswordAutosaveEnabled": "false",
    "IsBuiltInErrorPageEnabled": "false",
    "IsReputationCheckingRequired": "false",
    "AreDefaultScriptDialogsEnabled": "false",
    "AreHostObjectsAllowed": "false",
    "IsZoomControlEnabled": "true",
    "AreDefaultContextMenusEnabled": "true",
    "IsWebMessageEnabled": "true",
}


def test_webview_settings_are_the_spec_list():
    host = _strip_comments(read_lf(SHELL / "FoodOptimizer.Shell" / "WebViewHost.cs"))
    for name, value in WEBVIEW_SETTINGS.items():
        assert re.search(r"\.%s\s*=\s*%s;" % (name, value), host), name
    env = read_lf(SHELL / "FoodOptimizer.Shell" / "WebEnvironment.cs")
    assert "IsCustomCrashReportingEnabled = true" in env
