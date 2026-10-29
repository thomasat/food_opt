"""The Mac shell's own strings, read out of its sources: the Swift window
(desktop/FoodOptimizerApp.swift) and desktop/launcher.sh. The sync tests
compare them with desktop/shared/messages.json in both directions."""
import re

from tests.shared_texts import REPO, read_lf

SWIFT = REPO / "desktop" / "FoodOptimizerApp.swift"
LAUNCHER_SH = REPO / "desktop" / "launcher.sh"

_LITERAL = re.compile(r'"((?:[^"\\\n]|\\.)*)"')
_BLOCK = re.compile(r'"""(.*?)"""', re.S)
_STYLE = re.compile(r"<style>.*?</style>", re.S)
_TEXT_NODE = re.compile(r">([^<>]*)<")
_WORD = re.compile(r"[A-Za-z]{2,}")
_MACHINERY = re.compile(
    r"[<>{}=&]|\\\(|getElementById|\.style\b|textContent|[a-z-]+:\S")
_TOKEN = re.compile(r"[\w./:%-]+")


def _swift_text() -> str:
    return read_lf(SWIFT)


def swift_scope(anchor: str) -> str:
    """The declaration that holds the first line containing `anchor`: from
    that line (or the nearest `func`/`let`/`var` line above it) up to the
    next non-blank line indented no deeper than where it started."""
    lines = _swift_text().split("\n")
    hit = next((i for i, line in enumerate(lines) if anchor in line), None)
    assert hit is not None, f"no Swift line contains {anchor!r}"
    start = hit
    while start >= 0 and not any(k in lines[start] for k in ("func ", "let ", "var ")):
        start -= 1
    if start < 0:
        start = hit
    indent = len(lines[start]) - len(lines[start].lstrip())
    end = start + 1
    while end < len(lines):
        line = lines[end]
        if line.strip() and len(line) - len(line.lstrip()) <= indent:
            break
        end += 1
    return "\n".join(lines[start:end])


def swift_literals(text: str) -> list:
    """Every string a Swift text puts on screen, candidates included: its
    single-line literals (concatenations joined) and the text nodes of its
    HTML blocks."""
    text = "\n".join(line for line in text.split("\n")
                     if not line.lstrip().startswith("//"))
    nodes = []
    for block in _BLOCK.findall(text):
        block = _STYLE.sub("", block)
        for node in _TEXT_NODE.findall(block):
            # A node with an interpolation is kept for the forward check
            # (the "\(n) s" counter is only ever a text node); the screen
            # filter drops it as machinery.
            if _WORD.search(node):
                nodes.append(node)
    text = text.replace('"""', "\x00")
    text = re.sub(r'"\s*\+\s*"', "", text)
    return [s.strip() for s in _LITERAL.findall(text)] + [n.strip() for n in nodes]


def is_machinery(s: str) -> bool:
    """True for a literal that is code, not text a person reads."""
    if not _WORD.search(s):
        return True
    if _MACHINERY.search(s):
        return True
    return bool(_TOKEN.fullmatch(s)) and not s[:1].isupper()


def swift_screen_literals() -> set:
    return {s for s in swift_literals(_swift_text()) if not is_machinery(s)}


def launcher_sh_literals(anchor: str) -> list:
    """`status "…"` literals for the anchor "status", else every
    `NAME="…"` literal; each without its `|percent` tail."""
    text = read_lf(LAUNCHER_SH)
    if anchor == "status":
        pattern = r'(?<![\w-])status\s+"((?:[^"\\\n]|\\.)*)"'
    else:
        pattern = r'(?<![A-Za-z_0-9])' + re.escape(anchor) + r'="((?:[^"\\\n]|\\.)*)"'
    return [m.split("|")[0] for m in re.findall(pattern, text)]


def normalize(s: str) -> str:
    """Placeholders of every kind as `{}`: Swift's \\(…), ours {name}, and
    the shell's $NAME / ${NAME}."""
    s = re.sub(r"\\\([^()]*(?:\([^()]*\)[^()]*)*\)", "{}", s)
    s = re.sub(r"\{[a-z_]+\}", "{}", s)
    s = re.sub(r"\$\{[A-Za-z_][A-Za-z_0-9]*\}|\$[A-Za-z_][A-Za-z_0-9]*", "{}", s)
    return s


def strip_placeholders(s: str) -> str:
    return re.sub(r"\{[a-z_]+\}", "", s)


def swift_menu_items() -> list:
    """(title, keyEquivalent) for every NSMenuItem the Swift builds."""
    return re.findall(
        r'NSMenuItem\(\s*title:\s*"((?:[^"\\]|\\.)*)".*?keyEquivalent:\s*"((?:[^"\\]|\\.)*)"',
        _swift_text(), re.S)


def swift_array(name: str) -> list:
    """The string items of `let <name> = [ … ]`."""
    m = re.search(r"let " + re.escape(name) + r"\s*=\s*\[(.*?)\]", _swift_text(), re.S)
    assert m, f"no Swift array {name!r}"
    return re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))
