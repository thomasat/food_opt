"""The Windows docs (phase 5): desktop/windows/README.md's release checklist
covers the spec's Level 3 list, both checklists run the one shared demo
walkthrough (desktop/shared/manual_walkthrough.md), which was moved out of
desktop/README.md rather than rewritten, and the requirements block names
the measured disk figure."""
import json
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parent.parent
WALKTHROUGH = REPO / "desktop" / "shared" / "manual_walkthrough.md"
MAC_README = REPO / "desktop" / "README.md"
WIN_README = REPO / "desktop" / "windows" / "README.md"
SIZE_GATE = REPO / "desktop" / "windows" / "size_gate.json"


def _flat(path) -> str:
    return " ".join(path.read_text(encoding="utf-8").split())


def _checklist(text) -> str:
    start = text.index("## Manual checklist")
    end = text.find("\n## ", start + 1)
    return text[start:] if end < 0 else text[start:end]


def test_both_checklists_run_the_shared_walkthrough():
    for readme in (MAC_README, WIN_README):
        assert "shared/manual_walkthrough.md" in readme.read_text(encoding="utf-8"), readme


def test_walkthrough_was_moved_not_rewritten():
    walkthrough, mac = _flat(WALKTHROUGH), _flat(MAC_README)
    for needle in ("Try the yogurt pilot", "Yogurt pilot - Round 6 to score", "Round 6 results (pilot).xlsx",
                   "Save uploaded results", "Start Round 7", "Start this example over",
                   "Open the burger example", "Download all formulations (Excel)"):
        assert needle in walkthrough, needle
        assert needle not in mac, needle
    for heading in ("## The yogurt pilot", "## A project of your own", "## The all-formulations workbook"):
        assert heading in WALKTHROUGH.read_text(encoding="utf-8"), heading


def test_walkthrough_has_no_os_only_words():
    text = re.sub(r"\(Mac:[^)]*\)", "", WALKTHROUGH.read_text(encoding="utf-8"))
    for word in ("Cmd", "⌘", "Finder", "~/", "Activity Monitor", "Ctrl", "File Explorer"):
        assert word not in text, word


def test_windows_checklist_covers_level_3():
    checklist = _checklist(WIN_README.read_text(encoding="utf-8"))
    for needle in ("Mark of the Web", "SmartScreen", "Keep", "taskbar", "150%", "Quick access",
                   "Zone.Identifier", "Task Manager", "Sleep", "offline", "José Müller", "Upgrade",
                   "Uninstall", "WFP"):
        assert needle in checklist, needle
    items = re.findall(r"^- \*\*W(\d+)\*\*", checklist, re.M)
    assert items == [str(i) for i in range(1, 18)], items
    assert len(re.findall(r"^- \*\*W\d+\*\* \[S\]", checklist, re.M)) >= 8


def test_requirements_block_names_the_disk_figure():
    n = json.loads(SIZE_GATE.read_text(encoding="utf-8"))["disk_free_gb"]
    assert f"Needs Windows 10 or 11 (64-bit) and {n} GB free. Works offline." in _flat(WIN_README)
