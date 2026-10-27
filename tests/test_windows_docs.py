"""The Windows docs (phase 5): desktop/windows/README.md's release checklist
covers the spec's Level 3 list, both checklists run the one shared demo
walkthrough (desktop/shared/manual_walkthrough.md), which was moved out of
desktop/README.md rather than rewritten, and the requirements block names
the measured disk figure."""
import json
import pathlib
import re

from tests.test_release_guards import release_assets

REPO = pathlib.Path(__file__).resolve().parent.parent
WALKTHROUGH = REPO / "desktop" / "shared" / "manual_walkthrough.md"
MAC_README = REPO / "desktop" / "README.md"
WIN_README = REPO / "desktop" / "windows" / "README.md"
SIZE_GATE = REPO / "desktop" / "windows" / "size_gate.json"


def _flat(path) -> str:
    text = path if isinstance(path, str) else path.read_text(encoding="utf-8")
    return " ".join(text.split())


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
    assert f"Needs 64-bit Windows 10 (22H2) or Windows 11 and {n} GB free. Works offline." in _flat(WIN_README)


# ---------- the combined release ritual (phase 6, Task 7) ----------

def _section(text, heading):
    start = text.index(heading)
    end = text.find("\n## ", start + 1)
    return text[start:] if end < 0 else text[start:end]


def _ritual():
    return _section(MAC_README.read_text(encoding="utf-8"), "## Publishing a release (Mac and Windows together)")


def _windows_half():
    return _section(WIN_README.read_text(encoding="utf-8"), "## Release (Windows half)")


def test_release_ritual_follows_the_spec():
    ritual = _ritual()
    at = 0
    for needle in ("update_lock.sh", "desktop/VERSION", "git tag -a v", "build_dmg.sh", "windows/README.md",
                   "scp", "shasum -a 256", "gh release create v",
                   "--draft --verify-tag", "--notes-file", "gh release view", "--draft=false --latest",
                   "curl -I"):
        found = ritual.find(needle, at)
        assert found >= 0, f"{needle!r} missing, or out of order"
        at = found + len(needle)
    assert ritual.count("⚠") >= 5
    for name in release_assets("X.Y.Z").values():
        assert name in ritual, name
    assert "--notes-file desktop/release_notes/X.Y.Z.md" in ritual
    assert "gh workflow enable tests -R thomasat/food_opt" in ritual
    assert "release-080" in ritual
    assert "windows/download_email.md" in ritual


def test_windows_release_half():
    half = _windows_half()
    for needle in ("git clone --branch v", "build_installer.ps1", "run_level2.ps1", "-Cleanup",
                   "run_ui_tests.ps1", "W1", "L1", "test_our_processes_connect_only_to_loopback"):
        assert needle in half, needle
    hand_back = half[half.index("Hand back to the Mac"):]
    for name in (".build.txt", ".files.txt"):
        assert name in hand_back, name
    assert "report.json" not in hand_back


def test_ritual_names_no_real_version():
    for text in (_ritual(), _windows_half()):
        assert not re.search(r"\b\d+\.\d+\.\d+\b", text), re.search(r"\b\d+\.\d+\.\d+\b", text)


def test_windows_readme_pending_markers():
    found = set(re.findall(r"PENDING\(((?:[^()]|\([^()]*\))*)\)", WIN_README.read_text(encoding="utf-8")))
    assert found == {"signing step"}


def test_windows_readme_names_the_release_assets():
    out = WIN_README.read_text(encoding="utf-8")
    for needle in (".files.txt", "files_list_sha256"):
        assert needle in out, needle


def _mac_section(heading):
    return _section(MAC_README.read_text(encoding="utf-8"), heading)


def test_desktop_readme_says_how_to_run_the_tests():
    section = _flat(_mac_section("## Run the tests"))
    for needle in ("uv venv", "desktop/requirements.lock.txt", "pytest", "matplotlib", "0 failed", "12", "golden",
                   "--basetemp", "uv venv --python 3.13", "PYTHONDONTWRITEBYTECODE=1", "tests tests\\desktop_launcher tests\\desktop_page"):
        assert needle in section, needle


def test_windows_readme_lists_the_vm_quirks():
    section = _flat(_section(WIN_README.read_text(encoding="utf-8"), "## Test"))
    for needle in ("core.autocrlf false", "/SD 01/01/2020", "signed in", "sleep"):
        assert needle in section, needle


def test_level1_paths_are_generic():
    assert "p5-" not in WIN_README.read_text(encoding="utf-8")


def test_release_ritual_has_the_ready_snapshot_todo():
    ritual = _ritual()
    assert "tests/test_windows_ready.py" in ritual
    assert "wording_snapshot_080.json" in ritual
    assert not re.search(r"\b\d+\.\d+\.\d+\b", ritual)


def test_design_note_exists():
    path = REPO / "desktop" / "windows" / "DESIGN.md"
    text = path.read_text(encoding="utf-8")
    assert len(text.splitlines()) <= 120
    flat = " ".join(text.split())
    for needle in ("shared launcher", "Job Object", "per-user", "all-users", "No SBOM"):
        assert needle in flat, needle
