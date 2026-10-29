"""The texts sent at a release (phase 6, revised 2026-10-25): the release
page (desktop/release_notes/<version>.md), the Windows download email
(desktop/windows/download_email.md) and the root README's Windows section.

Owner, 2026-10-25: the main thing sent is a download link. The page holds
the two download links, three "what is new" lines, one update line and
about five lines for IT teams; the email is three lines; the README's
Windows section is the link and one requirements line. Every claim left is
read here from the code or the test that backs it."""
import ast
import json
import pathlib
import re

from tests.shared_texts import flowed, read_lf
from tests.test_release_guards import PENDING, URL_BASE, release_assets

REPO = pathlib.Path(__file__).resolve().parent.parent
WINDOWS = REPO / "desktop" / "windows"
VERSION = (REPO / "desktop" / "VERSION").read_text(encoding="utf-8").strip()
NOTES = REPO / "desktop" / "release_notes" / f"{VERSION}.md"
EMAIL = WINDOWS / "download_email.md"
README = REPO / "README.md"
WIN_README = WINDOWS / "README.md"
CONTACT = "sohum.patnaik@fsi.org"
COMMENT = re.compile(r"<!--.*?-->", re.S)


def _visible(text):
    """What a reader sees: HTML comments (the PENDING markers) taken out."""
    return COMMENT.sub("", text)


def _lines(text):
    return [line.strip() for line in _visible(text).splitlines() if line.strip()]


def _sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", " ".join(text.split())) if s]


def _section(text, heading):
    start = text.index(heading)
    end = text.find("\n## ", start + 1)
    return text[start:] if end < 0 else text[start:end]


def _start_here_smartscreen():
    text = flowed(read_lf(WINDOWS / "start_here.txt"))
    return re.search(r"Windows may not recognize the app yet\..*?Run anyway\.", text).group(0)


SMARTSCREEN = ("Windows may not recognize the app yet. Click More info. "
               "Check the publisher is Food System Innovations, then click Run anyway.")
EDGE = "If Edge warns about the download, choose Keep."
REQUIREMENTS = "Needs 64-bit Windows 10 (22H2) or Windows 11 and 2 GB free."


def _requirements():
    """The Windows README's copy-paste requirements line (its disk figure
    is size_gate.json's: tests/test_windows_docs.py)."""
    text = WIN_README.read_text(encoding="utf-8")
    block = text[text.index("## System requirements (copy-paste"):]
    return next(line for line in block.splitlines()[1:] if line.strip())


def _l2_constant(name):
    tree = ast.parse((WINDOWS / "tests" / "l2_support.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"no literal {name} in l2_support.py")


def _defines(ref):
    path, _, name = ref.partition("::")
    tree = ast.parse((REPO / path).read_text(encoding="utf-8"))
    return any(isinstance(n, ast.FunctionDef) and n.name == name for n in ast.walk(tree))


def _url(role):
    return f"{URL_BASE}/v{VERSION}/{release_assets(VERSION)[role]}"


# ---------- the release page ----------

# Under the Mac link (owner, 2026-10-25): who can install it. The Mac
# launcher refuses anything else (arm64, macOS 13+; desktop/README.md's
# requirements block says the same).
MAC_NEEDS = "Needs a Mac with Apple silicon, macOS 13 or later."


def test_mac_requirement_line_is_what_the_mac_app_enforces():
    readme = flowed((REPO / "desktop" / "README.md").read_text(encoding="utf-8"))
    assert "Apple silicon" in readme and "macOS 13" in readme
    launcher = (REPO / "desktop" / "launcher.sh").read_text(encoding="utf-8")
    assert '"arm64"' in launcher and "-lt 13" in launcher

def test_start_here_smartscreen_is_the_owners_sentence():
    assert _start_here_smartscreen() == SMARTSCREEN
    assert EDGE in flowed(read_lf(WINDOWS / "start_here.txt"))


def test_notes_are_the_owners_five_parts():
    """Two download links, what is new, the update line, the IT lines:
    nothing else (no sizes, no steps, no checksums, no Assets paragraph)."""
    text = _visible(NOTES.read_text(encoding="utf-8"))
    assert re.findall(r"^## (.+)$", text, re.M) == [f"What is new in {VERSION}", "For IT teams"]
    head = text[:text.index("## What is new")]
    assert _lines(head) == [f"**[Download for Mac]({_url('dmg')})**", MAC_NEEDS,
                            f"**[Download for Windows]({_url('exe')})**", EDGE, SMARTSCREEN]
    for gone in ("{{", "MB", "Assets", "Checksums", "SHA-256", "Source code", "IT sheet", "SBOM"):
        assert gone not in text, gone


def test_notes_say_what_changed_in_three_lines():
    section = _section(_visible(NOTES.read_text(encoding="utf-8")), "## What is new in")
    bullets = [line[2:] for line in section.splitlines() if line.startswith("- ")]
    assert len(bullets) == 3, bullets
    for bullet in bullets:
        for sentence in _sentences(bullet):
            assert len(sentence.split()) <= 15, sentence
    assert any("0.8.0" in b and "differ" in b for b in bullets), bullets


def test_notes_have_one_update_line_for_both():
    section = _section(_visible(NOTES.read_text(encoding="utf-8")), "## What is new in")
    after = [line for line in section.splitlines() if line.strip() and not line.startswith(("- ", "## "))]
    assert after == ["**Updating?** Close Food Optimizer, then install the new version. Your projects are kept."]


def _it_lines():
    section = _section(NOTES.read_text(encoding="utf-8"), "## For IT teams")
    return [line[2:] for line in section.splitlines() if line.startswith("- ")]


def test_notes_have_six_it_lines():
    lines = _it_lines()
    assert len(lines) == 6, lines


def test_notes_bullet_three_names_csv_and_get_help():
    section = _section(_visible(NOTES.read_text(encoding="utf-8")), "## What is new in")
    bullets = [line[2:] for line in section.splitlines() if line.startswith("- ")]
    assert bullets[2] == ("CSV files from Excel in other languages upload correctly. "
                          "Get Help now gives an email address.")


def test_it_exit_line_names_the_documented_codes():
    line = _it_lines()[1]
    assert line == "Exit 1: the app is open; retry. Exit 9: security software blocked a file."
    codes = _l2_constant("EXIT_CODES")
    assert 1 == codes["app_running"] and 9 == codes["prewarm_failed"]
    readme = WIN_README.read_text(encoding="utf-8")
    assert re.search(r"^\| `1` \|", readme, re.M) and re.search(r"^\| `9` \|", readme, re.M)


def test_it_install_line_is_the_documented_all_users_command():
    """The command is the Windows README's all-users one (whose switches are
    the Level 2 harness's SILENT plus /ALLUSERS), for this version's file."""
    line = _it_lines()[0]
    readme = WIN_README.read_text(encoding="utf-8")
    documented = re.search(r'^("FoodOptimizer-<ver>-Windows\.exe" [^\n]*/ALLUSERS)', readme, re.M).group(1)
    command = documented.replace("<ver>", VERSION)
    assert line.startswith("Silent install for all users (as SYSTEM or admin): ")
    assert line.endswith(f"`{command}`"), line
    assert command.endswith(" ".join(_l2_constant("SILENT")) + " /ALLUSERS")


def test_it_publisher_line_is_setups_and_pending_signing():
    """Food System Innovations is setup.iss's AppPublisher; until the signed
    build shows it, a PENDING marker holds the release tag
    (tests/test_release_guards.py::test_no_pending_marker_at_a_release_tag)."""
    line = _it_lines()[2]
    publisher = re.search(r"^AppPublisher=(.*)$", (WINDOWS / "setup.iss").read_text(encoding="ascii"), re.M).group(1)
    assert _visible(line).strip() == f"Publisher: {publisher}."
    assert {m.group(1) for m in PENDING.finditer(line)} == {"publisher shown on the signed build"}


def test_it_sends_nothing_line_points_to_the_zero_traffic_test():
    """"Sends nothing" is Level 2's zero-traffic test: Food Optimizer's
    processes connect only to loopback during a full session; the release
    ritual runs Level 2 (tests/test_windows_docs.py::test_windows_release_half)."""
    assert _it_lines()[3] == ("Runs only on this PC and sends nothing. "
                              "FoodOptimizer.exe and its Python connect only to 127.0.0.1.")
    assert _defines("desktop/windows/tests/test_l2_40_traffic.py::test_our_processes_connect_only_to_loopback")
    assert _defines("desktop/windows/tests/test_l2_40_traffic.py::test_the_capture_sees_our_python")
    assert "test_our_processes_connect_only_to_loopback" in WIN_README.read_text(encoding="utf-8")


def test_it_requirements_line_is_the_readmes():
    assert _it_lines()[4] == REQUIREMENTS
    assert _it_lines()[4] + " Works offline." == _requirements()


def test_it_contact_is_get_helps_address():
    assert _it_lines()[5] == f"Contact: {CONTACT}"
    assert CONTACT in (REPO / "desktop" / "shared" / "messages.json").read_text(encoding="utf-8")


def test_notes_links_are_the_two_downloads():
    urls = re.findall(r"https://\S+?(?=[)\s])", NOTES.read_text(encoding="utf-8"))
    assert urls == [_url("dmg"), _url("exe")]


def test_notes_pending_markers():
    assert {m.group(1) for m in PENDING.finditer(NOTES.read_text(encoding="utf-8"))} == {
        "publisher shown on the signed build", "exact SmartScreen words on a signed build",
        "Edge download warning words, from checklist W1"}


def test_start_here_pending_markers():
    src = REPO / "desktop" / "shared" / "start_here.src.md"
    assert {m.group(1) for m in PENDING.finditer(src.read_text(encoding="utf-8"))} == {
        "exact SmartScreen words on a signed build",
        "Edge download warning words, from checklist W1"}


def test_notes_sentences_are_short():
    """Copy rules for the food-scientist parts: short sentences. The IT
    lines may be technical (the command is one long token)."""
    text = _visible(NOTES.read_text(encoding="utf-8"))
    text = text[:text.index("## For IT teams")]
    text = re.sub(r"\*\*\[[^\]]+\]\([^)]+\)\*\*", "", text)
    for sentence in _sentences(re.sub(r"^#+ .*$|^- ", "", text, flags=re.M)):
        assert len(sentence.split()) <= 15, sentence


# ---------- the download email ----------

def test_email_is_four_lines():
    """The direct .exe link, the Edge line, Start Here's SmartScreen sentence
    and where to ask: no size, steps, subject or attachment lines."""
    lines = _lines(EMAIL.read_text(encoding="utf-8"))
    assert lines == [f"Food Optimizer for Windows: {_url('exe')}", EDGE, _start_here_smartscreen(),
                     f"Questions: {CONTACT}"]


def test_email_links_straight_to_the_exe():
    urls = re.findall(r"https?://\S+", EMAIL.read_text(encoding="utf-8"))
    assert urls == [_url("exe")]


def test_email_lines_follow_the_copy_rules():
    first, _, _, last = _lines(EMAIL.read_text(encoding="utf-8"))
    for line in (first.split(": https://")[0], last):
        assert len(line.split()) <= 10, line


def test_email_pending_markers():
    """The SmartScreen sentence names the publisher, which only a signed
    build shows: W1 records the real words before the tag."""
    assert {m.group(1) for m in PENDING.finditer(EMAIL.read_text(encoding="utf-8"))} == {
        "exact SmartScreen words on a signed build",
        "Edge download warning words, from checklist W1"}


# ---------- the root README ----------

def test_readme_windows_section_is_the_link_and_requirements():
    section = _section(README.read_text(encoding="utf-8"), "## Install on Windows")
    assert _lines(section) == ["## Install on Windows",
                               f"[**Download FoodOptimizer-{VERSION}-Windows.exe**]({_url('exe')})",
                               _requirements()]
    assert "PENDING(" not in section


def test_readme_holds_no_pending_marker():
    assert "PENDING(" not in README.read_text(encoding="utf-8")


# ---------- README and walkthrough wording ----------

def test_readme_names_the_real_restart_label():
    text = README.read_text(encoding="utf-8")
    assert "Start this example over" in text
    assert "sample over" not in text


def test_readme_numbers_tip_is_in_the_mac_section():
    text = README.read_text(encoding="utf-8")
    tip = "In Numbers, select the sheet tabs if workbook links do not open."
    assert text.count(tip) == 1
    mac = text[:text.index("## Install on Windows")]
    assert tip in mac


def test_walkthrough_has_no_numbered_step_references():
    text = (REPO / "desktop" / "shared" / "manual_walkthrough.md").read_text(encoding="utf-8")
    assert not re.search(r"step \d", text), re.findall(r".{20}step \d.{10}", text)
    assert 'in "A project of your own"' in text
