"""The Windows installer as text (Level 1, phase 5): its copy in
desktop/windows/installer_messages.isl (Inno's override format, layered on
Default.isl), the setup.iss directives and sections the spec fixes, and the
build script's installer steps. They run on any OS; the VM tests compile
setup.iss for real."""
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
WINDOWS = REPO / "desktop" / "windows"
ISL = WINDOWS / "installer_messages.isl"
ISS = WINDOWS / "setup.iss"


def parse_isl(path) -> dict[str, dict[str, str]]:
    """section -> key -> value of an Inno messages file. Comment lines start
    with ';'; a value runs to the end of its line. Task 11's C# loader
    mirrors this."""
    out: dict[str, dict[str, str]] = {}
    section = None
    for line in pathlib.Path(path).read_text(encoding="ascii").split("\n"):
        line = line.rstrip("\r")
        if not line.strip() or line.lstrip().startswith(";"):
            continue
        m = re.fullmatch(r"\[(\w+)\]", line.strip())
        if m:
            section = m.group(1)
            out.setdefault(section, {})
            continue
        key, sep, value = line.partition("=")
        assert sep and section, f"not a key=value line: {line!r}"
        out[section][key.strip()] = value
    return out


# The approved copy (owner, 2026-10-24): (section, keys, kind, text).
# Kind: T = title, L = label/button, B = body.
COPY = [
    ("Messages", ["SetupAppTitle"], "T", "Food Optimizer Setup"),
    ("Messages", ["SetupWindowTitle"], "T", "Install %1"),
    ("Messages", ["UninstallAppTitle"], "T", "Uninstall Food Optimizer"),
    ("Messages", ["UninstallAppFullTitle"], "T", "Uninstall %1"),
    ("Messages", ["WelcomeLabel1"], "T", "Install [name]"),
    ("Messages", ["WelcomeLabel2"], "B", "Version [ver]. Works offline.%n%nUpdating keeps your projects."),
    ("Messages", ["WizardPreparing"], "T", "Getting ready"),
    ("Messages", ["PreparingDesc"], "B", "Getting ready to install [name]."),
    ("Messages", ["CannotContinue"], "B", "Click Cancel to close."),
    ("Messages", ["ApplicationsFound", "ApplicationsFound2"], "B",
     "These apps use files the installer must update."),
    ("Messages", ["CloseApplications"], "L", "&Close them automatically"),
    ("Messages", ["DontCloseApplications"], "L", "&Leave them open"),
    ("Messages", ["ErrorCloseApplications"], "B",
     "Some apps could not be closed. Close them, then click Next."),
    ("Messages", ["WizardInstalling"], "T", "Installing"),
    ("Messages", ["InstallingLabel"], "B", "Installing [name] on this PC."),
    ("Messages", ["StatusClosingApplications"], "B", "Closing Food Optimizer"),
    ("Messages", ["StatusCreateDirs"], "B", "Creating folders"),
    ("Messages", ["StatusExtractFiles"], "B", "Copying files"),
    ("Messages", ["StatusCreateIcons"], "B", "Adding shortcuts"),
    ("Messages", ["StatusCreateIniEntries", "StatusCreateRegistryEntries"], "B", "Saving settings"),
    ("Messages", ["StatusSavingUninstall"], "B", "Saving uninstall details"),
    ("Messages", ["StatusRunProgram"], "B", "Finishing"),
    ("Messages", ["StatusRollback"], "B", "Undoing changes"),
    ("Messages", ["FinishedHeadingLabel"], "T", "[name] is installed"),
    ("Messages", ["FinishedLabelNoIcons", "FinishedLabel"], "B", "Open it from Start."),
    ("Messages", ["ClickFinish"], "B", "Click Finish to close."),
    ("Messages", ["ExitSetupTitle"], "T", "Stop installing?"),
    ("Messages", ["ExitSetupMessage"], "B", "Food Optimizer is not installed yet. Stop installing?"),
    ("Messages", ["SetupAppRunningError", "UninstallAppRunningError"], "B", "Close %1, then click OK."),
    ("Messages", ["ConfirmUninstall"], "B", "Uninstall %1? Your projects are kept."),
    ("Messages", ["UninstallStatusLabel"], "B", "Uninstalling %1 from this PC."),
    ("Messages", ["UninstalledAll"], "B",
     "%1 is uninstalled.%n%nYour projects stay in your user folder, under FoodOptimizer."),
    ("Messages", ["UninstalledMost"], "B", "%1 is uninstalled. Restart this PC to delete the last files."),
    ("Messages", ["UninstalledAndNeedsRestart"], "B", "Restart this PC to finish uninstalling %1."),
    ("Messages", ["WinVersionTooLowError", "OnlyOnTheseArchitectures"], "B",
     "This PC is not supported.%n%nNeeds 64-bit Windows 10 (22H2) or Windows 11."),
    ("Messages", ["AdminPrivilegesRequired", "PowerUserPrivilegesRequired"], "B",
     "Installing for all users needs admin rights."),
    ("Messages", ["SetupFileCorrupt", "SetupFileCorruptOrWrongVer"], "B",
     "This installer is damaged. Download it again."),
    ("Messages", ["LdrCannotCreateTemp", "LdrCannotExecTemp"], "B",
     "The installer could not start. Free some disk space, then try again."),
    ("Messages", ["ErrorReplacingExistingFile"], "B",
     "A file could not be replaced. Close Food Optimizer, then try again."),
    ("Messages", ["DiskSpaceWarningTitle"], "T", "Not enough space"),
    ("Messages", ["DiskSpaceWarning"], "B", "Needs %1 KB free. This drive has %2 KB.%n%nContinue anyway?"),
    ("CustomMessages", ["DesktopShortcut"], "L", "Add a desktop shortcut"),
    ("CustomMessages", ["OpenApp"], "L", "Open Food Optimizer"),
    ("CustomMessages", ["OpenStartHere"], "L", "Open Start Here"),
    ("CustomMessages", ["PrewarmStatus"], "B", "Loading the app's components"),
    ("CustomMessages", ["PrewarmFailed"], "B",
     "Food Optimizer could not load its components.%n%nAsk IT to allow Food Optimizer, "
     "or email sohum.patnaik@fsi.org."),
    ("CustomMessages", ["InstalledByIT"], "B", "Installed by your IT. Ask IT to update."),
    ("CustomMessages", ["SystemNeedsAllUsers"], "B", "Installing as SYSTEM needs /ALLUSERS."),
    ("CustomMessages", ["FolderTooLong"], "B",
     "This install folder path is too long. Use at most 80 characters."),
    ("CustomMessages", ["NeedsSpace"], "B", "Needs %1 GB free on this PC."),
]
REWRITTEN = {(section, key): text for section, keys, _kind, text in COPY for key in keys}


def _flatten(text):
    """The words a reader sees: no accelerator '&', '%n' as a space, and the
    placeholders as one word each."""
    text = text.replace("&", "").replace("%n", " ")
    return re.sub(r"\[name\]|\[ver\]|%\d", "X", text).strip()


def _words(sentence):
    return len(sentence.split())


@pytest.mark.parametrize("path", [ISL], ids=lambda p: p.name)
def test_installer_texts_are_ascii(path):
    data = path.read_bytes()
    assert all(b < 0x80 for b in data), path.name
    assert b"\r" not in data, path.name


def test_every_reachable_message_is_ours():
    parsed = parse_isl(ISL)
    assert set(parsed) == {"Messages", "CustomMessages"}
    for (section, key), text in REWRITTEN.items():
        assert parsed[section].get(key) == text, (section, key)
    extra = {(s, k) for s, keys in parsed.items() for k in keys} - set(REWRITTEN)
    assert extra == set()


def test_installer_messages_follow_the_copy_rules():
    from tests.test_food_bo import _BACKUP, _BANNED
    vocab = re.compile(r"\b(wizard|computer|applications?|remov\w*|programs?)\b", re.I)
    mac = re.compile("Finder|this Mac|\\bMac\\b|Cmd|\u2318|Applications")
    parsed = parse_isl(ISL)
    for section, keys, kind, _approved in COPY:
        for key in keys:
            text = parsed[section][key]
            flat = _flatten(text)
            if kind == "T":
                assert _words(flat) <= 8, (key, flat)
            elif kind == "L":
                assert _words(flat) <= 5, (key, flat)
            else:
                for sentence in re.split(r"(?<=[.!?])\s+", flat):
                    assert _words(sentence) <= 10, (key, sentence)
            plain = text.replace("&", "")
            assert not vocab.search(plain), (key, plain)
            assert not mac.search(plain), (key, plain)
            for pattern in [*_BANNED, _BACKUP]:
                assert not pattern.search(plain), (key, pattern.pattern, plain)


def test_prewarm_failure_names_it_and_the_address():
    custom = parse_isl(ISL)["CustomMessages"]
    assert "IT" in custom["PrewarmFailed"].split()
    assert "sohum.patnaik@fsi.org" in custom["PrewarmFailed"]
    assert custom["InstalledByIT"] == "Installed by your IT. Ask IT to update."
