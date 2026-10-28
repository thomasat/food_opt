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
    # Added after the final review (refusing a /DIR that holds other files);
    # awaiting the owner's approval.
    ("CustomMessages", ["FolderNotOurs"], "B", "This folder holds other files. Use an empty folder."),
    # Added in hardening; approved by the owner 2026-10-25.
    ("Messages", ["WizardUninstalling"], "T", "Uninstalling"),
    ("Messages", ["StatusUninstalling"], "B", "Uninstalling %1"),
    ("Messages", ["AbortRetryIgnoreSelectAction"], "T", "Choose what to do"),
    ("Messages", ["AbortRetryIgnoreRetry"], "L", "&Try again"),
    ("Messages", ["AbortRetryIgnoreIgnore"], "L", "&Continue anyway"),
    ("Messages", ["AbortRetryIgnoreCancel"], "L", "Stop installing"),
]
REWRITTEN = {(section, key): text for section, keys, _kind, text in COPY for key in keys}


def _flatten(text):
    """The words a reader sees: no accelerator '&', '%n' as a space, and the
    placeholders as one word each."""
    text = text.replace("&", "").replace("%n", " ")
    return re.sub(r"\[name\]|\[ver\]|%\d", "X", text).strip()


def _words(sentence):
    return len(sentence.split())


@pytest.mark.parametrize("path", [ISL, ISS], ids=lambda p: p.name)
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


# ---------- setup.iss (Task 4) ----------

APP_ID = "{DE876ED9-927A-45AD-AD9B-D6E4B4C9A4A3}"
SETUP_NAME = "FoodOptimizer.Setup.0E00CC2F-4E51-4EBF-A229-A910D5742A39"
SETUP_MUTEX = "{code:SetupMutexNames}"
APP_IDENTITY = WINDOWS / "FoodOptimizer" / "FoodOptimizer.Core" / "AppIdentity.cs"


def _iss_lines():
    """setup.iss's lines with ';' comment lines and ISPP '#' lines dropped,
    as (section, line)."""
    section = None
    for raw in ISS.read_text(encoding="ascii").split("\n"):
        line = raw.strip()
        if not line or line.startswith(";") or line.startswith("#") or line.startswith("//"):
            continue
        m = re.fullmatch(r"\[(\w+)\]", line)
        if m:
            section = m.group(1)
            continue
        yield section, line


def setup_directives() -> dict[str, str]:
    out = {}
    for section, line in _iss_lines():
        if section == "Setup":
            key, _, value = line.partition("=")
            assert key.strip() not in out, f"[Setup] {key} twice"
            out[key.strip()] = value.strip()
    return out


def _entry(line):
    """'Name: "a"; Flags: x y' -> {"Name": "a", "Flags": "x y"}."""
    out = {}
    for part in re.findall(r'(\w+):\s*("(?:[^"]|"")*"|[^;]*)\s*(?:;|$)', line):
        key, value = part
        value = value.strip()
        if value.startswith('"'):
            value = value[1:-1].replace('""', '"')
        out[key] = value
    return out


def section_entries(name) -> list[dict[str, str]]:
    return [_entry(line) for section, line in _iss_lines() if section == name]


SETUP_PINS = {
    "AppId": "{" + APP_ID, "AppName": "Food Optimizer", "AppVersion": "{#AppVersion}",
    "AppVerName": "Food Optimizer {#AppVersion}",
    "AppPublisher": "Food System Innovations", "AppContact": "sohum.patnaik@fsi.org",
    "AppSupportURL": "mailto:sohum.patnaik@fsi.org",
    "VersionInfoVersion": "{#AppVersion}.0", "VersionInfoProductVersion": "{#AppVersion}",
    "VersionInfoCompany": "Food System Innovations", "VersionInfoProductName": "Food Optimizer",
    "VersionInfoDescription": "Food Optimizer Setup",
    "PrivilegesRequired": "lowest", "PrivilegesRequiredOverridesAllowed": "commandline",
    "UsePreviousPrivileges": "no",
    "ArchitecturesAllowed": "x64compatible", "ArchitecturesInstallIn64BitMode": "x64compatible",
    "MinVersion": "10.0.19045",
    "DefaultDirName": "{autopf}\\Food Optimizer", "DisableDirPage": "yes",
    "DisableProgramGroupPage": "yes", "DisableReadyPage": "yes", "DisableWelcomePage": "no",
    "ShowLanguageDialog": "no",
    "SetupMutex": SETUP_MUTEX,
    "CloseApplications": "yes", "CloseApplicationsFilter": "*.exe,*.dll,*.pyd",
    "RestartApplications": "no",
    "SetupLogging": "yes", "UninstallDisplayName": "Food Optimizer",
    "UninstallDisplayIcon": "{app}\\FoodOptimizer.exe",
    "SetupIconFile": "icon.ico", "WizardStyle": "modern",
    "WizardImageFile": "installer_art\\wizard_100.bmp,installer_art\\wizard_150.bmp,installer_art\\wizard_200.bmp",
    "WizardSmallImageFile": "installer_art\\small_100.bmp,installer_art\\small_150.bmp,installer_art\\small_200.bmp",
    "Compression": "lzma2/max", "SolidCompression": "yes", "LZMANumBlockThreads": "4",
}


def _identity():
    text = APP_IDENTITY.read_text(encoding="utf-8")
    return {name: re.search(r'const string %s = @?"([^"]*)"' % name, text).group(1)
            for name in ("AppUserModelId", "SingleInstanceMutex", "RunningPerUser", "RunningAllUsers")}


def test_setup_directives_are_pinned():
    found = setup_directives()
    assert {k: found.get(k) for k in SETUP_PINS} == SETUP_PINS


def test_mutexes_and_app_id_match_the_shell():
    """No AppMutex: it is one list for both scopes, so the all-users shell
    would block the per-user uninstaller it runs. Pascal checks the
    running app's name per scope (test_running_checks_are_per_scope)."""
    ident = _identity()
    assert "AppMutex" not in setup_directives()
    assert re.search(r"^\s*AppMutex\s*=", ISS.read_text(encoding="ascii"), re.M | re.I) is None
    ids = [e["AppUserModelID"] for e in section_entries("Icons") if "AppUserModelID" in e]
    assert ids and set(ids) == {ident["AppUserModelId"]}


def test_setup_mutex_is_per_scope():
    """One installer at a time, per scope (I2): a per-user installer checks
    and holds only this session's name, so another user's installer (or
    anyone holding the Global name) can't hold it up; an all-users one also
    the Global name, every session's. Inno expands SetupMutex after
    InitializeSetup, so {code:} works there."""
    assert setup_directives()["SetupMutex"] == "{code:SetupMutexNames}"
    body = _function("SetupMutexNames")
    assert body.startswith("function SetupMutexNames(Param: String): String;")
    assert re.search(r"Result := '%s';\s+if IsAdminInstallMode then\s+"
                     r"Result := Result \+ ',Global\\%s';" % (re.escape(SETUP_NAME), re.escape(SETUP_NAME)),
                     body), body
    # The Global setup name appears nowhere else.
    assert ISS.read_text(encoding="ascii").count("Global\\" + SETUP_NAME) == 1


def test_running_checks_are_per_scope():
    """I2 (Review Focus 2): a per-user install or uninstall waits only for
    the app in this session (the Local name, which no other user can make
    here); an all-users one waits for the all-users app of every user (the
    Global name). The all-users installer marks its folder so the shell
    knows which name to hold. OK checks again; Cancel, and every silent run,
    stops with exit code 1 before anything changes, as AppMutex did."""
    ident = _identity()
    per, every = ident["RunningPerUser"], ident["RunningAllUsers"]
    assert per.startswith("Local\\") and every.startswith("Global\\")
    name = _function("RunningName")
    assert re.search(r"if IsAdminInstallMode then\s+Result := '%s'\s+else\s+Result := '%s';"
                     % (re.escape(every), re.escape(per)), name), name
    closed = _function("AppClosed")
    assert "while CheckForMutexes(RunningName) do" in closed
    assert ("if Quiet or (SuppressibleMsgBox(FmtMessage(Msg, ['Food Optimizer']), mbError, MB_OKCANCEL, IDCANCEL)"
            " <> IDOK) then begin") in closed
    assert closed.index("Log('refused: app_running')") < closed.index("Result := False;")
    # Setup checks where AppMutex was checked (Inno runs InitializeSetup
    # just before it); the uninstaller after the user confirms, at the step
    # Inno gives its own check (usAppMutexCheck), where Abort ends it.
    assert re.search(r"Result := AppClosed\(SetupMessage\(msgSetupAppRunningError\), False\);",
                     _function("InitializeSetup"))
    unins = _function("CurUninstallStepChanged")
    assert re.search(r"if \(CurUninstallStep = usAppMutexCheck\) and\s+"
                     r"not AppClosed\(SetupMessage\(msgUninstallAppRunningError\), UninstallSilent\) then\s+Abort;",
                     unins), unins
    assert "InitializeUninstall" not in _code_section()
    # The marker: copied only by the all-users installer, as a [Files]
    # entry, so a failed copy fails the install (Inno's file error; Abort
    # under /SUPPRESSMSGBOXES) instead of an IT copy that holds the
    # per-user name (test_the_all_users_marker_is_a_files_entry).
    assert MARKER_ENTRY in section_entries("Files")
    assert "SaveStringToFile(" not in _code_section()
    layout = (WINDOWS / "FoodOptimizer" / "FoodOptimizer.Core" / "InstallLayout.cs").read_text(encoding="utf-8")
    assert 'File.Exists(Path.Combine(Root, "all-users"))' in layout


def test_languages_layer_our_messages():
    assert section_entries("Languages") == [
        {"Name": "en", "MessagesFile": "compiler:Default.isl,installer_messages.isl"}]


MARKER_ENTRY = {"Source": "all-users", "DestDir": "{app}", "Flags": "ignoreversion", "Check": "IsAdminInstallMode"}


def test_one_files_entry_copies_the_image():
    """The image, then the all-users marker (all-users installs only)."""
    assert section_entries("Files") == [
        {"Source": "{#ImageDir}\\*", "DestDir": "{app}",
         "Flags": "ignoreversion recursesubdirs createallsubdirs"},
        MARKER_ENTRY]


def test_the_all_users_marker_is_a_files_entry():
    """Final review (ruling): the marker tells the shell it is the IT copy
    (the Global running name, no hand-off). A [Files] entry, not a Pascal
    write: a copy that fails stops the install (Inno's file error box;
    Abort, then rollback, under /SUPPRESSMSGBOXES), where a failed
    SaveStringToFile was only logged. Its source is a file of its own
    beside setup.iss; Inno deletes it at uninstall as it does every file
    it copied. Level 2: test_an_it_install_that_cannot_copy_its_marker_fails."""
    source = WINDOWS / MARKER_ENTRY["Source"]
    assert source.is_file()
    assert source.read_bytes() == b"all-users\n"
    assert section_entries("Files")[-1] == MARKER_ENTRY
    assert "{app}\\all-users" not in ISS.read_text(encoding="ascii")
    layout = (WINDOWS / "FoodOptimizer" / "FoodOptimizer.Core" / "InstallLayout.cs").read_text(encoding="utf-8")
    assert 'File.Exists(Path.Combine(Root, "all-users"))' in layout


def test_install_delete_clears_the_bundled_folders():
    assert section_entries("InstallDelete") == [
        {"Type": "filesandordirs", "Name": "{app}\\python"},
        {"Type": "filesandordirs", "Name": "{app}\\app"},
        {"Type": "filesandordirs", "Name": "{app}\\shared"},
        {"Type": "filesandordirs", "Name": "{app}\\runtimes"},
        {"Type": "files", "Name": "{app}\\*.dll"}]
    assert section_entries("UninstallDelete") == [{"Type": "dirifempty", "Name": "{app}"}]


def test_three_shortcuts():
    icons = {e["Name"]: e for e in section_entries("Icons")}
    assert set(icons) == {"{autoprograms}\\Food Optimizer", "{autoprograms}\\Food Optimizer - Start Here",
                          "{autodesktop}\\Food Optimizer"}
    app = icons["{autoprograms}\\Food Optimizer"]
    assert (app["Filename"], app["WorkingDir"]) == ("{app}\\FoodOptimizer.exe", "{app}")
    assert "Check" not in app
    assert icons["{autoprograms}\\Food Optimizer - Start Here"]["Filename"] == "{app}\\Start Here.txt"
    desktop = icons["{autodesktop}\\Food Optimizer"]
    assert desktop["Check"] == "WantDesktopIcon"
    assert {k: v for k, v in desktop.items() if k not in ("Name", "Check")} == {
        k: v for k, v in app.items() if k != "Name"}


def test_finish_page_offers_app_then_start_here():
    runs = section_entries("Run")
    assert [r["Filename"] for r in runs] == ["{app}\\FoodOptimizer.exe", "{app}\\Start Here.txt"]
    assert [r["Description"] for r in runs] == ["{cm:OpenApp}", "{cm:OpenStartHere}"]
    for r in runs:
        flags = r["Flags"].split()
        assert {"postinstall", "skipifsilent", "runasoriginaluser", "nowait"} <= set(flags), r
    assert "unchecked" not in runs[0]["Flags"].split()
    assert "unchecked" in runs[1]["Flags"].split() and "shellexec" in runs[1]["Flags"].split()


def test_prewarm_imports_the_spec_modules():
    text = ISS.read_text(encoding="ascii")
    assert "-I -X utf8 -B -c" in text
    assert "import torch, botorch, gpytorch, pandas, openpyxl, streamlit" in text
    assert "ExecAndLogOutput(" in text and "faulthandler.dump_traceback_later({#PrewarmSecs}, exit=True)" in text


def test_root_budget_matches_the_image_gate():
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_env", WINDOWS / "build_env.py")
    be = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(be)
    text = ISS.read_text(encoding="ascii")
    assert re.search(r"^#define MaxRoot (\d+)$", text, re.M).group(1) == str(be.ROOT_BUDGET)
    assert re.search(r"^#define PrewarmSecs (\d+)$", text, re.M).group(1) == "600"


REFUSALS = ("system_needs_allusers", "installed_by_it", "folder_too_long", "folder_not_ours", "needs_space")


def test_every_refusal_is_logged():
    text = ISS.read_text(encoding="ascii")
    for refusal in REFUSALS:
        assert f"Log('refused: {refusal}')" in text, refusal
    # In the order the plan fixes: scope first, then folder, then space.
    assert [text.index(f"refused: {r}") for r in REFUSALS] == sorted(
        text.index(f"refused: {r}") for r in REFUSALS)


def test_every_custom_message_is_used():
    text = ISS.read_text(encoding="ascii")
    for key in parse_isl(ISL)["CustomMessages"]:
        assert f"{{cm:{key}}}" in text or f"CustomMessage('{key}')" in text, key


def test_the_state_folder_is_deleted_only_by_its_name():
    text = ISS.read_text(encoding="ascii")
    assert "FOODOPT_STATE_DIR" in text and "FOODOPT_PROJECTS_DIR" in text
    assert "CompareText(ExtractFileName(" in text and "'FoodOptimizer') = 0" in text


def test_wizard_art_matches_the_logo():
    pytest.importorskip("PIL")
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_icon", WINDOWS / "make_icon.py")
    make_icon = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(make_icon)
    assert make_icon.main(["--check"]) == 0
    sizes = {"wizard_100": (164, 314), "wizard_150": (246, 471), "wizard_200": (328, 628),
             "small_100": (55, 55), "small_150": (83, 83), "small_200": (110, 110)}
    import struct
    for name, (w, h) in sizes.items():
        data = (WINDOWS / "installer_art" / f"{name}.bmp").read_bytes()
        assert data[:2] == b"BM", name
        width, height = struct.unpack_from("<ii", data, 18)
        bits = struct.unpack_from("<H", data, 28)[0]
        assert (width, abs(height), bits) == (w, h, 24), name


def test_setup_refuses_an_old_inno():
    text = ISS.read_text(encoding="ascii")
    assert "EncodeVer(6,3,0)" in text
    assert "#error Inno Setup 6.3 or later is required" in text
    # LockFolder's handles and pointers are Longint: Inno 6's Setup is
    # 32-bit; Inno 7 can build a 64-bit one (fix round 3).
    assert re.search(r"^#if VER >= EncodeVer\(7,0,0\)\n  #error [^\n]*Longint[^\n]*\n#endif$", text, re.M), text[:1500]


# ---------- build_installer.ps1 builds the installer (Task 5) ----------

BUILD = WINDOWS / "build_installer.ps1"
SIZE_GATE = WINDOWS / "size_gate.json"
BUILD_MESSAGES = (
    "refusing to build: Inno Setup 6.3 or later (ISCC.exe) not found",
    "$step = 'installer'",
    "ISCC exit $iscc",
    "size gate: not set (installer $installerBytes bytes, image $imageBytes bytes)",
    "build failed: size gate is not set (desktop/windows/size_gate.json)",
    # "build failed: size gate: <what> is ...": the step name is the prefix.
    "$step = 'size gate'",
    "$what is $mb MB; ceiling $ceiling MB (raise it in size_gate.json only on purpose)",
    "Built: $Installer ($installerMb MB, sha256 $installerSha)",
)


def _unquoted(text):
    """PowerShell's string concatenation taken off ('a ' + $x + ' b' reads
    `a $x b`), as tests/test_windows_build_script.py reads refusals."""
    return re.sub(r"'\s*\+\s*|\s*\+\s*'", "", text)


def test_build_script_builds_the_installer():
    text = BUILD.read_text(encoding="ascii")
    for needle in ("dotnet publish", "--shell", "setup.iss", "size_gate.json", "-Windows",
                   "/DDiskFreeGB=", "DOTNET_CLI_TELEMETRY_OPTOUT", "DOTNET_NOLOGO"):
        assert needle in text, needle
    flat = _unquoted(text)
    for message in BUILD_MESSAGES:
        assert message in flat, message
    # The ISCC lookup order of the plan (and of make_test_installer.ps1).
    order = [flat.index(s) for s in ("$env:FOODOPT_ISCC", "Get-Command ISCC.exe",
                                     "Programs\\Inno Setup 6\\ISCC.exe", "ProgramFiles(x86)",
                                     "$env:ProgramFiles")]
    assert order == sorted(order)


def test_release_record_keys_follow_the_image_record():
    text = BUILD.read_text(encoding="ascii")
    keys = re.findall(r'"([a-z0-9_]+)=', text) + re.findall(r"'([a-z0-9_]+)=", text)
    tail = ["inno_setup", "installer", "installer_bytes", "installer_sha256", "disk_free_gb",
            "size_gate", "signed"]
    for key in tail + ["dotnet_sdk", "shell_files"]:
        assert key in keys, key
    # The per-file list beside the installer, with its hash, is the last
    # key before 'signed=no'; no SBOM keys.
    block = text[text.index("Write-Lf $InstallerRecord"):]
    block = block[:block.index("))")]
    after = block[block.index("installer_sha256=") + 1:]
    found = re.findall(r'"([a-z0-9_]+)=|\'([a-z0-9_]+)=', after)
    assert [a or b for a, b in found] == ["disk_free_gb", "size_gate", "files_list_sha256", "signed"]
    for path in [BUILD, REPO / "desktop" / "README.md", WINDOWS / "README.md"]:
        assert "sbom" not in path.read_text(encoding="utf-8").lower(), path
    # The tests too, except this assertion and the release notes' banned words.
    # test_windows_docs.py pins the design note's "No SBOM" line.
    allowed = {pathlib.Path(__file__).resolve(), (REPO / "tests" / "test_release_texts.py").resolve(),
               (REPO / "tests" / "test_windows_docs.py").resolve()}
    for path in sorted((REPO / "tests").rglob("*.py")):
        if path.resolve() not in allowed:
            assert "sbom" not in path.read_text(encoding="utf-8").lower(), path


def test_every_files_source_gets_the_commit_time_before_iscc():
    """Inno stores each [Files] source's last-write time, so one commit gives
    one installer from any checkout only if every source has the commit's
    time: the image's files, and every file of the repo a [Files] entry
    names (the all-users marker kept its checkout time, gate at 6361cfe).
    The script asks build_env.py for the sources (iss-sources), so a new
    [Files] entry is covered, or fails the build (test_windows_build_env)."""
    text = BUILD.read_text(encoding="ascii")
    times = text[text.index("$step = 'file times'"):text.index("$step = 'installer'")]
    assert "$commitTime = [DateTimeOffset]::FromUnixTimeSeconds($commitSeconds).UtcDateTime" in times
    assert "Invoke-BuildEnv @('iss-sources', $SetupIss)" in times
    assert "if ($source -eq '{#ImageDir}\\*')" in times
    assert "Get-ChildItem -LiteralPath $Image -Recurse -File -Force" in times
    assert "Get-Item -LiteralPath (Join-Path $WindowsDir $source) -Force" in times
    assert times.count(".LastWriteTimeUtc = $commitTime") == 1, times
    # ISCC compiles the same setup.iss, after the times are set.
    assert "& $Iscc '/Qp'" in text[text.index("$step = 'installer'"):]
    assert "$SetupIss = Join-Path $WindowsDir 'setup.iss'" in text
    assert text.count("$SetupIss") == 3 and "(Join-Path $WindowsDir 'setup.iss')" not in text
    # Every [Files] entry's Source is one the script covers.
    sources = [e["Source"] for e in section_entries("Files")]
    assert sources[0] == "{#ImageDir}\\*"
    for source in sources[1:]:
        assert re.fullmatch(r"[\w.\- ]+", source) and (WINDOWS / source).is_file(), source


def test_a_rebuild_over_an_image_with_the_shell_is_allowed():
    """The 3b guard that keeps -OutDir off a home folder accepts the shell's
    published files at the image root, and still refuses a project file."""
    text = BUILD.read_text(encoding="ascii")
    for pattern in ("'*.dll'", "'*.exe'", "'*.pdb'", "'*.xml'", "'FoodOptimizer.*.json'", "'runtimes'"):
        assert pattern in text, pattern


def test_size_gate_is_set():
    import json
    gate = json.loads(SIZE_GATE.read_text(encoding="utf-8"))
    assert set(gate) == {"installer_max_bytes", "image_max_bytes", "disk_free_gb", "measured"}
    for key in ("installer_max_bytes", "image_max_bytes", "disk_free_gb"):
        assert isinstance(gate[key], int) and gate[key] > 0, key
    measured = gate["measured"]
    assert re.fullmatch(r"[0-9a-f]{40}", measured["commit"])
    mib10 = 10 * 1024 * 1024
    for key, ceiling in (("installer_bytes", "installer_max_bytes"), ("image_bytes", "image_max_bytes")):
        expected = -(-int(measured[key] * 1.10) // mib10) * mib10
        assert gate[ceiling] == expected, ceiling
    gib, mib = 1024 ** 3, 1024 ** 2
    assert gate["disk_free_gb"] == -(-(measured["image_bytes"] + 512 * mib) // gib)


def test_windows_requirements_name_the_disk_figure():
    import json
    from tests.shared_texts import flowed, read_lf
    n = json.loads(SIZE_GATE.read_text(encoding="utf-8"))["disk_free_gb"]
    text = flowed(read_lf(WINDOWS / "start_here.txt"))
    assert f"You need a 64-bit PC with Windows 10 (22H2) or Windows 11 and {n} GB free." in text


def test_dist_is_ignored():
    import subprocess
    r = subprocess.run(["git", "check-ignore", "-q", "desktop/windows/dist/x.exe"], cwd=REPO)
    assert r.returncode == 0


def test_no_variable_shadows_a_parameter():
    """PowerShell names are case-insensitive: a loop variable named $check
    is the script's [switch]$Check, and assigning it an array fails (the
    first size-gated build did)."""
    text = BUILD.read_text(encoding="ascii")
    block = text[text.index("param(\n"):text.index("\n)\n", text.index("param(\n"))]
    params = {m.lower() for m in re.findall(r"\[(?:string|switch)\]\$(\w+)", block)}
    assert {"version", "check", "testbuild", "outdir"} <= params
    assigned = {m.lower() for m in re.findall(r"foreach \(\$(\w+) in", text)}
    assigned |= {m.lower() for m in re.findall(r"^\s*\$(\w+)\s*=", text, re.M)}
    assert params & assigned <= {"outdir", "version"}, params & assigned


# ---------- the README's exit-code table (Task 12) ----------

def _l2_constant(name):
    """A literal constant of desktop/windows/tests/l2_support.py, read by
    AST: the module needs Windows to import."""
    import ast
    tree = ast.parse((WINDOWS / "tests" / "l2_support.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"no literal {name} in l2_support.py")


def test_exit_code_table_matches_l2_support():
    codes, intune = _l2_constant("EXIT_CODES"), _l2_constant("INTUNE")
    assert set(codes) == set(intune)
    assert all(isinstance(c, int) for c in codes.values()), codes
    text = (WINDOWS / "README.md").read_text(encoding="utf-8")
    rows = {int(m.group(1)): m.group(3).strip()
            for m in re.finditer(r"^\| `?(-?\d+)`? \| (.+?) \| (\w+) \|$", text, re.M)}
    for name, code in codes.items():
        assert rows.get(code) == intune[name], (name, code, rows)
    assert set(rows) == set(codes.values())


def test_new_scripts_follow_the_powershell_rules():
    """Phase 5's own scripts follow 3b's Windows PowerShell 5.1 rules."""
    for name in ("run_level2.ps1", "make_test_installer.ps1"):
        raw = (WINDOWS / "tests" / name).read_bytes()
        assert all(b < 0x80 for b in raw) and b"\r" not in raw, name
        text = raw.decode("ascii")
        for line in ("Set-StrictMode -Version 2", "$ErrorActionPreference = 'Stop'",
                     "$ProgressPreference = 'SilentlyContinue'"):
            assert line in text, (name, line)
        # No stderr redirection of a native command in PowerShell itself; a
        # .cmd line written as a string literal may redirect.
        code = re.sub(r"'[^'\n]*'", "''", text)
        assert "2>&1" not in code, name
        assert "$LASTEXITCODE" in text, name


# ---------- final-review fixes ----------

def _code_section():
    text = ISS.read_text(encoding="ascii")
    return text[text.index("[Code]"):]


def _function(name):
    code = _code_section()
    start = re.search(r"^(procedure|function) %s\b" % name, code, re.M).start()
    end = re.search(r"^end;$", code[start:], re.M).end()
    return code[start:start + end]


def test_an_upgrade_cannot_be_cancelled_once_files_are_deleted():
    """[InstallDelete] runs before the copy and rollback cannot restore it:
    on the Installing page of an upgrade, Cancel is refused."""
    body = _function("CancelButtonClick")
    assert "wpInstalling" in body and "Upgrading" in body and "Cancel := False" in body
    assert "Upgrading := FileExists(" in _function("PrepareToInstall")


def test_a_folder_that_is_not_ours_is_refused():
    """/DIR at a folder holding other files would let [InstallDelete]
    delete its python, app, shared and *.dll: refused up front."""
    body = _function("PrepareToInstall")
    assert "Log('refused: folder_not_ours')" in body
    assert "CustomMessage('FolderNotOurs')" in body
    assert body.index("refused: folder_not_ours") < body.index("refused: needs_space")
    assert parse_isl(ISL)["CustomMessages"]["FolderNotOurs"] == "This folder holds other files. Use an empty folder."


def test_the_state_folder_is_never_the_projects_folder():
    """Review Focus 2: FOODOPT_STATE_DIR may name the projects folder, or a
    folder holding it, written in another case or with a trailing \\. The
    uninstall then keeps the state folder rather than delete projects."""
    body = _function("CurUninstallStepChanged")
    guard = re.search(r"if SameOrInside\(ProjectsDir, Dir\) then\s+"
                      r"Log\('state folder kept \(it holds the projects folder\): ' \+ Dir\)\s+else ", body)
    assert guard, body
    assert guard.start() < body.index("DelTree(")
    same = _function("SameOrInside")
    assert same.startswith("function SameOrInside(const Inner, Outer: String): Boolean;")
    assert "RemoveBackslashUnlessRoot" in same and "AnsiLowercase" in same
    # Doubled backslashes, "\\.\\", ".." and "/" too: ExpandFileName is GetFullPathName.
    assert "RemoveBackslashUnlessRoot(ExpandFileName(Inner))" in same
    assert "RemoveBackslashUnlessRoot(ExpandFileName(Outer))" in same


# ---------- D1, D2: the IT install leaves users' copies alone ----------

CORE = WINDOWS / "FoodOptimizer" / "FoodOptimizer.Core"


def _routine_at(code, pos):
    """The procedure or function of [Code] that holds position pos."""
    starts = list(re.finditer(r"^(?:procedure|function) (\w+)", code[:pos], re.M))
    return _function(starts[-1].group(1))


def test_the_installer_never_acts_on_other_users():
    """D1 (owner): the elevated installer never deletes in a per-user
    location and never reads another user's profile or hive; a user's own
    next launch runs their per-user copy's uninstaller instead. It no
    longer pins Quick access either (the app does, D2). Every delete names
    this install's folder, this user's state folder or Setup's temp folder,
    directly or through a variable set from one of them."""
    text = ISS.read_text(encoding="ascii")
    for needle in ("HKU", "ProfileList", "ProfileImagePath", "ReplacePerUserCopies", "pintohome",
                   "PinToQuickAccess", "User Shell Folders"):
        assert needle not in text, needle
    code = _code_section()
    allowed = ("{app}", "StateDir", "ExpandConstant('{tmp}')")
    calls = list(re.finditer(r"\b(DelTree|DeleteFile|RemoveDir)\(([^,)]*)", code))
    assert calls
    for call in calls:
        arg = call.group(2).strip()
        if re.fullmatch(r"\w+", arg):
            routine = _routine_at(code, call.start())
            sources = re.findall(r"\b%s := ([^;]*);" % arg, routine)
            assert sources, (call.group(0), routine)
        else:
            sources = [arg]
        for source in sources:
            assert any(a in source for a in allowed), (call.group(0), source)
            # A state folder is a per-user location: only a per-user
            # uninstall deletes one (fix round 1 ruling).
            if "StateDir" in source:
                routine = _routine_at(code, call.start())
                local = routine.index(call.group(0))
                guard = re.search(r"if IsAdminInstallMode then\s+"
                                  r"Log\('state folder kept \(all-users uninstall\): ' \+ Dir\)\s+else ", routine)
                assert guard and guard.start() < local, routine


def test_the_app_finds_a_per_user_copy_where_the_installer_puts_it():
    """D1: the app reads the uninstall entry Inno writes for this AppId and
    falls back to the per-user default folder ({userpf}, which is
    %LOCALAPPDATA%\\Programs, plus DefaultDirName's folder)."""
    cleanup = (CORE / "Windows" / "PerUserCleanup.cs").read_text(encoding="utf-8")
    key = re.search(r"UninstallKey = '([^']+)';", _code_section()).group(1)
    assert f'@"{key}"' in cleanup
    prefix = "{autopf}\\"
    folder = setup_directives()["DefaultDirName"]
    assert folder.startswith(prefix)
    rule = (CORE / "PerUserCopy.cs").read_text(encoding="utf-8")
    assert 'public const string FolderName = "%s";' % folder[len(prefix):] in rule
    assert ('Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), '
            '"Programs", PerUserCopy.FolderName)') in cleanup
    # The switches the README gives IT for a silent uninstall.
    assert '["/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"]' in rule
    # Never with admin rights: the uninstaller found through HKCU and
    # %LOCALAPPDATA% is the user's to replace (fix round 1).
    assert "public static bool ShouldRun(bool allUsers, bool elevated) => allUsers && !elevated;" in rule
    run = cleanup[cleanup.index("public static void Run("):]
    skipped = run.index('log.Write("percopy.skipped", "elevated")')
    assert run.index("PerUserCopy.ShouldRun(") < skipped < run.index("PerUserCopy.Choose(")
    assert "IsSystem" in cleanup and "WindowsBuiltInRole.Administrator" in cleanup
    # The marker the all-users installer writes: never chosen, however spelled.
    assert 'fileExists(Path.Combine(dir, "all-users"))' in rule


def test_a_per_user_copy_hands_off_to_the_it_copy():
    """D1 hand-off (final review): the user's own shortcuts open the
    per-user copy, which never cleans itself up. Opened on a PC with an IT
    copy, it starts the IT copy's FoodOptimizer.exe and exits 0 before it
    holds any name, so the IT copy's cleanup (whose uninstaller checks the
    per-user running name) can run. The IT copy is found read-only: HKLM's
    uninstall entry (the x64 shell reads the 64-bit view, setup.iss's
    HKLM64), else %ProgramW6432%\\Food Optimizer; it must hold the app and
    the all-users marker and not be this copy's folder. The IT copy itself
    never hands off (no loop)."""
    program = (WINDOWS / "FoodOptimizer" / "FoodOptimizer.Shell" / "Program.cs").read_text(encoding="utf-8")
    main = program[program.index("static int Main()"):]
    handoff = main.index("ItCopyHandoff.Run(")
    assert handoff < main.index("SingleInstance.TryAcquire(") < main.index("InstallerMutex.Hold(")
    assert handoff < main.index("WinShell.SetAppUserModelId(")
    assert "if (ItCopyHandoff.Run(layout, log)) return 0;" in main
    run = (CORE / "Windows" / "ItCopyHandoff.cs").read_text(encoding="utf-8")
    body = run[run.index("public static bool Run("):]
    assert body.index("if (layout.AllUsers) return false;") < body.index("ItCopy.HandOffTarget(")
    assert "ItCopy.HandOffTarget(layout.AllUsers, layout.Root, InstallLocation(), DefaultDir(), File.Exists, " \
           "PerUserCleanup.IsReparsePoint)" in body
    # Logged first, then started: no shell execute, its own folder, no arguments.
    assert body.index('log.Write("percopy.handoff", dir);') < body.index("Process.Start(")
    assert "UseShellExecute = false," in body and "WorkingDirectory = dir," in body
    assert "ArgumentList" not in run and "Arguments" not in run
    assert "Registry.LocalMachine.OpenSubKey(PerUserCleanup.UninstallKey)" in run
    assert 'Environment.GetEnvironmentVariable("ProgramW6432")' in run
    assert "PerUserCopy.FolderName" in run
    rule = (CORE / "ItCopy.cs").read_text(encoding="utf-8")
    assert "if (allUsers) return null;" in rule
    assert 'fileExists(Path.Combine(dir, "FoodOptimizer.exe")) && fileExists(Path.Combine(dir, "all-users"))' in rule
    assert "PerUserCopy.OnALink(" in rule and "PerUserCopy.OddName(" in rule


def test_uninstall_keeps_state_when_an_it_copy_exists():
    """D1: the all-users app runs a per-user copy's uninstaller while it
    uses the same state folder (settings, WebView2): a per-user uninstall
    keeps it while an all-users copy is installed."""
    body = _function("CurUninstallStepChanged")
    post = body[body.index("if CurUninstallStep = usPostUninstall then"):]
    keep = re.search(r"else if MachineCopyExists then\s+"
                     r"Log\('state folder kept \(an all-users copy uses it\): ' \+ Dir\)\s+else ", post)
    assert keep, post
    assert keep.start() < post.index("DelTree(")
    assert "Result := RegKeyExists(HKLM64, UninstallKey);" in _function("MachineCopyExists")


def test_a_silent_uninstall_shows_no_box():
    """D1: the app starts the per-user uninstaller with /VERYSILENT
    /SUPPRESSMSGBOXES and nobody sees it. Inno shows no confirmation and
    no "projects are kept" box when silent; our own running-app box is
    skipped when the uninstall is silent, whatever the switches."""
    closed = _function("AppClosed")
    assert closed.startswith("function AppClosed(const Msg: String; const Quiet: Boolean): Boolean;")
    assert re.search(r"if Quiet or \(SuppressibleMsgBox\(", closed), closed
    unins = _function("CurUninstallStepChanged")
    assert "not AppClosed(SetupMessage(msgUninstallAppRunningError), UninstallSilent) then" in unins
    assert "Result := AppClosed(SetupMessage(msgSetupAppRunningError), False);" in _function("InitializeSetup")
    for name in ("CurUninstallStepChanged", "UnpinFromQuickAccess", "QuickAccessItem", "AppPinnedProjects"):
        assert "MsgBox" not in _function(name), name


def test_unpin_only_what_the_app_pinned():
    """D2: the app pins the projects folder at its first launch and records
    it in settings.json; the per-user uninstaller unpins only then (a pin
    the user made stays), and never while an all-users copy uses the
    folder. The installer no longer pins, but still makes the folder."""
    body = _function("CurUninstallStepChanged")
    assert re.search(r"if \(CurUninstallStep = usUninstall\) and not IsAdminInstallMode then begin\s+"
                     r"if MachineCopyExists then\s+"
                     r"Log\('quickaccess: kept \(an all-users copy uses it\)'\)\s+"
                     r"else if AppPinnedProjects then\s+"
                     r"UnpinFromQuickAccess\(ProjectsDir\)\s+"
                     r"else\s+Log\('quickaccess: not pinned by the app'\);\s+end;", body), body
    settings_name = re.search(r'Settings = "([^"]+)"', (CORE / "StateFiles.cs").read_text(encoding="utf-8")).group(1)
    pinned = _function("AppPinnedProjects")
    assert f"LoadStringFromFile(AddBackslash(StateDir) + '{settings_name}', S)" in pinned, pinned
    assert "Pos('\"quickAccessPinned\": true', String(S)) > 0" in pinned, pinned
    # The app writes exactly that text (indented JSON; ShellSettingsTests
    # pins the bytes).
    settings = (CORE / "ShellSettings.cs").read_text(encoding="utf-8")
    assert 'j.WriteBoolean("quickAccessPinned", QuickAccessPinned);' in settings
    # A folder already listed when the app looked (final review) is only
    # "checked": the uninstaller never reads that flag.
    assert 'j.WriteBoolean("quickAccessChecked", QuickAccessChecked);' in settings
    assert "quickAccessChecked" not in pinned
    assert "Indented = true" in settings
    assert re.search(r"if not IsAdminInstallMode then begin\s+if not ForceDirectories\(ProjectsDir\) then",
                     _function("CurStepChanged"))


def test_an_it_install_outside_program_files_is_locked_down():
    """M4: under Program Files an all-users copy inherits its rule (users
    read and run it, never change it). A /DIR elsewhere would inherit the
    drive's (C:\\ lets every signed-in user change a new folder). Fix
    round 2: before any file is copied (last in PrepareToInstall), an
    admin-mode install outside {commonpf64} gives the folder Program Files'
    rule in ONE call, on a handle to the folder itself (never what a link
    points at), held open with no delete sharing until the copy ends, so
    neither it nor a folder above it can be swapped. Refused, with nothing
    changed, when it or a folder above it is a link, when it holds anything
    (unless it is the registered all-users folder), or when a user owns it;
    refused when the lock fails or something got in first. No icacls, no
    recursion."""
    code = _code_section()
    text = ISS.read_text(encoding="ascii")
    assert "icacls" not in code and "/T" not in code and not re.search(r"\bExec\(", code)
    # The rule: owner Administrators, protected, SY and BA full, Users read and run.
    assert "AppDirRule = 'O:BAD:P(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)(A;OICI;0x1200a9;;;BU)';" in code
    generic_read = 0x00120089      # FILE_GENERIC_READ
    generic_execute = 0x001200A0   # FILE_GENERIC_EXECUTE
    assert generic_read | generic_execute == 0x1200A9
    # CreateFileW: READ_CONTROL | WRITE_DAC | WRITE_OWNER | FILE_READ_ATTRIBUTES
    # | FILE_LIST_DIRECTORY; share read+write only; OPEN_EXISTING;
    # BACKUP_SEMANTICS | OPEN_REPARSE_POINT.
    assert "AppDirAccess = $E0081;" in code and 0x20000 | 0x40000 | 0x80000 | 0x80 | 0x1 == 0xE0081
    assert "AppDirShare = 3;" in code and "AppDirOpenExisting = 3;" in code
    assert "AppDirFlags = $02200000;" in code and 0x02000000 | 0x00200000 == 0x02200000
    assert "OwnerDaclProtected = -2147483643;" in code and (-2147483643) & 0xFFFFFFFF == 0x80000005
    for name, dll in (("CreateFileW", "kernel32"), ("CloseHandle", "kernel32"), ("LocalFree", "kernel32"),
                      ("ConvertStringSecurityDescriptorToSecurityDescriptorW", "advapi32"),
                      ("GetSecurityDescriptorOwner", "advapi32"), ("GetSecurityDescriptorDacl", "advapi32"),
                      ("GetSecurityInfo", "advapi32"), ("SetSecurityInfo", "advapi32"),
                      ("IsWellKnownSid", "advapi32")):
        assert f"external '{name}@{dll}.dll stdcall';" in code, name

    link = _function("OnALink")
    assert "FindRec.Attributes and FILE_ATTRIBUTE_REPARSE_POINT <> 0" in link
    assert "Up := ExtractFileDir(P);" in link and "P := Up;" in link

    lock = _function("LockFolder")
    # PrepareToInstall can run twice: an earlier handle is closed first.
    assert lock.index("CloseAppDir;") < lock.index("AppDirHandle := CreateFileW(")
    steps = ["AppDirHandle := CreateFileW(App, AppDirAccess, AppDirShare, 0, AppDirOpenExisting, AppDirFlags, 0);",
             "if OnALink(App) or not OwnedByAdmins then",
             "ConvertStringSecurityDescriptorToSecurityDescriptorW(AppDirRule, 1, Descriptor, 0)",
             "Rc := SetSecurityInfo(AppDirHandle, SeFileObject, OwnerDaclProtected, Owner, 0, Dacl, 0);",
             "LocalFree(Descriptor);",
             "Log('acl: locked ' + App);"]
    at = [lock.index(step) for step in steps]
    assert at == sorted(at), lock
    assert lock.count("SetSecurityInfo(") == 1 and "SetNamedSecurityInfo" not in code
    owned = _function("OwnedByAdmins")
    assert "WinBuiltinAdministratorsSid" in owned and "WinLocalSystemSid" in owned and "LocalFree(Descriptor);" in owned
    assert "WinLocalSystemSid = 22;" in code and "WinBuiltinAdministratorsSid = 26;" in code

    down = _function("LockDownAppDir")
    assert down.startswith("function LockDownAppDir(const App: String): Boolean;")
    assert re.search(r"if SameOrInside\(App, ExpandConstant\('\{commonpf64\}'\)\) then\s+exit;", down), down
    # Links and a filled folder are refused before anything changes: no
    # folder is made and no rule is set until both checks pass.
    order = ["if OnALink(App) then", "if not RegisteredHere(App) and DirHasEntries(App) then begin",
             "if not ForceDirectories(App) then begin", "if LockFolder(App) then begin",
             "if RegisteredHere(App) or not DirHasEntries(App) then", "CloseAppDir;"]
    at = [down.index(step) for step in order]
    assert at == sorted(at), down
    for word in ("Abort", "MsgBox", "RaiseException", "DelTree", "DeleteFile"):
        assert word not in down + lock + link, word

    # Admin mode only, last in PrepareToInstall; the folder stays open
    # until the pre-warm has run.
    prep = _function("PrepareToInstall")
    tail = prep[prep.index("refused: needs_space"):]
    assert re.search(r"if IsAdminInstallMode and not LockDownAppDir\(App\) then begin\s+"
                     r"Log\('refused: folder_not_ours'\);\s+Result := CustomMessage\('FolderNotOurs'\);\s+end;\s+end;$",
                     tail), tail
    step = _function("CurStepChanged")
    assert "LockDownAppDir" not in step and step.index("Prewarm;") < step.index("CloseAppDir;")

    header = text.split("\n#if ")[0]
    assert "Win32 security calls (LockFolder)" in header and "icacls" not in header
    # A folder IT makes under C:\ keeps C:\'s rule (users may rename it):
    # the README says where a /DIR is safe (fix round 3).
    readme = (WINDOWS / "README.md").read_text(encoding="utf-8")
    assert "directly under `C:\\`" in readme and "only admins can change" not in readme


def test_a_failed_build_leaves_no_installer_or_record():
    text = BUILD.read_text(encoding="ascii")
    catch = text[text.rindex("} catch {"):]
    catch = catch[:catch.index("\n}\n")]
    for name in ("$Record", "$Installer", "$InstallerRecord", "$FilesOut"):
        assert name in catch, name


def test_build_writes_the_file_list():
    """<dist> gets <name>.files.txt, a byte copy of the per-file list."""
    text = BUILD.read_text(encoding="ascii")
    assert ".files.txt'" in text
    assert text.index("$step = 'size gate'") < text.index("$step = 'file list'") < text.index("$step = 'release record'")


def test_every_inno_exit_code_is_documented():
    """Spec section 5: the exit codes (Inno's 1-8, our 9) with the Intune
    remap; and the uninstaller's codes for Intune's uninstall command."""
    codes = _l2_constant("EXIT_CODES")
    assert set(codes.values()) == set(range(0, 10)), codes
    text = (WINDOWS / "README.md").read_text(encoding="utf-8")
    assert "Uninstall exit codes" in text
    unins = text[text.index("Uninstall exit codes"):]
    for row in ("| `0` |", "| `1` |"):
        assert row in unins, row


def test_listings_write_is_a_no_op_on_a_clean_tree(tmp_path):
    """Both `--write` forms leave the committed goldens byte-identical, and
    a stale golden makes the check form exit 1."""
    import shutil
    import subprocess
    import sys
    copy = tmp_path / "repo"
    for rel in ("desktop/shared/bundle_manifest.txt", "desktop/mac_bundle_files.txt",
                "desktop/windows/image_files.txt"):
        (copy / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / rel, copy / rel)
    shared = REPO / "desktop" / "shared"
    for pattern in ("launcher/**/*.py", "pages/status.html", "pages/status.css",
                    "pages/status.js", "messages.json"):
        for path in shared.glob(pattern):
            if path.is_file() and "__pycache__" not in path.parts:
                target = copy / "desktop" / "shared" / path.relative_to(shared)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
    tools = [[sys.executable, str(WINDOWS / "build_env.py"), "listing"],
             [sys.executable, str(REPO / "desktop" / "bundle_listing.py")]]
    goldens = [copy / "desktop/windows/image_files.txt", copy / "desktop/mac_bundle_files.txt"]
    for tool, golden in zip(tools, goldens):
        before = golden.read_bytes()
        assert before == (REPO / golden.relative_to(copy)).read_bytes()
        run = lambda *extra: subprocess.run([*tool, *extra, "--repo", str(copy)],
                                            capture_output=True, text=True)
        assert run().returncode == 0
        assert run("--write").returncode == 0
        assert golden.read_bytes() == before
        golden.write_bytes(before + b"stale/entry.txt\n")
        stale = run()
        assert stale.returncode == 1 and "stale/entry.txt" in stale.stdout
        assert run("--write").returncode == 0
        assert golden.read_bytes() == before
