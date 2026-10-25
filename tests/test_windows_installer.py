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
SETUP_MUTEX = ("FoodOptimizer.Setup.0E00CC2F-4E51-4EBF-A229-A910D5742A39,"
               "Global\\FoodOptimizer.Setup.0E00CC2F-4E51-4EBF-A229-A910D5742A39")
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
            for name in ("AppUserModelId", "SingleInstanceMutex", "InstallerMutex")}


def test_setup_directives_are_pinned():
    found = setup_directives()
    assert {k: found.get(k) for k in SETUP_PINS} == SETUP_PINS


def test_mutexes_and_app_id_match_the_shell():
    ident = _identity()
    assert setup_directives()["AppMutex"] == f"{ident['SingleInstanceMutex']},{ident['InstallerMutex']}"
    ids = [e["AppUserModelID"] for e in section_entries("Icons") if "AppUserModelID" in e]
    assert ids and set(ids) == {ident["AppUserModelId"]}


def test_languages_layer_our_messages():
    assert section_entries("Languages") == [
        {"Name": "en", "MessagesFile": "compiler:Default.isl,installer_messages.isl"}]


def test_one_files_entry_copies_the_image():
    assert section_entries("Files") == [
        {"Source": "{#ImageDir}\\*", "DestDir": "{app}",
         "Flags": "ignoreversion recursesubdirs createallsubdirs"}]


def test_install_delete_clears_the_bundled_folders():
    assert section_entries("InstallDelete") == [
        {"Type": "filesandordirs", "Name": "{app}\\python"},
        {"Type": "filesandordirs", "Name": "{app}\\app"},
        {"Type": "filesandordirs", "Name": "{app}\\shared"},
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


REFUSALS = ("system_needs_allusers", "installed_by_it", "folder_too_long", "needs_space")


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
