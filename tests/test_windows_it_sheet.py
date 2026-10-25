"""The IT sheet (desktop/windows/for_it_teams.md, phase 6): every concrete
fact in it (paths, switches, exit codes, the detection key, the disk
figure, file names, the launch command, the privacy settings) is read here
from the code that implements it, so a change there fails a test naming the
stale fact; and every section that makes a claim ends up in the evidence
table, whose every row names a real test."""
import ast
import builtins
import json
import pathlib
import pickle
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
WINDOWS = REPO / "desktop" / "windows"
SHEET = WINDOWS / "for_it_teams.md"
WIN_README = WINDOWS / "README.md"
ISS = WINDOWS / "setup.iss"
CORE = WINDOWS / "FoodOptimizer" / "FoodOptimizer.Core"
SHELL = WINDOWS / "FoodOptimizer" / "FoodOptimizer.Shell"

TITLE = "# Food Optimizer for Windows: information for IT teams"
SECTIONS = [
    "Release values", "Summary", "Publisher and contact", "System requirements", "Install and uninstall",
    "Intune", "Exit codes", "Signing and hashes", "What runs", "Network", "Data at rest",
    "Files, registry and services", "Allow-listing", "Software bill of materials and licences",
    "Vulnerabilities and updates", "How this sheet is tested",
]
CONTACT = "Support and security contact: Sohum Patnaik, sohum.patnaik@fsi.org."
PENDING_SET = {
    "publisher validation by Microsoft", "legal address", "website",
    "Windows 10 22H2 and Windows 11 x64 results", "all-users upgrade test",
    "certificate subject and CA chain", "identity EKU OID", "Arbitrary Code Guard check",
    "allow-listing rules tested on a signed build", "locked-down profile results",
    "vulnerability check before each release (owner)",
}
L2 = "desktop/windows/tests"
REQUIRED_EVIDENCE = {
    f"{L2}/test_l2_10_install.py::test_silent_per_user_install_needs_no_admin",
    f"{L2}/test_l2_10_install.py::test_installed_files_are_the_image",
    f"{L2}/test_l2_10_install.py::test_shortcuts_point_at_the_app",
    f"{L2}/test_l2_10_install.py::test_projects_folder_is_made_and_pinned",
    f"{L2}/test_l2_10_install.py::test_prewarm_ran_at_install",
    f"{L2}/test_l2_45_uninstall.py::test_silent_uninstall_keeps_projects",
    f"{L2}/test_l2_50_scopes.py::test_all_users_install_as_admin",
    f"{L2}/test_l2_50_scopes.py::test_intune_install_and_uninstall_as_system",
    f"{L2}/test_l2_50_scopes.py::test_system_without_allusers_is_refused",
    f"{L2}/test_l2_50_scopes.py::test_per_user_install_refuses_when_it_installed_it",
    f"{L2}/test_l2_50_scopes.py::test_all_users_install_replaces_a_per_user_copy",
    f"{L2}/test_l2_50_scopes.py::test_silent_install_while_the_app_runs",
    f"{L2}/test_l2_50_scopes.py::test_a_long_install_folder_is_refused",
    f"{L2}/test_l2_60_upgrade.py::test_upgrade_after_a_lock_change_leaves_nothing_stale",
    f"{L2}/test_l2_60_upgrade.py::test_a_failed_prewarm_says_so_and_exits_9",
    f"{L2}/test_l2_70_disk.py::test_too_little_free_space_is_refused",
    f"{L2}/test_l2_30_shell.py::test_shell_smoke_no_console_health_close_no_python",
    f"{L2}/test_l2_40_traffic.py::test_the_capture_sees_our_python",
    f"{L2}/test_l2_40_traffic.py::test_our_processes_connect_only_to_loopback",
    "tests/desktop_launcher/test_launcher_streamlit.py::test_real_streamlit_answers_only_on_its_base_path_and_host",
    "tests/desktop_launcher/test_launcher_guard.py::test_host_check_refuses_any_other_host",
    "tests/desktop_launcher/test_launcher_guard.py::test_guard_turns_proxies_off",
    "tests/desktop_launcher/test_launcher_guard.py::test_guard_stubs_streamlit_ip_lookups",
    "tests/test_windows_ready.py::test_ui_fetches_nothing_from_the_internet",
    "tests/test_windows_installer.py::test_exit_code_table_matches_l2_support",
    "tests/test_windows_build_env.py::test_file_manifest_is_sorted_lf_and_stable",
    "desktop/windows/FoodOptimizer/FoodOptimizer.UiTests/Installer/InstalledAppTests.cs::Try_again_on_the_installed_app_leaves_one_server",
    "tests/test_windows_it_sheet.py::test_webview2_privacy_settings_are_pinned",
    "tests/test_windows_it_sheet.py::test_a_saved_project_is_plain_json_and_never_unpickled",
    "tests/test_windows_it_sheet.py::test_nothing_writes_registry_services_or_tasks",
    "tests/test_windows_sbom.py::test_every_unsigned_file_is_attributed_once",
    "tests/test_windows_sbom.py::test_gpl_mentions_are_found_in_bundled_notices",
}
NO_EVIDENCE_NEEDED = {"Release values", "Publisher and contact", "Allow-listing", "Vulnerabilities and updates",
                      "How this sheet is tested"}


def _release_docs():
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location("foodopt_release_docs", REPO / "desktop" / "release_docs.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("foodopt_release_docs", module)
    spec.loader.exec_module(module)
    return module


rd = _release_docs()


def _text():
    return SHEET.read_text(encoding="utf-8")


def _flat(text):
    return " ".join(text.split())


def _sections(text):
    """[(heading, body)] for every '## ' heading, in order."""
    parts = re.split(r"^## (.+)$", text, flags=re.M)
    return [(parts[i].strip(), parts[i + 1]) for i in range(1, len(parts), 2)]


def _section(name):
    return dict(_sections(_text()))[name]


def _l2_constant(name):
    tree = ast.parse((WINDOWS / "tests" / "l2_support.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"no literal {name} in l2_support.py")


def _iss_directive(name):
    return re.search(r"^%s=(.*)$" % name, ISS.read_text(encoding="ascii"), re.M).group(1)


def _code_blocks(text):
    """Every fenced block's text, list-item indentation taken off."""
    blocks = []
    for indent, body in re.findall(r"^( *)```\n(.*?)^\1```$", text, re.M | re.S):
        blocks.append("\n".join(line[len(indent):] for line in body.strip("\n").splitlines()))
    return blocks


ROW = re.compile(r"^\| `?(-?\d+)`? \| (.+?) \| (\w+) \|$")


def _code_tables(text):
    """Each markdown table whose rows are `| code | meaning | type |`."""
    tables, current = [], []
    for line in text.splitlines() + [""]:
        m = ROW.match(line)
        if m:
            current.append(m.groups())
        elif not line.startswith("|"):
            if current:
                tables.append(current)
            current = []
    return tables


# ---------- shape ----------

def test_sections_are_the_spec_list():
    text = _text()
    assert text.splitlines()[0] == TITLE
    assert [h for h, _ in _sections(text)] == SECTIONS
    assert len(re.findall(r"^# ", text, re.M)) == 1
    first = text.split("\n\n")[1]
    assert "This file is the source." in first and "FoodOptimizer-<version>-IT-sheet.pdf" in first


def test_contact_and_no_commitment():
    text = _text()
    assert CONTACT in _flat(text)
    first = text.index("Food System Innovations")
    assert re.match(r"\s*PENDING\(publisher validation by Microsoft\)", text[first + len("Food System Innovations"):])
    commitment = re.compile(r"\b(within|response time|respond|SLA|business days?|hours|monthly|quarterly|"
                            r"cadence|guarantee)\b", re.I)
    for name in ("Publisher and contact", "Vulnerabilities and updates"):
        assert not commitment.search(_section(name)), (name, commitment.search(_section(name)))
    assert "https://github.com/thomasat/food_opt" in _section("Publisher and contact")
    assert (REPO / "LICENSE").read_text(encoding="utf-8").startswith("MIT License")
    assert "MIT" in _section("Publisher and contact")


def test_pending_markers_are_the_known_set():
    assert {m.group(1) for m in rd.PENDING.finditer(_text())} == PENDING_SET


def test_every_field_is_a_known_value():
    fields = set(rd.FIELD.findall(_text()))
    assert fields and fields <= rd.VALUE_KEYS, fields - rd.VALUE_KEYS


def test_sheet_is_swept():
    from tests.test_food_bo import _USER_FACING_TEXT
    assert "desktop/windows/for_it_teams.md" in _USER_FACING_TEXT


# ---------- facts read from the code ----------

def test_paths_match_the_code():
    text = _flat(_text())
    folder = _iss_directive("DefaultDirName")
    assert folder.startswith("{autopf}\\")
    name = folder[len("{autopf}\\"):]
    assert f"%LOCALAPPDATA%\\Programs\\{name}" in text
    assert f"C:\\Program Files\\{name}" in text
    user = re.search(r'const string Folder = "([^"]+)"', (CORE / "InstallLayout.cs").read_text(encoding="utf-8")).group(1)
    assert f"%LOCALAPPDATA%\\{user}" in text and f"%USERPROFILE%\\{user}" in text
    data = _flat(_section("Data at rest"))
    state = (CORE / "StateFiles.cs").read_text(encoding="utf-8")
    decl = re.search(r"public const string (.*?);", state, re.S).group(1)
    names = re.findall(r'=\s*"([^"]+)"', decl)
    assert len(names) >= 8, names
    for n in names + ["launcher.log.1", "shell.log.1"]:
        assert n in data, n
    tree = ast.parse((REPO / "desktop" / "shared" / "launcher" / "files.py").read_text(encoding="utf-8"))
    consts = {t.id: ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
              for t in node.targets if isinstance(t, ast.Name) and isinstance(node.value, ast.Constant)}
    assert consts["LOG_FILE"] in data and consts["LOCK_DIR"] in data
    assert f"newest {consts['KEEP_BACKUPS']} launch copies" in data
    rotate = re.search(r"Rotate\(string path, long maxBytes = ([\d_]+)\)",
                       (CORE / "ShellLog.cs").read_text(encoding="utf-8")).group(1)
    assert f"{int(rotate.replace('_', '')):,} bytes" in data


def test_switches_match_the_installer():
    text = _text()
    silent = " ".join(_l2_constant("SILENT"))
    assert silent in text and silent + " /ALLUSERS" in text
    readme = WIN_README.read_text(encoding="utf-8")
    switches = readme[readme.index("## Install switches and exit codes"):]
    commands = _code_blocks(switches[:switches.index("\n## ", 5)])
    assert len(commands) == 3, commands
    sheet_blocks = _code_blocks(text)
    for command in commands:
        assert command in sheet_blocks, command
    app_id = _l2_constant("APP_ID")
    assert _iss_directive("AppId") == "{" + app_id
    key = f"HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\{app_id}_is1"
    assert key in sheet_blocks, key
    intune = _section("Intune")
    assert key in intune and "DisplayVersion" in intune and "{{version}}" in intune
    assert "/LOG=" in text and "/NODESKTOPSHORTCUT" in text


def test_exit_code_tables_match_the_readme():
    sheet = _code_tables(_section("Exit codes"))
    readme = _code_tables(WIN_README.read_text(encoding="utf-8"))
    assert len(readme) == 2 and sheet == readme


def test_requirements_match_the_installer():
    req = _flat(_section("System requirements"))
    build = _iss_directive("MinVersion").split(".")[-1]
    assert build == "19045" and f"build {build}" in req
    assert "{{disk_free_gb}} GB free" in req
    doc = json.loads((REPO / "desktop" / "shared" / "messages.json").read_text(encoding="utf-8"))
    by_id = {e["id"]: e["text"]["windows"] for e in doc["messages"]}
    for key in ("webview2.missing.title", "webview2.missing.body"):
        assert by_id[key] in req, key
    assert "x64compatible" == _iss_directive("ArchitecturesAllowed") and "x64 emulation" in req


def test_what_runs_matches_the_launch():
    runs = _flat(_section("What runs"))
    assert "python.exe -I -X utf8 -B -m launcher" in runs
    layout = (CORE / "InstallLayout.cs").read_text(encoding="utf-8")
    assert '"-I", "-X", "utf8", "-B", "-m", "launcher",' in layout
    assert "127.0.0.1" in runs
    assert '"--server.address=127.0.0.1",' in (REPO / "desktop" / "shared" / "launcher" / "childenv.py").read_text(encoding="utf-8")
    assert "{{image_files}}" in runs and "{{image_gb}}" in runs


def test_webview2_privacy_settings_are_pinned():
    """The Network section's claims about the WebView2 runtime: crash
    reports stay on the PC, SmartScreen's reputation check (a lookup at
    Microsoft) is off, and every request the page makes passes our filter."""
    shell = "\n".join(p.read_text(encoding="utf-8") for p in sorted(SHELL.glob("*.cs")))
    for needle in ("IsCustomCrashReportingEnabled = true", "IsReputationCheckingRequired = false",
                   'AddWebResourceRequestedFilter("*"'):
        assert needle in shell, needle
    net = _flat(_section("Network"))
    assert "custom crash reporting" in net


def test_a_saved_project_is_plain_json_and_never_unpickled(tmp_path, monkeypatch):
    """Data at rest: a project file is JSON, and a pickle placed where a
    project goes is reported damaged without being run."""
    import storage
    monkeypatch.chdir(tmp_path)
    store = storage.LocalStorage()
    store.save("Round one", {"ingredients": ["oat"], "n": 1})
    assert json.loads((tmp_path / "Round one.pkl").read_bytes()) == {"ingredients": ["oat"], "n": 1}

    class Payload:
        def __reduce__(self):
            return (exec, ("import builtins; builtins._foodopt_unpickled = True",))
    (tmp_path / "evil.pkl").write_bytes(pickle.dumps(Payload()))
    with pytest.raises(storage.StorageError):
        storage.LocalStorage().load("evil")
    assert not hasattr(builtins, "_foodopt_unpickled")


def test_nothing_writes_registry_services_or_tasks():
    """Files, registry and services: the shell and its core never touch the
    registry; setup.iss has no [Registry] section, no service or scheduled
    task, and its only registry change is deleting a per-user copy's
    uninstall key (when an all-users install replaces that copy)."""
    for folder in (SHELL, CORE):
        for path in folder.rglob("*.cs"):
            text = path.read_text(encoding="utf-8")
            for needle in ("Microsoft.Win32.Registry", "RegistryKey", "Registry."):
                assert needle not in text, (path.name, needle)
    iss = ISS.read_text(encoding="ascii")
    assert "[Registry]" not in iss
    for needle in ("sc.exe", "schtasks", "CreateService", "RegWrite"):
        assert needle.lower() not in iss.lower(), needle
    deletes = re.findall(r"RegDelete\w*\(([^)]*)\)", iss)
    assert deletes and all("UninstallKey" in d for d in deletes), deletes
    assert "uninstall key" in _flat(_section("Files, registry and services"))


# ---------- the evidence table ----------

def _evidence_rows():
    body = _section("How this sheet is tested")
    rows = []
    for line in body.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")] if line.startswith("|") else []
        if len(cells) == 3 and cells[2].startswith("`"):
            rows.append((cells[0], cells[1], cells[2].strip("`")))
    return rows


def _resolves(ref):
    path, _, name = ref.partition("::")
    file = REPO / path
    if not file.is_file() or not name:
        return False
    text = file.read_text(encoding="utf-8")
    if path.endswith(".py"):
        return any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
                   for n in ast.walk(ast.parse(text)))
    if path.endswith(".cs"):
        return bool(re.search(r"\b(?:void|Task)\s+%s\s*\(" % re.escape(name), text))
    return False


def test_every_evidence_row_names_a_real_test():
    rows = _evidence_rows()
    assert rows
    for section, statement, ref in rows:
        assert section in SECTIONS, section
        assert statement, ref
        assert _resolves(ref), ref
    assert REQUIRED_EVIDENCE <= {ref for _, _, ref in rows}, REQUIRED_EVIDENCE - {ref for _, _, ref in rows}


def test_every_claim_section_has_evidence():
    covered = {section for section, _, _ in _evidence_rows()}
    assert set(SECTIONS) - NO_EVIDENCE_NEEDED <= covered, set(SECTIONS) - NO_EVIDENCE_NEEDED - covered
