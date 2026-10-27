"""The Windows README's facts for IT teams (desktop/windows/README.md,
"For IT teams (reference)"; phase 6, revised 2026-10-25). It is a repo
reference, never sent or rendered: the release page carries five lines
for IT (tests/test_release_texts.py). Every concrete fact here (paths,
switches, the detection key, file names, the launch command, the privacy
settings, the signers) is read from the code that implements it, so a
change there fails a test naming the stale fact."""
import ast
import builtins
import json
import pathlib
import pickle
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
WINDOWS = REPO / "desktop" / "windows"
WIN_README = WINDOWS / "README.md"
ISS = WINDOWS / "setup.iss"
CORE = WINDOWS / "FoodOptimizer" / "FoodOptimizer.Core"
SHELL = WINDOWS / "FoodOptimizer" / "FoodOptimizer.Shell"
HEADING = "## For IT teams (reference)"
PARTS = ["Requirements", "What runs", "Network", "Data at rest", "Files, registry and services",
         "Allow-listing"]


def _readme():
    return WIN_README.read_text(encoding="utf-8")


def _flat(text):
    return " ".join(text.split())


def _reference():
    text = _readme()
    start = text.index(HEADING)
    end = text.find("\n## ", start + 1)
    return text[start:] if end < 0 else text[start:end]


def _section(name):
    parts = re.split(r"^### (.+)$", _reference(), flags=re.M)
    return dict((parts[i].strip(), parts[i + 1]) for i in range(1, len(parts), 2))[name]


def _l2_constant(name):
    tree = ast.parse((WINDOWS / "tests" / "l2_support.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"no literal {name} in l2_support.py")


def _iss_directive(name):
    return re.search(r"^%s=(.*)$" % name, ISS.read_text(encoding="ascii"), re.M).group(1)


def test_reference_has_its_parts():
    assert re.findall(r"^### (.+)$", _reference(), re.M) == PARTS


# ---------- facts read from the code ----------

def test_paths_match_the_code():
    text = _flat(_readme())
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
    text = _readme()
    silent = " ".join(_l2_constant("SILENT"))
    assert silent in text and silent + " /ALLUSERS" in text
    app_id = _l2_constant("APP_ID")
    assert _iss_directive("AppId") == "{" + app_id
    assert f"HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\{app_id}_is1" in text
    assert "DisplayVersion" in text and "/LOG=" in text and "/NODESKTOPSHORTCUT" in text


def test_requirements_match_the_installer():
    req = _flat(_section("Requirements"))
    build = _iss_directive("MinVersion").split(".")[-1]
    assert build == "19045" and f"build {build}" in req
    doc = json.loads((REPO / "desktop" / "shared" / "messages.json").read_text(encoding="utf-8"))
    by_id = {e["id"]: e["text"].get("windows", e["text"]["mac"]) for e in doc["messages"]}
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


def test_allow_listing_names_every_signer_in_the_install():
    """The binaries keep their vendors' signatures (CPython's PSF, Microsoft's
    C++ and .NET runtimes); only the unsigned ones are re-signed. A rule for
    Food System Innovations alone would block python.exe."""
    pins = json.loads((WINDOWS / "pins.json").read_text(encoding="utf-8"))
    section = _flat(_section("Allow-listing"))
    for signer in ("Food System Innovations", pins["python"]["signer"], pins["vc_runtime"]["dll_signer"],
                   "Microsoft Corporation"):
        assert signer in section, signer
    assert "every binary it installs" not in section


def test_allow_listing_is_scoped_to_the_product():
    section = _flat(_section("Allow-listing"))
    for needle in ("Program Files\\Food Optimizer", "publisher", "product", "FoodOptimizer-<ver>-Windows.files.txt",
                   "AppLocker", "default rules", "per-user"):
        assert needle in section, needle
    assert "all four signers" not in section


def test_the_test_overrides_are_documented():
    text = _flat(_readme())
    for name in ("FOODOPT_STATE_DIR", "FOODOPT_PROJECTS_DIR", "FOODOPT_TEST_HOOKS"):
        assert name in text, name
