"""The Windows SBOM (phase 6): desktop/windows/sbom.py reads the licence of
every package the image ships from its own metadata, counts every GPL-family
mention in the shipped licence texts, attributes each re-signed binary to
the component that owns it, and writes a CycloneDX 1.6 document. It runs in
the build VM under the image's Python; here it runs on fake images built in
tmp_path, and (when FOODOPT_WIN_WHEELS names a folder) on the real Windows
wheels, read as zip data only."""
import ast
import importlib.util
import json
import os
import pathlib
import re
import site
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
WINDOWS = REPO / "desktop" / "windows"
SBOM_PY = WINDOWS / "sbom.py"
DECLARED = WINDOWS / "sbom_declared.json"
WINDOWS_LOCK = REPO / "desktop" / "requirements-windows.lock.txt"
SITE = "python/Lib/site-packages"


def _load():
    spec = importlib.util.spec_from_file_location("foodopt_sbom", SBOM_PY)
    module = importlib.util.module_from_spec(spec)
    sys.modules["foodopt_sbom"] = module
    spec.loader.exec_module(module)
    return module


sbom = _load()


# ---------- fixtures ----------

def fake_site(tmp, dists):
    """A site-packages folder: {name: (version, metadata_lines, {relpath:
    text})}. Each dist gets NAME-VERSION.dist-info/METADATA and a RECORD
    listing its files (and its own METADATA and RECORD)."""
    tmp = pathlib.Path(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    for name, (version, lines, files) in dists.items():
        info = f"{name.replace('-', '_')}-{version}.dist-info"
        (tmp / info).mkdir(parents=True, exist_ok=True)
        meta = [f"Metadata-Version: 2.4", f"Name: {name}", f"Version: {version}", *lines]
        (tmp / info / "METADATA").write_text("\n".join(meta) + "\n\n", encoding="utf-8")
        record = []
        for rel, text in files.items():
            path = tmp / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(text, bytes):
                path.write_bytes(text)
            else:
                path.write_text(text, encoding="utf-8")
            record.append(f"{rel},sha256=x,1")
        record += [f"{info}/METADATA,,", f"{info}/RECORD,,"]
        (tmp / info / "RECORD").write_text("\n".join(record) + "\n", encoding="utf-8")
    return tmp


def fake_lock(path, entries):
    """A uv-style lock: {name: version}, one fake hash each."""
    lines = ["# fake lock"]
    for name, version in entries.items():
        lines += [f"{name}=={version} \\", f"    --hash=sha256:{'ab' * 32}", "    # via -r x"]
    pathlib.Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return pathlib.Path(path)


MIT = ["Classifier: License :: OSI Approved :: MIT License"]


# ---------- survey (written first, Task 2 Step 1) ----------

def test_survey_prints_unknown_instead_of_failing(tmp_path, capsys):
    site_dir = fake_site(tmp_path / "site", {
        "good": ("1.0", MIT, {"good/__init__.py": ""}),
        "mystery": ("2.0", ["License: UNKNOWN"], {"mystery/__init__.py": ""}),
    })
    lock = fake_lock(tmp_path / "lock.txt", {"good": "1.0", "mystery": "2.0"})
    assert sbom.main(["survey", str(site_dir), str(lock)]) == 0
    rows = {line.split("\t")[0]: line.split("\t") for line in capsys.readouterr().out.splitlines()}
    assert rows["good"][1:4] == ["1.0", "MIT", "classifier"]
    assert rows["mystery"][1:3] == ["2.0", "UNKNOWN"]


# ---------- a fake image for build() ----------

RUNTIME = {"OpenSSL": "3.0.16", "SQLite": "3.50.4", "zlib": "1.3.1", "Expat": "2.7.1"}
TORCH = ("2.10.0", ["License: BSD-3-Clause"],
         {"torch/__init__.py": "", "torch/lib/libiomp5md.dll": b"MZ", "torch/lib/_C.pyd": b"MZ"})


def fake_image(tmp, dists=None, *, deps=None):
    """An image folder: site-packages holding torch plus `dists`, every
    file sbom_declared.json names as evidence (holding its phrase), the
    shell's FoodOptimizer.exe, and (when given) FoodOptimizer.deps.json.
    Returns (image, lock); the lock pins exactly the dists."""
    tmp = pathlib.Path(tmp)
    dists = {"torch": TORCH, **(dists or {})}
    image = tmp / "image"
    fake_site(image / SITE, dists)
    declared = json.loads(DECLARED.read_text(encoding="utf-8"))
    evidence = [c["evidence"] for c in declared["components"]]
    evidence += [o["evidence"] for o in declared.get("licence_overrides", {}).values()]
    for ev in evidence:
        paths = ([ev["file"]] if "file" in ev else []) + list(ev.get("files", []))
        for rel in paths:
            path = image / rel
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(ev.get("contains", "MZ"), encoding="utf-8")
    (image / "FoodOptimizer.exe").write_bytes(b"MZ")
    if deps is not None:
        (image / "FoodOptimizer.deps.json").write_text(json.dumps(deps), encoding="utf-8")
    lock = fake_lock(tmp / "lock.txt", {n: v for n, (v, _, _) in dists.items()})
    return image, lock


def run_build(image, lock, **kw):
    args = dict(version="0.9.0", commit="c0ffee", unsigned=(), signed=False, inno="6.5.4",
                runtime=RUNTIME)
    args.update(kw)
    return sbom.build(image, lock, DECLARED, **args)


def _props(component, name):
    return [p["value"] for p in component.get("properties", []) if p["name"] == name]


def _all_components(bom):
    return [bom["metadata"]["component"]] + bom["components"]


def _dist(tmp, name, version, lines, files=None):
    return sbom.read_dists(fake_site(tmp, {name: (version, lines, files or {f"{name}/__init__.py": ""})}))[name]


# ---------- the lock ----------

def test_lock_parses_every_package():
    entries = sbom.parse_lock(WINDOWS_LOCK)
    assert len(entries) == 62
    assert entries["torch"].version == "2.10.0"
    assert all(len(e.hashes) >= 1 for e in entries.values())


# ---------- licences (Review Focus 2) ----------

@pytest.mark.parametrize("lines, ids, name, source", [
    (["License-Expression: MIT AND (Apache-2.0 OR BSD-3-Clause)"],
     ("MIT", "Apache-2.0", "BSD-3-Clause"), None, "License-Expression"),
    (MIT, ("MIT",), None, "classifier"),
    (["Classifier: License :: OSI Approved :: BSD License"], (), "BSD (classifier)", "classifier"),
    (["License: Apache-2.0"], ("Apache-2.0",), None, "License"),
    (["License: BSD-3-Clause", "Classifier: License :: OSI Approved :: BSD License"],
     ("BSD-3-Clause",), None, "License"),
    (MIT + ["License: x"], ("Apache-2.0",), None, "override"),
], ids=["expression", "classifier", "vague-classifier", "license-field",
        "spdx-field-beats-vague-classifier", "override"])
def test_licence_sources_in_order(tmp_path, lines, ids, name, source):
    dist = _dist(tmp_path, "pkg", "1.0", lines)
    overrides = {"pkg": {"licence": "Apache-2.0", "evidence": {"file": "x", "contains": "y"}}} \
        if source == "override" else {}
    lic = sbom.licence_of(dist, overrides)
    assert (lic.ids, lic.name, lic.source) == (ids, name, source)
    if name == "BSD (classifier)":
        assert sbom.family(lic) == ["BSD"]


def test_unknown_licence_fails_the_build(tmp_path):
    image, lock = fake_image(tmp_path, {"mystery": ("2.0", ["License: UNKNOWN"], {"mystery/__init__.py": ""})})
    with pytest.raises(ValueError, match="no licence for mystery 2.0"):
        run_build(image, lock)


def test_gpl_mentions_are_found_in_bundled_notices(tmp_path):
    numpy = _dist(tmp_path / "a", "numpy", "2.4.3", ["License-Expression: BSD-3-Clause"], {
        "numpy/__init__.py": "",
        "numpy.libs/LICENSE_gfortran.txt": "Copyright FSF\nunder the GNU General Public License\nend\n"})
    assert sbom.gpl_mentions(numpy) == [
        ("numpy.libs/LICENSE_gfortran.txt", 2, "under the GNU General Public License")]
    python_style = _dist(tmp_path / "b", "typing-extensions", "4.16.0", ["License-Expression: PSF-2.0"], {
        "typing_extensions.py": "",
        "typing_extensions-4.16.0.dist-info/licenses/LICENSE": "releases have also been GPL-compatible;\n"})
    assert len(sbom.gpl_mentions(python_style)) == 1
    image, lock = fake_image(tmp_path / "c", {"x": ("1", [
        "Classifier: License :: OSI Approved :: GNU General Public License v3 (GPLv3)"], {"x/__init__.py": ""})})
    summary = sbom.summary(run_build(image, lock))
    assert summary["gpl_licensed"] == ["pypi:x@1"]
    image, lock = fake_image(tmp_path / "d", {"numpy": ("2.4.3", ["License-Expression: BSD-3-Clause"], {
        "numpy/__init__.py": "", "numpy.libs/LICENSE_gfortran.txt": "GNU General Public License\n"})})
    summary = sbom.summary(run_build(image, lock))
    assert summary["gpl_licensed"] == []
    assert {"name": "numpy", "version": "2.4.3", "file": "numpy.libs/LICENSE_gfortran.txt",
            "line": 1} in summary["gpl_mentions"]


WEBVIEW2 = "Microsoft.Web.WebView2/1.0.4314.35"
NETCORE = "runtimepack.Microsoft.NETCore.App.Runtime.win-x64/10.0.0"


def _deps():
    return {
        "targets": {".NETCoreApp,Version=v10.0/win-x64": {
            WEBVIEW2: {"runtime": {"lib/net462/Microsoft.Web.WebView2.Core.dll": {}},
                       "native": {"runtimes/win-x64/native/WebView2Loader.dll": {}}},
            NETCORE: {"runtime": {"System.Private.CoreLib.dll": {}}, "native": {"coreclr.dll": {}}},
            "FoodOptimizer/1.0.0": {"runtime": {"FoodOptimizer.dll": {}}},
        }},
        "libraries": {
            WEBVIEW2: {"type": "package", "path": "microsoft.web.webview2/1.0.4314.35"},
            NETCORE: {"type": "runtimepack"},
            "FoodOptimizer/1.0.0": {"type": "project"},
        },
    }


def _nuspec(root, pkg_id, version, licence):
    folder = pathlib.Path(root) / pkg_id.lower() / version
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{pkg_id.lower()}.nuspec").write_text(
        '<?xml version="1.0"?>\n<package xmlns="http://schemas.microsoft.com/packaging/2013/05/nuspec.xsd">'
        f"<metadata><id>{pkg_id}</id><version>{version}</version>{licence}</metadata></package>\n",
        encoding="utf-8")


def _nuget_root(tmp):
    root = pathlib.Path(tmp) / "nuget"
    _nuspec(root, "Microsoft.Web.WebView2", "1.0.4314.35", '<license type="file">LICENSE.txt</license>')
    _nuspec(root, "Microsoft.NETCore.App.Runtime.win-x64", "10.0.0", '<license type="expression">MIT</license>')
    return root


def test_nuget_licences_from_nuspec(tmp_path):
    deps = tmp_path / "FoodOptimizer.deps.json"
    deps.write_text(json.dumps(_deps()), encoding="utf-8")
    found = {c["id"]: c for c in sbom.nuget_components(deps, [tmp_path / "empty", _nuget_root(tmp_path)])}
    assert set(found) == {"Microsoft.Web.WebView2", "Microsoft.NETCore.App.Runtime.win-x64"}
    assert found["Microsoft.Web.WebView2"]["licence"].ids == ("LicenseRef-Microsoft.Web.WebView2",)
    assert found["Microsoft.Web.WebView2"]["licence"].source == "nuspec"
    assert found["Microsoft.NETCore.App.Runtime.win-x64"]["licence"].ids == ("MIT",)
    assert found["Microsoft.NETCore.App.Runtime.win-x64"]["type"] == "runtimepack"
    assert set(found["Microsoft.Web.WebView2"]["assets"]) == {"Microsoft.Web.WebView2.Core.dll", "WebView2Loader.dll"}
    with pytest.raises(ValueError, match="no nuspec for Microsoft.Web.WebView2 1.0.4314.35"):
        sbom.nuget_components(deps, [tmp_path / "empty"])


def test_declared_evidence_is_checked(tmp_path):
    image, lock = fake_image(tmp_path)
    run_build(image, lock)
    (image / "python" / "LICENSE.txt").unlink()
    with pytest.raises(ValueError, match="^sbom: evidence missing: python/LICENSE.txt$"):
        run_build(image, lock)


def test_declared_evidence_phrase_is_checked(tmp_path):
    image, lock = fake_image(tmp_path)
    (image / "python" / "LICENSE.txt").write_text("some other licence\n", encoding="utf-8")
    with pytest.raises(ValueError, match="^sbom: evidence phrase not in python/LICENSE.txt"):
        run_build(image, lock)


def test_every_unsigned_file_is_attributed_once(tmp_path):
    image, lock = fake_image(tmp_path, deps=_deps())
    unsigned = [f"{SITE}/torch/lib/_C.pyd", "Microsoft.Web.WebView2.Core.dll", "FoodOptimizer.exe"]
    (image / "Microsoft.Web.WebView2.Core.dll").write_bytes(b"MZ")
    bom = run_build(image, lock, unsigned=unsigned, nuget_roots=[_nuget_root(tmp_path)])
    owners = {c["bom-ref"]: int(_props(c, "foodopt:resigned_files")[0]) for c in _all_components(bom)}
    assert sum(owners.values()) == 3
    assert {ref for ref, n in owners.items() if n} == {
        "pypi:torch@2.10.0", "nuget:Microsoft.Web.WebView2@1.0.4314.35", "app:food-optimizer@0.9.0"}
    attributed = sbom.attribute(unsigned, sbom.read_dists(image / SITE),
                                sbom.nuget_components(image / "FoodOptimizer.deps.json", [_nuget_root(tmp_path)]),
                                ["FoodOptimizer.exe", "Microsoft.Web.WebView2.Core.dll"],
                                app_ref="app:food-optimizer@0.9.0")
    paths = [p for ps in attributed.values() for p in ps]
    assert sorted(paths) == sorted(unsigned)
    with pytest.raises(ValueError, match="unattributed: python/Lib/site-packages/zz/x.pyd"):
        run_build(image, lock, unsigned=unsigned + [f"{SITE}/zz/x.pyd"], nuget_roots=[_nuget_root(tmp_path)])


def test_signing_property_follows_the_build(tmp_path):
    image, lock = fake_image(tmp_path, {"other": ("1.0", MIT, {"other/__init__.py": ""})})
    unsigned = [f"{SITE}/torch/lib/_C.pyd", "FoodOptimizer.exe"]
    for signed, word in ((False, "unsigned (internal build)"), (True, "re-signed by Food System Innovations")):
        bom = run_build(image, lock, unsigned=unsigned, signed=signed)
        by_ref = {c["bom-ref"]: _props(c, "foodopt:signing") for c in _all_components(bom)}
        assert by_ref["pypi:torch@2.10.0"] == [word]
        assert by_ref["app:food-optimizer@0.9.0"] == [word]
        assert by_ref["pypi:other@1.0"] == ["vendor"]


def test_sbom_is_cyclonedx_1_6_shaped(tmp_path):
    image, lock = fake_image(tmp_path, {"other": ("1.0", ["License-Expression: MIT OR Apache-2.0"],
                                                  {"other/__init__.py": ""})})
    bom = run_build(image, lock)
    assert {"bomFormat", "specVersion", "version", "serialNumber", "metadata", "components"} <= set(bom)
    assert (bom["bomFormat"], bom["specVersion"], bom["version"]) == ("CycloneDX", "1.6", 1)
    assert re.fullmatch(r"urn:uuid:[0-9a-f-]{36}", bom["serialNumber"])
    assert "timestamp" not in bom["metadata"]
    app = bom["metadata"]["component"]
    assert app["supplier"]["name"] == "Food System Innovations"
    assert (app["type"], app["name"], app["version"]) == ("application", "Food Optimizer", "0.9.0")
    assert app["licenses"] == [{"license": {"id": "MIT"}}]
    refs = [c["bom-ref"] for c in bom["components"]]
    assert len(refs) == len(set(refs)) and refs == sorted(refs)
    for c in _all_components(bom):
        assert {"type", "name", "version", "bom-ref", "licenses"} <= set(c), c
        assert c["type"] in ("library", "framework", "platform", "file", "application"), c
        lic = c["licenses"]
        assert lic and (lic == [{"expression": lic[0].get("expression")}] and lic[0]["expression"]
                        or all(set(e) == {"license"} and len(e["license"]) == 1
                               and ("id" in e["license"] or "name" in e["license"]) for e in lic)), c
        for e in lic:
            assert not e.get("license", {}).get("id", "").startswith("LicenseRef-"), c
        assert all(isinstance(p["value"], str) for p in c.get("properties", [])), c
    other = next(c for c in bom["components"] if c["bom-ref"] == "pypi:other@1.0")
    assert other["licenses"] == [{"expression": "MIT OR Apache-2.0"}]
    assert other["purl"] == "pkg:pypi/other@1.0"
    assert _props(other, "foodopt:lock_hashes") == ["ab" * 32]
    declared = {c["bom-ref"] for c in bom["components"] if _props(c, "foodopt:licence_source") == ["declared"]}
    assert {"generic:cpython@3.13.7", "generic:inno-setup@6.5.4", "generic:openssl@3.0.16"} <= declared


def test_sbom_is_reproducible(tmp_path):
    image, lock = fake_image(tmp_path)
    first, second = sbom.dumps(run_build(image, lock)), sbom.dumps(run_build(image, lock))
    assert first == second and first.endswith(b"\n")
    assert sbom.dumps(run_build(image, lock, commit="other")) != first


def test_build_refuses_a_lock_mismatch(tmp_path):
    image, lock = fake_image(tmp_path, {"other": ("1.0", MIT, {"other/__init__.py": ""})})
    fake_lock(lock, {"torch": "2.10.0", "other": "1.1"})
    with pytest.raises(ValueError, match=r"^sbom: other 1\.0 .*version"):
        run_build(image, lock)
    fake_lock(lock, {"torch": "2.10.0", "other": "1.0", "absent": "3.0"})
    with pytest.raises(ValueError, match="^sbom: no metadata for absent 3.0"):
        run_build(image, lock)
    fake_lock(lock, {"torch": "2.10.0"})
    with pytest.raises(ValueError, match="^sbom: other 1.0 is not in the lock"):
        run_build(image, lock)


def test_build_cli_prints_the_record_lines(tmp_path, capsys):
    image, lock = fake_image(tmp_path)
    unsigned = tmp_path / "unsigned.txt"
    unsigned.write_text("FoodOptimizer.exe\n", encoding="utf-8")
    out = tmp_path / "out.cdx.json"
    argv = ["build", str(image), str(lock), str(out), "--version", "0.9.0", "--commit", "c0ffee",
            "--unsigned", str(unsigned), "--signed", "no", "--inno", "6.5.4",
            "--nuget-root", str(tmp_path / "none")]
    assert sbom.main(argv) == 0
    values = dict(line.split("=", 1) for line in capsys.readouterr().out.splitlines())
    import hashlib
    assert values["sbom_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert int(values["sbom_components"]) == len(json.loads(out.read_bytes())["components"])
    assert {"sbom_gpl_licensed", "sbom_gpl_mentions"} <= set(values)
    (image / "python" / "LICENSE.txt").unlink()
    out.unlink()
    assert sbom.main(argv) == 1
    # On stdout (the build reads it) and on stderr: build_installer.ps1's
    # Invoke-Native keeps stdout, so the reason reaches the console there.
    captured = capsys.readouterr()
    assert captured.out.startswith("sbom: evidence missing")
    assert captured.err.startswith("sbom: evidence missing")
    assert not out.exists()


def test_sbom_py_is_stdlib_only():
    tree = ast.parse(SBOM_PY.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] in sys.stdlib_module_names, alias.name
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative import"
            assert node.module.split(".")[0] in sys.stdlib_module_names, node.module


def test_declared_components_name_their_evidence():
    data = json.loads(DECLARED.read_text(encoding="utf-8"))
    for c in data["components"]:
        assert {"ref", "name", "type", "version_from", "licence", "evidence"} <= set(c), c
        assert re.fullmatch(r"(pins:[\w.]+|arg:inno|runtime:\w+|dist:[\w.-]+)", c["version_from"]), c
        assert ("file" in c["evidence"]) != ("files" in c["evidence"]), c
    for name, o in data["licence_overrides"].items():
        assert {"licence", "evidence"} <= set(o) and "contains" in o["evidence"], name


def _windows_only():
    tree = ast.parse((REPO / "tests" / "test_locks.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "WINDOWS_ONLY" for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("no WINDOWS_ONLY in tests/test_locks.py")


def test_every_locked_package_has_a_known_licence_on_this_machine():
    import importlib.metadata
    mac = sbom.parse_lock(REPO / "desktop" / "requirements.lock.txt")
    try:
        for entry in mac.values():
            importlib.metadata.distribution(entry.name)
    except importlib.metadata.PackageNotFoundError as e:
        pytest.skip(f"not the locked environment: {e}")
    here = sbom.read_dists(pathlib.Path(site.getsitepackages()[0]))
    overrides = sbom._overrides(DECLARED)
    wanted = set(sbom.parse_lock(WINDOWS_LOCK)) - {sbom.normalize(n) for n in _windows_only()}
    assert wanted <= set(here), sorted(wanted - set(here))
    for key in wanted:
        sbom.licence_of(here[key], overrides)


def test_every_windows_wheel_has_a_known_licence():
    folder = os.environ.get("FOODOPT_WIN_WHEELS")
    if not folder:
        pytest.skip("set FOODOPT_WIN_WHEELS to the downloaded Windows wheels")
    dists = sbom.read_dists(pathlib.Path(folder).expanduser())
    lock = sbom.parse_lock(WINDOWS_LOCK)
    assert {k: lock[k].version for k in lock} == {k: dists[k].version for k in lock if k in dists}
    overrides = sbom._overrides(DECLARED)
    for key in lock:
        sbom.licence_of(dists[key], overrides)
