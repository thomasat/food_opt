"""desktop/windows/build_env.py, the Python half of the Windows build: the
image layout, the strip rules, unchecked-hash precompiling, the long-path
gate, the PE import reader and the per-file manifest (phase 3b, spec §4).
These run on any OS against temporary folders; the VM tests run the same
functions against the built image."""
import hashlib
import importlib.util
import json
import pathlib
import py_compile
import re
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
BUILD_ENV = REPO / "desktop" / "windows" / "build_env.py"


def _load():
    spec = importlib.util.spec_from_file_location("build_env", BUILD_ENV)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


be = _load()


@pytest.fixture
def fake_image(tmp_path):
    image = tmp_path / be.IMAGE_NAME
    (image / be.SITE).mkdir(parents=True)
    (image / "python" / "python.exe").write_bytes(b"")
    return image


def _touch(root, rel, data=b""):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_constants_are_the_plans_numbers():
    assert be.IMAGE_NAME == "FoodOptimizer"
    assert (be.ROOT_BUDGET, be.PATH_LIMIT, be.MAX_RELATIVE) == (80, 240, 160)
    assert be.SITE == "python/Lib/site-packages"
    assert be.PTH_NAME == "foodopt.pth"
    assert be.PTH_TEXT == "../../../app\n../../../shared\n"
    # 80 + len("python\Lib\site-packages\") + longest package path <= 240
    assert be.ROOT_BUDGET + len(be.SITE) + 1 + (be.MAX_RELATIVE - len(be.SITE) - 1) == be.PATH_LIMIT
    assert [p for p, _ in be.STRIP_RULES] == [
        "python/include/", "python/libs/", "python/Scripts/",
        "python/Lib/site-packages/torch/include/",
        "python/Lib/site-packages/torch/bin/protoc*.exe",
        "python/Lib/site-packages/pip/", "python/Lib/site-packages/pip-*.dist-info/",
        "python/Lib/site-packages/setuptools/*.exe", "*.lib"]


def test_assemble_lays_out_the_image(fake_image):
    be.assemble(REPO, fake_image)
    golden = (REPO / "desktop" / "windows" / "image_files.txt").read_text(encoding="utf-8").splitlines()
    assert be.image_listing(fake_image) == be.expected_listing(REPO) == golden
    shared = REPO / "desktop" / "shared" / "launcher" / "sitecustomize.py"
    assert (fake_image / be.SITE / "sitecustomize.py").read_bytes() == shared.read_bytes()
    assert (fake_image / be.SITE / be.PTH_NAME).read_bytes() == be.PTH_TEXT.encode("utf-8")
    assert (fake_image / "Start Here.txt").read_bytes() == (
        REPO / "desktop" / "windows" / "start_here.txt").read_bytes()
    for entry in be.app_files(REPO):
        assert (fake_image / "app" / entry).read_bytes() == (REPO / entry).read_bytes()


def test_assemble_needs_site_packages(tmp_path):
    with pytest.raises(FileNotFoundError):
        be.assemble(REPO, tmp_path / be.IMAGE_NAME)


def test_listing_is_folders_with_slashes_sorted_without_pycache(fake_image):
    be.assemble(REPO, fake_image)
    _touch(fake_image, "app/__pycache__/x.cpython-313.pyc")
    listing = be.image_listing(fake_image)
    assert listing == sorted(listing)
    assert "python/" in listing and "app/" in listing and "app/data/" in listing
    assert not [line for line in listing if "__pycache__" in line]
    assert not [line for line in listing if line.startswith("python/") and line != "python/"]


def test_pth_puts_app_and_shared_on_the_path(fake_image):
    be.assemble(REPO, fake_image)
    site = fake_image / be.SITE
    code = "import json,site,sys; site.addsitedir(sys.argv[1]); print(json.dumps(sys.path))"
    out = subprocess.run([sys.executable, "-I", "-c", code, str(site)], capture_output=True,
                         text=True, encoding="utf-8", check=True).stdout
    paths = {pathlib.Path(p).resolve() for p in json.loads(out)}
    assert (fake_image / "app").resolve() in paths
    assert (fake_image / "shared").resolve() in paths


PLANTED = [
    "python/include/Python.h", "python/libs/python313.lib", "python/Scripts/streamlit.exe",
    f"{be.SITE}/torch/include/a.h", f"{be.SITE}/torch/bin/protoc.exe", f"{be.SITE}/torch/lib/dnnl.LIB",
    f"{be.SITE}/pip/__init__.py", f"{be.SITE}/pip-25.2.dist-info/RECORD", f"{be.SITE}/setuptools/cli-64.exe",
]
KEEPERS = [f"{be.SITE}/torch/lib/torch_cpu.dll", f"{be.SITE}/setuptools/__init__.py",
           "python/python.exe", "python/DLLs/_ssl.pyd"]


def test_strip_removes_every_rule_and_keeps_the_rest(fake_image):
    for rel in PLANTED + KEEPERS:
        _touch(fake_image, rel)
    counts = be.strip(fake_image)
    assert be.strip_violations(fake_image) == []
    for rel in KEEPERS:
        assert (fake_image / rel).is_file(), rel
    for rel in PLANTED:
        assert not (fake_image / rel).exists(), rel
    assert sum(counts.values()) == 9
    assert set(counts) == {p for p, _ in be.STRIP_RULES}


def test_strip_drops_record_lines_for_what_it_removed(fake_image):
    """A RECORD line for a stripped file (uv's console-script launcher, whose
    hash holds the build folder's path) goes; every other byte stays."""
    record = _touch(fake_image, f"{be.SITE}/streamlit-1.55.0.dist-info/RECORD", (
        b"../../Scripts/streamlit.exe,sha256=oFY9JYjcRq8XG9U4vXOt9fusc6,47616\r\n"
        b"streamlit/__init__.py,sha256=abc,10\r\n"
        b"\"torch/include/a,b.h\",sha256=def,3\n"
        b"torch/lib/dnnl.LIB,sha256=ghi,4\n"
        b"streamlit-1.55.0.dist-info/RECORD,,\n"
        b"../../Scripts/notes.txt,,"))
    kept_wheel = _touch(fake_image, f"{be.SITE}/idna-3.18.dist-info/RECORD",
                        b"idna/__init__.py,sha256=x,1\n")
    be.strip(fake_image)
    assert record.read_bytes() == (b"streamlit/__init__.py,sha256=abc,10\r\n"
                                   b"streamlit-1.55.0.dist-info/RECORD,,\n")
    assert kept_wheel.read_bytes() == b"idna/__init__.py,sha256=x,1\n"
    assert be.strip_violations(fake_image) == []


def test_strip_violations_name_what_is_left(fake_image):
    _touch(fake_image, f"{be.SITE}/x/y.lib")
    assert be.strip_violations(fake_image) == ["python/Lib/site-packages/x/y.lib"]


def test_long_path_gate_boundary(fake_image):
    _touch(fake_image, "a" * 160)
    assert be.long_path_violations(fake_image) == []
    assert be.longest_relative(fake_image) == (160, "a" * 160)
    _touch(fake_image, "b" * 161)
    found = be.long_path_violations(fake_image)
    assert found == ["161 " + "b" * 161]
    (fake_image / ("c" * 161)).mkdir()
    assert len(be.long_path_violations(fake_image)) == 2


def test_long_path_counts_backslash_separators(fake_image):
    rel = "d" * 79 + "/" + "e" * 80     # 160 characters with its separator
    _touch(fake_image, rel)
    assert be.longest_relative(fake_image) == (160, rel)
    assert be.long_path_violations(fake_image) == []


def test_repo_files_fit_the_budget():
    for entry in be.expected_listing(REPO):
        assert len(entry.rstrip("/")) <= be.MAX_RELATIVE, entry
        if entry.endswith(".py"):
            folder, _, name = entry.rpartition("/")
            pyc = f"{folder}/__pycache__/{name[:-3]}.cpython-313.pyc"
            assert len(pyc) <= be.MAX_RELATIVE, pyc


def _compile(path, mode):
    py_compile.compile(str(path), cfile=importlib.util.cache_from_source(str(path)),
                       doraise=True, invalidation_mode=mode)


def test_pyc_gaps_accept_only_unchecked_hash(fake_image):
    mode = py_compile.PycInvalidationMode
    good = _touch(fake_image, "app/good.py", b"x = 1\n")
    stamp = _touch(fake_image, "app/stamp.py", b"x = 2\n")
    edited = _touch(fake_image, "app/edited.py", b"x = 3\n")
    _touch(fake_image, "app/missing.py", b"x = 4\n")
    _compile(good, mode.UNCHECKED_HASH)
    _compile(stamp, mode.TIMESTAMP)
    _compile(edited, mode.UNCHECKED_HASH)
    edited.write_bytes(b"x = 33\n")
    assert be.pyc_gaps(fake_image) == ["app/edited.py", "app/missing.py", "app/stamp.py"]


def test_compile_all_then_no_gaps(fake_image):
    for rel in ("python/Lib/os2.py", f"{be.SITE}/pkg/__init__.py", "app/app.py",
                "shared/launcher/cli.py"):
        _touch(fake_image, rel, b"VALUE = 'x'\n")
    # A stale timestamp pyc is recompiled, not kept.
    _compile(fake_image / "app/app.py", py_compile.PycInvalidationMode.TIMESTAMP)
    assert be.compile_all(fake_image) is True
    assert be.pyc_gaps(fake_image) == []


def test_compile_from_the_cli(fake_image):
    _touch(fake_image, "app/a.py", b"A = 1\n")
    _touch(fake_image, "shared/launcher/b.py", b"B = 2\n")
    r = subprocess.run([sys.executable, "-I", "-B", str(BUILD_ENV), "compile", str(fake_image)],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stdout + r.stderr
    assert be.pyc_gaps(fake_image) == []


def test_compile_all_reports_a_syntax_error(fake_image):
    _touch(fake_image, "app/broken.py", b"def (:\n")
    assert be.compile_all(fake_image) is False


def test_pe_imports_reads_a_real_pe(tmp_path):
    pip_dir = pathlib.Path(importlib.util.find_spec("pip").origin).parent if importlib.util.find_spec("pip") else None
    exe = pip_dir / "_vendor" / "distlib" / "t64.exe" if pip_dir else None
    if exe is None or not exe.is_file():
        pytest.skip("pip's distlib t64.exe is not installed")
    assert "kernel32.dll" in be.pe_imports(exe)
    truncated = tmp_path / "cut.dll"
    truncated.write_bytes(exe.read_bytes()[:300])
    with pytest.raises(ValueError, match="cut.dll"):
        be.pe_imports(truncated)


@pytest.mark.parametrize("data", [b"", b"not a PE file\n", b"MZ", b"MZ" + b"\0" * 62,
                                  b"MZ" + b"\0" * 58 + b"\x40\0\0\0" + b"PE\0\0"],
                         ids=["empty", "text", "mz-only", "no-pe-signature", "cut-after-signature"])
def test_pe_imports_fails_on_a_file_it_cannot_parse(tmp_path, data):
    """3b review minor: an unparseable binary must not read as "imports
    nothing", or the VC runtime gate would pass it."""
    bad = tmp_path / "bad.pyd"
    bad.write_bytes(data)
    with pytest.raises(ValueError, match="bad.pyd"):
        be.pe_imports(bad)


def test_vc_runtime_gaps_name_a_binary_it_cannot_read(fake_image):
    _touch(fake_image, "python/DLLs/_bad.pyd", b"MZ" + b"\0" * 62)
    _touch(fake_image, "python/DLLs/_text.pyd", b"not a PE file\n")
    assert be.vc_runtime_gaps(fake_image) == [
        "python/DLLs/_bad.pyd: unreadable PE file (no PE signature)",
        "python/DLLs/_text.pyd: unreadable PE file (no MZ header)",
        "python/python.exe: unreadable PE file (no MZ header)"]   # the fixture's empty python.exe


def test_vc_runtime_gaps_rules(fake_image):
    site = be.SITE
    c10 = _touch(fake_image, f"{site}/torch/lib/c10.dll")
    x = _touch(fake_image, f"{site}/sklearn/_x.pyd")
    y = _touch(fake_image, f"{site}/numpy/_y.pyd")
    _touch(fake_image, f"{site}/sklearn/.libs/vcomp140.dll")
    imports = {c10: {"msvcp140.dll", "kernel32.dll"}, x: {"vcomp140.dll"}, y: {"msvcp140-abc.dll"}}

    def imports_of(path):
        return imports.get(pathlib.Path(path), set())

    assert be.vc_runtime_gaps(fake_image, imports_of=imports_of) == [
        f"{site}/torch/lib/c10.dll: msvcp140.dll"]
    _touch(fake_image, "python/msvcp140.dll")
    assert be.vc_runtime_gaps(fake_image, imports_of=imports_of) == []


def test_vc_runtime_gaps_accept_a_dot_libs_folder(fake_image):
    pyd = _touch(fake_image, f"{be.SITE}/scipy/_z.pyd")
    _touch(fake_image, f"{be.SITE}/scipy.libs/msvcp140_1.dll")
    assert be.vc_runtime_gaps(
        fake_image, imports_of=lambda p: {"msvcp140_1.dll"} if pathlib.Path(p) == pyd else set()) == []
    other = _touch(fake_image, "python/DLLs/_w.pyd")
    assert be.vc_runtime_gaps(
        fake_image, imports_of=lambda p: {"msvcp140_1.dll"} if pathlib.Path(p) == other else set()
    ) == ["python/DLLs/_w.pyd: msvcp140_1.dll"]
    assert pyd.exists()


def test_file_manifest_is_sorted_lf_and_stable(fake_image):
    _touch(fake_image, "app/b.py", b"b")
    _touch(fake_image, "app/a b.txt", b"spaced")
    _touch(fake_image, "Start Here.txt", b"hi")
    text, files, size = be.file_manifest(fake_image)
    assert text == be.file_manifest(fake_image)[0]
    lines = text.split("\n")
    assert lines[-1] == "" and "\r" not in text
    lines = lines[:-1]
    for line in lines:
        assert re.match(r"^[0-9a-f]{64}  \S", line), line
    paths = [line[66:] for line in lines]
    assert paths == sorted(paths)
    assert paths.count("app/a b.txt") == 1
    assert files == len(lines) == 4
    assert size == len(b"b") + len(b"spaced") + len(b"hi")
    assert f"{hashlib.sha256(b'spaced').hexdigest()}  app/a b.txt" in lines


def test_cli_files_writes_the_manifest_and_prints_its_hash(fake_image, tmp_path):
    out = tmp_path / "FoodOptimizer.files.txt"
    r = subprocess.run([sys.executable, "-I", "-B", str(BUILD_ENV), "files", str(fake_image), str(out)],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stdout + r.stderr
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    assert r.stdout.splitlines() == ["image_files=1", "image_bytes=0", f"image_sha256={sha}"]


def test_cli_gate_names_each_problem(fake_image):
    be.assemble(REPO, fake_image)
    _touch(fake_image, f"{be.SITE}/x/y.lib")
    r = subprocess.run([sys.executable, "-I", "-B", str(BUILD_ENV), "gate", str(REPO), str(fake_image)],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 1
    lines = r.stdout.splitlines()
    assert "strip: python/Lib/site-packages/x/y.lib" in lines
    assert [line for line in lines if line.startswith("pyc: ")]
    assert [line for line in lines if line.startswith("longest_path_chars=")]


def test_user_facing_text_files_skip_build_output(tmp_path):
    from tests.test_food_bo import _user_facing_text_files
    for rel in ("desktop/windows/build/x.txt", "desktop/windows/dist/y.txt",
                "desktop/windows/.cache/z.txt", "desktop/windows/image_files.txt",
                "desktop/windows/README.md"):
        _touch(tmp_path, rel, b"x")
    found = _user_facing_text_files(tmp_path)
    assert "desktop/windows/README.md" in found
    for rel in ("desktop/windows/build/x.txt", "desktop/windows/dist/y.txt",
                "desktop/windows/.cache/z.txt", "desktop/windows/image_files.txt"):
        assert rel not in found


def test_gate_checks_the_vc_runtime(fake_image, monkeypatch):
    be.assemble(REPO, fake_image)
    monkeypatch.setattr(be, "vc_runtime_gaps", lambda image: ["python/DLLs/_x.pyd: msvcp140.dll"])
    assert "vc runtime: python/DLLs/_x.pyd: msvcp140.dll" in be._gate(REPO, fake_image)


# ---------- the shell joins the image (phase 5) ----------

SHELL_FILES = {"FoodOptimizer.exe": b"MZ shell", "FoodOptimizer.dll": b"shell dll",
               "Microsoft.Web.WebView2.Core.dll": b"webview2", "x/y.json": b"{}"}


def fake_publish(tmp):
    publish = pathlib.Path(tmp) / "publish"
    for rel, data in SHELL_FILES.items():
        _touch(publish, rel, data)
    return publish


def test_shell_constants():
    assert be.RESERVED_TOP == ("python", "app", "shared", "Start Here.txt")
    assert be.SHELL_EXE == "FoodOptimizer.exe"


def test_add_shell_puts_the_publish_folder_at_the_root(fake_image, tmp_path):
    be.assemble(REPO, fake_image)
    publish = fake_publish(tmp_path)
    assert be.shell_listing(publish) == sorted(SHELL_FILES)
    assert be.add_shell(publish, fake_image) == sorted(SHELL_FILES)
    for rel, data in SHELL_FILES.items():
        assert (fake_image / rel).read_bytes() == data
    assert be.shell_gaps(publish, fake_image) == []


def test_add_shell_refuses_a_collision(fake_image, tmp_path):
    publish = fake_publish(tmp_path)
    _touch(publish, "App/x.dll", b"x")
    with pytest.raises(ValueError, match="shell file collides with the image: App/x.dll"):
        be.add_shell(publish, fake_image)
    assert not (fake_image / "App" / "x.dll").exists()


def test_add_shell_needs_the_exe(fake_image, tmp_path):
    publish = fake_publish(tmp_path)
    (publish / "FoodOptimizer.exe").unlink()
    with pytest.raises(ValueError, match="publish folder has no FoodOptimizer.exe"):
        be.add_shell(publish, fake_image)


def test_gate_listing_ignores_only_the_shell(fake_image, tmp_path):
    be.assemble(REPO, fake_image)
    publish = fake_publish(tmp_path)
    be.add_shell(publish, fake_image)
    golden = (REPO / "desktop" / "windows" / "image_files.txt").read_text(encoding="utf-8").splitlines()
    assert be.image_listing(fake_image, exclude=be.shell_listing(publish)) == golden
    assert be.image_listing(fake_image) != golden          # 3b's callers still see the shell
    _touch(fake_image, "stray.txt", b"x")
    assert be.image_listing(fake_image, exclude=be.shell_listing(publish)) != golden


def test_shell_gaps_find_a_changed_file(fake_image, tmp_path):
    publish = fake_publish(tmp_path)
    be.add_shell(publish, fake_image)
    (fake_image / "FoodOptimizer.dll").write_bytes(b"edited")
    assert be.shell_gaps(publish, fake_image) == ["differs FoodOptimizer.dll"]
    (fake_image / "x" / "y.json").unlink()
    assert be.shell_gaps(publish, fake_image) == ["differs FoodOptimizer.dll", "missing x/y.json"]


def test_cli_shell_and_gate_with_the_shell(fake_image, tmp_path):
    be.assemble(REPO, fake_image)
    publish = fake_publish(tmp_path)
    r = subprocess.run([sys.executable, "-I", "-B", str(BUILD_ENV), "shell", str(publish), str(fake_image)],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.splitlines() == [f"shell_files={len(SHELL_FILES)}"]
    (fake_image / "FoodOptimizer.dll").write_bytes(b"edited")
    r = subprocess.run([sys.executable, "-I", "-B", str(BUILD_ENV), "gate", str(REPO), str(fake_image),
                        "--shell", str(publish)], capture_output=True, text=True, encoding="utf-8")
    lines = r.stdout.splitlines()
    assert r.returncode == 1
    assert "shell: differs FoodOptimizer.dll" in lines
    assert not [line for line in lines if line.startswith("listing: ")]


def test_compiled_bytes_do_not_depend_on_the_image_folder(tmp_path):
    images = [tmp_path / "a" / be.IMAGE_NAME, tmp_path / "José build" / "x y" / be.IMAGE_NAME]
    for image in images:
        for rel in ("python/Lib/os2.py", "app/app.py", "shared/launcher/cli.py"):
            _touch(image, rel, b"def f():\n    raise ValueError\n")
        assert be.compile_all(image)
    assert be.file_manifest(images[0])[0] == be.file_manifest(images[1])[0]


def test_compiled_bytes_are_the_same_from_run_to_run(tmp_path):
    """marshal writes a string as a back-reference when something else in the
    process holds it, so a .pyc's bytes follow what that process compiled
    before it. A worker pool shares files out differently on each run; on
    scipy and sklearn's ~1600 files a few .pyc then differ on most runs
    (Review Focus 4: the same image hash from every build). Three runs, so a
    pool that happens to share alike twice still shows."""
    roots = [importlib.util.find_spec(name) for name in ("scipy", "sklearn")]
    if None in roots:
        pytest.skip("needs scipy and sklearn sources")
    manifests = []
    for n in range(3):
        image = tmp_path / str(n) / be.IMAGE_NAME
        for spec in roots:
            folder = pathlib.Path(spec.origin).parent
            for src in sorted(folder.rglob("*.py")):
                _touch(image, f"{be.SITE}/{folder.name}/{src.relative_to(folder).as_posix()}",
                       src.read_bytes())
        r = subprocess.run([sys.executable, "-I", "-B", str(BUILD_ENV), "compile", str(image)],
                           capture_output=True, text=True, encoding="utf-8")
        assert r.returncode == 0, r.stdout + r.stderr
        manifests.append(be.file_manifest(image)[0].splitlines())
    for other in manifests[1:]:
        assert [a for a, b in zip(manifests[0], other) if a != b] == []


_PE = b"MZ" + b"\0" * 58 + b"\x40\0\0\0" + b"PE\0\0"   # e_lfanew = 0x40 -> "PE\0\0"


def test_pe_files_go_by_header_not_extension(fake_image):
    _touch(fake_image, "python/a.dll", _PE)
    _touch(fake_image, "python/Lib/site-packages/x/b.node", _PE)
    _touch(fake_image, "python/c.dll", b"not a binary")
    _touch(fake_image, "python/d.pyd", b"MZ")
    assert be.pe_files(fake_image) == ["python/Lib/site-packages/x/b.node", "python/a.dll"]


def test_pe_header_beyond_the_first_4_kib_is_found(fake_image):
    head = b"MZ" + b"\0" * 58 + (8000).to_bytes(4, "little")
    _touch(fake_image, "python/far.dll", head + b"\0" * (8000 - len(head)) + b"PE\0\0")
    _touch(fake_image, "python/past_end.dll", head + b"\0" * 100)
    assert be.pe_files(fake_image) == ["python/far.dll"]


def test_cli_pe_files_writes_the_list(fake_image, tmp_path):
    _touch(fake_image, "python/a.dll", _PE)
    out = tmp_path / "pe.txt"
    r = subprocess.run([sys.executable, "-I", "-B", str(BUILD_ENV), "pe-files", str(fake_image), str(out)],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "pe_files=1" in r.stdout.splitlines()
    assert out.read_bytes() == b"python/a.dll\n"


# ---------- the installer's [Files] sources (gate at 6361cfe) ----------

ISS = REPO / "desktop" / "windows" / "setup.iss"


def test_iss_sources_are_every_files_source_of_setup_iss():
    """build_installer.ps1 gives each of these the commit's time before ISCC
    (Inno stores each source file's time): the image, and every file of the
    repo a [Files] entry names, so a checkout's own times never reach the
    installer's bytes. The all-users marker once kept its checkout time."""
    assert be.iss_file_sources(ISS) == ["{#ImageDir}\\*", "all-users"]
    assert be.iss_source_problems(ISS) == []


def test_iss_sources_cli_prints_each_source(capsys):
    assert be.main(["iss-sources", str(ISS)]) == 0
    assert capsys.readouterr().out.splitlines() == ["iss_source={#ImageDir}\\*", "iss_source=all-users"]


@pytest.mark.parametrize("entry, problem", [
    ('Source: "extra.txt"; DestDir: "{app}"', "iss source: missing extra.txt"),
    ('Source: "art\\*.bmp"; DestDir: "{app}"', "iss source: a pattern, name each file: art\\*.bmp"),
    ('Source: "{#Other}\\x"; DestDir: "{app}"', "iss source: not a file beside setup.iss: {#Other}\\x"),
    ('Source: "C:\\x\\y.txt"; DestDir: "{app}"', "iss source: not a file beside setup.iss: C:\\x\\y.txt"),
    ('Source: "..\\VERSION"; DestDir: "{app}"', "iss source: not a file beside setup.iss: ..\\VERSION"),
    ('DestDir: "{app}"', "iss source: an entry without a Source: DestDir: \"{app}\""),
])
def test_a_source_the_build_cannot_time_fails(tmp_path, capsys, entry, problem):
    """A [Files] source the build would not give the commit's time to fails
    the build (exit 1) rather than make the installer depend on the checkout."""
    iss = tmp_path / "setup.iss"
    iss.write_text("[Setup]\nAppName=x\n\n[Files]\n; a comment\n"
                   'Source: "{#ImageDir}\\*"; DestDir: "{app}"; Flags: recursesubdirs\n'
                   + entry + "\n\n[Code]\nfunction F: Boolean; begin end;\n", encoding="ascii")
    (tmp_path / "art").mkdir()
    assert be.iss_source_problems(iss) == [problem]
    assert be.main(["iss-sources", str(iss)]) == 1
    assert problem in capsys.readouterr().out.splitlines()
