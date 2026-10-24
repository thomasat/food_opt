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
    assert be.compile_all(fake_image, workers=1) is True
    assert be.pyc_gaps(fake_image) == []


def test_compile_all_in_parallel_from_the_cli(fake_image):
    _touch(fake_image, "app/a.py", b"A = 1\n")
    _touch(fake_image, "shared/launcher/b.py", b"B = 2\n")
    r = subprocess.run([sys.executable, "-I", "-B", str(BUILD_ENV), "compile", str(fake_image)],
                       capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stdout + r.stderr
    assert be.pyc_gaps(fake_image) == []


def test_compile_all_reports_a_syntax_error(fake_image):
    _touch(fake_image, "app/broken.py", b"def (:\n")
    assert be.compile_all(fake_image, workers=1) is False


def test_pe_imports_reads_a_real_pe(tmp_path):
    pip_dir = pathlib.Path(importlib.util.find_spec("pip").origin).parent if importlib.util.find_spec("pip") else None
    exe = pip_dir / "_vendor" / "distlib" / "t64.exe" if pip_dir else None
    if exe is None or not exe.is_file():
        pytest.skip("pip's distlib t64.exe is not installed")
    assert "kernel32.dll" in be.pe_imports(exe)
    text = tmp_path / "x.txt"
    text.write_text("not a PE file\n", encoding="utf-8")
    assert be.pe_imports(text) == set()
    truncated = tmp_path / "cut.dll"
    truncated.write_bytes(exe.read_bytes()[:300])
    assert be.pe_imports(truncated) == set()


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

