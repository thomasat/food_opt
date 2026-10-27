"""The Python half of the Windows build (build_installer.ps1 runs it with the
bundled python.exe -I -B): lay out the image, strip build-only files,
precompile every .py with unchecked-hash .pyc, and gate the result. The same
rules strip the folder and later check it, so they can't drift.

Stdlib only. Paths in results are image-relative with "/".

    python build_env.py assemble <repo> <image>
    python build_env.py strip <image>
    python build_env.py compile <image>
    python build_env.py shell <publish> <image>
    python build_env.py gate <repo> <image> [--shell <publish>]
    python build_env.py files <image> <out.txt>

Each command returns 0, or 1 with each problem on its own stdout line.
"""
import compileall
import csv
import fnmatch
from collections.abc import Collection
import hashlib
import importlib.util
import os
import pathlib
import posixpath
import py_compile
import re
import shutil
import struct
import subprocess
import sys

IMAGE_NAME = "FoodOptimizer"
ROOT_BUDGET = 80          # the install root, including its trailing "\"
PATH_LIMIT = 240
MAX_RELATIVE = PATH_LIMIT - ROOT_BUDGET
SITE = "python/Lib/site-packages"
PTH_NAME = "foodopt.pth"
PTH_TEXT = "../../../app\n../../../shared\n"
SHARED_FILES = ("launcher/**/*.py", "pages/status.html", "pages/status.css",
                "pages/status.js", "messages.json")
COMPILE_ROOTS = ("python/Lib", "app", "shared")
# A regex on the "/" path of files never precompiled. Only deliberately
# invalid files may be listed, one comment line each.
COMPILE_EXCLUDE: str | None = None

STRIP_RULES: tuple[tuple[str, str], ...] = (
    ("python/include/", "C headers, build only"),
    ("python/libs/", "import libraries, build only"),
    ("python/Scripts/", "console-script .exe stubs; the app runs python -m"),
    ("python/Lib/site-packages/torch/include/", "C++ headers, build only"),
    ("python/Lib/site-packages/torch/bin/protoc*.exe", "protobuf compiler, build only, unsigned"),
    ("python/Lib/site-packages/pip/", "installer, not used at run time"),
    ("python/Lib/site-packages/pip-*.dist-info/", "installer metadata"),
    ("python/Lib/site-packages/setuptools/*.exe", "launcher stubs for building scripts, unsigned"),
    ("*.lib", "static and import libraries, build only"),
)

VC_RUNTIME_NAME = re.compile(
    r"^(msvcp140(_\w+)?|vcruntime140(_\d)?|concrt140|vccorlib140|vcomp140)\.dll$")
PE_SUFFIXES = (".exe", ".dll", ".pyd")
# The image's own top-level entries: the shell's published files sit beside
# them at the root and may never share a name with one (case-insensitive).
RESERVED_TOP = ("python", "app", "shared", "Start Here.txt")
SHELL_EXE = "FoodOptimizer.exe"


# ---------- what goes in ----------

def app_files(repo) -> list[str]:
    """The lines of desktop/shared/bundle_manifest.txt (repo-relative)."""
    text = (pathlib.Path(repo) / "desktop" / "shared" / "bundle_manifest.txt").read_text(encoding="utf-8")
    return [line for line in text.split("\n") if line.strip()]


def shared_files(repo) -> list[str]:
    """The shipped files of desktop/shared, relative to it, sorted."""
    shared = pathlib.Path(repo) / "desktop" / "shared"
    found = set()
    for pattern in SHARED_FILES:
        for path in shared.glob(pattern):
            if path.is_file() and "__pycache__" not in path.parts:
                found.add(path.relative_to(shared).as_posix())
    return sorted(found)


def _with_folders(files) -> list[str]:
    entries = set()
    for rel in files:
        entries.add(rel)
        parts = rel.split("/")
        for i in range(1, len(parts)):
            entries.add("/".join(parts[:i]) + "/")
    return sorted(entries)


def expected_listing(repo) -> list[str]:
    files = ["Start Here.txt"]
    files += ["app/" + entry for entry in app_files(repo)]
    files += ["shared/" + rel for rel in shared_files(repo)]
    return sorted(set(_with_folders(files)) | {"python/"})


def _walk(root):
    """(rel "/" path, is_dir) for every entry under root, __pycache__ included."""
    root = pathlib.Path(root)
    for dirpath, dirnames, filenames in os.walk(root):
        here = pathlib.Path(dirpath)
        for name in dirnames:
            yield (here / name).relative_to(root).as_posix(), True
        for name in filenames:
            yield (here / name).relative_to(root).as_posix(), False


def image_listing(image, exclude: Collection[str] = ()) -> list[str]:
    """The golden-listing view of an image. Files in `exclude` (the shell's
    published files) are left out, and so is any folder left holding only
    excluded files."""
    image = pathlib.Path(image)
    excluded = set(exclude)
    entries = set()
    for rel, is_dir in _walk(image):
        if rel == "python":
            entries.add("python/")
            continue
        if rel.startswith("python/") or "__pycache__" in rel.split("/"):
            continue
        if not is_dir and rel in excluded:
            continue
        entries.add(rel + "/" if is_dir else rel)
    if excluded:
        held = {e.rsplit("/", 1)[0] + "/" for e in excluded if "/" in e}
        for folder in sorted((e for e in entries if e.endswith("/") and e != "python/"),
                             key=len, reverse=True):
            holds_excluded = any(e.startswith(folder) for e in held)
            if holds_excluded and not any(e != folder and e.startswith(folder) for e in entries):
                entries.discard(folder)
    return sorted(entries)


def _copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def assemble(repo, image) -> None:
    repo, image = pathlib.Path(repo), pathlib.Path(image)
    site = image / SITE
    if not site.is_dir():
        raise FileNotFoundError(f"{site} does not exist (install the lock first)")
    for entry in app_files(repo):
        _copy(repo / entry, image / "app" / entry)
    shared = repo / "desktop" / "shared"
    for rel in shared_files(repo):
        _copy(shared / rel, image / "shared" / rel)
    _copy(repo / "desktop" / "windows" / "start_here.txt", image / "Start Here.txt")
    _copy(shared / "launcher" / "sitecustomize.py", site / "sitecustomize.py")
    (site / PTH_NAME).write_bytes(PTH_TEXT.encode("utf-8"))


# ---------- the shell (phase 4's published FoodOptimizer.exe folder) ----------

def shell_listing(publish) -> list[str]:
    """Every file under the publish folder, "/" relative, codepoint-sorted."""
    return sorted(rel for rel, is_dir in _walk(publish) if not is_dir)


def add_shell(publish, image) -> list[str]:
    """Copy the published shell to the image root, path for path."""
    publish, image = pathlib.Path(publish), pathlib.Path(image)
    files = shell_listing(publish)
    reserved = {name.lower() for name in RESERVED_TOP}
    for rel in files:
        if rel.split("/")[0].lower() in reserved:
            raise ValueError(f"shell file collides with the image: {rel}")
    if SHELL_EXE not in files:
        raise ValueError(f"publish folder has no {SHELL_EXE}")
    for rel in files:
        _copy(publish / rel, image / rel)
    return files


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shell_gaps(publish, image) -> list[str]:
    """"missing <p>" / "differs <p>" for each published file the image lacks
    or holds with other bytes."""
    publish, image = pathlib.Path(publish), pathlib.Path(image)
    gaps = []
    for rel in shell_listing(publish):
        if not (image / rel).is_file():
            gaps.append(f"missing {rel}")
        elif _sha256(image / rel) != _sha256(publish / rel):
            gaps.append(f"differs {rel}")
    return gaps


# ---------- strip ----------

def _matches(pattern, rel, is_dir):
    if pattern.endswith("/"):
        return is_dir and fnmatch.fnmatchcase(rel.lower(), pattern[:-1].lower())
    return not is_dir and fnmatch.fnmatchcase(rel.lower(), pattern.lower())


def _rule_hits(image, pattern):
    """Entries matching one rule, outermost only (a matched folder counts once)."""
    image = pathlib.Path(image)
    hits = []
    for dirpath, dirnames, filenames in os.walk(image):
        here = pathlib.Path(dirpath)
        keep = []
        for name in dirnames:
            rel = (here / name).relative_to(image).as_posix()
            if _matches(pattern, rel, True):
                hits.append(rel)
            else:
                keep.append(name)
        dirnames[:] = keep
        for name in filenames:
            rel = (here / name).relative_to(image).as_posix()
            if _matches(pattern, rel, False):
                hits.append(rel)
    return sorted(hits)


def _stripped(rel):
    """Whether a strip rule removes this image-relative file path (itself, or
    a folder it is in)."""
    parts = rel.split("/")
    for pattern, _reason in STRIP_RULES:
        if _matches(pattern, rel, False):
            return True
        if any(_matches(pattern, "/".join(parts[:i]), True) for i in range(1, len(parts))):
            return True
    return False


def _drop_stripped_record_lines(image):
    """Take out of each wheel's RECORD the lines for files strip removed. They
    describe files the image no longer has, and uv's console-script launchers
    hash the build folder's path into theirs. Kept lines stay byte for byte."""
    for record in sorted((image / SITE).glob("*.dist-info/RECORD")):
        raw = record.read_bytes()
        kept = []
        for line in raw.splitlines(keepends=True):
            fields = next(csv.reader([line.decode("utf-8").rstrip("\r\n")]), [""])
            rel = posixpath.normpath(f"{SITE}/{fields[0]}") if fields and fields[0] else ""
            if not (rel and _stripped(rel)):
                kept.append(line)
        if len(kept) != len(raw.splitlines()):
            record.write_bytes(b"".join(kept))


def strip(image) -> dict[str, int]:
    image = pathlib.Path(image)
    counts = {}
    for pattern, _reason in STRIP_RULES:
        hits = _rule_hits(image, pattern)
        for rel in hits:
            path = image / rel
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        counts[pattern] = len(hits)
    _drop_stripped_record_lines(image)
    return counts


def strip_violations(image) -> list[str]:
    found = set()
    for pattern, _reason in STRIP_RULES:
        found.update(_rule_hits(image, pattern))
    return sorted(found)


# ---------- precompile ----------

def _exclude_rx():
    """COMPILE_EXCLUDE for compileall, which matches OS paths: each "/" also
    matches "\\"."""
    if COMPILE_EXCLUDE is None:
        return None
    return re.compile(COMPILE_EXCLUDE.replace("/", r"[/\\]"))


def compile_all(image) -> bool:
    """Every .py under COMPILE_ROOTS gets an unchecked-hash .pyc whose
    co_filename starts at the image (FoodOptimizer/...), compiled in one new
    Python process, one file at a time, in sorted order. marshal writes a
    string as a back-reference when something else in the process holds it,
    so the bytes follow what that process compiled before: a worker pool, or
    a process that did other work first, gives different bytes from run to
    run. The hash seed is fixed too. (About 25 s slower than a pool.)"""
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("PYTHON")}
    env["PYTHONHASHSEED"] = "0"
    # -I without -E (which would ignore the seed): -s -P and no PYTHON* vars.
    r = subprocess.run([sys.executable, "-s", "-P", "-B", str(pathlib.Path(__file__).resolve()),
                        "_compile", str(image)], env=env)
    return r.returncode == 0


def _compile_here(image) -> bool:
    image = pathlib.Path(image)
    ok = True
    for root in COMPILE_ROOTS:
        folder = image / root
        if not folder.is_dir():
            continue
        ok = compileall.compile_dir(
            str(folder), quiet=1, force=True, workers=1, rx=_exclude_rx(),
            stripdir=str(image), prependdir=IMAGE_NAME,
            invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH) and ok
    return bool(ok)


def pe_files(image) -> list[str]:
    """Image-relative "/" paths of every file with a PE header (MZ, and e_lfanew
    pointing at "PE\\0\\0"), whatever its extension. Reads 4 KiB at most per file,
    plus a seek when e_lfanew lies beyond it."""
    image = pathlib.Path(image)
    found = []
    for rel, is_dir in _walk(image):
        if is_dir:
            continue
        try:
            with open(image / rel, "rb") as f:
                head = f.read(4096)
                if head[:2] != b"MZ" or len(head) < 0x40:
                    continue
                offset = int.from_bytes(head[0x3C:0x40], "little")
                if offset + 4 <= len(head):
                    sig = head[offset:offset + 4]
                else:
                    f.seek(offset)
                    sig = f.read(4)
        except OSError:
            continue
        if sig == b"PE\0\0":
            found.append(rel)
    return sorted(found)


def _sources(image):
    image = pathlib.Path(image)
    for root in COMPILE_ROOTS:
        folder = image / root
        if not folder.is_dir():
            continue
        for rel, is_dir in _walk(folder):
            if not is_dir and rel.endswith(".py"):
                full = f"{root}/{rel}"
                if COMPILE_EXCLUDE is None or not re.search(COMPILE_EXCLUDE, full):
                    yield full


def pyc_gaps(image) -> list[str]:
    """Every .py whose .pyc is missing, not hash-based unchecked, or stale."""
    image = pathlib.Path(image)
    gaps = []
    for rel in _sources(image):
        source = image / rel
        pyc = pathlib.Path(importlib.util.cache_from_source(str(source)))
        try:
            head = pyc.read_bytes()[:16]
        except OSError:
            gaps.append(rel)
            continue
        if (len(head) < 16 or head[:4] != importlib.util.MAGIC_NUMBER
                or struct.unpack("<I", head[4:8])[0] != 1
                or head[8:16] != importlib.util.source_hash(source.read_bytes())):
            gaps.append(rel)
    return sorted(gaps)


# ---------- long-path gate ----------

def _lengths(image):
    for rel, _is_dir in _walk(image):
        yield len(rel), rel


def longest_relative(image) -> tuple[int, str]:
    """The longest image-relative path, counted with "\\" separators."""
    return max(_lengths(image), default=(0, ""))


def long_path_violations(image) -> list[str]:
    return [f"{n} {rel}" for n, rel in sorted(_lengths(image), key=lambda x: x[1])
            if ROOT_BUDGET + n > PATH_LIMIT]


# ---------- PE imports and the C++ runtime ----------

def pe_imports(path) -> set[str]:
    """Lower-cased DLL names from the import (1) and delay-import (13)
    directories of a PE32/PE32+ file. Anything it cannot parse raises
    ValueError naming the file: an unreadable binary must never read as
    "imports nothing", or the VC runtime gate would pass it (3b review)."""
    data = pathlib.Path(path).read_bytes()
    try:
        return _pe_imports(data)
    except (struct.error, ValueError, IndexError) as e:
        raise NotAPEFile(path, str(e) or type(e).__name__) from e


class NotAPEFile(ValueError):
    def __init__(self, path, reason):
        super().__init__(f"{path}: {reason}")
        self.reason = reason


def _pe_imports(data: bytes) -> set[str]:
    if data[:2] != b"MZ":
        raise ValueError("no MZ header")
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe:pe + 4] != b"PE\0\0":
        raise ValueError("no PE signature")
    sections, opt_size = struct.unpack_from("<H12xH", data, pe + 6)
    opt = pe + 24
    magic = struct.unpack_from("<H", data, opt)[0]
    if magic == 0x10B:
        image_base = struct.unpack_from("<I", data, opt + 28)[0]
        count = struct.unpack_from("<I", data, opt + 92)[0]
        dirs = opt + 96
    elif magic == 0x20B:
        image_base = struct.unpack_from("<Q", data, opt + 24)[0]
        count = struct.unpack_from("<I", data, opt + 108)[0]
        dirs = opt + 112
    else:
        raise ValueError(f"unknown optional header magic {magic:#x}")
    table = opt + opt_size
    spans = []
    for i in range(sections):
        vsize, va, raw_size, raw = struct.unpack_from("<IIII", data, table + 40 * i + 8)
        spans.append((va, max(vsize, raw_size), raw))

    def offset(rva):
        for va, size, raw in spans:
            if va <= rva < va + size:
                return rva - va + raw
        raise ValueError(rva)

    def cstring(rva):
        start = offset(rva)
        end = data.index(b"\0", start)
        return data[start:end].decode("ascii").lower()

    def directory(index):
        if index >= count:
            return 0
        return struct.unpack_from("<I", data, dirs + 8 * index)[0]

    names = set()
    rva = directory(1)
    if rva:
        at = offset(rva)
        for _ in range(4096):
            desc = struct.unpack_from("<5I", data, at)
            if not any(desc):
                break
            names.add(cstring(desc[3]))
            at += 20
    rva = directory(13)
    if rva:
        at = offset(rva)
        for _ in range(4096):
            desc = struct.unpack_from("<8I", data, at)
            if not any(desc):
                break
            name = desc[1] if desc[0] & 1 else desc[1] - image_base
            names.add(cstring(name))
            at += 32
    return names


def _names_under(folder):
    found = set()
    if folder.is_dir():
        for _dirpath, _dirnames, filenames in os.walk(folder):
            found.update(name.lower() for name in filenames)
    return found


def vc_runtime_gaps(image, imports_of=pe_imports) -> list[str]:
    """Every "<binary>: <name>" where a binary under python/ imports a VC
    runtime DLL that neither python/ nor its own package provides."""
    image = pathlib.Path(image)
    beside = {p.name.lower() for p in (image / "python").iterdir() if p.is_file()}
    site_prefix = SITE + "/"
    package_names = {}
    gaps = []
    for rel, is_dir in sorted(_walk(image / "python")):
        if is_dir or not rel.lower().endswith(PE_SUFFIXES):
            continue
        full = "python/" + rel
        try:
            imported = imports_of(image / full)
        except ValueError as e:
            gaps.append(f"{full}: unreadable PE file ({getattr(e, 'reason', e)})")
            continue
        wanted = {n for n in imported if VC_RUNTIME_NAME.match(n)}
        if not wanted:
            continue
        local = set(beside)
        if full.startswith(site_prefix) and "/" in full[len(site_prefix):]:
            top = full[len(site_prefix):].split("/")[0]
            if top not in package_names:
                package_names[top] = (_names_under(image / SITE / top)
                                      | _names_under(image / SITE / f"{top}.libs"))
            local |= package_names[top]
        gaps += [f"{full}: {name}" for name in sorted(wanted - local)]
    return gaps


# ---------- the per-file manifest ----------

def file_manifest(image) -> tuple[str, int, int]:
    image = pathlib.Path(image)
    rows = []
    total = 0
    for rel, is_dir in _walk(image):
        if is_dir:
            continue
        digest = hashlib.sha256()
        with open(image / rel, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
                total += len(chunk)
        rows.append((rel, digest.hexdigest()))
    rows.sort()
    text = "".join(f"{sha}  {rel}\n" for rel, sha in rows)
    return text, len(rows), total


# ---------- CLI ----------

def _gate(repo, image, publish=None) -> list[str]:
    repo, image = pathlib.Path(repo), pathlib.Path(image)
    problems = [f"strip: {rel}" for rel in strip_violations(image)]
    problems += [f"pyc: {rel}" for rel in pyc_gaps(image)]
    problems += [f"long path: {v}" for v in long_path_violations(image)]
    problems += [f"vc runtime: {gap}" for gap in vc_runtime_gaps(image)]
    golden = (repo / "desktop" / "windows" / "image_files.txt").read_text(encoding="utf-8").splitlines()
    shell = ()
    if publish is not None:
        problems += [f"shell: {gap}" for gap in shell_gaps(publish, image)]
        shell = shell_listing(publish)
    actual = image_listing(image, exclude=shell)
    problems += [f"listing: missing {e}" for e in sorted(set(golden) - set(actual))]
    problems += [f"listing: unexpected {e}" for e in sorted(set(actual) - set(golden))]
    return problems


def _listing_cmd(args) -> int:
    """`listing [--write] [--repo <path>]`: check desktop/windows/image_files.txt
    against expected_listing(repo), or rewrite it. A stale check prints a diff."""
    import difflib
    args = list(args)
    repo = pathlib.Path(__file__).resolve().parent.parent.parent
    if "--repo" in args:
        at = args.index("--repo")
        if at + 1 >= len(args):
            print("usage: build_env.py listing [--write] [--repo <path>]")
            return 2
        repo = pathlib.Path(args[at + 1])
        del args[at:at + 2]
    if args not in ([], ["--write"]):
        print("usage: build_env.py listing [--write] [--repo <path>]")
        return 2
    golden = repo / "desktop" / "windows" / "image_files.txt"
    want = "".join(f"{e}\n" for e in expected_listing(repo))
    have = golden.read_text(encoding="utf-8") if golden.exists() else ""
    if args == ["--write"]:
        golden.write_bytes(want.encode("utf-8"))
        return 0
    if have == want:
        return 0
    sys.stdout.writelines(difflib.unified_diff(have.splitlines(True), want.splitlines(True),
                                               "image_files.txt", "expected"))
    return 1


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd, args = (argv[0], argv[1:]) if argv else ("", [])
    if cmd == "listing":
        return _listing_cmd(args)
    if cmd == "assemble" and len(args) == 2:
        assemble(*args)
        return 0
    if cmd == "strip" and len(args) == 1:
        for pattern, n in strip(args[0]).items():
            print(f"stripped {n} {pattern}")
        left = strip_violations(args[0])
        for rel in left:
            print(f"strip: {rel}")
        return 1 if left else 0
    if cmd == "_compile" and len(args) == 1:      # compile_all's own process
        return 0 if _compile_here(args[0]) else 1
    if cmd == "compile" and len(args) == 1:
        if compile_all(args[0]):
            return 0
        print("compile: some files did not compile (see above)")
        return 1
    if cmd == "shell" and len(args) == 2:
        try:
            files = add_shell(*args)
        except ValueError as e:
            print(f"shell: {e}")
            return 1
        print(f"shell_files={len(files)}")
        return 0
    if cmd == "gate" and (len(args) == 2 or (len(args) == 4 and args[2] == "--shell")):
        problems = _gate(args[0], args[1], args[3] if len(args) == 4 else None)
        for line in problems:
            print(line)
        print(f"longest_path_chars={longest_relative(args[1])[0]}")
        return 1 if problems else 0
    if cmd == "files" and len(args) == 2:
        text, files, size = file_manifest(args[0])
        out = pathlib.Path(args[1])
        out.write_bytes(text.encode("utf-8"))
        print(f"image_files={files}")
        print(f"image_bytes={size}")
        print(f"image_sha256={hashlib.sha256(out.read_bytes()).hexdigest()}")
        return 0
    if cmd == "pe-files" and len(args) == 2:
        files = pe_files(args[0])
        pathlib.Path(args[1]).write_bytes("".join(f"{rel}\n" for rel in files).encode("utf-8"))
        print(f"pe_files={len(files)}")
        return 0
    print("usage: build_env.py assemble <repo> <image> | strip <image> | compile <image>"
          " | shell <publish> <image> | gate <repo> <image> [--shell <publish>]"
          " | files <image> <out.txt> | pe-files <image> <out.txt>"
          " | listing [--write] [--repo <path>]")
    return 2


if __name__ == "__main__":
    sys.exit(main())
