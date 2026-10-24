"""The two dependency locks: update_lock.sh writes both and keeps the Mac
lock's bytes unless its packages change, and the committed locks agree."""
import hashlib
import os
import pathlib
import re
import shutil
import subprocess

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
MAC_LOCK = REPO / "desktop" / "requirements.lock.txt"
WINDOWS_LOCK = REPO / "desktop" / "requirements-windows.lock.txt"
WINDOWS_ONLY = {"colorama", "watchdog"}
MAC_ONLY: set[str] = set()
MAC_LOCK_SHA256 = "457f16a64ade4ca1b6f7b63956309e137a23e35790f8bd9d253bd104d2771dad"


def pins(path):
    """{package: version} for every `name==version` line of a lock."""
    found = {}
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9_.-]+)==(\S+)", line)
        if m:
            found[re.sub(r"[_.]", "-", m.group(1).lower())] = m.group(2)
    return found


def _sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


# ---------- update_lock.sh behaviour, against a fake uv ----------

FAKE_UV = r"""#!/usr/bin/env bash
printf '%s\n' "$*" >> "$FAKE_UV_LOG"
cat > /dev/null
out=""; constraint=""; windows=0; prev=""
for arg in "$@"; do
  [ "$prev" = "-o" ] && out="$arg"
  [ "$prev" = "-c" ] && constraint="$arg"
  [ "$arg" = "--python-platform" ] && windows=1
  prev="$arg"
done
if [ "$windows" = "1" ]; then
  cp "$constraint" "$FAKE_C_COPY"
  cp "$FAKE_WIN_OUT" "$out"
  [ -n "${FAKE_WIN_VIA:-}" ] && printf '    # via -c %s\n' "$constraint" >> "$out"
  exit "${FAKE_WIN_EXIT:-0}"
fi
cp "$FAKE_MAC_OUT" "$out"
exit "${FAKE_MAC_EXIT:-0}"
"""


class LockSandbox:
    def __init__(self, root):
        self.root = root
        self.desktop = root / "desktop"
        self.desktop.mkdir(parents=True)
        shutil.copy(REPO / "desktop" / "update_lock.sh", self.desktop)
        shutil.copy(REPO / "requirements.txt", root)
        shutil.copy(MAC_LOCK, self.desktop)
        if WINDOWS_LOCK.exists():  # before it is committed, the script calls it new
            shutil.copy(WINDOWS_LOCK, self.desktop)
        self.uv = root / "fake-uv"
        self.uv.write_text(FAKE_UV, encoding="utf-8")
        self.uv.chmod(0o755)
        self.log = root / "uv.log"
        self.c_copy = root / "constraint-seen.txt"
        self.mac_out = root / "mac.out"
        self.win_out = root / "win.out"
        self.mac_out.write_bytes(MAC_LOCK.read_bytes())
        self.win_out.write_bytes(
            (WINDOWS_LOCK if WINDOWS_LOCK.exists() else MAC_LOCK).read_bytes())

    @property
    def mac_lock(self):
        return self.desktop / "requirements.lock.txt"

    @property
    def windows_lock(self):
        return self.desktop / "requirements-windows.lock.txt"

    def listing(self):
        return {p.name: p.read_bytes() for p in self.desktop.iterdir()}

    def run(self, platform="Darwin arm64", **env):
        full = dict(os.environ, FOODOPT_TEST_PLATFORM=platform,
                    FOODOPT_TEST_UV=str(self.uv), FAKE_UV_LOG=str(self.log),
                    FAKE_C_COPY=str(self.c_copy), FAKE_MAC_OUT=str(self.mac_out),
                    FAKE_WIN_OUT=str(self.win_out), **env)
        return subprocess.run(["bash", str(self.desktop / "update_lock.sh")],
                              env=full, capture_output=True, text=True, timeout=60)


@pytest.fixture
def lock_sandbox(tmp_path):
    return LockSandbox(tmp_path)


def test_a_comment_only_change_keeps_the_mac_lock_bytes(lock_sandbox):
    lines = MAC_LOCK.read_text(encoding="utf-8").splitlines(keepends=True)
    lines[1] = "#    ./desktop/update_lock.sh\n"
    via = next(i for i, line in enumerate(lines) if line.strip().startswith("# via"))
    lines[via] = lines[via].rstrip("\n") + " (and something else)\n"
    lock_sandbox.mac_out.write_text("".join(lines), encoding="utf-8")
    result = lock_sandbox.run()
    assert result.returncode == 0, result.stderr
    assert _sha256(lock_sandbox.mac_lock) == MAC_LOCK_SHA256
    assert ("Mac lock unchanged (same packages and hashes): "
            "desktop/requirements.lock.txt") in result.stdout


def test_a_version_change_rewrites_the_mac_lock(lock_sandbox):
    text = MAC_LOCK.read_text(encoding="utf-8")
    assert "altair==6.2.2" in text
    lock_sandbox.mac_out.write_text(text.replace("altair==6.2.2", "altair==6.2.3"),
                                    encoding="utf-8")
    result = lock_sandbox.run()
    assert result.returncode == 0, result.stderr
    assert lock_sandbox.mac_lock.read_bytes() == lock_sandbox.mac_out.read_bytes()
    assert "Wrote desktop/requirements.lock.txt (packages changed)" in result.stdout
    assert "re-measure EXPECTED_SETUP_MB" in result.stdout


def test_the_windows_lock_is_compiled_for_windows_against_the_new_mac_lock(lock_sandbox):
    text = MAC_LOCK.read_text(encoding="utf-8")
    lock_sandbox.mac_out.write_text(text.replace("altair==6.2.2", "altair==6.2.3"),
                                    encoding="utf-8")
    result = lock_sandbox.run()
    assert result.returncode == 0, result.stderr
    mac_call, windows_call = lock_sandbox.log.read_text(encoding="utf-8").splitlines()
    for flag in ("--python-version 3.13", "--generate-hashes", "--custom-compile-command"):
        assert flag in mac_call, mac_call
    assert "--python-platform" not in mac_call and " -c " not in mac_call, mac_call
    for flag in ("--python-platform x86_64-pc-windows-msvc", "--python-version 3.13",
                 "--generate-hashes", " -c "):
        assert flag in windows_call, windows_call
    assert lock_sandbox.c_copy.read_bytes() == lock_sandbox.mac_out.read_bytes()


def test_a_failed_compile_touches_neither_lock(lock_sandbox):
    text = MAC_LOCK.read_text(encoding="utf-8")
    lock_sandbox.mac_out.write_text(text.replace("altair==6.2.2", "altair==6.2.3"),
                                    encoding="utf-8")
    before = lock_sandbox.listing()
    result = lock_sandbox.run(FAKE_WIN_EXIT="1")
    assert result.returncode != 0
    assert lock_sandbox.listing() == before
    assert _sha256(lock_sandbox.mac_lock) == MAC_LOCK_SHA256


def test_the_windows_lock_names_the_committed_mac_lock(lock_sandbox):
    """uv writes the -c file's path into its `# via` lines; the compile's
    temporary copy must not reach the committed Windows lock."""
    lock_sandbox.windows_lock.unlink(missing_ok=True)
    result = lock_sandbox.run(FAKE_WIN_VIA="1")
    assert result.returncode == 0, result.stderr
    text = lock_sandbox.windows_lock.read_text(encoding="utf-8")
    assert text.rstrip("\n").endswith("# via -c desktop/requirements.lock.txt"), text[-200:]
    assert "mac.lock" not in text


@pytest.mark.parametrize("platform", ["Linux x86_64", "Darwin x86_64"])
def test_refuses_off_apple_silicon(lock_sandbox, platform):
    result = lock_sandbox.run(platform=platform)
    assert result.returncode == 2
    assert (f"refusing: compile the locks on an Apple Silicon Mac (this is {platform})"
            in result.stderr), result.stderr
    assert not lock_sandbox.log.exists()


# ---------- the committed locks ----------

def test_shared_packages_have_one_version():
    mac, windows = pins(MAC_LOCK), pins(WINDOWS_LOCK)
    differ = {p: (mac[p], windows[p]) for p in mac.keys() & windows.keys()
              if mac[p] != windows[p]}
    assert not differ, differ


def test_platform_only_packages_are_on_the_allowlist():
    mac, windows = set(pins(MAC_LOCK)), set(pins(WINDOWS_LOCK))
    assert windows - mac <= WINDOWS_ONLY, sorted(windows - mac)
    assert mac - windows <= MAC_ONLY, sorted(mac - windows)


def test_every_windows_pin_carries_a_hash():
    blocks, current = {}, None
    for line in WINDOWS_LOCK.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9_.-]+)==", line)
        if m:
            current = m.group(1)
            blocks[current] = 0
        elif current and line.strip().startswith("--hash=sha256:"):
            blocks[current] += 1
    assert blocks, "no pins in the Windows lock"
    assert not [p for p, n in blocks.items() if n == 0]


def test_the_windows_lock_installs_from_pypi():
    text = WINDOWS_LOCK.read_text(encoding="utf-8")
    assert "+cpu" not in text
    assert not re.search(r"^pytest==", text, re.M | re.I)
    assert "--index-url" not in text and "--extra-index-url" not in text


def test_mac_lock_bytes_change_only_on_purpose():
    """launcher.sh hashes the whole Mac lock (launcher.sh:152) to decide
    whether a Mac user's environment is current, so any byte change, even a
    comment, makes every Mac user redo the ~1 GB online setup. Update
    MAC_LOCK_SHA256 only in a release that bumps dependencies anyway."""
    assert _sha256(MAC_LOCK) == MAC_LOCK_SHA256
