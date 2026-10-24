"""The launcher's on-disk formats, byte for byte as desktop/launcher.sh
writes them, plus the atomic, retried writes Windows needs (Defender and
the search indexer hold files for a moment)."""
import hashlib
import os
import shutil
import sys
import time
from pathlib import Path
from typing import BinaryIO, Callable, Optional, Tuple

PORT_FILE = "server.port"
BASE_FILE = "server.base"
STATUS_FILE = "status.txt"
LOCK_DIR = "launch.lock"
LOG_FILE = "launcher.log"
BACKUPS = "backups"
LAST_SUM = ".last_sum"
KEEP_BACKUPS = 10


def _tmp(path: Path) -> Path:
    return path.with_name(path.name + ".tmp")


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def atomic_write(path: Path, data: bytes, attempts: int = 5, delay: float = 0.05) -> None:
    """Write <path>.tmp, then rename it over `path`. A PermissionError on the
    rename (a file briefly held open) is retried; the last one is raised."""
    path = Path(path)
    tmp = _tmp(path)
    with open(tmp, "wb") as fh:
        fh.write(data)
    for attempt in range(1, attempts + 1):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == attempts:
                _unlink(tmp)
                raise
            time.sleep(delay * attempt)
        except OSError:
            _unlink(tmp)
            raise


# ---------- status.txt ----------

def write_status(state_dir: Path, text: str, percent: Optional[int]) -> bool:
    """Publish one "<text>|<percent>" line. False, never an exception, when
    it cannot be written: a status line is never worth a failed launch."""
    line = f"{text}|{'' if percent is None else percent}\n"
    try:
        atomic_write(Path(state_dir) / STATUS_FILE, line.encode("utf-8"))
        return True
    except OSError:
        return False


def clear_status(state_dir: Path) -> None:
    for name in (STATUS_FILE, STATUS_FILE + ".tmp"):
        try:
            _unlink(Path(state_dir) / name)
        except OSError:
            pass


# ---------- server.port / server.base ----------

def write_port_file(state_dir: Path, port: int, pid: int) -> None:
    atomic_write(Path(state_dir) / PORT_FILE, f"{port} {pid}\n".encode("ascii"))


def _read_text(path: Path) -> Optional[str]:
    try:
        return path.read_bytes().decode("utf-8", "replace")
    except OSError:
        return None


def read_port_file(state_dir: Path) -> Optional[Tuple[int, Optional[int]]]:
    """(port, pid) from server.port; pid None when the line has no pid; None
    when there is no file or no numeric port."""
    text = _read_text(Path(state_dir) / PORT_FILE)
    if text is None:
        return None
    parts = text.split()
    if not parts or not parts[0].isdigit():
        return None
    pid = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
    return int(parts[0]), pid


def port_file_is(state_dir: Path, port: int, pid: int) -> bool:
    """True when server.port still says this launch's port and pid, so a
    launcher that exits late never removes a newer launch's file."""
    text = _read_text(Path(state_dir) / PORT_FILE)
    return text is not None and text.rstrip("\n") == f"{port} {pid}"


def write_base_file(state_dir: Path, base: str) -> None:
    atomic_write(Path(state_dir) / BASE_FILE, f"{base}\n".encode("ascii"))


def remove_port_files(state_dir: Path) -> None:
    for name in (PORT_FILE, PORT_FILE + ".tmp", BASE_FILE, BASE_FILE + ".tmp"):
        try:
            _unlink(Path(state_dir) / name)
        except OSError:
            pass


# ---------- launch.lock ----------

def _lock_text(lock_dir: Path) -> str:
    return (_read_text(Path(lock_dir) / "pid") or "").strip()


def lock_owner(lock_dir: Path) -> Optional[int]:
    text = _lock_text(lock_dir)
    return int(text) if text.isdigit() else None


def _take(lock_dir: Path, pid: int) -> bool:
    try:
        os.mkdir(lock_dir)
    except OSError:
        return False
    with open(Path(lock_dir) / "pid", "wb") as fh:
        fh.write(f"{pid}\n".encode("ascii"))
    return True


def acquire_lock(lock_dir: Path, pid: int, alive: Callable[[int], bool],
                 sleep: Callable[[float], None] = time.sleep,
                 tries: int = 5) -> Tuple[bool, Optional[int]]:
    """launcher.sh's acquire_lock: (True, None) when the lock is ours, else
    (False, owner). An empty pid is given `tries` seconds to appear (the
    owner may be between its mkdir and its pid write) before it is stolen;
    a dead owner's lock is stolen at once."""
    lock_dir = Path(lock_dir)
    if _take(lock_dir, pid):
        return True, None
    text = ""
    for _ in range(tries):
        text = _lock_text(lock_dir)
        if text:
            break
        sleep(1)
    other = int(text) if text.isdigit() else None
    if other is not None and alive(other):
        return False, other
    shutil.rmtree(lock_dir, ignore_errors=True)
    if _take(lock_dir, pid):
        return True, None
    return False, other


def release_lock(lock_dir: Path, pid: int) -> bool:
    """Remove the lock only when it is `pid`'s."""
    if lock_owner(lock_dir) != pid:
        return False
    shutil.rmtree(lock_dir, ignore_errors=True)
    return True


# ---------- backups ----------

def _projects(projects_dir: Path):
    names = [p.name for p in Path(projects_dir).iterdir()
             if p.name.endswith(".pkl") and not p.name.startswith(".") and p.is_file()]
    return sorted(names, key=lambda n: n.encode("utf-8"))


def backup_projects(projects_dir: Path, keep: int = KEEP_BACKUPS,
                    now: Optional[time.struct_time] = None) -> Optional[Path]:
    """Copy every top-level *.pkl into backups/YYYY-MM-DD_HHMMSS/ unless
    nothing changed since the last backup (backups/.last_sum, the sha256 of
    the files' bytes in bytewise name order). Keeps the newest `keep`
    directories. Returns the new directory, or None."""
    projects_dir = Path(projects_dir)
    names = _projects(projects_dir)
    if not names:
        return None
    digest = hashlib.sha256()
    for name in names:
        digest.update((projects_dir / name).read_bytes())
    current = digest.hexdigest()
    root = projects_dir / BACKUPS
    last = _read_text(root / LAST_SUM)
    if last == current:
        return None
    stamp = time.strftime("%Y-%m-%d_%H%M%S", now or time.localtime())
    target = root / stamp
    target.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copyfile(projects_dir / name, target / name)
    (root / LAST_SUM).write_bytes(current.encode("ascii"))
    dirs = sorted(p for p in root.iterdir() if p.is_dir())
    while len(dirs) > keep:
        shutil.rmtree(dirs.pop(0), ignore_errors=True)
    return target


# ---------- launcher.log ----------

def rotate_log(log: Path, lock_dir: Path) -> Tuple[bool, Optional[str]]:
    """Move launcher.log to launcher.log.1, only when no launch is in
    progress. (rotated, error): a log another process holds open (Windows
    cannot rename it) is left in place and appended to."""
    log = Path(log)
    if not log.is_file() or Path(lock_dir).is_dir():
        return False, None
    try:
        os.replace(log, log.with_name(log.name + ".1"))
    except OSError as err:
        return False, str(err)
    return True, None


class Log:
    """launcher.sh's say(): one dated line to the stream and to the log."""

    def __init__(self, stream=None):
        self.stream = sys.stdout if stream is None else stream
        self.path: Optional[Path] = None
        self._fh = None

    def open(self, path: Path) -> None:
        self.path = Path(path)
        self._fh = open(self.path, "ab")

    def say(self, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n"
        try:
            self.stream.write(line)
            self.stream.flush()
        except (OSError, ValueError, AttributeError):
            pass
        if self._fh is not None:
            try:
                self._fh.write(line.encode("utf-8"))
                self._fh.flush()
            except (OSError, ValueError):
                pass

    def child_handle(self) -> BinaryIO:
        """A fresh append handle on the log, for a child's stdout/stderr."""
        return open(self.path, "ab")

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None
