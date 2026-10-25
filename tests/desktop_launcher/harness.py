"""Run a launcher — desktop/launcher.sh or the shared Python launcher — in a
throwaway home folder and look at what it leaves behind. The behaviour
suite runs every test against each implementation in IMPLS."""
import hashlib
import http.client
import json
import os
import random
import shutil
import signal
import stat
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence

from tests.desktop_launcher.conftest import SHARED
from tests.shared_texts import REPO

from launcher import files, ports, procinfo

FAKE_CHILD = Path(__file__).resolve().parent / "fake_child.py"
BOOT = ("import runpy,sys; sys.path.insert(0, sys.argv.pop(1)); sys.argv[0]='launcher'; "
        "runpy.run_module('launcher', run_name='__main__', alter_sys=True)")


def wait_for(predicate: Callable[[], object], timeout: float, interval: float = 0.05) -> bool:
    end = time.monotonic() + timeout
    while True:
        if predicate():
            return True
        if time.monotonic() >= end:
            return False
        time.sleep(interval)


def free_port_block() -> int:
    for _ in range(50):
        base = random.randrange(30000, 39000, 10)
        if all(ports.port_is_free(base + i) for i in range(10)):
            return base
    raise RuntimeError("no free block of ten ports")


@dataclass
class World:
    root: Path
    home: Path
    state_dir: Path
    projects_dir: Path
    record: Path
    launches: List["Launch"] = field(default_factory=list)

    @property
    def port_file(self) -> Path:
        return self.state_dir / files.PORT_FILE

    @property
    def base_file(self) -> Path:
        return self.state_dir / files.BASE_FILE

    @property
    def status_file(self) -> Path:
        return self.state_dir / files.STATUS_FILE

    @property
    def lock_dir(self) -> Path:
        return self.state_dir / files.LOCK_DIR

    @property
    def log(self) -> Path:
        return self.state_dir / files.LOG_FILE

    @property
    def backups(self) -> Path:
        return self.projects_dir / files.BACKUPS

    def port_pid(self):
        got = files.read_port_file(self.state_dir)
        return got if got and got[1] is not None else None

    def base(self) -> Optional[str]:
        try:
            return self.base_file.read_bytes().decode("ascii").strip() or None
        except OSError:
            return None

    def health(self, path: Optional[str] = None) -> Optional[int]:
        pp = self.port_pid()
        if pp is None:
            return None
        port = pp[0]
        if path is None:
            base = self.base()
            path = f"/{base}/_stcore/health" if base else "/_stcore/health"
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            try:
                conn.request("GET", path, headers={"Host": f"127.0.0.1:{port}"})
                return conn.getresponse().status
            finally:
                conn.close()
        except (OSError, http.client.HTTPException):
            return None

    def records(self, kind: Optional[str] = None) -> list:
        try:
            lines = self.record.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        out = [json.loads(line) for line in lines if line.strip()]
        return [r for r in out if kind is None or r["kind"] == kind]

    def log_text(self) -> str:
        try:
            return self.log.read_bytes().decode("utf-8", "replace")
        except OSError:
            return ""


def make_world(tmp_path: Path) -> World:
    root = Path(os.path.realpath(tmp_path))
    home = root / "José Müller"
    home.mkdir()
    if sys.platform == "win32":
        state = home / "AppData" / "Local" / "FoodOptimizer"
    else:
        state = home / "Library" / "Application Support" / "FoodOptimizer"
    return World(root=root, home=home, state_dir=state,
                 projects_dir=home / "FoodOptimizer", record=root / "record.jsonl")


def _kill(pid: int) -> None:
    if procinfo.pid_alive(pid):
        procinfo.terminate(pid, grace=0.5)


def end_world(world: World) -> None:
    """Ends every launch (closes the lifeline, or SIGTERM on POSIX; kills
    after 5 s), then the server named in server.port if it is alive and is
    not a launch, then every recorded child."""
    for launch in world.launches:
        if launch.proc.poll() is None:
            try:
                launch.stop()
            except OSError:
                pass
            try:
                launch.proc.wait(5)
            except subprocess.TimeoutExpired:
                launch.proc.kill()
        if launch.proc.stdin is not None:
            try:
                launch.proc.stdin.close()
            except OSError:
                pass
        try:
            launch.proc.wait(5)
        except Exception:
            pass
    port_pid = world.port_pid()
    launch_pids = {launch.proc.pid for launch in world.launches}
    if port_pid is not None and port_pid[1] not in launch_pids:
        _kill(port_pid[1])
    for record in world.records():
        _kill(record["pid"])


@dataclass
class Launch:
    proc: subprocess.Popen
    out: Path
    lifeline: bool = False

    def wait(self, timeout: float) -> int:
        return self.proc.wait(timeout)

    def stop(self) -> None:
        if sys.platform != "win32":
            self.proc.send_signal(signal.SIGTERM)
        elif self.lifeline and self.proc.stdin is not None:
            self.proc.stdin.close()
        else:
            self.proc.kill()

    def output(self) -> str:
        try:
            return self.out.read_bytes().decode("utf-8", "replace")
        except OSError:
            return ""


def _env(world: World, env: Optional[dict]) -> dict:
    out = dict(os.environ)
    for key in [k for k in out if k.startswith("FOODOPT_")]:
        del out[key]
    out["HOME"] = str(world.home)
    if sys.platform == "win32":
        out["USERPROFILE"] = str(world.home)
    else:
        out["FOODOPT_TEST_ARCH"] = "arm64"
    out["FOODOPT_IDLE_TIMEOUT_SECS"] = "600"
    out["FOODOPT_PORT_BASE"] = str(free_port_block())
    out["FOODOPT_FAKE_RECORD"] = str(world.record)
    for key, value in (env or {}).items():
        if value is None:
            out.pop(key, None)
        else:
            out[key] = value
    return out


def _popen(world: World, cmd: list, env: dict, stdin) -> Launch:
    out = world.root / f"launch-{len(world.launches)}.out"
    with open(out, "wb") as fh:
        proc = subprocess.Popen(cmd, env=env, stdin=stdin, stdout=fh, stderr=subprocess.STDOUT,
                                cwd=world.root)
    launch = Launch(proc=proc, out=out, lifeline=stdin == subprocess.PIPE)
    world.launches.append(launch)
    return launch


class BashImpl:
    """desktop/launcher.sh in a fake app bundle, with setup already done and
    a venv python that hands every child to fake_child.py."""
    name = "launcher.sh"

    def resources(self, world: World) -> Path:
        return world.root / "bundle" / "Food Optimizer.app" / "Contents" / "Resources"

    def _prepare(self, world: World) -> Path:
        res = self.resources(world)
        script = res / "launcher.sh"
        if not script.exists():
            res.mkdir(parents=True)
            shutil.copyfile(REPO / "desktop" / "launcher.sh", script)
            script.chmod(0o755)
            (res / "app.py").write_bytes(b"")
            shutil.copyfile(REPO / "desktop" / "requirements.lock.txt",
                            res / "requirements.lock.txt")
        world.state_dir.mkdir(parents=True, exist_ok=True)
        lock_hash = hashlib.sha256((res / "requirements.lock.txt").read_bytes()).hexdigest()
        (world.state_dir / "setup_complete").write_text(
            f"{lock_hash}\npython=3.13.7\n", encoding="utf-8")
        stub = world.state_dir / "venv" / "bin" / "python"
        if not stub.exists():
            stub.parent.mkdir(parents=True)
            py, fake = sys.executable, str(FAKE_CHILD)
            stub.write_text(
                "#!/bin/bash\n"
                f'if [ "$1" = "-c" ]; then exec "{py}" -I -X utf8 -B "{fake}" warmup; fi\n'
                'if [ "$1" = "-m" ] && [ "$2" = "streamlit" ] && [ "$3" = "run" ]; then\n'
                f'  shift 3; exec "{py}" -I -X utf8 -B "{fake}" serve "$@"\n'
                "fi\n"
                'echo "venv stub: unexpected arguments: $*" >&2\nexit 99\n',
                encoding="utf-8")
            stub.chmod(stub.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        return script

    def start(self, world: World, *, env: Optional[dict] = None, args: Sequence[str] = (),
              stdin=subprocess.DEVNULL) -> Launch:
        script = self._prepare(world)
        return _popen(world, ["/bin/bash", str(script), *args], _env(world, env), stdin)


class PythonImpl:
    """The shared Python launcher, children replaced by fake_child.py."""
    name = "python"

    def app(self, world: World) -> Path:
        app = world.root / "app" / "app.py"
        if not app.exists():
            app.parent.mkdir(parents=True)
            app.write_bytes(b"")
        return app

    def command(self, world: World, args: Sequence[str] = (), child_script=FAKE_CHILD,
                app: Optional[Path] = None) -> list:
        cmd = [sys.executable, "-I", "-X", "utf8", "-B", "-c", BOOT, str(SHARED),
               "--state-dir", str(world.state_dir), "--projects-dir", str(world.projects_dir),
               "--app", str(app or self.app(world)), "--python", sys.executable]
        if child_script is not None:
            cmd += ["--child-script", str(child_script)]
        return cmd + list(args)

    def start(self, world: World, *, env: Optional[dict] = None, args: Sequence[str] = (),
              stdin=subprocess.DEVNULL, child_script=FAKE_CHILD,
              app: Optional[Path] = None) -> Launch:
        cmd = self.command(world, args, child_script=child_script, app=app)
        return _popen(world, cmd, _env(world, env), stdin)


class InstalledImpl(PythonImpl):
    """The launcher as an installed Food Optimizer runs it (Level 2): the
    install's own python.exe, `-m launcher` from the install's shared
    folder, the install's app. Children are still fake_child.py unless a
    test asks for the real one."""
    name = "installed"

    def __init__(self, app_dir: Path):
        self.app_dir = Path(app_dir)

    def app(self, world: World) -> Path:
        return self.app_dir / "app" / "app.py"

    def command(self, world: World, args: Sequence[str] = (), child_script=FAKE_CHILD,
                app: Optional[Path] = None) -> list:
        python = str(self.app_dir / "python" / "python.exe")
        cmd = [python, "-I", "-X", "utf8", "-B", "-m", "launcher",
               "--state-dir", str(world.state_dir), "--projects-dir", str(world.projects_dir),
               "--app", str(app or self.app(world)), "--python", python]
        if child_script is not None:
            cmd += ["--child-script", str(child_script)]
        return cmd + list(args)


# FOODOPT_INSTALLED_APP=<install folder> runs the suite against an install.
_INSTALLED = os.environ.get("FOODOPT_INSTALLED_APP")
IMPLS: list = [InstalledImpl(Path(_INSTALLED))] if _INSTALLED else [BashImpl(), PythonImpl()]
