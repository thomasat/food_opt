"""The launcher's run loop, in desktop/launcher.sh's order (see the package
docstring for the contract with the shell)."""
import argparse
import http.client
import os
import platform
import signal
import struct
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence

from launcher import childenv, connections, files, ports, procinfo
from launcher.status_lines import LOAD_LINE, STARTING_LINE

PACKAGE_DIR = Path(__file__).resolve().parent
MAX_SERVER_STARTS = 3
HEALTH_POLL_SECS = 0.5
WARMUP_POLL_SECS = 0.1


@dataclass(frozen=True)
class Options:
    state_dir: Path
    projects_dir: Path
    app: Path
    python: Path = Path(sys.executable)
    child_script: Path = PACKAGE_DIR / "child.py"
    stdin_lifeline: bool = False
    lock_held: bool = False
    base_path: str = "auto"
    warmup_timeout: float = 120
    health_timeout: float = 180
    idle_timeout: int = 900
    port_base: Optional[int] = None
    test_arch: Optional[str] = None


def _int_env(environ: Mapping[str, str], name: str) -> Optional[int]:
    value = (environ.get(name) or "").strip()
    return int(value) if value.isdigit() else None


class _Parser(argparse.ArgumentParser):
    """A usage error exits 1: argparse's own 2 is the code for "unsupported"."""

    def error(self, message):
        self.print_usage(sys.stderr)
        sys.stderr.write(f"{self.prog}: error: {message}\n")
        sys.exit(1)


def parse_args(argv: Sequence[str], environ: Mapping[str, str]) -> Options:
    p = _Parser(prog="launcher", description="Food Optimizer launcher")
    p.add_argument("--state-dir", required=True, type=Path)
    p.add_argument("--projects-dir", required=True, type=Path)
    p.add_argument("--app", required=True, type=Path)
    p.add_argument("--python", type=Path, default=Path(sys.executable))
    p.add_argument("--child-script", type=Path, default=PACKAGE_DIR / "child.py")
    p.add_argument("--stdin-lifeline", action="store_true")
    p.add_argument("--lock-held", action="store_true")
    p.add_argument("--base-path", choices=("auto", "none"), default="auto")
    p.add_argument("--warmup-timeout", type=float, default=120)
    p.add_argument("--health-timeout", type=float, default=180)
    ns = p.parse_args(list(argv))
    idle = _int_env(environ, "FOODOPT_IDLE_TIMEOUT_SECS")
    return Options(state_dir=ns.state_dir, projects_dir=ns.projects_dir, app=ns.app,
                   python=ns.python, child_script=ns.child_script,
                   stdin_lifeline=ns.stdin_lifeline, lock_held=ns.lock_held,
                   base_path=ns.base_path, warmup_timeout=ns.warmup_timeout,
                   health_timeout=ns.health_timeout,
                   idle_timeout=900 if idle is None else idle,
                   port_base=_int_env(environ, "FOODOPT_PORT_BASE"),
                   test_arch=environ.get("FOODOPT_TEST_ARCH") or None)


def preflight(platform_name: str, arch: Optional[str]) -> Optional[str]:
    """The log line when this machine is unsupported, else None. Never
    platform.machine() on Windows: under emulation it names the host (B1)."""
    if platform_name == "darwin":
        arch = arch or platform.machine()
        major = (platform.mac_ver()[0] or "0").split(".")[0]
        major = int(major) if major.isdigit() else 0
        if arch != "arm64" or major < 13:
            return f"unsupported machine: arch={arch} macos={major}"
        return None
    if platform_name == "win32":
        arch = arch or ("AMD64" if struct.calcsize("P") == 8 else "x86")
        try:
            build = sys.getwindowsversion().build
        except (AttributeError, OSError):
            build = 0
        if arch not in {"AMD64", "x64"} or build < 19045:
            return f"unsupported machine: arch={arch} windows={build}"
        return None
    return f"unsupported machine: platform={platform_name}"


def healthy(port: int, base: Optional[str], timeout: float = 2.0) -> bool:
    path = f"/{base}/_stcore/health" if base else "/_stcore/health"
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        try:
            conn.request("GET", path, headers={"Host": f"127.0.0.1:{port}"})
            return conn.getresponse().status == 200
        finally:
            conn.close()
    except (OSError, http.client.HTTPException):
        return False


def watch_idle(server_alive: Callable[[], bool], count: Callable[[], int], timeout: int,
               wait: Callable[[float], bool], tick: float = 5.0) -> str:
    """launcher.sh's idle watchdog, in ticks: time the machine spends asleep
    is never counted. "idle", "server_exited" or "stopped"."""
    idle = 0.0
    while server_alive():
        if wait(tick):
            return "stopped"
        if count() > 0:
            idle = 0.0
        else:
            idle += tick
            if idle >= timeout:
                return "idle"
    return "server_exited"


def _secs(value: float) -> str:
    return f"{value:g}"


class _Run:
    def __init__(self, opts: Options, environ: Mapping[str, str], log: files.Log):
        self.opts = opts
        self.environ = environ
        self.log = log
        self.stop = threading.Event()
        self.reason: Optional[str] = None
        self.pid = os.getpid()
        self.lock = opts.state_dir / files.LOCK_DIR
        self.status_failed = False
        self.server: Optional[subprocess.Popen] = None
        self.port: Optional[int] = None

    # ---------- small helpers ----------

    def request_stop(self, reason: str) -> None:
        if self.reason is None:
            self.reason = reason
        self.stop.set()

    def status(self, text: str) -> None:
        try:
            files.atomic_write(self.opts.state_dir / files.STATUS_FILE,
                               files.status_line(text, None))
        except OSError as err:
            if not self.status_failed:
                self.status_failed = True
                self.log.say(f"could not write status.txt: {err}")

    def spawn(self, kind: str, *rest: str) -> subprocess.Popen:
        handle = self.log.child_handle()
        try:
            return subprocess.Popen(
                childenv.child_command(self.opts.python, self.opts.child_script, kind, *rest),
                cwd=str(self.opts.projects_dir),
                env=childenv.child_env(self.environ, self.opts.state_dir),
                stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                **childenv.popen_kwargs())
        finally:
            handle.close()

    @staticmethod
    def end(proc: Optional[subprocess.Popen], grace: float = 5.0) -> None:
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(grace)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    def log_size(self) -> int:
        try:
            return (self.opts.state_dir / files.LOG_FILE).stat().st_size
        except OSError:
            return 0

    def log_since(self, offset: int) -> str:
        try:
            with open(self.opts.state_dir / files.LOG_FILE, "rb") as fh:
                fh.seek(offset)
                return fh.read().decode("utf-8", "replace")
        except OSError:
            return ""

    # ---------- the steps ----------

    def start_lifeline(self) -> None:
        # os.read on the raw fd, never sys.stdin.buffer: a thread blocked in
        # BufferedReader.read holds its lock, and the interpreter's shutdown,
        # which closes sys.stdin, then dies on that lock ("_enter_buffered_busy
        # ... daemon threads") on every exit while the shell holds stdin open.
        def read_to_eof():
            try:
                fd = sys.stdin.fileno()
                while os.read(fd, 65536):
                    pass
            except (OSError, ValueError, AttributeError):
                pass
            self.request_stop("eof")
        threading.Thread(target=read_to_eof, name="stdin-lifeline", daemon=True).start()

    def lock_owner_alive(self, pid: int) -> bool:
        """The lock's pid names a live process that started before it wrote
        the pid file. Windows reuses pids quickly: a younger process with
        that pid is not the owner, and its lock is stale."""
        if not procinfo.pid_alive(pid):
            return False
        try:
            written = (self.lock / "pid").stat().st_mtime
        except OSError:
            return True
        start = procinfo.process_start_time(pid)
        return start is None or start <= written + 2

    def replace_orphan(self) -> None:
        state = self.opts.state_dir
        got = files.read_port_file(state)
        if got and got[1] is not None and procinfo.pid_alive(got[1]):
            pid = got[1]
            try:
                mtime = (state / files.PORT_FILE).stat().st_mtime
            except OSError:
                mtime = 0.0
            if procinfo.is_our_server(pid, self.opts.python, mtime):
                self.log.say(f"stopping orphaned server (pid {pid}) from a previous session")
                procinfo.terminate(pid, 5.0)
            else:
                self.log.say(f"server.port named pid {pid}, which is not our server - left running")
        files.remove_port_files(state)

    def warm_up(self) -> Optional[int]:
        """The bounded, non-fatal warm-up import. An exit code when the
        launch was stopped meanwhile, else None."""
        self.status(LOAD_LINE)
        self.log.say(LOAD_LINE)
        started = time.monotonic()
        warm = self.spawn("warmup")
        gave_up = False
        while warm.poll() is None:
            if self.stop.wait(WARMUP_POLL_SECS):
                self.end(warm, 2.0)
                files.clear_status(self.opts.state_dir)
                return 143 if self.reason == "signal" else 0
            if time.monotonic() - started >= self.opts.warmup_timeout:
                self.log.say(f"warm-up import still running after "
                             f"{_secs(self.opts.warmup_timeout)}s - giving up on it")
                self.end(warm, 2.0)
                gave_up = True
                break
        if gave_up or warm.returncode != 0:
            self.log.say("warm-up import did not finish - starting the server anyway")
        self.log.say(f"components loaded in {int(time.monotonic() - started)} s")
        return None

    def wait_healthy(self, base: Optional[str], offset: int) -> str:
        started = time.monotonic()
        while True:
            if self.stop.is_set():
                return "stopped"
            if healthy(self.port, base):
                return "healthy"
            if self.server.poll() is not None:
                if f"Port {self.port} is not available" in self.log_since(offset):
                    return "port_taken"
                return "exited"
            if time.monotonic() - started >= self.opts.health_timeout:
                return "timeout"
            self.stop.wait(HEALTH_POLL_SECS)

    def cleanup(self) -> None:
        self.end(self.server)
        state = self.opts.state_dir
        files.clear_status(state)
        if self.server is not None and files.port_file_is(state, self.port, self.server.pid):
            files.remove_port_files(state)
        files.release_lock(self.lock, self.pid)

    def fatal(self, message: str) -> int:
        self.log.say(f"FATAL: {message}")
        self.cleanup()
        return 1

    def stopped_early(self) -> Optional[int]:
        """The exit code when a stop was asked (cleaned up and logged like
        stopped()), else None."""
        if not self.stop.is_set():
            return None
        self.stopped()
        return 143 if self.reason == "signal" else 0

    def stopped(self) -> int:
        self.cleanup()
        self.log.say("signal received - shut down" if self.reason == "signal"
                     else "the window closed - shut down")
        return 0

    # ---------- the run ----------

    def run(self) -> int:
        opts = self.opts
        for d in (opts.state_dir, opts.projects_dir):
            d.mkdir(parents=True, exist_ok=True)
            if sys.platform != "win32":
                try:
                    os.chmod(d, 0o700)
                except OSError:
                    pass
        rotated_error = None
        if not opts.lock_held:
            _, rotated_error = files.rotate_log(opts.state_dir / files.LOG_FILE, self.lock)
        self.log.open(opts.state_dir / files.LOG_FILE)
        self.log.say(f"launcher started (app={opts.app})")
        if rotated_error:
            self.log.say(f"could not rotate launcher.log: {rotated_error}")
        if opts.stdin_lifeline:
            self.start_lifeline()

        self.replace_orphan()

        if opts.lock_held:
            if files.lock_owner(self.lock) != self.pid:
                self.log.say("FATAL: --lock-held but launch.lock is not ours")
                return 1
        else:
            ok, other = files.acquire_lock(self.lock, self.pid, alive=self.lock_owner_alive)
            if not ok:
                self.log.say("another launch is already in progress "
                             f"(pid {other if other is not None else 'unknown'}) - exiting")
                return 0
        try:
            early = self.stopped_early()
            if early is not None:
                return early
            return self.run_locked()
        finally:
            files.release_lock(self.lock, self.pid)

    def run_locked(self) -> int:
        opts = self.opts
        files.clear_status(opts.state_dir)
        files.remove_port_files(opts.state_dir)
        self.port = ports.choose_port(opts.port_base)
        if not opts.projects_dir.is_dir():
            self.log.say(f"FATAL: could not enter data dir {opts.projects_dir}")
            return 1
        try:
            made = files.backup_projects(opts.projects_dir)
        except OSError as err:
            made = None
            self.log.say(f"could not back up projects: {err}")
        if made is not None:
            self.log.say(f"projects backed up to {made}")

        code = self.stopped_early()
        if code is None:
            code = self.warm_up()
        if code is not None:
            return code

        tried = []
        for attempt in range(1, MAX_SERVER_STARTS + 1):
            self.status(STARTING_LINE)
            self.log.say(STARTING_LINE)
            base = childenv.new_base_path() if opts.base_path == "auto" else None
            offset = self.log_size()
            self.server = self.spawn("serve", str(opts.app),
                                     *childenv.streamlit_flags(self.port, base))
            try:
                if base:
                    files.write_base_file(opts.state_dir, base)
                files.write_port_file(opts.state_dir, self.port, self.server.pid)
            except OSError:
                return self.fatal("could not write server.port")
            result = self.wait_healthy(base, offset)
            if result == "healthy":
                break
            if result == "stopped":
                return self.stopped()
            if result == "port_taken" and attempt < MAX_SERVER_STARTS:
                self.log.say(f"port {self.port} was taken - trying another port")
                files.remove_port_files(opts.state_dir)
                tried.append(self.port)
                self.port = ports.choose_port(opts.port_base, exclude=tried)
                continue
            if result == "timeout":
                return self.fatal(f"server did not become healthy within "
                                  f"{_secs(opts.health_timeout)}s")
            return self.fatal("server exited before becoming healthy")

        self.log.say(f"server healthy on port {self.port} (pid {self.server.pid})")
        files.clear_status(opts.state_dir)
        files.release_lock(self.lock, self.pid)

        if opts.stdin_lifeline:
            while self.server.poll() is None:
                if self.stop.wait(HEALTH_POLL_SECS):
                    break
            result = "stopped" if self.stop.is_set() else "server_exited"
        else:
            port = self.port
            result = watch_idle(lambda: self.server.poll() is None,
                                lambda: connections.established_count(port),
                                opts.idle_timeout, self.stop.wait)
        if result == "stopped":
            return self.stopped()
        if result == "idle":
            # launcher.sh counts whole 5 s ticks, so it reports the tick total.
            ticks = -(-opts.idle_timeout // 5) * 5
            self.log.say(f"no window connected for {ticks}s - shutting down")
        self.cleanup()
        self.log.say("exiting")
        return 0


def main(argv: Optional[Sequence[str]] = None,
         environ: Optional[Mapping[str, str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    environ = os.environ if environ is None else environ
    opts = parse_args(argv, environ)
    log = files.Log()
    message = preflight(sys.platform, opts.test_arch)
    if message:
        log.say(message)
        return 2
    run = _Run(opts, environ, log)
    for name in ("SIGTERM", "SIGINT"):
        if hasattr(signal, name):
            try:
                signal.signal(getattr(signal, name), lambda *a: run.request_stop("signal"))
            except (OSError, ValueError):
                pass
    try:
        return run.run()
    except Exception:
        # launcher.sh sends everything to its log; so does this: the log is
        # what the user attaches when the app cannot open.
        log.say("FATAL: unexpected error")
        for line in traceback.format_exc().rstrip("\n").split("\n"):
            log.say(line)
        try:
            run.cleanup()
        except Exception:
            pass
        return 1
    finally:
        log.close()
