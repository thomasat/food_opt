"""The child bootstrap: the network guard (sitecustomize.py), the Host
check and the user-config isolation (serve_guard.py), and child.py."""
import asyncio
import http.server
import importlib.util
import json
import os
import socket
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest

from tests.desktop_launcher.conftest import SHARED
from tests.shared_texts import REPO, read_lf

LAUNCHER = SHARED / "launcher"
GUARD = LAUNCHER / "sitecustomize.py"
SERVE_GUARD = LAUNCHER / "serve_guard.py"
CHILD = LAUNCHER / "child.py"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_LOAD_BY_PATH = textwrap.dedent("""
    import importlib.util, sys
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
""")


def _run(code, *args, env=None, cwd=None, timeout=120):
    return subprocess.run([sys.executable, "-I", "-c", _LOAD_BY_PATH + textwrap.dedent(code),
                           *map(str, args)],
                          capture_output=True, text=True, env=env, cwd=cwd, timeout=timeout)


# ---------- the C++ runtime preload ----------

# The guard installs itself when it is imported, so every test that loads
# it does so in a child process, never in this one.
_PRELOAD = """
    import json, sys
    g = load("sitecustomize", sys.argv[1])
    loaded = []
    got = g.preload_cpp_runtime(sys.argv[2], sys.argv[3], loaded.append)
    print(json.dumps([got, loaded]))
"""


def _preload(platform, exe):
    out = _run(_PRELOAD, GUARD, platform, exe)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_preload_loads_the_bundled_runtime_first(tmp_path):
    for name in ("python.exe", "msvcp140.dll", "msvcp140_1.dll"):
        (tmp_path / name).write_bytes(b"")
    want = [str(tmp_path / "msvcp140.dll"), str(tmp_path / "msvcp140_1.dll")]
    assert _preload("win32", str(tmp_path / "python.exe")) == [want, want]
    assert _preload("darwin", str(tmp_path / "python.exe")) == [[], []]


def test_preload_skips_missing_files(tmp_path):
    (tmp_path / "python.exe").write_bytes(b"")
    assert _preload("win32", str(tmp_path / "python.exe")) == [[], []]


# ---------- no proxy, no IP lookups ----------

class _Ok(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *a):
        pass


@pytest.fixture
def local_http():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Ok)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv.server_address[1]
    srv.shutdown()
    srv.server_close()


_FETCH = """
    import sys, urllib.request
    if sys.argv[1] == "guard":
        load("sitecustomize", sys.argv[2])
    try:
        print(urllib.request.urlopen(sys.argv[3], timeout=5).status)
    except Exception as err:
        print(type(err).__name__)
"""


def test_guard_turns_proxies_off(local_http):
    env = {**os.environ, "HTTP_PROXY": "http://127.0.0.1:9", "http_proxy": "http://127.0.0.1:9"}
    env.pop("NO_PROXY", None)
    env.pop("no_proxy", None)
    url = f"http://127.0.0.1:{local_http}/"
    control = _run(_FETCH, "none", GUARD, url, env=env)
    assert control.stdout.strip() == "URLError", control
    guarded = _run(_FETCH, "guard", GUARD, url, env=env)
    assert guarded.stdout.strip() == "200", guarded


_IP_LOOKUPS = """
    import json, sys
    if sys.argv[1] == "before":
        load("sitecustomize", sys.argv[2])
    import streamlit.net_util as n
    if sys.argv[1] == "after":
        load("sitecustomize", sys.argv[2])
    seen = []
    LOCAL = ("127.0.0.1", "::1", "localhost")
    def hook(event, args):
        if event == "socket.connect":
            addr = args[1]
            if isinstance(addr, tuple) and addr and addr[0] not in LOCAL:
                seen.append(str(addr))
        elif event == "socket.getaddrinfo" and args[0] not in LOCAL:
            seen.append(str(args[0]))
    sys.addaudithook(hook)
    print(n.get_internal_ip(), n.get_external_ip())
    from streamlit.web.server import server_util
    server_util.is_url_from_allowed_origins("http://evil.example:1/")
    print(json.dumps(seen))
"""


@pytest.mark.parametrize("when", ["before", "after"])
def test_guard_stubs_streamlit_ip_lookups(when):
    out = _run(_IP_LOOKUPS, when, GUARD)
    lines = out.stdout.strip().splitlines()
    assert lines[:1] == ["None None"], out
    assert json.loads(lines[1]) == [], out


def test_guard_is_installed_once():
    out = _run("""
        import sys, urllib.request
        calls = []
        urllib.request.install_opener = lambda o: calls.append(o)
        g = load("sitecustomize", sys.argv[1])
        g.install()
        g.install()
        print(len(calls), g.FOODOPT_GUARD)
    """, GUARD)
    assert out.stdout.strip() == "1 True", out


# ---------- the Host check ----------

class _TornadoServer:
    def __init__(self, handlers):
        import tornado.web
        self.app = tornado.web.Application(handlers)
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        self.port = s.getsockname()[1]
        s.close()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=lambda: asyncio.run(self._main()), daemon=True)
        self._thread.start()
        assert self._ready.wait(10)

    async def _main(self):
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        server = self.app.listen(self.port, address="127.0.0.1")
        self._ready.set()
        await self._stop.wait()
        server.stop()

    def close(self):
        self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(10)


def _status(port, request: bytes) -> int:
    with socket.create_connection(("127.0.0.1", port), timeout=5) as c:
        c.sendall(request)
        data = b""
        while b"\r\n" not in data:
            chunk = c.recv(4096)
            if not chunk:
                break
            data += chunk
    return int(data.split(b" ", 2)[1])


@pytest.fixture
def host_checked(monkeypatch):
    tornado_web = pytest.importorskip("tornado.web")
    # Restored after the test: the check patches tornado's Application.
    monkeypatch.setattr(tornado_web.Application, "find_handler",
                        tornado_web.Application.find_handler)

    class Ok(tornado_web.RequestHandler):
        def get(self):
            self.write("ok")

    srv = _TornadoServer([("/", Ok)])
    _load("foodopt_serve_guard_test", SERVE_GUARD).install_host_check(f"127.0.0.1:{srv.port}")
    yield srv.port
    srv.close()


def _get(host, version=b"1.1"):
    head = b"GET / HTTP/" + version + b"\r\n"
    if host is not None:
        head += b"Host: " + host.encode() + b"\r\n"
    return head + b"Connection: close\r\n\r\n"


def test_host_check_serves_its_own_host(host_checked):
    assert _status(host_checked, _get(f"127.0.0.1:{host_checked}")) == 200


@pytest.mark.parametrize("host", ["LOCALHOST:{p}", "localhost:{p}", "evil.example:{p}",
                                  "127.0.0.1:{q}"])
def test_host_check_refuses_any_other_host(host_checked, host):
    p = host_checked
    assert _status(p, _get(host.format(p=p, q=p + 1))) == 403


def test_host_check_refuses_a_request_without_a_host(host_checked):
    assert _status(host_checked, _get(None, version=b"1.0")) == 403


# ---------- user config files ----------

_CONFIG = """
    import sys
    from streamlit import config
    if sys.argv[1] == "guard":
        load("foodopt_serve_guard", sys.argv[2]).ignore_user_config_files()
    config.get_config_options(force_reparse=True)
    print(config.get_option("theme.base"), config.get_option("server.baseUrlPath"),
          config.get_option("server.useStarlette"))
"""


def test_user_config_files_are_ignored(tmp_path):
    home = tmp_path / "home"
    (home / ".streamlit").mkdir(parents=True)
    (home / ".streamlit" / "config.toml").write_text('[theme]\nbase="dark"\n', encoding="utf-8")
    cwd = tmp_path / "projects"
    (cwd / ".streamlit").mkdir(parents=True)
    (cwd / ".streamlit" / "config.toml").write_text(
        '[server]\nbaseUrlPath="evil"\nuseStarlette=true\n', encoding="utf-8")
    env = {**os.environ, "HOME": str(home), "USERPROFILE": str(home)}
    control = _run(_CONFIG, "none", SERVE_GUARD, env=env, cwd=cwd)
    assert control.stdout.strip() == "dark evil True", control
    guarded = _run(_CONFIG, "guard", SERVE_GUARD, env=env, cwd=cwd)
    assert guarded.stdout.rstrip("\n") == "None  False", guarded


# ---------- what the launcher relies on in its dependencies ----------

def _installed(package, rel):
    spec = importlib.util.find_spec(package)
    if spec is None or not spec.submodule_search_locations:
        pytest.skip(f"{package} is not installed")
    return Path(list(spec.submodule_search_locations)[0]) / rel


def test_port_taken_text_is_streamlits_own():
    assert "Port %s is not available" in read_lf(_installed("streamlit", "web/server/server.py"))


def test_torch_reads_the_user_base_from_sysconfig():
    assert 'get_config_var("userbase")' in read_lf(_installed("torch", "__init__.py"))


# ---------- child.py ----------

def test_child_warmup_imports_what_launcher_sh_imports():
    child = _load("foodopt_child_test", CHILD)
    assert "import " + ", ".join(child.WARMUP_MODULES) in read_lf(REPO / "desktop" / "launcher.sh")


def test_child_serve_refuses_without_a_port(tmp_path):
    out = subprocess.run([sys.executable, "-I", str(CHILD), "serve", "app.py"],
                         capture_output=True, text=True, cwd=tmp_path, timeout=60)
    assert out.returncode == 1
    assert "serve needs --server.port" in out.stderr
