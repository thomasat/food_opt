"""The Python launcher with the real child.py and real Streamlit: the base
path, the Host check and the user-config isolation, end to end. Slow."""
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from tests.desktop_launcher.conftest import healthy_launch, slow
from tests.desktop_launcher.harness import wait_for

from launcher import procinfo

PROBE = Path(__file__).resolve().parent / "probe_app.py"

pytestmark = slow


def _launch(py, world, **kw):
    return healthy_launch(py, world, timeout=120, child_script=None, app=PROBE, **kw)


def _raw(port, path, host, extra=b"Connection: close\r\n"):
    """Status and body of one raw HTTP/1.1 GET (headers read to their end,
    the body by its Content-Length)."""
    req = f"GET {path} HTTP/1.1\r\nHost: {host}\r\n".encode() + extra + b"\r\n"
    with socket.create_connection(("127.0.0.1", port), timeout=10) as c:
        c.sendall(req)
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = c.recv(65536)
            if not chunk:
                break
            data += chunk
        head, _, body = data.partition(b"\r\n\r\n")
        length = 0
        for line in head.split(b"\r\n")[1:]:
            name, _, value = line.partition(b":")
            if name.strip().lower() == b"content-length":
                length = int(value.strip())
        while len(body) < length:
            chunk = c.recv(65536)
            if not chunk:
                break
            body += chunk
    return int(head.split(b" ", 2)[1]), body[:length]


def _lan_ipv4():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.0.2.1", 9))   # TEST-NET: nothing is sent
        ip = s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()
    return None if ip.startswith("127.") or ip == "0.0.0.0" else ip


def test_real_streamlit_answers_only_on_its_base_path_and_host(py, world):
    _, port, _ = _launch(py, world)
    base = world.base()
    good = f"127.0.0.1:{port}"
    status, body = _raw(port, f"/{base}/_stcore/health", good)
    assert (status, body) == (200, b"ok")
    assert _raw(port, f"/{base}/_stcore/health", f"localhost:{port}")[0] == 403
    assert _raw(port, f"/{base}/_stcore/health", "evil.example")[0] == 403
    assert _raw(port, "/_stcore/health", good)[0] == 404
    upgrade = (b"Upgrade: websocket\r\nConnection: Upgrade\r\n"
               b"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n")
    assert _raw(port, f"/{base}/_stcore/stream", "evil.example", upgrade)[0] == 403
    lan = _lan_ipv4()
    if lan is not None:
        with pytest.raises(OSError):
            socket.create_connection((lan, port), timeout=2).close()


def test_real_streamlit_serves_the_base_path_from_the_environment(py, world):
    _, port, pid = _launch(py, world)
    base = world.base()
    assert _raw(port, f"/{base}/_stcore/health", f"127.0.0.1:{port}") == (200, b"ok")
    assert _raw(port, "/_stcore/health", f"127.0.0.1:{port}")[0] == 404
    assert base not in world.log_text()
    command = subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True,
                             text=True).stdout if sys.platform != "win32" else None
    if command is not None:
        assert command.strip() and base not in command


def test_real_streamlit_ignores_user_config_files(py, world):
    (world.home / ".streamlit").mkdir(parents=True)
    (world.home / ".streamlit" / "config.toml").write_text(
        '[server]\nbaseUrlPath="evil"\nuseStarlette=true\n[theme]\nbase="dark"\n',
        encoding="utf-8")
    (world.projects_dir / ".streamlit").mkdir(parents=True)
    (world.projects_dir / ".streamlit" / "config.toml").write_text(
        "[server]\nport=1\n", encoding="utf-8")
    _, port, _ = _launch(py, world)
    base = world.base()
    assert _raw(port, f"/{base}/", f"127.0.0.1:{port}")[0] == 200
    assert world.health("/evil/_stcore/health") == 404


def test_real_streamlit_shuts_down_on_stdin_eof(py, world):
    launch, _, pid = _launch(py, world, args=["--stdin-lifeline"], stdin=subprocess.PIPE)
    launch.proc.stdin.close()
    assert launch.wait(30) == 0
    assert wait_for(lambda: not procinfo.pid_alive(pid), 10)
