"""Choosing the server's port by actually binding it: Hyper-V and WSL
reserve ranges on Windows that refuse a bind (WinError 10013) though
nothing listens there."""
import socket
from typing import Collection, Optional

AVOID = 8501   # Streamlit's default; a developer's own server may hold it


def port_is_free(port: int) -> bool:
    """A bind to 127.0.0.1:port succeeds (no SO_REUSEADDR) and nothing
    accepts a connect there within 0.2 s. A refused connect and one that
    times out both count as free: Windows only reports a closed loopback
    port as refused after about 2 s of retries."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", port))
    except OSError:
        return False
    finally:
        s.close()
    c = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    c.settimeout(0.2)
    try:
        c.connect(("127.0.0.1", port))
    except (ConnectionRefusedError, TimeoutError):
        return True
    except OSError:
        return False
    finally:
        c.close()
    return False


def _any_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


def choose_port(base: Optional[int], exclude: Collection[int] = ()) -> int:
    """With a base (FOODOPT_PORT_BASE): the first free port from it. Without:
    a random free port the OS offers, never 8501."""
    if base is not None:
        port = base
        while port <= 65535:
            if port not in exclude and port_is_free(port):
                return port
            port += 1
        raise OSError(f"no free port from {base}")
    for _ in range(20):
        port = _any_free_port()
        if port != AVOID and port not in exclude and port_is_free(port):
            return port
    raise OSError("no free port")
