"""How the launcher starts its children: the command, the cleaned
environment, the Streamlit flags and the per-launch base path."""
import secrets
import sys
from pathlib import Path
from typing import Mapping, Optional

CHILD_FLAGS = ("-I", "-X", "utf8", "-B")
CREATE_NO_WINDOW = 0x08000000
PROXY_VARS = {"http_proxy", "https_proxy", "all_proxy", "no_proxy", "ftp_proxy"}


def child_env(environ: Mapping[str, str], state_dir: Path,
              base: Optional[str] = None) -> dict:
    """The parent's environment minus every PYTHON*, STREAMLIT_* and proxy
    variable. PYTHONUSERBASE points at a folder that is never created, so
    torch finds no user DLL folder to add to its search path. The server's
    secret base path travels here (STREAMLIT_SERVER_BASE_URL_PATH), never on
    its command line, where any local process could read it."""
    out = {k: v for k, v in environ.items()
           if not k.upper().startswith(("PYTHON", "STREAMLIT_"))
           and k.lower() not in PROXY_VARS}
    out["GIT_PYTHON_REFRESH"] = "quiet"
    out["PYTHONUSERBASE"] = str(Path(state_dir) / "no-user-base")
    if base:
        out["STREAMLIT_SERVER_BASE_URL_PATH"] = base
    return out


def streamlit_flags(port: int, base: Optional[str] = None) -> list:
    """Every option that matters, passed explicitly: no config file can
    change them."""
    flags = ["--server.headless=true",
             "--server.address=127.0.0.1",
             f"--server.port={port}",
             "--logger.hideWelcomeMessage=true"]
    flags += ["--browser.gatherUsageStats=false",
              "--client.toolbarMode=minimal",
              "--client.showErrorDetails=none",
              "--server.fileWatcherType=none",
              "--server.enableXsrfProtection=true",
              "--global.developmentMode=false",
              "--server.runOnSave=false",
              "--server.enableCORS=true",
              "--server.enableStaticServing=false",
              "--server.useStarlette=false",
              "--server.enableWebsocketCompression=false",
              "--server.maxUploadSize=200",
              "--browser.serverAddress=127.0.0.1",
              f"--browser.serverPort={port}",
              "--runner.magicEnabled=true"]
    return flags


def child_command(python: Path, child_script: Path, kind: str, *rest: str) -> list:
    return [str(python), *CHILD_FLAGS, str(child_script), kind, *rest]


def popen_kwargs(platform: Optional[str] = None) -> dict:
    """No console window for a child on Windows."""
    platform = sys.platform if platform is None else platform
    return {"creationflags": CREATE_NO_WINDOW} if platform == "win32" else {}


def new_base_path() -> str:
    """32 characters of [A-Za-z0-9_-], new on every launch."""
    return secrets.token_urlsafe(24)
