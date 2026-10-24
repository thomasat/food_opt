"""The shared texts the two desktop shells read, for the tests that check
them: desktop/shared/messages.json, loaded once, and the helpers that read
repo text the same way on every OS (UTF-8, LF line endings)."""
import functools
import json
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parent.parent
MESSAGES = REPO / "desktop" / "shared" / "messages.json"


def read_lf(path) -> str:
    """A repo text file as UTF-8 with LF line endings. Git on Windows may
    check a file out with CRLF, and the bytes compared here are the LF ones."""
    return pathlib.Path(path).read_bytes().decode("utf-8").replace("\r\n", "\n")


def flowed(text: str) -> str:
    """A file's prose as one line — a copy of tests/test_app_ui.py::_flowed,
    kept here so these tests need not import that AppTest-heavy module."""
    joined = re.sub(r'"\s*\+\s*"', "", text)          # Swift: "a " + "b"
    joined = re.sub(r"^[>\s]+", " ", joined, flags=re.M)  # quote / indent
    return re.sub(r"\s+", " ", joined)


@functools.lru_cache(maxsize=None)
def _doc_text() -> str:
    return read_lf(MESSAGES)


def messages_doc() -> dict:
    return json.loads(_doc_text())


def entries() -> list:
    doc = messages_doc()
    return list(doc["messages"]) + list(doc["menus"])


def texts(os_name: str) -> dict:
    """id -> the text for one OS ("mac" or "windows"), entries without one
    left out."""
    return {e["id"]: e["text"][os_name] for e in entries()
            if e["text"].get(os_name) is not None}
