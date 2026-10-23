"""The shared app, ready for Windows (Windows port phase 1).

The 0.8.0 wording snapshot is the Mac-safety guard: every module-level
string in wording.py stays exactly as 0.8.0 shipped it unless its name is in
ALLOWED_CHANGES, so a Windows fix cannot quietly change what a Mac user reads.
"""
import json
import pathlib

import wording

SNAPSHOT = pathlib.Path(__file__).parent / "fixtures" / "wording_snapshot_080.json"

# Names whose 0.8.0 value is meant to change (or that are new). Each entry
# says which fix changed it.
ALLOWED_CHANGES = set()


def _all_text(value):
    if isinstance(value, str):
        return True
    if isinstance(value, (list, tuple)):
        return all(_all_text(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _all_text(v) for k, v in value.items())
    return False


def _plain(value):
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


def collect_wording_constants():
    """Every module-level name in wording whose value is text, or a list,
    tuple or dict made only of text — as JSON-ready values."""
    return {name: _plain(value) for name, value in vars(wording).items()
            if not name.startswith("__") and _all_text(value)}


def test_wording_constants_match_080_snapshot():
    frozen = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    current = collect_wording_constants()
    changed = {k for k in frozen
               if k not in ALLOWED_CHANGES and current.get(k) != frozen[k]}
    assert changed == set()
    missing = {k for k in frozen if k not in current and k not in ALLOWED_CHANGES}
    assert missing == set()
