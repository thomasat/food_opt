import re
from pathlib import Path

import pytest

PINS = Path(__file__).resolve().parents[1] / "egbo-pins.txt"
LOCK = Path(__file__).resolve().parents[2] / "desktop" / "requirements.lock.txt"


def parse(path):
    found = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"([A-Za-z0-9._-]+)==([^\s\;]+)", line.strip())
        if m:
            found[re.sub(r"[-_.]+", "-", m.group(1)).lower()] = m.group(2)
    return found


def test_pins_match_app_lock():
    if not LOCK.exists():
        pytest.skip("app lock not present")
    pins, lock = parse(PINS), parse(LOCK)
    assert pins
    for name, version in pins.items():
        assert lock.get(name) == version, name
