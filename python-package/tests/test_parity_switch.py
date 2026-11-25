import pytest

from test_parity_with_app import load_app_module


def test_missing_app_fails_when_parity_required(monkeypatch):
    monkeypatch.setenv("EGBO_REQUIRE_PARITY", "1")
    monkeypatch.setitem(__import__("sys").modules, "food_bo", None)
    with pytest.raises(RuntimeError, match="EGBO_REQUIRE_PARITY"):
        load_app_module()


def test_missing_app_skips_without_the_switch(monkeypatch):
    monkeypatch.delenv("EGBO_REQUIRE_PARITY", raising=False)
    monkeypatch.setitem(__import__("sys").modules, "food_bo", None)
    with pytest.raises(pytest.skip.Exception):
        load_app_module()
