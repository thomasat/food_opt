"""FOOD_OPT_SIDEBAR=collapsed: a presenter's opt-in that opens the window
with the sidebar folded. Unset, the page config is exactly what it was."""
import ast
import os
import pathlib

import pytest

import wording

APP = pathlib.Path(__file__).resolve().parent.parent / "app.py"


def _page_config_statement():
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    for node in tree.body:
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == "set_page_config"):
            return ast.Module(body=[node], type_ignores=[])
    raise AssertionError("app.py has no top-level st.set_page_config call")


class _FakeSt:
    def __init__(self):
        self.calls = []

    def set_page_config(self, **kwargs):
        self.calls.append(kwargs)


def _page_config(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("FOOD_OPT_SIDEBAR", raising=False)
    else:
        monkeypatch.setenv("FOOD_OPT_SIDEBAR", value)
    st = _FakeSt()
    code = compile(_page_config_statement(), str(APP), "exec")
    exec(code, {"st": st, "os": os, "wording": wording})
    assert len(st.calls) == 1
    return st.calls[0]


def test_unset_keeps_the_page_config_unchanged(monkeypatch):
    assert _page_config(monkeypatch, None) == {
        "page_title": wording.APP_TITLE, "layout": "wide"}


@pytest.mark.parametrize("value", ["", "expanded", "auto", "Collapsed", "1"])
def test_other_values_keep_the_page_config_unchanged(monkeypatch, value):
    assert _page_config(monkeypatch, value) == {
        "page_title": wording.APP_TITLE, "layout": "wide"}


def test_collapsed_folds_the_sidebar(monkeypatch):
    assert _page_config(monkeypatch, "collapsed") == {
        "page_title": wording.APP_TITLE, "layout": "wide",
        "initial_sidebar_state": "collapsed"}


def test_the_flag_puts_nothing_on_screen():
    # The opt-in reads an environment variable only: no new string in
    # wording, and app.py names it nowhere but in the page config.
    source = APP.read_text(encoding="utf-8")
    assert source.count("FOOD_OPT_SIDEBAR") == 2      # the comment and the read
    assert "FOOD_OPT_SIDEBAR" not in pathlib.Path(wording.__file__).read_text(encoding="utf-8")
