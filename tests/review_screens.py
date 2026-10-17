"""Shared helpers for the Review's AppTest tests: install a demo project,
open it on the Review, and read the tab's elements in order (formerly
tests/test_review_cards.py, whose card tests went with the v3 Review)."""
import json
import os
import re

import pytest
from streamlit.testing.v1 import AppTest

import bo_views
import ui_insights
import wording
from food_bo import FoodOptimizer
from tests import demo_projects as D
from tests.test_app_ui import APP_PATH



def _install(name):
    state = json.loads((D.HERE / f"{name}.json").read_text())
    opt = FoodOptimizer(state["project_name"])
    opt.import_json(state)
    opt.save()


def _open(name):
    at = AppTest.from_file(APP_PATH, default_timeout=600)
    at.session_state["_loaded_project"] = name
    at.run()
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    return at


@pytest.fixture(scope="module")
def screens(tmp_path_factory):
    here = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("review-cards"))
    try:
        with bo_views.fresh():
            out = {}
            for name in D.ALL:
                _install(name)
                out[name] = _open(name)
            yield out
    finally:
        os.chdir(here)


def _leaves(node, folds=()):
    """(element, fold labels it sits in, column index or None) in screen
    order; a fold's own row comes before its contents."""
    def walk(n, folds, col):
        k = -1
        for child in n.children.values():
            if child.type == "column":
                k += 1
                yield from walk(child, folds, k)
            elif child.type == "expander":
                yield child, folds, col
                yield from walk(child, folds + (child.label,), col)
            elif getattr(child, "children", None):
                yield from walk(child, folds, col)
            else:
                yield child, folds, col
    yield from walk(node, folds, None)


def _text(el):
    if el.type in ("markdown", "caption", "subheader", "heading"):
        return el.value
    if el.type in ("button", "expander"):
        return el.label
    return ""


# Final demo (B1): the strip's Stop is ':orange[**Stop**] 1.7 below top; …'.
_LABEL = re.compile(r"^(?::red\[\*\*(?P<red>.+?)\*\*\]|(?::orange\[)?\*\*(?P<word>.+?)\*\*\]?)"
                    r"\s*(?P<line>.*)$")


def strip(at):
    """Review story 2: the status strip, in screen order — [{label, help,
    word, line, key}] per item (word ':red[…]' when drawn red, line as
    drawn, ':red[**…**]' for a warning) — and the strip's buttons
    [(type, label)], read from the leaves above the first section."""
    items, buttons, cur = [], [], None
    # Review learned: the Predictions item reads "Model" once the model's
    # verdicts are in (the gate keeps "Predictions").
    labels = {
              }
    for el, folds, _ in _leaves(at.tabs[3]):
        text = _text(el)
        if el.type == "markdown" and text.startswith("#### "):
            break
        if el.type == "caption" and text in labels:
            cur = {"label": text, "help": el.proto.help or None, "word": None, "line": None,
                   "key": None}
            items.append(cur)
        elif el.type == "button" and el.key and el.key.startswith(ui_insights.SEE_CHART_KEY):
            m = _LABEL.match(el.label)
            cur["word"] = f":red[{m['red']}]" if m["red"] else m["word"]
            cur["line"], cur["key"] = m["line"], el.key
        elif el.type == "button" and items:     # the header's method link comes first
            buttons.append((el.proto.type, el.label))
    return items, buttons


def _headings(at):
    return [_text(el)[5:] for el, _, _ in _leaves(at.tabs[3])
            if el.type == "markdown" and _text(el).startswith("#### ")]


def outside(at):
    """[(type, text)] outside the cards and outside every fold."""
    return [(el.type, _text(el)) for el, folds, col in _leaves(at.tabs[3])
            if col is None and not folds and _text(el)]


def _words(text):
    plain = re.sub(r":\w+\[(.*?)\]", r"\1", text).replace("**", "")
    return len([w for w in plain.split() if re.search(r"\w", w)])   # as the DOM count


# ------------------------------------------------------------------ #


