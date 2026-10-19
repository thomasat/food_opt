"""0.8.0 BO views, small fixes after the answer-key refresh: "regression to
the mean" only in Method and details, and the README's Limits line. (The
Why map, luck line and background-fit tests went with the old Insights
engine.)"""
import os

import pytest
from streamlit.testing.v1 import AppTest

import ui_insights
import wording
from tests.test_app_ui import APP_PATH
from tests.test_insights import _burger

README = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "README.md")


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def test_method_and_details_says_the_term():
    opt = _burger(rounds=5, noise=0.6)
    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.session_state["_loaded_project"] = opt.project_name
    at.run()
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.session_state[ui_insights.HOW_KEY] = True
    at.run()
    assert not at.exception
    how = next(e for e in at.tabs[3].expander if e.label == wording.FOLD_METHOD)
    # One-glance Review (owner, 2026-10-14): the fold is one line per term
    # and the luck line's statistics term is off the screen; the
    # caveat itself lives in the README's "How the Review works", Limits.
    # Review v3: the five terms, worked out for this project; no README line.
    assert [m.value for m in how.markdown] == [f"- {line}" for line in
                                               ui_insights.definitions(opt)]
    section = open(README, encoding="utf-8").read().split("## How the Review works", 1)[1]
    assert "Best so far is one noisy maximum and tends to read high." in \
        section.split("\n## ", 1)[0]
