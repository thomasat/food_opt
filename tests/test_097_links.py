"""0.8.0 BO views, Task 7 (spec 2026-09-30-bo-views §5; 04-navigation.md;
review I7, M12, M15): a link to Insights is a text-style button that asks
the user's own question, moves in one run without losing what was typed,
lands where it says and opens the fold it names; a move made by the app
lands at the top of the tab (the Results scroll bug)."""
import pathlib

import pytest
from tests.demo_build_reason import REVIEW_FOUR
from streamlit.testing.v1 import AppTest

import insights
import ui_helpers
import ui_insights
import wording
from tests.test_app_ui import APP_PATH
from tests.test_insights import _burger


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _app(opt, tab=None):
    at = AppTest.from_file(APP_PATH, default_timeout=240)
    at.session_state["_loaded_project"] = opt.project_name
    at.run()
    if tab is not None:
        at.session_state["main_tab"] = tab
        at.run()
    assert not at.exception
    return at


def _links(at):
    return [b for b in at.button if str(b.key or "").startswith("link_")]


def _scripts(at):
    return [e.proto.body for e in at.get("html")
            if getattr(e.proto, "unsafe_allow_javascript", False)]


def _blocks(node, out=None):
    out = [] if out is None else out
    for child in getattr(node, "children", {}).values():
        if hasattr(child, "proto") and hasattr(child.proto, "id"):
            out.append(child.proto.id)
        _blocks(child, out)
    return out


# ------------------------------------------------------------------ #
#  The mechanism (04 §5)
# ------------------------------------------------------------------ #


def test_go_to_tab_lands_at_the_top(monkeypatch):
    """04 §1: after Save results the move landed ~1,400 px down."""
    state = {}
    monkeypatch.setattr(ui_helpers.st, "session_state", state)
    monkeypatch.setattr(ui_helpers.st, "rerun", lambda: None)
    ui_helpers.go_to_tab(wording.TAB_RESULTS)
    assert state == {"_pending_tab": wording.TAB_RESULTS, ui_helpers.SCROLL_KEY: "top"}


def test_the_app_scrolls_once_after_a_move():
    opt = _burger(rounds=2)
    at = _app(opt)
    assert _scripts(at) == []
    at.session_state[ui_helpers.SCROLL_KEY] = "top"
    at.run()
    [script] = _scripts(at)
    assert "scrollTop = 0" in script
    assert ui_helpers.SCROLL_KEY not in at.session_state
    at.run()
    assert _scripts(at) == []                               # once


def test_a_section_is_scrolled_into_view():
    opt = _burger(rounds=2)
    at = _app(opt)
    at.session_state[ui_helpers.SCROLL_KEY] = ui_insights.STAND_SECTION
    at.run()
    [script] = _scripts(at)
    assert f".st-key-{ui_insights.STAND_SECTION}" in script and "scrollIntoView" in script


def test_insights_sections_carry_the_keys_links_land_on():
    opt = _burger(rounds=4)
    at = _app(opt, wording.TAB_INSIGHTS)
    ids = _blocks(at.tabs[3])
    # Review v3: Are we improving? and What next? (the Forecast check is gone).
    for key in (ui_insights.STAND_SECTION, ui_insights.NEXT_SECTION):
        assert any(i.endswith(f"-{key}") for i in ids), key
    assert at.session_state[ui_helpers.INSIGHTS_OPENED_KEY] == insights.data_key(opt)


# ------------------------------------------------------------------ #
#  E1: Results after Save results (04 §3, review I7)
# ------------------------------------------------------------------ #


def test_e1_is_gone_while_a_round_is_open():
    opt = _burger(rounds=4)
    opt.ask(2)
    assert _links(_app(opt, wording.TAB_RESULTS)) == []


def test_e1_waits_for_a_checkable_round():
    """Round 1 had nothing recorded before it: never checked."""
    opt = _burger(rounds=1, per_round=5)
    assert insights.rounds_not_checked(opt) == [1]
    assert _links(_app(opt, wording.TAB_RESULTS)) == []


def test_no_e1_in_a_process_only_project():
    """v3: Results has no "measured against predicted" link in any
    project (it was E1)."""
    from tests.test_094_make_a_round import _ferment
    opt = _ferment()
    opt.ask(2)
    for row in list(opt.pending_batch):
        opt.tell(row["recipe"], {"Acidity": 5.0}, formulation_no=row["formulation"],
                 batch_no=opt.pending_batch_no)
    opt.set_pending_batch(None)
    assert _links(_app(opt, wording.TAB_RESULTS)) == []


def test_opening_a_project_lands_at_the_top():
    """T47 review minor 7: opening a project is a move too."""
    opt = _burger(rounds=4)
    at = AppTest.from_file(APP_PATH, default_timeout=240)
    at.session_state["_loaded_project"] = opt.project_name
    at.session_state["_land_on_open"] = True
    at.run()
    assert not at.exception
    assert any("scrollTop = 0" in s for s in _scripts(at))


def test_the_tab_and_link_rules_in_the_source():
    """04 §5: the comments say links may open Insights and no flow moves there."""
    app = pathlib.Path(APP_PATH).read_text()
    assert "links the user clicks, open it" in app
    assert "No step of the loop moves the user there" in app
    assert "no flow moves there on its own" in ui_helpers.go_to_tab.__doc__
