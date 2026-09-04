"""Tests for ui_helpers: the two UI patterns every destructive/saving action uses."""
from streamlit.testing.v1 import AppTest

FLASH_SCRIPT = """
import streamlit as st
from ui_helpers import flash, render_flash
render_flash()
if st.button("go"):
    flash("success", "Done.")
    st.rerun()
"""


def test_flash_is_shown_on_the_run_after_rerun():
    at = AppTest.from_string(FLASH_SCRIPT)
    at.run()
    assert [s.value for s in at.success] == []
    at.button[0].click()
    at.run()
    assert [s.value for s in at.success] == ["Done."]


def test_flash_is_shown_only_once():
    at = AppTest.from_string(FLASH_SCRIPT)
    at.run()
    at.button[0].click()
    at.run()
    at.run()
    assert [s.value for s in at.success] == []


CONFIRM_SCRIPT = """
import streamlit as st
from ui_helpers import confirm_action
st.session_state.setdefault("done", 0)
if confirm_action("del", "Delete thing", "Really delete?", confirm_label="Yes, delete"):
    st.session_state["done"] += 1
st.write(f"done={st.session_state['done']}")
"""


def _btn(at, label):
    return next(b for b in at.button if b.label == label)


def test_confirm_action_needs_two_clicks():
    at = AppTest.from_string(CONFIRM_SCRIPT)
    at.run()
    _btn(at, "Delete thing").click()
    at.run()
    assert any("Really delete?" in w.value for w in at.warning)
    assert at.session_state["done"] == 0
    # The warning must sit above the confirm/cancel buttons, not below them:
    # walk main's ordered children and check the Warning element's position
    # precedes the Block (the st.columns row) holding the Yes/Cancel buttons.
    children = list(at.main.children.values())
    warning_idx = next(i for i, c in enumerate(children) if type(c).__name__ == "Warning")
    buttons_block_idx = next(
        i for i, c in enumerate(children)
        if type(c).__name__ == "Block" and any(b.label == "Yes, delete" for b in c.get("button"))
    )
    assert warning_idx < buttons_block_idx
    _btn(at, "Yes, delete").click()
    at.run()
    assert at.session_state["done"] == 1
    assert not any("Really delete?" in w.value for w in at.warning)


def test_confirm_action_cancel():
    at = AppTest.from_string(CONFIRM_SCRIPT)
    at.run()
    _btn(at, "Delete thing").click()
    at.run()
    _btn(at, "Cancel").click()
    at.run()
    assert at.session_state["done"] == 0
    assert not any("Really delete?" in w.value for w in at.warning)
