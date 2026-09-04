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
