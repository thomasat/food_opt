"""Small Streamlit UI patterns shared across app.py.

Why this exists: Streamlit discards everything rendered in a run that ends in
st.rerun(), so `st.success("Saved"); st.rerun()` shows nothing. The audit found
14 such sites. flash() queues the message in session state; render_flash() at
the top of the script shows it on the next run.
"""
import streamlit as st

_FLASH_KEY = "_flash_messages"


def flash(kind, message):
    """Queue a message for the top of the NEXT run. Call right before st.rerun().

    kind: "success" | "info" | "warning" | "error"
    """
    if kind not in ("success", "info", "warning", "error"):
        raise ValueError(f"unknown flash kind {kind!r}")
    st.session_state.setdefault(_FLASH_KEY, []).append((kind, message))


def render_flash():
    """Show and clear queued messages. Call once, near the top of app.py."""
    for kind, message in st.session_state.pop(_FLASH_KEY, []):
        getattr(st, kind)(message)
