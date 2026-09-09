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
    """Show and clear queued messages. Call once, near the top of app.py.

    The messages always render inside one container, so the page keeps the
    same shape whether or not a message is showing. Rendering them as bare
    top-level elements shifted everything below by one slot on the run that
    cleared them, which made the browser rebuild the tabs and drop the user
    back to the first tab (seen right after opening a project and clicking
    Generate recipes).
    """
    box = st.container()
    with box:
        for kind, message in st.session_state.pop(_FLASH_KEY, []):
            getattr(st, kind)(message)


def confirm_action(key, button_label, warning, confirm_label="Yes, continue", disabled=False):
    """Two-step confirmation for an irreversible action.

    Renders `button_label`. After it is clicked, shows `warning` above a
    primary confirm button and a Cancel button (a placeholder reserves the
    warning's position above the button row, and is filled only on runs
    where the user has not just confirmed, so the confirming run's element
    tree carries no warning). Returns True only on the run in which the user
    clicks the confirm button; the caller then performs the action, flashes
    a message, and reruns. `key` must be unique per call site.
    """
    pending_key = f"{key}__pending"
    if st.button(button_label, key=f"{key}__btn", disabled=disabled):
        st.session_state[pending_key] = True
    if not st.session_state.get(pending_key):
        return False
    slot = st.empty()                      # reserves the position above the buttons
    c1, c2 = st.columns(2)
    with c1:
        confirmed = st.button(confirm_label, key=f"{key}__yes", type="primary",
                              use_container_width=True)
    with c2:
        if st.button("Cancel", key=f"{key}__no", use_container_width=True):
            st.session_state[pending_key] = False
            st.rerun()
    if confirmed:
        st.session_state[pending_key] = False
        return True
    slot.warning(warning)                  # only on runs where the user has not confirmed
    return False
