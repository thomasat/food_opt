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


from datetime import datetime

# join_unit and goal_line live in food_bo (they are pure data formatting and
# closeness_details needs them too); the tab modules import them from here so
# there is one import site for screen helpers.
from food_bo import goal_line, join_unit, number_list  # noqa: F401  (re-exported)

# The three tabs, in loop order. The separator is U+00B7 MIDDLE DOT.
TAB_SETUP = "1 · Set up"
TAB_BATCH = "2 · Make a batch"
TAB_RESULTS = "3 · Results"


def go_to_tab(label):
    """Move to another tab and rerun. This is the ONLY way the app changes
    tabs, and it is called from four handlers only: Continue to make a batch,
    Save results, Start the next batch, and opening a project. A set-up edit,
    Generate, a correction or a plain rerun must never call it, and neither
    must a handler whose write failed."""
    st.session_state["main_tab"] = label
    st.rerun()


def plural(n, word):
    """'1 formulation', '3 formulations' — never '1 formulation(s)'."""
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def fmt_amount(value, unit="", decimals=2):
    """An amount as prose: '12.5 g', '12 g', '7/10', '' for a missing value.
    Trailing zeros are trimmed."""
    if value is None:
        return ""
    txt = f"{float(value):.{decimals}f}"
    if "." in txt:
        txt = txt.rstrip("0").rstrip(".")
    if txt in ("", "-0"):
        txt = "0"
    return join_unit(txt, unit)


def scale_error(obj, value):
    """The refusal for a measured value outside its scale, or '' when it fits.
    Results are never clamped: a firmness of 12 on a 0-10 scale is either a
    typo or a scale that is too narrow, and silently storing 10 hides both."""
    if value is None:
        return ""
    low, high = float(obj['min_val']), float(obj['max_val'])
    if low <= float(value) <= high:
        return ""
    unit = obj.get('unit')
    return (join_unit(f"{obj['name']} {float(value):g}", unit)
            + " is outside your scale of "
            + join_unit(f"{low:g} to {high:g}", unit)
            + ". Widen the scale in Set up, or check the value.")


def table_height(n_rows, max_rows=12):
    """Pixel height that shows up to max_rows rows of a st.dataframe without an
    inner scrollbar (35 px per row plus the header). Zero rows still get one
    row's height, so an empty table is not a sliver."""
    return 38 + 35 * max(1, min(int(n_rows), max_rows)) + 2


def saved_line(saved_at):
    """'Saved 14:32 · automatically, to this Mac', carrying the date when the
    last save was not today."""
    now = datetime.now().astimezone()
    if saved_at.date() == now.date():
        when = f"{saved_at:%H:%M}"
    else:
        when = saved_at.strftime("%d %b %H:%M").lstrip("0")
    return f"Saved {when} · automatically, to this Mac"


def readiness(opt):
    """(ready, the missing item named) for the foot button on tab 1 and the
    Generate button on tab 2. Ready means at least one ingredient and at least
    one measurement."""
    has_ingredient = any(
        v.get('category', 'ingredient') == 'ingredient' for v in opt.variables
    )
    if not has_ingredient:
        return False, "Add at least one ingredient."
    if not opt.objectives:
        return False, "Add at least one measurement."
    return True, ""


def landing_tab(opt):
    """Where opening a project lands: set-up while it is incomplete, the batch
    while one is unrecorded, results otherwise."""
    ready, _ = readiness(opt)
    if not ready:
        return TAB_SETUP
    if opt.pending_batch:
        return TAB_BATCH
    return TAB_RESULTS
