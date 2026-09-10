"""Small Streamlit UI patterns shared across app.py.

Why this exists: Streamlit discards everything rendered in a run that ends in
st.rerun(), so `st.success("Saved"); st.rerun()` shows nothing. The audit found
14 such sites. flash() queues the message in session state; render_flash() at
the top of the script shows it on the next run.
"""
from datetime import datetime

import streamlit as st

# join_unit and goal_line live in food_bo (they are pure data formatting and
# closeness_details needs them too); the tab modules import them from here so
# there is one import site for screen helpers.
from food_bo import goal_line, join_unit, number_list  # noqa: F401  (re-exported)

_FLASH_KEY = "_flash_messages"


def flash(kind, message):
    """Queue a message for the top of the NEXT run. Call right before st.rerun().

    kind: "success" | "info" | "warning" | "error"
    """
    if kind not in ("success", "info", "warning", "error"):
        raise ValueError(f"unknown flash kind {kind!r}")
    st.session_state.setdefault(_FLASH_KEY, []).append((kind, message))


def render_flash(box=None):
    """Show and clear queued messages, and return the container they went into.

    The messages always render inside one container, so the page keeps the
    same shape whether or not a message is showing. Rendering them as bare
    top-level elements shifted everything below by one slot on the run that
    cleared them, which made the browser rebuild the tabs and drop the user
    back to the first tab (seen right after opening a project and clicking
    Generate).

    Pass the container back to drain messages queued LATER in the same run —
    the sidebar runs after this point and can discard an unmakeable batch,
    and that notice belongs above the tabs on this run, not the next one.
    """
    box = st.container() if box is None else box
    with box:
        for kind, message in st.session_state.pop(_FLASH_KEY, []):
            getattr(st, kind)(message)
    return box


def saved_ok(opt):
    """True when the write that just ran reached the file. A green 'Added ...'
    over a change that never saved is a lie, so every handler asks first."""
    if getattr(opt, "save_error", None):
        st.error(opt.save_error)
        return False
    return True


ARMED_KEY = "_armed_confirmation"


def armed_confirmation():
    """The key of the one confirmation that is armed, or None.

    One key rather than a scan of every `{key}__pending` flag. A scan answers
    differently depending on where in the script it is asked from — a flag is
    only set once its own confirm_action has run — so two confirmations could
    end up armed on one frame, each with its own coloured Yes and its own
    copy of the project. This key is the single answer everything reads."""
    return st.session_state.get(ARMED_KEY)


def confirmation_open():
    """True while a confirmation is armed. Its "Yes" is the lit button, and
    a tab never shows two coloured buttons at once, so every tab's own primary
    steps aside while one is on screen."""
    return armed_confirmation() is not None


def other_confirmation(key):
    """True while a DIFFERENT confirmation is armed. Two armed at once would
    put two coloured Yes buttons on the tab, each keeping its own copy, so
    arming one greys the other's button until it is answered."""
    armed = armed_confirmation()
    return armed is not None and armed != key


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
        # One at a time. A click that arrives while another confirmation is
        # armed — a second click in the same frame, or a click already in
        # flight against a button this frame draws greyed — is ignored rather
        # than lighting a second Yes beside the first.
        if armed_confirmation() in (None, key):
            st.session_state[ARMED_KEY] = key
            st.session_state[pending_key] = True
    if armed_confirmation() != key or not st.session_state.get(pending_key):
        return False
    slot = st.empty()                      # reserves the position above the buttons
    c1, c2 = st.columns(2)
    with c1:
        confirmed = st.button(confirm_label, key=f"{key}__yes", type="primary",
                              use_container_width=True)
    with c2:
        if st.button("Cancel", key=f"{key}__no", use_container_width=True):
            st.session_state[pending_key] = False
            st.session_state.pop(ARMED_KEY, None)
            st.rerun()
    if confirmed:
        st.session_state[pending_key] = False
        st.session_state.pop(ARMED_KEY, None)
        return True
    slot.warning(warning)                  # only on runs where the user has not confirmed
    return False


# The three tabs, in loop order. The separator is U+00B7 MIDDLE DOT.
TAB_SETUP = "1 · Set up"
TAB_BATCH = "2 · Make a batch"
TAB_RESULTS = "3 · Results"


def go_to_tab(label):
    """Move to another tab and rerun. This is the ONLY way the app changes
    tabs, and it is called from six handlers only: Continue to make a batch,
    Back to set up, Save results, Save uploaded results, Start the next batch,
    and opening a project. A set-up edit, Generate, a correction or a plain
    rerun must never call it, and neither must a handler whose write failed.

    The move is deferred: the tabs widget already exists on this run, so
    assigning its key now would raise, and a write the frontend never asked
    for is ignored anyway. We park the target in "_pending_tab" and rerun;
    app.py drains it into "main_tab" BEFORE st.tabs renders, which is the one
    moment the widget takes a value from session state."""
    st.session_state["_pending_tab"] = label
    st.rerun()


def open_rows(opt):
    """Rows of the open batch that have not been recorded yet. Three screens
    count them — the line under the title, the result grid on tab 2 and the
    foot of tab 3 — and they must agree."""
    recorded = {int(i) for i in opt.formulation_ids}
    return [r for r in (opt.pending_batch or [])
            if r['formulation'] not in recorded]


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


def outside_message(name, value, low, high, unit, what, tail=""):
    """'Firmness 12 N is outside your scale of 0 to 10 N.' — the one builder
    for every out-of-bounds line, so a measurement and an amount are refused
    and warned about in the same words. `what` names the bounds, `tail` is any
    sentence that follows."""
    return (join_unit(f"{name} {float(value):g}", unit)
            + f" is outside {what} of "
            + join_unit(f"{float(low):g} to {float(high):g}", unit)
            + "." + tail)


def scale_error(obj, value):
    """The refusal for a measured value outside its scale, or '' when it fits.
    Results are never clamped: a firmness of 12 on a 0-10 scale is either a
    typo or a scale that is too narrow, and silently storing 10 hides both."""
    if value is None:
        return ""
    low, high = float(obj['min_val']), float(obj['max_val'])
    if low <= float(value) <= high:
        return ""
    return outside_message(obj['name'], value, low, high, obj.get('unit'),
                           "your scale",
                           " Widen the scale in Set up, or check the value.")


def bounds_warning(name, value, low, high, unit):
    """The caution for an amount outside what the project allows, or '' when it
    fits. Same builder as scale_error, so the two lines never drift apart."""
    if value is None:
        return ""
    if float(low) <= float(value) <= float(high):
        return ""
    return outside_message(name, value, low, high, unit, "its allowed amounts")


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


def clear_selection(key):
    """Ask for a select box to be emptied on the NEXT run.

    Popping a widget's key does not reach the browser: it keeps the old value
    and sends it back with the next click, so a row closed by Done reopened
    under the user's finger and swallowed that click. Assigning None instead is
    refused once the widget exists on this run. So the request is parked here
    and honoured by take_clear() just before the widget is created."""
    st.session_state[f"_clear_{key}"] = True


def take_clear(key):
    """Honour a pending clear_selection. Call immediately BEFORE the widget."""
    if st.session_state.pop(f"_clear_{key}", False):
        st.session_state[key] = None


def best_formulation_no(opt):
    """The number of the best formulation, or None. Three tabs name it — the
    repeat checkbox, the biggest-changes line and every 'Best moved' sentence —
    and they must all mean the same formulation."""
    i = opt.best_index()
    return None if i is None else int(opt.formulation_ids[i])


def best_move_sentence(before, after):
    """'Best moved from Formulation 3 to Formulation 7.' or '' when it stayed."""
    if before is None or after is None or before == after:
        return ""
    return f"Best moved from Formulation {before} to Formulation {after}."


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
