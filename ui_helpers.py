"""Small Streamlit UI patterns shared across app.py.

Why this exists: Streamlit discards everything rendered in a run that ends in
st.rerun(), so `st.success("Saved"); st.rerun()` shows nothing. The audit found
14 such sites. flash() queues the message in session state; render_flash() at
the top of the script shows it on the next run.
"""
import re
from datetime import datetime

import streamlit as st

import wording
# join_unit, goal_line and the two number formatters live in food_bo (they are
# pure data formatting, and closeness_details, the batch table and the
# what-is-it-trying column all need them there); the tab modules import them
# from here so there is one import site for screen helpers.
from food_bo import (  # noqa: F401  (re-exported)
    amount_range_placeholder, fmt_amount, fmt_setting, goal_line, join_unit,
    label_with_unit, number_list, outside_message, unit_after_number,
)
# Every word below lives in wording.py; these three names stay importable
# from here because ui_setup.py, ui_results.py and app.py already do
# `from ui_helpers import TAB_BATCH` and the like.
from wording import COPY_KEPT, TAB_BATCH, TAB_RESULTS, TAB_SETUP  # noqa: F401

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


RESTORE_KEY = "_restore_candidate"


def restore_armed():
    """True while a checked saved copy is waiting for `Yes, replace`.

    Restore is the one confirmation that is not a confirm_action: it is drawn
    by hand in the sidebar because it has a file to read and a summary to
    show first. It is still a confirmation — it replaces the project and keeps
    a copy — so it counts as one everywhere ARMED_KEY does, and the rest of
    the app dims behind it."""
    return st.session_state.get(RESTORE_KEY) is not None


def _confirmation_arming():
    """True on the run that is PUTTING a confirmation up, before any
    confirm_action has recorded it.

    Streamlit files a button's value in session state under its own key
    before the button is created, so a click that is about to arm a
    question has already arrived by the time the script starts. Without
    this, everything drawn ABOVE the confirm_action that will arm — a
    grid's `Save changes`, the foot's Continue — was still drawn coloured
    on that one run, beside the coloured Yes below it. confirm_action makes
    the same reading about its own trigger; this is the same reading for
    the rest of the tab.
    """
    if armed_confirmation() is not None:
        return False
    return any(str(key).endswith("__btn") and value is True
               for key, value in st.session_state.items())


def confirmation_open():
    """True while a confirmation is armed — or is being armed on this very
    run. Its "Yes" is the lit button, and a tab never shows two coloured
    buttons at once, so every tab's own primary steps aside while one is on
    screen."""
    return (armed_confirmation() is not None or restore_armed()
            or _confirmation_arming())


def other_confirmation(key):
    """True while a DIFFERENT confirmation is armed. Two armed at once would
    put two coloured Yes buttons on the tab, each keeping its own copy, so
    arming one greys the other's button until it is answered."""
    armed = armed_confirmation()
    return (armed is not None and armed != key) or restore_armed()


def disarm(key):
    """Take down a confirmation whose question has gone off screen, and say
    whether there was one. Call it from the SAME call site that armed it.

    A confirmation is armed by one click and answered on a later run, so the
    control it belongs to has to still be on screen to draw its Cancel. Empty
    the picker it was armed on — or change the row — and ARMED_KEY stays set
    with nothing anywhere to answer it: every coloured button in the app goes
    grey behind a question the user cannot reach. Only the armed key's own
    site may do this; disarming another's would take a Yes off the screen
    while the user was reading it."""
    if armed_confirmation() != key:
        return False
    st.session_state.pop(f"{key}__pending", None)
    st.session_state.pop(ARMED_KEY, None)
    return True


def confirm_action(key, button_label, warning, confirm_label=wording.YES_CONTINUE,
                   disabled=False, preserve=False, primary=False):
    """Two-step confirmation for an irreversible action.

    Renders `button_label`. After it is clicked, shows `warning` above a
    primary confirm button and a Cancel button (a placeholder reserves the
    warning's position above the button row, and is filled only on runs
    where the user has not just confirmed, so the confirming run's element
    tree carries no warning). Returns True only on the run in which the user
    clicks the confirm button; the caller then performs the action, flashes
    a message, and reruns. `key` must be unique per call site. `preserve` is
    for a confirmation whose Cancel reruns ABOVE one of the tab forms — a
    sidebar one, whose rerun never reaches the tabs at all, and tab 2's
    `Generate a different batch`, which is asked before the result grid is
    drawn into its reserved slot. Streamlit discards the session-state entry
    of every widget a run did not create, so without it the Cancel empties
    the sheet the user has already half-recorded.

    `primary` colours the trigger. Every confirmation in the app is reached
    from a grey button — the coloured one is the Yes it leads to — except
    the grids' own `Save changes`, which IS the tab's one lit action while
    there is an edit in hand and happens to have a deleted row in it. Once
    the question is up, the Yes below is the coloured one and the trigger
    goes grey with everything else, so the tab never shows two.
    """
    pending_key = f"{key}__pending"
    btn_key = f"{key}__btn"
    # The click that puts the question up has already arrived by the time
    # the script runs, and Streamlit files it in session state under the
    # button's own key BEFORE the button is drawn. So the trigger knows, at
    # the moment it is drawn, whether the Yes below is about to be the
    # coloured one — without that, the one run that arms a question drew two
    # coloured buttons.
    arming = (bool(st.session_state.get(btn_key))
              and armed_confirmation() in (None, key))
    up_already = (armed_confirmation() == key
                  and st.session_state.get(pending_key))
    lit = primary and not (arming or up_already)
    if st.button(button_label, key=btn_key, disabled=disabled,
                 type="primary" if lit else "secondary"):
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
        if st.button(wording.CANCEL, key=f"{key}__no", use_container_width=True):
            st.session_state[pending_key] = False
            st.session_state.pop(ARMED_KEY, None)
            if preserve:
                # This rerun happens above one of the tab forms; park what
                # they hold so the next run puts it back.
                preserve_tab_forms()
            st.rerun()
    if confirmed:
        st.session_state[pending_key] = False
        st.session_state.pop(ARMED_KEY, None)
        return True
    slot.warning(warning)                  # only on runs where the user has not confirmed
    return False


# The tab forms a rerun would otherwise throw away. Streamlit discards the
# session-state entry of every widget a run did not create, so a control that
# reruns from ABOVE one of these forms empties it: a Cancel in the sidebar
# (which runs before all three tabs) blanked the result grid and the open
# correction, and the `Whole batch` pick on tab 3 blanked the "already made"
# form directly beneath it.
_TAB_FORM_PREFIXES = ("own_", "past_", "correct_")
# The tab-form boxes whose keys fit none of those prefixes: the radio that
# chooses between typing a past formulation in and reading one off a CSV, the
# formulation total the open batch is shown and printed at, and how many
# formulations the next Generate will ask for. Named rather than swept in by
# prefix, because `add_past_formulation` beside the radio is a BUTTON, and a
# button's value cannot be assigned at all.
#
# Losing the formulation total is not a blank box: the batch table and the
# sheets silently go back to as-generated, and the bench weighs out different
# numbers from the ones that were on screen a click ago. Tab 1's own total
# box and the where-the-targets-come-from box are here for the same reason —
# both sit below the sidebar, so every sidebar Cancel ran above them.
_TAB_FORM_KEYS = ("add_past_mode", "scale_total", "how_many",
                  "formulation_total", "targets_source_box")
_GRID_KEY_RE = re.compile(r"^f\d+_")      # f7_Firmness, f7_note, f7_leave_out


def preserve_tab_forms():
    """Park every tab-form box at the value it is holding, so the next run
    puts it back. Call immediately before an st.rerun() raised anywhere above
    one of those forms. Parked, not merely left alone: an assignment made
    before the widget is created is the one way a value reaches the browser
    again."""
    for key in [k for k in st.session_state if isinstance(k, str)]:
        if (key.startswith(_TAB_FORM_PREFIXES) or key in _TAB_FORM_KEYS
                or _GRID_KEY_RE.match(key)):
            park_clear(key, st.session_state[key])


def go_to_tab(label):
    """Move to another tab and rerun. This is the ONLY way the app changes
    tabs, and it is called from eight handlers only: Next: make a batch,
    Back to set up, Save results, Save uploaded results, Start the next
    batch, Change a measurement or an ingredient, Change the total, and
    opening a project. A set-up edit, Generate, a correction or a plain
    rerun must never call it, and neither must a handler whose write
    failed.

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


def scale_error(obj, value):
    """The refusal for a measured value outside its range, or '' when it fits.
    Results are never clamped: a firmness of 12 on a 0-10 range is either a
    typo or a range that is too narrow, and silently storing 10 hides both."""
    if value is None:
        return ""
    low, high = float(obj['min_val']), float(obj['max_val'])
    if low <= float(value) <= high:
        return ""
    return outside_message(obj['name'], value, low, high, obj.get('unit'),
                           wording.YOUR_RANGE, wording.WIDEN_RANGE_HINT)


def bounds_caution(opt, name, value):
    """The line for an amount outside what the project allows, or '' when it
    fits. Built by the same helper that refuses an out-of-range measurement,
    so the two sentences read alike.

    It is a warning, not a refusal, wherever it is shown: an imported amount
    is a fact about work already done, and a formulation of the user's own is
    a formulation they mean to make. Both teach the model more than a blank.
    """
    return opt.bounds_caution(name, value)


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
    return wording.saved_line(when)


def clear_selection(key):
    """Ask for a select box to be emptied on the NEXT run.

    Popping a widget's key does not reach the browser: it keeps the old value
    and sends it back with the next click, so a row closed by Close reopened
    under the user's finger and swallowed that click. Assigning None instead is
    refused once the widget exists on this run. So the request is parked here
    and honoured by take_clear() just before the widget is created."""
    st.session_state[f"_clear_{key}"] = True


def park_clear(key, value):
    """Same request, for a box that empties to something other than None: a
    text box to "", a tick box to False, a number box to the value it opens
    with. This is what makes a project switch really empty the set-up and
    result forms in the browser — popping the key alone leaves the mounted
    widget to post its old value straight back."""
    st.session_state[f"_clear_{key}"] = ("value", value)


# The round screen's Batch size box. Named here rather than on tab 2,
# because tab 1 reads it too — a grid Save has to know whether the round it
# may be about to retire was being made to a size — and two screens reaching
# for one box by two spellings is a box neither of them owns.
BATCH_SIZE_KEY = "scale_total"


def typed_batch_size(opt):
    """What the Batch size box holds, or None while it is empty.

    Always None while the ingredients are not all in one unit: the box is
    not offered then, and a value left behind in it must not go on quietly
    meaning something on a screen nobody can see it acting on. A size of
    nothing is not a size either, so anything at or below zero reads as
    empty.
    """
    if opt.one_amount_unit() is None:
        return None
    value = st.session_state.get(BATCH_SIZE_KEY)
    if value is None or float(value) <= 0:
        return None
    return float(value)


def clear_scale_total():
    """Empty the round screen's own `Batch size` box for the next round.

    Parked, not popped. Popping a widget's key does not reach the browser —
    the mounted box posts its old value straight back — so a regenerated
    round came up re-sized to the size the round before it was made to, and
    the sheets were printed for it. The parked value lands before the box is
    drawn again (drain_clears).

    Call it AFTER preserve_tab_forms() wherever both are used: that parks
    every tab form at what it is still holding, which would put the old size
    back.
    """
    park_clear("scale_total", None)


def clear_formulation_total_box():
    """Empty Set up's `Default batch size` box.

    Parked for the same reason the round screen's is: popping a widget's key
    does not reach the browser, so a box left holding a size the project no
    longer has would write it straight back on the next run — and the notice
    saying the default went would be followed by the default coming back."""
    park_clear("formulation_total", None)


def take_clear(key, fresh=None):
    """Honour a pending clear. Call immediately BEFORE the widget is created.
    `fresh` is what a plain clear_selection should leave behind when the right
    empty value is only known here (the amount-unit box holds the newly opened
    project's unit)."""
    parked = st.session_state.pop(f"_clear_{key}", None)
    if parked is None:
        return
    if isinstance(parked, tuple) and parked and parked[0] == "value":
        st.session_state[key] = parked[1]
    else:
        st.session_state[key] = fresh


def drain_clears(fresh=None):
    """Honour every parked clear, at the one moment in a run when it is legal:
    the previous run's widgets are gone and this run's have not been created
    yet. app.py calls this once, just before the tabs render, which is what
    makes a project switch really empty the forms in the browser.

    `fresh` names the value for a plain clear_selection where only the caller
    knows it (the amount-unit box holds the newly opened project's unit).
    """
    fresh = fresh or {}
    for key in [k for k in st.session_state
                if isinstance(k, str) and k.startswith("_clear_")]:
        name = key[len("_clear_"):]
        take_clear(name, fresh.get(name))


def best_formulation_no(opt):
    """The number of the best formulation, or None. Three tabs name it —
    `Start from the best so far`, the biggest-changes line and every 'Best
    moved' sentence — and they must all mean the same formulation."""
    return opt.best_formulation_no()


def best_move_sentence(before, after):
    """'Best moved from Formulation 3 to Formulation 7.' or '' when it stayed."""
    if before is None or after is None or before == after:
        return ""
    return wording.best_moved(before, after)


def readiness(opt):
    """(ready, the missing item named) for the foot button on tab 1 and the
    Generate button on tab 2. Ready means something to vary — an ingredient
    OR a process setting — and at least one measurement.

    A fermentation project varies incubation temperature, time and culture
    dose and weighs nothing out; holding it incomplete until it listed an
    ingredient asked it to invent one."""
    if not opt.variables:
        return False, wording.NEED_A_VARIABLE
    if not opt.objectives:
        return False, wording.NEED_A_MEASUREMENT
    return True, ""


def landing_tab(opt):
    """Where opening a project lands: set-up while it is incomplete, the batch
    while one is unrecorded, set-up again while nothing has been made, and
    results once the project holds some.

    A project with no formulations sent the user to an empty Results tab, and
    the sample project skipped its own set-up entirely. Confirming the set-up
    is the step before making anything, so that is where those land. A
    not-scored formulation counts as one the project holds: it has a
    number, its amounts and a note, and tab 3 lists it."""
    ready, _ = readiness(opt)
    if not ready:
        return TAB_SETUP
    if opt.pending_batch:
        return TAB_BATCH
    if not opt.X_history and not opt.skipped:
        return TAB_SETUP
    return TAB_RESULTS


def _grid_nonce(name):
    return f"_{name}_nonce"


def grid_key(name):
    """The key tab 1's editable grid `name` is mounted under on this run.

    It carries a counter, and that is not decoration. A data editor's value
    cannot be assigned from session state AT ALL — Streamlit refuses the
    write outright, so the park-and-assign discipline every other box on the
    page uses is not available here — and its session-state entry is a
    record of what was typed (this cell changed, this row was added), not a
    frame. Left standing after a save, that record would add the new row a
    second time on the next run. So the only way to throw it away is to draw
    a NEW editor: turning the counter over does that, and Streamlit drops
    the state of the widget the run did not create.
    """
    return f"{name}_{st.session_state.get(_grid_nonce(name), 0)}"


def clear_grid(name):
    """Throw away what has been typed into one editable grid, by turning its
    key over and forgetting the frame parked for it. Called by Discard
    changes and by every Save that lands."""
    rekey_grid(name)
    unpark_grid(name)


def rekey_grid(name):
    """Turn one grid's key over WITHOUT forgetting what it was holding.

    For the grid the run is about to rerun past without drawing: its own
    record is gone either way (Streamlit keeps no widget a run did not
    create), and the frame parked for it is what it opens at next. The key
    has to move for that frame to be read — see parked_grid.
    """
    st.session_state[_grid_nonce(name)] = (
        st.session_state.get(_grid_nonce(name), 0) + 1)


# ------------------------------------------------------------------ #
#  Tab 1's three editable grids, named once
#
#  The keys live here rather than in ui_setup because reset_grids() below
#  is what every door that replaces a project's variable list calls, and
#  app.py reaches it through this module. ui_setup imports them back.
# ------------------------------------------------------------------ #
ING_GRID_KEY = "ingredient_grid"
MEAS_GRID_KEY = "measurement_grid"
PROP_GRID_KEY = "property_grid"
GRID_KEYS = (ING_GRID_KEY, MEAS_GRID_KEY, PROP_GRID_KEY)

# The Save/Discard pair each of the two full grids is drawn under. The
# properties grid has a plain button of its own and no deletion question.
ING_SAVE_KEY = "save_ingredient_grid"
MEAS_SAVE_KEY = "save_measurement_grid"
SAVE_KEYS = (ING_SAVE_KEY, MEAS_SAVE_KEY)

# Where a Save that refused parks its per-row errors until the slot under
# the grid draws them. Listed as a tuple so a fourth grid cannot be
# forgotten the way _property_grid_errors was.
ING_ERRORS_KEY = "_ingredient_grid_errors"
MEAS_ERRORS_KEY = "_measurement_grid_errors"
PROP_ERRORS_KEY = "_property_grid_errors"
GRID_ERROR_KEYS = (ING_ERRORS_KEY, MEAS_ERRORS_KEY, PROP_ERRORS_KEY)

# Which grid already has an edit in hand, read by the grid below it so that
# only one Save is coloured.
ING_PENDING_KEY = "_ingredient_grid_pending"

# What set of rows one grid's deletion question was armed over. ONE KEY PER
# GRID: shared, the second grid's bookkeeping ran after the first's on every
# run and popped the record out from under it.
_ARMED_DELETIONS = "_grid_deletions_armed"


def armed_deletions_key(key):
    return f"{key}__{_ARMED_DELETIONS}"


def parked_grid_key(name):
    """Where one grid's edited frame waits while a rerun it did not cause
    goes past it."""
    return f"_{name}_parked"


def park_grid(name, frame, keep_mark=False):
    """Keep one grid's edited frame, and the key it was typed under, so a
    rerun the grid was never drawn on does not throw the edit away.

    A Save on the grid ABOVE reruns before this one is drawn, and Streamlit
    throws away the session-state entry of every widget a run did not
    create — so without this the other grid's edit is gone and its banner
    with it, and the tab reported as saved work that never reached the
    project.

    The nonce is parked with the frame because the editor's own record is
    POSITIONAL, and a frame already holding an added row, drawn again under
    the same key that record belongs to, would add it a second time. So the
    frame is only ever read back after the key has turned over — which is
    exactly when the record is gone.
    """
    parked = st.session_state.get(parked_grid_key(name))
    # `keep_mark` is the run that drew the grid FROM the park: the frame is
    # still the only record of the edit, so it keeps the mark that says so.
    # Re-stamping it with the key it has just been drawn under would have
    # made it unreadable on the very next run, and the edit would have
    # lasted exactly one screen.
    nonce = (parked[0] if keep_mark and parked is not None
             else st.session_state.get(_grid_nonce(name), 0))
    st.session_state[parked_grid_key(name)] = (nonce, frame)


def parked_grid(name, saved):
    """The parked frame for `name` when it is the only record of the edit,
    else None.

    Two questions. Has the key turned over since it was parked? Until it
    has, the editor's own record is still live and is what carries the
    edit. And does the frame still fit the project — the same columns? A
    save elsewhere can add or drop one (the Baseline column arrives with
    the first recorded formulation), and an edit typed against the old
    shape is not one this grid can still draw.
    """
    parked = st.session_state.get(parked_grid_key(name))
    if parked is None:
        return None
    nonce, frame = parked
    if nonce >= st.session_state.get(_grid_nonce(name), 0):
        return None
    if list(frame.columns) != list(saved.columns):
        st.session_state.pop(parked_grid_key(name), None)
        return None
    return frame


def unpark_grid(name):
    st.session_state.pop(parked_grid_key(name), None)


def reset_grids():
    """Throw away every pending edit on tab 1 and every question armed over
    one.

    Called by each door that replaces the project's variable list — a
    project switch, Start this project over, Open a saved copy → Yes,
    replace, and an ingredients file that replaces the list. A data
    editor's session-state entry is POSITIONAL (row 0's Lowest changed),
    not a record of which ingredient it was typed against, so a record left
    standing writes one project's number onto whatever now sits in that
    row. Turning every key over is the only way to drop it.
    """
    for name in GRID_KEYS:
        clear_grid(name)
        unpark_grid(name)
    for key in SAVE_KEYS:
        st.session_state.pop(armed_deletions_key(key), None)
    for key in GRID_ERROR_KEYS:
        st.session_state.pop(key, None)
    st.session_state.pop(ING_PENDING_KEY, None)
