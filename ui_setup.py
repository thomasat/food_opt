"""Tab 1 · Set up: what the project can vary and what will be measured.

Three tiers, top to bottom (spec 1.5), and nothing else on the tab:

  1. the ingredients grid, with its Save/Discard while there is an edit in
     hand, its per-row errors and its collapsed `Or upload an ingredients
     file` alternative;
  2. the measurements grid, the same way;
  3. one collapsed `More settings` — Default batch size, where the targets
     came from, Limits, and the properties those limits read — and one
     collapsed `Advanced` at the very foot, holding the model settings and
     the two explanations.

Neither tier nests an expander inside itself: Streamlit forbids it, and a
tab that folded the optional half away twice made the reader open two
things to reach one. Exactly one coloured button lives here — `Next: make a
round` at the foot, or a grid's `Save changes`, or the `Yes` of a question
that is up.
"""
import json
import os

import pandas as pd
import streamlit as st

import storage as storage_backend
import wording
from food_bo import (
    GRID_ID, WORKBOOK_MIME, grid_signature, ingredients_template_workbook,
)
from ui_helpers import (
    COPY_KEPT, GRID_KEYS, ING_ERRORS_KEY, ING_GRID_KEY, ING_PENDING_KEY,
    ING_SAVE_KEY, MEAS_ERRORS_KEY, MEAS_GRID_KEY, MEAS_SAVE_KEY,
    PROP_ERRORS_KEY, PROP_GRID_KEY, TAB_BATCH,
    armed_confirmation, armed_deletions_key, best_formulation_no,
    best_move_sentence, clear_formulation_total_box, clear_grid, grid_key,
    clear_scale_total, confirm_action, confirmation_open,
    disarm, flash,
    go_to_tab, number_list, other_confirmation, park_clear, park_grid,
    parked_grid, plural,
    parked_grid_key, preserve_tab_forms, readiness, rekey_grid, reset_grids,
    saved_ok, table_height, typed_batch_size, unpark_grid,
)

_SAMPLE_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "sample_ingredients.csv")

# The one collapsed expander that maps the words on this tab to the words a
# specialist would use, and the tuple marking its three nested goal lines,
# both live in wording.py; importable from here too, since HOW_IT_WORKS and
# HOW_CLOSENESS are read by name from tests/test_food_bo.py's vocabulary
# guard — they are the two folds allowed to say the specialist words.
from wording import HOW_CLOSENESS, HOW_IT_WORKS  # noqa: E402,F401


# ------------------------------------------------------------------ #
#  Small shared text
# ------------------------------------------------------------------ #

def _unit_suffix(unit):
    return f" ({unit})" if unit else ""


# Where the notice about a round this tab has just retired waits. A toast
# fades, and the round it was about does not come back: the same sentence
# stays under the grid that took it away until a new round is made.
_ROUND_DISCARDED = "_round_discarded_line"


def _note_discarded_batch(opt, batch_no_before,
                          reason=wording.SETUP_CHANGED_REASON):
    """Say, once above the tabs and then for as long as it is true, that the
    write just now retired the open batch. The batch is named: the flash
    lands above the tabs, away from the table it is about, and the standing
    line is under the grid that did it.

    `reason` is what discarded it. The tab as a whole is the honest answer
    for the ingredient list and the allowed amounts — two ways to
    one place — but the total is one control the reader has just touched,
    and blaming "your set-up" sent them looking for what else they had
    done."""
    if _remember_discarded_round(opt, batch_no_before, reason):
        flash("info", st.session_state[_ROUND_DISCARDED])


def _remember_discarded_round(opt, batch_no_before,
                              reason=wording.SETUP_CHANGED_REASON):
    """Keep the notice for the line that stands under the grid, and say
    whether there was one.

    The grid's own save gets the flash from the model — it is the model that
    knows the round went — so this is the half the screen owes either way.
    """
    if batch_no_before is None or opt.pending_batch_no is not None:
        return False
    st.session_state[_ROUND_DISCARDED] = wording.batch_discarded_notice(
        batch_no_before, reason)
    return True


def _discarded_round_line(opt):
    """The standing notice under the grid, while it is still true. It goes
    the moment there is a round again: nothing was lost that the reader
    cannot now see."""
    if opt.pending_batch_no is not None:
        st.session_state.pop(_ROUND_DISCARDED, None)
        return
    line = st.session_state.get(_ROUND_DISCARDED)
    if line:
        st.info(line)


def _nothing_made_fits(opt):
    """True when a limit just added excludes every formulation already made."""
    if not opt.recipe_history:
        return False
    return not any(opt._check_constraints(r) for r in opt.recipe_history)


def _report_limit(opt, sentence):
    if not saved_ok(opt):
        return
    flash("success", f"{sentence} {wording.LIMIT_KEPT}")
    if _nothing_made_fits(opt):
        flash("warning", wording.NO_FORMULATION_FITS_LIMIT)
    st.rerun()


# ------------------------------------------------------------------ #
#  Sections
# ------------------------------------------------------------------ #

KIND_INGREDIENT = wording.KIND_INGREDIENT
KIND_SETTING = wording.KIND_SETTING


def _unscaled_tail(opt, before):
    """The sentence a unit change owes the open round when it has just split
    the ingredients across units: a batch size needs one unit, so there is no
    longer a box to change it in. Nothing is undone — since 0.5.0 the size
    moves the amounts themselves — so the sentence says the round keeps what
    it has. Empties the box too: a number left in it would go on quietly
    meaning nothing.

    `before` is only asked whether there was a size at all; the number itself
    is not named, because it is not one anything can be typed back to."""
    if before is None or opt.one_amount_unit() is not None:
        return ""
    clear_scale_total()
    if opt.pending_batch_no is None:
        return ""
    return wording.unscaled_tail(opt.pending_batch_no)


# The (project, stored total) tab 1's total box has already been opened for.
# After that an empty box means the user emptied it, which is an answer of its
# own and must reach the file.
_SEEDED_FORMULATION_TOTAL = "_formulation_total_seeded"
# The number the box was last put on screen holding. A widget's own key
# cannot answer this — it holds whatever the browser posted back, which is
# the same thing whether the user typed it or the app seeded it — so the one
# value a returned number can be compared against is tracked here.
_SHOWN_FORMULATION_TOTAL = "_formulation_total_shown"


def _formulation_total_mark(opt, total):
    """What the box is up to date with: whose project it belongs to AND the
    number stored on it. The project alone is not enough — the stored total
    can move under a session that has already drawn the box (a saved copy
    opened, a unit change that cleared it, the sample opened twice) and the
    seed would be skipped every time."""
    return (opt.project_name, total)


def _seed_formulation_total(opt):
    """Open the box at the total the project is stored with.

    The same debt tab 2's `_seed_scale_total` settles, for the same reason: a
    session that did not type the number draws an empty box, and the empty
    box then writes its own blank over the saved total on the very first
    render — here that would take the limit every suggestion is held to with
    it. Assigning a widget's key is legal only before the widget exists,
    which is why this runs first."""
    stored = getattr(opt, 'formulation_total', None)
    mark = _formulation_total_mark(opt, stored)
    # ... and re-seeded whenever the key has gone, whatever the mark says.
    # Streamlit discards the session-state entry of every widget a run did
    # not create, so any rerun raised ABOVE this box (a Cancel in the
    # sidebar, which runs before all three tabs) leaves the mark stamped and
    # the number gone. The box then opened empty over a project that has a
    # total — and an empty box used to mean "the user cleared it".
    if (st.session_state.get(_SEEDED_FORMULATION_TOTAL) == mark
            and "formulation_total" in st.session_state):
        return
    st.session_state[_SEEDED_FORMULATION_TOTAL] = mark
    st.session_state[_SHOWN_FORMULATION_TOTAL] = (
        None if stored is None else float(stored))
    if stored is not None:
        st.session_state["formulation_total"] = float(stored)


def _store_formulation_total(opt, typed, shown):
    """Keep what the box holds with the project, on a change only: this runs
    on every rerun, and a save per rerun would bump the file's mtime and make
    another open window see a false conflict.

    `shown` is the value the box was DRAWN with on this run. The write
    happens only when the two differ — that is, only when the user themselves
    moved the number — never because the box came up holding something else.
    Comparing against the stored total instead read a widget Streamlit had
    just thrown away as an answer of the user's: a Cancel in the sidebar, or
    an ingredient added, and the 100 g every suggestion was held to was gone
    with no message anywhere.

    A refusal is left on screen with the number still in the box — the mark
    is not re-stamped, so the next run does not quietly put the old total
    back over what the user is still typing."""
    stored = getattr(opt, 'formulation_total', None)
    if typed == shown:
        return
    if typed is None and stored is None:
        return
    try:
        if typed is None:
            opt.clear_formulation_total()
        else:
            opt.set_formulation_total(typed)
    except ValueError as e:
        st.error(str(e))
        return
    st.session_state[_SEEDED_FORMULATION_TOTAL] = _formulation_total_mark(
        opt, getattr(opt, 'formulation_total', None))
    st.session_state[_SHOWN_FORMULATION_TOTAL] = typed


def _formulation_total(opt):
    """The one box that says how big a formulation is.

    It sits under the ingredients table because it is a fact about the
    ingredients, not an optional rule: it writes the limit over every one of
    them that both the space-filling opening and the model obey, and the
    Limits list below shows what it wrote.

    Two projects are offered nothing: one that weighs nothing out, which has
    no total to build to, and one whose ingredients are in different units —
    a sum of 10 g and 40 ml is not a total of anything, and the line that
    says so is the one tab 2 has always shown."""
    if not opt.has_ingredients():
        return
    if opt.one_amount_unit() is None:
        st.caption(wording.NEEDS_ONE_UNIT)
        return
    _seed_formulation_total(opt)
    st.session_state.setdefault("formulation_total", None)
    # What the box was last put on screen holding. Not its own key: that
    # holds whatever the browser posted back, which reads the same whether
    # the user typed it or the app seeded it a moment ago.
    shown = st.session_state.get(_SHOWN_FORMULATION_TOTAL)
    shown = None if not shown else float(shown)
    batch_no = opt.pending_batch_no
    typed = st.number_input(
        wording.formulation_total_label(opt.one_amount_unit()),
        min_value=0.0, step=1.0,
        placeholder=wording.FORMULATION_TOTAL_PLACEHOLDER,
        key="formulation_total",
        help=wording.FORMULATION_TOTAL_HELP,
    )
    _store_formulation_total(opt, None if not typed else float(typed), shown)
    # The open batch was built to the old answer, so a changed total retires
    # it like every other set-up change — with the same notice. The notice
    # lands above the tabs on the NEXT run, so this one is ended here:
    # otherwise the batch vanished from tab 2 with nothing said until the
    # user's next click.
    if batch_no is not None and opt.pending_batch_no is None:
        _note_discarded_batch(opt, batch_no, wording.TOTAL_CHANGED_REASON)
        preserve_tab_forms()
        st.rerun()


# ------------------------------------------------------------------ #
#  The two editable grids
#
#  Nothing is written while the reader types. A grid that differs from the
#  saved project lights its own `Save changes`, with `Discard changes`
#  beside it, and the foot's Continue steps aside — the same rule the
#  per-row editors used to follow, now for the tab as a whole.
#
#  The work of turning an edited frame into writes is in food_bo
#  (apply_ingredient_grid / apply_measurement_grid): it is arithmetic over
#  the model's own objects, and AppTest cannot click a cell, so that is the
#  layer it can be tested at. What is here is the drawing, the one lit
#  button, the confirmation a deleted row owes, and the per-row errors.
# ------------------------------------------------------------------ #

# The grid keys, the key each grid's Save/Discard pair — and its deletion
# question — is drawn under, and where a Save that refused leaves its
# per-row errors for the slot under the grid to show on this same run: all
# named once, in ui_helpers, beside the reset_grids() that has to reach
# every one of them. Re-exported here because this module is where they are
# read.
# The picker the Delete beside the properties grid is armed from.
_PROP_DELETE = "prop_delete"
_ING_ERRORS = ING_ERRORS_KEY
_MEAS_ERRORS = MEAS_ERRORS_KEY
_PROP_ERRORS = PROP_ERRORS_KEY


def _number_column(label, help=None):
    """Every number on these grids is an amount, and an amount is written to
    two decimal places everywhere else in the app."""
    return st.column_config.NumberColumn(label, format="%.2f", help=help)


def _ingredient_columns(opt, frame):
    """The typed columns. Every label is read off wording — a bare string
    here would be a screen label the vocabulary guard never sees."""
    columns = {
        GRID_ID: None,                    # the hidden row identity
        wording.NAME_LABEL: st.column_config.TextColumn(
            wording.NAME_LABEL, required=True),
        wording.TYPE_LABEL: st.column_config.SelectboxColumn(
            wording.TYPE_LABEL, options=[KIND_INGREDIENT, KIND_SETTING],
            default=KIND_INGREDIENT, required=True,
            help=wording.VARIABLE_TYPE_HELP),
        wording.LOWEST_LABEL: _number_column(wording.LOWEST_LABEL),
        wording.HIGHEST_LABEL: _number_column(wording.HIGHEST_LABEL),
        wording.UNIT_LABEL: st.column_config.TextColumn(
            wording.UNIT_LABEL, default=opt.amount_unit or "g"),
        wording.VENDOR_LABEL: st.column_config.TextColumn(
            wording.VENDOR_LABEL, help=wording.VENDOR_HELP),
        wording.SKU_LABEL: st.column_config.TextColumn(
            wording.SKU_LABEL, help=wording.SKU_HELP),
    }
    if wording.BASELINE_LABEL in frame.columns:
        columns[wording.BASELINE_LABEL] = _number_column(
            wording.BASELINE_LABEL, help=wording.BASELINE_HELP)
    return columns


def _measurement_columns():
    return {
        GRID_ID: None,
        wording.MEASUREMENT_COLUMN: st.column_config.TextColumn(
            wording.MEASUREMENT_COLUMN, required=True),
        wording.GOAL_LABEL: st.column_config.SelectboxColumn(
            wording.GOAL_LABEL, options=list(wording.GOAL_LABELS.values()),
            default=wording.GOAL_LABELS['max'], required=True),
        wording.TARGET_LABEL: _number_column(wording.TARGET_LABEL),
        wording.LOWEST_MEASURABLE_LABEL: _number_column(
            wording.LOWEST_MEASURABLE_LABEL),
        wording.HIGHEST_MEASURABLE_LABEL: _number_column(
            wording.HIGHEST_MEASURABLE_LABEL),
        wording.UNIT_LABEL: st.column_config.TextColumn(wording.UNIT_LABEL),
        wording.SHARE_COLUMN: st.column_config.NumberColumn(
            wording.SHARE_COLUMN, min_value=0.0, max_value=100.0,
            format="%.2f", help=wording.SHARE_HELP),
    }


def _pending(saved, edited):
    """True while the grid differs from the project it was opened from.

    Read off the two frames rather than off the editor's own record of what
    was typed: a cell typed back to what it already held is not a change,
    and a grid that lit Save over one would ask the reader to save nothing.
    """
    return grid_signature(saved) != grid_signature(edited)


def _opening_frame(grid, saved):
    """(what one grid opens holding, whether that came from the park).

    The project, or the edit it was still holding when a rerun went past it
    without drawing it. Streamlit throws away the session-state entry of
    every widget a run did not create, so the Save on the grid ABOVE took
    this one's edit with it — the banner came down, the Save went, and the
    values on screen were the project's again while the reader believed
    they had been saved."""
    parked = parked_grid(grid, saved)
    return (saved, False) if parked is None else (parked, True)


def _grid_is_pending(grid):
    """True while `grid` has an edit parked — which is what a pending grid
    keeps on every run it is drawn."""
    return parked_grid_key(grid) in st.session_state


def _keep_pending(grid, pending, edited, from_park):
    """Park what this grid is holding, or forget it once there is nothing to
    hold. Called on every run: the run that loses the grid is the run that
    never reaches it."""
    if pending:
        park_grid(grid, edited, keep_mark=from_park)
    else:
        unpark_grid(grid)


def _grid_errors(slot, key):
    """The refusals a Save left behind, one line per row, under the grid
    they belong to. The row number is the one the grid shows down its left
    edge; a refusal about the grid as a whole has no number and says its
    sentence plainly."""
    errors = st.session_state.pop(key, None)
    if not errors:
        return
    with slot.container():
        for row, message in errors:
            st.error(message if row is None else wording.row_error(row, message))


def _save_and_discard(key, grid, lit):
    """`Save changes` and `Discard changes`, side by side. Save is the tab's
    one coloured button while it is the topmost grid with something to save;
    a second grid further down keeps its own pair, grey, so that the tab
    never shows two coloured buttons at once. Both grey while a confirmation
    is on screen: answering that is the one thing to do.

    `key` names the pair of buttons; `grid` is the editor they belong to,
    which is what Discard turns over. They are two different keys, and
    turning over the button's own would have left the edits standing under a
    Discard that appeared to do nothing.
    """
    b1, b2 = st.columns(2)
    with b1:
        save = st.button(wording.SAVE_CHANGES_BUTTON, key=f"{key}__save",
                         type="primary" if lit else "secondary",
                         disabled=confirmation_open(),
                         use_container_width=True)
    with b2:
        if st.button(wording.DISCARD_CHANGES_BUTTON, key=f"{key}__discard",
                     use_container_width=True):
            _discard_grid(grid)
    return save


def _discard_grid(grid):
    """Throw one grid's edit away and rerun. The OTHER grid keeps its own:
    its key turns over so that the frame it parked is what it opens at,
    because this rerun may never reach it to read its record."""
    clear_grid(grid)
    for other in GRID_KEYS:
        if other != grid:
            rekey_grid(other)
    st.rerun()


def _variables(opt, storage):
    """Ingredients and process settings, in one editable grid.

    It replaces an add form, a table, a row of five controls and a per-row
    editor — six places asking about one list. A row is typed where it is
    read; a new one goes on the empty line at the bottom; a row taken out is
    a deletion, asked about by name before Save applies it.

    Returns True while there is an edit in hand: `Save changes` is then the
    tab's one lit action and the foot steps aside, exactly as the per-row
    editors used to make it.
    """
    st.subheader(wording.VARIABLES_HEADER)
    st.caption(wording.INGREDIENT_GRID_CAPTION)
    saved = opt.ingredient_grid_frame()
    opening, from_park = _opening_frame(ING_GRID_KEY, saved)
    edited = st.data_editor(
        opening, key=grid_key(ING_GRID_KEY),
        num_rows="dynamic",
        column_config=_ingredient_columns(opt, saved),
        use_container_width=True,
        height=table_height(max(len(saved) + 1, 2), max_rows=20))
    slot = st.empty()            # where a refused Save writes its rows
    _grid_errors(slot, _ING_ERRORS)
    pending = _pending(saved, edited)
    _keep_pending(ING_GRID_KEY, pending, edited, from_park)
    # Read by the grid below, which keeps its own pair of buttons grey while
    # this one has something to save: one coloured button per tab.
    st.session_state[ING_PENDING_KEY] = pending
    if pending:
        st.caption(wording.unsaved_grid_caption(wording.VARIABLES_HEADER))
        _save_ingredients(opt, storage, edited)
    else:
        # Nothing to save, so nothing to ask about. A question armed over a
        # deletion that has since been discarded or undone is off screen
        # with no Yes to reach, and every coloured button in the app stays
        # grey behind it.
        _disarm_grid_deletion(ING_SAVE_KEY)
    _discarded_round_line(opt)
    if getattr(opt, "amount_unit_backfilled", False):
        # The file this project was saved in predates the unit; its amounts
        # may have been percentages or millilitres, and nothing on screen
        # would otherwise say the g was the app's guess and not the user's.
        st.caption(wording.made_before_units_caption(opt.amount_unit))
    with st.expander(wording.UPLOAD_INGREDIENTS_EXPANDER):
        _upload_ingredients(opt)
    return pending


def _round_at_risk(opt, edited, force=()):
    """The sentence a Save owes the open round it is about to take away, or
    "" when the round survives this edit or there is no round.

    Asked BEFORE the write, because a toast afterwards was the first the
    reader heard of it — and a round can hold formulations they typed in by
    hand, which nothing can generate back."""
    no = opt.ingredient_grid_retires_round(edited, force=force)
    if no is None:
        return ""
    rows = opt.pending_batch or []
    own = sum(1 for row in rows if row.get('note'))
    return wording.saving_discards_round(
        no, plural(len(rows), wording.FORMULATION), own)


def _save_ingredients(opt, storage, edited):
    """The one write the ingredients grid makes. A deleted row is confirmed
    by name first, an open round the save would take away is named in the
    same question, and a copy is kept before anything goes."""
    deletions = opt.ingredient_grid_deletions(edited)
    key = ING_SAVE_KEY
    # Before the colour is read and before the early return: a question that
    # is no longer this grid's question has to come down first, or `lit`
    # reads a confirmation that is about to be taken down anyway.
    _disarm_stale_deletion(key, deletions)
    lit = not confirmation_open()
    # Read here rather than off the tick box below, which is drawn after the
    # question and only while one is up.
    forced = set(deletions) if st.session_state.get("delete_ing_force") else ()
    at_risk = _round_at_risk(opt, edited, force=forced)
    if not deletions:
        _remember_armed_deletions(key, deletions)
        if not at_risk:
            if _save_and_discard(key, ING_GRID_KEY, lit):
                _apply_ingredient_grid(opt, edited)
            return
        # No row is going, but the open round is: the same two-step every
        # other irreversible action on this tab goes through.
        if confirm_action(key, wording.SAVE_CHANGES_BUTTON, at_risk,
                          confirm_label=wording.YES_SAVE_AND_DISCARD,
                          primary=lit, disabled=other_confirmation(key)):
            _apply_ingredient_grid(opt, edited)
        else:
            _discard_beside(key, ING_GRID_KEY)
        return
    confirmed = confirm_action(
        key, wording.SAVE_CHANGES_BUTTON,
        " ".join(p for p in (wording.delete_rows_warning(
            number_list(deletions)), at_risk) if p),
        confirm_label=wording.YES_DELETE, primary=lit,
        disabled=other_confirmation(key))
    _remember_armed_deletions(key, deletions)
    # Read here, before the tick box below is drawn: the run that confirms
    # is the run that disarms.
    force = bool(st.session_state.get("delete_ing_force", False))
    armed = (armed_confirmation() == key
             and st.session_state.get(f"{key}__pending"))
    # Drawn AFTER confirm_action, because the click that arms the question is
    # only recorded inside it. Rarely needed, and only ever seen while a
    # deletion is armed: deleting an ingredient that was used above 0 would
    # rewrite formulations nobody made.
    if armed:
        st.caption(wording.DELETE_VS_FIXING_CAPTION)
        st.checkbox(wording.DELETE_EVEN_IF_USED_CHECKBOX,
                    key="delete_ing_force")
    else:
        # Never carried into the next deletion, or the next project: a tick
        # left behind is one click away from discarding real results.
        st.session_state.pop("delete_ing_force", None)
    if not confirmed:
        _discard_beside(key, ING_GRID_KEY)
        return
    try:
        storage.archive(opt.project_name, "pre_delete", copy=True)
    except storage_backend.StorageError as e:
        st.error(str(e))
        return
    _apply_ingredient_grid(opt, edited,
                           force=set(deletions) if force else ())


def _discard_beside(key, grid):
    """Discard on its own, under a Save that confirm_action has drawn: that
    helper owns its own button row, so the pair cannot sit side by side while
    a deletion is waiting to be confirmed.

    Discarding while the question is up is the reader answering it with
    "none of it": the question goes down with the edit that raised it. Left
    armed, it greyed every coloured button in the app — on every tab — with
    no Yes anywhere to reach."""
    if st.button(wording.DISCARD_CHANGES_BUTTON, key=f"{key}__discard"):
        _disarm_grid_deletion(key)
        _discard_grid(grid)


# What set of rows one grid's deletion question was armed over — one key per
# grid, named in ui_helpers beside the reset that has to pop both.
_armed_deletions_key = armed_deletions_key


def _disarm_grid_deletion(key):
    """Take one grid's deletion question down, and forget what it was about.

    Called wherever the grid stops asking for a deletion: Discard, the row
    put back, the edit undone. Guarded on `disarm` having found something,
    so this can be called on every run of a tab that has no question up
    without touching a question belonging to somebody else."""
    if disarm(key):
        st.session_state.pop(_armed_deletions_key(key), None)


def _disarm_stale_deletion(key, deletions):
    """An armed Save belongs to the rows it was armed over. Put one back on
    the grid — or take the whole edit away — and the question on screen is
    about a different set of rows, so it is taken down rather than answered:
    a "Delete Water?" left standing over a grid that no longer deletes Water
    is one click from deleting something else.

    Asked BEFORE the early return for a grid with no deletions left, because
    no deletions at all is the commonest way for the question to stop being
    the question that is up."""
    armed = st.session_state.get(_armed_deletions_key(key))
    if armed is not None and armed != sorted(deletions):
        _disarm_grid_deletion(key)


def _remember_armed_deletions(key, deletions):
    """...which means remembering them at the moment of arming, and only
    then: the click that arms is recorded INSIDE confirm_action, so nothing
    before it can see the question go up."""
    if armed_confirmation() == key:
        st.session_state[_armed_deletions_key(key)] = sorted(deletions)
    else:
        st.session_state.pop(_armed_deletions_key(key), None)


def _apply_ingredient_grid(opt, edited, force=()):
    """Hand the finished grid to the model, and say what it did.

    Every consequence arrives as one message from there, because the model
    is what knows whether the default batch size survived, which limits were
    pruned and whether the open round is still open. The one sentence the
    screen owes on its own is the unit-split tail: it is the only one that
    has a box on this tab to empty as well as something to say.
    """
    scaled = typed_batch_size(opt)
    round_before = opt.pending_batch_no
    # The properties grid is drawn ingredient by ingredient, in this order.
    # A deletion, a rename or a reorder moves who sits in a row it may be
    # holding a pending edit for; a changed Highest does not.
    rows_before = opt.ingredient_names()
    errors, messages = opt.apply_ingredient_grid(edited, force=force)
    if errors:
        # st.rerun() does not return: the errors are drawn into the slot
        # under the grid on the run it raises.
        st.session_state[_ING_ERRORS] = errors
        st.rerun()
    if not saved_ok(opt):
        return
    # The model has already put the notice in `messages`; this is the copy
    # that stays under the grid after the flash has gone.
    _remember_discarded_round(opt, round_before)
    tail = _unscaled_tail(opt, scaled)
    if tail and messages:
        kind, line = messages[0]
        messages[0] = (kind, f"{line} {tail}")
    for kind, line in messages:
        flash(kind, line)
    clear_grid(ING_GRID_KEY)
    # This rerun happens ABOVE the measurements grid, so Streamlit is about
    # to throw its record away. Turning its key over — and keeping what it
    # parked — is what lets that frame be drawn in its place: the edit, the
    # banner and the Save all stay where the reader left them.
    rekey_grid(MEAS_GRID_KEY)
    if opt.ingredient_names() == rows_before:
        # The rows the properties grid is drawn from have not moved, so it
        # keeps its own edit exactly as the measurements grid does.
        rekey_grid(PROP_GRID_KEY)
    else:
        # They have. That edit is positional, not a record of which
        # ingredient it was typed against, so it is thrown away rather than
        # replayed onto whoever inherited the row — and the reader is told,
        # because a banner promised it was pending.
        if _grid_is_pending(PROP_GRID_KEY):
            flash("info", wording.PROPERTY_FIGURES_SET_ASIDE)
        clear_grid(PROP_GRID_KEY)
    st.session_state.pop(_armed_deletions_key(ING_SAVE_KEY), None)
    st.rerun()


def _load_label(opt):
    """What the button under an uploaded file does. It does not add to the
    list, it replaces it — every ingredient not in the file is dropped — and
    a button that said Load over a project's own eight ingredients did not
    say so."""
    return (wording.REPLACE_INGREDIENTS_BUTTON if opt.has_ingredients()
            else wording.LOAD_INGREDIENTS_BUTTON)


def _read_ingredients_file(uploaded):
    """An ingredient list in either shape: the template workbook's first
    sheet, or a comma-separated file with the same columns."""
    if str(getattr(uploaded, "name", "")).lower().endswith(".xlsx"):
        return pd.read_excel(uploaded, sheet_name=0)
    return pd.read_csv(uploaded)


def _upload_ingredients(opt):
    """The ingredient file, folded away: typing one ingredient is the common
    case, and a file is the shortcut for a project that already has one."""
    st.caption(wording.INGREDIENTS_FILE_CAPTION)
    uploaded = st.file_uploader(
        wording.UPLOAD_INGREDIENTS_FILE_LABEL, type=["xlsx", "csv"],
        # Keyed to the project: a file uploader cannot be emptied from session
        # state, so a shared key handed the next project the sheet this one
        # loaded, with a live Load ingredients under it.
        key=f"ingredients_file_{opt.project_name}",
        # The caption above lists the columns; the one thing it does not say
        # is what a blank Unit cell means, which is this project's own unit.
        help=wording.blank_unit_cell_help(opt.amount_unit),
    )
    if os.path.exists(_SAMPLE_CSV):
        st.download_button(
            wording.DOWNLOAD_TEMPLATE,
            data=ingredients_template_workbook(_SAMPLE_CSV),
            file_name=wording.INGREDIENTS_TEMPLATE_FILE_NAME,
            mime=WORKBOOK_MIME, key="ingredients_template")
    df = None
    if uploaded is not None:
        try:
            df = _read_ingredients_file(uploaded)
        except Exception:
            st.error(wording.FILE_UNREADABLE_RETRY)
    # The file itself is left alone. Taking it out of the uploader changed
    # the widget's identity, which shut every open expander on the page under
    # the user's hands; remembering which file was loaded stops a second Load
    # just as well.
    mark = None if uploaded is None else (
        getattr(uploaded, "file_id", None) or f"{uploaded.name}:{uploaded.size}")
    if df is not None and mark == st.session_state.get("_ingredients_loaded"):
        st.caption(wording.FILE_ALREADY_LOADED_CAPTION)
    elif df is not None:
        st.dataframe(df, hide_index=True, height=table_height(len(df)))
        if st.button(_load_label(opt), key="load_ingredients"):
            batch_no = opt.pending_batch_no
            try:
                removed = opt.load_ingredients_from_csv(df)
            except ValueError as e:
                st.error(str(e))
            else:
                if saved_ok(opt):
                    st.session_state["_ingredients_loaded"] = mark
                    flash("success", wording.loaded(plural(len(df), wording.INGREDIENT)))
                    # The new file can rename a unit or drop an ingredient a
                    # limit was written against.
                    _flash_removed_limits(opt, removed)
                    _note_discarded_batch(opt, batch_no)
                    # The file replaces the list outright, so every row on
                    # this tab is a different row now and no pending edit
                    # typed against the old one can be replayed onto it.
                    reset_grids()
                    st.rerun()


def _flash_removed_limits(opt, removed):
    """Name every ingredient limit an edit just emptied of meaning, one line
    each. Four doors reach it — a unit set on one ingredient, a new default
    unit, a reloaded ingredient file and a grid Save — so the sentences
    themselves live on the model and all four say them alike. What is left
    here is the one thing only a screen can do: empty the box the number the
    project has just lost is still sitting in."""
    for kind, line in opt.limit_removed_messages(removed):
        flash(kind, line)
    if any(qc.get('source') == 'formulation_total' for qc in removed or []):
        clear_formulation_total_box()


def _limit_who(opt, qc):
    """How one ingredient limit is named inside a sentence — 'the total of
    each formulation', 'all ingredients' or 'Water + Oil'. The same list
    _limit_label heads the row with, in the register a sentence needs."""
    if qc.get('source') == 'formulation_total':
        return wording.FORMULATION_TOTAL_LOWER
    if _limit_label(opt, qc) == wording.ALL_INGREDIENTS_LABEL:
        return wording.ALL_INGREDIENTS_LOWER
    return " + ".join(qc['ingredients'])


def _delete_limit(opt, storage, key, who, remove):
    """One limit's Delete: named, confirmed and copied first, exactly as
    every other Delete in the app is.

    It was a bare `Delete limit`, drawn once per limit with nothing on it to
    say which, and it removed the limit on the click — the only destructive
    button in the app with no question and no copy behind it."""
    if not confirm_action(key, wording.delete_limit_button(who),
                          wording.delete_limit_warning(who),
                          confirm_label=wording.YES_DELETE,
                          disabled=other_confirmation(key)):
        return
    try:
        storage.archive(opt.project_name, "pre_delete", copy=True)
    except storage_backend.StorageError as e:
        st.error(str(e))
        return
    remove()
    if saved_ok(opt):
        flash("success", wording.limit_deleted(who))
        st.rerun()


def _limit_label(opt, qc):
    """How one ingredient limit is named on screen — 'Total of each
    formulation', 'All ingredients' or 'Water + Oil'. It lives on the
    optimizer: the list under Limits, the line that reports a limit removed
    and the workbook's Set-up sheet all read from there, so they name it
    alike."""
    return opt.limit_label(qc)


_TARGETS_SOURCE_OPEN = "_targets_source_open"
_TARGETS_SOURCE_BOX = "targets_source_box"


def _targets_source_editor(opt):
    """The optional note on where the measurement targets came from: a
    caption once it is set, and a button that opens a one-line box to set or
    change it. Its Save is always secondary — unlike the measurement editor,
    opening this box does not take the foot's lit Continue away. Cancel
    closes it without saving, same word and same act as the measurement
    editor's own Cancel."""
    if opt.targets_source:
        st.caption(wording.targets_from_caption(opt.targets_source))
    if st.session_state.get(_TARGETS_SOURCE_OPEN):
        st.session_state.setdefault(_TARGETS_SOURCE_BOX, opt.targets_source)
        text = st.text_input(wording.TARGETS_SOURCE_LABEL,
                             key=_TARGETS_SOURCE_BOX,
                             placeholder=wording.TARGETS_SOURCE_PLACEHOLDER)
        b1, b2 = st.columns(2)
        with b1:
            if st.button(wording.SAVE_BUTTON, key="save_targets_source",
                         use_container_width=True):
                opt.set_targets_source(text)
                if saved_ok(opt):
                    st.session_state.pop(_TARGETS_SOURCE_OPEN, None)
                    st.rerun()
        with b2:
            if st.button(wording.CANCEL, key="cancel_targets_source",
                         use_container_width=True):
                st.session_state.pop(_TARGETS_SOURCE_OPEN, None)
                st.session_state.pop(_TARGETS_SOURCE_BOX, None)
                st.rerun()
    else:
        # Edit when there is a note above it, Add when there is not: a bare
        # "Edit this note" over nothing names a note the reader cannot see.
        label = (wording.TARGETS_SOURCE_BUTTON if opt.targets_source
                 else wording.ADD_TARGETS_SOURCE_BUTTON)
        if st.button(label, key="edit_targets_source"):
            st.session_state[_TARGETS_SOURCE_OPEN] = True
            st.rerun()


def _measurements(opt, storage):
    """The measurements, in a grid of their own.

    Share of score is the column that is typed into (spec 1.3): change one
    and the rest give way proportionally so the column still adds up to 100,
    and the app derives the importances from it. There is no Importance
    column any more, on this screen or on any sheet.

    Returns True while there is an edit in hand, for the same reason the
    grid above does.
    """
    st.subheader(wording.MEASUREMENTS_HEADER)
    st.caption(wording.MEASUREMENT_GRID_CAPTION)
    saved = opt.measurement_grid_frame()
    opening, from_park = _opening_frame(MEAS_GRID_KEY, saved)
    edited = st.data_editor(
        opening, key=grid_key(MEAS_GRID_KEY),
        num_rows="dynamic",
        column_config=_measurement_columns(), use_container_width=True,
        height=table_height(max(len(saved) + 1, 2), max_rows=20))
    slot = st.empty()
    _grid_errors(slot, _MEAS_ERRORS)
    pending = _pending(saved, edited)
    _keep_pending(MEAS_GRID_KEY, pending, edited, from_park)
    if pending:
        st.caption(wording.unsaved_grid_caption(wording.MEASUREMENTS_HEADER))
        _save_measurements(opt, storage, edited, lit=not _ingredients_pending())
    else:
        _disarm_grid_deletion(MEAS_SAVE_KEY)
    if opt.objectives:
        st.caption(opt.score_function_line())
    return pending


def _ingredients_pending():
    """True when the grid above this one already has an edit in hand. Its
    Save is then the coloured one: a tab shows one at a time, and the one
    higher up the page is the one the reader is looking at."""
    return st.session_state.get(ING_PENDING_KEY, False)


def _save_measurements(opt, storage, edited, lit=True):
    """The one write the measurements grid makes. A deleted row is confirmed
    by name first, exactly as one on the grid above is."""
    deletions = opt.measurement_grid_deletions(edited)
    key = MEAS_SAVE_KEY
    _disarm_stale_deletion(key, deletions)
    lit = lit and not confirmation_open()
    if not deletions:
        _remember_armed_deletions(key, deletions)
        if _save_and_discard(key, MEAS_GRID_KEY, lit):
            _apply_measurement_grid(opt, storage, edited)
        return
    confirmed = confirm_action(
        key, wording.SAVE_CHANGES_BUTTON,
        wording.delete_measurement_warning(number_list(deletions),
                                           many=len(deletions) > 1),
        confirm_label=wording.YES_DELETE, primary=lit,
        disabled=other_confirmation(key))
    _remember_armed_deletions(key, deletions)
    if not confirmed:
        _discard_beside(key, MEAS_GRID_KEY)
        return
    _apply_measurement_grid(opt, storage, edited)


def _apply_measurement_grid(opt, storage, edited):
    """Write the measurements grid.

    A copy is kept before every deletion on this tab, history or not, and
    before any save that recalculates a score already stored — not before
    every save: a unit corrected on one row is not something to keep a copy
    of, and a copy per keystroke is a copy of nothing. Which of those it is
    is the model's answer, handed in as a callback so the grid is read once
    per save rather than planned twice.
    """
    before = best_formulation_no(opt)
    copied = []

    def keep_a_copy():
        storage.archive(opt.project_name, "pre_edit", copy=True)
        copied.append(True)

    try:
        errors, messages = opt.apply_measurement_grid(edited,
                                                      archive=keep_a_copy)
    except storage_backend.StorageError as e:
        st.error(str(e))
        return
    if errors:
        st.session_state[_MEAS_ERRORS] = errors
        st.rerun()
    if not saved_ok(opt):
        return
    move = best_move_sentence(before, best_formulation_no(opt))
    green = [i for i, (kind, _) in enumerate(messages) if kind == "success"]
    if green:
        # The copy and the best moving are facts about the save as a whole,
        # so they land once, on its last green line.
        kind, line = messages[green[-1]]
        messages[green[-1]] = (kind, " ".join(
            p for p in (line, move, COPY_KEPT if copied else "") if p))
    for kind, line in messages:
        flash(kind, line)
    clear_grid(MEAS_GRID_KEY)
    # Above the properties grid, so its record is about to be dropped:
    # turning its key over is what lets the frame it parked be drawn in its
    # place. Its rows are the ingredients, which this save does not touch.
    rekey_grid(PROP_GRID_KEY)
    st.session_state.pop(_armed_deletions_key(MEAS_SAVE_KEY), None)
    st.rerun()


def _add_property(opt):
    """Name a property in the app. A property used to arrive only as an extra
    column in an ingredient CSV, which meant a project without one could not
    limit sodium at all without going back to a spreadsheet."""
    a1, a2 = st.columns([3, 1])
    with a1:
        st.session_state.setdefault("prop_new", "")
        typed = st.text_input(wording.ADD_PROPERTY_LABEL,
                              key="prop_new",
                              # The name carries the unit: nothing else on the
                              # screen can say whether 450 is mg or a percent.
                              placeholder=wording.ADD_PROPERTY_PLACEHOLDER)
    with a2:
        # Grey, like every other Add on this tab: the one coloured button is
        # Continue at the foot.
        if st.button(wording.ADD_PROPERTY_BUTTON, key="add_property"):
            try:
                added = opt.add_property(typed)
            except ValueError as e:
                st.error(str(e))
            else:
                if saved_ok(opt):
                    flash("success", wording.property_added(added))
                    park_clear("prop_new", "")
                    # The grid below gains a column; a record of edits made
                    # against the old set of columns would be replayed onto
                    # the new one.
                    clear_grid(PROP_GRID_KEY)
                    st.rerun()


def _property_columns(properties):
    """The row column is the ingredient's name and cannot be typed into: the
    grid above owns the rows. Every other column is a property, and every
    cell under it is one figure."""
    columns = {wording.PROPERTIES_ROW_COLUMN: st.column_config.TextColumn(
        wording.PROPERTIES_ROW_COLUMN, disabled=True)}
    for prop in properties:
        columns[prop] = _number_column(prop)
    return columns


def _apply_property_grid(opt, edited):
    """Hand the finished properties grid to the model, and say what it did.

    One green line for the save as a whole, and none at all when nothing
    moved: a grid saved unchanged has written nothing to report.
    """
    errors, messages = opt.apply_property_grid(edited)
    if errors:
        st.session_state[_PROP_ERRORS] = errors
        st.rerun()
    if not saved_ok(opt):
        return
    for kind, line in messages:
        flash(kind, line)
    clear_grid(PROP_GRID_KEY)
    st.rerun()


def _delete_property(opt, storage, properties):
    """A property goes from a picker and one Delete, not from a Delete per
    row: the list is beside a grid that already names every property across
    its head, and a column of Delete buttons said each name twice."""
    # Assigned before the widget is created, which is the only moment
    # Streamlit allows it: the property the picker was left on may have just
    # been deleted, and a select box whose stored value is not in its options
    # refuses to draw at all.
    if st.session_state.get(_PROP_DELETE) not in properties:
        st.session_state.pop(_PROP_DELETE, None)
    d1, d2 = st.columns([3, 1])
    with d1:
        prop = st.selectbox(wording.DELETE_PROPERTY_PICK_LABEL, properties,
                            key=_PROP_DELETE)
    key = f"rm_prop_{prop}"
    # The question belongs to the property it was armed over. Pick another
    # and the question on screen is about one the picker no longer shows, so
    # it is taken down rather than left arming every coloured button away.
    armed = armed_confirmation()
    if armed != key and str(armed or "").startswith("rm_prop_"):
        disarm(armed)
    limits = sum(1 for c in opt.constraints
                 if str(c['metric']).strip().lower() == prop.lower())
    limits_text = plural(limits, wording.LIMIT) if limits else None
    with d2:
        confirmed = confirm_action(
            key, wording.delete_button(prop),
            wording.delete_property_warning(prop, limits_text),
            confirm_label=wording.YES_DELETE,
            disabled=other_confirmation(key))
    if not confirmed:
        return
    try:
        storage.archive(opt.project_name, "pre_delete", copy=True)
        gone_limits = opt.remove_property(prop)
    except (ValueError, storage_backend.StorageError) as e:
        st.error(str(e))
        return
    if saved_ok(opt):
        gone = (wording.limit_went_with_it(plural(len(gone_limits),
                                                  wording.LIMIT))
                if gone_limits else "")
        flash("success", wording.property_deleted(prop, gone))
        # The grid has lost a column; a record of edits against the old one
        # would write a figure into a property that no longer exists.
        clear_grid(PROP_GRID_KEY)
        st.rerun()


def _properties(opt, storage):
    """Properties, as a grid of their own: rows are the ingredients, columns
    are the properties, and a cell is that ingredient's figure per 100 g
    (spec 1.5). It replaces a fold holding a picker and one box per property
    — one ingredient at a time, with no way to read the column down.

    `Save properties` is a SECONDARY button that applies on the click, where
    the two grids above light a primary `Save changes` and hand the foot's
    Continue aside. Two reasons, and both are about the one lit button. This
    grid lives inside a collapsed expander, so a primary Save here would be
    the tab's coloured button hidden inside a fold — the reader would see a
    grey Continue and nothing lit anywhere. And a third pending flag would
    have to be threaded through the topmost-grid rule to keep the count at
    one. The unsaved line below still says nothing is written while typing;
    what is typed survives until it is saved or the project changes.
    """
    st.markdown(wording.PROPERTIES_HEADING)
    properties = opt.grid_properties()
    # Every property the project knows, the row-column name included: a
    # property named "Ingredient" can only have arrived as a column of an
    # old ingredient file (the app itself refuses that name for a new one),
    # and it deletes the same way as any other even though it never shows
    # on the grid above.
    all_properties = opt.properties()
    names = opt.ingredient_names()
    if properties and names:
        # Property names carry their own basis as often as not ("Fat per
        # 100 g"), and a caption that then adds ", per 100 g" said it twice.
        said_already = all("per 100 g" in prop.lower() for prop in properties)
        st.caption(wording.properties_grid_caption(said_already))
        saved = opt.property_grid_frame()
        # The third grid on this tab keeps its edit the same way the two
        # above it do: a Save or a Discard up there reruns before this one
        # is drawn, and Streamlit throws away the entry of a widget the run
        # never created. It carries a banner of its own, so it has to be
        # able to keep what the banner promises.
        opening, from_park = _opening_frame(PROP_GRID_KEY, saved)
        edited = st.data_editor(
            opening, key=grid_key(PROP_GRID_KEY), num_rows="fixed",
            column_config=_property_columns(properties),
            use_container_width=True,
            height=table_height(max(len(saved), 1), max_rows=20))
        slot = st.empty()
        _grid_errors(slot, _PROP_ERRORS)
        pending = _pending(saved, edited)
        _keep_pending(PROP_GRID_KEY, pending, edited, from_park)
        if pending:
            st.caption(wording.unsaved_grid_caption(
                wording.PROPERTIES_NAME))
        if st.button(wording.SAVE_PROPERTIES_BUTTON, key="save_properties",
                     disabled=confirmation_open()):
            _apply_property_grid(opt, edited)
    _add_property(opt)
    if all_properties:
        _delete_property(opt, storage, all_properties)


def _property_limits(opt):
    """The finished-product limit, and the properties it is written
    against. Ingredients only: a property is a value each ingredient carries,
    and a process setting is weighed into nothing."""
    st.markdown(wording.FINISHED_PRODUCT_LIMIT_HEADING)
    # Per 100 g of what you make, not a total that grows with the formulation:
    # the same limit then means the same thing at 100 g and at 10 kg. Written
    # in the unit the ingredients are actually in. While the ingredients
    # differ, there is no 100 of anything yet — but that is already said
    # once, by the Default batch size box above (wording.NEEDS_ONE_UNIT), so
    # nothing is drawn here rather than saying it again in different words.
    unit = opt.one_amount_unit()
    if unit is not None:
        st.caption(wording.per_100_caption(unit or 'g'))
    properties = opt.grid_properties()
    if not properties:
        # The grid that names one is further down this expander, so the line
        # points at it rather than leaving an empty picker on screen.
        st.caption(wording.NO_PROPERTIES_YET_CAPTION)
        return
    metric = st.selectbox(wording.INGREDIENT_PROPERTY_LABEL, properties,
                          key="prop_metric")
    p1, p2 = st.columns(2)
    with p1:
        st.session_state.setdefault("prop_min", None)
        st.number_input(wording.per_100_box_label(wording.AT_LEAST_LABEL,
                                                 unit or 'g'),
                        placeholder=wording.NO_LIMIT_PLACEHOLDER,
                       key="prop_min")
    with p2:
        st.session_state.setdefault("prop_max", None)
        st.number_input(wording.per_100_box_label(wording.AT_MOST_LABEL,
                                                 unit or 'g'),
                        placeholder=wording.NO_LIMIT_PLACEHOLDER,
                       key="prop_max")
    if st.button(wording.ADD_PROPERTY_LIMIT_BUTTON, key="add_property_limit"):
        low, high = st.session_state["prop_min"], st.session_state["prop_max"]
        if low is None and high is None:
            st.error(wording.ENTER_LOWEST_HIGHEST_ERROR)
        else:
            try:
                opt.add_constraint(metric, min_val=low, max_val=high)
            except ValueError as e:
                st.error(str(e))
            else:
                _report_limit(opt, wording.limit_added_on(metric))


def _limit_gap_tail(opt, metric):
    """'· Water has no figure for it and counts as 0.' — the ingredients
    this limit is silently reading as zeroes. A limit that looks satisfied
    because half the formulation was never given a figure is the one way this
    arithmetic lies."""
    gaps = opt.ingredients_without_property(metric)
    if not gaps:
        return ""
    many = len(gaps) > 1
    return wording.limit_gap_tail(number_list(gaps) if many else gaps[0], many)


def _limits(opt, storage):
    """Limits, inside More settings. A heading and not a fold of its own:
    Streamlit cannot nest one expander in another, and a tab that folded the
    optional half away twice made the reader open two things to reach one.
    """
    # A limit is a sum, and a sum only has a unit when the ingredients share
    # one. When they do not, the labels stay bare and the limit itself is
    # refused in words that name the ingredient to re-enter.
    unit = opt.one_amount_unit() or ""
    # Both kinds of limit are about what you weigh out, so a project that
    # weighs nothing out — a fermentation run of settings alone — is not
    # offered the section at all. A setting's own Lowest and Highest are its
    # bounds.
    if not opt.has_ingredients():
        return
    st.markdown(wording.LIMITS_HEADING)
    # "of your ingredients", not "from your ingredient file": a property
    # is named in the app as often as it arrives in a file, and the grid
    # that names one is further down this expander.
    st.caption(wording.LIMITS_CAPTION)

    _property_limits(opt)

    # A limit written before 0.3.0 was a total, and the file does not say
    # so; the same stored number now means a per-100 g average. Said once,
    # above the list, and only while such a limit is still there.
    if any(c.get('basis') != 'per_100' for c in opt.constraints):
        st.caption(wording.old_limit_basis_caption(opt.one_amount_unit() or 'g'))

    for i, constraint in enumerate(opt.constraints):
        c1, c2 = st.columns([3, 1])
        with c1:
            bounds = ([wording.at_least(constraint['min'])]
                      if constraint['min'] is not None else [])
            bounds += ([wording.at_most(constraint['max'])]
                       if constraint['max'] is not None else [])
            st.text(f"{constraint['metric']}: {' and '.join(bounds)}"
                    + _limit_gap_tail(opt, constraint['metric']))
        with c2:
            _delete_limit(opt, storage, f"rm_constr_{i}",
                          constraint['metric'],
                          lambda i=i: opt.remove_constraint(i))

    names = opt.ingredient_names()

    # ONE amount limit, on the ingredients the user names. The total
    # over ALL of them is not written here — it is the Default batch size
    # box at the top of this expander, and this picker asks for a choice
    # like every other picker in the app.
    st.markdown(wording.LIMIT_ON_CHOSEN_INGREDIENTS_HEADING)
    picked = st.multiselect(wording.INGREDIENTS_TO_LIMIT_LABEL, names,
                            key="qty_pick",
                            placeholder=wording.CHOOSE_MANY_PLACEHOLDER)
    q1, q2 = st.columns(2)
    with q1:
        st.session_state.setdefault("qc_min", None)
        st.number_input(f"{wording.AT_LEAST_LABEL}{_unit_suffix(unit)}",
                        placeholder=wording.NO_LIMIT_PLACEHOLDER, key="qc_min")
    with q2:
        st.session_state.setdefault("qc_max", None)
        st.number_input(f"{wording.AT_MOST_LABEL}{_unit_suffix(unit)}",
                        placeholder=wording.NO_LIMIT_PLACEHOLDER, key="qc_max")
    # No "Set a maximum" tick box: a blank field already means no limit,
    # and a box the user forgot to tick silently threw their number away.
    # Nothing picked is no longer "every ingredient": the total over all
    # of them has its own box above, and a picker that quietly meant all
    # eight while showing none was the harder half of that one idea to read.
    if st.button(wording.ADD_INGREDIENT_LIMIT_BUTTON, key="add_amount_limit",
                 disabled=not picked) and picked:
        low, high = st.session_state["qc_min"], st.session_state["qc_max"]
        if low is None and high is None:
            st.error(wording.ENTER_LOWEST_HIGHEST_ERROR)
        else:
            try:
                opt.add_quantity_constraint(picked, min_val=low,
                                            max_val=high)
            except ValueError as e:
                st.error(str(e))
            else:
                _report_limit(opt,
                              wording.limit_added_on(" + ".join(picked)))

    for i, qc in enumerate(getattr(opt, "quantity_constraints", [])):
        if qc.get('source') == 'formulation_total':
            # Not listed at all. The default batch size is answered in its
            # own box at the top of this expander, with a help line that
            # says what it does; a third mention here, read-only, with a
            # caption telling the reader to go back up and empty a box, was
            # the one the reader could not act on where it stood.
            continue
        l1, l2 = st.columns([3, 1])
        with l1:
            # One line per limit, written by the optimizer: a limit sums
            # ingredients that share a unit, so it is written in that
            # unit ("at most 400 g", never a bare 400).
            st.text(opt.limit_text(qc))
        with l2:
            _delete_limit(opt, storage, f"rm_qc_{i}", _limit_who(opt, qc),
                          lambda i=i: opt.remove_quantity_constraint(i))


def _advanced(opt):
    """The bottom tier (spec 1.5): the model settings and the two
    explanations, in one small fold at the very foot of the tab.

    The explanations are plain markdown lists rather than folds of their own
    — Streamlit cannot nest an expander in an expander, and three things to
    open before the first sentence is the state this tier exists to end.
    """
    with st.expander(wording.ADVANCED_EXPANDER):
        st.markdown(wording.HOW_FORMULATIONS_CHOSEN_HEADING)
        st.caption(wording.STANDARD_VS_EXPERT_CAPTION)
        current = getattr(opt, "bo_config", None)
        # The radio's label is collapsed: the section it is the only control
        # in already names it, and a heading repeated as a label reads as two
        # things.
        mode = st.radio(wording.HOW_FORMULATIONS_CHOSEN_LABEL,
                        [wording.STANDARD_DEFAULT_OPTION, wording.EXPERT_SELECTED_OPTION],
                        index=1 if current else 0, key="bo_cfg_mode",
                        horizontal=True, label_visibility="collapsed")
        if mode == wording.STANDARD_DEFAULT_OPTION:
            if current is not None and st.button(wording.REVERT_TO_STANDARD_BUTTON,
                                                 key="bo_revert"):
                opt.set_bo_config(None)
                if saved_ok(opt):
                    flash("success", wording.USING_DEFAULT_MODEL_SETTINGS)
                    st.rerun()
        else:
            b1, b2 = st.columns(2)
            with b1:
                kernel = st.selectbox(wording.KERNEL_LABEL,
                                      wording.KERNEL_OPTIONS,
                                      key="bo_kernel")
                prior = st.selectbox(wording.LENGTHSCALE_PRIOR_LABEL,
                                     wording.LENGTHSCALE_PRIOR_OPTIONS,
                                     key="bo_prior")
            with b2:
                noise = st.selectbox(wording.NOISE_LABEL,
                                     wording.NOISE_OPTIONS, key="bo_noise")
                acq = st.selectbox(wording.ACQUISITION_LABEL,
                                   wording.ACQUISITION_OPTIONS,
                                   key="bo_acq")
            st.caption(wording.FIXED_TINY_NOISE_CAPTION)
            if st.button(wording.APPLY_EXPERT_SETTINGS_BUTTON, key="bo_apply"):
                opt.set_bo_config({"kernel": kernel, "lengthscale_prior": prior,
                                   "noise": noise, "acquisition": acq})
                if saved_ok(opt):
                    flash("success", wording.MODEL_SETTINGS_UPDATED)
                    st.rerun()
            if st.checkbox(wording.PASTE_EXPERT_SETTINGS_CHECKBOX, key="bo_paste"):
                text = st.text_area(
                    wording.EXPERT_SETTINGS_JSON_LABEL,
                    value='{"kernel": "matern52", "lengthscale_prior": "default", '
                          '"noise": "default", "acquisition": "qlognei"}',
                    key="bo_json",
                )
                if st.button(wording.APPLY_PASTED_SETTINGS_BUTTON, key="bo_apply_json"):
                    try:
                        opt.set_bo_config(json.loads(text))
                    except Exception as e:
                        st.error(wording.invalid_json(e))
                    else:
                        if saved_ok(opt):
                            flash("success", wording.MODEL_SETTINGS_UPDATED)
                            st.rerun()
        # Only under Standard: with the boxes on screen this line repeats
        # what they already show, four values at a time.
        if current and mode == wording.STANDARD_DEFAULT_OPTION:
            st.caption(wording.IN_USE_PREFIX
                       + ", ".join(f"{k}: {v}" for k, v in current.items()))

        # Four flat bullets, then the arithmetic behind the second one
        # directly beneath: the pair used to be two folds under the
        # measurements grid, where they were the last thing on the tab a
        # formulator needed and the first thing they saw.
        st.markdown(wording.HOW_IT_WORKS_HEADING)
        st.markdown("\n".join("- " + line for line in HOW_IT_WORKS))
        st.markdown(wording.HOW_CLOSENESS_HEADING)
        st.markdown("\n".join("- " + line for line in HOW_CLOSENESS))


def _more_settings(opt, storage):
    """The middle tier (spec 1.5): everything tab 1 asks at most once, in one
    collapsed expander, in the order a project needs it — how big a
    formulation is, where the targets came from, the hard rules, and the
    figures those rules read.

    Nothing in here is a fold of its own. Streamlit cannot nest one expander
    in another, and the point of the tier is that the tab has ONE thing to
    open rather than six.
    """
    with st.expander(wording.MORE_SETTINGS_EXPANDER):
        _formulation_total(opt)
        _targets_source_editor(opt)
        _limits(opt, storage)
        _properties(opt, storage)


def _foot(opt, pending=False):
    ready, missing = readiness(opt)
    # While a confirmation is armed, its "Yes" is the one coloured button and
    # answering it is the one thing to do; moving on can wait a click. An
    # unsaved grid is the same case: its `Save changes` is the lit one, and
    # Continue would leave the tab and throw the edit away.
    lit = ready and not confirmation_open() and not pending
    if st.button(wording.NEXT_MAKE_BATCH_BUTTON,
                 type="primary" if lit else "secondary",
                 disabled=not lit, key="continue_to_batch") and lit:
        go_to_tab(TAB_BATCH)
    if not ready:
        st.caption(missing)


def render(opt, storage):
    # The sample's own welcome, directly under the tab's title: gone the
    # moment any formulation exists, scored or not — a batch whose one row
    # was ticked Not scored has still been made, and "Next: make a round"
    # would be wrong about it.
    if (not opt.X_history and not opt.skipped
            and opt.project_name == wording.SAMPLE_PROJECT_NAME):
        st.caption(wording.SAMPLE_TAB1_DESCRIPTION)
    pending = _variables(opt, storage)
    st.divider()
    pending = _measurements(opt, storage) or pending
    st.divider()
    _more_settings(opt, storage)
    _advanced(opt)
    st.divider()
    _foot(opt, pending)
