"""Tab 1 · Set up: what the project can vary and what will be measured.

One column, in the order a formulator fills it in: the ingredients and the
process settings together, then the measurements, then the optional
sections. Exactly one coloured button lives here — `Next: make a round` at
the foot.
"""
import json
import os

import pandas as pd
import streamlit as st

import storage as storage_backend
import wording
from food_bo import (
    GRID_ID, WORKBOOK_MIME, goal_text, grid_signature,
    ingredients_template_workbook, measurement_range_text,
)
from ui_helpers import (
    COPY_KEPT, TAB_BATCH, armed_confirmation, best_formulation_no,
    best_move_sentence, clear_formulation_total_box, clear_grid, grid_key,
    clear_scale_total, confirm_action, confirmation_open,
    disarm, flash,
    go_to_tab, number_list, other_confirmation, park_clear, plural,
    preserve_tab_forms, readiness, saved_ok, table_height,
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


def _note_discarded_batch(opt, batch_no_before,
                          reason=wording.SETUP_CHANGED_REASON):
    """Flash the notice when the write just now retired the open batch. The
    batch is named: the notice lands above the tabs, away from the table it
    is about.

    `reason` is what discarded it. The tab as a whole is the honest answer
    for the ingredient list and the allowed amounts — two ways to
    one place — but the total is one control the reader has just touched,
    and blaming "your set-up" sent them looking for what else they had
    done."""
    if batch_no_before is not None and opt.pending_batch_no is None:
        flash("info", wording.batch_discarded_notice(batch_no_before, reason))


def _goal_text(obj):
    """'Target 6 N', 'Higher is better', 'Lower is better'. It lives in
    food_bo, because the workbook's Set-up sheet says it too."""
    return goal_text(obj)


def _range_text(obj):
    return measurement_range_text(obj)


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


def _scaled_now(opt):
    """The batch size the open round is being made to, or None. Only a real
    size counts: the box is not even offered while the ingredients differ in
    unit."""
    if opt.one_amount_unit() is None:
        return None
    value = st.session_state.get("scale_total")
    if value is None or float(value) <= 0:
        return None
    return float(value)


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

ING_GRID_KEY = "ingredient_grid"
MEAS_GRID_KEY = "measurement_grid"
# Where a Save that refused leaves its per-row errors, for the slot under
# the grid to show on this same run. The grid is drawn before the button is
# clicked, so the errors cannot simply be printed where they belong.
_ING_ERRORS = "_ingredient_grid_errors"
_MEAS_ERRORS = "_measurement_grid_errors"
GRID_KEYS = (ING_GRID_KEY, MEAS_GRID_KEY)


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
        wording.SKU_LABEL: st.column_config.TextColumn(wording.SKU_LABEL),
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
            clear_grid(grid)
            st.rerun()
    return save


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
    edited = st.data_editor(
        saved, key=grid_key(ING_GRID_KEY), num_rows="dynamic",
        column_config=_ingredient_columns(opt, saved),
        use_container_width=True,
        height=table_height(max(len(saved) + 1, 2), max_rows=20))
    slot = st.empty()            # where a refused Save writes its rows
    _grid_errors(slot, _ING_ERRORS)
    pending = _pending(saved, edited)
    # Read by the grid below, which keeps its own pair of buttons grey while
    # this one has something to save: one coloured button per tab.
    st.session_state["_ingredient_grid_pending"] = pending
    if pending:
        st.caption(wording.UNSAVED_CHANGES_CAPTION)
        _save_ingredients(opt, storage, edited)
    _formulation_total(opt)
    if getattr(opt, "amount_unit_backfilled", False):
        # The file this project was saved in predates the unit; its amounts
        # may have been percentages or millilitres, and nothing on screen
        # would otherwise say the g was the app's guess and not the user's.
        st.caption(wording.made_before_units_caption(opt.amount_unit))
    _set_properties(opt)
    with st.expander(wording.UPLOAD_INGREDIENTS_EXPANDER):
        _upload_ingredients(opt)
    return pending


def _save_ingredients(opt, storage, edited):
    """The one write the ingredients grid makes. A deleted row is confirmed
    by name first, and a copy is kept before anything goes."""
    deletions = opt.ingredient_grid_deletions(edited)
    key = "save_ingredient_grid"
    lit = not confirmation_open()
    if not deletions:
        if _save_and_discard(key, ING_GRID_KEY, lit):
            _apply_ingredient_grid(opt, edited)
        return
    _disarm_stale_deletion(key, deletions)
    confirmed = confirm_action(
        key, wording.SAVE_CHANGES_BUTTON,
        wording.delete_rows_warning(number_list(deletions)),
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
    a deletion is waiting to be confirmed."""
    if st.button(wording.DISCARD_CHANGES_BUTTON, key=f"{key}__discard"):
        clear_grid(grid)
        st.rerun()


_ARMED_DELETIONS = "_grid_deletions_armed"


def _disarm_stale_deletion(key, deletions):
    """An armed Save belongs to the rows it was armed over. Put one back on
    the grid and the question on screen is about a different set of rows, so
    it is taken down rather than answered — a "Delete Water?" left standing
    over a grid that no longer deletes Water is one click from deleting
    something else."""
    armed = st.session_state.get(_ARMED_DELETIONS)
    if armed is not None and armed != sorted(deletions):
        disarm(key)
        st.session_state.pop(_ARMED_DELETIONS, None)


def _remember_armed_deletions(key, deletions):
    """...which means remembering them at the moment of arming, and only
    then: the click that arms is recorded INSIDE confirm_action, so nothing
    before it can see the question go up."""
    if armed_confirmation() == key:
        st.session_state[_ARMED_DELETIONS] = sorted(deletions)
    else:
        st.session_state.pop(_ARMED_DELETIONS, None)


def _apply_ingredient_grid(opt, edited, force=()):
    """Hand the finished grid to the model, and say what it did.

    Every consequence arrives as one message from there, because the model
    is what knows whether the default batch size survived, which limits were
    pruned and whether the open round is still open. The one sentence the
    screen owes on its own is the unit-split tail: it is the only one that
    has a box on this tab to empty as well as something to say.
    """
    scaled = _scaled_now(opt)
    errors, messages = opt.apply_ingredient_grid(edited, force=force)
    if errors:
        st.session_state[_ING_ERRORS] = errors
        st.rerun()
        return
    if not saved_ok(opt):
        return
    tail = _unscaled_tail(opt, scaled)
    if tail and messages:
        kind, line = messages[0]
        messages[0] = (kind, f"{line} {tail}")
    for kind, line in messages:
        flash(kind, line)
    clear_grid(ING_GRID_KEY)
    st.session_state.pop(_ARMED_DELETIONS, None)
    st.rerun()


def _set_properties(opt):
    """The one control left over from the old row of five.

    Properties get a grid of their own inside More settings in the next
    wave (spec 1.5); until then they keep a path rather than losing one, and
    it is folded away because a project without properties has nothing to
    set and a project with them sets them once."""
    properties = opt.properties()
    names = [v['name'] for v in opt.variables
             if v.get('category', 'ingredient') == 'ingredient']
    if not properties or not names:
        return
    with st.expander(wording.SET_PROPERTIES_BUTTON):
        pick = st.selectbox(wording.PROPERTIES_PICK_LABEL, names,
                            key="prop_pick")
        _property_value_editor(opt, pick, properties)


def _pkey(pick, prop):
    return f"setprop_{pick}_{prop}"


def _property_value_editor(opt, pick, properties):
    """One box per property for the picked ingredient. An empty box counts as 0 in the per-100 average — and
    every limit on that property names the ingredients it is reading as
    zeroes. The caption above the boxes is what names the properties; the
    button cannot, because a property name is the project's own and may run
    to "Sodium mg per 100 g"."""
    # Property names carry their own basis as often as not ("Fat per 100 g"),
    # and a caption that then adds ", per 100 g" said it three times.
    said_already = all("per 100 g" in prop.lower() for prop in properties)
    st.caption(wording.properties_for_caption(number_list(properties), pick,
                                              said_already))
    boxes = st.columns(min(4, len(properties)))
    for j, prop in enumerate(properties):
        with boxes[j % len(boxes)]:
            # Opened on the value this ingredient already has. The key is
            # per (ingredient, property), so picking another ingredient
            # creates boxes that have never been seeded and this seeds them
            # — which is what stops one row's figures showing under
            # another's name.
            st.session_state.setdefault(
                _pkey(pick, prop),
                float(opt.property_value(pick, prop))
                if opt.has_property_value(pick, prop) else None)
            st.number_input(prop, key=_pkey(pick, prop),
                           placeholder=wording.PROPERTY_PLACEHOLDER)
    # One button, not two: the editor lives inside a fold now, and a fold
    # has its own way of closing.
    if st.button(wording.SAVE_BUTTON, key="save_props"):
        for prop in properties:
            opt.set_property_value(pick, prop,
                                   st.session_state.get(_pkey(pick, prop)))
        if saved_ok(opt):
            flash("success", wording.properties_saved(
                number_list(properties), pick))
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
    edited = st.data_editor(
        saved, key=grid_key(MEAS_GRID_KEY), num_rows="dynamic",
        column_config=_measurement_columns(), use_container_width=True,
        height=table_height(max(len(saved) + 1, 2), max_rows=20))
    slot = st.empty()
    _grid_errors(slot, _MEAS_ERRORS)
    pending = _pending(saved, edited)
    if pending:
        st.caption(wording.UNSAVED_CHANGES_CAPTION)
        _save_measurements(opt, storage, edited, lit=not _ingredients_pending())
    if opt.objectives:
        st.caption(opt.score_function_line())
    _targets_source_editor(opt)

    # Four flat bullets, then the arithmetic behind the second one folded
    # directly beneath: one fold of nine bullets answered a question most
    # readers never asked, in the middle of the four that say what happens.
    with st.expander(wording.HOW_IT_WORKS_EXPANDER):
        st.markdown("\n".join("- " + line for line in HOW_IT_WORKS))
    with st.expander(wording.HOW_CLOSENESS_EXPANDER):
        st.markdown("\n".join("- " + line for line in HOW_CLOSENESS))
    return pending


def _ingredients_pending():
    """True when the grid above this one already has an edit in hand. Its
    Save is then the coloured one: a tab shows one at a time, and the one
    higher up the page is the one the reader is looking at."""
    return st.session_state.get("_ingredient_grid_pending", False)


def _save_measurements(opt, storage, edited, lit=True):
    deletions = opt.measurement_grid_deletions(edited)
    key = "save_measurement_grid"
    lit = lit and not confirmation_open()
    if not deletions:
        if _save_and_discard(key, MEAS_GRID_KEY, lit):
            _apply_measurement_grid(opt, storage, edited)
        return
    confirmed = confirm_action(
        key, wording.SAVE_CHANGES_BUTTON,
        wording.delete_measurement_warning(number_list(deletions),
                                           many=len(deletions) > 1),
        confirm_label=wording.YES_DELETE, primary=lit,
        disabled=other_confirmation(key))
    if not confirmed:
        _discard_beside(key, MEAS_GRID_KEY)
        return
    _apply_measurement_grid(opt, storage, edited)


def _apply_measurement_grid(opt, storage, edited):
    """Write the measurements grid. A copy is kept first whenever there is a
    history to rescore: this save has no confirmation of its own to promise
    one, and every stored overall score can move under it."""
    before = best_formulation_no(opt)
    copied = False
    # A copy is kept before every deletion on this tab, history or not, and
    # before any save that recalculates a score already stored. Not before
    # every save: a unit corrected on one row is not something to keep a
    # copy of, and a copy per keystroke is a copy of nothing.
    if (opt.measurement_grid_deletions(edited)
            or (opt.Y_history and opt.measurement_grid_rescores(edited))):
        try:
            storage.archive(opt.project_name, "pre_edit", copy=True)
        except storage_backend.StorageError as e:
            st.error(str(e))
            return
        copied = True
    errors, messages = opt.apply_measurement_grid(edited)
    if errors:
        st.session_state[_MEAS_ERRORS] = errors
        st.rerun()
        return
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
                    st.rerun()


def _property_list(opt, storage, properties):
    """Every property, with a Delete that names what goes with it."""
    for prop in properties:
        limits = sum(1 for c in opt.constraints
                     if str(c['metric']).strip().lower() == prop.lower())
        key = f"rm_prop_{prop}"
        limits_text = plural(limits, wording.LIMIT) if limits else None
        c1, c2 = st.columns([3, 1])
        with c1:
            st.text(prop)
        with c2:
            confirmed = confirm_action(
                key, wording.delete_button(prop),
                wording.delete_property_warning(prop, limits_text),
                confirm_label=wording.YES_DELETE, disabled=other_confirmation(key),
            )
        if confirmed:
            try:
                storage.archive(opt.project_name, "pre_delete", copy=True)
                removed = opt.remove_property(prop)
            except (ValueError, storage_backend.StorageError) as e:
                st.error(str(e))
            else:
                if saved_ok(opt):
                    gone = (wording.limit_went_with_it(plural(len(removed), wording.LIMIT))
                            if removed else "")
                    flash("success", wording.property_deleted(prop, gone))
                    st.rerun()


def _property_limits(opt, storage):
    """The finished-product limit, and the properties it is written
    against. Ingredients only: a property is a value each ingredient carries,
    and a process setting is weighed into nothing."""
    st.markdown(wording.FINISHED_PRODUCT_LIMIT_HEADING)
    # Per 100 g of what you make, not a total that grows with the formulation:
    # the same limit then means the same thing at 100 g and at 10 kg. Written
    # in the unit the ingredients are actually in.
    unit = opt.one_amount_unit()
    if unit is None:
        # The ingredients differ, so there is no 100 of anything yet, and the
        # limit itself is refused in words that name the fix.
        st.caption(wording.PER_100G_UNRESOLVED_CAPTION)
    else:
        st.caption(wording.per_100_caption(unit or 'g'))
    _add_property(opt)
    properties = opt.properties()
    if not properties:
        return
    _property_list(opt, storage, properties)
    metric = st.selectbox(wording.INGREDIENT_PROPERTY_LABEL, properties,
                          key="prop_metric")
    p1, p2 = st.columns(2)
    with p1:
        st.session_state.setdefault("prop_min", None)
        st.number_input(wording.AT_LEAST_LABEL, placeholder=wording.NO_LIMIT_PLACEHOLDER,
                       key="prop_min")
    with p2:
        st.session_state.setdefault("prop_max", None)
        st.number_input(wording.AT_MOST_LABEL, placeholder=wording.NO_LIMIT_PLACEHOLDER,
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
    with st.expander(wording.LIMITS_EXPANDER):
        # "of your ingredients", not "from your ingredient file": a property
        # is named in the app as often as it arrives in a file, and the box
        # that names one is two lines below this caption.
        st.caption(wording.LIMITS_CAPTION)

        _property_limits(opt, storage)

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

        names = [v['name'] for v in opt.variables
                 if v.get('category', 'ingredient') == 'ingredient']

        # ONE amount limit, on the ingredients the user names. The total
        # over ALL of them is not written here — it is the box under the
        # ingredients table, and this picker asks for a choice like every
        # other picker in the app.
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
        # of them has its own box under the ingredients table, and a picker
        # that quietly meant all eight while showing none was the harder
        # half of that one idea to read.
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
            l1, l2 = st.columns([3, 1])
            with l1:
                # One line per limit, written by the optimizer: a limit sums
                # ingredients that share a unit, so it is written in that
                # unit ("at most 400 g", never a bare 400), and the total
                # reads as the one number it is rather than as the
                # half-percent band it is enforced as.
                st.text(opt.limit_text(qc))
            if qc.get('source') == 'formulation_total':
                # The total's row is a reading of the box at the top of this
                # tab, not a second control for it. A Delete here let one
                # rule be taken off in two places, and the cold read could
                # not tell which of the two was the real one.
                with l1:
                    st.caption(wording.FORMULATION_TOTAL_IN_LIMITS_CAPTION)
                continue
            with l2:
                _delete_limit(opt, storage, f"rm_qc_{i}", _limit_who(opt, qc),
                              lambda i=i: opt.remove_quantity_constraint(i))


def _advanced(opt):
    with st.expander(wording.HOW_FORMULATIONS_CHOSEN_EXPANDER):
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
    _limits(opt, storage)
    _advanced(opt)
    st.divider()
    _foot(opt, pending)
