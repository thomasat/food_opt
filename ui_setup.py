"""Tab 1 · Set up: what the project can vary and what will be measured.

One column, in the order a formulator fills it in: the ingredients and the
process settings together, then the measurements, then the optional
sections. Exactly one coloured button lives here — `Next: make a batch` at
the foot.
"""
import json
import os

import pandas as pd
import streamlit as st

import storage as storage_backend
import wording
from ui_helpers import (
    COPY_KEPT, TAB_BATCH, armed_confirmation, best_formulation_no,
    best_move_sentence, confirm_action, confirmation_open, disarm, flash,
    fmt_amount, fmt_setting, go_to_tab, join_unit, label_with_unit,
    number_list, other_confirmation, park_clear, plural, readiness, saved_ok,
    table_height, unit_after_number,
)

_SAMPLE_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "sample_ingredients.csv")

# The one collapsed expander that maps the words on this tab to the words a
# specialist would use, and the tuple marking its three nested goal lines,
# both live in wording.py; importable from here too, since HOW_IT_WORKS is
# read by name from tests/test_food_bo.py's vocabulary guard.
from wording import HOW_IT_WORKS, HOW_IT_WORKS_NESTED  # noqa: E402,F401


# ------------------------------------------------------------------ #
#  Small shared text
# ------------------------------------------------------------------ #

def _unit_suffix(unit):
    return f" ({unit})" if unit else ""


def _note_discarded_batch(opt, batch_no_before):
    """Flash the notice when the write just now retired the open batch."""
    if batch_no_before is not None and opt.pending_batch_no is None:
        flash("info", wording.batch_discarded_notice())


def _goal_text(obj):
    """'Target 6 N', 'Higher is better', 'Lower is better'. A '/10' rides on
    the measurement's own name instead of on every number in its row."""
    if obj['goal'] == 'target':
        return join_unit(wording.target_value(obj['target']),
                         unit_after_number(obj.get('unit')))
    return wording.GOAL_LABELS.get(obj['goal'], obj['goal'])


def _range_text(obj):
    return join_unit(wording.range_text(obj['min_val'], obj['max_val']),
                     unit_after_number(obj.get('unit')))


def _mkey(editing, field):
    """Widget keys are per measurement, so opening Edit shows that
    measurement's values rather than whatever was last typed elsewhere."""
    who = "new" if editing is None else editing['name']
    return f"meas_{who}_{field}"


def _clear_measurement_keys(editing):
    for field in ("name", "unit", "goal", "target", "min", "max", "importance"):
        st.session_state.pop(_mkey(editing, field), None)


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
    """The total the open batch is scaled to, or None. Only a real scaling
    counts: the box is not even offered while the ingredients differ in unit."""
    if opt.one_amount_unit() is None:
        return None
    value = st.session_state.get("scale_total")
    if value is None or float(value) <= 0:
        return None
    return float(value)


def _unscaled_tail(opt, before, before_unit):
    """The sentence a unit change owes the open batch when it has just split
    the ingredients across units: scaling needs one unit, so the batch is back
    to as-generated. Empties the box too — a number left in it would go on
    quietly meaning nothing."""
    if before is None or opt.one_amount_unit() is not None:
        return ""
    st.session_state.pop("scale_total", None)
    if opt.pending_batch_no is None:
        return ""
    total_text = join_unit(f"{before:g}", before_unit)
    return wording.unscaled_tail(opt.pending_batch_no, total_text)


def _variables(opt, storage):
    """Ingredients and process settings, in one open section. The form first, then one table of everything, then one row
    of controls, with the file upload folded away beneath.

    They were four places — an Ingredients subheader with its own uploader, a
    Change the ingredient list expander, a Process settings expander and an
    Ingredients fold — which asked the same question ('what changes between
    formulations?') in four different shapes."""
    st.subheader(wording.VARIABLES_HEADER)
    _add_variable(opt)
    _variable_table(opt)
    if getattr(opt, "amount_unit_backfilled", False):
        # The file this project was saved in predates the unit; its amounts
        # may have been percentages or millilitres, and nothing on screen
        # would otherwise say the g was the app's guess and not the user's.
        st.caption(wording.made_before_units_caption(opt.amount_unit))
    _variable_controls(opt, storage)
    with st.expander(wording.UPLOAD_INGREDIENTS_EXPANDER):
        _upload_ingredients(opt)


def _add_variable(opt):
    """One form for both types. Type is a radio rather than two forms: an
    ingredient and a setting are the same four answers — what it is called,
    how low, how high, and in what."""
    mid_run = bool(opt.X_history)
    st.session_state.setdefault("var_kind", KIND_INGREDIENT)
    kind = st.session_state["var_kind"]
    setting = kind == KIND_SETTING
    # The unit box opens on what that type is written in: g for an ingredient,
    # blank for a setting, because a cook temperature is never 175 g. Assigned
    # before the box is created, which is the one moment Streamlit allows it.
    if st.session_state.get("_var_kind_shown") != kind:
        st.session_state["_var_kind_shown"] = kind
        st.session_state["var_unit"] = "" if setting else (opt.amount_unit or "")
    wants_baseline = setting and mid_run
    widths = [2, 2, 1, 1, 1] + ([1] if wants_baseline else [])
    cols = st.columns(widths)
    with cols[0]:
        st.session_state.setdefault("var_name", "")
        st.text_input(wording.NAME_LABEL, key="var_name",
                      placeholder=(wording.SETTING_NAME_PLACEHOLDER if setting
                                   else wording.INGREDIENT_NAME_PLACEHOLDER))
    with cols[1]:
        st.radio(wording.TYPE_LABEL, [KIND_INGREDIENT, KIND_SETTING], key="var_kind",
                 horizontal=True,
                 help=wording.VARIABLE_TYPE_HELP)
    with cols[2]:
        # The opening value comes from session state, never from a `value=`
        # argument: a project switch assigns these keys (see app.py's
        # _FORM_FRESH), and Streamlit warns on screen when a widget is given
        # both a default and a session-state value.
        st.session_state.setdefault("var_low", 0.0)
        if mid_run and not setting:
            # Fixed at 0 and shown as 0: a number left in the box by a
            # setting typed a moment ago would go on to be sent as the
            # ingredient's lowest, which the box says it cannot be.
            st.session_state["var_low"] = 0.0
        st.number_input(
            wording.LOWEST_LABEL, key="var_low", disabled=mid_run and not setting,
            help=(wording.NEW_INGREDIENT_FIXED_LOW_HELP
                  if mid_run and not setting else None),
        )
    with cols[3]:
        st.session_state.setdefault("var_high", 100.0)
        st.number_input(wording.HIGHEST_LABEL, key="var_high")
    with cols[4]:
        st.session_state.setdefault("var_unit", opt.amount_unit or "")
        st.text_input(wording.UNIT_LABEL, key="var_unit",
                     placeholder=wording.VARIABLE_UNIT_PLACEHOLDER)
    if wants_baseline:
        with cols[5]:
            st.session_state.setdefault("var_base", None)
            st.number_input(
                wording.BASELINE_LABEL, key="var_base",
                placeholder=wording.BASELINE_REQUIRED_PLACEHOLDER,
                help=wording.BASELINE_HELP,
            )
    # One box per property, on their own row: a property is an ingredient's
    # value, so a process setting is never asked for one — and neither is this
    # form while Set property values is open below it, or the same property
    # would have two boxes on one screen.
    editing_values = st.session_state.get("_props_for") is not None
    properties = [] if (setting or editing_values) else opt.properties()
    if properties:
        prop_cols = st.columns(min(4, len(properties)))
        for j, prop in enumerate(properties):
            with prop_cols[j % len(prop_cols)]:
                st.session_state.setdefault(_prop_key(prop), None)
                st.number_input(prop, key=_prop_key(prop),
                               placeholder=wording.NO_VALUE_PLACEHOLDER)
    # Grey: the tab's one coloured button is Continue at the foot.
    if st.button(wording.ADD_VARIABLE_BUTTON, key="add_variable"):
        _add_variable_now(opt, setting, wants_baseline, properties)


def _prop_key(prop):
    """The add form's box for one property. Keyed by name, so a project switch
    empties it (app.py parks every var_prop_ key)."""
    return f"var_prop_{prop}"


def _set_typed_properties(opt, name, properties):
    """Write the property values typed on the add form, and empty the boxes.
    Nothing is written for a box left blank: no value is not 0, and the limit
    line says which ingredients have none."""
    for prop in properties:
        value = st.session_state.get(_prop_key(prop))
        if value is not None:
            opt.set_property_value(name, prop, value)
        park_clear(_prop_key(prop), None)


def _add_variable_now(opt, setting, wants_baseline, properties=()):
    name = st.session_state["var_name"]
    low, high = st.session_state["var_low"], st.session_state["var_high"]
    unit = st.session_state.get("var_unit", "")
    batch_no = opt.pending_batch_no
    if setting:
        if wants_baseline and st.session_state.get("var_base") is None:
            st.error(wording.ADD_BASELINE_ERROR)
            return
        try:
            opt.add_process_parameter(
                name, low, high,
                baseline=(st.session_state.get("var_base")
                          if wants_baseline else None),
                unit=unit,
            )
        except ValueError as e:
            st.error(str(e))
            return
        if saved_ok(opt):
            flash("success", wording.added(str(name).strip()))
            _note_discarded_batch(opt, batch_no)
            st.rerun()
        return

    scaled, scaled_unit = _scaled_now(opt), opt.one_amount_unit()
    try:
        # Adding a name the project already has is an edit, and it can set
        # that ingredient's unit — so it can leave an amount limit adding
        # grams to millilitres, exactly as Set unit can.
        removed = opt.add_ingredient(name, low, high, unit=unit)
    except ValueError as e:
        st.error(str(e))
        return
    if saved_ok(opt):
        _set_typed_properties(opt, str(name).strip(), properties)
        if not saved_ok(opt):
            return
        added_line = wording.added(str(name).strip())
        tail = _unscaled_tail(opt, scaled, scaled_unit)
        flash("success", f"{added_line} {tail}" if tail else added_line)
        _flash_removed_limits(opt, removed)
        _note_discarded_batch(opt, batch_no)
        st.rerun()


def _ordered_variables(opt):
    """Ingredients first, then process settings, each in the order they were
    added. The table and the picker beneath it read the same way down."""
    ingredients = [v for v in opt.variables
                   if v.get('category', 'ingredient') == 'ingredient']
    settings = [v for v in opt.variables if v.get('category') == 'process']
    return ingredients + settings


def _held_at(opt, var):
    """What a paused row is held at in every new formulation. A cook
    temperature is dialled in, an ingredient is weighed out, and each is
    written in its own unit — a paused setting 'held at 175 g' priced a
    setting in grams."""
    unit = opt.unit_of(var['name'])
    value = opt._frozen_value(var)
    if var.get('category') == 'process':
        return fmt_setting(value, unit)
    return fmt_amount(value, unit)


def _property_cell(opt, var, prop):
    """One ingredient's value for one property, as the table writes it."""
    if var.get('category', 'ingredient') != 'ingredient':
        return ""
    if not opt.has_property_value(var['name'], prop):
        return ""
    return f"{opt.property_value(var['name'], prop):g}"


def _variable_table(opt):
    rows = _ordered_variables(opt)
    if not rows:
        return
    properties = opt.properties()
    # A column that says the same thing on every row is a column of noise, so
    # Status arrives with the first paused row and Baseline with the first
    # setting that has one.
    any_paused = any(not v.get('active', True) for v in rows)
    any_baseline = any(v.get('_absent_value') is not None for v in rows)
    # The three headers are the add form's own labels, so the table and the
    # boxes above it name the same four answers with the same four words.
    frame = pd.DataFrame([{
        wording.TYPE_LABEL: (KIND_INGREDIENT
                             if v.get('category', 'ingredient') == 'ingredient'
                             else KIND_SETTING),
        wording.NAME_LABEL: v['name'],
        wording.LOWEST_LABEL: float(v['bounds'][0]),
        wording.HIGHEST_LABEL: float(v['bounds'][1]),
        # Plain Lowest and Highest with a Unit column of their own: "Lowest
        # (g)" over a row measured in ml was a lie, and the water really is
        # in ml.
        wording.UNIT_LABEL: opt.unit_of(v['name']),
        **({wording.BASELINE_LABEL: (fmt_setting(v.get('_absent_value'),
                                     opt.unit_of(v['name']))
                         if v.get('_absent_value') is not None else "")}
           if any_baseline else {}),
        **({wording.STATUS_LABEL: (wording.ACTIVE_STATUS if v.get('active', True)
                       else wording.paused_status(_held_at(opt, v)))}
           if any_paused else {}),
        # One column per property, blank where an ingredient has no value —
        # a 0 is a value and must not read like a gap. A process setting is
        # weighed into nothing, so its cells are blank too.
        **{prop: _property_cell(opt, v, prop) for prop in properties},
    } for v in rows])
    st.dataframe(frame, hide_index=True, key="variable_table",
                 height=table_height(len(frame), max_rows=20))


def _disarm_other_removals(pick):
    """An armed Delete belongs to the row it was armed on. Changing the pick
    would otherwise leave a confirmation armed with nothing on screen to
    answer it, and the tab's Continue greyed behind it for ever."""
    armed = armed_confirmation()
    if armed and armed.startswith("rm_var_") and armed != f"rm_var_{pick}":
        disarm(armed)


def _variable_controls(opt, storage):
    """One row for everything you can do to a row of the table: pause it or
    resume it, set its unit, delete it."""
    rows = _ordered_variables(opt)
    if not rows:
        return
    properties = opt.properties()
    widths = [2.4, 1, 1.2, 1, 1.6] + ([1.6] if properties else [])
    cols = st.columns(widths)
    with cols[0]:
        pick = st.selectbox(wording.INGREDIENT_OR_SETTING_LABEL,
                            [v['name'] for v in rows],
                            key="var_pick")
    var = opt._var_by_name(pick)
    is_ingredient = var.get('category', 'ingredient') == 'ingredient'
    _disarm_other_removals(pick)
    with cols[1]:
        _pause_or_resume(opt, var, pick)
    with cols[2]:
        st.session_state.setdefault("unit_value", "")
        # "New unit", not "Unit": the add form above has a Unit box of its
        # own, and two of them on one row asked the reader which was which.
        # The placeholder is the unit the picked row is in today.
        typed = st.text_input(wording.NEW_UNIT_LABEL, key="unit_value",
                              placeholder=opt.unit_of(pick) or "g")
    with cols[3]:
        if st.button(wording.SET_UNIT_BUTTON, key="set_unit"):
            _set_unit_now(opt, pick, typed)
    with cols[4]:
        _remove_variable(opt, storage, pick, is_ingredient)
    if properties:
        with cols[5]:
            # Disabled rather than hidden for a setting: a control that comes
            # and goes as the pick changes reads as a fault in the app.
            if st.button(wording.SET_PROPERTY_VALUES_BUTTON, key="set_props",
                         disabled=not is_ingredient,
                         help=(wording.ONLY_INGREDIENT_HAS_PROPERTIES
                               if not is_ingredient else None)):
                st.session_state["_props_for"] = pick
                # Seeded here, the one moment a widget's value can be set:
                # the boxes do not exist yet on the run that follows.
                for prop in properties:
                    park_clear(_pkey(pick, prop),
                               float(opt.property_value(pick, prop))
                               if opt.has_property_value(pick, prop) else None)
                st.rerun()
        if st.session_state.get("_props_for") == pick and is_ingredient:
            _property_value_editor(opt, pick, properties)


def _pkey(pick, prop):
    return f"setprop_{pick}_{prop}"


def _property_value_editor(opt, pick, properties):
    """One box per property for the picked ingredient, opened by Set property
    values. Blank means no value, which counts as 0 in the per-100 average —
    and every limit on that property names the ingredients it is reading as
    zeroes."""
    st.caption(wording.values_for_caption(pick))
    boxes = st.columns(min(4, len(properties)))
    for j, prop in enumerate(properties):
        with boxes[j % len(boxes)]:
            st.session_state.setdefault(_pkey(pick, prop), None)
            st.number_input(prop, key=_pkey(pick, prop),
                           placeholder=wording.NO_VALUE_PLACEHOLDER)
    b1, b2 = st.columns(2)
    with b1:
        if st.button(wording.SAVE_VALUES_BUTTON, key="save_props", use_container_width=True):
            for prop in properties:
                opt.set_property_value(pick, prop,
                                       st.session_state.get(_pkey(pick, prop)))
            if saved_ok(opt):
                flash("success", wording.property_values_saved(pick))
                st.session_state.pop("_props_for", None)
                st.rerun()
    with b2:
        if st.button(wording.CLOSE_BUTTON, key="close_props", use_container_width=True):
            st.session_state.pop("_props_for", None)
            st.rerun()


def _pause_or_resume(opt, var, pick):
    """Whichever of the two applies to the row that is picked. A paused row is
    left out of new formulations; nothing is removed."""
    batch_no = opt.pending_batch_no
    if not var.get('active', True):
        if st.button(wording.RESUME_BUTTON, key="resume_var",
                     help=wording.RESUME_HELP):
            opt.reactivate_variable(pick)
            if saved_ok(opt):
                flash("success", wording.resumed(pick))
                _note_discarded_batch(opt, batch_no)
                st.rerun()
        return
    alone = len(opt.active_variables()) <= 1
    if st.button(wording.PAUSE_BUTTON, key="pause_var", disabled=alone,
                 help=(wording.PAUSE_DISABLED_HELP if alone else
                       wording.PAUSE_HELP)):
        try:
            opt.deactivate_variable(pick)
        except ValueError as e:
            st.error(str(e))
        else:
            if saved_ok(opt):
                flash("success", wording.paused(pick))
                _note_discarded_batch(opt, batch_no)
                st.rerun()


def _set_unit_now(opt, pick, typed):
    """Change one row's unit, ingredient or process setting. Nothing is
    rescored and the open batch stands: a unit is how a number is written,
    not the number."""
    if not str(typed).strip():
        # A blank box looks like a no-op and is not one: it would rewrite the
        # ingredient to no unit at all, and could take an amount limit with it.
        st.error(wording.UNIT_REQUIRED_ERROR)
        return
    scaled, scaled_unit = _scaled_now(opt), opt.one_amount_unit()
    try:
        removed = opt.set_variable_unit(pick, typed)
    except ValueError as e:
        st.error(str(e))
        return
    if saved_ok(opt):
        written = opt.unit_of(pick)
        # What changed is how the number is written, not the number: nothing
        # is converted and nothing is rescored, and only this sentence says so.
        # An ingredient has amounts; a process setting has one value, and
        # "the amounts" named something a cook temperature does not have.
        var = opt._var_by_name(pick)
        is_ingredient = var.get('category', 'ingredient') == 'ingredient'
        said = wording.unit_changed(pick, written, is_ingredient)
        # Scaling needs one unit, and this change may have taken it away; the
        # batch is back to as-generated, so say so.
        tail = _unscaled_tail(opt, scaled, scaled_unit)
        flash("success", f"{said} {tail}" if tail else said)
        # An amount limit is a sum, and this change may have left one adding
        # grams to millilitres. It is gone; say which.
        _flash_removed_limits(opt, removed)
        # Emptied for the next ingredient: a unit left in the box is one click
        # away from being applied to another row.
        park_clear("unit_value", "")
        st.rerun()


def _remove_variable(opt, storage, pick, is_ingredient):
    """Delete one row for good. Always confirmed, always copied first, history
    or not: a deletion is a deletion and the user is told the same thing every
    time."""
    key = f"rm_var_{pick}"
    confirmed = confirm_action(
        key, wording.delete_button(pick),
        wording.delete_variable_warning(pick, is_ingredient),
        confirm_label=wording.YES_DELETE,
        disabled=other_confirmation(key),
    )
    # Read here, before the tick box below is cleared: the run that confirms
    # is the run that disarms.
    force = bool(st.session_state.get("delete_ing_force", False))
    armed = (armed_confirmation() == key
             and st.session_state.get(f"{key}__pending"))
    # The tick box is drawn AFTER confirm_action, because the click that arms
    # the confirmation is only recorded inside it: asking first showed the box
    # one click late. It is rarely needed and only ever seen while a deletion
    # is armed — deleting an ingredient that was used above 0 would rewrite
    # formulations nobody made.
    if armed and is_ingredient:
        st.caption(wording.DELETE_VS_PAUSE_CAPTION)
        st.checkbox(wording.DELETE_EVEN_IF_USED_CHECKBOX,
                    key="delete_ing_force")
    elif not armed:
        # Never carried into the next deletion, or the next project: a tick
        # left behind is one click away from discarding real results.
        st.session_state.pop("delete_ing_force", None)
    if confirmed:
        batch_no = opt.pending_batch_no
        try:
            storage.archive(opt.project_name, "pre_delete", copy=True)
            if is_ingredient:
                opt.remove_ingredient(pick, force=force)
            else:
                opt.remove_process_parameter(pick)
        except (ValueError, storage_backend.StorageError) as e:
            st.error(str(e))
        else:
            if saved_ok(opt):
                flash("success", wording.deleted(pick))
                _note_discarded_batch(opt, batch_no)
                st.rerun()


def _load_label(opt):
    """What the button under an uploaded file does. It does not add to the
    list, it replaces it — every ingredient not in the file is dropped — and
    a button that said Load over a project's own eight ingredients did not
    say so."""
    return (wording.REPLACE_INGREDIENTS_BUTTON if opt.has_ingredients()
            else wording.LOAD_INGREDIENTS_BUTTON)


def _upload_ingredients(opt):
    """The ingredient file, folded away: typing one ingredient is the common
    case, and a file is the shortcut for a project that already has one."""
    st.caption(wording.INGREDIENTS_CSV_CAPTION)
    uploaded = st.file_uploader(
        wording.UPLOAD_INGREDIENTS_CSV_LABEL, type=["csv"],
        # Keyed to the project: a file uploader cannot be emptied from session
        # state, so a shared key handed the next project the sheet this one
        # loaded, with a live Load ingredients under it.
        key=f"ingredients_csv_{opt.project_name}",
        # The caption above lists the columns; the one thing it does not say
        # is what a blank Unit cell means, which is this project's own unit.
        help=wording.blank_unit_cell_help(opt.amount_unit),
    )
    if os.path.exists(_SAMPLE_CSV):
        with open(_SAMPLE_CSV, "rb") as handle:
            st.download_button(wording.DOWNLOAD_CSV_TEMPLATE, data=handle.read(),
                               file_name="ingredients_template.csv",
                               mime="text/csv", key="ingredients_template")
    df = None
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception:
            st.error(wording.CSV_UNREADABLE_RETRY)
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
    each. Three edits can do it — a unit set on one ingredient, a new default
    unit, a reloaded ingredient file — and all three say it the same way."""
    for qc in removed:
        if 'metric' in qc:
            # A property limit is an average over the amounts, so it is the
            # ingredients as a whole that stopped sharing a unit — there is no
            # list of its own to name.
            flash("warning", wording.property_limit_removed(qc['metric']))
            continue
        label = _limit_label(opt, qc)
        if qc.get('reason') == 'missing':
            gone = qc.get('missing') or []
            many = len(gone) > 1
            who = wording.no_longer_ingredients(
                number_list(gone) if many else gone[0], many)
            flash("warning", wording.quantity_limit_removed_missing(label, who))
        else:
            flash("warning", wording.quantity_limit_removed_unit_mismatch(label))


def _limit_who(opt, qc):
    """How one ingredient limit is named inside a sentence — 'all
    ingredients' or 'Water + Oil'. The same list _limit_label heads the row
    with, in the register a sentence needs."""
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
    """How one ingredient limit is named on screen — 'All ingredients' or
    'Water + Oil'. The list under Limits and the line that reports a limit
    removed both read from here, so they name it alike."""
    names = [v['name'] for v in opt.variables
             if v.get('category', 'ingredient') == 'ingredient']
    if names and set(qc['ingredients']) == set(names):
        return wording.ALL_INGREDIENTS_LABEL
    return " + ".join(qc['ingredients'])


def _measurement_editor(opt, storage, editing):
    """The add/edit fields. `editing` is the measurement being changed, or
    None when adding a new one. No st.form: the Target box greys itself out
    the moment the goal changes, which a form would defer to its submit."""
    if editing is None:
        st.session_state.setdefault(_mkey(None, "name"), "")
        name = st.text_input(wording.NAME_LABEL, key=_mkey(None, "name"),
                             placeholder=wording.MEASUREMENT_NAME_PLACEHOLDER)
    else:
        st.session_state.setdefault(_mkey(editing, "name"), editing['name'])
        st.text_input(wording.NAME_LABEL, key=_mkey(editing, "name"), disabled=True)
        name = editing['name']

    c1, c2 = st.columns(2)
    with c1:
        st.session_state.setdefault(_mkey(editing, "unit"),
                                    (editing or {}).get('unit', ""))
        unit = st.text_input(wording.UNIT_LABEL, key=_mkey(editing, "unit"),
                             placeholder=wording.MEASUREMENT_UNIT_PLACEHOLDER)
    with c2:
        st.session_state.setdefault(_mkey(editing, "goal"),
                                    (editing or {}).get('goal', "max"))
        goal = st.selectbox(wording.GOAL_LABEL, list(wording.GOAL_LABELS),
                            key=_mkey(editing, "goal"),
                            format_func=wording.GOAL_LABELS.get,
                            help=wording.GOAL_SELECT_HELP)

    st.session_state.setdefault(_mkey(editing, "target"),
                                float((editing or {}).get('target') or 0.0))
    target = st.number_input(wording.TARGET_LABEL, key=_mkey(editing, "target"),
                             disabled=(goal != 'target'))

    st.markdown(wording.RANGE_HEADING)
    s1, s2 = st.columns(2)
    with s1:
        st.session_state.setdefault(_mkey(editing, "min"),
                                    float((editing or {}).get('min_val', 0.0)))
        lowest = st.number_input(wording.LOWEST_MEASURABLE_LABEL, key=_mkey(editing, "min"))
    with s2:
        st.session_state.setdefault(_mkey(editing, "max"),
                                    float((editing or {}).get('max_val', 10.0)))
        highest = st.number_input(wording.HIGHEST_MEASURABLE_LABEL, key=_mkey(editing, "max"))
    st.caption(wording.RANGE_HINT_CAPTION)

    st.session_state.setdefault(_mkey(editing, "importance"),
                                float((editing or {}).get('weight', 1.0)))
    importance = st.number_input(
        wording.IMPORTANCE_LABEL, min_value=0.1, max_value=100.0, step=0.1,
        key=_mkey(editing, "importance"),
        help=wording.IMPORTANCE_HELP,
    )

    if editing is None:
        if st.button(wording.ADD_MEASUREMENT_BUTTON, key="add_measurement"):
            # add_objective REPLACES a measurement of the same name, which
            # would silently overwrite its goal, target and range and rescore
            # every result with no copy kept. Editing is a different door.
            if any(str(name).strip().lower() == o['name'].lower()
                   for o in opt.objectives):
                st.error(wording.MEASUREMENT_EXISTS_ERROR)
                return
            try:
                opt.add_objective(
                    name, importance, goal,
                    target=(target if goal == 'target' else None),
                    min_val=lowest, max_val=highest, unit=unit,
                )
            except (ValueError, storage_backend.StorageError) as e:
                st.error(str(e))
            else:
                if not saved_ok(opt):
                    return
                _clear_measurement_keys(None)
                flash("success", wording.added(str(name).strip()))
                st.rerun()
        return

    b1, b2 = st.columns(2)
    # The edit being made is the one thing to do while its row is open, as a
    # correction is on tab 3; the foot's Continue steps aside (it would
    # navigate away and throw the edit away).
    lit = not confirmation_open()
    with b1:
        save = st.button(wording.SAVE_CHANGES_BUTTON, key="save_measurement",
                         type="primary" if lit else "secondary",
                         disabled=not lit, use_container_width=True) and lit
    with b2:
        if st.button(wording.CANCEL, key="cancel_measurement", use_container_width=True):
            _clear_measurement_keys(editing)
            st.session_state.pop("_editing_measurement", None)
            st.rerun()
    if save:
        _apply_measurement_edit(opt, storage, editing, importance, goal, target,
                                lowest, highest, unit)


def _rescores(editing, importance, goal, target, lowest, highest):
    """True when this edit changes how every stored result scores. Importance
    is not the only such field: the goal, the target and either end of the
    range all feed closeness, so all of them recalculate the history."""
    def moved(before, after):
        return abs(float(after) - float(before)) > 1e-9
    if moved(editing['weight'], importance):
        return True
    if goal != editing['goal']:
        return True
    before_target = editing.get('target')
    after_target = target if goal == 'target' else None
    if (before_target is None) != (after_target is None):
        return True
    if before_target is not None and moved(before_target, after_target):
        return True
    return (moved(editing.get('min_val'), lowest)
            or moved(editing.get('max_val'), highest))


def _apply_measurement_edit(opt, storage, editing, importance, goal, target,
                            lowest, highest, unit):
    # editing is the live dict; update_objective mutates it, so read it first.
    changed_importance = abs(float(importance) - float(editing['weight'])) > 1e-9
    rescores = _rescores(editing, importance, goal, target, lowest, highest)
    before = best_formulation_no(opt)
    if rescores:
        # No history guard: a copy is kept before every destructive action
        # here, so the user always has one door back.
        try:
            storage.archive(opt.project_name, "pre_edit", copy=True)
        except storage_backend.StorageError as e:
            st.error(str(e))
            return
    try:
        opt.update_objective(
            editing['name'], weight=importance, goal=goal,
            target=(target if goal == 'target' else None),
            min_val=lowest, max_val=highest, unit=unit,
        )
    except ValueError as e:
        st.error(str(e))
        return
    if not saved_ok(opt):
        return
    _clear_measurement_keys(editing)
    st.session_state.pop("_editing_measurement", None)
    if rescores:
        after = best_formulation_no(opt)
        # Nothing has been scored yet: there is no overall score to
        # recalculate, and saying otherwise invents a history.
        recalculated = wording.RECALCULATED_SUFFIX if opt.Y_history else ""
        sentence = (
            wording.importance_changed(editing['name'], importance)
            if changed_importance else wording.updated(editing['name'])
        ) + recalculated
        parts = [sentence, best_move_sentence(before, after),
                 # This edit has no confirmation before it, so the copy it
                 # kept is named here, as a correction names its own.
                 COPY_KEPT]
        flash("success", " ".join(p for p in parts if p))
    else:
        flash("success", wording.updated(editing['name']))
    st.rerun()


def _remove_measurement(opt, storage, name):
    before = best_formulation_no(opt)
    try:
        storage.archive(opt.project_name, "pre_edit", copy=True)
    except storage_backend.StorageError as e:
        st.error(str(e))
        return
    opt.remove_objective(name)
    if not saved_ok(opt):
        return
    after = best_formulation_no(opt)
    sentence = wording.measurement_deleted(name) + (
        wording.RECALCULATED_SUFFIX if opt.Y_history else "")
    move = best_move_sentence(before, after)
    flash("success", f"{sentence} {move}".strip())
    st.rerun()


def _measurements(opt, storage):
    """Draw the measurements section. Returns True while a measurement is
    open for editing: `Save changes` is then the tab's one lit action and the
    foot steps aside."""
    st.subheader(wording.MEASUREMENTS_HEADER)
    editing_name = st.session_state.get("_editing_measurement")
    editing = next((o for o in opt.objectives if o['name'] == editing_name), None)
    if editing is not None:
        _measurement_editor(opt, storage, editing)
    elif not opt.objectives:
        _measurement_editor(opt, storage, None)
    else:
        with st.expander(wording.ADD_A_MEASUREMENT_EXPANDER):
            _measurement_editor(opt, storage, None)

    if not opt.objectives:
        return editing is not None

    ordered = opt.measurements_by_importance()
    st.dataframe(pd.DataFrame([{
        # No Priority column: it was the row's position in a table already
        # sorted by importance, which is the same fact written twice.
        wording.MEASUREMENT_COLUMN: label_with_unit(o['name'], o.get('unit')),
        wording.GOAL_LABEL: _goal_text(o),
        wording.RANGE_COLUMN: _range_text(o),
        wording.IMPORTANCE_LABEL: float(o['weight']),
        wording.COL_SHARE: opt.share_text(o['name']),
    } for o in ordered]), hide_index=True, key="measurement_table",
        height=table_height(len(ordered)))
    for obj in ordered:
        e1, e2 = st.columns(2)
        with e1:
            if st.button(wording.edit_button(obj['name']), key=f"edit_meas_{obj['name']}"):
                st.session_state["_editing_measurement"] = obj['name']
                st.rerun()
        with e2:
            if confirm_action(
                f"rm_meas_{obj['name']}", wording.delete_button(obj['name']),
                wording.delete_measurement_warning(obj['name']),
                confirm_label=wording.YES_DELETE,
                disabled=other_confirmation(f"rm_meas_{obj['name']}"),
            ):
                _remove_measurement(opt, storage, obj['name'])

    st.caption(opt.score_function_line())

    with st.expander(wording.HOW_IT_WORKS_EXPANDER):
        st.markdown("\n".join(
            ("    - " if i in HOW_IT_WORKS_NESTED else "- ") + line
            for i, line in enumerate(HOW_IT_WORKS)))
    return editing is not None


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
    """'· Water has no value and counts as 0.' — the ingredients this limit
    is silently reading as zeroes. A limit that looks satisfied because half
    the recipe was never given a value is the one way this arithmetic lies."""
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

        # ONE amount limit. There used to be two controls for one idea: a
        # group limit whose picker, with every ingredient ticked, wrote
        # exactly what the second control wrote. The picker's empty state is
        # now every ingredient, which is the common case and reads as one.
        st.markdown(wording.LIMIT_ON_CHOSEN_INGREDIENTS_HEADING)
        picked = st.multiselect(wording.INGREDIENTS_TO_LIMIT_LABEL, names,
                                key="qty_pick", placeholder=wording.ALL_INGREDIENTS_LABEL)
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
        if st.button(wording.ADD_INGREDIENT_LIMIT_BUTTON, key="add_amount_limit"):
            low, high = st.session_state["qc_min"], st.session_state["qc_max"]
            if low is None and high is None:
                st.error(wording.ENTER_LOWEST_HIGHEST_ERROR)
            else:
                try:
                    if picked:
                        opt.add_quantity_constraint(picked, min_val=low,
                                                    max_val=high)
                    else:
                        opt.add_total_mass_constraint(min_val=low, max_val=high)
                except ValueError as e:
                    st.error(str(e))
                else:
                    who = " + ".join(picked) if picked else wording.ALL_INGREDIENTS_LOWER
                    _report_limit(opt, wording.limit_added_on(who))

        for i, qc in enumerate(getattr(opt, "quantity_constraints", [])):
            label = _limit_label(opt, qc)
            # A limit sums ingredients that share a unit, so the
            # limit is written in it: "at most 400 g", never a bare 400.
            limited = {opt.unit_of(n) for n in qc['ingredients']}
            qc_unit = limited.pop() if len(limited) == 1 else ""
            bounds = ([join_unit(wording.at_least(qc['min']), qc_unit)]
                      if qc['min'] is not None else [])
            bounds += ([join_unit(wording.at_most(qc['max']), qc_unit)]
                       if qc['max'] is not None else [])
            l1, l2 = st.columns([3, 1])
            with l1:
                st.text(f"{label}: {' and '.join(bounds)}")
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


def _foot(opt, editing=False):
    ready, missing = readiness(opt)
    # While a confirmation is armed, its "Yes" is the one coloured button and
    # answering it is the one thing to do; moving on can wait a click. An open
    # measurement editor is the same case: Save changes is the lit one, and
    # Continue would leave the tab and throw the edit away.
    lit = ready and not confirmation_open() and not editing
    if st.button(wording.NEXT_MAKE_BATCH_BUTTON,
                 type="primary" if lit else "secondary",
                 disabled=not lit, key="continue_to_batch") and lit:
        go_to_tab(TAB_BATCH)
    if not ready:
        st.caption(missing)


def render(opt, storage):
    _variables(opt, storage)
    st.divider()
    editing = _measurements(opt, storage)
    st.divider()
    _limits(opt, storage)
    _advanced(opt)
    st.divider()
    _foot(opt, editing)
