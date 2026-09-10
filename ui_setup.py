"""Tab 1 · Set up: what the project can vary and what will be measured.

One column, in the order a formulator fills it in: what you can vary — the
ingredients and the process settings together — then the measurements, then
the optional sections. Exactly one coloured button lives here — `Continue to
make a batch` at the foot.
"""
import json
import os

import pandas as pd
import streamlit as st

import storage as storage_backend
from ui_helpers import (
    ARMED_KEY, TAB_BATCH, armed_confirmation, best_formulation_no,
    best_move_sentence, confirm_action, confirmation_open, flash, fmt_amount,
    fmt_setting, go_to_tab, join_unit, label_with_unit, number_list,
    other_confirmation, park_clear, plural, readiness, saved_ok, table_height,
    unit_after_number,
)

GOAL_LABELS = {
    "max": "Higher is better",
    "min": "Lower is better",
    "target": "Hit a target",
}

_SAMPLE_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "sample_ingredients.csv")

_LIMIT_KEPT = ("Formulations already made are kept. The next batch will "
               "respect this limit.")

# food_bo drops the open batch inside add_ingredient, deactivate_variable,
# add_process_parameter and friends, so app.py's makeability check never sees
# the mismatch. Every handler here that can change the ingredient list, a
# process setting or an allowed amount says so itself.
_BATCH_DISCARDED = ("The open batch was discarded because the ingredient list "
                    "or its allowed amounts changed since it was generated.")

# The one sentence on any screen that says "weight": the technical gloss at
# the foot of How closeness is worked out, which is there to join the word
# the tab uses to the one a statistician would.
_GLOSS = ("Importance is the weight of each measurement in the overall score; "
          "closeness is its normalised score between 0 and 1.")


# ------------------------------------------------------------------ #
#  Small shared text
# ------------------------------------------------------------------ #

def _unit_suffix(unit):
    return f" ({unit})" if unit else ""


def _note_discarded_batch(opt, batch_no_before):
    """Flash the notice when the write just now retired the open batch."""
    if batch_no_before is not None and opt.pending_batch_no is None:
        flash("info", _BATCH_DISCARDED)


def _goal_text(obj):
    """'Target 6 N', 'Higher is better', 'Lower is better'. A '/10' rides on
    the measurement's own name instead of on every number in its row."""
    if obj['goal'] == 'target':
        return join_unit(f"Target {float(obj['target']):g}",
                         unit_after_number(obj.get('unit')))
    return GOAL_LABELS.get(obj['goal'], obj['goal'])


def _scale_text(obj):
    return join_unit(f"{float(obj['min_val']):g} to {float(obj['max_val']):g}",
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
    flash("success", f"{sentence} {_LIMIT_KEPT}")
    if _nothing_made_fits(opt):
        flash("warning", "No formulation you have made fits this limit.")
    st.rerun()


# ------------------------------------------------------------------ #
#  Sections
# ------------------------------------------------------------------ #

KIND_INGREDIENT = "Ingredient"
KIND_SETTING = "Process setting"


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
    return (f"Batch {opt.pending_batch_no} is no longer scaled to "
            + join_unit(f"{before:g}", before_unit)
            + "; scaling needs all ingredients in one unit.")


def _variables(opt, storage):
    """What you can vary: the ingredients and the process settings, in one
    open section. The form first, then one table of everything, then one row
    of controls, with the file upload folded away beneath.

    They were four places — an Ingredients subheader with its own uploader, a
    Change the ingredient list expander, a Process settings expander and an
    Ingredients fold — which asked the same question ('what changes between
    formulations?') in four different shapes."""
    st.subheader("What you can vary")
    st.caption("Ingredients and process settings you will change between "
               "formulations.")
    _add_variable(opt)
    _variable_table(opt)
    if getattr(opt, "amount_unit_backfilled", False):
        # The file this project was saved in predates the unit; its amounts
        # may have been percentages or millilitres, and nothing on screen
        # would otherwise say the g was the app's guess and not the user's.
        st.caption("Made before units were recorded; amounts are in "
                   f"{opt.amount_unit}. Set each unit below if that is "
                   "wrong.")
    _variable_controls(opt, storage)
    with st.expander("Or upload a list"):
        _upload_ingredients(opt)


def _add_variable(opt):
    """One form for both kinds. Kind is a radio rather than two forms: an
    ingredient and a setting are the same four answers — what it is called,
    how low, how high, and in what."""
    mid_run = bool(opt.X_history)
    st.session_state.setdefault("var_kind", KIND_INGREDIENT)
    kind = st.session_state["var_kind"]
    setting = kind == KIND_SETTING
    # The unit box opens on what that kind is written in: g for an ingredient,
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
        st.text_input("Name", key="var_name",
                      placeholder=("e.g. Cook temperature" if setting
                                   else "e.g. Water"))
    with cols[1]:
        st.radio("Kind", [KIND_INGREDIENT, KIND_SETTING], key="var_kind",
                 horizontal=True)
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
            "Lowest", key="var_low", disabled=mid_run and not setting,
            help=("A new ingredient starts at 0 in every formulation already "
                  "made, so its lowest is fixed at 0 for now."
                  if mid_run and not setting else None),
        )
    with cols[3]:
        st.session_state.setdefault("var_high", 100.0)
        st.number_input("Highest", key="var_high")
    with cols[4]:
        st.session_state.setdefault("var_unit", opt.amount_unit or "")
        st.text_input("Unit", key="var_unit", placeholder="°C, min, %")
    if wants_baseline:
        with cols[5]:
            st.session_state.setdefault("var_base", None)
            st.number_input(
                "Baseline", key="var_base", placeholder="required",
                help="The setting you used for every formulation already "
                     "made, so those results still count.",
            )
    # Grey: the tab's one coloured button is Continue at the foot.
    if st.button("Add", key="add_variable"):
        _add_variable_now(opt, setting, wants_baseline)


def _add_variable_now(opt, setting, wants_baseline):
    name = st.session_state["var_name"]
    low, high = st.session_state["var_low"], st.session_state["var_high"]
    unit = st.session_state.get("var_unit", "")
    batch_no = opt.pending_batch_no
    if setting:
        if wants_baseline and st.session_state.get("var_base") is None:
            st.error("Enter the baseline: the setting you used for every "
                     "formulation already made.")
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
            flash("success", f"Added {str(name).strip()}.")
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
        added = f"Added {str(name).strip()}."
        tail = _unscaled_tail(opt, scaled, scaled_unit)
        flash("success", f"{added} {tail}" if tail else added)
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


def _variable_table(opt):
    rows = _ordered_variables(opt)
    if not rows:
        return
    # A column that says the same thing on every row is a column of noise, so
    # Status arrives with the first paused row and Baseline with the first
    # setting that has one.
    any_paused = any(not v.get('active', True) for v in rows)
    any_baseline = any(v.get('_absent_value') is not None for v in rows)
    frame = pd.DataFrame([{
        "Kind": (KIND_INGREDIENT if v.get('category', 'ingredient') == 'ingredient'
                 else KIND_SETTING),
        "Name": v['name'],
        "Lowest": float(v['bounds'][0]),
        "Highest": float(v['bounds'][1]),
        # Plain Lowest and Highest with a Unit column of their own: "Lowest
        # (g)" over a row measured in ml was a lie, and the water really is
        # in ml.
        "Unit": opt.unit_of(v['name']),
        **({"Baseline": (fmt_setting(v.get('_absent_value'),
                                     opt.unit_of(v['name']))
                         if v.get('_absent_value') is not None else "")}
           if any_baseline else {}),
        **({"Status": ("active" if v.get('active', True)
                       else f"paused · held at {_held_at(opt, v)}")}
           if any_paused else {}),
    } for v in rows])
    st.dataframe(frame, hide_index=True, key="variable_table",
                 height=table_height(len(frame), max_rows=20))


def _disarm_other_removals(pick):
    """An armed Remove belongs to the row it was armed on. Changing the pick
    would otherwise leave a confirmation armed with nothing on screen to
    answer it, and the tab's Continue greyed behind it for ever."""
    armed = armed_confirmation()
    if armed and armed.startswith("rm_var_") and armed != f"rm_var_{pick}":
        st.session_state.pop(f"{armed}__pending", None)
        st.session_state.pop(ARMED_KEY, None)


def _variable_controls(opt, storage):
    """One row for everything you can do to a row of the table: pause it or
    resume it, set its unit, remove it."""
    rows = _ordered_variables(opt)
    if not rows:
        return
    c1, c2, c3, c4, c5 = st.columns([2.4, 1, 1.2, 1, 1.6])
    with c1:
        pick = st.selectbox("Ingredient or setting", [v['name'] for v in rows],
                            key="var_pick")
    var = opt._var_by_name(pick)
    is_ingredient = var.get('category', 'ingredient') == 'ingredient'
    _disarm_other_removals(pick)
    with c2:
        _pause_or_resume(opt, var, pick)
    with c3:
        st.session_state.setdefault("unit_value", "")
        # "New unit", not "Unit": the add form above has a Unit box of its
        # own, and two of them on one row asked the reader which was which.
        # The placeholder is the unit the picked row is in today.
        typed = st.text_input("New unit", key="unit_value",
                              placeholder=opt.unit_of(pick) or "g")
    with c4:
        if st.button("Set unit", key="set_unit"):
            _set_unit_now(opt, pick, typed)
    with c5:
        _remove_variable(opt, storage, pick, is_ingredient)


def _pause_or_resume(opt, var, pick):
    """Whichever of the two applies to the row that is picked. A paused row is
    left out of new formulations; nothing is deleted."""
    batch_no = opt.pending_batch_no
    if not var.get('active', True):
        if st.button("Resume", key="resume_var",
                     help="Put it back into new formulations."):
            opt.reactivate_variable(pick)
            if saved_ok(opt):
                flash("success", f"Resumed {pick}.")
                _note_discarded_batch(opt, batch_no)
                st.rerun()
        return
    alone = len(opt.active_variables()) <= 1
    if st.button("Pause", key="pause_var", disabled=alone,
                 help=("At least two ingredients or settings must stay active "
                       "before one can be paused." if alone else
                       "Leave it out of new formulations, keeping every "
                       "result already recorded.")):
        try:
            opt.deactivate_variable(pick)
        except ValueError as e:
            st.error(str(e))
        else:
            if saved_ok(opt):
                flash("success", f"Paused {pick}.")
                _note_discarded_batch(opt, batch_no)
                st.rerun()


def _set_unit_now(opt, pick, typed):
    """Change one row's unit, ingredient or process setting. Nothing is
    rescored and the open batch stands: a unit is how a number is written,
    not the number."""
    if not str(typed).strip():
        # A blank box looks like a no-op and is not one: it would rewrite the
        # ingredient to no unit at all, and could take an amount limit with it.
        st.error("Enter a unit, such as g or ml.")
        return
    scaled, scaled_unit = _scaled_now(opt), opt.one_amount_unit()
    try:
        removed = opt.set_variable_unit(pick, typed)
    except ValueError as e:
        st.error(str(e))
        return
    if saved_ok(opt):
        written = opt.unit_of(pick)
        said = (f"{pick} is measured in {written}." if written
                else f"{pick} is shown without a unit.")
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
    """Remove one row for good. Always confirmed, always copied first, history
    or not: a removal is a removal and the user is told the same thing every
    time."""
    key = f"rm_var_{pick}"
    warning = (f"Remove {pick} from this project permanently? " if is_ingredient
               else f"Remove {pick}? ")
    confirmed = confirm_action(
        key, "Remove",
        warning + "Formulations already made will be recorded without it. "
                  "A copy of the project is kept first.",
        confirm_label="Yes, remove",
        disabled=other_confirmation(key),
    )
    # Read here, before the tick box below is cleared: the run that confirms
    # is the run that disarms.
    force = bool(st.session_state.get("delete_ing_force", False))
    armed = (armed_confirmation() == key
             and st.session_state.get(f"{key}__pending"))
    # The tick box is drawn AFTER confirm_action, because the click that arms
    # the confirmation is only recorded inside it: asking first showed the box
    # one click late. It is rarely needed and only ever seen while a removal
    # is armed — removing an ingredient that was used above 0 would rewrite
    # formulations nobody made.
    if armed and is_ingredient:
        st.caption("Deleting removes it from every formulation already made; "
                   "pausing keeps the data.")
        st.checkbox("Delete even if it was used (discards that information)",
                    key="delete_ing_force")
    elif not armed:
        # Never carried into the next removal, or the next project: a tick
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
                flash("success", f"Removed {pick}.")
                _note_discarded_batch(opt, batch_no)
                st.rerun()


def _load_label(opt):
    """What the button under an uploaded file does. It does not add to the
    list, it replaces it — every ingredient not in the file is dropped — and
    a button that said Load over a project's own eight ingredients did not
    say so."""
    return "Replace ingredients" if opt.has_ingredients() else "Load ingredients"


def _upload_ingredients(opt):
    """The ingredient file, folded away: typing one ingredient is the common
    case, and a file is the shortcut for a project that already has one."""
    st.caption("A CSV with the columns Name, Min, Max and, optionally, Unit. "
               "Extra columns become properties you can set limits on.")
    uploaded = st.file_uploader(
        "Upload ingredients CSV", type=["csv"],
        # Keyed to the project: a file uploader cannot be emptied from session
        # state, so a shared key handed the next project the sheet this one
        # loaded, with a live Load ingredients under it.
        key=f"ingredients_csv_{opt.project_name}",
        # The caption above lists the columns; the one thing it does not say
        # is what a blank Unit cell means, which is this project's own unit.
        help=f"A blank Unit cell is in {opt.amount_unit}.",
    )
    if os.path.exists(_SAMPLE_CSV):
        with open(_SAMPLE_CSV, "rb") as handle:
            st.download_button("Download CSV template", data=handle.read(),
                               file_name="ingredients_template.csv",
                               mime="text/csv", key="ingredients_template")
    df = None
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception:
            st.error("This file could not be read as a CSV. If it came from "
                     "Excel, use File > Save As and pick CSV format, then "
                     "try again.")
    # The file itself is left alone. Taking it out of the uploader changed
    # the widget's identity, which shut every open expander on the page under
    # the user's hands; remembering which file was loaded stops a second Load
    # just as well.
    mark = None if uploaded is None else (
        getattr(uploaded, "file_id", None) or f"{uploaded.name}:{uploaded.size}")
    if df is not None and mark == st.session_state.get("_ingredients_loaded"):
        st.caption("This file is already loaded. Choose another to replace "
                   "the ingredient list.")
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
                    flash("success", f"Loaded {plural(len(df), 'ingredient')}.")
                    # The new file can rename a unit or drop an ingredient a
                    # limit was written against.
                    _flash_removed_limits(opt, removed)
                    _note_discarded_batch(opt, batch_no)
                    st.rerun()


def _flash_removed_limits(opt, removed):
    """Name every amount limit an edit just emptied of meaning, one line
    each. Three edits can do it — a unit set on one ingredient, a new default
    unit, a reloaded ingredient file — and all three say it the same way."""
    for qc in removed:
        if 'metric' in qc:
            # A property limit is an average over the amounts, so it is the
            # ingredients as a whole that stopped sharing a unit — there is no
            # list of its own to name.
            flash("warning", f"The limit on {qc['metric']} was removed because "
                             "the ingredients no longer share a unit.")
            continue
        label = _limit_label(opt, qc)
        if qc.get('reason') == 'missing':
            gone = qc.get('missing') or []
            who = (f"{number_list(gone)} are no longer ingredients"
                   if len(gone) > 1 else f"{gone[0]} is no longer an ingredient")
            flash("warning", f"The limit on {label} was removed because {who}.")
        else:
            flash("warning", f"The limit on {label} was removed because those "
                             "ingredients no longer share a unit.")


def _limit_label(opt, qc):
    """How one amount limit is named on screen — 'Total amount (all
    ingredients)' or 'Water + Oil'. The list under Limits and the line that
    reports a limit removed both read from here, so they name it alike."""
    names = [v['name'] for v in opt.variables
             if v.get('category', 'ingredient') == 'ingredient']
    if names and set(qc['ingredients']) == set(names):
        return "Total amount (all ingredients)"
    return " + ".join(qc['ingredients'])


def _measurement_editor(opt, storage, editing):
    """The add/edit fields. `editing` is the measurement being changed, or
    None when adding a new one. No st.form: the live share line has to follow
    what is typed."""
    if editing is None:
        st.session_state.setdefault(_mkey(None, "name"), "")
        name = st.text_input("Name", key=_mkey(None, "name"),
                             placeholder="e.g. Firmness")
    else:
        st.session_state.setdefault(_mkey(editing, "name"), editing['name'])
        st.text_input("Name", key=_mkey(editing, "name"), disabled=True)
        name = editing['name']

    c1, c2 = st.columns(2)
    with c1:
        st.session_state.setdefault(_mkey(editing, "unit"),
                                    (editing or {}).get('unit', ""))
        unit = st.text_input("Unit", key=_mkey(editing, "unit"), placeholder="e.g. N")
    with c2:
        st.session_state.setdefault(_mkey(editing, "goal"),
                                    (editing or {}).get('goal', "max"))
        goal = st.selectbox("Goal", list(GOAL_LABELS), key=_mkey(editing, "goal"),
                            format_func=GOAL_LABELS.get,
                            help="Whether you want this measurement higher, "
                                 "lower, or at a target.")

    st.session_state.setdefault(_mkey(editing, "target"),
                                float((editing or {}).get('target') or 0.0))
    target = st.number_input("Target", key=_mkey(editing, "target"),
                             disabled=(goal != 'target'))

    st.markdown("**Scale**")
    s1, s2 = st.columns(2)
    with s1:
        st.session_state.setdefault(_mkey(editing, "min"),
                                    float((editing or {}).get('min_val', 0.0)))
        lowest = st.number_input("Lowest possible", key=_mkey(editing, "min"))
    with s2:
        st.session_state.setdefault(_mkey(editing, "max"),
                                    float((editing or {}).get('max_val', 10.0)))
        highest = st.number_input("Highest possible", key=_mkey(editing, "max"))
    st.caption("The ends of your scale or instrument range, not the values "
               "you expect")

    st.session_state.setdefault(_mkey(editing, "importance"),
                                float((editing or {}).get('weight', 1.0)))
    importance = st.number_input(
        "Importance", min_value=0.1, max_value=100.0, step=0.1,
        key=_mkey(editing, "importance"),
        help="Any positive number. 2 counts twice as much as 1.",
    )

    # Only once there is something to name: "This measurement: 40% of the
    # overall score" on an empty form is a share of nothing.
    if str(name).strip():
        others = sum(float(o['weight']) for o in opt.objectives
                     if o['name'] != name)
        total = others + float(importance)
        share = (float(importance) / total * 100.0) if total > 0 else 0.0
        st.markdown(f"{str(name).strip()}: {share:.0f}% of the overall score")

    if editing is None:
        if st.button("Add measurement", key="add_measurement"):
            # add_objective REPLACES a measurement of the same name, which
            # would silently overwrite its goal, target and scale and rescore
            # every result with no copy kept. Editing is a different door.
            if any(str(name).strip().lower() == o['name'].lower()
                   for o in opt.objectives):
                st.error("That measurement already exists. Use Edit on its "
                         "row to change it.")
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
                flash("success", f"Added {str(name).strip()}.")
                st.rerun()
        return

    b1, b2 = st.columns(2)
    # The edit being made is the one thing to do while its row is open, as a
    # correction is on tab 3; the foot's Continue steps aside (it would
    # navigate away and throw the edit away).
    lit = not confirmation_open()
    with b1:
        save = st.button("Save changes", key="save_measurement",
                         type="primary" if lit else "secondary",
                         disabled=not lit, use_container_width=True) and lit
    with b2:
        if st.button("Cancel", key="cancel_measurement", use_container_width=True):
            _clear_measurement_keys(editing)
            st.session_state.pop("_editing_measurement", None)
            st.rerun()
    if save:
        _apply_measurement_edit(opt, storage, editing, importance, goal, target,
                                lowest, highest, unit)


def _rescores(editing, importance, goal, target, lowest, highest):
    """True when this edit changes how every stored result scores. Importance
    is not the only such field: the goal, the target and either end of the
    scale all feed closeness, so all of them recalculate the history."""
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
        recalculated = (" Every overall score was recalculated."
                        if opt.Y_history else "")
        sentence = (
            f"{editing['name']} importance changed to {float(importance):.1f}."
            if changed_importance else f"Updated {editing['name']}."
        ) + recalculated
        parts = [sentence, best_move_sentence(before, after),
                 # This edit has no confirmation before it, so the copy it
                 # kept is named here, as a correction names its own.
                 "A copy of the project was kept first."]
        flash("success", " ".join(p for p in parts if p))
    else:
        flash("success", f"Updated {editing['name']}.")
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
    sentence = f"{name} removed." + (" Every overall score was recalculated."
                                     if opt.Y_history else "")
    move = best_move_sentence(before, after)
    flash("success", f"{sentence} {move}".strip())
    st.rerun()


def _measurements(opt, storage):
    """Draw the measurements section. Returns True while a measurement is
    open for editing: `Save changes` is then the tab's one lit action and the
    foot steps aside."""
    st.subheader("Measurements and targets")
    editing_name = st.session_state.get("_editing_measurement")
    editing = next((o for o in opt.objectives if o['name'] == editing_name), None)
    if editing is not None:
        _measurement_editor(opt, storage, editing)
    elif not opt.objectives:
        _measurement_editor(opt, storage, None)
    else:
        with st.expander("Add a measurement"):
            _measurement_editor(opt, storage, None)

    if not opt.objectives:
        return editing is not None

    ordered = opt.measurements_by_importance()
    st.dataframe(pd.DataFrame([{
        "Priority": i + 1,
        "Measurement": label_with_unit(o['name'], o.get('unit')),
        "Goal": _goal_text(o),
        "Scale": _scale_text(o),
        "Importance": float(o['weight']),
        "Share": f"{opt.importance_share(o['name']) * 100:.0f}%",
    } for i, o in enumerate(ordered)]), hide_index=True, key="measurement_table",
        height=table_height(len(ordered)))
    for obj in ordered:
        e1, e2 = st.columns(2)
        with e1:
            if st.button(f"Edit {obj['name']}", key=f"edit_meas_{obj['name']}"):
                st.session_state["_editing_measurement"] = obj['name']
                st.rerun()
        with e2:
            if confirm_action(
                f"rm_meas_{obj['name']}", f"Remove {obj['name']}",
                f"Remove {obj['name']}? Every overall score is recalculated "
                "without it. A copy of the project is kept first.",
                confirm_label="Yes, remove",
                disabled=other_confirmation(f"rm_meas_{obj['name']}"),
            ):
                _remove_measurement(opt, storage, obj['name'])

    st.caption(opt.score_function_line())

    with st.expander("How closeness is worked out"):
        st.markdown(
            "- **Higher is better:** closeness = (measured − lowest) ÷ "
            "(highest − lowest), so the top of your scale scores 1 and the "
            "bottom scores 0.\n"
            "- **Lower is better:** the reverse — the bottom of your scale "
            "scores 1 and the top scores 0.\n"
            "- **Hit a target:** closeness is 1 at the target and falls evenly "
            "with distance from it; a full scale width away scores 0.\n\n"
            "Each closeness is multiplied by that measurement's importance, "
            "and the results are added up.\n\n"
            + _GLOSS
        )
    return editing is not None


def _limits(opt):
    # A limit is a sum, and a sum only has a unit when the ingredients share
    # one. When they do not, the labels stay bare and the limit itself is
    # refused with "Choose ingredients that share a unit."
    unit = opt.one_amount_unit() or ""
    with st.expander("Limits (optional)"):
        st.caption("Limits hold the next batch to an amount or a property; to "
                   "cap something you measure, make it a measurement.")

        properties = set()
        for props in opt.ingredient_properties.values():
            properties.update(props.keys())
        if properties:
            st.markdown("**Limit on the finished formulation**")
            # Per 100 g of what you make, not a total that grows with the
            # batch: the same limit then means the same thing at 100 g and at
            # 10 kg. Written in the unit the ingredients are actually in.
            st.caption(f"Per 100 {opt.one_amount_unit() or 'g'} of "
                       "formulation, worked out from your ingredient file's "
                       "property columns.")
            metric = st.selectbox("Property", sorted(properties), key="prop_metric")
            p1, p2 = st.columns(2)
            with p1:
                st.session_state.setdefault("prop_min", None)
                st.number_input("Min value", placeholder="no limit",
                                key="prop_min")
            with p2:
                st.session_state.setdefault("prop_max", None)
                st.number_input("Max value", placeholder="no limit",
                                key="prop_max")
            if st.button("Add a property limit", key="add_property_limit"):
                low, high = st.session_state["prop_min"], st.session_state["prop_max"]
                if low is None and high is None:
                    st.error("Enter a minimum, a maximum, or both.")
                else:
                    try:
                        opt.add_constraint(metric, min_val=low, max_val=high)
                    except ValueError as e:
                        st.error(str(e))
                    else:
                        _report_limit(opt, f"Limit added on {metric}.")
        else:
            st.caption("Upload an ingredient CSV with extra columns such as "
                       "Cost or Sodium per 100 g to set property limits.")

        # A limit written before 0.3.0 was a total, and the file does not say
        # so; the same stored number now means a per-100 g average. Said once,
        # above the list, and only while such a limit is still there.
        if any(c.get('basis') != 'per_100' for c in opt.constraints):
            st.caption(f"A limit set before this version is now read per 100 "
                       f"{opt.one_amount_unit() or 'g'} of formulation.")

        for i, constraint in enumerate(opt.constraints):
            c1, c2 = st.columns([3, 1])
            with c1:
                bounds = ([f"at least {constraint['min']:g}"]
                          if constraint['min'] is not None else [])
                bounds += ([f"at most {constraint['max']:g}"]
                           if constraint['max'] is not None else [])
                st.text(f"{constraint['metric']}: {' and '.join(bounds)}")
            with c2:
                if st.button("Remove", key=f"rm_constr_{i}"):
                    metric = constraint['metric']
                    opt.remove_constraint(i)
                    if saved_ok(opt):
                        flash("success", f"Limit on {metric} removed. The next "
                                         "batch is no longer held to it.")
                        st.rerun()

        names = [v['name'] for v in opt.variables
                 if v.get('category', 'ingredient') == 'ingredient']
        if not names:
            st.caption("Load ingredients first to add amount limits.")
            return

        st.markdown("**Amount limit**")
        picked = st.multiselect("Ingredients to limit together", names,
                                key="qty_pick")
        q1, q2 = st.columns(2)
        with q1:
            st.session_state.setdefault("qc_min", None)
            st.number_input(f"Min sum{_unit_suffix(unit)}",
                            placeholder="no limit", key="qc_min")
        with q2:
            st.session_state.setdefault("qc_max", None)
            st.number_input(f"Max sum{_unit_suffix(unit)}",
                            placeholder="no limit", key="qc_max")
        # No "Set a maximum" tick box: a blank field already means no limit,
        # exactly as it does for the total amount below, and a box the user
        # forgot to tick silently threw their number away.
        if st.button("Add an amount limit", key="add_amount_limit"):
            low, high = st.session_state["qc_min"], st.session_state["qc_max"]
            if not picked:
                st.error("Choose at least one ingredient.")
            elif low is None and high is None:
                st.error("Enter a minimum, a maximum, or both.")
            else:
                try:
                    opt.add_quantity_constraint(picked, min_val=low, max_val=high)
                except ValueError as e:
                    st.error(str(e))
                else:
                    _report_limit(opt, f"Limit added on {', '.join(picked)}.")

        t1, t2, t3 = st.columns(3)
        with t1:
            st.session_state.setdefault("tm_min", None)
            st.number_input(f"Total amount min{_unit_suffix(unit)}",
                            placeholder="no limit", key="tm_min")
        with t2:
            st.session_state.setdefault("tm_max", None)
            st.number_input(f"Total amount max{_unit_suffix(unit)}",
                            placeholder="no limit", key="tm_max")
        with t3:
            if st.button("Add a total amount limit", key="add_total_limit"):
                low, high = st.session_state["tm_min"], st.session_state["tm_max"]
                if low is None and high is None:
                    st.error("Enter a minimum, a maximum, or both.")
                else:
                    try:
                        opt.add_total_mass_constraint(min_val=low, max_val=high)
                    except ValueError as e:
                        st.error(str(e))
                    else:
                        _report_limit(opt, "Total amount limit added.")

        for i, qc in enumerate(getattr(opt, "quantity_constraints", [])):
            label = _limit_label(opt, qc)
            # An amount limit sums ingredients that share a unit, so the
            # limit is written in it: "at most 400 g", never a bare 400.
            limited = {opt.unit_of(n) for n in qc['ingredients']}
            qc_unit = limited.pop() if len(limited) == 1 else ""
            bounds = ([join_unit(f"at least {qc['min']:g}", qc_unit)]
                      if qc['min'] is not None else [])
            bounds += ([join_unit(f"at most {qc['max']:g}", qc_unit)]
                       if qc['max'] is not None else [])
            l1, l2 = st.columns([3, 1])
            with l1:
                st.text(f"{label}: {' and '.join(bounds)}")
            with l2:
                if st.button("Remove", key=f"rm_qc_{i}"):
                    opt.remove_quantity_constraint(i)
                    if saved_ok(opt):
                        # No .lower(): ingredient names are names.
                        flash("success", f"Limit on {label} removed. The next "
                                         "batch is no longer held to it.")
                        st.rerun()


def _advanced(opt):
    with st.expander("Advanced model settings"):
        st.caption("Standard uses tested defaults and fits most projects. "
                   "Expert-selected lets a specialist set the model's kernel, "
                   "prior, noise handling and acquisition once at the start.")
        current = getattr(opt, "bo_config", None)
        mode = st.radio("Model settings", ["Standard (default)", "Expert-selected"],
                        index=1 if current else 0, key="bo_cfg_mode", horizontal=True)
        if mode == "Standard (default)":
            if current is not None and st.button("Revert to standard settings",
                                                 key="bo_revert"):
                opt.set_bo_config(None)
                if saved_ok(opt):
                    flash("success", "Using default model settings.")
                    st.rerun()
        else:
            b1, b2 = st.columns(2)
            with b1:
                kernel = st.selectbox("Kernel",
                                      ["matern52", "matern32", "rbf", "linear", "poly2"],
                                      key="bo_kernel")
                prior = st.selectbox("Lengthscale prior", ["default", "long", "short"],
                                     key="bo_prior")
            with b2:
                noise = st.selectbox("Noise", ["default", "low", "fixed_tiny"],
                                     key="bo_noise")
                acq = st.selectbox("Acquisition", ["qlognei", "qlogei", "qucb"],
                                   key="bo_acq")
            st.caption("`fixed_tiny` noise suits a deterministic measurement, "
                       "not a sensory panel — keep `default` unless you have a "
                       "specific reason.")
            if st.button("Apply expert settings", key="bo_apply"):
                opt.set_bo_config({"kernel": kernel, "lengthscale_prior": prior,
                                   "noise": noise, "acquisition": acq})
                if saved_ok(opt):
                    flash("success", "Model settings updated.")
                    st.rerun()
            if st.checkbox("Or paste expert settings as JSON", key="bo_paste"):
                text = st.text_area(
                    "Expert settings JSON",
                    value='{"kernel": "matern52", "lengthscale_prior": "default", '
                          '"noise": "default", "acquisition": "qlognei"}',
                    key="bo_json",
                )
                if st.button("Apply pasted settings", key="bo_apply_json"):
                    try:
                        opt.set_bo_config(json.loads(text))
                    except Exception as e:
                        st.error(f"Invalid JSON: {e}")
                    else:
                        if saved_ok(opt):
                            flash("success", "Model settings updated.")
                            st.rerun()
        if current:
            st.caption("Active model settings: "
                       + ", ".join(f"{k}: {v}" for k, v in current.items()))


def _foot(opt, editing=False):
    ready, missing = readiness(opt)
    # While a confirmation is armed, its "Yes" is the one coloured button and
    # answering it is the one thing to do; moving on can wait a click. An open
    # measurement editor is the same case: Save changes is the lit one, and
    # Continue would leave the tab and throw the edit away.
    lit = ready and not confirmation_open() and not editing
    if st.button("Continue to make a batch",
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
    _limits(opt)
    _advanced(opt)
    st.divider()
    _foot(opt, editing)
