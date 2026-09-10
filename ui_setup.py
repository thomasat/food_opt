"""Tab 1 · Set up: what the project is made of and what will be measured.

One column, in the order a formulator fills it in: the unit amounts are in,
the ingredients, the measurements, then the optional sections. Exactly one
coloured button lives here — `Continue to make a batch` at the foot.
"""
import json
import os

import pandas as pd
import streamlit as st

import storage as storage_backend
from ui_helpers import (
    TAB_BATCH, confirm_action, confirmation_open, flash, go_to_tab,
    other_confirmation, plural,
    readiness, saved_ok, table_height,
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
    """'Target 6 N', 'Higher is better', 'Lower is better'."""
    if obj['goal'] == 'target':
        from food_bo import join_unit
        return join_unit(f"Target {float(obj['target']):g}", obj.get('unit'))
    return GOAL_LABELS.get(obj['goal'], obj['goal'])


def _scale_text(obj):
    from food_bo import join_unit
    return join_unit(f"{float(obj['min_val']):g} to {float(obj['max_val']):g}",
                     obj.get('unit'))


def _best_formulation_no(opt):
    i = opt.best_index()
    return None if i is None else int(opt.formulation_ids[i])


def _best_move_sentence(before, after):
    """'Best moved from Formulation 3 to Formulation 7.' or '' when it stayed."""
    if before is None or after is None or before == after:
        return ""
    return f"Best moved from Formulation {before} to Formulation {after}."


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

def _amount_unit(opt):
    st.session_state.setdefault("amount_unit", opt.amount_unit)
    typed = str(st.text_input(
        "Amounts are in", key="amount_unit", placeholder="g",
        help="One unit for every amount in this project, such as g, % or kg. "
             "It shows in every ingredient header, batch sheet and off-by line.",
    )).strip()
    # Compare stripped with stripped: comparing a stripped store against an
    # unstripped box re-saved on every rerun, and a failing save then bounced
    # the script into a rerun loop.
    if typed != opt.amount_unit:
        opt.set_amount_unit(typed)
        saved_ok(opt)


def _ingredients(opt, storage):
    unit = opt.amount_unit
    st.subheader("Ingredients")
    st.caption("One row per ingredient: Name, Min, Max.")
    uploaded = st.file_uploader(
        "Upload ingredients CSV", type=["csv"], key="ingredients_csv",
        help="Columns Name, Min, Max. Extra columns such as Cost or Protein "
             "per 100 g become properties you can set limits on.",
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
    if df is not None:
        st.dataframe(df, hide_index=True, height=table_height(len(df)))
        if st.button("Load ingredients", key="load_ingredients"):
            batch_no = opt.pending_batch_no
            try:
                opt.load_ingredients_from_csv(df)
            except ValueError as e:
                st.error(str(e))
            else:
                if saved_ok(opt):
                    # Clear the uploader too: leaving the file and its preview
                    # on screen invites a second Load, which errors once
                    # results exist.
                    st.session_state.pop("ingredients_csv", None)
                    flash("success", f"Loaded {plural(len(df), 'ingredient')}.")
                    _note_discarded_batch(opt, batch_no)
                    st.rerun()

    ingredients = [v for v in opt.variables
                   if v.get('category', 'ingredient') == 'ingredient']
    if ingredients:
        # A Status column that says "active" on every row is a column of noise.
        any_paused = any(not v.get('active', True) for v in ingredients)
        rows = pd.DataFrame([{
            "Name": v['name'],
            f"Min{_unit_suffix(unit)}": float(v['bounds'][0]),
            f"Max{_unit_suffix(unit)}": float(v['bounds'][1]),
            **({"Status": "active" if v.get('active', True) else "paused"}
               if any_paused else {}),
        } for v in ingredients])
        st.dataframe(rows, hide_index=True, key="ingredient_table",
                     height=table_height(len(rows), max_rows=20))

    with st.expander("Change the ingredient list"):
        _change_ingredient_list(opt, storage)


def _change_ingredient_list(opt, storage):
    unit = opt.amount_unit
    has_history = bool(opt.X_history)
    st.markdown("**Add an ingredient**")
    ic1, ic2, ic3 = st.columns([2, 1, 1])
    with ic1:
        st.text_input("Ingredient name", key="ing_name")
    with ic2:
        st.number_input(f"Min{_unit_suffix(unit)}", value=0.0, key="ing_min",
                        disabled=has_history)
    with ic3:
        st.number_input(f"Max{_unit_suffix(unit)}", value=100.0, key="ing_max")
    if has_history:
        st.caption("A new ingredient starts at 0 in every formulation already "
                   "made, so its minimum is fixed at 0 for now.")
    if st.button("Add ingredient", key="add_ingredient"):
        batch_no = opt.pending_batch_no
        try:
            opt.add_ingredient(st.session_state["ing_name"],
                               st.session_state["ing_min"],
                               st.session_state["ing_max"])
        except ValueError as e:
            st.error(str(e))
        else:
            if saved_ok(opt):
                flash("success",
                      f"Added {str(st.session_state['ing_name']).strip()}.")
                _note_discarded_batch(opt, batch_no)
                st.rerun()

    active = opt.active_variables()
    inactive = opt.inactive_variables()
    st.divider()
    st.markdown("**Pause an ingredient**")
    st.caption("A paused ingredient is left out of new formulations; nothing "
               "is deleted and you can resume at any time.")
    if len(active) > 1:
        pause = st.multiselect("Pause", [v['name'] for v in active], key="pause_pick")
        if st.button("Pause selected", disabled=not pause, key="pause_go"):
            batch_no = opt.pending_batch_no
            try:
                for name in pause:
                    opt.deactivate_variable(name)
            except ValueError as e:
                st.error(str(e))
            else:
                if saved_ok(opt):
                    flash("success", f"Paused: {', '.join(pause)}.")
                    _note_discarded_batch(opt, batch_no)
                    st.rerun()
    else:
        st.caption("At least two ingredients or settings must stay active "
                   "before one can be paused.")
    if inactive:
        st.dataframe(pd.DataFrame([{
            "Name": v['name'],
            "Type": "ingredient" if v.get('category', 'ingredient') == 'ingredient'
                    else "process setting",
            f"Held at{_unit_suffix(unit)}": opt._frozen_value(v),
        } for v in inactive]), hide_index=True, key="paused_table")
        resume = st.multiselect("Resume", [v['name'] for v in inactive],
                                key="resume_pick")
        if st.button("Resume selected", disabled=not resume, key="resume_go"):
            batch_no = opt.pending_batch_no
            for name in resume:
                opt.reactivate_variable(name)
            if saved_ok(opt):
                flash("success", f"Resumed: {', '.join(resume)}.")
                _note_discarded_batch(opt, batch_no)
                st.rerun()

    st.divider()
    # A checkbox, not an expander: Streamlit forbids nesting expanders.
    if st.checkbox("Show permanent deletion (rarely needed)", key="show_delete_ing"):
        st.caption("Deleting removes the ingredient from every formulation "
                   "already made. It is refused if the ingredient was ever "
                   "used above 0, because that would rewrite formulations "
                   "nobody made. Pausing keeps the data.")
        names = [v['name'] for v in opt.variables
                 if v.get('category', 'ingredient') == 'ingredient']
        if not names:
            st.caption("No ingredients loaded.")
        else:
            pick = st.selectbox("Ingredient", names, key="delete_ing_pick")
            force = st.checkbox("Delete even if it was used (discards that "
                                "information)", key="delete_ing_force")
            if confirm_action(
                "delete_ing", "Delete permanently",
                f"Delete {pick} from this project permanently? Formulations "
                "already made will be recorded without it. A copy of the "
                "project is kept first.",
                confirm_label="Yes, delete",
                disabled=other_confirmation("delete_ing"),
            ):
                batch_no = opt.pending_batch_no
                try:
                    storage.archive(opt.project_name, "pre_delete", copy=True)
                    opt.remove_ingredient(pick, force=force)
                except (ValueError, storage_backend.StorageError) as e:
                    st.error(str(e))
                else:
                    if saved_ok(opt):
                        flash("success", f"Deleted {pick}.")
                        _note_discarded_batch(opt, batch_no)
                        st.rerun()


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
    with b1:
        save = st.button("Save changes", key="save_measurement",
                         use_container_width=True)
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
    before = _best_formulation_no(opt)
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
        after = _best_formulation_no(opt)
        sentence = (
            f"{editing['name']} importance changed to {float(importance):.1f}. "
            "Every overall score was recalculated."
            if changed_importance else
            f"Updated {editing['name']}. Every overall score was recalculated."
        )
        move = _best_move_sentence(before, after)
        flash("success", f"{sentence} {move}".strip())
    else:
        flash("success", f"Updated {editing['name']}.")
    st.rerun()


def _remove_measurement(opt, storage, name):
    before = _best_formulation_no(opt)
    try:
        storage.archive(opt.project_name, "pre_edit", copy=True)
    except storage_backend.StorageError as e:
        st.error(str(e))
        return
    opt.remove_objective(name)
    if not saved_ok(opt):
        return
    after = _best_formulation_no(opt)
    sentence = f"{name} removed. Every overall score was recalculated."
    move = _best_move_sentence(before, after)
    flash("success", f"{sentence} {move}".strip())
    st.rerun()


def _measurements(opt, storage):
    st.subheader("Measurements")
    editing_name = st.session_state.get("_editing_measurement")
    editing = next((o for o in opt.objectives if o['name'] == editing_name), None)
    if editing is not None:
        _measurement_editor(opt, storage, editing)
    elif not opt.objectives:
        st.caption("What you will measure on every formulation.")
        _measurement_editor(opt, storage, None)
    else:
        with st.expander("Add a measurement"):
            _measurement_editor(opt, storage, None)

    if not opt.objectives:
        return

    ordered = opt.measurements_by_importance()
    st.dataframe(pd.DataFrame([{
        "Priority": i + 1,
        "Measurement": o['name'],
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
            "and the results are added up."
        )


def _process_settings(opt, storage):
    with st.expander("Process settings (optional)"):
        st.caption("Settings such as cook temperature or mixing time that can "
                   "vary between formulations.")
        mid_run = bool(opt.X_history)
        pc = st.columns(4)
        with pc[0]:
            st.text_input("Setting name", key="pp_name",
                          placeholder="e.g. Cook temperature")
        with pc[1]:
            st.number_input("Min value", value=0.0, key="pp_min")
        with pc[2]:
            st.number_input("Max value", value=100.0, key="pp_max")
        with pc[3]:
            if mid_run:
                st.number_input(
                    "Baseline", value=None, placeholder="required", key="pp_base",
                    help="The setting you used for every formulation already "
                         "made, so those results still count.",
                )
            else:
                st.number_input(
                    "Baseline", value=0.0, key="pp_base",
                    help="Only needed once results exist: the setting used for "
                         "every formulation already made.",
                )
        if st.button("Add process setting", key="add_process_setting"):
            if mid_run and st.session_state["pp_base"] is None:
                st.error("Enter the baseline: the setting you used for every "
                         "formulation already made.")
            else:
                batch_no = opt.pending_batch_no
                try:
                    opt.add_process_parameter(
                        st.session_state["pp_name"], st.session_state["pp_min"],
                        st.session_state["pp_max"],
                        baseline=(st.session_state["pp_base"] if mid_run else None),
                    )
                except ValueError as e:
                    st.error(str(e))
                else:
                    if saved_ok(opt):
                        flash("success",
                              f"Added {str(st.session_state['pp_name']).strip()}.")
                        _note_discarded_batch(opt, batch_no)
                        st.rerun()

        process = [v for v in opt.variables if v.get('category') == 'process']
        for i, pv in enumerate(process):
            p1, p2 = st.columns([3, 1])
            with p1:
                low, high = pv['bounds']
                base = pv.get('_absent_value')
                base_txt = "" if base is None else f", baseline {base:g}"
                paused = "" if pv.get('active', True) else "  (paused)"
                st.text(f"{pv['name']}: {low:g} to {high:g}{base_txt}{paused}")
            with p2:
                # Always confirmed, always copied first, history or not: a
                # removal is a removal and the user is told the same thing
                # every time.
                if confirm_action(
                    f"rm_pp_{i}", "Remove",
                    f"Remove {pv['name']}? Formulations already made will "
                    "be recorded without it. A copy of the project is "
                    "kept first.",
                    confirm_label="Yes, remove",
                    disabled=other_confirmation(f"rm_pp_{i}"),
                ):
                    batch_no = opt.pending_batch_no
                    try:
                        storage.archive(opt.project_name, "pre_delete", copy=True)
                    except storage_backend.StorageError as e:
                        st.error(str(e))
                    else:
                        opt.remove_process_parameter(pv['name'])
                        if saved_ok(opt):
                            flash("success", f"Removed {pv['name']}.")
                            _note_discarded_batch(opt, batch_no)
                            st.rerun()


def _limits(opt):
    unit = opt.amount_unit
    with st.expander("Limits (optional)"):
        st.caption("Limits apply to ingredient amounts and to properties from "
                   "your ingredient file. To cap something you measure, add it "
                   "as a measurement.")

        properties = set()
        for props in opt.ingredient_properties.values():
            properties.update(props.keys())
        if properties:
            st.markdown("**Property limit**")
            metric = st.selectbox("Property", sorted(properties), key="prop_metric")
            p1, p2 = st.columns(2)
            with p1:
                st.number_input("Min value", value=None, placeholder="no limit",
                                key="prop_min")
            with p2:
                st.number_input("Max value", value=None, placeholder="no limit",
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
            st.caption("Your ingredient file has no property columns, such as "
                       "Cost or Sodium per 100 g.")

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
            st.number_input(f"Min sum{_unit_suffix(unit)}", value=None,
                            placeholder="no limit", key="qc_min")
        with q2:
            st.number_input(f"Max sum{_unit_suffix(unit)}", value=None,
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
            st.number_input(f"Total amount min{_unit_suffix(unit)}", value=None,
                            placeholder="no limit", key="tm_min")
        with t2:
            st.number_input(f"Total amount max{_unit_suffix(unit)}", value=None,
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
            label = ("Total amount (all ingredients)"
                     if set(qc['ingredients']) == set(names)
                     else " + ".join(qc['ingredients']))
            bounds = ([f"at least {qc['min']:g}"] if qc['min'] is not None else [])
            bounds += ([f"at most {qc['max']:g}"] if qc['max'] is not None else [])
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


def _foot(opt):
    ready, missing = readiness(opt)
    # While a confirmation is armed, its "Yes" is the one coloured button and
    # answering it is the one thing to do; moving on can wait a click.
    lit = ready and not confirmation_open()
    if st.button("Continue to make a batch",
                 type="primary" if lit else "secondary",
                 disabled=not lit, key="continue_to_batch") and lit:
        go_to_tab(TAB_BATCH)
    if not ready:
        st.caption(missing)


def render(opt, storage):
    _amount_unit(opt)
    st.divider()
    _ingredients(opt, storage)
    st.divider()
    _measurements(opt, storage)
    st.divider()
    _process_settings(opt, storage)
    _limits(opt)
    _advanced(opt)
    st.divider()
    _foot(opt)
