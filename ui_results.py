"""Tab 3 · Results: the best formulation, every formulation, corrections.

The one coloured button is at the foot: `Start the next batch`, or `Back to
batch N` while a batch is still unrecorded. Everything destructive is behind a
confirmation that keeps a copy first and says so — once, in the confirmation.
"""
import pandas as pd
import streamlit as st

import storage as storage_backend
from ui_helpers import (
    TAB_BATCH, TAB_SETUP, best_formulation_no, best_move_sentence,
    bounds_warning, clear_selection, confirm_action, confirmation_open, flash,
    fmt_amount, fmt_setting, goal_line, go_to_tab, join_unit, label_with_unit,
    open_rows, other_confirmation, plural, readiness, saved_ok, scale_error,
    table_height, take_clear, unit_after_number,
)


def _all_numbers(opt):
    """Every number this project has issued and still holds, in order."""
    numbers = [int(n) for n in opt.formulation_ids]
    numbers += [int(s['formulation']) for s in opt.skipped]
    return sorted(numbers)


def _amount_rows(opt, recipe):
    """The `Amounts to make it` rows: ingredients by amount, largest first,
    then the process settings, each written with its own unit — the water in
    ml beside the protein in g, and a cook temperature never in either."""
    category = {v['name']: v.get('category', 'ingredient') for v in opt.variables}
    pairs = opt.recipe_lines(recipe)
    ingredients = [p for p in pairs if category.get(p[0]) != 'process']
    settings = [p for p in pairs if category.get(p[0]) == 'process']

    def amount(name, value):
        if category.get(name) == 'process':
            # 180 °C, never 180.00 °C: a setting is dialled in, not weighed.
            return fmt_setting(value, opt.unit_of(name))
        return fmt_amount(value, opt.unit_of(name))

    return [{"Ingredient or setting": name, "Amount": amount(name, value)}
            for name, value in ingredients + settings]


def _number(cell):
    """A CSV cell as a float, or None when it is blank or not a number. The
    import loop reports an unreadable cell by row; the checks above it step
    over one rather than crashing on the way to the message."""
    try:
        if pd.isna(cell):
            return None
        return float(cell)
    except (TypeError, ValueError):
        return None


def _range_warning(opt, name, value):
    """The line for an imported amount outside what the project allows, or
    '' when it fits. Built by the same helper that refuses an out-of-scale
    measurement, so the two sentences read alike.
    It is a warning, not a refusal: the amount is a fact about work already
    done, and the model learns more from it than from a blank."""
    var = next((v for v in opt.variables if v['name'] == name), None)
    if var is None or value is None:
        return ""
    low, high = (float(b) for b in var['bounds'])
    return bounds_warning(name, value, low, high, opt.unit_of(name))


def _progress_line(opt):
    last = opt.last_batch_no()   # counts a batch whose rows were all left out
    if last is None or not opt.Y_history:
        return ""
    earlier = [float(y) for y, b in zip(opt.Y_history, opt.batch_history)
               if b != last]
    if not earlier:
        # Nothing to compare it with. The flash above the tabs already says
        # "Batch 1 recorded."; saying it again four lines lower is the same
        # sentence twice on one screen.
        return ""
    best_now = max(float(y) for y in opt.Y_history)
    best_before = max(earlier)
    if best_now > best_before + 1e-9:
        return (f"Batch {last} recorded · best improved "
                f"{best_before:.2f} → {best_now:.2f}")
    return f"Batch {last} recorded · no improvement."


def _best(opt):
    index = opt.best_index()
    if index is None:
        return
    number = int(opt.formulation_ids[index])
    batch = opt.batch_history[index]
    heading = f"Best so far: Formulation {number}"
    if batch is not None:
        heading += f" (batch {batch})"
    st.subheader(heading)
    line = _progress_line(opt)
    if line:
        st.caption(line)

    details = opt.closeness_details(index)
    if details:
        st.dataframe(pd.DataFrame([
            {"Measurement": d['name'], "Goal": d['goal'],
             "Measured": d['measured'], "Off by": d['off_by']}
            for d in details
        ]), hide_index=True, key="best_off_by",
            height=table_height(len(details)))

    st.markdown("**Amounts to make it**")
    recipe = opt.recipe_history[index]
    st.table(pd.DataFrame(_amount_rows(opt, recipe),
                          columns=["Ingredient or setting", "Amount"]))
    # Ingredients only: a process setting sitting at 0 is a setting, not an
    # ingredient somebody left out.
    unused = [v['name'] for v in opt.variables
              if v.get('category', 'ingredient') == 'ingredient'
              and not float(recipe.get(v['name'], 0.0))]
    if unused:
        st.caption("Not used: " + ", ".join(unused))
    ceiling = opt.utility_ceiling()
    # "(partial)", exactly as the All formulations row writes it: a score
    # missing one measurement is not the same number as a full one.
    scored = opt.results_history[index] if index < len(opt.results_history) else {}
    partial = " (partial)" if any(o['name'] not in scored
                                  for o in opt.objectives) else ""
    st.caption(f"Overall score {float(opt.Y_history[index]):.2f}{partial} of "
               f"{ceiling:.2f} · {ceiling:.2f} is every measurement on target "
               "· not comparable across projects.")


def _all_formulations(opt):
    st.markdown("**All formulations**")
    o1, o2 = st.columns([2, 1])
    with o1:
        order = st.selectbox("Sort",
                             ["Best first", "Newest first", "Batch order"],
                             key="results_order")
    with o2:
        show_amounts = st.toggle("Show amounts", key="show_amounts")
    frame = opt.history_frame(order=order, include_amounts=show_amounts)
    st.dataframe(frame, hide_index=True, key="all_formulations",
                 height=table_height(len(frame), max_rows=20))
    st.download_button("Download all formulations (CSV)", data=opt.history_csv(),
                       file_name=f"{opt.project_name} formulations.csv",
                       mime="text/csv", key="download_formulations",
                       help="Amounts are unitless in this file so it can be "
                            "imported back; units are shown on screen. "
                            "Formulations you left out are not included.")


def _correct(opt):
    """The correction row: the select box and the boxes to retype. Returns the
    open correction, or None — the foot reads that and steps aside, because
    `Save correction` is the one lit action until the row is answered.

    Its button row is only RESERVED here. Arming a confirmation in one of the
    sections below does not rerun, so a Save correction drawn now would still
    be coloured on the very run that puts a Yes beside it; render() fills the
    slot once the confirmations have had their say."""
    numbers = [int(n) for n in opt.formulation_ids]
    if not numbers:
        return None
    take_clear("correct_formulation")
    choice = st.selectbox("Correct a result", numbers, index=None,
                          placeholder="Formulation", key="correct_formulation")
    if choice is None:
        return None
    index = opt.index_of_formulation(choice)
    if index is None:
        return None
    ordered = opt.measurements_by_importance()
    if not ordered:
        st.info("Add a measurement in Set up before correcting a result.")
        return None
    current = opt.results_history[index]
    typed = {}
    cols = None
    for j, obj in enumerate(ordered):
        if j % 4 == 0:
            cols = st.columns(min(4, len(ordered) - j))
        with cols[j % 4]:
            # No clamping here either: the same reading refused on tab 2 must
            # be refusable here, not silently pulled back to the scale end.
            typed[obj['name']] = st.number_input(
                f"{label_with_unit(obj['name'], obj.get('unit'))} · "
                f"{goal_line(obj)}",
                value=(float(current[obj['name']])
                       if current.get(obj['name']) is not None else None),
                placeholder=f"{obj['min_val']:g}–{obj['max_val']:g}",
                help="Leave blank to keep the value already recorded.",
                key=f"correct_{choice}_{obj['name']}",
            )
    return {"choice": choice, "index": index, "ordered": ordered,
            "current": current, "typed": typed, "slot": st.container()}


def _save_correction(opt, storage, pending):
    """The correction's button row, drawn into the slot reserved at the row's
    position once every confirmation on the tab has been drawn."""
    choice, index = pending['choice'], pending['index']
    ordered, current, typed = (pending['ordered'], pending['current'],
                               pending['typed'])
    b1, b2 = st.columns(2)
    # The correction is the one thing to do while its row is open, so it is
    # lit — unless a confirmation is armed, which outranks everything.
    lit = not confirmation_open()
    with b1:
        save = st.button("Save correction", key="save_correction",
                         type="primary" if lit else "secondary",
                         disabled=not lit, use_container_width=True) and lit
    with b2:
        # The select box cannot be cleared by the user once it holds a value.
        if st.button("Close", key="done_correcting", use_container_width=True):
            clear_selection("correct_formulation")
            st.rerun()
    if not save:
        return
    if not any(v is not None for v in typed.values()):
        st.error("Enter a value for at least one measurement.")
        return
    for obj in ordered:
        problem = scale_error(obj, typed[obj['name']])
        if problem:
            st.error(problem)
            return
    # A measurement that could not be scored is left out of the stored row, so
    # a corrected result has exactly the shape a recorded one has: tell() drops
    # the Nones, and edit_result would otherwise keep them.
    changes = []
    final = {k: v for k, v in current.items() if v is not None}
    for obj in ordered:
        value = typed[obj['name']]
        if value is None:
            continue
        was = final.get(obj['name'])
        if was is None or abs(float(was) - float(value)) > 1e-9:
            changes.append((obj['name'], was, float(value)))
        final[obj['name']] = float(value)
    if not changes:
        # Nothing to write, so nothing to copy first, and nothing to claim.
        flash("success", f"Formulation {choice} is unchanged.")
        st.rerun()
    before = best_formulation_no(opt)
    # A correction overwrites a reading nobody can retype from memory, so the
    # project is copied first — as it is before every other destructive act.
    try:
        storage.archive(opt.project_name, "pre_edit", copy=True)
    except storage_backend.StorageError as e:
        st.error(str(e))
        return
    opt.edit_result(index, final)
    if not saved_ok(opt):
        return
    after = best_formulation_no(opt)
    # Every number wears its unit: 66 could be °C or a panel score, and this
    # sentence is the only confirmation that the right one was typed.
    units = {o['name']: unit_after_number(o.get('unit')) for o in ordered}

    def _n(name, value):
        return join_unit(f"{float(value):g}", units.get(name, ""))

    sentences = [
        f"Formulation {choice} {name} corrected "
        f"{_n(name, was)} → {_n(name, now)}."
        for name, was, now in changes if was is not None
    ]
    sentences += [
        f"Formulation {choice} {name} recorded as {_n(name, now)}."
        for name, was, now in changes if was is None
    ]
    move = best_move_sentence(before, after)
    if move:
        sentences.append(move)
    sentences.append("A copy of the project was kept first.")
    flash("success", " ".join(sentences))
    st.rerun()


def _foot_label(opt):
    """What the foot of this tab offers. An open batch outranks everything:
    the work to do is the batch on the bench, and every other screen already
    says so in these words."""
    if opt.pending_batch:
        return (f"Back to batch {opt.pending_batch_no} · "
                f"{len(open_rows(opt))} to record")
    return "Start the next batch"


def _foot(opt, correcting=False):
    label = _foot_label(opt)
    # While a confirmation is armed, its "Yes" is the one coloured button and
    # answering it is the one thing to do; the next batch can wait a click.
    # An open correction row is the same case: Save correction is the lit one.
    lit = not confirmation_open() and not correcting
    if st.button(label, type="primary" if lit else "secondary",
                 disabled=not lit, key="foot_batch") and lit:
        go_to_tab(TAB_BATCH)


def _progress_chart(opt):
    with st.expander("Progress chart"):
        if not opt.Y_history:
            st.caption("No results yet.")
            return
        st.line_chart(pd.DataFrame({
            "Formulation": [int(n) for n in opt.formulation_ids],
            "Overall score": [float(y) for y in opt.Y_history],
            "Best so far": opt.best_so_far(),
        }).set_index("Formulation"), height=220)
        st.caption("Each formulation's overall score, and the best so far. "
                   "When the top line stops rising, you are close to the best "
                   "this ingredient list can do.")


def _undo(opt, storage):
    with st.expander("Undo the last batch"):
        # Left-out formulations count: a batch nobody managed to make is still
        # the last batch, and undoing must not reach past it.
        last = opt.last_batch_no()
        if last is None:
            if opt.X_history or opt.skipped:
                st.caption("These formulations were recorded before batches "
                           "existed, so there is no batch to undo. You can "
                           "delete one formulation at a time below.")
            else:
                st.caption("No batch to undo yet.")
            return
        if opt.pending_batch:
            st.caption("Record or discard the open batch first.")
            st.button("Undo the last batch", disabled=True, key="undo_batch__btn")
            return
        count = sum(1 for b in opt.batch_history if b == last)
        count += sum(1 for s in opt.skipped if s.get('batch') == last)
        # Said once, in the confirmation: a caption above it repeats it.
        if confirm_action(
            "undo_batch", "Undo the last batch",
            # Formulations, not results: one of them may have been left out
            # and have no result at all, and formulation is the app's noun.
            f"Removes batch {last} and its {plural(count, 'formulation')}. "
            "A copy is kept first.",
            confirm_label="Yes, undo", disabled=other_confirmation("undo_batch"),
        ):
            try:
                storage.archive(opt.project_name, "pre_undo", copy=True)
            except storage_backend.StorageError as e:
                st.error(str(e))
            else:
                try:
                    opt.undo_last_batch()
                except ValueError as e:
                    # The open batch appeared between the click and the
                    # confirmation: say why, do not take it down as well.
                    st.error(str(e))
                    return
                if not saved_ok(opt):
                    return
                st.session_state.pop("scale_total", None)
                st.session_state.pop("_results_upload", None)
                flash("success", f"Batch {last} removed. A copy was kept first.")
                st.rerun()


def _delete_formulation(opt, storage):
    with st.expander("Delete a formulation"):
        numbers = _all_numbers(opt)
        if not numbers:
            st.caption("Nothing to delete yet.")
            return
        take_clear("delete_formulation")
        choice = st.selectbox("Formulation", numbers, index=None,
                              placeholder="Formulation",
                              key="delete_formulation")
        if choice is None:
            return
        go = confirm_action(
            "delete_formulation", f"Delete Formulation {choice}",
            f"Delete Formulation {choice}? Later formulations keep their "
            "numbers. A copy is kept first.",
            confirm_label="Yes, delete",
            disabled=other_confirmation("delete_formulation"),
        )
        # The select box cannot be cleared by the user once it holds a value.
        # The confirmation brings its own Cancel, so this one steps aside while
        # that is on screen rather than showing the word twice.
        if (not st.session_state.get("delete_formulation__pending")
                and st.button("Choose a different formulation",
                              key="cancel_delete_formulation")):
            clear_selection("delete_formulation")
            st.rerun()
        if go:
            try:
                storage.archive(opt.project_name, "pre_delete", copy=True)
            except storage_backend.StorageError as e:
                st.error(str(e))
            else:
                opt.delete_formulation(choice)
                if not saved_ok(opt):
                    return
                clear_selection("delete_formulation")
                flash("success", f"Formulation {choice} deleted. A copy was "
                                 "kept first.")
                st.rerun()


def _import(opt):
    with st.expander("Import past formulations from a CSV"):
        variables = [v['name'] for v in opt.variables]
        measurements = [o['name'] for o in opt.objectives]
        if variables or measurements:
            st.caption("One row per formulation you already made. The columns "
                       "must match these names exactly: "
                       + ", ".join(variables + measurements) + ".")
        else:
            st.caption("One row per formulation you already made. Add "
                       "ingredients and measurements first; the columns must "
                       "match their names exactly.")
        uploaded = st.file_uploader(
            "Upload formulations CSV", type=["csv"],
            # Per project: an uploader cannot be emptied from session state,
            # so a shared key offered the next project this one's file.
            key=f"import_csv_{opt.project_name}")
        # The parse is behind a button, as it is on tab 2: reading the file on
        # every rerun left the sheet on screen after it had been imported, and
        # a second click on Import recorded every row twice.
        if uploaded is not None and st.button("Check this sheet",
                                              key="check_import"):
            try:
                st.session_state["_import_rows"] = pd.read_csv(uploaded)
            except Exception:
                st.session_state.pop("_import_rows", None)
                st.error("This file could not be read as a CSV. If it came "
                         "from Excel, use File > Save As and pick CSV format, "
                         "then try again.")
        rows = st.session_state.get("_import_rows")
        if rows is None:
            return
        st.dataframe(rows, hide_index=True)
        missing = [c for c in variables + measurements if c not in rows.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
            return
        blank_amounts = [c for c in variables if rows[c].isna().any()]
        if blank_amounts:
            st.error("These amount columns have blank cells: "
                     + ", ".join(blank_amounts))
            return
        if not st.button("Import all rows", key="import_rows"):
            return
        # Nothing is recorded until the whole file has been read: a reading
        # outside its scale is a typo or a scale that is too narrow, and it
        # would otherwise arrive as the best formulation in the project.
        cautions = []
        for position, (_, row) in enumerate(rows.iterrows(), start=1):
            for obj in opt.objectives:
                problem = scale_error(obj, _number(row[obj['name']]))
                if problem:
                    st.error(f"Row {position}: {problem}")
                    return
            for name in variables:
                caution = _range_warning(opt, name, _number(row[name]))
                if caution:
                    cautions.append(f"Row {position}: {caution}")
        imported, failure = 0, None
        try:
            for _, row in rows.iterrows():
                # A blank measurement is a partial result here too, exactly as
                # it is in the results grid and in an uploaded batch sheet.
                results = {name: float(row[name]) for name in measurements
                           if not pd.isna(row[name])}
                opt.import_formulation(
                    {name: float(row[name]) for name in variables}, results)
                # Stop at the first row that did not reach the disk rather than
                # reporting a whole file as imported.
                if not saved_ok(opt):
                    return
                imported += 1
        except (ValueError, TypeError) as e:
            failure = e
        if failure is not None:
            message = f"Stopped at row {imported + 1}: {failure}"
            if imported:
                message += (f" The {plural(imported, 'row')} before it were "
                            "imported and saved.")
            st.error(message)
            return
        st.session_state.pop("_import_rows", None)
        flash("success", f"Imported {plural(imported, 'formulation')}.")
        for caution in cautions:
            flash("warning", caution)
        st.rerun()


def render(opt, storage):
    if not opt.X_history and not opt.skipped:
        st.markdown("No results yet.")
        # A project with no ingredients cannot make a batch: sending the user
        # to a tab holding a greyed Generate is a lit button to a dead end.
        # And a batch already on the bench is not a first batch to make: the
        # foot of every other screen calls it "Back to batch 1 · 3 to record".
        ready, _ = readiness(opt)
        if not ready:
            label, target = "Set up this project", TAB_SETUP
        elif opt.pending_batch:
            label, target = _foot_label(opt), TAB_BATCH
        else:
            label, target = "Make your first batch", TAB_BATCH
        lit = not confirmation_open()
        if st.button(label, type="primary" if lit else "secondary",
                     disabled=not lit, key="first_batch") and lit:
            go_to_tab(target)
        # A fresh project is exactly when someone imports past work.
        _import(opt)
        return
    if not opt.objectives:
        st.info("Add a measurement in Set up to score these formulations "
                "again. Nothing recorded has been lost.")
    _best(opt)
    st.divider()
    _all_formulations(opt)
    pending = _correct(opt)
    st.divider()
    # The foot keeps its place on screen but is drawn last, so it can see a
    # confirmation armed by a click in one of the collapsed sections below it
    # and step aside on the same run — otherwise the tab shows two coloured
    # buttons until the next click.
    foot = st.container()
    st.divider()
    _progress_chart(opt)
    _undo(opt, storage)
    _delete_formulation(opt, storage)
    _import(opt)
    with foot:
        _foot(opt, pending is not None)
    if pending is not None:
        with pending['slot']:
            _save_correction(opt, storage, pending)
