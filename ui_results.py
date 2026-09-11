"""Tab 3 · Results: the best formulation, every formulation, corrections.

The one coloured button is at the foot: `Start the next batch`, or `Back to
batch N` while a batch is still unrecorded. Everything destructive is behind a
confirmation that keeps a copy first and says so — once, in the confirmation.
"""
import pandas as pd
import streamlit as st

import storage as storage_backend
import wording
from ui_helpers import (
    COPY_KEPT, TAB_BATCH, TAB_SETUP, best_formulation_no, best_move_sentence,
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

    return [{wording.INGREDIENT_OR_SETTING_LABEL: name,
             wording.AMOUNT_COLUMN: amount(name, value)}
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


def _bounds_caution(opt, name, value):
    """The line for an imported amount outside what the project allows, or
    '' when it fits. Built by the same helper that refuses an out-of-range
    measurement, so the two sentences read alike.
    It is a warning, not a refusal: the amount is a fact about work already
    done, and the model learns more from it than from a blank."""
    var = next((v for v in opt.variables if v['name'] == name), None)
    if var is None or value is None:
        return ""
    low, high = (float(b) for b in var['bounds'])
    return bounds_warning(name, value, low, high, opt.unit_of(name))


def _progress_line(opt):
    last = opt.last_batch_no()   # counts a batch nobody managed to make
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
        return wording.batch_recorded_progress(last, best_before, best_now)
    return wording.batch_recorded_no_improvement(last)


def _best(opt):
    """The best formulation. Returns True when it has already said what a
    partial score cannot be compared with, so the table below does not say the
    same sentence again on the same screen."""
    index = opt.best_index()
    if index is None:
        return False
    number = int(opt.formulation_ids[index])
    batch = opt.batch_history[index]
    st.subheader(wording.best_so_far_heading(number, batch))
    line = _progress_line(opt)
    if line:
        st.caption(line)

    details = opt.closeness_details(index)
    if details:
        # Off by is a distance from a target, so it only appears once a
        # measurement has one: a column of dashes said nothing on every row.
        has_target = any(o['goal'] == 'target' for o in opt.objectives)
        st.dataframe(pd.DataFrame([
            {wording.MEASUREMENT_COLUMN: d['name'], wording.GOAL_LABEL: d['goal'],
             wording.MEASURED_COLUMN: d['measured'],
             **({wording.OFF_BY_COLUMN: d['off_by']} if has_target else {})}
            for d in details
        ]), hide_index=True, key="best_off_by",
            height=table_height(len(details)))

    st.markdown(wording.AMOUNTS_TO_MAKE_IT_HEADING)
    recipe = opt.recipe_history[index]
    st.table(pd.DataFrame(_amount_rows(opt, recipe),
                          columns=[wording.INGREDIENT_OR_SETTING_LABEL,
                                   wording.AMOUNT_COLUMN]))
    # Ingredients only: a process setting sitting at 0 is a setting, not an
    # ingredient somebody left out.
    unused = [v['name'] for v in opt.variables
              if v.get('category', 'ingredient') == 'ingredient'
              and not float(recipe.get(v['name'], 0.0))]
    if unused:
        st.caption(wording.NOT_USED_PREFIX + ", ".join(unused))
    ceiling = opt.utility_ceiling()
    # "· partial", exactly as the All formulations row writes it: a score
    # missing one measurement is not the same number as a full one.
    scored = opt.results_history[index] if index < len(opt.results_history) else {}
    partial = any(o['name'] not in scored for o in opt.objectives)
    # What the ceiling MEANS is said once, under the measurements table on
    # Set up ("Every measurement at its goal scores 2.50."). Repeating it
    # here read as a claim about the formulation on screen — false whenever
    # it is off target, and flatly contradictory beside "· partial".
    st.caption(wording.overall_score_caption(
        float(opt.Y_history[index]), ceiling, partial))
    if partial:
        st.caption(wording.PARTIAL_SCORES_CAPTION)
    return partial


def _all_formulations(opt, said_partial=False):
    st.markdown(wording.ALL_FORMULATIONS_HEADING)
    o1, o2 = st.columns([2, 1])
    with o1:
        # "Best first" and "Newest first" are also compared, verbatim, inside
        # food_bo.history_frame — they are a protocol with that module, not
        # display prose, so they stay literal here. The third option names
        # the batch, so it comes from wording (food_bo imports the same
        # constant) rather than repeating the word as its own literal.
        order = st.selectbox(wording.SORT_LABEL,
                             ["Best first", "Newest first",
                              wording.SORT_BATCH_ORDER],
                             key="results_order")
    with o2:
        show_amounts = st.toggle(wording.SHOW_AMOUNTS_TOGGLE, key="show_amounts")
    frame = opt.history_frame(order=order, include_amounts=show_amounts)
    st.dataframe(frame, hide_index=True, key="all_formulations",
                 height=table_height(len(frame), max_rows=20))
    # "Overall score" here is a lookup into food_bo's own history_frame
    # schema, not a header this module produces — it stays literal.
    if not said_partial and any("· partial" in str(v)
                                for v in frame["Overall score"]):
        st.caption(wording.PARTIAL_SCORES_CAPTION)
    st.download_button(wording.DOWNLOAD_ALL_FORMULATIONS_BUTTON, data=opt.history_csv(),
                       file_name=f"{opt.project_name} formulations.csv",
                       mime="text/csv", key="download_formulations",
                       help=wording.DOWNLOAD_ALL_FORMULATIONS_HELP)


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
    choice = st.selectbox(wording.CORRECT_A_RESULT_LABEL, numbers, index=None,
                          placeholder=wording.FORMULATION_CAP, key="correct_formulation")
    if opt.skipped:
        # The picker offers fewer numbers than All formulations lists, and
        # the reason is not visible from the box.
        st.caption(wording.FORMULATIONS_NOT_MADE_NO_RESULT_CAPTION)
    if choice is None:
        return None
    index = opt.index_of_formulation(choice)
    if index is None:
        return None
    ordered = opt.measurements_by_importance()
    if not ordered:
        st.info(wording.ADD_MEASUREMENT_BEFORE_CORRECTING_INFO)
        return None
    current = opt.results_history[index]
    typed = {}
    cols = None
    for j, obj in enumerate(ordered):
        if j % 4 == 0:
            cols = st.columns(min(4, len(ordered) - j))
        with cols[j % 4]:
            # No clamping here either: the same reading refused on tab 2 must
            # be refusable here, not silently pulled back to the range end.
            typed[obj['name']] = st.number_input(
                f"{label_with_unit(obj['name'], obj.get('unit'))} · "
                f"{goal_line(obj)}",
                value=(float(current[obj['name']])
                       if current.get(obj['name']) is not None else None),
                placeholder=f"{obj['min_val']:g}–{obj['max_val']:g}",
                help=wording.LEAVE_BLANK_KEEP_VALUE_HELP,
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
        save = st.button(wording.SAVE_CORRECTION_BUTTON, key="save_correction",
                         type="primary" if lit else "secondary",
                         disabled=not lit, use_container_width=True) and lit
    with b2:
        # The select box cannot be cleared by the user once it holds a value.
        if st.button(wording.CLOSE_BUTTON, key="done_correcting", use_container_width=True):
            clear_selection("correct_formulation")
            st.rerun()
    if not save:
        return
    if not any(v is not None for v in typed.values()):
        st.error(wording.ENTER_VALUE_AT_LEAST_ONE_ERROR)
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
        flash("success", wording.formulation_unchanged(choice))
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
        wording.formulation_corrected(choice, name, _n(name, was), _n(name, now))
        for name, was, now in changes if was is not None
    ]
    sentences += [
        wording.formulation_recorded_as(choice, name, _n(name, now))
        for name, was, now in changes if was is None
    ]
    move = best_move_sentence(before, after)
    if move:
        sentences.append(move)
    sentences.append(COPY_KEPT)
    flash("success", " ".join(sentences))
    st.rerun()


def _foot_label(opt):
    """What the foot of this tab offers. An open batch outranks everything:
    the work to do is the batch on the bench, and every other screen already
    says so in these words."""
    if opt.pending_batch:
        return wording.back_to_batch_label(opt.pending_batch_no, len(open_rows(opt)))
    return wording.START_NEXT_BATCH


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
    with st.expander(wording.PROGRESS_CHART_EXPANDER):
        if not opt.Y_history:
            st.caption(wording.NO_RESULTS_YET)
            return
        st.line_chart(pd.DataFrame({
            wording.FORMULATION_CAP: [int(n) for n in opt.formulation_ids],
            wording.OVERALL_SCORE_COLUMN: [float(y) for y in opt.Y_history],
            wording.BEST_SO_FAR_COLUMN: opt.best_so_far(),
        }).set_index(wording.FORMULATION_CAP), height=220)
        st.caption(wording.PROGRESS_CHART_CAPTION)


def _remove_batch_or_formulation(opt, storage):
    """One section for both ways a formulation leaves the project: the whole
    of the last batch, or one formulation. They were two expanders, and the
    first was a heading and a button with the same words, so clicking the
    heading read as having deleted the batch."""
    with st.expander(wording.DELETE_BATCH_OR_FORMULATION_EXPANDER):
        _undo(opt, storage)
        st.divider()
        _remove_formulation(opt, storage)


def _undo(opt, storage):
    # Left-out formulations count: a batch nobody managed to make is still
    # the last batch, and deleting must not reach past it.
    last = opt.last_batch_no()
    if last is None:
        if opt.X_history or opt.skipped:
            st.caption(wording.BATCH_NOT_RECORDED_BEFORE_VERSION_CAPTION)
        else:
            st.caption(wording.no_batch_to_delete_caption())
        return
    if opt.pending_batch:
        st.caption(wording.batch_open_record_first_caption())
        st.button(wording.DELETE_LAST_BATCH_BUTTON, disabled=True, key="undo_batch__btn")
        return
    count = sum(1 for b in opt.batch_history if b == last)
    count += sum(1 for s in opt.skipped if s.get('batch') == last)
    # Said once, in the confirmation: a caption above it repeats it.
    if confirm_action(
        "undo_batch", wording.DELETE_LAST_BATCH_BUTTON,
        # Formulations, not results: one of them may never have been made
        # and have no result at all, and formulation is the app's noun.
        wording.delete_last_batch_warning(last, plural(count, wording.FORMULATION)),
        confirm_label=wording.YES_DELETE, disabled=other_confirmation("undo_batch"),
    ):
        try:
            storage.archive(opt.project_name, "pre_delete", copy=True)
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
            flash("success", wording.batch_deleted(last))
            st.rerun()


def _remove_formulation(opt, storage):
    numbers = _all_numbers(opt)
    if not numbers:
        st.caption(wording.no_formulation_to_delete_caption())
        return
    take_clear("delete_formulation")
    choice = st.selectbox(wording.FORMULATION_TO_DELETE_LABEL, numbers, index=None,
                          placeholder=wording.FORMULATION_CAP,
                          key="delete_formulation")
    if choice is None:
        return
    go = confirm_action(
        "delete_formulation", wording.delete_formulation_button(choice),
        wording.delete_formulation_warning(choice),
        confirm_label=wording.YES_DELETE,
        disabled=other_confirmation("delete_formulation"),
    )
    # The select box cannot be cleared by the user once it holds a value.
    # The confirmation brings its own Cancel, so this one steps aside while
    # that is on screen rather than showing the word twice.
    if (not st.session_state.get("delete_formulation__pending")
            and st.button(wording.CLOSE_BUTTON, key="cancel_delete_formulation")):
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
            flash("success", wording.formulation_deleted(choice))
            st.rerun()


def _import(opt):
    with st.expander(wording.IMPORT_FORMULATIONS_EXPANDER):
        variables = [v['name'] for v in opt.variables]
        # By importance, as every other list of measurements on every tab.
        measurements = [o['name'] for o in opt.measurements_by_importance()]
        if variables or measurements:
            st.caption(wording.import_columns_caption(
                ", ".join(variables + measurements)))
        else:
            st.caption(wording.import_columns_caption_empty())
        uploaded = st.file_uploader(
            wording.UPLOAD_FORMULATIONS_CSV_LABEL, type=["csv"],
            # Per project: an uploader cannot be emptied from session state,
            # so a shared key offered the next project this one's file.
            key=f"import_csv_{opt.project_name}")
        # The parse is behind a button, as it is on tab 2: reading the file on
        # every rerun left the sheet on screen after it had been imported, and
        # a second click on Import recorded every row twice.
        if uploaded is not None and st.button(wording.CHECK_THIS_FILE,
                                              key="check_import"):
            try:
                st.session_state["_import_rows"] = pd.read_csv(uploaded)
            except Exception:
                st.session_state.pop("_import_rows", None)
                st.error(wording.CSV_UNREADABLE_RETRY)
        rows = st.session_state.get("_import_rows")
        if rows is None:
            return
        st.dataframe(rows, hide_index=True)
        missing = [c for c in variables + measurements if c not in rows.columns]
        if missing:
            st.error(wording.missing_columns(", ".join(missing)))
            return
        blank_amounts = [c for c in variables if rows[c].isna().any()]
        if blank_amounts:
            st.error(wording.blank_amount_columns(", ".join(blank_amounts)))
            return
        if not st.button(wording.IMPORT_ALL_ROWS_BUTTON, key="import_rows"):
            return
        # Nothing is recorded until the whole file has been read: a reading
        # outside its range is a typo or a range that is too narrow, and it
        # would otherwise arrive as the best formulation in the project.
        cautions = []
        for position, (_, row) in enumerate(rows.iterrows(), start=1):
            for obj in opt.objectives:
                problem = scale_error(obj, _number(row[obj['name']]))
                if problem:
                    st.error(wording.row_error(position, problem))
                    return
            for name in variables:
                caution = _bounds_caution(opt, name, _number(row[name]))
                if caution:
                    cautions.append(wording.row_error(position, caution))
        imported, failure = 0, None
        try:
            for _, row in rows.iterrows():
                # A blank measurement is a partial result here too, exactly as
                # it is in the results grid and in an uploaded bench sheet.
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
            message = wording.stopped_at_row(imported + 1, failure)
            if imported:
                message += wording.rows_before_saved(plural(imported, wording.ROW))
            st.error(message)
            return
        st.session_state.pop("_import_rows", None)
        flash("success", wording.imported(plural(imported, wording.FORMULATION)))
        for caution in cautions:
            flash("warning", caution)
        st.rerun()


def render(opt, storage):
    if not opt.X_history and not opt.skipped:
        st.markdown(wording.NO_RESULTS_YET)
        # A project with no ingredients cannot make a batch: sending the user
        # to a tab holding a greyed Generate is a lit button to a dead end.
        # And a batch already on the bench is not a first batch to make: the
        # foot of every other screen calls it "Back to batch 1 · 3 to record".
        ready, _ = readiness(opt)
        if not ready:
            label, target = wording.SET_UP_THIS_PROJECT_BUTTON, TAB_SETUP
        elif opt.pending_batch:
            label, target = _foot_label(opt), TAB_BATCH
        else:
            label, target = wording.MAKE_YOUR_FIRST_BATCH_BUTTON, TAB_BATCH
        lit = not confirmation_open()
        if st.button(label, type="primary" if lit else "secondary",
                     disabled=not lit, key="first_batch") and lit:
            go_to_tab(target)
        # A fresh project is exactly when someone imports past work.
        _import(opt)
        return
    if not opt.objectives:
        st.info(wording.ADD_MEASUREMENT_RESCORE_INFO)
    said_partial = _best(opt)
    st.divider()
    _all_formulations(opt, said_partial)
    pending = _correct(opt)
    st.divider()
    # The foot keeps its place on screen but is drawn last, so it can see a
    # confirmation armed by a click in one of the collapsed sections below it
    # and step aside on the same run — otherwise the tab shows two coloured
    # buttons until the next click.
    foot = st.container()
    st.divider()
    _progress_chart(opt)
    _remove_batch_or_formulation(opt, storage)
    _import(opt)
    with foot:
        _foot(opt, pending is not None)
    if pending is not None:
        with pending['slot']:
            _save_correction(opt, storage, pending)
