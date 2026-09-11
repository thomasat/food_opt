"""Tab 3 · Results: the best formulation, every formulation, and one
collapsed section — `Edit past formulations` — for every way the record is
fixed after the fact: a correction, a deletion, and a formulation made before
this project existed, typed in or read off a CSV.

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
    bounds_caution, clear_selection, confirm_action, confirmation_open,
    disarm, flash, fmt_amount, fmt_setting, goal_line, go_to_tab,
    label_with_unit, number_list, open_rows, other_confirmation, park_clear,
    plural, preserve_tab_forms, readiness, saved_ok, scale_error,
    table_height, take_clear,
)


def _all_numbers(opt):
    """Every number this project has issued and still holds, in order."""
    numbers = [int(n) for n in opt.formulation_ids]
    numbers += [int(s['formulation']) for s in opt.skipped]
    return sorted(numbers)


def _deletable_numbers(opt):
    """The numbers this section may offer, which is every number the project
    holds except the open batch's own.

    A batch recorded one sheet at a time keeps its recorded rows while the
    batch stays open. Deleting one of those took the result but left the row
    on tab 2's grid, waiting to be recorded all over again — a delete that
    put work back on the bench."""
    open_numbers = {int(r['formulation']) for r in (opt.pending_batch or [])
                    if isinstance(r, dict) and 'formulation' in r}
    return [n for n in _all_numbers(opt) if n not in open_numbers]


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
    # The measurements nobody took, named exactly as the All formulations
    # row names them: a score missing one is not the same number as a full
    # one, and the reader should not have to go and find out which.
    recorded = opt.results_history[index] if index < len(opt.results_history) else {}
    unmeasured = [o['name'] for o in opt.measurements_by_importance()
                  if o['name'] not in recorded]
    partial = bool(unmeasured)
    # What the ceiling MEANS is said once, under the measurements table on
    # Set up ("Every measurement at its goal scores 2.50."). Repeating it
    # here read as a claim about the formulation on screen — false whenever
    # it is off target, and flatly contradictory beside the missing ones.
    #
    # With every measurement deleted the ceiling is 0, and "Overall score
    # 0.00 of 0.00" is a sentence about nothing; the banner at the top of
    # the tab already says what to do about it.
    if ceiling > 0:
        st.caption(wording.overall_score_caption(
            float(opt.Y_history[index]), ceiling,
            number_list(unmeasured) if unmeasured else ""))
    if partial:
        st.caption(wording.PARTIAL_SCORES_CAPTION)
    return partial


def _amount_format(opt, frame):
    """How each amount column of the All formulations table is written out:
    two decimals, because that is the precision a balance works to and what
    every other table in the app shows — `Show amounts` printed 11.875 beside
    the bench sheet's 11.88. A process setting is not an amount and keeps
    fmt_setting's own rule."""
    settings = {opt._amount_column(v['name']) for v in opt.variables
                if v.get('category') == 'process'}
    amounts = {opt._amount_column(v['name']) for v in opt.variables}

    def weighed(value):
        # A row nobody made may hold no amount for a variable added later.
        if value is None or pd.isna(value):
            return ""
        return fmt_amount(value)

    return {c: (fmt_setting if c in settings else weighed)
            for c in frame.columns if c in amounts}


def _all_formulations(opt, said_partial=False):
    st.markdown(wording.ALL_FORMULATIONS_HEADING)
    o1, o2 = st.columns([2, 1])
    with o1:
        # The three options are a protocol with food_bo.history_frame, which
        # compares them verbatim; both modules read the one list in wording.
        order = st.selectbox(wording.SORT_LABEL, wording.SORT_OPTIONS,
                             key="results_order")
    with o2:
        show_amounts = st.toggle(wording.SHOW_AMOUNTS_TOGGLE, key="show_amounts")
    frame = opt.history_frame(order=order, include_amounts=show_amounts)
    st.dataframe(frame.style.format(_amount_format(opt, frame)),
                 hide_index=True, key="all_formulations",
                 height=table_height(len(frame), max_rows=20))
    # "Overall score" here is a lookup into food_bo's own history_frame
    # schema, not a header this module produces — it stays literal.
    if not said_partial and any(wording.NOT_MEASURED in str(v)
                                for v in frame["Overall score"]):
        st.caption(wording.PARTIAL_SCORES_CAPTION)
    st.download_button(wording.DOWNLOAD_ALL_FORMULATIONS_BUTTON, data=opt.history_csv(),
                       file_name=f"{opt.project_name} formulations.csv",
                       mime="text/csv", key="download_formulations",
                       help=wording.DOWNLOAD_ALL_FORMULATIONS_HELP)


def _correct_amount_key(no, name):
    """The box for one amount under `Correct a formulation`. The
    correct_amount_ prefix is what app.py empties on a project switch, so a
    half-typed correction never follows the user into the next project."""
    return f"correct_amount_{no}_{name}"


def _correct_measurement_key(no, name):
    return f"correct_{no}_{name}"


def _past_key(name):
    """The amount box for one variable under `Add a formulation you already
    made`; `past_m_` is its measurement. Both prefixes are what app.py
    empties on a project switch."""
    return f"past_{name}"


def _past_measurement_key(name):
    return f"past_m_{name}"


def _in_fours(items):
    """Each item beside the column to draw it in, four to a row. Both forms
    in this section lay their boxes out this way — the correction's and the
    typed-in past formulation's — and a grid that wrapped differently in one
    of them would read as a different kind of form."""
    cols = None
    for j, item in enumerate(items):
        if j % 4 == 0:
            cols = st.columns(min(4, len(items) - j))
        yield item, cols[j % 4]


def _recorded_amount(var, recipe):
    """What a past formulation holds for one variable.

    A row older than the ingredient has no key for it at all, and the model
    already reads that row at the variable's absent value — `_encode` fills
    the same blank the same way. So that, not None, is both what the box
    opens on and what a change is measured against: seeding from one and
    comparing against the other made every untouched row look corrected."""
    return float(recipe.get(var['name'], var.get('_absent_value', 0.0)))


def _amount_boxes(opt, key_of, recipe=None):
    """One number box per variable, keyed by `key_of(name)`. With a recipe the
    boxes open on what that row recorded; without one they open blank.

    A formulation is corrected as often for what went into the bowl — a
    misread balance, a line transposed off the bench sheet — as for what came
    off the panel, and until now only the measurements could be fixed."""
    typed = {}
    for var, col in _in_fours(opt.variables):
        with col:
            name = var['name']
            low, high = (float(b) for b in var['bounds'])
            # No min_value/max_value: an amount outside what the project
            # allows is a fact about work already done, and clamping it would
            # quietly record a formulation nobody made. It is a caution on
            # the way out, exactly as an imported amount is.
            st.session_state.setdefault(
                key_of(name),
                None if recipe is None else _recorded_amount(var, recipe))
            # The All formulations table's own header, so an amount is typed
            # in the unit that table prints it in.
            typed[name] = st.number_input(
                opt._amount_column(name), placeholder=f"{low:g}–{high:g}",
                # Only where the box opens on a recorded amount: on the
                # typed-in past formulation there is nothing to keep.
                help=(None if recipe is None
                      else wording.KEEP_RECORDED_AMOUNT_HELP),
                key=key_of(name))
    return typed


def _measurement_boxes(ordered, key_of, current=None):
    """One number box per measurement, by importance. With `current` the boxes
    open on what was recorded and a blank keeps it; without, a blank is a
    measurement nobody took and the row is stored partial."""
    typed = {}
    for obj, col in _in_fours(ordered):
        with col:
            name = obj['name']
            recorded = None if current is None else current.get(name)
            st.session_state.setdefault(
                key_of(name), None if recorded is None else float(recorded))
            # No clamping here either: the same reading refused on tab 2 must
            # be refusable here, not silently pulled back to the range end.
            typed[name] = st.number_input(
                f"{label_with_unit(name, obj.get('unit'))} · {goal_line(obj)}",
                placeholder=f"{obj['min_val']:g}–{obj['max_val']:g}",
                help=(None if current is None
                      else wording.LEAVE_BLANK_KEEP_VALUE_HELP),
                key=key_of(name))
    return typed


def _correct(opt):
    """The correction row: the select box, the amounts the formulation was
    really made with, and the measurements to retype. Returns the open
    correction, or None — the foot reads that and steps aside, because
    `Save correction` is the one lit action until the row is answered.

    Its button row is only RESERVED here. Arming a confirmation in one of the
    sections below does not rerun, so a Save correction drawn now would still
    be coloured on the very run that puts a Yes beside it; render() fills the
    slot once the confirmations have had their say."""
    numbers = [int(n) for n in opt.formulation_ids]
    if not numbers:
        # Not "No results yet.": that sentence is already the whole screen
        # above this section on a project holding nothing.
        st.caption(wording.no_formulation_to_correct_caption())
        return None
    take_clear("correct_formulation")
    # The label says what picking one DOES; the placeholder says what the box
    # holds. A bare "Formulation" on both left the reader to infer the verb
    # from a heading three rows up.
    choice = st.selectbox(wording.CORRECT_WHICH_LABEL, numbers, index=None,
                          placeholder=wording.FORMULATION_CAP,
                          key="correct_formulation")
    if opt.skipped:
        # The picker offers fewer numbers than All formulations lists, and
        # the reason is not visible from the box.
        st.caption(wording.FORMULATIONS_NOT_MADE_NO_RESULT_CAPTION)
    if choice is None:
        return None
    index = opt.index_of_formulation(choice)
    if index is None:
        return None
    recipe = opt.recipe_history[index]
    amounts = _amount_boxes(opt, lambda name: _correct_amount_key(choice, name),
                            recipe)
    ordered = opt.measurements_by_importance()
    current = opt.results_history[index]
    typed = _measurement_boxes(
        ordered, lambda name: _correct_measurement_key(choice, name), current)
    return {"choice": choice, "index": index, "ordered": ordered,
            "current": current, "typed": typed, "recipe": recipe,
            "amounts": amounts, "slot": st.container()}


def _close_correction(opt, choice):
    """Forget what was typed into this row's boxes. They are gone from the
    screen on the next run, so popping is enough — nothing is mounted in the
    browser to post the old value back."""
    for var in opt.variables:
        st.session_state.pop(_correct_amount_key(choice, var['name']), None)
    for obj in opt.objectives:
        st.session_state.pop(_correct_measurement_key(choice, obj['name']), None)
    clear_selection("correct_formulation")


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
            _close_correction(opt, choice)
            st.rerun()
    if not save:
        return
    # Amounts first, and a blank one is a refusal: nothing is written, no
    # copy is kept, and what was typed stays on screen to be finished.
    recipe = dict(pending['recipe'])
    amount_changes = []
    for var in opt.variables:
        name = var['name']
        value = pending['amounts'].get(name)
        if value is None:
            st.error(wording.ENTER_EVERY_AMOUNT)
            return
        # Against the same value the box was seeded with, so a row older than
        # the ingredient is not reported as corrected for having been looked
        # at — and does not get a copy of the project kept for nothing.
        if abs(_recorded_amount(var, pending['recipe']) - float(value)) > 1e-9:
            amount_changes.append(name)
        recipe[name] = float(value)
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
            changes.append(obj['name'])
        final[obj['name']] = float(value)
    if not changes and not amount_changes:
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
    if amount_changes:
        # The encoded row goes with the amounts, or the model keeps steering
        # the next batch towards a formulation nobody made.
        opt.edit_amounts(index, recipe)
        if not saved_ok(opt):
            return
    if changes:
        opt.edit_result(index, final)
        if not saved_ok(opt):
            return
    after = best_formulation_no(opt)
    # One sentence for the whole correction: a row can change its amounts and
    # its measurements in one save, and naming every number that moved made a
    # flash longer than the table above it.
    sentences = [wording.formulation_corrected(choice)]
    move = best_move_sentence(before, after)
    if move:
        sentences.append(move)
    sentences.append(COPY_KEPT)
    cautions = [c for c in (bounds_caution(opt, name, recipe[name])
                            for name in amount_changes) if c]
    flash("success", " ".join(sentences))
    for caution in cautions:
        flash("warning", caution)
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
    #
    # And a project that cannot make a batch — every measurement deleted, say
    # — must not offer a lit button to a tab holding a greyed Generate.
    ready, _ = readiness(opt)
    lit = ready and not confirmation_open() and not correcting
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


def _batch_numbers(opt):
    """Every batch this project still holds, in order. A batch nobody managed
    to make counts: its formulations have numbers, and this is the only place
    they can be taken out together.

    The open batch is not offered: its rows are still on the bench, and the
    list beside this box cannot hold them."""
    seen = {int(b) for b in opt.batch_history if b is not None}
    seen |= {int(s['batch']) for s in opt.skipped if s.get('batch') is not None}
    seen.discard(opt.pending_batch_no)
    return sorted(seen)


def _formulations_of_batch(opt, batch_no):
    """Every number that batch issued and the project still holds, recorded
    and not made alike."""
    numbers = [int(n) for n, b in zip(opt.formulation_ids, opt.batch_history)
               if b is not None and int(b) == int(batch_no)]
    numbers += [int(s['formulation']) for s in opt.skipped
                if s.get('batch') is not None
                and int(s['batch']) == int(batch_no)]
    return sorted(numbers)


def _disarm_delete():
    """Take the delete confirmation down when the selection that raised it is
    gone, and redraw. Without the redraw the sidebar and the other two tabs —
    already drawn this run, behind the armed flag — stay grey until the next
    click, and the button that would clear it is one of the grey ones."""
    if disarm("delete_formulations"):
        # This section is drawn above `Add a formulation you already made`,
        # so the redraw would otherwise take the half-typed formulation in it
        # with it: Streamlit discards every widget the run did not reach.
        preserve_tab_forms()
        st.rerun()


def _delete_formulations(opt, storage):
    """Any number of formulations, recorded or not made, behind one
    confirmation.

    `Delete the last batch` was a button of its own that could only ever
    reach the last one, and it named what it would take only in the
    confirmation. `Whole batch` fills the list instead, so what is about to
    go is on screen — and can be added to or taken back out — before anything
    is confirmed."""
    numbers = _deletable_numbers(opt)
    if not numbers:
        _disarm_delete()
        st.caption(wording.no_formulation_to_delete_caption())
        return
    c1, c2 = st.columns([2, 1])
    with c1:
        take_clear("delete_formulations")
        picked = st.multiselect(wording.FORMULATIONS_TO_DELETE_LABEL, numbers,
                                placeholder=wording.FORMULATION_CAP,
                                key="delete_formulations")
    with c2:
        batches = _batch_numbers(opt)
        take_clear("delete_whole_batch")
        batch = st.selectbox(wording.WHOLE_BATCH_LABEL, batches, index=None,
                             placeholder=wording.BATCH_CAP,
                             disabled=not batches, key="delete_whole_batch")
    if batch is not None:
        # A shortcut into the list beside it, never a delete of its own. The
        # widget already exists on this run, so the filled selection is
        # parked and assigned before the box is drawn again.
        offered = set(numbers)
        park_clear("delete_formulations",
                   sorted((set(int(n) for n in picked)
                           | set(_formulations_of_batch(opt, batch)))
                          & offered))
        clear_selection("delete_whole_batch")
        # Nothing has been deleted: this rerun only fills the list beside the
        # box. The form below it must survive it (see _disarm_delete).
        preserve_tab_forms()
        st.rerun()
    if not picked:
        # The list the question was asked about is empty, so the question is
        # gone from the screen with nothing left to answer it.
        _disarm_delete()
        return
    chosen = sorted(int(n) for n in picked)
    if len(chosen) == 1:
        label = wording.delete_formulation_button(chosen[0])
        question = wording.delete_formulation_warning(chosen[0])
        done = wording.formulation_deleted(chosen[0])
    else:
        label = wording.delete_formulations_button(
            plural(len(chosen), wording.FORMULATION))
        question = wording.delete_formulations_warning(number_list(chosen))
        done = wording.formulations_deleted(number_list(chosen))
    # Said once, in the confirmation: a caption above it repeats it.
    if not confirm_action("delete_formulations", label, question,
                          confirm_label=wording.YES_DELETE,
                          disabled=other_confirmation("delete_formulations")):
        return
    try:
        storage.archive(opt.project_name, "pre_delete", copy=True)
    except storage_backend.StorageError as e:
        st.error(str(e))
        return
    opt.delete_formulations(chosen)
    if not saved_ok(opt):
        return
    # A scaled table and a parsed bench sheet both name formulations that may
    # have just left the project — but the open batch's own rows are never in
    # the list above (see _deletable_numbers), so an open batch keeps both:
    # its sheet was read for formulations this delete cannot have touched.
    if not opt.pending_batch:
        st.session_state.pop("scale_total", None)
        st.session_state.pop("_results_upload", None)
    park_clear("delete_formulations", [])
    # The form below this section is not what the user just deleted from.
    preserve_tab_forms()
    flash("success", done)
    st.rerun()


def _add_past(opt):
    """A formulation made before this project existed: typed in, or read off
    a CSV. Both land through import_formulation, so both carry no batch and
    the same note, and the CSV is no longer a section of its own that had to
    be found before past work could be entered at all."""
    mode = st.radio(wording.ADD_PAST_FORMULATION_LABEL,
                    [wording.TYPE_IT_IN, wording.UPLOAD_A_CSV],
                    horizontal=True, label_visibility="collapsed",
                    key="add_past_mode")
    if mode == wording.UPLOAD_A_CSV:
        _import(opt)
    else:
        _type_in_past(opt)


def _type_in_past(opt):
    if not opt.variables:
        st.caption(wording.import_columns_caption_empty())
        return
    # Every variable, paused ones included: this formulation was made, and it
    # was made with some amount of each. Tab 2's own-formulation form pins a
    # paused variable to the value generated formulations hold it at, because
    # that one is a new formulation under today's set — this one is a fact
    # about work already done.
    _amount_boxes(opt, _past_key)
    ordered = opt.measurements_by_importance()
    # A blank measurement is a partial result here, exactly as it is in the
    # results grid and in an imported CSV.
    _measurement_boxes(ordered, _past_measurement_key)
    st.session_state.setdefault("past_note", wording.IMPORTED_NOTE)
    st.text_input(wording.NOTE, key="past_note")
    # Secondary: the foot's Start the next batch is this tab's coloured
    # button, and answering an armed confirmation outranks both.
    if st.button(wording.ADD_THIS_FORMULATION, key="add_past_formulation",
                 disabled=confirmation_open()):
        _add_typed_past(opt, ordered)


def _add_typed_past(opt, ordered):
    recipe = {}
    for var in opt.variables:
        value = st.session_state.get(_past_key(var['name']))
        if value is None:
            # No rerun: the refusal stays on screen beside the boxes, and what
            # was typed stays in them to be finished.
            st.error(wording.ENTER_EVERY_AMOUNT)
            return
        recipe[var['name']] = float(value)
    results = {}
    for obj in ordered:
        value = st.session_state.get(_past_measurement_key(obj['name']))
        problem = scale_error(obj, value)
        if problem:
            st.error(problem)
            return
        if value is not None:
            results[obj['name']] = float(value)
    note = str(st.session_state.get("past_note") or "").strip()
    try:
        opt.import_formulation(recipe, results, note or wording.IMPORTED_NOTE)
    except (ValueError, TypeError) as e:
        st.error(str(e))
        return
    if not saved_ok(opt):
        return
    # A caution, never a refusal: an amount outside what the project allows is
    # a fact about work already done, and the model learns from it.
    cautions = [c for c in (bounds_caution(opt, name, value)
                            for name, value in recipe.items()) if c]
    number = int(opt.formulation_ids[-1])
    for var in opt.variables:
        park_clear(_past_key(var['name']), None)
    for obj in ordered:
        park_clear(_past_measurement_key(obj['name']), None)
    park_clear("past_note", wording.IMPORTED_NOTE)
    flash("success", wording.formulation_added(number))
    for caution in cautions:
        flash("warning", caution)
    st.rerun()


def _import_columns(opt, rows):
    """Which column of the uploaded file holds each variable and each
    measurement, and which of them the file has no column for at all.

    Two spellings are accepted per name: the bare one the caption lists, and
    the one the downloaded All formulations file heads it with, which carries
    the unit — `Water (ml)`. Anything else in the file is ignored."""
    col_for, missing = {}, []
    headed = [(v['name'], opt._amount_column(v['name'])) for v in opt.variables]
    headed += [(o['name'], opt._measurement_column(o))
               for o in opt.measurements_by_importance()]
    for name, with_unit in headed:
        if name in rows.columns:
            col_for[name] = name
        elif with_unit in rows.columns:
            col_for[name] = with_unit
        else:
            missing.append(name)
    return col_for, missing


def _import(opt):
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
    col_for, missing = _import_columns(opt, rows)
    if missing:
        st.error(wording.missing_columns(", ".join(missing)))
        return
    blank_amounts = [c for c in variables if rows[col_for[c]].isna().any()]
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
            problem = scale_error(obj, _number(row[col_for[obj['name']]]))
            if problem:
                st.error(wording.row_error(position, problem))
                return
        for name in variables:
            caution = bounds_caution(opt, name, _number(row[col_for[name]]))
            if caution:
                cautions.append(wording.row_error(position, caution))
    imported, nothing_measured, failure, reached = 0, 0, None, 0
    try:
        for position, (_, row) in enumerate(rows.iterrows(), start=1):
            reached = position
            # A blank measurement is a partial result here too, exactly as
            # it is in the results grid and in an uploaded bench sheet.
            results = {name: float(row[col_for[name]]) for name in measurements
                       if not pd.isna(row[col_for[name]])}
            if not results:
                # A formulation nobody made: `Download all formulations`
                # includes those rows, amounts and note and all, and they
                # have no result to teach the model. Left out, and counted,
                # rather than stopping a file that is otherwise importable.
                nothing_measured += 1
                continue
            opt.import_formulation(
                {name: float(row[col_for[name]]) for name in variables},
                results)
            # Stop at the first row that did not reach the disk rather than
            # reporting a whole file as imported.
            if not saved_ok(opt):
                return
            imported += 1
    except (ValueError, TypeError) as e:
        failure = e
    if failure is not None:
        # The file's own row number, not a count: rows can now be skipped,
        # and "Stopped at row 3" has to name the row the reader can look at.
        message = wording.stopped_at_row(reached, failure)
        if imported:
            message += wording.rows_before_saved(plural(imported, wording.ROW))
        st.error(message)
        return
    st.session_state.pop("_import_rows", None)
    said = wording.imported(plural(imported, wording.FORMULATION))
    if nothing_measured:
        said += wording.rows_with_nothing_measured(
            plural(nothing_measured, wording.ROW), nothing_measured > 1)
    flash("success", said)
    for caution in cautions:
        flash("warning", caution)
    st.rerun()


def _edit_past(opt, storage):
    """`Edit past formulations`: one collapsed section for every way the
    record is fixed after the fact. Returns the open correction, or None —
    the foot above it reads that and steps aside.

    The three parts were three controls of their own, and the only way to
    enter work done before the project existed hid inside the third."""
    if not (opt.X_history or opt.skipped or opt.variables):
        return None
    with st.expander(wording.EDIT_PAST_FORMULATIONS_EXPANDER):
        st.markdown(wording.CORRECT_A_FORMULATION_HEADING)
        pending = _correct(opt)
        st.divider()
        st.markdown(wording.DELETE_FORMULATIONS_HEADING)
        # Reserved, and filled below once the form under it has been drawn.
        # Two things in the delete part rerun without touching the disk — the
        # `Whole batch` pick, which fills the list beside it, and taking a
        # stale confirmation down — and Streamlit discards the session-state
        # entry of every widget the run did not create, so either one blanked
        # the half-typed formulation in the form beneath.
        _delete_formulations(opt, storage)
        st.divider()
        st.markdown(wording.ADD_PAST_FORMULATION_HEADING)
        _add_past(opt)
    return pending


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
        # A fresh project is exactly when someone enters past work: there is
        # nothing yet to correct or delete, but the section is where both the
        # typed formulation and the CSV live.
        _edit_past(opt, storage)
        return
    if not opt.objectives:
        st.info(wording.ADD_MEASUREMENT_RESCORE_INFO)
    said_partial = _best(opt)
    st.divider()
    _all_formulations(opt, said_partial)
    st.divider()
    # The foot keeps its place on screen but is drawn last, so it can see a
    # confirmation armed by a click in one of the collapsed sections below it
    # and step aside on the same run — otherwise the tab shows two coloured
    # buttons until the next click.
    foot = st.container()
    st.divider()
    _progress_chart(opt)
    pending = _edit_past(opt, storage)
    with foot:
        _foot(opt, pending is not None)
    if pending is not None:
        with pending['slot']:
            _save_correction(opt, storage, pending)
