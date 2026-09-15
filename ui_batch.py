"""Tab 2 · Make a batch: generate formulations, make them, record results.

Nothing here fits a model or writes to disk on a plain rerun, and the result
grid deliberately avoids st.form so `Save results` can light up the moment
every kept row has a value. What the screen shows, the workbook the bench
carries away shows too: both go through `scaled_recipe`.
"""
import pandas as pd
import streamlit as st

import wording
from food_bo import WORKBOOK_MIME
from ui_helpers import (
    TAB_RESULTS, TAB_SETUP, best_formulation_no, bounds_caution, confirm_action,
    confirmation_open, flash, fmt_setting, go_to_tab, goal_line,
    clear_scale_total, join_unit, label_with_unit, number_list, open_rows,
    park_clear, preserve_tab_forms, readiness, saved_ok, scale_error,
    scaled_caution, table_height, unit_after_number,
)

# Measurements wrap at four per row so a pilot with ten instrument readings
# still fits on screen.
_PER_ROW = 4


def _result_key(formulation_no, measurement):
    """The stable key for one measurement of one formulation. Formulation
    numbers are global and never reissued, so these keys never collide across
    batches and never need clearing."""
    return f"f{formulation_no}_{measurement}"


def _left_out(formulation_no):
    """Read the Not scored flag before its checkbox is drawn — the checkbox is
    rendered last in the row, after the boxes the user came to fill."""
    return bool(st.session_state.get(f"f{formulation_no}_leave_out", False))


# The (project, batch) this session has already opened the total box for.
# After that an empty box means the user emptied it, which is a value of its
# own and must reach the file.
_SEEDED_TOTAL = "_scale_total_seeded"


def _seed_mark(opt, total):
    """What the box is up to date with: whose batch it belongs to AND the
    number stored on it. The batch's identity alone was not enough — the
    stored total can change under a session that has already rendered that
    same batch (Open a saved copy, Reload project after a save error,
    reopening a project whose new batch is number 1 again), and the seed was
    skipped every time."""
    return (opt.project_name, opt.pending_batch_no, total)


def _seed_scale_total(opt):
    """Open the box at the total its batch is stored with.

    A session that did not type the number knows nothing about it — a
    reopened window, a project switched back to, a saved copy just opened — and
    drew an empty box. The empty box then wrote its own blank over the saved
    total on the very first render, taking `batch_totals` down with it once
    results were in.

    Keyed rather than done once: a project switch parks the box empty rather
    than popping it, so the key is there holding None when the next project
    arrives, and `setdefault` would have passed straight over it. Assigning a
    widget's key is legal here and nowhere later — this runs before the box
    is created."""
    if opt.has_formulation_total():
        return          # no box on this tab to seed: tab 1 holds the total
    stored = getattr(opt, 'pending_batch_total', None)
    mark = _seed_mark(opt, stored)
    if st.session_state.get(_SEEDED_TOTAL) == mark:
        return
    st.session_state[_SEEDED_TOTAL] = mark
    if stored is not None and opt.one_amount_unit() is not None:
        st.session_state["scale_total"] = float(stored)


def _store_total(opt, scale_to):
    """Keep what the box holds with the batch, so a reopened window and tab 3
    both still know what the bench weighed out.

    A blank is written only when there was a box to blank. A project that
    weighs nothing out never draws one, and a value left behind in its
    session must not be read as the user clearing a total they were never
    shown.

    The mark is re-stamped afterwards: with the stored number in it, the
    session's own write would otherwise read as a total that changed
    somewhere else, and the next run would fill a deliberately emptied box
    back in."""
    if scale_to is None and not opt.has_ingredients():
        return
    if opt.has_formulation_total():
        # The box is not drawn while the project has a total of its own, so
        # there is no answer of the user's to record — and a value left
        # behind in the session must not blank the total a past batch of this
        # project was made to.
        return
    opt.set_pending_batch_total(scale_to)
    st.session_state[_SEEDED_TOTAL] = _seed_mark(opt, scale_to)


def _scale_to(opt):
    """The total every formulation is scaled to, or None while the box is empty
    — and always None while the ingredients are not all in one unit, because
    the box is not offered then and a value left behind in it must not go on
    quietly rewriting amounts nobody can see it acting on.

    The box starts EMPTY, and empty means 'as generated'. It used to open
    pre-filled with the first formulation's total, which was wrong in both
    directions: the other rows still showed their own (larger) totals, and
    typing that same number back was a silent no-op. An empty box has one
    meaning, any number in it has the other, and clearing it undoes the
    scaling."""
    if opt.one_amount_unit() is None:
        return None
    if opt.has_formulation_total():
        return None     # tab 1's total is the answer; the box is not drawn
    value = st.session_state.get("scale_total")
    if value is None or float(value) <= 0:
        return None
    return float(value)


def _any_value_typed(opt):
    """True once a result has been typed into a row that is still being kept.
    A not-scored row is skipped: a disabled number_input still returns its stored
    value, which would flip the batch sheets grey on a tick."""
    for row in open_rows(opt):
        number = row['formulation']
        if _left_out(number):
            continue
        for obj in opt.objectives:
            if st.session_state.get(_result_key(number, obj['name'])) is not None:
                return True
    return False


def _incomplete(missing):
    st.button(wording.GENERATE_FORMULATIONS_DISABLED, disabled=True, key="generate")
    st.caption(missing)
    if st.button(wording.BACK_TO_SETUP, key="back_to_setup"):
        go_to_tab(TAB_SETUP)


def _generate(opt, n, batch_no=None, discarded=None):
    with st.spinner(wording.GENERATING_SPINNER):
        try:
            # A regenerated batch keeps its number and says which numbers
            # retired, and ask() is told both before it opens the batch: a
            # batch renumbered afterwards had already spent the number it was
            # opened with, so the batch AFTER a regenerate skipped one.
            opt.ask(n_suggestions=int(n), batch_no=batch_no,
                    discarded=discarded)
        except ValueError as e:
            st.error(str(e))
            return
        except Exception:
            # A raw traceback is a dead end for a nontechnical user.
            st.error(wording.GENERATE_FAILED)
            return
    st.session_state.pop("_results_upload", None)
    clear_scale_total()
    if not saved_ok(opt):
        return
    flash("success", wording.batch_ready(opt.pending_batch_no))
    st.rerun()   # a rerun, never a tab move: Generate keeps you here


def _no_batch(opt):
    # The opening value comes from session state (see app.py's _FORM_FRESH):
    # Streamlit warns on screen when a widget carries both a `value=` and a
    # session-state entry, and a project switch assigns these keys.
    st.session_state.setdefault("batch_size", 3)
    size = st.number_input(wording.FORMULATIONS_TO_GENERATE, min_value=1,
                           max_value=10, step=1, key="batch_size")
    n = int(size)
    # While a confirmation is armed its "Yes" is the one coloured button, and
    # answering it is the one thing to do; generating can wait a click.
    lit = not confirmation_open()
    if st.button(wording.generate_button_label(n),
                 type="primary" if lit else "secondary",
                 disabled=not lit, key="generate") and lit:
        _generate(opt, n)
    # One line for both sides of the fifth formulation, word for word the
    # How it works bullet. Two captions for the two halves of one rule left
    # a reader who had only seen one of them thinking there were two.
    st.caption(wording.HOW_CHOSEN)


def _own_key(name):
    """The box for one variable under `Add a formulation of your own`. The
    own_ prefix is what app.py empties on a project switch, so a half-typed
    formulation never follows the user into the next project."""
    return f"own_{name}"


def _own_recipe(opt):
    """What the boxes hold, or None while any of them is empty.

    A held variable is pinned exactly as a generated formulation pins it
    (opt._frozen_value), so the stored amounts name every variable the
    project has — which is what the batch table, the sheets and tell() all
    expect. Inventing a second rule here would put one formulation's held
    ingredient at a different amount from its neighbour's in the same batch.
    """
    recipe = {}
    for var in opt.active_variables():
        value = st.session_state.get(_own_key(var['name']))
        if value is None:
            return None
        recipe[var['name']] = float(value)
    for var in opt.inactive_variables():
        recipe[var['name']] = opt._frozen_value(var)
    return recipe


def _clear_own(opt):
    """Empty the form for the next one. Popping a widget key does not reach
    the browser — the mounted box posts its old value straight back — so each
    is parked and assigned before the boxes are drawn again."""
    for var in opt.active_variables():
        park_clear(_own_key(var['name']), None)
    park_clear("own_note", "")


def _start_from_best(opt, best_no):
    """Fill the boxes from the best formulation so far, and say in the note
    what the row is. Assigning the keys here would raise — the boxes already
    exist on this run — so the values are parked and land before they are
    drawn on the next one."""
    index = opt.index_of_formulation(best_no)
    if index is None:
        return
    recipe = opt.recipe_history[index]
    for var in opt.active_variables():
        # A variable added after that formulation was recorded has no amount
        # in it. Its box opens EMPTY rather than at zero: zero is an amount
        # the user never chose, and Add refuses a blank, which is the ask.
        recorded = recipe.get(var['name'])
        park_clear(_own_key(var['name']),
                   None if recorded is None else float(recorded))
    park_clear("own_note", wording.repeat_of_formulation(best_no))
    st.rerun()


def _add_own(opt):
    recipe = _own_recipe(opt)
    if recipe is None:
        # No rerun: the refusal stays on screen beside the boxes, and what
        # was typed stays in them to be finished.
        st.error(wording.ENTER_EVERY_AMOUNT)
        return
    note = str(st.session_state.get("own_note") or "").strip()
    number = opt.add_to_pending_batch(recipe, note or wording.OWN_FORMULATION_NOTE)
    if not saved_ok(opt):
        return
    # A caution, never a refusal: an amount outside what the project allows is
    # still a formulation the user means to make, and the model learns from it.
    cautions = [c for c in (bounds_caution(opt, v['name'], recipe[v['name']])
                            for v in opt.active_variables()) if c]
    _clear_own(opt)
    flash("success", wording.own_formulation_added(number, opt.pending_batch_no))
    for caution in cautions:
        flash("warning", caution)
    st.rerun()


def _own_formulation(opt):
    """`Add a formulation of your own`: the formulation the scientist wants to
    try, added to the batch beside the generated ones. It replaced the Repeat
    checkbox, which could only ever repeat the best one — `Start from the best
    so far` does that in one click and leaves the amounts editable.

    Both buttons are secondary. One button per tab is the coloured one:
    Generate before a batch is open, Save results once it is.
    """
    with st.expander(wording.ADD_OWN_EXPANDER):
        if not opt.pending_batch:
            # Adding one with nothing open opens the batch, so the order
            # matters: generated formulations can only join it first.
            st.caption(wording.ADD_OWN_NO_BATCH_CAPTION)
        for var in opt.active_variables():
            low, high = (float(b) for b in var['bounds'])
            # No min_value/max_value: clamping would turn a deliberate 30 g
            # into a silent 25, exactly as it would a measured value. The
            # allowed amounts are the placeholder, and going outside them is
            # a caution on the way in.
            st.session_state.setdefault(_own_key(var['name']), None)
            # The batch table's own header, so the box asks for the amount in
            # the unit the table beneath it prints: `Pea protein (g)`, and a
            # cook temperature in °C rather than in nothing at all.
            st.number_input(
                opt._amount_column(var['name']),
                placeholder=f"{low:g}–{high:g}",
                key=_own_key(var['name']),
            )
        st.session_state.setdefault("own_note", "")
        # The note is why this formulation is worth a place in the batch; it
        # rides onto the table, the sheet and the stored result.
        st.text_input(wording.NOTE, placeholder=wording.OWN_NOTE_PLACEHOLDER,
                      key="own_note")
        # While a confirmation is armed, answering it is the one thing to do.
        blocked = confirmation_open()
        best_no = best_formulation_no(opt)
        if best_no is None:
            add = st.button(wording.ADD_TO_THIS_BATCH, key="add_own_formulation",
                            disabled=blocked)
        else:
            c1, c2 = st.columns(2)
            with c1:
                if st.button(wording.START_FROM_BEST, key="start_from_best",
                             disabled=blocked, use_container_width=True):
                    _start_from_best(opt, best_no)
            with c2:
                add = st.button(wording.ADD_TO_THIS_BATCH,
                                key="add_own_formulation", disabled=blocked,
                                use_container_width=True)
        if add:
            _add_own(opt)


def _amount_format(opt, frame):
    """How each column of the batch table is written out: two decimals for the
    amounts and the total, because that is the precision a balance works to,
    and fmt_setting for a process setting, because a cook temperature is
    neither an amount nor 180.00 nor 188.494. The column header already
    carries the setting's unit, so only the number is formatted here."""
    settings = {opt._amount_column(v['name']) for v in opt.variables
                if v.get('category') == 'process'}
    # A total across several units is already written out ("10.00 g · 40.00
    # ml"); formatting a string as a number would raise.
    skip = {wording.FORMULATION_CAP, wording.NOTE} | {
        c for c in frame.columns if frame[c].dtype == object}
    return {c: (fmt_setting if c in settings else "{:.2f}")
            for c in frame.columns if c not in skip}


def _title(opt):
    """The batch's own heading, and the notice a regenerate owes the numbers
    it retired. Both are about the batch as a whole, so they come before the
    first step rather than inside it."""
    st.markdown(wording.make_these(opt.pending_batch_no, len(opt.pending_batch)))
    if opt.pending_batch_discarded:
        st.caption(wording.batch_discarded_caption(
            number_list(opt.pending_batch_discarded),
            len(opt.pending_batch_discarded) > 1))


def _batch_table(opt, scale_to):
    """The make-these table and the lines it owes the reader.

    Nothing is rewritten to make a row add up: a suggestion that could not be
    moved onto the total without breaking a limit stands at the band edge,
    and a formulation of the user's own stands exactly as typed. Each such
    row gets one line saying what it does add up to — the same sentence the
    sheet carries — because the table is what the bench reads before it
    prints anything.

    With no total in force at all, one line says so. The cold read watched a
    batch come out at 73 to 89 g with nothing on the screen to say the 100 g
    rule had stopped applying.
    """
    frame = opt.batch_frame(opt.pending_batch, scale_to=scale_to)
    st.dataframe(
        frame.style.format(_amount_format(opt, frame)),
        hide_index=True, key="batch_table", height=table_height(len(frame)),
    )
    if scale_to is None:
        if opt.has_ingredients():
            st.caption(wording.NOT_HELD_TO_A_TOTAL)
    else:
        for line in opt.total_mismatch_lines(opt.pending_batch, scale_to):
            st.caption(line)
    # The what-is-it-trying column is not drawn during the cold start (every
    # cell under it repeated its own header); this is the line that says what
    # those formulations are instead.
    if opt.compared_with_column() == wording.COMPARED_WITH_ALLOWED:
        st.caption(wording.HOW_CHOSEN)


def _scaled_cautions(opt, rows, scale_to):
    """The amounts on the table, checked against what the project allows.

    ONE line, however many ingredients on however many rows are outside: the
    total is what did it, and the fix is the same one every time. A caption
    per ingredient per row put eight lines of raw numbers between the box and
    the step below it, and said nothing the one line does not.
    """
    for caution in opt.scaled_cautions(rows, scale_to):
        st.caption(caution)


def _scale_control(opt, unit, scale_to):
    """The box that rewrites every formulation to a total, and the line under
    it. `unit` is the unit the ingredients share, or None when they differ.

    Two projects are offered nothing at all: one that weighs nothing out (a
    fermentation project of settings alone has no total to scale to), and one
    whose ingredients are in different units — scaling 10 g of powder and
    40 ml of water to "400" is not a total of anything, and that one says so.
    """
    if not opt.has_ingredients():
        return
    if opt.has_formulation_total():
        # The project already says how big a formulation is, and every row in
        # the table was BUILT to that total rather than rewritten to it. A
        # second box for the same number would let the bench answer the
        # question twice, differently.
        return
    if unit is None:
        st.caption(wording.NEEDS_ONE_UNIT)
        return
    st.session_state.setdefault("scale_total", None)
    # "Formulation total", not "Range": Range is the measurement's range on
    # tab 1, and one word cannot be two things across two tabs.
    st.number_input(
        wording.batch_total_label(unit),
        min_value=0.0, step=1.0, placeholder=wording.BATCH_TOTAL_PLACEHOLDER,
        key="scale_total",
        help=wording.BATCH_TOTAL_HELP,
    )
    # The same refusal tab 1's box gives the same number: a total the allowed
    # amounts cannot add up to is arithmetic with no answer, and printing the
    # sheets for it sends the bench out with amounts the project says it does
    # not allow. A warning, not a block — these rows already exist and the
    # cautions below say what it did to them.
    if scale_to is not None:
        lowest, highest = opt.total_reach()
        if scale_to > highest:
            st.caption(wording.total_not_reachable_at_most(
                opt.batch_total_text(scale_to),
                opt.batch_total_text(highest)))
        elif scale_to < lowest:
            st.caption(wording.total_not_reachable_at_least(
                opt.batch_total_text(scale_to),
                opt.batch_total_text(lowest)))
    # No "Amounts shown for this total." under the box: the box holds the
    # total, the help says what it does, and the line under the two download
    # buttons names the number the files were written for. Three sentences
    # for one fact; this was the one that carried nothing of its own.


def _downloads(opt, scale_to):
    """Step 2: the one file the bench works from, the total it is written to,
    and the one line that names it.

    One download, not three. A batch used to leave the app as a sheet to fill
    in, a set of sheets to print and a preview of those sheets on screen —
    three things to choose between before any of them could be carried to a
    bench. The workbook is all three: a summary sheet the whole batch is
    weighed out from and written back onto, and one sheet per formulation to
    print and carry."""
    rows = opt.pending_batch
    # The sheets are the lit thing until the first result is typed, and they
    # step aside while a confirmation is waiting for an answer.
    lit = not _any_value_typed(opt) and not confirmation_open()
    st.download_button(
        wording.DOWNLOAD_BATCH_SHEETS,
        data=opt.workbook_bytes(rows, scale_to),
        file_name=wording.workbook_file_name(opt.project_name,
                                             opt.pending_batch_no),
        mime=WORKBOOK_MIME, key="download_batch_sheets",
        type="primary" if lit else "secondary",
        use_container_width=True,
    )
    # The box belongs with the file it changes, not with the table: the table
    # shows what it does, the sheets are what it is for.
    _scale_control(opt, opt.one_amount_unit(), scale_to)
    if scale_to is not None:
        # The file carries the amounts on screen, so the size it was written
        # for is named directly under it — and beside it, the way to change
        # it. The number is tab 1's, and the line that names it was the one
        # place on this tab a reader could see it without being told where it
        # is set.
        line, change = st.columns([3, 1])
        with line:
            st.caption(wording.sheets_show_total_caption(
                opt.batch_total_text(scale_to)))
        with change:
            # Grey, and greyed while a confirmation is waiting: the sheets
            # are this step's lit button, and a tab never shows two.
            if st.button(wording.CHANGE_THE_TOTAL_BUTTON,
                         key="change_the_total",
                         disabled=confirmation_open(),
                         use_container_width=True):
                # The grid below has not been drawn on this run, so what the
                # bench has already typed into it is parked before the rerun
                # that leaves the tab.
                preserve_tab_forms()
                go_to_tab(TAB_SETUP)
    # The total is what pushed an amount out of what the project allows, so
    # the line reads under the box that did it rather than under a table two
    # steps above.
    _scaled_cautions(opt, rows, scale_to)


def _regenerate(opt, rows, numbers):
    batch_no = opt.pending_batch_no
    # Only the generated rows are asked for again. A formulation of the
    # user's own is theirs — it carries a note saying what it is — and
    # counting it here would quietly ask the model for one more than it
    # chose last time. A batch of nothing but own rows still regenerates
    # one, so the button never produces an empty batch.
    n = max(1, sum(1 for r in rows if not r.get('note')))
    opt.set_pending_batch(None)     # the old numbers retire here
    clear_scale_total()
    st.session_state.pop("_results_upload", None)
    _generate(opt, n, batch_no=batch_no, discarded=numbers)


def _recorded_row(opt, number, ordered):
    """The read-only line a formulation gets once its results are in — a batch
    recorded one sheet at a time reopens here with those rows already done. The
    note is part of the record, so it is shown with the numbers rather than
    being kept for tab 3."""
    index = opt.index_of_formulation(number)
    results = opt.results_history[index] if index is not None else {}
    line = " · ".join(
        join_unit(f"{label_with_unit(o['name'], o.get('unit'))} "
                  f"{float(results[o['name']]):g}",
                  unit_after_number(o.get('unit')))
        for o in ordered if o['name'] in results
    )
    unmeasured = [o['name'] for o in ordered if o['name'] not in results]
    if unmeasured:
        # Named, not "partial": the row has room to say which measurement
        # nobody took, and the reader would otherwise have to go and look.
        line = wording.recorded_line_partial(line, number_list(unmeasured))
    note = (opt.notes_history[index] if index is not None
            and index < len(opt.notes_history) else "")
    if note:
        line = wording.recorded_line_note(line, note)
    st.caption(line or wording.RECORDED_FALLBACK)


def _record_results(opt):
    rows = opt.pending_batch
    to_record = open_rows(opt)
    open_numbers = {r['formulation'] for r in to_record}
    ordered = opt.measurements_by_importance()
    st.caption(wording.RECORD_RESULTS_CAPTION)
    left_out, entered, partly = set(), 0, 0

    for row in rows:
        number = row['formulation']
        st.markdown(wording.formulation_heading(number))
        if number not in open_numbers:
            _recorded_row(opt, number, ordered)
            st.divider()
            continue
        skip = _left_out(number)
        if skip:
            left_out.add(number)
        typed = 0
        cols = None
        outside = []
        for j, obj in enumerate(ordered):
            if j % _PER_ROW == 0:
                cols = st.columns(min(_PER_ROW, len(ordered) - j))
            with cols[j % _PER_ROW]:
                # No min_value/max_value: clamping turns a 12 N reading into a
                # silent 10 N. Out-of-range values are refused on save instead.
                # The box opens empty from session state rather than from a
                # `value=`, which Streamlit warns about once a project switch
                # has assigned the key.
                st.session_state.setdefault(_result_key(number, obj['name']),
                                            None)
                value = st.number_input(
                    f"{label_with_unit(obj['name'], obj.get('unit'))} · "
                    f"{goal_line(obj)}",
                    placeholder=f"{obj['min_val']:g}–{obj['max_val']:g}",
                    key=_result_key(number, obj['name']), disabled=skip,
                )
                typed += value is not None
                if not skip:
                    problem = scale_error(obj, value)
                    if problem:
                        outside.append(problem)
        # Said as it is typed, under the boxes it was typed into: Save is
        # grey until every row has a value, so a reading outside its range
        # was refused only on a press the user could not make — and the grey
        # button explained nothing. Save still refuses it, in these words.
        for problem in outside:
            st.caption(problem)
        if row.get('note'):
            # A repeat of the best formulation arrives already saying so.
            st.session_state.setdefault(f"f{number}_note", row['note'])
        # Never disabled: why a formulation was not scored is the only
        # record of what went wrong, and a Note that greys out on the tick
        # can only be typed by someone who knew to type it first.
        st.text_input(wording.NOTE, key=f"f{number}_note")
        # Last in the row, per spec: the boxes the user came to fill come first.
        st.checkbox(wording.NOT_SCORED, key=f"f{number}_leave_out",
                    help=wording.NOT_SCORED_HELP)
        if not skip:
            # Complete means EVERY measurement has a number. One of three
            # typed is not a third of a result, and counting it as complete
            # told the user the batch was further along than it was.
            if ordered and typed == len(ordered):
                entered += 1
            elif typed:
                partly += 1
        st.divider()

    kept = [r for r in to_record if r['formulation'] not in left_out]
    ready = bool(kept) and all(
        any(st.session_state.get(_result_key(r['formulation'], o['name'])) is not None
            for o in opt.objectives)
        for r in kept
    )
    if to_record and not kept:
        st.info(wording.NOTHING_TO_SAVE)
    # "complete", not "to record": this counts the rows that HAVE every
    # measurement, and every other screen uses "to record" for the rows that
    # do not ("Back to Batch 2 · 2 to record"). One word could not mean both.
    #
    # The denominator is every row still open, ticked ones included, so it
    # matches the sheets in the technician's hand: "1 of 1 complete · 1 not
    # scored" counted a batch of two as a batch of one.
    counter = wording.complete_counter(entered, len(to_record))
    if partly:
        counter += wording.partly_filled_suffix(partly)
    if left_out:
        counter += wording.not_scored_counter_suffix(len(left_out))
    st.caption(counter)
    lit = ready and not confirmation_open()
    if st.button(wording.SAVE_RESULTS, type="primary" if lit else "secondary",
                 disabled=not lit, key="save_results") and lit:
        _save_results(opt, kept, left_out, to_record)


def _save_results(opt, kept, left_out, to_record):
    batch_no = opt.pending_batch_no
    # Check every value against its range before writing anything: a
    # half-saved batch is worse than a refused one.
    typed = {}
    for row in kept:
        number = row['formulation']
        results = {}
        for obj in opt.objectives:
            value = st.session_state.get(_result_key(number, obj['name']))
            problem = scale_error(obj, value)
            if problem:
                st.error(problem)
                return
            results[obj['name']] = value
        typed[number] = results

    for row in kept:
        number = row['formulation']
        note = str(st.session_state.get(f"f{number}_note") or "").strip()
        try:
            opt.tell(row['recipe'], typed[number], formulation_no=number,
                     batch_no=batch_no, note=note)
        except (ValueError, TypeError) as e:
            st.error(wording.could_not_save(e))
            return
        if not saved_ok(opt):
            return
    for row in to_record:
        number = row['formulation']
        if number in left_out:
            # Why it was not scored is often typed before the box is
            # ticked, and it is the only record of what went wrong.
            note = str(st.session_state.get(f"f{number}_note") or "").strip()
            # "Not scored" first, always: with only the typed note, the All
            # formulations row for a not-scored formulation said nothing
            # about having no result.
            opt.record_skipped(number, batch_no, row['recipe'],
                               note=(wording.not_scored_with_note(note) if note
                                     else wording.NOT_SCORED))
    opt.set_pending_batch(None)
    clear_scale_total()
    st.session_state.pop("_results_upload", None)
    if not saved_ok(opt):
        # A move would rerun past app.py's end-of-script check and hide it.
        return
    flash("success", wording.batch_recorded_flash(batch_no))
    go_to_tab(TAB_RESULTS)


def _read_results_file(opt, uploaded):
    """An uploaded results file in either shape it can arrive in: the
    workbook's own summary sheet, read back as one row per formulation, or a
    comma-separated file with the columns that sheet's rows are named for."""
    if str(getattr(uploaded, "name", "")).lower().endswith(".xlsx"):
        return opt.results_from_workbook(uploaded)
    return pd.read_csv(uploaded)


def _upload_preview(opt, parsed, left_out):
    """The numbers the file was read as, before anything is saved.

    The check step counted the formulations it found and showed none of
    them: a firmness of 74 written where 7.4 was meant passed it without
    anybody seeing the number. One row per formulation, in the columns the
    grid above uses.
    """
    ordered = opt.measurements_by_importance()
    rows = [{wording.FORMULATION_CAP: int(no),
             **{label_with_unit(o['name'], o.get('unit')):
                results.get(o['name']) for o in ordered},
             wording.NOT_SCORED: "", wording.NOTE: note or ""}
            for no, results, note in parsed]
    rows += [{wording.FORMULATION_CAP: int(no),
              **{label_with_unit(o['name'], o.get('unit')): None
                 for o in ordered},
              wording.NOT_SCORED: wording.TICKED_BOX, wording.NOTE: note or ""}
             for no, note in left_out]
    if not rows:
        return
    rows.sort(key=lambda r: r[wording.FORMULATION_CAP])
    frame = pd.DataFrame(rows)
    st.caption(wording.UPLOAD_PREVIEW_CAPTION)
    st.dataframe(frame.style.format(
        {c: _blank_or_number for c in frame.columns
         if frame[c].dtype != object and c != wording.FORMULATION_CAP}),
        hide_index=True, key="upload_preview",
        height=table_height(len(frame)))


def _blank_or_number(value):
    """A measurement nobody took is a blank cell, never 'nan'."""
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):g}"


def _upload(opt):
    with st.expander(wording.UPLOAD_EXPANDER):
        st.caption(wording.UPLOAD_HELP_CAPTION)
        sheet_file = st.file_uploader(
            wording.UPLOAD_RESULTS_FILE, type=["xlsx", "csv"],
            # Per project: an uploader cannot be emptied from session state,
            # so a shared key offered the next project this one's sheet.
            key=f"results_file_{opt.project_name}")
        if sheet_file is not None and st.button(wording.CHECK_THIS_FILE,
                                                key="check_sheet"):
            try:
                st.session_state["_results_upload"] = _read_results_file(
                    opt, sheet_file)
            except ValueError as e:
                # The workbook reader knows which sheet it wanted and which
                # ones the file has; that is worth more than one sentence
                # about files in general.
                st.session_state.pop("_results_upload", None)
                st.error(str(e))
            except Exception:
                st.session_state.pop("_results_upload", None)
                st.error(wording.FILE_UNREADABLE)
        sheet = st.session_state.get("_results_upload")
        if sheet is None:
            return
        try:
            parsed, left_out = opt.parse_batch_results(
                sheet, opt.pending_batch, with_skipped=True)
        except ValueError as e:
            st.error(str(e))
            st.session_state.pop("_results_upload", None)
            return
        st.info(wording.upload_found_caption(
            len(parsed), len(opt.pending_batch),
            ", ".join(f"{wording.FORMULATION_CAP} {no}" for no, _, _ in parsed))
            + (wording.not_scored_counter_suffix(len(left_out)) if left_out
               else ""))
        _upload_preview(opt, parsed, left_out)
        if st.button(wording.SAVE_UPLOADED_RESULTS, key="save_uploaded"):
            batch_no = opt.pending_batch_no
            by_number = {r['formulation']: r['recipe'] for r in opt.pending_batch}
            try:
                for number, results, note in parsed:
                    opt.tell(by_number[number], results, formulation_no=number,
                             batch_no=batch_no, note=note)
                    # Stop at the first row that did not reach the disk rather
                    # than telling the user a whole sheet was recorded.
                    if not saved_ok(opt):
                        return
                for number, note in left_out:
                    # The box was ticked on the sheet, so the row is recorded
                    # as not scored in the same words the grid's tick writes.
                    opt.record_skipped(
                        number, batch_no, by_number[number],
                        note=(wording.not_scored_with_note(note) if note
                              else wording.NOT_SCORED))
                    if not saved_ok(opt):
                        return
            except (ValueError, TypeError, KeyError) as e:
                st.error(wording.could_not_save(e))
                return
            st.session_state.pop("_results_upload", None)
            # A row ticked Not scored has been dealt with, and open_rows
            # counts only the scored ones — without this the file that
            # finished a batch left it open on a row it had just recorded.
            done = ({no for no, _, _ in parsed}
                    | {no for no, _ in left_out})
            left = [row for row in open_rows(opt)
                    if row['formulation'] not in done]
            if left:
                # Rows still to record: the batch stays open, numbers and all,
                # so the whole-batch sentence would be a lie.
                flash("success", wording.upload_partial_flash(
                    len(parsed) + len(left_out), len(opt.pending_batch),
                    batch_no, len(left)))
                st.rerun()
            opt.set_pending_batch(None)
            clear_scale_total()
            if not saved_ok(opt):
                # A move would rerun past app.py's end-of-script check and
                # hide it, and the batch is still open on disk.
                return
            flash("success", wording.batch_recorded_flash(batch_no))
            go_to_tab(TAB_RESULTS)


def render(opt, storage):
    """The open batch reads as the work: make these formulations, print the
    sheets, record what you measured — each step headed with its number, the
    three folded extras after Save, and the one button that throws the batch
    away last of all.

    The screen is DRAWN out of that order on purpose, and both reasons are
    bugs the order fixes.

    Arming a confirmation does not rerun. `Generate a different batch` now
    renders last, so a download or a Save button drawn before the arming
    click would still be coloured on the run that puts the warning on screen.
    Every step below the table reserves its position in a container; the
    question is asked first, then the slots are filled.

    And `Add to this batch` and `Start from the best so far` both end in
    st.rerun(), while Streamlit discards the session-state entry of every
    widget the run did not create — so an expander that reruns before the
    result grid exists took every measurement, note and Not-made tick
    already typed into the open batch with it. The grid is filled first.
    """
    ready, missing = readiness(opt)
    if not ready:
        _incomplete(missing)
        return
    if not opt.pending_batch:
        _no_batch(opt)
        _own_formulation(opt)
        return

    rows = opt.pending_batch
    _seed_scale_total(opt)
    typed = _scale_to(opt)
    # Kept with the batch, so tab 3 can still say what the bench weighed out
    # once the batch is closed. A no-op on a rerun that changed nothing.
    _store_total(opt, typed)
    # The project's own total wins over the box, and the one accessor is what
    # keeps the table, the workbook and tab 3 naming the same number.
    scale_to = opt.sheet_total(typed)

    _title(opt)
    st.markdown(wording.STEP_MAKE_HEADING)
    _batch_table(opt, scale_to)
    st.markdown(wording.STEP_PRINT_HEADING)
    print_slot = st.container()
    st.markdown(wording.STEP_RECORD_HEADING)
    record_slot = st.container()
    own_slot = st.container()
    upload_slot = st.container()

    # Only a formulation of the user's own carries a note when the batch is
    # opened; a generated row has none. With no generated row in it there is
    # no Generate control anywhere on this screen, and this is the button
    # that gets one back — so the line sits directly above it.
    if rows and all(r.get('note') for r in rows):
        st.caption(wording.ONLY_OWN_FORMULATIONS_CAPTION)
    numbers = [r['formulation'] for r in rows]
    regenerate = confirm_action(
        "regenerate", wording.GENERATE_DIFFERENT_BATCH,
        # The number the next generate will actually issue, read off the
        # project rather than guessed from the rows on screen.
        wording.regenerate_warning(opt.pending_batch_no,
                                   number_list(numbers),
                                   opt.next_formulation_no,
                                   len(numbers) > 1),
        confirm_label=wording.YES_DISCARD,
        # The question is asked above the grid, so its Cancel reruns before
        # the grid exists: without this it emptied every measurement, note
        # and Not-made tick the sheet already carried.
        preserve=True,
    )

    with print_slot:
        _downloads(opt, scale_to)
    with record_slot:
        _record_results(opt)
    with own_slot:
        _own_formulation(opt)
    with upload_slot:
        _upload(opt)

    if regenerate:
        _regenerate(opt, rows, numbers)
