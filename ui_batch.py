"""Tab 2 · Make a trial: generate formulations, make them, record results.

Nothing here fits a model or writes to disk on a plain rerun, and the result
grid deliberately avoids st.form so `Save results` can light up the moment
every kept row has a value. What the screen shows, the bench sheet and the
printable sheets show too: all three go through `scaled_recipe`.
"""
import html
from datetime import datetime

import pandas as pd
import streamlit as st

import wording
from ui_helpers import (
    TAB_RESULTS, TAB_SETUP, best_formulation_no, confirm_action,
    confirmation_open, flash, fmt_amount, fmt_setting, go_to_tab, goal_line,
    join_unit, label_with_unit, number_list, open_rows, readiness,
    saved_ok, scale_error, table_height, unit_after_number,
)

# Measurements wrap at four per row so a pilot with ten instrument readings
# still fits on screen.
_PER_ROW = 4


def _result_key(formulation_no, measurement):
    """The stable key for one measurement of one formulation. Formulation
    numbers are global and never reissued, so these keys never collide across
    trials and never need clearing."""
    return f"f{formulation_no}_{measurement}"


def _left_out(formulation_no):
    """Read the Not made flag before its checkbox is drawn — the checkbox is
    rendered last in the row, after the boxes the user came to fill."""
    return bool(st.session_state.get(f"f{formulation_no}_leave_out", False))


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
    value = st.session_state.get("scale_total")
    if value is None or float(value) <= 0:
        return None
    return float(value)


def _any_value_typed(opt):
    """True once a result has been typed into a row that is still being kept.
    A row nobody made is skipped: a disabled number_input still returns its stored
    value, which would flip the bench sheet grey on a tick."""
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


def _generate(opt, n, repeat, best_no, batch_no=None, discarded=None):
    with st.spinner(wording.GENERATING_SPINNER):
        try:
            opt.ask(n_suggestions=int(n))
        except ValueError as e:
            st.error(str(e))
            return
        except Exception:
            # A raw traceback is a dead end for a nontechnical user.
            st.error(wording.GENERATE_FAILED)
            return
    if repeat and best_no is not None:
        index = opt.index_of_formulation(best_no)
        if index is not None:
            # Named on the trial table and carried into the stored result: an
            # unexplained extra row is not a repeat, it is a mystery.
            opt.add_to_pending_batch(opt.recipe_history[index],
                                     note=wording.repeat_of_formulation(best_no))
    if batch_no is not None or discarded is not None:
        # A regenerated trial keeps its number and says which numbers retired.
        opt.set_pending_batch(opt.pending_batch, batch_no=batch_no,
                              discarded=discarded)
    st.session_state.pop("_results_upload", None)
    st.session_state.pop("scale_total", None)
    if not saved_ok(opt):
        return
    flash("success", wording.batch_ready(opt.pending_batch_no))
    st.rerun()   # a rerun, never a tab move: Generate keeps you here


def _no_batch(opt):
    # The opening value comes from session state (see app.py's _FORM_FRESH):
    # Streamlit warns on screen when a widget carries both a `value=` and a
    # session-state entry, and a project switch assigns these keys.
    st.session_state.setdefault("batch_size", 3)
    size = st.number_input(wording.NEW_FORMULATIONS_IN_BATCH, min_value=1,
                           max_value=10, step=1, key="batch_size")
    best_no = best_formulation_no(opt)
    repeat = False
    if best_no is not None:
        repeat = st.checkbox(
            wording.repeat_checkbox_label(best_no),
            key="repeat_best",
            help=wording.REPEAT_HELP,
        )
    n = int(size)
    # The repeat is a formulation the user will have to make, so the button
    # counts it: ticking the box on a trial of three makes four.
    making = n + (1 if repeat else 0)
    # While a confirmation is armed its "Yes" is the one coloured button, and
    # answering it is the one thing to do; generating can wait a click.
    lit = not confirmation_open()
    if st.button(wording.generate_button_label(making),
                 type="primary" if lit else "secondary",
                 disabled=not lit, key="generate") and lit:
        _generate(opt, n, repeat, best_no)
    if len(opt.X_history) < 5:
        st.caption(wording.FIRST_FIVE_SPREAD)
    else:
        st.caption(wording.EACH_BATCH_AIMS_CLOSER)


def _amount_format(opt, frame):
    """How each column of the trial table is written out: two decimals for the
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


def _batch_table(opt):
    rows = opt.pending_batch
    unit = opt.one_amount_unit()
    st.markdown(wording.make_these(opt.pending_batch_no, len(rows)))
    scale_to = _scale_to(opt)
    frame = opt.batch_frame(rows, scale_to=scale_to)
    st.dataframe(
        frame.style.format(_amount_format(opt, frame)),
        hide_index=True, key="batch_table", height=table_height(len(frame)),
    )
    _scale_control(opt, unit, scale_to)
    if opt.pending_batch_discarded:
        st.caption(wording.batch_discarded_caption(
            number_list(opt.pending_batch_discarded)))

    best_no = best_formulation_no(opt)
    if (opt.pending_batch_no or 0) > 1 and best_no is not None:
        index = opt.index_of_formulation(best_no)
        if index is not None:
            # Against the amounts on the table above, not the ones underneath
            # them: while the trial is scaled to a total, a change read off
            # the generated amounts is a number nothing on screen shows. The
            # formulation it compares is scaled to that same total.
            changes = opt.biggest_changes(
                opt.scaled_recipe(rows[0]['recipe'], scale_to),
                opt.scaled_recipe(opt.recipe_history[index], scale_to), n=2)
            if changes:
                # Each change in that ingredient's own unit: +10.00 ml of
                # water beside +2.00 g of protein.
                parts = ", ".join(
                    f"{name} {'+' if delta > 0 else '−'}"
                    f"{fmt_amount(abs(delta), opt.unit_of(name))}"
                    for name, delta in changes
                )
                # Named: the line reads one row of the trial, so a trial of
                # three must not sound as though it describes all of them.
                st.caption(wording.biggest_changes_caption(
                    rows[0]['formulation'], best_no, parts))


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
    if unit is None:
        st.caption(wording.NEEDS_ONE_UNIT)
        return
    st.session_state.setdefault("scale_total", None)
    # "Formulation total", not "Range": Range is the measurement's range on
    # tab 1, and one word cannot be two things across two tabs.
    st.number_input(
        wording.batch_total_label(unit),
        min_value=0.0, step=1.0, placeholder=wording.AS_GENERATED_PLACEHOLDER,
        key="scale_total",
        help=wording.BATCH_TOTAL_HELP,
    )
    if scale_to is not None:
        st.caption(wording.AMOUNTS_SHOWN_FOR_TOTAL)


def _sheet_lines(opt, row, scale_to):
    """One printable sheet as a list of plain-text lines. Shared by the on-screen
    sheets and the downloadable HTML so the two can never drift apart."""
    recipe = opt.scaled_recipe(row['recipe'], scale_to)
    made_on = opt.pending_batch_created or datetime.now().astimezone().strftime("%Y-%m-%d")
    ingredients = [v for v in opt.variables
                   if v.get('category', 'ingredient') == 'ingredient']
    process = [v for v in opt.variables if v.get('category') == 'process']
    lines = [f"{opt.project_name} · {made_on}",
             f"{wording.FORMULATION_CAP} {row['formulation']} · "
             f"{wording.BATCH} {opt.pending_batch_no}",
             ""]
    for var in ingredients:
        lines.append(f"{var['name']}: "
                     + fmt_amount(recipe.get(var['name'], 0.0),
                                  opt.unit_of(var['name'])))
    if ingredients:
        # One total per unit: "10.00 g · 40.00 ml" when the sheet mixes them.
        # A sheet with nothing to weigh out claims no total at all.
        lines.append(wording.TOTAL_PREFIX + opt.total_text(recipe))
    if process:
        if ingredients:
            lines.append("")      # a blank line only separates two lists
        for var in process:
            # A setting is not an amount, so it never wears the project's unit
            # and never the two decimals a balance works to.
            lines.append(f"{var['name']}: "
                         + fmt_setting(recipe.get(var['name'], 0.0),
                                       var.get('unit')))
    lines.append("")
    for obj in opt.measurements_by_importance():
        lines.append(wording.MEASURED_PREFIX
                     + f"{label_with_unit(obj['name'], obj.get('unit'))}"
                     f" · {goal_line(obj)}: ______________________")
    lines.append("")
    # A repeat says so on the sheet the technician carries: two sheets with
    # identical amounts and nothing printed to say why is how a formulation
    # gets made twice by mistake.
    note = str(row.get('note') or "").strip()
    lines.append(wording.note_line(note) if note else wording.NOTE_BLANK_LINE)
    lines.append(wording.NOT_MADE_CHECKBOX_SHEET)
    return lines


def _sheets_html(opt, scale_to):
    """Every sheet in one self-contained file, one page each. The in-app
    expander cannot be printed on its own — Cmd-P would take the sidebar and
    the other tabs with it."""
    blocks = []
    for row in opt.pending_batch:
        body = "\n".join(html.escape(line) for line in _sheet_lines(opt, row, scale_to))
        blocks.append(f"<pre class='sheet'>{body}</pre>")
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(opt.project_name)} {wording.BATCH} "
        f"{opt.pending_batch_no}</title>"
        "<style>"
        "body{font-family:ui-monospace,Menlo,Consolas,monospace;margin:0}"
        ".sheet{padding:24px 32px;font-size:13px;line-height:1.6;"
        "page-break-after:always;break-after:page;white-space:pre-wrap}"
        ".sheet:last-child{page-break-after:auto;break-after:auto}"
        "</style></head><body>" + "".join(blocks) + "</body></html>"
    )


def _printable(opt, scale_to):
    for row in opt.pending_batch:
        # st.text, not st.markdown: markdown italicises paired underscores and
        # turns a run of three into a horizontal rule, so a measurement named
        # L_a_b would silently mangle the sheet.
        for line in _sheet_lines(opt, row, scale_to):
            st.text(line)
        st.divider()


def _download_row(opt, scale_to):
    rows = opt.pending_batch
    # The bench sheet is the lit thing until the first result is typed, and it
    # steps aside while a confirmation is waiting for an answer.
    lit = not _any_value_typed(opt) and not confirmation_open()

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            wording.DOWNLOAD_BENCH_SHEET,
            data=opt.batch_csv(rows, scale_to=scale_to),
            file_name=f"{opt.project_name} trial {opt.pending_batch_no}.csv",
            mime="text/csv", key="download_batch_sheet",
            type="primary" if lit else "secondary",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            wording.DOWNLOAD_FORMULATION_SHEETS,
            data=_sheets_html(opt, scale_to),
            file_name=f"{opt.project_name} trial {opt.pending_batch_no} sheets.html",
            mime="text/html", key="download_sheets", use_container_width=True,
        )
    if scale_to is not None:
        # Both files carry the amounts on screen, so the size they were
        # written for is named directly under the two buttons.
        st.caption(wording.sheets_use_total_caption(
            join_unit(f"{scale_to:g}", opt.one_amount_unit() or "")))
    with st.expander(wording.PREVIEW_SHEETS):
        _printable(opt, scale_to)


def _downloads(opt):
    rows = opt.pending_batch
    scale_to = _scale_to(opt)
    numbers = [r['formulation'] for r in rows]

    # The downloads belong above `Generate a different trial`, but the question
    # has to be asked first: arming a confirmation does not rerun, so a download
    # rendered before it would still be coloured on the run that puts the
    # warning on screen. The container reserves the position instead.
    slot = st.container()
    regenerate = confirm_action(
        "regenerate", wording.GENERATE_DIFFERENT_BATCH,
        wording.regenerate_warning(opt.pending_batch_no,
                                   ", ".join(str(n) for n in numbers)),
        confirm_label=wording.YES_DISCARD,
    )
    with slot:
        _download_row(opt, scale_to)

    if regenerate:
        batch_no = opt.pending_batch_no
        n = len(rows)
        opt.set_pending_batch(None)     # the old numbers retire here
        st.session_state.pop("scale_total", None)
        st.session_state.pop("_results_upload", None)
        _generate(opt, n, False, None, batch_no=batch_no, discarded=numbers)


def _recorded_row(opt, number, ordered):
    """The read-only line a formulation gets once its results are in — a trial
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
    if any(o['name'] not in results for o in ordered):
        line = wording.recorded_line_partial(line)
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
    st.subheader(wording.RECORD_RESULTS_HEADER)
    st.caption(wording.RECORD_RESULTS_CAPTION)
    left_out, entered = set(), 0

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
        has_value = False
        cols = None
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
                has_value = has_value or value is not None
        if row.get('note'):
            # A repeat of the best formulation arrives already saying so.
            st.session_state.setdefault(f"f{number}_note", row['note'])
        # Never disabled: why a formulation was not made is the only record
        # of what went wrong, and a Note that greys out on the tick can only
        # be typed by someone who knew to type it first.
        st.text_input(wording.NOTE, key=f"f{number}_note")
        # Last in the row, per spec: the boxes the user came to fill come first.
        st.checkbox(wording.NOT_MADE_LABEL, key=f"f{number}_leave_out",
                    help=wording.NOT_MADE_HELP)
        if has_value and not skip:
            entered += 1
        st.divider()

    kept = [r for r in to_record if r['formulation'] not in left_out]
    ready = bool(kept) and all(
        any(st.session_state.get(_result_key(r['formulation'], o['name'])) is not None
            for o in opt.objectives)
        for r in kept
    )
    if to_record and not kept:
        st.info(wording.NOTHING_TO_SAVE)
    lit = ready and not confirmation_open()
    if st.button(wording.SAVE_RESULTS, type="primary" if lit else "secondary",
                 disabled=not lit, key="save_results") and lit:
        _save_results(opt, kept, left_out, to_record)
    # "filled in", not "to record": this counts the rows that HAVE a value,
    # and every other screen uses "to record" for the rows that do not
    # ("Back to trial 2 · 2 to record"). One word could not mean both.
    # Nothing here reaches the file until Save results is pressed, which is
    # the tail.
    counter = wording.filled_in_counter(entered, len(kept))
    if left_out:
        counter += wording.not_made_counter_suffix(len(left_out))
    st.caption(counter + wording.SAVED_WHEN_SUFFIX)


def _save_results(opt, kept, left_out, to_record):
    batch_no = opt.pending_batch_no
    # Check every value against its range before writing anything: a
    # half-saved trial is worse than a refused one.
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
            # Why it was not made is often typed before the box is ticked, and
            # it is the only record of what went wrong.
            note = str(st.session_state.get(f"f{number}_note") or "").strip()
            # "Not made" first, always: with only the typed note, the All
            # formulations row for a formulation nobody made said nothing
            # about not having been made.
            opt.record_skipped(number, batch_no, row['recipe'],
                               note=(wording.not_made_with_note(note) if note
                                     else wording.NOT_MADE_NOTE_PREFIX))
    opt.set_pending_batch(None)
    st.session_state.pop("scale_total", None)
    st.session_state.pop("_results_upload", None)
    if not saved_ok(opt):
        # A move would rerun past app.py's end-of-script check and hide it.
        return
    flash("success", wording.batch_recorded_flash(batch_no))
    go_to_tab(TAB_RESULTS)


def _upload(opt):
    with st.expander(wording.UPLOAD_EXPANDER):
        st.caption(wording.UPLOAD_HELP_CAPTION)
        sheet_file = st.file_uploader(
            wording.UPLOAD_RESULTS_CSV, type=["csv"],
            # Per project: an uploader cannot be emptied from session state,
            # so a shared key offered the next project this one's sheet.
            key=f"results_csv_{opt.project_name}")
        if sheet_file is not None and st.button(wording.CHECK_THIS_FILE,
                                                key="check_sheet"):
            try:
                st.session_state["_results_upload"] = pd.read_csv(sheet_file)
            except Exception:
                st.session_state.pop("_results_upload", None)
                st.error(wording.CSV_UNREADABLE)
        sheet = st.session_state.get("_results_upload")
        if sheet is None:
            return
        try:
            parsed = opt.parse_batch_results(sheet, opt.pending_batch)
        except ValueError as e:
            st.error(str(e))
            st.session_state.pop("_results_upload", None)
            return
        st.info(wording.upload_found_caption(
            len(parsed), len(opt.pending_batch),
            ", ".join(f"{wording.FORMULATION_CAP} {no}" for no, _, _ in parsed)))
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
            except (ValueError, TypeError, KeyError) as e:
                st.error(wording.could_not_save(e))
                return
            st.session_state.pop("_results_upload", None)
            left = open_rows(opt)
            if left:
                # Rows still to record: the trial stays open, numbers and all,
                # so the whole-trial sentence would be a lie.
                flash("success", wording.upload_partial_flash(
                    len(parsed), len(opt.pending_batch), batch_no, len(left)))
                st.rerun()
            opt.set_pending_batch(None)
            st.session_state.pop("scale_total", None)
            if not saved_ok(opt):
                # A move would rerun past app.py's end-of-script check and
                # hide it, and the trial is still open on disk.
                return
            flash("success", wording.batch_recorded_flash(batch_no))
            go_to_tab(TAB_RESULTS)


def render(opt, storage):
    ready, missing = readiness(opt)
    if not ready:
        _incomplete(missing)
        return
    if not opt.pending_batch:
        _no_batch(opt)
        return
    _batch_table(opt)
    _downloads(opt)
    _record_results(opt)
    _upload(opt)
