"""Tab 2 · Make a batch: generate formulations, make them, record results.

Nothing here fits a model or writes to disk on a plain rerun, and the result
grid deliberately avoids st.form so `Save results` can light up the moment
every kept row has a value. What the screen shows, the batch sheet and the
printable sheets show too: all three go through `scaled_recipe`.
"""
import html
from datetime import datetime

import pandas as pd
import streamlit as st

from ui_helpers import (
    TAB_RESULTS, TAB_SETUP, best_formulation_no, confirm_action,
    confirmation_open, flash, fmt_amount, fmt_setting, go_to_tab, goal_line,
    join_unit, label_with_unit, number_list, open_rows, plural, readiness,
    saved_ok, scale_error, table_height, unit_after_number,
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
    """Read the Leave out flag before its checkbox is drawn — the checkbox is
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
    A left-out row is skipped: a disabled number_input still returns its stored
    value, which would flip the batch sheet grey on a tick."""
    for row in open_rows(opt):
        number = row['formulation']
        if _left_out(number):
            continue
        for obj in opt.objectives:
            if st.session_state.get(_result_key(number, obj['name'])) is not None:
                return True
    return False


def _incomplete(missing):
    st.button("Generate formulations", disabled=True, key="generate")
    st.caption(missing)
    if st.button("Back to set up", key="back_to_setup"):
        go_to_tab(TAB_SETUP)


def _generate(opt, n, repeat, best_no, batch_no=None, discarded=None):
    with st.spinner("Choosing the next formulations…"):
        try:
            opt.ask(n_suggestions=int(n))
        except ValueError as e:
            st.error(str(e))
            return
        except Exception:
            # A raw traceback is a dead end for a nontechnical user.
            st.error("The app could not choose formulations this time. Try "
                     "again with fewer formulations. If it keeps happening, "
                     "loosen any limits you added recently, or open "
                     "Help › Get Help.")
            return
    if repeat and best_no is not None:
        index = opt.index_of_formulation(best_no)
        if index is not None:
            # Named on the batch table and carried into the stored result: an
            # unexplained extra row is not a repeat, it is a mystery.
            opt.add_to_pending_batch(opt.recipe_history[index],
                                     note=f"repeat of Formulation {best_no}")
    if batch_no is not None or discarded is not None:
        # A regenerated batch keeps its number and says which numbers retired.
        opt.set_pending_batch(opt.pending_batch, batch_no=batch_no,
                              discarded=discarded)
    st.session_state.pop("_results_upload", None)
    st.session_state.pop("scale_total", None)
    if not saved_ok(opt):
        return
    flash("success", f"Batch {opt.pending_batch_no} is ready to make.")
    st.rerun()   # a rerun, never a tab move: Generate keeps you here


def _no_batch(opt):
    # The opening value comes from session state (see app.py's _FORM_FRESH):
    # Streamlit warns on screen when a widget carries both a `value=` and a
    # session-state entry, and a project switch assigns these keys.
    st.session_state.setdefault("batch_size", 3)
    size = st.number_input("New formulations in this batch", min_value=1,
                           max_value=10, step=1, key="batch_size")
    best_no = best_formulation_no(opt)
    repeat = False
    if best_no is not None:
        repeat = st.checkbox(
            f"Include a repeat of Formulation {best_no}", key="repeat_best",
            help="Adds the best formulation to this batch as an extra "
                 "formulation, so you can check it again.",
        )
    n = int(size)
    # The repeat is a formulation the user will have to make, so the button
    # counts it: ticking the box on a batch of three makes four.
    making = n + (1 if repeat else 0)
    # While a confirmation is armed its "Yes" is the one coloured button, and
    # answering it is the one thing to do; generating can wait a click.
    lit = not confirmation_open()
    if st.button(f"Generate {making} formulations",
                 type="primary" if lit else "secondary",
                 disabled=not lit, key="generate") and lit:
        _generate(opt, n, repeat, best_no)
    if len(opt.X_history) < 5:
        st.caption("The first few formulations spread across the amounts you "
                   "allowed; later batches aim closer to your targets.")
    else:
        st.caption("Each batch aims closer to your targets.")
    st.caption("Only one batch is open at a time.")


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
    skip = {"Formulation", "Note"} | {c for c in frame.columns
                                      if frame[c].dtype == object}
    return {c: (fmt_setting if c in settings else "{:.2f}")
            for c in frame.columns if c not in skip}


def _batch_table(opt):
    rows = opt.pending_batch
    unit = opt.one_amount_unit()
    st.markdown(f"**Batch {opt.pending_batch_no} · make "
                f"{'this' if len(rows) == 1 else 'these'} "
                f"{plural(len(rows), 'formulation')}**")
    scale_to = _scale_to(opt)
    frame = opt.batch_frame(rows, scale_to=scale_to)
    st.dataframe(
        frame.style.format(_amount_format(opt, frame)),
        hide_index=True, key="batch_table", height=table_height(len(frame)),
    )
    if unit is None:
        # Scaling 10 g of powder and 40 ml of water to "400" is not a total
        # of anything, so the box is not offered and one line says why.
        st.caption("Scaling needs all ingredients in one unit.")
    else:
        st.session_state.setdefault("scale_total", None)
        st.number_input(
            f"Scale each formulation to a total of ({unit})" if unit
            else "Scale each formulation to a total of",
            min_value=0.0, step=1.0, placeholder="as generated",
            key="scale_total",
            help="Leave this empty to weigh out the amounts as they were "
                 "generated. Type a total and every formulation is rewritten "
                 "to it, on screen and in both downloads.",
        )
        if scale_to is None:
            st.caption("Shown as generated.")
        else:
            # The table on screen is scaled as well, so this is not a fact
            # about the downloads alone.
            st.caption("Shown and printed at a total of "
                       + join_unit(f"{scale_to:g}", unit) + ".")
    if opt.pending_batch_discarded:
        st.caption(f"Formulations {number_list(opt.pending_batch_discarded)} "
                   "were discarded and their numbers will not be used again.")

    best_no = best_formulation_no(opt)
    if (opt.pending_batch_no or 0) > 1 and best_no is not None:
        index = opt.index_of_formulation(best_no)
        if index is not None:
            changes = opt.biggest_changes(rows[0]['recipe'],
                                          opt.recipe_history[index], n=2)
            if changes:
                # Each change in that ingredient's own unit: +10.00 ml of
                # water beside +2.00 g of protein.
                parts = ", ".join(
                    f"{name} {'+' if delta > 0 else '−'}"
                    f"{fmt_amount(abs(delta), opt.unit_of(name))}"
                    for name, delta in changes
                )
                st.caption(f"Biggest changes from Formulation {best_no}: {parts}.")


def _sheet_lines(opt, row, scale_to):
    """One printable sheet as a list of plain-text lines. Shared by the on-screen
    sheets and the downloadable HTML so the two can never drift apart."""
    recipe = opt.scaled_recipe(row['recipe'], scale_to)
    made_on = opt.pending_batch_created or datetime.now().astimezone().strftime("%Y-%m-%d")
    ingredients = [v for v in opt.variables
                   if v.get('category', 'ingredient') == 'ingredient']
    process = [v for v in opt.variables if v.get('category') == 'process']
    lines = [f"{opt.project_name} · {made_on}",
             f"Formulation {row['formulation']} · batch {opt.pending_batch_no}",
             ""]
    for var in ingredients:
        lines.append(f"{var['name']}: "
                     + fmt_amount(recipe.get(var['name'], 0.0),
                                  opt.unit_of(var['name'])))
    # One total per unit: "10.00 g · 40.00 ml" when the sheet mixes them.
    lines.append("Total: " + opt.total_text(recipe))
    if process:
        lines.append("")
        for var in process:
            # A setting is not an amount, so it never wears the project's unit
            # and never the two decimals a balance works to.
            lines.append(f"{var['name']}: "
                         + fmt_setting(recipe.get(var['name'], 0.0),
                                       var.get('unit')))
    lines.append("")
    for obj in opt.measurements_by_importance():
        lines.append(f"Measured {label_with_unit(obj['name'], obj.get('unit'))}"
                     f" · {goal_line(obj)}: ______________________")
    lines.append("")
    lines.append("Note: ______________________________________________")
    lines.append("Not made [  ]")
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
        f"<title>{html.escape(opt.project_name)} batch {opt.pending_batch_no}</title>"
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
    # The batch sheet is the lit thing until the first result is typed, and it
    # steps aside while a confirmation is waiting for an answer.
    lit = not _any_value_typed(opt) and not confirmation_open()

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "Download batch sheet",
            data=opt.batch_csv(rows, scale_to=scale_to),
            file_name=f"{opt.project_name} batch {opt.pending_batch_no}.csv",
            mime="text/csv", key="download_batch_sheet",
            type="primary" if lit else "secondary",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            "Download printable sheets",
            data=_sheets_html(opt, scale_to),
            file_name=f"{opt.project_name} batch {opt.pending_batch_no} sheets.html",
            mime="text/html", key="download_sheets", use_container_width=True,
        )
    with st.expander("Printable formulation sheets"):
        _printable(opt, scale_to)


def _downloads(opt):
    rows = opt.pending_batch
    scale_to = _scale_to(opt)
    numbers = [r['formulation'] for r in rows]

    # The downloads belong above `Generate a different batch`, but the question
    # has to be asked first: arming a confirmation does not rerun, so a download
    # rendered before it would still be coloured on the run that puts the
    # warning on screen. The container reserves the position instead.
    slot = st.container()
    regenerate = confirm_action(
        "regenerate", "Generate a different batch",
        f"Batch {opt.pending_batch_no} (Formulations "
        f"{', '.join(str(n) for n in numbers)}) will be discarded. "
        "Those numbers will not be used again.",
        confirm_label="Yes, discard",
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
    if any(o['name'] not in results for o in ordered):
        line = f"{line} · (partial)" if line else "(partial)"
    note = (opt.notes_history[index] if index is not None
            and index < len(opt.notes_history) else "")
    if note:
        line = f"{line} · Note: {note}" if line else f"Note: {note}"
    st.caption(line or "recorded")


def _record_results(opt):
    rows = opt.pending_batch
    to_record = open_rows(opt)
    open_numbers = {r['formulation'] for r in to_record}
    ordered = opt.measurements_by_importance()
    st.subheader("Record results")
    st.caption("Enter your panel mean. Leave a measurement blank if it could "
               "not be scored.")
    left_out, entered = set(), 0

    for row in rows:
        number = row['formulation']
        st.markdown(f"**Formulation {number}**")
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
                # silent 10 N. Out-of-scale values are refused on save instead.
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
        st.text_input("Note", key=f"f{number}_note", disabled=skip)
        # Last in the row, per spec: the boxes the user came to fill come first.
        st.checkbox("Leave out", key=f"f{number}_leave_out",
                    help="Not made, or failed. The formulation keeps its "
                         "number and its amounts, and is stored without "
                         "results.")
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
        st.info("Nothing to save — at least one formulation needs results.")
    lit = ready and not confirmation_open()
    if st.button("Save results", type="primary" if lit else "secondary",
                 disabled=not lit, key="save_results") and lit:
        _save_results(opt, kept, left_out, to_record)
    counter = f"{entered} of {plural(len(kept), 'formulation')} entered"
    if left_out:
        counter += f" · {len(left_out)} left out"
    st.caption(counter + " · saved when you press Save results")


def _save_results(opt, kept, left_out, to_record):
    batch_no = opt.pending_batch_no
    # Check every value against its scale before writing anything: a half-saved
    # batch is worse than a refused one.
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
            st.error(f"Could not save these results: {e}")
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
            # formulations row for a left-out formulation said nothing about
            # having been left out.
            opt.record_skipped(number, batch_no, row['recipe'],
                               note=f"Not made · {note}" if note else "Not made")
    opt.set_pending_batch(None)
    st.session_state.pop("scale_total", None)
    st.session_state.pop("_results_upload", None)
    if not saved_ok(opt):
        # A move would rerun past app.py's end-of-script check and hide it.
        return
    flash("success", f"Batch {batch_no} recorded.")
    go_to_tab(TAB_RESULTS)


def _upload(opt):
    with st.expander("Or upload results from a CSV"):
        st.caption("Download the batch sheet above, fill in one column per "
                   "measurement, and upload it here. Rows are matched by "
                   "Formulation number.")
        sheet_file = st.file_uploader("Results sheet", type=["csv"],
                                      key="results_csv")
        if sheet_file is not None and st.button("Check this sheet",
                                                key="check_sheet"):
            try:
                st.session_state["_results_upload"] = pd.read_csv(sheet_file)
            except Exception:
                st.session_state.pop("_results_upload", None)
                st.error("This file could not be read as a CSV. If it came "
                         "from Excel, use File > Save As and pick CSV format.")
        sheet = st.session_state.get("_results_upload")
        if sheet is None:
            return
        try:
            parsed = opt.parse_batch_results(sheet, opt.pending_batch)
        except ValueError as e:
            st.error(str(e))
            st.session_state.pop("_results_upload", None)
            return
        st.info(f"Found results for {len(parsed)} of "
                f"{len(opt.pending_batch)} formulations: "
                + ", ".join(f"Formulation {no}" for no, _, _ in parsed) + ".")
        if st.button("Save uploaded results", key="save_uploaded"):
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
                st.error(f"Could not save these results: {e}")
                return
            st.session_state.pop("_results_upload", None)
            left = open_rows(opt)
            if left:
                # Rows still to record: the batch stays open, numbers and all,
                # so the whole-batch sentence would be a lie.
                flash("success",
                      f"Recorded {len(parsed)} of {len(opt.pending_batch)} "
                      f"formulations in batch {batch_no} · "
                      f"{len(left)} to make.")
                st.rerun()
            opt.set_pending_batch(None)
            st.session_state.pop("scale_total", None)
            if not saved_ok(opt):
                # A move would rerun past app.py's end-of-script check and
                # hide it, and the batch is still open on disk.
                return
            flash("success", f"Batch {batch_no} recorded.")
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
