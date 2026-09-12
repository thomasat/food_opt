"""Tab 2 · Make a batch: generate formulations, make them, record results.

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
    TAB_RESULTS, TAB_SETUP, best_formulation_no, bounds_caution, confirm_action,
    confirmation_open, flash, fmt_amount, fmt_setting, go_to_tab, goal_line,
    join_unit, label_with_unit, number_list, open_rows, park_clear, readiness,
    saved_ok, scale_error, table_height, unit_after_number,
)

# Measurements wrap at four per row so a pilot with ten instrument readings
# still fits on screen.
_PER_ROW = 4

# What follows a line on a printable sheet: nothing, a ruled line to write one
# number on, or a ruled area for the note.
_PROSE, _RULE, _AREA = "prose", "rule", "area"


def _result_key(formulation_no, measurement):
    """The stable key for one measurement of one formulation. Formulation
    numbers are global and never reissued, so these keys never collide across
    batches and never need clearing."""
    return f"f{formulation_no}_{measurement}"


def _left_out(formulation_no):
    """Read the Not made flag before its checkbox is drawn — the checkbox is
    rendered last in the row, after the boxes the user came to fill."""
    return bool(st.session_state.get(f"f{formulation_no}_leave_out", False))


# The (project, batch) this session has already opened the total box for.
# After that an empty box means the user emptied it, which is a value of its
# own and must reach the file.
_SEEDED_TOTAL = "_scale_total_seeded"


def _seed_scale_total(opt):
    """Open the box at the total its batch is stored with.

    A session that did not type the number knows nothing about it — a
    reopened window, a project switched back to — and drew an empty box. The
    empty box then wrote its own blank over the saved total on the very first
    render, taking `batch_totals` down with it once results were in.

    Seeded once per batch per session, and keyed by project and batch number
    because a project switch parks the box empty rather than popping it: the
    key is there, holding None, when the next project arrives. Assigning a
    widget's key is legal here and nowhere later — this runs before the box
    is created."""
    mark = (opt.project_name, opt.pending_batch_no)
    if st.session_state.get(_SEEDED_TOTAL) == mark:
        return
    st.session_state[_SEEDED_TOTAL] = mark
    stored = getattr(opt, 'pending_batch_total', None)
    if stored is not None and opt.one_amount_unit() is not None:
        st.session_state["scale_total"] = float(stored)


def _store_total(opt, scale_to):
    """Keep what the box holds with the batch, so a reopened window and tab 3
    both still know what the bench weighed out.

    A blank is written only when there was a box to blank. A project that
    weighs nothing out never draws one, and a value left behind in its
    session must not be read as the user clearing a total they were never
    shown."""
    if scale_to is None and not opt.has_ingredients():
        return
    opt.set_pending_batch_total(scale_to)


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

    A paused variable is pinned exactly as a generated formulation pins it
    (opt._frozen_value), so the stored amounts name every variable the
    project has — which is what the batch table, the sheets and tell() all
    expect. Inventing a second rule here would put one formulation's paused
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
            number_list(opt.pending_batch_discarded)))


def _batch_table(opt, scale_to):
    rows = opt.pending_batch
    frame = opt.batch_frame(rows, scale_to=scale_to)
    st.dataframe(
        frame.style.format(_amount_format(opt, frame)),
        hide_index=True, key="batch_table", height=table_height(len(frame)),
    )

    best_no = best_formulation_no(opt)
    if (opt.pending_batch_no or 0) > 1 and best_no is not None:
        index = opt.index_of_formulation(best_no)
        if index is not None:
            # Against the amounts on the table above, not the ones underneath
            # them: while the batch is scaled to a total, a change read off
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
                # Named: the line reads one row of the batch, so a batch of
                # three must not sound as though it describes all of them.
                st.caption(wording.biggest_changes_caption(
                    rows[0]['formulation'], best_no, parts))


def _scaled_cautions(opt, rows, scale_to):
    """The amounts on the table, checked against what the project allows.

    A formulation total the generated amounts were never chosen for can push
    an ingredient past its own Lowest or Highest — and the sheets are printed
    from these numbers, so the bench weighs out an amount the project says it
    does not allow. One line per ingredient, however many rows are outside:
    the fix is the same one every time.
    """
    if scale_to is None:
        return
    said = set()
    for row in rows:
        recipe = opt.scaled_recipe(row['recipe'], scale_to)
        for var in opt.variables:
            # Ingredients only: a formulation total scales what you weigh
            # out, and leaves a cook temperature exactly where it was.
            if var.get('category', 'ingredient') != 'ingredient':
                continue
            name = var['name']
            if name in said:
                continue
            caution = bounds_caution(opt, name, recipe.get(name))
            if caution:
                said.add(name)
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
    # No "Amounts shown for this total." under the box: the box holds the
    # total, the help says what it does, and the line under the two download
    # buttons names the number the files were written for. Three sentences
    # for one fact; this was the one that carried nothing of its own.


def _sheet_lines(opt, row, scale_to):
    """One printable sheet as (text, kind) pairs, kind being what follows the
    text on paper: nothing, a ruled line to write one number on, or a ruled
    area for the note. Shared by the in-app preview and the downloadable HTML
    so the two can never drift apart.

    The rules are drawn in CSS, never typed. A run of underscores wandered out
    of line the moment a measurement had a longer name than its neighbour, and
    it is not something a pen can write on straight."""
    recipe = opt.scaled_recipe(row['recipe'], scale_to)
    made_on = opt.pending_batch_created or datetime.now().astimezone().strftime("%Y-%m-%d")
    ingredients = [v for v in opt.variables
                   if v.get('category', 'ingredient') == 'ingredient']
    process = [v for v in opt.variables if v.get('category') == 'process']
    lines = [(f"{opt.project_name} · {made_on}", _PROSE),
             (f"{wording.FORMULATION_CAP} {row['formulation']} · "
              f"{wording.BATCH} {opt.pending_batch_no}", _PROSE),
             ("", _PROSE)]
    for var in ingredients:
        lines.append((f"{var['name']}: "
                      + fmt_amount(recipe.get(var['name'], 0.0),
                                   opt.unit_of(var['name'])), _PROSE))
    if ingredients:
        # One total per unit: "10.00 g · 40.00 ml" when the sheet mixes them.
        # A sheet with nothing to weigh out claims no total at all.
        lines.append((wording.TOTAL_PREFIX + opt.total_text(recipe), _PROSE))
    if process:
        if ingredients:
            lines.append(("", _PROSE))   # a blank line only separates two lists
        for var in process:
            # A setting is not an amount, so it never wears the project's unit
            # and never the two decimals a balance works to.
            lines.append((f"{var['name']}: "
                          + fmt_setting(recipe.get(var['name'], 0.0),
                                        var.get('unit')), _PROSE))
    lines.append(("", _PROSE))
    for obj in opt.measurements_by_importance():
        lines.append((wording.MEASURED_PREFIX
                      + f"{label_with_unit(obj['name'], obj.get('unit'))}"
                      f" · {goal_line(obj)}:", _RULE))
    lines.append(("", _PROSE))
    # A repeat says so on the sheet the technician carries: two sheets with
    # identical amounts and nothing printed to say why is how a formulation
    # gets made twice by mistake.
    note = str(row.get('note') or "").strip()
    lines.append((wording.note_line(note), _PROSE) if note
                 else (wording.NOTE_SHEET_LABEL, _AREA))
    lines.append((wording.NOT_MADE_CHECKBOX_SHEET, _PROSE))
    return lines


# Every rule is scoped to .fo-sheet. The preview is injected into the app's
# own page, where a bare body{} rule would restyle the whole window; the
# download wraps the same block in a document of its own.
_SHEET_CSS = (
    ".fo-sheet{font-family:ui-monospace,Menlo,Consolas,monospace;"
    "font-size:13px;line-height:1.6;padding:24px 32px;"
    "page-break-after:always;break-after:page}"
    ".fo-sheet:last-child{page-break-after:auto;break-after:auto}"
    ".fo-sheet p{margin:0;white-space:pre-wrap}"
    # currentColor, so the rules print black on paper and stay visible in
    # either of the app's themes on screen.
    ".fo-rule{display:block;height:1.4em;opacity:.5;"
    "border-bottom:1px solid currentColor}"
    ".fo-area{display:block;height:4.2em;opacity:.5;border-radius:3px;"
    "border:1px solid currentColor}"
)


def _sheets_body(opt, scale_to):
    """Every sheet as one block of HTML, styles included. This is what the
    preview renders and what the download wraps, so the printed page and the
    screen can never disagree."""
    blocks = []
    for row in opt.pending_batch:
        parts = []
        for text, kind in _sheet_lines(opt, row, scale_to):
            safe = html.escape(text)
            if kind == _RULE:
                parts.append(f"<p>{safe}<span class='fo-rule'></span></p>")
            elif kind == _AREA:
                parts.append(f"<p>{safe}<span class='fo-area'></span></p>")
            else:
                # A blank line is a blank line on paper, and an empty <p>
                # has no height at all.
                parts.append(f"<p>{safe or '&nbsp;'}</p>")
        blocks.append("<div class='fo-sheet'>" + "".join(parts) + "</div>")
    return f"<style>{_SHEET_CSS}</style>" + "".join(blocks)


def _sheets_html(opt, scale_to):
    """Every sheet in one self-contained file, one page each. The in-app
    expander cannot be printed on its own — Cmd-P would take the sidebar and
    the other tabs with it."""
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(opt.project_name)} {wording.BATCH} "
        f"{opt.pending_batch_no}</title>"
        "<style>body{margin:0}</style></head><body>"
        + _sheets_body(opt, scale_to) + "</body></html>"
    )


def _printable(opt, scale_to):
    # The same HTML the download carries, rather than a text rendering of it:
    # two renderings of one sheet is two sheets to keep in step, and st.text
    # cannot draw a line to write on. (st.markdown is still no use here — it
    # would italicise a measurement named L_a_b.)
    st.html(_sheets_body(opt, scale_to))


def _downloads(opt, scale_to):
    """Step 2: the two files to print, the total they are written to, and the
    one line that names it."""
    rows = opt.pending_batch
    # The bench sheet is the lit thing until the first result is typed, and it
    # steps aside while a confirmation is waiting for an answer.
    lit = not _any_value_typed(opt) and not confirmation_open()

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            wording.DOWNLOAD_BENCH_SHEET,
            data=opt.batch_csv(rows, scale_to=scale_to),
            file_name=f"{opt.project_name} {wording.BATCH} {opt.pending_batch_no}.csv",
            mime="text/csv", key="download_batch_sheet",
            type="primary" if lit else "secondary",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            wording.DOWNLOAD_FORMULATION_SHEETS,
            data=_sheets_html(opt, scale_to),
            file_name=f"{opt.project_name} {wording.BATCH} {opt.pending_batch_no} sheets.html",
            mime="text/html", key="download_sheets", use_container_width=True,
        )
    # The box belongs with the files it changes, not with the table: the
    # table shows what it does, the sheets are what it is for.
    _scale_control(opt, opt.one_amount_unit(), scale_to)
    if scale_to is not None:
        # Both files carry the amounts on screen, so the size they were
        # written for is named directly under them.
        st.caption(wording.sheets_show_total_caption(
            opt.batch_total_text(scale_to)))
    # The total is what pushed an amount out of what the project allows, so
    # the line reads under the box that did it rather than under a table two
    # steps above.
    _scaled_cautions(opt, rows, scale_to)


def _preview(opt, scale_to):
    with st.expander(wording.PREVIEW_SHEETS):
        _printable(opt, scale_to)


def _regenerate(opt, rows, numbers):
    batch_no = opt.pending_batch_no
    # Only the generated rows are asked for again. A formulation of the
    # user's own is theirs — it carries a note saying what it is — and
    # counting it here would quietly ask the model for one more than it
    # chose last time. A batch of nothing but own rows still regenerates
    # one, so the button never produces an empty batch.
    n = max(1, sum(1 for r in rows if not r.get('note')))
    opt.set_pending_batch(None)     # the old numbers retire here
    st.session_state.pop("scale_total", None)
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
        if row.get('note'):
            # A repeat of the best formulation arrives already saying so.
            st.session_state.setdefault(f"f{number}_note", row['note'])
        # Never disabled: why a formulation was not made is the only record
        # of what went wrong, and a Note that greys out on the tick can only
        # be typed by someone who knew to type it first.
        st.text_input(wording.NOTE, key=f"f{number}_note")
        # Last in the row, per spec: the boxes the user came to fill come first.
        st.checkbox(wording.NOT_MADE, key=f"f{number}_leave_out",
                    help=wording.NOT_MADE_HELP)
        if not skip:
            # Filled in means EVERY measurement has a value. One of three
            # typed is not a third of a result, and counting it as filled in
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
    # "filled in", not "to record": this counts the rows that HAVE a value,
    # and every other screen uses "to record" for the rows that do not
    # ("Back to batch 2 · 2 to record"). One word could not mean both.
    # Nothing here reaches the file until Save results is pressed, and the
    # line says so directly above the button that does it.
    counter = wording.filled_in_counter(entered, len(kept))
    if partly:
        counter += wording.partly_filled_suffix(partly)
    if left_out:
        counter += wording.not_made_counter_suffix(len(left_out))
    st.caption(counter + wording.SAVED_WHEN_SUFFIX)
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
            # Why it was not made is often typed before the box is ticked, and
            # it is the only record of what went wrong.
            note = str(st.session_state.get(f"f{number}_note") or "").strip()
            # "Not made" first, always: with only the typed note, the All
            # formulations row for a formulation nobody made said nothing
            # about not having been made.
            opt.record_skipped(number, batch_no, row['recipe'],
                               note=(wording.not_made_with_note(note) if note
                                     else wording.NOT_MADE))
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
                # Rows still to record: the batch stays open, numbers and all,
                # so the whole-batch sentence would be a lie.
                flash("success", wording.upload_partial_flash(
                    len(parsed), len(opt.pending_batch), batch_no, len(left)))
                st.rerun()
            opt.set_pending_batch(None)
            st.session_state.pop("scale_total", None)
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
    scale_to = _scale_to(opt)
    # Kept with the batch, so tab 3 can still say what the bench weighed out
    # once the batch is closed. A no-op on a rerun that changed nothing.
    _store_total(opt, scale_to)

    _title(opt)
    st.markdown(wording.STEP_MAKE_HEADING)
    _batch_table(opt, scale_to)
    st.markdown(wording.STEP_PRINT_HEADING)
    print_slot = st.container()
    st.markdown(wording.STEP_RECORD_HEADING)
    record_slot = st.container()
    own_slot = st.container()
    preview_slot = st.container()
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
        wording.regenerate_warning(opt.pending_batch_no,
                                   ", ".join(str(n) for n in numbers)),
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
    with preview_slot:
        _preview(opt, scale_to)
    with upload_slot:
        _upload(opt)

    if regenerate:
        _regenerate(opt, rows, numbers)
