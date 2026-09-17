"""Tab 2 · Make a round: generate formulations, make them, record results.

The result grid deliberately avoids st.form so `Save results` can light up
the moment every kept row has a value. What the screen shows, the workbook
the bench carries away shows too — and since 0.5.0 so does what is recorded:
the Batch size box moves the stored amounts (`FoodOptimizer.scale_round`)
rather than rewriting a picture of them on the way out.

One write on a plain rerun, and only one: a change to the Batch size box.
"""
import pandas as pd
import streamlit as st

import wording
from food_bo import WORKBOOK_MIME, UploadedWorkbook, uploaded_parts
from ui_helpers import (
    BATCH_SIZE_KEY, TAB_RESULTS, TAB_SETUP, amount_range_placeholder,
    best_formulation_no,
    bounds_caution, confirm_action,
    confirmation_open, flash, fmt_setting, go_to_tab, goal_line,
    clear_scale_total, join_unit, label_with_unit, number_list, open_rows,
    park_clear, readiness, saved_ok, scale_error,
    table_height, typed_batch_size, unit_after_number,
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


# The Batch size box. Its session key keeps its old spelling — clear_scale_total
# and app.py's form list are written to it — while what it asks has changed:
# it is the weight of ONE formulation in the round on screen, and changing it
# rewrites the round rather than the picture of it. Named in ui_helpers,
# because tab 1 reads the same box.
_BATCH_SIZE_KEY = BATCH_SIZE_KEY

# The (project, round, size) the box has already been opened for. The size is
# part of the mark because it is also the number scale_round was last called
# with: a box holding anything else is the bench changing it.
_SEEDED_TOTAL = "_scale_total_seeded"

# Set by a size that was refused, read by the seeding on the next run: the
# box goes back to the size the round is actually made to, rather than
# sitting on a number nothing on the page is a number of.


def _seed_mark(opt, size):
    """What the box is up to date with: whose round it belongs to AND the
    size on it. The round's identity alone was not enough — the stored size
    can change under a session that has already rendered that same round
    (Open a saved copy, Reload project after a save error, reopening a
    project whose new round is number 1 again), and the seed was skipped
    every time.

    An EMPTIED box is not an answer. There is no size of nothing: the round
    keeps the amounts the box already moved, and clearing the stored size
    while leaving those amounts would be worse than either. So the box is
    re-seeded from the round's own size on the next run — it never sits
    blank over a round that has one, saying nothing about the numbers
    printed under it."""
    return (opt.project_name, opt.pending_batch_no, size)


def _prefill(opt):
    """What the Batch size box opens holding, or None for a blank box.

    Only a number the round is ALREADY made to: the size it is stored with,
    else the project's default batch size. Nothing else — a box that opened
    at the mean of the rows' own sums put a number on screen that no
    formulation weighed, refused it out loud when the allowed amounts could
    not reach it, and could not be confirmed by typing it back. A blank box
    and the placeholder ask the question without answering it.
    """
    stored = getattr(opt, 'pending_batch_total', None)
    if stored is not None:
        return float(stored)
    project = getattr(opt, 'formulation_total', None)
    return None if project is None else float(project)


def _seed_batch_size(opt):
    """Open the box at the size the round is being made to.

    A session that did not type the number knows nothing about it — a
    reopened window, a project switched back to, a saved copy just opened —
    and drew an empty box. Keyed rather than done once: a project switch
    parks the box empty rather than popping it, so the key is there holding
    None when the next project arrives, and `setdefault` would have passed
    straight over it. Assigning a widget's key is legal here and nowhere
    later — this runs before the box is created."""
    if opt.one_amount_unit() is None:
        return          # no box is drawn while the units differ
    size = _prefill(opt)
    mark = _seed_mark(opt, size)
    emptied = size is not None and typed_batch_size(opt) is None
    if st.session_state.get(_SEEDED_TOTAL) == mark and not emptied:
        return
    st.session_state[_SEEDED_TOTAL] = mark
    if size is not None:
        st.session_state[_BATCH_SIZE_KEY] = float(size)


def _previous_size_line(opt):
    """The line this round owes the last one when it is not being made to
    the same size, or "" when there is nothing to say.

    Only for a round with no size of its own: one the bench has re-sized is
    at a number the bench typed, and saying where it came from is telling
    them what they just did. A size belongs to the round it was typed for,
    and the next round starts at the project's default — silently, until
    now, with the old number still on the tab next door.
    """
    if getattr(opt, 'pending_batch_total', None) is not None:
        return ""
    size = opt.open_round_size()
    no = opt.pending_batch_no
    if size is None or no is None:
        return ""
    previous = opt.last_batch_no()
    if previous is None:
        return ""
    was = opt.batch_total(previous)
    if was is None or abs(float(was) - float(size)) < 1e-9:
        return ""
    return wording.round_uses_the_default_size(
        no, opt.batch_total_text(size), previous,
        opt.batch_total_text(was))


def _unreachable(opt, typed):
    """Why this size cannot be made, or "" when it can.

    The same arithmetic Set up's box does, in the same sentences, with the
    noun of THIS box: a size outside what the allowed amounts add up to has
    no answer, and no search can find one.

    A project with a row written = rest has no top at all — that row takes
    whatever is left, so any size the other rows fit inside can be made —
    and its floor is said in the balance's own words, which name the row
    and what the others need. Reading the batch size at both ends told one
    round its floor and its ceiling were both 100 g, and the box accepted
    nothing at all.
    """
    if typed is None:
        return ""
    lowest, highest = opt.total_reach()
    if typed > highest:
        return wording.total_not_reachable_at_most(
            opt.batch_total_text(typed), opt.batch_total_text(highest),
            noun=wording.BATCH_SIZE_NOUN)
    if typed < lowest:
        balance = opt.balance_row_name()
        if balance is not None:
            return wording.balance_would_go_negative(
                balance, opt.batch_total_text(typed),
                opt.batch_total_text(lowest),
                unit=opt.unit_of(balance),
                noun=wording.BATCH_SIZE_NOUN)
        return wording.total_not_reachable_at_least(
            opt.batch_total_text(typed), opt.batch_total_text(lowest),
            noun=wording.BATCH_SIZE_NOUN)
    return ""


def _apply_batch_size(opt, typed):
    """Make the round to the size in the box, the moment that size changes,
    and hand back the refusal when it cannot be made at all.

    Only a CHANGE scales. The box opens prefilled, and a round the bench has
    not re-sized keeps the amounts the model chose — including the one
    suggestion a limit would not let be snapped onto the project's default,
    which says so in its own line under the table. Scaling on the prefill
    would rewrite that row behind the caption explaining it.

    A size the ingredients cannot add up to is not made. The app printed
    sheets for 250 g directly under a line saying 250 g was impossible, and
    sent the bench out with amounts the project says it does not allow. The
    round keeps the size it has, the box keeps the number that was refused
    next to the line that says why, and the refusal is the only thing that
    changes on screen.
    """
    if typed is None:
        return ""
    mark = st.session_state.get(_SEEDED_TOTAL)
    applied = mark[2] if isinstance(mark, tuple) and len(mark) == 3 else None
    if applied is not None and float(applied) == typed:
        return ""
    trouble = _unreachable(opt, typed)
    if trouble:
        # The refused number stays in the box, beside the reason it was not
        # made: re-seeding the box on the next run took the NEXT number the
        # bench typed with it (a reachable 120 after a refused 150 came back
        # as the old 100), and a box that changes under the hand is worse
        # than one holding a number the line beneath it explains.
        return trouble + " " + wording.round_stays_at(
            opt.batch_total_text(opt.open_round_size()))
    opt.scale_round(typed)
    st.session_state[_SEEDED_TOTAL] = _seed_mark(opt, typed)
    return ""


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
    # "how_many", not "batch_size": Batch size is the weight of one
    # formulation, and this box asks how many of them to make.
    st.session_state.setdefault("how_many", 3)
    how_many = st.number_input(wording.FORMULATIONS_TO_GENERATE, min_value=1,
                               max_value=10, step=1, key="how_many")
    n = int(how_many)
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

    A fixed variable is pinned exactly as a generated formulation pins it —
    at the one amount its Lowest and Highest agree on — so the stored amounts
    name every variable the project has, which is what the batch table, the
    sheets and tell() all expect. Inventing a second rule here would put one
    formulation's fixed ingredient at a different amount from its
    neighbour's in the same batch.
    """
    recipe = {}
    for var in opt.varying_variables():
        value = st.session_state.get(_own_key(var['name']))
        if value is None:
            return None
        recipe[var['name']] = float(value)
    for var in opt.fixed_variables():
        recipe[var['name']] = opt._fixed_value(var)
    return recipe


# What `Start from the best so far` typed into the boxes, and which
# formulation it came from. Kept so the form can say, before Add, that the
# app's own numbers sit outside today's allowed amounts — and so that being
# told off for them afterwards is not the first the reader hears of it.
_PREFILLED_FROM = "_own_prefilled_from"
_PREFILLED_AMOUNTS = "_own_prefilled_amounts"


def _clear_own(opt):
    """Empty the form for the next one. Popping a widget key does not reach
    the browser — the mounted box posts its old value straight back — so each
    is parked and assigned before the boxes are drawn again."""
    for var in opt.varying_variables():
        park_clear(_own_key(var['name']), None)
    park_clear("own_note", "")
    st.session_state.pop(_PREFILLED_FROM, None)
    st.session_state.pop(_PREFILLED_AMOUNTS, None)


def _start_from_best(opt, best_no):
    """Fill the boxes from the best formulation so far, and say in the note
    what the row is. Assigning the keys here would raise — the boxes already
    exist on this run — so the values are parked and land before they are
    drawn on the next one."""
    index = opt.index_of_formulation(best_no)
    if index is None:
        return
    recipe = opt.recipe_history[index]
    prefilled = {}
    for var in opt.varying_variables():
        # A variable added after that formulation was recorded has no amount
        # in it. Its box opens EMPTY rather than at zero: zero is an amount
        # the user never chose, and Add refuses a blank, which is the ask.
        recorded = recipe.get(var['name'])
        # To the precision a balance works to, like every other amount on
        # screen: the boxes opened at 21.738359306488338.
        value = None if recorded is None else round(float(recorded), 2)
        prefilled[var['name']] = value
        park_clear(_own_key(var['name']), value)
    park_clear("own_note", wording.repeat_of_formulation(best_no))
    st.session_state[_PREFILLED_FROM] = int(best_no)
    st.session_state[_PREFILLED_AMOUNTS] = prefilled
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
    # Not said twice, though: amounts the app typed into the boxes itself have
    # already been said once, in the form, before Add was pressed.
    cautions = ([] if _prefilled_unchanged(opt) else
                [c for c in (bounds_caution(opt, v['name'], recipe[v['name']])
                             for v in opt.varying_variables()) if c])
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
        for var in opt.varying_variables():
            low, high = (float(b) for b in var['bounds'])
            # No min_value/max_value: clamping would turn a deliberate 30 g
            # into a silent 25, exactly as it would a measured value. The
            # allowed amounts are the placeholder, and going outside them is
            # a caution on the way in.
            st.session_state.setdefault(_own_key(var['name']), None)
            # The batch table's own header, so the box asks for the amount in
            # the unit the table beneath it prints: `Pea protein (g)`, and a
            # cook temperature in °C rather than in nothing at all. An amount
            # reads to two decimals, as it does in every table and on every
            # sheet; a setting is dialled in and keeps its own.
            st.number_input(
                opt._amount_column(var['name']),
                placeholder=amount_range_placeholder(low, high),
                key=_own_key(var['name']),
                **({} if var.get('category') == 'process'
                   else {"format": "%.2f"}),
            )
        _worked_out_boxes(opt)
        _prefilled_caption(opt)
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


def _worked_out_boxes(opt):
    """The rows that are worked out, under the ones that are typed: greyed,
    and holding the amount the formula makes of what is in the boxes above.

    A worked-out row has no box of its own to type into — its amount is
    arithmetic over the others — but the bench still weighs it, so leaving
    it off the form would have asked for a formulation the table then showed
    a row of that nobody had seen. It fills in as the boxes above it do; it
    reads `worked out` until they are all answered.
    """
    worked = [v for v in opt.variables if opt.has_formula(v)]
    if not worked:
        return
    st.caption(wording.WORKED_OUT_BOXES_CAPTION)
    typed = {var['name']: st.session_state.get(_own_key(var['name']))
             for var in opt.varying_variables()}
    filled = {}
    if all(value is not None for value in typed.values()):
        filled = opt.fill_formulas(
            {**{name: float(value) for name, value in typed.items()},
             **{v['name']: opt._fixed_value(v)
                for v in opt.fixed_variables()}})
    for var in worked:
        value = filled.get(var['name'])
        # No key. A keyed box keeps the first value it was given — session
        # state wins over the argument on every later run — and this one has
        # to follow the boxes above it as they are typed into.
        st.number_input(
            opt._amount_column(var['name']),
            value=None if value is None else round(float(value), 2),
            placeholder=wording.WORKED_OUT, disabled=True,
            **({} if var.get('category') == 'process'
               else {"format": "%.2f"}),
        )


def _prefilled_unchanged(opt):
    """True while the boxes still hold exactly what Start from the best so
    far typed into them. A number the reader has since changed is theirs,
    and the caution on the way in belongs to it."""
    prefilled = st.session_state.get(_PREFILLED_AMOUNTS)
    if not prefilled:
        return False
    for var in opt.varying_variables():
        held = st.session_state.get(_own_key(var['name']))
        want = prefilled.get(var['name'])
        if (held is None) != (want is None):
            return False
        if held is not None and round(float(held), 2) != round(float(want), 2):
            return False
    return True


def _prefilled_caption(opt):
    """The one line the form owes amounts the app typed into it that today's
    Lowest and Highest no longer allow. Said here, before Add, rather than
    as a warning after it: the reader was told off for numbers they had not
    chosen."""
    no = st.session_state.get(_PREFILLED_FROM)
    if no is None or not _prefilled_unchanged(opt):
        return
    outside = any(bounds_caution(opt, var['name'],
                                 st.session_state.get(_own_key(var['name'])))
                  for var in opt.varying_variables()
                  if st.session_state.get(_own_key(var['name'])) is not None)
    if outside:
        st.caption(wording.prefilled_outside_caption(no))


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
    # The size first, then the corrections line. Three captions under one
    # table, two of them about the size, used to sit with the one that is
    # not in the middle of them.
    if scale_to is None:
        if opt.has_ingredients():
            st.caption(wording.NO_BATCH_SIZE_OF_ITS_OWN)
    else:
        for line in opt.total_mismatch_lines(opt.pending_batch, scale_to):
            st.caption(line)
    # The table is read only, whatever it holds — a worked-out row's amount
    # included. What was actually weighed is corrected where it is
    # recorded, not by overtyping a cell here.
    st.caption(wording.CORRECTIONS_ON_RESULTS_CAPTION)
    # The what-is-it-trying column is not drawn during the cold start (every
    # cell under it repeated its own header); this is the line that says what
    # those formulations are instead.
    if opt.compared_with_column() == wording.COMPARED_WITH_ALLOWED:
        st.caption(wording.HOW_CHOSEN)


def _scaled_cautions(opt, rows, scale_to, sized):
    """The amounts on the table, checked against what the project allows.

    ONE line, however many ingredients on however many rows are outside: the
    total is what did it, and the fix is the same one every time. A caption
    per ingredient per row put eight lines of raw numbers between the box and
    the step below it, and said nothing the one line does not.
    """
    for caution in opt.scaled_cautions(rows, scale_to, sized):
        st.caption(caution)


def _batch_size_control(opt, unit, typed, scale_to, sized, refusal=""):
    """The Batch size box, above the round table, and the lines under it.
    `unit` is the unit the ingredients share, or None when they differ.

    Always drawn, whether or not Set up holds a default: the size of what the
    bench is about to weigh out is the first thing it needs to know, and
    until 0.5.0 a project with a default showed nothing here at all — the
    number was two tabs away, and making ONE round bigger meant editing the
    project. Changing it here makes this round to that size and leaves the
    default alone.

    One project is offered nothing: one that weighs nothing out (a
    fermentation project of settings alone has no size), and one whose
    ingredients are in different units — making 10 g of powder and 40 ml of
    water "400" is not a size of anything, and that one says so.
    """
    if not opt.has_ingredients():
        return
    if unit is None:
        st.caption(wording.NEEDS_ONE_UNIT)
        return
    st.session_state.setdefault(_BATCH_SIZE_KEY, None)
    st.number_input(
        wording.batch_size_label(unit),
        min_value=0.0, step=1.0, placeholder=wording.BATCH_SIZE_PLACEHOLDER,
        key=_BATCH_SIZE_KEY,
        help=wording.batch_size_help(unit),
    )
    line = _previous_size_line(opt)
    if line:
        st.caption(line)
    # The refusal the size earned, worked out where the size is applied so
    # that the sentence on screen and the round on disk can never disagree:
    # the round was NOT made to this size, and the table, the download and
    # its caption below are all still the size it was made to.
    if refusal:
        st.caption(refusal)
    # The size is what pushed an amount out of what the project allows, so
    # the line reads under the box that did it.
    _scaled_cautions(opt, opt.pending_batch, scale_to, sized)


def _downloads(opt, scale_to, sized, said_size=False):
    """Step 2: the one file the bench works from, the size it is written to,
    and the one line that names it.

    One download, not three. A round used to leave the app as a sheet to fill
    in, a set of sheets to print and a preview of those sheets on screen —
    three things to choose between before any of them could be carried to a
    bench. The workbook is all three: a summary sheet the whole round is
    weighed out from and written back onto, and one sheet per formulation to
    print and carry.

    No `Change the total` beside the line any more: the Batch size box is on
    this screen, above the table, and a button that left the tab to answer a
    question the tab already asks was the long way round.
    """
    rows = opt.pending_batch
    # The sheets are the lit thing until the first result is typed, and they
    # step aside while a confirmation is waiting for an answer.
    lit = not _any_value_typed(opt) and not confirmation_open()
    st.download_button(
        wording.DOWNLOAD_BATCH_SHEETS,
        data=opt.workbook_bytes(rows, scale_to, sized),
        file_name=wording.workbook_file_name(opt.project_name,
                                             opt.pending_batch_no),
        mime=WORKBOOK_MIME, key="download_batch_sheets",
        type="primary" if lit else "secondary",
        use_container_width=True,
    )
    if scale_to is not None and not said_size:
        # The file carries the amounts on screen, so the size it was written
        # for is named directly under it — unless the refusal above the
        # table has just said that same number ("The round stays at 100 g"),
        # in which case this repeats one fact with a different subject one
        # line apart.
        st.caption(wording.sheets_show_total_caption(
            opt.batch_total_text(scale_to)))


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
    # do not ("Back to Round 2 · 2 to record"). One word could not mean both.
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
    comma-separated file with the columns that sheet's rows are named for.

    Both come back as an UploadedWorkbook, so what the Save button reads is
    one shape: a file of columns says nothing about lots or about what was
    weighed, and says it by carrying nothing.
    """
    if str(getattr(uploaded, "name", "")).lower().endswith(".xlsx"):
        return opt.results_from_workbook(uploaded)
    return UploadedWorkbook(pd.read_csv(uploaded), {}, {})


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
        # The rows, what the Actual cells said, and the lots: three things
        # the file carried, kept apart all the way to Save.
        sheet, weighed, lots = uploaded_parts(
            st.session_state.get("_results_upload"))
        if sheet is None:
            return
        try:
            parsed, left_out = opt.parse_batch_results(
                sheet, opt.pending_batch, with_skipped=True, weighed=weighed)
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
                    # What was weighed, where the sheet's Actual cells say
                    # something different from what was printed. The note
                    # already says so: parse_batch_results put the marker in
                    # front of it.
                    opt.tell(opt.amounts_as_weighed(by_number[number],
                                                    weighed.get(number)),
                             results, formulation_no=number,
                             batch_no=batch_no, note=note)
                    # Stop at the first row that did not reach the disk rather
                    # than telling the user a whole sheet was recorded.
                    if not saved_ok(opt):
                        return
                for number, note in left_out:
                    # The box was ticked on the sheet, so the row is recorded
                    # as not scored in the same words the grid's tick writes.
                    opt.record_skipped(
                        number, batch_no,
                        opt.amounts_as_weighed(by_number[number],
                                               weighed.get(number)),
                        note=(wording.not_scored_with_note(note) if note
                              else wording.NOT_SCORED))
                    if not saved_ok(opt):
                        return
                # The lots the sheet came back with belong to the round, not
                # to any one formulation, so they are kept once at the end.
                opt.store_lots(batch_no, lots)
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

    _seed_batch_size(opt)
    typed = typed_batch_size(opt)
    # A change to the box makes the round to that size — the amounts
    # themselves, not a picture of them — and remembers it with the round, so
    # a reopened window and tab 3 both still know what the bench weighed out.
    refusal = _apply_batch_size(opt, typed)
    # A refused size has already named the size the round IS, right above
    # the table; the caption under the download must not say it again.
    said_size = bool(refusal)
    # After the scaling, never before: scale_round replaces the rows, and a
    # list read a line earlier would be the amounts nobody is making.
    rows = opt.pending_batch
    # The round's own size wins over the project's default, and the one
    # accessor is what keeps the table, the workbook and tab 3 naming the
    # same number.
    scale_to = opt.open_round_size()
    # Whether scale_round has been over these rows. Only this screen knows
    # it, so it is said here once and handed to everything that draws a
    # caution off it rather than being guessed at inside the model.
    sized = getattr(opt, 'pending_batch_total', None) is not None

    _title(opt)
    st.markdown(wording.STEP_MAKE_HEADING)
    # Above the table: the size is what the table is a table of.
    _batch_size_control(opt, opt.one_amount_unit(), typed, scale_to,
                        sized, refusal)
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
        _downloads(opt, scale_to, sized, said_size)
    with record_slot:
        _record_results(opt)
    with own_slot:
        _own_formulation(opt)
    with upload_slot:
        _upload(opt)

    if regenerate:
        _regenerate(opt, rows, numbers)
