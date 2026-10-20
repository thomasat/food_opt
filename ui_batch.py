"""Tab 2 · Make a round: generate formulations, make them, record results.

The result grid deliberately avoids st.form so `Save results` can light up
when at least one formulation has a measurement. What the screen shows, the workbook
the bench carries away shows too — and since 0.5.0 so does what is recorded:
the Batch size box moves the stored amounts (`FoodOptimizer.scale_round`)
rather than rewriting a picture of them on the way out.

Changed batch sizes and committed measurement drafts are persisted; unchanged
reruns do not write.
"""
import hashlib
import io
import json
from dataclasses import dataclass

import openpyxl
import pandas as pd
import streamlit as st

import wording
import custom_records
import sample_projects
from food_bo import WORKBOOK_MIME, UploadedWorkbook, uploaded_parts
from storage import StorageError
from ui_helpers import (
    BATCH_SIZE_KEY, SCROLL_KEY, TAB_RESULTS, TAB_SETUP, amount_range_placeholder,
    best_formulation_no, bounds_caution, confirm_action, confirmation_open, flash,
    flash_waiting, fmt_setting, go_to_tab, goal_line, clear_scale_total, entry_label,
    join_unit, label_with_unit, number_list, open_rows, park_clear, preserve_tab_forms,
    readiness, saved_ok, scale_error, table_height, typed_batch_size,
    unit_after_number,
)
from food_bo import COLD_START_RUNS
import bo_views
import insights
import ui_insights
from ui_helpers import REVIEW_DUE_KEY, ask_tab, review_due
# Ftue fix C: the Add fold's place and the Repeat link's request; v3: the
from ui_helpers import ADD_OWN_KEY, ADD_OWN_PLACE


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
    # The one button here that does something, so it is the lit one (spec
    # 2026-09-30 §3, audit V3).
    lit = not confirmation_open()
    if st.button(wording.BACK_TO_SETUP, key="back_to_setup",
                 type="primary" if lit else "secondary", disabled=not lit):
        go_to_tab(TAB_SETUP)


def ask_alone(opt, **kwargs):
    """opt.ask(**kwargs) holding insights.RNG_LOCK: no other tab's Insights
    step can touch torch's global RNG while a Round is being made (review
    B2; Plan A review B I-3). The lock is taken here, at the call site,
    never inside ask().

    The Round's generation is recorded here too (spec 2026-09-30-bo-views
    §3.4, review B4): the seed, the rows and the regime ask() is about to
    use, so "Why these suggestions" can rebuild its acquisition exactly,
    and — once ask() has returned — the formulation numbers it issued, in
    order, so a pick is known by its number and never by its Note (T3
    I-2). Display metadata only, written after ask() has made the Round; a
    failure to record it never stops Generate."""
    with insights.RNG_LOCK:
        try:
            record = bo_views.generation_record(opt, kwargs.get("n_suggestions", 1))
        except Exception:
            record = None
        out = opt.ask(**kwargs)
    if record is not None and opt.pending_batch_no is not None:
        try:
            record["picks"] = [int(row['formulation']) for row in opt.pending_batch or []]
            opt.record_generation(opt.pending_batch_no, record)
        except Exception:
            pass
    return out


def _generate(opt, n, batch_no=None, discarded=None):
    with st.spinner(wording.GENERATING_SPINNER):
        try:
            # A regenerated batch keeps its number and says which numbers
            # retired, and ask() is told both before it opens the batch: a
            # batch renumbered afterwards had already spent the number it was
            # opened with, so the batch AFTER a regenerate skipped one.
            ask_alone(opt, n_suggestions=int(n), batch_no=batch_no,
                      discarded=discarded)
        except ValueError as e:
            st.error(str(e))
            return
        except Exception:
            # A raw traceback is a dead end for a nontechnical user.
            st.error(wording.for_project(opt, wording.GENERATE_FAILED))
            return
    st.session_state.pop("_results_upload", None)
    clear_scale_total()
    if not saved_ok(opt):
        return
    flash("success", wording.batch_ready(opt.pending_batch_no))
    st.rerun()   # a rerun, never a tab move: Generate keeps you here


# What the Generate count opens on before anything has been generated.
_FIRST_COUNT = 3
_COUNT_SEED = "_how_many_seed"


def last_generate_count(opt):
    """How many formulations the project's latest Generate asked for (its
    Round's generation record), 1–10; 3 before the first (02 M1)."""
    records = opt.generation_records()
    if not records:
        return _FIRST_COUNT
    try:
        n = int(records[max(records)].get("n", _FIRST_COUNT))
    except (TypeError, ValueError):
        return _FIRST_COUNT
    return min(10, max(1, n))


def _review_status(opt, due=None, lit=False):
    """Review guide: one line above the suggestions (or above Generate) in
    the Review cards' own words — 'Review: Predictions Reliable ·
    Progress Flat' — and the link "Open Review". Red when a card is (Off,
    Check panel first). Drawn while the workers fit as "Checking
    predictions…", so the line is there from the first draw and never moves
    the lit button under the pointer. True when drawn: its link is then
    the screen's one link to Review (R-I4), and E2's and E3's sentences
    keep their place without theirs.

    `due`: a Round part-saved here and not yet reviewed (Save results kept
    the Round open, so the reader lands back on this screen): the link is
    then the button "Review Round N", lit (`lit`) until a result is typed,
    when Save results is the lit one again.

    v3 (owner: no verdicts, no forecasts on Make a round): only the
    "Review Round N" button is left, while a Round is due its Review; the
    cards' words line and its "Open Review" link are gone."""
    if due is None:
        return False
    with st.container(horizontal=True, gap="small", vertical_alignment="center",
                      key="review_status"):
        st.button(wording.review_round_label(due), key="review_round_batch",
                  type="primary" if lit else "secondary",
                  disabled=confirmation_open(), help=wording.REVIEW_ROUND_HELP,
                  on_click=ask_tab, args=(wording.TAB_INSIGHTS,))
    return True


def _review_lit(opt, due):
    """Review guide: "Review Round N" is this screen's lit button while the
    Round is due a Review and nothing is typed, uploaded or armed: the
    sheets' rule (_downloads), which it then takes over from."""
    return (due is not None and not _any_value_typed(opt) and not confirmation_open()
            and not _upload_pending(opt))


def _no_batch(opt, tab_open=False):
    # The opening value comes from session state (see app.py's _FORM_FRESH):
    # Streamlit warns on screen when a widget carries both a `value=` and a
    # session-state entry, and a project switch assigns these keys.
    # "how_many", not "batch_size": Batch size is the weight of one
    # formulation, and this box asks how many of them to make.
    #
    # 02 M1: it opens on the count the project last generated. The box is
    # not drawn while a Round is open, so Streamlit forgets it, and it came
    # back at 3 after every Round. Seeded once per project and latest
    # Round; a count typed here is the reader's until then.
    records = opt.generation_records()
    seed = (opt.project_name, max(records) if records else None)
    if st.session_state.get(_COUNT_SEED) != seed or "how_many" not in st.session_state:
        st.session_state["how_many"] = last_generate_count(opt)
        st.session_state[_COUNT_SEED] = seed
    how_many = st.number_input(wording.for_project(opt, wording.FORMULATIONS_TO_GENERATE), min_value=1,
                               max_value=10, step=1, key="how_many")
    n = int(how_many)
    # While a confirmation is armed its "Yes" is the one coloured button, and
    # answering it is the one thing to do; generating can wait a click.
    lit = not confirmation_open()
    # v3: no stop rule, so Generate is always the lit button here.
    if st.button(wording.for_project(opt, wording.generate_button_label(n)),
                 type="primary" if lit else "secondary",
                 disabled=not lit, key="generate") and lit:
        _generate(opt, n)
    # v3: no Next step line here (it read the forecasts); Review's What
    # next? says what to do.
    # Only while it is news: before five formulations have results (spec
    # 2026-09-30 §3, audit V16). Set up › Advanced › How it works keeps it.
    if opt.complete_count() < COLD_START_RUNS:
        st.caption(wording.for_project(opt, wording.first_formulations_caption(COLD_START_RUNS)),
                   help=wording.for_project(opt, wording.HOW_CHOSEN_2))
    line = _waiting_caption(opt)
    if line:
        st.caption(line)


def _waiting_caption(opt):
    """One line while any formulation waits (spec 2026-09-30 §3, audit
    V17): '2 formulations are waiting for Cook loss (enter it on Results);
    suggestions may land close to them.', plus why the suggestions still
    cover the allowed amounts when that is the reason. The measurement is
    named only when every waiting formulation waits for the same ones;
    otherwise 'waiting for results'. In the project's own words: trials in
    a project of process settings alone (final review M-1)."""
    waiting = opt.awaited_rows()
    if not waiting:
        return ""
    sets = {tuple(opt.waiting_on(i)) for i in waiting}
    one = next(iter(sets)) if len(sets) == 1 else ()
    line = wording.waiting_line(len(waiting), number_list(one), len(one) > 1)
    if opt.complete_count() < COLD_START_RUNS <= len(opt.Y_history):
        line += " " + wording.still_covering(COLD_START_RUNS)
    return wording.for_project(opt, line)


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


# What `Start from Best so far` typed into the boxes, and which
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
    drawn on the next one. The amounts are at the Formulation total
    (insights.start_recipe), the ones the model learned it at: a Best so
    far made to a 500 g batch size opens at 100 g, not at 500 (spec
    2026-09-30-bo-views §1, review M1)."""
    if _fill_from(opt, best_no, park_clear):
        st.rerun()


def _fill_from(opt, best_no, put):
    """Start from Best so far's filling, shared by its button (`put`:
    park_clear, then a rerun) and the Repeat link's request (`put`: _put,
    on the run that draws the boxes). Returns whether anything was
    filled."""
    index = opt.index_of_formulation(best_no)
    if index is None:
        return False
    # At the total, but only the rows that formulation was recorded with
    # (T1 review I2): start_recipe decodes every current row.
    recorded_names = opt.recipe_history[index]
    recipe = {k: v for k, v in insights.start_recipe(opt, index).items()
              if k in recorded_names}
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
        put(_own_key(var['name']), value)
    put("own_note", wording.repeat_of_formulation(best_no))
    st.session_state[_PREFILLED_FROM] = int(best_no)
    st.session_state[_PREFILLED_AMOUNTS] = prefilled
    return True


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
    checkbox, which could only ever repeat the best one — `Start from Best so
    far` does that in one click and leaves the amounts editable.

    Both buttons are secondary. One button per tab is the coloured one:
    Generate before a batch is open, Save results once it is.
    """
    # Keyed, in a keyed place (ui_helpers.ADD_OWN_PLACE). Its boxes are
    # drawn whether it is open or not, as before.
    with st.container(key=ADD_OWN_PLACE), \
            st.expander(wording.for_project(opt, wording.ADD_OWN_EXPANDER), key=ADD_OWN_KEY,
                        on_change="rerun"):
        if not opt.pending_batch:
            # Adding one with nothing open opens the batch, so the order
            # matters: generated formulations can only join it first.
            st.caption(wording.ADD_OWN_NO_BATCH_CAPTION, help=wording.ADD_OWN_NO_BATCH_HELP)
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
                if st.button(wording.START_FROM_BEST_SO_FAR, key="start_from_best",
                             disabled=blocked, width="stretch"):
                    _start_from_best(opt, best_no)
            with c2:
                add = st.button(wording.ADD_TO_THIS_BATCH,
                                key="add_own_formulation", disabled=blocked,
                                width="stretch")
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
    st.caption(wording.WORKED_OUT_BOXES_CAPTION, help=wording.WORKED_OUT_BOXES_HELP)
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
    """True while the boxes still hold exactly what Start from Best so far
    typed into them. A number the reader has since changed is theirs,
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


def _title(opt):
    """The batch's own heading, and the notice a regenerate owes the numbers
    it retired. Both are about the batch as a whole, so they come before the
    first step rather than inside it."""
    # In the project's own words (04 M-3): "Round 4 · Make 3 trials" in a
    # project of process settings alone.
    st.markdown(wording.for_project(opt, wording.make_these(
        opt.pending_batch_no, len(opt.pending_batch))))
    if opt.pending_batch_discarded:
        st.caption(wording.for_project(opt, wording.batch_discarded_caption(
            number_list(opt.pending_batch_discarded),
            len(opt.pending_batch_discarded) > 1)))


# ------------------------------------------------------------------ #
#  v3 · the Round table (final-v3/2-make-round.png; wording.md §2): the
#  rows are the ingredients and settings in Set up's order, the columns
#  Best so far and then each formulation of the open Round. Each cell is
#  the amount alone, its number orange when it is more than Best so far's
#  by DIRECTION_SHARE of the row's allowed range or more, purple when it
#  is less (owner, v3 final cleanup; it replaced the bracketed change and
#  the grey shading). No forecasts on this screen (owner, v3): the model
#  only picks.
# ------------------------------------------------------------------ #

# A change of at least this share of a row's allowed range (Lowest–Highest)
# colours the number; _SHARE_TOL keeps 5 % up to float error at 5 %.
DIRECTION_SHARE = 0.05
_SHARE_TOL = 1e-9
# Two Overall scores this close are the same score (ties are named).
SCORE_TIE = 1e-6
# Theme-keyed table colours: Best so far's column tint (blue = best so far
# only). The direction colours are ui_insights.direction_colour's.
_COMPARE_COLOURS = {
    "light": {"best": "#e8f0fe"},
    "dark": {"best": "#1c2a4a"},
}


def direction_of(change, span):
    """'more' | 'less' when `change` is at least DIRECTION_SHARE of `span`
    (with _SHARE_TOL), else ''."""
    if span <= 0 or abs(change) / span < DIRECTION_SHARE - _SHARE_TOL:
        return ""
    return "more" if change > 0 else "less"


@dataclass
class CompareCell:
    text: str            # what the cell says (lines joined by "\n")
    share: float = 0.0   # |change from Best so far| / allowed range
    tag: str = ""        # wording.AT_UPPER_LIMIT / AT_LOWER_LIMIT / ""
    direction: str = ""  # 'more' | 'less' | '' (direction_of)


@dataclass
class RoundCompare:
    rows: list           # the row labels, the Total last when there is one
    columns: list        # the column headers, Best so far first when there is one
    cells: list          # [column][row] CompareCell
    best_no: int | None  # the best shown (first made on a tie), None in the cold start
    tied: list           # the other formulations tied with it
    row_header: str
    unit: str | None     # the one unit every row shares, else None
    numbers: list        # [(suggestion k, formulation number)] relabelled columns
    formulations: list   # the formulation number of each column after Best so far
    total_row: bool


def best_shown(opt):
    """(index, [tied formulation numbers]) of Best so far as this screen
    names it: on a tie (scores within SCORE_TIE) the first made, the others
    named. (None, []) with no best."""
    best = opt.best_index()
    if best is None:
        return None, []
    top = float(opt.Y_history[best])
    tied = sorted((int(opt.formulation_ids[i]), i) for i in opt.best_candidates()
                  if abs(float(opt.Y_history[i]) - top) <= SCORE_TIE)
    first = tied[0][1] if tied else best
    return first, [no for no, i in tied if i != first]


def _compare_index(opt):
    """Best so far's index while the Round is compared with it: past the
    cold start, with its amounts on file. Else None."""
    if opt._best_to_compare() is None:
        return None
    first, _ = best_shown(opt)
    if first is None or first >= len(opt.recipe_history):
        return None
    return first


def _total_limit(opt):
    """(lowest, highest) of the limit over every ingredient (the total), or
    None. Either end may be None."""
    if opt.has_formulation_total():
        # The total typed in Set up, as its own number (limit_text writes it
        # so), not the band it is enforced as.
        return float(opt.formulation_total), float(opt.formulation_total)
    names = {v['name'] for v in opt._ingredients()}
    for qc in getattr(opt, 'quantity_constraints', None) or []:
        if names and set(qc.get('ingredients') or []) == names:
            lo, hi = qc.get('min'), qc.get('max')
            if qc.get('exactly') is not None and lo is None and hi is None:
                lo = hi = qc['exactly']
            if lo is not None or hi is not None:
                return lo, hi
    return None


def _limit_tag(var, value):
    """AT_UPPER_LIMIT / AT_LOWER_LIMIT when `value` (at the Formulation
    total) sits on the row's Highest / Lowest; '' otherwise and for a fixed
    row."""
    try:
        lo, hi = (float(b) for b in var['bounds'])
        v = float(value)
    except (KeyError, TypeError, ValueError):
        return ""
    span = hi - lo
    if span <= 0:
        return ""
    near = lambda b: abs(v - b) <= max(1e-6 * (1.0 + abs(b)), 1e-4 * span)
    if near(hi):
        return wording.AT_UPPER_LIMIT
    if near(lo):
        return wording.AT_LOWER_LIMIT
    return ""


def _number(opt, var, value):
    """An amount to two decimals; a setting as its dial reads (fmt_setting)."""
    if var.get('category') == 'process':
        return fmt_setting(value)
    return f"{float(value):.2f}"


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def suggestion_numbers(opt):
    """{formulation number: k} for the open Round's app suggestions while no
    row of it is recorded (or set aside): Make a round's table and its record
    grid call them 'Suggestion 1', 2, 3 in that order. A row of the user's
    own keeps its number; the workbook, the upload and the records keep
    every formulation's number. Empty once anything is recorded."""
    rows = list(opt.pending_batch or [])
    if not rows or len(open_rows(opt)) != len(rows):
        return {}
    out = {}
    for row in rows:
        if not row.get('note'):
            out[int(row['formulation'])] = len(out) + 1
    return out


def _column_label(opt, row, k):
    """'Suggestion k' for an app suggestion not yet made (k given), else
    'F16'; with ' · Less-tried area' when the distance rule says it
    explores."""
    label = wording.suggestion_column(k) if k else wording.short_no(row['formulation'])
    try:
        role = opt.suggestion_role(row['recipe'])
    except Exception:
        role = None
    if role == wording.SUGGESTION_DIFFERENT:
        label = f"{label} · {wording.LESS_TRIED}"
    return label


def round_compare(opt, scale_to=None):
    """The open Round's table, worked out (no Streamlit): see RoundCompare.
    Amounts as the bench weighs them (shown_recipe, at `scale_to`); the
    change from Best so far, its direction and the limit tags read at the
    Formulation total, so they never move with the Batch size box."""
    variables = list(opt.variables)
    rows = list(opt.pending_batch or [])
    # One unit down the whole table: in the header, once. Otherwise each
    # row carries its own ('Bake time (min)').
    unit = opt.one_amount_unit() if not opt._process_settings() else None
    labels = ([wording.worked_out_label(v['name']) if opt.has_formula(v) else v['name']
               for v in variables] if unit else
              [opt._amount_column(v['name'], mark=True) for v in variables])
    best = _compare_index(opt)
    _, tied = best_shown(opt) if best is not None else (None, [])
    # Suggestions not yet made are numbered 1, 2, 3 on screen (no row of the
    # Round recorded yet); a row of the user's own keeps its number.
    shown_as = suggestion_numbers(opt)
    columns, cells, numbers = [], [], []
    with_total = opt.has_ingredients()
    limit = _total_limit(opt) if with_total else None
    total_label = None
    if with_total:
        one = opt.one_amount_unit()
        if limit is not None and one is not None:
            total_label = wording.total_row(
                None if limit[0] is None else f"{float(limit[0]):.2f}",
                None if limit[1] is None else f"{float(limit[1]):.2f}", one)
        else:
            total_label = wording.total_row(unit=one or "")

    def total_cell(recipe):
        value = opt._total_cell(recipe)
        return CompareCell(f"{value:.2f}" if isinstance(value, (int, float)) else str(value))

    ref = ref_total = None
    if best is not None:
        made = opt._recorded_recipe(best)
        ref = opt.scaled_recipe(made, scale_to) if scale_to is not None else made
        ref_total = insights.start_recipe(opt, best)
        score = float(opt.Y_history[best])
        columns.append(wording.best_column(int(opt.formulation_ids[best]), f"{score:.1f}"))
        col = []
        for v in variables:
            x = _as_float(ref.get(v['name']))
            col.append(CompareCell("" if x is None else _number(opt, v, x)))
        if with_total:
            col.append(total_cell(ref))
        cells.append(col)
    for row in rows:
        k = shown_as.get(int(row['formulation']))
        if k:
            numbers.append((k, int(row['formulation'])))
        columns.append(_column_label(opt, row, k))
        shown, _ = opt.shown_recipe(row, scale_to)
        at_total = insights.at_total(opt, row['recipe'])
        col = []
        for v in variables:
            name = v['name']
            x = _as_float(shown.get(name))
            if x is None:
                col.append(CompareCell(str(shown.get(name, ""))))
                continue
            text = _number(opt, v, x)
            share, way = 0.0, ""
            if ref is not None:
                a, b = _as_float(at_total.get(name)), _as_float(ref_total.get(name))
                try:
                    lo, hi = (float(e) for e in v['bounds'])
                except (KeyError, TypeError, ValueError):
                    lo = hi = 0.0
                if a is not None and b is not None and hi > lo:
                    share = abs(a - b) / (hi - lo)
                    way = direction_of(a - b, hi - lo)
            tag = _limit_tag(v, at_total.get(name))
            if tag:
                text += "\n" + tag
            col.append(CompareCell(text, share, tag, way))
        if with_total:
            col.append(total_cell(shown))
        cells.append(col)
    if with_total:
        labels = labels + [total_label]
    return RoundCompare(
        rows=labels, columns=columns, cells=cells,
        best_no=None if best is None else int(opt.formulation_ids[best]), tied=tied,
        row_header=wording.compare_row_header(unit), unit=unit, numbers=numbers,
        formulations=[int(r['formulation']) for r in rows], total_row=with_total)


def compare_styles(table, which="light"):
    """[column][row] CSS for the table in theme `which`: Best so far's
    column blue; a number with a direction in its colour (more orange, less
    purple: ui_insights.direction_colour); everything else plain."""
    tokens = _COMPARE_COLOURS[which]
    out = []
    best = table.best_no is not None
    for c, col in enumerate(table.cells):
        css = []
        for r, cell in enumerate(col):
            parts = ["text-align: center !important"]
            total = table.total_row and r == len(col) - 1
            if best and c == 0:
                parts.append(f"background-color: {tokens['best']}")
            elif cell.direction and not total:
                kind = "above" if cell.direction == "more" else "below"
                parts.append(f"color: {ui_insights.direction_colour(kind, which)}")
            css.append("; ".join(parts))
        out.append(css)
    return out


def limit_tag_style(which="light"):
    """The table style for a cell's limit line (drawn as *at upper limit*):
    always the theme's muted grey, smaller than the number and upright,
    whatever colour the number has (owner, v3 final cleanup)."""
    grey = ui_insights._MUTED_DARK if which == "dark" else ui_insights._MUTED_LIGHT
    return {"selector": "td em",
            "props": f"color: {grey} !important; font-size: 0.8em; font-style: normal"}


def compare_frames(opt, table, which="light"):
    """(cells, css): two frames of one shape, the row labels as the index,
    in the project's own words (Trial, T10 in a study of trials). st.table
    draws each cell as Markdown: a number with a direction is bold, a limit
    tag goes on a line of its own (a hard break) in emphasis, which
    limit_tag_style draws muted and small, and the Total row is bold."""
    say = lambda text: ui_insights._short(opt, text)
    last = len(table.rows) - 1 if table.total_row else None
    bold = lambda text, r: f"**{text}**" if r == last and text else text

    def shown(cell, r):
        number, _, tag = cell.text.partition("\n")
        if cell.direction and number and r != last:
            number = f"**{number}**"
        text = bold(number, r)
        return f"{text}  \n*{tag}*" if tag else text
    index = pd.Index([bold(say(r), k) for k, r in enumerate(table.rows)],
                     name=say(table.row_header))
    columns = [say(c) for c in table.columns]
    cells = pd.DataFrame({c: [shown(cell, r) for r, cell in enumerate(col)]
                          for c, col in zip(columns, table.cells)}, index=index)
    css = pd.DataFrame({c: col for c, col in zip(columns, compare_styles(table, which))},
                       index=index)
    return cells, css


def _improve_caption(opt, table):
    """The caption under the Round title, and the pilot's engine line."""
    say = lambda text: ui_insights._short(opt, text)
    rows = opt.pending_batch or []
    suggested = any(not r.get('note') for r in rows)
    picked = (sample_projects.is_yogurt(opt.project_name)
            and any(sample_projects.ENGINE_PICK in str(r.get('note') or "") for r in rows))
    if table.best_no is None:
        if suggested:
            st.caption(say(wording.COVER_CAPTION), help=say(wording.HOW_CHOSEN_2))
    elif suggested or picked:
        nos = sorted([table.best_no] + list(table.tied))
        st.caption(say(wording.improve_on_caption(nos)),
                   help=say(wording.improve_on_help(nos)))
    if picked:
        st.caption(wording.CHOSEN_IN_PILOT)


def _draw_compare(opt, table):
    """The table, then its key with the rest in the key's "?"."""
    say = lambda text: ui_insights._short(opt, text)
    cells, css = compare_frames(opt, table, ui_insights.theme())
    styled = cells.style.apply(lambda _: css, axis=None).set_table_styles(
        [{"selector": "thead th", "props": "text-align: center !important"},
         limit_tag_style(ui_insights.theme())])
    if table.tied:
        # The tie in the Best so far column's cells' "?" (st.table draws no
        # header help); the key's "?" says it too.
        tip = say(wording.best_tied_help(table.best_no, table.tied))
        tips = pd.DataFrame("", index=cells.index, columns=cells.columns)
        tips[cells.columns[0]] = tip
        styled = styled.set_tooltips(tips)
    st.table(styled)
    if table.best_no is None:
        return
    row = "ingredient" if opt.has_ingredients() else "setting"
    pct = DIRECTION_SHARE * 100
    tips = [wording.compare_key_help(table.best_no, pct, row)]
    if table.tied:
        tips.append(wording.best_tied_help(table.best_no, table.tied))
    if any(wording.LESS_TRIED in c for c in table.columns):
        from food_bo import FAR_FROM_TESTED
        tips.append(wording.less_tried_help(round(FAR_FROM_TESTED * 100)))
    if table.numbers:
        tips.append(wording.SUGGESTION_NUMBERS_HELP)
    # Minor 15 (2026-09-29 review): a suggestion that nearly repeats a
    # formulation still waiting for results, named by its column.
    prefixes = _near_copy_prefixes(opt, opt.pending_batch)
    for label, no in zip(table.columns[len(table.columns) - len(table.formulations):],
                         table.formulations):
        if no in prefixes:
            tips.append(f"{label.split(' · ')[0]}: {prefixes[no].strip()}")
    st.caption(ui_insights.direction_key_html(
        "", say(wording.compare_key_more(table.best_no)),
        say(wording.compare_key_less(table.best_no)), wording.compare_key_plain(pct)),
        help=say(" ".join(tips)), unsafe_allow_html=True)


def _near_copy_prefixes(opt, batch):
    """{Formulation number: prefix text} for every suggestion in `batch`
    that nearly repeats a formulation still waiting for results (Minor 15,
    2026-09-29 review) — never for the user's own formulations, which carry
    a note. Screen only: v3 says it in the Round table key's "?"; the
    workbook never carries it.

    Routed through `wording.for_project`: a process-only study's table calls
    every row a Trial, and the prefix has to use the same word for the row
    it names, or a cell reads 'Nearly the same as Formulation 7' beside a
    table that has never once said Formulation.

    During the cold start there is no Compared with column to prefix at all
    (batch_frame drops it, COMPARED_WITH_ALLOWED), so nothing here would
    ever be read; skipped rather than computed and thrown away."""
    if opt.compared_with_column() == wording.COMPARED_WITH_ALLOWED:
        return {}
    prefixes = {}
    for row in opt._batch_rows(batch):
        if row.get('note'):
            continue
        match = insights.near_copy(opt, row['recipe'])
        if match is not None:
            prefixes[row['formulation']] = wording.for_project(
                opt, wording.near_copy_prefix(
                    match.formulation, number_list(match.waiting_on)))
    return prefixes


def _batch_table(opt, scale_to, tab_open=False, table=None):
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

    v3: drawn as round_compare's table (ingredients down, Best so far and
    the Round across), never with a forecast. Returns (editing, None).
    """
    frame = opt.batch_frame(opt.pending_batch, scale_to=scale_to)
    edit_key = f"edit_round_{opt.project_name}_{opt.pending_batch_no}"
    editing = st.session_state.get(edit_key, False)
    if not editing:
        if st.button(wording.for_project(opt, wording.EDIT_FORMULATIONS), key=edit_key + "_start", icon=":material/edit:"):
            st.session_state[edit_key] = True
            preserve_tab_forms()
            st.rerun()
        _draw_compare(opt, table if table is not None else round_compare(opt, scale_to))
    else:
        st.info(wording.for_project(opt, wording.EDIT_FORMULATIONS_ACTIVE), icon=":material/edit:")
        st.caption(wording.EDIT_FORMULATIONS_CAPTION)
        editable_numbers = {row["formulation"] for row in open_rows(opt)}
        # v3: amounts only; the role words are not on this screen.
        frame = frame.drop(columns=[opt.compared_with_column()], errors="ignore")
        frame = frame[frame["Formulation"].isin(editable_numbers)].reset_index(drop=True)
        independent = {opt._amount_column(v['name'], mark=True): v['name']
                       for v in opt.variables if not opt.has_formula(v)}
        columns = {c: st.column_config.NumberColumn(c, help=wording.EDITABLE_AMOUNT_HELP)
                   for c in frame if c in independent}
        columns.update({c: st.column_config.Column(c, help=wording.READ_ONLY_AMOUNT_HELP)
                        for c in frame if c not in independent})
        columns[wording.FORMULATION_CAP] = st.column_config.Column(
            wording.for_project(opt, wording.FORMULATION_CAP))
        edited = st.data_editor(frame, hide_index=True, num_rows="fixed",
                               column_config=columns,
                               disabled=[c for c in frame if c not in independent],
                               key=edit_key + "_grid", width="stretch", placeholder="")
        save, cancel = st.columns(2)
        if save.button("Save changes", key=edit_key + "_save", type="primary"):
            changes = {}
            for i, row in edited.iterrows():
                values = {name: row[col] for col, name in independent.items()
                          if col in frame and row[col] != frame.loc[i, col]}
                if values:
                    changes[int(row["Formulation"])] = values
            try:
                opt.edit_pending_formulations(changes)
            except (ValueError, OSError, StorageError) as exc:
                st.error(str(exc))
            else:
                if saved_ok(opt):
                    st.session_state[edit_key] = False
                    st.session_state.pop(edit_key + "_grid", None)
                    flash("success", wording.FORMULATIONS_SAVED_NEW_WORKBOOK)
                    preserve_tab_forms()
                    st.rerun()
        if cancel.button("Cancel", key=edit_key + "_cancel"):
            st.session_state[edit_key] = False
            st.session_state.pop(edit_key + "_grid", None)
            preserve_tab_forms()
            st.rerun()
    # The size first, then the corrections line. Three captions under one
    # table, two of them about the size, used to sit with the one that is
    # not in the middle of them.
    if scale_to is None:
        if opt.has_ingredients():
            st.caption(wording.NO_BATCH_SIZE_OF_ITS_OWN)
    else:
        for line in opt.total_mismatch_lines(opt.pending_batch, scale_to):
            st.caption(line)
    # v3: the cold start's line (HOW_CHOSEN_2) is the "?" of the caption
    # under the title, and the table's key says what it is compared with.
    return editing, None


def own_round_caption(rows):
    """Persona fixes 2: a Round of the user's own rows says where they came
    from — "only your own formulations" only when no row carries a note of
    its own (the default note the app types does not count); else the
    entered line that points at the notes."""
    notes = {str(r.get('note') or "").strip() for r in rows}
    if notes - {"", wording.OWN_FORMULATION_NOTE}:
        return wording.ENTERED_ROUND_CAPTION
    return wording.ONLY_OWN_FORMULATIONS_CAPTION


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
        st.caption(wording.for_project(opt, wording.NEEDS_ONE_UNIT))
        return
    st.session_state.setdefault(_BATCH_SIZE_KEY, None)
    st.number_input(
        wording.for_project(opt, wording.batch_size_label(unit)),
        min_value=0.0, step=1.0, placeholder=wording.BATCH_SIZE_PLACEHOLDER,
        key=_BATCH_SIZE_KEY,
        disabled=bool(st.session_state.get(f"edit_round_{opt.project_name}_{opt.pending_batch_no}")),
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


def _has_preparation_sheet(book):
    """The workbook offered has a Preparation sheet: read from the file
    itself, so the line under the download lists the sheets it really has."""
    try:
        names = openpyxl.load_workbook(io.BytesIO(book), read_only=True).sheetnames
    except Exception:            # an unreadable file: the two-sheet line stands
        return False
    return wording.PREPARATION_SHEET in names


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
    lit = (not _any_value_typed(opt) and not confirmation_open()
           and not _upload_pending(opt) and review_due(opt) is None)
    print_pack = st.checkbox(wording.for_project(opt, wording.PRINT_PACK_CHECKBOX), key="workbook_print_pack")
    book = opt.workbook_bytes(rows, scale_to, sized, print_pack=print_pack)
    st.download_button(
        wording.DOWNLOAD_BATCH_SHEETS,
        data=book,
        file_name=wording.workbook_file_name(opt.project_name,
                                             opt.pending_batch_no),
        mime=WORKBOOK_MIME, key="download_batch_sheets",
        type="primary" if lit else "secondary",
        width="stretch",
        on_click=_sheets_are_out,
        help=wording.for_project(opt, wording.workbook_tabs_help(print_pack)),
    )
    # v3: one line under the button naming the sheets (it follows the box:
    # the file a tick produces has no Results sheet); the pre-mix sheet and
    # the tick in its "?".
    st.caption(wording.for_project(opt, wording.workbook_sheets(
                   print_pack, preparation=_has_preparation_sheet(book))),
               help=wording.for_project(opt, wording.workbook_sheets_help(print_pack)))
    if scale_to is not None and not said_size:
        # The file carries the amounts on screen, so the size it was written
        # for is named directly under it — unless the refusal above the
        # table has just said that same number ("The round stays at 100 g"),
        # in which case this repeats one fact with a different subject one
        # line apart.
        st.caption(wording.sheets_show_total_caption(
            opt.batch_total_text(scale_to)))


def _sheets_are_out():
    """The workbook has left the app in this session, so the upload door
    opens. A download button has no return value to read: the click is
    recorded here, which is the one place Streamlit gives it."""
    st.session_state[_SHEETS_DOWNLOADED] = True


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
    skipped = next((r for r in opt.skipped if r['formulation'] == number), None)
    if skipped is not None:
        st.caption(skipped.get('note') or wording.NOT_SCORED)
        return
    results = opt.results_history[index] if index is not None else {}
    line = " · ".join(
        join_unit(f"{label_with_unit(o['name'], o.get('unit'))} "
                  f"{float(results[o['name']]):g}",
                  unit_after_number(o.get('unit')))
        for o in ordered if o['name'] in results
    )
    # Only what it still waits for: a measurement added after this row was
    # recorded was never expected of it.
    unmeasured = opt.waiting_on(index) if index is not None else []
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
    # The project's own name, not id(opt): a memory address is reused the
    # moment the object it belonged to is collected, and app.py replaces
    # st.session_state.optimizer outright on a CLASS_VERSION refresh.
    seed = (opt.project_name, opt.pending_batch_no,
            tuple(o['name'] for o in ordered))
    if st.session_state.get('_result_draft_seed') != seed:
        for row in to_record:
            number = row['formulation']
            draft = opt.result_drafts.get(number, {})
            for obj in ordered:
                st.session_state[_result_key(number, obj['name'])] = (
                    draft.get('results', {}).get(obj['name']))
            st.session_state[f'f{number}_note'] = draft.get('note', row.get('note') or '')
            st.session_state[f'f{number}_leave_out'] = draft.get('not_scored', False)
        st.session_state['_result_draft_seed'] = seed
    else:
        # 05 N1: a save on Set up reruns from ABOVE this tab, and Streamlit
        # drops the state of every box that run did not reach. The seed is
        # unchanged (a Highest widened, a unit corrected), so without this
        # the boxes came back empty and the empty boxes were then saved over
        # the drafts on file. Only a box with no state at all is filled, and
        # only from what was saved: one the reader is typing in is theirs.
        for row in to_record:
            number = row['formulation']
            draft = opt.result_drafts.get(number, {})
            for obj in ordered:
                key = _result_key(number, obj['name'])
                if key not in st.session_state:
                    st.session_state[key] = draft.get('results', {}).get(obj['name'])
            if f'f{number}_note' not in st.session_state:
                st.session_state[f'f{number}_note'] = draft.get('note', row.get('note') or '')
            if f'f{number}_leave_out' not in st.session_state:
                st.session_state[f'f{number}_leave_out'] = draft.get('not_scored', False)
    st.caption(wording.RECORD_RESULTS_LINE, help=wording.RECORD_RESULTS_HELP)
    left_out, entered, partly = set(), 0, 0
    # As the table above names them: 'Suggestion 1' until a row is recorded.
    shown_as = suggestion_numbers(opt)

    for row in rows:
        number = row['formulation']
        k = shown_as.get(int(number))
        st.markdown(wording.for_project(opt, wording.suggestion_heading(k) if k
                                        else wording.formulation_heading(number)))
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
                    f"{entry_label(obj['name'], obj.get('unit'))} · "
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
        # available for entered formulations; an out-of-range reading is
        # explained immediately and refused again on save.
        for problem in outside:
            st.caption(problem)
        if row.get('note'):
            # A repeat of the best formulation arrives already saying so.
            st.session_state.setdefault(f"f{number}_note", row['note'])
        # Note and Not scored behind one "More" per formulation (spec
        # 2026-09-30 §3, audit V11): the boxes the user came to fill come
        # first. The label says what is set inside, and the fold opens when
        # Not scored is ticked or a note was typed here (review M11).
        note = str(st.session_state.get(f"f{number}_note") or "").strip()
        with st.expander(wording.more_label(skip, bool(note)),
                         expanded=skip or note not in ("", str(row.get('note') or "").strip())):
            # Never disabled: why a formulation was not scored is the only
            # record of what went wrong, and a Note that greys out on the
            # tick can only be typed by someone who knew to type it first.
            st.text_input(wording.NOTE, key=f"f{number}_note")
            st.checkbox(wording.NOT_SCORED, key=f"f{number}_leave_out",
                        help=wording.NOT_SCORED_TICK_HELP)
        if not skip:
            # Complete means EVERY measurement has a number. One of three
            # typed is not a third of a result, and counting it as complete
            # told the user the batch was further along than it was.
            if ordered and typed == len(ordered):
                entered += 1
            elif typed:
                partly += 1
        st.divider()

    drafts = {}
    for row in to_record:
        number = row['formulation']
        values = {o['name']: st.session_state.get(_result_key(number, o['name']))
                  for o in ordered}
        values = {name: value for name, value in values.items() if value is not None}
        note = st.session_state.get(f'f{number}_note') or ''
        if values or note != (row.get('note') or '') or number in left_out:
            drafts[number] = dict(results=values, note=note, not_scored=number in left_out)
    opt.save_result_drafts(drafts)
    if drafts and saved_ok(opt):
        st.caption(wording.RESULT_DRAFT_SAVED)
    kept = [r for r in to_record if r['formulation'] not in left_out
            and drafts.get(r['formulation'], {}).get('results')]
    ready = bool(kept)
    if to_record and len(left_out) == len(to_record):
        st.info(wording.for_project(opt, wording.NOTHING_TO_SAVE))
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
                 disabled=not lit, key="save_results", help=wording.SAVE_RESULTS_HELP) and lit:
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
    st.session_state.pop("_results_upload", None)
    if not saved_ok(opt):
        # A move would rerun past app.py's end-of-script check and hide it.
        return
    saved = [row['formulation'] for row in kept]
    _after_save(opt, batch_no)
    if open_rows(opt):
        flash('success', wording.upload_partial_flash(
            len(kept) + len(left_out), len(opt.pending_batch),
            batch_no, len(open_rows(opt))), sticky=True)
        flash_waiting(opt, saved, sticky=True)
        # The reader is at Save results, far below the line (01 O3).
        st.session_state[SCROLL_KEY] = "top"
        st.rerun()
    opt.set_pending_batch(None)
    clear_scale_total()
    if not saved_ok(opt):
        return
    flash("success", wording.batch_recorded_flash(batch_no), sticky=True)
    flash_waiting(opt, saved, sticky=True)
    go_to_tab(TAB_RESULTS)


def _after_save(opt, batch_no=None):
    """After Save results or Save uploaded results (review I10): Results'
    View formulation box back on Best so far (ui_results.reset_view); and
    (review guide) the Round now due a Review, whole or part saved."""
    import ui_results     # ui_results is a sibling tab, drawn after this one
    ui_results.reset_view(opt)
    if batch_no is not None:
        st.session_state[REVIEW_DUE_KEY] = batch_no


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


def _upload_preview(opt, parsed, left_out, weighed=None, lots=None):
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
        height=table_height(len(frame)), placeholder="")
    by_number = {r['formulation']: r['recipe'] for r in opt.pending_batch}
    imported = {no for no, _, _ in parsed} | {no for no, _ in left_out}
    if weighed:
        with st.expander(wording.UPLOAD_AMOUNTS_HEADING, expanded=True):
            comparisons = []
            for number in sorted(imported):
                actual = weighed.get(number, {})
                if not actual:
                    continue
                planned = by_number[number]
                for name, amount in actual.items():
                    unit = opt.unit_of(name)
                    comparisons.append({wording.FORMULATION_CAP: number,
                        wording.NAME_LABEL: name,
                        wording.AMOUNT_COLUMN: planned.get(name, 0),
                        wording.ACTUAL_COLUMN: amount, wording.UNIT_LABEL: unit})
                recipe = opt.amounts_as_weighed(planned, actual)
                st.caption(wording.upload_amounts_total(
                    number, opt.total_text(planned), opt.total_text(recipe)))
                mismatch = opt.total_mismatch(number, recipe, opt.open_round_size())
                if mismatch:
                    st.caption(mismatch)
                for name, amount in actual.items():
                    caution = opt.bounds_caution(
                        name, amount, opt.amount_scale(opt.open_round_size()))
                    if caution:
                        st.caption(caution)
                if opt.one_amount_unit() and opt.ingredient_total(recipe) > 0:
                    caution = opt.scaled_limit_caution(
                        [recipe], opt.ingredient_total(recipe), sized=True)
                    if caution:
                        st.caption(caution)
            if comparisons:
                frame = pd.DataFrame(comparisons)
                unit = opt.one_amount_unit()
                if unit and all(r[wording.UNIT_LABEL] == unit for r in comparisons):
                    frame = frame.drop(columns=[wording.UNIT_LABEL]).rename(columns={
                        c: f'{c} ({unit})' for c in
                        (wording.AMOUNT_COLUMN, wording.ACTUAL_COLUMN)})
                st.dataframe(frame, hide_index=True, key='upload_amounts_preview',
                    column_config={c: st.column_config.NumberColumn(c, format='%.2f')
                        for c in frame.columns if c not in
                        (wording.FORMULATION_CAP, wording.NAME_LABEL, wording.UNIT_LABEL)}, placeholder="")
    if lots:
        with st.expander(wording.UPLOAD_LOTS_HEADING):
            st.dataframe(pd.DataFrame([{wording.NAME_LABEL: name, wording.LOT_COLUMN: lot}
                for name, lot in lots.items()]), hide_index=True, key='upload_lots_preview', placeholder="")


def _blank_or_number(value):
    """A measurement nobody took is a blank cell, never 'nan'.

    Two decimals are the app's rule for an AMOUNT. These are measurements: a
    panel score of 6 reads 6, and a sodium reading of 39000 reads 39000.
    """
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):g}"


_SHEETS_DOWNLOADED = "_sheets_downloaded"


def _file_mark(opt, sheet_file):
    """Which file, for which project and Round: read again only when it
    changes."""
    return (opt.project_name, hashlib.sha256(sheet_file.getvalue()).hexdigest(),
            json.dumps(opt.pending_batch, sort_keys=True, default=str))


def _upload_key(opt):
    return f"results_file_{opt.project_name}"


def _upload_pending(opt):
    """Whether an uploaded results file is the next step (02 M5): read and
    waiting for Save uploaded results, or new and read further down this
    run. A file already read and refused is not."""
    if st.session_state.get("_results_upload") is not None:
        return True
    sheet_file = st.session_state.get(_upload_key(opt))
    if sheet_file is None:
        return False
    return _file_mark(opt, sheet_file) != st.session_state.get("_results_file_mark")


def _upload(opt):
    # Open, once the workbook has left the app in this session. The reader
    # who downloaded the sheets is coming back with a file in their hand,
    # and the door they need was collapsed under a heading beginning "Or".
    came_back = bool(st.session_state.get(_SHEETS_DOWNLOADED))
    with st.expander(wording.UPLOAD_EXPANDER, expanded=came_back):
        st.caption(wording.for_project(opt, wording.UPLOAD_SHEETS_ARE_BACK_CAPTION if came_back
                   else wording.UPLOAD_HELP_CAPTION),
                   help=wording.for_project(opt, wording.UPLOAD_SHEETS_ARE_BACK_HELP if came_back
                                            else wording.UPLOAD_HELP_DETAIL))
        sheet_file = st.file_uploader(
            wording.UPLOAD_RESULTS_FILE, type=["xlsx", "csv"],
            # Per project: an uploader cannot be emptied from session state,
            # so a shared key offered the next project this one's sheet.
            key=_upload_key(opt))
        # The Round 6 sample: the pilot's own results, to upload right here.
        if sample_projects.offers_round6_download(opt):
            st.download_button(
                wording.DOWNLOAD_ROUND6_RESULTS,
                data=sample_projects.round6_results_bytes(),
                file_name=sample_projects.ROUND6_RESULTS_FILE_NAME,
                mime=WORKBOOK_MIME, key="sample_round6_results")
        check_file = sheet_file is not None and st.button(wording.CHECK_THIS_FILE, key="check_sheet")
        file_mark = None
        if sheet_file is not None:
            file_mark = _file_mark(opt, sheet_file)
        if sheet_file is not None and (check_file or file_mark != st.session_state.get("_results_file_mark")):
            st.session_state["_results_file_mark"] = file_mark
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
        extra_records = getattr(st.session_state.get('_results_upload'), 'custom_records', None) or []
        try:
            if sheet.empty and extra_records:
                parsed, left_out = [], []
            else:
                parsed, left_out = opt.parse_batch_results(
                    sheet, opt.pending_batch, with_skipped=True, weighed=weighed)
        except ValueError as e:
            st.error(str(e))
            st.session_state.pop("_results_upload", None)
            return
        if parsed or left_out:
            st.info(wording.for_project(opt, wording.upload_found_caption(
                len(parsed), len(opt.pending_batch),
                ", ".join(f"{wording.FORMULATION_CAP} {no}" for no, _, _ in parsed))
                + (wording.not_scored_counter_suffix(len(left_out)) if left_out
                   else "")))
            _upload_preview(opt, parsed, left_out, weighed, lots)
        if extra_records:
            definitions = {f['id']: f for f in custom_records.fields(opt, enabled=False)}
            st.caption(wording.CUSTOM_IMPORT_FOUND)
            st.dataframe(pd.DataFrame([{wording.CUSTOM_SCOPE_LABEL: wording.for_project(opt, wording.CUSTOM_SCOPE_NAMES[scope]),
                wording.CUSTOM_SUBJECT: subject, wording.CUSTOM_FIELD_NAME: definitions[key]['name'],
                wording.CUSTOM_VALUE: value} for scope, subject, key, value in extra_records]), hide_index=True, placeholder="")
        records = getattr(st.session_state.get("_results_upload"), "bench_records", None)
        if records:
            st.caption(wording.for_project(opt, wording.BENCH_RECORDS_CAPTION))
            with st.expander(wording.BENCH_RECORDS_SHEET):
                st.dataframe(pd.DataFrame(records).rename(columns=wording.BENCH_RECORD_COLUMNS),
                             hide_index=True, width="stretch", placeholder="")
        # The next step once the preview shows (02 M5): lit, and the
        # download steps aside (_downloads, _upload_pending). Not while
        # results are typed into the grid: Save results is lit then.
        lit = not confirmation_open() and not _any_value_typed(opt)
        if st.button(wording.SAVE_UPLOADED_RESULTS, key="save_uploaded",
                     type="primary" if lit else "secondary"):
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
                custom_records.save_values(opt, batch_no, extra_records)
                if not saved_ok(opt):
                    return
                opt.store_bench_records(batch_no, records)
                if not saved_ok(opt):
                    return
                opt.store_lots(batch_no, lots)
                if not saved_ok(opt):
                    return
            except (ValueError, TypeError, KeyError) as e:
                st.error(wording.could_not_save(e))
                return
            st.session_state.pop("_results_upload", None)
            if extra_records and not parsed and not left_out:
                flash("success", wording.CUSTOM_SAVED)
                st.rerun()
            _after_save(opt, batch_no)
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
                    batch_no, len(left)), sticky=True)
                flash_waiting(opt, [no for no, _, _ in parsed], sticky=True)
                st.session_state[SCROLL_KEY] = "top"
                st.rerun()
            opt.set_pending_batch(None)
            clear_scale_total()
            if not saved_ok(opt):
                # A move would rerun past app.py's end-of-script check and
                # hide it, and the batch is still open on disk.
                return
            flash("success", wording.batch_recorded_flash(batch_no), sticky=True)
            flash_waiting(opt, [no for no, _, _ in parsed], sticky=True)
            go_to_tab(TAB_RESULTS)


def render(opt, storage, tab_open=False):
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

    And `Add to this batch` and `Start from Best so far` both end in
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
        _no_batch(opt, tab_open)
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
    table = round_compare(opt, scale_to)
    _improve_caption(opt, table)
    # Review guide: filled after the confirmation below is asked, so an
    # armed one greys "Review Round N" on the same run.
    status_slot = st.container()
    st.markdown(wording.for_project(opt, wording.STEP_MAKE_HEADING))
    # Above the table: the size is what the table is a table of.
    _batch_size_control(opt, opt.one_amount_unit(), typed, scale_to,
                        sized, refusal)
    editing, _ = _batch_table(opt, scale_to, tab_open, table)
    if editing:
        preserve_tab_forms()
        st.caption(wording.for_project(opt, wording.FINISH_FORMULATION_EDITS))
        return
    custom_records.editor(opt, opt.pending_batch_no, [r['formulation'] for r in rows])
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
        st.caption(wording.for_project(opt, own_round_caption(rows)))
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

    if tab_open:
        with status_slot:
            due = review_due(opt)
            _review_status(opt, due, _review_lit(opt, due))
    with print_slot:
        _downloads(opt, scale_to, sized, said_size)
    with record_slot:
        # v3: no forecast before tasting (owner: no forecasts on this screen).
        _record_results(opt)
    with own_slot:
        _own_formulation(opt)
    with upload_slot:
        _upload(opt)

    if regenerate:
        _regenerate(opt, rows, numbers)
