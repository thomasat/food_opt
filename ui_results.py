"""Tab 3 · Results: the last Round's results against their targets (v3:
measurements only, no forecast), how one formulation's score is worked
out, the best formulation, every formulation, and one
collapsed section — `Correct, delete or add past formulations` — for every
way the record is fixed after the fact: a correction, a deletion, and a
formulation made before this project existed, typed in or read off a CSV.

The one coloured button is at the foot: `Start the next round`, or `Back to
Round N` while a round is still unrecorded. Everything destructive is behind a
confirmation that keeps a copy first and says so — once, in the confirmation.
"""
import pandas as pd
import streamlit as st

import insights
import sample_projects
import storage as storage_backend
import ui_insights
import wording
import custom_records
from food_bo import ROUND_FIELD, WORKBOOK_MIME
from ui_helpers import (
    COPY_KEPT, TAB_BATCH, TAB_SETUP, amount_range_placeholder, ask_tab,
    best_formulation_no, best_move_sentence,
    bounds_caution, clear_selection, confirm_action, confirmation_open,
    SCROLL_KEY, clear_scale_total, disarm, flash, fmt_amount, fmt_setting, goal_line,
    entry_label, go_to_tab, review_due, number_list, open_rows, other_confirmation,
    park_clear,
    plural, preserve_tab_forms, readiness, render_flash, saved_ok, scale_error,
    table_height, take_clear, waiting_text,
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
    """The `Amounts to make it` rows: ingredients in SET-UP order, then the
    process settings, each written with its own unit — the water in ml beside
    the protein in g, and a cook temperature never in either.

    Set-up order, not largest first: it is the order every sheet lists them
    in and the order the table on tab 1 reads down, and the same recipe in
    two orders on two screens made the reader check it line by line."""
    category = {v['name']: v.get('category', 'ingredient') for v in opt.variables}
    shown = {name for name, _ in opt.recipe_lines(recipe)}
    shown.update(v['name'] for v in opt.variables
                 if v.get('category') == 'process' and v['name'] in recipe)
    pairs = [(v['name'], float(recipe[v['name']])) for v in opt.variables
             if v['name'] in shown and v['name'] in recipe]
    ingredients = [p for p in pairs if category.get(p[0]) != 'process']
    settings = [p for p in pairs if category.get(p[0]) == 'process']

    def amount(name, value):
        if category.get(name) == 'process':
            # 180 °C, never 180.00 °C: a setting is dialled in, not weighed.
            return fmt_setting(value, opt.unit_of(name))
        return fmt_amount(value, opt.unit_of(name))

    # Use the amounts actually displayed, including user-entered formulations
    # and recorded scaling. Process settings and pre-mix component percentages
    # must never enter this denominator.
    total = sum(value for _, value in ingredients)
    comparable = opt._shows_shares() and total > 0
    rows = []
    for name, value in ingredients + settings:
        row = {wording.INGREDIENT_OR_SETTING_LABEL: name,
               wording.AMOUNT_COLUMN: amount(name, value)}
        if opt.has_ingredients():
            row[wording.RESULT_PERCENT_COLUMN] = (
                f"{100 * value / total:.2f}" if comparable and category.get(name) != 'process'
                else "" if category.get(name) == 'process' else "—")
        rows.append(row)
    return rows


def _amount_table(opt, recipe):
    rows = _amount_rows(opt, recipe)
    columns = [wording.INGREDIENT_OR_SETTING_LABEL, wording.AMOUNT_COLUMN]
    if opt.has_ingredients():
        columns.append(wording.RESULT_PERCENT_COLUMN)
    st.table(pd.DataFrame(rows, columns=columns, index=[""] * len(rows)))
    if opt.has_ingredients():
        if not opt._shows_shares():
            st.caption(wording.RESULT_PERCENT_MIXED_HELP)
        elif not any(row.get(wording.RESULT_PERCENT_COLUMN) not in ("", "—", None) for row in rows):
            st.caption(wording.RESULT_PERCENT_ZERO_HELP)
        elif opt._process_settings():
            st.caption(wording.RESULT_PERCENT_PROCESS_HELP)


# The View formulation box's entry for Best so far, whichever formulation
# that is. A word, not None: Streamlit reads a selectbox value of None as
# "nothing chosen", and the box then read "Choose an option".
BEST_CHOICE = "best"

# Where this tab's own confirmations land (0.8.0 e2e fixes B3, review I10):
# a keyed slot directly above the View formulation box, scrolled to, so a
# "Formulation 19 moved to Not scored." no longer appears 501 px above a
# reader looking at the button they pressed (01 O3). Save results keeps the
# top of the page: it moves here and lands at the top.
FLASH_PLACE = "results"
FLASH_SLOT = "results_flash"
# The correction part of "Correct, delete or add past formulations", which
# "Change a measurement or an ingredient" scrolls to while a recorded
# formulation is being corrected (05 I8).
CORRECTION_SLOT = "results_correction"
# The "Correct, delete or add past formulations" fold, keyed so the button
# above can open it (polish round I8).
EDIT_PAST_KEY = "results_edit_past"


def _view_key(opt):
    return f"result_view_{opt.project_name}"


def reset_view(opt):
    """After Save results or Save uploaded results (review I10; 02 I5, M11):
    the View formulation box goes back to Best so far — never a stale
    formulation viewed before, such as one moved to Not scored, which hid
    the Round summary and its link for every later Round. Set, not popped:
    a value set here is sent to the browser again, under its new label.
    Called before the Results tab is drawn on this run."""
    if opt.best_index() is not None:
        st.session_state[_view_key(opt)] = BEST_CHOICE
    else:
        st.session_state.pop(_view_key(opt), None)


def flash_here(kind, message):
    """A confirmation of something done on this tab, in its slot above the
    View formulation box, scrolled into view and kept there until the next
    action (review I10)."""
    flash(kind, message, sticky=True, place=FLASH_PLACE)
    st.session_state[SCROLL_KEY] = FLASH_SLOT


def _confirmations():
    """The slot flash_here writes to: always drawn, so the tab keeps its
    shape whether a message shows or not."""
    with st.container(key=FLASH_SLOT):
        render_flash(place=FLASH_PLACE)


def _selected_formulation(opt):
    """A default that follows the best score, plus read-only history choices."""
    best_index = opt.best_index()
    best_no = int(opt.formulation_ids[best_index]) if best_index is not None else None
    numbers = _all_numbers(opt)
    if len(numbers) < 2:
        return numbers[0] if numbers else None
    options = ([BEST_CHOICE] if best_no is not None else []) + [n for n in numbers if n != best_no]
    key = _view_key(opt)
    # The Best so far entry's label names its formulation, and the browser
    # keeps the label it was shown: after Best so far moved, the box kept
    # naming the old one above a heading naming the new one (0.8.0 review,
    # UX minor 11). A value set here is sent to the browser again, so the
    # box comes back to Best so far under its new label — also when the
    # formulation picked in it is the one that became Best so far.
    seen = f"_result_view_best_{opt.project_name}"
    stale = key in st.session_state and st.session_state[key] not in options
    moved = (st.session_state.get(seen, best_no) != best_no
             and st.session_state.get(key) == BEST_CHOICE)
    if stale or moved:
        if BEST_CHOICE in options:
            st.session_state[key] = BEST_CHOICE
        else:
            st.session_state.pop(key, None)
    st.session_state[seen] = best_no

    labels = {choice: formulation_option(opt, best_no if choice == BEST_CHOICE else choice,
                                         best=choice == BEST_CHOICE)
              for choice in options}
    # The browser keeps the label it was shown for the value it holds: after
    # Move to Not scored the box kept "· waiting for Cook loss" above "Not
    # scored" (answer-key read-through). When the label of the formulation
    # picked changed, its value is sent again, once, so the label is redrawn.
    shown_key = f"_result_view_label_{opt.project_name}"
    current = st.session_state.get(key)
    shown = st.session_state.get(shown_key)
    if (current in labels and shown is not None and shown[0] == current
            and shown[1] != labels[current]):
        st.session_state[key] = current
    if current in labels:
        st.session_state[shown_key] = (current, labels[current])

    chosen = st.selectbox(wording.for_project(opt, wording.RESULT_VIEW_LABEL), options,
                          format_func=labels.__getitem__, key=key,
                          help=wording.for_project(opt, wording.RESULT_VIEW_HELP))
    return best_no if chosen in (None, BEST_CHOICE) else chosen


def formulation_option(opt, number, best=False):
    """How a formulation reads in a picker: number, Round and status
    ("Best so far · Formulation 2 (Round 1)", "… · Not scored", "… ·
    waiting for Cook loss"). The View formulation box and the correction
    picker say it the same way (polish round I8)."""
    index = opt.index_of_formulation(number)
    skipped = next((r for r in opt.skipped if int(r['formulation']) == number), {})
    batch = opt.batch_history[index] if index is not None else skipped.get(ROUND_FIELD)
    label = wording.result_view_option(
        number, batch, best=best, unscored=index is None or opt.is_set_aside(index))
    # A formulation still waiting says what for (spec 2026-09-30 §4).
    if index is not None and not opt.is_set_aside(index) and not opt.fully_measured(index):
        label = wording.result_view_waiting(label, number_list(opt.waiting_on(index)))
    return wording.for_project(opt, label)


def _viewed_number(opt):
    """The formulation the View formulation box is showing, or None."""
    best = opt.best_index()
    best_no = int(opt.formulation_ids[best]) if best is not None else None
    numbers = _all_numbers(opt)
    if len(numbers) < 2:
        return numbers[0] if numbers else None
    chosen = st.session_state.get(_view_key(opt))
    return best_no if chosen in (None, BEST_CHOICE) else chosen


def _unscored_details(opt, number):
    row = next(r for r in opt.skipped if int(r['formulation']) == number)
    batch = row.get(ROUND_FIELD)
    # The box above names it (spec 2026-09-30 §4, review I12).
    st.info(wording.for_project(opt, wording.RESULT_UNSCORED_HELP))
    reason = wording.note_reason(row.get('note'))
    if reason:
        st.caption(reason)
    total = opt.recorded_total(batch) if opt.one_amount_unit() is not None else None
    shown_row = {'formulation': number, 'recipe': row.get('recipe') or {}}
    shown, _ = opt.shown_recipe(shown_row, total)
    st.markdown(wording.for_project(opt, wording.amounts_to_make_it_heading(opt.batch_total_text(total))))
    _amount_table(opt, shown)
    for caution in (opt.scaled_cautions([shown_row], total, total is not None)
                    + opt.total_mismatch_lines([shown_row], total)):
        st.caption(caution)


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


# ------------------------------------------------------------------ #
#  Round N results (final-v3, 3 · Results): the last Round's ratings
#  against their targets, and how one formulation's score is worked out.
#  Measurements only: no forecast, no range, no verdict.
# ------------------------------------------------------------------ #

_TIE = 1e-6          # Overall scores this close are equal (owner rule)
_FULL = 1e-9         # a fraction of a share this close to 1 is on target
# A new best's row: the best-so-far blue, as a tint under the reader's text
# colour in each theme; a total row: Streamlit's own secondary background.
_BEST_TINT = {"light": "#e8f0fc", "dark": "#1b2a4a"}
_TOTAL_TINT = {"light": "#f0f2f6", "dark": "#262730"}
BREAKDOWN_KEY = "score_breakdown"


def _rating_text(value):
    """12, 4.5, 12.25: a rating as it was entered, never 12.000000."""
    return f"{round(float(value), 2):g}"


def _scales(opt):
    return {(float(o.get('min_val', 0.0)), float(o.get('max_val', 10.0)))
            for o in opt.objectives}


def _common_scale(opt):
    """(low, high) when every measurement shares one scale, else None."""
    scales = _scales(opt)
    return next(iter(scales)) if len(scales) == 1 else None


def results_round(opt):
    """The Round 'Round N results' shows: the last one with anything
    recorded or moved to Not scored, or None (only past formulations)."""
    rounds = {int(b) for b in opt.batch_history if b is not None}
    rounds |= {int(r[ROUND_FIELD]) for r in opt.skipped
               if r.get(ROUND_FIELD) is not None}
    return max(rounds) if rounds else None


def _short(opt, number):
    return ui_insights._short(opt, f"F{int(number)}")


def _off(part):
    """A rating that lost points: off its target, or short of the end of
    the scale a higher-or-lower goal wants."""
    return (part['applies'] and part['utility'] is not None
            and part['utility'] < 1.0 - _FULL)


def new_best_rows(opt, round_no):
    """Positions of the Round's formulations that are a new Best so far:
    the Round's highest score among best_candidates, when it is above
    every earlier formulation's (an earlier Round's, or one made before the
    project) by more than _TIE; all of them when tied. [] when the Round
    did not raise the best."""
    pool = set(opt.best_candidates())
    mine = [i for i in pool if opt.batch_history[i] == round_no]
    if not mine:
        return []
    top = max(float(opt.Y_history[i]) for i in mine)
    earlier = [float(opt.Y_history[i]) for i in pool
               if opt.batch_history[i] is None or int(opt.batch_history[i]) < round_no]
    if earlier and top <= max(earlier) + _TIE:
        return []
    return sorted((i for i in mine if float(opt.Y_history[i]) >= top - _TIE),
                  key=lambda i: int(opt.formulation_ids[i]))


def round_results(opt, round_no=None):
    """'Round N results' as data: {'round', 'columns', 'frame' (text
    cells), 'off' {(row, column)}, 'best' [rows], 'caption'}. One row per
    formulation of the Round, in number order: Formulation, Overall score,
    then each measurement (most important first) with its target in the
    header. None when there is no Round to show."""
    round_no = results_round(opt) if round_no is None else round_no
    if round_no is None:
        return None
    common = _common_scale(opt)
    objs = opt.measurements_by_importance()
    headers = {o['name']: wording.round_results_column(
        o['name'], o.get('goal'), o.get('target'), o.get('unit', ''),
        None if common else (o.get('min_val', 0.0), o.get('max_val', 10.0)))
        for o in objs}
    form = wording.for_project(opt, wording.FORMULATION_CAP)
    score = wording.OVERALL_SCORE_COLUMN
    columns = [form, score] + [headers[o['name']] for o in objs]
    best = new_best_rows(opt, round_no)
    tied = len(best) > 1
    entries = [(int(opt.formulation_ids[i]), i) for i in range(len(opt.formulation_ids))
               if opt.batch_history[i] == round_no]
    entries += [(int(r['formulation']), None) for r in opt.skipped
                if r.get(ROUND_FIELD) is not None and int(r[ROUND_FIELD]) == round_no]
    rows, off, best_rows = [], set(), []
    for pos, (number, index) in enumerate(sorted(entries)):
        row = {c: "" for c in columns}
        row[form] = wording.round_results_row_label(
            _short(opt, number), best=index in best, tied=tied)
        if index in best:
            best_rows.append(pos)
        if index is None or opt.is_set_aside(index):
            row[score] = wording.NOT_SCORED
        elif not opt.fully_measured(index):
            row[score] = wording.MEASURED_WAITING
        else:
            row[score] = wording.score_shown(opt.Y_history[index])
        if index is not None:
            for part in opt.score_breakdown(index):
                column = headers[part['name']]
                if not part['applies']:
                    row[column] = wording.DOES_NOT_APPLY
                elif part['rating'] is not None:
                    row[column] = _rating_text(part['rating'])
                if _off(part):
                    off.add((pos, column))
        rows.append(row)
    caption = wording.round_results_caption(
        common, sample_projects.panel_size(opt),
        targets_only=all(o.get('goal') == 'target' for o in objs))
    return {"round": round_no, "columns": columns,
            "frame": pd.DataFrame(rows, columns=columns), "off": off,
            "best": best_rows, "caption": wording.for_project(opt, caption)}


def score_breakdown_frame(opt, index):
    """How formulation `index`'s Overall score is worked out, as data:
    {'frame', 'off' {(row, column)}, 'total' (the sum of the points),
    'line'}. One row per measurement — Attribute, Rating, Target, Share,
    Points earned — then the Overall score row, Share and Points summed.
    Every number is food_bo's own (score_breakdown, the parts
    _compute_utility adds), so the total is the Overall score."""
    parts = opt.score_breakdown(index)
    cols = (wording.SCORE_ATTRIBUTE_COLUMN, wording.SCORE_RATING_COLUMN,
            wording.SCORE_TARGET_COLUMN, wording.SCORE_SHARE_COLUMN,
            wording.SCORE_POINTS_COLUMN)
    rows, off = [], set()
    for pos, part in enumerate(parts):
        if part['goal'] == 'target':
            target = f"{part['target']:g}"
        elif part['goal'] == 'max':
            target = f"{part['high']:g} ({wording.SCORE_GOAL_HIGHER})"
        else:
            target = f"{part['low']:g} ({wording.SCORE_GOAL_LOWER})"
        rows.append({
            cols[0]: part['name'],
            cols[1]: (wording.DOES_NOT_APPLY if not part['applies']
                      else "" if part['rating'] is None else _rating_text(part['rating'])),
            cols[2]: target,
            cols[3]: ("" if part['share_of_score'] is None
                      else f"{round(part['share_of_score'], 1):g}"),
            cols[4]: "" if part['points'] is None else f"{part['points']:.1f}"})
        if _off(part):
            off.update({(pos, cols[1]), (pos, cols[4])})
    total = sum(p['points'] for p in parts if p['points'] is not None)
    shares = sum(p['share_of_score'] for p in parts if p['share_of_score'] is not None)
    rows.append({cols[0]: wording.OVERALL_SCORE_COLUMN, cols[1]: "", cols[2]: "",
                 cols[3]: f"{round(shares, 1):g}", cols[4]: f"{total:.1f}"})
    common = _common_scale(opt)
    return {"frame": pd.DataFrame(rows, columns=list(cols)), "off": off, "total": total,
            "line": wording.score_point_line(None if common is None
                                             else common[1] - common[0])}


def _styled(frame, off, tinted=(), bold=(), total_row=None):
    """`frame` for st.table: off-target cells in the theme's orange, the
    `tinted` rows in the best-so-far blue, `bold` {(row, column)} cells
    bold, a total row on the secondary background and bold. Theme-keyed,
    never one theme's colours."""
    which = ui_insights.theme()
    orange = ui_insights.outline_colour()

    def css(data):
        out = pd.DataFrame("", index=data.index, columns=data.columns)
        for r in range(len(data)):
            for c in data.columns:
                rules = []
                if r in tinted:
                    rules.append(f"background-color: {_BEST_TINT[which]}")
                if r == total_row:
                    rules.append(f"background-color: {_TOTAL_TINT[which]}")
                if (r, c) in off:
                    rules.append(f"color: {orange}")
                out.iloc[r, list(data.columns).index(c)] = "; ".join(rules)
        return out

    # Bold as Markdown, which st.table renders in every cell: it draws a
    # Styler's colours but not its font-weight.
    strong = set(bold) | ({(total_row, c) for c in frame.columns}
                          if total_row is not None else set())
    marked = frame.copy()
    for r, c in strong:
        if str(marked.iat[r, list(frame.columns).index(c)]):
            marked.iat[r, list(frame.columns).index(c)] = (
                f"**{marked.iat[r, list(frame.columns).index(c)]}**")
    # The first column is the index: st.table always draws one, and a
    # column of 0, 1, 2 beside the formulations read as a fourth number.
    first = frame.columns[0]
    shown = marked.set_index(first)
    return shown.style.apply(lambda data: css(frame).drop(columns=first).set_axis(data.index),
                             axis=None)


def _round_results_section(opt):
    """'Round N results', its caption and table, then the fold that works
    one formulation's score out. Nothing when there is no Round yet."""
    if not opt.objectives:
        return
    table = round_results(opt)
    if table is None or table["frame"].empty:
        return
    st.markdown(f"#### {wording.round_results_heading(table['round'])}")
    st.caption(table["caption"])
    score = wording.OVERALL_SCORE_COLUMN
    st.table(_styled(table["frame"], table["off"], tinted=set(table["best"]),
                     bold={(r, score) for r in table["best"]}),
             width="content")
    _score_breakdown(opt, table)


def _score_breakdown(opt, table):
    """The fold: pick a formulation (the Round's best by default), and its
    Overall score worked out measurement by measurement."""
    scored = [int(n) for n in opt.formulation_ids]
    if not scored or not opt.objectives:
        return
    mine = [i for i in range(len(opt.formulation_ids))
            if opt.batch_history[i] == table["round"]]
    best_rows = new_best_rows(opt, table["round"])
    pool = [i for i in mine if opt.fully_measured(i)] or mine
    default_index = (best_rows[0] if best_rows
                     else max(pool, key=lambda i: float(opt.Y_history[i])) if pool else None)
    default = (int(opt.formulation_ids[default_index]) if default_index is not None
               else sorted(scored)[-1])
    key = f"{BREAKDOWN_KEY}_{opt.project_name}"
    seen = f"_{key}_round"
    if (st.session_state.get(seen) != table["round"]
            or st.session_state.get(key) not in scored):
        st.session_state[key] = default
    st.session_state[seen] = table["round"]
    with st.expander(wording.SCORE_WORKED_OUT_FOLD, key=f"{BREAKDOWN_KEY}_fold"):
        # A label that never changes with Best so far or a result arriving:
        # the browser keeps the label it was shown, and a stale one is no
        # option at all.
        rounds = {int(n): b for n, b in zip(opt.formulation_ids, opt.batch_history)}
        number = st.selectbox(
            wording.for_project(opt, wording.SCORE_WORKED_OUT_PICK), sorted(scored),
            format_func=lambda n: wording.for_project(
                opt, wording.result_view_option(n, rounds.get(n))), key=key)
        index = opt.index_of_formulation(number)
        if index is None:
            return
        shown = score_breakdown_frame(opt, index)
        st.caption(shown["line"])
        last = len(shown["frame"]) - 1
        st.table(_styled(shown["frame"], shown["off"], total_row=last), width="content")


def _progress_line(opt):
    last = opt.last_batch_no()   # counts a batch nobody managed to make
    # The formulations Best so far compares, and only those.
    pool = opt.best_candidates()
    if last is None or not pool:
        return ""
    # Results' own before/after rule, whatever the Round (review M18): Best
    # so far against the next highest, by the I5 yardstick. A tie made by a
    # late Cook loss or a correction is said here, not only in the flash.
    ranked = sorted(pool, key=lambda i: float(opt.Y_history[i]), reverse=True)
    partner = _tie_partner(opt, ranked[0])
    tie = _tie(opt, partner, ranked[0]) if partner is not None else ""
    if tie:
        return tie
    earlier = [float(opt.Y_history[i]) for i in pool
               if opt.batch_history[i] != last]
    if not earlier:
        # Nothing to compare it with. The flash above the tabs already says
        # "Round 1 recorded."; saying it again four lines lower is the same
        # sentence twice on one screen.
        return ""
    best_now = max(float(opt.Y_history[i]) for i in pool)
    best_before = max(earlier)
    if best_now > best_before + 1e-9:
        return wording.batch_recorded_progress(last, best_before, best_now)
    return wording.batch_recorded_no_improvement(last)


def _yardstick(opt):
    """How much Repeats differ, for Results' own before/after rule (spec
    2026-09-30 review I5, M18): the checks Insights has already worked out
    leave a disagreeing Repeat out; Results never checks Rounds itself."""
    return insights.noise_yardstick(opt, ui_insights.cached_checks(opt))


def _tie_partner(opt, best):
    """The row Best so far (row `best`) is compared with for a tie: the next
    highest DIFFERENT formulation — a Repeat of it is the same make-up,
    never a tie (Plan B parked item). The line and the flash both ask
    here, so they never disagree (re-review R-2). None without one. The
    rule is insights.tie_partner's, which Insights asks too (ftue review
    R-I9)."""
    return insights.tie_partner(opt, best)


def _same_formulation(opt, i, j):
    """Rows `i` and `j` are one make-up: one a Repeat of the other."""
    return insights._same(opt, insights.start_recipe(opt, i), insights.start_recipe(opt, j))


def _tie(opt, old, new):
    """The tie sentence when Best so far moved from row `old` to row `new`
    by less than one new result needs to differ (rise_threshold(yard, 1):
    2.33 × the noise SD, t on the Repeats' df), else ''. A lead
    of 0.00 is an exact tie, said as one; a Repeat of the same make-up is
    never a tie (Plan B parked items)."""
    if _same_formulation(opt, old, new):
        return ""
    lead = float(opt.Y_history[new]) - float(opt.Y_history[old])
    # Fix A follow-up (ruling on ties): a lead is a tie under what one new
    # result needs to read Rising, rise_threshold(yard, 1), so Results'
    # tie and Insights' Rising never disagree.
    need = insights.rise_threshold(_yardstick(opt), 1)
    if lead < 0 or lead >= need:
        return ""
    # An exact tie on the stored scores, up to float error (owner rule:
    # full precision with a tolerance, never the shown text).
    if lead <= _TIE:
        return wording.tie_equal(int(opt.formulation_ids[new]), int(opt.formulation_ids[old]),
                                 wording.score_shown(opt.Y_history[new]))
    # Specific wording: the lead against the rise one new result needs, to
    # one decimal as scores are shown; both to two when one would read the
    # lead no smaller than the need ("2.30 lower; 2.33 needed", never "2.3
    # lower; 2.3 needed").
    digits = 1 if round(lead, 1) < round(need, 1) and round(lead, 1) > 0 else 2
    return wording.tie_line(int(opt.formulation_ids[new]), wording.score_shown(opt.Y_history[new]),
                            f"{lead:.{digits}f}", int(opt.formulation_ids[old]),
                            f"{need:.{digits}f}")


def _move_sentence(opt, before, after):
    """best_move_sentence, or the tie sentence when the new Best so far
    leads by less than Repeats differ (§4 ties): the flash says what the
    line under Best so far says."""
    move = best_move_sentence(before, after)
    if not move:
        return ""
    old, new = opt.index_of_formulation(before), opt.index_of_formulation(after)
    if old is None or new is None:
        return move
    partner = _tie_partner(opt, new)
    return (_tie(opt, partner, new) if partner is not None else "") or move


def _best(opt):
    """Details for the selected formulation, defaulting to the best. The box
    names it, so there is no heading (spec 2026-09-30 §4, review I12); its
    Overall score is the first line under the box. Returns (said_partial,
    missing, settled): said_partial is True when it has already said what a
    dash means, so the table below does not say it again; missing is the
    reserved slot for the missing-values panel of a formulation that waits
    or was moved to Not scored, or None, which render() fills last;
    settled is True when the formulation on screen has all its results."""
    if opt.Y_history and opt.best_index() is None:
        # Nothing has all its results, so there is no best to name.
        st.caption(wording.for_project(opt, wording.NO_COMPLETE_FORMULATION))
    _confirmations()
    number = _selected_formulation(opt)
    if number is None:
        return False, None, False
    index = opt.index_of_formulation(number)
    if index is None:
        _unscored_details(opt, number)
        return False, None, False
    batch = opt.batch_history[index]
    is_best = index == opt.best_index()
    ceiling = opt.utility_ceiling()
    # A formulation still waiting shows what it waits for, never the lowered
    # number — named exactly as the All formulations row names it.
    waiting = opt.waiting_on(index) if not opt.fully_measured(index) else None
    # A dash is a measurement it is scored without; a waiting cell reads
    # "waiting" (audit V9): the dash caption only when a dash is on screen.
    partial = bool(opt.not_expected(index))
    missing = None
    if waiting is not None:
        st.caption(wording.not_scored_missing(number_list(waiting)) + "."
                   if opt.is_set_aside(index)
                   else wording.overall_score_waiting(number_list(waiting)))
    elif ceiling > 0:
        # With every measurement deleted the ceiling is 0, and "Overall
        # score 0.00 of 0" is a sentence about nothing.
        st.caption(wording.overall_score_line(float(opt.Y_history[index]), ceiling,
                                              number_list(opt.not_expected(index))),
                   help=wording.OVERALL_SCORE_HELP)
    if is_best:
        # One rule line (audit V8, review I10), then what the last Round
        # did, then what is still waiting.
        without = opt.best_scored_without()
        st.caption(wording.for_project(opt, wording.BEST_RULE_WITHOUT
                                       if without else wording.BEST_RULE),
                   help=wording.for_project(opt, wording.BEST_RULE_WITHOUT_HELP
                                            if without else wording.BEST_RULE_HELP))
        repeats = best_repeats_text(opt)
        if repeats:
            st.caption(wording.for_project(opt, repeats))
        line = _progress_line(opt)
        if line:
            st.caption(wording.for_project(opt, line))
        for text in _waiting_lines(opt):
            st.caption(wording.for_project(opt, text))

    details = opt.closeness_details(index)
    if details:
        # Off by is a distance from a target, so it only appears once a
        # measurement has one: a column of dashes said nothing on every row.
        has_target = any(o['goal'] == 'target' for o in opt.objectives)
        # Round-2 fixes (P2 #15): the table named, as every other one here.
        st.markdown(f"**{wording.for_project(opt, wording.OFF_BY_TITLE)}**")
        st.dataframe(pd.DataFrame([
            {wording.MEASUREMENT_COLUMN: d['name'], wording.GOAL_LABEL: d['goal'],
             wording.MEASURED_COLUMN: d['measured'],
             **({wording.OFF_BY_COLUMN: d['off_by']} if has_target else {})}
            for d in details
        ]), hide_index=True, key="best_off_by",
            height=table_height(len(details)))

    recipe = opt.recipe_history[index]
    # The amounts stored are as generated. When the batch this formulation
    # came from was printed to a total, the bench weighed out those numbers
    # instead, and they are the ones to hand back — scaled by the same rule
    # the sheets used, and with the total named in the heading so the two
    # cannot be confused. `Start from Best so far` opens at the amounts at
    # the Formulation total: those are the ones the models learn it at (spec
    # 2026-09-30-bo-views §1, review M1).
    total = (opt.recorded_total(batch)
             if opt.one_amount_unit() is not None else None)
    # The row as the sheet carried it: a formulation of the user's own — the
    # note is what marks one — was printed exactly as typed, and nothing at
    # all is rewritten under a project total.
    note = opt.notes_history[index] if index < len(opt.notes_history) else ""
    row = {'formulation': number, 'recipe': recipe, 'note': note}
    shown, _ = opt.shown_recipe(row, total)
    st.markdown(wording.for_project(opt, wording.amounts_to_make_it_heading(opt.batch_total_text(total))))
    # A blank index: st.table always draws one, and a list of ingredients
    # numbered from 0 beside a properties grid numbered from 1 had the cold
    # reader reading two different tables of the same ingredients.
    _amount_table(opt, shown)
    # The amounts above are the ones the bench weighed out, so the same lines
    # tab 2 shows under the box belong under the table that shows them.
    for caution in (opt.scaled_cautions([row], total, total is not None)
                    + opt.total_mismatch_lines([row], total)):
        st.caption(caution)
    # Ingredients only: a process setting sitting at 0 is a setting, not an
    # ingredient somebody left out.
    unused = [name for name in opt.ingredient_names()
              if not float(recipe.get(name, 0.0))]
    if unused:
        st.caption(wording.NOT_USED_PREFIX + ", ".join(unused))
    if waiting is not None:
        # Reserved here, drawn last by render(): a confirmation armed in
        # the fold below must grey these on the same run.
        missing = {"number": number, "names": waiting, "slot": st.container()}
    if partial:
        st.caption(_complete_results_caption(opt), help=_complete_results_help(opt))
    return partial, missing, waiting is None


def best_repeats_text(opt, checks=None):
    """A6 (review I7): Best so far's Repeats under it, "Repeats of
    Formulation 14: 67.3, 66.3, 75.6. Mean of all: 75.2.", or "". One
    decimal, as every screen shows scores; the mean is taken from
    the stored scores. A Repeat disagreeing_repeat leaves out of the noise
    is still listed, marked. Display only: Best so far is still the single
    highest score (ranking by the mean is an owner decision). `checks`:
    the Round-by-Round checks, when worked out (Results never checks
    Rounds itself)."""
    best = opt.best_index()
    if best is None:
        return ""
    group = next((g for g in insights.repeat_groups(opt, opt.best_candidates())
                  if best in g), None)
    if not group:
        return ""
    checks = ui_insights.cached_checks(opt) if checks is None else checks
    flagged = insights.disagreeing_repeat(opt, checks) if checks else None
    others = sorted((i for i in group if i != best), key=lambda i: int(opt.formulation_ids[i]))
    scores = [wording.repeat_score(wording.score_shown(opt.Y_history[i]),
                                   disagrees=flagged is not None and flagged.row == i)
              for i in others]
    mean = sum(float(opt.Y_history[i]) for i in group) / len(group)
    return wording.best_repeats_line(int(opt.formulation_ids[best]), scores,
                                     wording.score_shown(mean))


def _waiting_lines(opt):
    """'Waiting for Cook loss: Formulation 22 (Round 6).' — one line per
    set of measurements the waiting formulations lack (spec 2026-09-30 §4,
    review M7). A formulation moved to Not scored is not waiting."""
    groups = {}
    for i in opt.awaited_rows():
        groups.setdefault(tuple(opt.waiting_on(i)), []).append(i)
    lines = []
    for names, rows in groups.items():
        numbers = [int(opt.formulation_ids[i]) for i in rows]
        lines.append(wording.waiting_list(number_list(names), numbers,
                                          opt.batch_history[rows[0]] if len(rows) == 1
                                          else None))
    return lines


def _complete_results_help(opt):
    return wording.for_project(opt, wording.complete_results_help(
        not opt.best_scored_without()))


def _complete_results_caption(opt):
    """What a dash means, without contradicting the star: while no
    formulation has every measurement, Best so far has a dash itself."""
    return wording.for_project(opt, wording.complete_results_caption(
        not opt.best_scored_without()))


def _missing_key(no, name):
    return f"missing_{no}_{name}"


def _missing_panel(opt, number, names, correcting=False):
    """Under a formulation still waiting for results, or moved to Not
    scored: one box per missing measurement and Save, the lit button —
    entering the value is the first way out, and it used to be the hardest
    (review 0.8.0) — then the secondary one: Move to Not scored, behind a
    confirmation, or Wait for its results again. Greyed and ignored while
    another confirmation is open.

    `correcting`: a correction is open in Correct, delete or add past
    formulations, and its Save correction is the one lit button on the tab;
    this Save stays usable, in grey. With nothing to name (every measurement
    the row was measured on deleted) there is nothing to type, so no boxes
    and no Save."""
    index = opt.index_of_formulation(number)
    lit = not confirmation_open()
    if names:
        objs = {o['name']: o for o in opt.objectives}
        typed = _measurement_boxes([objs[n] for n in names if n in objs],
                                   lambda name: _missing_key(number, name))
        entered = any(v is not None for v in typed.values())
        # Enabled always (spec 2026-09-30 review M12): a box commits on
        # blur, so a Save disabled until then would swallow the first click.
        # Lit only once a value is typed; an empty click says what to do.
        if st.button(wording.save_missing(number_list(names)),
                     key=f"save_missing_{number}",
                     type="primary" if lit and entered and not correcting else "secondary",
                     disabled=not lit) and lit:
            _save_missing(opt, number, typed)
    key = f"set_aside_{number}"
    if opt.is_set_aside(index):
        if st.button(wording.WAIT_AGAIN, key=f"wait_again_{number}",
                     disabled=not lit) and lit:
            opt.wait_again(number)
            if saved_ok(opt):
                flash_here("success", wording.for_project(
                    opt, wording.waiting_again(number, number_list(names))))
                st.rerun()
        return
    if confirm_action(key, wording.MOVE_TO_NOT_SCORED,
                      wording.for_project(opt, wording.never_coming_warning(
                          number, number_list(names))),
                      # Names the act (spec §4, review I9), and is not a
                      # second copy of the button that opened it (01 O13).
                      confirm_label=wording.YES_MOVE_TO_NOT_SCORED,
                      disabled=other_confirmation(key),
                      help=wording.for_project(opt, wording.move_to_not_scored_help(len(names)))):
        opt.set_aside_formulation(number)
        if saved_ok(opt):
            flash_here("success", wording.for_project(opt, wording.moved_to_not_scored(number)))
            st.rerun()


def _save_missing(opt, number, typed):
    """Write the values typed into the missing boxes onto the recorded row.
    Nothing recorded is overwritten, so no copy is kept first; a blank box
    stays missing. A row moved to Not scored that still lacks a value stays
    Not scored, and the flash says so rather than 'scored'."""
    objs = {o['name']: o for o in opt.objectives}
    for name, value in typed.items():
        problem = scale_error(objs[name], value)
        if problem:
            st.error(problem)
            return
    entered = {n: float(v) for n, v in typed.items() if v is not None}
    if not entered:
        st.error(wording.ENTER_A_MEASUREMENT)
        return
    index = opt.index_of_formulation(number)
    before = best_formulation_no(opt)
    opt.edit_result(index, {**opt.results_history[index], **entered})
    if not saved_ok(opt):
        return
    for name in typed:
        st.session_state.pop(_missing_key(number, name), None)
    if opt.is_set_aside(index):
        first = wording.saved_not_scored(number, number_list(opt.waiting_on(index)))
    else:
        first = wording.formulation_scored(number, waiting_text(opt, index))
    sentences = _scored_then_move(number, first,
                                  _move_sentence(opt, before, best_formulation_no(opt)))
    # Above the View formulation box, where the reader is (01 O3: the tie
    # flash was 287 px above the Save Cook loss they had just pressed).
    flash_here("success", wording.for_project(opt, " ".join(sentences)))
    st.rerun()


def _scored_then_move(number, first, move):
    """The flash's sentences after a result arrives: `first`, then what
    Best so far did. A tie already says "Formulation 4 scored 70.50, …",
    so it replaces "Formulation 4 scored." rather than repeating it."""
    if move.startswith(f"{wording.FORMULATION_CAP} {number} scored "):
        return [move]
    return [first, move] if move else [first]


def _amount_format(opt, frame):
    """How each amount column of the All formulations table is written out:
    two decimals, because that is the precision a balance works to and what
    every other table in the app shows — `Show amounts` printed 11.875 beside
    the batch sheet's 11.88. A process setting is not an amount and keeps
    fmt_setting's own rule."""
    settings = {opt._amount_column(v['name']) for v in opt.variables
                if v.get('category') == 'process'}
    amounts = {opt._amount_column(v['name']) for v in opt.variables}

    def weighed(value):
        # A not-scored row may hold no amount for a variable added later,
        # and a measurement left blank stays blank rather than reading "nan".
        # A cell that is not a number at all — a note, a date, a score with
        # a sentence behind it — is handed back as it came.
        if value is None:
            return ""
        if not isinstance(value, str) and pd.isna(value):
            return ""
        try:
            return fmt_amount(float(value))
        except (TypeError, ValueError):
            return value

    # Every number in the table reads to two decimals: the amounts, the
    # measurements and the score alike. Left to pandas, a panel score typed
    # as 5 came out as 5.000000 beside an amount written 11.88. A process
    # setting keeps fmt_setting (180, not 180.00), and a text column — a
    # score reading "waiting for Juiciness", Note, Best — is left alone.
    # Every column but the two counts and the settings. Read off the dtype,
    # a column that holds one blank went to `object` and slipped the net
    # entirely: the amounts of a row recorded before a pre-mix moved printed
    # as 31.13258332014084 beside their neighbours' 31.13.
    numeric = {c for c in frame.columns
               if c not in amounts
               and c not in (wording.FORMULATION_CAP, wording.ROUND_CAP)}
    formats = {c: (fmt_setting if c in settings else weighed)
               for c in frame.columns if c in amounts}
    formats.update({c: weighed for c in numeric})
    return formats


def _table_style(opt, frame):
    """The All formulations table as it is drawn: every number to two
    decimals, and a dash where a measurement does not apply to the row. The
    dash is display only — the cell stays a blank number, so Streamlit keeps
    the column numeric and sorts it as numbers."""
    styler = frame.style.format(_amount_format(opt, frame))
    columns = {o['name']: opt._measurement_column(o) for o in opt.objectives}
    dashes = {}
    for position, number in enumerate(frame[wording.FORMULATION_CAP]):
        index = opt.index_of_formulation(int(number))
        if index is None:
            continue
        for name in opt.not_expected(index):
            if columns.get(name) in frame.columns:
                dashes.setdefault(columns[name], []).append(frame.index[position])
    for column, rows in dashes.items():
        styler = styler.format(lambda _: wording.DOES_NOT_APPLY,
                               subset=pd.IndexSlice[rows, [column]])
    return styler


def _all_formulations(opt, said_partial=False):
    st.markdown(wording.for_project(opt, wording.ALL_FORMULATIONS_HEADING))
    o1, o2 = st.columns([2, 1])
    with o1:
        # The three options are a protocol with food_bo.history_frame, which
        # compares them verbatim; both modules read the one list in wording.
        order = st.selectbox(wording.SORT_LABEL, wording.SORT_OPTIONS,
                             key="results_order")
    with o2:
        show_amounts = st.toggle(wording.SHOW_AMOUNTS_TOGGLE, key="show_amounts")
    frame = opt.history_frame(order=order, include_amounts=show_amounts)
    # Round-2 fixes (P2 #15): a column as wide as its name ("Coconut cream
    # (g)" was cut to "Coconut cr…"); the table scrolls sideways instead.
    config = {c: st.column_config.Column(width=max(80, 8 * len(str(c)) + 24))
              for c in frame.columns if len(str(c)) > 9}
    config[wording.FORMULATION_CAP] = st.column_config.Column(
        wording.for_project(opt, wording.FORMULATION_CAP))
    st.dataframe(_table_style(opt, frame),
                     column_config=config,
                 hide_index=True, key="all_formulations",
                 height=table_height(len(frame), max_rows=20))
    # What a dash means, only while a dash is in the table (audit V9): a
    # waiting cell is not one.
    if not said_partial and any(opt.not_expected(i) for i in range(len(opt.Y_history))):
        st.caption(_complete_results_caption(opt), help=_complete_results_help(opt))
    # One workbook, not a comma-separated file: the same table the screen
    # shows, and a Set-up sheet beside it saying what the targets and the
    # allowed amounts were. A column of numbers with nothing to read it
    # against is a file nobody can use six months later.
    st.download_button(wording.for_project(opt, wording.DOWNLOAD_ALL_FORMULATIONS_BUTTON),
                       data=opt.all_formulations_workbook(),
                       file_name=wording.all_formulations_file_name(
                           opt.project_name),
                       mime=WORKBOOK_MIME, key="download_formulations",
                       help=wording.DOWNLOAD_ALL_FORMULATIONS_HELP)


def _correct_amount_key(no, name):
    """The box for one amount under `Correct a formulation`. The
    correct_amount_ prefix is what app.py empties on a project switch, so a
    half-typed correction never follows the user into the next project."""
    return f"correct_amount_{no}_{name}"


def _correct_measurement_key(no, name):
    return f"correct_{no}_{name}"


def _correct_note_key(no):
    """The Note box on a not-scored formulation's scoring row. `correct_` is
    what preserve_tab_forms keeps across a rerun that does not draw it."""
    return f"correct_note_{no}"


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


def _amount_boxes(opt, key_of, recipe=None, lock_worked_out=False):
    """One number box per variable, keyed by `key_of(name)`. With a recipe the
    boxes open on what that row recorded; without one they open blank.

    A formulation is corrected as often for what went into the bowl — a
    misread balance, a line transposed off the batch sheet — as for what came
    off the panel, and until now only the measurements could be fixed.

    A row that is worked out from a rule never offers its own Lowest and
    Highest as the hint: they are dormant, the Set up grid has replaced
    them with a word, and the box was showing the numbers the row carried
    BEFORE it was given a rule — a third answer to a question two other
    screens had already answered. `lock_worked_out` greys it as well, which
    is what the same row's box on Make a round does: on the form that
    RECORDS a formulation the app works the row out. On the form that
    CORRECTS one it stays typeable, because what the balance actually gave
    is the whole point of correcting, and the printed sheet leaves that
    row's Actual (g) cell open for exactly that.
    """
    for name, group in opt.premixes.items():
        if group['mode'] == 'weighed':
            st.caption(wording.premix_members_line(
                name, ", ".join(p['name'] for p in group['parts'])))
    typed = {}
    for var, col in _in_fours(opt.variables):
        with col:
            name = var['name']
            # The placeholder is the allowed amounts at the size this row
            # was made to: the boxes were prefilled with 250 g grams under
            # a hint reading "0–15", which is the band per 100 g.
            scale = (opt.recipe_scale(recipe)
                     if recipe and var.get("category", "ingredient") == "ingredient"
                     else 1.0)
            low, high = (float(b) * scale for b in var['bounds'])
            worked_out = opt.has_formula(var)
            # No min_value/max_value: an amount outside what the project
            # allows is a fact about work already done, and clamping it would
            # quietly record a formulation nobody made. It is a caution on
            # the way out, exactly as an imported amount is.
            # Two decimals, as the sheet the bench weighed from printed it:
            # a box holding 22.241068936455125 under a label that says (g)
            # claims the balance read that, and the sheet said 22.24.
            st.session_state.setdefault(
                key_of(name),
                None if recipe is None
                else round(_recorded_amount(var, recipe), 2))
            # The All formulations table's own header, so an amount is typed
            # in the unit that table prints it in.
            typed[name] = st.number_input(
                opt._amount_column(name),
                placeholder=(wording.WORKED_OUT if worked_out
                             else amount_range_placeholder(low, high)),
                disabled=worked_out and lock_worked_out,
                key=key_of(name), format="%.2f")
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
            # The unit in the label, as the record form says it: the box
            # holds a bare number (answer-key read-through).
            typed[name] = st.number_input(
                f"{entry_label(name, obj.get('unit'))} · {goal_line(obj)}",
                placeholder=f"{obj['min_val']:g}–{obj['max_val']:g}",
                key=key_of(name))
    return typed


def _score_row(opt, choice):
    """The row a not-scored formulation opens under `Correct a formulation`.

    Nothing recorded is being changed here: the amounts are the ones the
    project generated and stored, shown as they are shown everywhere else,
    and the one thing missing is a result. So they are read, not offered in
    boxes — an editable amount invites an edit nobody came here to make —
    and the measurement boxes open empty. The Note box opens on why the row
    was not scored, which is the only thing anybody has typed about it."""
    row = next((s for s in opt.skipped
                if int(s['formulation']) == int(choice)), None)
    if row is None:
        return None
    recipe = dict(row.get('recipe') or {})
    # The amounts the bench would have weighed out, handed back exactly as
    # the best-so-far block hands them back: the stored ones are as
    # generated, so when this row's batch was printed to a total the sheet
    # carried different numbers, and the heading names which of the two is
    # on screen.
    batch = row.get(ROUND_FIELD)
    total = (opt.recorded_total(batch)
             if opt.one_amount_unit() is not None else None)
    # No note here: a not-scored row's note says why nobody scored it, not
    # that the user wrote the formulation out themselves.
    shown_row = {'formulation': int(choice), 'recipe': recipe}
    shown, _ = opt.shown_recipe(shown_row, total)
    st.markdown(wording.for_project(opt, wording.amounts_to_make_it_heading(opt.batch_total_text(total))))
    # A blank index: st.table always draws one, and a list of ingredients
    # numbered from 0 beside a properties grid numbered from 1 had the cold
    # reader reading two different tables of the same ingredients.
    _amount_table(opt, shown)
    # It is the total, not the formulation, that pushes an amount out of the
    # allowed ones — the same lines tab 2 shows under its box, and the best
    # block under the same table.
    for caution in (opt.scaled_cautions([shown_row], total, total is not None)
                    + opt.total_mismatch_lines([shown_row], total)):
        st.caption(caution)
    ordered = opt.measurements_by_importance()
    typed = _measurement_boxes(
        ordered, lambda name: _correct_measurement_key(choice, name))
    # Why it was not scored is the only thing anyone typed about this row,
    # and it stays the row's note: the box opens on the reason, without the
    # marker in front of it, and what is left here is what the scored row
    # carries. Emptying it is allowed — the reason may be exactly what
    # stopped being true.
    st.session_state.setdefault(_correct_note_key(choice),
                                wording.note_reason(row.get('note')))
    note = st.text_input(wording.NOTE, key=_correct_note_key(choice))
    return {"choice": int(choice), "index": None, "ordered": ordered,
            "current": {}, "typed": typed, "recipe": recipe, "note": note,
            "amounts": None, "skipped": True, "slot": st.container()}


def _correct(opt):
    """The correction row: the select box, the amounts the formulation was
    really made with, and the measurements to retype. Returns the open
    correction, or None — the foot reads that and steps aside, because
    `Save correction` is the one lit action until the row is answered.

    Its button row is only RESERVED here. Arming a confirmation in one of the
    sections below does not rerun, so a Save correction drawn now would still
    be coloured on the very run that puts a Yes beside it; render() fills the
    slot once the confirmations have had their say."""
    scored = [int(n) for n in opt.formulation_ids]
    not_scored = [int(s['formulation']) for s in opt.skipped]
    # A recorded formulation moved to Not scored reads as Not scored here
    # too, as in the View box and the table, but it is still a recorded row
    # and opens as one (final review M-4).
    set_aside = {int(opt.formulation_ids[i]) for i in range(len(opt.formulation_ids))
                 if opt.is_set_aside(i)}
    # One run of numbers, in number order: the list came out 1, 2, 4, 3 —
    # scored ones first and the rest after — which is not an order anybody
    # could read. A not-scored one says so on its own line instead.
    numbers = sorted(scored + not_scored)
    if not numbers:
        # Not "No results yet.": that sentence is already the whole screen
        # above this section on a project holding nothing.
        st.caption(wording.for_project(opt, wording.no_formulation_to_correct_caption()))
        return None
    take_clear("correct_formulation")
    # The label says what picking one DOES; the placeholder says what the box
    # holds. A bare "Formulation" on both left the reader to infer the verb
    # from a heading three rows up.
    best = opt.best_index()
    best_no = int(opt.formulation_ids[best]) if best is not None else None
    choice = st.selectbox(
        wording.for_project(opt, wording.CORRECT_WHICH_LABEL), numbers, index=None,
        placeholder=wording.CHOOSE_A_FORMULATION_PLACEHOLDER,
        # As the View formulation box reads them (polish round I8), not
        # bare numbers.
        format_func=lambda n: formulation_option(opt, n, best=n == best_no),
        key="correct_formulation")
    if not_scored or set_aside:
        # A not-scored number in the list is not a correction, and nothing
        # about the box says what picking one does. A set-aside one is
        # scored here too, by entering its missing value.
        st.caption(wording.for_project(opt, wording.NOT_SCORED_CAN_BE_SCORED_CAPTION))
    if choice is None:
        return None
    if int(choice) in not_scored:
        return _score_row(opt, int(choice))
    index = opt.index_of_formulation(choice)
    if index is None:
        return None
    recipe = opt.recipe_history[index]
    # Said once, above every box on the form: the boxes open pre-filled, so a
    # per-box "leave blank" described a state the reader was not in.
    st.caption(wording.CORRECTION_CAPTION)
    amounts = _amount_boxes(opt, lambda name: _correct_amount_key(choice, name),
                            recipe)
    ordered = opt.measurements_by_importance()
    current = opt.results_history[index]
    typed = _measurement_boxes(
        ordered, lambda name: _correct_measurement_key(choice, name), current)
    # The same field a not-scored row gets. A note is part of the record —
    # which bowl it was, what went wrong — and it was the one thing a scored
    # formulation could not have corrected.
    recorded_note = (opt.notes_history[index]
                     if index < len(opt.notes_history) else "")
    st.session_state.setdefault(_correct_note_key(choice), recorded_note)
    note = st.text_input(wording.NOTE, key=_correct_note_key(choice))
    return {"choice": choice, "index": index, "ordered": ordered,
            "current": current, "typed": typed, "recipe": recipe,
            "amounts": amounts, "note": note, "recorded_note": recorded_note,
            "slot": st.container()}


def _close_correction(opt, choice):
    """Forget what was typed into this row's boxes. They are gone from the
    screen on the next run, so popping is enough — nothing is mounted in the
    browser to post the old value back."""
    for var in opt.variables:
        st.session_state.pop(_correct_amount_key(choice, var['name']), None)
    for obj in opt.objectives:
        st.session_state.pop(_correct_measurement_key(choice, obj['name']), None)
    st.session_state.pop(_correct_note_key(choice), None)
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
        # A not-scored row has no result to correct: this writes its first.
        label = (wording.SAVE_RESULT_BUTTON if pending.get("skipped")
                 else wording.SAVE_CORRECTION_BUTTON)
        save = st.button(label, key="save_correction",
                         type="primary" if lit else "secondary",
                         disabled=not lit, width="stretch") and lit
    with b2:
        # The select box cannot be cleared by the user once it holds a value.
        if st.button(wording.CLOSE_BUTTON, key="done_correcting", width="stretch"):
            _close_correction(opt, choice)
            st.rerun()
    if not save:
        return
    if pending.get("skipped"):
        # A not-scored formulation has no recorded result to change: this
        # writes its first one.
        _save_score(opt, storage, pending)
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
        # At the box's own precision: it was SEEDED with the recorded
        # amount rounded to two decimals, so an untouched box always looked
        # changed against the raw number behind it.
        recorded = _recorded_amount(var, pending['recipe'])
        if abs(round(recorded, 2) - round(float(value), 2)) > 1e-9:
            amount_changes.append(name)
            recipe[name] = float(value)
        elif name not in recipe:
            # A row older than this ingredient has no amount for it at all.
            recipe[name] = float(value)
        # Otherwise the recorded amount stands, to the last digit. The box
        # shows it at two decimals because that is what the balance reads
        # and what the sheet printed; writing that display back turned
        # 15.18876389 into 15.19 under a line promising that anything left
        # alone stays as recorded.
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
    note_changed = (not pending.get("skipped")
                    and str(pending.get("note") or "").strip()
                    != str(pending.get("recorded_note") or "").strip())
    if not changes and not amount_changes and not note_changed:
        # Nothing to write, so nothing to copy first, and nothing to claim.
        flash("success", wording.for_project(opt, wording.formulation_unchanged(choice)))
        st.rerun()
    before = best_formulation_no(opt)
    # A correction overwrites a reading nobody can retype from memory, so the
    # project is copied first — as it is before every other destructive act.
    try:
        storage.archive(opt.project_name, "pre_correction", copy=True)
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
    if note_changed:
        opt.edit_note(index, str(pending.get("note") or "").strip())
        if not saved_ok(opt):
            return
    after = best_formulation_no(opt)
    # One sentence for the whole correction: a row can change its amounts and
    # its measurements in one save, and naming every number that moved made a
    # flash longer than the table above it.
    sentences = [wording.formulation_corrected_not_scored(
                     choice, number_list(opt.waiting_on(index)))
                 if opt.is_set_aside(index)
                 else wording.formulation_corrected(choice, waiting_text(opt, index))]
    move = _move_sentence(opt, before, after)
    if move:
        sentences.append(move)
    sentences.append(COPY_KEPT)
    scale = opt.recipe_scale(recipe)
    cautions = [c for c in (bounds_caution(opt, name, recipe[name], scale)
                            for name in amount_changes) if c]
    flash("success", wording.for_project(opt, " ".join(sentences)))
    for caution in cautions:
        flash("warning", wording.for_project(opt, caution))
    # The correction is done, so the row closes itself exactly as Close
    # closes it. Left open, its lit Save correction stayed the one coloured
    # thing on the tab and the foot had no next action to offer.
    _close_correction(opt, choice)
    st.rerun()


def _save_score(opt, storage, pending):
    """Score a formulation that was left not scored. Its row moves into the
    scored history under the number and the batch it was generated with —
    nothing is renumbered — and the model reads it from the next batch on."""
    choice, ordered, typed = (pending['choice'], pending['ordered'],
                              pending['typed'])
    for obj in ordered:
        problem = scale_error(obj, typed[obj['name']])
        if problem:
            st.error(problem)
            return
    # A measurement nobody took is left out, exactly as tab 2 leaves it out,
    # and the row is stored partial. Every box empty is not a result at all,
    # and is refused in the sentence tab 2 uses.
    results = {obj['name']: float(typed[obj['name']]) for obj in ordered
               if typed[obj['name']] is not None}
    if not results:
        st.error(wording.ENTER_A_MEASUREMENT)
        return
    before = best_formulation_no(opt)
    # The row leaves `skipped` for good, so the project is copied first — as
    # it is before every other write that cannot be retyped from memory.
    try:
        storage.archive(opt.project_name, "pre_correction", copy=True)
    except storage_backend.StorageError as e:
        st.error(str(e))
        return
    try:
        opt.score_skipped(choice, results,
                          note=str(pending.get('note') or "").strip())
    except (ValueError, TypeError) as e:
        st.error(wording.could_not_save(e))
        return
    if not saved_ok(opt):
        return
    index = opt.index_of_formulation(choice)
    sentences = _scored_then_move(choice, wording.formulation_scored(
        choice, None if index is None else waiting_text(opt, index)),
        _move_sentence(opt, before, best_formulation_no(opt)))
    # The row left `skipped` for good and a copy was kept first, so this says
    # so in the same sentence every correction ends with.
    sentences.append(COPY_KEPT)
    flash("success", wording.for_project(opt, " ".join(sentences)))
    _close_correction(opt, choice)
    st.rerun()


def _foot_label(opt):
    """What the foot of this tab offers. An open batch outranks everything:
    the work to do is the batch on the bench, and every other screen already
    says so in these words."""
    if opt.pending_batch:
        return wording.back_to_batch_label(opt.pending_batch_no, len(open_rows(opt)))
    return wording.START_NEXT_BATCH


def _review_button(opt, no, lit):
    """Review guide: after Save results, the screen's one lit button opens
    4 · Review (Set up → Make a round → Results → Review → Make a round).
    A callback, as every link is: no half-typed box on this tab is lost."""
    st.button(wording.review_round_label(no), type="primary" if lit else "secondary",
              disabled=confirmation_open(), key="review_round",
              help=wording.REVIEW_ROUND_HELP,
              on_click=ask_tab, args=(wording.TAB_INSIGHTS,))


def _foot(opt, correcting=False, typed=False, review=False):
    label = _foot_label(opt)
    # While a confirmation is armed, its "Yes" is the one coloured button and
    # answering it is the one thing to do; the next batch can wait a click.
    # An open correction row is the same case: Save correction is the lit one.
    #
    # And a project that cannot make a batch — every measurement deleted, say
    # — must not offer a lit button to a tab holding a greyed Generate.
    #
    # A waiting or Not scored formulation on screen never disables it (spec
    # 2026-09-30 §4, review I11): it stays the lit one until a value is
    # typed into that formulation's box (`typed`), then grey but enabled.
    ready, _ = readiness(opt)
    enabled = ready and not confirmation_open() and not correcting
    # Review guide: while a Round is due a Review, "Review Round N" is
    # the lit one and this waits a step, enabled.
    lit = enabled and not typed and not review
    # v3: no stop rule, so nothing here holds the next Round back.
    if st.button(label, type="primary" if lit else "secondary",
                 disabled=not enabled, key="foot_batch") and enabled:
        go_to_tab(TAB_BATCH)


def _batch_numbers(opt):
    """Every batch this project still holds, in order. A batch nobody managed
    to make counts: its formulations have numbers, and this is the only place
    they can be taken out together.

    The open batch is not offered: its rows are still on the bench, and the
    list beside this box cannot hold them."""
    seen = {int(b) for b in opt.batch_history if b is not None}
    seen |= {int(s[ROUND_FIELD]) for s in opt.skipped
             if s.get(ROUND_FIELD) is not None}
    seen.discard(opt.pending_batch_no)
    return sorted(seen)


def _formulations_of_batch(opt, batch_no):
    """Every number that batch issued and the project still holds, scored
    and not scored alike."""
    numbers = [int(n) for n, b in zip(opt.formulation_ids, opt.batch_history)
               if b is not None and int(b) == int(batch_no)]
    numbers += [int(s['formulation']) for s in opt.skipped
                if s.get(ROUND_FIELD) is not None
                and int(s[ROUND_FIELD]) == int(batch_no)]
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
    """Any number of formulations, scored or not, behind one
    confirmation.

    `Delete the last batch` was a button of its own that could only ever
    reach the last one, and it named what it would take only in the
    confirmation. `Whole batch` fills the list instead, so what is about to
    go is on screen — and can be added to or taken back out — before anything
    is confirmed."""
    numbers = _deletable_numbers(opt)
    if not numbers:
        _disarm_delete()
        st.caption(wording.for_project(opt, wording.no_formulation_to_delete_caption()))
        return
    c1, c2 = st.columns([2, 1])
    with c1:
        take_clear("delete_formulations")
        picked = st.multiselect(wording.FORMULATIONS_TO_DELETE_LABEL, numbers,
                                placeholder=wording.CHOOSE_MANY_PLACEHOLDER,
                                key="delete_formulations")
    with c2:
        batches = _batch_numbers(opt)
        take_clear("delete_whole_batch")
        batch = st.selectbox(wording.WHOLE_BATCH_LABEL, batches, index=None,
                             placeholder=wording.CHOOSE_A_BATCH_PLACEHOLDER,
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
        label = wording.delete_formulations_button(number_list(chosen),
                                                  len(chosen))
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
    park_clear("delete_formulations", [])
    # The form below this section is not what the user just deleted from.
    preserve_tab_forms()
    # A scaled table and a parsed batch sheet both name formulations that may
    # have just left the project — but the open batch's own rows are never in
    # the list above (see _deletable_numbers), so an open batch keeps both:
    # its sheet was read for formulations this delete cannot have touched.
    #
    # After preserve_tab_forms, never before: that parks the box at what it
    # is still holding, which would put the old total straight back.
    if not opt.pending_batch:
        clear_scale_total()
        st.session_state.pop("_results_upload", None)
    flash("success", wording.for_project(opt, done))
    st.rerun()


def _add_past(opt):
    """A formulation made before this project existed: typed in, or read off
    a CSV. Both land through import_formulation, so both carry no batch and
    the same note, and the CSV is no longer a section of its own that had to
    be found before past work could be entered at all."""
    mode = st.radio(wording.for_project(opt, wording.ADD_PAST_FORMULATION_LABEL),
                    [wording.TYPE_IT_IN, wording.UPLOAD_A_FILE],
                    horizontal=True, label_visibility="collapsed",
                    key="add_past_mode")
    if mode == wording.UPLOAD_A_FILE:
        _import(opt)
    else:
        _type_in_past(opt)


def _type_in_past(opt):
    if not opt.variables:
        st.caption(wording.import_columns_caption_empty())
        return
    # Every variable, held ones included: this formulation was made, and it
    # was made with some amount of each. Tab 2's own-formulation form pins a
    # held variable to the value generated formulations hold it at, because
    # that one is a new formulation under today's set — this one is a fact
    # about work already done.
    _amount_boxes(opt, _past_key, lock_worked_out=True)
    ordered = opt.measurements_by_importance()
    # A blank measurement is a partial result here, exactly as it is in the
    # results grid and in an imported CSV.
    _measurement_boxes(ordered, _past_measurement_key)
    st.session_state.setdefault("past_note", wording.IMPORTED_NOTE)
    st.text_input(wording.NOTE, key="past_note")
    # Secondary: the foot's Start the next round is this tab's coloured
    # button, and answering an armed confirmation outranks both.
    if st.button(wording.for_project(opt, wording.ADD_THIS_FORMULATION), key="add_past_formulation",
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
    scale = opt.recipe_scale(recipe)
    cautions = [c for c in (bounds_caution(opt, name, value, scale)
                            for name, value in recipe.items()) if c]
    number = int(opt.formulation_ids[-1])
    for var in opt.variables:
        park_clear(_past_key(var['name']), None)
    for obj in ordered:
        park_clear(_past_measurement_key(obj['name']), None)
    park_clear("past_note", wording.IMPORTED_NOTE)
    flash("success", wording.for_project(opt, wording.formulation_added(number)))
    for caution in cautions:
        flash("warning", wording.for_project(opt, caution))
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


def _read_past_formulations(uploaded):
    """Formulations made before this project existed, in either shape: the
    first sheet of a workbook, or a comma-separated file."""
    if str(getattr(uploaded, "name", "")).lower().endswith(".xlsx"):
        raw = pd.read_excel(uploaded, sheet_name=0, header=None)
        # The app's own export opens with a one-cell title row ("Recorded
        # amounts"); the header is the first row that names more than one
        # column. A plain sheet whose first row is the header still works.
        header_row = 0
        for i in range(min(len(raw), 5)):
            if (raw.iloc[i].notna().sum() > 1
                    and raw.iloc[i, 0] != wording.RECORDED_AMOUNTS):
                header_row = i
                break
        frame = raw.iloc[header_row + 1:].reset_index(drop=True)
        frame.columns = [str(c) for c in raw.iloc[header_row]]
        return frame.dropna(how="all")
    return pd.read_csv(uploaded)


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
        wording.UPLOAD_FORMULATIONS_FILE_LABEL, type=["xlsx", "csv"],
        # Per project: an uploader cannot be emptied from session state,
        # so a shared key offered the next project this one's file.
        key=f"import_file_{opt.project_name}")
    # The parse is behind a button, as it is on tab 2: reading the file on
    # every rerun left the sheet on screen after it had been imported, and
    # a second click on Import recorded every row twice.
    if uploaded is not None and st.button(wording.CHECK_THIS_FILE,
                                          key="check_import"):
        try:
            st.session_state["_import_rows"] = _read_past_formulations(uploaded)
        except Exception:
            st.session_state.pop("_import_rows", None)
            st.error(wording.FILE_UNREADABLE_RETRY)
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
        if all(pd.isna(row[col_for[name]]) for name in measurements):
            # Left out below (nothing measured): no caution for a row that
            # is never recorded.
            continue
        scale = opt.recipe_scale({name: _number(row[col_for[name]])
                                  for name in variables})
        for name in variables:
            caution = bounds_caution(opt, name, _number(row[col_for[name]]),
                                     scale)
            if caution:
                cautions.append(wording.row_error(position, caution))
    imported, nothing_measured, failure, reached = 0, 0, None, 0
    try:
        for position, (_, row) in enumerate(rows.iterrows(), start=1):
            reached = position
            # A blank measurement is a partial result here too, exactly as
            # it is in the results grid and in an uploaded batch sheet.
            results = {name: float(row[col_for[name]]) for name in measurements
                       if not pd.isna(row[col_for[name]])}
            if not results:
                # A not-scored formulation: `Download all formulations`
                # includes those rows, amounts and note and all, and they
                # have no result to teach the model. Left out, and counted,
                # rather than stopping a file that is otherwise importable.
                nothing_measured += 1
                continue
            # The file's own Note column, when it has one: a round trip of
            # `Download all formulations (Excel)` otherwise turned every note
            # in the project into "Made earlier". A blank cell still means
            # the row came from before this project, and says so.
            note = ""
            if "Note" in rows.columns and not pd.isna(row["Note"]):
                note = str(row["Note"]).strip()
            opt.import_formulation(
                {name: float(row[col_for[name]]) for name in variables},
                results, note=note or wording.IMPORTED_NOTE)
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
    flash("success", wording.for_project(opt, said))
    for caution in cautions:
        flash("warning", wording.for_project(opt, caution))
    st.rerun()


def _correcting(opt, pending):
    """True while the correction part holds a RECORDED formulation (not a
    Not-scored one being scored for the first time)."""
    return (pending is not None and pending.get("index") is not None
            and not any(int(r['formulation']) == int(pending.get("choice", -1))
                        for r in opt.skipped))


def _edit_past(opt, storage):
    """`Correct, delete or add past formulations`: one collapsed section for
    every way the record is fixed after the fact. Returns the open
    correction, or None — the foot above it reads that and steps aside.

    The three parts were three controls of their own, and the only way to
    enter work done before the project existed hid inside the third."""
    if not (opt.X_history or opt.skipped or opt.variables):
        return None
    # Named for all three things it holds (spec 2026-09-30 §9 I13).
    with st.expander(wording.for_project(opt, wording.EDIT_PAST_FOLD), key=EDIT_PAST_KEY,
                     on_change="rerun"):
        # The heading follows the pick: with a not-scored row picked, the
        # section is writing that row's FIRST result, and "Correct" named
        # something there was nothing of yet.
        picked = st.session_state.get("correct_formulation")
        scoring = picked is not None and any(
            int(row['formulation']) == int(picked) for row in opt.skipped)
        # The anchor "Change a measurement or an ingredient" scrolls to
        # while a correction is open (05 I8): the heading, not the boxes.
        with st.container(key=CORRECTION_SLOT):
            st.markdown(wording.for_project(opt, wording.SCORE_A_FORMULATION_HEADING
                                            if scoring
                                            else wording.CORRECT_A_FORMULATION_HEADING))
        pending = _correct(opt)
        st.divider()
        st.markdown(wording.for_project(opt, wording.DELETE_FORMULATIONS_HEADING))
        # Two things in the delete part rerun without touching the disk — the
        # `Add a whole batch to the list` pick and taking a stale confirmation
        # down — and Streamlit discards the session-state entry of every
        # widget the run did not create. Each of them calls
        # preserve_tab_forms() first so the half-typed formulation in the
        # form beneath survives (a reserved container left stale buttons).
        _delete_formulations(opt, storage)
        st.divider()
        st.markdown(wording.for_project(opt, wording.ADD_PAST_FORMULATION_HEADING))
        _add_past(opt)
    return pending


def render(opt, storage):
    if not opt.X_history and not opt.skipped:
        st.markdown(wording.NO_RESULTS_YET)
        # A project with no ingredients cannot make a batch: sending the user
        # to a tab holding a greyed Generate is a lit button to a dead end.
        # And a batch already on the bench is not a first batch to make: the
        # foot of every other screen calls it "Back to Round 1 · 3 to record".
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
        st.info(wording.for_project(opt, wording.ADD_MEASUREMENT_RESCORE_INFO))
    # Review guide: the one lit button after Save results, at the top where
    # Save results lands; drawn last, like the foot, so a confirmation armed
    # below greys it on the same run.
    # v3: the last Round's results first, the way they were measured.
    _round_results_section(opt)
    due = review_due(opt)
    review_slot = st.container() if due is not None else None
    said_partial, missing, settled = _best(opt)
    # A measurement's range too narrow, or an ingredient's amount capped too
    # low, is often exactly what a formulation on screen reveals — so the
    # one way back to Set up is right here, not several sections down. Its
    # slot is reserved now but drawn last, same as the foot below, so a
    # confirmation armed in one of the collapsed sections beneath it still
    # greys this button on the same run.
    change_setup = st.container()
    st.divider()
    _all_formulations(opt, said_partial)
    custom_records.history(opt)
    st.divider()
    # The foot keeps its place on screen but is drawn last, so it can see a
    # confirmation armed by a click in one of the collapsed sections below it
    # and step aside on the same run — otherwise the tab shows two coloured
    # buttons until the next click.
    foot = st.container()
    st.divider()
    # The Progress chart moved to Insights (0.8.0), under "Is it still improving?".
    pending = _edit_past(opt, storage)
    with change_setup:
        # Not beside a formulation still waiting or Not scored: its missing
        # value is the way on, not Set up (spec 2026-09-30 §4).
        lit = not confirmation_open()
        if settled and st.button(wording.change_setup_from_results(
                settings_only=not opt.has_ingredients()),
                                 key="change_setup_from_results", disabled=not lit) and lit:
            # 02 I8 (05; polish round): a recorded formulation is on
            # screen, so a measurement to change is one of its values. The
            # correction fold opens on it, or stays on the one already open
            # there, and the page goes to it. What is typed in the fold
            # below is kept.
            preserve_tab_forms()
            if not _correcting(opt, pending):
                number = _viewed_number(opt)
                if number is not None:
                    park_clear("correct_formulation", number)
            park_clear(EDIT_PAST_KEY, True)
            st.session_state[SCROLL_KEY] = CORRECTION_SLOT
            st.rerun()
    with foot:
        typed = missing is not None and any(
            st.session_state.get(_missing_key(missing['number'], n)) is not None
            for n in missing['names'])
        _foot(opt, pending is not None, typed, review=due is not None)
    if review_slot is not None:
        with review_slot:
            _review_button(opt, due, lit=not confirmation_open() and pending is None
                           and not typed)
    if pending is not None:
        with pending['slot']:
            _save_correction(opt, storage, pending)
    # Both rerun when clicked, so they come after every other widget on the
    # tab has been drawn: what is typed in the fold below survives.
    if missing is not None:
        with missing['slot']:
            _missing_panel(opt, missing['number'], missing['names'],
                           correcting=pending is not None)
