"""Compact bench workbook and compatibility bridge to the original workbook reader.

Import metadata is stored on a very hidden worksheet, outside the working tabs.
Legacy cell comments are used only as an in-memory compatibility bridge for
existing readers and older workbooks; new downloads contain no machine comments.
"""
import io
import json
from copy import copy

from openpyxl import Workbook, load_workbook
from openpyxl.comments import Comment
from openpyxl.styles import Font, PatternFill, Protection, Alignment, Side
from openpyxl.worksheet.hyperlink import Hyperlink
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.cell_range import CellRange

import wording

MARKER = 'food-opt-workbook-v1:'
PREP = 'food-opt-preparation:'
OVERVIEW = wording.OVERVIEW_SHEET
RESULTS = wording.RESULTS_SHEET
PREPARATION = wording.PREPARATION_SHEET


METADATA_SHEET = '_FoodOptimizer'
METADATA_VERSION = 'food-opt-metadata-v1'
_METADATA_PREFIXES = (MARKER, PREP, 'food-opt-custom-record:')
_METADATA_ERROR = wording.WORKBOOK_METADATA_ERROR


def hide_metadata(book):
    """Move internal comments into chunked text, keeping all working tabs clean.

    Chunks stay below Excel's cell text limit, even for large rounds. This sheet
    is not available through Excel's normal Unhide command. User comments stay.
    """
    if METADATA_SHEET in book.sheetnames:
        raise ValueError(_METADATA_ERROR)
    records = []
    for sheet in book:
        for row in sheet:
            for cell in row:
                if cell.comment and cell.comment.text.startswith(_METADATA_PREFIXES):
                    records.append([sheet.title, cell.coordinate, cell.comment.text])
                    cell.comment = None
    if not records:
        return
    payload = json.dumps(records, ensure_ascii=True)
    metadata = book.create_sheet(METADATA_SHEET)
    metadata['A1'] = METADATA_VERSION
    for index, offset in enumerate(range(0, len(payload), 20000), 2):
        cell = metadata.cell(index, 1, payload[offset:offset + 20000])
        cell.data_type = 's'
    metadata.sheet_state = 'veryHidden'
    metadata.protection.sheet = True


def restore_metadata(book):
    """Restore markers in memory for import; accept legacy comment-only files."""
    if METADATA_SHEET not in book.sheetnames:
        return
    metadata = book[METADATA_SHEET]
    try:
        if metadata['A1'].value != METADATA_VERSION:
            raise ValueError
        records = json.loads(''.join(metadata.cell(r, 1).value
                                    for r in range(2, metadata.max_row + 1)))
        if not isinstance(records, list) or not records:
            raise ValueError
        seen = set()
        for record in records:
            if (not isinstance(record, list) or len(record) != 3
                    or not all(isinstance(v, str) for v in record)):
                raise ValueError
            title, address, text = record
            if (title == METADATA_SHEET or title not in book.sheetnames
                    or (title, address) in seen
                    or not text.startswith(_METADATA_PREFIXES)):
                raise ValueError
            cell = book[title][address]
            if cell.coordinate != address:
                raise ValueError
            cell.comment = Comment(text, 'Food Opt')
            seen.add((title, address))
    except (ValueError, TypeError, KeyError, AttributeError):
        raise ValueError(_METADATA_ERROR) from None
    book.remove(metadata)


def sheet_link(cell, title):
    """A workbook-internal destination, not an external file URL."""
    cell.hyperlink = Hyperlink(ref=cell.coordinate, location=f"'{title}'!A1")
    cell.font = Font(color='0563C1', underline='single')


def separate_formulations(sheet, columns, last_row, paired=False):
    """Keep amounts and percentages together, with a strong boundary per formulation.

    Borders survive monochrome printing; shaded headers provide a second cue.
    Do not touch merged method banners or overwrite input-cell shading.
    """
    divider = Side(style='medium', color='596C80')
    for index, column in enumerate(columns):
        for row in range(3, last_row + 1):
            cell = sheet.cell(row, column)
            if cell.__class__.__name__ == 'MergedCell':
                continue
            if any(region.min_row <= row <= region.max_row and
                   region.min_col <= column <= region.max_col
                   for region in sheet.merged_cells.ranges):
                continue
            border = copy(cell.border)
            border.left = divider
            cell.border = border
            if str(cell.value or '').startswith(wording.FORMULATION_CAP + ' '):
                cell.fill = PatternFill('solid', fgColor=('DDEBF7' if index % 2 == 0 else 'E2EFDA'))
                cell.font = Font(bold=True, color='17365D')
                cell.alignment = Alignment(horizontal='center', wrap_text=True)
                sheet.row_dimensions[row].height = max(28, sheet.row_dimensions[row].height or 0)
                if paired:
                    sheet.cell(row, column + 1).fill = copy(cell.fill)


def copy_cells(source, target, first=1, last=None, start=1):
    for row in source.iter_rows(min_row=first, max_row=last or source.max_row):
        for cell in row:
            if cell.__class__.__name__ == 'MergedCell':
                continue
            dest = target.cell(start + cell.row - first, cell.column)
            dest.value = cell.value
            dest.data_type = cell.data_type
            dest.font = copy(cell.font)
            dest.fill = copy(cell.fill)
            dest.border = copy(cell.border)
            dest.alignment = copy(cell.alignment)
            dest.protection = copy(cell.protection)
            dest.number_format = cell.number_format
            dest.comment = copy(cell.comment)
    end = last or source.max_row
    for merged in source.merged_cells.ranges:
        if merged.min_row >= first and merged.max_row <= end:
            shifted = CellRange(str(merged))
            shifted.shift(row_shift=start - first)
            target.merge_cells(str(shifted))
    for row_number in range(first, end + 1):
        target.row_dimensions[start + row_number - first].height = source.row_dimensions[row_number].height
    for column, dimension in source.column_dimensions.items():
        target.column_dimensions[column].width = dimension.width


def stamp(book, opt, rows, total):
    sheet = book[wording.batch_sheet_name(opt.pending_batch_no)]
    data = {'round': opt.pending_batch_no,
            'recipes': {str(row['formulation']):
                        {n: round(float(v), 8) for n, v in opt.shown_recipe(row, total)[0].items()}
                        for row in rows}}
    sheet['A1'].comment = Comment(MARKER + json.dumps(data), 'Food Opt')


def compact(book, opt, rows, note=""):
    source = book[wording.batch_sheet_name(opt.pending_batch_no)]
    split = next(c.row for row in source for c in row
                 if c.column == 1 and c.value == wording.MEASUREMENTS_SHEET_HEADING)
    result = Workbook()
    overview = result.active
    overview.title = OVERVIEW
    copy_cells(source, overview, last=split - 1)
    navigation = wording.SHEET_NAVIGATION
    # The line that says these amounts were re-sized survives the fold: it
    # is the first thing the bench needs and the navigation replaces the
    # banner it was written into.
    overview['A2'] = ((note + ' ') if note else '') + navigation + wording.OVERVIEW_SHEET_INTRO
    overview['A2'].alignment = Alignment(wrap_text=True)
    overview.row_dimensions[2].height = 45
    for row in overview:
        for cell in row:
            if cell.value == wording.PREMIX_LOT_ON_ITS_PAGE:
                cell.value = wording.PREMIX_LOT_ON_PREPARATION
                sheet_link(cell, PREPARATION)
    preparation = result.create_sheet(PREPARATION)
    preparation['A1'] = wording.PREPARATION_SHEET_TITLE
    preparation['A2'] = navigation + wording.PREPARATION_SHEET_INTRO
    start = 4
    for title in opt._premix_sheet_names([r['formulation'] for r in rows]).values():
        page = book[title]
        copy_cells(page, preparation, start=start)
        preparation.cell(start, 1).comment = Comment(PREP + title, 'Food Opt')
        start += page.max_row + 3
    if start == 4:
        preparation['A4'] = wording.PREPARATION_SHEET_EMPTY
    measured = result.create_sheet(RESULTS)
    measured['A1'] = wording.RESULTS_SHEET_TITLE
    measured['A2'] = navigation + wording.RESULTS_SHEET_INTRO
    copy_cells(source, measured, first=split, start=4)
    # The unit is in the label the print pack already writes
    # (food_bo.entry_label), so the compact sheet has nothing to patch.
    formulation_columns = [c.column for c in source[3]
                           if opt._formulation_column_number(c.value) is not None]
    # Keep measurement and actual columns aligned; hide percentage spacer
    # columns from the overview so Results reads as one column per formulation.
    for column in range(2, measured.max_column + 1):
        if column not in formulation_columns:
            measured.column_dimensions[get_column_letter(column)].hidden = True
    # The Results sheet is the only measurement entry surface. Actual values
    # use one additional matrix on this same sheet, rather than N extra tabs.
    if opt.records('actual'):
        start = measured.max_row + 3
        measured.cell(start, 1, wording.ACTUAL_SHEET_HEADING)
        measured.cell(start + 1, 1, wording.ACTUAL_SHEET_HELP)
        measured.merge_cells(start_row=start + 1, start_column=1, end_row=start + 1, end_column=max(formulation_columns or [2]))
        measured.cell(start + 1, 1).alignment = Alignment(wrap_text=True)
        measured.row_dimensions[start + 1].height = 30
        measured.cell(start + 2, 1, wording.INGREDIENT_OR_SETTING_COLUMN)
        for j, row in zip(formulation_columns, rows):
            measured.cell(start + 2, j, wording.formulation_sheet_name(row['formulation']))
        for i, var in enumerate(opt.variables, start + 3):
            measured.cell(i, 1, opt._amount_column(var['name'], mark=True))
            for j in formulation_columns:
                cell = measured.cell(i, j)
                cell.protection = Protection(locked=False)
                cell.fill = PatternFill('solid', fgColor='FFF2CC')
                cell.number_format = '0.00'
    # The overview groups an amount and a percentage under each formulation.
    # Results has one visible column per formulation, including actual amounts.
    # End at the process settings, before method and round-total sections.
    plan_end = (3 + len(list(opt._formulation_ingredient_lines()))
                + int(opt.has_ingredients()) + len(opt._process_settings()))
    separate_formulations(overview, formulation_columns, plan_end, paired=opt._shows_shares())
    separate_formulations(measured, formulation_columns, measured.max_row)
    for sheet in result:
        linkrow = sheet.max_row + 3
        for c, title in enumerate((OVERVIEW, PREPARATION, RESULTS), 1):
            # Results hides percentage columns: never put navigation in one.
            column = 1 if c == 1 else (2 * c - 2 if sheet == measured else c)
            cell = sheet.cell(linkrow, column, title)
            sheet_link(cell, title)
        sheet.cell(linkrow + 1, 1, wording.SHEET_LINK_FALLBACK)
        sheet['A1'].font = Font(size=15, bold=True)
        sheet.column_dimensions['A'].width = 58
        sheet.row_dimensions[2].height = 65
        if sheet.title != OVERVIEW:
            sheet.merge_cells(start_row=2, start_column=1, end_row=2, end_column=max(3, sheet.max_column))
        sheet['A2'].alignment = Alignment(wrap_text=True)
        sheet.freeze_panes = 'B4'
        sheet.sheet_view.showGridLines = False
        sheet.protection.sheet = True
        sheet.protection.selectLockedCells = False
        sheet.protection.selectUnlockedCells = False
        sheet.sheet_properties.pageSetUpPr.fitToPage = True
        sheet.page_setup.orientation = 'landscape'
        sheet.page_setup.paperSize = sheet.PAPERSIZE_A4
        sheet.page_setup.fitToWidth = 1
        sheet.page_setup.fitToHeight = 0
        for c in range(2, sheet.max_column + 1):
            sheet.column_dimensions[get_column_letter(c)].width = max(18, sheet.column_dimensions[get_column_letter(c)].width or 0)
    if not opt.has_ingredients() and opt._process_settings():
        for sheet in result:
            for address in ('A2',):
                sheet[address] = wording.for_project(opt, sheet[address].value)
            for row in sheet:
                for cell in row:
                    number = opt._formulation_column_number(cell.value)
                    if (number is not None and cell.column in formulation_columns
                            and cell.fill.fgColor.rgb in ('00DDEBF7', '00E2EFDA')):
                        cell.value = f"{wording.TRIAL_CAP} {number}"
                    elif cell.value == wording.INGREDIENT_OR_SETTING_LABEL:
                        cell.value = wording.PROCESS_SETTINGS_HEADER
    result.active = 0
    return result


def prepare_import(source, opt, batch_no):
    """Verify issued plans, then expand compact sheets for the legacy reader."""
    if hasattr(source, 'seek'):
        source.seek(0)
    book = load_workbook(source, data_only=False)
    restore_metadata(book)
    formula_cells = [(sheet.title, cell.coordinate) for sheet in book for row in sheet
                     for cell in row if cell.data_type == 'f']
    if formula_cells:
        if hasattr(source, 'seek'):
            source.seek(0)
        cached = load_workbook(source, data_only=True)
        for title, address in formula_cells:
            cell = book[title][address]
            value = cached[title][address].value
            if value is None and not cell.protection.locked:
                raise ValueError(wording.WORKBOOK_FORMULA_ERROR)
            cell.value = value
    marker = next((cell.comment.text[len(MARKER):] for sheet in book
                   for cell in [sheet['A1']] if cell.comment and cell.comment.text.startswith(MARKER)), None)
    if marker:
        issued = json.loads(marker)
        if issued['round'] != batch_no:
            raise ValueError(wording.WORKBOOK_WRONG_ROUND)
        for row in opt.pending_batch or []:
            old = issued['recipes'].get(str(row['formulation']))
            if old is not None:
                current = {n: round(float(v), 8) for n, v in opt.shown_recipe(row, opt.open_round_size())[0].items()}
                if old != current:
                    raise ValueError(wording.WORKBOOK_ROUND_CHANGED)
    if OVERVIEW not in book.sheetnames:
        out = io.BytesIO(); book.save(out); out.seek(0)
        return out
    if not marker or RESULTS not in book.sheetnames or PREPARATION not in book.sheetnames:
        raise ValueError(wording.WORKBOOK_MISSING_SHEET)
    original = Workbook()
    summary = original.active
    summary.title = wording.batch_sheet_name(batch_no)
    copy_cells(book[OVERVIEW], summary)
    measurements = book[RESULTS]
    if any(c.data_type == 'f' for row in measurements for c in row if not c.protection.locked):
        raise ValueError(wording.WORKBOOK_RESULTS_ARE_NUMBERS)
    # Copy only the measurement block, leaving the optional Actual matrix out.
    actual = next((c.row for row in measurements for c in row
                   if c.column == 1 and c.value == wording.ACTUAL_SHEET_HEADING), None)
    end = actual - 1 if actual else measurements.max_row
    copy_cells(measurements, summary, first=4, last=end, start=summary.max_row + 2)
    if actual:
        for cell in measurements[actual + 2][1:]:
            number = opt._formulation_column_number(cell.value)
            if number is None:
                continue
            page = original.create_sheet(wording.formulation_sheet_name(number))
            page.cell(1, 3, opt._actual_column_head())
            for i, var in enumerate(opt.variables, 2):
                label = opt._amount_column(var['name'], mark=True)
                source_row = next((r for r in range(actual + 3, measurements.max_row + 1)
                                   if measurements.cell(r, 1).value in (label, label.replace(f" · {wording.WORKED_OUT}", f" · {wording.OLD_CALCULATED_LABEL}"))), None)
                if source_row:
                    page.cell(i, 2, label)
                    value = measurements.cell(source_row, cell.column).value
                    if value is not None:
                        if measurements.cell(source_row, cell.column).data_type == 'f':
                            raise ValueError(wording.WORKBOOK_ACTUALS_ARE_NUMBERS)
                        page.cell(i, 3, value)
    prep = book[PREPARATION]
    starts = [(c.row, c.comment.text[len(PREP):]) for row in prep for c in row
              if c.comment and c.comment.text.startswith(PREP)]
    for i, (start, title) in enumerate(starts):
        end = starts[i + 1][0] - 1 if i + 1 < len(starts) else prep.max_row
        copy_cells(prep, original.create_sheet(title), first=start, last=end)
    out = io.BytesIO(); original.save(out); out.seek(0)
    return out
