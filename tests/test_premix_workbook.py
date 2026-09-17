"""Bench-facing pre-mix pages, totals, protection, and Excel round trips."""
import io

import pytest
from openpyxl import load_workbook

from food_bo import FoodOptimizer
import wording


@pytest.fixture
def premix_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer('pre-mix pages', robust=False)
    opt.set_amount_unit('g')
    opt.add_ingredient('Water', 0, 100)
    opt.add_objective('Taste', 1, goal='max', min_val=0, max_val=10)
    opt.add_premix('Dry blend', 'portioned')
    opt.set_premix_parts('Dry blend', [
        {'name': 'Flour', 'share': 60, 'unit': 'g', 'vendor': 'Mill'},
        {'name': 'Salt', 'share': 40, 'unit': 'g'}])
    opt.add_ingredient('Dry blend', 10, 60)
    opt.add_premix('Fat phase', 'weighed')
    opt.set_premix_parts('Fat phase', [
        {'name': 'Coconut oil', 'share': 50, 'unit': 'g'},
        {'name': 'Sunflower oil', 'share': 50, 'unit': 'g'}])
    opt.add_ingredient('Coconut oil', 0, 20)
    opt.add_ingredient('Sunflower oil', 0, 20)
    opt.set_pending_batch([
        {'Water': 60, 'Dry blend': 30, 'Coconut oil': 4, 'Sunflower oil': 6},
        {'Water': 40, 'Dry blend': 50, 'Coconut oil': 7, 'Sunflower oil': 3}], batch_no=1)
    opt._snapshot_premixes(1)
    return opt


def workbook(opt, total=100):
    return load_workbook(io.BytesIO(opt.workbook_bytes(opt.pending_batch, total)))


def row_for(sheet, label, column=1):
    return next(row for row in range(1, sheet.max_row + 1)
                if sheet.cell(row, column).value == label)


def test_portioned_page_is_first_and_names_its_round_quantity(premix_project):
    book = workbook(premix_project, 200)
    assert book.sheetnames[:2] == ['Pre-mix · Dry blend', 'Round 1']
    assert book.active['A1'].value == 'Dry blend · make 160.00 g for this round'
    sheet = book.active
    assert sheet['B4'].value == 96
    assert sheet['B5'].value == 64
    assert sheet['B6'].value == 160
    assert sheet['A6'].value == wording.TOTAL_LABEL
    assert sheet['E4'].value == 'Mill'


def test_prep_page_uses_the_rounds_original_makeup(premix_project):
    premix_project.set_premix_parts('Dry blend', [
        {'name': 'Flour', 'share': 90, 'unit': 'g'},
        {'name': 'Salt', 'share': 10, 'unit': 'g'}])
    sheet = workbook(premix_project).active
    assert sheet['B4'].value == 48
    assert sheet['B5'].value == 32


def test_portioned_blend_is_one_line_and_weighed_parts_are_indented(premix_project):
    book = workbook(premix_project)
    sheet = book['Formulation 1']
    labels = [sheet.cell(r, 2).value for r in range(1, sheet.max_row + 1)]
    assert labels.count('Dry blend') == 1
    assert 'Flour' not in labels
    group = row_for(sheet, 'Fat phase', 2)
    assert sheet.cell(group, 2).font.bold
    for name in ('Coconut oil', 'Sunflower oil'):
        r = row_for(sheet, name, 2)
        assert sheet.cell(r, 2).alignment.indent == 1
        assert not sheet.cell(r, 4).protection.locked
    total = row_for(sheet, 'Fat phase · total', 2)
    assert sheet.cell(total, 3).value == 10
    assert sheet.cell(total, 4).protection.locked


def test_summary_shopping_totals_match_the_printed_formulations(premix_project):
    sheet = workbook(premix_project, 200)['Round 1']
    r = row_for(sheet, wording.SHOPPING_TOTAL_HEADING)
    assert sheet.cell(r + 1, 2).value == 'Total for this round (g)'
    assert {sheet.cell(i, 1).value: sheet.cell(i, 2).value
            for i in range(r + 2, r + 7)} == {
        'Water': 200, 'Flour': 96, 'Salt': 64, 'Coconut oil': 22, 'Sunflower oil': 18}


def test_prep_page_only_unlocks_lot_cells(premix_project):
    sheet = workbook(premix_project).active
    assert sheet.protection.sheet
    assert {c.coordinate for row in sheet for c in row if not c.protection.locked} == {'D4', 'D5'}
    assert sheet['D4'].fill.fgColor.rgb == '00FFF2CC'


def test_long_or_invalid_names_get_unique_legal_tabs(premix_project):
    batch = premix_project.pending_batch
    for name in ('A/' + 'x' * 40, 'A?' + 'x' * 40, 'Round 1', 'x' * 20 + "'long"):
        premix_project.add_premix(name, 'portioned')
        premix_project.set_premix_parts(name, [{'name': 'Flour', 'share': 100, 'unit': 'g'}])
    premix_project.set_pending_batch(batch, batch_no=1)
    book = workbook(premix_project)
    assert len({n.lower() for n in book.sheetnames}) == len(book.sheetnames)
    assert all(len(n) <= 31 and not any(c in n for c in '[]:*?/\\') for n in book.sheetnames)
    assert all(not n.startswith("'") and not n.endswith("'") for n in book.sheetnames)
    assert 'Round 1' in book.sheetnames and 'Formulation 1' in book.sheetnames


def test_setup_sheet_names_groups_parts_and_modes(premix_project):
    sheet = load_workbook(io.BytesIO(premix_project.all_formulations_workbook()))[wording.SET_UP_SHEET]
    r = row_for(sheet, wording.PREMIXES_HEADING)
    values = {c.value for row in sheet.iter_rows(min_row=r) for c in row if c.value is not None}
    assert {'Dry blend', 'Fat phase', 'Flour', 'Salt', 'Coconut oil', 'Sunflower oil',
            wording.PREMIX_MADE_AS_PORTIONED, wording.PREMIX_MADE_AS_WEIGHED} <= values
    first_group = row_for(sheet, 'Fat phase')
    assert sheet.cell(first_group, 5).value is None


def test_lots_and_actual_part_amounts_survive_workbook_upload(premix_project):
    book = workbook(premix_project)
    book.active['D4'] = 'F-123'
    page = book['Formulation 1']
    page.cell(row_for(page, 'Coconut oil', 2), 4, 4.5)
    page.cell(row_for(page, 'Taste', 2), 4, 8)
    source = io.BytesIO()
    book.save(source)
    source.seek(0)
    upload = premix_project.results_from_workbook(source)
    assert upload.lots == {'Flour': 'F-123'}
    assert upload.actual == {1: {'Coconut oil': 4.5}}
    assert upload.frame['Taste'].tolist() == [8]


def test_an_empty_premix_cannot_generate_a_round(premix_project):
    premix_project.add_premix('New pre-mix', 'portioned')
    from ui_helpers import readiness
    ready, message = readiness(premix_project)
    assert not ready and 'New pre-mix' in message
    with pytest.raises(ValueError, match='New pre-mix'):
        premix_project.ask(1)


def test_regenerating_a_recorded_round_preserves_its_makeup(premix_project, monkeypatch):
    opt = premix_project
    recipe = dict(opt.pending_batch[0])
    opt.tell(recipe, {'Taste': 8}, formulation_no=1, batch_no=1)
    opt.set_premix_parts('Dry blend', [
        {'name': 'Flour', 'share': 90, 'unit': 'g'},
        {'name': 'Salt', 'share': 10, 'unit': 'g'}])
    observed = []
    def generate(*args):
        observed.append(opt._property_parts('Dry blend')[0]['share'])
        return [recipe]
    monkeypatch.setattr(opt, '_ask_cold_start', generate)
    opt.set_pending_batch(None)
    opt.ask(1, batch_no=1)
    assert observed == [60]
    assert opt.premix_version_of('Dry blend', 0)[0]['share'] == 60
    opt.set_pending_batch(None)
    opt.ask(1)
    assert observed == [60, 90]
    assert opt.premix_version_of('Dry blend', 0)[0]['share'] == 60


def test_switching_a_weighed_group_does_not_count_old_parts_in_rule_preview(
        premix_project):
    opt = premix_project
    opt.set_pending_batch(None)
    opt.add_ingredient('Coconut oil', 3, 10)
    opt.add_ingredient('Sunflower oil', 3, 10)
    opt.add_ingredient('Dry blend', 1, 2)
    opt.set_formulation_total(10)
    opt.set_formula('Water', '= rest')
    frame = opt.ingredient_grid_frame()
    row = frame.index[frame[wording.NAME_LABEL] == 'Fat phase'][0]
    frame.loc[row, wording.MADE_AS_LABEL] = wording.PREMIX_MADE_AS_PORTIONED
    errors, _ = opt.apply_ingredient_grid(frame)
    assert errors == []
    assert set(opt._by_name()) == {'Water', 'Dry blend', 'Fat phase'}
