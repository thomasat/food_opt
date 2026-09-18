"""User-facing editing and workbook interoperability regressions."""
import io
import math

import pytest
from openpyxl import load_workbook

from food_bo import FoodOptimizer
import wording


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer('editable', robust=False)
    opt.add_ingredient('Protein', 5, 20)
    opt.add_ingredient('Water', 0, 100)
    opt.set_formulation_total(100)
    opt.set_formula('Water', '= rest')
    opt.add_objective('Taste', 1, goal='max', min_val=0, max_val=10)
    opt.set_pending_batch([{'Protein': 10, 'Water': 90}, {'Protein': 15, 'Water': 85}], batch_no=1)
    return opt


def data(book):
    out = io.BytesIO(); book.save(out); out.seek(0)
    return out


def result_row(sheet, label='Taste', col=1):
    return next(c.row for row in sheet for c in row if c.column == col and (c.value == label or str(c.value).startswith(label + " (") or str(c.value).startswith(label + " ·")))


def test_edit_recalculates_preserves_original_and_reloads(project):
    assert project.edit_pending_formulations({1: {'Protein': 12}})
    row = project.pending_batch[0]
    assert row['recipe'] == {'Protein': 12, 'Water': 88}
    assert row['original_recipe'] == {'Protein': 10, 'Water': 90}
    assert FoodOptimizer('editable').pending_batch[0] == row
    assert project.storage.list_archives()
    assert not project.edit_pending_formulations({1: {'Protein': 12}})


@pytest.mark.parametrize('changes', [{1: {'Protein': math.nan}}, {1: {'Protein': 21}},
                                    {1: {'Water': 80}}, {3: {'Protein': 10}},
                                    {1: {'Protein': 12}, 2: {'Protein': -1}}])
def test_invalid_edits_are_atomic(project, changes):
    before = project.export_json()
    with pytest.raises(ValueError):
        project.edit_pending_formulations(changes)
    assert project.export_json() == before


def test_compact_workbook_results_and_actual_import(project):
    project.set_records('actual', True)
    book = load_workbook(io.BytesIO(project.workbook_bytes(project.pending_batch, 100, print_pack=False)))
    assert book.sheetnames == ['Round overview', 'Preparation', 'Results']
    assert book.active.title == 'Round overview'
    results = book['Results']
    results.cell(result_row(results), 2, 8)
    actual = result_row(results, 'Actual amounts and settings — optional')
    results.cell(actual + 3, 2, 11)
    upload = project.results_from_workbook(data(book))
    assert upload.frame['Taste'].tolist() == [8]
    assert upload.actual == {1: {'Protein': 11}}


def test_old_workbook_after_edit_is_refused(project):
    book = load_workbook(io.BytesIO(project.workbook_bytes(project.pending_batch, 100, print_pack=False)))
    project.edit_pending_formulations({1: {'Protein': 12}})
    with pytest.raises(ValueError, match='changed after'):
        project.results_from_workbook(data(book))


def test_legacy_results_on_both_surfaces_merge_and_conflicts_fail(project):
    book = load_workbook(io.BytesIO(project.workbook_bytes(project.pending_batch, 100)))
    book['Round 1'].cell(result_row(book['Round 1']), 2, 8)
    page = book['Formulation 2']
    page.cell(result_row(page, col=2), 4, 7)
    upload = project.results_from_workbook(data(book))
    assert upload.frame['Taste'].tolist() == [8, 7]
    page = book['Formulation 1']
    page.cell(result_row(page, col=2), 4, 9)
    with pytest.raises(ValueError, match='conflicting Taste'):
        project.results_from_workbook(data(book))


def test_legacy_without_metadata_still_imports(project):
    book = load_workbook(io.BytesIO(project.workbook_bytes(project.pending_batch, 100)))
    book['Round 1']['A1'].comment = None
    book['Round 1'].cell(result_row(book['Round 1']), 2, 8)
    assert project.results_from_workbook(data(book)).frame['Taste'].tolist() == [8]


def test_compact_preparation_lots_and_records(project):
    project.add_premix('Seasoning', 'portioned')
    project.set_premix_parts('Seasoning', [{'name': 'Salt', 'share': 100, 'unit': 'g'}])
    project.add_ingredient('Seasoning', 1, 3)
    project.set_records('lot', True)
    project.set_records('actual', True)
    project.set_pending_batch([{'Protein': 10, 'Seasoning': 2, 'Water': 88}], batch_no=2)
    book = load_workbook(io.BytesIO(project.workbook_bytes(project.pending_batch, 100, print_pack=False)))
    sheet = book['Preparation']
    sheet.cell(result_row(sheet, wording.PREMIX_LOT_LABEL), 2, 'Blend-42')
    sheet.cell(result_row(sheet, wording.PREMIX_BLENDED_BY_LABEL), 2, 'Alex')
    book['Results'].cell(result_row(book['Results']), 2, 7)
    upload = project.results_from_workbook(data(book))
    assert upload.lots['Seasoning'] == 'Blend-42'
    assert any(r['value'] == 'Alex' for r in upload.bench_records)


@pytest.mark.parametrize('kind', ['Advanced burger: calculated ingredients', 'Okara fermentation'])
def test_examples_generate_feasible_rounds(tmp_path, monkeypatch, kind):
    import sample_projects
    from storage import LocalStorage
    monkeypatch.chdir(tmp_path)
    opt = sample_projects.build(kind, sample_projects.NAMES[kind], LocalStorage(), LocalStorage())
    assert not opt.X_history
    assert 'Illustrative' in opt.targets_source
    suggestions = opt.ask(3)
    assert len(suggestions) == 3
    for recipe in suggestions:
        assert opt._check_constraints(recipe)
        if kind.startswith('Advanced'):
            assert recipe['Hydration water'] == pytest.approx(2.2 * (recipe['Textured pea protein'] + recipe['Textured soy protein']))
            assert sum(v for k, v in recipe.items() if k != 'Mixing time after oils') == pytest.approx(100)
    opt.set_pending_batch(suggestions)
    book = load_workbook(io.BytesIO(opt.workbook_bytes(opt.pending_batch, opt.open_round_size(), print_pack=False)))
    assert book.sheetnames == ['Round overview', 'Preparation', 'Results']


def test_recorded_row_in_partial_round_cannot_be_edited(project):
    project.tell(project.pending_batch[0]['recipe'], {'Taste': 8}, formulation_no=1, batch_no=1)
    assert project.pending_batch
    with pytest.raises(ValueError, match='Only unrecorded'):
        project.edit_pending_formulations({1: {'Protein': 12}})
    assert project.edit_pending_formulations({2: {'Protein': 12}})
    assert project.recipe_history[0] == {'Protein': 10, 'Water': 90}


def test_editor_save_and_cancel_preserve_result_drafts(project):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    app = str(Path(__file__).resolve().parents[1] / 'app.py')
    at = AppTest.from_file(app, default_timeout=120).run()
    at.number_input(key='f1_Taste').set_value(8).run()
    next(b for b in at.button if b.label == wording.EDIT_FORMULATIONS).click().run()
    assert not at.exception
    key = 'edit_round_editable_1'
    at.session_state[key + '_grid'] = {'edited_rows': {0: {'Protein (g)': 12}}, 'added_rows': [], 'deleted_rows': []}
    at.run()
    at.session_state[key + '_grid'] = {'edited_rows': {0: {'Protein (g)': 12}}, 'added_rows': [], 'deleted_rows': []}
    at.button(key=key + '_save').click().run()
    assert not at.exception
    assert FoodOptimizer('editable').pending_batch[0]['recipe']['Protein'] == 12
    assert at.number_input(key='f1_Taste').value == 8
    at.button(key=key + '_start').click().run()
    at.button(key=key + '_cancel').click().run()
    assert not at.exception
    assert at.number_input(key='f1_Taste').value == 8


def test_actual_and_measurement_columns_align_for_every_formulation(project):
    project.set_records('actual', True)
    book = load_workbook(io.BytesIO(project.workbook_bytes(project.pending_batch, 100, print_pack=False)))
    sheet = book['Results']
    actual = result_row(sheet, 'Actual amounts and settings — optional')
    for col in range(2, sheet.max_column + 1):
        number = project._formulation_column_number(sheet.cell(actual + 2, col).value)
        if number:
            sheet.cell(result_row(sheet), col, 6 + number)
            sheet.cell(actual + 3, col, 10 + number)
    upload = project.results_from_workbook(data(book))
    assert upload.frame['Taste'].tolist() == [7, 8]
    assert upload.actual == {1: {'Protein': 11}, 2: {'Protein': 12}}


def test_compact_measurement_units_are_visible_before_entry(project):
    project.add_objective('Cook loss', 1, goal='min', min_val=0, max_val=40, unit='%')
    project.set_pending_batch([{'Protein': 10, 'Water': 90}], batch_no=2)
    book = load_workbook(io.BytesIO(project.workbook_bytes(project.pending_batch, 100, print_pack=False)))
    sheet = book['Results']
    row = result_row(sheet, 'Cook loss (%)')
    sheet.cell(row, 2, 12.5)
    sheet.cell(result_row(sheet), 2, 7)
    assert project.results_from_workbook(data(book)).frame['Cook loss'].tolist() == [12.5]


def test_excel_cached_measurement_formula_is_preserved(project):
    from zipfile import ZipFile, ZIP_DEFLATED
    book = load_workbook(io.BytesIO(project.workbook_bytes(project.pending_batch, 100, print_pack=False)))
    sheet = book['Results']
    sheet.cell(result_row(sheet), 2, '=4+4')
    saved = data(book)
    with pytest.raises(ValueError, match='no saved result'):
        project.results_from_workbook(saved)
    # Excel supplies the cached number on recalculation; openpyxl cannot
    # calculate, so emulate the standard OOXML value Excel would write.
    original = ZipFile(data(book))
    out = io.BytesIO()
    with ZipFile(out, 'w', ZIP_DEFLATED) as rewritten:
        for name in original.namelist():
            payload = original.read(name)
            if name == 'xl/worksheets/sheet3.xml':
                payload = payload.replace(b'<f>4+4</f><v></v>', b'<f>4+4</f><v>8</v>')
            rewritten.writestr(name, payload)
    out.seek(0)
    assert project.results_from_workbook(out).frame['Taste'].tolist() == [8]
