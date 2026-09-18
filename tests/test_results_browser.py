"""Results browsing shows recorded amounts and meaningful ingredient percentages."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from food_bo import FoodOptimizer
from ui_results import _amount_rows
import wording

APP = str(Path(__file__).resolve().parents[1] / 'app.py')


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer('details', robust=False)
    opt.add_ingredient('Protein', 0, 100, unit='g')
    opt.add_ingredient('Water', 0, 100, unit='g')
    opt.add_process_parameter('Mixing time', 0, 120, unit='s')
    opt.add_objective('Taste', 1, goal='max', min_val=0, max_val=10)
    opt.tell({'Protein': 20, 'Water': 80, 'Mixing time': 0}, {'Taste': 4}, formulation_no=1, batch_no=1)
    opt.tell({'Protein': 30, 'Water': 70, 'Mixing time': 100}, {'Taste': 8}, formulation_no=2, batch_no=1)
    opt.record_skipped(3, 1, {'Protein': 40, 'Water': 60, 'Mixing time': 60}, note='Panel unavailable')
    return opt


def amounts(at):
    return next(t.value for t in at.tabs[2].table if wording.AMOUNT_COLUMN in t.value.columns)


def test_best_other_and_unscored_details_are_read_only(project):
    at = AppTest.from_file(APP, default_timeout=120).run()
    before = FoodOptimizer('details').export_json()
    picker = at.selectbox(key='result_view_details')
    assert picker.options == ['Best so far · Formulation 2 (Round 1)',
                              'Formulation 1 (Round 1)', 'Formulation 3 (Round 1) · Not scored']
    assert any(h.value == wording.best_so_far_heading(2, 1) for h in at.subheader)
    assert amounts(at)[wording.RESULT_PERCENT_COLUMN].tolist() == ['30.00', '70.00', '']
    picker.select(1).run()
    assert not at.exception
    assert any(h.value == 'Formulation 1 (Round 1)' for h in at.subheader)
    assert not any(h.value.startswith('Best so far:') for h in at.subheader)
    assert amounts(at)[wording.RESULT_PERCENT_COLUMN].tolist() == ['20.00', '80.00', '']
    assert amounts(at)[wording.AMOUNT_COLUMN].tolist() == ['20.00 g', '80.00 g', '0 s']
    assert float(next(d.value for d in at.tabs[2].dataframe if 'Measured' in d.value.columns)['Measured'].iloc[0]) == 4
    at.selectbox(key='result_view_details').select(3).run()
    assert any(i.value == wording.RESULT_UNSCORED_HELP for i in at.info)
    assert amounts(at)[wording.RESULT_PERCENT_COLUMN].tolist() == ['40.00', '60.00', '']
    assert not any('Measured' in d.value.columns for d in at.tabs[2].dataframe)
    at.selectbox(key='result_view_details').select(None).run()
    assert any(h.value == wording.best_so_far_heading(2, 1) for h in at.subheader)
    assert FoodOptimizer('details').export_json() == before


def test_percentages_use_displayed_recorded_size(project):
    project._batch_totals()[1] = 200
    project.save()
    at = AppTest.from_file(APP, default_timeout=120).run()
    frame = amounts(at)
    assert frame[wording.AMOUNT_COLUMN].tolist() == ['60.00 g', '140.00 g', '100 s']
    assert frame[wording.RESULT_PERCENT_COLUMN].tolist() == ['30.00', '70.00', '']


def test_best_default_follows_new_results_and_deleted_selection_recovers(project):
    at = AppTest.from_file(APP, default_timeout=120).run()
    opt = at.session_state['optimizer']
    opt.tell({'Protein': 50, 'Water': 50, 'Mixing time': 90}, {'Taste': 9}, formulation_no=4, batch_no=2)
    at.run()
    assert any(h.value == wording.best_so_far_heading(4, 2) for h in at.subheader)
    at.selectbox(key='result_view_details').select(3).run()
    opt = at.session_state['optimizer']
    opt.skipped = []
    opt.save()
    at.run()
    assert not at.exception
    assert any(h.value == wording.best_so_far_heading(4, 2) for h in at.subheader)


def test_mixed_units_do_not_create_meaningless_percentages(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer('mixed', robust=False)
    opt.add_ingredient('Powder', 0, 100, unit='g')
    opt.add_ingredient('Liquid', 0, 100, unit='ml')
    opt.add_objective('Taste', 1, goal='max', min_val=0, max_val=10)
    opt.tell({'Powder': 20, 'Liquid': 80}, {'Taste': 5}, formulation_no=1, batch_no=1)
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert amounts(at)[wording.RESULT_PERCENT_COLUMN].tolist() == ['—', '—']
    assert any(c.value == wording.RESULT_PERCENT_MIXED_HELP for c in at.caption)


def test_process_only_and_unscored_only_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer('process', robust=False)
    opt.add_process_parameter('Temperature', 20, 50, unit='°C')
    opt.add_objective('Yield', 1, goal='max', min_val=0, max_val=100)
    opt.record_skipped(1, 1, {'Temperature': 30})
    opt.record_skipped(2, 1, {'Temperature': 40})
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert at.selectbox(key='result_view_process').label == 'View trial'
    assert wording.RESULT_PERCENT_COLUMN not in amounts(at)
    assert amounts(at)[wording.AMOUNT_COLUMN].tolist() == ['30 °C']
    at.selectbox(key='result_view_process').select(2).run()
    assert amounts(at)[wording.AMOUNT_COLUMN].tolist() == ['40 °C']
    assert not any(h.value.startswith('Best so far:') for h in at.subheader)


def test_premix_percentage_counts_blend_once_and_zero_total_has_no_percentage(project):
    project.add_premix('Seasoning', 'portioned')
    project.set_premix_parts('Seasoning', [{'name': 'Salt', 'share': 60, 'unit': 'g'},
                                        {'name': 'Spice', 'share': 40, 'unit': 'g'}])
    project.add_ingredient('Seasoning', 0, 10, unit='g')
    rows = _amount_rows(project, {'Protein': 20, 'Water': 75, 'Seasoning': 5, 'Mixing time': 100})
    by_name = {row[wording.INGREDIENT_OR_SETTING_LABEL]: row for row in rows}
    assert by_name['Seasoning'][wording.RESULT_PERCENT_COLUMN] == '5.00'
    assert 'Salt' not in by_name and 'Spice' not in by_name
    assert sum(float(row[wording.RESULT_PERCENT_COLUMN]) for row in rows if row[wording.RESULT_PERCENT_COLUMN]) == 100
    assert _amount_rows(project, {'Protein': 0, 'Water': 0, 'Mixing time': 0}) == [
        {wording.INGREDIENT_OR_SETTING_LABEL: 'Mixing time', wording.AMOUNT_COLUMN: '0 s', wording.RESULT_PERCENT_COLUMN: ''}]
