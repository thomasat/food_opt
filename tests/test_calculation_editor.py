"""Calculation editing preserves canonical storage and transaction safeguards."""
import io
import pandas as pd
import pytest
from food_bo import FoodOptimizer
import wording
from calculation_editor import (display_frame, canonical_frame, canonical_calculation,
                                display_calculation, _validate)

@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer('calculations', robust=False)
    opt.add_ingredient('Flour', 10, 20, unit='g')
    opt.add_ingredient('Starch', 5, 10, unit='g')
    opt.add_ingredient('Water', 0, 100, unit='g')
    opt.set_formulation_total(100)
    return opt


def test_compact_display_preserves_original_expression_for_rename(project):
    project.set_formula('Water', '=  2 * (Flour + Starch)')
    original = project.ingredient_grid_frame()
    shown = display_frame(original)
    assert shown.iloc[2][wording.FORMULA_LABEL] == '2 * (Flour + Starch)'
    shown.at[1, wording.NAME_LABEL] = 'Oat flour'
    canonical = canonical_frame(shown, original)
    assert canonical.iloc[2][wording.FORMULA_LABEL] == '=  2 * (Flour + Starch)'
    errors, _ = project.apply_ingredient_grid(canonical)
    assert not errors
    assert 'Oat flour' in project._var_by_name('Water')['formula']


def test_fill_alias_and_legacy_expression_roundtrip(project):
    for text in ('Fill to total', 'Fill to batch size', '= rest', '=REST'):
        assert display_calculation(canonical_calculation(text)) == 'Fill to total'
    project.set_formula('Water', '= rest')
    original = project.ingredient_grid_frame()
    pd.testing.assert_frame_equal(canonical_frame(display_frame(original), original), original)


@pytest.mark.parametrize('expression', ['2 * (Flour + Starch)', '5% of batch size',
                                        '10% of Flour', 'Flour / 2', 'Flour + Starch'])
def test_supported_expressions_validate_without_writing(project, expression):
    before = project.export_json()
    frame = project.ingredient_grid_frame()
    frame.at[3, wording.FORMULA_LABEL] = canonical_calculation(expression)
    errors, plan = _validate(project, frame)
    assert not errors and plan is not None
    assert project.export_json() == before


@pytest.mark.parametrize('expression', ['Flour * Starch', 'Flour / Starch', 'Flour / 0',
                                        'SUM(Flour)', 'Water + 1', 'Unknown + 1'])
def test_invalid_expressions_leave_project_unchanged(project, expression):
    before = project.export_json()
    frame = project.ingredient_grid_frame()
    frame.at[3, wording.FORMULA_LABEL] = canonical_calculation(expression)
    errors, _ = _validate(project, frame)
    assert errors
    assert project.export_json() == before


def test_cycles_and_two_fill_rows_refused(project):
    frame = project.ingredient_grid_frame()
    for a, b in [('= Water',' = Flour'),('= rest','= rest')]:
        frame.at[1, wording.FORMULA_LABEL] = a.strip()
        frame.at[3, wording.FORMULA_LABEL] = b.strip()
        assert _validate(project, frame)[0]


@pytest.mark.parametrize('header',['Rule','Formula','Calculation'])
def test_old_and_new_import_headers(project, header):
    csv = io.StringIO(f'Name,Lowest,Highest,Unit,{header}\nProtein,10,20,g,\nWater,0,100,g,= rest\n')
    project.load_ingredients_from_csv(pd.read_csv(csv))
    assert project._var_by_name('Water')['formula'] == '= rest'


def test_dialog_stages_fill_without_writing(project):
    from streamlit.testing.v1 import AppTest
    from ui_helpers import ING_GRID_KEY, parked_grid_key
    def render():
        from calculation_editor import open_editor
        from food_bo import FoodOptimizer
        opt = FoodOptimizer('calculations')
        open_editor(opt, opt.ingredient_grid_frame())
    at = AppTest.from_function(render, default_timeout=120).run()
    at.selectbox(key='calculation_row_calculations').select(3).run()
    next(r for r in at.radio if r.label=='How is this amount determined?').set_value('Fill to total').run()
    next(b for b in at.button if b.label=='Use fill to total').click().run()
    assert not at.exception
    assert not FoodOptimizer('calculations')._var_by_name('Water').get('formula')
    staged = at.session_state[parked_grid_key(ING_GRID_KEY)][1]
    assert staged.at[3, wording.FORMULA_LABEL] == '= rest'
    errors, _ = project.apply_ingredient_grid(staged)
    assert not errors
    assert project._var_by_name('Water')['formula'] == '= rest'


def test_use_calculation_writes_the_row_and_lights_save_changes(project):
    """`Use calculation` closed the panel and changed nothing, with nothing
    said: the component's own trigger reruns the app, and the dialog was
    drawn only where the button that opens it is pressed, so everything the
    panel said about the calculation — the refusal included — went with it.

    The apply is a function now, and what it hands back is what the grid is
    staged with."""
    from calculation_editor import apply_calculation
    frame = project.ingredient_grid_frame()
    index = frame.index[frame[wording.NAME_LABEL] == 'Water'][0]
    errors, candidate = apply_calculation(project, frame, index,
                                          '2 * (Flour + Starch)')
    assert errors == []
    assert candidate.at[index, wording.FORMULA_LABEL] == '= 2 * (Flour + Starch)'
    # Nothing is written until the grid's own Save changes is pressed.
    assert not project._var_by_name('Water').get('formula')
    # ...and a calculation the project cannot take comes back as the
    # refusal, with nothing staged.
    errors, candidate = apply_calculation(project, frame, index,
                                          'Flour * Starch')
    assert errors and candidate is None


def test_the_editor_stays_open_while_a_calculation_is_being_written(project):
    """Every keystroke in the expression box reruns the app. The dialog was
    rendered only inside `if st.button(...)`, so the first rerun closed it
    under the reader's hand."""
    from streamlit.testing.v1 import AppTest
    from calculation_editor import EDITOR_OPEN
    def render():
        import streamlit as st
        import calculation_editor
        from food_bo import FoodOptimizer
        key = calculation_editor.EDITOR_OPEN
        opt = FoodOptimizer('calculations')
        if st.button('Edit calculation', key='edit_calculation'):
            st.session_state[key] = True
        if st.session_state.get(key):
            calculation_editor.open_editor(opt, opt.ingredient_grid_frame())
    at = AppTest.from_function(render, default_timeout=120).run()
    at.button(key='edit_calculation').click().run()
    assert not at.exception, at.exception
    assert EDITOR_OPEN in at.session_state.filtered_state
    assert any(r.label == wording.CALCULATION_MODE_QUESTION for r in at.radio)
    # A rerun that is not a click on that button still finds the panel open.
    at.run()
    assert any(r.label == wording.CALCULATION_MODE_QUESTION for r in at.radio)
