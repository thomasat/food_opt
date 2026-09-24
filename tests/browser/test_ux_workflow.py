"""The composition entry, the editable plan and the three-tab download, in
Chrome. Runs against a served copy named in APP_URL; see conftest.py."""
import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("APP_URL"),
    reason="set APP_URL to a served copy of the app")
pytest.importorskip("playwright.sync_api")
pytest.importorskip("openpyxl")

from openpyxl import load_workbook        # noqa: E402
from grid_helpers import *                # noqa: E402,F401,F403


@pytest.fixture
def page(playwright_session):
    browser, page = open_sample(playwright_session)
    yield page
    browser.close()


def test_the_calculation_help_explains_filling_to_the_total(page):
    tab(page, SET_UP_TAB)
    assert not page.get_by_text('Example project', exact=True).count()
    page.get_by_text('Calculation help', exact=True).click()
    settle(page)
    assert page.get_by_text('Fill to total', exact=False).count()
    page.keyboard.press('Escape')


def test_a_premix_composition_can_be_typed_as_amounts_instead(page):
    tab(page, SET_UP_TAB)
    open_fold(page, 'Dry blend · parts')
    parts = grid_named(page, 'Part')
    assert 'Unit' not in grid_head(page, parts)
    assert 'Composition (%)' in grid_head(page, parts)
    group = fold(page, 'Dry blend · parts')
    group.get_by_text('Enter ingredient amounts instead', exact=True).click()
    settle(page)
    parts = grid_named(page, 'Part')
    assert 'Amount (g)' in grid_head(page, parts), grid_head(page, parts)
    set_cell(page, parts, 0, 1, '60')
    fold_click(page, 'Dry blend · parts', 'Save changes')
    assert not alerts(page, 'Error'), alerts(page, 'Error')
    open_fold(page, 'Dry blend · parts')
    group.get_by_text('Enter ingredient amounts instead', exact=True).click()
    settle(page)
    parts = grid_named(page, 'Part')
    # 60 g of a 94 g blend is what the percentages come back as.
    assert abs(float(grid_rows(page, parts)[0][1]) - 60 / 94 * 100) < 0.02


def test_a_round_can_be_edited_and_downloaded_as_three_tabs(page):
    ensure_round(page)
    click(page, 'Edit formulations')
    plan = grid_named(page, 'Formulation')
    set_cell(page, plan, 0, 1, '6')
    click(page, 'Save changes')
    assert not alerts(page, 'Error'), alerts(page, 'Error')
    assert not page.locator('[data-testid="stException"]').count()
    with page.expect_download() as event:
        page.get_by_role('button', name='Download the round sheets (Excel)',
                         exact=True).click()
    path = '/tmp/foodopt-ux-round.xlsx'
    event.value.save_as(path)
    book = load_workbook(path)
    assert [s.title for s in book if s.sheet_state == 'visible'] == [
        'Round overview', 'Preparation', 'Results']
