"""Unfinished bench measurements survive storage without becoming observations."""
import copy

import pytest

from food_bo import FoodOptimizer


@pytest.fixture
def draft_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer('drafts')
    opt.add_ingredient('Water', 0, 100)
    opt.add_objective('Taste', 1, goal='max')
    opt.set_pending_batch([{'Water': 50}])
    opt.save_result_drafts({1: {'results': {'Taste': 7}, 'note': 'Bench A',
                               'not_scored': False}})
    return opt


def test_drafts_roundtrip_without_becoming_observations(draft_project):
    opt = FoodOptimizer('drafts')
    assert opt.result_drafts == draft_project.result_drafts
    assert opt.formulation_ids == []
    assert FoodOptimizer.validate_state(opt.export_json())
    opt.set_pending_batch(None)
    assert FoodOptimizer('drafts').result_drafts == {}


@pytest.mark.parametrize('invalid', [None, [], {'no': {}},
    {'1': {'results': {'Taste': 'seven'}, 'note': '', 'not_scored': False}},
    {'1': {'results': {'Taste': float('nan')}, 'note': '', 'not_scored': False}},
    {'1': {'results': {}, 'note': [], 'not_scored': False}},
    {'99': {'results': {'Taste': 7}, 'note': '', 'not_scored': False}}])
def test_bad_drafts_are_refused_before_import(draft_project, invalid):
    state = copy.deepcopy(draft_project.export_json())
    state['result_drafts'] = invalid
    with pytest.raises(ValueError):
        FoodOptimizer.validate_state(state)


def test_old_projects_start_without_drafts(draft_project):
    state = draft_project.export_json()
    state.pop('result_drafts')
    state['CLASS_VERSION'] = 13
    restored = FoodOptimizer('legacy')
    restored.import_json(state)
    assert restored.result_drafts == {}


def test_renaming_a_measurement_keeps_its_draft_and_deleting_removes_it(draft_project):
    draft_project.rename_objective('Taste', 'Flavour')
    restored = FoodOptimizer('drafts')
    assert restored.result_drafts[1]['results'] == {'Flavour': 7}
    restored.remove_objective('Flavour')
    assert FoodOptimizer('drafts').result_drafts[1]['results'] == {}
