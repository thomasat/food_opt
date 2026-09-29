"""Complete results only (0.8.0): a formulation teaches the suggestion model
once every measurement that applies to it has a value, and a measurement
added after a formulation was recorded does not apply to it.
Spec: docs/superpowers/specs/2026-09-28-complete-results-design.md (rev. 2)."""
import itertools

import pytest
import torch

import wording
from food_bo import COLD_START_RUNS, FoodOptimizer
from storage import LocalStorage


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


_n = itertools.count()


def _recipe(no):
    return {"Pea": 4 + (no * 0.37) % 4, "Oil": 2 + (no * 0.61) % 4}


def _full(no):
    r = _recipe(no)
    return {"Firmness": 2 + 0.9 * r["Pea"], "Juiciness": 1 + 1.2 * r["Oil"],
            "Cook loss": 30 - 3 * r["Pea"]}


def _tell(opt, no, round_no=1, drop=()):
    """Record formulation `no` with every CURRENT measurement except `drop`.
    Only current names: tell() stores any key it is handed, and a value
    filed under a measurement added later would make it apply."""
    names = {o['name'] for o in opt.objectives}
    opt.tell(_recipe(no), {k: v for k, v in _full(no).items()
                           if k in names and k not in drop},
             formulation_no=no, batch_no=round_no)


def _add_cook_loss(opt):
    opt.add_objective("Cook loss", 20, goal="min", min_val=0, max_val=40)


def _patty(rows=0, cook_loss=True):
    """Firmness (target 6 of 0-10, share 50), Juiciness (higher, 30) and,
    unless cook_loss=False, Cook loss (lower, 0-40, 20). `rows` fully
    measured formulations numbered 1..rows, three to a Round."""
    opt = FoodOptimizer(f"complete_{next(_n)}")
    opt.set_amount_unit("g")
    opt.add_ingredient("Pea", 4, 8)
    opt.add_ingredient("Oil", 2, 6)
    opt.add_objective("Firmness", 50, goal="target", target=6,
                      min_val=0, max_val=10)
    opt.add_objective("Juiciness", 30, goal="max", min_val=0, max_val=10)
    if cook_loss:
        _add_cook_loss(opt)
    for k in range(rows):
        _tell(opt, k + 1, k // 3 + 1)
    return opt


def _u(no):
    """Closeness of formulation `no` to each goal, by hand."""
    v = _full(no)
    return {"Firmness": max(0.0, 1 - abs(v["Firmness"] / 10 - 0.6)),
            "Juiciness": v["Juiciness"] / 10,
            "Cook loss": 1 - v["Cook loss"] / 40}


def _not_for(opt, name):
    return next(o for o in opt.objectives if o['name'] == name)['not_for']


# ---------------------------------------------------------------- Task 2


def test_a_new_measurement_does_not_apply_to_formulations_already_recorded():
    opt = _patty(rows=4, cook_loss=False)
    _add_cook_loss(opt)
    assert _not_for(opt, "Cook loss") == [1, 2, 3, 4]
    assert _not_for(opt, "Firmness") == []


def test_a_round_recorded_one_sheet_at_a_time():
    """Recorded rows of the open Round are exempt; its unrecorded rows and
    a Not scored row are not (spec §3; review B1)."""
    opt = _patty(rows=3, cook_loss=False)
    opt.set_pending_batch([_recipe(4), _recipe(5), _recipe(6), _recipe(7)])
    assert [r['formulation'] for r in opt.pending_batch] == [4, 5, 6, 7]
    _tell(opt, 4, opt.pending_batch_no)
    _tell(opt, 6, opt.pending_batch_no)
    opt.record_skipped(5, opt.pending_batch_no, _recipe(5))
    _add_cook_loss(opt)
    assert _not_for(opt, "Cook loss") == [1, 2, 3, 4, 6]


def test_replace_rename_and_changes_keep_not_for():
    opt = _patty(rows=3, cook_loss=False)
    _add_cook_loss(opt)
    opt.add_objective("Cook loss", 20, goal="min", min_val=0, max_val=50)
    assert _not_for(opt, "Cook loss") == [1, 2, 3]
    opt.rename_objective("Cook loss", "Loss on cooking")
    assert _not_for(opt, "Loss on cooking") == [1, 2, 3]
    opt.update_objective("Loss on cooking", max_val=60)
    opt.set_shares({"Firmness": 40, "Juiciness": 30, "Loss on cooking": 30})
    assert _not_for(opt, "Loss on cooking") == [1, 2, 3]


def test_delete_then_add_again_is_a_new_measurement():
    opt = _patty(rows=2)
    assert _not_for(opt, "Cook loss") == []
    _tell(opt, 3, 1)
    opt.remove_objective("Cook loss")
    _add_cook_loss(opt)
    assert _not_for(opt, "Cook loss") == [1, 2, 3]


def test_an_import_is_exempt_from_what_it_has_no_value_for():
    opt = _patty(rows=2)
    opt.import_formulation(_recipe(9), {"Firmness": 6.0})
    number = int(opt.formulation_ids[-1])
    assert number == 3
    assert number in _not_for(opt, "Juiciness")
    assert number in _not_for(opt, "Cook loss")
    assert number not in _not_for(opt, "Firmness")


def _v15_state(opt):
    state = opt.export_json()
    for obj in state['objectives']:
        obj.pop('not_for', None)
    state['CLASS_VERSION'] = 15
    return state


def test_migration_exempts_every_recorded_row_without_a_value():
    opt = _patty(rows=3)
    _tell(opt, 4, 2, drop=("Cook loss",))
    _tell(opt, 5, 2, drop=("Juiciness",))
    other = FoodOptimizer("migrated")
    other.import_json(_v15_state(opt))
    assert _not_for(other, "Cook loss") == [4]
    assert _not_for(other, "Juiciness") == [5]
    assert _not_for(other, "Firmness") == []


def test_migration_runs_after_the_identity_backfill():
    """A 0.2.x file has no formulation numbers: they are backfilled first,
    then not_for is read off them, then the scores (review I1)."""
    opt = _patty(rows=3)
    _tell(opt, 4, 2, drop=("Cook loss",))
    state = _v15_state(opt)
    for key in ('formulation_ids', 'batch_history', 'notes_history',
                'next_formulation_no', 'next_batch_number'):
        state.pop(key, None)
    state['CLASS_VERSION'] = 6
    other = FoodOptimizer("legacy")
    other.import_json(state)
    assert other.formulation_ids == [1, 2, 3, 4]
    assert _not_for(other, "Cook loss") == [4]


@pytest.mark.parametrize("bad", ["3", [1, "2"], [True], None, {"n": 1}, [1.5]])
def test_the_copy_validator_refuses_a_malformed_not_for(bad):
    opt = _patty(rows=2)
    state = opt.export_json()
    state['objectives'][0]['not_for'] = bad
    with pytest.raises(ValueError):
        FoodOptimizer.validate_state(state)


def test_the_copy_validator_accepts_a_file_without_not_for():
    opt = _patty(rows=2)
    assert FoodOptimizer.validate_state(_v15_state(opt))['version'] == 15


def test_class_version_16_round_trips():
    opt = _patty(rows=2, cook_loss=False)
    _add_cook_loss(opt)
    state = opt.export_json()
    assert FoodOptimizer.CLASS_VERSION == 16
    assert state['CLASS_VERSION'] == 16
    assert FoodOptimizer.validate_state(state)['version'] == 16
    other = FoodOptimizer("round_trip")
    other.import_json(state)
    assert _not_for(other, "Cook loss") == [1, 2]


def test_an_older_file_is_saved_back_with_not_for_on_open():
    opt = _patty(rows=2)
    _tell(opt, 3, 1, drop=("Cook loss",))
    state = _v15_state(opt)
    state['project_name'] = "older"      # import_json takes the file's name
    store = LocalStorage()
    store.save("older", state)
    FoodOptimizer("older")
    saved = store.load("older")
    assert saved['CLASS_VERSION'] == 16
    cook = next(o for o in saved['objectives'] if o['name'] == "Cook loss")
    assert cook['not_for'] == [3]


# ---------------------------------------------------------------- Task 3


def test_earlier_rows_are_scored_on_their_measurements_rescaled():
    opt = _patty(rows=3, cook_loss=False)
    _add_cook_loss(opt)
    for i, no in enumerate((1, 2, 3)):
        u = _u(no)
        assert opt.Y_history[i] == pytest.approx(
            (50 * u["Firmness"] + 30 * u["Juiciness"]) * 100 / 80)
        assert opt.fully_measured(i)
        assert opt.waiting_on(i) == []
        assert opt.not_expected(i) == ["Cook loss"]


def test_rows_recorded_after_the_measurement_need_it():
    opt = _patty(rows=3, cook_loss=False)
    _add_cook_loss(opt)
    _tell(opt, 4, 2, drop=("Cook loss",))
    assert opt.waiting_on(3) == ["Cook loss"]
    assert not opt.fully_measured(3)
    assert opt.waiting_rows() == [3]
    u = _u(4)
    # Stored as it always was (a blank counts 0); never shown, never taught.
    assert opt.Y_history[3] == pytest.approx(50 * u["Firmness"] + 30 * u["Juiciness"])


def test_open_round_rows_need_a_measurement_added_before_they_are_recorded():
    opt = _patty(rows=3, cook_loss=False)
    opt.set_pending_batch([_recipe(4), _recipe(5)])
    _add_cook_loss(opt)
    _tell(opt, 4, opt.pending_batch_no, drop=("Cook loss",))
    assert opt.waiting_on(3) == ["Cook loss"]


def test_a_value_entered_on_an_earlier_row_then_counts():
    opt = _patty(rows=3, cook_loss=False)
    _add_cook_loss(opt)
    results = dict(opt.results_history[0], **{"Cook loss": _full(1)["Cook loss"]})
    opt.edit_result(0, results)
    u = _u(1)
    assert opt.Y_history[0] == pytest.approx(
        50 * u["Firmness"] + 30 * u["Juiciness"] + 20 * u["Cook loss"])
    assert opt.not_expected(0) == []


def test_a_not_scored_row_scored_later_needs_the_new_measurement():
    opt = _patty(rows=3, cook_loss=False)
    opt.set_pending_batch([_recipe(4)])
    opt.record_skipped(4, opt.pending_batch_no, _recipe(4))
    opt.set_pending_batch(None)
    _add_cook_loss(opt)
    opt.score_skipped(4, {"Firmness": 6.0, "Juiciness": 5.0})
    assert opt.waiting_on(opt.index_of_formulation(4)) == ["Cook loss"]


def test_old_values_count_after_a_delete_and_add_again():
    """The values a delete left on the rows were measured: the measurement
    added again under the same name applies to them (spec §3)."""
    opt = _patty(rows=2)
    opt.remove_objective("Cook loss")      # the other two go to 62.5 / 37.5
    _add_cook_loss(opt)
    assert _not_for(opt, "Cook loss") == [1, 2]
    for i, no in enumerate((1, 2)):
        u = _u(no)
        assert opt.fully_measured(i)
        assert opt.Y_history[i] == pytest.approx(
            62.5 * u["Firmness"] + 37.5 * u["Juiciness"] + 20 * u["Cook loss"])


def test_deleting_a_measurement_releases_every_row_waiting_on_it():
    opt = _patty(rows=2)
    _tell(opt, 3, 1, drop=("Cook loss",))
    assert opt.waiting_rows() == [2]
    opt.remove_objective("Cook loss")
    assert opt.waiting_rows() == []


def test_an_import_is_scored_on_what_it_has():
    opt = _patty(rows=2)
    opt.import_formulation(_recipe(9), {"Firmness": 6.0})
    assert opt.fully_measured(2)
    assert opt.Y_history[2] == pytest.approx(100.0)   # on target, rescaled


def test_score_without_exempts_one_formulation_and_rescores_it():
    opt = _patty(rows=2)
    _tell(opt, 3, 1, drop=("Cook loss",))
    opt.score_without(3, "Cook loss")
    assert 3 in _not_for(opt, "Cook loss")
    assert opt.fully_measured(2)
    u = _u(3)
    assert opt.Y_history[2] == pytest.approx(
        (50 * u["Firmness"] + 30 * u["Juiciness"]) * 100 / 80)
    again = FoodOptimizer(opt.project_name)
    assert again.Y_history[2] == pytest.approx(opt.Y_history[2])
    # Undo is entering the value.
    opt.edit_result(2, dict(opt.results_history[2], **{"Cook loss": 10.0}))
    assert opt.not_expected(2) == []


def test_score_without_refuses_an_unknown_measurement():
    opt = _patty(rows=1)
    with pytest.raises(ValueError):
        opt.score_without(1, "Crunch")


def test_a_row_with_nothing_that_applies_is_never_fully_measured():
    """Every measurement it was measured on deleted, a new one added after
    it: nothing to be scored on. Never taught to the model as a 0, and no
    division by zero (review focus 3)."""
    opt = _patty(rows=2, cook_loss=False)
    opt.remove_objective("Juiciness")
    opt.remove_objective("Firmness")
    opt.add_objective("Taste", 1, goal="max", min_val=0, max_val=10)
    assert opt.Y_history == [0.0, 0.0]
    assert opt.waiting_on(0) == [] and not opt.fully_measured(0)
    assert opt.waiting_rows() == [0, 1]


def test_migration_scores_exempt_rows_on_what_they_have():
    opt = _patty(rows=3)
    _tell(opt, 4, 2, drop=("Cook loss",))
    other = FoodOptimizer("migrated_scores")
    other.import_json(_v15_state(opt))
    u = _u(4)
    assert other.Y_history[3] == pytest.approx(
        (50 * u["Firmness"] + 30 * u["Juiciness"]) * 100 / 80)
    assert other.fully_measured(3)
