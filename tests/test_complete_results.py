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


def test_migration_exempts_rows_recorded_before_the_first_value():
    """Rev. 3: a measurement with no `not_for` exempts the recorded rows with
    no value for it that came BEFORE the first recorded row that has one.
    A blank after that is a result still to come: it waits."""
    opt = _patty(rows=0)
    _tell(opt, 1, 1, drop=("Cook loss",))
    _tell(opt, 2, 1, drop=("Cook loss",))
    _tell(opt, 3, 1)
    _tell(opt, 4, 2, drop=("Cook loss",))
    _tell(opt, 5, 2, drop=("Juiciness",))
    other = FoodOptimizer("migrated")
    other.import_json(_v15_state(opt))
    assert _not_for(other, "Cook loss") == [1, 2]
    assert _not_for(other, "Juiciness") == []
    assert _not_for(other, "Firmness") == []
    assert other.waiting_on(other.index_of_formulation(4)) == ["Cook loss"]
    assert other.waiting_on(other.index_of_formulation(5)) == ["Juiciness"]


def test_migration_exempts_every_row_when_none_has_a_value():
    opt = _patty(rows=0)
    for no in (1, 2, 3):
        _tell(opt, no, 1, drop=("Cook loss",))
    other = FoodOptimizer("migrated_none")
    other.import_json(_v15_state(opt))
    assert _not_for(other, "Cook loss") == [1, 2, 3]
    assert other.waiting_rows() == []


def test_migration_follows_recording_order_not_numbers():
    """Recording order is the history's order: a Round recorded one sheet
    at a time can record 6 before 5."""
    opt = _patty(rows=0)
    _tell(opt, 6, 2, drop=("Cook loss",))
    _tell(opt, 4, 2)
    _tell(opt, 5, 2, drop=("Cook loss",))
    other = FoodOptimizer("migrated_order")
    other.import_json(_v15_state(opt))
    assert _not_for(other, "Cook loss") == [6]


def test_a_migrated_project_keeps_its_best_and_its_later_partial_row_waits():
    """A 0.7.x project: the star is on a complete row, and a later row was
    saved with Cook loss still to come. After load the star stays, and the
    partial row reads waiting — it is not rescaled past the star (review #3)."""
    opt = _patty(rows=0)
    for no in range(1, 7):
        opt.tell({"Pea": 5.0, "Oil": 3.0},
                 {"Firmness": 5.0, "Juiciness": 6.0, "Cook loss": 20.0},
                 formulation_no=no, batch_no=1)
    opt.tell({"Pea": 6.0, "Oil": 5.0}, {"Firmness": 6.0, "Juiciness": 9.5},
             formulation_no=7, batch_no=2)
    star = opt.best_formulation_no()
    assert star != 7
    other = FoodOptimizer("migrated_star")
    other.import_json(_v15_state(opt))
    assert other.best_formulation_no() == star
    assert _not_for(other, "Cook loss") == []
    frame = other.history_frame()
    assert _row(frame, 7)[wording.OVERALL_SCORE_COLUMN] == "waiting for Cook loss"
    assert _row(frame, star)[wording.BEST_SO_FAR_COLUMN] == "★"


def test_migration_runs_after_the_identity_backfill():
    """A 0.2.x file has no formulation numbers: they are backfilled first,
    then not_for is read off them, then the scores (review I1)."""
    opt = _patty(rows=0)
    _tell(opt, 1, 1, drop=("Cook loss",))
    _tell(opt, 2, 1)
    _tell(opt, 3, 1)
    _tell(opt, 4, 2, drop=("Cook loss",))
    state = _v15_state(opt)
    for key in ('formulation_ids', 'batch_history', 'notes_history',
                'next_formulation_no', 'next_batch_number'):
        state.pop(key, None)
    state['CLASS_VERSION'] = 6
    other = FoodOptimizer("legacy")
    other.import_json(state)
    assert other.formulation_ids == [1, 2, 3, 4]
    assert _not_for(other, "Cook loss") == [1]


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


def test_class_version_17_round_trips():
    opt = _patty(rows=2, cook_loss=False)
    _add_cook_loss(opt)
    state = opt.export_json()
    assert FoodOptimizer.CLASS_VERSION == 17
    assert state['CLASS_VERSION'] == 17
    assert FoodOptimizer.validate_state(state)['version'] == 17
    other = FoodOptimizer("round_trip")
    other.import_json(state)
    assert _not_for(other, "Cook loss") == [1, 2]


def test_an_older_file_is_saved_back_with_not_for_on_open():
    opt = _patty(rows=0)
    _tell(opt, 3, 1, drop=("Cook loss",))
    _tell(opt, 4, 1)
    state = _v15_state(opt)
    state['project_name'] = "older"      # import_json takes the file's name
    store = LocalStorage()
    store.save("older", state)
    FoodOptimizer("older")
    saved = store.load("older")
    assert saved['CLASS_VERSION'] == 17
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
    opt = _patty(rows=0)
    _tell(opt, 4, 1, drop=("Cook loss",))
    for no in (5, 6, 7):
        _tell(opt, no, 1)
    other = FoodOptimizer("migrated_scores")
    other.import_json(_v15_state(opt))
    u = _u(4)
    assert other.Y_history[0] == pytest.approx(
        (50 * u["Firmness"] + 30 * u["Juiciness"]) * 100 / 80)
    assert other.fully_measured(0)


# ---------------------------------------------------------------- Task 4


def _spy_gp(opt):
    seen = []
    real = opt._build_gp

    def spy(X, Y, dim, tf):
        seen.append((X.detach().clone(), Y.detach().clone()))
        return real(X, Y, dim, tf)

    opt._build_gp = spy
    return seen


def test_a_waiting_row_is_not_taught_to_the_model():
    opt = _patty(rows=6)
    _tell(opt, 7, 3, drop=("Cook loss",))
    seen = _spy_gp(opt)
    opt.ask(1)
    X, Y = seen[0]
    assert X.shape[0] == 6
    assert torch.equal(Y, torch.tensor(opt.Y_history[:6], dtype=torch.double).unsqueeze(-1))


def test_once_its_value_is_entered_it_is_taught():
    opt = _patty(rows=6)
    _tell(opt, 7, 3, drop=("Cook loss",))
    opt.edit_result(6, _full(7))
    seen = _spy_gp(opt)
    opt.ask(1)
    assert torch.equal(seen[0][1], torch.tensor(opt.Y_history, dtype=torch.double).unsqueeze(-1))


def test_expected_improvement_and_its_baseline_see_only_complete_rows():
    opt = _patty(rows=6)
    opt.set_bo_config({"acquisition": "qlogei"})
    _tell(opt, 7, 3, drop=("Cook loss",))
    opt.Y_history[6] = 1000.0        # a waiting row must never be the best_f
    seen = []
    real = opt._build_acqf
    opt._build_acqf = lambda gp, X, Y: seen.append((X.shape[0], float(Y.max()))) or real(gp, X, Y)
    opt.ask(1)
    assert seen[0][0] == 6 and seen[0][1] < 1000.0


def test_the_cold_start_counts_only_complete_formulations():
    opt = _patty(rows=COLD_START_RUNS - 1)
    _tell(opt, COLD_START_RUNS, 2, drop=("Cook loss",))
    seen = _spy_gp(opt)
    opt.ask(1)
    assert seen == []                                   # still spread out
    assert opt.compared_with_column() == wording.COMPARED_WITH_ALLOWED
    opt.set_pending_batch(None)
    opt.edit_result(COLD_START_RUNS - 1, _full(COLD_START_RUNS))
    opt.ask(1)
    assert len(seen) == 1                               # the model now
    assert opt.compared_with_column() != wording.COMPARED_WITH_ALLOWED


def test_the_cold_start_fallback_needs_a_complete_formulation(monkeypatch):
    opt = _patty(rows=0)
    _tell(opt, 1, 1, drop=("Cook loss",))
    calls = []
    monkeypatch.setattr(opt, "_cold_start_pool", lambda *a: [])
    monkeypatch.setattr(opt, "_ask_optimize", lambda *a: calls.append(1) or [])
    with pytest.raises(ValueError):
        opt.ask(1)
    assert calls == []
    _tell(opt, 2, 1)
    with pytest.raises(ValueError):
        opt.ask(1)
    assert calls == [1]


def test_no_best_until_a_formulation_has_all_its_results():
    opt = _patty(rows=0)
    _tell(opt, 1, 1, drop=("Cook loss",))
    assert opt.best_index() is None
    assert opt.best_formulation_no() is None
    assert opt.best_so_far() == [None]
    assert opt.history_frame()[wording.BEST_SO_FAR_COLUMN].tolist() == [""]
    assert opt.vs_best_text(_recipe(5)) == ""
    _tell(opt, 2, 1)
    _tell(opt, 3, 1, drop=("Juiciness",))
    assert opt.best_formulation_no() == 2
    assert opt.best_so_far() == [None, opt.Y_history[1], opt.Y_history[1]]


def test_export_trajectory_marks_waiting_rows():
    opt = _patty(rows=2)
    _tell(opt, 3, 1, drop=("Cook loss",))
    text = opt.export_trajectory()
    line = next(l for l in text.splitlines() if "Formulation 3:" in l)
    assert "waiting for Cook loss" in line and "score" not in line
    assert not line.startswith("*")
    none = _patty(rows=0)
    _tell(none, 1, 1, drop=("Cook loss",))
    assert not any(l.startswith("*") for l in none.export_trajectory().splitlines())


# ---------------------------------------------------------------- Task 5
import pandas as pd


def _row(frame, number):
    return frame[frame["Formulation"] == number].iloc[0]


def test_the_table_names_what_a_formulation_waits_for():
    opt = _patty(rows=2)
    _tell(opt, 3, 1, drop=("Cook loss",))
    _tell(opt, 4, 1, drop=("Cook loss", "Juiciness"))
    frame = opt.history_frame()
    assert _row(frame, 3)[wording.OVERALL_SCORE_COLUMN] == "waiting for Cook loss"
    assert (_row(frame, 4)[wording.OVERALL_SCORE_COLUMN]
            == "waiting for Juiciness and Cook loss")
    assert _row(frame, 1)[wording.OVERALL_SCORE_COLUMN] == f"{opt.Y_history[0]:.1f}"


def test_a_measurement_that_does_not_apply_is_a_blank_number_in_the_frame():
    opt = _patty(rows=2, cook_loss=False)
    _add_cook_loss(opt)
    frame = opt.history_frame()
    # The cell stays a number (NaN) so the column sorts as numbers; the dash
    # is the Results table's display formatting (review #5).
    assert pd.isna(_row(frame, 1)["Cook loss"])
    assert pd.api.types.is_numeric_dtype(frame["Cook loss"])
    assert _row(frame, 1)[wording.OVERALL_SCORE_COLUMN] == f"{opt.Y_history[0]:.1f}"
    details = {d['name']: d for d in opt.closeness_details(0)}
    assert details["Cook loss"]["measured"] == "—"
    assert details["Cook loss"]["off_by"] == "—"
    assert wording.NOT_MEASURED not in str(frame.to_dict())


def test_best_first_puts_scored_then_waiting_then_not_scored():
    opt = _patty(rows=2)
    _tell(opt, 3, 1, drop=("Cook loss",))
    opt.Y_history[2] = 1000.0                 # a high number never jumps the queue
    opt.set_pending_batch([_recipe(4)])
    opt.record_skipped(4, opt.pending_batch_no, _recipe(4))
    order = list(opt.history_frame(order=wording.SORT_BEST_FIRST)["Formulation"])
    best = opt.best_formulation_no()
    assert order == [best, 3 - best, 3, 4]


def test_downloads_leave_a_waiting_score_blank_and_raw_values_alone():
    opt = _patty(rows=2, cook_loss=False)
    _add_cook_loss(opt)
    _tell(opt, 3, 1, drop=("Cook loss",))
    frame = opt.history_export_frame()
    assert _row(frame, 3)[wording.OVERALL_SCORE_COLUMN] == ""
    assert _row(frame, 1)[wording.OVERALL_SCORE_COLUMN] == round(opt.Y_history[0], 2)
    assert pd.isna(_row(frame, 1)["Cook loss"])      # a blank, not a dash


def test_a_downloaded_waiting_row_imports_as_scored_on_what_it_has():
    """Download all formulations, then import it elsewhere: the blank
    measurement becomes an exemption, not a wait (review focus 4)."""
    opt = _patty(rows=2)
    _tell(opt, 3, 1, drop=("Cook loss",))
    row = _row(opt.history_export_frame(), 3)
    other = _patty(rows=0)
    results = {n: float(row[n]) for n in ("Firmness", "Juiciness", "Cook loss")
               if not pd.isna(row[n])}
    other.import_formulation({"Pea": float(row["Pea (g)"]),
                              "Oil": float(row["Oil (g)"])}, results)
    assert other.fully_measured(0)
    assert other.waiting_rows() == []


# ---------------------------------------------------------------- Task 8


def test_the_insights_score_model_learns_only_from_complete_rows():
    import bo_views
    import insights
    opt = _patty(rows=6)
    _tell(opt, 7, 3, drop=("Cook loss",))
    rows = list(range(7))

    def counted(name):
        return sum(insights.measured(opt, name, i) is not None for i in rows)
    assert insights.measured(opt, insights.SCORE, 6) is None
    assert counted(insights.SCORE) == 6
    assert counted("Firmness") == 7
    assert counted("Cook loss") == 6
    # The rows a Generate learns from (its generation record) leave the
    # waiting one out too.
    assert bo_views._complete_rows(opt) == list(range(6))


def test_the_progress_chart_leaves_a_gap_for_a_waiting_row():
    import math
    import ui_insights
    opt = _patty(rows=2)
    _tell(opt, 3, 1, drop=("Cook loss",))
    _tell(opt, 4, 2)
    frame = ui_insights.progress_frame(opt)
    score = frame[wording.OVERALL_SCORE_COLUMN].tolist()
    assert math.isnan(score[2]) and not math.isnan(score[3])
    best = frame[wording.BEST_SO_FAR_COLUMN].tolist()
    assert best[2] == best[1]


def test_repeat_spread_leaves_out_a_rescaled_repeat():
    """A repeat recorded before Cook loss was added (rescaled) beside one
    recorded after (full): the rescaled score is not on the full one's
    footing (review fixes, Task 1), so there is no pair to compare."""
    import insights
    opt = _patty(rows=1, cook_loss=False)
    _add_cook_loss(opt)
    _tell(opt, 2, 1)
    opt.recipe_history[1] = dict(opt.recipe_history[0])
    opt.X_history[1] = list(opt.X_history[0])
    import ui_results
    # The two are Repeats of each other...
    assert insights.repeat_groups(opt, [0, 1]) == [[0, 1]]
    # ...but the Repeats line compares only what Best so far compares.
    assert insights.repeat_groups(opt, opt.best_candidates()) == []
    assert ui_results.best_repeats_text(opt) == ""


def test_every_new_string_passes_the_vocabulary_guard():
    from tests.test_food_bo import _BANNED
    strings = [
        wording.waiting_for("Cook loss"), wording.waiting_for(),
        wording.complete_results_caption(), wording.NO_COMPLETE_FORMULATION,
        wording.overall_score_waiting("Cook loss"),
        wording.saved_waiting([12], "Cook loss"),
        wording.saved_waiting([7, 8, 9, 10, 11, 12], "Cook loss"),
        wording.formulation_corrected(12, "Cook loss"),
        wording.formulation_scored(12, "Cook loss"),
    ]
    for text in strings:
        assert not any(p.search(text) for p in _BANNED), text
    assert wording.saved_waiting([12], "Cook loss") == \
        "Formulation 12 saved · waiting for Cook loss."
    assert wording.saved_waiting([12, 7, 8, 9, 10, 11], "Cook loss") == \
        "Formulations 7–12 saved · waiting for Cook loss."
    assert wording.saved_waiting([7, 9, 12], "Cook loss") == \
        "Formulations 7, 9 and 12 saved · waiting for Cook loss."
