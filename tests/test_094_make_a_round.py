"""0.8.0 BO views, Task 4 (spec 2026-09-30-bo-views §2.1, §3.1, §5; review
B4, I1, I4, I5, I12, M4, M5, M9, M12, M15): Make a round records how each
Round was generated. v3: the track line, E2, E3 and the Why these fold left
Make a round (no forecasts on that screen); their tests went with them
(tests/test_v3_make_round.py covers the screen)."""

import pytest
from streamlit.testing.v1 import AppTest

import bo_views
import insights
import ui_batch
import wording
from food_bo import FoodOptimizer
from tests.test_app_ui import APP_PATH
from tests.test_insights import _burger


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with bo_views.fresh():
        yield


def _open_round(rounds=4, n=3):
    opt = _burger(rounds=rounds)
    ui_batch.ask_alone(opt, n_suggestions=n)
    return opt


def _app(opt, tab=wording.TAB_BATCH):
    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.session_state["_loaded_project"] = opt.project_name
    at.run()
    at.session_state["main_tab"] = tab
    at.run()
    assert not at.exception
    return at


def _rerun(at, tab=wording.TAB_BATCH):
    """AppTest keeps no tab choice between runs: say it again."""
    at.session_state["main_tab"] = tab
    at.run()
    assert not at.exception


# ------------------------------------------------------------------ #
#  The generation record (review B4)
# ------------------------------------------------------------------ #

def test_generate_records_how_the_round_was_made():
    """T47 review minor 2: one row still waiting for a result is not among
    the rows ask() learns from, so it is not in the record either."""
    opt = _burger(rounds=4)
    last = len(opt.Y_history) - 1
    waiting = int(opt.formulation_ids[last])
    opt.results_history[last] = {k: v for k, v in list(opt.results_history[last].items())[:1]}
    assert not opt.fully_measured(last)
    ui_batch.ask_alone(opt, n_suggestions=3)
    again = FoodOptimizer(opt.project_name)
    record = again.round_generation[again.pending_batch_no]
    assert record["regime"] == "warm" and record["n"] == 3
    assert record["seed"] == min(r["formulation"] for r in again.pending_batch)
    assert record["rows"] == [int(n) for n in again.formulation_ids if int(n) != waiting]
    assert record["picks"] == [r["formulation"] for r in again.pending_batch]
    assert bo_views.generation(again, again.pending_batch_no).exact


def test_a_cold_round_is_recorded_as_cold():
    opt = _burger(rounds=1)
    ui_batch.ask_alone(opt, n_suggestions=2)
    assert opt.round_generation[opt.pending_batch_no]["regime"] == "cold"


def test_recording_never_moves_a_suggestion():
    """The record is made before ask() and changes nothing it reads."""
    a, b = _burger(rounds=4, seed=3), _burger(rounds=4, seed=3)
    ui_batch.ask_alone(a, n_suggestions=3)
    with insights.RNG_LOCK:
        b.ask(n_suggestions=3)
    assert [r["recipe"] for r in a.pending_batch] == [r["recipe"] for r in b.pending_batch]


# ------------------------------------------------------------------ #
#  Why these suggestions (§3.1)
# ------------------------------------------------------------------ #

def test_a_cold_round_says_why_it_has_no_values():
    opt = _burger(rounds=1)
    ui_batch.ask_alone(opt, n_suggestions=2)
    w = bo_views.why(opt)
    assert w.cold


# ------------------------------------------------------------------ #
#  The track line's words (Review's; review I1)
# ------------------------------------------------------------------ #

def _track(no_skill=None, rough=False, gap=None, see=False, record=True):
    fact = None if no_skill is None else insights.SkillFact(no_skill, 16, [2, 3, 4, 5],
                                                            "no_skill")
    return insights.RoundTrack(rough, 11, 12, gap, see, fact, [3, 4, 5], record)


# ------------------------------------------------------------------ #
#  Review Focus (plan)
# ------------------------------------------------------------------ #

def test_a_regenerated_round_is_recorded_again_and_stays_exact():
    """Generate a different Round keeps its number and asks again with a
    new seed: the record follows it."""
    opt = _open_round()
    no, old = opt.pending_batch_no, dict(opt.round_generation[opt.pending_batch_no])
    opt.set_pending_batch(None)
    ui_batch.ask_alone(opt, n_suggestions=3, batch_no=no)
    new = opt.round_generation[no]
    assert new["seed"] != old["seed"] and new["seed"] == min(
        r["formulation"] for r in opt.pending_batch)
    assert bo_views.why(opt).exact


def _ferment():
    opt = FoodOptimizer("ferment_why")
    opt.add_process_parameter("Mixing time", 1, 10, unit="min")
    opt.add_process_parameter("Temperature", 20, 40, unit="°C")
    opt.add_objective("Acidity", 1.0, goal="max", min_val=0, max_val=10, unit="pH")
    for k in range(6):
        opt.tell({"Mixing time": 1 + 1.5 * k, "Temperature": 20 + (7 * k) % 20},
                 {"Acidity": 3 + 0.5 * k - 0.02 * k * k}, formulation_no=k + 1, batch_no=1)
    return opt
