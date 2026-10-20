"""0.8.0 BO views, Task 4 (spec 2026-09-30-bo-views §2.1, §3.1, §5; review
B4, I1, I4, I5, I12, M4, M5, M9, M12, M15): Make a round records how each
Round was generated. v3: the track line, E2, E3 and the Why these fold left
Make a round (no forecasts on that screen); their tests went with them
(tests/test_v3_make_round.py covers the screen)."""

import pytest
import insights
import ui_batch
from food_bo import FoodOptimizer
from tests.test_insights import _burger


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


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
