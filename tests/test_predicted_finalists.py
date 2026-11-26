"""Task 4 (algorithm update, stable 940a0ce A7): formulations ranked by the
model's prediction, beside Best so far (which stays the observed score)."""
import itertools

import pytest
import torch

from food_bo import FoodOptimizer

_n = itertools.count()


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _project(rows):
    """rows: (Pea, Oil, Score out of 10). Score is the only measurement."""
    opt = FoodOptimizer(f"finalists_{next(_n)}")
    opt.set_amount_unit("g")
    opt.add_ingredient("Pea", 4, 8)
    opt.add_ingredient("Oil", 2, 6)
    opt.add_objective("Score", 100, goal="max", min_val=0, max_val=10)
    for k, (pea, oil, score) in enumerate(rows):
        opt.tell({"Pea": pea, "Oil": oil}, {"Score": score},
                 formulation_no=k + 1, batch_no=1)
    return opt


# One lone formulation far from the rest scores 90, the one highest observed
# score; six at one point score 87 to 89; the low ones disagree with each
# other, so the model learns the panel is noisy and pulls the lone score
# back below the six it has corroborated (by about 3 points of the score).
CLUSTER = range(1, 7)
LUCKY = (
    [(7.8, 5.8, 9.0)]
    + [(5.0, 3.0, v) for v in (8.7, 8.8, 8.9, 8.8, 8.7, 8.8)]
    + [(4.2, 2.2, v) for v in (0.0, 8.9, 1.0, 8.0)]
    + [(6.6, 4.4, v) for v in (0.5, 8.5)]
)


def test_finalists_prefer_consistent_over_lucky():
    opt = _project(LUCKY)
    # The lone row is the best observed by a clear point, not by a tie.
    assert opt.best_index() == 0
    assert opt.Y_history[0] - max(opt.Y_history[1:]) >= 1.0 - 1e-9
    top = opt.finalists(3)
    assert top[0] != 0
    assert top[0] in CLUSTER
    pred = opt.predicted_utilities()
    assert pred[top[0]][0] - pred[0][0] >= 2.0


def test_too_few_rows_gives_none():
    opt = _project([(5, 3, 2.0), (6, 4, 7.0), (7, 5, 5.0)])
    assert opt.predicted_utilities() is None
    assert opt.finalists(3) == [1, 2, 0]  # the observed order


def test_predictions_are_mean_and_sd_per_row():
    opt = _project(LUCKY)
    pred = opt.predicted_utilities()
    assert len(pred) == len(LUCKY)
    assert all(sd >= 0 for _, sd in pred)
    assert pred[2][0] == pytest.approx(87, abs=5)


def test_global_rng_untouched():
    opt = _project(LUCKY)
    before = torch.get_rng_state()
    assert opt.predicted_utilities() is not None
    assert torch.equal(before, torch.get_rng_state())


def test_waiting_rows_excluded():
    opt = FoodOptimizer(f"finalists_{next(_n)}")
    opt.set_amount_unit("g")
    opt.add_ingredient("Pea", 4, 8)
    opt.add_ingredient("Oil", 2, 6)
    opt.add_objective("Score", 60, goal="max", min_val=0, max_val=10)
    opt.add_objective("Juice", 40, goal="max", min_val=0, max_val=10)
    for k, (pea, oil, score) in enumerate(LUCKY):
        done = {"Score": score, "Juice": 5.0}
        if k == 1:                     # the best-looking one still waits
            done.pop("Juice")
        opt.tell({"Pea": pea, "Oil": oil}, done,
                 formulation_no=k + 1, batch_no=1)
    assert opt.waiting_rows() == [1]
    every = opt.finalists(len(LUCKY))
    assert 1 not in every
    assert len(every) == len(LUCKY) - 1
    assert len(opt.predicted_utilities()) == len(LUCKY)
