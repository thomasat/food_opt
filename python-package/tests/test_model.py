import pytest
import torch

from egbo._model import pick_restart, suggest, validate_bo_config


def test_pick_restart_ties_go_to_smallest_candidate():
    candidates = torch.tensor([[[0.7, 0.2]], [[0.3, 0.9]], [[0.1, 0.1]]])
    values = torch.tensor([1.0, 1.0 + 5e-5, 0.5])
    assert pick_restart(candidates, values) == 1


def test_pick_restart_nan_never_wins():
    candidates = torch.tensor([[[0.1]], [[0.5]], [[0.9]]])
    values = torch.tensor([float("nan"), 0.2, 0.1])
    assert pick_restart(candidates, values) == 1


def _data():
    X = torch.tensor([[0.1, 0.2, 0.3], [0.9, 0.1, 0.5], [0.4, 0.8, 0.2],
                      [0.6, 0.5, 0.9], [0.2, 0.7, 0.6], [0.8, 0.9, 0.1]],
                     dtype=torch.double)
    Y = torch.tensor([[1.0], [3.5], [2.0], [5.0], [2.8], [4.1]],
                     dtype=torch.double)
    bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                          dtype=torch.double)
    return X, Y, bounds


def test_suggest_deterministic():
    X, Y, bounds = _data()
    a = suggest(X, Y, bounds, 2, 7)
    b = suggest(X, Y, bounds, 2, 7)
    assert a.shape == (2, 3)
    assert torch.equal(a, b)


def test_suggest_respects_fixed_and_constraints():
    X, Y, bounds = _data()
    # x0 + x1 <= 1.2  ->  -x0 - x1 >= -1.2
    cons = [(torch.tensor([0, 1]), torch.tensor([-1.0, -1.0], dtype=torch.double), -1.2)]
    out = suggest(X, Y, bounds, 2, 3, inequality_constraints=cons,
                  fixed_features={2: 0.25})
    assert torch.allclose(out[:, 2], torch.full((2,), 0.25, dtype=torch.double))
    assert bool(((out[:, 0] + out[:, 1]) <= 1.2 + 1e-6).all())


def test_validate_bo_config_falls_back():
    assert validate_bo_config({"kernel": "nonsense"})["kernel"] == "matern52"
    assert validate_bo_config(None) is None


def _scored(rows, **kwargs):
    from egbo import Optimizer
    opt = Optimizer(seed=0, **kwargs)
    opt.add_ingredient("A", 0, 10)
    opt.add_ingredient("B", 0, 10)
    opt.add_objective("Taste", 1)
    for a, b, taste in rows:
        opt.tell({"A": a, "B": b}, {"Taste": taste})
    return opt


def test_predicted_utilities_none_below_five_rows():
    opt = _scored([(1, 1, 2.0), (5, 5, 7.0), (9, 9, 5.0), (2, 8, 4.0)])
    assert opt.predicted_utilities() is None
    assert opt.finalists(3) == [1, 2, 3]       # observed order


def test_predicted_utilities_mean_and_sd_and_rng_untouched():
    rows = [(1, 1, 2.0), (5, 5, 7.0), (9, 9, 5.0), (2, 8, 4.0), (8, 2, 6.0),
            (5, 6, 7.5)]
    opt = _scored(rows)
    before = torch.random.get_rng_state().clone()
    pred = opt.predicted_utilities()
    assert torch.equal(torch.random.get_rng_state(), before)
    assert len(pred) == 6 and all(sd >= 0 for _, sd in pred)
    assert sorted(opt.finalists(10)) == list(range(6))
    assert len(opt.finalists()) == 3
