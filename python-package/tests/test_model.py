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
