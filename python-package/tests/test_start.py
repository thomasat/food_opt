import numpy as np
import torch

from egbo._start import fill_from_repeats, select, sobol_pool


def test_pool_prefix_stable():
    small = sobol_pool(2048, 4, 7, 0, {})
    large = sobol_pool(8192, 4, 7, 0, {})
    assert torch.equal(small, large[:2048])


def test_three_then_two_equals_five():
    five = sobol_pool(5, 3, 11, 0, {})
    first = sobol_pool(5, 3, 11, 0, {})[:3]
    second = sobol_pool(5, 3, 11, 3, {})[:2]
    assert torch.equal(five, torch.cat([first, second]))


def test_fixed_columns_overwritten():
    pool = sobol_pool(8, 3, 1, 0, {1: 0.25})
    assert torch.all(pool[:, 1] == 0.25)


def _vec(rec):
    return np.array([rec["a"]], dtype=float)


def test_select_skips_issued_and_infeasible():
    candidates = [{"a": 1.0}, {"a": 2.0}, {"a": 3.0}, {"a": 3.0}, {"a": 4.0}, {"a": 5.0}]
    results, repeats, chosen = select(
        candidates, 2, lambda r: r["a"] != 2.0, _vec, np.array([[1.0]]))
    assert results == [{"a": 3.0}, {"a": 4.0}]
    assert repeats == [{"a": 1.0}, {"a": 3.0}]
    assert [c[0] for c in chosen] == [3.0, 4.0]


def test_fill_from_repeats_prefers_distinct():
    repeat_recipes = [{"a": 1.0}, {"a": 1.0}, {"a": 2.0}]
    out = fill_from_repeats(repeat_recipes, [np.array([1.0])], 2, _vec)
    assert out == [{"a": 2.0}, {"a": 1.0}]
