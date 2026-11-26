import pytest

from egbo._score import Objective, score, utility


def test_utility_max_min_target():
    assert utility(Objective("a", 1, "max"), 7) == 0.7
    assert utility(Objective("a", 1, "min"), 7) == pytest.approx(0.3)
    assert utility(Objective("a", 1, "target", target=6), 8) == pytest.approx(0.8)


def test_utility_clamps_outside_scale():
    assert utility(Objective("a", 1, "max"), 12) == 1.0
    assert utility(Objective("a", 1, "max"), -3) == 0.0
    assert utility(Objective("a", 1, "target", target=6), 12) == pytest.approx(0.6)


def test_zero_width_scale_uses_unit_range():
    obj = Objective("a", 1, "max", min_val=5, max_val=5)
    assert utility(obj, 5) == 0.0
    assert utility(obj, 5.5) == pytest.approx(0.5)


def test_score_is_weighted_sum():
    objs = [Objective("a", 2, "max"), Objective("b", 1, "max")]
    assert score(objs, {"a": 5, "b": 10}) == pytest.approx(2.0)
