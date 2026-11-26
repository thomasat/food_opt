import pytest

from egbo._limits import (PropertyLimit, QuantityLimit, at_total, check,
                          linear_constraints, missing_property_values,
                          off_total, project_onto_total,
                          snapped_if_it_still_fits, total_band, total_reach)
from egbo._space import Variable, ingredient_total


def two(low=0, high=10):
    return [Variable("A", low, high), Variable("B", low, high)]


def test_quantity_limit_as_inequality():
    (idx, coeffs, rhs), = linear_constraints(
        two(), [QuantityLimit(["A", "B"], max_val=12)], [], {})
    assert idx.tolist() == [0, 1]
    assert coeffs.tolist() == [-10.0, -10.0]
    assert rhs == -12


def test_fixed_ingredient_moves_to_offset():
    v = [Variable("A", 0, 10), Variable("B", 4, 4)]
    (idx, coeffs, rhs), = linear_constraints(
        v, [QuantityLimit(["A", "B"], max_val=12)], [], {})
    assert idx.tolist() == [0]
    assert coeffs.tolist() == [-10.0]
    assert rhs == -(12 - 4)


def test_widened_frame_writes_limits_over_it_and_keeps_own_range():
    """D in 4..10 searched over the frame 0..10: "D >= 8" is u >= 0.8, and
    D's own range adds u >= 0.4 (u <= 1 is the box itself)."""
    v = [Variable("A", 0, 10), Variable("D", 4, 10)]
    frame = [(0.0, 10.0), (0.0, 10.0)]
    rows = linear_constraints(v, [QuantityLimit(["D"], min_val=8)], [], {},
                              frame)
    assert [(i.tolist(), c.tolist(), r) for i, c, r in rows] == [
        ([1], [10.0], 8.0), ([1], [1.0], pytest.approx(0.4))]
    # Without a frame, each variable's own range, as before.
    (idx, coeffs, rhs), = linear_constraints(
        v, [QuantityLimit(["D"], min_val=8)], [], {})
    assert coeffs.tolist() == [6.0] and rhs == 8 - 4


def test_empty_formulation_fails_a_property_limit_only():
    """No ingredients is no amount per 100 g: it meets no property limit
    (min or max), though the linear form reads it as 0 >= 0."""
    v = two(0, 100)
    props = {"A": {"protein": 80.0}, "B": {"protein": 0.0}}
    empty = {"A": 0.0, "B": 0.0}
    for pl in (PropertyLimit("Protein", min_val=5),
               PropertyLimit("Protein", max_val=50)):
        assert not check(v, [], [pl], props, empty)
    assert check(v, [], [PropertyLimit("Protein")], props, empty)
    assert check(v, [], [], props, empty)
    assert check(v, [QuantityLimit(["A", "B"], max_val=10)], [], props, empty)


def test_property_limit_per_100g():
    v = two(0, 100)
    props = {"A": {"protein": 80.0}, "B": {"protein": 0.0}}
    pls = [PropertyLimit("Protein", min_val=5)]
    assert check(v, [], pls, props, {"A": 10, "B": 90})
    assert not check(v, [], pls, props, {"A": 5, "B": 95})
    assert len(linear_constraints(v, [], pls, props)) == 1
    assert missing_property_values(v, {"A": {"protein": 1}}, "Protein") == ["B"]
    assert not check(v, [], pls, {"A": {"protein": 1}}, {"A": 10, "B": 90})


def test_unreachable_limit_refused():
    v = [Variable("A", 5, 5), Variable("B", 5, 5)]
    with pytest.raises(ValueError, match="A \\+ B"):
        linear_constraints(v, [QuantityLimit(["A", "B"], max_val=8)], [], {})


def test_project_onto_total_hits_total():
    out = project_onto_total(two(), {"A": 1.0, "B": 2.0}, 12)
    assert out["A"] + out["B"] == pytest.approx(12)
    assert project_onto_total(two(), {"A": 1, "B": 2}, 25) is None
    assert project_onto_total(two(), {"A": 1, "B": 2}, -1) is None


def test_total_reach():
    v = [Variable("A", 1, 10), Variable("B", 2, 5),
         Variable("T", 0, 9, kind="setting")]
    assert total_reach(v) == (3, 15)
    assert total_band(100) == pytest.approx((99.5, 100.5))


def test_snapped_kept_only_if_limits_hold():
    v = [Variable("Pea", 0, 20), Variable("Gluten", 0, 20),
         Variable("Water", 0, 100)]
    qls = [QuantityLimit(["Pea", "Gluten"], max_val=20)]
    recipe = {"Pea": 10.0, "Gluten": 10.0, "Water": 79.68}
    # projecting to 100 would add 0.32 g, spread over all three
    out = snapped_if_it_still_fits(v, qls, [], {}, recipe, 100)
    assert out == recipe
    free = {"Pea": 5.0, "Gluten": 5.0, "Water": 89.68}
    out = snapped_if_it_still_fits(v, qls, [], {}, free, 100)
    assert sum(out.values()) == pytest.approx(100)


def test_off_total_and_at_total():
    v = [Variable("A", 0, 100), Variable("B", 0, 100),
         Variable("T", 0, 9, kind="setting")]
    assert not off_total(v, {"A": 50, "B": 49}, 100)
    assert off_total(v, {"A": 50, "B": 47}, 100)
    out = at_total(v, {"A": 50, "B": 47, "T": 3}, 100)
    assert out["A"] + out["B"] == pytest.approx(100)
    assert out["T"] == 3
    assert ingredient_total(v, out) == pytest.approx(100)
