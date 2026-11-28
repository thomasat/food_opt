"""A size refused as past what the amounts reach names the size to type at
two decimals, rounded inward so the number said is one the box accepts
(WIN11 smoke 2026-11-27: "enter 98.6905 g or less")."""
import pytest

import ui_batch
from food_bo import FoodOptimizer


@pytest.fixture
def opt(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    o = FoodOptimizer("bounds", robust=False)
    o.add_ingredient("Oat base", 10.0041, 50.12345)
    o.add_ingredient("Sugar", 20.0, 48.56705)
    o.add_objective("Taste", 1, goal="max", min_val=0, max_val=10)
    return o


def test_the_round_box_names_the_largest_size_at_two_decimals(opt):
    # Reach 30.0041 – 98.6905 g.
    assert "enter 98.69 g or less" in ui_batch._unreachable(opt, 110.0)
    assert ui_batch._unreachable(opt, 98.69) == ""


def test_the_round_box_names_the_smallest_size_rounded_up(opt):
    assert "enter 30.01 g or more" in ui_batch._unreachable(opt, 5.0)
    assert ui_batch._unreachable(opt, 30.01) == ""


def test_set_up_names_the_reach_at_two_decimals(opt):
    with pytest.raises(ValueError, match=r"98\.69 g") as high:
        opt.set_formulation_total(110)
    assert "98.6905" not in str(high.value)
    with pytest.raises(ValueError, match=r"30\.01 g") as low:
        opt.set_formulation_total(5)
    assert "30.0041" not in str(low.value)


def test_a_round_number_stays_round(opt):
    assert opt.size_bound_text(100.0, up=False) == "100 g"
    assert opt.size_bound_text(99.99999999999, up=False) == "100 g"
    assert opt.size_bound_text(113.75, up=False) == "113.75 g"
    assert opt.size_bound_text(72.0500001, up=True) == "72.06 g"
