"""Range edits: an ingredient's min does not go below 0, and a range edit
may not newly put a limit out of reach (one already out of reach does not
block it)."""
import pytest

from egbo import Optimizer


def _two():
    opt = Optimizer(seed=0)
    opt.add_ingredient("A", 0, 10)
    opt.add_ingredient("B", 0, 10)
    opt.add_objective("Taste", 1)
    return opt


def test_negative_min_refused_for_an_ingredient():
    opt = _two()
    with pytest.raises(ValueError, match="C: min -1 is below 0"):
        opt.add_ingredient("C", -1, 5)
    with pytest.raises(ValueError, match="A: min -0.5 is below 0"):
        opt.set_range("A", -0.5, 5)
    assert [(v.name, v.low, v.high) for v in opt.variables] == [
        ("A", 0, 10), ("B", 0, 10)]


def test_negative_min_allowed_for_a_setting():
    opt = _two()
    opt.add_process_parameter("Chill", -18, 4)
    opt.set_range("Chill", -20, 4)
    assert opt.variables[-1].low == -20


def test_negative_min_already_on_file_is_let_stand():
    opt = _two()
    opt.variables[0].low = -1.0          # an older file's row
    opt.set_range("A", -1, 8)
    assert (opt.variables[0].low, opt.variables[0].high) == (-1, 8)


def test_narrowing_that_strands_a_quantity_limit_refused():
    opt = _two()
    opt.add_quantity_constraint(["A", "B"], min_val=15)
    opt.ask(2)
    pending = opt.pending()
    with pytest.raises(ValueError) as err:
        opt.set_range("A", 0, 4)
    assert str(err.value) == (
        "The limit on A + B cannot be met: the amounts allowed only reach 14 "
        "at most (with A at 0 to 4)")
    assert (opt.variables[0].low, opt.variables[0].high) == (0, 10)
    assert opt.pending() == pending
    opt.set_range("A", 0, 6)             # still reaches 16


def test_raising_a_min_that_strands_a_max_limit_refused():
    opt = _two()
    opt.add_quantity_constraint(["A", "B"], max_val=12)
    opt.set_range("A", 8, 10)                 # 8 at least, still under 12
    with pytest.raises(ValueError) as err:
        opt.set_range("B", 5, 10)
    assert str(err.value) == (
        "The limit on A + B cannot be met: the lowest amounts add up to 13 "
        "(with B at 5 to 10)")


def test_narrowing_that_strands_a_property_limit_refused():
    opt = _two()
    opt.set_property_values("Protein", {"A": 20, "B": 0})
    opt.add_constraint("Protein", min_val=10)
    with pytest.raises(ValueError, match="The limit on Protein cannot be met"
                                         ".*\\(with B at 15 to 20\\)"):
        opt.set_range("B", 15, 20)
    assert opt.variables[1].low == 0


def test_a_limit_already_out_of_reach_does_not_block():
    opt = _two()
    opt.set_range("A", 0, 4)
    opt.add_quantity_constraint(["A", "B"], min_val=15)   # 14 at most already
    opt.set_range("B", 0, 9)                              # allowed
    assert opt.variables[1].high == 9


def test_a_new_ingredient_is_not_a_range_edit():
    opt = _two()
    opt.add_quantity_constraint(["A", "B"], min_val=15)
    opt.add_ingredient("C", 0, 1)
    assert [v.name for v in opt.variables] == ["A", "B", "C"]
