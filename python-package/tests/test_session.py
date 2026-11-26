"""Mid-run changes that a script would otherwise not see, and an ask()
that fails leaving the project as it was."""
import warnings

import pytest

from egbo import Optimizer


def _two():
    opt = Optimizer(seed=0)
    opt.add_ingredient("Pea protein", 2, 8)
    opt.add_ingredient("Coconut oil", 0, 5)
    opt.add_objective("Creaminess", weight=1)
    return opt


def _record(opt, n):
    for k, r in enumerate(opt.ask(n)):
        opt.tell(r, {"Creaminess": 2 + k})


def test_mid_run_min_lowered_to_zero_warns():
    opt = _two()
    _record(opt, 3)
    with pytest.warns(UserWarning) as caught:
        opt.add_ingredient("Sugar", 2, 6)
    assert [str(w.message) for w in caught] == [
        "Sugar: min 2 lowered to 0, because the 3 recorded formulations had "
        "none; pass keep_lowest=True to keep 2"]
    assert opt.variables[-1].low == 0.0


@pytest.mark.parametrize("call", [
    lambda o: o.add_ingredient("Sugar", 0, 6),
    lambda o: o.add_ingredient("Sugar", 2, 6, keep_lowest=True),
    lambda o: o.add_ingredient("Sugar", 2, 6, baseline=3),
    lambda o: o.set_range("Pea protein", 3, 8),
])
def test_no_warning_when_nothing_is_lowered(call):
    opt = _two()
    _record(opt, 3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        call(opt)


def test_no_warning_before_any_result():
    opt = _two()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        opt.add_ingredient("Sugar", 2, 6)
    assert opt.variables[-1].low == 2.0


@pytest.mark.parametrize("warm", [False, True])
def test_refused_ask_keeps_the_open_batch(warm):
    opt = _two()
    if warm:
        _record(opt, 5)
    batch = opt.ask(2)
    pending, retired = opt.pending(), list(opt.retired)
    opt.add_quantity_constraint(["Pea protein", "Coconut oil"], min_val=20)
    with pytest.raises(ValueError, match="No formulation meets all the limits"):
        opt.ask(2)
    assert opt.pending() == pending and opt.retired == retired
    assert opt.tell(batch[0], {"Creaminess": 5}) == pending[0]["formulation"]


def test_re_adding_a_setting_keeps_its_step():
    opt = _two()
    opt.add_process_parameter("Mix", 1, 10, step=0.5)
    opt.add_process_parameter("Mix", 2, 10)
    assert opt.variables[-1].step == 0.5
    opt.add_process_parameter("Mix", 2, 10, step=1)
    assert opt.variables[-1].step == 1


def test_ask_with_results_but_no_measurement_refused():
    opt = _two()
    _record(opt, 5)
    opt.remove_objective("Creaminess")
    with pytest.raises(ValueError) as err:
        opt.ask(2)
    assert str(err.value) == ("No measurement to learn from: add one with "
                              "add_objective() (5 formulations recorded)")
    assert opt.pending() == []


def test_ask_with_no_results_and_no_measurement_is_the_opening_design():
    opt = Optimizer(seed=0)
    opt.add_ingredient("Pea protein", 2, 8)
    assert len(opt.ask(2)) == 2
