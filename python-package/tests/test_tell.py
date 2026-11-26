"""tell() is permanent, so what it records must be what was made and
measured: every variable's amount, results on their scale."""
import pytest

from egbo import Optimizer


def _two():
    opt = Optimizer(seed=0)
    opt.add_ingredient("Pea protein", 2, 8)
    opt.add_ingredient("Coconut oil", 0, 5)
    opt.add_objective("Creaminess", weight=1)
    return opt


def test_result_outside_its_scale_refused():
    opt = _two()
    recipe = opt.ask(1)[0]
    with pytest.raises(ValueError) as err:
        opt.tell(recipe, {"Creaminess": 1500})
    assert str(err.value) == (
        "Creaminess: 1500 is outside its scale 0 to 10; set the scale with "
        "add_objective('Creaminess', ..., min_val=, max_val=)")
    with pytest.raises(ValueError, match="Creaminess: -4 is outside its "
                                         "scale 0 to 10"):
        opt.tell(recipe, {"Creaminess": -4})
    assert opt.history() == []
    opt.tell(recipe, {"Creaminess": 10})     # the ends are on the scale


def test_result_on_its_own_scale_accepted():
    opt = Optimizer(seed=0)
    opt.add_ingredient("Pea protein", 2, 8)
    opt.add_objective("Hardness", weight=1, min_val=200, max_val=2000)
    opt.tell({"Pea protein": 4}, {"Hardness": 1500})
    assert opt.history()[0]["score"] == pytest.approx(
        (1500 - 200) / 1800)


def test_fixed_variable_left_out_refused():
    opt = _two()
    opt.fix("Coconut oil", 2)
    batch = opt.ask(2)
    with pytest.raises(ValueError) as err:
        opt.tell({"Pea protein": batch[0]["Pea protein"]}, {"Creaminess": 7})
    assert str(err.value) == ("Missing amount for Coconut oil (2 variables: "
                              "Pea protein, Coconut oil)")
    assert [r["formulation"] for r in opt.pending()] == [1, 2]
    assert opt.tell(batch[0], {"Creaminess": 7}) == 1


def test_negative_amount_refused():
    opt = _two()
    with pytest.raises(ValueError, match="Pea protein: amount -3 is below 0; "
                                         "amounts start at 0"):
        opt.tell({"Pea protein": -3, "Coconut oil": 1}, {"Creaminess": 5})


def test_a_setting_may_be_negative():
    opt = _two()
    opt.add_process_parameter("Chill", -18, 4)
    opt.tell({"Pea protein": 3, "Coconut oil": 1, "Chill": -10},
             {"Creaminess": 5})
    assert len(opt.history()) == 1


@pytest.mark.parametrize("recipe, results, message", [
    ([5, 1], {"Creaminess": 5}, "recipe must be a dict of amounts by "
                                "variable name, not list"),
    ({"Pea protein": 3, "Coconut oil": 1}, [5],
     "results must be a dict of values by measurement name, not list"),
])
def test_non_dict_refused(recipe, results, message):
    with pytest.raises(ValueError, match=message):
        _two().tell(recipe, results)


def test_unknown_names_say_what_is_known():
    opt = _two()
    with pytest.raises(ValueError, match="'Pea protien' is not a variable; "
                                         "the 2 variables are Pea protein, "
                                         "Coconut oil"):
        opt.tell({"Pea protien": 3, "Coconut oil": 1}, {"Creaminess": 5})
    with pytest.raises(ValueError, match="'Creamines' is not a measurement; "
                                         "the 1 measurement is Creaminess"):
        opt.tell({"Pea protein": 3, "Coconut oil": 1}, {"Creamines": 5})


@pytest.mark.parametrize("value", [float("inf"), True])
def test_non_finite_amount_refused(value):
    with pytest.raises(ValueError, match="Pea protein: amount .* is not a "
                                         "finite number"):
        _two().tell({"Pea protein": value, "Coconut oil": 1}, {"Creaminess": 5})
