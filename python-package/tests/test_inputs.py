"""Setup inputs: every number is a finite number, arguments after max_val
are keywords, and unknown names say what is known."""
import json
import math

import pytest

from egbo import Optimizer

NAN, INF = float("nan"), float("inf")


def _two():
    opt = Optimizer(seed=0)
    opt.add_ingredient("Pea protein", 2, 8)
    opt.add_ingredient("Coconut oil", 0, 5)
    opt.add_objective("Creaminess", weight=1)
    return opt


@pytest.mark.parametrize("call, message", [
    (lambda o: o.add_ingredient("X", NAN, 8), "X: min nan"),
    (lambda o: o.add_ingredient("X", 0, INF), "X: max inf"),
    (lambda o: o.add_ingredient("X", True, 8), "X: min True"),
    (lambda o: o.add_ingredient("X", "1", 8), "X: min '1'"),
    (lambda o: o.add_process_parameter("Mix", 1, NAN), "Mix: max nan"),
    (lambda o: o.add_process_parameter("Mix", 1, 5, step=NAN), "Mix: step nan"),
    (lambda o: o.fix("Pea protein", NAN), "Pea protein: min nan"),
    (lambda o: o.set_range("Coconut oil", 0, INF), "Coconut oil: max inf"),
    (lambda o: o.add_objective("Sweet", NAN), "Sweet: weight nan"),
    (lambda o: o.add_objective("Sweet", 1, min_val=NAN), "Sweet: scale min nan"),
    (lambda o: o.add_objective("Sweet", 1, max_val=INF), "Sweet: scale max inf"),
    (lambda o: o.add_objective("Sweet", 1, goal="target", target=NAN),
     "Sweet: target nan"),
    (lambda o: o.add_quantity_constraint(["Pea protein"], max_val=NAN),
     "Pea protein: limit max nan"),
    (lambda o: o.add_total_mass_constraint(min_val=INF),
     "Pea protein \\+ Coconut oil: limit min inf"),
    (lambda o: o.set_formulation_total(NAN), "Formulation total nan"),
    (lambda o: o.add_constraint("Protein", min_val=NAN), "Protein: limit min nan"),
    (lambda o: o.set_property_values("Protein", {"Pea protein": INF}),
     "Pea protein: Protein value inf"),
])
def test_non_finite_numbers_refused(call, message):
    opt = _two()
    before = opt.to_dict()
    with pytest.raises(ValueError, match=message + " is not a finite number"):
        call(opt)
    assert opt.to_dict() == before


@pytest.mark.parametrize("step", [0, -1])
def test_step_must_be_above_zero(step):
    opt = _two()
    with pytest.raises(ValueError, match=f"Mix: step {step} must be above 0"):
        opt.add_process_parameter("Mix", 1, 5, step=step)


def test_baseline_must_be_finite():
    opt = _two()
    for r in opt.ask(2):
        opt.tell(r, {"Creaminess": 5})
    with pytest.raises(ValueError, match="Sugar: baseline nan is not a "
                                         "finite number"):
        opt.add_ingredient("Sugar", 1, 3, baseline=NAN)
    with pytest.raises(ValueError, match="Mix: baseline inf is not a "
                                         "finite number"):
        opt.add_process_parameter("Mix", 1, 5, baseline=INF)


def test_arguments_after_max_are_keywords():
    opt = _two()
    with pytest.raises(TypeError):
        opt.add_ingredient("Sugar", 1, 5, 3)
    with pytest.raises(TypeError):
        opt.add_process_parameter("Mix", 1, 10, 0.5)


def test_one_name_for_a_quantity_limit():
    opt = _two()
    opt.add_quantity_constraint("Pea protein", max_val=5)
    assert opt.quantity_limits[-1].ingredients == ["Pea protein"]


def test_unknown_names_say_what_is_known():
    opt = _two()
    with pytest.raises(ValueError, match="No ingredient named 'Pea protien'; "
                                         "the 2 ingredients are Pea protein, "
                                         "Coconut oil"):
        opt.add_quantity_constraint(["Pea protien"], max_val=5)
    with pytest.raises(ValueError, match="No ingredient named 'Sugar'; the 2 "
                                         "ingredients are"):
        opt.set_property_values("Protein", {"Sugar": 1})
    with pytest.raises(ValueError, match="No variable named 'Salt'; the 2 "
                                         "variables are Pea protein, Coconut "
                                         "oil"):
        opt.fix("Salt", 1)
    with pytest.raises(ValueError, match="No measurement named 'Creamines'; "
                                         "the 1 measurement is Creaminess"):
        opt.remove_objective("Creamines")


@pytest.mark.parametrize("seed", [1.7, "abc", True, NAN])
def test_seed_is_a_whole_number(seed):
    with pytest.raises(ValueError, match="seed must be a whole number"):
        Optimizer(seed=seed)
    assert Optimizer(seed=7.0).sobol_seed == 7


def test_add_objective_returns_none():
    opt = _two()
    assert opt.add_objective("Creaminess", weight=2) is None
    assert opt.add_objective("Sweet", weight=1) is None


def test_save_refuses_a_non_finite_number(tmp_path):
    opt = _two()
    opt.variables[0].high = INF           # past every check
    with pytest.raises(ValueError):
        opt.save(tmp_path / "o.json")
    assert not (tmp_path / "o.json").exists()
