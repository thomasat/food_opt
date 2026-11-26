"""Inputs that would be silently changed are refused; state is left alone."""
import json
import os

import pytest
import torch

from egbo import Optimizer


def _two(seed=0, **kwargs):
    opt = Optimizer(seed=seed, **kwargs)
    opt.add_ingredient("Pea protein", 0, 5)
    opt.add_ingredient("Oil", 0, 5)
    opt.add_objective("Taste", weight=1)
    return opt


def _taste(recipe):
    value = 10 - (recipe["Pea protein"] - 3) ** 2 - (recipe.get("Oil", 0) - 1) ** 2
    return {"Taste": max(0.0, min(10.0, value))}


def _warm(opt):
    for recipe in opt.ask(5):
        opt.tell(recipe, _taste(recipe))


def _with_total():
    opt = Optimizer(seed=0)
    opt.add_ingredient("A", 0, 60)
    opt.add_ingredient("B", 0, 60)
    opt.add_objective("Taste", weight=1)
    opt.set_formulation_total(100)
    opt.ask(2)
    return opt


# 1
def test_range_change_making_total_unreachable_refused():
    opt = _with_total()
    before = opt.to_dict()
    with pytest.raises(ValueError, match="100") as err:
        opt.set_range("B", 0, 20)
    assert "0 to 80" in str(err.value)
    assert "clear_formulation_total()" in str(err.value)
    assert opt.to_dict() == before
    with pytest.raises(ValueError, match="clear_formulation_total"):
        opt.fix("B", 10)
    assert opt.to_dict() == before


def test_range_change_allowed_after_clearing_total():
    opt = _with_total()
    opt.clear_formulation_total()
    opt.set_range("B", 0, 20)
    assert opt.formulation_total is None


# 2
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True])
def test_property_values_must_be_finite_numbers(bad):
    opt = _two()
    with pytest.raises(ValueError, match="Oil.*Protein|Protein.*Oil"):
        opt.set_property_values("Protein", {"Oil": bad})
    assert opt.properties == {}


# 3
def test_remove_unknown_objective_refused():
    opt = _two()
    opt.add_objective("Texture", weight=3)
    before = [o.weight for o in opt.objectives]
    with pytest.raises(ValueError, match="No measurement named 'Smell'"):
        opt.remove_objective("Smell")
    assert [o.weight for o in opt.objectives] == before


# 4
def test_tell_missing_no_baseline_setting_refused():
    opt = _two()
    opt.add_process_parameter("Bake", 10, 20)
    with pytest.raises(ValueError, match="Missing value for 'Bake'"):
        opt.tell({"Pea protein": 1.0}, {"Taste": 5})
    assert opt.history() == [] and opt.next_formulation_no == 1
    opt.tell({"Pea protein": 1.0, "Bake": 12.0}, {"Taste": 5})
    assert len(opt.history()) == 1


# 5
@pytest.mark.parametrize("config", [
    {"kernal": "rbf"}, {"kernel": "rbff"}, {"acquisition": "ei"},
    {"noise": "loud"}, {"lengthscale_prior": 3}, ["kernel"],
])
def test_bo_config_typos_refused(config):
    with pytest.raises(ValueError):
        Optimizer(bo_config=config)


def test_bo_config_case_and_space_accepted():
    opt = Optimizer(bo_config={"kernel": " RBF "})
    assert opt.bo_config["kernel"] == "rbf"


# 10
def test_n_init_below_one_refused():
    with pytest.raises(ValueError, match="n_init"):
        Optimizer(n_init=0)


@pytest.mark.parametrize("n", [0, -1, 1.5, True, "2"])
def test_ask_count_must_be_a_positive_integer(n):
    opt = _two()
    with pytest.raises(ValueError):
        opt.ask(n)
    assert opt.pending() == []


def test_ask_accepts_integral_float():
    assert len(_two().ask(2.0)) == 2


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_tell_result_must_be_finite(bad):
    opt = _two()
    with pytest.raises(ValueError, match="Taste"):
        opt.tell({"Pea protein": 1.0}, {"Taste": bad})
    assert opt.history() == []


# 11
def test_save_is_atomic(tmp_path, monkeypatch):
    opt = _two()
    path = tmp_path / "o.json"
    opt.save(path)
    original = path.read_text(encoding="utf-8")
    opt.ask(1)

    def boom(*args):
        raise OSError("disk gone")
    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        opt.save(path)
    assert path.read_text(encoding="utf-8") == original
    assert [p.name for p in tmp_path.iterdir()] == ["o.json"]


def test_from_dict_missing_key_named():
    data = _two().to_dict()
    del data["recipes"]
    with pytest.raises(ValueError, match="egbo file is missing 'recipes'"):
        Optimizer.from_dict(data)


@pytest.mark.parametrize("version", [True, 0, -1, 1.0, "1"])
def test_from_dict_format_version_must_be_a_positive_int(version):
    data = _two().to_dict()
    data["format_version"] = version
    with pytest.raises(ValueError):
        Optimizer.from_dict(data)


def test_saved_file_still_loads(tmp_path):
    path = tmp_path / "o.json"
    _two().save(path)
    assert json.loads(path.read_text(encoding="utf-8"))["format"] == "egbo"
    assert Optimizer.load(path).to_dict() == _two().to_dict()


# 12
def test_refused_property_limit_replacement_keeps_previous():
    opt = _two()
    opt.set_property_values("Protein", {"Pea protein": 80, "Oil": 0})
    opt.add_constraint("Protein", min_val=1)
    opt.fix("Oil", 3)
    before = opt.to_dict()
    with pytest.raises(ValueError):
        opt.add_constraint("Protein", min_val=60)
    assert opt.to_dict() == before
    assert [(p.metric, p.min_val) for p in opt.property_limits] == [("Protein", 1.0)]


# 13
def test_warm_ask_leaves_global_torch_rng_alone():
    opt = _two()
    _warm(opt)
    torch.manual_seed(1234)
    before = torch.get_rng_state()
    opt.ask(2)
    assert torch.equal(torch.get_rng_state(), before)


# 14
def test_limits_leave_the_open_batch_alone():
    opt = _two()
    opt.set_property_values("Protein", {"Pea protein": 80, "Oil": 0})
    opt.ask(3)
    before = opt.pending()
    opt.add_quantity_constraint(["Pea protein", "Oil"], max_val=9)
    assert opt.pending() == before
    opt.add_constraint("Protein", min_val=1)
    assert opt.pending() == before


@pytest.mark.xfail(strict=True, reason=(
    "The app's opening-Round sampler (food_bo.py _sample_feasible) lands on "
    "the all-zero formulation, the one point a min-only property limit no "
    "ingredient can reach still admits, and hands back a Round of 1 with it "
    "instead of refusing. egbo mirrors the app for parity."))
def test_impossible_property_limit_says_why_at_cold_start():
    opt = _two()
    opt.set_property_values("Protein", {"Pea protein": 0.8, "Oil": 0})
    opt.add_constraint("Protein", min_val=3)
    with pytest.raises(ValueError) as err:
        opt.ask(2)
    assert str(err.value) == (
        "No formulation meets all the limits. The limit on Protein cannot be "
        "met: the amounts allowed only reach 0.8 per 100 g at most")


def test_impossible_quantity_limit_says_why_after_the_first_rounds():
    opt = _two()
    for recipe in opt.ask(5):
        opt.tell(recipe, _taste(recipe))
    opt.add_quantity_constraint(["Pea protein", "Oil"], min_val=20)
    with pytest.raises(ValueError) as err:
        opt.ask(1)
    message = str(err.value)
    assert message.startswith("No formulation meets all the limits. ")
    assert len(message) > len("No formulation meets all the limits. ")
