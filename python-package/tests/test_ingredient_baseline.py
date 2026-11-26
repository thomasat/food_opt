"""An ingredient added mid-run at the amount earlier formulations used:
add_ingredient(..., baseline=). The past rows are read at the baseline by
every reader; the stored recipes are never rewritten."""
import pytest

from egbo import Optimizer
from egbo._limits import QuantityLimit, at_total, check, off_total
from egbo._space import encode, ingredient_total, search_bounds


def _taste(r):
    return {"Taste": max(0.0, min(10.0, 10 - ((r["A"] - 30) / 10) ** 2
                                  - ((r["B"] - 20) / 10) ** 2))}


def _project(rows=5):
    opt = Optimizer(seed=11)
    opt.add_ingredient("A", 0, 60)
    opt.add_ingredient("B", 0, 50)
    opt.add_objective("Taste", 1)
    for r in opt.ask(rows):
        opt.tell(r, _taste(r))
    return opt


def test_past_rows_read_at_the_baseline():
    opt = _project()
    before = [dict(r) for r in opt.recipes]
    opt.add_ingredient("Sorbate", 0.5, 1.2, baseline=0.8)
    var = opt.variables[-1]
    assert (var.low, var.high, var.absent) == (0.5, 1.2, 0.8)
    assert [encode(opt.variables, r)[-1] for r in opt.recipes] == [0.8] * 5
    # Inside the range, so the frame is not widened down to 0.
    assert search_bounds(opt.variables, opt.recipes, [])[-1] == (0.5, 1.2)
    assert opt.recipes == before
    rows = opt.ask(3)
    assert all(0.5 - 1e-9 <= r["Sorbate"] <= 1.2 + 1e-9 for r in rows)


def test_every_reader_counts_the_baseline():
    opt = _project()
    opt.add_ingredient("Sorbate", 0.5, 1.2, baseline=0.8)
    past = opt.recipes[0]
    assert "Sorbate" not in past
    own = past["A"] + past["B"]
    assert ingredient_total(opt.variables, past) == pytest.approx(own + 0.8)
    scaled = at_total(opt.variables, past, 2 * (own + 0.8))
    assert scaled["Sorbate"] == pytest.approx(1.6)
    assert off_total(opt.variables, past, own) is True
    assert not off_total(opt.variables, past, own + 0.8)
    limits = [QuantityLimit(["Sorbate"], min_val=0.7)]
    assert check(opt.variables, limits, [], {}, past)
    assert not check(opt.variables, [QuantityLimit(["Sorbate"], max_val=0.7)],
                     [], {}, past)


def test_baseline_outside_the_range_refused():
    opt = _project()
    with pytest.raises(ValueError, match="Sorbate: baseline 0.3 is outside "
                                         "0.5 to 1.2"):
        opt.add_ingredient("Sorbate", 0.5, 1.2, baseline=0.3)
    assert "Sorbate" not in [v.name for v in opt.variables]


def test_baseline_before_any_result_refused():
    opt = Optimizer(seed=1)
    with pytest.raises(ValueError, match="Sorbate: baseline 0.8 .* 0 "
                                         "formulations are recorded"):
        opt.add_ingredient("Sorbate", 0.5, 1.2, baseline=0.8)
    opt.add_ingredient("Sorbate", 0.5, 1.2, baseline=0)    # 0 is none
    assert opt.variables[-1].absent == 0.0
    with pytest.raises(ValueError, match="Sorbate: baseline 0.9 .* 0 "
                                         "formulations are recorded"):
        opt.add_ingredient("Sorbate", 0.5, 1.2, baseline=0.9)   # existing


def test_baseline_of_zero_is_none():
    opt = _project()
    with pytest.warns(UserWarning, match="Salt: min 0.5 lowered to 0"):
        opt.add_ingredient("Salt", 0.5, 1.2, baseline=0)
    assert (opt.variables[-1].low, opt.variables[-1].absent) == (0.0, 0.0)
    opt.add_ingredient("Sugar", 0.5, 1.2, keep_lowest=True, baseline=0)
    assert (opt.variables[-1].low, opt.variables[-1].absent) == (0.5, 0.0)


def test_baseline_on_an_existing_ingredient():
    opt = _project()
    with pytest.raises(ValueError) as err:
        opt.add_ingredient("A", 0, 60, baseline=10)
    assert str(err.value) == ("A: baseline 10 refused; all 5 recorded "
                              "formulations hold their own A amounts")
    opt.add_ingredient("Sorbate", 0.5, 1.2, baseline=0.8)
    opt.add_ingredient("Sorbate", 0.5, 1.2, baseline=0.9)      # corrected
    assert opt.variables[-1].absent == 0.9
    with pytest.raises(ValueError, match="outside 0.5 to 1.2"):
        opt.add_ingredient("Sorbate", 0.5, 1.2, baseline=2)
    opt.add_ingredient("Sorbate", 0.5, 1.2)                     # kept
    assert opt.variables[-1].absent == 0.9
    opt.add_ingredient("Sorbate", 0.5, 1.2, baseline=0)        # cleared
    assert opt.variables[-1].absent == 0.0


def test_baseline_survives_save_and_load(tmp_path):
    opt = _project()
    opt.add_ingredient("Sorbate", 0.5, 1.2, baseline=0.8)
    path = tmp_path / "o.json"
    opt.save(path)
    loaded = Optimizer.load(path)
    assert loaded.variables[-1].absent == 0.8
    assert Optimizer.from_dict(opt.to_dict()).to_dict() == opt.to_dict()
    assert loaded.ask(2) == opt.ask(2)
