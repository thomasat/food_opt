"""Generate records the numbers ask() issued (0.8.0 BO views, T3 I-2): the
Generate call site keeps the picks, in order, and they survive a save."""
import pytest

from food_bo import FoodOptimizer


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _truth(r):
    return {"Firmness": 2 + 0.9 * r["Pea"] - 0.2 * r["Oil"],
            "Juiciness": 1 + 1.2 * r["Oil"] - 0.05 * r["Pea"] ** 2 / 4}


def _project(name, bo_config=None, robust=False, rows=7):
    """Pea 4-8, Oil 2-6, Water filling to 100 g; `rows` formulations told."""
    opt = FoodOptimizer(name, robust=robust)
    opt.set_amount_unit("g")
    opt.add_ingredient("Pea", 4, 8)
    opt.add_ingredient("Oil", 2, 6)
    opt.add_ingredient("Water", 0, 100)
    opt.add_objective("Firmness", 50, goal="target", target=6, min_val=0, max_val=10)
    opt.add_objective("Juiciness", 30, goal="max", min_val=0, max_val=10)
    opt.set_formulation_total(100)
    opt.set_formula("Water", "= rest")
    if bo_config:
        opt.set_bo_config(bo_config)
    for k in range(rows):
        r = opt.fill_formulas({"Pea": 4 + (k * 0.37) % 4, "Oil": 2 + (k * 0.61) % 4})
        opt.tell(r, _truth(r), formulation_no=k + 1, batch_no=k // 3 + 1)
    return opt



def test_generate_records_the_numbers_ask_issued():
    """T3 I-2: the Generate call site keeps the picks, in order, and they
    survive a save."""
    import ui_batch
    opt = _project("picks_recorded")
    ui_batch.ask_alone(opt, n_suggestions=3)
    record = opt.round_generation[opt.pending_batch_no]
    assert record["picks"] == [int(r["formulation"]) for r in opt.pending_batch]
    again = FoodOptimizer("picks_recorded")
    assert again.round_generation[opt.pending_batch_no]["picks"] == record["picks"]
