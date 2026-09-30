"""The owner's bench shape for Insights timings (perf.md §1.1): 8
ingredients plus Water filling to 100 g plus 2 process settings (10
varying rows), 3-6 measurements, 6 formulations per Round, results from a
fixed smooth truth with noise. Built with tell(), not ask(): quick, and
the same numbers every time."""
import torch

from food_bo import FoodOptimizer

EXTRA = [("Colour", 0.5, "max", None, 0, 10), ("Hardness", 0.6, "target", 5, 0, 10),
         ("Springiness", 0.4, "max", None, 0, 10)]
INGREDIENTS = [("Pea protein", 10, 20), ("Oil", 2, 8), ("Starch", 1, 5), ("Fibre", 0.5, 3),
               ("Salt", 0.5, 2), ("Methylcellulose", 0.5, 3), ("Yeast extract", 0.2, 1.5),
               ("Beet juice", 0.2, 2)]


def bench(n, m, per_round=6, seed=4, name=None):
    """`n` scored formulations with `m` measurements, in Rounds of `per_round`."""
    opt = FoodOptimizer(name or f"bench-{n}x{m}")
    opt.set_amount_unit("g")
    for k, lo, hi in INGREDIENTS:
        opt.add_ingredient(k, lo, hi)
    opt.add_ingredient("Water", 0, 100)
    opt.set_formulation_total(100)
    opt.set_formula("Water", "= rest")
    opt.add_process_parameter("Cook temperature", 60, 90, unit="°C")
    opt.add_process_parameter("Cook time", 5, 30, unit="min")
    opt.add_objective("Firmness", 1.5, goal="target", target=6, min_val=0, max_val=10, unit="N")
    opt.add_objective("Juiciness", 1.0, goal="max", min_val=0, max_val=10, unit="/10")
    opt.add_objective("Cook loss", 1.0, goal="min", min_val=0, max_val=40, unit="%")
    for extra, w, goal, target, lo, hi in EXTRA[:max(0, m - 3)]:
        opt.add_objective(extra, w, goal=goal, target=target, min_val=lo, max_val=hi, unit="/10")
    opt.add_quantity_constraint(["Oil", "Starch"], max_val=11)
    g = torch.Generator().manual_seed(seed)
    for no in range(1, n + 1):
        tell(opt, no, g, m, per_round)
    return opt


def tell(opt, no, g, m, per_round=6):
    """Record formulation `no` (Round (no - 1) // per_round + 1)."""
    u = torch.rand(10, generator=g).tolist()
    rec = {k: lo + (hi - lo) * u[i] for i, (k, lo, hi) in enumerate(INGREDIENTS)}
    if rec["Oil"] + rec["Starch"] > 11:
        rec["Oil"] = 11 - rec["Starch"] - 0.1
    rec["Cook temperature"] = 60 + 30 * u[8]
    rec["Cook time"] = 5 + 25 * u[9]
    e = torch.randn(6, generator=g).tolist()
    res = {"Firmness": 0.4 * rec["Pea protein"] + 0.3 * rec["Starch"] + 0.3 * e[0],
           "Juiciness": min(10, 1 + 1.0 * rec["Oil"] + 0.3 * e[1]),
           "Cook loss": max(0, 30 - 0.5 * rec["Pea protein"] - 2 * rec["Methylcellulose"]
                            + 0.1 * rec["Cook temperature"] + e[2])}
    extra = [min(10, max(0, 2 + 2 * rec["Beet juice"] + 0.3 * e[3])),
             min(10, max(0, 2 + 0.3 * rec["Pea protein"] - 0.4 * rec["Oil"] + 0.3 * e[4])),
             min(10, max(0, 3 + 1.2 * rec["Methylcellulose"] + 0.2 * rec["Fibre"] + 0.3 * e[5]))]
    for (extra_name, *_), v in zip(EXTRA[:max(0, m - 3)], extra):
        res[extra_name] = v
    opt.tell(opt.fill_formulas(rec), res, formulation_no=no, batch_no=1 + (no - 1) // per_round)
