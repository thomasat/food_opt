"""Three rounds of egbo against a toy objective."""

from egbo import Optimizer


def taste(recipe):
    """Stand-in for a tasting: peaks at 5 g pea protein, 2 g coconut oil."""
    score = 10 - (recipe["Pea protein"] - 5) ** 2 - 2 * (recipe["Coconut oil"] - 2) ** 2
    return {"Creaminess": max(0.0, score)}


opt = Optimizer(seed=0)
opt.add_ingredient("Pea protein", 2, 8)
opt.add_ingredient("Coconut oil", 0, 5)
opt.add_objective("Creaminess", weight=1, goal="max", min_val=0, max_val=10)
opt.add_quantity_constraint(["Pea protein", "Coconut oil"], max_val=10)

for round_no in range(1, 4):
    for recipe in opt.ask(4):
        opt.tell(recipe, taste(recipe))
    print(f"Round {round_no}: {len(opt.history())} formulations so far")

best = max(opt.history(), key=lambda row: row["score"])
print("Best:", {k: round(v, 2) for k, v in best["recipe"].items()},
      "score", round(best["score"], 3))

# The expert step: stop varying coconut oil, keep the history.
opt.fix("Coconut oil", 2)
print("Now searching over:", opt.active_variables())
opt.save("pilot.json")
opt = Optimizer.load("pilot.json")
