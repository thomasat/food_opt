# egbo

Suggests formulations to try next, from your tasting results. You choose what varies.

## Install

Python 3.11 or newer:

```
pip install "git+https://github.com/thomasat/food_opt#subdirectory=python-package"
```

To match the desktop app's suggestions exactly (needs Python 3.12 or newer):

```
pip install -c https://raw.githubusercontent.com/thomasat/food_opt/main/python-package/egbo-pins.txt "git+https://github.com/thomasat/food_opt#subdirectory=python-package"
```

## A round

```python
from egbo import Optimizer

opt = Optimizer(seed=0)
opt.add_ingredient("Pea protein", 2, 8)       # name, lowest, highest
opt.add_ingredient("Coconut oil", 0, 5)
opt.add_objective("Creaminess", weight=2)     # 0–10 scale, higher is better
opt.add_objective("Sweetness", weight=1, goal="target", target=6)

batch = opt.ask(3)                            # 3 formulations to make
tasted = [{"Creaminess": 7, "Sweetness": 5},  # one result per formulation
          {"Creaminess": 4, "Sweetness": 8},
          {"Creaminess": 6, "Sweetness": 6}]
for recipe, result in zip(batch, tasted):
    opt.tell(recipe, result)
opt.save("pilot.json")

opt = Optimizer.load("pilot.json")            # next session
opt.fix("Coconut oil", 2)                     # expert step: stop varying it
batch = opt.ask(3)
```

## Methods

| Method | Does |
|---|---|
| `add_ingredient(name, min_val, max_val, baseline=)` | Add an ingredient, or change its range |
| `add_process_parameter(name, min_val, max_val, step=)` | Add a setting, e.g. mix time |
| `add_objective(name, weight, goal=, target=, min_val=, max_val=)` | Goal `"max"`, `"min"` or `"target"` |
| `fix(name, value)` / `set_range(name, min_val, max_val)` | Stop varying / vary again |
| `add_quantity_constraint([names], min_val=, max_val=)` | Limit a sum of amounts |
| `set_formulation_total(total)` | All ingredients add up to this |
| `set_property_values(metric, {ingredient: value})` | Values per 100 g |
| `add_constraint(metric, min_val=, max_val=)` | Limit a property of the formulation |
| `ask(n)` / `tell(recipe, results)` | Get a batch / record one result |
| `history()` / `pending()` | Recorded rows / suggestions not yet recorded |
| `finalists(k=3)` | Positions in `history()` of the k rows predicted best |
| `save(path)` / `Optimizer.load(path)` | Keep the project in a JSON file |

## Good to know

- Score: each measurement counts 0–1 on its scale (`min_val`..`max_val`, default 0–10), times its weight. A result off its scale is refused.
- `tell()` can't be undone. Re-running that notebook cell records the rows again.
- `ask()` again replaces the suggestions not yet recorded.
- Rounded amounts are fine: a recipe that matches no suggestion counts as your own formulation.
- A property limit applies to the mass-weighted average, per 100 g.

## Tests

From the repository root:

```
pip install -e "python-package[test]" && pytest python-package/tests
```

MIT license.
