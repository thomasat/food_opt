# egbo

![license: MIT](https://img.shields.io/badge/license-MIT-blue)

Expert-guided batch Bayesian optimization of formulation recipes.

- You pick which ingredients and settings vary; the optimizer searches only those.
- Each round it suggests a batch of formulations; you taste or measure them.
- You tell it the results; it suggests the next batch.
- Between rounds you can fix an ingredient, widen a range or add a new one.
- Every formulation and result is kept in the history and in the saved file.

## Install

```
pip install "git+https://github.com/thomasat/food_opt#subdirectory=python-package"
```

For a smaller CPU-only torch, add `--extra-index-url https://download.pytorch.org/whl/cpu`.

For suggestions identical to the desktop app, install with its pinned versions
(download `egbo-pins.txt` from `python-package/` first):

```
pip install -c egbo-pins.txt "git+https://github.com/thomasat/food_opt#subdirectory=python-package"
```

Without the pins, suggestions can differ slightly.

## Quickstart

```python
from egbo import Optimizer

opt = Optimizer(seed=0)
opt.add_ingredient("Pea protein", 2, 8)
opt.add_ingredient("Coconut oil", 0, 5)
opt.add_objective("Creaminess", weight=1, goal="max", min_val=0, max_val=10)
opt.add_quantity_constraint(["Pea protein", "Coconut oil"], max_val=10)

batch = opt.ask(4)                        # list of {name: amount}
opt.tell(batch[0], {"Creaminess": 8.5})   # one result per measurement

opt.fix("Coconut oil", 2)                 # leaves the search, history kept
opt.set_range("Coconut oil", 0, 5)        # back in
opt.pending()                             # suggestions not yet recorded
opt.history()                             # recipe, results and score per row
opt.save("pilot.json"); opt = Optimizer.load("pilot.json")
```

A runnable version is in `examples/quickstart.py`.

## The open batch

- `ask()` holds its rows as the open batch until you `tell` them.
- A second `ask()` discards the rows still open and returns a different batch.
- Re-running a notebook cell that calls `ask()` therefore gives a new batch.
- Round the amounts and `tell` the rounded recipe: it matches no open row, so it
  counts as your own formulation.

## Seeding

`seed` sets the opening (first, Sobol) batch only. Later suggestions depend on
the data, not on `seed`. With `seed=None` a random one is drawn and saved.

## Not included

Calculations, pre-mixes, Excel workbooks and the app's project files.

## License

MIT, see `LICENSE`.
