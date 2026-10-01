"""The process-only demo project (Insights for a first-time user, spec
2026-10-01 rev. 2; spec-review I17, R-I17): none of the six demo projects
is a study of process settings alone, so the first-screen pins had nothing
to say "Trial 15" on. This builds one, with the demo projects' shape (Line
A's: Rounds of 5, 4, 4, 4 and 3, a Repeat of Best so far in Round 5, Round
6 open with 4 suggestions) and the generation records the Generate call
site keeps (ui_batch.ask_alone), so Round 6 is a warm Round of generated
picks and "Why the model suggested Round 6" has somewhere to land.

It is run once, by hand, and the JSON it writes is the fixture:

    python -m tests.process_only_fixture

Every number comes from random.Random(SEED), the project's sobol_seed and
ask(), which seeds itself from the next formulation number; nothing depends
on the clock but the timestamps, which no test reads. The tests load the
saved file (tests/demo_projects.py), never this builder: a rebuild under
another BoTorch could pick other amounts, and a fixture is never
regenerated."""
import json
import os
import pathlib
import random
import tempfile

import bo_views
import insights
import wording
from food_bo import FoodOptimizer

NAME = "Oven trial - process settings"
SEED = 20261001
OUT = pathlib.Path(__file__).parent / "fixtures" / "demo_projects"


def setup():
    opt = FoodOptimizer(NAME)
    opt.sobol_seed = SEED
    opt.add_process_parameter("Bake temperature", 160, 220, unit="°C", step=5)
    opt.add_process_parameter("Bake time", 8, 20, unit="min", step=1)
    opt.add_process_parameter("Proof time", 30, 90, unit="min", step=5)
    opt.add_objective("Crumb softness", 40, goal="target", target=7, min_val=0, max_val=10,
                      unit="/10")
    opt.add_objective("Crust colour", 35, goal="target", target=6, min_val=0, max_val=10,
                      unit="/10")
    opt.add_objective("Moisture loss", 25, goal="min", min_val=0, max_val=30, unit="%")
    opt.set_shares({"Crumb softness": 40, "Crust colour": 35, "Moisture loss": 25})
    opt.set_targets_source("Benchmark: the current retail loaf, rated by the same panel")
    opt.set_method("Same dough each trial; proof, then bake one 500 g loaf; rate after 2 h")
    return opt


def truth(r):
    """Panel means a baker would find plausible."""
    t, b, p = r["Bake temperature"], r["Bake time"], r["Proof time"]
    soft = 7.6 - 0.0025 * (t - 190) ** 2 - 0.05 * (b - 13) ** 2 + 0.015 * (p - 45)
    colour = 2.0 + 0.045 * (t - 160) + 0.18 * (b - 8)
    loss = 6.0 + 0.06 * (t - 160) + 0.55 * (b - 8) - 0.01 * (p - 30)
    return soft, colour, loss


def _clip(v, lo, hi):
    return min(hi, max(lo, v))


def panel(rng, recipe):
    soft, colour, loss = truth(recipe)
    return {"Crumb softness": round(_clip(soft + rng.gauss(0, 0.3), 0, 10), 1),
            "Crust colour": round(_clip(colour + rng.gauss(0, 0.3), 0, 10), 1),
            "Moisture loss": round(_clip(loss + rng.gauss(0, 0.8), 0, 30), 1)}


def generate(opt, n):
    """ask(n) as the Generate call site makes it (ui_batch.ask_alone): the
    generation record first, under RNG_LOCK, its picks once ask() returns."""
    with insights.RNG_LOCK:
        record = bo_views.generation_record(opt, n)
        opt.ask(n_suggestions=n)
    record["picks"] = [int(row["formulation"]) for row in opt.pending_batch or []]
    opt.record_generation(opt.pending_batch_no, record)


def record_round(opt, rng, n, repeat_best=False):
    generate(opt, n)
    if repeat_best:
        best = opt.best_index()
        # The note Start from Best so far writes; the screen says it
        # through for_project ("Repeat of Trial 14").
        opt.add_to_pending_batch(dict(opt.recipe_history[best]),
                                 note=wording.repeat_of_formulation(opt.formulation_ids[best]))
    for row in list(opt.pending_batch):
        opt.tell(row["recipe"], panel(rng, row["recipe"]), formulation_no=row["formulation"],
                 note=row.get("note"))
    opt.set_pending_batch(None)


def build():
    rng = random.Random(SEED)
    opt = setup()
    for r, n in enumerate([5, 4, 4, 4, 3], start=1):
        record_round(opt, rng, n, repeat_best=(r == 5))
    generate(opt, 4)             # Round 6, open, nothing recorded
    return opt


def main():
    here = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="process-only-") as folder:
        os.chdir(folder)        # LocalStorage writes <name>.pkl into the cwd
        try:
            opt = build()
            state = opt.export_json()
        finally:
            os.chdir(here)
    path = OUT / f"{NAME}.json"
    path.write_text(json.dumps(state, indent=2))
    print(path, "rows", len(opt.Y_history), "open", len(opt.pending_batch or []))


if __name__ == "__main__":
    main()
