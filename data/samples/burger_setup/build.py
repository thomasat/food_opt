"""Example setup - plant-based burger: how to set up a project, with no results.

Run with the devenv python, from anywhere:
    devenv/bin/python data/samples/burger_setup/build.py [OUT_DIR]
It writes "Example setup - plant-based burger.json" (default: beside this
file). tests/test_sample_projects.py checks the shipped file still matches.

The set-up is the 0.7.x burger sample's (sample_projects.build at 994ac14),
unchanged: calculated rows (Hydration water = 2.2 x the textured proteins,
Remaining water "= rest"), three pre-mixes, one process setting, three
measurements with targets and shares, a formulation total of 100 g, a fat
limit, a limit on the solids, a pre-mix smallest quantity, Lot and Actual
records, four bench records, where the targets came from and the method.

Every value is illustrative: no ingredient property, limit or target was
measured, and the project carries no results and no open Round (never fake
results). It carries no date either.
"""
import copy
import hashlib
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

import custom_records  # noqa: E402
from food_bo import FoodOptimizer  # noqa: E402

NAME = "Example setup - plant-based burger"
INGREDIENTS_CSV = os.path.join(HERE, "burger_ingredients.csv")

# The six rows that carry the solids. A formed plant-based patty is 55-65 %
# moisture; six bands moving independently add up to 25 g of swing on the
# dry side, and the row that fills to the total hands every gram of it to
# the water. Holding the solids between 36 and 43 g is how a formulator
# writes that down - and it is the only way to write it, because the two
# water rows are calculated and cannot carry a limit.
SOLIDS = ['Textured pea protein', 'Textured soy protein', 'Dry blend',
          'Wheat gluten', 'Coconut oil', 'Sunflower oil']
# Cook loss is (raw - cooked) / raw x 100, so the bench needs two weights to
# work it out and a place to write them; the two temperatures are what a
# methylcellulose system stands or falls on.
BENCH_RECORDS = ['Raw weight (g)', 'Cooked weight (g)',
                 'Water temperature at addition (°C)',
                 'Mass temperature out of the bowl (°C)']

# The example's own texts (0.7.x wording.SAMPLE_TARGETS_SOURCE and
# SAMPLE_METHOD). They are the project's data, shown wherever any project's
# targets note and method are shown; nothing in the app names them.
TARGETS_SOURCE = (
    "Scores are against an 80/20 beef control cooked to 71 °C core, which our panel rates "
    "firmness 6 and juiciness 7 on the anchored 0–10 scale (12 panellists, two sessions, "
    "February 2026). Above firmness 7 the patty eats rubbery; below juiciness 5 it eats dry "
    "and chalky. Contains soy and wheat (gluten)."
)
METHOD = (
    "1. Hydrate the textured pea and soy proteins in the Hydration water only "
    "(2.2 × their combined mass), 10 min at 45 °C, covered. Do not drain.\n"
    "2. Put the Remaining water in the bowl at ≤ 5 °C. Add the Dry blend, Wheat gluten "
    "and Seasoning blend and mix 60 s on low. (The dry blend carries the "
    "methylcellulose: it hydrates only cold.)\n"
    "3. Add the hydrated proteins. Mix 30 s on low.\n"
    "4. Add the fats and oils, coconut oil chopped to ~5 mm, and mix for this "
    "formulation's Mixing time after fat.\n"
    "5. Form 100 g patties, 100 mm across, 12 mm thick. Chill 20 min at 4 °C, covered.\n"
    "6. Griddle at 180 °C, 3 min per side, to 74 °C core. Serve within 3 min.\n"
    "Water temp at addition ____ °C (≤ 5)     Mass temp out of the bowl ____ °C (≤ 12)\n"
    "Raw weight ____ g     Cooked and blotted weight ____ g     "
    "Cook loss = (raw − cooked)/raw × 100\n"
    "Contains soy and wheat (gluten).\n"
    "Every formulation is made to the batch size printed above. These amounts and "
    "targets demonstrate the app; confirm a suitable preparation and measurement "
    "protocol before using them for real trials."
)


def setup(opt):
    """The 0.7.x burger set-up, step for step, on a blank project."""
    opt.set_amount_unit('g')
    opt.load_ingredients_from_csv(pd.read_csv(INGREDIENTS_CSV))
    # 5 s because that is what a planetary mixer's dial does; 109.04 s is a
    # number nobody can set.
    opt.add_process_parameter('Mixing time after fat', 45, 150, unit='s', step=5)
    opt.add_objective('Juiciness', 1, goal='target', target=7, min_val=0, max_val=10, unit='/10')
    opt.add_objective('Firmness', 1, goal='target', target=6, min_val=0, max_val=10, unit='/10')
    # 0-40, not 0-100: a real patty runs 15-25 % cook loss, and on a 0-100
    # range the whole interesting stretch moved the score by two points.
    opt.add_objective('Cook loss', 1, goal='min', min_val=0, max_val=40, unit='%')
    opt.set_shares({'Firmness': 45, 'Juiciness': 35, 'Cook loss': 20})
    opt.set_formulation_total(100)
    opt.add_constraint('Fat per 100 g', max_val=16)
    opt.add_chosen_quantity_constraint(SOLIDS, min_val=36, max_val=43)
    # 150 g, not the 100 g floor: at exactly 100 g a pre-mix page prints
    # Amount (g) and Composition (%) as the same column twice.
    opt.set_premix_smallest_quantity('Dry blend', 150)
    opt.set_records('lot', True)
    opt.set_records('actual', True)
    for field in BENCH_RECORDS:
        custom_records.add_field(opt, field, 'formulation')
    _steady_record_ids(opt)
    opt.set_targets_source(TARGETS_SOURCE)
    opt.set_method(METHOD)
    return opt


def _steady_record_ids(opt):
    """A bench record's id is a random uuid; the shipped file must equal a
    rebuild, so each id is made from the record's name instead (the project
    holds no values yet, so nothing refers to the old ids)."""
    data = copy.deepcopy(custom_records.state(opt))
    assert not data['values']
    for f in data['fields']:
        f['id'] = hashlib.md5(f"{f['scope']}:{f['name']}".encode("utf-8")).hexdigest()
    opt.custom_records = data
    opt.save()


def build(out_dir=None, name=NAME):
    """Build the project; LocalStorage writes <name>.pkl into the cwd, so
    chdir into out_dir for the build when it is given."""
    prev = os.getcwd()
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        os.chdir(out_dir)
    try:
        # A pkl left from an earlier run would be loaded, not rebuilt.
        if os.path.exists(f"{name}.pkl"):
            os.remove(f"{name}.pkl")
        opt = setup(FoodOptimizer(name))
        opt.save()
        return opt
    finally:
        os.chdir(prev)


if __name__ == "__main__":
    out = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else HERE)
    opt = build(out_dir=out)
    os.replace(os.path.join(out, f"{NAME}.pkl"), os.path.join(out, f"{NAME}.json"))
    print(f"{NAME}: {len(opt.variables)} rows, {len(opt.premixes)} pre-mixes, "
          f"{len(opt.objectives)} measurements, total {opt.formulation_total:g} g, "
          f"results {len(opt.X_history)}")
