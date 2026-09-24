"""One guided teaching example; no measured results or validated recipe."""
from pathlib import Path
import pandas as pd
import custom_records
from food_bo import FoodOptimizer
import wording

OPTIONS = ['Burger formulation']
NAMES = {OPTIONS[0]: wording.SAMPLE_PROJECT_NAME}

# The six rows that carry the solids. A formed plant-based patty is 55-65 %
# moisture; six bands moving independently add up to 25 g of swing on the
# dry side, and the row that fills to the total hands every gram of it to
# the water. Holding the solids between 36 and 43 g is how a formulator
# writes that down — and it is the only way to write it, because the two
# water rows are calculated and cannot carry a limit.
SOLIDS = ['Textured pea protein', 'Textured soy protein', 'Dry blend',
          'Wheat gluten', 'Coconut oil', 'Sunflower oil']
# Cook loss is (raw − cooked) / raw × 100, so the bench needs two weights to
# work it out and a place to write them; the two temperatures are what a
# methylcellulose system stands or falls on.
BENCH_RECORDS = ['Raw weight (g)', 'Cooked weight (g)',
                 'Water temperature at addition (°C)',
                 'Mass temperature out of the bowl (°C)']


def build(kind, name, fresh_storage, storage):
    if kind not in OPTIONS:
        raise ValueError('Choose the burger teaching example.')
    opt = FoodOptimizer(name, storage=fresh_storage)
    opt.storage = storage
    opt.set_amount_unit('g')
    opt.load_ingredients_from_csv(pd.read_csv(Path(__file__).parent / 'data' / 'sample_ingredients.csv'))
    # 5 s because that is what a planetary mixer's dial does; 109.04 s is a
    # number nobody can set.
    opt.add_process_parameter('Mixing time after fat', 45, 150, unit='s', step=5)
    opt.add_objective('Juiciness', 1, goal='target', target=7, min_val=0, max_val=10, unit='/10')
    opt.add_objective('Firmness', 1, goal='target', target=6, min_val=0, max_val=10, unit='/10')
    # 0-40, not 0-100: a real patty runs 15-25 % cook loss, and on a 0-100
    # scale the whole interesting range moved the score by two points.
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
    opt.set_targets_source(wording.SAMPLE_TARGETS_SOURCE)
    opt.set_method(wording.SAMPLE_METHOD)
    return opt
