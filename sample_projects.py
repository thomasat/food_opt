"""One guided teaching example; no measured results or validated recipe."""
from pathlib import Path
import pandas as pd
from food_bo import FoodOptimizer
import wording

OPTIONS = ['Burger formulation']
NAMES = {OPTIONS[0]: wording.SAMPLE_PROJECT_NAME}


def build(kind, name, fresh_storage, storage):
    if kind not in OPTIONS:
        raise ValueError('Choose the burger teaching example.')
    opt = FoodOptimizer(name, storage=fresh_storage)
    opt.storage = storage
    opt.set_amount_unit('g')
    opt.load_ingredients_from_csv(pd.read_csv(Path(__file__).parent / 'data' / 'sample_ingredients.csv'))
    opt.add_process_parameter('Mixing time after fat', 45, 150, unit='s')
    opt.add_objective('Juiciness', 1, goal='target', target=7, min_val=0, max_val=10, unit='/10')
    opt.add_objective('Firmness', 1, goal='target', target=6, min_val=0, max_val=10, unit='/10')
    opt.add_objective('Cook loss', 1, goal='min', min_val=0, max_val=100, unit='%')
    opt.set_shares({'Firmness': 45, 'Juiciness': 35, 'Cook loss': 20})
    opt.set_formulation_total(100)
    opt.add_constraint('Fat per 100 g', max_val=16)
    opt.set_records('lot', True)
    opt.set_records('actual', True)
    opt.set_targets_source(wording.SAMPLE_TARGETS_SOURCE)
    opt.set_method(wording.SAMPLE_METHOD)
    return opt
