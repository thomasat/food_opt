"""Clearly labeled teaching examples; no measured results or validated recipes."""
from pathlib import Path

import pandas as pd

from food_bo import FoodOptimizer
import wording

OPTIONS = ['Burger formulation', 'Advanced burger: calculated ingredients', 'Okara fermentation']
NAMES = {OPTIONS[0]: wording.SAMPLE_PROJECT_NAME,
         OPTIONS[1]: 'Advanced burger example', OPTIONS[2]: 'Okara fermentation example'}


def build(kind, name, fresh_storage, storage):
    opt = FoodOptimizer(name, storage=fresh_storage)
    opt.storage = storage
    if kind == OPTIONS[1]:
        opt.set_amount_unit('g')
        frame = pd.read_csv(Path(__file__).parent / 'data' / 'sample_ingredients.csv')
        pea = frame['Name'].eq('Textured pea protein')
        frame.loc[pea, ['Lowest', 'Highest']] = [4, 6.5]
        frame.loc[frame['Name'].eq('Water'), 'Name'] = 'Remaining water'
        soy = frame.loc[pea].copy()
        soy['Name'] = 'Textured soy protein'
        # Property values are examples, not supplier specifications.
        water = frame.loc[frame['Name'].eq('Remaining water')].copy()
        water['Name'] = 'Hydration water'
        water['Rule'] = '= 2.2 * (Textured pea protein + Textured soy protein)'
        frame = pd.concat([frame, soy, water], ignore_index=True)
        opt.load_ingredients_from_csv(frame)
        opt.add_process_parameter('Mixing time after oils', 45, 150, unit='s')
        opt.add_objective('Firmness', 1, goal='target', target=6, min_val=0, max_val=10, unit='/10')
        opt.add_objective('Juiciness', 1, goal='target', target=7, min_val=0, max_val=10, unit='/10')
        opt.add_objective('Cook loss', 1, goal='min', min_val=0, max_val=40, unit='%')
        opt.set_shares({'Firmness': 45, 'Juiciness': 35, 'Cook loss': 20})
        opt.set_formulation_total(100)
        opt.add_constraint('Fat per 100 g', max_val=16)
        opt.set_records('lot', True)
        opt.set_records('actual', True)
        opt.set_targets_source(wording.SAMPLE_TARGETS_SOURCE + ' The shared hydration ratio of 2.2 is an illustrative protocol assumption; establish a suitable ratio for the selected protein grades in pilot trials.')
        opt.set_method(
            'Teaching example: Hydration water = 2.2 × (textured pea protein + textured soy protein). '
            'Hydrate both proteins using only the calculated Hydration water. Remaining water is a separate '
            'weighing amount that brings all ingredients to 100 g; do not add Hydration water again. '
            'Combine the remaining water, dry blend and gluten, then add the hydrated proteins, seasoning '
            'and oils. Vary only the specified mixing time. Establish and document one fixed hydration, '
            'forming, cooking and measurement protocol before collecting real results. Example ranges '
            'and ingredient properties are illustrative, not supplier specifications.')
    elif kind == OPTIONS[2]:
        for label, low, high, unit in [
            ('Fermentation temperature', 25, 35, '°C'),
            ('Fermentation duration', 24, 72, 'h'),
            ('Inoculum level', 1, 5, '% wet substrate mass'),
            ('Initial pH', 5, 7, '')]:
            opt.add_process_parameter(label, low, high, unit=unit)
        opt.add_objective('Patty firmness', 1, goal='target', target=20, min_val=0, max_val=100, unit='N')
        opt.add_objective('Off-flavour intensity', 1, goal='min', min_val=0, max_val=10, unit='/10')
        opt.add_objective('Cook loss', 1, goal='min', min_val=0, max_val=40, unit='%')
        opt.set_shares({'Patty firmness': 40, 'Off-flavour intensity': 40, 'Cook loss': 20})
        opt.set_targets_source(
            'Illustrative numeric fermentation experiment, not a validated fermentation protocol. '
            'Background: Wang et al. (2015), https://doi.org/10.7506/spkx1002-6630-201509017, '
            'optimized okara fermentation using pH, inoculum, temperature and duration. '
            'That study measured okara composition; the ranges and burger endpoints here are teaching '
            'choices, not its published optimum. Choose suitable ranges for your organism and process.')
        opt.set_method(
            'Keep the microbial strain, okara source and moisture, substrate mass, vessel, aeration and '
            'burger recipe constant. Select numeric fermentation settings for each independent fermentation '
            'batch. Record batch ID, strain, substrate lot and date in its note. Prepare burgers with the '
            'same fixed recipe and measure firmness, off-flavour intensity and cook loss at a defined '
            'endpoint using a fixed protocol. Several patties from one fermentation batch are subsamples, '
            'not independent fermentation replicates; enter their prespecified aggregate as one result. '
            'Use independent fermentation batches for replication. This example does not model strain '
            'choices or complete time courses.')
    else:
        raise ValueError('Choose an available example project.')
    return opt
