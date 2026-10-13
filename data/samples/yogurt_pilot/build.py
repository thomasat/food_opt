"""The yogurt-style pilot as two sample projects (the pilot data is public).

Run with the devenv python, from anywhere:
    devenv/bin/python data/samples/yogurt_pilot/build.py [OUT_DIR]
It writes <project>.json (default: beside this file) for the two shipped
projects, plus "Round 5" which is built by tests only.

Every pilot Iteration is one Round; its three Batches are recorded in Batch
order as the team's own formulations, exactly as the app records them:
  2 · Make a round > "Add a formulation of your own" > Add to this Round
      = opt.add_to_pending_batch(recipe, note)          (ui_batch._add_own)
  Record results > Save results
      = opt.tell(row recipe, results, formulation_no, batch_no, note)
        for each row, then opt.set_pending_batch(None)  (ui_batch._save_results)

Scoring: each measurement is a Target on a 0-15 scale with a share of 25, so
the app's Overall score is sum 25 * (1 - |x - target| / 15)
= 100 * (1 - sum|x - target| / 60), the pilot's Total_Utility_Score x 100.

Batch total (inferred from the pilot). The pilot recorded no total limit, but
its data shows one. The 7 recorded ingredients add up to exactly 98.691 g in
every Batch of Rounds 3-5 and in 6.1, 6.3 and 8.3; to exactly 97.691 g in 6.2,
7.x, 9.x and 10.x; and to 98.211 g (8.1) and 98.131 g (8.2) in between. The
team's starting set (Rounds 1-2) ranges 88.9-101.4 g. So every Batch EGBO chose
(Rounds 3-10) sits inside 97.691-98.691 g, mostly on one edge or the other:
the signature of a two-sided limit on the total of these 7 ingredients (no
other ingredient is known). The projects carry it as one amount limit over all
7, named BATCH_TOTAL_NAME, at least BATCH_TOTAL_MIN and at most BATCH_TOTAL_MAX
(the two edges rounded outward to 0.1 g). It only bounds what the app
suggests: no recorded formulation is altered or rescaled, and the Rounds 1-2
rows outside it stay as recorded (the model still learns from them).
Not a Formulation total: that is one number held to +-0.5 %, which would
exclude both edges the pilot used (97.70-98.68 g at 98.19 g), says a
formulation off it does not add up to the total, and setting one discards an
open Round.

Projects
  Yogurt pilot - Round 6 to score   Rounds 1-5 scored; Round 6 open with the
                                    pilot's three Round-6 formulations, unscored.
  Yogurt pilot - all 10 Rounds      Rounds 1-10 scored.
  Round 5 (built by tests only)     Rounds 1-5 scored, nothing open.
"""
import csv
import os
import sys
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..")))

from food_bo import FoodOptimizer  # noqa: E402

DATA = HERE

ZERO = 1e-6            # |x| below this is an optimiser zero

# CSV column -> (name in the app, Lowest, Highest) in g. Caps from the pilot
# where the data shows one; otherwise the data's range rounded outward, with
# Lowest 0 where the data reaches 0.
INGREDIENTS = OrderedDict([
    ("Coconut_cream", ("Coconut cream", 48.0, 61.0)),        # data 48.38-60.82
    ("Tap_water", ("Tap water", 25.0, 42.0)),                # data 25.01-42 (cap)
    ("Lemon_juice", ("Lemon juice", 0.0, 3.0)),              # data 0-2.82
    ("Salt", ("Salt", 0.0, 1.8)),                            # data 0-1.8 (cap)
    ("Nutritional_yeast", ("Nutritional yeast", 0.0, 2.5)),  # data 0-2.5 (cap)
    ("Tapioca_starch", ("Tapioca starch", 1.0, 2.6)),        # data 1.04-2.6 (cap)
    ("Xanthan_gum", ("Xanthan gum", 0.1, 0.65)),             # data 0.15-0.65 (cap)
])

# CSV column -> (name in the app, target). Scale 0-15, equal shares.
MEASUREMENTS = OrderedDict([
    ("Consistency", ("Consistency", 12.0)),
    ("Creaminess", ("Creaminess", 5.0)),
    ("Tangy", ("Tangy", 5.0)),
    ("Thickness_Uniformity", ("Thickness uniformity", 13.0)),
])
SCALE = (0.0, 15.0)

# The limit inferred from the pilot's totals (module docstring): every Batch
# EGBO chose adds up to 97.691-98.691 g over the 7 ingredients.
BATCH_TOTAL_NAME = "Batch total (inferred from the pilot)"
BATCH_TOTAL_MIN = 97.6
BATCH_TOTAL_MAX = 98.7
SHARE = 100.0 / len(MEASUREMENTS)


def _rows(name):
    with open(os.path.join(DATA, name), newline="") as fh:
        return list(csv.DictReader(fh))


def _batch_key(batch):
    it, k = batch.split(".")
    return int(it), int(k)


def pilot():
    """{iteration: [(batch, recipe, results, utility x 100)]} in Batch order."""
    amounts = {r["Batch"]: r for r in _rows("yogurt_ingredient_data.csv")}
    sensory = {r["Batch"]: r for r in _rows("yogurt_sensory_data.csv")}
    utility = {r["Batch"]: float(r["Total_Utility_Score"]) * 100
               for r in _rows("yogurt_utility_per_recipe.csv")}
    rounds = OrderedDict()
    for batch in sorted(amounts, key=_batch_key):
        a, s = amounts[batch], sensory[batch]
        recipe = {}
        for col, (name, _, _) in INGREDIENTS.items():
            v = float(a[col])
            recipe[name] = 0.0 if abs(v) < ZERO else v
        results = {name: float(s[col]) for col, (name, _) in MEASUREMENTS.items()}
        rounds.setdefault(int(a["Iteration"]), []).append(
            (batch, recipe, results, utility[batch]))
    return rounds


def setup(name):
    # A pkl left from an earlier run would be loaded, not rebuilt.
    if os.path.exists(f"{name}.pkl"):
        os.remove(f"{name}.pkl")
    opt = FoodOptimizer(name)
    opt.set_amount_unit("g")
    for _, (ing, lo, hi) in INGREDIENTS.items():
        opt.add_ingredient(ing, lo, hi)
    # No Formulation total (module docstring): a limit on the 7 ingredients'
    # total instead, inferred from the pilot's batches.
    opt.add_total_mass_constraint(BATCH_TOTAL_MIN, BATCH_TOTAL_MAX,
                                  name=BATCH_TOTAL_NAME)
    for _, (m, target) in MEASUREMENTS.items():
        opt.add_objective(m, SHARE, goal="target", target=target,
                          min_val=SCALE[0], max_val=SCALE[1], unit="/15")
    opt.set_shares({m: SHARE for m, _ in MEASUREMENTS.values()})
    return opt


STARTING_ROUNDS = 2    # Rounds 1-2: the team's starting set; 3-10 chosen by EGBO


def note_of(batch):
    """The Note typed for a pilot Batch, saying where it came from: 'Pilot
    starting set 1.1' (Rounds 1-2) or 'Pilot batch 3.1 · EGBO' (Rounds
    3-10, chosen by EGBO, the engine behind this app)."""
    if _batch_key(batch)[0] <= STARTING_ROUNDS:
        return f"Pilot starting set {batch}"
    return f"Pilot batch {batch} · EGBO"


def add_round(opt, batches):
    """2 · Make a round > Add a formulation of your own, once per pilot
    Batch, its provenance typed as the Note (note_of)."""
    for batch, recipe, _, _ in batches:
        opt.add_to_pending_batch(dict(recipe), note_of(batch))


def score_round(opt, batches):
    """Record results > Save results: every open row scored, then the Round
    closes (ui_batch._save_results)."""
    by_note = {note_of(b): res for b, _, res, _ in batches}
    batch_no = opt.pending_batch_no
    for row in list(opt.pending_batch):
        note = row.get("note") or ""
        opt.tell(row["recipe"], dict(by_note[note]), formulation_no=row["formulation"],
                 batch_no=batch_no, note=note)
    opt.set_pending_batch(None)


def build(name, scored, open_round=None, out_dir=None):
    """Build one project; LocalStorage writes <name>.pkl into the cwd, so
    chdir into out_dir for the build when it is given."""
    prev = os.getcwd()
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        os.chdir(out_dir)
    try:
        rounds = pilot()
        opt = setup(name)
        for it in range(1, scored + 1):
            add_round(opt, rounds[it])
            score_round(opt, rounds[it])
        if open_round is not None:
            add_round(opt, rounds[open_round])
        opt.save()
        return opt
    finally:
        os.chdir(prev)


def check_scores(opt):
    """The app's Overall score against the pilot's utility x 100."""
    want = [u for it in pilot().values() for _, _, _, u in it]
    worst = 0.0
    for i, got in enumerate(opt.Y_history):
        worst = max(worst, abs(float(got) - want[i]))
    return worst, len(opt.Y_history)


if __name__ == "__main__":
    out = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else HERE)
    projects = [
        build("Yogurt pilot - Round 6 to score", scored=5, open_round=6,
              out_dir=out),
        build("Yogurt pilot - all 10 Rounds", scored=10, out_dir=out),
    ]
    for p in projects:
        worst, n = check_scores(p)
        os.replace(os.path.join(out, f"{p.project_name}.pkl"),
                   os.path.join(out, f"{p.project_name}.json"))
        print(f"{p.project_name}: {n} scored, open "
              f"{len(p.pending_batch or [])}, max |score - pilot| {worst:.2e}")
