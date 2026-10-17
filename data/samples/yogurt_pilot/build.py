"""The yogurt-style pilot as two sample projects (the pilot data is public).

Run with the devenv python, from anywhere:
    devenv/bin/python data/samples/yogurt_pilot/build.py [OUT_DIR]
It writes <project>.json (default: beside this file) for the two shipped
projects, plus "Round 5" which is built by tests only, and "Round 6 results
(pilot).xlsx", Round 6's workbook with the pilot's ratings typed in
(write_round6_results).

Every pilot Iteration is one Round; its three Batches are recorded in Batch
order as formulations of your own, exactly as the app records them:
  2 · Make a round > "Add a formulation of your own" > Add to this Round
      = opt.add_to_pending_batch(recipe, note)          (ui_batch._add_own)
  Record results > Save results
      = opt.tell(row recipe, results, formulation_no, batch_no, note)
        for each row, then opt.set_pending_batch(None)  (ui_batch._save_results)

Scoring: each measurement is a Target on a 0-15 scale with a share of 25, so
the app's Overall score is sum 25 * (1 - |x - target| / 15)
= 100 * (1 - sum|x - target| / 60), the pilot's Total_Utility_Score x 100.

Limits: the campaign's own settings (with them, the campaign's code
regenerates its recorded batches), not inferred from the data. Per 100 g of
yogurt the 7 varied ingredients have the Lowest-Highest in INGREDIENTS, and
their total is held to 97.69051-98.69051 g: 98.69051 g is 100 g less the two
ingredients held constant in every batch (paper, Appendix G.2.1), soy protein
isolate (1.00 %) and tricalcium phosphate (0.31 %). Those two are not in the
CSVs and are not ingredients here. The projects carry the total as
one amount limit over all 7, named BATCH_TOTAL_NAME ("Ingredient total"), at
least BATCH_TOTAL_MIN and at most BATCH_TOTAL_MAX. Every Batch EGBO chose
(Rounds 3-10) sits inside every limit (tests/test_sample_projects.py). Rounds
1-2 were a random (Sobol) starting set the engine drew within the ingredient
limits but not the total (88.9-101.4 g); they stay as recorded, and the
model still learns from them. No recorded formulation is altered or rescaled.
Not a Formulation total: that is one number held to +-0.5 %, would say a
formulation off it does not add up to the total, and setting one discards an
open Round.

Results: each rating is the median of a trained panel of five, 0-15, one per
attribute per Batch; no Batch was repeated. Each Batch was mixed, cooked at
90 C for 8 min and chilled (no culture or fermentation: the paper's method is
mix, cook, chill, with no culture among its ingredients). The campaign's
budget was fixed at 10 Rounds of 3.

No dates. The pilot was tasted Feb-May 2026, but its CSVs carry no date for
any Batch, so no pilot row carries one: build() blanks timestamps_history
(None per row, which 3 · Results and both exports show as a blank Date
recorded) and pending_batch_created (the open Round 6's sheets then say
Printed <the day they are printed>, as for any batch with no date). The
build date is never shown as a tasting date. A result recorded later in the
app is stamped with its own day, as always.

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

# CSV column -> (name in the app, Lowest, Highest) in g per 100 g: the
# campaign's own bounds (module docstring), not the data's range.
INGREDIENTS = OrderedDict([
    ("Coconut_cream", ("Coconut cream", 46.0, 61.0)),
    ("Tap_water", ("Tap water", 25.0, 42.0)),
    ("Lemon_juice", ("Lemon juice", 0.0, 3.2)),
    ("Salt", ("Salt", 0.0, 1.8)),
    ("Nutritional_yeast", ("Nutritional yeast", 0.0, 2.5)),
    ("Tapioca_starch", ("Tapioca starch", 0.9, 2.6)),
    ("Xanthan_gum", ("Xanthan gum", 0.15, 0.65)),
])

# CSV column -> (name in the app, target). Scale 0-15, equal shares.
MEASUREMENTS = OrderedDict([
    ("Consistency", ("Consistency", 12.0)),
    ("Creaminess", ("Creaminess", 5.0)),
    ("Tangy", ("Tangy", 5.0)),
    ("Thickness_Uniformity", ("Thickness uniformity", 13.0)),
])
SCALE = (0.0, 15.0)

# The campaign's limit on the 7 ingredients' total (module docstring): 100 g
# less the two constant ingredients, and 1 g below that.
BATCH_TOTAL_NAME = "Ingredient total"
BATCH_TOTAL_MIN = 97.69051
BATCH_TOTAL_MAX = 98.69051
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
    # No Formulation total (module docstring): the campaign's limit on the 7
    # ingredients' total instead.
    opt.add_total_mass_constraint(BATCH_TOTAL_MIN, BATCH_TOTAL_MAX,
                                  name=BATCH_TOTAL_NAME)
    for _, (m, target) in MEASUREMENTS.items():
        opt.add_objective(m, SHARE, goal="target", target=target,
                          min_val=SCALE[0], max_val=SCALE[1], unit="/15")
    opt.set_shares({m: SHARE for m, _ in MEASUREMENTS.values()})
    return opt


STARTING_ROUNDS = 2    # Rounds 1-2: a random (Sobol) starting set; 3-10 chosen by EGBO


def note_of(batch):
    """The Note typed for a pilot Batch, saying where it came from: 'Random
    starting set 1.1' (Rounds 1-2, drawn at random by the engine) or 'Pilot
    batch 3.1 · EGBO' (Rounds 3-10, chosen by EGBO, the engine behind this
    app)."""
    if _batch_key(batch)[0] <= STARTING_ROUNDS:
        return f"Random starting set {batch}"
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


def undate(opt):
    """No dates (module docstring): the pilot's CSVs date no Batch, so the
    day this script ran is not written as the day any pilot row was
    recorded, nor as the day the open Round was opened."""
    opt.timestamps_history = [None] * len(opt.timestamps_history)
    opt.pending_batch_created = None


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
        undate(opt)
        opt.save()
        return opt
    finally:
        os.chdir(prev)


ROUND6_RESULTS = "Round 6 results (pilot).xlsx"


def _cell_value(value):
    """12, not 12.0: a whole panel median as the bench would type it."""
    return int(value) if float(value).is_integer() else float(value)


def write_round6_results(path=None, work_dir=None):
    """The shipped ROUND6_RESULTS: Round 6's workbook exactly as 2 · Make a
    round downloads it (the compact one, at the Round's own size), with
    the pilot's Round 6 ratings typed into its Results sheet. The notes
    are the ones the Round's rows carry (note_of). Its title names the
    Round and the project but no print date: the pilot printed nothing on
    the day this runs. `work_dir` holds the throwaway project build."""
    import io
    import tempfile

    from openpyxl import load_workbook

    import wording

    path = path or os.path.join(HERE, ROUND6_RESULTS)
    with tempfile.TemporaryDirectory() as scratch:
        opt = build("Yogurt pilot - Round 6 to score", scored=5, open_round=6,
                    out_dir=work_dir or scratch)
        book = load_workbook(io.BytesIO(opt.workbook_bytes(
            opt.pending_batch, opt.open_round_size(), print_pack=False)))
    overview = book[wording.OVERVIEW_SHEET]
    overview["A1"] = f"{wording.batch_sheet_name(opt.pending_batch_no)} · {opt.project_name}"
    by_note = {note_of(b): res for b, _, res, _ in pilot()[6]}
    sheet = book[wording.RESULTS_SHEET]
    header = next(c.row for row in sheet for c in row
                  if c.column == 1 and c.value == wording.MEASUREMENT_COLUMN)
    columns = {opt._formulation_column_number(c.value): c.column
               for c in sheet[header] if opt._formulation_column_number(c.value)}
    for row in opt.pending_batch:
        results = by_note[row["note"]]
        for r in range(header + 1, sheet.max_row + 1):
            label = str(sheet.cell(r, 1).value or "")
            name = next((n for n in results if label.startswith(n + " (")), None)
            if name is not None:
                sheet.cell(r, columns[row["formulation"]], _cell_value(results[name]))
    book.save(path)
    return path


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
    print("wrote", write_round6_results(os.path.join(out, ROUND6_RESULTS)))
    for p in projects:
        worst, n = check_scores(p)
        os.replace(os.path.join(out, f"{p.project_name}.pkl"),
                   os.path.join(out, f"{p.project_name}.json"))
        print(f"{p.project_name}: {n} scored, open "
              f"{len(p.pending_batch or [])}, max |score - pilot| {worst:.2e}")
