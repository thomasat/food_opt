"""What the Insights tab shows, worked out from a project without changing it.

Nothing here writes to the optimizer. Every model is fitted for display only,
with the optimizer's own builder so an expert's model settings apply, inside
a forked torch RNG so the next suggestion is exactly what it would have been.
The suggestions themselves still come from the one Overall-score model that
FoodOptimizer._ask_optimize fits; the models here never reach it."""
import hashlib
import json
import threading
from dataclasses import dataclass

import torch
from botorch.utils.transforms import normalize, unnormalize

from food_bo import CLOSE_TO_THE_BEST

SCORE = "__overall_score__"


# The owner's hard rule across browser tabs (review B2; Plan A review B
# I-3): Streamlit runs each session's script in a thread of one process, and
# torch has one global RNG. Every in-process Insights step that touches it
# goes through _isolated() — fits, conditioning, predictions, the predicted
# best — and holds this lock for its whole fork-and-reseed; the one ask()
# call site holds it too (ui_batch.ask_alone). So no Insights step can
# restore the RNG in the middle of a suggestion. Re-entrant: an _isolated()
# inside another is fine. Worker processes have their own.
RNG_LOCK = threading.RLock()


def outputs(opt):
    out = [(o['name'], float(o['min_val']), float(o['max_val']))
           for o in opt.objectives]
    out.append((SCORE, 0.0, float(opt.utility_ceiling())))
    return out


def measured(opt, name, i):
    # A formulation moved to Not scored counts for nothing here, its partial
    # values included: the confirmation says so.
    if opt.is_set_aside(i):
        return None
    if name == SCORE:
        # A formulation still waiting for results has no Overall score to
        # learn from yet: its missing measurement counts as zero in it.
        return float(opt.Y_history[i]) if opt.fully_measured(i) else None
    value = (opt.results_history[i] if i < len(opt.results_history) else {}).get(name)
    return None if value is None else float(value)


def _range_of(opt, name):
    for n, lo, hi in outputs(opt):
        if n == name:
            return lo, hi
    raise KeyError(name)


def xy(opt, name, rows):
    """(X, Y) of `rows` for `name`: the encoded make-ups normalised to the
    rows' Lowest-Highest, and the measured values as a column. The make-ups
    are model_X's: a formulation made off the Formulation total is learned
    at the total, as the suggestions learn it (spec 2026-09-30-bo-views §1)."""
    X = normalize(torch.tensor(opt.model_X(rows), dtype=torch.double),
                  opt._get_bounds())
    Y = torch.tensor([measured(opt, name, i) for i in rows],
                     dtype=torch.double).unsqueeze(-1)
    return X, Y


def _digest(payload):
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def _model_settings(opt):
    """What every model here depends on besides its rows: the measurements
    (their Ranges, goals and shares, so the Overall score too), an expert's
    model settings, and the rows' Lowest-Highest and encoding."""
    return {"objectives": opt.objectives, "cfg": getattr(opt, "bo_config", None),
            "bounds": opt._get_bounds().tolist(),
            # model_X places a row made off the total at the total: a new
            # total moves those rows (review I11).
            "total": getattr(opt, "formulation_total", None)}


def data_key(opt):
    """Everything a fitted model or a Round-ahead check depends on. The
    rows as the models learn them (model_X): a row made off the total moves
    when a formula's constant is edited (T1 review M3)."""
    return _digest({
        "X": opt.model_X(range(len(opt.X_history))), "Y": opt.Y_history,
        "R": opt.results_history,
        "ids": opt.formulation_ids, "rounds": opt.batch_history,
        "aside": [opt.is_set_aside(i) for i in range(len(opt.Y_history))],
        **_model_settings(opt),
    })


def setup_key(opt):
    """data_key plus everything else the formulation predicted to score
    highest depends on: the rows and their Lowest/Highest, calculations and
    fixed amounts, every limit, the Formulation total, the ingredient
    properties and the pre-mix make-ups (with the open Round's number and
    its Not scored rows, which decide which make-up a new Round uses)."""
    payload = {
        "data": data_key(opt), "variables": opt.variables,
        "limits": opt.constraints,
        "amounts": getattr(opt, "quantity_constraints", []),
        "total": opt.formulation_total,
        "properties": opt.ingredient_properties, "premixes": opt.premixes,
        "round": opt.pending_batch_no, "skipped": opt.skipped,
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


# --- review four (ruling 1, display and status only, 2026-10-02) ---


def tie_partner(opt, best):
    """W1 (review R-I9): ui_results._tie_partner's rule, moved here so
    Results and Insights ask one place: the row Best so far (row `best`) is
    compared with for a tie, the next highest DIFFERENT formulation (a
    Repeat is never a tie). None without one."""
    ranked = sorted(opt.best_candidates(), key=lambda i: float(opt.Y_history[i]), reverse=True)
    return next((i for i in ranked
                 if i != best and not _same(opt, start_recipe(opt, best), start_recipe(opt, i))),
                None)


_EPS = 1e-9


def _var(opt, name):
    return opt._by_name()[name]


def _is_ingredient(v):
    return v.get('category', 'ingredient') == 'ingredient'


def is_setting(opt, name):
    """Whether this row is a process setting (it has values, not amounts)."""
    return not _is_ingredient(_var(opt, name))


def at_total(opt, recipe):
    """A formulation's make-up at the project's Formulation total: a row
    made to another batch size is brought back to the total, so a
    prediction or a comparison never moves with the Batch size box. The
    same rule (off_total, 1 %) as the models' own inputs, model_X (spec
    2026-09-30-bo-views §1, review B1)."""
    recipe = dict(recipe)
    if opt.off_total(recipe):
        # A categorical row's choice kept as it is (T1 review M2).
        return opt.at_model_total(recipe)
    return opt.fill_formulas(recipe)


def start_recipe(opt, i):
    return at_total(opt, opt._recorded_recipe(i))


def _same(opt, a, b):
    for v in opt.variables:
        if opt.has_formula(v):
            continue
        x, y = a.get(v['name']), b.get(v['name'])
        if v.get('type') != 'continuous':
            if x != y:
                return False
            continue
        lo, hi = (float(t) for t in v['bounds'])
        tol = 0.0 if opt.is_fixed(v) else 0.005 * (hi - lo)
        if abs(float(x or 0.0) - float(y or 0.0)) > tol + 1e-12:
            return False
    return True


def _matches(opt, rows):
    """Every pair among `rows` that `_same` calls a match."""
    recipes = {i: start_recipe(opt, i) for i in rows}
    return [(i, j) for a, i in enumerate(rows) for j in rows[a + 1:]
            if _same(opt, recipes[i], recipes[j])]


def repeat_groups(opt, rows, matches=None):
    """The Repeat groups among `rows`: formulations `_same` calls a match,
    joined when any two of them match (a chain of matches is one group,
    since `_same` is not transitive). Groups of one are left out.
    `matches`: `_matches` over any rows that include `rows`, computed
    once by a caller that needs the groups of several subsets; only the
    pairs with both ends in `rows` join them."""
    if matches is None:
        matches = _matches(opt, rows)
    parent = {i: i for i in rows}

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, j in matches:
        if i in parent and j in parent:
            parent[root(j)] = root(i)
    groups = {}
    for i in rows:
        groups.setdefault(root(i), []).append(i)
    return [g for g in groups.values() if len(g) > 1]


# ------------------------------------------------------------------ #
#  Wave B: predictions for the open Round, spreads, predicted best
# ------------------------------------------------------------------ #


def distance(opt, a, b):
    """The largest change between two make-ups, each varying amount or
    setting as a share of its own Lowest-Highest span; None when a choice
    (a categorical row) differs."""
    largest = 0.0
    for v in opt.varying_variables():
        if v.get('type') != 'continuous':
            if a.get(v['name']) != b.get(v['name']):
                return None
            continue
        lo, hi = (float(t) for t in v['bounds'])
        d = abs(float(a.get(v['name'], 0.0) or 0.0) - float(b.get(v['name'], 0.0) or 0.0))
        largest = max(largest, d / (hi - lo))
    return largest


# Minor 15 (2026-09-29 review): a suggestion within this share of each
# varying row's own span of a formulation still waiting for results may be
# repeating it before either can be told apart — the model has not been
# taught that waiting formulation's result yet, so it cannot know. Tighter
# than CLOSE_TO_THE_BEST (0.15, "close to the best formulation so far"): a
# near-copy is a much narrower claim, that two specific rows are hard to
# tell apart at all.
NEAR_COPY_SPAN = 0.05


@dataclass
class NearCopy:
    formulation: int   # the waiting formulation's own number
    waiting_on: list    # what it still needs, most important first


def _fixed_rows_match(opt, a, b):
    """Whether every pinned-at-one-amount row agrees between two make-ups.
    In practice this is never the disagreement — a fixed row decodes to
    today's pinned amount on both sides, recorded or not — but the rule is
    checked anyway, exactly as the ones that can differ are."""
    return all(abs(float(a.get(v['name'], 0.0) or 0.0)
                  - float(b.get(v['name'], 0.0) or 0.0)) < 1e-6
              for v in opt.fixed_variables())


def near_copy(opt, recipe):
    """The nearest formulation still waiting for results that `recipe` may
    be repeating, or None. Read-only: decides nothing the search sees, and
    is never asked before a Round is generated — only after, to flag a row
    already chosen.

    A match needs every varying row (continuous, as a share of its own
    Lowest-Highest span; categorical, the same choice) within
    NEAR_COPY_SPAN, and every fixed row the same, both compared at the
    Formulation total (`at_total`) rather than whatever the round happens to
    be scaled to on screen. A waiting formulation from `recipe`'s own Round
    is excluded: it is a row of the batch being built, not one the model
    should already have learned from.

    A formulation moved to Not scored is not waiting (awaited_rows): no
    result is coming for it, so repeating it is not a repeat of anything
    still to arrive.

    Ties go to the lowest Formulation number, as `nearest` does."""
    a = at_total(opt, recipe)
    own_round = opt.pending_batch_no
    best = None
    for i in opt.awaited_rows():
        batch = opt.batch_history[i] if i < len(opt.batch_history) else None
        if own_round is not None and batch == own_round:
            continue
        b = start_recipe(opt, i)
        if not _fixed_rows_match(opt, a, b):
            continue
        d = distance(opt, a, b)
        if d is None or d > NEAR_COPY_SPAN + _EPS:
            continue
        key = (d, int(opt.formulation_ids[i]))
        if best is None or key < best[0]:
            best = (key, i)
    if best is None:
        return None
    i = best[1]
    return NearCopy(int(opt.formulation_ids[i]), opt.waiting_on(i))


# ------------------------------------------------------------------ #
#  0.8.0 speed and simplicity, Plan B: where you stand, and which
#  ingredient changes the prediction most near what you have tested (spec
#  2026-09-30 §2.3, §2.5; review I7, I8). The yardstick is handed in: a
#  number, noise_yardstick(...).value, so nothing here needs the checks.
# ------------------------------------------------------------------ #


# A row or a limit within this share of its own span (a limit: of its bound)
# counts as reached (review I7).
BINDING = 0.01


                            # by more than the yardstick


                               # calculation reads another of them


# ------------------------------------------------------------------ #
#  Charts proto (2026-10-01): "What you've tried" (parallel coordinates)
#  and "How the app chooses" (the path through Best so far). Model-free
#  data here; the charts are ui_insights'.
# ------------------------------------------------------------------ #


# Charts proto follow-up: the straight line in the full space from Best so
# far (t = 0) to a suggestion (t = 1), every row moving together.
LINE_FROM, LINE_TO, LINE_POINTS = -0.25, 1.5, 71


