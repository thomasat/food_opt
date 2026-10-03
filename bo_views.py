"""The numbers behind the views that explain the search (spec
2026-09-30-bo-views §3; review B4, I5, I9, I10, I11, I12, M4, M5, M10).

The suggestion model's acquisition, rebuilt on the display side without
touching ask(): the same training rows (insights.xy, the rows ask() learns
from), the same builder, the same default fit and the Round's own seed, then
opt._build_acqf, built fresh for every evaluation (optimize_acqf changes the
object it is given, 02 M5). A Round whose generation was recorded (ui_batch
records it at the Generate call site) and whose rows still match is rebuilt
exactly; any other Round is recomputed from today's records with the
display model of the same rows, and says so.

Every step holds insights.RNG_LOCK inside a forked torch RNG, so nothing
here can move a suggestion. The model is fitted once in this process and
every question is answered here with it (fix round 1, T3 I-3): sending it to
a worker only fitted it again, and could start the pool. A worker (kind
"acq", `answer`) still gives the same numbers from JSON for a caller that
has not fitted. No screen text lives here."""
import collections
import contextlib
import dataclasses
import math
import random
import threading
from dataclasses import dataclass

import botorch
import gpytorch
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.input import Warp
from botorch.utils.sampling import get_polytope_samples
from botorch.utils.transforms import normalize
from gpytorch.mlls import ExactMarginalLogLikelihood

import insights
import wording
from food_bo import COLD_START_RUNS, ROUND_FIELD, FoodOptimizer, LinearForm

# The pool of allowed formulations (review I9): hit-and-run over the
# normalised continuous columns, seed 0, 3,000 points, thinning 10. Fewer
# than POOL_MIN distinct points is a degenerate pool (pool_points). The pool
# no longer gives the role (0.8.0 e2e fixes A2, review I3: role()).
POOL_SIZE = 3000
POOL_SEED = 0
POOL_THINNING = 10
POOL_MIN = 500
# Every evaluation runs in chunks of this many points (review M10).
CHUNK = 256
# The outputs kept in this process (review B4: outputs, not models),
# shared by every session's script thread (T3 M-3: under a lock).
_CACHE = collections.OrderedDict()
_CACHE_SIZE = 64
_CACHE_LOCK = threading.Lock()
_VERSIONS = (torch.__version__, gpytorch.__version__, botorch.__version__)


def acquisition(opt):
    """'qlognei' (the default), 'qlogei' or 'qucb': what _build_acqf builds."""
    return (getattr(opt, "bo_config", None) or {}).get("acquisition", "qlognei")


# ------------------------------------------------------------------ #
#  Which rows, which seed: the generation record (review B4)
# ------------------------------------------------------------------ #

def _complete_rows(opt, before=None):
    n = len(opt.Y_history) if before is None else before
    return [i for i in range(n) if opt.fully_measured(i)]


def generation_key(opt, rows):
    """What a rebuild of a Round depends on: the rows as the suggestion
    model learns them (insights.xy), their scores, the model settings and
    robust. A record whose key still matches is rebuilt exactly."""
    X, Y = insights.xy(opt, insights.SCORE, rows)
    return insights._digest({"X": X.tolist(), "Y": Y.tolist(),
                             "cfg": getattr(opt, "bo_config", None),
                             "robust": bool(getattr(opt, "robust", False))})


def generation_record(opt, n):
    """The record the Generate call site keeps (review B4), made just
    before ask(n): the seed ask() will use (next_formulation_no), the
    formulation numbers of the rows it learns from, n, the regime, and the
    key of those rows. The call site adds "picks" after ask() returns: the
    formulation numbers ask() issued, in order (T3 I-2)."""
    rows = _complete_rows(opt)
    warm = opt.complete_count() >= COLD_START_RUNS
    return {"seed": int(opt.next_formulation_no),
            "rows": [int(opt.formulation_ids[i]) for i in rows], "n": int(n),
            "regime": "warm" if warm else "cold",
            "key": generation_key(opt, rows) if warm else ""}


@dataclass
class Generation:
    round: int
    seed: int
    rows: list          # positions of the rows the Round was (or is taken as) learned from
    regime: str         # "warm" or "cold"
    exact: bool         # the record is there and its rows still match
    picks: list = dataclasses.field(default_factory=list)   # numbers ask() issued, in
                        # order (T3 I-2); [] without a record that has them


def _round_numbers(opt, r):
    """Every Formulation number Round `r` issued: recorded, Not scored, and
    still open."""
    numbers = [int(n) for n, b in zip(opt.formulation_ids, opt.batch_history) if b == r]
    numbers += [int(s['formulation']) for s in getattr(opt, "skipped", []) or []
                if s.get(ROUND_FIELD) == r]
    if r == opt.pending_batch_no:
        numbers += [int(row['formulation']) for row in opt.pending_batch or []]
    return numbers


def generation(opt, r):
    """How Round `r` was generated: its record when there is one (exact
    when its rows still match, review B4), else taken from today's records
    — the seed its lowest Formulation number, the rows the fully measured
    ones recorded before it (02 §3)."""
    record = (getattr(opt, "round_generation", None) or {}).get(r)
    ids = [int(x) for x in opt.formulation_ids]
    if record is not None:
        rows = [ids.index(n) for n in record["rows"] if n in ids]
        exact = (len(rows) == len(record["rows"]) and record["regime"] == "warm"
                 and all(opt.fully_measured(i) for i in rows)
                 and record.get("key") == generation_key(opt, rows))
        return Generation(r, int(record["seed"]), rows, record["regime"], exact,
                          [int(n) for n in record.get("picks") or []])
    first = [i for i, b in enumerate(opt.batch_history) if b == r]
    rows = _complete_rows(opt, first[0] if first else None)
    numbers = _round_numbers(opt, r)
    seed = min(numbers) if numbers else int(opt.next_formulation_no)
    regime = "warm" if len(rows) >= COLD_START_RUNS else "cold"
    return Generation(r, seed, rows, regime, False)


# ------------------------------------------------------------------ #
#  The rebuilt model and its acquisition
# ------------------------------------------------------------------ #

@dataclass
class Rebuilt:
    gen: Generation
    gp: object
    X: object           # the training rows, normalised
    Y: object
    rng: object         # torch's RNG state where ask() built its acquisition
    cfg: object
    exact: bool


def fit_default(cfg, X, Y, robust, seed):
    """(gp, RNG state) as _ask_optimize makes them: torch seeded with the
    Round's seed, the project's builder, Warp when robust, botorch's default
    fit (not insights.FIT_OPTIONS). Call inside a forked RNG."""
    torch.manual_seed(int(seed))
    dim = X.shape[-1]
    input_tf = Warp(d=dim, indices=list(range(dim))) if robust else None
    gp = FoodOptimizer._build_gp(_Settings(cfg), X, Y, dim, input_tf)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
    return gp, torch.get_rng_state()


class _Settings:
    """What _build_gp and _build_acqf read of a project: its settings."""

    def __init__(self, cfg):
        self.bo_config = cfg


def build_acqf(cfg, gp, X, Y, rng, pending=None):
    """A fresh acquisition, with the RNG where ask() had it (its sampler
    and qLogNEI's pruning draw from it), and `pending` as X_pending (the
    Round's earlier picks, review I12). Call inside a forked RNG."""
    torch.set_rng_state(rng)
    acqf = FoodOptimizer._build_acqf(_Settings(cfg), gp, X, Y)
    if pending is not None:
        acqf.set_X_pending(pending)
    return acqf


def rebuild(opt, gen, fitted=None):
    """The suggestion model of Round `gen`: fitted again on its rows with
    its seed when the generation is exact; otherwise the display model of
    the same rows (`fitted`, insights.fit's, from the memo when it is
    there: no new fit, review I10), labelled recomputed. A `fitted` made on
    another number of rows is not used (T3 M-6). Callers hold a memo
    (insights.memo_in_use), so a recomputed Round costs no fit (T3 M-2).
    None for a cold Round (no acquisition, review I12) or too few rows."""
    if gen.regime != "warm" or len(gen.rows) < insights.MIN_ROWS:
        return None
    if fitted is not None and getattr(fitted, "n_rows", None) != len(gen.rows):
        fitted = None
    X, Y = insights.xy(opt, insights.SCORE, gen.rows)
    cfg = getattr(opt, "bo_config", None)
    with insights.RNG_LOCK, torch.random.fork_rng():
        if gen.exact:
            gp, rng = fit_default(cfg, X, Y, bool(getattr(opt, "robust", False)), gen.seed)
        else:
            if fitted is None:
                fitted = insights.fit(opt, insights.SCORE, gen.rows)
            gp = fitted.model
            if gp is None:
                return None
            torch.manual_seed(int(gen.seed))
            rng = torch.get_rng_state()
    gp.eval()
    return Rebuilt(gen, gp, X, Y, rng, cfg, gen.exact)


def _values(cfg, raw):
    """An acquisition's value in points: expected improvement for the two
    log forms (their log is never shown, review I5), the optimistic score
    itself for qUCB."""
    kind = (cfg or {}).get("acquisition", "qlognei")
    return raw if kind == "qucb" else raw.exp()


def joint_value(model, B):
    """The acquisition of the whole set B (q x d) at once, in points, no
    pending: what qLogEI's picks 2+ are measured from (T3 I-1)."""
    with insights.RNG_LOCK, torch.random.fork_rng(), torch.no_grad():
        acqf = build_acqf(model.cfg, model.gp, model.X, model.Y, model.rng)
        raw = acqf(B.unsqueeze(0)).reshape(-1)
    return float(_values(model.cfg, raw)[0])


def evaluate_here(model, P, pending=None):
    """(mean, sd, value) of the suggestion model at normalised points P:
    the model's own mean and sd (no Repeat noise, no calibration) and the
    acquisition in points, in CHUNK-point pieces, the acquisition built
    fresh (review M10)."""
    means, sds, values = [], [], []
    with insights.RNG_LOCK, torch.random.fork_rng(), torch.no_grad():
        acqf = build_acqf(model.cfg, model.gp, model.X, model.Y, model.rng, pending)
        for k in range(0, P.shape[0], CHUNK):
            part = P[k:k + CHUNK]
            post = model.gp.posterior(part)
            means.append(post.mean.reshape(-1))
            sds.append(post.variance.clamp_min(0.0).sqrt().reshape(-1))
            values.append(acqf(part.unsqueeze(1)).reshape(-1))
    return (torch.cat(means), torch.cat(sds), _values(model.cfg, torch.cat(values)))


def answer(request):
    """insights_worker's "acq" kind (review I10): fit on X, Y with the
    seed and the default fit, build the acquisition fresh, and answer the
    mean, the sd and the value at every point."""
    tensor = lambda key: torch.tensor(request[key], dtype=torch.double)
    X, Y, P = tensor("X"), tensor("Y"), tensor("points")
    pending = tensor("pending") if request.get("pending") else None
    with torch.random.fork_rng():
        gp, rng = fit_default(request["cfg"], X, Y, bool(request["robust"]), request["seed"])
    gp.eval()
    model = Rebuilt(None, gp, X, Y, rng, request["cfg"], True)
    mean, sd, value = evaluate_here(model, P, pending)
    return {"mean": mean.tolist(), "sd": sd.tolist(), "value": value.tolist()}


@dataclass
class Values:
    mean: object        # tensor: the suggestion model's mean, points
    sd: object          # its own sd (narrower than the 95 % interval)
    value: object       # expected improvement (or qUCB's optimistic score), points
    exact: bool


def _key(opt, model, P, pending):
    """Review I11: everything an output depends on."""
    return insights._digest({
        "X": model.X.tolist(), "Y": model.Y.tolist(), "seed": model.gen.seed,
        "exact": model.exact, "cfg": model.cfg,
        "robust": bool(getattr(opt, "robust", False)),
        "model": None if model.exact else insights.hyper_state(model.gp),
        "P": P.tolist(), "pending": None if pending is None else pending.tolist(),
        "pool": [POOL_SEED, POOL_SIZE], "versions": _VERSIONS})


def evaluate(opt, model, P, pending=None):
    """Values at normalised points P, from the cache when this exact
    question was answered before, else worked out here with the model
    already fitted (T3 I-3: never sent to a worker, never starting one)."""
    key = _key(opt, model, P, pending)
    with _CACHE_LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]
    got = Values(*evaluate_here(model, P, pending), model.exact)
    _remember(key, got)
    return got


def _remember(key, value):
    with _CACHE_LOCK:
        _CACHE[key] = value
        _CACHE.move_to_end(key)
        while len(_CACHE) > _CACHE_SIZE:
            _CACHE.popitem(last=False)


def points(opt, recipes):
    """Recipes (at the Formulation total) as normalised points."""
    return normalize(torch.tensor([opt._encode(r) for r in recipes], dtype=torch.double),
                     opt._get_bounds())


# ------------------------------------------------------------------ #
#  The allowed formulations, sampled evenly (review I9)
# ------------------------------------------------------------------ #

def _total_equality(opt):
    """The Formulation total's band as one equality (sum = the total), and
    the band's two inequalities it replaces; None when the total is not a
    constraint on the search (no total, or a Fill to total row makes every
    formulation add up by construction)."""
    qc = next((q for q in getattr(opt, "quantity_constraints", [])
               if q.get('source') == 'formulation_total'), None)
    if qc is None or not opt.has_formulation_total():
        return None
    known = {v['name'] for v in opt.variables if v['type'] == 'continuous'}
    form = LinearForm()
    for name in qc['ingredients']:
        if name in known:
            form = form + opt._linear_form(name)
    idx, coeffs, offset = opt._form_over_columns(form, opt._search_columns())
    if not idx:
        return None
    t_idx = torch.tensor(idx, dtype=torch.long)
    t_coeffs = torch.tensor(coeffs, dtype=torch.double)
    band = [(t_idx, t_coeffs, qc['min'] - offset) if qc['min'] is not None else None,
            (t_idx, -t_coeffs, -(qc['max'] - offset)) if qc['max'] is not None else None]
    return (t_idx, t_coeffs, float(opt.formulation_total) - offset), band


def _same(a, b):
    return (a is not None and b is not None and torch.equal(a[0], b[0])
            and torch.allclose(a[1], b[1]) and math.isclose(float(a[2]), float(b[2]),
                                                             rel_tol=0, abs_tol=1e-9))


def _categorical_columns(opt):
    out, col = [], 0
    for var in opt.variables:
        if opt.has_formula(var):
            continue
        if var['type'] == 'continuous':
            col += 1
        elif var['type'] == 'categorical':
            out.append(list(range(col, col + len(var['options']))))
            col += len(var['options'])
    return out


def pool_key(opt):
    """What the pool depends on: the rows, their frame, the limits and the
    total (not the results), and the open Round and its Not scored rows,
    which decide which pre-mix make-up is used (T3 M-4)."""
    return insights._digest({
        "variables": opt.variables, "bounds": opt._get_bounds().tolist(),
        "limits": opt.constraints, "amounts": getattr(opt, "quantity_constraints", []),
        "total": getattr(opt, "formulation_total", None),
        "properties": opt.ingredient_properties, "premixes": opt.premixes,
        "fixed": opt._get_fixed_features(), "round": opt.pending_batch_no,
        "skipped": getattr(opt, "skipped", None), "seed": POOL_SEED, "n": POOL_SIZE})


def feasible_pool(opt):
    """POOL_SIZE allowed formulations, normalised (review I9): hit-and-run
    over the continuous columns inside every limit, the total an equality,
    the fixed rows pinned; a categorical row's option drawn uniformly (fixed
    seed). None when the limits leave no room or the sampler fails."""
    key = ("pool", pool_key(opt))
    with _CACHE_LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)      # T3 M-3: kept while it is in use
            return _CACHE[key]
    bounds = opt._get_bounds()
    dim = bounds.shape[1]
    try:
        with opt._premixes_for_generation():
            ineq = list(opt._get_botorch_constraints())
            total = _total_equality(opt)
        eq = []
        if total is not None:
            equality, band = total
            ineq = [c for c in ineq if not any(_same(c, b) for b in band)]
            eq.append(equality)
        for col, z in opt._get_fixed_features().items():
            eq.append((torch.tensor([col]), torch.tensor([1.0], dtype=torch.double), float(z)))
        groups = _categorical_columns(opt)
        for group in groups:
            for col in group:
                eq.append((torch.tensor([col]), torch.tensor([1.0], dtype=torch.double), 0.0))
        with insights.RNG_LOCK, torch.random.fork_rng():
            P = get_polytope_samples(
                POOL_SIZE, torch.stack([torch.zeros(dim), torch.ones(dim)]).double(),
                inequality_constraints=ineq or None, equality_constraints=eq or None,
                seed=POOL_SEED, n_thinning=POOL_THINNING)
        rng = random.Random(POOL_SEED)
        for group in groups:
            for k in range(P.shape[0]):
                P[k, group] = 0.0
                P[k, rng.choice(group)] = 1.0
        P = P.clamp(0.0, 1.0)
    except Exception:
        P = None
    _remember(key, P)
    return P


def pool_points(P):
    """How many distinct allowed formulations the pool holds (rounded to
    1e-6): hit-and-run returns POOL_SIZE rows or fails, but a (nearly)
    degenerate region gives the same point again and again (T3 M-1)."""
    if P is None:
        return 0
    return int(torch.unique(torch.round(P * 1e6), dim=0).shape[0])


def role(opt, row):
    """A row of the open Round's role (0.8.0 e2e fixes A2, review I3): the
    Compared with cell's own word, by FoodOptimizer.suggestion_role's
    distance rule, so the two tables never disagree. A row with a note (the
    user's own, a Repeat, an edited suggestion) is "your own formulation",
    by the cell's own rule (compared_with_text). Model-free: shown whatever
    the Prediction check says (it reverts 11152ff's withholding)."""
    if row.get('note'):
        return wording.own_source(row.get('note'))
    return opt.suggestion_role(row['recipe'])


# ------------------------------------------------------------------ #
#  §3.1 Why these suggestions (Make a round, a Round open)
# ------------------------------------------------------------------ #

@dataclass
class WhyRow:
    formulation: int
    own: bool           # a formulation of the user's own
    edited: bool
    pick: int | None    # its place among the Round's picks (1 = first); None for own
    mean: float
    sd: float
    value: float        # expected improvement (qUCB: optimistic score), points
    pending: bool       # valued with the earlier picks counted (picks 2+)
    nearest: int | None
    differences: list   # [(name, difference in its own unit, unit)], the two largest
    role: str           # role(): the Compared with cell's own word


@dataclass
class Why:
    kind: str           # acquisition(opt)
    exact: bool
    cold: bool          # no model yet: one line instead (review I12)
    rows: list          # [WhyRow]
    pool_mean: object   # tensors over the pool, or None
    pool_sd: object
    tested_mean: object
    tested_sd: object


def _differences(opt, recipe, other, k=2):
    """The `k` largest differences of `recipe` from `other` (both at the
    Formulation total), largest share of its span first, in each row's
    own unit (review M4): FoodOptimizer._variable_deltas, the rule the
    Compared with cell ranks by (0.8.0 e2e fixes B2, review M3), so the
    two tables never list different changes for one reason. A difference
    under one display step is left out (03 M6: "Honey 0.00 g")."""
    return [(name, d, opt.unit_of(name))
            for name, d in opt._variable_deltas(recipe, other)[:k]]


def _issued(opt, number):
    """The make-up ask() issued as formulation `number`, at the total: the
    open Round's row as generated, else its recorded or Not scored row."""
    for row in opt.pending_batch or []:
        if int(row['formulation']) == number:
            return insights.at_total(opt, row.get('original_recipe') or row['recipe'])
    ids = [int(x) for x in opt.formulation_ids]
    if number in ids:
        return insights.start_recipe(opt, ids.index(number))
    for row in getattr(opt, "skipped", []) or []:
        if int(row['formulation']) == number and row.get('recipe'):
            return insights.at_total(opt, row['recipe'])
    return None


def _picks(opt, gen, rows):
    """(own flags, pick numbers in the order ask() issued them): from the
    generation record's "picks" (T3 I-2), else — a Round made before they
    were kept — the rows without a note of their own (an edited row
    counts)."""
    if gen.picks:
        issued = [n for n in gen.picks]
        own = [int(row['formulation']) not in issued for row in rows]
        return own, issued
    own = [bool(row.get('note')) and not row.get('original_recipe') for row in rows]
    return own, [int(row['formulation']) for row, mine in zip(rows, own) if not mine]


def why(opt, fitted=None):
    """The open Round's rows valued by the acquisition that chose them
    (review I12): pick 1 exact (q = 1); picks 2+ with the earlier picks
    counted — qLogNEI's improvement over the tested and earlier picks, the
    increment qLogEI's joint value gains from this pick (T3 I-1); qUCB has
    no such form, so every row shows its own optimistic score. The user's
    own and edited rows at their amounts as they stand, marked. The role
    is role()'s distance rule, never the pool. None without an open Round."""
    r = opt.pending_batch_no
    if not opt.pending_batch or r is None:
        return None
    gen = generation(opt, r)
    kind = acquisition(opt)
    if gen.regime != "warm":
        return Why(kind, False, True, [], None, None, None, None)
    model = rebuild(opt, gen, fitted)
    if model is None:
        return Why(kind, False, True, [], None, None, None, None)
    rows = list(opt.pending_batch)
    recipes = [insights.at_total(opt, row['recipe']) for row in rows]
    own, issued = _picks(opt, gen, rows)
    P = points(opt, recipes)
    alone = evaluate(opt, model, P)
    pool = feasible_pool(opt)
    pool_values = evaluate(opt, model, pool) if pool is not None else None
    tested = evaluate(opt, model, model.X)
    out = []
    for k, row in enumerate(rows):
        no = int(row['formulation'])
        pick = issued.index(no) + 1 if not own[k] and no in issued else None
        value = float(alone.value[k])
        pending = pick is not None and pick > 1 and kind != "qucb"
        if pending:
            earlier = [_issued(opt, n) for n in issued[:pick - 1]]
            earlier = points(opt, [e for e in earlier if e is not None])
            if earlier.shape[0] == 0:
                pending = False
            elif kind == "qlogei":
                joint = float(evaluate(opt, model, P[k:k + 1], pending=earlier).value[0])
                value = max(0.0, joint - joint_value(model, earlier))
            else:
                value = float(evaluate(opt, model, P[k:k + 1], pending=earlier).value[0])
        i, _ = insights.nearest(opt, recipes[k])
        near = None if i is None else int(opt.formulation_ids[i])
        diffs = [] if i is None else _differences(opt, recipes[k], insights.start_recipe(opt, i))
        out.append(WhyRow(no, own[k], bool(row.get('original_recipe')) and not own[k],
                          pick, float(alone.mean[k]), float(alone.sd[k]), value,
                          pending, near, diffs, role(opt, row)))
    return Why(kind, model.exact, False, out,
               None if pool_values is None else pool_values.mean,
               None if pool_values is None else pool_values.sd,
               tested.mean, tested.sd)


# ------------------------------------------------------------------ #
#  §3.2 Expected improvement by Round (Insights, Progress)
# ------------------------------------------------------------------ #

@dataclass
class RoundGain:
    round: int
    best_measured: float | None     # Best so far among the rows recorded before it
    highest: float | None           # highest predicted mean over the pool
    gain: float | None              # the first pick's expected improvement; None: no bar
    exact: bool
    why_none: str | None            # "cold", "own" or None


def _first_pick(opt, r, gen=None):
    """The Round's first generated row, at the total: the first number ask()
    issued, from the generation record (T3 I-2); without one, the open
    Round's first row without a note of its own, else the recorded row with
    the lowest number among those without one (an edited row counts). None
    for a Round of the user's own rows only."""
    gen = generation(opt, r) if gen is None else gen
    if gen.picks:
        return _issued(opt, gen.picks[0])
    if r == opt.pending_batch_no and opt.pending_batch:
        for row in opt.pending_batch:
            if not row.get('note') or row.get('original_recipe'):
                return insights.at_total(opt, row.get('original_recipe') or row['recipe'])
    made = [(int(n), i) for i, (n, b) in enumerate(zip(opt.formulation_ids, opt.batch_history))
            if b == r and (not opt.notes_history[i]
                           or opt.notes_history[i] == wording.EDITED_FORMULATION)]
    if not made:
        return None
    return insights.start_recipe(opt, min(made)[1])


def round_gain(opt, r, fitted=None, highest=True):
    """RoundGain for Round `r`: the expected improvement its first pick was
    chosen for (exact from its record, else recomputed with the display
    model of the rows before it), the highest predicted mean over the pool
    (`highest` False: not worked out, the pool left alone) and Best so far
    before it. `fitted`: the display model of those rows, when the caller
    has it."""
    gen = generation(opt, r)
    candidates = set(opt.best_candidates())            # T3 M-8: once, not per row
    before = [i for i in gen.rows if i in candidates]
    best = max((float(opt.Y_history[i]) for i in before), default=None)
    if gen.regime != "warm":
        return RoundGain(r, best, None, None, False, "cold")
    pick = _first_pick(opt, r, gen)
    if pick is None:
        return RoundGain(r, best, None, None, False, "own")
    model = rebuild(opt, gen, fitted)
    if model is None:
        return RoundGain(r, best, None, None, False, "cold")
    value = float(evaluate(opt, model, points(opt, [pick])).value[0])
    pool = feasible_pool(opt) if highest else None
    top = None if pool is None else float(evaluate(opt, model, pool).mean.max())
    return RoundGain(r, best, top, value, model.exact, None)


def round_history(opt, models=None):
    """round_gain for every Round made after MIN_ROWS rows, in order.
    `models`: {Round: the display model of the rows before it} (the
    Round-ahead check's own Overall-score model), so a recomputed Round
    costs no new fit (review I10)."""
    models = models or {}
    out = []
    for r, first in insights.round_order(opt):
        if first < insights.MIN_ROWS:
            continue
        out.append(round_gain(opt, r, models.get(r)))
    if opt.pending_batch_no is not None and opt.pending_batch and all(
            g.round != opt.pending_batch_no for g in out):
        out.append(round_gain(opt, opt.pending_batch_no, models.get(opt.pending_batch_no)))
    return out


def latest_gain(opt, fitted=None):
    """§3.2's line (review I5): the open Round's figure, else the last
    Round made, with its number; the pool is not needed for it. None
    without a Round."""
    r = opt.pending_batch_no if opt.pending_batch else None
    if r is None:
        order = insights.round_order(opt)
        r = order[-1][0] if order else None
    return None if r is None else round_gain(opt, r, fitted, highest=False)


def next_model(opt):
    """The suggestion model the next Generate fits, rebuilt exactly as an
    exact Round is (fit_default under RNG_LOCK): today's fully measured rows
    as the model learns them (model_X), the seed ask() will use
    (next_formulation_no), botorch's default fit. None while cold. So the
    figure before Generate and the one Generate records come from one
    model (final review I-1, coordinator ruling)."""
    rows = _complete_rows(opt)
    regime = "warm" if opt.complete_count() >= COLD_START_RUNS else "cold"
    return rebuild(opt, Generation(None, int(opt.next_formulation_no), rows, regime, True))



@dataclass
class MapValues:
    """The explored–unexplored map's numbers (§8.2, review R-I16), all from
    one suggestion model: its own mean and sd (no Repeat noise, no
    calibration) over the allowed pool, at the rows it learned from, and at
    the open Round's rows when the model is that Round's."""
    pool_mean: object       # tensors over feasible_pool, or None without a pool
    pool_sd: object
    tested_mean: object     # at the model's training rows
    tested_sd: object
    tested: list            # their Formulation numbers, in the same order
    picks: list             # [(Formulation number, mean, sd)] of the open Round's rows
    exact: bool             # the model is the Round's own, rebuilt exactly
    round: int | None       # the open Round the picks are from; None: the next Generate's


def map_values(opt, model):
    """W3, Insights for a first-time user (spec 2026-10-01 rev. 2 §8.2,
    review R-I16): the explored–unexplored map's values, from `model` (the
    open Round's rebuilt model, else next_model) over feasible_pool(opt),
    the rows it learned from and — the open Round's model only — that
    Round's rows at their amounts as they stand, as the Why fold values
    them. Through evaluate(), so the global RNG never moves and a question
    the Why fold already asked is answered from the cache. None without a
    model."""
    if model is None:
        return None
    pool = feasible_pool(opt)
    pool_values = evaluate(opt, model, pool) if pool is not None else None
    tested = evaluate(opt, model, model.X)
    picks = []
    r = model.gen.round if model.gen is not None else None
    if r is not None and r == opt.pending_batch_no and opt.pending_batch:
        rows = list(opt.pending_batch)
        got = evaluate(opt, model, points(opt, [insights.at_total(opt, row['recipe'])
                                                for row in rows]))
        picks = [(int(row['formulation']), float(got.mean[k]), float(got.sd[k]))
                 for k, row in enumerate(rows)]
    ids = [int(n) for n in opt.formulation_ids]
    return MapValues(None if pool_values is None else pool_values.mean,
                     None if pool_values is None else pool_values.sd,
                     tested.mean, tested.sd, [ids[i] for i in model.gen.rows], picks,
                     model.exact, r if picks else None)


# ask()'s own optimizer settings (food_bo._ask_optimize; a test ties them).
ASK_RESTARTS = 20
ASK_RAW_SAMPLES = 1024


def next_optimum(opt, model):
    """The expected improvement, in points, of the formulation the next
    Generate would pick first (final review I-1, coordinator ruling): a
    q = 1 optimize_acqf on `model`'s acquisition with ask()'s bounds,
    limits, fixed rows and settings, from the RNG state ask() has there
    (sequential=True makes ask()'s first pick this very search), then the
    point decoded, snapped and stepped as ask() issues it and valued at the
    total, as the Why fold values pick 1. In this process, inside
    insights._isolated: the global RNG never moves. Raises if the search
    fails (the caller falls back to the pool).

    Kept in this process's output cache, keyed by the model and the setup
    (answer-key refresh item 2): a page rerun can leave the run that
    started the search finishing in its own thread, with its session-state
    write dropped; its answer still serves the next run."""
    key = insights._digest({"next_optimum": _key(opt, model, torch.zeros(0), None),
                            "setup": insights.setup_key(opt)})
    with _CACHE_LOCK:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]
    value = _next_optimum_here(opt, model)
    _remember(key, value)
    return value


def _next_optimum_here(opt, model):
    from botorch.optim import optimize_acqf
    from botorch.utils.transforms import unnormalize
    dim = model.X.shape[-1]
    with opt._premixes_for_generation(None):
        bounds = opt._get_bounds()
        ineq = opt._get_botorch_constraints()
        fixed = opt._get_fixed_features() or None
        with insights._isolated():
            acqf = build_acqf(model.cfg, model.gp, model.X, model.Y, model.rng)
            x, _ = optimize_acqf(
                acq_function=acqf, bounds=torch.stack([torch.zeros(dim), torch.ones(dim)]).double(),
                q=1, num_restarts=ASK_RESTARTS, raw_samples=ASK_RAW_SAMPLES, sequential=True,
                inequality_constraints=ineq, fixed_features=fixed)
        recipe = opt._decode(unnormalize(x[0], bounds).detach().numpy().flatten())
        if opt.has_formulation_total():
            recipe = opt._snapped_if_it_still_fits(recipe)
    recipe = opt._settings_on_their_step(recipe)
    return float(evaluate(opt, model, points(opt, [insights.at_total(opt, recipe)])).value[0])


def next_gain(opt, model):
    """§3.2's figure with no Round open (final review I-1): the expected
    improvement of the formulation the next Generate picks first under
    `model` (next_model), by ask()'s own search (next_optimum); the largest
    over the allowed pool only if that search fails. RoundGain(None, …). A Round already
    made is history, in round_history; its figure was for formulations
    since measured. None without a pool or a model."""
    rows = _complete_rows(opt)
    candidates = set(opt.best_candidates())
    best = max((float(opt.Y_history[i]) for i in rows if i in candidates), default=None)
    if len(rows) < COLD_START_RUNS:
        return RoundGain(None, best, None, None, False, "cold")
    if model is None:
        return None
    try:
        gain = next_optimum(opt, model)
    except Exception:
        # The search failed: the largest over the allowed pool instead.
        pool = feasible_pool(opt)
        if pool is None:
            return None
        gain = float(evaluate(opt, model, pool).value.max())
    return RoundGain(None, best, None, gain, False, None)


# ------------------------------------------------------------------ #
#  C1 (spec 2026-10-01; 04 I-1, review I11): the next Round's figure is
#  kept beside the project (insights_cache, `gain`), so a reopen neither
#  fits next_model nor searches: 1.6 s at 60 x 4, 2.4 s at 120 x 6.
# ------------------------------------------------------------------ #

# Part of next_gain_key: bump it whenever a change here, in next_model or
# in ask()'s first-pick search could change the figure, and a saved one
# from before is never used again.
GAIN_SALT = "0.8.0-e2e-1"


def next_gain_key(opt):
    """Everything the next Round's figure depends on (review I11): the
    setup (insights.setup_key: the rows and results, model settings,
    Lowest/Highest, calculations and fixed rows, every limit, the total,
    the ingredient properties, pre-mixes and Not scored), the seed the
    next Generate uses (next_formulation_no: "Generate a different batch"
    or a discarded Round moves it with the data unchanged), robust, the
    model settings, the library versions, ask()'s search settings, the
    cold-start count and GAIN_SALT."""
    return insights._digest({
        "setup": insights.setup_key(opt), "seed": int(opt.next_formulation_no),
        "robust": bool(getattr(opt, "robust", False)), "cfg": getattr(opt, "bo_config", None),
        "versions": list(_VERSIONS), "restarts": ASK_RESTARTS, "raw": ASK_RAW_SAMPLES,
        "cold": COLD_START_RUNS, "salt": GAIN_SALT})


def _gain_cache_key(key):
    return insights._digest({"next_gain": key})


def saved_next_gain(opt, saved=None):
    """The next Round's figure without a fit or a search, when it is known:
    `saved`, a (next_gain_key, RoundGain) read from beside the project,
    when its key is this setup's; else this process's output cache (a
    figure another session or run of the app worked out). None otherwise."""
    key = next_gain_key(opt)
    if saved is not None and saved[0] == key and saved[1] is not None:
        _remember(_gain_cache_key(key), saved[1])
        return saved[1]
    with _CACHE_LOCK:
        hit = _CACHE.get(_gain_cache_key(key))
        if hit is not None:
            _CACHE.move_to_end(_gain_cache_key(key))
        return hit


def next_gain_for(opt, saved=None):
    """next_gain under next_model, unless saved_next_gain knows it: what a
    screen shows with no Round open. A figure worked out is kept in the
    process's cache under next_gain_key (a failure, None, is not), beside
    next_optimum's own entry, which still serves a run that has fitted its
    model."""
    known = saved_next_gain(opt, saved)
    if known is not None:
        return known
    key = next_gain_key(opt)
    got = next_gain(opt, next_model(opt))
    if got is not None:
        _remember(_gain_cache_key(key), got)
    return got


@contextlib.contextmanager
def fresh():
    """Tests: empty the output cache before and after."""
    _CACHE.clear()
    try:
        yield
    finally:
        _CACHE.clear()
