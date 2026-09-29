"""What the Insights tab shows, worked out from a project without changing it.

Nothing here writes to the optimizer. Every model is fitted for display only,
with the optimizer's own builder so an expert's model settings apply, inside
a forked torch RNG so the next suggestion is exactly what it would have been.
The suggestions themselves still come from the one Overall-score model that
FoodOptimizer._ask_optimize fits; the models here never reach it."""
import contextlib
import hashlib
import json
import math
import statistics
from dataclasses import dataclass

import torch
from botorch.acquisition import PosteriorMean
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood

import wording
from food_bo import CLOSE_TO_THE_BEST, number_list

SCORE = "__overall_score__"
MIN_ROWS = 5
Z95 = 1.959963984540054
_SEED = 20260928


@contextlib.contextmanager
def _isolated():
    """Repeatable, and invisible to everything outside: the global RNG is
    restored on the way out."""
    with torch.random.fork_rng():
        torch.manual_seed(_SEED)
        yield


@dataclass
class Fitted:
    name: str
    model: object
    error: str
    n_rows: int
    low: float
    high: float
    widen: float = 1.0


@dataclass
class Prediction:
    mean: float
    low: float
    high: float
    # The model's own mean and its calibrated predictive sd (already
    # widened), before any clipping to the Range: what a miss in standard
    # deviations is measured against. None where a caller made its own.
    raw_mean: float | None = None
    sd: float | None = None


def outputs(opt):
    out = [(o['name'], float(o['min_val']), float(o['max_val']))
           for o in opt.objectives]
    out.append((SCORE, 0.0, float(opt.utility_ceiling())))
    return out


def measured(opt, name, i):
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


def _fit_gp(opt, name, rows):
    """One GP on `rows` (all measured), with the optimizer's own builder so
    an expert's model settings apply. Raises if the fit fails."""
    bounds = opt._get_bounds()
    with _isolated():
        X = normalize(torch.tensor([opt.X_history[i] for i in rows],
                                   dtype=torch.double), bounds)
        Y = torch.tensor([measured(opt, name, i) for i in rows],
                         dtype=torch.double).unsqueeze(-1)
        gp = opt._build_gp(X, Y, bounds.shape[1], None)
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
    return gp


def fit(opt, name, rows):
    lo, hi = _range_of(opt, name)
    use = [i for i in rows if measured(opt, name, i) is not None]
    if len(use) < MIN_ROWS:
        return Fitted(name, None, None, len(use), lo, hi)
    try:
        gp = _fit_gp(opt, name, use)
    except Exception as exc:   # one model failing must not take the tab down
        return Fitted(name, None, str(exc) or type(exc).__name__, len(use), lo, hi)
    return Fitted(name, gp, None, len(use), lo, hi, _widen(opt, name, use))


FOLDS = 5


def _cv_z(opt, name, use):
    """Standardised misses (y - mean) / sd of each row when it was held
    out: rows go to FOLDS folds by row order, and the model is fitted again
    (hyperparameters too) on the other folds, then predicts the held-out
    fold with the variation between Repeats included. A fold whose fit
    fails is skipped. Only `use` is ever seen, so a Round-ahead check stays
    honest."""
    bounds = opt._get_bounds()
    z = []
    for k in range(FOLDS):
        held = [i for n, i in enumerate(use) if n % FOLDS == k]
        if not held:
            continue
        try:
            gp = _fit_gp(opt, name, [i for n, i in enumerate(use) if n % FOLDS != k])
            with torch.no_grad(), _isolated():
                X = normalize(torch.tensor([opt.X_history[i] for i in held],
                                           dtype=torch.double), bounds)
                post = gp.posterior(X, observation_noise=True)
                mean = post.mean.reshape(-1).tolist()
                sd = post.variance.clamp_min(0.0).sqrt().reshape(-1).tolist()
        except Exception:
            continue
        for i, m, s in zip(held, mean, sd):
            if s > 0:
                z.append((measured(opt, name, i) - m) / s)
    return z


def _widen(opt, name, use):
    """How much to widen this model's likely range: the root-mean-square
    of its cross-validated misses, never below 1. With few rows the fitted
    noise comes out too small and the range too narrow (Round-ahead
    coverage near 78 % at realistic noise); refitting on each fold sees
    that, where the same-rows leave-one-out did not. Only widens; with no
    usable fold the range stays as fitted."""
    try:
        z = _cv_z(opt, name, use)
        c = math.sqrt(sum(v * v for v in z) / len(z)) if z else 1.0
    except Exception:
        return 1.0
    return max(1.0, c) if math.isfinite(c) else 1.0


def predict(opt, fitted, X_rows):
    if fitted.model is None or not X_rows:
        return []
    bounds = opt._get_bounds()
    with torch.no_grad(), _isolated():
        X = normalize(torch.tensor(X_rows, dtype=torch.double), bounds)
        post = fitted.model.posterior(X, observation_noise=True)
        mean = post.mean.reshape(-1).tolist()
        sd = (post.variance.clamp_min(0.0).sqrt() * fitted.widen).reshape(-1).tolist()
    clip = lambda v: min(fitted.high, max(fitted.low, v))
    # The band is worked from the model's own mean, then clipped: centring it
    # on a mean already clipped to the Range would shift it.
    return [Prediction(clip(m), clip(m - Z95 * s), clip(m + Z95 * s), m, s)
            for m, s in zip(mean, sd)]


def fit_all(opt):
    rows = list(range(len(opt.Y_history)))
    return {name: fit(opt, name, rows) for name, _, _ in outputs(opt)}


def _digest(payload):
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def _model_settings(opt):
    """What every model here depends on besides its rows: the measurements
    (their Ranges, goals and shares, so the Overall score too), an expert's
    model settings, and the rows' Lowest-Highest and encoding."""
    return {"objectives": opt.objectives, "cfg": getattr(opt, "bo_config", None),
            "bounds": opt._get_bounds().tolist()}


def data_key(opt):
    """Everything a fitted model or a Round-ahead check depends on."""
    return _digest({
        "X": opt.X_history, "Y": opt.Y_history, "R": opt.results_history,
        "ids": opt.formulation_ids, "rounds": opt.batch_history,
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


def round_order(opt):
    seen, out = set(), []
    for i, r in enumerate(opt.batch_history):
        if r is not None and r not in seen:
            seen.add(r)
            out.append((r, i))
    return out


@dataclass
class Check:
    formulation: int
    round: int
    name: str
    predicted: float
    low: float
    high: float
    measured: float
    inside: bool
    fully_measured: bool
    raw_mean: float | None = None   # Prediction.raw_mean / .sd: for the misses chart
    sd: float | None = None


def _row_key(opt, i):
    """One recorded row as a Round-ahead check sees it: its encoded make-up,
    its Overall score, its results and its Formulation number (which
    decides the measurements that apply to it)."""
    at = lambda seq: seq[i] if i < len(seq) else None
    return _digest([opt.X_history[i], opt.Y_history[i], at(opt.results_history),
                    at(opt.formulation_ids)])


def round_keys(opt):
    """{Round: key}: everything that Round's checks depend on and nothing
    else, so a key changes only when its checks could. That is the rows
    recorded before the Round's first row (what its models learn from),
    the Round's own rows (what they predict) and the model settings
    data_key covers; not a later Round, nor a row's Round number."""
    settings = _digest(_model_settings(opt))
    rows = [_row_key(opt, i) for i in range(len(opt.Y_history))]
    return {r: _digest({"settings": settings, "round": r, "first": first,
                        "train": rows[:first],
                        "targets": [(i, rows[i]) for i, b in enumerate(opt.batch_history)
                                    if b == r]})
            for r, first in round_order(opt)}


def _check_round(opt, r, first, names):
    checks = []
    train = list(range(first))
    targets = [i for i, b in enumerate(opt.batch_history) if b == r]
    for name in names:
        fitted = fit(opt, name, train)
        if fitted.model is None:
            continue
        rows = [i for i in targets if measured(opt, name, i) is not None]
        try:
            preds = predict(opt, fitted, [opt.X_history[i] for i in rows])
        except Exception:   # like a failed fit: no checks for this pair
            continue
        for i, p in zip(rows, preds):
            value = measured(opt, name, i)
            checks.append(Check(
                int(opt.formulation_ids[i]), r, name, p.mean, p.low, p.high,
                value, p.low <= value <= p.high, opt.fully_measured(i),
                p.raw_mean, p.sd))
    return checks


def check_rounds(opt, cache=None, checking=None):
    """Each Round's formulations predicted from the rows recorded before
    it, Round by Round.

    `cache`, a dict the caller keeps between calls, holds each Round's
    checks under its round_keys key: a Round whose key is there is not
    fitted again, and on the way out the dict holds this project's Rounds
    alone. Every fit runs in its own `_isolated()` with the same seed, so a
    Round's checks come out the same whichever other Rounds are fitted with
    it: kept or fresh, the answer is identical.

    `checking(r, last)`, when given, is entered around each Round actually
    fitted (one that could be checked), for the screen's progress line."""
    names = [n for n, _, _ in outputs(opt)]
    order = round_order(opt)
    keys = round_keys(opt) if cache is not None else {}
    checks = []
    for r, first in order:
        key = keys.get(r)
        if key is not None and key in cache:
            checks.extend(cache[key])
            continue
        line = (checking(r, order[-1][0]) if checking is not None and first >= MIN_ROWS
                else contextlib.nullcontext())
        with line:
            found = _check_round(opt, r, first, names)
        if cache is not None:
            cache[key] = found
        checks.extend(found)
    if cache is not None:
        for gone in set(cache) - set(keys.values()):
            del cache[gone]
    return checks


def rounds_not_checked(opt):
    """The Rounds with fewer than MIN_ROWS formulations recorded before
    them: nothing earlier to predict them from, so they are never checked."""
    return [r for r, first in round_order(opt) if first < MIN_ROWS]


# Wave C: how far each prediction missed, in standard deviations.

@dataclass
class Miss:
    formulation: int
    round: int
    name: str
    z: float
    inside: bool


def misses(checks):
    """Each check's miss in standard deviations of its likely range:
    (measured - unclipped mean) / calibrated sd. A check with no sd is
    left out."""
    return [Miss(c.formulation, c.round, c.name, (c.measured - c.raw_mean) / c.sd, c.inside)
            for c in checks if c.sd is not None and c.sd > 0 and c.raw_mean is not None]


MISSES_GATE = 10


def misses_ready(checks):
    """The same threshold as the headline: 10 measurement checks."""
    return sum(c.name != SCORE for c in checks) >= MISSES_GATE


def binomial_tail(m, n, p=0.05):
    if m <= 0:
        return 1.0
    return sum(math.comb(n, k) * p ** k * (1 - p) ** (n - k)
               for k in range(m, n + 1))


@dataclass
class Headline:
    state: str
    rounds: list
    inside: int
    total: int
    verdict: str
    score_below: bool

    def _score_rule_for_tests(self):
        return self.score_below


def _ordered_rounds(checks):
    out = []
    for c in checks:
        if c.round not in out:
            out.append(c.round)
    return out


def headline(opt, checks):
    values = [c for c in checks if c.name != SCORE]
    rounds = _ordered_rounds(values)
    score_rounds = _ordered_rounds([c for c in checks if c.name == SCORE])
    if not rounds:
        chosen = score_rounds[-3:]
        return Headline("none", [], 0, 0, None,
                        _score_below(checks, chosen))
    k = min(3, len(rounds))
    chosen = rounds[-k:]
    while (sum(c.round in chosen for c in values) < 10 and k < len(rounds)):
        k += 1
        chosen = rounds[-k:]
    picked = [c for c in values if c.round in chosen]
    below = _score_below(checks, chosen)
    if len(picked) < 10:
        return Headline("early", chosen, sum(c.inside for c in picked),
                        len(values), None, below)
    inside = sum(c.inside for c in picked)
    tail = binomial_tail(len(picked) - inside, len(picked))
    verdict = ("holding" if tail >= 0.10 else
               "caution" if tail >= 0.01 else "unreliable")
    return Headline("verdict", chosen, inside, len(picked), verdict, below)


def _score_below(checks, chosen):
    scores = [c for c in checks if c.name == SCORE and c.round in chosen
              and c.fully_measured]
    return (len(scores) >= 10 and
            sum(c.measured < c.predicted for c in scores) >= 0.8 * len(scores))


@dataclass
class DetailRow:
    name: str
    checked: int
    inside: int
    half_width: float
    typical_miss: float
    r2: float


def details(opt, checks):
    rows = []
    for name, _, _ in outputs(opt):
        mine = [c for c in checks if c.name == name]
        if not mine:
            continue
        meas = [c.measured for c in mine]
        mean = sum(meas) / len(meas)
        ss_tot = sum((m - mean) ** 2 for m in meas)
        ss_res = sum((c.measured - c.predicted) ** 2 for c in mine)
        rows.append(DetailRow(
            name, len(mine), sum(c.inside for c in mine),
            statistics.median((c.high - c.low) / 2 for c in mine),
            statistics.median(abs(c.measured - c.predicted) for c in mine),
            None if len(mine) < MIN_ROWS or ss_tot == 0 else 1 - ss_res / ss_tot))
    return rows


_EPS = 1e-9


def _var(opt, name):
    return opt._by_name()[name]


def _is_ingredient(v):
    return v.get('category', 'ingredient') == 'ingredient'


def path_choices(opt):
    varying = [v for v in opt.varying_variables() if v.get('type') == 'continuous']
    return ([v['name'] for v in varying if _is_ingredient(v)] +
            [v['name'] for v in varying if not _is_ingredient(v)])


def is_setting(opt, name):
    """Whether this row is a process setting (it has values, not amounts)."""
    return not _is_ingredient(_var(opt, name))


def has_setting_choice(opt):
    """Whether any process setting is among the path choices."""
    return any(not _is_ingredient(_var(opt, n)) for n in path_choices(opt))


def at_total(opt, recipe):
    """A formulation's make-up at the project's Formulation total: a row
    made to another batch size is brought back to the total, so a
    prediction or a comparison never moves with the Batch size box."""
    recipe = dict(recipe)
    if opt.has_formulation_total():
        total = float(opt.formulation_total)
        own = opt.ingredient_total(recipe)
        tolerance = total * opt.FORMULATION_TOTAL_TOLERANCE
        if own > 0 and abs(own - total) > tolerance:
            return opt.scaled_recipe(recipe, scale_to=total)
    return opt.fill_formulas(recipe)


def start_recipe(opt, i):
    return at_total(opt, opt._recorded_recipe(i))


@dataclass
class PathEnd:
    amount: float
    reason: str | None


@dataclass
class Path:
    name: str
    unit: str
    mode: str
    fill_name: str | None
    amounts: list
    recipes: list
    start_amount: float
    low_end: PathEnd
    high_end: PathEnd
    start_breaks: str | None
    skipped: list
    outside: bool
    tested: tuple | None
    fixed: list
    follows: list


def _mode(opt, name):
    if not _is_ingredient(_var(opt, name)):
        return "process"
    if opt.balance_row_name():
        return "fill"
    if opt.has_formulation_total():
        return "proportional"
    return "free"


def _others(opt, name):
    return [v for v in opt.varying_variables()
            if _is_ingredient(v) and v['name'] != name and v.get('type') == 'continuous']


def _widened(var, start):
    """This row's own Lowest-Highest, widened to include its start amount —
    the same rule the chosen row gets in `build_path`, so a formulation
    recorded before a range was narrowed still draws a path instead of
    reading as permanently past a bound it predates."""
    lo, hi = (float(b) for b in var['bounds'])
    s = float(start.get(var['name'], lo))
    return min(lo, s), max(hi, s)


def _recipe_at(opt, name, start, mode, t):
    r = dict(start)
    delta = t - float(start[name])
    r[name] = t
    if mode == "proportional" and abs(delta) > _EPS:
        others = _others(opt, name)
        if delta > 0:
            room = {v['name']: float(start[v['name']]) - _widened(v, start)[0]
                    for v in others}
        else:
            room = {v['name']: _widened(v, start)[1] - float(start[v['name']])
                    for v in others}
        total_room = sum(max(0.0, x) for x in room.values())
        if total_room <= _EPS or abs(delta) > total_room + _EPS:
            return None   # nothing can give way, or not enough of it
        for n, x in room.items():
            r[n] = float(start[n]) - delta * max(0.0, x) / total_room
    return opt.fill_formulas(r)


def _limits_broken(opt, recipe, moving, start):
    """The first limit `recipe` breaks, as the text Set up shows, or None.

    A moving OTHER ingredient running past its own Lowest/Highest is not
    named here in practice: `_recipe_at` already reports that as the
    aggregate "no room" wording (see `_reason`) before a recipe naming one
    specific ingredient is ever produced, in proportional mode, for the
    direction that ran out. This loop stays as the defensive fallback for
    any other case a start already widened away from — see `_widened`."""
    for n in moving:
        lo, hi = _widened(_var(opt, n), start)
        v = float(recipe.get(n, 0.0))
        if v < lo - 1e-6 * (1 + abs(lo)):
            return wording.path_past_lowest(n)
        if v > hi + 1e-6 * (1 + abs(hi)):
            return wording.path_past_highest(n)
    for n, form in opt._resolved_forms().items():
        v = float(recipe.get(n, 0.0))
        if v < -1e-6 * (1.0 + abs(v)):
            return wording.path_below_zero(n)
    for qc in getattr(opt, 'quantity_constraints', []):
        total = sum(float(recipe.get(n, 0.0)) for n in qc['ingredients'])
        slack = 1e-6 * (1.0 + abs(total))
        if qc['min'] is not None and total < float(qc['min']) - slack:
            return opt.limit_text(qc)
        if qc['max'] is not None and total > float(qc['max']) + slack:
            return opt.limit_text(qc)
    scale = 1.0 + sum(abs(float(a or 0.0)) for a in recipe.values())
    for c in opt.constraints:
        if opt.ingredients_without_property(c['metric']):
            continue
        for bound, sign in ((c['min'], -1), (c['max'], 1)):
            if bound is None:
                continue
            slack = 1e-9 * scale * (1.0 + abs(float(bound)))
            res = opt._property_residual(recipe, c['metric'], bound)
            if sign * res > slack:
                return opt.property_limit_text(c)
    return None


def _reason(opt, name, start, mode, t, moving):
    r = _recipe_at(opt, name, start, mode, t)
    if r is None:
        return (wording.PATH_NO_ROOM_DOWN if t > start[name]
                else wording.PATH_NO_ROOM_UP), None
    return _limits_broken(opt, r, moving, start), r


def _edge(opt, name, start, mode, moving, ok_t, bad_t):
    """Bisect between an allowed amount and a broken one."""
    for _ in range(40):
        mid = (ok_t + bad_t) / 2
        if _reason(opt, name, start, mode, mid, moving)[0] is None:
            ok_t = mid
        else:
            bad_t = mid
    return ok_t, _reason(opt, name, start, mode, bad_t, moving)[0]


def build_path(opt, name, start, points=60):
    var = _var(opt, name)
    mode = _mode(opt, name)
    lo, hi = (float(b) for b in var['bounds'])
    s = float(start[name])
    outside = s < lo - _EPS or s > hi + _EPS
    lo, hi = min(lo, s), max(hi, s)
    moving = [v['name'] for v in _others(opt, name)] if mode == "proportional" else []
    skipped = [(opt.property_limit_text(c), c['metric']) for c in opt.constraints
               if (c.get('min') is not None or c.get('max') is not None)
               and opt.ingredients_without_property(c['metric'])]
    fixed = [(v['name'], float(start.get(v['name'], v['bounds'][0])), opt.unit_of(v['name']))
             for v in opt.fixed_variables()]
    watched = {name, *moving}
    follows = [n for n, f in opt._resolved_forms().items()
               if n != opt.balance_row_name() and watched & set(f.names())]
    tested = _tested_span(opt, name)
    base = dict(name=name, unit=opt.unit_of(name), mode=mode,
                fill_name=opt.balance_row_name() if mode == "fill" else None,
                start_amount=s, skipped=skipped, outside=outside, tested=tested,
                fixed=fixed, follows=follows)
    start_reason, start_recipe_value = _reason(opt, name, start, mode, s, moving)
    if start_reason is not None:
        return Path(amounts=[], recipes=[], low_end=PathEnd(s, None),
                    high_end=PathEnd(s, None), start_breaks=start_reason, **base)
    grid = sorted({lo + (hi - lo) * k / (points - 1) for k in range(points)} | {s})
    ends = {}
    computed = {s: start_recipe_value}   # each grid point's recipe, computed once
    for side, seq in (("low", [t for t in reversed(grid) if t <= s]),
                      ("high", [t for t in grid if t >= s])):
        last = s
        ends[side] = PathEnd(seq[-1], None)
        for t in seq:
            reason, r = _reason(opt, name, start, mode, t, moving)
            if reason is None:
                computed[t] = r
                last = t
                continue
            amount, why = _edge(opt, name, start, mode, moving, last, t)
            ends[side] = PathEnd(amount, why)
            break
    keep = sorted({t for t in grid if ends["low"].amount <= t <= ends["high"].amount}
                  | {ends["low"].amount, ends["high"].amount})
    recipes = [computed[t] if t in computed else _recipe_at(opt, name, start, mode, t)
               for t in keep]
    return Path(amounts=keep, recipes=recipes, low_end=ends["low"],
                high_end=ends["high"], start_breaks=None, **base)


def _tested_span(opt, name):
    values = [float(start_recipe(opt, i).get(name, 0.0))
              for i in range(len(opt.Y_history))]
    return (min(values), max(values)) if values else None


def path_predictions(opt, fitted, path):
    X = [opt._encode(r) for r in path.recipes]
    return {name: predict(opt, f, X) for name, f in fitted.items()
            if f.model is not None}


@dataclass
class BestFacts:
    score: float
    formulation: int
    round: int | None
    rounds_since: int
    latest: bool
    without: list       # measurements Best so far is scored without ([] as a rule)


def best_facts(opt):
    i = opt.best_index()
    if i is None:
        return None
    order = round_order(opt)
    since = [r for r, first in order if first > i]
    r = opt.batch_history[i]
    latest = r is not None and bool(order) and order[-1][0] == r
    return BestFacts(float(opt.Y_history[i]), int(opt.formulation_ids[i]), r,
                     len(since), latest, opt.best_scored_without())


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


def repeat_spread(opt):
    """The largest |Overall score difference| between any two formulations
    that `_same` calls a match, both among opt.best_candidates(): a missing
    measurement counts as 0 in the Overall score, so a partly measured
    repeat would read as a spread nobody saw at the bench, and a score
    rescaled without a measurement is not on the others' footing.

    Pairwise, not anchor groups: `_same`'s per-row tolerance is not
    transitive (a chain of small steps can drift past it end to end), so
    comparing each new row only against one earlier "anchor" row can miss a
    matching pair -- and the largest spread -- entirely. Every pair of
    formulations is compared independently instead."""
    rows = opt.best_candidates()
    recipes = {i: start_recipe(opt, i) for i in rows}
    best = None
    for a, i in enumerate(rows):
        for j in rows[a + 1:]:
            if _same(opt, recipes[i], recipes[j]):
                spread = abs(opt.Y_history[i] - opt.Y_history[j])
                if best is None or spread > best:
                    best = spread
    return best


# Wave C: two measurements against each other (measured values only).

def pair_choices(opt):
    """The measurements, largest Share of score first (ties keep the
    project's order): the two defaults of "Can you have both?" lead."""
    order = {o['name']: k for k, o in enumerate(opt.objectives)}
    return [o['name'] for o in sorted(
        opt.objectives, key=lambda o: (-float(o['weight']), order[o['name']]))]


@dataclass
class PairPoint:
    formulation: int
    a: float
    b: float
    score: float | None     # None: still waiting for results
    best: bool


def pair_points(opt, a, b):
    """Every scored formulation with a value for both `a` and `b`, fully
    measured or not; only a fully measured one carries its Overall score."""
    best = opt.best_index()
    out = []
    for i in range(len(opt.Y_history)):
        x, y = measured(opt, a, i), measured(opt, b, i)
        if x is None or y is None:
            continue
        full = opt.fully_measured(i)
        out.append(PairPoint(int(opt.formulation_ids[i]), x, y,
                             float(opt.Y_history[i]) if full else None, i == best))
    return out


# ------------------------------------------------------------------ #
#  Wave B: predictions for the open Round, spreads, predicted best
# ------------------------------------------------------------------ #

def round_predictions(opt, fitted, rows):
    """{Formulation number: {output name: Prediction}} for `rows` (open
    Round rows, each {'formulation', 'recipe'}), each predicted at its
    make-up at the Formulation total. Outputs with no model are left out."""
    if not rows:
        return {}
    X = [opt._encode(at_total(opt, r['recipe'])) for r in rows]
    per = {name: predict(opt, f, X) for name, f in fitted.items()
           if f.model is not None}
    return {int(r['formulation']): {name: preds[k] for name, preds in per.items()}
            for k, r in enumerate(rows)}


@dataclass
class Spread:
    name: str
    spread: float


@dataclass
class Spreads:
    moved: list     # [Spread], largest first
    held: list      # names the limits hold in place from this start


# A path over less than this share of its own Lowest-Highest is held in
# place by the limits: its flat line would read as "no effect", which the
# predictions never said.
HELD = 0.02


def has_ingredient_choice(opt):
    """Whether any ingredient is among the path choices."""
    return any(_is_ingredient(_var(opt, n)) for n in path_choices(opt))


def spreads(opt, fitted, start):
    """How far each path choice moves the predicted Overall score from
    `start`: highest minus lowest mean along its path, largest first (ties
    keep path_choices order). A choice whose path the limits box in (under
    HELD of its Lowest-Highest) is named in `held` instead of given a
    spread. A path that cannot be drawn is left out; with no Overall-score
    model there is nothing to say."""
    score = fitted.get(SCORE)
    if score is None or score.model is None:
        return Spreads([], [])
    moved, held = [], []
    for name in path_choices(opt):
        path = build_path(opt, name, start)
        if path.start_breaks or not path.recipes:
            continue
        lo, hi = (float(b) for b in _var(opt, name)['bounds'])
        if path.amounts[-1] - path.amounts[0] < HELD * (hi - lo):
            held.append(name)
            continue
        means = [p.mean for p in predict(opt, score, [opt._encode(r) for r in path.recipes])]
        moved.append(Spread(name, max(means) - min(means)))
    return Spreads(sorted(moved, key=lambda s: -s.spread), held)


def best_refusal(opt):
    """Why no Round could be made now (the refusals `ask` raises before it
    searches), or None."""
    for c in opt.constraints:
        if c['min'] is None and c['max'] is None:
            continue
        gaps = opt.ingredients_without_property(c['metric'])
        if gaps:
            return wording.missing_property_values(c['metric'], number_list(gaps))
    missing = opt.premix_readiness_error()
    if missing:
        return missing
    if not opt.varying_variables():
        return wording.EVERYTHING_IS_FIXED
    return None


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


def nearest(opt, recipe):
    """(index, distance) of the nearest formulation among
    opt.best_candidates() (its score is quoted beside the predicted best),
    ties to the lowest Formulation number; (None, None) when none can be
    compared."""
    best = None
    for i in opt.best_candidates():
        d = distance(opt, recipe, start_recipe(opt, i))
        if d is None:
            continue
        key = (d, int(opt.formulation_ids[i]))
        if best is None or key < best[0]:
            best = (key, i)
    return (None, None) if best is None else (best[1], best[0][0])


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

    Ties go to the lowest Formulation number, as `nearest` does."""
    a = at_total(opt, recipe)
    own_round = opt.pending_batch_no
    best = None
    for i in opt.waiting_rows():
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


@dataclass
class PredictedBest:
    recipe: dict
    score: Prediction
    close: int | None       # index of the nearest formulation when close, else None
    compare: int            # index of the table's second row


def predicted_best(opt, fitted):
    """The single formulation the Overall-score model predicts highest,
    found the way suggestions are made and never written anywhere.

    Returns a PredictedBest, or a one-line reason (str) when `ask` would
    refuse to make a Round. Raises when the search fails or its answer
    breaks a limit; the caller shows one line."""
    reason = best_refusal(opt)
    if reason:
        return reason
    score = fitted[SCORE]
    bounds = opt._get_bounds()
    dim = bounds.shape[1]
    with opt._premixes_for_generation():
        with _isolated():
            candidate, _ = optimize_acqf(
                acq_function=PosteriorMean(score.model),
                bounds=torch.stack([torch.zeros(dim), torch.ones(dim)]).double(),
                q=1, num_restarts=10, raw_samples=512,
                inequality_constraints=opt._get_botorch_constraints(),
                fixed_features=opt._get_fixed_features() or None,
            )
        recipe = opt._decode(unnormalize(candidate[0], bounds).detach().numpy().flatten())
        if opt.has_formulation_total():
            recipe = opt._snapped_if_it_still_fits(recipe)
        recipe = opt._settings_on_their_step(recipe)
        # A categorical row holds a word, which the limit check would read
        # as an amount; no limit can name one.
        if not opt._check_constraints({k: v for k, v in recipe.items()
                                       if not isinstance(v, str)}):
            raise ValueError("the predicted best breaks a limit")
    [prediction] = predict(opt, score, [opt._encode(recipe)])
    i, d = nearest(opt, recipe)
    close = i if i is not None and d <= CLOSE_TO_THE_BEST + 1e-9 else None
    return PredictedBest(recipe, prediction, close,
                         close if close is not None else opt.best_index())


# ------------------------------------------------------------------ #
#  Wave C: More charts — three ingredients trading places
# ------------------------------------------------------------------ #

def mixture_choices(opt):
    """The rows "What happens when three ingredients trade places?" offers,
    in project order: the varying ingredients and, when there is one, the
    Fill to total row. Without a Fill to total row an ingredient that a
    calculation reads is left out: moving it would move the calculation and
    take the formulation off its total. [] without a Formulation total."""
    if not opt.has_formulation_total():
        return []
    fill = opt.balance_row_name()
    read = set()
    if fill is None:
        for form in opt._resolved_forms().values():
            read |= form.names()
    varying = {v['name'] for v in opt.varying_variables()
               if _is_ingredient(v) and v.get('type') == 'continuous'}
    return [v['name'] for v in opt.variables
            if v['name'] == fill or (v['name'] in varying and v['name'] not in read)]


GRID_STEPS = 30          # barycentric step 1/30: 496 points over the whole triangle
# The grid covers only the amounts each of the three can take (0.8.0 review
# fixes §5.1): a region that is a fraction f of the triangle is sampled
# ceil(1/sqrt(f)) times finer, up to this many times, so it keeps about as
# many points as the whole triangle had.
MAX_REFINE = 10
# Why a Mixture has no chart (Mixture.refusal); the screen words each one.
NO_ROOM = "no_room"            # the three already sit at their Lowest or Highest
FILL_FOLLOWS = "fill_follows"  # the Fill to total row is one of the three and a
                               # calculation reads another of them


def _simplex(steps=GRID_STEPS):
    return [(i / steps, j / steps, (steps - i - j) / steps)
            for i in range(steps + 1) for j in range(steps + 1 - i)]


@dataclass
class MixturePoint:
    x: tuple               # barycentric (x1, x2, x3), summing to 1
    amounts: tuple         # the three amounts
    recipe: dict | None    # None when the point breaks a limit
    broken: str | None     # the limit it breaks, as Set up words it
    tested: bool           # within CLOSE_TO_THE_BEST of a dot


@dataclass
class MixtureDot:
    formulation: int
    x: tuple               # barycentric, after projecting onto the slice
    amounts: tuple


@dataclass
class Mixture:
    names: list
    units: list
    start: dict
    total: float           # S: the three amounts' sum in the start
    lower: list            # L*: implied lowest of each
    upper: list            # U*: implied highest of each
    span: float            # R = S - sum(L*)
    points: list
    dots: list
    refusal: str | None    # NO_ROOM / FILL_FOLLOWS: one line instead of a chart
    start_breaks: str | None
    follows: list          # calculation rows that move with the three
    fill_moves: str | None # the Fill to total row, when it takes up a change
    skipped: list          # property limits that cannot be checked
    top: list | None = None  # each one's largest amount inside the drawn region
    steps: int = GRID_STEPS  # the grid's barycentric steps, after refining


def _numeric(recipe):
    """A make-up without its categorical rows: a choice ("Tray") is not an
    amount, and _limits_broken adds every value up for its tolerance."""
    return {k: v for k, v in recipe.items() if not isinstance(v, str)}


def _own_bounds(opt, name, start):
    """A chosen row's Lowest-Highest widened to its start amount; the Fill
    to total row's are 0 to the total (its own two cells are dormant)."""
    if name == opt.balance_row_name():
        return 0.0, float(opt.formulation_total)
    return _widened(_var(opt, name), start)


def _near_other_rows(opt, names, start, rec):
    """Whether `rec` matches `start` on every choice (a categorical row) and
    sits within CLOSE_TO_THE_BEST of it, as a share of each span, on every
    other varying row (process settings included)."""
    for v in opt.varying_variables():
        if v['name'] in names:
            continue
        if v.get('type') != 'continuous':
            if rec.get(v['name']) != start.get(v['name']):
                return False
            continue
        lo, hi = (float(t) for t in v['bounds'])
        d = abs(float(rec.get(v['name'], 0.0) or 0.0) - float(start.get(v['name'], 0.0) or 0.0))
        if d / (hi - lo) > CLOSE_TO_THE_BEST + _EPS:
            return False
    return True


def _mixture_dots(opt, names, start, S, low, R, cap=(1.0, 1.0, 1.0)):
    """The fully measured formulations close to the slice, each projected
    onto it (its three amounts scaled to sum to S) and kept only when that
    lands inside the drawn region (x_k <= cap_k)."""
    dots = []
    for i in range(len(opt.Y_history)):
        if not opt.fully_measured(i):
            continue
        rec = start_recipe(opt, i)
        if not _near_other_rows(opt, names, start, rec):
            continue
        amounts = [float(rec.get(n, 0.0) or 0.0) for n in names]
        s2 = sum(amounts)
        if s2 <= 0 or abs(s2 - S) / R > CLOSE_TO_THE_BEST + _EPS:
            continue
        a = [v * S / s2 for v in amounts]
        x = tuple((a[k] - low[k]) / R for k in range(3))
        if min(x) < -1e-6 or any(x[k] > cap[k] + 1e-6 for k in range(3)):
            continue
        dots.append(MixtureDot(int(opt.formulation_ids[i]), x, tuple(a)))
    return dots


def mixture_slice(opt, names, start, steps=GRID_STEPS):
    """The three amounts in `names` trading places with their sum kept at
    their sum in `start`; every other row stays as in `start`, and the
    calculations (and a Fill to total row not among the three) are worked
    out again. Each grid point is checked limit by limit, like a path."""
    names = list(names)
    fill = opt.balance_row_name()
    units = [opt.unit_of(n) for n in names]
    start = dict(start)
    S = sum(float(start.get(n, 0.0) or 0.0) for n in names)
    bounds = [_own_bounds(opt, n, start) for n in names]
    low = [max(bounds[k][0], S - sum(bounds[j][1] for j in range(3) if j != k)) for k in range(3)]
    up = [min(bounds[k][1], S - sum(bounds[j][0] for j in range(3) if j != k)) for k in range(3)]
    R = S - sum(low)
    moving = [n for n in names if n != fill]
    follows = [n for n, f in opt._resolved_forms().items()
               if n != fill and set(names) & f.names()]
    skipped = [(opt.property_limit_text(c), c['metric']) for c in opt.constraints
               if (c.get('min') is not None or c.get('max') is not None)
               and opt.ingredients_without_property(c['metric'])]
    base = dict(names=names, units=units, start=start, total=S, lower=low, upper=up,
                span=R, points=[], dots=[], refusal=None, start_breaks=None,
                follows=follows, fill_moves=None, skipped=skipped)
    if fill in names and follows:
        return Mixture(**dict(base, refusal=FILL_FOLLOWS))
    if R <= 1e-9 * max(1.0, abs(S)):
        return Mixture(**dict(base, refusal=NO_ROOM))
    breaks = _limits_broken(opt, _numeric(opt.fill_formulas(dict(start))), moving, start)
    if breaks:
        return Mixture(**dict(base, start_breaks=breaks))
    # The drawn region: inside the triangle, each at most its implied
    # Highest (U*). The L-pseudo corner L*+R can lie past it.
    cap = [min(1.0, (up[k] - low[k]) / R) for k in range(3)]
    top = [low[k] + cap[k] * R for k in range(3)]
    inside = lambda x: all(x[k] <= cap[k] + 1e-9 for k in range(3))
    base_grid = _simplex(steps)
    share = sum(inside(x) for x in base_grid) / len(base_grid)
    refine = min(MAX_REFINE, max(1, math.ceil(1 / math.sqrt(max(share, 1e-12)))))
    steps = steps * refine
    dots = _mixture_dots(opt, names, start, S, low, R, cap)
    spans = [bounds[k][1] - bounds[k][0] for k in range(3)]
    points = []
    for x in (x for x in _simplex(steps) if inside(x)):
        a = tuple(low[k] + x[k] * R for k in range(3))
        r = dict(start)
        for n, v in zip(names, a):
            if n != fill:
                r[n] = v
        r = opt.fill_formulas(r)
        broken = _limits_broken(opt, _numeric(r), moving, start)
        tested = any(max(abs(a[k] - d.amounts[k]) / spans[k] for k in range(3))
                     <= CLOSE_TO_THE_BEST + _EPS for d in dots)
        points.append(MixturePoint(x, a, None if broken else r, broken, tested))
    return Mixture(**dict(base, points=points, dots=dots, top=top, steps=steps,
                          fill_moves=fill if (follows and fill and fill not in names) else None))


def allowed_share(opt, names, start, steps=10):
    """How much of the three's trade space is allowed from `start`: the
    share of a coarse grid over the whole triangle that keeps every limit,
    each one's own Lowest-Highest included. 0 when they cannot trade places
    (no room, the fill follows a calculation, or the start breaks a limit).
    Picks the default three (0.8.0 review fixes §5.2)."""
    mix = mixture_slice(opt, names, start, steps=1)
    if mix.refusal or mix.start_breaks:
        return 0.0
    fill = opt.balance_row_name()
    moving = [n for n in names if n != fill]
    ok = 0
    grid = _simplex(steps)
    for x in grid:
        r = dict(mix.start)
        for n, k in zip(names, range(3)):
            if n != fill:
                r[n] = mix.lower[k] + x[k] * mix.span
        if _limits_broken(opt, _numeric(opt.fill_formulas(r)), moving, mix.start) is None:
            ok += 1
    return ok / len(grid)


def mixture_predictions(opt, fitted, mix):
    """The predicted Overall score at each allowed point of `mix`, None at a
    point that breaks a limit; [] without an Overall-score model."""
    score = fitted.get(SCORE)
    if score is None or score.model is None:
        return []
    allowed = [p.recipe for p in mix.points if p.recipe is not None]
    preds = iter(predict(opt, score, [opt._encode(r) for r in allowed]))
    return [next(preds) if p.recipe is not None else None for p in mix.points]
