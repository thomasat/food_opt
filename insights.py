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
from botorch.fit import fit_gpytorch_mll
from botorch.utils.transforms import normalize
from gpytorch.mlls import ExactMarginalLogLikelihood

import wording

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


@dataclass
class Prediction:
    mean: float
    low: float
    high: float


def outputs(opt):
    out = [(o['name'], float(o['min_val']), float(o['max_val']))
           for o in opt.objectives]
    out.append((SCORE, 0.0, float(opt.utility_ceiling())))
    return out


def measured(opt, name, i):
    if name == SCORE:
        return float(opt.Y_history[i])
    value = (opt.results_history[i] if i < len(opt.results_history) else {}).get(name)
    return None if value is None else float(value)


def _range_of(opt, name):
    for n, lo, hi in outputs(opt):
        if n == name:
            return lo, hi
    raise KeyError(name)


def fit(opt, name, rows):
    lo, hi = _range_of(opt, name)
    use = [i for i in rows if measured(opt, name, i) is not None]
    if len(use) < MIN_ROWS:
        return Fitted(name, None, None, len(use), lo, hi)
    bounds = opt._get_bounds()
    try:
        with _isolated():
            X = normalize(torch.tensor([opt.X_history[i] for i in use],
                                       dtype=torch.double), bounds)
            Y = torch.tensor([measured(opt, name, i) for i in use],
                             dtype=torch.double).unsqueeze(-1)
            gp = opt._build_gp(X, Y, bounds.shape[1], None)
            fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
    except Exception as exc:   # one model failing must not take the tab down
        return Fitted(name, None, str(exc) or type(exc).__name__, len(use), lo, hi)
    return Fitted(name, gp, None, len(use), lo, hi)


def predict(opt, fitted, X_rows):
    if fitted.model is None or not X_rows:
        return []
    bounds = opt._get_bounds()
    with torch.no_grad(), _isolated():
        X = normalize(torch.tensor(X_rows, dtype=torch.double), bounds)
        post = fitted.model.posterior(X, observation_noise=True)
        mean = post.mean.reshape(-1).tolist()
        sd = post.variance.clamp_min(0.0).sqrt().reshape(-1).tolist()
    out = []
    for m, s in zip(mean, sd):
        m = min(fitted.high, max(fitted.low, m))
        out.append(Prediction(m, max(fitted.low, m - Z95 * s),
                              min(fitted.high, m + Z95 * s)))
    return out


def fit_all(opt):
    rows = list(range(len(opt.Y_history)))
    return {name: fit(opt, name, rows) for name, _, _ in outputs(opt)}


def data_key(opt):
    """Everything a fitted model or a Round-ahead check depends on."""
    payload = {
        "X": opt.X_history, "Y": opt.Y_history, "R": opt.results_history,
        "ids": opt.formulation_ids, "rounds": opt.batch_history,
        "objectives": opt.objectives, "cfg": getattr(opt, "bo_config", None),
        "bounds": opt._get_bounds().tolist(),
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


def check_rounds(opt):
    checks = []
    names = [n for n, _, _ in outputs(opt)]
    for r, first in round_order(opt):
        train = list(range(first))
        targets = [i for i, b in enumerate(opt.batch_history) if b == r]
        for name in names:
            fitted = fit(opt, name, train)
            if fitted.model is None:
                continue
            rows = [i for i in targets if measured(opt, name, i) is not None]
            preds = predict(opt, fitted, [opt.X_history[i] for i in rows])
            for i, p in zip(rows, preds):
                value = measured(opt, name, i)
                checks.append(Check(
                    int(opt.formulation_ids[i]), r, name, p.mean, p.low, p.high,
                    value, p.low <= value <= p.high, opt.fully_measured(i)))
    return checks


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
        if len(mine) < 5:
            continue
        meas = [c.measured for c in mine]
        mean = sum(meas) / len(meas)
        ss_tot = sum((m - mean) ** 2 for m in meas)
        ss_res = sum((c.measured - c.predicted) ** 2 for c in mine)
        rows.append(DetailRow(
            name, len(mine), sum(c.inside for c in mine),
            statistics.median((c.high - c.low) / 2 for c in mine),
            statistics.median(abs(c.measured - c.predicted) for c in mine),
            None if ss_tot == 0 else 1 - ss_res / ss_tot))
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


def start_recipe(opt, i):
    recipe = dict(opt._recorded_recipe(i))
    if opt.has_formulation_total():
        total = float(opt.formulation_total)
        own = opt.ingredient_total(recipe)
        tolerance = total * opt.FORMULATION_TOTAL_TOLERANCE
        if own > 0 and abs(own - total) > tolerance:
            return opt.scaled_recipe(recipe, scale_to=total)
    return opt.fill_formulas(recipe)


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
    skipped = [opt.property_limit_text(c) for c in opt.constraints
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
