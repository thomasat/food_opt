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
