"""What the Insights tab shows, worked out from a project without changing it.

Nothing here writes to the optimizer. Every model is fitted for display only,
with the optimizer's own builder so an expert's model settings apply, inside
a forked torch RNG so the next suggestion is exactly what it would have been.
The suggestions themselves still come from the one Overall-score model that
FoodOptimizer._ask_optimize fits; the models here never reach it."""
import contextlib
import hashlib
import json
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
