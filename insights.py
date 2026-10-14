"""What the Insights tab shows, worked out from a project without changing it.

Nothing here writes to the optimizer. Every model is fitted for display only,
with the optimizer's own builder so an expert's model settings apply, inside
a forked torch RNG so the next suggestion is exactly what it would have been.
The suggestions themselves still come from the one Overall-score model that
FoodOptimizer._ask_optimize fits; the models here never reach it."""
import contextlib
import contextvars
import dataclasses
import functools
import hashlib
import itertools
import json
import math
import statistics
import threading
import types
from dataclasses import dataclass

import torch
from botorch.acquisition import PosteriorMean
from botorch.exceptions.errors import InfeasibilityError
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.special import ndtri, stdtrit

import wording
from food_bo import CLOSE_TO_THE_BEST, FoodOptimizer, number_list

SCORE = "__overall_score__"
MIN_ROWS = 5
Z95 = 1.959963984540054
_SEED = 20260928


# The owner's hard rule across browser tabs (review B2; Plan A review B
# I-3): Streamlit runs each session's script in a thread of one process, and
# torch has one global RNG. Every in-process Insights step that touches it
# goes through _isolated() — fits, conditioning, predictions, the predicted
# best — and holds this lock for its whole fork-and-reseed; the one ask()
# call site holds it too (ui_batch.ask_alone). So no Insights step can
# restore the RNG in the middle of a suggestion. Re-entrant: an _isolated()
# inside another is fine. Worker processes have their own.
RNG_LOCK = threading.RLock()


@contextlib.contextmanager
def _isolated():
    """Repeatable, and invisible to everything outside: the global RNG is
    restored on the way out, and nothing else in this process uses it in
    between (RNG_LOCK)."""
    with RNG_LOCK, torch.random.fork_rng():
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


# 0.8.0 speed (spec 2026-09-30 §10.4): the scipy options of every display
# fit here, never the suggestion model's. scipy's default ftol is 2.2e-9;
# 1e-6 takes 2.8-3.3x fewer steps on a 10-row project with Round-ahead
# coverage 90.8 / 88.8 % against 90.9 / 88.6 % (perf.md §2.2).
FIT_OPTIONS = {"ftol": 1e-6}
# Part of every model_key: bump it whenever a change here could change a
# fitted model's numbers, and a saved fit from before is never used again.
# That includes a torch, gpytorch or botorch upgrade (review A M-6).
MODEL_SALT = "0.8.0-ftue-fa-folds-1"
# The fitted models a caller keeps (insights_jobs, the saved cache): model_key
# -> {"state": hyper_state(gp), "widen": float} or {"error": str}. Set with
# memo_in_use(); None: every fit is made fresh, as before.
_MEMO = contextvars.ContextVar("insights_memo", default=None)


@contextlib.contextmanager
def memo_in_use(memo):
    """Within the block, fit() takes a model from `memo` when its key is
    there (conditioning only, no optimising) and puts every model it fits
    into it. Per thread: another session's script run never sees it."""
    token = _MEMO.set(memo)
    try:
        yield memo
    finally:
        _MEMO.reset(token)


def builder(cfg):
    """FoodOptimizer._build_gp for a project whose expert model settings
    are `cfg` (None: the default model), without the project: what an
    insights_worker builds with. In this process fits use opt._build_gp,
    the same method."""
    return functools.partial(FoodOptimizer._build_gp, types.SimpleNamespace(bo_config=cfg))


def fit_xy(build, X, Y):
    """One display GP on normalised X (n x d) and Y (n x 1), made by
    `build` (opt._build_gp, or builder(cfg) in a worker, so an expert's
    model settings apply) and fitted with FIT_OPTIONS under the fixed seed.
    The one fit both this process and insights_worker make, so the two
    give the same numbers. Raises if the fit fails."""
    with _isolated():
        gp = build(X, Y, X.shape[-1], None)
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp),
                         optimizer_kwargs={"options": dict(FIT_OPTIONS)})
    return gp


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


def _fit_gp(opt, name, rows):
    """One GP on `rows` (all measured). Raises if the fit fails."""
    X, Y = xy(opt, name, rows)
    return fit_xy(opt._build_gp, X, Y)


def hyper_state(gp):
    """A fitted GP's parameters and buffers as plain numbers:
    {name: [dtype, nested list]}. JSON data, never a pickled object."""
    return {k: [str(v.dtype).removeprefix("torch."), v.detach().tolist()]
            for k, v in gp.state_dict().items()}


def rebuild(opt, name, rows, state):
    """The GP fit() made on `rows`, from its hyper_state: built on the same
    rows by the same builder and conditioned, with nothing optimised."""
    X, Y = xy(opt, name, rows)
    with _isolated():
        gp = opt._build_gp(X, Y, X.shape[-1], None)
        gp.load_state_dict({k: torch.tensor(v, dtype=getattr(torch, d))
                            for k, (d, v) in state.items()})
        gp.eval()
    return gp


def model_key(opt, name, use):
    """Everything one display model and its widening depend on: the model
    settings, the output, each training row (make-up and results), the
    folds (their number and each row's Round, which groups them: ftue fix
    A, cold read I9), the fit options and MODEL_SALT."""
    return _digest({"salt": MODEL_SALT, "settings": _digest(_model_settings(opt)),
                    "name": name, "rows": [_row_key(opt, i) for i in use],
                    "folds": FOLDS, "rounds": [_round_of(opt, i) for i in use],
                    "fit": FIT_OPTIONS})


def fit(opt, name, rows):
    lo, hi = _range_of(opt, name)
    use = [i for i in rows if measured(opt, name, i) is not None]
    if len(use) < MIN_ROWS:
        return Fitted(name, None, None, len(use), lo, hi)
    memo = _MEMO.get()
    key = None if memo is None else model_key(opt, name, use)
    hit = None if memo is None else memo.get(key)
    if hit is not None:
        if hit.get("error") is not None:
            return Fitted(name, None, hit["error"], len(use), lo, hi)
        try:
            return Fitted(name, rebuild(opt, name, use, hit["state"]), None, len(use),
                          lo, hi, float(hit["widen"]))
        except Exception:   # a damaged entry: fitted again below
            pass
    try:
        gp = _fit_gp(opt, name, use)
    except Exception as exc:   # one model failing must not take the tab down
        error = str(exc) or type(exc).__name__
        if memo is not None:
            memo[key] = {"error": error}
        return Fitted(name, None, error, len(use), lo, hi)
    widen = _widen(opt, name, use)
    if memo is not None:
        memo[key] = {"state": hyper_state(gp), "widen": widen}
    return Fitted(name, gp, None, len(use), lo, hi, widen)


FOLDS = 5


def _round_of(opt, i):
    return opt.batch_history[i] if i < len(opt.batch_history) else None


def fold_units(opt, rows):
    """One label per position of `rows`: the rows that must be held out
    together (ftue fix A, cold read I9). A Round's formulations sit close
    together, and a Repeat is a near copy, so a row held out with its Round
    or its Repeat left in training is predicted too well and the widening
    comes out too small. Each Round is one unit (a row with no Round is its
    own); when every row is in one Round (the cold start) each row is its
    own unit. Then every Repeat group (repeat_groups) joins the units of
    its rows into one."""
    labels = [("round", r) if r is not None else ("row", i)
              for i, r in ((i, _round_of(opt, i)) for i in rows)]
    if len(set(labels)) < 2:
        labels = [("row", i) for i in rows]
    parent = list(range(len(rows)))

    def root(p):
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p

    def join(a, b):
        parent[root(b)] = root(a)

    first = {}
    for p, label in enumerate(labels):
        if label in first:
            join(first[label], p)
        else:
            first[label] = p
    where = {i: p for p, i in enumerate(rows)}
    for group in repeat_groups(opt, list(rows)):
        for i in group[1:]:
            join(where[group[0]], where[i])
    return [root(p) for p in range(len(rows))]


def folds(units):
    """[(kept, held)] positions for cross-validation, from fold_units'
    labels (an int n: n rows, each its own unit): the units, in the order
    they first appear, are dealt to FOLDS folds in turn, so a
    Round, and a Repeat group, is always held out whole (ftue fix A, cold
    read I9; before, row k went to fold k % FOLDS). A fold with nothing
    held or nothing kept is left out."""
    if isinstance(units, int):
        units = list(range(units))
    fold = {u: j % FOLDS for j, u in enumerate(dict.fromkeys(units))}
    out = []
    for f in range(FOLDS):
        held = [p for p, u in enumerate(units) if fold[u] == f]
        kept = [p for p, u in enumerate(units) if fold[u] != f]
        if held and kept:
            out.append((kept, held))
    return out


def cv_folds(opt, rows):
    """folds(fold_units(opt, rows)): the folds the widening of a model
    fitted on `rows` is worked out from, here and in insights_jobs."""
    return folds(fold_units(opt, list(rows)))


def held_z(gp, Xt, Yt):
    """The standardised misses (y - mean) / sd of held-out rows (Xt, Yt)
    under `gp`, with the variation between Repeats included."""
    with torch.no_grad(), _isolated():
        post = gp.posterior(Xt, observation_noise=True)
        mean = post.mean.reshape(-1).tolist()
        sd = post.variance.clamp_min(0.0).sqrt().reshape(-1).tolist()
    return [(y - m) / s for y, m, s in zip(Yt.reshape(-1).tolist(), mean, sd) if s > 0]


def fold_z(build, X, Y, Xt, Yt):
    """held_z of one fold under a model fitted (hyperparameters too) on
    (X, Y): what an insights_worker answers. Raises if the fit fails."""
    return held_z(fit_xy(build, X, Y), Xt, Yt)


def _cv_z(opt, name, use):
    """Standardised misses of each row when it was held out: rows go to
    FOLDS folds a Round, and a Repeat group, at a time (cv_folds), and each
    fold is predicted by a model fitted on the others. A fold whose fit
    fails is skipped. Only `use` is ever seen, so a Round-ahead check stays
    honest."""
    z = []
    for kept, held in cv_folds(opt, use):
        try:
            gp = _fit_gp(opt, name, [use[p] for p in kept])
            z.extend(held_z(gp, *xy(opt, name, [use[p] for p in held])))
        except Exception:
            continue
    return z


def widen_from(z):
    """The root-mean-square of cross-validated misses, never below 1."""
    c = math.sqrt(sum(v * v for v in z) / len(z)) if z else 1.0
    return max(1.0, c) if math.isfinite(c) else 1.0


def _widen(opt, name, use):
    """How much to widen this model's likely range: widen_from its
    cross-validated misses. With few rows the fitted noise comes out too
    small and the range too narrow (Round-ahead coverage near 78 % at
    realistic noise); refitting on each fold sees that, where the same-rows
    leave-one-out did not. Only widens; with no usable fold the range stays
    as fitted."""
    try:
        return widen_from(_cv_z(opt, name, use))
    except Exception:
        return 1.0


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
    raw_mean: float | None = None   # Prediction.raw_mean / .sd: the verdict's |z|
    sd: float | None = None
    # How much the likely range [low, high] was widened on top of sd (the
    # Overall score's before-the-fact widening, spec 2026-09-30 §9 I1); a
    # miss in standard deviations is still (measured - raw_mean) / sd.
    widen: float = 1.0
    # The Round-ahead model's fitted noise as an sd in the output's units
    # (fitted_noise_sd); None when the noise is not fitted (fixed_tiny).
    # Ftue fix A, cold read I7: the Repeat check's σ.
    noise: float | None = None
    # Sceptic review (fix round 1): a 95 % range covering the output's whole
    # Range (each end within NO_FORECAST_EDGE of the Range's ends) is no
    # forecast: never counted inside or outside (ranged). Set by check_rounds.
    no_forecast: bool = False


# Sceptic review: "whole Range" — each end within 1 % of the Range's width.
NO_FORECAST_EDGE = 0.01


def covers_range(opt, name, low, high):
    """Whether a 95 % range [low, high] of output `name` covers its whole
    Range: low ≤ Range low + NO_FORECAST_EDGE × width and high ≥ Range high
    − NO_FORECAST_EDGE × width. False for a name with no Range."""
    try:
        lo, hi = _range_of(opt, name)
    except KeyError:
        return False
    edge = NO_FORECAST_EDGE * (float(hi) - float(lo))
    return float(low) <= float(lo) + edge and float(high) >= float(hi) - edge


def mark_no_forecast(opt, checks):
    """`checks` with no_forecast set from each one's final 95 % range."""
    return [dataclasses.replace(c, no_forecast=covers_range(opt, c.name, c.low, c.high))
            for c in checks]


def ranged(checks):
    """The checks that forecast something: every coverage count (inside or
    outside a 95 % range, and its allowed misses) is over these only."""
    return [c for c in checks or () if not getattr(c, "no_forecast", False)]


def _row_key(opt, i):
    """One recorded row as a Round-ahead check sees it: its make-up as the
    models learn it (model_X, review I11), its Overall score, its results
    and its Formulation number (which decides the measurements that apply
    to it)."""
    at = lambda seq: seq[i] if i < len(seq) else None
    return _digest([opt.model_X([i])[0], opt.Y_history[i], at(opt.results_history),
                    at(opt.formulation_ids), opt.is_set_aside(i)])


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
                        # The training rows' Rounds group the folds (I9).
                        "train_rounds": [_round_of(opt, i) for i in range(first)],
                        "targets": [(i, rows[i]) for i, b in enumerate(opt.batch_history)
                                    if b == r]})
            for r, first in round_order(opt)}


def _check_round(opt, r, first, names):
    checks = []
    train = list(range(first))
    targets = [i for i, b in enumerate(opt.batch_history) if b == r]
    tiny = (getattr(opt, "bo_config", None) or {}).get("noise") == "fixed_tiny"
    for name in names:
        fitted = fit(opt, name, train)
        if fitted.model is None:
            continue
        try:
            noise = None if tiny else fitted_noise_sd(fitted.model)
        except Exception:
            noise = None
        rows = [i for i in targets if measured(opt, name, i) is not None]
        try:
            # At the make-ups the models learn (model_X): a Round made at
            # another batch size is predicted where it was made.
            preds = predict(opt, fitted, opt.model_X(rows))
        except Exception:   # like a failed fit: no checks for this pair
            continue
        for i, p in zip(rows, preds):
            value = measured(opt, name, i)
            checks.append(Check(
                int(opt.formulation_ids[i]), r, name, p.mean, p.low, p.high,
                value, p.low <= value <= p.high, opt.fully_measured(i),
                p.raw_mean, p.sd, noise=noise))
    return checks


def check_rounds(opt, cache=None, checking=None, only=None):
    """Each Round's formulations predicted from the rows recorded before
    it, Round by Round, the Overall score's likely range then widened by
    nest_score_widen (spec 2026-09-30 §9 I1).

    `cache`, a dict the caller keeps between calls, holds each Round's
    checks under its round_keys key, before that widening: a Round whose key
    is there is not fitted again, and on the way out the dict holds this
    project's Rounds alone. Every fit runs in its own `_isolated()` with the
    same seed, so a Round's checks come out the same whichever other Rounds
    are fitted with it: kept or fresh, the answer is identical.

    `checking(r, last)`, when given, is entered around each Round actually
    fitted (one that could be checked), for the screen's progress line.

    `only`, a set of (Round, output name) pairs, works out those checks
    alone, for a screen filling in while the rest are still being fitted; a
    partial answer is never written to `cache`. It is taken with every
    pair their widening reads (widen_pairs), so a check comes out the same
    whether it is worked out alone or with every other: one 95 % range
    per forecast, whatever is cached (2026-10-07)."""
    names = [n for n, _, _ in outputs(opt)]
    order = round_order(opt)
    if only is not None:
        only = widen_pairs(opt, only)
    keys = round_keys(opt) if cache is not None else {}
    checks = []
    for r, first in order:
        wanted = names if only is None else [n for n in names if (r, n) in only]
        if not wanted:
            continue
        key = keys.get(r)
        if key is not None and key in cache:
            checks.extend(c for c in cache[key] if c.name in wanted)
            continue
        line = (checking(r, order[-1][0]) if checking is not None and first >= MIN_ROWS
                else contextlib.nullcontext())
        with line:
            found = _check_round(opt, r, first, wanted)
        if cache is not None and only is None:
            cache[key] = found
        checks.extend(found)
    if cache is not None and only is None:
        for gone in set(cache) - set(keys.values()):
            del cache[gone]
    return mark_no_forecast(opt, nest_score_widen(opt, nest_measure_widen(opt, checks)))


def widen_pairs(opt, pairs):
    """`pairs`, (Round, output name) pairs of Round-ahead checks, with
    every pair their widening reads: a check in Round r is widened by the
    same output's checks in every Round before r (nest_measure_widen,
    nest_score_widen), so those are needed too. Worked out from the Rounds
    alone; a pair whose Round has no check is harmless (nothing is fitted
    for it)."""
    rounds = [r for r, _ in round_order(opt)]
    out = set(pairs)
    for r, name in pairs:
        if r in rounds:
            out.update((q, name) for q in rounds[:rounds.index(r)])
    return out


def nest_measure_widen(opt, checks):
    """Round-2 fixes (P1 #1): each measurement's checks widened by
    measure_widen of that measurement's checks in the Rounds before it, as
    nest_score_widen does for the Overall score. Make a round widens its
    Predicted measurements by measure_widen of every Round but the open
    one, so the 95 % range on Make a round, the Forecast check and Results'
    In / Above / Below is one range for one forecast. Widened about the
    model's own mean, clipped to the measurement's Range; `inside` follows."""
    ranges = {n: (lo, hi) for n, lo, hi in outputs(opt) if n != SCORE}
    factor = {}
    for name in ranges:
        mine = [c for c in checks if c.name == name]
        earlier = []
        for r in _ordered_rounds(mine):
            factor[(name, r)] = measure_widen(earlier, name)
            earlier.extend(c for c in mine if c.round == r)
    out = []
    for c in checks:
        f = factor.get((c.name, c.round), 1.0)
        if f == 1.0 or c.sd is None or c.raw_mean is None:
            out.append(c)
            continue
        lo_r, hi_r = ranges[c.name]
        half = Z95 * c.sd * f
        low = min(hi_r, max(lo_r, c.raw_mean - half))
        high = min(hi_r, max(lo_r, c.raw_mean + half))
        out.append(dataclasses.replace(c, low=low, high=high,
                                       inside=low <= c.measured <= high, widen=f))
    return out


def nest_score_widen(opt, checks):
    """`checks` with each Round's Overall-score checks widened by
    score_widen of the checks of the Rounds before it, and nothing else
    changed (spec 2026-09-30 §9 I1). Make a round widens its Overall-score
    column by score_widen of every check so far, so the check that later
    judges that Round counts the range that was on screen, and stays
    before the fact: no Round's widening has seen that Round's results.
    The range is widened about the model's own mean, then clipped to 0 and
    the ceiling; `inside` follows the widened range."""
    top = float(opt.utility_ceiling())
    rounds = _ordered_rounds([c for c in checks if c.name == SCORE])
    out, earlier = [], []
    factor = {}
    for r in rounds:
        factor[r] = score_widen(earlier)
        earlier.extend(c for c in checks if c.name == SCORE and c.round == r)
    for c in checks:
        f = factor.get(c.round, 1.0) if c.name == SCORE else 1.0
        if f == 1.0 or c.sd is None or c.raw_mean is None:
            out.append(c)
            continue
        half = Z95 * c.sd * f
        low = min(top, max(0.0, c.raw_mean - half))
        high = min(top, max(0.0, c.raw_mean + half))
        out.append(dataclasses.replace(c, low=low, high=high,
                                       inside=low <= c.measured <= high, widen=f))
    return out


def rounds_not_checked(opt):
    """The Rounds with fewer than MIN_ROWS formulations recorded before
    them: nothing earlier to predict them from, so they are never checked."""
    return [r for r, first in round_order(opt) if first < MIN_ROWS]


def binomial_tail(m, n, p=0.05):
    if m <= 0:
        return 1.0
    return sum(math.comb(n, k) * p ** k * (1 - p) ** (n - k)
               for k in range(m, n + 1))


@dataclass
class ScoreRecord:
    """How one output's own predictions did, fully measured formulations
    only (0.8.0 review fixes §3.3, §9 I7). The headline's is the Overall
    score's over every Round checked (0.8.0 e2e fixes, review I1)."""
    inside: int
    total: int
    mean_miss: float        # measured minus predicted, on average: < 0 is below
    gap_sd: float | None = None   # the spread of those gaps (sample sd); None for one
    rounds: list = None     # the Rounds it spans, in order
    # The standard error of mean_miss with the Round as the unit from
    # ROUND_UNIT_MIN Rounds (review B1); None: gap_sd / √total is used.
    se: float | None = None
    # se's degrees of freedom (ftue fix A, cold read I5): Rounds − 1 when
    # each Round's mean error is one value, else values − 1. None: worked
    # out from rounds and total (bias_df).
    df: int | None = None


SCORE_RECORD_MIN = 5
# From this many Rounds checked, a steady bias is judged with each Round's
# mean error as one value (review B1): a Round's formulations sit close
# together and share the model's error, so their errors are not independent.
ROUND_UNIT_MIN = 4
# A likely range whose typical half-width is over this share of the
# measurement's Range says little when results land inside it (§3.2, §9 I8).
WIDE = 0.25


@dataclass
class Headline:
    state: str
    rounds: list
    inside: int
    total: int
    verdict: str
    score: ScoreRecord | None = None
    wide: list = None       # outputs (SCORE included) whose likely range is wide


def _ordered_rounds(checks):
    out = []
    for c in checks:
        if c.round not in out:
            out.append(c.round)
    return out


def headline(opt, checks):
    # Sceptic review (fix round 1): no-forecast checks are counted neither
    # inside nor outside (ranged); the window is still the Rounds checked,
    # and the wide-range rule still sees them.
    rounds = _ordered_rounds([c for c in checks if c.name != SCORE])
    values = [c for c in ranged(checks) if c.name != SCORE]
    score_rounds = _ordered_rounds([c for c in checks if c.name == SCORE])
    if not rounds:
        return Headline("none", [], 0, 0, None)
    k = min(3, len(rounds))
    chosen = rounds[-k:]
    while (sum(c.round in chosen for c in values) < 10 and k < len(rounds)):
        k += 1
        chosen = rounds[-k:]
    picked = [c for c in values if c.round in chosen]
    # The Overall score's own line spans every Round checked (0.8.0 e2e
    # fixes, review I1), not the verdict's window.
    record = score_record(checks, score_rounds)
    wide = wide_ranges(opt, [c for c in checks if c.round in chosen])
    if len(picked) < 10:
        return Headline("early", chosen, sum(c.inside for c in picked),
                        len(values), None, record, wide)
    inside = sum(c.inside for c in picked)
    tail = binomial_tail(len(picked) - inside, len(picked))
    verdict = ("holding" if tail >= 0.10 else
               "caution" if tail >= 0.01 else "unreliable")
    return Headline("verdict", chosen, inside, len(picked), verdict, record, wide)


def error_record(checks, name, rounds=None):
    """How `name`'s Round-ahead predictions did over `rounds` (every Round
    checked when None), fully measured formulations only; None with fewer
    than SCORE_RECORD_MIN. Its standard error takes each Round's mean error
    as one value from ROUND_UNIT_MIN Rounds, else each value (review B1)."""
    mine = [c for c in ranged(checks) if c.name == name and c.fully_measured
            and (rounds is None or c.round in rounds)]
    if len(mine) < SCORE_RECORD_MIN:
        return None
    gaps = [c.measured - c.predicted for c in mine]
    sd = statistics.stdev(gaps) if len(gaps) > 1 else None
    order = _ordered_rounds(mine)
    if len(order) >= ROUND_UNIT_MIN:
        means = [statistics.fmean(c.measured - c.predicted for c in mine if c.round == r)
                 for r in order]
        se = statistics.stdev(means) / math.sqrt(len(means))
        df = len(means) - 1
    else:
        se = None if sd is None else sd / math.sqrt(len(gaps))
        df = len(gaps) - 1
    return ScoreRecord(sum(c.inside for c in mine), len(mine), sum(gaps) / len(gaps), sd,
                       order, se, df)


def score_record(checks, rounds):
    """The Overall score's checks in `rounds` (every Round checked when
    None), fully measured formulations only; None with fewer than
    SCORE_RECORD_MIN. It replaced the 0.8.0 "most came in below" line,
    which needed 8 in 10 below and almost never showed while the score ran
    optimistic in 7 checks out of 10."""
    return error_record(checks, SCORE, rounds)


def t_critical(df, p=0.975):
    """Student's t quantile t_{p, df}: 3.18 at 3 df, 2.57 at 5, 1.96 in the
    limit (ftue fix A, cold read I5)."""
    return float(stdtrit(df, p))


def bias_df(record):
    """The degrees of freedom of `record`'s standard error (cold read I5):
    record.df, else Rounds − 1 from ROUND_UNIT_MIN Rounds (each Round's mean
    error one value), else values − 1."""
    df = getattr(record, "df", None)
    if df is not None:
        return df
    rounds = getattr(record, "rounds", None) or []
    return len(rounds) - 1 if len(rounds) >= ROUND_UNIT_MIN else record.total - 1


def bias_t(record):
    """(t_{0.975, df}, df): how many standard errors a steady bias must
    beat (ftue fix A, cold read I5), for the screen to say. None for a
    record with no degrees of freedom."""
    df = bias_df(record)
    return None if df is None or df < 1 else (t_critical(df), df)


def bias_gap(record, yardstick):
    """The average gap when it is a steady bias, else None (answer-key
    read-through: one rule for Insights and Make a round; review B1). It
    must be more than `yardstick` (the noise threshold: how much Repeats
    differ, 1 point without them for the Overall score; 0 for a measurement
    with no Repeat SD, so no floor) AND more than t_{0.975, df} standard
    errors (ftue fix A, cold read I5: 2 standard errors at 4 Rounds, 3 df,
    flagged a bias that is not there about 1 time in 7): the record's
    Round-unit one (record.se, df = Rounds − 1), else sd / sqrt(n) of the
    gaps (df = n − 1), so a few large misses, or one off Round, are not a
    bias."""
    if record is None or yardstick is None or abs(record.mean_miss) <= yardstick:
        return None
    se = getattr(record, "se", None)
    if se is None and record.gap_sd is not None:
        se = record.gap_sd / math.sqrt(record.total)
    t = bias_t(record)
    if se is not None and t is not None and abs(record.mean_miss) <= t[0] * se:
        return None
    return record.mean_miss


def score_bias(opt, checks, h=None, yard=None):
    """bias_gap for the Overall score's record over every Round checked."""
    h = headline(opt, checks) if h is None else h
    yard = noise_yardstick(opt, checks) if yard is None else yard
    return bias_gap(h.score, yard.value)


@dataclass
class Bias:
    """One output's steady bias (review B1): measured minus predicted on
    average over `rounds`; below 0, predictions ran high."""
    name: str
    mean: float
    rounds: list
    record: ScoreRecord = None


def _noise_floor(opt, name, matches=None, without=()):
    """The noise threshold of `name` for the bias rule (review I2): the
    Repeat SD from YARDSTICK_DF degrees of freedom (at least
    REPEAT_SD_SHOWN), else 1 point for the Overall score and no floor (0)
    for a measurement."""
    spread = repeat_sd(opt, name, matches, without)
    if (spread.sd is not None and spread.sd >= REPEAT_SD_SHOWN
            and spread.df >= YARDSTICK_DF):
        return spread.sd
    return assumed_noise(opt) if name == SCORE else 0.0


def steady_bias(opt, checks, flagged=None):
    """Every output whose predictions ran steadily high or low over every
    Round checked (review B1): bias_gap of its error_record against its
    noise floor, a disagreeing Repeat kept in the Repeat SD (ftue fix A,
    cold read I7). The Overall score first, then the measurements in
    Results' order. A fact for one line, never advice: a steady bias is the
    model's (its optimism at its own maximum, or its mean shrinking towards
    the average), not the bench's. `flagged`: accepted for callers, no
    longer used."""
    if not checks:
        return []
    # Ftue fix A (cold read I7): a disagreeing Repeat stays in the Repeat SD.
    without = ()
    out, matches = [], None
    for name in [SCORE] + [o['name'] for o in opt.measurements_by_importance()]:
        record = error_record(checks, name)
        if record is None:
            continue
        if matches is None:
            matches = _matches(opt, [i for i in range(len(opt.Y_history))
                                     if opt.fully_measured(i)])
        gap = bias_gap(record, _noise_floor(opt, name, matches, without))
        if gap is not None:
            out.append(Bias(name, gap, list(record.rounds), record))
    return out


def score_steady_bias(opt, checks, flagged=None):
    """The Overall score's steady bias alone (steady_bias's rule and
    inputs), or None: measured minus predicted, so above 0 the
    predictions ran low. Final review I-1: Progress's luck line reads
    against it."""
    if not checks:
        return None
    record = error_record(checks, SCORE)
    if record is None:
        return None
    # Ftue fix A (cold read I7): a disagreeing Repeat stays in the Repeat SD.
    matches = _matches(opt, [i for i in range(len(opt.Y_history))
                             if opt.fully_measured(i)])
    return bias_gap(record, _noise_floor(opt, SCORE, matches))


def wide_ranges(opt, checks):
    """The outputs among `checks` whose typical likely range is wide: a
    median half-width over WIDE of the output's Range (0 to the ceiling for
    the Overall score), in the order the tab lists them."""
    out = []
    for name, lo, hi in outputs(opt):
        mine = [c for c in checks if c.name == name]
        if mine and hi > lo and statistics.median(
                (c.high - c.low) / 2 for c in mine) > WIDE * (hi - lo):
            out.append(name)
    order = [o['name'] for o in opt.measurements_by_importance()] + [SCORE]
    return sorted(out, key=order.index)

@dataclass
class DetailRow:
    name: str
    checked: int
    inside: int
    half_width: float
    typical_miss: float
    repeat_sd: float | None     # spread between Repeats; None: no Repeats yet
    ratio: float | None         # typical miss / (MAD_OF_NORMAL * repeat_sd)
    widen: float | None = None  # the current fit's widening; None without a model
    pairs: int = 0              # pairs of Repeats repeat_sd rests on
    left_out: int = 0           # pairs left out because a Repeat disagreed
    # 0.8.0 e2e fixes (review B1, B2): the mean error over every Round
    # checked (None without a fully measured check) and whether it is a
    # steady bias; the model's fitted noise sd in the output's units, as
    # context (None without one), and whether the noise is fixed instead.
    mean_error: float | None = None
    bias: bool = False
    fitted_noise: float | None = None
    noise_fixed: bool = False
    # repeat_sd's degrees of freedom (ftue fix A, cold read I11): the Repeat
    # SD is the noise floor from YARDSTICK_DF of them.
    df: int = 0


# The median absolute value of a standard normal: a typical miss (a median)
# is compared with a typical difference between Repeats on the same footing
# (spec 2026-09-30 §2.5, review M6).
MAD_OF_NORMAL = 0.6744897501960817


def details(opt, checks, fitted=None, flagged=None):
    """One row per output checked at least once, in Results' order. R² is
    gone (0.8.0 review fixes §3.4): on formulations the suggestions chose
    it is below 0 even for a perfect prediction one time in seven. The
    typical miss is compared with the typical difference between Repeats,
    like with like (spec 2026-09-30 review M6). `fitted`: the current
    models, for the "Likely range widened by" column (review M5).
    `flagged`: accepted for callers; a disagreeing Repeat now stays in every
    output's Repeats (ftue fix A, cold read I7), so left_out is 0. Each row also carries its mean error over every
    Round checked, whether that is a steady bias (steady_bias's rule), and
    the model's fitted noise sd (0.8.0 e2e fixes, review B1, B2)."""
    rows = []
    matches = None
    tiny = (getattr(opt, "bo_config", None) or {}).get("noise") == "fixed_tiny"
    ordered = [o['name'] for o in opt.measurements_by_importance()] + [SCORE]
    for name in ordered:
        mine = [c for c in checks if c.name == name]
        if not mine:
            continue
        miss = statistics.median(abs(c.measured - c.predicted) for c in mine)
        if matches is None:
            # The recipe comparisons, once for every output (review M-4):
            # each output's rows are among the fully measured ones.
            matches = _matches(opt, [i for i in range(len(opt.Y_history))
                                     if opt.fully_measured(i)])
        # Ftue fix A (cold read I7): a disagreeing Repeat stays in.
        spread = repeat_sd(opt, name, matches)
        left_out = 0
        sd = spread.sd
        f = (fitted or {}).get(name)
        full = [c.measured - c.predicted for c in mine if c.fully_measured]
        floor = (spread.sd if spread.sd is not None and spread.sd >= REPEAT_SD_SHOWN
                 and spread.df >= YARDSTICK_DF else assumed_noise(opt) if name == SCORE
                 else 0.0)
        gp = f.model if f is not None else None
        noise = None if gp is None or tiny else fitted_noise_sd(gp)
        rows.append(DetailRow(
            name, len(ranged(mine)), sum(c.inside for c in ranged(mine)),
            statistics.median((c.high - c.low) / 2 for c in mine), miss,
            # sd 0: the Repeats agreed exactly, and there is no ratio.
            sd, None if not sd else miss / (MAD_OF_NORMAL * sd),
            f.widen if f is not None and f.model is not None else None, spread.pairs,
            left_out,
            mean_error=statistics.fmean(full) if full else None,
            bias=bias_gap(error_record(checks, name), floor) is not None,
            fitted_noise=noise, noise_fixed=gp is not None and noise is None,
            df=spread.df))
    return rows


# ------------------------------------------------------------------ #
#  0.8.0 speed and simplicity, Plan B: one yardstick, one verdict, one
#  next step (spec 2026-09-30 §2; review I2, I4, I5, I6). Every function
#  here takes the checks as an argument, so they can come from a cache.
# ------------------------------------------------------------------ #

# A pooled spread between Repeats is the yardstick only from this many
# degrees of freedom (review I5: "don't benchmark on one pair"; ftue fix A,
# cold read I11: degrees of freedom, not pairs, since one formulation made
# three times is 3 pairs but 2 df).
YARDSTICK_DF = 3
# The over-wide verdict (review I6): every result inside, at least this many
# checks, and a median |z| this small. Simulated with calibrated misses it
# fires 2.9 % of the time at 20 checks and 0.4 % at 33.
OVER_WIDE_MIN = 20
OVER_WIDE_Z = 0.4


@dataclass
class Yardstick:
    value: float | None     # the noise SD: Repeats' sd from 3+ df, else 1 point (None
                            # for a measurement)
    sd: float | None        # the pooled sd between Repeats
    pairs: int              # how many pairs of Repeats sd rests on
    from_repeats: bool      # value is sd (3+ df, at least REPEAT_SD_SHOWN)
    # 0.8.0 e2e fixes (review B2): the Overall score's 1 point is assumed,
    # not measured, while sd has fewer than YARDSTICK_DF degrees of freedom;
    # the screen says so and Next step asks for a Repeat of Best so far.
    assumed: bool = False
    df: int = 0             # sd's degrees of freedom (review I13)
    groups: int = 0         # formulations made more than once
    # Pairs left out because a Repeat disagrees (polish round: Line B's
    # only Repeat disagrees, and "no Repeats yet" read as false).
    left_out: int = 0


@dataclass
class Disagreement:
    later: int              # the Repeat's Formulation number
    earlier: int            # the Formulation number it repeats
    diff: float             # later minus earlier, Overall score
    row: int                # the later one's row
    limit: float | None = None  # √2 × 1.96 × sigma, the limit it broke, points
    # Ftue fix A (cold read I7): the σ the limit rests on and where it came
    # from: "model" (the Round-ahead model's fitted noise SD), "repeats"
    # (the Repeat SD without this pair, from YARDSTICK_DF df) or "assumed"
    # (the assumed 1 point, while that Repeat SD has fewer df). `df`: that
    # Repeat SD's degrees of freedom (0 without one); `noise`: the model's
    # fitted noise SD (None when not fitted); `repeat_sd`: the Repeat SD
    # without this pair (None without one).
    sigma: float | None = None
    basis: str = "assumed"
    df: int = 0
    noise: float | None = None
    repeat_sd: float | None = None
    # Fix A follow-up (owner, decision 2): while that Repeat SD has fewer
    # than YARDSTICK_DF df, the pair must also beat √2 × the later one's own
    # Round-ahead 95 % half-width (the earlier rule); that limit, in points.
    # None from YARDSTICK_DF df, when the Repeat SD alone sets the limit.
    predictive: float | None = None


# The Overall score's noise SD while Repeats have fewer than YARDSTICK_DF
# degrees of freedom: the assumed 1 point (noise_yardstick's fallback).
ASSUMED_NOISE_SD = 1.0
# Final demo (B5): the tasting noise choices on What next? (points, 1 SD).
TASTING_NOISE_CHOICES = (0.5, 1.0, 2.0, 3.0)


def assumed_noise(opt):
    """Final demo (B5): the Overall score's noise SD while Repeats give
    under YARDSTICK_DF df — the project's own tasting noise when set
    (FoodOptimizer.tasting_noise), else the assumed 1 point. Review-side
    only: suggestions never read it."""
    v = getattr(opt, "tasting_noise", None) if opt is not None else None
    try:
        v = None if v is None or isinstance(v, bool) else float(v)
    except (TypeError, ValueError):
        v = None
    return v if v is not None and math.isfinite(v) and v > 0 else ASSUMED_NOISE_SD


def set_tasting_noise(opt, value):
    """Store the project's tasting noise (None or the assumed 1 point: back
    to the default, None) and save. Nothing else changes."""
    v = None if value is None or abs(float(value) - ASSUMED_NOISE_SD) < 1e-9 else float(value)
    opt.tasting_noise = v
    opt.save()
    return v


def repeat_sigma(noise, spread, assumed=ASSUMED_NOISE_SD):
    """(σ, basis) for the Repeat check (ftue fix A, cold read I7): the larger
    of the model's fitted noise SD `noise` (None when not fitted) and the
    Overall score's noise SD without the pair tested: `spread`'s sd from
    YARDSTICK_DF degrees of freedom (at least REPEAT_SD_SHOWN), else the
    assumed 1 point, as noise_yardstick has it. The display model's fitted
    noise alone runs well below what Repeats show (0.3-0.4 points on Lines
    A to C under the default prior), and would make an ordinary Repeat
    disagree."""
    usable = (spread is not None and spread.sd is not None
              and spread.sd >= REPEAT_SD_SHOWN and spread.df >= YARDSTICK_DF)
    rep, basis = (spread.sd, "repeats") if usable else (assumed, "assumed")
    if noise is not None and noise > rep:
        return noise, "model"
    return rep, basis


def disagreeing_repeat(opt, checks, matches=None):
    """The one Repeat whose Overall score differs from its original by more
    than two results should (review I4), or None. The Repeat SD is taken
    with the later one left out (leave-one-out: the pair tested never sets
    its own limit).

    - From YARDSTICK_DF degrees of freedom: |d| over √2 × t_{0.975, df} ×
      that Repeat SD alone (fix A follow-up, owner's decision 2; t, not
      1.96, whenever the SD comes from Repeats, the coordinator's ruling on
      03 drift: its F20 vs F14, 1.62 apart, is under √2 × 3.18 × 0.49 =
      2.20).
    - Before: |d| over BOTH √2 × 1.96 × σ, σ the larger of the later one's
      Round-ahead model's fitted noise SD and the assumed 1 point
      (repeat_sigma; ftue fix A, cold read I7), AND √2 × the later one's
      own Round-ahead 95 % half-width (the earlier rule). 03 clean's
      ordinary Repeat (2.84 points, true noise about 1.3) no longer reads
      as a disagreement on the assumed 1 point alone.

    Only
    a later one in the verdict's Rounds counts, a group of three or more
    compares each with its first, either direction counts, and of several
    the largest |d| is named. The flagged pair stays in every reported
    Repeat SD. `matches`: as for repeat_groups."""
    window = set(headline(opt, checks).rounds)
    if not window:
        return None
    ahead = {c.formulation: c for c in checks
             if c.name == SCORE and c.fully_measured and c.round in window}
    if not ahead:
        return None
    pool = opt.best_candidates()
    if matches is None:
        matches = _matches(opt, pool)
    best = None
    for group in repeat_groups(opt, pool, matches):
        first, *rest = sorted(group)
        for i in rest:
            c = ahead.get(int(opt.formulation_ids[i]))
            if c is None:
                continue
            noise = getattr(c, "noise", None)
            spread = repeat_sd(opt, SCORE, matches, (i,))
            if (spread.sd is not None and spread.sd >= REPEAT_SD_SHOWN
                    and spread.df >= YARDSTICK_DF):
                sigma, basis, predictive = spread.sd, "repeats", None
                limit = math.sqrt(2) * t_critical(spread.df) * sigma
            else:
                sigma, basis = repeat_sigma(noise, spread, assumed_noise(opt))
                # The half-width shown for that check, its I1 widening included.
                half = Z95 * c.sd * c.widen if c.sd else (c.high - c.low) / 2
                predictive = math.sqrt(2) * half
                limit = max(math.sqrt(2) * Z95 * sigma, predictive)
            d = float(opt.Y_history[i]) - float(opt.Y_history[first])
            if abs(d) > limit and (best is None or abs(d) > abs(best.diff)):
                best = Disagreement(int(opt.formulation_ids[i]),
                                    int(opt.formulation_ids[first]), d, i, limit,
                                    sigma, basis, spread.df, noise, spread.sd, predictive)
    return best


def noise_yardstick(opt, checks=(), name=SCORE):
    """The noise SD a difference is judged by (review I5): the pooled sd
    between Repeats of `name`, from YARDSTICK_DF degrees of freedom or more
    and at least REPEAT_SD_SHOWN; else 1 point for the Overall score (None
    for a measurement). Every rule that asks "more than Repeats differ?"
    starts here; Rising also takes √2 and the Round's size into account
    (rise_threshold). A disagreeing Repeat stays in (ftue fix A, cold read
    I7: leaving it out biased the Repeat SD down), so `checks` no longer
    change the answer and left_out is 0."""
    spread = repeat_sd(opt, name, None)
    usable = (spread.sd is not None and spread.sd >= REPEAT_SD_SHOWN
              and spread.df >= YARDSTICK_DF)
    fallback = assumed_noise(opt) if name == SCORE else None
    return Yardstick(spread.sd if usable else fallback, spread.sd, spread.pairs, usable,
                     assumed=name == SCORE and spread.df < YARDSTICK_DF,
                     df=spread.df, groups=spread.groups, left_out=0)


def verdict_state(opt, h, checks):
    """One verdict, in order (review I6): "none", "early", "unreliable",
    "caution", "wide" (every output in the verdict's Rounds wide),
    "over_wide", or "within"."""
    if h.state != "verdict":
        return h.state
    if h.verdict in ("unreliable", "caution"):
        return h.verdict
    window = [c for c in checks if c.round in h.rounds]
    names = {c.name for c in window}
    if names and names <= set(h.wide or []):
        return "wide"
    z = median_z(h, checks)
    if (h.inside == h.total and h.total >= OVER_WIDE_MIN and z is not None
            and z <= OVER_WIDE_Z):
        return "over_wide"
    return "within"


def median_z(h, checks):
    """The median |standardised error| of the measurements in the
    verdict's Rounds (0.67 when the 95 % intervals are right), or None:
    the over-wide rule's basis, said with it (03 §5.2)."""
    window = [c for c in checks if c.round in h.rounds]
    z = [abs((c.measured - c.raw_mean) / c.sd) for c in window
         if c.name != SCORE and c.sd and c.raw_mean is not None]
    return statistics.median(z) if z else None


# --- review diagnosis (display only, 2026-10-02) ---
# Where the verdict's misses fell, said as a description, never as a cause:
# Round-ahead errors cannot show drift or a panel problem (spec
# 2026-10-01-insights-first-time-design; e2e fixes), only a disagreeing
# Repeat can. "New ground" is suggestion_role's own rule: farther than
# FAR_FROM_TESTED (L-infinity share of each varying row's span) from every
# formulation recorded before the miss's Round.
MOSTLY = 0.5        # more than this share of the misses: "mostly"
ONE_WAY = 0.8       # this share of the misses on their measurement's usual side
ONE_WAY_MIN = 2     # misses a measurement needs before its side counts


@dataclass
class MissItem:
    name: str
    misses: int
    checked: int
    above: int              # measured above the interval (predictions ran low)
    below: int
    mean_error: float | None   # mean measured − predicted over the misses


@dataclass
class MissDiagnosis:
    kind: str               # "new_ground", "one_way" or "unexplained"
    rounds: list
    misses: int
    checked: int
    far: int                # misses on formulations far from anything tested before
    far_checked: int        # values checked on such formulations
    one_way: int            # misses on their measurement's usual side
    items: list             # MissItem, most misses first
    score: object = None    # the Overall score's ScoreRecord (Headline.score)
    score_ok: bool = False  # its misses within allowed_misses: as expected
    # "one_way": the usual side of every measurement with ONE_WAY_MIN misses
    # or more, "above", "below" or "mixed" (they differ); None otherwise.
    side: str | None = None


def far_from_tested(opt, formulation, round_no):
    """True when Formulation `formulation` sits farther than
    FAR_FROM_TESTED from every formulation recorded before Round
    `round_no` (FoodOptimizer.suggestion_role's "explores" rule); None
    when the formulation is no longer on file."""
    from food_bo import FAR_FROM_TESTED
    ids = [int(x) for x in opt.formulation_ids]
    if formulation not in ids:
        return None
    mine = start_recipe(opt, ids.index(formulation))
    ds = [distance(opt, mine, r) for r in opt._recorded_before(round_no)]
    return all(d is None or d > FAR_FROM_TESTED + _EPS for d in ds)


def miss_diagnosis(opt, checks, h=None):
    """Where the measurements' misses in the verdict's Rounds fell (display
    only): by measurement, side and mean error, and how many sat on
    formulations far from anything tested before their Round. None with no
    miss. `kind`: "new_ground" when more than half the misses are far AND
    far values missed at least as often as the rest (so the misses are not
    merely where most checks were); else "one_way" when at least ONE_WAY
    of them sit on their measurement's usual side (measurements with
    ONE_WAY_MIN misses or more); else "unexplained"."""
    h = h or headline(opt, checks)
    values = [c for c in ranged(checks) if c.name != SCORE and c.round in (h.rounds or [])]
    misses = [c for c in values if not c.inside]
    if not misses:
        return None
    seen = {}

    def far(c):
        key = (c.formulation, c.round)
        if key not in seen:
            seen[key] = bool(far_from_tested(opt, c.formulation, c.round))
        return seen[key]

    order = [n for n, _, _ in outputs(opt) if n != SCORE]
    items = []
    for name in order + sorted({c.name for c in values} - set(order)):
        mine = [c for c in misses if c.name == name]
        if not mine:
            continue
        above = sum(c.measured > c.high for c in mine)
        items.append(MissItem(name, len(mine), sum(c.name == name for c in values), above,
                              len(mine) - above,
                              statistics.fmean(c.measured - c.predicted for c in mine)))
    items.sort(key=lambda m: -m.misses)
    n_far = sum(far(c) for c in misses)
    far_checked = sum(far(c) for c in values)
    near_checked = len(values) - far_checked
    # Far values miss at least as often as near ones (a near-free Round counts).
    concentrated = (near_checked == 0 or
                    n_far * near_checked >= (len(misses) - n_far) * far_checked)
    one_way = sum(max(m.above, m.below) for m in items if m.misses >= ONE_WAY_MIN)
    side = None
    if n_far > MOSTLY * len(misses) and concentrated:
        kind = "new_ground"
    elif one_way >= ONE_WAY * len(misses):
        kind = "one_way"
        sides = {"above" if m.above >= m.below else "below"
                 for m in items if m.misses >= ONE_WAY_MIN}
        side = sides.pop() if len(sides) == 1 else "mixed"
    else:
        kind = "unexplained"
    score = h.score
    score_ok = bool(score is not None and score.total
                    and score.total - score.inside <= allowed_misses(score.total))
    return MissDiagnosis(kind, list(h.rounds), len(misses), len(values), n_far,
                         far_checked, one_way, items, score, score_ok, side)


# The CV-widening "check your panel's consistency" (ui_insights.wide_cause)
# is backed only when Repeats show the noise is large: the Repeat SD, from
# YARDSTICK_DF degrees of freedom (noise_yardstick's own rule), at least
# this share of the spread of scores between formulations.
REPEATS_VARY = 0.5


@dataclass
class RepeatsVary:
    sd: float           # the Repeat SD, points
    df: int
    pairs: int
    spread: float       # sd of the scored formulations' Overall scores


def repeats_vary(opt, yard=None):
    """RepeatsVary when the Repeats measure the noise and it is large
    against the differences between formulations, else None."""
    yard = yard if yard is not None else noise_yardstick(opt)
    if not getattr(yard, "from_repeats", False) or yard.value is None:
        return None
    ys = [float(opt.Y_history[i]) for i in opt.best_candidates()]
    if len(ys) < 3:
        return None
    spread = statistics.stdev(ys)
    if yard.value < REPEATS_VARY * spread:
        return None
    return RepeatsVary(float(yard.value), int(yard.df), int(yard.pairs), spread)


@dataclass
class NextStep:
    kind: str       # "repeat", "misses", "early", "no_skill", "untested",
                    # "explore", "close", "rising" or "see"
    gap: float | None = None                    # "misses": the Overall score's steady bias
    disagreement: Disagreement | None = None    # "repeat"
    binding: object = None                      # "close": what binds the predicted best
    wide: list = dataclasses.field(default_factory=list)   # "rising": outputs still wide
    skill: object = None                        # "no_skill": the SkillFact (BO views, I1)
    gain: float | None = None                   # "explore": the expected improvement, points
    bias_rounds: list = dataclasses.field(default_factory=list)  # "misses": gap's Rounds
    # "close", "explore", "see", "rising": the noise threshold is assumed (fewer
    # than 3 Repeat pairs), so the row asks for a Repeat of Best so far (B2, I5).
    assumed: bool = False
    # "misses" with no disagreeing Repeat: where the misses fell (display only).
    diagnosis: MissDiagnosis | None = None
    # "close": every limit the predicted best sits at (review critique: name
    # them all, not the first only).
    bindings: list = dataclasses.field(default_factory=list)
    # "close" (specific wording): the noise threshold the expected gain
    # (`gain`) fell under.
    noise: float | None = None


def next_step(opt, checks, standing=None, gain=None):
    """What to do next, first match wins (0.8.0 e2e fixes, review I2):
    a disagreeing Repeat ("repeat", the one row that says to check the
    bench, conditionally: review B1); an unreliable verdict ("misses", with
    the Overall score's steady bias as a fact, never advice); too early;
    no skill; then the rest. `standing`: the
    predicted best against Best so far (an object with .higher,
    .tested_near and .binding), or None when it could not be worked out;
    the rows that need it are then skipped. `gain`: the expected
    improvement Progress shows (points), or None where it shows none.
    "Close to the top" needs BOTH the predicted gain and the expected
    improvement at or under the noise threshold (fix round 1, ledger
    ruling C1); a predicted gain under it with an expected improvement
    above it is "explore": the model still expects that much from
    formulations it is unsure about. Never called at the gate (fewer than
    MIN_ROWS scored) or after a failed fit: the screen says the gate line
    there."""
    h = headline(opt, checks)
    state = verdict_state(opt, h, checks)
    yard = noise_yardstick(opt, checks)
    flagged = disagreeing_repeat(opt, checks)
    if flagged is not None:
        return NextStep("repeat", disagreement=flagged)
    if state == "unreliable":
        gap = score_bias(opt, checks, h, yard)
        return NextStep("misses", gap=gap,
                        bias_rounds=list(h.score.rounds or []) if gap is not None else [],
                        diagnosis=miss_diagnosis(opt, checks, h))
    # "none" here is past the gate with no Round checked yet, so it reads
    # as row 3; the ruling's "none" (row 9) is the no-data gate the caller says.
    if state in ("none", "early", "wide"):
        return NextStep("early")
    # BO views (review I1): before untested and close, which both rest on
    # the model's predicted best.
    fact = skill(opt, checks)
    if fact is not None and fact.state == "no_skill":
        return NextStep("no_skill", skill=fact)
    answer = improvement(opt, yardstick=yard)
    assumed = bool(getattr(yard, "assumed", False))
    # Review I4: while the A3 gate withholds effect claims, the rows that
    # cite the model's predicted best (untested, close, explore) fall
    # through to rising or see.
    # Ftue review R-I11 (display only): under More misses than expected the
    # record does not back the predicted best either, so they fall through
    # too.
    if (standing is not None and state != "caution"
            and skill_gate(opt, checks, fact) is None):
        if standing.higher and not standing.tested_near:
            return NextStep("untested")
        if (not standing.higher and answer is not None
                and answer.state in ("not_lately", "barely")):
            if gain is not None and gain > yard.value:
                return NextStep("explore", gain=gain, assumed=assumed)
            return NextStep("close", binding=standing.binding[0] if standing.binding else None,
                            assumed=assumed, bindings=list(standing.binding or []),
                            gain=gain, noise=yard.value)
    if answer is not None and answer.state == "yes":
        # Said with the width caution the verdict gives (answer-key
        # read-through, Line C).
        return NextStep("rising", wide=list(h.wide or []), assumed=assumed)
    return NextStep("see", assumed=assumed)


# ------------------------------------------------------------------ #
#  Insights for a first-time user (spec 2026-10-01 rev. 2), W1: the three
#  status lines' facts. T0 stub with its final signature; W1 fills it in.
#  The words are wording's (PREDICTIONS_WORDS, PROGRESS_WORDS, pred_*,
#  progress_*, next_plain_*); W2's tooltips read these facts.
# ------------------------------------------------------------------ #

# Review R-B1: the Predictions rows, worst first. The first that holds sets
# the status; PREDICTION_STATUS gives its word's key.
# Integration ruling (ledger): caution sits before too_few, "a warning is
# never folded": More misses than expected is said even while the skill
# gate still counts fewer than 10 checks.
PREDICTION_ROWS = ("off_repeat", "off_coverage", "gate", "caution", "too_few", "wide",
                   "no_skill", "unclear", "reliable")
PREDICTION_STATUS = {row: ("off" if row.startswith("off") else
                           "reliable" if row == "reliable" else "not_yet")
                     for row in PREDICTION_ROWS}
# Review R-B1, R-I10: the Progress words' keys. improvement()'s "yes",
# "barely" and "not_lately" are rising, within_noise and flat; None is
# too_early (one Round, or no earlier candidate) or no_new_best (one Round
# since).
PROGRESS_STATES = ("rising", "within_noise", "flat", "too_early", "no_new_best")


@dataclass
class StatusFacts:
    """What the three status lines say, and the numbers behind each
    (review R-B1). None in place of the whole when there are no status
    lines (nothing scored, a failed fit)."""
    scored: int                                  # complete_count()
    prediction_row: str | None = None            # a PREDICTION_ROWS key
    clauses: list = dataclasses.field(default_factory=list)
    # ("wide", [names]) · ("over_wide",) · ("bias", gap_points) · ("guard",)
    headline: object = None                      # Headline
    verdict: str | None = None                   # verdict_state()
    skill: object = None                         # SkillFact or None
    disagreement: object = None                  # Disagreement or None
    bias: float | None = None                    # the Overall score's steady bias, points
    guard: object = None                         # Guard or None
    yardstick: object = None                     # Yardstick
    progress: str | None = None                  # a PROGRESS_STATES key
    improvement: object = None                   # Improvement or None
    best: object = None                          # BestFacts
    tie: int | None = None                       # row of an equal-score runner-up (R-I9)
    luck: bool = False                           # said at level 0 only when Reliable
    next: object = None                          # NextStep, or None at the gate
    # W1: the Overall-score checks the skill gate counts (row 4 says them:
    # formulations, never the verdict's values; review B1).
    checked: int = 0
    # W1 (R-I9): the predicted Repeat of Best so far the luck clause names,
    # points; None when luck is not said.
    luck_repeat: float | None = None
    # W1 (R-I8): the lines whose facts are still being worked out while the
    # workers fit ("progress", "next"): their places keep their captions.
    pending: tuple = ()
    # Review diagnosis: where the misses fell, under Off (coverage) or More
    # misses than expected with no disagreeing Repeat; None otherwise.
    diagnosis: MissDiagnosis | None = None
    # Review four, ruling 1 (owner, 2026-10-02): the verdict word is the
    # Overall score's (verdict, above, is score_verdict_state's); the
    # measurements' own verdict (verdict_state) is kept here, for the
    # card's secondary line and the tooltip, which report both.
    measure_verdict: str | None = None
    # Specific wording: which measurements missed, how often, against the
    # 1 in 20 a correct range gives (value_misses); None while they do not
    # miss more than expected.
    misses: object = None

    @property
    def measures_miss(self):
        """The measurements miss more than expected (their own verdict
        caution or unreliable): a secondary, non-red line on the card."""
        return self.measure_verdict in ("unreliable", "caution")

    @property
    def predictions(self):
        """'reliable', 'not_yet' or 'off'; None without a row."""
        return PREDICTION_STATUS.get(self.prediction_row)


def prediction_row(scored, verdict, flagged, fact, checked):
    """R-B1's Predictions rows, worst first: the first that holds. `scored`:
    complete_count(); `verdict`: verdict_state(); `flagged`:
    disagreeing_repeat(); `fact`: skill(); `checked`: the Overall-score
    checks the skill gate counts. A disagreeing Repeat beside an unreliable
    record reads as the coverage (Next carries the Repeat). More misses than
    expected comes before too few checks (integration ruling: a warning is
    never folded). Ten checks or
    more with no skill (every check on its baseline) is not clearly
    better."""
    if scored < MIN_ROWS:
        return "gate"
    if flagged is not None and verdict != "unreliable":
        return "off_repeat"
    if verdict == "unreliable":
        return "off_coverage"
    if verdict == "caution":
        return "caution"
    if verdict in ("none", "early") or (fact is None and checked < SKILL_MIN):
        return "too_few"
    if verdict == "wide":
        return "wide"
    if fact is not None and fact.state == "no_skill":
        return "no_skill"
    if fact is None or fact.state == "unclear":
        return "unclear"
    return "reliable"


# R-B1: the rows the clauses are appended to (4-9), and the guard's two
# "fitting noise" triggers (R-I11; trigger (c) is the skill gate, which a
# Reliable row has passed).
_CLAUSED = ("too_few", "caution", "wide", "no_skill", "unclear", "reliable")
_NOISE_TRIGGERS = ("floor", "noise")
_PROGRESS_WORD = {"yes": "rising", "barely": "within_noise", "not_lately": "flat"}


# --- review four (ruling 1, display and status only, 2026-10-02) ---
# The suggestions are made from the Overall-score model, so the Predictions
# word reads the Overall score's own Round-ahead record: its coverage over
# every Round checked (Headline.score, fully measured formulations) and its
# skill. The measurements' coverage (verdict_state) stays a fact beside it.
def score_verdict_state(opt, h, checks):
    """The Overall score's verdict: "none" (no Round checked), "early"
    (fewer than SKILL_MIN Overall-score checks), "unreliable" or "caution"
    (the binomial tail of its misses under 0.01 or 0.10, as the
    measurements' verdict), "wide" (its own 95 % ranges wide in the
    verdict's Rounds), else "within"."""
    record = getattr(h, "score", None)
    if h.state == "none" and record is None:
        return "none"
    if record is None or record.total < SKILL_MIN:
        return "early"
    tail = binomial_tail(record.total - record.inside, record.total)
    if tail < 0.01:
        return "unreliable"
    if tail < 0.10:
        return "caution"
    if SCORE in (h.wide or []):
        return "wide"
    return "within"


def score_prediction_row(scored, verdict, fact, checked):
    """prediction_row's rows on the Overall score's verdict: the disagreeing
    Repeat is no longer a Predictions row (it is the Next card's, in red:
    ruling 1); the rest as before, worst first."""
    if scored < MIN_ROWS:
        return "gate"
    if verdict == "unreliable":
        return "off_coverage"
    if verdict == "caution":
        return "caution"
    if verdict in ("none", "early") or (fact is None and checked < SKILL_MIN):
        return "too_few"
    if verdict == "wide":
        return "wide"
    if fact is not None and fact.state == "no_skill":
        return "no_skill"
    if fact is None or fact.state == "unclear":
        return "unclear"
    return "reliable"


def score_predictions(opt, checks):
    """The Predictions card's word key ("reliable", "not_yet" or "off"), as
    status_facts reads it (review four polish: Expected gain is greyed by
    this word, so its line never contradicts the card)."""
    checks = list(checks or ())
    h = headline(opt, checks)
    verdict = score_verdict_state(opt, h, checks)
    fact = skill(opt, checks) if checks else None
    checked = sum(1 for c in checks if c.name == SCORE and c.fully_measured)
    return PREDICTION_STATUS[score_prediction_row(opt.complete_count(), verdict, fact,
                                                  checked)]


def status_facts(opt, checks, standing=None, gain=None):
    """W1 (review R-B1): the facts of the three status lines, worst rule
    first, from headline, verdict_state, skill, disagreeing_repeat,
    score_steady_bias, noise_guard, improvement, best_facts and tie_partner;
    `standing` and `gain` as next_step takes them. None when nothing is
    scored or waiting (every formulation moved to Not scored).

    - Predictions: prediction_row's row; clauses on rows 4-9 in the
      ruling's order: some outputs (not all) wide, over_wide, the Overall
      score's steady bias (row 2 too, as its own clause), and the noise
      guard's floor or noise trigger under Reliable alone (R-I11).
    - Progress: improvement()'s answer as its word; with none, No new best
      after one Round since, else Too early. `tie`: an equal-score runner-up
      at two decimals (R-I9); `luck`: Best so far's luck line, said at level
      0 only while Predictions is Reliable with no guard clause (R-I9, I7).
    - Next: next_step's row past the gate (None at it).
    `guard`: the full noise guard under Reliable (its model from the memo
    in use, else fitted); the skill gate alone otherwise."""
    rows = list(range(len(opt.Y_history)))
    if not rows or all(opt.is_set_aside(i) for i in rows):
        return None
    checks = list(checks or ())
    scored = opt.complete_count()
    h = headline(opt, checks)
    # Review four, ruling 1: the word on the Overall score's record; the
    # measurements' verdict kept beside it (measure_verdict).
    measure_verdict = verdict_state(opt, h, checks)
    verdict = score_verdict_state(opt, h, checks)
    flagged = disagreeing_repeat(opt, checks) if checks else None
    fact = skill(opt, checks) if checks else None
    checked = sum(1 for c in checks if c.name == SCORE and c.fully_measured)
    yard = noise_yardstick(opt, checks)
    bias = score_steady_bias(opt, checks, flagged) if checks else None
    row = score_prediction_row(scored, verdict, fact, checked)
    clauses, guard = [], None
    if row in _CLAUSED:
        wide = list(h.wide or [])
        if wide and row != "wide":
            clauses.append(("wide", wide))
        if measure_verdict == "over_wide":
            clauses.append(("over_wide",))
        if bias is not None:
            clauses.append(("bias", bias))
    elif row == "off_coverage" and bias is not None:
        clauses.append(("bias", bias))
    if row != "gate":
        model = fit(opt, SCORE, rows) if row == "reliable" else None
        guard = noise_guard(opt, model, checks, fact=fact)
        if row == "reliable" and any(r in guard.reasons for r in _NOISE_TRIGGERS):
            clauses.append(("guard",))
    best = best_facts(opt)
    progress = answer = tie = None
    if best is not None:
        answer = improvement(opt, best, yardstick=yard)
        progress = (_PROGRESS_WORD[answer.state] if answer is not None
                    else "no_new_best" if best.rounds_since == 1 else "too_early")
        top = opt.best_index()
        other = None if top is None else tie_partner(opt, top)
        if (other is not None
                and f"{float(opt.Y_history[other]):.2f}" == f"{float(opt.Y_history[top]):.2f}"):
            tie = other
    luck = bool(row == "reliable" and ("guard",) not in clauses and standing is not None
                and getattr(standing, "luck", False)
                and getattr(standing.best, "repeat", None) is not None)
    step = None if row == "gate" else next_step(opt, checks, standing, gain)
    # Where the measurements' misses fell (the tooltip's), whenever they
    # miss more than expected and no Repeat disagrees.
    diagnosis = (miss_diagnosis(opt, checks, h)
                 if measure_verdict in ("unreliable", "caution") and flagged is None else None)
    misses = (value_misses(opt, checks, h)
              if measure_verdict in ("unreliable", "caution") else None)
    return StatusFacts(scored, row, clauses, h, verdict, fact, flagged, bias, guard, yard,
                       progress, answer, best, tie, luck, step, checked,
                       float(standing.best.repeat.mean) if luck else None,
                       diagnosis=diagnosis, measure_verdict=measure_verdict, misses=misses)


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
    base = dict(name=name, unit=opt.unit_of(name), mode=mode,
                fill_name=opt.balance_row_name() if mode == "fill" else None,
                start_amount=s, skipped=skipped, outside=outside,
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


def tested_near(opt, path):
    """For each point of `path`, whether a fully measured formulation lies
    within CLOSE_TO_THE_BEST of it on every varying row — the tab's own
    "not tested near here" rule, which question 4 and the triangle use
    (0.8.0 review fixes §9 I13). Each row's own min-max, as before, left
    85 % of the curve white while untested by this rule."""
    tested = [start_recipe(opt, i) for i in range(len(opt.Y_history))
              if opt.fully_measured(i)]
    out = []
    for r in path.recipes:
        ds = [distance(opt, r, t) for t in tested]
        out.append(any(d is not None and d <= CLOSE_TO_THE_BEST + _EPS for d in ds))
    return out


def untested_runs(path, near):
    """The stretches of `path` not tested near here, as (from, to) amounts:
    each run of untested points, reaching halfway to its tested neighbours
    so the shading has no gaps."""
    a, runs, k = path.amounts, [], 0
    while k < len(a):
        if near[k]:
            k += 1
            continue
        j = k
        while j + 1 < len(a) and not near[j + 1]:
            j += 1
        lo = a[0] if k == 0 else (a[k - 1] + a[k]) / 2
        hi = a[-1] if j == len(a) - 1 else (a[j] + a[j + 1]) / 2
        runs.append((lo, hi))
        k = j + 1
    return runs


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


def scored_rounds(opt):
    """round_order for the Rounds with at least one formulation Best so far
    compares. A Round still all waiting, or all moved to Not scored, has
    not been scored: it is neither the latest Round nor a Round "without a
    higher score" (0.8.0 review fixes, Task 6 review I-1)."""
    pool = {opt.batch_history[i] for i in opt.best_candidates()}
    return [(r, first) for r, first in round_order(opt) if r in pool]


def best_facts(opt):
    """Best so far, and where it stands among the scored Rounds."""
    i = opt.best_index()
    if i is None:
        return None
    order = scored_rounds(opt)
    since = [r for r, first in order if first > i]
    r = opt.batch_history[i]
    latest = r is not None and bool(order) and order[-1][0] == r
    return BestFacts(float(opt.Y_history[i]), int(opt.formulation_ids[i]), r,
                     len(since), latest, opt.best_scored_without())


@dataclass
class Improvement:
    state: str              # "yes", "barely" or "not_lately"
    rise: float | None      # the latest Round's best minus the Best so far before it
    sd: float | None        # the Overall score's spread between Repeats
    rounds: int             # Rounds since Best so far ("not_lately")
    # Ftue fix A (cold read I6): what the rise is judged against,
    # rise_threshold(noise SD, q), and q, the scored formulations of the
    # latest Round the rise is the top of. None and 0 for "not_lately".
    threshold: float | None = None
    q: int = 0
    # Fix A follow-up (owner, decision 3): the degrees of freedom of the
    # Repeat SD the threshold uses t on; 0 when the noise SD is the assumed
    # 1 point (z).
    df: int = 0


# The one-sided false-alarm rate of Rising (ftue fix A, cold read I6).
RISE_ALPHA = 0.05


def rise_z(q):
    """z_q = Φ⁻¹((1 − RISE_ALPHA)^(1/q)): the one-sided normal quantile at
    which the top of q results as good as the old best beats it by chance
    with probability RISE_ALPHA (Šidák). 1.645 at q = 1, 2.23 at q = 4."""
    return float(ndtri((1.0 - RISE_ALPHA) ** (1.0 / max(1, int(q)))))


def rise_quantile(q, df=None):
    """The max-of-q factor: rise_z(q) for the assumed noise SD (`df`
    None or 0), else its analogue on t, t_df⁻¹((1 − RISE_ALPHA)^(1/q)), for
    a noise SD estimated from Repeats with `df` degrees of freedom (fix A
    follow-up, owner's decision 3). 4.15 at q = 4 on 3 df."""
    if not df:
        return rise_z(q)
    return float(stdtrit(int(df), (1.0 - RISE_ALPHA) ** (1.0 / max(1, int(q)))))


def rise_threshold(sd, q, df=None):
    """What a rise must beat to read Rising (ftue fix A, cold read I6): the
    noise SD × √2 (a difference of two results) × rise_quantile(q, df) (the
    top of q new results; t on `df` when the SD comes from Repeats). With 4
    new formulations as good as the old best, Rising then shows about 1
    time in 20 by chance with a known SD, about 1 in 30 with one estimated
    on 3 df; against one noise SD it showed about 1 time in 2. `sd` may be
    a Yardstick (noise_yardstick): its value, and its df when it comes from
    Repeats. Ties ask rise_threshold(yard, 1), so a tie and Rising agree."""
    if isinstance(sd, Yardstick):
        df = sd.df if sd.from_repeats else None
        sd = sd.value
    return sd * math.sqrt(2) * rise_quantile(q, df)


# A spread between Repeats under this reads 0.0 on screen, so the tab
# neither shows it nor judges a rise against it: Repeats that matched
# exactly say nothing about chance (Task 6 review M-1).
REPEAT_SD_SHOWN = 0.05


def improvement(opt, facts=None, spread=None, yardstick=None):
    """The answer to "Is it still improving?" (0.8.0 review fixes §3.7,
    §9 I9), or None when there is nothing to say yet. Rounds are the
    scored ones (scored_rounds).

    - Not lately: no higher score in the last 2 or more Rounds.
    - When Best so far came from the latest Round, the rise is its score
      minus the Best so far before that Round: Barely when it is below
      rise_threshold(noise SD, q) (ftue fix A, cold read I6), Yes
      otherwise. The noise SD is the Overall score's spread between
      Repeats (1 point without Repeats, or with a spread under
      REPEAT_SD_SHOWN); q is how many scored formulations the latest Round
      has, the rise being the top of them. A noise SD from Repeats takes t
      on their degrees of freedom for z (fix A follow-up, owner's decision
      3; Improvement.df), the assumed 1 point keeps z. Both are compared to the
      hundredth, as the answer can print them, so its words never
      contradict its numbers.
    - Anything else (one Round only, or no higher score in just the
      Round since): None, and Best so far's own line says it.

    `facts` and `spread`: best_facts(opt) and repeat_sd(opt, SCORE), when
    the caller has them already. `yardstick` (noise_yardstick, spec
    2026-09-30 review I5), when given, replaces `spread`: the spread
    between Repeats counts only from YARDSTICK_DF df, else 1 point."""
    f = best_facts(opt) if facts is None else facts
    if f is None:
        return None
    if yardstick is not None:
        sd = yardstick.value if yardstick.from_repeats else None
        df = yardstick.df
    else:
        spread = repeat_sd(opt, SCORE) if spread is None else spread
        sd, df = spread.sd, spread.df
    if sd is not None and sd < REPEAT_SD_SHOWN:
        sd = None
    df = df if sd is not None else 0
    if f.rounds_since >= 2:
        return Improvement("not_lately", None, sd, f.rounds_since)
    if f.rounds_since or not f.latest:
        return None
    earlier = [float(opt.Y_history[i]) for i in opt.best_candidates()
               if opt.batch_history[i] != f.round]
    if not earlier:
        return None
    rise = f.score - max(earlier)
    q = sum(1 for i in opt.best_candidates() if opt.batch_history[i] == f.round)
    threshold = rise_threshold(sd if sd is not None else assumed_noise(opt), q, df)
    barely = round(rise, 2) < round(threshold, 2)
    return Improvement("barely" if barely else "yes", rise, sd, 0, threshold, q, df)


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


@dataclass
class RepeatSpread:
    sd: float | None        # pooled sd between Repeats; None without any
    pairs: int              # how many pairs of Repeats it rests on
    # Its degrees of freedom, Σ(n − 1) over the groups, and how many
    # formulations were made more than once (0.8.0 e2e fixes, review I13;
    # 03 I7: "8 pairs" of one formulation made four times is 5 df with two
    # others).
    df: int = 0
    groups: int = 0


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


def repeat_sd(opt, name, matches=None, without=()):
    """The spread between Repeats for one output: the pooled standard
    deviation within Repeat groups (0.8.0 review fixes §9 I9). The Overall
    score uses the formulations Best so far compares; a measurement, the
    fully measured formulations with a value for it. `matches`: as for
    repeat_groups, over the fully measured formulations or more.
    `without`: rows left out (a Repeat that disagrees, spec 2026-09-30
    review I4)."""
    if name == SCORE:
        rows = [i for i in opt.best_candidates() if i not in without]
        value = lambda i: float(opt.Y_history[i])
    else:
        rows = [i for i in range(len(opt.Y_history))
                if opt.fully_measured(i) and measured(opt, name, i) is not None
                and i not in without]
        value = lambda i: measured(opt, name, i)
    groups = repeat_groups(opt, rows, matches)
    if not groups:
        return RepeatSpread(None, 0)
    ss = dof = pairs = 0
    for g in groups:
        ys = [value(i) for i in g]
        mean = sum(ys) / len(ys)
        ss += sum((y - mean) ** 2 for y in ys)
        dof += len(ys) - 1
        pairs += len(ys) * (len(ys) - 1) // 2
    return RepeatSpread(math.sqrt(ss / dof), pairs, dof, len(groups))


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


def spreads(opt, fitted, start, name=SCORE):
    """How far each path choice moves the predicted `name` (the Overall
    score unless a measurement is named, 0.8.0 review fixes §4.1) from
    `start`: highest minus lowest mean along its path, largest first (ties
    keep path_choices order). A choice whose path the limits box in (under
    HELD of its Lowest-Highest) is named in `held` instead of given a
    spread. A path that cannot be drawn is left out; with no model for
    `name` there is nothing to say."""
    score = fitted.get(name)
    if score is None or score.model is None:
        return Spreads([], [])
    moved, held = [], []
    for choice in path_choices(opt):
        path = build_path(opt, choice, start)
        if path.start_breaks or not path.recipes:
            continue
        lo, hi = (float(b) for b in _var(opt, choice)['bounds'])
        if path.amounts[-1] - path.amounts[0] < HELD * (hi - lo):
            held.append(choice)
            continue
        means = [p.mean for p in predict(opt, score, [opt._encode(r) for r in path.recipes])]
        moved.append(Spread(choice, max(means) - min(means)))
    return Spreads(sorted(moved, key=lambda s: -s.spread), held)


# Two spreads this close are named together (§4.2): the named top was wrong
# in 55-70 % of the trust audit's misses when the second was within 20 %.
NEAR_TIE = 0.8
# Amounts that moved together this closely are named (§4.3).
TOGETHER = 0.7     # round-2 fixes: one |r| threshold on screen (MOVED_R's)
# A row whose amounts span less than this share of its Lowest-Highest never
# varied: float noise from a spreadsheet round-trip or a unit conversion
# is not a change anybody made (Task 8 review, Minor 2).
VARIED = 1e-6


def near_tie(moved):
    """The top two when the second is at least NEAR_TIE of the first."""
    if len(moved) >= 2 and moved[0].spread > 0 and moved[1].spread >= NEAR_TIE * moved[0].spread:
        return moved[:2]
    return None


def moved_together(opt):
    """(a, b) of correlated_pair, or None (the 0.8.0 callers' form)."""
    pair = correlated_pair(opt)
    return None if pair is None else pair[:2]


def correlated_pair(opt):
    """(a, b, r) — the two varying ingredients whose amounts across the
    fully measured formulations are most closely correlated, when |r| is at
    least TOGETHER, and that |r| (said with it, 03 §5.5) — or None (§4.3,
    §9 minor 16). The Fill to total row, a row a calculation works out and
    the process settings are left out, and so is a row whose amount never
    varied."""
    fill = opt.balance_row_name()
    span = {v['name']: float(v['bounds'][1]) - float(v['bounds'][0])
            for v in opt.varying_variables()
            if _is_ingredient(v) and v.get('type') == 'continuous'
            and v['name'] != fill and not opt.has_formula(v)}
    rows = [start_recipe(opt, i) for i in range(len(opt.Y_history)) if opt.fully_measured(i)]
    if len(rows) < 3:
        return None
    cols = {n: [float(r.get(n, 0.0) or 0.0) for r in rows] for n in span}
    cols = {n: v for n, v in cols.items() if max(v) - min(v) > max(VARIED * span[n], _EPS)}
    best = None
    for a, b in itertools.combinations(list(cols), 2):
        r = abs(statistics.correlation(cols[a], cols[b]))
        if r >= TOGETHER - 1e-12 and (best is None or r > best[0] + 1e-12):
            best = (r, a, b)
    return None if best is None else (best[1], best[2], best[0])


# Review story 3: two rows whose tested amounts moved together this closely
# (|Pearson r|) are noted under the profiler: no model can separate them.
MOVED_R = 0.7


def moved_pairs(opt, threshold=MOVED_R):
    """[(a, b, r)]: every two varying continuous rows (the profiler's
    choices, ingredients and process settings; not a calculated row) whose
    amounts across the fully measured formulations have |r| >= `threshold`
    (to two decimals), strongest first; r signed. A row that never varied is left out."""
    names = [n for n in path_choices(opt) if not opt.has_formula(_var(opt, n))]
    rows = [start_recipe(opt, i) for i in range(len(opt.Y_history)) if opt.fully_measured(i)]
    if len(rows) < 3:
        return []
    span = {n: float(_var(opt, n)['bounds'][1]) - float(_var(opt, n)['bounds'][0])
            for n in names}
    cols = {n: [float(r.get(n, 0.0) or 0.0) for r in rows] for n in names}
    cols = {n: v for n, v in cols.items() if max(v) - min(v) > max(VARIED * span[n], _EPS)}
    out = []
    for a, b in itertools.combinations(list(cols), 2):
        r = statistics.correlation(cols[a], cols[b])
        # Compared as shown, to two decimals: an r the note would print as
        # 0.70 is noted, so the number and the rule never disagree.
        if round(abs(r), 2) >= threshold - 1e-12:
            out.append((a, b, r))
    return sorted(out, key=lambda t: -abs(t[2]))


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


@dataclass
class PredictedBest:
    recipe: dict
    score: Prediction
    close: int | None       # index of the nearest formulation when close, else None
    compare: int            # index of the table's second row
    factor: float = 1.0     # how much the likely ranges here were widened
    repeat: Prediction | None = None   # Best so far made again, predicted


def column_factor(opt, checks):
    """Make a round's Overall-score widening (review I1, A M-3): score_widen
    of every check except the open Round's own, so the column for its
    rows still to be made is widened as that Round's check will be, by the
    Rounds before it only."""
    return score_widen([c for c in checks if c.round != opt.pending_batch_no])


def score_widen(checks):
    """How much to widen a likely range of the Overall score picked for
    being highest: the root-mean-square of the project's own standardised
    Overall-score misses (fully measured formulations), never below 1; 1
    with fewer than SCORE_RECORD_MIN of them (0.8.0 review fixes §3.6: the
    predicted best's range held 82 % of the time, 92 % widened so)."""
    z = [(c.measured - c.raw_mean) / c.sd for c in checks
         if c.name == SCORE and c.fully_measured and c.sd and c.raw_mean is not None]
    if len(z) < SCORE_RECORD_MIN:
        return 1.0
    rms = math.sqrt(sum(v * v for v in z) / len(z))
    return max(1.0, rms) if math.isfinite(rms) else 1.0


def measure_widen(checks, name, skip_round=None):
    """Review critique (fix 10): a measurement's own calibration, as the
    Overall score's: the root-mean-square of its standardised Round-ahead
    misses (measured − mean) ÷ sd — the model's own sd, before any
    widening (round-2 fixes: the checks now carry their nested widening,
    nest_measure_widen) — never below 1; 1 with fewer than
    SCORE_RECORD_MIN. `skip_round`: the open Round's own checks left out."""
    z = [(c.measured - c.raw_mean) / c.sd for c in checks or ()
         if c.name == name and c.round != skip_round and c.sd and c.raw_mean is not None]
    if len(z) < SCORE_RECORD_MIN:
        return 1.0
    rms = math.sqrt(sum(v * v for v in z) / len(z))
    return max(1.0, rms) if math.isfinite(rms) else 1.0


IMPLIED_POINTS = 401     # normal quantiles per measurement for expected_score


def expected_score(opt, per):
    """Round-2 fixes (P1 #4): the Overall score a formulation is expected
    to get from its predicted measurements `per` ({name: Prediction}):
    E[U] over each measurement's forecast, Normal(model mean, sd), not the
    score of the means. The score is a sum over measurements, so each
    measurement's expectation is taken alone (IMPLIED_POINTS normal
    quantiles, the others at their means) and the gains added. A target's
    score falls either side of it, so E[U] sits below U(means) whenever a
    forecast is unsure. None when a measurement has no forecast."""
    names = [o['name'] for o in opt.objectives]
    if not names or any(per.get(n) is None for n in names):
        return None
    centre = lambda p: p.raw_mean if p.raw_mean is not None else p.mean  # noqa: E731
    means = {n: centre(per[n]) for n in names}
    base = opt._compute_utility(means)
    qs = [statistics.NormalDist().inv_cdf((k + 0.5) / IMPLIED_POINTS)
          for k in range(IMPLIED_POINTS)]
    total = base
    for n in names:
        sd = per[n].sd
        if not sd:
            continue
        mine = dict(means)
        acc = 0.0
        for q in qs:
            mine[n] = means[n] + q * sd
            acc += opt._compute_utility(mine)
        total += acc / IMPLIED_POINTS - base
    return total


def widened(fitted, p, factor):
    """`p` with its likely range widened by `factor` about the model's own
    mean, then clipped to the Range; the shown mean is unchanged."""
    if factor == 1.0 or p.sd is None or p.raw_mean is None:
        return p
    sd = p.sd * factor
    clip = lambda v: min(fitted.high, max(fitted.low, v))
    return Prediction(p.mean, clip(p.raw_mean - Z95 * sd), clip(p.raw_mean + Z95 * sd),
                      p.raw_mean, sd)


def no_amounts_reason(opt):
    """Answer 2's line when no amounts meet the project's limits together
    (the search's polytope is empty; ask() stops at the same place): a row
    calculated as = rest with no default batch size to fill — its floor
    then reads 0 - every other amount >= 0 — else the limits in general."""
    rest = opt._balance_row()
    if rest is not None and not opt.has_formulation_total():
        return wording.best_needs_batch_size(rest['name'])
    return wording.BEST_NO_AMOUNTS


def predicted_best(opt, fitted, checks=()):
    """The single formulation the Overall-score model predicts highest,
    found the way suggestions are made and never written anywhere. Its
    likely range, and Best so far's own predicted repeat, are widened by
    column_factor(opt, checks), as Make a round's column is.

    Returns a PredictedBest, or a one-line reason (str) when `ask` would
    refuse to make a Round or no amounts meet the limits together. Raises
    when the search fails or its answer breaks a limit; the caller shows
    one line."""
    reason = best_refusal(opt)
    if reason:
        return reason
    score = fitted[SCORE]
    bounds = opt._get_bounds()
    dim = bounds.shape[1]
    with opt._premixes_for_generation():
        with _isolated():
            try:
                candidate, _ = optimize_acqf(
                    acq_function=PosteriorMean(score.model),
                    bounds=torch.stack([torch.zeros(dim), torch.ones(dim)]).double(),
                    q=1, num_restarts=10, raw_samples=512,
                    inequality_constraints=opt._get_botorch_constraints(),
                    fixed_features=opt._get_fixed_features() or None,
                )
            except InfeasibilityError:
                return no_amounts_reason(opt)
        recipe = opt._decode(unnormalize(candidate[0], bounds).detach().numpy().flatten())
        if opt.has_formulation_total():
            recipe = opt._snapped_if_it_still_fits(recipe)
        recipe = opt._settings_on_their_step(recipe)
        # A categorical row holds a word, which the limit check would read
        # as an amount; no limit can name one.
        if not opt._check_constraints({k: v for k, v in recipe.items()
                                       if not isinstance(v, str)}):
            raise ValueError("the predicted best breaks a limit")
    # The factor Make a round's column uses, so the same make-up reads the
    # same range on both screens mid-Round (final review M-2).
    factor = column_factor(opt, checks)
    [prediction] = predict(opt, score, [opt._encode(recipe)])
    i, d = nearest(opt, recipe)
    close = i if i is not None and d <= CLOSE_TO_THE_BEST + 1e-9 else None
    best = opt.best_index()
    repeat = None
    if best is not None:
        [repeat] = predict(opt, score, [opt._encode(start_recipe(opt, best))])
        repeat = widened(score, repeat, factor)
    return PredictedBest(recipe, widened(score, prediction, factor), close,
                         close if close is not None else best, factor, repeat)


# ------------------------------------------------------------------ #
#  0.8.0 speed and simplicity, Plan B: where you stand, and which
#  ingredient changes the prediction most near what you have tested (spec
#  2026-09-30 §2.3, §2.5; review I7, I8). The yardstick is handed in: a
#  number, noise_yardstick(...).value, so nothing here needs the checks.
# ------------------------------------------------------------------ #

@dataclass
class NearChange:
    name: str
    near: float | None      # highest minus lowest mean over the tested-near stretch that
                            # holds the start; None when the start is not tested near
    whole: float            # the same over the whole path


@dataclass
class NearChanges:
    moved: list             # [NearChange], largest tested-near change first
    held: list              # names the limits keep in place from this start (see HELD)
    start_tested: bool      # the start itself is tested near


def _run_around(near, k):
    """The indices of the unbroken run of tested-near points holding point
    `k`, or [] when `k` itself is not tested near."""
    if not near[k]:
        return []
    lo = hi = k
    while lo > 0 and near[lo - 1]:
        lo -= 1
    while hi < len(near) - 1 and near[hi + 1]:
        hi += 1
    return list(range(lo, hi + 1))


def tested_near_changes(opt, fitted, start, name=SCORE):
    """How far each path choice moves the predicted `name` from `start`,
    near what has been tested (review I8): highest minus lowest mean over
    the stretch of its path, around the start, that tested_near marks; the
    whole path's change beside it. Largest tested-near change first (ties
    and choices with none keep path_choices order after them). Like
    spreads, a choice the limits box in is `held` and a path that cannot be
    drawn is left out; with no model for `name` there is nothing to say."""
    model = fitted.get(name)
    if model is None or model.model is None:
        return NearChanges([], [], False)
    moved, held, start_tested = [], [], False
    for choice in path_choices(opt):
        path = build_path(opt, choice, start)
        if path.start_breaks or not path.recipes:
            continue
        lo, hi = (float(b) for b in _var(opt, choice)['bounds'])
        if path.amounts[-1] - path.amounts[0] < HELD * (hi - lo):
            held.append(choice)
            continue
        means = [p.mean for p in predict(opt, model, [opt._encode(r) for r in path.recipes])]
        near = tested_near(opt, path)
        k = min(range(len(path.amounts)), key=lambda j: abs(path.amounts[j] - path.start_amount))
        run = _run_around(near, k)
        start_tested = start_tested or bool(run)
        moved.append(NearChange(
            choice, (max(means[j] for j in run) - min(means[j] for j in run)) if run else None,
            max(means) - min(means)))
    moved.sort(key=lambda c: (c.near is None, -(c.near or 0.0)))
    return NearChanges(moved, held, start_tested)


@dataclass
class Binding:
    kind: str               # "highest", "lowest" or "limit"
    name: str               # the row's name; for a limit, its metric or its rows
    text: str               # the row's name, or the limit as Set up words it


# A row or a limit within this share of its own span (a limit: of its bound)
# counts as reached (review I7).
BINDING = 0.01


def binding_limits(opt, recipe):
    """What the formulation `recipe` sits at (review I7, Line 3): each
    varying row within BINDING of its span from its Lowest or Highest (the
    Fill to total row, fixed and calculated rows left out), then each
    ingredient limit and property limit within BINDING of its bound (the
    Formulation total itself and every Exactly limit left out: every
    formulation sits at it, and a fixed amount is not a limit to widen)."""
    out = []
    fill = opt.balance_row_name()
    for v in opt.varying_variables():
        if v.get('type') != 'continuous' or v['name'] == fill:
            continue
        lo, hi = (float(b) for b in v['bounds'])
        span = hi - lo
        if span <= 0:
            continue
        x = float(recipe.get(v['name'], lo) or 0.0)
        if x >= hi - BINDING * span:
            out.append(Binding("highest", v['name'], v['name']))
        elif x <= lo + BINDING * span:
            out.append(Binding("lowest", v['name'], v['name']))
    for qc in getattr(opt, 'quantity_constraints', []):
        if qc.get('source') == 'formulation_total' or qc.get('exactly') is not None:
            continue
        total = sum(float(recipe.get(n, 0.0) or 0.0) for n in qc['ingredients'])
        if any(b is not None and abs(total - float(b)) <= BINDING * max(abs(float(b)), _EPS)
               for b in (qc['min'], qc['max'])):
            out.append(Binding("limit", " + ".join(qc['ingredients']), opt.limit_text(qc)))
    amount = max(opt.ingredient_total(recipe), _EPS)
    numeric = _numeric(recipe)
    for c in opt.constraints:
        if opt.ingredients_without_property(c['metric']):
            continue
        if any(b is not None and abs(opt._property_residual(numeric, c['metric'], b))
               <= BINDING * max(abs(float(b)), _EPS) * amount
               for b in (c['min'], c['max'])):
            out.append(Binding("limit", c['metric'], opt.property_limit_text(c)))
    return out


# ------------------------------------------------------------------ #
#  Insights for a first-time user (spec 2026-10-01 rev. 2), W3: "Where you
#  have tested", model-free (§8.1, review R-I15). T0 stubs with their final
#  signatures; W3 fills them in. The words are wording's (strips_caption,
#  gap_*, strip_gaps_line, limit_hit_line, STRIP_*).
# ------------------------------------------------------------------ #

@dataclass
class StripTick:
    value: float            # at the Formulation total (start_recipe), the row's own unit
    formulation: int
    round: int | None
    score: float | None     # Overall score; None while waiting or Not scored
    state: str              # "tested", "waiting", "not_scored" or "suggested"
    latest: bool = False    # from the latest Round (highlighted)


@dataclass
class Strip:
    name: str
    unit: str
    low: float              # Lowest–Highest; the Fill-to-total row: 0–total
    high: float
    reach: tuple | None     # (lowest, highest) over bo_views.feasible_pool, else (low, high)
    ticks: list             # [StripTick]
    fill: bool = False      # the Fill-to-total row: drawn, never given a gap line
    setting: bool = False   # a process setting (its own unit)
    options: dict | None = None   # a choice row: {option: times tested}


@dataclass
class StripGap:
    name: str
    kind: str               # "above", "below", "between" or "option"
    low: float | None       # the stretch's ends, in the row's unit
    high: float | None
    size: float             # its length as a share of Highest − Lowest
    option: str | None = None


@dataclass
class LimitHit:
    round: int
    hits: int               # picks at this limit (binding_limits, BINDING)
    picks: int              # generated picks counted (own formulations and Repeats left out)
    binding: Binding


def _strip_rows(opt):
    """The rows a strip is drawn for, ingredients first, then process
    settings, each group in project order: every varying row (continuous or
    a choice) and the Fill to total row (review R-I15: drawn, never given a
    gap line). Calculated and fixed rows have nothing to show."""
    fill = opt.balance_row_name()
    rows = [v for v in opt.variables
            if v['name'] == fill or (not opt.has_formula(v) and not opt.is_fixed(v))]
    return ([v for v in rows if _is_ingredient(v)]
            + [v for v in rows if not _is_ingredient(v)])


def _strip_reach(opt, pool):
    """{row: (lowest, highest)} over the allowed pool (bo_views.feasible_pool,
    normalised), decoded as ask() decodes a point, so the Fill to total row
    reads what is left of the total; {} without a pool."""
    if pool is None or pool.shape[0] == 0:
        return {}
    numeric = [v['name'] for v in _strip_rows(opt) if v.get('type') == 'continuous']
    low, high = {}, {}
    for point in unnormalize(pool, opt._get_bounds()):
        recipe = opt._decode(point.numpy())
        for name in numeric:
            x = recipe.get(name)
            if x is None:
                continue
            x = float(x)
            low[name] = min(low.get(name, x), x)
            high[name] = max(high.get(name, x), x)
    return {name: (low[name], high[name]) for name in low}


def _strip_ticks(opt):
    """[(recipe at the total, StripTick)]: every recorded formulation —
    with all results in ("tested"), waiting ("waiting") or moved to Not
    scored ("not_scored") — every Not scored suggestion that kept its
    make-up, and the open Round's rows still to make ("suggested")."""
    from food_bo import ROUND_FIELD
    rounds = [r for r in opt.batch_history if r is not None]
    latest = max(rounds) if rounds else None
    out = []
    for i in range(len(opt.Y_history)):
        r = opt.batch_history[i] if i < len(opt.batch_history) else None
        if opt.fully_measured(i):
            state, score = "tested", float(opt.Y_history[i])
        else:
            state, score = ("not_scored" if opt.is_set_aside(i) else "waiting"), None
        out.append((start_recipe(opt, i),
                    StripTick(0.0, int(opt.formulation_ids[i]), r, score, state,
                              latest is not None and r == latest)))
    for row in getattr(opt, "skipped", []) or []:
        if row.get('recipe'):
            out.append((at_total(opt, row['recipe']),
                        StripTick(0.0, int(row['formulation']), row.get(ROUND_FIELD), None,
                                  "not_scored")))
    recorded = {int(n) for n in opt.formulation_ids}
    recorded |= {int(row['formulation']) for row in getattr(opt, "skipped", []) or []}
    for row in opt.pending_batch or []:
        if int(row['formulation']) not in recorded:
            out.append((at_total(opt, row['recipe']),
                        StripTick(0.0, int(row['formulation']), opt.pending_batch_no, None,
                                  "suggested")))
    return out


def tested_strips(opt, pool=None):
    """W3: one Strip per varying ingredient and process setting, and the
    Fill to total row (review R-I15), model-free. Each tick is a formulation
    at the Formulation total (start_recipe, at_total: the make-up the models
    learn), in the row's own unit. The strip spans the row's Lowest–Highest;
    its reachable part is the lowest to highest of the row over `pool`
    (bo_views.feasible_pool(opt), normalised), else Lowest–Highest. A choice
    row counts the formulations with all results in per option."""
    reach = _strip_reach(opt, pool)
    made = _strip_ticks(opt)
    fill = opt.balance_row_name()
    out = []
    for v in _strip_rows(opt):
        name = v['name']
        setting = not _is_ingredient(v)
        if v.get('type') != 'continuous':
            options = {o: 0 for o in v.get('options', [])}
            for recipe, tick in made:
                if tick.state == "tested" and recipe.get(name) in options:
                    options[recipe.get(name)] += 1
            out.append(Strip(name, "", 0.0, 0.0, None, [], False, setting, options))
            continue
        low, high = (float(b) for b in v['bounds'])
        if name == fill and opt.has_formulation_total() and high <= low:
            low, high = 0.0, float(opt.formulation_total)
        ticks = [dataclasses.replace(tick, value=float(recipe.get(name, 0.0) or 0.0))
                 for recipe, tick in made]
        out.append(Strip(name, opt.unit_of(name), low, high, reach.get(name, (low, high)),
                         ticks, name == fill, setting))
    return out


def _stretches(strip):
    """The untested stretches of a strip's reachable part (review R-I15):
    between consecutive tested values, or between a reachable end and the
    nearest tested value, each at least CLOSE_TO_THE_BEST of Highest −
    Lowest. A tested value outside the reachable part (made before a limit
    changed) counts at the nearer end. Nothing without a tested value."""
    span = strip.high - strip.low
    lo, hi = strip.reach if strip.reach is not None else (strip.low, strip.high)
    values = sorted(min(max(t.value, lo), hi) for t in strip.ticks if t.state == "tested")
    if span <= 0 or hi <= lo or not values:
        return []
    least = CLOSE_TO_THE_BEST * span - _EPS
    out = []
    if values[0] - lo >= least:
        out.append(StripGap(strip.name, "below", lo, values[0], (values[0] - lo) / span))
    for a, b in zip(values, values[1:]):
        if b - a >= least:
            out.append(StripGap(strip.name, "between", a, b, (b - a) / span))
    if hi - values[-1] >= least:
        out.append(StripGap(strip.name, "above", values[-1], hi, (hi - values[-1]) / span))
    return out


def gap_rows(strips):
    """The strips the gap rule reads: every amount or process setting row,
    the Fill to total row and choice rows left out."""
    return [s for s in strips if not s.fill and s.options is None]


def all_stretches(strips):
    """Every untested stretch (_stretches) of every gap_rows strip, in
    strip order, low to high within a strip. Model-free."""
    return [g for s in gap_rows(strips) for g in _stretches(s)]


def strip_gaps(opt, strips, k=2):
    """W3: the `k` largest untested stretches of the strips' reachable parts
    (_stretches: the 1-D rule of review R-I15, never a stretch the limits
    rule out), largest first, the Fill to total row left out; then up to
    `k` options of a choice row no formulation with all results in has
    used ("option"). Model-free."""
    stretches = sorted(all_stretches(strips), key=lambda g: -g.size)
    options = [StripGap(s.name, "option", None, None, 1.0, o)
               for s in strips if s.options is not None
               for o, n in s.options.items() if n == 0]
    return stretches[:k] + options[:k]


def _round_picks(opt, r):
    """The make-ups (at the total) of Round `r`'s generated picks, own
    formulations and Repeats left out: from its generation record's picks
    when it has them (bo_views._picks), else the rows without a note of
    their own, as bo_views._first_pick counts them (an edited row counts,
    at the make-up it was issued with)."""
    import bo_views     # bo_views imports this module
    from food_bo import ROUND_FIELD
    gen = bo_views.generation(opt, r)
    if gen.picks:
        numbers = list(gen.picks)
    elif r == opt.pending_batch_no and opt.pending_batch:
        own, numbers = bo_views._picks(opt, gen, list(opt.pending_batch))
    else:
        mine = lambda note: bool(note) and note not in (wording.EDITED_FORMULATION,
                                                       wording.NOT_SCORED)
        numbers = [int(n) for n, b, note in zip(opt.formulation_ids, opt.batch_history,
                                                 opt.notes_history)
                   if b == r and not mine(note)]
        numbers += [int(s['formulation']) for s in getattr(opt, "skipped", []) or []
                    if s.get(ROUND_FIELD) == r and not mine(s.get('note'))]
    recipes = [bo_views._issued(opt, n) for n in numbers]
    return gen, [x for x in recipes if x is not None]


def limit_hits(opt):
    """W3 (review R-I15): the limits at least half of a Round's generated
    picks sit at — binding_limits, BINDING of the span or bound — counted
    from 2 picks: the open Round's, else the last Round the model chose
    (warm); most hits first. [LimitHit], [] without such a Round."""
    import bo_views     # bo_views imports this module
    r = opt.pending_batch_no if opt.pending_batch else None
    if r is None:
        warm = [n for n, _ in round_order(opt) if bo_views.generation(opt, n).regime == "warm"]
        if not warm:
            return []
        r = warm[-1]
    gen, recipes = _round_picks(opt, r)
    if gen.regime != "warm" or len(recipes) < 2:
        return []
    counts, first = {}, {}
    for recipe in recipes:
        for b in binding_limits(opt, recipe):
            key = (b.kind, b.name)
            counts[key] = counts.get(key, 0) + 1
            first.setdefault(key, b)
    hits = [LimitHit(r, n, len(recipes), first[key]) for key, n in counts.items()
            if 2 * n >= len(recipes)]
    hits.sort(key=lambda h: -h.hits)
    return hits


@dataclass
class Standing:
    best: PredictedBest
    higher: bool            # the predicted best beats Best so far's predicted Repeat by
                            # more than the yardstick
    tested_near: bool       # a scored formulation lies within CLOSE_TO_THE_BEST of it
    close_no: int | None    # that formulation's number
    binding: list           # [Binding] at the predicted best
    luck: bool              # Best so far's measured score beats its own predicted Repeat
                            # by more than the yardstick


def best_standing(opt, best, yardstick, score_bias=None):
    """The predicted best against Best so far (review I7): like with like,
    its mean against Best so far's predicted Repeat (the measured score
    when there is none), judged by `yardstick` (points) — and, as shown,
    above Best so far's measured score by more than the yardstick too.
    `score_bias`: the Overall score's steady bias (score_steady_bias). When
    predictions run low by at least the gap between the measured score and
    its predicted Repeat, the gap is the model's, not chance, and there is
    no luck line (final review I-1)."""
    top = opt.best_index()
    measured_top = None if top is None else float(opt.Y_history[top])
    ref = best.repeat.mean if best.repeat is not None else measured_top
    # "Higher" must also hold on the numbers shown (answer-key read-through,
    # Timing check): the predicted best in whole points above Best so far's
    # measured score in two decimals by more than the yardstick, so neither
    # line 2 nor "What to do next" calls 92 higher than 92.50.
    shown = (measured_top is None
             or round(best.score.mean) - round(measured_top, 2) > yardstick)
    higher = ref is not None and best.score.mean - ref > yardstick and shown
    # Compared as shown too (review M1): the Repeat in whole points, the
    # measured score in two decimals, so "luck" never sits beside a Repeat
    # that reads higher than the score.
    luck = (best.repeat is not None and measured_top is not None
            and best.repeat.mean < measured_top - yardstick
            and round(best.repeat.mean) < round(measured_top, 2)
            and not (score_bias is not None and score_bias > 0
                     and score_bias >= measured_top - best.repeat.mean))
    close_no = None if best.close is None else int(opt.formulation_ids[best.close])
    return Standing(best, higher, best.close is not None, close_no,
                    binding_limits(opt, best.recipe), luck)


def above_best(low, mean, ref, measured_top, yardstick):
    """The triangle's "Predicted above Best so far" outline at one point
    (spec §3.3; T56 I1): its whole interval above Best so far's predicted
    Repeat (`low`, the model mean − 1.96 × model sd, above `ref`), and
    best_standing's own rule for the predicted score `mean` — above `ref`
    by more than the noise threshold, and so as shown against the measured
    Best so far (whole points against two decimals) — so the outline never
    marks what Progress calls "none"."""
    if None in (low, mean, ref):
        return False
    if not (low > ref and mean - ref > yardstick):
        return False
    return measured_top is None or round(mean) - round(measured_top, 2) > yardstick


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
# ceil(1/sqrt(f)) times finer, up to this many times, so it keeps roughly
# as many points as the whole triangle had or more: up to about four times
# as many, since the ceiling rounds a share just under 1 up to a step twice
# as fine (Task 2 review, Minor 2).
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


# ------------------------------------------------------------------ #
#  0.8.0 speed and simplicity, Plan A: Make a round's track-record line
#  (spec 2026-09-30 §3; review I16).
# ------------------------------------------------------------------ #

@dataclass
class RoundTrack:
    rough: bool             # fewer landed inside than 1 in 20 misses explains
    inside: int
    total: int
    gap: float | None       # its average miss when it is a bias (bias_gap); else None
    see_insights: bool      # the verdict (the measurements) is unreliable
    # BO views (review I1): the Overall score's skill when it is at or below
    # its no-skill cut (a warning, for E2 and E3), the verdict's Rounds, and whether
    # the coverage part is said (False: the skill part alone).
    no_skill: object = None
    rounds: list = dataclasses.field(default_factory=list)
    record: bool = True
    # Specific wording: the measurements' misses (value_misses) when
    # see_insights; None otherwise.
    misses: object = None


def round_track(opt, checks):
    """Make a round's line under its table, from the Overall score's own
    record over every Round checked (review I16; 0.8.0 e2e fixes, review
    I1): rough only when fewer
    landed inside than 1 in 20 misses explains (binomial tail under 0.10);
    the gap when it is a bias by Insights' own rule (bias_gap), so the two
    screens say one thing (answer-key read-through); holding otherwise. None — no line — without a record (fewer than
    SCORE_RECORD_MIN fully measured checks), when the verdict is early,
    wide ("Not judged yet" on Insights, review C I-5) or over-wide, and
    when the Overall score's own range is wide (final review I-1).
    BO views (review I1): a skill at or below its cut sets `no_skill` and
    is said even when coverage holds; where the coverage part would not be
    said (no record, over-wide, the Overall score wide) the line carries
    the skill alone (`record` False). Never while early or wide."""
    h = headline(opt, checks)
    record = h.score
    state = verdict_state(opt, h, checks)
    if state in ("none", "early", "wide"):
        return None
    fact = skill(opt, checks)
    no_skill = fact if fact is not None and fact.state == "no_skill" else None
    # Insights says the Overall score's own interval is wide (final review
    # I-1), or the intervals are wider than needed: no "12 of 12 inside"
    # beside that. A skill at or below its cut is still said (review I1).
    said = (record is not None and state != "over_wide" and SCORE not in (h.wide or []))
    if not said:
        if no_skill is None:
            return None
        return RoundTrack(False, 0, 0, None, state == "unreliable", no_skill,
                          list(h.rounds), record=False,
                          misses=value_misses(opt, checks, h) if state == "unreliable" else None)
    rough = binomial_tail(record.total - record.inside, record.total) < 0.10
    # The record's own Rounds, every Round checked (0.8.0 e2e fixes, I1).
    return RoundTrack(rough, record.inside, record.total, score_bias(opt, checks, h),
                      state == "unreliable", no_skill, list(record.rounds or h.rounds),
                      misses=value_misses(opt, checks, h) if state == "unreliable" else None)


# ------------------------------------------------------------------ #
#  0.8.0 BO views (spec 2026-09-30-bo-views §2; review B3, I1-I4): is the
#  model predicting? Round-ahead skill against the average of the rows
#  recorded before each Round, and a display-side guard that stops the tab
#  from stating noise as ingredient effects. Nothing here reaches ask().
# ------------------------------------------------------------------ #

# Skill is said from this many checks (review I2): below it, nothing.
SKILL_MIN = 10
# The no-skill cut c_n (fix round 1, ledger ruling on T2 I-1): the 10th
# percentile of a correct model's skill over windows of n checks whose true
# skill is 0.2 or more — a finite-data model whose Round-ahead errors match
# its own claimed uncertainty, on the 24 "good" simulated projects
# (tests/test_092_skill.py runs this gate and pins the table) — never above
# SKILL_CUT_MAX. At or below the cut: "no better than the average of earlier
# results" (the no-skill Next step, the guard's trigger (c), the track line);
# between the cut and SKILL_SMALL: "not clearly better than the average"
# (shown, no advice); SKILL_SMALL or more: "better than the average".
# Re-measured by ftue fix A on the I9 fixture (Round-grouped CV folds);
# it was {10: 0.00, 12: 0.03, 15: 0.11, 20: 0.25, 25: 0.32, 30: 0.45}.
# A correct model now reads at or below −0.2 in 4.1 % of windows at n = 10
# (6.9 % before) and below 0.2 in 12.5 % at n = 10, 11.0 % at n = 15
# (15.3 % and 12.3 % before).
SKILL_GATE_Q10 = {10: 0.11, 12: 0.11, 15: 0.14, 20: 0.22, 25: 0.31, 30: 0.34}
SKILL_CUT_MAX = -0.2
SKILL_SMALL = 0.2


def skill_cut(n):
    """c_n for a skill over `n` checks: the gate's 10th percentile at the
    largest tabulated n at or below it (the percentile rises with n, so
    this errs towards fewer no-skill warnings), capped at SKILL_CUT_MAX.
    As measured, every tabulated percentile is above −0.2, so the cut is
    −0.2 at every n (final review M-5): the table moves it only if a later
    gate measures a percentile below −0.2."""
    at = max((k for k in SKILL_GATE_Q10 if k <= n), default=min(SKILL_GATE_Q10))
    return min(SKILL_GATE_Q10[at], SKILL_CUT_MAX)


@dataclass
class SkillFact:
    value: float
    n: int              # checks pooled
    rounds: list        # the Rounds they came from, in order
    state: str          # "no_skill", "unclear" or "skill"
    cut: float = SKILL_CUT_MAX   # skill_cut(n), the no-skill cut it was judged by


def _skill_state(value, n=SKILL_MIN):
    return ("no_skill" if value <= skill_cut(n) else "unclear" if value < SKILL_SMALL
            else "skill")


def skill(opt, checks, name=SCORE):
    """Round-ahead skill of `name` over every Round checked (review B3):
    1 − Σ(measured − predicted)² ÷ Σ(measured − baseline)², the squared
    errors pooled over the checks (review I2), the baseline of each check
    the mean of `name` over the rows recorded before its Round (the rows
    its model learned from). For the Overall score only formulations with
    all their results count. None with fewer than SKILL_MIN checks or when
    every check sits exactly on its baseline (nothing to divide by)."""
    mine = [c for c in checks
            if c.name == name and (name != SCORE or c.fully_measured)]
    if len(mine) < SKILL_MIN:
        return None
    first = dict(round_order(opt))
    base = {}
    model = mean = 0.0
    pooled = []
    for c in mine:
        if c.round not in base:
            seen = [measured(opt, name, i) for i in range(first.get(c.round, 0))]
            seen = [v for v in seen if v is not None]
            base[c.round] = sum(seen) / len(seen) if seen else None
        if base[c.round] is None:
            continue
        pooled.append(c)
        model += (c.measured - c.predicted) ** 2
        mean += (c.measured - base[c.round]) ** 2
    # n is the checks actually pooled (review T2 m-3).
    if len(pooled) < SKILL_MIN or mean <= 0:
        return None
    value = 1.0 - model / mean
    n = len(pooled)
    return SkillFact(value, n, _ordered_rounds(pooled), _skill_state(value, n), skill_cut(n))


def allowed_misses(n):
    """The most misses out of `n` values the verdict still calls As
    expected (03 §8): the largest m with binomial_tail(m, n) ≥ 0.10."""
    m = 0
    while m < n and binomial_tail(m + 1, n) >= 0.10:
        m += 1
    return m


# The guard's triggers (review I3): the correlation length within this
# factor of its lower limit; the fitted noise below this share of the
# Repeat SD (from YARDSTICK_DF degrees of freedom).
GUARD_FLOOR = 1.05
GUARD_NOISE = 0.5
# A kernel whose own lower limit is 0 (an expert's Matern or RBF, gpytorch's
# Positive()) is judged against this floor on the normalised inputs instead
# (fix round 1, ledger ruling on T2 I-2: about BoTorch's default 0.025 × 2).
EXPERT_FLOOR = 0.05


@dataclass
class SkillGate:
    """Why the Overall score's Round-ahead record withholds ingredient-effect
    claims (0.8.0 e2e fixes A3, review I4): "wait" (fewer than SKILL_MIN
    checks, `n` so far), "unclear" (skill under SKILL_SMALL) or "no_skill"
    (at or below the cut)."""
    state: str
    n: int
    fact: SkillFact | None = None


def skill_gate(opt, checks, fact=None):
    """The A3 rule (review I4): effect claims — the ranking, near-ties,
    Compare all's bars, the expected improvement, the layers — need the
    Overall score's skill at SKILL_SMALL or more, over SKILL_MIN checks or
    more. None when they may be made; else why not. `fact`: skill() when
    the caller has it. Kept apart from the no-skill cut: Next step's
    no_skill row and the track line still use the cut alone."""
    fact = skill(opt, checks) if fact is None else fact
    if fact is None:
        n = sum(1 for c in checks if c.name == SCORE and c.fully_measured)
        # SKILL_MIN or more checks with no skill: every check sat on its
        # baseline, nothing to divide by. Not "waiting" for checks it has.
        return SkillGate("wait", n) if n < SKILL_MIN else None
    if fact.state in ("no_skill", "unclear"):
        return SkillGate(fact.state, fact.n, fact)
    return None


@dataclass
class Guard:
    reasons: list       # "floor", "noise", "skill", in that order
    # "skill": the A3 gate behind it (0.8.0 e2e fixes, review I4): no skill,
    # skill not clearly better than the average, or too few checks yet.
    gate: SkillGate | None = None

    @property
    def fires(self):
        return bool(self.reasons)

    @property
    def skill_only(self):
        """Trigger (c) alone: said as "not reported until…", never as
        "fitting noise" (review I3)."""
        return self.reasons == ["skill"]


def _lengthscale_at_floor(gp):
    """Whether any kernel's smallest correlation length (its actual one,
    after the constraint's transform) sits within GUARD_FLOOR of that
    kernel's own lower limit (review I3: read from the module), or of
    EXPERT_FLOOR when that limit is 0 (fix round 1, T2 I-2)."""
    for module in gp.covar_module.modules():
        constraint = getattr(module, "raw_lengthscale_constraint", None)
        if constraint is None or not hasattr(module, "lengthscale"):
            continue
        low = float(torch.as_tensor(constraint.lower_bound).min())
        if low <= 0:
            low = EXPERT_FLOOR
        if float(module.lengthscale.detach().min()) <= GUARD_FLOOR * low:
            return True
    return False


def fitted_noise_sd(gp):
    """The display model's fitted noise as an sd in the output's own units
    (review I3): √noise × the training Y's sd (Standardize(m=1)). None when
    the noise is not fitted (Custom model settings, fixed_tiny)."""
    likelihood = getattr(gp, "likelihood", None)
    noise = getattr(likelihood, "noise", None)
    if noise is None or likelihood.__class__.__name__.startswith("Fixed"):
        return None
    tf = getattr(gp, "outcome_transform", None)
    scale = float(tf.stdvs.reshape(-1)[0]) if tf is not None and hasattr(tf, "stdvs") else 1.0
    return math.sqrt(max(0.0, float(noise.detach().reshape(-1)[0]))) * scale


def noise_guard(opt, fitted, checks=(), name=SCORE, fact=None):
    """Whether the display model of `name` is likely fitting noise (review
    I3): (a) its correlation length at its lower limit; (b) its fitted
    noise sd below GUARD_NOISE × the Repeat SD of `name`, from
    YARDSTICK_DF df or more, never under fixed_tiny; (c) for the
    Overall score only, the A3 skill gate (0.8.0 e2e fixes, review I4):
    fewer than SKILL_MIN checks, or skill under SKILL_SMALL — said as
    "not reported until…", never as "fitting noise" (`fact`: skill() when
    the caller has it). `fitted`: insights.Fitted, or None."""
    reasons = []
    gp = getattr(fitted, "model", None)
    if gp is not None:
        try:
            if _lengthscale_at_floor(gp):
                reasons.append("floor")
        except Exception:
            pass
        tiny = (getattr(opt, "bo_config", None) or {}).get("noise") == "fixed_tiny"
        sd = None if tiny else fitted_noise_sd(gp)
        # The yardstick only when there is a fitted noise to judge (T2 m-5).
        yard = None if sd is None else noise_yardstick(opt, checks, name)
        if (yard is not None and yard.sd and yard.df >= YARDSTICK_DF
                and sd < GUARD_NOISE * yard.sd):
            reasons.append("noise")
    gate = None
    if name == SCORE:
        gate = skill_gate(opt, checks, fact)
        if gate is not None:
            reasons.append("skill")
    return Guard(reasons, gate)


# ------------------------------------------------------------------ #
#  Insights for a first-time user (spec 2026-10-01 rev. 2), W2: the facts
#  "Model and method" and "How each status is decided" are written from
#  (§7, review R-I12, R-I13). T0 stubs with their final signatures; W2
#  fills them in. The words are wording's (model_summary, status_rule_rows).
# ------------------------------------------------------------------ #

@dataclass
class ModelFacts:
    """Read off a model actually built by FoodOptimizer._build_gp on this
    project's own xy(opt, SCORE, rows), never from hard-coded defaults
    (review R-I13), plus the settings that are not in bo_config.
    wording.model_summary(facts) writes the sentences."""
    cold: bool                       # fewer than COLD_START_RUNS scored: the model is to come
    scored: int                      # formulations with all results in
    cold_start: int                  # COLD_START_RUNS
    values: bool                     # process settings alone
    total: bool                      # the project has a Formulation total
    recommended: bool = True         # bo_config is None
    inputs: int = 0                  # model columns
    kernel: str = "rbf"              # a wording.KERNEL_NAMES key, else the class name
    ard: bool = True
    lengthscale_prior: str | None = None    # "LogNormal(1.89, 1.73)"
    lengthscale_floor: float | None = None
    outputscale: bool = False        # a ScaleKernel wraps the kernel
    outputscale_prior: str | None = None
    noise: str = "fitted"            # "fitted" or "fixed"
    noise_prior: str | None = None
    noise_floor: float | None = None
    standardised: bool = True        # Standardize(m=1)
    acquisition: str = "qlognei"     # bo_views.acquisition(opt)
    beta: float | None = None        # qUCB's beta
    q: int | None = None             # the open or last Round's number generated
    restarts: int = 0                # bo_views.ASK_RESTARTS
    raw_samples: int = 0             # bo_views.ASK_RAW_SAMPLES
    mc_samples: int = 0              # the QMC sample count
    folds: int = FOLDS
    fit_tolerance: float = FIT_OPTIONS["ftol"]
    robust: bool = False             # opt.robust: Input Warping in ask() only


# Read off the built objects (review R-I13): the kernel's class, and its
# smoothness for a Matérn or its degree for a polynomial, as the
# wording.KERNEL_NAMES key the expert box would choose; another class is
# said by its own name.
_KERNEL_KEYS = {"RBFKernel": "rbf", "LinearKernel": "linear"}
_ACQUISITION_KEYS = {"qLogNoisyExpectedImprovement": "qlognei",
                     "qLogExpectedImprovement": "qlogei", "qUpperConfidenceBound": "qucb"}


def _number(value):
    return float(torch.as_tensor(value).reshape(-1)[0])


def _prior_text(prior):
    """'LogNormal(2.39, 1.73)', 'Gamma(3.00, 6.00)': a prior as its
    parameters are set on the object; None without one."""
    if prior is None:
        return None
    kind = type(prior).__name__.removesuffix("Prior")
    # The distribution's own parameters, in its own order (torch's
    # arg_constraints): LogNormal's location and its log-sd, Gamma's
    # concentration and rate.
    names = list(getattr(prior, "arg_constraints", None) or ())
    if len(names) != 2:
        return kind
    return "{}({}, {})".format(kind, *(f"{_number(getattr(prior, k)):.2f}".replace("-", "−")
                                       for k in names))


def _floor(module, name):
    """A positive lower limit of a constrained hyperparameter, else None
    (gpytorch's Positive() has 0)."""
    constraint = getattr(module, f"raw_{name}_constraint", None)
    low = getattr(constraint, "lower_bound", None)
    if low is None:
        return None
    low = _number(low)
    return low if low > 0 else None


def _kernel_facts(covar):
    """(kernel key, ARD, lengthscale prior, its floor, outputscale, its
    prior) of a built covar_module."""
    scaled = type(covar).__name__ == "ScaleKernel"
    base = covar.base_kernel if scaled else covar
    cls = type(base).__name__
    if cls == "MaternKernel":
        nu = float(base.nu)
        key = {2.5: "matern52", 1.5: "matern32"}.get(nu, f"Matérn {nu:g}")
    elif cls == "PolynomialKernel":
        key = "poly2" if int(base.power) == 2 else f"{cls} {int(base.power)}"
    else:
        key = _KERNEL_KEYS.get(cls, cls)
    lengths = bool(getattr(base, "has_lengthscale", False))
    ard = lengths and (getattr(base, "ard_num_dims", None) or 0) > 1
    return (key, ard,
            _prior_text(getattr(base, "lengthscale_prior", None)) if lengths else None,
            _floor(base, "lengthscale") if lengths else None,
            scaled, _prior_text(getattr(covar, "outputscale_prior", None)) if scaled else None)


def _generated(opt):
    """q: the number the open Round's generation record asked for, else the
    last Round's; None without a record."""
    records = opt.generation_records()
    record = records.get(opt.pending_batch_no) if opt.pending_batch_no is not None else None
    if record is None and records:
        record = records[max(records)]
    try:
        return None if record is None else int(record["n"])
    except (KeyError, TypeError, ValueError):
        return None


def model_summary(opt):
    """W2 (review R-I13): ModelFacts for this project, read off the
    Overall-score model FoodOptimizer._build_gp builds on the project's own
    xy(opt, SCORE, rows) and the acquisition _build_acqf builds on it
    (milliseconds, no fit, inside _isolated so the global RNG never moves),
    plus the settings that are not in bo_config. Never calls ask(). Below
    COLD_START_RUNS scored there is no model yet: the cold facts alone."""
    import bo_views   # bo_views imports this module
    from food_bo import COLD_START_RUNS
    scored = opt.complete_count()
    facts = ModelFacts(cold=scored < COLD_START_RUNS, scored=scored,
                       cold_start=COLD_START_RUNS, values=not opt.has_ingredients(),
                       total=opt.has_formulation_total(),
                       recommended=not getattr(opt, "bo_config", None),
                       restarts=bo_views.ASK_RESTARTS, raw_samples=bo_views.ASK_RAW_SAMPLES,
                       robust=bool(getattr(opt, "robust", False)), q=_generated(opt))
    if facts.cold:
        return facts
    rows = [i for i in range(len(opt.Y_history)) if opt.fully_measured(i)]
    X, Y = xy(opt, SCORE, rows)
    import warnings
    with _isolated(), warnings.catch_warnings():
        # An unfitted model's baseline posterior may need jitter: said once
        # per build on the console, and nothing to the reader.
        warnings.simplefilter("ignore")
        gp = opt._build_gp(X, Y, X.shape[-1], None)
        acqf = opt._build_acqf(gp, X, Y)
    facts.inputs = int(X.shape[-1])
    (facts.kernel, facts.ard, facts.lengthscale_prior, facts.lengthscale_floor,
     facts.outputscale, facts.outputscale_prior) = _kernel_facts(gp.covar_module)
    likelihood = gp.likelihood
    if type(likelihood).__name__.startswith("Fixed"):
        facts.noise = "fixed"
    else:
        noise = likelihood.noise_covar
        facts.noise = "fitted"
        facts.noise_prior = _prior_text(getattr(noise, "noise_prior", None))
        facts.noise_floor = _floor(noise, "noise")
    facts.standardised = type(getattr(gp, "outcome_transform", None)).__name__ == "Standardize"
    name = type(acqf).__name__
    facts.acquisition = _ACQUISITION_KEYS.get(name, name)
    if facts.acquisition == "qucb":
        # qUpperConfidenceBound keeps √(β π / 2) only.
        facts.beta = round(2.0 * float(acqf.beta_prime) ** 2 / math.pi, 6)
    sampler = getattr(acqf, "sampler", None)
    facts.mc_samples = int(sampler.sample_shape[0]) if sampler is not None else 0
    return facts


# "How each status is decided" (review I13, R-I13): one key per row of
# wording.status_rule_rows, in its order: R-B1's Predictions rows worst
# first, the three clauses, the Progress words, then next_step's rows in
# their first-match order (NEXT_ROWS).
NEXT_ROWS = ("repeat", "misses", "early", "no_skill", "untested", "explore", "close",
             "rising", "see")
RULE_KEYS = (*(("predictions", k) for k in PREDICTION_ROWS),
             ("clause", "wide"), ("clause", "bias"), ("clause", "guard"),
             *(("progress", k) for k in ("rising", "within_noise", "flat", "no_new_best",
                                         "too_early")),
             *(("next", k) for k in NEXT_ROWS))


def _rule_rows(opt, checks=None):
    """[(key, (line, status, exact rule))], every row, this project's cut
    (skill_cut at its checks' n) and form."""
    fact = skill(opt, checks) if checks else None
    cut = fact.cut if fact is not None else skill_cut(SKILL_MIN)
    rows = wording.status_rule_rows(wording.signed(cut, 2).lstrip("+"),
                                    settings_only=not opt.has_ingredients())
    out = list(zip(RULE_KEYS, rows))
    # Round-2 fixes (P3 #17): Rising's rule in the screen's own words and
    # number, and rows for Plateau and the stop at the top score.
    need = _rise_need_text(opt)
    fixed = []
    for key, (line, status, rule) in out:
        if key == ("progress", "rising"):
            rule = wording.method_rising(need)
        elif key == ("progress", "within_noise"):
            rule = wording.METHOD_WITHIN_NOISE
        fixed.append((key, (line, status, rule)))
        if key == ("progress", "flat"):
            fixed.append((("progress", "plateau"),
                          (line, wording.METHOD_PLATEAU_WORD, wording.PLATEAU_RULE)))
    fixed.append((("next", "stop"), (wording.STATUS_NEXT, wording.CARD_CEILING,
                                     wording.METHOD_STOP)))
    return fixed


def _rise_need_text(opt):
    """'3.0': the rise the latest scored Round needed to read Rising
    (rise_threshold of the noise yardstick and that Round's size); None
    with no Round scored."""
    try:
        pool = opt.best_candidates()
        rounds = [opt.batch_history[i] for i in pool if i < len(opt.batch_history)
                  and opt.batch_history[i] is not None]
        if not rounds:
            return None
        q = sum(1 for r in rounds if r == max(rounds))
        return f"{rise_threshold(noise_yardstick(opt), max(1, q)):.1f}"
    except Exception:
        return None


def status_rule(opt, key, checks=None):
    """The exact rule of one RULE_KEYS row, e.g. ("next", "explore")."""
    return dict(_rule_rows(opt, checks))[key][2]


def status_label(opt, key, checks=None):
    """The status column of one RULE_KEYS row: 'Not yet (gate)', 'Explore'."""
    return dict(_rule_rows(opt, checks))[key][1]


def status_rules(opt, checks=None):
    """W2 (review I13, R-I13): "How each status is decided", one row per
    state of review B1's table and per Next row, in first-match order:
    [(line, status, exact rule)], from wording.status_rule_rows with this
    project's cut and form. At the gate (M9) only what applies: the gate
    row and the Progress rows; no Next row is tried before it."""
    rows = _rule_rows(opt, checks)
    if opt.complete_count() < MIN_ROWS:
        rows = [(k, r) for k, r in rows
                if k == ("predictions", "gate") or k[0] == "progress"]
    return [r for _, r in rows]


# ------------------------------------------------------------------ #
#  Charts proto (2026-10-01): "What you've tried" (parallel coordinates)
#  and "How the app chooses" (the path through Best so far). Model-free
#  data here; the charts are ui_insights'.
# ------------------------------------------------------------------ #

@dataclass
class TriedAxis:
    name: str
    unit: str
    low: float              # Lowest (bottom of the axis)
    high: float             # Highest (top)
    setting: bool = False
    options: list | None = None   # a choice row: its options, spaced evenly


@dataclass
class TriedLine:
    formulation: int
    round: int | None
    state: str              # "tested", "waiting", "not_scored" or "suggested"
    score: float | None
    best: bool
    values: dict            # {axis name: amount, setting or option}


def tried_axes(opt):
    """One axis per varying ingredient and process setting (continuous or a
    choice), ingredients first, in project order; the Fill to total row is
    left out: it follows the others."""
    fill = opt.balance_row_name()
    out = []
    for v in _strip_rows(opt):
        if v['name'] == fill:
            continue
        setting = not _is_ingredient(v)
        if v.get('type') != 'continuous':
            out.append(TriedAxis(v['name'], "", 0.0, 1.0, setting, list(v.get('options', []))))
            continue
        low, high = (float(b) for b in v['bounds'])
        out.append(TriedAxis(v['name'], opt.unit_of(v['name']), low, high, setting))
    return out


def tried_lines(opt):
    """One TriedLine per recorded formulation (tested, waiting, Not scored)
    and per suggestion of the open Round still to make, at the Formulation
    total (the make-up the models learn), as the strips take them."""
    best = opt.best_index()
    best_no = None if best is None else int(opt.formulation_ids[best])
    names = [a.name for a in tried_axes(opt)]
    out = []
    for recipe, tick in _strip_ticks(opt):
        values = {}
        for n in names:
            x = recipe.get(n)
            values[n] = x if isinstance(x, str) else float(x or 0.0)
        out.append(TriedLine(tick.formulation, tick.round, tick.state, tick.score,
                             tick.state == "tested" and tick.formulation == best_no, values))
    return out


def axis_share(axis, value):
    """Where `value` sits on `axis`, 0 at its Lowest, 1 at its Highest; a
    choice's options spaced evenly (one option: the middle). A value
    recorded outside Lowest–Highest is drawn outside 0–1."""
    if axis.options is not None:
        if value not in axis.options:
            return None
        n = len(axis.options)
        return 0.5 if n == 1 else axis.options.index(value) / (n - 1)
    span = axis.high - axis.low
    return 0.5 if span <= 0 else (float(value) - axis.low) / span


def tested_near_path(opt, path):
    """The fully measured formulations within CLOSE_TO_THE_BEST of some
    point of `path` (tested_near's rule, per formulation): [(index, its
    amount of the path's row at the total, its Overall score)]."""
    out = []
    for i in range(len(opt.Y_history)):
        if not opt.fully_measured(i):
            continue
        t = start_recipe(opt, i)
        ds = [distance(opt, r, t) for r in path.recipes]
        if any(d is not None and d <= CLOSE_TO_THE_BEST + _EPS for d in ds):
            out.append((i, float(t.get(path.name, 0.0) or 0.0), float(opt.Y_history[i])))
    return out


def near_path(opt, path, recipe):
    """`recipe`'s amount of the path's row when it lies within
    CLOSE_TO_THE_BEST of the path, else None."""
    ds = [distance(opt, r, recipe) for r in path.recipes]
    if any(d is not None and d <= CLOSE_TO_THE_BEST + _EPS for d in ds):
        return float(recipe.get(path.name, 0.0) or 0.0)
    return None


def most_changed(opt, start, recipes):
    """The path choice `recipes` move furthest from `start`, as a share of
    its Lowest–Highest: the largest move over the recipes, ties (two rows
    taken to a limit) to the larger total move; None without one."""
    best, best_key = None, None
    for n in path_choices(opt):
        v = _var(opt, n)
        lo, hi = (float(b) for b in v['bounds'])
        if hi <= lo or not recipes:
            continue
        moves = [abs(float(r.get(n, 0.0) or 0.0) - float(start.get(n, 0.0) or 0.0)) / (hi - lo)
                 for r in recipes]
        key = (round(max(moves), 9), round(sum(moves), 9))
        if best_key is None or key > best_key:
            best, best_key = n, key
    return best


# Charts proto follow-up: the straight line in the full space from Best so
# far (t = 0) to a suggestion (t = 1), every row moving together.
LINE_FROM, LINE_TO, LINE_POINTS = -0.25, 1.5, 71


def _line_recipe(opt, start, end, t):
    """The make-up a share `t` of the way from `start` to `end`: each
    varying continuous row in proportion, a choice row the start's until
    halfway, the end's after; the formula rows (the Fill to total row
    among them) written in again."""
    r = dict(start)
    for v in opt.varying_variables():
        n = v['name']
        if v.get('type') != 'continuous':
            r[n] = start.get(n) if t < 0.5 else end.get(n, start.get(n))
            continue
        a, b = float(start.get(n, 0.0) or 0.0), float(end.get(n, 0.0) or 0.0)
        r[n] = a + t * (b - a)
    return opt.fill_formulas(r)


def _line_moving(opt):
    return [v['name'] for v in opt.varying_variables() if v.get('type') == 'continuous']


def _line_breaks(opt, start, end, t):
    """The first limit the line breaks at `t` (its own Lowest–Highest per
    row, the total, every limit, a formula row below zero), or None."""
    return _limits_broken(opt, _line_recipe(opt, start, end, t), _line_moving(opt), start)


def line_path(opt, start, end, lo=LINE_FROM, hi=LINE_TO, points=LINE_POINTS):
    """Path-like (amounts = t, recipes) along the straight line from
    `start` (t = 0) to `end` (t = 1) over [lo, hi], clipped where it leaves
    the allowed region as build_path clips one row's path (bisected edge,
    the limit it breaks named in low_end/high_end). Both ends in the
    region keep 0..1 whole: the region is convex."""
    grid = sorted({lo + (hi - lo) * k / (points - 1) for k in range(points)} | {0.0, 1.0})
    base = dict(name="line", unit="", mode="line", fill_name=opt.balance_row_name(),
                start_amount=0.0, skipped=[], outside=False, fixed=[], follows=[])
    why = _line_breaks(opt, start, end, 0.0)
    if why is not None:
        return Path(amounts=[], recipes=[], low_end=PathEnd(0.0, None),
                    high_end=PathEnd(0.0, None), start_breaks=why, **base)
    ends = {}
    for side, seq in (("low", [t for t in reversed(grid) if t <= 0.0]),
                      ("high", [t for t in grid if t >= 0.0])):
        last = 0.0
        ends[side] = PathEnd(seq[-1], None)
        for t in seq:
            reason = _line_breaks(opt, start, end, t)
            if reason is None:
                last = t
                continue
            ok, bad = last, t
            for _ in range(40):
                mid = (ok + bad) / 2
                if _line_breaks(opt, start, end, mid) is None:
                    ok = mid
                else:
                    bad = mid
            ends[side] = PathEnd(ok, _line_breaks(opt, start, end, bad))
            break
    keep = sorted({t for t in grid if ends["low"].amount <= t <= ends["high"].amount}
                  | {ends["low"].amount, ends["high"].amount})
    return Path(amounts=keep, recipes=[_line_recipe(opt, start, end, t) for t in keep],
                low_end=ends["low"], high_end=ends["high"], start_breaks=None, **base)


def tested_near_line(opt, path):
    """tested_near_path for a line_path: [(index, t of the nearest point
    of the line, Overall score)] for each fully measured formulation
    within CLOSE_TO_THE_BEST of it."""
    out = []
    for i in range(len(opt.Y_history)):
        if not opt.fully_measured(i):
            continue
        rec = start_recipe(opt, i)
        ds = [distance(opt, r, rec) for r in path.recipes]
        near = [(d, t) for d, t in zip(ds, path.amounts) if d is not None]
        if near and min(near)[0] <= CLOSE_TO_THE_BEST + _EPS:
            out.append((i, min(near)[1], float(opt.Y_history[i])))
    return out


# --- specific wording (owner, 2026-10-12: "I don't like vague wording like
# 'Measurements: more misses than expected.'") ---
# A correct 95 % range misses 1 value in 20: the count a line names as
# "expected" beside the misses it saw.
MISS_RATE = 0.05


@dataclass
class ValueMisses:
    """The measurements' misses in the verdict's Rounds (headline's window,
    the values verdict_state counts): `items` [(name, misses, checked)] for
    each measurement that missed, most misses first (Results' order on a
    tie)."""
    rounds: list
    misses: int
    total: int
    items: list

    @property
    def expected(self):
        """The misses a correct 95 % range gives on `total` values."""
        return MISS_RATE * self.total

    @property
    def allowed(self):
        """The most misses the verdict still calls As expected (the
        binomial allowance, allowed_misses): the screen's "up to N
        normal", so the line and the verdict agree."""
        return allowed_misses(self.total)


def value_misses(opt, checks, h=None):
    """ValueMisses over the headline's Rounds; None with no value checked."""
    h = h or headline(opt, checks)
    values = [c for c in ranged(checks) if c.name != SCORE and c.round in (h.rounds or [])]
    if not values:
        return None
    order = [n for n, _, _ in outputs(opt) if n != SCORE]
    order += sorted({c.name for c in values} - set(order))
    items = []
    for name in order:
        mine = [c for c in values if c.name == name]
        missed = sum(not c.inside for c in mine)
        if missed:
            items.append((name, missed, len(mine)))
    items.sort(key=lambda it: -it[1])
    return ValueMisses(list(h.rounds or []), sum(it[1] for it in items), len(values), items)
