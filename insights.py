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
MODEL_SALT = "0.8.0-speed-1"
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
    folds, the fit options and MODEL_SALT."""
    return _digest({"salt": MODEL_SALT, "settings": _digest(_model_settings(opt)),
                    "name": name, "rows": [_row_key(opt, i) for i in use],
                    "folds": FOLDS, "fit": FIT_OPTIONS})


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


def folds(n):
    """[(kept, held)] positions of n rows: row k goes to fold k % FOLDS; a
    fold with nothing held is left out."""
    out = []
    for k in range(FOLDS):
        held = [p for p in range(n) if p % FOLDS == k]
        if held:
            out.append(([p for p in range(n) if p % FOLDS != k], held))
    return out


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
    FOLDS folds by row order (folds()), and each fold is predicted by a
    model fitted on the others. A fold whose fit fails is skipped. Only
    `use` is ever seen, so a Round-ahead check stays honest."""
    z = []
    for kept, held in folds(len(use)):
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
    """Everything a fitted model or a Round-ahead check depends on."""
    return _digest({
        "X": opt.X_history, "Y": opt.Y_history, "R": opt.results_history,
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
                p.raw_mean, p.sd))
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
    partial answer is never written to `cache`. The Overall score's
    widening is final only when `only` holds every Round's Overall score."""
    names = [n for n, _, _ in outputs(opt)]
    order = round_order(opt)
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
    return nest_score_widen(opt, checks)


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
    """How the Overall score's own predictions did in the headline's Rounds:
    fully measured formulations only (0.8.0 review fixes §3.3, §9 I7)."""
    inside: int
    total: int
    mean_miss: float        # measured minus predicted, on average: < 0 is below
    gap_sd: float | None = None   # the spread of those gaps (sample sd); None for one


SCORE_RECORD_MIN = 5
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
    values = [c for c in checks if c.name != SCORE]
    rounds = _ordered_rounds(values)
    score_rounds = _ordered_rounds([c for c in checks if c.name == SCORE])
    if not rounds:
        return Headline("none", [], 0, 0, None)
    k = min(3, len(rounds))
    chosen = rounds[-k:]
    while (sum(c.round in chosen for c in values) < 10 and k < len(rounds)):
        k += 1
        chosen = rounds[-k:]
    picked = [c for c in values if c.round in chosen]
    record = score_record(checks, chosen)
    wide = wide_ranges(opt, [c for c in checks if c.round in chosen])
    if len(picked) < 10:
        return Headline("early", chosen, sum(c.inside for c in picked),
                        len(values), None, record, wide)
    inside = sum(c.inside for c in picked)
    tail = binomial_tail(len(picked) - inside, len(picked))
    verdict = ("holding" if tail >= 0.10 else
               "caution" if tail >= 0.01 else "unreliable")
    return Headline("verdict", chosen, inside, len(picked), verdict, record, wide)


def score_record(checks, rounds):
    """The Overall score's checks in `rounds`, fully measured formulations
    only; None with fewer than SCORE_RECORD_MIN. It replaced the 0.8.0
    "most came in below" line, which needed 8 in 10 below and almost never
    showed while the score ran optimistic in 7 checks out of 10."""
    mine = [c for c in checks if c.name == SCORE and c.round in rounds and c.fully_measured]
    if len(mine) < SCORE_RECORD_MIN:
        return None
    gaps = [c.measured - c.predicted for c in mine]
    return ScoreRecord(sum(c.inside for c in mine), len(mine), sum(gaps) / len(gaps),
                       statistics.stdev(gaps) if len(gaps) > 1 else None)


def bias_gap(record, yardstick):
    """The Overall score's average gap when it is a bias worth saying, else
    None (answer-key read-through: one rule for Insights and Make a round).
    It must be more than `yardstick` (how much Repeats differ, 1 point
    without them) AND clear of noise: more than 2 x sd / sqrt(n) of the
    gaps themselves, so a few large misses either way are not a bias."""
    if record is None or yardstick is None or abs(record.mean_miss) <= yardstick:
        return None
    if record.gap_sd is not None and abs(record.mean_miss) <= (
            2 * record.gap_sd / math.sqrt(record.total)):
        return None
    return record.mean_miss


def score_bias(opt, checks, h=None, yard=None):
    """bias_gap for the Overall score's record in the verdict's Rounds."""
    h = headline(opt, checks) if h is None else h
    yard = noise_yardstick(opt, checks) if yard is None else yard
    return bias_gap(h.score, yard.value)


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
    `flagged`: disagreeing_repeat's answer, left out of every output's
    Repeats (review I4)."""
    rows = []
    matches = None
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
        spread = repeat_sd(opt, name, matches, () if flagged is None else (flagged.row,))
        left_out = (0 if flagged is None
                    else repeat_sd(opt, name, matches).pairs - spread.pairs)
        sd = spread.sd
        f = (fitted or {}).get(name)
        rows.append(DetailRow(
            name, len(mine), sum(c.inside for c in mine),
            statistics.median((c.high - c.low) / 2 for c in mine), miss,
            # sd 0: the Repeats agreed exactly, and there is no ratio.
            sd, None if not sd else miss / (MAD_OF_NORMAL * sd),
            f.widen if f is not None and f.model is not None else None, spread.pairs,
            left_out))
    return rows


# ------------------------------------------------------------------ #
#  0.8.0 speed and simplicity, Plan B: one yardstick, one verdict, one
#  next step (spec 2026-09-30 §2; review I2, I4, I5, I6). Every function
#  here takes the checks as an argument, so they can come from a cache.
# ------------------------------------------------------------------ #

# A pooled spread between Repeats is the yardstick only from this many pairs
# (review I5): "don't benchmark on one pair".
YARDSTICK_PAIRS = 3
# The over-wide verdict (review I6): every result inside, at least this many
# checks, and a median |z| this small. Simulated with calibrated misses it
# fires 2.9 % of the time at 20 checks and 0.4 % at 33.
OVER_WIDE_MIN = 20
OVER_WIDE_Z = 0.4


@dataclass
class Yardstick:
    value: float | None     # the threshold: Repeats' sd from 3+ pairs, else 1 point (None
                            # for a measurement)
    sd: float | None        # the pooled sd between Repeats, a disagreeing one left out
    pairs: int              # how many pairs of Repeats sd rests on
    from_repeats: bool      # value is sd (3+ pairs, at least REPEAT_SD_SHOWN)


@dataclass
class Disagreement:
    later: int              # the Repeat's Formulation number
    earlier: int            # the Formulation number it repeats
    diff: float             # later minus earlier, Overall score
    row: int                # the later one's row: left out of every Repeat spread
    limit: float | None = None  # √2 × the half-width it broke (03 §2 #13), points


def disagreeing_repeat(opt, checks, matches=None):
    """The one Repeat whose Overall score differs from its original by more
    than two results should (review I4), or None: |d| over √2 times the
    half-width of the later one's own Round-ahead check. Only a later one
    in the verdict's Rounds counts, a group of three or more compares each
    with its first, either direction counts, and of several the largest
    |d| is named. `matches`: as for repeat_groups."""
    window = set(headline(opt, checks).rounds)
    if not window:
        return None
    ahead = {c.formulation: c for c in checks
             if c.name == SCORE and c.fully_measured and c.round in window}
    if not ahead:
        return None
    best = None
    for group in repeat_groups(opt, opt.best_candidates(), matches):
        first, *rest = sorted(group)
        for i in rest:
            c = ahead.get(int(opt.formulation_ids[i]))
            if c is None:
                continue
            # The half-width shown for that check, its I1 widening included.
            half = Z95 * c.sd * c.widen if c.sd else (c.high - c.low) / 2
            d = float(opt.Y_history[i]) - float(opt.Y_history[first])
            if abs(d) > math.sqrt(2) * half and (best is None or abs(d) > abs(best.diff)):
                best = Disagreement(int(opt.formulation_ids[i]),
                                    int(opt.formulation_ids[first]), d, i,
                                    math.sqrt(2) * half)
    return best


def noise_yardstick(opt, checks=(), name=SCORE):
    """What a difference must beat to be more than chance (review I5): the
    pooled sd between Repeats of `name`, a disagreeing Repeat left out,
    from YARDSTICK_PAIRS pairs or more and at least REPEAT_SD_SHOWN; else 1
    point for the Overall score (None for a measurement). Every rule that
    asks "more than Repeats differ?" asks this. The disagreeing Repeat is
    left out only when `checks` are handed in."""
    flagged = disagreeing_repeat(opt, checks) if checks else None
    spread = repeat_sd(opt, name, None, () if flagged is None else (flagged.row,))
    usable = (spread.sd is not None and spread.sd >= REPEAT_SD_SHOWN
              and spread.pairs >= YARDSTICK_PAIRS)
    fallback = 1.0 if name == SCORE else None
    return Yardstick(spread.sd if usable else fallback, spread.sd, spread.pairs, usable)


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
    verdict's Rounds (0.67 when the 95 % intervals are right), or None:
    the over-wide rule's basis, said with it (03 §5.2)."""
    window = [c for c in checks if c.round in h.rounds]
    z = [abs((c.measured - c.raw_mean) / c.sd) for c in window
         if c.name != SCORE and c.sd and c.raw_mean is not None]
    return statistics.median(z) if z else None


@dataclass
class NextStep:
    kind: str       # "bias", "misses", "repeat", "early", "no_skill", "untested",
                    # "close", "rising" or "see"
    gap: float | None = None                    # "bias": the Overall score's average miss
    disagreement: Disagreement | None = None    # "repeat"
    binding: object = None                      # "close": what binds the predicted best
    wide: list = dataclasses.field(default_factory=list)   # "rising": outputs still wide
    skill: object = None                        # "no_skill": the SkillFact (BO views, I1)


def next_step(opt, checks, standing=None):
    """What to do next (review I2), first match wins. `standing`: the
    predicted best against Best so far (an object with .higher,
    .tested_near and .binding), or None when it could not be worked out;
    the rows that need it are then skipped. Never called at the gate
    (fewer than MIN_ROWS scored) or after a failed fit: the screen says
    the gate line there."""
    h = headline(opt, checks)
    state = verdict_state(opt, h, checks)
    yard = noise_yardstick(opt, checks)
    if state == "unreliable":
        gap = score_bias(opt, checks, h, yard)
        if gap is not None:
            return NextStep("bias", gap=gap)
        return NextStep("misses")
    flagged = disagreeing_repeat(opt, checks)
    if flagged is not None:
        return NextStep("repeat", disagreement=flagged)
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
    if standing is not None:
        if standing.higher and not standing.tested_near:
            return NextStep("untested")
        if (not standing.higher and answer is not None
                and answer.state in ("not_lately", "barely")):
            return NextStep("close", binding=standing.binding[0] if standing.binding else None)
    if answer is not None and answer.state == "yes":
        # Said with the width caution the verdict gives (answer-key
        # read-through, Line C).
        return NextStep("rising", wide=list(h.wide or []))
    return NextStep("see")


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
        return opt.scaled_recipe(recipe, scale_to=float(opt.formulation_total))
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
      the Overall score's spread between Repeats (1 point without Repeats,
      or with a spread under REPEAT_SD_SHOWN), Yes otherwise. Both are
      compared to the hundredth, as the answer can print them, so its
      words never contradict its numbers.
    - Anything else (one Round only, or no higher score in just the
      Round since): None, and Best so far's own line says it.

    `facts` and `spread`: best_facts(opt) and repeat_sd(opt, SCORE), when
    the caller has them already. `yardstick` (noise_yardstick, spec
    2026-09-30 review I5), when given, replaces `spread`: the spread
    between Repeats counts only from YARDSTICK_PAIRS pairs, else 1 point."""
    f = best_facts(opt) if facts is None else facts
    if f is None:
        return None
    if yardstick is not None:
        sd = yardstick.value if yardstick.from_repeats else None
    else:
        sd = (repeat_sd(opt, SCORE) if spread is None else spread).sd
    if sd is not None and sd < REPEAT_SD_SHOWN:
        sd = None
    if f.rounds_since >= 2:
        return Improvement("not_lately", None, sd, f.rounds_since)
    if f.rounds_since or not f.latest:
        return None
    earlier = [float(opt.Y_history[i]) for i in opt.best_candidates()
               if opt.batch_history[i] != f.round]
    if not earlier:
        return None
    rise = f.score - max(earlier)
    threshold = sd if sd is not None else 1.0
    barely = round(rise, 2) < round(threshold, 2)
    return Improvement("barely" if barely else "yes", rise, sd, 0)


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
    return RepeatSpread(math.sqrt(ss / dof), pairs)


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
TOGETHER = 0.8
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


def best_standing(opt, best, yardstick):
    """The predicted best against Best so far (review I7): like with like,
    its mean against Best so far's predicted Repeat (the measured score
    when there is none), judged by `yardstick` (points) — and, as shown,
    above Best so far's measured score by more than the yardstick too."""
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
            and round(best.repeat.mean) < round(measured_top, 2))
    close_no = None if best.close is None else int(opt.formulation_ids[best.close])
    return Standing(best, higher, best.close is not None, close_no,
                    binding_limits(opt, best.recipe), luck)


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


def round_track(opt, checks):
    """Make a round's line under its table, from the Overall score's own
    record in the verdict's Rounds (review I16): rough only when fewer
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
                          list(h.rounds), record=False)
    rough = binomial_tail(record.total - record.inside, record.total) < 0.10
    return RoundTrack(rough, record.inside, record.total, score_bias(opt, checks, h),
                      state == "unreliable", no_skill, list(h.rounds))


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
SKILL_GATE_Q10 = {10: 0.00, 12: 0.03, 15: 0.11, 20: 0.25, 25: 0.32, 30: 0.45}
SKILL_CUT_MAX = -0.2
SKILL_SMALL = 0.2


def skill_cut(n):
    """c_n for a skill over `n` checks: the gate's 10th percentile at the
    largest tabulated n at or below it (the percentile rises with n, so
    this errs towards fewer no-skill warnings), capped at SKILL_CUT_MAX."""
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
# Repeat SD (from YARDSTICK_PAIRS pairs).
GUARD_FLOOR = 1.05
GUARD_NOISE = 0.5


@dataclass
class Guard:
    reasons: list       # "floor", "noise", "skill", in that order

    @property
    def fires(self):
        return bool(self.reasons)

    @property
    def skill_only(self):
        """Trigger (c) alone: said as "not reported until…", never as
        "fitting noise" (review I3)."""
        return self.reasons == ["skill"]


def _lengthscale_at_floor(gp):
    """Whether any kernel's smallest correlation length sits within
    GUARD_FLOOR of that kernel's own lower limit (review I3: read from the
    module, so an expert's kernel is judged by its own bound)."""
    for module in gp.covar_module.modules():
        constraint = getattr(module, "raw_lengthscale_constraint", None)
        if constraint is None or not hasattr(module, "lengthscale"):
            continue
        low = float(torch.as_tensor(constraint.lower_bound).min())
        if low > 0 and float(module.lengthscale.detach().min()) <= GUARD_FLOOR * low:
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
    YARDSTICK_PAIRS pairs or more, never under fixed_tiny; (c) for the
    Overall score only, Round-ahead skill at or below its cut (`fact`:
    skill() when the caller has it). `fitted`: insights.Fitted, or None."""
    reasons = []
    gp = getattr(fitted, "model", None)
    if gp is not None:
        try:
            if _lengthscale_at_floor(gp):
                reasons.append("floor")
        except Exception:
            pass
        tiny = (getattr(opt, "bo_config", None) or {}).get("noise") == "fixed_tiny"
        yard = noise_yardstick(opt, checks, name)
        sd = None if tiny else fitted_noise_sd(gp)
        if (sd is not None and yard.sd and yard.pairs >= YARDSTICK_PAIRS
                and sd < GUARD_NOISE * yard.sd):
            reasons.append("noise")
    if name == SCORE:
        fact = skill(opt, checks) if fact is None else fact
        if fact is not None and fact.state == "no_skill":
            reasons.append("skill")
    return Guard(reasons)
