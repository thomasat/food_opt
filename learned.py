"""What has the model learned? (review learned, prototype 2026-10-02).

Honest, checkable evidence from the app's own Round-ahead checks
(insights.check_rounds): each formulation predicted from the rows recorded
before its Round, then compared with what was measured.

- Ordering: within each Round, every pair of formulations whose measured
  values differ is "ordered right" when the Round-ahead prediction put them
  in the same order. Chance is half the pairs. A measured tie is not
  counted. Persona fixes 2 (coordinator: an order claim only when the
  forecast made one): a pair whose forecasts differ by less than one
  tasting noise of the result (pair_noise: the noise Progress uses) is
  "too close to call" and not counted either, nor is "the top one" when
  the top two forecasts are that close.
- Verdict: Predictable when the share right is clearly above chance (an
  exact within-Round permutation test, review critique: the measured
  values shuffled among each Round's formulations independently, the
  statistic right − wrong summed over Rounds, p < VERDICT_P, with at least
  MIN_PAIRS pairs); Partly when it leans right (share above 50 %,
  p < PARTLY_P, at least PARTLY_MIN pairs); Not yet otherwise, or when the
  measurement barely varies within Rounds (more than FLAT_TIES of its
  pairs measured equal). The chart's chance range is the same null's, so
  its dot sits past the range exactly when the verdict is Predictable.
- Clear effect (review learned 2, strict): moving one ingredient or setting
  from the 10th to the 90th percentile of its tested amounts, averaged
  over the tested formulations, has a 95 % posterior interval that
  excludes 0, with the same sign, under EACH of the last STABLE_REFITS
  Round-ahead refits (fitted through Rounds k-3 … k); the row did not move
  together with another (|r| >= insights.MOVED_R); and the latest change is
  at least one noise SD of the result (its Repeats' SD when they give one,
  else the larger of the latest model's fitted noise and the Round-ahead
  forecasts' root-mean-square miss; the Overall score's assumed 1 point as
  the floor). Anything less is no claim.
- Before you taste: the open Round's forecast from the rows recorded before
  the Round, the very model its Round-ahead check later uses.
- What next preview: FoodOptimizer.ask() on a deep copy whose saving is
  switched off, holding the RNG lock, so the project and the global RNG
  are untouched.

Nothing here writes to the project."""
import copy
import itertools
import math
import random
from dataclasses import dataclass, field

import torch
from botorch.utils.transforms import normalize

import insights

VERDICT_P = 0.05
MIN_PAIRS = 5
PARTLY_P = 0.20
PARTLY_MIN = 5
FLAT_TIES = 0.5
STABLE_REFITS = 4
STRETCH = (0.10, 0.90)

PREDICTABLE, PARTLY, NOT_YET = "predictable", "partly", "not_yet"


def chance_tail(right, pairs):
    """P(at least `right` of `pairs` right by a coin toss): the one-sided
    binomial p-value against 50 %."""
    if pairs <= 0:
        return 1.0
    return sum(math.comb(pairs, k) for k in range(right, pairs + 1)) / 2 ** pairs


PERM_EXACT_MAX = 7      # Rounds up to this size: every ordering; larger: PERM_SAMPLES draws
PERM_SAMPLES = 4000


def _round_stat(mine, measured, noise):
    """right − wrong over the counted pairs of one Round's checks `mine`
    (by formulation), the measured values taken from `measured` in the same
    order: a pair counts when its measured values differ and its forecasts
    are not too close (round_call's rule)."""
    s = 0
    for i, a in enumerate(mine):
        for j in range(i + 1, len(mine)):
            b = mine[j]
            ma, mb = measured[i], measured[j]
            if ma == mb or too_close(a, b, noise):
                continue
            ra, rb = _rank_value(a), _rank_value(b)
            if ra == rb:
                s -= 1
                continue
            s += 1 if (ra > rb) == (ma > mb) else -1
    return s


def round_null(mine, noise=0.0):
    """{S: probability} of right − wrong for one Round under the null: the
    measured values shuffled among its formulations (forecasts fixed). Exact
    up to PERM_EXACT_MAX formulations, else PERM_SAMPLES seeded draws."""
    vals = [c.measured for c in mine]
    out = {}
    if len(mine) <= PERM_EXACT_MAX:
        perms = list(itertools.permutations(range(len(vals))))
    else:
        rng = random.Random(len(vals))
        perms = []
        for _ in range(PERM_SAMPLES):
            k = list(range(len(vals)))
            rng.shuffle(k)
            perms.append(k)
    for p in perms:
        s = _round_stat(mine, [vals[k] for k in p], noise)
        out[s] = out.get(s, 0) + 1
    total = float(len(perms))
    return {k: v / total for k, v in out.items()}


def convolve(nulls):
    """The null of the summed statistic: the Rounds' nulls convolved."""
    acc = {0: 1.0}
    for d in nulls:
        nxt = {}
        for a, pa in acc.items():
            for b, pb in d.items():
                nxt[a + b] = nxt.get(a + b, 0.0) + pa * pb
        acc = nxt
    return acc


def perm_tail(null, observed):
    """P(S ≥ observed) under `null` ({S: probability})."""
    return min(1.0, sum(p for s, p in null.items() if s >= observed - 1e-9))


def chance_range(null, alpha=0.05):
    """(low, high) of S that chance reaches: high the largest S with
    P(S ≥ high) ≥ alpha, low the smallest with P(S ≤ low) ≥ alpha. The
    observed S beats chance (p < alpha) exactly when it is above high."""
    keys = sorted(null)
    high = max(k for k in keys if perm_tail(null, k) >= alpha - 1e-12)
    low = min(k for k in keys if sum(null[j] for j in keys if j <= k) >= alpha - 1e-12)
    return low, high


def _rank_value(c):
    # The model's own mean, before clipping to the Range: two predictions
    # clipped at the top would otherwise read as a tie.
    return c.raw_mean if c.raw_mean is not None else c.predicted


@dataclass
class PairCall:
    high: int            # the formulation predicted higher
    low: int
    right: bool | None   # None: not counted (measured equal, or too close)
    close: bool = False  # forecasts under one noise SD apart: too close to call


@dataclass
class RoundCall:
    round: int
    pairs: list
    top_predicted: int | None
    top_measured: list   # the formulations measured highest (ties: several)
    top_close: bool = False     # the top two forecasts under one noise SD apart

    @property
    def right(self):
        return sum(1 for p in self.pairs if p.right)

    @property
    def counted(self):
        return sum(1 for p in self.pairs if p.right is not None)

    @property
    def close(self):
        """Pairs too close to call (measured apart, forecasts within noise)."""
        return sum(1 for p in self.pairs if p.close)

    @property
    def ties(self):
        return sum(1 for p in self.pairs if p.right is None and not p.close)

    @property
    def top_right(self):
        """Whether the formulation predicted highest was measured highest;
        None when every formulation measured the same."""
        if self.top_predicted is None or len(self.top_measured) == len(self.pairs_members()):
            return None
        if self.top_close:
            return None
        return self.top_predicted in self.top_measured

    def pairs_members(self):
        return {n for p in self.pairs for n in (p.high, p.low)}


def at_cap(c):
    """Round-2 fixes (P1 #3): whether check `c`'s forecast is at the top of
    its Range: the model's own mean above it, the shown forecast clipped
    there (both read "≥ 100")."""
    return c.raw_mean is not None and c.raw_mean > c.predicted + 1e-9


def too_close(a, b, noise):
    """Whether the forecasts of checks `a` and `b` differ by less than one
    noise SD (`noise`; 0 or None: never), or are both at the top of the
    Range (round-2 fixes: two forecasts that both read "≥ 100" make no
    order to call)."""
    if at_cap(a) and at_cap(b):
        return True
    return bool(noise) and abs(_rank_value(a) - _rank_value(b)) < noise - 1e-12


def round_call(checks, name, r, noise=0.0):
    """One Round's pairs for `name`, each by the model's predicted order; a
    pair measured apart whose forecasts are under `noise` apart is too close
    to call (not counted)."""
    mine = sorted([c for c in checks if c.name == name and c.round == r
                   and (name != insights.SCORE or c.fully_measured)],
                  key=lambda c: c.formulation)
    pairs = []
    for i, a in enumerate(mine):
        for b in mine[i + 1:]:
            hi, lo = (a, b) if _rank_value(a) >= _rank_value(b) else (b, a)
            close = False
            if hi.measured == lo.measured:
                right = None
            elif too_close(hi, lo, noise):
                right, close = None, True
            else:
                right = _rank_value(hi) > _rank_value(lo) and hi.measured > lo.measured
            pairs.append(PairCall(hi.formulation, lo.formulation, right, close))
    if len(mine) < 2:
        return RoundCall(r, [], None, [])
    ranked = sorted(mine, key=_rank_value, reverse=True)
    top = ranked[0]
    best = max(c.measured for c in mine)
    return RoundCall(r, pairs, top.formulation,
                     [c.formulation for c in mine if c.measured == best],
                     too_close(ranked[0], ranked[1], noise))


@dataclass
class Ordering:
    name: str
    right: int = 0
    pairs: int = 0          # counted pairs (measured values differ)
    ties: int = 0           # pairs measured equal
    close: int = 0          # pairs too close to call (forecasts within noise)
    top_right: int = 0
    top_rounds: int = 0
    top_chance: float = 0.0
    rounds: list = field(default_factory=list)
    verdict: str = NOT_YET
    reason: str = "few"     # "few", "flat", "chance" or "" (Predictable/Partly)
    nulls: list = field(default_factory=list)   # per Round {S: probability}
    # Round-2 fixes (P1 #2): Holm across the results checked (orderings):
    # the adjusted p-value the Predictable verdict reads, and the level the
    # chart's chance bar is drawn at, so the dot clears the bar exactly
    # when the verdict is Predictable. One result alone: unadjusted.
    p_holm: float | None = None
    level: float = VERDICT_P
    tested: int = 1         # how many results the Holm step counted

    @property
    def share(self):
        return self.right / self.pairs if self.pairs else None

    @property
    def stat(self):
        """right − wrong over the counted pairs."""
        return 2 * self.right - self.pairs

    @property
    def null(self):
        return convolve(self.nulls)

    @property
    def p(self):
        """Review critique: the exact within-Round permutation p-value of
        right − wrong (the binomial on pairs treated a Round's 3 pairs as
        independent)."""
        if not self.pairs:
            return 1.0
        if not self.nulls:
            # Built by hand, with no Rounds behind it: the binomial on pairs.
            return chance_tail(self.right, self.pairs)
        return perm_tail(self.null, self.stat)

    def chance_share(self):
        """(low, high) of the chance range as a share right (0–1) at this
        result's own number of pairs: what shuffling within Rounds reaches
        (one-sided VERDICT_P each end); None without pairs."""
        if not self.pairs or not self.nulls:
            return None
        lo, hi = chance_range(self.null, self.level)
        to = lambda s: min(1.0, max(0.0, (s + self.pairs) / (2 * self.pairs)))  # noqa: E731
        return to(lo), to(hi)

    @property
    def chance(self):
        return self.pairs / 2


def judge(o):
    """`o` with its verdict and reason set by the rule in the module
    docstring."""
    every = o.pairs + o.ties + o.close
    if every and o.ties / every > FLAT_TIES:
        o.verdict, o.reason = NOT_YET, "flat"
    elif o.pairs < PARTLY_MIN:
        o.verdict, o.reason = NOT_YET, "few"
    elif o.pairs >= MIN_PAIRS and (o.p if o.p_holm is None else o.p_holm) < VERDICT_P:
        o.verdict, o.reason = PREDICTABLE, ""
    elif o.right * 2 > o.pairs and o.p < PARTLY_P:
        o.verdict, o.reason = PARTLY, ""
    else:
        o.verdict, o.reason = NOT_YET, "chance" if o.pairs >= MIN_PAIRS else "few"
    return o


def ordering(checks, name, noise=0.0):
    """Ordering accuracy of `name` over every Round checked, judged; pairs
    too close to call (`noise`) left out."""
    o = Ordering(name)
    for r in insights._ordered_rounds([c for c in checks if c.name == name]):
        call = round_call(checks, name, r, noise)
        if not call.pairs:
            continue
        o.rounds.append(r)
        mine = sorted([c for c in checks if c.name == name and c.round == r
                       and (name != insights.SCORE or c.fully_measured)],
                      key=lambda c: c.formulation)
        o.nulls.append(round_null(mine, noise))
        o.right += call.right
        o.pairs += call.counted
        o.ties += call.ties
        o.close += call.close
        if call.top_right is not None:
            members = call.pairs_members()
            o.top_rounds += 1
            o.top_right += int(call.top_right)
            o.top_chance += len(call.top_measured) / len(members)
    return judge(o)


def names(opt):
    """The Overall score, then each measurement by importance."""
    return [insights.SCORE] + [o['name'] for o in opt.measurements_by_importance()]


def orderings(opt, checks, noise=None):
    """{name: Ordering}; `noise` {name: SD} (pair_noise of `checks` when
    None): a pair closer than that is too close to call."""
    noise = pair_noise(opt, checks) if noise is None else noise
    return holm({n: ordering(checks, n, noise.get(n) or 0.0) for n in names(opt)})


def holm(ords, alpha=VERDICT_P):
    """Round-2 fixes (P1 #2): Holm's step-down adjustment of the
    within-Round permutation p-values across the results checked (those
    with a counted pair), then each verdict judged again: Predictable needs
    the adjusted p below `alpha`; Partly still reads the result's own p.
    Each result's chance bar is drawn at its Holm level, alpha ÷ (m − k)
    for the k-th smallest p, held at the first level that fails for every
    result after it — so a dot is past its bar exactly when the result is
    Predictable."""
    tested = sorted((o for o in ords.values() if o.pairs), key=lambda o: o.p)
    m = len(tested)
    worst, failed = 0.0, None
    for k, o in enumerate(tested):
        level = alpha / (m - k)
        if failed is None and o.p >= level - 1e-15:
            failed = level
        worst = max(worst, min(1.0, (m - k) * o.p))
        o.p_holm, o.level, o.tested = worst, failed if failed is not None else level, m
        judge(o)
    return ords


def _repeat_noise(opt, name):
    """`name`'s SD between Repeats from the same threshold Progress uses
    (insights.noise_yardstick: YARDSTICK_DF degrees of freedom, at least
    REPEAT_SD_SHOWN), else None."""
    try:
        yard = insights.noise_yardstick(opt, name=name)
    except Exception:
        return None
    if yard.from_repeats and yard.value:
        return float(yard.value)
    return None


def overall_noise(opt):
    """(SD, assumed): the Overall score's tasting noise as Progress has it —
    the Repeats' SD from YARDSTICK_DF degrees of freedom, else the assumed
    1 point."""
    sd = _repeat_noise(opt, insights.SCORE)
    return (sd, False) if sd is not None else (insights.assumed_noise(opt), True)


def points_per_unit(obj):
    """How many Overall-score points one unit of a measurement moves, from
    FoodOptimizer._compute_utility: its share over its Range width, for a
    target, a higher-is-better and a lower-is-better measurement alike."""
    width = float(obj.get('max_val', 10.0)) - float(obj.get('min_val', 0.0))
    width = width if width else 1.0
    return float(obj.get('weight', 0.0)) / width if obj.get('goal') in (
        'target', 'max', 'min') else 0.0


def scored_measurements(opt):
    """The measurements that move the Overall score (points_per_unit > 0)."""
    return [o for o in getattr(opt, "objectives", None) or () if points_per_unit(o) > 0]


def derived_noise(opt, name, sigma=None):
    """Review critique (owner's decision): a measurement's tasting noise
    derived from the Overall score's, every scored measurement assumed to
    add the same share of the Overall variance: σ_a = σ × w_a ÷ (s_a × √k),
    w_a its Range width, s_a its Share of score, k the scored measurements.
    The yogurt: 1 × 15 ÷ (25 × 2) = 0.3. 0 when it moves no points."""
    sigma = overall_noise(opt)[0] if sigma is None else sigma
    objs = {o['name']: o for o in getattr(opt, "objectives", None) or ()}
    ppu = points_per_unit(objs[name]) if name in objs else 0.0
    k = len(scored_measurements(opt))
    return float(sigma) / (ppu * math.sqrt(k)) if ppu > 0 and k else 0.0


def pair_noise(opt, checks=()):
    """{name: the tasting noise an order claim must reach} (coordinator's
    ruling, persona fixes 2: the same noise Progress uses, so the screen
    tells one story). The Overall score: overall_noise. A measurement: its
    Repeats' SD from the same threshold; else derived_noise (review
    critique: 0.3 on a 0–15 target worth 25 of 4 equal measurements; it
    read 0.6, as if one measurement carried all of the Overall noise).
    `checks` is accepted for callers and not used."""
    sigma, _ = overall_noise(opt)
    out = {insights.SCORE: float(sigma)}
    for n in names(opt):
        if n == insights.SCORE:
            continue
        sd = _repeat_noise(opt, n)
        if sd is None:
            sd = derived_noise(opt, n, sigma)
        out[n] = float(sd)
    return out


def round_bias(cks, noise):
    """Persona fixes 2: measured − forecast on average over a Round's checks
    `cks` of one result, when every one sits on the same side of its
    forecast by more than `noise` (at least two checks); else None."""
    if len(cks) < 2 or not noise:
        return None
    d = [c.measured - c.predicted for c in cks]
    if all(x > noise for x in d) or all(x < -noise for x in d):
        return sum(d) / len(d)
    return None


def before(checks, r):
    """The checks of the Rounds before Round `r` (in check order)."""
    order = insights._ordered_rounds(checks)
    if r not in order:
        return list(checks)
    earlier = set(order[:order.index(r)])
    return [c for c in checks if c.round in earlier]


def latest_round(checks):
    order = insights._ordered_rounds(checks)
    return order[-1] if order else None


def called_it(opt, checks, r=None, noise=None):
    """[(name, RoundCall)] for Round `r` (the latest checked when None), for
    the measurements that were Predictable BEFORE that Round — the ones its
    forecast showed — so nothing is picked after the fact. The Overall score
    is never in it (its forecast is the Make a round column). Empty when
    none was."""
    r = latest_round(checks) if r is None else r
    if r is None:
        return r, []
    noise = pair_noise(opt, checks) if noise is None else noise
    earlier = before(checks, r)
    picked = [n for n, o in orderings(opt, earlier, noise).items()
              if n != insights.SCORE and o.verdict == PREDICTABLE]
    return r, [(n, round_call(checks, n, r, noise.get(n) or 0.0)) for n in picked]


CALLED, CALLED_PARTLY, CALLED_MISSED = "called", "partly", "missed"
CALLED_PANELS = 2       # Predictable measurements drawn beside the Overall score


def round_checks(checks, name, r):
    """`name`'s checks of Round `r`, by formulation (the Overall score:
    fully measured only)."""
    return sorted([c for c in checks if c.name == name and c.round == r
                   and (name != insights.SCORE or c.fully_measured)],
                  key=lambda c: c.formulation)


def is_latest(opt, r):
    """Whether Round `r` is the project's latest Round: none recorded or
    opened after it (an open Round with nothing saved yet counts)."""
    order = insights.round_order(opt)
    pending = getattr(opt, "pending_batch_no", None) if getattr(opt, "pending_batch", None) else None
    return bool(order) and order[-1][0] == r and pending in (None, r)


def all_tied(cks):
    """Whether every measured value of a Round's checks is the same (nothing
    to order: its panel is left out)."""
    return len({c.measured for c in cks}) <= 1


def shown(items, cap=1 + CALLED_PANELS):
    """The items Called it? draws: those with something to order (not every
    value measured the same), at most `cap`, in order."""
    return [it for it in items if not all_tied(it[2])][:cap]


@dataclass
class CalledSummary:
    """Called it?'s facts for one Round, judged on ORDER (review learned 2,
    coordinator: the order is the claim): `drawn` the items drawn, `right`
    the names whose order held (every counted pair right, the one forecast
    highest measured highest), `off` the names it did not, `tied` the names
    left out (all measured the same), `lead` the name the headline leads
    with (the first measurement drawn, else the Overall score) and `top`
    its formulation forecast highest; `above` the Overall scores measured
    above their 95 % range (misses, like those below); `close` the names
    drawn whose every pair was too close to call (persona fixes 2)."""
    verdict: str
    drawn: list
    right: list
    off: list
    tied: list
    lead: str | None
    top: int | None
    above: int
    close: list = field(default_factory=list)


def order_held(call):
    """Every counted pair in the forecast order and the top one picked; at
    least one pair counted (all too close: nothing held)."""
    return call.counted > 0 and call.right == call.counted and call.top_right is not False


def order_counts(call):
    """Whether any pair of `call` counts (measured apart, forecasts apart)."""
    return call.counted > 0


def called_summary(items):
    """CalledSummary of `items` [(name, RoundCall, [Check])], the Overall
    score first: called when every drawn item's order held, missed when
    none did, partly otherwise."""
    drawn = shown(items)
    right = [n for n, call, _ in drawn if order_held(call)]
    close = [n for n, call, _ in drawn if not order_counts(call)]
    off = [n for n, call, _ in drawn if order_counts(call) and not order_held(call)]
    tied = [n for n, _, cks in items if all_tied(cks) and cks]
    above = sum(1 for n, _, cks in items if n == insights.SCORE
                for c in cks if c.measured > c.high)
    if not drawn or not (right or off):
        verdict = CALLED_MISSED if off else CALLED
    elif not off:
        verdict = CALLED
    elif right:
        verdict = CALLED_PARTLY
    else:
        verdict = CALLED_MISSED
    measures = [it for it in drawn if it[0] != insights.SCORE]
    lead_item = (measures or drawn or [None])[0]
    lead = lead_item[0] if lead_item else None
    top = lead_item[1].top_predicted if lead_item else None
    return CalledSummary(verdict, [n for n, _, _ in drawn], right, off, tied, lead, top, above,
                         close)


# ------------------------------------------------------------------ #
#  Skill against two baselines (the expert line)
# ------------------------------------------------------------------ #

def skill_vs_best(opt, checks, name=insights.SCORE):
    """Round-ahead skill against Best so far: 1 − Σ(measured − predicted)² ÷
    Σ(measured − b)², b the value of `name` at the best-scoring formulation
    recorded before the check's Round. None with fewer than SKILL_MIN."""
    mine = [c for c in checks if c.name == name and (name != insights.SCORE or c.fully_measured)]
    if len(mine) < insights.SKILL_MIN:
        return None
    first = dict(insights.round_order(opt))
    model = base = 0.0
    n = 0
    for c in mine:
        rows = [i for i in range(first.get(c.round, 0)) if opt.fully_measured(i)
                and not opt.is_set_aside(i)]
        if not rows:
            continue
        top = max(rows, key=lambda i: opt.Y_history[i])
        b = insights.measured(opt, name, top)
        if b is None:
            continue
        n += 1
        model += (c.measured - c.predicted) ** 2
        base += (c.measured - b) ** 2
    if n < insights.SKILL_MIN or base <= 0:
        return None
    return 1.0 - model / base, n


def coverage(checks, name=insights.SCORE):
    mine = [c for c in insights.ranged(checks) if c.name == name and (name != insights.SCORE or c.fully_measured)]
    return sum(c.inside for c in mine), len(mine)


# ------------------------------------------------------------------ #
#  Stable effects
# ------------------------------------------------------------------ #

@dataclass
class Effect:
    name: str           # the output
    var: str            # the ingredient or setting
    lo: float           # the tested stretch, 10th to 90th percentile
    hi: float
    mean: float         # the predicted change from lo to hi, latest model
    low: float          # its 95 % interval
    high: float
    signs: list         # the sign of the mean under each refit, oldest first
    stable: bool        # all three parts of the clear-effect rule hold
    steady: bool = False    # (a) every refit's interval excludes 0, one sign
    together: bool = False  # (b) moved together with another row
    noise: float | None = None  # (c) the result's noise SD it must reach

    @property
    def up(self):
        return self.mean > 0


def _columns(opt):
    """[(name, column of model_X)] for every continuous row the models see."""
    out, col = [], 0
    for var in opt.variables:
        if opt.has_formula(var):
            continue
        if var['type'] == 'continuous':
            out.append((var['name'], col))
            col += 1
        elif var['type'] == 'categorical':
            col += len(var['options'])
    return out


def _quantile(values, q):
    v = sorted(values)
    pos = (len(v) - 1) * q
    lo = math.floor(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def effect_of(opt, fitted, rows, col):
    """(lo, hi, mean, sd) of the average predicted change of `fitted`'s
    output when column `col` moves from the 10th to the 90th percentile of
    its tested amounts, every other amount as tested; None when it hardly
    varied. The mean and sd are of the model's own posterior (no
    observation noise, no widening): the 95 % posterior interval."""
    use = [i for i in rows if insights.measured(opt, fitted.name, i) is not None]
    if fitted.model is None or len(use) < insights.MIN_ROWS:
        return None
    X = torch.tensor(opt.model_X(use), dtype=torch.double)
    bounds = opt._get_bounds()
    span = float(bounds[1, col] - bounds[0, col])
    amounts = X[:, col].tolist()
    lo, hi = _quantile(amounts, STRETCH[0]), _quantile(amounts, STRETCH[1])
    if span <= 0 or hi - lo <= 1e-6 * span:
        return None
    A, B = X.clone(), X.clone()
    A[:, col], B[:, col] = lo, hi
    n = len(use)
    w = torch.cat([-torch.ones(n), torch.ones(n)]).double() / n
    with torch.no_grad(), insights._isolated():
        post = fitted.model.posterior(normalize(torch.cat([A, B]), bounds))
        mu = post.mean.reshape(-1)
        cov = post.mvn.covariance_matrix
        if cov.dim() > 2:
            cov = cov.reshape(2 * n, 2 * n)
    mean = float(w @ mu)
    sd = math.sqrt(max(0.0, float(w @ cov @ w)))
    return lo, hi, mean, sd


def refit_rows(opt):
    """[(Round, rows)]: the rows the models were fitted on through each of the
    last STABLE_REFITS Rounds, oldest first — exactly the training rows of
    the next Round's Round-ahead check, and every row for the latest — so
    every one is a model already fitted (and kept) for the checks."""
    order = insights.round_order(opt)
    out = []
    for k, (r, _first) in enumerate(order):
        end = order[k + 1][1] if k + 1 < len(order) else len(opt.Y_history)
        out.append((r, list(range(end))))
    return out[-STABLE_REFITS:]


def forecast_miss(checks, name):
    """The Round-ahead forecasts' root-mean-square miss on `name` (the
    Overall score: fully measured only), or None with no check."""
    mine = [c for c in checks or () if c.name == name
            and (name != insights.SCORE or c.fully_measured)]
    if not mine:
        return None
    return math.sqrt(sum((c.measured - c.predicted) ** 2 for c in mine) / len(mine))


def noise_of(opt, name, fitted, checks=()):
    """The noise SD an effect on `name` must reach: its Repeats' SD when
    they give one (insights.noise_yardstick); else the larger of the latest
    model's fitted noise and the Round-ahead forecasts' typical miss (the
    fitted noise alone runs small on few rows: a conservative stand-in); the
    Overall score never under its assumed 1 point. None when nothing is
    known (then no claim)."""
    yard = insights.noise_yardstick(opt, name=name)
    if yard.from_repeats and yard.value is not None:
        value = float(yard.value)
    else:
        fitted_sd = None
        if fitted is not None and fitted.model is not None:
            try:
                fitted_sd = insights.fitted_noise_sd(fitted.model)
            except Exception:
                fitted_sd = None
        known = [v for v in (fitted_sd, forecast_miss(checks, name)) if v is not None]
        value = max(known) if known else None
    if name == insights.SCORE:
        value = max(value or 0.0, insights.assumed_noise(opt))
    return value


def judge_effect(got, together, noise):
    """(steady, stable) by the clear-effect rule from `got`, each refit's
    (lo, hi, mean, sd) oldest first (None where not worked out)."""
    if not got or any(g is None for g in got):
        return False, False
    signs = set()
    for _lo, _hi, mean, sd in got:
        low, high = mean - insights.Z95 * sd, mean + insights.Z95 * sd
        if not (low > 0 or high < 0):
            return False, False
        signs.add(1 if mean > 0 else -1)
    steady = len(signs) == 1
    big = noise is not None and abs(got[-1][2]) >= noise - 1e-12
    return steady, steady and not together and big


def effects(opt, checks=()):
    """{output: [Effect]} for every output and continuous row, each judged
    by the clear-effect rule (module docstring; `checks`, the Round-ahead
    checks, give the noise stand-in); {} with fewer than STABLE_REFITS
    Rounds."""
    refits = refit_rows(opt)
    if len(refits) < STABLE_REFITS:
        return {}
    cols = _columns(opt)
    try:
        moved = {n for a, b, _r in insights.moved_pairs(opt) for n in (a, b)}
    except Exception:
        moved = set()
    out = {}
    for name in names(opt):
        per = []
        fits = [insights.fit(opt, name, rows) for _, rows in refits]
        noise = noise_of(opt, name, fits[-1], checks)
        for var, col in cols:
            got = []
            for f, (_, rows) in zip(fits, refits):
                try:
                    got.append(effect_of(opt, f, rows, col))
                except Exception:
                    got.append(None)
            last = got[-1]
            if last is None:
                continue
            lo, hi, mean, sd = last
            low, high = mean - insights.Z95 * sd, mean + insights.Z95 * sd
            signs = [None if g is None else (1 if g[2] > 0 else -1 if g[2] < 0 else 0)
                     for g in got]
            together = var in moved
            steady, ok = judge_effect(got, together, noise)
            per.append(Effect(name, var, lo, hi, mean, low, high, signs, ok, steady, together,
                              noise))
        out[name] = per
    return out


def stable(effect_map, name):
    """The clear effects on `name` (all three parts of the rule), largest
    first."""
    return sorted([e for e in effect_map.get(name, []) if e.stable],
                  key=lambda e: -abs(e.mean))


# ------------------------------------------------------------------ #
#  Before you taste
# ------------------------------------------------------------------ #

def round_start(opt):
    """The index of the open Round's first recorded row, else the number of
    rows: the open Round is forecast from the rows before it."""
    no = opt.pending_batch_no
    for i, b in enumerate(opt.batch_history):
        if no is not None and b == no:
            return i
    return len(opt.Y_history)


def forecast(opt, rows, wanted):
    """{Formulation: {name: Prediction}} for the open Round's `rows` (each
    {'formulation', 'recipe'}), for the outputs `wanted`, from the models
    fitted on the rows recorded before the Round — the models its
    Round-ahead check uses once its results are saved."""
    train = list(range(round_start(opt)))
    fitted = {n: insights.fit(opt, n, train) for n in wanted}
    return insights.round_predictions(opt, fitted, rows)


def top_of(values):
    """The key with the largest value (None for none)."""
    return max(values, key=values.get) if values else None


# ------------------------------------------------------------------ #
#  What next: the next suggestions, previewed on a copy
# ------------------------------------------------------------------ #

class _NoStorage:
    """A storage that keeps nothing: the preview copy never writes."""

    persist_empty_on_init = False
    persist_after_load = False

    def __getattr__(self, _name):
        return lambda *a, **k: None


def preview_copy(opt):
    """A deep copy of `opt` that can never save."""
    # The project's storage is never copied: the copy gets one that keeps
    # nothing (deepcopy's memo maps the one to the other).
    clone = copy.deepcopy(opt, {id(opt.storage): _NoStorage()})
    clone.save = lambda *a, **k: None
    return clone


def preview_next(opt, n):
    """The `n` formulations Generate would make now, without making them:
    ask() on preview_copy, holding the RNG lock with the global RNG
    restored. [recipe]; ask() seeds itself from the project, so the copy
    asks exactly what Generate would."""
    clone = preview_copy(opt)
    with insights.RNG_LOCK, torch.random.fork_rng():
        return clone.ask(n_suggestions=int(n))
