"""The GP model and the reproducible acquisition optimizer: a port of
food_bo.py validate_bo_config / _build_covar / _build_gp / _build_acqf /
_pick_restart / _top_starts / _optimize_reproducibly and the model half of
_ask_optimize."""
from __future__ import annotations

import math

import torch
from botorch.acquisition import (
    qLogExpectedImprovement,
    qLogNoisyExpectedImprovement,
)
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Warp
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.optim.initializers import gen_batch_initial_conditions
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.kernels import (
    LinearKernel,
    MaternKernel,
    PolynomialKernel,
    RBFKernel,
    ScaleKernel,
)
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior

_KERNELS = {"matern52", "matern32", "rbf", "linear", "poly2"}
_LENGTHSCALE = {"default", "long", "short"}
_NOISE = {"default", "low", "fixed_tiny"}
_ACQ = {"qlognei", "qlogei", "qucb"}
_LS_PRIORS = {"default": (3.0, 6.0), "long": (3.0, 1.0), "short": (3.0, 12.0)}

# Two restarts of the acquisition optimiser whose values agree to this
# relative tolerance are the same quality of suggestion; the candidates,
# compared rounded to ACQ_TIE_DECIMALS (normalised units), decide.
ACQ_TIE_RTOL = 1e-4
ACQ_TIE_DECIMALS = 3


def check_bo_config(spec):
    """Refuse an unknown key or value (validate_bo_config would silently
    replace it with the default)."""
    if not spec:
        return
    if not isinstance(spec, dict):
        raise ValueError(f"bo_config must be a dict, not {spec!r}")
    allowed = {"kernel": _KERNELS, "lengthscale_prior": _LENGTHSCALE,
               "noise": _NOISE, "acquisition": _ACQ}
    for key, value in spec.items():
        if key not in allowed:
            raise ValueError(f"bo_config: unknown setting '{key}'; use "
                             f"{', '.join(sorted(allowed))}")
        if str(value).strip().lower() not in allowed[key]:
            raise ValueError(f"bo_config: {key} {value!r} is not one of "
                             f"{', '.join(sorted(allowed[key]))}")


def validate_bo_config(spec):
    """Coerce a raw config dict to valid values. Returns None for an empty/None
    spec (None => library defaults, identical to the standard arm)."""
    if not spec:
        return None

    def pick(key, allowed, default):
        v = str(spec.get(key, default)).strip().lower()
        return v if v in allowed else default

    return {
        "kernel": pick("kernel", _KERNELS, "matern52"),
        "lengthscale_prior": pick("lengthscale_prior", _LENGTHSCALE, "default"),
        "noise": pick("noise", _NOISE, "default"),
        "acquisition": pick("acquisition", _ACQ, "qlognei"),
    }


def _build_covar(cfg, dim):
    kernel = cfg.get("kernel", "matern52")
    if kernel in ("matern52", "matern32"):
        nu = 2.5 if kernel == "matern52" else 1.5
        conc, rate = _LS_PRIORS.get(cfg.get("lengthscale_prior", "default"), _LS_PRIORS["default"])
        base = MaternKernel(nu=nu, ard_num_dims=dim, lengthscale_prior=GammaPrior(conc, rate))
    elif kernel == "rbf":
        conc, rate = _LS_PRIORS.get(cfg.get("lengthscale_prior", "default"), _LS_PRIORS["default"])
        base = RBFKernel(ard_num_dims=dim, lengthscale_prior=GammaPrior(conc, rate))
    elif kernel == "linear":
        base = LinearKernel()
    elif kernel == "poly2":
        base = PolynomialKernel(power=2)
    else:
        base = MaternKernel(nu=2.5, ard_num_dims=dim)
    return ScaleKernel(base)


def pick_restart(candidates, values):
    """The restart a suggestion is taken from: of those within ACQ_TIE_RTOL
    of the best value (a floor of 1 in absolute terms, for log values near
    zero), the lexicographically smallest candidate."""
    # A NaN value is the worst a restart can be (never the best, never tied);
    # a +inf is the best, and among several the candidates decide.
    values = torch.nan_to_num(values.detach().reshape(-1), nan=-math.inf)
    best = float(values.max())
    if math.isinf(best):
        tied = [j for j in range(len(values)) if float(values[j]) == best]
        if not tied:
            tied = [int(values.argmax())]
    else:
        floor = best - ACQ_TIE_RTOL * max(1.0, abs(best))
        tied = [j for j in range(len(values)) if float(values[j]) >= floor]
    return min(tied, key=lambda j: tuple(
        round(float(x), ACQ_TIE_DECIMALS) for x in candidates[j].reshape(-1)))


def _top_starts(*args, options=None, **kwargs):
    """optimize_acqf's starting points: the best raw samples, not a
    Boltzmann draw among them that a last-bit change could flip."""
    return gen_batch_initial_conditions(
        *args, options={**(options or {}), "topn": True}, **kwargs)


def build_gp(train_X_norm, train_Y, dim, robust, bo_config):
    """Build the GP. With bo_config falsy this is the default GP; a config
    swaps in the expert's kernel/noise."""
    input_tf = Warp(d=dim, indices=list(range(dim))) if robust else None
    cfg = bo_config
    if not cfg:
        return SingleTaskGP(
            train_X_norm, train_Y,
            outcome_transform=Standardize(m=1), input_transform=input_tf,
        )
    kwargs = dict(
        outcome_transform=Standardize(m=1), input_transform=input_tf,
        covar_module=_build_covar(cfg, dim),
    )
    if cfg.get('noise') == 'fixed_tiny':
        yvar = float(max(1e-8, 1e-6 * (train_Y.var().item() + 1e-12)))
        return SingleTaskGP(
            train_X_norm, train_Y,
            train_Yvar=torch.full_like(train_Y, yvar), **kwargs,
        )
    gp = SingleTaskGP(train_X_norm, train_Y, **kwargs)
    if cfg.get('noise') == 'low':
        gp.likelihood.noise_covar.register_prior(
            'noise_prior', GammaPrior(1.1, 50.0), 'raw_noise',
        )
    return gp


def build_acqf(gp, train_X_norm, train_Y, bo_config):
    acq = (bo_config or {}).get('acquisition', 'qlognei')
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]))
    if acq == 'qlogei':
        return qLogExpectedImprovement(
            model=gp, best_f=float(train_Y.max().item()), sampler=sampler,
        )
    if acq == 'qucb':
        return qUpperConfidenceBound(model=gp, beta=0.2, sampler=sampler)
    return qLogNoisyExpectedImprovement(
        model=gp, X_baseline=train_X_norm, sampler=sampler,
    )


def optimize_reproducibly(acq_func, q, dim, inequality_constraints, fixed_features):
    """optimize_acqf(sequential=True), made so that a last-bit change in
    the numbers cannot move a suggestion: every restart is its own scipy
    problem, restarts start from the best raw samples (_top_starts), and of
    restarts with near-equal values the candidate decides (pick_restart).
    One suggestion at a time, each optimised with the ones before it
    pending, exactly as BoTorch's sequential loop does."""
    options = dict(
        bounds=torch.stack([torch.zeros(dim), torch.ones(dim)]).double(),
        q=1,
        num_restarts=20,
        raw_samples=1024,
        return_best_only=False,
        inequality_constraints=inequality_constraints,
        fixed_features=fixed_features or None,
        options={"max_optimization_problem_aggregation_size": 1},
        ic_generator=_top_starts,
    )
    base_pending = acq_func.X_pending
    chosen = []
    for _ in range(q):
        if chosen:
            pending = torch.cat(chosen, dim=-2)
            if base_pending is not None:
                pending = torch.cat([base_pending, pending], dim=-2)
            acq_func.set_X_pending(pending)
        candidates, values = optimize_acqf(acq_function=acq_func, **options)
        pick = pick_restart(candidates, values)
        chosen.append(candidates[pick].detach().reshape(1, -1))
    acq_func.set_X_pending(base_pending)
    return torch.cat(chosen, dim=-2)


def suggest(train_X, train_Y, bounds, q, seed, *, robust=False, bo_config=None,
            inequality_constraints=None, fixed_features=None):
    """q suggestions in raw units, shape (q, d): the model half of
    _ask_optimize. bounds is (2, d) raw; the constraints and fixed features
    are over the normalised [0, 1] frame."""
    # The seeding is local: the caller's global torch RNG is left as it was.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        dim = train_X.shape[-1]
        train_X_norm = normalize(train_X, bounds)
        gp = build_gp(train_X_norm, train_Y, dim, robust, bo_config)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        acq_func = build_acqf(gp, train_X_norm, train_Y, bo_config)
        candidate_norm = optimize_reproducibly(
            acq_func, q, dim,
            [] if inequality_constraints is None else inequality_constraints,
            fixed_features)
        return unnormalize(candidate_norm, bounds).detach()
