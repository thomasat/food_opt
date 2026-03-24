"""
Expert-Guided Variable Selection Simulation on Hartmann6.

Setup:
- True function: Hartmann6 (d=6 relevant variables)
- Embedded in N=100 dimensions (94 irrelevant variables)
- The objective only depends on variables 0..5; variables 6..99 are inert.

Expert model:
- At each BO iteration, for each relevant variable NOT yet in the expert's
  active set, add it with probability p.
- For each irrelevant variable NOT yet in the active set, add it with
  probability q.
- High p (good at finding relevant) and low q (avoids noise) = good expert.

Comparisons (T=20 total evaluations, n_init=5 Sobol):
1. vanilla_bo    – BO on all N=100 dimensions
2. oracle        – BO on the 6 true dimensions only
3. rembo         – Random Embedding BO (target_dim=6)
4. turbo         – Trust Region BO on N=100
5. saasbo        – Sparse Axis-Aligned Subspace BO on N=100
6. expert_good   – p=0.5, q=0.005
7. expert_medium – p=0.2, q=0.03
8. expert_poor   – p=0.05, q=0.08

Usage:
    python experiments/run_expert_simulation.py [--seeds 10] [--budget 20]
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from botorch.models import SingleTaskGP, SaasFullyBayesianSingleTaskGP
from botorch.fit import fit_gpytorch_mll, fit_fully_bayesian_model_nuts
from botorch.optim import optimize_acqf
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.models.transforms.outcome import Standardize
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

# ---------------------------------------------------------------------------
# 1. HARTMANN6 EMBEDDED IN 100-D
# ---------------------------------------------------------------------------

D_RELEVANT = 6     # true Hartmann6 dimensions
D_TOTAL = 100       # total ambient dimensions
D_IRRELEVANT = D_TOTAL - D_RELEVANT

# Standard Hartmann6 coefficients
_HARTMANN6_ALPHA = np.array([1.0, 1.2, 3.0, 3.2])
_HARTMANN6_A = np.array([
    [10, 3, 17, 3.5, 1.7, 8],
    [0.05, 10, 17, 0.1, 8, 14],
    [3, 3.5, 1.7, 10, 17, 8],
    [17, 8, 0.05, 10, 0.1, 14],
])
_HARTMANN6_P = 1e-4 * np.array([
    [1312, 1696, 5569, 124, 8283, 5886],
    [2329, 4135, 8307, 3736, 1004, 9991],
    [2348, 1451, 3522, 2883, 3047, 6650],
    [4047, 8828, 8732, 5743, 1091, 381],
])

# Global optimum of Hartmann6 ≈ 3.3224 (negated, so max ≈ 3.3224)
HARTMANN6_GLOBAL_MAX = 3.3224


def hartmann6(x6: np.ndarray) -> float:
    """Standard Hartmann6 on [0,1]^6. Returns positive value (maximisation)."""
    outer = 0.0
    for i in range(4):
        inner = 0.0
        for j in range(6):
            inner += _HARTMANN6_A[i, j] * (x6[j] - _HARTMANN6_P[i, j]) ** 2
        outer += _HARTMANN6_ALPHA[i] * np.exp(-inner)
    return float(outer)


def evaluate_embedded(x100: np.ndarray, noise_std: float = 0.0) -> float:
    """Evaluate Hartmann6 using only the first 6 of 100 variables."""
    val = hartmann6(x100[:D_RELEVANT])
    if noise_std > 0:
        val += np.random.normal(0, noise_std)
    return val


# All variables live in [0, 1]^100
BOUNDS_100 = torch.stack([
    torch.zeros(D_TOTAL, dtype=torch.double),
    torch.ones(D_TOTAL, dtype=torch.double),
])

RELEVANT_SET = set(range(D_RELEVANT))        # {0,1,2,3,4,5}
IRRELEVANT_SET = set(range(D_RELEVANT, D_TOTAL))  # {6,...,99}


# ---------------------------------------------------------------------------
# 2. EXPERT VARIABLE SELECTION MODEL
# ---------------------------------------------------------------------------

@dataclass
class ExpertConfig:
    name: str
    p: float   # prob of adding each missing relevant variable per iteration
    q: float   # prob of adding each missing irrelevant variable per iteration


EXPERT_CONFIGS = {
    "expert_good":   ExpertConfig("expert_good",   p=0.5,  q=0.005),
    "expert_medium": ExpertConfig("expert_medium",  p=0.2,  q=0.03),
    "expert_poor":   ExpertConfig("expert_poor",    p=0.05, q=0.08),
}


def expert_update_variables(
    current_vars: set,
    config: ExpertConfig,
    rng: np.random.RandomState,
) -> set:
    """One step of expert variable selection.

    For each relevant variable NOT in current_vars, add with probability p.
    For each irrelevant variable NOT in current_vars, add with probability q.
    Returns the updated set.
    """
    new_vars = set(current_vars)

    # Relevant variables not yet selected
    missing_relevant = RELEVANT_SET - current_vars
    for v in missing_relevant:
        if rng.random() < config.p:
            new_vars.add(v)

    # Irrelevant variables not yet selected
    missing_irrelevant = IRRELEVANT_SET - current_vars
    for v in missing_irrelevant:
        if rng.random() < config.q:
            new_vars.add(v)

    return new_vars


# ---------------------------------------------------------------------------
# 3. BO HELPERS
# ---------------------------------------------------------------------------

def _fit_and_suggest(
    train_X_norm: torch.Tensor,
    train_Y: torch.Tensor,
    d: int,
    seed: int,
) -> torch.Tensor:
    """Fit GP and return next candidate (normalised)."""
    gp = SingleTaskGP(train_X_norm, train_Y, outcome_transform=Standardize(m=1))
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    try:
        fit_gpytorch_mll(mll)
    except Exception:
        pass

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))
    acq = qLogNoisyExpectedImprovement(
        model=gp, X_baseline=train_X_norm, sampler=sampler,
    )
    candidate_norm, _ = optimize_acqf(
        acq_function=acq,
        bounds=torch.stack([torch.zeros(d), torch.ones(d)]).double(),
        q=1,
        num_restarts=10,
        raw_samples=512,
    )
    return candidate_norm.squeeze(0)


# ---------------------------------------------------------------------------
# 4. METHOD IMPLEMENTATIONS
# ---------------------------------------------------------------------------

def run_vanilla_bo(budget: int, n_init: int, seed: int,
                   noise_std: float) -> Dict:
    """Vanilla BO on all 100 dimensions."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    d = D_TOTAL
    all_x, all_y, best_values = [], [], []

    sobol = SobolEngine(dimension=d, scramble=True, seed=seed)
    x_init = sobol.draw(n_init).double()  # already in [0,1]^d

    for i in range(n_init):
        x = x_init[i].numpy()
        y = evaluate_embedded(x, noise_std)
        all_x.append(x)
        all_y.append(y)
        best_values.append(max(all_y))

    for step in range(budget - n_init):
        train_X = torch.tensor(np.array(all_x), dtype=torch.double)
        train_Y = torch.tensor(all_y, dtype=torch.double).unsqueeze(-1)
        cand = _fit_and_suggest(train_X, train_Y, d, seed + step)
        x_new = cand.detach().numpy()
        y_new = evaluate_embedded(x_new, noise_std)
        all_x.append(x_new)
        all_y.append(y_new)
        best_values.append(max(all_y))

    return {"best_values": best_values, "all_y": all_y, "final_best": max(all_y)}


def run_oracle_bo(budget: int, n_init: int, seed: int,
                  noise_std: float) -> Dict:
    """BO on only the 6 true Hartmann6 dimensions."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    d = D_RELEVANT
    all_x, all_y, best_values = [], [], []

    sobol = SobolEngine(dimension=d, scramble=True, seed=seed)
    x_init = sobol.draw(n_init).double()

    for i in range(n_init):
        x6 = x_init[i].numpy()
        y = hartmann6(x6) + (np.random.normal(0, noise_std) if noise_std > 0 else 0)
        all_x.append(x6)
        all_y.append(y)
        best_values.append(max(all_y))

    for step in range(budget - n_init):
        train_X = torch.tensor(np.array(all_x), dtype=torch.double)
        train_Y = torch.tensor(all_y, dtype=torch.double).unsqueeze(-1)
        cand = _fit_and_suggest(train_X, train_Y, d, seed + step)
        x_new = cand.detach().numpy()
        y_new = hartmann6(x_new) + (np.random.normal(0, noise_std) if noise_std > 0 else 0)
        all_x.append(x_new)
        all_y.append(y_new)
        best_values.append(max(all_y))

    return {"best_values": best_values, "all_y": all_y, "final_best": max(all_y)}


def run_rembo(budget: int, n_init: int, seed: int,
              noise_std: float, target_dim: int = D_RELEVANT) -> Dict:
    """REMBO on N=100, embedding to target_dim."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    d = D_TOTAL
    k = target_dim

    rng = np.random.RandomState(seed)
    A = torch.tensor(rng.randn(d, k) / np.sqrt(k), dtype=torch.double)
    x_center = torch.full((d,), 0.5, dtype=torch.double)

    # Find z-bounds by sampling
    sobol = SobolEngine(dimension=d, scramble=True, seed=seed)
    x_samp = sobol.draw(4096).double()
    z_samp = (x_samp - x_center) @ A
    z_min = z_samp.min(dim=0).values
    z_max = z_samp.max(dim=0).values
    z_bounds = torch.stack([z_min, z_max])

    def z_to_x(z):
        x = x_center + A @ z
        return torch.clamp(x, 0.0, 1.0).detach().numpy()

    all_z, all_y, best_values = [], [], []

    sobol_z = SobolEngine(dimension=k, scramble=True, seed=seed)
    z_init = z_min + sobol_z.draw(n_init).double() * (z_max - z_min)

    for i in range(n_init):
        x = z_to_x(z_init[i])
        y = evaluate_embedded(x, noise_std)
        all_z.append(z_init[i].numpy())
        all_y.append(y)
        best_values.append(max(all_y))

    for step in range(budget - n_init):
        train_Z = torch.tensor(np.array(all_z), dtype=torch.double)
        train_Y = torch.tensor(all_y, dtype=torch.double).unsqueeze(-1)
        train_Z_norm = normalize(train_Z, z_bounds)

        cand_norm = _fit_and_suggest(train_Z_norm, train_Y, k, seed + step)
        z_new = unnormalize(cand_norm, z_bounds)
        x_new = z_to_x(z_new)
        y_new = evaluate_embedded(x_new, noise_std)

        all_z.append(z_new.detach().numpy())
        all_y.append(y_new)
        best_values.append(max(all_y))

    return {"best_values": best_values, "all_y": all_y, "final_best": max(all_y)}


def run_turbo(budget: int, n_init: int, seed: int,
              noise_std: float) -> Dict:
    """TuRBO on N=100."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    d = D_TOTAL
    bounds = BOUNDS_100

    length = 0.8
    length_min = 0.5 ** 7
    length_max = 1.6
    success_tol = 3
    failure_tol = max(d, 1)
    n_successes = 0
    n_failures = 0

    all_x, all_y, best_values = [], [], []

    sobol = SobolEngine(dimension=d, scramble=True, seed=seed)
    x_init = sobol.draw(n_init).double()

    for i in range(n_init):
        x = x_init[i].numpy()
        y = evaluate_embedded(x, noise_std)
        all_x.append(x)
        all_y.append(y)
        best_values.append(max(all_y))

    best_y = max(all_y)

    for step in range(budget - n_init):
        train_X = torch.tensor(np.array(all_x), dtype=torch.double)
        train_Y = torch.tensor(all_y, dtype=torch.double).unsqueeze(-1)

        gp = SingleTaskGP(train_X, train_Y, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        try:
            fit_gpytorch_mll(mll)
        except Exception:
            pass

        best_idx = int(np.argmax(all_y))
        x_center = train_X[best_idx]

        covar = gp.covar_module
        if hasattr(covar, 'base_kernel'):
            ls = covar.base_kernel.lengthscale.detach().squeeze()
        else:
            ls = covar.lengthscale.detach().squeeze()
        weights = ls / ls.mean()
        weights = weights / torch.prod(weights).pow(1.0 / d)

        tr_lb = torch.clamp(x_center - weights * length / 2, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + weights * length / 2, 0.0, 1.0)

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))
        acq = qLogNoisyExpectedImprovement(
            model=gp, X_baseline=train_X, sampler=sampler,
        )
        candidate, _ = optimize_acqf(
            acq_function=acq,
            bounds=torch.stack([tr_lb, tr_ub]).double(),
            q=1, num_restarts=10, raw_samples=512,
        )

        x_new = candidate.squeeze(0).detach().numpy()
        y_new = evaluate_embedded(x_new, noise_std)
        all_x.append(x_new)
        all_y.append(y_new)
        best_values.append(max(all_y))

        if y_new > best_y:
            n_successes += 1
            n_failures = 0
            best_y = y_new
        else:
            n_successes = 0
            n_failures += 1

        if n_successes >= success_tol:
            length = min(2.0 * length, length_max)
            n_successes = 0
        elif n_failures >= failure_tol:
            length /= 2.0
            n_failures = 0
        if length < length_min:
            length = 0.8
            n_successes = 0
            n_failures = 0

    return {"best_values": best_values, "all_y": all_y, "final_best": max(all_y)}


def run_saasbo(budget: int, n_init: int, seed: int,
               noise_std: float) -> Dict:
    """SAASBO on N=100."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    d = D_TOTAL
    all_x, all_y, best_values = [], [], []

    sobol = SobolEngine(dimension=d, scramble=True, seed=seed)
    x_init = sobol.draw(n_init).double()

    for i in range(n_init):
        x = x_init[i].numpy()
        y = evaluate_embedded(x, noise_std)
        all_x.append(x)
        all_y.append(y)
        best_values.append(max(all_y))

    for step in range(budget - n_init):
        train_X = torch.tensor(np.array(all_x), dtype=torch.double)
        train_Y = torch.tensor(all_y, dtype=torch.double).unsqueeze(-1)

        gp = SaasFullyBayesianSingleTaskGP(train_X=train_X, train_Y=train_Y)
        try:
            fit_fully_bayesian_model_nuts(
                gp, warmup_steps=256, num_samples=128, disable_progbar=True,
            )
        except Exception:
            gp = SingleTaskGP(train_X, train_Y, outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            try:
                fit_gpytorch_mll(mll)
            except Exception:
                pass

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))
        acq = qLogNoisyExpectedImprovement(
            model=gp, X_baseline=train_X, sampler=sampler,
        )
        candidate, _ = optimize_acqf(
            acq_function=acq,
            bounds=torch.stack([torch.zeros(d), torch.ones(d)]).double(),
            q=1, num_restarts=10, raw_samples=512,
        )

        x_new = candidate.squeeze(0).detach().numpy()
        y_new = evaluate_embedded(x_new, noise_std)
        all_x.append(x_new)
        all_y.append(y_new)
        best_values.append(max(all_y))

    return {"best_values": best_values, "all_y": all_y, "final_best": max(all_y)}


def run_expert_bo(budget: int, n_init: int, seed: int,
                  noise_std: float, config: ExpertConfig) -> Dict:
    """Expert-guided BO.

    The expert starts with an empty active variable set.  Before every
    BO iteration (including the initial Sobol phase) the expert updates
    the set.  BO is then run only over the active variables; inactive
    variables are fixed at 0.5 (midpoint of [0,1]).
    """
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    np.random.seed(seed)

    active_vars: set = set()
    all_x_full = []   # full 100-d points (for evaluation)
    all_y = []
    best_values = []

    # We also track the active-set trajectory for diagnostics
    active_set_sizes = []
    n_relevant_selected = []

    for t in range(budget):
        # --- Expert updates variable set ---
        active_vars = expert_update_variables(active_vars, config, rng)
        active_list = sorted(active_vars)
        d_active = len(active_list)

        active_set_sizes.append(d_active)
        n_relevant_selected.append(len(RELEVANT_SET & active_vars))

        if d_active == 0:
            # Expert hasn't selected anything yet – sample randomly in full space
            x_full = rng.random(D_TOTAL)
            y = evaluate_embedded(x_full, noise_std)
            all_x_full.append(x_full)
            all_y.append(y)
            best_values.append(max(all_y))
            continue

        # --- Map stored full-d points to active subspace ---
        # Only use points from iterations where active_list was a subset of
        # what we've collected (all of them, we just slice)
        X_active_hist = np.array([x[active_list] for x in all_x_full])
        Y_hist = np.array(all_y)

        if t < n_init or len(all_y) < 2:
            # Sobol initialisation in active subspace
            sobol = SobolEngine(dimension=d_active, scramble=True,
                                seed=seed + t)
            x_sub = sobol.draw(1).double().squeeze(0).numpy()
        else:
            # BO step in active subspace
            train_X = torch.tensor(X_active_hist, dtype=torch.double)
            train_Y = torch.tensor(Y_hist, dtype=torch.double).unsqueeze(-1)

            try:
                cand = _fit_and_suggest(train_X, train_Y, d_active,
                                        seed + t)
                x_sub = cand.detach().numpy()
            except Exception:
                # Fallback to random if GP fitting fails
                x_sub = rng.random(d_active)

        # Reconstruct full 100-d point (inactive dims fixed at 0.5)
        x_full = np.full(D_TOTAL, 0.5)
        x_full[active_list] = x_sub

        y = evaluate_embedded(x_full, noise_std)
        all_x_full.append(x_full)
        all_y.append(y)
        best_values.append(max(all_y))

    return {
        "best_values": best_values,
        "all_y": all_y,
        "final_best": max(all_y),
        "active_set_sizes": active_set_sizes,
        "n_relevant_selected": n_relevant_selected,
    }


# ---------------------------------------------------------------------------
# 5. RUNNER & PLOTTING
# ---------------------------------------------------------------------------

ALL_METHODS = [
    "vanilla_bo", "oracle", "rembo", "turbo", "saasbo",
    "expert_good", "expert_medium", "expert_poor",
]

METHOD_STYLES = {
    "vanilla_bo":     {"color": "grey",      "ls": "--", "label": "Vanilla BO (d=100)"},
    "oracle":         {"color": "black",     "ls": "-",  "label": "Oracle (d=6)"},
    "rembo":          {"color": "tab:blue",  "ls": "-.", "label": "REMBO (k=6)"},
    "turbo":          {"color": "tab:cyan",  "ls": "-.", "label": "TuRBO (d=100)"},
    "saasbo":         {"color": "tab:purple","ls": "-.", "label": "SAASBO (d=100)"},
    "expert_good":    {"color": "tab:green", "ls": "-",  "label": "Expert good (p=.5, q=.005)"},
    "expert_medium":  {"color": "tab:orange","ls": "-",  "label": "Expert medium (p=.2, q=.03)"},
    "expert_poor":    {"color": "tab:red",   "ls": "-",  "label": "Expert poor (p=.05, q=.08)"},
}


def run_simulation(
    budget: int = 20,
    n_init: int = 5,
    n_seeds: int = 10,
    noise_std: float = 0.0,
    methods_to_run: Optional[List[str]] = None,
) -> Dict:
    """Run the full expert simulation benchmark.

    Returns:
        dict of method_name -> {mean_trace, std_trace, all_traces, ...}
    """
    if methods_to_run is None:
        methods_to_run = ALL_METHODS

    results = {}

    for method in methods_to_run:
        print(f"  Running {method} ...", flush=True)
        traces = []
        extra = {}

        for seed in range(n_seeds):
            if method == "vanilla_bo":
                r = run_vanilla_bo(budget, n_init, seed, noise_std)
            elif method == "oracle":
                r = run_oracle_bo(budget, n_init, seed, noise_std)
            elif method == "rembo":
                r = run_rembo(budget, n_init, seed, noise_std)
            elif method == "turbo":
                r = run_turbo(budget, n_init, seed, noise_std)
            elif method == "saasbo":
                r = run_saasbo(budget, n_init, seed, noise_std)
            elif method in EXPERT_CONFIGS:
                r = run_expert_bo(budget, n_init, seed, noise_std,
                                  EXPERT_CONFIGS[method])
                # Collect diagnostics from last seed for plotting
                extra["active_set_sizes"] = r.get("active_set_sizes")
                extra["n_relevant_selected"] = r.get("n_relevant_selected")
            else:
                raise ValueError(f"Unknown method: {method}")

            traces.append(r["best_values"])
            print(f"    seed {seed}: final_best = {r['final_best']:.4f}")

        traces = np.array(traces)
        results[method] = {
            "mean_trace": traces.mean(axis=0).tolist(),
            "std_trace": traces.std(axis=0).tolist(),
            "all_traces": traces.tolist(),
            "mean_final": float(traces[:, -1].mean()),
            "std_final": float(traces[:, -1].std()),
            **extra,
        }

    return results


def plot_convergence(results: Dict, budget: int, save_path: str = "plots/expert_simulation_convergence.png"):
    """Plot convergence curves with ±1 std shading."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    iterations = np.arange(1, budget + 1)

    for method, data in results.items():
        style = METHOD_STYLES.get(method, {"color": "grey", "ls": "-", "label": method})
        mean = np.array(data["mean_trace"])
        std = np.array(data["std_trace"])
        ax.plot(iterations, mean, color=style["color"], ls=style["ls"],
                label=style["label"], linewidth=2)
        ax.fill_between(iterations, mean - std, mean + std,
                         color=style["color"], alpha=0.12)

    ax.axhline(HARTMANN6_GLOBAL_MAX, color="gold", ls=":", lw=1.5,
               label=f"Global optimum ({HARTMANN6_GLOBAL_MAX:.4f})")
    ax.set_xlabel("Evaluation", fontsize=13)
    ax.set_ylabel("Best Hartmann6 Value Found", fontsize=13)
    ax.set_title("Expert-Guided BO vs Baselines — Hartmann6 in 100-D", fontsize=14)
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"  Saved convergence plot to {save_path}")
    plt.close(fig)


def plot_expert_diagnostics(results: Dict, budget: int,
                            save_path: str = "plots/expert_simulation_diagnostics.png"):
    """Plot expert variable-set growth (relevant vs total active)."""
    expert_methods = [m for m in results if m.startswith("expert_")]
    if not expert_methods:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    iterations = np.arange(1, budget + 1)

    for method in expert_methods:
        data = results[method]
        style = METHOD_STYLES.get(method, {"color": "grey", "ls": "-", "label": method})
        if "active_set_sizes" in data and data["active_set_sizes"] is not None:
            axes[0].plot(iterations, data["active_set_sizes"],
                         color=style["color"], ls=style["ls"],
                         label=style["label"], linewidth=2)
            axes[1].plot(iterations, data["n_relevant_selected"],
                         color=style["color"], ls=style["ls"],
                         label=style["label"], linewidth=2)

    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Total Active Variables")
    axes[0].set_title("Active Set Size Over Time")
    axes[0].axhline(D_RELEVANT, color="black", ls=":", lw=1, label="True relevant (6)")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("# Relevant Variables Selected")
    axes[1].set_title("Relevant Variable Recovery")
    axes[1].axhline(D_RELEVANT, color="black", ls=":", lw=1, label="All 6 relevant")
    axes[1].set_ylim(-0.5, D_RELEVANT + 0.5)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"  Saved diagnostics plot to {save_path}")
    plt.close(fig)


def plot_final_bar(results: Dict,
                   save_path: str = "plots/expert_simulation_final.png"):
    """Bar chart of final best values."""
    methods = list(results.keys())
    means = [results[m]["mean_final"] for m in methods]
    stds = [results[m]["std_final"] for m in methods]
    colors = [METHOD_STYLES.get(m, {}).get("color", "grey") for m in methods]
    labels = [METHOD_STYLES.get(m, {}).get("label", m) for m in methods]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(methods))
    ax.bar(x, means, yerr=stds, color=colors, alpha=0.8, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Final Best Hartmann6 Value", fontsize=12)
    ax.set_title("Final Performance Comparison", fontsize=14)
    ax.axhline(HARTMANN6_GLOBAL_MAX, color="gold", ls=":", lw=1.5,
               label=f"Global optimum ({HARTMANN6_GLOBAL_MAX:.4f})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"  Saved bar chart to {save_path}")
    plt.close(fig)


def print_results(results: Dict):
    """Pretty-print final results table."""
    print("\n" + "=" * 72)
    print("EXPERT SIMULATION RESULTS — Hartmann6 in 100-D")
    print("=" * 72)
    print(f"\n{'Method':<30} {'Final Best (mean ± std)':<25} {'Gap to opt'}")
    print("-" * 72)

    for method, data in sorted(results.items(), key=lambda x: -x[1]["mean_final"]):
        label = METHOD_STYLES.get(method, {}).get("label", method)
        perf = f"{data['mean_final']:.4f} ± {data['std_final']:.4f}"
        gap = HARTMANN6_GLOBAL_MAX - data["mean_final"]
        print(f"{label:<30} {perf:<25} {gap:+.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Expert-guided BO simulation on Hartmann6 (d=6 in N=100)")
    parser.add_argument("--budget", type=int, default=20,
                        help="Total evaluations per run (T)")
    parser.add_argument("--n-init", type=int, default=5,
                        help="Initial Sobol samples")
    parser.add_argument("--seeds", type=int, default=10,
                        help="Number of random seeds")
    parser.add_argument("--noise", type=float, default=0.0,
                        help="Observation noise std (0 = noiseless)")
    parser.add_argument("--methods", nargs="+", default=None,
                        choices=ALL_METHODS,
                        help="Methods to run (default: all)")
    args = parser.parse_args()

    os.makedirs("plots", exist_ok=True)

    print(f"Hartmann6 in 100-D expert simulation")
    print(f"  budget={args.budget}, n_init={args.n_init}, "
          f"seeds={args.seeds}, noise={args.noise}")
    print()

    results = run_simulation(
        budget=args.budget,
        n_init=args.n_init,
        n_seeds=args.seeds,
        noise_std=args.noise,
        methods_to_run=args.methods,
    )

    print_results(results)

    plot_convergence(results, args.budget)
    plot_expert_diagnostics(results, args.budget)
    plot_final_bar(results)
