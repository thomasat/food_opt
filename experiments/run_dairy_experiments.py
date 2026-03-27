"""
Expanded dairy-category optimization experiments.

Goal: for each dairy category, minimize Euclidean distance in normalized
(fat, sodium) space between a plant-based formulation and the animal-based
reference product.

Same 28 ingredients (ingredients_dairy.csv) across all categories.
Normalization: fat_norm = fat_per_100g / 100, sodium_norm = sodium_per_100g / 1000.

Methods compared:
  1. LLM + BO (k=5):   LLM selects 5 ingredients, BO optimizes those fractions
  2. Vanilla BO (d=28): BO over all 28 ingredient fractions
  3. Random Search:     uniform random on the simplex
  4. TuRBO (d=28):      trust-region BO
  5. SAASBO (d=28):     sparse axis-aligned subspace BO
  6. REMBO (d=28->k=5): random embedding BO

Usage:
    python run_dairy_experiments.py [--budget 30] [--seeds 5] [--no-plot]
    python run_dairy_experiments.py --categories mozzarella cheddar --budget 20
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

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
# 1. DATA
# ---------------------------------------------------------------------------

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_ingredients(
    path: str = os.path.join(_ROOT, "data", "ingredients_dairy.csv"),
) -> pd.DataFrame:
    return pd.read_csv(path)


def get_nutrition_arrays(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    fat_norm = df["fat_per_100g"].values / 100.0
    sodium_norm = df["sodium_per_100g"].values / 1000.0
    return fat_norm, sodium_norm


# ---------------------------------------------------------------------------
# 2. DAIRY CATEGORY TARGETS  (per 100 g of animal-based product)
# ---------------------------------------------------------------------------
# Each entry: (fat_per_100g, sodium_per_100g)

DAIRY_TARGETS: Dict[str, Tuple[float, float]] = {
    "barista milk":   (3.5,  44.0),
    "butter":         (81.0, 625.0),
    "cheddar":        (33.0, 620.0),
    "coffee creamer": (10.0, 50.0),
    "cream cheese":   (34.0, 321.0),
    "ice cream":      (11.0, 80.0),
    "milk":           (3.3,  44.0),
    "mozzarella":     (22.0, 500.0),
    "sour cream":     (20.0, 38.0),
    "yogurt":         (3.3,  46.0),
}

# ---------------------------------------------------------------------------
# 3. LLM INGREDIENT SELECTIONS  (Claude's up-front dimensionality reduction)
#
# For each category the 5 indices are chosen to give good coverage of the
# (fat, sodium) target while spanning the reachable region:
#   idx  0  potato starch        fat=0.001  sodium=0.010
#   idx  1  coconut oil          fat=0.990  sodium=0.000
#   idx  2  tapioca starch       fat=0.000  sodium=0.000
#   idx  3  pea protein          fat=0.020  sodium=0.800
#   idx  4  canola oil           fat=1.000  sodium=0.000
#   idx  6  potato protein       fat=0.010  sodium=0.200
#   idx 11  pea milk             fat=0.015  sodium=0.120
#   idx 12  oat flour            fat=0.070  sodium=0.003
#   idx 13  oat cream            fat=0.100  sodium=0.040
#   idx 15  carrageenan          fat=0.000  sodium=0.700
#   idx 16  mushroom extract     fat=0.003  sodium=0.030
#   idx 18  cashew milk          fat=0.020  sodium=0.080
#   idx 20  oatmilk              fat=0.015  sodium=0.100
#   idx 22  cashews              fat=0.440  sodium=0.012
#   idx 24  soy protein isolate  fat=0.005  sodium=1.000
#   idx 25  soybeans             fat=0.200  sodium=0.002
#   idx 26  soymilk              fat=0.018  sodium=0.050
# ---------------------------------------------------------------------------

LLM_SELECTIONS: Dict[str, List[int]] = {
    # barista milk  target (0.035, 0.044) – low fat, low sodium
    # oatmilk (close fat), soymilk (close sodium), pea milk (sodium lever),
    # oat flour (fat lever), tapioca starch (neutral filler)
    "barista milk":   [20, 26, 11, 12, 2],

    # butter  target (0.81, 0.625) – very high fat, high sodium
    # coconut oil (fat), canola oil (fat), soy protein isolate (sodium),
    # pea protein (sodium), carrageenan (sodium)
    "butter":         [1, 4, 24, 3, 15],

    # cheddar  target (0.33, 0.62) – high fat, high sodium
    # coconut oil (fat), cashews (moderate fat), soy protein isolate (sodium),
    # pea protein (sodium), tapioca starch (filler)
    "cheddar":        [1, 22, 24, 3, 2],

    # coffee creamer  target (0.10, 0.05) – moderate fat, low sodium
    # oat cream (near-exact match), coconut oil (fat lever),
    # cashew milk (sodium fine-tune), soymilk (sodium), tapioca starch (filler)
    "coffee creamer": [13, 1, 18, 26, 2],

    # cream cheese  target (0.34, 0.321) – high fat, moderate sodium
    # coconut oil (fat), cashews (moderate fat), pea protein (sodium),
    # potato protein (sodium), tapioca starch (filler)
    "cream cheese":   [1, 22, 3, 6, 2],

    # ice cream  target (0.11, 0.08) – moderate fat, low sodium
    # coconut oil (fat), oat cream (close), cashew milk (sodium),
    # pea milk (sodium lever), tapioca starch (filler)
    "ice cream":      [1, 13, 18, 11, 2],

    # milk  target (0.033, 0.044) – low fat, low sodium
    # oatmilk, soymilk, pea milk, oat flour, tapioca starch
    "milk":           [20, 26, 11, 12, 2],

    # mozzarella  target (0.22, 0.50) – (original experiment)
    # coconut oil (fat), pea protein (sodium), soy protein isolate (sodium),
    # cashews (moderate fat), tapioca starch (filler)
    "mozzarella":     [1, 3, 24, 22, 2],

    # sour cream  target (0.20, 0.038) – high fat, very low sodium
    # coconut oil (fat), soybeans (fat=0.20, exact match!), cashews (fat),
    # tapioca starch (filler), mushroom extract (fine sodium)
    "sour cream":     [1, 25, 22, 2, 16],

    # yogurt  target (0.033, 0.046) – low fat, low sodium
    # oatmilk, soymilk, oat flour, cashew milk, tapioca starch
    "yogurt":         [20, 26, 12, 18, 2],
}


# ---------------------------------------------------------------------------
# 4. OBJECTIVE
# ---------------------------------------------------------------------------

def make_objective(
    fat_norm: np.ndarray,
    sodium_norm: np.ndarray,
    target_fat_norm: float,
    target_sodium_norm: float,
):
    """Returns callable: x (weight fractions) -> negative L2 distance."""
    def objective(x: np.ndarray) -> float:
        fat_mix = np.dot(x, fat_norm)
        sodium_mix = np.dot(x, sodium_norm)
        dist = np.sqrt(
            (fat_mix - target_fat_norm) ** 2
            + (sodium_mix - target_sodium_norm) ** 2
        )
        return -float(dist)
    return objective


# ---------------------------------------------------------------------------
# 5. SIMPLEX HELPERS
# ---------------------------------------------------------------------------

def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


# ---------------------------------------------------------------------------
# 6. OPTIMISATION METHODS  (identical to run_mozzarella_experiment.py)
# ---------------------------------------------------------------------------

def random_search(objective, d: int, budget: int = 30, seed: int = 0) -> Dict:
    rng = np.random.RandomState(seed)
    best_values, all_y = [], []
    for _ in range(budget):
        x = rng.dirichlet(np.ones(d))
        y = objective(x)
        all_y.append(y)
        best_values.append(max(all_y))
    return {"best_values": best_values, "all_y": all_y, "final_best": max(all_y)}


def vanilla_bo(objective, d: int, budget: int = 30, n_init: int = 5,
               seed: int = 0) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    z_lo, z_hi = -3.0, 3.0
    bounds = torch.tensor([[z_lo]*d, [z_hi]*d], dtype=torch.double)

    all_z, all_y, best_values = [], [], []

    sobol = SobolEngine(dimension=d, scramble=True, seed=seed)
    z_init = unnormalize(sobol.draw(n_init).double(), bounds)
    for i in range(n_init):
        z = z_init[i].numpy()
        y = objective(_softmax(z))
        all_z.append(z); all_y.append(y); best_values.append(max(all_y))

    for _ in range(budget - n_init):
        train_Z = torch.tensor(np.array(all_z), dtype=torch.double)
        train_Y = torch.tensor(all_y, dtype=torch.double).unsqueeze(-1)
        train_Z_norm = normalize(train_Z, bounds)

        gp = SingleTaskGP(train_Z_norm, train_Y, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        try:
            fit_gpytorch_mll(mll)
        except Exception:
            pass

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))
        acq = qLogNoisyExpectedImprovement(model=gp, X_baseline=train_Z_norm, sampler=sampler)
        cand, _ = optimize_acqf(
            acq_function=acq,
            bounds=torch.stack([torch.zeros(d), torch.ones(d)]).double(),
            q=1, num_restarts=10, raw_samples=512,
        )
        z_new = unnormalize(cand.squeeze(0), bounds).detach().numpy()
        y_new = objective(_softmax(z_new))
        all_z.append(z_new); all_y.append(y_new); best_values.append(max(all_y))

    return {"best_values": best_values, "all_y": all_y, "final_best": max(all_y)}


def llm_reduced_bo(objective, d_full: int, selected_indices: List[int],
                   budget: int = 30, n_init: int = 5, seed: int = 0) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    k = len(selected_indices)
    z_lo, z_hi = -3.0, 3.0
    bounds = torch.tensor([[z_lo]*k, [z_hi]*k], dtype=torch.double)

    def z_to_x(z):
        fracs = _softmax(z)
        x = np.zeros(d_full)
        for i, idx in enumerate(selected_indices):
            x[idx] = fracs[i]
        return x

    all_z, all_y, best_values = [], [], []

    sobol = SobolEngine(dimension=k, scramble=True, seed=seed)
    z_init = unnormalize(sobol.draw(n_init).double(), bounds)
    for i in range(n_init):
        z = z_init[i].numpy()
        y = objective(z_to_x(z))
        all_z.append(z); all_y.append(y); best_values.append(max(all_y))

    for _ in range(budget - n_init):
        train_Z = torch.tensor(np.array(all_z), dtype=torch.double)
        train_Y = torch.tensor(all_y, dtype=torch.double).unsqueeze(-1)
        train_Z_norm = normalize(train_Z, bounds)

        gp = SingleTaskGP(train_Z_norm, train_Y, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        try:
            fit_gpytorch_mll(mll)
        except Exception:
            pass

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))
        acq = qLogNoisyExpectedImprovement(model=gp, X_baseline=train_Z_norm, sampler=sampler)
        cand, _ = optimize_acqf(
            acq_function=acq,
            bounds=torch.stack([torch.zeros(k), torch.ones(k)]).double(),
            q=1, num_restarts=10, raw_samples=512,
        )
        z_new = unnormalize(cand.squeeze(0), bounds).detach().numpy()
        y_new = objective(z_to_x(z_new))
        all_z.append(z_new); all_y.append(y_new); best_values.append(max(all_y))

    return {"best_values": best_values, "all_y": all_y, "final_best": max(all_y)}


def rembo_bo(objective, d: int, budget: int = 30, n_init: int = 5,
             seed: int = 0, target_dim: int = 5) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    k = target_dim
    z_lo, z_hi = -3.0, 3.0
    bounds_full = torch.tensor([[z_lo]*d, [z_hi]*d], dtype=torch.double)

    rng = np.random.RandomState(seed)
    A = torch.tensor(rng.randn(d, k) / np.sqrt(k), dtype=torch.double)
    x_center = torch.zeros(d, dtype=torch.double)

    sobol_samples = SobolEngine(dimension=d, scramble=True, seed=seed)
    x_samples = unnormalize(sobol_samples.draw(4096).double(), bounds_full)
    z_samples = (x_samples - x_center) @ A
    z_min = z_samples.min(dim=0).values
    z_max = z_samples.max(dim=0).values
    z_bounds = torch.stack([z_min, z_max])

    def z_to_x(z):
        x_unc = x_center + A @ z
        x_unc = torch.clamp(x_unc, bounds_full[0, 0], bounds_full[1, 0])
        return _softmax(x_unc.detach().numpy())

    all_z, all_y, best_values = [], [], []

    sobol_z = SobolEngine(dimension=k, scramble=True, seed=seed)
    z_init = z_min + sobol_z.draw(n_init).double() * (z_max - z_min)
    for i in range(n_init):
        y = objective(z_to_x(z_init[i]))
        all_z.append(z_init[i].numpy()); all_y.append(y); best_values.append(max(all_y))

    for _ in range(budget - n_init):
        train_Z = torch.tensor(np.array(all_z), dtype=torch.double)
        train_Y = torch.tensor(all_y, dtype=torch.double).unsqueeze(-1)
        train_Z_norm = normalize(train_Z, z_bounds)

        gp = SingleTaskGP(train_Z_norm, train_Y, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        try:
            fit_gpytorch_mll(mll)
        except Exception:
            pass

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))
        acq = qLogNoisyExpectedImprovement(model=gp, X_baseline=train_Z_norm, sampler=sampler)
        cand, _ = optimize_acqf(
            acq_function=acq,
            bounds=torch.stack([torch.zeros(k), torch.ones(k)]).double(),
            q=1, num_restarts=10, raw_samples=512,
        )
        z_new = unnormalize(cand.squeeze(0), z_bounds)
        y_new = objective(z_to_x(z_new))
        all_z.append(z_new.detach().numpy()); all_y.append(y_new); best_values.append(max(all_y))

    return {"best_values": best_values, "all_y": all_y, "final_best": max(all_y)}


def turbo_bo(objective, d: int, budget: int = 30, n_init: int = 5,
             seed: int = 0) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    z_lo, z_hi = -3.0, 3.0
    bounds = torch.tensor([[z_lo]*d, [z_hi]*d], dtype=torch.double)

    length_init, length_min, length_max = 0.8, 0.5 ** 7, 1.6
    success_tolerance, failure_tolerance = 3, max(d, 1)
    length, n_successes, n_failures = length_init, 0, 0

    all_z, all_y, best_values = [], [], []

    sobol = SobolEngine(dimension=d, scramble=True, seed=seed)
    z_init = unnormalize(sobol.draw(n_init).double(), bounds)
    for i in range(n_init):
        z = z_init[i].numpy()
        y = objective(_softmax(z))
        all_z.append(z); all_y.append(y); best_values.append(max(all_y))

    best_y = max(all_y)

    for _ in range(budget - n_init):
        train_Z = torch.tensor(np.array(all_z), dtype=torch.double)
        train_Y = torch.tensor(all_y, dtype=torch.double).unsqueeze(-1)
        train_Z_norm = normalize(train_Z, bounds)

        gp = SingleTaskGP(train_Z_norm, train_Y, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        try:
            fit_gpytorch_mll(mll)
        except Exception:
            pass

        best_idx = int(np.argmax(all_y))
        z_center_norm = train_Z_norm[best_idx]

        covar = gp.covar_module
        if hasattr(covar, "base_kernel"):
            lengthscales = covar.base_kernel.lengthscale.detach().squeeze()
        else:
            lengthscales = covar.lengthscale.detach().squeeze()
        weights = lengthscales / lengthscales.mean()
        weights = weights / torch.prod(weights).pow(1.0 / d)

        tr_lb = torch.clamp(z_center_norm - weights * length / 2, 0.0, 1.0)
        tr_ub = torch.clamp(z_center_norm + weights * length / 2, 0.0, 1.0)

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))
        acq = qLogNoisyExpectedImprovement(model=gp, X_baseline=train_Z_norm, sampler=sampler)
        cand, _ = optimize_acqf(
            acq_function=acq,
            bounds=torch.stack([tr_lb, tr_ub]).double(),
            q=1, num_restarts=10, raw_samples=512,
        )
        z_new = unnormalize(cand.squeeze(0), bounds).detach().numpy()
        y_new = objective(_softmax(z_new))
        all_z.append(z_new); all_y.append(y_new); best_values.append(max(all_y))

        if y_new > best_y:
            n_successes += 1; n_failures = 0; best_y = y_new
        else:
            n_successes = 0; n_failures += 1

        if n_successes >= success_tolerance:
            length = min(2.0 * length, length_max); n_successes = 0
        elif n_failures >= failure_tolerance:
            length = length / 2.0; n_failures = 0

        if length < length_min:
            length = length_init; n_successes = 0; n_failures = 0

    return {"best_values": best_values, "all_y": all_y, "final_best": max(all_y)}


def saasbo_bo(objective, d: int, budget: int = 30, n_init: int = 5,
              seed: int = 0, warmup_steps: int = 256,
              num_samples: int = 128) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    z_lo, z_hi = -3.0, 3.0
    bounds = torch.tensor([[z_lo]*d, [z_hi]*d], dtype=torch.double)

    all_z, all_y, best_values = [], [], []

    sobol = SobolEngine(dimension=d, scramble=True, seed=seed)
    z_init = unnormalize(sobol.draw(n_init).double(), bounds)
    for i in range(n_init):
        z = z_init[i].numpy()
        y = objective(_softmax(z))
        all_z.append(z); all_y.append(y); best_values.append(max(all_y))

    for _ in range(budget - n_init):
        train_Z = torch.tensor(np.array(all_z), dtype=torch.double)
        train_Y = torch.tensor(all_y, dtype=torch.double).unsqueeze(-1)
        train_Z_norm = normalize(train_Z, bounds)

        gp = SaasFullyBayesianSingleTaskGP(train_X=train_Z_norm, train_Y=train_Y)
        try:
            fit_fully_bayesian_model_nuts(
                gp, warmup_steps=warmup_steps, num_samples=num_samples,
                disable_progbar=True,
            )
        except Exception:
            gp = SingleTaskGP(train_Z_norm, train_Y, outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            try:
                fit_gpytorch_mll(mll)
            except Exception:
                pass

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))
        acq = qLogNoisyExpectedImprovement(model=gp, X_baseline=train_Z_norm, sampler=sampler)
        cand, _ = optimize_acqf(
            acq_function=acq,
            bounds=torch.stack([torch.zeros(d), torch.ones(d)]).double(),
            q=1, num_restarts=10, raw_samples=512,
        )
        z_new = unnormalize(cand.squeeze(0), bounds).detach().numpy()
        y_new = objective(_softmax(z_new))
        all_z.append(z_new); all_y.append(y_new); best_values.append(max(all_y))

    return {"best_values": best_values, "all_y": all_y, "final_best": max(all_y)}


# ---------------------------------------------------------------------------
# 7. SINGLE-CATEGORY RUNNER
# ---------------------------------------------------------------------------

def run_single_category(
    category: str,
    df: pd.DataFrame,
    fat_norm: np.ndarray,
    sodium_norm: np.ndarray,
    budget: int = 30,
    n_seeds: int = 5,
    n_init: int = 5,
) -> Dict[str, list]:
    """Run all 6 methods for one dairy category. Returns {method_name: [traces]}."""
    fat_target, sodium_target = DAIRY_TARGETS[category]
    target_fat_norm = fat_target / 100.0
    target_sodium_norm = sodium_target / 1000.0
    d = len(df)

    objective = make_objective(fat_norm, sodium_norm, target_fat_norm, target_sodium_norm)
    selected = LLM_SELECTIONS[category]
    k = len(selected)

    method_names = [
        f"LLM + BO (k={k})",
        f"Vanilla BO (d={d})",
        f"Random Search (d={d})",
        f"TuRBO (d={d})",
        f"SAASBO (d={d})",
        f"REMBO (d={d}->k=5)",
    ]
    all_traces = {name: [] for name in method_names}

    print(f"\n{'='*60}")
    print(f"  {category.upper()}")
    print(f"  target: fat={target_fat_norm:.3f}, sodium={target_sodium_norm:.3f}")
    print(f"  LLM-selected: {[df['name'].iloc[i] for i in selected]}")
    print(f"{'='*60}")

    for seed in range(n_seeds):
        print(f"  seed {seed+1}/{n_seeds}", end="", flush=True)

        res = llm_reduced_bo(objective, d, selected, budget=budget, n_init=n_init, seed=seed)
        all_traces[method_names[0]].append(res["best_values"])

        res = vanilla_bo(objective, d, budget=budget, n_init=n_init, seed=seed)
        all_traces[method_names[1]].append(res["best_values"])

        res = random_search(objective, d, budget=budget, seed=seed)
        all_traces[method_names[2]].append(res["best_values"])

        res = turbo_bo(objective, d, budget=budget, n_init=n_init, seed=seed)
        all_traces[method_names[3]].append(res["best_values"])

        res = saasbo_bo(objective, d, budget=budget, n_init=n_init, seed=seed,
                        warmup_steps=64, num_samples=32)
        all_traces[method_names[4]].append(res["best_values"])

        res = rembo_bo(objective, d, budget=budget, n_init=n_init, seed=seed)
        all_traces[method_names[5]].append(res["best_values"])

        print(" done")

    return all_traces


# ---------------------------------------------------------------------------
# 8. PLOTTING
# ---------------------------------------------------------------------------

_METHOD_STYLE = {
    "LLM":     ("#2ecc71", "-",  2.5),
    "Vanilla": ("#e74c3c", "-.", 2.0),
    "Random":  ("#999999", "--", 1.5),
    "TuRBO":   ("#9b59b6", "-.", 2.0),
    "SAASBO":  ("#e67e22", ":",  2.0),
    "REMBO":   ("#3498db", "--", 2.0),
}


def _style(name: str):
    for key, val in _METHOD_STYLE.items():
        if key in name:
            return val
    return ("#333333", "-", 1.5)


def plot_category_convergence(
    category: str,
    all_traces: Dict[str, list],
    budget: int,
    save_dir: str,
):
    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    iters = np.arange(1, budget + 1)

    for name, traces in all_traces.items():
        arr = np.array(traces)
        mean_dist = -arr.mean(axis=0)
        std_dist = arr.std(axis=0)
        color, ls, lw = _style(name)
        ax.plot(iters, mean_dist, label=name, color=color, linestyle=ls, linewidth=lw)
        ax.fill_between(iters, mean_dist - std_dist, mean_dist + std_dist,
                        alpha=0.15, color=color)

    ax.set_xlabel("Evaluation", fontsize=12)
    ax.set_ylabel("L2 Distance (lower = better)", fontsize=12)
    ax.set_title(f"Plant-Based {category.title()}: Nutrition Matching", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, budget)
    plt.tight_layout()
    path = os.path.join(save_dir, f"dairy_{category.replace(' ', '_')}_convergence.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved {path}")


def plot_category_bar(
    category: str,
    all_traces: Dict[str, list],
    save_dir: str,
):
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    names = list(all_traces.keys())
    means, stds, colors = [], [], []
    for name in names:
        finals = [-t[-1] for t in all_traces[name]]
        means.append(np.mean(finals))
        stds.append(np.std(finals))
        colors.append(_style(name)[0])

    ax.bar(range(len(names)), means, yerr=stds, color=colors,
           capsize=5, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=9)
    ax.set_ylabel("Final L2 Distance (lower = better)", fontsize=11)
    ax.set_title(f"Final Performance: {category.title()}", fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(save_dir, f"dairy_{category.replace(' ', '_')}_final.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved {path}")


def plot_summary_heatmap(
    all_results: Dict[str, Dict[str, list]],
    save_dir: str,
):
    """Heatmap: rows = categories, columns = methods, values = final distance."""
    categories = list(all_results.keys())
    method_names = list(all_results[categories[0]].keys())

    mat = np.zeros((len(categories), len(method_names)))
    for i, cat in enumerate(categories):
        for j, meth in enumerate(method_names):
            finals = [-t[-1] for t in all_results[cat][meth]]
            mat[i, j] = np.mean(finals)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(len(method_names)))
    ax.set_xticklabels([m.replace(" ", "\n") for m in method_names], fontsize=9)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels([c.title() for c in categories], fontsize=10)

    for i in range(len(categories)):
        for j in range(len(method_names)):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if mat[i, j] > mat.mean() else "black")

    ax.set_title("Final L2 Distance by Category & Method (lower = better)", fontsize=13)
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    path = os.path.join(save_dir, "dairy_summary_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# 9. MAIN
# ---------------------------------------------------------------------------

def run_all(
    categories: List[str] | None = None,
    budget: int = 30,
    n_seeds: int = 5,
    n_init: int = 5,
    plot: bool = True,
):
    df = load_ingredients()
    d = len(df)
    fat_norm, sodium_norm = get_nutrition_arrays(df)

    if categories is None:
        categories = list(DAIRY_TARGETS.keys())

    print(f"Ingredients: {d}, Budget: {budget}, Seeds: {n_seeds}")
    print(f"Categories: {categories}\n")

    plot_dir = os.path.join(_ROOT, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    results_dir = os.path.join(_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    all_results: Dict[str, Dict[str, list]] = {}

    for cat in categories:
        traces = run_single_category(
            cat, df, fat_norm, sodium_norm,
            budget=budget, n_seeds=n_seeds, n_init=n_init,
        )
        all_results[cat] = traces

        if plot:
            plot_category_convergence(cat, traces, budget, plot_dir)
            plot_category_bar(cat, traces, plot_dir)

    # Summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    method_names = list(all_results[categories[0]].keys())
    header = f"{'Category':<18}" + "".join(f"{m:<22}" for m in method_names)
    print(header)
    print("-" * len(header))
    for cat in categories:
        row = f"{cat:<18}"
        for meth in method_names:
            finals = [-t[-1] for t in all_results[cat][meth]]
            row += f"{np.mean(finals):.4f} +/- {np.std(finals):.4f}  "
        print(row)

    if plot and len(categories) > 1:
        plot_summary_heatmap(all_results, plot_dir)

    # Save raw results to JSON
    serializable = {}
    for cat, traces in all_results.items():
        serializable[cat] = {
            meth: [[float(v) for v in t] for t in tlist]
            for meth, tlist in traces.items()
        }
    results_path = os.path.join(results_dir, "dairy_experiments.json")
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nRaw results saved to {results_path}")

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Expanded dairy optimization experiments")
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--n-init", type=int, default=5)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--categories", nargs="+", default=None,
        choices=list(DAIRY_TARGETS.keys()),
        help="Run only selected categories (default: all 10)",
    )
    args = parser.parse_args()

    run_all(
        categories=args.categories,
        budget=args.budget,
        n_seeds=args.seeds,
        n_init=args.n_init,
        plot=not args.no_plot,
    )
