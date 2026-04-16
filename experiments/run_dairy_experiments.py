"""
Expanded dairy-category optimization experiments.

Goal: for each dairy category, minimize weighted Euclidean distance in a
multi-nutrient space between a plant-based formulation and the reference
animal-based product.

This version matches the same nutrition dimensions used in the meat
experiments:
  - energy
  - protein
  - fat
  - saturated fat
  - carbs
  - fiber
  - sugars
  - sodium

Ingredients come from data/usda_nutrition_data.csv and are restricted to
rows marked in_dairy == "Y", which corresponds to the expanded dairy
ingredient list from nectar_ingredients.csv enriched with USDA nutrition.

Methods compared:
  1. LLM + BO (k=5): LLM selects ingredients, then BO
  2. Vanilla BO (full d)
  3. Random Search
  4. TuRBO
  5. SAASBO
  6. REMBO
  7. SEBO (Liu et al. 2023): SAAS prior + L0-relaxed homotopy

Usage:
    python run_dairy_experiments.py [--budget 30] [--seeds 5] [--no-plot]
    python run_dairy_experiments.py --categories mozzarella cheddar --budget 20
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from botorch.acquisition import AcquisitionFunction, qLogNoisyExpectedImprovement
from botorch.fit import fit_fully_bayesian_model_nuts, fit_gpytorch_mll
from botorch.models import SaasFullyBayesianSingleTaskGP, SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NUTRIENT_COLUMNS = [
    "energy_kcal_per_100g",
    "protein_g_per_100g",
    "fat_g_per_100g",
    "saturated_fat_g_per_100g",
    "carbs_g_per_100g",
    "fiber_g_per_100g",
    "sugars_g_per_100g",
    "sodium_mg_per_100g",
]

NUTRIENT_LABELS = {
    "energy_kcal_per_100g": "energy",
    "protein_g_per_100g": "protein",
    "fat_g_per_100g": "fat",
    "saturated_fat_g_per_100g": "sat_fat",
    "carbs_g_per_100g": "carbs",
    "fiber_g_per_100g": "fiber",
    "sugars_g_per_100g": "sugars",
    "sodium_mg_per_100g": "sodium",
}

NUTRIENT_WEIGHTS = {
    "energy_kcal_per_100g": 0.75,
    "protein_g_per_100g": 1.5,
    "fat_g_per_100g": 1.25,
    "saturated_fat_g_per_100g": 1.0,
    "carbs_g_per_100g": 1.0,
    "fiber_g_per_100g": 0.75,
    "sugars_g_per_100g": 0.5,
    "sodium_mg_per_100g": 1.25,
}

# Approximate animal-based reference profiles per 100 g.
DAIRY_TARGETS: Dict[str, Dict[str, float]] = {
    "barista milk": {
        "energy_kcal_per_100g": 61.0,
        "protein_g_per_100g": 3.1,
        "fat_g_per_100g": 3.5,
        "saturated_fat_g_per_100g": 2.2,
        "carbs_g_per_100g": 4.8,
        "fiber_g_per_100g": 0.0,
        "sugars_g_per_100g": 5.0,
        "sodium_mg_per_100g": 44.0,
    },
    "butter": {
        "energy_kcal_per_100g": 717.0,
        "protein_g_per_100g": 0.9,
        "fat_g_per_100g": 81.0,
        "saturated_fat_g_per_100g": 51.0,
        "carbs_g_per_100g": 0.1,
        "fiber_g_per_100g": 0.0,
        "sugars_g_per_100g": 0.1,
        "sodium_mg_per_100g": 625.0,
    },
    "cheddar": {
        "energy_kcal_per_100g": 403.0,
        "protein_g_per_100g": 25.0,
        "fat_g_per_100g": 33.0,
        "saturated_fat_g_per_100g": 21.0,
        "carbs_g_per_100g": 1.3,
        "fiber_g_per_100g": 0.0,
        "sugars_g_per_100g": 0.5,
        "sodium_mg_per_100g": 620.0,
    },
    "coffee creamer": {
        "energy_kcal_per_100g": 120.0,
        "protein_g_per_100g": 2.0,
        "fat_g_per_100g": 10.0,
        "saturated_fat_g_per_100g": 6.0,
        "carbs_g_per_100g": 5.0,
        "fiber_g_per_100g": 0.0,
        "sugars_g_per_100g": 4.0,
        "sodium_mg_per_100g": 50.0,
    },
    "cream cheese": {
        "energy_kcal_per_100g": 342.0,
        "protein_g_per_100g": 6.2,
        "fat_g_per_100g": 34.0,
        "saturated_fat_g_per_100g": 19.0,
        "carbs_g_per_100g": 4.1,
        "fiber_g_per_100g": 0.0,
        "sugars_g_per_100g": 3.2,
        "sodium_mg_per_100g": 321.0,
    },
    "ice cream": {
        "energy_kcal_per_100g": 207.0,
        "protein_g_per_100g": 3.5,
        "fat_g_per_100g": 11.0,
        "saturated_fat_g_per_100g": 7.0,
        "carbs_g_per_100g": 24.0,
        "fiber_g_per_100g": 0.0,
        "sugars_g_per_100g": 21.0,
        "sodium_mg_per_100g": 80.0,
    },
    "milk": {
        "energy_kcal_per_100g": 61.0,
        "protein_g_per_100g": 3.3,
        "fat_g_per_100g": 3.3,
        "saturated_fat_g_per_100g": 1.9,
        "carbs_g_per_100g": 4.8,
        "fiber_g_per_100g": 0.0,
        "sugars_g_per_100g": 5.0,
        "sodium_mg_per_100g": 44.0,
    },
    "mozzarella": {
        "energy_kcal_per_100g": 280.0,
        "protein_g_per_100g": 22.0,
        "fat_g_per_100g": 22.0,
        "saturated_fat_g_per_100g": 13.0,
        "carbs_g_per_100g": 3.0,
        "fiber_g_per_100g": 0.0,
        "sugars_g_per_100g": 1.0,
        "sodium_mg_per_100g": 500.0,
    },
    "sour cream": {
        "energy_kcal_per_100g": 198.0,
        "protein_g_per_100g": 2.4,
        "fat_g_per_100g": 20.0,
        "saturated_fat_g_per_100g": 10.0,
        "carbs_g_per_100g": 4.6,
        "fiber_g_per_100g": 0.0,
        "sugars_g_per_100g": 3.4,
        "sodium_mg_per_100g": 38.0,
    },
    "yogurt": {
        "energy_kcal_per_100g": 61.0,
        "protein_g_per_100g": 3.5,
        "fat_g_per_100g": 3.3,
        "saturated_fat_g_per_100g": 2.1,
        "carbs_g_per_100g": 4.7,
        "fiber_g_per_100g": 0.0,
        "sugars_g_per_100g": 4.7,
        "sodium_mg_per_100g": 46.0,
    },
}

# GPT-5.4-authored ingredient selections for the expanded dairy ingredient set
# (usda_nutrition_data.csv filtered to in_dairy == "Y").
#
# Ingredient indices:
#    1 allulose
#    2 almond milk
#    8 canola oil
#    9 carrageenan
#   10 cashew cream
#   12 cashews
#   18 cocoa butter
#   19 coconut cream
#   22 coconut oil
#   32 corn syrup solids
#   39 gellan gum
#   46 inulin
#   56 oat cream
#   59 oat milk
#   62 pea milk
#   63 pea protein
#   69 potato protein
#   77 soy protein isolate
#   80 soymilk
LLM_SELECTIONS: Dict[str, List[int]] = {
    # barista milk - milk-like fat with low sodium and emulsified steaming behavior
    "barista milk": [59, 80, 2, 8, 39],

    # butter - very high fat / saturated fat with a modest sodium lever
    "butter": [22, 18, 8, 77, 9],

    # cheddar - high fat, high protein, elevated sodium
    "cheddar": [22, 12, 77, 63, 9],

    # coffee creamer - creamy fat profile with some sweetness and low sodium
    "coffee creamer": [56, 19, 80, 1, 39],

    # cream cheese - high fat, moderate protein, spreadable structure
    "cream cheese": [22, 10, 77, 69, 9],

    # ice cream - fat plus strong sugar/carbohydrate levers
    "ice cream": [19, 56, 1, 32, 39],

    # milk - low-fat, low-sodium beverage base
    "milk": [59, 80, 2, 62, 39],

    # mozzarella - fat/protein balance with gel structure
    "mozzarella": [22, 77, 63, 12, 9],

    # sour cream - rich acidic cream profile with moderate structure
    "sour cream": [19, 10, 22, 80, 39],

    # yogurt - cultured milk analogue with beverage/gel levers and fiber
    "yogurt": [59, 80, 62, 46, 39],
}


def load_ingredients(
    path: str = os.path.join(_ROOT, "data", "usda_nutrition_data.csv"),
) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["in_dairy"] == "Y"].copy()

    for col in NUTRIENT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["ingredient"] = df["ingredient"].fillna("").astype(str)
    df.reset_index(drop=True, inplace=True)
    return df


def get_nutrition_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[NUTRIENT_COLUMNS].to_numpy(dtype=float)


def get_target_vector(category: str) -> np.ndarray:
    return np.array([DAIRY_TARGETS[category][col] for col in NUTRIENT_COLUMNS], dtype=float)


def get_weight_vector() -> np.ndarray:
    return np.array([NUTRIENT_WEIGHTS[col] for col in NUTRIENT_COLUMNS], dtype=float)


def get_scale_vector(nutrition_matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    data_range = nutrition_matrix.max(axis=0) - nutrition_matrix.min(axis=0)
    target_floor = np.maximum(np.abs(target), 1.0)
    return np.maximum(data_range, 0.25 * target_floor)


def _weighted_distance(
    profile: np.ndarray,
    target: np.ndarray,
    scales: np.ndarray,
    weights: np.ndarray,
) -> float:
    residual = (profile - target) / scales
    return float(np.sqrt(np.sum(weights * residual ** 2)))


def make_objective(
    nutrition_matrix: np.ndarray,
    target: np.ndarray,
    scales: np.ndarray,
    weights: np.ndarray,
):
    def objective(x: np.ndarray) -> float:
        profile = x @ nutrition_matrix
        return -_weighted_distance(profile, target, scales, weights)

    return objective


def llm_select_ingredients(category: str, df: pd.DataFrame) -> List[int]:
    selected = LLM_SELECTIONS[category]
    print(f"  LLM-selected: {[df['ingredient'].iloc[i] for i in selected]}")
    return selected


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def _simplex_initial_logits(dim: int, n_init: int, seed: int) -> np.ndarray:
    """Simplex-native initialization mapped back to logits."""
    rng = np.random.RandomState(seed)
    log_floor = np.exp(-12.0)
    simplex_points = [np.full(dim, 1.0 / dim)]
    for _ in range(max(0, n_init - 1)):
        simplex_points.append(rng.dirichlet(np.ones(dim)))
    simplex = np.array(simplex_points[:n_init], dtype=float)
    simplex = np.clip(simplex, log_floor, 1.0)
    simplex = simplex / simplex.sum(axis=1, keepdims=True)
    return np.log(simplex)


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
    z_lo, z_hi = -12.0, 0.0
    bounds = torch.tensor([[z_lo] * d, [z_hi] * d], dtype=torch.double)

    all_z, all_y, best_values = [], [], []

    z_init = _simplex_initial_logits(d, n_init, seed)
    for i in range(n_init):
        z = z_init[i]
        y = objective(_softmax(z))
        all_z.append(z)
        all_y.append(y)
        best_values.append(max(all_y))

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
            q=1,
            num_restarts=10,
            raw_samples=512,
        )
        z_new = unnormalize(cand.squeeze(0), bounds).detach().numpy()
        y_new = objective(_softmax(z_new))
        all_z.append(z_new)
        all_y.append(y_new)
        best_values.append(max(all_y))

    return {"best_values": best_values, "all_y": all_y, "final_best": max(all_y)}


def llm_reduced_bo(objective, d_full: int, selected_indices: List[int],
                   budget: int = 30, n_init: int = 5, seed: int = 0) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    k = len(selected_indices)
    z_lo, z_hi = -12.0, 0.0
    bounds = torch.tensor([[z_lo] * k, [z_hi] * k], dtype=torch.double)

    def z_to_x(z):
        fracs = _softmax(z)
        x = np.zeros(d_full)
        for i, idx in enumerate(selected_indices):
            x[idx] = fracs[i]
        return x

    all_z, all_y, best_values = [], [], []

    z_init = _simplex_initial_logits(k, n_init, seed)
    for i in range(n_init):
        z = z_init[i]
        y = objective(z_to_x(z))
        all_z.append(z)
        all_y.append(y)
        best_values.append(max(all_y))

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
            q=1,
            num_restarts=10,
            raw_samples=512,
        )
        z_new = unnormalize(cand.squeeze(0), bounds).detach().numpy()
        y_new = objective(z_to_x(z_new))
        all_z.append(z_new)
        all_y.append(y_new)
        best_values.append(max(all_y))

    return {"best_values": best_values, "all_y": all_y, "final_best": max(all_y)}


def random_subset_bo(objective, d_full: int, k: int = 5,
                     budget: int = 30, n_init: int = 5, seed: int = 0) -> Dict:
    subset_rng = np.random.RandomState(seed + 10_000)
    selected_indices = sorted(subset_rng.choice(d_full, size=k, replace=False).tolist())
    return llm_reduced_bo(objective, d_full, selected_indices,
                          budget=budget, n_init=n_init, seed=seed)


def rembo_bo(objective, d: int, budget: int = 30, n_init: int = 5,
             seed: int = 0, target_dim: int = 5) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    k = target_dim
    z_lo, z_hi = -3.0, 3.0
    bounds_full = torch.tensor([[z_lo] * d, [z_hi] * d], dtype=torch.double)

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
        all_z.append(z_init[i].numpy())
        all_y.append(y)
        best_values.append(max(all_y))

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
            q=1,
            num_restarts=10,
            raw_samples=512,
        )
        z_new = unnormalize(cand.squeeze(0), z_bounds)
        y_new = objective(z_to_x(z_new))
        all_z.append(z_new.detach().numpy())
        all_y.append(y_new)
        best_values.append(max(all_y))

    return {"best_values": best_values, "all_y": all_y, "final_best": max(all_y)}


def turbo_bo(objective, d: int, budget: int = 30, n_init: int = 5,
             seed: int = 0) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    z_lo, z_hi = -12.0, 0.0
    bounds = torch.tensor([[z_lo] * d, [z_hi] * d], dtype=torch.double)

    length_init, length_min, length_max = 0.8, 0.5 ** 7, 1.6
    success_tolerance, failure_tolerance = 3, max(d, 1)
    length, n_successes, n_failures = length_init, 0, 0

    all_z, all_y, best_values = [], [], []

    z_init = _simplex_initial_logits(d, n_init, seed)
    for i in range(n_init):
        z = z_init[i]
        y = objective(_softmax(z))
        all_z.append(z)
        all_y.append(y)
        best_values.append(max(all_y))

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
            q=1,
            num_restarts=10,
            raw_samples=512,
        )
        z_new = unnormalize(cand.squeeze(0), bounds).detach().numpy()
        y_new = objective(_softmax(z_new))
        all_z.append(z_new)
        all_y.append(y_new)
        best_values.append(max(all_y))

        if y_new > best_y:
            n_successes += 1
            n_failures = 0
            best_y = y_new
        else:
            n_successes = 0
            n_failures += 1

        if n_successes >= success_tolerance:
            length = min(2.0 * length, length_max)
            n_successes = 0
        elif n_failures >= failure_tolerance:
            length = length / 2.0
            n_failures = 0

        if length < length_min:
            length = length_init
            n_successes = 0
            n_failures = 0

    return {"best_values": best_values, "all_y": all_y, "final_best": max(all_y)}


def saasbo_bo(objective, d: int, budget: int = 30, n_init: int = 5,
              seed: int = 0, warmup_steps: int = 256,
              num_samples: int = 128) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    z_lo, z_hi = -12.0, 0.0
    bounds = torch.tensor([[z_lo] * d, [z_hi] * d], dtype=torch.double)

    all_z, all_y, best_values = [], [], []

    z_init = _simplex_initial_logits(d, n_init, seed)
    for i in range(n_init):
        z = z_init[i]
        y = objective(_softmax(z))
        all_z.append(z)
        all_y.append(y)
        best_values.append(max(all_y))

    for _ in range(budget - n_init):
        train_Z = torch.tensor(np.array(all_z), dtype=torch.double)
        train_Y = torch.tensor(all_y, dtype=torch.double).unsqueeze(-1)
        train_Z_norm = normalize(train_Z, bounds)

        gp = SaasFullyBayesianSingleTaskGP(train_X=train_Z_norm, train_Y=train_Y)
        try:
            fit_fully_bayesian_model_nuts(
                gp,
                warmup_steps=warmup_steps,
                num_samples=num_samples,
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
            q=1,
            num_restarts=10,
            raw_samples=512,
        )
        z_new = unnormalize(cand.squeeze(0), bounds).detach().numpy()
        y_new = objective(_softmax(z_new))
        all_z.append(z_new)
        all_y.append(y_new)
        best_values.append(max(all_y))

    return {"best_values": best_values, "all_y": all_y, "final_best": max(all_y)}


class _PenalizedAcq(AcquisitionFunction):
    """Wraps a base acquisition with a subtractive penalty term.

    forward(X) = base_acq(X) - lambda_t * penalty_func(X)
    """

    def __init__(self, base_acq, penalty_func, lambda_t: float):
        super().__init__(model=base_acq.model)
        self.base_acq = base_acq
        self.penalty_func = penalty_func
        self.lambda_t = float(lambda_t)

    def forward(self, X):
        return self.base_acq(X) - self.lambda_t * self.penalty_func(X)


def sebo_bo(objective, d: int, budget: int = 30, n_init: int = 5,
            seed: int = 0, lambda_max: float = 1.0, l0_a: float = 0.05,
            warmup_steps: int = 64, num_samples: int = 32) -> Dict:
    """SEBO: Sparsity-Exploiting BO (Liu et al. 2023).

    Combines a SAAS-prior GP with a penalized acquisition function whose
    smooth L0 approximation is applied to the resulting ingredient
    fractions x = softmax(z). A homotopy continuation strategy linearly
    grows the sparsity weight lambda from 0 to lambda_max over the BO
    budget. Falls back to a SingleTaskGP if NUTS fitting fails.

    Reference: Liu, Feng, Eriksson, Letham, Bakshy, "Sparse Bayesian
    Optimization", AISTATS 2023 (https://arxiv.org/abs/2203.01900).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    z_lo, z_hi = -12.0, 0.0
    bounds = torch.tensor([[z_lo] * d, [z_hi] * d], dtype=torch.double)

    all_z, all_y, best_values = [], [], []

    z_init = _simplex_initial_logits(d, n_init, seed)
    for i in range(n_init):
        z = z_init[i]
        y = objective(_softmax(z))
        all_z.append(z)
        all_y.append(y)
        best_values.append(max(all_y))

    a_squared = float(l0_a) ** 2

    def sparsity_penalty(X: torch.Tensor) -> torch.Tensor:
        # X normalized in [0, 1]^d; map back to logit space, then softmax.
        z = X * (z_hi - z_lo) + z_lo
        x = torch.softmax(z, dim=-1)
        # Smooth L0 approximation: count ingredients with non-negligible mass.
        active = 1.0 - torch.exp(-x.pow(2) / a_squared)
        return active.sum(dim=(-1, -2))

    n_bo_steps = budget - n_init
    for step in range(n_bo_steps):
        train_Z = torch.tensor(np.array(all_z), dtype=torch.double)
        train_Y = torch.tensor(all_y, dtype=torch.double).unsqueeze(-1)
        train_Z_norm = normalize(train_Z, bounds)

        gp = SaasFullyBayesianSingleTaskGP(train_X=train_Z_norm, train_Y=train_Y)
        try:
            fit_fully_bayesian_model_nuts(
                gp,
                warmup_steps=warmup_steps,
                num_samples=num_samples,
                disable_progbar=True,
            )
        except Exception:
            gp = SingleTaskGP(train_Z_norm, train_Y, outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            try:
                fit_gpytorch_mll(mll)
            except Exception:
                pass

        # Homotopy: lambda grows linearly from 0 to lambda_max.
        progress = (step + 1) / max(1, n_bo_steps)
        lambda_t = lambda_max * progress

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))
        base_acq = qLogNoisyExpectedImprovement(
            model=gp, X_baseline=train_Z_norm, sampler=sampler,
        )
        penalized_acq = _PenalizedAcq(base_acq, sparsity_penalty, lambda_t)
        cand, _ = optimize_acqf(
            acq_function=penalized_acq,
            bounds=torch.stack([torch.zeros(d), torch.ones(d)]).double(),
            q=1,
            num_restarts=10,
            raw_samples=512,
        )
        z_new = unnormalize(cand.squeeze(0), bounds).detach().numpy()
        y_new = objective(_softmax(z_new))
        all_z.append(z_new)
        all_y.append(y_new)
        best_values.append(max(all_y))

    return {"best_values": best_values, "all_y": all_y, "final_best": max(all_y)}


def summarize_profile(
    nutrition_matrix: np.ndarray,
    selected: List[int],
    target: np.ndarray,
    scales: np.ndarray,
    weights: np.ndarray,
) -> Tuple[np.ndarray, float]:
    rng = np.random.RandomState(123)
    draws = rng.dirichlet(np.ones(len(selected)), size=4096)
    profiles = draws @ nutrition_matrix[selected]
    residuals = (profiles - target[None, :]) / scales[None, :]
    dists = np.sqrt(np.sum(weights[None, :] * residuals ** 2, axis=1))
    best = int(np.argmin(dists))
    return profiles[best], float(dists[best])


ALL_METHOD_KEYS = ["llm", "vanilla", "random", "turbo", "saasbo", "sebo", "rembo", "random_subset"]


def run_single_category(
    category: str,
    df: pd.DataFrame,
    nutrition_matrix: np.ndarray,
    budget: int = 30,
    n_seeds: int = 5,
    n_init: int = 5,
    methods: List[str] | None = None,
) -> Dict[str, object]:
    target = get_target_vector(category)
    weights = get_weight_vector()
    scales = get_scale_vector(nutrition_matrix, target)
    d = len(df)

    objective = make_objective(nutrition_matrix, target, scales, weights)
    selected = llm_select_ingredients(category, df)
    approx_profile, approx_dist = summarize_profile(
        nutrition_matrix,
        selected,
        target,
        scales,
        weights,
    )
    k = len(selected)

    method_registry = {
        "llm":            (f"LLM + BO (k={k})",           lambda s: llm_reduced_bo(objective, d, selected, budget=budget, n_init=n_init, seed=s)),
        "vanilla":        (f"Vanilla BO (d={d})",          lambda s: vanilla_bo(objective, d, budget=budget, n_init=n_init, seed=s)),
        "random":         (f"Random Search (d={d})",       lambda s: random_search(objective, d, budget=budget, seed=s)),
        "turbo":          (f"TuRBO (d={d})",               lambda s: turbo_bo(objective, d, budget=budget, n_init=n_init, seed=s)),
        "saasbo":         (f"SAASBO (d={d})",              lambda s: saasbo_bo(objective, d, budget=budget, n_init=n_init, seed=s, warmup_steps=64, num_samples=32)),
        "sebo":           (f"SEBO (d={d})",                lambda s: sebo_bo(objective, d, budget=budget, n_init=n_init, seed=s, warmup_steps=64, num_samples=32)),
        "rembo":          (f"REMBO (d={d}->k=5)",          lambda s: rembo_bo(objective, d, budget=budget, n_init=n_init, seed=s)),
        "random_subset":  (f"Random Subset + BO (k={k})",  lambda s: random_subset_bo(objective, d, k=k, budget=budget, n_init=n_init, seed=s)),
    }

    if methods is None:
        methods = ALL_METHOD_KEYS
    active = [(key, method_registry[key]) for key in methods]

    method_names = [name for _, (name, _) in active]
    all_traces = {name: [] for name in method_names}
    seed_records = []

    print(f"\n{'=' * 72}")
    print(f"  {category.upper()}")
    target_str = ", ".join(
        f"{NUTRIENT_LABELS[col]}={DAIRY_TARGETS[category][col]:.1f}"
        for col in NUTRIENT_COLUMNS
    )
    print(f"  target: {target_str}")
    print(f"  selected: {[df['ingredient'].iloc[i] for i in selected]}")
    approx_str = ", ".join(
        f"{NUTRIENT_LABELS[col]}={approx_profile[j]:.1f}"
        for j, col in enumerate(NUTRIENT_COLUMNS)
    )
    print(f"  best sampled subset fit: dist={approx_dist:.3f} ({approx_str})")
    print(f"{'=' * 72}")

    for seed in range(n_seeds):
        print(f"  seed {seed + 1}/{n_seeds}", end="", flush=True)
        seed_record = {"seed": int(seed), "methods": {}}

        for _, (name, run_fn) in active:
            res = run_fn(seed)
            all_traces[name].append(res["best_values"])
            seed_record["methods"][name] = {
                "best_values": [float(v) for v in res["best_values"]],
                "all_y": [float(v) for v in res["all_y"]],
                "final_best": float(res["final_best"]),
            }

        seed_records.append(seed_record)
        print(" done")

    return {
        "traces": all_traces,
        "selected_indices": [int(i) for i in selected],
        "selected_ingredients": [str(df["ingredient"].iloc[i]) for i in selected],
        "target": {
            NUTRIENT_LABELS[col]: float(DAIRY_TARGETS[category][col])
            for col in NUTRIENT_COLUMNS
        },
        "approx_profile": {
            NUTRIENT_LABELS[col]: float(approx_profile[j])
            for j, col in enumerate(NUTRIENT_COLUMNS)
        },
        "approx_distance": float(approx_dist),
        "seed_records": seed_records,
    }


_METHOD_STYLE = {
    "LLM": ("#2ecc71", "-", 2.5),
    "Vanilla": ("#e74c3c", "-.", 2.0),
    "Random Subset": ("#d4ac0d", "--", 2.0),
    "Random": ("#999999", "--", 1.5),
    "TuRBO": ("#9b59b6", "-.", 2.0),
    "SAASBO": ("#e67e22", ":", 2.0),
    "SEBO": ("#16a085", ":", 2.0),
    "REMBO": ("#3498db", "--", 2.0),
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
    run_tag: str | None = None,
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
    ax.set_ylabel("Weighted Nutrition Distance (lower = better)", fontsize=12)
    ax.set_title(f"Plant-Based {category.title()}: Multi-Nutrient Matching", fontsize=14)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, budget)
    plt.tight_layout()
    suffix = f"_{run_tag}" if run_tag else ""
    path = os.path.join(save_dir, f"dairy_{category.replace(' ', '_')}_convergence{suffix}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved {path}")


def plot_category_bar(
    category: str,
    all_traces: Dict[str, list],
    save_dir: str,
    run_tag: str | None = None,
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
    ax.set_ylabel("Final Weighted Distance (lower = better)", fontsize=11)
    ax.set_title(f"Final Performance: {category.title()}", fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    suffix = f"_{run_tag}" if run_tag else ""
    path = os.path.join(save_dir, f"dairy_{category.replace(' ', '_')}_final{suffix}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved {path}")


def plot_summary_heatmap(
    all_results: Dict[str, Dict[str, list]],
    save_dir: str,
    run_tag: str | None = None,
):
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

    ax.set_title("Final Weighted Nutrition Distance by Category & Method", fontsize=13)
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    suffix = f"_{run_tag}" if run_tag else ""
    path = os.path.join(save_dir, f"dairy_summary_heatmap{suffix}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved {path}")


def run_all(
    categories: List[str] | None = None,
    budget: int = 30,
    n_seeds: int = 5,
    n_init: int = 5,
    plot: bool = True,
    methods: List[str] | None = None,
):
    df = load_ingredients()
    d = len(df)
    nutrition_matrix = get_nutrition_matrix(df)

    if categories is None:
        categories = list(DAIRY_TARGETS.keys())

    print(f"Ingredients: {d}, Budget: {budget}, Seeds: {n_seeds}")
    print(f"Nutrients: {[NUTRIENT_LABELS[c] for c in NUTRIENT_COLUMNS]}")
    print(f"Categories: {categories}")
    print(f"Methods: {methods or ALL_METHOD_KEYS}\n")

    plot_dir = os.path.join(_ROOT, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    results_dir = os.path.join(_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(results_dir, f"dairy_experiments_{timestamp}.json")
    raw_path = os.path.join(results_dir, f"dairy_experiments_raw_{timestamp}.json")

    all_results: Dict[str, Dict[str, list]] = {}
    raw_results: Dict[str, Dict[str, object]] = {}
    raw_payload = {
        "config": {
            "domain": "dairy",
            "budget": int(budget),
            "n_seeds": int(n_seeds),
            "n_init": int(n_init),
            "plot": bool(plot),
            "categories": list(categories),
            "methods": methods or ALL_METHOD_KEYS,
            "n_ingredients": int(d),
            "ingredient_names": [str(x) for x in df["ingredient"].tolist()],
            "nutrients": [NUTRIENT_LABELS[c] for c in NUTRIENT_COLUMNS],
            "weights": {
                NUTRIENT_LABELS[col]: float(NUTRIENT_WEIGHTS[col])
                for col in NUTRIENT_COLUMNS
            },
        },
        "results": raw_results,
    }

    for cat in categories:
        category_result = run_single_category(
            cat,
            df,
            nutrition_matrix,
            budget=budget,
            n_seeds=n_seeds,
            n_init=n_init,
            methods=methods,
        )
        traces = category_result["traces"]
        all_results[cat] = traces
        raw_results[cat] = category_result

        summary_payload = {
            done_cat: {
                meth: [[float(v) for v in t] for t in tlist]
                for meth, tlist in done_traces.items()
            }
            for done_cat, done_traces in all_results.items()
        }
        with open(summary_path, "w") as f:
            json.dump(summary_payload, f, indent=2)
        with open(raw_path, "w") as f:
            json.dump(raw_payload, f, indent=2)
        print(f"  checkpointed summary to {summary_path}")
        print(f"  checkpointed raw data to {raw_path}")

        if plot:
            plot_category_convergence(cat, traces, budget, plot_dir, run_tag=timestamp)
            plot_category_bar(cat, traces, plot_dir, run_tag=timestamp)

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    method_names = list(all_results[categories[0]].keys())
    header = f"{'Category':<18}" + "".join(f"{m:<26}" for m in method_names)
    print(header)
    print("-" * len(header))
    for cat in categories:
        row = f"{cat:<18}"
        for meth in method_names:
            finals = [-t[-1] for t in all_results[cat][meth]]
            row += f"{np.mean(finals):.4f} +/- {np.std(finals):.4f}  "
        print(row)

    if plot and len(categories) > 1:
        plot_summary_heatmap(all_results, plot_dir, run_tag=timestamp)
    print(f"\nSummary results saved to {summary_path}")
    print(f"Raw results saved to {raw_path}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Expanded dairy optimization experiments")
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--n-init", type=int, default=5)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        choices=list(DAIRY_TARGETS.keys()),
        help="Run only selected categories (default: all configured dairy categories)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        choices=ALL_METHOD_KEYS,
        help="Run only selected methods (default: all). "
             f"Choices: {', '.join(ALL_METHOD_KEYS)}",
    )
    args = parser.parse_args()

    run_all(
        categories=args.categories,
        budget=args.budget,
        n_seeds=args.seeds,
        n_init=args.n_init,
        plot=not args.no_plot,
        methods=args.methods,
    )
