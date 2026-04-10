"""
Expanded meat-category optimization experiments.

Goal: for each meat category, minimize weighted Euclidean distance in a
multi-nutrient space between a plant-based formulation and the reference
animal-based product.

Unlike the dairy runner, which matches only fat and sodium, this script
matches a broader nutrition profile:
  - energy
  - protein
  - fat
  - saturated fat
  - carbs
  - fiber
  - sugars
  - sodium

Ingredients come from data/usda_nutrition_data.csv and are restricted to
rows marked in_meat == "Y".

Methods compared:
  1. LLM + BO (k=5): LLM selects ingredients, then BO
  2. Vanilla BO (full d)
  3. Random Search
  4. TuRBO
  5. SAASBO
  6. REMBO

Usage:
    python run_meat_experiments.py [--budget 30] [--seeds 5] [--no-plot]
    python run_meat_experiments.py --categories bacon burgers --budget 20
"""

import json
import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from botorch.acquisition import qLogNoisyExpectedImprovement
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
MEAT_TARGETS: Dict[str, Dict[str, float]] = {
    "bacon": {
        "energy_kcal_per_100g": 541.0,
        "protein_g_per_100g": 37.0,
        "fat_g_per_100g": 42.0,
        "saturated_fat_g_per_100g": 14.0,
        "carbs_g_per_100g": 1.5,
        "fiber_g_per_100g": 0.0,
        "sugars_g_per_100g": 0.0,
        "sodium_mg_per_100g": 1717.0,
    },
    "bratwurst": {
        "energy_kcal_per_100g": 333.0,
        "protein_g_per_100g": 12.0,
        "fat_g_per_100g": 29.0,
        "saturated_fat_g_per_100g": 10.0,
        "carbs_g_per_100g": 2.0,
        "fiber_g_per_100g": 0.0,
        "sugars_g_per_100g": 0.5,
        "sodium_mg_per_100g": 846.0,
    },
    "breaded chicken fillets": {
        "energy_kcal_per_100g": 250.0,
        "protein_g_per_100g": 18.0,
        "fat_g_per_100g": 12.0,
        "saturated_fat_g_per_100g": 2.5,
        "carbs_g_per_100g": 15.0,
        "fiber_g_per_100g": 1.0,
        "sugars_g_per_100g": 1.0,
        "sodium_mg_per_100g": 600.0,
    },
    "breakfast sausage patties": {
        "energy_kcal_per_100g": 300.0,
        "protein_g_per_100g": 13.0,
        "fat_g_per_100g": 26.0,
        "saturated_fat_g_per_100g": 8.0,
        "carbs_g_per_100g": 2.0,
        "fiber_g_per_100g": 0.0,
        "sugars_g_per_100g": 1.0,
        "sodium_mg_per_100g": 900.0,
    },
    "burgers": {
        "energy_kcal_per_100g": 254.0,
        "protein_g_per_100g": 17.0,
        "fat_g_per_100g": 17.0,
        "saturated_fat_g_per_100g": 7.0,
        "carbs_g_per_100g": 9.0,
        "fiber_g_per_100g": 0.0,
        "sugars_g_per_100g": 0.0,
        "sodium_mg_per_100g": 620.0,
    },
    "chicken nuggets": {
        "energy_kcal_per_100g": 296.0,
        "protein_g_per_100g": 15.0,
        "fat_g_per_100g": 18.0,
        "saturated_fat_g_per_100g": 3.5,
        "carbs_g_per_100g": 18.0,
        "fiber_g_per_100g": 1.5,
        "sugars_g_per_100g": 0.5,
        "sodium_mg_per_100g": 540.0,
    },
    "deli slices - ham": {
        "energy_kcal_per_100g": 110.0,
        "protein_g_per_100g": 18.0,
        "fat_g_per_100g": 4.0,
        "saturated_fat_g_per_100g": 1.5,
        "carbs_g_per_100g": 2.0,
        "fiber_g_per_100g": 0.0,
        "sugars_g_per_100g": 1.0,
        "sodium_mg_per_100g": 1200.0,
    },
    "deli slices - turkey": {
        "energy_kcal_per_100g": 104.0,
        "protein_g_per_100g": 17.0,
        "fat_g_per_100g": 2.0,
        "saturated_fat_g_per_100g": 0.6,
        "carbs_g_per_100g": 3.0,
        "fiber_g_per_100g": 0.0,
        "sugars_g_per_100g": 1.0,
        "sodium_mg_per_100g": 950.0,
    },
    "hot dogs": {
        "energy_kcal_per_100g": 290.0,
        "protein_g_per_100g": 11.0,
        "fat_g_per_100g": 26.0,
        "saturated_fat_g_per_100g": 9.0,
        "carbs_g_per_100g": 3.0,
        "fiber_g_per_100g": 0.0,
        "sugars_g_per_100g": 0.5,
        "sodium_mg_per_100g": 1090.0,
    },
    "meatballs": {
        "energy_kcal_per_100g": 286.0,
        "protein_g_per_100g": 15.0,
        "fat_g_per_100g": 21.0,
        "saturated_fat_g_per_100g": 8.0,
        "carbs_g_per_100g": 9.0,
        "fiber_g_per_100g": 0.5,
        "sugars_g_per_100g": 3.0,
        "sodium_mg_per_100g": 780.0,
    },
    "pulled pork": {
        "energy_kcal_per_100g": 250.0,
        "protein_g_per_100g": 20.0,
        "fat_g_per_100g": 17.0,
        "saturated_fat_g_per_100g": 6.0,
        "carbs_g_per_100g": 3.0,
        "fiber_g_per_100g": 0.0,
        "sugars_g_per_100g": 2.0,
        "sodium_mg_per_100g": 700.0,
    },
    "steak fillets": {
        "energy_kcal_per_100g": 220.0,
        "protein_g_per_100g": 26.0,
        "fat_g_per_100g": 12.0,
        "saturated_fat_g_per_100g": 4.8,
        "carbs_g_per_100g": 0.0,
        "fiber_g_per_100g": 0.0,
        "sugars_g_per_100g": 0.0,
        "sodium_mg_per_100g": 60.0,
    },
    "unbreaded chicken fillet": {
        "energy_kcal_per_100g": 165.0,
        "protein_g_per_100g": 31.0,
        "fat_g_per_100g": 3.6,
        "saturated_fat_g_per_100g": 1.0,
        "carbs_g_per_100g": 0.0,
        "fiber_g_per_100g": 0.0,
        "sugars_g_per_100g": 0.0,
        "sodium_mg_per_100g": 74.0,
    },
    "unbreaded chicken strips & chunks": {
        "energy_kcal_per_100g": 170.0,
        "protein_g_per_100g": 26.0,
        "fat_g_per_100g": 5.0,
        "saturated_fat_g_per_100g": 1.2,
        "carbs_g_per_100g": 2.0,
        "fiber_g_per_100g": 0.0,
        "sugars_g_per_100g": 0.0,
        "sodium_mg_per_100g": 450.0,
    },
}

# GPT-5.4-authored ingredient selections, following the same pattern as the
# dairy script: one fixed expert subset per category for up-front dimensionality
# reduction before BO.
#
# Ingredient indices (from usda_nutrition_data.csv filtered to in_meat == "Y"):
#   16  canola oil
#   21  coconut oil
#   23  corn starch
#   35  jackfruit
#   40  liquid smoke
#   43  methylcellulose
#   46  mushroom mycelium
#   50  mycoprotein
#   55  onion
#   61  pea protein
#   65  potato protein
#   82  soy protein isolate
#   83  soy sauce
#   93  tapioca starch
#   94  textured vegetable protein
#   102 wheat flour
#   103 wheat gluten
#   108 yeast extract
LLM_SELECTIONS: Dict[str, List[int]] = {
    # bacon - high fat, high saturated fat, high sodium, smoky/savory
    "bacon": [21, 82, 108, 40, 43],

    # bratwurst - fatty sausage with sodium, structure, and protein
    "bratwurst": [21, 82, 43, 108, 65],

    # breaded chicken fillets - lean protein core plus breading/binding
    "breaded chicken fillets": [82, 103, 23, 43, 108],

    # breakfast sausage patties - fatty, salty, structured breakfast sausage
    "breakfast sausage patties": [21, 82, 43, 108, 61],

    # burgers - protein/fiber base, fat lever, binder, savory lever
    "burgers": [82, 94, 16, 43, 108],

    # chicken nuggets - protein + starch/breading + binder + savory
    "chicken nuggets": [82, 103, 23, 43, 108],

    # deli ham - lean protein, sodium, binder, smoke/ham-like flavor
    "deli slices - ham": [82, 65, 43, 83, 40],

    # deli turkey - leaner deli profile with sodium and sliceable structure
    "deli slices - turkey": [82, 103, 43, 83, 40],

    # hot dogs - fatty emulsified sausage with strong sodium lever
    "hot dogs": [21, 82, 43, 108, 83],

    # meatballs - protein base, carbs/binder, aromatics, savory
    "meatballs": [82, 94, 102, 55, 108],

    # pulled pork - fibrous base, smoke, savory sodium, binder
    "pulled pork": [35, 82, 40, 108, 43],

    # steak fillets - high protein, modest fat, umami/fibrous structure
    "steak fillets": [103, 82, 16, 108, 50],

    # unbreaded chicken fillet - lean protein structure with mild fat and binding
    "unbreaded chicken fillet": [82, 103, 43, 16, 108],

    # unbreaded chicken strips & chunks - lean protein with chunkable texture
    "unbreaded chicken strips & chunks": [82, 103, 43, 108, 50],
}


def load_ingredients(
    path: str = os.path.join(_ROOT, "data", "usda_nutrition_data.csv"),
) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["in_meat"] == "Y"].copy()

    for col in NUTRIENT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["ingredient"] = df["ingredient"].fillna("").astype(str)
    df.reset_index(drop=True, inplace=True)
    return df


def get_nutrition_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[NUTRIENT_COLUMNS].to_numpy(dtype=float)


def get_target_vector(category: str) -> np.ndarray:
    return np.array([MEAT_TARGETS[category][col] for col in NUTRIENT_COLUMNS], dtype=float)


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
    bounds = torch.tensor([[z_lo] * d, [z_hi] * d], dtype=torch.double)

    all_z, all_y, best_values = [], [], []

    sobol = SobolEngine(dimension=d, scramble=True, seed=seed)
    z_init = unnormalize(sobol.draw(n_init).double(), bounds)
    for i in range(n_init):
        z = z_init[i].numpy()
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


def reduced_bo(objective, d_full: int, selected_indices: List[int],
               budget: int = 30, n_init: int = 5, seed: int = 0) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    k = len(selected_indices)
    z_lo, z_hi = -3.0, 3.0
    bounds = torch.tensor([[z_lo] * k, [z_hi] * k], dtype=torch.double)

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
    z_lo, z_hi = -3.0, 3.0
    bounds = torch.tensor([[z_lo] * d, [z_hi] * d], dtype=torch.double)

    length_init, length_min, length_max = 0.8, 0.5 ** 7, 1.6
    success_tolerance, failure_tolerance = 3, max(d, 1)
    length, n_successes, n_failures = length_init, 0, 0

    all_z, all_y, best_values = [], [], []

    sobol = SobolEngine(dimension=d, scramble=True, seed=seed)
    z_init = unnormalize(sobol.draw(n_init).double(), bounds)
    for i in range(n_init):
        z = z_init[i].numpy()
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
    z_lo, z_hi = -3.0, 3.0
    bounds = torch.tensor([[z_lo] * d, [z_hi] * d], dtype=torch.double)

    all_z, all_y, best_values = [], [], []

    sobol = SobolEngine(dimension=d, scramble=True, seed=seed)
    z_init = unnormalize(sobol.draw(n_init).double(), bounds)
    for i in range(n_init):
        z = z_init[i].numpy()
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


def run_single_category(
    category: str,
    df: pd.DataFrame,
    nutrition_matrix: np.ndarray,
    budget: int = 30,
    n_seeds: int = 5,
    n_init: int = 5,
) -> Dict[str, list]:
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

    method_names = [
        f"LLM + BO (k={k})",
        f"Vanilla BO (d={d})",
        f"Random Search (d={d})",
        f"TuRBO (d={d})",
        f"SAASBO (d={d})",
        f"REMBO (d={d}->k=5)",
    ]
    all_traces = {name: [] for name in method_names}

    print(f"\n{'=' * 72}")
    print(f"  {category.upper()}")
    target_str = ", ".join(
        f"{NUTRIENT_LABELS[col]}={MEAT_TARGETS[category][col]:.1f}"
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

        res = reduced_bo(objective, d, selected, budget=budget, n_init=n_init, seed=seed)
        all_traces[method_names[0]].append(res["best_values"])

        res = vanilla_bo(objective, d, budget=budget, n_init=n_init, seed=seed)
        all_traces[method_names[1]].append(res["best_values"])

        res = random_search(objective, d, budget=budget, seed=seed)
        all_traces[method_names[2]].append(res["best_values"])

        res = turbo_bo(objective, d, budget=budget, n_init=n_init, seed=seed)
        all_traces[method_names[3]].append(res["best_values"])

        res = saasbo_bo(
            objective,
            d,
            budget=budget,
            n_init=n_init,
            seed=seed,
            warmup_steps=64,
            num_samples=32,
        )
        all_traces[method_names[4]].append(res["best_values"])

        res = rembo_bo(objective, d, budget=budget, n_init=n_init, seed=seed)
        all_traces[method_names[5]].append(res["best_values"])

        print(" done")

    return all_traces


_METHOD_STYLE = {
    "LLM": ("#2ecc71", "-", 2.5),
    "Vanilla": ("#e74c3c", "-.", 2.0),
    "Random": ("#999999", "--", 1.5),
    "TuRBO": ("#9b59b6", "-.", 2.0),
    "SAASBO": ("#e67e22", ":", 2.0),
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
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, budget)
    plt.tight_layout()
    path = os.path.join(save_dir, f"meat_{category.replace(' ', '_').replace('&', 'and')}_convergence.png")
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
    ax.set_ylabel("Final Weighted Distance (lower = better)", fontsize=11)
    ax.set_title(f"Final Performance: {category.title()}", fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(save_dir, f"meat_{category.replace(' ', '_').replace('&', 'and')}_final.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved {path}")


def plot_summary_heatmap(
    all_results: Dict[str, Dict[str, list]],
    save_dir: str,
):
    categories = list(all_results.keys())
    method_names = list(all_results[categories[0]].keys())

    mat = np.zeros((len(categories), len(method_names)))
    for i, cat in enumerate(categories):
        for j, meth in enumerate(method_names):
            finals = [-t[-1] for t in all_results[cat][meth]]
            mat[i, j] = np.mean(finals)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
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
    path = os.path.join(save_dir, "meat_summary_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved {path}")


def run_all(
    categories: List[str] | None = None,
    budget: int = 30,
    n_seeds: int = 5,
    n_init: int = 5,
    plot: bool = True,
):
    df = load_ingredients()
    d = len(df)
    nutrition_matrix = get_nutrition_matrix(df)

    if categories is None:
        categories = list(MEAT_TARGETS.keys())

    print(f"Ingredients: {d}, Budget: {budget}, Seeds: {n_seeds}")
    print(f"Nutrients: {[NUTRIENT_LABELS[c] for c in NUTRIENT_COLUMNS]}")
    print(f"Categories: {categories}\n")

    plot_dir = os.path.join(_ROOT, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    results_dir = os.path.join(_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)

    all_results: Dict[str, Dict[str, list]] = {}

    for cat in categories:
        traces = run_single_category(
            cat,
            df,
            nutrition_matrix,
            budget=budget,
            n_seeds=n_seeds,
            n_init=n_init,
        )
        all_results[cat] = traces

        if plot:
            plot_category_convergence(cat, traces, budget, plot_dir)
            plot_category_bar(cat, traces, plot_dir)

    print("\n" + "=" * 92)
    print("RESULTS SUMMARY")
    print("=" * 92)
    method_names = list(all_results[categories[0]].keys())
    header = f"{'Category':<30}" + "".join(f"{m:<26}" for m in method_names)
    print(header)
    print("-" * len(header))
    for cat in categories:
        row = f"{cat:<30}"
        for meth in method_names:
            finals = [-t[-1] for t in all_results[cat][meth]]
            row += f"{np.mean(finals):.4f} +/- {np.std(finals):.4f}  "
        print(row)

    if plot and len(categories) > 1:
        plot_summary_heatmap(all_results, plot_dir)

    results_path = os.path.join(results_dir, "meat_experiments.json")
    existing = {}
    if os.path.exists(results_path):
        try:
            with open(results_path) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = {}

    serializable = dict(existing)
    for cat, traces in all_results.items():
        serializable[cat] = {
            meth: [[float(v) for v in t] for t in tlist]
            for meth, tlist in traces.items()
        }

    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nRaw results saved to {results_path}")

    return all_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Expanded meat optimization experiments")
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--n-init", type=int, default=5)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        choices=list(MEAT_TARGETS.keys()),
        help="Run only selected categories (default: all configured meat categories)",
    )
    args = parser.parse_args()

    run_all(
        categories=args.categories,
        budget=args.budget,
        n_seeds=args.seeds,
        n_init=args.n_init,
        plot=not args.no_plot,
    )
