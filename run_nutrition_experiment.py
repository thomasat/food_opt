"""
Nutrition-distance BO experiment for meeting demo.

Setup:
- 20 ingredients with fat (g/100g) and sodium (mg/100g) from ingredients.csv
- Decision variables: concentration of each ingredient (grams), subject to bounds
- Objective: minimize L2 distance to a target nutrition profile
- Target: a "moderately lean, moderate-sodium" product

Methods compared:
1. Random search (baseline)
2. Vanilla BO on all 20 ingredients
3. LLM-reduced BO: LLM picks the ~5 most relevant ingredients for
   hitting a nutrition target, BO only over those

Usage:
    python run_nutrition_experiment.py [--budget 30] [--seeds 5] [--plot]
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.models.transforms.outcome import Standardize
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine


# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------

def load_ingredients(path: str = "ingredients.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


# ---------------------------------------------------------------------------
# 2. NUTRITION OBJECTIVE
# ---------------------------------------------------------------------------

def make_nutrition_objective(
    df: pd.DataFrame,
    target_fat: float,
    target_sodium: float,
    fat_weight: float = 1.0,
    sodium_weight: float = 1.0,
):
    """Returns a callable: x (ingredient grams) -> negative L2 distance to target.

    We negate so that BO can maximize (higher = better = closer to target).

    Nutrition of a formulation: weighted sum of per-ingredient nutrition.
      fat_total = sum_i x_i * fat_per_100g_i / 100
      sodium_total = sum_i x_i * sodium_per_100g_i / 100
    """
    fat_per_g = df["fat_per_100g"].values / 100.0       # fat per 1g of ingredient
    sodium_per_g = df["sodium_per_100g"].values / 100.0  # sodium(mg) per 1g of ingredient

    # Normalization: so fat and sodium contribute equally
    # Estimate rough ranges from bounds
    x_max = df["max"].values
    max_fat = float(np.dot(x_max, fat_per_g))
    max_sodium = float(np.dot(x_max, sodium_per_g))
    fat_scale = max(max_fat, 1.0)
    sodium_scale = max(max_sodium, 1.0)

    def objective(x: np.ndarray) -> float:
        fat_total = np.dot(x, fat_per_g)
        sodium_total = np.dot(x, sodium_per_g)

        # Normalized L2 distance
        dist = np.sqrt(
            fat_weight * ((fat_total - target_fat) / fat_scale) ** 2
            + sodium_weight * ((sodium_total - target_sodium) / sodium_scale) ** 2
        )
        return -float(dist)  # negate for maximization

    return objective


# ---------------------------------------------------------------------------
# 3. RANDOM SEARCH
# ---------------------------------------------------------------------------

def random_search(
    objective,
    bounds: np.ndarray,  # (2, d)
    budget: int = 30,
    seed: int = 0,
) -> Dict:
    rng = np.random.RandomState(seed)
    d = bounds.shape[1]
    best_values = []
    all_y = []

    for _ in range(budget):
        x = bounds[0] + rng.rand(d) * (bounds[1] - bounds[0])
        y = objective(x)
        all_y.append(y)
        best_values.append(max(all_y))

    return {"best_values": best_values, "all_y": all_y, "final_best": max(all_y)}


# ---------------------------------------------------------------------------
# 4. VANILLA BO (full dimensional)
# ---------------------------------------------------------------------------

def vanilla_bo(
    objective,
    bounds_np: np.ndarray,  # (2, d)
    budget: int = 30,
    n_init: int = 5,
    seed: int = 0,
) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    d = bounds_np.shape[1]
    bounds = torch.tensor(bounds_np, dtype=torch.double)

    all_x = []
    all_y = []
    best_values = []

    # Sobol init
    sobol = SobolEngine(dimension=d, scramble=True, seed=seed)
    x_init = unnormalize(sobol.draw(n_init).double(), bounds)

    for i in range(n_init):
        x = x_init[i].numpy()
        y = objective(x)
        all_x.append(x)
        all_y.append(y)
        best_values.append(max(all_y))

    for step in range(budget - n_init):
        train_X = torch.tensor(np.array(all_x), dtype=torch.double)
        train_Y = torch.tensor(all_y, dtype=torch.double).unsqueeze(-1)
        train_X_norm = normalize(train_X, bounds)

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
            q=1, num_restarts=10, raw_samples=512,
        )

        x_new = unnormalize(candidate_norm.squeeze(0), bounds).detach().numpy()
        y_new = objective(x_new)
        all_x.append(x_new)
        all_y.append(y_new)
        best_values.append(max(all_y))

    return {"best_values": best_values, "all_y": all_y, "final_best": max(all_y)}


# ---------------------------------------------------------------------------
# 5. LLM-REDUCED BO (optimize only over selected ingredients)
# ---------------------------------------------------------------------------

def llm_reduced_bo(
    objective,
    bounds_np: np.ndarray,       # (2, d)
    selected_indices: List[int],  # which ingredients to optimize
    budget: int = 30,
    n_init: int = 5,
    seed: int = 0,
) -> Dict:
    """BO only over selected ingredients; others fixed at their midpoint."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    d_full = bounds_np.shape[1]
    k = len(selected_indices)
    fixed_indices = [i for i in range(d_full) if i not in selected_indices]

    # Fixed values = midpoint of bounds for non-selected ingredients
    x_fixed = (bounds_np[0] + bounds_np[1]) / 2.0

    # Sub-bounds for selected ingredients
    sub_bounds_np = np.stack([
        bounds_np[0][selected_indices],
        bounds_np[1][selected_indices],
    ])  # (2, k)
    sub_bounds = torch.tensor(sub_bounds_np, dtype=torch.double)

    def sub_to_full(z: np.ndarray) -> np.ndarray:
        x = x_fixed.copy()
        for i, idx in enumerate(selected_indices):
            x[idx] = z[i]
        return x

    all_z = []
    all_y = []
    best_values = []

    # Sobol init in k-dim
    sobol = SobolEngine(dimension=k, scramble=True, seed=seed)
    z_init = unnormalize(sobol.draw(n_init).double(), sub_bounds)

    for i in range(n_init):
        z = z_init[i].numpy()
        x = sub_to_full(z)
        y = objective(x)
        all_z.append(z)
        all_y.append(y)
        best_values.append(max(all_y))

    for step in range(budget - n_init):
        train_Z = torch.tensor(np.array(all_z), dtype=torch.double)
        train_Y = torch.tensor(all_y, dtype=torch.double).unsqueeze(-1)
        train_Z_norm = normalize(train_Z, sub_bounds)

        gp = SingleTaskGP(train_Z_norm, train_Y, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        try:
            fit_gpytorch_mll(mll)
        except Exception:
            pass

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))
        acq = qLogNoisyExpectedImprovement(
            model=gp, X_baseline=train_Z_norm, sampler=sampler,
        )

        candidate_norm, _ = optimize_acqf(
            acq_function=acq,
            bounds=torch.stack([torch.zeros(k), torch.ones(k)]).double(),
            q=1, num_restarts=10, raw_samples=512,
        )

        z_new = unnormalize(candidate_norm.squeeze(0), sub_bounds).detach().numpy()
        x_new = sub_to_full(z_new)
        y_new = objective(x_new)
        all_z.append(z_new)
        all_y.append(y_new)
        best_values.append(max(all_y))

    return {"best_values": best_values, "all_y": all_y, "final_best": max(all_y)}


# ---------------------------------------------------------------------------
# 6. LLM INGREDIENT SELECTION (simulated)
# ---------------------------------------------------------------------------

def llm_select_ingredients(df: pd.DataFrame, target_fat: float, target_sodium: float) -> List[int]:
    """Simulate what an LLM would pick: ingredients with highest fat or sodium content,
    since those are the levers that actually move the nutrition profile.

    An LLM would reason: "to hit a fat target, I need to control the high-fat
    ingredients (oils). To hit a sodium target, I need salt, soy_sauce, yeast_extract."

    Returns indices of selected ingredients.
    """
    # Score each ingredient by how much nutritional leverage it provides
    fat_leverage = df["fat_per_100g"].values * df["max"].values / 100.0
    sodium_leverage = df["sodium_per_100g"].values * df["max"].values / 100.0

    # Normalize leverages
    max_fl = max(fat_leverage.max(), 1e-6)
    max_sl = max(sodium_leverage.max(), 1e-6)
    combined = fat_leverage / max_fl + sodium_leverage / max_sl

    # Pick top 5
    selected = list(np.argsort(combined)[-5:])

    print(f"  LLM-selected ingredients: {[df['name'].iloc[i] for i in selected]}")
    return selected


# ---------------------------------------------------------------------------
# 7. MAIN EXPERIMENT
# ---------------------------------------------------------------------------

def run_experiment(
    budget: int = 30,
    n_seeds: int = 5,
    n_init: int = 5,
    target_fat: float = 8.0,      # grams fat in the total formulation
    target_sodium: float = 400.0,  # mg sodium in the total formulation
):
    df = load_ingredients()
    d = len(df)
    bounds_np = np.stack([df["min"].values, df["max"].values]).astype(float)

    objective = make_nutrition_objective(df, target_fat, target_sodium)
    selected = llm_select_ingredients(df, target_fat, target_sodium)

    methods = {
        "Random Search": {},
        f"Vanilla BO (d={d})": {},
        f"LLM-Reduced BO (k={len(selected)})": {},
    }

    all_traces = {name: [] for name in methods}

    for seed in range(n_seeds):
        print(f"Seed {seed+1}/{n_seeds}")

        # Random search
        res = random_search(objective, bounds_np, budget=budget, seed=seed)
        all_traces["Random Search"].append(res["best_values"])

        # Vanilla BO
        res = vanilla_bo(objective, bounds_np, budget=budget, n_init=n_init, seed=seed)
        all_traces[f"Vanilla BO (d={d})"].append(res["best_values"])

        # LLM-reduced BO
        res = llm_reduced_bo(objective, bounds_np, selected, budget=budget, n_init=n_init, seed=seed)
        all_traces[f"LLM-Reduced BO (k={len(selected)})"].append(res["best_values"])

    return all_traces, budget


# ---------------------------------------------------------------------------
# 8. PLOTTING
# ---------------------------------------------------------------------------

def plot_convergence(all_traces: Dict, budget: int, save_path: str = "convergence.png"):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    colors = {"Random Search": "#999999", }
    styles = {}
    for i, name in enumerate(all_traces):
        if "Random" in name:
            colors[name] = "#999999"
            styles[name] = "--"
        elif "Vanilla" in name:
            colors[name] = "#e74c3c"
            styles[name] = "-."
        else:
            colors[name] = "#2ecc71"
            styles[name] = "-"

    iters = np.arange(1, budget + 1)

    for name, traces in all_traces.items():
        traces_arr = np.array(traces)
        mean = traces_arr.mean(axis=0)
        std = traces_arr.std(axis=0)

        # Convert from negative distance to positive (for cleaner y-axis)
        mean_dist = -mean
        std_dist = std

        ax.plot(iters, mean_dist, label=name, color=colors[name],
                linestyle=styles[name], linewidth=2)
        ax.fill_between(iters, mean_dist - std_dist, mean_dist + std_dist,
                        alpha=0.15, color=colors[name])

    ax.set_xlabel("Evaluation", fontsize=12)
    ax.set_ylabel("L2 Distance to Target (lower = better)", fontsize=12)
    ax.set_title("Convergence: Nutrition Target Optimization", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, budget)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")
    plt.close()


def plot_final_bar(all_traces: Dict, save_path: str = "final_comparison.png"):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    names = list(all_traces.keys())
    final_means = []
    final_stds = []

    for name in names:
        finals = [-t[-1] for t in all_traces[name]]  # negate back to distance
        final_means.append(np.mean(finals))
        final_stds.append(np.std(finals))

    colors = []
    for name in names:
        if "Random" in name:
            colors.append("#999999")
        elif "Vanilla" in name:
            colors.append("#e74c3c")
        else:
            colors.append("#2ecc71")

    bars = ax.bar(range(len(names)), final_means, yerr=final_stds,
                  color=colors, capsize=5, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=10)
    ax.set_ylabel("Final L2 Distance (lower = better)", fontsize=11)
    ax.set_title("Final Performance Comparison", fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Plot saved to {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--n-init", type=int, default=5)
    parser.add_argument("--target-fat", type=float, default=8.0)
    parser.add_argument("--target-sodium", type=float, default=400.0)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    print(f"Target: fat={args.target_fat}g, sodium={args.target_sodium}mg")
    print(f"Budget: {args.budget}, Seeds: {args.seeds}")

    all_traces, budget = run_experiment(
        budget=args.budget,
        n_seeds=args.seeds,
        n_init=args.n_init,
        target_fat=args.target_fat,
        target_sodium=args.target_sodium,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for name, traces in all_traces.items():
        finals = [-t[-1] for t in traces]
        print(f"  {name:<35} {np.mean(finals):.4f} +/- {np.std(finals):.4f}")

    if not args.no_plot:
        plot_convergence(all_traces, budget)
        plot_final_bar(all_traces)
