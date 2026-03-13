"""
Plant-based mozzarella optimization experiment.

Goal: minimize Euclidean distance in normalized (fat, sodium) space between
a plant-based formulation and animal-based mozzarella.

Ingredients are weight fractions summing to 1.
Normalization: fat_norm = fat_per_100g / 100, sodium_norm = sodium_per_100g / 1000.

Animal-based mozzarella target (per 100g): fat ≈ 22g, sodium ≈ 500mg
  → normalized target: (0.22, 0.50)

Methods compared:
  1. LLM + BO (k=5):  LLM selects 5 ingredients, BO optimizes those fractions
  2. Vanilla BO (d=28): BO over all 28 ingredient fractions
  3. Random Search (d=28): uniform random on the simplex

Usage:
    python run_mozzarella_experiment.py [--budget 30] [--seeds 5] [--plot]
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
# 1. DATA & TARGET
# ---------------------------------------------------------------------------

TARGET_FAT_NORM = 0.22      # 22g fat / 100g, normalized by 100
TARGET_SODIUM_NORM = 0.50   # 500mg sodium / 100g, normalized by 1000


def load_ingredients(path: str = "ingredients_dairy.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def get_nutrition_arrays(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Return normalized fat and sodium arrays for each ingredient."""
    fat_norm = df["fat_per_100g"].values / 100.0     # in [0, 1]
    sodium_norm = df["sodium_per_100g"].values / 1000.0  # in [0, 1]
    return fat_norm, sodium_norm


# ---------------------------------------------------------------------------
# 2. OBJECTIVE
# ---------------------------------------------------------------------------

def make_objective(fat_norm: np.ndarray, sodium_norm: np.ndarray):
    """Returns callable: x (weight fractions summing to ~1) -> negative L2 distance."""
    def objective(x: np.ndarray) -> float:
        fat_mix = np.dot(x, fat_norm)
        sodium_mix = np.dot(x, sodium_norm)
        dist = np.sqrt((fat_mix - TARGET_FAT_NORM)**2 + (sodium_mix - TARGET_SODIUM_NORM)**2)
        return -float(dist)  # negate for maximization
    return objective


# ---------------------------------------------------------------------------
# 3. SIMPLEX SAMPLING (ingredients sum to 1)
# ---------------------------------------------------------------------------

def sample_simplex(d: int, n: int, rng: np.random.RandomState) -> np.ndarray:
    """Sample n points uniformly from the d-dimensional simplex."""
    # Dirichlet(1,...,1) gives uniform distribution on simplex
    return rng.dirichlet(np.ones(d), size=n)


def sample_sub_simplex(k: int, n: int, d_full: int,
                       selected_indices: List[int],
                       rng: np.random.RandomState) -> np.ndarray:
    """Sample on the sub-simplex: only selected ingredients active, rest = 0."""
    z = rng.dirichlet(np.ones(k), size=n)
    X = np.zeros((n, d_full))
    for i, idx in enumerate(selected_indices):
        X[:, idx] = z[:, i]
    return X


# ---------------------------------------------------------------------------
# 4. RANDOM SEARCH (d=28, simplex)
# ---------------------------------------------------------------------------

def random_search(objective, d: int, budget: int = 30, seed: int = 0) -> Dict:
    rng = np.random.RandomState(seed)
    best_values = []
    all_y = []

    for _ in range(budget):
        x = rng.dirichlet(np.ones(d))
        y = objective(x)
        all_y.append(y)
        best_values.append(max(all_y))

    return {"best_values": best_values, "all_y": all_y, "final_best": max(all_y)}


# ---------------------------------------------------------------------------
# 5. VANILLA BO (d=28, simplex via softmax reparametrization)
# ---------------------------------------------------------------------------

def _softmax(z: np.ndarray) -> np.ndarray:
    """Map unconstrained R^d -> simplex via softmax."""
    e = np.exp(z - z.max())
    return e / e.sum()


def vanilla_bo(objective, d: int, budget: int = 30, n_init: int = 5,
               seed: int = 0) -> Dict:
    """BO in unconstrained space, mapped to simplex via softmax."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Work in unconstrained z-space; bounds [-3, 3] per dim
    z_lo, z_hi = -3.0, 3.0
    bounds_np = np.array([[z_lo]*d, [z_hi]*d])
    bounds = torch.tensor(bounds_np, dtype=torch.double)

    all_z = []
    all_y = []
    best_values = []

    # Sobol init
    sobol = SobolEngine(dimension=d, scramble=True, seed=seed)
    z_init = sobol.draw(n_init).double()
    z_init = unnormalize(z_init, bounds)

    for i in range(n_init):
        z = z_init[i].numpy()
        x = _softmax(z)
        y = objective(x)
        all_z.append(z)
        all_y.append(y)
        best_values.append(max(all_y))

    for step in range(budget - n_init):
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
        acq = qLogNoisyExpectedImprovement(
            model=gp, X_baseline=train_Z_norm, sampler=sampler,
        )
        candidate_norm, _ = optimize_acqf(
            acq_function=acq,
            bounds=torch.stack([torch.zeros(d), torch.ones(d)]).double(),
            q=1, num_restarts=10, raw_samples=512,
        )
        z_new = unnormalize(candidate_norm.squeeze(0), bounds).detach().numpy()
        x_new = _softmax(z_new)
        y_new = objective(x_new)
        all_z.append(z_new)
        all_y.append(y_new)
        best_values.append(max(all_y))

    return {"best_values": best_values, "all_y": all_y, "final_best": max(all_y)}


# ---------------------------------------------------------------------------
# 6. LLM-REDUCED BO (k=5 on sub-simplex)
# ---------------------------------------------------------------------------

def llm_reduced_bo(objective, d_full: int, selected_indices: List[int],
                   budget: int = 30, n_init: int = 5, seed: int = 0) -> Dict:
    """BO over k selected ingredients on the sub-simplex (softmax in k-dim)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    k = len(selected_indices)
    z_lo, z_hi = -3.0, 3.0
    bounds_np = np.array([[z_lo]*k, [z_hi]*k])
    bounds = torch.tensor(bounds_np, dtype=torch.double)

    def z_to_x(z: np.ndarray) -> np.ndarray:
        fracs = _softmax(z)
        x = np.zeros(d_full)
        for i, idx in enumerate(selected_indices):
            x[idx] = fracs[i]
        return x

    all_z = []
    all_y = []
    best_values = []

    sobol = SobolEngine(dimension=k, scramble=True, seed=seed)
    z_init = unnormalize(sobol.draw(n_init).double(), bounds)

    for i in range(n_init):
        z = z_init[i].numpy()
        x = z_to_x(z)
        y = objective(x)
        all_z.append(z)
        all_y.append(y)
        best_values.append(max(all_y))

    for step in range(budget - n_init):
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
        acq = qLogNoisyExpectedImprovement(
            model=gp, X_baseline=train_Z_norm, sampler=sampler,
        )
        candidate_norm, _ = optimize_acqf(
            acq_function=acq,
            bounds=torch.stack([torch.zeros(k), torch.ones(k)]).double(),
            q=1, num_restarts=10, raw_samples=512,
        )
        z_new = unnormalize(candidate_norm.squeeze(0), bounds).detach().numpy()
        x_new = z_to_x(z_new)
        y_new = objective(x_new)
        all_z.append(z_new)
        all_y.append(y_new)
        best_values.append(max(all_y))

    return {"best_values": best_values, "all_y": all_y, "final_best": max(all_y)}


# ---------------------------------------------------------------------------
# 7. LLM INGREDIENT SELECTION
# ---------------------------------------------------------------------------

LLM_PROMPT = """You are an expert food scientist. We are trying to create a plant-based mozzarella matching animal-based mozzarella on nutrition. Here's an initial set of ingredients. Please choose a subset of 5 ingredients for Bayesian optimization."""


def llm_select_ingredients(df: pd.DataFrame) -> List[int]:
    """Call Claude API to select 5 ingredients. Falls back to built-in reasoning."""
    ingredient_info = []
    fat_norm, sodium_norm = get_nutrition_arrays(df)
    for i, row in df.iterrows():
        ingredient_info.append(
            f"  {i}: {row['name']} (fat_norm={fat_norm[i]:.3f}, sodium_norm={sodium_norm[i]:.3f})"
        )
    ingredient_list = "\n".join(ingredient_info)

    full_prompt = (
        f"{LLM_PROMPT}\n\n"
        f"Target (animal-based mozzarella, normalized): fat={TARGET_FAT_NORM}, sodium={TARGET_SODIUM_NORM}\n\n"
        f"Ingredients (index: name, normalized fat, normalized sodium):\n{ingredient_list}\n\n"
        f"Return ONLY a JSON list of 5 integer indices, e.g. [3, 5, 7, 12, 24].\n"
        f"Pick ingredients that give the best control over both fat and sodium to match the target."
    )

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if api_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=200,
                messages=[{"role": "user", "content": full_prompt}],
            )
            text = response.content[0].text.strip()
            # Extract JSON list
            import re
            match = re.search(r'\[[\d,\s]+\]', text)
            if match:
                selected = json.loads(match.group())
                if len(selected) == 5 and all(0 <= i < len(df) for i in selected):
                    print(f"  LLM API selected: {[df['name'].iloc[i] for i in selected]}")
                    return selected
        except Exception as e:
            print(f"  LLM API call failed ({e}), using built-in reasoning.")

    # Built-in reasoning (Claude's own analysis):
    # To match mozzarella (fat=0.22, sodium=0.50), we need:
    #   - A fat source (but not too high, since oils are ~1.0 and target is 0.22)
    #   - A sodium source (target is 0.50)
    #   - Neutral fillers to dilute
    # Best 5:
    #   coconut oil (idx 1): fat=0.99  -> main fat lever
    #   pea protein (idx 3): sodium=0.80 -> sodium lever + some fat
    #   soy protein isolate (idx 24): sodium=1.0 -> strongest sodium lever
    #   cashews (idx 22): fat=0.44 -> moderate fat, helps fine-tune
    #   tapioca starch (idx 2): fat=0, sodium=0 -> neutral filler for balance
    selected = [1, 3, 24, 22, 2]
    print(f"  LLM (built-in) selected: {[df['name'].iloc[i] for i in selected]}")
    return selected


# ---------------------------------------------------------------------------
# 8. MAIN EXPERIMENT
# ---------------------------------------------------------------------------

def run_experiment(budget: int = 30, n_seeds: int = 5, n_init: int = 5):
    df = load_ingredients()
    d = len(df)
    fat_norm, sodium_norm = get_nutrition_arrays(df)
    objective = make_objective(fat_norm, sodium_norm)

    print(f"Target (normalized): fat={TARGET_FAT_NORM}, sodium={TARGET_SODIUM_NORM}")
    print(f"Ingredients: {d}, Budget: {budget}, Seeds: {n_seeds}\n")

    # Print ingredient table
    print(f"{'Idx':<4} {'Name':<25} {'fat_norm':<10} {'sodium_norm':<12}")
    print("-" * 51)
    for i in range(d):
        print(f"{i:<4} {df['name'].iloc[i]:<25} {fat_norm[i]:<10.3f} {sodium_norm[i]:<12.3f}")
    print()

    # LLM selection
    selected = llm_select_ingredients(df)
    k = len(selected)

    method_names = [
        f"LLM + BO (k={k})",
        f"Vanilla BO (d={d})",
        f"Random Search (d={d})",
    ]
    all_traces = {name: [] for name in method_names}

    for seed in range(n_seeds):
        print(f"Seed {seed+1}/{n_seeds}")

        # LLM + BO
        res = llm_reduced_bo(objective, d, selected, budget=budget, n_init=n_init, seed=seed)
        all_traces[method_names[0]].append(res["best_values"])

        # Vanilla BO
        res = vanilla_bo(objective, d, budget=budget, n_init=n_init, seed=seed)
        all_traces[method_names[1]].append(res["best_values"])

        # Random search
        res = random_search(objective, d, budget=budget, seed=seed)
        all_traces[method_names[2]].append(res["best_values"])

    return all_traces, budget, df, selected, fat_norm, sodium_norm


# ---------------------------------------------------------------------------
# 9. PLOTTING
# ---------------------------------------------------------------------------

def plot_convergence(all_traces: Dict, budget: int, save_path: str = "mozzarella_convergence.png"):
    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))
    iters = np.arange(1, budget + 1)

    for name, traces in all_traces.items():
        traces_arr = np.array(traces)
        mean = traces_arr.mean(axis=0)
        std = traces_arr.std(axis=0)
        mean_dist = -mean  # convert back to distance
        std_dist = std

        if "LLM" in name:
            color, ls, lw = "#2ecc71", "-", 2.5
        elif "Vanilla" in name:
            color, ls, lw = "#e74c3c", "-.", 2
        else:
            color, ls, lw = "#999999", "--", 1.5

        ax.plot(iters, mean_dist, label=name, color=color, linestyle=ls, linewidth=lw)
        ax.fill_between(iters, mean_dist - std_dist, mean_dist + std_dist,
                        alpha=0.15, color=color)

    ax.set_xlabel("Evaluation", fontsize=12)
    ax.set_ylabel("L2 Distance to Mozzarella (lower = better)", fontsize=12)
    ax.set_title("Plant-Based Mozzarella: Nutrition Matching", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, budget)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Convergence plot saved to {save_path}")
    plt.close()


def plot_final_bar(all_traces: Dict, save_path: str = "mozzarella_final.png"):
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.5))
    names = list(all_traces.keys())
    final_means = []
    final_stds = []

    for name in names:
        finals = [-t[-1] for t in all_traces[name]]
        final_means.append(np.mean(finals))
        final_stds.append(np.std(finals))

    colors = []
    for name in names:
        if "LLM" in name:
            colors.append("#2ecc71")
        elif "Vanilla" in name:
            colors.append("#e74c3c")
        else:
            colors.append("#999999")

    ax.bar(range(len(names)), final_means, yerr=final_stds,
           color=colors, capsize=5, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=10)
    ax.set_ylabel("Final L2 Distance (lower = better)", fontsize=11)
    ax.set_title("Final Performance: Plant-Based Mozzarella", fontsize=13)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Bar chart saved to {save_path}")
    plt.close()


def plot_nutrition_space(df, selected, fat_norm, sodium_norm,
                         save_path: str = "mozzarella_nutrition_space.png"):
    """Plot ingredients in (fat, sodium) space with target and LLM selection highlighted."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # All ingredients
    ax.scatter(fat_norm, sodium_norm, s=50, c="#cccccc", edgecolors="black",
               linewidth=0.5, zorder=2, label="Other ingredients")

    # LLM-selected
    ax.scatter(fat_norm[selected], sodium_norm[selected], s=120, c="#2ecc71",
               edgecolors="black", linewidth=1, zorder=3, label="LLM-selected (k=5)")

    # Target
    ax.scatter([TARGET_FAT_NORM], [TARGET_SODIUM_NORM], s=200, c="#e74c3c",
               marker="*", edgecolors="black", linewidth=1, zorder=4,
               label="Mozzarella target")

    # Labels
    for i in range(len(df)):
        offset = (5, 5) if i not in selected else (5, -12)
        fontweight = "bold" if i in selected else "normal"
        fontsize = 8 if i not in selected else 9
        alpha = 0.5 if i not in selected else 1.0
        ax.annotate(df["name"].iloc[i], (fat_norm[i], sodium_norm[i]),
                    textcoords="offset points", xytext=offset,
                    fontsize=fontsize, fontweight=fontweight, alpha=alpha)

    ax.set_xlabel("Normalized Fat (fat_per_100g / 100)", fontsize=12)
    ax.set_ylabel("Normalized Sodium (sodium_per_100g / 1000)", fontsize=12)
    ax.set_title("Ingredient Nutrition Space", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Nutrition space plot saved to {save_path}")
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
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    all_traces, budget, df, selected, fat_norm, sodium_norm = run_experiment(
        budget=args.budget, n_seeds=args.seeds, n_init=args.n_init,
    )

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for name, traces in all_traces.items():
        finals = [-t[-1] for t in traces]
        print(f"  {name:<35} dist = {np.mean(finals):.4f} +/- {np.std(finals):.4f}")

    if not args.no_plot:
        plot_convergence(all_traces, budget)
        plot_final_bar(all_traces)
        plot_nutrition_space(df, selected, fat_norm, sodium_norm)
