"""
Dimension sweep: how does vanilla BO degrade as we add more ingredients?

Compares vanilla BO vs. LLM-reduced BO (always k=5) as the total ingredient
pool grows from 20 to 100. The LLM always picks the same ~5 high-leverage
ingredients, while vanilla BO must search the full space.

Usage:
    python run_dimension_sweep.py [--budget 30] [--seeds 5]
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Allow importing sibling module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_nutrition_experiment import (
    make_nutrition_objective,
    random_search,
    vanilla_bo,
    llm_reduced_bo,
    llm_select_ingredients,
)


def _run_method(method_name, objective, bounds_np, selected, budget, n_init, n_seeds):
    """Run a single method for n_seeds and return aggregated results."""
    traces = []
    for seed in range(n_seeds):
        if method_name == "Random":
            res = random_search(objective, bounds_np, budget=budget, seed=seed)
        elif method_name == "Vanilla BO":
            res = vanilla_bo(objective, bounds_np, budget=budget, n_init=n_init, seed=seed)
        elif method_name == "LLM-Reduced BO":
            res = llm_reduced_bo(objective, bounds_np, selected, budget=budget, n_init=n_init, seed=seed)
        traces.append(res["best_values"])
    traces_arr = np.array(traces)
    final_dists = -traces_arr[:, -1]
    return {
        "mean_trace": -traces_arr.mean(axis=0),
        "std_trace": traces_arr.std(axis=0),
        "mean_final": float(final_dists.mean()),
        "std_final": float(final_dists.std()),
    }


def run_sweep(
    dims: list = [20, 40, 60, 80, 100],
    budget: int = 30,
    n_seeds: int = 5,
    n_init: int = 5,
    target_fat: float = 8.0,
    target_sodium: float = 400.0,
):
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df_full = pd.read_csv(os.path.join(_root, "data", "ingredients_large.csv"))
    assert len(df_full) >= max(dims), f"CSV has {len(df_full)} rows, need {max(dims)}"

    results = {d: {} for d in dims}

    for d in dims:
        print(f"\n--- D={d} ingredients ---")
        df = df_full.iloc[:d].copy()
        bounds_np = np.stack([df["min"].values, df["max"].values]).astype(float)
        objective = make_nutrition_objective(df, target_fat, target_sodium)
        selected = llm_select_ingredients(df, target_fat, target_sodium)

        for method_name in ["Random", "Vanilla BO", "LLM-Reduced BO"]:
            results[d][method_name] = _run_method(
                method_name, objective, bounds_np, selected, budget, n_init, n_seeds
            )
            print(f"  {method_name}: {results[d][method_name]['mean_final']:.4f} ± {results[d][method_name]['std_final']:.4f}")

    return results


_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def plot_sweep(results, budget, save_path=os.path.join(_ROOT, "plots", "dimension_sweep.png")):
    dims = sorted(results.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: final distance vs. dimension
    ax = axes[0]
    for method_name, color, ls in [
        ("Random", "#999999", "--"),
        ("Vanilla BO", "#e74c3c", "-."),
        ("LLM-Reduced BO", "#2ecc71", "-"),
    ]:
        means = [results[d][method_name]["mean_final"] for d in dims]
        stds = [results[d][method_name]["std_final"] for d in dims]
        ax.errorbar(dims, means, yerr=stds, label=method_name, color=color,
                    linestyle=ls, linewidth=2, markersize=8, marker="o", capsize=4)

    ax.set_xlabel("Number of Ingredients (D)", fontsize=12)
    ax.set_ylabel("Final L2 Distance (lower = better)", fontsize=12)
    ax.set_title("Performance vs. Search Space Dimension", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Right: convergence curves for min and max D
    ax = axes[1]
    d_small = dims[0]
    d_large = dims[-1]
    iters = np.arange(1, budget + 1)

    for d, alpha in [(d_small, 0.5), (d_large, 1.0)]:
        for method_name, color, ls in [
            ("Vanilla BO", "#e74c3c", "-."),
            ("LLM-Reduced BO", "#2ecc71", "-"),
        ]:
            mean_trace = results[d][method_name]["mean_trace"]
            label = f"{method_name} (D={d})" if alpha == 1.0 else f"_nolegend_"
            if alpha < 1.0:
                label = f"{method_name} (D={d})"
            ax.plot(iters, mean_trace, label=label, color=color,
                    linestyle=ls, linewidth=2, alpha=alpha)

    ax.set_xlabel("Evaluation", fontsize=12)
    ax.set_ylabel("L2 Distance to Target", fontsize=12)
    ax.set_title(f"Convergence: D={d_small} vs. D={d_large}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--dims", type=int, nargs="+", default=[20, 40, 60, 80, 100])
    args = parser.parse_args()

    results = run_sweep(dims=args.dims, budget=args.budget, n_seeds=args.seeds)
    plot_sweep(results, args.budget)
