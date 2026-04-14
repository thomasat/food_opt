"""
BO Simulation Comparisons — 5 Key Insights.

Produces a single multi-panel figure illustrating:

1. Vanilla BO struggles in high dimensions (dim scaling).
2. Knowing the right variable set helps a lot (oracle gap).
3. Even moderate expertise helps a lot (expert quality).
4. Dependence on expert accuracy p/q (heatmap).
5. Adaptive expansion helps (adaptive vs. static expert).

Defaults target Hartmann6 at D=500, where vanilla BO clearly fails and
oracle BO remains effective. Reuses all BO methods from
`run_expert_simulation.py`; the only new method is `run_static_expert_bo`
(for insight 5).

Usage:
    # Quick smoke test
    python experiments/run_bo_comparisons.py --experiments dim_scaling \\
        --budget 20 --seeds 2

    # Full run at D=500
    python experiments/run_bo_comparisons.py --dim 500 --budget 50 --seeds 5
"""

import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.quasirandom import SobolEngine

# Allow importing sibling module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_expert_simulation as res  # noqa: E402
from run_expert_simulation import (  # noqa: E402
    ExpertConfig,
    EXPERT_CONFIGS,
    expert_update_variables,
    run_vanilla_bo,
    run_random_search,
    run_expert_bo,
    _fit_and_suggest,
    _set_ambient_dim,
    _set_test_function,
    D_RELEVANT,
)


# ---------------------------------------------------------------------------
# 1. NEW BO METHOD: STATIC EXPERT
# ---------------------------------------------------------------------------

def run_static_expert_bo(
    budget: int,
    n_init: int,
    seed: int,
    noise_std: float,
    config: ExpertConfig,
    n_selection_rounds: int = 5,
) -> Dict:
    """BO with a one-shot expert selection.

    The expert runs `n_selection_rounds` rounds of `expert_update_variables`
    up-front, then the variable set is frozen. BO is run on that fixed
    subspace for the full budget. Inactive dims fixed at 0.5.
    """
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    np.random.seed(seed)

    # --- One-shot selection ---
    active_vars: set = set()
    for _ in range(n_selection_rounds):
        active_vars = expert_update_variables(active_vars, config, rng)

    active_list = sorted(active_vars)
    d_active = len(active_list)
    d_total = res.D_TOTAL

    all_x_full, all_y, best_values = [], [], []

    if d_active == 0:
        # Degenerate: nothing selected, fall back to random in full space.
        for _ in range(budget):
            x_full = rng.random(d_total)
            y = res._eval_fn(x_full, noise_std)
            all_x_full.append(x_full)
            all_y.append(y)
            best_values.append(max(all_y))
        return {
            "best_values": best_values,
            "all_y": all_y,
            "final_best": max(all_y),
            "frozen_set_size": 0,
            "frozen_n_relevant": 0,
        }

    # --- Sobol init in the frozen subspace ---
    sobol = SobolEngine(dimension=d_active, scramble=True, seed=seed)
    x_init_sub = sobol.draw(n_init).double().numpy()

    for i in range(n_init):
        x_full = np.full(d_total, 0.5)
        x_full[active_list] = x_init_sub[i]
        y = res._eval_fn(x_full, noise_std)
        all_x_full.append(x_full)
        all_y.append(y)
        best_values.append(max(all_y))

    # --- BO on frozen subspace ---
    for step in range(budget - n_init):
        X_sub = np.array([x[active_list] for x in all_x_full])
        train_X = torch.tensor(X_sub, dtype=torch.double)
        train_Y = torch.tensor(all_y, dtype=torch.double).unsqueeze(-1)

        try:
            cand = _fit_and_suggest(train_X, train_Y, d_active, seed + step)
            x_sub = cand.detach().numpy()
        except Exception:
            x_sub = rng.random(d_active)

        x_full = np.full(d_total, 0.5)
        x_full[active_list] = x_sub
        y = res._eval_fn(x_full, noise_std)
        all_x_full.append(x_full)
        all_y.append(y)
        best_values.append(max(all_y))

    return {
        "best_values": best_values,
        "all_y": all_y,
        "final_best": max(all_y),
        "frozen_set_size": d_active,
        "frozen_n_relevant": len(res.RELEVANT_SET & active_vars),
    }


# ---------------------------------------------------------------------------
# 2. D-AWARE EXPERT CONFIGS
# ---------------------------------------------------------------------------

def scaled_expert_configs(d_total: int) -> Dict[str, ExpertConfig]:
    """Rescale q by 100/D so expected irrelevant additions per iteration
    stay comparable to the original D=100 setting."""
    scale = 100.0 / max(d_total, 1)
    return {
        "oracle":        ExpertConfig("oracle",        p=1.0,  q=0.0),
        "expert_good":   ExpertConfig("expert_good",   p=0.3,  q=0.01 * scale),
        "expert_medium": ExpertConfig("expert_medium", p=0.15, q=0.03 * scale),
        "expert_poor":   ExpertConfig("expert_poor",   p=0.05, q=0.06 * scale),
    }


# ---------------------------------------------------------------------------
# 3. SUB-EXPERIMENTS
# ---------------------------------------------------------------------------

def _run_seeds(runner, n_seeds: int, **kwargs) -> Dict:
    """Run `runner(seed=..., **kwargs)` across seeds, collect traces."""
    traces = []
    for seed in range(n_seeds):
        r = runner(seed=seed, **kwargs)
        traces.append(r["best_values"])
    traces = np.array(traces)
    return {
        "mean_trace": traces.mean(axis=0),
        "std_trace": traces.std(axis=0),
        "all_traces": traces,
        "mean_final": float(traces[:, -1].mean()),
        "std_final": float(traces[:, -1].std()),
    }


def experiment_dimension_scaling(
    dims: List[int],
    budget: int,
    n_init: int,
    n_seeds: int,
    noise_std: float,
) -> Dict:
    """For each D, run vanilla BO and oracle; show regret vs D."""
    results = {}
    for D in dims:
        print(f"  [dim_scaling] D={D}")
        _set_ambient_dim(D)
        configs = scaled_expert_configs(D)

        vanilla = _run_seeds(
            run_vanilla_bo, n_seeds,
            budget=budget, n_init=n_init, noise_std=noise_std,
        )
        oracle = _run_seeds(
            run_expert_bo, n_seeds,
            budget=budget, n_init=n_init, noise_std=noise_std,
            config=configs["oracle"],
        )
        results[D] = {"vanilla_bo": vanilla, "oracle": oracle}
        print(f"    vanilla: {vanilla['mean_final']:.4f}  oracle: {oracle['mean_final']:.4f}")
    return results


def experiment_oracle_vs_vanilla(
    budget: int, n_init: int, n_seeds: int, noise_std: float,
) -> Dict:
    """Convergence curves: random, vanilla, oracle at current D."""
    configs = scaled_expert_configs(res.D_TOTAL)
    out = {}
    for name, runner, extra in [
        ("random",  run_random_search, {}),
        ("vanilla", run_vanilla_bo,    {}),
        ("oracle",  run_expert_bo,     {"config": configs["oracle"]}),
    ]:
        print(f"  [oracle_vs_vanilla] {name}")
        out[name] = _run_seeds(
            runner, n_seeds,
            budget=budget, n_init=n_init, noise_std=noise_std, **extra,
        )
    return out


def experiment_expert_quality(
    budget: int, n_init: int, n_seeds: int, noise_std: float,
) -> Dict:
    """Convergence curves: vanilla, oracle, expert_good/medium/poor."""
    configs = scaled_expert_configs(res.D_TOTAL)
    out = {}
    print("  [expert_quality] vanilla")
    out["vanilla"] = _run_seeds(
        run_vanilla_bo, n_seeds,
        budget=budget, n_init=n_init, noise_std=noise_std,
    )
    for name in ["oracle", "expert_good", "expert_medium", "expert_poor"]:
        print(f"  [expert_quality] {name}")
        out[name] = _run_seeds(
            run_expert_bo, n_seeds,
            budget=budget, n_init=n_init, noise_std=noise_std,
            config=configs[name],
        )
    return out


def experiment_pq_heatmap(
    p_values: List[float],
    q_values: List[float],
    budget: int,
    n_init: int,
    n_seeds: int,
    noise_std: float,
) -> Dict:
    """Sweep expert (p, q) grid; report mean final regret."""
    regret = np.full((len(q_values), len(p_values)), np.nan)
    mean_final = np.full_like(regret, np.nan)
    for i, q in enumerate(q_values):
        for j, p in enumerate(p_values):
            cfg = ExpertConfig(f"p{p}_q{q}", p=p, q=q)
            print(f"  [pq_heatmap] p={p:.3f} q={q:.4f}")
            r = _run_seeds(
                run_expert_bo, n_seeds,
                budget=budget, n_init=n_init, noise_std=noise_std,
                config=cfg,
            )
            mean_final[i, j] = r["mean_final"]
            regret[i, j] = res.GLOBAL_MAX - r["mean_final"]
    return {
        "p_values": list(p_values),
        "q_values": list(q_values),
        "regret": regret,
        "mean_final": mean_final,
    }


def experiment_adaptive_vs_static(
    budget: int,
    n_init: int,
    n_seeds: int,
    noise_std: float,
    config_names: List[str] = ("expert_good", "expert_medium"),
) -> Dict:
    """Compare adaptive expansion vs frozen one-shot expert selection."""
    configs = scaled_expert_configs(res.D_TOTAL)
    out = {}
    for name in config_names:
        cfg = configs[name]
        print(f"  [adaptive_vs_static] {name} adaptive")
        adaptive = _run_seeds(
            run_expert_bo, n_seeds,
            budget=budget, n_init=n_init, noise_std=noise_std,
            config=cfg,
        )
        print(f"  [adaptive_vs_static] {name} static")
        static = _run_seeds(
            run_static_expert_bo, n_seeds,
            budget=budget, n_init=n_init, noise_std=noise_std,
            config=cfg, n_selection_rounds=n_init,
        )
        out[name] = {"adaptive": adaptive, "static": static}
    return out


# ---------------------------------------------------------------------------
# 4. PLOTTING
# ---------------------------------------------------------------------------

def _plot_dimension_scaling(results: Dict, ax):
    dims = sorted(results.keys())
    global_max = res.GLOBAL_MAX
    for method, color, label in [
        ("vanilla_bo", "tab:red", "Vanilla BO"),
        ("oracle",     "black",   "Oracle (d=6)"),
    ]:
        regrets = [global_max - results[D][method]["mean_final"] for D in dims]
        stds = [results[D][method]["std_final"] for D in dims]
        ax.errorbar(dims, regrets, yerr=stds, marker="o", color=color,
                    label=label, linewidth=2, capsize=4)
    ax.set_xlabel("Ambient dim D")
    ax.set_ylabel("Simple regret (lower = better)")
    ax.set_title("1. Vanilla BO degrades with D")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()


def _plot_convergence_panel(results: Dict, ax, title: str,
                            styles: Optional[Dict] = None):
    global_max = res.GLOBAL_MAX
    default_palette = {
        "random":        ("#999999", ":",  "Random"),
        "vanilla":       ("tab:red", "--", "Vanilla BO"),
        "oracle":        ("black",   "-",  "Oracle"),
        "expert_good":   ("tab:green",  "-", "Expert good"),
        "expert_medium": ("tab:orange", "-", "Expert medium"),
        "expert_poor":   ("tab:purple", "-", "Expert poor"),
    }
    palette = {**default_palette, **(styles or {})}

    for name, data in results.items():
        color, ls, label = palette.get(name, ("grey", "-", name))
        mean = np.asarray(data["mean_trace"])
        std = np.asarray(data["std_trace"])
        iters = np.arange(1, len(mean) + 1)
        ax.plot(iters, mean, color=color, ls=ls, label=label, linewidth=2)
        ax.fill_between(iters, mean - std, mean + std, color=color, alpha=0.12)

    ax.axhline(global_max, color="gold", ls=":", lw=1.5,
               label=f"Optimum ({global_max:.2f})")
    ax.set_xlabel("Evaluation")
    ax.set_ylabel("Best value found")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")


def _plot_pq_heatmap(results: Dict, ax):
    regret = results["regret"]
    p_vals = results["p_values"]
    q_vals = results["q_values"]
    im = ax.imshow(regret, aspect="auto", origin="lower", cmap="viridis_r")
    ax.set_xticks(range(len(p_vals)))
    ax.set_xticklabels([f"{p:g}" for p in p_vals])
    ax.set_yticks(range(len(q_vals)))
    ax.set_yticklabels([f"{q:g}" for q in q_vals])
    ax.set_xlabel("p (prob. adding relevant)")
    ax.set_ylabel("q (prob. adding irrelevant)")
    ax.set_title("4. Expert accuracy p/q → regret")
    plt.colorbar(im, ax=ax, label="Simple regret")
    # annotate cells
    for i in range(regret.shape[0]):
        for j in range(regret.shape[1]):
            val = regret[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color="white", fontsize=7)


def _plot_adaptive_vs_static(results: Dict, ax):
    color_by_cfg = {"expert_good": "tab:green", "expert_medium": "tab:orange"}
    for cfg_name, pair in results.items():
        color = color_by_cfg.get(cfg_name, "grey")
        for variant, ls, suffix in [("adaptive", "-", "adaptive"),
                                    ("static",   "--", "static")]:
            data = pair[variant]
            mean = np.asarray(data["mean_trace"])
            std = np.asarray(data["std_trace"])
            iters = np.arange(1, len(mean) + 1)
            ax.plot(iters, mean, color=color, ls=ls, linewidth=2,
                    label=f"{cfg_name} ({suffix})")
            ax.fill_between(iters, mean - std, mean + std,
                            color=color, alpha=0.10)
    ax.axhline(res.GLOBAL_MAX, color="gold", ls=":", lw=1.5,
               label=f"Optimum ({res.GLOBAL_MAX:.2f})")
    ax.set_xlabel("Evaluation")
    ax.set_ylabel("Best value found")
    ax.set_title("5. Adaptive expansion vs. frozen selection")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")


def _plot_summary_bar(all_results: Dict, ax):
    """Summary bar chart of final regret across the headline methods.
    Uses the expert-quality experiment when available."""
    quality = all_results.get("quality")
    if quality is None:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "(run expert_quality for summary bar)",
                ha="center", va="center", transform=ax.transAxes)
        return
    order = ["vanilla", "expert_poor", "expert_medium", "expert_good", "oracle"]
    labels = [o for o in order if o in quality]
    means = [res.GLOBAL_MAX - quality[m]["mean_final"] for m in labels]
    stds  = [quality[m]["std_final"] for m in labels]
    colors = {"vanilla": "tab:red", "expert_poor": "tab:purple",
              "expert_medium": "tab:orange", "expert_good": "tab:green",
              "oracle": "black"}
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, color=[colors[m] for m in labels],
           alpha=0.85, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Simple regret")
    ax.set_title(f"Summary @ D={res.D_TOTAL}")
    ax.grid(axis="y", alpha=0.3)


def plot_combined(all_results: Dict, save_path: str):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    if "dim_scaling" in all_results:
        _plot_dimension_scaling(all_results["dim_scaling"], axes[0, 0])
    else:
        axes[0, 0].set_axis_off()

    if "oracle" in all_results:
        _plot_convergence_panel(all_results["oracle"], axes[0, 1],
                                "2. Oracle vs. vanilla BO")
    else:
        axes[0, 1].set_axis_off()

    if "quality" in all_results:
        _plot_convergence_panel(all_results["quality"], axes[0, 2],
                                "3. Expert quality matters")
    else:
        axes[0, 2].set_axis_off()

    if "pq_heatmap" in all_results:
        _plot_pq_heatmap(all_results["pq_heatmap"], axes[1, 0])
    else:
        axes[1, 0].set_axis_off()

    if "adaptive" in all_results:
        _plot_adaptive_vs_static(all_results["adaptive"], axes[1, 1])
    else:
        axes[1, 1].set_axis_off()

    _plot_summary_bar(all_results, axes[1, 2])

    fig.suptitle(
        f"BO Comparisons: Hartmann6 in D={res.D_TOTAL}",
        fontsize=15, y=0.995,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"Saved combined figure to {save_path}")
    plt.close(fig)


def plot_individual(all_results: Dict, save_dir: str):
    """Also save each panel as a standalone figure."""
    def _save(name, plot_fn, **kwargs):
        fig, ax = plt.subplots(figsize=(7, 5))
        plot_fn(ax=ax, **kwargs)
        fig.tight_layout()
        path = os.path.join(save_dir, f"bo_cmp_{name}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  saved {path}")

    if "dim_scaling" in all_results:
        _save("1_dim_scaling",
              lambda ax: _plot_dimension_scaling(all_results["dim_scaling"], ax))
    if "oracle" in all_results:
        _save("2_oracle",
              lambda ax: _plot_convergence_panel(
                  all_results["oracle"], ax, "Oracle vs. vanilla BO"))
    if "quality" in all_results:
        _save("3_quality",
              lambda ax: _plot_convergence_panel(
                  all_results["quality"], ax, "Expert quality matters"))
    if "pq_heatmap" in all_results:
        _save("4_pq_heatmap",
              lambda ax: _plot_pq_heatmap(all_results["pq_heatmap"], ax))
    if "adaptive" in all_results:
        _save("5_adaptive",
              lambda ax: _plot_adaptive_vs_static(all_results["adaptive"], ax))


# ---------------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------------

ALL_EXPERIMENTS = ["dim_scaling", "oracle", "quality", "pq_heatmap", "adaptive"]


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="BO comparisons: 5 insights about expert-guided BO")
    parser.add_argument("--experiments", nargs="+",
                        choices=ALL_EXPERIMENTS + ["all"],
                        default=["all"])
    parser.add_argument("--dim", type=int, default=500,
                        help="Headline ambient dim (exps 2-5)")
    parser.add_argument("--dim-sweep", type=int, nargs="+",
                        default=[50, 100, 250, 500, 1000],
                        help="Dim list for experiment 1")
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--function", default="hartmann6",
                        choices=["hartmann6", "ackley"])
    parser.add_argument("--save-dir", default="plots")
    parser.add_argument("--individual-figures", action="store_true")
    # quick/cheap p/q heatmap defaults (still informative, much faster)
    parser.add_argument("--pq-p", type=float, nargs="+",
                        default=[0.05, 0.1, 0.2, 0.3, 0.5])
    parser.add_argument("--pq-q", type=float, nargs="+",
                        default=[0.0, 0.002, 0.005, 0.01, 0.02, 0.05])
    parser.add_argument("--quick", action="store_true",
                        help="Smaller grid / fewer seeds for fast iteration")
    args = parser.parse_args()

    if args.quick:
        args.pq_p = [0.05, 0.2, 0.5]
        args.pq_q = [0.0, 0.005, 0.02]
        args.seeds = max(2, args.seeds // 2)

    exps = set(args.experiments)
    if "all" in exps:
        exps = set(ALL_EXPERIMENTS)

    _set_test_function(args.function)
    os.makedirs(args.save_dir, exist_ok=True)
    all_results: Dict = {}

    print(f"Config: function={args.function}, D={args.dim}, budget={args.budget}, "
          f"n_init={args.n_init}, seeds={args.seeds}, noise={args.noise}")

    if "dim_scaling" in exps:
        print("\n=== Experiment 1: Dimension Scaling ===")
        all_results["dim_scaling"] = experiment_dimension_scaling(
            dims=args.dim_sweep,
            budget=args.budget, n_init=args.n_init,
            n_seeds=args.seeds, noise_std=args.noise,
        )

    # Set the headline dim for experiments 2-5
    _set_ambient_dim(args.dim)

    if "oracle" in exps:
        print("\n=== Experiment 2: Oracle vs Vanilla ===")
        all_results["oracle"] = experiment_oracle_vs_vanilla(
            budget=args.budget, n_init=args.n_init,
            n_seeds=args.seeds, noise_std=args.noise,
        )

    if "quality" in exps:
        print("\n=== Experiment 3: Expert Quality ===")
        all_results["quality"] = experiment_expert_quality(
            budget=args.budget, n_init=args.n_init,
            n_seeds=args.seeds, noise_std=args.noise,
        )

    if "pq_heatmap" in exps:
        print("\n=== Experiment 4: p/q Heatmap ===")
        all_results["pq_heatmap"] = experiment_pq_heatmap(
            p_values=args.pq_p, q_values=args.pq_q,
            budget=args.budget, n_init=args.n_init,
            n_seeds=args.seeds, noise_std=args.noise,
        )

    if "adaptive" in exps:
        print("\n=== Experiment 5: Adaptive vs Static ===")
        all_results["adaptive"] = experiment_adaptive_vs_static(
            budget=args.budget, n_init=args.n_init,
            n_seeds=args.seeds, noise_std=args.noise,
        )

    combined_path = os.path.join(args.save_dir, "bo_comparisons.png")
    plot_combined(all_results, combined_path)
    if args.individual_figures:
        plot_individual(all_results, args.save_dir)

    # --- terse text summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if "quality" in all_results:
        q = all_results["quality"]
        for m in ["vanilla", "expert_poor", "expert_medium", "expert_good", "oracle"]:
            if m in q:
                r = res.GLOBAL_MAX - q[m]["mean_final"]
                print(f"  {m:<15} regret = {r:.4f}  (final={q[m]['mean_final']:.4f})")
    if "dim_scaling" in all_results:
        print("\nDim scaling (vanilla vs oracle final value):")
        for D, r in sorted(all_results["dim_scaling"].items()):
            print(f"  D={D:<5} vanilla={r['vanilla_bo']['mean_final']:.4f}  "
                  f"oracle={r['oracle']['mean_final']:.4f}")


if __name__ == "__main__":
    main()
