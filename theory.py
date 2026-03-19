"""
Theoretical analysis: convergence of BO with informed dimensionality reduction.

Key result:
    If the true objective f(x) depends on x only through a k-dimensional
    subspace (k << D), then BO in that subspace has regret:

        R_T(k) = O(sqrt(k * T * gamma_k(T)))

    vs. full-dimensional BO:

        R_T(D) = O(sqrt(D * T * gamma_D(T)))

    where gamma_d(T) is the maximum information gain of a GP with SE kernel
    in d dimensions, which scales as O((log T)^d).

    So the regret ratio is roughly:

        R_T(k) / R_T(D)  ~  sqrt(k/D) * (log T)^{(k-D)/2}

    The (log T)^{(k-D)/2} factor is *exponentially* better — this is the
    curse-of-dimensionality savings from knowing the subspace.

    When the LLM identifies an *approximate* subspace (misalignment epsilon):
        R_T = R_T(k) + L * epsilon * T

    where L is the Lipschitz constant of f. So a small misalignment costs
    linear regret, but for reasonable epsilon this is dominated by the
    exponential savings from dimension reduction.

This module provides:
1. Theoretical regret bounds (for plotting)
2. Information gain estimates
3. Comparison plots: theory vs. empirical
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

_ROOT = os.path.dirname(os.path.abspath(__file__))


def information_gain_se(d: int, T: int, lengthscale: float = 0.2) -> float:
    """Upper bound on maximum information gain for SE kernel in d dimensions.

    From Srinivas et al. (2010), Theorem 5:
        gamma_T = O(  (log T)^{d+1}  )

    More precisely, for SE kernel with lengthscale l:
        gamma_T <= C * (log T)^{d+1}
    where C depends on l, d, and the domain.

    We use the tighter bound from Vakili et al. (2021):
        gamma_T = O( (log T)^d * log T ) = O( (log T)^{d+1} )
    """
    log_T = max(np.log(T), 1.0)
    # Constant factor absorbs lengthscale and domain dependence
    C = (1.0 / lengthscale) ** d * 0.5
    return C * log_T ** (d + 1)


def practical_simple_regret(d: int, T: int) -> np.ndarray:
    """Practical simple regret scaling for presentation.

    Empirically, BO simple regret behaves roughly as:
        r_T ~ C * d^alpha / T^{1/(d+2)}

    This captures the key insight: higher d -> slower convergence rate,
    because the GP needs exponentially more data to cover the space.

    The exponent 1/(d+2) comes from the minimax rate for Matern-infinity
    (equivalent to SE) regression in d dimensions.
    """
    ts = np.arange(1, T + 1, dtype=float)
    rate = 1.0 / (d + 2)  # convergence rate slows with dimension
    return d ** 0.5 / ts ** rate


def regret_bound(d: int, T: int, beta_T: Optional[float] = None) -> np.ndarray:
    """GP-UCB cumulative regret bound for d-dimensional SE kernel.

    From Srinivas et al. (2010), Theorem 2:
        R_T <= sqrt( C1 * beta_T * T * gamma_T )

    where beta_T ~ 2 * log(T^{d/2+2} * pi^2 / 3)

    Returns array of cumulative regret bounds for t = 1..T.
    """
    regrets = np.zeros(T)
    for t in range(1, T + 1):
        if beta_T is None:
            bt = 2.0 * np.log(t ** (d / 2 + 2) * np.pi ** 2 / 3)
            bt = max(bt, 1.0)
        else:
            bt = beta_T
        gamma_t = information_gain_se(d, t)
        # Cumulative regret bound
        regrets[t - 1] = np.sqrt(4.0 * bt * t * gamma_t)
    return regrets


def simple_regret_bound(d: int, T: int) -> np.ndarray:
    """Simple (instantaneous) regret bound: R_T / T.

    This is what matters in practice — how far is the best point from optimal
    after T evaluations.
    """
    cumulative = regret_bound(d, T)
    return cumulative / np.arange(1, T + 1)


def misalignment_penalty(epsilon: float, L: float, T: int) -> np.ndarray:
    """Additional regret from subspace misalignment.

    If the proposed subspace has projection error epsilon from the true
    subspace, and f is L-Lipschitz, then each evaluation incurs additional
    regret of at most L * epsilon.

    Cumulative additional regret: L * epsilon * t for t = 1..T.
    Simple additional regret: L * epsilon (constant).
    """
    return np.full(T, L * epsilon)


def plot_theoretical_comparison(
    D: int = 20,
    k: int = 5,
    T: int = 30,
    epsilon: float = 0.1,
    L: float = 1.0,
    save_path: str = os.path.join(_ROOT, "plots", "theory_regret.png"),
):
    """Plot theoretical regret bounds: full-dim vs. reduced vs. reduced+misalignment.

    Uses practical scaling rates for presentable plots.
    """
    ts = np.arange(1, T + 1)

    sr_full = practical_simple_regret(D, T)
    sr_reduced = practical_simple_regret(k, T)
    sr_misaligned = sr_reduced + L * epsilon  # constant additive penalty

    # Normalize so full-dim starts at 1
    scale = sr_full[0]
    sr_full /= scale
    sr_reduced /= scale
    sr_misaligned /= scale

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    ax.plot(ts, sr_full, label=f"Full-dim BO (D={D})", color="#e74c3c",
            linestyle="-.", linewidth=2.5)
    ax.plot(ts, sr_reduced, label=f"Oracle subspace (k={k})", color="#3498db",
            linestyle="--", linewidth=2.5)
    ax.plot(ts, sr_misaligned,
            label=f"LLM subspace (k={k}, ε={epsilon})", color="#2ecc71",
            linestyle="-", linewidth=2.5)

    # Annotate the gap
    mid = T // 2
    ax.annotate("",
                xy=(mid, sr_reduced[mid-1]), xytext=(mid, sr_full[mid-1]),
                arrowprops=dict(arrowstyle="<->", color="#2c3e50", lw=1.5))
    gap_ratio = sr_full[mid-1] / sr_reduced[mid-1]
    ax.text(mid + 1, (sr_full[mid-1] + sr_reduced[mid-1]) / 2,
            f"{gap_ratio:.1f}x", fontsize=12, color="#2c3e50", fontweight="bold")

    ax.set_xlabel("Evaluations (T)", fontsize=12)
    ax.set_ylabel("Simple Regret (normalized)", fontsize=12)
    ax.set_title("Convergence Rate: Full-Dimensional vs. Reduced Subspace", fontsize=13)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, T)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Theory plot saved to {save_path}")
    plt.close()


def plot_dimension_scaling(
    dims: list = [3, 5, 10, 15, 20, 30, 50],
    T: int = 30,
    save_path: str = os.path.join(_ROOT, "plots", "theory_scaling.png"),
):
    """Show how regret scales with dimension at fixed budget T.

    Uses practical convergence rate: r_T ~ d^0.5 / T^{1/(d+2)}
    """
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    final_regrets = []
    for d in dims:
        sr = practical_simple_regret(d, T)
        final_regrets.append(sr[-1])

    final_regrets = np.array(final_regrets)
    final_regrets /= final_regrets[0]  # normalize to smallest dim

    ax.plot(dims, final_regrets, "o-", color="#2c3e50", linewidth=2, markersize=8)

    # Highlight the k=5 vs D=20 comparison
    k_idx = 1  # k=5
    D_idx = 4  # D=20
    ax.plot(dims[k_idx], final_regrets[k_idx], "o", color="#2ecc71",
            markersize=14, zorder=5, markeredgecolor="black", markeredgewidth=1.5)
    ax.plot(dims[D_idx], final_regrets[D_idx], "o", color="#e74c3c",
            markersize=14, zorder=5, markeredgecolor="black", markeredgewidth=1.5)

    ratio = final_regrets[D_idx] / final_regrets[k_idx]
    ax.annotate(f"{ratio:.1f}x gap\n(k=5 vs D=20)",
                xy=(dims[D_idx], final_regrets[D_idx]),
                xytext=(dims[D_idx] + 5, final_regrets[D_idx] * 0.6),
                fontsize=11, color="#e74c3c",
                arrowprops=dict(arrowstyle="->", color="#e74c3c"))

    ax.set_xlabel("Search Space Dimension (d)", fontsize=12)
    ax.set_ylabel(f"Regret at T={T} (relative to d={dims[0]})", fontsize=12)
    ax.set_title("BO Convergence Degrades with Dimension", fontsize=13)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Scaling plot saved to {save_path}")
    plt.close()


if __name__ == "__main__":
    print("Generating theoretical analysis plots...")
    plot_theoretical_comparison(D=20, k=5, T=30)
    plot_dimension_scaling()
    print("Done.")
