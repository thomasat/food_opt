"""Sobol cold start: one scrambled sequence, fast-forwarded past issued points."""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from botorch.utils.sampling import get_polytope_samples
from torch.quasirandom import SobolEngine

COLD_START_RUNS = 5
COLD_START_POOLS = (2048, 8192)
REPEAT_TOLERANCE = 1e-9
# Points the feasible sampler draws per formulation asked for: some fail the
# snap onto the total or repeat a formulation already issued.
FEASIBLE_SAMPLE_DRAWS = 4


def sobol_pool(size: int, dim: int, sobol_seed: int, already: int,
               fixed: dict[int, float]) -> torch.Tensor:
    """`size` normalized points of the project's one Sobol sequence, after
    skipping `already`. A fresh engine each time, so a larger pool opens with
    exactly the points a smaller one held."""
    sobol = SobolEngine(dimension=dim, scramble=True, seed=sobol_seed)
    if already > 0:
        sobol.fast_forward(already)
    pool = sobol.draw(size).double()
    for col, z in fixed.items():
        pool[:, col] = z
    return pool


def repeats(vector: np.ndarray, rows) -> bool:
    """`vector` matches a row of `rows` to REPEAT_TOLERANCE in every column."""
    if rows is None or len(rows) == 0:
        return False
    rows = np.asarray(rows, dtype=float)
    if rows.ndim != 2 or rows.shape[1] != vector.shape[0]:
        return False
    return bool((np.abs(rows - vector).max(axis=1) <= REPEAT_TOLERANCE).any())


def select(candidates: list[dict], n: int, feasible: Callable[[dict], bool],
           as_issued: Callable[[dict], np.ndarray], issued: np.ndarray | None
           ) -> tuple[list[dict], list[dict], list[np.ndarray]]:
    """First `n` feasible candidates that repeat neither `issued` nor each
    other. Returns (results, feasible repeats in sequence order, chosen vectors)."""
    results: list[dict] = []
    skipped: list[dict] = []
    chosen: list[np.ndarray] = []
    for candidate in candidates:
        if not feasible(candidate):
            continue
        vector = as_issued(candidate)
        if repeats(vector, issued) or repeats(vector, chosen):
            skipped.append(candidate)
            continue
        results.append(candidate)
        chosen.append(vector)
        if len(results) >= n:
            break
    return results, skipped, chosen


def fill_from_repeats(repeat_recipes: list[dict], chosen, k: int,
                      as_issued: Callable[[dict], np.ndarray]) -> list[dict]:
    """`k` of `repeat_recipes` (sequence order): first those matching neither
    this Round's picks nor each other, then the rest."""
    seen = list(chosen)
    first, rest = [], []
    for rec in repeat_recipes:
        vector = as_issued(rec)
        if repeats(vector, seen):
            rest.append(rec)
        else:
            first.append(rec)
            seen.append(vector)
    return (first + rest)[:k]


def polytope_points(n: int, dim: int, inequalities, fixed: dict[int, float],
                    seed: int) -> torch.Tensor:
    """`FEASIBLE_SAMPLE_DRAWS * n` normalized points from a hit-and-run chain
    inside the limits (`inequalities`, over the [0, 1] frame), with each
    fixed column pinned as an equality. Run in its own torch random state."""
    equalities = [(torch.tensor([col], dtype=torch.long),
                   torch.tensor([1.0], dtype=torch.double), float(z))
                  for col, z in fixed.items()] or None
    with torch.random.fork_rng(devices=[]):
        return get_polytope_samples(
            n=FEASIBLE_SAMPLE_DRAWS * n,
            bounds=torch.stack([torch.zeros(dim), torch.ones(dim)]).double(),
            inequality_constraints=inequalities or None,
            equality_constraints=equalities,
            seed=int(seed), n_burnin=2000, n_thinning=64)
