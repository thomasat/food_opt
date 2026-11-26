"""Measurement scoring: a port of food_bo.py score_parts / _compute_utility
(the branch where every measurement applies)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Objective:
    name: str
    weight: float
    goal: str = "max"            # "max" | "min" | "target"
    target: float | None = None  # defaults to (min_val + max_val) / 2
    min_val: float = 0.0
    max_val: float = 10.0


def utility(obj: Objective, value: float) -> float:
    val = float(value)
    min_v, max_v = obj.min_val, obj.max_val
    rng = max_v - min_v
    if rng == 0:
        rng = 1.0
    norm_val = max(0.0, min(1.0, (val - min_v) / rng))
    if obj.goal == "max":
        return norm_val
    if obj.goal == "min":
        return 1.0 - norm_val
    if obj.goal == "target":
        targ_val = obj.target if obj.target is not None else (max_v + min_v) / 2
        norm_targ = (targ_val - min_v) / rng
        return max(0.0, 1.0 - abs(norm_val - norm_targ))
    return 0.0


def score(objectives: list[Objective], results: dict[str, float]) -> float:
    total = 0.0
    for obj in objectives:
        raw = results.get(obj.name)
        if raw is None:
            continue
        total += obj.weight * utility(obj, raw)
    return total
