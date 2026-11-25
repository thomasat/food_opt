"""Search space: a port of food_bo.py _encode / _decode / _search_bounds /
_get_bounds / _get_fixed_features / _settings_on_their_step (continuous
variables only)."""
from __future__ import annotations

from dataclasses import dataclass

import torch

_FIXED_SPAN_FLOOR = 1e-9


@dataclass
class Variable:
    name: str
    low: float
    high: float
    kind: str = "ingredient"     # "ingredient" | "setting"
    step: float | None = None    # settings only
    # Value in rows recorded before the variable existed: 0 for an ingredient,
    # the baseline for a setting (None when no baseline was given).
    absent: float | None = None

    @property
    def fixed(self) -> bool:
        return float(self.low) == float(self.high)


def encode(variables: list[Variable], recipe: dict) -> list[float]:
    return [float(recipe.get(v.name, 0.0 if v.absent is None else v.absent))
            for v in variables]


def decode(variables: list[Variable], vector) -> dict[str, float]:
    return {v.name: float(vector[i]) for i, v in enumerate(variables)}


def search_bounds(variables: list[Variable], recipes: list[dict],
                  model_rows: list[list[float]]) -> list[tuple[float, float]]:
    spans = []
    for col, var in enumerate(variables):
        lo, hi = float(var.low), float(var.high)
        if var.absent is not None and any(var.name not in r for r in recipes):
            lo, hi = min(lo, float(var.absent)), max(hi, float(var.absent))
        if hi <= lo:
            seen = [float(row[col]) for row in model_rows if col < len(row)]
            lo, hi = min([lo] + seen), max([hi] + seen)
            if hi - lo <= _FIXED_SPAN_FLOOR * max(1.0, abs(lo)):
                hi = lo + 1.0
        spans.append((lo, hi))
    return spans


def bounds_tensor(spans) -> torch.Tensor:
    return torch.tensor([[lo for lo, _ in spans], [hi for _, hi in spans]],
                        dtype=torch.double)


def fixed_features(variables: list[Variable], spans) -> dict[int, float]:
    fixed = {}
    for col, var in enumerate(variables):
        if var.fixed:
            lo, hi = spans[col]
            span = hi - lo
            z = 0.0 if span <= 0 else (float(var.low) - lo) / span
            fixed[col] = float(min(max(z, 0.0), 1.0))
    return fixed


def active_names(variables: list[Variable]) -> list[str]:
    return [v.name for v in variables if not v.fixed]


def snap_settings(variables: list[Variable], recipe: dict) -> dict:
    out = dict(recipe)
    for var in variables:
        if var.kind != "setting":
            continue
        step = var.step
        value = out.get(var.name)
        if not step or value is None:
            continue
        snapped = round(float(value) / float(step)) * float(step)
        out[var.name] = min(max(snapped, float(var.low)), float(var.high))
    return out


def ingredient_total(variables: list[Variable], recipe: dict) -> float:
    return sum(float(recipe.get(v.name, 0.0)) for v in variables
               if v.kind == "ingredient")
