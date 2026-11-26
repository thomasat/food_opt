"""The Optimizer: a port of food_bo.py FoodOptimizer's set-up, active set and
ask/tell, for continuous ingredients and process settings."""
from __future__ import annotations

import math
import numbers
import dataclasses
import os

import numpy as np
import torch
from botorch.exceptions.errors import BotorchError
from botorch.utils.transforms import unnormalize

from egbo._limits import (
    PropertyLimit,
    QuantityLimit,
    _property_limit_refusal,
    _quantity_limit_refusal,
    at_total,
    check,
    limited_total_reach,
    linear_constraints,
    missing_property_values,
    off_total,
    project_onto_total,
    snapped_if_it_still_fits,
    total_band,
    total_reach,
    widened,
)
from egbo._model import check_bo_config, suggest, validate_bo_config
from egbo import _state
from egbo._score import Objective, score
from egbo._space import (
    Variable,
    active_names,
    bounds_tensor,
    decode,
    encode,
    fixed_features,
    search_bounds,
    snap_settings,
)
from egbo._start import (
    COLD_START_POOLS,
    COLD_START_RUNS,
    fill_from_repeats,
    polytope_points,
    repeats as _repeats,
    select,
    sobol_pool,
)

NO_FORMULATION = "No formulation meets all the limits"
_TOTAL = "formulation_total"


def _num(value) -> str:
    return f"{float(value):g}"


class Optimizer:
    """Expert-guided batch Bayesian optimizer: you choose the variables, `ask` suggests a batch, `tell` records a result."""

    def __init__(self, seed: int | None = None, *, robust: bool = False,
                 bo_config: dict | None = None, n_init: int = COLD_START_RUNS):
        # seed is the Sobol scramble only; warm rounds seed on the counter.
        self.sobol_seed = (int.from_bytes(os.urandom(4), "big") if seed is None
                           else int(seed))
        self.robust = bool(robust)
        check_bo_config(bo_config)
        self.bo_config = validate_bo_config(bo_config)
        if (isinstance(n_init, bool) or not isinstance(n_init, numbers.Real)
                or n_init != int(n_init) or int(n_init) < 1):
            raise ValueError(f"n_init must be a whole number of at least 1, "
                             f"not {n_init!r}")
        self.n_init = int(n_init)
        self.variables: list[Variable] = []
        self.objectives: list[Objective] = []
        self.quantity_limits: list[QuantityLimit] = []
        self.property_limits: list[PropertyLimit] = []
        self.properties: dict[str, dict[str, float]] = {}
        self.formulation_total: float | None = None
        self.next_formulation_no = 1
        self.recipes: list[dict] = []
        self.results: list[dict] = []
        self.scores: list[float] = []
        self.formulation_ids: list[int] = []
        self.open_rows: list[dict] = []    # [{"formulation", "recipe"}]
        self.retired: list[dict] = []      # dropped rows, on their generated basis

    # ------------------------------------------------------------ variables

    def _var(self, name):
        return next((v for v in self.variables if v.name == name), None)

    def _check_new_variable(self, name, min_val, max_val, kind):
        """_check_new_variable 1999: same name and kind is an edit."""
        name = str(name).strip()
        if not name:
            raise ValueError("A variable needs a name")
        if float(min_val) > float(max_val):
            raise ValueError(f"{name}: min {_num(min_val)} is above max "
                             f"{_num(max_val)}")
        for v in self.variables:
            if v.name.lower() == name.lower() and v.kind != kind:
                other = "an ingredient" if v.kind == "ingredient" else "a process setting"
                raise ValueError(f"{v.name} is already {other}")
        for v in self.variables:
            if v.name != name and v.name.lower() == name.lower():
                raise ValueError(f"{name}: already added as '{v.name}'")
        for obj in self.objectives:
            if obj.name.lower() == name.lower():
                raise ValueError(f"{name} is already a measurement")
        return name

    def _limit_refusals(self):
        refusals = {}
        for pl in self.property_limits:
            message = _property_limit_refusal(self.variables, self.properties, pl)
            if message:
                refusals[("property", pl.metric)] = message
        for ql in self.quantity_limits:
            message = _quantity_limit_refusal(self.variables, ql)
            if message:
                key = (("total",) if ql.source == _TOTAL
                       else ("quantity", tuple(ql.ingredients)))
                refusals[key] = message
        return refusals

    def _check_fixed_feasible(self, name, low, high, kind):
        """_check_fixed_feasible 10060: fixing a variable may not break a
        limit that held before."""
        if float(low) != float(high):
            return
        before = self._limit_refusals()
        var = self._var(name)
        if var is None:
            self.variables.append(Variable(name, float(low), float(high), kind))
            try:
                after = self._limit_refusals()
            finally:
                self.variables.pop()
        else:
            was = (var.low, var.high)
            var.low, var.high = float(low), float(high)
            try:
                after = self._limit_refusals()
            finally:
                var.low, var.high = was
        newly = [message for key, message in after.items() if key not in before]
        if newly:
            raise ValueError(f"{newly[0]} (with {name} fixed at {_num(low)})")

    def add_ingredient(self, name, min_val, max_val, keep_lowest=False,
                       baseline=None) -> None:
        """Add an ingredient or change its range. Mid-run, a new ingredient's min resets to 0 (past recipes lacked it) unless `keep_lowest`, or unless `baseline` gives the amount every recorded formulation already used (within min..max; 0 is none). Drops the open batch."""
        name = self._check_new_variable(name, min_val, max_val, "ingredient")
        low, high = float(min_val), float(max_val)
        # A baseline of 0 is none: the past formulations had none of it.
        cleared = baseline is not None and float(baseline) == 0.0
        baseline = None if baseline is None or cleared else float(baseline)
        var = self._var(name)
        if var is not None:
            stored = float(var.absent) or None
            if baseline is not None and stored is None:
                raise ValueError(
                    f"{name}: no baseline to set; it was in every recorded "
                    "formulation at its own amounts")
            moved = baseline is not None and baseline != stored
            if moved and not low <= baseline <= high:
                raise ValueError(f"{name}: baseline {_num(baseline)} is "
                                 f"outside {_num(low)} to {_num(high)}")
            absent = (baseline if moved else 0.0 if cleared
                      else float(var.absent))
            self._check_total_stays_reachable(
                [dataclasses.replace(v, low=low, high=high, absent=absent)
                 if v is var else v for v in self.variables], name, low, high)
            self._check_fixed_feasible(name, low, high, "ingredient")
            var.low, var.high, var.absent = low, high, absent
        else:
            if baseline is not None:
                if not self.recipes:
                    raise ValueError(
                        f"{name}: baseline {_num(baseline)} is the amount "
                        "recorded formulations used, and 0 formulations are "
                        "recorded")
                if not low <= baseline <= high:
                    raise ValueError(f"{name}: baseline {_num(baseline)} is "
                                     f"outside {_num(low)} to {_num(high)}")
            elif self.recipes and not keep_lowest:
                low = 0.0                  # absent from past recipes is 0
            new = Variable(name, low, high, "ingredient",
                           absent=0.0 if baseline is None else baseline)
            self._check_total_stays_reachable(self.variables + [new],
                                              name, low, high)
            self._check_fixed_feasible(name, low, high, "ingredient")
            self.variables.append(new)
        self._sync_formulation_total()
        self.discard_pending()

    def add_process_parameter(self, name, min_val, max_val, baseline=None,
                              step=None) -> None:
        """Add a process setting; once recipes exist it needs a `baseline` (the value past runs used). Results are snapped to `step`. Drops the open batch."""
        name = self._check_new_variable(name, min_val, max_val, "setting")
        low, high = float(min_val), float(max_val)
        step = None if step is None else float(step)
        var = self._var(name)
        if var is not None:
            stored = var.absent
            carried = (float(baseline) if baseline is not None
                       and stored is not None else stored)
            # Fixing away from the baseline is allowed: the baseline is a
            # fact about past runs, the range a decision about the next.
            fixing = low == high and (baseline is None or stored is None
                                      or float(baseline) == float(stored))
            if carried is not None and not fixing and not low <= carried <= high:
                raise ValueError(f"{name}: baseline {_num(carried)} is outside "
                                 f"{_num(low)} to {_num(high)}")
            self._check_fixed_feasible(name, low, high, "setting")
            var.low, var.high, var.step = low, high, step
            if carried is not None:
                var.absent = float(carried)
            self.discard_pending()
            return
        self._check_fixed_feasible(name, low, high, "setting")
        absent = None
        if self.recipes:
            if baseline is None:
                raise ValueError(f"{name}: give a baseline, the value the "
                                 "recorded formulations were made at")
            absent = float(baseline)
            if not low <= absent <= high:
                raise ValueError(f"{name}: baseline {_num(absent)} is outside "
                                 f"{_num(low)} to {_num(high)}")
        self.variables.append(Variable(name, low, high, "setting", step=step,
                                       absent=absent))
        self.discard_pending()

    def set_range(self, name, min_val, max_val) -> None:
        """Change the range of an ingredient or setting. Drops the open batch."""
        var = self._var(name)
        if var is None:
            raise ValueError(f"No variable named '{name}'")
        if var.kind == "ingredient":
            self.add_ingredient(name, min_val, max_val)
        else:
            self.add_process_parameter(name, min_val, max_val, step=var.step)

    def fix(self, name, value) -> None:
        """Hold a variable at one value: it leaves the search, its history stays."""
        self.set_range(name, value, value)

    def active_variables(self) -> list[str]:
        """Names of the variables still searched (not fixed)."""
        return active_names(self.variables)

    # ----------------------------------------------------------- objectives

    def _recompute_scores(self):
        self.scores = [score(self.objectives, r) for r in self.results]

    def add_objective(self, name, weight, goal="max", target=None,
                      min_val=None, max_val=None) -> bool:
        """Add or replace a measurement. Weights are relative; the scale `min_val`..`max_val` defaults to 0..10. Not allowed once results exist, except to replace one."""
        name = str(name).strip()
        if not name:
            raise ValueError("A measurement needs a name")
        if any(name.lower() == v.name.lower() for v in self.variables):
            raise ValueError(f"{name} is already a variable")
        for obj in self.objectives:
            if obj.name != name and obj.name.lower() == name.lower():
                raise ValueError(f"{name}: already added as '{obj.name}'")
        weight = float(weight)
        if weight <= 0:
            raise ValueError(f"{name}: weight {_num(weight)} must be above 0")
        min_val = float(min_val) if min_val is not None else 0.0
        max_val = float(max_val) if max_val is not None else 10.0
        if min_val >= max_val:
            raise ValueError(f"{name}: scale min {_num(min_val)} must be below "
                             f"max {_num(max_val)}")
        if goal not in ("max", "min", "target"):
            raise ValueError(f"{name}: goal must be 'max', 'min' or 'target', "
                             f"not {goal!r}")
        if goal == "target":
            if target is None:
                raise ValueError(f"{name}: a target goal needs a target")
            target = float(target)
            if not min_val <= target <= max_val:
                raise ValueError(f"{name}: target {_num(target)} is outside the "
                                 f"scale {_num(min_val)} to {_num(max_val)}")
        else:
            target = None
        replaced = any(obj.name == name for obj in self.objectives)
        if not replaced and self.results:
            raise ValueError(f"Cannot add measurement '{name}' after results "
                             "are recorded")
        self.objectives = [obj for obj in self.objectives if obj.name != name]
        self.objectives.append(Objective(name, weight, goal, target,
                                         min_val, max_val))
        self._recompute_scores()
        return replaced

    def remove_objective(self, name) -> None:
        """Drop a measurement; the remaining weights are rescaled to sum to 100."""
        if not any(obj.name == name for obj in self.objectives):
            raise ValueError(f"No measurement named '{name}'")
        self.objectives = [obj for obj in self.objectives if obj.name != name]
        # _shares_to_100 3200: what is left shares the whole 100.
        total = sum(float(obj.weight) for obj in self.objectives)
        if self.objectives and total > 0 and abs(total - 100.0) > 1e-9:
            for obj in self.objectives:
                obj.weight = float(obj.weight) * 100.0 / total
        self._recompute_scores()

    # --------------------------------------------------------------- limits

    def _ingredient_names(self):
        return [v.name for v in self.variables if v.kind == "ingredient"]

    def _fixed_ingredients(self, names=None):
        return [v.name for v in self.variables
                if v.kind == "ingredient" and v.fixed
                and (names is None or v.name in names)]

    def _add_quantity_limit(self, names, min_val, max_val, source):
        label = " + ".join(names)
        if (min_val is not None and max_val is not None
                and float(min_val) >= float(max_val)):
            raise ValueError(f"{label}: min {_num(min_val)} must be below max "
                             f"{_num(max_val)}")
        kept = list(self.quantity_limits)
        self.quantity_limits = [ql for ql in self.quantity_limits
                                if set(ql.ingredients) != set(names)
                                or ql.source != source]
        entry = QuantityLimit(list(names),
                              None if min_val is None else float(min_val),
                              None if max_val is None else float(max_val),
                              source)
        self.quantity_limits.append(entry)
        if source is None:
            refusal = _quantity_limit_refusal(self.variables, entry)
            fixed = self._fixed_ingredients(names)
            if refusal and fixed:
                self.quantity_limits = kept
                raise ValueError(f"{refusal} ({', '.join(fixed)} fixed)")

    def add_quantity_constraint(self, ingredients, min_val=None,
                                max_val=None) -> None:
        """Limit the summed amount of some ingredients. Does not drop the open batch."""
        names = list(ingredients)
        if not names:
            raise ValueError("A quantity limit needs at least one ingredient")
        known = set(self._ingredient_names())
        for name in names:
            if name not in known:
                raise ValueError(f"No ingredient named '{name}'")
        self._add_quantity_limit(names, min_val, max_val, None)

    def add_total_mass_constraint(self, min_val=None, max_val=None) -> None:
        """Limit the summed amount of all ingredients."""
        self.add_quantity_constraint(self._ingredient_names(), min_val, max_val)

    def _total_index(self):
        return next((i for i, ql in enumerate(self.quantity_limits)
                     if ql.source == _TOTAL), None)

    def _keep_limit_position(self, index):
        if index is None or index >= len(self.quantity_limits) - 1:
            return
        self.quantity_limits.insert(index, self.quantity_limits.pop())

    def set_formulation_total(self, total) -> None:
        """Fix the total of all ingredients. It is enforced as a +-0.5 % band, and suggestions are snapped to the exact total when that still meets every limit. Refused if the ranges or limits cannot reach it."""
        value = float(total)
        lowest, highest = total_reach(self.variables)
        if value > highest or value < lowest:
            raise ValueError(f"Formulation total {_num(value)} is out of reach: "
                             f"the ingredients add up to {_num(lowest)} to "
                             f"{_num(highest)}")
        limited = limited_total_reach(self.variables, self.quantity_limits,
                                      self.property_limits, self.properties)
        if limited is not None and (value > limited[1] + 1e-9
                                    or value < limited[0] - 1e-9):
            raise ValueError(f"Formulation total {_num(value)} is out of reach: "
                             f"the limits allow {_num(limited[0])} to "
                             f"{_num(limited[1])}")
        if value <= 0:
            raise ValueError(f"Formulation total must be above 0, not {_num(value)}")
        index = self._total_index()
        if value == self.formulation_total and index is not None:
            return
        if index is not None:
            self.quantity_limits.pop(index)
        self._add_quantity_limit(self._ingredient_names(), *total_band(value),
                                 _TOTAL)
        self._keep_limit_position(index)
        self.formulation_total = value
        self.discard_pending()

    def clear_formulation_total(self) -> None:
        """Remove the formulation total."""
        index = self._total_index()
        if index is None and self.formulation_total is None:
            return
        if index is not None:
            self.quantity_limits.pop(index)
        self.formulation_total = None
        self.discard_pending()

    def _check_total_stays_reachable(self, variables, name, low, high):
        """Refuse, before anything changes, a range the formulation total
        could no longer be reached with (the app would drop the total)."""
        if self.formulation_total is None:
            return
        lowest, highest = total_reach(variables)
        if not lowest <= self.formulation_total <= highest:
            raise ValueError(
                f"{name}: {_num(low)} to {_num(high)} makes the ingredients add "
                f"up to {_num(lowest)} to {_num(highest)}, which cannot reach "
                f"the formulation total {_num(self.formulation_total)}; call "
                "clear_formulation_total() first")

    def _sync_formulation_total(self):
        """_sync_formulation_total 7294: the total's limit follows the
        ingredient list (the range check has already passed)."""
        index = self._total_index()
        if index is not None:
            self.quantity_limits.pop(index)
        if self.formulation_total is None:
            return
        names = self._ingredient_names()
        self._add_quantity_limit(names, *total_band(self.formulation_total), _TOTAL)
        self._keep_limit_position(index)

    def set_property_values(self, metric, values: dict[str, float]) -> None:
        """Set an ingredient property, per 100 g of that ingredient (for example protein in g per 100 g). `None` clears a value."""
        if not str(metric).strip():
            raise ValueError("A property needs a name")
        known = set(self._ingredient_names())
        for name, value in values.items():
            if name not in known:
                raise ValueError(f"No ingredient named '{name}'")
            if value is not None and (not isinstance(value, numbers.Real)
                                      or isinstance(value, bool)
                                      or not math.isfinite(value)):
                raise ValueError(f"{name}: {metric} value {value!r} is not a "
                                 "finite number")
        wanted = str(metric).strip().lower()
        for name, value in values.items():
            props = self.properties.setdefault(name, {})
            for key in [k for k in props if str(k).strip().lower() == wanted]:
                props.pop(key)
            if value is not None:
                props[metric] = float(value)

    def add_constraint(self, metric, min_val=None, max_val=None) -> None:
        """Limit a property of the whole formulation: its mass-weighted average over the ingredients, using the per-100 g values. Replaces a limit on the same metric; a refused replacement keeps the old one."""
        if (min_val is not None and max_val is not None
                and float(min_val) >= float(max_val)):
            raise ValueError(f"{metric}: min {_num(min_val)} must be below max "
                             f"{_num(max_val)}")
        wanted = str(metric).strip().lower()
        kept = list(self.property_limits)
        self.property_limits = [pl for pl in self.property_limits
                                if str(pl.metric).strip().lower() != wanted]
        entry = PropertyLimit(metric,
                              None if min_val is None else float(min_val),
                              None if max_val is None else float(max_val))
        self.property_limits.append(entry)
        refusal = _property_limit_refusal(self.variables, self.properties, entry)
        fixed = self._fixed_ingredients()
        if refusal and fixed:
            self.property_limits = kept
            raise ValueError(f"{refusal} ({', '.join(fixed)} fixed)")

    # ---------------------------------------------------------------- rounds

    def _as_generated(self, recipe):
        recipe = dict(recipe)
        if (self.formulation_total is not None
                and off_total(self.variables, recipe, self.formulation_total)):
            return at_total(self.variables, recipe, self.formulation_total)
        return recipe

    def _model_row(self, index):
        """model_X 8514: a row made off the total is learned at the total."""
        row = encode(self.variables, self.recipes[index])
        if self.formulation_total is None:
            return row
        recipe = decode(self.variables, row)
        if off_total(self.variables, recipe, self.formulation_total):
            return encode(self.variables,
                          at_total(self.variables, recipe, self.formulation_total))
        return row

    def _generated_row(self, recipe):
        row = np.asarray(encode(self.variables, self._as_generated(recipe)),
                         dtype=float)
        return row if np.all(np.isfinite(row)) else None

    def _issued_matrix(self):
        """_issued_matrix 8339: recorded, open and retired formulations."""
        recipes = [decode(self.variables, encode(self.variables, r))
                   for r in self.recipes]
        recipes += [row["recipe"] for row in self.open_rows]
        recipes += self.retired
        rows = [r for r in map(self._generated_row, recipes) if r is not None]
        if not rows:
            return None
        width = rows[-1].shape[0]
        rows = [r for r in rows if r.shape[0] == width]
        return np.vstack(rows) if rows else None

    def _as_issued(self, recipe):
        return np.asarray(encode(self.variables,
                                 snap_settings(self.variables, recipe)),
                          dtype=float)

    def _feasible(self, recipe):
        return check(self.variables, self.quantity_limits,
                     self.property_limits, self.properties, recipe)

    def _snap_to_total(self, recipe):
        return snapped_if_it_still_fits(
            self.variables, self.quantity_limits, self.property_limits,
            self.properties, recipe, self.formulation_total)

    def _ask_optimize(self, n, spans, bounds, constraints):
        train_X = torch.tensor([self._model_row(i) for i in range(len(self.recipes))],
                               dtype=torch.double)
        train_Y = torch.tensor(self.scores, dtype=torch.double).unsqueeze(-1)
        out = suggest(train_X, train_Y, bounds, n, int(self.next_formulation_no),
                      robust=self.robust, bo_config=self.bo_config,
                      inequality_constraints=constraints,
                      fixed_features=fixed_features(self.variables, spans))
        return [decode(self.variables, out[i].tolist()) for i in range(n)]

    def _linear_constraints(self, spans):
        return linear_constraints(self.variables, self.quantity_limits,
                                  self.property_limits, self.properties, spans)

    def _ask_cold_start(self, n, spans, bounds):
        """_ask_cold_start 8154: the next points of the one Sobol sequence."""
        issued = self._issued_matrix()
        fixed = fixed_features(self.variables, spans)
        already = max(0, int(self.next_formulation_no) - 1)
        results, repeats, chosen = [], [], []
        outside = [v for _, v in widened(self.variables, spans)]
        for size in COLD_START_POOLS:
            pool = sobol_pool(size, len(self.variables), self.sobol_seed,
                              already, fixed)
            candidates = [decode(self.variables,
                                 unnormalize(pool[i], bounds).numpy().flatten())
                          for i in range(size)]
            # A widened variable decodes anywhere in its frame; a point
            # outside its own range is not a formulation it may be given.
            if outside:
                candidates = [c for c in candidates if all(
                    float(v.low) - 1e-9 * (1 + abs(float(v.low))) <= float(c[v.name])
                    <= float(v.high) + 1e-9 * (1 + abs(float(v.high)))
                    for v in outside)]
            if self.formulation_total is not None:
                candidates = [c for c in (
                    project_onto_total(self.variables, rec, self.formulation_total)
                    for rec in candidates) if c is not None]
            results, repeats, chosen = select(candidates, n, self._feasible,
                                              self._as_issued, issued)
            if len(results) >= n:
                return results
        if self.recipes:
            try:
                picked = self._ask_optimize(n, spans, bounds,
                                            self._linear_constraints(spans))
            except Exception:
                picked = []
            if picked:
                if self.formulation_total is not None:
                    picked = [self._snap_to_total(rec) for rec in picked]
                return picked
        if len(results) + len(repeats) >= n:
            return results + fill_from_repeats(repeats, chosen, n - len(results),
                                               self._as_issued)
        # Limits too thin for any pool to land in often (A + B within a
        # hair): sample inside them instead. Only here, where the Round would
        # otherwise come back short or be refused.
        results = results + self._sample_feasible(n - len(results), spans,
                                                  bounds, chosen)
        if results:
            return results
        if self.formulation_total is not None:
            ingredients = [v for v in self.variables if v.kind == "ingredient"]
            if all(v.fixed for v in ingredients):
                fixed_sum = sum(float(v.low) for v in ingredients)
                raise ValueError(f"The fixed amounts add up to {_num(fixed_sum)}, "
                                 f"not the formulation total "
                                 f"{_num(self.formulation_total)}")
            raise ValueError(f"{NO_FORMULATION} at the formulation total "
                             f"{_num(self.formulation_total)}")
        raise ValueError(NO_FORMULATION)

    def _sample_feasible(self, n, spans, bounds, chosen):
        """Up to `n` new formulations from a hit-and-run chain inside the
        limits, seeded on the next formulation number. Each point is snapped
        onto the total only where that keeps every limit, checked, and
        passed over if it repeats an issued formulation, `chosen` or an
        earlier point. Any failure is an empty list: the refusal stands."""
        if n <= 0:
            return []
        try:
            points = polytope_points(
                n, len(self.variables), self._linear_constraints(spans),
                fixed_features(self.variables, spans),
                int(self.next_formulation_no))
            issued = self._issued_matrix()
            seen = list(chosen)
            out = []
            for i in range(points.shape[0]):
                rec = decode(self.variables,
                             unnormalize(points[i], bounds).numpy().flatten())
                if self.formulation_total is not None:
                    rec = self._snap_to_total(rec)
                if not self._feasible(rec):
                    continue
                vector = self._as_issued(rec)
                if _repeats(vector, issued) or _repeats(vector, seen):
                    continue
                out.append(rec)
                seen.append(vector)
                if len(out) >= n:
                    break
            return out
        except Exception:
            return []

    def ask(self, n: int = 1) -> list[dict[str, float]]:
        """Suggest `n` formulations and hold them as the open batch. A second `ask` discards the open rows still unrecorded. The first rounds (`n_init` recorded) are a Sobol design; later ones come from a fitted model. Does not change the global torch RNG."""
        if (isinstance(n, bool) or not isinstance(n, numbers.Real)
                or not math.isfinite(n) or n != int(n) or int(n) < 1):
            raise ValueError(f"Ask for a whole number of formulations, at "
                             f"least 1, not {n!r}")
        n = int(n)
        for pl in self.property_limits:
            if pl.min_val is None and pl.max_val is None:
                continue
            gaps = missing_property_values(self.variables, self.properties,
                                           pl.metric)
            if gaps:
                raise ValueError(f"{pl.metric}: no value for {', '.join(gaps)}")
        if not self.variables:
            raise ValueError("Add an ingredient or a process setting first")
        if not self.active_variables():
            raise ValueError("Every variable is fixed; widen a range to search")
        self.discard_pending()
        spans = search_bounds(self.variables, self.recipes,
                              [self._model_row(i) for i in range(len(self.recipes))])
        bounds = bounds_tensor(spans)
        try:
            rows = self._generate(n, spans, bounds)
        except ValueError as exc:
            # The app shows why on its Limits screen; a script only has this.
            reasons = list(self._limit_refusals().values())
            if str(exc).startswith(NO_FORMULATION) and reasons:
                raise ValueError(f"{exc}. {reasons[0]}") from exc
            raise
        for rec in rows:
            self.open_rows.append({"formulation": int(self.next_formulation_no),
                                   "recipe": dict(rec)})
            self.next_formulation_no += 1
        return [dict(rec) for rec in rows]

    def _generate(self, n, spans, bounds):
        if len(self.recipes) < self.n_init:
            rows = self._ask_cold_start(n, spans, bounds)
        else:
            constraints = self._linear_constraints(spans)
            try:
                rows = self._ask_optimize(n, spans, bounds, constraints)
            except (BotorchError, torch.linalg.LinAlgError) as exc:
                # An empty polytope (infeasible limits) fails inside BoTorch.
                raise ValueError(NO_FORMULATION) from exc
            if self.formulation_total is not None:
                rows = [self._snap_to_total(rec) for rec in rows]
        return [snap_settings(self.variables, rec) for rec in rows]

    def pending(self) -> list[dict]:
        """The open (suggested, not yet recorded) formulations."""
        return [{"formulation": row["formulation"], "recipe": dict(row["recipe"])}
                for row in self.open_rows]

    def discard_pending(self) -> None:
        """Drop the open batch. Unrecorded rows are remembered so the opening design never suggests them again."""
        recorded = set(self.formulation_ids)
        for row in self.open_rows:
            if row["formulation"] not in recorded:
                self.retired.append(self._as_generated(row["recipe"]))
        self.open_rows = []

    def tell(self, recipe: dict, results: dict) -> int:
        """Record a result. If the recipe equals an open row exactly (same amounts) that row is closed; otherwise it is recorded as your own formulation under a new number. Permanent in this version. Returns the formulation number."""
        if not self.objectives:
            raise ValueError("Add a measurement before recording results")
        names = {v.name for v in self.variables}
        amounts = {}
        for key, value in recipe.items():
            if key not in names:
                raise ValueError(f"'{key}' is not a variable")
            if (not isinstance(value, numbers.Real) or isinstance(value, bool)
                    or not math.isfinite(value)):
                raise ValueError(f"{key}: amount {value!r} is not a number")
            amounts[key] = float(value)
        measured = {obj.name for obj in self.objectives}
        for key in results:
            if key not in measured:
                raise ValueError(f"'{key}' is not a measurement")
        values = {}
        for obj in self.objectives:
            value = results.get(obj.name)
            if value is None:
                raise ValueError(f"Missing result for '{obj.name}'")
            if not isinstance(value, numbers.Real) or isinstance(value, bool):
                raise ValueError(f"Result for '{obj.name}' is not a number: "
                                 f"{value!r}")
            if math.isnan(value):
                raise ValueError(f"Missing result for '{obj.name}'")
            if not math.isfinite(value):
                raise ValueError(f"Result for '{obj.name}' is not finite: "
                                 f"{value!r}")
            values[obj.name] = float(value)
        for v in self.variables:
            if v.absent is None and v.name not in amounts:
                raise ValueError(f"Missing value for '{v.name}'")
        vector = encode(self.variables, amounts)
        number = None
        for k, row in enumerate(self.open_rows):
            if encode(self.variables, row["recipe"]) == vector:
                number = int(row["formulation"])
                del self.open_rows[k]
                break
        if number is None:
            number = int(self.next_formulation_no)
            self.next_formulation_no += 1
        self.recipes.append(amounts)
        self.results.append(values)
        self.scores.append(score(self.objectives, values))
        self.formulation_ids.append(number)
        return number

    def history(self) -> list[dict]:
        """Every recorded formulation: number, recipe, results and score."""
        return [{"formulation": no, "recipe": dict(r), "results": dict(res),
                 "score": s}
                for no, r, res, s in zip(self.formulation_ids, self.recipes,
                                         self.results, self.scores)]

    # ------------------------------------------------------------ save/load

    def to_dict(self) -> dict:
        """The full state as plain JSON-ready data."""
        return _state.to_dict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Optimizer":
        """Rebuild an Optimizer from `to_dict` output."""
        return _state.from_dict(cls, d)

    def save(self, path) -> None:
        """Write the state to a JSON file (replaced atomically)."""
        _state.save(self, path)

    @classmethod
    def load(cls, path) -> "Optimizer":
        """Read an Optimizer saved with `save`."""
        return _state.load(cls, path)
