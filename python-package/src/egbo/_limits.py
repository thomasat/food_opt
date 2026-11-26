"""Limits and the formulation total: a port of food_bo.py
_get_botorch_constraints / _form_over_columns / _property_coeff /
_check_constraints / limit_slack / _project_onto_total / _share_to_total /
_snapped_if_it_still_fits / off_total / at_model_total (no formula rows,
pre-mixes or batch size, so every ingredient's linear form is itself)."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from egbo._space import Variable, amount, ingredient_total

FORMULATION_TOTAL_TOLERANCE = 0.005
MODEL_TOTAL_MARGIN = 0.01


@dataclass
class QuantityLimit:
    ingredients: list[str]
    min_val: float | None = None
    max_val: float | None = None
    source: str | None = None        # "formulation_total" for the total's band


@dataclass
class PropertyLimit:
    metric: str
    min_val: float | None = None
    max_val: float | None = None


Properties = dict[str, dict[str, float]]   # ingredient -> {metric: value per 100 g}


def limit_slack(value) -> float:
    return 1e-6 * (1.0 + abs(float(value)))


def _amount(value) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _has_value(properties, name, metric) -> bool:
    props = properties.get(name, {}) or {}
    wanted = str(metric).strip().lower()
    return any(str(key).strip().lower() == wanted for key in props)


def _property_value(properties, name, metric) -> float:
    props = properties.get(name, {}) or {}
    if metric in props:
        return float(props[metric])
    lowered = str(metric).strip().lower()
    for key, value in props.items():
        if str(key).strip().lower() == lowered:
            return float(value)
    return 0.0


def missing_property_values(variables: list[Variable], properties: Properties,
                            metric: str) -> list[str]:
    names, seen = [], set()
    for v in variables:
        if v.kind != "ingredient":
            continue
        if v.name not in seen and not _has_value(properties, v.name, metric):
            seen.add(v.name)
            names.append(v.name)
    return names


def _property_coeff(variables, properties, metric, limit):
    limit = float(limit)
    names = {v.name for v in variables if v.kind == "ingredient"}
    return lambda name: ((_property_value(properties, name, metric) - limit)
                         if name in names else 0.0)


def _form_over_columns(variables, terms, frame=None):
    """(columns, coefficients, offset) in the [0, 1] frame; a fixed variable
    moves the offset rather than riding in the form.

    `frame` is the search span per column (search_bounds). A variable the
    frame widened (added mid-run with its min kept above 0) is decoded over
    the widened span, so its coefficients are written over it too: over its
    own range, "D >= 8" on D in 4..10 widened to 0..10 would mean D >= 6.67."""
    by_name = {v.name: (col, v) for col, v in enumerate(variables)}
    indices, coeffs, offset = [], [], 0.0
    for name, coeff in terms.items():
        if coeff == 0:
            continue
        if name not in by_name:
            continue
        col, var = by_name[name]
        v_min, v_max = float(var.low), float(var.high)
        if v_max != v_min:
            f_min, f_max = (v_min, v_max) if frame is None else frame[col]
            indices.append(col)
            coeffs.append(coeff * (f_max - f_min))
            offset += coeff * f_min
        else:
            offset += coeff * v_min
    return indices, coeffs, offset


def widened(variables, frame) -> list[tuple[int, Variable]]:
    """(column, variable) of each searched variable whose frame is wider
    than its own range."""
    return [(col, v) for col, v in enumerate(variables)
            if not v.fixed
            and tuple(map(float, frame[col])) != (float(v.low), float(v.high))]


def _form_reach(variables, terms):
    by_name = {v.name: v for v in variables}
    lo = hi = 0.0
    for name, coeff in terms.items():
        if coeff == 0 or name not in by_name:
            continue
        a, b = coeff * float(by_name[name].low), coeff * float(by_name[name].high)
        lo += min(a, b)
        hi += max(a, b)
    return lo, hi


def _property_terms(variables, coeff_of):
    terms = {}
    for v in variables:
        coeff = coeff_of(v.name)
        if coeff == 0:
            continue
        terms[v.name] = terms.get(v.name, 0.0) + coeff
    return terms


def _achievable_property(variables, properties, metric):
    values = [_property_value(properties, v.name, metric) for v in variables
              if v.kind == "ingredient"]
    if not values:
        return 0.0, 0.0
    low, high = min(values), max(values)
    if high - low <= 1e-12:
        return low, high

    def span(value):
        return _form_reach(variables, _property_terms(
            variables, _property_coeff(variables, properties, metric, value)))

    if span(high)[0] >= 0:
        lowest = high
    else:
        under, over = high, low
        for _ in range(60):
            mid = (under + over) / 2.0
            if span(mid)[0] < 0:
                under = mid
            else:
                over = mid
        lowest = under
    if span(low)[1] <= 0:
        highest = low
    else:
        over, under = low, high
        for _ in range(60):
            mid = (over + under) / 2.0
            if span(mid)[1] > 0:
                over = mid
            else:
                under = mid
        highest = over
    return lowest, highest


def _property_limit_refusal(variables, properties, pl):
    if missing_property_values(variables, properties, pl.metric):
        return None
    lo, hi = _achievable_property(variables, properties, pl.metric)
    if pl.min_val is not None and hi < pl.min_val:
        return (f"The limit on {pl.metric} (at least {pl.min_val:g} per "
                f"100 g) cannot be met: the amounts allowed only reach "
                f"{hi:.4g} per 100 g at most")
    if pl.max_val is not None and lo > pl.max_val:
        return (f"The limit on {pl.metric} (at most {pl.max_val:g} per "
                f"100 g) cannot be met: the amounts allowed never get below "
                f"{lo:.4g} per 100 g")
    return None


def _quantity_limit_refusal(variables, ql, pinned="fixed amounts"):
    names = set(ql.ingredients)
    lo, hi = _form_reach(variables, {n: 1.0 for n in names})
    broken = ((ql.min_val is not None and hi < ql.min_val)
              or (ql.max_val is not None and lo > ql.max_val))
    if ql.source == "formulation_total":
        return ("The fixed amounts break the formulation total"
                if broken else None)
    label = " + ".join(ql.ingredients)
    if ql.min_val is not None and hi < ql.min_val:
        return (f"The limit on {label} (at least {ql.min_val:g}) cannot be "
                f"met: the amounts allowed only reach {hi:.4g} at most")
    if ql.max_val is not None and lo > ql.max_val:
        return (f"The limit on {label} (at most {ql.max_val:g}) cannot be "
                f"met: the {pinned} add up to {lo:.4g}")
    return None


def linear_constraints(variables, quantity_limits, property_limits, properties,
                       frame=None):
    """BoTorch inequality constraints over the [0, 1] search frame. `frame`
    is search_bounds' span per column; None is each variable's own range."""
    out = []
    known = {v.name for v in variables}

    for pl in property_limits:
        for bound, sense in (("min_val", 1.0), ("max_val", -1.0)):
            limit = getattr(pl, bound)
            if limit is None:
                continue
            terms = _property_terms(
                variables, _property_coeff(variables, properties, pl.metric,
                                           limit))
            indices, coeffs, offset = _form_over_columns(variables, terms, frame)
            if not indices:
                message = _property_limit_refusal(variables, properties, pl)
                if message:
                    raise ValueError(message)
                continue
            out.append((torch.tensor(indices, dtype=torch.long),
                        torch.tensor([sense * c for c in coeffs],
                                     dtype=torch.double),
                        -sense * offset))

    for ql in quantity_limits:
        terms = {}
        for name in ql.ingredients:
            if name in known:
                terms[name] = terms.get(name, 0.0) + 1.0
        indices, coeffs, offset = _form_over_columns(variables, terms, frame)
        if not indices:
            message = _quantity_limit_refusal(variables, ql)
            if message:
                raise ValueError(message)
            continue
        t_idx = torch.tensor(indices, dtype=torch.long)
        t_coeffs = torch.tensor(coeffs, dtype=torch.double)
        if ql.min_val is not None:
            out.append((t_idx, t_coeffs, ql.min_val - offset))
        if ql.max_val is not None:
            out.append((t_idx, -t_coeffs, -(ql.max_val - offset)))

    # A widened variable's own min and max: the box the acquisition searches
    # is the whole frame, so without these a suggestion could fall to 0.
    for col, var in ([] if frame is None else widened(variables, frame)):
        low, high = float(var.low), float(var.high)
        f_low, f_high = frame[col]
        span = f_high - f_low
        idx = torch.tensor([col], dtype=torch.long)
        if low > f_low:
            out.append((idx, torch.tensor([1.0], dtype=torch.double),
                        (low - f_low) / span))
        if high < f_high:
            out.append((idx, torch.tensor([-1.0], dtype=torch.double),
                        -(high - f_low) / span))
    return out


def check(variables, quantity_limits, property_limits, properties, recipe) -> bool:
    # The tolerance rides on the amounts as given, as the app's does.
    scale = 1.0 + sum(abs(float(a or 0.0)) for a in recipe.values())
    # No ingredients is no amount per 100 g, so it meets no property limit;
    # the linear form would read it as 0 >= 0 and let it through.
    empty = ingredient_total(variables, recipe) <= 1e-9 * scale
    # A recipe made before an ingredient was added holds its baseline.
    recipe = {**{v.name: v.absent for v in variables
                 if v.kind == "ingredient" and v.name not in recipe
                 and v.absent}, **recipe}
    for pl in property_limits:
        if pl.min_val is None and pl.max_val is None:
            continue
        if empty or missing_property_values(variables, properties, pl.metric):
            return False
        coeff_of = None
        for bound, sign in ((pl.min_val, -1), (pl.max_val, 1)):
            if bound is None:
                continue
            coeff_of = _property_coeff(variables, properties, pl.metric, bound)
            residual = sum(coeff_of(name) * float(amount or 0.0)
                           for name, amount in recipe.items())
            slack = 1e-9 * scale * (1.0 + abs(float(bound)))
            if sign < 0 and residual < -slack:
                return False
            if sign > 0 and residual > slack:
                return False
    for ql in quantity_limits:
        total_val = sum(recipe.get(name, 0.0) for name in ql.ingredients)
        slack = limit_slack(total_val)
        if ql.min_val is not None and total_val < float(ql.min_val) - slack:
            return False
        if ql.max_val is not None and total_val > float(ql.max_val) + slack:
            return False
    return True


def total_band(total: float) -> tuple[float, float]:
    value = float(total)
    return (value * (1.0 - FORMULATION_TOTAL_TOLERANCE),
            value * (1.0 + FORMULATION_TOTAL_TOLERANCE))


def total_reach(variables) -> tuple[float, float]:
    ingredients = [v for v in variables if v.kind == "ingredient"]
    return (sum(float(v.low) for v in ingredients),
            sum(float(v.high) for v in ingredients))


def _share_to_total(free, room_of, need):
    free = list(free)
    for _ in range(40):
        residual = need - sum(free)
        if abs(residual) <= 1e-9:
            break
        headroom = ([c - f for f, c in zip(free, room_of)] if residual > 0
                    else list(free))
        share = sum(headroom)
        if share <= 1e-12:
            break
        for k, head in enumerate(headroom):
            free[k] = min(room_of[k],
                          max(0.0, free[k] + residual * head / share))
    return free


def project_onto_total(variables, recipe, total) -> dict | None:
    ingredients = [v for v in variables if v.kind == "ingredient"]
    constant = sum(float(v.low) for v in ingredients if v.fixed)
    names, lows, caps, start = [], [], [], []
    for var in ingredients:
        if var.fixed:
            continue
        low, high = float(var.low), float(var.high)
        value = _amount(recipe.get(var.name, var.absent))
        names.append(var.name)
        lows.append(low)
        caps.append(max(0.0, high - low))
        start.append(min(max(value - low, 0.0), max(0.0, high - low)))

    base = constant + sum(lows)
    need = float(total) - base
    snapped = dict(recipe)
    for i, name in enumerate(names):
        snapped[name] = lows[i] + start[i]

    if not names:
        if abs(need) > 1e-6 * (1.0 + abs(float(total))):
            return None
        for var in ingredients:
            if var.fixed:
                snapped[var.name] = float(var.low)
        return snapped

    room = sum(caps)
    if need < -1e-9 or need > room + 1e-9:
        return None
    up_need = min(max(need, 0.0), room)
    free = _share_to_total(start, caps, up_need)
    for k, name in enumerate(names):
        snapped[name] = lows[k] + free[k]
    return snapped


def snapped_if_it_still_fits(variables, quantity_limits, property_limits,
                             properties, recipe, total) -> dict:
    snapped = project_onto_total(variables, recipe, total)
    if snapped is None or not check(variables, quantity_limits,
                                    property_limits, properties, snapped):
        return recipe
    return snapped


def off_total(variables, recipe, total) -> bool:
    own = ingredient_total(variables, recipe)
    total = float(total)
    return own > 0 and abs(own - total) > MODEL_TOTAL_MARGIN * total


def at_total(variables, recipe, total) -> dict:
    own = ingredient_total(variables, recipe)
    if own <= 0:
        return dict(recipe)
    factor = float(total) / own
    out = dict(recipe)
    for var in variables:
        if var.kind == "ingredient":
            out[var.name] = amount(var, recipe) * factor
        else:
            out[var.name] = float(recipe.get(var.name, 0.0))
    return out


def _limited_total_reach(variables, quantity_limits, property_limits,
                         properties):
    """(lowest, highest) the ingredients can add up to once the limits hold
    too (one small linear programme each way), or None when no limit
    narrows anything or no formulation meets them all. The formulation
    total's own limit is left out: the total is what is being asked about."""
    limits = [ql for ql in quantity_limits
              if ql.source != "formulation_total"
              and (ql.min_val is not None or ql.max_val is not None)]
    props = [pl for pl in property_limits
             if (pl.min_val is not None or pl.max_val is not None)
             and not missing_property_values(variables, properties, pl.metric)]
    if not limits and not props:
        return None
    from scipy.optimize import linprog
    index = {v.name: i for i, v in enumerate(variables)}

    def row(terms):
        coeffs = [0.0] * len(variables)
        for name, coeff in terms.items():
            if name in index:
                coeffs[index[name]] += coeff
        return coeffs

    a_ub, b_ub = [], []

    def at_least(terms, value):
        a_ub.append([-c for c in row(terms)])
        b_ub.append(-float(value))

    def at_most(terms, value):
        a_ub.append(row(terms))
        b_ub.append(float(value))

    for ql in limits:
        members = set(ql.ingredients)
        terms = _property_terms(variables,
                                lambda name: 1.0 if name in members else 0.0)
        if ql.min_val is not None:
            at_least(terms, ql.min_val)
        if ql.max_val is not None:
            at_most(terms, ql.max_val)
    for pl in props:
        if pl.min_val is not None:
            at_least(_property_terms(variables, _property_coeff(
                variables, properties, pl.metric, pl.min_val)), 0.0)
        if pl.max_val is not None:
            at_most(_property_terms(variables, _property_coeff(
                variables, properties, pl.metric, pl.max_val)), 0.0)
    names = {v.name for v in variables if v.kind == "ingredient"}
    objective = row(_property_terms(
        variables, lambda name: 1.0 if name in names else 0.0))
    bounds = [(float(v.low), float(v.high)) for v in variables]
    ends = []
    for sign in (1.0, -1.0):
        try:
            answer = linprog([sign * c for c in objective],
                             A_ub=a_ub or None, b_ub=b_ub or None,
                             bounds=bounds, method="highs")
        except ValueError:
            return None
        if answer.status != 0:
            return None
        ends.append(round(sign * float(answer.fun), 9))
    return ends[0], ends[1]


def limited_total_reach(variables, quantity_limits, property_limits,
                        properties) -> tuple[float, float] | None:
    """total_reach narrowed by the limits, or None when they narrow nothing
    (a port of food_bo.py limited_total_reach, no formula rows)."""
    limited = _limited_total_reach(variables, quantity_limits,
                                   property_limits, properties)
    if limited is None:
        return None
    plain = total_reach(variables)
    low, high = max(plain[0], limited[0]), min(plain[1], limited[1])
    if abs(low - plain[0]) < 1e-9 and abs(high - plain[1]) < 1e-9:
        return None
    return low, high
