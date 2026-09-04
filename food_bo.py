import json
import os
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
from torch.quasirandom import SobolEngine

from botorch.acquisition import (
    qLogExpectedImprovement,
    qLogNoisyExpectedImprovement,
)
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Warp
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.kernels import (
    LinearKernel,
    MaternKernel,
    PolynomialKernel,
    RBFKernel,
    ScaleKernel,
)
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior

from storage import LocalStorage, StorageError


# --------------------------------------------------------------------------- #
#  Expert-selectable BO hyperparameters (optional "arm 3").
#  bo_config == None  =>  library defaults, i.e. byte-identical to the standard
#  non-adaptive arm. A validated dict swaps in the expert's chosen kernel /
#  lengthscale prior / noise handling / acquisition. Chosen ONCE at project
#  start (not per iteration): the GP's lengthscale/noise VALUES still refit from
#  data each iteration via MLE; only the structural config is fixed a priori.
# --------------------------------------------------------------------------- #
_KERNELS = {"matern52", "matern32", "rbf", "linear", "poly2"}
_LENGTHSCALE = {"default", "long", "short"}
_NOISE = {"default", "low", "fixed_tiny"}
_ACQ = {"qlognei", "qlogei", "qucb"}
_LS_PRIORS = {"default": (3.0, 6.0), "long": (3.0, 1.0), "short": (3.0, 12.0)}
DEFAULT_BO_CONFIG = {
    "kernel": "matern52", "lengthscale_prior": "default",
    "noise": "default", "acquisition": "qlognei",
}


def validate_bo_config(spec):
    """Coerce a raw config dict to valid values. Returns None for an empty/None
    spec (None => library defaults, identical to the standard arm)."""
    if not spec:
        return None

    def pick(key, allowed, default):
        v = str(spec.get(key, default)).strip().lower()
        return v if v in allowed else default

    return {
        "kernel": pick("kernel", _KERNELS, "matern52"),
        "lengthscale_prior": pick("lengthscale_prior", _LENGTHSCALE, "default"),
        "noise": pick("noise", _NOISE, "default"),
        "acquisition": pick("acquisition", _ACQ, "qlognei"),
    }


def _build_covar(cfg, dim):
    kernel = cfg.get("kernel", "matern52")
    if kernel in ("matern52", "matern32"):
        nu = 2.5 if kernel == "matern52" else 1.5
        conc, rate = _LS_PRIORS.get(cfg.get("lengthscale_prior", "default"), _LS_PRIORS["default"])
        base = MaternKernel(nu=nu, ard_num_dims=dim, lengthscale_prior=GammaPrior(conc, rate))
    elif kernel == "rbf":
        conc, rate = _LS_PRIORS.get(cfg.get("lengthscale_prior", "default"), _LS_PRIORS["default"])
        base = RBFKernel(ard_num_dims=dim, lengthscale_prior=GammaPrior(conc, rate))
    elif kernel == "linear":
        base = LinearKernel()
    elif kernel == "poly2":
        base = PolynomialKernel(power=2)
    else:
        base = MaternKernel(nu=2.5, ard_num_dims=dim)
    return ScaleKernel(base)


class FoodOptimizer:
    CLASS_VERSION = 6  # bump when adding methods/attrs to force session refresh

    def __init__(self, project_name="experiment", robust=False, storage=None):
        """Initialize or load a food optimization project.

        Args:
            project_name: Name used for the .pkl save file.
            robust: If True, uses Input Warping for cliffs/traps.
                    If False (default), uses standard GP for smooth problems.
            storage: Persistence backend. Defaults to LocalStorage() (the
                     desktop app's on-disk .pkl files); a cloud backend can
                     be injected instead.
        """
        self.project_name = project_name
        self.robust = robust
        self.filename = f"{project_name}.pkl"   # informational; kept for compat
        self.storage = storage if storage is not None else LocalStorage()

        self.variables = []
        self.objectives = []
        self.ingredient_properties = {}
        self.constraints = []
        self.quantity_constraints = []
        self.screening_model = None
        self.bo_config = None  # None => library defaults; dict => expert-chosen (arm 3)

        self.X_history = []
        self.Y_history = []
        self.recipe_history = []
        self.results_history = []
        self.timestamps_history = []  # UTC ISO per tell(); parallel to results_history
        self.pending_batch = None  # suggested-but-unrated recipes (survives sessions)
        self.load_error = None  # set to a plain-language string if load() fails
        self.save_error = None  # set when a cloud save fails; cleared on success

        try:
            exists = self.storage.exists(project_name)
        except StorageError as e:
            # Backend down: must NOT save (would overwrite a real project
            # with a blank one). Surface as a load error instead.
            self.load_error = str(e)
        else:
            if exists:
                self.load()
            elif self.storage.persist_empty_on_init:
                self.save()

    # ------------------------------------------------------------------ #
    #  Setup: Ingredients & Process Parameters
    # ------------------------------------------------------------------ #

    def add_ingredient(self, name, min_val, max_val):
        """Add a single ingredient. Safe to call mid-run (adaptive EGBO): the
        ingredient is treated as absent (=0) in every prior recipe, and the
        encoded history is rebuilt so the GP stays dimensionally consistent."""
        for var in self.variables:
            if var['name'] == name:
                return
        min_val, max_val = float(min_val), float(max_val)
        if self.X_history:
            if len(self.recipe_history) != len(self.X_history):
                raise ValueError(
                    "Cannot add a variable mid-run: some experiments were recorded "
                    "without stored recipes. Start a fresh project or re-import history."
                )
            min_val = 0.0  # absent-in-past encodes as 0; it must be within bounds
        self.variables.append({
            'name': name,
            'type': 'continuous',
            'bounds': (min_val, max_val),
            'category': 'ingredient',
            'active': True,
        })
        if self.X_history:
            self._reencode_history()
        self.pending_batch = None
        self.save()

    def load_ingredients_from_csv(self, df):
        """Bulk-load ingredients from a DataFrame (used by the app).

        Raises ValueError if experiments already exist, since reloading
        would invalidate encoded history vectors.
        """
        if self.X_history:
            raise ValueError(
                "Cannot reload ingredients after experiments have been recorded. "
                "Use Hard Reset to start a new project, or restore from a backup."
            )

        # Accept any capitalization/whitespace for the required headers, and
        # fail with a plain-language error (the app shows ValueError text to
        # the user) instead of a KeyError when one is missing.
        canonical = {'name': 'Name', 'min': 'Min', 'max': 'Max', 'type': 'Type'}
        df = df.rename(columns={
            c: canonical[c.strip().lower()]
            for c in df.columns if c.strip().lower() in canonical
        })
        missing = [c for c in ('Name', 'Min', 'Max') if c not in df.columns]
        if missing:
            raise ValueError(
                f"The ingredients file is missing required column(s): "
                f"{', '.join(missing)}. Expected columns: Name, Min, Max "
                f"(plus optional property columns like Cost or Protein)."
            )

        process_vars = [v for v in self.variables if v.get('category') == 'process']
        self.variables = []
        self.ingredient_properties = {}

        standard_cols = {'Name', 'Min', 'Max', 'Type'}
        prop_cols = [c for c in df.columns if c not in standard_cols]

        for _, row in df.iterrows():
            name = row['Name']
            try:
                min_val, max_val = float(row['Min']), float(row['Max'])
            except (ValueError, TypeError):
                raise ValueError(
                    f"Ingredient '{row['Name']}': Min and Max must be numbers. "
                    f"Please check that column for text or blank cells and try "
                    f"again."
                )
            if min_val >= max_val:
                raise ValueError(
                    f"Ingredient '{name}': Min ({min_val}) must be less than Max ({max_val})"
                )
            self.variables.append({
                'name': name,
                'type': 'continuous',
                'bounds': (min_val, max_val),
                'category': 'ingredient',
                'active': True,
            })

            props = {}
            for col in prop_cols:
                try:
                    val = float(row[col])
                    if not pd.isna(val):
                        props[col.lower()] = val
                except (ValueError, TypeError):
                    pass
            self.ingredient_properties[name] = props

        self.variables.extend(process_vars)
        self.pending_batch = None
        self.save()

    def add_process_parameter(self, name, min_val, max_val, baseline=None):
        """Add a process parameter (e.g. baking temperature, mixing time).

        Added mid-run it requires `baseline` — the value used in ALL prior
        batches — because past experiments ran at a fixed setting, not at 0.
        History then encodes at that baseline (its 'absent' value), and min is
        NOT forced to 0 (unlike an ingredient). `baseline` must lie in [min, max].
        """
        for var in self.variables:
            if var['name'] == name:
                return
        min_val, max_val = float(min_val), float(max_val)
        var = {
            'name': name,
            'type': 'continuous',
            'bounds': (min_val, max_val),
            'category': 'process',
            'active': True,
        }
        if self.X_history:
            if len(self.recipe_history) != len(self.X_history):
                raise ValueError(
                    "Cannot add a variable mid-run: some experiments were recorded "
                    "without stored recipes. Start a fresh project or re-import history."
                )
            if baseline is None:
                raise ValueError(
                    "A process parameter added mid-run needs a baseline (the value "
                    "used in all prior batches) so past experiments encode correctly."
                )
            baseline = float(baseline)
            if not (min_val <= baseline <= max_val):
                raise ValueError(
                    f"baseline {baseline} must lie within [{min_val}, {max_val}]."
                )
            var['_absent_value'] = baseline
        self.variables.append(var)
        if self.X_history:
            self._reencode_history()
        self.pending_batch = None
        self.save()

    def remove_process_parameter(self, name):
        """Remove a process parameter by name."""
        self.variables = [
            v for v in self.variables
            if not (v['name'] == name and v.get('category') == 'process')
        ]
        self.pending_batch = None
        self.save()

    def load_screening_model(self, model_obj):
        self.screening_model = model_obj

    # ------------------------------------------------------------------ #
    #  Setup: Objectives
    # ------------------------------------------------------------------ #

    def add_objective(self, name, weight, goal='max', target=None,
                      min_val=None, max_val=None):
        """Add or replace an objective. Returns True if an objective of the
        same name was replaced. Stored scores are recomputed either way so the
        history and the model never disagree with the current weights."""
        name = str(name).strip()
        if not name:
            raise ValueError("Objective name cannot be empty.")
        weight = float(weight)
        if weight <= 0:
            raise ValueError("Weight must be greater than 0.")
        min_val = float(min_val) if min_val is not None else 0.0
        max_val = float(max_val) if max_val is not None else 10.0
        if min_val >= max_val:
            raise ValueError("Range Min must be less than Range Max.")
        if goal == 'target':
            if target is None:
                raise ValueError("Enter a target value for a 'Hit a target' objective.")
            target = float(target)
            if not (min_val <= target <= max_val):
                raise ValueError(
                    f"Target {target:g} must lie within the range {min_val:g} to {max_val:g}."
                )
        else:
            target = None
        replaced = any(obj['name'] == name for obj in self.objectives)
        self.objectives = [obj for obj in self.objectives if obj['name'] != name]
        self.objectives.append({
            'name': name, 'weight': weight, 'goal': goal,
            'target': target, 'min_val': min_val, 'max_val': max_val,
        })
        self.pending_batch = None
        self._recompute_utilities()
        self.save()
        return replaced

    def remove_objective(self, name):
        """Remove an objective and recalculate stored utility scores."""
        self.objectives = [obj for obj in self.objectives if obj['name'] != name]
        self.pending_batch = None
        self._recompute_utilities()
        self.save()

    def _recompute_utilities(self):
        for i, results_dict in enumerate(self.results_history):
            if i < len(self.Y_history):
                self.Y_history[i] = self._compute_utility(results_dict)

    def utility_ceiling(self):
        """The Overall Score a perfect recipe would get: the sum of weights."""
        return float(sum(obj['weight'] for obj in self.objectives))

    # ------------------------------------------------------------------ #
    #  Setup: Constraints
    # ------------------------------------------------------------------ #

    def add_constraint(self, metric, min_val=None, max_val=None):
        """Add a property-based constraint (e.g. total fat, total sodium)."""
        self.constraints.append({
            'metric': metric,
            'min': float(min_val) if min_val is not None else None,
            'max': float(max_val) if max_val is not None else None,
        })
        self.save()

    def remove_constraint(self, index):
        """Remove a property constraint by index."""
        if 0 <= index < len(self.constraints):
            self.constraints.pop(index)
            self.save()

    def add_quantity_constraint(self, ingredients, min_val=None, max_val=None):
        """Add a constraint on the sum of selected ingredient quantities.

        Args:
            ingredients: List of ingredient names whose quantities to sum.
            min_val: Minimum allowed sum (or None for no lower bound).
            max_val: Maximum allowed sum (or None for no upper bound).
        """
        self.quantity_constraints.append({
            'ingredients': list(ingredients),
            'min': float(min_val) if min_val is not None else None,
            'max': float(max_val) if max_val is not None else None,
        })
        self.save()

    def add_total_mass_constraint(self, min_val=None, max_val=None):
        """Shortcut: constrain the total mass (sum of all ingredients)."""
        all_ingredients = [
            v['name'] for v in self.variables
            if v.get('category', 'ingredient') == 'ingredient'
        ]
        self.add_quantity_constraint(all_ingredients, min_val, max_val)

    def remove_quantity_constraint(self, index):
        """Remove a quantity constraint by index."""
        if 0 <= index < len(self.quantity_constraints):
            self.quantity_constraints.pop(index)
            self.save()

    # ------------------------------------------------------------------ #
    #  Utility Scoring
    # ------------------------------------------------------------------ #

    def _compute_utility(self, results_dict):
        """Compute a weighted utility score from raw results."""
        total_utility = 0.0
        for obj in self.objectives:
            raw_val = results_dict.get(obj['name'])
            if raw_val is None:
                continue
            val = float(raw_val)

            min_v = obj.get('min_val', 0.0)
            max_v = obj.get('max_val', 10.0)
            rng = max_v - min_v
            if rng == 0:
                rng = 1.0

            norm_val = max(0.0, min(1.0, (val - min_v) / rng))

            if obj['goal'] == 'max':
                utility = norm_val
            elif obj['goal'] == 'min':
                utility = 1.0 - norm_val
            elif obj['goal'] == 'target':
                targ_val = obj.get('target', (max_v + min_v) / 2)
                norm_targ = (targ_val - min_v) / rng
                utility = max(0.0, 1.0 - abs(norm_val - norm_targ))
            else:
                utility = 0.0

            total_utility += obj['weight'] * utility
        return total_utility

    # ------------------------------------------------------------------ #
    #  History Editing
    # ------------------------------------------------------------------ #

    def edit_result(self, index, new_results_dict):
        """Edit a previously saved result and recalculate its utility."""
        if index < 0 or index >= len(self.Y_history):
            raise IndexError("Result index out of range")
        if index < len(self.results_history):
            self.results_history[index] = dict(new_results_dict)
        self.Y_history[index] = self._compute_utility(new_results_dict)
        self.save()

    def delete_result(self, index):
        """Delete an experiment by index."""
        if index < 0 or index >= len(self.X_history):
            raise IndexError("Result index out of range")
        self.X_history.pop(index)
        self.Y_history.pop(index)
        if index < len(self.recipe_history):
            self.recipe_history.pop(index)
        if index < len(self.results_history):
            self.results_history.pop(index)
        if index < len(self.timestamps_history):
            self.timestamps_history.pop(index)
        self.save()

    def rewind_to(self, index):
        """Keep only experiments 0..index (inclusive), discard the rest."""
        if index < 0 or index >= len(self.X_history):
            raise IndexError("Experiment index out of range")
        keep = index + 1
        self.X_history = self.X_history[:keep]
        self.Y_history = self.Y_history[:keep]
        self.recipe_history = self.recipe_history[:keep]
        self.results_history = self.results_history[:keep]
        self.timestamps_history = self.timestamps_history[:keep]
        self.pending_batch = None
        self.save()

    # ------------------------------------------------------------------ #
    #  Internal: Encoding / Decoding / Bounds
    # ------------------------------------------------------------------ #

    def _encode(self, recipe_dict):
        """Encode a recipe dict into a flat numeric vector.

        Missing keys default to 0.0 / absent so that a variable added mid-run
        (adaptive EGBO) re-encodes prior recipes correctly: the new variable was
        at zero concentration in every past mixture, which is exactly EGBO's
        'earlier observations remain valid' property.
        """
        vector = []
        for var in self.variables:
            if var['type'] == 'continuous':
                # Missing key => the variable's 'absent' value: 0 for an
                # ingredient (not in the recipe), or a process parameter's
                # baseline (prior batches ran at a fixed setting, not 0).
                absent = var.get('_absent_value', 0.0)
                vector.append(float(recipe_dict.get(var['name'], absent)))
            elif var['type'] == 'categorical':
                chosen = recipe_dict.get(var['name'], None)
                for opt in var['options']:
                    vector.append(1.0 if opt == chosen else 0.0)
        return vector

    def _decode(self, vector):
        """Decode a flat numeric vector back into a recipe dict."""
        recipe = {}
        idx = 0
        for var in self.variables:
            if var['type'] == 'continuous':
                recipe[var['name']] = float(vector[idx])
                idx += 1
            elif var['type'] == 'categorical':
                n_opts = len(var['options'])
                one_hot_segment = vector[idx:idx + n_opts]
                best_idx = np.argmax(one_hot_segment)
                recipe[var['name']] = var['options'][best_idx]
                idx += n_opts
        return recipe

    def _get_bounds(self):
        """Return a (2, dim) tensor of [mins, maxs] for all variables."""
        bounds_min, bounds_max = [], []
        for var in self.variables:
            if var['type'] == 'continuous':
                bounds_min.append(var['bounds'][0])
                bounds_max.append(var['bounds'][1])
            elif var['type'] == 'categorical':
                for _ in var['options']:
                    bounds_min.append(0.0)
                    bounds_max.append(1.0)
        return torch.tensor([bounds_min, bounds_max], dtype=torch.double)

    # ------------------------------------------------------------------ #
    #  Internal: Constraint Helpers
    # ------------------------------------------------------------------ #

    def _get_botorch_constraints(self):
        """Build BoTorch inequality constraints from property + quantity constraints."""
        constraints_list = []
        var_indices = {
            var['name']: i
            for i, var in enumerate(self.variables)
            if var['type'] == 'continuous'
        }

        # Property-based constraints (ingredient_properties * quantity)
        for constr in self.constraints:
            metric = constr['metric']
            indices, coeffs = [], []
            offset_lhs = 0.0

            for var_name, idx in var_indices.items():
                prop_val = self.ingredient_properties.get(var_name, {}).get(metric, 0.0)
                if prop_val != 0:
                    var_def = next(v for v in self.variables if v['name'] == var_name)
                    v_min, v_max = var_def['bounds']
                    indices.append(idx)
                    coeffs.append(prop_val * (v_max - v_min))
                    offset_lhs += prop_val * v_min

            if not indices:
                continue
            t_idx = torch.tensor(indices, dtype=torch.long)
            t_coeffs = torch.tensor(coeffs, dtype=torch.double)

            if constr['min'] is not None:
                constraints_list.append((t_idx, t_coeffs, constr['min'] - offset_lhs))
            if constr['max'] is not None:
                constraints_list.append((t_idx, -t_coeffs, -(constr['max'] - offset_lhs)))

        # Quantity constraints (direct sum of ingredient quantities)
        for qc in getattr(self, 'quantity_constraints', []):
            indices, coeffs = [], []
            offset = 0.0

            for ing_name in qc['ingredients']:
                if ing_name in var_indices:
                    idx = var_indices[ing_name]
                    var_def = next(v for v in self.variables if v['name'] == ing_name)
                    v_min, v_max = var_def['bounds']
                    indices.append(idx)
                    coeffs.append(v_max - v_min)
                    offset += v_min

            if not indices:
                continue
            t_idx = torch.tensor(indices, dtype=torch.long)
            t_coeffs = torch.tensor(coeffs, dtype=torch.double)

            if qc['min'] is not None:
                constraints_list.append((t_idx, t_coeffs, qc['min'] - offset))
            if qc['max'] is not None:
                constraints_list.append((t_idx, -t_coeffs, -(qc['max'] - offset)))

        return constraints_list

    def _check_constraints(self, recipe_dict):
        """Return True if a recipe satisfies all constraints."""
        # Property-based constraints
        for constr in self.constraints:
            metric = constr['metric']
            total_val = 0.0
            for var_name, props in self.ingredient_properties.items():
                if var_name in recipe_dict:
                    total_val += recipe_dict[var_name] * props.get(metric, 0.0)
            if constr['min'] is not None and total_val < constr['min']:
                return False
            if constr['max'] is not None and total_val > constr['max']:
                return False

        # Quantity constraints
        for qc in getattr(self, 'quantity_constraints', []):
            total_val = sum(recipe_dict.get(name, 0.0) for name in qc['ingredients'])
            if qc['min'] is not None and total_val < qc['min']:
                return False
            if qc['max'] is not None and total_val > qc['max']:
                return False

        return True

    # ------------------------------------------------------------------ #
    #  Core Loop: Ask / Tell
    # ------------------------------------------------------------------ #

    def ask(self, n_suggestions=1, n_init_random=5):
        """Suggest the next batch of recipes to try.

        Inactive variables (see deactivate_variable) are held at their frozen
        value: the search runs over the active set only, while the surrogate
        still sees every past observation.
        """
        if not self.active_variables():
            raise ValueError(
                "Every variable is inactive — reactivate at least one before "
                "generating recipes."
            )
        bounds_tensor = self._get_bounds()
        dim = bounds_tensor.shape[1]

        # Cold start: space-filling Sobol sequence
        if len(self.X_history) < n_init_random:
            return self._ask_cold_start(n_suggestions, bounds_tensor, dim)

        # Warm: GP-based Bayesian optimization
        return self._ask_optimize(n_suggestions, bounds_tensor, dim)

    def _ask_cold_start(self, n_suggestions, bounds_tensor, dim):
        """Generate initial recipes using Sobol sampling."""
        print(f"DEBUG: Cold start (Sobol batch of {n_suggestions})...")
        sobol = SobolEngine(dimension=dim, scramble=True, seed=len(self.X_history))
        pool_norm = sobol.draw(2048).double()

        # Pin inactive variables so the Sobol design also lives in X_S.
        for col, z in self._get_fixed_features().items():
            pool_norm[:, col] = z

        candidates = [
            self._decode(unnormalize(pool_norm[i], bounds_tensor).numpy().flatten())
            for i in range(2048)
        ]

        # Optional screening model
        if self.screening_model is not None:
            scored = []
            for rec in candidates:
                if self._check_constraints(rec):
                    try:
                        if hasattr(self.screening_model, 'predict'):
                            score = self.screening_model.predict([list(rec.values())])[0]
                        else:
                            score = self.screening_model(rec)
                        scored.append((score, rec))
                    except Exception:
                        pass
            scored.sort(key=lambda x: x[0], reverse=True)
            return [x[1] for x in scored[:n_suggestions]]

        # Standard: pick first feasible candidates
        results = []
        for candidate in candidates:
            if self._check_constraints(candidate):
                results.append(candidate)
            if len(results) >= n_suggestions:
                break

        if not results:
            raise ValueError(
                "No valid recipes found — constraints may be too restrictive. "
                "Try widening ingredient ranges or relaxing constraints."
            )
        return results

    def _ask_optimize(self, n_suggestions, bounds_tensor, dim):
        """Generate recipes using a GP + the configured acquisition (default qLogNEI)."""
        print(f"DEBUG: Optimization step (batch of {n_suggestions})...")
        torch.manual_seed(len(self.X_history))

        train_X = torch.tensor(self.X_history, dtype=torch.double)
        train_Y = torch.tensor(self.Y_history, dtype=torch.double).unsqueeze(-1)
        train_X_norm = normalize(train_X, bounds_tensor)

        input_tf = Warp(d=dim, indices=list(range(dim))) if self.robust else None

        gp = self._build_gp(train_X_norm, train_Y, dim, input_tf)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        acq_func = self._build_acqf(gp, train_X_norm, train_Y)

        # fixed_features drops the inactive columns from the optimization outright
        # (and folds them into the linear constraints), so the candidate is the true
        # argmax over the restricted domain X_S rather than a full-space argmax with
        # coordinates zeroed after the fact.
        candidate_norm, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=torch.stack([torch.zeros(dim), torch.ones(dim)]).double(),
            q=n_suggestions,
            num_restarts=20,
            raw_samples=1024,
            sequential=True,
            inequality_constraints=self._get_botorch_constraints(),
            fixed_features=self._get_fixed_features() or None,
        )

        return [
            self._decode(unnormalize(candidate_norm[i], bounds_tensor).detach().numpy().flatten())
            for i in range(n_suggestions)
        ]

    def tell(self, recipe_dict, results_dict):
        """Record an experiment's recipe and results."""
        if not self.objectives:
            raise ValueError("No objectives defined!")
        for obj in self.objectives:
            if results_dict.get(obj['name']) is None:
                raise ValueError(f"Missing data for {obj['name']}")

        self.X_history.append(self._encode(recipe_dict))
        self.Y_history.append(self._compute_utility(results_dict))
        self.recipe_history.append(dict(recipe_dict))
        self.results_history.append(dict(results_dict))
        self.timestamps_history.append(datetime.now(timezone.utc).isoformat())
        self.save()

    def set_pending_batch(self, batch_or_none):
        """Persist (or clear) the suggested-but-not-yet-rated batch so a user
        who closes the tab mid-experiment finds their recipes on return."""
        self.pending_batch = batch_or_none
        self.save()

    # ------------------------------------------------------------------ #
    #  Adaptivity + expert BO config (optional arms 2 & 3)
    # ------------------------------------------------------------------ #

    def _reencode_history(self):
        """Rebuild X_history from recipe_history under the current variable set.
        Call after any change to self.variables. recipe_history holds the raw
        dicts, so this stays lossless for prior experiments."""
        if self.recipe_history:
            self.X_history = [self._encode(r) for r in self.recipe_history]

    # ------------------------------------------------------------------ #
    #  Non-monotone active set: deactivate / reactivate / remove
    #
    #  Standard EGBO grows the active set monotonically (S_1 <= S_2 <= ...),
    #  which means expert false positives accumulate and never leave: the
    #  expected active-set size tends to the full ambient dimension as the
    #  number of expert rounds grows, so adaptive EGBO degenerates towards
    #  vanilla BO. Allowing the expert to prune gives the active set a bounded
    #  steady state instead.
    #
    #  Pruning is implemented as *deactivation*, not deletion:
    #    - the variable keeps its column in the encoded history, so every past
    #      observation stays in the GP (a recorded experiment is still a valid
    #      observation of f — it is only outside the current search domain);
    #    - the acquisition function is maximized over the active set only;
    #    - reactivation is free, which is what a non-monotone active set needs.
    # ------------------------------------------------------------------ #

    def active_variables(self):
        """Variables in the current active set S_r."""
        return [v for v in self.variables if v.get('active', True)]

    def inactive_variables(self):
        """Variables pruned from S_r but still carried in the history/GP."""
        return [v for v in self.variables if not v.get('active', True)]

    def _var_by_name(self, name):
        for var in self.variables:
            if var['name'] == name:
                return var
        raise ValueError(f"No variable named {name!r}.")

    def _frozen_value(self, var):
        """The value an inactive variable is held at during search."""
        lo, hi = float(var['bounds'][0]), float(var['bounds'][1])
        for key in ('_frozen_at', '_absent_value'):
            if key in var:
                return float(var[key])
        if var.get('category') == 'process':
            return lo  # a process parameter has no meaningful 'off' state
        return min(max(0.0, lo), hi)

    def _achievable_range(self, coeff_of, pinned):
        """Range of sum_i coeff_i * x_i attainable when `pinned` variables are held
        fixed and the rest range over their bounds. Handles negative coefficients."""
        lo = hi = 0.0
        for var in self.variables:
            if var['type'] != 'continuous':
                continue
            coeff = coeff_of(var['name'])
            if coeff == 0:
                continue
            if var['name'] in pinned:
                lo += coeff * pinned[var['name']]
                hi += coeff * pinned[var['name']]
            else:
                a = coeff * float(var['bounds'][0])
                b = coeff * float(var['bounds'][1])
                lo += min(a, b)
                hi += max(a, b)
        return lo, hi

    def _assert_constraints_satisfiable(self, pinned):
        """Raise if holding `pinned` ({name: value}) fixed makes any constraint
        unsatisfiable. Pinning an ingredient to 0 can strand a lower bound that
        the ingredient was carrying, which would otherwise surface later as an
        opaque acquisition-optimization failure."""
        for constr in self.constraints:
            metric = constr['metric']
            lo, hi = self._achievable_range(
                lambda n: self.ingredient_properties.get(n, {}).get(metric, 0.0),
                pinned,
            )
            if constr['min'] is not None and hi < constr['min']:
                raise ValueError(
                    f"Constraint '{metric} >= {constr['min']}' becomes unsatisfiable: "
                    f"the remaining active variables reach at most {hi:.4g}. "
                    f"Relax the constraint before pruning."
                )
            if constr['max'] is not None and lo > constr['max']:
                raise ValueError(
                    f"Constraint '{metric} <= {constr['max']}' becomes unsatisfiable: "
                    f"the pinned values already total {lo:.4g}."
                )

        for i, qc in enumerate(getattr(self, 'quantity_constraints', [])):
            names = set(qc['ingredients'])
            label = " + ".join(qc['ingredients'])
            lo, hi = self._achievable_range(lambda n: 1.0 if n in names else 0.0, pinned)
            if qc['min'] is not None and hi < qc['min']:
                raise ValueError(
                    f"Quantity constraint [{i}] '{label} >= {qc['min']}' becomes "
                    f"unsatisfiable: the remaining active ingredients reach at most "
                    f"{hi:.4g}. Relax the constraint before pruning."
                )
            if qc['max'] is not None and lo > qc['max']:
                raise ValueError(
                    f"Quantity constraint [{i}] '{label} <= {qc['max']}' becomes "
                    f"unsatisfiable: the pinned values already total {lo:.4g}."
                )

    def _get_fixed_features(self):
        """{column: normalized_value} for the inactive columns, in the [0,1]^d frame
        that `ask` hands to optimize_acqf. Returns {} when everything is active, so
        the standard monotone behavior is byte-identical."""
        fixed = {}
        col = 0
        for var in self.variables:
            if var['type'] == 'continuous':
                if not var.get('active', True):
                    lo, hi = float(var['bounds'][0]), float(var['bounds'][1])
                    span = hi - lo
                    z = 0.0 if span <= 0 else (self._frozen_value(var) - lo) / span
                    fixed[col] = float(min(max(z, 0.0), 1.0))
                col += 1
            elif var['type'] == 'categorical':
                n_opts = len(var['options'])
                if not var.get('active', True):
                    for j in range(n_opts):
                        fixed[col + j] = 0.0
                col += n_opts
        return fixed

    def deactivate_variable(self, name, value=None):
        """Prune `name` from the active set, pinning it at `value` (default: 0 for
        an ingredient, the lower bound or stored baseline for a process parameter).

        Nothing is destroyed: past experiments keep contributing to the GP, and
        reactivate_variable puts the variable back. Raises if pruning would make a
        constraint unsatisfiable or would empty the active set.
        """
        var = self._var_by_name(name)
        if not var.get('active', True):
            return
        if len(self.active_variables()) <= 1:
            raise ValueError(
                "Cannot deactivate the last active variable — BO needs at least one "
                "free dimension to search over."
            )
        frozen = self._frozen_value(var) if value is None else float(value)
        lo, hi = float(var['bounds'][0]), float(var['bounds'][1])
        if not (lo <= frozen <= hi):
            raise ValueError(
                f"Frozen value {frozen} for '{name}' must lie within [{lo}, {hi}]."
            )

        pinned = {v['name']: self._frozen_value(v) for v in self.inactive_variables()}
        pinned[name] = frozen
        self._assert_constraints_satisfiable(pinned)

        var['active'] = False
        var['_frozen_at'] = frozen
        self.pending_batch = None
        self.save()

    def reactivate_variable(self, name):
        """Return a pruned variable to the active set (S_r is non-monotone, so a
        variable removed in one round may be re-added in a later one)."""
        var = self._var_by_name(name)
        if var.get('active', True):
            return
        var['active'] = True
        var.pop('_frozen_at', None)
        self.pending_batch = None
        self.save()

    def remove_ingredient(self, name, force=False):
        """Permanently delete an ingredient and drop its column from the history.

        Refuses by default if the ingredient was ever used at a nonzero amount,
        because dropping its column silently rewrites those experiments into
        recipes that were never run. Prefer deactivate_variable, which keeps the
        data. force=True deletes anyway and discards that information.
        """
        var = self._var_by_name(name)
        if var.get('category', 'ingredient') != 'ingredient':
            raise ValueError(
                f"'{name}' is a process parameter — use remove_process_parameter."
            )
        if self.X_history and len(self.recipe_history) != len(self.X_history):
            raise ValueError(
                "Cannot remove a variable: some experiments were recorded without "
                "stored recipes, so the history cannot be re-encoded. Deactivate it "
                "instead, or start a fresh project."
            )

        used = [
            i for i, r in enumerate(self.recipe_history)
            if float(r.get(name, 0.0)) != 0.0
        ]
        if used and not force:
            shown = ", ".join(f"#{i}" for i in used[:5])
            more = f" (+{len(used) - 5} more)" if len(used) > 5 else ""
            raise ValueError(
                f"'{name}' was used at a nonzero amount in experiment(s) {shown}{more}. "
                f"Deleting it would discard that information. Deactivate it instead to "
                f"stop searching over it while keeping the data, or pass force=True."
            )

        remaining = [v for v in self.variables if v['name'] != name]
        if not any(v.get('active', True) for v in remaining):
            raise ValueError("Cannot remove the last active variable.")

        self.variables = remaining
        self.ingredient_properties.pop(name, None)
        for recipe in self.recipe_history:
            recipe.pop(name, None)

        kept = []
        for qc in getattr(self, 'quantity_constraints', []):
            qc['ingredients'] = [n for n in qc['ingredients'] if n != name]
            if qc['ingredients']:
                kept.append(qc)
        self.quantity_constraints = kept

        self._reencode_history()
        self.pending_batch = None
        self.save()

    def set_bo_config(self, spec):
        """Set expert-selected BO hyperparameters (arm 3). Pass None/{} for the
        library defaults (arm 1). Chosen once at project start, not per iteration."""
        self.bo_config = validate_bo_config(spec)
        self.save()

    def fork(self, new_project_name):
        """Branch the current state into a new project, saved under a new name.
        Used to split one run into adaptive vs non-adaptive at the first re-query:
        both share an identical pre-fork history."""
        clone = FoodOptimizer(new_project_name, storage=self.storage)
        state = self.export_json()
        state['project_name'] = new_project_name
        clone.import_json(state)
        clone.screening_model = None
        clone.save()
        return clone

    def export_trajectory(self):
        """Human-readable optimization trajectory for an expert re-query (adaptive
        arm). The app appends the 'available to add' pool (full CSV minus active)."""
        if not self.Y_history:
            return "No experiments recorded yet."
        best_i = int(np.argmax(self.Y_history))
        active = [v['name'] for v in self.active_variables()]
        inactive = [
            f"{v['name']} (pinned at {self._frozen_value(v):.3g})"
            for v in self.inactive_variables()
        ]
        all_names = [v['name'] for v in self.variables]
        lines = [f"Active variables ({len(active)}): {active}"]
        if inactive:
            lines.append(f"Pruned / inactive ({len(inactive)}): {inactive}")
        lines.append("")
        for i, y in enumerate(self.Y_history):
            rec = self.recipe_history[i] if i < len(self.recipe_history) else {}
            # Show every variable that was actually used, including since-pruned
            # ones, so the expert can see what a pruned variable contributed.
            comp = ", ".join(f"{k}={rec[k]:.3g}" for k in all_names if rec.get(k))
            res = self.results_history[i] if i < len(self.results_history) else {}
            attrs = ", ".join(f"{k}={v:.3g}" for k, v in res.items())
            mark = "*" if i == best_i else " "
            lines.append(
                f"{mark} iter {i + 1}: utility={y:.4f} | {comp} | attrs: {attrs}"
            )
        return "\n".join(lines)

    def _build_gp(self, train_X_norm, train_Y, dim, input_tf):
        """Build the GP. With bo_config == None this is byte-identical to the
        original default GP; a config swaps in the expert's kernel/noise."""
        cfg = getattr(self, 'bo_config', None)
        if not cfg:
            return SingleTaskGP(
                train_X_norm, train_Y,
                outcome_transform=Standardize(m=1), input_transform=input_tf,
            )
        kwargs = dict(
            outcome_transform=Standardize(m=1), input_transform=input_tf,
            covar_module=_build_covar(cfg, dim),
        )
        if cfg.get('noise') == 'fixed_tiny':
            yvar = float(max(1e-8, 1e-6 * (train_Y.var().item() + 1e-12)))
            return SingleTaskGP(
                train_X_norm, train_Y,
                train_Yvar=torch.full_like(train_Y, yvar), **kwargs,
            )
        gp = SingleTaskGP(train_X_norm, train_Y, **kwargs)
        if cfg.get('noise') == 'low':
            gp.likelihood.noise_covar.register_prior(
                'noise_prior', GammaPrior(1.1, 50.0), 'raw_noise',
            )
        return gp

    def _build_acqf(self, gp, train_X_norm, train_Y):
        cfg = getattr(self, 'bo_config', None)
        acq = (cfg or {}).get('acquisition', 'qlognei')
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]))
        if acq == 'qlogei':
            return qLogExpectedImprovement(
                model=gp, best_f=float(train_Y.max().item()), sampler=sampler,
            )
        if acq == 'qucb':
            return qUpperConfidenceBound(model=gp, beta=0.2, sampler=sampler)
        return qLogNoisyExpectedImprovement(
            model=gp, X_baseline=train_X_norm, sampler=sampler,
        )

    # ------------------------------------------------------------------ #
    #  Persistence: Save / Load / Export / Import
    # ------------------------------------------------------------------ #

    def save(self):
        # Delegates to the storage backend. StorageError (a cloud backend
        # failure) is swallowed into save_error so the user's in-memory work
        # survives; the app shows it as a banner. Local I/O errors propagate.
        try:
            self.storage.save(self.project_name, self.export_json())
        except StorageError as e:
            self.save_error = str(e)
        else:
            self.save_error = None

    def load(self):
        """Load the project via the storage backend. Sets self.load_error to a
        plain-language message on failure instead of silently producing a
        blank project. Returns True on success, False on failure."""
        self.load_error = None
        try:
            state = self.storage.load(self.project_name)
        except StorageError as e:
            self.load_error = str(e)
            return False
        if state is None:
            self.load_error = (
                "This project could not be found. It may have been renamed "
                "or archived."
            )
            return False
        try:
            self.import_json(state)
        except Exception:
            self.load_error = (
                "This project file is damaged and could not be opened. "
                "If you have a backup, use Restore from backup; otherwise "
                "check the FoodOptimizer > backups folder in your home "
                "folder for a recent copy."
            )
            return False
        # Historical behavior for local files: re-save migrates a legacy
        # pickle or an older-CLASS_VERSION JSON file to the current format;
        # it's a no-op for an already-current file, so a plain read-only open
        # doesn't rewrite the file and invalidate another session's conflict
        # stamp. Cloud loads are read-only.
        if self.storage.persist_after_load and state.get('CLASS_VERSION', 0) < self.CLASS_VERSION:
            self.save()
        return True

    def export_json(self):
        """Export full project state as a JSON-serializable dict."""

        def _make_serializable(obj):
            if isinstance(obj, tuple):
                return list(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        state = {
            'project_name': self.project_name,
            'variables': self.variables,
            'objectives': self.objectives,
            'ingredient_properties': self.ingredient_properties,
            'constraints': self.constraints,
            'quantity_constraints': self.quantity_constraints,
            'robust': self.robust,
            'X_history': self.X_history,
            'Y_history': self.Y_history,
            'recipe_history': self.recipe_history,
            'results_history': self.results_history,
            'timestamps_history': self.timestamps_history,
            'pending_batch': self.pending_batch,
            'bo_config': self.bo_config,
            'CLASS_VERSION': self.CLASS_VERSION,
        }
        return json.loads(json.dumps(state, default=_make_serializable))

    @staticmethod
    def validate_state(state):
        """Check a backup dict before importing it. Returns a summary dict
        (name, experiments, ingredients, version) or raises ValueError with a
        message suitable for the UI. import_json assigns attributes one by
        one, so validating first is what keeps a bad file from leaving the
        optimizer half-mutated."""
        bad = "This file is not a Food Optimizer backup."
        if not isinstance(state, dict):
            raise ValueError(bad)
        required = {
            'variables': list, 'objectives': list,
            'recipe_history': list, 'results_history': list,
        }
        if not all(k in state for k in required):
            raise ValueError(bad)
        for key, typ in required.items():
            if not isinstance(state[key], typ):
                raise ValueError(f"This backup's '{key}' section has the wrong shape.")
        version = state.get('CLASS_VERSION')
        if not isinstance(version, int):
            raise ValueError(bad)
        if version > FoodOptimizer.CLASS_VERSION:
            raise ValueError(
                "This backup was made with a newer version of Food Optimizer. "
                "Update the app, then try again."
            )
        if len(state['recipe_history']) != len(state['results_history']):
            raise ValueError("This backup is inconsistent: recipes and results differ in count.")
        ingredients = sum(
            1 for v in state['variables']
            if isinstance(v, dict) and v.get('category', 'ingredient') == 'ingredient'
        )
        return {
            'name': state.get('project_name', '(unnamed)'),
            'experiments': len(state['recipe_history']),
            'ingredients': ingredients,
            'version': version,
        }

    def import_json(self, state):
        """Restore project state from a JSON dict (as produced by export_json)."""
        self.project_name = state.get('project_name', self.project_name)
        self.filename = f"{self.project_name}.pkl"
        self.variables = state.get('variables', [])
        self.objectives = state.get('objectives', [])
        self.ingredient_properties = state.get('ingredient_properties', {})
        self.constraints = state.get('constraints', [])
        self.quantity_constraints = state.get('quantity_constraints', [])
        self.robust = state.get('robust', False)
        self.bo_config = validate_bo_config(state.get('bo_config', None))
        self.recipe_history = state.get('recipe_history', [])
        self.results_history = state.get('results_history', [])
        self.timestamps_history = state.get('timestamps_history', [])
        self.pending_batch = state.get('pending_batch', None)
        while len(self.timestamps_history) < len(self.results_history):
            self.timestamps_history.append(None)  # pre-feature files/backups

        for var in self.variables:
            if 'bounds' in var and isinstance(var['bounds'], list):
                var['bounds'] = tuple(var['bounds'])
            var.setdefault('category', 'ingredient')
            var.setdefault('active', True)

        # Rebuild encoded vectors and utility scores from raw data
        if self.recipe_history and self.variables:
            self.X_history = [self._encode(r) for r in self.recipe_history]
        else:
            self.X_history = state.get('X_history', [])

        if self.results_history and self.objectives:
            self.Y_history = [self._compute_utility(r) for r in self.results_history]
        else:
            self.Y_history = state.get('Y_history', [])

        # A successful import means the in-memory state is valid again, so a
        # restore-from-backup clears any earlier damaged-file error.
        self.load_error = None
