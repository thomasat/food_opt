import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch.quasirandom import SobolEngine

from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Warp
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood


class FoodOptimizer:
    CLASS_VERSION = 3  # bump when adding methods/attrs to force session refresh

    def __init__(self, project_name="experiment", robust=False):
        """Initialize or load a food optimization project.

        Args:
            project_name: Name used for the .pkl save file.
            robust: If True, uses Input Warping for cliffs/traps.
                    If False (default), uses standard GP for smooth problems.
        """
        self.project_name = project_name
        self.robust = robust
        self.filename = f"{project_name}.pkl"

        self.variables = []
        self.objectives = []
        self.ingredient_properties = {}
        self.constraints = []
        self.quantity_constraints = []
        self.screening_model = None

        self.X_history = []
        self.Y_history = []
        self.recipe_history = []
        self.results_history = []

        if os.path.exists(self.filename):
            self.load()
        else:
            self.save()

    # ------------------------------------------------------------------ #
    #  Setup: Ingredients & Process Parameters
    # ------------------------------------------------------------------ #

    def add_ingredient(self, name, min_val, max_val):
        """Add a single ingredient (used for benchmarking)."""
        for var in self.variables:
            if var['name'] == name:
                return
        self.variables.append({
            'name': name,
            'type': 'continuous',
            'bounds': (float(min_val), float(max_val)),
            'category': 'ingredient',
        })
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

        process_vars = [v for v in self.variables if v.get('category') == 'process']
        self.variables = []
        self.ingredient_properties = {}

        standard_cols = {'Name', 'Min', 'Max', 'Type'}
        prop_cols = [c for c in df.columns if c not in standard_cols]

        for _, row in df.iterrows():
            name = row['Name']
            min_val, max_val = float(row['Min']), float(row['Max'])
            if min_val >= max_val:
                raise ValueError(
                    f"Ingredient '{name}': Min ({min_val}) must be less than Max ({max_val})"
                )
            self.variables.append({
                'name': name,
                'type': 'continuous',
                'bounds': (min_val, max_val),
                'category': 'ingredient',
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
        self.save()

    def add_process_parameter(self, name, min_val, max_val):
        """Add a process parameter (e.g. baking temperature, mixing time)."""
        for var in self.variables:
            if var['name'] == name:
                return
        self.variables.append({
            'name': name,
            'type': 'continuous',
            'bounds': (float(min_val), float(max_val)),
            'category': 'process',
        })
        self.save()

    def remove_process_parameter(self, name):
        """Remove a process parameter by name."""
        self.variables = [
            v for v in self.variables
            if not (v['name'] == name and v.get('category') == 'process')
        ]
        self.save()

    def load_screening_model(self, model_obj):
        self.screening_model = model_obj

    # ------------------------------------------------------------------ #
    #  Setup: Objectives
    # ------------------------------------------------------------------ #

    def add_objective(self, name, weight, goal='max', target=None,
                      min_val=None, max_val=None):
        self.objectives = [obj for obj in self.objectives if obj['name'] != name]
        self.objectives.append({
            'name': name,
            'weight': float(weight),
            'goal': goal,
            'target': float(target) if target is not None else None,
            'min_val': float(min_val) if min_val is not None else 0.0,
            'max_val': float(max_val) if max_val is not None else 10.0,
        })
        self.save()

    def remove_objective(self, name):
        """Remove an objective and recalculate stored utility scores."""
        self.objectives = [obj for obj in self.objectives if obj['name'] != name]
        if self.results_history:
            for i, results_dict in enumerate(self.results_history):
                if i < len(self.Y_history):
                    self.Y_history[i] = self._compute_utility(results_dict)
        self.save()

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
        self.save()

    # ------------------------------------------------------------------ #
    #  Internal: Encoding / Decoding / Bounds
    # ------------------------------------------------------------------ #

    def _encode(self, recipe_dict):
        """Encode a recipe dict into a flat numeric vector."""
        vector = []
        for var in self.variables:
            if var['type'] == 'continuous':
                vector.append(recipe_dict[var['name']])
            elif var['type'] == 'categorical':
                chosen = recipe_dict[var['name']]
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
        bounds_tensor = self._get_bounds()
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
        """Suggest the next batch of recipes to try."""
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
        """Generate recipes using GP + qNoisyExpectedImprovement."""
        print(f"DEBUG: Optimization step (batch of {n_suggestions})...")
        torch.manual_seed(len(self.X_history))

        train_X = torch.tensor(self.X_history, dtype=torch.double)
        train_Y = torch.tensor(self.Y_history, dtype=torch.double).unsqueeze(-1)
        train_X_norm = normalize(train_X, bounds_tensor)

        input_tf = Warp(d=dim, indices=list(range(dim))) if self.robust else None

        gp = SingleTaskGP(
            train_X_norm,
            train_Y,
            outcome_transform=Standardize(m=1),
            input_transform=input_tf,
        )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]))
        acq_func = qLogNoisyExpectedImprovement(
            model=gp,
            X_baseline=train_X_norm,
            sampler=sampler,
        )

        candidate_norm, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=torch.stack([torch.zeros(dim), torch.ones(dim)]).double(),
            q=n_suggestions,
            num_restarts=20,
            raw_samples=1024,
            sequential=True,
            inequality_constraints=self._get_botorch_constraints(),
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
        self.save()

    # ------------------------------------------------------------------ #
    #  Persistence: Save / Load / Export / Import
    # ------------------------------------------------------------------ #

    def save(self):
        state = self.__dict__.copy()
        state.pop('screening_model', None)
        with open(self.filename, 'wb') as f:
            pickle.dump(state, f)

    def load(self):
        try:
            with open(self.filename, 'rb') as f:
                state = pickle.load(f)
                self.__dict__.update(state)
            self.screening_model = None

            # Backward compatibility for older pickle files
            if not hasattr(self, 'quantity_constraints'):
                self.quantity_constraints = []
            if not hasattr(self, 'recipe_history'):
                self.recipe_history = []
            if not hasattr(self, 'results_history'):
                self.results_history = []
            for var in self.variables:
                if 'category' not in var:
                    var['category'] = 'ingredient'

            # Recalculate utility scores to pick up formula changes
            if self.results_history and self.objectives:
                for i, results_dict in enumerate(self.results_history):
                    if i < len(self.Y_history):
                        self.Y_history[i] = self._compute_utility(results_dict)
        except Exception:
            print("Warning: Load failed.")

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
            'CLASS_VERSION': self.CLASS_VERSION,
        }
        return json.loads(json.dumps(state, default=_make_serializable))

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
        self.recipe_history = state.get('recipe_history', [])
        self.results_history = state.get('results_history', [])

        for var in self.variables:
            if 'bounds' in var and isinstance(var['bounds'], list):
                var['bounds'] = tuple(var['bounds'])

        # Rebuild encoded vectors and utility scores from raw data
        if self.recipe_history and self.variables:
            self.X_history = [self._encode(r) for r in self.recipe_history]
        else:
            self.X_history = state.get('X_history', [])

        if self.results_history and self.objectives:
            self.Y_history = [self._compute_utility(r) for r in self.results_history]
        else:
            self.Y_history = state.get('Y_history', [])

        self.save()
