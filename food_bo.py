import torch
import numpy as np
import pickle
import os
import pandas as pd

# BoTorch / GPyTorch Imports
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.transforms import normalize, unnormalize
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import Warp
from torch.quasirandom import SobolEngine

# Robust Acquisition Function
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler

class FoodOptimizer:
    def __init__(self, project_name="experiment", robust=False):
        """
        robust (bool): 
          If False (Default), uses standard GP. Faster convergence on smooth problems (e.g. Synergy).
          If True, uses Input Warping. Slower start, but handles cliffs/traps (e.g. Phase Separation).
        """
        self.project_name = project_name
        self.robust = robust
        self.filename = f"{project_name}.pkl"
        
        # --- STATE ---
        self.variables = [] 
        self.objectives = [] 
        self.ingredient_properties = {} 
        self.constraints = [] 
        self.screening_model = None 
        
        self.X_history = []  
        self.Y_history = []  
        
        if os.path.exists(self.filename):
            self.load()
        else:
            self.save()

    # --- SETUP METHODS ---

    def add_ingredient(self, name, min_val, max_val):
        """Manually add a single ingredient (Used for Benchmarking)."""
        for var in self.variables:
            if var['name'] == name: return
        self.variables.append({
            'name': name,
            'type': 'continuous',
            'bounds': (float(min_val), float(max_val))
        })
        self.save()

    def load_ingredients_from_csv(self, df):
        """Bulk load from CSV (Used for App)."""
        self.variables = [] 
        self.ingredient_properties = {}
        
        standard_cols = ['Name', 'Min', 'Max', 'Type']
        prop_cols = [c for c in df.columns if c not in standard_cols]
        
        for _, row in df.iterrows():
            name = row['Name']
            self.variables.append({
                'name': name,
                'type': 'continuous',
                'bounds': (float(row['Min']), float(row['Max']))
            })
            
            props = {}
            for col in prop_cols:
                props[col.lower()] = float(row[col])
            self.ingredient_properties[name] = props
        self.save()

    def load_screening_model(self, model_obj):
        self.screening_model = model_obj

    def add_objective(self, name, weight, goal='max', target=None, min_val=None, max_val=None):
        self.objectives = [obj for obj in self.objectives if obj['name'] != name]
        self.objectives.append({
            'name': name,
            'weight': float(weight),
            'goal': goal,
            'target': float(target) if target is not None else None,
            'min_val': float(min_val) if min_val is not None else 0.0,
            'max_val': float(max_val) if max_val is not None else 10.0
        })
        self.save()

    def add_constraint(self, metric, min_val=None, max_val=None):
        self.constraints.append({
            'metric': metric,
            'min': float(min_val) if min_val is not None else None,
            'max': float(max_val) if max_val is not None else None
        })
        self.save()

    # --- INTERNAL HELPERS ---
    def _get_botorch_constraints(self):
        # ... (Constraint logic identical to previous versions) ...
        bounds_tensor = self._get_bounds()
        constraints_list = []
        var_indices = {var['name']: i for i, var in enumerate(self.variables) if var['type'] == 'continuous'}
        
        for constr in self.constraints:
            metric = constr['metric']
            indices = []
            coeffs = []
            offset_lhs = 0.0
            
            for var_name, idx in var_indices.items():
                prop_val = self.ingredient_properties.get(var_name, {}).get(metric, 0.0)
                if prop_val != 0:
                    var_def = next(v for v in self.variables if v['name'] == var_name)
                    v_min, v_max = var_def['bounds']
                    v_range = v_max - v_min
                    indices.append(idx)
                    coeffs.append(prop_val * v_range)
                    offset_lhs += prop_val * v_min

            if not indices: continue
            tensor_idx = torch.tensor(indices, dtype=torch.long)
            tensor_coeffs = torch.tensor(coeffs, dtype=torch.double)
            
            if constr['min'] is not None:
                rhs = constr['min'] - offset_lhs
                constraints_list.append((tensor_idx, tensor_coeffs, rhs))
            if constr['max'] is not None:
                rhs = -(constr['max'] - offset_lhs)
                constraints_list.append((tensor_idx, -tensor_coeffs, rhs))
        return constraints_list

    def _check_constraints(self, recipe_dict):
        for constr in self.constraints:
            metric = constr['metric']
            total_val = 0.0
            for var_name, props in self.ingredient_properties.items():
                if var_name in recipe_dict:
                    total_val += recipe_dict[var_name] * props.get(metric, 0.0)
            if constr['min'] is not None and total_val < constr['min']: return False
            if constr['max'] is not None and total_val > constr['max']: return False
        return True

    def _encode(self, recipe_dict):
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
        recipe = {}
        idx = 0
        for var in self.variables:
            if var['type'] == 'continuous':
                recipe[var['name']] = float(vector[idx])
                idx += 1
            elif var['type'] == 'categorical':
                n_opts = len(var['options'])
                one_hot_segment = vector[idx : idx + n_opts]
                best_idx = np.argmax(one_hot_segment)
                recipe[var['name']] = var['options'][best_idx]
                idx += n_opts
        return recipe

    def _get_bounds(self):
        bounds_min = []
        bounds_max = []
        for var in self.variables:
            if var['type'] == 'continuous':
                bounds_min.append(var['bounds'][0])
                bounds_max.append(var['bounds'][1])
            elif var['type'] == 'categorical':
                for _ in var['options']:
                    bounds_min.append(0.0)
                    bounds_max.append(1.0)
        return torch.tensor([bounds_min, bounds_max], dtype=torch.double)

    # --- CORE LOOP ---

    def ask(self, n_suggestions=1, n_init_random=5):
        bounds_tensor = self._get_bounds()
        dim = bounds_tensor.shape[1]

        # 1. COLD START (Sobol)
        if len(self.X_history) < n_init_random:
            print(f"DEBUG: Cold start (Sobol batch of {n_suggestions})...")
            sobol = SobolEngine(dimension=dim, scramble=True, seed=len(self.X_history))
            pool_norm = sobol.draw(2048).double()
            
            candidates = []
            for i in range(2048):
                vec = unnormalize(pool_norm[i], bounds_tensor).numpy().flatten()
                candidates.append(self._decode(vec))
            
            # Screening Model Logic (Optional)
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
                        except: pass
                scored.sort(key=lambda x: x[0], reverse=True)
                return [x[1] for x in scored[:n_suggestions]]

            # Standard Random
            results = []
            idx = 0
            while len(results) < n_suggestions and idx < len(candidates):
                if self._check_constraints(candidates[idx]):
                    results.append(candidates[idx])
                idx += 1
            return results

        # 2. OPTIMIZATION (GP + qNEI)
        print(f"DEBUG: Optimization Step (Batch of {n_suggestions})...")
        train_X = torch.tensor(self.X_history, dtype=torch.double)
        train_Y = torch.tensor(self.Y_history, dtype=torch.double).unsqueeze(-1)
        train_X_norm = normalize(train_X, bounds_tensor)
        
        # --- ROBUSTNESS TOGGLE ---
        if self.robust:
            # Slower start, but handles cliffs (Rastrigin)
            input_tf = Warp(d=dim, indices=list(range(dim)))
        else:
            # Faster start, assumes smooth synergy (Hartmann)
            input_tf = None
            
        gp = SingleTaskGP(
            train_X_norm, 
            train_Y, 
            outcome_transform=Standardize(m=1),
            input_transform=input_tf
        )
        
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        
        # Use qNoisyExpectedImprovement for stability
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]))
        acq_func = qLogNoisyExpectedImprovement(
            model=gp,
            X_baseline=train_X_norm,
            sampler=sampler
        )
        
        candidate_norm, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=torch.stack([torch.zeros(dim), torch.ones(dim)]).double(),
            q=n_suggestions, 
            num_restarts=20,    # Increased for stability
            raw_samples=1024,   # Increased for stability
            sequential=True,
            inequality_constraints=self._get_botorch_constraints()
        )
        
        results = []
        for i in range(n_suggestions):
            vec = unnormalize(candidate_norm[i], bounds_tensor).detach().numpy().flatten()
            results.append(self._decode(vec))
            
        return results

    def tell(self, recipe_dict, results_dict):
        total_utility = 0.0
        if not self.objectives: raise ValueError("No objectives defined!")

        for obj in self.objectives:
            raw_val = results_dict.get(obj['name'])
            if raw_val is None: raise ValueError(f"Missing data for {obj['name']}")
            val = float(raw_val)
            
            # Normalize [0, 1]
            min_v = obj.get('min_val', 0.0)
            max_v = obj.get('max_val', 10.0)
            rng = max_v - min_v
            if rng == 0: rng = 1.0
            
            norm_val = (val - min_v) / rng
            norm_val = max(0.0, min(1.0, norm_val))
            
            # Calculate Utility
            if obj['goal'] == 'max': 
                utility = norm_val
            elif obj['goal'] == 'min': 
                utility = 1.0 - norm_val
            elif obj['goal'] == 'target':
                targ_val = obj.get('target', (max_v+min_v)/2)
                norm_targ = (targ_val - min_v) / rng
                dist = norm_val - norm_targ
                utility = 1.0 - (dist ** 2)
            
            total_utility += obj['weight'] * utility
        
        x_vec = self._encode(recipe_dict)
        self.X_history.append(x_vec)
        self.Y_history.append(total_utility)
        self.save()

    def save(self):
        state = self.__dict__.copy()
        if 'screening_model' in state: del state['screening_model']
        with open(self.filename, 'wb') as f: pickle.dump(state, f)

    def load(self):
        try:
            with open(self.filename, 'rb') as f:
                state = pickle.load(f)
                self.__dict__.update(state)
            self.screening_model = None 
        except: print("Warning: Load failed.")