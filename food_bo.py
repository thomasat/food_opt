import torch
import numpy as np
import pickle
import os
import pandas as pd # Added pandas

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.transforms import normalize, unnormalize
from botorch.models.transforms.outcome import Standardize
from torch.quasirandom import SobolEngine
from botorch.acquisition import qUpperConfidenceBound

class FoodOptimizer:
    def __init__(self, project_name="experiment"):
        self.project_name = project_name
        self.filename = f"{project_name}.pkl"
        
        # --- STATE ---
        self.variables = [] 
        self.objectives = [] 
        self.ingredient_properties = {} 
        self.constraints = [] 
        self.screening_model = None # New: Holds the low-fi model
        
        self.X_history = []  
        self.Y_history = []  
        
        if os.path.exists(self.filename):
            self.load()
        else:
            self.save()

    # --- SETUP METHODS ---

    def load_ingredients_from_csv(self, df):
        """
        Parses a DataFrame to bulk-add ingredients.
        Expected Cols: 'Name', 'Min', 'Max'
        Optional Cols: 'Cost', 'Protein', etc.
        """
        # Clear existing to prevent duplicates during re-upload
        self.variables = [] 
        self.ingredient_properties = {}
        
        # Identify property columns (anything that isn't Name/Min/Max/Type)
        standard_cols = ['Name', 'Min', 'Max', 'Type']
        prop_cols = [c for c in df.columns if c not in standard_cols]
        
        for _, row in df.iterrows():
            name = row['Name']
            min_val = row['Min']
            max_val = row['Max']
            
            # Add Variable
            self.variables.append({
                'name': name,
                'type': 'continuous', # Defaulting to continuous for CSV simplicity
                'bounds': (float(min_val), float(max_val))
            })
            
            # Extract Properties
            props = {}
            for col in prop_cols:
                props[col.lower()] = float(row[col])
            
            self.ingredient_properties[name] = props
            
        print(f"Loaded {len(self.variables)} ingredients from CSV.")
        self.save()

    def load_screening_model(self, model_obj):
        """Loads a python object to use for screening."""
        self.screening_model = model_obj
        print("Low-fidelity model loaded.")
        # We don't pickle the external model to avoid size/compatibility issues
        # It must be re-loaded per session in the UI.

    def add_objective(self, name, weight, goal='max'):
        # Clear old objective if same name exists
        self.objectives = [obj for obj in self.objectives if obj['name'] != name]
        self.objectives.append({
            'name': name,
            'weight': float(weight),
            'goal': goal
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

        # 1. COLD START (Sobol + Model Screening)
        if len(self.X_history) < n_init_random:
            print(f"DEBUG: Cold start (Sobol batch of {n_suggestions})...")
            
            # Draw large pool
            sobol = SobolEngine(dimension=dim, scramble=True, seed=len(self.X_history))
            pool_norm = sobol.draw(2048).double()
            
            candidates = []
            for i in range(2048):
                vec = unnormalize(pool_norm[i], bounds_tensor).numpy().flatten()
                candidates.append(self._decode(vec))
            
            # If screening model exists, use it
            if self.screening_model is not None:
                print("DEBUG: Using Low-Fidelity Model for screening.")
                scored = []
                for rec in candidates:
                    if self._check_constraints(rec):
                        try:
                            # Assume model is callable or has .predict
                            if hasattr(self.screening_model, 'predict'):
                                # Some models expect DF, some dict. Wrap in list/df if needed.
                                score = self.screening_model.predict([list(rec.values())])[0]
                            else:
                                score = self.screening_model(rec)
                            scored.append((score, rec))
                        except Exception as e:
                            pass 
                
                scored.sort(key=lambda x: x[0], reverse=True)
                return [x[1] for x in scored[:n_suggestions]]

            # Standard random validation
            results = []
            idx = 0
            while len(results) < n_suggestions and idx < len(candidates):
                if self._check_constraints(candidates[idx]):
                    results.append(candidates[idx])
                idx += 1
            return results

        # 2. OPTIMIZATION
        train_X = torch.tensor(self.X_history, dtype=torch.double)
        train_Y = torch.tensor(self.Y_history, dtype=torch.double).unsqueeze(-1)
        train_X_norm = normalize(train_X, bounds_tensor)
        
        gp = SingleTaskGP(train_X_norm, train_Y, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        
        qUCB = qUpperConfidenceBound(model=gp, beta=2.0)
        inequality_constraints = self._get_botorch_constraints()
        
        candidate_norm, _ = optimize_acqf(
            acq_function=qUCB,
            bounds=torch.stack([torch.zeros(dim), torch.ones(dim)]).double(),
            q=n_suggestions, 
            num_restarts=10,
            raw_samples=512,
            sequential=True,
            inequality_constraints=inequality_constraints
        )
        
        results = []
        for i in range(n_suggestions):
            vec = unnormalize(candidate_norm[i], bounds_tensor).detach().numpy().flatten()
            results.append(self._decode(vec))
            
        return results

    def tell(self, recipe_dict, results_dict):
        total_score = 0.0
        if not self.objectives: raise ValueError("No objectives defined!")

        for obj in self.objectives:
            raw_val = results_dict.get(obj['name'])
            if raw_val is None: raise ValueError(f"Missing data for {obj['name']}")
            
            if obj['goal'] == 'max': total_score += obj['weight'] * float(raw_val)
            else: total_score -= obj['weight'] * float(raw_val)
        
        x_vec = self._encode(recipe_dict)
        self.X_history.append(x_vec)
        self.Y_history.append(total_score)
        self.save()

    def save(self):
        # We don't save the screening_model in pickle (security/size). 
        # User must re-upload it.
        state = self.__dict__.copy()
        if 'screening_model' in state:
            del state['screening_model']
        with open(self.filename, 'wb') as f:
            pickle.dump(state, f)

    def load(self):
        try:
            with open(self.filename, 'rb') as f:
                state = pickle.load(f)
                self.__dict__.update(state)
            self.screening_model = None # Reset model on load
            print(f"Loaded project {self.project_name}")
        except Exception:
            print("Warning: Load failed.")