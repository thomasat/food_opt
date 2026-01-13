import torch
import numpy as np
import pickle
import os

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.transforms import normalize, unnormalize
from botorch.models.transforms.outcome import Standardize
from torch.quasirandom import SobolEngine

# Switched to qUpperConfidenceBound for Batch support
from botorch.acquisition import qUpperConfidenceBound

class FoodOptimizer:
    def __init__(self, project_name="experiment"):
        self.project_name = project_name
        self.filename = f"{project_name}.pkl"
        
        # --- STATE ---
        self.variables = [] 
        self.objectives = [] 
        self.X_history = []  
        self.Y_history = []  
        
        if os.path.exists(self.filename):
            self.load()
        else:
            # FIX: Create the file immediately if it doesn't exist
            self.save()

    # --- DEFINITION METHODS ---

    def add_ingredient(self, name, min_val, max_val):
        for var in self.variables:
            if var['name'] == name:
                return
        self.variables.append({
            'name': name,
            'type': 'continuous',
            'bounds': (float(min_val), float(max_val))
        })
        self.save() 

    def add_categorical(self, name, options):
        for var in self.variables:
            if var['name'] == name:
                return
        self.variables.append({
            'name': name,
            'type': 'categorical',
            'options': options
        })
        self.save() 

    def add_objective(self, name, weight, goal='max'):
        for obj in self.objectives:
            if obj['name'] == name:
                return
        self.objectives.append({
            'name': name,
            'weight': float(weight),
            'goal': goal
        })
        self.save() 

    # --- INTERNAL HELPERS ---
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

    # --- CORE LOOP (Updated for Batching) ---

    def ask(self, n_suggestions=1, n_init_random=5):
        """
        Generates a batch of `n_suggestions` recipes.
        Returns: List of dictionaries.
        """
        bounds_tensor = self._get_bounds()
        dim = bounds_tensor.shape[1]

        # 1. COLD START (Sobol)
        if len(self.X_history) < n_init_random:
            print(f"DEBUG: Cold start (Sobol batch of {n_suggestions})...")
            sobol = SobolEngine(dimension=dim, scramble=True, seed=len(self.X_history))
            cand_norm = sobol.draw(n_suggestions).double()
            
            results = []
            for i in range(n_suggestions):
                vec = unnormalize(cand_norm[i], bounds_tensor).numpy().flatten()
                results.append(self._decode(vec))
            return results

        # 2. OPTIMIZATION (GP + qUCB)
        print(f"DEBUG: Optimization Step (Batch of {n_suggestions})...")
        train_X = torch.tensor(self.X_history, dtype=torch.double)
        train_Y = torch.tensor(self.Y_history, dtype=torch.double).unsqueeze(-1)
        
        train_X_norm = normalize(train_X, bounds_tensor)
        
        gp = SingleTaskGP(train_X_norm, train_Y, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        
        qUCB = qUpperConfidenceBound(model=gp, beta=2.0)
        
        candidate_norm, _ = optimize_acqf(
            acq_function=qUCB,
            bounds=torch.stack([torch.zeros(dim), torch.ones(dim)]).double(),
            q=n_suggestions, 
            num_restarts=10,
            raw_samples=512,
            sequential=True
        )
        
        results = []
        for i in range(n_suggestions):
            vec = unnormalize(candidate_norm[i], bounds_tensor).detach().numpy().flatten()
            results.append(self._decode(vec))
            
        return results

    def tell(self, recipe_dict, results_dict):
        total_score = 0.0
        
        if not self.objectives:
            raise ValueError("No objectives defined! Call add_objective() first.")

        for obj in self.objectives:
            raw_val = results_dict.get(obj['name'])
            
            if raw_val is None:
                raise ValueError(f"Missing data for objective '{obj['name']}'.")
                
            if obj['goal'] == 'max':
                total_score += obj['weight'] * float(raw_val)
            else:
                total_score -= obj['weight'] * float(raw_val)
        
        x_vec = self._encode(recipe_dict)
        
        self.X_history.append(x_vec)
        self.Y_history.append(total_score)
        
        print(f"Recorded -> Weighted Score: {total_score:.4f}")
        self.save()

    def save(self):
        with open(self.filename, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load(self):
        try:
            with open(self.filename, 'rb') as f:
                state = pickle.load(f)
                self.__dict__.update(state)
            print(f"Loaded project '{self.project_name}' with {len(self.X_history)} trials.")
        except Exception as e:
            print(f"Warning: Could not load state ({e}). Starting fresh.")