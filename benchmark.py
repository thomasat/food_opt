"""
Synthetic Benchmark for LLM-Structured Dimensionality Reduction in BO.

Establishes ground-truth food-formulation problems where:
- The search space is high-dimensional (d=30 ingredients)
- The true objective depends on k=5 latent factors
- A known loading matrix A maps ingredients -> factors
- We can measure subspace recovery quality and sample efficiency

Usage:
    python benchmark.py [--trials 30] [--seeds 5] [--budget 25]
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from scipy.linalg import subspace_angles
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.models.transforms.outcome import Standardize
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

# ---------------------------------------------------------------------------
# 1. INGREDIENT DEFINITIONS (ground-truth food science structure)
# ---------------------------------------------------------------------------

# 30 ingredients spanning a realistic plant-based formulation
INGREDIENTS = [
    # Proteins (0-4)
    "pea_protein_isolate", "soy_protein_concentrate", "wheat_gluten",
    "chickpea_flour", "rice_protein",
    # Fats (5-9)
    "coconut_oil", "canola_oil", "cocoa_butter", "sunflower_oil", "shea_butter",
    # Starches & Binders (10-14)
    "tapioca_starch", "potato_starch", "methylcellulose", "xanthan_gum",
    "konjac_glucomannan",
    # Flavoring & Color (15-19)
    "beet_juice_powder", "yeast_extract", "smoked_paprika", "onion_powder",
    "natural_flavor_blend",
    # Moisture & Acids (20-24)
    "water", "apple_cider_vinegar", "lemon_juice_concentrate",
    "tomato_paste", "soy_sauce",
    # Minerals & Minor (25-29)
    "salt", "calcium_carbonate", "iron_supplement", "b12_supplement",
    "transglutaminase",
]

D_FULL = len(INGREDIENTS)  # 30
K_LATENT = 5  # number of true latent factors

# Factor names (for interpretability)
FACTOR_NAMES = [
    "protein_structure",   # structural integrity, chew, bite
    "fat_system",          # juiciness, mouthfeel, flavor release
    "binding_network",     # cohesion, water retention, texture
    "flavor_profile",      # taste, aroma, color
    "moisture_balance",    # water activity, shelf life, tenderness
]


def build_ground_truth_loading_matrix() -> np.ndarray:
    """Build the 'true' loading matrix A (k x d) mapping ingredients to factors.

    Each row is a latent factor. Each column is an ingredient.
    Values reflect approximate food-science relationships:
    - Nonzero where an ingredient meaningfully contributes to a factor
    - Magnitude reflects relative importance
    - Some ingredients load onto multiple factors (realistic cross-talk)

    Returns:
        A: np.ndarray of shape (K_LATENT, D_FULL) = (5, 30)
    """
    A = np.zeros((K_LATENT, D_FULL))

    # Factor 0: protein_structure
    # Proteins are primary; transglutaminase cross-links them
    A[0, 0] = 0.35   # pea protein isolate (dominant)
    A[0, 1] = 0.25   # soy protein concentrate
    A[0, 2] = 0.30   # wheat gluten (strong structure)
    A[0, 3] = 0.08   # chickpea flour (minor)
    A[0, 4] = 0.10   # rice protein (filler)
    A[0, 29] = 0.20  # transglutaminase (cross-linker, big effect)

    # Factor 1: fat_system
    # Fats drive juiciness; coconut oil melts at body temp (key for "bite")
    A[1, 5] = 0.40   # coconut oil (dominant, melting point matters)
    A[1, 6] = 0.20   # canola oil
    A[1, 7] = 0.25   # cocoa butter (solid fat, texture)
    A[1, 8] = 0.15   # sunflower oil
    A[1, 9] = 0.10   # shea butter
    # Cross-talk: methylcellulose traps fat
    A[1, 12] = 0.08  # methylcellulose (fat encapsulation)

    # Factor 2: binding_network
    # Starches and hydrocolloids create the gel matrix
    A[2, 10] = 0.25  # tapioca starch
    A[2, 11] = 0.20  # potato starch
    A[2, 12] = 0.35  # methylcellulose (dominant binder)
    A[2, 13] = 0.15  # xanthan gum
    A[2, 14] = 0.25  # konjac (synergy with xanthan)
    # Cross-talk: wheat gluten also binds
    A[2, 2] = 0.12   # wheat gluten (network formation)

    # Factor 3: flavor_profile
    # Flavor, color, and savory notes
    A[3, 15] = 0.20  # beet juice powder (color + earthy)
    A[3, 16] = 0.30  # yeast extract (umami, dominant flavor)
    A[3, 17] = 0.15  # smoked paprika
    A[3, 18] = 0.15  # onion powder
    A[3, 19] = 0.25  # natural flavor blend
    A[3, 24] = 0.12  # soy sauce (umami cross-talk)
    A[3, 25] = 0.08  # salt (flavor enhancer)

    # Factor 4: moisture_balance
    # Water activity, tenderness, shelf life
    A[4, 20] = 0.40  # water (dominant)
    A[4, 21] = 0.10  # apple cider vinegar
    A[4, 22] = 0.08  # lemon juice concentrate
    A[4, 23] = 0.12  # tomato paste (moisture + flavor)
    A[4, 24] = 0.10  # soy sauce
    # Cross-talk: starches absorb moisture
    A[4, 10] = -0.10  # tapioca starch (absorbs water)
    A[4, 11] = -0.08  # potato starch (absorbs water)

    return A


# ---------------------------------------------------------------------------
# 2. OBJECTIVE FUNCTIONS (operate in latent factor space)
# ---------------------------------------------------------------------------

def _hartmann5_on_factors(z: np.ndarray) -> float:
    """Modified Hartmann function for 5 latent factors.

    Adapted from the standard Hartmann6 to 5 dimensions.
    Has a known global optimum for validation.
    Returns NEGATIVE value (we maximize, Hartmann is typically minimized).
    """
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A_h = np.array([
        [10, 3, 17, 3.5, 8],
        [0.05, 10, 17, 0.1, 8],
        [3, 3.5, 1.7, 10, 17],
        [17, 8, 0.05, 10, 0.1],
    ])
    P = 1e-4 * np.array([
        [1312, 1696, 5569, 124, 8283],
        [2329, 4135, 8307, 3736, 1004],
        [2348, 1451, 3522, 2883, 3047],
        [4047, 8828, 8732, 5743, 1091],
    ])

    outer_sum = 0.0
    for i in range(4):
        inner_sum = 0.0
        for j in range(5):
            inner_sum += A_h[i, j] * (z[j] - P[i, j]) ** 2
        outer_sum += alpha[i] * np.exp(-inner_sum)

    return float(outer_sum)  # positive = better (we maximize)


def _ackley_on_factors(z: np.ndarray) -> float:
    """Ackley function on latent factors (5D). Highly multimodal.

    Returns negated value (higher = better for maximization).
    Global optimum at z = 0.5 (after shifting to [0,1] domain).
    """
    # Shift so optimum is at 0.5 in [0,1]^5
    x = 2 * (z - 0.5)  # map to [-1, 1]
    d = len(x)
    sum_sq = np.sum(x ** 2)
    sum_cos = np.sum(np.cos(2 * np.pi * x))
    val = -20 * np.exp(-0.2 * np.sqrt(sum_sq / d)) - np.exp(sum_cos / d) + 20 + np.e
    return -float(val)  # negate for maximization


def _synergy_on_factors(z: np.ndarray) -> float:
    """Custom 'synergy' function modeling realistic food interactions.

    Key properties:
    - Protein-fat interaction (factors 0 & 1): synergistic peak
    - Binding threshold (factor 2): cliff effect below 0.3
    - Flavor optimum (factor 3): target around 0.6
    - Moisture sweet spot (factor 4): quadratic with optimum at 0.55

    This is designed to be hard for vanilla BO but easy if you know the structure.
    """
    protein = z[0]
    fat = z[1]
    binding = z[2]
    flavor = z[3]
    moisture = z[4]

    # Protein-fat synergy: both need to be moderate-high
    synergy = 2.0 * protein * fat * np.exp(-0.5 * (protein - 0.6)**2 - 0.5 * (fat - 0.5)**2)

    # Binding cliff: below 0.3, texture collapses
    if binding < 0.3:
        binding_score = binding / 0.3 * 0.2  # harsh penalty
    else:
        binding_score = 0.2 + 0.8 * (1 - np.exp(-3 * (binding - 0.3)))

    # Flavor target: optimum around 0.6
    flavor_score = np.exp(-8 * (flavor - 0.6)**2)

    # Moisture sweet spot: optimum around 0.55
    moisture_score = 1.0 - 4.0 * (moisture - 0.55)**2
    moisture_score = max(0.0, moisture_score)

    return float(synergy + binding_score + flavor_score + moisture_score)


# ---------------------------------------------------------------------------
# 3. SYNTHETIC PROBLEM CLASS
# ---------------------------------------------------------------------------

@dataclass
class SyntheticFoodProblem:
    """A high-dimensional food formulation problem with known latent structure.

    The objective is: f(x) = g(A @ x_normalized)
    where x is in ingredient space (d=30), A is the loading matrix (5x30),
    and g is a known function in factor space.
    """
    name: str
    objective_fn: Callable[[np.ndarray], float]
    loading_matrix: np.ndarray  # (k, d)
    ingredient_bounds: np.ndarray  # (2, d) — [min; max]
    noise_std: float = 0.0

    @property
    def d(self) -> int:
        return self.loading_matrix.shape[1]

    @property
    def k(self) -> int:
        return self.loading_matrix.shape[0]

    def evaluate(self, x: np.ndarray, noisy: bool = True) -> float:
        """Evaluate the objective at ingredient vector x.

        Args:
            x: ingredient quantities, shape (d,)
            noisy: whether to add observation noise
        Returns:
            scalar objective value
        """
        # Normalize x to [0, 1] per ingredient
        bounds_min = self.ingredient_bounds[0]
        bounds_max = self.ingredient_bounds[1]
        rng = bounds_max - bounds_min
        rng[rng == 0] = 1.0
        x_norm = (x - bounds_min) / rng

        # Project to latent factors
        z = self.loading_matrix @ x_norm

        # Normalize z to [0, 1] per factor (based on possible range from A @ [0,1]^d)
        # For each factor, min = sum of negative coeffs, max = sum of positive coeffs
        z_min = np.minimum(self.loading_matrix, 0).sum(axis=1)
        z_max = np.maximum(self.loading_matrix, 0).sum(axis=1)
        z_rng = z_max - z_min
        z_rng[z_rng == 0] = 1.0
        z_norm = (z - z_min) / z_rng

        z_norm = np.clip(z_norm, 0, 1)

        val = self.objective_fn(z_norm)
        if noisy and self.noise_std > 0:
            val += np.random.normal(0, self.noise_std)
        return val

    def evaluate_batch(self, X: np.ndarray, noisy: bool = True) -> np.ndarray:
        """Evaluate at multiple points. X shape: (n, d)."""
        return np.array([self.evaluate(X[i], noisy=noisy) for i in range(len(X))])


def make_default_bounds() -> np.ndarray:
    """Realistic ingredient bounds in grams per 100g formulation."""
    mins = np.array([
        # Proteins (0-4)
        0, 0, 0, 0, 0,
        # Fats (5-9)
        0, 0, 0, 0, 0,
        # Starches & Binders (10-14)
        0, 0, 0, 0, 0,
        # Flavoring & Color (15-19)
        0, 0, 0, 0, 0,
        # Moisture & Acids (20-24)
        20, 0, 0, 0, 0,
        # Minerals & Minor (25-29)
        0, 0, 0, 0, 0,
    ], dtype=float)

    maxs = np.array([
        # Proteins (0-4): up to ~25g each
        25, 20, 15, 10, 10,
        # Fats (5-9): up to ~15g each
        15, 10, 8, 8, 5,
        # Starches & Binders (10-14): smaller amounts
        8, 8, 3, 1.5, 1.5,
        # Flavoring & Color (15-19): small amounts
        2, 3, 1, 2, 2,
        # Moisture & Acids (20-24)
        60, 2, 1, 3, 2,
        # Minerals & Minor (25-29): trace to small
        3, 0.5, 0.05, 0.01, 0.5,
    ], dtype=float)

    return np.stack([mins, maxs])  # (2, 30)


def create_problems(noise_std: float = 0.05) -> Dict[str, SyntheticFoodProblem]:
    """Create the benchmark problem suite."""
    A = build_ground_truth_loading_matrix()
    bounds = make_default_bounds()

    return {
        "hartmann5": SyntheticFoodProblem(
            name="PlantBurger-Hartmann",
            objective_fn=_hartmann5_on_factors,
            loading_matrix=A,
            ingredient_bounds=bounds,
            noise_std=noise_std,
        ),
        "ackley": SyntheticFoodProblem(
            name="PlantBurger-Ackley",
            objective_fn=_ackley_on_factors,
            loading_matrix=A,
            ingredient_bounds=bounds,
            noise_std=noise_std,
        ),
        "synergy": SyntheticFoodProblem(
            name="PlantBurger-Synergy",
            objective_fn=_synergy_on_factors,
            loading_matrix=A,
            ingredient_bounds=bounds,
            noise_std=noise_std,
        ),
    }


# ---------------------------------------------------------------------------
# 4. SUBSPACE QUALITY METRICS
# ---------------------------------------------------------------------------

def subspace_alignment(A_true: np.ndarray, A_proposed: np.ndarray) -> Dict[str, float]:
    """Measure how well a proposed subspace aligns with the true one.

    Args:
        A_true: ground-truth loading matrix, shape (k, d)
        A_proposed: proposed loading matrix, shape (k', d)

    Returns:
        dict with:
        - principal_angles: list of principal angles in degrees
        - mean_angle: mean principal angle (lower = better, 0 = perfect)
        - projection_error: Frobenius norm of (P_true - P_proposed) / sqrt(k)
          where P = A^T (A A^T)^{-1} A is the projection matrix
        - variance_captured: fraction of true subspace variance captured
    """
    # Compute orthonormal bases via SVD
    U_true, _, _ = np.linalg.svd(A_true, full_matrices=False)  # (k, k) x ... but we want column space
    # Actually, for A of shape (k, d), the row space is what matters
    # The subspace in R^d spanned by the rows of A
    U_true_T = np.linalg.svd(A_true.T, full_matrices=False)[0][:, :A_true.shape[0]]  # (d, k)
    U_prop_T = np.linalg.svd(A_proposed.T, full_matrices=False)[0][:, :A_proposed.shape[0]]  # (d, k')

    # Principal angles
    angles_rad = subspace_angles(U_true_T, U_prop_T)
    angles_deg = np.degrees(angles_rad)

    # Projection matrices
    P_true = U_true_T @ U_true_T.T
    P_prop = U_prop_T @ U_prop_T.T
    k = A_true.shape[0]
    proj_error = np.linalg.norm(P_true - P_prop, 'fro') / np.sqrt(k)

    # Variance captured: how much of A_true's row space is inside A_proposed's row space
    # = sum of cos^2(principal angles) / k
    variance_captured = float(np.mean(np.cos(angles_rad) ** 2))

    return {
        "principal_angles_deg": angles_deg.tolist(),
        "mean_angle_deg": float(np.mean(angles_deg)),
        "projection_error": float(proj_error),
        "variance_captured": variance_captured,
    }


# ---------------------------------------------------------------------------
# 5. BASELINE SUBSPACE METHODS
# ---------------------------------------------------------------------------

def random_subspace(d: int, k: int, seed: int = 0) -> np.ndarray:
    """ALEBO-style random linear embedding. Shape (k, d)."""
    rng = np.random.RandomState(seed)
    A = rng.randn(k, d)
    # Orthogonalize rows
    Q, _ = np.linalg.qr(A.T)
    return Q[:, :k].T  # (k, d)


def pca_subspace(X_data: np.ndarray, k: int) -> np.ndarray:
    """PCA on observed data. Requires prior evaluations. Shape (k, d)."""
    X_centered = X_data - X_data.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    return Vt[:k]  # (k, d) — top k principal components


def expert_subspace(d: int, expert_indices: List[int]) -> np.ndarray:
    """Simulated human expert: picks a subset of variables, fixes the rest.

    Returns an axis-aligned projection onto the selected variables.
    Shape (len(expert_indices), d).
    """
    k = len(expert_indices)
    A = np.zeros((k, d))
    for i, idx in enumerate(expert_indices):
        A[i, idx] = 1.0
    return A


def simulated_expert_selection(problem: SyntheticFoodProblem,
                                n_select: int = 5) -> List[int]:
    """Simulate a food scientist picking the most important ingredients.

    Heuristic: pick the ingredients with highest total absolute loading
    across factors — but with noise to simulate imperfect human judgment.
    A real expert would know ~70-80% of the important variables.
    """
    A = problem.loading_matrix
    importance = np.sum(np.abs(A), axis=0)  # (d,)

    # Add noise to simulate human imperfection (they might miss some)
    noise = np.random.randn(len(importance)) * 0.05
    noisy_importance = importance + noise

    # Pick top n_select
    return list(np.argsort(noisy_importance)[-n_select:])


# ---------------------------------------------------------------------------
# 6. BO IN SUBSPACE
# ---------------------------------------------------------------------------

def bo_in_subspace(
    problem: SyntheticFoodProblem,
    subspace_matrix: np.ndarray,
    budget: int = 25,
    n_init: int = 5,
    seed: int = 0,
    fixed_point: Optional[np.ndarray] = None,
) -> Dict:
    """Run BO in a given subspace and return optimization trace.

    The subspace_matrix A has shape (k, d). We optimize over z in R^k,
    then reconstruct x = A^+ z (pseudoinverse) projected onto feasible bounds.

    Args:
        problem: the synthetic food problem
        subspace_matrix: (k, d) projection matrix
        budget: total evaluation budget
        n_init: initial random points
        seed: random seed
        fixed_point: (d,) point to anchor the projection (midpoint of bounds if None)

    Returns:
        dict with keys: best_values (list), all_y (list), final_best (float)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    k, d = subspace_matrix.shape
    A = torch.tensor(subspace_matrix, dtype=torch.double)
    A_pinv = torch.linalg.pinv(A)  # (d, k)

    bounds_np = problem.ingredient_bounds  # (2, d)
    x_min = torch.tensor(bounds_np[0], dtype=torch.double)
    x_max = torch.tensor(bounds_np[1], dtype=torch.double)

    if fixed_point is None:
        x_anchor = (x_min + x_max) / 2  # midpoint
    else:
        x_anchor = torch.tensor(fixed_point, dtype=torch.double)

    # Determine z-space bounds by sampling corners
    # Conservative approach: use Sobol sampling to find feasible z range
    sobol = SobolEngine(dimension=d, scramble=True, seed=seed)
    x_samples = sobol.draw(4096).double()
    x_samples = x_min + x_samples * (x_max - x_min)
    z_samples = (A @ x_samples.T).T  # (4096, k)
    z_min = z_samples.min(dim=0).values
    z_max = z_samples.max(dim=0).values
    z_bounds = torch.stack([z_min, z_max])  # (2, k)

    def z_to_x(z: torch.Tensor) -> np.ndarray:
        """Map from subspace back to ingredient space."""
        x = x_anchor + A_pinv @ (z - A @ x_anchor)
        x = torch.clamp(x, x_min, x_max)
        return x.detach().numpy()

    # Storage
    all_z = []
    all_y = []
    best_values = []

    # Initial random samples in z-space
    sobol_z = SobolEngine(dimension=k, scramble=True, seed=seed)
    z_init = sobol_z.draw(n_init).double()
    z_init = z_min + z_init * (z_max - z_min)

    for i in range(n_init):
        z = z_init[i]
        x = z_to_x(z)
        y = problem.evaluate(x)
        all_z.append(z.numpy())
        all_y.append(y)
        best_values.append(max(all_y))

    # BO loop
    for step in range(budget - n_init):
        train_Z = torch.tensor(np.array(all_z), dtype=torch.double)
        train_Y = torch.tensor(all_y, dtype=torch.double).unsqueeze(-1)

        # Normalize to [0,1]
        train_Z_norm = normalize(train_Z, z_bounds)

        gp = SingleTaskGP(train_Z_norm, train_Y, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        try:
            fit_gpytorch_mll(mll)
        except Exception:
            pass  # if fitting fails, use current hyperparams

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))
        acq = qLogNoisyExpectedImprovement(
            model=gp,
            X_baseline=train_Z_norm,
            sampler=sampler,
        )

        candidate_norm, _ = optimize_acqf(
            acq_function=acq,
            bounds=torch.stack([torch.zeros(k), torch.ones(k)]).double(),
            q=1,
            num_restarts=10,
            raw_samples=512,
        )

        z_new = unnormalize(candidate_norm.squeeze(0), z_bounds)
        x_new = z_to_x(z_new)
        y_new = problem.evaluate(x_new)

        all_z.append(z_new.detach().numpy())
        all_y.append(y_new)
        best_values.append(max(all_y))

    return {
        "best_values": best_values,
        "all_y": all_y,
        "final_best": max(all_y),
    }


def bo_full_dimensional(
    problem: SyntheticFoodProblem,
    budget: int = 25,
    n_init: int = 5,
    seed: int = 0,
) -> Dict:
    """Vanilla BO in the full d-dimensional ingredient space (baseline)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    d = problem.d
    bounds_np = problem.ingredient_bounds
    bounds = torch.tensor(bounds_np, dtype=torch.double)

    all_x = []
    all_y = []
    best_values = []

    # Initial Sobol samples
    sobol = SobolEngine(dimension=d, scramble=True, seed=seed)
    x_init_norm = sobol.draw(n_init).double()
    x_init = unnormalize(x_init_norm, bounds)

    for i in range(n_init):
        x = x_init[i].numpy()
        y = problem.evaluate(x)
        all_x.append(x)
        all_y.append(y)
        best_values.append(max(all_y))

    # BO loop
    for step in range(budget - n_init):
        train_X = torch.tensor(np.array(all_x), dtype=torch.double)
        train_Y = torch.tensor(all_y, dtype=torch.double).unsqueeze(-1)
        train_X_norm = normalize(train_X, bounds)

        gp = SingleTaskGP(train_X_norm, train_Y, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        try:
            fit_gpytorch_mll(mll)
        except Exception:
            pass

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([256]))
        acq = qLogNoisyExpectedImprovement(
            model=gp,
            X_baseline=train_X_norm,
            sampler=sampler,
        )

        candidate_norm, _ = optimize_acqf(
            acq_function=acq,
            bounds=torch.stack([torch.zeros(d), torch.ones(d)]).double(),
            q=1,
            num_restarts=10,
            raw_samples=512,
        )

        x_new = unnormalize(candidate_norm.squeeze(0), bounds).detach().numpy()
        y_new = problem.evaluate(x_new)

        all_x.append(x_new)
        all_y.append(y_new)
        best_values.append(max(all_y))

    return {
        "best_values": best_values,
        "all_y": all_y,
        "final_best": max(all_y),
    }


# ---------------------------------------------------------------------------
# 7. BENCHMARK RUNNER
# ---------------------------------------------------------------------------

def run_benchmark(
    problem_name: str = "synergy",
    budget: int = 25,
    n_init: int = 5,
    n_seeds: int = 5,
    llm_subspace: Optional[np.ndarray] = None,
) -> Dict:
    """Run full benchmark comparison on a single problem.

    Args:
        problem_name: one of "hartmann5", "ackley", "synergy"
        budget: total evaluation budget per run
        n_init: initial random evaluations
        n_seeds: number of random seeds for statistical significance
        llm_subspace: (k, d) matrix from LLM. If None, uses ground-truth as oracle.

    Returns:
        dict of method_name -> {mean_trace, std_trace, mean_final, std_final, subspace_metrics}
    """
    problems = create_problems(noise_std=0.05)
    problem = problems[problem_name]
    A_true = problem.loading_matrix
    k = problem.k
    d = problem.d

    # If no LLM subspace provided, use ground truth as oracle upper bound
    if llm_subspace is None:
        llm_subspace = A_true

    methods = {}

    # 1. Oracle (ground-truth subspace)
    methods["oracle"] = {"matrix": A_true}

    # 2. LLM-proposed subspace
    methods["llm"] = {"matrix": llm_subspace}

    # 3. Random subspace (ALEBO-style)
    # Use multiple random seeds and average
    methods["random_alebo"] = {"matrix": None}  # generated per seed

    # 4. Expert-simulated
    methods["expert"] = {"matrix": None}  # generated per seed

    # 5. Full-dimensional BO
    methods["full_dim"] = {"matrix": None}  # special case

    results = {}

    for method_name in methods:
        print(f"  Running {method_name}...")
        traces = []

        for seed in range(n_seeds):
            if method_name == "full_dim":
                result = bo_full_dimensional(problem, budget=budget, n_init=n_init, seed=seed)
            elif method_name == "random_alebo":
                A_rand = random_subspace(d, k, seed=seed)
                result = bo_in_subspace(problem, A_rand, budget=budget, n_init=n_init, seed=seed)
            elif method_name == "expert":
                expert_idx = simulated_expert_selection(problem, n_select=k)
                A_exp = expert_subspace(d, expert_idx)
                result = bo_in_subspace(problem, A_exp, budget=budget, n_init=n_init, seed=seed)
            else:
                A = methods[method_name]["matrix"]
                result = bo_in_subspace(problem, A, budget=budget, n_init=n_init, seed=seed)

            traces.append(result["best_values"])

        traces = np.array(traces)  # (n_seeds, budget)
        mean_trace = traces.mean(axis=0)
        std_trace = traces.std(axis=0)

        # Subspace metrics
        if method_name == "full_dim":
            sub_metrics = {"note": "full-dimensional, no subspace"}
        elif method_name == "random_alebo":
            # Average metrics over seeds
            sub_metrics = {"note": "averaged over random seeds"}
        elif method_name == "expert":
            sub_metrics = {"note": "simulated expert variable selection"}
        else:
            A_m = methods[method_name]["matrix"]
            sub_metrics = subspace_alignment(A_true, A_m)

        results[method_name] = {
            "mean_trace": mean_trace.tolist(),
            "std_trace": std_trace.tolist(),
            "mean_final": float(mean_trace[-1]),
            "std_final": float(std_trace[-1]),
            "subspace_metrics": sub_metrics,
        }

    return results


def print_results(results: Dict):
    """Pretty-print benchmark results."""
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    # Final performance comparison
    print(f"\n{'Method':<20} {'Final Best (mean +/- std)':<30} {'Subspace Quality'}")
    print("-" * 70)

    for method, data in sorted(results.items(), key=lambda x: -x[1]["mean_final"]):
        perf = f"{data['mean_final']:.4f} +/- {data['std_final']:.4f}"

        sm = data["subspace_metrics"]
        if "mean_angle_deg" in sm:
            quality = f"angle={sm['mean_angle_deg']:.1f} deg, var_captured={sm['variance_captured']:.2%}"
        elif "note" in sm:
            quality = sm["note"]
        else:
            quality = ""

        print(f"{method:<20} {perf:<30} {quality}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run synthetic food BO benchmark")
    parser.add_argument("--problem", default="synergy", choices=["hartmann5", "ackley", "synergy"])
    parser.add_argument("--budget", type=int, default=25)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--n-init", type=int, default=5)
    args = parser.parse_args()

    print(f"Running benchmark: {args.problem} (budget={args.budget}, seeds={args.seeds})")
    results = run_benchmark(
        problem_name=args.problem,
        budget=args.budget,
        n_init=args.n_init,
        n_seeds=args.seeds,
    )
    print_results(results)
