"""Tests for the expert simulation on Hartmann6 in 100-D."""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Hartmann6 function tests
# ---------------------------------------------------------------------------

class TestHartmann6:
    def test_global_optimum(self):
        from experiments.run_expert_simulation import hartmann6, HARTMANN6_GLOBAL_MAX
        x_opt = np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])
        val = hartmann6(x_opt)
        assert abs(val - HARTMANN6_GLOBAL_MAX) < 0.01

    def test_domain_bounds(self):
        """Function should return finite values at corners."""
        from experiments.run_expert_simulation import hartmann6
        for _ in range(10):
            x = np.random.rand(6)
            val = hartmann6(x)
            assert np.isfinite(val)
            assert val >= 0  # Hartmann6 (negated for min) is non-negative

    def test_embedded_ignores_irrelevant(self):
        """evaluate_embedded should give same value regardless of dims 6..99."""
        from experiments.run_expert_simulation import evaluate_embedded, D_TOTAL
        x_rel = np.random.rand(6)
        x1 = np.zeros(D_TOTAL)
        x1[:6] = x_rel
        x2 = np.random.rand(D_TOTAL)
        x2[:6] = x_rel
        assert evaluate_embedded(x1) == evaluate_embedded(x2)


# ---------------------------------------------------------------------------
# Rastrigin function tests
# ---------------------------------------------------------------------------

class TestRastrigin:
    def test_global_optimum(self):
        """Rastrigin global min is 0 at origin; negated global max = 0.0 at x=0.5."""
        from experiments.run_expert_simulation import _rastrigin6, RASTRIGIN_GLOBAL_MAX
        x_opt = np.full(6, 0.5)   # maps to origin in [-5.12, 5.12]
        val = _rastrigin6(x_opt)
        assert abs(val - RASTRIGIN_GLOBAL_MAX) < 1e-10

    def test_domain_bounds(self):
        """Function should return finite values in [0,1]^6."""
        from experiments.run_expert_simulation import _rastrigin6
        for _ in range(10):
            x = np.random.rand(6)
            val = _rastrigin6(x)
            assert np.isfinite(val)
            assert val <= 0  # negated Rastrigin is non-positive

    def test_embedded_ignores_irrelevant(self):
        """evaluate_embedded_rastrigin should give same value regardless of dims 6..99."""
        from experiments.run_expert_simulation import evaluate_embedded_rastrigin, D_TOTAL
        x_rel = np.random.rand(6)
        x1 = np.zeros(D_TOTAL)
        x1[:6] = x_rel
        x2 = np.random.rand(D_TOTAL)
        x2[:6] = x_rel
        assert evaluate_embedded_rastrigin(x1) == evaluate_embedded_rastrigin(x2)


# ---------------------------------------------------------------------------
# Levy function tests
# ---------------------------------------------------------------------------

class TestLevy:
    def test_global_optimum(self):
        """Levy global min is 0 at x=1; in [0,1]^6 that maps to x_unit=0.6."""
        from experiments.run_expert_simulation import _levy6, LEVY_GLOBAL_MAX
        x_opt = np.full(6, 0.6)    # maps to x=1.0 in [-5, 5]
        val = _levy6(x_opt)
        assert abs(val - LEVY_GLOBAL_MAX) < 0.01

    def test_domain_bounds(self):
        """Function should return finite values in [0,1]^6."""
        from experiments.run_expert_simulation import _levy6
        for _ in range(10):
            x = np.random.rand(6)
            val = _levy6(x)
            assert np.isfinite(val)
            assert val <= 0  # negated Levy is non-positive

    def test_embedded_ignores_irrelevant(self):
        """evaluate_embedded_levy should give same value regardless of dims 6..99."""
        from experiments.run_expert_simulation import evaluate_embedded_levy, D_TOTAL
        x_rel = np.random.rand(6)
        x1 = np.zeros(D_TOTAL)
        x1[:6] = x_rel
        x2 = np.random.rand(D_TOTAL)
        x2[:6] = x_rel
        assert evaluate_embedded_levy(x1) == evaluate_embedded_levy(x2)


# ---------------------------------------------------------------------------
# Rosenbrock function tests
# ---------------------------------------------------------------------------

class TestRosenbrock:
    def test_global_optimum(self):
        """Rosenbrock global min is 0 at x=1; in [0,1]^6 that maps to x_unit=0.4."""
        from experiments.run_expert_simulation import _rosenbrock6, ROSENBROCK_GLOBAL_MAX
        x_opt = np.full(6, 0.4)   # maps to x=1.0 in [-5, 10]
        val = _rosenbrock6(x_opt)
        assert abs(val - ROSENBROCK_GLOBAL_MAX) < 0.01

    def test_domain_bounds(self):
        """Function should return finite values in [0,1]^6."""
        from experiments.run_expert_simulation import _rosenbrock6
        for _ in range(10):
            x = np.random.rand(6)
            val = _rosenbrock6(x)
            assert np.isfinite(val)
            assert val <= 0  # negated Rosenbrock is non-positive

    def test_embedded_ignores_irrelevant(self):
        """evaluate_embedded_rosenbrock should give same value regardless of dims 6..99."""
        from experiments.run_expert_simulation import evaluate_embedded_rosenbrock, D_TOTAL
        x_rel = np.random.rand(6)
        x1 = np.zeros(D_TOTAL)
        x1[:6] = x_rel
        x2 = np.random.rand(D_TOTAL)
        x2[:6] = x_rel
        assert evaluate_embedded_rosenbrock(x1) == evaluate_embedded_rosenbrock(x2)


# ---------------------------------------------------------------------------
# Styblinski-Tang function tests
# ---------------------------------------------------------------------------

class TestStyblinskiTang:
    def test_global_optimum(self):
        """Styblinski-Tang global min ≈ -39.16599*d at x≈-2.903534; negated max ≈ 234.996."""
        from experiments.run_expert_simulation import _styblinski_tang6, STYBLINSKI_TANG_GLOBAL_MAX
        # x_i = -2.903534 maps to x_unit = (-2.903534 + 5) / 10 = 0.2096466
        x_opt = np.full(6, 0.2096466)
        val = _styblinski_tang6(x_opt)
        assert abs(val - STYBLINSKI_TANG_GLOBAL_MAX) < 0.1

    def test_domain_bounds(self):
        """Function should return finite values in [0,1]^6."""
        from experiments.run_expert_simulation import _styblinski_tang6
        for _ in range(10):
            x = np.random.rand(6)
            val = _styblinski_tang6(x)
            assert np.isfinite(val)

    def test_embedded_ignores_irrelevant(self):
        """evaluate_embedded_styblinski_tang should give same value regardless of dims 6..99."""
        from experiments.run_expert_simulation import evaluate_embedded_styblinski_tang, D_TOTAL
        x_rel = np.random.rand(6)
        x1 = np.zeros(D_TOTAL)
        x1[:6] = x_rel
        x2 = np.random.rand(D_TOTAL)
        x2[:6] = x_rel
        assert evaluate_embedded_styblinski_tang(x1) == evaluate_embedded_styblinski_tang(x2)


# ---------------------------------------------------------------------------
# Expert variable selection tests
# ---------------------------------------------------------------------------

class TestExpertUpdate:
    def test_monotonic_growth(self):
        """Active set should never shrink."""
        from experiments.run_expert_simulation import (
            expert_update_variables, EXPERT_CONFIGS,
        )
        rng = np.random.RandomState(0)
        cfg = EXPERT_CONFIGS["expert_good"]
        v = set()
        for _ in range(50):
            v_new = expert_update_variables(v, cfg, rng)
            assert v_new >= v  # superset
            v = v_new

    def test_good_expert_finds_relevant_fast(self):
        """Good expert (p=0.5) should find most relevant vars in ~10 iters."""
        from experiments.run_expert_simulation import (
            expert_update_variables, EXPERT_CONFIGS, RELEVANT_SET,
        )
        rng = np.random.RandomState(42)
        cfg = EXPERT_CONFIGS["expert_good"]
        v = set()
        for _ in range(10):
            v = expert_update_variables(v, cfg, rng)
        assert len(RELEVANT_SET & v) >= 4  # should find at least 4 of 6

    def test_poor_expert_adds_irrelevant(self):
        """Poor expert (q=0.08) should accumulate many irrelevant vars."""
        from experiments.run_expert_simulation import (
            expert_update_variables, EXPERT_CONFIGS, IRRELEVANT_SET,
        )
        rng = np.random.RandomState(0)
        cfg = EXPERT_CONFIGS["expert_poor"]
        v = set()
        for _ in range(20):
            v = expert_update_variables(v, cfg, rng)
        n_irr = len(IRRELEVANT_SET & v)
        # With q=0.08 and 94 irrelevant vars, expect ~7-8 per step early on
        assert n_irr > 5, f"Expected >5 irrelevant vars, got {n_irr}"

    def test_p1_q0_finds_only_relevant(self):
        """With p=1, q=0, expert should select exactly the relevant set."""
        from experiments.run_expert_simulation import (
            expert_update_variables, ExpertConfig, RELEVANT_SET,
        )
        rng = np.random.RandomState(0)
        cfg = ExpertConfig("perfect", p=1.0, q=0.0)
        v = set()
        v = expert_update_variables(v, cfg, rng)
        assert v == RELEVANT_SET


# ---------------------------------------------------------------------------
# Integration: short run
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_oracle_short_run(self):
        from experiments.run_expert_simulation import run_expert_bo, EXPERT_CONFIGS
        r = run_expert_bo(budget=6, n_init=3, seed=0, noise_std=0.0,
                          config=EXPERT_CONFIGS["oracle"])
        assert len(r["best_values"]) == 6
        assert r["best_values"] == sorted(r["best_values"])  # monotonically non-decreasing
        assert r["final_best"] == r["best_values"][-1]
        # Oracle should select exactly 6 relevant vars on first iteration
        assert r["n_relevant_selected"][0] == 6

    def test_expert_short_run(self):
        from experiments.run_expert_simulation import run_expert_bo, EXPERT_CONFIGS
        r = run_expert_bo(budget=6, n_init=3, seed=0, noise_std=0.0,
                          config=EXPERT_CONFIGS["expert_good"])
        assert len(r["best_values"]) == 6
        assert len(r["active_set_sizes"]) == 6
        assert r["best_values"] == sorted(r["best_values"])
