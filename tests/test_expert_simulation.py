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
        from experiments.run_expert_simulation import run_oracle_bo
        r = run_oracle_bo(budget=6, n_init=3, seed=0, noise_std=0.0)
        assert len(r["best_values"]) == 6
        assert r["best_values"] == sorted(r["best_values"])  # monotonically non-decreasing
        assert r["final_best"] == r["best_values"][-1]

    def test_expert_short_run(self):
        from experiments.run_expert_simulation import run_expert_bo, EXPERT_CONFIGS
        r = run_expert_bo(budget=6, n_init=3, seed=0, noise_std=0.0,
                          config=EXPERT_CONFIGS["expert_good"])
        assert len(r["best_values"]) == 6
        assert len(r["active_set_sizes"]) == 6
        assert r["best_values"] == sorted(r["best_values"])
