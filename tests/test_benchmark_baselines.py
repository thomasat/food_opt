"""Tests for high-dimensional BO baselines: REMBO, TuRBO, SAASBO."""

import numpy as np
import pytest

from benchmark import (
    create_problems,
    bo_rembo,
    bo_turbo,
    bo_saasbo,
    run_benchmark,
    ALL_METHODS,
)

BUDGET = 8
N_INIT = 5


@pytest.fixture
def problem():
    return create_problems(noise_std=0.05)["synergy"]


# ------------------------------------------------------------------ #
#  REMBO
# ------------------------------------------------------------------ #

class TestREMBO:
    def test_returns_correct_keys(self, problem):
        result = bo_rembo(problem, budget=BUDGET, n_init=N_INIT, seed=0)
        assert "best_values" in result
        assert "all_y" in result
        assert "final_best" in result

    def test_trace_length_matches_budget(self, problem):
        result = bo_rembo(problem, budget=BUDGET, n_init=N_INIT, seed=0)
        assert len(result["best_values"]) == BUDGET
        assert len(result["all_y"]) == BUDGET

    def test_best_values_monotonically_increase(self, problem):
        result = bo_rembo(problem, budget=BUDGET, n_init=N_INIT, seed=0)
        bv = result["best_values"]
        for i in range(1, len(bv)):
            assert bv[i] >= bv[i - 1]

    def test_custom_target_dim(self, problem):
        result = bo_rembo(problem, budget=BUDGET, n_init=N_INIT, seed=0, target_dim=3)
        assert len(result["best_values"]) == BUDGET

    def test_different_seeds_give_different_results(self, problem):
        r1 = bo_rembo(problem, budget=BUDGET, n_init=N_INIT, seed=0)
        r2 = bo_rembo(problem, budget=BUDGET, n_init=N_INIT, seed=42)
        # Traces should differ (different random embeddings)
        assert r1["all_y"] != r2["all_y"]


# ------------------------------------------------------------------ #
#  TuRBO
# ------------------------------------------------------------------ #

class TestTuRBO:
    def test_returns_correct_keys(self, problem):
        result = bo_turbo(problem, budget=BUDGET, n_init=N_INIT, seed=0)
        assert "best_values" in result
        assert "all_y" in result
        assert "final_best" in result

    def test_trace_length_matches_budget(self, problem):
        result = bo_turbo(problem, budget=BUDGET, n_init=N_INIT, seed=0)
        assert len(result["best_values"]) == BUDGET
        assert len(result["all_y"]) == BUDGET

    def test_best_values_monotonically_increase(self, problem):
        result = bo_turbo(problem, budget=BUDGET, n_init=N_INIT, seed=0)
        bv = result["best_values"]
        for i in range(1, len(bv)):
            assert bv[i] >= bv[i - 1]

    def test_different_seeds_give_different_results(self, problem):
        r1 = bo_turbo(problem, budget=BUDGET, n_init=N_INIT, seed=0)
        r2 = bo_turbo(problem, budget=BUDGET, n_init=N_INIT, seed=42)
        assert r1["all_y"] != r2["all_y"]


# ------------------------------------------------------------------ #
#  SAASBO
# ------------------------------------------------------------------ #

class TestSAASBO:
    def test_returns_correct_keys(self, problem):
        result = bo_saasbo(problem, budget=BUDGET, n_init=N_INIT, seed=0,
                           warmup_steps=32, num_samples=16)
        assert "best_values" in result
        assert "all_y" in result
        assert "final_best" in result

    def test_trace_length_matches_budget(self, problem):
        result = bo_saasbo(problem, budget=BUDGET, n_init=N_INIT, seed=0,
                           warmup_steps=32, num_samples=16)
        assert len(result["best_values"]) == BUDGET
        assert len(result["all_y"]) == BUDGET

    def test_best_values_monotonically_increase(self, problem):
        result = bo_saasbo(problem, budget=BUDGET, n_init=N_INIT, seed=0,
                           warmup_steps=32, num_samples=16)
        bv = result["best_values"]
        for i in range(1, len(bv)):
            assert bv[i] >= bv[i - 1]


# ------------------------------------------------------------------ #
#  Benchmark runner integration
# ------------------------------------------------------------------ #

class TestBenchmarkIntegration:
    def test_methods_list_includes_new_baselines(self):
        assert "rembo" in ALL_METHODS
        assert "turbo" in ALL_METHODS
        assert "saasbo" in ALL_METHODS

    def test_run_benchmark_with_method_filter(self):
        results = run_benchmark(
            problem_name="synergy",
            budget=BUDGET,
            n_init=N_INIT,
            n_seeds=1,
            methods_to_run=["rembo", "turbo"],
        )
        assert "rembo" in results
        assert "turbo" in results
        assert "oracle" not in results
        assert "full_dim" not in results

    def test_run_benchmark_single_method(self):
        results = run_benchmark(
            problem_name="synergy",
            budget=BUDGET,
            n_init=N_INIT,
            n_seeds=1,
            methods_to_run=["rembo"],
        )
        assert len(results) == 1
        assert "rembo" in results
        assert len(results["rembo"]["mean_trace"]) == BUDGET

    def test_run_benchmark_none_runs_all(self):
        results = run_benchmark(
            problem_name="synergy",
            budget=BUDGET,
            n_init=N_INIT,
            n_seeds=1,
            methods_to_run=None,
        )
        # Should include all methods
        for m in ALL_METHODS:
            assert m in results
