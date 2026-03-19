"""Tests for the theoretical analysis module."""

import numpy as np
import pytest

from theory import (
    information_gain_se,
    misalignment_penalty,
    practical_simple_regret,
    regret_bound,
    simple_regret_bound,
)


class TestInformationGain:
    def test_positive(self):
        assert information_gain_se(d=5, T=100) > 0

    def test_increases_with_dimension(self):
        g5 = information_gain_se(d=5, T=100)
        g20 = information_gain_se(d=20, T=100)
        assert g20 > g5

    def test_increases_with_T(self):
        g10 = information_gain_se(d=5, T=10)
        g100 = information_gain_se(d=5, T=100)
        assert g100 > g10

    def test_lengthscale_effect(self):
        g_short = information_gain_se(d=5, T=100, lengthscale=0.1)
        g_long = information_gain_se(d=5, T=100, lengthscale=1.0)
        assert g_short > g_long  # shorter lengthscale => more info needed


class TestPracticalSimpleRegret:
    def test_shape(self):
        result = practical_simple_regret(d=5, T=30)
        assert result.shape == (30,)

    def test_decreasing(self):
        result = practical_simple_regret(d=5, T=30)
        # Regret should decrease over time
        assert result[0] > result[-1]

    def test_higher_dim_higher_regret(self):
        r5 = practical_simple_regret(d=5, T=30)
        r20 = practical_simple_regret(d=20, T=30)
        # Higher dimension -> worse final regret
        assert r20[-1] > r5[-1]

    def test_all_positive(self):
        result = practical_simple_regret(d=10, T=50)
        assert np.all(result > 0)


class TestRegretBound:
    def test_shape(self):
        result = regret_bound(d=5, T=30)
        assert result.shape == (30,)

    def test_cumulative_increases(self):
        result = regret_bound(d=5, T=30)
        # Cumulative regret should generally increase
        assert result[-1] > result[0]

    def test_higher_dim_higher_bound(self):
        r5 = regret_bound(d=5, T=30)
        r20 = regret_bound(d=20, T=30)
        assert r20[-1] > r5[-1]

    def test_custom_beta(self):
        result = regret_bound(d=5, T=30, beta_T=2.0)
        assert result.shape == (30,)
        assert result[-1] > 0


class TestSimpleRegretBound:
    def test_shape(self):
        result = simple_regret_bound(d=5, T=30)
        assert result.shape == (30,)

    def test_positive(self):
        result = simple_regret_bound(d=5, T=30)
        assert np.all(result > 0)


class TestMisalignmentPenalty:
    def test_constant_penalty(self):
        result = misalignment_penalty(epsilon=0.1, L=1.0, T=30)
        assert result.shape == (30,)
        assert np.all(result == pytest.approx(0.1))

    def test_scales_with_epsilon(self):
        r1 = misalignment_penalty(epsilon=0.1, L=1.0, T=10)
        r2 = misalignment_penalty(epsilon=0.2, L=1.0, T=10)
        assert np.all(r2 > r1)

    def test_scales_with_lipschitz(self):
        r1 = misalignment_penalty(epsilon=0.1, L=1.0, T=10)
        r2 = misalignment_penalty(epsilon=0.1, L=2.0, T=10)
        assert np.all(r2 > r1)
