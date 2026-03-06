"""Shared pytest fixtures for pricing tests."""

import pytest


@pytest.fixture
def standard_params():
    """Standard Black–Scholes parameters for tests (S=100, K=100, T=1, r=0.05, sigma=0.2, q=0)."""
    return {
        "S": 100.0,
        "K": 100.0,
        "T": 1.0,
        "r": 0.05,
        "sigma": 0.2,
        "q": 0.0,
    }
