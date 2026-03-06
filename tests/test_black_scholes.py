"""
Tests for Black–Scholes pricing (European options).

- Put–call parity: C - P = S - K*exp(-r*T)
- Call price bounds: max(0, S - K*exp(-r*T)) <= C <= S
- Put price bounds: max(0, K*exp(-r*T) - S) <= P <= K*exp(-r*T)
"""

import numpy as np
import pytest

from src.pricing.black_scholes import black_scholes_call, black_scholes_put


# -----------------------------------------------------------------------------
# Put–call parity
# -----------------------------------------------------------------------------
# In plain English: a long call plus a short put (same K, T) is equivalent to
# a forward: you effectively agree to buy the stock at K at time T. So the
# *value* of that combo today is the present value of (S_T - K), i.e.
#   C - P = S - K*exp(-r*T).
# So: C = P + (S - K*exp(-r*T)). We test that our formulas satisfy this.
# -----------------------------------------------------------------------------


def test_put_call_parity_scalar():
    """Put–call parity holds for a single set of parameters."""
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    C = black_scholes_call(S, K, T, r, sigma)
    P = black_scholes_put(S, K, T, r, sigma)
    pv_strike = K * np.exp(-r * T)
    # C - P should equal S - K*exp(-r*T)
    assert np.isclose(C - P, S - pv_strike, rtol=1e-9, atol=1e-12)


def test_put_call_parity_multiple_spots():
    """Put–call parity holds for many stock prices at once."""
    S = np.array([80.0, 100.0, 120.0])
    K, T, r, sigma = 100.0, 0.5, 0.02, 0.25
    C = black_scholes_call(S, K, T, r, sigma)
    P = black_scholes_put(S, K, T, r, sigma)
    pv_strike = K * np.exp(-r * T)
    np.testing.assert_allclose(C - P, S - pv_strike, rtol=1e-9, atol=1e-12)


# -----------------------------------------------------------------------------
# Call price bounds
# -----------------------------------------------------------------------------
# In plain English: the European call is worth at least the intrinsic value
# of a forward, max(0, S - K*exp(-r*T)), and at most S (you could just buy
# the stock). So:  max(0, S - K*exp(-r*T)) <= C <= S.
# -----------------------------------------------------------------------------


def test_call_price_lower_bound():
    """Call price >= max(0, S - K*exp(-r*T))."""
    S = np.array([85.0, 100.0, 115.0])
    K, T, r, sigma = 100.0, 1.0, 0.03, 0.2
    C = black_scholes_call(S, K, T, r, sigma)
    pv_strike = K * np.exp(-r * T)
    lower = np.maximum(0.0, S - pv_strike)
    assert np.all(C >= lower - 1e-10)  # small tolerance for float


def test_call_price_upper_bound():
    """Call price <= S (cannot exceed stock price)."""
    S = np.array([90.0, 100.0, 110.0])
    K, T, r, sigma = 100.0, 0.25, 0.05, 0.3
    C = black_scholes_call(S, K, T, r, sigma)
    assert np.all(C <= S + 1e-10)


# -----------------------------------------------------------------------------
# Put price bounds
# -----------------------------------------------------------------------------
# In plain English: the European put is worth at least the intrinsic value
# of (K*exp(-r*T) - S) when that is positive, and at most K*exp(-r*T) (the
# most you can get at expiry is K). So:  max(0, K*exp(-r*T) - S) <= P <= K*exp(-r*T).
# -----------------------------------------------------------------------------


def test_put_price_lower_bound():
    """Put price >= max(0, K*exp(-r*T) - S)."""
    S = np.array([85.0, 100.0, 115.0])
    K, T, r, sigma = 100.0, 1.0, 0.03, 0.2
    P = black_scholes_put(S, K, T, r, sigma)
    pv_strike = K * np.exp(-r * T)
    lower = np.maximum(0.0, pv_strike - S)
    assert np.all(P >= lower - 1e-10)


def test_put_price_upper_bound():
    """Put price <= K*exp(-r*T) (max payoff at expiry is K)."""
    S = np.array([90.0, 100.0, 110.0])
    K, T, r, sigma = 100.0, 0.25, 0.05, 0.3
    P = black_scholes_put(S, K, T, r, sigma)
    pv_strike = K * np.exp(-r * T)
    assert np.all(P <= pv_strike + 1e-10)
