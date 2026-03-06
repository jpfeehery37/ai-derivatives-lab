"""
Binomial (CRR) option pricing. As n increases, the tree converges to GBM and price to Black–Scholes.
"""

import numpy as np
import pandas as pd

from src.pricing.black_scholes import black_scholes_call, _validate_bs_inputs


def _validate_binomial_inputs(S, K, T, r, sigma, q=0.0):
    """Raise ValueError if any input is invalid (mirror Black–Scholes)."""
    S = np.atleast_1d(S)
    K = np.atleast_1d(K)
    T = np.atleast_1d(T)
    sigma = np.atleast_1d(sigma)
    q = np.atleast_1d(q)
    if np.any(S <= 0):
        raise ValueError("S (stock price) must be positive")
    if np.any(K <= 0):
        raise ValueError("K (strike price) must be positive")
    if np.any(T <= 0):
        raise ValueError("T (time to maturity) must be positive")
    if np.any(sigma <= 0):
        raise ValueError("sigma (volatility) must be positive")
    if np.any(q < 0):
        raise ValueError("q (dividend yield) must be non-negative")


def binomial_price(S, K, T, r, sigma, n=100, option_type="call", exercise="european", q=0.0):
    """
    Binomial (CRR) option price. Scalar S; returns scalar.

    As n increases, the tree converges to GBM and the price to Black–Scholes.
    American exercise: at each node take max(continuation value, early exercise).

    Parameters
    ----------
    S, K, T, r, sigma : float
        Spot, strike, time to maturity, rate, volatility.
    n : int, optional
        Number of steps. Default 100.
    option_type : str, optional
        'call' or 'put'. Default 'call'.
    exercise : str, optional
        'european' or 'american'. Default 'european'.
    q : float, optional
        Continuous dividend yield. Default 0.0.
    """
    _validate_binomial_inputs(S, K, T, r, sigma, q)
    S, K, T, r, sigma, q = float(S), float(K), float(T), float(r), float(sigma), float(q)
    if n < 1:
        raise ValueError("n must be at least 1")

    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    discount = np.exp(-r * dt)

    # Terminal stock prices: S * u^j * d^(n-j), j = 0..n
    j = np.arange(n + 1)
    terminal_S = S * (u ** j) * (d ** (n - j))
    if option_type == "call":
        terminal_value = np.maximum(terminal_S - K, 0.0)
    else:
        terminal_value = np.maximum(K - terminal_S, 0.0)

    # Backward induction
    value = terminal_value.copy()
    for step in range(n - 1, -1, -1):
        value = discount * (p * value[1 : step + 2] + (1 - p) * value[0 : step + 1])
        if exercise == "american":
            # Current stock levels at this step
            j_step = np.arange(step + 1)
            S_step = S * (u ** j_step) * (d ** (step - j_step))
            if option_type == "call":
                intrinsic = np.maximum(S_step - K, 0.0)
            else:
                intrinsic = np.maximum(K - S_step, 0.0)
            value = np.maximum(value, intrinsic)
    return float(value[0])


def convergence_to_bs(S, K, T, r, sigma, steps=None):
    """
    Compare binomial European call price to Black–Scholes for several step counts.

    Returns a DataFrame with columns: n, binomial_price, bs_price, error.
    """
    if steps is None:
        steps = [10, 25, 50, 100, 200, 500]
    _validate_binomial_inputs(S, K, T, r, sigma)
    S, K, T, r, sigma = float(S), float(K), float(T), float(r), float(sigma)
    bs = black_scholes_call(S, K, T, r, sigma, 0.0)
    rows = []
    for n in steps:
        bn = binomial_price(S, K, T, r, sigma, n=n, option_type="call", exercise="european", q=0.0)
        rows.append({"n": n, "binomial_price": bn, "bs_price": bs, "error": abs(bn - bs)})
    return pd.DataFrame(rows)
