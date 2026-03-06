"""
Implied volatility (IV): the volatility σ that makes the Black–Scholes price match the market price.

No closed form exists; we solve numerically (Newton–Raphson with vega, or Brent when vega is tiny).
"""

import warnings
import numpy as np
from scipy.optimize import brentq

from src.pricing.black_scholes import black_scholes_call, black_scholes_put, _validate_bs_inputs
from src.pricing.greeks import vega as vega_per_pct


def _raw_vega(S, K, T, r, sigma, q=0.0):
    """Vega per unit sigma (derivative of option price w.r.t. sigma). Used in Newton."""
    return float(vega_per_pct(S, K, T, r, sigma, q) * 100.0)


def implied_vol(market_price, S, K, T, r, option_type="call", q=0.0, method="newton", tol=1e-6, max_iter=100):
    """
    Implied volatility: σ such that BS(σ) equals market_price.

    Parameters
    ----------
    market_price : float
        Observed option price (must be > 0).
    S, K, T, r : float
        Spot, strike, time to maturity, risk-free rate.
    option_type : str, optional
        'call' or 'put'. Default 'call'.
    q : float, optional
        Continuous dividend yield. Default 0.0.
    method : str, optional
        'newton' or 'brent'. Default 'newton'.
    tol : float, optional
        Convergence tolerance for |BS(sigma) - market_price|. Default 1e-6.
    max_iter : int, optional
        Maximum Newton iterations. Default 100.

    Returns
    -------
    float
        Implied volatility, or np.nan if no convergence (with a warning).
    """
    if market_price <= 0:
        raise ValueError("market_price must be positive")
    S, K, T, r, q = float(S), float(K), float(T), float(r), float(q)
    _validate_bs_inputs(S, K, T, r, 0.25, q)  # sigma dummy for validation

    # Intrinsic (dividend-adjusted)
    if option_type == "call":
        intrinsic = max(0.0, S * np.exp(-q * T) - K * np.exp(-r * T))
    else:
        intrinsic = max(0.0, K * np.exp(-r * T) - S * np.exp(-q * T))
    if market_price <= intrinsic:
        return 0.0

    def price(sig):
        if option_type == "call":
            return black_scholes_call(S, K, T, r, sig, q)
        return black_scholes_put(S, K, T, r, sig, q)

    # Initial guess: sqrt(2 * |ln(S/K) + r*T| / T), clamped
    log_ratio = np.log(S / K) + r * T
    sigma_0 = np.sqrt(2.0 * np.abs(log_ratio) / T) if T > 1e-10 else 0.2
    sigma_0 = np.clip(sigma_0, 1e-4, 5.0)

    if method == "brent":
        return _implied_vol_brent(price, market_price, sigma_0, tol)

    # Newton
    sigma = sigma_0
    for _ in range(max_iter):
        p = price(sigma)
        err = p - market_price
        if np.abs(err) <= tol:
            return float(sigma)
        v = _raw_vega(S, K, T, r, sigma, q)
        if v < 1e-10:
            return _implied_vol_brent(price, market_price, sigma_0, tol)
        sigma = sigma - err / v
        if sigma <= 0:
            sigma = 1e-4
        if sigma > 10:
            sigma = 5.0
    warnings.warn("implied_vol did not converge", UserWarning)
    return np.nan


def _implied_vol_brent(price_func, market_price, sigma_0, tol):
    """Solve price_func(sigma) - market_price = 0 with brentq on [1e-6, 10]."""
    def objective(sig):
        return price_func(sig) - market_price
    try:
        return float(brentq(objective, 1e-6, 10.0, xtol=tol))
    except (ValueError, ZeroDivisionError):
        warnings.warn("implied_vol (Brent) did not converge", UserWarning)
        return np.nan


def implied_vol_vectorized(prices, S, K, T, r, option_types, q=0.0):
    """
    Implied volatility for multiple options. Loops over rows; returns 1D array.

    Parameters
    ----------
    prices : array-like
        Option prices (one per row).
    S, K, T, r : float or array-like
        If array-like, same length as prices.
    option_types : array-like of str
        'call' or 'put' per row.
    q : float, optional
        Dividend yield. Default 0.0.

    Returns
    -------
    np.ndarray
        1D array of implied vols; np.nan where solver fails.
    """
    prices = np.atleast_1d(prices)
    n = len(prices)
    S = np.broadcast_to(np.atleast_1d(S), n)
    K = np.broadcast_to(np.atleast_1d(K), n)
    T = np.broadcast_to(np.atleast_1d(T), n)
    r = np.broadcast_to(np.atleast_1d(r), n)
    option_types = np.broadcast_to(np.atleast_1d(option_types), n)
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        try:
            out[i] = implied_vol(
                float(prices[i]), float(S[i]), float(K[i]), float(T[i]), float(r[i]),
                option_type=str(option_types[i]), q=float(q)
            )
        except (ValueError, Exception):
            out[i] = np.nan
    return out
