"""
Put–call parity: C - P = S*exp(-q*T) - K*exp(-r*T).

Violations usually indicate data or quote errors rather than arbitrage;
this module provides the residual and a simple parity checker.
"""

import numpy as np


def parity_residual(C, P, S, K, T, r, q=0.0):
    """
    Put–call parity residual: (C - P) - (S*exp(-q*T) - K*exp(-r*T)).

    Under no-arbitrage, the residual is zero. Vectorized over arrays.

    Parameters
    ----------
    C, P : float or array-like
        Call and put prices (same expiry and strike).
    S, K, T, r, q : float or array-like
        Spot, strike, time to maturity, rate, dividend yield.
    """
    C = np.asarray(C, dtype=float)
    P = np.asarray(P, dtype=float)
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    r = np.asarray(r, dtype=float)
    q = np.asarray(q, dtype=float)
    rhs = S * np.exp(-q * T) - K * np.exp(-r * T)
    return (C - P) - rhs


def check_put_call_parity(C, P, S, K, T, r, q=0.0, tol=0.05):
    """
    Check put–call parity and return residual, pass/fail, and forward info.

    Parameters
    ----------
    C, P, S, K, T, r, q : float or array-like
        As in parity_residual.
    tol : float, optional
        Tolerance for |residual| to consider parity satisfied. Default 0.05.

    Returns
    -------
    dict
        residual : parity residual (C - P) - (S*e^{-q*T} - K*e^{-r*T}).
        passes : bool or array of bool; True where |residual| < tol.
        implied_forward : K + (C - P)*exp(r*T) (forward implied by C, P, K, r).
        theoretical_forward : S*exp((r - q)*T).
    """
    res = parity_residual(C, P, S, K, T, r, q)
    C = np.asarray(C, dtype=float)
    P = np.asarray(P, dtype=float)
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    r = np.asarray(r, dtype=float)
    q = np.asarray(q, dtype=float)
    implied_fwd = K + (C - P) * np.exp(r * T)
    theoretical_fwd = S * np.exp((r - q) * T)
    passes = np.abs(res) < tol
    return {
        "residual": res,
        "passes": passes,
        "implied_forward": implied_fwd,
        "theoretical_forward": theoretical_fwd,
    }
