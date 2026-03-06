"""
Black–Scholes pricing for European options.

These functions give the option value *today* (present value under the model),
not the payoff at expiration. They assume European exercise (no early exercise),
no dividends, constant risk-free rate and volatility, and lognormal stock price.
"""

import numpy as np
from scipy.stats import norm


def _validate_bs_inputs(S, K, T, r, sigma):
    """Raise ValueError if any input is invalid for Black–Scholes (e.g. non-positive)."""
    S, K, T, sigma = np.atleast_1d(S), np.atleast_1d(K), np.atleast_1d(T), np.atleast_1d(sigma)
    if np.any(S <= 0):
        raise ValueError("S (stock price) must be positive")
    if np.any(K <= 0):
        raise ValueError("K (strike price) must be positive")
    if np.any(T <= 0):
        raise ValueError("T (time to maturity) must be positive")
    if np.any(sigma <= 0):
        raise ValueError("sigma (volatility) must be positive")


def d1(S, K, T, r, sigma):
    """
    Black–Scholes d1 term.

    Used in the formula for the call/put price. With d2, it encodes the
    risk-neutral probability that the option finishes in the money.

    Parameters
    ----------
    S : float or array-like
        Current stock price.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Risk-free interest rate (annual, continuous compounding).
    sigma : float
        Volatility (annual standard deviation of log-returns).
    """
    S = np.array(S, dtype=float)
    numerator = np.log(S / K) + (r + 0.5 * sigma**2) * T
    denominator = sigma * np.sqrt(T)
    return numerator / denominator


def d2(S, K, T, r, sigma):
    """
    Black–Scholes d2 term: d1 - sigma * sqrt(T).

    Used with d1 in the pricing formulas; d2 appears in the probability
    terms that weight the expected payoff.
    """
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def black_scholes_call(S, K, T, r, sigma):
    """
    Black–Scholes price for a European call option (value today, not payoff at expiry).

    This is the present value of the expected payoff under the risk-neutral
    measure. Output is a price in the same units as S (e.g. dollars per share).

    Parameters
    ----------
    S : float or array-like
        Current stock price.
    K : float
        Strike price.
    T : float
        Time to maturity in years.
    r : float
        Risk-free interest rate (annual, continuous compounding).
    sigma : float
        Volatility (annual standard deviation of log-returns).
    """
    _validate_bs_inputs(S, K, T, r, sigma)
    S = np.array(S, dtype=float)
    d_1 = d1(S, K, T, r, sigma)
    d_2 = d2(S, K, T, r, sigma)
    return S * norm.cdf(d_1) - K * np.exp(-r * T) * norm.cdf(d_2)


def black_scholes_put(S, K, T, r, sigma):
    """
    Black–Scholes price for a European put option (value today, not payoff at expiry).

    Present value of the expected payoff under the risk-neutral measure.
    Same parameters as black_scholes_call.
    """
    _validate_bs_inputs(S, K, T, r, sigma)
    S = np.array(S, dtype=float)
    d_1 = d1(S, K, T, r, sigma)
    d_2 = d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d_2) - S * norm.cdf(-d_1)
