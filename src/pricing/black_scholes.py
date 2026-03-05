import numpy as np
from scipy.stats import norm


def d1(S, K, T, r, sigma):
    """
    Black–Scholes d1 term.

    S: current stock price
    K: strike price
    T: time to maturity in years
    r: risk-free interest rate (annual, continuous compounding)
    sigma: volatility (annual standard deviation of returns)
    """
    S = np.array(S, dtype=float)
    numerator = np.log(S / K) + (r + 0.5 * sigma**2) * T
    denominator = sigma * np.sqrt(T)
    return numerator / denominator


def d2(S, K, T, r, sigma):
    """
    Black–Scholes d2 term: d1 - sigma * sqrt(T)
    """
    return d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def black_scholes_call(S, K, T, r, sigma):
    """
    Black–Scholes price for a European call option.

    S: current stock price
    K: strike price
    T: time to maturity in years
    r: risk-free interest rate (annual, continuous compounding)
    sigma: volatility (annual standard deviation of returns)
    """
    S = np.array(S, dtype=float)
    d_1 = d1(S, K, T, r, sigma)
    d_2 = d2(S, K, T, r, sigma)
    return S * norm.cdf(d_1) - K * np.exp(-r * T) * norm.cdf(d_2)


def black_scholes_put(S, K, T, r, sigma):
    """
    Black–Scholes price for a European put option.

    Parameters are the same as for black_scholes_call.
    """
    S = np.array(S, dtype=float)
    d_1 = d1(S, K, T, r, sigma)
    d_2 = d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d_2) - S * norm.cdf(-d_1)
    