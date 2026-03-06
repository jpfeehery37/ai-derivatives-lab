"""
Analytical Black–Scholes Greeks (delta, gamma, theta, vega, rho).

All sensitivities are for European options with optional continuous dividend yield q.
Theta is per calendar day; vega and rho are per 1 percentage point move.
"""

import numpy as np
from scipy.stats import norm

from src.pricing.black_scholes import (
    _validate_bs_inputs,
    d1,
    d2,
    black_scholes_call,
    black_scholes_put,
)


def delta_call(S, K, T, r, sigma, q=0.0):
    """
    Delta of a European call: sensitivity of call price to a small change in S.

    Financially: approximate number of shares to replicate the call. In [0, 1] for non-dividend.

    Parameters
    ----------
    S, K, T, r, sigma, q : float or array-like
        Standard Black–Scholes parameters. q = continuous dividend yield (default 0).
    """
    _validate_bs_inputs(S, K, T, r, sigma, q)
    S = np.array(S, dtype=float)
    d_1 = d1(S, K, T, r, sigma, q)
    return np.exp(-q * T) * norm.cdf(d_1)


def delta_put(S, K, T, r, sigma, q=0.0):
    """
    Delta of a European put: sensitivity of put price to a small change in S.

    Financially: approximate number of shares to short to replicate the put. In [-1, 0] for non-dividend.

    Parameters
    ----------
    S, K, T, r, sigma, q : float or array-like
        Standard Black–Scholes parameters.
    """
    _validate_bs_inputs(S, K, T, r, sigma, q)
    S = np.array(S, dtype=float)
    d_1 = d1(S, K, T, r, sigma, q)
    return np.exp(-q * T) * (norm.cdf(d_1) - 1.0)


def gamma(S, K, T, r, sigma, q=0.0):
    """
    Gamma: sensitivity of delta to a small change in S (same for call and put).

    Financially: curvature of option value vs S; always positive for long options.

    Parameters
    ----------
    S, K, T, r, sigma, q : float or array-like
        Standard Black–Scholes parameters.
    """
    _validate_bs_inputs(S, K, T, r, sigma, q)
    S = np.array(S, dtype=float)
    d_1 = d1(S, K, T, r, sigma, q)
    sigma_sqrt_T = sigma * np.sqrt(T)
    # Avoid division by zero when T or sigma is tiny
    denom = S * sigma_sqrt_T
    return np.exp(-q * T) * norm.pdf(d_1) / np.where(denom > 1e-14, denom, np.nan)


def theta_call(S, K, T, r, sigma, q=0.0):
    """
    Theta of a European call: sensitivity to time (time decay).

    Returned in dollars per calendar day (annual theta divided by 365). Usually negative.

    Parameters
    ----------
    S, K, T, r, sigma, q : float or array-like
        Standard Black–Scholes parameters.
    """
    _validate_bs_inputs(S, K, T, r, sigma, q)
    S = np.array(S, dtype=float)
    d_1 = d1(S, K, T, r, sigma, q)
    d_2 = d2(S, K, T, r, sigma, q)
    sqrt_T = np.sqrt(T)
    # Annual theta: -S*exp(-q*T)*phi(d1)*sigma/(2*sqrt(T)) - r*K*exp(-r*T)*N(d2) + q*S*exp(-q*T)*N(d1)
    term1 = -S * np.exp(-q * T) * norm.pdf(d_1) * sigma / (2.0 * np.where(sqrt_T > 1e-14, sqrt_T, np.nan))
    term2 = -r * K * np.exp(-r * T) * norm.cdf(d_2)
    term3 = q * S * np.exp(-q * T) * norm.cdf(d_1)
    theta_annual = term1 + term2 + term3
    return theta_annual / 365.0


def theta_put(S, K, T, r, sigma, q=0.0):
    """
    Theta of a European put: sensitivity to time (time decay).

    Returned in dollars per calendar day. Usually negative for ATM options.

    Parameters
    ----------
    S, K, T, r, sigma, q : float or array-like
        Standard Black–Scholes parameters.
    """
    _validate_bs_inputs(S, K, T, r, sigma, q)
    S = np.array(S, dtype=float)
    d_1 = d1(S, K, T, r, sigma, q)
    d_2 = d2(S, K, T, r, sigma, q)
    sqrt_T = np.sqrt(T)
    # Annual theta: -S*exp(-q*T)*phi(d1)*sigma/(2*sqrt(T)) + r*K*exp(-r*T)*N(-d2) - q*S*exp(-q*T)*N(-d1)
    term1 = -S * np.exp(-q * T) * norm.pdf(d_1) * sigma / (2.0 * np.where(sqrt_T > 1e-14, sqrt_T, np.nan))
    term2 = r * K * np.exp(-r * T) * norm.cdf(-d_2)
    term3 = -q * S * np.exp(-q * T) * norm.cdf(-d_1)
    theta_annual = term1 + term2 + term3
    return theta_annual / 365.0


def vega(S, K, T, r, sigma, q=0.0):
    """
    Vega: sensitivity of option value to a 1 percentage point increase in volatility.

    Same for call and put. Returned per 1% move in sigma (divide by 100).

    Parameters
    ----------
    S, K, T, r, sigma, q : float or array-like
        Standard Black–Scholes parameters.
    """
    _validate_bs_inputs(S, K, T, r, sigma, q)
    S = np.array(S, dtype=float)
    d_1 = d1(S, K, T, r, sigma, q)
    # Raw vega = S*exp(-q*T)*sqrt(T)*phi(d1); per 1% vol we divide by 100
    vega_raw = S * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d_1)
    return vega_raw / 100.0


def rho_call(S, K, T, r, sigma, q=0.0):
    """
    Rho of a European call: sensitivity to a 1 percentage point increase in risk-free rate.

    Returned per 1% move in r (divide by 100).

    Parameters
    ----------
    S, K, T, r, sigma, q : float or array-like
        Standard Black–Scholes parameters.
    """
    _validate_bs_inputs(S, K, T, r, sigma, q)
    S = np.array(S, dtype=float)
    d_2 = d2(S, K, T, r, sigma, q)
    rho_raw = K * T * np.exp(-r * T) * norm.cdf(d_2)
    return rho_raw / 100.0


def rho_put(S, K, T, r, sigma, q=0.0):
    """
    Rho of a European put: sensitivity to a 1 percentage point increase in risk-free rate.

    Returned per 1% move in r. Typically negative.

    Parameters
    ----------
    S, K, T, r, sigma, q : float or array-like
        Standard Black–Scholes parameters.
    """
    _validate_bs_inputs(S, K, T, r, sigma, q)
    S = np.array(S, dtype=float)
    d_2 = d2(S, K, T, r, sigma, q)
    rho_raw = -K * T * np.exp(-r * T) * norm.cdf(-d_2)
    return rho_raw / 100.0


def verify_greeks_numerically(S, K, T, r, sigma, q=0.0, epsilon=0.01):
    """
    Compare analytical Greeks to finite-difference approximations (call-based).

    Uses black_scholes_call for bumping. Returns a dict with analytical vs numerical
    and absolute error for delta, gamma, vega, and theta.

    Parameters
    ----------
    S, K, T, r, sigma, q : float
        Single set of parameters (scalars).
    epsilon : float
        Bump size for finite differences (default 0.01).
    """
    _validate_bs_inputs(S, K, T, r, sigma, q)
    S, K, T, r, sigma, q = float(S), float(K), float(T), float(r), float(sigma), float(q)

    def bs_call(s, sig, t):
        return black_scholes_call(s, K, t, r, sig, q)

    # Delta: (C(S+ε) - C(S-ε)) / (2*ε)
    C_up = bs_call(S + epsilon, sigma, T)
    C_dn = bs_call(S - epsilon, sigma, T)
    delta_num = (C_up - C_dn) / (2.0 * epsilon)
    delta_ana = float(delta_call(S, K, T, r, sigma, q))

    # Gamma: (C(S+ε) - 2*C(S) + C(S-ε)) / ε^2
    C_mid = bs_call(S, sigma, T)
    gamma_num = (C_up - 2.0 * C_mid + C_dn) / (epsilon**2)
    gamma_ana = float(gamma(S, K, T, r, sigma, q))

    # Vega: (C(σ+ε) - C(σ-ε)) / (2*ε); vega we report per 1% so numerical is same scale
    C_sig_up = bs_call(S, sigma + epsilon, T)
    C_sig_dn = bs_call(S, sigma - epsilon, T)
    vega_num_per_pct = (C_sig_up - C_sig_dn) / (2.0 * epsilon) / 100.0
    vega_ana = float(vega(S, K, T, r, sigma, q))

    # Theta: (C(T-ε) - C(T)) / ε (one-sided, time shrinking); theta per day
    T_shrink = max(T - epsilon, epsilon * 0.5)
    C_T = bs_call(S, sigma, T)
    C_T_shrink = bs_call(S, sigma, T_shrink)
    theta_num_per_day = (C_T_shrink - C_T) / (T - T_shrink) / 365.0
    theta_ana = float(theta_call(S, K, T, r, sigma, q))

    return {
        "delta": {"analytical": delta_ana, "numerical": delta_num, "error": abs(delta_ana - delta_num)},
        "gamma": {"analytical": gamma_ana, "numerical": gamma_num, "error": abs(gamma_ana - gamma_num)},
        "vega": {"analytical": vega_ana, "numerical": vega_num_per_pct, "error": abs(vega_ana - vega_num_per_pct)},
        "theta": {"analytical": theta_ana, "numerical": theta_num_per_day, "error": abs(theta_ana - theta_num_per_day)},
    }
