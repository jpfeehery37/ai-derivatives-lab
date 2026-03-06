"""
Comprehensive tests for Phase 1: Black–Scholes (with q), Greeks, implied vol,
parity, binomial, and parity checker.
"""

import numpy as np
import pytest

from src.pricing.black_scholes import (
    black_scholes_call,
    black_scholes_put,
    d1,
    d2,
    _validate_bs_inputs,
)
from src.pricing.greeks import (
    delta_call,
    delta_put,
    gamma,
    theta_call,
    vega,
    verify_greeks_numerically,
)
from src.pricing.implied_vol import implied_vol
from src.pricing.parity import parity_residual, check_put_call_parity
from src.pricing.binomial import binomial_price, convergence_to_bs


# -----------------------------------------------------------------------------
# Black–Scholes: known values and parity with q
# -----------------------------------------------------------------------------


def test_bs_known_call_put(standard_params):
    """Known call ≈ 10.4506, put ≈ 5.5735 for S=100, K=100, T=1, r=0.05, sigma=0.2, q=0."""
    p = standard_params
    C = black_scholes_call(p["S"], p["K"], p["T"], p["r"], p["sigma"], p["q"])
    P = black_scholes_put(p["S"], p["K"], p["T"], p["r"], p["sigma"], p["q"])
    assert C == pytest.approx(10.4506, abs=0.01)
    assert P == pytest.approx(5.5735, abs=0.01)


def test_put_call_parity_with_q(standard_params):
    """C - P == S*exp(-q*T) - K*exp(-r*T) (atol 1e-10)."""
    p = standard_params
    C = black_scholes_call(p["S"], p["K"], p["T"], p["r"], p["sigma"], p["q"])
    P = black_scholes_put(p["S"], p["K"], p["T"], p["r"], p["sigma"], p["q"])
    rhs = p["S"] * np.exp(-p["q"] * p["T"]) - p["K"] * np.exp(-p["r"] * p["T"])
    assert np.isclose(C - P, rhs, atol=1e-10, rtol=0)


def test_bs_q_zero_matches_old_behavior(standard_params):
    """With q=0, call/put behave as before (no q argument)."""
    p = standard_params
    C0 = black_scholes_call(p["S"], p["K"], p["T"], p["r"], p["sigma"])
    Cq = black_scholes_call(p["S"], p["K"], p["T"], p["r"], p["sigma"], 0.0)
    P0 = black_scholes_put(p["S"], p["K"], p["T"], p["r"], p["sigma"])
    Pq = black_scholes_put(p["S"], p["K"], p["T"], p["r"], p["sigma"], 0.0)
    assert C0 == Cq and P0 == Pq


def test_bs_validation_S_positive():
    """ValueError for S <= 0."""
    with pytest.raises(ValueError, match="S"):
        black_scholes_call(0.0, 100, 1, 0.05, 0.2)
    with pytest.raises(ValueError, match="S"):
        black_scholes_call(-1.0, 100, 1, 0.05, 0.2)


def test_bs_validation_K_positive():
    """ValueError for K <= 0."""
    with pytest.raises(ValueError, match="K"):
        black_scholes_call(100, 0, 1, 0.05, 0.2)


def test_bs_validation_T_positive():
    """ValueError for T <= 0."""
    with pytest.raises(ValueError, match="T"):
        black_scholes_call(100, 100, 0, 0.05, 0.2)


def test_bs_validation_sigma_positive():
    """ValueError for sigma <= 0."""
    with pytest.raises(ValueError, match="sigma"):
        black_scholes_call(100, 100, 1, 0.05, 0.0)


# -----------------------------------------------------------------------------
# Greeks
# -----------------------------------------------------------------------------


def test_delta_call_bounds(standard_params):
    """delta_call in [0, 1] for standard params."""
    p = standard_params
    d = delta_call(p["S"], p["K"], p["T"], p["r"], p["sigma"], p["q"])
    assert 0 <= d <= 1


def test_delta_put_bounds(standard_params):
    """delta_put in [-1, 0] for standard params."""
    p = standard_params
    d = delta_put(p["S"], p["K"], p["T"], p["r"], p["sigma"], p["q"])
    assert -1 <= d <= 0


def test_delta_parity_delta_call_minus_delta_put(standard_params):
    """delta_call - delta_put == exp(-q*T)."""
    p = standard_params
    dc = delta_call(p["S"], p["K"], p["T"], p["r"], p["sigma"], p["q"])
    dp = delta_put(p["S"], p["K"], p["T"], p["r"], p["sigma"], p["q"])
    assert np.isclose(dc - dp, np.exp(-p["q"] * p["T"]), atol=1e-10)


def test_gamma_positive(standard_params):
    """gamma > 0."""
    p = standard_params
    g = gamma(p["S"], p["K"], p["T"], p["r"], p["sigma"], p["q"])
    assert g > 0


def test_vega_positive(standard_params):
    """vega > 0."""
    p = standard_params
    v = vega(p["S"], p["K"], p["T"], p["r"], p["sigma"], p["q"])
    assert v > 0


def test_theta_call_negative_atm(standard_params):
    """theta < 0 for ATM (time decay)."""
    p = standard_params
    t = theta_call(p["S"], p["K"], p["T"], p["r"], p["sigma"], p["q"])
    assert t < 0


def test_verify_greeks_numerically_errors_small(standard_params):
    """verify_greeks_numerically: all errors < 1e-4."""
    p = standard_params
    result = verify_greeks_numerically(
        p["S"], p["K"], p["T"], p["r"], p["sigma"], p["q"], epsilon=0.01
    )
    for key in ("delta", "gamma", "vega", "theta"):
        assert result[key]["error"] < 1e-4, f"{key} error too large"


# -----------------------------------------------------------------------------
# Implied vol
# -----------------------------------------------------------------------------


def test_implied_vol_recover_from_call(standard_params):
    """Recover sigma from BS call price to within 1e-5."""
    p = standard_params
    C = black_scholes_call(p["S"], p["K"], p["T"], p["r"], p["sigma"], p["q"])
    iv = implied_vol(C, p["S"], p["K"], p["T"], p["r"], option_type="call", q=p["q"])
    assert iv == pytest.approx(p["sigma"], abs=1e-5)


def test_implied_vol_recover_from_put(standard_params):
    """Recover sigma from BS put price to within 1e-5."""
    p = standard_params
    P = black_scholes_put(p["S"], p["K"], p["T"], p["r"], p["sigma"], p["q"])
    iv = implied_vol(P, p["S"], p["K"], p["T"], p["r"], option_type="put", q=p["q"])
    assert iv == pytest.approx(p["sigma"], abs=1e-5)


def test_implied_vol_impossible_price_returns_nan(standard_params):
    """Impossible price (e.g. above max) yields np.nan and warning."""
    p = standard_params
    # Call price cannot exceed S; use something above S
    with pytest.warns(UserWarning, match="converge"):
        iv = implied_vol(200.0, p["S"], p["K"], p["T"], p["r"], option_type="call")
    assert np.isnan(iv)


def test_implied_vol_at_intrinsic_returns_zero(standard_params):
    """Price at or below intrinsic → 0.0."""
    p = standard_params
    intrinsic_call = max(0.0, p["S"] * np.exp(-p["q"] * p["T"]) - p["K"] * np.exp(-p["r"] * p["T"]))
    iv = implied_vol(intrinsic_call, p["S"], p["K"], p["T"], p["r"], option_type="call", q=p["q"])
    assert iv == 0.0


def test_implied_vol_negative_price_raises():
    """market_price <= 0 raises ValueError."""
    with pytest.raises(ValueError, match="market_price"):
        implied_vol(0.0, 100, 100, 1, 0.05)


# -----------------------------------------------------------------------------
# Binomial
# -----------------------------------------------------------------------------


def test_binomial_european_call_converges_to_bs(standard_params):
    """European binomial call vs BS: n=200, error < 0.01."""
    p = standard_params
    bs = black_scholes_call(p["S"], p["K"], p["T"], p["r"], p["sigma"], p["q"])
    bn = binomial_price(p["S"], p["K"], p["T"], p["r"], p["sigma"], n=200, option_type="call", exercise="european", q=p["q"])
    assert abs(bn - bs) < 0.01


def test_american_call_equals_european_no_dividend(standard_params):
    """American call == European for q=0 (no early exercise optimal)."""
    p = standard_params
    eu = binomial_price(p["S"], p["K"], p["T"], p["r"], p["sigma"], n=100, option_type="call", exercise="european", q=0.0)
    am = binomial_price(p["S"], p["K"], p["T"], p["r"], p["sigma"], n=100, option_type="call", exercise="american", q=0.0)
    assert eu == pytest.approx(am, abs=0.001)


# -----------------------------------------------------------------------------
# Parity module
# -----------------------------------------------------------------------------


def test_parity_residual_near_zero_for_bs_synthetic(standard_params):
    """parity_residual ~ 0 for (C, P) from Black–Scholes."""
    p = standard_params
    C = black_scholes_call(p["S"], p["K"], p["T"], p["r"], p["sigma"], p["q"])
    P = black_scholes_put(p["S"], p["K"], p["T"], p["r"], p["sigma"], p["q"])
    res = parity_residual(C, P, p["S"], p["K"], p["T"], p["r"], p["q"])
    assert abs(res) < 1e-10


def test_check_put_call_parity_passes_within_tol(standard_params):
    """check_put_call_parity passes when residual within tol."""
    p = standard_params
    C = black_scholes_call(p["S"], p["K"], p["T"], p["r"], p["sigma"], p["q"])
    P = black_scholes_put(p["S"], p["K"], p["T"], p["r"], p["sigma"], p["q"])
    out = check_put_call_parity(C, P, p["S"], p["K"], p["T"], p["r"], p["q"], tol=0.05)
    assert out["passes"] is True or np.all(out["passes"])


def test_check_put_call_parity_fails_outside_tol(standard_params):
    """check_put_call_parity fails when residual outside tol."""
    p = standard_params
    C = black_scholes_call(p["S"], p["K"], p["T"], p["r"], p["sigma"], p["q"])
    P = black_scholes_put(p["S"], p["K"], p["T"], p["r"], p["sigma"], p["q"])
    # Artificially break parity
    out = check_put_call_parity(C + 10.0, P, p["S"], p["K"], p["T"], p["r"], p["q"], tol=0.05)
    assert out["passes"] is False or not np.any(out["passes"])
