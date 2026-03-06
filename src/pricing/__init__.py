"""Options pricing: Black–Scholes, Greeks, implied vol, parity, binomial."""

from src.pricing.black_scholes import (
    black_scholes_call,
    black_scholes_put,
    d1,
    d2,
)
from src.pricing.greeks import (
    delta_call,
    delta_put,
    gamma,
    theta_call,
    theta_put,
    vega,
    rho_call,
    rho_put,
    verify_greeks_numerically,
)
from src.pricing.implied_vol import implied_vol, implied_vol_vectorized
from src.pricing.parity import parity_residual, check_put_call_parity
from src.pricing.binomial import binomial_price, convergence_to_bs

__all__ = [
    "black_scholes_call",
    "black_scholes_put",
    "d1",
    "d2",
    "delta_call",
    "delta_put",
    "gamma",
    "theta_call",
    "theta_put",
    "vega",
    "rho_call",
    "rho_put",
    "verify_greeks_numerically",
    "implied_vol",
    "implied_vol_vectorized",
    "parity_residual",
    "check_put_call_parity",
    "binomial_price",
    "convergence_to_bs",
]
