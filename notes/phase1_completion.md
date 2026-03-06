# Phase 1 completion — Options pricing lab

This note summarizes the modules added for Phase 1 and how they connect.

**Greeks** (`src/pricing/greeks.py`): Analytical Black–Scholes sensitivities (delta, gamma, theta, vega, rho) with optional dividend yield `q`. Theta is per calendar day; vega and rho are per 1 percentage point. They are used for hedging and risk, and `verify_greeks_numerically` checks them against finite differences. All rely on the same BS formulas as in `black_scholes.py`.

**Implied volatility** (`src/pricing/implied_vol.py`): Inverts Black–Scholes to find the volatility that matches a given market price. No closed form, so we use Newton–Raphson (with vega) or Brent when vega is tiny. BS and Greeks together support this: we price with BS and use vega as the derivative for Newton.

**Parity** (`src/pricing/parity.py`): Put–call parity is \(C - P = S e^{-qT} - K e^{-rT}\). We compute the residual and a checker (pass/fail, implied vs theoretical forward). Violations usually indicate data or quote issues. It uses the same discounting as BS (with `q`).

**New payoffs** (`src/strategies/payoffs.py`): Bull call spread, bear put spread, butterfly, iron condor, and strangle, each with expiration payoff and P&L (including premium). These are built from the existing `call_payoff` and `put_payoff`; they describe strategy outcomes at expiry, while BS gives *prices today*.

**Binomial** (`src/pricing/binomial.py`): CRR tree pricer for European and American options with optional `q`. As the number of steps increases, the tree converges to GBM and the European price to Black–Scholes. `convergence_to_bs` illustrates this with a table of step counts vs BS.

Overall: **BS + Greeks + parity** form the core (prices, sensitivities, and a no-arbitrage check). **Implied vol** inverts BS using those same formulas. **Binomial** provides a discrete-time model that converges to BS and supports American exercise.
