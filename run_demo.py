import numpy as np
import matplotlib.pyplot as plt

from src.strategies.payoffs import (
    call_payoff,
    put_payoff,
    straddle_payoff,
    covered_call_expiration_pnl,
    bull_call_spread_payoff,
    bull_call_spread_pnl,
    iron_condor_payoff,
    iron_condor_pnl,
)
from src.pricing import (
    black_scholes_call,
    black_scholes_put,
    delta_call,
    delta_put,
    gamma,
    theta_call,
    theta_put,
    vega,
    rho_call,
    rho_put,
    verify_greeks_numerically,
    implied_vol,
)


def main():
    # Example option parameters
    S0 = 100.0  # current stock price
    K = 100.0   # strike price
    T = 1.0     # time to maturity in years
    r = 0.02    # risk-free rate (2%)
    sigma = 0.2 # volatility (20%)

    # Range of possible stock prices at maturity (for payoff / P&L diagrams)
    S_range = np.linspace(0.5 * K, 1.5 * K, 200)

    # Compute payoffs and P&L at expiration
    call = call_payoff(S_range, K)
    put = put_payoff(S_range, K)
    straddle = straddle_payoff(S_range, K)
    covered = covered_call_expiration_pnl(S_range, K, S0=S0)

    # Plot: these curves show what happens at expiration only (not value today)
    plt.figure(figsize=(10, 6))
    plt.plot(S_range, call, label="Call payoff")
    plt.plot(S_range, put, label="Put payoff")
    plt.plot(S_range, straddle, label="Straddle payoff")
    plt.plot(S_range, covered, label="Covered call (expiration P&L)")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.axvline(K, color="gray", linestyle="--", label="Strike K")
    plt.title("Option strategy payoffs and P&L at expiration")
    plt.xlabel("Stock price at maturity (S)")
    plt.ylabel("Payoff / P&L")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Black–Scholes gives value *today* (present value), not expiration payoff
    print("Below: option *prices today* (present value under Black–Scholes).")
    print("The plot above shows *expiration* outcomes; these are related but not the same.\n")
    call_price = black_scholes_call(S0, K, T, r, sigma)
    put_price = black_scholes_put(S0, K, T, r, sigma)
    print("Black–Scholes prices at S0 = 100:")
    print(f"  Call price: {call_price:.4f}")
    print(f"  Put price:  {put_price:.4f}")

    # --- Phase 1: Greeks at S0=100, K=100, T=1, r=0.02, sigma=0.2 ---
    print("\n--- Greeks (S0=100, K=100, T=1, r=0.02, sigma=0.2) ---")
    print(f"  delta_call: {delta_call(S0, K, T, r, sigma):.4f}")
    print(f"  delta_put:  {delta_put(S0, K, T, r, sigma):.4f}")
    print(f"  gamma:      {gamma(S0, K, T, r, sigma):.6f}")
    print(f"  theta_call: {theta_call(S0, K, T, r, sigma):.4f} (per day)")
    print(f"  theta_put:  {theta_put(S0, K, T, r, sigma):.4f} (per day)")
    print(f"  vega:       {vega(S0, K, T, r, sigma):.4f} (per 1% vol)")
    print(f"  rho_call:   {rho_call(S0, K, T, r, sigma):.4f}")
    print(f"  rho_put:    {rho_put(S0, K, T, r, sigma):.4f}")
    greek_errors = verify_greeks_numerically(S0, K, T, r, sigma, epsilon=0.01)
    print("  Numerical verification (max absolute error):")
    for name, data in greek_errors.items():
        print(f"    {name}: {data['error']:.2e}")

    # Implied vol from BS call price (expect ~0.20)
    iv = implied_vol(call_price, S0, K, T, r, option_type="call")
    print(f"\n  Implied vol from BS call price: {iv:.4f} (expected ~0.20)")

    # Bull call spread and iron condor: payoff and P&L
    K1, K2 = 90, 110
    bull_payoff = bull_call_spread_payoff(S_range, K1, K2)
    net_debit = 5.0  # example
    bull_pnl = bull_call_spread_pnl(S_range, K1, K2, net_debit)
    K1p, K2p, K3p, K4p = 85, 95, 105, 115
    ic_payoff = iron_condor_payoff(S_range, K1p, K2p, K3p, K4p)
    net_credit = 3.0  # example
    ic_pnl = iron_condor_pnl(S_range, K1p, K2p, K3p, K4p, net_credit)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(S_range, bull_payoff, label="Bull call spread payoff")
    ax1.plot(S_range, bull_pnl, label="Bull call spread P&L (net debit=5)")
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.axvline(K1, color="gray", linestyle="--")
    ax1.axvline(K2, color="gray", linestyle="--")
    ax1.set_title("Bull call spread (K1=90, K2=110)")
    ax1.set_xlabel("Stock price at maturity")
    ax1.legend()
    ax1.grid(True)
    ax2.plot(S_range, ic_payoff, label="Iron condor payoff")
    ax2.plot(S_range, ic_pnl, label="Iron condor P&L (net credit=3)")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title("Iron condor (85/95/105/115)")
    ax2.set_xlabel("Stock price at maturity")
    ax2.legend()
    ax2.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Phase 2: Market data infrastructure (optional) ---
    try:
        from scripts.phase2_demo import run_phase2_demo
        run_phase2_demo()
    except Exception as e:
        print(f"Phase 2 demo skipped: {e}")


if __name__ == "__main__":
    main()
