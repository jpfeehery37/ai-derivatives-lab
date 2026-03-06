"""
Options on Transocean (RIG): fetch live price, then price options and plot payoffs.

Run from project root with venv active:
  python scripts/rig_options_example.py
"""

import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

from src.pricing.black_scholes import black_scholes_call, black_scholes_put
from src.strategies.payoffs import call_payoff, put_payoff

TICKER = "RIG"


def main():
    # 1. Get Transocean's current (latest) price
    stock = yf.Ticker(TICKER)
    hist = stock.history(period="5d")
    if hist.empty:
        print(f"Could not fetch data for {TICKER}. Check ticker and connection.")
        return
    S0 = float(hist["Close"].iloc[-1])
    print(f"{TICKER} (Transocean) latest close: ${S0:.2f}")

    # 2. Option parameters (you can change these)
    K = round(S0, 0)  # at-the-money strike (round to nearest dollar)
    T = 1.0            # 1 year to expiry
    r = 0.05            # 5% risk-free rate (example)
    sigma = 0.5         # 50% vol (high vol typical for names like RIG; adjust as you like)

    # 3. Black–Scholes prices
    call_price = black_scholes_call(S0, K, T, r, sigma)
    put_price = black_scholes_put(S0, K, T, r, sigma)
    print(f"Strike K = {K}, T = {T} yr, r = {r:.0%}, sigma = {sigma:.0%}")
    print(f"  Call (BS): ${call_price:.2f}")
    print(f"  Put  (BS): ${put_price:.2f}")

    # 4. Payoff diagram (expiration)
    S_range = np.linspace(max(0.5 * K, 1), 1.5 * K, 200)
    call = call_payoff(S_range, K)
    put = put_payoff(S_range, K)

    plt.figure(figsize=(9, 5))
    plt.plot(S_range, call, label="Long call payoff")
    plt.plot(S_range, put, label="Long put payoff")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.axvline(K, color="gray", linestyle="--", label=f"Strike K={K}")
    plt.axvline(S0, color="blue", linestyle=":", alpha=0.7, label=f"Spot {S0:.1f}")
    plt.title(f"{TICKER} (Transocean) option payoffs at expiration")
    plt.xlabel("Stock price at expiration")
    plt.ylabel("Payoff")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
