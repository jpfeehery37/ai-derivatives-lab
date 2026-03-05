import numpy as np
import matplotlib.pyplot as plt

from src.strategies.payoffs import (
    call_payoff,
    put_payoff,
    straddle_payoff,
    covered_call_payoff,
)
from src.pricing.black_scholes import (
    black_scholes_call,
    black_scholes_put,
)


def main():
    # Example option parameters
    S0 = 100.0  # current stock price
    K = 100.0   # strike price
    T = 1.0     # time to maturity in years
    r = 0.02    # risk-free rate (2%)
    sigma = 0.2 # volatility (20%)

    # Range of possible stock prices at maturity for payoff diagrams
    S_range = np.linspace(0.5 * K, 1.5 * K, 200)

    # Compute payoffs
    call = call_payoff(S_range, K)
    put = put_payoff(S_range, K)
    straddle = straddle_payoff(S_range, K)
    covered = covered_call_payoff(S_range, K, S0=S0)

    # Plot payoff diagrams
    plt.figure(figsize=(10, 6))
    plt.plot(S_range, call, label="Call payoff")
    plt.plot(S_range, put, label="Put payoff")
    plt.plot(S_range, straddle, label="Straddle payoff")
    plt.plot(S_range, covered, label="Covered call payoff")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.axvline(K, color="gray", linestyle="--", label="Strike K")
    plt.title("Option Strategy Payoff Diagrams at Maturity")
    plt.xlabel("Stock price at maturity (S)")
    plt.ylabel("Payoff")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Show the payoff figure
    plt.show()

    # Demonstrate Black–Scholes pricing at S0
    call_price = black_scholes_call(S0, K, T, r, sigma)
    put_price = black_scholes_put(S0, K, T, r, sigma)

    print("Black–Scholes prices at S0 = 100:")
    print(f"  Call price: {call_price:.4f}")
    print(f"  Put price:  {put_price:.4f}")


if __name__ == "__main__":
    main()
    