import numpy as np


def call_payoff(S, K):
    """
    Long call option payoff: max(S - K, 0)

    S: current or future stock price(s)
    K: strike price
    """
    S = np.array(S)
    return np.maximum(S - K, 0.0)


def put_payoff(S, K):
    """
    Long put option payoff: max(K - S, 0)

    S: current or future stock price(s)
    K: strike price
    """
    S = np.array(S)
    return np.maximum(K - S, 0.0)


def straddle_payoff(S, K):
    """
    Long straddle payoff: long call + long put at same strike

    S: current or future stock price(s)
    K: strike price
    """
    return call_payoff(S, K) + put_payoff(S, K)


def covered_call_payoff(S, K, S0=None):
    """
    Covered call payoff: long stock + short call

    S: current or future stock price(s)
    K: strike price of the short call
    S0: initial stock price paid (if None, assume S0 = K for illustration)
    """
    S = np.array(S)
    if S0 is None:
        S0 = K

    # Profit from owning the stock
    stock_pnl = S - S0
    # Loss from having sold the call option
    call_pnl = -call_payoff(S, K)
    return stock_pnl + call_pnl 
    