import numpy as np


def call_payoff(S, K):
    """
    Long call option expiration payoff: max(S - K, 0).

    Returns expiration payoff only (no option premium or cost basis).
    Financially: the payoff at expiry from holding the right to buy the
    underlying at strike K.

    Parameters
    ----------
    S : float or array-like
        Spot price(s) at expiration (scalar or array).
    K : float
        Strike price.
    """
    S = np.array(S)
    return np.maximum(S - K, 0.0)


def put_payoff(S, K):
    """
    Long put option expiration payoff: max(K - S, 0).

    Returns expiration payoff only (no option premium or cost basis).
    Financially: the payoff at expiry from holding the right to sell the
    underlying at strike K.

    Parameters
    ----------
    S : float or array-like
        Spot price(s) at expiration (scalar or array).
    K : float
        Strike price.
    """
    S = np.array(S)
    return np.maximum(K - S, 0.0)


def straddle_payoff(S, K):
    """
    Long straddle expiration payoff: long call + long put at same strike.

    Returns expiration payoff only (no premiums or cost basis).
    Financially: payoff at expiry from holding one call and one put at strike K.

    Parameters
    ----------
    S : float or array-like
        Spot price(s) at expiration (scalar or array).
    K : float
        Strike price (same for both options).
    """
    return call_payoff(S, K) + put_payoff(S, K)


def covered_call_expiration_pnl(S, K, S0=None):
    """
    Covered call expiration profit/loss: long stock at S0 + short call at K.

    Returns expiration profit/loss (includes cost basis S0), not pure payoff.
    Financially: at expiry you have the stock (bought at S0) and owe the
    call payoff to the buyer; this is the net P&L of that strategy.

    Parameters
    ----------
    S : float or array-like
        Spot price(s) at expiration (scalar or array).
    K : float
        Strike price of the short call.
    S0 : float, optional
        Initial stock price paid. If None, defaults to K for illustration.
    """
    S = np.array(S)
    if S0 is None:
        S0 = K
    stock_pnl = S - S0
    call_pnl = -call_payoff(S, K)
    return stock_pnl + call_pnl


def covered_call_payoff(S, K, S0=None):
    """
    Deprecated. Use covered_call_expiration_pnl instead.

    This function returns expiration P&L (includes cost basis S0), not a
    pure payoff. Kept for backward compatibility.
    """
    return covered_call_expiration_pnl(S, K, S0)
