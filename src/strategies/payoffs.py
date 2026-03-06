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


# --- Spreads and combinations (expiration payoff only; _pnl includes premium) ---


def bull_call_spread_payoff(S, K1, K2):
    """
    Bull call spread at expiration: long call K1, short call K2 (K1 < K2).

    Expiration payoff only. Profits when S rises moderately; caps gain above K2.
    """
    S = np.array(S)
    return call_payoff(S, K1) - call_payoff(S, K2)


def bull_call_spread_pnl(S, K1, K2, net_debit):
    """P&L at expiration: bull call spread payoff minus net debit paid."""
    return bull_call_spread_payoff(S, K1, K2) - net_debit


def bear_put_spread_payoff(S, K1, K2):
    """
    Bear put spread at expiration: long put K2, short put K1 (K1 < K2).

    Expiration payoff only. Profits when S falls; loss capped below K1.
    """
    S = np.array(S)
    return put_payoff(S, K2) - put_payoff(S, K1)


def bear_put_spread_pnl(S, K1, K2, net_debit):
    """P&L at expiration: bear put spread payoff minus net debit paid."""
    return bear_put_spread_payoff(S, K1, K2) - net_debit


def butterfly_payoff(S, K1, K2, K3):
    """
    Long butterfly at expiration: long 1 call K1, short 2 calls K2, long 1 call K3.

    K1 < K2 < K3 with K2 = (K1 + K3) / 2. Expiration payoff only; profits when S near K2.
    """
    S = np.array(S)
    return call_payoff(S, K1) - 2.0 * call_payoff(S, K2) + call_payoff(S, K3)


def butterfly_pnl(S, K1, K2, K3, net_debit):
    """P&L at expiration: butterfly payoff minus net debit paid."""
    return butterfly_payoff(S, K1, K2, K3) - net_debit


def iron_condor_payoff(S, K1, K2, K3, K4):
    """
    Iron condor at expiration: long put K1, short put K2, short call K3, long call K4.

    K1 < K2 < K3 < K4. Expiration payoff only; profits when S stays between K2 and K3.
    """
    S = np.array(S)
    return put_payoff(S, K1) - put_payoff(S, K2) + call_payoff(S, K4) - call_payoff(S, K3)


def iron_condor_pnl(S, K1, K2, K3, K4, net_credit):
    """P&L at expiration: iron condor payoff plus net credit received (premium)."""
    return iron_condor_payoff(S, K1, K2, K3, K4) + net_credit


def strangle_payoff(S, Kp, Kc):
    """
    Long strangle at expiration: long put Kp, long call Kc (Kp < Kc).

    Expiration payoff only; profits on large moves in either direction.
    """
    S = np.array(S)
    return put_payoff(S, Kp) + call_payoff(S, Kc)


def strangle_pnl(S, Kp, Kc, net_debit):
    """P&L at expiration: strangle payoff minus net debit paid."""
    return strangle_payoff(S, Kp, Kc) - net_debit
