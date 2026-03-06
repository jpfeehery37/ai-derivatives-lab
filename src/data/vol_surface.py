"""
The volatility surface is implied volatility as a function of strike and expiration.
In real markets it is not flat: the smile (IV varying with strike) and skew (OTM
puts often richer than OTM calls) reflect demand for downside protection and
supply of yield. For equity indices, OTM puts are more expensive than
equidistant OTM calls (positive skew) — crash insurance demand.
"""

import numpy as np
import pandas as pd

from src.pricing.implied_vol import implied_vol_vectorized
from src.pricing.greeks import delta_call, delta_put
from src.data.rates import get_rate_for_expiry


def compute_iv_surface(
    df: pd.DataFrame,
    r: float | None = None,
    q: float = 0.0,
    use_rate_for_expiry: bool = True,
) -> pd.DataFrame:
    """
    Compute implied volatility for each row of a cleaned options chain; add IV, log_moneyness, delta_approx.

    Uses implied_vol_vectorized from src.pricing.implied_vol. If r is None and
    use_rate_for_expiry is True, uses get_rate_for_expiry(days_to_exp) per row.
    Drops rows where iv is NaN, <= 0, or > 5.0. Prints a one-line summary.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned chain with columns: mid, strike, spot, T, option_type, days_to_exp.
    r : float or None, optional
        Risk-free rate (decimal). If None, use rate per expiry when use_rate_for_expiry is True.
    q : float, optional
        Dividend yield. Default 0.0.
    use_rate_for_expiry : bool, optional
        If True and r is None, fetch rate per row by days_to_exp. Default True.

    Returns
    -------
    pd.DataFrame
        Same rows (minus invalid IV) with added columns: iv, log_moneyness, delta_approx.
    """
    df = df.copy()
    n = len(df)
    prices = df["mid"].astype(float).values
    S = df["spot"].astype(float).values
    K = df["strike"].astype(float).values
    T = df["T"].astype(float).values
    option_types = df["option_type"].astype(str).values
    if r is not None:
        r_arr = np.full(n, float(r))
    else:
        if use_rate_for_expiry:
            days = df["days_to_exp"].astype(int).values
            r_arr = np.array([get_rate_for_expiry(int(d)) for d in days], dtype=float)
        else:
            r_arr = np.full(n, 0.05)  # fallback when r is None and use_rate_for_expiry is False
    iv_arr = implied_vol_vectorized(prices, S, K, T, r_arr, option_types, q=q)
    df["iv"] = iv_arr
    df["log_moneyness"] = np.log(df["strike"].astype(float) / df["spot"].astype(float))
    # delta_approx: use iv as sigma; NaN where iv is NaN
    delta_vals = np.full(n, np.nan, dtype=float)
    for i in range(n):
        if np.isnan(iv_arr[i]) or iv_arr[i] <= 0 or iv_arr[i] > 5.0:
            continue
        s, k, t, sig = float(S[i]), float(K[i]), float(T[i]), float(iv_arr[i])
        rr = float(r_arr[i])
        if df["option_type"].iloc[i] == "call":
            delta_vals[i] = delta_call(s, k, t, rr, sig, q)
        else:
            delta_vals[i] = delta_put(s, k, t, rr, sig, q)
    df["delta_approx"] = delta_vals
    # Drop unreasonable IV
    valid = df["iv"].notna() & (df["iv"] > 0) & (df["iv"] <= 5.0)
    out = df.loc[valid].copy().reset_index(drop=True)
    expirations = sorted(out["expiry"].unique().tolist())
    iv_min = out["iv"].min()
    iv_max = out["iv"].max()
    print(
        f"IV surface: {len(out)} options, expirations: {expirations}, IV range: [{iv_min:.4f}, {iv_max:.4f}]"
    )
    return out


def get_atm_iv(surface_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each expiration, find the ATM implied volatility (option closest to moneyness == 1).

    Returns the term structure of ATM volatility: one row per expiry with
    expiry, days_to_exp, T, atm_iv.

    Parameters
    ----------
    surface_df : pd.DataFrame
        Output of compute_iv_surface (has expiry, days_to_exp, T, moneyness, iv).

    Returns
    -------
    pd.DataFrame
        Columns: expiry, days_to_exp, T, atm_iv. Sorted by days_to_exp.
    """
    if surface_df.empty:
        return pd.DataFrame(columns=["expiry", "days_to_exp", "T", "atm_iv"])
    rows = []
    for exp, grp in surface_df.groupby("expiry", sort=False):
        grp = grp.copy()
        grp["dist"] = (grp["moneyness"] - 1.0).abs()
        idx = grp["dist"].idxmin()
        row = grp.loc[idx]
        rows.append({
            "expiry": exp,
            "days_to_exp": row["days_to_exp"],
            "T": row["T"],
            "atm_iv": row["iv"],
        })
    out = pd.DataFrame(rows)
    out = out.sort_values("days_to_exp").reset_index(drop=True)
    return out


def get_skew(surface_df: pd.DataFrame, delta_target: float = 0.25) -> pd.DataFrame:
    """
    For each expiration, compute skew = IV(25-delta put) - IV(25-delta call).

    Uses delta_approx to find options closest to +/- delta_target. Positive skew
    means puts are more expensive (typical equity index). Returns one row per
    expiry where both sides exist.

    Parameters
    ----------
    surface_df : pd.DataFrame
        Output of compute_iv_surface (has expiry, days_to_exp, option_type, delta_approx, iv).
    delta_target : float, optional
        Target absolute delta (e.g. 0.25 for 25-delta). Default 0.25.

    Returns
    -------
    pd.DataFrame
        Columns: expiry, days_to_exp, put_iv, call_iv, skew.
    """
    if surface_df.empty:
        return pd.DataFrame(columns=["expiry", "days_to_exp", "put_iv", "call_iv", "skew"])
    rows = []
    for exp, grp in surface_df.groupby("expiry", sort=False):
        puts = grp[grp["option_type"] == "put"].copy()
        calls = grp[grp["option_type"] == "call"].copy()
        if puts.empty or calls.empty:
            continue
        puts["d_dist"] = (puts["delta_approx"] - (-delta_target)).abs()
        calls["d_dist"] = (calls["delta_approx"] - delta_target).abs()
        put_idx = puts["d_dist"].idxmin()
        call_idx = calls["d_dist"].idxmin()
        put_iv = puts.loc[put_idx, "iv"]
        call_iv = calls.loc[call_idx, "iv"]
        if pd.isna(put_iv) or pd.isna(call_iv):
            continue
        skew = float(put_iv - call_iv)
        days = grp["days_to_exp"].iloc[0]
        rows.append({
            "expiry": exp,
            "days_to_exp": days,
            "put_iv": put_iv,
            "call_iv": call_iv,
            "skew": skew,
        })
    if not rows:
        return pd.DataFrame(columns=["expiry", "days_to_exp", "put_iv", "call_iv", "skew"])
    out = pd.DataFrame(rows)
    return out.sort_values("days_to_exp").reset_index(drop=True)


def summarize_surface(surface_df: pd.DataFrame) -> dict:
    """
    Return a summary dict: ticker, spot, n_options, expirations, atm_iv_term, skew_term, iv_min, iv_max, iv_mean.

    Parameters
    ----------
    surface_df : pd.DataFrame
        Output of compute_iv_surface.

    Returns
    -------
    dict
        Keys as above; atm_iv_term and skew_term are DataFrames from get_atm_iv and get_skew.
    """
    if surface_df.empty:
        return {
            "ticker": None,
            "spot": None,
            "n_options": 0,
            "expirations": [],
            "atm_iv_term": pd.DataFrame(),
            "skew_term": pd.DataFrame(),
            "iv_min": np.nan,
            "iv_max": np.nan,
            "iv_mean": np.nan,
        }
    ticker = str(surface_df["ticker"].iloc[0])
    spot = float(surface_df["spot"].iloc[0])
    expirations = sorted(surface_df["expiry"].unique().tolist())
    return {
        "ticker": ticker,
        "spot": spot,
        "n_options": len(surface_df),
        "expirations": expirations,
        "atm_iv_term": get_atm_iv(surface_df),
        "skew_term": get_skew(surface_df),
        "iv_min": float(surface_df["iv"].min()),
        "iv_max": float(surface_df["iv"].max()),
        "iv_mean": float(surface_df["iv"].mean()),
    }
