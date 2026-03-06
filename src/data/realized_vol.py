"""
Realized volatility is the actual annualized standard deviation of log returns
observed over a historical window, in contrast to implied volatility (the market's
forward-looking expectation). The difference IV - RV (variance risk premium) is
a key concept for understanding whether options are cheap or expensive relative
to recent realized risk.
"""

import numpy as np
import pandas as pd


def compute_realized_vol(
    prices: pd.Series,
    window: int = 30,
    trading_days: int = 252,
    annualize: bool = True,
) -> pd.Series:
    """
    Compute rolling realized volatility from a price series.

    Method: annualized standard deviation of log returns over a rolling window.
    log_returns = ln(P_t / P_{t-1}); rv_t = std(log_returns over window) * sqrt(trading_days) if annualize.
    Returns a Series of the same length as prices (NaN for first window-1 values).
    Name: "realized_vol_{window}d".

    Parameters
    ----------
    prices : pd.Series
        Price series (e.g. Close); index preserved in output.
    window : int, optional
        Rolling window size. Default 30.
    trading_days : int, optional
        Days per year for annualization. Default 252.
    annualize : bool, optional
        If True, multiply by sqrt(trading_days). Default True.

    Returns
    -------
    pd.Series
        Rolling realized vol; name "realized_vol_{window}d". Same index as prices.
    """
    if window < 2:
        raise ValueError("window must be at least 2")
    log_returns = np.log(prices / prices.shift(1))
    # Rolling std of log returns
    rv = log_returns.rolling(window=window, min_periods=window).std()
    if annualize:
        rv = rv * np.sqrt(trading_days)
    rv = rv.rename(f"realized_vol_{window}d")
    return rv


def compute_parkinson_vol(
    high: pd.Series,
    low: pd.Series,
    window: int = 30,
    trading_days: int = 252,
) -> pd.Series:
    """
    Parkinson volatility estimator using high-low range (more efficient than close-to-close).

    Formula per day: (1 / (4 * ln(2))) * ln(H/L)^2. Rolling mean over window,
    then annualize: sqrt(mean * trading_days). Returns Series named "parkinson_vol_{window}d".

    Parameters
    ----------
    high, low : pd.Series
        Daily high and low prices; same index.
    window : int, optional
        Rolling window. Default 30.
    trading_days : int, optional
        Days per year. Default 252.

    Returns
    -------
    pd.Series
        Name "parkinson_vol_{window}d".
    """
    if window < 1:
        raise ValueError("window must be at least 1")
    # Avoid log(0): clip low to a small positive where high/low are valid
    hl = np.maximum(high.astype(float), low.astype(float) + 1e-12)
    low_safe = np.minimum(low.astype(float), hl - 1e-12)
    daily_term = (np.log(high.astype(float) / low_safe) ** 2) / (4.0 * np.log(2))
    roll_mean = daily_term.rolling(window=window, min_periods=window).mean()
    ann = np.sqrt(roll_mean * trading_days)
    return ann.rename(f"parkinson_vol_{window}d")


def compute_vol_cone(
    prices: pd.Series,
    windows: list[int] | None = None,
    percentiles: list[float] | None = None,
    trading_days: int = 252,
) -> pd.DataFrame:
    """
    Compute a volatility cone: realized vol summary across multiple lookback windows.

    For each window, compute rolling realized vol then percentiles and current value.
    Default windows: [10, 21, 30, 60, 90, 252]. Default percentiles: [10, 25, 50, 75, 90].
    Used to assess whether current implied vol is cheap or expensive vs history.

    Parameters
    ----------
    prices : pd.Series
        Price series (e.g. Close).
    windows : list of int or None, optional
        Lookback windows. Default [10, 21, 30, 60, 90, 252].
    percentiles : list of float or None, optional
        Percentiles to compute. Default [10, 25, 50, 75, 90].
    trading_days : int, optional
        Days per year. Default 252.

    Returns
    -------
    pd.DataFrame
        Index: window. Columns: p10, p25, p50, p75, p90, current.
    """
    if windows is None:
        windows = [10, 21, 30, 60, 90, 252]
    if percentiles is None:
        percentiles = [10.0, 25.0, 50.0, 75.0, 90.0]
    rows = []
    for w in windows:
        rv = compute_realized_vol(prices, window=w, trading_days=trading_days, annualize=True)
        pct_cols = [f"p{int(p)}" for p in percentiles]
        pct_vals = np.nanpercentile(rv.values, percentiles)
        current = rv.iloc[-1] if len(rv) and pd.notna(rv.iloc[-1]) else np.nan
        row = dict(zip(pct_cols, pct_vals))
        row["current"] = current
        rows.append(row)
    out = pd.DataFrame(rows, index=pd.Index(windows, name="window"))
    return out


def compute_vrp(
    realized_vol: pd.Series,
    implied_vol_series: pd.Series,
) -> pd.Series:
    """
    Compute the variance risk premium: IV - RV (both as decimals).

    Aligns both series by date index before subtracting. Positive VRP means
    options are expensive relative to realized vol. Returns Series named "vrp".

    Parameters
    ----------
    realized_vol : pd.Series
        Realized volatility (decimal).
    implied_vol_series : pd.Series
        Implied volatility (decimal), e.g. from surface or index.

    Returns
    -------
    pd.Series
        VRP = IV - RV; name "vrp". Index: alignment of the two inputs.
    """
    common = realized_vol.align(implied_vol_series, join="inner")
    rv, iv = common[0], common[1]
    vrp = iv - rv
    return vrp.rename("vrp")
