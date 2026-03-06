"""
Options chain data from yfinance is often messy: stale quotes, zero volume,
crossed markets, missing IVs. This module fetches, validates, and standardizes
it into a clean DataFrame ready for IV calculation and surface construction.
"""

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None


def get_underlying_price(ticker: str) -> float:
    """
    Fetch the current spot price for a ticker using yfinance.

    Returns the most recent closing price as a float (price today, not payoff).
    Financially: the spot S used as input to Black–Scholes and IV.

    Parameters
    ----------
    ticker : str
        Stock or index symbol (e.g. "SPY", "AAPL").

    Returns
    -------
    float
        Most recent closing price.

    Raises
    ------
    ValueError
        If the ticker is invalid or data is unavailable.
    """
    if yf is None:
        raise ValueError("yfinance is not installed")
    obj = yf.Ticker(ticker)
    hist = obj.history(period="5d")
    if hist is None or hist.empty:
        raise ValueError(f"No price data available for ticker '{ticker}'")
    close = hist["Close"].iloc[-1]
    if pd.isna(close) or close <= 0:
        raise ValueError(f"Invalid or missing closing price for ticker '{ticker}'")
    return float(close)


def get_options_chain(
    ticker: str,
    expiry: str | None = None,
    option_type: str = "both",
    min_volume: int = 0,
    min_open_interest: int = 0,
) -> pd.DataFrame:
    """
    Fetch the full options chain for ticker using yfinance; standardize and filter.

    If expiry is None, fetches all available expirations and concatenates.
    If expiry is a string (e.g. "2025-09-19"), fetches that single expiration.
    option_type: "call", "put", or "both". Adds computed columns (mid, spread, T,
    moneyness, etc.) and optionally filters by volume and open interest.

    Parameters
    ----------
    ticker : str
        Underlying symbol.
    expiry : str or None, optional
        Expiration date "YYYY-MM-DD" or None for all expirations.
    option_type : str, optional
        "call", "put", or "both". Default "both".
    min_volume : int, optional
        Drop rows with volume < this (only when > 0). Default 0.
    min_open_interest : int, optional
        Drop rows with open_interest < this (only when > 0). Default 0.

    Returns
    -------
    pd.DataFrame
        One DataFrame with columns: strike, last_price, bid, ask, volume,
        open_interest, yf_iv, ticker, expiry, option_type, mid, spread, spread_pct,
        days_to_exp, T, moneyness, spot. Sorted by expiry, option_type, strike.

    Raises
    ------
    ValueError
        If no data is returned after filtering.
    """
    if yf is None:
        raise ValueError("yfinance is not installed")
    spot_price = get_underlying_price(ticker)
    obj = yf.Ticker(ticker)
    expirations = obj.options
    if not expirations:
        raise ValueError(f"No option expirations available for ticker '{ticker}'")
    if expiry is not None:
        expirations = [e for e in expirations if e == expiry]
        if not expirations:
            raise ValueError(f"Expiry '{expiry}' not found for ticker '{ticker}'")
    today = pd.Timestamp.now().normalize()
    rows = []
    for exp_str in expirations:
        try:
            chain = obj.option_chain(exp_str)
        except Exception:
            continue
        exp_dt = pd.Timestamp(exp_str)
        days_to_exp = (exp_dt - today).days
        T = days_to_exp / 365.0
        for kind, df in [("call", chain.calls), ("put", chain.puts)]:
            if option_type != "both" and kind != option_type:
                continue
            if df is None or df.empty:
                continue
            df = df.copy()
            # Normalize column names (yfinance may use lastPrice, openInterest, impliedVolatility)
            if "lastPrice" in df.columns:
                df = df.rename(columns={"lastPrice": "last_price"})
            if "openInterest" in df.columns:
                df = df.rename(columns={"openInterest": "open_interest"})
            if "impliedVolatility" in df.columns:
                df = df.rename(columns={"impliedVolatility": "yf_iv"})
            if "last_price" not in df.columns:
                df["last_price"] = np.nan
            if "open_interest" not in df.columns:
                df["open_interest"] = 0
            if "yf_iv" not in df.columns:
                df["yf_iv"] = np.nan
            keep = ["strike", "last_price", "bid", "ask", "volume", "open_interest", "yf_iv"]
            missing = [c for c in keep if c not in df.columns]
            for c in missing:
                if c == "volume":
                    df["volume"] = 0
                elif c == "open_interest":
                    df["open_interest"] = 0
                else:
                    df[c] = np.nan
            df = df[keep].copy()
            df["ticker"] = ticker
            df["expiry"] = exp_str
            df["option_type"] = kind
            df["mid"] = (df["bid"].astype(float) + df["ask"].astype(float)) / 2.0
            df["spread"] = df["ask"].astype(float) - df["bid"].astype(float)
            df["spread_pct"] = np.where(df["mid"] != 0, df["spread"] / df["mid"], np.nan)
            df["days_to_exp"] = days_to_exp
            df["T"] = T
            df["moneyness"] = df["strike"].astype(float) / spot_price
            df["spot"] = spot_price
            rows.append(df)
    if not rows:
        raise ValueError(f"No options chain data for ticker '{ticker}' (expiry={expiry})")
    out = pd.concat(rows, ignore_index=True)
    if min_volume > 0:
        before = len(out)
        out = out[out["volume"].fillna(0).astype(int) >= min_volume]
        if before > len(out):
            print(f"Filtered by min_volume={min_volume}: removed {before - len(out)} rows")
    if min_open_interest > 0:
        before = len(out)
        out = out[out["open_interest"].fillna(0).astype(int) >= min_open_interest]
        if before > len(out):
            print(f"Filtered by min_open_interest={min_open_interest}: removed {before - len(out)} rows")
    out = out.sort_values(["expiry", "option_type", "strike"]).reset_index(drop=True)
    if out.empty:
        raise ValueError("No options chain rows remaining after filtering")
    return out


def clean_chain(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply data quality filters to an options chain DataFrame.

    Removes rows with invalid bid/ask (crossed or locked), zero/negative mid,
    wide spreads (>50% of mid), expired or zero strike, and missing/zero last_price.
    Financially: ensures IV and surface construction use only tradeable quotes.

    Parameters
    ----------
    df : pd.DataFrame
        Options chain with columns: bid, ask, mid, spread_pct, T, strike, last_price.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with index reset. Prints kept N/M (P%).
    """
    if df.empty:
        print("Cleaned chain: kept 0/0 rows (0%)")
        return df.copy()
    n_before = len(df)
    df = df.copy()
    if "mid" not in df.columns:
        df["mid"] = (df["bid"].astype(float) + df["ask"].astype(float)) / 2.0
    # 1. bid > 0
    mask = df["bid"].astype(float) > 0
    # 2. ask >= bid (no crossed/locked market)
    mask &= df["ask"].astype(float) >= df["bid"].astype(float)
    # 3. mid > 0
    mask &= df["mid"].astype(float) > 0
    # 4. spread_pct <= 0.5 (illiquid if wider than 50%)
    mask &= df["spread_pct"].fillna(0) <= 0.5
    # 5. T > 0 (not expired)
    mask &= df["T"].astype(float) > 0
    # 6. strike > 0
    mask &= df["strike"].astype(float) > 0
    # last_price not NaN and not zero
    mask &= df["last_price"].notna() & (df["last_price"].astype(float) > 0)
    out = df.loc[mask].copy().reset_index(drop=True)
    pct = 100.0 * len(out) / n_before if n_before else 0
    print(f"Cleaned chain: kept {len(out)}/{n_before} rows ({pct:.1f}%)")
    return out


def get_expirations(ticker: str) -> list[str]:
    """
    Return a sorted list of available expiration dates for ticker as "YYYY-MM-DD" strings.

    Parameters
    ----------
    ticker : str
        Underlying symbol.

    Returns
    -------
    list of str
        Sorted expiration date strings.
    """
    if yf is None:
        raise ValueError("yfinance is not installed")
    obj = yf.Ticker(ticker)
    opts = obj.options
    if not opts:
        return []
    return sorted(list(opts))


def get_price_history(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for ticker using yfinance.

    Returns a DataFrame with columns Open, High, Low, Close, Volume and DatetimeIndex.
    Financially: used for realized volatility and vol cone (price history, not payoff).

    Parameters
    ----------
    ticker : str
        Underlying symbol.
    period : str, optional
        yfinance period (e.g. "1y", "6mo"). Default "1y".
    interval : str, optional
        Bar interval (e.g. "1d", "1wk"). Default "1d".

    Returns
    -------
    pd.DataFrame
        Columns: Open, High, Low, Close, Volume. Index: DatetimeIndex.

    Raises
    ------
    ValueError
        If data is empty.
    """
    if yf is None:
        raise ValueError("yfinance is not installed")
    obj = yf.Ticker(ticker)
    hist = obj.history(period=period, interval=interval)
    if hist is None or hist.empty:
        raise ValueError(f"No price history for ticker '{ticker}' (period={period})")
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in hist.columns]
    out = hist[keep].copy()
    out.index = pd.DatetimeIndex(out.index)
    if out.empty:
        raise ValueError(f"Price history empty for ticker '{ticker}'")
    return out
