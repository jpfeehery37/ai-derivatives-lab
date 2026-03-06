"""
The risk-free rate used in Black–Scholes is typically proxied by the yield on
short-term US Treasury bills, available from the FRED (Federal Reserve Economic
Data) API. Using a current rate matters for precise IV calculation: a fixed 5%
can misprice options when the actual short rate is far from 5%, especially for
short-dated options where the discount factor is sensitive to r.
"""

import csv
import io
import warnings

# Module-level cache: keyed by maturity string, value is decimal rate (float)
_rate_cache: dict[str, float] = {}

# FRED series IDs by maturity (1m = 1 month, etc.)
_FRED_SERIES = {
    "1m": "DGS1MO",
    "3m": "DGS3MO",
    "6m": "DGS6MO",
    "1y": "DGS1",
    "2y": "DGS2",
    "5y": "DGS5",
    "10y": "DGS10",
}


def get_risk_free_rate(maturity: str = "3m") -> float:
    """
    Fetch the current US risk-free rate from FRED as a decimal (e.g. 0.053).

    Supported maturities and FRED series: 1m->DGS1MO, 3m->DGS3MO, 6m->DGS6MO,
    1y->DGS1, 2y->DGS2, 5y->DGS5, 10y->DGS10. Uses FRED CSV endpoint (no API key).
    On request failure, falls back to 0.05 and emits a warning. Result is cached
    per maturity for the session.

    Parameters
    ----------
    maturity : str, optional
        One of "1m", "3m", "6m", "1y", "2y", "5y", "10y". Default "3m".

    Returns
    -------
    float
        Rate as decimal (e.g. 0.05 for 5%). Always a Python float, not numpy scalar.
    """
    if maturity in _rate_cache:
        return _rate_cache[maturity]
    if maturity not in _FRED_SERIES:
        raise ValueError(
            f"Unsupported maturity '{maturity}'. Use one of: {list(_FRED_SERIES.keys())}"
        )
    series_id = _FRED_SERIES[maturity]
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=10) as resp:
            text = resp.read().decode("utf-8")
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
        if not rows:
            raise ValueError("Empty FRED response")
        # Column with values is the series ID (e.g. DGS3MO)
        if series_id not in rows[0]:
            raise ValueError(f"Series column {series_id} not in FRED CSV")
        values = []
        for row in rows:
            raw = row.get(series_id, "").strip()
            if raw and raw != ".":
                try:
                    values.append(float(raw))
                except ValueError:
                    pass
        if not values:
            raise ValueError("No valid values in FRED series")
        rate_pct = values[-1]  # most recent non-null
        rate = float(rate_pct / 100.0)
    except Exception as e:
        warnings.warn(
            f"Failed to fetch risk-free rate from FRED ({e}); using default 0.05",
            UserWarning,
            stacklevel=2,
        )
        rate = 0.05
    _rate_cache[maturity] = rate
    return rate


def get_rate_for_expiry(days: int) -> float:
    """
    Return the appropriate risk-free rate for a given days-to-expiry.

    Selects the nearest FRED maturity: 1m (<=45d), 3m (<=90d), 6m (<=180d),
    1y (<=365d), 2y (<=730d), 5y (<=1825d), else 10y. Financially: matches
    option tenor to the correct discount curve point.

    Parameters
    ----------
    days : int
        Calendar days to expiration.

    Returns
    -------
    float
        Risk-free rate as decimal.
    """
    if days <= 45:
        maturity = "1m"
    elif days <= 90:
        maturity = "3m"
    elif days <= 180:
        maturity = "6m"
    elif days <= 365:
        maturity = "1y"
    elif days <= 730:
        maturity = "2y"
    elif days <= 1825:
        maturity = "5y"
    else:
        maturity = "10y"
    return get_risk_free_rate(maturity)
