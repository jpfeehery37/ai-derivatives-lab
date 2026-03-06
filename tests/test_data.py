"""
Pytest tests for Phase 2 data modules. Use monkeypatching to mock yfinance and
network calls so tests do not depend on live data.
"""

import io
import csv
import pytest
import numpy as np
import pandas as pd

from src.data.fetcher import clean_chain, get_options_chain
from src.data.rates import get_risk_free_rate, get_rate_for_expiry, _rate_cache
from src.data.vol_surface import compute_iv_surface, get_atm_iv, get_skew
from src.data.realized_vol import (
    compute_realized_vol,
    compute_parkinson_vol,
    compute_vol_cone,
    compute_vrp,
)
from src.pricing.black_scholes import black_scholes_call, black_scholes_put


# -----------------------------------------------------------------------------
# Fixtures: shared chain-like DataFrames
# -----------------------------------------------------------------------------


@pytest.fixture
def chain_with_crossed_markets():
    """DataFrame with one row where bid > ask (crossed)."""
    return pd.DataFrame({
        "strike": [100.0, 105.0, 110.0],
        "last_price": [5.0, 3.0, 1.0],
        "bid": [4.8, 3.5, 0.9],
        "ask": [5.2, 3.1, 1.1],
        "volume": [10, 20, 30],
        "open_interest": [100, 200, 300],
        "yf_iv": [0.2, 0.19, 0.18],
        "mid": [5.0, (3.5 + 3.1) / 2, 1.0],
        "spread": [0.4, -0.4, 0.2],
        "spread_pct": [0.08, 0.12, 0.2],
        "T": [0.25, 0.25, 0.25],
        "days_to_exp": [91, 91, 91],
    })


@pytest.fixture
def chain_wide_spreads():
    """DataFrame with some rows having spread_pct > 0.5."""
    return pd.DataFrame({
        "strike": [100.0, 105.0],
        "last_price": [5.0, 3.0],
        "bid": [1.0, 2.0],
        "ask": [10.0, 3.5],  # spread_pct = 9/5.5 > 0.5 and 1.5/2.75 < 0.5
        "volume": [10, 20],
        "open_interest": [100, 200],
        "yf_iv": [0.2, 0.19],
        "mid": [5.5, 2.75],
        "spread": [9.0, 1.5],
        "spread_pct": [9.0 / 5.5, 1.5 / 2.75],  # >0.5 and <0.5
        "T": [0.25, 0.25],
        "days_to_exp": [91, 91],
    })


@pytest.fixture
def chain_zero_bid():
    """DataFrame with one row bid <= 0."""
    return pd.DataFrame({
        "strike": [100.0, 105.0],
        "last_price": [5.0, 3.0],
        "bid": [0.0, 2.9],
        "ask": [5.2, 3.1],
        "volume": [10, 20],
        "open_interest": [100, 200],
        "yf_iv": [0.2, 0.19],
        "mid": [2.6, 3.0],
        "spread": [5.2, 0.2],
        "spread_pct": [0.5, 0.067],
        "T": [0.25, 0.25],
        "days_to_exp": [91, 91],
    })


@pytest.fixture
def synthetic_chain_for_iv():
    """Cleaned-chain-like DataFrame with BS prices at sigma=0.2 for compute_iv_surface."""
    S, r, q, sigma = 100.0, 0.05, 0.0, 0.2
    exp = "2025-06-20"
    T = 0.25
    days = 91
    strikes = [95.0, 100.0, 105.0]
    rows = []
    for k in strikes:
        c = black_scholes_call(S, k, T, r, sigma, q)
        p = black_scholes_put(S, k, T, r, sigma, q)
        rows.append({
            "strike": k, "last_price": c, "bid": c - 0.05, "ask": c + 0.05,
            "volume": 100, "open_interest": 500, "yf_iv": np.nan,
            "mid": c, "spread": 0.1, "spread_pct": 0.02,
            "T": T, "days_to_exp": days, "moneyness": k / S, "spot": S,
            "option_type": "call", "expiry": exp, "ticker": "TEST",
        })
    return pd.DataFrame(rows)


@pytest.fixture
def surface_with_skew():
    """Surface-like DataFrame where OTM puts have higher IV than OTM calls (positive skew)."""
    return pd.DataFrame({
        "expiry": ["2025-06-20"] * 6,
        "days_to_exp": [91] * 6,
        "option_type": ["put", "put", "put", "call", "call", "call"],
        "strike": [95, 98, 100, 100, 102, 105],
        "iv": [0.28, 0.24, 0.20, 0.20, 0.22, 0.26],  # puts richer
        "delta_approx": [-0.35, -0.28, -0.20, 0.20, 0.28, 0.35],
        "moneyness": [0.95, 0.98, 1.0, 1.0, 1.02, 1.05],
    })


# -----------------------------------------------------------------------------
# fetcher.py
# -----------------------------------------------------------------------------


def test_clean_chain_removes_crossed_markets(chain_with_crossed_markets):
    """clean_chain removes rows where ask < bid."""
    df = chain_with_crossed_markets
    out = clean_chain(df)
    assert len(out) == 2
    assert not (out["bid"] > out["ask"]).any()


def test_clean_chain_removes_wide_spreads(chain_wide_spreads):
    """Rows with spread_pct > 0.5 are removed."""
    df = pd.DataFrame({
        "strike": [100.0, 105.0],
        "last_price": [5.0, 3.0],
        "bid": [1.0, 2.0],
        "ask": [10.0, 3.5],
        "mid": [5.5, 2.75],
        "spread": [9.0, 1.5],
        "spread_pct": [0.99, 0.2],
        "T": [0.25, 0.25],
        "days_to_exp": [91, 91],
    })
    out = clean_chain(df)
    assert len(out) == 1
    assert out["spread_pct"].iloc[0] <= 0.5


def test_clean_chain_removes_zero_bid(chain_zero_bid):
    """Rows with bid <= 0 are removed."""
    df = pd.DataFrame({
        "strike": [100.0, 105.0],
        "last_price": [5.0, 3.0],
        "bid": [0.0, 2.9],
        "ask": [5.2, 3.1],
        "mid": [2.6, 3.0],
        "spread": [5.2, 0.2],
        "spread_pct": [0.4, 0.067],
        "T": [0.25, 0.25],
        "days_to_exp": [91, 91],
    })
    out = clean_chain(df)
    assert len(out) == 1
    assert (out["bid"] > 0).all()


def test_get_options_chain_adds_computed_columns(monkeypatch):
    """Mock yfinance so get_options_chain returns DataFrame with mid, spread, T, moneyness."""
    import src.data.fetcher as fetcher_mod
    fake_calls = pd.DataFrame({
        "strike": [100.0],
        "lastPrice": [5.0],
        "bid": [4.9],
        "ask": [5.1],
        "volume": [10],
        "openInterest": [100],
        "impliedVolatility": [0.2],
    })
    fake_puts = pd.DataFrame({
        "strike": [100.0],
        "lastPrice": [4.0],
        "bid": [3.9],
        "ask": [4.1],
        "volume": [5],
        "openInterest": [50],
        "impliedVolatility": [0.19],
    })

    class FakeChain:
        calls = fake_calls
        puts = fake_puts

    def fake_option_chain(date):
        return FakeChain()

    def fake_get_price(ticker):
        return 100.0

    monkeypatch.setattr(fetcher_mod, "get_underlying_price", fake_get_price)
    class FakeTicker:
        options = ("2025-09-19",)
        def option_chain(self, date):
            return FakeChain()
    from types import SimpleNamespace
    monkeypatch.setattr(fetcher_mod, "yf", SimpleNamespace(Ticker=lambda t: FakeTicker()))
    out = get_options_chain("SPY", expiry="2025-09-19", option_type="both", min_volume=0, min_open_interest=0)
    assert "mid" in out.columns
    assert "spread" in out.columns
    assert "T" in out.columns
    assert "moneyness" in out.columns
    assert "spot" in out.columns


# -----------------------------------------------------------------------------
# rates.py
# -----------------------------------------------------------------------------


def test_get_rate_for_expiry_selects_correct_maturity(monkeypatch):
    """days=30 -> 1m, days=60 -> 3m, etc."""
    # Clear cache to avoid previous test pollution
    _rate_cache.clear()
    got = {}
    def capture(maturity):
        got["maturity"] = maturity
        return 0.05
    monkeypatch.setattr("src.data.rates.get_risk_free_rate", capture)
    get_rate_for_expiry(30)
    assert got["maturity"] == "1m"
    get_rate_for_expiry(60)
    assert got["maturity"] == "3m"
    get_rate_for_expiry(150)
    assert got["maturity"] == "6m"
    get_rate_for_expiry(200)
    assert got["maturity"] == "1y"
    get_rate_for_expiry(400)
    assert got["maturity"] == "2y"
    get_rate_for_expiry(700)
    assert got["maturity"] == "2y"
    get_rate_for_expiry(1000)
    assert got["maturity"] == "5y"
    get_rate_for_expiry(2000)
    assert got["maturity"] == "10y"


def test_get_risk_free_rate_fallback_on_network_error(monkeypatch):
    """When request fails, return 0.05 with a warning."""
    _rate_cache.clear()
    def raise_err(*args, **kwargs):
        raise OSError("network error")
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen", raise_err)
    with pytest.warns(UserWarning, match="Failed to fetch"):
        r = get_risk_free_rate("3m")
    assert r == 0.05
    assert isinstance(r, float)


# -----------------------------------------------------------------------------
# vol_surface.py
# -----------------------------------------------------------------------------


def test_compute_iv_surface_recovers_known_iv(synthetic_chain_for_iv):
    """Synthetic chain with BS prices at sigma=0.2; IV should recover ~0.2."""
    df = synthetic_chain_for_iv
    out = compute_iv_surface(df, r=0.05, q=0.0, use_rate_for_expiry=False)
    assert len(out) >= 1
    assert (out["iv"] - 0.2).abs().max() < 0.001


def test_get_atm_iv_returns_one_row_per_expiry(synthetic_chain_for_iv):
    """Output has exactly one row per unique expiry."""
    df = synthetic_chain_for_iv
    surface = compute_iv_surface(df, r=0.05, use_rate_for_expiry=False)
    atm = get_atm_iv(surface)
    assert len(atm) == surface["expiry"].nunique()
    assert "atm_iv" in atm.columns


def test_get_skew_positive_for_equity_index_like_data(surface_with_skew):
    """Synthetic surface with OTM puts > OTM calls IV -> skew > 0."""
    skew_df = get_skew(surface_with_skew, delta_target=0.25)
    assert len(skew_df) >= 1
    assert (skew_df["skew"] > 0).all()


# -----------------------------------------------------------------------------
# realized_vol.py
# -----------------------------------------------------------------------------


def test_compute_realized_vol_constant_prices_zero_vol():
    """Flat price series -> realized vol 0 (after window)."""
    prices = pd.Series(100.0, index=pd.date_range("2024-01-01", periods=50, freq="B"))
    rv = compute_realized_vol(prices, window=20)
    assert rv.notna().any()
    assert (rv.dropna() == 0).all()


def test_compute_realized_vol_known_value():
    """Log-normal prices with known sigma; rolling rv converges to that sigma."""
    np.random.seed(42)
    n = 500
    sigma_true = 0.25
    trading_days = 252
    # Daily vol = sigma_true / sqrt(252) so annualized std of daily returns = sigma_true
    sigma_daily = sigma_true / np.sqrt(trading_days)
    mu_daily = -0.5 * sigma_daily ** 2
    log_ret = np.random.normal(mu_daily, sigma_daily, n)
    prices = pd.Series(100 * np.exp(np.cumsum(log_ret)), index=pd.date_range("2022-01-01", periods=n, freq="B"))
    rv = compute_realized_vol(prices, window=252, trading_days=trading_days)
    last_rv = rv.iloc[-1]
    assert abs(last_rv - sigma_true) < 0.05


def test_compute_vol_cone_shape():
    """Output has len(windows) rows and expected columns."""
    prices = pd.Series(
        np.exp(np.cumsum(np.random.randn(300) * 0.01)),
        index=pd.date_range("2023-01-01", periods=300, freq="B"),
    )
    windows = [10, 21, 30]
    cone = compute_vol_cone(prices, windows=windows, percentiles=[10, 50, 90])
    assert len(cone) == len(windows)
    assert "p10" in cone.columns and "p50" in cone.columns and "p90" in cone.columns
    assert "current" in cone.columns


def test_compute_vrp_aligns_dates():
    """Two series with offset dates -> vrp only at overlapping dates."""
    idx1 = pd.date_range("2024-01-01", periods=10, freq="B")
    idx2 = pd.date_range("2024-01-05", periods=10, freq="B")
    rv = pd.Series(0.2, index=idx1)
    iv = pd.Series(0.25, index=idx2)
    vrp = compute_vrp(rv, iv)
    common = idx1.intersection(idx2)
    assert len(vrp) == len(common)
    np.testing.assert_allclose(vrp.values, 0.05)
