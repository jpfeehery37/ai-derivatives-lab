"""
Phase 2 demo: market data infrastructure end-to-end.
Uses print only (no matplotlib). Run from project root: python -m scripts.phase2_demo
"""


def run_phase2_demo():
    """Run the full Phase 2 demo: SPY spot, rates, chain, IV surface, realized vol, VRP, vol cone."""
    import pandas as pd
    from src.data.fetcher import (
        get_underlying_price,
        get_options_chain,
        clean_chain,
        get_price_history,
    )
    from src.data.rates import get_risk_free_rate, get_rate_for_expiry
    from src.data.vol_surface import compute_iv_surface, summarize_surface
    from src.data.realized_vol import (
        compute_realized_vol,
        compute_vol_cone,
        compute_vrp,
    )

    ticker = "SPY"
    hist = None
    surface = None
    summary = {}
    current_rv = float("nan")
    print("=" * 60)
    print("Phase 2 — Market Data Infrastructure Demo")
    print("=" * 60)

    # 1. SPY spot
    try:
        spot = get_underlying_price(ticker)
        print(f"\n--- 1. SPY spot ---\nSPY spot: ${spot:.2f}")
    except Exception as e:
        print(f"\n--- 1. SPY spot ---\nFailed: {e}")
        return

    # 2. Risk-free rate (3m)
    try:
        r_3m = get_risk_free_rate("3m")
        print(f"\n--- 2. Risk-free rate ---\nRisk-free rate (3m): {r_3m * 100:.2f}%")
    except Exception as e:
        print(f"\n--- 2. Risk-free rate ---\nFailed: {e}")
        r_3m = 0.05

    # 3. Fetch and clean options chain
    try:
        print("\n--- 3. Options chain ---")
        chain = get_options_chain(ticker, expiry=None, option_type="both", min_volume=10, min_open_interest=100)
        print(f"Chain shape before clean: {chain.shape[0]} rows")
        chain_clean = clean_chain(chain)
        print(f"Chain shape after clean: {chain_clean.shape[0]} rows")
    except Exception as e:
        print(f"Failed: {e}")
        return

    # 4. IV surface and summary
    try:
        print("\n--- 4. IV surface ---")
        surface = compute_iv_surface(chain_clean, r=None, q=0.0, use_rate_for_expiry=True)
        summary = summarize_surface(surface)
        print(f"Ticker: {summary['ticker']}")
        print(f"Spot: {summary['spot']}")
        print(f"n_options: {summary['n_options']}")
        print(f"expirations: {summary['expirations']}")
        print(f"IV min: {summary['iv_min']:.4f}, mean: {summary['iv_mean']:.4f}, max: {summary['iv_max']:.4f}")
        print("\nATM IV term structure:")
        print(summary["atm_iv_term"].to_string(index=False))
        print("\nSkew (25-delta put - call):")
        print(summary["skew_term"].to_string(index=False))
    except Exception as e:
        print(f"Failed: {e}")

    # 5. Price history and 30d realized vol
    try:
        print("\n--- 5. Price history & realized vol ---")
        hist = get_price_history(ticker, period="1y", interval="1d")
        current_rv = float("nan")
        rv = compute_realized_vol(hist["Close"], window=30, trading_days=252, annualize=True)
        current_rv = rv.iloc[-1] if rv.notna().any() else float("nan")
        print(f"Current 30d realized vol: {current_rv * 100:.2f}%")
        print("Last 5 realized vol values:")
        print(rv.dropna().tail(5).to_string())
    except Exception as e:
        print(f"Failed: {e}")
        rv = None

    # 6. ATM IV vs 30d RV and VRP
    try:
        print("\n--- 6. ATM IV vs 30d RV ---")
        atm_iv_pct = float("nan")
        if surface is not None and not surface.empty and summary.get("atm_iv_term") is not None:
            atm_df = summary["atm_iv_term"]
            if not atm_df.empty:
                # Nearest expiry: first row (sorted by days_to_exp)
                atm_iv_pct = atm_df["atm_iv"].iloc[0] * 100
        rv_pct = current_rv * 100 if pd.notna(current_rv) else float("nan")
        vrp_pct = atm_iv_pct - rv_pct if pd.notna(atm_iv_pct) and pd.notna(rv_pct) else float("nan")
        print(f"ATM IV: {atm_iv_pct:.2f}%  |  30d RV: {rv_pct:.2f}%  |  VRP: {vrp_pct:.2f}%")
    except Exception as e:
        print(f"Failed: {e}")

    # 7. Volatility cone
    try:
        print("\n--- 7. Volatility cone ---")
        if hist is None or hist.empty:
            hist = get_price_history(ticker, period="1y", interval="1d")
        cone = compute_vol_cone(hist["Close"], trading_days=252)
        print(cone.to_string())
    except Exception as e:
        print(f"Failed: {e}")

    print("\n" + "=" * 60)
    print("Phase 2 demo complete.")
    print("=" * 60)
