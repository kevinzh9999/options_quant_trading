"""
forward_price.py
----------------
Put-Call Parity implied forward price calculation for futures options.

PCP for futures options: C - P = e^(-rT) * (F - K)
=> F = K + (C - P) * e^(rT)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calc_implied_forward(
    chain_df: pd.DataFrame,
    T: float,
    r: float,
    futures_close: float,
    window_pct: float = 0.10,
) -> tuple[float, int]:
    """
    Estimate implied forward price from put-call parity across multiple strikes.

    Parameters
    ----------
    chain_df : pd.DataFrame
        Option chain for a single expiry. Must have columns:
        exercise_price, call_put ('C'/'CALL'/'P'/'PUT'), close, volume.
    T : float
        Time to expiry in years.
    r : float
        Risk-free rate (annualized, continuous).
    futures_close : float
        Futures close price for this expiry, used to define the search window
        and as fallback if no valid PCP pairs are found.
    window_pct : float
        Strike search window as fraction of futures_close (default ±10%).

    Returns
    -------
    (implied_fwd, n_estimates) : tuple[float, int]
        implied_fwd  — median PCP-implied forward; equals futures_close if n_estimates == 0.
        n_estimates  — number of strike pairs used; 0 means fallback to futures_close.
    """
    discount = np.exp(r * T)  # e^(rT)

    # Build K → {cp_key: (price, volume)} mapping
    cp_prices: dict[float, dict[str, tuple[float, float]]] = {}
    for _, row in chain_df.iterrows():
        k = float(row["exercise_price"])
        cp = str(row["call_put"])
        px = float(row["close"] or 0)
        vol = float(row.get("volume", 0) or 0)
        if px > 0:
            cp_prices.setdefault(k, {})[cp] = (px, vol)

    lo = futures_close * (1.0 - window_pct)
    hi = futures_close * (1.0 + window_pct)

    fwd_estimates: list[float] = []
    for k, sides in cp_prices.items():
        if not (lo <= k <= hi):
            continue
        c_key = next((x for x in sides if x in ("C", "CALL")), None)
        p_key = next((x for x in sides if x in ("P", "PUT")), None)
        if c_key is None or p_key is None:
            continue
        c_px, _ = sides[c_key]
        p_px, _ = sides[p_key]
        fwd_estimates.append(k + (c_px - p_px) * discount)

    if fwd_estimates:
        return float(np.median(fwd_estimates)), len(fwd_estimates)
    return futures_close, 0
