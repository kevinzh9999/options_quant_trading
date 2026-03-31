"""
cffex_calendar.py
-----------------
CFFEX index futures contract month utilities.

Tushare IML continuous contract naming (as of 2026):
  IM.CFX / IML.CFX = current month main contract (当月)
  IML1.CFX         = next month (次月)
  IML2.CFX         = current quarter end month (当季: nearest 3/6/9/12 >= next month)
  IML3.CFX         = next quarter end month (下季)

Example (2026-03-17, MO2603 not yet expired):
  active months: ['2603', '2604', '2606', '2609']
  IM.CFX  = 2603 → 8014.0
  IML1.CFX = 2604 → 7951.0
  IML2.CFX = 2606 → 7770.8
  IML3.CFX = 2609 → 7564.2

Option expiry → IM futures mapping:
  MO2603 → 2603 → IM.CFX   = 8014.0  (exact match)
  MO2604 → 2604 → IML1.CFX = 7951.0  (exact match)
  MO2605 → 2606 → IML2.CFX = 7770.8  (no 2605, nearest >= 2605 is 2606)
  MO2606 → 2606 → IML2.CFX = 7770.8  (exact match)
  MO2609 → 2609 → IML3.CFX = 7564.2  (exact match)
"""

from __future__ import annotations

import re
import pandas as pd


# IML index → Tushare ts_code
# index 0 = current month = IM.CFX (also IML.CFX, same price)
# index 1 = next month    = IML1.CFX
# index 2 = cur quarter   = IML2.CFX
# index 3 = next quarter  = IML3.CFX
_IML_CODES = ["IM.CFX", "IML1.CFX", "IML2.CFX", "IML3.CFX"]


def _yymm(year: int, month: int) -> str:
    """Convert year/month to YYMM string, handling month overflow."""
    while month > 12:
        year += 1
        month -= 12
    return f"{year % 100:02d}{month:02d}"


def active_im_months(trade_date_str: str) -> list[str]:
    """
    Return the 4 active CFFEX IM futures contract months for a given trade date.

    CFFEX rules: current month, next month, current quarter end, next quarter end.
    Quarter end months: 3, 6, 9, 12.

    Returns YYMM strings in order matching _IML_CODES:
      [cur_month, next_month, cur_quarter_end, next_quarter_end]

    Example: '20260317' → ['2603', '2604', '2606', '2609']
    """
    dt = pd.Timestamp(str(trade_date_str))
    y, m = dt.year, dt.month

    current = _yymm(y, m)
    next_m  = _yymm(y, m + 1)
    q_end   = ((m - 1) // 3 + 1) * 3   # nearest quarter end month (3/6/9/12)
    cur_q   = _yymm(y, q_end)
    next_q  = _yymm(y, q_end + 3)
    next_q2 = _yymm(y, q_end + 6)  # extra candidate for dedup (e.g. March: cur_q==current)

    seen, months = set(), []
    for code in [current, next_m, cur_q, next_q, next_q2]:
        if code not in seen:
            seen.add(code)
            months.append(code)
        if len(months) == 4:
            break
    return months


def get_im_futures_prices(db, trade_date: str) -> dict[str, float]:
    """
    Query all active IM futures contract prices for trade_date.

    Returns {yymm: close_price}, e.g.:
      {'2603': 8014.0, '2604': 7951.0, '2606': 7770.8, '2609': 7564.2}

    Strategy:
    1. Try specific contracts (IM2603.CFX, IM2604.CFX, ...) — more accurate.
    2. Fall back to IML continuous contracts (IM.CFX, IML1.CFX, IML2.CFX, IML3.CFX)
       using correct index mapping: IM.CFX=current, IML1=next, IML2=cur_qtr, IML3=next_qtr.
    """
    # Strategy 1: specific contract codes
    try:
        df = db.query_df(
            f"SELECT ts_code, close FROM futures_daily "
            f"WHERE ts_code LIKE 'IM%' AND ts_code NOT LIKE 'IML%' "
            f"AND close IS NOT NULL AND close > 0 "
            f"AND trade_date = '{trade_date}'"
        )
        if df is not None and not df.empty:
            prices: dict[str, float] = {}
            for _, row in df.iterrows():
                m = re.match(r"IM(\d{4})\.CFX", str(row["ts_code"]))
                if m:
                    prices[m.group(1)] = float(row["close"])
            if prices:
                return prices
    except Exception:
        pass

    # Strategy 2: IML continuous contracts with corrected index mapping
    active = active_im_months(trade_date)
    prices = {}
    for i, iml_code in enumerate(_IML_CODES):
        if i >= len(active):
            break
        try:
            row = db.query_df(
                f"SELECT close FROM futures_daily "
                f"WHERE ts_code='{iml_code}' AND trade_date='{trade_date}'"
            )
            if row is not None and not row.empty:
                prices[active[i]] = float(row["close"].iloc[0])
        except Exception:
            pass
    return prices


def map_expiry_to_futures_price(
    expire_months: list[str],
    im_prices: dict[str, float],
    fallback: float,
) -> dict[str, tuple[str, float]]:
    """
    Map option expiry months to the nearest available IM futures price.

    For each expire_month:
    1. Exact match in im_prices → use it directly
    2. No exact match → use the nearest IM contract month >= expire_month
    3. Nothing found → fallback (main contract price)

    Returns {expire_month: (matched_im_month, price)}
    Example:
      expire_months=['2603','2604','2605','2606','2609']
      im_prices={'2603':8014, '2604':7951, '2606':7771, '2609':7564}
      → {'2603':('2603',8014), '2604':('2604',7951),
         '2605':('2606',7771), '2606':('2606',7771), '2609':('2609',7564)}
    """
    sorted_im = sorted(im_prices.keys())
    result: dict[str, tuple[str, float]] = {}
    for opt_m in expire_months:
        if opt_m in im_prices:
            result[opt_m] = (opt_m, im_prices[opt_m])
        else:
            nearest = next((m for m in sorted_im if m >= opt_m), None)
            if nearest:
                result[opt_m] = (nearest, im_prices[nearest])
            else:
                # Fallback: use the furthest available contract
                if sorted_im:
                    furthest = sorted_im[-1]
                    result[opt_m] = (furthest, im_prices[furthest])
                else:
                    result[opt_m] = ("fallback", fallback)
    return result
