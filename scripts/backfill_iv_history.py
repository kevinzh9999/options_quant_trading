#!/usr/bin/env python3
"""
backfill_iv_history.py
----------------------
从 options_daily 回算历史 ATM IV / RV / RR / IV Term Structure，
写入 daily_model_output。

用法：
    python scripts/backfill_iv_history.py                    # 补全所有缺失日
    python scripts/backfill_iv_history.py --date 20250601    # 单日
    python scripts/backfill_iv_history.py --force             # 覆盖已有数据
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

warnings.filterwarnings("ignore")

from data.storage.db_manager import DBManager
from config.config_loader import ConfigLoader

# Reuse existing calculation functions
from scripts.portfolio_analysis import (
    _parse_mo, get_atm_iv, _build_expire_map, RISK_FREE,
)
from models.pricing.forward_price import calc_implied_forward
from models.pricing.implied_vol import calc_implied_vol

# ---------------------------------------------------------------------------
# IV Term Structure: 近月IV - 远月IV
# ---------------------------------------------------------------------------

def calc_iv_term_spread(
    chain_df: pd.DataFrame,
    trade_date: str,
    expire_map: dict,
    spot: float,
) -> Optional[float]:
    """Calculate IV term spread = near-month ATM IV - far-month ATM IV.

    Positive = inverted (near > far, short-term panic).
    Negative = normal contango.
    """
    today_ts = pd.Timestamp(trade_date)
    sorted_months = sorted(chain_df["expire_month"].unique())

    # Find valid months with DTE >= 7
    valid = []
    for em in sorted_months:
        ed = expire_map.get(em, "")
        if not ed:
            continue
        dte = (pd.Timestamp(ed) - today_ts).days
        if dte >= 7:
            valid.append((em, ed, dte))

    if len(valid) < 2:
        return None

    # Near = shortest DTE, Far = second shortest
    near_month, near_ed, near_dte = valid[0]
    far_month, far_ed, far_dte = valid[1]

    near_T = max(near_dte / 365.0, 1 / 365)
    far_T = max(far_dte / 365.0, 1 / 365)

    def _atm_iv_for_month(month, T):
        sub = chain_df[chain_df["expire_month"] == month].copy()
        if sub.empty:
            return None
        fwd, n = calc_implied_forward(sub, T, RISK_FREE, spot)
        strikes = sub["exercise_price"].unique()
        atm_k = float(strikes[np.argmin(np.abs(strikes - fwd))])
        atm_rows = sub[sub["exercise_price"] == atm_k]
        ivs = []
        for _, row in atm_rows.iterrows():
            mkt = float(row["close"])
            if mkt <= 0:
                continue
            cp = str(row["call_put"])
            try:
                iv = calc_implied_vol(mkt, fwd, atm_k, T, RISK_FREE, cp)
                if iv and 0.01 < iv < 5.0:
                    ivs.append(iv)
            except Exception:
                pass
        return np.mean(ivs) if ivs else None

    near_iv = _atm_iv_for_month(near_month, near_T)
    far_iv = _atm_iv_for_month(far_month, far_T)

    if near_iv is not None and far_iv is not None:
        return near_iv - far_iv
    return None


# ---------------------------------------------------------------------------
# RR calculation (reuse vol_monitor logic)
# ---------------------------------------------------------------------------

def calc_rr_for_date(
    chain_df: pd.DataFrame,
    trade_date: str,
    expire_map: dict,
    spot: float,
) -> Optional[float]:
    """Calculate 25D Risk Reversal for a given date."""
    try:
        from scripts.vol_monitor import calc_skew_table, calc_rr_bf

        today_ts = pd.Timestamp(trade_date)
        sorted_months = sorted(chain_df["expire_month"].unique())

        # Find near month (DTE >= 14)
        near_month = None
        for em in sorted_months:
            ed = expire_map.get(em, "")
            if not ed:
                continue
            dte = (pd.Timestamp(ed) - today_ts).days
            if dte >= 14:
                near_month = em
                break

        if not near_month:
            return None

        near_ed = expire_map.get(near_month, "")
        skew_df = calc_skew_table(chain_df, near_month, spot, near_ed, trade_date)
        rr, _ = calc_rr_bf(skew_df)
        return rr if rr != 0 else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main backfill
# ---------------------------------------------------------------------------

def build_chain_df(db: DBManager, trade_date: str, expire_map: dict) -> pd.DataFrame:
    """Build chain_df from options_daily for a given date."""
    df = db.query_df(
        f"SELECT ts_code, close, settle, volume, oi "
        f"FROM options_daily "
        f"WHERE ts_code LIKE 'MO%' AND trade_date='{trade_date}' AND close > 0"
    )
    if df is None or df.empty:
        return pd.DataFrame()

    records = []
    for _, row in df.iterrows():
        parsed = _parse_mo(str(row["ts_code"]))
        if parsed is None:
            continue
        expire_month, cp, strike = parsed
        expire_date = expire_map.get(expire_month, "")
        if not expire_date:
            continue
        records.append({
            "ts_code": row["ts_code"],
            "expire_month": expire_month,
            "expire_date": expire_date,
            "call_put": cp,
            "exercise_price": strike,
            "close": float(row["close"]),
            "volume": float(row["volume"]) if row["volume"] else 0.0,
            "oi": float(row["oi"]) if row["oi"] else 0.0,
        })

    return pd.DataFrame(records) if records else pd.DataFrame()


def backfill_one_day(
    db: DBManager,
    trade_date: str,
    expire_map: dict,
    spot: float,
    futures_price: float,
    rv_5d: Optional[float],
    rv_20d: Optional[float],
    hurst_60d: Optional[float],
) -> dict:
    """Backfill all indicators for one day. Returns dict of computed values."""
    result = {"trade_date": trade_date, "underlying": "IM"}

    chain_df = build_chain_df(db, trade_date, expire_map)
    if chain_df.empty:
        return result

    # 1. ATM IV (structural, PCP Forward)
    try:
        # Suppress prints from get_atm_iv
        import io, contextlib
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            atm_iv, _ = get_atm_iv(spot, chain_df, trade_date,
                                    futures_prices=None)
        result["atm_iv"] = round(atm_iv, 6)
    except Exception:
        pass

    # 2. Market ATM IV (futures-price based, for VRP)
    try:
        from scripts.vol_monitor import calc_market_atm_iv
        sorted_months = sorted(chain_df["expire_month"].unique())
        near_month = None
        today_ts = pd.Timestamp(trade_date)
        for em in sorted_months:
            ed = expire_map.get(em, "")
            if ed and (pd.Timestamp(ed) - today_ts).days >= 14:
                near_month = em
                break
        if near_month:
            near_ed = expire_map.get(near_month, "")
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                mkt_iv = calc_market_atm_iv(chain_df, near_month,
                                            futures_price, near_ed, trade_date)
            if mkt_iv and mkt_iv > 0:
                result["atm_iv_market"] = round(mkt_iv, 6)
    except Exception:
        pass

    # 3. IV Term Structure
    try:
        spread = calc_iv_term_spread(chain_df, trade_date, expire_map, spot)
        if spread is not None:
            result["iv_term_spread"] = round(spread, 6)
    except Exception:
        pass

    # 4. 25D Risk Reversal
    try:
        rr = calc_rr_for_date(chain_df, trade_date, expire_map, spot)
        if rr is not None:
            result["rr_25d"] = round(rr, 6)
    except Exception:
        pass

    # 5. RV values
    if rv_5d is not None:
        result["realized_vol_5d"] = round(rv_5d, 6)
    if rv_20d is not None:
        result["realized_vol_20d"] = round(rv_20d, 6)

    # 6. VRP = market_iv - blended_rv (if both available)
    mkt_iv = result.get("atm_iv_market")
    if mkt_iv and rv_5d and rv_20d:
        blended_rv = 0.5 * rv_5d + 0.5 * rv_20d  # simplified (no GARCH for historical)
        result["vrp"] = round(mkt_iv - blended_rv, 6)

    # 7. Hurst
    if hurst_60d is not None:
        result["hurst_60d"] = round(hurst_60d, 4)

    return result


def main():
    parser = argparse.ArgumentParser(description="回算历史IV/RV/RR数据")
    parser.add_argument("--date", default="", help="单日YYYYMMDD，空=全部缺失日")
    parser.add_argument("--force", action="store_true", help="覆盖已有数据")
    args = parser.parse_args()

    db = DBManager(ConfigLoader().get_db_path())

    # Get all dates from index_min
    r = db.query_df(
        "SELECT DISTINCT substr(datetime,1,10) as d "
        "FROM index_min WHERE symbol='000852' AND period=300 ORDER BY d"
    )
    all_dates = [d.replace('-', '') for d in r['d'].tolist()]

    if args.date:
        dates_to_fill = [args.date]
    elif args.force:
        dates_to_fill = all_dates
    else:
        # Find dates missing atm_iv
        existing = db.query_df(
            "SELECT trade_date FROM daily_model_output "
            "WHERE atm_iv > 0 AND underlying='IM'"
        )
        existing_dates = set(existing['trade_date'].tolist()) if existing is not None else set()
        dates_to_fill = [d for d in all_dates if d not in existing_dates]

    print(f"Backfilling {len(dates_to_fill)} dates ({dates_to_fill[0]}~{dates_to_fill[-1]})")

    # Build expire_map
    expire_map = _build_expire_map(db)
    print(f"Expire map: {len(expire_map)} months")

    # Load daily data for RV/spot calculation
    daily = db.query_df(
        "SELECT trade_date, close FROM index_daily "
        "WHERE ts_code='000852.SH' ORDER BY trade_date"
    )
    daily['close'] = daily['close'].astype(float)

    # Load futures data
    futures = db.query_df(
        "SELECT trade_date, close FROM futures_daily "
        "WHERE ts_code='IM.CFX' ORDER BY trade_date"
    )
    if futures is not None:
        futures['close'] = futures['close'].astype(float)
        fut_map = dict(zip(futures['trade_date'], futures['close']))
    else:
        fut_map = {}

    # Hurst calculation
    try:
        from scripts.morning_briefing import _calc_hurst
        has_hurst = True
    except Exception:
        has_hurst = False

    success = 0
    errors = 0

    for i, td in enumerate(dates_to_fill):
        # Spot price
        sub = daily[daily['trade_date'] <= td]
        if len(sub) < 25:
            continue
        spot = float(sub.iloc[-1]['close'])
        futures_price = fut_map.get(td, spot)

        # RV_5d and RV_20d
        closes = sub['close'].values
        if len(closes) >= 6:
            log_rets = np.diff(np.log(closes[-6:]))
            rv_5d = float(np.std(log_rets, ddof=1) * np.sqrt(252))
        else:
            rv_5d = None

        if len(closes) >= 21:
            log_rets_20 = np.diff(np.log(closes[-21:]))
            rv_20d = float(np.std(log_rets_20, ddof=1) * np.sqrt(252))
        else:
            rv_20d = None

        # Hurst
        hurst = None
        if has_hurst and len(closes) >= 60:
            try:
                hurst = float(_calc_hurst(closes[-60:]))
            except Exception:
                pass

        try:
            result = backfill_one_day(
                db, td, expire_map, spot, futures_price,
                rv_5d, rv_20d, hurst,
            )

            atm_iv = result.get("atm_iv")
            if atm_iv:
                # Upsert to daily_model_output
                # Check if row exists
                existing = db.query_df(
                    f"SELECT trade_date FROM daily_model_output "
                    f"WHERE trade_date='{td}' AND underlying='IM'"
                )
                if existing is not None and not existing.empty:
                    # Update specific fields
                    sets = []
                    params = []
                    for key in ["atm_iv", "atm_iv_market", "realized_vol_5d",
                                "realized_vol_20d", "vrp", "rr_25d",
                                "iv_term_spread", "hurst_60d"]:
                        if key in result and result[key] is not None:
                            # iv_term_spread might not exist as column yet
                            sets.append(f"{key}=?")
                            params.append(result[key])
                    if sets:
                        params.extend([td, "IM"])
                        try:
                            db._conn.execute(
                                f"UPDATE daily_model_output SET {','.join(sets)} "
                                f"WHERE trade_date=? AND underlying=?",
                                params
                            )
                            db._conn.commit()
                        except Exception as e:
                            # Column might not exist, try without iv_term_spread
                            if "iv_term_spread" in str(e):
                                sets2 = [s for s in sets if "iv_term_spread" not in s]
                                params2 = [p for s, p in zip(sets, params[:-2])
                                           if "iv_term_spread" not in s]
                                params2.extend([td, "IM"])
                                db._conn.execute(
                                    f"UPDATE daily_model_output SET {','.join(sets2)} "
                                    f"WHERE trade_date=? AND underlying=?",
                                    params2
                                )
                                db._conn.commit()
                            else:
                                raise
                else:
                    # Insert new row
                    db._conn.execute(
                        "INSERT OR IGNORE INTO daily_model_output "
                        "(trade_date, underlying, atm_iv, atm_iv_market, "
                        "realized_vol_20d, vrp, rr_25d, hurst_60d) "
                        "VALUES (?,?,?,?,?,?,?,?)",
                        (td, "IM", result.get("atm_iv"),
                         result.get("atm_iv_market"),
                         result.get("realized_vol_20d"),
                         result.get("vrp"), result.get("rr_25d"),
                         result.get("hurst_60d"))
                    )
                    db._conn.commit()

                success += 1
                rr_str = f"RR={result.get('rr_25d', 'N/A')}" if result.get('rr_25d') else "RR=N/A"
                ts_str = f"TS={result.get('iv_term_spread', 'N/A'):.4f}" if result.get('iv_term_spread') else "TS=N/A"
                if (i + 1) % 10 == 0 or i < 3:
                    print(f"  [{i+1}/{len(dates_to_fill)}] {td}: "
                          f"IV={atm_iv*100:.1f}% "
                          f"mktIV={result.get('atm_iv_market',0)*100:.1f}% "
                          f"RV5={rv_5d*100:.1f}% " if rv_5d else "",
                          f"{rr_str} {ts_str}")
            else:
                errors += 1
                if (i + 1) % 20 == 0:
                    print(f"  [{i+1}/{len(dates_to_fill)}] {td}: no ATM IV")

        except Exception as e:
            errors += 1
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(dates_to_fill)}] {td}: ERROR {e}")

    print(f"\nDone: {success} success, {errors} errors out of {len(dates_to_fill)} dates")

    # Summary stats
    r = db.query_df(
        "SELECT COUNT(*) cnt, "
        "SUM(CASE WHEN atm_iv > 0 THEN 1 ELSE 0 END) as iv_cnt, "
        "SUM(CASE WHEN rr_25d IS NOT NULL AND rr_25d != 0 THEN 1 ELSE 0 END) as rr_cnt, "
        "SUM(CASE WHEN vrp IS NOT NULL THEN 1 ELSE 0 END) as vrp_cnt "
        "FROM daily_model_output WHERE underlying='IM'"
    )
    if r is not None and not r.empty:
        print(f"\ndaily_model_output coverage:")
        print(f"  Total rows: {int(r.iloc[0]['cnt'])}")
        print(f"  ATM IV: {int(r.iloc[0]['iv_cnt'])}")
        print(f"  RR 25D: {int(r.iloc[0]['rr_cnt'])}")
        print(f"  VRP: {int(r.iloc[0]['vrp_cnt'])}")


if __name__ == "__main__":
    main()
