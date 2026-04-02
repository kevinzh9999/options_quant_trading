#!/usr/bin/env python3
"""
style_spread_research.py
------------------------
研究 IM-IH style spread 作为日内信号质量过滤器的效果。

方法：
1. 对每笔交易记录开仓时刻的 style spread（IM return - IH return，用 completed bar）
2. 测试不同过滤阈值对 PnL / WR 的影响
3. 稳健性检验：最优阈值 ±20% 扰动

Usage:
    python scripts/style_spread_research.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data.storage.db_manager import DBManager
from config.config_loader import ConfigLoader

# ─── Constants ──────────────────────────────────────────────────────────────
IM_MULT = 200
IC_MULT = 200

DATES = [
    "20260204","20260205","20260206","20260209","20260210","20260211",
    "20260212","20260213","20260225","20260226","20260227","20260302",
    "20260303","20260304","20260305","20260306","20260309","20260310",
    "20260311","20260212","20260313","20260316","20260317","20260318",
    "20260319","20260320","20260323","20260324","20260325","20260326",
    "20260327","20260328","20260401","20260402",
]
# deduplicate while preserving order
seen = set()
DATES = [d for d in DATES if not (d in seen or seen.add(d))]

_SPOT_SYM = {"IM": "000852", "IF": "000300", "IH": "000016", "IC": "000905"}
_SPOT_IDX = {"IM": "000852.SH", "IF": "000300.SH", "IH": "000016.SH", "IC": "000905.SH"}

# Filter threshold candidates  [lo, hi]  (units: fraction, e.g. 0.001 = 0.1%)
FILTER_THRESHOLDS = [
    ("[-0.05,+0.03]", -0.0005, +0.0003),   # strict (half original)
    ("[-0.10,+0.06]", -0.0010, +0.0006),   # medium
    ("[-0.15,+0.10]", -0.0015, +0.0010),   # loose
    ("[-0.20,+0.15]", -0.0020, +0.0015),   # very loose
]


# ─── Utility ────────────────────────────────────────────────────────────────
def _utc_to_bj(utc_str: str) -> str:
    h = int(utc_str[:2]) + 8
    if h >= 24:
        h -= 24
    return f"{h:02d}:{utc_str[3:5]}"


def _build_15m_from_5m(bar_5m: pd.DataFrame) -> pd.DataFrame:
    if len(bar_5m) < 3:
        return pd.DataFrame()
    df = bar_5m.copy()
    df["dt"] = pd.to_datetime(df["datetime"] if "datetime" in df.columns else df.index)
    df = df.set_index("dt")
    resampled = df.resample("15min", label="right", closed="right").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()
    return resampled.reset_index(drop=True)


def _calc_minutes(t1: str, t2: str) -> int:
    try:
        h1, m1 = int(t1[:2]), int(t1[3:5])
        h2, m2 = int(t2[:2]), int(t2[3:5])
        return (h2 * 60 + m2) - (h1 * 60 + m1)
    except Exception:
        return 0


# ─── Load style spread data ─────────────────────────────────────────────────
def load_style_spreads(db: DBManager) -> Dict[str, pd.DataFrame]:
    """
    Load 5min bars for IM / IH / IC / IF and compute bar-to-bar returns
    and style spreads for every bar.

    Returns dict: date_str -> DataFrame with columns:
        datetime, utc_hm, bj_time,
        im_ret, ih_ret, ic_ret,
        ss_im_ih,   # IM - IH style spread
        ss_ic_ih,   # IC - IH style spread
    """
    print("Loading spot 5min bars for IM/IH/IC ...")
    dfs: Dict[str, pd.DataFrame] = {}
    for sym in ("IM", "IH", "IC"):
        spot = _SPOT_SYM[sym]
        df = db.query_df(
            f"SELECT datetime, close FROM index_min "
            f"WHERE symbol='{spot}' AND period=300 ORDER BY datetime"
        )
        if df is None or df.empty:
            print(f"  WARNING: no data for {sym} ({spot})")
            dfs[sym] = pd.DataFrame(columns=["datetime", "close"])
        else:
            df["close"] = df["close"].astype(float)
            dfs[sym] = df

    # Align on datetime
    if any(dfs[s].empty for s in ("IM", "IH", "IC")):
        print("ERROR: missing bar data for at least one symbol")
        return {}

    merged = dfs["IM"].rename(columns={"close": "im_close"})
    merged = merged.merge(
        dfs["IH"].rename(columns={"close": "ih_close"}), on="datetime", how="inner"
    )
    merged = merged.merge(
        dfs["IC"].rename(columns={"close": "ic_close"}), on="datetime", how="inner"
    )

    # bar-to-bar return (fraction)
    merged["im_ret"] = merged["im_close"].pct_change().fillna(0.0)
    merged["ih_ret"] = merged["ih_close"].pct_change().fillna(0.0)
    merged["ic_ret"] = merged["ic_close"].pct_change().fillna(0.0)

    # style spreads
    merged["ss_im_ih"] = merged["im_ret"] - merged["ih_ret"]
    merged["ss_ic_ih"] = merged["ic_ret"] - merged["ih_ret"]

    # Extract UTC HH:MM for fast lookup
    merged["utc_hm"] = merged["datetime"].str[11:16]
    merged["bj_time"] = merged["utc_hm"].apply(_utc_to_bj)
    merged["date"] = merged["datetime"].str[:10].str.replace("-", "")

    # Build per-date lookup dict
    result: Dict[str, pd.DataFrame] = {}
    for date, grp in merged.groupby("date"):
        result[date] = grp.reset_index(drop=True)

    print(f"  Style spread data ready: {len(result)} dates, {len(merged)} bars total")
    return result


# ─── Core backtest with style spread capture ────────────────────────────────
def run_day_with_spread(
    sym: str,
    td: str,
    db: DBManager,
    spread_data: Dict[str, pd.DataFrame],
    spread_col: str = "ss_im_ih",
) -> List[Dict]:
    """
    Run one-day backtest identical to backtest_signals_day.run_day() but
    additionally records the style spread at each trade entry.

    Returns list of trade dicts, each with extra key 'entry_spread'.
    """
    from strategies.intraday.A_share_momentum_signal_v2 import (
        SignalGeneratorV2, SentimentData, check_exit, is_open_allowed,
        STOP_LOSS_PCT, TRAILING_STOP_HIVOL, TRAILING_STOP_NORMAL,
        SYMBOL_PROFILES, _DEFAULT_PROFILE, SIGNAL_ROUTING,
    )

    date_dash = f"{td[:4]}-{td[4:6]}-{td[6:]}"
    spot_sym = _SPOT_SYM.get(sym)

    # Load bars
    all_bars = db.query_df(
        f"SELECT datetime, open, high, low, close, volume "
        f"FROM index_min WHERE symbol='{spot_sym}' AND period=300 ORDER BY datetime"
    ) if spot_sym else None
    if all_bars is None or all_bars.empty:
        return []
    for c in ["open", "high", "low", "close", "volume"]:
        all_bars[c] = all_bars[c].astype(float)

    today_mask = all_bars["datetime"].str.startswith(date_dash)
    today_indices = all_bars.index[today_mask].tolist()
    if not today_indices:
        return []

    # Daily bars
    idx_code = _SPOT_IDX.get(sym, f"{sym}.CFX")
    daily_all = db.query_df(
        f"SELECT trade_date, close as open, close as high, close as low, close, 0 as volume "
        f"FROM index_daily WHERE ts_code='{idx_code}' ORDER BY trade_date"
    )
    if daily_all is not None and not daily_all.empty:
        daily_all["close"] = daily_all["close"].astype(float)
    else:
        daily_all = None

    # Z-Score
    spot_all = db.query_df(
        f"SELECT trade_date, close FROM index_daily WHERE ts_code='{idx_code}' ORDER BY trade_date"
    )
    if spot_all is not None:
        spot_all["close"] = spot_all["close"].astype(float)

    def _zscore_for_date(target_date: str):
        if spot_all is None or spot_all.empty:
            return 0.0, 0.0
        sub = spot_all[spot_all["trade_date"] < target_date].tail(30)
        if len(sub) < 20:
            return 0.0, 0.0
        closes = sub["close"].values
        ema = float(pd.Series(closes).ewm(span=20).mean().iloc[-1])
        std = float(pd.Series(closes).rolling(20).std().iloc[-1])
        return ema, std

    ema20, std20 = _zscore_for_date(td)

    # GARCH regime
    is_high_vol = True
    dmo = db.query_df(
        "SELECT garch_forecast_vol FROM daily_model_output "
        "WHERE underlying='IM' AND garch_forecast_vol > 0 "
        f"AND trade_date < '{td}' ORDER BY trade_date DESC LIMIT 1"
    )
    if dmo is not None and not dmo.empty:
        is_high_vol = (float(dmo.iloc[0].iloc[0]) * 100 / 24.9) > 1.2

    # Sentiment
    sentiment = None
    try:
        sdf = db.query_df(
            "SELECT atm_iv, atm_iv_market, vrp, term_structure_shape, rr_25d "
            "FROM daily_model_output WHERE underlying='IM' "
            f"AND trade_date < '{td}' ORDER BY trade_date DESC LIMIT 2"
        )
        if sdf is not None and len(sdf) >= 1:
            cur = sdf.iloc[0]
            prev = sdf.iloc[1] if len(sdf) >= 2 else cur
            sentiment = SentimentData(
                atm_iv=float(cur.get("atm_iv_market") or cur.get("atm_iv") or 0),
                atm_iv_prev=float(prev.get("atm_iv_market") or prev.get("atm_iv") or 0),
                rr_25d=float(cur.get("rr_25d") or 0),
                rr_25d_prev=float(prev.get("rr_25d") or 0),
                vrp=float(cur.get("vrp") or 0),
                term_structure=str(cur.get("term_structure_shape") or ""),
            )
    except Exception:
        pass

    # Signal generator (always v2)
    gen = SignalGeneratorV2({"min_signal_score": 60})

    # Per-symbol threshold
    _sym_prof = SYMBOL_PROFILES.get(sym, _DEFAULT_PROFILE)
    effective_threshold = _sym_prof.get("signal_threshold", 60)

    # Morning briefing d_override
    d_override = None
    try:
        briefing_row = db.query_df(
            "SELECT d_override_long, d_override_short FROM morning_briefing "
            f"WHERE trade_date = '{td}' LIMIT 1"
        )
        if briefing_row is not None and len(briefing_row) > 0:
            d_long = briefing_row.iloc[0].get("d_override_long")
            d_short = briefing_row.iloc[0].get("d_override_short")
            if d_long is not None and d_short is not None:
                d_override = {"LONG": float(d_long), "SHORT": float(d_short)}
    except Exception:
        pass

    # Daily df (truncated, no future leakage)
    daily_df = None
    if daily_all is not None:
        daily_df = daily_all[daily_all["trade_date"] < td].tail(30).reset_index(drop=True)
        if daily_df.empty:
            daily_df = None

    # prevClose
    prev_c = 0.0
    if daily_df is not None and len(daily_df) >= 1:
        prev_rows = daily_df[daily_df["trade_date"] < td]
        if len(prev_rows) > 0:
            prev_c = float(prev_rows.iloc[-1]["close"])

    _today_open = 0.0
    _gap_pct = 0.0
    if prev_c > 0 and today_indices:
        _today_open = float(all_bars.loc[today_indices[0], "open"])
        _gap_pct = (_today_open - prev_c) / prev_c

    # Style spread lookup for this date
    day_spreads = spread_data.get(td, pd.DataFrame())
    spread_by_utc: Dict[str, float] = {}
    if not day_spreads.empty:
        for _, row in day_spreads.iterrows():
            spread_by_utc[row["utc_hm"]] = float(row[spread_col])

    # Backtest loop
    position: Optional[Dict] = None
    completed_trades: List[Dict] = []
    last_exit_utc = ""
    last_exit_dir = ""
    COOLDOWN_MINUTES = 15

    for idx in today_indices:
        bar_5m = all_bars.loc[:idx].tail(200).copy()
        if len(bar_5m) < 15:
            continue

        bar_5m_signal = bar_5m.iloc[:-1]
        if len(bar_5m_signal) < 15:
            continue

        price = float(bar_5m.iloc[-1]["close"])
        high = float(bar_5m.iloc[-1]["high"])
        low = float(bar_5m.iloc[-1]["low"])
        signal_price = float(bar_5m_signal.iloc[-1]["close"])
        dt_str = str(all_bars.loc[idx, "datetime"])
        utc_hm = dt_str[11:16]
        bj_time = _utc_to_bj(utc_hm)

        z_val = (signal_price - ema20) / std20 if std20 > 0 else None

        bar_15m = _build_15m_from_5m(bar_5m_signal)

        result = gen.score_all(
            sym, bar_5m_signal, bar_15m, daily_df, None, sentiment,
            zscore=z_val, is_high_vol=is_high_vol, d_override=d_override,
        )

        score = result["total"] if result else 0
        direction = result["direction"] if result else ""

        # style spread for the completed bar (bar_5m_signal[-1] == bar before current)
        # We want the spread of the last completed bar, so look up by utc_hm of
        # the signal bar (bar_5m_signal.iloc[-1])
        signal_bar_dt = str(bar_5m_signal.iloc[-1]["datetime"])
        signal_bar_utc = signal_bar_dt[11:16]
        entry_spread = spread_by_utc.get(signal_bar_utc, float("nan"))

        action_str = ""
        if position is not None:
            stop_price = position.get("stop_loss", 0)
            bar_stopped = False
            if stop_price > 0:
                if position["direction"] == "LONG" and low <= stop_price:
                    bar_stopped = True
                elif position["direction"] == "SHORT" and high >= stop_price:
                    bar_stopped = True

            if bar_stopped:
                entry_p = position["entry_price"]
                exit_p = stop_price
                pnl_pts = (exit_p - entry_p) if position["direction"] == "LONG" else (entry_p - exit_p)
                elapsed = _calc_minutes(position["entry_time_utc"], utc_hm)
                completed_trades.append({
                    **_trade_record(position, bj_time, exit_p, pnl_pts, elapsed, "STOP_LOSS"),
                })
                last_exit_utc = utc_hm
                last_exit_dir = position["direction"]
                position = None
                action_str = "STOP_LOSS"
            else:
                if position["direction"] == "LONG":
                    position["highest_since"] = max(position["highest_since"], high)
                else:
                    position["lowest_since"] = min(position["lowest_since"], low)

                reverse_score = 0
                if result and direction and direction != position["direction"]:
                    reverse_score = score

                exit_info = check_exit(
                    position, price, bar_5m_signal,
                    bar_15m if not bar_15m.empty else None,
                    utc_hm, reverse_score, is_high_vol=is_high_vol, symbol=sym,
                )
                if exit_info["should_exit"]:
                    exit_vol = exit_info["exit_volume"]
                    reason = exit_info["exit_reason"]
                    entry_p = position["entry_price"]
                    exit_p = price
                    pnl_pts = (exit_p - entry_p) if position["direction"] == "LONG" else (entry_p - exit_p)
                    elapsed = _calc_minutes(position["entry_time_utc"], utc_hm)

                    if exit_vol >= position["volume"]:
                        completed_trades.append(
                            _trade_record(position, bj_time, exit_p, pnl_pts, elapsed, reason)
                        )
                        last_exit_utc = utc_hm
                        last_exit_dir = position["direction"]
                        position = None
                        action_str = reason
                    else:
                        completed_trades.append(
                            _trade_record(position, bj_time, exit_p, pnl_pts, elapsed,
                                          reason + "(半仓)", partial=True)
                        )
                        position["volume"] -= exit_vol
                        position["half_closed"] = True
                        action_str = reason + "_PARTIAL"

        in_cooldown = False
        if last_exit_utc and direction == last_exit_dir:
            cd_elapsed = _calc_minutes(last_exit_utc, utc_hm)
            if 0 < cd_elapsed < COOLDOWN_MINUTES:
                in_cooldown = True

        if (position is None and not action_str and result and not in_cooldown
                and score >= effective_threshold and direction and is_open_allowed(utc_hm)):
            entry_p = price
            stop = (entry_p * (1 - STOP_LOSS_PCT) if direction == "LONG"
                    else entry_p * (1 + STOP_LOSS_PCT))
            position = {
                "entry_price": entry_p,
                "direction": direction,
                "entry_time_utc": utc_hm,
                "highest_since": high,
                "lowest_since": low,
                "stop_loss": stop,
                "volume": 1,
                "half_closed": False,
                "bars_below_mid": 0,
                "entry_score": score,
                "entry_v_score": result.get("s_volatility", 0),
                "entry_daily_mult": result.get("daily_mult", 1.0),
                "entry_raw_total": result.get("raw_total", 0),
                "entry_gap_pct": _gap_pct,
                "entry_gap_aligned": ((_gap_pct < 0 and direction == "SHORT") or
                                      (_gap_pct > 0 and direction == "LONG")),
                "entry_rebound_pct": 0.0,
                "entry_total_drop": (signal_price - prev_c) / prev_c if prev_c > 0 else 0.0,
                "entry_intraday_drop": (signal_price - _today_open) / _today_open if _today_open > 0 else 0.0,
                "entry_filter_mult": result.get("intraday_filter", 1.0),
                "entry_zscore": z_val,
                "entry_spread": entry_spread,  # <<< key field for this research
            }

    # Force-close EOD
    if position is not None:
        last_price = float(all_bars.loc[today_indices[-1], "close"])
        entry_p = position["entry_price"]
        exit_p = last_price
        pnl = (exit_p - entry_p) if position["direction"] == "LONG" else (entry_p - exit_p)
        elapsed = _calc_minutes(position["entry_time_utc"],
                                 str(all_bars.loc[today_indices[-1], "datetime"])[11:16])
        completed_trades.append(
            _trade_record(position, _utc_to_bj(str(all_bars.loc[today_indices[-1], "datetime"])[11:16]),
                          exit_p, pnl, elapsed, "EOD_FORCE")
        )

    return completed_trades


def _trade_record(position: Dict, exit_time: str, exit_price: float,
                  pnl_pts: float, minutes: int, reason: str,
                  partial: bool = False) -> Dict:
    return {
        "entry_time": _utc_to_bj(position["entry_time_utc"]),
        "entry_price": position["entry_price"],
        "exit_time": exit_time,
        "exit_price": exit_price,
        "direction": position["direction"],
        "pnl_pts": pnl_pts,
        "reason": reason,
        "minutes": minutes,
        "partial": partial,
        "entry_score": position.get("entry_score", 0),
        "entry_daily_mult": position.get("entry_daily_mult", 1.0),
        "entry_spread": position.get("entry_spread", float("nan")),
    }


# ─── Analysis ────────────────────────────────────────────────────────────────
def collect_all_trades(
    sym: str, dates: List[str], db: DBManager,
    spread_data: Dict[str, pd.DataFrame],
    spread_col: str = "ss_im_ih",
) -> pd.DataFrame:
    """Run backtest for all dates, return DataFrame of trades with spread."""
    all_trades = []
    for td in dates:
        trades = run_day_with_spread(sym, td, db, spread_data, spread_col=spread_col)
        for t in trades:
            t["date"] = td
        all_trades.extend(trades)
    if not all_trades:
        return pd.DataFrame()
    df = pd.DataFrame(all_trades)
    df["full"] = ~df.get("partial", False).fillna(False)
    return df


def filter_summary(trades_df: pd.DataFrame, lo: float, hi: float) -> Dict:
    """
    Simulate applying a spread filter: block trades where spread < lo or spread > hi.
    Returns summary dict.
    """
    full = trades_df[trades_df["full"]].copy()
    has_spread = full["entry_spread"].notna()

    # Trades we keep = spread within [lo, hi] OR spread is NaN (conservative: keep NaN)
    blocked_mask = has_spread & ((full["entry_spread"] < lo) | (full["entry_spread"] > hi))
    kept = full[~blocked_mask]
    blocked = full[blocked_mask]

    n_total = len(full)
    n_blocked = len(blocked)
    n_kept = len(kept)
    pnl = kept["pnl_pts"].sum()
    wins = (kept["pnl_pts"] > 0).sum()
    wr = wins / n_kept * 100 if n_kept > 0 else 0.0
    return {
        "n_total": n_total,
        "n_blocked": n_blocked,
        "n_kept": n_kept,
        "pnl": pnl,
        "wr": wr,
    }


def print_filter_table(sym: str, trades_df: pd.DataFrame,
                       spread_label: str, spread_col_label: str):
    if trades_df.empty:
        print(f"  No trades for {sym}")
        return

    full = trades_df[trades_df["full"]]
    baseline_pnl = full["pnl_pts"].sum()
    baseline_n = len(full)
    baseline_wins = (full["pnl_pts"] > 0).sum()
    baseline_wr = baseline_wins / baseline_n * 100 if baseline_n > 0 else 0

    print(f"\n{sym} ({spread_col_label}):")
    print(f"{'阈值':<18} | {'过滤笔数':>8} | {'剩余笔数':>8} | {'WR':>7} | {'PnL':>8} | {'vs baseline':>12}")
    print("-" * 75)
    print(f"  {'无过滤':<16} | {'0':>8} | {baseline_n:>8} | {baseline_wr:>6.1f}% | {baseline_pnl:>+8.0f} | {'—':>12}")

    best_delta = None
    best_threshold = None
    for label, lo, hi in FILTER_THRESHOLDS:
        r = filter_summary(trades_df, lo, hi)
        delta = r["pnl"] - baseline_pnl
        delta_str = f"{delta:+.0f}"
        wr_str = f"{r['wr']:.1f}%"
        pnl_str = f"{r['pnl']:+.0f}"
        print(f"  {label:<16} | {r['n_blocked']:>8} | {r['n_kept']:>8} | {wr_str:>7} | {pnl_str:>8} | {delta_str:>12}")
        if best_delta is None or r["pnl"] > best_delta[0]:
            best_delta = (r["pnl"], delta)
            best_threshold = (label, lo, hi)

    return best_threshold


def print_robustness(sym: str, trades_df: pd.DataFrame,
                     best_threshold: Tuple, spread_col_label: str):
    """Print ±20% robustness check for the best threshold."""
    label, lo, hi = best_threshold
    print(f"\n稳健性 ({sym} {spread_col_label}, 最优={label}):")
    print(f"{'阈值变体':<22} | {'过滤笔数':>8} | {'剩余笔数':>8} | {'WR':>7} | {'PnL':>8}")
    print("-" * 65)

    variants = [
        ("最优×1.0 (原始)", lo, hi),
        ("最优×0.8 (收紧)", lo * 0.8, hi * 0.8),
        ("最优×1.2 (放松)", lo * 1.2, hi * 1.2),
    ]
    results = []
    for vlabel, vlo, vhi in variants:
        r = filter_summary(trades_df, vlo, vhi)
        results.append((vlabel, r))
        print(f"  {vlabel:<20} | {r['n_blocked']:>8} | {r['n_kept']:>8} | {r['wr']:>6.1f}% | {r['pnl']:>+8.0f}")

    if len(results) >= 2:
        base_pnl = results[0][1]["pnl"]
        for vlabel, r in results[1:]:
            if base_pnl != 0:
                chg = abs(r["pnl"] - base_pnl) / abs(base_pnl) * 100
                stable = "稳健" if chg <= 30 else "不稳健"
                print(f"  → {vlabel}: PnL变化 {chg:.0f}% ({stable})")


def spread_distribution(trades_df: pd.DataFrame, sym: str):
    """Print distribution stats of entry spread values."""
    full = trades_df[trades_df["full"]].copy()
    has_spread = full["entry_spread"].notna()
    spreads = full.loc[has_spread, "entry_spread"] * 100  # convert to bps (actually %)

    if spreads.empty:
        print(f"  {sym}: no spread data")
        return

    wins = full.loc[has_spread & (full["pnl_pts"] > 0), "entry_spread"] * 100
    losses = full.loc[has_spread & (full["pnl_pts"] <= 0), "entry_spread"] * 100

    print(f"\n{sym} spread分布 (%, {len(spreads)}笔):")
    print(f"  全体:  min={spreads.min():.3f}  p10={spreads.quantile(0.1):.3f}"
          f"  p25={spreads.quantile(0.25):.3f}  median={spreads.median():.3f}"
          f"  p75={spreads.quantile(0.75):.3f}  p90={spreads.quantile(0.9):.3f}"
          f"  max={spreads.max():.3f}")
    print(f"  盈利({len(wins)}笔): median={wins.median():.3f}  mean={wins.mean():.3f}")
    print(f"  亏损({len(losses)}笔): median={losses.median():.3f}  mean={losses.mean():.3f}")

    # PnL by spread quintile
    q_labels = ["Q1(最小)", "Q2", "Q3", "Q4", "Q5(最大)"]
    full_hs = full[has_spread].copy()
    full_hs["spread_q"] = pd.qcut(full_hs["entry_spread"], 5, labels=q_labels)
    print(f"  按spread分位PnL:")
    for q in q_labels:
        sub = full_hs[full_hs["spread_q"] == q]
        if sub.empty:
            continue
        pnl = sub["pnl_pts"].sum()
        n = len(sub)
        wr = (sub["pnl_pts"] > 0).sum() / n * 100
        rng = f"[{sub['entry_spread'].min()*100:.3f},{sub['entry_spread'].max()*100:.3f}]"
        print(f"    {q}: {n:3d}笔 WR={wr:.0f}% PnL={pnl:+.0f}pt  range={rng}%")


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    db = DBManager(ConfigLoader().get_db_path())

    # Load style spread data once
    spread_data = load_style_spreads(db)
    if not spread_data:
        print("ERROR: Failed to load style spread data. Exiting.")
        return

    print("\n" + "=" * 80)
    print(" Style Spread过滤效果研究")
    print("=" * 80)

    # ── IM trades ──────────────────────────────────────────────────────────
    print("\n[IM] 收集回测交易（v2, thr=60）...")
    im_trades = collect_all_trades("IM", DATES, db, spread_data, spread_col="ss_im_ih")
    print(f"  IM: {len(im_trades[im_trades['full']])} full trades" if not im_trades.empty else "  IM: no trades")

    # ── IC trades (two spread flavors) ────────────────────────────────────
    print("\n[IC] 收集回测交易（v2, thr=65）...")
    ic_trades_im_ih = collect_all_trades("IC", DATES, db, spread_data, spread_col="ss_im_ih")
    ic_trades_ic_ih = collect_all_trades("IC", DATES, db, spread_data, spread_col="ss_ic_ih")
    print(f"  IC: {len(ic_trades_im_ih[ic_trades_im_ih['full']])} full trades" if not ic_trades_im_ih.empty else "  IC: no trades")

    print("\n" + "=" * 80)
    print(" === Style Spread过滤效果 ===")
    print("=" * 80)

    # IM spread distribution
    if not im_trades.empty:
        spread_distribution(im_trades, "IM")

    # IC spread distribution (IM-IH)
    if not ic_trades_im_ih.empty:
        spread_distribution(ic_trades_im_ih, "IC(IM-IH spread)")

    print("\n" + "=" * 80)

    # IM filter table
    best_im = None
    if not im_trades.empty:
        best_im = print_filter_table("IM", im_trades, "ss_im_ih", "IM-IH spread")

    # IC filter tables
    best_ic_im_ih = None
    if not ic_trades_im_ih.empty:
        best_ic_im_ih = print_filter_table("IC", ic_trades_im_ih, "ss_im_ih", "IM-IH spread")

    best_ic_ic_ih = None
    if not ic_trades_ic_ih.empty:
        best_ic_ic_ih = print_filter_table("IC", ic_trades_ic_ih, "ss_ic_ih", "IC-IH spread")

    # Robustness
    print("\n" + "=" * 80)
    print(" === 稳健性检验（最优阈值 ±20%） ===")
    print("=" * 80)

    if best_im and not im_trades.empty:
        print_robustness("IM", im_trades, best_im, "IM-IH spread")

    if best_ic_im_ih and not ic_trades_im_ih.empty:
        print_robustness("IC", ic_trades_im_ih, best_ic_im_ih, "IM-IH spread")

    if best_ic_ic_ih and not ic_trades_ic_ih.empty:
        print_robustness("IC", ic_trades_ic_ih, best_ic_ic_ih, "IC-IH spread")

    # Combined best-threshold summary
    print("\n" + "=" * 80)
    print(" === 最优合并汇总（IM + IC） ===")
    print("=" * 80)
    print(f"{'阈值变体':<22} | {'IM PnL':>8} | {'IC PnL':>8} | {'合计':>8} | {'vs no-filter':>12}")
    print("-" * 70)

    # No filter baselines
    im_base = im_trades[im_trades["full"]]["pnl_pts"].sum() if not im_trades.empty else 0
    ic_base = ic_trades_im_ih[ic_trades_im_ih["full"]]["pnl_pts"].sum() if not ic_trades_im_ih.empty else 0
    total_base = im_base + ic_base
    print(f"  {'无过滤':<20} | {im_base:>+8.0f} | {ic_base:>+8.0f} | {total_base:>+8.0f} | {'—':>12}")

    for label, lo, hi in FILTER_THRESHOLDS:
        im_r = filter_summary(im_trades, lo, hi) if not im_trades.empty else {"pnl": 0}
        ic_r = filter_summary(ic_trades_im_ih, lo, hi) if not ic_trades_im_ih.empty else {"pnl": 0}
        total = im_r["pnl"] + ic_r["pnl"]
        delta = total - total_base
        print(f"  {label:<20} | {im_r['pnl']:>+8.0f} | {ic_r['pnl']:>+8.0f} | {total:>+8.0f} | {delta:>+12.0f}")

    print()


if __name__ == "__main__":
    main()
