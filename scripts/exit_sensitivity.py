#!/usr/bin/env python3
"""
exit_sensitivity.py
-------------------
Exit系统参数敏感性分析。每次只变一个参数组，其他保持当前值。

参数组：
  G1  STOP_LOSS_PCT         固定止损百分比 (当前 0.5%)
  G2  Trailing scale        跟踪止盈阶梯（等比缩放，当前 1.0x）
  G3  Trailing bonus        15m确认加宽 (当前 +0.2%)
  G4  ME hold_minutes       MomentumExhausted最小持仓时间 (当前 20分钟)
  G5  ME ratio              MomentumExhausted窄幅阈值 (当前 boll*0.20)
  G6  TIME_STOP_MINUTES     时间止损 (当前 60分钟)
  G7  LUNCH_CLOSE mode      午休平仓模式 (当前 loss_only=亏损才平)
  G8  MID_BREAK bars        中轨跌破确认bar数 (当前 2)

使用方法：
    python scripts/exit_sensitivity.py --symbol IM
    python scripts/exit_sensitivity.py --symbol IM --group G1,G2,G4
    python scripts/exit_sensitivity.py --symbol IM --group G1 --slippage 2
    python scripts/exit_sensitivity.py --symbol IC
"""
from __future__ import annotations

import argparse
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

# ─────────────────────────────────────────────────────────────
# 回测日期（35天）
# ─────────────────────────────────────────────────────────────
DEFAULT_DATES = (
    "20260204,20260205,20260206,20260209,20260210,20260211,20260212,20260213,"
    "20260225,20260226,20260227,20260302,20260303,20260304,20260305,20260306,"
    "20260309,20260310,20260311,20260312,20260313,20260316,20260317,20260318,"
    "20260319,20260320,20260323,20260324,20260325,20260326,20260327,20260328,"
    "20260401,20260402,20260403"
)

IM_MULT = 200
COOLDOWN_MINUTES = 15

# ─────────────────────────────────────────────────────────────
# Current (baseline) values
# ─────────────────────────────────────────────────────────────
BASELINE = {
    "stop_loss_pct":     0.005,
    "trail_scale":       1.0,      # multiplies [0.5%,0.6%,0.8%,1.0%] ladder
    "trail_bonus":       0.002,    # +0.2% when 15m confirmed & PnL>0.5%
    "me_hold_min":       20,       # P5 MOMENTUM_EXHAUSTED minimum hold (minutes)
    "me_ratio":          0.20,     # P5 narrow-range threshold (fraction of boll_width)
    "time_stop_min":     60,       # P7 TIME_STOP_MINUTES
    "lunch_mode":        "loss_only",  # "loss_only" | "always"
    "mid_break_bars":    2,        # P6 MID_BREAK consecutive bars
}

# ─────────────────────────────────────────────────────────────
# Sweep definitions: (param_name, [values_to_test])
# ─────────────────────────────────────────────────────────────
SWEEP_GROUPS = {
    "G1": ("stop_loss_pct",  [0.003, 0.004, 0.005, 0.006, 0.007, 0.010]),
    "G2": ("trail_scale",    [0.5, 0.7, 1.0, 1.3, 1.5, 2.0]),
    "G3": ("trail_bonus",    [0.0, 0.001, 0.002, 0.003, 0.005]),
    "G4": ("me_hold_min",    [0, 10, 15, 20, 30, 45]),
    "G5": ("me_ratio",       [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]),
    "G6": ("time_stop_min",  [30, 45, 60, 90, 120, 999]),
    "G7": ("lunch_mode",     ["always", "loss_only", "never"]),
    "G8": ("mid_break_bars", [1, 2, 3, 4]),
}

GROUP_DESC = {
    "G1": "StopLoss %",
    "G2": "TrailingStop scale (ladder × X)",
    "G3": "TrailingStop 15m-confirm bonus",
    "G4": "ME min hold (minutes)",
    "G5": "ME narrow-range ratio (× boll_width)",
    "G6": "TimeStop (minutes)",
    "G7": "LunchClose mode",
    "G8": "MidBreak bars",
}


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _utc_to_bj(utc_str: str) -> str:
    h = int(utc_str[:2]) + 8
    if h >= 24:
        h -= 24
    return f"{h:02d}:{utc_str[3:5]}"


def _calc_minutes(t1: str, t2: str) -> int:
    try:
        h1, m1 = int(t1[:2]), int(t1[3:5])
        h2, m2 = int(t2[:2]), int(t2[3:5])
        return (h2 * 60 + m2) - (h1 * 60 + m1)
    except Exception:
        return 0


def _build_15m_from_5m(bar_5m: pd.DataFrame) -> pd.DataFrame:
    if len(bar_5m) < 3:
        return pd.DataFrame()
    df = bar_5m.copy()
    df["dt"] = pd.to_datetime(df["datetime"] if "datetime" in df.columns else df.index)
    df = df.set_index("dt")
    resampled = df.resample("15min", label="left", closed="left").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()
    return resampled.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# Parameterized check_exit
# ─────────────────────────────────────────────────────────────

def _calc_boll(closes: pd.Series, period: int = 20):
    if len(closes) < period:
        return float("nan"), float("nan")
    mid = float(closes.rolling(period).mean().iloc[-1])
    std = float(closes.rolling(period).std().iloc[-1])
    return mid, std


def _boll_zone(price: float, mid: float, std: float) -> str:
    upper = mid + 2 * std
    lower = mid - 2 * std
    half_up = mid + std
    half_dn = mid - std
    if price > upper:
        return "ABOVE_UPPER"
    elif price > half_up:
        return "UPPER_ZONE"
    elif price > mid:
        return "MID_UPPER"
    elif price > half_dn:
        return "MID_LOWER"
    elif price > lower:
        return "LOWER_ZONE"
    else:
        return "BELOW_LOWER"


def check_exit_param(
    position: dict,
    current_price: float,
    bar_5m: pd.DataFrame,
    bar_15m,
    current_time_utc: str,
    symbol: str = "",
    # ── tunable params ──────────────────────────────────────
    stop_loss_pct:  float = 0.005,
    trail_scale:    float = 1.0,
    trail_bonus:    float = 0.002,
    me_hold_min:    int   = 20,
    me_ratio:       float = 0.20,
    time_stop_min:  int   = 60,
    lunch_mode:     str   = "loss_only",   # "always"|"loss_only"|"never"
    mid_break_bars: int   = 2,
    # ── fixed constants ─────────────────────────────────────
    EOD_CLOSE_UTC:    str = "06:45",
    LUNCH_CLOSE_UTC:  str = "03:25",
    TRAILING_STOP_LUNCH: float = 0.003,
) -> dict:
    """
    Parameterized version of check_exit.
    All exit logic is identical to the production version except the
    hard-coded numbers are replaced by function parameters.
    """
    entry_price = position["entry_price"]
    direction   = position["direction"]
    entry_time  = position.get("entry_time_utc", "")
    highest     = position.get("highest_since", entry_price)
    lowest      = position.get("lowest_since",  entry_price)
    volume      = position.get("volume", 1)

    boll_price = current_price   # in backtest spot=futures

    NO_EXIT = {"should_exit": False, "exit_volume": 0,
               "exit_reason": "", "exit_urgency": "NORMAL"}

    # P1: EOD
    if current_time_utc >= EOD_CLOSE_UTC:
        return {"should_exit": True, "exit_volume": volume,
                "exit_reason": "EOD_CLOSE", "exit_urgency": "URGENT"}

    # P1b: Stop loss
    loss_pct = ((entry_price - current_price) / entry_price if direction == "LONG"
                else (current_price - entry_price) / entry_price)
    if loss_pct > stop_loss_pct:
        return {"should_exit": True, "exit_volume": volume,
                "exit_reason": "STOP_LOSS", "exit_urgency": "URGENT"}

    # P2: Lunch close
    if current_time_utc >= LUNCH_CLOSE_UTC and current_time_utc < "05:00":
        profitable = (current_price > entry_price) if direction == "LONG" else (current_price < entry_price)
        if lunch_mode == "always":
            return {"should_exit": True, "exit_volume": volume,
                    "exit_reason": "LUNCH_CLOSE", "exit_urgency": "URGENT"}
        elif lunch_mode == "loss_only":
            if not profitable:
                return {"should_exit": True, "exit_volume": volume,
                        "exit_reason": "LUNCH_CLOSE", "exit_urgency": "URGENT"}
            # Profitable: tighten trailing
            if direction == "LONG" and highest > entry_price:
                dd = (highest - current_price) / highest
                if dd > TRAILING_STOP_LUNCH:
                    return {"should_exit": True, "exit_volume": volume,
                            "exit_reason": "LUNCH_TRAIL", "exit_urgency": "NORMAL"}
            elif direction == "SHORT" and lowest < entry_price:
                du = (current_price - lowest) / lowest
                if du > TRAILING_STOP_LUNCH:
                    return {"should_exit": True, "exit_volume": volume,
                            "exit_reason": "LUNCH_TRAIL", "exit_urgency": "NORMAL"}
        # lunch_mode == "never" → skip

    # Bollinger bands
    b5_mid, b5_std = float("nan"), float("nan")
    zone_5m = zone_15m = ""

    if bar_5m is not None and len(bar_5m) >= 20:
        c5 = bar_5m["close"].astype(float)
        b5_mid, b5_std = _calc_boll(c5)
        if not np.isnan(b5_mid) and b5_std > 0:
            zone_5m = _boll_zone(boll_price, b5_mid, b5_std)

    if bar_15m is not None and len(bar_15m) >= 20:
        c15 = bar_15m["close"].astype(float)
        b15_mid, b15_std = _calc_boll(c15)
        if not np.isnan(b15_mid) and b15_std > 0:
            zone_15m = _boll_zone(boll_price, b15_mid, b15_std)

    # Hold time
    hold_minutes = 0
    if entry_time and current_time_utc:
        try:
            h1, m1 = int(entry_time[:2]), int(entry_time[3:5])
            h2, m2 = int(current_time_utc[:2]), int(current_time_utc[3:5])
            hold_minutes = (h2 * 60 + m2) - (h1 * 60 + m1)
        except Exception:
            pass

    # P3: Dynamic trailing stop (scaled)
    # Base ladder (production values) × trail_scale
    # IC禁用trailing_stop by SYMBOL_PROFILES (not re-implemented here; IC sweep is still informative)
    if hold_minutes < 15:
        base_trail = 0.005
    elif hold_minutes < 30:
        base_trail = 0.006
    elif hold_minutes < 60:
        base_trail = 0.008
    else:
        base_trail = 0.010

    trail_pct = base_trail * trail_scale

    # Bonus: 15m trend confirmed + profitable >0.5%
    if zone_15m:
        if direction == "LONG":
            pnl_pct    = (current_price - entry_price) / entry_price
            fifteen_ok = zone_15m in ("MID_UPPER", "UPPER_ZONE", "ABOVE_UPPER")
        else:
            pnl_pct    = (entry_price - current_price) / entry_price
            fifteen_ok = zone_15m in ("MID_LOWER", "LOWER_ZONE", "BELOW_LOWER")
        if pnl_pct > 0.005 and fifteen_ok:
            trail_pct += trail_bonus

    if direction == "LONG" and highest > entry_price:
        dd = (highest - current_price) / highest
        if dd > trail_pct:
            return {"should_exit": True, "exit_volume": volume,
                    "exit_reason": "TRAILING_STOP", "exit_urgency": "NORMAL"}
    elif direction == "SHORT" and lowest < entry_price:
        du = (current_price - lowest) / lowest
        if du > trail_pct:
            return {"should_exit": True, "exit_volume": volume,
                    "exit_reason": "TRAILING_STOP", "exit_urgency": "NORMAL"}

    # P4: Trend complete
    if zone_5m and zone_15m:
        if direction == "LONG":
            if zone_5m == "ABOVE_UPPER" and zone_15m == "ABOVE_UPPER":
                return {"should_exit": True, "exit_volume": volume,
                        "exit_reason": "TREND_COMPLETE", "exit_urgency": "NORMAL"}
        else:
            if zone_5m == "BELOW_LOWER" and zone_15m == "BELOW_LOWER":
                return {"should_exit": True, "exit_volume": volume,
                        "exit_reason": "TREND_COMPLETE", "exit_urgency": "NORMAL"}

    # P5: Momentum exhausted
    if hold_minutes >= me_hold_min and bar_5m is not None and len(bar_5m) >= 23 and b5_std > 0:
        last3_c = bar_5m["close"].astype(float).iloc[-3:]
        last3_h = bar_5m["high"].astype(float).iloc[-3:]
        last3_l = bar_5m["low"].astype(float).iloc[-3:]
        total_range  = float(last3_h.max() - last3_l.min())
        boll_width   = 4 * b5_std
        if total_range < boll_width * me_ratio:
            close_change  = float(last3_c.iloc[-1]) - float(last3_c.iloc[0])
            still_trending = False
            if direction == "LONG" and close_change > boll_width * 0.05:
                still_trending = True
            elif direction == "SHORT" and close_change < -boll_width * 0.05:
                still_trending = True
            if not still_trending:
                if direction == "LONG" and zone_15m in ("ABOVE_UPPER", "UPPER_ZONE"):
                    return {"should_exit": True, "exit_volume": volume,
                            "exit_reason": "MOMENTUM_EXHAUSTED", "exit_urgency": "NORMAL"}
                elif direction == "SHORT" and zone_15m in ("BELOW_LOWER", "LOWER_ZONE"):
                    return {"should_exit": True, "exit_volume": volume,
                            "exit_reason": "MOMENTUM_EXHAUSTED", "exit_urgency": "NORMAL"}

    # P6: Mid-band break
    if zone_5m and not np.isnan(b5_mid):
        if direction == "LONG":
            five_below   = boll_price < b5_mid
            fifteen_below = zone_15m in ("MID_LOWER", "LOWER_ZONE", "BELOW_LOWER") if zone_15m else False
        else:
            five_below   = boll_price > b5_mid
            fifteen_below = zone_15m in ("MID_UPPER", "UPPER_ZONE", "ABOVE_UPPER") if zone_15m else False

        if five_below:
            bars_bm = position.get("bars_below_mid", 0) + 1
            position["bars_below_mid"] = bars_bm
            if bars_bm >= mid_break_bars and fifteen_below:
                return {"should_exit": True, "exit_volume": volume,
                        "exit_reason": "MID_BREAK", "exit_urgency": "NORMAL"}
        else:
            position["bars_below_mid"] = 0

    # P7: Time stop
    if entry_time and current_time_utc:
        try:
            h1, m1 = int(entry_time[:2]), int(entry_time[3:5])
            h2, m2 = int(current_time_utc[:2]), int(current_time_utc[3:5])
            elapsed = (h2 * 60 + m2) - (h1 * 60 + m1)
            if elapsed > time_stop_min:
                profitable = (current_price > entry_price if direction == "LONG"
                              else current_price < entry_price)
                if not profitable:
                    return {"should_exit": True, "exit_volume": volume,
                            "exit_reason": "TIME_STOP", "exit_urgency": "NORMAL"}
        except Exception:
            pass

    return NO_EXIT


# ─────────────────────────────────────────────────────────────
# Core backtest for one day with explicit exit params
# ─────────────────────────────────────────────────────────────

def run_day_param(
    sym: str,
    td: str,
    db: DBManager,
    # day-level cached data (passed in to avoid re-querying)
    all_bars: pd.DataFrame,
    daily_all: Optional[pd.DataFrame],
    ema20: float,
    std20: float,
    is_high_vol: bool,
    sentiment,
    d_override,
    gen,
    effective_threshold: int,
    slippage: float = 0,
    # exit params
    stop_loss_pct:  float = 0.005,
    trail_scale:    float = 1.0,
    trail_bonus:    float = 0.002,
    me_hold_min:    int   = 20,
    me_ratio:       float = 0.20,
    time_stop_min:  int   = 60,
    lunch_mode:     str   = "loss_only",
    mid_break_bars: int   = 2,
) -> List[Dict]:
    from strategies.intraday.A_share_momentum_signal_v2 import is_open_allowed

    date_dash = f"{td[:4]}-{td[4:6]}-{td[6:]}"
    today_mask    = all_bars["datetime"].str.startswith(date_dash)
    today_indices = all_bars.index[today_mask].tolist()
    if not today_indices:
        return []

    daily_df = None
    if daily_all is not None:
        daily_df = daily_all[daily_all["trade_date"] < td].tail(30).reset_index(drop=True)
        if daily_df.empty:
            daily_df = None

    # prevClose (最后一个 < today)
    prev_c = 0.0
    _today_open = 0.0
    _gap_pct = 0.0
    if daily_df is not None and len(daily_df) > 0:
        prev_rows = daily_df[daily_df["trade_date"] < td]
        if len(prev_rows) > 0:
            prev_c = float(prev_rows.iloc[-1]["close"])
    if prev_c > 0 and today_indices:
        _today_open = float(all_bars.loc[today_indices[0], "open"])
        _gap_pct = (_today_open - prev_c) / prev_c

    position: Optional[Dict] = None
    completed_trades: List[Dict] = []
    last_exit_utc = ""
    last_exit_dir = ""

    # Stop-loss price uses stop_loss_pct param
    def _make_stop(entry_p, direction):
        if direction == "LONG":
            return entry_p * (1 - stop_loss_pct)
        else:
            return entry_p * (1 + stop_loss_pct)

    for idx in today_indices:
        bar_5m = all_bars.loc[:idx].tail(200).copy()
        if len(bar_5m) < 15:
            continue

        bar_5m_signal = bar_5m.iloc[:-1]
        if len(bar_5m_signal) < 15:
            continue

        price  = float(bar_5m.iloc[-1]["close"])
        high   = float(bar_5m.iloc[-1]["high"])
        low    = float(bar_5m.iloc[-1]["low"])
        signal_price = float(bar_5m_signal.iloc[-1]["close"])
        dt_str = str(all_bars.loc[idx, "datetime"])
        utc_hm = dt_str[11:16]

        z_val = (signal_price - ema20) / std20 if std20 > 0 else None
        bar_15m = _build_15m_from_5m(bar_5m_signal)

        result = gen.score_all(
            sym, bar_5m_signal, bar_15m, daily_df, None, sentiment,
            zscore=z_val, is_high_vol=is_high_vol, d_override=d_override,
        )

        score     = result["total"]     if result else 0
        direction = result["direction"] if result else ""

        action_str = ""

        # ── exit check ───────────────────────────────────────────────
        if position is not None:
            # Bar high/low stop-loss check (uses stop_loss_pct-based stop stored in position)
            stop_price = position.get("stop_loss", 0)
            bar_stopped = False
            if stop_price > 0:
                if position["direction"] == "LONG" and low <= stop_price:
                    bar_stopped = True
                elif position["direction"] == "SHORT" and high >= stop_price:
                    bar_stopped = True

            if bar_stopped:
                entry_p = position["entry_price"]
                exit_p  = (stop_price - slippage if position["direction"] == "LONG"
                           else stop_price + slippage)
                pnl_pts = (exit_p - entry_p) if position["direction"] == "LONG" else (entry_p - exit_p)
                elapsed = _calc_minutes(position["entry_time_utc"], utc_hm)
                completed_trades.append({
                    "direction": position["direction"],
                    "pnl_pts":   pnl_pts,
                    "reason":    "STOP_LOSS",
                    "minutes":   elapsed,
                    "entry_time": _utc_to_bj(position["entry_time_utc"]),
                    "exit_time":  _utc_to_bj(utc_hm),
                })
                last_exit_utc = utc_hm
                last_exit_dir = position["direction"]
                position      = None
            else:
                # Update extremes
                if position["direction"] == "LONG":
                    position["highest_since"] = max(position["highest_since"], high)
                else:
                    position["lowest_since"] = min(position["lowest_since"], low)

                exit_info = check_exit_param(
                    position, price,
                    bar_5m_signal,
                    bar_15m if not bar_15m.empty else None,
                    utc_hm,
                    symbol=sym,
                    stop_loss_pct=stop_loss_pct,
                    trail_scale=trail_scale,
                    trail_bonus=trail_bonus,
                    me_hold_min=me_hold_min,
                    me_ratio=me_ratio,
                    time_stop_min=time_stop_min,
                    lunch_mode=lunch_mode,
                    mid_break_bars=mid_break_bars,
                )

                if exit_info["should_exit"]:
                    exit_vol = exit_info["exit_volume"]
                    reason   = exit_info["exit_reason"]
                    entry_p  = position["entry_price"]
                    exit_p   = (price - slippage if position["direction"] == "LONG"
                                else price + slippage)
                    pnl_pts  = (exit_p - entry_p) if position["direction"] == "LONG" else (entry_p - exit_p)
                    elapsed  = _calc_minutes(position["entry_time_utc"], utc_hm)

                    completed_trades.append({
                        "direction": position["direction"],
                        "pnl_pts":  pnl_pts,
                        "reason":   reason,
                        "minutes":  elapsed,
                        "entry_time": _utc_to_bj(position["entry_time_utc"]),
                        "exit_time":  _utc_to_bj(utc_hm),
                        "partial": exit_vol < position["volume"],
                    })
                    if exit_vol >= position["volume"]:
                        last_exit_utc = utc_hm
                        last_exit_dir = position["direction"]
                        position      = None
                        action_str    = f"EXIT {reason}"
                    else:
                        position["volume"] -= exit_vol
                        action_str = f"PARTIAL {reason}"

        # ── entry check ──────────────────────────────────────────────
        in_cooldown = False
        if last_exit_utc and direction == last_exit_dir:
            cd_elapsed = _calc_minutes(last_exit_utc, utc_hm)
            if 0 < cd_elapsed < COOLDOWN_MINUTES:
                in_cooldown = True

        if (position is None and not action_str and result
                and not in_cooldown
                and score >= effective_threshold
                and direction
                and is_open_allowed(utc_hm)):
            entry_p = price + slippage if direction == "LONG" else price - slippage
            stop    = _make_stop(entry_p, direction)
            position = {
                "entry_price":    entry_p,
                "direction":      direction,
                "entry_time_utc": utc_hm,
                "highest_since":  high,
                "lowest_since":   low,
                "stop_loss":      stop,
                "volume":         1,
                "bars_below_mid": 0,
            }

    # Force close remaining position
    if position is not None:
        last_price = float(all_bars.loc[today_indices[-1], "close"])
        entry_p    = position["entry_price"]
        exit_p     = (last_price - slippage if position["direction"] == "LONG"
                      else last_price + slippage)
        pnl = (exit_p - entry_p) if position["direction"] == "LONG" else (entry_p - exit_p)
        elapsed = _calc_minutes(
            position["entry_time_utc"],
            str(all_bars.loc[today_indices[-1], "datetime"])[11:16]
        )
        completed_trades.append({
            "direction": position["direction"],
            "pnl_pts":  pnl,
            "reason":   "EOD_FORCE",
            "minutes":  elapsed,
            "entry_time": _utc_to_bj(position["entry_time_utc"]),
            "exit_time":  _utc_to_bj(
                str(all_bars.loc[today_indices[-1], "datetime"])[11:16]
            ),
        })

    return completed_trades


# ─────────────────────────────────────────────────────────────
# Pre-load all per-day data (one pass) to avoid repeated DB queries
# ─────────────────────────────────────────────────────────────

def preload_day_data(sym: str, dates: List[str], db: DBManager, version: str = "auto"):
    """Query DB once per symbol, return dict keyed by date."""
    from strategies.intraday.A_share_momentum_signal_v2 import (
        SignalGeneratorV2, SignalGeneratorV3, SentimentData,
        SIGNAL_ROUTING, SYMBOL_PROFILES, _DEFAULT_PROFILE,
    )

    _SPOT_SYM = {"IM": "000852", "IF": "000300", "IH": "000016", "IC": "000905"}
    _SPOT_IDX = {"IM": "000852.SH", "IF": "000300.SH", "IH": "000016.SH", "IC": "000905.SH"}
    spot_sym = _SPOT_SYM.get(sym, sym)
    idx_code = _SPOT_IDX.get(sym, f"{sym}.CFX")

    print(f"  Loading index_min for {spot_sym}...")
    all_bars = db.query_df(
        f"SELECT datetime, open, high, low, close, volume "
        f"FROM index_min WHERE symbol='{spot_sym}' AND period=300 ORDER BY datetime"
    )
    if all_bars is None or all_bars.empty:
        raise RuntimeError(f"No index_min data for {spot_sym}")
    for c in ["open", "high", "low", "close", "volume"]:
        all_bars[c] = all_bars[c].astype(float)

    print(f"  Loading index_daily for {idx_code}...")
    daily_all = db.query_df(
        f"SELECT trade_date, close as open, close as high, close as low, close, 0 as volume "
        f"FROM index_daily WHERE ts_code='{idx_code}' ORDER BY trade_date"
    )
    if daily_all is not None:
        daily_all["close"] = daily_all["close"].astype(float)

    spot_all = db.query_df(
        f"SELECT trade_date, close FROM index_daily WHERE ts_code='{idx_code}' ORDER BY trade_date"
    )
    if spot_all is not None:
        spot_all["close"] = spot_all["close"].astype(float)

    def _zscore_for_date(td):
        if spot_all is None or spot_all.empty:
            return 0.0, 0.0
        sub = spot_all[spot_all["trade_date"] < td].tail(30)
        if len(sub) < 20:
            return 0.0, 0.0
        closes = sub["close"].values
        ema = float(pd.Series(closes).ewm(span=20).mean().iloc[-1])
        std = float(pd.Series(closes).rolling(20).std().iloc[-1])
        return ema, std

    # Signal generator (use same version as backtest)
    _ver = version if version != "auto" else SIGNAL_ROUTING.get(sym, "v2")
    gen = SignalGeneratorV3({"min_signal_score": 60}) if _ver == "v3" else SignalGeneratorV2({"min_signal_score": 60})

    _sym_prof = SYMBOL_PROFILES.get(sym, _DEFAULT_PROFILE)
    effective_threshold = _sym_prof.get("signal_threshold", 60)

    per_day = {}
    print(f"  Precomputing per-day state for {len(dates)} dates...")
    for td in dates:
        ema20, std20 = _zscore_for_date(td)

        is_high_vol = True
        dmo = db.query_df(
            "SELECT garch_forecast_vol FROM daily_model_output "
            "WHERE underlying='IM' AND garch_forecast_vol > 0 "
            f"AND trade_date < '{td}' ORDER BY trade_date DESC LIMIT 1"
        )
        if dmo is not None and not dmo.empty:
            is_high_vol = (float(dmo.iloc[0].iloc[0]) * 100 / 24.9) > 1.2

        sentiment = None
        try:
            sdf = db.query_df(
                "SELECT atm_iv, atm_iv_market, vrp, term_structure_shape, rr_25d "
                "FROM daily_model_output WHERE underlying='IM' "
                f"AND trade_date < '{td}' ORDER BY trade_date DESC LIMIT 2"
            )
            if sdf is not None and len(sdf) >= 1:
                from strategies.intraday.A_share_momentum_signal_v2 import SentimentData
                cur, prev = sdf.iloc[0], (sdf.iloc[1] if len(sdf) >= 2 else sdf.iloc[0])
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

        d_override = None
        try:
            briefing_row = db.query_df(
                "SELECT d_override_long, d_override_short FROM morning_briefing "
                f"WHERE trade_date = '{td}' LIMIT 1"
            )
            if briefing_row is not None and len(briefing_row) > 0:
                d_long  = briefing_row.iloc[0].get("d_override_long")
                d_short = briefing_row.iloc[0].get("d_override_short")
                if d_long is not None and d_short is not None:
                    d_override = {"LONG": float(d_long), "SHORT": float(d_short)}
        except Exception:
            pass

        per_day[td] = dict(
            ema20=ema20, std20=std20,
            is_high_vol=is_high_vol,
            sentiment=sentiment,
            d_override=d_override,
        )

    return all_bars, daily_all, gen, effective_threshold, per_day


# ─────────────────────────────────────────────────────────────
# Aggregate stats
# ─────────────────────────────────────────────────────────────

def _stats(trades: List[Dict], slippage: float = 0) -> Dict:
    full = [t for t in trades if not t.get("partial")]
    if not full:
        return {"pnl": 0, "n": 0, "wr": 0.0, "avg": 0.0,
                "be_slip": 0.0, "win_days": 0}
    pnl_total = sum(t["pnl_pts"] for t in full)
    n = len(full)
    wins = sum(1 for t in full if t["pnl_pts"] > 0)
    wr   = wins / n * 100 if n > 0 else 0
    avg  = pnl_total / n
    be_slip = avg / 2 if n > 0 else 0   # breakeven slippage per side
    return {"pnl": pnl_total, "n": n, "wr": wr, "avg": avg,
            "be_slip": be_slip, "wins": wins}


def _reason_breakdown(trades: List[Dict]) -> Dict[str, int]:
    full = [t for t in trades if not t.get("partial")]
    counts: Dict[str, int] = {}
    for t in full:
        r = t.get("reason", "?")
        counts[r] = counts.get(r, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


# ─────────────────────────────────────────────────────────────
# Run one sweep group
# ─────────────────────────────────────────────────────────────

def run_sweep_group(
    group_id: str,
    sym: str,
    dates: List[str],
    all_bars: pd.DataFrame,
    daily_all: Optional[pd.DataFrame],
    gen,
    effective_threshold: int,
    per_day: Dict,
    slippage: float = 0,
) -> List[Dict]:
    """Run all values for one group. Returns list of result rows."""
    param_name, values = SWEEP_GROUPS[group_id]
    baseline_params = dict(BASELINE)

    results = []
    for val in values:
        params = dict(baseline_params)
        params[param_name] = val
        is_baseline = (val == baseline_params[param_name])

        all_trades = []
        for td in dates:
            day = per_day[td]
            trades = run_day_param(
                sym=sym, td=td, db=None,
                all_bars=all_bars, daily_all=daily_all,
                ema20=day["ema20"], std20=day["std20"],
                is_high_vol=day["is_high_vol"],
                sentiment=day["sentiment"],
                d_override=day["d_override"],
                gen=gen,
                effective_threshold=effective_threshold,
                slippage=slippage,
                stop_loss_pct=params["stop_loss_pct"],
                trail_scale=params["trail_scale"],
                trail_bonus=params["trail_bonus"],
                me_hold_min=params["me_hold_min"],
                me_ratio=params["me_ratio"],
                time_stop_min=params["time_stop_min"],
                lunch_mode=params["lunch_mode"],
                mid_break_bars=params["mid_break_bars"],
            )
            all_trades.extend(trades)

        st = _stats(all_trades)
        rb = _reason_breakdown(all_trades)
        results.append({
            "group":      group_id,
            "param":      param_name,
            "value":      val,
            "is_baseline": is_baseline,
            **st,
            "reasons":    rb,
        })

    return results


# ─────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────

def _fmt_val(val) -> str:
    if isinstance(val, float):
        if val < 0.1:
            return f"{val*100:.1f}%"
        return f"{val:.2f}"
    return str(val)


def print_group_results(group_id: str, results: List[Dict], sym: str, n_days: int):
    param_name, _ = SWEEP_GROUPS[group_id]
    desc = GROUP_DESC[group_id]
    baseline_val = BASELINE[param_name]

    print(f"\n{'═'*90}")
    print(f"  {group_id}: {desc}")
    print(f"  Symbol={sym}  |  {n_days} days  |  Baseline={_fmt_val(baseline_val)}")
    print(f"{'─'*90}")
    print(f"  {'Value':>10}  {'PnL(pt)':>8}  {'N':>4}  {'WR%':>6}  {'Avg/T':>7}  {'BE-slip':>7}  {'Status':12}  Exit breakdown")
    print(f"{'─'*90}")

    baseline_pnl = next((r["pnl"] for r in results if r["is_baseline"]), 0)

    for r in results:
        val_str   = _fmt_val(r["value"])
        baseline_marker = " ←BASE" if r["is_baseline"] else ""
        delta     = r["pnl"] - baseline_pnl
        delta_str = f"({delta:+.0f})" if not r["is_baseline"] else "        "
        status    = "BETTER" if delta > 5 and not r["is_baseline"] else ("WORSE" if delta < -5 and not r["is_baseline"] else "~")
        rb_str = "  ".join(f"{k}:{v}" for k, v in list(r["reasons"].items())[:5])
        print(
            f"  {val_str:>10}  {r['pnl']:>+8.0f} {delta_str:>8}  {r['n']:>4}"
            f"  {r['wr']:>5.1f}%  {r['avg']:>+7.1f}  {r['be_slip']:>6.1f}pt"
            f"  {status+baseline_marker:16}  {rb_str}"
        )

    # Find best
    best = max(results, key=lambda r: r["pnl"])
    if not best["is_baseline"]:
        print(f"\n  >> Best value: {_fmt_val(best['value'])} → PnL={best['pnl']:+.0f}pt"
              f"  (vs baseline {baseline_pnl:+.0f}pt,  Δ={best['pnl']-baseline_pnl:+.0f}pt)")
    else:
        print(f"\n  >> Baseline is already optimal for {group_id}")


def print_summary_table(all_results: Dict[str, List[Dict]], sym: str):
    """Cross-group summary: best value vs baseline for each group."""
    print(f"\n{'═'*75}")
    print(f"  SENSITIVITY SUMMARY — {sym}")
    print(f"{'─'*75}")
    print(f"  {'Group':5}  {'Param':22}  {'Baseline':>8}  {'Best val':>10}  {'Δ PnL':>8}  {'Verdict'}")
    print(f"{'─'*75}")

    for gid, results in all_results.items():
        param_name = SWEEP_GROUPS[gid][0]
        base_val   = BASELINE[param_name]
        base_pnl   = next((r["pnl"] for r in results if r["is_baseline"]), 0)
        best       = max(results, key=lambda r: r["pnl"])
        delta      = best["pnl"] - base_pnl
        if abs(delta) < 5:
            verdict = "STABLE"
        elif delta > 0:
            verdict = f"IMPROVE +{delta:.0f}pt → {_fmt_val(best['value'])}"
        else:
            verdict = f"BASELINE OPTIMAL ({delta:+.0f}pt)"
        print(f"  {gid:5}  {param_name:22}  {_fmt_val(base_val):>8}  {_fmt_val(best['value']):>10}  {delta:>+8.0f}  {verdict}")

    print(f"{'═'*75}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Exit参数敏感性分析")
    parser.add_argument("--symbol",   default="IM",   help="IM / IC / IF (default IM)")
    parser.add_argument("--date",     default=DEFAULT_DATES,
                        help="Comma-separated YYYYMMDD dates (default: 35-day set)")
    parser.add_argument("--group",    default="ALL",
                        help="Comma-separated group IDs: G1,G2,...,G8 or ALL")
    parser.add_argument("--slippage", type=float, default=0,
                        help="Slippage per trade in points (default 0)")
    parser.add_argument("--version",  choices=["v2", "v3", "auto"], default="auto",
                        help="Signal version (default auto)")
    args = parser.parse_args()

    dates = [d.strip() for d in args.date.split(",") if d.strip()]
    sym   = args.symbol.upper()

    if args.group.upper() == "ALL":
        groups = list(SWEEP_GROUPS.keys())
    else:
        groups = [g.strip().upper() for g in args.group.split(",")]
        invalid = [g for g in groups if g not in SWEEP_GROUPS]
        if invalid:
            print(f"Unknown groups: {invalid}. Valid: {list(SWEEP_GROUPS.keys())}")
            sys.exit(1)

    db = DBManager(ConfigLoader().get_db_path())

    print(f"\n{'═'*75}")
    print(f"  EXIT SENSITIVITY ANALYSIS — {sym}")
    print(f"  Dates: {dates[0]} ~ {dates[-1]}  ({len(dates)} days)")
    print(f"  Groups: {groups}  |  Slippage: {args.slippage}pt  |  Version: {args.version}")
    print(f"{'═'*75}")

    print(f"\nPreloading data...")
    all_bars, daily_all, gen, effective_threshold, per_day = preload_day_data(
        sym, dates, db, version=args.version
    )
    print(f"  Loaded {len(all_bars)} bars.  Threshold={effective_threshold}")

    # Baseline run (always included for reference)
    print(f"\nRunning baseline...")
    baseline_all_trades = []
    for td in dates:
        day = per_day[td]
        trades = run_day_param(
            sym=sym, td=td, db=None,
            all_bars=all_bars, daily_all=daily_all,
            ema20=day["ema20"], std20=day["std20"],
            is_high_vol=day["is_high_vol"],
            sentiment=day["sentiment"],
            d_override=day["d_override"],
            gen=gen,
            effective_threshold=effective_threshold,
            slippage=args.slippage,
            **{k: BASELINE[k] for k in BASELINE},
        )
        baseline_all_trades.extend(trades)

    bst = _stats(baseline_all_trades)
    brb = _reason_breakdown(baseline_all_trades)
    print(f"\n  BASELINE: PnL={bst['pnl']:+.0f}pt  N={bst['n']}  WR={bst['wr']:.1f}%"
          f"  Avg={bst['avg']:+.1f}pt  BE-slip={bst['be_slip']:.1f}pt")
    rb_str = "  ".join(f"{k}:{v}" for k, v in list(brb.items())[:6])
    print(f"  Exit reasons: {rb_str}")

    # Run sweeps
    all_results: Dict[str, List[Dict]] = {}
    for gid in groups:
        param_name, values = SWEEP_GROUPS[gid]
        n_combos = len(values)
        print(f"\nSweeping {gid}: {param_name}  [{n_combos} values × {len(dates)} days]...")
        results = run_sweep_group(
            gid, sym, dates, all_bars, daily_all,
            gen, effective_threshold, per_day,
            slippage=args.slippage,
        )
        all_results[gid] = results
        print_group_results(gid, results, sym, len(dates))

    # Cross-group summary
    if len(groups) > 1:
        print_summary_table(all_results, sym)


if __name__ == "__main__":
    main()
