#!/usr/bin/env python3
"""
parameter_sensitivity.py
------------------------
V2核心参数系统性敏感性分析。

每次只变一个参数组，其他保持当前值（基准值）。
通过 monkey-patch 模块级变量/SYMBOL_PROFILES 实现参数覆盖。

运行方式：
    python scripts/parameter_sensitivity.py
    python scripts/parameter_sensitivity.py --symbols IM  # 只跑IM
    python scripts/parameter_sensitivity.py --group G1    # 只跑某一组
    python scripts/parameter_sensitivity.py --fast        # 缩减到20天
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from data.storage.db_manager import DBManager
from config.config_loader import ConfigLoader

# ---------------------------------------------------------------------------
# 回测日期列表（34天干净数据）
# ---------------------------------------------------------------------------

ALL_DATES = [
    "20260204","20260205","20260206","20260209","20260210","20260211","20260212","20260213",
    "20260225","20260226","20260227",
    "20260302","20260303","20260304","20260305","20260306",
    "20260309","20260310","20260311","20260312","20260313",
    "20260316","20260317","20260318","20260319","20260320",
    "20260323","20260324","20260325","20260326","20260327","20260328",
    "20260401","20260402",
]

# 快速模式用20天（跳过部分较早的日期）
FAST_DATES = ALL_DATES[-20:]

# 品种乘数（IM=200, IC=200）
CONTRACT_MULT = {"IM": 200, "IC": 200}

# 每个品种的基准阈值
BASE_THRESHOLD = {"IM": 60, "IC": 65}


# ---------------------------------------------------------------------------
# 基准参数（当前生产值）
# ---------------------------------------------------------------------------

BASE_PARAMS = {
    # --- 止损 ---
    "STOP_LOSS_PCT":         0.005,

    # --- Trailing Stop 阶梯（持仓时间 → 止盈宽度）---
    "trail_0_15":    0.005,   # <15min
    "trail_15_30":   0.006,   # 15-30min
    "trail_30_60":   0.008,   # 30-60min
    "trail_60_plus": 0.010,   # >60min

    # --- 午休跟踪止盈 ---
    "TRAILING_STOP_LUNCH":   0.003,

    # --- MOMENTUM_EXHAUSTED 最小持仓分钟 ---
    "ME_MIN_HOLD_MINUTES":   20,

    # --- 动量lookback（IM/IC，5m bar数量）---
    "IM_momentum_lookback_5m": 12,
    "IC_momentum_lookback_5m": 12,

    # --- 日内涨跌幅过滤阈值（高波动区间）---
    "intraday_thr_hi_1":   0.01,  # 1% 轻度
    "intraday_thr_hi_2":   0.02,  # 2% 中度
    "intraday_thr_hi_3":   0.03,  # 3% 极端

    # --- 日线逆势/顺势乘数（IM/IC）---
    "IM_dm_trend":      1.2,
    "IM_dm_contrarian": 0.8,
    "IC_dm_trend":      1.2,
    "IC_dm_contrarian": 0.8,

    # --- 成交量得分阈值 ---
    "VOLUME_SURGE_RATIO": 1.5,
    "VOLUME_LOW_RATIO":   0.5,

    # --- 动量得分断点（5m mom_5m绝对值 → base分）---
    "mom_thr_35": 0.003,   # >0.003 → 35分
    "mom_thr_25": 0.002,   # >0.002 → 25分
    "mom_thr_15": 0.001,   # >0.001 → 15分

    # --- 波动率(ATR)阈值（ATR_SHORT/ATR_LONG ratio）---
    "atr_ratio_30": 0.7,   # <0.7 → 30分
    "atr_ratio_25": 0.9,   # <0.9 → 25分
    "atr_ratio_15": 1.1,   # <1.1 → 15分
    "atr_ratio_5":  1.5,   # <1.5 → 5分
}


# ---------------------------------------------------------------------------
# 参数组定义
# ---------------------------------------------------------------------------

PARAM_GROUPS: Dict[str, Dict] = {
    "G1_stop_loss": {
        "desc": "止损幅度 STOP_LOSS_PCT",
        "values": [
            {"STOP_LOSS_PCT": 0.003},
            {"STOP_LOSS_PCT": 0.004},
            {"STOP_LOSS_PCT": 0.005},   # ← 基准
            {"STOP_LOSS_PCT": 0.006},
            {"STOP_LOSS_PCT": 0.008},
        ],
        "label_fn": lambda p: f"SL={p['STOP_LOSS_PCT']*100:.1f}%",
    },
    "G2_trailing_stop": {
        "desc": "Trailing stop阶梯宽度（整体同倍缩放）",
        "values": [
            {"trail_0_15": 0.003, "trail_15_30": 0.004, "trail_30_60": 0.006, "trail_60_plus": 0.008},
            {"trail_0_15": 0.004, "trail_15_30": 0.005, "trail_30_60": 0.007, "trail_60_plus": 0.009},
            {"trail_0_15": 0.005, "trail_15_30": 0.006, "trail_30_60": 0.008, "trail_60_plus": 0.010},  # ← 基准
            {"trail_0_15": 0.006, "trail_15_30": 0.007, "trail_30_60": 0.010, "trail_60_plus": 0.012},
            {"trail_0_15": 0.008, "trail_15_30": 0.010, "trail_30_60": 0.012, "trail_60_plus": 0.015},
        ],
        "label_fn": lambda p: f"trail={p['trail_0_15']*100:.1f}%-{p['trail_60_plus']*100:.1f}%",
    },
    "G3_me_min_hold": {
        "desc": "MOMENTUM_EXHAUSTED最小持仓时间（分钟）",
        "values": [
            {"ME_MIN_HOLD_MINUTES": 0},
            {"ME_MIN_HOLD_MINUTES": 10},
            {"ME_MIN_HOLD_MINUTES": 20},   # ← 基准
            {"ME_MIN_HOLD_MINUTES": 30},
            {"ME_MIN_HOLD_MINUTES": 45},
        ],
        "label_fn": lambda p: f"ME_min={p['ME_MIN_HOLD_MINUTES']}min",
    },
    "G4_momentum_lookback": {
        "desc": "动量lookback（5m bar数）",
        "values": [
            {"IM_momentum_lookback_5m": 6,  "IC_momentum_lookback_5m": 6},
            {"IM_momentum_lookback_5m": 9,  "IC_momentum_lookback_5m": 9},
            {"IM_momentum_lookback_5m": 12, "IC_momentum_lookback_5m": 12},   # ← 基准
            {"IM_momentum_lookback_5m": 18, "IC_momentum_lookback_5m": 18},
            {"IM_momentum_lookback_5m": 24, "IC_momentum_lookback_5m": 24},
        ],
        "label_fn": lambda p: f"lb5m={p['IM_momentum_lookback_5m']}",
    },
    "G5_daily_mult": {
        "desc": "日线方向乘数（顺势/逆势）",
        "values": [
            {"IM_dm_trend": 1.0, "IM_dm_contrarian": 1.0, "IC_dm_trend": 1.0, "IC_dm_contrarian": 1.0},
            {"IM_dm_trend": 1.1, "IM_dm_contrarian": 0.9, "IC_dm_trend": 1.1, "IC_dm_contrarian": 0.9},
            {"IM_dm_trend": 1.2, "IM_dm_contrarian": 0.8, "IC_dm_trend": 1.2, "IC_dm_contrarian": 0.8},  # ← 基准
            {"IM_dm_trend": 1.3, "IM_dm_contrarian": 0.7, "IC_dm_trend": 1.3, "IC_dm_contrarian": 0.7},
            {"IM_dm_trend": 1.5, "IM_dm_contrarian": 0.5, "IC_dm_trend": 1.5, "IC_dm_contrarian": 0.5},
        ],
        "label_fn": lambda p: f"dm={p['IM_dm_trend']:.1f}/{p['IM_dm_contrarian']:.1f}",
    },
    "G6_intraday_filter": {
        "desc": "日内涨跌幅阈值（高波动区间1/2/3%断点）",
        "values": [
            {"intraday_thr_hi_1": 0.005, "intraday_thr_hi_2": 0.015, "intraday_thr_hi_3": 0.025},
            {"intraday_thr_hi_1": 0.008, "intraday_thr_hi_2": 0.018, "intraday_thr_hi_3": 0.028},
            {"intraday_thr_hi_1": 0.010, "intraday_thr_hi_2": 0.020, "intraday_thr_hi_3": 0.030},  # ← 基准
            {"intraday_thr_hi_1": 0.012, "intraday_thr_hi_2": 0.025, "intraday_thr_hi_3": 0.035},
            {"intraday_thr_hi_1": 0.015, "intraday_thr_hi_2": 0.030, "intraday_thr_hi_3": 0.040},
        ],
        "label_fn": lambda p: f"thr={p['intraday_thr_hi_1']*100:.1f}%/{p['intraday_thr_hi_2']*100:.1f}%/{p['intraday_thr_hi_3']*100:.1f}%",
    },
    "G7_volume_ratio": {
        "desc": "成交量放量阈值 VOLUME_SURGE_RATIO",
        "values": [
            {"VOLUME_SURGE_RATIO": 1.2, "VOLUME_LOW_RATIO": 0.4},
            {"VOLUME_SURGE_RATIO": 1.3, "VOLUME_LOW_RATIO": 0.45},
            {"VOLUME_SURGE_RATIO": 1.5, "VOLUME_LOW_RATIO": 0.5},   # ← 基准
            {"VOLUME_SURGE_RATIO": 1.8, "VOLUME_LOW_RATIO": 0.6},
            {"VOLUME_SURGE_RATIO": 2.0, "VOLUME_LOW_RATIO": 0.7},
        ],
        "label_fn": lambda p: f"surge={p['VOLUME_SURGE_RATIO']:.1f}",
    },
    "G8_mom_thresholds": {
        "desc": "动量得分断点（5m绝对涨幅）",
        "values": [
            {"mom_thr_35": 0.002, "mom_thr_25": 0.0015, "mom_thr_15": 0.0008},
            {"mom_thr_35": 0.0025, "mom_thr_25": 0.0018, "mom_thr_15": 0.0009},
            {"mom_thr_35": 0.003, "mom_thr_25": 0.002,  "mom_thr_15": 0.001},   # ← 基准
            {"mom_thr_35": 0.004, "mom_thr_25": 0.003,  "mom_thr_15": 0.0015},
            {"mom_thr_35": 0.005, "mom_thr_25": 0.004,  "mom_thr_15": 0.002},
        ],
        "label_fn": lambda p: f"mom35={p['mom_thr_35']*100:.2f}%",
    },
}


# ---------------------------------------------------------------------------
# Monkey-patch helpers
# ---------------------------------------------------------------------------

def _apply_params(params: Dict) -> None:
    """Apply parameter overrides to signal module via monkey-patch."""
    import strategies.intraday.A_share_momentum_signal_v2 as sig

    # --- 止损 ---
    if "STOP_LOSS_PCT" in params:
        sig.STOP_LOSS_PCT = params["STOP_LOSS_PCT"]

    # --- Trailing stop ---
    # We override check_exit's trail_pct ladder by patching the module-level
    # constants TRAILING_STOP_HIVOL / TRAILING_STOP_NORMAL for the 0.8/0.5%
    # fallback, but the real ladder is hardcoded inside check_exit.
    # We patch check_exit itself via closure.
    trail_keys = {"trail_0_15", "trail_15_30", "trail_30_60", "trail_60_plus"}
    if trail_keys & set(params.keys()):
        t0 = params.get("trail_0_15",  BASE_PARAMS["trail_0_15"])
        t1 = params.get("trail_15_30", BASE_PARAMS["trail_15_30"])
        t2 = params.get("trail_30_60", BASE_PARAMS["trail_30_60"])
        t3 = params.get("trail_60_plus", BASE_PARAMS["trail_60_plus"])
        _patch_trailing_stop(sig, t0, t1, t2, t3)

    if "TRAILING_STOP_LUNCH" in params:
        sig.TRAILING_STOP_LUNCH = params["TRAILING_STOP_LUNCH"]

    # --- ME_MIN_HOLD_MINUTES ---
    # Hardcoded in check_exit as `hold_minutes >= 20`. Patch via function override.
    if "ME_MIN_HOLD_MINUTES" in params:
        _patch_me_min_hold(sig, params["ME_MIN_HOLD_MINUTES"])

    # --- Momentum lookback ---
    if "IM_momentum_lookback_5m" in params:
        sig.SYMBOL_PROFILES["IM"]["momentum_lookback_5m"] = params["IM_momentum_lookback_5m"]
    if "IC_momentum_lookback_5m" in params:
        sig.SYMBOL_PROFILES["IC"]["momentum_lookback_5m"] = params["IC_momentum_lookback_5m"]

    # --- Daily mult ---
    if "IM_dm_trend" in params:
        sig.SYMBOL_PROFILES["IM"]["dm_trend"] = params["IM_dm_trend"]
    if "IM_dm_contrarian" in params:
        sig.SYMBOL_PROFILES["IM"]["dm_contrarian"] = params["IM_dm_contrarian"]
    if "IC_dm_trend" in params:
        sig.SYMBOL_PROFILES["IC"]["dm_trend"] = params["IC_dm_trend"]
    if "IC_dm_contrarian" in params:
        sig.SYMBOL_PROFILES["IC"]["dm_contrarian"] = params["IC_dm_contrarian"]

    # --- Volume ratio ---
    if "VOLUME_SURGE_RATIO" in params:
        sig.VOLUME_SURGE_RATIO = params["VOLUME_SURGE_RATIO"]
    if "VOLUME_LOW_RATIO" in params:
        sig.VOLUME_LOW_RATIO = params["VOLUME_LOW_RATIO"]

    # --- Momentum score thresholds ---
    mom_keys = {"mom_thr_35", "mom_thr_25", "mom_thr_15"}
    if mom_keys & set(params.keys()):
        t35 = params.get("mom_thr_35", BASE_PARAMS["mom_thr_35"])
        t25 = params.get("mom_thr_25", BASE_PARAMS["mom_thr_25"])
        t15 = params.get("mom_thr_15", BASE_PARAMS["mom_thr_15"])
        _patch_score_momentum(sig, t35, t25, t15)

    # --- Intraday filter thresholds ---
    idf_keys = {"intraday_thr_hi_1", "intraday_thr_hi_2", "intraday_thr_hi_3"}
    if idf_keys & set(params.keys()):
        thr1 = params.get("intraday_thr_hi_1", BASE_PARAMS["intraday_thr_hi_1"])
        thr2 = params.get("intraday_thr_hi_2", BASE_PARAMS["intraday_thr_hi_2"])
        thr3 = params.get("intraday_thr_hi_3", BASE_PARAMS["intraday_thr_hi_3"])
        _patch_intraday_filter(sig, thr1, thr2, thr3)


def _restore_params() -> None:
    """Restore all patched parameters to baseline values."""
    import importlib
    import strategies.intraday.A_share_momentum_signal_v2 as sig

    sig.STOP_LOSS_PCT = BASE_PARAMS["STOP_LOSS_PCT"]
    sig.TRAILING_STOP_LUNCH = BASE_PARAMS["TRAILING_STOP_LUNCH"]
    sig.VOLUME_SURGE_RATIO = BASE_PARAMS["VOLUME_SURGE_RATIO"]
    sig.VOLUME_LOW_RATIO = BASE_PARAMS["VOLUME_LOW_RATIO"]

    # Restore profile values
    sig.SYMBOL_PROFILES["IM"]["momentum_lookback_5m"] = BASE_PARAMS["IM_momentum_lookback_5m"]
    sig.SYMBOL_PROFILES["IC"]["momentum_lookback_5m"] = BASE_PARAMS["IC_momentum_lookback_5m"]
    sig.SYMBOL_PROFILES["IM"]["dm_trend"] = BASE_PARAMS["IM_dm_trend"]
    sig.SYMBOL_PROFILES["IM"]["dm_contrarian"] = BASE_PARAMS["IM_dm_contrarian"]
    sig.SYMBOL_PROFILES["IC"]["dm_trend"] = BASE_PARAMS["IC_dm_trend"]
    sig.SYMBOL_PROFILES["IC"]["dm_contrarian"] = BASE_PARAMS["IC_dm_contrarian"]

    # Restore patched functions to originals
    _restore_check_exit(sig)
    _restore_score_momentum(sig)
    _restore_intraday_filter(sig)


# We keep references to the original functions at import time
import strategies.intraday.A_share_momentum_signal_v2 as _sig_module
_ORIG_CHECK_EXIT = _sig_module.check_exit
_ORIG_SCORE_MOMENTUM_METHOD = None   # will be set after class instantiation check
_ORIG_INTRADAY_FILTER = None


def _patch_trailing_stop(sig, t0: float, t1: float, t2: float, t3: float):
    """Replace check_exit with version using new trailing stop ladder."""
    import numpy as np
    import pandas as pd
    from strategies.intraday.A_share_momentum_signal_v2 import (
        _boll_zone, _calc_boll, SYMBOL_PROFILES, _DEFAULT_PROFILE,
        EOD_CLOSE_UTC, LUNCH_CLOSE_UTC,
    )

    def _patched_check_exit(
        position, current_price, bar_5m, bar_15m, current_time_utc,
        reverse_signal_score=0, is_high_vol=True, symbol="", spot_price=0.0,
    ):
        entry_price = position["entry_price"]
        direction = position["direction"]
        entry_time = position.get("entry_time_utc", "")
        highest = position.get("highest_since", entry_price)
        lowest = position.get("lowest_since", entry_price)
        volume = position.get("volume", 1)

        boll_price = spot_price if spot_price > 0 else current_price
        NO_EXIT = {"should_exit": False, "exit_volume": 0,
                   "exit_reason": "", "exit_urgency": "NORMAL"}

        # P1: EOD
        if current_time_utc >= EOD_CLOSE_UTC:
            return {"should_exit": True, "exit_volume": volume,
                    "exit_reason": "EOD_CLOSE", "exit_urgency": "URGENT"}

        # P1b: Stop loss
        if direction == "LONG":
            loss_pct = (entry_price - current_price) / entry_price
        else:
            loss_pct = (current_price - entry_price) / entry_price
        if loss_pct > sig.STOP_LOSS_PCT:
            return {"should_exit": True, "exit_volume": volume,
                    "exit_reason": "STOP_LOSS", "exit_urgency": "URGENT"}

        # P2: Lunch close
        if current_time_utc >= LUNCH_CLOSE_UTC and current_time_utc < "05:00":
            if direction == "LONG":
                profitable = current_price > entry_price
            else:
                profitable = current_price < entry_price
            if not profitable:
                return {"should_exit": True, "exit_volume": volume,
                        "exit_reason": "LUNCH_CLOSE", "exit_urgency": "URGENT"}
            if direction == "LONG" and highest > entry_price:
                dd = (highest - current_price) / highest
                if dd > sig.TRAILING_STOP_LUNCH:
                    return {"should_exit": True, "exit_volume": volume,
                            "exit_reason": "LUNCH_TRAIL", "exit_urgency": "NORMAL"}
            elif direction == "SHORT" and lowest < entry_price:
                du = (current_price - lowest) / lowest
                if du > sig.TRAILING_STOP_LUNCH:
                    return {"should_exit": True, "exit_volume": volume,
                            "exit_reason": "LUNCH_TRAIL", "exit_urgency": "NORMAL"}

        # Bollinger bands
        b5_mid, b5_std = float("nan"), float("nan")
        b15_mid, b15_std = float("nan"), float("nan")
        zone_5m = ""
        zone_15m = ""

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

        hold_minutes = 0
        if entry_time and current_time_utc:
            try:
                h1, m1 = int(entry_time[:2]), int(entry_time[3:5])
                h2, m2 = int(current_time_utc[:2]), int(current_time_utc[3:5])
                hold_minutes = (h2 * 60 + m2) - (h1 * 60 + m1)
            except Exception:
                pass

        # P3: Dynamic trailing stop — NEW LADDER
        _prof = SYMBOL_PROFILES.get(symbol, _DEFAULT_PROFILE) if symbol else _DEFAULT_PROFILE
        if _prof.get("trailing_stop_enabled", True):
            if hold_minutes < 15:
                trail_pct = t0
            elif hold_minutes < 30:
                trail_pct = t1
            elif hold_minutes < 60:
                trail_pct = t2
            else:
                trail_pct = t3

            # Bonus: profitable >0.5% + 15m trend confirmed
            if zone_15m:
                if direction == "LONG":
                    pnl_pct = (current_price - entry_price) / entry_price
                    fifteen_ok = zone_15m in ("MID_UPPER", "UPPER_ZONE", "ABOVE_UPPER")
                else:
                    pnl_pct = (entry_price - current_price) / entry_price
                    fifteen_ok = zone_15m in ("MID_LOWER", "LOWER_ZONE", "BELOW_LOWER")
                if pnl_pct > 0.005 and fifteen_ok:
                    trail_pct += 0.002

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

        # P5: Momentum exhausted — uses patched min hold from ME_MIN_HOLD_MINUTES
        # (check_exit uses module-level _ME_MIN_HOLD, see _patch_me_min_hold)
        me_min = getattr(sig, "_ME_MIN_HOLD_MINUTES", 20)
        if hold_minutes >= me_min and bar_5m is not None and len(bar_5m) >= 23 and b5_std > 0:
            last3_c = bar_5m["close"].astype(float).iloc[-3:]
            last3_h = bar_5m["high"].astype(float).iloc[-3:]
            last3_l = bar_5m["low"].astype(float).iloc[-3:]
            total_range = float(last3_h.max() - last3_l.min())
            boll_width = 4 * b5_std
            if total_range < boll_width * 0.20:
                close_change = float(last3_c.iloc[-1]) - float(last3_c.iloc[0])
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
                five_below = boll_price < b5_mid
                fifteen_below = zone_15m in ("MID_LOWER", "LOWER_ZONE", "BELOW_LOWER") if zone_15m else False
            else:
                five_below = boll_price > b5_mid
                fifteen_below = zone_15m in ("MID_UPPER", "UPPER_ZONE", "ABOVE_UPPER") if zone_15m else False
            if five_below:
                bars_below_mid = position.get("bars_below_mid", 0) + 1
                position["bars_below_mid"] = bars_below_mid
                if bars_below_mid >= 2 and fifteen_below:
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
                if elapsed > sig.TIME_STOP_MINUTES:
                    profitable = (current_price > entry_price) if direction == "LONG" \
                        else (current_price < entry_price)
                    if not profitable:
                        return {"should_exit": True, "exit_volume": volume,
                                "exit_reason": "TIME_STOP", "exit_urgency": "NORMAL"}
            except Exception:
                pass

        return NO_EXIT

    sig.check_exit = _patched_check_exit


def _restore_check_exit(sig):
    sig.check_exit = _ORIG_CHECK_EXIT


def _patch_me_min_hold(sig, minutes: int):
    """Patch ME_MIN_HOLD_MINUTES used by check_exit."""
    # We store on module for use in _patch_trailing_stop's closure
    sig._ME_MIN_HOLD_MINUTES = minutes
    # Also need to patch check_exit to respect this value —
    # call _patch_trailing_stop with current trail values to rebuild check_exit
    _patch_trailing_stop(
        sig,
        BASE_PARAMS["trail_0_15"],
        BASE_PARAMS["trail_15_30"],
        BASE_PARAMS["trail_30_60"],
        BASE_PARAMS["trail_60_plus"],
    )


def _patch_score_momentum(sig, thr35: float, thr25: float, thr15: float):
    """Patch SignalGeneratorV2._score_momentum thresholds."""
    original_method = sig.SignalGeneratorV2._score_momentum

    def _patched_score_momentum(self, close_5m, bar_15m, daily_bar):
        import numpy as np
        lb = sig.MOM_5M_LOOKBACK
        if len(close_5m) < lb + 1:
            return 0, ""
        mom_5m = (close_5m[-1] - close_5m[-lb - 1]) / close_5m[-lb - 1]
        dir_5m = "LONG" if mom_5m > 0 else "SHORT" if mom_5m < 0 else ""
        mom_15m = 0.0
        dir_15m = ""
        if bar_15m is not None and len(bar_15m) >= sig.MOM_15M_LOOKBACK + 1:
            c15 = bar_15m["close"].values
            mom_15m = (c15[-1] - c15[-sig.MOM_15M_LOOKBACK - 1]) / c15[-sig.MOM_15M_LOOKBACK - 1]
            dir_15m = "LONG" if mom_15m > 0 else "SHORT" if mom_15m < 0 else ""
        if dir_15m and dir_5m != dir_15m:
            return 0, ""
        abs_mom = abs(mom_5m)
        if abs_mom > thr35:
            base = 35
        elif abs_mom > thr25:
            base = 25
        elif abs_mom > thr15:
            base = 15
        else:
            base = 0
        consistency_bonus = 15 if dir_15m == dir_5m and dir_15m else 0
        return min(50, base + consistency_bonus), dir_5m

    sig.SignalGeneratorV2._score_momentum = _patched_score_momentum
    sig.SignalGeneratorV2._score_momentum._original = original_method


def _restore_score_momentum(sig):
    """Restore original _score_momentum."""
    m = sig.SignalGeneratorV2._score_momentum
    if hasattr(m, "_original"):
        sig.SignalGeneratorV2._score_momentum = m._original


def _patch_intraday_filter(sig, thr1: float, thr2: float, thr3: float):
    """Patch _intraday_filter static method thresholds."""
    original = sig.SignalGeneratorV2._intraday_filter

    @staticmethod
    def _patched_intraday_filter(intraday_return: float, direction: str,
                                  zscore=None) -> float:
        abs_ret = abs(intraday_return)
        if abs_ret < thr1:
            return 1.0
        z = zscore if zscore is not None else 0.0
        if intraday_return > thr3:
            base = 0.8 if direction == "LONG" else 0.3
        elif intraday_return > thr2:
            base = 0.9 if direction == "LONG" else 0.5
        elif intraday_return > thr1:
            base = 1.0 if direction == "LONG" else 0.7
        elif intraday_return < -thr3:
            base = 0.8 if direction == "SHORT" else (0.7 if z < -2.0 else 0.3)
        elif intraday_return < -thr2:
            base = 0.9 if direction == "SHORT" else (0.8 if z < -2.0 else 0.5)
        elif intraday_return < -thr1:
            base = 1.0 if direction == "SHORT" else (1.0 if z < -2.0 else 0.7)
        else:
            base = 1.0
        return base

    _patched_intraday_filter._original = original
    sig.SignalGeneratorV2._intraday_filter = _patched_intraday_filter


def _restore_intraday_filter(sig):
    m = sig.SignalGeneratorV2._intraday_filter
    if hasattr(m, "_original"):
        sig.SignalGeneratorV2._intraday_filter = m._original


# ---------------------------------------------------------------------------
# Single-run backtest (reuse run_day from backtest_signals_day)
# ---------------------------------------------------------------------------

def _run_backtest(sym: str, dates: List[str], db: DBManager,
                  threshold: Optional[int] = None) -> Dict:
    """Run backtest for one symbol/parameter combo. Returns metrics dict."""
    from scripts.backtest_signals_day import run_day, _SIGNAL_THRESHOLD

    # Temporarily override threshold for this run
    import scripts.backtest_signals_day as bsd
    orig_thr = bsd._SIGNAL_THRESHOLD
    if threshold is not None:
        bsd._SIGNAL_THRESHOLD = threshold

    all_trades = []
    for td in dates:
        try:
            trades = run_day(sym, td, db, verbose=False, slippage=0, version="v2")
            all_trades.extend(trades)
        except Exception as e:
            pass

    bsd._SIGNAL_THRESHOLD = orig_thr

    # Compute metrics
    full_trades = [t for t in all_trades if not t.get("partial")]
    total_pnl_pts = sum(t["pnl_pts"] for t in full_trades)
    n_trades = len(full_trades)
    wins = [t for t in full_trades if t["pnl_pts"] > 0]
    losses = [t for t in full_trades if t["pnl_pts"] <= 0]
    win_rate = len(wins) / n_trades if n_trades > 0 else 0.0
    avg_win = float(np.mean([t["pnl_pts"] for t in wins])) if wins else 0.0
    avg_loss = abs(float(np.mean([t["pnl_pts"] for t in losses]))) if losses else 1.0
    pl_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0
    avg_pnl = total_pnl_pts / n_trades if n_trades > 0 else 0.0
    breakeven_slip = avg_pnl / 2 if avg_pnl > 0 else 0.0
    mult = CONTRACT_MULT.get(sym, 200)
    yuan = total_pnl_pts * mult

    return {
        "pnl_pts": total_pnl_pts,
        "yuan": yuan,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "pl_ratio": pl_ratio,
        "avg_pnl": avg_pnl,
        "breakeven_slip": breakeven_slip,
    }


# ---------------------------------------------------------------------------
# Run one parameter group
# ---------------------------------------------------------------------------

def run_group(
    group_name: str,
    group_def: Dict,
    symbols: List[str],
    dates: List[str],
    db: DBManager,
) -> Dict:
    """
    Run all parameter values for one group, returns results dict.
    Returns: {sym: [(label, metrics), ...]}
    """
    results: Dict[str, List[Tuple[str, Dict]]] = {sym: [] for sym in symbols}
    label_fn = group_def["label_fn"]
    values = group_def["values"]

    print(f"\n{'─'*70}")
    print(f"  {group_name}: {group_def['desc']}")
    print(f"  {len(values)} values × {len(symbols)} symbols × {len(dates)} days")
    print(f"{'─'*70}")

    for i, param_override in enumerate(values):
        # Merge with base params for complete picture
        full_params = {**BASE_PARAMS, **param_override}
        label = label_fn(param_override)

        # Apply patch
        _restore_params()
        _apply_params(param_override)

        t0 = time.time()
        sym_metrics = {}
        for sym in symbols:
            thr = BASE_THRESHOLD.get(sym, 60)
            m = _run_backtest(sym, dates, db, threshold=thr)
            sym_metrics[sym] = m

        elapsed = time.time() - t0
        base_marker = " ← BASE" if i == _find_base_idx(group_def) else ""

        # Print row for each symbol
        for sym in symbols:
            m = sym_metrics[sym]
            results[sym].append((label, m))
            print(f"  [{sym}] {label:42s} | "
                  f"PnL={m['pnl_pts']:+6.0f}pt  yuan={m['yuan']:>+10,.0f}  "
                  f"n={m['n_trades']:3d}  WR={m['win_rate']*100:4.1f}%  "
                  f"BE={m['breakeven_slip']:.1f}pt"
                  f"{base_marker}")

        print(f"  {'':42s}   ({elapsed:.1f}s)")

    # Restore to base
    _restore_params()
    return results


def _find_base_idx(group_def: Dict) -> int:
    """Find index of the baseline parameter set within a group."""
    values = group_def["values"]
    label_fn = group_def["label_fn"]
    # Check against BASE_PARAMS
    for i, pov in enumerate(values):
        is_base = all(
            abs(float(pov.get(k, BASE_PARAMS.get(k, 0))) - float(BASE_PARAMS.get(k, 0))) < 1e-9
            for k in pov.keys()
        )
        if is_base:
            return i
    return -1


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(all_group_results: Dict[str, Dict], symbols: List[str]) -> None:
    """Print per-group summary: best value vs base, delta, verdict."""
    print(f"\n\n{'═'*80}")
    print(f"  PARAMETER SENSITIVITY SUMMARY")
    print(f"{'═'*80}")

    for sym in symbols:
        print(f"\n  ── {sym} (baseline thr={BASE_THRESHOLD.get(sym, 60)}) ──")
        print(f"  {'Group':20s}  {'Best':42s}  {'BestPnL':>8}  {'BasePnL':>8}  {'Delta':>8}  {'Verdict'}")
        print(f"  {'─'*20}  {'─'*42}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*12}")

        for grp_name, grp_results in all_group_results.items():
            sym_rows = grp_results.get(sym, [])
            if not sym_rows:
                continue

            # Find base row
            base_idx = _find_base_idx(PARAM_GROUPS[grp_name])
            base_pnl = sym_rows[base_idx][1]["pnl_pts"] if 0 <= base_idx < len(sym_rows) else 0.0
            best_label, best_m = max(sym_rows, key=lambda x: x[1]["pnl_pts"])
            best_pnl = best_m["pnl_pts"]
            delta = best_pnl - base_pnl

            if delta > 20:
                verdict = "IMPROVE ↑"
            elif delta < -20:
                verdict = "DEGRADE ↓"
            else:
                verdict = "STABLE ─"

            print(f"  {grp_name:20s}  {best_label:42s}  {best_pnl:>+8.0f}  "
                  f"{base_pnl:>+8.0f}  {delta:>+8.0f}  {verdict}")

    # Overall recommendation
    print(f"\n{'─'*80}")
    print(f"  RECOMMENDATIONS")
    print(f"{'─'*80}")
    for grp_name, grp_results in all_group_results.items():
        for sym in symbols:
            sym_rows = grp_results.get(sym, [])
            if not sym_rows:
                continue
            base_idx = _find_base_idx(PARAM_GROUPS[grp_name])
            base_pnl = sym_rows[base_idx][1]["pnl_pts"] if 0 <= base_idx < len(sym_rows) else 0.0
            best_label, best_m = max(sym_rows, key=lambda x: x[1]["pnl_pts"])
            best_pnl = best_m["pnl_pts"]
            delta = best_pnl - base_pnl
            base_label = sym_rows[base_idx][0] if 0 <= base_idx < len(sym_rows) else "?"
            if delta > 20:
                pct = delta / abs(base_pnl) * 100 if base_pnl != 0 else float("inf")
                print(f"  [{sym}] {grp_name}: 优化空间 {delta:+.0f}pt ({pct:.0f}%) — "
                      f"当前={base_label} → 建议={best_label}")

    print()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="V2 parameter sensitivity analysis")
    parser.add_argument("--symbols", default="IM,IC",
                        help="Comma-separated symbols (default: IM,IC)")
    parser.add_argument("--group", default=None,
                        help="Run only this group (e.g. G1_stop_loss). Default: all groups")
    parser.add_argument("--fast", action="store_true",
                        help="Use only last 20 dates instead of all 34")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]
    dates = FAST_DATES if args.fast else ALL_DATES

    print(f"\n{'═'*80}")
    print(f"  V2 PARAMETER SENSITIVITY ANALYSIS")
    print(f"  Symbols: {', '.join(symbols)}")
    print(f"  Dates: {len(dates)} days ({dates[0]} ~ {dates[-1]})")
    print(f"  {'Fast mode (20 days)' if args.fast else 'Full mode (34 days)'}")
    print(f"  Baseline thresholds: {BASE_THRESHOLD}")
    print(f"{'═'*80}")

    db = DBManager(ConfigLoader().get_db_path())

    # Select groups to run
    if args.group:
        groups_to_run = {k: v for k, v in PARAM_GROUPS.items() if k == args.group}
        if not groups_to_run:
            print(f"ERROR: group '{args.group}' not found. Available: {list(PARAM_GROUPS.keys())}")
            sys.exit(1)
    else:
        groups_to_run = PARAM_GROUPS

    # Pre-import to ensure module is loaded
    import strategies.intraday.A_share_momentum_signal_v2  # noqa
    import scripts.backtest_signals_day  # noqa

    all_group_results: Dict[str, Dict] = {}
    total_start = time.time()

    for grp_name, grp_def in groups_to_run.items():
        grp_results = run_group(grp_name, grp_def, symbols, dates, db)
        all_group_results[grp_name] = grp_results

    total_elapsed = time.time() - total_start
    print(f"\n  Total elapsed: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")

    # Print summary
    print_summary(all_group_results, symbols)


if __name__ == "__main__":
    main()
