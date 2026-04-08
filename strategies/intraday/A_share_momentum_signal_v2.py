"""
A_share_momentum_signal_v2.py
------------------------------
日内信号系统 v2 + v3。

v2：统一三维度评分（动量50/波动率30/成交量20）。
v3：品种差异化策略，基于 momentum_research.py 实证结果。
    - IM: 纯动量（AC(1)=+0.065, 1天回看夏普0.72）
    - IH: 均值回归为主（AC(1)=-0.021, 短期动量负, 涨>1%回调58%）
    - IF: 混合模式（上午动量79.4%一致性, 跌>2%反弹75%）
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from strategies.intraday.signal import IntradaySignal


# ---------------------------------------------------------------------------
# 期权情绪乘数（0.5 ~ 1.5）
# ---------------------------------------------------------------------------

@dataclass
class SentimentData:
    """期权情绪指标数据（来自 vol_monitor_snapshots 或 daily_model_output）。"""
    atm_iv: float = 0.0           # 市场ATM IV（期货价格based）
    atm_iv_prev: float = 0.0      # 昨日ATM IV
    rr_25d: float = 0.0           # 25D Risk Reversal（正=看跌偏向）
    rr_25d_prev: float = 0.0      # 昨日 RR
    pcr: float = 1.0              # Put/Call 成交量比
    term_structure: str = ""       # 期限结构形态（"正常升水"/"倒挂"等）
    vrp: float = 0.0              # 波动率风险溢价


def calc_sentiment_multiplier(
    direction: str,
    sentiment: SentimentData | None,
) -> Tuple[float, str]:
    """计算期权情绪乘数（0.5~1.5）。

    规则：
    1. IV 动态：IV 急升 + 做多 → 减分（恐慌做多危险）；IV 急降 + 做空 → 减分
    2. Skew 变化：RR 走强(看跌加剧) + 做多 → 减分；RR 走弱 + 做空 → 减分
    3. PCR（逆向指标）：>1.5 极度悲观 → 做多加分；<0.7 极度乐观 → 做空加分
    4. 期限结构倒挂 → 整体减分（市场恐慌）
    5. VRP 异常：VRP<0 → 做空波动率危险 → 减分

    Returns: (multiplier, reason_str)
    """
    if sentiment is None or not direction:
        return 1.0, ""

    adjustments: list[Tuple[float, str]] = []

    # --- 1. IV 动态 ---
    if sentiment.atm_iv > 0 and sentiment.atm_iv_prev > 0:
        iv_chg_pp = (sentiment.atm_iv - sentiment.atm_iv_prev) * 100
        if abs(iv_chg_pp) > 1.0:  # IV 变动 > 1pp
            if iv_chg_pp > 2.0:
                # IV 急升（恐慌）
                if direction == "LONG":
                    adjustments.append((-0.15, f"IV急升{iv_chg_pp:+.1f}pp"))
                else:
                    adjustments.append((+0.10, f"IV急升{iv_chg_pp:+.1f}pp"))
            elif iv_chg_pp > 1.0:
                if direction == "LONG":
                    adjustments.append((-0.08, f"IV升{iv_chg_pp:+.1f}pp"))
            elif iv_chg_pp < -2.0:
                # IV 急降（乐观）
                if direction == "SHORT":
                    adjustments.append((-0.15, f"IV急降{iv_chg_pp:+.1f}pp"))
                else:
                    adjustments.append((+0.10, f"IV降{iv_chg_pp:+.1f}pp"))
            elif iv_chg_pp < -1.0:
                if direction == "SHORT":
                    adjustments.append((-0.08, f"IV降{iv_chg_pp:+.1f}pp"))

    # --- 2. Skew 变化（RR 走强 = 看跌加剧）---
    if sentiment.rr_25d != 0 and sentiment.rr_25d_prev != 0:
        rr_chg_pp = (sentiment.rr_25d - sentiment.rr_25d_prev) * 100
        if abs(rr_chg_pp) > 0.5:
            if rr_chg_pp > 1.0 and direction == "LONG":
                adjustments.append((-0.10, f"Skew看跌加剧{rr_chg_pp:+.1f}pp"))
            elif rr_chg_pp > 0.5 and direction == "LONG":
                adjustments.append((-0.05, f"Skew偏空"))
            elif rr_chg_pp < -1.0 and direction == "SHORT":
                adjustments.append((-0.10, f"Skew看跌减弱{rr_chg_pp:+.1f}pp"))
            elif rr_chg_pp < -0.5 and direction == "SHORT":
                adjustments.append((-0.05, f"Skew偏多"))

    # --- 3. PCR（逆向指标）---
    if sentiment.pcr > 0:
        if sentiment.pcr > 1.5:
            # 极度悲观 → 逆向做多
            if direction == "LONG":
                adjustments.append((+0.15, f"PCR={sentiment.pcr:.1f}逆向"))
            else:
                adjustments.append((-0.10, f"PCR={sentiment.pcr:.1f}悲观"))
        elif sentiment.pcr < 0.7:
            # 极度乐观 → 逆向做空
            if direction == "SHORT":
                adjustments.append((+0.15, f"PCR={sentiment.pcr:.1f}逆向"))
            else:
                adjustments.append((-0.10, f"PCR={sentiment.pcr:.1f}乐观"))

    # --- 4. 期限结构倒挂（轻微惩罚，避免过度压制信号）---
    if "倒挂" in sentiment.term_structure:
        adjustments.append((-0.05, "期限倒挂"))

    if not adjustments:
        return 1.0, ""

    total_adj = sum(a for a, _ in adjustments)
    multiplier = max(0.5, min(1.5, 1.0 + total_adj))
    reason = ",".join(r for _, r in adjustments)
    return round(multiplier, 2), reason


# ---------------------------------------------------------------------------
# v2 常量（保留不变）
# ---------------------------------------------------------------------------

MOM_5M_LOOKBACK = 12
MOM_15M_LOOKBACK = 6
MOM_DAILY_LOOKBACK = 5
ATR_SHORT = 5
ATR_LONG = 40
VOLUME_SURGE_RATIO = 1.5
VOLUME_LOW_RATIO = 0.5
INTRADAY_REVERSAL_THRESHOLD = 0.015

# Q分分位数阈值（历史同时段分位数）
VOLUME_PCT_HIGH = 0.75    # 分位>75% → Q=20（该时段历史性放量）
VOLUME_PCT_LOW = 0.25     # 分位<25% → Q=0（该时段历史性缩量）


def compute_volume_profile(bar_5m_all: pd.DataFrame, before_date: str = "",
                           lookback_days: int = 20) -> Dict[str, list]:
    """
    构建历史同时段volume列表，用于Q分的分位数计算。

    返回 {time_slot: [vol1, vol2, ...]}，time_slot为UTC "HH:MM"格式。
    每个slot保留最近lookback_days个交易日的volume值。

    Args:
        bar_5m_all: 全量5分钟K线（需有datetime和volume列）
        before_date: 只用此日期之前的数据（YYYYMMDD或YYYY-MM-DD）
        lookback_days: 使用最近N个交易日
    """
    df = bar_5m_all.copy()
    if "datetime" in df.columns:
        dt_col = df["datetime"].astype(str)
    elif isinstance(df.index, pd.DatetimeIndex):
        dt_col = df.index.strftime("%Y-%m-%d %H:%M:%S")
    else:
        return {}

    if before_date:
        bd = before_date.replace("-", "")
        date_dash = f"{bd[:4]}-{bd[4:6]}-{bd[6:]}"
        mask = dt_col < date_dash
        df = df[mask].copy()
        dt_col = dt_col[mask]

    if len(df) == 0:
        return {}

    df = df.copy()
    df["_slot"] = dt_col.str[11:16]
    df["_date"] = dt_col.str[:10]
    df["volume"] = df["volume"].astype(float)

    unique_dates = sorted(df["_date"].unique())
    if len(unique_dates) > lookback_days:
        cutoff = unique_dates[-lookback_days]
        df = df[df["_date"] >= cutoff]

    profile: Dict[str, list] = {}
    for slot, grp in df.groupby("_slot"):
        profile[slot] = grp["volume"].tolist()
    return profile


def volume_percentile_q(vol: float, hist_vols: list) -> int:
    """根据历史同时段volume列表计算Q分（分位数法）。"""
    if not hist_vols or len(hist_vols) < 5:
        return 10  # 数据不足，给中性分
    pct = sum(1 for v in hist_vols if v <= vol) / len(hist_vols)
    if pct > VOLUME_PCT_HIGH:
        return 20
    elif pct > VOLUME_PCT_LOW:
        return 10
    return 0

_TIME_WEIGHTS = [
    (time(1, 35), time(2, 30), 1.0),
    (time(2, 30), time(3, 30), 1.1),
    (time(5, 0),  time(5, 30), 1.0),
    (time(5, 30), time(6, 30), 1.0),
    (time(6, 30), time(6, 50), 0.6),
]

NO_TRADE_BEFORE = time(1, 35)
NO_TRADE_AFTER = time(6, 50)


# ---------------------------------------------------------------------------
# 平仓信号系统
# ---------------------------------------------------------------------------

STOP_LOSS_PCT = 0.005           # 默认止损0.5%（IM用0.3%，见SYMBOL_PROFILES）
TRAILING_STOP_HIVOL = 0.008     # 高波动跟踪止盈0.8%
TRAILING_STOP_NORMAL = 0.005    # 正常跟踪止盈0.5%
TRAILING_STOP_LUNCH = 0.003     # 午休前紧止盈0.3%
TIME_STOP_MINUTES = 60
EOD_CLOSE_UTC = "06:45"         # 14:45 BJ
LUNCH_CLOSE_UTC = "03:25"       # 11:25 BJ
NO_OPEN_LUNCH_START = "03:20"   # 11:20 BJ
NO_OPEN_LUNCH_END = "05:05"     # 13:05 BJ
NO_OPEN_EOD = "06:30"           # 14:30 BJ


def _boll_zone(price: float, mid: float, std: float) -> str:
    """Classify price position relative to Bollinger bands."""
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


def _calc_boll(closes: pd.Series, period: int = 20):
    """Return (mid, std) for latest bar, or (nan, nan)."""
    if len(closes) < period:
        return float("nan"), float("nan")
    mid = float(closes.rolling(period).mean().iloc[-1])
    std = float(closes.rolling(period).std().iloc[-1])
    return mid, std


def _score_trend_startup(
    close_5m: np.ndarray, high_5m: np.ndarray, low_5m: np.ndarray,
    volume_5m: np.ndarray, direction: str,
    vol_percentile: float = -1,
) -> tuple[int, str]:
    """趋势启动检测器：突破+振幅放大+放量三重确认时给bonus。

    条件（同时满足）：
    1. 收盘价突破前5根bar的高/低点（方向确认）
    2. 当前bar振幅 > 前20根均振幅 × 1.5（波动率扩张）
    3. 成交量放量确认：分位数>60%���有vol_profile时），或rolling均值×1.2（fallback）
    """
    if len(close_5m) < 22 or not direction:
        return 0, ""
    last_close = float(close_5m[-1])
    recent_high = float(np.max(high_5m[-6:-1]))
    recent_low = float(np.min(low_5m[-6:-1]))
    current_range = float(high_5m[-1] - low_5m[-1])
    avg_range = float(np.mean(
        [high_5m[i] - low_5m[i] for i in range(-21, -1)]))

    if avg_range <= 0:
        return 0, ""

    # 放量确认：优先用分位数，fallback到rolling均值
    if vol_percentile >= 0:
        vol_ok = vol_percentile > 0.6
    else:
        current_vol = float(volume_5m[-1])
        avg_vol = float(np.mean(volume_5m[-21:-1]))
        vol_ok = avg_vol > 0 and current_vol > avg_vol * 1.2

    if (direction == "LONG" and last_close > recent_high
            and current_range > avg_range * 1.5
            and vol_ok):
        return 15, "TS^"
    if (direction == "SHORT" and last_close < recent_low
            and current_range > avg_range * 1.5
            and vol_ok):
        return 15, "TSv"
    return 0, ""


def _score_boll_breakout(
    close_5m: np.ndarray, bar_15m: pd.DataFrame | None,
    direction: str, volume_5m: np.ndarray,
) -> tuple[int, str]:
    """布林带中轨突破加分（0~20分）。

    做空：前3根在中轨上方 → 当根跌破中轨 → 加分
    做多：前3根在中轨下方 → 当根涨破中轨 → 加分
    双周期确认（5m+15m同时突破）→ 额外加分

    Returns: (bonus_score, note_str)
    """
    if len(close_5m) < 24:  # 需要20根算BOLL + 几根确认
        return 0, ""

    closes_s = pd.Series(close_5m)
    mid, std = _calc_boll(closes_s, 20)
    if np.isnan(mid) or std <= 0:
        return 0, ""

    cur = float(close_5m[-1])
    bonus = 0
    note = ""

    if direction == "SHORT":
        # 检查：前5根在中轨上方，当根跌破（避免来回穿越假突破）
        prev_above = all(float(close_5m[i]) > mid for i in range(-6, -1))
        cur_below = cur < mid
        if prev_above and cur_below:
            bonus = 10
            note = "B5v"
            # 放量确认（比前3根bar均量高即可）
            if len(volume_5m) >= 4:
                avg_vol = float(np.mean(volume_5m[-4:-1]))
                if avg_vol > 0 and float(volume_5m[-1]) > avg_vol:
                    bonus += 2
            # 窄带突破更有价值（带宽 < 均值的0.8）
            boll_width = 2 * std / mid if mid > 0 else 0
            if len(close_5m) >= 40:
                hist_stds = pd.Series(close_5m[-40:]).rolling(20).std().dropna()
                if len(hist_stds) >= 5:
                    avg_bw = float(hist_stds.mean()) * 2 / mid
                    if boll_width < avg_bw * 0.8:
                        bonus += 3

    elif direction == "LONG":
        prev_below = all(float(close_5m[i]) < mid for i in range(-6, -1))
        cur_above = cur > mid
        if prev_below and cur_above:
            bonus = 10
            note = "B5^"
            # 放量确认（比前3根bar均量高即可）
            if len(volume_5m) >= 4:
                avg_vol = float(np.mean(volume_5m[-4:-1]))
                if avg_vol > 0 and float(volume_5m[-1]) > avg_vol:
                    bonus += 2
            boll_width = 2 * std / mid if mid > 0 else 0
            if len(close_5m) >= 40:
                hist_stds = pd.Series(close_5m[-40:]).rolling(20).std().dropna()
                if len(hist_stds) >= 5:
                    avg_bw = float(hist_stds.mean()) * 2 / mid
                    if boll_width < avg_bw * 0.8:
                        bonus += 3

    if bonus == 0:
        return 0, ""

    # 15分钟同方向确认 → 额外加分
    if bar_15m is not None and len(bar_15m) >= 24:
        c15 = bar_15m["close"].values.astype(float)
        c15_s = pd.Series(c15)
        mid15, std15 = _calc_boll(c15_s, 20)
        if not np.isnan(mid15) and std15 > 0:
            if direction == "SHORT":
                prev15_above = all(float(c15[i]) > mid15 for i in range(-3, -1))
                cur15_below = float(c15[-1]) < mid15
                if prev15_above and cur15_below:
                    bonus += 5
                    note = "B5v+15v"
                elif float(c15[-1]) < mid15:
                    # 15分钟已在中轨下方（趋势确认）
                    bonus += 3
                    note = "B5v+15ok"
            elif direction == "LONG":
                prev15_below = all(float(c15[i]) < mid15 for i in range(-3, -1))
                cur15_above = float(c15[-1]) > mid15
                if prev15_below and cur15_above:
                    bonus += 5
                    note = "B5^+15^"
                elif float(c15[-1]) > mid15:
                    bonus += 3
                    note = "B5^+15ok"

    return min(bonus, 20), note


def is_open_allowed(utc_hm: str) -> bool:
    """Check if opening new positions is allowed at this time."""
    if utc_hm < "01:45":      # before 09:45 BJ (avoid opening noise)
        return False
    if NO_OPEN_LUNCH_START <= utc_hm < NO_OPEN_LUNCH_END:  # 11:20-13:05 BJ
        return False
    if utc_hm >= NO_OPEN_EOD:  # after 14:15 BJ
        return False
    return True


# ── 开盘振幅过滤器 ─────────────────────────────────────────
# 215天验证：开盘30min振幅<0.4%的日子策略均PnL=-21pt/天
# 过滤后+274pt改善，误杀率仅15%
OPEN_RANGE_FILTER_PCT = 0.004   # 0.4%
OPEN_RANGE_BARS = 6             # 前6根5分钟bar = 30分钟


def check_low_amplitude(bar_5m: pd.DataFrame) -> bool:
    """判断当天是否为低振幅日（开盘30分钟振幅<0.4%）。

    Args:
        bar_5m: 当天的5分钟K线（至少需要前6根bar）

    Returns:
        True if low amplitude (should suppress new entries), False otherwise
    """
    if bar_5m is None or len(bar_5m) < OPEN_RANGE_BARS:
        return False
    first_bars = bar_5m.iloc[:OPEN_RANGE_BARS]
    open_price = float(first_bars.iloc[0]["open"])
    if open_price <= 0:
        return False
    high_max = float(first_bars["high"].astype(float).max())
    low_min = float(first_bars["low"].astype(float).min())
    amplitude = (high_max - low_min) / open_price
    return amplitude < OPEN_RANGE_FILTER_PCT


def check_exit(
    position: dict,
    current_price: float,
    bar_5m: pd.DataFrame,
    bar_15m: pd.DataFrame | None,
    current_time_utc: str,
    reverse_signal_score: int = 0,
    is_high_vol: bool = True,
    symbol: str = "",
    spot_price: float = 0.0,
) -> dict:
    """
    Multi-timeframe Bollinger exit system.

    Priority: EOD > StopLoss > Lunch > TrailingStop > TrendComplete >
              MomentumExhausted > MidBreak > TimeStop

    Price convention (live monitor):
      - current_price / highest / lowest / entry_price: 期货价格（止损/跟踪止盈/PnL）
      - spot_price: 现货价格（Bollinger zone判断，与bar_5m/bar_15m同源）
      - 回测时 spot_price=0 → fallback 到 current_price（回测全用现货）
    """
    entry_price = position["entry_price"]
    direction = position["direction"]
    entry_time = position.get("entry_time_utc", "")
    highest = position.get("highest_since", entry_price)
    lowest = position.get("lowest_since", entry_price)
    volume = position.get("volume", 1)
    bars_below_mid = position.get("bars_below_mid", 0)

    # Bollinger zone判断用现货价格（与bar_5m同源），回测时fallback到current_price
    boll_price = spot_price if spot_price > 0 else current_price

    NO_EXIT = {"should_exit": False, "exit_volume": 0,
               "exit_reason": "", "exit_urgency": "NORMAL"}

    # P1: EOD close
    if current_time_utc >= EOD_CLOSE_UTC:
        return {"should_exit": True, "exit_volume": volume,
                "exit_reason": "EOD_CLOSE", "exit_urgency": "URGENT"}

    # P1b: Stop loss (MUST be before lunch close — prevents large losses at lunch)
    # Per-symbol stop loss: IM=0.3%（217天稳健性三检通过+366pt），其他0.5%
    _prof = SYMBOL_PROFILES.get(symbol, _DEFAULT_PROFILE) if symbol else _DEFAULT_PROFILE
    _sl_pct = _prof.get("stop_loss_pct", STOP_LOSS_PCT)
    if direction == "LONG":
        loss_pct = (entry_price - current_price) / entry_price
    else:
        loss_pct = (current_price - entry_price) / entry_price
    if loss_pct > _sl_pct:
        return {"should_exit": True, "exit_volume": volume,
                "exit_reason": "STOP_LOSS", "exit_urgency": "URGENT"}

    # P2: Lunch close (11:25 BJ)
    if current_time_utc >= LUNCH_CLOSE_UTC and current_time_utc < "05:00":
        if direction == "LONG":
            profitable = current_price > entry_price
        else:
            profitable = current_price < entry_price
        if not profitable:
            return {"should_exit": True, "exit_volume": volume,
                    "exit_reason": "LUNCH_CLOSE", "exit_urgency": "URGENT"}
        # Profitable: tighten trailing stop to 0.3%
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

    # Compute Bollinger bands
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

    # Hold time (used by P3 trailing stop and P5 momentum exhausted)
    hold_minutes = 0
    if entry_time and current_time_utc:
        try:
            h1, m1 = int(entry_time[:2]), int(entry_time[3:5])
            h2, m2 = int(current_time_utc[:2]), int(current_time_utc[3:5])
            hold_minutes = (h2 * 60 + m2) - (h1 * 60 + m1)
        except Exception:
            pass

    # P3: Dynamic trailing stop (per-symbol: IC禁用，让利润奔跑)
    _prof = SYMBOL_PROFILES.get(symbol, _DEFAULT_PROFILE) if symbol else _DEFAULT_PROFILE
    if _prof.get("trailing_stop_enabled", True):
        if hold_minutes < 15:
            trail_pct = 0.005   # 0.5% — just opened, protect capital
        elif hold_minutes < 30:
            trail_pct = 0.006   # 0.6%
        elif hold_minutes < 60:
            trail_pct = 0.008   # 0.8%
        else:
            trail_pct = 0.010   # 1.0% — long trend, max room

        # Bonus: if profitable >0.5% and 15m trend confirmed, widen by 0.2%
        if zone_15m:
            if direction == "LONG":
                pnl_pct = (current_price - entry_price) / entry_price
                fifteen_ok = zone_15m in ("MID_UPPER", "UPPER_ZONE", "ABOVE_UPPER")
            else:
                pnl_pct = (entry_price - current_price) / entry_price
                fifteen_ok = zone_15m in ("MID_LOWER", "LOWER_ZONE", "BELOW_LOWER")
            if pnl_pct > 0.005 and fifteen_ok:
                trail_pct += 0.002  # +0.2% bonus

        # Per-symbol trailing scale（IC=2.0x需要更宽trailing）
        trail_pct *= _prof.get("trailing_stop_scale", 1.0)

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

    # P3b: Band Reversal — 研究结果(2026-04-07)
    # Phase 1: 因子有效（bounce_from_lower WR=69% +14bps, 1911样本）
    # Phase 2: 但作为exit信号净效果 -32%（退出后再入场被止损 + 趋势日错过利润）
    # 结论: 单独作为exit不可行。需要搭配：
    #   a. 退出后同方向冷却期（防止反转行情中再入场止损）
    #   b. 或只在低振幅日/震荡regime启用
    #   c. 或作为反手信号而非纯退出
    # 暂不启用，留待进一步研究

    # P4: Trend complete — requires 5m ABOVE upper + 15m ABOVE upper
    # Both timeframes at the most extreme zone = trend truly exhausted
    # UPPER_ZONE alone is not enough (trend still pushing)
    if zone_5m and zone_15m:
        if direction == "LONG":
            if zone_5m == "ABOVE_UPPER" and zone_15m == "ABOVE_UPPER":
                return {"should_exit": True, "exit_volume": volume,
                        "exit_reason": "TREND_COMPLETE", "exit_urgency": "NORMAL"}
        else:
            if zone_5m == "BELOW_LOWER" and zone_15m == "BELOW_LOWER":
                return {"should_exit": True, "exit_volume": volume,
                        "exit_reason": "TREND_COMPLETE", "exit_urgency": "NORMAL"}

    # P5: Momentum exhausted (3 bars narrow range + 15m extreme + NOT trending)
    # 最小持仓20分钟：5-15min止出75%是过早的（32天回测验证）
    if hold_minutes >= 20 and bar_5m is not None and len(bar_5m) >= 23 and b5_std > 0:
        last3_c = bar_5m["close"].astype(float).iloc[-3:]
        last3_h = bar_5m["high"].astype(float).iloc[-3:]
        last3_l = bar_5m["low"].astype(float).iloc[-3:]
        total_range = float(last3_h.max() - last3_l.min())
        boll_width = 4 * b5_std
        if total_range < boll_width * 0.10:
            # Narrow range — but is price still trending?
            close_change = float(last3_c.iloc[-1]) - float(last3_c.iloc[0])
            still_trending = False
            if direction == "LONG" and close_change > boll_width * 0.05:
                still_trending = True  # slow grind up, not stalled
            elif direction == "SHORT" and close_change < -boll_width * 0.05:
                still_trending = True  # slow grind down

            if not still_trending:
                if direction == "LONG" and zone_15m in ("ABOVE_UPPER", "UPPER_ZONE"):
                    return {"should_exit": True, "exit_volume": volume,
                            "exit_reason": "MOMENTUM_EXHAUSTED", "exit_urgency": "NORMAL"}
                elif direction == "SHORT" and zone_15m in ("BELOW_LOWER", "LOWER_ZONE"):
                    return {"should_exit": True, "exit_volume": volume,
                            "exit_reason": "MOMENTUM_EXHAUSTED", "exit_urgency": "NORMAL"}

    # P6: Mid-band break (requires BOTH timeframes)
    # 5m breaks mid + 15m also below mid → trend over
    # 5m breaks mid but 15m still above → just a pullback, hold
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
            # Need 2 bars + 15m confirmation
            if bars_below_mid >= 3 and fifteen_below:
                return {"should_exit": True, "exit_volume": volume,
                        "exit_reason": "MID_BREAK", "exit_urgency": "NORMAL"}
        else:
            position["bars_below_mid"] = 0

    # P7: Time stop (60min without profit)
    if entry_time and current_time_utc:
        try:
            h1, m1 = int(entry_time[:2]), int(entry_time[3:5])
            h2, m2 = int(current_time_utc[:2]), int(current_time_utc[3:5])
            elapsed = (h2 * 60 + m2) - (h1 * 60 + m1)
            if elapsed > TIME_STOP_MINUTES:
                profitable = (current_price > entry_price) if direction == "LONG" \
                    else (current_price < entry_price)
                if not profitable:
                    return {"should_exit": True, "exit_volume": volume,
                            "exit_reason": "TIME_STOP", "exit_urgency": "NORMAL"}
        except Exception:
            pass

    return NO_EXIT


# ---------------------------------------------------------------------------
# 均值回归逻辑链
# ---------------------------------------------------------------------------
# 第一层: is_high_vol (GARCH ratio > 1.2)
# 第二层: Z-Score硬过滤 (只在高波动区间生效)
# 第三层: 日内涨跌幅加强 (只在高波动区间加强)
# 第四层: RSI回归确认 (高波动+超卖/超买时RSI反转加分)
# ---------------------------------------------------------------------------

DEFAULT_GARCH_LONG_RUN = 24.9  # 长期GARCH均值(%)，从volatility_history


def _calc_rsi(closes: np.ndarray, period: int = 14) -> float:
    """计算RSI（0-100）。"""
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes)
    recent = deltas[-(period):]
    gains = np.where(recent > 0, recent, 0)
    losses = np.where(recent < 0, -recent, 0)
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


def _rsi_reversal_bonus(
    close_5m: np.ndarray, zscore: float, direction: str,
) -> tuple[int, str]:
    """
    第四层：RSI回归确认。

    Z<-2 + RSI从低位回升 → 做多+10分
    Z>+2 + RSI从高位回落 → 做空+10分
    """
    if len(close_5m) < 20:
        return 0, ""

    rsi_now = _calc_rsi(close_5m, 14)
    rsi_prev = _calc_rsi(close_5m[:-3], 14)  # 3根K线前的RSI

    if zscore < -2.0:
        if direction == "LONG" and rsi_now > rsi_prev and rsi_now < 40:
            return 10, f"RSI{rsi_now:.0f}^"
    elif zscore > 2.0:
        if direction == "SHORT" and rsi_now < rsi_prev and rsi_now > 60:
            return 10, f"RSI{rsi_now:.0f}v"

    return 0, ""


def _apply_zscore_filter(
    total: int, direction: str, zscore: float | None,
    is_high_vol: bool = True,
) -> tuple[int, str]:
    """
    第二层：Z-Score硬过滤。只在高波动区间(is_high_vol)生效。

    均值回归逻辑：超卖时阻断做空、鼓励做多，超买时反之。

    高波动区间:
      Z < -2.5: 做空→0，做多×1.3
      Z < -2.0: 做空×0.3，做多×1.2
      Z > +2.5: 做多→0，做空×1.3
      Z > +2.0: 做多×0.3，做空×1.2

    注：曾测试"顺势不阻断"（Z<-2做空不惩罚），slip=0时+776pt优于+735pt，
    但breakeven从4.0pt恶化到3.0pt（多释放的交易摊薄滑点），revert。
    """
    if zscore is None or not direction:
        return total, ""

    if not is_high_vol:
        return total, ""  # 低波动区间不过滤

    # 梯度Z-Score过滤（高波动区间）
    if direction == "SHORT":
        if zscore < -3.0:
            return 0, "Z<-3 BLOCK"
        elif zscore < -2.5:
            return 0, "Z<-2.5 BLOCK"
        elif zscore < -2.0:
            return int(total * 0.3), "Z<-2 x0.3"
        elif zscore < -1.5:
            return int(total * 0.6), "Z<-1.5 x0.6"
    elif direction == "LONG":
        if zscore < -3.0:
            return min(100, int(total * 1.5)), "Z<-3 +50%"
        elif zscore < -2.5:
            return min(100, int(total * 1.3)), "Z<-2.5 +30%"
        elif zscore < -2.0:
            return min(100, int(total * 1.2)), "Z<-2 +20%"
        elif zscore < -1.5:
            return min(100, int(total * 1.1)), "Z<-1.5 +10%"

    # 对称的超买处理
    if direction == "LONG":
        if zscore > 3.0:
            return 0, "Z>3 BLOCK"
        elif zscore > 2.5:
            return 0, "Z>2.5 BLOCK"
        elif zscore > 2.0:
            return int(total * 0.3), "Z>2 x0.3"
        elif zscore > 1.5:
            return int(total * 0.6), "Z>1.5 x0.6"
    elif direction == "SHORT":
        if zscore > 3.0:
            return min(100, int(total * 1.5)), "Z>3 +50%"
        elif zscore > 2.5:
            return min(100, int(total * 1.3)), "Z>2.5 +30%"
        elif zscore > 2.0:
            return min(100, int(total * 1.2)), "Z>2 +20%"
        elif zscore > 1.5:
            return min(100, int(total * 1.1)), "Z>1.5 +10%"

    return total, ""


# ---------------------------------------------------------------------------
# v3 品种配置表
# ---------------------------------------------------------------------------

# 时段倍数 key = (utc_start, utc_end)
_SESSION_UTC = {
    "0935-1030": (time(1, 35), time(2, 30)),
    "1030-1130": (time(2, 30), time(3, 30)),
    "1300-1330": (time(5, 0),  time(5, 30)),
    "1330-1430": (time(5, 30), time(6, 30)),
    "1430-1450": (time(6, 30), time(6, 50)),
}

# ---------------------------------------------------------------------------
# 品种 → 信号版本路由（每个品种用回测表现最佳的版本）
# ---------------------------------------------------------------------------
SIGNAL_ROUTING: Dict[str, str] = {
    "IF": "v2",   # 均值回归型，v2逆势+120pt > v3无逆势+102pt
    "IH": "v2",   # 均值回归型，v2逆势+99pt > v3无逆势+60pt（干净数据验证）
    "IM": "v2",   # 动量型，波动大利润厚
    "IC": "v2",   # 动量型，thr=65优化
}

SYMBOL_PROFILES: Dict[str, Dict] = {
    "IM": {
        "style": "MOMENTUM",
        "momentum_lookback_5m": 12,   # 60分钟（v2验证最优，保持不变）
        "momentum_lookback_15m": 6,   # 90分钟
        "momentum_lookback_daily": 5, # 5天（与v2一致，回测验证更稳定）
        "reversal_filter": False,     # IM不做反转过滤（实证：超跌不反弹52.9%，超涨不回调62%）
        "reversal_threshold": 0.0,
        "trend_weight": 60,
        "vol_weight": 25,
        "volume_weight": 15,
        "reversal_weight": 0,
        "daily_align_bonus": 1.2,     # 保守与v2一致
        "daily_conflict_penalty": 0.7, # 与v2一致
        "dm_trend": 1.1,              # 215天验证：1.1/0.9 > 1.2/0.8（+691pt合计）
        "dm_contrarian": 0.9,         # 轻度逆势惩罚，避免过度打折逆势交易
        "trailing_stop_scale": 1.5,   # 215天验证：1.5x > 1.0x（IM+259pt）
        "stop_loss_pct": 0.003,       # 217天稳健性三检通过：+366pt，邻域✅分半✅单日18%✅
        "session_multiplier": {
            "0935-1030": 1.0,
            "1030-1130": 1.1,
            "1300-1330": 1.0,
            "1330-1430": 1.0,
            "1430-1450": 0.7,
        },
    },
    "IF": {
        "style": "HYBRID",
        "momentum_lookback_5m": 12,   # 60分钟（grid search验证12最优，之前配18但硬编码没生效）
        "momentum_lookback_15m": 6,   # 90分钟
        "momentum_lookback_daily": 20, # 20天实证最优
        "reversal_filter": True,
        "reversal_threshold": 0.02,   # 跌>2%后反弹75%
        "trend_weight": 50,
        "vol_weight": 30,
        "volume_weight": 20,
        "reversal_weight": 0,
        "daily_align_bonus": 1.2,
        "daily_conflict_penalty": 0.7,
        "dm_trend": 1.0,             # IF均值回归型：中性dm（逆势59%WR，不惩罚）
        "dm_contrarian": 1.0,        # 34天验证：中性+255pt vs 当前+146pt（+75%）
        "session_multiplier": {
            "0935-1030": 1.2,         # 开盘动量最强
            "1030-1130": 1.1,
            "1300-1330": 0.7,
            "1330-1430": 0.9,
            "1430-1450": 0.5,         # 基本不做
        },
    },
    "IH": {
        "style": "MEAN_REVERSION",
        "momentum_lookback_5m": 12,   # 60分钟（grid search验证12最优）
        "momentum_lookback_15m": 6,
        "momentum_lookback_daily": 20,
        "reversal_filter": True,
        "reversal_threshold": 0.01,   # 涨>1%就开始回调58%
        "trend_weight": 35,
        "vol_weight": 30,
        "volume_weight": 15,
        "reversal_weight": 20,        # 反转维度
        "daily_align_bonus": 1.2,
        "daily_conflict_penalty": 0.7,
        "dm_trend": 1.1,              # 215天验证：1.1/0.9统一
        "dm_contrarian": 0.9,
        "session_multiplier": {
            "0935-1030": 1.0,
            "1030-1130": 1.0,
            "1300-1330": 1.0,
            "1330-1430": 1.0,
            "1430-1450": 0.3,         # 尾盘反转率43%最高，不做趋势
        },
    },
    "IC": {
        "style": "MOMENTUM",
        "momentum_lookback_5m": 12,
        "momentum_lookback_15m": 6,
        "momentum_lookback_daily": 5,
        "reversal_filter": False,     # 动量品种，类似IM
        "reversal_threshold": 0.0,
        "trend_weight": 60,
        "vol_weight": 25,
        "volume_weight": 15,
        "reversal_weight": 0,
        "daily_align_bonus": 1.2,
        "daily_conflict_penalty": 0.7,
        "dm_trend": 1.1,              # 215天验证：1.1/0.9 > 1.2/0.8
        "dm_contrarian": 0.9,
        "signal_threshold": 60,       # 动态lb+th55/60: IC从65降至60（配合lb=4震荡模式）
        "trailing_stop_scale": 2.0,   # IC趋势中震荡大，需要更宽trailing（5/5周稳健验证）
        "session_multiplier": {
            "0935-1030": 1.0,
            "1030-1130": 1.1,
            "1300-1330": 1.0,
            "1330-1430": 1.0,
            "1430-1450": 0.7,
        },
    },
}

# 默认配置（未知品种 fallback 到 v2 式统一参数）
_DEFAULT_PROFILE: Dict = {
    "style": "HYBRID",
    "momentum_lookback_5m": 12,
    "momentum_lookback_15m": 6,
    "momentum_lookback_daily": 5,
    "reversal_filter": True,
    "reversal_threshold": 0.015,
    "trend_weight": 50,
    "vol_weight": 30,
    "volume_weight": 20,
    "reversal_weight": 0,
    "daily_align_bonus": 1.2,
    "daily_conflict_penalty": 0.7,
    "session_multiplier": {
        "0935-1030": 1.0, "1030-1130": 1.1, "1300-1330": 1.0,
        "1330-1430": 1.0, "1430-1450": 0.6,
    },
}


# ---------------------------------------------------------------------------
# 共用工具函数
# ---------------------------------------------------------------------------

def _get_time_weight(utc_time: time) -> float:
    for start, end, weight in _TIME_WEIGHTS:
        if start <= utc_time < end:
            return weight
    return 1.0


def _find_session_start(bar_5m: pd.DataFrame) -> int:
    """Find the array index of the last session's first bar.

    Returns the positional index (0-based) of today's first bar within bar_5m.
    If no session boundary is found, returns 0 (entire array is same session).
    """
    n = len(bar_5m)
    if n < 2:
        return 0
    # Extract datetime values as numpy array for fast access
    if isinstance(bar_5m.index, pd.DatetimeIndex):
        dts = bar_5m.index.values
    elif "datetime" in bar_5m.columns:
        dts = pd.to_datetime(bar_5m["datetime"]).values
    else:
        return 0
    # Scan backwards for a gap > 30 minutes (= session boundary)
    for i in range(n - 1, 0, -1):
        gap_ns = int(dts[i]) - int(dts[i - 1])
        gap_sec = gap_ns / 1e9
        if gap_sec > 1800:  # 30 minutes
            return i
    return 0


def _atr(highs, lows, closes, period: int) -> float:
    if len(highs) < period + 1:
        return 0.0
    tr_list = []
    for i in range(-period, 0):
        h, l, c_prev = highs[i], lows[i], closes[i - 1]
        tr = max(h - l, abs(h - c_prev), abs(l - c_prev))
        tr_list.append(tr)
    return float(np.mean(tr_list)) if tr_list else 0.0


def _extract_today_bars(bar_5m: pd.DataFrame) -> pd.DataFrame:
    if bar_5m.empty:
        return bar_5m
    if isinstance(bar_5m.index, pd.DatetimeIndex):
        last_date = bar_5m.index[-1].date()
        return bar_5m[bar_5m.index.date == last_date]
    elif "datetime" in bar_5m.columns:
        dt_col = pd.to_datetime(bar_5m["datetime"])
        last_date = dt_col.iloc[-1].date()
        return bar_5m[dt_col.dt.date == last_date]
    return bar_5m


def _get_utc_time(bar_5m: pd.DataFrame) -> Optional[time]:
    try:
        if isinstance(bar_5m.index, pd.DatetimeIndex):
            return bar_5m.index[-1].time()
        elif "datetime" in bar_5m.columns:
            val = bar_5m.iloc[-1]["datetime"]
            if isinstance(val, (int, float, np.integer)):
                ts = pd.Timestamp(int(val), unit="ns")
                return ts.time()
            return pd.Timestamp(val).time()
    except Exception:
        pass
    return None


def _get_session_weight(utc_time: time, session_map: Dict[str, float]) -> float:
    for label, (start, end) in _SESSION_UTC.items():
        if start <= utc_time < end:
            return session_map.get(label, 1.0)
    return 1.0


# ---------------------------------------------------------------------------
# v2 信号生成器（保留不变）
# ---------------------------------------------------------------------------

class SignalGeneratorV2:
    """简化三维度信号评分系统（A股特化参数）。"""

    def __init__(self, config: Dict | None = None):
        cfg = config or {}
        self.min_signal_score: int = cfg.get("min_signal_score", 55)
        self.debug: bool = cfg.get("debug", False)
        self._opening_bars: int = cfg.get("opening_bars", 6)

    def update(
        self, symbol: str, bar_5m: pd.DataFrame,
        bar_15m: pd.DataFrame | None, daily_bar: pd.DataFrame | None,
        quote_data: Dict | None = None,
        sentiment: Optional[SentimentData] = None,
        zscore: float | None = None,
        is_high_vol: bool = True,
        d_override: Dict[str, float] | None = None,
    ) -> Optional[IntradaySignal]:
        result = self.score_all(
            symbol, bar_5m, bar_15m, daily_bar, quote_data, sentiment,
            zscore=zscore, is_high_vol=is_high_vol, d_override=d_override,
        )
        if result is None:
            return None
        if result["total"] < self.min_signal_score or not result["direction"]:
            return None
        # 开仓时间窗口检查（在 update 中拦截，不在 score_all 中拦截，
        # 以便面板始终能显示评分）
        utc_time = _get_utc_time(bar_5m)
        if utc_time and not is_open_allowed(utc_time.strftime("%H:%M")):
            return None
        close = float(bar_5m.iloc[-1]["close"])
        atr_val = result.get("atr_short", close * 0.005)
        if atr_val <= 0:
            atr_val = close * 0.005
        if result["direction"] == "LONG":
            stop = close - min(atr_val * 2, close * 0.01)
        else:
            stop = close + min(atr_val * 2, close * 0.01)
        sent_str = ""
        if result.get("sentiment_mult", 1.0) != 1.0:
            sent_str = f"|sent={result['sentiment_mult']:.2f}"
        return IntradaySignal(
            symbol=symbol,
            datetime=str(bar_5m.index[-1]) if isinstance(
                bar_5m.index, pd.DatetimeIndex) else str(
                bar_5m.iloc[-1].get("datetime", "")),
            direction=result["direction"], score=result["total"],
            entry_price=close, stop_loss=round(stop, 1),
            signal_type="INTRADAY",
            components={
                "momentum": {"score": result["s_momentum"]},
                "volatility": {"score": result["s_volatility"]},
                "volume": {"score": result["s_volume"]},
            },
            reason=(f"v2|mom={result['s_momentum']}"
                    f"|vol={result['s_volatility']}"
                    f"|qty={result['s_volume']}"
                    f"|tw={result.get('time_weight', 1.0):.1f}"
                    f"|id={result.get('intraday_filter', 1.0):.1f}"
                    f"{sent_str}"),
        )

    def score_all(
        self, symbol: str, bar_5m: pd.DataFrame,
        bar_15m: pd.DataFrame | None, daily_bar: pd.DataFrame | None,
        quote_data: Dict | None = None,
        sentiment: Optional[SentimentData] = None,
        zscore: float | None = None,
        is_high_vol: bool = True,
        d_override: Dict[str, float] | None = None,
        vol_profile: Dict[str, list] | None = None,
    ) -> Dict | None:
        prof = SYMBOL_PROFILES.get(symbol, _DEFAULT_PROFILE)
        lb_5m = prof.get("momentum_lookback_5m", MOM_5M_LOOKBACK)
        if bar_5m is None or len(bar_5m) < lb_5m + 1:
            return None
        utc_time = _get_utc_time(bar_5m)
        if utc_time and (utc_time < NO_TRADE_BEFORE or utc_time > NO_TRADE_AFTER):
            return None

        close_5m = bar_5m["close"].values
        high_5m = bar_5m["high"].values
        low_5m = bar_5m["low"].values
        volume_5m = bar_5m["volume"].values
        current_close = float(close_5m[-1])

        s_mom, mom_dir = self._score_momentum(close_5m, bar_15m, daily_bar, lb_5m)
        s_vol, atr_short = self._score_volatility(high_5m, low_5m, close_5m)

        # Q分：优先用历史同时段分位数（消除跨日volume偏差），fallback到rolling均值
        if vol_profile and utc_time:
            slot = utc_time.strftime("%H:%M")
            hist_vols = vol_profile.get(slot)
            if hist_vols and len(hist_vols) >= 5:
                s_qty = volume_percentile_q(float(volume_5m[-1]), hist_vols)
            else:
                s_qty = self._score_volume(volume_5m, mom_dir)
        else:
            s_qty = self._score_volume(volume_5m, mom_dir)

        # 布林带突破加分（0~20分，仅动量已确认方向时生效）
        s_breakout, breakout_note = 0, ""
        if s_mom > 0 and mom_dir:
            s_breakout, breakout_note = _score_boll_breakout(
                close_5m, bar_15m, mom_dir, volume_5m)

        # 趋势启动检测器（0~15分，突破+振幅+放量三重确认）
        s_startup, startup_note = 0, ""
        if mom_dir:
            # 传入volume分位数（如有），消��跨日偏差
            _vol_pct = -1.0
            if vol_profile and utc_time:
                slot = utc_time.strftime("%H:%M")
                _hv = vol_profile.get(slot)
                if _hv and len(_hv) >= 5:
                    _cur_v = float(volume_5m[-1])
                    _vol_pct = sum(1 for v in _hv if v <= _cur_v) / len(_hv)
            s_startup, startup_note = _score_trend_startup(
                close_5m, high_5m, low_5m, volume_5m, mom_dir,
                vol_percentile=_vol_pct)
            if startup_note:
                breakout_note = (breakout_note + "+" + startup_note
                                 if breakout_note else startup_note)

        daily_mult = self._daily_direction_multiplier(daily_bar, mom_dir, symbol)
        # Morning Briefing d_override: 覆盖daily_mult（和monitor一致）
        if d_override and mom_dir:
            daily_mult = d_override.get(mom_dir, daily_mult)
        raw_total = s_mom + s_vol + s_qty + s_breakout + s_startup
        adjusted = raw_total * daily_mult

        # 第三层：日内涨跌幅过滤（用当前价 vs 昨日收盘，含跳空gap）
        intraday_filter = 1.0
        prev_close = 0.0
        if daily_bar is not None and len(daily_bar) >= 2:
            # 从K线提取当前交易日，找严格小于当天的最新日线
            today_bars = _extract_today_bars(bar_5m)
            today_date = ""
            if len(today_bars) > 0:
                try:
                    if isinstance(today_bars.index, pd.DatetimeIndex):
                        today_date = today_bars.index[0].strftime("%Y%m%d")
                    elif "datetime" in today_bars.columns:
                        today_date = str(today_bars.iloc[0]["datetime"])[:10].replace("-", "")
                except Exception:
                    pass
            if today_date and "trade_date" in daily_bar.columns:
                prev_rows = daily_bar[daily_bar["trade_date"] < today_date]
                if len(prev_rows) > 0:
                    prev_close = float(prev_rows.iloc[-1]["close"])
            if prev_close <= 0:
                prev_close = float(daily_bar.iloc[-2]["close"])  # fallback
        if prev_close <= 0:
            # fallback: 今日开盘价
            today_bars = _extract_today_bars(bar_5m)
            if len(today_bars) > 0:
                prev_close = float(today_bars.iloc[0]["open"])

        if prev_close > 0:
            intraday_return = (current_close - prev_close) / prev_close
            if is_high_vol:
                intraday_filter = self._intraday_filter(intraday_return, mom_dir, zscore)
            else:
                intraday_filter = self._intraday_filter_mild(intraday_return, mom_dir)
            adjusted *= intraday_filter

        # 时段权重：用per-symbol session_multiplier（prof已在函数开头加载）
        tw = _get_session_weight(utc_time, prof["session_multiplier"]) if utc_time else 1.0
        adjusted *= tw

        # 期权情绪乘数
        sent_mult, sent_reason = calc_sentiment_multiplier(mom_dir, sentiment)
        adjusted *= sent_mult

        total = max(0, min(100, int(round(adjusted))))

        # 第二层：Z-Score硬过滤（只在高波动区间生效）
        pre_z_total = total
        total, z_filter = _apply_zscore_filter(total, mom_dir, zscore, is_high_vol)

        # 第四层：RSI回归确认（高波动+极端Z时）
        rsi_bonus = 0
        rsi_note = ""
        if is_high_vol and zscore is not None and abs(zscore) > 2.0:
            rsi_bonus, rsi_note = _rsi_reversal_bonus(close_5m, zscore, mom_dir)
            total = min(100, total + rsi_bonus)

        # RSI当前值（面板显示用）
        rsi_val = _calc_rsi(close_5m) if len(close_5m) >= 15 else 50.0

        return {
            "total": total, "direction": mom_dir,
            "s_momentum": s_mom, "s_volatility": s_vol, "s_volume": s_qty,
            "s_breakout": s_breakout, "breakout_note": breakout_note,
            "daily_mult": daily_mult, "intraday_filter": intraday_filter,
            "time_weight": tw, "raw_total": raw_total, "atr_short": atr_short,
            "sentiment_mult": sent_mult, "sentiment_reason": sent_reason,
            "pre_z_total": pre_z_total, "z_filter": z_filter,
            "rsi": rsi_val, "rsi_bonus": rsi_bonus, "rsi_note": rsi_note,
            "is_high_vol": is_high_vol,
        }

    # 动态 lookback 参数（振幅驱动）
    # 稳健性验证: 时间分半两半都正, 12个邻域全正(+425~+1062), 逐月无亏损
    _DYN_LB_LOW = 4      # 震荡模式 lookback（20分钟窗口）
    _DYN_LB_HIGH = 12    # 趋势模式 lookback（60分钟窗口，=旧 baseline）
    _DYN_AMP_THR = 0.015  # 振幅切换阈值 1.5%

    def _score_momentum(self, close_5m, bar_15m, daily_bar,
                         lookback_5m: int = MOM_5M_LOOKBACK) -> Tuple[int, str]:
        # === 动态 lookback: 振幅 < 1.5% 用 lb=4（震荡），否则 lb=12（趋势） ===
        # 216天回测: +5186 vs baseline +4124 (+26%), 时间分半前+21%后+30%
        lb = self._DYN_LB_HIGH  # 默认趋势模式
        if len(close_5m) >= 6:
            recent = close_5m[-min(48, len(close_5m)):]
            amp = (max(recent) - min(recent)) / recent[0] if recent[0] > 0 else 0
            if amp < self._DYN_AMP_THR:
                lb = self._DYN_LB_LOW

        if len(close_5m) < lb + 1:
            return 0, ""
        mom_5m = (close_5m[-1] - close_5m[-lb - 1]) / close_5m[-lb - 1]
        dir_5m = "LONG" if mom_5m > 0 else "SHORT" if mom_5m < 0 else ""
        mom_15m = 0.0
        dir_15m = ""
        if bar_15m is not None and len(bar_15m) >= MOM_15M_LOOKBACK + 1:
            c15 = bar_15m["close"].values
            mom_15m = (c15[-1] - c15[-MOM_15M_LOOKBACK - 1]) / c15[-MOM_15M_LOOKBACK - 1]
            dir_15m = "LONG" if mom_15m > 0 else "SHORT" if mom_15m < 0 else ""
        if dir_15m and dir_5m != dir_15m:
            return 0, ""
        abs_mom = abs(mom_5m)
        if abs_mom > 0.003:
            base = 35
        elif abs_mom > 0.002:
            base = 25
        elif abs_mom > 0.001:
            base = 15
        else:
            base = 0
        consistency_bonus = 15 if dir_15m == dir_5m and dir_15m else 0
        return min(50, base + consistency_bonus), dir_5m

    def _score_volatility(self, high_5m, low_5m, close_5m) -> Tuple[int, float]:
        atr_s = _atr(high_5m, low_5m, close_5m, ATR_SHORT)
        atr_l = _atr(high_5m, low_5m, close_5m, ATR_LONG)
        if atr_l <= 0 or atr_s <= 0:
            return 15, atr_s
        ratio = atr_s / atr_l
        if ratio < 0.7:
            score = 30
        elif ratio < 0.9:
            score = 25
        elif ratio < 1.1:
            score = 15
        elif ratio < 1.5:
            score = 5
        else:
            score = 0
        return score, atr_s

    def _score_volume(self, volume_5m, direction) -> int:
        if len(volume_5m) < 20:
            return 10
        recent_vol = float(volume_5m[-1])
        avg_vol = float(np.mean(volume_5m[-20:]))
        if avg_vol <= 0:
            return 10
        ratio = recent_vol / avg_vol
        if ratio > VOLUME_SURGE_RATIO:
            return 20
        elif ratio > VOLUME_LOW_RATIO:
            return 10
        return 0

    def _daily_direction_multiplier(self, daily_bar, signal_dir,
                                    symbol: str = "") -> float:
        if daily_bar is None or len(daily_bar) < MOM_DAILY_LOOKBACK + 1:
            return 1.0
        closes = daily_bar["close"].values
        daily_mom = (closes[-1] - closes[-MOM_DAILY_LOOKBACK - 1]) / closes[-MOM_DAILY_LOOKBACK - 1]
        if abs(daily_mom) < 0.002:
            return 1.0
        # Per-symbol dm from SYMBOL_PROFILES（IF中性1.0/1.0, IM/IC动量1.2/0.8）
        prof = SYMBOL_PROFILES.get(symbol, _DEFAULT_PROFILE) if symbol else _DEFAULT_PROFILE
        dm_trend = prof.get("dm_trend", 1.1)
        dm_contra = prof.get("dm_contrarian", 0.9)
        daily_dir = "LONG" if daily_mom > 0 else "SHORT"
        if daily_dir == signal_dir:
            return dm_trend
        elif signal_dir and daily_dir != signal_dir:
            return dm_contra
        return 1.0

    @staticmethod
    def _intraday_filter_mild(intraday_return: float, direction: str) -> float:
        """低波动区间：温和版日内过滤。

        逻辑与高波动模式一致（逆势罚、顺势轻罚），阈值更低。
        旧版把顺势/逆势搞反了（顺势追跌f=0.3），导致丢失+1840pt利润。

        涨>3%:   做多(追涨)0.8  做空(逆势)0.5
        涨1.5-3%: 做多(追涨)0.9  做空(逆势)0.7
        跌>3%:   做空(追跌)0.8  做多(逆势)0.5
        跌1.5-3%: 做空(追跌)0.9  做多(逆势)0.7
        <1.5%: 1.0
        """
        thr = INTRADAY_REVERSAL_THRESHOLD  # 0.015
        abs_ret = abs(intraday_return)
        if abs_ret < thr:
            return 1.0

        if intraday_return > thr * 2:       # 涨>3%
            return 0.8 if direction == "LONG" else 0.5
        elif intraday_return > thr:          # 涨1.5-3%
            return 0.9 if direction == "LONG" else 0.7
        elif intraday_return < -thr * 2:     # 跌>3%
            return 0.8 if direction == "SHORT" else 0.5
        elif intraday_return < -thr:         # 跌1.5-3%
            return 0.9 if direction == "SHORT" else 0.7
        return 1.0

    @staticmethod
    def _intraday_filter(
        intraday_return: float, direction: str, zscore: float | None = None,
    ) -> float:
        """高波动区间：宽阈值的方向感知过滤。

        逆势重罚，顺势>2%轻罚，1-2%不罚。

        涨>3%: 做多0.8 做空0.3 (极端大涨)
        涨2-3%: 做多0.9 做空0.5
        涨1-2%: 做多1.0 做空0.7
        跌>3%: 做空0.8 做多0.3 (Z<-2时0.7)
        跌2-3%: 做空0.9 做多0.5 (Z<-2时0.8)
        跌1-2%: 做空1.0 做多0.7 (Z<-2时1.0)
        <1%: 1.0

        注：IV/VRP恐慌环境动态调整经回测验证无效（释放的信号被止损系统
        快速止出，实际PnL反而下降），保持静态阈值。
        """
        abs_ret = abs(intraday_return)
        if abs_ret < 0.01:
            return 1.0

        z = zscore if zscore is not None else 0.0

        if intraday_return > 0.03:
            base = 0.8 if direction == "LONG" else 0.3
        elif intraday_return > 0.02:
            base = 0.9 if direction == "LONG" else 0.5
        elif intraday_return > 0.01:
            base = 1.0 if direction == "LONG" else 0.7
        elif intraday_return < -0.03:
            if direction == "SHORT":
                base = 0.8
            else:
                base = 0.7 if z < -2.0 else 0.3
        elif intraday_return < -0.02:
            if direction == "SHORT":
                base = 0.9
            else:
                base = 0.8 if z < -2.0 else 0.5
        elif intraday_return < -0.01:
            if direction == "SHORT":
                base = 1.0
            else:
                base = 1.0 if z < -2.0 else 0.7
        else:
            base = 1.0

        return base


# ---------------------------------------------------------------------------
# v3 信号生成器：品种差异化策略
# ---------------------------------------------------------------------------

class SignalGeneratorV3:
    """品种差异化信号系统，基于实证研究结果。

    IM = 纯动量（短回看，无反转过滤，日线一致加分更多）
    IH = 均值回归为主（新增反转维度，低阈值反转触发）
    IF = 混合（上午动量为主，下午偏反转，高阈值）
    """

    def __init__(self, config: Dict | None = None):
        cfg = config or {}
        self.min_signal_score: int = cfg.get("min_signal_score", 55)
        self.debug: bool = cfg.get("debug", False)
        self._opening_bars: int = cfg.get("opening_bars", 6)

    def _profile(self, symbol: str) -> Dict:
        return SYMBOL_PROFILES.get(symbol, _DEFAULT_PROFILE)

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def update(
        self, symbol: str, bar_5m: pd.DataFrame,
        bar_15m: pd.DataFrame | None, daily_bar: pd.DataFrame | None,
        quote_data: Dict | None = None,
        sentiment: Optional[SentimentData] = None,
        zscore: float | None = None,
        is_high_vol: bool = True,
        d_override: Dict[str, float] | None = None,
    ) -> Optional[IntradaySignal]:
        result = self.score_all(
            symbol, bar_5m, bar_15m, daily_bar, quote_data, sentiment,
            zscore=zscore, is_high_vol=is_high_vol, d_override=d_override,
        )
        if result is None:
            return None
        if result["total"] < self.min_signal_score or not result["direction"]:
            return None

        close = float(bar_5m.iloc[-1]["close"])
        atr_val = result.get("atr_short", close * 0.005)
        if atr_val <= 0:
            atr_val = close * 0.005
        if result["direction"] == "LONG":
            stop = close - min(atr_val * 2, close * 0.01)
        else:
            stop = close + min(atr_val * 2, close * 0.01)

        prof = self._profile(symbol)
        reason_parts = [
            f"v3|{prof['style'][:3]}",
            f"mom={result['s_momentum']}",
            f"vol={result['s_volatility']}",
            f"qty={result['s_volume']}",
        ]
        if result.get("s_reversal", 0):
            reason_parts.append(f"rev={result['s_reversal']}")
        reason_parts.append(f"tw={result.get('time_weight', 1.0):.1f}")
        if result.get("sentiment_mult", 1.0) != 1.0:
            reason_parts.append(f"sent={result['sentiment_mult']:.2f}")

        return IntradaySignal(
            symbol=symbol,
            datetime=str(bar_5m.index[-1]) if isinstance(
                bar_5m.index, pd.DatetimeIndex) else str(
                bar_5m.iloc[-1].get("datetime", "")),
            direction=result["direction"], score=result["total"],
            entry_price=close, stop_loss=round(stop, 1),
            signal_type="INTRADAY",
            components={
                "momentum": {"score": result["s_momentum"]},
                "volatility": {"score": result["s_volatility"]},
                "volume": {"score": result["s_volume"]},
                "reversal": {"score": result.get("s_reversal", 0)},
            },
            reason="|".join(reason_parts),
        )

    def score_all(
        self, symbol: str, bar_5m: pd.DataFrame,
        bar_15m: pd.DataFrame | None, daily_bar: pd.DataFrame | None,
        quote_data: Dict | None = None,
        sentiment: Optional[SentimentData] = None,
        zscore: float | None = None,
        is_high_vol: bool = True,
        d_override: Dict[str, float] | None = None,
        vol_profile: Dict[str, list] | None = None,
    ) -> Dict | None:
        """使用 v2 的 50/30/20 评分体系，叠加品种差异化参数。"""
        prof = self._profile(symbol)
        lb_5m = prof["momentum_lookback_5m"]

        if bar_5m is None or len(bar_5m) < lb_5m + 1:
            return None

        utc_time = _get_utc_time(bar_5m)
        if utc_time and (utc_time < NO_TRADE_BEFORE or utc_time > NO_TRADE_AFTER):
            return None

        close_5m = bar_5m["close"].values
        high_5m = bar_5m["high"].values
        low_5m = bar_5m["low"].values
        volume_5m = bar_5m["volume"].values
        current_close = float(close_5m[-1])

        # --- 维度1: 动量 (50分) — 使用品种专属lookback ---
        s_mom, mom_dir = self._score_momentum(close_5m, bar_15m, prof)

        # --- 维度2: 波动率 (30分) ---
        s_vol, atr_short = self._score_volatility(high_5m, low_5m, close_5m)

        # --- 维度3: 成交量 (20分) — 同时段分位数法 ---
        if vol_profile and utc_time:
            slot = utc_time.strftime("%H:%M")
            hist_vols = vol_profile.get(slot)
            if hist_vols and len(hist_vols) >= 5:
                s_qty = volume_percentile_q(float(volume_5m[-1]), hist_vols)
            else:
                s_qty = self._score_volume(volume_5m)
        else:
            s_qty = self._score_volume(volume_5m)

        # --- 维度4: 反转 (IH专属, 0-20附加分, 可覆盖方向) ---
        s_rev = 0
        rev_dir = ""
        today_bars = _extract_today_bars(bar_5m)
        intraday_return = 0.0
        if len(today_bars) > 0:
            open_price = float(today_bars.iloc[0]["open"])
            if open_price > 0:
                intraday_return = (current_close - open_price) / open_price

        w_rev = prof.get("reversal_weight", 0)
        if w_rev > 0 and prof["reversal_filter"]:
            s_rev, rev_dir = self._score_reversal(
                intraday_return, close_5m, prof)

        # --- 方向决定 ---
        direction, s_mom, s_rev = self._resolve_direction(
            mom_dir, s_mom, rev_dir, s_rev, prof)

        # --- 布林带突破加分 (0~20分，仅动量已确认方向时生效) ---
        s_breakout, breakout_note = 0, ""
        if s_mom > 0 and direction:
            s_breakout, breakout_note = _score_boll_breakout(
                close_5m, bar_15m, direction, volume_5m)

        # --- 日线方向乘数（品种专属 lookback 和 bonus）---
        daily_mult = self._daily_direction_multiplier(
            daily_bar, direction, prof)
        # Morning Briefing d_override: 覆盖daily_mult（和monitor一致）
        if d_override and direction:
            daily_mult = d_override.get(direction, daily_mult)

        raw_total = s_mom + s_vol + s_qty + s_rev + s_breakout
        adjusted = float(raw_total) * daily_mult

        # --- 反转过滤 ---
        # IF (reversal_filter=True, w_rev=0): 标准v2式乘数（追涨×0.3，反转×1.3）
        # IH (reversal_filter=True, w_rev>0): 通过反转维度处理，不用乘数
        # IM (reversal_filter=False): 不惩罚追涨（实证：IM涨后继续涨），
        #     但保留v2的反转方向加成（跌后做多×1.3）
        intraday_filter = 1.0
        if prof["reversal_filter"] and w_rev == 0:
            thresh = prof["reversal_threshold"]
            intraday_filter = self._intraday_multiplier(
                intraday_return, direction, thresh)
            adjusted *= intraday_filter
        elif not prof["reversal_filter"]:
            # IM模式：只保留反转方向加成，不惩罚追势
            intraday_filter = self._momentum_intraday_filter(
                intraday_return, direction)
            adjusted *= intraday_filter

        # --- 时段权重（品种专属）---
        tw = _get_session_weight(
            utc_time, prof["session_multiplier"]) if utc_time else 1.0
        adjusted *= tw

        # --- 期权情绪乘数 ---
        sent_mult, sent_reason = calc_sentiment_multiplier(direction, sentiment)
        adjusted *= sent_mult

        total = max(0, min(100, int(round(adjusted))))

        # 第二层：Z-Score硬过滤（只在高波动区间生效）
        pre_z_total = total
        total, z_filter = _apply_zscore_filter(total, direction, zscore, is_high_vol)

        # 第四层：RSI回归确认
        rsi_bonus = 0
        rsi_note = ""
        if is_high_vol and zscore is not None and abs(zscore) > 2.0:
            rsi_bonus, rsi_note = _rsi_reversal_bonus(close_5m, zscore, direction)
            total = min(100, total + rsi_bonus)

        rsi_val = _calc_rsi(close_5m) if len(close_5m) >= 15 else 50.0

        return {
            "total": total,
            "direction": direction,
            "style": prof["style"],
            "s_momentum": s_mom,
            "s_volatility": s_vol,
            "s_volume": s_qty,
            "s_reversal": s_rev,
            "s_breakout": s_breakout, "breakout_note": breakout_note,
            "daily_mult": daily_mult,
            "intraday_filter": intraday_filter,
            "time_weight": tw,
            "raw_total": raw_total,
            "atr_short": atr_short,
            "sentiment_mult": sent_mult,
            "sentiment_reason": sent_reason,
            "pre_z_total": pre_z_total,
            "z_filter": z_filter,
            "rsi": rsi_val,
            "rsi_bonus": rsi_bonus,
            "rsi_note": rsi_note,
            "is_high_vol": is_high_vol,
        }

    # ------------------------------------------------------------------
    # 动量评分 (50分) — 与 v2 相同的尺度，品种专属 lookback
    # ------------------------------------------------------------------

    def _score_momentum(
        self, close_5m: np.ndarray, bar_15m: pd.DataFrame | None,
        prof: Dict,
    ) -> Tuple[int, str]:
        lb_5m = prof["momentum_lookback_5m"]
        lb_15m = prof["momentum_lookback_15m"]

        if len(close_5m) < lb_5m + 1:
            return 0, ""

        mom_5m = (close_5m[-1] - close_5m[-lb_5m - 1]) / close_5m[-lb_5m - 1]
        dir_5m = "LONG" if mom_5m > 0 else "SHORT" if mom_5m < 0 else ""

        dir_15m = ""
        if bar_15m is not None and len(bar_15m) >= lb_15m + 1:
            c15 = bar_15m["close"].values
            mom_15m = (c15[-1] - c15[-lb_15m - 1]) / c15[-lb_15m - 1]
            dir_15m = "LONG" if mom_15m > 0 else "SHORT" if mom_15m < 0 else ""

        if dir_15m and dir_5m != dir_15m:
            return 0, ""

        abs_mom = abs(mom_5m)

        # 与 v2 完全相同的阈值和分数
        if abs_mom > 0.003:
            base = 35
        elif abs_mom > 0.002:
            base = 25
        elif abs_mom > 0.001:
            base = 15
        else:
            base = 0

        consistency_bonus = 15 if dir_15m == dir_5m and dir_15m else 0
        return min(50, base + consistency_bonus), dir_5m

    # ------------------------------------------------------------------
    # 波动率评分 (30分)
    # ------------------------------------------------------------------

    @staticmethod
    def _score_volatility(high_5m, low_5m, close_5m) -> Tuple[int, float]:
        atr_s = _atr(high_5m, low_5m, close_5m, ATR_SHORT)
        atr_l = _atr(high_5m, low_5m, close_5m, ATR_LONG)
        if atr_l <= 0 or atr_s <= 0:
            return 15, atr_s
        ratio = atr_s / atr_l
        if ratio < 0.7:
            score = 30
        elif ratio < 0.9:
            score = 25
        elif ratio < 1.1:
            score = 15
        elif ratio < 1.5:
            score = 5
        else:
            score = 0
        return score, atr_s

    # ------------------------------------------------------------------
    # 成交量评分 (20分)
    # ------------------------------------------------------------------

    @staticmethod
    def _score_volume(volume_5m: np.ndarray) -> int:
        if len(volume_5m) < 20:
            return 10
        recent_vol = float(volume_5m[-1])
        avg_vol = float(np.mean(volume_5m[-20:]))
        if avg_vol <= 0:
            return 10
        ratio = recent_vol / avg_vol
        if ratio > VOLUME_SURGE_RATIO:
            return 20
        elif ratio > VOLUME_LOW_RATIO:
            return 10
        return 0

    # ------------------------------------------------------------------
    # 反转维度（IH 专属，0-20 内部尺度，带方向）
    # ------------------------------------------------------------------

    @staticmethod
    def _score_reversal(
        intraday_return: float, close_5m: np.ndarray, prof: Dict,
    ) -> Tuple[int, str]:
        """反转维度评分。返回 (score, direction)。

        超跌 → 做多信号；超涨 → 做空信号。
        同时检查最近K线是否出现减速迹象（确认反转启动）。
        """
        w_rev = prof.get("reversal_weight", 0)
        thresh = prof["reversal_threshold"]
        if w_rev <= 0 or thresh <= 0:
            return 0, ""

        score = 0
        direction = ""

        # 减速检查：最近3根K线的变化递减
        decel = False
        if len(close_5m) >= 4:
            d1 = close_5m[-2] - close_5m[-3]
            d2 = close_5m[-1] - close_5m[-2]
            if intraday_return < 0:
                decel = d2 > d1  # 跌幅收窄
            else:
                decel = d2 < d1  # 涨幅收窄

        if intraday_return < -thresh:
            # 超跌做多
            direction = "LONG"
            abs_ret = abs(intraday_return)
            if abs_ret > thresh * 2:
                score = w_rev          # 满分
            elif abs_ret > thresh * 1.5:
                score = int(w_rev * 0.75)
            else:
                score = int(w_rev * 0.5)
            if decel:
                score = min(w_rev, score + int(w_rev * 0.25))

        elif intraday_return > thresh:
            # 超涨做空
            direction = "SHORT"
            if intraday_return > thresh * 2:
                score = w_rev
            elif intraday_return > thresh * 1.5:
                score = int(w_rev * 0.75)
            else:
                score = int(w_rev * 0.5)
            if decel:
                score = min(w_rev, score + int(w_rev * 0.25))

        return score, direction

    # ------------------------------------------------------------------
    # 方向冲突解决
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_direction(
        mom_dir: str, s_mom: int,
        rev_dir: str, s_rev: int,
        prof: Dict,
    ) -> Tuple[str, int, int]:
        """解决动量和反转方向冲突。

        IH（均值回归型）：反转维度得分 > 15 且方向冲突 → 采用反转方向
        其他品种：动量优先
        """
        if not rev_dir or not mom_dir or rev_dir == mom_dir:
            direction = mom_dir or rev_dir
            return direction, s_mom, s_rev

        # 方向冲突
        if prof["style"] == "MEAN_REVERSION":
            w_rev = prof.get("reversal_weight", 0)
            if s_rev >= int(w_rev * 0.75):
                # 反转信号强，采用反转方向，削减动量分
                return rev_dir, int(s_mom * 0.3), s_rev
            else:
                # 反转信号弱，取消双方
                return "", 0, 0
        else:
            # 动量优先，反转分清零
            return mom_dir, s_mom, 0

    # ------------------------------------------------------------------
    # 日线方向乘数
    # ------------------------------------------------------------------

    @staticmethod
    def _daily_direction_multiplier(
        daily_bar: pd.DataFrame | None, signal_dir: str, prof: Dict,
    ) -> float:
        lb = prof["momentum_lookback_daily"]
        if daily_bar is None or len(daily_bar) < lb + 1:
            return 1.0
        closes = daily_bar["close"].values
        daily_mom = (closes[-1] - closes[-lb - 1]) / closes[-lb - 1]
        if abs(daily_mom) < 0.002:
            return 1.0
        daily_dir = "LONG" if daily_mom > 0 else "SHORT"
        if daily_dir == signal_dir:
            return prof.get("daily_align_bonus", 1.2)
        elif signal_dir and daily_dir != signal_dir:
            return prof.get("daily_conflict_penalty", 0.7)
        return 1.0

    # ------------------------------------------------------------------
    # 日内涨跌幅乘数（IF 用，非独立维度）
    # ------------------------------------------------------------------

    @staticmethod
    def _intraday_multiplier(
        intraday_return: float, direction: str, threshold: float,
    ) -> float:
        """日内涨跌幅乘数（加强版）。"""
        abs_ret = abs(intraday_return)
        if abs_ret < threshold:
            return 1.0

        extreme = threshold * 1.33  # ~2.0% if threshold=1.5%

        if intraday_return > extreme:
            if direction == "LONG":
                return 0.0
            elif direction == "SHORT":
                return 1.5
        elif intraday_return > threshold:
            if direction == "LONG":
                return 0.1
            elif direction == "SHORT":
                return 1.3

        if intraday_return < -extreme:
            if direction == "SHORT":
                return 0.0
            elif direction == "LONG":
                return 1.5
        elif intraday_return < -threshold:
            if direction == "SHORT":
                return 0.1
            elif direction == "LONG":
                return 1.3

        return 1.0

    @staticmethod
    def _momentum_intraday_filter(
        intraday_return: float, direction: str,
    ) -> float:
        """IM专用：不惩罚追势，只奖励逆势入场。

        IM实证：涨后不回调(38%回调率)，跌后不反弹(52.9%)。
        所以追涨追跌不惩罚（×1.0），但逆势入场仍给奖励（×1.3）。
        """
        thresh = 0.015
        if abs(intraday_return) < thresh:
            return 1.0
        if intraday_return > thresh and direction == "SHORT":
            return 1.3  # 涨多了做空（虽然概率不高，但给加分）
        if intraday_return < -thresh and direction == "LONG":
            return 1.3  # 跌多了做多
        return 1.0  # 追涨追跌不惩罚
