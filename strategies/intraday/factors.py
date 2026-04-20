"""
因子化评分框架 — entry和exit都基于原子因子（atomic_factors.py）的组合。

Entry: ScoringFactor子类（M/V/Q/B/S），由FactorCombiner组合。
Exit:  ExitCondition子类（StopLoss/Trailing/TC/ME/MidBreak/TimeStop），由ExitEvaluator评估。
两者共享同一套原子因子。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from strategies.intraday import atomic_factors as af


# ═══════════════════════════════════════════════════════════════════════════
# Entry 侧
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ScoringContext:
    """打包 score_all 的所有输入。"""
    symbol: str
    close_5m: np.ndarray
    high_5m: np.ndarray
    low_5m: np.ndarray
    volume_5m: np.ndarray
    bar_15m: Optional[pd.DataFrame]
    daily_bar: Optional[pd.DataFrame]
    utc_time: Optional[time]
    vol_profile: Optional[Dict[str, list]]
    profile: Dict
    current_close: float
    bar_5m_df: Optional[pd.DataFrame] = None


@dataclass
class FactorResult:
    score: int = 0
    direction: str = ""
    meta: Dict = field(default_factory=dict)


class ScoringFactor(ABC):
    name: str = ""
    max_score: int = 0

    @abstractmethod
    def score(self, ctx: ScoringContext) -> FactorResult:
        ...


# ── 动态lookback常量 ──
_DYN_LB_LOW = 4
_DYN_LB_HIGH = 12
_DYN_AMP_THR = 0.015
_MOM_15M_LB = 6
_VOLUME_PCT_HIGH = 0.75
_VOLUME_PCT_LOW = 0.25
_VOLUME_SURGE = 1.5
_VOLUME_LOW = 0.5


class MomentumFactor(ScoringFactor):
    """M维度（0-50）：基于 af.momentum + af.amplitude 原子因子。"""
    name = "momentum"
    max_score = 50

    def score(self, ctx: ScoringContext) -> FactorResult:
        close = ctx.close_5m
        # 动态lookback：基于 af.amplitude
        lb = _DYN_LB_HIGH
        if len(close) >= 6:
            amp = af.amplitude(close, 48)
            if amp < _DYN_AMP_THR:
                lb = _DYN_LB_LOW

        # 5m动量
        mom_5m = af.momentum(close, lb)
        if mom_5m == 0.0 and len(close) < lb + 1:
            return FactorResult(0, "")
        dir_5m = af.momentum_direction(mom_5m)

        # 15m确认
        dir_15m = ""
        if ctx.bar_15m is not None and len(ctx.bar_15m) >= _MOM_15M_LB + 1:
            mom_15m = af.momentum(ctx.bar_15m["close"].values, _MOM_15M_LB)
            dir_15m = af.momentum_direction(mom_15m)
        if dir_15m and dir_5m != dir_15m:
            return FactorResult(0, "")

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
        return FactorResult(min(50, base + consistency_bonus), dir_5m,
                            {"lb": lb, "mom_5m": mom_5m})


class VolatilityFactor(ScoringFactor):
    """V维度（0-30）：基于 af.atr_ratio 原子因子。"""
    name = "volatility"
    max_score = 30

    def score(self, ctx: ScoringContext) -> FactorResult:
        atr_s = af.atr(ctx.high_5m, ctx.low_5m, ctx.close_5m, 5)
        ratio = af.atr_ratio(ctx.high_5m, ctx.low_5m, ctx.close_5m, 5, 40)
        if atr_s <= 0:
            return FactorResult(15, meta={"atr_short": atr_s})
        if ratio < 0.7:
            s = 30
        elif ratio < 0.9:
            s = 25
        elif ratio < 1.1:
            s = 15
        elif ratio < 1.5:
            s = 5
        else:
            s = 0
        return FactorResult(s, meta={"atr_short": atr_s, "atr_ratio": ratio})


class VolumeFactor(ScoringFactor):
    """Q维度（0-20）：基于 af.volume_percentile / af.volume_ratio 原子因子。"""
    name = "volume"
    max_score = 20

    def score(self, ctx: ScoringContext) -> FactorResult:
        volume = ctx.volume_5m
        if ctx.vol_profile and ctx.utc_time:
            slot = ctx.utc_time.strftime("%H:%M")
            hist_vols = ctx.vol_profile.get(slot)
            if hist_vols and len(hist_vols) >= 5:
                pct = af.volume_percentile(float(volume[-1]), hist_vols)
                if pct > _VOLUME_PCT_HIGH:
                    return FactorResult(20, meta={"method": "percentile", "pct": pct})
                elif pct > _VOLUME_PCT_LOW:
                    return FactorResult(10, meta={"method": "percentile", "pct": pct})
                return FactorResult(0, meta={"method": "percentile", "pct": pct})
        # fallback
        if len(volume) < 20:
            return FactorResult(10, meta={"method": "fallback_short"})
        ratio = af.volume_ratio(float(volume[-1]), float(np.mean(volume[-20:])))
        if ratio > _VOLUME_SURGE:
            return FactorResult(20, meta={"method": "rolling", "ratio": ratio})
        elif ratio > _VOLUME_LOW:
            return FactorResult(10, meta={"method": "rolling", "ratio": ratio})
        return FactorResult(0, meta={"method": "rolling", "ratio": ratio})


class BreakoutFactor(ScoringFactor):
    """B维度（0-20）：布林带突破。需要先有动量方向。"""
    name = "breakout"
    max_score = 20

    def score(self, ctx: ScoringContext, mom_dir: str = "",
              s_mom: int = 0) -> FactorResult:
        if s_mom <= 0 or not mom_dir:
            return FactorResult(0)
        from strategies.intraday.A_share_momentum_signal_v2 import _score_boll_breakout
        s, note = _score_boll_breakout(ctx.close_5m, ctx.bar_15m,
                                        mom_dir, ctx.volume_5m)
        return FactorResult(s, meta={"note": note})


class StartupFactor(ScoringFactor):
    """S维度（0-15）：趋势启动。需要先有动量方向。"""
    name = "startup"
    max_score = 15

    def score(self, ctx: ScoringContext, mom_dir: str = "") -> FactorResult:
        if not mom_dir:
            return FactorResult(0)
        vol_pct = -1.0
        if ctx.vol_profile and ctx.utc_time:
            slot = ctx.utc_time.strftime("%H:%M")
            hv = ctx.vol_profile.get(slot)
            if hv and len(hv) >= 5:
                vol_pct = af.volume_percentile(float(ctx.volume_5m[-1]), hv)
        from strategies.intraday.A_share_momentum_signal_v2 import _score_trend_startup
        s, note = _score_trend_startup(ctx.close_5m, ctx.high_5m, ctx.low_5m,
                                        ctx.volume_5m, mom_dir,
                                        vol_percentile=vol_pct)
        return FactorResult(s, meta={"note": note})


# ── Entry组合器 ──

class FactorCombiner:
    """Entry侧：因子求和 → 乘数管道 → 硬过滤。输出与旧score_all()格式一致。"""

    def __init__(self, factors: List[ScoringFactor],
                 weights: Optional[Dict[str, float]] = None):
        self.factors = factors
        self.weights = weights or {}

    def combine(self, ctx: ScoringContext,
                daily_bar=None, sentiment=None, zscore=None,
                is_high_vol: bool = True, d_override=None) -> Dict:

        # Phase 1: 因子计算
        mom_dir = ""
        s_mom = 0
        vol_result = FactorResult(0)
        qty_result = FactorResult(0)
        brk_result = FactorResult(0)
        stp_result = FactorResult(0)

        mom_result = FactorResult(0)
        for f in self.factors:
            w = self.weights.get(f.name, 1.0)
            if w == 0:
                continue
            if f.name == "momentum":
                mom_result = f.score(ctx)
                mom_dir = mom_result.direction
                s_mom = int(round(mom_result.score * w))
            elif f.name == "volatility":
                vol_result = f.score(ctx)
            elif f.name == "volume":
                qty_result = f.score(ctx)
            elif f.name == "breakout":
                brk_result = f.score(ctx, mom_dir=mom_dir, s_mom=s_mom)
            elif f.name == "startup":
                stp_result = f.score(ctx, mom_dir=mom_dir)

        s_vol = int(round(vol_result.score * self.weights.get("volatility", 1.0)))
        s_qty = int(round(qty_result.score * self.weights.get("volume", 1.0)))
        s_breakout = int(round(brk_result.score * self.weights.get("breakout", 1.0)))
        s_startup = int(round(stp_result.score * self.weights.get("startup", 1.0)))
        atr_short = vol_result.meta.get("atr_short", 0.0)

        brk_note = brk_result.meta.get("note", "")
        stp_note = stp_result.meta.get("note", "")
        breakout_note = brk_note
        if stp_note:
            breakout_note = (brk_note + "+" + stp_note) if brk_note else stp_note

        # Phase 2: 乘数管道
        from strategies.intraday.A_share_momentum_signal_v2 import (
            _get_session_weight, calc_sentiment_multiplier,
            _apply_zscore_filter, _rsi_reversal_bonus, _calc_rsi,
            _extract_today_bars, MOM_DAILY_LOOKBACK, SignalGeneratorV2,
        )

        prof = ctx.profile
        daily_mult = 1.0
        if daily_bar is not None and len(daily_bar) >= MOM_DAILY_LOOKBACK + 1:
            closes = daily_bar["close"].values
            daily_mom = af.momentum(closes, MOM_DAILY_LOOKBACK)
            if abs(daily_mom) >= 0.002:
                dm_trend = prof.get("dm_trend", 1.1)
                dm_contra = prof.get("dm_contrarian", 0.9)
                daily_dir = af.momentum_direction(daily_mom)
                if daily_dir == mom_dir:
                    daily_mult = dm_trend
                elif mom_dir:
                    daily_mult = dm_contra
        if d_override and mom_dir:
            daily_mult = d_override.get(mom_dir, daily_mult)

        raw_total = s_mom + s_vol + s_qty + s_breakout + s_startup
        adjusted = raw_total * daily_mult

        # intraday_filter
        intraday_filter = 1.0
        prev_close = 0.0
        if daily_bar is not None and len(daily_bar) >= 2 and ctx.bar_5m_df is not None:
            today_bars = _extract_today_bars(ctx.bar_5m_df)
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
                prev_close = float(daily_bar.iloc[-2]["close"])
        if prev_close <= 0 and ctx.bar_5m_df is not None:
            today_bars = _extract_today_bars(ctx.bar_5m_df)
            if len(today_bars) > 0:
                prev_close = float(today_bars.iloc[0]["open"])
        if prev_close > 0:
            intraday_return = (ctx.current_close - prev_close) / prev_close
            if is_high_vol:
                intraday_filter = SignalGeneratorV2._intraday_filter(intraday_return, mom_dir, zscore)
            else:
                intraday_filter = SignalGeneratorV2._intraday_filter_mild(intraday_return, mom_dir)
            adjusted *= intraday_filter

        tw = _get_session_weight(ctx.utc_time, prof.get("session_multiplier", {})) if ctx.utc_time else 1.0
        adjusted *= tw

        sent_mult, sent_reason = calc_sentiment_multiplier(mom_dir, sentiment)
        adjusted *= sent_mult

        total = max(0, min(100, int(round(adjusted))))

        # Phase 3: 硬过滤
        pre_z_total = total
        total, z_filter = _apply_zscore_filter(total, mom_dir, zscore, is_high_vol)

        rsi_bonus = 0
        rsi_note = ""
        if is_high_vol and zscore is not None and abs(zscore) > 2.0:
            rsi_bonus, rsi_note = _rsi_reversal_bonus(ctx.close_5m, zscore, mom_dir)
            total = min(100, total + rsi_bonus)

        rsi_val = af.rsi(ctx.close_5m) if len(ctx.close_5m) >= 15 else 50.0

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
            # 原始连续数值（诊断用）
            "raw_mom_5m": mom_result.meta.get("mom_5m", 0.0) if mom_result.meta else 0.0,
            "raw_atr_ratio": vol_result.meta.get("atr_ratio", 0.0) if vol_result.meta else 0.0,
            "raw_vol_pct": qty_result.meta.get("pct", -1.0) if qty_result.meta else -1.0,
            "raw_vol_ratio": qty_result.meta.get("ratio", -1.0) if qty_result.meta else -1.0,
            "raw_vol_method": qty_result.meta.get("method", "") if qty_result.meta else "",
        }


# ═══════════════════════════════════════════════════════════════════════════
# Exit 侧
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ExitContext:
    """打包 check_exit 的所有输入。"""
    position: dict          # entry_price, direction, entry_time_utc, highest_since, lowest_since, volume, bars_below_mid
    current_price: float
    bar_5m: Optional[pd.DataFrame]
    bar_15m: Optional[pd.DataFrame]
    current_time_utc: str
    symbol: str
    profile: Dict
    # 预计算的原子因子值（由ExitEvaluator填充）
    hold_minutes: int = 0
    loss_pct: float = 0.0
    pnl_pct_val: float = 0.0
    zone_5m: str = ""
    zone_15m: str = ""
    b5_mid: float = 0.0
    b5_std: float = 0.0
    b15_mid: float = 0.0
    b15_std: float = 0.0
    boll_price: float = 0.0


class ExitCondition(ABC):
    """Exit条件抽象基类。"""
    name: str = ""
    priority: int = 99     # 数字越小优先级越高
    urgency: str = "NORMAL"

    @abstractmethod
    def check(self, ctx: ExitContext) -> Optional[dict]:
        """返回exit dict或None。"""
        ...


class StopLossCondition(ExitCondition):
    """基于 af.pnl_pct 原子因子。"""
    name = "STOP_LOSS"
    priority = 10
    urgency = "URGENT"

    def check(self, ctx: ExitContext) -> Optional[dict]:
        sl_pct = ctx.profile.get("stop_loss_pct", 0.005)
        if ctx.loss_pct > sl_pct:
            return {"should_exit": True, "exit_volume": ctx.position.get("volume", 1),
                    "exit_reason": "STOP_LOSS", "exit_urgency": "URGENT"}
        return None


class LunchCloseCondition(ExitCondition):
    """午休平仓：亏损→LUNCH_CLOSE，盈利→紧trailing→LUNCH_TRAIL。"""
    name = "LUNCH_CLOSE"
    priority = 15  # 在STOP_LOSS(10)之后，TRAILING(30)之前
    urgency = "URGENT"

    def check(self, ctx: ExitContext) -> Optional[dict]:
        from strategies.intraday.A_share_momentum_signal_v2 import LUNCH_CLOSE_UTC, TRAILING_STOP_LUNCH
        if ctx.current_time_utc < LUNCH_CLOSE_UTC or ctx.current_time_utc >= "05:00":
            return None
        pos = ctx.position
        direction = pos["direction"]
        entry_price = pos["entry_price"]
        volume = pos.get("volume", 1)
        profitable = ctx.pnl_pct_val > 0
        if not profitable:
            return {"should_exit": True, "exit_volume": volume,
                    "exit_reason": "LUNCH_CLOSE", "exit_urgency": "URGENT"}
        # 盈利：紧trailing
        highest = pos.get("highest_since", entry_price)
        lowest = pos.get("lowest_since", entry_price)
        if direction == "LONG" and highest > entry_price:
            dd = af.trailing_drawdown(ctx.current_price, highest, "LONG")
            if dd > TRAILING_STOP_LUNCH:
                return {"should_exit": True, "exit_volume": volume,
                        "exit_reason": "LUNCH_TRAIL", "exit_urgency": "NORMAL"}
        elif direction == "SHORT" and lowest < entry_price:
            du = af.trailing_drawdown(ctx.current_price, lowest, "SHORT")
            if du > TRAILING_STOP_LUNCH:
                return {"should_exit": True, "exit_volume": volume,
                        "exit_reason": "LUNCH_TRAIL", "exit_urgency": "NORMAL"}
        return None


class TrailingStopCondition(ExitCondition):
    """基于 af.trailing_drawdown + af.hold_time + af.boll_zone 原子因子。"""
    name = "TRAILING_STOP"
    priority = 30
    urgency = "NORMAL"

    def check(self, ctx: ExitContext) -> Optional[dict]:
        if not ctx.profile.get("trailing_stop_enabled", True):
            return None
        pos = ctx.position
        direction = pos["direction"]
        entry_price = pos["entry_price"]
        highest = pos.get("highest_since", entry_price)
        lowest = pos.get("lowest_since", entry_price)

        # 持仓时间→trail宽度
        if ctx.hold_minutes < 15:
            trail_pct = 0.005
        elif ctx.hold_minutes < 30:
            trail_pct = 0.006
        elif ctx.hold_minutes < 60:
            trail_pct = 0.008
        else:
            trail_pct = 0.010

        # 15m趋势确认加宽
        if ctx.zone_15m:
            if direction == "LONG":
                fifteen_ok = ctx.zone_15m in ("MID_UPPER", "UPPER_ZONE", "ABOVE_UPPER")
            else:
                fifteen_ok = ctx.zone_15m in ("MID_LOWER", "LOWER_ZONE", "BELOW_LOWER")
            if ctx.pnl_pct_val > 0.005 and fifteen_ok:
                trail_pct += 0.002

        trail_pct *= ctx.profile.get("trailing_stop_scale", 1.0)

        dd = af.trailing_drawdown(ctx.current_price,
                                   highest if direction == "LONG" else lowest,
                                   direction)
        if dd > trail_pct:
            profitable = (ctx.current_price > entry_price) if direction == "LONG" \
                else (ctx.current_price < entry_price)
            if profitable:
                return {"should_exit": True, "exit_volume": pos.get("volume", 1),
                        "exit_reason": "TRAILING_STOP", "exit_urgency": "NORMAL"}
        return None


class TrendCompleteCondition(ExitCondition):
    """基于 af.boll_zone（5m+15m双极端）原子因子。

    修复(2026-04-20)：加两个保护条件防止秒平——
    1. 最小持仓10分钟（方案1）：刚开仓不触发
    2. 开仓时已在极端zone的不触发（方案4）：只有'从内部涨到极端'才算趋势完成
    """
    name = "TREND_COMPLETE"
    priority = 40
    urgency = "NORMAL"
    MIN_HOLD_MINUTES = 10  # 最小持仓时间

    def check(self, ctx: ExitContext) -> Optional[dict]:
        if not ctx.zone_5m or not ctx.zone_15m:
            return None
        # 方案1：最小持仓时间保护
        if ctx.hold_minutes < self.MIN_HOLD_MINUTES:
            return None
        direction = ctx.position["direction"]
        # 方案4：如果开仓时已在极端zone，不触发TC
        # （开仓时记录entry_zone到position dict，若无则不阻止——兼容旧持仓）
        entry_zone = ctx.position.get("entry_zone_5m", "")
        if direction == "LONG":
            if entry_zone == "ABOVE_UPPER":
                return None  # 开仓时就在上轨之上，不算趋势完成
            if ctx.zone_5m == "ABOVE_UPPER" and ctx.zone_15m == "ABOVE_UPPER":
                return {"should_exit": True, "exit_volume": ctx.position.get("volume", 1),
                        "exit_reason": "TREND_COMPLETE", "exit_urgency": "NORMAL"}
        else:
            if entry_zone == "BELOW_LOWER":
                return None  # 开仓时就在下轨之下
            if ctx.zone_5m == "BELOW_LOWER" and ctx.zone_15m == "BELOW_LOWER":
                return {"should_exit": True, "exit_volume": ctx.position.get("volume", 1),
                        "exit_reason": "TREND_COMPLETE", "exit_urgency": "NORMAL"}
        return None


class MomentumExhaustedCondition(ExitCondition):
    """基于 af.narrow_range + af.boll_zone + af.price_trending + af.hold_time 原子因子。"""
    name = "MOMENTUM_EXHAUSTED"
    priority = 50
    urgency = "NORMAL"

    def check(self, ctx: ExitContext) -> Optional[dict]:
        min_hold = ctx.profile.get("me_min_hold", 20)
        if ctx.hold_minutes < min_hold:
            return None
        if ctx.bar_5m is None or len(ctx.bar_5m) < 23 or ctx.b5_std <= 0:
            return None
        direction = ctx.position["direction"]
        me_ratio = ctx.profile.get("me_ratio", 0.10)
        nr = af.narrow_range(ctx.bar_5m, 3, ctx.b5_std)
        if nr < 0 or nr >= me_ratio:
            return None
        trending = af.price_trending(ctx.bar_5m, 3, ctx.b5_std, direction)
        if trending:
            return None
        if direction == "LONG" and ctx.zone_15m in ("ABOVE_UPPER", "UPPER_ZONE"):
            return {"should_exit": True, "exit_volume": ctx.position.get("volume", 1),
                    "exit_reason": "MOMENTUM_EXHAUSTED", "exit_urgency": "NORMAL"}
        elif direction == "SHORT" and ctx.zone_15m in ("BELOW_LOWER", "LOWER_ZONE"):
            return {"should_exit": True, "exit_volume": ctx.position.get("volume", 1),
                    "exit_reason": "MOMENTUM_EXHAUSTED", "exit_urgency": "NORMAL"}
        return None


class MidBreakCondition(ExitCondition):
    """基于 af.boll_zone（5m破中轨）+ 15m确认 + 连续计数。"""
    name = "MID_BREAK"
    priority = 60
    urgency = "NORMAL"

    def check(self, ctx: ExitContext) -> Optional[dict]:
        if not ctx.zone_5m or np.isnan(ctx.b5_mid):
            return None
        direction = ctx.position["direction"]
        boll_price = ctx.boll_price

        if direction == "LONG":
            five_below = boll_price < ctx.b5_mid
            fifteen_below = ctx.zone_15m in ("MID_LOWER", "LOWER_ZONE", "BELOW_LOWER") if ctx.zone_15m else False
        else:
            five_below = boll_price > ctx.b5_mid
            fifteen_below = ctx.zone_15m in ("MID_UPPER", "UPPER_ZONE", "ABOVE_UPPER") if ctx.zone_15m else False

        bars_count = ctx.profile.get("mid_break_bars", 3)
        if five_below:
            bm = ctx.position.get("bars_below_mid", 0) + 1
            ctx.position["bars_below_mid"] = bm
            if bm >= bars_count and fifteen_below:
                return {"should_exit": True, "exit_volume": ctx.position.get("volume", 1),
                        "exit_reason": "MID_BREAK", "exit_urgency": "NORMAL"}
        else:
            ctx.position["bars_below_mid"] = 0
        return None


class TimeStopCondition(ExitCondition):
    """基于 af.hold_time + af.pnl_pct 原子因子。"""
    name = "TIME_STOP"
    priority = 70
    urgency = "NORMAL"

    def check(self, ctx: ExitContext) -> Optional[dict]:
        max_hold = ctx.profile.get("time_stop_minutes", 60)
        if ctx.hold_minutes > max_hold and ctx.pnl_pct_val <= 0:
            return {"should_exit": True, "exit_volume": ctx.position.get("volume", 1),
                    "exit_reason": "TIME_STOP", "exit_urgency": "NORMAL"}
        return None


class ExitEvaluator:
    """Exit侧组合器：预计算原子因子，按优先级评估exit条件。"""

    def __init__(self, conditions: List[ExitCondition],
                 weights: Optional[Dict[str, float]] = None):
        self.conditions = sorted(conditions, key=lambda c: c.priority)
        self.weights = weights or {}  # name -> weight, 0=禁用

    def evaluate(self, position: dict, current_price: float,
                 bar_5m: Optional[pd.DataFrame], bar_15m: Optional[pd.DataFrame],
                 current_time_utc: str, symbol: str = "",
                 spot_price: float = 0.0, is_high_vol: bool = True) -> dict:
        """评估所有exit条件，返回最高优先级的结果。与旧check_exit输出格式一致。"""
        from strategies.intraday.A_share_momentum_signal_v2 import (
            SYMBOL_PROFILES, _DEFAULT_PROFILE, EOD_CLOSE_UTC,
            LUNCH_CLOSE_UTC, TRAILING_STOP_LUNCH,
        )

        prof = SYMBOL_PROFILES.get(symbol, _DEFAULT_PROFILE) if symbol else _DEFAULT_PROFILE
        direction = position["direction"]
        entry_price = position["entry_price"]
        volume = position.get("volume", 1)
        boll_price = spot_price if spot_price > 0 else current_price

        NO_EXIT = {"should_exit": False, "exit_volume": 0,
                   "exit_reason": "", "exit_urgency": "NORMAL"}

        # ── 硬约束（不因子化）──
        # P1: EOD
        if current_time_utc >= EOD_CLOSE_UTC:
            return {"should_exit": True, "exit_volume": volume,
                    "exit_reason": "EOD_CLOSE", "exit_urgency": "URGENT"}

        # P2: Lunch — 移到因子化条件中（优先级在STOP_LOSS之后），不在硬约束里

        # ── 预计算原子因子 ──
        hold_min = af.hold_time(position.get("entry_time_utc", ""), current_time_utc)
        loss_pct = -af.pnl_pct(current_price, entry_price, direction)  # 正值=亏损
        pnl_val = af.pnl_pct(current_price, entry_price, direction)

        zone_5m = zone_15m = ""
        b5_mid = b5_std = b15_mid = b15_std = float("nan")
        if bar_5m is not None and len(bar_5m) >= 20:
            c5 = bar_5m["close"].astype(float)
            b5_mid, b5_std = af.boll_params(c5)
            if not np.isnan(b5_mid) and b5_std > 0:
                zone_5m = af.boll_zone(boll_price, b5_mid, b5_std)
        if bar_15m is not None and len(bar_15m) >= 20:
            c15 = bar_15m["close"].astype(float)
            b15_mid, b15_std = af.boll_params(c15)
            if not np.isnan(b15_mid) and b15_std > 0:
                zone_15m = af.boll_zone(boll_price, b15_mid, b15_std)

        ectx = ExitContext(
            position=position, current_price=current_price,
            bar_5m=bar_5m, bar_15m=bar_15m,
            current_time_utc=current_time_utc, symbol=symbol, profile=prof,
            hold_minutes=hold_min, loss_pct=loss_pct, pnl_pct_val=pnl_val,
            zone_5m=zone_5m, zone_15m=zone_15m,
            b5_mid=b5_mid, b5_std=b5_std, b15_mid=b15_mid, b15_std=b15_std,
            boll_price=boll_price,
        )

        # ── 按优先级评估因子化条件 ──
        for cond in self.conditions:
            w = self.weights.get(cond.name, 1.0)
            if w == 0:
                continue  # 禁用
            result = cond.check(ectx)
            if result is not None:
                return result

        return NO_EXIT


# ═══════════════════════════════════════════════════════════════════════════
# 工厂函数
# ═══════════════════════════════════════════════════════════════════════════

def create_default_combiner(weights: Optional[Dict[str, float]] = None) -> FactorCombiner:
    factors = [MomentumFactor(), VolatilityFactor(), VolumeFactor(),
               BreakoutFactor(), StartupFactor()]
    return FactorCombiner(factors, weights)


def create_default_exit_evaluator(weights: Optional[Dict[str, float]] = None) -> ExitEvaluator:
    conditions = [
        StopLossCondition(),
        LunchCloseCondition(),
        TrailingStopCondition(),
        TrendCompleteCondition(),
        MomentumExhaustedCondition(),
        MidBreakCondition(),
        TimeStopCondition(),
    ]
    return ExitEvaluator(conditions, weights)
