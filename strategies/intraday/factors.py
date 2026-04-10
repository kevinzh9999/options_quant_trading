"""
因子化评分框架 — 替代 score_all() 的硬编码逻辑。

每个评分维度（M/V/Q/B/S）抽象为 ScoringFactor 子类。
FactorCombiner 负责：因子求和 → 乘数管道 → 过滤器。
输出格式与旧 score_all() 完全一致，接口零改动。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 数据容器
# ---------------------------------------------------------------------------

@dataclass
class ScoringContext:
    """打包 score_all 的所有输入，传给每个因子。"""
    symbol: str
    close_5m: np.ndarray
    high_5m: np.ndarray
    low_5m: np.ndarray
    volume_5m: np.ndarray
    bar_15m: Optional[pd.DataFrame]
    daily_bar: Optional[pd.DataFrame]
    utc_time: Optional[time]
    vol_profile: Optional[Dict[str, list]]
    profile: Dict               # SYMBOL_PROFILES 条目
    current_close: float
    bar_5m_df: Optional[pd.DataFrame] = None   # 原始 DataFrame（breakout/startup需要）


@dataclass
class FactorResult:
    """单个因子的输出。"""
    score: int = 0
    direction: str = ""         # "LONG"/"SHORT"/""（只有 Momentum 设置方向）
    meta: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 因子基类
# ---------------------------------------------------------------------------

class ScoringFactor(ABC):
    """评分因子抽象基类。"""
    name: str = ""
    max_score: int = 0

    @abstractmethod
    def score(self, ctx: ScoringContext) -> FactorResult:
        ...


# ---------------------------------------------------------------------------
# 具体因子
# ---------------------------------------------------------------------------

# 常量（从 A_share_momentum_signal_v2 模块级变量复制，避免循环导入）
_MOM_15M_LB = 6
_DYN_LB_LOW = 4
_DYN_LB_HIGH = 12
_DYN_AMP_THR = 0.015
_ATR_SHORT = 5
_ATR_LONG = 40
_VOLUME_SURGE = 1.5
_VOLUME_LOW = 0.5
_VOLUME_PCT_HIGH = 0.75
_VOLUME_PCT_LOW = 0.25


def _atr(high, low, close, period):
    """Average True Range."""
    n = len(high)
    if n < period + 1:
        return 0.0
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
    )
    return float(np.mean(tr[-period:]))


class MomentumFactor(ScoringFactor):
    """M维度：动量评分（0-50）+ 方向判定。含动态lookback。"""
    name = "momentum"
    max_score = 50

    def score(self, ctx: ScoringContext) -> FactorResult:
        close = ctx.close_5m
        # 动态 lookback
        lb = _DYN_LB_HIGH
        if len(close) >= 6:
            recent = close[-min(48, len(close)):]
            amp = (max(recent) - min(recent)) / recent[0] if recent[0] > 0 else 0
            if amp < _DYN_AMP_THR:
                lb = _DYN_LB_LOW

        if len(close) < lb + 1:
            return FactorResult(0, "")

        mom_5m = (close[-1] - close[-lb - 1]) / close[-lb - 1]
        dir_5m = "LONG" if mom_5m > 0 else "SHORT" if mom_5m < 0 else ""

        # 15m确认
        dir_15m = ""
        if ctx.bar_15m is not None and len(ctx.bar_15m) >= _MOM_15M_LB + 1:
            c15 = ctx.bar_15m["close"].values
            mom_15m = (c15[-1] - c15[-_MOM_15M_LB - 1]) / c15[-_MOM_15M_LB - 1]
            dir_15m = "LONG" if mom_15m > 0 else "SHORT" if mom_15m < 0 else ""
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
    """V维度：波动率评分（0-30）。ATR短/长比率。"""
    name = "volatility"
    max_score = 30

    def score(self, ctx: ScoringContext) -> FactorResult:
        atr_s = _atr(ctx.high_5m, ctx.low_5m, ctx.close_5m, _ATR_SHORT)
        atr_l = _atr(ctx.high_5m, ctx.low_5m, ctx.close_5m, _ATR_LONG)
        if atr_l <= 0 or atr_s <= 0:
            return FactorResult(15, meta={"atr_short": atr_s})
        ratio = atr_s / atr_l
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
    """Q维度：成交量评分（0-20）。优先用历史分位，fallback到rolling。"""
    name = "volume"
    max_score = 20

    def score(self, ctx: ScoringContext) -> FactorResult:
        volume = ctx.volume_5m
        # 优先用历史同时段分位
        if ctx.vol_profile and ctx.utc_time:
            slot = ctx.utc_time.strftime("%H:%M")
            hist_vols = ctx.vol_profile.get(slot)
            if hist_vols and len(hist_vols) >= 5:
                cur_vol = float(volume[-1])
                pct = sum(1 for v in hist_vols if v <= cur_vol) / len(hist_vols)
                if pct > _VOLUME_PCT_HIGH:      # 严格大于（与旧volume_percentile_q一致）
                    return FactorResult(20, meta={"method": "percentile", "pct": pct})
                elif pct > _VOLUME_PCT_LOW:    # 严格大于
                    return FactorResult(10, meta={"method": "percentile", "pct": pct})
                return FactorResult(0, meta={"method": "percentile", "pct": pct})

        # fallback: rolling均值
        if len(volume) < 20:
            return FactorResult(10, meta={"method": "fallback_short"})
        recent_vol = float(volume[-1])
        avg_vol = float(np.mean(volume[-20:]))
        if avg_vol <= 0:
            return FactorResult(10, meta={"method": "fallback_zero"})
        ratio = recent_vol / avg_vol
        if ratio > _VOLUME_SURGE:
            return FactorResult(20, meta={"method": "rolling", "ratio": ratio})
        elif ratio > _VOLUME_LOW:
            return FactorResult(10, meta={"method": "rolling", "ratio": ratio})
        return FactorResult(0, meta={"method": "rolling", "ratio": ratio})


class BreakoutFactor(ScoringFactor):
    """B维度：布林带突破加分（0-20）。需要先有动量方向。"""
    name = "breakout"
    max_score = 20

    def score(self, ctx: ScoringContext, mom_dir: str = "",
              s_mom: int = 0) -> FactorResult:
        if s_mom <= 0 or not mom_dir:
            return FactorResult(0)
        # 调用模块级函数（逻辑不变）
        from strategies.intraday.A_share_momentum_signal_v2 import _score_boll_breakout
        s, note = _score_boll_breakout(ctx.close_5m, ctx.bar_15m,
                                        mom_dir, ctx.volume_5m)
        return FactorResult(s, meta={"note": note})


class StartupFactor(ScoringFactor):
    """S维度：趋势启动检测（0-15）。需要先有动量方向。"""
    name = "startup"
    max_score = 15

    def score(self, ctx: ScoringContext, mom_dir: str = "") -> FactorResult:
        if not mom_dir:
            return FactorResult(0)
        # 计算volume分位
        vol_pct = -1.0
        if ctx.vol_profile and ctx.utc_time:
            slot = ctx.utc_time.strftime("%H:%M")
            hv = ctx.vol_profile.get(slot)
            if hv and len(hv) >= 5:
                cur_v = float(ctx.volume_5m[-1])
                vol_pct = sum(1 for v in hv if v <= cur_v) / len(hv)
        from strategies.intraday.A_share_momentum_signal_v2 import _score_trend_startup
        s, note = _score_trend_startup(ctx.close_5m, ctx.high_5m, ctx.low_5m,
                                        ctx.volume_5m, mom_dir,
                                        vol_percentile=vol_pct)
        return FactorResult(s, meta={"note": note})


# ---------------------------------------------------------------------------
# 因子组合器
# ---------------------------------------------------------------------------

class FactorCombiner:
    """替代 score_all() 方法体。

    流程：因子求和 → 乘数管道（dm/f/tw/sent）→ 硬过滤（zscore/rsi）。
    输出 dict 格式与旧 score_all() 完全一致。
    """

    def __init__(self, factors: List[ScoringFactor],
                 weights: Optional[Dict[str, float]] = None):
        self.factors = factors
        self.weights = weights or {}       # factor_name -> weight, default 1.0

    def combine(self, ctx: ScoringContext,
                # 乘数管道参数（当前中性化，未来可恢复）
                daily_bar: Optional[pd.DataFrame] = None,
                sentiment=None,
                zscore: Optional[float] = None,
                is_high_vol: bool = True,
                d_override: Optional[Dict[str, float]] = None,
                ) -> Dict:
        """运行所有因子，合成最终得分。返回与旧 score_all() 格式一致的 dict。"""

        # ── Phase 1: 计算各因子 ──
        mom_result = None
        vol_result = None
        qty_result = None
        brk_result = FactorResult(0)
        stp_result = FactorResult(0)

        mom_dir = ""
        s_mom = 0

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

        # 应用权重
        s_vol = int(round((vol_result.score if vol_result else 0) * self.weights.get("volatility", 1.0)))
        s_qty = int(round((qty_result.score if qty_result else 0) * self.weights.get("volume", 1.0)))
        s_breakout = int(round(brk_result.score * self.weights.get("breakout", 1.0)))
        s_startup = int(round(stp_result.score * self.weights.get("startup", 1.0)))

        atr_short = vol_result.meta.get("atr_short", 0.0) if vol_result else 0.0

        # 合成breakout_note
        brk_note = brk_result.meta.get("note", "")
        stp_note = stp_result.meta.get("note", "")
        breakout_note = brk_note
        if stp_note:
            breakout_note = (brk_note + "+" + stp_note) if brk_note else stp_note

        # ── Phase 2: 乘数管道 ──
        from strategies.intraday.A_share_momentum_signal_v2 import (
            _get_session_weight, calc_sentiment_multiplier,
            _apply_zscore_filter, _rsi_reversal_bonus, _calc_rsi,
            _extract_today_bars, SYMBOL_PROFILES, _DEFAULT_PROFILE,
            INTRADAY_REVERSAL_THRESHOLD, MOM_DAILY_LOOKBACK,
            SignalGeneratorV2,
        )

        prof = ctx.profile

        # daily_mult
        daily_mult = 1.0
        if daily_bar is not None and len(daily_bar) >= MOM_DAILY_LOOKBACK + 1:
            closes = daily_bar["close"].values
            daily_mom = (closes[-1] - closes[-MOM_DAILY_LOOKBACK - 1]) / closes[-MOM_DAILY_LOOKBACK - 1]
            if abs(daily_mom) >= 0.002:
                dm_trend = prof.get("dm_trend", 1.1)
                dm_contra = prof.get("dm_contrarian", 0.9)
                daily_dir = "LONG" if daily_mom > 0 else "SHORT"
                if daily_dir == mom_dir:
                    daily_mult = dm_trend
                elif mom_dir and daily_dir != mom_dir:
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

        # time_weight
        tw = _get_session_weight(ctx.utc_time, prof.get("session_multiplier", {})) if ctx.utc_time else 1.0
        adjusted *= tw

        # sentiment_mult
        sent_mult, sent_reason = calc_sentiment_multiplier(mom_dir, sentiment)
        adjusted *= sent_mult

        total = max(0, min(100, int(round(adjusted))))

        # ── Phase 3: 硬过滤 ──
        pre_z_total = total
        total, z_filter = _apply_zscore_filter(total, mom_dir, zscore, is_high_vol)

        rsi_bonus = 0
        rsi_note = ""
        if is_high_vol and zscore is not None and abs(zscore) > 2.0:
            rsi_bonus, rsi_note = _rsi_reversal_bonus(ctx.close_5m, zscore, mom_dir)
            total = min(100, total + rsi_bonus)

        rsi_val = _calc_rsi(ctx.close_5m) if len(ctx.close_5m) >= 15 else 50.0

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


# 避免循环导入：在 combine() 内部才 import SignalGeneratorV2
# 这里提前声明以便类型引用
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from strategies.intraday.A_share_momentum_signal_v2 import SignalGeneratorV2


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------

def create_default_combiner(weights: Optional[Dict[str, float]] = None) -> FactorCombiner:
    """创建默认的5因子组合器（与旧 score_all 行为一致）。"""
    factors = [
        MomentumFactor(),
        VolatilityFactor(),
        VolumeFactor(),
        BreakoutFactor(),
        StartupFactor(),
    ]
    return FactorCombiner(factors, weights)
