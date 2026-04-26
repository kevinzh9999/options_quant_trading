"""Regime gates and enhancements for Daily XGB.

Three layers:
  G3s (SHORT block): suppress SHORT in structurally bullish regime
  N5  (LONG enh):    extended-bull → 10d×4 ATR; dip-bull → 15d×4 ATR
  M7  (SHORT enh):   extended-bear → 10d×4 ATR; rip-bear → 15d×4 ATR

If no enhancement matches, defaults to 5d hold and ATR×1.5 SL.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .config import DailyXGBConfig


@dataclass(frozen=True)
class RegimeState:
    """Per-day regime indicators required for gate decisions."""
    close_sma60: float
    close_sma200: float
    slope_60d: float
    vol_regime: float

    def is_valid(self) -> bool:
        return not (
            np.isnan(self.close_sma60)
            or np.isnan(self.close_sma200)
        )


@dataclass(frozen=True)
class TradeParams:
    """Output of regime decision: hold + SL config."""
    hold_days: int
    atr_k: float
    enhancement: str  # "default" | "n5_strict" | "n5_dip" | "m7_extended" | "m7_rip"


def block_short(state: RegimeState, cfg: DailyXGBConfig) -> bool:
    """G3s: True 表示该 SHORT 信号应被屏蔽（处于强 bull regime）."""
    if not state.is_valid():
        return False
    return (state.close_sma60 > cfg.g3s_strict_sma60_thr
            and state.close_sma200 > cfg.g3s_strict_sma200_thr)


def long_enhancement(state: RegimeState, cfg: DailyXGBConfig) -> Optional[TradeParams]:
    """N5: 返回 LONG enhancement 参数 (若 regime 触发)."""
    if not state.is_valid():
        return None
    # Strict bull (extended uptrend)
    if (state.close_sma60 > cfg.n5_strict_sma60_thr
            and state.close_sma200 > cfg.n5_strict_sma200_thr):
        return TradeParams(cfg.n5_strict_hold_days, cfg.n5_strict_atr_k, "n5_strict")
    # Dip bull (pullback in bull)
    if (state.close_sma200 > cfg.n5_dip_sma200_thr
            and state.close_sma60 < cfg.n5_dip_sma60_thr):
        return TradeParams(cfg.n5_dip_hold_days, cfg.n5_dip_atr_k, "n5_dip")
    return None


def short_enhancement(state: RegimeState, cfg: DailyXGBConfig) -> Optional[TradeParams]:
    """M7: SHORT enhancement parameters (mirrors N5)."""
    if not state.is_valid():
        return None
    # Extended bear
    if (state.close_sma60 < cfg.m7_extended_sma60_thr
            and state.close_sma200 < cfg.m7_extended_sma200_thr):
        return TradeParams(cfg.m7_extended_hold_days, cfg.m7_extended_atr_k, "m7_extended")
    # Rip bear (rally in bear)
    if (state.close_sma200 < cfg.m7_rip_sma200_thr
            and state.close_sma60 > cfg.m7_rip_sma60_thr):
        return TradeParams(cfg.m7_rip_hold_days, cfg.m7_rip_atr_k, "m7_rip")
    return None


def decide_trade_params(direction: str,
                          state: RegimeState,
                          cfg: DailyXGBConfig) -> Optional[TradeParams]:
    """Resolve hold/SL params for a given signal direction. Returns None if signal must be skipped."""
    direction = direction.upper()
    if direction == "SHORT" and block_short(state, cfg):
        return None  # 屏蔽 SHORT in bull
    if direction == "LONG":
        enh = long_enhancement(state, cfg)
        if enh is not None:
            return enh
    elif direction == "SHORT":
        enh = short_enhancement(state, cfg)
        if enh is not None:
            return enh
    return TradeParams(cfg.hold_days_default, cfg.atr_k_default, "default")


def compute_regime(closes: np.ndarray, idx: int) -> RegimeState:
    """Compute RegimeState for a single date index (closes is full series)."""
    if idx >= len(closes):
        return RegimeState(np.nan, np.nan, np.nan, np.nan)
    close = closes[idx]
    sma60 = closes[max(0, idx - 59): idx + 1].mean() if idx >= 59 else np.nan
    sma200 = closes[max(0, idx - 199): idx + 1].mean() if idx >= 199 else np.nan

    if idx >= 60:
        c = closes[idx - 59: idx + 1].astype(float)
        slope = float(np.polyfit(range(60), c, 1)[0])
        slope_60d = slope / max(c.mean(), 1)
    else:
        slope_60d = np.nan

    if idx >= 60:
        rets = np.diff(np.log(closes[idx - 60: idx + 1].astype(float)))
        rv5 = float(np.std(rets[-5:]) * np.sqrt(252)) if len(rets) >= 5 else np.nan
        rv60 = float(np.std(rets) * np.sqrt(252)) if len(rets) > 1 else np.nan
        vol_regime = rv5 / max(rv60, 1e-6) if not (np.isnan(rv5) or np.isnan(rv60)) else np.nan
    else:
        vol_regime = np.nan

    return RegimeState(
        close_sma60=close / sma60 if not np.isnan(sma60) and sma60 > 0 else np.nan,
        close_sma200=close / sma200 if not np.isnan(sma200) and sma200 > 0 else np.nan,
        slope_60d=slope_60d,
        vol_regime=vol_regime,
    )


__all__ = [
    "RegimeState", "TradeParams",
    "block_short", "long_enhancement", "short_enhancement",
    "decide_trade_params", "compute_regime",
]
