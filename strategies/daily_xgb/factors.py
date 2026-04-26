"""Daily XGB factors — extends strategies/daily/factors.py with regime factors.

3 regime factors added on top of the 18-factor default pipeline:
  - close_sma60_ratio: close / SMA(60)
  - slope_60d:         60-day price slope (normalized)
  - vol_regime:        RV5 / RV60 (vol expansion ratio)
"""
from __future__ import annotations

import numpy as np

from strategies.daily.factors import (
    DailyContext,
    DailyFactor,
    DailyFactorPipeline,
    build_default_pipeline,
    load_default_context,
    add_forward_returns,
)


class CloseSma60Factor(DailyFactor):
    name = "close_sma60_ratio"
    category = "regime"

    def compute(self, td, ctx):
        h = ctx.px_history(td, 60)
        if h is None or len(h) < 60:
            return None
        return float(h.iloc[-1]["close"] / h["close"].mean())


class Slope60dFactor(DailyFactor):
    name = "slope_60d"
    category = "regime"

    def compute(self, td, ctx):
        h = ctx.px_history(td, 60)
        if h is None or len(h) < 60:
            return None
        c = h["close"].astype(float).values
        slope = float(np.polyfit(range(60), c, 1)[0])
        mean = float(c.mean())
        return slope / max(mean, 1)


class VolRegimeFactor(DailyFactor):
    """Recent 5d realized vol / 60d realized vol."""
    name = "vol_regime"
    category = "regime"

    def compute(self, td, ctx):
        h = ctx.px_history(td, 61)
        if h is None or len(h) < 61:
            return None
        c = h["close"].astype(float).values
        rets = np.diff(np.log(c))
        rv5 = float(np.std(rets[-5:]) * np.sqrt(252))
        rv60 = float(np.std(rets) * np.sqrt(252))
        return rv5 / max(rv60, 1e-6)


def build_full_pipeline() -> DailyFactorPipeline:
    """Build pipeline with default 18 + 3 regime factors = 21 total."""
    base = build_default_pipeline()
    base.factors.extend([CloseSma60Factor(), Slope60dFactor(), VolRegimeFactor()])
    base.feature_names = [f.name for f in base.factors]
    return base


__all__ = [
    "CloseSma60Factor",
    "Slope60dFactor",
    "VolRegimeFactor",
    "build_full_pipeline",
    "load_default_context",
    "add_forward_returns",
    "DailyContext",
]
