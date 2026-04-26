"""Daily XGB strategy — 跨日 trend-following swing strategy.

Walk-forward 因子化预测 + regime-aware enhancement (G3s+N5+M7) + ATR SL.

Walk-forward backtest: +1,572K/yr 1×lot, Calmar 2.93, Sharpe 5.64.
Conservative deployment: 1 lot/signal + cap 10 concurrent → +1,343K/yr realistic.

Strategy is fully isolated from intraday strategy:
  - Own DB tables (daily_xgb_*)
  - Own JSON pipes (tmp/daily_xgb_*)
  - Own TQ executor process (DXGB_ order prefix)
"""

from .config import DailyXGBConfig

__all__ = ["DailyXGBConfig"]
