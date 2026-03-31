"""
signal.py
---------
职责：均值回归信号生成。

基于 OU 过程参数估计和 RSI/布林带判断价格偏离程度，
识别短期超买超卖机会。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import pandas as pd

logger = logging.getLogger(__name__)


class MeanReversionRegime(str, Enum):
    """均值回归状态"""
    OVERBOUGHT = "overbought"     # 超买（向下均值回归机会）
    OVERSOLD = "oversold"         # 超卖（向上均值回归机会）
    NEUTRAL = "neutral"           # 中性


@dataclass
class MeanReversionSignal:
    """
    均值回归信号。

    Attributes
    ----------
    signal_date : str
        信号日期
    instrument : str
        合约代码
    underlying : str
        品种代码
    regime : MeanReversionRegime
        当前状态
    rsi : float
        RSI 指标值（0-100）
    bb_zscore : float
        布林带 Z-score（价格偏离均值的标准差倍数）
    ou_speed : float
        OU 过程均值回归速度（越大收敛越快）
    ou_half_life : float
        OU 过程半衰期（交易日）
    entry_price : float
        建议入场价格
    exit_price : float
        目标出场价格（均线）
    stop_loss : float
        止损价格
    notes : str
        备注
    """
    signal_date: str
    instrument: str
    underlying: str
    regime: MeanReversionRegime
    rsi: float = 50.0
    bb_zscore: float = 0.0
    ou_speed: float = 0.0
    ou_half_life: float = float("inf")
    entry_price: float = 0.0
    exit_price: float = 0.0
    stop_loss: float = 0.0
    notes: str = ""

    @property
    def is_actionable(self) -> bool:
        return self.regime != MeanReversionRegime.NEUTRAL


class MeanReversionSignalGenerator:
    """
    均值回归信号生成器。

    Parameters
    ----------
    rsi_window : int
        RSI 计算窗口，默认 14
    rsi_overbought : float
        RSI 超买阈值，默认 70
    rsi_oversold : float
        RSI 超卖阈值，默认 30
    bb_window : int
        布林带窗口，默认 20
    bb_std : float
        布林带标准差倍数，默认 2.0
    ou_lookback : int
        OU 过程拟合窗口，默认 60
    """

    def __init__(
        self,
        rsi_window: int = 14,
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0,
        bb_window: int = 20,
        bb_std: float = 2.0,
        ou_lookback: int = 60,
    ) -> None:
        self.rsi_window = rsi_window
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.bb_window = bb_window
        self.bb_std = bb_std
        self.ou_lookback = ou_lookback

    def generate(
        self,
        trade_date: str,
        instrument: str,
        underlying: str,
        price_df: pd.DataFrame,
    ) -> MeanReversionSignal:
        """
        生成均值回归信号。

        Parameters
        ----------
        price_df : pd.DataFrame
            包含 close 列的日线 DataFrame，按日期升序排列

        Notes
        -----
        - RSI > rsi_overbought 且价格在布林带上轨外 → OVERBOUGHT
        - RSI < rsi_oversold 且价格在布林带下轨外 → OVERSOLD
        - OU 过程半衰期过长（>30日）时，不生成信号（均值回归太慢）
        """
        raise NotImplementedError("TODO: 实现均值回归信号生成")

    def _calc_rsi(self, close: pd.Series) -> float:
        """计算 RSI"""
        raise NotImplementedError("TODO: 实现 RSI 计算")

    def _calc_bb_zscore(self, close: pd.Series) -> float:
        """计算布林带 Z-score"""
        raise NotImplementedError("TODO: 实现布林带 Z-score 计算")

    def _fit_ou_process(self, close: pd.Series) -> tuple[float, float]:
        """
        拟合 OU 过程参数。

        Returns
        -------
        tuple
            (mean_reversion_speed, half_life_in_days)
        """
        raise NotImplementedError("TODO: 实现 OU 过程拟合（使用 models.statistics.ou_process）")
