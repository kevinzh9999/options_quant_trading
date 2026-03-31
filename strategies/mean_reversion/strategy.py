"""
strategy.py
-----------
职责：均值回归策略主类。

基于 RSI、布林带和 OU 过程识别短期超买超卖，
在均值回归完成后获利平仓。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from strategies.base import BaseStrategy, Signal, SignalDirection, SignalStrength, StrategyConfig
from .signal import MeanReversionSignalGenerator, MeanReversionSignal, MeanReversionRegime

logger = logging.getLogger(__name__)


@dataclass
class MeanReversionConfig(StrategyConfig):
    """
    均值回归策略配置。

    Attributes
    ----------
    rsi_window : int
        RSI 计算窗口
    rsi_overbought : float
        超买阈值
    rsi_oversold : float
        超卖阈值
    bb_window : int
        布林带窗口
    bb_std : float
        布林带标准差倍数
    max_half_life : float
        最大允许半衰期（交易日），超过此值不入场
    stop_loss_pct : float
        止损幅度（百分比）
    """
    rsi_window: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    bb_window: int = 20
    bb_std: float = 2.0
    max_half_life: float = 20.0
    stop_loss_pct: float = 0.02


class MeanReversionStrategy(BaseStrategy):
    """
    均值回归策略。

    支持标的：IF/IH/IC/IM 期货（短期超买超卖修正）

    Parameters
    ----------
    config : MeanReversionConfig
        策略配置
    """

    def __init__(self, config: MeanReversionConfig) -> None:
        super().__init__(config)
        self.mr_config: MeanReversionConfig = config
        self._signal_generator = MeanReversionSignalGenerator(
            rsi_window=config.rsi_window,
            rsi_overbought=config.rsi_overbought,
            rsi_oversold=config.rsi_oversold,
            bb_window=config.bb_window,
            bb_std=config.bb_std,
        )

    @property
    def name(self) -> str:
        return "均值回归策略"

    def generate_signals(
        self,
        trade_date: str,
        market_data: dict[str, pd.DataFrame],
    ) -> list[Signal]:
        """
        生成均值回归信号。

        Notes
        -----
        - 超卖时生成做多信号，超买时生成做空信号
        - OU 半衰期过长时不入场
        - 目标出场价为布林带中轨（均线）
        """
        raise NotImplementedError("TODO: 实现均值回归信号生成主流程")

    def on_fill(
        self,
        order_id: str,
        instrument: str,
        direction: str,
        volume: int,
        price: float,
        trade_date: str,
    ) -> None:
        """成交回调"""
        raise NotImplementedError("TODO: 实现成交回调")

    @staticmethod
    def _mr_signal_to_signal(
        mr_signal: MeanReversionSignal,
        strategy_id: str,
        position_size: int,
    ) -> Signal:
        """将 MeanReversionSignal 转换为通用 Signal"""
        direction = (
            SignalDirection.LONG
            if mr_signal.regime == MeanReversionRegime.OVERSOLD
            else SignalDirection.SHORT
            if mr_signal.regime == MeanReversionRegime.OVERBOUGHT
            else SignalDirection.NEUTRAL
        )
        return Signal(
            strategy_id=strategy_id,
            signal_date=mr_signal.signal_date,
            instrument=mr_signal.instrument,
            direction=direction,
            strength=SignalStrength.MODERATE,
            target_volume=position_size,
            price_ref=mr_signal.entry_price,
            metadata={
                "regime": mr_signal.regime.value,
                "rsi": mr_signal.rsi,
                "bb_zscore": mr_signal.bb_zscore,
                "ou_half_life": mr_signal.ou_half_life,
                "exit_price": mr_signal.exit_price,
                "stop_loss": mr_signal.stop_loss,
            },
        )
