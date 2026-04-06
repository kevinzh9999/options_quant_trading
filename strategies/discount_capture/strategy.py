"""
strategy.py
-----------
IM 贴水捕获策略主类，继承 BaseStrategy。

策略逻辑：
  做多 IM 期货 + 保护性 Put（下行保护），捕获贴水（discount）随时间收敛至零的收益。
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from strategies.base import BaseStrategy, Signal, SignalDirection, SignalStrength, StrategyConfig
from strategies.discount_capture.signal import DiscountSignal
from strategies.discount_capture.position import DiscountPosition

logger = logging.getLogger(__name__)


class DiscountCaptureStrategy(BaseStrategy):
    """
    IM 贴水捕获策略。

    做多 IM 期货 + 保护性 Put，捕获贴水随到期收敛的收益。

    Parameters
    ----------
    config : StrategyConfig
        策略配置
    db_manager : optional
        数据库管理器（优先使用；为 None 时尝试自动加载）
    account_equity : float
        账户权益（用于仓位计算），默认 1,000,000 元
    """

    DEFAULT_PARAMS = {
        "min_discount_rate": 0.08,       # 最低年化贴水率入场阈值
        "target_days_range": [30, 120],  # 目标剩余到期天数区间
        "put_delta": -0.15,              # 目标保护 Put Delta
        "max_loss_per_trade": 0.05,      # 每笔最大亏损（相对权益）
        "roll_days_before_expiry": 5,    # 到期前N天滚仓
    }

    def __init__(
        self,
        config: StrategyConfig,
        db_manager=None,
        account_equity: float = 1_000_000.0,
    ):
        super().__init__(config)
        self._db = db_manager
        self.account_equity = account_equity
        self.params = {**self.DEFAULT_PARAMS}
        if hasattr(config, "params") and config.params:
            self.params.update(config.params)

        self._discount_signal: DiscountSignal = None
        self._discount_position: DiscountPosition = None
        self.initialize()

    def initialize(self) -> None:
        """初始化信号生成器和仓位管理器。"""
        db = self._get_db()
        if db is not None:
            self._discount_signal = DiscountSignal(db)
            self._discount_position = DiscountPosition(
                account_equity=self.account_equity,
                max_allocation=0.3,
                max_loss_ratio=self.params.get("max_loss_per_trade", 0.05),
            )
            logger.info("DiscountCaptureStrategy 初始化完成")
        else:
            logger.warning("DiscountCaptureStrategy: 无数据库连接，信号生成将受限")

    def _get_db(self):
        """获取数据库连接（优先使用注入的实例）。"""
        if self._db is not None:
            return self._db
        try:
            from data.storage.db_manager import DBManager, get_db
            from config.config_loader import ConfigLoader
            config = ConfigLoader()
            db = get_db(config)
            self._db = db
            return db
        except Exception as e:
            logger.warning("无法自动加载数据库: %s", e)
            return None

    def generate_signals(
        self,
        trade_date: str,
        market_data: dict[str, pd.DataFrame] = None,
    ) -> list[Signal]:
        """
        生成贴水捕获信号列表。

        Parameters
        ----------
        trade_date : str
            交易日期，格式 YYYYMMDD
        market_data : dict, optional
            市场数据（此策略不依赖外部传入）

        Returns
        -------
        list[Signal]
            Signal 对象列表
        """
        if self._discount_signal is None:
            logger.warning("信号生成器未初始化，跳过 %s", trade_date)
            return []

        try:
            sig_data = self._discount_signal.generate_signal(trade_date)
        except Exception as e:
            logger.warning("贴水信号生成失败: %s", e)
            return []

        signal_str = sig_data.get("signal", "NONE")
        if signal_str == "NONE":
            return []

        # 映射强度
        strength_map = {
            "STRONG": SignalStrength.STRONG,
            "MEDIUM": SignalStrength.MODERATE,
            "WEAK": SignalStrength.WEAK,
        }
        strength = strength_map.get(signal_str, SignalStrength.WEAK)

        min_rate = self.params.get("min_discount_rate", 0.08)
        if sig_data.get("annualized_discount", 0) < min_rate:
            return []

        instrument = sig_data.get("recommended_contract", "IM.CFX") or "IM.CFX"
        ann_discount = sig_data.get("annualized_discount", 0.0)
        confidence = min(ann_discount / 0.20, 1.0)  # 归一化置信度

        signal = Signal(
            strategy_id=self.strategy_id,
            signal_date=trade_date,
            instrument=instrument,
            direction=SignalDirection.LONG,
            strength=strength,
            target_volume=1,  # 仓位由 calculate_positions 确定
            price_ref=0.0,
            confidence=confidence,
            metadata={
                "annualized_discount": ann_discount,
                "discount_percentile": sig_data.get("discount_percentile", 50.0),
                "days_to_expiry": sig_data.get("days_to_expiry", 0),
                "iml_code": sig_data.get("iml_code", ""),
                "all_contracts": (
                    sig_data["all_contracts"].to_dict("records")
                    if not sig_data.get("all_contracts", pd.DataFrame()).empty
                    else []
                ),
            },
            notes=(
                f"贴水捕获: {instrument} 年化贴水率={ann_discount*100:.1f}%  "
                f"剩余{sig_data.get('days_to_expiry', 0)}天  "
                f"历史{sig_data.get('discount_percentile', 0):.0f}百分位"
            ),
        )

        return [signal]

    def on_fill(
        self,
        order_id: str,
        instrument: str,
        direction: str,
        volume: int,
        price: float,
        trade_date: str,
    ) -> None:
        """成交回调（记录日志）。"""
        logger.info(
            "[DiscountCapture] 成交: %s  %s  %s  %d手  价格=%.2f  日期=%s",
            order_id, instrument, direction, volume, price, trade_date,
        )

    def calculate_positions(self, signals: list[Signal]) -> list[dict]:
        """
        根据信号计算具体持仓。

        Parameters
        ----------
        signals : list[Signal]

        Returns
        -------
        list[dict]
            每个 dict 含 instrument, lots, side, notes
        """
        if not signals or self._discount_position is None:
            return []

        positions = []
        for sig in signals:
            meta = sig.metadata or {}
            # 从 metadata 获取参考价格（若无则用 price_ref）
            fut_price = sig.price_ref or 8000.0  # 兜底默认值

            lots = self._discount_position.calculate_futures_lots(fut_price)
            positions.append({
                "instrument": sig.instrument,
                "lots": lots,
                "side": "long",
                "strategy": self.strategy_id,
                "notes": sig.notes,
                "annualized_discount": meta.get("annualized_discount", 0),
                "days_to_expiry": meta.get("days_to_expiry", 0),
            })

        return positions

    def on_bar(self, bar_data: Any) -> None:
        """行情更新回调（当前不处理）。"""
        pass

    def get_status(self) -> dict:
        """获取策略状态摘要。"""
        return {
            "strategy_id": self.strategy_id,
            "enabled": self.config.enabled,
            "params": self.params,
            "has_db": self._db is not None,
            "has_signal": self._discount_signal is not None,
        }
