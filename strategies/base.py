"""
base.py
-------
职责：定义多策略框架的抽象基类和通用数据结构。

所有具体策略（波动率套利、趋势跟踪、价差交易、均值回归）
都继承 BaseStrategy，保证统一的接口规范。
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ======================================================================
# 通用信号数据结构
# ======================================================================

class SignalDirection(str, Enum):
    """统一信号方向枚举"""
    LONG = "long"           # 做多
    SHORT = "short"         # 做空
    NEUTRAL = "neutral"     # 中性/无操作
    CLOSE = "close"         # 平仓信号


class SignalStrength(str, Enum):
    """信号强度等级"""
    STRONG = "strong"       # 强信号
    MODERATE = "moderate"   # 中等信号
    WEAK = "weak"           # 弱信号


@dataclass
class Signal:
    """
    统一信号数据结构。

    所有策略的信号输出均使用此结构，便于执行层统一处理。

    Attributes
    ----------
    strategy_id : str
        发出信号的策略 ID
    signal_date : str
        信号生成日期，格式 YYYYMMDD
    instrument : str
        交易标的代码（合约代码或品种代码）
    direction : SignalDirection
        交易方向
    strength : SignalStrength
        信号强度
    target_volume : int
        目标持仓手数（0 表示清仓）
    price_ref : float
        参考价格（用于限价单定价）
    confidence : float
        信号置信度 [0, 1]
    metadata : dict
        策略特定的附加信息（如 VRP 值、行权价推荐等）
    notes : str
        备注
    """
    strategy_id: str
    signal_date: str
    instrument: str
    direction: SignalDirection
    strength: SignalStrength
    target_volume: int = 0
    price_ref: float = 0.0
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)
    notes: str = ""

    @property
    def is_actionable(self) -> bool:
        """信号是否应触发实际交易"""
        return self.direction != SignalDirection.NEUTRAL

    def to_dict(self) -> dict:
        """转换为字典，用于日志记录和数据库存储"""
        return {
            "strategy_id": self.strategy_id,
            "signal_date": self.signal_date,
            "instrument": self.instrument,
            "direction": self.direction.value,
            "strength": self.strength.value,
            "target_volume": self.target_volume,
            "price_ref": self.price_ref,
            "confidence": self.confidence,
            "notes": self.notes,
        }


# ======================================================================
# 策略配置
# ======================================================================

@dataclass
class StrategyConfig:
    """
    策略配置基类。

    每个策略通过继承此类定义自己的参数，
    框架通过此接口统一管理策略参数的序列化和验证。

    Attributes
    ----------
    strategy_id : str
        策略唯一标识符（如 'vol_arb_IO', 'trend_IF'）
    enabled : bool
        是否启用该策略
    universe : list[str]
        交易标的列表（品种代码）
    max_position : int
        单标的最大持仓手数
    dry_run : bool
        干运行模式（只生成信号，不实际下单）
    """
    strategy_id: str
    enabled: bool = True
    universe: list[str] = field(default_factory=list)
    max_position: int = 10
    dry_run: bool = True

    def validate(self) -> None:
        """验证配置合法性，子类可覆盖以添加额外校验"""
        if not self.strategy_id:
            raise ValueError("strategy_id 不能为空")
        if self.max_position <= 0:
            raise ValueError("max_position 必须大于 0")


# ======================================================================
# 策略基类
# ======================================================================

class BaseStrategy(ABC):
    """
    策略抽象基类。

    所有具体策略必须实现以下抽象方法：
    - generate_signals(): 核心信号生成逻辑
    - on_fill(): 成交回调处理

    框架负责调用 run() 方法，策略实现只需关注业务逻辑。

    Parameters
    ----------
    config : StrategyConfig
        策略配置对象
    """

    def __init__(self, config: StrategyConfig) -> None:
        self.config = config
        self.strategy_id = config.strategy_id
        self.logger = logging.getLogger(f"strategy.{self.strategy_id}")
        self._signals: list[Signal] = []

    # ------------------------------------------------------------------
    # 抽象方法（子类必须实现）
    # ------------------------------------------------------------------

    @abstractmethod
    def generate_signals(
        self,
        trade_date: str,
        market_data: dict[str, pd.DataFrame],
    ) -> list[Signal]:
        """
        生成当日交易信号。

        Parameters
        ----------
        trade_date : str
            交易日期，格式 YYYYMMDD
        market_data : dict[str, pd.DataFrame]
            市场数据字典，key 为品种代码，value 为日线/分钟线 DataFrame

        Returns
        -------
        list[Signal]
            当日信号列表（可为空列表）
        """
        ...

    @abstractmethod
    def on_fill(
        self,
        order_id: str,
        instrument: str,
        direction: str,
        volume: int,
        price: float,
        trade_date: str,
    ) -> None:
        """
        订单成交回调。

        Parameters
        ----------
        order_id : str
            订单 ID
        instrument : str
            合约代码
        direction : str
            交易方向（'buy' / 'sell'）
        volume : int
            成交手数
        price : float
            成交价格
        trade_date : str
            成交日期
        """
        ...

    # ------------------------------------------------------------------
    # 框架调用方法（子类通常不需要覆盖）
    # ------------------------------------------------------------------

    def run(
        self,
        trade_date: str,
        market_data: dict[str, pd.DataFrame],
    ) -> list[Signal]:
        """
        执行策略主流程（框架入口）。

        框架调用此方法，内部调用 generate_signals()，
        并做基础的日志记录和异常捕获。
        """
        if not self.config.enabled:
            self.logger.debug("策略已禁用，跳过 %s", trade_date)
            return []

        self.logger.info("运行策略 [%s] 日期=%s", self.strategy_id, trade_date)
        try:
            signals = self.generate_signals(trade_date, market_data)
            self._signals.extend(signals)
            self.logger.info(
                "策略 [%s] 生成 %d 个信号", self.strategy_id, len(signals)
            )
            return signals
        except Exception as exc:
            self.logger.exception(
                "策略 [%s] 运行异常: %s", self.strategy_id, exc
            )
            return []

    def get_signal_history(self) -> list[Signal]:
        """获取历史信号列表（用于分析和调试）"""
        return list(self._signals)

    @property
    def name(self) -> str:
        """策略显示名称，子类可覆盖"""
        return self.__class__.__name__

    def __repr__(self) -> str:
        return (
            f"{self.name}(id={self.strategy_id!r}, "
            f"enabled={self.config.enabled}, "
            f"universe={self.config.universe})"
        )
