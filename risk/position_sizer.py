"""
position_sizer.py
-----------------
职责：根据信号强度和风控约束确定期权下单数量（仓位管理）。
核心方法：固定名义 Vega 仓位法（Fixed Vega Sizing）
- 每次开仓的目标净 Vega = 账户净值 × vega_target_pct / IV
- 实际手数 = 目标 Vega / 单合约 Vega / 合约乘数

同时支持：
- 固定手数法（简单但忽略波动率水平）
- 凯利公式（基于历史胜率和盈亏比，高级模式）
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

from config import Config
from strategies.vol_arb.signal_types import VolArbSignalStrength as SignalStrength, VRPSignal

logger = logging.getLogger(__name__)


class SizingMethod(str, Enum):
    """仓位计算方法"""
    FIXED_LOTS = "fixed_lots"         # 固定手数
    FIXED_VEGA = "fixed_vega"         # 固定 Vega 名义值（推荐）
    KELLY = "kelly"                    # 凯利公式


class PositionSizer:
    """
    仓位计算器。

    Parameters
    ----------
    config : Config
        系统配置
    method : SizingMethod
        仓位计算方法，默认固定 Vega
    """

    def __init__(
        self,
        config: Config,
        method: SizingMethod = SizingMethod.FIXED_VEGA,
    ) -> None:
        self.config = config
        self.method = method

    # ------------------------------------------------------------------
    # 主计算入口
    # ------------------------------------------------------------------

    def calc_position_size(
        self,
        signal: VRPSignal,
        account_balance: float,
        single_contract_vega: float,
        current_vega_exposure: float,
        contract_multiplier: int = 100,
    ) -> int:
        """
        计算建议的新开仓手数。

        Parameters
        ----------
        signal : VRPSignal
            VRP 信号（含信号强度）
        account_balance : float
            账户总权益（元）
        single_contract_vega : float
            单合约 Vega（点位/1%波动率，正值）
        current_vega_exposure : float
            当前已有的净 Vega 敞口（元/1%波动率）
        contract_multiplier : int
            合约乘数，IO/MO = 100

        Returns
        -------
        int
            建议开仓手数（已取整，最小为 0）

        Notes
        -----
        - 信号强度调整系数：STRONG=1.0, MODERATE=0.7, WEAK=0.4
        - 最终手数受 max_vega_exposure 约束
        - 结果向下取整（保守）
        """
        raise NotImplementedError("TODO: 实现仓位计算主逻辑")

    # ------------------------------------------------------------------
    # 各方法实现
    # ------------------------------------------------------------------

    def _fixed_vega_size(
        self,
        account_balance: float,
        single_contract_vega: float,
        strength_multiplier: float,
        current_vega_exposure: float,
        contract_multiplier: int,
    ) -> int:
        """
        固定 Vega 名义值仓位法。

        目标 Vega 名义值（元）= account_balance × vega_target_pct × strength_multiplier
        手数 = (目标 Vega - 当前 Vega) / (single_contract_vega × contract_multiplier)

        Notes
        -----
        vega_target_pct 从配置中读取（未来可添加到 config.yaml）
        默认约为 0.5%（即账户净值的 0.5% 对应每1%波动率变动的盈亏）
        """
        raise NotImplementedError("TODO: 实现固定 Vega 仓位计算")

    def _fixed_lots_size(self, signal: VRPSignal) -> int:
        """
        固定手数法（简单基准）。
        STRONG=4手, MODERATE=2手, WEAK=1手
        """
        raise NotImplementedError("TODO: 实现固定手数法")

    def _kelly_size(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        account_balance: float,
        single_contract_value: float,
    ) -> int:
        """
        凯利公式仓位法（保守版：使用半凯利）。

        Parameters
        ----------
        win_rate : float
            历史胜率（0~1）
        avg_win : float
            平均盈利（元/手）
        avg_loss : float
            平均亏损（元/手，正值）
        account_balance : float
            账户权益
        single_contract_value : float
            单合约名义价值

        Returns
        -------
        int
            建议手数

        Notes
        -----
        Kelly f = (p × b - q) / b，其中 b = avg_win/avg_loss，q = 1-p
        实际使用 f/2（半凯利）以控制波动
        """
        raise NotImplementedError("TODO: 实现凯利公式仓位")

    @staticmethod
    def strength_to_multiplier(strength: SignalStrength) -> float:
        """将信号强度转换为仓位系数"""
        return {
            SignalStrength.STRONG: 1.0,
            SignalStrength.MODERATE: 0.7,
            SignalStrength.WEAK: 0.4,
        }[strength]
