"""
signal_types.py
---------------
职责：定义波动率套利策略特有的信号数据结构。

从原 signals/signal_types.py 迁移，扩展了 vol_arb 特有字段。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class VolArbSignalDirection(str, Enum):
    """波动率套利信号方向"""
    SHORT_VOL = "short_vol"    # 做空波动率（卖出期权）
    LONG_VOL = "long_vol"      # 做多波动率（买入期权）
    NEUTRAL = "neutral"         # 中性/无操作


class VolArbSignalStrength(str, Enum):
    """信号强度等级"""
    STRONG = "strong"          # 强信号（VRP 远超阈值）
    MODERATE = "moderate"      # 中等信号
    WEAK = "weak"              # 弱信号（刚过阈值）


@dataclass
class VRPSignal:
    """
    波动率风险溢价（VRP）交易信号。

    Attributes
    ----------
    signal_date : str
        信号生成日期，格式 YYYYMMDD
    underlying : str
        标的品种，如 IO / MO
    direction : VolArbSignalDirection
        信号方向
    strength : VolArbSignalStrength
        信号强度
    garch_vol : float
        GARCH 预测波动率（年化小数）
    atm_iv : float
        近月 ATM 隐含波动率（年化小数）
    vrp : float
        波动率风险溢价 = (atm_iv - garch_vol) / garch_vol
    vrp_threshold : float
        触发信号的 VRP 阈值
    recommended_expire : str
        推荐到期日（YYYYMMDD）
    recommended_strikes : list[float]
        推荐行权价列表（如 Strangle 的两个行权价）
    target_delta : float
        目标组合 Delta 敞口（通常为 0，表示 Delta 中性）
    notes : str
        备注信息
    """
    signal_date: str
    underlying: str
    direction: VolArbSignalDirection
    strength: VolArbSignalStrength
    garch_vol: float
    atm_iv: float
    vrp: float
    vrp_threshold: float
    recommended_expire: str = ""
    recommended_strikes: list[float] = field(default_factory=list)
    target_delta: float = 0.0
    notes: str = ""

    @property
    def is_actionable(self) -> bool:
        """信号是否应触发实际交易（排除 NEUTRAL）"""
        return self.direction != VolArbSignalDirection.NEUTRAL

    def to_dict(self) -> dict:
        """转换为字典，用于日志记录和数据库存储"""
        return {
            "signal_date": self.signal_date,
            "underlying": self.underlying,
            "direction": self.direction.value,
            "strength": self.strength.value,
            "garch_vol": round(self.garch_vol, 6),
            "atm_iv": round(self.atm_iv, 6),
            "vrp": round(self.vrp, 6),
            "vrp_threshold": self.vrp_threshold,
            "recommended_expire": self.recommended_expire,
            "recommended_strikes": self.recommended_strikes,
            "target_delta": self.target_delta,
            "notes": self.notes,
        }


@dataclass
class RollSignal:
    """持仓滚仓信号（临近到期时生成）"""
    signal_date: str
    underlying: str
    current_expire: str       # 当前持仓到期日
    target_expire: str        # 目标展期到期日
    days_to_expire: int       # 距离到期的交易日数
    reason: str = ""          # 滚仓原因
