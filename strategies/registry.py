"""
registry.py
-----------
职责：策略注册表，支持按名称动态实例化策略。

使用方式：
    registry = StrategyRegistry()
    registry.register("vol_arb", VolArbStrategy)
    strategy = registry.create("vol_arb", config=my_config)
"""

from __future__ import annotations

import logging
from typing import Type

from .base import BaseStrategy, StrategyConfig

logger = logging.getLogger(__name__)


class StrategyRegistry:
    """
    策略注册表。

    支持按字符串名称注册和实例化策略类，
    便于从配置文件动态加载策略。

    Examples
    --------
    >>> registry = StrategyRegistry()
    >>> registry.register("vol_arb", VolArbStrategy)
    >>> s = registry.create("vol_arb", config=StrategyConfig(strategy_id="vol_arb_IO"))
    """

    def __init__(self) -> None:
        self._registry: dict[str, Type[BaseStrategy]] = {}

    def register(self, name: str, cls: Type[BaseStrategy]) -> None:
        """
        注册策略类。

        Parameters
        ----------
        name : str
            策略注册名称（唯一标识，如 'vol_arb'）
        cls : Type[BaseStrategy]
            策略类（必须继承 BaseStrategy）

        Raises
        ------
        ValueError
            如果 name 已注册且 cls 不同
        TypeError
            如果 cls 不是 BaseStrategy 的子类
        """
        if not (isinstance(cls, type) and issubclass(cls, BaseStrategy)):
            raise TypeError(f"{cls} 必须是 BaseStrategy 的子类")
        if name in self._registry and self._registry[name] is not cls:
            raise ValueError(
                f"策略名称 '{name}' 已注册为 {self._registry[name].__name__}，"
                f"请使用不同的名称或先调用 unregister()"
            )
        self._registry[name] = cls
        logger.debug("注册策略: %s -> %s", name, cls.__name__)

    def unregister(self, name: str) -> None:
        """取消注册策略"""
        self._registry.pop(name, None)

    def create(self, name: str, config: StrategyConfig) -> BaseStrategy:
        """
        按名称实例化策略。

        Parameters
        ----------
        name : str
            策略注册名称
        config : StrategyConfig
            策略配置对象

        Returns
        -------
        BaseStrategy
            策略实例

        Raises
        ------
        KeyError
            如果 name 未注册
        """
        if name not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise KeyError(
                f"策略 '{name}' 未注册。已注册策略：{available or '（空）'}"
            )
        cls = self._registry[name]
        return cls(config)

    def list_strategies(self) -> list[str]:
        """返回所有已注册策略名称"""
        return sorted(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __repr__(self) -> str:
        names = ", ".join(self.list_strategies())
        return f"StrategyRegistry([{names}])"


# 全局注册表单例
_global_registry = StrategyRegistry()


def get_registry() -> StrategyRegistry:
    """获取全局策略注册表"""
    return _global_registry


# ---------------------------------------------------------------------------
# Register built-in strategies
# ---------------------------------------------------------------------------

def _register_builtin_strategies() -> None:
    """Register VolArbStrategy and TrendFollowingStrategy into the global registry."""
    try:
        from strategies.vol_arb.strategy import VolArbStrategy
        _global_registry.register("vol_arb", VolArbStrategy)
    except Exception as exc:
        logger.debug("Could not register vol_arb: %s", exc)

    try:
        from strategies.trend_following.strategy import TrendFollowingStrategy
        _global_registry.register("trend_following", TrendFollowingStrategy)
    except Exception as exc:
        logger.debug("Could not register trend_following: %s", exc)

    try:
        from strategies.intraday.strategy import IntradayStrategy
        _global_registry.register("intraday", IntradayStrategy)
    except Exception as exc:
        logger.debug("Could not register intraday: %s", exc)


_register_builtin_strategies()
