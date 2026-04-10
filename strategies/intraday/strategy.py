"""
strategy.py
-----------
日内/隔日期货方向性交易策略主控。

继承 BaseStrategy，核心循环在 on_bar() 中按5分钟K线驱动。
每根K线流程：
  1. 止损检查
  2. 信号生成（IF/IH/IM 各自评分）
  3. 按强度排序 → 风控检查 → 开仓
  4. 移动止盈
  5. 尾盘平仓
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from strategies.base import (
    BaseStrategy, Signal, SignalDirection, SignalStrength, StrategyConfig,
)
from strategies.intraday.signal import IntradaySignal
from strategies.intraday.A_share_momentum_signal_v2 import SignalGeneratorV2 as IntradaySignalGenerator
from strategies.intraday.position import IntradayPositionManager
from strategies.intraday.risk import IntradayRiskManager

logger = logging.getLogger(__name__)


@dataclass
class IntradayConfig(StrategyConfig):
    """日内策略配置"""
    strategy_id: str = "intraday"
    universe: list = field(default_factory=lambda: ["IF", "IH", "IM", "IC"])
    max_position: int = 2

    # 信号参数
    opening_range_minutes: int = 30
    vwap_deviation_threshold: float = 0.002
    volume_surge_ratio: float = 1.5
    trend_fast_period: int = 10
    trend_slow_period: int = 30
    min_signal_score: int = 50

    # 仓位参数
    max_lots_per_symbol: int = 1
    max_total_lots: int = 2
    stop_loss_bps: int = 100
    use_lock: bool = True

    # 可交易品种（只有这些品种会开仓占用position_mgr槽位）
    # 非交易品种仍然生成评分供面板显示，但不开仓
    tradeable: set = field(default_factory=lambda: {"IM", "IC"})

    # 移动止盈
    trailing_stop_levels: list = field(default_factory=lambda: [
        {"profit_bps": 50, "lock_bps": 0},
        {"profit_bps": 80, "lock_bps": 50},
        {"profit_bps": 120, "lock_bps": 80},
    ])

    # 风控
    max_daily_loss: float = 50_000
    max_daily_trades_per_symbol: int = 5
    max_consecutive_losses: int = 3

    # 手续费（按成交额万分比）
    commission_open_rate: float = 0.000023
    commission_intraday_close_rate: float = 0.000345
    commission_overnight_close_rate: float = 0.000023


class IntradayStrategy(BaseStrategy):
    """日内/隔日期货方向性交易策略。"""

    def __init__(self, config: IntradayConfig | None = None):
        if config is None:
            config = IntradayConfig()
        super().__init__(config)
        self.intraday_config: IntradayConfig = config

        sig_cfg = {
            "opening_range_minutes": config.opening_range_minutes,
            "vwap_deviation_threshold": config.vwap_deviation_threshold,
            "volume_surge_ratio": config.volume_surge_ratio,
            "trend_fast_period": config.trend_fast_period,
            "trend_slow_period": config.trend_slow_period,
            "min_signal_score": config.min_signal_score,
        }
        self.signal_gen = IntradaySignalGenerator(sig_cfg)

        pos_cfg = {
            "max_lots_per_symbol": config.max_lots_per_symbol,
            "max_total_lots": config.max_total_lots,
            "stop_loss_bps": config.stop_loss_bps,
            "use_lock": config.use_lock,
        }
        self.position_mgr = IntradayPositionManager(pos_cfg)

        risk_cfg = {
            "max_daily_loss": config.max_daily_loss,
            "max_daily_trades_per_symbol": config.max_daily_trades_per_symbol,
            "max_consecutive_losses": config.max_consecutive_losses,
            "stop_loss_bps": config.stop_loss_bps,
        }
        self.risk_mgr = IntradayRiskManager(risk_cfg)

        self.symbols = list(config.universe)
        self.tradeable = set(config.tradeable)

    # ------------------------------------------------------------------
    # BaseStrategy 接口（日线级别 — 日内策略主要用 on_bar）
    # ------------------------------------------------------------------

    def generate_signals(
        self, trade_date: str, market_data: dict
    ) -> list[Signal]:
        return []

    def on_fill(self, order_id, instrument, direction, volume, price, trade_date):
        pass

    # ------------------------------------------------------------------
    # 核心：每根K线调用
    # ------------------------------------------------------------------

    def on_bar(
        self,
        bar_data: Dict[str, pd.DataFrame],
        bar_15m: Dict[str, pd.DataFrame] | None,
        daily_data: Dict[str, pd.DataFrame] | None,
        current_time: str,
        next_trade_date: str = "",
        quote_data: Dict[str, Dict] | None = None,
        zscore_params: Dict | None = None,
        is_high_vol: bool = False,
        sentiment=None,
        d_override: Dict[str, float] | None = None,
        vol_profiles: Dict | None = None,
    ) -> List[Dict]:
        """
        每根5分钟K线调用一次。

        Args:
            bar_data: {symbol: 当日5m bars DataFrame (up to current bar)}
            bar_15m:  {symbol: 当日15m bars DataFrame}
            daily_data: {symbol: 最近日线 DataFrame}
            current_time: 当前bar时间戳 (UTC)
            next_trade_date: 下一交易日日期 YYYYMMDD
            quote_data: {symbol: {bid_price1, ask_price1, ...}}
            zscore_params: {symbol: {ema20, std20}}
            is_high_vol: GARCH regime flag
            sentiment: SentimentData
        Returns:
            操作列表
        """
        actions: List[Dict] = []
        time_str = current_time.split(" ")[-1][:5]

        # 当前价格（用现货close，与entry_price/stop_loss同源）
        # strategy层的position_mgr在monitor中只做占位管理，
        # 实际止损/PnL由monitor的shadow系统用期货价格处理
        prices: Dict[str, float] = {}
        for sym in self.symbols:
            if sym in bar_data and len(bar_data[sym]) > 0:
                prices[sym] = float(bar_data[sym].iloc[-1]["close"])

        # 1. 止损检查
        stop_triggers = self.position_mgr.check_stop_loss(prices)
        for pid, trigger_price in stop_triggers:
            result = self.position_mgr.close_position(
                pid, trigger_price, "STOP_LOSS",
                use_lock=True, current_time=current_time,
                next_trade_date=next_trade_date,
            )
            if result:
                self.risk_mgr.on_trade_complete(result["pnl"], result["symbol"])
                actions.append(result)

        # 2. 信号生成（传入zscore/is_high_vol/sentiment）
        signals: List[IntradaySignal] = []
        for sym in self.symbols:
            if sym not in bar_data or len(bar_data[sym]) < 2:
                continue
            b15 = bar_15m.get(sym) if bar_15m else None
            daily = daily_data.get(sym) if daily_data else None
            qd = quote_data.get(sym) if quote_data else None

            # Compute Z-Score for this symbol
            z_val = None
            zp = zscore_params.get(sym) if zscore_params else None
            if zp and sym in prices and zp.get("std20", 0) > 0:
                z_val = (prices[sym] - zp["ema20"]) / zp["std20"]

            _vp = vol_profiles.get(sym) if vol_profiles else None
            sig = self.signal_gen.update(
                sym, bar_data[sym], b15, daily, qd,
                sentiment=sentiment,
                zscore=z_val,
                is_high_vol=is_high_vol,
                d_override=d_override,
                vol_profile=_vp,
            )
            if sig:
                signals.append(sig)

        # 3. 按强度排序
        signals.sort(key=lambda s: s.score, reverse=True)

        # 4. 逐信号处理（只有tradeable品种进入position_mgr开仓）
        from strategies.intraday.A_share_momentum_signal_v2 import SYMBOL_PROFILES, _DEFAULT_PROFILE
        for sig in signals:
            if sig.symbol not in self.tradeable:
                continue
            # Per-symbol threshold（与backtest一致）
            _thr = SYMBOL_PROFILES.get(sig.symbol, _DEFAULT_PROFILE).get("signal_threshold", 60)
            if sig.score < _thr:
                continue
            allowed, reason = self.risk_mgr.check_pre_trade(
                sig, self.position_mgr
            )
            if not allowed:
                logger.debug(f"[STRATEGY] {sig.symbol} BLOCKED by risk_mgr: {reason}")
                continue

            # 矛盾持仓处理
            net = self.position_mgr._net_position_for(sig.symbol)
            if (net > 0 and sig.direction == "SHORT") or \
               (net < 0 and sig.direction == "LONG"):
                if sig.score >= 70:
                    for pid, pos in list(self.position_mgr.positions.items()):
                        if pos.symbol == sig.symbol and not pos.is_lock:
                            result = self.position_mgr.close_position(
                                pid, prices.get(sig.symbol, sig.entry_price),
                                "SIGNAL_FLIP", use_lock=True,
                                current_time=current_time,
                                next_trade_date=next_trade_date,
                            )
                            if result:
                                self.risk_mgr.on_trade_complete(
                                    result["pnl"], result["symbol"]
                                )
                                actions.append(result)
                            break
                else:
                    continue

            # 开仓
            pos = self.position_mgr.open_position(sig, sig.entry_price)
            if pos:
                actions.append({
                    "action": "OPEN",
                    "symbol": sig.symbol,
                    "direction": sig.direction,
                    "price": sig.entry_price,
                    "score": sig.score,
                    "signal_type": sig.signal_type,
                    "stop_loss": sig.stop_loss,
                    "reason": sig.reason,
                })

        # 5. 移动止盈
        self.position_mgr.update_trailing_stop(
            prices, self.intraday_config.trailing_stop_levels
        )

        # 6. 尾盘平仓
        try:
            weekday = pd.Timestamp(current_time).weekday()
        except Exception:
            weekday = 0

        eod_closes = self.position_mgr.check_eod_close(
            time_str, weekday, prices, next_trade_date
        )
        for pid, reason, use_lock in eod_closes:
            pos = self.position_mgr.positions.get(pid)
            if pos is None:
                continue
            price = prices.get(pos.symbol)
            if price is None:
                continue
            result = self.position_mgr.close_position(
                pid, price, reason, use_lock=use_lock,
                current_time=current_time,
                next_trade_date=next_trade_date,
            )
            if result:
                self.risk_mgr.on_trade_complete(result["pnl"], result["symbol"])
                actions.append(result)

        return actions

    # ------------------------------------------------------------------
    # 每日开盘
    # ------------------------------------------------------------------

    def on_daily_open(
        self, trade_date: str, prices: Dict[str, float]
    ) -> List[Dict]:
        unlock_actions = self.position_mgr.process_daily_open(
            trade_date, prices
        )
        self.position_mgr.reset_daily()
        self.risk_mgr.reset_daily()
        return unlock_actions

    # ------------------------------------------------------------------
    # 信号格式化
    # ------------------------------------------------------------------

    def format_signal_alert(self, signal: IntradaySignal) -> str:
        comp = signal.components
        try:
            bj_dt = pd.Timestamp(signal.datetime) + pd.Timedelta(hours=8)
            display_time = bj_dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            display_time = signal.datetime

        dir_cn = "做多" if signal.direction == "LONG" else "做空"
        hold_cn = "日内" if signal.signal_type == "INTRADAY" else "隔日"

        mult = self.position_mgr.contract_multipliers.get(signal.symbol, 300)
        sl_dist = abs(signal.entry_price - signal.stop_loss)
        sl_pct = sl_dist / signal.entry_price * 100 if signal.entry_price else 0
        sl_amt = sl_dist * mult

        dim_names = {
            "opening_breakout": ("开盘突破", 25),
            "vwap": ("VWAP", 21),
            "multi_tf": ("多周期", 16),
            "volume": ("成交量", 11),
            "daily_levels": ("日线位置", 7),
            "orderbook": ("盘口", 8),
            "bollinger": ("布林带", 12),
        }

        lines = [
            "═" * 50,
            f"  交易信号 | {signal.symbol} | {dir_cn} | 强度 {signal.score}/100",
            "═" * 50,
            f"  时间: {display_time}",
            f"  入场: {signal.entry_price:.1f}",
            f"  止损: {signal.stop_loss:.1f}（-{sl_pct:.1f}%, -{sl_amt:,.0f}元）",
            f"  类型: {hold_cn}",
            "",
            "  信号明细：",
        ]
        for key, (name, mx) in dim_names.items():
            if key in comp:
                lines.append(f"    {name:<8}: +{comp[key]['score']}/{mx}")
        lines.append(f"    {'总分':<8}: {signal.score}/100")
        lines.append("═" * 50)
        return "\n".join(lines)
