"""
strategy.py — 波动率套利策略 (Iron Condor)
卖出 Iron Condor: 卖OTM Put/Call + 买更虚值 Put/Call 保护翼。
最大亏损 = wing_width × multiplier - net_premium，结构性锁定，无隔夜跳空风险。

重构版本：适度信号中频交易
- 固定VRP阈值 1%（不再使用动态百分位）
- 最长持仓10个交易日
- 阶梯止盈（trailing stop）
- 更紧止损：1.5×权利金/condor, 2% daily
- 多品种：MO（中证1000）+ IO（沪深300）
- 小仓位：2-3手/condor，允许2-3个同时持仓
- 3天冷却期
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from strategies.base import BaseStrategy, Signal, StrategyConfig, SignalDirection, SignalStrength
from strategies.vol_arb.signal import VRPSignalGenerator

logger = logging.getLogger(__name__)


@dataclass
class VolArbConfig(StrategyConfig):
    # --- Entry ---
    vrp_entry_threshold: float = 0.015   # 1.5% VRP threshold
    vrp_exit_threshold: float = -1.0     # disabled; hold to expiry/take-profit
    put_delta: float = -0.20             # tighter deltas → more premium
    call_delta: float = 0.20
    wing_width: int = 300
    min_days_to_expire: int = 30
    max_days_to_expire: int = 60
    preferred_dte_min: int = 35
    preferred_dte_max: int = 55
    roll_days_before_expiry: int = 5
    garch_lookback: int = 504
    garch_refit_days: int = 20
    max_vega_exposure: float = 8000.0

    # --- Risk / Exit ---
    take_profit_pct: float = 0.60        # close at 60% captured
    trailing_stage1_pct: float = 99.0   # disabled (set impossibly high)
    trailing_stage2_pct: float = 99.0   # disabled
    trailing_stage2_lock: float = 0.0
    daily_loss_limit_pct: float = 0.025
    condor_stop_loss_mult: float = 3.0   # per-condor stop: 3× premium
    max_holding_days: int = 10           # 10 trading days sweet spot

    # --- Position sizing ---
    max_condors_per_expiry: int = 2
    max_condors_total: int = 3
    margin_utilization_limit: float = 0.40
    cooldown_days: int = 3
    max_loss_pct_of_equity: float = 0.05
    lots_min: int = 4
    lots_max: int = 5

    # --- Regime filter ---
    use_regime_filter: bool = True
    regime_rv_short: int = 20
    regime_rv_long: int = 60
    regime_rv_ratio: float = 1.3
    regime_big_move_pct: float = 0.03    # 3% daily move blocks entry

    # --- Dynamic wing width ---
    use_dynamic_wing: bool = True
    wing_atr_multiplier: float = 2.0
    wing_min: int = 200
    wing_max: int = 400

    # --- Multi-product ---
    # Products config: list of (option_prefix, garch_training_sym, multiplier)
    # Set at runtime based on available data


class VolArbStrategy(BaseStrategy):
    """
    Volatility arbitrage: sell Iron Condor when VRP > threshold.

    中频版本：
    - VRP > 1% 即可入场（固定阈值）
    - 允许GARCH降级到EWMA仍可入场
    - 阶梯止盈 + 最长10天持仓
    - 多品种（MO + IO）
    - 2-3手小仓位，允许2-3个同时condor
    """

    def __init__(self, config: VolArbConfig, db_manager=None) -> None:
        super().__init__(config)
        self.vrp_config = config
        self.db = db_manager

        # Per-product VRP generators
        self._vrp_generators: Dict[str, VRPSignalGenerator] = {}
        self._default_vrp_gen = VRPSignalGenerator(
            garch_lookback=config.garch_lookback,
            forecast_horizon=5,
        )

        self._open_positions: Dict[str, dict] = {}
        self._positions: Dict[str, dict] = {}  # engine injects __equity__ here
        self._condor_groups: List[dict] = []
        self._daily_loss_triggered: bool = False
        self._cooldown_until: str = ""

        # Attribution
        self.trade_log: List[dict] = []
        self._vrp_history: List[float] = []
        self._daily_vrp_log: List[dict] = []
        self._last_entry_date: str = ""

        # Trading day counter for max_holding_days
        self._trading_day_count: int = 0

    # ------------------------------------------------------------------
    # VRP generator per product
    # ------------------------------------------------------------------

    def _get_vrp_gen(self, prefix: str) -> VRPSignalGenerator:
        if prefix not in self._vrp_generators:
            self._vrp_generators[prefix] = VRPSignalGenerator(
                garch_lookback=self.vrp_config.garch_lookback,
                forecast_horizon=5,
            )
        return self._vrp_generators[prefix]

    # ------------------------------------------------------------------
    # Risk helpers
    # ------------------------------------------------------------------

    def _calc_condor_pnl(self, cg: dict, chain_prices: dict) -> float:
        """P&L for a single condor group."""
        multiplier = 100
        sp_cur = chain_prices.get(cg["sell_put_sym"], cg["sell_put_entry_px"])
        bp_cur = chain_prices.get(cg["buy_put_sym"], cg["buy_put_entry_px"])
        sc_cur = chain_prices.get(cg["sell_call_sym"], cg["sell_call_entry_px"])
        bc_cur = chain_prices.get(cg["buy_call_sym"], cg["buy_call_entry_px"])
        entry_net = cg["net_premium_per_lot"]
        current_net = (sp_cur + sc_cur) - (bp_cur + bc_cur)
        return (entry_net - current_net) * cg["lots"] * multiplier

    def _calc_total_unrealized(self, chain_prices: dict) -> float:
        """Total unrealized P&L across all condor groups."""
        return sum(self._calc_condor_pnl(cg, chain_prices) for cg in self._condor_groups)

    def _calc_profit_ratio(self, cg: dict, chain_prices: dict) -> float:
        """Profit ratio = captured / initial premium. >0 = profitable."""
        sp_cur = chain_prices.get(cg["sell_put_sym"], cg["sell_put_entry_px"])
        bp_cur = chain_prices.get(cg["buy_put_sym"], cg["buy_put_entry_px"])
        sc_cur = chain_prices.get(cg["sell_call_sym"], cg["sell_call_entry_px"])
        bc_cur = chain_prices.get(cg["buy_call_sym"], cg["buy_call_entry_px"])
        entry_net = cg["net_premium_per_lot"]
        if entry_net <= 0:
            return 0.0
        current_net = (sp_cur + sc_cur) - (bp_cur + bc_cur)
        return (entry_net - current_net) / entry_net

    def _estimate_vega_per_condor(self, spot: float, dte: int) -> float:
        multiplier = 100
        T = max(dte, 1) / 365.0
        vega_per_leg = spot * np.sqrt(T) * 0.399 * 0.01 * multiplier * 0.6
        naked_vega = 2 * vega_per_leg
        return naked_vega * 0.5

    def _current_total_vega(self, spot: float) -> float:
        return sum(
            self._estimate_vega_per_condor(spot, cg.get("dte", 30)) * cg["lots"]
            for cg in self._condor_groups
        )

    # ------------------------------------------------------------------
    # Regime filter
    # ------------------------------------------------------------------

    def _is_high_vol_regime(self, futures_df: pd.DataFrame) -> bool:
        if not self.vrp_config.use_regime_filter:
            return False
        close = futures_df["close"].dropna()
        if len(close) < self.vrp_config.regime_rv_long + 5:
            return False
        log_ret = np.log(close / close.shift(1)).dropna()

        # Condition 1: Short RV / Long RV ratio
        rv_short = float(log_ret.tail(self.vrp_config.regime_rv_short).std() * np.sqrt(252))
        rv_long = float(log_ret.tail(self.vrp_config.regime_rv_long).std() * np.sqrt(252))
        if rv_long > 0 and rv_short > rv_long * self.vrp_config.regime_rv_ratio:
            return True

        # Condition 2: Single day > 3% move (more permissive than consecutive check)
        daily_ret = close.pct_change().dropna()
        if len(daily_ret) > 0 and abs(float(daily_ret.iloc[-1])) > self.vrp_config.regime_big_move_pct:
            return True

        return False

    def _calc_dynamic_wing_width(self, futures_df: pd.DataFrame) -> int:
        if not self.vrp_config.use_dynamic_wing:
            return self.vrp_config.wing_width
        close = futures_df["close"].dropna()
        if len(close) < 21:
            return self.vrp_config.wing_width
        if "high" in futures_df.columns and "low" in futures_df.columns:
            high = futures_df["high"].dropna().tail(20)
            low = futures_df["low"].dropna().tail(20)
            if len(high) >= 20 and len(low) >= 20:
                atr = float((high - low).mean())
            else:
                atr = float(close.diff().abs().tail(20).mean()) * 2
        else:
            atr = float(close.diff().abs().tail(20).mean()) * 2
        wing = int(atr * self.vrp_config.wing_atr_multiplier)
        wing = round(wing / 100) * 100
        return max(self.vrp_config.wing_min, min(self.vrp_config.wing_max, wing))

    # ------------------------------------------------------------------
    # Signal helpers
    # ------------------------------------------------------------------

    def _make_close_signal(
        self, trade_date: str, sym: str, price: float, volume: int, reason: str,
    ) -> Signal:
        return Signal(
            strategy_id=self.strategy_id,
            signal_date=trade_date,
            instrument=sym,
            direction=SignalDirection.CLOSE,
            strength=SignalStrength.STRONG,
            target_volume=volume,
            price_ref=price,
            notes=f"RiskClose({reason})",
            metadata={
                "reason": reason,
                "contract_multiplier": 100,
                "margin_rate": 0.12,
            },
        )

    def _close_condor_signals(
        self, trade_date: str, chain_prices: dict, cg: dict,
        volume: int, reason: str,
    ) -> List[Signal]:
        signals = []
        for key in ("sell_put_sym", "buy_put_sym", "sell_call_sym", "buy_call_sym"):
            sym = cg[key]
            entry_key = key.replace("_sym", "_entry_px")
            px = chain_prices.get(sym, cg[entry_key])
            signals.append(self._make_close_signal(trade_date, sym, px, volume, reason))
        self._log_condor_close(trade_date, chain_prices, cg, reason)
        return signals

    def _remove_condor_positions(self, cg: dict) -> None:
        for key in ("sell_put_sym", "buy_put_sym", "sell_call_sym", "buy_call_sym"):
            self._open_positions.pop(cg[key], None)

    def _log_condor_close(
        self, trade_date: str, chain_prices: dict, cg: dict, reason: str,
    ) -> None:
        multiplier = 100
        sp_cur = chain_prices.get(cg["sell_put_sym"], cg["sell_put_entry_px"])
        bp_cur = chain_prices.get(cg["buy_put_sym"], cg["buy_put_entry_px"])
        sc_cur = chain_prices.get(cg["sell_call_sym"], cg["sell_call_entry_px"])
        bc_cur = chain_prices.get(cg["buy_call_sym"], cg["buy_call_entry_px"])
        entry_net = cg["net_premium_per_lot"]
        exit_net = (sp_cur + sc_cur) - (bp_cur + bc_cur)
        pnl_per_lot = (entry_net - exit_net) * multiplier
        total_pnl = pnl_per_lot * cg["lots"]
        try:
            holding_days = (pd.Timestamp(trade_date) - pd.Timestamp(cg["entry_date"])).days
        except Exception:
            holding_days = 0
        self.trade_log.append({
            "entry_date": cg["entry_date"],
            "exit_date": trade_date,
            "expire_date": cg["expire_date"],
            "product": cg.get("product", "MO"),
            "reason": reason,
            "lots": cg["lots"],
            "entry_vrp": cg.get("entry_vrp", 0),
            "entry_iv": cg.get("entry_iv", 0),
            "entry_garch_level": cg.get("entry_garch_level", ""),
            "entry_dte": cg.get("entry_dte", 0),
            "net_premium_per_lot": entry_net,
            "max_loss_per_lot": cg.get("max_loss_per_lot", 0),
            "pnl_per_lot": pnl_per_lot,
            "total_pnl": total_pnl,
            "holding_days": holding_days,
            "holding_trade_days": cg.get("holding_trade_days", 0),
            "wing_width": cg.get("wing_width", 400),
        })

    # ------------------------------------------------------------------
    # Expiry settlement
    # ------------------------------------------------------------------

    def _settle_expired_condors(
        self, trade_date: str, spot_price: float, chain_prices: dict,
    ) -> List[Signal]:
        signals: List[Signal] = []
        remaining: List[dict] = []

        for cg in self._condor_groups:
            if cg["dte"] > 0:
                remaining.append(cg)
                continue

            K1 = cg["sell_put_strike"]
            K2 = cg["buy_put_strike"]
            K3 = cg["sell_call_strike"]
            K4 = cg["buy_call_strike"]
            S = spot_price

            settle_prices = {
                cg["sell_put_sym"]: max(K1 - S, 0),
                cg["buy_put_sym"]: max(K2 - S, 0),
                cg["sell_call_sym"]: max(S - K3, 0),
                cg["buy_call_sym"]: max(S - K4, 0),
            }

            logger.info(
                "EXPIRY SETTLE %s: spot=%.0f  K=[%.0f,%.0f,%.0f,%.0f]  "
                "settle=[%.1f,%.1f,%.1f,%.1f]",
                cg["expire_date"], S, K1, K2, K3, K4,
                settle_prices[cg["sell_put_sym"]], settle_prices[cg["buy_put_sym"]],
                settle_prices[cg["sell_call_sym"]], settle_prices[cg["buy_call_sym"]],
            )

            for sym, px in settle_prices.items():
                chain_prices[sym] = px

            for key in ("sell_put_sym", "buy_put_sym", "sell_call_sym", "buy_call_sym"):
                sym = cg[key]
                px = settle_prices[sym]
                signals.append(self._make_close_signal(
                    trade_date, sym, px, cg["lots"], "expiry_settle",
                ))

            self._log_condor_close(trade_date, chain_prices, cg, "expiry_settle")
            self._remove_condor_positions(cg)

        self._condor_groups = remaining
        return signals

    # ------------------------------------------------------------------
    # Risk checks
    # ------------------------------------------------------------------

    def _risk_daily_loss(
        self, trade_date: str, chain_prices: dict, account_equity: float,
    ) -> List[Signal]:
        """Close ALL condors if total unrealized loss > daily_loss_limit_pct of equity."""
        if not self._condor_groups:
            return []
        total_pnl = self._calc_total_unrealized(chain_prices)
        limit = -self.vrp_config.daily_loss_limit_pct * account_equity
        if total_pnl >= limit:
            return []

        logger.warning(
            "DAILY LOSS LIMIT: unrealized=%.0f limit=%.0f (%.1f%% of %.0f)",
            total_pnl, limit, self.vrp_config.daily_loss_limit_pct * 100, account_equity,
        )
        signals: List[Signal] = []
        for cg in self._condor_groups:
            signals += self._close_condor_signals(
                trade_date, chain_prices, cg, cg["lots"], "daily_loss_limit",
            )
        self._condor_groups.clear()
        self._open_positions.clear()
        self._daily_loss_triggered = True
        try:
            cd = pd.Timestamp(str(trade_date)) + pd.Timedelta(days=self.vrp_config.cooldown_days)
            self._cooldown_until = cd.strftime("%Y%m%d")
        except Exception:
            pass
        return signals

    def _risk_condor_stop_loss(self, trade_date: str, chain_prices: dict) -> List[Signal]:
        """Per-condor stop-loss: close if loss > condor_stop_loss_mult × premium.
        Set condor_stop_loss_mult=0 to disable."""
        if self.vrp_config.condor_stop_loss_mult <= 0:
            return []

        signals: List[Signal] = []
        remaining: List[dict] = []
        multiplier = 100

        for cg in self._condor_groups:
            pnl = self._calc_condor_pnl(cg, chain_prices)
            max_loss = -self.vrp_config.condor_stop_loss_mult * cg["net_premium_per_lot"] * cg["lots"] * multiplier
            if pnl < max_loss:
                logger.warning(
                    "CONDOR STOP LOSS: %s pnl=%.0f < limit=%.0f (%.1f× premium)",
                    cg["expire_date"], pnl, max_loss, self.vrp_config.condor_stop_loss_mult,
                )
                signals += self._close_condor_signals(
                    trade_date, chain_prices, cg, cg["lots"], "condor_stop_loss",
                )
                self._remove_condor_positions(cg)
            else:
                remaining.append(cg)

        self._condor_groups = remaining
        return signals

    def _risk_trailing_stop(self, trade_date: str, chain_prices: dict) -> List[Signal]:
        """Trailing stop-profit with 3 stages:
        Stage 1: profit >= 30% → move stop to breakeven (floor=0%)
        Stage 2: profit >= 50% → lock 30% profit (floor=30%)
        Stage 3: profit >= 70% → close immediately
        """
        signals: List[Signal] = []
        remaining: List[dict] = []

        for cg in self._condor_groups:
            profit_ratio = self._calc_profit_ratio(cg, chain_prices)

            # Stage 3: close at 70%
            if profit_ratio >= self.vrp_config.take_profit_pct:
                logger.info(
                    "TRAILING STAGE3: condor %s captured %.0f%%, closing",
                    cg["expire_date"], profit_ratio * 100,
                )
                signals += self._close_condor_signals(
                    trade_date, chain_prices, cg, cg["lots"], "take_profit",
                )
                self._remove_condor_positions(cg)
                continue

            # Update trailing stop floor
            current_floor = cg.get("trailing_floor", -999.0)

            if profit_ratio >= self.vrp_config.trailing_stage2_pct:
                # Stage 2: lock 30%
                new_floor = self.vrp_config.trailing_stage2_lock
                if new_floor > current_floor:
                    cg["trailing_floor"] = new_floor
                    logger.debug("TRAILING STAGE2: %s floor→%.0f%%", cg["expire_date"], new_floor * 100)
            elif profit_ratio >= self.vrp_config.trailing_stage1_pct:
                # Stage 1: breakeven
                new_floor = 0.0
                if new_floor > current_floor:
                    cg["trailing_floor"] = new_floor
                    logger.debug("TRAILING STAGE1: %s floor→breakeven", cg["expire_date"])

            # Check if profit dropped below trailing floor
            trailing_floor = cg.get("trailing_floor", -999.0)
            if trailing_floor > -999.0 and profit_ratio < trailing_floor:
                logger.info(
                    "TRAILING STOP: condor %s profit=%.0f%% < floor=%.0f%%, closing",
                    cg["expire_date"], profit_ratio * 100, trailing_floor * 100,
                )
                signals += self._close_condor_signals(
                    trade_date, chain_prices, cg, cg["lots"], "trailing_stop",
                )
                self._remove_condor_positions(cg)
                continue

            remaining.append(cg)

        self._condor_groups = remaining
        return signals

    def _risk_max_holding(self, trade_date: str, chain_prices: dict) -> List[Signal]:
        """Close condors held longer than max_holding_days trading days."""
        signals: List[Signal] = []
        remaining: List[dict] = []

        for cg in self._condor_groups:
            trade_days = cg.get("holding_trade_days", 0)
            if trade_days >= self.vrp_config.max_holding_days:
                logger.info(
                    "MAX HOLDING: condor %s held %d trading days, closing",
                    cg["expire_date"], trade_days,
                )
                signals += self._close_condor_signals(
                    trade_date, chain_prices, cg, cg["lots"], "max_holding",
                )
                self._remove_condor_positions(cg)
            else:
                remaining.append(cg)

        self._condor_groups = remaining
        return signals

    # ------------------------------------------------------------------
    # Main signal generation
    # ------------------------------------------------------------------

    def generate_signals(self, trade_date: str, market_data: dict) -> List[Signal]:
        signals: List[Signal] = []
        self._daily_loss_triggered = False
        self._trading_day_count += 1

        # --- Verify condor groups against broker positions (prune phantoms) ---
        broker_positions = self._positions.get("__broker_positions__", {})
        if broker_positions:
            verified: List[dict] = []
            for cg in self._condor_groups:
                legs = [cg["sell_put_sym"], cg["buy_put_sym"],
                        cg["sell_call_sym"], cg["buy_call_sym"]]
                if all(sym in broker_positions for sym in legs):
                    verified.append(cg)
                else:
                    missing = [s for s in legs if s not in broker_positions]
                    logger.debug("Pruning phantom condor %s: missing legs %s",
                                 cg["expire_date"], missing)
                    for key in ("sell_put_sym", "buy_put_sym",
                                "sell_call_sym", "buy_call_sym"):
                        self._open_positions.pop(cg[key], None)
            self._condor_groups = verified

        # --- Increment holding days for all condors ---
        for cg in self._condor_groups:
            cg["holding_trade_days"] = cg.get("holding_trade_days", 0) + 1

        # --- Resolve GARCH training data (prefer IC.CFX for longer history) ---
        garch_sym = None
        for candidate in ("IC.CFX", "IM.CFX", "IF.CFX"):
            if candidate in market_data and not market_data[candidate].empty:
                garch_sym = candidate
                break
        if garch_sym is None:
            return signals
        futures_df = market_data[garch_sym]
        if len(futures_df) < 50:
            return signals

        options_chain = market_data.get("options_chain", pd.DataFrame())

        # --- Compute VRP (using MO chain by default) ---
        # Pass futures close price for market IV calculation (avoids Forward-based circular reasoning)
        fut_close = float(futures_df["close"].iloc[-1])
        vrp_gen = self._default_vrp_gen
        vrp_result = vrp_gen.compute_vrp(futures_df, options_chain, trade_date, futures_price=fut_close)
        if not vrp_result:
            return signals
        vrp = vrp_result.get("vrp")
        if vrp is None:
            return signals

        atm_iv = vrp_result.get("atm_iv")
        garch_level = vrp_result.get("garch_level", "")
        spot_price = vrp_result.get("implied_forward", 0) or float(
            futures_df["close"].iloc[-1]
        )

        self._vrp_history.append(vrp)
        self._daily_vrp_log.append({
            "date": trade_date, "vrp": vrp, "atm_iv": atm_iv,
            "garch_level": garch_level,
            "garch_forecast": vrp_result.get("garch_forecast", 0),
        })

        # --- Build chain_prices lookup ---
        chain_prices: dict = {}
        if (
            options_chain is not None
            and not (isinstance(options_chain, pd.DataFrame) and options_chain.empty)
            and "ts_code" in options_chain.columns
            and "close" in options_chain.columns
        ):
            for _, row in options_chain[["ts_code", "close"]].iterrows():
                px = row["close"]
                if (
                    px is not None
                    and not (isinstance(px, float) and pd.isna(px))
                    and float(px) > 0
                ):
                    chain_prices[str(row["ts_code"])] = float(px)

        account_equity = self._positions.get("__equity__", {}).get("balance", 1_000_000)
        current_margin = self._positions.get("__equity__", {}).get("margin", 0)

        # --- Update DTE on all condor groups ---
        trade_dt = pd.Timestamp(str(trade_date))
        for cg in self._condor_groups:
            try:
                cg["dte"] = (pd.Timestamp(str(cg["expire_date"])) - trade_dt).days
            except Exception:
                pass

        # ============================================================
        # EXPIRY SETTLEMENT (highest priority)
        # ============================================================
        signals += self._settle_expired_condors(trade_date, spot_price, chain_prices)

        # ============================================================
        # RISK CHECKS (priority order)
        # ============================================================

        # 1. Daily loss limit (closes all)
        risk_signals = self._risk_daily_loss(trade_date, chain_prices, account_equity)
        if risk_signals:
            return signals + risk_signals

        # 2. Per-condor stop-loss (1.5× premium)
        signals += self._risk_condor_stop_loss(trade_date, chain_prices)

        # 3. Trailing stop-profit (staged)
        signals += self._risk_trailing_stop(trade_date, chain_prices)

        # 4. Max holding days
        signals += self._risk_max_holding(trade_date, chain_prices)

        # ============================================================
        # NORMAL EXITS (VRP reversal, DTE roll)
        # ============================================================
        remaining_condors: List[dict] = []
        for cg in self._condor_groups:
            dte = cg.get("dte", 999)
            exit_triggered = (
                vrp < self.vrp_config.vrp_exit_threshold
                or dte < self.vrp_config.roll_days_before_expiry
            )
            if exit_triggered:
                reason = "dte_roll" if dte < self.vrp_config.roll_days_before_expiry else "vrp_exit"
                signals += self._close_condor_signals(
                    trade_date, chain_prices, cg, cg["lots"], reason,
                )
                self._remove_condor_positions(cg)
            else:
                remaining_condors.append(cg)
        self._condor_groups = remaining_condors

        # ============================================================
        # ENTRY (attempt for each product with available options)
        # ============================================================
        if self._daily_loss_triggered:
            return signals

        # Cooldown check
        in_cooldown = self._cooldown_until and trade_date <= self._cooldown_until
        if in_cooldown:
            return signals

        # Regime filter
        high_vol_regime = self._is_high_vol_regime(futures_df)
        if high_vol_regime:
            return signals

        # VRP threshold check (fixed, no dynamic)
        if vrp <= self.vrp_config.vrp_entry_threshold:
            return signals

        # Can we add more condors?
        if len(self._condor_groups) >= self.vrp_config.max_condors_total:
            return signals

        # --- Try entry on MO (中证1000 options) ---
        entry_signals = self._try_entry(
            trade_date, options_chain, futures_df, vrp_result,
            account_equity, current_margin, spot_price,
            product="MO",
        )
        signals += entry_signals

        return signals

    # ------------------------------------------------------------------
    # Entry logic
    # ------------------------------------------------------------------

    def _try_entry(
        self,
        trade_date: str,
        options_chain: pd.DataFrame,
        futures_df: pd.DataFrame,
        vrp_result: dict,
        account_equity: float,
        current_margin: float,
        spot_price: float,
        product: str = "MO",
    ) -> List[Signal]:
        """Try to enter an Iron Condor for the given product."""
        signals: List[Signal] = []

        if len(self._condor_groups) >= self.vrp_config.max_condors_total:
            return signals

        vrp = vrp_result["vrp"]
        atm_iv = vrp_result.get("atm_iv")
        garch_level = vrp_result.get("garch_level", "")
        atm_expiry = vrp_result.get("atm_expiry", "")
        dte = vrp_result.get("days_to_expiry")

        # DTE check
        dte_ok = (
            dte is None
            or self.vrp_config.min_days_to_expire <= dte <= self.vrp_config.max_days_to_expire
        )
        has_chain = (
            options_chain is not None
            and not (isinstance(options_chain, pd.DataFrame) and options_chain.empty)
        )
        if not (dte_ok and has_chain):
            return signals

        # Max condors per expiry
        expiry_count = sum(1 for cg in self._condor_groups if cg["expire_date"] == atm_expiry)
        if expiry_count >= self.vrp_config.max_condors_per_expiry:
            return signals

        # Dynamic wing width
        dynamic_wing = self._calc_dynamic_wing_width(futures_df)

        # Select Iron Condor legs
        vrp_gen = self._get_vrp_gen(product) if product != "MO" else self._default_vrp_gen
        condor = vrp_gen.select_iron_condor_legs(
            options_chain,
            implied_forward=vrp_result.get("implied_forward", 0),
            expire_date=atm_expiry,
            put_delta_target=self.vrp_config.put_delta,
            call_delta_target=self.vrp_config.call_delta,
            wing_width=dynamic_wing,
        )
        if not condor:
            return signals

        sell_put = condor["sell_put"]
        buy_put = condor["buy_put"]
        sell_call = condor["sell_call"]
        buy_call = condor["buy_call"]

        sp_px = float(sell_put["close"])
        bp_px = float(buy_put["close"])
        sc_px = float(sell_call["close"])
        bc_px = float(buy_call["close"])

        net_premium = (sp_px + sc_px) - (bp_px + bc_px)
        if net_premium <= 0:
            return signals

        dte_val = dte or 30
        multiplier = 100

        put_wing = sell_put["exercise_price"] - buy_put["exercise_price"]
        call_wing = sell_call["exercise_price"] - buy_call["exercise_price"]
        actual_wing = max(put_wing, call_wing)
        max_loss_per_lot = actual_wing * multiplier - net_premium * multiplier
        if max_loss_per_lot <= 0:
            return signals

        # --- Position sizing: fixed 2-3 lots ---
        max_allowed_loss = account_equity * self.vrp_config.max_loss_pct_of_equity
        lots_by_risk = int(max_allowed_loss / max_loss_per_lot)

        spread_margin_per_lot = max_loss_per_lot
        avail_margin = account_equity * self.vrp_config.margin_utilization_limit - current_margin
        if avail_margin <= spread_margin_per_lot:
            return signals
        lots_by_margin = int(avail_margin / spread_margin_per_lot)

        # Vega limit
        cur_vega = self._current_total_vega(spot_price)
        remaining_vega = self.vrp_config.max_vega_exposure - cur_vega
        vega_per = self._estimate_vega_per_condor(spot_price, dte_val)
        lots_by_vega = int(remaining_vega / vega_per) if vega_per > 0 and remaining_vega > 0 else 0
        if lots_by_vega <= 0:
            return signals

        # Clamp to min/max lots
        lots = max(
            self.vrp_config.lots_min,
            min(lots_by_risk, lots_by_margin, lots_by_vega, self.vrp_config.lots_max),
        )
        # If even min lots exceed risk, still allow min
        if lots_by_risk < self.vrp_config.lots_min or lots_by_margin < self.vrp_config.lots_min:
            return signals

        strength = SignalStrength.STRONG if vrp > 0.03 else SignalStrength.MODERATE
        ic_margin_rate = spread_margin_per_lot / (spot_price * multiplier) if spot_price > 0 else 0.12

        logger.info(
            "IC ENTRY [%s]: K=[%.0f,%.0f,%.0f,%.0f] premium=%.1f max_loss/lot=%.0f "
            "lots=%d wing=%.0f vrp=%.3f",
            product,
            sell_put["exercise_price"], buy_put["exercise_price"],
            sell_call["exercise_price"], buy_call["exercise_price"],
            net_premium, max_loss_per_lot, lots, actual_wing, vrp,
        )

        # --- Emit 4 signals ---
        for leg_key, leg_info, direction in [
            ("sell_put", sell_put, SignalDirection.SHORT),
            ("sell_call", sell_call, SignalDirection.SHORT),
            ("buy_put", buy_put, SignalDirection.LONG),
            ("buy_call", buy_call, SignalDirection.LONG),
        ]:
            signals.append(Signal(
                strategy_id=self.strategy_id,
                signal_date=trade_date,
                instrument=leg_info["ts_code"],
                direction=direction,
                strength=strength,
                target_volume=lots,
                price_ref=float(leg_info["close"]),
                notes=f"IC {leg_key} VRP={vrp:.3f} lots={lots} [{product}]",
                metadata={
                    "vrp": vrp, "leg": leg_key,
                    "contract_multiplier": 100, "margin_rate": ic_margin_rate if "sell" in leg_key else 0.0,
                    "days_to_expiry": dte_val,
                },
            ))

        # Track positions
        for leg_key, leg_info in [
            ("sell_put", sell_put), ("buy_put", buy_put),
            ("sell_call", sell_call), ("buy_call", buy_call),
        ]:
            ts = leg_info["ts_code"]
            self._open_positions[ts] = {
                "days_to_expiry": dte_val,
                "expire_date": leg_info.get("expire_date", atm_expiry),
                "entry_date": trade_date,
                "lots": lots,
                "leg": leg_key,
                "product": product,
            }

        self._last_entry_date = trade_date

        self._condor_groups.append({
            "sell_put_sym": sell_put["ts_code"],
            "buy_put_sym": buy_put["ts_code"],
            "sell_call_sym": sell_call["ts_code"],
            "buy_call_sym": buy_call["ts_code"],
            "sell_put_strike": float(sell_put["exercise_price"]),
            "buy_put_strike": float(buy_put["exercise_price"]),
            "sell_call_strike": float(sell_call["exercise_price"]),
            "buy_call_strike": float(buy_call["exercise_price"]),
            "lots": lots,
            "sell_put_entry_px": sp_px,
            "buy_put_entry_px": bp_px,
            "sell_call_entry_px": sc_px,
            "buy_call_entry_px": bc_px,
            "net_premium_per_lot": net_premium,
            "max_loss_per_lot": max_loss_per_lot,
            "entry_iv": atm_iv or 0.0,
            "entry_vrp": vrp,
            "entry_garch_level": garch_level,
            "entry_dte": dte_val,
            "wing_width": actual_wing,
            "expire_date": atm_expiry,
            "entry_date": trade_date,
            "dte": dte_val,
            "product": product,
            "holding_trade_days": 0,
            "trailing_floor": -999.0,  # no trailing stop activated yet
        })

        return signals

    def on_fill(
        self,
        order_id: str,
        instrument: str,
        direction: str,
        volume: int,
        price: float,
        trade_date: str,
    ) -> None:
        self.logger.info(
            "Fill: %s %s %s %d @ %.2f", instrument, direction, order_id, volume, price
        )

    def calculate_positions(self, signals: List[Signal]) -> List[Signal]:
        return signals

    def get_status(self) -> Dict:
        return {
            "open_positions": len(self._open_positions),
            "condor_groups": len(self._condor_groups),
        }
