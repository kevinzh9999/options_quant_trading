"""
pnl_attribution.py
------------------
职责：P&L（盈亏）归因分析。
将每日/持仓期间的总盈亏分解为以下分量：
- Delta PnL：标的价格变动的贡献
- Gamma PnL：价格大幅变动的非线性收益（凸性）
- Theta PnL：时间衰减带来的收益（做空波动率的主要收益来源）
- Vega PnL：隐含波动率变动的贡献
- 其他（残差）：模型误差、滑点、利息成本等

P&L 归因是策略监控和优化的重要工具。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DailyPnLAttribution:
    """单日 P&L 归因结果"""
    date: str
    total_pnl: float          # 当日总盈亏（元）
    realized_pnl: float       # 已实现盈亏（平仓 + 手续费）
    unrealized_pnl: float     # 未实现盈亏变动
    delta_pnl: float          # Delta 贡献
    gamma_pnl: float          # Gamma 贡献
    theta_pnl: float          # Theta 贡献（时间价值收入）
    vega_pnl: float           # Vega 贡献（IV 变动影响）
    residual_pnl: float       # 残差（total - 各分量之和）

    @property
    def explained_ratio(self) -> float:
        """各 Greeks 分量解释的比例（理论上应接近 1）"""
        if abs(self.total_pnl) < 1e-6:
            return 1.0
        explained = self.total_pnl - self.residual_pnl
        return explained / self.total_pnl


class PnLAttributor:
    """
    P&L 归因计算器。

    使用一阶泰勒展开估算各 Greeks 对盈亏的贡献：
    ΔV ≈ Δ·ΔS + ½·Γ·ΔS² + Θ·Δt + V·Δσ

    Greeks 单位约定（来自 daily_model_output）：
    - net_delta: 元/点  （标的变动1点，组合盈亏变动 net_delta 元）
    - net_gamma: 元/点² （标的变动ΔS点，二阶项 = 0.5 × gamma × ΔS²）
    - net_theta: 元/天  （每过一天，组合因时间衰减获得 net_theta 元）
    - net_vega:  元/1%σ （IV 变动1个百分点，组合盈亏变动 net_vega 元）
    """

    def __init__(self, db=None):
        self.db = db

    # ------------------------------------------------------------------
    # 从数据库读取归因所需数据
    # ------------------------------------------------------------------

    def _get_model_output(self, trade_date: str) -> Optional[Dict]:
        """读取某日的 daily_model_output。"""
        if self.db is None:
            return None
        df = self.db.query_df(
            "SELECT * FROM daily_model_output "
            "WHERE trade_date = ? AND underlying = 'IM'",
            (trade_date,),
        )
        if df is None or df.empty:
            return None
        return df.iloc[0].to_dict()

    def _get_account_snapshot(self, trade_date: str) -> Optional[Dict]:
        """读取某日的账户快照。"""
        if self.db is None:
            return None
        df = self.db.query_df(
            "SELECT * FROM account_snapshots WHERE trade_date = ?",
            (trade_date,),
        )
        if df is None or df.empty:
            return None
        return df.iloc[0].to_dict()

    def _get_im_close(self, trade_date: str) -> Optional[float]:
        """读取 IM.CFX 收盘价。"""
        if self.db is None:
            return None
        df = self.db.query_df(
            "SELECT close FROM futures_daily "
            "WHERE ts_code = 'IM.CFX' AND trade_date = ?",
            (trade_date,),
        )
        if df is None or df.empty:
            return None
        return float(df["close"].iloc[0])

    def _get_prev_trade_date(self, trade_date: str) -> Optional[str]:
        """获取前一个交易日。"""
        if self.db is None:
            return None
        df = self.db.query_df(
            "SELECT trade_date FROM futures_daily "
            "WHERE ts_code = 'IM.CFX' AND trade_date < ? "
            "ORDER BY trade_date DESC LIMIT 1",
            (trade_date,),
        )
        if df is None or df.empty:
            return None
        return str(df["trade_date"].iloc[0])

    def _get_realized_pnl(self, trade_date: str) -> float:
        """
        从 trade_records 计算当日已实现盈亏。

        对每笔平仓记录（offset = CLOSE/CLOSETODAY），
        通过 position_snapshots（前一日）的 open_price_avg 获得开仓均价。
        """
        if self.db is None:
            return 0.0

        prev_date = self._get_prev_trade_date(trade_date)
        if not prev_date:
            return 0.0

        # 前日持仓（用于获取开仓均价）
        prev_pos = self.db.query_df(
            "SELECT symbol, direction, open_price_avg, volume "
            "FROM position_snapshots WHERE trade_date = ?",
            (prev_date,),
        )
        if prev_pos is None or prev_pos.empty:
            return 0.0

        pos_map: Dict[str, Dict] = {}
        for _, row in prev_pos.iterrows():
            pos_map[row["symbol"]] = {
                "direction": row["direction"],
                "open_price_avg": float(row["open_price_avg"]),
            }

        # 当日成交记录
        trades = self.db.query_df(
            "SELECT symbol, direction, offset, volume, price, commission "
            "FROM trade_records WHERE trade_date = ?",
            (trade_date,),
        )
        if trades is None or trades.empty:
            return 0.0

        # 合约乘数映射
        MO_MULT = 100   # MO 期权
        IM_MULT = 200   # IM 期货
        IF_MULT = 300   # IF 期货
        IH_MULT = 300   # IH 期货

        def _multiplier(sym: str) -> int:
            s = sym.upper()
            if "MO" in s:
                return MO_MULT
            if "IM" in s:
                return IM_MULT
            if "IF" in s:
                return IF_MULT
            if "IH" in s:
                return IH_MULT
            return 100  # default

        realized = 0.0
        for _, tr in trades.iterrows():
            offset = str(tr["offset"]).upper()
            if offset not in ("CLOSE", "CLOSETODAY"):
                continue

            sym = tr["symbol"]
            vol = int(tr["volume"])
            close_price = float(tr["price"])
            commission = float(tr.get("commission", 0) or 0)
            mult = _multiplier(sym)

            # 查找 TQ 格式的 symbol（position_snapshots 用 CFFEX.MO2604-P-7800）
            tq_sym = f"CFFEX.{sym}" if not sym.startswith("CFFEX.") else sym
            pos_info = pos_map.get(tq_sym)

            if pos_info is None:
                # 日内开平（当日新开仓当日平仓），无前日持仓记录
                # 无法从前日数据得到开仓价，跳过（贡献已包含在 float_profit 变动中）
                continue

            open_price = pos_info["open_price_avg"]
            pos_dir = pos_info["direction"]

            # PnL = (平仓价 - 开仓价) × 手数 × 乘数 × 方向符号
            if pos_dir == "LONG":
                pnl = (close_price - open_price) * vol * mult
            else:  # SHORT
                pnl = (open_price - close_price) * vol * mult

            realized += pnl - commission

        return realized

    # ------------------------------------------------------------------
    # 主归因计算
    # ------------------------------------------------------------------

    def attribute_daily_pnl(
        self,
        trade_date: str,
        prev_date: str | None = None,
    ) -> Optional[DailyPnLAttribution]:
        """
        计算单日 P&L 归因。

        使用前一日（BOD）的 Greeks 和当日价格/IV 变动。

        归因公式：
            Delta PnL = net_delta × ΔS
            Gamma PnL = 0.5 × net_gamma × ΔS²
            Theta PnL = net_theta × 1  （一个交易日）
            Vega PnL  = net_vega × ΔIV  （ΔIV 单位：百分点）
            Residual  = Total PnL - (Delta + Gamma + Theta + Vega)
        """
        if prev_date is None:
            prev_date = self._get_prev_trade_date(trade_date)
        if not prev_date:
            logger.warning("无法获取 %s 的前一交易日", trade_date)
            return None

        # BOD Greeks（前一日收盘时的组合 Greeks）
        prev_model = self._get_model_output(prev_date)
        if prev_model is None:
            logger.warning("无 %s 的 daily_model_output", prev_date)
            return None

        today_model = self._get_model_output(trade_date)
        if today_model is None:
            logger.warning("无 %s 的 daily_model_output", trade_date)
            return None

        # ΔS = 今日IM收盘 - 昨日IM收盘
        prev_close = self._get_im_close(prev_date)
        today_close = self._get_im_close(trade_date)
        if prev_close is None or today_close is None:
            logger.warning("IM.CFX 收盘价缺失: prev=%s today=%s", prev_close, today_close)
            return None

        delta_s = today_close - prev_close

        # ΔIV = 今日ATM_IV - 昨日ATM_IV（百分点）
        prev_iv = prev_model.get("atm_iv")
        today_iv = today_model.get("atm_iv")
        delta_iv_pp = 0.0  # 百分点
        if prev_iv is not None and today_iv is not None:
            delta_iv_pp = (today_iv - prev_iv) * 100  # 转换为百分点

        # BOD Greeks（前一日收盘值）
        bod_delta = float(prev_model.get("net_delta") or 0)
        bod_gamma = float(prev_model.get("net_gamma") or 0)
        bod_theta = float(prev_model.get("net_theta") or 0)
        bod_vega = float(prev_model.get("net_vega") or 0)

        # Greeks 归因计算
        pnl_delta = bod_delta * delta_s
        pnl_gamma = 0.5 * bod_gamma * delta_s * delta_s
        pnl_theta = bod_theta * 1.0  # 1 个交易日
        pnl_vega = bod_vega * delta_iv_pp

        # 总盈亏（从账户快照的权益变动计算）
        prev_acct = self._get_account_snapshot(prev_date)
        today_acct = self._get_account_snapshot(trade_date)

        total_pnl = 0.0
        if prev_acct and today_acct:
            # 权益 = balance + float_profit
            prev_equity = float(prev_acct.get("balance", 0)) + float(prev_acct.get("float_profit", 0))
            today_equity = float(today_acct.get("balance", 0)) + float(today_acct.get("float_profit", 0))
            total_pnl = today_equity - prev_equity
        else:
            logger.warning("账户快照缺失，total_pnl 使用 Greeks 之和")
            total_pnl = pnl_delta + pnl_gamma + pnl_theta + pnl_vega

        # 已实现盈亏
        realized_pnl = self._get_realized_pnl(trade_date)
        unrealized_pnl = total_pnl - realized_pnl

        # 残差
        explained = pnl_delta + pnl_gamma + pnl_theta + pnl_vega
        residual = total_pnl - explained

        return DailyPnLAttribution(
            date=trade_date,
            total_pnl=round(total_pnl, 2),
            realized_pnl=round(realized_pnl, 2),
            unrealized_pnl=round(unrealized_pnl, 2),
            delta_pnl=round(pnl_delta, 2),
            gamma_pnl=round(pnl_gamma, 2),
            theta_pnl=round(pnl_theta, 2),
            vega_pnl=round(pnl_vega, 2),
            residual_pnl=round(residual, 2),
        )

    def attribute_period_pnl(
        self,
        start_date: str,
        end_date: str,
        daily_attributions: list[DailyPnLAttribution],
    ) -> dict[str, float]:
        """汇总持仓期间的 P&L 归因。"""
        result = {
            "total": 0.0, "realized": 0.0, "unrealized": 0.0,
            "delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0,
            "residual": 0.0,
        }
        for a in daily_attributions:
            result["total"] += a.total_pnl
            result["realized"] += a.realized_pnl
            result["unrealized"] += a.unrealized_pnl
            result["delta"] += a.delta_pnl
            result["gamma"] += a.gamma_pnl
            result["theta"] += a.theta_pnl
            result["vega"] += a.vega_pnl
            result["residual"] += a.residual_pnl
        return result

    def to_dataframe(self, attributions: list[DailyPnLAttribution]) -> pd.DataFrame:
        """将归因结果列表转换为 DataFrame，用于可视化。"""
        rows = []
        for a in attributions:
            rows.append({
                "date": a.date,
                "total": a.total_pnl,
                "realized": a.realized_pnl,
                "unrealized": a.unrealized_pnl,
                "delta": a.delta_pnl,
                "gamma": a.gamma_pnl,
                "theta": a.theta_pnl,
                "vega": a.vega_pnl,
                "residual": a.residual_pnl,
                "explained_ratio": a.explained_ratio,
            })
        return pd.DataFrame(rows)
