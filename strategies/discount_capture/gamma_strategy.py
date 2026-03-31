"""
gamma_strategy.py
-----------------
贴水收割 + Gamma Scalping 组合策略。

三条收益来源叠加：
1. 贴水收敛：做多IM季月期货，持有到到期赚取贴水（年化10-12%）
2. Put保护：买入虚值Put限制下行风险
3. Gamma Scalping：利用Put的正Gamma，通过Delta对冲把日内波动转化为利润
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from models.pricing.greeks import calc_delta, calc_gamma, calc_theta
from models.pricing.implied_vol import bs_price, calc_implied_vol

logger = logging.getLogger(__name__)

FUTURES_MULT = 200
OPTION_MULT = 100
_MO_RE = re.compile(r"^MO(\d{4})-(C|P)-(\d+)")


def _parse_mo_ts_code(ts_code: str) -> Optional[Tuple[str, str, float]]:
    """解析MO期权ts_code → (expire_month, call_put, strike)"""
    m = _MO_RE.match(str(ts_code))
    if m:
        return m.group(1), m.group(2), float(m.group(3))
    return None


class DiscountGammaStrategy:
    """
    贴水收割 + Gamma Scalping 组合策略。

    Parameters
    ----------
    db_manager : DBManager
    config : dict
        策略参数
    """

    def __init__(self, db_manager, config: Dict = None):
        self.db = db_manager
        cfg = config or {}

        # 合约选择参数
        self.min_dte = cfg.get("min_dte", 30)
        self.max_dte = cfg.get("max_dte", 180)
        self.min_volume = cfg.get("min_volume", 10000)
        self.roll_days = cfg.get("roll_days", 10)

        # Put选择参数
        self.put_otm_min = cfg.get("put_otm_min", 0.05)
        self.put_otm_max = cfg.get("put_otm_max", 0.15)
        self.put_min_volume = cfg.get("put_min_volume", 50)
        self.iv_high_threshold = cfg.get("iv_high_threshold", 0.28)
        self.iv_low_threshold = cfg.get("iv_low_threshold", 0.22)
        self.put_roll_days = cfg.get("put_roll_days", 10)

        # Delta对冲参数
        self.delta_hedge_ratio = cfg.get("delta_hedge_ratio", 0.6)

        self.risk_free_rate = cfg.get("risk_free_rate", 0.02)

    def select_best_futures_contract(self, trade_date: str) -> Optional[Dict]:
        """
        选择贴水最优的IM期货合约。

        Returns
        -------
        dict or None
            {symbol, futures_price, spot_price, discount_points,
             annualized_rate, days_to_expiry, iml_code, contract_month}
        """
        from utils.cffex_calendar import active_im_months, get_im_futures_prices, _IML_CODES

        # 获取现货价格
        spot_price = self._get_spot_price(trade_date)
        if spot_price is None:
            return None

        # 获取所有活跃合约价格
        im_prices = get_im_futures_prices(self.db, trade_date)
        if not im_prices:
            return None

        active_months = active_im_months(trade_date)

        # 获取到期日映射
        expire_dates = self._get_expire_dates()

        # 评估每个合约
        candidates = []
        today = pd.Timestamp(trade_date)

        for i, month in enumerate(active_months):
            if month not in im_prices:
                continue

            futures_price = im_prices[month]
            iml_code = _IML_CODES[i] if i < len(_IML_CODES) else f"IM{month}.CFX"

            # 计算到期天数
            expire_str = expire_dates.get(month, "")
            if expire_str:
                dte = max((pd.Timestamp(expire_str) - today).days, 1)
            else:
                y = 2000 + int(month[:2])
                m_num = int(month[2:])
                dte = max((pd.Timestamp(y, m_num, 20) - today).days, 1)

            # 过滤
            if dte < self.min_dte or dte > self.max_dte:
                continue

            discount_points = futures_price - spot_price
            if discount_points >= 0:
                continue  # 没有贴水

            ann_rate = abs(discount_points) / spot_price * (365 / dte)

            candidates.append({
                "symbol": f"IM{month}",
                "futures_price": futures_price,
                "spot_price": spot_price,
                "discount_points": discount_points,
                "annualized_rate": ann_rate,
                "days_to_expiry": dte,
                "iml_code": iml_code,
                "contract_month": month,
            })

        if not candidates:
            return None

        # 选年化贴水率最大的
        return max(candidates, key=lambda x: x["annualized_rate"])

    def select_optimal_put(
        self,
        futures_price: float,
        trade_date: str,
        expire_month: str,
    ) -> Optional[Dict]:
        """
        动态选择最优的保护性Put。

        最优Put标准：Gamma/Theta比率最高（Scalping效率最优）。

        Returns
        -------
        dict or None
        """
        # 获取该到期月份的Put期权链
        options_chain = self._get_put_chain(trade_date, expire_month)
        if options_chain is None or options_chain.empty:
            return None

        expire_date_str = self._get_expire_date_for_month(expire_month)
        if not expire_date_str:
            return None

        today = pd.Timestamp(trade_date)
        expire_ts = pd.Timestamp(expire_date_str)
        dte = max((expire_ts - today).days, 1)
        T = dte / 365.0

        if dte < self.put_roll_days:
            return None  # 太近到期不买新Put

        candidates = []
        for _, row in options_chain.iterrows():
            strike = row["strike"]
            close_price = row["close"]

            if close_price <= 0 or np.isnan(close_price):
                continue

            volume = row.get("volume", 0)
            if volume < self.put_min_volume:
                continue

            # 虚值程度
            otm_pct = (futures_price - strike) / futures_price
            if otm_pct < self.put_otm_min or otm_pct > self.put_otm_max:
                continue

            # 计算IV
            iv = calc_implied_vol(
                market_price=close_price,
                S=futures_price,
                K=strike,
                T=T,
                r=self.risk_free_rate,
                option_type="P",
            )
            if iv is None or iv <= 0:
                iv = 0.25  # fallback

            # 计算Greeks
            delta = calc_delta(futures_price, strike, T, self.risk_free_rate, iv, "P")
            gamma = calc_gamma(futures_price, strike, T, self.risk_free_rate, iv)
            theta = calc_theta(futures_price, strike, T, self.risk_free_rate, iv, "P")

            # Gamma/|Theta|比率
            if abs(theta) < 1e-6:
                gamma_theta_ratio = 0.0
            else:
                gamma_theta_ratio = gamma / abs(theta)

            # 评分
            # 保护距离合理性评分
            if 0.05 <= otm_pct <= 0.08:
                distance_score = 1.0
            elif 0.08 < otm_pct <= 0.12:
                distance_score = 0.7
            elif otm_pct < 0.05:
                distance_score = 0.4
            else:
                distance_score = 0.3

            # IV动态调整：高IV选更虚值的
            iv_adj = 1.0
            if iv > self.iv_high_threshold and otm_pct > 0.10:
                iv_adj = 1.2  # 高IV+远行权价加分
            elif iv < self.iv_low_threshold and otm_pct < 0.08:
                iv_adj = 1.2  # 低IV+近行权价加分

            liquidity_score = min(volume / 500, 1.0)

            score = (gamma_theta_ratio * 1e4 * 0.5
                     + distance_score * 0.3
                     + liquidity_score * 0.2) * iv_adj

            candidates.append({
                "ts_code": row.get("ts_code", ""),
                "strike": strike,
                "put_price": close_price,
                "cost": close_price * OPTION_MULT,
                "iv": iv,
                "delta": delta,
                "gamma": gamma,
                "theta": theta,
                "gamma_theta_ratio": gamma_theta_ratio,
                "protection_distance": futures_price - strike,
                "protection_pct": otm_pct,
                "volume": volume,
                "score": score,
                "expire_date": expire_date_str,
                "dte": dte,
            })

        if not candidates:
            return None

        return max(candidates, key=lambda x: x["score"])

    def calc_initial_put_lots(
        self,
        futures_lots: int,
        put_delta: float,
    ) -> int:
        """
        计算需要买入的Put手数（目标对冲delta_hedge_ratio的delta）。

        1手期货delta = 200元/点
        1手Put delta = put_delta × 100元/点
        """
        futures_delta = futures_lots * FUTURES_MULT
        target_hedge = futures_delta * self.delta_hedge_ratio
        if abs(put_delta) < 1e-6:
            return 1
        put_per_lot = abs(put_delta) * OPTION_MULT
        lots = int(round(target_hedge / put_per_lot))
        return max(lots, 1)

    # ─── helpers ───────────────────────────────────────────

    def _get_spot_price(self, trade_date: str) -> Optional[float]:
        """获取现货价格（000852.SH优先）"""
        try:
            r = self.db.query_df(
                f"SELECT close FROM index_daily "
                f"WHERE ts_code='000852.SH' AND trade_date='{trade_date}'"
            )
            if r is not None and not r.empty:
                return float(r["close"].iloc[0])
            r = self.db.query_df(
                "SELECT close FROM index_daily "
                "WHERE ts_code='000852.SH' ORDER BY trade_date DESC LIMIT 1"
            )
            if r is not None and not r.empty:
                return float(r["close"].iloc[0])
        except Exception:
            pass
        return None

    def _get_put_chain(self, trade_date: str, expire_month: str) -> Optional[pd.DataFrame]:
        """获取指定到期月份的Put期权链（含行权价）"""
        try:
            # 从options_daily获取数据，ts_code parse行权价
            df = self.db.query_df(
                f"SELECT od.ts_code, od.close, od.volume, od.oi, "
                f"oc.exercise_price, oc.expire_date "
                f"FROM options_daily od "
                f"JOIN options_contracts oc ON od.ts_code = oc.ts_code "
                f"WHERE od.ts_code LIKE 'MO{expire_month}-P-%' "
                f"AND od.trade_date = '{trade_date}' "
                f"AND oc.call_put = 'P'"
            )
            if df is not None and not df.empty:
                df["strike"] = df["exercise_price"].astype(float)
                return df
            # fallback: parse from ts_code
            df = self.db.query_df(
                f"SELECT ts_code, close, volume, oi "
                f"FROM options_daily "
                f"WHERE ts_code LIKE 'MO{expire_month}-P-%' "
                f"AND trade_date = '{trade_date}'"
            )
            if df is not None and not df.empty:
                strikes = []
                for tc in df["ts_code"]:
                    parsed = _parse_mo_ts_code(str(tc))
                    strikes.append(parsed[2] if parsed else np.nan)
                df["strike"] = strikes
                df = df.dropna(subset=["strike"])
                # 获取expire_date
                expire_date = self._get_expire_date_for_month(expire_month)
                df["expire_date"] = expire_date
                return df
        except Exception as e:
            logger.warning("获取Put链失败 %s %s: %s", trade_date, expire_month, e)
        return None

    def _get_expire_dates(self) -> Dict[str, str]:
        """获取所有MO到期月份→到期日映射"""
        expire_map = {}
        try:
            df = self.db.query_df(
                "SELECT DISTINCT ts_code, expire_date FROM options_contracts "
                "WHERE ts_code LIKE 'MO%'"
            )
            if df is not None and not df.empty:
                for _, row in df.iterrows():
                    parsed = _parse_mo_ts_code(str(row["ts_code"]))
                    if parsed and row["expire_date"]:
                        em = parsed[0]
                        if em not in expire_map:
                            expire_map[em] = str(row["expire_date"]).replace("-", "")
        except Exception:
            pass
        return expire_map

    def _get_expire_date_for_month(self, expire_month: str) -> Optional[str]:
        """获取某月份的到期日"""
        try:
            r = self.db.query_df(
                f"SELECT expire_date FROM options_contracts "
                f"WHERE ts_code LIKE 'MO{expire_month}-%' LIMIT 1"
            )
            if r is not None and not r.empty:
                return str(r["expire_date"].iloc[0]).replace("-", "")
        except Exception:
            pass
        return None
