"""
signal.py
---------
IM 期货贴水信号生成模块。

贴水（Contango Discount）：股指期货因分红预期等原因，期货价格 < 现货价格。
年化贴水率越高，持有多头期货至到期的隐含收益越大。
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DiscountSignal:
    """
    IM 贴水信号计算器。

    Parameters
    ----------
    db_manager : DBManager
        数据库管理器实例
    risk_free_rate : float
        无风险利率（年化），默认 0.02
    """

    def __init__(self, db_manager, risk_free_rate: float = 0.02):
        self.db = db_manager
        self.r = risk_free_rate

    def calculate_discount(self, trade_date: str) -> pd.DataFrame:
        """
        计算各活跃 IM 合约月份的贴水指标。

        Parameters
        ----------
        trade_date : str
            交易日期，格式 YYYYMMDD

        Returns
        -------
        pd.DataFrame
            columns: contract_month, futures_price, spot_price,
                     absolute_discount, annualized_discount_rate,
                     days_to_expiry, theoretical_basis, iml_code
        """
        from utils.cffex_calendar import (
            get_im_futures_prices,
            active_im_months,
            _IML_CODES,
        )

        # 1. 获取现货价格：优先使用 000852.SH（中证1000现货指数），回退到 IM.CFX
        spot_price = None
        spot_source = None
        try:
            spot_row = self.db.query_df(
                f"SELECT close FROM index_daily "
                f"WHERE ts_code='000852.SH' AND trade_date='{trade_date}'"
            )
            if spot_row is not None and not spot_row.empty:
                spot_price = float(spot_row["close"].iloc[0])
                spot_source = "000852.SH"
            else:
                # 取最近一条（防止当日数据未更新）
                spot_row = self.db.query_df(
                    "SELECT close FROM index_daily "
                    "WHERE ts_code='000852.SH' ORDER BY trade_date DESC LIMIT 1"
                )
                if spot_row is not None and not spot_row.empty:
                    spot_price = float(spot_row["close"].iloc[0])
                    spot_source = "000852.SH(最新)"
        except Exception:
            pass

        if spot_price is None:
            # 回退：IM.CFX 主力期货价格近似现货
            try:
                spot_row = self.db.query_df(
                    f"SELECT close FROM futures_daily "
                    f"WHERE ts_code='IM.CFX' AND trade_date='{trade_date}'"
                )
                if spot_row is None or spot_row.empty:
                    spot_row = self.db.query_df(
                        "SELECT close FROM futures_daily "
                        "WHERE ts_code='IM.CFX' ORDER BY trade_date DESC LIMIT 1"
                    )
                if spot_row is not None and not spot_row.empty:
                    spot_price = float(spot_row["close"].iloc[0])
                    spot_source = "IM.CFX(回退)"
            except Exception as e:
                logger.warning("获取现货价格失败: %s", e)

        if spot_price is None:
            logger.warning("无法获取现货价格（已尝试 000852.SH 和 IM.CFX）")
            return pd.DataFrame()

        logger.debug("现货价格 %.2f  来源: %s", spot_price, spot_source)

        # 2. 获取所有活跃 IML 合约价格
        try:
            im_prices = get_im_futures_prices(self.db, trade_date)
        except Exception as e:
            logger.warning("获取 IM 期货价格失败: %s", e)
            im_prices = {}

        if not im_prices:
            logger.warning("trade_date=%s 无 IM 期货价格数据", trade_date)
            return pd.DataFrame()

        # 3. 获取活跃月份列表及对应 IML 代码
        active_months = active_im_months(trade_date)

        # 4. 从 options_contracts 获取到期日映射
        expire_date_map: dict[str, str] = {}
        try:
            months_in = active_months + list(im_prices.keys())
            months_set = sorted(set(months_in))
            # options_contracts 的 expire_month 字段推导：从 ts_code MO{YYMM}-... 解析
            # 直接用 delist_date 作为 expire_date
            df_contracts = self.db.query_df(
                "SELECT ts_code, delist_date FROM options_contracts "
                "WHERE ts_code LIKE 'MO%'"
            )
            if df_contracts is not None and not df_contracts.empty:
                import re
                _mo_re = re.compile(r'^MO(\d{4})-')
                for _, row in df_contracts.iterrows():
                    m = _mo_re.match(str(row["ts_code"]))
                    if m and row["delist_date"]:
                        em = m.group(1)
                        if em not in expire_date_map:
                            expire_date_map[em] = str(row["delist_date"]).replace("-", "")
        except Exception as e:
            logger.warning("获取到期日映射失败: %s", e)

        # 5. 计算每个合约月份的贴水指标
        today_ts = pd.Timestamp(trade_date)
        records = []

        for i, month in enumerate(active_months):
            if month not in im_prices:
                continue

            futures_price = im_prices[month]
            iml_code = _IML_CODES[i] if i < len(_IML_CODES) else f"IM{month}.CFX"

            # 到期日 & 剩余天数
            expire_date_str = expire_date_map.get(month, "")
            if expire_date_str:
                try:
                    expire_ts = pd.Timestamp(expire_date_str)
                    raw_days = (expire_ts - today_ts).days
                    if raw_days < 0:
                        # 已到期合约，跳过
                        continue
                    days_to_expiry = max(raw_days, 1)
                except Exception:
                    days_to_expiry = 45  # 近似值
            else:
                # 用月末估算
                yymm = month
                y = 2000 + int(yymm[:2])
                m_num = int(yymm[2:])
                # CFFEX 到期日通常是第三个周五
                raw_days = (pd.Timestamp(y, m_num, 20) - today_ts).days
                if raw_days < 0:
                    continue
                days_to_expiry = max(raw_days, 1)

            T = days_to_expiry / 365.0

            # 贴水计算
            absolute_discount = futures_price - spot_price  # 负数 = 贴水
            if spot_price > 0 and days_to_expiry > 0:
                annualized_discount_rate = abs(absolute_discount) / spot_price * (365 / days_to_expiry)
            else:
                annualized_discount_rate = 0.0

            # 理论基差（假设 q=0）
            theoretical_basis = spot_price * self.r * T

            records.append({
                "contract_month":         month,
                "iml_code":               iml_code,
                "futures_price":          futures_price,
                "spot_price":             spot_price,
                "absolute_discount":      absolute_discount,
                "annualized_discount_rate": annualized_discount_rate,
                "days_to_expiry":         days_to_expiry,
                "theoretical_basis":      theoretical_basis,
            })

        # 尝试加入 IML3（如果已经在 active_months 里的第4个位置）
        if len(active_months) > 3 and active_months[3] in im_prices:
            pass  # already included above (index 3)

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        # 只保留有实际贴水的（期货价 < 现货价）
        # 不强制过滤，让调用方决定
        return df.sort_values("days_to_expiry").reset_index(drop=True)

    def get_discount_history(
        self,
        contract_type: str = "IML1",
        start_date: str = "20220722",
        end_date: str = None,
    ) -> pd.DataFrame:
        """
        获取连续合约的历史贴水时间序列。

        Parameters
        ----------
        contract_type : str
            连续合约代码，如 IML1, IML2, IML3
        start_date : str
            起始日期，格式 YYYYMMDD
        end_date : str, optional
            结束日期，默认为今天

        Returns
        -------
        pd.DataFrame
            columns: trade_date, futures_price, spot_price,
                     absolute_discount, raw_discount_rate
        """
        if end_date is None:
            end_date = date.today().strftime("%Y%m%d")

        ts_code = f"{contract_type}.CFX" if not contract_type.endswith(".CFX") else contract_type

        try:
            # 获取期货连续合约价格
            df_fut = self.db.query_df(
                f"SELECT trade_date, close as futures_price "
                f"FROM futures_daily "
                f"WHERE ts_code='{ts_code}' "
                f"AND trade_date >= '{start_date}' "
                f"AND trade_date <= '{end_date}' "
                f"ORDER BY trade_date ASC"
            )

            # 获取现货价格：优先 000852.SH，回退 IM.CFX
            df_spot = None
            try:
                df_spot = self.db.query_df(
                    f"SELECT trade_date, close as spot_price "
                    f"FROM index_daily "
                    f"WHERE ts_code='000852.SH' "
                    f"AND trade_date >= '{start_date}' "
                    f"AND trade_date <= '{end_date}' "
                    f"ORDER BY trade_date ASC"
                )
            except Exception:
                pass
            if df_spot is None or df_spot.empty:
                df_spot = self.db.query_df(
                    f"SELECT trade_date, close as spot_price "
                    f"FROM futures_daily "
                    f"WHERE ts_code='IM.CFX' "
                    f"AND trade_date >= '{start_date}' "
                    f"AND trade_date <= '{end_date}' "
                    f"ORDER BY trade_date ASC"
                )

            if df_fut is None or df_fut.empty:
                logger.warning("无 %s 历史数据", ts_code)
                return pd.DataFrame()

            if df_spot is None or df_spot.empty:
                logger.warning("无现货历史数据（已尝试 000852.SH 和 IM.CFX）")
                return pd.DataFrame()

            # 合并
            df = pd.merge(df_fut, df_spot, on="trade_date", how="inner")
            if df.empty:
                return pd.DataFrame()

            df["futures_price"] = df["futures_price"].astype(float)
            df["spot_price"] = df["spot_price"].astype(float)
            df["absolute_discount"] = df["futures_price"] - df["spot_price"]
            df["raw_discount_rate"] = df["absolute_discount"] / df["spot_price"]

            return df.sort_values("trade_date").reset_index(drop=True)

        except Exception as e:
            logger.warning("获取历史贴水失败: %s", e)
            return pd.DataFrame()

    def get_discount_percentile(
        self,
        current_discount_rate: float,
        contract_type: str = "IML1",
        lookback_days: int = 504,
    ) -> float:
        """
        计算当前贴水率在历史分布中的百分位。

        Parameters
        ----------
        current_discount_rate : float
            当前贴水率（正值，代表贴水幅度大小）
        contract_type : str
            合约类型
        lookback_days : int
            历史回溯天数（约2年）

        Returns
        -------
        float
            百分位 0-100（越高说明当前贴水越大）
        """
        try:
            hist = self.get_discount_history(contract_type=contract_type)
            if hist.empty or len(hist) < 20:
                return 50.0

            # 取最近 lookback_days 条
            hist_recent = hist.tail(lookback_days)
            # 历史贴水率取绝对值（贴水为负，取绝对值代表贴水幅度）
            hist_rates = hist_recent["raw_discount_rate"].abs().values

            # 百分位：current_discount_rate 在历史中的位置
            percentile = float(np.mean(hist_rates <= current_discount_rate) * 100)
            return round(percentile, 1)

        except Exception as e:
            logger.warning("计算百分位失败: %s", e)
            return 50.0

    def generate_signal(self, trade_date: str) -> dict:
        """
        生成贴水捕获信号。

        Parameters
        ----------
        trade_date : str
            交易日期，格式 YYYYMMDD

        Returns
        -------
        dict
            keys: signal, recommended_contract, annualized_discount,
                  discount_percentile, days_to_expiry, all_contracts
        """
        result = {
            "signal": "NONE",
            "recommended_contract": None,
            "annualized_discount": 0.0,
            "discount_percentile": 50.0,
            "days_to_expiry": 0,
            "all_contracts": pd.DataFrame(),
        }

        try:
            df = self.calculate_discount(trade_date)
            if df.empty:
                logger.warning("trade_date=%s 无贴水数据", trade_date)
                return result

            result["all_contracts"] = df

            # 筛选有效期合约：30-120 天
            valid = df[
                (df["days_to_expiry"] >= 30) &
                (df["days_to_expiry"] <= 120) &
                (df["absolute_discount"] < 0)  # 只选贴水合约（期货 < 现货）
            ].copy()

            if valid.empty:
                # 放宽条件，取所有有贴水的
                valid = df[df["absolute_discount"] < 0].copy()

            if valid.empty:
                return result

            # 找贴水率最高的合约
            best_idx = valid["annualized_discount_rate"].idxmax()
            best = valid.loc[best_idx]

            ann_rate = float(best["annualized_discount_rate"])
            iml_code = str(best["iml_code"])
            days_tte = int(best["days_to_expiry"])
            month = str(best["contract_month"])

            # 信号强度
            if ann_rate > 0.15:
                signal = "STRONG"
            elif ann_rate >= 0.10:
                signal = "MEDIUM"
            elif ann_rate >= 0.05:
                signal = "WEAK"
            else:
                signal = "NONE"

            # 历史百分位（用 raw_discount_rate 幅度）
            raw_rate = abs(float(best["absolute_discount"]) / float(best["spot_price"]))
            # 确定合约类型标识（如 IML1, IML2）
            contract_label = iml_code.replace(".CFX", "")
            percentile = self.get_discount_percentile(
                current_discount_rate=raw_rate,
                contract_type=contract_label,
            )

            result.update({
                "signal": signal,
                "recommended_contract": f"IM{month}",
                "iml_code": iml_code,
                "annualized_discount": ann_rate,
                "raw_discount_rate": -raw_rate,  # 负值表示贴水
                "discount_percentile": percentile,
                "days_to_expiry": days_tte,
            })

        except Exception as e:
            logger.warning("生成贴水信号失败: %s", e)

        return result
