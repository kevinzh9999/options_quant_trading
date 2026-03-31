"""
vol_surface.py
--------------
职责：构建和分析期权波动率曲面。
- 按行权价（moneyness）和到期日组织隐含波动率数据
- 计算 ATM IV、波动率偏斜（skew）、峰度（kurtosis of smile）
- 提供插值功能（对任意行权价/到期日查询 IV）
- 生成用于可视化的波动率曲面数据

波动率曲面是 VRP 信号生成的核心输入之一。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from models.pricing.implied_vol import calc_implied_vol_batch

logger = logging.getLogger(__name__)

# 每个到期日至少需要的有效 IV 点数
MIN_SMILE_POINTS = 3


@dataclass
class VolSmile:
    """单个到期日的波动率微笑数据"""
    expire_date: str                    # 到期日，格式 YYYYMMDD
    T: float                            # 到期时间（年）
    strikes: np.ndarray                 # 行权价数组（升序）
    ivs: np.ndarray                     # 对应 IV 数组（小数）
    forward: float                      # 标的远期价格（F = S·e^{rT}）
    atm_iv: float = 0.0                 # ATM 隐含波动率
    skew_25d: float = 0.0               # 25-Delta 偏斜（RR25）
    butterfly_25d: float = 0.0          # 25-Delta 蝶式（BF25）

    def __post_init__(self) -> None:
        if len(self.strikes) > 0:
            self.atm_iv = self._interpolate_atm()

    def _interpolate_atm(self) -> float:
        """插值计算 ATM IV（行权价最接近远期价格的 IV）"""
        if len(self.strikes) == 0:
            return 0.0
        # 线性插值到 forward price
        return float(np.interp(self.forward, self.strikes, self.ivs))

    def get_iv(self, strike: float) -> float:
        """
        查询指定行权价的 IV（线性插值，超出范围时使用端点值）。

        Parameters
        ----------
        strike : float
            目标行权价

        Returns
        -------
        float
            插值后的 IV
        """
        if len(self.strikes) == 0:
            return self.atm_iv
        # np.interp 自动使用端点值外推
        return float(np.interp(strike, self.strikes, self.ivs))


class VolSurface:
    """
    期权波动率曲面。

    Parameters
    ----------
    trade_date : str
        行情日期，格式 YYYYMMDD
    underlying : str
        标的代码，如 IO / MO
    spot_price : float
        标的现价
    risk_free_rate : float
        无风险利率（年化小数）
    """

    def __init__(
        self,
        trade_date: str,
        underlying: str,
        spot_price: float,
        risk_free_rate: float,
    ) -> None:
        self.trade_date = trade_date
        self.underlying = underlying
        self.spot_price = spot_price
        self.risk_free_rate = risk_free_rate
        self._smiles: dict[str, VolSmile] = {}  # expire_date -> VolSmile

    # ------------------------------------------------------------------
    # 构建波动率曲面
    # ------------------------------------------------------------------

    def build_from_options_df(
        self,
        options_df: pd.DataFrame,
        min_volume: int = 0,
        min_oi: int = 0,
    ) -> "VolSurface":
        """
        从期权日线数据构建波动率曲面。

        Parameters
        ----------
        options_df : pd.DataFrame
            当日期权数据，需含 ts_code, strike_price, call_put, expire_date,
            close（或 settle）, volume, oi 列
        min_volume : int
            过滤成交量低于此值的合约（流动性筛选）
        min_oi : int
            过滤持仓量低于此值的合约

        Returns
        -------
        VolSurface
            self（支持链式调用）
        """
        self._smiles = {}
        if options_df.empty:
            return self

        df = options_df.copy()

        # 流动性过滤
        if min_volume > 0 and "volume" in df.columns:
            df = df[df["volume"] >= min_volume]
        if min_oi > 0 and "oi" in df.columns:
            df = df[df["oi"] >= min_oi]

        if df.empty:
            return self

        # 批量计算 IV
        iv_series = calc_implied_vol_batch(
            df, self.spot_price, self.risk_free_rate, self.trade_date
        )
        df = df.set_index("ts_code").copy()
        df["_iv"] = iv_series

        # 过滤 IV 计算失败的合约
        df = df.dropna(subset=["_iv"])
        df = df[df["_iv"] > 0]

        # 按到期日分组，构建 VolSmile
        ref_date = pd.Timestamp(self.trade_date)
        for expire_date, group in df.groupby("expire_date"):
            expire_str = str(expire_date)
            expire_ts = pd.Timestamp(expire_str)
            T = max((expire_ts - ref_date).days / 365.0, 0.0)

            if len(group) < MIN_SMILE_POINTS:
                continue

            # 排序
            group_sorted = group.sort_values("strike_price")
            strikes = group_sorted["strike_price"].values.astype(float)
            ivs = group_sorted["_iv"].values.astype(float)

            forward = self.spot_price * np.exp(self.risk_free_rate * T)
            smile = VolSmile(
                expire_date=expire_str,
                T=T,
                strikes=strikes,
                ivs=ivs,
                forward=forward,
            )
            self._smiles[expire_str] = smile
            logger.debug(
                "构建 %s 期权微笑：%d 个行权价，ATM IV=%.2f%%",
                expire_str, len(strikes), smile.atm_iv * 100
            )

        return self

    # ------------------------------------------------------------------
    # 查询接口
    # ------------------------------------------------------------------

    def get_atm_iv(self, expire_date: str) -> Optional[float]:
        """
        获取指定到期日的 ATM 隐含波动率。

        Returns
        -------
        float | None
            ATM IV（年化小数），到期日不存在时返回 None
        """
        smile = self._smiles.get(expire_date)
        if smile is None:
            return None
        return smile.atm_iv if smile.atm_iv > 0 else None

    def get_nearest_atm_iv(self, target_T: float = 30 / 365) -> Optional[float]:
        """
        获取到期时间最接近目标值的期权的 ATM IV。

        常用于获取"近月 ATM IV"与 GARCH 预测值比较。

        Parameters
        ----------
        target_T : float
            目标到期时间（年），默认 30/365 ≈ 30天

        Returns
        -------
        float | None
        """
        if not self._smiles:
            return None

        # 找到 T 最接近 target_T 的到期日
        best_expire = min(
            self._smiles.keys(),
            key=lambda exp: abs(self._smiles[exp].T - target_T),
        )
        return self._smiles[best_expire].atm_iv or None

    def get_smile(self, expire_date: str) -> Optional[VolSmile]:
        """获取指定到期日的 VolSmile 对象"""
        return self._smiles.get(expire_date)

    def get_all_expire_dates(self) -> list[str]:
        """获取所有可用到期日（YYYYMMDD，升序）"""
        return sorted(self._smiles.keys())

    # ------------------------------------------------------------------
    # 分析工具
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """
        将波动率曲面展开为 DataFrame，用于可视化和存储。

        Returns
        -------
        pd.DataFrame
            列：trade_date, underlying, expire_date, T, strike_price, iv, moneyness
        """
        rows = []
        for expire_date, smile in sorted(self._smiles.items()):
            for strike, iv in zip(smile.strikes, smile.ivs):
                rows.append({
                    "trade_date": self.trade_date,
                    "underlying": self.underlying,
                    "expire_date": expire_date,
                    "T": smile.T,
                    "strike_price": strike,
                    "iv": iv,
                    "moneyness": strike / self.spot_price,
                })
        if not rows:
            return pd.DataFrame(columns=[
                "trade_date", "underlying", "expire_date", "T",
                "strike_price", "iv", "moneyness"
            ])
        return pd.DataFrame(rows)

    def term_structure(self) -> pd.DataFrame:
        """
        返回不同到期日的 ATM IV 期限结构。

        Returns
        -------
        pd.DataFrame
            列：expire_date, T（年）, atm_iv；按 T 升序排列
        """
        rows = [
            {
                "expire_date": exp,
                "T": smile.T,
                "atm_iv": smile.atm_iv,
            }
            for exp, smile in self._smiles.items()
            if smile.atm_iv > 0
        ]
        if not rows:
            return pd.DataFrame(columns=["expire_date", "T", "atm_iv"])
        return pd.DataFrame(rows).sort_values("T").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 新 API：接收 ImpliedVolCalculator 输出的 options_with_iv DataFrame
    # ------------------------------------------------------------------

    def build_surface(self, options_with_iv: pd.DataFrame) -> pd.DataFrame:
        """
        从带 IV 的期权链数据构建波动率曲面透视表。

        Parameters
        ----------
        options_with_iv : pd.DataFrame
            ImpliedVolCalculator.calculate_iv_for_chain 的输出，
            须含 moneyness, expire_date, iv, is_valid 列

        Returns
        -------
        pd.DataFrame
            index=moneyness（四舍五入到 2 位小数），
            columns=expire_date，
            values=IV（认购认沽的平均 IV）
        """
        df = _filter_valid(options_with_iv)
        if df.empty:
            return pd.DataFrame()

        # 将 moneyness 离散化到 2 位小数，便于合并认购认沽
        df = df.copy()
        df["_mon"] = df["moneyness"].round(2)

        pivot = (
            df.groupby(["_mon", "expire_date"])["iv"]
            .mean()
            .unstack("expire_date")
        )
        pivot.index.name = "moneyness"
        return pivot.sort_index()

    def get_skew(
        self, options_with_iv: pd.DataFrame, expire_date: str
    ) -> pd.DataFrame:
        """
        提取特定到期日的波动率偏斜（smile / skew）。

        Parameters
        ----------
        options_with_iv : pd.DataFrame
            ImpliedVolCalculator 输出，须含 exercise_price, call_put,
            moneyness, iv, expire_date, is_valid 列
        expire_date : str
            目标到期日 YYYYMMDD

        Returns
        -------
        pd.DataFrame
            列：exercise_price, moneyness, iv_call, iv_put（缺少一侧为 NaN）
            按 moneyness 升序排列
        """
        df = _filter_valid(options_with_iv)
        df = df[df["expire_date"].astype(str) == str(expire_date)]
        if df.empty:
            return pd.DataFrame(columns=["exercise_price", "moneyness", "iv_call", "iv_put"])

        calls = (
            df[df["call_put"] == "C"]
            .groupby("exercise_price")[["moneyness", "iv"]]
            .first()
            .rename(columns={"iv": "iv_call"})
        )
        puts = (
            df[df["call_put"] == "P"]
            .groupby("exercise_price")[["moneyness", "iv"]]
            .first()
            .rename(columns={"iv": "iv_put"})
        )

        result = calls[["iv_call"]].join(puts[["iv_put"]], how="outer")
        # 取 moneyness（优先从 calls，否则 puts）
        mon_calls = calls["moneyness"] if "moneyness" in calls.columns else pd.Series(dtype=float)
        mon_puts = puts["moneyness"] if "moneyness" in puts.columns else pd.Series(dtype=float)
        result["moneyness"] = mon_calls.combine_first(mon_puts)
        result = result.reset_index().rename(columns={"index": "exercise_price"})
        result = result[["exercise_price", "moneyness", "iv_call", "iv_put"]]
        return result.sort_values("moneyness").reset_index(drop=True)

    def get_term_structure(
        self,
        options_with_iv: pd.DataFrame,
        underlying_price: float,
    ) -> pd.DataFrame:
        """
        提取平值期权的波动率期限结构。

        对每个到期日找行权价最接近标的价格的合约，
        取认购认沽 IV 均值作为该到期日的 ATM IV。

        Parameters
        ----------
        options_with_iv : pd.DataFrame
            ImpliedVolCalculator 输出，须含 exercise_price, expire_date,
            T, iv, is_valid 列
        underlying_price : float
            标的当日价格

        Returns
        -------
        pd.DataFrame
            列：expire_date, T, atm_iv；按 T 升序排列
        """
        df = _filter_valid(options_with_iv)
        if df.empty:
            return pd.DataFrame(columns=["expire_date", "T", "atm_iv"])

        rows = []
        for expire_date, group in df.groupby("expire_date"):
            strikes = group["exercise_price"].unique()
            atm_strike = min(strikes, key=lambda k: abs(float(k) - underlying_price))
            atm_group = group[group["exercise_price"] == atm_strike]
            atm_iv = atm_group["iv"].mean()
            T_val = float(atm_group["T"].iloc[0]) if "T" in atm_group.columns else float("nan")
            rows.append({
                "expire_date": str(expire_date),
                "T": T_val,
                "atm_iv": float(atm_iv),
            })

        return (
            pd.DataFrame(rows)
            .sort_values("T")
            .reset_index(drop=True)
        )

    def get_risk_reversal(
        self,
        options_with_iv: pd.DataFrame,
        expire_date: str,
        delta_target: float = 0.25,
    ) -> float:
        """
        计算 Risk Reversal（25-delta call IV − 25-delta put IV）。

        负值表示 put 比 call 贵（A 股正常的负偏斜）。

        Parameters
        ----------
        options_with_iv : pd.DataFrame
            须含 delta, iv, call_put, expire_date, is_valid 列
        expire_date : str
            目标到期日
        delta_target : float
            目标 delta 绝对值，默认 0.25

        Returns
        -------
        float
            RR = IV(call, Δ≈+delta_target) − IV(put, Δ≈-delta_target)
            无法计算时返回 NaN
        """
        call_iv = self._find_delta_iv(options_with_iv, expire_date, delta_target, "C")
        put_iv = self._find_delta_iv(options_with_iv, expire_date, -delta_target, "P")
        if call_iv is None or put_iv is None:
            return float("nan")
        return call_iv - put_iv

    def get_butterfly(
        self,
        options_with_iv: pd.DataFrame,
        expire_date: str,
        delta_target: float = 0.25,
    ) -> float:
        """
        计算 Butterfly（(25d call IV + 25d put IV) / 2 − ATM IV）。

        正值表示微笑（两端高中间低）。

        Parameters
        ----------
        options_with_iv : pd.DataFrame
            须含 delta, iv, call_put, expire_date, is_valid 列
        expire_date : str
            目标到期日
        delta_target : float
            目标 delta 绝对值，默认 0.25

        Returns
        -------
        float
            BF 值，无法计算时返回 NaN
        """
        call_iv = self._find_delta_iv(options_with_iv, expire_date, delta_target, "C")
        put_iv = self._find_delta_iv(options_with_iv, expire_date, -delta_target, "P")

        # ATM：moneyness 最接近 1.0 的合约 IV 均值
        df = _filter_valid(options_with_iv)
        df = df[df["expire_date"].astype(str) == str(expire_date)]
        if df.empty or call_iv is None or put_iv is None:
            return float("nan")

        strikes = df["exercise_price"].unique()
        if "moneyness" in df.columns:
            atm_strike = min(strikes, key=lambda k: abs(float(
                df[df["exercise_price"] == k]["moneyness"].iloc[0]) - 1.0))
        else:
            # fallback: use forward = spot
            forward = self.spot_price
            atm_strike = min(strikes, key=lambda k: abs(float(k) - forward))

        atm_iv = df[df["exercise_price"] == atm_strike]["iv"].mean()
        if pd.isna(atm_iv):
            return float("nan")

        return (call_iv + put_iv) / 2.0 - float(atm_iv)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _find_delta_iv(
        self,
        options_with_iv: pd.DataFrame,
        expire_date: str,
        delta_target: float,
        call_put: str,
    ) -> Optional[float]:
        """
        在指定到期日找 delta 最接近 delta_target 的合约，返回其 IV。

        Parameters
        ----------
        delta_target : float
            目标 delta，正数（call）或负数（put）
        call_put : str
            "C" 或 "P"
        """
        df = _filter_valid(options_with_iv)
        df = df[
            (df["expire_date"].astype(str) == str(expire_date))
            & (df["call_put"] == call_put)
            & df["delta"].notna()
        ]
        if df.empty:
            return None
        idx = (df["delta"] - delta_target).abs().idxmin()
        iv = df.loc[idx, "iv"]
        return float(iv) if pd.notna(iv) else None


# ======================================================================
# 模块级工具函数
# ======================================================================

def _filter_valid(options_with_iv: pd.DataFrame) -> pd.DataFrame:
    """过滤 is_valid=True 且 iv 非空的行。"""
    if options_with_iv.empty:
        return options_with_iv
    df = options_with_iv
    if "is_valid" in df.columns:
        df = df[df["is_valid"]]
    df = df[df["iv"].notna() & (df["iv"] > 0)]
    return df
