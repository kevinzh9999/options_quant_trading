"""
implied_vol.py
--------------
职责：期权隐含波动率（Implied Volatility, IV）的计算。
- Black-Scholes（BS）公式正向定价
- 牛顿-拉弗森法反推 IV（快速，一般情况）
- 二分法反推 IV（稳健，兜底方案）
- Brent 方法（scipy.optimize.brentq，自动选择）

适用标的：沪深300股指期权（IO）、中证1000股指期权（MO）
期权类型：欧式期权（股指期权均为欧式）
"""

from __future__ import annotations

import logging
import re as _re
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from scipy import optimize, stats

logger = logging.getLogger(__name__)

# IV 搜索边界（避免极端情况下不收敛）
IV_LOWER_BOUND = 1e-6   # 最小 IV（接近0）
IV_UPPER_BOUND = 10.0   # 最大 IV（1000%，理论上界）
IV_INIT_GUESS = 0.3     # 牛顿法初始猜测值（30%）


class OptionType(str, Enum):
    """期权类型"""
    CALL = "C"
    PUT = "P"


# ======================================================================
# Black-Scholes 定价公式
# ======================================================================

def bs_d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    """
    计算 BS 公式中的 d1 和 d2。

    d1 = (ln(S/K) + (r + σ²/2)·T) / (σ·√T)
    d2 = d1 - σ·√T

    Returns
    -------
    tuple[float, float]
        (d1, d2)
    """
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: OptionType | str,
) -> float:
    """
    Black-Scholes 欧式期权定价公式。

    Parameters
    ----------
    S : float
        标的现价（指数点位）
    K : float
        行权价
    T : float
        到期时间（年，如 30天 = 30/365）
    r : float
        无风险利率（连续复利，小数，如 0.025 表示 2.5%）
    sigma : float
        波动率（年化小数，如 0.20 表示 20%）
    option_type : OptionType | str
        期权类型：'C' 或 'P'

    Returns
    -------
    float
        期权理论价格（指数点位）

    Notes
    -----
    d1 = (ln(S/K) + (r + σ²/2)·T) / (σ·√T)
    d2 = d1 - σ·√T
    Call = S·N(d1) - K·e^{-rT}·N(d2)
    Put  = K·e^{-rT}·N(-d2) - S·N(-d1)
    """
    if T <= 0:
        # 已到期：返回内在价值
        intrinsic = S - K if (str(option_type) == OptionType.CALL or option_type == "C") else K - S
        return max(intrinsic, 0.0)

    d1, d2 = bs_d1_d2(S, K, T, r, sigma)
    discount = np.exp(-r * T)

    if str(option_type) == OptionType.CALL or option_type == "C":
        return S * stats.norm.cdf(d1) - K * discount * stats.norm.cdf(d2)
    else:
        return K * discount * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)


def bs_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """
    计算 BS Vega（内部使用，原始单位，未除以100）。

    Vega = S·√T·N'(d1)
    """
    d1, _ = bs_d1_d2(S, K, T, r, sigma)
    return S * np.sqrt(T) * stats.norm.pdf(d1)


# ======================================================================
# 隐含波动率反推
# ======================================================================

def calc_implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType | str,
    method: str = "brent",
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Optional[float]:
    """
    反推期权隐含波动率。

    Parameters
    ----------
    market_price : float
        期权市场价格（指数点位）
    S : float
        标的现价
    K : float
        行权价
    T : float
        到期时间（年）
    r : float
        无风险利率（年化小数）
    option_type : OptionType | str
        期权类型：'C' 或 'P'
    method : str
        求解方法：'newton'（牛顿法）/ 'bisect'（二分法）/ 'brent'（Brent 方法）
    tol : float
        收敛精度，默认 1e-6
    max_iter : int
        最大迭代次数

    Returns
    -------
    float | None
        隐含波动率（年化小数），求解失败或无解返回 None

    Notes
    -----
    - 深度实值/虚值期权流动性差，IV 可能无解（期权价格低于内在价值时）
    - T <= 0 时直接返回 None（已到期合约）
    - 推荐使用 Brent 方法，在收敛性和速度间取得平衡
    """
    if T <= 0:
        return None

    # 检查价格有效性（不能低于内在价值）
    if str(option_type) == OptionType.CALL or option_type == "C":
        intrinsic = max(S - K * np.exp(-r * T), 0.0)
    else:
        intrinsic = max(K * np.exp(-r * T) - S, 0.0)

    if market_price < intrinsic - tol:
        logger.debug("期权价格 %.4f 低于内在价值 %.4f，无法求解 IV", market_price, intrinsic)
        return None

    ot = OptionType(option_type) if isinstance(option_type, str) else option_type

    if method == "newton":
        return _newton_iv(market_price, S, K, T, r, ot, tol, max_iter)
    elif method == "bisect":
        return _bisect_iv(market_price, S, K, T, r, ot, tol, max_iter)
    else:  # brent (default)
        return _brent_iv(market_price, S, K, T, r, ot, tol)


def _newton_iv(
    target: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType,
    tol: float,
    max_iter: int,
) -> Optional[float]:
    """
    牛顿-拉弗森法求解 IV。

    迭代公式：σ_{n+1} = σ_n - (BS(σ_n) - market_price) / Vega(σ_n)
    Vega 接近 0 时（深度实值/虚值）自动切换 Brent 方法。
    """
    sigma = IV_INIT_GUESS
    for _ in range(max_iter):
        price = bs_price(S, K, T, r, sigma, option_type)
        vega = bs_vega(S, K, T, r, sigma)
        if abs(vega) < 1e-10:
            # Vega 太小，切换到 Brent
            return _brent_iv(target, S, K, T, r, option_type, tol)
        diff = price - target
        if abs(diff) < tol:
            return sigma
        sigma -= diff / vega
        sigma = max(min(sigma, IV_UPPER_BOUND), IV_LOWER_BOUND)
    logger.debug("牛顿法未收敛，切换到 Brent")
    return _brent_iv(target, S, K, T, r, option_type, tol)


def _bisect_iv(
    target: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType,
    tol: float,
    max_iter: int,
) -> Optional[float]:
    """二分法求解 IV。"""
    lo, hi = IV_LOWER_BOUND, IV_UPPER_BOUND
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        price = bs_price(S, K, T, r, mid, option_type)
        if abs(price - target) < tol:
            return mid
        if price < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _brent_iv(
    target: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: OptionType,
    tol: float,
) -> Optional[float]:
    """Brent 方法（scipy.optimize.brentq）求解 IV。"""
    def objective(sigma: float) -> float:
        return bs_price(S, K, T, r, sigma, option_type) - target

    try:
        # 检查端点符号
        f_lo = objective(IV_LOWER_BOUND)
        f_hi = objective(IV_UPPER_BOUND)
        if f_lo * f_hi > 0:
            # 无根（价格超出范围）
            return None
        iv = optimize.brentq(objective, IV_LOWER_BOUND, IV_UPPER_BOUND, xtol=tol)
        return float(iv)
    except (ValueError, RuntimeError) as e:
        logger.debug("Brent 求解失败: %s", e)
        return None


def calc_implied_vol_batch(
    options_df: pd.DataFrame,
    spot_price: float,
    risk_free_rate: float,
    trade_date: Optional[str] = None,
) -> pd.Series:
    """
    批量计算 DataFrame 中所有期权的隐含波动率。

    Parameters
    ----------
    options_df : pd.DataFrame
        期权数据，需含 ts_code, strike_price, call_put, expire_date, close 列
    spot_price : float
        标的现价
    risk_free_rate : float
        无风险利率（年化小数）
    trade_date : str, optional
        交易日期 YYYYMMDD；不提供时使用 expire_date 推算（仅用于测试）

    Returns
    -------
    pd.Series
        以 ts_code 为索引的 IV 序列，求解失败的合约填充 NaN

    Notes
    -----
    - expire_date 格式为 YYYYMMDD，内部自动计算到期时间 T
    - 使用收盘价（close）计算 IV
    """
    if options_df.empty:
        return pd.Series(dtype=float)

    results = {}
    ref_date = pd.Timestamp(trade_date) if trade_date else pd.Timestamp.today()

    for _, row in options_df.iterrows():
        ts_code = row.get("ts_code", "")
        try:
            expire_ts = pd.Timestamp(str(row["expire_date"]))
            T = max((expire_ts - ref_date).days / 365.0, 0.0)
            iv = calc_implied_vol(
                market_price=float(row["close"]),
                S=spot_price,
                K=float(row["strike_price"]),
                T=T,
                r=risk_free_rate,
                option_type=str(row["call_put"]),
            )
        except Exception as e:
            logger.debug("IV 计算异常 %s: %s", ts_code, e)
            iv = None
        results[ts_code] = iv

    return pd.Series(results, dtype=float)


# ======================================================================
# ImpliedVolCalculator 类：期权链批量 IV 计算与 ATM IV 提取
# ======================================================================

# 标的代码 → 期货品种代码映射（用于查找底层期货价格）
_UNDERLYING_TO_FUTURES = {
    "IO": "IF",   # 沪深300股指期权 → 沪深300期货
    "MO": "IM",   # 中证1000股指期权 → 中证1000期货
    "HO": "IH",   # 上证50股指期权 → 上证50期货
}

# 期权代码解析正则，例：MO2604-P-7200.CFX  IO2606-C-4200.CFX
_OPT_EXPIRY_RE = _re.compile(r'^(MO|IO|HO)(\d{4})-')


def get_underlying_future_for_option(option_ts_code: str) -> str:
    """
    从期权合约代码推导对应期货合约代码（同到期月份）。

    Parameters
    ----------
    option_ts_code : str
        期权合约代码，如 "MO2604-P-7200.CFX"

    Returns
    -------
    str
        对应期货合约代码，如 "IM2604.CFX"

    Examples
    --------
    >>> get_underlying_future_for_option("MO2604-P-7200.CFX")
    'IM2604.CFX'
    >>> get_underlying_future_for_option("IO2606-C-4200.CFX")
    'IF2606.CFX'
    """
    m = _OPT_EXPIRY_RE.match(str(option_ts_code))
    if not m:
        raise ValueError(f"无法从期权代码解析到期月份: {option_ts_code}")
    opt_underlying = m.group(1)   # "MO" / "IO" / "HO"
    expire_month   = m.group(2)   # "2604"
    futures_prefix = _UNDERLYING_TO_FUTURES.get(opt_underlying.upper(), opt_underlying)
    return f"{futures_prefix}{expire_month}.CFX"

_MIN_T_DAYS = 7          # 剩余天数低于此值标记 is_valid=False
_ATM_IV_MIN = 0.01       # ATM IV 合理下界（1%）
_ATM_IV_MAX = 5.0        # ATM IV 合理上界（500%）


class ImpliedVolCalculator:
    """
    从期权市场数据批量计算隐含波动率。

    Parameters
    ----------
    risk_free_rate : float
        无风险利率（年化），A股一般用 SHIBOR 或国债利率，默认 2%
    dividend_yield : float
        标的连续分红率（年化），股指期权可近似为指数股息率，默认 0
    """

    def __init__(
        self,
        risk_free_rate: float = 0.02,
        dividend_yield: float = 0.0,
    ) -> None:
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield

    # ------------------------------------------------------------------
    # 期权链 IV 计算
    # ------------------------------------------------------------------

    def calculate_iv_for_chain(
        self,
        options_chain: pd.DataFrame,
        underlying_price: float,
        trade_date: str,
        underlying_prices_by_expiry: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        为一组期权链（同一交易日、同一标的）计算隐含波动率。

        Parameters
        ----------
        options_chain : pd.DataFrame
            期权链，须含列：exercise_price, call_put, expire_date, close
            可选列：volume（有则用于 is_valid 过滤）、expire_month
        underlying_price : float
            标的当日收盘价（主力合约或默认价，当 underlying_prices_by_expiry 未提供时使用）
        trade_date : str
            交易日 YYYYMMDD，用于计算剩余到期时间
        underlying_prices_by_expiry : dict[str, float], optional
            按到期月份（如 "2604"）分别指定期货标的价格，用于修正 A 股股指期货贴水偏差。
            键为四位到期月份字符串，值为对应期货合约收盘价。
            键可从 expire_month 列读取，或从 expire_date 列推算（YYYYMMDD → "YYMM"）。
            缺失的月份回退到 underlying_price。

        Returns
        -------
        pd.DataFrame
            原始列 + iv, T, moneyness, delta, is_valid 五列
        """
        from models.pricing.black_scholes import BlackScholes

        if options_chain.empty:
            result = options_chain.copy()
            for col in ("iv", "T", "moneyness", "delta", "is_valid"):
                result[col] = pd.Series(dtype=float if col != "is_valid" else bool)
            return result

        ref_date = pd.Timestamp(trade_date)
        result = options_chain.copy()

        iv_list, T_list, moneyness_list, delta_list, valid_list = [], [], [], [], []

        for _, row in result.iterrows():
            try:
                expire_ts = pd.Timestamp(str(row["expire_date"]))
                T = max((expire_ts - ref_date).days / 365.0, 0.0)
                K = float(row["exercise_price"])
                cp = str(row["call_put"]).upper()
                price = float(row["close"]) if pd.notna(row["close"]) else float("nan")

                # 按到期月份选择对应期货标的价格（修正贴水偏差）
                S = underlying_price
                if underlying_prices_by_expiry:
                    if "expire_month" in row.index and pd.notna(row["expire_month"]):
                        S = underlying_prices_by_expiry.get(str(row["expire_month"]), underlying_price)
                    elif "expire_date" in row.index and pd.notna(row["expire_date"]):
                        exp_str = str(row["expire_date"]).replace("-", "")
                        S = underlying_prices_by_expiry.get(exp_str[2:6], underlying_price)

                # 计算 IV（close 为 NaN 或非正时跳过）
                if np.isnan(price) or price <= 0:
                    iv_list.append(float("nan"))
                    T_list.append(T)
                    moneyness_list.append(S / K if cp == "C" else K / S)
                    delta_list.append(float("nan"))
                    valid_list.append(False)
                    continue

                iv = BlackScholes.implied_volatility(
                    market_price=price,
                    S=S,
                    K=K,
                    T=T,
                    r=self.risk_free_rate,
                    q=self.dividend_yield,
                    option_type=cp,
                )

                # 计算 delta
                if not np.isnan(iv) and T > 0:
                    delta = BlackScholes.delta(
                        S, K, T, self.risk_free_rate, iv,
                        self.dividend_yield, cp
                    )
                else:
                    delta = float("nan")

                # moneyness：认购 = S/K，认沽 = K/S
                moneyness = S / K if cp == "C" else K / S

                # is_valid 判断
                days_left = (expire_ts - ref_date).days
                is_valid = (
                    days_left >= _MIN_T_DAYS
                    and not np.isnan(iv)
                    and (not pd.isna(price))
                    and price > 0
                )
                # 成交量为0时标记 False
                if "volume" in row.index and pd.notna(row["volume"]) and row["volume"] == 0:
                    is_valid = False

            except Exception as e:
                logger.debug("calculate_iv row error: %s", e)
                iv, T, delta, moneyness, is_valid = float("nan"), 0.0, float("nan"), float("nan"), False

            iv_list.append(iv)
            T_list.append(T)
            moneyness_list.append(moneyness)
            delta_list.append(delta)
            valid_list.append(is_valid)

        result["iv"] = iv_list
        result["T"] = T_list
        result["moneyness"] = moneyness_list
        result["delta"] = delta_list
        result["is_valid"] = valid_list

        return result

    # ------------------------------------------------------------------
    # ATM IV 提取
    # ------------------------------------------------------------------

    def get_atm_iv(
        self,
        options_chain_with_iv: pd.DataFrame,
        underlying_price: float,
    ) -> float:
        """
        从已计算 IV 的期权链中提取平值（ATM）隐含波动率。

        找行权价最接近标的价格的合约，取认购和认沽 IV 的平均值。
        若只有一侧有效，则取单侧 IV。

        Parameters
        ----------
        options_chain_with_iv : pd.DataFrame
            calculate_iv_for_chain 的输出（含 iv、is_valid 列）
        underlying_price : float
            标的价格

        Returns
        -------
        float
            ATM 隐含波动率（年化），无法计算时返回 NaN
        """
        if options_chain_with_iv.empty or "iv" not in options_chain_with_iv.columns:
            return float("nan")

        valid = options_chain_with_iv[
            options_chain_with_iv.get("is_valid", pd.Series([True] * len(options_chain_with_iv)))
        ]
        if valid.empty:
            return float("nan")

        # 找最近行权价
        strikes = valid["exercise_price"].unique()
        if len(strikes) == 0:
            return float("nan")
        atm_strike = min(strikes, key=lambda k: abs(float(k) - underlying_price))

        atm_rows = valid[valid["exercise_price"] == atm_strike]
        ivs = atm_rows["iv"].dropna()
        ivs = ivs[(ivs >= _ATM_IV_MIN) & (ivs <= _ATM_IV_MAX)]

        if ivs.empty:
            return float("nan")
        return float(ivs.mean())

    # ------------------------------------------------------------------
    # 历史 ATM IV 时间序列构建
    # ------------------------------------------------------------------

    def build_iv_history(
        self,
        db_manager,
        underlying: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        构建历史每日 ATM IV 时间序列。

        Parameters
        ----------
        db_manager : DBManager
            数据库管理器
        underlying : str
            标的代码，如 "IO"
        start_date : str
            起始日期 YYYYMMDD
        end_date : str
            结束日期 YYYYMMDD

        Returns
        -------
        pd.DataFrame
            列：trade_date, atm_iv, underlying_price
            无数据的日期 atm_iv 为 NaN
        """
        # 获取交易日历
        calendar = db_manager.get_trade_calendar(start_date=start_date, end_date=end_date)
        if calendar.empty:
            # 无交易日历：按自然日生成（会包含非交易日，但 IV 均为 NaN）
            dates = pd.date_range(start_date, end_date, freq="B").strftime("%Y%m%d").tolist()
        else:
            dates = calendar["trade_date"].tolist()

        # 期货品种映射
        futures_code = _UNDERLYING_TO_FUTURES.get(underlying.upper(), underlying)

        records = []
        for trade_date in dates:
            underlying_price = self._get_underlying_price(
                db_manager, futures_code, trade_date
            )
            if underlying_price is None or np.isnan(underlying_price):
                records.append({
                    "trade_date": trade_date,
                    "atm_iv": float("nan"),
                    "underlying_price": float("nan"),
                })
                continue

            chain = db_manager.get_options_chain(underlying, trade_date)
            if chain.empty:
                records.append({
                    "trade_date": trade_date,
                    "atm_iv": float("nan"),
                    "underlying_price": underlying_price,
                })
                continue

            chain_iv = self.calculate_iv_for_chain(chain, underlying_price, trade_date)
            atm_iv = self.get_atm_iv(chain_iv, underlying_price)
            records.append({
                "trade_date": trade_date,
                "atm_iv": atm_iv,
                "underlying_price": underlying_price,
            })

        return pd.DataFrame(records, columns=["trade_date", "atm_iv", "underlying_price"])

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _get_underlying_price(
        self, db_manager, futures_code: str, trade_date: str
    ) -> Optional[float]:
        """
        从期货日线数据中获取主力合约价格。

        策略：取同日期持仓量最大的合约收盘价作为主力合约价格。
        """
        try:
            df = db_manager.get_futures_daily(
                ts_code=f"%{futures_code}%",
                start_date=trade_date,
                end_date=trade_date,
            )
            if df.empty:
                return None
            # 按 oi 降序取主力合约（oi 最大）
            if "oi" in df.columns:
                df = df.sort_values("oi", ascending=False)
            return float(df.iloc[0]["close"])
        except Exception as e:
            logger.debug("获取期货价格失败 %s %s: %s", futures_code, trade_date, e)
            return None
