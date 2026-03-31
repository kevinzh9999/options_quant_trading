"""
tushare_client.py
-----------------
职责：封装 Tushare Pro API，提供结构化数据拉取接口。

特性：
- 所有请求通过 _call_api 统一入口：sleep 频率控制 + 重试 + 日志
- 返回列名与 schemas.py 字段名完全一致
- 日期字段统一为字符串 YYYYMMDD
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from datetime import date
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Tushare 原始列名 → 本地 schemas 字段名
_FUTURES_DAILY_RENAME: dict[str, str] = {
    "vol": "volume",
}
_FUTURES_MIN_RENAME: dict[str, str] = {
    "trade_time": "datetime",
    "vol":        "volume",
}
_OPTIONS_DAILY_RENAME: dict[str, str] = {
    "vol":       "volume",
    "opt_code":  "underlying_code",
}
_OPTIONS_CONTRACTS_RENAME: dict[str, str] = {
    "opt_code":      "underlying_code",
    "maturity_date": "expire_date",
}
_TRADE_CAL_RENAME: dict[str, str] = {
    "cal_date": "trade_date",
}
_INDEX_DAILY_RENAME: dict[str, str] = {
    "vol":    "volume",
    "amount": "amount",
}

# 期货日线最终保留列（与 futures_daily 表对齐）
_FUTURES_DAILY_COLS = [
    "ts_code", "trade_date", "open", "high", "low", "close",
    "volume", "oi", "settle", "pre_close", "pre_settle",
]
_FUTURES_MIN_COLS = [
    "ts_code", "datetime", "open", "high", "low", "close", "volume",
]
_OPTIONS_DAILY_COLS = [
    "ts_code", "trade_date", "exchange", "underlying_code",
    "exercise_price", "call_put", "expire_date",
    "close", "settle", "volume", "oi", "pre_close", "pre_settle",
]
_OPTIONS_CONTRACTS_COLS = [
    "ts_code", "exchange", "underlying_code", "exercise_price", "call_put",
    "expire_date", "list_date", "delist_date", "contract_unit", "exercise_type",
]
_TRADE_CAL_COLS = [
    "exchange", "trade_date", "is_open", "pretrade_date",
]
_INDEX_DAILY_COLS = [
    "ts_code", "trade_date", "open", "high", "low", "close", "volume", "amount",
]


class TushareClient:
    """
    Tushare Pro API 封装。

    Parameters
    ----------
    token : str
        Tushare Pro API Token
    max_retry : int
        API 调用失败时最大重试次数
    sleep_interval : float
        每次 API 调用前的等待秒数（防止频率限制）
    """

    def __init__(
        self,
        token: str,
        max_retry: int = 3,
        sleep_interval: float = 0.5,
    ) -> None:
        self.token = token
        self.max_retry = max_retry
        self.sleep_interval = sleep_interval
        self._api = None  # 懒加载

    # ------------------------------------------------------------------
    # 内部：API 实例 & 统一调用入口
    # ------------------------------------------------------------------

    def _get_api(self):
        """懒加载 tushare pro_api 实例（仅在首次调用时初始化）"""
        if self._api is None:
            import tushare as ts  # noqa: F401 — optional dependency
            ts.set_token(self.token)
            self._api = ts.pro_api()
            logger.info("Tushare API 已初始化")
        return self._api

    def _call_api(self, api_name: str, _timeout: int = 30, **kwargs) -> pd.DataFrame:
        """
        统一 API 调用入口：sleep 频率控制 + 单次调用超时 + 重试 + 日志。

        Parameters
        ----------
        api_name : str
            Tushare API 方法名，如 "fut_daily"
        _timeout : int
            单次 API 调用超时秒数，默认 30 秒
        **kwargs
            传递给对应 API 的参数

        Returns
        -------
        pd.DataFrame
            API 原始返回数据；None 结果转为空 DataFrame。
        """
        api = self._get_api()
        fn = getattr(api, api_name)
        last_exc: Exception = RuntimeError("未知错误")
        for attempt in range(1, self.max_retry + 1):
            try:
                time.sleep(self.sleep_interval)
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _ex:
                    future = _ex.submit(fn, **kwargs)
                    try:
                        result = future.result(timeout=_timeout)
                    except concurrent.futures.TimeoutError:
                        raise TimeoutError(
                            f"Tushare {api_name} 调用超时（{_timeout}s），参数={kwargs}"
                        )
                rows = 0 if result is None else len(result)
                logger.info(
                    "Tushare %s 调用成功，参数=%s，返回 %d 行",
                    api_name, kwargs, rows,
                )
                return result if result is not None else pd.DataFrame()
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Tushare %s 第 %d/%d 次失败: %s",
                    api_name, attempt, self.max_retry, exc,
                )
                if attempt < self.max_retry:
                    time.sleep(2 ** attempt)  # 指数退避
        raise last_exc

    @staticmethod
    def _rename_and_select(
        df: pd.DataFrame,
        rename: dict[str, str],
        cols: list[str],
    ) -> pd.DataFrame:
        """重命名列，再按 cols 裁剪/补全（缺失列填 NaN）。"""
        # 只重命名实际存在的列
        actual_rename = {k: v for k, v in rename.items() if k in df.columns}
        if actual_rename:
            df = df.rename(columns=actual_rename)
        return df.reindex(columns=cols)

    # ------------------------------------------------------------------
    # 期货数据
    # ------------------------------------------------------------------

    def get_futures_daily(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        获取期货日线行情（fut_daily 接口）。

        Returns
        -------
        pd.DataFrame
            列：ts_code, trade_date, open, high, low, close, volume, oi,
                settle, pre_close, pre_settle；按 trade_date 升序。
        """
        df = self._call_api(
            "fut_daily",
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
        )
        if df.empty:
            return pd.DataFrame(columns=_FUTURES_DAILY_COLS)
        df = self._rename_and_select(df, _FUTURES_DAILY_RENAME, _FUTURES_DAILY_COLS)
        return df.sort_values("trade_date").reset_index(drop=True)

    def get_futures_min(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        freq: str = "5min",
    ) -> pd.DataFrame:
        """
        获取期货分钟线行情（ft_mins 接口，需额外积分）。

        Notes
        -----
        - `trade_time` → `datetime`，保持 "YYYY-MM-DD HH:MM:SS" 格式
        - 分钟数据量大，建议按天拆分调用

        Returns
        -------
        pd.DataFrame
            列：ts_code, datetime, open, high, low, close, volume；按 datetime 升序。
        """
        df = self._call_api(
            "ft_mins",
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
            freq=freq,
        )
        if df.empty:
            return pd.DataFrame(columns=_FUTURES_MIN_COLS)
        df = self._rename_and_select(df, _FUTURES_MIN_RENAME, _FUTURES_MIN_COLS)
        return df.sort_values("datetime").reset_index(drop=True)

    def get_futures_mapping(self, exchange: str = "CFFEX") -> pd.DataFrame:
        """
        获取期货主力/连续合约映射关系（fut_mapping 接口）。

        Returns
        -------
        pd.DataFrame
            列：ts_code（具体合约）, trade_date, mapping_ts_code（主力标识）
        用途：确定每天的主力合约，构建主力连续数据。
        """
        df = self._call_api("fut_mapping", exchange=exchange)
        if df.empty:
            return pd.DataFrame(columns=["ts_code", "trade_date", "mapping_ts_code"])
        cols = ["ts_code", "trade_date", "mapping_ts_code"]
        return df.reindex(columns=cols).sort_values(
            ["ts_code", "trade_date"]
        ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 期权数据
    # ------------------------------------------------------------------

    def get_options_daily(
        self,
        exchange: str,
        trade_date: str,
    ) -> pd.DataFrame:
        """
        获取期权日线行情（opt_daily 接口）。

        Parameters
        ----------
        exchange : str
            交易所代码：CFFEX（股指期权）/ SSE（上交所 ETF 期权）/ SZSE
        trade_date : str
            交易日 YYYYMMDD

        Returns
        -------
        pd.DataFrame
            列：ts_code, trade_date, exchange, underlying_code, exercise_price,
                call_put, expire_date, close, settle, volume, oi, pre_close, pre_settle
        """
        df = self._call_api(
            "opt_daily",
            exchange=exchange,
            trade_date=trade_date,
        )
        if df.empty:
            return pd.DataFrame(columns=_OPTIONS_DAILY_COLS)
        df = self._rename_and_select(df, _OPTIONS_DAILY_RENAME, _OPTIONS_DAILY_COLS)
        return df.sort_values(["ts_code"]).reset_index(drop=True)

    def get_options_contracts(
        self,
        exchange: str = "CFFEX",
        underlying: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        获取期权合约基本信息（opt_basic 接口）。

        Notes
        -----
        - 包含历史已退市合约，建议一次性拉取后定期更新
        - `opt_code` → `underlying_code`，`maturity_date` → `expire_date`

        Returns
        -------
        pd.DataFrame
            列：ts_code, exchange, underlying_code, exercise_price, call_put,
                expire_date, list_date, delist_date, contract_unit, exercise_type
        """
        kwargs: dict = {"exchange": exchange}
        if underlying:
            kwargs["underlying"] = underlying
        df = self._call_api("opt_basic", **kwargs)
        if df.empty:
            return pd.DataFrame(columns=_OPTIONS_CONTRACTS_COLS)
        df = self._rename_and_select(df, _OPTIONS_CONTRACTS_RENAME, _OPTIONS_CONTRACTS_COLS)
        return df

    # ------------------------------------------------------------------
    # 商品期货
    # ------------------------------------------------------------------

    def get_commodity_daily(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        获取商品期货日线行情（fut_daily 接口，与股指期货同一接口）。

        交易所代码：SHFE / DCE / CZCE / INE / GFEX

        Returns
        -------
        pd.DataFrame
            与 get_futures_daily 结构相同。
        """
        return self.get_futures_daily(ts_code, start_date, end_date)

    # ------------------------------------------------------------------
    # 交易日历
    # ------------------------------------------------------------------

    def get_trade_calendar(
        self,
        exchange: str = "CFFEX",
        start_date: str = "20100101",
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        获取交易日历（trade_cal 接口）。

        Notes
        -----
        - `cal_date` → `trade_date`
        - end_date 默认为今天

        Returns
        -------
        pd.DataFrame
            列：exchange, trade_date, is_open, pretrade_date；按 trade_date 升序。
        """
        if end_date is None:
            end_date = date.today().strftime("%Y%m%d")
        df = self._call_api(
            "trade_cal",
            exchange=exchange,
            start_date=start_date,
            end_date=end_date,
        )
        if df.empty:
            return pd.DataFrame(columns=_TRADE_CAL_COLS)
        df = self._rename_and_select(df, _TRADE_CAL_RENAME, _TRADE_CAL_COLS)
        return df.sort_values("trade_date").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 指数
    # ------------------------------------------------------------------

    def get_index_daily(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        获取指数日线行情（index_daily 接口）。

        用途：计算已实现波动率时作为标的替代（缺少期货分钟线时）

        Returns
        -------
        pd.DataFrame
            列：ts_code, trade_date, open, high, low, close, volume；按 trade_date 升序。
        """
        df = self._call_api(
            "index_daily",
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date,
        )
        if df.empty:
            return pd.DataFrame(columns=_INDEX_DAILY_COLS)
        df = self._rename_and_select(df, _INDEX_DAILY_RENAME, _INDEX_DAILY_COLS)
        return df.sort_values("trade_date").reset_index(drop=True)
