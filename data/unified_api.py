"""
unified_api.py
--------------
职责：统一数据访问接口，屏蔽底层数据源差异。
上层模块（models/signals/risk 等）只依赖本模块，不直接调用 TushareClient 或 TqClient。

数据读取策略：
1. 优先从本地数据库（SQLite）读取
2. 若本地数据不足（缺失日期），自动从 Tushare 补充历史数据
3. 实时分钟数据可从天勤 TqSdk 补充（当 TqClient 已连接时）

symbol 参数支持多种格式，内部 _normalize_symbol() 统一转换为 Tushare ts_code 格式。
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd

from config import Config
from data.sources.tq_client import TqClient
from data.sources.tushare_client import TushareClient
from data.storage.db_manager import DBManager

logger = logging.getLogger(__name__)

# ======================================================================
# 交易所代码映射（含股票/ETF 市场，比 TqClient 更全）
# ======================================================================

# Tushare 后缀 -> TqSdk 前缀
_TS_TO_TQ_EXCHANGE: dict[str, str] = {
    "CFX": "CFFEX",
    "SHF": "SHFE",
    "DCE": "DCE",
    "ZCE": "CZCE",
    "INE": "INE",
    "GFX": "GFEX",
    "SH":  "SSE",
    "SZ":  "SZSE",
    "BJ":  "BSE",
}
# TqSdk 前缀 -> Tushare 后缀
_TQ_TO_TS_EXCHANGE: dict[str, str] = {v: k for k, v in _TS_TO_TQ_EXCHANGE.items()}

# 已知 TqSdk 交易所前缀
_TQ_PREFIXES = frozenset(_TQ_TO_TS_EXCHANGE.keys())
# 已知 Tushare 交易所后缀
_TS_SUFFIXES = frozenset(_TS_TO_TQ_EXCHANGE.keys())

# 品种代码 -> Tushare 交易所后缀（用于裸代码解析）
_PRODUCT_TO_TS_SUFFIX: dict[str, str] = {}
for _p in ["IF", "IH", "IC", "IM", "IO", "MO", "HO", "TF", "T", "TS"]:
    _PRODUCT_TO_TS_SUFFIX[_p] = "CFX"
for _p in ["RB", "HC", "CU", "AL", "ZN", "PB", "NI", "SN", "AU", "AG",
           "SP", "BU", "RU", "FU", "WR"]:
    _PRODUCT_TO_TS_SUFFIX[_p] = "SHF"
for _p in ["I", "J", "JM", "M", "Y", "P", "C", "CS", "A", "B",
           "BB", "FB", "L", "V", "PP", "EB", "EG", "PG", "RR", "JD", "LH"]:
    _PRODUCT_TO_TS_SUFFIX[_p] = "DCE"
for _p in ["SR", "CF", "OI", "TA", "MA", "FG", "RM", "ZC", "AP", "CJ",
           "PF", "SA", "UR", "PK", "WH", "PM", "RS", "JR", "LR", "SF", "SM"]:
    _PRODUCT_TO_TS_SUFFIX[_p] = "ZCE"
for _p in ["SC", "NR", "LU", "BC"]:
    _PRODUCT_TO_TS_SUFFIX[_p] = "INE"
for _p in ["LC", "SI"]:
    _PRODUCT_TO_TS_SUFFIX[_p] = "GFX"

# 常用指数代码 -> Tushare ts_code
_INDEX_CODE_TO_TS: dict[str, str] = {
    "000001": "000001.SH",   # 上证综指
    "000016": "000016.SH",   # 上证50
    "000300": "000300.SH",   # 沪深300
    "000905": "000905.SH",   # 中证500
    "000852": "000852.SH",   # 中证1000
    "399001": "399001.SZ",   # 深证成指
    "399006": "399006.SZ",   # 创业板指
}

# 期权标的 -> Tushare API 交易所参数
_UNDERLYING_TO_EXCHANGE: dict[str, str] = {
    "IO": "CFFEX",
    "MO": "CFFEX",
    "HO": "CFFEX",
}

# freq 字符串 -> 秒数
_FREQ_TO_SECONDS: dict[str, int] = {
    "1min": 60, "5min": 300, "15min": 900, "30min": 1800, "60min": 3600,
}


class UnifiedDataAPI:
    """
    统一数据 API。

    数据获取策略：
    1. 历史数据 -> 本地 SQLite 优先
    2. 本地缺失 -> Tushare 补充并存入本地
    3. 近期分钟数据 -> 可选 TqSdk 补充

    Parameters
    ----------
    config : Config
        系统配置对象
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.db = DBManager(config.db_path)
        self.tushare = TushareClient(config.tushare_token)
        self._tq: Optional[TqClient] = None

    # ------------------------------------------------------------------
    # symbol 格式标准化
    # ------------------------------------------------------------------

    def _normalize_symbol(self, symbol: str) -> str:
        """
        将任意格式的合约代码统一转换为 Tushare ts_code 格式。

        支持的输入格式：
        - "IF"            -> "IF.CFX"    (CFFEX 主力/连续)
        - "IF2406"        -> "IF2406.CFX"  (具体合约)
        - "CFFEX.IF2406"  -> "IF2406.CFX"  (TqSdk 格式)
        - "IF2406.CFX"    -> "IF2406.CFX"  (Tushare 格式，原样返回)
        - "IO2406-C-3800.CFX" -> 原样返回（已是 TS 格式）
        """
        if "." in symbol:
            # 取最后一个 "." 后的部分作为后缀
            last_dot = symbol.rfind(".")
            suffix = symbol[last_dot + 1:]
            if suffix in _TS_SUFFIXES:
                return symbol  # 已是 Tushare 格式
            # 取第一个 "." 前的部分作为前缀
            first_dot = symbol.index(".")
            prefix = symbol[:first_dot]
            if prefix in _TQ_PREFIXES:
                # TqSdk 格式 -> Tushare 格式
                code = symbol[first_dot + 1:]
                ts_suffix = _TQ_TO_TS_EXCHANGE.get(prefix, prefix)
                return f"{code}.{ts_suffix}"
            return symbol  # 未知格式，原样返回

        # 无点号：裸代码，提取品种字母部分
        m = re.match(r"^([A-Za-z]+)", symbol)
        if not m:
            return symbol  # 纯数字等无法识别的格式
        product = m.group(1).upper()
        ts_suffix = _PRODUCT_TO_TS_SUFFIX.get(product)
        if ts_suffix is None:
            logger.warning("未知品种 %r，无法确定交易所", product)
            return symbol
        return f"{symbol.upper()}.{ts_suffix}"

    # ------------------------------------------------------------------
    # 期货日线
    # ------------------------------------------------------------------

    def get_futures_daily(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        auto_download: bool = True,
    ) -> pd.DataFrame:
        """
        获取期货日线数据（本地优先，自动补充缺失）。

        Parameters
        ----------
        symbol : str
            品种代码，支持多格式：IF / IF2406 / CFFEX.IF2406 / IF2406.CFX
        start_date : str
            起始日期 YYYYMMDD
        end_date : str
            结束日期 YYYYMMDD
        auto_download : bool
            本地缺失时是否自动从 Tushare 下载

        Returns
        -------
        pd.DataFrame
            列：ts_code, trade_date, open, high, low, close, volume, oi, settle
            按 trade_date 升序排列
        """
        ts_code = self._normalize_symbol(symbol)
        df = self.db.get_futures_daily(ts_code, start_date, end_date)

        if auto_download and self.config.tushare_token:
            local_latest = self.db.get_latest_date("futures_daily", ts_code)
            fill_start = start_date
            if local_latest and local_latest >= end_date:
                return df  # 本地已完整覆盖
            if local_latest and local_latest >= start_date:
                fill_start = _next_date(local_latest)

            try:
                remote = self.tushare.get_futures_daily(ts_code, fill_start, end_date)
                if not remote.empty:
                    self.db.upsert_dataframe("futures_daily", remote)
                    df = self.db.get_futures_daily(ts_code, start_date, end_date)
            except Exception as exc:
                logger.warning("Tushare 期货日线补充失败 [%s]: %s", ts_code, exc)

        return df

    # ------------------------------------------------------------------
    # 期货分钟线
    # ------------------------------------------------------------------

    def get_futures_min(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        freq: str = "5min",
    ) -> pd.DataFrame:
        """
        获取期货分钟线数据。

        数据源优先级：本地 SQLite -> Tushare -> TqSdk（当已连接时）

        Parameters
        ----------
        symbol : str
            品种代码（同 get_futures_daily，支持多格式）
        start_date : str
            起始日期 YYYYMMDD
        end_date : str
            结束日期 YYYYMMDD
        freq : str
            频率 "1min" / "5min" / "15min" / "30min" / "60min"

        Returns
        -------
        pd.DataFrame
            列：ts_code, datetime, open, high, low, close, volume
            按 datetime 升序排列
        """
        ts_code = self._normalize_symbol(symbol)
        df = self.db.get_futures_min(ts_code, start_date, end_date)

        if df.empty and self.config.tushare_token:
            try:
                remote = self.tushare.get_futures_min(ts_code, start_date, end_date, freq=freq)
                if not remote.empty:
                    self.db.upsert_dataframe("futures_min", remote)
                    df = self.db.get_futures_min(ts_code, start_date, end_date)
            except Exception as exc:
                logger.warning("Tushare 分钟线补充失败 [%s]: %s", ts_code, exc)

        # TqSdk 补充（仅当已连接时）
        if df.empty and self._tq is not None and self._tq._api is not None:
            try:
                tq_symbol = TqClient.convert_symbol_ts_to_tq(ts_code)
                seconds = _FREQ_TO_SECONDS.get(freq, 300)
                klines = self._tq.get_kline(tq_symbol, seconds)
                if not klines.empty:
                    remote = _tq_kline_to_min_df(klines, ts_code)
                    start_dt = f"{start_date[:4]}-{start_date[4:6]}-{start_date[6:]} 00:00:00"
                    end_dt = f"{end_date[:4]}-{end_date[4:6]}-{end_date[6:]} 23:59:59"
                    remote = remote[
                        (remote["datetime"] >= start_dt) & (remote["datetime"] <= end_dt)
                    ]
                    if not remote.empty:
                        self.db.upsert_dataframe("futures_min", remote)
                        df = self.db.get_futures_min(ts_code, start_date, end_date)
            except Exception as exc:
                logger.warning("TqSdk 分钟线补充失败 [%s]: %s", ts_code, exc)

        return df

    # ------------------------------------------------------------------
    # 期权数据
    # ------------------------------------------------------------------

    def get_options_daily(
        self,
        underlying: str,
        trade_date: str,
        call_put: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        获取期权日线数据（单交易日快照）。

        Parameters
        ----------
        underlying : str
            标的代码，如 IO / MO / 510050
        trade_date : str
            交易日 YYYYMMDD
        call_put : str, optional
            "C" 认购 / "P" 认沽；None 返回全部

        Returns
        -------
        pd.DataFrame
            列：ts_code, trade_date, exchange, underlying_code,
                exercise_price, call_put, expire_date, close, settle, volume, oi
        """
        df = self.db.get_options_daily(underlying, trade_date, call_put)

        if df.empty and self.config.tushare_token:
            exchange = _get_exchange_for_underlying(underlying)
            try:
                remote = self.tushare.get_options_daily(
                    exchange=exchange, trade_date=trade_date
                )
                if not remote.empty:
                    self.db.upsert_dataframe("options_daily", remote)
                    df = self.db.get_options_daily(underlying, trade_date, call_put)
            except Exception as exc:
                logger.warning("Tushare 期权日线补充失败 [%s %s]: %s",
                               underlying, trade_date, exc)

        return df

    def get_options_chain(
        self,
        underlying: str,
        trade_date: str,
        expire_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        获取期权链（合约基本信息 + 当日行情合并）。

        Parameters
        ----------
        underlying : str
            标的代码，如 IO / MO
        trade_date : str
            交易日 YYYYMMDD
        expire_date : str, optional
            指定到期日；None 返回全部到期月份

        Returns
        -------
        pd.DataFrame
            列：ts_code, exercise_price, call_put, expire_date,
                contract_unit, exercise_type, close, settle, volume, oi
        """
        return self.db.get_options_chain(underlying, trade_date, expire_date)

    def get_options_contracts(
        self,
        underlying: Optional[str] = None,
        active_on: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        获取期权合约基本信息。

        Parameters
        ----------
        underlying : str, optional
            标的代码
        active_on : str, optional
            指定日期（YYYYMMDD），只返回该日已上市且未到期的合约

        Returns
        -------
        pd.DataFrame
            列：ts_code, exchange, underlying_code, exercise_price, call_put,
                expire_date, list_date, delist_date
        """
        conditions: list[str] = []
        params: list = []
        if underlying:
            conditions.append("underlying_code LIKE ?")
            params.append(f"%{underlying}%")
        if active_on:
            conditions.append("list_date <= ?")
            params.append(active_on)
            conditions.append("delist_date >= ?")
            params.append(active_on)

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        df = self.db.query(
            f"SELECT * FROM options_contracts {where}"
            " ORDER BY expire_date, exercise_price",
            tuple(params) if params else None,
        )

        if df.empty and self.config.tushare_token:
            try:
                remote = self.tushare.get_options_contracts(underlying=underlying)
                if not remote.empty:
                    self.db.upsert_dataframe("options_contracts", remote)
                    df = self.db.query(
                        f"SELECT * FROM options_contracts {where}"
                        " ORDER BY expire_date, exercise_price",
                        tuple(params) if params else None,
                    )
            except Exception as exc:
                logger.warning("Tushare 期权合约信息补充失败: %s", exc)

        return df

    # ------------------------------------------------------------------
    # 交易日历
    # ------------------------------------------------------------------

    def get_trade_calendar(
        self,
        exchange: str = "CFFEX",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        获取交易日历（仅返回 is_open=1 的交易日）。

        Parameters
        ----------
        exchange : str
            交易所代码，默认 CFFEX
        start_date : str, optional
            起始日期 YYYYMMDD
        end_date : str, optional
            结束日期 YYYYMMDD

        Returns
        -------
        pd.DataFrame
            列：exchange, trade_date, is_open, pretrade_date；按 trade_date 升序
        """
        df = self.db.get_trade_calendar(exchange, start_date, end_date)

        if df.empty and self.config.tushare_token:
            try:
                remote = self.tushare.get_trade_calendar(
                    exchange=exchange,
                    start_date=start_date or "20100101",
                    end_date=end_date,
                )
                if not remote.empty:
                    self.db.upsert_dataframe("trade_calendar", remote)
                    df = self.db.get_trade_calendar(exchange, start_date, end_date)
            except Exception as exc:
                logger.warning("Tushare 交易日历补充失败: %s", exc)

        return df

    def get_latest_trading_date(self, exchange: str = "CFFEX") -> str:
        """
        获取最近一个交易日的日期。

        优先从本地 trade_calendar 查询，缺失时回落 Tushare。

        Returns
        -------
        str
            最近交易日，YYYYMMDD 格式
        """
        today = date.today().strftime("%Y%m%d")
        df = self.db.query(
            "SELECT trade_date FROM trade_calendar"
            " WHERE exchange=? AND is_open=1 AND trade_date<=?"
            " ORDER BY trade_date DESC LIMIT 1",
            (exchange, today),
        )
        if not df.empty:
            return str(df.iloc[0]["trade_date"])

        # 回落 Tushare：拉取近 30 天日历
        if self.config.tushare_token:
            lookback_start = _offset_date(today, -30)
            try:
                cal = self.tushare.get_trade_calendar(
                    exchange=exchange,
                    start_date=lookback_start,
                    end_date=today,
                )
                if not cal.empty:
                    self.db.upsert_dataframe("trade_calendar", cal)
                    open_dates = cal[cal["is_open"] == 1]["trade_date"]
                    if not open_dates.empty:
                        return str(open_dates.iloc[-1])
            except Exception as exc:
                logger.warning("获取最近交易日失败: %s", exc)

        return today  # 最终兜底

    # ------------------------------------------------------------------
    # 指数日线
    # ------------------------------------------------------------------

    def get_index_daily(
        self,
        index_code: str,
        start_date: str,
        end_date: str,
        save_to_db: bool = False,
    ) -> pd.DataFrame:
        """
        获取指数日线数据。

        优先从本地 index_daily 表读取；如果 save_to_db=True 则同时写入数据库。

        Parameters
        ----------
        index_code : str
            指数代码，支持裸代码（"000300"）或带后缀（"000300.SH"）
            常用：000300(沪深300) / 000016(上证50) / 000905(中证500) / 000852(中证1000)
        save_to_db : bool
            True = 从 Tushare 获取后写入 index_daily 表（用于增量更新）

        Returns
        -------
        pd.DataFrame
            列：ts_code, trade_date, open, high, low, close, volume, amount
            按 trade_date 升序排列
        """
        ts_code = _normalize_index_code(index_code)

        # 先尝试从本地 DB 读取
        if self.db is not None:
            try:
                df_local = self.db.query_df(
                    f"SELECT * FROM index_daily "
                    f"WHERE ts_code='{ts_code}' "
                    f"AND trade_date >= '{start_date}' "
                    f"AND trade_date <= '{end_date}' "
                    f"ORDER BY trade_date ASC"
                )
                if df_local is not None and not df_local.empty:
                    return df_local
            except Exception:
                pass  # 表可能不存在，继续向 Tushare 请求

        if not self.config.tushare_token:
            logger.debug("无 Tushare token，跳过指数日线获取")
            return pd.DataFrame()

        try:
            df = self.tushare.get_index_daily(ts_code, start_date, end_date)
        except Exception as exc:
            logger.warning("Tushare 指数日线获取失败 [%s]: %s", ts_code, exc)
            return pd.DataFrame()

        if save_to_db and self.db is not None and not df.empty:
            try:
                self.db.upsert_dataframe("index_daily", df)
            except Exception as exc:
                logger.warning("写入 index_daily 失败 [%s]: %s", ts_code, exc)

        return df

    # ------------------------------------------------------------------
    # 数据预加载
    # ------------------------------------------------------------------

    def ensure_data_available(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        data_type: str = "futures_daily",
    ) -> bool:
        """
        确认指定数据在本地可用，不可用则自动下载。

        Parameters
        ----------
        symbol : str
            品种代码（同 get_futures_daily，支持多格式）
        start_date : str
            起始日期 YYYYMMDD
        end_date : str
            结束日期 YYYYMMDD
        data_type : str
            数据类型："futures_daily" / "futures_min" / "options_daily"

        Returns
        -------
        bool
            本地数据可用（非空）则返回 True
        """
        ts_code = self._normalize_symbol(symbol)

        if data_type == "futures_daily":
            df = self.get_futures_daily(ts_code, start_date, end_date, auto_download=True)
        elif data_type == "futures_min":
            df = self.get_futures_min(ts_code, start_date, end_date)
        elif data_type == "options_daily":
            # 对于期权日线，symbol 作为 underlying，start_date 作为 trade_date
            df = self.get_options_daily(ts_code, start_date)
        else:
            logger.warning("ensure_data_available: 未知数据类型 %r", data_type)
            return False

        available = not df.empty
        if not available:
            logger.warning("数据不可用: %s %s %s~%s", data_type, ts_code, start_date, end_date)
        return available

    # ------------------------------------------------------------------
    # 向后兼容：get_trade_dates / is_trade_date
    # ------------------------------------------------------------------

    def get_trade_dates(
        self,
        start_date: str,
        end_date: str,
        exchange: str = "SSE",
    ) -> list[str]:
        """返回指定区间内的交易日列表（is_open=1）"""
        df = self.get_trade_calendar(exchange, start_date, end_date)
        if df.empty:
            return []
        return df["trade_date"].tolist()

    def is_trade_date(self, date_str: str, exchange: str = "SSE") -> bool:
        """判断指定日期是否为交易日"""
        result = self.db.query_scalar(
            "SELECT is_open FROM trade_calendar WHERE exchange=? AND trade_date=?",
            [exchange, date_str],
        )
        if result is not None:
            return bool(result)
        dates = self.get_trade_dates(date_str, date_str, exchange)
        return date_str in dates

    # ------------------------------------------------------------------
    # 代码映射（静态方法，向后兼容）
    # ------------------------------------------------------------------

    @staticmethod
    def tushare_to_tq_symbol(ts_code: str) -> str:
        """Tushare ts_code -> TqSdk symbol (支持股票/ETF/期货/期权)"""
        if "." not in ts_code:
            return ts_code
        last_dot = ts_code.rfind(".")
        symbol = ts_code[:last_dot]
        suffix = ts_code[last_dot + 1:]
        tq_exchange = _TS_TO_TQ_EXCHANGE.get(suffix, suffix)
        return f"{tq_exchange}.{symbol}"

    @staticmethod
    def tq_to_tushare_symbol(tq_symbol: str) -> str:
        """TqSdk symbol -> Tushare ts_code"""
        if "." not in tq_symbol:
            return tq_symbol
        first_dot = tq_symbol.index(".")
        tq_exchange = tq_symbol[:first_dot]
        symbol = tq_symbol[first_dot + 1:]
        ts_suffix = _TQ_TO_TS_EXCHANGE.get(tq_exchange, tq_exchange)
        return f"{symbol}.{ts_suffix}"

    # ------------------------------------------------------------------
    # 天勤客户端懒加载
    # ------------------------------------------------------------------

    def _get_tq_client(self) -> TqClient:
        """懒加载并连接天勤客户端"""
        if self._tq is None:
            tq_cfg = self.config.get_tq_config()
            self._tq = TqClient(
                auth_account=tq_cfg["account"],
                auth_password=tq_cfg["password"],
            )
            self._tq.connect()
        return self._tq


# ======================================================================
# 模块级工具函数
# ======================================================================

def _next_date(date_str: str) -> str:
    """返回 YYYYMMDD 格式日期的次日"""
    d = datetime.strptime(date_str, "%Y%m%d") + timedelta(days=1)
    return d.strftime("%Y%m%d")


def _offset_date(date_str: str, days: int) -> str:
    """返回 YYYYMMDD 日期偏移 days 天后的日期（days 可为负数）"""
    d = datetime.strptime(date_str, "%Y%m%d") + timedelta(days=days)
    return d.strftime("%Y%m%d")


def _get_exchange_for_underlying(underlying: str) -> str:
    """根据期权标的代码推断 Tushare API 所需的交易所参数"""
    upper = underlying.upper()
    if upper in _UNDERLYING_TO_EXCHANGE:
        return _UNDERLYING_TO_EXCHANGE[upper]
    if underlying.startswith(("51", "58")):   # 上交所 ETF
        return "SSE"
    if underlying.startswith(("15", "16", "17")):  # 深交所 ETF
        return "SZSE"
    return "CFFEX"  # 默认


def _normalize_index_code(index_code: str) -> str:
    """裸指数代码 -> Tushare ts_code（如 '000300' -> '000300.SH'）"""
    if "." in index_code:
        return index_code
    return _INDEX_CODE_TO_TS.get(index_code, f"{index_code}.SH")


def _tq_kline_to_min_df(klines: pd.DataFrame, ts_code: str) -> pd.DataFrame:
    """TqSdk K 线 DataFrame -> futures_min 表格式"""
    df = klines.copy()
    if "datetime" in df.columns and pd.api.types.is_datetime64_any_dtype(df["datetime"]):
        df["datetime"] = df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df["ts_code"] = ts_code
    cols = ["ts_code", "datetime", "open", "high", "low", "close", "volume"]
    return df.reindex(columns=cols).dropna(subset=["datetime"])
