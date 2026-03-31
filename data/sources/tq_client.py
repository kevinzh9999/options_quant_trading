"""
tq_client.py
------------
职责：封装天勤 TqSdk API，提供：
- 历史 K 线批量获取（回测/数据补充用）
- 实时行情订阅（盘中监控用）
- 期权链行情快照获取

注意：TqSdk 使用异步事件循环，该模块做同步化封装供策略层调用。
"""

from __future__ import annotations

import logging
import time
from typing import Callable

import pandas as pd

logger = logging.getLogger(__name__)

# 单次 wait_update 的最长等待秒数。
# 收盘后（15:00+）交易所不再推送行情，不加 deadline 会永久阻塞。
_TQ_WAIT_TIMEOUT: int = 30


def _wait(api, timeout: int = _TQ_WAIT_TIMEOUT) -> bool:
    """
    调用 api.wait_update(deadline=...) 并在超时时返回 False。

    TqSdk 的 deadline 参数为 Unix 时间戳（float）。
    正常返回 True 表示收到了更新；False 表示超时未收到更新。
    """
    return api.wait_update(deadline=time.time() + timeout)

# 交易所代码映射：Tushare 后缀 ↔ TqSdk 前缀
# ts_suffix -> tq_prefix
_TS_TO_TQ_EXCHANGE: dict[str, str] = {
    "CFX": "CFFEX",
    "SHF": "SHFE",
    "DCE": "DCE",
    "ZCE": "CZCE",
    "INE": "INE",
    "GFX": "GFEX",
}
# tq_prefix -> ts_suffix
_TQ_TO_TS_EXCHANGE: dict[str, str] = {v: k for k, v in _TS_TO_TQ_EXCHANGE.items()}

# K 线最终保留列
_KLINE_COLS = [
    "datetime", "open", "high", "low", "close", "volume", "open_oi", "close_oi",
]

# 行情快照字段
_QUOTE_FIELDS = (
    "last_price", "bid_price1", "ask_price1", "bid_volume1", "ask_volume1",
    "volume", "open_interest", "upper_limit", "lower_limit",
    "highest", "lowest", "open", "close",
    "strike_price", "expire_datetime", "option_class",
)


class TqClient:
    """
    天勤 TqSdk 封装客户端。

    Parameters
    ----------
    auth_account : str
        天勤平台账户（邮箱），用于 TqAuth 行情权限认证
    auth_password : str
        天勤平台密码，用于 TqAuth
    broker_id : str, optional
        期货公司代码，宏源期货为 "H宏源期货"（TqAccount broker_id）。
        提供时启用实盘 TqAccount 连接；省略时仅使用 TqAuth（模拟/行情模式）。
    account_id : str, optional
        期货公司资金账号（TqAccount account_id）
    broker_password : str, optional
        期货账户密码（TqAccount password，对应环境变量 TQ_BROKER_PASSWORD）

    Notes
    -----
    TqSdk 实盘连接正确写法（来自官方文档）::

        TqApi(
            TqAccount("H宏源期货", account_id, broker_password),
            auth=TqAuth(auth_account, auth_password),
        )

    仅行情/回测时（无需实盘账户）::

        TqApi(auth=TqAuth(auth_account, auth_password))
    """

    def __init__(
        self,
        auth_account: str,
        auth_password: str,
        broker_id: str = "",
        account_id: str = "",
        broker_password: str = "",
    ) -> None:
        self.auth_account = auth_account
        self.auth_password = auth_password
        self.broker_id = broker_id
        self.account_id = account_id
        self.broker_password = broker_password
        self._api = None  # tqsdk.TqApi 实例，按需初始化

    # ------------------------------------------------------------------
    # 连接管理
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """
        初始化 TqApi 连接（已连接时跳过）。

        - 若提供了 broker_id / account_id / broker_password，使用实盘模式：
          TqApi(TqAccount(broker_id, account_id, broker_password), auth=TqAuth(...))
        - 否则使用行情/模拟模式：
          TqApi(auth=TqAuth(auth_account, auth_password))

        建议在策略启动时调用一次，之后复用连接。
        """
        if self._api is not None:
            logger.debug("TqClient 已连接，跳过重复初始化")
            return
        try:
            from tqsdk import TqApi, TqAuth, TqAccount  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "tqsdk 未安装，请执行: pip install tqsdk\n"
                f"原始错误: {exc}"
            ) from exc

        auth = TqAuth(self.auth_account, self.auth_password)
        if self.broker_id and self.account_id and self.broker_password:
            # 实盘模式：TqAccount + TqAuth
            self._api = TqApi(
                TqAccount(self.broker_id, self.account_id, self.broker_password),
                auth=auth,
            )
            logger.info(
                "TqSdk 实盘已连接: broker=%s, account=%s***",
                self.broker_id, self.account_id[:4] if self.account_id else "",
            )
        else:
            # 行情/模拟模式：仅 TqAuth
            self._api = TqApi(auth=auth)
            logger.info("TqSdk 已连接（行情模式），账户: %s***", self.auth_account[:4])

    def disconnect(self) -> None:
        """关闭 TqApi 连接，释放资源"""
        if self._api is not None:
            self._api.close()
            self._api = None
            logger.info("TqSdk 已断开连接")

    def __enter__(self) -> "TqClient":
        self.connect()
        return self

    def __exit__(self, *args) -> None:
        self.disconnect()

    def _require_connected(self) -> None:
        if self._api is None:
            raise RuntimeError("TqClient 尚未连接，请先调用 connect()")

    # ------------------------------------------------------------------
    # 历史 K 线
    # ------------------------------------------------------------------

    def get_kline(
        self,
        symbol: str,
        duration_seconds: int,
        data_length: int = 8964,
    ) -> pd.DataFrame:
        """
        获取合约历史 K 线数据。

        Parameters
        ----------
        symbol : str
            天勤合约代码，如 CFFEX.IF2406 / CFFEX.IO2406-C-3800
        duration_seconds : int
            K 线周期秒数：60=1分钟, 300=5分钟, 86400=日线
        data_length : int
            获取数量，最大约 8964 根

        Returns
        -------
        pd.DataFrame
            列：datetime, open, high, low, close, volume, open_oi, close_oi
            datetime 为 pandas Timestamp，已转为 Asia/Shanghai 时区
        """
        self._require_connected()
        klines = self._api.get_kline_serial(symbol, duration_seconds, data_length)
        _wait(self._api)
        df = klines.copy()
        # TqSdk 返回 datetime 字段为纳秒时间戳，转换为本地时间
        if "datetime" in df.columns and pd.api.types.is_integer_dtype(df["datetime"]):
            df["datetime"] = (
                pd.to_datetime(df["datetime"], unit="ns", utc=True)
                .dt.tz_convert("Asia/Shanghai")
            )
        # close_oi 字段兼容
        if "close_oi" not in df.columns and "close_interest" in df.columns:
            df["close_oi"] = df["close_interest"]
        df = df.reindex(columns=_KLINE_COLS)
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    # 实时行情
    # ------------------------------------------------------------------

    def get_quote(self, symbol: str) -> dict:
        """
        获取合约最新快照行情。

        Parameters
        ----------
        symbol : str
            天勤合约代码

        Returns
        -------
        dict
            包含 last_price, bid_price1, ask_price1, volume, open_interest 等字段
        """
        self._require_connected()
        quote = self._api.get_quote(symbol)
        _wait(self._api)
        return {f: getattr(quote, f, None) for f in _QUOTE_FIELDS}

    def get_option_quotes(
        self,
        underlying: str,
        exchange: str,
    ) -> pd.DataFrame:
        """
        获取标的下所有期权合约的行情快照。

        Parameters
        ----------
        underlying : str
            标的代码，如 \"IO\"、\"510050\"
        exchange : str
            交易所 TqSdk 前缀，如 \"CFFEX\"、\"SSE\"

        Returns
        -------
        pd.DataFrame
            每行一个期权合约，包含 symbol 及行情快照字段
        """
        self._require_connected()
        # 获取交易所下所有合约列表，过滤出属于该标的的期权
        ins_list = self._api.query_options(
            underlying_symbol=f"{exchange}.{underlying}"
        )
        _wait(self._api)
        rows = []
        for symbol in ins_list:
            quote = self._api.get_quote(symbol)
            _wait(self._api)
            row = {"symbol": symbol}
            row.update({f: getattr(quote, f, None) for f in _QUOTE_FIELDS})
            rows.append(row)
        if not rows:
            cols = ["symbol"] + list(_QUOTE_FIELDS)
            return pd.DataFrame(columns=cols)
        return pd.DataFrame(rows)

    def subscribe_quotes(
        self,
        symbols: list[str],
        callback: Callable[[str, dict], None],
    ) -> None:
        """
        批量订阅合约行情，有更新时调用 callback（阻塞运行直到手动中断）。

        Parameters
        ----------
        symbols : list[str]
            天勤合约代码列表
        callback : Callable[[str, dict], None]
            行情更新回调，参数为 (symbol, quote_dict)

        Notes
        -----
        - 阻塞方法，适用于盘中监控场景
        - 通过 KeyboardInterrupt 退出循环
        """
        self._require_connected()
        quotes = {s: self._api.get_quote(s) for s in symbols}
        logger.info("已订阅 %d 个合约行情，开始监听...", len(symbols))
        try:
            while True:
                self._api.wait_update()
                for symbol, quote in quotes.items():
                    if self._api.is_changing(quote):
                        snapshot = {f: getattr(quote, f, None) for f in _QUOTE_FIELDS}
                        callback(symbol, snapshot)
        except KeyboardInterrupt:
            logger.info("行情订阅已停止")

    # ------------------------------------------------------------------
    # 代码映射（静态方法）
    # ------------------------------------------------------------------

    @staticmethod
    def convert_symbol_tq_to_ts(tq_symbol: str) -> str:
        """
        天勤代码 → Tushare 代码。

        Examples
        --------
        >>> TqClient.convert_symbol_tq_to_ts("CFFEX.IF2406")
        'IF2406.CFX'
        >>> TqClient.convert_symbol_tq_to_ts("SHFE.RBL4")
        'RBL4.SHF'
        """
        exchange, code = tq_symbol.split(".", 1)
        ts_suffix = _TQ_TO_TS_EXCHANGE.get(exchange, exchange)
        return f"{code}.{ts_suffix}"

    @staticmethod
    def convert_symbol_ts_to_tq(ts_symbol: str) -> str:
        """
        Tushare 代码 → 天勤代码。

        Examples
        --------
        >>> TqClient.convert_symbol_ts_to_tq("IF2406.CFX")
        'CFFEX.IF2406'
        >>> TqClient.convert_symbol_ts_to_tq("RBL4.SHF")
        'SHFE.RBL4'
        """
        # 期权代码含多个点号，如 IO2406-C-3800.CFX，只取最后一段作为后缀
        last_dot = ts_symbol.rfind(".")
        code = ts_symbol[:last_dot]
        ts_suffix = ts_symbol[last_dot + 1:]
        tq_exchange = _TS_TO_TQ_EXCHANGE.get(ts_suffix, ts_suffix)
        return f"{tq_exchange}.{code}"
