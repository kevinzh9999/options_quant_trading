"""
account_manager.py
------------------
职责：封装天勤 TqSdk 的账户和持仓数据读取，面向宏源期货实盘账户。
提供以下能力：
- 账户资金概况（权益、可用、保证金、风险度）
- 全量持仓列表（期货 + 期权）
- 期权持仓子集（含行权价、到期日、认购/认沽方向）
- 期货持仓子集
- 组合 Greeks 汇总（对接 models/greeks.py，待实现）
- 各持仓保证金明细

设计注意事项：
- 天勤 API 在非交易时段（如夜间、节假日）可能返回空数据或抛出异常，
  每个数据读取方法均有 try/except 保护，非交易时段降级为空列表/默认值。
- 本模块不持有 TqApi 连接的生命周期，连接由外部（TqClient 或主脚本）管理，
  通过 tq_api 参数注入。
"""

from __future__ import annotations

import logging
import math
import re
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    # 避免在未安装 tqsdk 的环境下导入失败（如单元测试）
    from tqsdk import TqApi

logger = logging.getLogger(__name__)

# 天勤合约代码中期权品种的前缀/后缀特征，用于区分期货和期权
_OPTION_EXCHANGES = {"CFFEX"}          # 中金所期权：IO、MO
_OPTION_CODE_SUFFIXES = ("-C-", "-P-") # 期权代码含行权价分隔符

# 期权代码解析正则：{交易所}.{品种}{到期月}-{C|P}-{行权价}
# 例：CFFEX.IO2406-C-3800
_OPTION_SYMBOL_RE = re.compile(
    r"^(?P<exchange>[A-Z]+)\."
    r"(?P<product>[A-Z]+)"
    r"(?P<expire_month>\d+)"
    r"-(?P<call_put>[CP])-"
    r"(?P<strike_price>[\d.]+)$"
)


def _is_option(symbol: str) -> bool:
    """
    判断天勤合约代码是否为期权。

    Parameters
    ----------
    symbol : str
        天勤合约代码，如 CFFEX.IO2406-C-3800 / CFFEX.IF2406

    Returns
    -------
    bool
        True 表示期权合约
    """
    return any(sep in symbol for sep in _OPTION_CODE_SUFFIXES)


def _safe_float(val: Any, default: float = 0.0) -> float:
    """将值转为 float，NaN 或转换失败时返回 default。"""
    try:
        f = float(val)
        return default if math.isnan(f) else f
    except (TypeError, ValueError):
        return default


class AccountManager:
    """
    宏源期货账户持仓管理器。

    封装天勤 TqSdk 账户相关 API，提供结构化的资金和持仓数据。
    上层模块（风控层、分析层、执行层）通过本类读取账户状态，
    无需直接操作 TqApi 对象。

    Parameters
    ----------
    tq_api : TqApi
        已连接的天勤 TqApi 实例，由调用方负责连接生命周期管理
    broker : str
        期货公司代码，宏源期货为 "H宏源期货"（TqAccount broker_id）
    account_id : str
        期货公司资金账号（TqAccount account_id）
    broker_password : str
        期货账户密码（TqAccount password，对应环境变量 TQ_BROKER_PASSWORD）

    Notes
    -----
    推荐通过 :meth:`connect_live` 类方法创建实例，该方法按文档正确组合
    ``TqAccount`` 和 ``TqAuth``::

        manager = AccountManager.connect_live(
            broker_id="H宏源期货",
            account_id="YOUR_ACCOUNT_ID",
            broker_password="YOUR_BROKER_PASSWORD",
            auth_account="your@email.com",
            auth_password="YOUR_TQ_PLATFORM_PASSWORD",
        )
    """

    def __init__(
        self,
        tq_api: "TqApi",
        broker: str = "H宏源期货",
        account_id: str = "",
        broker_password: str = "",
    ) -> None:
        self.tq_api = tq_api
        self.broker = broker
        self.account_id = account_id
        self.broker_password = broker_password

    # ------------------------------------------------------------------
    # 工厂方法：创建实盘连接
    # ------------------------------------------------------------------

    @classmethod
    def connect_live(
        cls,
        broker_id: str,
        account_id: str,
        broker_password: str,
        auth_account: str,
        auth_password: str,
    ) -> "AccountManager":
        """
        按天勤文档正确初始化实盘连接并返回 AccountManager 实例。

        TqSdk 实盘连接需要同时提供：
        - TqAccount(broker_id, account_id, password)  # 期货公司账户
        - TqAuth(account, password)                   # 天勤平台认证

        Parameters
        ----------
        broker_id : str
            期货公司代码，宏源期货为 "H宏源期货"
        account_id : str
            期货公司资金账号
        broker_password : str
            期货账户密码（对应环境变量 TQ_BROKER_PASSWORD）
        auth_account : str
            天勤平台账户（邮箱），用于 TqAuth
        auth_password : str
            天勤平台密码，用于 TqAuth

        Returns
        -------
        AccountManager
            持有已连接 TqApi 的管理器实例

        Examples
        --------
        >>> manager = AccountManager.connect_live(
        ...     broker_id="H宏源期货",
        ...     account_id="12345678",
        ...     broker_password="broker_pass",
        ...     auth_account="your@email.com",
        ...     auth_password="tq_platform_pass",
        ... )
        """
        try:
            from tqsdk import TqAccount, TqApi, TqAuth
        except ImportError as exc:
            raise ImportError(
                "tqsdk 未安装，请执行: pip install tqsdk\n"
                f"原始错误: {exc}"
            ) from exc

        api = TqApi(
            TqAccount(broker_id, account_id, broker_password),
            auth=TqAuth(auth_account, auth_password),
        )
        logger.info(
            "TqSdk 实盘已连接: broker=%s, account=%s***",
            broker_id, account_id[:4] if account_id else "",
        )
        return cls(
            tq_api=api,
            broker=broker_id,
            account_id=account_id,
            broker_password=broker_password,
        )

    # ------------------------------------------------------------------
    # 账户资金概况
    # ------------------------------------------------------------------

    def get_account_summary(self) -> dict[str, Any]:
        """
        获取账户资金概况。

        Returns
        -------
        dict
            balance, available, margin, margin_ratio, float_profit,
            position_profit, close_profit, commission, risk_ratio

        Raises
        ------
        RuntimeError
            天勤 API 调用失败时
        """
        try:
            account = self.tq_api.get_account()
        except Exception as exc:
            logger.warning("获取账户数据失败（可能为非交易时段）: %s", exc)
            raise RuntimeError(f"账户数据不可用: {exc}") from exc

        balance = _safe_float(account.balance)
        margin  = _safe_float(account.margin)
        margin_ratio = margin / balance if balance > 0 else 0.0

        return {
            "balance":          balance,
            "available":        _safe_float(account.available),
            "margin":           margin,
            "margin_ratio":     round(margin_ratio, 6),
            "float_profit":     _safe_float(account.float_profit),
            "position_profit":  _safe_float(account.position_profit),
            "close_profit":     _safe_float(account.close_profit),
            "commission":       _safe_float(account.commission),
            "risk_ratio":       _safe_float(getattr(account, "risk_ratio", 0.0)),
        }

    # ------------------------------------------------------------------
    # 持仓查询
    # ------------------------------------------------------------------

    def get_all_positions(self) -> list[dict[str, Any]]:
        """
        获取所有持仓（期货 + 期权）列表。

        每个持仓方向展开为独立记录，仅返回 volume > 0 的方向。

        Returns
        -------
        list[dict]
            字段：symbol, direction ("LONG"/"SHORT"), volume, volume_today,
                  open_price_avg, last_price, float_profit,
                  margin, instrument_type ("FUTURE"/"OPTION")
        """
        try:
            positions = self.tq_api.get_position()
        except Exception as exc:
            logger.warning("获取持仓数据失败: %s", exc)
            return []

        result: list[dict[str, Any]] = []
        for symbol, pos in positions.items():
            instrument_type = "OPTION" if _is_option(symbol) else "FUTURE"

            # 多头
            vol_long = int(_safe_float(pos.volume_long, 0))
            if vol_long > 0:
                result.append({
                    "symbol":          symbol,
                    "direction":       "LONG",
                    "volume":          vol_long,
                    "volume_today":    int(_safe_float(pos.volume_long_today, 0)),
                    "open_price_avg":  _safe_float(pos.open_price_long),
                    "last_price":      _safe_float(pos.last_price),
                    "float_profit":    _safe_float(pos.float_profit_long),
                    "margin":          _safe_float(pos.margin_long),
                    "instrument_type": instrument_type,
                })

            # 空头
            vol_short = int(_safe_float(pos.volume_short, 0))
            if vol_short > 0:
                result.append({
                    "symbol":          symbol,
                    "direction":       "SHORT",
                    "volume":          vol_short,
                    "volume_today":    int(_safe_float(pos.volume_short_today, 0)),
                    "open_price_avg":  _safe_float(pos.open_price_short),
                    "last_price":      _safe_float(pos.last_price),
                    "float_profit":    _safe_float(pos.float_profit_short),
                    "margin":          _safe_float(pos.margin_short),
                    "instrument_type": instrument_type,
                })

        return result

    def get_option_positions(self) -> list[dict[str, Any]]:
        """
        获取期权持仓列表，附加 strike_price / call_put / expire_date / underlying。

        Returns
        -------
        list[dict]
        """
        option_pos: list[dict[str, Any]] = []
        for pos in self.get_all_positions():
            if pos["instrument_type"] != "OPTION":
                continue
            try:
                parsed = self.parse_option_symbol(pos["symbol"])
            except ValueError:
                logger.warning("无法解析期权代码: %s", pos["symbol"])
                parsed = {}

            entry = dict(pos)
            entry["strike_price"] = parsed.get("strike_price", float("nan"))
            entry["call_put"]     = parsed.get("call_put", "")   # "CALL" or "PUT"
            entry["expire_date"]  = parsed.get("expire_month", "")
            entry["underlying"]   = (
                f"{parsed['exchange']}.{parsed['product']}{parsed['expire_month']}"
                if "exchange" in parsed else ""
            )
            option_pos.append(entry)

        return option_pos

    def get_futures_positions(self) -> list[dict[str, Any]]:
        """
        获取期货持仓列表（过滤掉期权）。

        Returns
        -------
        list[dict]
            字段与 get_all_positions() 一致
        """
        return [p for p in self.get_all_positions() if p["instrument_type"] == "FUTURE"]

    # ------------------------------------------------------------------
    # Greeks 汇总（待实现）
    # ------------------------------------------------------------------

    def get_position_greeks(
        self,
        spot_prices: Optional[dict[str, float]] = None,
        risk_free_rate: float = 0.02,
        contract_multiplier: int = 100,
    ) -> dict[str, float]:
        """
        计算当前期权持仓组合的汇总 Greeks。

        Notes
        -----
        依赖 models/greeks.py 中的 calc_portfolio_greeks()（尚未实现）。
        """
        raise NotImplementedError("TODO: 对接 models/greeks.calc_portfolio_greeks() 计算组合 Greeks")

    # ------------------------------------------------------------------
    # 保证金明细
    # ------------------------------------------------------------------

    def get_margin_detail(self) -> list[dict[str, Any]]:
        """
        各持仓保证金明细（从 get_all_positions() 提取）。

        Returns
        -------
        list[dict]
            字段：symbol, direction, volume, margin
        """
        return [
            {
                "symbol":    p["symbol"],
                "direction": p["direction"],
                "volume":    p["volume"],
                "margin":    p["margin"],
            }
            for p in self.get_all_positions()
        ]

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def is_account_ready(self) -> bool:
        """
        检查账户数据是否可用（balance > 0 且非 NaN）。

        Returns
        -------
        bool
        """
        try:
            account = self.tq_api.get_account()
            balance = float(account.balance)
            return not math.isnan(balance) and balance > 0
        except Exception:
            return False

    @staticmethod
    def parse_option_symbol(symbol: str) -> dict[str, Any]:
        """
        解析天勤期权合约代码，提取关键字段。

        Parameters
        ----------
        symbol : str
            天勤期权代码，如 CFFEX.IO2406-C-3800

        Returns
        -------
        dict
            exchange, product, expire_month, call_put ("CALL"/"PUT"),
            strike_price (float)

        Raises
        ------
        ValueError
            代码格式不符合期权规范时

        Examples
        --------
        >>> AccountManager.parse_option_symbol("CFFEX.IO2406-C-3800")
        {'exchange': 'CFFEX', 'product': 'IO', 'expire_month': '2406',
         'call_put': 'CALL', 'strike_price': 3800.0}
        """
        m = _OPTION_SYMBOL_RE.match(symbol)
        if not m:
            raise ValueError(
                f"无法解析期权代码 '{symbol}'，"
                "格式应为 EXCHANGE.PRODUCT+EXPIRE-C/P-STRIKE，"
                "如 CFFEX.IO2406-C-3800"
            )
        return {
            "exchange":     m.group("exchange"),
            "product":      m.group("product"),
            "expire_month": m.group("expire_month"),
            "call_put":     "CALL" if m.group("call_put") == "C" else "PUT",
            "strike_price": float(m.group("strike_price")),
        }
