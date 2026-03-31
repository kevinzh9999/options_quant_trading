"""
db_manager.py
-------------
职责：SQLite 数据库管理器。
- 初始化数据库（建表 + 建索引）
- 提供写入、查询、更新的通用接口
- 支持批量 upsert（INSERT OR REPLACE）
- 提供期货/期权/交易日历的领域专用查询方法
- 支持 :memory: 数据库（持久单连接）

所有写入操作幂等；所有查询在无结果时返回空 DataFrame 而非 None。
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

from .schemas import ALL_TABLES, DAILY_MODEL_OUTPUT_ALTER_SQLS, SIGNAL_LOG_ALTER_SQLS

logger = logging.getLogger(__name__)

# ts_code 含 % 时使用 LIKE，否则使用 =
_LIKE_OPERATORS = {True: "LIKE", False: "="}


class DBManager:
    """
    SQLite 数据库管理器。

    Parameters
    ----------
    db_path : str
        数据库文件路径，不存在时自动创建；传入 ":memory:" 使用纯内存数据库。
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        # :memory: 必须保持单连接；文件数据库也用持久连接简化管理
        if db_path != ":memory:":
            p = Path(db_path)
            p.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES, timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self.initialize_tables()

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------

    def initialize_tables(self) -> None:
        """执行所有 CREATE TABLE IF NOT EXISTS 语句（幂等）。"""
        for ddl in ALL_TABLES:
            self._conn.executescript(ddl)
        # 增量 ALTER TABLE（新增列，列已存在时静默忽略）
        for alter_sql in DAILY_MODEL_OUTPUT_ALTER_SQLS + SIGNAL_LOG_ALTER_SQLS:
            try:
                self._conn.execute(alter_sql)
            except Exception:
                pass  # column already exists
        self._conn.commit()
        logger.debug("数据库表初始化完成: %s", self.db_path)

    # ------------------------------------------------------------------
    # 通用写入
    # ------------------------------------------------------------------

    def upsert_dataframe(self, table_name: str, df: pd.DataFrame) -> int:
        """
        将 DataFrame 批量写入指定表（INSERT OR REPLACE 语义）。

        Parameters
        ----------
        table_name : str
            目标表名
        df : pd.DataFrame
            数据，列名必须与表字段名一致

        Returns
        -------
        int
            写入的行数
        """
        if df.empty:
            return 0
        cols = list(df.columns)
        placeholders = ", ".join("?" * len(cols))
        col_list = ", ".join(cols)
        sql = f"INSERT OR REPLACE INTO {table_name} ({col_list}) VALUES ({placeholders})"
        # itertuples 比 to_dict 快，None 自动变 NULL
        data = [tuple(row) for row in df.itertuples(index=False, name=None)]
        self._conn.executemany(sql, data)
        self._conn.commit()
        logger.debug("upsert %d rows into %s", len(df), table_name)
        return len(df)

    # ------------------------------------------------------------------
    # 通用查询
    # ------------------------------------------------------------------

    def query(self, sql: str, params: Optional[tuple] = None) -> pd.DataFrame:
        """
        执行 SQL 查询，返回 DataFrame。

        Parameters
        ----------
        sql : str
            SELECT 语句，占位符用 ?
        params : tuple, optional
            绑定参数

        Returns
        -------
        pd.DataFrame
            查询结果；无数据时返回带列名的空 DataFrame。
        """
        cursor = self._conn.execute(sql, params or ())
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description] if cursor.description else []
        if not rows:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame([dict(r) for r in rows], columns=cols)

    # ------------------------------------------------------------------
    # 期货查询
    # ------------------------------------------------------------------

    def get_futures_daily(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        查询期货日线数据。

        Parameters
        ----------
        ts_code : str
            合约代码，如 "IF2406.CFX"；支持 LIKE 模式如 "%IF%"
        start_date : str
            起始日期 YYYYMMDD
        end_date : str
            结束日期 YYYYMMDD

        Returns
        -------
        pd.DataFrame
            按 trade_date 升序排列；无数据时返回空 DataFrame
        """
        op = _LIKE_OPERATORS["%" in ts_code]
        return self.query(
            f"SELECT * FROM futures_daily"
            f" WHERE ts_code {op} ? AND trade_date >= ? AND trade_date <= ?"
            f" ORDER BY trade_date",
            (ts_code, start_date, end_date),
        )

    def get_futures_min(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        freq: str = "5min",
    ) -> pd.DataFrame:
        """
        查询期货分钟线数据。

        Parameters
        ----------
        ts_code : str
            合约代码
        start_date : str
            起始日期 YYYYMMDD（内部转换为 YYYY-MM-DD 前缀做范围过滤）
        end_date : str
            结束日期 YYYYMMDD
        freq : str
            频率（"1min"/"5min" 等）。数据库存储原始频率；若需聚合请在调用方处理。

        Returns
        -------
        pd.DataFrame
            按 datetime 升序排列
        """
        start_dt = _to_datetime_prefix(start_date)
        end_dt = _to_datetime_prefix(end_date, end_of_day=True)
        logger.debug("get_futures_min freq=%s (aggregation delegated to caller)", freq)
        op = _LIKE_OPERATORS["%" in ts_code]
        return self.query(
            f"SELECT * FROM futures_min"
            f" WHERE ts_code {op} ? AND datetime >= ? AND datetime <= ?"
            f" ORDER BY datetime",
            (ts_code, start_dt, end_dt),
        )

    # ------------------------------------------------------------------
    # 期权查询
    # ------------------------------------------------------------------

    def get_options_daily(
        self,
        underlying: str,
        trade_date: str,
        call_put: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        查询期权日线数据（单交易日快照）。

        Parameters
        ----------
        underlying : str
            标的代码，如 "IO" 或 "510050"
        trade_date : str
            交易日 YYYYMMDD
        call_put : str, optional
            "C" 认购 / "P" 认沽；为 None 时返回全部

        Returns
        -------
        pd.DataFrame
            按 exercise_price, call_put 排序
        """
        sql = (
            "SELECT * FROM options_daily"
            " WHERE underlying_code LIKE ? AND trade_date = ?"
        )
        params: list[Any] = [f"%{underlying}%", trade_date]
        if call_put:
            sql += " AND call_put = ?"
            params.append(call_put)
        sql += " ORDER BY exercise_price, call_put"
        return self.query(sql, tuple(params))

    def get_options_chain(
        self,
        underlying: str,
        trade_date: str,
        expire_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        获取期权链：合约基本信息 + 当日行情合并。

        Parameters
        ----------
        underlying : str
            标的代码
        trade_date : str
            交易日 YYYYMMDD
        expire_date : str, optional
            指定到期日 YYYYMMDD；为 None 时返回全部到期月份

        Returns
        -------
        pd.DataFrame
            包含 ts_code, exercise_price, call_put, expire_date,
            contract_unit, exercise_type, close, settle, volume, oi
        """
        sql = """
            SELECT
                c.ts_code,
                c.exercise_price,
                c.call_put,
                c.expire_date,
                c.contract_unit,
                c.exercise_type,
                d.close,
                d.settle,
                d.volume,
                d.oi
            FROM options_contracts c
            LEFT JOIN options_daily d
                ON c.ts_code = d.ts_code AND d.trade_date = ?
            WHERE c.underlying_code LIKE ?
        """
        params: list[Any] = [trade_date, f"%{underlying}%"]
        if expire_date:
            sql += " AND c.expire_date = ?"
            params.append(expire_date)
        sql += " ORDER BY c.expire_date, c.exercise_price, c.call_put"
        return self.query(sql, tuple(params))

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
        查询交易日历（只返回 is_open=1 的交易日）。

        Parameters
        ----------
        exchange : str
            交易所代码
        start_date : str, optional
            起始日期 YYYYMMDD
        end_date : str, optional
            结束日期 YYYYMMDD

        Returns
        -------
        pd.DataFrame
            按 trade_date 升序排列
        """
        conditions = ["exchange = ?", "is_open = 1"]
        params: list[Any] = [exchange]
        if start_date:
            conditions.append("trade_date >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("trade_date <= ?")
            params.append(end_date)
        where = " AND ".join(conditions)
        return self.query(
            f"SELECT * FROM trade_calendar WHERE {where} ORDER BY trade_date",
            tuple(params),
        )

    # ------------------------------------------------------------------
    # 元数据工具
    # ------------------------------------------------------------------

    def get_latest_date(
        self,
        table_name: str,
        ts_code: Optional[str] = None,
    ) -> Optional[str]:
        """
        查询指定表中某合约（或全表）的最新日期。
        用于增量更新时确定从哪天开始拉取新数据。

        Parameters
        ----------
        table_name : str
            表名（须含 trade_date 列）
        ts_code : str, optional
            合约代码；为 None 时查全表最大日期

        Returns
        -------
        str | None
            最大日期 YYYYMMDD，表为空时返回 None
        """
        if ts_code:
            op = _LIKE_OPERATORS["%" in ts_code]
            cursor = self._conn.execute(
                f"SELECT MAX(trade_date) FROM {table_name} WHERE ts_code {op} ?",
                (ts_code,),
            )
        else:
            cursor = self._conn.execute(
                f"SELECT MAX(trade_date) FROM {table_name}"
            )
        row = cursor.fetchone()
        return row[0] if row and row[0] is not None else None

    def table_exists(self, table: str) -> bool:
        """检查表是否存在"""
        cursor = self._conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        )
        return bool(cursor.fetchone()[0])

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def close(self) -> None:
        """关闭数据库连接，释放资源。"""
        self._conn.close()
        logger.debug("数据库连接已关闭: %s", self.db_path)

    def __enter__(self) -> "DBManager":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # ------------------------------------------------------------------
    # 向后兼容别名（供 unified_api.py 等现有代码使用）
    # ------------------------------------------------------------------

    def upsert_df(self, table: str, df: pd.DataFrame) -> int:
        """向后兼容：等同于 upsert_dataframe。"""
        return self.upsert_dataframe(table, df)

    def upsert_rows(self, table: str, rows: list[dict[str, Any]]) -> int:
        """向后兼容：将字典列表转为 DataFrame 后 upsert。"""
        if not rows:
            return 0
        return self.upsert_dataframe(table, pd.DataFrame(rows))

    def query_df(self, sql: str, params: Sequence[Any] | None = None) -> pd.DataFrame:
        """向后兼容：等同于 query。"""
        return self.query(sql, tuple(params) if params else ())

    def query_scalar(self, sql: str, params: Sequence[Any] | None = None) -> Any:
        """向后兼容：执行查询返回第一行第一列。"""
        cursor = self._conn.execute(sql, tuple(params) if params else ())
        row = cursor.fetchone()
        return row[0] if row is not None else None

    def get_max_date(self, table: str, date_col: str = "trade_date") -> str | None:
        """向后兼容：等同于 get_latest_date（无 ts_code 过滤）。"""
        cursor = self._conn.execute(f"SELECT MAX({date_col}) FROM {table}")
        row = cursor.fetchone()
        return row[0] if row and row[0] is not None else None


# ======================================================================
# 内部工具
# ======================================================================

def _to_datetime_prefix(date_yyyymmdd: str, end_of_day: bool = False) -> str:
    """将 YYYYMMDD 转为分钟线 datetime 列可用的比较字符串。"""
    y, m, d = date_yyyymmdd[:4], date_yyyymmdd[4:6], date_yyyymmdd[6:]
    if end_of_day:
        return f"{y}-{m}-{d} 23:59:59"
    return f"{y}-{m}-{d} 00:00:00"
