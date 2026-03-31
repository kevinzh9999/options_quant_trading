"""
test_schemas.py
---------------
测试 data/storage/schemas.py

覆盖：
- ALL_TABLES 包含所有预期表
- ALL_DDL 向后兼容别名
- 每个 SQL 字符串包含正确的表名和 PRIMARY KEY
- 关键字段存在于对应 SQL 中
- 索引定义存在
- DBManager 能用这些 DDL 建表（集成验证）
- 新字段名（underlying_code / exercise_price / pre_close 等）正确
"""

from __future__ import annotations

import sqlite3
import textwrap
from pathlib import Path

import pytest

from data.storage.schemas import (
    ALL_TABLES,
    ALL_DDL,
    FUTURES_DAILY_SQL,
    FUTURES_MIN_SQL,
    OPTIONS_DAILY_SQL,
    OPTIONS_CONTRACTS_SQL,
    COMMODITY_DAILY_SQL,
    TRADE_CALENDAR_SQL,
    STRATEGY_SIGNALS_SQL,
    STRATEGY_TRADES_SQL,
    STRATEGY_PNL_SQL,
)


# ======================================================================
# ALL_TABLES / ALL_DDL
# ======================================================================

class TestAllTables:

    def test_contains_nine_tables(self):
        assert len(ALL_TABLES) == 9

    def test_all_ddl_is_alias(self):
        assert ALL_DDL is ALL_TABLES

    def test_all_entries_are_strings(self):
        for ddl in ALL_TABLES:
            assert isinstance(ddl, str), f"非字符串条目: {ddl!r}"

    def test_expected_sqls_all_present(self):
        expected = {
            FUTURES_DAILY_SQL, FUTURES_MIN_SQL, OPTIONS_DAILY_SQL,
            OPTIONS_CONTRACTS_SQL, COMMODITY_DAILY_SQL, TRADE_CALENDAR_SQL,
            STRATEGY_SIGNALS_SQL, STRATEGY_TRADES_SQL, STRATEGY_PNL_SQL,
        }
        assert set(ALL_TABLES) == expected


# ======================================================================
# 各 SQL 字符串内容验证
# ======================================================================

class TestFuturesDailySql:

    def test_table_name(self):
        assert "futures_daily" in FUTURES_DAILY_SQL

    def test_primary_key(self):
        assert "PRIMARY KEY" in FUTURES_DAILY_SQL
        assert "ts_code" in FUTURES_DAILY_SQL
        assert "trade_date" in FUTURES_DAILY_SQL

    def test_required_columns(self):
        for col in ["open", "high", "low", "close", "volume", "oi",
                    "settle", "pre_close", "pre_settle"]:
            assert col in FUTURES_DAILY_SQL, f"缺少字段: {col}"

    def test_has_trade_date_index(self):
        assert "idx_fd_trade_date" in FUTURES_DAILY_SQL


class TestFuturesMinSql:

    def test_table_name(self):
        assert "futures_min" in FUTURES_MIN_SQL

    def test_primary_key_composite(self):
        assert "PRIMARY KEY" in FUTURES_MIN_SQL
        assert "ts_code" in FUTURES_MIN_SQL
        assert "datetime" in FUTURES_MIN_SQL

    def test_required_columns(self):
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in FUTURES_MIN_SQL


class TestOptionsDailySql:

    def test_table_name(self):
        assert "options_daily" in OPTIONS_DAILY_SQL

    def test_new_field_names(self):
        assert "underlying_code" in OPTIONS_DAILY_SQL
        assert "exercise_price" in OPTIONS_DAILY_SQL
        # 旧字段名不应出现
        assert "strike_price" not in OPTIONS_DAILY_SQL

    def test_pre_fields(self):
        assert "pre_close" in OPTIONS_DAILY_SQL
        assert "pre_settle" in OPTIONS_DAILY_SQL

    def test_indexes(self):
        assert "idx_od_trade_date" in OPTIONS_DAILY_SQL
        assert "idx_od_underlying_date" in OPTIONS_DAILY_SQL

    def test_primary_key(self):
        assert "PRIMARY KEY" in OPTIONS_DAILY_SQL


class TestOptionsContractsSql:

    def test_table_name(self):
        assert "options_contracts" in OPTIONS_CONTRACTS_SQL

    def test_new_field_names(self):
        assert "underlying_code" in OPTIONS_CONTRACTS_SQL
        assert "exercise_price" in OPTIONS_CONTRACTS_SQL
        assert "strike_price" not in OPTIONS_CONTRACTS_SQL

    def test_extra_fields(self):
        assert "contract_unit" in OPTIONS_CONTRACTS_SQL
        assert "exercise_type" in OPTIONS_CONTRACTS_SQL

    def test_primary_key_single(self):
        assert "PRIMARY KEY" in OPTIONS_CONTRACTS_SQL
        assert "ts_code" in OPTIONS_CONTRACTS_SQL

    def test_indexes(self):
        assert "idx_oc_underlying" in OPTIONS_CONTRACTS_SQL
        assert "idx_oc_expire" in OPTIONS_CONTRACTS_SQL


class TestCommodityDailySql:

    def test_table_name(self):
        assert "commodity_daily" in COMMODITY_DAILY_SQL

    def test_required_columns(self):
        for col in ["ts_code", "trade_date", "exchange", "underlying",
                    "open", "high", "low", "close", "volume", "oi",
                    "settle", "amount"]:
            assert col in COMMODITY_DAILY_SQL, f"缺少字段: {col}"

    def test_indexes(self):
        assert "idx_cd_trade_date" in COMMODITY_DAILY_SQL
        assert "idx_cd_underlying" in COMMODITY_DAILY_SQL


class TestTradeCalendarSql:

    def test_table_name(self):
        assert "trade_calendar" in TRADE_CALENDAR_SQL

    def test_required_columns(self):
        for col in ["exchange", "trade_date", "is_open", "pretrade_date"]:
            assert col in TRADE_CALENDAR_SQL

    def test_primary_key_composite(self):
        assert "PRIMARY KEY" in TRADE_CALENDAR_SQL
        assert "exchange" in TRADE_CALENDAR_SQL
        assert "trade_date" in TRADE_CALENDAR_SQL

    def test_has_index(self):
        assert "idx_tc_trade_date" in TRADE_CALENDAR_SQL


class TestStrategySignalsSql:

    def test_table_name(self):
        assert "strategy_signals" in STRATEGY_SIGNALS_SQL

    def test_autoincrement_id(self):
        assert "AUTOINCREMENT" in STRATEGY_SIGNALS_SQL

    def test_required_columns(self):
        for col in ["strategy_name", "trade_date", "symbol", "direction",
                    "signal_type", "strength", "target_volume",
                    "metadata_json", "created_at"]:
            assert col in STRATEGY_SIGNALS_SQL, f"缺少字段: {col}"

    def test_indexes(self):
        assert "idx_ss_strategy_date" in STRATEGY_SIGNALS_SQL
        assert "idx_ss_symbol_date" in STRATEGY_SIGNALS_SQL


class TestStrategyTradesSql:

    def test_table_name(self):
        assert "strategy_trades" in STRATEGY_TRADES_SQL

    def test_autoincrement_id(self):
        assert "AUTOINCREMENT" in STRATEGY_TRADES_SQL

    def test_required_columns(self):
        for col in ["strategy_name", "trade_date", "symbol", "direction",
                    "volume", "price", "commission", "slippage", "created_at"]:
            assert col in STRATEGY_TRADES_SQL, f"缺少字段: {col}"

    def test_indexes(self):
        assert "idx_st_strategy_date" in STRATEGY_TRADES_SQL


class TestStrategyPnlSql:

    def test_table_name(self):
        assert "strategy_pnl" in STRATEGY_PNL_SQL

    def test_required_columns(self):
        for col in ["strategy_name", "trade_date", "realized_pnl",
                    "unrealized_pnl", "commission", "net_pnl"]:
            assert col in STRATEGY_PNL_SQL, f"缺少字段: {col}"

    def test_primary_key_composite(self):
        assert "PRIMARY KEY" in STRATEGY_PNL_SQL
        assert "strategy_name" in STRATEGY_PNL_SQL
        assert "trade_date" in STRATEGY_PNL_SQL

    def test_indexes(self):
        assert "idx_sp_strategy" in STRATEGY_PNL_SQL
        assert "idx_sp_date" in STRATEGY_PNL_SQL


# ======================================================================
# 集成验证：DBManager 能用 ALL_TABLES 成功建表
# ======================================================================

class TestSchemasIntegration:

    @pytest.fixture
    def conn(self, tmp_path: Path):
        db_path = tmp_path / "test.db"
        c = sqlite3.connect(str(db_path))
        yield c
        c.close()

    def test_all_tables_created(self, conn):
        for ddl in ALL_TABLES:
            conn.executescript(ddl)
        conn.commit()

        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row[0] for row in cursor.fetchall()}
        expected = {
            "futures_daily", "futures_min", "options_daily",
            "options_contracts", "commodity_daily", "trade_calendar",
            "strategy_signals", "strategy_trades", "strategy_pnl",
        }
        assert expected.issubset(tables)

    def test_all_indexes_created(self, conn):
        for ddl in ALL_TABLES:
            conn.executescript(ddl)
        conn.commit()

        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' ORDER BY name"
        )
        indexes = {row[0] for row in cursor.fetchall()}
        expected_indexes = {
            "idx_fd_trade_date", "idx_fd_ts_code",
            "idx_fm_ts_code", "idx_fm_datetime",
            "idx_od_trade_date", "idx_od_underlying_date", "idx_od_expire_date",
            "idx_oc_underlying", "idx_oc_expire",
            "idx_cd_trade_date", "idx_cd_underlying",
            "idx_tc_trade_date",
            "idx_ss_strategy_date", "idx_ss_symbol_date",
            "idx_st_strategy_date", "idx_st_symbol_date",
            "idx_sp_strategy", "idx_sp_date",
        }
        assert expected_indexes.issubset(indexes), (
            f"缺少索引: {expected_indexes - indexes}"
        )

    def test_options_daily_new_columns(self, conn):
        for ddl in ALL_TABLES:
            conn.executescript(ddl)
        cursor = conn.execute("PRAGMA table_info(options_daily)")
        cols = {row[1] for row in cursor.fetchall()}
        assert "underlying_code" in cols
        assert "exercise_price" in cols
        assert "pre_close" in cols
        assert "pre_settle" in cols

    def test_options_contracts_new_columns(self, conn):
        for ddl in ALL_TABLES:
            conn.executescript(ddl)
        cursor = conn.execute("PRAGMA table_info(options_contracts)")
        cols = {row[1] for row in cursor.fetchall()}
        assert "underlying_code" in cols
        assert "exercise_price" in cols
        assert "contract_unit" in cols
        assert "exercise_type" in cols

    def test_futures_daily_pre_columns(self, conn):
        for ddl in ALL_TABLES:
            conn.executescript(ddl)
        cursor = conn.execute("PRAGMA table_info(futures_daily)")
        cols = {row[1] for row in cursor.fetchall()}
        assert "pre_close" in cols
        assert "pre_settle" in cols

    def test_strategy_signals_autoincrement(self, conn):
        for ddl in ALL_TABLES:
            conn.executescript(ddl)
        conn.execute("""
            INSERT INTO strategy_signals
                (strategy_name, trade_date, symbol, direction, signal_type,
                 strength, target_volume, metadata_json, created_at)
            VALUES ('vol_arb', '20240102', 'IF2406.CFX', 'long', 'entry',
                    'strong', 2, '{}', '2024-01-02 09:30:00')
        """)
        conn.commit()
        cursor = conn.execute("SELECT id FROM strategy_signals")
        row = cursor.fetchone()
        assert row[0] == 1

    def test_strategy_pnl_net_pnl_column(self, conn):
        for ddl in ALL_TABLES:
            conn.executescript(ddl)
        conn.execute("""
            INSERT INTO strategy_pnl
                (strategy_name, trade_date, realized_pnl, unrealized_pnl,
                 commission, net_pnl)
            VALUES ('vol_arb', '20240102', 1000.0, 500.0, 50.0, 1450.0)
        """)
        conn.commit()
        cursor = conn.execute("SELECT net_pnl FROM strategy_pnl")
        assert cursor.fetchone()[0] == 1450.0

    def test_idempotent_double_run(self, conn):
        """IF NOT EXISTS 保证重复执行不报错"""
        for ddl in ALL_TABLES:
            conn.executescript(ddl)
        for ddl in ALL_TABLES:
            conn.executescript(ddl)  # 第二次不应抛异常
