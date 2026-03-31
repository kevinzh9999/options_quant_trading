"""
test_db_manager.py
------------------
测试 data/storage/db_manager.py

全部使用 :memory: 数据库，无需 tmp_path。

覆盖：
- initialize_tables 幂等性
- table_exists
- upsert_dataframe：基本写入 / 幂等更新 / 空 DataFrame
- upsert_rows / upsert_df 向后兼容别名
- query：有数据 / 无数据（带列名的空 DF）
- query_df / query_scalar 向后兼容别名
- get_futures_daily：精确匹配 / LIKE 模糊 / 日期范围
- get_futures_min：日期范围
- get_options_daily：标的过滤 / call_put 过滤
- get_options_chain：JOIN 逻辑 / expire_date 过滤
- get_trade_calendar：日期范围 / is_open 过滤
- get_latest_date：全表 / 按 ts_code / 空表
- get_max_date 向后兼容
- close / context manager
"""

from __future__ import annotations

import pandas as pd
import pytest

from data.storage.db_manager import DBManager


# ======================================================================
# Fixture：每个测试函数独享一个内存 DB
# ======================================================================

@pytest.fixture
def db() -> DBManager:
    manager = DBManager(":memory:")
    yield manager
    manager.close()


# ======================================================================
# 初始化
# ======================================================================

class TestInit:

    def test_all_tables_exist(self, db: DBManager):
        for table in [
            "futures_daily", "futures_min", "options_daily",
            "options_contracts", "commodity_daily", "trade_calendar",
            "strategy_signals", "strategy_trades", "strategy_pnl",
        ]:
            assert db.table_exists(table), f"表 {table} 未创建"

    def test_initialize_tables_idempotent(self, db: DBManager):
        """重复调用不应抛异常"""
        db.initialize_tables()
        db.initialize_tables()

    def test_context_manager(self):
        with DBManager(":memory:") as mgr:
            assert mgr.table_exists("futures_daily")


# ======================================================================
# upsert_dataframe
# ======================================================================

FUTURES_ROW = {
    "ts_code": "IF2406.CFX", "trade_date": "20240102",
    "open": 4010.0, "high": 4060.0, "low": 4000.0,
    "close": 4040.0, "volume": 5000.0, "oi": 20000.0, "settle": 4035.0,
    "pre_close": 4000.0, "pre_settle": 3990.0,
}


class TestUpsertDataframe:

    def test_basic_write(self, db: DBManager):
        n = db.upsert_dataframe("futures_daily", pd.DataFrame([FUTURES_ROW]))
        assert n == 1

    def test_empty_df_returns_zero(self, db: DBManager):
        assert db.upsert_dataframe("futures_daily", pd.DataFrame()) == 0

    def test_idempotent(self, db: DBManager):
        df = pd.DataFrame([FUTURES_ROW])
        db.upsert_dataframe("futures_daily", df)
        db.upsert_dataframe("futures_daily", df)
        count = db.query_scalar("SELECT COUNT(*) FROM futures_daily")
        assert count == 1

    def test_upsert_updates_value(self, db: DBManager):
        db.upsert_dataframe("futures_daily", pd.DataFrame([FUTURES_ROW]))
        updated = {**FUTURES_ROW, "close": 9999.0}
        db.upsert_dataframe("futures_daily", pd.DataFrame([updated]))
        close = db.query_scalar(
            "SELECT close FROM futures_daily WHERE ts_code=? AND trade_date=?",
            ("IF2406.CFX", "20240102"),
        )
        assert close == 9999.0

    def test_multiple_rows(self, db: DBManager):
        rows = [
            {**FUTURES_ROW, "trade_date": f"2024010{i}"}
            for i in range(1, 6)
        ]
        n = db.upsert_dataframe("futures_daily", pd.DataFrame(rows))
        assert n == 5

    def test_returns_row_count(self, db: DBManager):
        df = pd.DataFrame([FUTURES_ROW, {**FUTURES_ROW, "trade_date": "20240103"}])
        assert db.upsert_dataframe("futures_daily", df) == 2


# ======================================================================
# 向后兼容别名：upsert_df / upsert_rows
# ======================================================================

class TestBackwardCompatWrite:

    def test_upsert_df(self, db: DBManager):
        n = db.upsert_df("futures_daily", pd.DataFrame([FUTURES_ROW]))
        assert n == 1

    def test_upsert_rows(self, db: DBManager):
        n = db.upsert_rows("futures_daily", [FUTURES_ROW])
        assert n == 1

    def test_upsert_rows_empty(self, db: DBManager):
        assert db.upsert_rows("futures_daily", []) == 0


# ======================================================================
# query / query_df / query_scalar
# ======================================================================

class TestQuery:

    @pytest.fixture(autouse=True)
    def seed(self, db: DBManager):
        rows = [
            {**FUTURES_ROW, "trade_date": f"2024010{i}", "close": 4000.0 + i}
            for i in range(1, 4)
        ]
        db.upsert_dataframe("futures_daily", pd.DataFrame(rows))

    def test_query_returns_df(self, db: DBManager):
        df = db.query("SELECT * FROM futures_daily")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_query_with_params(self, db: DBManager):
        df = db.query(
            "SELECT * FROM futures_daily WHERE trade_date=?",
            ("20240101",),
        )
        assert len(df) == 1

    def test_query_empty_returns_empty_df_with_columns(self, db: DBManager):
        df = db.query(
            "SELECT * FROM futures_daily WHERE trade_date=?",
            ("99991231",),
        )
        assert isinstance(df, pd.DataFrame)
        assert df.empty
        assert "ts_code" in df.columns

    def test_query_df_alias(self, db: DBManager):
        df = db.query_df("SELECT * FROM futures_daily")
        assert len(df) == 3

    def test_query_scalar_count(self, db: DBManager):
        assert db.query_scalar("SELECT COUNT(*) FROM futures_daily") == 3

    def test_query_scalar_none_on_empty(self, db: DBManager):
        val = db.query_scalar(
            "SELECT close FROM futures_daily WHERE trade_date=?",
            ("99991231",),
        )
        assert val is None


# ======================================================================
# get_futures_daily
# ======================================================================

class TestGetFuturesDaily:

    @pytest.fixture(autouse=True)
    def seed(self, db: DBManager):
        rows = [
            {**FUTURES_ROW, "ts_code": "IF2406.CFX", "trade_date": "20240102"},
            {**FUTURES_ROW, "ts_code": "IF2406.CFX", "trade_date": "20240103"},
            {**FUTURES_ROW, "ts_code": "IH2406.CFX", "trade_date": "20240102"},
        ]
        db.upsert_dataframe("futures_daily", pd.DataFrame(rows))

    def test_exact_match(self, db: DBManager):
        df = db.get_futures_daily("IF2406.CFX", "20240101", "20240131")
        assert len(df) == 2
        assert (df["ts_code"] == "IF2406.CFX").all()

    def test_like_match(self, db: DBManager):
        df = db.get_futures_daily("%IF%", "20240101", "20240131")
        assert len(df) == 2  # IF2406 matches, IH2406 does not

    def test_like_wildcard_all(self, db: DBManager):
        df = db.get_futures_daily("%.CFX", "20240101", "20240131")
        assert len(df) == 3

    def test_date_range_filter(self, db: DBManager):
        df = db.get_futures_daily("IF2406.CFX", "20240103", "20240103")
        assert len(df) == 1
        assert df.iloc[0]["trade_date"] == "20240103"

    def test_sorted_ascending(self, db: DBManager):
        df = db.get_futures_daily("IF2406.CFX", "20240101", "20240131")
        assert df["trade_date"].tolist() == sorted(df["trade_date"].tolist())

    def test_no_match_empty_df(self, db: DBManager):
        df = db.get_futures_daily("XX9999.CFX", "20240101", "20240131")
        assert df.empty


# ======================================================================
# get_futures_min
# ======================================================================

class TestGetFuturesMin:

    @pytest.fixture(autouse=True)
    def seed(self, db: DBManager):
        rows = [
            {"ts_code": "IF2406.CFX", "datetime": "2024-01-02 09:35:00",
             "open": 4010.0, "high": 4015.0, "low": 4005.0, "close": 4012.0, "volume": 100.0},
            {"ts_code": "IF2406.CFX", "datetime": "2024-01-02 09:40:00",
             "open": 4012.0, "high": 4020.0, "low": 4010.0, "close": 4018.0, "volume": 80.0},
            {"ts_code": "IF2406.CFX", "datetime": "2024-01-03 09:35:00",
             "open": 4018.0, "high": 4025.0, "low": 4015.0, "close": 4022.0, "volume": 90.0},
        ]
        db.upsert_dataframe("futures_min", pd.DataFrame(rows))

    def test_returns_correct_day(self, db: DBManager):
        df = db.get_futures_min("IF2406.CFX", "20240102", "20240102")
        assert len(df) == 2
        assert df["datetime"].str.startswith("2024-01-02").all()

    def test_multi_day(self, db: DBManager):
        df = db.get_futures_min("IF2406.CFX", "20240102", "20240103")
        assert len(df) == 3

    def test_sorted_by_datetime(self, db: DBManager):
        df = db.get_futures_min("IF2406.CFX", "20240102", "20240103")
        assert df["datetime"].tolist() == sorted(df["datetime"].tolist())

    def test_empty_result(self, db: DBManager):
        df = db.get_futures_min("IF2406.CFX", "20230101", "20230101")
        assert df.empty

    def test_freq_parameter_accepted(self, db: DBManager):
        """freq 参数不影响返回结果（聚合由调用方负责）"""
        df1 = db.get_futures_min("IF2406.CFX", "20240102", "20240102", freq="1min")
        df2 = db.get_futures_min("IF2406.CFX", "20240102", "20240102", freq="5min")
        assert len(df1) == len(df2)


# ======================================================================
# get_options_daily
# ======================================================================

OPT_ROW_C = {
    "ts_code": "IO2406-C-3800.CFX", "trade_date": "20240102",
    "exchange": "CFFEX", "underlying_code": "IO",
    "exercise_price": 3800.0, "call_put": "C", "expire_date": "20240621",
    "close": 250.0, "settle": 245.0, "volume": 500.0, "oi": 1200.0,
    "pre_close": 240.0, "pre_settle": 238.0,
}
OPT_ROW_P = {**OPT_ROW_C,
             "ts_code": "IO2406-P-3800.CFX", "call_put": "P",
             "close": 120.0, "settle": 118.0}


class TestGetOptionsDaily:

    @pytest.fixture(autouse=True)
    def seed(self, db: DBManager):
        db.upsert_dataframe("options_daily",
                            pd.DataFrame([OPT_ROW_C, OPT_ROW_P]))

    def test_returns_both_directions(self, db: DBManager):
        df = db.get_options_daily("IO", "20240102")
        assert len(df) == 2

    def test_filter_call(self, db: DBManager):
        df = db.get_options_daily("IO", "20240102", call_put="C")
        assert len(df) == 1
        assert df.iloc[0]["call_put"] == "C"

    def test_filter_put(self, db: DBManager):
        df = db.get_options_daily("IO", "20240102", call_put="P")
        assert len(df) == 1
        assert df.iloc[0]["call_put"] == "P"

    def test_wrong_date_empty(self, db: DBManager):
        df = db.get_options_daily("IO", "20230101")
        assert df.empty

    def test_underlying_partial_match(self, db: DBManager):
        """underlying_code 列存储为 "IO"，用 LIKE %IO% 应能匹配"""
        df = db.get_options_daily("IO", "20240102")
        assert len(df) == 2


# ======================================================================
# get_options_chain
# ======================================================================

CONTRACT_ROW_C = {
    "ts_code": "IO2406-C-3800.CFX", "exchange": "CFFEX",
    "underlying_code": "IO", "exercise_price": 3800.0, "call_put": "C",
    "expire_date": "20240621", "list_date": "20230921",
    "delist_date": "20240621", "contract_unit": 100.0, "exercise_type": "E",
}
CONTRACT_ROW_P = {**CONTRACT_ROW_C,
                  "ts_code": "IO2406-P-3800.CFX", "call_put": "P"}
CONTRACT_ROW_FAR = {**CONTRACT_ROW_C,
                    "ts_code": "IO2409-C-3800.CFX",
                    "expire_date": "20240920",
                    "delist_date": "20240920"}


class TestGetOptionsChain:

    @pytest.fixture(autouse=True)
    def seed(self, db: DBManager):
        db.upsert_dataframe(
            "options_contracts",
            pd.DataFrame([CONTRACT_ROW_C, CONTRACT_ROW_P, CONTRACT_ROW_FAR]),
        )
        db.upsert_dataframe("options_daily", pd.DataFrame([OPT_ROW_C, OPT_ROW_P]))

    def test_returns_all_expirations(self, db: DBManager):
        df = db.get_options_chain("IO", "20240102")
        assert len(df) == 3  # 2 near + 1 far (LEFT JOIN, far has no daily data)

    def test_filter_by_expire_date(self, db: DBManager):
        df = db.get_options_chain("IO", "20240102", expire_date="20240621")
        assert len(df) == 2
        assert (df["expire_date"] == "20240621").all()

    def test_daily_fields_present_for_matched(self, db: DBManager):
        df = db.get_options_chain("IO", "20240102", expire_date="20240621")
        assert "close" in df.columns
        assert df.loc[df["call_put"] == "C", "close"].iloc[0] == 250.0

    def test_left_join_null_for_no_daily(self, db: DBManager):
        """远月合约在 20240102 无日线数据，close 应为 NaN"""
        df = db.get_options_chain("IO", "20240102")
        far = df[df["ts_code"] == "IO2409-C-3800.CFX"]
        assert len(far) == 1
        assert pd.isna(far.iloc[0]["close"])

    def test_contract_fields_present(self, db: DBManager):
        df = db.get_options_chain("IO", "20240102", expire_date="20240621")
        for col in ["exercise_price", "contract_unit", "exercise_type"]:
            assert col in df.columns


# ======================================================================
# get_trade_calendar
# ======================================================================

class TestGetTradeCalendar:

    @pytest.fixture(autouse=True)
    def seed(self, db: DBManager):
        rows = [
            {"exchange": "CFFEX", "trade_date": "20240102",
             "is_open": 1, "pretrade_date": "20231229"},
            {"exchange": "CFFEX", "trade_date": "20240103",
             "is_open": 1, "pretrade_date": "20240102"},
            {"exchange": "CFFEX", "trade_date": "20240104",
             "is_open": 0, "pretrade_date": "20240103"},  # 非交易日
            {"exchange": "SSE",   "trade_date": "20240102",
             "is_open": 1, "pretrade_date": "20231229"},
        ]
        db.upsert_dataframe("trade_calendar", pd.DataFrame(rows))

    def test_returns_only_open_days(self, db: DBManager):
        df = db.get_trade_calendar("CFFEX")
        assert (df["is_open"] == 1).all()

    def test_exchange_filter(self, db: DBManager):
        df = db.get_trade_calendar("SSE")
        assert len(df) == 1
        assert df.iloc[0]["exchange"] == "SSE"

    def test_date_range_filter(self, db: DBManager):
        df = db.get_trade_calendar("CFFEX", start_date="20240103")
        assert all(d >= "20240103" for d in df["trade_date"])

    def test_sorted_ascending(self, db: DBManager):
        df = db.get_trade_calendar("CFFEX")
        assert df["trade_date"].tolist() == sorted(df["trade_date"].tolist())

    def test_empty_result(self, db: DBManager):
        df = db.get_trade_calendar("DCE")
        assert df.empty


# ======================================================================
# get_latest_date
# ======================================================================

class TestGetLatestDate:

    @pytest.fixture(autouse=True)
    def seed(self, db: DBManager):
        rows = [
            {**FUTURES_ROW, "ts_code": "IF2406.CFX", "trade_date": "20240101"},
            {**FUTURES_ROW, "ts_code": "IF2406.CFX", "trade_date": "20240115"},
            {**FUTURES_ROW, "ts_code": "IH2406.CFX", "trade_date": "20240110"},
        ]
        db.upsert_dataframe("futures_daily", pd.DataFrame(rows))

    def test_latest_date_whole_table(self, db: DBManager):
        assert db.get_latest_date("futures_daily") == "20240115"

    def test_latest_date_by_ts_code(self, db: DBManager):
        assert db.get_latest_date("futures_daily", "IF2406.CFX") == "20240115"
        assert db.get_latest_date("futures_daily", "IH2406.CFX") == "20240110"

    def test_latest_date_like_pattern(self, db: DBManager):
        """LIKE 模式应返回匹配合约中的最大日期"""
        result = db.get_latest_date("futures_daily", "%IF%")
        assert result == "20240115"

    def test_empty_table_returns_none(self, db: DBManager):
        assert db.get_latest_date("options_daily") is None

    def test_no_match_returns_none(self, db: DBManager):
        assert db.get_latest_date("futures_daily", "XX9999.CFX") is None

    def test_get_max_date_compat(self, db: DBManager):
        """向后兼容别名"""
        assert db.get_max_date("futures_daily") == "20240115"
