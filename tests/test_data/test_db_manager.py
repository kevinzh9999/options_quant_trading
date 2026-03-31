"""
test_db_manager.py
------------------
tests/test_data/ 子集：聚焦规格要求的6项检查。
完整测试见 tests/test_db_manager.py。

覆盖：
- test_create_tables
- test_upsert_futures_daily
- test_upsert_duplicate
- test_get_futures_daily_date_range
- test_get_options_chain
- test_get_latest_date
"""

from __future__ import annotations

import pandas as pd
import pytest

from data.storage.db_manager import DBManager


@pytest.fixture
def db():
    m = DBManager(":memory:")
    yield m
    m.close()


FUTURES_ROW = {
    "ts_code": "IF2406.CFX", "trade_date": "20240102",
    "open": 4010.0, "high": 4060.0, "low": 4000.0,
    "close": 4040.0, "volume": 5000.0, "oi": 20000.0, "settle": 4035.0,
    "pre_close": 4000.0, "pre_settle": 3990.0,
}

OPT_C = {
    "ts_code": "IO2406-C-3800.CFX", "trade_date": "20240102",
    "exchange": "CFFEX", "underlying_code": "IO",
    "exercise_price": 3800.0, "call_put": "C", "expire_date": "20240621",
    "close": 250.0, "settle": 245.0, "volume": 500.0, "oi": 1200.0,
    "pre_close": 240.0, "pre_settle": 238.0,
}
OPT_P = {**OPT_C, "ts_code": "IO2406-P-3800.CFX", "call_put": "P",
         "close": 120.0, "settle": 118.0}

CONTRACT_C = {
    "ts_code": "IO2406-C-3800.CFX", "exchange": "CFFEX",
    "underlying_code": "IO", "exercise_price": 3800.0, "call_put": "C",
    "expire_date": "20240621", "list_date": "20230921",
    "delist_date": "20240621", "contract_unit": 100.0, "exercise_type": "E",
}
CONTRACT_P = {**CONTRACT_C, "ts_code": "IO2406-P-3800.CFX", "call_put": "P"}


def test_create_tables(db):
    """确认所有主要表创建成功"""
    for table in ["futures_daily", "futures_min", "options_daily",
                  "options_contracts", "commodity_daily", "trade_calendar"]:
        assert db.table_exists(table), f"表 {table} 未创建"


def test_upsert_futures_daily(db):
    """写入测试数据并查询验证"""
    n = db.upsert_dataframe("futures_daily", pd.DataFrame([FUTURES_ROW]))
    assert n == 1
    df = db.get_futures_daily("IF2406.CFX", "20240101", "20240131")
    assert len(df) == 1
    assert df.iloc[0]["close"] == 4040.0


def test_upsert_duplicate(db):
    """写入重复数据验证 upsert（INSERT OR REPLACE）行为"""
    db.upsert_dataframe("futures_daily", pd.DataFrame([FUTURES_ROW]))
    updated = {**FUTURES_ROW, "close": 9999.0}
    db.upsert_dataframe("futures_daily", pd.DataFrame([updated]))
    count = db.query_scalar("SELECT COUNT(*) FROM futures_daily")
    assert count == 1  # 不产生重复行
    close = db.query_scalar(
        "SELECT close FROM futures_daily WHERE ts_code=? AND trade_date=?",
        ("IF2406.CFX", "20240102"),
    )
    assert close == 9999.0  # 值已更新


def test_get_futures_daily_date_range(db):
    """验证日期范围查询"""
    rows = [{**FUTURES_ROW, "trade_date": f"2024010{i}"} for i in range(1, 6)]
    db.upsert_dataframe("futures_daily", pd.DataFrame(rows))

    df = db.get_futures_daily("IF2406.CFX", "20240102", "20240104")
    assert len(df) == 3
    assert df["trade_date"].min() == "20240102"
    assert df["trade_date"].max() == "20240104"


def test_get_options_chain(db):
    """验证期权链查询的 LEFT JOIN 逻辑"""
    # 合约（含远月）
    far = {**CONTRACT_C, "ts_code": "IO2409-C-3800.CFX", "expire_date": "20240920",
           "delist_date": "20240920"}
    db.upsert_dataframe("options_contracts",
                        pd.DataFrame([CONTRACT_C, CONTRACT_P, far]))
    # 日线只有近月两条
    db.upsert_dataframe("options_daily", pd.DataFrame([OPT_C, OPT_P]))

    df = db.get_options_chain("IO", "20240102")
    assert len(df) == 3  # 2 近月 + 1 远月（无日线，close=NaN）

    far_row = df[df["ts_code"] == "IO2409-C-3800.CFX"]
    assert len(far_row) == 1
    assert pd.isna(far_row.iloc[0]["close"])

    near = db.get_options_chain("IO", "20240102", expire_date="20240621")
    assert len(near) == 2
    assert near.loc[near["call_put"] == "C", "close"].iloc[0] == 250.0


def test_get_latest_date(db):
    """验证最新日期查询"""
    rows = [
        {**FUTURES_ROW, "ts_code": "IF2406.CFX", "trade_date": "20240101"},
        {**FUTURES_ROW, "ts_code": "IF2406.CFX", "trade_date": "20240115"},
        {**FUTURES_ROW, "ts_code": "IH2406.CFX", "trade_date": "20240110"},
    ]
    db.upsert_dataframe("futures_daily", pd.DataFrame(rows))

    assert db.get_latest_date("futures_daily") == "20240115"
    assert db.get_latest_date("futures_daily", "IF2406.CFX") == "20240115"
    assert db.get_latest_date("futures_daily", "IH2406.CFX") == "20240110"
    assert db.get_latest_date("options_daily") is None  # 空表
