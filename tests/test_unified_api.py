"""
test_unified_api.py
-------------------
测试 data/unified_api.py

覆盖：
- _normalize_symbol：多种输入格式 -> Tushare ts_code
- tushare_to_tq_symbol / tq_to_tushare_symbol：代码转换
- get_futures_daily：本地命中 / 本地未命中触发 Tushare / auto_download=False
- get_futures_min：本地命中 / 回落 Tushare / TqSdk 补充路径
- get_options_daily：本地命中 / Tushare 补充
- get_options_chain：委托 db.get_options_chain
- get_options_contracts：active_on 日期过滤
- get_trade_calendar：本地命中 / 回落 Tushare
- get_latest_trading_date：本地命中 / 回落 Tushare
- get_index_daily：直接从 Tushare / 无 token 返回空
- ensure_data_available：有数据 / 无数据
- get_trade_dates / is_trade_date：向后兼容接口
- _next_date 工具函数
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from pathlib import Path
import pandas as pd
import pytest

from data.unified_api import UnifiedDataAPI, _next_date
from config import Config


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    """临时 DB，Tushare token 置空（禁用自动补充）"""
    return Config({
        "database": {"path": str(tmp_path / "test.db")},
        "tushare": {"token": ""},
    })


@pytest.fixture
def api(cfg: Config) -> UnifiedDataAPI:
    return UnifiedDataAPI(cfg)


@pytest.fixture
def cfg_with_token(tmp_path: Path) -> Config:
    return Config({
        "database": {"path": str(tmp_path / "test.db")},
        "tushare": {"token": "FAKE_TOKEN"},
    })


@pytest.fixture
def api_with_token(cfg_with_token: Config) -> UnifiedDataAPI:
    return UnifiedDataAPI(cfg_with_token)


# ======================================================================
# _normalize_symbol
# ======================================================================

class TestNormalizeSymbol:

    @pytest.mark.parametrize("symbol,expected", [
        # 已是 Tushare 格式
        ("IF2406.CFX",            "IF2406.CFX"),
        ("IO2406-C-3800.CFX",     "IO2406-C-3800.CFX"),
        ("RBL4.SHF",              "RBL4.SHF"),
        ("MA409.ZCE",             "MA409.ZCE"),
        ("SC2406.INE",            "SC2406.INE"),
        ("SI2406.GFX",            "SI2406.GFX"),
        # TqSdk 格式
        ("CFFEX.IF2406",          "IF2406.CFX"),
        ("CFFEX.IO2406-C-3800",   "IO2406-C-3800.CFX"),
        ("SHFE.RBL4",             "RBL4.SHF"),
        ("CZCE.MA409",            "MA409.ZCE"),
        # 裸合约代码（带月份）
        ("IF2406",                "IF2406.CFX"),
        ("IH2406",                "IH2406.CFX"),
        ("RB2501",                "RB2501.SHF"),
        ("MA409",                 "MA409.ZCE"),
        # 裸品种代码（主力）
        ("IF",                    "IF.CFX"),
        ("IH",                    "IH.CFX"),
        ("RB",                    "RB.SHF"),
    ])
    def test_normalize(self, api: UnifiedDataAPI, symbol, expected):
        assert api._normalize_symbol(symbol) == expected

    def test_unknown_product_returns_as_is(self, api: UnifiedDataAPI):
        assert api._normalize_symbol("UNKNOWN") == "UNKNOWN"

    def test_already_tushare_stock(self, api: UnifiedDataAPI):
        assert api._normalize_symbol("510050.SH") == "510050.SH"

    def test_tq_sse_format(self, api: UnifiedDataAPI):
        assert api._normalize_symbol("SSE.510050") == "510050.SH"


# ======================================================================
# 代码映射
# ======================================================================

class TestSymbolConversion:

    @pytest.mark.parametrize("ts_code,expected_tq", [
        ("IF2406.CFX",          "CFFEX.IF2406"),
        ("IO2406-C-3800.CFX",   "CFFEX.IO2406-C-3800"),
        ("510050.SH",           "SSE.510050"),
        ("000300.SZ",           "SZSE.000300"),
    ])
    def test_tushare_to_tq(self, ts_code, expected_tq):
        assert UnifiedDataAPI.tushare_to_tq_symbol(ts_code) == expected_tq

    @pytest.mark.parametrize("tq_symbol,expected_ts", [
        ("CFFEX.IF2406",            "IF2406.CFX"),
        ("CFFEX.IO2406-C-3800",     "IO2406-C-3800.CFX"),
        ("SSE.510050",              "510050.SH"),
        ("SZSE.000300",             "000300.SZ"),
    ])
    def test_tq_to_tushare(self, tq_symbol, expected_ts):
        assert UnifiedDataAPI.tq_to_tushare_symbol(tq_symbol) == expected_ts

    def test_roundtrip(self):
        original = "IF2406.CFX"
        assert UnifiedDataAPI.tq_to_tushare_symbol(
            UnifiedDataAPI.tushare_to_tq_symbol(original)
        ) == original


# ======================================================================
# get_futures_daily
# ======================================================================

FUTURES_ROW = {
    "ts_code": "IF2406.CFX", "trade_date": "20240102",
    "open": 4010.0, "high": 4060.0, "low": 4000.0,
    "close": 4040.0, "volume": 5000.0, "oi": 20000.0, "settle": 4035.0,
}


class TestGetFuturesDaily:

    def test_returns_local_data(self, api: UnifiedDataAPI):
        api.db.upsert_rows("futures_daily", [FUTURES_ROW])
        df = api.get_futures_daily("IF2406.CFX", "20240101", "20240131")
        assert len(df) == 1
        assert df.iloc[0]["ts_code"] == "IF2406.CFX"

    def test_normalizes_tq_symbol(self, api: UnifiedDataAPI):
        api.db.upsert_rows("futures_daily", [FUTURES_ROW])
        df = api.get_futures_daily("CFFEX.IF2406", "20240101", "20240131")
        assert len(df) == 1

    def test_normalizes_bare_code(self, api: UnifiedDataAPI):
        api.db.upsert_rows("futures_daily", [FUTURES_ROW])
        df = api.get_futures_daily("IF2406", "20240101", "20240131")
        assert len(df) == 1

    def test_no_tushare_when_token_empty(self, api: UnifiedDataAPI):
        with patch.object(api.tushare, "get_futures_daily") as mock_ts:
            df = api.get_futures_daily("IF2406.CFX", "20240101", "20240131")
        mock_ts.assert_not_called()
        assert df.empty

    def test_auto_download_false_no_tushare(self, api_with_token: UnifiedDataAPI):
        with patch.object(api_with_token.tushare, "get_futures_daily") as mock_ts:
            api_with_token.get_futures_daily(
                "IF2406.CFX", "20240101", "20240131", auto_download=False
            )
        mock_ts.assert_not_called()

    def test_tushare_called_when_local_empty(self, api_with_token: UnifiedDataAPI):
        remote = pd.DataFrame([FUTURES_ROW])
        with patch.object(api_with_token.tushare, "get_futures_daily", return_value=remote):
            df = api_with_token.get_futures_daily("IF2406.CFX", "20240101", "20240131")
        assert len(df) == 1

    def test_tushare_fills_missing_tail(self, api_with_token: UnifiedDataAPI):
        """本地有早期数据，只补充尾部缺失"""
        api_with_token.db.upsert_rows("futures_daily", [FUTURES_ROW])
        new_row = {**FUTURES_ROW, "trade_date": "20240103"}
        remote = pd.DataFrame([new_row])
        with patch.object(api_with_token.tushare, "get_futures_daily",
                          return_value=remote) as mock_ts:
            df = api_with_token.get_futures_daily("IF2406.CFX", "20240101", "20240131")
        # Tushare 应只被调用从 20240103 开始（fill_start = next_date("20240102")）
        args, _ = mock_ts.call_args
        assert args[1] >= "20240103"  # start_date 参数
        assert len(df) == 2

    def test_tushare_not_called_when_local_covers(self, api_with_token: UnifiedDataAPI):
        """本地数据已覆盖整个范围，不应调用 Tushare"""
        api_with_token.db.upsert_rows("futures_daily", [FUTURES_ROW])
        # Seed latest date to end_date or beyond
        row_at_end = {**FUTURES_ROW, "trade_date": "20240131"}
        api_with_token.db.upsert_rows("futures_daily", [row_at_end])
        with patch.object(api_with_token.tushare, "get_futures_daily") as mock_ts:
            api_with_token.get_futures_daily("IF2406.CFX", "20240101", "20240131")
        mock_ts.assert_not_called()

    def test_tushare_failure_returns_local(self, api_with_token: UnifiedDataAPI):
        api_with_token.db.upsert_rows("futures_daily", [FUTURES_ROW])
        with patch.object(api_with_token.tushare, "get_futures_daily",
                          side_effect=RuntimeError("API error")):
            df = api_with_token.get_futures_daily("IF2406.CFX", "20240101", "20240131")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1  # 本地数据仍返回


# ======================================================================
# get_futures_min
# ======================================================================

MIN_ROW = {
    "ts_code": "IF2406.CFX", "datetime": "2024-01-02 09:35:00",
    "open": 4010.0, "high": 4015.0, "low": 4005.0,
    "close": 4012.0, "volume": 100.0,
}


class TestGetFuturesMin:

    def test_returns_local_data(self, api: UnifiedDataAPI):
        api.db.upsert_rows("futures_min", [MIN_ROW])
        df = api.get_futures_min("IF2406.CFX", "20240102", "20240102")
        assert len(df) == 1

    def test_normalizes_symbol(self, api: UnifiedDataAPI):
        api.db.upsert_rows("futures_min", [MIN_ROW])
        df = api.get_futures_min("CFFEX.IF2406", "20240102", "20240102")
        assert len(df) == 1

    def test_tushare_called_when_empty(self, api_with_token: UnifiedDataAPI):
        remote = pd.DataFrame([MIN_ROW])
        with patch.object(api_with_token.tushare, "get_futures_min", return_value=remote):
            df = api_with_token.get_futures_min("IF2406.CFX", "20240102", "20240102")
        assert len(df) == 1

    def test_no_tushare_when_token_empty(self, api: UnifiedDataAPI):
        with patch.object(api.tushare, "get_futures_min") as mock_ts:
            df = api.get_futures_min("IF2406.CFX", "20240102", "20240102")
        mock_ts.assert_not_called()
        assert df.empty

    def test_tq_fallback_when_available(self, api_with_token: UnifiedDataAPI):
        """TqClient 已连接时，Tushare 失败后走 TqSdk 路径"""
        mock_klines = pd.DataFrame([{
            "datetime": pd.Timestamp("2024-01-02 09:35:00", tz="Asia/Shanghai"),
            "open": 4010.0, "high": 4015.0, "low": 4005.0,
            "close": 4012.0, "volume": 100.0,
            "open_oi": 20000.0, "close_oi": 20050.0,
        }])
        mock_tq = MagicMock()
        mock_tq._api = MagicMock()  # 模拟已连接
        mock_tq.get_kline.return_value = mock_klines
        api_with_token._tq = mock_tq

        with patch.object(api_with_token.tushare, "get_futures_min",
                          side_effect=RuntimeError("timeout")):
            df = api_with_token.get_futures_min("IF2406.CFX", "20240102", "20240102")

        mock_tq.get_kline.assert_called_once()
        assert len(df) == 1

    def test_tushare_failure_returns_empty(self, api_with_token: UnifiedDataAPI):
        with patch.object(api_with_token.tushare, "get_futures_min",
                          side_effect=RuntimeError("API error")):
            df = api_with_token.get_futures_min("IF2406.CFX", "20240102", "20240102")
        assert isinstance(df, pd.DataFrame)


# ======================================================================
# get_options_daily
# ======================================================================

OPT_ROW = {
    "ts_code": "IO2406-C-3800.CFX", "trade_date": "20240102",
    "exchange": "CFFEX", "underlying_code": "IO",
    "exercise_price": 3800.0, "call_put": "C", "expire_date": "20240621",
    "close": 250.0, "settle": 245.0, "volume": 500.0, "oi": 1200.0,
}


class TestGetOptionsDaily:

    def test_returns_local_data(self, api: UnifiedDataAPI):
        api.db.upsert_rows("options_daily", [OPT_ROW])
        df = api.get_options_daily("IO", "20240102")
        assert len(df) == 1

    def test_filter_call_put(self, api: UnifiedDataAPI):
        put_row = {**OPT_ROW, "ts_code": "IO2406-P-3800.CFX", "call_put": "P"}
        api.db.upsert_rows("options_daily", [OPT_ROW, put_row])
        df_c = api.get_options_daily("IO", "20240102", call_put="C")
        assert len(df_c) == 1
        assert df_c.iloc[0]["call_put"] == "C"
        df_p = api.get_options_daily("IO", "20240102", call_put="P")
        assert len(df_p) == 1

    def test_no_match_returns_empty(self, api: UnifiedDataAPI):
        df = api.get_options_daily("IO", "99991231")
        assert df.empty

    def test_tushare_called_when_local_empty(self, api_with_token: UnifiedDataAPI):
        remote = pd.DataFrame([OPT_ROW])
        with patch.object(api_with_token.tushare, "get_options_daily",
                          return_value=remote) as mock_ts:
            df = api_with_token.get_options_daily("IO", "20240102")
        mock_ts.assert_called_once_with(exchange="CFFEX", trade_date="20240102")
        assert len(df) == 1

    def test_no_tushare_when_token_empty(self, api: UnifiedDataAPI):
        with patch.object(api.tushare, "get_options_daily") as mock_ts:
            api.get_options_daily("IO", "20240102")
        mock_ts.assert_not_called()


# ======================================================================
# get_options_chain
# ======================================================================

CONTRACT_ROW = {
    "ts_code": "IO2406-C-3800.CFX", "exchange": "CFFEX",
    "underlying_code": "IO", "exercise_price": 3800.0, "call_put": "C",
    "expire_date": "20240621", "list_date": "20230921", "delist_date": "20240621",
    "contract_unit": 100, "exercise_type": "E",
}


class TestGetOptionsChain:

    def test_returns_chain_data(self, api: UnifiedDataAPI):
        api.db.upsert_rows("options_contracts", [CONTRACT_ROW])
        api.db.upsert_rows("options_daily", [OPT_ROW])
        df = api.get_options_chain("IO", "20240102")
        assert len(df) == 1
        assert "exercise_price" in df.columns
        assert "close" in df.columns

    def test_filter_by_expire_date(self, api: UnifiedDataAPI):
        api.db.upsert_rows("options_contracts", [CONTRACT_ROW])
        df = api.get_options_chain("IO", "20240102", expire_date="20240621")
        assert len(df) == 1
        df2 = api.get_options_chain("IO", "20240102", expire_date="20241231")
        assert df2.empty

    def test_left_join_no_daily(self, api: UnifiedDataAPI):
        """合约存在但当日无行情 -> 仍返回合约行，价格为 NaN"""
        api.db.upsert_rows("options_contracts", [CONTRACT_ROW])
        df = api.get_options_chain("IO", "20991231")  # 无行情数据
        assert len(df) == 1
        assert pd.isna(df.iloc[0]["close"])


# ======================================================================
# get_options_contracts
# ======================================================================

class TestGetOptionsContracts:

    def test_returns_all(self, api: UnifiedDataAPI):
        api.db.upsert_rows("options_contracts", [CONTRACT_ROW])
        df = api.get_options_contracts()
        assert len(df) == 1

    def test_filter_by_underlying(self, api: UnifiedDataAPI):
        api.db.upsert_rows("options_contracts", [CONTRACT_ROW])
        df = api.get_options_contracts(underlying="IO")
        assert len(df) == 1
        df2 = api.get_options_contracts(underlying="MO")
        assert df2.empty

    def test_active_on_filter(self, api: UnifiedDataAPI):
        api.db.upsert_rows("options_contracts", [CONTRACT_ROW])
        assert api.get_options_contracts(active_on="20230920").empty  # 上市前
        assert len(api.get_options_contracts(active_on="20240101")) == 1  # 活跃中
        assert api.get_options_contracts(active_on="20240622").empty  # 到期后


# ======================================================================
# get_trade_calendar
# ======================================================================

CAL_ROWS = [
    {"exchange": "CFFEX", "trade_date": "20240102", "is_open": 1, "pretrade_date": "20231229"},
    {"exchange": "CFFEX", "trade_date": "20240103", "is_open": 1, "pretrade_date": "20240102"},
    {"exchange": "CFFEX", "trade_date": "20240104", "is_open": 0, "pretrade_date": "20240103"},
]


class TestGetTradeCalendar:

    def test_returns_open_days_only(self, api: UnifiedDataAPI):
        api.db.upsert_rows("trade_calendar", CAL_ROWS)
        df = api.get_trade_calendar("CFFEX", "20240101", "20240110")
        assert len(df) == 2
        assert "20240104" not in df["trade_date"].tolist()  # is_open=0

    def test_returns_dataframe(self, api: UnifiedDataAPI):
        api.db.upsert_rows("trade_calendar", CAL_ROWS)
        df = api.get_trade_calendar("CFFEX")
        assert isinstance(df, pd.DataFrame)
        assert "trade_date" in df.columns

    def test_empty_when_no_data_no_token(self, api: UnifiedDataAPI):
        df = api.get_trade_calendar("CFFEX", "20240101", "20240131")
        assert df.empty

    def test_tushare_called_when_local_empty(self, api_with_token: UnifiedDataAPI):
        remote = pd.DataFrame([CAL_ROWS[0]])
        with patch.object(api_with_token.tushare, "get_trade_calendar",
                          return_value=remote) as mock_ts:
            df = api_with_token.get_trade_calendar("CFFEX", "20240101", "20240131")
        mock_ts.assert_called_once()
        assert not df.empty


# ======================================================================
# get_latest_trading_date
# ======================================================================

class TestGetLatestTradingDate:

    def test_returns_local_latest(self, api: UnifiedDataAPI):
        api.db.upsert_rows("trade_calendar", CAL_ROWS)
        result = api.get_latest_trading_date("CFFEX")
        assert result == "20240103"  # 最近 is_open=1 日

    def test_tushare_fallback_when_no_local(self, api_with_token: UnifiedDataAPI):
        remote = pd.DataFrame([
            {"exchange": "CFFEX", "trade_date": "20240103",
             "is_open": 1, "pretrade_date": "20240102"},
        ])
        with patch.object(api_with_token.tushare, "get_trade_calendar",
                          return_value=remote):
            result = api_with_token.get_latest_trading_date("CFFEX")
        assert result == "20240103"

    def test_returns_str(self, api: UnifiedDataAPI):
        api.db.upsert_rows("trade_calendar", CAL_ROWS)
        result = api.get_latest_trading_date("CFFEX")
        assert isinstance(result, str)
        assert len(result) == 8


# ======================================================================
# get_index_daily
# ======================================================================

class TestGetIndexDaily:

    def test_no_token_returns_empty(self, api: UnifiedDataAPI):
        df = api.get_index_daily("000300", "20240101", "20240131")
        assert df.empty

    def test_normalizes_bare_code(self, api_with_token: UnifiedDataAPI):
        remote = pd.DataFrame([{
            "ts_code": "000300.SH", "trade_date": "20240102",
            "open": 3400.0, "high": 3450.0, "low": 3390.0,
            "close": 3420.0, "volume": 150000.0,
        }])
        with patch.object(api_with_token.tushare, "get_index_daily",
                          return_value=remote) as mock_ts:
            df = api_with_token.get_index_daily("000300", "20240101", "20240131")
        mock_ts.assert_called_once_with("000300.SH", "20240101", "20240131")
        assert len(df) == 1

    def test_accepts_full_ts_code(self, api_with_token: UnifiedDataAPI):
        remote = pd.DataFrame([{
            "ts_code": "000905.SH", "trade_date": "20240102",
            "open": 5000.0, "high": 5050.0, "low": 4990.0,
            "close": 5020.0, "volume": 80000.0,
        }])
        with patch.object(api_with_token.tushare, "get_index_daily",
                          return_value=remote) as mock_ts:
            api_with_token.get_index_daily("000905.SH", "20240101", "20240131")
        mock_ts.assert_called_once_with("000905.SH", "20240101", "20240131")

    def test_tushare_failure_returns_empty(self, api_with_token: UnifiedDataAPI):
        with patch.object(api_with_token.tushare, "get_index_daily",
                          side_effect=RuntimeError("timeout")):
            df = api_with_token.get_index_daily("000300", "20240101", "20240131")
        assert df.empty


# ======================================================================
# ensure_data_available
# ======================================================================

class TestEnsureDataAvailable:

    def test_returns_true_when_data_exists(self, api: UnifiedDataAPI):
        api.db.upsert_rows("futures_daily", [FUTURES_ROW])
        assert api.ensure_data_available("IF2406.CFX", "20240101", "20240131") is True

    def test_returns_false_when_no_data_no_token(self, api: UnifiedDataAPI):
        assert api.ensure_data_available("IF2406.CFX", "20240101", "20240131") is False

    def test_downloads_and_returns_true(self, api_with_token: UnifiedDataAPI):
        remote = pd.DataFrame([FUTURES_ROW])
        with patch.object(api_with_token.tushare, "get_futures_daily",
                          return_value=remote):
            result = api_with_token.ensure_data_available(
                "IF2406.CFX", "20240101", "20240131"
            )
        assert result is True

    def test_unknown_data_type_returns_false(self, api: UnifiedDataAPI):
        assert api.ensure_data_available(
            "IF2406.CFX", "20240101", "20240131", data_type="unknown"
        ) is False


# ======================================================================
# 向后兼容：get_trade_dates / is_trade_date
# ======================================================================

class TestTradeDates:

    def _seed_calendar(self, api: UnifiedDataAPI):
        rows = [
            {"exchange": "SSE", "trade_date": "20240102", "is_open": 1, "pretrade_date": "20231229"},
            {"exchange": "SSE", "trade_date": "20240103", "is_open": 1, "pretrade_date": "20240102"},
            {"exchange": "SSE", "trade_date": "20240104", "is_open": 0, "pretrade_date": "20240103"},
        ]
        api.db.upsert_rows("trade_calendar", rows)

    def test_returns_open_dates_only(self, api: UnifiedDataAPI):
        self._seed_calendar(api)
        dates = api.get_trade_dates("20240101", "20240110")
        assert "20240102" in dates
        assert "20240103" in dates
        assert "20240104" not in dates

    def test_empty_when_no_data_no_token(self, api: UnifiedDataAPI):
        assert api.get_trade_dates("20240101", "20240131") == []

    def test_is_trade_date_true(self, api: UnifiedDataAPI):
        self._seed_calendar(api)
        assert api.is_trade_date("20240102") is True

    def test_is_trade_date_false_non_trading(self, api: UnifiedDataAPI):
        self._seed_calendar(api)
        assert api.is_trade_date("20240104") is False


# ======================================================================
# _next_date 工具函数
# ======================================================================

class TestNextDate:

    def test_normal(self):
        assert _next_date("20240131") == "20240201"

    def test_end_of_year(self):
        assert _next_date("20231231") == "20240101"
