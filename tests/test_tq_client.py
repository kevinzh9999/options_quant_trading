"""
test_tq_client.py
-----------------
测试 data/sources/tq_client.py

TqSdk 是实盘依赖库，所有测试均使用 mock，不建立真实连接。
覆盖：
- connect() 前调用任何方法抛出 RuntimeError
- connect() 已连接时跳过重复初始化
- tqsdk 未安装时 connect() 抛出 ImportError
- connect / disconnect 状态管理
- context manager (__enter__/__exit__)
- get_kline 列名规范化、datetime 转换、close_oi 兼容
- get_quote 返回字段子集
- get_option_quotes 返回 DataFrame
- subscribe_quotes callback 被调用
- convert_symbol_tq_to_ts / convert_symbol_ts_to_tq 全交易所映射
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from data.sources.tq_client import TqClient


# ======================================================================
# Helpers
# ======================================================================

def make_client() -> TqClient:
    return TqClient(auth_account="TEST_ACCOUNT", auth_password="TEST_PASS")


def inject_mock_api(client: TqClient) -> MagicMock:
    """直接注入一个 mock api，跳过 connect()"""
    api = MagicMock()
    client._api = api
    return api


def make_mock_tqsdk() -> MagicMock:
    """返回一个模拟 tqsdk 模块"""
    mock_tqsdk = MagicMock()
    mock_tqsdk.TqApi.return_value = MagicMock()
    return mock_tqsdk


# ======================================================================
# 未连接状态保护
# ======================================================================

class TestNotConnected:

    def test_get_quote_raises(self):
        c = make_client()
        with pytest.raises(RuntimeError, match="尚未连接"):
            c.get_quote("CFFEX.IF2406")

    def test_get_kline_raises(self):
        c = make_client()
        with pytest.raises(RuntimeError, match="尚未连接"):
            c.get_kline("CFFEX.IF2406", 300)

    def test_get_option_quotes_raises(self):
        c = make_client()
        with pytest.raises(RuntimeError, match="尚未连接"):
            c.get_option_quotes("IO", "CFFEX")

    def test_subscribe_quotes_raises(self):
        c = make_client()
        with pytest.raises(RuntimeError, match="尚未连接"):
            c.subscribe_quotes(["CFFEX.IF2406"], callback=lambda s, q: None)


# ======================================================================
# connect / disconnect
# ======================================================================

class TestConnect:

    def test_connect_sets_api(self):
        c = make_client()
        mock_tqsdk = make_mock_tqsdk()
        with patch.dict("sys.modules", {"tqsdk": mock_tqsdk}):
            c.connect()
        assert c._api is not None
        mock_tqsdk.TqApi.assert_called_once()

    def test_connect_passes_auth(self):
        c = make_client()
        mock_tqsdk = make_mock_tqsdk()
        with patch.dict("sys.modules", {"tqsdk": mock_tqsdk}):
            c.connect()
        # TqAuth 应使用账户和密码初始化
        mock_tqsdk.TqAuth.assert_called_once_with("TEST_ACCOUNT", "TEST_PASS")

    def test_connect_skips_if_already_connected(self):
        c = make_client()
        existing_api = MagicMock()
        c._api = existing_api
        mock_tqsdk = make_mock_tqsdk()
        with patch.dict("sys.modules", {"tqsdk": mock_tqsdk}):
            c.connect()
        # 不应再次调用 TqApi
        mock_tqsdk.TqApi.assert_not_called()
        assert c._api is existing_api

    def test_connect_raises_import_error_when_tqsdk_missing(self):
        c = make_client()
        with patch.dict("sys.modules", {"tqsdk": None}):
            with pytest.raises(ImportError, match="tqsdk 未安装"):
                c.connect()

    def test_disconnect_clears_api(self):
        c = make_client()
        api = inject_mock_api(c)
        c.disconnect()
        assert c._api is None
        api.close.assert_called_once()

    def test_disconnect_noop_when_not_connected(self):
        c = make_client()
        c.disconnect()  # 不应抛异常

    def test_context_manager(self):
        c = make_client()
        mock_tqsdk = make_mock_tqsdk()
        with patch.dict("sys.modules", {"tqsdk": mock_tqsdk}):
            with c:
                assert c._api is not None
        assert c._api is None


# ======================================================================
# get_kline
# ======================================================================

class TestGetKline:

    def _make_kline_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "datetime": [1704153600_000_000_000, 1704153900_000_000_000],
            "open":     [4010.0, 4020.0],
            "high":     [4015.0, 4025.0],
            "low":      [4005.0, 4015.0],
            "close":    [4012.0, 4022.0],
            "volume":   [100.0, 120.0],
            "open_oi":  [20000.0, 20100.0],
            "close_oi": [20050.0, 20150.0],
        })

    def test_returns_required_columns(self):
        c = make_client()
        api = inject_mock_api(c)
        api.get_kline_serial.return_value = self._make_kline_df()
        df = c.get_kline("CFFEX.IF2406", 300)
        for col in ["datetime", "open", "high", "low", "close",
                    "volume", "open_oi", "close_oi"]:
            assert col in df.columns, f"缺少列: {col}"

    def test_integer_datetime_converted(self):
        c = make_client()
        api = inject_mock_api(c)
        api.get_kline_serial.return_value = self._make_kline_df()
        df = c.get_kline("CFFEX.IF2406", 300)
        assert pd.api.types.is_datetime64_any_dtype(df["datetime"])

    def test_close_interest_renamed_to_close_oi(self):
        c = make_client()
        api = inject_mock_api(c)
        raw = self._make_kline_df().drop(columns=["close_oi"])
        raw["close_interest"] = [20050.0, 20150.0]
        api.get_kline_serial.return_value = raw
        df = c.get_kline("CFFEX.IF2406", 300)
        assert "close_oi" in df.columns

    def test_correct_args_passed_to_tqsdk(self):
        c = make_client()
        api = inject_mock_api(c)
        api.get_kline_serial.return_value = self._make_kline_df()
        c.get_kline("CFFEX.IO2406-C-3800", 86400, data_length=500)
        api.get_kline_serial.assert_called_once_with("CFFEX.IO2406-C-3800", 86400, 500)

    def test_default_data_length(self):
        c = make_client()
        api = inject_mock_api(c)
        api.get_kline_serial.return_value = self._make_kline_df()
        c.get_kline("CFFEX.IF2406", 300)
        _, args, _ = api.get_kline_serial.mock_calls[0]
        assert args[2] == 8964


# ======================================================================
# get_quote
# ======================================================================

class TestGetQuote:

    def test_returns_dict(self):
        c = make_client()
        api = inject_mock_api(c)
        mock_quote = MagicMock()
        mock_quote.last_price = 4050.0
        mock_quote.bid_price1 = 4049.0
        mock_quote.ask_price1 = 4051.0
        api.get_quote.return_value = mock_quote
        result = c.get_quote("CFFEX.IF2406")
        assert isinstance(result, dict)
        assert result["last_price"] == 4050.0

    def test_missing_fields_return_none(self):
        c = make_client()
        api = inject_mock_api(c)

        class MinimalQuote:
            last_price = 4000.0

        api.get_quote.return_value = MinimalQuote()
        result = c.get_quote("CFFEX.IF2406")
        assert result["last_price"] == 4000.0
        assert result.get("strike_price") is None


# ======================================================================
# get_option_quotes
# ======================================================================

class TestGetOptionQuotes:

    def test_returns_dataframe(self):
        c = make_client()
        api = inject_mock_api(c)
        mock_quote = MagicMock()
        mock_quote.last_price = 250.0
        api.query_options.return_value = ["CFFEX.IO2406-C-3800", "CFFEX.IO2406-P-3800"]
        api.get_quote.return_value = mock_quote
        df = c.get_option_quotes("IO", "CFFEX")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "symbol" in df.columns

    def test_query_options_called_with_underlying(self):
        c = make_client()
        api = inject_mock_api(c)
        api.query_options.return_value = []
        c.get_option_quotes("IO", "CFFEX")
        api.query_options.assert_called_once_with(underlying_symbol="CFFEX.IO")

    def test_empty_returns_empty_dataframe(self):
        c = make_client()
        api = inject_mock_api(c)
        api.query_options.return_value = []
        df = c.get_option_quotes("IO", "CFFEX")
        assert df.empty
        assert "symbol" in df.columns


# ======================================================================
# subscribe_quotes
# ======================================================================

class TestSubscribeQuotes:

    def test_callback_called_on_update(self):
        c = make_client()
        api = inject_mock_api(c)
        mock_quote = MagicMock()
        api.get_quote.return_value = mock_quote

        received = []

        def callback(symbol, quote_dict):
            received.append((symbol, quote_dict))

        # wait_update 第一次返回正常，第二次抛 KeyboardInterrupt 退出
        api.wait_update.side_effect = [None, KeyboardInterrupt()]
        api.is_changing.return_value = True

        c.subscribe_quotes(["CFFEX.IF2406"], callback=callback)

        assert len(received) == 1
        assert received[0][0] == "CFFEX.IF2406"

    def test_callback_not_called_when_not_changing(self):
        c = make_client()
        api = inject_mock_api(c)
        api.get_quote.return_value = MagicMock()
        api.wait_update.side_effect = [None, KeyboardInterrupt()]
        api.is_changing.return_value = False

        received = []
        c.subscribe_quotes(["CFFEX.IF2406"], callback=lambda s, q: received.append(s))

        assert len(received) == 0


# ======================================================================
# convert_symbol_tq_to_ts
# ======================================================================

class TestConvertSymbolTqToTs:

    @pytest.mark.parametrize("tq_symbol,expected", [
        ("CFFEX.IF2406",          "IF2406.CFX"),
        ("CFFEX.IO2406-C-3800",   "IO2406-C-3800.CFX"),
        ("SHFE.RBL4",             "RBL4.SHF"),
        ("DCE.ML4",               "ML4.DCE"),
        ("CZCE.MA409",            "MA409.ZCE"),
        ("INE.SC2406",            "SC2406.INE"),
        ("GFEX.SI2406",           "SI2406.GFX"),
    ])
    def test_mapping(self, tq_symbol, expected):
        assert TqClient.convert_symbol_tq_to_ts(tq_symbol) == expected


# ======================================================================
# convert_symbol_ts_to_tq
# ======================================================================

class TestConvertSymbolTsToTq:

    @pytest.mark.parametrize("ts_symbol,expected", [
        ("IF2406.CFX",          "CFFEX.IF2406"),
        ("IO2406-C-3800.CFX",   "CFFEX.IO2406-C-3800"),
        ("RBL4.SHF",            "SHFE.RBL4"),
        ("ML4.DCE",             "DCE.ML4"),
        ("MA409.ZCE",           "CZCE.MA409"),
        ("SC2406.INE",          "INE.SC2406"),
        ("SI2406.GFX",          "GFEX.SI2406"),
    ])
    def test_mapping(self, ts_symbol, expected):
        assert TqClient.convert_symbol_ts_to_tq(ts_symbol) == expected

    def test_roundtrip(self):
        original = "IO2406-C-3800.CFX"
        assert TqClient.convert_symbol_tq_to_ts(
            TqClient.convert_symbol_ts_to_tq(original)
        ) == original
