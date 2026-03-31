"""
test_tushare_client.py
----------------------
测试 data/sources/tushare_client.py

所有测试都使用 mock，不发起真实 Tushare API 请求。
覆盖：
- 懒加载 _get_api（首次调用时初始化）
- _call_api 成功 / 失败重试 / 超出重试上限 / None 结果
- get_futures_daily：vol→volume 重命名、列裁剪、排序、空结果
- get_futures_min：trade_time→datetime, vol→volume, 排序
- get_futures_mapping：列选取、排序
- get_options_daily(exchange, trade_date)：opt_code→underlying_code, vol→volume
- get_options_contracts：opt_code→underlying_code, maturity_date→expire_date
- get_commodity_daily：委托给 get_futures_daily
- get_trade_calendar：cal_date→trade_date, end_date 默认今天
- get_index_daily：vol→volume, 排序
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from data.sources.tushare_client import TushareClient


# ======================================================================
# Helpers
# ======================================================================

def make_client(sleep: float = 0.0, max_retry: int = 3) -> TushareClient:
    return TushareClient(token="FAKE_TOKEN", max_retry=max_retry, sleep_interval=sleep)


def make_mock_api(**method_returns: pd.DataFrame) -> MagicMock:
    """返回一个模拟 pro_api 对象，每个方法名对应一个 DataFrame"""
    api = MagicMock()
    for method, df in method_returns.items():
        getattr(api, method).return_value = df
    return api


# ======================================================================
# _get_api 懒加载
# ======================================================================

class TestGetApi:

    def test_api_not_initialized_at_construction(self):
        client = make_client()
        assert client._api is None

    def test_api_initialized_lazily(self):
        client = make_client()
        mock_ts = MagicMock()
        mock_ts.pro_api.return_value = MagicMock()
        with patch.dict("sys.modules", {"tushare": mock_ts}):
            api = client._get_api()
        assert api is not None
        mock_ts.set_token.assert_called_once_with("FAKE_TOKEN")
        mock_ts.pro_api.assert_called_once()

    def test_api_initialized_only_once(self):
        client = make_client()
        mock_api = MagicMock()
        client._api = mock_api  # 已设置，不应再次初始化
        result = client._get_api()
        assert result is mock_api


# ======================================================================
# _call_api
# ======================================================================

class TestCallApi:

    def test_success_on_first_try(self):
        client = make_client()
        expected = pd.DataFrame({"a": [1, 2]})
        client._api = make_mock_api(fut_daily=expected)
        with patch("time.sleep"):
            result = client._call_api("fut_daily", ts_code="IF2406.CFX")
        pd.testing.assert_frame_equal(result, expected)
        client._api.fut_daily.assert_called_once_with(ts_code="IF2406.CFX")

    def test_sleep_called_before_each_attempt(self):
        client = make_client(sleep=0.5)
        client._api = make_mock_api(fut_daily=pd.DataFrame({"a": [1]}))
        with patch("time.sleep") as mock_sleep:
            client._call_api("fut_daily")
        # 第一次调用前至少 sleep 一次
        mock_sleep.assert_any_call(0.5)

    def test_retry_on_failure_then_success(self):
        client = make_client()
        expected = pd.DataFrame({"a": [1]})
        mock_api = MagicMock()
        mock_api.fut_daily.side_effect = [RuntimeError("connection error"), expected]
        client._api = mock_api
        with patch("time.sleep"):
            result = client._call_api("fut_daily")
        pd.testing.assert_frame_equal(result, expected)
        assert mock_api.fut_daily.call_count == 2

    def test_raises_after_max_retries(self):
        client = make_client(max_retry=2)
        mock_api = MagicMock()
        mock_api.fut_daily.side_effect = RuntimeError("persistent error")
        client._api = mock_api
        with patch("time.sleep"), pytest.raises(RuntimeError, match="persistent error"):
            client._call_api("fut_daily")
        assert mock_api.fut_daily.call_count == 2

    def test_none_result_returns_empty_df(self):
        client = make_client()
        mock_api = MagicMock()
        mock_api.fut_daily.return_value = None
        client._api = mock_api
        with patch("time.sleep"):
            result = client._call_api("fut_daily")
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_exponential_backoff_on_retry(self):
        client = make_client(max_retry=3)
        mock_api = MagicMock()
        mock_api.fut_daily.side_effect = [
            RuntimeError("err"),
            RuntimeError("err"),
            pd.DataFrame({"a": [1]}),
        ]
        client._api = mock_api
        sleep_calls = []
        with patch("time.sleep", side_effect=lambda s: sleep_calls.append(s)):
            client._call_api("fut_daily")
        # 退避：第一次失败后 sleep(2), 第二次失败后 sleep(4)
        backoff_calls = [s for s in sleep_calls if s > 1]
        assert 2 in backoff_calls


# ======================================================================
# get_futures_daily
# ======================================================================

class TestGetFuturesDaily:

    def _make_raw(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "ts_code": "IF2406.CFX", "trade_date": "20240102",
            "open": 4010.0, "high": 4060.0, "low": 4000.0,
            "close": 4040.0, "vol": 5000.0, "oi": 20000.0,
            "settle": 4030.0, "pre_close": 4005.0, "pre_settle": 4010.0,
        }])

    def test_vol_renamed_to_volume(self):
        client = make_client()
        client._api = make_mock_api(fut_daily=self._make_raw())
        with patch("time.sleep"):
            df = client.get_futures_daily("IF2406.CFX", "20240101", "20240131")
        assert "volume" in df.columns
        assert "vol" not in df.columns

    def test_returns_required_columns(self):
        client = make_client()
        client._api = make_mock_api(fut_daily=self._make_raw())
        with patch("time.sleep"):
            df = client.get_futures_daily("IF2406.CFX", "20240101", "20240131")
        for col in ["ts_code", "trade_date", "open", "high", "low", "close",
                    "volume", "oi", "settle", "pre_close", "pre_settle"]:
            assert col in df.columns, f"缺少列: {col}"

    def test_sorted_by_trade_date(self):
        client = make_client()
        raw = pd.DataFrame([
            {"ts_code": "IF2406.CFX", "trade_date": "20240103",
             "open": 1, "high": 1, "low": 1, "close": 1, "vol": 1, "oi": 1, "settle": 1},
            {"ts_code": "IF2406.CFX", "trade_date": "20240101",
             "open": 1, "high": 1, "low": 1, "close": 1, "vol": 1, "oi": 1, "settle": 1},
        ])
        client._api = make_mock_api(fut_daily=raw)
        with patch("time.sleep"):
            df = client.get_futures_daily("IF2406.CFX", "20240101", "20240131")
        assert df["trade_date"].tolist() == ["20240101", "20240103"]

    def test_empty_returns_empty_df_with_columns(self):
        client = make_client()
        client._api = make_mock_api(fut_daily=pd.DataFrame())
        with patch("time.sleep"):
            df = client.get_futures_daily("IF2406.CFX", "20240101", "20240131")
        assert df.empty
        assert "volume" in df.columns


# ======================================================================
# get_futures_min
# ======================================================================

class TestGetFuturesMin:

    def _make_raw(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "ts_code": "IF2406.CFX",
            "trade_time": "2024-01-02 09:35:00",
            "open": 4010.0, "high": 4015.0, "low": 4005.0,
            "close": 4012.0, "vol": 100.0,
        }])

    def test_trade_time_renamed_to_datetime(self):
        client = make_client()
        client._api = make_mock_api(ft_mins=self._make_raw())
        with patch("time.sleep"):
            df = client.get_futures_min("IF2406.CFX", "20240102", "20240102")
        assert "datetime" in df.columns
        assert "trade_time" not in df.columns

    def test_vol_renamed_to_volume(self):
        client = make_client()
        client._api = make_mock_api(ft_mins=self._make_raw())
        with patch("time.sleep"):
            df = client.get_futures_min("IF2406.CFX", "20240102", "20240102")
        assert "volume" in df.columns
        assert "vol" not in df.columns

    def test_returns_required_columns(self):
        client = make_client()
        client._api = make_mock_api(ft_mins=self._make_raw())
        with patch("time.sleep"):
            df = client.get_futures_min("IF2406.CFX", "20240102", "20240102")
        for col in ["ts_code", "datetime", "open", "high", "low", "close", "volume"]:
            assert col in df.columns

    def test_sorted_by_datetime(self):
        client = make_client()
        raw = pd.DataFrame([
            {"ts_code": "IF2406.CFX", "trade_time": "2024-01-02 10:00:00",
             "open": 1, "high": 1, "low": 1, "close": 1, "vol": 1},
            {"ts_code": "IF2406.CFX", "trade_time": "2024-01-02 09:30:00",
             "open": 1, "high": 1, "low": 1, "close": 1, "vol": 1},
        ])
        client._api = make_mock_api(ft_mins=raw)
        with patch("time.sleep"):
            df = client.get_futures_min("IF2406.CFX", "20240102", "20240102")
        assert df["datetime"].tolist() == ["2024-01-02 09:30:00", "2024-01-02 10:00:00"]

    def test_freq_passed_to_api(self):
        client = make_client()
        mock_api = MagicMock()
        mock_api.ft_mins.return_value = pd.DataFrame()
        client._api = mock_api
        with patch("time.sleep"):
            client.get_futures_min("IF2406.CFX", "20240102", "20240102", freq="1min")
        mock_api.ft_mins.assert_called_once_with(
            ts_code="IF2406.CFX", start_date="20240102", end_date="20240102", freq="1min"
        )

    def test_empty_returns_empty_df_with_columns(self):
        client = make_client()
        client._api = make_mock_api(ft_mins=pd.DataFrame())
        with patch("time.sleep"):
            df = client.get_futures_min("IF2406.CFX", "20240102", "20240102")
        assert df.empty
        assert "datetime" in df.columns


# ======================================================================
# get_futures_mapping
# ======================================================================

class TestGetFuturesMapping:

    def _make_raw(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"ts_code": "IFZ4.CFX", "trade_date": "20240102", "mapping_ts_code": "IF.CFX"},
            {"ts_code": "IFZ4.CFX", "trade_date": "20240101", "mapping_ts_code": "IF.CFX"},
        ])

    def test_returns_required_columns(self):
        client = make_client()
        client._api = make_mock_api(fut_mapping=self._make_raw())
        with patch("time.sleep"):
            df = client.get_futures_mapping(exchange="CFFEX")
        for col in ["ts_code", "trade_date", "mapping_ts_code"]:
            assert col in df.columns

    def test_sorted_by_ts_code_and_trade_date(self):
        client = make_client()
        raw = pd.DataFrame([
            {"ts_code": "IFZ4.CFX", "trade_date": "20240103", "mapping_ts_code": "IF.CFX"},
            {"ts_code": "IFZ4.CFX", "trade_date": "20240101", "mapping_ts_code": "IF.CFX"},
        ])
        client._api = make_mock_api(fut_mapping=raw)
        with patch("time.sleep"):
            df = client.get_futures_mapping()
        assert df["trade_date"].tolist() == ["20240101", "20240103"]

    def test_empty_returns_empty_df_with_columns(self):
        client = make_client()
        client._api = make_mock_api(fut_mapping=pd.DataFrame())
        with patch("time.sleep"):
            df = client.get_futures_mapping()
        assert df.empty
        assert "mapping_ts_code" in df.columns


# ======================================================================
# get_options_daily
# ======================================================================

class TestGetOptionsDaily:

    def _make_raw(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "ts_code": "IO2406-C-3800.CFX",
            "trade_date": "20240102",
            "exchange": "CFFEX",
            "opt_code": "IO",
            "exercise_price": 3800.0,
            "call_put": "C",
            "expire_date": "20240621",
            "close": 250.0, "settle": 245.0,
            "vol": 500.0, "oi": 1200.0,
            "pre_close": 240.0, "pre_settle": 242.0,
        }])

    def test_opt_code_renamed_to_underlying_code(self):
        client = make_client()
        client._api = make_mock_api(opt_daily=self._make_raw())
        with patch("time.sleep"):
            df = client.get_options_daily(exchange="CFFEX", trade_date="20240102")
        assert "underlying_code" in df.columns
        assert "opt_code" not in df.columns

    def test_vol_renamed_to_volume(self):
        client = make_client()
        client._api = make_mock_api(opt_daily=self._make_raw())
        with patch("time.sleep"):
            df = client.get_options_daily(exchange="CFFEX", trade_date="20240102")
        assert "volume" in df.columns

    def test_returns_required_columns(self):
        client = make_client()
        client._api = make_mock_api(opt_daily=self._make_raw())
        with patch("time.sleep"):
            df = client.get_options_daily(exchange="CFFEX", trade_date="20240102")
        for col in ["ts_code", "trade_date", "exchange", "underlying_code",
                    "exercise_price", "call_put", "expire_date",
                    "close", "settle", "volume", "oi", "pre_close", "pre_settle"]:
            assert col in df.columns, f"缺少列: {col}"

    def test_api_called_with_exchange_and_trade_date(self):
        client = make_client()
        mock_api = MagicMock()
        mock_api.opt_daily.return_value = pd.DataFrame()
        client._api = mock_api
        with patch("time.sleep"):
            client.get_options_daily(exchange="SSE", trade_date="20240102")
        mock_api.opt_daily.assert_called_once_with(exchange="SSE", trade_date="20240102")

    def test_empty_returns_empty_df_with_columns(self):
        client = make_client()
        client._api = make_mock_api(opt_daily=pd.DataFrame())
        with patch("time.sleep"):
            df = client.get_options_daily(exchange="CFFEX", trade_date="20240102")
        assert df.empty
        assert "underlying_code" in df.columns


# ======================================================================
# get_options_contracts
# ======================================================================

class TestGetOptionsContracts:

    def _make_raw(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "ts_code": "IO2406-C-3800.CFX",
            "exchange": "CFFEX",
            "opt_code": "IO",
            "exercise_price": 3800.0,
            "call_put": "C",
            "maturity_date": "20240621",
            "list_date": "20230921",
            "delist_date": "20240621",
            "contract_unit": 100,
            "exercise_type": "E",
        }])

    def test_opt_code_renamed_to_underlying_code(self):
        client = make_client()
        client._api = make_mock_api(opt_basic=self._make_raw())
        with patch("time.sleep"):
            df = client.get_options_contracts(exchange="CFFEX")
        assert "underlying_code" in df.columns
        assert "opt_code" not in df.columns

    def test_maturity_date_renamed_to_expire_date(self):
        client = make_client()
        client._api = make_mock_api(opt_basic=self._make_raw())
        with patch("time.sleep"):
            df = client.get_options_contracts(exchange="CFFEX")
        assert "expire_date" in df.columns
        assert "maturity_date" not in df.columns

    def test_returns_required_columns(self):
        client = make_client()
        client._api = make_mock_api(opt_basic=self._make_raw())
        with patch("time.sleep"):
            df = client.get_options_contracts()
        for col in ["ts_code", "exchange", "underlying_code", "exercise_price",
                    "call_put", "expire_date", "list_date", "delist_date",
                    "contract_unit", "exercise_type"]:
            assert col in df.columns, f"缺少列: {col}"

    def test_underlying_kwarg_passed_to_api(self):
        client = make_client()
        mock_api = MagicMock()
        mock_api.opt_basic.return_value = pd.DataFrame()
        client._api = mock_api
        with patch("time.sleep"):
            client.get_options_contracts(exchange="CFFEX", underlying="IO")
        mock_api.opt_basic.assert_called_once_with(exchange="CFFEX", underlying="IO")

    def test_empty_returns_empty_df_with_columns(self):
        client = make_client()
        client._api = make_mock_api(opt_basic=pd.DataFrame())
        with patch("time.sleep"):
            df = client.get_options_contracts()
        assert df.empty
        assert "expire_date" in df.columns


# ======================================================================
# get_commodity_daily
# ======================================================================

class TestGetCommodityDaily:

    def test_delegates_to_get_futures_daily(self):
        client = make_client()
        raw = pd.DataFrame([{
            "ts_code": "RBL4.SHF", "trade_date": "20240102",
            "open": 3800.0, "high": 3850.0, "low": 3790.0,
            "close": 3820.0, "vol": 2000.0, "oi": 50000.0, "settle": 3815.0,
        }])
        client._api = make_mock_api(fut_daily=raw)
        with patch("time.sleep"):
            df = client.get_commodity_daily("RBL4.SHF", "20240101", "20240131")
        assert "volume" in df.columns
        assert len(df) == 1


# ======================================================================
# get_trade_calendar
# ======================================================================

class TestGetTradeCalendar:

    def _make_raw(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"exchange": "SSE", "cal_date": "20240102", "is_open": 1, "pretrade_date": "20231229"},
            {"exchange": "SSE", "cal_date": "20240103", "is_open": 1, "pretrade_date": "20240102"},
        ])

    def test_cal_date_renamed_to_trade_date(self):
        client = make_client()
        client._api = make_mock_api(trade_cal=self._make_raw())
        with patch("time.sleep"):
            df = client.get_trade_calendar()
        assert "trade_date" in df.columns
        assert "cal_date" not in df.columns

    def test_returns_required_columns(self):
        client = make_client()
        client._api = make_mock_api(trade_cal=self._make_raw())
        with patch("time.sleep"):
            df = client.get_trade_calendar()
        for col in ["exchange", "trade_date", "is_open", "pretrade_date"]:
            assert col in df.columns

    def test_sorted_by_trade_date(self):
        client = make_client()
        raw = pd.DataFrame([
            {"exchange": "SSE", "cal_date": "20240103", "is_open": 1, "pretrade_date": "20240102"},
            {"exchange": "SSE", "cal_date": "20240101", "is_open": 0, "pretrade_date": "20231229"},
        ])
        client._api = make_mock_api(trade_cal=raw)
        with patch("time.sleep"):
            df = client.get_trade_calendar()
        assert df["trade_date"].tolist() == ["20240101", "20240103"]

    def test_end_date_defaults_to_today(self):
        client = make_client()
        mock_api = MagicMock()
        mock_api.trade_cal.return_value = pd.DataFrame()
        client._api = mock_api
        with patch("time.sleep"):
            client.get_trade_calendar(exchange="CFFEX", start_date="20240101")
        _, kwargs = mock_api.trade_cal.call_args
        assert kwargs["end_date"] is not None
        assert len(kwargs["end_date"]) == 8  # YYYYMMDD 格式

    def test_end_date_explicit(self):
        client = make_client()
        mock_api = MagicMock()
        mock_api.trade_cal.return_value = pd.DataFrame()
        client._api = mock_api
        with patch("time.sleep"):
            client.get_trade_calendar(exchange="CFFEX", end_date="20241231")
        _, kwargs = mock_api.trade_cal.call_args
        assert kwargs["end_date"] == "20241231"

    def test_empty_returns_empty_df_with_columns(self):
        client = make_client()
        client._api = make_mock_api(trade_cal=pd.DataFrame())
        with patch("time.sleep"):
            df = client.get_trade_calendar()
        assert df.empty
        assert "trade_date" in df.columns


# ======================================================================
# get_index_daily
# ======================================================================

class TestGetIndexDaily:

    def _make_raw(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "ts_code": "000300.SH", "trade_date": "20240102",
            "open": 3400.0, "high": 3450.0, "low": 3390.0,
            "close": 3420.0, "vol": 150000.0,
        }])

    def test_vol_renamed_to_volume(self):
        client = make_client()
        client._api = make_mock_api(index_daily=self._make_raw())
        with patch("time.sleep"):
            df = client.get_index_daily("000300.SH", "20240101", "20240131")
        assert "volume" in df.columns
        assert "vol" not in df.columns

    def test_returns_required_columns(self):
        client = make_client()
        client._api = make_mock_api(index_daily=self._make_raw())
        with patch("time.sleep"):
            df = client.get_index_daily("000300.SH", "20240101", "20240131")
        for col in ["ts_code", "trade_date", "open", "high", "low", "close", "volume"]:
            assert col in df.columns

    def test_sorted_by_trade_date(self):
        client = make_client()
        raw = pd.DataFrame([
            {"ts_code": "000300.SH", "trade_date": "20240103",
             "open": 1, "high": 1, "low": 1, "close": 1, "vol": 1},
            {"ts_code": "000300.SH", "trade_date": "20240101",
             "open": 1, "high": 1, "low": 1, "close": 1, "vol": 1},
        ])
        client._api = make_mock_api(index_daily=raw)
        with patch("time.sleep"):
            df = client.get_index_daily("000300.SH", "20240101", "20240131")
        assert df["trade_date"].tolist() == ["20240101", "20240103"]

    def test_empty_returns_empty_df_with_columns(self):
        client = make_client()
        client._api = make_mock_api(index_daily=pd.DataFrame())
        with patch("time.sleep"):
            df = client.get_index_daily("000300.SH", "20240101", "20240131")
        assert df.empty
        assert "volume" in df.columns
