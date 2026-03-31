"""
test_account_manager.py
-----------------------
测试账户持仓管理模块（data/sources/account_manager.py）。

全部使用 Mock，不建立真实天勤连接。

覆盖：
- _is_option 判断
- parse_option_symbol：认购/认沽 / 非法代码
- get_account_summary：字段完整 / margin_ratio 计算 / NaN 处理
- get_all_positions：多空方向展开 / 空持仓
- get_option_positions：过滤 + 额外字段
- get_futures_positions：过滤
- get_margin_detail：字段提取
- is_account_ready：正常 / NaN / 异常
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest

from data.sources.account_manager import AccountManager, _is_option


# ======================================================================
# 工具函数
# ======================================================================

class TestIsOption:

    def test_call_option_detected(self):
        assert _is_option("CFFEX.IO2406-C-3800") is True

    def test_put_option_detected(self):
        assert _is_option("CFFEX.MO2406-P-5400") is True

    def test_futures_not_option(self):
        assert _is_option("CFFEX.IF2406") is False
        assert _is_option("CFFEX.IM2406") is False

    def test_empty_string(self):
        assert _is_option("") is False


# ======================================================================
# parse_option_symbol
# ======================================================================

class TestParseOptionSymbol:

    def test_parse_call_option(self):
        result = AccountManager.parse_option_symbol("CFFEX.IO2406-C-3800")
        assert result["exchange"]     == "CFFEX"
        assert result["product"]      == "IO"
        assert result["expire_month"] == "2406"
        assert result["call_put"]     == "CALL"
        assert result["strike_price"] == 3800.0

    def test_parse_put_option(self):
        result = AccountManager.parse_option_symbol("CFFEX.MO2406-P-5400")
        assert result["call_put"]     == "PUT"
        assert result["strike_price"] == 5400.0
        assert result["product"]      == "MO"

    def test_parse_decimal_strike(self):
        result = AccountManager.parse_option_symbol("CFFEX.IO2406-C-3850")
        assert result["strike_price"] == 3850.0

    def test_invalid_symbol_raises(self):
        with pytest.raises(ValueError, match="无法解析"):
            AccountManager.parse_option_symbol("CFFEX.IF2406")

    def test_invalid_no_exchange(self):
        with pytest.raises(ValueError):
            AccountManager.parse_option_symbol("IO2406-C-3800")


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def mock_tq_api():
    api = MagicMock()

    mock_account = MagicMock()
    mock_account.balance         = 1_000_000.0
    mock_account.available       = 600_000.0
    mock_account.margin          = 380_000.0
    mock_account.float_profit    = 5_000.0
    mock_account.position_profit = 5_000.0
    mock_account.close_profit    = 2_000.0
    mock_account.commission      = 300.0
    mock_account.risk_ratio      = 0.38
    api.get_account.return_value = mock_account

    mock_pos_if = MagicMock()
    mock_pos_if.volume_long        = 1
    mock_pos_if.volume_short       = 0
    mock_pos_if.volume_long_today  = 1
    mock_pos_if.volume_short_today = 0
    mock_pos_if.open_price_long    = 3780.0
    mock_pos_if.open_price_short   = 0.0
    mock_pos_if.last_price         = 3800.0
    mock_pos_if.float_profit_long  = 4000.0
    mock_pos_if.float_profit_short = 0.0
    mock_pos_if.margin_long        = 300_000.0
    mock_pos_if.margin_short       = 0.0

    mock_pos_io = MagicMock()
    mock_pos_io.volume_long        = 0
    mock_pos_io.volume_short       = 2
    mock_pos_io.volume_long_today  = 0
    mock_pos_io.volume_short_today = 2
    mock_pos_io.open_price_long    = 0.0
    mock_pos_io.open_price_short   = 55.0
    mock_pos_io.last_price         = 48.0
    mock_pos_io.float_profit_long  = 0.0
    mock_pos_io.float_profit_short = 1_400.0
    mock_pos_io.margin_long        = 0.0
    mock_pos_io.margin_short       = 80_000.0

    api.get_position.return_value = {
        "CFFEX.IF2406":        mock_pos_if,
        "CFFEX.IO2406-C-3800": mock_pos_io,
    }
    return api


@pytest.fixture
def mgr(mock_tq_api):
    return AccountManager(tq_api=mock_tq_api, broker="宏源期货", account_id="TEST123456")


# ======================================================================
# get_account_summary
# ======================================================================

class TestGetAccountSummary:

    def test_returns_required_keys(self, mgr):
        summary = mgr.get_account_summary()
        for key in ("balance", "available", "margin", "margin_ratio",
                    "float_profit", "position_profit", "close_profit",
                    "commission", "risk_ratio"):
            assert key in summary, f"缺少键: {key}"

    def test_balance_correct(self, mgr):
        assert mgr.get_account_summary()["balance"] == 1_000_000.0

    def test_margin_ratio_calculation(self, mgr):
        summary = mgr.get_account_summary()
        # margin / balance = 380000 / 1000000 = 0.38
        assert abs(summary["margin_ratio"] - 0.38) < 1e-4

    def test_nan_balance_becomes_zero(self):
        api = MagicMock()
        account = MagicMock()
        account.balance         = float("nan")
        account.available       = float("nan")
        account.margin          = float("nan")
        account.float_profit    = float("nan")
        account.position_profit = 0.0
        account.close_profit    = 0.0
        account.commission      = 0.0
        api.get_account.return_value = account
        m = AccountManager(tq_api=api)
        summary = m.get_account_summary()
        assert summary["balance"]      == 0.0
        assert summary["margin"]       == 0.0
        assert summary["margin_ratio"] == 0.0

    def test_api_exception_raises_runtime_error(self):
        api = MagicMock()
        api.get_account.side_effect = RuntimeError("连接断开")
        m = AccountManager(tq_api=api)
        with pytest.raises(RuntimeError, match="账户数据不可用"):
            m.get_account_summary()


# ======================================================================
# get_all_positions
# ======================================================================

class TestGetAllPositions:

    def test_returns_two_positions(self, mgr):
        positions = mgr.get_all_positions()
        assert len(positions) == 2

    def test_required_fields_present(self, mgr):
        for pos in mgr.get_all_positions():
            for field in ("symbol", "direction", "volume", "volume_today",
                          "open_price_avg", "last_price", "float_profit",
                          "margin", "instrument_type"):
                assert field in pos

    def test_if_is_future_long(self, mgr):
        futures = [p for p in mgr.get_all_positions() if "IF2406" in p["symbol"]]
        assert len(futures) == 1
        assert futures[0]["instrument_type"] == "FUTURE"
        assert futures[0]["direction"]        == "LONG"
        assert futures[0]["volume"]           == 1

    def test_io_is_option_short(self, mgr):
        options = [p for p in mgr.get_all_positions() if "IO2406" in p["symbol"]]
        assert len(options) == 1
        assert options[0]["instrument_type"] == "OPTION"
        assert options[0]["direction"]        == "SHORT"
        assert options[0]["volume"]           == 2

    def test_empty_positions(self):
        api = MagicMock()
        api.get_position.return_value = {}
        assert AccountManager(tq_api=api).get_all_positions() == []

    def test_api_exception_returns_empty(self):
        api = MagicMock()
        api.get_position.side_effect = ConnectionError("连接超时")
        assert AccountManager(tq_api=api).get_all_positions() == []

    def test_zero_volume_excluded(self):
        api = MagicMock()
        pos = MagicMock()
        pos.volume_long  = 0
        pos.volume_short = 0
        api.get_position.return_value = {"CFFEX.IF2406": pos}
        assert AccountManager(tq_api=api).get_all_positions() == []


# ======================================================================
# get_option_positions
# ======================================================================

class TestGetOptionPositions:

    def test_returns_only_options(self, mgr):
        options = mgr.get_option_positions()
        assert len(options) == 1
        assert "IO2406-C-3800" in options[0]["symbol"]

    def test_no_futures_in_result(self, mgr):
        symbols = [p["symbol"] for p in mgr.get_option_positions()]
        assert not any("IF2406" in s for s in symbols)

    def test_extra_fields_present(self, mgr):
        opt = mgr.get_option_positions()[0]
        for field in ("strike_price", "call_put", "expire_date", "underlying"):
            assert field in opt

    def test_strike_price_correct(self, mgr):
        assert mgr.get_option_positions()[0]["strike_price"] == 3800.0

    def test_call_put_correct(self, mgr):
        assert mgr.get_option_positions()[0]["call_put"] == "CALL"

    def test_underlying_correct(self, mgr):
        assert mgr.get_option_positions()[0]["underlying"] == "CFFEX.IO2406"


# ======================================================================
# get_futures_positions
# ======================================================================

class TestGetFuturesPositions:

    def test_returns_only_futures(self, mgr):
        futures = mgr.get_futures_positions()
        assert len(futures) == 1
        assert "IF2406" in futures[0]["symbol"]

    def test_no_options_in_result(self, mgr):
        symbols = [p["symbol"] for p in mgr.get_futures_positions()]
        assert not any(_is_option(s) for s in symbols)


# ======================================================================
# get_margin_detail
# ======================================================================

class TestGetMarginDetail:

    def test_returns_two_entries(self, mgr):
        assert len(mgr.get_margin_detail()) == 2

    def test_required_fields(self, mgr):
        for entry in mgr.get_margin_detail():
            for f in ("symbol", "direction", "volume", "margin"):
                assert f in entry

    def test_if_margin_correct(self, mgr):
        entries = [e for e in mgr.get_margin_detail() if "IF2406" in e["symbol"]]
        assert entries[0]["margin"] == 300_000.0

    def test_total_margin_matches_account(self, mgr):
        total = sum(e["margin"] for e in mgr.get_margin_detail())
        assert abs(total - 380_000.0) < 1.0


# ======================================================================
# is_account_ready
# ======================================================================

class TestIsAccountReady:

    def test_ready_when_balance_positive(self, mgr):
        assert mgr.is_account_ready() is True

    def test_not_ready_when_balance_nan(self):
        api = MagicMock()
        account = MagicMock()
        account.balance = float("nan")
        api.get_account.return_value = account
        assert AccountManager(tq_api=api).is_account_ready() is False

    def test_not_ready_when_balance_zero(self):
        api = MagicMock()
        account = MagicMock()
        account.balance = 0.0
        api.get_account.return_value = account
        assert AccountManager(tq_api=api).is_account_ready() is False

    def test_not_ready_on_exception(self):
        api = MagicMock()
        api.get_account.side_effect = RuntimeError("断线")
        assert AccountManager(tq_api=api).is_account_ready() is False
