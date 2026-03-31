"""
test_unified_api.py
-------------------
tests/test_data/ 子集：聚焦规格要求的3项检查。
完整测试见 tests/test_unified_api.py。

覆盖：
- test_symbol_normalization：各种格式都能正确识别
- test_data_source_fallback：本地无数据时从 Tushare 下载
- test_data_deduplication：两个源拼接时无重复
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data.unified_api import UnifiedDataAPI
from data.storage.db_manager import DBManager


# ======================================================================
# Helpers
# ======================================================================

def make_api(db: DBManager | None = None) -> UnifiedDataAPI:
    """
    构造使用内存 DB 的 UnifiedDataAPI，无需真实 token。
    注入 mock config + 替换内部 db 引用。
    """
    mock_config = MagicMock()
    mock_config.db_path = ":memory:"
    mock_config.tushare_token = "FAKE_TOKEN"
    mock_config.tq_auth_account = ""
    mock_config.tq_auth_password = ""
    api = UnifiedDataAPI(config=mock_config)
    if db is not None:
        api.db = db
    return api


FUTURES_ROW = {
    "ts_code": "IF2406.CFX", "trade_date": "20240102",
    "open": 4010.0, "high": 4060.0, "low": 4000.0,
    "close": 4040.0, "volume": 5000.0, "oi": 20000.0,
    "settle": 4035.0, "pre_close": 4000.0, "pre_settle": 3990.0,
}


# ======================================================================
# test_symbol_normalization
# ======================================================================

class TestSymbolNormalization:
    """各种格式的合约代码都能被正确识别"""

    @pytest.fixture
    def api(self):
        return make_api()

    def test_ts_format_passthrough(self, api):
        assert api._normalize_symbol("IF2406.CFX") == "IF2406.CFX"

    def test_tq_format_recognized(self, api):
        assert api._normalize_symbol("CFFEX.IF2406") == "IF2406.CFX"

    def test_bare_contract_recognized(self, api):
        assert api._normalize_symbol("IF2406") == "IF2406.CFX"

    def test_option_tq_format(self, api):
        result = api._normalize_symbol("CFFEX.IO2406-C-3800")
        assert result == "IO2406-C-3800.CFX"

    def test_shfe_tq_format(self, api):
        result = api._normalize_symbol("SHFE.RBL4")
        assert result == "RBL4.SHF"

    def test_bare_product_returns_with_exchange(self, api):
        """品种代码应返回带交易所后缀的格式"""
        result = api._normalize_symbol("IF")
        assert result.endswith(".CFX")

    @pytest.mark.parametrize("symbol,expected", [
        ("IF2406.CFX",      "IF2406.CFX"),
        ("CFFEX.IF2406",    "IF2406.CFX"),
        ("IF2406",          "IF2406.CFX"),
        ("CFFEX.IM2409",    "IM2409.CFX"),
    ])
    def test_normalization_parametrized(self, api, symbol, expected):
        assert api._normalize_symbol(symbol) == expected


# ======================================================================
# test_data_source_fallback
# ======================================================================

class TestDataSourceFallback:
    """本地无数据时应从 Tushare 下载"""

    def test_falls_back_to_tushare_when_local_empty(self):
        db = DBManager(":memory:")
        api = make_api(db)

        tushare_df = pd.DataFrame([FUTURES_ROW])
        api.tushare.get_futures_daily = MagicMock(return_value=tushare_df)

        df = api.get_futures_daily("IF2406.CFX", "20240102", "20240102",
                                   auto_download=True)
        assert not df.empty
        api.tushare.get_futures_daily.assert_called_once()

    def test_no_fallback_when_auto_download_false(self):
        db = DBManager(":memory:")
        api = make_api(db)
        api.tushare.get_futures_daily = MagicMock(return_value=pd.DataFrame())

        df = api.get_futures_daily("IF2406.CFX", "20240102", "20240102",
                                   auto_download=False)
        assert df.empty
        api.tushare.get_futures_daily.assert_not_called()

    def test_uses_local_data_first(self):
        db = DBManager(":memory:")
        db.upsert_dataframe("futures_daily", pd.DataFrame([FUTURES_ROW]))
        api = make_api(db)
        api.tushare.get_futures_daily = MagicMock()

        df = api.get_futures_daily("IF2406.CFX", "20240102", "20240102",
                                   auto_download=False)
        assert not df.empty
        api.tushare.get_futures_daily.assert_not_called()


# ======================================================================
# test_data_deduplication
# ======================================================================

class TestDataDeduplication:
    """写入 DB 不产生重复行"""

    def test_no_duplicate_after_double_upsert(self):
        db = DBManager(":memory:")
        api = make_api(db)
        api.tushare.get_futures_daily = MagicMock(
            return_value=pd.DataFrame([FUTURES_ROW])
        )

        # 下载两次相同区间
        api.get_futures_daily("IF2406.CFX", "20240102", "20240102",
                              auto_download=True)
        api.get_futures_daily("IF2406.CFX", "20240102", "20240102",
                              auto_download=True)

        count = db.query_scalar(
            "SELECT COUNT(*) FROM futures_daily WHERE ts_code=? AND trade_date=?",
            ("IF2406.CFX", "20240102"),
        )
        assert count == 1  # upsert 语义，不重复

    def test_incremental_update_no_overlap(self):
        """增量更新：已有行保留，新行追加"""
        db = DBManager(":memory:")
        db.upsert_dataframe("futures_daily", pd.DataFrame([FUTURES_ROW]))

        api = make_api(db)
        new_row = {**FUTURES_ROW, "trade_date": "20240103", "close": 4050.0}
        api.tushare.get_futures_daily = MagicMock(
            return_value=pd.DataFrame([new_row])
        )

        api.get_futures_daily("IF2406.CFX", "20240103", "20240103",
                              auto_download=True)

        count = db.query_scalar("SELECT COUNT(*) FROM futures_daily")
        assert count == 2  # 原始 1 行 + 新增 1 行
