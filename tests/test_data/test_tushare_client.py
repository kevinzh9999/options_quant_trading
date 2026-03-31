"""
test_tushare_client.py
----------------------
tests/test_data/ 子集：聚焦字段映射、频率控制、重试逻辑。
完整测试见 tests/test_tushare_client.py。

覆盖：
- 字段映射（vol -> volume）
- 频率控制（sleep 被调用）
- 重试逻辑（API 失败后自动重试）
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data.sources.tushare_client import TushareClient


def make_client(sleep: float = 0.0, max_retry: int = 3) -> TushareClient:
    return TushareClient(token="FAKE_TOKEN", max_retry=max_retry, sleep_interval=sleep)


def make_mock_api(**method_returns) -> MagicMock:
    """返回注入 client._api 的 mock pro_api 对象"""
    api = MagicMock()
    for method, retval in method_returns.items():
        getattr(api, method).return_value = retval
    return api


def raw_futures_daily() -> pd.DataFrame:
    """Tushare fut_daily 返回的原始列名（含 vol 而非 volume）"""
    return pd.DataFrame({
        "ts_code":    ["IF2406.CFX"],
        "trade_date": ["20240102"],
        "open":       [4010.0],
        "high":       [4060.0],
        "low":        [4000.0],
        "close":      [4040.0],
        "vol":        [5000.0],   # 原始列名
        "oi":         [20000.0],
        "settle":     [4035.0],
        "pre_close":  [4000.0],
        "pre_settle": [3990.0],
    })


# ======================================================================
# 字段映射
# ======================================================================

class TestFieldMapping:

    def test_vol_renamed_to_volume(self):
        """Tushare 的 vol 字段应映射为 volume"""
        client = make_client()
        client._api = make_mock_api(fut_daily=raw_futures_daily())

        with patch("time.sleep"):
            df = client.get_futures_daily("IF2406.CFX", "20240102", "20240102")

        assert "volume" in df.columns
        assert "vol" not in df.columns

    def test_required_columns_present(self):
        """返回 DataFrame 应包含所有规范列名"""
        client = make_client()
        client._api = make_mock_api(fut_daily=raw_futures_daily())

        with patch("time.sleep"):
            df = client.get_futures_daily("IF2406.CFX", "20240102", "20240102")

        for col in ("ts_code", "trade_date", "open", "high", "low",
                    "close", "volume", "oi", "settle"):
            assert col in df.columns, f"缺少列: {col}"

    def test_empty_response_returns_empty_df_with_columns(self):
        """空响应应返回空 DataFrame（含正确列名，不报错）"""
        client = make_client()
        client._api = make_mock_api(fut_daily=pd.DataFrame())

        with patch("time.sleep"):
            df = client.get_futures_daily("IF2406.CFX", "20240102", "20240102")

        assert df.empty
        assert "ts_code" in df.columns


# ======================================================================
# 频率控制
# ======================================================================

class TestSleepControl:

    def test_sleep_called_before_request(self):
        """每次 _call_api 都应调用 time.sleep"""
        client = make_client(sleep=0.5)
        client._api = make_mock_api(fut_daily=raw_futures_daily())

        with patch("time.sleep") as mock_sleep:
            client.get_futures_daily("IF2406.CFX", "20240102", "20240102")

        mock_sleep.assert_any_call(0.5)

    def test_sleep_interval_respected(self):
        """sleep_interval 为 0.1 时，sleep(0.1) 应被调用"""
        client = make_client(sleep=0.1)
        client._api = make_mock_api(fut_daily=raw_futures_daily())

        with patch("time.sleep") as mock_sleep:
            client.get_futures_daily("IF2406.CFX", "20240102", "20240102")

        mock_sleep.assert_any_call(0.1)


# ======================================================================
# 重试逻辑
# ======================================================================

class TestRetryLogic:

    def test_retries_on_failure_then_succeeds(self):
        """API 第一次失败，第二次成功"""
        client = make_client(sleep=0.0, max_retry=3)
        mock_api = MagicMock()
        mock_api.fut_daily.side_effect = [
            Exception("网络超时"),
            raw_futures_daily(),
        ]
        client._api = mock_api

        with patch("time.sleep"):
            df = client.get_futures_daily("IF2406.CFX", "20240102", "20240102")

        assert not df.empty
        assert mock_api.fut_daily.call_count == 2

    def test_raises_after_max_retries(self):
        """超过最大重试次数应抛出异常"""
        client = make_client(sleep=0.0, max_retry=2)
        mock_api = MagicMock()
        mock_api.fut_daily.side_effect = Exception("服务不可用")
        client._api = mock_api

        with patch("time.sleep"), pytest.raises(Exception, match="服务不可用"):
            client.get_futures_daily("IF2406.CFX", "20240102", "20240102")

        assert mock_api.fut_daily.call_count == 2  # max_retry 次

    def test_no_extra_call_on_success(self):
        """首次成功时只调用一次"""
        client = make_client(sleep=0.0, max_retry=3)
        client._api = make_mock_api(fut_daily=raw_futures_daily())

        with patch("time.sleep"):
            client.get_futures_daily("IF2406.CFX", "20240102", "20240102")

        assert client._api.fut_daily.call_count == 1

    def test_exponential_backoff_on_retry(self):
        """重试时的指数退避 sleep 应被调用"""
        client = make_client(sleep=0.0, max_retry=3)
        mock_api = MagicMock()
        mock_api.fut_daily.side_effect = [
            Exception("失败1"),
            raw_futures_daily(),
        ]
        client._api = mock_api

        sleep_calls = []
        with patch("time.sleep", side_effect=lambda s: sleep_calls.append(s)):
            client.get_futures_daily("IF2406.CFX", "20240102", "20240102")

        # 第一次失败后的退避 sleep(2**1)=2
        backoff = [s for s in sleep_calls if s >= 2]
        assert len(backoff) >= 1
