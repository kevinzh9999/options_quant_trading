"""
test_quality_check.py
---------------------
tests/test_data/ 子集：聚焦规格要求的3项检查。
完整测试见 tests/test_quality_check.py。

覆盖：
- test_missing_dates_detection：构造缺失交易日，验证能检出
- test_price_anomaly_detection：构造异常涨跌幅，验证能检出
- test_ohlc_violation：构造 high < low，验证能检出
"""

from __future__ import annotations

import pandas as pd
import pytest

from data.quality_check import DataQualityChecker


def make_checker(**kwargs):
    return DataQualityChecker(**kwargs)


def make_futures_df(n: int = 5, bad_rows: dict | None = None) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({
            "ts_code":    "IF2406.CFX",
            "trade_date": f"2024010{i + 1}",
            "open":   4000.0 + i,
            "high":   4010.0 + i,
            "low":    3990.0 + i,
            "close":  4005.0 + i,
            "volume": 5000.0,
        })
    df = pd.DataFrame(rows)
    if bad_rows:
        for idx, updates in bad_rows.items():
            for col, val in updates.items():
                df.at[idx, col] = val
    return df


# ======================================================================
# test_missing_dates_detection
# ======================================================================

class TestMissingDatesDetection:

    def test_detects_single_missing_date(self):
        """提供5个交易日中缺少1天，应检出1个缺失日期"""
        trade_dates = [f"2024010{i}" for i in range(1, 6)]  # 5天
        # 数据中只有4天
        df = pd.DataFrame({"trade_date": ["20240101", "20240102", "20240103", "20240105"]})
        checker = make_checker(trade_dates=trade_dates)
        missing = checker.check_missing_dates(df)
        assert missing == ["20240104"]

    def test_detects_multiple_missing_dates(self):
        """连续缺失多个交易日"""
        trade_dates = [f"2024010{i}" for i in range(1, 8)]  # 7天
        df = pd.DataFrame({"trade_date": ["20240101", "20240107"]})
        checker = make_checker(trade_dates=trade_dates)
        missing = checker.check_missing_dates(df)
        assert len(missing) == 5
        assert "20240102" in missing

    def test_no_missing_when_complete(self):
        """数据完整时返回空列表"""
        dates = ["20240101", "20240102", "20240103"]
        df = pd.DataFrame({"trade_date": dates})
        checker = make_checker(trade_dates=dates)
        assert checker.check_missing_dates(df) == []

    def test_no_trade_dates_returns_empty(self):
        """未提供交易日历时返回空列表"""
        df = pd.DataFrame({"trade_date": ["20240101"]})
        assert make_checker().check_missing_dates(df) == []

    def test_missing_dates_in_check_futures_daily(self):
        """check_futures_daily 结果中的 missing_dates 字段"""
        trade_dates = [f"2024010{i}" for i in range(1, 6)]
        df = make_futures_df(n=2)  # 只有2天
        result = make_checker(trade_dates=trade_dates).check_futures_daily(df, "IF2406.CFX")
        assert len(result["missing_dates"]) == 3


# ======================================================================
# test_price_anomaly_detection
# ======================================================================

class TestPriceAnomalyDetection:

    def test_detects_large_price_jump(self):
        """涨跌幅超过15%应被标记为异常"""
        # row 0: 4000 → row 1: 6000 (50% jump)
        df = make_futures_df(bad_rows={1: {"close": 6000.0}})
        result = make_checker().check_futures_daily(df, "IF2406.CFX")
        assert len(result["price_anomalies"]) > 0

    def test_anomaly_contains_change_pct(self):
        """异常记录应包含 change_pct 字段"""
        df = make_futures_df(bad_rows={1: {"close": 6000.0}})
        result = make_checker().check_futures_daily(df, "IF2406.CFX")
        anomaly = result["price_anomalies"][0]
        assert "change_pct" in anomaly
        assert abs(anomaly["change_pct"]) > 0.15

    def test_normal_price_change_not_anomaly(self):
        """正常涨跌幅（< 15%）不应产生异常"""
        # 相邻行价格变化约 0.025%，远低于阈值
        df = make_futures_df(n=5)
        result = make_checker().check_futures_daily(df, "IF2406.CFX")
        assert result["price_anomalies"] == []

    def test_anomaly_severity_is_warning(self):
        """价格异常的 severity 为 warning（不影响 is_clean）"""
        df = make_futures_df(bad_rows={1: {"close": 6000.0}})
        result = make_checker().check_futures_daily(df, "IF2406.CFX")
        for anomaly in result["price_anomalies"]:
            assert anomaly["severity"] == "warning"

    def test_check_price_outliers_detects_outlier(self):
        """check_price_outliers 工具方法同样检出异常"""
        df = pd.DataFrame({"close": [100, 101, 150, 151]})
        outliers = make_checker().check_price_outliers(df, max_daily_change=0.2)
        assert 2 in outliers


# ======================================================================
# test_ohlc_violation
# ======================================================================

class TestOhlcViolation:

    def test_high_lt_low_detected(self):
        """high < low 应被检出为 OHLC 违规"""
        df = make_futures_df(bad_rows={0: {"high": 3980.0, "low": 3995.0}})
        result = make_checker().check_futures_daily(df, "IF2406.CFX")
        issues = [v["issue"] for v in result["ohlc_violations"]]
        assert "high_lt_low" in issues

    def test_high_lt_close_detected(self):
        """high < close 应被检出"""
        df = make_futures_df(bad_rows={0: {"high": 3990.0, "close": 4005.0}})
        result = make_checker().check_futures_daily(df, "IF2406.CFX")
        issues = [v["issue"] for v in result["ohlc_violations"]]
        assert "high_lt_close" in issues

    def test_non_positive_close_detected(self):
        """close <= 0 应被检出"""
        df = make_futures_df(bad_rows={0: {"close": -1.0}})
        result = make_checker().check_futures_daily(df, "IF2406.CFX")
        issues = [v["issue"] for v in result["ohlc_violations"]]
        assert "non_positive_close" in issues

    def test_ohlc_violation_marks_is_clean_false(self):
        """存在 OHLC 违规时 is_clean 应为 False"""
        df = make_futures_df(bad_rows={0: {"high": 3980.0, "low": 3995.0}})
        result = make_checker().check_futures_daily(df, "IF2406.CFX")
        assert result["is_clean"] is False

    def test_clean_data_no_violations(self):
        """合法的 OHLC 数据不应有违规"""
        result = make_checker().check_futures_daily(make_futures_df(), "IF2406.CFX")
        assert result["ohlc_violations"] == []
        assert result["is_clean"] is True

    def test_violation_record_contains_index(self):
        """违规记录应包含行索引"""
        df = make_futures_df(bad_rows={2: {"high": 3980.0, "low": 3995.0}})
        result = make_checker().check_futures_daily(df, "IF2406.CFX")
        indices = [v["index"] for v in result["ohlc_violations"]]
        assert 2 in indices
