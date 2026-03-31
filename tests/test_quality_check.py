"""
test_quality_check.py
---------------------
测试 data/quality_check.py

覆盖：
- QualityReport.is_clean / summary
- check_null_values / check_missing_dates / check_price_outliers
- check_futures_daily（dict 接口）：正常数据 / OHLC 错误 / 成交量异常 / 价格异常 / 空值 / 缺失日期
- check_futures_min（dict 接口）：正常数据 / OHLC 错误 / daily_bar_counts / time_gaps
- check_options_daily（dict 接口）：call_put_balance / 负价格
- check_data_alignment：重叠行 / 价格差异 / 成交量差异
- run_full_check (旧接口 str → QualityReport)
- run_full_check (新接口 db_manager → Dict)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from data.quality_check import DataQualityChecker, QualityReport


# ======================================================================
# Helpers
# ======================================================================

def make_checker(**kwargs) -> DataQualityChecker:
    return DataQualityChecker(**kwargs)


def make_futures_df(n: int = 5, bad_rows: dict | None = None) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({
            "ts_code": "IF2406.CFX",
            "trade_date": f"2024010{i + 1}",
            "open":   4000.0 + i,
            "high":   4010.0 + i,
            "low":    3990.0 + i,
            "close":  4005.0 + i,
            "volume": 5000.0,
            "settle": 4003.0 + i,
        })
    df = pd.DataFrame(rows)
    if bad_rows:
        for idx, updates in bad_rows.items():
            for col, val in updates.items():
                df.at[idx, col] = val
    return df


def make_options_df(n: int = 4, with_call_put: bool = False) -> pd.DataFrame:
    rows = []
    cp_cycle = ["C", "P", "C", "P"]
    for i in range(n):
        row = {
            "ts_code":    f"IO2406-C-{3800 + i * 100}.CFX",
            "trade_date": "20240102",
            "open":  250.0 + i,
            "high":  260.0 + i,
            "low":   240.0 + i,
            "close": 255.0 + i,
            "volume": 500.0,
        }
        if with_call_put:
            row["call_put"] = cp_cycle[i % 4]
        rows.append(row)
    return pd.DataFrame(rows)


def make_min_df(n: int = 6) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({
            "ts_code":  "IF2406.CFX",
            "datetime": f"2024-01-02 09:{35 + i * 5:02d}:00",
            "open":     4010.0 + i,
            "high":     4015.0 + i,
            "low":      4005.0 + i,
            "close":    4012.0 + i,
            "volume":   100.0,
        })
    return pd.DataFrame(rows)


# ======================================================================
# QualityReport（向后兼容对象）
# ======================================================================

class TestQualityReport:

    def test_is_clean_no_errors(self):
        r = QualityReport("t", 10, [], {}, [], warnings=["w"])
        assert r.is_clean is True

    def test_is_clean_with_errors(self):
        r = QualityReport("t", 10, [], {}, [], errors=["e"])
        assert r.is_clean is False

    def test_summary_contains_table_name(self):
        r = QualityReport("futures_daily", 100, [], {}, [])
        assert "futures_daily" in r.summary()

    def test_summary_shows_missing_date_count(self):
        r = QualityReport("t", 10, ["20240101", "20240102"], {}, [])
        summary = r.summary()
        assert "2" in summary


# ======================================================================
# 工具方法
# ======================================================================

class TestCheckNullValues:

    def test_no_nulls(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        assert make_checker().check_null_values(df) == {}

    def test_with_nulls(self):
        df = pd.DataFrame({"a": [1, None], "b": [None, None]})
        result = make_checker().check_null_values(df)
        assert result["a"] == 1
        assert result["b"] == 2

    def test_empty_df(self):
        assert make_checker().check_null_values(pd.DataFrame()) == {}


class TestCheckMissingDates:

    def test_no_missing(self):
        dates = ["20240101", "20240102", "20240103"]
        df = pd.DataFrame({"trade_date": dates})
        assert make_checker(trade_dates=dates).check_missing_dates(df) == []

    def test_with_missing(self):
        expected = ["20240101", "20240102", "20240103"]
        df = pd.DataFrame({"trade_date": ["20240101", "20240103"]})
        missing = make_checker(trade_dates=expected).check_missing_dates(df)
        assert missing == ["20240102"]

    def test_no_trade_dates_returns_empty(self):
        df = pd.DataFrame({"trade_date": ["20240101"]})
        assert make_checker().check_missing_dates(df) == []

    def test_custom_expected_dates(self):
        df = pd.DataFrame({"trade_date": ["20240101"]})
        missing = make_checker().check_missing_dates(
            df, expected_dates=["20240101", "20240102"]
        )
        assert missing == ["20240102"]


class TestCheckPriceOutliers:

    def test_no_outliers(self):
        df = pd.DataFrame({"close": [100, 101, 102, 103]})
        assert make_checker().check_price_outliers(df) == []

    def test_detects_outlier(self):
        df = pd.DataFrame({"close": [100, 101, 150, 151]})
        outliers = make_checker().check_price_outliers(df, max_daily_change=0.2)
        assert 2 in outliers

    def test_single_row_no_outlier(self):
        df = pd.DataFrame({"close": [100]})
        assert make_checker().check_price_outliers(df) == []

    def test_missing_price_col(self):
        df = pd.DataFrame({"volume": [100, 200]})
        assert make_checker().check_price_outliers(df) == []


# ======================================================================
# check_futures_daily — dict 接口
# ======================================================================

class TestCheckFuturesDaily:

    def test_returns_dict_with_required_keys(self):
        result = make_checker().check_futures_daily(make_futures_df(), "IF2406.CFX")
        for key in ("total_rows", "missing_dates", "price_anomalies",
                    "volume_anomalies", "ohlc_violations", "null_counts", "is_clean"):
            assert key in result, f"缺少键: {key}"

    def test_clean_data_is_clean(self):
        result = make_checker().check_futures_daily(make_futures_df(), "IF2406.CFX")
        assert result["is_clean"] is True
        assert result["ohlc_violations"] == []
        assert result["volume_anomalies"] == []

    def test_total_rows_correct(self):
        result = make_checker().check_futures_daily(make_futures_df(n=7), "IF2406.CFX")
        assert result["total_rows"] == 7

    def test_detects_high_lt_low(self):
        df = make_futures_df(bad_rows={0: {"high": 3980.0, "low": 3995.0}})
        result = make_checker().check_futures_daily(df, "IF2406.CFX")
        assert result["is_clean"] is False
        issues = [v["issue"] for v in result["ohlc_violations"]]
        assert "high_lt_low" in issues

    def test_detects_non_positive_close(self):
        df = make_futures_df(bad_rows={1: {"close": -1.0}})
        result = make_checker().check_futures_daily(df, "IF2406.CFX")
        issues = [v["issue"] for v in result["ohlc_violations"]]
        assert "non_positive_close" in issues

    def test_detects_zero_volume(self):
        df = make_futures_df(bad_rows={0: {"volume": 0.0}})
        result = make_checker().check_futures_daily(df, "IF2406.CFX")
        assert result["is_clean"] is False
        issues = [v["issue"] for v in result["volume_anomalies"]]
        assert "zero_volume" in issues

    def test_detects_volume_spike(self):
        df = make_futures_df(n=10)
        # set one row to 1000x the normal 5000
        df.at[5, "volume"] = 5_000_000.0
        result = make_checker().check_futures_daily(df, "IF2406.CFX")
        issues = [v["issue"] for v in result["volume_anomalies"]]
        assert "spike" in issues

    def test_detects_price_anomaly(self):
        # 50% jump from row 0 to row 1
        df = make_futures_df(bad_rows={1: {"close": 6100.0}})
        result = make_checker().check_futures_daily(df, "IF2406.CFX")
        assert len(result["price_anomalies"]) > 0
        assert result["price_anomalies"][0]["severity"] == "warning"

    def test_null_counts_reported(self):
        df = make_futures_df()
        df.at[2, "close"] = None
        result = make_checker().check_futures_daily(df, "IF2406.CFX")
        assert "close" in result["null_counts"]

    def test_missing_dates(self):
        trade_dates = [f"2024010{i}" for i in range(1, 7)]
        df = make_futures_df(n=3)  # only 3 of 6 days
        result = make_checker(trade_dates=trade_dates).check_futures_daily(
            df, "IF2406.CFX"
        )
        assert len(result["missing_dates"]) == 3

    def test_empty_df_returns_clean(self):
        result = make_checker().check_futures_daily(pd.DataFrame(), "IF2406.CFX")
        assert result["total_rows"] == 0
        assert result["is_clean"] is True


# ======================================================================
# check_futures_min — dict 接口
# ======================================================================

class TestCheckFuturesMin:

    def test_returns_dict_with_required_keys(self):
        result = make_checker().check_futures_min(make_min_df(), "IF2406.CFX")
        for key in ("total_rows", "missing_dates", "ohlc_violations",
                    "null_counts", "daily_bar_counts", "time_gaps", "is_clean"):
            assert key in result

    def test_clean_data_is_clean(self):
        result = make_checker().check_futures_min(make_min_df(), "IF2406.CFX")
        assert result["is_clean"] is True
        assert result["ohlc_violations"] == []

    def test_total_rows_correct(self):
        result = make_checker().check_futures_min(make_min_df(n=4), "IF2406.CFX")
        assert result["total_rows"] == 4

    def test_detects_high_lt_low(self):
        df = make_min_df()
        df.at[0, "high"] = 4000.0
        df.at[0, "low"]  = 4010.0
        result = make_checker().check_futures_min(df, "IF2406.CFX")
        assert result["is_clean"] is False
        issues = [v["issue"] for v in result["ohlc_violations"]]
        assert "high_lt_low" in issues

    def test_daily_bar_counts_populated(self):
        result = make_checker().check_futures_min(make_min_df(n=6), "IF2406.CFX")
        # all 6 bars are on 2024-01-02
        assert result["daily_bar_counts"].get("2024-01-02") == 6

    def test_time_gap_detected(self):
        df = make_min_df(n=3)
        # introduce a 30-minute gap between bar 1 and bar 2
        df.at[2, "datetime"] = "2024-01-02 10:20:00"
        result = make_checker().check_futures_min(df, "IF2406.CFX", freq_minutes=5)
        assert len(result["time_gaps"]) > 0

    def test_empty_df(self):
        result = make_checker().check_futures_min(pd.DataFrame(), "IF2406.CFX")
        assert result["total_rows"] == 0
        assert result["is_clean"] is True

    def test_missing_dates_with_trade_calendar(self):
        trade_dates = {"20240102", "20240103"}
        # Only provide data for 20240102
        result = make_checker(trade_dates=trade_dates).check_futures_min(
            make_min_df(n=3), "IF2406.CFX"
        )
        assert "20240103" in result["missing_dates"]


# ======================================================================
# check_options_daily — dict 接口
# ======================================================================

class TestCheckOptionsDaily:

    def test_returns_dict_with_required_keys(self):
        result = make_checker().check_options_daily(make_options_df())
        for key in ("total_rows", "daily_contract_counts", "call_put_balance",
                    "ohlc_violations", "null_counts", "is_clean"):
            assert key in result

    def test_clean_data_is_clean(self):
        result = make_checker().check_options_daily(make_options_df())
        assert result["is_clean"] is True

    def test_total_rows_correct(self):
        result = make_checker().check_options_daily(make_options_df(n=6))
        assert result["total_rows"] == 6

    def test_negative_price_detected(self):
        df = make_options_df()
        df.at[0, "close"] = -5.0
        result = make_checker().check_options_daily(df)
        assert result["is_clean"] is False
        issues = [v["issue"] for v in result["ohlc_violations"]]
        assert "negative_price" in issues

    def test_call_put_balance_populated(self):
        df = make_options_df(n=4, with_call_put=True)
        result = make_checker().check_options_daily(df)
        balance = result["call_put_balance"]
        assert "20240102" in balance
        assert balance["20240102"]["C"] == 2
        assert balance["20240102"]["P"] == 2

    def test_daily_contract_counts_populated(self):
        df = make_options_df(n=4)
        result = make_checker().check_options_daily(df)
        assert result["daily_contract_counts"].get("20240102") == 4

    def test_empty_df(self):
        result = make_checker().check_options_daily(pd.DataFrame())
        assert result["total_rows"] == 0
        assert result["is_clean"] is True


# ======================================================================
# check_data_alignment
# ======================================================================

class TestCheckDataAlignment:

    def _make_pair(self, n: int = 5, price_diff: float = 0.0):
        rows_ts = []
        rows_tq = []
        for i in range(n):
            dt = f"2024-01-0{i + 1} 15:00:00"
            rows_ts.append({"ts_code": "IF2406.CFX", "datetime": dt,
                            "close": 4000.0 + i, "volume": 5000.0})
            rows_tq.append({"ts_code": "IF2406.CFX", "datetime": dt,
                            "close": 4000.0 + i + price_diff, "volume": 5000.0})
        return pd.DataFrame(rows_ts), pd.DataFrame(rows_tq)

    def test_returns_required_keys(self):
        df_ts, df_tq = self._make_pair()
        result = make_checker().check_data_alignment(df_ts, df_tq, ["ts_code", "datetime"])
        for key in ("overlap_rows", "price_mismatches", "volume_mismatches", "alignment_score"):
            assert key in result

    def test_perfect_alignment(self):
        df_ts, df_tq = self._make_pair()
        result = make_checker().check_data_alignment(df_ts, df_tq, ["ts_code", "datetime"])
        assert result["overlap_rows"] == 5
        assert result["price_mismatches"] == []
        assert result["alignment_score"] == 1.0

    def test_detects_price_mismatch(self):
        # price_diff 5 on base price 4000 → ~0.12% diff → above 0.1% threshold
        df_ts, df_tq = self._make_pair(price_diff=5.0)
        result = make_checker().check_data_alignment(df_ts, df_tq, ["ts_code", "datetime"])
        assert len(result["price_mismatches"]) > 0

    def test_detects_volume_mismatch(self):
        df_ts, df_tq = self._make_pair()
        df_tq.at[0, "volume"] = 10_000.0  # 100% diff
        result = make_checker().check_data_alignment(df_ts, df_tq, ["ts_code", "datetime"])
        assert len(result["volume_mismatches"]) > 0

    def test_empty_inputs(self):
        result = make_checker().check_data_alignment(
            pd.DataFrame(), pd.DataFrame(), ["datetime"]
        )
        assert result["overlap_rows"] == 0
        assert result["alignment_score"] == 1.0

    def test_no_overlap(self):
        df_ts = pd.DataFrame({"datetime": ["2024-01-01"], "close": [4000.0]})
        df_tq = pd.DataFrame({"datetime": ["2024-01-02"], "close": [4001.0]})
        result = make_checker().check_data_alignment(df_ts, df_tq, ["datetime"])
        assert result["overlap_rows"] == 0


# ======================================================================
# run_full_check — 旧接口 (str → QualityReport)
# ======================================================================

class TestRunFullCheckLegacy:

    def test_dispatches_futures_daily(self):
        checker = make_checker()
        report = checker.run_full_check("futures_daily", make_futures_df(),
                                        ts_code="IF2406.CFX")
        assert isinstance(report, QualityReport)
        assert report.table == "IF2406.CFX"

    def test_dispatches_options_daily(self):
        checker = make_checker()
        report = checker.run_full_check("options_daily", make_options_df())
        assert isinstance(report, QualityReport)
        assert report.table == "options_daily"

    def test_dispatches_futures_min(self):
        checker = make_checker()
        report = checker.run_full_check("futures_min", make_min_df(),
                                        ts_code="IF2406.CFX")
        assert isinstance(report, QualityReport)
        assert report.table == "IF2406.CFX"

    def test_generic_table(self):
        checker = make_checker()
        df = pd.DataFrame({"close": [100, 200], "volume": [10, 20]})
        report = checker.run_full_check("some_table", df)
        assert isinstance(report, QualityReport)
        assert report.table == "some_table"

    def test_none_df_handled(self):
        checker = make_checker()
        report = checker.run_full_check("futures_daily", None, ts_code="X")
        assert report.total_rows == 0

    def test_violations_appear_in_outlier_rows(self):
        df = make_futures_df(bad_rows={0: {"high": 3980.0, "low": 3995.0}})
        checker = make_checker()
        report = checker.run_full_check("futures_daily", df, ts_code="IF2406.CFX")
        assert 0 in report.outlier_rows
        assert not report.is_clean


# ======================================================================
# run_full_check — 新接口 (db_manager → Dict)
# ======================================================================

class TestRunFullCheckDbManager:

    def _make_db(self, futures_df=None, options_df=None, min_df=None):
        db = MagicMock()

        def query_side_effect(sql):
            if "futures_daily" in sql:
                return futures_df if futures_df is not None else pd.DataFrame()
            elif "options_daily" in sql:
                return options_df if options_df is not None else pd.DataFrame()
            elif "futures_min" in sql:
                return min_df if min_df is not None else pd.DataFrame()
            return pd.DataFrame()

        db.query.side_effect = query_side_effect
        return db

    def test_returns_dict(self):
        db = self._make_db()
        result = make_checker().run_full_check(db)
        assert isinstance(result, dict)

    def test_futures_daily_keys_present(self):
        df = make_futures_df()
        db = self._make_db(futures_df=df)
        result = make_checker().run_full_check(db)
        key = "futures_daily:IF2406.CFX"
        assert key in result
        assert "is_clean" in result[key]

    def test_options_daily_key_present(self):
        db = self._make_db(options_df=make_options_df(n=2))
        result = make_checker().run_full_check(db)
        assert "options_daily" in result

    def test_futures_min_key_present(self):
        db = self._make_db(min_df=make_min_df())
        result = make_checker().run_full_check(db)
        key = "futures_min:IF2406.CFX"
        assert key in result

    def test_query_exception_recorded(self):
        db = MagicMock()
        db.query.side_effect = RuntimeError("DB error")
        result = make_checker().run_full_check(db)
        assert "error" in result.get("futures_daily", {})

    def test_empty_tables_not_reported(self):
        db = self._make_db()  # all empty
        result = make_checker().run_full_check(db)
        # No futures_daily:xxx keys since df was empty
        assert not any(k.startswith("futures_daily:") for k in result)
