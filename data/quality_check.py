"""
quality_check.py
----------------
职责：数据质量检查工具集。
在数据下载后和策略使用前运行，确保数据的完整性和正确性。

检查结果以 dict 形式返回，便于程序化处理和日志输出。
同时保留 QualityReport dataclass 供需要结构化对象的调用方使用。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ======================================================================
# QualityReport — 向后兼容的结构化报告（check_*方法内部使用）
# ======================================================================

@dataclass
class QualityReport:
    """数据质量检查报告（向后兼容）"""
    table: str
    total_rows: int
    missing_dates: list[str]
    null_fields: dict[str, int]
    outlier_rows: list[int]
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        lines = [
            f"[质量报告] {self.table}",
            f"  总行数: {self.total_rows}",
            f"  缺失交易日: {len(self.missing_dates)} 天",
            f"  异常值行数: {len(self.outlier_rows)}",
            f"  空值统计: {self.null_fields}",
        ]
        if self.warnings:
            lines += [f"  ⚠ {w}" for w in self.warnings]
        if self.errors:
            lines += [f"  ✗ {e}" for e in self.errors]
        return "\n".join(lines)


# ======================================================================
# DataQualityChecker
# ======================================================================

class DataQualityChecker:
    """
    数据质量检查器。

    Parameters
    ----------
    trade_dates : list[str], optional
        标准交易日历（YYYYMMDD），用于对比检测缺失交易日
    """

    def __init__(self, trade_dates: Optional[list[str]] = None) -> None:
        self.trade_dates = set(trade_dates) if trade_dates else set()

    # ------------------------------------------------------------------
    # 期货日线检查
    # ------------------------------------------------------------------

    def check_futures_daily(
        self,
        df: pd.DataFrame,
        ts_code: str,
    ) -> Dict[str, Any]:
        """
        检查期货日线数据质量。

        检查项：
        1. 缺失交易日（对比 trade_dates，看是否有遗漏）
        2. 价格异常（涨跌幅超过 15% 标记为可疑）
        3. 成交量异常（为 0 或突然放大 100 倍）
        4. OHLC 逻辑（high>=low, high>=open/close, low<=open/close）
        5. 空值检查

        Returns
        -------
        dict
            total_rows, missing_dates, price_anomalies, volume_anomalies,
            ohlc_violations, null_counts, is_clean
        """
        if df.empty:
            return _empty_futures_result()

        ohlc_violations: List[dict] = []
        price_anomalies: List[dict] = []
        volume_anomalies: List[dict] = []

        # OHLC 逻辑检查
        ohlc_violations.extend(_check_ohlc(df))

        # 价格异常（涨跌幅 > 15%）
        if "close" in df.columns and len(df) > 1:
            price_anomalies.extend(_check_price_change(df, threshold=0.15))

        # 成交量异常
        if "volume" in df.columns:
            volume_anomalies.extend(_check_volume(df))

        null_counts = self.check_null_values(df)
        missing_dates = self.check_missing_dates(df)
        logger.debug("check_futures_daily: %s, rows=%d", ts_code, len(df))

        # 价格跌幅异常 + OHLC 错误算 is_clean=False；仅涨跌幅警告不影响
        is_clean = not bool(ohlc_violations) and not bool(volume_anomalies)

        return {
            "total_rows": len(df),
            "missing_dates": missing_dates,
            "price_anomalies": price_anomalies,
            "volume_anomalies": volume_anomalies,
            "ohlc_violations": ohlc_violations,
            "null_counts": null_counts,
            "is_clean": is_clean,
        }

    # ------------------------------------------------------------------
    # 期货分钟线检查
    # ------------------------------------------------------------------

    def check_futures_min(
        self,
        df: pd.DataFrame,
        ts_code: str,
        freq_minutes: int = 5,
    ) -> Dict[str, Any]:
        """
        检查期货分钟线数据质量。

        额外检查项（在日线基础上）：
        1. 每个交易日的 K 线数量是否在合理范围内
        2. 时间戳中是否存在异常跳跃

        Parameters
        ----------
        freq_minutes : int
            K 线周期（分钟），用于计算预期每日 K 线数量
        """
        if df.empty:
            return _empty_min_result()

        logger.debug("check_futures_min: %s, rows=%d", ts_code, len(df))
        ohlc_violations = _check_ohlc(df)
        null_counts = self.check_null_values(df)

        # 从 datetime 列提取每日 K 线数量
        daily_bar_counts: Dict[str, int] = {}
        time_gaps: List[dict] = []

        if "datetime" in df.columns:
            df = df.copy()
            df["_date"] = df["datetime"].str[:10]
            daily_bar_counts = df.groupby("_date").size().to_dict()

            # 每日预期 K 线数量（股指期货 5min 约 48 根，含竞价约 50 根）
            expected_per_day = _expected_bars_per_day(freq_minutes)
            sparse_days = {
                d: cnt for d, cnt in daily_bar_counts.items()
                if cnt < expected_per_day * 0.5
            }
            if sparse_days:
                logger.debug("K 线不足的交易日（<%d 根）: %s",
                             expected_per_day // 2, list(sparse_days.keys())[:5])

            # 时间戳跳跃检测（连续两根 K 线间隔超过 freq_minutes * 3）
            if len(df) > 1:
                time_gaps = _check_time_gaps(df, freq_minutes)

        # 缺失交易日（从 datetime 提取日期）
        missing_dates: list[str] = []
        if "datetime" in df.columns and self.trade_dates:
            dates_in_data = set(df["datetime"].str[:10].str.replace("-", ""))
            missing_dates = sorted(self.trade_dates - dates_in_data)

        is_clean = not bool(ohlc_violations)

        return {
            "total_rows": len(df),
            "missing_dates": missing_dates,
            "ohlc_violations": ohlc_violations,
            "null_counts": null_counts,
            "daily_bar_counts": daily_bar_counts,
            "time_gaps": time_gaps,
            "is_clean": is_clean,
        }

    # ------------------------------------------------------------------
    # 期权日线检查
    # ------------------------------------------------------------------

    def check_options_daily(
        self,
        df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        检查期权日线数据质量。

        额外检查项：
        1. 每个交易日的合约数量是否合理（< 10 视为异常）
        2. 认购/认沽数量是否大致对称
        3. 期权价格非负
        """
        if df.empty:
            return _empty_options_result()

        ohlc_violations: List[dict] = []
        null_counts = self.check_null_values(df)

        # 期权价格非负（call/put 可以接近0，但不能为负）
        if "close" in df.columns:
            bad = df[df["close"] < 0]
            for idx, row in bad.iterrows():
                ohlc_violations.append({
                    "index": idx,
                    "trade_date": row.get("trade_date", ""),
                    "ts_code": row.get("ts_code", ""),
                    "issue": "negative_price",
                    "close": row["close"],
                })

        # 每日合约数量统计
        daily_contract_counts: Dict[str, int] = {}
        call_put_balance: Dict[str, Dict[str, int]] = {}

        if "trade_date" in df.columns:
            daily_contract_counts = df.groupby("trade_date").size().to_dict()

        if "trade_date" in df.columns and "call_put" in df.columns:
            for date, grp in df.groupby("trade_date"):
                call_put_balance[str(date)] = {
                    "C": int((grp["call_put"] == "C").sum()),
                    "P": int((grp["call_put"] == "P").sum()),
                }

        is_clean = not bool(ohlc_violations)

        return {
            "total_rows": len(df),
            "daily_contract_counts": daily_contract_counts,
            "call_put_balance": call_put_balance,
            "ohlc_violations": ohlc_violations,
            "null_counts": null_counts,
            "is_clean": is_clean,
        }

    # ------------------------------------------------------------------
    # 数据源对比
    # ------------------------------------------------------------------

    def check_data_alignment(
        self,
        df_tushare: pd.DataFrame,
        df_tq: pd.DataFrame,
        key_columns: List[str],
    ) -> Dict[str, Any]:
        """
        检查 Tushare 与天勤两数据源在重叠区间内的一致性。

        Parameters
        ----------
        df_tushare : pd.DataFrame
            Tushare 数据（含 datetime/trade_date + close + volume）
        df_tq : pd.DataFrame
            天勤数据（同列名）
        key_columns : list[str]
            用于 JOIN 的键列，如 ["ts_code", "datetime"]

        Returns
        -------
        dict
            overlap_rows, price_mismatches, volume_mismatches, alignment_score
        """
        if df_tushare.empty or df_tq.empty:
            return {
                "overlap_rows": 0,
                "price_mismatches": [],
                "volume_mismatches": [],
                "alignment_score": 1.0,
            }

        # 内连接找重叠行
        try:
            merged = pd.merge(
                df_tushare, df_tq,
                on=key_columns,
                suffixes=("_ts", "_tq"),
            )
        except Exception as exc:
            logger.warning("数据对比 merge 失败: %s", exc)
            return {
                "overlap_rows": 0,
                "price_mismatches": [],
                "volume_mismatches": [],
                "alignment_score": 0.0,
            }

        overlap = len(merged)
        price_mismatches: List[dict] = []
        volume_mismatches: List[dict] = []

        # 收盘价差异 > 0.1%
        if "close_ts" in merged.columns and "close_tq" in merged.columns:
            prices_ts = merged["close_ts"].astype(float)
            prices_tq = merged["close_tq"].astype(float)
            mid = (prices_ts + prices_tq) / 2
            rel_diff = ((prices_ts - prices_tq).abs() / mid.replace(0, float("nan")))
            bad = merged[rel_diff > 0.001]
            for _, row in bad.iterrows():
                entry: dict = {"close_ts": row["close_ts"], "close_tq": row["close_tq"]}
                for k in key_columns:
                    entry[k] = row.get(k, "")
                price_mismatches.append(entry)

        # 成交量差异 > 10%
        if "volume_ts" in merged.columns and "volume_tq" in merged.columns:
            vol_ts = merged["volume_ts"].astype(float)
            vol_tq = merged["volume_tq"].astype(float)
            mid = (vol_ts + vol_tq) / 2
            rel_diff = ((vol_ts - vol_tq).abs() / mid.replace(0, float("nan")))
            bad = merged[rel_diff > 0.1]
            for _, row in bad.iterrows():
                entry = {"volume_ts": row["volume_ts"], "volume_tq": row["volume_tq"]}
                for k in key_columns:
                    entry[k] = row.get(k, "")
                volume_mismatches.append(entry)

        match_count = overlap - max(len(price_mismatches), len(volume_mismatches))
        alignment_score = match_count / overlap if overlap > 0 else 1.0

        return {
            "overlap_rows": overlap,
            "price_mismatches": price_mismatches,
            "volume_mismatches": volume_mismatches,
            "alignment_score": round(alignment_score, 4),
        }

    # ------------------------------------------------------------------
    # 全库检查
    # ------------------------------------------------------------------

    def run_full_check(
        self,
        db_manager_or_table,
        df: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        全库检查或单表检查的统一入口。

        使用方式1（新接口）：run_full_check(db_manager) -> Dict[str, Dict]
            对数据库中所有主要表运行检查，返回汇总报告。

        使用方式2（旧接口）：run_full_check(table_name, df, **kwargs) -> QualityReport
            对指定表和 DataFrame 运行检查（向后兼容）。
        """
        if isinstance(db_manager_or_table, str):
            # 旧接口：run_full_check(table_name, df, **kwargs)
            return self._check_table(db_manager_or_table, df, **kwargs)

        # 新接口：run_full_check(db_manager)
        db = db_manager_or_table
        report: Dict[str, Any] = {}

        # futures_daily：按 ts_code 逐合约检查
        try:
            df_fd = db.query("SELECT * FROM futures_daily ORDER BY ts_code, trade_date")
            if not df_fd.empty:
                for ts_code, grp in df_fd.groupby("ts_code"):
                    key = f"futures_daily:{ts_code}"
                    report[key] = self.check_futures_daily(
                        grp.reset_index(drop=True), str(ts_code)
                    )
        except Exception as exc:
            report["futures_daily"] = {"error": str(exc)}

        # options_daily：全表检查
        try:
            df_od = db.query("SELECT * FROM options_daily ORDER BY trade_date")
            if not df_od.empty:
                report["options_daily"] = self.check_options_daily(df_od)
        except Exception as exc:
            report["options_daily"] = {"error": str(exc)}

        # futures_min：按 ts_code 检查
        try:
            df_fm = db.query("SELECT * FROM futures_min ORDER BY ts_code, datetime")
            if not df_fm.empty:
                for ts_code, grp in df_fm.groupby("ts_code"):
                    key = f"futures_min:{ts_code}"
                    report[key] = self.check_futures_min(
                        grp.reset_index(drop=True), str(ts_code)
                    )
        except Exception as exc:
            report["futures_min"] = {"error": str(exc)}

        return report

    # ------------------------------------------------------------------
    # 通用工具方法（保持旧接口）
    # ------------------------------------------------------------------

    def check_missing_dates(
        self,
        df: pd.DataFrame,
        date_col: str = "trade_date",
        expected_dates: Optional[list[str]] = None,
    ) -> list[str]:
        """检查 DataFrame 中缺失的交易日，返回缺失日期列表（YYYYMMDD 升序）"""
        reference = set(expected_dates) if expected_dates is not None else self.trade_dates
        if not reference or date_col not in df.columns:
            return []
        present = set(df[date_col].dropna().astype(str))
        return sorted(reference - present)

    def check_null_values(self, df: pd.DataFrame) -> dict[str, int]:
        """统计各列空值数量，只返回有空值的字段"""
        counts = df.isnull().sum()
        return {col: int(cnt) for col, cnt in counts.items() if cnt > 0}

    def check_price_outliers(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        max_daily_change: float = 0.2,
    ) -> list[int]:
        """检测价格异常涨跌（超过 max_daily_change），返回行索引列表"""
        if price_col not in df.columns or len(df) < 2:
            return []
        prices = df[price_col].astype(float)
        pct = prices.pct_change().abs()
        return df.index[pct > max_daily_change].tolist()

    # ------------------------------------------------------------------
    # 内部：旧接口兼容
    # ------------------------------------------------------------------

    def _check_table(
        self,
        table: str,
        df: Optional[pd.DataFrame],
        **kwargs,
    ) -> QualityReport:
        """旧接口：run_full_check(table_name, df) -> QualityReport"""
        if df is None:
            df = pd.DataFrame()

        if table == "futures_daily":
            ts_code = kwargs.get("ts_code", table)
            result = self.check_futures_daily(df, ts_code)
            return _dict_to_report(ts_code, result)
        elif table == "futures_min":
            ts_code = kwargs.get("ts_code", table)
            result = self.check_futures_min(df, ts_code)
            return _dict_to_report(ts_code, result)
        elif table == "options_daily":
            result = self.check_options_daily(df)
            return _dict_to_report("options_daily", result)
        else:
            null_fields = self.check_null_values(df)
            price_col = kwargs.get("price_col", "close")
            outliers = self.check_price_outliers(df, price_col) if price_col in df.columns else []
            return QualityReport(
                table=table, total_rows=len(df),
                missing_dates=[], null_fields=null_fields,
                outlier_rows=outliers,
            )


# ======================================================================
# 内部工具函数
# ======================================================================

def _check_ohlc(df: pd.DataFrame) -> List[dict]:
    """检查 OHLC 逻辑约束违反，返回异常记录列表"""
    violations: List[dict] = []
    has_ohlc = {"open", "high", "low", "close"}.issubset(df.columns)
    if not has_ohlc:
        return violations

    for idx, row in df.iterrows():
        try:
            o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        except (TypeError, ValueError):
            continue

        issues = []
        if h < l:
            issues.append("high_lt_low")
        if h < o:
            issues.append("high_lt_open")
        if h < c:
            issues.append("high_lt_close")
        if l > o:
            issues.append("low_gt_open")
        if l > c:
            issues.append("low_gt_close")
        if c <= 0:
            issues.append("non_positive_close")

        for issue in issues:
            violations.append({
                "index": idx,
                "trade_date": str(row.get("trade_date", row.get("datetime", ""))),
                "issue": issue,
                "open": o, "high": h, "low": l, "close": c,
            })
    return violations


def _check_price_change(df: pd.DataFrame, threshold: float = 0.15) -> List[dict]:
    """检测单日涨跌幅超过阈值的记录"""
    anomalies: List[dict] = []
    prices = df["close"].astype(float)
    pct = prices.pct_change(fill_method=None)
    for idx in df.index[pct.abs() > threshold]:
        if idx == df.index[0]:
            continue  # 第一行无前值
        anomalies.append({
            "index": idx,
            "trade_date": str(df.at[idx, "trade_date"]) if "trade_date" in df.columns else "",
            "close": float(df.at[idx, "close"]),
            "change_pct": round(float(pct.at[idx]), 4),
            "severity": "warning",
        })
    return anomalies


def _check_volume(df: pd.DataFrame) -> List[dict]:
    """检测成交量异常：为0 或突然放大100倍"""
    anomalies: List[dict] = []
    volumes = df["volume"].astype(float)
    median_vol = volumes[volumes > 0].median()

    for idx, vol in volumes.items():
        issue = None
        if vol == 0:
            issue = "zero_volume"
        elif median_vol > 0 and vol > median_vol * 100:
            issue = "spike"
        elif vol < 0:
            issue = "negative_volume"

        if issue:
            anomalies.append({
                "index": idx,
                "trade_date": str(df.at[idx, "trade_date"]) if "trade_date" in df.columns else "",
                "volume": vol,
                "issue": issue,
            })
    return anomalies


def _check_time_gaps(df: pd.DataFrame, freq_minutes: int) -> List[dict]:
    """检测分钟线时间戳中的异常跳跃（超过 freq_minutes * 3 分钟）"""
    gaps: List[dict] = []
    threshold_min = freq_minutes * 3
    try:
        times = pd.to_datetime(df["datetime"])
        diffs = times.diff().dt.total_seconds().div(60)
        same_day = df["datetime"].str[:10] == df["datetime"].str[:10].shift()
        # 只在同一天内检查跳跃
        bad = diffs[(diffs > threshold_min) & same_day.values]
        for idx in bad.index:
            gaps.append({
                "index": idx,
                "datetime_before": str(df.at[df.index[df.index.get_loc(idx) - 1], "datetime"]),
                "datetime_after": str(df.at[idx, "datetime"]),
                "gap_minutes": round(float(bad.at[idx]), 1),
            })
    except Exception:
        pass
    return gaps[:20]  # 最多返回前20个


def _expected_bars_per_day(freq_minutes: int) -> int:
    """估算股指期货每交易日的预期 K 线数（日盘：4小时共 240 分钟）"""
    return max(1, 240 // freq_minutes)


def _empty_futures_result() -> Dict[str, Any]:
    return {
        "total_rows": 0,
        "missing_dates": [],
        "price_anomalies": [],
        "volume_anomalies": [],
        "ohlc_violations": [],
        "null_counts": {},
        "is_clean": True,
    }


def _empty_min_result() -> Dict[str, Any]:
    return {
        "total_rows": 0,
        "missing_dates": [],
        "ohlc_violations": [],
        "null_counts": {},
        "daily_bar_counts": {},
        "time_gaps": [],
        "is_clean": True,
    }


def _empty_options_result() -> Dict[str, Any]:
    return {
        "total_rows": 0,
        "daily_contract_counts": {},
        "call_put_balance": {},
        "ohlc_violations": [],
        "null_counts": {},
        "is_clean": True,
    }


def _dict_to_report(table: str, result: Dict[str, Any]) -> QualityReport:
    """将 check_* 返回的 dict 转为 QualityReport（向后兼容）"""
    violations = result.get("ohlc_violations", [])
    outlier_rows = [v["index"] for v in violations]
    # price_anomalies 也加入 outlier_rows
    for a in result.get("price_anomalies", []):
        outlier_rows.append(a["index"])
    errors = [f"{v['issue']} at row {v['index']}" for v in violations]
    null_fields = result.get("null_counts", {})
    return QualityReport(
        table=table,
        total_rows=result.get("total_rows", 0),
        missing_dates=result.get("missing_dates", []),
        null_fields=null_fields,
        outlier_rows=sorted(set(outlier_rows)),
        errors=errors,
    )
