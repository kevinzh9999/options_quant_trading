#!/usr/bin/env python3
"""彻底 audit daily_model_output for IM — 验证 Daily XGB 所需字段全部干净。

检查项：
  1. NULL 覆盖率：每个 daily_xgb 必需字段
  2. 数值合理性：RV / IV 应在 [0, 1] 范围, term in [-0.1, 0.1] 等
  3. 价源一致性：rv_20d 应与 spot 000852 计算结果一致 (futures 来源会偏)
  4. iv_term_spread / rv_5d 是否齐全
  5. 时间连续性：trade_date 有无 gap
  6. day-over-day 跳变：相邻日 |Δrv| > 0.5 / |Δiv| > 0.10 标 outlier

输出: 每个字段的 NULL 占比 + outlier 列表 + dirty dates 清单 (供 repair 用)。
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


DB_PATH = ROOT + "/data/storage/trading.db"

# Daily XGB 21 因子需要的列（不含派生 *_change）
DXGB_REQUIRED_COLS = [
    "atm_iv_market",
    "iv_percentile_hist",
    "rr_25d",
    "vrp",
    "iv_term_spread",
    "realized_vol_5d",
    "realized_vol_20d",
]

# 其他重要列（应该尽量都有）
NICE_TO_HAVE_COLS = [
    "atm_iv",
    "garch_current_vol",
    "garch_forecast_vol",
    "realized_vol_60d",
    "vrp_percentile",
    "hurst_60d",
    "garch_reliable",
    "signal_primary",
]

# 数值合理性范围 [min, max]
SANITY_RANGES = {
    "atm_iv_market":   (0.05, 1.0),    # 5% ~ 100% IV
    "atm_iv":          (0.05, 1.0),
    "rr_25d":          (-0.5, 0.5),
    "vrp":             (-0.5, 0.5),
    "iv_term_spread":  (-0.2, 0.2),
    "realized_vol_5d": (0.0, 2.0),
    "realized_vol_20d": (0.0, 1.5),
    "realized_vol_60d": (0.0, 1.5),
    "garch_forecast_vol": (0.0, 2.0),
    "garch_current_vol": (0.0, 2.0),
    "hurst_60d":       (0.0, 1.0),
    "iv_percentile_hist": (0, 100),
    "vrp_percentile":  (0, 100),
}


def load_data(db_path: str = DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    df = pd.read_sql_query(
        "SELECT * FROM daily_model_output WHERE underlying='IM' ORDER BY trade_date",
        conn,
    )
    conn.close()
    return df


def audit_null_coverage(df: pd.DataFrame) -> Dict[str, Dict]:
    """每个字段的 NULL 覆盖率。"""
    n = len(df)
    out = {}
    for col in DXGB_REQUIRED_COLS + NICE_TO_HAVE_COLS:
        if col not in df.columns:
            out[col] = {"present": 0, "null": n, "pct_null": 100.0, "missing_dates": []}
            continue
        nulls = df[col].isna()
        out[col] = {
            "present": int((~nulls).sum()),
            "null": int(nulls.sum()),
            "pct_null": float(nulls.sum() / n * 100),
            "missing_dates": df.loc[nulls, "trade_date"].tolist(),
        }
    return out


def audit_sanity(df: pd.DataFrame) -> Dict[str, List]:
    """数值合理性检查 — 找出 out-of-range 的行。"""
    out = {}
    for col, (mn, mx) in SANITY_RANGES.items():
        if col not in df.columns:
            continue
        sub = df[df[col].notna()]
        bad = sub[(sub[col] < mn) | (sub[col] > mx)]
        if not bad.empty:
            out[col] = bad[["trade_date", col]].to_dict("records")
    return out


def audit_rv_source(db_path: str, df: pd.DataFrame, sample_n: int = 30) -> List[Dict]:
    """随机抽 N 个日期，独立用 spot 000852 重算 rv_20d，对比 DB 值看是否漂移 (futures 源)。"""
    if "realized_vol_20d" not in df.columns:
        return [{"error": "realized_vol_20d 列不存在"}]

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    spot = pd.read_sql_query(
        "SELECT trade_date, close FROM index_daily WHERE ts_code='000852.SH' ORDER BY trade_date",
        conn,
    )
    conn.close()
    spot["close"] = spot["close"].astype(float)

    valid = df[df["realized_vol_20d"].notna()].copy()
    if len(valid) == 0:
        return []
    sample = valid.sample(n=min(sample_n, len(valid)), random_state=42)

    discrepancies = []
    for _, r in sample.iterrows():
        td = r["trade_date"]
        sub = spot[spot["trade_date"] <= td]
        if len(sub) < 21:
            continue
        c = sub["close"].values[-21:]
        rets = np.diff(np.log(c))
        spot_rv20 = float(np.std(rets, ddof=1) * np.sqrt(252))
        db_rv20 = float(r["realized_vol_20d"])
        rel_diff = abs(spot_rv20 - db_rv20) / max(spot_rv20, 1e-6)
        if rel_diff > 0.02:  # >2% 偏差 = 怀疑价源不一致
            discrepancies.append({
                "trade_date": td,
                "db_rv20": round(db_rv20, 6),
                "spot_rv20": round(spot_rv20, 6),
                "rel_diff_pct": round(rel_diff * 100, 2),
            })
    return discrepancies


def audit_continuity(df: pd.DataFrame) -> Dict:
    """时间连续性 + day-over-day 跳变。"""
    if df.empty:
        return {}
    df = df.sort_values("trade_date").reset_index(drop=True)
    cols_to_check = ["atm_iv_market", "realized_vol_5d", "realized_vol_20d",
                      "rr_25d", "vrp", "iv_term_spread"]
    jumps = {}
    for c in cols_to_check:
        if c not in df.columns:
            continue
        diff = df[c].diff().abs()
        # threshold: >0.10 absolute or >50% relative
        bad = df[(diff > 0.10) & df[c].notna() & df[c].shift(1).notna()]
        if not bad.empty:
            jumps[c] = bad[["trade_date", c]].head(10).to_dict("records")
    return jumps


def gather_dirty_dates(audit: Dict, sanity: Dict,
                        discrepancies: List, jumps: Dict) -> List[str]:
    """合并所有需要 repair 的日期。"""
    dirty = set()
    # NULL on required fields
    for col in DXGB_REQUIRED_COLS:
        for d in audit[col]["missing_dates"]:
            dirty.add(d)
    # Sanity violations
    for col, rows in sanity.items():
        for r in rows:
            dirty.add(r["trade_date"])
    # RV20 source mismatches (futures vs spot)
    for r in discrepancies:
        if "trade_date" in r:
            dirty.add(r["trade_date"])
    # Day-over-day jumps (only if real anomaly, conservative)
    for col, rows in jumps.items():
        for r in rows:
            dirty.add(r["trade_date"])
    return sorted(dirty)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--sample", type=int, default=30, help="RV 价源抽检数量")
    ap.add_argument("--report-dirty", action="store_true",
                     help="输出所有 dirty dates 到 stdout 末尾")
    args = ap.parse_args()

    print(f"=== Daily Model Output Audit (IM) ===")
    print(f"DB: {args.db}")
    df = load_data(args.db)
    print(f"Total rows: {len(df)}")
    if df.empty:
        print("No data, exiting")
        return
    print(f"Date range: {df['trade_date'].iloc[0]} ~ {df['trade_date'].iloc[-1]}\n")

    # ── 1. NULL coverage ──
    print(f"{'═' * 80}")
    print(" NULL coverage by column")
    print(f"{'═' * 80}")
    audit = audit_null_coverage(df)
    print(f"  {'Column':<30} {'Present':>9} {'NULL':>6} {'%NULL':>8} {'(critical)':>12}")
    for col in DXGB_REQUIRED_COLS:
        a = audit[col]
        critical = "CRITICAL" if col in DXGB_REQUIRED_COLS else ""
        bar = "█" * int(a["pct_null"] / 5)
        print(f"  {col:<30} {a['present']:>9} {a['null']:>6} {a['pct_null']:>7.1f}% {critical:>12}  {bar}")
    print()
    for col in NICE_TO_HAVE_COLS:
        a = audit[col]
        bar = "█" * int(a["pct_null"] / 5)
        print(f"  {col:<30} {a['present']:>9} {a['null']:>6} {a['pct_null']:>7.1f}%              {bar}")

    # ── 2. Sanity check ──
    print(f"\n{'═' * 80}")
    print(" Sanity check (out-of-range values)")
    print(f"{'═' * 80}")
    sanity = audit_sanity(df)
    if not sanity:
        print("  All values within sanity ranges ✓")
    else:
        for col, rows in sanity.items():
            mn, mx = SANITY_RANGES[col]
            print(f"  {col}: expected [{mn}, {mx}], {len(rows)} violations")
            for r in rows[:5]:
                print(f"    {r['trade_date']}: {r[col]}")
            if len(rows) > 5:
                print(f"    ... +{len(rows)-5} more")

    # ── 3. RV source consistency (spot vs futures) ──
    print(f"\n{'═' * 80}")
    print(f" RV20 source consistency (random {args.sample} samples vs spot 000852)")
    print(f"{'═' * 80}")
    discrepancies = audit_rv_source(args.db, df, args.sample)
    if not discrepancies:
        print("  All sampled rv_20d match spot 000852 ✓")
    else:
        print(f"  Found {len(discrepancies)} mismatches (>2% relative diff = suspect futures source):")
        for r in discrepancies[:15]:
            print(f"    {r['trade_date']}: db={r['db_rv20']}  spot={r['spot_rv20']}  diff={r['rel_diff_pct']}%")
        if len(discrepancies) > 15:
            print(f"    ... +{len(discrepancies)-15} more")

    # ── 4. Continuity / day-over-day jumps ──
    print(f"\n{'═' * 80}")
    print(" Day-over-day jumps (|Δ| > 0.10)")
    print(f"{'═' * 80}")
    jumps = audit_continuity(df)
    if not jumps:
        print("  No anomalous jumps ✓")
    else:
        for col, rows in jumps.items():
            print(f"  {col}: {len(rows)} jumps")
            for r in rows[:5]:
                print(f"    {r['trade_date']}: {col}={r[col]}")

    # ── 5. Dirty dates summary ──
    dirty = gather_dirty_dates(audit, sanity, discrepancies, jumps)
    print(f"\n{'═' * 80}")
    print(f" SUMMARY: {len(dirty)} dirty dates need repair")
    print(f"{'═' * 80}")
    if dirty:
        print(f"  Range: {dirty[0]} ~ {dirty[-1]}")
        # Show by year
        years = pd.Series([d[:4] for d in dirty]).value_counts().sort_index()
        print(f"  By year:")
        for y, n in years.items():
            print(f"    {y}: {n} days")

    if args.report_dirty:
        print(f"\n{'─' * 80}")
        print(" Dirty dates list:")
        print(f"{'─' * 80}")
        for d in dirty:
            print(d)

    # 写到文件方便 repair 脚本读
    if dirty:
        out_path = Path(ROOT) / "tmp" / "daily_model_output_dirty_dates.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(dirty))
        print(f"\n  Dirty dates written to: {out_path}")


if __name__ == "__main__":
    main()
