#!/usr/bin/env python3
"""彻底 repair daily_model_output for IM — 重新计算所有 dirty/incomplete 日期。

策略:
  1. 读 tmp/daily_model_output_dirty_dates.txt (audit 输出)
     或者 --rebuild-all 重算所有日期
  2. 对每个 trade_date 调 daily_record._eod_model_output(td)
     - 现在用 UPDATE-or-INSERT 语义（preserves pnl_* and research cols）
     - 自动用 spot 000852 算 RV
     - 自动写 iv_term_spread + realized_vol_5d + rr_25d + hurst_60d
  3. 多进程并行（可选）
  4. 完成后建议重跑 audit 验证

用法:
    python scripts/repair_daily_model_output.py --dirty-only
    python scripts/repair_daily_model_output.py --range 20260316-20260402
    python scripts/repair_daily_model_output.py --rebuild-all     # 慎用 ~30min
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import List

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DB_PATH = ROOT + "/data/storage/trading.db"
DIRTY_FILE = ROOT + "/tmp/daily_model_output_dirty_dates.txt"


def get_all_im_dates(db_path: str) -> List[str]:
    """所有有 IM.CFX 数据的日期。"""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = conn.execute(
        "SELECT DISTINCT trade_date FROM futures_daily "
        "WHERE ts_code='IM.CFX' ORDER BY trade_date"
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


def parse_date_range(spec: str) -> List[str]:
    """parse 'YYYYMMDD-YYYYMMDD' into list of dates in range."""
    start, end = spec.split("-")
    all_dates = get_all_im_dates(DB_PATH)
    return [d for d in all_dates if start <= d <= end]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirty-only", action="store_true",
                     help="只修复 audit 标记的 dirty dates")
    ap.add_argument("--range", default=None, help="日期范围 YYYYMMDD-YYYYMMDD")
    ap.add_argument("--rebuild-all", action="store_true",
                     help="重算所有 IM 日期（慎用 ~30min）")
    ap.add_argument("--date", default=None, help="单个日期 YYYYMMDD")
    ap.add_argument("--limit", type=int, default=None, help="限制处理数量（测试用）")
    args = ap.parse_args()

    # Determine target dates
    if args.date:
        targets = [args.date]
    elif args.range:
        targets = parse_date_range(args.range)
    elif args.rebuild_all:
        targets = get_all_im_dates(DB_PATH)
    elif args.dirty_only:
        if not Path(DIRTY_FILE).exists():
            print(f"[ERROR] {DIRTY_FILE} not found, run audit first")
            sys.exit(1)
        targets = Path(DIRTY_FILE).read_text().strip().split("\n")
    else:
        print("Specify one of: --dirty-only / --range / --rebuild-all / --date")
        sys.exit(1)

    targets = [t for t in targets if t]   # filter empties
    if args.limit:
        targets = targets[: args.limit]

    print(f"=== Daily Model Output Repair ===")
    print(f"Targets: {len(targets)} dates")
    print(f"Range: {targets[0]} ~ {targets[-1]}")
    print()

    # Import within main to avoid loading on import (logger setup etc.)
    from scripts.daily_record import _eod_model_output, _open_db

    db = _open_db()
    success = 0
    skipped = 0
    failed = 0
    failures = []

    for i, td in enumerate(targets, 1):
        try:
            result = _eod_model_output(td, db)
            if result is None:
                skipped += 1
                continue
            success += 1
            if i % 50 == 0 or i <= 3 or i == len(targets):
                rv5 = result.get("realized_vol_5d") or 0
                rv20 = result.get("realized_vol_20d") or 0
                term = result.get("iv_term_spread") or 0
                rr = result.get("rr_25d") or 0
                iv_mkt = result.get("atm_iv_market") or 0
                print(f"  [{i}/{len(targets)}] {td}: "
                      f"rv5={rv5:.4f}  rv20={rv20:.4f}  "
                      f"iv_mkt={iv_mkt:.4f}  term={term:+.4f}  rr={rr:+.4f}")
        except Exception as e:
            failed += 1
            failures.append((td, str(e)))
            print(f"  [{i}/{len(targets)}] {td}: FAILED — {e}")

    print()
    print(f"=== Repair Summary ===")
    print(f"  Total:    {len(targets)}")
    print(f"  Success:  {success}")
    print(f"  Skipped:  {skipped}  (data missing/insufficient)")
    print(f"  Failed:   {failed}")
    if failures:
        print(f"\n  Failures:")
        for td, err in failures[:10]:
            print(f"    {td}: {err}")

    print(f"\nNext step: re-run audit to verify")
    print(f"  python scripts/audit_daily_model_output.py")


if __name__ == "__main__":
    main()
