#!/usr/bin/env python3
"""IC threshold验证（独立进程，跟IM并行跑）。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONUNBUFFERED"] = "1"

from data.storage.db_manager import get_db
from scripts.threshold_validation import get_dates, run_threshold_scan, run_profiling_summary

THRESHOLDS = [50, 55, 60, 65]
db = get_db()
ic_dates = get_dates(db, "000905")

ic_seg1 = [d for d in ic_dates if d < '20200701']
ic_seg2 = [d for d in ic_dates if '20200701' <= d < '20221001']
ic_seg3 = [d for d in ic_dates if d >= '20221001']

for seg_name, seg_dates in [("Seg1(2018~mid2020)", ic_seg1),
                             ("Seg2(mid2020~2022)", ic_seg2),
                             ("Seg3(2022~now)", ic_seg3)]:
    if not seg_dates:
        continue
    print(f"\n[IC] {seg_name} ({len(seg_dates)}天)")
    run_threshold_scan('IC', seg_dates, THRESHOLDS, db)

print(f"\n[IC] Seg3 profiling ({len(ic_seg3)}天)...")
prof = run_profiling_summary('IC', ic_seg3, db)
if len(prof) > 0:
    print(prof.to_string())
print("\nIC验证完成")
