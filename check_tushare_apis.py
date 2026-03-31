# 保存为 check_tushare_apis.py 然后运行
import tushare as ts
import os
from dotenv import load_dotenv
load_dotenv()

ts.set_token(os.getenv("TUSHARE_TOKEN"))
pro = ts.pro_api()

# 1. 全球指数（A50/SPX/恒生）
print("=== index_global ===")
try:
    df = pro.index_global(ts_code="XIN9", trade_date="20260327")
    print(f"  OK: {len(df)} rows" if len(df) > 0 else "  OK but empty")
    print(f"  {df.head()}")
except Exception as e:
    print(f"  FAIL: {e}")

# 2. 外汇日线
print("\n=== fx_daily ===")
try:
    df = pro.fx_daily(ts_code="USDCNY.FXCM", start_date="20260320", end_date="20260327")
    print(f"  OK: {len(df)} rows" if len(df) > 0 else "  OK but empty")
except Exception as e:
    print(f"  FAIL: {e}")

# 3. 全A股日线（涨跌家数用）
print("\n=== daily (sample) ===")
try:
    df = pro.daily(trade_date="20260327", fields="ts_code,pct_chg")
    print(f"  OK: {len(df)} stocks")
    up = len(df[df["pct_chg"] > 0])
    down = len(df[df["pct_chg"] < 0])
    limit_up = len(df[df["pct_chg"] >= 9.9])
    limit_down = len(df[df["pct_chg"] <= -9.9])
    print(f"  涨:{up} 跌:{down} 涨停:{limit_up} 跌停:{limit_down}")
except Exception as e:
    print(f"  FAIL: {e}")

# 4. 沪深两市成交额
print("\n=== index_daily (market turnover) ===")
try:
    df = pro.index_daily(ts_code="000001.SH", start_date="20260325", end_date="20260327")
    print(f"  OK: {len(df)} rows")
    print(f"  {df[['trade_date','close','vol','amount']].to_string()}")
except Exception as e:
    print(f"  FAIL: {e}")

print("\n=== Done ===")