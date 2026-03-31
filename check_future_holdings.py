import tushare as ts
import os
from dotenv import load_dotenv
load_dotenv()
ts.set_token(os.getenv("TUSHARE_TOKEN"))
pro = ts.pro_api()

# 1. 全球指数
print("=== index_global 历史深度 ===")
for code, name in [("XIN9","A50"), ("SPX","标普500"), ("IXIC","纳斯达克"), ("HSI","恒生")]:
    try:
        # 测试各年代第一周
        for year in [2010, 2012, 2015, 2018, 2020, 2024, 2026]:
            df = pro.index_global(ts_code=code, start_date=f"{year}0101", end_date=f"{year}0115")
            if len(df) > 0:
                print(f"  {name}({code}): {year}年 ✅ {len(df)}条")
            else:
                print(f"  {name}({code}): {year}年 ❌ 空")
    except Exception as e:
        print(f"  {name}({code}): FAIL - {e}")
    print()

# 2. 全A股日线（用确定有交易的日期）
print("=== daily 历史深度 ===")
for date in ["20100104","20120104","20150105","20180102","20200102","20240102","20260327"]:
    try:
        df = pro.daily(trade_date=date, fields="ts_code,pct_chg")
        print(f"  {date}: {len(df)} stocks")
    except Exception as e:
        print(f"  {date}: FAIL - {e}")

# 3. 北向资金
print("\n=== moneyflow_hsgt 历史深度 ===")
for date in ["20141117","20150105","20180102","20200102","20240102","20260327"]:
    try:
        df = pro.moneyflow_hsgt(trade_date=date)
        print(f"  {date}: {len(df)} rows")
    except Exception as e:
        print(f"  {date}: FAIL - {e}")

# 4. 融资融券
print("\n=== margin 历史深度 ===")
for date in ["20100104","20120104","20150105","20180102","20200102","20240102","20260327"]:
    try:
        df = pro.margin(trade_date=date, exchange_id="SSE")
        print(f"  {date}: {len(df)} rows")
    except Exception as e:
        print(f"  {date}: FAIL - {e}")

# 5. ETF份额
print("\n=== fund_share 历史深度 ===")
for code in ["510300.SH","560010.SH"]:
    for year in [2010, 2015, 2018, 2020, 2024, 2026]:
        try:
            df = pro.fund_share(ts_code=code, start_date=f"{year}0101", end_date=f"{year}0201")
            print(f"  {code} {year}年: {len(df)}条")
        except Exception as e:
            print(f"  {code} {year}年: FAIL - {e}")
    print()

# 6. 期货持仓排名
print("=== fut_holding 历史深度 ===")
for date in ["20100104","20150105","20180102","20200102","20240102","20260327"]:
    try:
        df = pro.fut_holding(trade_date=date, exchange="CFFEX")
        print(f"  {date}: {len(df)} rows")
    except Exception as e:
        print(f"  {date}: FAIL - {e}")

# 7. 沪深成交额（index_daily已知可用，确认最早日期）
print("\n=== index_daily 历史深度 ===")
for date in ["20100104","20150105","20200102","20260327"]:
    try:
        sh = pro.index_daily(ts_code="000001.SH", trade_date=date)
        sz = pro.index_daily(ts_code="399001.SZ", trade_date=date)
        print(f"  {date}: SH={len(sh)}条 SZ={len(sz)}条")
    except Exception as e:
        print(f"  {date}: FAIL - {e}")

print("\n=== Done ===")