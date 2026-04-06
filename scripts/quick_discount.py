"""
IM 股指期货历史贴水率分析
直接在项目根目录运行: python scripts/quick_discount.py

现货基准：000852.SH（中证1000现货指数）
另外单独显示 IM.CFX 主力期货相对现货的贴水（当月贴水）
"""
import sqlite3
import sys
import os
import pandas as pd
import numpy as np

# 找数据库
db_candidates = [
    'data/storage/trading.db',
    os.path.expanduser('~/options_quant_trading/data/storage/trading.db'),
]
db_path = None
for p in db_candidates:
    if os.path.exists(p):
        db_path = p
        break

if not db_path:
    try:
        sys.path.insert(0, os.getcwd())
        from config.config_loader import ConfigLoader
        db_path = ConfigLoader().get_db_path()
    except Exception:
        pass

if not db_path:
    print("找不到数据库文件，请在项目根目录运行此脚本")
    sys.exit(1)

print(f"数据库: {db_path}\n")
conn = sqlite3.connect(db_path, timeout=30)
conn.execute("PRAGMA journal_mode=WAL")
conn.execute("PRAGMA busy_timeout=30000")

# ── 1. 现货：000852.SH（中证1000） ─────────────────────────────────────────
spot_df = None
try:
    spot_df = pd.read_sql(
        "SELECT trade_date, close as spot_close FROM index_daily "
        "WHERE ts_code='000852.SH' ORDER BY trade_date",
        conn,
    )
except Exception:
    pass

if spot_df is None or spot_df.empty:
    print("⚠  index_daily 中无 000852.SH 数据，回退到 IM.CFX 作为现货近似\n"
          "   提示：运行 python data/download_scripts/download_index_daily.py --start 20150101 下载现货数据\n")
    spot_df = pd.read_sql(
        "SELECT trade_date, close as spot_close FROM futures_daily "
        "WHERE ts_code='IM.CFX' ORDER BY trade_date",
        conn,
    )
    spot_source = "IM.CFX（近似）"
else:
    spot_source = "000852.SH（中证1000现货）"

# ── 2. 期货：IM.CFX（主力）+ IML1/2/3 ────────────────────────────────────
fut_df = pd.read_sql(
    "SELECT trade_date, ts_code, close "
    "FROM futures_daily "
    "WHERE ts_code IN ('IM.CFX','IML.CFX','IML1.CFX','IML2.CFX','IML3.CFX') "
    "ORDER BY trade_date",
    conn,
)
conn.close()

if fut_df.empty:
    print("无 IM 期货数据")
    sys.exit(1)

# 透视期货
pivot_fut = fut_df.pivot(index='trade_date', columns='ts_code', values='close')

# 合并现货
spot_df = spot_df.set_index('trade_date')
df = pivot_fut.join(spot_df, how='inner').dropna(subset=['spot_close'])

if df.empty:
    print("现货和期货数据无交集（日期不重叠）")
    sys.exit(1)

print(f"现货基准  : {spot_source}")
print(f"数据范围  : {df.index[0]} ~ {df.index[-1]}，共 {len(df)} 个交易日\n")

# ── 3. 当月贴水（IM.CFX vs 现货）────────────────────────────────────────
main_col = 'IM.CFX' if 'IM.CFX' in df.columns else 'IML.CFX'
if main_col in df.columns:
    main_discount_abs = df[main_col] - df['spot_close']
    main_discount_pct = main_discount_abs / df['spot_close']
    cur_abs  = main_discount_abs.iloc[-1]
    cur_pct  = main_discount_pct.iloc[-1]
    cur_spot = df['spot_close'].iloc[-1]
    cur_fut  = df[main_col].iloc[-1]
    print(f"{'='*60}")
    print(f"  IM.CFX 主力合约 vs 现货（{df.index[-1]}）")
    print(f"{'='*60}")
    print(f"  现货（{spot_source}）: {cur_spot:.2f}")
    print(f"  IM.CFX 主力价格      : {cur_fut:.2f}")
    print(f"  绝对贴水             : {cur_abs:+.2f} 点")
    print(f"  原始贴水率           : {cur_pct*100:+.3f}%")
    print()

# ── 4. 各连续合约历史贴水统计 ────────────────────────────────────────────
# IML1=次月(~22交易日), IML2=当季(~66交易日), IML3=隔季(~132交易日)
contracts = {
    'IML1.CFX': {'name': '次月 (IML1)', 'est_days': 22},
    'IML2.CFX': {'name': '当季 (IML2)', 'est_days': 66},
    'IML3.CFX': {'name': '隔季 (IML3)', 'est_days': 132},
}

results = {}
for code, info in contracts.items():
    if code not in df.columns:
        continue

    sub = df[[code, 'spot_close']].dropna()
    if sub.empty:
        continue

    discount_abs = sub[code] - sub['spot_close']
    discount_raw = discount_abs / sub['spot_close']   # 负值 = 贴水
    annualized   = discount_raw * (252 / info['est_days'])

    current_ann = annualized.iloc[-1]
    # 百分位：贴水越深（越负）= 越低百分位；信号角度用绝对值
    pct = (annualized > current_ann).mean() * 100  # 高百分位 = 贴水深

    results[code] = {
        'name':            info['name'],
        'annualized':      annualized,
        'discount_raw':    discount_raw,
    }

    print(f"{'='*60}")
    print(f"  {info['name']}  ({code})")
    print(f"{'='*60}")
    print(f"  均值          : {annualized.mean()*100:>8.2f}%")
    print(f"  中位数        : {annualized.median()*100:>8.2f}%")
    print(f"  25百分位(浅)  : {annualized.quantile(0.25)*100:>8.2f}%")
    print(f"  75百分位(深)  : {annualized.quantile(0.75)*100:>8.2f}%")
    print(f"  最小值(最浅)  : {annualized.min()*100:>8.2f}%  ({annualized.idxmin()})")
    print(f"  最大值(最深)  : {annualized.max()*100:>8.2f}%  ({annualized.idxmax()})")
    print(f"  {'─'*40}")
    print(f"  当前年化贴水  : {current_ann*100:>8.2f}%  （历史深度 {pct:.0f} 百分位）")
    print(f"  当前期货价    : {sub[code].iloc[-1]:.2f}")
    print(f"  当前现货价    : {sub['spot_close'].iloc[-1]:.2f}")
    print(f"  绝对贴水      : {discount_abs.iloc[-1]:+.2f} 点")
    print()

# ── 5. 汇总信号 ────────────────────────────────────────────────────────
print(f"{'='*60}")
print(f"  当前贴水率总览  ({df.index[-1]})")
print(f"{'='*60}")
for code, info in results.items():
    ann     = info['annualized'].iloc[-1]
    pct     = (info['annualized'] > ann).mean() * 100
    # 贴水为负值，取负后判断强度
    mag     = -ann  # 正值 = 贴水幅度
    signal  = "强" if mag > 0.15 else "中" if mag > 0.10 else "弱" if mag > 0.05 else "无"
    print(f"  {info['name']:14s}: 年化 {ann*100:>7.2f}%  |  深度 {pct:>5.1f} 百分位  |  信号: {signal}")

print()
print(f"现货基准: {spot_source}")
print("贴水率为负值，绝对值越大 = 贴水越深。深度百分位越高 = 当前贴水越深（历史少见）")
print("信号: 年化贴水>15%=强, 10-15%=中, 5-10%=弱, <5%=无")
print()
print("注: 若要使用准确现货价，请先运行:")
print("  python data/download_scripts/download_index_daily.py --start 20150101")
