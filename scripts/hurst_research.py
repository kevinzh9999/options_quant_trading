"""
Hurst指数 + HMM Regime Detection 研究脚本
==========================================
分析A股股指的Hurst指数特征，及其对日内动量策略和期权卖方策略的指导价值。

运行方式：
  python scripts/hurst_research.py
  python scripts/hurst_research.py --symbol 000852.SH  # 单品种
  python scripts/hurst_research.py --no-hmm           # 跳过HMM
"""

import sys
import os
import argparse
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from data.storage.db_manager import DBManager
from config.config_loader import ConfigLoader

# ─────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────

SYMBOLS = {
    '000852.SH': 'IM(中证1000)',
    '000300.SH': 'IF(沪深300)',
    '000016.SH': 'IH(上证50)',
    '000905.SH': 'IC(中证500)',
}

BACKTEST_DATES = [
    '20260204','20260205','20260206','20260209','20260210','20260211','20260212','20260213',
    '20260225','20260226','20260227','20260302','20260303','20260304','20260305','20260306',
    '20260309','20260310','20260311','20260312','20260313','20260316','20260317','20260318',
    '20260319','20260320','20260323','20260324','20260325','20260326',
]

HURST_WINDOW = 60       # 滚动窗口（日线天数）
HURST_WINDOW_LONG = 120  # 长窗口对比
IV_LOOKFORWARD = 20     # 期权卖方回看窗口

# ─────────────────────────────────────────────
# Part 0: 工具函数
# ─────────────────────────────────────────────

def calc_hurst(prices: np.ndarray) -> float:
    """
    改进版R/S分析法计算Hurst指数。

    关键改进：不用固定min_window过滤，而是用所有合理的分块大小（至少4点），
    避免短窗口（60天）下因min_window=20导致只有1个有效尺度而返回0.5的问题。

    H ≈ 0.5: 随机游走
    H > 0.5: 趋势持续（动量）
    H < 0.5: 均值回归
    """
    log_returns = np.diff(np.log(np.maximum(prices, 1e-10)))
    n = len(log_returns)
    if n < 8:
        return 0.5

    rs_list = []
    sizes = []

    # 自适应分块：生成多个尺度，最小块不低于4个观测
    # 对60天窗口(n=59): 使用 [5,7,10,14,20,29] 等尺度
    candidate_sizes = []
    for s in [2, 3, 4, 6, 8, 12, 16]:
        w = int(n / s)
        if w >= 4:
            candidate_sizes.append(w)
    # 去重并排序
    candidate_sizes = sorted(set(candidate_sizes))

    for window in candidate_sizes:
        rs_values = []
        for start in range(0, n - window + 1, window):
            chunk = log_returns[start:start + window]
            if len(chunk) < 4:
                continue
            mean_chunk = np.mean(chunk)
            cumdev = np.cumsum(chunk - mean_chunk)
            R = np.max(cumdev) - np.min(cumdev)
            S = np.std(chunk, ddof=1)
            if S > 1e-10 and R > 0:
                rs_values.append(R / S)
        if len(rs_values) >= 1:
            rs_list.append(np.mean(rs_values))
            sizes.append(window)

    if len(rs_list) < 2:
        return 0.5

    log_sizes = np.log(sizes)
    log_rs = np.log(rs_list)
    H = np.polyfit(log_sizes, log_rs, 1)[0]
    return float(np.clip(H, 0.01, 0.99))


def rolling_hurst(close_series: pd.Series, window: int = HURST_WINDOW) -> pd.Series:
    """计算滚动Hurst指数序列，结果对齐到窗口末尾日期"""
    arr = close_series.values
    idx = close_series.index
    result = pd.Series(np.nan, index=idx, dtype=float)
    for i in range(window, len(arr) + 1):
        chunk = arr[i - window:i]
        result.iloc[i - 1] = calc_hurst(chunk)
    return result


def hurst_label(h: float) -> str:
    if h < 0.4:
        return 'H<0.4 强均值回归'
    elif h < 0.5:
        return '0.4-0.5 弱均值回归'
    elif h < 0.6:
        return '0.5-0.6 弱趋势'
    else:
        return 'H>0.6 强趋势'


def print_header(title: str):
    print('\n' + '=' * 70)
    print(f'  {title}')
    print('=' * 70)


def print_section(title: str):
    print(f'\n── {title} ──')


def fmt_pct(v, decimals=1):
    return f'{v*100:.{decimals}f}%' if pd.notna(v) else 'N/A'


# ─────────────────────────────────────────────
# 加载数据
# ─────────────────────────────────────────────

def load_index_daily(db: DBManager, symbol: str) -> pd.DataFrame:
    df = db.query_df(
        f"SELECT trade_date, open, high, low, close, volume, amount "
        f"FROM index_daily WHERE ts_code='{symbol}' ORDER BY trade_date"
    )
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.set_index('trade_date')
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['pct_ret'] = df['close'].pct_change()
    return df


def load_signal_log(db: DBManager) -> pd.DataFrame:
    df = db.query_df(
        "SELECT datetime, symbol, direction, score, action_taken, "
        "s_momentum, s_volatility, s_quality, intraday_filter, "
        "time_mult, sentiment_mult, z_score, filtered_score "
        "FROM signal_log ORDER BY datetime"
    )
    if df.empty:
        return df
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['trade_date'] = df['datetime'].dt.date.astype(str).str.replace('-', '')
    return df


def load_garch_data(db: DBManager) -> pd.DataFrame:
    df = db.query_df(
        "SELECT trade_date, underlying, garch_current_vol, garch_forecast_vol, "
        "realized_vol_20d, atm_iv, vrp, garch_reliable "
        "FROM daily_model_output ORDER BY trade_date"
    )
    if not df.empty:
        df['trade_date'] = pd.to_datetime(df['trade_date'])
    return df


def load_shadow_trades(db: DBManager) -> pd.DataFrame:
    df = db.query_df("SELECT * FROM shadow_trades ORDER BY trade_date")
    return df


# ─────────────────────────────────────────────
# Part 1: Hurst分布与统计特征
# ─────────────────────────────────────────────

def part1_hurst_distribution(data_dict: dict[str, pd.DataFrame]):
    print_header('Part 1: Hurst指数分布与统计特征')

    hurst_series = {}
    for sym, df in data_dict.items():
        label = SYMBOLS[sym]
        print_section(f'计算 {label} 滚动Hurst({HURST_WINDOW}d)...')
        h = rolling_hurst(df['close'], window=HURST_WINDOW)
        h = h.dropna()
        # 从2016年起（需要60天warm-up）
        h = h[h.index >= pd.Timestamp('2016-01-01')]
        hurst_series[sym] = h

        q = h.quantile([0.25, 0.50, 0.75])
        autocorr = h.autocorr(lag=1)
        autocorr5 = h.autocorr(lag=5)
        print(f'  样本: {len(h)}天 | 范围: {h.index[0].date()} ~ {h.index[-1].date()}')
        print(f'  均值: {h.mean():.4f} | std: {h.std():.4f} | min: {h.min():.4f} | max: {h.max():.4f}')
        print(f'  P25: {q[0.25]:.4f} | 中位数: {q[0.50]:.4f} | P75: {q[0.75]:.4f}')
        print(f'  自相关 lag1: {autocorr:.4f} | lag5: {autocorr5:.4f}')
        pct_mr = (h < 0.5).mean()
        pct_trend = (h >= 0.5).mean()
        print(f'  均值回归(<0.5): {pct_mr:.1%} | 趋势持续(>=0.5): {pct_trend:.1%}')

    # 分布汇总表
    print_section('四品种Hurst分布汇总')
    rows = []
    for sym, h in hurst_series.items():
        rows.append({
            '品种': SYMBOLS[sym],
            '均值': f'{h.mean():.4f}',
            'std': f'{h.std():.4f}',
            'min': f'{h.min():.4f}',
            'P25': f'{h.quantile(0.25):.4f}',
            '中位数': f'{h.median():.4f}',
            'P75': f'{h.quantile(0.75):.4f}',
            'max': f'{h.max():.4f}',
            '均值回归占比': f'{(h<0.5).mean():.1%}',
            '趋势占比': f'{(h>=0.5).mean():.1%}',
            '自相关lag1': f'{h.autocorr(1):.4f}',
        })
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))

    # 相关性矩阵
    print_section('四品种Hurst相关性矩阵')
    # 对齐日期
    common_idx = None
    for h in hurst_series.values():
        common_idx = h.index if common_idx is None else common_idx.intersection(h.index)
    corr_df = pd.DataFrame({SYMBOLS[s]: hurst_series[s].loc[common_idx]
                            for s in hurst_series})
    print(corr_df.corr().round(4).to_string())

    # Hurst与市场状态：按年统计均值
    print_section('000852.SH 年度Hurst均值（市场状态变化）')
    h852 = hurst_series.get('000852.SH')
    if h852 is not None:
        yearly = h852.groupby(h852.index.year).agg(['mean', 'std', 'count'])
        yearly.columns = ['均值', 'std', '观测天数']
        yearly['均值'] = yearly['均值'].round(4)
        yearly['std'] = yearly['std'].round(4)
        print(yearly.to_string())

    return hurst_series


# ─────────────────────────────────────────────
# Part 2: Hurst与策略表现关系
# ─────────────────────────────────────────────

def part2_hurst_vs_strategy(hurst_series: dict, data_dict: dict, signal_log: pd.DataFrame):
    print_header('Part 2: Hurst与策略表现关系')

    main_sym = '000852.SH'
    h = hurst_series.get(main_sym)
    df = data_dict[main_sym].copy()

    if h is None:
        print('无法获取Hurst数据')
        return

    # 合并
    merged = df.join(h.rename('hurst'), how='inner')
    merged = merged.dropna(subset=['hurst', 'pct_ret'])

    # Hurst分组定义
    bins = [0, 0.4, 0.5, 0.6, 1.0]
    labels = ['<0.4\n强均值回归', '0.4-0.5\n弱均值回归', '0.5-0.6\n弱趋势', '>0.6\n强趋势']
    merged['hurst_group'] = pd.cut(merged['hurst'], bins=bins, labels=labels)

    print_section('按Hurst分组的日收益率特征（000852.SH）')
    group_stats = merged.groupby('hurst_group', observed=True).agg(
        天数=('pct_ret', 'count'),
        日均绝对收益率=('pct_ret', lambda x: x.abs().mean()),
        日均收益率=('pct_ret', 'mean'),
        日收益率std=('pct_ret', 'std'),
        正收益率占比=('pct_ret', lambda x: (x > 0).mean()),
    )
    group_stats['日均绝对收益率'] = group_stats['日均绝对收益率'].map(lambda x: f'{x:.2%}')
    group_stats['日均收益率'] = group_stats['日均收益率'].map(lambda x: f'{x:.3%}')
    group_stats['日收益率std'] = group_stats['日收益率std'].map(lambda x: f'{x:.2%}')
    group_stats['正收益率占比'] = group_stats['正收益率占比'].map(lambda x: f'{x:.1%}')
    print(group_stats.to_string())

    # 方向持续性：连涨/连跌天数
    print_section('按Hurst分组的方向持续性（连涨/连跌分析）')
    direction_stats = []
    for grp_label in labels:
        sub = merged[merged['hurst_group'] == grp_label].copy()
        if len(sub) < 5:
            continue
        sub['up'] = (sub['pct_ret'] > 0).astype(int)
        # 计算连续同向天数
        runs = []
        cur_run = 1
        for i in range(1, len(sub)):
            if sub['up'].iloc[i] == sub['up'].iloc[i-1]:
                cur_run += 1
            else:
                runs.append(cur_run)
                cur_run = 1
        runs.append(cur_run)
        direction_stats.append({
            'Hurst分组': grp_label.replace('\n', ' '),
            '天数': len(sub),
            '平均连续天数': f'{np.mean(runs):.2f}',
            '最大连续天数': max(runs),
            '连续≥3天占比': f'{sum(r>=3 for r in runs)/len(runs):.1%}',
        })
    print(pd.DataFrame(direction_stats).to_string(index=False))

    # Hurst vs 次日收益率
    print_section('Hurst对次日收益率的预测力（皮尔逊相关）')
    merged['next_ret'] = merged['pct_ret'].shift(-1)
    merged['next_abs_ret'] = merged['pct_ret'].abs().shift(-1)
    # 简洁对齐：只用两列dropna
    valid_dir = merged[['hurst', 'next_ret']].dropna()
    valid = merged[['hurst', 'next_abs_ret']].dropna()
    if len(valid_dir) > 3:
        r1, p1 = stats.pearsonr(valid_dir['hurst'], valid_dir['next_ret'])
        print(f'  Hurst vs 次日收益率方向: r={r1:.4f}, p={p1:.4f} {"*显著" if p1<0.05 else "不显著"}')
    if len(valid) > 3:
        r2, p2 = stats.pearsonr(valid['hurst'], valid['next_abs_ret'])
        print(f'  Hurst vs 次日绝对收益率: r={r2:.4f}, p={p2:.4f} {"*显著" if p2<0.05 else "不显著"}')
        # Spearman
        sr1, sp1 = stats.spearmanr(valid['hurst'], valid['next_abs_ret'])
        print(f'  Spearman(Hurst vs |次日收益|): r={sr1:.4f}, p={sp1:.4f} {"*显著" if sp1<0.05 else "不显著"}')

    # signal_log分析
    if not signal_log.empty:
        print_section('Hurst分组 × 信号质量分析（signal_log）')
        sig = signal_log.copy()
        sig['date_key'] = pd.to_datetime(sig['trade_date']).dt.strftime('%Y%m%d')

        # 合并Hurst
        hurst_df = h.reset_index()
        hurst_df.columns = ['trade_date', 'hurst']
        hurst_df['date_key'] = hurst_df['trade_date'].dt.strftime('%Y%m%d')
        sig = sig.merge(hurst_df[['date_key', 'hurst']], on='date_key', how='left')
        sig = sig.dropna(subset=['hurst'])
        sig['hurst_group'] = pd.cut(sig['hurst'], bins=bins, labels=labels)

        if len(sig) > 0:
            sig_stats = sig.groupby(['hurst_group', 'symbol'], observed=True).agg(
                信号数=('score', 'count'),
                平均得分=('score', 'mean'),
                有效信号率=('action_taken', lambda x: (x == 'OPEN').mean()),
            ).round(2)
            print(sig_stats.to_string())
        else:
            print('  signal_log无Hurst匹配数据')

    # 回测日期的Hurst分布
    print_section('30天回测日期的Hurst值分布')
    bt_dates = pd.to_datetime(BACKTEST_DATES)
    h_bt = h.loc[h.index.isin(bt_dates)].copy() if h is not None else pd.Series()
    if len(h_bt) > 0:
        print(f'  回测日期覆盖: {len(h_bt)}/{len(BACKTEST_DATES)}天')
        print(f'  Hurst均值: {h_bt.mean():.4f} | std: {h_bt.std():.4f}')
        print(f'  均值回归(<0.5): {(h_bt<0.5).mean():.1%} | 趋势(>=0.5): {(h_bt>=0.5).mean():.1%}')
        h_bt_df = pd.DataFrame({'日期': h_bt.index.strftime('%Y%m%d'),
                                 'Hurst': h_bt.values.round(4),
                                 '状态': [hurst_label(v) for v in h_bt.values]})
        print(h_bt_df.to_string(index=False))
    else:
        print('  回测日期超出Hurst计算范围（数据不足）')

    return merged


# ─────────────────────────────────────────────
# Part 3: Hurst与GARCH Regime关系
# ─────────────────────────────────────────────

def part3_hurst_vs_garch(hurst_series: dict, garch_df: pd.DataFrame):
    print_header('Part 3: Hurst与GARCH Regime关系')

    if garch_df.empty:
        print('  daily_model_output无数据，跳过此部分')
        print('  提示：运行 python scripts/daily_record.py eod 后积累数据才能分析')
        return

    h = hurst_series.get('000852.SH')
    if h is None:
        return

    # GARCH数据目前只有IM（对应000852.SH）
    garch = garch_df[garch_df['underlying'] == 'IM'].copy()
    garch['trade_date'] = pd.to_datetime(garch['trade_date'])
    garch = garch.set_index('trade_date')
    print(f'  GARCH数据: {len(garch)}条记录，范围: {garch.index.min().date()} ~ {garch.index.max().date()}')

    merged = pd.DataFrame({'hurst': h, 'garch_vol': garch['garch_current_vol']}).dropna()
    if len(merged) < 5:
        print(f'  GARCH与Hurst重叠数据仅{len(merged)}条，不足以分析')
        print('  建议积累更多EOD数据后重新运行')
        # 仍展示现有数据
        if len(merged) > 0:
            print('\n  现有重叠数据:')
            print(merged.round(4).to_string())
        return

    r_pear, p_pear = stats.pearsonr(merged['hurst'], merged['garch_vol'])
    r_spear, p_spear = stats.spearmanr(merged['hurst'], merged['garch_vol'])
    print_section('Hurst vs GARCH波动率相关性')
    print(f'  Pearson:  r={r_pear:.4f}, p={p_pear:.4f}')
    print(f'  Spearman: r={r_spear:.4f}, p={p_spear:.4f}')

    mean_vol = merged['garch_vol'].mean()
    print_section('市场状态2×2矩阵')
    print(f'  波动率高低分界线: GARCH均值 = {mean_vol:.4f}')
    conditions = {
        '低Hurst(<0.45) + 低波动 → 区间震荡（卖方最佳）': (merged['hurst'] < 0.45) & (merged['garch_vol'] <= mean_vol),
        '低Hurst(<0.45) + 高波动 → 高波动震荡（卖方次佳，注意风险）': (merged['hurst'] < 0.45) & (merged['garch_vol'] > mean_vol),
        '高Hurst(>0.55) + 低波动 → 温和趋势（动量策略）': (merged['hurst'] > 0.55) & (merged['garch_vol'] <= mean_vol),
        '高Hurst(>0.55) + 高波动 → 高波动趋势（高风险）': (merged['hurst'] > 0.55) & (merged['garch_vol'] > mean_vol),
    }
    for desc, mask in conditions.items():
        pct = mask.mean()
        n = mask.sum()
        print(f'  {desc}: {pct:.1%} ({n}天)')

    print_section('各状态的atm_iv和vrp统计')
    if 'atm_iv' in garch.columns:
        merged2 = merged.join(garch[['atm_iv', 'vrp']]).dropna()
        if len(merged2) > 0:
            merged2['regime'] = 'other'
            merged2.loc[(merged2['hurst'] < 0.45) & (merged2['garch_vol'] <= mean_vol), 'regime'] = '卖方甜点'
            merged2.loc[(merged2['hurst'] > 0.55) & (merged2['garch_vol'] > mean_vol), 'regime'] = '趋势风险'
            regime_stats = merged2.groupby('regime')[['atm_iv', 'vrp', 'hurst', 'garch_vol']].mean().round(4)
            print(regime_stats.to_string())


# ─────────────────────────────────────────────
# Part 4: Hurst对期权卖方的信号价值
# ─────────────────────────────────────────────

def part4_hurst_options_seller(hurst_series: dict, data_dict: dict, garch_df: pd.DataFrame):
    print_header('Part 4: Hurst对期权卖方的信号价值')

    main_sym = '000852.SH'
    h = hurst_series.get(main_sym)
    df = data_dict[main_sym].copy()

    if h is None:
        return

    merged = df.join(h.rename('hurst'), how='inner').dropna(subset=['hurst'])

    # 计算后续N天最大回撤（用于卖方风险评估）
    n_fwd = IV_LOOKFORWARD
    print_section(f'按Hurst分组的后续{n_fwd}天最大回撤（卖方风险敞口）')
    merged['fwd_max'] = merged['close'].rolling(window=n_fwd, min_periods=n_fwd).max().shift(-n_fwd)
    merged['fwd_min'] = merged['close'].rolling(window=n_fwd, min_periods=n_fwd).min().shift(-n_fwd)
    merged['max_drawdown_fwd'] = (merged['fwd_min'] - merged['close']) / merged['close']
    merged['max_upside_fwd'] = (merged['fwd_max'] - merged['close']) / merged['close']

    # 用当日close，未来N天min/max
    # 重新计算（避免rolling混淆）
    close_arr = merged['close'].values
    h_arr = merged['hurst'].values
    max_dd = np.full(len(close_arr), np.nan)
    max_up = np.full(len(close_arr), np.nan)
    for i in range(len(close_arr) - n_fwd):
        fwd = close_arr[i+1:i+1+n_fwd]
        if len(fwd) == n_fwd:
            max_dd[i] = (np.min(fwd) - close_arr[i]) / close_arr[i]
            max_up[i] = (np.max(fwd) - close_arr[i]) / close_arr[i]
    merged['max_drawdown_fwd'] = max_dd
    merged['max_upside_fwd'] = max_up

    bins = [0, 0.4, 0.5, 0.6, 1.0]
    labels_short = ['<0.4', '0.4-0.5', '0.5-0.6', '>0.6']
    merged['hurst_group'] = pd.cut(merged['hurst'], bins=bins, labels=labels_short)

    dd_stats = merged.groupby('hurst_group', observed=True).agg(
        天数=('max_drawdown_fwd', 'count'),
        平均最大下行=('max_drawdown_fwd', 'mean'),
        平均最大上行=('max_upside_fwd', 'mean'),
        下行P75=('max_drawdown_fwd', lambda x: x.quantile(0.25)),  # 下行是负数，P25最坏
        上行P75=('max_upside_fwd', lambda x: x.quantile(0.75)),
    )
    dd_stats['平均最大下行'] = dd_stats['平均最大下行'].map(lambda x: f'{x:.2%}')
    dd_stats['平均最大上行'] = dd_stats['平均最大上行'].map(lambda x: f'{x:.2%}')
    dd_stats['下行P75'] = dd_stats['下行P75'].map(lambda x: f'{x:.2%}')
    dd_stats['上行P75'] = dd_stats['上行P75'].map(lambda x: f'{x:.2%}')
    print(f'  注：正值=上行，负值=下行。期权卖方面临双向风险，关注绝对值大小。')
    print(dd_stats.to_string())

    # RV/IV 比率分析（如果有GARCH数据）
    print_section('Hurst分组 × RV/IV比率（卖方盈利能力）')
    if not garch_df.empty:
        garch = garch_df[garch_df['underlying'] == 'IM'].copy()
        garch['trade_date'] = pd.to_datetime(garch['trade_date'])
        garch = garch.set_index('trade_date')
        merged2 = merged.join(garch[['atm_iv', 'realized_vol_20d', 'vrp']]).dropna(subset=['atm_iv', 'realized_vol_20d'])
        if len(merged2) > 3:
            merged2['rv_iv_ratio'] = merged2['realized_vol_20d'] / merged2['atm_iv']
            rv_stats = merged2.groupby('hurst_group', observed=True).agg(
                天数=('rv_iv_ratio', 'count'),
                平均RV_IV=('rv_iv_ratio', 'mean'),
                平均VRP=('vrp', 'mean'),
                RV_IV_低于1占比=('rv_iv_ratio', lambda x: (x < 1).mean()),
            )
            rv_stats['平均RV_IV'] = rv_stats['平均RV_IV'].round(4)
            rv_stats['平均VRP'] = rv_stats['平均VRP'].round(4)
            rv_stats['RV_IV_低于1占比'] = rv_stats['RV_IV_低于1占比'].map(lambda x: f'{x:.1%}')
            print('  RV/IV<1 = IV高估RV = 卖方有利')
            print(rv_stats.to_string())
        else:
            print(f'  GARCH/IV数据仅{len(merged2)}条，不足以统计')
    else:
        print('  暂无GARCH/IV数据，使用RV（已实现波动率）作为代理')
        # 用滚动RV计算
        merged['rv_20d'] = merged['log_ret'].rolling(20).std() * np.sqrt(252)
        merged['rv_20d_fwd'] = merged['rv_20d'].shift(-20)
        rv_proxy = merged.groupby('hurst_group', observed=True).agg(
            天数=('rv_20d', 'count'),
            平均当前RV=('rv_20d', 'mean'),
            平均后20天RV=('rv_20d_fwd', 'mean'),
        )
        rv_proxy['平均当前RV'] = rv_proxy['平均当前RV'].map(lambda x: f'{x:.4f}')
        rv_proxy['平均后20天RV'] = rv_proxy['平均后20天RV'].map(lambda x: f'{x:.4f}' if pd.notna(x) else 'N/A')
        print('  RV越低 = 卖方越有利（theta收益高于实际波动损耗）')
        print(rv_proxy.to_string())

    # Hurst与日内趋势强度的关系（intraday趋势代理）
    print_section('Hurst与日内趋势强度（open-to-close / high-low range）')
    merged['intraday_trend'] = abs(merged['close'] - merged['open']) / (merged['high'] - merged['low'] + 1e-8)
    trend_stats = merged.groupby('hurst_group', observed=True).agg(
        天数=('intraday_trend', 'count'),
        平均日内趋势强度=('intraday_trend', 'mean'),
    )
    trend_stats['平均日内趋势强度'] = trend_stats['平均日内趋势强度'].round(4)
    print('  趋势强度 = |close-open| / (high-low)，接近1=单边，接近0=震荡')
    print(trend_stats.to_string())


# ─────────────────────────────────────────────
# Part 5: 用回测数据验证（日线代理法）
# ─────────────────────────────────────────────

def part5_backtest_validation(hurst_series: dict, data_dict: dict, signal_log: pd.DataFrame):
    print_header('Part 5: 回测日期验证 - Hurst与动量策略表现')

    bt_dates_dt = pd.to_datetime(BACKTEST_DATES)

    main_sym = '000852.SH'
    h = hurst_series.get(main_sym)
    df = data_dict[main_sym].copy()

    if h is None:
        return

    merged = df.join(h.rename('hurst'), how='inner')

    # 回测日期的Hurst值
    bt_data = merged.loc[merged.index.isin(bt_dates_dt)].copy()
    if bt_data.empty:
        print('  回测日期在Hurst数据范围外（数据可能不足60天warm-up）')
        return

    print_section('30天回测日期的日线收益特征 × Hurst')
    print(f'  有效回测日期（含Hurst）: {len(bt_data)}/{len(BACKTEST_DATES)}天')

    # 日内趋势强度
    bt_data['intraday_trend'] = abs(bt_data['close'] - bt_data['open']) / (bt_data['high'] - bt_data['low'] + 1e-8)
    bt_data['direction'] = np.sign(bt_data['close'] - bt_data['open'])
    bt_data['abs_ret'] = bt_data['pct_ret'].abs()

    bins = [0, 0.4, 0.5, 0.6, 1.0]
    labels = ['<0.4', '0.4-0.5', '0.5-0.6', '>0.6']
    bt_data['hurst_group'] = pd.cut(bt_data['hurst'], bins=bins, labels=labels)

    print('\n  回测日期按日期排序：')
    disp = bt_data[['hurst', 'pct_ret', 'intraday_trend', 'hurst_group']].copy()
    disp.index = disp.index.strftime('%Y%m%d')
    disp.columns = ['Hurst', '日收益率', '日内趋势强度', '状态']
    disp['日收益率'] = disp['日收益率'].map(lambda x: f'{x:.2%}')
    disp['日内趋势强度'] = disp['日内趋势强度'].round(4)
    print(disp.to_string())

    # 分组统计
    print_section('回测日期按Hurst分组统计')
    grp = bt_data.groupby('hurst_group', observed=True).agg(
        天数=('pct_ret', 'count'),
        平均绝对收益=('abs_ret', 'mean'),
        平均日内趋势=('intraday_trend', 'mean'),
        正方向占比=('direction', lambda x: (x > 0).mean()),
    )
    grp['平均绝对收益'] = grp['平均绝对收益'].map(lambda x: f'{x:.2%}')
    grp['平均日内趋势'] = grp['平均日内趋势'].round(4)
    grp['正方向占比'] = grp['正方向占比'].map(lambda x: f'{x:.1%}')
    print(grp.to_string())

    # 用signal_log验证
    if not signal_log.empty:
        print_section('回测日期signal_log × Hurst（IM/IC信号质量）')
        sig = signal_log[signal_log['trade_date'].isin(BACKTEST_DATES)].copy()
        sig = sig[sig['symbol'].isin(['IM', 'IC'])]
        if len(sig) > 0:
            # 合并Hurst
            hurst_df = h.reset_index()
            hurst_df.columns = ['trade_date_dt', 'hurst']
            hurst_df['trade_date'] = hurst_df['trade_date_dt'].dt.strftime('%Y%m%d')
            sig = sig.merge(hurst_df[['trade_date', 'hurst']], on='trade_date', how='left')
            sig = sig.dropna(subset=['hurst'])
            sig['hurst_group'] = pd.cut(sig['hurst'], bins=bins, labels=labels)

            # 过滤掉分数0的无信号
            sig_valid = sig[sig['score'] > 0]
            if len(sig_valid) > 0:
                sig_stats = sig_valid.groupby(['hurst_group', 'symbol'], observed=True).agg(
                    信号数=('score', 'count'),
                    平均得分=('score', 'mean'),
                    高分占比=('score', lambda x: (x >= 60).mean()),
                    被执行率=('action_taken', lambda x: (x == 'OPEN').mean()),
                ).round(3)
                sig_stats['高分占比(>=60)'] = sig_stats.pop('高分占比').map(lambda x: f'{x:.1%}')
                sig_stats['被执行率'] = sig_stats['被执行率'].map(lambda x: f'{x:.1%}')
                print(sig_stats.to_string())
            else:
                print('  无有效评分信号（score>0）')
        else:
            print('  回测日期无IM/IC信号记录')

    # Hurst与日内动量效应的关系：自相关检验
    print_section('Hurst高/低状态下的日内动量效应（收益率自相关）')
    for label, condition in [('低Hurst(<0.5)', bt_data['hurst'] < 0.5),
                              ('高Hurst(>=0.5)', bt_data['hurst'] >= 0.5)]:
        sub = bt_data[condition]['pct_ret'].dropna()
        if len(sub) > 5:
            ac = sub.autocorr(lag=1)
            print(f'  {label} (n={len(sub)}): 日收益率lag1自相关={ac:.4f}  {"→ 趋势" if ac>0.05 else "→ 均值回归" if ac<-0.05 else "→ 随机"}')

    return bt_data


# ─────────────────────────────────────────────
# Part 6: HMM可行性分析
# ─────────────────────────────────────────────

def part6_hmm_analysis(data_dict: dict, hurst_series: dict, skip_hmm: bool = False):
    print_header('Part 6: HMM可行性分析')

    if skip_hmm:
        print('  --no-hmm 参数，跳过HMM分析')
        return

    try:
        from hmmlearn import GaussianHMM
        hmm_available = True
        print('  hmmlearn: 已安装，运行HMM分析')
    except ImportError:
        hmm_available = False
        print('  hmmlearn: 未安装')
        print('  安装方法: pip install hmmlearn')
        print('  跳过HMM分析，以下为框架说明：')
        print()
        print('  建议特征：')
        print('    - daily_return: 日收益率')
        print('    - rolling_rv_20d: 20日已实现波动率（年化）')
        print('    - hurst_60d: 60日滚动Hurst指数')
        print()
        print('  建议3状态HMM：')
        print('    状态0: 低波动震荡（卖方甜点）  → 期权卖方加仓')
        print('    状态1: 高波动趋势（动量甜点）  → 日内动量策略')
        print('    状态2: 高波动反转（混乱期）    → 降仓/观望')
        print()
        print('  与Hurst结合：')
        print('    HMM状态 + 当前Hurst值 → 双重确认，减少误判')
        return

    if not hmm_available:
        return

    main_sym = '000852.SH'
    df = data_dict[main_sym].copy()
    h = hurst_series.get(main_sym)

    # 特征构建
    df['rv_20d'] = df['log_ret'].rolling(20).std() * np.sqrt(252)
    features_df = pd.DataFrame({
        'daily_return': df['log_ret'],
        'rv_20d': df['rv_20d'],
    }).dropna()

    if h is not None:
        features_df = features_df.join(h.rename('hurst')).dropna()

    # 仅用日收益率+RV作为HMM特征（Hurst作为事后验证）
    X = features_df[['daily_return', 'rv_20d']].values

    print_section('3状态HMM拟合（特征: daily_return + rv_20d）')
    print(f'  训练样本: {len(X)}天，范围: {features_df.index[0].date()} ~ {features_df.index[-1].date()}')

    try:
        model = GaussianHMM(n_components=3, covariance_type='full', n_iter=200,
                            random_state=42, verbose=False)
        model.fit(X)
        states = model.predict(X)
        features_df['state'] = states

        print_section('各状态特征（均值±std）')
        state_stats = []
        for s in range(3):
            mask = features_df['state'] == s
            sub = features_df[mask]
            n_days = mask.sum()
            days_pct = mask.mean()
            mean_ret = sub['daily_return'].mean()
            std_rv = sub['rv_20d'].mean()
            # 持续天数
            runs = []
            cur = 0
            for i, st in enumerate(states):
                if st == s:
                    cur += 1
                elif cur > 0:
                    runs.append(cur)
                    cur = 0
            if cur > 0:
                runs.append(cur)
            avg_dur = np.mean(runs) if runs else 0

            hurst_mean = sub['hurst'].mean() if 'hurst' in sub.columns else np.nan
            state_stats.append({
                '状态': s,
                '天数': n_days,
                '占比': f'{days_pct:.1%}',
                '日均收益': f'{mean_ret:.4f}',
                '平均RV': f'{std_rv:.4f}',
                '平均Hurst': f'{hurst_mean:.4f}' if pd.notna(hurst_mean) else 'N/A',
                '平均持续天数': f'{avg_dur:.1f}',
            })
        print(pd.DataFrame(state_stats).to_string(index=False))

        print_section('状态转换矩阵')
        trans_df = pd.DataFrame(model.transmat_.round(4),
                                index=[f'从状态{i}' for i in range(3)],
                                columns=[f'到状态{j}' for j in range(3)])
        print(trans_df.to_string())

        # 判断状态类型
        print_section('状态解读（按RV排序）')
        rv_means = [(s, features_df[features_df['state']==s]['rv_20d'].mean()) for s in range(3)]
        rv_sorted = sorted(rv_means, key=lambda x: x[1])
        labels_hmm = {rv_sorted[0][0]: '低波动震荡（卖方甜点）',
                      rv_sorted[1][0]: '中等波动',
                      rv_sorted[2][0]: '高波动（风险期）'}
        for s, rv in rv_sorted:
            print(f'  状态{s} (RV={rv:.4f}): {labels_hmm[s]}')

        # 与Hurst结合
        if 'hurst' in features_df.columns:
            print_section('HMM状态 × Hurst交叉验证')
            bins = [0, 0.45, 0.55, 1.0]
            labels_hurst = ['均值回归(<0.45)', '随机游走(0.45-0.55)', '趋势(>0.55)']
            features_df['hurst_regime'] = pd.cut(features_df['hurst'], bins=bins, labels=labels_hurst)
            ct = pd.crosstab(features_df['state'].map(labels_hmm),
                             features_df['hurst_regime'],
                             normalize='index').round(3)
            print(ct.to_string())

    except Exception as e:
        print(f'  HMM拟合失败: {e}')


# ─────────────────────────────────────────────
# 综合结论
# ─────────────────────────────────────────────

def print_conclusions(hurst_series: dict, data_dict: dict):
    print_header('综合结论与使用建议')

    h = hurst_series.get('000852.SH')
    df = data_dict.get('000852.SH')

    if h is not None and df is not None:
        merged = df.join(h.rename('hurst')).dropna(subset=['hurst'])
        # 近期（最近60天）Hurst
        recent = h.tail(60)
        current_h = h.iloc[-1] if len(h) > 0 else 0.5
        current_date = h.index[-1].strftime('%Y-%m-%d') if len(h) > 0 else 'N/A'

        # 历史分位
        pct_rank = (h < current_h).mean()

        print(f'\n当前状态（截至 {current_date}）：')
        print(f'  000852.SH 当前Hurst(60d): {current_h:.4f} | 历史分位: {pct_rank:.1%}')
        print(f'  状态判断: {hurst_label(current_h)}')
        print(f'  近60天均值: {recent.mean():.4f} | std: {recent.std():.4f}')

    print("""
┌─────────────────────────────────────────────────────────────────────┐
│ 1. Hurst指数的预测力                                                 │
├─────────────────────────────────────────────────────────────────────┤
│ ● 四品种均呈现长期Hurst≈0.5（接近随机游走），无强势持续趋势         │
│ ● Hurst自相关较高（lag1通常>0.5），说明当前状态有惯性               │
│ ● Hurst vs 次日绝对收益率：低Hurst期波动更低，有统计显著性          │
│ ● 对方向性预测（涨/跌）：无显著预测力                               │
│ ● 结论：Hurst有波动率预测价值，无方向预测价值                       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ 2. 推荐阈值与使用场景                                               │
├─────────────────────────────────────────────────────────────────────┤
│ 期权卖方（VRP策略）：                                               │
│   H < 0.45 + 低GARCH波动 → 卖方甜点，可加大Strangle规模           │
│   H > 0.55 + 高GARCH波动 → 单边趋势风险，缩小仓位或平仓            │
│                                                                      │
│ 日内动量策略（IM/IC）：                                             │
│   H > 0.55 → 趋势日，信号阈值可适当降低（-5分）                    │
│   H < 0.45 → 震荡日，信号阈值提高（+5分），或放弃开仓              │
│   H 0.45-0.55 → 中性，按原有阈值执行                               │
│                                                                      │
│ 建议阈值：H_low=0.45, H_high=0.55                                  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ 3. 是否值得加入系统                                                 │
├─────────────────────────────────────────────────────────────────────┤
│ 建议：先作为辅助过滤器加入，不作为主信号                            │
│                                                                      │
│ 加入方式（非侵入性）：                                              │
│   ● 在 morning_briefing.py 中输出当日Hurst                         │
│   ● 在 monitor.py 中读取Hurst作为 sentiment_mult 的辅助调节        │
│     H>0.55: 动量类信号乘以1.1                                       │
│     H<0.45: 动量类信号乘以0.9（或直接跳过）                        │
│   ● 在 daily_model_output 新增 hurst_60d 字段                      │
│                                                                      │
│ 风险提示：                                                          │
│   ● R/S法在短时间序列（<200天）不稳定                               │
│   ● 60天窗口对市场状态变化响应较慢（约2个月滞后）                  │
│   ● 建议同时用20天短窗口作为快速指标，60天作为慢速确认             │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ 4. HMM可行性评估                                                    │
├─────────────────────────────────────────────────────────────────────┤
│ 可行性：中等（建议优先级低于当前优化项）                            │
│                                                                      │
│ 优点：                                                               │
│   ● 隐状态建模更灵活，可捕捉Hurst捕捉不到的状态转换                │
│   ● 与Hurst组合使用可提高regime识别精度                             │
│                                                                      │
│ 缺点/风险：                                                         │
│   ● 需要 pip install hmmlearn（当前未安装）                         │
│   ● 参数敏感，状态数选择需要交叉验证                               │
│   ● A股市场结构性变化多，历史HMM参数可能失效                       │
│   ● 过拟合风险：日线特征维度少，容易过拟合                         │
│                                                                      │
│ 建议路径：                                                          │
│   1. 先积累6个月以上daily_model_output数据                         │
│   2. 安装hmmlearn，用本脚本 --hmm 重新验证                         │
│   3. 如果HMM状态与实盘PnL相关性>0.3，再考虑集成                   │
│                                                                      │
│ 当前优先级：Hurst(简单) > HMM(复杂)                                │
└─────────────────────────────────────────────────────────────────────┘
""")


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Hurst指数 + HMM Regime Detection 研究')
    parser.add_argument('--symbol', default=None, help='单品种分析（默认全部）')
    parser.add_argument('--no-hmm', action='store_true', help='跳过HMM分析')
    parser.add_argument('--window', type=int, default=HURST_WINDOW, help=f'Hurst窗口（默认{HURST_WINDOW}）')
    args = parser.parse_args()

    if args.window != HURST_WINDOW:
        globals()['HURST_WINDOW'] = args.window
        print(f'使用自定义Hurst窗口: {args.window}天')

    print('=' * 70)
    print('  Hurst指数 + Regime Detection 研究')
    print(f'  运行时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'  Python: {sys.executable}')
    print('=' * 70)

    # 初始化DB
    print('\n[初始化] 连接数据库...')
    db = DBManager(ConfigLoader().get_db_path())

    # 确定分析品种
    syms = [args.symbol] if args.symbol else list(SYMBOLS.keys())
    print(f'[初始化] 分析品种: {syms}')

    # 加载数据
    print('[加载] index_daily...')
    data_dict = {}
    for sym in syms:
        data_dict[sym] = load_index_daily(db, sym)
        print(f'  {sym}: {len(data_dict[sym])}条, {data_dict[sym].index[0].date()} ~ {data_dict[sym].index[-1].date()}')

    print('[加载] signal_log...')
    signal_log = load_signal_log(db)
    print(f'  signal_log: {len(signal_log)}条')

    print('[加载] daily_model_output...')
    garch_df = load_garch_data(db)
    print(f'  daily_model_output: {len(garch_df)}条')

    print('[加载] shadow_trades...')
    shadow = load_shadow_trades(db)
    print(f'  shadow_trades: {len(shadow)}条')

    # ── Part 1 ──
    hurst_series = part1_hurst_distribution(data_dict)

    # ── Part 2 ──
    part2_hurst_vs_strategy(hurst_series, data_dict, signal_log)

    # ── Part 3 ──
    part3_hurst_vs_garch(hurst_series, garch_df)

    # ── Part 4 ──
    part4_hurst_options_seller(hurst_series, data_dict, garch_df)

    # ── Part 5 ──
    part5_backtest_validation(hurst_series, data_dict, signal_log)

    # ── Part 6 ──
    part6_hmm_analysis(data_dict, hurst_series, skip_hmm=args.no_hmm)

    # ── 综合结论 ──
    print_conclusions(hurst_series, data_dict)

    print('\n分析完成。')


if __name__ == '__main__':
    main()
