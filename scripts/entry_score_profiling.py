#!/usr/bin/env python3
"""
Entry Score Profiling — 验证entry_score是否正向预测行情延伸空间。

计算固定窗口和实际窗口两套MFE/MAE指标，按score bin × regime × direction分组分析。
自动检测climax假设并输出verdict。
"""
import sys, os, argparse, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTHONUNBUFFERED"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import time

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

SPOT_MAP = {"IM": "000852", "IC": "000905", "IF": "000300", "IH": "000016"}

def load_trades(db_path, symbol, start_date=None, end_date=None):
    """从backtest跑出交易数据（不依赖shadow_trades，用回测保证一致性）。"""
    from data.storage.db_manager import get_db
    from scripts.backtest_signals_day import run_day
    db = get_db()

    dates_df = db.query_df(
        f"SELECT DISTINCT substr(datetime,1,10) as d FROM index_min "
        f"WHERE symbol='{SPOT_MAP[symbol]}' AND period=300 ORDER BY d"
    )
    all_dates = [d.replace('-', '') for d in dates_df['d'].tolist()]

    if start_date:
        all_dates = [d for d in all_dates if d >= start_date]
    if end_date:
        all_dates = [d for d in all_dates if d <= end_date]

    trades = []
    for td in all_dates:
        day_trades = run_day(symbol, td, db, verbose=False)
        full = [t for t in day_trades if not t.get("partial")]
        for t in full:
            t["trade_date"] = td
            t["symbol"] = symbol
        trades.extend(full)

    df = pd.DataFrame(trades)
    if len(df) == 0:
        return df

    # 标准化列名
    df = df.rename(columns={
        "entry_score": "entry_score",
        "pnl_pts": "pnl_pts",
    })
    return df


def load_ohlcv(db_path, symbol, start_date=None, end_date=None):
    """加载现货5分钟K线。"""
    from data.storage.db_manager import get_db
    db = get_db()
    spot = SPOT_MAP[symbol]
    sql = (f"SELECT datetime, open, high, low, close, volume FROM index_min "
           f"WHERE symbol='{spot}' AND period=300 ORDER BY datetime")
    df = db.query_df(sql)
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = df[c].astype(float)
    # datetime是UTC，转为BJ用于跟entry_time对比
    # entry_time已经是BJ格式（bar结束时间+8h）
    df['bj_time'] = pd.to_datetime(df['datetime']) + pd.Timedelta(hours=8)
    df = df.set_index('bj_time')
    df['trade_date'] = df.index.strftime('%Y%m%d')

    if start_date:
        df = df[df['trade_date'] >= start_date]
    if end_date:
        df = df[df['trade_date'] <= end_date]
    return df


# ---------------------------------------------------------------------------
# Metric calculation
# ---------------------------------------------------------------------------

def calc_fixed_window_metrics(entry_price, direction, ohlcv_window):
    """固定窗口MFE/MAE。ohlcv_window是entry之后N根bar。"""
    if len(ohlcv_window) == 0:
        return {'mfe': 0, 'mae': 0, 'window_truncated': True}

    highs = ohlcv_window['high'].values
    lows = ohlcv_window['low'].values

    if direction == "LONG":
        mfe = float(np.max(highs) - entry_price)
        mae = float(entry_price - np.min(lows))
    else:
        mfe = float(entry_price - np.min(lows))
        mae = float(np.max(highs) - entry_price)

    return {'mfe': max(0, mfe), 'mae': max(0, mae), 'window_truncated': False}


def calc_actual_window_metrics(entry_price, exit_price, direction, ohlcv_window):
    """实际持仓窗口MFE/MAE。"""
    if len(ohlcv_window) == 0:
        # same-bar
        pnl = (exit_price - entry_price) if direction == "LONG" else (entry_price - exit_price)
        return {'mfe': max(0, pnl), 'mae': max(0, -pnl), 'holding_bars': 1, 'same_bar': True}

    highs = ohlcv_window['high'].values
    lows = ohlcv_window['low'].values

    if direction == "LONG":
        mfe = float(np.max(highs) - entry_price)
        mae = float(entry_price - np.min(lows))
    else:
        mfe = float(entry_price - np.min(lows))
        mae = float(np.max(highs) - entry_price)

    return {'mfe': max(0, mfe), 'mae': max(0, mae),
            'holding_bars': len(ohlcv_window), 'same_bar': False}


def classify_regime(trade_date, ohlcv_df, amp_threshold=0.004):
    """开盘30分钟振幅regime分类。"""
    day_bars = ohlcv_df[ohlcv_df['trade_date'] == trade_date]
    if len(day_bars) < 6:
        return 'unknown'
    open_bars = day_bars.iloc[:6]  # 前6根5min = 30分钟
    amp = (open_bars['high'].max() - open_bars['low'].min()) / open_bars['open'].iloc[0]
    return 'high_amp' if amp >= amp_threshold else 'low_amp'


def find_bar_index(ohlcv_df, trade_date, entry_time_bj):
    """找entry_time对应的bar index。entry_time是BJ HH:MM格式。"""
    day_bars = ohlcv_df[ohlcv_df['trade_date'] == trade_date]
    if len(day_bars) == 0:
        return None, day_bars

    # entry_time是bar结束时间（如09:50表示09:45bar完成后的entry）
    # 对应的bar是close时间<=entry_time的最后一根
    entry_hm = entry_time_bj
    for i in range(len(day_bars)):
        bar_bj = day_bars.index[i].strftime('%H:%M')
        # bar的BJ时间+5min = bar结束时间
        bar_h, bar_m = int(bar_bj[:2]), int(bar_bj[3:5])
        bar_m += 5
        if bar_m >= 60:
            bar_h += 1; bar_m -= 60
        bar_end = f"{bar_h:02d}:{bar_m:02d}"
        if bar_end == entry_hm:
            return i, day_bars
    return None, day_bars


def enrich_trades(trades_df, ohlcv_df, n_bars_list=[12, 24, 48], amp_threshold=0.004):
    """给每笔交易加上MFE/MAE/regime等列。"""
    records = []
    skipped = 0

    for _, trade in trades_df.iterrows():
        td = trade['trade_date']
        entry_time = trade['entry_time']
        exit_time = trade.get('exit_time', '')
        entry_price = trade['entry_price']
        exit_price = trade.get('exit_price', entry_price)
        direction = trade['direction']
        score = trade.get('entry_score', 0)

        bar_idx, day_bars = find_bar_index(ohlcv_df, td, entry_time)
        if bar_idx is None:
            skipped += 1
            continue

        rec = {
            'trade_date': td,
            'symbol': trade['symbol'],
            'direction': direction,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'entry_score': score,
            'pnl_pts': trade.get('pnl_pts', 0),
            'exit_reason': trade.get('reason', ''),
        }

        # Regime
        rec['regime'] = classify_regime(td, ohlcv_df, amp_threshold)

        # Fixed-window MFE/MAE for each N
        for n in n_bars_list:
            window = day_bars.iloc[bar_idx + 1: bar_idx + 1 + n]
            # 截断到当天收盘（14:55 BJ）
            window = window[window.index.time <= time(14, 55)]
            fm = calc_fixed_window_metrics(entry_price, direction, window)
            rec[f'mfe_fixed_{n}'] = fm['mfe']
            rec[f'mae_fixed_{n}'] = fm['mae']
            rec[f'truncated_{n}'] = fm['window_truncated'] or len(window) < n

        # Actual-window MFE/MAE
        exit_idx = None
        if exit_time:
            exit_idx_result, _ = find_bar_index(ohlcv_df, td, exit_time)
            if exit_idx_result is not None:
                exit_idx = exit_idx_result

        if exit_idx is not None and exit_idx > bar_idx:
            actual_window = day_bars.iloc[bar_idx + 1: exit_idx + 1]
            am = calc_actual_window_metrics(entry_price, exit_price, direction, actual_window)
        else:
            am = calc_actual_window_metrics(entry_price, exit_price, direction, pd.DataFrame())
        rec['mfe_actual'] = am['mfe']
        rec['mae_actual'] = am['mae']
        rec['holding_bars'] = am['holding_bars']
        rec['same_bar'] = am['same_bar']

        # Actual win
        rec['actual_win'] = 1 if trade.get('pnl_pts', 0) > 0 else 0

        # Fixed win (1.5R target)
        sl_pct = 0.003 if trade['symbol'] == 'IM' else 0.005
        target = entry_price * sl_pct * 1.5
        rec['fixed_win_24'] = 1 if rec.get('mfe_fixed_24', 0) >= target else 0

        records.append(rec)

    if skipped:
        print(f"  [WARN] {skipped} 笔交易未找到对应bar，已跳过")

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

SCORE_BINS = [50, 60, 65, 75, 85, 100]
SCORE_LABELS = ['[50,60)', '[60,65)', '[65,75)', '[75,85)', '[85,100]']


def auto_adjust_bins(df, min_samples=20):
    """如果最高档样本不足则合并。"""
    bins, labels = SCORE_BINS.copy(), SCORE_LABELS.copy()
    df['score_bin'] = pd.cut(df['entry_score'], bins=bins, labels=labels, right=False, include_lowest=True)

    # 检查最高档
    top_count = (df['score_bin'] == labels[-1]).sum()
    if top_count < min_samples and len(labels) > 3:
        # 合并最后两档
        bins = bins[:-1]
        labels = labels[:-1]
        labels[-1] = labels[-1].split(',')[0] + ',100]'
        df['score_bin'] = pd.cut(df['entry_score'], bins=bins, labels=labels, right=False, include_lowest=True)
        print(f"  [INFO] 最高档样本不足({top_count}<{min_samples})，合并为{labels[-1]}")

    return df, bins, labels


def aggregate_stats(df, group_col='score_bin'):
    """按分组聚合统计。"""
    agg = df.groupby(group_col).agg(
        trade_count=('entry_score', 'count'),
        mean_mfe_fixed_24=('mfe_fixed_24', 'mean'),
        median_mfe_fixed_24=('mfe_fixed_24', 'median'),
        p75_mfe_fixed_24=('mfe_fixed_24', lambda x: np.percentile(x, 75)),
        mean_mae_fixed_24=('mae_fixed_24', 'mean'),
        median_mae_fixed_24=('mae_fixed_24', 'median'),
        mean_mfe_actual=('mfe_actual', 'mean'),
        median_mfe_actual=('mfe_actual', 'median'),
        mean_mae_actual=('mae_actual', 'mean'),
        mean_holding_bars=('holding_bars', 'mean'),
        median_holding_bars=('holding_bars', 'median'),
        actual_win_rate=('actual_win', 'mean'),
        fixed_win_rate_24=('fixed_win_24', 'mean'),
    ).reset_index()

    agg['mfe_mae_ratio'] = agg['median_mfe_fixed_24'] / agg['median_mae_fixed_24'].replace(0, np.nan)
    return agg


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_overall_2x2(df, stats, output_path):
    """总体2x2图。"""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Entry Score Profiling — Overall', fontsize=14)

    # 左上: Fixed MFE boxplot
    df.boxplot(column='mfe_fixed_24', by='score_bin', ax=axes[0, 0])
    axes[0, 0].set_title('Fixed-window MFE (24 bars)')
    axes[0, 0].set_xlabel('Score Bin')
    axes[0, 0].set_ylabel('MFE (points)')

    # 右上: Fixed MAE boxplot
    df.boxplot(column='mae_fixed_24', by='score_bin', ax=axes[0, 1])
    axes[0, 1].set_title('Fixed-window MAE (24 bars)')
    axes[0, 1].set_xlabel('Score Bin')
    axes[0, 1].set_ylabel('MAE (points)')

    # 左下: Holding bars
    axes[1, 0].bar(stats['score_bin'].astype(str), stats['mean_holding_bars'])
    axes[1, 0].set_title('Mean Holding Bars')
    axes[1, 0].set_xlabel('Score Bin')
    axes[1, 0].set_ylabel('Bars')

    # 右下: MFE vs MAE scatter
    colors = {'[50,60)': 'gray', '[60,65)': 'blue', '[65,75)': 'green',
              '[75,85)': 'orange', '[85,100]': 'red'}
    for bin_label in df['score_bin'].unique():
        sub = df[df['score_bin'] == bin_label]
        c = colors.get(str(bin_label), 'black')
        axes[1, 1].scatter(sub['mfe_fixed_24'], sub['mae_fixed_24'],
                           alpha=0.3, s=15, c=c, label=str(bin_label))
    axes[1, 1].set_xlabel('MFE (points)')
    axes[1, 1].set_ylabel('MAE (points)')
    axes[1, 1].set_title('MFE vs MAE by Score Bin')
    axes[1, 1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  图1: {output_path}")


def plot_by_regime(df, output_path):
    """Regime拆分对比图。"""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Entry Score Profiling — By Regime', fontsize=14)

    for col, (ax, title) in enumerate(zip(
            [axes[0, 0], axes[0, 1]], ['High Amplitude', 'Low Amplitude'])):
        regime = 'high_amp' if col == 0 else 'low_amp'
        sub = df[df['regime'] == regime]
        if len(sub) > 0:
            sub.boxplot(column='mfe_fixed_24', by='score_bin', ax=ax)
        ax.set_title(f'{title} — Fixed MFE (24 bars)')

    for col, (ax, title) in enumerate(zip(
            [axes[1, 0], axes[1, 1]], ['High Amplitude', 'Low Amplitude'])):
        regime = 'high_amp' if col == 0 else 'low_amp'
        sub = df[df['regime'] == regime]
        if len(sub) > 0:
            stats = aggregate_stats(sub)
            ax.bar(stats['score_bin'].astype(str), stats['actual_win_rate'] * 100)
        ax.set_title(f'{title} — Win Rate (%)')
        ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  图2: {output_path}")


def plot_by_direction(df, output_path):
    """方向拆分对比图。"""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Entry Score Profiling — By Direction', fontsize=14)

    for col, (ax, d) in enumerate(zip([axes[0, 0], axes[0, 1]], ['LONG', 'SHORT'])):
        sub = df[df['direction'] == d]
        if len(sub) > 0:
            sub.boxplot(column='mfe_fixed_24', by='score_bin', ax=ax)
        ax.set_title(f'{d} — Fixed MFE (24 bars)')

    for col, (ax, d) in enumerate(zip([axes[1, 0], axes[1, 1]], ['LONG', 'SHORT'])):
        sub = df[df['direction'] == d]
        if len(sub) > 0:
            sub.boxplot(column='mae_fixed_24', by='score_bin', ax=ax)
        ax.set_title(f'{d} — Fixed MAE (24 bars)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  图3: {output_path}")


def plot_actual_vs_fixed(stats, output_path):
    """Actual vs Fixed对比图。"""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Actual vs Fixed Window Comparison', fontsize=14)

    x = np.arange(len(stats))
    w = 0.35

    axes[0].bar(x - w / 2, stats['median_mfe_actual'], w, label='Actual', alpha=0.8)
    axes[0].bar(x + w / 2, stats['median_mfe_fixed_24'], w, label='Fixed 24bar', alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(stats['score_bin'].astype(str), rotation=45)
    axes[0].set_title('Median MFE: Actual vs Fixed')
    axes[0].set_ylabel('Points')
    axes[0].legend()

    axes[1].bar(x - w / 2, stats['mean_mae_actual'], w, label='Actual', alpha=0.8)
    axes[1].bar(x + w / 2, stats['mean_mae_fixed_24'], w, label='Fixed 24bar', alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(stats['score_bin'].astype(str), rotation=45)
    axes[1].set_title('Mean MAE: Actual vs Fixed')
    axes[1].set_ylabel('Points')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  图4: {output_path}")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def auto_detect_climax(stats):
    """自动检测climax假设。"""
    mfes = stats['median_mfe_fixed_24'].values
    if len(mfes) < 3:
        return {'verdict': 'INCONCLUSIVE', 'details': 'insufficient bins'}

    # 检查前N-1档是否单调递增，最后一档是否下降
    increasing = all(mfes[i] <= mfes[i + 1] for i in range(len(mfes) - 2))
    last_drop = mfes[-1] < mfes[-2] * 0.9  # 下降超10%

    if increasing and last_drop:
        return {'verdict': 'SUPPORTED',
                'details': f'MFE increases from {mfes[0]:.1f} to {mfes[-2]:.1f}, then drops to {mfes[-1]:.1f} (-{(1-mfes[-1]/mfes[-2])*100:.0f}%)'}
    elif all(mfes[i] <= mfes[i + 1] for i in range(len(mfes) - 1)):
        return {'verdict': 'REJECTED',
                'details': f'MFE increases monotonically: {" → ".join(f"{m:.1f}" for m in mfes)}'}
    else:
        return {'verdict': 'INCONCLUSIVE',
                'details': f'non-monotonic pattern: {" → ".join(f"{m:.1f}" for m in mfes)}'}


def generate_text_report(symbol, trades_enriched, stats_all, stats_regime, stats_direction):
    """生成文本报告。"""
    n = len(trades_enriched)
    date_range = f"{trades_enriched['trade_date'].min()} ~ {trades_enriched['trade_date'].max()}"
    n_high = (trades_enriched['regime'] == 'high_amp').sum()
    n_low = (trades_enriched['regime'] == 'low_amp').sum()
    n_long = (trades_enriched['direction'] == 'LONG').sum()
    n_short = (trades_enriched['direction'] == 'SHORT').sum()

    lines = [
        "=" * 80,
        "ENTRY SCORE PROFILING REPORT",
        "=" * 80,
        f"Data: {symbol} | Trades: {n} | Date range: {date_range}",
        f"Regime split: high_amp={n_high} ({n_high/n*100:.0f}%) | low_amp={n_low} ({n_low/n*100:.0f}%)",
        f"Direction split: LONG={n_long} | SHORT={n_short}",
        "",
        "--- Overall Statistics by Score Bin ---",
        f"{'Bin':<12s} {'N':>5s} {'Med_MFE':>9s} {'Med_MAE':>9s} {'MFE/MAE':>8s} {'Hold':>6s} {'WinR':>6s} {'FxWin':>6s}",
    ]

    for _, r in stats_all.iterrows():
        warn = "  ⚠ <30" if r['trade_count'] < 30 else ""
        ratio_str = f"{r['mfe_mae_ratio']:.2f}" if pd.notna(r['mfe_mae_ratio']) else "N/A"
        lines.append(
            f"{str(r['score_bin']):<12s} {int(r['trade_count']):>5d} "
            f"{r['median_mfe_fixed_24']:>9.1f} {r['median_mae_fixed_24']:>9.1f} "
            f"{ratio_str:>8s} {r['mean_holding_bars']:>6.1f} "
            f"{r['actual_win_rate']*100:>5.1f}% {r['fixed_win_rate_24']*100:>5.1f}%{warn}"
        )

    # By regime
    for regime_name, regime_key in [("High Amplitude Days", "high_amp"), ("Low Amplitude Days", "low_amp")]:
        sub_stats = stats_regime.get(regime_key)
        if sub_stats is not None and len(sub_stats) > 0:
            lines.append(f"\n--- By Regime: {regime_name} ---")
            for _, r in sub_stats.iterrows():
                ratio_str = f"{r['mfe_mae_ratio']:.2f}" if pd.notna(r['mfe_mae_ratio']) else "N/A"
                lines.append(
                    f"{str(r['score_bin']):<12s} {int(r['trade_count']):>5d} "
                    f"{r['median_mfe_fixed_24']:>9.1f} {r['median_mae_fixed_24']:>9.1f} "
                    f"{ratio_str:>8s} {r['actual_win_rate']*100:>5.1f}%"
                )

    # Key findings
    climax = auto_detect_climax(stats_all)
    lines.append(f"\n--- Key Findings ---")
    lines.append(f"1. Climax hypothesis: {climax['verdict']}")
    lines.append(f"   {climax['details']}")

    # Actual vs Fixed gap
    gap = stats_all['median_mfe_fixed_24'].mean() - stats_all['median_mfe_actual'].mean()
    lines.append(f"2. Actual vs Fixed MFE gap: exit strategy leaves {gap:.1f}pt on average")

    # Regime effect
    high_stats = stats_regime.get('high_amp')
    low_stats = stats_regime.get('low_amp')
    if high_stats is not None and low_stats is not None and len(high_stats) > 0 and len(low_stats) > 0:
        high_mfe = high_stats['median_mfe_fixed_24'].mean()
        low_mfe = low_stats['median_mfe_fixed_24'].mean()
        lines.append(f"3. Regime effect: high_amp median MFE={high_mfe:.1f}pt, low_amp={low_mfe:.1f}pt")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Entry Score Profiling")
    parser.add_argument('--symbol', default='IM')
    parser.add_argument('--start', default='20250516', help='YYYYMMDD')
    parser.add_argument('--end', default='20260409', help='YYYYMMDD')
    parser.add_argument('--bins', default='fixed', choices=['fixed', 'qcut'])
    parser.add_argument('--amp-threshold', type=float, default=0.004)
    parser.add_argument('--output-dir', default='tmp/entry_score_profile')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 60}")
    print(f"  Entry Score Profiling | {args.symbol}")
    print(f"  {args.start} ~ {args.end}")
    print(f"{'=' * 60}")

    # Load data
    print("\n加载交易数据（回测）...")
    from config.config_loader import ConfigLoader
    db_path = ConfigLoader().get_db_path()
    trades = load_trades(db_path, args.symbol, args.start, args.end)
    print(f"  {len(trades)} 笔交易")

    if len(trades) < 50:
        print(f"  ⚠ 样本过少（{len(trades)}<50），建议等数据积累更多")

    print("加载K线数据...")
    ohlcv = load_ohlcv(db_path, args.symbol, args.start, args.end)
    print(f"  {len(ohlcv)} 根bar")

    # Enrich
    print("计算MFE/MAE/regime...")
    enriched = enrich_trades(trades, ohlcv, amp_threshold=args.amp_threshold)
    print(f"  {len(enriched)} 笔有效交易")

    if len(enriched) == 0:
        print("无有效交易，退出")
        return

    # Bin
    if args.bins == 'qcut':
        enriched['score_bin'] = pd.qcut(enriched['entry_score'], 4, duplicates='drop')
    else:
        enriched, _, _ = auto_adjust_bins(enriched)

    # 样本数警告
    for bin_label in enriched['score_bin'].unique():
        n = (enriched['score_bin'] == bin_label).sum()
        if n < 30:
            print(f"  ⚠ {bin_label}: {n} 笔 (< 30)")

    # Aggregate
    stats_all = aggregate_stats(enriched)

    stats_regime = {}
    for regime in ['high_amp', 'low_amp']:
        sub = enriched[enriched['regime'] == regime]
        if len(sub) >= 10:
            stats_regime[regime] = aggregate_stats(sub)

    stats_direction = {}
    for d in ['LONG', 'SHORT']:
        sub = enriched[enriched['direction'] == d]
        if len(sub) >= 10:
            stats_direction[d] = aggregate_stats(sub)

    # Plot
    print("\n生成图表...")
    plot_overall_2x2(enriched, stats_all, str(output_dir / "entry_score_profile_all.png"))
    plot_by_regime(enriched, str(output_dir / "entry_score_profile_by_regime.png"))
    plot_by_direction(enriched, str(output_dir / "entry_score_profile_by_direction.png"))
    plot_actual_vs_fixed(stats_all, str(output_dir / "entry_score_profile_actual_vs_fixed.png"))

    # Report
    print("\n生成报告...")
    report = generate_text_report(args.symbol, enriched, stats_all, stats_regime, stats_direction)
    print(report)

    report_path = output_dir / "entry_score_profile_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\n报告已保存: {report_path}")

    # Save enriched data
    csv_path = output_dir / "entry_score_profile_data.csv"
    enriched.to_csv(csv_path, index=False)
    print(f"数据已保存: {csv_path}")


if __name__ == "__main__":
    main()
