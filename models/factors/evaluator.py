"""
Layer 1.5: Factor Evaluator（因子评估器）
=========================================
评估因子的预测力：IC、分组收益、单调性、因子间相关性。
支持按振幅regime分组评估（04-04研究结论：高/低振幅日表现可能完全不同）。
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from models.factors.base import Factor


class FactorEvaluator:
    """
    因子评估器。

    支持：
    1. IC (Information Coefficient) — Spearman rank correlation
    2. IC_IR — IC稳定性
    3. 分组收益（Q1~Q5）
    4. 单调性检验
    5. 按振幅regime分组评估
    """

    def __init__(self, bar_5m: pd.DataFrame,
                 forward_periods: list = None,
                 daily_range: pd.Series = None):
        """
        Args:
            bar_5m: 5分钟K线，columns=[open, high, low, close, volume]
            forward_periods: 评估的未来收益bar数，默认[1, 3, 5, 10]
            daily_range: 可选，每天的日内振幅(%)，用于regime分组评估
                         index应为bar_5m中每根bar所属日期
        """
        if forward_periods is None:
            forward_periods = [1, 3, 5, 10]
        self.bar_5m = bar_5m
        self.forward_periods = forward_periods
        self.daily_range = daily_range

        # 预计算 target (未来N根bar的收益率)
        self.targets = {}
        for n in forward_periods:
            self.targets[n] = bar_5m['close'].pct_change(n).shift(-n)

    def evaluate(self, factor: Factor) -> dict:
        """评估单个因子，返回完整评估报告。"""
        values = factor.compute_series(self.bar_5m)
        values = values.replace([np.inf, -np.inf], np.nan)

        result = {
            'name': factor.name,
            'category': factor.category,
            'stats': self._calc_stats(values),
            'ic': self._calc_ic(values),
            'group_returns': self._calc_group_returns(values),
            'monotonicity': self._calc_monotonicity(values),
        }

        # 按振幅regime分组
        if self.daily_range is not None:
            result['regime_ic'] = self._calc_regime_ic(values)

        return result

    def _calc_stats(self, values: pd.Series) -> dict:
        v = values.dropna()
        return {
            'count': len(v),
            'mean': float(v.mean()) if len(v) > 0 else 0,
            'std': float(v.std()) if len(v) > 0 else 0,
            'skew': float(v.skew()) if len(v) > 2 else 0,
            'pct_nan': float(values.isna().mean()),
        }

    def _calc_ic(self, values: pd.Series) -> dict:
        """计算每个forward period的IC和IC_IR。

        包含两种IC：
        - 全局IC：全部bar做一个Spearman（可能因自相关高估显著性）
        - Daily IC：每天算一个IC，取均值和标准差（更接近实盘决策频率）
        """
        ics = {}
        for n in self.forward_periods:
            target = self.targets[n].reindex(values.index)
            valid = pd.concat([values.rename('f'), target.rename('t')], axis=1).dropna()
            if len(valid) < 30:
                ics[f'IC_{n}bar'] = float('nan')
                continue

            # 全局IC
            ic = valid['f'].corr(valid['t'], method='spearman')
            ics[f'IC_{n}bar'] = round(float(ic), 4)

            # Daily IC：每天算一个IC，更稳健
            valid_with_date = valid.copy()
            valid_with_date['date'] = valid_with_date.index.date
            daily_ics = []
            for _, day_df in valid_with_date.groupby('date'):
                if len(day_df) >= 10:
                    d_ic = day_df['f'].corr(day_df['t'], method='spearman')
                    if not np.isnan(d_ic):
                        daily_ics.append(d_ic)
            if daily_ics:
                d_mean = np.mean(daily_ics)
                d_std = np.std(daily_ics, ddof=1)
                ics[f'dIC_{n}bar'] = round(float(d_mean), 4)
                ics[f'dICIR_{n}bar'] = round(float(d_mean / d_std), 4) if d_std > 0 else float('nan')
                ics[f'dIC_ndays_{n}bar'] = len(daily_ics)
            else:
                ics[f'dIC_{n}bar'] = float('nan')
                ics[f'dICIR_{n}bar'] = float('nan')

        return ics

    def _calc_group_returns(self, values: pd.Series, n_groups: int = 5) -> dict:
        """按因子值分组，看各组的平均未来收益。"""
        groups = {}
        for n in self.forward_periods:
            target = self.targets[n].reindex(values.index)
            valid = pd.concat([values.rename('factor'), target.rename('target')], axis=1).dropna()
            if len(valid) < 50:
                continue
            try:
                valid['group'] = pd.qcut(valid['factor'], n_groups,
                                         labels=[f'Q{i+1}' for i in range(n_groups)],
                                         duplicates='drop')
                gr = valid.groupby('group')['target'].mean()
                groups[f'{n}bar'] = {k: round(float(v) * 10000, 2) for k, v in gr.items()}  # in bps
            except ValueError:
                pass
        return groups

    def _calc_monotonicity(self, values: pd.Series) -> dict:
        """检验分组收益是否单调。"""
        mono = {}
        for n in self.forward_periods:
            target = self.targets[n].reindex(values.index)
            valid = pd.concat([values.rename('factor'), target.rename('target')], axis=1).dropna()
            if len(valid) < 50:
                continue
            try:
                valid['group'] = pd.qcut(valid['factor'], 5, labels=range(5), duplicates='drop')
                gr = valid.groupby('group')['target'].mean()
                corr, pval = spearmanr(range(len(gr)), gr.values)
                mono[f'{n}bar'] = {'corr': round(float(corr), 3), 'pval': round(float(pval), 4)}
            except (ValueError, TypeError):
                pass
        return mono

    def _calc_regime_ic(self, values: pd.Series) -> dict:
        """按日内振幅regime分组计算IC。"""
        if self.daily_range is None:
            return {}

        # Align daily_range with bar_5m index
        dr = self.daily_range.reindex(values.index)
        result = {}

        for regime_name, mask in [
            ('low_range(<1%)', dr < 1.0),
            ('mid_range(1-2%)', (dr >= 1.0) & (dr < 2.0)),
            ('high_range(>=2%)', dr >= 2.0),
        ]:
            for n in [3, 5]:  # only key periods
                target = self.targets[n].reindex(values.index)
                valid = pd.concat([
                    values.rename('f'), target.rename('t'), mask.rename('m')
                ], axis=1).dropna()
                sub = valid[valid['m']]
                if len(sub) < 20:
                    continue
                ic = sub['f'].corr(sub['t'], method='spearman')
                result[f'{regime_name}_IC_{n}bar'] = round(float(ic), 4)

        return result

    def batch_evaluate(self, factors: List[Factor]) -> tuple:
        """批量评估 + 相关性矩阵。"""
        results = []
        series_dict = {}

        for f in factors:
            result = self.evaluate(f)
            results.append(result)
            series_dict[f.name] = f.compute_series(self.bar_5m)

        corr_df = pd.DataFrame(series_dict).corr()
        return results, corr_df

    def print_report(self, factors: List[Factor]):
        """打印可读的评估报告。"""
        results, corr = self.batch_evaluate(factors)

        n_bars = len(self.bar_5m)
        print("=" * 100)
        print(f"  FACTOR EVALUATION REPORT | {len(factors)} factors | {n_bars} bars")
        print("=" * 100)

        # IC汇总表（全局 + Daily）
        sort_key = self.forward_periods[1]
        print(f"\n--- Global IC (全局，可能因自相关高估) ---")
        header = f"  {'Factor':<25} {'Cat':<10}"
        for n in self.forward_periods:
            header += f" {'IC_'+str(n):>8}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for r in sorted(results, key=lambda x: abs(x['ic'].get(f'IC_{sort_key}bar', 0)), reverse=True):
            row = f"  {r['name']:<25} {r['category']:<10}"
            for n in self.forward_periods:
                ic = r['ic'].get(f'IC_{n}bar', float('nan'))
                row += f" {ic:>+8.4f}" if not np.isnan(ic) else f" {'N/A':>8}"
            print(row)

        print(f"\n--- Daily IC (每天算一个IC，更稳健) ---")
        header2 = f"  {'Factor':<25}"
        for n in self.forward_periods:
            header2 += f" {'dIC_'+str(n):>8} {'dICIR':>7}"
        print(header2)
        print("  " + "-" * (len(header2) - 2))

        for r in sorted(results, key=lambda x: abs(x['ic'].get(f'dIC_{sort_key}bar', 0)), reverse=True):
            row = f"  {r['name']:<25}"
            for n in self.forward_periods:
                dic = r['ic'].get(f'dIC_{n}bar', float('nan'))
                dicir = r['ic'].get(f'dICIR_{n}bar', float('nan'))
                row += f" {dic:>+8.4f}" if not np.isnan(dic) else f" {'N/A':>8}"
                row += f" {dicir:>+7.3f}" if not np.isnan(dicir) else f" {'N/A':>7}"
            print(row)

        # 单调性
        print(f"\n--- Monotonicity (Q1→Q5, forward={self.forward_periods[1]}bar) ---")
        for r in results:
            key = f'{self.forward_periods[1]}bar'
            mono = r['monotonicity'].get(key, {})
            if mono:
                print(f"  {r['name']:<25} corr={mono['corr']:+.3f}  p={mono['pval']:.4f}")

        # Regime IC (if available)
        has_regime = any(r.get('regime_ic') for r in results)
        if has_regime:
            print(f"\n--- Regime IC (by daily range) ---")
            header = f"  {'Factor':<25}"
            regimes = ['low_range(<1%)', 'mid_range(1-2%)', 'high_range(>=2%)']
            for regime in regimes:
                header += f" {regime:>18}"
            print(header)
            print("  " + "-" * (len(header) - 2))
            for r in results:
                ri = r.get('regime_ic', {})
                if not ri:
                    continue
                row = f"  {r['name']:<25}"
                for regime in regimes:
                    ic = ri.get(f'{regime}_IC_{self.forward_periods[1]}bar', float('nan'))
                    row += f" {ic:>+18.4f}" if not np.isnan(ic) else f" {'N/A':>18}"
                print(row)

        # 相关性矩阵
        print(f"\n--- Factor Correlation Matrix ---")
        print(corr.round(3).to_string())

        # 分组收益（top 3因子）
        best = sorted(results,
                      key=lambda r: abs(r['ic'].get(f'IC_{self.forward_periods[1]}bar', 0)),
                      reverse=True)
        print(f"\n--- Top Factor Group Returns ({self.forward_periods[1]}bar, in bps) ---")
        for r in best[:3]:
            gr = r['group_returns'].get(f'{self.forward_periods[1]}bar', {})
            if gr:
                gr_str = "  ".join(f"{k}:{v:+.1f}" for k, v in gr.items())
                print(f"  {r['name']:<25} {gr_str}")

        print("=" * 100)
