"""
performance_dashboard.py
------------------------
Dashboard 页面：策略绩效

功能:
- 累计收益率、年化收益率、最大回撤、夏普比率、胜率
- 累计收益曲线 vs IM.CFX
- 日盈亏柱状图 + 30日均线
- 回撤曲线
- 月度统计表
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[2])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# ──────────────────────────────────────────────
# DB helpers
# ──────────────────────────────────────────────

@st.cache_resource
def _get_db():
    try:
        from data.storage.db_manager import DBManager, get_db
        from config.config_loader import ConfigLoader
        config = ConfigLoader()
        db = get_db(config)
        db.initialize_tables()
        return db
    except Exception as e:
        st.error(f"数据库连接失败: {e}")
        return None


def _load_equity_history(db) -> pd.DataFrame:
    try:
        df = db.query_df(
            "SELECT trade_date, balance FROM account_snapshots ORDER BY trade_date ASC"
        )
        if df is None:
            return pd.DataFrame()
        # Deduplicate: keep one row per trade_date (latest snapshot)
        df = df.drop_duplicates(subset="trade_date", keep="last")
        return df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def _load_im_daily(db, n: int = 2000) -> pd.DataFrame:
    try:
        df = db.query_df(
            f"""SELECT trade_date, close FROM futures_daily
                WHERE ts_code = 'IM.CFX'
                ORDER BY trade_date DESC LIMIT {n}"""
        )
        if df is None or df.empty:
            return pd.DataFrame()
        return df.sort_values("trade_date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


def _calc_performance_metrics(equity_df: pd.DataFrame) -> dict:
    if equity_df.empty or len(equity_df) < 2:
        return {}
    balance = equity_df["balance"].astype(float)
    daily_returns = balance.pct_change().dropna()
    trading_days = len(equity_df)
    cum_return = (balance.iloc[-1] / balance.iloc[0]) - 1

    metrics = {
        "cum_return": cum_return,
        "trading_days": trading_days,
    }

    if trading_days > 20:
        ann_return = (1 + cum_return) ** (252 / trading_days) - 1
        metrics["ann_return"] = ann_return
    else:
        metrics["ann_return"] = None

    # Max drawdown
    roll_max = balance.cummax()
    drawdown = (balance - roll_max) / roll_max
    metrics["max_drawdown"] = float(drawdown.min())
    metrics["drawdown_series"] = drawdown.values.tolist()

    if trading_days >= 20 and daily_returns.std() > 0:
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        metrics["sharpe"] = float(sharpe)
    else:
        metrics["sharpe"] = None

    if len(daily_returns) > 0:
        win_rate = (daily_returns > 0).sum() / len(daily_returns)
        metrics["win_rate"] = float(win_rate)
    else:
        metrics["win_rate"] = None

    metrics["daily_returns"] = daily_returns.values.tolist()
    metrics["daily_return_dates"] = equity_df["trade_date"].iloc[1:].tolist()

    return metrics


def _na_str(val, fmt: str = ".2%") -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "数据积累中"
    return format(val, fmt)


# ──────────────────────────────────────────────
# Render
# ──────────────────────────────────────────────

def render() -> None:
    st.title("💰 策略绩效")

    db = _get_db()
    if db is None:
        st.error("无法连接数据库，请检查配置。")
        return

    equity_df = _load_equity_history(db)
    im_df = _load_im_daily(db)

    latest_date = equity_df["trade_date"].iloc[-1] if not equity_df.empty else "N/A"
    st.caption(f"数据更新至: {latest_date}")

    metrics = _calc_performance_metrics(equity_df)
    n_days = len(equity_df)

    # ── KPI Row ──
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("累计收益率", _na_str(metrics.get("cum_return")))
    k2.metric("年化收益率", _na_str(metrics.get("ann_return")))
    k3.metric("最大回撤", _na_str(metrics.get("max_drawdown")))
    k4.metric("夏普比率", _na_str(metrics.get("sharpe"), ".2f") if metrics.get("sharpe") is not None else "数据积累中")
    k5.metric("胜率", _na_str(metrics.get("win_rate")) if metrics.get("win_rate") is not None else "数据积累中")

    st.divider()

    if n_days < 3:
        st.info("数据积累中，需要更多交易日（当前 %d 天，建议 ≥ 20 天获得完整统计）" % n_days)
        if not equity_df.empty:
            st.dataframe(equity_df, use_container_width=True)
        return

    # ── Chart 1: Cumulative Return ──
    st.subheader("累计收益率曲线")
    fig1 = go.Figure()

    balance = equity_df["balance"].astype(float)
    cum_ret = (balance / balance.iloc[0] - 1) * 100
    fig1.add_trace(go.Scatter(
        x=equity_df["trade_date"].tolist(),
        y=cum_ret.tolist(),
        mode="lines", name="策略累计收益",
        line=dict(color="#3498db", width=2),
    ))

    # IM.CFX benchmark
    if not im_df.empty:
        # Align start date
        start_date = equity_df["trade_date"].iloc[0]
        im_aligned = im_df[im_df["trade_date"] >= start_date].copy()
        if not im_aligned.empty and len(im_aligned) >= 2:
            base_im = im_aligned["close"].iloc[0]
            im_cum = (im_aligned["close"] / base_im - 1) * 100
            fig1.add_trace(go.Scatter(
                x=im_aligned["trade_date"].tolist(),
                y=im_cum.tolist(),
                mode="lines", name="IM.CFX（基准）",
                line=dict(color="#e67e22", width=1.5, dash="dash"),
            ))

    fig1.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)")
    fig1.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)", title="累计收益率 (%)"),
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=10, r=10, t=40, b=10),
        height=320,
    )
    st.plotly_chart(fig1, use_container_width=True)

    # ── Chart 2: Daily P&L ──
    st.subheader("日盈亏柱状图")
    daily_returns = metrics.get("daily_returns", [])

    if len(daily_returns) >= 3:
        balance_vals = balance.values
        daily_pnl = np.diff(balance_vals)  # absolute P&L
        pnl_dates = equity_df["trade_date"].iloc[1:].tolist()
        bar_colors = ["#e74c3c" if v >= 0 else "#2ecc71" for v in daily_pnl]

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=pnl_dates, y=daily_pnl.tolist(),
            marker_color=bar_colors,
            name="日盈亏",
        ))
        # 30-day rolling mean
        pnl_series = pd.Series(daily_pnl)
        if len(pnl_series) >= 10:
            roll_win = min(30, len(pnl_series))
            rolling_mean = pnl_series.rolling(roll_win, min_periods=1).mean()
            fig2.add_trace(go.Scatter(
                x=pnl_dates,
                y=rolling_mean.tolist(),
                mode="lines", name=f"{roll_win}日均线",
                line=dict(color="#9b59b6", width=2),
            ))
        fig2.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)")
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)", title="盈亏 (元)"),
            legend=dict(orientation="h", y=1.02),
            margin=dict(l=10, r=10, t=40, b=10),
            height=280,
            barmode="relative",
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("日盈亏数据不足（< 3 天）")

    # ── Chart 3: Drawdown ──
    st.subheader("回撤曲线")
    dd_series = metrics.get("drawdown_series", [])
    if len(dd_series) >= 3:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=equity_df["trade_date"].tolist(),
            y=[v * 100 for v in dd_series],
            mode="lines", name="回撤",
            line=dict(color="#e74c3c", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(231,76,60,0.2)",
        ))
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)", title="回撤 (%)"),
            margin=dict(l=10, r=10, t=30, b=10),
            height=250,
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("回撤数据不足（< 3 天）")

    # ── Chart 4: P&L Attribution Placeholder ──
    st.subheader("盈亏归因")
    st.info("需要更多数据积累（Greeks 归因需要多日持仓和模型输出数据）")

    # ── Monthly Table ──
    st.subheader("月度统计")
    if len(equity_df) < 5:
        st.info("月度统计需要更多数据")
    else:
        eq = equity_df.copy()
        eq["trade_date"] = pd.to_datetime(eq["trade_date"])
        eq["year_month"] = eq["trade_date"].dt.to_period("M").astype(str)
        eq["balance"] = eq["balance"].astype(float)
        eq["daily_pnl"] = eq["balance"].diff()
        eq["daily_ret"] = eq["balance"].pct_change()

        monthly_rows = []
        for ym, grp in eq.groupby("year_month"):
            grp = grp.dropna(subset=["daily_ret"])
            if grp.empty:
                continue
            month_grp = eq[eq["year_month"] == ym]["balance"].dropna()
            if len(month_grp) < 2:
                monthly_ret = 0.0
            else:
                monthly_ret = (month_grp.iloc[-1] / month_grp.iloc[0] - 1)

            # Max drawdown within month
            if len(month_grp) >= 2:
                roll_max = month_grp.cummax()
                dd = (month_grp - roll_max) / roll_max
                max_dd = float(dd.min())
            else:
                max_dd = 0.0

            trade_count_q = 0
            try:
                ym_start = str(grp["trade_date"].min().date())
                ym_end = str(grp["trade_date"].max().date())
                tc_df = db.query_df(
                    f"SELECT COUNT(*) as cnt FROM trade_records WHERE trade_date BETWEEN '{ym_start}' AND '{ym_end}'"
                )
                if tc_df is not None and not tc_df.empty:
                    trade_count_q = int(tc_df.iloc[0]["cnt"])
            except Exception:
                pass

            win_days = (grp["daily_ret"] > 0).sum()
            total_days = len(grp["daily_ret"])
            wr = win_days / total_days if total_days > 0 else 0

            monthly_rows.append({
                "年月": ym,
                "月收益率": f"{monthly_ret:.2%}",
                "最大回撤": f"{max_dd:.2%}",
                "成交笔数": trade_count_q,
                "胜率": f"{wr:.2%}",
            })

        if monthly_rows:
            monthly_df = pd.DataFrame(monthly_rows)

            def _color_monthly_ret(val):
                try:
                    v = float(val.replace("%", "")) / 100
                    if v > 0:
                        return "color: #e74c3c"
                    elif v < 0:
                        return "color: #2ecc71"
                    return ""
                except Exception:
                    return ""

            styled_monthly = monthly_df.style.applymap(_color_monthly_ret, subset=["月收益率"])
            st.dataframe(styled_monthly, use_container_width=True, hide_index=True)
        else:
            st.info("暂无月度统计数据")


if __name__ == "__main__":
    render()
