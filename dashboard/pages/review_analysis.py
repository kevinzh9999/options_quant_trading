"""
review_analysis.py
------------------
Dashboard 页面：复盘分析 (Review & Analysis)

信号准确率统计、策略绩效追踪、盘感 vs 系统对比。
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
from plotly.subplots import make_subplots

_BG = "rgba(0,0,0,0)"
_GRID = "rgba(255,255,255,0.1)"
PRODUCT_COLORS = {"IF": "#3498db", "IH": "#e74c3c", "IM": "#e67e22", "IC": "#9b59b6"}


@st.cache_resource
def _get_db():
    try:
        from data.storage.db_manager import DBManager
        from config.config_loader import ConfigLoader
        db = DBManager(ConfigLoader().get_db_path())
        db.initialize_tables()
        return db
    except Exception as e:
        st.error(f"数据库连接失败: {e}")
        return None


# ── 数据加载 ──────────────────────────────────────────

def _load_all_signals(db) -> pd.DataFrame:
    df = db.query_df("SELECT * FROM signal_log ORDER BY datetime")
    return df if df is not None else pd.DataFrame()


def _load_futures_min(db) -> pd.DataFrame:
    """加载5分钟数据用于计算信号后收益"""
    df = db.query_df(
        "SELECT symbol, datetime, close FROM futures_min ORDER BY datetime"
    )
    return df if df is not None else pd.DataFrame()


def _load_futures_daily(db, symbol: str = "IM.CFX") -> pd.DataFrame:
    df = db.query_df(
        f"SELECT trade_date, close FROM futures_daily "
        f"WHERE ts_code='{symbol}' ORDER BY trade_date"
    )
    return df if df is not None else pd.DataFrame()


def _load_trade_decisions(db) -> pd.DataFrame:
    df = db.query_df("SELECT * FROM trade_decisions ORDER BY datetime")
    return df if df is not None else pd.DataFrame()


def _load_model_history(db) -> pd.DataFrame:
    df = db.query_df(
        "SELECT * FROM daily_model_output WHERE underlying='IM' ORDER BY trade_date"
    )
    return df if df is not None else pd.DataFrame()


# ── 信号准确率 ────────────────────────────────────────

def _render_signal_accuracy(signals: pd.DataFrame, futures_daily: pd.DataFrame):
    st.subheader("信号准确率统计")

    if signals.empty:
        st.info("暂无信号数据")
        return

    # 提取有效信号（有方向的）
    sig = signals.copy()
    sig["dir"] = sig.apply(
        lambda r: r.get("direction") or r.get("direction_v2") or r.get("direction_v3") or "",
        axis=1,
    )
    sig["scr"] = sig.apply(
        lambda r: int(r.get("score") or r.get("score_v2") or r.get("score_v3") or 0),
        axis=1,
    )
    sig["ver"] = sig.apply(
        lambda r: r.get("signal_version") or ("v2" if r.get("score_v2") else "v3" if r.get("score_v3") else ""),
        axis=1,
    )
    sig = sig[sig["dir"].isin(["LONG", "SHORT"])].copy()

    if sig.empty:
        st.info("无有效方向性信号")
        return

    sig["date"] = sig["datetime"].str[:10]
    sig["hour"] = sig["datetime"].str[11:13].astype(int, errors="ignore")

    # 时段分类
    def _session(h):
        try:
            h = int(h)
        except (ValueError, TypeError):
            return "其他"
        if h < 3:  # 上午盘 (9:30-11:30 → UTC+8偏移后)
            return "上午"
        elif h < 5:
            return "午后"
        else:
            return "尾盘"
    sig["session"] = sig["hour"].apply(_session)

    # 计算信号后收益（用日线数据近似：信号日 → 次日收盘变动方向）
    if not futures_daily.empty:
        daily = futures_daily.copy()
        daily["next_close"] = daily["close"].shift(-1)
        daily["daily_return"] = (daily["next_close"] - daily["close"]) / daily["close"]
        daily_map = dict(zip(daily["trade_date"], daily["daily_return"]))

        sig["trade_date"] = sig["date"].str.replace("-", "")
        sig["next_day_return"] = sig["trade_date"].map(daily_map)

        # 信号是否正确
        sig["correct"] = (
            ((sig["dir"] == "LONG") & (sig["next_day_return"] > 0)) |
            ((sig["dir"] == "SHORT") & (sig["next_day_return"] < 0))
        )
    else:
        sig["next_day_return"] = np.nan
        sig["correct"] = np.nan

    # 按品种统计
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**按品种/版本**")
        valid = sig.dropna(subset=["correct"])
        if not valid.empty:
            stats = valid.groupby(["symbol", "ver"]).agg(
                总信号=("correct", "count"),
                正确=("correct", "sum"),
                准确率=("correct", "mean"),
            ).reset_index()
            stats["准确率"] = (stats["准确率"] * 100).round(1).astype(str) + "%"
            st.dataframe(stats, use_container_width=True, hide_index=True)
        else:
            st.caption("次日收益数据不足")

    with col2:
        st.markdown("**按时段**")
        valid = sig.dropna(subset=["correct"])
        if not valid.empty:
            stats = valid.groupby("session").agg(
                总信号=("correct", "count"),
                正确=("correct", "sum"),
                准确率=("correct", "mean"),
            ).reset_index()
            stats["准确率"] = (stats["准确率"] * 100).round(1).astype(str) + "%"
            st.dataframe(stats, use_container_width=True, hide_index=True)

    # 得分 vs 收益散点图
    valid = sig.dropna(subset=["next_day_return"])
    if not valid.empty and len(valid) > 5:
        st.markdown("**信号得分 vs 次日收益**")
        signed_score = valid["scr"].copy()
        signed_score[valid["dir"] == "SHORT"] = -signed_score[valid["dir"] == "SHORT"]

        fig = go.Figure()
        for symbol in valid["symbol"].unique():
            sub = valid[valid["symbol"] == symbol]
            ss = signed_score[sub.index]
            fig.add_trace(go.Scatter(
                x=ss, y=sub["next_day_return"] * 100,
                mode="markers",
                name=symbol,
                marker=dict(
                    color=PRODUCT_COLORS.get(symbol, "#999"),
                    size=6, opacity=0.7,
                ),
            ))

        fig.add_hline(y=0, line_color="gray", opacity=0.3)
        fig.add_vline(x=0, line_color="gray", opacity=0.3)
        fig.update_layout(
            height=350,
            plot_bgcolor=_BG, paper_bgcolor=_BG,
            xaxis=dict(title="带方向得分 (空为负)", gridcolor=_GRID),
            yaxis=dict(title="次日收益 (%)", gridcolor=_GRID),
            legend=dict(orientation="h", y=1.02),
            margin=dict(l=40, r=20, t=40, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

    # 信号频次统计
    st.markdown("**信号频次统计**")
    freq = sig.groupby("date").size().reset_index(name="信号数")
    if not freq.empty:
        avg_per_day = freq["信号数"].mean()
        st.caption(f"日均信号: {avg_per_day:.1f}个 | 总天数: {len(freq)} | 总信号: {len(sig)}")


# ── 策略绩效追踪 ──────────────────────────────────────

def _render_strategy_performance(model_hist: pd.DataFrame):
    st.subheader("策略绩效追踪")

    if model_hist.empty:
        st.info("暂无模型输出数据")
        return

    # 从 daily_model_output 提取 PnL 时间序列
    pnl_cols = ["pnl_total", "pnl_delta", "pnl_gamma", "pnl_theta", "pnl_vega"]
    available = [c for c in pnl_cols if c in model_hist.columns]

    if not available:
        st.info("daily_model_output 中无 PnL 数据")
        return

    df = model_hist[["trade_date"] + available].dropna(subset=["pnl_total"])
    if df.empty:
        st.info("无有效 PnL 记录")
        return

    # 累计收益
    df = df.copy()
    df["cumulative"] = df["pnl_total"].astype(float).cumsum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["trade_date"], y=df["cumulative"],
        name="累计PnL", line=dict(color="#3498db"),
        fill="tozeroy", fillcolor="rgba(52,152,219,0.1)",
    ))
    fig.update_layout(
        title="累计收益曲线",
        height=300,
        plot_bgcolor=_BG, paper_bgcolor=_BG,
        xaxis=dict(gridcolor=_GRID),
        yaxis=dict(title="累计盈亏 (元)", gridcolor=_GRID),
        margin=dict(l=40, r=20, t=50, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 收益分解柱状图
    if len(df) > 0:
        st.markdown("**收益分解**")
        decomp = {}
        for col in ["pnl_delta", "pnl_gamma", "pnl_theta", "pnl_vega", "pnl_residual"]:
            if col in df.columns:
                vals = df[col].astype(float)
                decomp[col.replace("pnl_", "").title()] = vals.sum()

        if decomp:
            labels = list(decomp.keys())
            values = list(decomp.values())
            colors = ["#e74c3c" if v >= 0 else "#2ecc71" for v in values]

            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=labels, y=values,
                marker_color=colors,
                text=[f"{v:+,.0f}" for v in values],
                textposition="outside",
            ))
            fig2.update_layout(
                title="期间收益归因合计",
                height=300,
                plot_bgcolor=_BG, paper_bgcolor=_BG,
                xaxis=dict(gridcolor=_GRID),
                yaxis=dict(title="盈亏 (元)", gridcolor=_GRID),
                margin=dict(l=40, r=20, t=50, b=30),
            )
            st.plotly_chart(fig2, use_container_width=True)


# ── 盘感 vs 系统 ─────────────────────────────────────

def _render_human_vs_system(decisions: pd.DataFrame, futures_daily: pd.DataFrame):
    st.subheader("盘感 vs 系统")

    if decisions.empty:
        st.info("暂无交易决策记录（从 trade_decisions 表读取）")
        return

    dec = decisions.copy()
    dec["date"] = dec["datetime"].str[:10].str.replace("-", "")

    # 匹配次日收益
    if not futures_daily.empty:
        daily = futures_daily.copy()
        daily["next_close"] = daily["close"].shift(-1)
        daily["daily_return"] = (daily["next_close"] - daily["close"]) / daily["close"]

        # 用 IM.CFX 的收益
        daily_map = dict(zip(daily["trade_date"], daily["daily_return"]))
        dec["next_day_return"] = dec["date"].map(daily_map)

        # 信号收益 = 次日收益 × 方向
        dec["signal_return"] = dec["next_day_return"]
        dec.loc[dec["signal_direction"] == "SHORT", "signal_return"] *= -1
    else:
        dec["signal_return"] = np.nan

    # 分组统计
    executed = dec[dec["decision"].isin(["EXECUTED", "OPEN"])]
    skipped = dec[dec["decision"] == "SKIPPED"]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**✅ 执行的信号**")
        if not executed.empty and "signal_return" in executed.columns:
            valid = executed.dropna(subset=["signal_return"])
            if not valid.empty:
                avg_ret = valid["signal_return"].mean() * 100
                win_rate = (valid["signal_return"] > 0).mean() * 100
                st.metric("平均事后收益", f"{avg_ret:+.2f}%")
                st.metric("正确率", f"{win_rate:.0f}%")
                st.metric("次数", f"{len(valid)}")
            else:
                st.caption("无次日收益数据")
        else:
            st.caption(f"共 {len(executed)} 次执行")

    with col2:
        st.markdown("**⏭️ 跳过的信号**")
        if not skipped.empty and "signal_return" in skipped.columns:
            valid = skipped.dropna(subset=["signal_return"])
            if not valid.empty:
                avg_ret = valid["signal_return"].mean() * 100
                win_rate = (valid["signal_return"] > 0).mean() * 100
                st.metric("如果执行的平均收益", f"{avg_ret:+.2f}%")
                st.metric("本应正确率", f"{win_rate:.0f}%")
                st.metric("次数", f"{len(valid)}")
            else:
                st.caption("无次日收益数据")
        else:
            st.caption(f"共 {len(skipped)} 次跳过")

    # 决策时间线
    st.markdown("**决策时间线**")
    timeline = []
    for _, r in dec.iterrows():
        decision = str(r.get("decision", ""))
        icon = "✅" if decision in ("EXECUTED", "OPEN") else "⏭️" if decision == "SKIPPED" else "📝"
        timeline.append({
            "时间": str(r.get("datetime", ""))[:19],
            "品种": r.get("symbol", ""),
            "方向": r.get("signal_direction", ""),
            "得分": r.get("signal_score", ""),
            "决策": f"{icon} {decision}",
            "备注": r.get("manual_note", "") or "",
        })

    if timeline:
        st.dataframe(pd.DataFrame(timeline), use_container_width=True, hide_index=True)


# ── 主渲染 ────────────────────────────────────────────

def render():
    st.title("复盘分析")

    db = _get_db()
    if db is None:
        return

    signals = _load_all_signals(db)
    futures_daily = _load_futures_daily(db)
    decisions = _load_trade_decisions(db)
    model_hist = _load_model_history(db)

    tab1, tab2, tab3 = st.tabs(["信号准确率", "策略绩效", "盘感vs系统"])

    with tab1:
        _render_signal_accuracy(signals, futures_daily)

    with tab2:
        _render_strategy_performance(model_hist)

    with tab3:
        _render_human_vs_system(decisions, futures_daily)
