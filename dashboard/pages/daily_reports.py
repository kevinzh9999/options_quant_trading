"""
daily_reports.py
----------------
Dashboard 页面：每日报告 (Daily Reports)

读取 daily_reports 表、logs/ 目录和 daily_model_output 展示 EOD 报告、
持仓分析、PnL 归因图表、交易笔记和多日对比视图。
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
_LOGS_DIR = Path(ROOT) / "logs"


# ── DB ────────────────────────────────────────────────

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

def _available_dates(db) -> list[str]:
    """获取有 daily_model_output 或 daily_reports 的所有日期"""
    dates = set()
    for table in ("daily_model_output", "daily_reports", "account_snapshots"):
        try:
            df = db.query_df(f"SELECT DISTINCT trade_date FROM {table}")
            if df is not None and not df.empty:
                dates.update(df["trade_date"].tolist())
        except Exception:
            pass
    # 也检查 logs/eod 目录
    eod_dir = _LOGS_DIR / "eod"
    if eod_dir.exists():
        for f in eod_dir.glob("*.md"):
            dates.add(f.stem)
    return sorted(dates, reverse=True)


def _load_report(db, trade_date: str, report_type: str) -> str | None:
    try:
        df = db.query_df(
            f"SELECT content FROM daily_reports "
            f"WHERE trade_date='{trade_date}' AND report_type='{report_type}'"
        )
        if df is not None and not df.empty:
            return str(df["content"].iloc[0])
    except Exception:
        pass
    return None


def _load_log_file(trade_date: str, subdir: str = "eod") -> str | None:
    """从 logs/ 目录读取 Markdown 文件"""
    path = _LOGS_DIR / subdir / f"{trade_date}.md"
    if path.exists():
        return path.read_text(encoding="utf-8")
    # 也检查根 logs 目录
    path2 = _LOGS_DIR / f"{trade_date}.md"
    if path2.exists():
        return path2.read_text(encoding="utf-8")
    return None


def _load_notes(db, trade_date: str) -> dict | None:
    try:
        df = db.query_df(
            f"SELECT * FROM daily_notes WHERE trade_date='{trade_date}'"
        )
        if df is not None and not df.empty:
            return df.iloc[0].to_dict()
    except Exception:
        pass
    return None


def _load_model_output(db, trade_date: str) -> dict | None:
    try:
        df = db.query_df(
            f"SELECT * FROM daily_model_output "
            f"WHERE trade_date='{trade_date}' AND underlying='IM'"
        )
        if df is not None and not df.empty:
            return df.iloc[0].to_dict()
    except Exception:
        pass
    return None


def _load_model_history(db, start_date: str, end_date: str) -> pd.DataFrame:
    try:
        df = db.query_df(
            f"SELECT * FROM daily_model_output "
            f"WHERE underlying='IM' AND trade_date>='{start_date}' AND trade_date<='{end_date}' "
            f"ORDER BY trade_date"
        )
        return df if df is not None else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _load_equity_range(db, start_date: str, end_date: str) -> pd.DataFrame:
    try:
        df = db.query_df(
            f"SELECT trade_date, balance FROM account_snapshots "
            f"WHERE trade_date>='{start_date}' AND trade_date<='{end_date}' "
            f"ORDER BY trade_date"
        )
        if df is not None and not df.empty:
            return df.drop_duplicates("trade_date", keep="last")
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


# ── 单日报告视图 ──────────────────────────────────────

def _render_single_day(db, trade_date: str):
    st.subheader(f"日期: {trade_date}")

    tabs = st.tabs(["EOD报告", "持仓分析", "PnL归因", "交易笔记"])

    with tabs[0]:
        # 优先从 daily_reports 表读，fallback logs/eod/
        content = _load_report(db, trade_date, "eod")
        if content is None:
            content = _load_log_file(trade_date, "eod")
        if content is None:
            content = _load_log_file(trade_date)  # root logs dir
        if content:
            st.markdown(content)
        else:
            st.info("该日期无EOD报告")

    with tabs[1]:
        content = _load_report(db, trade_date, "analysis")
        if content is None:
            content = _load_log_file(trade_date, "analysis")
        if content:
            st.markdown(content)
        else:
            st.info("该日期无持仓分析报告")

    with tabs[2]:
        _render_pnl_attribution(db, trade_date)

    with tabs[3]:
        notes = _load_notes(db, trade_date)
        if notes:
            if notes.get("market_observation"):
                st.markdown("**市场观察**")
                st.write(notes["market_observation"])
            if notes.get("trade_rationale"):
                st.markdown("**交易理由**")
                st.write(notes["trade_rationale"])
            if notes.get("deviations"):
                st.markdown("**偏差/异常**")
                st.write(notes["deviations"])
            if notes.get("lessons"):
                st.markdown("**经验教训**")
                st.write(notes["lessons"])
        else:
            # fallback: logs/notes/ Markdown file
            note_md = _load_log_file(trade_date, "notes")
            if note_md:
                st.markdown(note_md)
            else:
                st.info("该日期无交易笔记")


def _render_pnl_attribution(db, trade_date: str):
    """PnL 归因柱状图"""
    model = _load_model_output(db, trade_date)
    if model is None:
        st.info("该日期无PnL归因数据")
        return

    components = {
        "Delta": model.get("pnl_delta"),
        "Gamma": model.get("pnl_gamma"),
        "Theta": model.get("pnl_theta"),
        "Vega": model.get("pnl_vega"),
        "残差": model.get("pnl_residual"),
    }

    # 过滤 None/NaN
    filtered = {}
    for k, v in components.items():
        if v is not None:
            try:
                fv = float(v)
                if not np.isnan(fv):
                    filtered[k] = fv
            except (ValueError, TypeError):
                pass

    if not filtered:
        st.info("该日期无PnL归因数据")
        return

    total = model.get("pnl_total")
    if total is not None:
        try:
            total = float(total)
        except (ValueError, TypeError):
            total = None

    # 柱状图
    labels = list(filtered.keys())
    values = list(filtered.values())
    colors = ["#e74c3c" if v >= 0 else "#2ecc71" for v in values]  # A股惯例

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f"{v:+,.0f}" for v in values],
        textposition="outside",
    ))

    if total is not None and not np.isnan(total):
        fig.add_hline(y=total, line_dash="dash", line_color="white", opacity=0.5,
                      annotation_text=f"合计 {total:+,.0f}")

    fig.update_layout(
        title=f"PnL 归因分解 | {trade_date}",
        height=350,
        plot_bgcolor=_BG, paper_bgcolor=_BG,
        xaxis=dict(gridcolor=_GRID),
        yaxis=dict(title="盈亏 (元)", gridcolor=_GRID),
        margin=dict(l=40, r=20, t=50, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)

    # KPI 卡片
    if total is not None and not np.isnan(total):
        cols = st.columns(4)
        realized = model.get("pnl_realized")
        unrealized = model.get("pnl_unrealized")
        cols[0].metric("总PnL", f"{total:+,.0f}")
        if realized is not None:
            try:
                cols[1].metric("已实现", f"{float(realized):+,.0f}")
            except (ValueError, TypeError):
                pass
        if unrealized is not None:
            try:
                cols[2].metric("未实现", f"{float(unrealized):+,.0f}")
            except (ValueError, TypeError):
                pass


# ── 多日对比视图 ──────────────────────────────────────

def _render_comparison(db, dates: list[str]):
    if len(dates) < 2:
        st.info("请选择至少2个日期进行对比")
        return

    start = min(dates)
    end = max(dates)

    model_hist = _load_model_history(db, start, end)
    equity_hist = _load_equity_range(db, start, end)

    if model_hist.empty and equity_hist.empty:
        st.info("所选范围内无数据")
        return

    # 多行图
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=("账户权益", "ATM IV", "VRP", "净Delta", "净Theta", "净Vega"),
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
    )

    # 权益曲线
    if not equity_hist.empty:
        fig.add_trace(go.Scatter(
            x=equity_hist["trade_date"], y=equity_hist["balance"],
            name="权益", line=dict(color="#3498db"),
        ), row=1, col=1)

    if not model_hist.empty:
        # ATM IV
        for col_name, label, color in [
            ("atm_iv", "ATM IV", "#e67e22"),
            ("atm_iv_market", "Market IV", "#3498db"),
        ]:
            if col_name in model_hist.columns:
                vals = model_hist[col_name].dropna()
                if not vals.empty:
                    fig.add_trace(go.Scatter(
                        x=model_hist.loc[vals.index, "trade_date"],
                        y=vals * 100,
                        name=label, line=dict(color=color),
                    ), row=1, col=2)

        # VRP
        if "vrp" in model_hist.columns:
            vrp = model_hist["vrp"].dropna()
            if not vrp.empty:
                fig.add_trace(go.Scatter(
                    x=model_hist.loc[vrp.index, "trade_date"],
                    y=vrp * 100,
                    name="VRP", line=dict(color="#9b59b6"),
                    fill="tozeroy",
                ), row=2, col=1)

        # Greeks
        for col_name, label, color, r, c in [
            ("net_delta", "Delta", "#e74c3c", 2, 2),
            ("net_theta", "Theta", "#2ecc71", 3, 1),
            ("net_vega", "Vega", "#3498db", 3, 2),
        ]:
            if col_name in model_hist.columns:
                vals = model_hist[col_name].dropna()
                if not vals.empty:
                    fig.add_trace(go.Scatter(
                        x=model_hist.loc[vals.index, "trade_date"],
                        y=vals,
                        name=label, line=dict(color=color),
                    ), row=r, col=c)

    fig.update_layout(
        height=800,
        plot_bgcolor=_BG, paper_bgcolor=_BG,
        showlegend=False,
        margin=dict(l=40, r=20, t=40, b=30),
    )
    for i in range(1, 4):
        for j in range(1, 3):
            fig.update_xaxes(gridcolor=_GRID, row=i, col=j)
            fig.update_yaxes(gridcolor=_GRID, row=i, col=j)

    st.plotly_chart(fig, use_container_width=True)


# ── 主渲染 ────────────────────────────────────────────

def render():
    st.title("每日报告")

    db = _get_db()
    if db is None:
        return

    dates = _available_dates(db)
    if not dates:
        st.info("暂无报告数据")
        return

    tab_single, tab_compare = st.tabs(["单日报告", "多日对比"])

    with tab_single:
        selected = st.selectbox("选择日期", dates, index=0)
        if selected:
            _render_single_day(db, selected)

    with tab_compare:
        st.markdown("选择日期范围进行多指标对比")
        col1, col2 = st.columns(2)
        with col1:
            start_idx = min(len(dates) - 1, 5)
            start_date = st.selectbox("起始日期", dates, index=start_idx, key="cmp_start")
        with col2:
            end_date = st.selectbox("结束日期", dates, index=0, key="cmp_end")

        if start_date and end_date:
            range_dates = [d for d in dates if start_date >= d >= end_date]
            if range_dates:
                _render_comparison(db, range_dates)
