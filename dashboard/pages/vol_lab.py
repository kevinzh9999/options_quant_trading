"""
vol_lab.py
----------
Dashboard 页面：波动率研究 (Volatility Lab)

IV历史走势、Skew历史、期限结构快照对比、IV锥。
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

def _to_date(df: pd.DataFrame) -> pd.DataFrame:
    """Convert trade_date string (YYYYMMDD) to datetime for proper X-axis."""
    if "trade_date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
    return df


def _load_model_history(db, lookback: int = 900) -> pd.DataFrame:
    """优先从 volatility_history 读取（886天），fallback daily_model_output。"""
    df = db.query_df(
        f"SELECT * FROM volatility_history "
        f"WHERE atm_iv IS NOT NULL AND atm_iv > 0 "
        f"ORDER BY trade_date DESC LIMIT {lookback}"
    )
    if df is not None and len(df) > 20:
        return _to_date(df.sort_values("trade_date").reset_index(drop=True))
    # fallback
    df = db.query_df(
        f"SELECT * FROM daily_model_output "
        f"WHERE underlying='IM' ORDER BY trade_date DESC LIMIT {lookback}"
    )
    if df is not None and not df.empty:
        return _to_date(df.sort_values("trade_date").reset_index(drop=True))
    return pd.DataFrame()


def _load_vol_snapshots(db, date_prefix: str) -> pd.DataFrame:
    df = db.query_df(
        f"SELECT * FROM vol_monitor_snapshots "
        f"WHERE datetime LIKE '{date_prefix}%' ORDER BY datetime"
    )
    return df if df is not None else pd.DataFrame()


def _load_im_closes(db, lookback: int = 600) -> pd.DataFrame:
    df = db.query_df(
        f"SELECT trade_date, close FROM futures_daily "
        f"WHERE ts_code='IM.CFX' ORDER BY trade_date DESC LIMIT {lookback}"
    )
    if df is not None and not df.empty:
        return df.sort_values("trade_date").reset_index(drop=True)
    return pd.DataFrame()


def _load_snapshot_dates(db) -> list[str]:
    """获取有 vol_monitor_snapshots 数据的日期"""
    df = db.query_df(
        "SELECT DISTINCT substr(datetime, 1, 10) as date "
        "FROM vol_monitor_snapshots ORDER BY date DESC"
    )
    if df is not None and not df.empty:
        return df["date"].tolist()
    return []


# ── IV 历史走势 ───────────────────────────────────────

def _render_iv_history(model_hist: pd.DataFrame):
    st.subheader("IV 历史走势")

    if model_hist.empty:
        st.info("暂无历史数据")
        return

    fig = go.Figure()

    # volatility_history: atm_iv/garch_sigma/rv_20d in percent
    # daily_model_output: atm_iv_market/garch_forecast_vol/realized_vol_20d in decimal
    series = [
        ("atm_iv", "ATM IV", "#e67e22", None),
        ("atm_iv_market", "Market IV", "#e67e22", None),  # fallback
        ("garch_sigma", "GARCH", "#9b59b6", "dot"),
        ("garch_forecast_vol", "GARCH", "#9b59b6", "dot"),  # fallback
        ("rv_20d", "RV20", "#95a5a6", "dash"),
        ("realized_vol_20d", "RV20", "#95a5a6", "dash"),  # fallback
    ]

    shown_names = set()
    for col, name, color, dash in series:
        if name in shown_names:
            continue
        if col in model_hist.columns:
            vals = model_hist[col].dropna()
            if not vals.empty:
                # Auto-detect units: if mean < 1 → decimal → ×100
                y_vals = vals * 100 if vals.mean() < 1 else vals
                fig.add_trace(go.Scatter(
                    x=model_hist.loc[vals.index, "date"],
                    y=y_vals,
                    name=name,
                    line=dict(color=color, dash=dash),
                ))
                shown_names.add(name)

    fig.update_layout(
        title="Market IV / 结构IV / GARCH预测 / RV20 叠加",
        height=400,
        plot_bgcolor=_BG, paper_bgcolor=_BG,
        xaxis=dict(gridcolor=_GRID),
        yaxis=dict(title="波动率 (%)", gridcolor=_GRID),
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=40, r=20, t=50, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)

    # VRP 分布 (vrp_rv from volatility_history, or vrp from daily_model_output)
    vrp_col = "vrp_rv" if "vrp_rv" in model_hist.columns else "vrp"
    if vrp_col in model_hist.columns:
        vrp_raw = model_hist[vrp_col].dropna()
        # volatility_history vrp_rv is in percent (e.g. 1.91)
        # daily_model_output vrp is decimal (e.g. 0.0191)
        vrp = vrp_raw * 100 if vrp_raw.abs().mean() < 0.5 else vrp_raw

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**VRP 历史走势**")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=model_hist.loc[vrp_raw.index, "date"],
                y=vrp,
                name="VRP",
                fill="tozeroy",
                fillcolor="rgba(155,89,182,0.1)",
                line=dict(color="#9b59b6"),
            ))
            fig2.add_hline(y=0, line_color="gray", opacity=0.3)
            fig2.update_layout(
                height=280,
                plot_bgcolor=_BG, paper_bgcolor=_BG,
                xaxis=dict(gridcolor=_GRID),
                yaxis=dict(title="VRP (pp)", gridcolor=_GRID),
                margin=dict(l=40, r=20, t=30, b=30),
            )
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            st.markdown("**VRP 分布**")
            fig3 = go.Figure()
            fig3.add_trace(go.Histogram(
                x=vrp, nbinsx=30,
                marker_color="#9b59b6", opacity=0.7,
            ))
            if not vrp.empty:
                current = vrp.iloc[-1]
                fig3.add_vline(x=current, line_dash="dash", line_color="white",
                               annotation_text=f"当前 {current:.1f}pp")
            fig3.update_layout(
                height=280,
                plot_bgcolor=_BG, paper_bgcolor=_BG,
                xaxis=dict(title="VRP (pp)", gridcolor=_GRID),
                yaxis=dict(title="频次", gridcolor=_GRID),
                margin=dict(l=40, r=20, t=30, b=30),
            )
            st.plotly_chart(fig3, use_container_width=True)


# ── Skew 历史 ─────────────────────────────────────────

def _render_skew_history(db):
    st.subheader("Skew 历史")

    vol_snaps = db.query_df(
        "SELECT datetime, rr_25d, bf_25d FROM vol_monitor_snapshots ORDER BY datetime"
    )
    if vol_snaps is None or vol_snaps.empty:
        st.info("暂无 Skew 数据（来自 vol_monitor_snapshots 表）")
        return

    # 每日取最后一条作为日线数据
    vol_snaps["date"] = vol_snaps["datetime"].str[:10]
    daily = vol_snaps.groupby("date").last().reset_index()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**25D Risk Reversal (RR)**")
        fig = go.Figure()
        if "rr_25d" in daily.columns:
            rr = daily["rr_25d"].dropna() * 100
            if not rr.empty:
                fig.add_trace(go.Scatter(
                    x=daily.loc[rr.index, "date"], y=rr,
                    name="25D RR", line=dict(color="#e74c3c"),
                    fill="tozeroy", fillcolor="rgba(231,76,60,0.1)",
                ))
        fig.add_hline(y=0, line_color="gray", opacity=0.3)
        fig.update_layout(
            height=300,
            plot_bgcolor=_BG, paper_bgcolor=_BG,
            xaxis=dict(gridcolor=_GRID),
            yaxis=dict(title="RR (%)", gridcolor=_GRID),
            margin=dict(l=40, r=20, t=30, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)
        if not rr.empty:
            st.caption(f"当前 RR = {rr.iloc[-1]:.2f}pp | 正值=Put偏贵（看空偏斜）")

    with col2:
        st.markdown("**25D Butterfly (BF)**")
        fig = go.Figure()
        if "bf_25d" in daily.columns:
            bf = daily["bf_25d"].dropna() * 100
            if not bf.empty:
                fig.add_trace(go.Scatter(
                    x=daily.loc[bf.index, "date"], y=bf,
                    name="25D BF", line=dict(color="#2ecc71"),
                    fill="tozeroy", fillcolor="rgba(46,204,113,0.1)",
                ))
        fig.update_layout(
            height=300,
            plot_bgcolor=_BG, paper_bgcolor=_BG,
            xaxis=dict(gridcolor=_GRID),
            yaxis=dict(title="BF (%)", gridcolor=_GRID),
            margin=dict(l=40, r=20, t=30, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)
        if "bf" in dir() and not bf.empty:
            st.caption(f"当前 BF = {bf.iloc[-1]:.2f}pp | 高值=尾部风险定价高")


# ── 期限结构历史快照 ──────────────────────────────────

def _render_term_structure_snapshot(db):
    st.subheader("期限结构历史快照")

    dates = _load_snapshot_dates(db)
    if not dates:
        st.info("暂无期限结构数据")
        return

    col1, col2 = st.columns(2)
    with col1:
        date1 = st.selectbox("日期1 (蓝)", dates, index=0, key="ts_d1")
    with col2:
        date2 = st.selectbox("日期2 (橙)", dates, index=min(1, len(dates) - 1), key="ts_d2")

    snap1 = _load_vol_snapshots(db, date1)
    snap2 = _load_vol_snapshots(db, date2)

    fig = go.Figure()

    for snap, label, color in [(snap1, date1, "#3498db"), (snap2, date2, "#e67e22")]:
        if snap.empty:
            continue
        last = snap.iloc[-1]
        months = []
        ivs = []
        for col in ["iv_m1", "iv_m2", "iv_m3"]:
            val = last.get(col)
            if val is not None and not pd.isna(val):
                months.append(col.replace("iv_", "").upper())
                ivs.append(float(val) * 100)
        if months:
            fig.add_trace(go.Bar(
                x=months, y=ivs,
                name=label,
                marker_color=color,
                opacity=0.7,
                text=[f"{v:.1f}%" for v in ivs],
                textposition="outside",
            ))

    fig.update_layout(
        title="期限结构对比",
        height=350,
        barmode="group",
        plot_bgcolor=_BG, paper_bgcolor=_BG,
        xaxis=dict(title="到期月份", gridcolor=_GRID),
        yaxis=dict(title="IV (%)", gridcolor=_GRID),
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=40, r=20, t=50, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 变化明细
    if not snap1.empty and not snap2.empty:
        l1 = snap1.iloc[-1]
        l2 = snap2.iloc[-1]
        diffs = []
        for col in ["iv_m1", "iv_m2", "iv_m3"]:
            v1 = l1.get(col)
            v2 = l2.get(col)
            if v1 is not None and v2 is not None:
                try:
                    diff = (float(v1) - float(v2)) * 100
                    diffs.append({"月份": col.replace("iv_", "").upper(),
                                  f"{date1}": f"{float(v1)*100:.1f}%",
                                  f"{date2}": f"{float(v2)*100:.1f}%",
                                  "变化": f"{diff:+.2f}pp"})
                except (ValueError, TypeError):
                    pass
        if diffs:
            st.dataframe(pd.DataFrame(diffs), use_container_width=True, hide_index=True)


# ── IV 锥 ────────────────────────────────────────────

def _render_vol_cone(im_closes: pd.DataFrame, model_hist: pd.DataFrame):
    st.subheader("IV 锥 (Volatility Cone)")

    if im_closes.empty or len(im_closes) < 60:
        st.info("IM日线数据不足")
        return

    closes = im_closes["close"].astype(float).values
    log_ret = np.diff(np.log(closes))

    windows = [5, 10, 20, 60]
    percentiles = [10, 25, 50, 75, 90]

    # 计算各窗口的RV分布
    cone_data = {}
    for w in windows:
        rvs = []
        for i in range(w, len(log_ret)):
            rv = np.std(log_ret[i - w:i]) * np.sqrt(252) * 100
            rvs.append(rv)
        if rvs:
            cone_data[w] = {
                p: float(np.percentile(rvs, p))
                for p in percentiles
            }
            cone_data[w]["current"] = rvs[-1] if rvs else 0

    if not cone_data:
        st.info("计算数据不足")
        return

    # 当前 IV
    current_iv = None
    if not model_hist.empty:
        for col in ["atm_iv", "atm_iv_market"]:
            if col in model_hist.columns:
                val = model_hist.iloc[-1].get(col)
                if val is not None:
                    try:
                        fv = float(val)
                        current_iv = fv if fv > 1 else fv * 100
                        break
                    except (ValueError, TypeError):
                        pass

    fig = go.Figure()

    # 锥的各分位线
    x_vals = windows
    colors = {
        10: "rgba(231,76,60,0.2)",
        25: "rgba(241,196,15,0.2)",
        50: "rgba(52,152,219,0.5)",
        75: "rgba(241,196,15,0.2)",
        90: "rgba(231,76,60,0.2)",
    }

    # 填充区域
    for p in [90, 75]:
        upper = [cone_data[w][p] for w in x_vals]
        lower = [cone_data[w][100 - p] for w in x_vals]
        fig.add_trace(go.Scatter(
            x=x_vals, y=upper,
            mode="lines", line=dict(width=0),
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=x_vals, y=lower,
            mode="lines", line=dict(width=0),
            fill="tonexty",
            fillcolor=colors[p],
            name=f"P{100-p}-P{p}",
        ))

    # 中位线
    fig.add_trace(go.Scatter(
        x=x_vals, y=[cone_data[w][50] for w in x_vals],
        mode="lines+markers",
        name="中位 (P50)",
        line=dict(color="#3498db", width=2),
    ))

    # 当前RV
    fig.add_trace(go.Scatter(
        x=x_vals, y=[cone_data[w]["current"] for w in x_vals],
        mode="lines+markers",
        name="当前RV",
        line=dict(color="white", width=2, dash="dash"),
        marker=dict(size=8),
    ))

    # 当前IV
    if current_iv is not None:
        fig.add_hline(y=current_iv, line_dash="dot", line_color="#e67e22",
                      annotation_text=f"当前IV {current_iv:.1f}%")

    fig.update_layout(
        title="不同回看期的RV分布 + 当前IV位置",
        height=400,
        plot_bgcolor=_BG, paper_bgcolor=_BG,
        xaxis=dict(
            title="回看期 (天)",
            tickvals=windows,
            ticktext=[f"{w}D" for w in windows],
            gridcolor=_GRID,
        ),
        yaxis=dict(title="波动率 (%)", gridcolor=_GRID),
        legend=dict(orientation="h", y=1.02),
        margin=dict(l=40, r=20, t=50, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)

    # 数据表
    rows = []
    for w in windows:
        row = {"回看期": f"{w}D"}
        for p in percentiles:
            row[f"P{p}"] = f"{cone_data[w][p]:.1f}%"
        row["当前RV"] = f"{cone_data[w]['current']:.1f}%"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if current_iv is not None:
        # 当前IV在RV20分布中的位置
        rv20_dist = []
        for i in range(20, len(log_ret)):
            rv20_dist.append(np.std(log_ret[i - 20:i]) * np.sqrt(252) * 100)
        if rv20_dist:
            pct = np.mean(np.array(rv20_dist) <= current_iv) * 100
            st.caption(f"当前IV ({current_iv:.1f}%) 在RV20历史分布中的百分位: **{pct:.0f}%**")


# ── Z-Score 历史 ──────────────────────────────────────

def _render_zscore_history(model_hist: pd.DataFrame, db):
    st.subheader("Z-Score 历史 (IM via 000852.SH)")

    # 优先从 volatility_history 的 spot_zscore 列
    if "spot_zscore" in model_hist.columns and "date" in model_hist.columns:
        z_data = model_hist[["date", "spot_zscore"]].dropna(subset=["spot_zscore", "date"])
        if not z_data.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=z_data["date"], y=z_data["spot_zscore"],
                name="Z-Score", line=dict(color="#3498db"),
            ))
            fig.add_hline(y=-2, line_dash="dash", line_color="#e74c3c", opacity=0.7,
                          annotation_text="Z=-2 (超卖)")
            fig.add_hline(y=2, line_dash="dash", line_color="#2ecc71", opacity=0.7,
                          annotation_text="Z=+2 (超买)")
            fig.add_hline(y=0, line_color="gray", opacity=0.3)

            # 标注Z<-2和Z>+2的区域
            for _, row in z_data.iterrows():
                z = row["spot_zscore"]
                if z < -2:
                    fig.add_vrect(
                        x0=row["date"], x1=row["date"],
                        fillcolor="rgba(231,76,60,0.15)", line_width=0,
                    )
                elif z > 2:
                    fig.add_vrect(
                        x0=row["date"], x1=row["date"],
                        fillcolor="rgba(46,204,113,0.15)", line_width=0,
                    )

            fig.update_layout(
                title="Z-Score = (现货 - EMA20) / STD20",
                height=400,
                plot_bgcolor=_BG, paper_bgcolor=_BG,
                xaxis=dict(gridcolor=_GRID),
                yaxis=dict(title="Z-Score", gridcolor=_GRID),
                margin=dict(l=40, r=20, t=50, b=30),
            )
            st.plotly_chart(fig, use_container_width=True)

            # 统计
            z = z_data["spot_zscore"]
            current = z.iloc[-1]
            below_m2 = (z < -2).sum()
            above_p2 = (z > 2).sum()
            st.caption(
                f"当前 Z={current:+.2f} | "
                f"Z<-2 出现 {below_m2} 次 ({below_m2/len(z)*100:.1f}%) | "
                f"Z>+2 出现 {above_p2} 次 ({above_p2/len(z)*100:.1f}%) | "
                f"共 {len(z)} 天"
            )
            return

    # fallback: 从000852.SH日线计算
    spot_df = db.query_df(
        "SELECT trade_date, close FROM index_daily "
        "WHERE ts_code='000852.SH' AND close > 0 ORDER BY trade_date"
    )
    if spot_df is None or len(spot_df) < 30:
        st.info("数据不足")
        return

    spot_df["close"] = spot_df["close"].astype(float)
    spot_df["ema20"] = spot_df["close"].ewm(span=20).mean()
    spot_df["std20"] = spot_df["close"].rolling(20).std()
    spot_df["zscore"] = (spot_df["close"] - spot_df["ema20"]) / spot_df["std20"]
    spot_df = spot_df.dropna()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=spot_df["trade_date"], y=spot_df["zscore"],
        name="Z-Score", line=dict(color="#3498db"),
    ))
    fig.add_hline(y=-2, line_dash="dash", line_color="#e74c3c", opacity=0.7)
    fig.add_hline(y=2, line_dash="dash", line_color="#2ecc71", opacity=0.7)
    fig.add_hline(y=0, line_color="gray", opacity=0.3)
    fig.update_layout(
        title="Z-Score = (000852.SH - EMA20) / STD20",
        height=400,
        plot_bgcolor=_BG, paper_bgcolor=_BG,
        xaxis=dict(gridcolor=_GRID),
        yaxis=dict(title="Z-Score", gridcolor=_GRID),
        margin=dict(l=40, r=20, t=50, b=30),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── 主渲染 ────────────────────────────────────────────

def render():
    st.title("波动率研究")

    db = _get_db()
    if db is None:
        return

    model_hist = _load_model_history(db)
    im_closes = _load_im_closes(db)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["IV历史", "Skew历史", "期限结构", "IV锥", "Z-Score"])

    with tab1:
        _render_iv_history(model_hist)

    with tab2:
        _render_skew_history(db)

    with tab3:
        _render_term_structure_snapshot(db)

    with tab4:
        _render_vol_cone(im_closes, model_hist)

    with tab5:
        _render_zscore_history(model_hist, db)
