"""
vol_monitor.py
--------------
Dashboard 页面：波动率监控

功能:
- RV vs ATM IV 历史对比（含 VRP 填充区域）
- GARCH 条件波动率历史
- 波动率期限结构（柱状图）
- 波动率微笑（按到期月分色）
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
from datetime import datetime


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


def _load_futures_daily(db, ts_code: str, days: int) -> pd.DataFrame:
    try:
        df = db.query_df(
            f"""SELECT trade_date, open, high, low, close, volume, oi
                FROM futures_daily
                WHERE ts_code = '{ts_code}'
                ORDER BY trade_date DESC
                LIMIT {days}"""
        )
        if df is None or df.empty:
            return pd.DataFrame()
        return df.sort_values("trade_date").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def _compute_rv(_close_tuple):
    """Compute RV20 and RV60. Accept tuple for hashability."""
    close = pd.Series(_close_tuple)
    results = {}
    try:
        from models.volatility.realized_vol import RealizedVolCalculator
        for w in (20, 60):
            rv = RealizedVolCalculator.from_daily(close, window=w)
            results[w] = list(rv.values)
    except Exception:
        log_ret = np.log(close / close.shift(1))
        for w in (20, 60):
            rv = log_ret.rolling(w).std() * np.sqrt(252)
            results[w] = list(rv.values)
    return results


def _load_atm_iv_history(db, days: int) -> pd.DataFrame:
    """优先从 volatility_history 读取（886天预计算数据）。"""
    try:
        df = db.query_df(
            f"SELECT trade_date, atm_iv, garch_sigma, rv_20d, vrp_rv as vrp "
            f"FROM volatility_history "
            f"WHERE atm_iv IS NOT NULL AND atm_iv > 0 "
            f"ORDER BY trade_date DESC LIMIT {days}"
        )
        if df is not None and len(df) > 20:
            df = df.sort_values("trade_date").reset_index(drop=True)
            df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
            return df
    except Exception:
        pass
    # fallback
    try:
        df = db.query_df(
            f"SELECT trade_date, atm_iv, garch_forecast_vol as garch_sigma, "
            f"realized_vol_20d as rv_20d, vrp "
            f"FROM daily_model_output "
            f"WHERE underlying='IM' ORDER BY trade_date DESC LIMIT {days}"
        )
        if df is not None and not df.empty:
            df = df.sort_values("trade_date").reset_index(drop=True)
            df["date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
            return df
    except Exception:
        pass
    return pd.DataFrame()


def _load_mo_chain_latest(db) -> pd.DataFrame:
    try:
        df = db.query_df(
            """SELECT od.ts_code, od.trade_date, od.close, od.settle,
                      oc.exercise_price, oc.call_put, oc.delist_date as expire_date,
                      oc.underlying_symbol
               FROM options_daily od
               JOIN options_contracts oc ON oc.ts_code = od.ts_code
               WHERE od.trade_date = (SELECT MAX(trade_date) FROM options_daily)
                 AND od.close > 0
               ORDER BY oc.exercise_price ASC"""
        )
        if df is None or df.empty:
            return pd.DataFrame()
        df["expire_date"] = df["expire_date"].astype(str)
        df["expire_month"] = df["expire_date"].str[:6]
        return df
    except Exception:
        return pd.DataFrame()


def _calc_iv_for_chain(chain_df: pd.DataFrame, spot: float) -> pd.DataFrame:
    if chain_df.empty or spot <= 0:
        return chain_df
    try:
        from models.pricing.black_scholes import BlackScholes
        today = datetime.now()
        ivs = []
        for _, row in chain_df.iterrows():
            try:
                expire_str = str(row.get("expire_date", ""))
                if len(expire_str) >= 8:
                    exp_date = datetime.strptime(expire_str[:8], "%Y%m%d")
                    T = max((exp_date - today).days / 365.0, 1 / 365)
                else:
                    T = 30 / 365
                K = float(row.get("exercise_price", spot))
                price = float(row.get("close", 0) or 0)
                cp = str(row.get("call_put", "C")).upper()
                if price > 0 and K > 0 and T > 0:
                    iv = BlackScholes.implied_volatility(
                        market_price=price, S=spot, K=K, T=T,
                        r=0.02, q=0, option_type=cp
                    )
                    ivs.append(iv if iv and 0.001 < iv < 5.0 else np.nan)
                else:
                    ivs.append(np.nan)
            except Exception:
                ivs.append(np.nan)
        chain_df = chain_df.copy()
        chain_df["iv"] = ivs
        return chain_df
    except Exception:
        return chain_df


@st.cache_data(ttl=3600)
def _fit_garch_cached(_close_tuple):
    """Fit GJR-GARCH on close prices."""
    close = pd.Series(_close_tuple)
    try:
        from models.volatility.garch_model import GJRGARCHModel
        log_ret = np.log(close / close.shift(1)).dropna() * 100
        model = GJRGARCHModel()
        fit = model.fit(log_ret)
        cond_vol = model.get_conditional_vol()
        params = fit.params if hasattr(fit, "params") else {}
        persistence = fit.persistence if hasattr(fit, "persistence") else np.nan
        garch_5d = model.forecast_period_avg(horizon=5)
        return {
            "cond_vol": list(cond_vol.values) if hasattr(cond_vol, "values") else list(cond_vol),
            "params": dict(params) if params else {},
            "persistence": float(persistence) if not np.isnan(float(persistence if persistence is not None else np.nan)) else np.nan,
            "garch_5d": float(garch_5d) if garch_5d is not None else np.nan,
            "converged": bool(fit.converged) if hasattr(fit, "converged") else True,
        }
    except Exception as e:
        return {"error": str(e)}


def _days_from_range(r: str) -> int:
    return {"近3月": 63, "近6月": 126, "近1年": 252, "全部": 2000}.get(r, 126)


# ──────────────────────────────────────────────
# Render
# ──────────────────────────────────────────────

def render() -> None:
    st.title("📈 波动率监控")

    db = _get_db()
    if db is None:
        st.error("无法连接数据库，请检查配置。")
        return

    ctrl1, ctrl2 = st.columns(2)
    with ctrl1:
        product = st.selectbox("品种", ["IM", "IF", "IH", "IC"], index=0)
    with ctrl2:
        time_range = st.selectbox("时间范围", ["近3月", "近6月", "近1年", "全部"], index=1)

    days = _days_from_range(time_range)
    ts_code = f"{product}.CFX"

    futures_df = _load_futures_daily(db, ts_code, days)
    model_history = _load_atm_iv_history(db, days)

    latest_date = futures_df["trade_date"].iloc[-1] if not futures_df.empty else "N/A"
    st.caption(f"数据更新至: {latest_date}")

    spot = float(futures_df["close"].iloc[-1]) if not futures_df.empty else 0.0

    # ── Chart 1: RV vs ATM IV (from volatility_history) ──
    st.subheader("RV vs ATM IV 历史对比")

    if model_history.empty or len(model_history) < 5:
        st.info("波动率数据不足（需要运行 volatility_history_research.py）")
    else:
        h = model_history
        x_col = "date" if "date" in h.columns else "trade_date"
        fig1 = go.Figure()

        # RV20
        if "rv_20d" in h.columns:
            rv = h["rv_20d"].dropna()
            y = rv if rv.mean() > 1 else rv * 100
            fig1.add_trace(go.Scatter(
                x=h.loc[rv.index, x_col], y=y,
                mode="lines", name="RV20", line=dict(color="#3498db", width=1.5),
            ))

        # ATM IV
        if "atm_iv" in h.columns:
            iv = h["atm_iv"].dropna()
            y = iv if iv.mean() > 1 else iv * 100
            fig1.add_trace(go.Scatter(
                x=h.loc[iv.index, x_col], y=y,
                mode="lines", name="ATM IV", line=dict(color="#e74c3c", width=2),
            ))

        # GARCH
        if "garch_sigma" in h.columns:
            gs = h["garch_sigma"].dropna()
            if not gs.empty:
                y = gs if gs.mean() > 1 else gs * 100
                fig1.add_trace(go.Scatter(
                    x=h.loc[gs.index, x_col], y=y,
                    mode="lines", name="GARCH", line=dict(color="#9b59b6", width=1, dash="dot"),
                ))

        # VRP as shaded area between IV and RV (not raw VRP values which blow up scale)
        if "atm_iv" in h.columns and "rv_20d" in h.columns:
            both = h.dropna(subset=["atm_iv", "rv_20d"])
            if not both.empty:
                iv_y = both["atm_iv"] if both["atm_iv"].mean() > 1 else both["atm_iv"] * 100
                rv_y = both["rv_20d"] if both["rv_20d"].mean() > 1 else both["rv_20d"] * 100
                # Fill between IV and RV: green when IV>RV (VRP>0), red when IV<RV
                fig1.add_trace(go.Scatter(
                    x=both[x_col], y=iv_y, mode="lines", line=dict(width=0),
                    showlegend=False,
                ))
                fig1.add_trace(go.Scatter(
                    x=both[x_col], y=rv_y, mode="lines", line=dict(width=0),
                    fill="tonexty", fillcolor="rgba(144,238,144,0.15)",
                    name="VRP区域", showlegend=True,
                ))

        fig1.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)", title="波动率 (%)"),
            legend=dict(orientation="h", y=1.02),
            margin=dict(l=10, r=10, t=40, b=10), height=350,
        )
        st.plotly_chart(fig1, use_container_width=True)

    # ── Chart 2: GARCH Conditional Vol ──
    st.subheader("GARCH 条件波动率")
    garch_data = None

    if not model_history.empty and "garch_sigma" in model_history.columns:
        gs = model_history["garch_sigma"].dropna()
        if not gs.empty:
            x_col = "date" if "date" in model_history.columns else "trade_date"
            y = gs if gs.mean() > 1 else gs * 100
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=model_history.loc[gs.index, x_col], y=y,
                mode="lines", name="GARCH条件波动率",
                line=dict(color="#9b59b6", width=2),
            ))
            long_run = float(y.mean())
            fig2.add_hline(y=long_run, line_dash="dash",
                           line_color="rgba(255,200,0,0.7)",
                           annotation_text=f"均值 {long_run:.1f}%")
            fig2.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)", title="条件波动率 (%)"),
                margin=dict(l=10, r=10, t=30, b=10), height=300,
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("GARCH数据为空")
    else:
        st.info("无GARCH数据（需要运行 volatility_history_research.py）")

    # ── Charts 3 & 4: Term Structure & Vol Skew ──
    st.subheader("波动率期限结构 & 微笑")

    chain_df = _load_mo_chain_latest(db)
    if chain_df.empty:
        st.info("暂无期权链数据（需要 options_daily 及 options_contracts 数据）")
    else:
        chain_with_iv = _calc_iv_for_chain(chain_df, spot)
        chain_valid = chain_with_iv.dropna(subset=["iv"])

        if chain_valid.empty:
            st.info("期权 IV 计算结果全部无效，请检查期权数据")
        else:
            chart_c1, chart_c2 = st.columns(2)

            # Chart 3: Term Structure
            with chart_c1:
                st.markdown("**期限结构 (ATM IV by 到期月)**")
                if "expire_month" in chain_valid.columns:
                    atm_rows = []
                    for month, grp in chain_valid.groupby("expire_month"):
                        grp = grp.copy()
                        if spot > 0:
                            grp["_dist"] = (grp["exercise_price"].astype(float) - spot).abs()
                            atm_idx = grp["_dist"].idxmin()
                            atm_iv = grp.loc[atm_idx, "iv"]
                        else:
                            atm_iv = grp["iv"].mean()
                        atm_rows.append({"月份": month, "ATM_IV": atm_iv * 100})
                    ts_manual = pd.DataFrame(atm_rows).sort_values("月份")
                    if not ts_manual.empty:
                        vals = ts_manual["ATM_IV"].values
                        is_contango = all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1)) if len(vals) > 1 else True
                        bar_color = "#3498db" if is_contango else "#e74c3c"
                        fig3 = go.Figure(go.Bar(
                            x=ts_manual["月份"].tolist(),
                            y=ts_manual["ATM_IV"].tolist(),
                            marker_color=bar_color,
                            name="ATM IV",
                        ))
                        fig3.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            xaxis_title="到期月",
                            yaxis_title="ATM IV (%)",
                            margin=dict(l=10, r=10, t=30, b=10),
                            height=300,
                        )
                        st.plotly_chart(fig3, use_container_width=True)
                        st.caption("蓝色=正向结构 | 红色=反向结构")
                    else:
                        st.info("期限结构数据不足")
                else:
                    st.info("期权数据缺少到期月信息")

            # Chart 4: Vol Skew
            with chart_c2:
                st.markdown("**波动率微笑 (按到期月)**")
                colors = ["#3498db", "#e74c3c", "#2ecc71", "#e67e22", "#9b59b6", "#1abc9c"]
                fig4 = go.Figure()
                x_label = "Moneyness (K/F)" if spot > 0 else "行权价"

                if "expire_month" in chain_valid.columns:
                    months = sorted(chain_valid["expire_month"].unique())
                    for i, month in enumerate(months[:6]):
                        grp = chain_valid[chain_valid["expire_month"] == month].copy()
                        if len(grp) < 2:
                            continue
                        grp_sorted = grp.sort_values("exercise_price")
                        if spot > 0:
                            x_vals = (grp_sorted["exercise_price"].astype(float) / spot).tolist()
                        else:
                            x_vals = grp_sorted["exercise_price"].astype(float).tolist()
                        y_vals = (grp_sorted["iv"] * 100).tolist()
                        fig4.add_trace(go.Scatter(
                            x=x_vals, y=y_vals,
                            mode="lines+markers",
                            name=month,
                            line=dict(color=colors[i % len(colors)]),
                        ))
                    fig4.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        xaxis_title=x_label,
                        yaxis_title="IV (%)",
                        legend=dict(orientation="h", y=1.02),
                        margin=dict(l=10, r=10, t=30, b=10),
                        height=300,
                    )
                    st.plotly_chart(fig4, use_container_width=True)
                else:
                    st.info("缺少到期月数据")

    # ── Info expander ──
    with st.expander("模型参数详情", expanded=False):
        ec1, ec2 = st.columns(2)
        with ec1:
            st.markdown("**VRP 当前信号**")
            if not model_history.empty:
                row = model_history.iloc[-1]
                vrp_val = row.get("vrp")
                sig = row.get("signal")
                atm_iv_val = row.get("atm_iv")
                rv22 = row.get("realized_vol_20d")
                if vrp_val is not None and not pd.isna(vrp_val):
                    st.metric("VRP", f"{vrp_val:.4f}")
                if atm_iv_val is not None and not pd.isna(atm_iv_val):
                    st.metric("ATM IV", f"{atm_iv_val*100:.2f}%")
                if rv22 is not None and not pd.isna(rv22):
                    st.metric("RV22", f"{rv22*100:.2f}%")
                if sig and not pd.isna(sig):
                    st.info(f"信号: {sig}")
            else:
                st.info("暂无 VRP 数据")

        with ec2:
            st.markdown("**GARCH 参数**")
            if garch_data and "error" not in garch_data:
                params = garch_data.get("params", {})
                persistence = garch_data.get("persistence", np.nan)
                converged = garch_data.get("converged", False)
                garch_5d = garch_data.get("garch_5d", np.nan)
                for k, v in params.items():
                    if isinstance(v, (int, float)):
                        st.write(f"**{k}**: {v:.6f}")
                    else:
                        st.write(f"**{k}**: {v}")
                if isinstance(persistence, float) and not np.isnan(persistence):
                    st.write(f"**Persistence**: {persistence:.6f}")
                    if persistence < 1 and persistence > 0:
                        half_life = np.log(2) / np.log(1 / persistence)
                        st.write(f"**波动率半衰期**: {half_life:.1f} 天")
                if isinstance(garch_5d, float) and not np.isnan(garch_5d):
                    st.write(f"**GARCH 5日预测**: {garch_5d*100:.2f}%")
                st.write(f"**收敛**: {'是' if converged else '否'}")
            else:
                st.info("GARCH 参数不可用（数据不足或拟合失败）")


if __name__ == "__main__":
    render()
