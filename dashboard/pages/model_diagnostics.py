"""
model_diagnostics.py
--------------------
Dashboard 页面：模型诊断

功能:
- GJR-GARCH 参数展示及收敛状态
- 预测 RV vs 实际 RV 散点图
- 标准化残差分布（直方图 + 正态曲线）
- 残差 QQ 图
- 预测误差时间序列
- Ljung-Box / ARCH-LM 诊断检验
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


@st.cache_data(ttl=3600)
def _fit_garch_cached(_close_tuple):
    """Fit GJR-GARCH on IM.CFX close, return diagnostics dict."""
    close = pd.Series(_close_tuple)
    try:
        from models.volatility.garch_model import GJRGARCHModel
        log_ret = np.log(close / close.shift(1)).dropna() * 100
        model = GJRGARCHModel()
        fit = model.fit(log_ret)

        cond_vol = model.get_conditional_vol()
        std_resid = model.get_standardized_residuals()
        diag = model.diagnose()

        params = dict(fit.params) if hasattr(fit, "params") else {}
        persistence = float(fit.persistence) if hasattr(fit, "persistence") else np.nan
        converged = bool(fit.converged) if hasattr(fit, "converged") else True

        # Log-likelihood, AIC, BIC
        log_lik = float(fit.loglikelihood) if hasattr(fit, "loglikelihood") else np.nan
        aic = float(fit.aic) if hasattr(fit, "aic") else np.nan
        bic = float(fit.bic) if hasattr(fit, "bic") else np.nan

        return {
            "params": params,
            "persistence": persistence,
            "converged": converged,
            "log_lik": log_lik,
            "aic": aic,
            "bic": bic,
            "cond_vol": list(cond_vol.values) if hasattr(cond_vol, "values") else list(cond_vol),
            "std_resid": list(std_resid.values) if hasattr(std_resid, "values") else list(std_resid),
            "diagnostics": diag if isinstance(diag, dict) else {},
            "n_obs": len(log_ret),
        }
    except Exception as e:
        return {"error": str(e)}


def _load_model_history(db) -> pd.DataFrame:
    try:
        df = db.query_df(
            """SELECT trade_date, atm_iv, garch_forecast_vol, realized_vol_20d, vrp
               FROM daily_model_output
               ORDER BY trade_date ASC"""
        )
        if df is None:
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()


def _load_im_close(db, n: int = 500) -> pd.Series:
    try:
        df = db.query_df(
            f"""SELECT trade_date, close FROM futures_daily
                WHERE ts_code = 'IM.CFX'
                ORDER BY trade_date DESC LIMIT {n}"""
        )
        if df is None or df.empty:
            return pd.Series(dtype=float)
        df = df.sort_values("trade_date").reset_index(drop=True)
        return df.set_index("trade_date")["close"]
    except Exception:
        return pd.Series(dtype=float)


# ──────────────────────────────────────────────
# Render
# ──────────────────────────────────────────────

def render() -> None:
    st.title("🔬 模型诊断")

    db = _get_db()
    if db is None:
        st.error("无法连接数据库，请检查配置。")
        return

    close_series = _load_im_close(db)
    model_history = _load_model_history(db)

    latest_date = close_series.index[-1] if not close_series.empty else "N/A"
    st.caption(f"数据更新至: {latest_date}")

    # Fit GARCH
    garch_data = None
    if len(close_series) >= 30:
        garch_data = _fit_garch_cached(tuple(close_series.values.tolist()))
    else:
        st.warning(f"IM.CFX 数据不足（{len(close_series)} 行，需要 ≥ 30 行），无法拟合 GARCH")

    # ── Section 1: Model Params ──
    st.subheader("GARCH 模型参数")

    if garch_data and "error" not in garch_data:
        params = garch_data.get("params", {})
        persistence = garch_data.get("persistence", np.nan)
        converged = garch_data.get("converged", False)
        log_lik = garch_data.get("log_lik", np.nan)
        aic = garch_data.get("aic", np.nan)
        bic = garch_data.get("bic", np.nan)
        n_obs = garch_data.get("n_obs", 0)

        param_col, fit_col = st.columns(2)

        with param_col:
            st.markdown("**参数估计**")
            param_labels = {
                "omega": "ω (常数项)",
                "alpha": "α (ARCH 效应)",
                "gamma": "γ (杠杆效应)",
                "beta": "β (GARCH 效应)",
                "Const": "ω (常数项)",
                "alpha[1]": "α (ARCH 效应)",
                "gamma[1]": "γ (杠杆效应)",
                "beta[1]": "β (GARCH 效应)",
            }
            for k, v in params.items():
                label = param_labels.get(k, k)
                if isinstance(v, (int, float)) and not np.isnan(float(v)):
                    st.write(f"**{label}**: {float(v):.6f}")

            if not np.isnan(persistence):
                st.write(f"**Persistence** (α+β+γ/2): {persistence:.6f}")
                if 0 < persistence < 1:
                    half_life = np.log(2) / np.log(1 / persistence)
                    st.write(f"**波动率半衰期**: {half_life:.1f} 天")

                # Long-run vol
                omega = params.get("omega", params.get("Const", None))
                if omega and persistence < 1:
                    long_run_var = float(omega) / (1 - persistence)
                    long_run_vol = np.sqrt(long_run_var * 252) * 100
                    st.write(f"**长期均衡波动率**: {long_run_vol:.2f}%")

        with fit_col:
            st.markdown("**拟合质量**")
            st.write(f"**收敛**: {'✓ 是' if converged else '✗ 否'}")
            st.write(f"**观测数**: {n_obs}")
            if not np.isnan(log_lik):
                st.write(f"**对数似然**: {log_lik:.4f}")
            if not np.isnan(aic):
                st.write(f"**AIC**: {aic:.4f}")
            if not np.isnan(bic):
                st.write(f"**BIC**: {bic:.4f}")

            # Diagnostic tests
            st.markdown("**诊断检验**")
            diag = garch_data.get("diagnostics", {})

            lb_pval = None
            arch_pval = None

            if diag:
                # Handle various key names
                for key in ("ljung_box", "lb_pvalue", "lb_stat"):
                    if key in diag:
                        v = diag[key]
                        if isinstance(v, dict):
                            lb_pval = v.get("pvalue", v.get("p_value"))
                        elif isinstance(v, (int, float)):
                            lb_pval = v
                        break
                for key in ("arch_lm", "arch_pvalue", "lm_pvalue"):
                    if key in diag:
                        v = diag[key]
                        if isinstance(v, dict):
                            arch_pval = v.get("pvalue", v.get("p_value"))
                        elif isinstance(v, (int, float)):
                            arch_pval = v
                        break

            if lb_pval is not None:
                color = "green" if lb_pval > 0.05 else "red"
                status = "正常" if lb_pval > 0.05 else "有序列相关"
                st.markdown(
                    f"Ljung-Box p值: <span style='color:{color}'>{lb_pval:.4f} ({status})</span>",
                    unsafe_allow_html=True
                )
            else:
                st.write("Ljung-Box: 暂无数据")

            if arch_pval is not None:
                color = "green" if arch_pval > 0.05 else "red"
                status = "正常" if arch_pval > 0.05 else "仍有ARCH效应"
                st.markdown(
                    f"ARCH-LM p值: <span style='color:{color}'>{arch_pval:.4f} ({status})</span>",
                    unsafe_allow_html=True
                )
            else:
                st.write("ARCH-LM: 暂无数据")
    else:
        if garch_data and "error" in garch_data:
            st.error(f"GARCH 拟合失败: {garch_data['error']}")
        else:
            st.info("GARCH 参数不可用")

    st.divider()

    # ── Section 2: Predicted vs Actual RV Scatter ──
    st.subheader("GARCH 预测 vs 实际 RV 散点图")

    n_hist = len(model_history)
    if n_hist < 20:
        st.info(f"需要更多数据积累（当前 {n_hist} 天，需要 ≥ 20 天）")
    else:
        valid = model_history.dropna(subset=["garch_forecast_vol", "realized_vol_20d"])
        if len(valid) < 5:
            st.info(f"有效数据点不足（{len(valid)} 行，需要 ≥ 5 行）")
        else:
            x_vals = (valid["garch_forecast_vol"] * 100).tolist()
            y_vals = (valid["realized_vol_20d"] * 100).tolist()
            all_vals = x_vals + y_vals
            v_min, v_max = min(all_vals), max(all_vals)

            fig_sc = go.Figure()
            fig_sc.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                mode="markers",
                name="预测 vs 实际",
                marker=dict(color="#3498db", size=7, opacity=0.7),
                text=valid["trade_date"].tolist(),
                hovertemplate="日期: %{text}<br>GARCH预测: %{x:.2f}%<br>实际RV22: %{y:.2f}%",
            ))
            # 45° reference line
            fig_sc.add_trace(go.Scatter(
                x=[v_min, v_max], y=[v_min, v_max],
                mode="lines", name="理想线 (45°)",
                line=dict(color="rgba(255,200,0,0.6)", dash="dash"),
            ))
            fig_sc.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="GARCH 5日预测波动率 (%)",
                           showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
                yaxis=dict(title="实际 RV22 (%)",
                           showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
                margin=dict(l=10, r=10, t=30, b=10),
                height=320,
            )
            st.plotly_chart(fig_sc, use_container_width=True)

    st.divider()

    # ── Charts 3 & 4: Residuals ──
    residuals_col, qq_col = st.columns(2)

    with residuals_col:
        st.subheader("标准化残差分布")
        if garch_data and "error" not in garch_data:
            std_resid = garch_data.get("std_resid", [])
            if std_resid:
                resid_arr = np.array(std_resid)
                resid_clean = resid_arr[~np.isnan(resid_arr)]
                if len(resid_clean) > 10:
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Histogram(
                        x=resid_clean.tolist(),
                        nbinsx=50,
                        name="标准化残差",
                        marker_color="#3498db",
                        opacity=0.7,
                        histnorm="probability density",
                    ))
                    # Normal curve overlay
                    x_norm = np.linspace(resid_clean.min(), resid_clean.max(), 200)
                    y_norm = (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x_norm ** 2)
                    fig_hist.add_trace(go.Scatter(
                        x=x_norm.tolist(), y=y_norm.tolist(),
                        mode="lines", name="标准正态",
                        line=dict(color="#e74c3c", width=2),
                    ))
                    fig_hist.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        xaxis=dict(title="标准化残差",
                                   showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
                        yaxis=dict(title="密度",
                                   showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
                        legend=dict(orientation="h", y=1.02),
                        margin=dict(l=10, r=10, t=40, b=10),
                        height=300,
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)

                    # Summary stats
                    from scipy import stats as scipy_stats
                    skew = float(scipy_stats.skew(resid_clean))
                    kurt = float(scipy_stats.kurtosis(resid_clean))
                    _, jb_pval = scipy_stats.jarque_bera(resid_clean)
                    st.write(f"偏度: {skew:.4f} | 超额峰度: {kurt:.4f} | JB p值: {jb_pval:.4f}")
                else:
                    st.info("残差数据不足")
            else:
                st.info("无残差数据")
        else:
            st.info("GARCH 模型未就绪")

    with qq_col:
        st.subheader("残差 QQ 图")
        if garch_data and "error" not in garch_data:
            std_resid = garch_data.get("std_resid", [])
            if std_resid:
                resid_arr = np.array(std_resid)
                resid_clean = resid_arr[~np.isnan(resid_arr)]
                if len(resid_clean) > 10:
                    try:
                        from scipy import stats as scipy_stats
                        qq_result = scipy_stats.probplot(resid_clean, dist="norm")
                        theoretical_q = qq_result[0][0].tolist()
                        sample_q = qq_result[0][1].tolist()
                        fit_line_x = [min(theoretical_q), max(theoretical_q)]
                        slope = qq_result[1][0]
                        intercept = qq_result[1][1]
                        fit_line_y = [slope * x + intercept for x in fit_line_x]

                        fig_qq = go.Figure()
                        fig_qq.add_trace(go.Scatter(
                            x=theoretical_q, y=sample_q,
                            mode="markers", name="分位数",
                            marker=dict(color="#3498db", size=4, opacity=0.7),
                        ))
                        fig_qq.add_trace(go.Scatter(
                            x=fit_line_x, y=fit_line_y,
                            mode="lines", name="正态参考线",
                            line=dict(color="#e74c3c", dash="dash", width=2),
                        ))
                        fig_qq.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            xaxis=dict(title="理论分位数",
                                       showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
                            yaxis=dict(title="样本分位数",
                                       showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
                            legend=dict(orientation="h", y=1.02),
                            margin=dict(l=10, r=10, t=40, b=10),
                            height=300,
                        )
                        st.plotly_chart(fig_qq, use_container_width=True)
                    except ImportError:
                        st.info("需要 scipy 库来绘制 QQ 图")
                    except Exception as e:
                        st.warning(f"QQ 图绘制失败: {e}")
                else:
                    st.info("残差数据不足")
            else:
                st.info("无残差数据")
        else:
            st.info("GARCH 模型未就绪")

    st.divider()

    # ── Chart 5: Prediction Error Time Series ──
    st.subheader("预测误差时序（GARCH 5日预测 - RV22）")

    if n_hist < 5:
        st.info(f"预测误差时序需要 ≥ 5 天数据（当前 {n_hist} 天）")
    else:
        valid_err = model_history.dropna(subset=["garch_forecast_vol", "realized_vol_20d"]).copy()
        if len(valid_err) < 5:
            st.info(f"有效数据不足（{len(valid_err)} 行）")
        else:
            valid_err["pred_error"] = (valid_err["garch_forecast_vol"] - valid_err["realized_vol_20d"]) * 100
            bar_colors = ["#e74c3c" if v >= 0 else "#2ecc71" for v in valid_err["pred_error"]]

            fig_err = go.Figure()
            fig_err.add_trace(go.Bar(
                x=valid_err["trade_date"].tolist(),
                y=valid_err["pred_error"].tolist(),
                marker_color=bar_colors,
                name="预测误差 (pp)",
            ))
            fig_err.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)")
            fig_err.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)",
                           title="预测误差 (百分点)"),
                margin=dict(l=10, r=10, t=30, b=10),
                height=260,
            )
            st.plotly_chart(fig_err, use_container_width=True)

            # RMSE
            rmse = np.sqrt((valid_err["pred_error"] ** 2).mean())
            mae = valid_err["pred_error"].abs().mean()
            bias = valid_err["pred_error"].mean()
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("RMSE (pp)", f"{rmse:.4f}")
            mc2.metric("MAE (pp)", f"{mae:.4f}")
            mc3.metric("偏差 (pp)", f"{bias:+.4f}")

    st.divider()

    # ── Section: 5-Day Forecast Backtest (from DB backfill) ──
    st.subheader("GARCH 5日预测回测验证（滚动回填）")
    st.caption("每天 EOD 自动回填5个交易日前的预测误差，积累后展示真实预测能力。")

    try:
        bt_df = db.query_df(
            """SELECT trade_date, garch_forecast_vol, rv_5d_actual, forecast_error
               FROM daily_model_output
               WHERE underlying='IM'
                 AND rv_5d_actual IS NOT NULL
                 AND forecast_error IS NOT NULL
               ORDER BY trade_date ASC"""
        )
    except Exception:
        bt_df = pd.DataFrame()

    MIN_BT = 20
    if bt_df is None or bt_df.empty:
        st.info("尚无回填数据。请在 EOD 流程积累至少 5 天后，数据将开始自动填入。")
    elif len(bt_df) < MIN_BT:
        st.info(f"数据积累中（已有 {len(bt_df)} 条，建议积累 ≥ {MIN_BT} 条后分析系统性偏差）")
        st.dataframe(
            bt_df.assign(
                forecast_pct=lambda d: d["garch_forecast_vol"] * 100,
                actual_pct=lambda d: d["rv_5d_actual"] * 100,
                error_pp=lambda d: d["forecast_error"] * 100,
            )[["trade_date", "forecast_pct", "actual_pct", "error_pp"]].rename(columns={
                "trade_date": "预测日",
                "forecast_pct": "GARCH预测(%)",
                "actual_pct": "实际5日RV(%)",
                "error_pp": "误差(pp)",
            }),
            use_container_width=True, hide_index=True,
        )
    else:
        bt_df["forecast_pct"] = bt_df["garch_forecast_vol"] * 100
        bt_df["actual_pct"]   = bt_df["rv_5d_actual"]       * 100
        bt_df["error_pp"]     = bt_df["forecast_error"]     * 100

        # KPI metrics
        mae   = bt_df["error_pp"].abs().mean()
        bias  = bt_df["error_pp"].mean()
        rmse  = float(np.sqrt((bt_df["error_pp"] ** 2).mean()))
        n_obs = len(bt_df)

        # t-test for systematic bias
        try:
            from scipy import stats as scipy_stats
            t_stat, t_pval = scipy_stats.ttest_1samp(bt_df["error_pp"], 0)
            bias_sig = f"{'显著' if t_pval < 0.05 else '不显著'}（p={t_pval:.3f}）"
        except Exception:
            bias_sig = "无法计算"

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("样本数", n_obs)
        k2.metric("MAE (pp)", f"{mae:.2f}")
        k3.metric("平均偏差 (pp)", f"{bias:+.2f}", help="正=系统性高估，负=系统性低估")
        k4.metric("RMSE (pp)", f"{rmse:.2f}")

        if abs(bias) > 0.5:
            bias_dir = "高估" if bias > 0 else "低估"
            st.warning(f"检测到系统性偏差：平均{bias_dir} {abs(bias):.2f}pp — {bias_sig}")
        else:
            st.success(f"无显著系统性偏差（{bias_sig}）")

        # Scatter: forecast vs actual
        sc_col, err_col = st.columns(2)

        with sc_col:
            st.markdown("**预测 vs 实际散点图**")
            all_v = bt_df["forecast_pct"].tolist() + bt_df["actual_pct"].tolist()
            v_min, v_max = min(all_v), max(all_v)

            fig_sc = go.Figure()
            fig_sc.add_trace(go.Scatter(
                x=bt_df["forecast_pct"].tolist(),
                y=bt_df["actual_pct"].tolist(),
                mode="markers",
                name="预测 vs 实际",
                marker=dict(color="#3498db", size=7, opacity=0.75),
                text=bt_df["trade_date"].tolist(),
                hovertemplate="预测日: %{text}<br>GARCH预测: %{x:.2f}%<br>实际5日RV: %{y:.2f}%",
            ))
            fig_sc.add_trace(go.Scatter(
                x=[v_min, v_max], y=[v_min, v_max],
                mode="lines", name="理想线 (45°)",
                line=dict(color="rgba(255,200,0,0.6)", dash="dash"),
            ))
            fig_sc.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="GARCH 5日预测 (%)",
                           showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
                yaxis=dict(title="实际 5日 RV (%)",
                           showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
                legend=dict(orientation="h", y=1.02),
                margin=dict(l=10, r=10, t=10, b=10),
                height=300,
            )
            st.plotly_chart(fig_sc, use_container_width=True)

        with err_col:
            st.markdown("**预测误差时序**")
            bar_colors = ["#e74c3c" if v >= 0 else "#2ecc71" for v in bt_df["error_pp"]]
            fig_err = go.Figure()
            fig_err.add_trace(go.Bar(
                x=bt_df["trade_date"].tolist(),
                y=bt_df["error_pp"].tolist(),
                marker_color=bar_colors,
                name="误差 (pp)",
                hovertemplate="日期: %{x}<br>误差: %{y:+.2f}pp",
            ))
            fig_err.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.3)")
            fig_err.add_hline(y=bias, line_dash="dash",
                              line_color="rgba(255,165,0,0.7)",
                              annotation_text=f"均值 {bias:+.2f}pp",
                              annotation_position="top right")
            fig_err.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
                yaxis=dict(title="预测误差 (pp)",
                           showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
                margin=dict(l=10, r=10, t=10, b=10),
                height=300,
            )
            st.plotly_chart(fig_err, use_container_width=True)

        # Detail table
        with st.expander("查看明细数据"):
            show_df = bt_df[["trade_date", "forecast_pct", "actual_pct", "error_pp"]].copy()
            show_df.columns = ["预测日", "GARCH预测(%)", "实际5日RV(%)", "误差(pp)"]
            show_df = show_df.sort_values("预测日", ascending=False)
            st.dataframe(show_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── Section: Diagnostic test summary ──
    st.subheader("统计检验汇总")

    if garch_data and "error" not in garch_data:
        diag = garch_data.get("diagnostics", {})
        if not diag:
            st.info("诊断检验数据不可用（模型未返回 diagnose() 结果）")
        else:
            rows = []
            test_map = {
                "ljung_box": "Ljung-Box（残差序列相关）",
                "arch_lm": "ARCH-LM（剩余 ARCH 效应）",
                "lb_pvalue": "Ljung-Box（残差序列相关）",
                "arch_pvalue": "ARCH-LM（剩余 ARCH 效应）",
            }
            seen = set()
            for key, label in test_map.items():
                if key not in diag or label in seen:
                    continue
                seen.add(label)
                val = diag[key]
                if isinstance(val, dict):
                    pval = val.get("pvalue", val.get("p_value", None))
                elif isinstance(val, (int, float)):
                    pval = val
                else:
                    pval = None
                if pval is not None:
                    status = "通过 ✓" if pval > 0.05 else "未通过 ✗"
                    interpretation = "无显著问题" if pval > 0.05 else "存在显著问题"
                    rows.append({
                        "检验": label,
                        "p值": f"{float(pval):.4f}",
                        "结论 (α=0.05)": status,
                        "含义": interpretation,
                    })
            if rows:
                diag_df = pd.DataFrame(rows)

                def _color_status(val):
                    if "✓" in str(val):
                        return "color: #2ecc71"
                    elif "✗" in str(val):
                        return "color: #e74c3c"
                    return ""

                styled_diag = diag_df.style.applymap(_color_status, subset=["结论 (α=0.05)"])
                st.dataframe(styled_diag, use_container_width=True, hide_index=True)
            else:
                st.info("诊断检验结果解析失败")
    else:
        st.info("请先确保 GARCH 模型成功拟合")


if __name__ == "__main__":
    render()
