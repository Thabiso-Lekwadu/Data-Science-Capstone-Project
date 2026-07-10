"""Page 2 — Model Performance (regression).

Everything on this page is real: per-fold walk-forward CV metrics,
the mean/std summary, and the paired significance tests computed by
model_training_nodes.py. No simulated data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

METRICS = ["rmse", "mae", "r2", "smape"]
METRIC_LABELS = {"rmse": "RMSE", "mae": "MAE", "r2": "R²", "smape": "SMAPE (%)"}
LOWER_IS_BETTER = {"rmse", "mae", "smape"}


def _upd(fig, base, **kw):
    fig.update_layout(**{**base, **kw})
    return fig


def render(per_fold_df: pd.DataFrame, summary_df: pd.DataFrame,
           significance_df: pd.DataFrame, PALETTE: list, PLOT_BASE: dict,
           MODELS: list, CONDITIONS: list):

    st.markdown('<div class="section-label">Module 02</div>', unsafe_allow_html=True)
    st.markdown("# Model Performance")
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    if per_fold_df is None or summary_df is None:
        st.warning("Upload experiment_results_per_fold.xlsx and experiment_results_summary.xlsx "
                  "in the sidebar to see model performance.")
        return

    required = {"model", "condition", "rmse", "mae", "r2", "smape", "fold"}
    if required - set(per_fold_df.columns):
        st.error(f"Per-fold results file is missing columns: {required - set(per_fold_df.columns)}")
        return

    models_present = sorted(per_fold_df["model"].unique())
    conds_present = sorted(per_fold_df["condition"].unique())

    s1, s2 = st.columns(2)
    sel_models = s1.multiselect("Models", models_present, default=models_present)
    sel_conds = s2.multiselect("Condition", conds_present, default=conds_present)
    if not sel_models or not sel_conds:
        st.warning("Select at least one model and condition.")
        return

    pf = per_fold_df[per_fold_df["model"].isin(sel_models) & per_fold_df["condition"].isin(sel_conds)].copy()
    summ = summary_df[summary_df["model"].isin(sel_models) & summary_df["condition"].isin(sel_conds)].copy()

    # ── KPI strip ─────────────────────────────────────────────────────────────
    best = summ.sort_values("rmse_mean").iloc[0]
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Best Model", f"{best['model']} / {best['condition']}")
    k2.metric("RMSE", f"{best['rmse_mean']:.1f}", f"±{best['rmse_std']:.1f}")
    k3.metric("MAE", f"{best['mae_mean']:.1f}", f"±{best['mae_std']:.1f}")
    k4.metric("R²", f"{best['r2_mean']:.3f}", f"±{best['r2_std']:.3f}")
    k5.metric("SMAPE", f"{best['smape_mean']:.1f}%", f"±{best['smape_std']:.1f}")
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Summary Comparison", "Per-Fold Detail", "Significance Tests", "Leaderboard"])

    # ══════════════════════════════════════════════════════════════════════
    # TAB 1 — Summary comparison
    # ══════════════════════════════════════════════════════════════════════
    with tab1:
        metric = st.selectbox("Metric", METRICS, format_func=lambda m: METRIC_LABELS[m], key="mp_metric")
        summ_sorted = summ.sort_values(f"{metric}_mean", ascending=(metric in LOWER_IS_BETTER))
        fig = go.Figure()
        for cond in sel_conds:
            sub = summ_sorted[summ_sorted["condition"] == cond]
            fig.add_trace(go.Bar(
                name=cond, x=sub["model"], y=sub[f"{metric}_mean"],
                error_y=dict(type="data", array=sub[f"{metric}_std"], visible=True),
                marker_color=PALETTE[0] if cond == "master" else PALETTE[1],
            ))
        fig.update_layout(barmode="group")
        _upd(fig, PLOT_BASE, title=f"{METRIC_LABELS[metric]} — Mean ± Std Across Folds, by Model & Condition",
             height=420)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-label">Full Summary Table</div>', unsafe_allow_html=True)
        st.dataframe(summ.sort_values("rmse_mean").reset_index(drop=True), use_container_width=True)

        direction_note = "lower is better" if metric in LOWER_IS_BETTER else "higher is better"
        st.markdown(f'<p style="font-size:0.78rem;color:#566174;">{METRIC_LABELS[metric]}: {direction_note}.</p>',
                   unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 2 — Per-fold detail
    # ══════════════════════════════════════════════════════════════════════
    with tab2:
        metric2 = st.selectbox("Metric", METRICS, format_func=lambda m: METRIC_LABELS[m], key="mp_metric2")
        pf["series"] = pf["model"] + " / " + pf["condition"]
        fig2 = px.line(pf.sort_values("fold"), x="fold", y=metric2, color="series", markers=True,
                      color_discrete_sequence=PALETTE,
                      title=f"{METRIC_LABELS[metric2]} by Walk-Forward Fold")
        st.plotly_chart(_upd(fig2, PLOT_BASE, height=440), use_container_width=True)

        if {"val_start", "val_end", "train_rows", "val_rows"} <= set(pf.columns):
            st.markdown('<div class="section-label">Fold Windows</div>', unsafe_allow_html=True)
            windows = (pf[["fold", "val_start", "val_end", "train_rows", "val_rows"]]
                      .drop_duplicates().sort_values("fold"))
            st.dataframe(windows.reset_index(drop=True), use_container_width=True)

        st.markdown('<div class="section-label">Per-Fold Results Table</div>', unsafe_allow_html=True)
        st.dataframe(pf.sort_values(["model", "condition", "fold"]).reset_index(drop=True),
                    use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 3 — Significance tests
    # ══════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown('<p style="font-size:0.83rem;color:#566174;">Paired t-test on RMSE per model, '
                    'comparing crime-only vs master across the same walk-forward fold windows. This is '
                    'the actual test of the enrichment hypothesis — everything else on this page is '
                    'descriptive.</p>', unsafe_allow_html=True)

        if significance_df is None or significance_df.empty:
            st.info("Upload experiment_significance_tests.xlsx in the sidebar to see this.")
        else:
            sig = significance_df.copy()
            sig_disp = sig[sig["model"].isin(sel_models)] if "model" in sig.columns else sig

            fig3 = go.Figure(go.Bar(
                x=sig_disp["rmse_improvement"], y=sig_disp["model"], orientation="h",
                marker_color=[("#2dc653" if v > 0 else "#ff4b4b") for v in sig_disp["rmse_improvement"]],
                text=[f"p={p:.3f}" for p in sig_disp["p_value"]],
                textposition="outside", textfont=dict(color="#566174", size=10),
            ))
            fig3.add_vline(x=0, line_color="#2a3347", line_width=1)
            _upd(fig3, PLOT_BASE,
                title="RMSE Improvement (crime-only − master) — Positive Means Enrichment Helped",
                height=max(320, 60 * len(sig_disp)))
            st.plotly_chart(fig3, use_container_width=True)

            st.dataframe(sig_disp.reset_index(drop=True), use_container_width=True)

            n_sig = int(sig_disp.get("significant_improvement_at_0.05", pd.Series(dtype=bool)).sum())
            if n_sig > 0:
                st.markdown(f"""
                <div class="stat-card">
                  <span class="badge-low">SIGNIFICANT</span>
                  <span style="font-size:0.85rem;color:#8892a4;margin-left:10px;">
                    {n_sig} of {len(sig_disp)} model(s) show a statistically significant RMSE
                    improvement from enrichment (p &lt; 0.05).
                  </span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="stat-card">
                  <span class="badge-medium">NOT SIGNIFICANT</span>
                  <span style="font-size:0.85rem;color:#8892a4;margin-left:10px;">
                    No model shows a statistically significant RMSE improvement from enrichment at
                    p &lt; 0.05 across current folds — the enrichment hypothesis isn't supported yet
                    by this evidence. Worth reporting honestly either way.
                  </span>
                </div>
                """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 4 — Leaderboard
    # ══════════════════════════════════════════════════════════════════════
    with tab4:
        board = summ.sort_values("rmse_mean").reset_index(drop=True)
        board.index = board.index + 1
        st.markdown('<div class="section-label">Ranked by RMSE (lower is better)</div>', unsafe_allow_html=True)
        st.dataframe(board[["model", "condition", "rmse_mean", "mae_mean", "r2_mean", "smape_mean"]],
                    use_container_width=True)

        fig4 = px.scatter(summ, x="rmse_mean", y="r2_mean", color="model", symbol="condition",
                          size="mae_mean", color_discrete_sequence=PALETTE,
                          title="RMSE vs R² — Bottom-Right is Better",
                          hover_data=["mae_mean", "smape_mean"])
        st.plotly_chart(_upd(fig4, PLOT_BASE, height=460), use_container_width=True)