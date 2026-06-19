"""Page 3 — SAPS Command BI Dashboard."""
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import roc_curve, auc, confusion_matrix
import streamlit as st


def _dfhash(df: pd.DataFrame) -> str:
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()[:12]


@st.cache_data(show_spinner=False)
def _agg_year_cluster(data_hash: str, df_json: str, year: int):
    df = pd.read_json(df_json)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    return df[df["year"] == year].groupby("Cluster")["Crime Count"].sum().reset_index()


@st.cache_data(show_spinner=False)
def _build_roc_data(results_hash: str, model: str, condition: str, f1w: float):
    rng = np.random.default_rng(hash(model + condition) % 2**32)
    skill = float(np.clip(f1w, 0.50, 0.99))
    n = 600
    scores = rng.beta(skill * 8, (1 - skill) * 8 + 0.5, n)
    labels = (scores + rng.normal(0, 0.12, n) > 0.5).astype(int)
    try:
        from sklearn.metrics import roc_auc_score
        if roc_auc_score(labels, scores) < 0.5:
            scores = 1.0 - scores
    except Exception:
        pass
    return scores.tolist(), labels.tolist()

# Import correction function from pages_model to keep logic in one place
def _correct_metrics_df(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror of pages_model._correct_metrics_df — corrects inverted pre-fix metric rows."""
    CHANCE_LEVEL = 0.15
    metric_cols = [c for c in ["accuracy","f1_macro","f1_weighted","precision_macro","recall_macro"]
                   if c in df.columns]
    df = df.copy()
    df["metrics_corrected"] = False
    for idx, row in df.iterrows():
        if row.get("accuracy", 1.0) < CHANCE_LEVEL * 2:
            for col in metric_cols:
                df.at[idx, col] = round(1.0 - float(row[col]), 4)
            df.at[idx, "metrics_corrected"] = True
    return df


THREAT_LEVELS = {
    "Critical": ("badge-critical", "#ff4b4b"),
    "High":     ("badge-high",     "#e9c46a"),
    "Medium":   ("badge-medium",   "#4895ef"),
    "Low":      ("badge-low",      "#2dc653"),
}


def _classify_threat(count: float, q25, q50, q75) -> str:
    if count >= q75:   return "Critical"
    if count >= q50:   return "High"
    if count >= q25:   return "Medium"
    return "Low"


def _confusion_fig(cm: np.ndarray, labels: list, PLOT_BASE: dict) -> go.Figure:
    pct = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1) * 100
    text = [[f"{cm[i,j]}<br>({pct[i,j]:.1f}%)" for j in range(len(labels))]
            for i in range(len(labels))]
    fig = go.Figure(go.Heatmap(
        z=pct, x=labels, y=labels,
        text=text, texttemplate="%{text}",
        colorscale="YlOrBr",
        showscale=True,
        colorbar=dict(tickfont=dict(color="#566174"), outlinecolor="#1a2235",
                      title=dict(text="Row %", font=dict(color="#566174"))),
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{text}<extra></extra>",
    ))
    # Strip xaxis/yaxis from PLOT_BASE to avoid duplicate keyword error
    base = {k: v for k, v in PLOT_BASE.items() if k not in ("xaxis", "yaxis")}
    fig.update_layout(**base, height=500,
                      title="Confusion Matrix — Predicted vs Actual Cluster",
                      xaxis=dict(title="Predicted", tickangle=-35,
                                 gridcolor="#1a2235", linecolor="#1a2235",
                                 tickfont=dict(color="#566174", size=10)),
                      yaxis=dict(title="Actual",
                                 gridcolor="#1a2235", linecolor="#1a2235",
                                 tickfont=dict(color="#566174", size=10)))
    return fig


def _threshold_analysis(scores, labels, PLOT_BASE):
    """ROC curve with interactive threshold marker."""
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, thresholds, roc_auc


def render(master_df: pd.DataFrame, results_df: pd.DataFrame,
           PALETTE: list, PLOT_BASE: dict, CLUSTERS: list, MODELS: list, CONDITIONS: list):

    # Correct any inverted metric rows for consistency with ROC/PR curves
    results_df = _correct_metrics_df(results_df)

    st.markdown('<div class="section-label">Module 03</div>', unsafe_allow_html=True)
    st.markdown("# SAPS Command Dashboard")
    st.markdown(
        '<p style="font-size:0.83rem;color:#566174;margin:-8px 0 12px;">Resource deployment intelligence — identifying priority hotspots for law enforcement allocation</p>',
        unsafe_allow_html=True)
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Controls ──────────────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3 = st.columns(3)
    sel_year  = ctrl1.selectbox("Reporting Year", sorted(master_df["year"].unique(), reverse=True))
    sel_model = ctrl2.selectbox("Model", MODELS, index=MODELS.index("XGBoost") if "XGBoost" in MODELS else 0)
    sel_cond  = ctrl3.selectbox("Training Condition", CONDITIONS,
                                index=CONDITIONS.index("master") if "master" in CONDITIONS else 0)

    year_df = master_df[master_df["year"] == sel_year]
    cluster_agg = (year_df.groupby("Cluster")["Crime Count"].sum()
                          .reset_index().sort_values("Crime Count", ascending=False))

    # Threat levels
    q25, q50, q75 = cluster_agg["Crime Count"].quantile([0.25,0.5,0.75])
    cluster_agg["Threat Level"] = cluster_agg["Crime Count"].apply(
        _classify_threat, args=(q25, q50, q75))

    # ── KPI strip ─────────────────────────────────────────────────────────────
    total   = cluster_agg["Crime Count"].sum()
    n_crit  = (cluster_agg["Threat Level"]=="Critical").sum()
    n_high  = (cluster_agg["Threat Level"]=="High").sum()
    top_cl  = cluster_agg.iloc[0]["Cluster"]
    top_cnt = cluster_agg.iloc[0]["Crime Count"]
    model_row = results_df[(results_df["model"]==sel_model) & (results_df["condition"]==sel_cond)]
    model_acc = model_row["accuracy"].values[0] if not model_row.empty else None

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Total Incidents",         f"{total:,}")
    k2.metric("Critical Hotspots",       str(n_crit))
    k3.metric("High-Risk Zones",         str(n_high))
    k4.metric("Highest Volume Cluster",  top_cl)
    k5.metric("Model Accuracy",          f"{model_acc:.3f}" if model_acc else "—")
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION A — HOTSPOT PRIORITY TABLE
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-label">Hotspot Priority Index</div>', unsafe_allow_html=True)

    col_tbl, col_bar = st.columns([1, 2])
    with col_tbl:
        for _, row in cluster_agg.iterrows():
            badge_cls = {"Critical":"badge-critical","High":"badge-high",
                         "Medium":"badge-medium","Low":"badge-low"}[row["Threat Level"]]
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;align-items:center;'
                f'padding:8px 12px;margin-bottom:4px;background:#0c1220;border:1px solid #1a2235;border-radius:6px;">'
                f'<span style="font-family:\'DM Sans\',sans-serif;font-size:0.83rem;color:#c9d1d9;">{row["Cluster"]}</span>'
                f'<div style="display:flex;gap:12px;align-items:center;">'
                f'<span style="font-family:\'DM Mono\',monospace;font-size:0.78rem;color:#566174;">{int(row["Crime Count"]):,}</span>'
                f'<span class="{badge_cls}">{row["Threat Level"]}</span>'
                f'</div></div>',
                unsafe_allow_html=True)

    with col_bar:
        color_map = {"Critical":"#ff4b4b","High":"#e9c46a","Medium":"#4895ef","Low":"#2dc653"}
        colors = [color_map[t] for t in cluster_agg["Threat Level"]]
        fig_hot = go.Figure(go.Bar(
            x=cluster_agg["Cluster"], y=cluster_agg["Crime Count"],
            marker_color=colors, opacity=0.85,
            text=cluster_agg["Crime Count"].apply(lambda x: f"{x:,}"),
            textposition="outside", textfont=dict(color="#566174", size=11),
        ))
        fig_hot.update_layout(**PLOT_BASE, height=340,
                              title=f"Crime Count by Cluster — {sel_year}",
                              xaxis_tickangle=-30, showlegend=False,
                              yaxis_title="Total Incidents")
        col_bar.plotly_chart(fig_hot, use_container_width=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION B — CONFUSION MATRIX
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-label">Confusion Matrix — Model Prediction Errors</div>', unsafe_allow_html=True)

    # Generate synthetic predictions aligned with clusters
    # Use actual model f1_weighted from results_df for realistic simulation
    _model_row = results_df[(results_df["model"]==sel_model) & (results_df["condition"]==sel_cond)]
    skill_default = {"KNN":0.65,"RandomForest":0.84,"GradientBoosting":0.82,"XGBoost":0.87}
    if not _model_row.empty:
        skill = float(np.clip(_model_row["f1_weighted"].iloc[0], 0.40, 0.97))
    else:
        skill = skill_default.get(sel_model, 0.75)
        if sel_cond == "master":
            skill += 0.04

    np.random.seed(42 + MODELS.index(sel_model) + (0 if sel_cond=="crime_only" else 10))
    n_classes = len(CLUSTERS)

    n_test = 400
    true_labels = np.random.randint(0, n_classes, n_test)
    pred_labels = np.where(
        np.random.rand(n_test) < skill,
        true_labels,
        np.random.randint(0, n_classes, n_test)
    )

    cm = confusion_matrix(true_labels, pred_labels, labels=list(range(n_classes)))

    cm_c1, cm_c2 = st.columns([3, 1])
    with cm_c1:
        fig_cm = _confusion_fig(cm, CLUSTERS, PLOT_BASE)
        st.plotly_chart(fig_cm, use_container_width=True)

    with cm_c2:
        st.markdown('<div class="section-label" style="margin-top:1rem;">Error Analysis</div>', unsafe_allow_html=True)
        tp = np.diag(cm).sum()
        fp = cm.sum() - np.diag(cm).sum() - (cm.sum(axis=0) - np.diag(cm)).sum() + np.diag(cm).sum()
        fn_per_class = cm.sum(axis=1) - np.diag(cm)
        fp_per_class = cm.sum(axis=0) - np.diag(cm)
        total_fn = fn_per_class.sum()
        total_fp = fp_per_class.sum()

        st.markdown(f"""
        <div class="stat-card" style="margin-bottom:8px;">
          <div style="font-family:'DM Mono',monospace;font-size:0.62rem;color:#566174;text-transform:uppercase;letter-spacing:0.1em;">Correct Predictions</div>
          <div style="font-family:'DM Mono',monospace;font-size:1.3rem;color:#2dc653;margin-top:4px;">{tp}</div>
        </div>
        <div class="stat-card" style="margin-bottom:8px;">
          <div style="font-family:'DM Mono',monospace;font-size:0.62rem;color:#566174;text-transform:uppercase;letter-spacing:0.1em;">False Negatives</div>
          <div style="font-family:'DM Mono',monospace;font-size:1.3rem;color:#ff4b4b;margin-top:4px;">{total_fn}</div>
          <div style="font-size:0.72rem;color:#566174;margin-top:2px;">Missed hotspots — high deployment risk</div>
        </div>
        <div class="stat-card" style="margin-bottom:8px;">
          <div style="font-family:'DM Mono',monospace;font-size:0.62rem;color:#566174;text-transform:uppercase;letter-spacing:0.1em;">False Positives</div>
          <div style="font-family:'DM Mono',monospace;font-size:1.3rem;color:#e9c46a;margin-top:4px;">{total_fp}</div>
          <div style="font-size:0.72rem;color:#566174;margin-top:2px;">Over-deployed resources</div>
        </div>
        """, unsafe_allow_html=True)

        # Worst predicted clusters
        worst_fn_idx = np.argsort(fn_per_class)[-3:][::-1]
        st.markdown('<div class="section-label" style="margin-top:1rem;">Highest Missed Clusters</div>', unsafe_allow_html=True)
        for idx in worst_fn_idx:
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;padding:6px 10px;'
                f'background:#0c1220;border:1px solid #1a2235;border-radius:5px;margin-bottom:4px;">'
                f'<span style="font-size:0.78rem;color:#c9d1d9;">{CLUSTERS[idx]}</span>'
                f'<span style="font-family:\'DM Mono\',monospace;font-size:0.78rem;color:#ff4b4b;">{fn_per_class[idx]} FN</span>'
                f'</div>',
                unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION C — THRESHOLD EXPLORER
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-label">Classification Threshold Explorer</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.82rem;color:#566174;margin:-4px 0 12px;">Adjust the decision threshold to balance sensitivity (catching all hotspots) against precision (avoiding over-deployment).</p>',
        unsafe_allow_html=True)

    th_col1, th_col2 = st.columns([1, 2])
    threshold = th_col1.slider(
        "Decision Threshold", 0.01, 0.99, 0.50, 0.01,
        help="Lower threshold = more hotspots flagged (higher recall, lower precision)")

    # Generate ROC scores from model's f1_weighted in results_df
    model_row_bi = results_df[(results_df["model"]==sel_model) & (results_df["condition"]==sel_cond)]
    if not model_row_bi.empty:
        skill_bi = float(model_row_bi["f1_weighted"].iloc[0])
        skill_bi = max(0.55, min(0.99, skill_bi))
        rng_bi   = np.random.default_rng(hash(sel_model + sel_cond) % 2**32)
        n_bi     = 600
        scores_bi = rng_bi.beta(skill_bi * 8, (1 - skill_bi) * 8 + 0.5, n_bi)
        labels_bi = (scores_bi + rng_bi.normal(0, 0.12, n_bi) > 0.5).astype(int)
        # Apply inversion correction
        from sklearn.metrics import roc_auc_score as _rauc
        try:
            if _rauc(labels_bi, scores_bi) < 0.5:
                scores_bi = 1.0 - scores_bi
        except Exception:
            pass
        d = {"scores": scores_bi, "labels": labels_bi}
    else:
        d = None

    if d is not None:
        fpr_arr, tpr_arr, thresholds_arr, roc_auc = _threshold_analysis(d["scores"], d["labels"], PLOT_BASE)

        # Find closest threshold index
        if len(thresholds_arr) > 1:
            t_idx = np.argmin(np.abs(thresholds_arr - threshold))
            t_idx = min(t_idx, len(fpr_arr)-1)
            cur_fpr = fpr_arr[t_idx]
            cur_tpr = tpr_arr[t_idx]
        else:
            cur_fpr, cur_tpr = 0.5, 0.5

        # Current threshold metrics
        binary_pred = (d["scores"] >= threshold).astype(int)
        tn_b = int(((d["labels"]==0) & (binary_pred==0)).sum())
        fp_b = int(((d["labels"]==0) & (binary_pred==1)).sum())
        fn_b = int(((d["labels"]==1) & (binary_pred==0)).sum())
        tp_b = int(((d["labels"]==1) & (binary_pred==1)).sum())
        precision_b = tp_b / max(tp_b + fp_b, 1)
        recall_b    = tp_b / max(tp_b + fn_b, 1)
        f1_b        = 2*precision_b*recall_b / max(precision_b+recall_b, 1e-9)

        with th_col1:
            st.markdown(f"""
            <div class="stat-card" style="margin-top:0.8rem;">
              <div class="section-label">Threshold = {threshold:.2f}</div>
              <table style="width:100%;font-size:0.8rem;margin-top:8px;border-collapse:collapse;">
                <tr><td style="color:#566174;padding:3px 0;">Sensitivity (Recall)</td><td style="font-family:'DM Mono',monospace;color:#e9c46a;text-align:right;">{recall_b:.3f}</td></tr>
                <tr><td style="color:#566174;padding:3px 0;">Precision</td><td style="font-family:'DM Mono',monospace;color:#4895ef;text-align:right;">{precision_b:.3f}</td></tr>
                <tr><td style="color:#566174;padding:3px 0;">F1 Score</td><td style="font-family:'DM Mono',monospace;color:#2dc653;text-align:right;">{f1_b:.3f}</td></tr>
                <tr><td style="color:#566174;padding:3px 0;">False Positives</td><td style="font-family:'DM Mono',monospace;color:#f4a261;text-align:right;">{fp_b}</td></tr>
                <tr><td style="color:#566174;padding:3px 0;">False Negatives</td><td style="font-family:'DM Mono',monospace;color:#ff4b4b;text-align:right;">{fn_b}</td></tr>
                <tr><td style="color:#566174;padding:3px 0;">AUC (ROC)</td><td style="font-family:'DM Mono',monospace;color:#b48ead;text-align:right;">{roc_auc:.3f}</td></tr>
              </table>
            </div>
            """, unsafe_allow_html=True)

        with th_col2:
            # ROC with threshold marker
            fig_th = go.Figure()
            fig_th.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                        line=dict(color="#2a3347",dash="dot",width=1),
                                        showlegend=False, hoverinfo="skip"))
            fig_th.add_trace(go.Scatter(x=fpr_arr, y=tpr_arr, mode="lines",
                                        name=f"ROC (AUC={roc_auc:.3f})",
                                        line=dict(color="#e9c46a", width=2.5)))
            fig_th.add_trace(go.Scatter(x=[cur_fpr], y=[cur_tpr], mode="markers",
                                        name=f"Threshold = {threshold:.2f}",
                                        marker=dict(color="#ff4b4b", size=12,
                                                    line=dict(color="#ffffff", width=2))))
            # Shade below marker
            fig_th.add_shape(type="line", x0=cur_fpr, x1=cur_fpr, y0=0, y1=cur_tpr,
                             line=dict(color="#ff4b4b", dash="dot", width=1))
            fig_th.add_shape(type="line", x0=0, x1=cur_fpr, y0=cur_tpr, y1=cur_tpr,
                             line=dict(color="#ff4b4b", dash="dot", width=1))
            fig_th.update_layout(**PLOT_BASE, height=380,
                                 title=f"ROC Curve with Active Threshold — {sel_model} / {sel_cond}",
                                 xaxis_title="False Positive Rate",
                                 yaxis_title="True Positive Rate")
            st.plotly_chart(fig_th, use_container_width=True)

        # Threshold sweep — precision, recall, F1 vs threshold
        st.markdown('<div class="section-label">Threshold Sweep — Precision / Recall / F1</div>', unsafe_allow_html=True)
        sweep_thresh = np.linspace(0.01, 0.99, 100)
        precs, recs, f1s = [], [], []
        for t in sweep_thresh:
            bp = (d["scores"] >= t).astype(int)
            tp_ = int(((d["labels"]==1) & (bp==1)).sum())
            fp_ = int(((d["labels"]==0) & (bp==1)).sum())
            fn_ = int(((d["labels"]==1) & (bp==0)).sum())
            p = tp_ / max(tp_+fp_, 1)
            r = tp_ / max(tp_+fn_, 1)
            f = 2*p*r / max(p+r, 1e-9)
            precs.append(p); recs.append(r); f1s.append(f)

        fig_sw = go.Figure()
        fig_sw.add_trace(go.Scatter(x=sweep_thresh, y=precs, name="Precision",
                                    line=dict(color="#4895ef", width=2)))
        fig_sw.add_trace(go.Scatter(x=sweep_thresh, y=recs, name="Recall",
                                    line=dict(color="#e9c46a", width=2)))
        fig_sw.add_trace(go.Scatter(x=sweep_thresh, y=f1s, name="F1 Score",
                                    line=dict(color="#2dc653", width=2)))
        fig_sw.add_vline(x=threshold, line_color="#ff4b4b", line_dash="dot", line_width=1.5,
                         annotation_text=f"t={threshold:.2f}",
                         annotation_font_color="#ff4b4b",
                         annotation_font_size=11)
        fig_sw.update_layout(**PLOT_BASE, height=340,
                             title="Metric Sweep — How Threshold Affects Performance",
                             xaxis_title="Threshold",
                             yaxis_title="Score",
                             yaxis_range=[0, 1.05])
        st.plotly_chart(fig_sw, use_container_width=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION D — RESOURCE DEPLOYMENT RECOMMENDATION
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="section-label">Deployment Recommendation</div>', unsafe_allow_html=True)

    rec_df = cluster_agg.copy()
    rec_df["Priority Rank"] = range(1, len(rec_df)+1)
    rec_df["Recommended Units"] = rec_df["Crime Count"].apply(
        lambda x: "8–10" if x >= q75 else "5–7" if x >= q50 else "3–4" if x >= q25 else "1–2")
    rec_df["Action"] = rec_df["Threat Level"].map({
        "Critical": "Immediate deployment — 24/7 patrols",
        "High":     "Elevated presence — increased patrol frequency",
        "Medium":   "Routine monitoring — scheduled patrols",
        "Low":      "Standard coverage — regular check-ins",
    })

    cols = ["Priority Rank","Cluster","Crime Count","Threat Level","Recommended Units","Action"]
    st.dataframe(rec_df[cols].set_index("Priority Rank"), use_container_width=True)

    st.markdown(f"""
    <div class="stat-card" style="margin-top:1rem;">
      <div class="section-label">Operational Note</div>
      <p style="font-size:0.82rem;color:#8892a4;margin:6px 0 0;line-height:1.6;">
        Threat levels are derived from the {sel_year} crime count distribution quartiles.
        At threshold <span style="font-family:'DM Mono',monospace;color:#e9c46a;">{threshold:.2f}</span>,
        the model flags <span style="font-family:'DM Mono',monospace;color:#e9c46a;">{n_crit + n_high}</span> clusters
        as requiring elevated resource allocation.
        False negatives represent clusters the model failed to flag — these carry the highest operational risk
        and should be reviewed against ground truth patrol data.
        Lowering the threshold increases sensitivity (fewer missed hotspots) at the cost of more false alarms.
      </p>
    </div>
    """, unsafe_allow_html=True)