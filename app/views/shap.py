"""Page — SHAP Feature Importance.

Uses uploaded models + the automatically-engineered feature matrix (derived
in App.py from whichever raw dataset(s) were uploaded — no separate
"pre-engineered" upload needed). Computes SHAP values with
shap.TreeExplainer (works uniformly across RandomForest, XGBoost, LightGBM,
CatBoost), and shows which features actually drive Crime Count predictions.
"""
from __future__ import annotations

import sys

import numpy as np
import io
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model_utils import feature_columns, models_for_condition, align_features_to_model

try:
    import shap
    SHAP_AVAILABLE = True
    SHAP_IMPORT_ERROR = None
except Exception as e:  # broader than ImportError — shap can fail on binary/numpy mismatches too
    SHAP_AVAILABLE = False
    SHAP_IMPORT_ERROR = f"{type(e).__name__}: {e}"


def _upd(fig, base, **kw):
    fig.update_layout(**{**base, **kw})
    return fig


@st.cache_data(show_spinner=False)
def _sample_rows(features_json: str, n: int, seed: int = 42) -> pd.DataFrame:
    df = pd.read_json(io.StringIO(features_json))
    return df.sample(min(n, len(df)), random_state=seed).reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def _make_explainer(_model, model_key: str):
    return shap.TreeExplainer(_model)


def render(master_feat: pd.DataFrame, crime_feat: pd.DataFrame, uploaded_models: dict,
           PALETTE: list, PLOT_BASE: dict, MODEL_NAMES: list, CONDITIONS: list):
    st.markdown('<div class="section-label">Module 03</div>', unsafe_allow_html=True)
    st.markdown("# SHAP Feature Importance")
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    if not SHAP_AVAILABLE:
        st.error("The `shap` package couldn't be imported in **this** Python environment.")
        st.code(
            f"Python executable Streamlit is running under:\n{sys.executable}\n\n"
            f"Import error:\n{SHAP_IMPORT_ERROR}",
            language="text",
        )
        st.markdown(
            '<p style="font-size:0.83rem;color:#8892a4;line-height:1.7;">'
            'This is almost always an environment mismatch: <code>pip install shap</code> installed it '
            'into a <em>different</em> Python than the one running Streamlit. Install it into the exact '
            'interpreter shown above, then fully stop and restart <code>streamlit run App.py</code> — '
            'a browser refresh alone will not pick up a newly installed package.</p>',
            unsafe_allow_html=True,
        )
        st.code(f'"{sys.executable}" -m pip install shap', language="powershell")
        return

    available_conditions = [c for c in CONDITIONS
                             if (c == "master" and master_feat is not None)
                             or (c == "crime_only" and crime_feat is not None)]
    if not available_conditions:
        st.warning("Upload ① the master dataset or ② the crime-only dataset in the sidebar to use this page.")
        return

    condition = st.radio("Condition", available_conditions, horizontal=True,
                         format_func=lambda c: "Master (enriched)" if c == "master" else "Crime Only",
                         key="shap_condition")

    active_feat = master_feat if condition == "master" else crime_feat
    if active_feat is None:
        st.warning(f"Upload the {'master' if condition == 'master' else 'crime-only'} dataset "
                   "in the sidebar to use this condition.")
        return

    models = models_for_condition(uploaded_models, condition)
    if not models:
        st.error(f"No uploaded model files match condition '{condition}'. Upload the .pkl files "
                 f"named e.g. randomforest_{condition}.pkl, xgboost_{condition}.pkl, etc. "
                 "in the sidebar (③ Model files).")
        return
    missing = [m for m in MODEL_NAMES if m not in models]
    if missing:
        st.info(f"No uploaded file for: {', '.join(missing)} ({condition}) — skipped.")

    c1, c2 = st.columns(2)
    model_name = c1.selectbox("Model", list(models.keys()), key="shap_model")
    sample_n = c2.slider("Sample size (rows)", min(50, len(active_feat)),
                         min(3000, len(active_feat)),
                         min(800, len(active_feat)), key="shap_n")

    feat_cols = feature_columns(active_feat)
    features_json = active_feat[feat_cols].to_json()
    X_sample = _sample_rows(features_json, sample_n)

    model = models[model_name]

    try:
        X_sample = align_features_to_model(X_sample, model)
    except ValueError as e:
        st.error(str(e))
        return

    with st.spinner(f"Computing SHAP values for {model_name}..."):
        try:
            explainer = _make_explainer(model, model_key=f"{model_name}:{condition}")
            shap_values = explainer.shap_values(X_sample)
        except Exception as e:
            st.error(f"SHAP computation failed for {model_name}: {e}")
            return

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    mean_abs = np.abs(shap_values).mean(axis=0)
    imp_df = pd.DataFrame({"feature": X_sample.columns, "mean_abs_shap": mean_abs})
    imp_df = imp_df.sort_values("mean_abs_shap", ascending=False)

    st.markdown('<div class="section-label">Global Feature Importance (Mean |SHAP|)</div>',
               unsafe_allow_html=True)
    top_n = st.slider("Show top N features", 5, min(40, len(imp_df)), min(15, len(imp_df)), key="shap_topn")
    top_imp = imp_df.head(top_n).sort_values("mean_abs_shap")

    fig_bar = go.Figure(go.Bar(
        x=top_imp["mean_abs_shap"], y=top_imp["feature"], orientation="h",
        marker=dict(color=top_imp["mean_abs_shap"].tolist(),
                    colorscale=[[0, "#1a2235"], [0.4, "#4895ef"], [1, "#e9c46a"]],
                    showscale=False),
        text=top_imp["mean_abs_shap"].round(2), textposition="outside",
        textfont=dict(color="#566174", size=10),
    ))
    _upd(fig_bar, PLOT_BASE, title=f"{model_name} ({condition}) — Mean |SHAP| Feature Importance",
        height=max(360, 22 * top_n))
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown('<div class="section-label">SHAP Beeswarm — Direction & Magnitude</div>',
               unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.8rem;color:#566174;">Each dot is one row in the sample. '
               'Position left/right shows whether that feature pushed the prediction down or up.</p>',
               unsafe_allow_html=True)
    try:
        import matplotlib.pyplot as plt
        top_features = imp_df.head(top_n)["feature"].tolist()
        idx = [list(X_sample.columns).index(f) for f in top_features]
        fig_bee = plt.figure(figsize=(8, max(4, 0.35 * top_n)))
        shap.summary_plot(shap_values[:, idx], X_sample.iloc[:, idx], show=False, plot_size=None)
        st.pyplot(fig_bee, use_container_width=True)
        plt.close(fig_bee)
    except Exception as e:
        st.info(f"Beeswarm plot unavailable ({e}) — the bar chart above still gives the ranking.")

    st.markdown('<div class="section-label">Feature Importance Table</div>', unsafe_allow_html=True)
    st.dataframe(imp_df.reset_index(drop=True), use_container_width=True)

    if condition == "master":
        socio_cols = ["population_density", "poor_households",
                      "population_unemployment", "population_education"]
        socio_present = [c for c in socio_cols if c in imp_df["feature"].values]
        if socio_present:
            socio_rank = imp_df[imp_df["feature"].isin(socio_present)]
            avg_rank = (imp_df.reset_index(drop=True).reset_index()
                       .set_index("feature").loc[socio_present, "index"].mean()) + 1
            st.markdown(f"""
            <div class="stat-card">
              <div class="section-label">Enrichment Signal Check</div>
              <p style="font-size:0.82rem;color:#8892a4;margin:6px 0 0;line-height:1.6;">
                The socioeconomic features rank at an average position of
                <span style="color:#e9c46a;font-family:'DM Mono',monospace;">{avg_rank:.1f}</span>
                out of {len(imp_df)} features by SHAP importance.
              </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="section-label">Grouped Importance — Socioeconomic Block vs Individual Features</div>',
                       unsafe_allow_html=True)
            st.markdown(
                '<p style="font-size:0.8rem;color:#566174;line-height:1.6;">'
                'These four features correlate with each other (they scale with cluster population size), '
                'which splits SHAP credit between them. Summing their |SHAP| shows the block\'s real '
                'combined contribution — no features dropped to produce this.</p>', unsafe_allow_html=True)

            non_socio = imp_df[~imp_df["feature"].isin(socio_present)].copy()
            grouped_row = pd.DataFrame([{
                "feature": "Socioeconomic Block (combined)",
                "mean_abs_shap": socio_rank["mean_abs_shap"].sum(),
            }])
            grouped_view = pd.concat([non_socio, grouped_row], ignore_index=True)
            grouped_view = grouped_view.sort_values("mean_abs_shap").tail(top_n)
            grouped_view["is_group"] = grouped_view["feature"] == "Socioeconomic Block (combined)"

            fig_grp = go.Figure(go.Bar(
                x=grouped_view["mean_abs_shap"], y=grouped_view["feature"], orientation="h",
                marker_color=["#e9c46a" if g else "#4895ef" for g in grouped_view["is_group"]],
                text=grouped_view["mean_abs_shap"].round(2), textposition="outside",
                textfont=dict(color="#566174", size=10),
            ))
            _upd(fig_grp, PLOT_BASE,
                title=f"{model_name} ({condition}) — Individual vs. Combined Socioeconomic Block",
                height=max(340, 22 * len(grouped_view)))
            st.plotly_chart(fig_grp, use_container_width=True)

            combined_share = socio_rank["mean_abs_shap"].sum() / imp_df["mean_abs_shap"].sum() * 100
            st.markdown(f"""
            <div class="stat-card">
              <div class="section-label">Combined Socioeconomic Share</div>
              <p style="font-size:0.82rem;color:#8892a4;margin:6px 0 0;line-height:1.6;">
                Summed together, the four socioeconomic features account for
                <span style="color:#e9c46a;font-family:'DM Mono',monospace;">{combined_share:.1f}%</span>
                of total |SHAP| mass for this model.
              </p>
            </div>
            """, unsafe_allow_html=True)