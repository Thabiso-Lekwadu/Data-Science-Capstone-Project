"""Page — Prediction Explorer.

Pick a Cluster, Type of Crime, and a set of feature values; every uploaded
"master" (enriched) model — Random Forest, XGBoost, LightGBM, CatBoost —
predicts Crime Count for that scenario side by side.

Lag/rolling feature defaults are auto-suggested from the most recent known
record for that (Cluster, Type of Crime) pair in the automatically-engineered
master features (derived from the ① master dataset upload in App.py). Fully
editable — override them to explore other scenarios.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from model_utils import (
    feature_columns, cluster_and_crime_type_columns, build_input_row,
    models_for_condition, DATE_COL, align_features_to_model,
)

SOCIO_LABELS = {
    "population_density":      "Population Density",
    "poor_households":         "Poor Households",
    "population_unemployment": "Population Unemployment",
    "population_education":    "Population Education",
}
LAG_ROLL_PREFIXES = ("crime_count_lag_", "crime_count_roll_mean_", "crime_count_roll_std_")


def _upd(fig, base, **kw):
    fig.update_layout(**{**base, **kw})
    return fig


def _historical_last_row(features_df, cluster, crime_type):
    ccol, tcol = f"cluster_{cluster}", f"crime_type_{crime_type}"
    if ccol not in features_df.columns or tcol not in features_df.columns:
        return None
    sub = features_df[(features_df[ccol] == 1) & (features_df[tcol] == 1)]
    if sub.empty:
        return None
    sub = sub.copy()
    sub[DATE_COL] = pd.to_datetime(sub[DATE_COL])
    return sub.sort_values(DATE_COL).iloc[-1]


def render(master_feat: pd.DataFrame, uploaded_models: dict, PALETTE: list, PLOT_BASE: dict):
    st.markdown('<div class="section-label">Module 04</div>', unsafe_allow_html=True)
    st.markdown("# Prediction Explorer")
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    if master_feat is None:
        st.warning("Upload ① the master dataset in the sidebar — features are engineered automatically.")
        return

    feat_cols = feature_columns(master_feat)
    clusters, crime_types = cluster_and_crime_type_columns(feat_cols)
    if not clusters or not crime_types:
        st.error("Couldn't find Cluster/Type of Crime columns after engineering the uploaded dataset. "
                 "Make sure ① is the raw master_dataset.xlsx schema (date, Cluster, Type of Crime, "
                 "Crime Count, socioeconomic columns).")
        return

    models = models_for_condition(uploaded_models, "master")
    if not models:
        st.error("No uploaded model files match condition 'master'. Upload .pkl files named "
                 "e.g. randomforest_master.pkl, xgboost_master.pkl, etc. in the sidebar (③ Model files).")
        return

    c1, c2 = st.columns(2)
    cluster = c1.selectbox("Cluster", clusters, key="pred_cluster")
    crime_type = c2.selectbox("Type of Crime", crime_types, key="pred_crime_type")

    last_row = _historical_last_row(master_feat, cluster, crime_type)

    default_year = int(last_row["year"]) + 1 if last_row is not None and "year" in last_row else 2026
    default_month = int(last_row["month"]) if last_row is not None and "month" in last_row else 12

    st.markdown('<div class="section-label">Prediction Target Period</div>', unsafe_allow_html=True)
    year = st.number_input("Year", min_value=2000, max_value=2100, value=default_year, key="pred_year")

    st.markdown('<div class="section-label">Socioeconomic Inputs</div>', unsafe_allow_html=True)
    socio_cols = [c for c in feat_cols if c in SOCIO_LABELS]
    socio_values = {}
    s_cols = st.columns(len(socio_cols)) if socio_cols else []
    for i, col in enumerate(socio_cols):
        default_val = float(last_row[col]) if last_row is not None and col in last_row else \
                      float(master_feat[col].median())
        socio_values[col] = s_cols[i].number_input(
            SOCIO_LABELS[col], value=round(default_val, 2), key=f"pred_{col}")

    with st.expander("Advanced: lag & rolling Crime Count features (auto-suggested — edit to explore scenarios)"):
        if last_row is None:
            st.info("No history found for this Cluster/Type combination — defaulting all lag/rolling "
                   "features to 0. Predictions for a never-observed pairing should be treated cautiously.")
        lag_roll_cols = [c for c in feat_cols if c.startswith(LAG_ROLL_PREFIXES)]
        lag_roll_values = {}
        n_lr_cols = 3
        lr_grid = st.columns(n_lr_cols)
        for i, col in enumerate(lag_roll_cols):
            default_val = float(last_row[col]) if last_row is not None and col in last_row else 0.0
            lag_roll_values[col] = lr_grid[i % n_lr_cols].number_input(
                col.replace("crime_count_", "").replace("_", " ").title(),
                value=round(default_val, 2), key=f"pred_{col}")

    calendar_values = {}
    if "year" in feat_cols:
        calendar_values["year"] = int(year)
    if "quarter" in feat_cols:
        calendar_values["quarter"] = (default_month - 1) // 3 + 1
    if "month" in feat_cols:
        calendar_values["month"] = default_month
    if "month_sin" in feat_cols:
        calendar_values["month_sin"] = float(np.sin(2 * np.pi * default_month / 12))
    if "month_cos" in feat_cols:
        calendar_values["month_cos"] = float(np.cos(2 * np.pi * default_month / 12))

    all_values = {**socio_values, **lag_roll_values, **calendar_values}

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    if st.button("Predict Crime Count", type="primary", key="pred_button"):
        X_row = build_input_row(feat_cols, cluster, crime_type, all_values)

        preds, raw_preds, errors = {}, {}, {}
        for name, model in models.items():
            try:
                X_aligned = align_features_to_model(X_row, model)
                # Explicit float64 cast — a single-row DataFrame built from a
                # plain dict can end up with an object-dtype column, which
                # LightGBM rejects outright ("DataFrame.dtypes must be int,
                # float or bool"); harmless for the other three models.
                X_aligned = X_aligned.astype(float)
                raw = float(model.predict(X_aligned)[0])
                raw_preds[name] = raw
                # Crime Count is a non-negative integer — a plain regressor
                # has no floor at zero and can extrapolate below it (this is
                # what was showing up as XGBoost's negative predictions).
                # Clipping and rounding here is what turns a raw regression
                # output into an actual, reportable "specific crime count".
                preds[name] = int(round(max(0.0, raw)))
            except Exception as e:
                import traceback
                errors[name] = f"{e}\n\n{traceback.format_exc(limit=3)}"

        if errors:
            for name, err in errors.items():
                with st.expander(f"⚠️ {name} failed to predict — click for details"):
                    st.code(err, language="text")

        if not preds:
            st.error("No model produced a prediction.")
            return

        pred_df = pd.DataFrame(list(preds.items()), columns=["Model", "Predicted Crime Count"])
        pred_df["Raw Model Output"] = pred_df["Model"].map(raw_preds)
        pred_df = pred_df.sort_values("Predicted Crime Count", ascending=False)

        clipped = {n: r for n, r in raw_preds.items() if r < 0}
        if clipped:
            rows_html = "".join(
                f"<li><b>{n}</b>: raw output {r:.2f} → floored to 0</li>" for n, r in clipped.items()
            )
            st.markdown(f"""
            <div class="stat-card" style="border-color:#e9c46a;">
              <span class="badge-high">CLIPPED PREDICTION{'S' if len(clipped) > 1 else ''}</span>
              <p style="font-size:0.82rem;color:#8892a4;margin:10px 0 0;line-height:1.7;">
                {len(clipped)} of {len(pred_df)} model(s) actually predicted a <b>negative</b> raw value
                for this scenario, floored to 0 for display:
                <ul style="margin:6px 0 0 18px;">{rows_html}</ul>
                A large negative raw value (e.g. below −5 or so) usually means the model is
                extrapolating outside what it saw in training — most often because the lag/rolling
                Crime Count features above defaulted to 0 (no history for this Cluster/Type pairing)
                and/or the target year is beyond the training data's range. A "0" here isn't
                necessarily a confident "no crime expected" — check the raw value and the feature
                row below before reporting it as such.
              </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f'<div class="section-label">Predicted Crime Count — {cluster} / {crime_type} / {int(year)}</div>',
                    unsafe_allow_html=True)
        cols = st.columns(len(pred_df))
        for c, (_, row_) in zip(cols, pred_df.iterrows()):
            raw_val = row_["Raw Model Output"]
            delta = f"raw: {raw_val:.2f}" if raw_val < 0 else None
            c.metric(row_["Model"], f"{row_['Predicted Crime Count']:,} incidents", delta=delta, delta_color="off")

        fig = go.Figure(go.Bar(
            x=pred_df["Model"], y=pred_df["Predicted Crime Count"],
            marker_color=PALETTE[:len(pred_df)],
            text=pred_df["Predicted Crime Count"],
            textposition="outside", textfont=dict(color="#566174"),
        ))
        _upd(fig, PLOT_BASE, height=380,
             title=f"Predicted Crime Count — {cluster} / {crime_type} / {int(year)}")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Show raw vs. floored predictions, and the exact feature row sent to the models"):
            st.dataframe(pred_df[["Model", "Raw Model Output", "Predicted Crime Count"]]
                        .reset_index(drop=True), use_container_width=True)
            st.dataframe(X_row.T.rename(columns={0: "value"}), use_container_width=True)