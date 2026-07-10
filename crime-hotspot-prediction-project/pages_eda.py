"""Page 1 — Exploratory Data Analysis.

Four tools:
  1. Univariate Explorer   — distribution of any single column, either dataset
  2. Bivariate Explorer    — relationship between any two columns, either dataset
  3. Time Series Explorer  — trends, year-on-year change, top regions
  4. Enrichment Justification — the evidence behind "add socioeconomic features"
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

READABLE = {
    "Crime Count":             "Crime Count",
    "population_density":      "Population Density",
    "poor_households":         "Poor Households",
    "population_unemployment": "Population Unemployment",
    "population_education":    "Population Education",
    "Cluster":                 "Cluster",
    "Type of Crime":           "Type of Crime",
    "year":                    "Year",
}
SOCIO = ["population_density", "poor_households",
         "population_unemployment", "population_education"]


def _upd(fig, base, **kw):
    fig.update_layout(**{**base, **kw})
    return fig


def _label(c):
    return READABLE.get(c, c)


def _numeric_cols(df):
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in ("year",)] + \
           (["year"] if "year" in df.columns else [])


def _categorical_cols(df):
    return [c for c in ["Cluster", "Type of Crime"] if c in df.columns]


def _prep(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if "year" not in df.columns:
        df["year"] = df["date"].dt.year
    return df


# ── Univariate ──────────────────────────────────────────────────────────────

def _univariate(df, PALETTE, PLOT_BASE):
    st.markdown("## Univariate Explorer")
    st.markdown('<p style="font-size:0.83rem;color:#566174;">Look at the shape of one variable '
                'at a time — its spread, skew, and outliers.</p>', unsafe_allow_html=True)

    num_cols = _numeric_cols(df)
    cat_cols = _categorical_cols(df)
    all_cols = num_cols + cat_cols

    c1, c2 = st.columns(2)
    col = c1.selectbox("Column", all_cols, format_func=_label, key="uni_col")
    is_numeric = col in num_cols

    if is_numeric:
        chart = c2.selectbox("Chart Type", ["Histogram", "Box Plot", "Violin-style (Box + Points)"], key="uni_chart")
        if chart == "Histogram":
            nbins = st.slider("Bins", 10, 100, 40, key="uni_bins")
            fig = px.histogram(df, x=col, nbins=nbins, color_discrete_sequence=PALETTE,
                                title=f"Distribution of {_label(col)}")
        elif chart == "Box Plot":
            fig = px.box(df, y=col, color_discrete_sequence=PALETTE,
                         title=f"Box Plot — {_label(col)}")
        else:
            fig = px.strip(df, y=col, color_discrete_sequence=PALETTE,
                           title=f"Point Spread — {_label(col)}")
            fig.add_trace(go.Box(y=df[col], name="", marker_color=PALETTE[0],
                                  boxpoints=False, fillcolor="rgba(0,0,0,0)"))
        st.plotly_chart(_upd(fig, PLOT_BASE), use_container_width=True)

        st.markdown('<div class="section-label">Summary Statistics</div>', unsafe_allow_html=True)
        desc = df[col].describe().to_frame().T
        desc.index = [_label(col)]
        st.dataframe(desc, use_container_width=True)
    else:
        chart = c2.selectbox("Chart Type", ["Bar Chart", "Horizontal Bar"], key="uni_chart_cat")
        counts = df[col].value_counts().reset_index()
        counts.columns = [col, "Count"]
        if chart == "Bar Chart":
            fig = px.bar(counts, x=col, y="Count", color_discrete_sequence=PALETTE,
                         title=f"Frequency of {_label(col)}")
            fig.update_layout(xaxis_tickangle=-30)
        else:
            fig = px.bar(counts.sort_values("Count"), x="Count", y=col, orientation="h",
                         color_discrete_sequence=PALETTE, title=f"Frequency of {_label(col)}")
        st.plotly_chart(_upd(fig, PLOT_BASE), use_container_width=True)
        st.dataframe(counts, use_container_width=True)


# ── Bivariate ───────────────────────────────────────────────────────────────

def _bivariate(df, PALETTE, PLOT_BASE):
    st.markdown("## Bivariate Explorer")
    st.markdown('<p style="font-size:0.83rem;color:#566174;">Pick any two variables to check '
                'for a relationship — this is how we justified pulling in socioeconomic data.</p>',
                unsafe_allow_html=True)

    num_cols = _numeric_cols(df)
    cat_cols = _categorical_cols(df)
    all_cols = num_cols + cat_cols

    c1, c2, c3, c4 = st.columns(4)
    x_col = c1.selectbox("X", all_cols, index=0, format_func=_label, key="bi_x")
    y_col = c2.selectbox("Y", all_cols, index=min(1, len(all_cols) - 1), format_func=_label, key="bi_y")
    color_col = c3.selectbox("Colour By", ["None"] + cat_cols, key="bi_color")
    color_col = None if color_col == "None" else color_col
    chart = c4.selectbox("Chart Type",
        ["Scatter Plot", "Line Chart", "Box Plot", "Bar Chart (mean)", "Correlation Heatmap"],
        key="bi_chart")

    if chart == "Correlation Heatmap":
        cols = [c for c in num_cols if c != "year"]
        corr = df[cols].corr().round(3)
        corr.index = [_label(c) for c in corr.index]
        corr.columns = [_label(c) for c in corr.columns]
        fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1, title="Pearson Correlation Matrix")
        st.plotly_chart(_upd(fig, PLOT_BASE), use_container_width=True)
        return

    sample = df.sample(min(len(df), 2000), random_state=42) if chart in ("Scatter Plot",) else df

    if chart == "Scatter Plot":
        fig = px.scatter(sample, x=x_col, y=y_col, color=color_col,
                         color_discrete_sequence=PALETTE, trendline="ols",
                         title=f"{_label(x_col)} vs {_label(y_col)}")
        fig.update_traces(marker_opacity=0.65)
    elif chart == "Line Chart":
        agg = df.groupby([x_col] + ([color_col] if color_col else []))[y_col].mean().reset_index()
        fig = px.line(agg, x=x_col, y=y_col, color=color_col, markers=True,
                     color_discrete_sequence=PALETTE, title=f"{_label(y_col)} by {_label(x_col)}")
    elif chart == "Box Plot":
        fig = px.box(df, x=x_col, y=y_col, color=color_col,
                    color_discrete_sequence=PALETTE, title=f"{_label(y_col)} by {_label(x_col)}")
        fig.update_layout(xaxis_tickangle=-30)
    else:  # Bar Chart (mean)
        agg = df.groupby([x_col] + ([color_col] if color_col else []))[y_col].mean().reset_index()
        fig = px.bar(agg, x=x_col, y=y_col, color=color_col, barmode="group",
                    color_discrete_sequence=PALETTE, title=f"Mean {_label(y_col)} by {_label(x_col)}")
        fig.update_layout(xaxis_tickangle=-30)

    st.plotly_chart(_upd(fig, PLOT_BASE), use_container_width=True)

    if x_col in num_cols and y_col in num_cols:
        r = df[[x_col, y_col]].corr().iloc[0, 1]
        st.markdown(f'<div class="stat-card">Pearson correlation between '
                    f'<b>{_label(x_col)}</b> and <b>{_label(y_col)}</b>: '
                    f'<span style="font-family:\'DM Mono\',monospace;color:#e9c46a;">{r:.3f}</span></div>',
                    unsafe_allow_html=True)


# ── Time series ─────────────────────────────────────────────────────────────

def _time_series(df, PALETTE, PLOT_BASE):
    st.markdown("## Time Series Explorer")
    st.markdown('<p style="font-size:0.83rem;color:#566174;">Trends, year-on-year change, and '
                'which clusters drive overall crime volume.</p>', unsafe_allow_html=True)

    clusters = sorted(df["Cluster"].unique())
    types = sorted(df["Type of Crime"].unique())
    c1, c2 = st.columns(2)
    sel_clusters = c1.multiselect("Clusters", clusters, default=clusters, key="ts_clusters")
    sel_types = c2.multiselect("Crime Types", types, default=types, key="ts_types")

    fdf = df[df["Cluster"].isin(sel_clusters) & df["Type of Crime"].isin(sel_types)]
    if fdf.empty:
        st.warning("No data for this selection.")
        return

    # Overall trend
    st.markdown('<div class="section-label">Overall Trend</div>', unsafe_allow_html=True)
    trend_by = st.radio("Break down by", ["Total", "Cluster", "Type of Crime"],
                        horizontal=True, key="ts_trend_by")
    if trend_by == "Total":
        agg = fdf.groupby("year")["Crime Count"].sum().reset_index()
        fig = px.line(agg, x="year", y="Crime Count", markers=True,
                     color_discrete_sequence=PALETTE, title="Total Crime Count Over Time")
    else:
        group_col = "Cluster" if trend_by == "Cluster" else "Type of Crime"
        agg = fdf.groupby(["year", group_col])["Crime Count"].sum().reset_index()
        fig = px.line(agg, x="year", y="Crime Count", color=group_col, markers=True,
                     color_discrete_sequence=PALETTE, title=f"Crime Count Over Time by {group_col}")
    st.plotly_chart(_upd(fig, PLOT_BASE), use_container_width=True)

    ts_c1, ts_c2 = st.columns(2)

    with ts_c1:
        st.markdown('<div class="section-label">Year-on-Year Change</div>', unsafe_allow_html=True)
        yoy = (fdf.groupby(["year", "Cluster"])["Crime Count"].sum()
                  .groupby("Cluster").pct_change().mul(100).reset_index())
        yoy.columns = ["year", "Cluster", "YoY %"]
        yoy = yoy.dropna()
        top_yoy_clusters = st.multiselect("Limit to clusters", clusters,
                                          default=clusters[:min(6, len(clusters))], key="ts_yoy_clusters")
        yoy_f = yoy[yoy["Cluster"].isin(top_yoy_clusters)]
        fig_yoy = px.line(yoy_f, x="year", y="YoY %", color="Cluster", markers=True,
                          color_discrete_sequence=PALETTE, title="Year-on-Year Crime Change (%)")
        fig_yoy.add_hline(y=0, line_color="#2a3347", line_dash="dot")
        st.plotly_chart(_upd(fig_yoy, PLOT_BASE), use_container_width=True)

    with ts_c2:
        st.markdown('<div class="section-label">Top Crime Regions</div>', unsafe_allow_html=True)
        top_n = st.slider("Show top N clusters", 3, len(clusters), min(10, len(clusters)), key="ts_topn")
        totals = (fdf.groupby(["Cluster", "year"])["Crime Count"].sum()
                     .groupby("Cluster").mean().reset_index()
                     .rename(columns={"Crime Count": "Mean Annual Crime Count"})
                     .sort_values("Mean Annual Crime Count", ascending=False).head(top_n))
        fig_top = px.bar(totals.sort_values("Mean Annual Crime Count"),
                         x="Mean Annual Crime Count", y="Cluster", orientation="h",
                         color_discrete_sequence=PALETTE, title=f"Top {top_n} Clusters by Mean Annual Crime Count")
        st.plotly_chart(_upd(fig_top, PLOT_BASE), use_container_width=True)

    st.markdown('<div class="section-label">Crime Type Composition Over Time</div>', unsafe_allow_html=True)
    ct_trend = fdf.groupby(["year", "Type of Crime"])["Crime Count"].sum().reset_index()
    fig_ct = px.area(ct_trend, x="year", y="Crime Count", color="Type of Crime",
                     color_discrete_sequence=PALETTE, title="Crime Count by Type — Stacked Trend")
    st.plotly_chart(_upd(fig_ct, PLOT_BASE), use_container_width=True)


# ── Enrichment justification ────────────────────────────────────────────────

def _justification(master_df, PALETTE, PLOT_BASE):
    st.markdown("## Enrichment Justification")
    st.markdown(
        '<p style="font-size:0.83rem;color:#566174;">The central hypothesis: crime-only data is '
        'feature-poor, and adding socioeconomic indicators should carry real predictive signal. '
        'These are the checks run before committing to that framing.</p>',
        unsafe_allow_html=True)

    corr_vals = {READABLE[s]: master_df[["Crime Count", s]].corr().iloc[0, 1] for s in SOCIO}
    corr_df = pd.DataFrame(list(corr_vals.items()), columns=["Feature", "Correlation"])
    corr_df = corr_df.sort_values("Correlation")
    corr_df["Colour"] = corr_df["Correlation"].apply(lambda x: "#2dc653" if x > 0 else "#ff4b4b")

    fig = go.Figure(go.Bar(
        x=corr_df["Correlation"], y=corr_df["Feature"], orientation="h",
        marker_color=corr_df["Colour"].tolist(),
        text=corr_df["Correlation"].round(3), textposition="outside",
        textfont=dict(color="#566174", size=11),
    ))
    fig.add_vline(x=0, line_color="#2a3347", line_width=1)
    _upd(fig, PLOT_BASE, title="Correlation with Crime Count — Socioeconomic Features",
         xaxis_range=[-1, 1], showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-label">Multicollinearity Check</div>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.8rem;color:#566174;">Tree ensembles (Random Forest, XGBoost, '
                'LightGBM, CatBoost) are not thrown off by correlated predictors the way linear models '
                'are — they just split on whichever correlated feature helps most at each node. That\'s '
                'why the enriched feature set keeps every raw socioeconomic column instead of applying '
                'PCA or dropping collinear pairs.</p>', unsafe_allow_html=True)
    corr_full = master_df[SOCIO].corr().round(3)
    corr_full.index = [READABLE[c] for c in corr_full.index]
    corr_full.columns = [READABLE[c] for c in corr_full.columns]
    fig2 = px.imshow(corr_full, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r",
                     zmin=-1, zmax=1, title="Socioeconomic Feature Multicollinearity")
    st.plotly_chart(_upd(fig2, PLOT_BASE), use_container_width=True)

    # ── VIF — quantifies collinearity per feature (not just pairwise) ──────
    st.markdown('<div class="section-label">Variance Inflation Factor (VIF)</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.8rem;color:#566174;line-height:1.6;">'
        'The correlation matrix above only shows pairs. VIF checks each feature against '
        '<em>all the others at once</em> — VIF for feature X is 1 / (1 − R²) from regressing X on every '
        'other socioeconomic feature. Above ~5 is usually flagged as concerning for linear models; '
        'above ~10 is severe.</p>', unsafe_allow_html=True)
    try:
        from sklearn.linear_model import LinearRegression
        vif_rows = []
        socio_data = master_df[SOCIO].dropna()
        for col in SOCIO:
            others = [c for c in SOCIO if c != col]
            r2 = LinearRegression().fit(socio_data[others], socio_data[col]).score(
                socio_data[others], socio_data[col])
            vif = float("inf") if r2 >= 0.999999 else 1.0 / (1.0 - r2)
            vif_rows.append({"Feature": READABLE[col], "R² vs. other features": round(r2, 4),
                             "VIF": round(vif, 1) if np.isfinite(vif) else "∞"})
        vif_df = pd.DataFrame(vif_rows).sort_values("R² vs. other features", ascending=False)
        st.dataframe(vif_df.set_index("Feature"), use_container_width=True)

        max_vif_row = vif_df.iloc[0]
        if isinstance(max_vif_row["VIF"], (int, float)) and max_vif_row["VIF"] > 10:
            severity, badge = "severe", "badge-critical"
        elif isinstance(max_vif_row["VIF"], (int, float)) and max_vif_row["VIF"] > 5:
            severity, badge = "moderate", "badge-high"
        else:
            severity, badge = "mild", "badge-low"

        st.markdown(f"""
        <div class="stat-card">
          <span class="{badge}">{severity.upper()} COLLINEARITY</span>
          <p style="font-size:0.82rem;color:#8892a4;margin:10px 0 0;line-height:1.7;">
            <b>This isn't a pipeline bug.</b> These four indicators are recorded as raw headcounts per
            cluster (poor households, unemployed people, etc.), not rates — so all four naturally scale
            with a cluster's population size. A more populous cluster tends to have more of everything
            in absolute terms, which is exactly the kind of "size effect" that shows up as strong
            correlation here. If Quantec exposes population totals for these clusters later, converting
            these to per-capita rates (e.g. unemployment ÷ population) would separate the underlying
            socioeconomic condition from cluster size and reduce this collinearity at the source.<br><br>
            <b>Given no rate denominator is currently available, and the models are all tree ensembles,</b>
            the practical fix isn't dropping features — collinear predictors don't bias tree splits or
            hurt predictive accuracy, only individual-feature SHAP attribution (see the Feature
            Importance page's "Grouped Importance" panel, which sums the correlated block into one
            number instead of diluting credit across four highly related columns).
          </p>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.info(f"VIF computation skipped: {e}")

    st.markdown("""
    <div class="stat-card">
      <div class="section-label">Why This Shaped the Modelling Approach</div>
      <p style="font-size:0.82rem;color:#8892a4;margin:6px 0 0;line-height:1.7;">
        <b>Target:</b> reframed from predicting Cluster (classification) to predicting Crime Count
        (regression) — the cluster-classification framing didn't hold up in practice.<br>
        <b>Models:</b> Random Forest, XGBoost, LightGBM, CatBoost only — all tree ensembles, so no
        scaling, log-transform, or PCA is needed, and multicollinearity between socioeconomic features
        (visible above) doesn't hurt them.<br>
        <b>Validation:</b> expanding-window walk-forward cross-validation, not a single train/test split
        — Crime Count is a time series, so folds respect chronological order.<br>
        <b>The actual test</b> of whether enrichment helps is the paired significance test on the
        Model Performance page — this page only shows the raw correlations that motivated the hypothesis.
      </p>
    </div>
    """, unsafe_allow_html=True)


# ── Main render ──────────────────────────────────────────────────────────────

def render(master_df: pd.DataFrame, crime_only_df, PALETTE: list, PLOT_BASE: dict):
    st.markdown('<div class="section-label">Module 01</div>', unsafe_allow_html=True)
    st.markdown("# Exploratory Data Analysis")
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    dataset_choice = st.radio("Dataset", ["Master (Crime + Socioeconomic)", "Crime Only"],
                              horizontal=True, key="eda_dataset")
    active_df = master_df if dataset_choice.startswith("Master") else crime_only_df
    if active_df is None:
        st.warning("This dataset isn't loaded — upload it in the sidebar.")
        return
    active_df = _prep(active_df)

    tabs = st.tabs(["Univariate Explorer", "Bivariate Explorer",
                    "Time Series Explorer", "Enrichment Justification"])

    with tabs[0]:
        _univariate(active_df, PALETTE, PLOT_BASE)
    with tabs[1]:
        _bivariate(active_df, PALETTE, PLOT_BASE)
    with tabs[2]:
        _time_series(active_df, PALETTE, PLOT_BASE)
    with tabs[3]:
        if master_df is None:
            st.warning("Enrichment justification needs the master dataset — upload it in the sidebar.")
        else:
            _justification(_prep(master_df), PALETTE, PLOT_BASE)