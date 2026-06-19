"""Page 1 — Exploratory Data Analysis (Research Questions)."""
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

NUMERIC = ["Crime Count", "population_density", "poor_households",
           "population_unemployment", "population_education"]


def _dfhash(df: pd.DataFrame) -> str:
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()[:12]


@st.cache_data(show_spinner=False)
def _agg_cluster_year(data_hash: str, df_json: str):
    df = pd.read_json(df_json)
    return df.groupby(["year", "Cluster"])["Crime Count"].sum().reset_index()

@st.cache_data(show_spinner=False)
def _agg_cluster_total(data_hash: str, df_json: str):
    df = pd.read_json(df_json)
    return (df.groupby(["Cluster", "year"])["Crime Count"].sum()
              .groupby("Cluster").mean().reset_index()
              .rename(columns={"Crime Count": "Mean Annual Crime Count"})
              .sort_values("Mean Annual Crime Count", ascending=False))

@st.cache_data(show_spinner=False)
def _agg_yoy(data_hash: str, df_json: str):
    df = pd.read_json(df_json)
    yoy = (df.groupby(["year", "Cluster"])["Crime Count"].sum()
             .groupby("Cluster").pct_change().mul(100).reset_index())
    yoy.columns = ["year", "Cluster", "YoY %"]
    return yoy.dropna()

@st.cache_data(show_spinner=False)
def _agg_hotspot_flags(data_hash: str, df_json: str):
    df = pd.read_json(df_json)
    yr_agg = df.groupby(["year", "Cluster"])["Crime Count"].sum().reset_index()
    q75 = yr_agg.groupby("year")["Crime Count"].quantile(0.75).rename("q75")
    yr_agg = yr_agg.merge(q75, on="year")
    yr_agg["Hotspot"] = (yr_agg["Crime Count"] >= yr_agg["q75"]).astype(int)
    return yr_agg

@st.cache_data(show_spinner=False)
def _agg_corr(data_hash: str, df_json: str, cols_key: str):
    df = pd.read_json(df_json)
    cols = cols_key.split("|")
    available = [c for c in cols if c in df.columns]
    return df[available].corr().round(3)

READABLE = {
    "Crime Count":              "Crime Count",
    "population_density":       "Population Density",
    "poor_households":          "Poor Households",
    "population_unemployment":  "Population Unemployment",
    "population_education":     "Population Education",
}


def _upd(fig, base, **kw):
    fig.update_layout(**{**base, **kw})
    return fig


# ── chart builders ──────────────────────────────────────────────────────────

def _bar(df, x, y, color_col, palette, base, title, orient="v", **kw):
    if orient == "h":
        fig = px.bar(df, x=y, y=x, orientation="h", color=color_col,
                     color_discrete_sequence=palette, title=title)
    else:
        fig = px.bar(df, x=x, y=y, color=color_col,
                     color_discrete_sequence=palette, title=title)
    return _upd(fig, base, **kw)


def _line(df, x, y, color, palette, base, title, **kw):
    fig = px.line(df, x=x, y=y, color=color, color_discrete_sequence=palette,
                  markers=True, title=title)
    fig.update_traces(line_width=2, marker_size=6)
    return _upd(fig, base, **kw)


def _area(df, x, y, color, palette, base, title, **kw):
    fig = px.area(df, x=x, y=y, color=color,
                  color_discrete_sequence=palette, title=title)
    return _upd(fig, base, **kw)


def _scatter(df, x, y, color, size, palette, base, title, **kw):
    fig = px.scatter(df, x=x, y=y, color=color, size=size, size_max=18,
                     color_discrete_sequence=palette,
                     trendline="ols", title=title)
    fig.update_traces(marker_opacity=0.7)
    return _upd(fig, base, **kw)


def _box(df, x, y, color, palette, base, title, **kw):
    fig = px.box(df, x=x, y=y, color=color,
                 color_discrete_sequence=palette, title=title)
    fig.update_layout(showlegend=False)
    return _upd(fig, base, xaxis_tickangle=-30, **kw)


def _strip(df, x, y, color, palette, base, title, **kw):
    fig = px.strip(df, x=x, y=y, color=color,
                   color_discrete_sequence=palette, title=title)
    return _upd(fig, base, xaxis_tickangle=-30, **kw)


def _heatmap_pivot(pivot, base, title, cs="YlOrBr"):
    fig = px.imshow(pivot, aspect="auto", color_continuous_scale=cs, title=title)
    fig.update_coloraxes(colorbar_tickfont_color="#566174")
    return _upd(fig, base)


def _histogram(df, col, color_col, palette, base, title, nbins=50, **kw):
    fig = px.histogram(df, x=col, nbins=nbins, color=color_col,
                       color_discrete_sequence=palette, barmode="overlay",
                       title=title)
    fig.update_traces(opacity=0.75)
    return _upd(fig, base, **kw)


def _render_chart(chart_type, rq_df, x, y, color, size_col, palette, base, title):
    """Dispatch to the chosen chart type."""
    ct = chart_type
    if ct == "Bar Chart":
        return _bar(rq_df, x, y, color, palette, base, title, orient="v")
    if ct == "Horizontal Bar":
        return _bar(rq_df, x, y, color, palette, base, title, orient="h")
    if ct == "Line Chart":
        return _line(rq_df, x, y, color, palette, base, title)
    if ct == "Area Chart":
        return _area(rq_df, x, y, color, palette, base, title)
    if ct == "Scatter Plot":
        return _scatter(rq_df, x, y, color, size_col, palette, base, title)
    if ct == "Box Plot":
        return _box(rq_df, x, y, color, palette, base, title)
    if ct == "Strip Plot":
        return _strip(rq_df, x, y, color, palette, base, title)
    if ct == "Histogram":
        return _histogram(rq_df, y, color, palette, base, title)
    # fallback
    return _bar(rq_df, x, y, color, palette, base, title)


# ── main render ─────────────────────────────────────────────────────────────

def render(df: pd.DataFrame, crime_only_df, PALETTE: list, PLOT_BASE: dict):
    st.markdown('<div class="section-label">Module 01</div>', unsafe_allow_html=True)
    st.markdown("# Exploratory Data Analysis")
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if "year" not in df.columns:
        df["year"] = df["date"].dt.year

    dh = _dfhash(df)

    # KPIs
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Records",          f"{len(df):,}")
    k2.metric("Clusters",         df["Cluster"].nunique())
    k3.metric("Crime Categories", df["Type of Crime"].nunique())
    k4.metric("Year Range",       f"{df['year'].min()}–{df['year'].max()}")
    k5.metric("Avg Crime Count",  f"{df['Crime Count'].mean():,.0f}")
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Global filters
    fc1, fc2, fc3 = st.columns([2, 2, 1])
    clusters = fc1.multiselect("Filter — Cluster",
        sorted(df["Cluster"].unique()), default=list(df["Cluster"].unique()))
    crime_types = fc2.multiselect("Filter — Crime Type",
        sorted(df["Type of Crime"].unique()), default=list(df["Type of Crime"].unique()))
    yr = fc3.slider("Year Range",
        int(df["year"].min()), int(df["year"].max()),
        (int(df["year"].min()), int(df["year"].max())))

    fdf = df[
        df["Cluster"].isin(clusters or df["Cluster"].unique()) &
        df["Type of Crime"].isin(crime_types or df["Type of Crime"].unique()) &
        df["year"].between(*yr)
    ].copy()

    if fdf.empty:
        st.warning("No records match current filters.")
        return

    tabs = st.tabs([
        "Univariate Analysis",
        "Top Crime Locations",
        "Crime Patterns",
        "Seasonal Trends",
        "Hotspot Persistence",
        "Crime Predictors",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 0 — UNIVARIATE ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[0]:
        st.markdown("## Univariate Analysis")
        st.markdown(
            '<p style="font-size:0.83rem;color:#566174;">Explore the distribution of each feature '
            'individually. Choose a feature and chart type to understand its spread, central tendency, '
            'and outliers.</p>',
            unsafe_allow_html=True)

        u1, u2 = st.columns([2, 1])
        uni_col   = u1.selectbox("Feature to Analyse", NUMERIC, key="uni_col")
        uni_chart = u2.selectbox("Chart Type",
            ["Histogram", "Box Plot", "Bar Chart (Mean by Cluster)",
             "Bar Chart (Total by Cluster)", "Bar Chart (Mean by Crime Type)"],
            key="uni_chart")

        if uni_chart == "Histogram":
            n_bins = st.slider("Number of bins", 10, 100, 40, key="uni_bins")
            fig_uni = go.Figure(go.Histogram(
                x=fdf[uni_col].dropna(), nbinsx=n_bins,
                marker_color="#e9c46a", opacity=0.8,
                marker_line_color="#070b14", marker_line_width=0.5,
            ))
            fig_uni.update_layout(**PLOT_BASE, title=f"Histogram — {uni_col}",
                                  bargap=0.02, showlegend=False,
                                  xaxis_title=uni_col, yaxis_title="Count")
            st.plotly_chart(fig_uni, use_container_width=True)

            # Summary stats alongside
            st.markdown('<div class="section-label">Distribution Summary</div>', unsafe_allow_html=True)
            s = fdf[uni_col].describe()
            sc1,sc2,sc3,sc4,sc5 = st.columns(5)
            sc1.metric("Mean",   f"{s['mean']:.1f}")
            sc2.metric("Median", f"{fdf[uni_col].median():.1f}")
            sc3.metric("Std",    f"{s['std']:.1f}")
            sc4.metric("Min",    f"{s['min']:.1f}")
            sc5.metric("Max",    f"{s['max']:.1f}")

        elif uni_chart == "Box Plot":
            group_by = st.selectbox("Group by", ["Cluster","Type of Crime"], key="uni_box_grp")
            fig_uni = px.box(fdf, x=group_by, y=uni_col, color=group_by,
                             color_discrete_sequence=PALETTE,
                             title=f"Box Plot — {uni_col} by {group_by}")
            fig_uni.update_layout(**PLOT_BASE, xaxis_tickangle=-30, showlegend=False)
            st.plotly_chart(fig_uni, use_container_width=True)

        elif uni_chart == "Bar Chart (Mean by Cluster)":
            agg_u = fdf.groupby("Cluster")[uni_col].mean().reset_index().sort_values(uni_col, ascending=True)
            fig_uni = go.Figure(go.Bar(
                x=agg_u[uni_col], y=agg_u["Cluster"], orientation="h",
                marker_color="#e9c46a", opacity=0.8,
                text=agg_u[uni_col].round(0).astype(int),
                textposition="outside", textfont=dict(color="#566174", size=11),
            ))
            fig_uni.update_layout(**PLOT_BASE, title=f"Mean {uni_col} by Cluster",
                                  xaxis_title=f"Mean {uni_col}", showlegend=False)
            st.plotly_chart(fig_uni, use_container_width=True)

        elif uni_chart == "Bar Chart (Total by Cluster)":
            agg_u = fdf.groupby("Cluster")[uni_col].sum().reset_index().sort_values(uni_col, ascending=True)
            fig_uni = go.Figure(go.Bar(
                x=agg_u[uni_col], y=agg_u["Cluster"], orientation="h",
                marker_color="#4895ef", opacity=0.8,
                text=agg_u[uni_col].round(0).astype(int),
                textposition="outside", textfont=dict(color="#566174", size=11),
            ))
            fig_uni.update_layout(**PLOT_BASE, title=f"Total {uni_col} by Cluster",
                                  xaxis_title=f"Total {uni_col}", showlegend=False)
            st.plotly_chart(fig_uni, use_container_width=True)

        else:  # Mean by Crime Type
            agg_u = fdf.groupby("Type of Crime")[uni_col].mean().reset_index().sort_values(uni_col, ascending=True)
            fig_uni = go.Figure(go.Bar(
                x=agg_u[uni_col], y=agg_u["Type of Crime"], orientation="h",
                marker_color="#2dc653", opacity=0.8,
                text=agg_u[uni_col].round(0).astype(int),
                textposition="outside", textfont=dict(color="#566174", size=11),
            ))
            fig_uni.update_layout(**PLOT_BASE, title=f"Mean {uni_col} by Crime Type",
                                  xaxis_title=f"Mean {uni_col}", showlegend=False)
            st.plotly_chart(fig_uni, use_container_width=True)

        # Descriptive stats table always shown
        st.markdown('<div class="section-label">Full Descriptive Statistics</div>', unsafe_allow_html=True)
        st.dataframe(fdf[NUMERIC].describe().T.round(2), use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # RQ1 — Top 5 locations with highest crime rate
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[1]:
        st.markdown("## What are the top locations with the highest crime rate?")
        st.markdown(
            '<p style="font-size:0.83rem;color:#566174;">Comparing total recorded incidents per cluster '
            'helps identify where law enforcement resources are most urgently needed.</p>',
            unsafe_allow_html=True)

        rq1c1, rq1c2 = st.columns([2, 1])
        n_top = rq1c1.slider("Number of locations to show", 3, len(df["Cluster"].unique()), 5, key="rq1_n")
        chart1 = rq1c2.selectbox("Chart Type",
            ["Horizontal Bar", "Bar Chart", "Line Chart"], key="rq1_chart")

        # Use mean annual crime count per cluster to avoid inflating totals
        # by summing across all years and all crime types simultaneously.
        # Mean annual count reflects typical yearly crime burden per cluster.
        cluster_total = (
            fdf.groupby(["Cluster","year"])["Crime Count"].sum()  # total per cluster per year
               .groupby("Cluster").mean()                          # average across years
               .reset_index()
               .rename(columns={"Crime Count":"Mean Annual Crime Count"})
               .sort_values("Mean Annual Crime Count", ascending=False)
               .head(n_top)
        )
        cluster_total["Crime Count"] = cluster_total["Mean Annual Crime Count"].round(0).astype(int)

        fig1 = _render_chart(chart1, cluster_total,
                             x="Cluster", y="Crime Count",
                             color="Cluster", size_col="Crime Count",
                             palette=PALETTE, base=PLOT_BASE,
                             title=f"Top {n_top} Clusters by Total Crime Count")
        fig1.update_layout(showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)

        # Breakdown by crime type for the top clusters
        st.markdown('<div class="section-label">Crime Type Breakdown — Top Clusters</div>', unsafe_allow_html=True)
        top_names = cluster_total["Cluster"].tolist()
        breakdown = (fdf[fdf["Cluster"].isin(top_names)]
                       .groupby(["Cluster", "Type of Crime"])["Crime Count"]
                       .sum().reset_index())
        chart1b = st.selectbox("Breakdown Chart Type",
            ["Bar Chart", "Horizontal Bar", "Area Chart"], key="rq1b_chart")

        fig1b = _render_chart(chart1b, breakdown,
                              x="Cluster", y="Crime Count",
                              color="Type of Crime", size_col="Crime Count",
                              palette=PALETTE, base=PLOT_BASE,
                              title="Crime Type Split — Top Clusters")
        st.plotly_chart(fig1b, use_container_width=True)

        # Summary table
        st.markdown('<div class="section-label">Summary Table</div>', unsafe_allow_html=True)
        tbl = cluster_total.copy()
        tbl["% of Total"] = (tbl["Crime Count"] / fdf["Crime Count"].sum() * 100).round(2)
        tbl["Rank"] = range(1, len(tbl) + 1)
        st.dataframe(tbl[["Rank", "Cluster", "Crime Count", "% of Total"]].set_index("Rank"),
                     use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # RQ2 — When is crime most likely to occur?
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[2]:
        st.markdown("## When is crime most likely to occur?")
        st.markdown(
            '<p style="font-size:0.83rem;color:#566174;">Analysing crime by year reveals which periods '
            'see the most activity. With annual data, year-level patterns show long-term surges and drops.</p>',
            unsafe_allow_html=True)

        rq2c1, rq2c2 = st.columns([2, 1])
        chart2 = rq2c1.selectbox("Chart Type",
            ["Bar Chart", "Line Chart", "Area Chart", "Box Plot"], key="rq2_chart")
        color2 = rq2c2.selectbox("Colour By", ["Cluster", "Type of Crime"], key="rq2_color")

        year_agg = fdf.groupby(["year", color2])["Crime Count"].sum().reset_index()
        fig2 = _render_chart(chart2, year_agg,
                             x="year", y="Crime Count",
                             color=color2, size_col="Crime Count",
                             palette=PALETTE, base=PLOT_BASE,
                             title=f"Crime Count by Year — coloured by {color2}")
        st.plotly_chart(fig2, use_container_width=True)

        # Annual total with trend line
        st.markdown('<div class="section-label">Overall Annual Trend</div>', unsafe_allow_html=True)
        annual_total = fdf.groupby("year")["Crime Count"].sum().reset_index()
        fig2b = go.Figure()
        fig2b.add_trace(go.Bar(x=annual_total["year"], y=annual_total["Crime Count"],
                               marker_color="#e9c46a", opacity=0.7, name="Annual Total"))
        # simple moving average
        if len(annual_total) >= 3:
            ma = annual_total["Crime Count"].rolling(3, center=True).mean()
            fig2b.add_trace(go.Scatter(x=annual_total["year"], y=ma, mode="lines",
                                       name="3-Year Moving Avg",
                                       line=dict(color="#ff4b4b", width=2.5)))
        _upd(fig2b, PLOT_BASE, title="Total Crimes per Year with 3-Year Moving Average",
             xaxis_title="Year", yaxis_title="Total Crime Count")
        st.plotly_chart(fig2b, use_container_width=True)

        # Peak year callout
        peak_yr = annual_total.loc[annual_total["Crime Count"].idxmax()]
        st.markdown(f"""
        <div class="stat-card">
          <div class="section-label">Peak Year</div>
          <p style="font-size:0.83rem;color:#8892a4;margin:6px 0 0;">
            The highest total crime count in the filtered dataset occurred in
            <span style="color:#e9c46a;font-family:'DM Mono',monospace;">{int(peak_yr['year'])}</span>
            with <span style="color:#e9c46a;font-family:'DM Mono',monospace;">{int(peak_yr['Crime Count']):,}</span> incidents.
          </p>
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # RQ3 — Seasonal patterns or long-term trends
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[3]:
        st.markdown("## Are there seasonal patterns or long-term trends in crime rates?")
        st.markdown(
            '<p style="font-size:0.83rem;color:#566174;">Looking at how crime changes year-on-year '
            'and across clusters reveals structural trends versus short-term fluctuations.</p>',
            unsafe_allow_html=True)

        rq3c1, rq3c2 = st.columns([2, 1])
        chart3 = rq3c1.selectbox("Chart Type",
            ["Heatmap", "Line Chart", "Area Chart", "Bar Chart"], key="rq3_chart")
        sel_cl3 = rq3c2.multiselect("Select Clusters",
            sorted(fdf["Cluster"].unique()),
            default=list(fdf["Cluster"].unique()[:5]), key="rq3_cl")

        sub3 = fdf[fdf["Cluster"].isin(sel_cl3 or fdf["Cluster"].unique())]

        if chart3 == "Heatmap":
            pivot3 = sub3.pivot_table(values="Crime Count", index="Cluster",
                                      columns="year", aggfunc="sum")
            fig3 = _heatmap_pivot(pivot3, PLOT_BASE,
                                  "Crime Intensity Heatmap — Cluster × Year", cs="YlOrBr")
        else:
            ts3 = sub3.groupby(["year", "Cluster"])["Crime Count"].sum().reset_index()
            fig3 = _render_chart(chart3, ts3, x="year", y="Crime Count",
                                 color="Cluster", size_col="Crime Count",
                                 palette=PALETTE, base=PLOT_BASE,
                                 title="Crime Trend by Cluster over Time")
        st.plotly_chart(fig3, use_container_width=True)

        # Year-on-year % change
        st.markdown('<div class="section-label">Year-on-Year Change (%)</div>', unsafe_allow_html=True)
        yoy = (sub3.groupby(["year", "Cluster"])["Crime Count"].sum()
                   .groupby("Cluster").pct_change().mul(100).reset_index())
        yoy.columns = ["year", "Cluster", "YoY Change (%)"]
        yoy = yoy.dropna()

        chart3b = st.selectbox("YoY Chart Type",
            ["Bar Chart", "Line Chart", "Scatter Plot"], key="rq3b_chart")
        fig3b = _render_chart(chart3b, yoy,
                              x="year", y="YoY Change (%)",
                              color="Cluster", size_col=None,
                              palette=PALETTE, base=PLOT_BASE,
                              title="Year-on-Year Crime Change (%)")
        if chart3b == "Bar Chart":
            fig3b.update_layout(barmode="group")
        fig3b.add_hline(y=0, line_color="#2a3347", line_dash="dot")
        st.plotly_chart(fig3b, use_container_width=True)

        # Crime type trend
        st.markdown('<div class="section-label">Crime Type Long-term Trend</div>', unsafe_allow_html=True)
        ct_trend = fdf.groupby(["year", "Type of Crime"])["Crime Count"].sum().reset_index()
        chart3c = st.selectbox("Crime Type Trend Chart",
            ["Line Chart", "Area Chart", "Bar Chart"], key="rq3c_chart")
        fig3c = _render_chart(chart3c, ct_trend,
                              x="year", y="Crime Count",
                              color="Type of Crime", size_col="Crime Count",
                              palette=PALETTE, base=PLOT_BASE,
                              title="Crime Count by Type — Annual Trend")
        st.plotly_chart(fig3c, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # RQ4 — How frequently do hotspots persist vs dissipate?
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[4]:
        st.markdown("## How frequently do hotspots persist versus dissipate over time?")
        st.markdown(
            '<p style="font-size:0.83rem;color:#566174;">A cluster is flagged as a hotspot in a given year '
            'if its crime count exceeds the 75th percentile across all clusters that year. '
            'Persistence is measured by how many consecutive years a cluster stays in the top tier.</p>',
            unsafe_allow_html=True)

        # Build hotspot flag per cluster per year
        yr_agg = fdf.groupby(["year", "Cluster"])["Crime Count"].sum().reset_index()
        thresh_by_year = yr_agg.groupby("year")["Crime Count"].quantile(0.75).rename("q75")
        yr_agg = yr_agg.merge(thresh_by_year, on="year")
        yr_agg["Hotspot"] = yr_agg["Crime Count"] >= yr_agg["q75"]

        # Persistence count
        persist = yr_agg.groupby("Cluster")["Hotspot"].sum().reset_index()
        persist.columns = ["Cluster", "Years as Hotspot"]
        persist = persist.sort_values("Years as Hotspot", ascending=False)
        total_years = yr_agg["year"].nunique()
        persist["% Time as Hotspot"] = (persist["Years as Hotspot"] / total_years * 100).round(1)
        persist["Status"] = persist["Years as Hotspot"].apply(
            lambda x: "Persistent" if x >= total_years * 0.6
            else "Intermittent" if x >= total_years * 0.3
            else "Transient")

        rq4c1, rq4c2 = st.columns([2, 1])
        chart4 = rq4c1.selectbox("Chart Type",
            ["Horizontal Bar", "Bar Chart", "Scatter Plot"], key="rq4_chart")

        fig4 = _render_chart(chart4, persist,
                             x="Cluster", y="Years as Hotspot",
                             color="Status", size_col="Years as Hotspot",
                             palette=["#ff4b4b", "#e9c46a", "#2dc653"],
                             base=PLOT_BASE,
                             title="Hotspot Persistence — Years Each Cluster Exceeded 75th Percentile")
        st.plotly_chart(fig4, use_container_width=True)

        # Hotspot presence over time — tile chart
        st.markdown('<div class="section-label">Hotspot Presence — Year by Cluster</div>', unsafe_allow_html=True)
        pivot4 = yr_agg.pivot_table(values="Hotspot", index="Cluster",
                                    columns="year", aggfunc="sum").fillna(0)
        fig4b = px.imshow(pivot4.astype(int), aspect="auto",
                          color_continuous_scale=["#0c1220", "#e9c46a"],
                          title="Hotspot Flag per Year (1 = Above 75th Percentile)")
        fig4b.update_coloraxes(colorbar_tickfont_color="#566174", showscale=False)
        _upd(fig4b, PLOT_BASE)
        st.plotly_chart(fig4b, use_container_width=True)

        st.markdown('<div class="section-label">Persistence Classification</div>', unsafe_allow_html=True)
        tbl4 = persist[["Cluster", "Years as Hotspot", "% Time as Hotspot", "Status"]].set_index("Cluster")
        st.dataframe(tbl4, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # RQ5 — What factors contribute most to predicting crime?
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[5]:
        st.markdown("## What factors contribute most to predicting crime?")
        st.markdown(
            '<p style="font-size:0.83rem;color:#566174;">Exploring how socioeconomic variables relate '
            'to crime count shows which features carry predictive signal. '
            'Strong correlations suggest a variable will be useful in the model.</p>',
            unsafe_allow_html=True)

        SOCIO = ["population_density", "poor_households",
                 "population_unemployment", "population_education"]
        SOCIO_LABEL = {s: READABLE[s] for s in SOCIO}

        # Correlation bar — each socioeconomic feature vs Crime Count
        corr_vals = {READABLE[s]: fdf[["Crime Count", s]].corr().iloc[0, 1] for s in SOCIO}
        corr_df = pd.DataFrame(list(corr_vals.items()), columns=["Feature", "Correlation"])
        corr_df = corr_df.sort_values("Correlation", ascending=True)
        corr_df["Colour"] = corr_df["Correlation"].apply(lambda x: "#2dc653" if x > 0 else "#ff4b4b")

        fig5a = go.Figure(go.Bar(
            x=corr_df["Correlation"], y=corr_df["Feature"],
            orientation="h",
            marker_color=corr_df["Colour"].tolist(),
            text=corr_df["Correlation"].round(3),
            textposition="outside",
            textfont=dict(color="#566174", size=11),
        ))
        fig5a.add_vline(x=0, line_color="#2a3347", line_width=1)
        _upd(fig5a, PLOT_BASE,
             title="Correlation with Crime Count — Socioeconomic Features",
             xaxis_title="Pearson Correlation",
             xaxis_range=[-1, 1], showlegend=False)
        st.plotly_chart(fig5a, use_container_width=True)

        # Interactive scatter — choose socioeconomic feature vs crime count
        st.markdown('<div class="section-label">Relationship Explorer</div>', unsafe_allow_html=True)
        r5c1, r5c2, r5c3 = st.columns(3)
        sel_feat = r5c1.selectbox("Socioeconomic Feature",
            [READABLE[s] for s in SOCIO], key="rq5_feat")
        chart5b   = r5c2.selectbox("Chart Type",
            ["Scatter Plot", "Box Plot", "Bar Chart", "Histogram"], key="rq5_chart")
        color5    = r5c3.selectbox("Colour By", ["Cluster", "Type of Crime"], key="rq5_color")

        raw_feat = {v: k for k, v in READABLE.items()}[sel_feat]
        sample5 = fdf.sample(min(len(fdf), 1500), random_state=42)

        if chart5b == "Histogram":
            fig5b = _histogram(sample5, raw_feat, color5, PALETTE, PLOT_BASE,
                               title=f"Distribution of {sel_feat}")
        elif chart5b == "Box Plot":
            fig5b = _box(sample5, color5, "Crime Count", color5, PALETTE, PLOT_BASE,
                         title=f"Crime Count Distribution by {color5}")
        elif chart5b == "Bar Chart":
            agg5 = sample5.groupby([color5, raw_feat])["Crime Count"].mean().reset_index()
            fig5b = _bar(agg5, color5, "Crime Count", color5, PALETTE, PLOT_BASE,
                         title=f"Avg Crime Count by {color5}")
        else:
            fig5b = _scatter(sample5, raw_feat, "Crime Count", color5,
                             "Crime Count", PALETTE, PLOT_BASE,
                             title=f"{sel_feat} vs Crime Count (with trend line)")
        st.plotly_chart(fig5b, use_container_width=True)

        # Full correlation matrix
        st.markdown('<div class="section-label">Full Correlation Matrix</div>', unsafe_allow_html=True)
        corr_full = fdf[NUMERIC].corr().round(3)
        corr_full.index   = [READABLE.get(c, c) for c in corr_full.index]
        corr_full.columns = [READABLE.get(c, c) for c in corr_full.columns]
        fig5c = px.imshow(corr_full, text_auto=True, aspect="auto",
                          color_continuous_scale="RdBu_r",
                          title="Pearson Correlation — All Numeric Features",
                          zmin=-1, zmax=1)
        _upd(fig5c, PLOT_BASE)
        fig5c.update_coloraxes(colorbar_tickfont_color="#566174")
        st.plotly_chart(fig5c, use_container_width=True)

        st.markdown("""
        <div class="stat-card">
          <div class="section-label">How to Read This</div>
          <p style="font-size:0.82rem;color:#8892a4;margin:6px 0 0;line-height:1.6;">
            Values close to <span style="color:#2dc653;font-family:'DM Mono',monospace;">+1</span>
            mean the two features increase together — a strong positive predictor of crime.
            Values close to <span style="color:#ff4b4b;font-family:'DM Mono',monospace;">–1</span>
            mean they move in opposite directions.
            Values near <span style="font-family:'DM Mono',monospace;">0</span> suggest little linear relationship.
            Features with high absolute correlation to Crime Count are the most useful predictors for the model.
          </p>
        </div>
        """, unsafe_allow_html=True)