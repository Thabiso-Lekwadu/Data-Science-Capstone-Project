"""
Crime Hotspot Prediction – Intelligence Dashboard
Run: streamlit run App.py --server.port 8503
"""
from __future__ import annotations
import sys
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

import streamlit as st

st.set_page_config(
    page_title="Crime Hotspot Intelligence | SAPS",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');
*, html, body { box-sizing: border-box; }
.stApp { background: #070b14; color: #c9d1d9; font-family: 'DM Sans', sans-serif; }
section[data-testid="stSidebar"] { background: #0c1220; border-right: 1px solid #1a2235; padding-top: 0 !important; }
section[data-testid="stSidebar"] > div { padding-top: 0 !important; }
div[data-testid="stRadio"] > label { display: none; }
div[data-testid="stRadio"] > div { gap: 2px !important; }
div[data-testid="stRadio"] > div > label {
    background: transparent !important; border: none !important; border-radius: 6px !important;
    color: #566174 !important; font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important; font-weight: 500 !important;
    padding: 10px 16px !important; cursor: pointer !important;
    display: block !important; width: 100% !important;
}
div[data-testid="stRadio"] > div > label:hover { color: #e2e8f0 !important; background: #131c2e !important; }
[data-testid="stMetric"] { background: #0c1220; border: 1px solid #1a2235; border-radius: 10px; padding: 1.1rem 1.3rem 1rem; }
[data-testid="stMetricLabel"] p { color: #566174 !important; font-size: 0.7rem !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; font-family: 'DM Mono', monospace !important; }
[data-testid="stMetricValue"] { color: #e9c46a !important; font-family: 'DM Mono', monospace !important; font-size: 1.55rem !important; }
.stTabs [data-baseweb="tab-list"] { background: transparent; border-bottom: 1px solid #1a2235; gap: 0; }
.stTabs [data-baseweb="tab"] { background: transparent; border: none; color: #566174; font-family: 'DM Sans', sans-serif; font-size: 0.82rem; font-weight: 500; padding: 10px 20px; border-radius: 0; }
.stTabs [aria-selected="true"] { background: transparent !important; color: #e9c46a !important; border-bottom: 2px solid #e9c46a !important; }
div[data-testid="stSelectbox"] label p, div[data-testid="stMultiSelect"] label p,
div[data-testid="stSlider"] label p, div[data-testid="stNumberInput"] label p {
    color: #566174 !important; font-size: 0.72rem !important; letter-spacing: 0.07em !important;
    text-transform: uppercase !important; font-family: 'DM Mono', monospace !important;
}
h1 { font-family: 'Syne', sans-serif !important; font-weight: 800 !important; color: #e2e8f0 !important; font-size: 1.6rem !important; letter-spacing: -0.02em !important; }
h2 { font-family: 'Syne', sans-serif !important; font-weight: 700 !important; color: #e2e8f0 !important; font-size: 1.1rem !important; }
.section-label { font-family: 'DM Mono', monospace; font-size: 0.65rem; letter-spacing: 0.14em; text-transform: uppercase; color: #e9c46a; margin-bottom: 6px; }
.section-divider { border: none; border-top: 1px solid #1a2235; margin: 1.5rem 0; }
.stat-card { background: #0c1220; border: 1px solid #1a2235; border-radius: 10px; padding: 1rem 1.2rem; }
.badge-critical { display:inline-block; background:#ff4b4b18; border:1px solid #ff4b4b; color:#ff4b4b; border-radius:4px; padding:2px 10px; font-size:0.72rem; font-family:'DM Mono',monospace; font-weight:500; }
.badge-high     { display:inline-block; background:#e9c46a18; border:1px solid #e9c46a; color:#e9c46a; border-radius:4px; padding:2px 10px; font-size:0.72rem; font-family:'DM Mono',monospace; font-weight:500; }
.badge-medium   { display:inline-block; background:#4895ef18; border:1px solid #4895ef; color:#4895ef; border-radius:4px; padding:2px 10px; font-size:0.72rem; font-family:'DM Mono',monospace; font-weight:500; }
.badge-low      { display:inline-block; background:#2dc65318; border:1px solid #2dc653; color:#2dc653; border-radius:4px; padding:2px 10px; font-size:0.72rem; font-family:'DM Mono',monospace; font-weight:500; }
[data-testid="stDataFrame"] { border: 1px solid #1a2235; border-radius: 8px; overflow: hidden; }
div[data-testid="stFileUploader"] { border: 1px dashed #1a2235; border-radius: 8px; padding: 4px 8px; background: #0a0f1a; }
div[data-testid="stFileUploader"] label p { color: #566174 !important; font-size: 0.72rem !important; text-transform: uppercase !important; font-family: 'DM Mono', monospace !important; letter-spacing: 0.07em !important; }
</style>
""", unsafe_allow_html=True)

import pandas as pd

DATA_DIR   = Path("/app/data")
MODELS     = ["KNN","RandomForest","GradientBoosting","XGBoost"]
CONDITIONS = ["crime_only","master"]
PALETTE    = ["#e9c46a","#4895ef","#ff4b4b","#2dc653","#b48ead",
               "#f4a261","#64b5f6","#ef9a9a","#66bb6a","#ce93d8"]
PLOT_BASE  = dict(
    paper_bgcolor="#0c1220", plot_bgcolor="#070b14",
    font=dict(color="#566174", family="DM Sans"),
    title_font=dict(color="#e2e8f0", size=13, family="Syne"),
    xaxis=dict(gridcolor="#1a2235", linecolor="#1a2235", tickfont=dict(color="#566174", size=11)),
    yaxis=dict(gridcolor="#1a2235", linecolor="#1a2235", tickfont=dict(color="#566174", size=11)),
    legend=dict(bgcolor="#0c1220", bordercolor="#1a2235", font=dict(color="#8892a4", size=11)),
    margin=dict(t=50, b=40, l=55, r=20),
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="background:#0a0f1a;border-bottom:1px solid #1a2235;padding:20px 16px 18px;margin:-1rem -1rem 1.2rem;">
      <div style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:0.18em;color:#e9c46a;text-transform:uppercase;margin-bottom:6px;">Intelligence System</div>
      <div style="font-family:'Syne',sans-serif;font-size:1.15rem;font-weight:800;color:#e2e8f0;line-height:1.25;">Crime Hotspot<br>Prediction</div>
      <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#566174;margin-top:6px;">SAPS Analytical Platform</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("page", [
        "Exploratory Analysis",
        "Model Performance",
        "Command Dashboard",
    ])

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Data Sources</div>', unsafe_allow_html=True)

    up_master      = st.file_uploader("master_dataset.xlsx",          type=["xlsx","csv"])
    up_crime_only  = st.file_uploader("crime_processed.xlsx",         type=["xlsx","csv"])
    up_results     = st.file_uploader("experiment_results.xlsx",      type=["xlsx","csv"])

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:0.62rem;color:#2a3347;padding-bottom:8px;">v1.0 — Capstone Project</div>', unsafe_allow_html=True)


# ── Data loading ─────────────────────────────────────────────────────────────
@st.cache_data
def _load_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    import io
    buf = io.BytesIO(file_bytes)
    return pd.read_csv(buf) if filename.endswith(".csv") else pd.read_excel(buf, engine="openpyxl")


@st.cache_data
def _load_path(path_str: str) -> pd.DataFrame | None:
    p = Path(path_str)
    if not p.exists():
        return None
    try:
        return pd.read_excel(p, engine="openpyxl") if path_str.endswith("xlsx") else pd.read_csv(p)
    except Exception:
        return None


def _resolve(uploaded, docker_path: Path) -> pd.DataFrame | None:
    if uploaded is not None:
        try:
            return _load_file(uploaded.getvalue(), uploaded.name)
        except Exception as e:
            st.sidebar.error(f"Could not read file: {e}")
            return None
    return _load_path(str(docker_path))


def _prep_master(df):
    if df is None:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if "year" not in df.columns:
        df["year"] = df["date"].dt.year
    return df


def _prep_crime_only(df):
    """
    crime_processed has: date, Cluster, Type of Crime, Crime Count.
    Add year column if missing.
    """
    if df is None:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if "year" not in df.columns:
        df["year"] = df["date"].dt.year
    return df


master_df     = _prep_master(_resolve(up_master,     DATA_DIR / "03_primary/master_dataset.xlsx"))
crime_only_df = _prep_crime_only(_resolve(up_crime_only, DATA_DIR / "02_intermediate/crime_processed.xlsx"))
results_df    = _resolve(up_results, DATA_DIR / "08_reporting/experiment_comparison.xlsx")

# Derive cluster list from whichever dataset is available
CLUSTERS = (
    sorted(master_df["Cluster"].unique().tolist())     if master_df is not None
    else sorted(crime_only_df["Cluster"].unique().tolist()) if crime_only_df is not None
    else ["Bela Bela","Buffalo City","Cape Town","Ekurhuleni","eThekwini",
          "Johannesburg","Mangaung","Msunduzi","Nelson Mandela Bay","Tshwane"]
)


# ── Upload gate ───────────────────────────────────────────────────────────────
def _gate(ready: bool, needed: str) -> bool:
    if not ready:
        st.markdown(f"""
        <div style="background:#0c1220;border:1px solid #1a2235;border-radius:10px;
                    padding:3rem 2rem;text-align:center;margin-top:3rem;">
          <div style="font-family:'DM Mono',monospace;font-size:0.62rem;letter-spacing:0.14em;
                      color:#566174;text-transform:uppercase;margin-bottom:10px;">No Data Loaded</div>
          <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;
                      color:#e2e8f0;margin-bottom:8px;">Upload required to continue</div>
          <div style="font-size:0.83rem;color:#566174;line-height:1.7;">
            Upload <span style="font-family:'DM Mono',monospace;color:#e9c46a;">{needed}</span>
            via the sidebar to view this page.
          </div>
        </div>
        """, unsafe_allow_html=True)
        return False
    return True


# ── Route ─────────────────────────────────────────────────────────────────────
if page == "Exploratory Analysis":
    if _gate(master_df is not None, "master_dataset.xlsx"):
        from pages_eda import render
        render(master_df, crime_only_df, PALETTE, PLOT_BASE)

elif page == "Model Performance":
    if _gate(results_df is not None, "experiment_results.xlsx"):
        from pages_model import render
        render(results_df, PALETTE, PLOT_BASE, MODELS, CONDITIONS,
               master_df=master_df, crime_only_df=crime_only_df)

else:
    needed = "master_dataset.xlsx  +  experiment_results.xlsx"
    if _gate(master_df is not None and results_df is not None, needed):
        from pages_bi import render
        render(master_df, results_df, PALETTE, PLOT_BASE, CLUSTERS, MODELS, CONDITIONS)