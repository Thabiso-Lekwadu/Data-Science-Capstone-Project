"""
Crime Hotspot Prediction – Intelligence Dashboard
Run: streamlit run App.py --server.port 8503

STRICTLY upload-driven: nothing on any page renders until the relevant
file(s) have been uploaded in this sidebar. There is no fallback to a local
data/ folder or any other default — closing/reopening the browser or
restarting the app clears everything back to empty, and pages only ever
reflect what's actually been uploaded in the current session.
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

from model_utils import (
    MODEL_NAMES, CONDITIONS, engineer_features, required_raw_columns_present,
    load_uploaded_models, models_for_condition,
)

PALETTE    = ["#e9c46a", "#4895ef", "#ff4b4b", "#2dc653", "#b48ead",
              "#f4a261", "#64b5f6", "#ef9a9a", "#66bb6a", "#ce93d8"]
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
        "SHAP Feature Importance",
        "Model Performance",
        "Prediction Explorer",
    ])

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Data Sources — Upload Only</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size:0.72rem;color:#566174;line-height:1.5;margin-bottom:10px;">'
        'Nothing renders until you upload it here. There is no local-disk fallback — '
        'restart the app or clear an upload and the corresponding charts disappear.</p>',
        unsafe_allow_html=True)

    up_master = st.file_uploader(
        "① Master dataset (crime + socioeconomic)", type=["xlsx", "csv"],
        help="master_dataset.xlsx — the raw merged dataset (date, Cluster, Type of Crime, "
             "Crime Count, socioeconomic columns), BEFORE feature engineering. "
             "Powers Exploratory Analysis directly, and is automatically feature-engineered "
             "in-app (one-hot + lag/rolling) to power SHAP and Prediction Explorer for the "
             "'master' condition — you do not need a second, pre-engineered upload for this.")
    up_crime_only = st.file_uploader(
        "② Crime-only dataset", type=["xlsx", "csv"],
        help="crime_processed.xlsx — same shape as ① but without socioeconomic columns. "
             "A genuinely different dataset (the baseline condition), not a duplicate of ①. "
             "Powers Exploratory Analysis and, engineered in-app, the 'crime_only' condition "
             "on SHAP and Prediction Explorer.")

    st.markdown('<div class="section-label" style="margin-top:14px;">Trained Models</div>', unsafe_allow_html=True)
    up_models = st.file_uploader(
        "③ Model files (.pkl)", type=["pkl"], accept_multiple_files=True,
        help="Upload the .pkl files from your Kedro run's data/06_models/ — e.g. "
             "randomforest_master.pkl, xgboost_crime_only.pkl. Matched automatically by "
             "filename. Needed for SHAP and Prediction Explorer.")

    st.markdown('<div class="section-label" style="margin-top:14px;">CV Results</div>', unsafe_allow_html=True)
    up_per_fold = st.file_uploader(
        "④ Per-fold CV results", type=["xlsx", "csv"],
        help="experiment_results_per_fold.xlsx. Powers Model Performance.")
    up_summary = st.file_uploader(
        "⑤ CV summary", type=["xlsx", "csv"],
        help="experiment_results_summary.xlsx. Powers Model Performance.")
    up_sig = st.file_uploader(
        "⑥ Significance tests", type=["xlsx", "csv"],
        help="experiment_significance_tests.xlsx. Powers Model Performance.")

    if st.button("🗑 Clear everything", help="Drops all cached uploads/computations for this session."):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown('<div style="font-family:\'DM Mono\',monospace;font-size:0.62rem;color:#2a3347;padding-bottom:8px;">v3.0 — Capstone Project (upload-only)</div>', unsafe_allow_html=True)


# ── Data loading — upload only, nothing else ────────────────────────────────
@st.cache_data(show_spinner=False)
def _load_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    import io
    buf = io.BytesIO(file_bytes)
    return pd.read_csv(buf) if filename.endswith(".csv") else pd.read_excel(buf, engine="openpyxl")


def _resolve_upload(uploaded, label: str):
    """Returns (dataframe_or_None, has_data_bool). No disk fallback of any kind."""
    if uploaded is None:
        return None, False
    try:
        return _load_file(uploaded.getvalue(), uploaded.name), True
    except Exception as e:
        st.sidebar.error(f"Could not read {label}: {e}")
        return None, False


master_df,     has_master = _resolve_upload(up_master,     "master dataset")
crime_only_df, has_crime  = _resolve_upload(up_crime_only,  "crime-only dataset")
per_fold_df,   has_pf     = _resolve_upload(up_per_fold,    "per-fold results")
summary_df,    has_summ   = _resolve_upload(up_summary,     "CV summary")
sig_df,        has_sig    = _resolve_upload(up_sig,         "significance tests")

# Uploaded models — matched to (model_name, condition) by filename pattern
uploaded_models, unmatched_model_files = load_uploaded_models(up_models)
if unmatched_model_files:
    st.sidebar.warning(
        f"{len(unmatched_model_files)} uploaded file(s) didn't match an expected model filename "
        f"and were skipped: {', '.join(unmatched_model_files)}")

# On-the-fly feature engineering, derived from the raw uploads — this is what
# replaces the old separate "master_dataset_features.xlsx" upload. Computed
# lazily (only if the corresponding raw dataset was uploaded), cached by the
# raw data's own content.
master_feat, crime_feat = None, None
if has_master:
    err = required_raw_columns_present(master_df)
    if err:
        st.sidebar.error(f"Master dataset: {err}")
    else:
        master_feat = engineer_features(master_df.to_json())
if has_crime:
    err = required_raw_columns_present(crime_only_df)
    if err:
        st.sidebar.error(f"Crime-only dataset: {err}")
    else:
        crime_feat = engineer_features(crime_only_df.to_json())

# Visible, page-agnostic proof of what's actually driving the charts right now.
_SOURCE_ROWS = [
    ("① master_dataset (raw)",        master_df,      has_master),
    ("② crime_processed (raw)",       crime_only_df,  has_crime),
    ("   → engineered (master)",      master_feat,    master_feat is not None),
    ("   → engineered (crime_only)",  crime_feat,      crime_feat is not None),
    ("③ trained models",              None,            len(uploaded_models) > 0),
    ("④ experiment_results_per_fold", per_fold_df,     has_pf),
    ("⑤ experiment_results_summary",  summary_df,      has_summ),
    ("⑥ experiment_significance_tests", sig_df,        has_sig),
]
with st.expander("📡  Active data sources — nothing below exists until it's uploaded", expanded=False):
    for _name, _df, _present in _SOURCE_ROWS:
        _icon = "🟢" if _present else "⚪"
        if _name.startswith("③"):
            _detail = f"{len(uploaded_models)} model file(s) matched" if _present else "not uploaded"
        else:
            _detail = f"{len(_df):,} rows" if (_df is not None) else "not uploaded"
        st.markdown(f'<span style="font-family:\'DM Mono\',monospace;font-size:0.78rem;">'
                    f'{_icon} <b>{_name}</b> — {_detail}</span>', unsafe_allow_html=True)


# ── Upload gate ───────────────────────────────────────────────────────────────
def _gate(ready: bool, needed: str) -> bool:
    if not ready:
        st.markdown(f"""
        <div style="background:#0c1220;border:1px solid #1a2235;border-radius:10px;
                    padding:3rem 2rem;text-align:center;margin-top:3rem;">
          <div style="font-family:'DM Mono',monospace;font-size:0.62rem;letter-spacing:0.14em;
                      color:#566174;text-transform:uppercase;margin-bottom:10px;">No Data Uploaded</div>
          <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;
                      color:#e2e8f0;margin-bottom:8px;">Upload required to continue</div>
          <div style="font-size:0.83rem;color:#566174;line-height:1.7;">
            Upload <span style="font-family:'DM Mono',monospace;color:#e9c46a;">{needed}</span>
            via the sidebar to view this page. Nothing is loaded from disk automatically.
          </div>
        </div>
        """, unsafe_allow_html=True)
        return False
    return True


# ── Route ─────────────────────────────────────────────────────────────────────
if page == "Exploratory Analysis":
    if _gate(master_df is not None or crime_only_df is not None,
             "① master dataset (or at least ② crime-only dataset)"):
        from pages_eda import render
        render(master_df, crime_only_df, PALETTE, PLOT_BASE)

elif page == "SHAP Feature Importance":
    if _gate(master_feat is not None or crime_feat is not None,
             "① master dataset (or ② crime-only dataset) — features are engineered automatically"):
        from pages_shap import render
        render(master_feat, crime_feat, uploaded_models, PALETTE, PLOT_BASE, MODEL_NAMES, CONDITIONS)

elif page == "Model Performance":
    if _gate(per_fold_df is not None and summary_df is not None,
             "④ per-fold CV results + ⑤ CV summary"):
        from pages_model import render
        render(per_fold_df, summary_df, sig_df, PALETTE, PLOT_BASE, MODEL_NAMES, CONDITIONS)

else:  # Prediction Explorer
    if _gate(master_feat is not None,
             "① master dataset — features are engineered automatically"):
        from pages_predict import render
        render(master_feat, uploaded_models, PALETTE, PLOT_BASE)