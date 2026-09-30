import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
 
from utils import load_data
from preprocessing import preprocess_pipeline
from feature_engineering import engineer_features
 
# --------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------
st.set_page_config(
    page_title="Outlier & Distribution Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# --------------------------------------------------------
# DATA LOADING — full pipeline (raw -> cleaned -> engineered)
# --------------------------------------------------------
DATA_PATH = r"C:\Users\pavan\Downloads\dashboard\application_train.csv"
 
 
@st.cache_data
def get_data(path: str) -> pd.DataFrame:
    df = load_data(path)
    df = preprocess_pipeline(df)
    df = engineer_features(df)
    return df
 
 
try:
    df = get_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"Could not find file at: {DATA_PATH}\n\nUpdate DATA_PATH at the top of this file.")
    st.stop()
 
# --------------------------------------------------------
# HEADER
# --------------------------------------------------------
st.title("📈 Outlier & Distribution Analysis")
st.caption("Identifying unusual numerical values before deeper analysis")
 
# --------------------------------------------------------
# VARIABLES TO ANALYSE
# --------------------------------------------------------
ANALYSIS_VARIABLES = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
]
present_vars = [c for c in ANALYSIS_VARIABLES if c in df.columns]
missing_vars = [c for c in ANALYSIS_VARIABLES if c not in df.columns]
 
if missing_vars:
    st.info(f"Not found in this dataset (skipped): {', '.join(missing_vars)}")
 
if not present_vars:
    st.error("None of the expected analysis variables were found in the dataset.")
    st.stop()
 
with st.sidebar:
    st.header("IQR Settings")
    iqr_k = st.slider("IQR multiplier (k)", 0.5, 3.0, 1.5, 0.1,
                       help="Standard Tukey fence uses k = 1.5. Lower k flags more values as outliers.")
 
# --------------------------------------------------------
# IQR OUTLIER DETECTION
# --------------------------------------------------------
def iqr_bounds(series: pd.Series, k: float):
    clean = series.dropna()
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr
 
 
@st.cache_data
def scan_outliers(_df: pd.DataFrame, columns: list, k: float):
    rows = []
    masks = {}
    for col in columns:
        lo, hi = iqr_bounds(_df[col], k)
        mask = (_df[col] < lo) | (_df[col] > hi)
        mask = mask.fillna(False)
        masks[col] = mask
        n_valid = int(_df[col].notna().sum())
        rows.append({
            "Column": col,
            "Lower Bound": lo,
            "Upper Bound": hi,
            "Outlier Count": int(mask.sum()),
            "Outlier %": round(mask.sum() / n_valid * 100, 2) if n_valid else 0.0,
        })
    summary = pd.DataFrame(rows).sort_values("Outlier %", ascending=False).reset_index(drop=True)
    return summary, masks
 
 
outlier_summary, masks = scan_outliers(df, present_vars, iqr_k)
vars_with_outliers = int((outlier_summary["Outlier Count"] > 0).sum())
numeric_cols = df.select_dtypes(include=[np.number]).columns
 
# --------------------------------------------------------
# KPI CARDS
# --------------------------------------------------------
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("🔢 Numerical Columns", f"{len(numeric_cols)}")
kpi2.metric("⚠️ Variables with Outliers", f"{vars_with_outliers} / {len(present_vars)}")
kpi3.metric("💰 Maximum Income", f"{df['AMT_INCOME_TOTAL'].max():,.0f}" if "AMT_INCOME_TOTAL" in df else "N/A")
kpi4.metric("🏦 Maximum Credit", f"{df['AMT_CREDIT'].max():,.0f}" if "AMT_CREDIT" in df else "N/A")
kpi5.metric("🧾 Maximum Annuity", f"{df['AMT_ANNUITY'].max():,.0f}" if "AMT_ANNUITY" in df else "N/A")
 
st.markdown("---")
 
# --------------------------------------------------------
# ROW 1 — Income distribution | Income outliers
# --------------------------------------------------------
c1, c2 = st.columns(2)
 
with c1:
    st.subheader("💡 Income Distribution")
    if "AMT_INCOME_TOTAL" in df.columns:
        lo, hi = iqr_bounds(df["AMT_INCOME_TOTAL"], iqr_k)
        fig = px.histogram(df, x="AMT_INCOME_TOTAL", nbins=60, height=380)
        fig.add_vline(x=lo, line_dash="dash", line_color="#E63946", annotation_text="Lower fence")
        fig.add_vline(x=hi, line_dash="dash", line_color="#E63946", annotation_text="Upper fence")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("AMT_INCOME_TOTAL not found.")
 
with c2:
    st.subheader("📦 Income Outliers")
    if "AMT_INCOME_TOTAL" in df.columns:
        fig = px.box(df, y="AMT_INCOME_TOTAL", points="outliers", height=380)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("AMT_INCOME_TOTAL not found.")
 
# --------------------------------------------------------
# ROW 2 — Credit outliers | Annuity outliers
# --------------------------------------------------------
c3, c4 = st.columns(2)
 
with c3:
    st.subheader("📦 Credit Outliers")
    if "AMT_CREDIT" in df.columns:
        fig = px.box(df, y="AMT_CREDIT", points="outliers", height=380, color_discrete_sequence=["#2ca02c"])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("AMT_CREDIT not found.")
 
with c4:
    st.subheader("📦 Annuity Outliers")
    if "AMT_ANNUITY" in df.columns:
        fig = px.box(df, y="AMT_ANNUITY", points="outliers", height=380, color_discrete_sequence=["#ff7f0e"])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("AMT_ANNUITY not found.")
 
st.markdown("---")
 
# --------------------------------------------------------
# ROW 3 — Income vs Credit scatter
# --------------------------------------------------------
st.subheader("🔎 Income vs Credit")
if {"AMT_INCOME_TOTAL", "AMT_CREDIT"} <= set(df.columns):
    combined_mask = masks.get("AMT_INCOME_TOTAL", pd.Series(False, index=df.index)) | \
                    masks.get("AMT_CREDIT", pd.Series(False, index=df.index))
    scatter_df = df[["AMT_INCOME_TOTAL", "AMT_CREDIT"]].copy()
    scatter_df["Status"] = np.where(combined_mask, "Outlier", "Normal")
 
    fig = px.scatter(
        scatter_df, x="AMT_INCOME_TOTAL", y="AMT_CREDIT", color="Status",
        color_discrete_map={"Normal": "#2E86AB", "Outlier": "#E63946"},
        opacity=0.5, height=450,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Points flagged red fall outside the IQR fence on income and/or credit.")
else:
    st.info("AMT_INCOME_TOTAL and/or AMT_CREDIT not found.")
 
st.markdown("---")
 
# --------------------------------------------------------
# OUTLIER SUMMARY TABLE
# --------------------------------------------------------
st.subheader("📋 Outlier Summary Across Analysis Variables")
st.dataframe(
    outlier_summary.style.format({
        "Lower Bound": "{:,.1f}", "Upper Bound": "{:,.1f}", "Outlier %": "{:.2f}%"
    }).background_gradient(subset=["Outlier %"], cmap="Oranges"),
    use_container_width=True,
)
 
st.markdown("---")
 
# --------------------------------------------------------
# TECHNIQUES TO DISCUSS
# --------------------------------------------------------
st.subheader("🧭 Techniques to Discuss")
 
with st.expander("1. IQR Method"):
    st.markdown("""
    *   Flags a value as an outlier if it falls outside `[Q1 − k×IQR, Q3 + k×IQR]`, typically `k = 1.5`.
    *   Robust to skew and doesn't assume a normal distribution — a good default for right-skewed financial fields like income and credit.
    *   Best used for **screening/flagging**, not automatic removal.
    """)
 
with st.expander("2. Percentile Capping"):
    st.markdown("""
    *   Clips values above/below a chosen percentile (e.g. 1st/99th) to that percentile's value, instead of dropping the row.
    *   Preserves sample size — useful when every applicant needs to remain represented (e.g. for model training).
    *   Risk: an arbitrary percentile can still cap a genuinely high-income applicant.
    """)
 
with st.expander("3. Winsorization"):
    st.markdown("""
    *   Symmetric variant of percentile capping — trims both tails by the same proportion and replaces trimmed values with the nearest retained value.
    *   Common in finance to reduce the influence of extreme values on means/variances without discarding rows.
    """)
 
with st.expander("4. Log Transformation"):
    st.markdown("""
    *   Applies `log(1 + x)` to compress large values and pull in the right tail.
    *   Doesn't remove or reclassify outliers — it changes the *scale* so extreme values carry less leverage in models like linear/logistic regression.
    """)
 
with st.expander("5. Business-Rule Validation"):
    st.markdown("""
    *   `DAYS_EMPLOYED = 365243` is a known sentinel/error code, not a real outlier.
    *   `AMT_CREDIT` far below `AMT_GOODS_PRICE` may indicate a data entry issue in a loan-logic sense.
    *   `CNT_CHILDREN` exceeding `CNT_FAM_MEMBERS` is logically inconsistent.
    *   Statistical methods tell you *where* to look; business rules tell you *why* a value is there.
    """)
 
st.markdown("---")
 
# --------------------------------------------------------
# LIVE COMPARISON — technique effect on income
# --------------------------------------------------------
st.subheader("🔬 Live Comparison: Technique Effect on Income")
 
if "AMT_INCOME_TOTAL" in df.columns:
    technique = st.selectbox("Technique:", ["Percentile Capping", "Winsorization", "Log Transformation"])
    income = df["AMT_INCOME_TOTAL"].dropna()
 
    if technique == "Percentile Capping":
        lo_pct, hi_pct = st.slider("Percentile range to keep", 0.0, 100.0, (1.0, 99.0), 0.5)
        lo_val, hi_val = np.percentile(income, [lo_pct, hi_pct])
        transformed = income.clip(lower=lo_val, upper=hi_val)
        label = f"Capped [{lo_pct:.1f}–{hi_pct:.1f} pct]"
    elif technique == "Winsorization":
        limit = st.slider("Winsorize limit per tail (%)", 0.1, 10.0, 1.0, 0.1) / 100
        lo_val, hi_val = income.quantile(limit), income.quantile(1 - limit)
        transformed = income.clip(lower=lo_val, upper=hi_val)
        label = f"Winsorized ({limit * 100:.1f}% / {limit * 100:.1f}%)"
    else:
        transformed = np.log1p(income)
        label = "log1p(Income)"
 
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=income, name="Original", opacity=0.6, marker_color="#2E86AB", nbinsx=60))
    fig.add_trace(go.Histogram(x=transformed, name=label, opacity=0.6, marker_color="#E63946", nbinsx=60))
    fig.update_layout(barmode="overlay", height=400, xaxis_title="AMT_INCOME_TOTAL", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("AMT_INCOME_TOTAL not found — live comparison unavailable.")
 
st.markdown("---")
 
# --------------------------------------------------------
# CLASSIFICATION EXERCISE
# --------------------------------------------------------
st.subheader("🧩 Classify the Flagged Values")
st.warning(
    "**Do not automatically remove every outlier.** For each flagged row, decide: is this a "
    "**true extreme customer**, a **data entry issue**, or a **potential invalid value**?"
)
 
st.markdown("""
*   **True extreme customer** — plausible, just at the edge of the population (e.g. a genuinely high earner).
*   **Data entry issue** — looks like a system artifact (e.g. `DAYS_EMPLOYED = 365243`, a stray extra zero).
*   **Potential invalid value** — inconsistent with other fields for that row and needs follow-up (e.g. credit far exceeding any plausible income multiple, negative counts).
""")
 
review_col = st.selectbox("Column to review flagged rows for:", present_vars, key="review_col")
review_mask = masks[review_col]
flagged = df[review_mask].copy()
 
if flagged.empty:
    st.success(f"No IQR outliers flagged for **{review_col}** at k = {iqr_k}.")
else:
    id_col = "SK_ID_CURR" if "SK_ID_CURR" in df.columns else None
    context_cols = [c for c in [id_col, "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY",
                                 "DAYS_EMPLOYED", "CNT_CHILDREN", "CNT_FAM_MEMBERS"]
                     if c and c in df.columns]
    review_df = flagged[context_cols].head(200).reset_index(drop=True)
    review_df["Classification"] = "Not yet reviewed"
 
    edited = st.data_editor(
        review_df,
        column_config={
            "Classification": st.column_config.SelectboxColumn(
                "Classification",
                options=["Not yet reviewed", "True extreme customer",
                         "Data entry issue", "Potential invalid value"],
                required=True,
            )
        },
        use_container_width=True,
        height=400,
        key=f"editor_{review_col}",
    )
 
    counts = edited["Classification"].value_counts()
    st.caption("Classification tally: " + " · ".join(f"{k}: {v}" for k, v in counts.items()))
    st.caption(f"Showing up to 200 of {len(flagged):,} flagged rows for **{review_col}**.")
 
st.markdown("---")
 
# --------------------------------------------------------
# KEY INSIGHTS
# --------------------------------------------------------
st.subheader("📌 Key Insights")
 
insights = [
    f"**{vars_with_outliers} of {len(present_vars)} analysed variables** show at least one IQR-flagged outlier at k = {iqr_k}.",
    "**Income and credit fields are right-skewed** — a handful of genuinely high-value applicants will always sit outside a symmetric IQR fence; don't treat every flagged point as an error.",
    "**DAYS_EMPLOYED deserves special attention** — the 365243 sentinel value is a known data artifact from the source system, not a real employment duration.",
    "**Workflow:** use IQR/percentile methods to flag candidates, read the distributions and business context, then classify each — only data-entry or clearly invalid values are typically corrected or removed; true extreme customers are usually kept.",
]
for point in insights:
    st.markdown(f"- {point}")
 
st.markdown("---")
st.caption("Outlier & Distribution Analysis • application_train.csv • Home Credit Default Risk schema")
