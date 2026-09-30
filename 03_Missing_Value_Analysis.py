import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
 
from utils import load_data
from preprocessing import missing_value_report
 
# --------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------
st.set_page_config(
    page_title="Missing Value Analysis",
    page_icon="🕳️",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# --------------------------------------------------------
# DATA LOADING (raw — NOT run through preprocess_pipeline,
# since this page exists to analyze missingness before cleaning)
# --------------------------------------------------------
DATA_PATH = r"C:\Users\pavan\Downloads\dashboard\application_train.csv"
 
@st.cache_data
def get_raw_data(path: str) -> pd.DataFrame:
    return load_data(path)
 
try:
    df = get_raw_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"Could not find file at: {DATA_PATH}\n\nUpdate DATA_PATH at the top of this file.")
    st.stop()
 
has_target = "TARGET" in df.columns
 
# --------------------------------------------------------
# HEADER
# --------------------------------------------------------
st.title("🕳️ Missing Value Analysis")
st.caption("Where the gaps are in application_train.csv, and whether they matter")
 
# --------------------------------------------------------
# CORE MISSINGNESS CALCULATIONS
# --------------------------------------------------------
total_rows, total_cols = df.shape
total_cells = total_rows * total_cols
missing_per_col = df.isnull().sum()
missing_pct_per_col = (missing_per_col / total_rows * 100).round(2)
 
cols_with_missing = (missing_per_col > 0).sum()
cols_no_missing = total_cols - cols_with_missing
cols_high_missing = (missing_pct_per_col > 50).sum()
cols_complete_missing = (missing_pct_per_col == 100).sum()
 
total_missing_cells = int(missing_per_col.sum())
overall_missing_pct = (total_missing_cells / total_cells * 100)
 
missing_per_row = df.isnull().sum(axis=1)
rows_with_any_missing = (missing_per_row > 0).sum()
rows_with_any_missing_pct = rows_with_any_missing / total_rows * 100
 
worst_col = missing_pct_per_col.idxmax() if cols_with_missing > 0 else "None"
worst_col_pct = missing_pct_per_col.max() if cols_with_missing > 0 else 0
 
# --------------------------------------------------------
# KPI CARDS
# --------------------------------------------------------
row1 = st.columns(4)
row1[0].metric("Total Cells", f"{total_cells:,}")
row1[1].metric("Missing Cells", f"{total_missing_cells:,}")
row1[2].metric("Overall Missing %", f"{overall_missing_pct:.2f}%")
row1[3].metric("Columns w/ Missing Data", f"{cols_with_missing} / {total_cols}")
 
row2 = st.columns(4)
row2[0].metric("Columns >50% Missing", f"{cols_high_missing}")
row2[1].metric("Columns 100% Missing", f"{cols_complete_missing}")
row2[2].metric("Rows w/ ≥1 Missing Value", f"{rows_with_any_missing_pct:.1f}%")
row2[3].metric("Worst Column", worst_col, delta=f"{worst_col_pct:.1f}% missing", delta_color="off")
 
st.markdown("---")
 
# --------------------------------------------------------
# ROW 1 — Missing % by column (ranked bar)
# --------------------------------------------------------
st.subheader("Missing % by Column")
 
if cols_with_missing > 0:
    top_n = st.slider("Show top N columns by missing %", 5, min(60, cols_with_missing), min(20, cols_with_missing))
 
    missing_df = missing_pct_per_col[missing_pct_per_col > 0].sort_values(ascending=False).head(top_n)
    plot_df = missing_df.reset_index()
    plot_df.columns = ["Column", "Missing %"]
 
    fig = px.bar(
        plot_df, x="Missing %", y="Column", orientation="h",
        color="Missing %", color_continuous_scale="Reds",
        text="Missing %",
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        height=max(400, 22 * len(plot_df)),
        coloraxis_showscale=False,
    )
    # reference lines for common missingness thresholds
    fig.add_vline(x=50, line_dash="dash", line_color="gray", annotation_text="50%")
    fig.add_vline(x=80, line_dash="dash", line_color="black", annotation_text="80%")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.success("No missing values found in this dataset.")
 
st.markdown("---")
 
# --------------------------------------------------------
# ROW 2 — Missing values by data type | Missing values per row distribution
# --------------------------------------------------------
c1, c2 = st.columns(2)
 
with c1:
    st.subheader("Missing Cells by Data Type")
    dtype_map = df.dtypes.astype(str)
    is_numeric = df.dtypes.apply(lambda d: pd.api.types.is_numeric_dtype(d))
    dtype_group = pd.Series(np.where(is_numeric, "Numeric", "Categorical / Text"), index=df.columns)
 
    dtype_missing = pd.DataFrame({
        "Missing Cells": missing_per_col,
        "Type": dtype_group,
    })
    dtype_summary = dtype_missing.groupby("Type")["Missing Cells"].sum().reset_index()
 
    if dtype_summary["Missing Cells"].sum() > 0:
        fig = px.pie(dtype_summary, names="Type", values="Missing Cells", hole=0.5)
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No missing cells to break down by type.")
 
with c2:
    st.subheader("Missing Fields per Applicant")
    row_missing_df = missing_per_row.value_counts().reset_index()
    row_missing_df.columns = ["Missing Fields", "Applicants"]
    row_missing_df = row_missing_df.sort_values("Missing Fields")
 
    fig = px.bar(row_missing_df, x="Missing Fields", y="Applicants")
    fig.update_layout(xaxis_title="Number of Missing Fields (per applicant)", yaxis_title="Applicants")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"{rows_with_any_missing:,} of {total_rows:,} applicants ({rows_with_any_missing_pct:.1f}%) have at least one missing field.")
 
st.markdown("---")
 
# --------------------------------------------------------
# ROW 3 — Missingness heatmap (sampled)
# --------------------------------------------------------
st.subheader("Missing Value Heatmap (sampled)")
 
if cols_with_missing > 0:
    heatmap_col_n = st.slider("Columns to include (ranked by missing %)", 5, min(40, cols_with_missing), min(20, cols_with_missing), key="heatmap_cols")
    heatmap_row_n = st.slider("Rows to sample", 50, 500, 200, step=50, key="heatmap_rows")
 
    top_missing_cols = missing_pct_per_col[missing_pct_per_col > 0].sort_values(ascending=False).head(heatmap_col_n).index.tolist()
    sample_df = df[top_missing_cols].sample(min(heatmap_row_n, len(df)), random_state=42)
    missing_matrix = sample_df.isnull().astype(int)
 
    fig = px.imshow(
        missing_matrix.T,
        color_continuous_scale=["#2E86AB", "#E63946"],
        aspect="auto",
        labels=dict(x="Sampled Applicant", y="Column", color="Missing"),
    )
    fig.update_layout(height=max(400, 20 * len(top_missing_cols)), coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Red = missing, Blue = present. Sampled for rendering performance — not the full dataset.")
else:
    st.info("No missing values to visualize.")
 
st.markdown("---")
 
# --------------------------------------------------------
# ROW 4 — Nullity correlation (which columns go missing together)
# --------------------------------------------------------
st.subheader("Nullity Correlation — Columns That Go Missing Together")
 
if cols_with_missing >= 2:
    corr_col_n = st.slider("Columns to include", 2, min(30, cols_with_missing), min(15, cols_with_missing), key="corr_cols")
    top_corr_cols = missing_pct_per_col[missing_pct_per_col > 0].sort_values(ascending=False).head(corr_col_n).index.tolist()
 
    nullity_matrix = df[top_corr_cols].isnull().astype(int)
    nullity_corr = nullity_matrix.corr()
 
    fig = px.imshow(
        nullity_corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        aspect="auto", labels=dict(color="Correlation"),
    )
    fig.update_layout(height=max(400, 25 * len(top_corr_cols)))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Values near +1 mean two columns tend to be missing together (often the same source form/section). Near -1 means rarely missing together.")
else:
    st.info("Need at least 2 columns with missing values for nullity correlation.")
 
st.markdown("---")
 
# --------------------------------------------------------
# ROW 5 — Does missingness relate to default risk?
# --------------------------------------------------------
if has_target:
    st.subheader("Missingness vs Default Rate")
 
    candidate_cols = missing_pct_per_col[(missing_pct_per_col > 0) & (missing_pct_per_col < 100)].sort_values(ascending=False).head(15).index.tolist()
 
    if candidate_cols:
        col_choice = st.selectbox("Column:", candidate_cols)
        temp = df[[col_choice, "TARGET"]].copy()
        temp["Status"] = np.where(temp[col_choice].isnull(), "Missing", "Present")
 
        rate_df = temp.groupby("Status")["TARGET"].agg(["mean", "count"]).reset_index()
        rate_df.columns = ["Status", "Default Rate", "Applications"]
        rate_df["Default Rate"] = (rate_df["Default Rate"] * 100).round(2)
 
        fig = px.bar(
            rate_df, x="Status", y="Default Rate", color="Status", text="Default Rate",
            color_discrete_map={"Missing": "#E63946", "Present": "#2E86AB"},
            hover_data=["Applications"],
        )
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(showlegend=False, yaxis_title="Default Rate (%)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("If 'Missing' and 'Present' default rates differ noticeably, the missingness itself may carry signal — consider a missing-indicator flag rather than pure imputation.")
    else:
        st.info("No partially-missing columns available to compare.")
else:
    st.info("TARGET column not found — skipping missingness-vs-default comparison.")
 
st.markdown("---")
 
# --------------------------------------------------------
# ROW 6 — Full missing value report table
# --------------------------------------------------------
st.subheader("Full Missing Value Report")
 
report = missing_value_report(df)
if not report.empty:
    report_display = report.reset_index().rename(columns={"index": "Column"})
    st.dataframe(report_display, use_container_width=True, hide_index=True)
 
    csv = report_display.to_csv(index=False).encode("utf-8")
    st.download_button("Download report as CSV", data=csv, file_name="missing_value_report.csv", mime="text/csv")
else:
    st.success("No missing values found in this dataset.")
 
st.markdown("---")
 
# --------------------------------------------------------
# KEY INSIGHTS
# --------------------------------------------------------
st.subheader("📌 Key Insights")
 
insights = []
insights.append(f"**Overall completeness:** {100 - overall_missing_pct:.2f}% of all cells are populated ({total_missing_cells:,} missing out of {total_cells:,}).")
insights.append(f"**Columns affected:** {cols_with_missing} of {total_cols} columns have at least one missing value; {cols_no_missing} are fully complete.")
 
if cols_complete_missing > 0:
    fully_missing_cols = missing_pct_per_col[missing_pct_per_col == 100].index.tolist()
    insights.append(f"**Completely empty columns:** {', '.join(fully_missing_cols[:10])}{' …' if len(fully_missing_cols) > 10 else ''} — consider dropping these entirely.")
 
if cols_high_missing > 0:
    insights.append(f"**High-missingness columns (>50%):** {cols_high_missing} columns — mostly likely housing/apartment detail and asset fields; missingness here often reflects the asset simply not existing (e.g. no car → `OWN_CAR_AGE` blank) rather than a data error.")
 
insights.append(f"**Row-level impact:** {rows_with_any_missing_pct:.1f}% of applicants have at least one missing field, so a naive drop-all-missing-rows approach would discard a large share of the portfolio.")
 
for point in insights:
    st.markdown(f"- {point}")
 
st.markdown("---")
st.caption("Missing Value Analysis • application_train.csv • Home Credit Default Risk schema")
 
