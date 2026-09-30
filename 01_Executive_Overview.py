
import streamlit as st
import pandas as pd
import numpy as np
 
from preprocessing import preprocess_pipeline
from utils import load_data, engineer_features
import charts
 
# ------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Executive Portfolio Overview",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ------------------------------------------------------------------
# DATA PIPELINE
# ------------------------------------------------------------------
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
 
if "TARGET" not in df.columns:
    st.error("This dashboard requires a TARGET column (0 = non-default, 1 = default).")
    st.stop()
 
# ------------------------------------------------------------------
# SIDEBAR FILTERS
# ------------------------------------------------------------------
st.sidebar.title("🔎 Filters")
st.sidebar.caption("Dataset: application_train.csv")
st.sidebar.caption(f"Rows loaded: {len(df):,}")
 
filtered_df = df.copy()
 
def multiselect_filter(col_name, label):
    global filtered_df
    if col_name in df.columns:
        options = sorted(df[col_name].dropna().unique().tolist())
        selected = st.sidebar.multiselect(label, options, default=[])
        if selected:
            filtered_df = filtered_df[filtered_df[col_name].isin(selected)]
 
multiselect_filter("NAME_CONTRACT_TYPE", "Contract Type")
multiselect_filter("CODE_GENDER", "Gender")
multiselect_filter("NAME_INCOME_TYPE", "Income Type")
multiselect_filter("NAME_EDUCATION_TYPE", "Education")
 
st.sidebar.markdown("---")
st.sidebar.caption(f"Filtered rows: {len(filtered_df):,}")
 
# ------------------------------------------------------------------
# HEADER
# ------------------------------------------------------------------
st.title("📊 Executive Portfolio Overview")
st.caption("A high-level overview of the Home Credit portfolio for management")
 
# ------------------------------------------------------------------
# KPI CARDS
# ------------------------------------------------------------------
total_customers = filtered_df["SK_ID_CURR"].nunique() if "SK_ID_CURR" in filtered_df.columns else len(filtered_df)
total_applications = len(filtered_df)
default_customers = int(filtered_df["TARGET"].sum())
non_default_customers = total_applications - default_customers
default_rate = (default_customers / total_applications * 100) if total_applications else np.nan
 
total_credit = filtered_df["AMT_CREDIT"].sum() if "AMT_CREDIT" in filtered_df.columns else np.nan
avg_credit = filtered_df["AMT_CREDIT"].mean() if "AMT_CREDIT" in filtered_df.columns else np.nan
median_credit = filtered_df["AMT_CREDIT"].median() if "AMT_CREDIT" in filtered_df.columns else np.nan
 
avg_income = filtered_df["AMT_INCOME_TOTAL"].mean() if "AMT_INCOME_TOTAL" in filtered_df.columns else np.nan
median_income = filtered_df["AMT_INCOME_TOTAL"].median() if "AMT_INCOME_TOTAL" in filtered_df.columns else np.nan
 
avg_annuity = filtered_df["AMT_ANNUITY"].mean() if "AMT_ANNUITY" in filtered_df.columns else np.nan
avg_goods_price = filtered_df["AMT_GOODS_PRICE"].mean() if "AMT_GOODS_PRICE" in filtered_df.columns else np.nan
 
row1 = st.columns(4)
row1[0].metric("Total Customers", f"{total_customers:,}")
row1[1].metric("Total Applications", f"{total_applications:,}")
row1[2].metric("Default Customers", f"{default_customers:,}")
row1[3].metric("Non-Default Customers", f"{non_default_customers:,}")
 
row2 = st.columns(4)
row2[0].metric("Default Rate", f"{default_rate:.2f}%" if pd.notna(default_rate) else "N/A")
row2[1].metric("Total Credit Amount", f"${total_credit:,.0f}" if pd.notna(total_credit) else "N/A")
row2[2].metric("Average Credit Amount", f"${avg_credit:,.0f}" if pd.notna(avg_credit) else "N/A")
row2[3].metric("Median Credit Amount", f"${median_credit:,.0f}" if pd.notna(median_credit) else "N/A")
 
row3 = st.columns(4)
row3[0].metric("Average Customer Income", f"${avg_income:,.0f}" if pd.notna(avg_income) else "N/A")
row3[1].metric("Median Income", f"${median_income:,.0f}" if pd.notna(median_income) else "N/A")
row3[2].metric("Average Annuity", f"${avg_annuity:,.0f}" if pd.notna(avg_annuity) else "N/A")
row3[3].metric("Average Goods Price", f"${avg_goods_price:,.0f}" if pd.notna(avg_goods_price) else "N/A")
 
st.markdown("---")
 
# ------------------------------------------------------------------
# ROW 1 — Default vs Non-Default (bar) | Default Percentage (donut)
# ------------------------------------------------------------------
c1, c2 = st.columns(2)
with c1:
    st.subheader("Default vs Non-Default")
    st.plotly_chart(charts.outcome_count_bar(filtered_df), use_container_width=True)
 
with c2:
    st.subheader("Default Percentage")
    st.plotly_chart(charts.outcome_pie_chart(filtered_df), use_container_width=True)
 
# ------------------------------------------------------------------
# ROW 2 — Applications by Contract Type (bar)
# ------------------------------------------------------------------
st.subheader("Applications by Contract Type")
if "NAME_CONTRACT_TYPE" in filtered_df.columns:
    st.plotly_chart(charts.category_count_bar(filtered_df, "NAME_CONTRACT_TYPE"), use_container_width=True)
else:
    st.info("NAME_CONTRACT_TYPE column not found.")
 
st.markdown("---")
 
# ------------------------------------------------------------------
# ROW 3 — Credit Amount Distribution | Income Distribution (histograms)
# ------------------------------------------------------------------
c3, c4 = st.columns(2)
with c3:
    st.subheader("Credit Amount Distribution")
    if "AMT_CREDIT" in filtered_df.columns:
        st.plotly_chart(
            charts.distribution_histogram(filtered_df, "AMT_CREDIT", clip_quantile=0.99),
            use_container_width=True,
        )
    else:
        st.info("AMT_CREDIT column not found.")
 
with c4:
    st.subheader("Income Distribution")
    if "AMT_INCOME_TOTAL" in filtered_df.columns:
        st.plotly_chart(
            charts.distribution_histogram(filtered_df, "AMT_INCOME_TOTAL", clip_quantile=0.99),
            use_container_width=True,
        )
    else:
        st.info("AMT_INCOME_TOTAL column not found.")
 
st.markdown("---")
 
# ------------------------------------------------------------------
# ROW 4 — Credit by Income Type (treemap) | Default Rate by Income Type (h-bar)
# ------------------------------------------------------------------
c5, c6 = st.columns(2)
with c5:
    st.subheader("Credit by Income Type")
    if {"NAME_INCOME_TYPE", "AMT_CREDIT"}.issubset(filtered_df.columns):
        st.plotly_chart(
            charts.treemap_by_category(filtered_df, "NAME_INCOME_TYPE", "AMT_CREDIT", agg="sum"),
            use_container_width=True,
        )
    else:
        st.info("NAME_INCOME_TYPE or AMT_CREDIT column not found.")
 
with c6:
    st.subheader("Default Rate by Income Type")
    if "NAME_INCOME_TYPE" in filtered_df.columns:
        st.plotly_chart(
            charts.default_rate_by_segment(filtered_df, "NAME_INCOME_TYPE", min_segment_size=30),
            use_container_width=True,
        )
    else:
        st.info("NAME_INCOME_TYPE column not found.")
 
st.markdown("---")
 
# ------------------------------------------------------------------
# ROW 5 — Income vs Credit (scatter)
# ------------------------------------------------------------------
st.subheader("Income vs Credit")
if {"AMT_INCOME_TOTAL", "AMT_CREDIT"}.issubset(filtered_df.columns):
    scatter_df = filtered_df.copy()
    income_cap = scatter_df["AMT_INCOME_TOTAL"].quantile(0.99)
    scatter_df = scatter_df[scatter_df["AMT_INCOME_TOTAL"] <= income_cap]
    st.plotly_chart(
        charts.scatter_by_outcome(scatter_df, "AMT_INCOME_TOTAL", "AMT_CREDIT"),
        use_container_width=True,
    )
    st.caption("Top 1% of incomes excluded for readability.")
else:
    st.info("AMT_INCOME_TOTAL or AMT_CREDIT column not found.")
 
st.markdown("---")
 
# ------------------------------------------------------------------
# REQUIRED INSIGHTS (auto-generated from the current filtered view)
# ------------------------------------------------------------------
st.subheader("📌 Key Insights")
 
insights = []
 
insights.append(f"**Overall default rate:** {default_rate:.2f}% ({default_customers:,} of {total_applications:,} applications).")
 
if pd.notna(total_credit):
    insights.append(f"**Total credit exposure:** ${total_credit:,.0f} across the current view.")
 
if "NAME_INCOME_TYPE" in filtered_df.columns:
    largest_segment = filtered_df["NAME_INCOME_TYPE"].value_counts().idxmax()
    largest_segment_pct = filtered_df["NAME_INCOME_TYPE"].value_counts(normalize=True).max() * 100
    insights.append(f"**Largest customer segment:** {largest_segment} ({largest_segment_pct:.1f}% of applications).")
 
if "NAME_INCOME_TYPE" in filtered_df.columns:
    income_risk = (
        filtered_df.groupby("NAME_INCOME_TYPE")["TARGET"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "rate", "count": "n"})
    )
    income_risk = income_risk[income_risk["n"] >= 30]
    if not income_risk.empty:
        highest_risk_income = income_risk["rate"].idxmax()
        highest_risk_rate = income_risk["rate"].max() * 100
        insights.append(f"**Highest-risk income segment:** {highest_risk_income} (default rate {highest_risk_rate:.2f}%).")
 
if pd.notna(median_credit):
    insights.append(f"**Typical credit amount:** ${median_credit:,.0f} (median); mean is ${avg_credit:,.0f}.")
 
if pd.notna(median_income):
    insights.append(f"**Typical customer income:** ${median_income:,.0f} (median); mean is ${avg_income:,.0f}.")
 
for point in insights:
    st.markdown(f"- {point}")
 
# ------------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------------
st.markdown("---")
st.caption("Executive Portfolio Overview • application_train.csv • Home Credit Default Risk schema")
 
