
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
 
# --------------------------------------------------------
# 1. PAGE CONFIGURATION & CACHED DATA LOADING
# --------------------------------------------------------
st.set_page_config(page_title="Home Credit Data Quality Dashboard", layout="wide")
 
@st.cache_data
def load_and_profile_data():
    # Replace this with your actual local file path or URL
    # For testing, we generate a representative sample if file not found
    try:
        df = pd.read_csv("application_train.csv")
    except FileNotFoundError:
        # Create a mock dataset mimicking application_train properties for instant rendering
        np.random.seed(42)
        n_rows = 10000
        mock_data = {
            'SK_ID_CURR': np.arange(100002, 100002 + n_rows),
            'TARGET': np.random.choice([0, 1], size=n_rows, p=[0.92, 0.08]),
            'NAME_CONTRACT_TYPE': np.random.choice(['Cash loans', 'Revolving loans'], size=n_rows),
            'CODE_GENDER': np.random.choice(['M', 'F', 'XNA'], size=n_rows, p=[0.35, 0.64, 0.01]),
            'AMT_INCOME_TOTAL': np.random.exponential(scale=150000, size=n_rows) + 25000,
            'DAYS_EMPLOYED': np.random.choice([-1000, -2500, 365243, -500], size=n_rows, p=[0.4, 0.4, 0.15, 0.05]),
            'OWN_CAR_AGE': np.random.choice([np.nan, 2.0, 5.0, 12.0, 20.0], size=n_rows, p=[0.65, 0.1, 0.1, 0.1, 0.05]),
            'EXT_SOURCE_1': np.random.uniform(0, 1, size=n_rows),
            'ORGANIZATION_TYPE': np.random.choice(['Business Entity Type 3', 'XNA', 'Self-employed'], size=n_rows)
        }
        # Add random missing values across columns
        df = pd.DataFrame(mock_data)
        for col in df.columns:
            if col not in ['SK_ID_CURR', 'TARGET']:
                df.loc[df.sample(frac=np.random.uniform(0, 0.05)).index, col] = np.nan
        df.loc[df.sample(n=5).index, :] = df.iloc[0].values  # Add a few duplicates for testing
 
    # Structural calculations
    total_rows, total_cols = df.shape
    num_cols = df.select_dtypes(include=[np.number]).shape[1]
    cat_cols = df.select_dtypes(include=['object', 'category']).shape[1]
    missing_cells = df.isnull().sum().sum()
    total_cells = total_rows * total_cols
    completeness = ((total_cells - missing_cells) / total_cells) * 100
    duplicate_rows = df.duplicated().sum()
    memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    unique_customers = df['SK_ID_CURR'].nunique()
 
    # Generate Profile Table
    profile_rows = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        m_count = df[col].isnull().sum()
        m_pct = (m_count / total_rows) * 100
        u_val = df[col].nunique()
 
        if pd.api.types.is_numeric_dtype(df[col]):
            vmin = df[col].min()
            vmax = df[col].max()
            vmean = df[col].mean()
            vmedian = df[col].median()
        else:
            vmin, vmax, vmean, vmedian = "N/A", "N/A", "N/A", "N/A"
 
        profile_rows.append({
            "Column Name": col, "Data Type": dtype, "Missing Count": m_count,
            "Missing %": round(m_pct, 2), "Unique Values": u_val,
            "Minimum": vmin, "Maximum": vmax, "Mean": vmean, "Median": vmedian
        })
    profile_df = pd.DataFrame(profile_rows)
 
    return df, profile_df, total_rows, total_cols, num_cols, cat_cols, missing_cells, duplicate_rows, memory_usage_mb, unique_customers, completeness
 
df, profile_df, rows, cols, num_cols, cat_cols, missing, duplicates, memory, unique_cust, completeness = load_and_profile_data()
 
# --------------------------------------------------------
# 2. HEADER SECTION
# --------------------------------------------------------
st.title("📊 Home Credit Data Quality Dashboard")
st.subheader("Dataset Profiling & Preprocessing Assessment Component")
st.markdown("---")
 
# --------------------------------------------------------
# 3. KPI CARDS LAYER
# --------------------------------------------------------
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("📋 Total Rows", f"{rows:,}")
kpi2.metric("📐 Total Columns", f"{cols}")
kpi3.metric("🔢 Numerical Columns", f"{num_cols}")
kpi4.metric("🔤 Categorical Columns", f"{cat_cols}")
 
kpi5, kpi6, kpi7, kpi8 = st.columns(4)
kpi5.metric("🔍 Missing Cells", f"{missing:,}")
kpi6.metric("👥 Duplicate Rows", f"{duplicates}")
kpi7.metric("💾 Total Memory Usage", f"{memory:.2f} MB")
kpi8.metric("🆔 Unique Customers (SK_ID)", f"{unique_cust:,}")
 
st.markdown("---")
 
# --------------------------------------------------------
# 4. VISUALIZATIONS SECTION
# --------------------------------------------------------
col1, col2 = st.columns(2)
 
with col1:
    st.subheader("💡 Column Data Types Distribution")
    type_counts = profile_df['Data Type'].value_counts().reset_index()
    type_counts.columns = ['Data Type', 'Count']
    fig_types = px.bar(type_counts, x='Count', y='Data Type', orientation='h',
                        color='Data Type', height=300)
    st.plotly_chart(fig_types, use_container_width=True)
 
with col2:
    st.subheader("📈 Dataset Completeness Gauge")
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=completeness,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "#1f77b4"},
               'steps': [
                   {'range': [0, 60], 'color': "#ff9999"},
                   {'range': [60, 85], 'color': "#ffcc99"},
                   {'range': [85, 100], 'color': "#b3ffb3"}]}))
    fig_gauge.update_layout(height=300, margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig_gauge, use_container_width=True)
 
col3, col4 = st.columns(2)
 
with col3:
    st.subheader("⚠️ Missing vs Available Data per Feature")
    missing_chart_df = profile_df.sort_values(by='Missing %', ascending=False).head(15)
    fig_missing = go.Figure()
    fig_missing.add_trace(go.Bar(y=missing_chart_df['Column Name'], x=100 - missing_chart_df['Missing %'],
                                  name='Available %', orientation='h', marker_color='#2ca02c'))
    fig_missing.add_trace(go.Bar(y=missing_chart_df['Column Name'], x=missing_chart_df['Missing %'],
                                  name='Missing %', orientation='h', marker_color='#d62728'))
    fig_missing.update_layout(barmode='stack', height=400, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_missing, use_container_width=True)
 
with col4:
    st.subheader("🔎 Cardinality: Unique Values by Categorical Column")
    cat_profile = profile_df[profile_df['Column Name'].isin(df.select_dtypes(include=['object', 'category']).columns)]
    if not cat_profile.empty:
        fig_cardinality = px.bar(cat_profile.sort_values(by='Unique Values', ascending=True),
                                  x='Unique Values', y='Column Name', orientation='h', text_auto=True, height=400)
        st.plotly_chart(fig_cardinality, use_container_width=True)
    else:
        st.info("No categorical columns available to plot cardinality.")
 
st.markdown("---")
 
# --------------------------------------------------------
# 5. DATA PROFILING TABLE
# --------------------------------------------------------
st.subheader("📋 Comprehensive Data Profiling Table")
st.dataframe(profile_df, use_container_width=True)
st.markdown("---")
 
# --------------------------------------------------------
# 6. REQUIRED ANALYSIS EXPANDER PANELS
# --------------------------------------------------------
st.subheader("🗒️ Required Quality Analysis & Assessment")
 
with st.expander("1. Which columns have quality issues?"):
    st.markdown("""
    *   **Anomalous Time Fields:** Columns measuring relative time metrics like `DAYS_EMPLOYED` contain an error code value of **365243** (which equals 1,000 years). This must be systematically flagged and handled.
    *   **Negative Durations:** Feature categories indexing chronological metrics (`DAYS_BIRTH`, `DAYS_REGISTRATION`, `DAYS_ID_PUBLISH`) are stored as negative numbers relative to the application day.
    """)
 
with st.expander("2. Which columns contain extreme missingness?"):
    high_missing = profile_df[profile_df['Missing %'] > 40][['Column Name', 'Missing %']]
    if not high_missing.empty:
        st.dataframe(high_missing.reset_index(drop=True), use_container_width=True)
    st.markdown("""
    *   **External Features & Assets:** Housing parameters (`APARTMENTS_AVG`, `BASEMENTAREA_AVG`) and structural assets like `OWN_CAR_AGE` consistently cross 40-65% missing thresholds.
    *   **Context:** Missing fields on asset metrics often indicate the structural absence of the asset (e.g., missing `OWN_CAR_AGE` implies the borrower does not own a vehicle).
    """)
 
with st.expander("3. Which columns may require datatype conversion?"):
    st.markdown("""
    *   **Categorical Encodings:** Columns like `FLAG_OWN_CAR` and `FLAG_OWN_REALTY` store binary states as strings (`Y`/`N`). These must be mapped to flags (`1`/`0`).
    *   **Arbitrary Classifications:** High volumes of integer classifications flagged under generic labels (like `FLAG_DOCUMENT_X` features) operate intrinsically as booleans rather than scalar integers.
    """)
 
with st.expander("4. Are duplicate customers present?"):
    if rows == unique_cust:
        st.success(f"✅ Clean Structure: Total Rows ({rows:,}) exactly matches Unique Customer Counts ({unique_cust:,}). There are zero cross-sectional duplicate customer anomalies.")
    else:
        st.warning(f"⚠️ Discrepancy Found: Total Rows ({rows:,}) does not match Unique Customer IDs ({unique_cust:,}). Cross-sectional duplicates or historical sequence states exist.")
 
with st.expander("5. Are there categorical inconsistencies?"):
    st.markdown("""
    *   **Missing System Code Labels:** Fields like `CODE_GENDER` contain unknown code states (e.g., `XNA`), which function effectively as structured missing string tags.
    *   **Free-Text Drift:** Columns like `ORGANIZATION_TYPE` mix broad categories with highly specific ones (e.g., `Business Entity Type 3`), which can inflate cardinality and complicate grouping.
    """)
 
st.markdown("---")
st.caption("Data Quality Dashboard • application_train.csv • Home Credit Default Risk schema")
 
