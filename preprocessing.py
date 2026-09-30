
"""
preprocessing.py
-----------------
Data cleaning helpers for application_train.csv, meant to run BEFORE
utils.engineer_features() in the dashboard pipeline.
 
Typical usage in a dashboard file:
 
    from preprocessing import preprocess_pipeline
    from utils import load_data, engineer_features
 
    df = load_data(r"C:\\Users\\pavan\\Downloads\\dashboard\\application_train.csv")
    df = preprocess_pipeline(df)
    df = engineer_features(df)
 
Scope: this focuses on cleaning for DASHBOARD/VISUALIZATION use, not on
building an ML-ready encoded matrix — so it drops junk values, fixes known
data-entry anomalies, standardizes types/labels, and reports missingness,
rather than imputing/encoding everything for a model.
 
All functions are defensive: if a column doesn't exist in the given
dataframe, that step is skipped instead of raising an error.
"""
 
import pandas as pd
import numpy as np
 
 
# ------------------------------------------------------------------
# 1. DUPLICATES
# ------------------------------------------------------------------
def drop_duplicates(df: pd.DataFrame, subset: str = "SK_ID_CURR") -> pd.DataFrame:
    """Drop duplicate rows, preferring SK_ID_CURR (application id) if present."""
    if subset in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=subset, keep="first")
        removed = before - len(df)
        if removed:
            print(f"[preprocessing] Dropped {removed} duplicate rows on '{subset}'.")
    else:
        before = len(df)
        df = df.drop_duplicates(keep="first")
        removed = before - len(df)
        if removed:
            print(f"[preprocessing] Dropped {removed} fully duplicate rows.")
    return df
 
 
# ------------------------------------------------------------------
# 2. KNOWN DATA-ENTRY ANOMALIES (specific to this dataset)
# ------------------------------------------------------------------
def fix_known_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix documented placeholder/anomaly values in application_train:
      - DAYS_EMPLOYED == 365243  -> not currently employed placeholder -> NaN
      - CODE_GENDER == 'XNA'     -> unknown gender placeholder -> NaN
      - ORGANIZATION_TYPE == 'XNA' -> unknown org placeholder -> NaN
      - DAYS_* columns are stored negative (days before application);
        this does not alter them, just documents the convention.
    """
    if "DAYS_EMPLOYED" in df.columns:
        n_flagged = (df["DAYS_EMPLOYED"] == 365243).sum()
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
        if n_flagged:
            print(f"[preprocessing] Replaced {n_flagged} placeholder DAYS_EMPLOYED values (365243) with NaN.")
 
    if "CODE_GENDER" in df.columns:
        n_xna = (df["CODE_GENDER"] == "XNA").sum()
        df["CODE_GENDER"] = df["CODE_GENDER"].replace("XNA", np.nan)
        if n_xna:
            print(f"[preprocessing] Replaced {n_xna} 'XNA' CODE_GENDER values with NaN.")
 
    if "ORGANIZATION_TYPE" in df.columns:
        n_xna = (df["ORGANIZATION_TYPE"] == "XNA").sum()
        df["ORGANIZATION_TYPE"] = df["ORGANIZATION_TYPE"].replace("XNA", np.nan)
        if n_xna:
            print(f"[preprocessing] Replaced {n_xna} 'XNA' ORGANIZATION_TYPE values with NaN.")
 
    return df
 
 
# ------------------------------------------------------------------
# 3. MISSING VALUES
# ------------------------------------------------------------------
def missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a small summary table of missing values per column (non-zero only)."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    report = pd.DataFrame({
        "Missing Count": missing,
        "Missing %": (missing / len(df) * 100).round(2),
    })
    return report
 
 
def drop_high_missing_columns(df: pd.DataFrame, threshold: float = 60.0) -> pd.DataFrame:
    """Drop columns with more than `threshold`% missing values."""
    missing_pct = df.isnull().mean() * 100
    cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
    if cols_to_drop:
        print(f"[preprocessing] Dropping {len(cols_to_drop)} columns with >{threshold}% missing values.")
        df = df.drop(columns=cols_to_drop)
    return df
 
 
def fill_missing_values(
    df: pd.DataFrame,
    numeric_strategy: str = "median",
    categorical_fill: str = "Unknown",
) -> pd.DataFrame:
    """
    Fill remaining missing values so dashboard charts don't silently drop rows.
      - numeric_strategy: 'median', 'mean', or 'zero'
      - categorical_fill: label used for missing categorical values
    Note: for TARGET / SK_ID_CURR (identifiers/labels), missing values are left as-is.
    """
    protected_cols = {"TARGET", "SK_ID_CURR"}
 
    for col in df.columns:
        if col in protected_cols or df[col].isnull().sum() == 0:
            continue
 
        if pd.api.types.is_numeric_dtype(df[col]):
            if numeric_strategy == "median":
                df[col] = df[col].fillna(df[col].median())
            elif numeric_strategy == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif numeric_strategy == "zero":
                df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna(categorical_fill)
 
    return df
 
 
# ------------------------------------------------------------------
# 4. TYPE / LABEL STANDARDIZATION
# ------------------------------------------------------------------
def standardize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure key id/flag columns have sensible dtypes."""
    if "SK_ID_CURR" in df.columns:
        df["SK_ID_CURR"] = df["SK_ID_CURR"].astype("Int64")
 
    if "TARGET" in df.columns:
        df["TARGET"] = pd.to_numeric(df["TARGET"], errors="coerce").astype("Int64")
 
    flag_cols = [c for c in df.columns if c.startswith("FLAG_")]
    for c in flag_cols:
        if df[c].dropna().isin([0, 1]).all():
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
 
    return df
 
 
def standardize_categorical_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace and normalize casing inconsistencies in object columns."""
    obj_cols = df.select_dtypes(include="object").columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip()
        df.loc[df[c].isin(["nan", "None", ""]), c] = np.nan
    return df
 
 
# ------------------------------------------------------------------
# 5. OUTLIER HANDLING (light-touch, dashboard-friendly)
# ------------------------------------------------------------------
def winsorize_column(df: pd.DataFrame, col: str, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """Cap a numeric column's extreme values at given lower/upper quantiles."""
    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
        lo, hi = df[col].quantile([lower, upper])
        df[col] = df[col].clip(lower=lo, upper=hi)
    return df
 
 
def winsorize_amount_columns(df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """Apply winsorization to the main AMT_* monetary columns to tame extreme outliers."""
    amount_cols = [c for c in ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"] if c in df.columns]
    for c in amount_cols:
        df = winsorize_column(df, c, lower, upper)
    return df
 
 
# ------------------------------------------------------------------
# MASTER PIPELINE
# ------------------------------------------------------------------
def preprocess_pipeline(
    df: pd.DataFrame,
    drop_high_missing: bool = True,
    high_missing_threshold: float = 60.0,
    fill_missing: bool = True,
    winsorize_amounts: bool = False,
) -> pd.DataFrame:
    """
    Run the full cleaning pipeline in a sensible order:
      1. Drop duplicates
      2. Fix known anomalies (365243, XNA, etc.)
      3. Standardize dtypes and categorical labels
      4. Optionally drop very sparse columns
      5. Optionally fill remaining missing values
      6. Optionally winsorize monetary columns
 
    winsorize_amounts defaults to False because both dashboards already
    clip outliers at plot-time (via utils.clip_outliers) — turn this on
    only if you want the underlying data itself capped everywhere.
    """
    df = df.copy()
    df = drop_duplicates(df)
    df = fix_known_anomalies(df)
    df = standardize_dtypes(df)
    df = standardize_categorical_labels(df)
 
    if drop_high_missing:
        df = drop_high_missing_columns(df, threshold=high_missing_threshold)
 
    if fill_missing:
        df = fill_missing_values(df)
 
    if winsorize_amounts:
        df = winsorize_amount_columns(df)
 
    return df
 
