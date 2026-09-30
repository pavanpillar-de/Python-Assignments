
import pandas as pd
import numpy as np
 
 
 
# ------------------------------------------------------------------
# INDIVIDUAL FEATURE BLOCKS
# ------------------------------------------------------------------
def add_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert DAYS_BIRTH (negative days) into a readable AGE_YEARS column."""
    if "DAYS_BIRTH" in df.columns:
        df["AGE_YEARS"] = (-df["DAYS_BIRTH"] / 365.25).round(0)
    return df
 
 
def add_employment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert DAYS_EMPLOYED into YEARS_EMPLOYED.
    Home Credit uses 365243 as a placeholder for 'not currently employed'
    (e.g. pensioners) — this is replaced with NaN before conversion.
    """
    if "DAYS_EMPLOYED" in df.columns:
        df["DAYS_EMPLOYED_CLEAN"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
        df["YEARS_EMPLOYED"] = (-df["DAYS_EMPLOYED_CLEAN"] / 365.25).round(1)
    return df
 
 
def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add common credit-risk ratio features used across the dashboards."""
    if {"AMT_CREDIT", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["CREDIT_INCOME_RATIO"] = (df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]).round(2)
 
    if {"AMT_ANNUITY", "AMT_INCOME_TOTAL"}.issubset(df.columns):
        df["ANNUITY_INCOME_RATIO"] = (df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]).round(2)
 
    if {"AMT_ANNUITY", "AMT_CREDIT"}.issubset(df.columns):
        # Rough proxy for loan duration in "annuity payments"
        df["CREDIT_TERM"] = (df["AMT_CREDIT"] / df["AMT_ANNUITY"]).round(1)
 
    if {"AMT_GOODS_PRICE", "AMT_CREDIT"}.issubset(df.columns):
        # How much credit exceeds the price of the goods being financed
        df["CREDIT_GOODS_RATIO"] = (df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]).round(2)
 
    return df
 
 
def add_outcome_label(df: pd.DataFrame) -> pd.DataFrame:
    """Add a human-readable OUTCOME column from TARGET (0/1)."""
    if "TARGET" in df.columns:
        df["OUTCOME"] = df["TARGET"].map({0: "Repaid", 1: "Default"})
    return df
 
 
def clip_outliers(series: pd.Series, upper_quantile: float = 0.99) -> pd.Series:
    """Cap extreme values at a given quantile — useful before plotting."""
    return series.clip(upper=series.quantile(upper_quantile))
 
 
# ------------------------------------------------------------------
# MASTER PIPELINE
# ------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run all feature engineering steps in sequence and return the enriched df."""
    df = df.copy()
    df = add_age_features(df)
    df = add_employment_features(df)
    df = add_ratio_features(df)
    df = add_outcome_label(df)
    return df
 
 
# ------------------------------------------------------------------
# HELPERS FOR DASHBOARD LOGIC
# ------------------------------------------------------------------
def get_numeric_risk_candidates(df: pd.DataFrame) -> list:
    """Return the list of engineered/raw numeric columns worth checking against TARGET."""
    candidates = [
        "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "AGE_YEARS", "YEARS_EMPLOYED",
        "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
        "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_TERM", "CREDIT_GOODS_RATIO",
        "CNT_CHILDREN", "CNT_FAM_MEMBERS", "REGION_RATING_CLIENT",
        "REGION_RATING_CLIENT_W_CITY", "DAYS_LAST_PHONE_CHANGE",
        "OBS_30_CNT_SOCIAL_CIRCLE", "DEF_30_CNT_SOCIAL_CIRCLE",
    ]
    return [c for c in candidates if c in df.columns]
 
 
def get_categorical_segment_options(df: pd.DataFrame) -> list:
    """Return categorical columns commonly used for segmenting default rate."""
    candidates = [
        "NAME_EDUCATION_TYPE", "CODE_GENDER", "NAME_FAMILY_STATUS",
        "NAME_INCOME_TYPE", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE",
        "NAME_CONTRACT_TYPE", "ORGANIZATION_TYPE", "WEEKDAY_APPR_PROCESS_START",
    ]
    return [c for c in candidates if c in df.columns]