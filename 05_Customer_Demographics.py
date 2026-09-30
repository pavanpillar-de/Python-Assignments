import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
 
st.set_page_config(page_title="Customer Demographics", page_icon="🧑‍🤝‍🧑", layout="wide")
 
DATA_PATH = r"C:\Users\pavan\Downloads\dashboard\application_train.csv"
 

# ---------------------------------------------------------------------------
# Age groups
# ---------------------------------------------------------------------------
AGE_BINS = [20, 30, 40, 50, 60, 200]
AGE_LABELS = ["20-30", "31-40", "41-50", "51-60", "60+"]


def add_age_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Bucket AGE into the standard 20-30 / 31-40 / 41-50 / 51-60 / 60+ groups."""
    df = df.copy()
    if "AGE" not in df.columns:
        return df
    df["AGE_GROUP"] = pd.cut(
        df["AGE"].astype(float),
        bins=AGE_BINS,
        labels=AGE_LABELS,
        right=True,
        include_lowest=True,
    )
    return df


# ---------------------------------------------------------------------------
# Small stat helpers (kept local so this file has no other project deps)
# ---------------------------------------------------------------------------
def _most_common(series: pd.Series, default: str = "N/A") -> str:
    series = series.dropna()
    return str(series.mode().iloc[0]) if not series.empty else default


def _safe_mean(series: pd.Series, default: float = 0.0) -> float:
    series = pd.to_numeric(series, errors="coerce").dropna()
    return float(series.mean()) if not series.empty else default


def _safe_median(series: pd.Series, default: float = 0.0) -> float:
    series = pd.to_numeric(series, errors="coerce").dropna()
    return float(series.median()) if not series.empty else default


# ---------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------
def compute_kpis(df: pd.DataFrame) -> dict:
    """
    Returns the 6 KPI-card values plus a couple of extras
    (total customers, avg children, avg family size).
    """
    return {
        "avg_age": _safe_mean(df.get("AGE", pd.Series(dtype=float))),
        "median_age": _safe_median(df.get("AGE", pd.Series(dtype=float))),
        "most_common_gender": _most_common(df.get("CODE_GENDER", pd.Series(dtype=object))),
        "most_common_education": _most_common(df.get("NAME_EDUCATION_TYPE", pd.Series(dtype=object))),
        "most_common_income_type": _most_common(df.get("NAME_INCOME_TYPE", pd.Series(dtype=object))),
        "most_common_family_status": _most_common(df.get("NAME_FAMILY_STATUS", pd.Series(dtype=object))),
        "total_customers": len(df),
        "avg_children": _safe_mean(df.get("CNT_CHILDREN", pd.Series(dtype=float))),
        "avg_family_size": _safe_mean(df.get("CNT_FAM_MEMBERS", pd.Series(dtype=float))),
    }


# ---------------------------------------------------------------------------
# Per-dimension distributions
# Each returns a tidy 2-column DataFrame: [<dimension>, "Count"], ready to
# hand straight to a chart function.
# ---------------------------------------------------------------------------
def gender_distribution(df: pd.DataFrame) -> pd.DataFrame:
    return _value_counts_df(df, "CODE_GENDER", "Gender")


def age_distribution(df: pd.DataFrame) -> pd.Series:
    """Returns the raw AGE series (histogram needs raw values, not counts)."""
    return df.get("AGE", pd.Series(dtype=float)).dropna()


def education_distribution(df: pd.DataFrame) -> pd.DataFrame:
    return _value_counts_df(df, "NAME_EDUCATION_TYPE", "Education")


def family_status_distribution(df: pd.DataFrame) -> pd.DataFrame:
    return _value_counts_df(df, "NAME_FAMILY_STATUS", "Family Status")


def income_type_distribution(df: pd.DataFrame) -> pd.DataFrame:
    return _value_counts_df(df, "NAME_INCOME_TYPE", "Income Type")


def occupation_distribution(df: pd.DataFrame) -> pd.DataFrame:
    return _value_counts_df(df, "OCCUPATION_TYPE", "Occupation")


def children_distribution(df: pd.DataFrame, cap_at: int = 5) -> pd.DataFrame:
    """Number of children, capped for display (e.g. '5+')."""
    col = "CNT_CHILDREN_CAPPED" if "CNT_CHILDREN_CAPPED" in df.columns else "CNT_CHILDREN"
    if col not in df.columns:
        return pd.DataFrame(columns=["Children", "Count"])
    counts = df[col].value_counts().sort_index().reset_index()
    counts.columns = ["Children", "Count"]
    counts["Children"] = counts["Children"].astype(str).replace(str(cap_at), f"{cap_at}+")
    return counts


def family_size_distribution(df: pd.DataFrame, cap_at: int = 7) -> pd.DataFrame:
    """Family size, capped for display (e.g. '7+')."""
    col = "CNT_FAM_MEMBERS_CAPPED" if "CNT_FAM_MEMBERS_CAPPED" in df.columns else "CNT_FAM_MEMBERS"
    if col not in df.columns:
        return pd.DataFrame(columns=["Family Size", "Count"])
    counts = df[col].value_counts().sort_index().reset_index()
    counts.columns = ["Family Size", "Count"]
    counts["Family Size"] = counts["Family Size"].astype(str).replace(f"{cap_at}.0", f"{cap_at}+")
    return counts


def _value_counts_df(df: pd.DataFrame, column: str, label: str) -> pd.DataFrame:
    if column not in df.columns:
        return pd.DataFrame(columns=[label, "Count"])
    counts = df[column].value_counts(dropna=True).reset_index()
    counts.columns = [label, "Count"]
    return counts


# ---------------------------------------------------------------------------
# Cross-tabs / relationships
# ---------------------------------------------------------------------------
def age_group_by_gender(df: pd.DataFrame) -> pd.DataFrame:
    """Long-format [AGE_GROUP, CODE_GENDER, Count] table for the grouped bar chart."""
    if "AGE_GROUP" not in df.columns or "CODE_GENDER" not in df.columns:
        return pd.DataFrame(columns=["AGE_GROUP", "CODE_GENDER", "Count"])
    sub = df.dropna(subset=["AGE_GROUP", "CODE_GENDER"])
    grouped = (
        sub.groupby(["AGE_GROUP", "CODE_GENDER"], observed=True)
        .size()
        .reset_index(name="Count")
    )
    return grouped


def age_vs_income(df: pd.DataFrame, sample_size: int = 5000) -> pd.DataFrame:
    """
    Returns [AGE, income, CODE_GENDER] rows for the scatter plot, sampled
    down for chart performance if the dataset is large.
    """
    income_col = "AMT_INCOME_TOTAL_CAPPED" if "AMT_INCOME_TOTAL_CAPPED" in df.columns else "AMT_INCOME_TOTAL"
    cols = [c for c in ["AGE", income_col, "CODE_GENDER"] if c in df.columns]
    sub = df[cols].dropna(subset=[c for c in ["AGE", income_col] if c in cols])
    if len(sub) > sample_size:
        sub = sub.sample(sample_size, random_state=42)
    return sub.rename(columns={income_col: "AMT_INCOME_TOTAL"})


# ---------------------------------------------------------------------------
# Insight narrative
# ---------------------------------------------------------------------------
def build_customer_profile_insight(kpis: dict, df: pd.DataFrame) -> str:
    """Compose the 'typical customer' narrative shown in the Insights section."""
    gender = kpis["most_common_gender"]
    age = round(kpis["avg_age"])
    education = kpis["most_common_education"]
    family_status = kpis["most_common_family_status"]
    income_type = kpis["most_common_income_type"]
    children = round(kpis["avg_children"], 1)

    top_age_group = "N/A"
    if "AGE_GROUP" in df.columns and df["AGE_GROUP"].notna().any():
        top_age_group = df["AGE_GROUP"].mode().iloc[0]

    top_occupation = _most_common(df.get("OCCUPATION_TYPE", pd.Series(dtype=object)))

    return (
        f"The typical Home Credit applicant is a **{gender.lower() if gender != 'N/A' else 'N/A'}** "
        f"in the **{top_age_group}** age bracket (average age ≈ **{age}**), most often working as a "
        f"**{top_occupation.lower()}** under a **{income_type.lower()}** income arrangement. "
        f"They typically hold a **{education.lower()}** education, are **{family_status.lower()}**, "
        f"and have on average **{children}** children."
    )


# ---------------------------------------------------------------------------
# One-call convenience wrapper
# ---------------------------------------------------------------------------
def run_full_analysis(df: pd.DataFrame) -> dict:
    """
    Runs every analysis step and bundles the results into a single dict.
    Handy for app.py, notebooks, or unit tests that want everything at once.
    """
    df = add_age_groups(df)
    kpis = compute_kpis(df)

    return {
        "df": df,  # dataframe with AGE_GROUP added
        "kpis": kpis,
        "gender": gender_distribution(df),
        "age": age_distribution(df),
        "education": education_distribution(df),
        "family_status": family_status_distribution(df),
        "income_type": income_type_distribution(df),
        "occupation": occupation_distribution(df),
        "children": children_distribution(df),
        "family_size": family_size_distribution(df),
        "age_group_by_gender": age_group_by_gender(df),
        "age_vs_income": age_vs_income(df),
        "insight": build_customer_profile_insight(kpis, df),
    }
 