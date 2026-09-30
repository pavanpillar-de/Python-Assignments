
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
 
# ------------------------------------------------------------------
# SHARED COLOR PALETTE
# ------------------------------------------------------------------
BLUE = "#2E86AB"
RED = "#E63946"
OUTCOME_COLORS = {"Repaid": BLUE, "Default": RED}
TARGET_COLORS = {0: BLUE, 1: RED}
 
 
def _ensure_outcome_col(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with an OUTCOME column, deriving it from TARGET if missing."""
    if "OUTCOME" not in df.columns and "TARGET" in df.columns:
        df = df.copy()
        df["OUTCOME"] = df["TARGET"].map({0: "Repaid", 1: "Default"})
    return df
 
 
# ------------------------------------------------------------------
# OUTCOME / VOLUME CHARTS
# ------------------------------------------------------------------
def outcome_pie_chart(df: pd.DataFrame) -> go.Figure:
    """Donut chart of Repaid vs Default counts."""
    df = _ensure_outcome_col(df)
    counts = df["OUTCOME"].value_counts().reset_index()
    counts.columns = ["Outcome", "Count"]
    fig = px.pie(
        counts, names="Outcome", values="Count", hole=0.5,
        color="Outcome", color_discrete_map=OUTCOME_COLORS,
    )
    fig.update_traces(textinfo="percent+label")
    return fig
 
 
def category_count_bar(df: pd.DataFrame, col: str, title: str = None) -> go.Figure:
    """Simple bar chart of value counts for a categorical column."""
    counts = df[col].value_counts().reset_index()
    counts.columns = [col, "Count"]
    fig = px.bar(counts, x=col, y="Count", color=col, text="Count", title=title)
    fig.update_layout(showlegend=False)
    return fig
 
 
def outcome_count_bar(df: pd.DataFrame) -> go.Figure:
    """Bar chart (not donut) of raw Default vs Non-Default counts."""
    df = _ensure_outcome_col(df)
    counts = df["OUTCOME"].value_counts().reindex(["Repaid", "Default"]).reset_index()
    counts.columns = ["Outcome", "Count"]
    fig = px.bar(
        counts, x="Outcome", y="Count", color="Outcome", text="Count",
        color_discrete_map=OUTCOME_COLORS,
    )
    fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig.update_layout(showlegend=False, yaxis_title="Customers")
    return fig
 
 
def treemap_by_category(
    df: pd.DataFrame,
    category_col: str,
    value_col: str,
    agg: str = "sum",
    title: str = None,
) -> go.Figure:
    """
    Treemap sizing each category by an aggregated numeric value
    (e.g. total AMT_CREDIT by NAME_INCOME_TYPE).
    """
    grp = df.groupby(category_col)[value_col].agg(agg).reset_index()
    grp.columns = [category_col, value_col]
    grp = grp.sort_values(value_col, ascending=False)
    fig = px.treemap(
        grp, path=[category_col], values=value_col,
        color=value_col, color_continuous_scale="Blues", title=title,
    )
    fig.update_traces(textinfo="label+value")
    fig.update_layout(coloraxis_showscale=False)
    return fig
 
 
# ------------------------------------------------------------------
# DEFAULT RATE BY SEGMENT
# ------------------------------------------------------------------
def default_rate_by_segment(
    df: pd.DataFrame,
    segment_col: str,
    min_segment_size: int = 50,
    top_n: int = None,
    horizontal: bool = True,
) -> go.Figure:
    """
    Bar chart of default rate (%) grouped by a categorical column.
    Segments below `min_segment_size` applicants are excluded.
    """
    seg_df = (
        df.groupby(segment_col)["TARGET"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "Default Rate", "count": "Applications"})
    )
    seg_df = seg_df[seg_df["Applications"] >= min_segment_size]
    seg_df["Default Rate"] = (seg_df["Default Rate"] * 100).round(2)
    seg_df = seg_df.sort_values("Default Rate", ascending=False)
    if top_n:
        seg_df = seg_df.head(top_n)
 
    if horizontal:
        fig = px.bar(
            seg_df, x="Default Rate", y=segment_col, orientation="h",
            color="Default Rate", color_continuous_scale="Reds",
            hover_data=["Applications"], text="Default Rate",
        )
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
    else:
        fig = px.bar(
            seg_df, x=segment_col, y="Default Rate", text="Default Rate",
            color="Default Rate", color_continuous_scale="Reds",
            hover_data=["Applications"],
        )
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(coloraxis_showscale=False)
 
    return fig
 
 
# ------------------------------------------------------------------
# DISTRIBUTIONS
# ------------------------------------------------------------------
def distribution_histogram(
    df: pd.DataFrame,
    col: str,
    by_outcome: bool = False,
    nbins: int = 40,
    clip_quantile: float = None,
) -> go.Figure:
    """
    Histogram of a numeric column. If by_outcome=True, overlays Repaid vs
    Default. clip_quantile (e.g. 0.99) caps extreme outliers for readability.
    """
    plot_df = df.copy()
    if clip_quantile:
        cap = plot_df[col].quantile(clip_quantile)
        plot_df[col] = plot_df[col].clip(upper=cap)
 
    if by_outcome:
        plot_df = _ensure_outcome_col(plot_df)
        fig = px.histogram(
            plot_df, x=col, color="OUTCOME", nbins=nbins, barmode="overlay",
            opacity=0.65, color_discrete_map=OUTCOME_COLORS,
        )
    else:
        fig = px.histogram(plot_df, x=col, nbins=nbins)
        fig.update_layout(showlegend=False)
 
    fig.update_layout(yaxis_title="Applicants")
    return fig
 
 
def ratio_violin_by_outcome(df: pd.DataFrame, col: str, clip_quantile: float = 0.99) -> go.Figure:
    """Violin+box plot comparing a numeric column's distribution by outcome."""
    df = _ensure_outcome_col(df)
    plot_df = df.copy()
    if clip_quantile:
        cap = plot_df[col].quantile(clip_quantile)
        plot_df[col] = plot_df[col].clip(upper=cap)
 
    fig = px.violin(
        plot_df, x="OUTCOME", y=col, color="OUTCOME",
        color_discrete_map=OUTCOME_COLORS, box=True, points=False,
    )
    fig.update_layout(showlegend=False)
    return fig
 
 
def ratio_box_by_target(df: pd.DataFrame, col: str, clip_quantile: float = 0.99) -> go.Figure:
    """Boxplot of a numeric column split by raw TARGET (0/1)."""
    plot_df = df.copy()
    if clip_quantile:
        cap = plot_df[col].quantile(clip_quantile)
        plot_df[col] = plot_df[col].clip(upper=cap)
 
    fig = px.box(
        plot_df, x="TARGET", y=col, color="TARGET",
        color_discrete_map=TARGET_COLORS,
        labels={"TARGET": "Outcome (0=Repaid, 1=Default)"},
    )
    return fig
 
 
# ------------------------------------------------------------------
# CORRELATION / DRIVERS
# ------------------------------------------------------------------
def correlation_with_target_bar(df: pd.DataFrame, numeric_cols: list) -> go.Figure:
    """
    Horizontal bar chart ranking numeric features by their correlation with
    TARGET (diverging red/blue scale — negative = protective, positive = risk).
    """
    corr_series = (
        df[numeric_cols + ["TARGET"]]
        .corr(numeric_only=True)["TARGET"]
        .drop("TARGET")
        .sort_values()
    )
    corr_df = corr_series.reset_index()
    corr_df.columns = ["Feature", "Correlation with Default"]
 
    fig = px.bar(
        corr_df, x="Correlation with Default", y="Feature", orientation="h",
        color="Correlation with Default", color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
    )
    fig.update_layout(height=max(400, 25 * len(corr_df)), coloraxis_showscale=False)
    return fig
 
 
def scatter_by_outcome(df: pd.DataFrame, x_col: str, y_col: str, sample_size: int = 5000) -> go.Figure:
    """Scatter plot of two numeric columns colored by outcome, sampled for performance."""
    df = _ensure_outcome_col(df)
    sample_df = df.sample(min(sample_size, len(df)), random_state=42)
    fig = px.scatter(
        sample_df, x=x_col, y=y_col, color="OUTCOME", opacity=0.4,
        color_discrete_map=OUTCOME_COLORS,
    )
    return fig
 
 
# ------------------------------------------------------------------
# GROUPED / RATE-BASED CHARTS
# ------------------------------------------------------------------
def rate_by_bucket_bar(
    df: pd.DataFrame,
    bucket_col: str,
    x_title: str = None,
    min_segment_size: int = 50,
) -> go.Figure:
    """
    Bar chart of default rate (%) by an already-bucketed/discrete numeric
    column (e.g. number of children, region rating). Unlike
    default_rate_by_segment, keeps natural bucket order instead of sorting
    by rate.
    """
    grp = (
        df.groupby(bucket_col)["TARGET"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "Default Rate", "count": "Applications"})
    )
    grp = grp[grp["Applications"] >= min_segment_size]
    grp["Default Rate"] = (grp["Default Rate"] * 100).round(2)
 
    fig = px.bar(
        grp, x=bucket_col, y="Default Rate", text="Default Rate",
        color="Default Rate", color_continuous_scale="Reds",
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(xaxis_title=x_title or bucket_col, coloraxis_showscale=False)
    return fig
 
 
# ------------------------------------------------------------------
# TABLES (returned as DataFrames, rendered via st.dataframe by caller)
# ------------------------------------------------------------------
def top_risk_segments_table(
    df: pd.DataFrame,
    group_cols: list,
    min_segment_size: int = 50,
    top_n: int = 10,
) -> pd.DataFrame:
    """Return the top N riskiest segments (by default rate) for 1-2 group columns."""
    table = (
        df.groupby(group_cols)["TARGET"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "Default Rate", "count": "Applications"})
    )
    table = table[table["Applications"] >= min_segment_size]
    table["Default Rate"] = (table["Default Rate"] * 100).round(2)
    table = table.sort_values("Default Rate", ascending=False).head(top_n)
    return table
 
 
def flag_signal_table(df: pd.DataFrame, flag_cols: list, min_segment_size: int = 50) -> pd.DataFrame:
    """
    For a list of binary FLAG_* columns, compute the default-rate gap between
    flag=1 vs flag=0, ranked by strongest absolute signal.
    """
    results = []
    for f in flag_cols:
        if df[f].nunique() <= 2:
            sub = df.groupby(f)["TARGET"].agg(["mean", "count"])
            for val, row in sub.iterrows():
                results.append({
                    "Flag": f, "Value": int(val), "Default Rate": row["mean"] * 100,
                    "Applications": int(row["count"]),
                })
 
    flag_df = pd.DataFrame(results)
    if flag_df.empty:
        return flag_df
 
    flag_df = flag_df[flag_df["Applications"] >= min_segment_size]
    pivot = flag_df.pivot_table(index="Flag", columns="Value", values="Default Rate")
    pivot = pivot.rename(columns={0: "Flag = 0", 1: "Flag = 1"}).dropna()
    pivot["Gap (pp)"] = pivot.get("Flag = 1", np.nan) - pivot.get("Flag = 0", np.nan)
    pivot = pivot.sort_values("Gap (pp)", key=abs, ascending=False).round(2)
    return pivot
 