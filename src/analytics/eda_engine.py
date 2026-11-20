"""
EDA Engine — computes all analytics from artifacts/raw.csv for the dashboard.
Returns plain Python/Pandas objects; no Streamlit imports here.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats


_DATA_PATH = os.path.join("artifacts", "raw.csv")

_EDU_ORDER = [
    "some high school",
    "high school",
    "some college",
    "associate's degree",
    "bachelor's degree",
    "master's degree",
]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(_DATA_PATH)
    df["avg_score"] = df[["math_score", "reading_score", "writing_score"]].mean(axis=1)
    df["parental_level_of_education"] = pd.Categorical(
        df["parental_level_of_education"], categories=_EDU_ORDER, ordered=True
    )
    return df


def get_summary_stats(df: pd.DataFrame) -> dict:
    score_cols = ["math_score", "reading_score", "writing_score"]
    return {
        col: {
            "mean":   round(df[col].mean(), 2),
            "median": round(df[col].median(), 2),
            "std":    round(df[col].std(), 2),
            "min":    int(df[col].min()),
            "max":    int(df[col].max()),
            "q25":    round(df[col].quantile(0.25), 2),
            "q75":    round(df[col].quantile(0.75), 2),
        }
        for col in score_cols
    }


def get_kpis(df: pd.DataFrame) -> dict:
    prep_boost = (
        df[df["test_preparation_course"] == "completed"]["math_score"].mean()
        - df[df["test_preparation_course"] == "none"]["math_score"].mean()
    )
    lunch_boost = (
        df[df["lunch"] == "standard"]["math_score"].mean()
        - df[df["lunch"] == "free/reduced"]["math_score"].mean()
    )
    return {
        "total_students":    len(df),
        "avg_math":          round(df["math_score"].mean(), 1),
        "avg_reading":       round(df["reading_score"].mean(), 1),
        "avg_writing":       round(df["writing_score"].mean(), 1),
        "test_prep_boost":   round(prep_boost, 1),
        "lunch_boost":       round(lunch_boost, 1),
        "top_ethnicity":     df.groupby("race_ethnicity")["math_score"].mean().idxmax(),
        "pct_perfect_math":  round((df["math_score"] == 100).mean() * 100, 1),
        "pct_below_40_math": round((df["math_score"] < 40).mean() * 100, 1),
    }


def get_score_distribution(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([
        df[["math_score"]].rename(columns={"math_score": "score"}).assign(subject="Math"),
        df[["reading_score"]].rename(columns={"reading_score": "score"}).assign(subject="Reading"),
        df[["writing_score"]].rename(columns={"writing_score": "score"}).assign(subject="Writing"),
    ], ignore_index=True)


def get_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df[["math_score", "reading_score", "writing_score", "avg_score"]].corr().round(3)


def get_gender_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("gender")[["math_score", "reading_score", "writing_score"]]
        .mean().round(2).reset_index()
    )


def get_edu_impact(df: pd.DataFrame) -> pd.DataFrame:
    result = (
        df.groupby("parental_level_of_education", observed=True)[
            ["math_score", "reading_score", "writing_score"]
        ].mean().round(2).reset_index()
    )
    result["parental_level_of_education"] = result["parental_level_of_education"].astype(str)
    return result


def get_lunch_impact(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("lunch")[["math_score", "reading_score", "writing_score"]]
        .mean().round(2).reset_index()
    )


def get_test_prep_impact(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("test_preparation_course")[["math_score", "reading_score", "writing_score"]]
        .mean().round(2).reset_index()
    )


def get_ethnicity_stats(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("race_ethnicity")
        .agg(
            count=("math_score", "count"),
            avg_math=("math_score", "mean"),
            avg_reading=("reading_score", "mean"),
            avg_writing=("writing_score", "mean"),
        ).round(2).reset_index()
    )


def get_score_bands(df: pd.DataFrame) -> pd.DataFrame:
    bins   = [0, 40, 60, 75, 90, 101]
    labels = ["< 40 (At Risk)", "40–60 (Below Avg)", "60–75 (Average)", "75–90 (Good)", "90–100 (Excellent)"]
    df2 = df.copy()
    df2["band"] = pd.cut(df2["math_score"], bins=bins, labels=labels, right=False)
    counts = df2["band"].value_counts().reindex(labels).fillna(0).astype(int).reset_index()
    counts.columns = ["Score Band", "Count"]
    counts["Percentage"] = (counts["Count"] / len(df2) * 100).round(1)
    return counts


def test_prep_ttest(df: pd.DataFrame) -> dict:
    completed = df[df["test_preparation_course"] == "completed"]["math_score"]
    none      = df[df["test_preparation_course"] == "none"]["math_score"]
    t_stat, p_val = stats.ttest_ind(completed, none)
    return {
        "t_statistic": round(t_stat, 4),
        "p_value":     round(p_val, 6),
        "significant": p_val < 0.05,
    }


def score_percentile(df: pd.DataFrame, score: float) -> float:
    return round(stats.percentileofscore(df["math_score"], score, kind="rank"), 1)
