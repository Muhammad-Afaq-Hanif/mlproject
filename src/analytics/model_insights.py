"""
Model Insights — loads saved model artifacts and computes inference diagnostics
for the dashboard without re-running training.
"""
import os
import json

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.core.utils import load_object


_PREPROCESSOR_PATH = os.path.join("artifacts", "transform.pkl")
_MODEL_PATH        = os.path.join("artifacts", "Model.pkl")
_SCORES_PATH       = os.path.join("artifacts", "model_scores.json")
_TEST_PATH         = os.path.join("artifacts", "test.csv")
_TARGET            = "math_score"


def get_model_leaderboard() -> pd.DataFrame:
    """Read saved scores JSON and return a sorted leaderboard DataFrame."""
    if not os.path.exists(_SCORES_PATH):
        return pd.DataFrame(columns=["Rank", "Model", "R² Score", "Grade"])

    with open(_SCORES_PATH) as f:
        scores = json.load(f)

    df = (
        pd.DataFrame(list(scores.items()), columns=["Model", "R² Score"])
        .sort_values("R² Score", ascending=False)
        .reset_index(drop=True)
    )
    df["R² Score"] = df["R² Score"].round(4)
    df.insert(0, "Rank", range(1, len(df) + 1))
    df["Grade"] = df["R² Score"].apply(
        lambda x: "🥇 Excellent" if x >= 0.90 else
                  "🥈 Great"     if x >= 0.85 else
                  "🥉 Good"      if x >= 0.80 else
                  "⚠️ Fair"      if x >= 0.70 else "❌ Weak"
    )
    return df[["Rank", "Model", "R² Score", "Grade"]]


def get_inference_metrics() -> dict:
    """Run the saved best model on the test set; return metrics and prediction arrays."""
    try:
        preprocessor = load_object(_PREPROCESSOR_PATH)
        model        = load_object(_MODEL_PATH)
        test_df      = pd.read_csv(_TEST_PATH)

        X_test  = test_df.drop(columns=[_TARGET])
        y_test  = test_df[_TARGET].to_numpy()
        y_pred  = model.predict(preprocessor.transform(X_test))
        residuals = y_test - y_pred

        return {
            "y_test":    y_test.tolist(),
            "y_pred":    y_pred.tolist(),
            "residuals": residuals.tolist(),
            "r2":        round(r2_score(y_test, y_pred), 4),
            "mae":       round(mean_absolute_error(y_test, y_pred), 3),
            "rmse":      round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 3),
            "model_name": type(model).__name__,
        }
    except Exception as e:
        return {"error": str(e)}


def get_feature_importance() -> pd.DataFrame | None:
    """Return feature importances for tree-based models; None for others."""
    try:
        preprocessor = load_object(_PREPROCESSOR_PATH)
        model        = load_object(_MODEL_PATH)

        if not hasattr(model, "feature_importances_"):
            return None

        num_features = ["reading_score", "writing_score"]
        cat_features = (
            preprocessor
            .named_transformers_["categorical"]
            .named_steps["encoder"]
            .get_feature_names_out([
                "gender", "race_ethnicity", "parental_level_of_education",
                "lunch", "test_preparation_course",
            ])
            .tolist()
        )
        all_features = num_features + cat_features
        importances  = model.feature_importances_
        n = min(len(importances), len(all_features))

        df = (
            pd.DataFrame({"Feature": all_features[:n], "Importance": importances[:n]})
            .sort_values("Importance", ascending=False)
            .reset_index(drop=True)
        )
        df["Importance"] = df["Importance"].round(4)
        df["Group"] = df["Feature"].apply(
            lambda f: "Numeric Scores"
            if f in ["reading_score", "writing_score"]
            else f.split("_")[0].title()
        )
        return df

    except Exception:
        return None
