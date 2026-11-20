"""
Prediction pipeline — loads saved model artifacts and returns a single prediction.
"""
import os

import pandas as pd

from src.core.utils import load_object


_PREPROCESSOR_PATH = os.path.join("artifacts", "transform.pkl")
_MODEL_PATH        = os.path.join("artifacts", "Model.pkl")


def predict_math_score(input_df: pd.DataFrame) -> float:
    """
    Apply saved preprocessor and model to *input_df*.

    Parameters
    ----------
    input_df : pd.DataFrame
        One row with the 7 input features expected by the model.

    Returns
    -------
    float
        Predicted math score.
    """
    preprocessor = load_object(_PREPROCESSOR_PATH)
    model        = load_object(_MODEL_PATH)
    transformed  = preprocessor.transform(input_df)
    return float(model.predict(transformed)[0])
