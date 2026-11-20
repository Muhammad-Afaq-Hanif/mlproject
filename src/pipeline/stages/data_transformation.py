"""
Data Transformation Stage — builds and fits a sklearn ColumnTransformer
that handles numeric imputation/scaling and categorical encoding.
"""
import os
import sys
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.core.exceptions import log_exception
from src.core.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_path: str = field(
        default_factory=lambda: os.path.join("artifacts", "transform.pkl")
    )


class DataTransformation:
    """Fits a preprocessor on training data and transforms both splits."""

    NUMERIC_FEATURES = ["reading_score", "writing_score"]
    CATEGORICAL_FEATURES = [
        "gender",
        "race_ethnicity",
        "parental_level_of_education",
        "lunch",
        "test_preparation_course",
    ]
    TARGET = "math_score"

    def __init__(self) -> None:
        self.config = DataTransformationConfig()

    def _build_preprocessor(self) -> ColumnTransformer:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ])
        categorical_pipeline = Pipeline([
            ("imputer",  SimpleImputer(strategy="most_frequent")),
            ("encoder",  OneHotEncoder(handle_unknown="ignore")),
            ("scaler",   StandardScaler(with_mean=False)),
        ])
        return ColumnTransformer([
            ("numeric",      numeric_pipeline,      self.NUMERIC_FEATURES),
            ("categorical",  categorical_pipeline,  self.CATEGORICAL_FEATURES),
        ])

    def run(self, train_path: str, test_path: str) -> tuple[np.ndarray, np.ndarray, str]:
        """Fit on train, transform both; returns (train_arr, test_arr, preprocessor_path)."""
        logging.info("DataTransformation: starting")
        try:
            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)

            preprocessor = self._build_preprocessor()

            X_train = train_df.drop(columns=[self.TARGET])
            y_train = train_df[self.TARGET]
            X_test  = test_df.drop(columns=[self.TARGET])
            y_test  = test_df[self.TARGET]

            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr  = preprocessor.transform(X_test)

            train_arr = np.c_[X_train_arr, y_train.to_numpy()]
            test_arr  = np.c_[X_test_arr,  y_test.to_numpy()]

            save_object(self.config.preprocessor_path, preprocessor)
            logging.info("DataTransformation: complete")

            return train_arr, test_arr, self.config.preprocessor_path

        except Exception:
            log_exception(*sys.exc_info())
            raise
