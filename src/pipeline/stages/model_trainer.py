"""
Model Training Stage — evaluates multiple regressors with GridSearchCV,
selects the best by R² score, and persists it along with a leaderboard JSON.
"""
import os
import sys
import json
import logging
from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.core.exceptions import log_exception
from src.core.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    model_path: str  = field(default_factory=lambda: os.path.join("artifacts", "Model.pkl"))
    scores_path: str = field(default_factory=lambda: os.path.join("artifacts", "model_scores.json"))


class ModelTrainer:
    """Trains and evaluates candidate regressors; persists the best model."""

    MODELS = {
        "Linear Regression":       LinearRegression(),
        "Decision Tree":           DecisionTreeRegressor(),
        "Random Forest":           RandomForestRegressor(),
        "Gradient Boosting":       GradientBoostingRegressor(),
        "K-Nearest Neighbors":     KNeighborsRegressor(),
        "XGBoost":                 XGBRegressor(),
        "CatBoost":                CatBoostRegressor(verbose=0),
        "AdaBoost":                AdaBoostRegressor(),
    }

    PARAMS = {
        "Linear Regression":   {},
        "Decision Tree":       {"max_depth": [None, 10, 20, 30], "min_samples_split": [2, 5, 10]},
        "Random Forest":       {"n_estimators": [100, 200], "max_depth": [None, 10, 20], "min_samples_split": [2, 5]},
        "Gradient Boosting":   {"n_estimators": [100, 150], "learning_rate": [0.1, 0.05], "max_depth": [3, 5]},
        "K-Nearest Neighbors": {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]},
        "XGBoost":             {"n_estimators": [100, 150], "learning_rate": [0.1, 0.05], "max_depth": [3, 5]},
        "CatBoost":            {"iterations": [100, 200], "learning_rate": [0.1, 0.05], "depth": [3, 5]},
        "AdaBoost":            {"n_estimators": [50, 100], "learning_rate": [0.1, 0.5, 1.0]},
    }

    def __init__(self) -> None:
        self.config = ModelTrainerConfig()

    def run(self, train_arr: np.ndarray, test_arr: np.ndarray) -> None:
        """Evaluate all models, save the best one and the leaderboard JSON."""
        logging.info("ModelTrainer: starting evaluation")
        try:
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test,  y_test  = test_arr[:, :-1],  test_arr[:, -1]

            scores, fitted_models = evaluate_models(
                X_train, y_train, X_test, y_test,
                models=self.MODELS,
                params=self.PARAMS,
            )

            best_name  = max(scores, key=scores.get)
            best_model = fitted_models[best_name]
            best_score = scores[best_name]

            logging.info(f"ModelTrainer: best model → {best_name}  R²={best_score:.4f}")

            save_object(self.config.model_path, best_model)

            with open(self.config.scores_path, "w") as f:
                json.dump(scores, f, indent=2)

            logging.info("ModelTrainer: artifacts saved")

        except Exception:
            log_exception(*sys.exc_info())
            raise
