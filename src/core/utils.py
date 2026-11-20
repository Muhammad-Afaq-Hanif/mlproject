"""
Shared utility functions for the EduScore Predictor project.
"""
import os
import sys
import json
import logging

import dill
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

from src.core.exceptions import log_exception


def save_object(file_path: str, obj: object) -> None:
    """Serialise *obj* to *file_path* using dill, creating parent dirs as needed."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            dill.dump(obj, f)
    except Exception:
        log_exception(*sys.exc_info())


def load_object(file_path: str) -> object:
    """Deserialise and return the object stored at *file_path*."""
    try:
        with open(file_path, "rb") as f:
            return dill.load(f)
    except Exception:
        log_exception(*sys.exc_info())


def evaluate_models(
    X_train, y_train, X_test, y_test, models: dict, params: dict
) -> tuple[dict, dict]:
    """
    Fit each model (optionally with GridSearchCV) and return:
        report      → {model_name: r2_score}
        best_models → {model_name: fitted_estimator}
    """
    try:
        report: dict = {}
        best_models: dict = {}

        for name, model in models.items():
            model_params = params.get(name)

            if model_params:
                logging.info(f"Tuning hyperparameters for {name}")
                gs = GridSearchCV(model, model_params, cv=3, n_jobs=-1, verbose=0)
                gs.fit(X_train, y_train)
                fitted = gs.best_estimator_
            else:
                logging.info(f"Using default params for {name}")
                model.fit(X_train, y_train)
                fitted = model

            score = r2_score(y_test, fitted.predict(X_test))
            report[name] = score
            best_models[name] = fitted
            logging.info(f"  {name}: R²={score:.4f}")

        return report, best_models

    except Exception:
        log_exception(*sys.exc_info())
        return {}, {}
