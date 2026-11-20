"""
Training pipeline orchestrator.

Runs the three stages in sequence:
  1. DataIngestion   — loads and splits the raw dataset
  2. DataTransformation — fits preprocessor, transforms features
  3. ModelTrainer    — evaluates models, saves best + leaderboard
"""
import logging
import sys

from src.core.exceptions import log_exception
from src.pipeline.stages.data_ingestion import DataIngestion
from src.pipeline.stages.data_transformation import DataTransformation
from src.pipeline.stages.model_trainer import ModelTrainer


def run_training_pipeline() -> None:
    """Execute the full end-to-end training workflow."""
    logging.info("=== Training Pipeline Started ===")
    try:
        train_path, test_path = DataIngestion().run()
        train_arr, test_arr, _ = DataTransformation().run(train_path, test_path)
        ModelTrainer().run(train_arr, test_arr)
        logging.info("=== Training Pipeline Completed ===")
    except Exception:
        log_exception(*sys.exc_info())
        raise
