"""
Data Ingestion Stage — reads the raw CSV source, saves it under artifacts/,
and writes train/test splits.
"""
import os
import sys
import logging
from dataclasses import dataclass, field

import pandas as pd
from sklearn.model_selection import train_test_split

from src.core.exceptions import log_exception


@dataclass
class DataIngestionConfig:
    raw_data_path: str  = field(default_factory=lambda: os.path.join("artifacts", "raw.csv"))
    train_data_path: str = field(default_factory=lambda: os.path.join("artifacts", "train.csv"))
    test_data_path: str  = field(default_factory=lambda: os.path.join("artifacts", "test.csv"))


class DataIngestion:
    """Reads source data, saves raw copy, and performs train/test split."""

    SOURCE_PATH = os.path.join("data", "students.csv")

    def __init__(self) -> None:
        self.config = DataIngestionConfig()

    def run(self) -> tuple[str, str]:
        """Execute ingestion. Returns (train_path, test_path)."""
        logging.info("DataIngestion: starting")
        try:
            df = pd.read_csv(self.SOURCE_PATH)
            logging.info(f"DataIngestion: loaded {len(df)} rows from {self.SOURCE_PATH}")

            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)
            df.to_csv(self.config.raw_data_path, index=False)

            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            train_df.to_csv(self.config.train_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)

            logging.info("DataIngestion: complete")
            return self.config.train_data_path, self.config.test_data_path

        except Exception:
            log_exception(*sys.exc_info())
            raise
