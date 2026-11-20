"""
Core logging configuration for the EduScore Predictor project.

Creates a timestamped log file in the project-root `logs/` directory
on every interpreter start. Import this module early so the global
`logging` object is configured before any other module uses it.
"""
import logging
import os
from datetime import datetime

_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs")
os.makedirs(_LOG_DIR, exist_ok=True)

_log_filename = datetime.now().strftime("log_%Y-%m-%d_%H-%M-%S.log")
_log_path = os.path.join(_LOG_DIR, _log_filename)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(_log_path),
    ],
)
