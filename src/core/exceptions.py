"""
Centralised exception logging for the EduScore Predictor project.

`log_exception` can be called directly from a bare ``except`` block via
``sys.exc_info()``, and is also installed as the global ``sys.excepthook``
so that unhandled exceptions are automatically captured.
"""
import sys
import traceback
import logging

from src.core import logger as _  # ensure logging is configured


def log_exception(exc_type, exc_value, exc_traceback) -> None:
    """Log a full structured traceback for the given exception info triple."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    separator = "-" * 60
    header = f"\n Exception Caught\n{separator}"
    logging.error(header)

    for filename, line, func, text in traceback.extract_tb(exc_traceback):
        detail = (
            f"  File     : {filename}\n"
            f"  Line     : {line}\n"
            f"  Function : {func}\n"
            f"  Code     : {text}\n"
            f"{separator}"
        )
        logging.error(detail)

    summary = (
        f"  Type    : {exc_type.__name__}\n"
        f"  Message : {exc_value}\n"
        f"{separator}"
    )
    logging.error(summary)


# Install as the global exception hook (non-interactive scripts)
sys.excepthook = log_exception
