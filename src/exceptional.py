import sys
import traceback
from src.logger import logging

def log_exception(exc_type, exc_value, exc_traceback):
    # Skip logging for Ctrl+C exit
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    # Header
    header = "\n An Error Occurred:\n" + "-" * 50
    print(header)
    logging.error(header)

    # Traceback info
    traceback_details = traceback.extract_tb(exc_traceback)
    for filename, line, func, text in traceback_details:
        info = (
            f" File     : {filename}\n"
            f" Line     : {line}\n"
            f" Function : {func}\n"
            f" Code     : {text}\n" +
            "-" * 50
        )
        print(info)
        logging.error(info)

    # Final error type & message
    final_message = (
        f" Error Type   : {exc_type.__name__}\n"
        f" Error Message: {exc_value}\n" +
        "-" * 50
    )
    print(final_message)
    logging.error(final_message)

# Set as global exception hook
sys.excepthook = log_exception


