import logging
from datetime import datetime
import os


def get_logger(name=None):
    # Create a log filename based on current timestamp
    log_file = f"{datetime.now().strftime('%m_%d_%Y')}.log"

    # Setup directory path
    log_dir = os.path.join(os.getcwd(), "run_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger(name or __name__)
    logger.setLevel(logging.INFO)

    # Attach a file handler once per logger to avoid duplicated log lines.
    if not logger.handlers:
        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s in %(name)s: %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        # Prevent duplicate logs when API and UI run together.
        logger.propagate = False
    
    return logger