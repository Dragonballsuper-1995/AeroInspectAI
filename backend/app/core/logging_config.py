"""Logging configuration."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(log_file: str = "aeroinspect.log") -> logging.Logger:
    """Configure and return a logger for the application."""
    
    logger = logging.getLogger("aeroinspect")
    logger.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


logger = setup_logging("logs/aeroinspect.log")
