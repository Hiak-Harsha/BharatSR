"""
BharatSR — Core Logging Setup
Provides structured and standardized logging across the backend.
"""

import logging
import sys


def setup_logging(log_level: str = "INFO") -> None:
    """Configures application-wide logging format and handlers."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def get_logger(name: str = "bharatsr") -> logging.Logger:
    """Returns a named logger instance."""
    return logging.getLogger(name)


logger = get_logger()
