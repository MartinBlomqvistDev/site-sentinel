"""
Project-wide logging setup.

Usage in any module or script:

    from site_sentinel.logging_utils import get_logger
    logger = get_logger(__name__)
    logger.info("Processing %d files", n)
"""

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a named logger with a consistent format.

    Safe to call multiple times with the same name — handlers are only
    attached once, so there's no risk of duplicate log lines.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger
