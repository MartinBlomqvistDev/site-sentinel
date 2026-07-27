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
        # Force UTF-8 on the stream. Several pipeline messages contain "→", and
        # on a Windows console stdout defaults to cp1252, so logging raised
        # UnicodeEncodeError and printed a traceback for every fold instead of
        # the result. Logging swallows the error, so the run continued and the
        # numbers were simply lost.
        stream = sys.stdout
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                # Detached or redirected in a way that cannot be reconfigured;
                # the errors="replace" fallback below still keeps output going.
                pass
        handler = logging.StreamHandler(stream)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger
