"""Logging utilities for the Smart Email Assistant."""
from __future__ import annotations

import logging
from functools import lru_cache


@lru_cache(maxsize=1)
def get_logger(name: str = "smart_email_assistant") -> logging.Logger:
    """Return a configured logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


__all__ = ["get_logger"]
