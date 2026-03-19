"""Project logging helpers."""

from __future__ import annotations

import logging
import os


DEFAULT_LOG_LEVEL = "INFO"
LOG_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}


def _resolved_level(level: str | int | None) -> int:
    if isinstance(level, int):
        return level
    raw_level = os.environ.get("SUPPORT_GRAPH_LOG_LEVEL", DEFAULT_LOG_LEVEL)
    if level is not None:
        raw_level = level
    return LOG_LEVELS.get(str(raw_level).upper(), logging.INFO)


def configure_logging(level: str | int | None = None) -> None:
    logger = logging.getLogger("support_graph")
    logger.setLevel(_resolved_level(level))
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(handler)
    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    if name.startswith("support_graph"):
        return logging.getLogger(name)
    return logging.getLogger(f"support_graph.{name}")


__all__ = ["configure_logging", "get_logger"]
