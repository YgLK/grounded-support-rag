"""Project logging helpers."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

__all__ = ["configure_logging", "get_logger"]

DEFAULT_LOG_LEVEL = "INFO"
LOG_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}
DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def _sanitize_filename_part(value: str | None, fallback: str) -> str:
    raw = (value or "").strip().lower()
    if not raw:
        return fallback
    sanitized = "".join(
        char if char.isalnum() or char in {"-", "_"} else "-" for char in raw
    ).strip("-_")
    return sanitized or fallback


def _build_log_file_path(log_dir: str | Path, command_name: str | None = None) -> Path:
    directory = Path(log_dir)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
    command = _sanitize_filename_part(command_name, "grounded-support-rag")
    return directory / f"{timestamp}-{command}.log"


def _resolved_level(level: str | int | None) -> int:
    if isinstance(level, int):
        return level
    raw_level = DEFAULT_LOG_LEVEL if level is None else level
    return LOG_LEVELS.get(str(raw_level).upper(), logging.INFO)


def configure_logging(
    level: str | int | None = None,
    *,
    log_dir: str | Path | None = None,
    command_name: str | None = None,
) -> Path | None:
    logger = logging.getLogger("support_graph")
    logger.setLevel(_resolved_level(level))
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    resolved_log_dir = log_dir
    log_path: Path | None = None
    if resolved_log_dir is not None and str(resolved_log_dir).strip():
        log_path = _build_log_file_path(resolved_log_dir, command_name=command_name)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return log_path


def get_logger(name: str) -> logging.Logger:
    if name.startswith("support_graph"):
        return logging.getLogger(name)
    return logging.getLogger(f"support_graph.{name}")
