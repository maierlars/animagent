"""Structured logging (structlog) configuration.

Logs go to stderr so that stdout stays usable for machine-readable output.
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: str = "INFO", json_logs: bool = False) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=numeric, stream=sys.stderr, format="%(message)s")
    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer() if json_logs else structlog.dev.ConsoleRenderer(colors=False)
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=False,
    )


def get_logger(name: str) -> structlog.typing.FilteringBoundLogger:
    return structlog.get_logger(name)
