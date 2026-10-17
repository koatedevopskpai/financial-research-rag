"""
Structured JSON logging with correlation IDs — built on structlog.

Design decisions:
  - JSON output for machine parsing (log aggregation in Azure Monitor, Datadog, etc.)
  - Correlation IDs per request to trace end-to-end flow across pipeline steps
  - Timestamps in ISO 8601 with timezone
  - Log levels respected — DEBUG is noisy, INFO for normal ops, WARNING for degradation

Usage:
    from app.utils.logging import get_logger
    logger = get_logger(__name__)
    logger.info("Document ingested", filename="report.pdf", chunks=42)
"""

from __future__ import annotations

import logging
import sys

import structlog

from app.config import settings


def setup_logging() -> None:
    """
    Configure structlog with JSON rendering and standard library integration.

    Call once at application startup (called by setup_observability in monitoring.py).
    """
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer()
            if sys.stderr.isatty()
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Set root logger level from settings
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.app_log_level.upper(), logging.INFO),
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structlog logger for the given module name.

    Call this at module level to get a logger instance:
        logger = get_logger(__name__)
    """
    return structlog.stdlib.get_logger(name)
