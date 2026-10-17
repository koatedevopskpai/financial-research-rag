"""
OpenTelemetry setup for distributed tracing and metrics.

Configuration:
  - Traces are exported via OTLP if otel_exporter_otlp_endpoint is configured
  - Falls back to stdout tracing (console exporter) if no OTLP endpoint
  - FastAPI instrumentation is auto-configured for request/response tracing
"""

from __future__ import annotations

from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from app.config import settings
from app.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def setup_observability() -> None:
    """
    Initialise logging + OpenTelemetry tracing.

    Call once at application startup (from main.py lifespan handler).
    """
    # Always set up logging first
    setup_logging()

    # Configure OpenTelemetry resource — identifies this service in traces
    resource = Resource.create(
        {
            SERVICE_NAME: settings.otel_service_name,
            "service.version": "0.1.0",
            "deployment.environment": "reference",
        }
    )

    # Create tracer provider
    tracer_provider = TracerProvider(resource=resource)

    # Configure span exporter
    if settings.otel_exporter_otlp_endpoint:
        # Production: export to configured OTLP endpoint (e.g. Azure Monitor, Datadog)
        otlp_exporter = OTLPSpanExporter(
            endpoint=settings.otel_exporter_otlp_endpoint,
        )
        tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
        logger.info(
            "OTLP tracing configured",
            endpoint=settings.otel_exporter_otlp_endpoint,
        )
    else:
        # Development: print spans to console
        tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        logger.info("Console tracing configured (no OTLP endpoint)")

    # Set the global tracer provider
    trace.set_tracer_provider(tracer_provider)

    logger.info(
        "Observability initialised",
        service=settings.otel_service_name,
        log_level=settings.app_log_level,
    )


def instrument_fastapi(app: Any) -> None:
    """
    Instrument a FastAPI application for automatic request/response tracing.

    Call after the app is fully configured but before it starts serving:
        app = FastAPI()
        ...
        instrument_fastapi(app)
    """
    FastAPIInstrumentor.instrument_app(app)
    logger.info("FastAPI instrumentation applied")
