"""
One instrumentation call, two destinations.

This module is the "no lock-in" argument as executable code. The agent below
emits OpenInference spans over OpenTelemetry — an open semantic convention, not
a vendor SDK. Where those spans land is an environment variable:

    (unset)                      → local Phoenix at http://localhost:6006
    PHOENIX_COLLECTOR_ENDPOINT   → self-hosted or Phoenix Cloud
    ARIZE_SPACE_ID + ARIZE_API_KEY → Arize AX

Nothing in agent.py, evaluators.py, or the scripts changes between those three.
That is the whole point: dev-time tracing and production observability run on
one instrumentation, so "start on the OSS one and graduate" is a configuration
change rather than a migration.

The span *kinds* matter as much as the destination. OpenInference types spans as
LLM / TOOL / AGENT / CHAIN / RETRIEVER, and the evaluators in this repo filter on
those kinds when they read a trajectory back out of the span tree. Getting the
kind right is not cosmetic — it is the join key between instrumentation and
evaluation.
"""

from __future__ import annotations

import os

from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry.sdk.trace import TracerProvider

_PROVIDER: TracerProvider | None = None

DEFAULT_PROJECT = "jwst-research-agent"


def destination() -> str:
    """Which backend the current environment points at. Printed by every script
    so a demo never leaves the audience guessing where the traces went."""
    if os.getenv("ARIZE_SPACE_ID") and os.getenv("ARIZE_API_KEY"):
        return "arize-ax"
    if os.getenv("PHOENIX_COLLECTOR_ENDPOINT"):
        return f"phoenix ({os.environ['PHOENIX_COLLECTOR_ENDPOINT']})"
    return "phoenix (http://localhost:6006)"


def setup_tracing(project_name: str | None = None) -> TracerProvider:
    """Register the tracer provider and auto-instrument LangChain/LangGraph.

    Idempotent: calling it twice returns the first provider rather than
    registering a second exporter. Double registration is the classic way to
    double every token count in a trace, and a cost number that is wrong in the
    flattering direction is the worst kind of wrong.
    """
    global _PROVIDER
    if _PROVIDER is not None:
        return _PROVIDER

    project = project_name or os.getenv("PHOENIX_PROJECT_NAME", DEFAULT_PROJECT)

    if os.getenv("ARIZE_SPACE_ID") and os.getenv("ARIZE_API_KEY"):
        # Arize AX. Same OTLP protocol, same OpenInference spans, different
        # endpoint and auth headers.
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.resources import Resource

        provider = TracerProvider(
            resource=Resource.create(
                {
                    "model_id": project,
                    "model_version": os.getenv("ARIZE_MODEL_VERSION", "1.0.0"),
                }
            )
        )
        provider.add_span_processor(
            BatchSpanProcessor(
                OTLPSpanExporter(
                    endpoint=os.getenv("ARIZE_ENDPOINT", "https://otlp.arize.com/v1"),
                    headers={
                        "space_id": os.environ["ARIZE_SPACE_ID"],
                        "api_key": os.environ["ARIZE_API_KEY"],
                    },
                )
            )
        )
    else:
        from phoenix.otel import register

        provider = register(
            project_name=project,
            endpoint=os.getenv("PHOENIX_COLLECTOR_ENDPOINT"),
            batch=True,
            verbose=False,
        )

    LangChainInstrumentor().instrument(tracer_provider=provider)
    _PROVIDER = provider
    return provider


def flush() -> None:
    """Force-export before the process exits.

    Batch processors drop whatever is still buffered at exit. A script that ends
    by printing "200 traces sent" and then loses forty of them is worse than one
    that never claimed a number.
    """
    if _PROVIDER is not None:
        _PROVIDER.force_flush()
