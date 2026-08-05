"""Phoenix / OpenTelemetry tracing.

The LLM calls here go straight to Ollama over httpx rather than through a
LangChain chat model, so no auto-instrumentor sees them. Spans are therefore
emitted by hand, following OpenInference semantic conventions so Phoenix
renders them as proper LLM and RETRIEVER spans instead of anonymous blocks.

Tracing is entirely optional: with PHOENIX_COLLECTOR_ENDPOINT unset every
helper here degrades to a no-op context manager and the graph runs unchanged.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_TRACER: Any = None
_ENABLED: bool | None = None

# How much of each payload is attached to a span. These were 12000 / 2000 / 20,
# which put roughly 200KB on the wire per question: four LLM calls carrying
# prompt and completion in full, plus twenty passages per retrieval. Phoenix's
# ingest could not keep up and its exporter began timing out at ten seconds a
# batch, several times a run — one B3 question took 789s traced and 42s with
# tracing off. The trace is for reading, not for storing the corpus a second
# time, so the defaults now keep the shape of every payload and the beginning of
# its content. Raise them by environment variable when a specific prompt has to
# be read back in full.
_LLM_CHARS = int(os.getenv("PHOENIX_MAX_LLM_CHARS", "4000"))
_DOC_CHARS = int(os.getenv("PHOENIX_MAX_DOC_CHARS", "500"))
_MAX_DOCS = int(os.getenv("PHOENIX_MAX_DOCUMENTS", "5"))
_JSON_CHARS = int(os.getenv("PHOENIX_MAX_JSON_CHARS", "2000"))


def _truthy(v: str | None) -> bool:
    return (v or "").strip().lower() in {"1", "true", "yes", "on"}


def enabled() -> bool:
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = bool(os.getenv("PHOENIX_COLLECTOR_ENDPOINT")) and not _truthy(
            os.getenv("PHOENIX_DISABLED")
        )
    return _ENABLED


def setup(project_name: str | None = None) -> Any:
    """Register the Phoenix tracer provider. Safe to call more than once."""
    global _TRACER
    if _TRACER is not None or not enabled():
        return _TRACER
    try:
        from phoenix.otel import register

        # `register` treats a bare host as a base URL inconsistently across
        # versions; pointing at the OTLP HTTP path directly avoids the 405.
        endpoint = os.environ["PHOENIX_COLLECTOR_ENDPOINT"].rstrip("/")
        if not endpoint.endswith("/v1/traces"):
            endpoint = f"{endpoint}/v1/traces"

        provider = register(
            project_name=project_name or os.getenv("PHOENIX_PROJECT", "frus-agentic-rag"),
            endpoint=endpoint,
            auto_instrument=False,
            batch=True,
            set_global_tracer_provider=True,
        )
        _TRACER = provider.get_tracer("frus_agentic_rag")
    except Exception as exc:
        print(f"[observability] Phoenix disabled: {type(exc).__name__}: {exc}", flush=True)
        _TRACER = None
    return _TRACER


def _tracer() -> Any:
    if _TRACER is None and enabled():
        setup()
    return _TRACER


def _set(span: Any, key: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, (str, bool, int, float)):
        span.set_attribute(key, value)
    else:
        span.set_attribute(key, json.dumps(value, ensure_ascii=False, default=str)[:_JSON_CHARS])


@contextmanager
def span(name: str, kind: str = "CHAIN", **attributes: Any) -> Iterator[Any]:
    """One OpenInference span. A no-op when Phoenix is not configured."""
    tracer = _tracer()
    if tracer is None:
        yield None
        return

    from openinference.semconv.trace import SpanAttributes

    with tracer.start_as_current_span(name) as sp:
        sp.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, kind)
        for k, v in attributes.items():
            _set(sp, k, v)
        try:
            yield sp
        except Exception as exc:
            sp.record_exception(exc)
            raise


def set_input(sp: Any, value: Any) -> None:
    if sp is None:
        return
    from openinference.semconv.trace import SpanAttributes

    _set(sp, SpanAttributes.INPUT_VALUE, value)


def set_output(sp: Any, value: Any) -> None:
    if sp is None:
        return
    from openinference.semconv.trace import SpanAttributes

    _set(sp, SpanAttributes.OUTPUT_VALUE, value)


@contextmanager
def llm_span(model: str, system: str, user: str, schema: str | None = None) -> Iterator[Any]:
    """LLM span carrying both messages so Phoenix shows the real prompt."""
    tracer = _tracer()
    if tracer is None:
        yield None
        return

    from openinference.semconv.trace import (
        MessageAttributes,
        SpanAttributes,
    )

    name = f"ollama.{schema}" if schema else "ollama.generate"
    with tracer.start_as_current_span(name) as sp:
        sp.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
        sp.set_attribute(SpanAttributes.LLM_MODEL_NAME, model)
        sp.set_attribute(SpanAttributes.LLM_PROVIDER, "ollama")
        sp.set_attribute(SpanAttributes.LLM_SYSTEM, "ollama")
        for i, (role, content) in enumerate([("system", system), ("user", user)]):
            prefix = f"{SpanAttributes.LLM_INPUT_MESSAGES}.{i}"
            sp.set_attribute(f"{prefix}.{MessageAttributes.MESSAGE_ROLE}", role)
            sp.set_attribute(f"{prefix}.{MessageAttributes.MESSAGE_CONTENT}", content[:_LLM_CHARS])
        _set(sp, SpanAttributes.INPUT_VALUE, user[:_LLM_CHARS])
        if schema:
            _set(sp, SpanAttributes.LLM_INVOCATION_PARAMETERS, {"format": schema})
        try:
            yield sp
        except Exception as exc:
            sp.record_exception(exc)
            raise


def record_llm_output(sp: Any, content: str, usage: dict | None = None) -> None:
    if sp is None:
        return
    from openinference.semconv.trace import MessageAttributes, SpanAttributes

    prefix = f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.0"
    sp.set_attribute(f"{prefix}.{MessageAttributes.MESSAGE_ROLE}", "assistant")
    sp.set_attribute(f"{prefix}.{MessageAttributes.MESSAGE_CONTENT}", content[:_LLM_CHARS])
    _set(sp, SpanAttributes.OUTPUT_VALUE, content[:_LLM_CHARS])
    if usage:
        if (v := usage.get("prompt_eval_count")) is not None:
            sp.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, int(v))
        if (v := usage.get("eval_count")) is not None:
            sp.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, int(v))
        total = (usage.get("prompt_eval_count") or 0) + (usage.get("eval_count") or 0)
        if total:
            sp.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, int(total))


@contextmanager
def retriever_span(tool: str, query: str, **attributes: Any) -> Iterator[Any]:
    tracer = _tracer()
    if tracer is None:
        yield None
        return

    from openinference.semconv.trace import SpanAttributes

    with tracer.start_as_current_span(f"retrieve.{tool}") as sp:
        sp.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "RETRIEVER")
        _set(sp, SpanAttributes.INPUT_VALUE, query)
        for k, v in attributes.items():
            _set(sp, k, v)
        try:
            yield sp
        except Exception as exc:
            sp.record_exception(exc)
            raise


def record_documents(sp: Any, evidence: list) -> None:
    """Attach retrieved chunks so Phoenix's retrieval views work."""
    if sp is None:
        return
    from openinference.semconv.trace import DocumentAttributes, SpanAttributes

    for i, e in enumerate(evidence[:_MAX_DOCS]):
        prefix = f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{i}"
        sp.set_attribute(f"{prefix}.{DocumentAttributes.DOCUMENT_ID}", e.evidence_id)
        sp.set_attribute(f"{prefix}.{DocumentAttributes.DOCUMENT_CONTENT}", e.text[:_DOC_CHARS])
        sp.set_attribute(f"{prefix}.{DocumentAttributes.DOCUMENT_SCORE}", float(e.score))
        sp.set_attribute(
            f"{prefix}.{DocumentAttributes.DOCUMENT_METADATA}",
            json.dumps(
                {
                    "volume_id": e.volume_id,
                    "document_id": e.document_id,
                    "date_from": e.date_from,
                    "hop": e.hop,
                    "url": e.url,
                },
                ensure_ascii=False,
            ),
        )
    sp.set_attribute("retrieval.document_count", len(evidence))


def flush() -> None:
    """Force-export before a short-lived CLI process exits."""
    if _TRACER is None:
        return
    try:
        from opentelemetry import trace

        provider = trace.get_tracer_provider()
        if hasattr(provider, "force_flush"):
            provider.force_flush(5000)
    except Exception:
        pass
