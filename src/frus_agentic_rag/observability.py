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
from contextlib import contextmanager
from collections.abc import Iterator
from typing import Any

_TRACER: Any = None
_ENABLED: bool | None = None


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
    except Exception as exc:  # noqa: BLE001 — observability must never break the run
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
        span.set_attribute(key, json.dumps(value, ensure_ascii=False, default=str)[:8000])


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
            sp.set_attribute(f"{prefix}.{MessageAttributes.MESSAGE_CONTENT}", content[:12000])
        _set(sp, SpanAttributes.INPUT_VALUE, user[:12000])
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
    sp.set_attribute(f"{prefix}.{MessageAttributes.MESSAGE_CONTENT}", content[:12000])
    _set(sp, SpanAttributes.OUTPUT_VALUE, content[:12000])
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

    for i, e in enumerate(evidence[:20]):
        prefix = f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{i}"
        sp.set_attribute(f"{prefix}.{DocumentAttributes.DOCUMENT_ID}", e.evidence_id)
        sp.set_attribute(f"{prefix}.{DocumentAttributes.DOCUMENT_CONTENT}", e.text[:2000])
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
    except Exception:  # noqa: BLE001, S110 — never fail a run on telemetry
        pass
