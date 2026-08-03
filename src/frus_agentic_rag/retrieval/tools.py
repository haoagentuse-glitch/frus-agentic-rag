"""The five tools the graph may call.

This is the whole surface the LLM can reach. It never emits SQL, a file path or
a URL: it fills typed arguments, and code turns those into store queries and
canonical citations.
"""

from __future__ import annotations

import asyncio
from typing import Protocol

from frus_agentic_rag import observability as obs
from frus_agentic_rag.config import get_settings
from frus_agentic_rag.retrieval.hybrid import _COLS, _open, hybrid_search
from frus_agentic_rag.retrieval.models import Evidence, SearchFilters, canonical_url


class Toolbox(Protocol):
    """What the graph depends on. The fake implementation satisfies this too."""

    async def hybrid_search(
        self, query: str, filters: SearchFilters, top_k: int = 10, hop: str = ""
    ) -> list[Evidence]: ...

    async def lookup_document(self, volume_id: str, document_id: str) -> list[Evidence]: ...

    async def timeline_search(
        self,
        query: str,
        date_from: str | None,
        date_to: str | None,
        filters: SearchFilters,
        top_k: int = 10,
        hop: str = "",
    ) -> list[Evidence]: ...

    async def get_adjacent_context(self, chunk_id: str) -> list[Evidence]: ...

    async def get_series_status(self, volume_or_period: str) -> dict: ...


def _to_evidence(rows: list[dict], hop: str = "") -> list[Evidence]:
    return [
        Evidence(
            evidence_id=r["chunk_id"],
            volume_id=r["volume_id"],
            document_id=r["document_id"],
            doc_number=r.get("doc_number", ""),
            subtype=r["subtype"],
            head=r.get("head", ""),
            date_from=r.get("date_from", ""),
            date_to=r.get("date_to", ""),
            text=r["text"],
            hop=hop,
        )
        for r in rows
    ]


class LiveToolbox:
    """Backed by the real LanceDB index."""

    async def hybrid_search(
        self, query: str, filters: SearchFilters, top_k: int = 10, hop: str = ""
    ) -> list[Evidence]:
        with obs.retriever_span(
            "hybrid_search",
            query,
            **{
                "frus.hop": hop,
                "frus.top_k": top_k,
                "frus.filters": filters.model_dump(exclude_defaults=True),
            },
        ) as sp:
            hits = await hybrid_search(query, filters, top_k, hop)
            obs.record_documents(sp, hits)
        return hits

    async def lookup_document(self, volume_id: str, document_id: str) -> list[Evidence]:
        def _run() -> list[Evidence]:
            tbl = _open()
            rows = (
                tbl.search()
                .where(
                    f"volume_id = '{volume_id}' AND document_id = '{document_id}'"
                    if _safe(volume_id) and _safe(document_id)
                    else "false"
                )
                .select(_COLS)
                .limit(64)
                .to_list()
            )
            rows.sort(key=lambda r: r["chunk_id"])
            return _to_evidence(rows, hop="lookup")

        return await asyncio.to_thread(_run)

    async def timeline_search(
        self,
        query: str,
        date_from: str | None,
        date_to: str | None,
        filters: SearchFilters,
        top_k: int = 10,
        hop: str = "",
    ) -> list[Evidence]:
        merged = filters.model_copy(
            update={
                "date_from": date_from or filters.date_from,
                "date_to": date_to or filters.date_to,
            }
        )
        with obs.retriever_span(
            "timeline_search",
            query,
            **{"frus.hop": hop, "frus.date_from": date_from, "frus.date_to": date_to},
        ) as sp:
            hits = await hybrid_search(query, merged, top_k, hop)
            hits.sort(key=lambda e: (e.date_from or "9999", e.evidence_id))
            obs.record_documents(sp, hits)
        return hits

    async def get_adjacent_context(self, chunk_id: str) -> list[Evidence]:
        """The chunks either side of one chunk, within the same document."""

        def _run() -> list[Evidence]:
            try:
                volume_id, document_id, ordinal = chunk_id.rsplit(":", 2)
                i = int(ordinal)
            except ValueError:
                return []
            if not (_safe(volume_id) and _safe(document_id)):
                return []
            tbl = _open()
            rows = (
                tbl.search()
                .where(f"volume_id = '{volume_id}' AND document_id = '{document_id}'")
                .select([*_COLS, "ordinal"])
                .limit(64)
                .to_list()
            )
            near = [r for r in rows if abs(r.get("ordinal", 0) - i) == 1]
            near.sort(key=lambda r: r.get("ordinal", 0))
            return _to_evidence(near, hop="adjacent")

        return await asyncio.to_thread(_run)

    async def get_series_status(self, volume_or_period: str) -> dict:
        """Answer 'is this published?' from the manifest, never from the model."""

        def _run() -> dict:
            from frus_agentic_rag.ingest.manifest import load_manifest

            rows = load_manifest().to_pylist()
            needle = volume_or_period.lower().strip()
            hits = [
                r
                for r in rows
                if needle in r["volume_id"].lower()
                or needle in (r["title_complete"] or "").lower()
                or needle in (r["sub_series"] or "").lower()
            ][:20]
            return {
                "query": volume_or_period,
                "matches": [
                    {
                        "volume_id": r["volume_id"],
                        "title": r["title_complete"],
                        "status": r["status"],
                        "documents": r["n_historical_documents"],
                        "content_dates": [r["content_date_from"], r["content_date_to"]],
                        "url": canonical_url(r["volume_id"]),
                    }
                    for r in hits
                ],
                "n_published_total": sum(1 for r in rows if r["status"] == "published"),
                "n_planned_total": sum(1 for r in rows if r["status"] == "planned"),
            }

        return await asyncio.to_thread(_run)


def _safe(v: str) -> bool:
    return bool(v) and all(c.isalnum() or c in "-._" for c in v)


_TOOLBOX: Toolbox | None = None


def get_toolbox() -> Toolbox:
    global _TOOLBOX
    if _TOOLBOX is None:
        _TOOLBOX = LiveToolbox()
    return _TOOLBOX


def set_toolbox(tb: Toolbox | None) -> None:
    """Injection point for the fake-tool walking skeleton and tests."""
    global _TOOLBOX
    _TOOLBOX = tb


def default_top_k() -> int:
    return get_settings().top_k
