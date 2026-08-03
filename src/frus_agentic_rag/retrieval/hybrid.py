"""Hybrid retrieval: LanceDB BM25 + BGE-M3 dense, fused with RRF.

Both B0 and every agentic system call this same function with the same top_k,
which is what makes the ablation a comparison of the graph rather than of two
different retrievers.
"""

from __future__ import annotations

import asyncio
import re

from frus_agentic_rag.config import get_settings
from frus_agentic_rag.corpus.index import CHUNKS_TABLE, connect
from frus_agentic_rag.models import Evidence, SearchFilters

_COLS = [
    "chunk_id",
    "volume_id",
    "document_id",
    "doc_number",
    "subtype",
    "head",
    "date_from",
    "date_to",
    "text",
]

# LanceDB's SQL filter takes string literals; nothing user-authored reaches it
# unescaped, and the LLM only ever fills the typed SearchFilters fields.
_SAFE = re.compile(r"^[\w\-.:/ ]{0,120}$")


def _lit(value: str) -> str:
    if not _SAFE.match(value):
        raise ValueError(f"unsafe filter literal: {value!r}")
    return "'" + value.replace("'", "''") + "'"


def build_where(filters: SearchFilters) -> str | None:
    clauses: list[str] = []
    if filters.volume_ids:
        vals = ", ".join(_lit(v) for v in filters.volume_ids)
        clauses.append(f"volume_id IN ({vals})")
    if filters.subtypes:
        vals = ", ".join(_lit(s) for s in filters.subtypes)
        clauses.append(f"subtype IN ({vals})")
    if filters.date_from:
        clauses.append(f"date_to >= {_lit(filters.date_from)}")
    if filters.date_to:
        clauses.append(f"date_from <= {_lit(filters.date_to)}")
    return " AND ".join(clauses) if clauses else None


def _open():
    db = connect()
    if CHUNKS_TABLE not in db.table_names():
        raise FileNotFoundError("chunks table missing; run `frus ingest` then `frus index`")
    return db.open_table(CHUNKS_TABLE)


def _rows(query, limit: int, where: str | None) -> list[dict]:
    if where:
        query = query.where(where, prefilter=True)
    return query.select(_COLS).limit(limit).to_list()


def bm25_search(query: str, filters: SearchFilters, limit: int) -> list[dict]:
    tbl = _open()
    # Tantivy treats these as syntax; FRUS titles are full of them.
    cleaned = re.sub(r'[+\-!(){}\[\]^"~*?:\\/]', " ", query).strip()
    if not cleaned:
        return []
    return _rows(tbl.search(cleaned, query_type="fts"), limit, build_where(filters))


def dense_search(query: str, filters: SearchFilters, limit: int) -> list[dict]:
    from frus_agentic_rag.corpus.embed import encode

    tbl = _open()
    vec = encode([query])[0]
    where = build_where(filters)
    # Chunks whose vector was never filled would otherwise rank as noise.
    where = f"({where}) AND embedded = true" if where else "embedded = true"
    return _rows(tbl.search(vec, vector_column_name="vector"), limit, where)


def rrf_fuse(ranked_lists: list[list[dict]], k: int, top_k: int, hop: str = "") -> list[Evidence]:
    scores: dict[str, float] = {}
    payload: dict[str, dict] = {}
    ranks: dict[str, dict[str, int]] = {}

    for list_idx, rows in enumerate(ranked_lists):
        name = "bm25" if list_idx == 0 else "dense"
        for rank, row in enumerate(rows, start=1):
            cid = row["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            payload.setdefault(cid, row)
            ranks.setdefault(cid, {})[name] = rank

    ordered = sorted(scores, key=lambda c: scores[c], reverse=True)[:top_k]
    return [
        Evidence(
            evidence_id=cid,
            volume_id=payload[cid]["volume_id"],
            document_id=payload[cid]["document_id"],
            doc_number=payload[cid].get("doc_number", ""),
            subtype=payload[cid]["subtype"],
            head=payload[cid].get("head", ""),
            date_from=payload[cid].get("date_from", ""),
            date_to=payload[cid].get("date_to", ""),
            text=payload[cid]["text"],
            score=round(scores[cid], 6),
            rank_bm25=ranks[cid].get("bm25"),
            rank_dense=ranks[cid].get("dense"),
            hop=hop,
        )
        for cid in ordered
    ]


def hybrid_search_sync(
    query: str,
    filters: SearchFilters | None = None,
    top_k: int | None = None,
    hop: str = "",
) -> list[Evidence]:
    settings = get_settings()
    filters = filters or SearchFilters()
    top_k = top_k or settings.top_k
    cand = settings.candidate_k

    lexical = bm25_search(query, filters, cand)
    try:
        dense = dense_search(query, filters, cand)
    except Exception:
        # A corpus with BM25 but no vectors yet is a supported intermediate state.
        dense = []
    return rrf_fuse([lexical, dense], settings.rrf_k, top_k, hop=hop)


async def hybrid_search(
    query: str,
    filters: SearchFilters | None = None,
    top_k: int = 10,
    hop: str = "",
) -> list[Evidence]:
    return await asyncio.to_thread(hybrid_search_sync, query, filters, top_k, hop)
