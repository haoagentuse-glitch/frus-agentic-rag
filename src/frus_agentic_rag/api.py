"""FastAPI surface. Thin: the graph is the product, this just exposes it."""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from frus_agentic_rag import observability as obs
from frus_agentic_rag.agent.graph import SYSTEMS, mermaid
from frus_agentic_rag.config import get_settings
from frus_agentic_rag.models import Answer, SearchFilters

app = FastAPI(
    title="FRUS Bounded Agentic RAG",
    description=(
        "Question answering over the full published Foreign Relations of the United "
        "States corpus. Answers cite FRUS documents or abstain; there is no web "
        "fallback and no answering from model memory."
    ),
    version="0.1.0",
)


@app.on_event("startup")
async def _startup() -> None:
    obs.setup()


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    language: Literal["zh-TW", "en"] | None = None
    system: Literal["B0", "B1", "B2", "B3", "B0-2step"] = "B3"
    thread_id: str | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)
    volume_ids: list[str] = Field(default_factory=list)
    date_from: str | None = None
    date_to: str | None = None


@app.get("/health")
async def health() -> dict:
    from frus_agentic_rag.agent.llm import get_client
    from frus_agentic_rag.corpus.index import CHUNKS_TABLE, connect

    settings = get_settings()
    out: dict = {"status": "ok", "model": settings.ollama_model}
    try:
        out["ollama"] = await get_client().health()
    except Exception as exc:
        out["ollama"] = {"error": str(exc)}
        out["status"] = "degraded"
    try:
        tbl = connect().open_table(CHUNKS_TABLE)
        out["index"] = {
            "rows": tbl.count_rows(),
            "embedded": tbl.count_rows("embedded = true"),
        }
    except Exception as exc:
        out["index"] = {"error": str(exc)}
        out["status"] = "degraded"
    from frus_agentic_rag.retrieval import rerank as rr

    out["reranker"] = rr.warm()
    out["tracing"] = {"phoenix": obs.enabled()}
    return out


@app.get("/stats")
async def stats() -> dict:
    from frus_agentic_rag.corpus.index import corpus_stats

    return corpus_stats()


@app.get("/graph/{system}")
async def graph(system: str) -> dict:
    if system not in SYSTEMS:
        raise HTTPException(404, f"unknown system {system}; expected one of {SYSTEMS}")
    return {"system": system, "mermaid": mermaid(system)}


@app.post("/search")
async def search(req: SearchRequest) -> dict:
    from frus_agentic_rag.retrieval.hybrid import hybrid_search

    hits = await hybrid_search(
        req.query,
        SearchFilters(
            volume_ids=req.volume_ids,
            date_from=req.date_from,
            date_to=req.date_to,
            subtypes=["historical-document"],
        ),
        req.top_k,
    )
    return {"query": req.query, "hits": [h.model_dump() for h in hits]}


@app.post("/ask", response_model=Answer)
async def ask(req: AskRequest) -> Answer:
    from frus_agentic_rag.agent.run import answer, answer_2step

    try:
        if req.system == "B0-2step":
            return await answer_2step(req.question, language=req.language)
        return await answer(
            req.question,
            language=req.language,
            system=req.system,
            thread_id=req.thread_id,
        )
    except Exception as exc:
        raise HTTPException(500, f"{type(exc).__name__}: {exc}") from exc


@app.get("/series-status")
async def series_status(q: str) -> dict:
    """Whether a volume or period is published. Answered from the manifest."""
    from frus_agentic_rag.retrieval.tools import get_toolbox

    return await get_toolbox().get_series_status(q)
