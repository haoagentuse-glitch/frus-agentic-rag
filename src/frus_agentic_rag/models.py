"""Shared data types. These cross every layer, so they live in one place."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from frus_agentic_rag.config import FRUS_CANONICAL_BASE

SectionType = Literal["historical-document", "editorial-note"]


def canonical_url(volume_id: str, document_id: str | None = None) -> str:
    """Build the history.state.gov URL. Only code may produce these — never the LLM."""
    if document_id:
        return f"{FRUS_CANONICAL_BASE}/{volume_id}/{document_id}"
    return f"{FRUS_CANONICAL_BASE}/{volume_id}"


class VolumeMeta(BaseModel):
    volume_id: str
    title_complete: str = ""
    title_volume: str = ""
    sub_series: str = ""
    volume_number: str = ""
    publication_year: str = ""
    content_date_from: str = ""
    content_date_to: str = ""
    editors: list[str] = Field(default_factory=list)
    # "published" if the volume carries document divs, "planned" if it is a stub.
    status: Literal["published", "planned"] = "published"
    n_document_divs: int = 0
    n_historical_documents: int = 0
    sha256: str = ""
    size_bytes: int = 0


class FrusDocument(BaseModel):
    """One `div[@type='document']` — the canonical citable unit."""

    volume_id: str
    document_id: str  # xml:id, e.g. "d4"
    doc_number: str = ""  # @n
    subtype: SectionType
    head: str = ""
    date_from: str = ""  # frus:doc-dateTime-min
    date_to: str = ""
    date_display: str = ""
    persons: list[str] = Field(default_factory=list)
    text: str = ""
    source_note: str = ""

    @property
    def url(self) -> str:
        return canonical_url(self.volume_id, self.document_id)


class Chunk(BaseModel):
    chunk_id: str  # f"{volume_id}:{document_id}:{ordinal}" — stable across rebuilds
    volume_id: str
    document_id: str
    ordinal: int
    n_chunks_in_doc: int
    subtype: SectionType
    head: str = ""
    doc_number: str = ""
    date_from: str = ""
    date_to: str = ""
    persons: list[str] = Field(default_factory=list)
    text: str
    n_tokens: int = 0


class SearchFilters(BaseModel):
    """Structured, LLM-fillable filters. No free-form SQL ever reaches the store."""

    volume_ids: list[str] = Field(default_factory=list)
    date_from: str | None = None  # ISO yyyy-mm-dd
    date_to: str | None = None
    persons: list[str] = Field(default_factory=list)
    subtypes: list[SectionType] = Field(default_factory=list)


class Evidence(BaseModel):
    """A retrieved chunk plus provenance. Citation gate validates against these."""

    evidence_id: str  # == chunk_id; what the LLM is allowed to cite
    volume_id: str
    document_id: str
    doc_number: str = ""
    subtype: SectionType
    head: str = ""
    date_from: str = ""
    date_to: str = ""
    text: str
    score: float = 0.0
    rank_bm25: int | None = None
    rank_dense: int | None = None
    hop: str = ""  # which subquery surfaced it

    @property
    def url(self) -> str:
        return canonical_url(self.volume_id, self.document_id)

    def citation(self) -> str:
        num = f"Doc. {self.doc_number}" if self.doc_number else self.document_id
        return f"[{self.volume_id} {num}] {self.url}"


class Claim(BaseModel):
    text: str
    evidence_ids: list[str] = Field(default_factory=list)


class Answer(BaseModel):
    question: str
    language: Literal["zh-TW", "en"]
    outcome: Literal["answer", "abstain"]
    answer_text: str = ""
    claims: list[Claim] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)

    # The evidence pipeline, one field per stage. Without these, a single recall
    # number cannot say whether a miss came from the retriever, the grader or
    # the citation gate — `evidence` above is already filtered and truncated.
    retrieved_document_ids: list[str] = Field(
        default_factory=list, description="union over every retrieval round, untruncated"
    )
    accepted_document_ids: list[str] = Field(
        default_factory=list, description="what the grader accepted"
    )
    cited_document_ids: list[str] = Field(
        default_factory=list, description="what survived the citation gate"
    )

    limitations: str = ""
    abstain_reason: str = ""
    system: str = "B0"
    llm_calls: int = 0
    retrieval_calls: int = 0
    latency_s: float = 0.0
    trace: list[dict] = Field(default_factory=list)
