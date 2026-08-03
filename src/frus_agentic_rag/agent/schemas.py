"""Structured-output contracts for the planner, grader and synthesizer.

Every LLM boundary in this system is a Pydantic schema handed to Ollama as a
JSON schema. Nothing here is parsed out of prose with a regex.

Every list carries a max_length. That is load-bearing, not tidiness: Pydantic
emits it as `maxItems`, and without it the grammar-constrained decoder happily
emits array elements until the read timeout — measured at 14k tokens on a
single grade call before this was added.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Route = Literal["lookup", "simple", "timeline", "complex", "status"]
Language = Literal["zh-TW", "en"]
HopVerdict = Literal["supported", "partial", "unsupported"]


class SubQuery(BaseModel):
    """One retrieval hop. Always English: the corpus is English."""

    # Sanitised in nodes.plan_query: the model likes to answer with paths.
    hop_id: str = Field(default="", description="short slug, e.g. 'nixon-visit-date'")
    query: str = Field(description="English retrieval query")
    date_from: str | None = Field(default=None, description="ISO yyyy-mm-dd or null")
    date_to: str | None = Field(default=None, description="ISO yyyy-mm-dd or null")
    persons: list[str] = Field(default_factory=list, max_length=5)
    volume_ids: list[str] = Field(default_factory=list, max_length=5)


class QueryPlan(BaseModel):
    route: Route
    answer_language: Language
    needs_retrieval: bool = True
    subqueries: list[SubQuery] = Field(default_factory=list, max_length=3)
    required_evidence: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="What must be shown by evidence for the answer to stand",
    )


class HopGrade(BaseModel):
    hop_id: str = ""
    verdict: HopVerdict
    accepted_evidence_ids: list[str] = Field(default_factory=list, max_length=8)
    # No free-form "missing" field. Given one, the model writes a paragraph of
    # reasoning and blows through num_predict; the corrective query carries the
    # same information in a form the retriever can actually use.
    corrective_query: str = Field(
        default="", description="one short English retrieval query, under 20 words"
    )


class EvidenceGrade(BaseModel):
    hops: list[HopGrade] = Field(default_factory=list, max_length=3)
    overall: HopVerdict


class DraftClaim(BaseModel):
    text: str
    evidence_ids: list[str] = Field(default_factory=list, max_length=5)


class AgentAnswer(BaseModel):
    # No length bounds on the strings here. Ollama's schema-to-grammar
    # converter returns "failed to parse grammar" (400) for `maxLength`, and
    # rejects `minLength`/`minItems` too — bisected against a live server.
    # String length is bounded by num_predict instead; "you must cite
    # something" is enforced by the prompt and by the citation gate.
    answer_text: str
    claims: list[DraftClaim] = Field(max_length=8)
    limitations: str = ""
