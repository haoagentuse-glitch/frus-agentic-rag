"""Structured-output contracts for the planner, grader and synthesizer.

Every LLM boundary in this system is a Pydantic schema handed to Ollama as a
JSON schema. Nothing here is parsed out of prose with a regex.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Route = Literal["lookup", "simple", "timeline", "complex", "status"]
Language = Literal["zh-TW", "en"]
HopVerdict = Literal["supported", "partial", "unsupported"]


class SubQuery(BaseModel):
    """One retrieval hop. Always English: the corpus is English."""

    hop_id: str = Field(description="short slug, e.g. 'nixon-visit-date'")
    query: str = Field(description="English retrieval query")
    date_from: str | None = Field(default=None, description="ISO yyyy-mm-dd or null")
    date_to: str | None = Field(default=None, description="ISO yyyy-mm-dd or null")
    persons: list[str] = Field(default_factory=list)
    volume_ids: list[str] = Field(default_factory=list)


class QueryPlan(BaseModel):
    route: Route
    answer_language: Language
    needs_retrieval: bool = True
    subqueries: list[SubQuery] = Field(default_factory=list, max_length=3)
    required_evidence: list[str] = Field(
        default_factory=list,
        description="What must be shown by evidence for the answer to stand",
    )


class HopGrade(BaseModel):
    hop_id: str
    verdict: HopVerdict
    accepted_evidence_ids: list[str] = Field(default_factory=list)
    missing: str = ""
    corrective_query: str = ""


class EvidenceGrade(BaseModel):
    hops: list[HopGrade] = Field(default_factory=list)
    overall: HopVerdict


class DraftClaim(BaseModel):
    text: str
    evidence_ids: list[str] = Field(default_factory=list)


class AgentAnswer(BaseModel):
    answer_text: str
    claims: list[DraftClaim] = Field(default_factory=list)
    limitations: str = ""
