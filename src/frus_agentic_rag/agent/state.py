"""Graph state and the budget arithmetic that guarantees termination."""

from __future__ import annotations

import time
from typing import Annotated, Any, Literal, TypedDict

from frus_agentic_rag import observability as obs
from frus_agentic_rag.agent.schemas import HopGrade, Language, Route, SubQuery
from frus_agentic_rag.models import Evidence


class TraceEvent(TypedDict, total=False):
    node: str
    started_at: float
    ended_at: float
    duration_s: float
    llm_calls: int
    retrieval_calls: int
    route: str
    detail: dict[str, Any]
    error: str


def _extend(a: list, b: list) -> list:
    return [*a, *b]


class AgentState(TypedDict, total=False):
    question: str
    answer_language: Language
    system: str  # B0 | B1 | B2 | B3 — which nodes are live

    route: Route
    needs_retrieval: bool
    subqueries: list[SubQuery]
    required_evidence: list[str]

    evidence: list[Evidence]
    accepted_evidence_ids: list[str]
    hop_grades: list[HopGrade]
    missing_hops: list[str]

    retrieval_rounds: int
    llm_calls: int
    corrections: int

    draft_answer: str | None
    claims: list[dict]
    limitations: str
    citation_errors: list[str]
    citations: list[str]

    outcome: Literal["answer", "abstain"] | None
    abstain_reason: str
    trace: Annotated[list[TraceEvent], _extend]


def new_state(question: str, language: Language, system: str = "B3") -> AgentState:
    return AgentState(
        question=question,
        answer_language=language,
        system=system,
        route="simple",
        needs_retrieval=True,
        subqueries=[],
        required_evidence=[],
        evidence=[],
        accepted_evidence_ids=[],
        hop_grades=[],
        missing_hops=[],
        retrieval_rounds=0,
        llm_calls=0,
        corrections=0,
        draft_answer=None,
        claims=[],
        limitations="",
        citation_errors=[],
        citations=[],
        outcome=None,
        abstain_reason="",
        trace=[],
    )


class NodeTimer:
    """Appends one TraceEvent per node, and opens the matching Phoenix span.

    The span wraps the whole node body, so LLM and retriever spans raised
    inside it nest underneath rather than floating at the root.
    """

    def __init__(self, node: str) -> None:
        self.node = node
        self.event: TraceEvent = {"node": node, "started_at": time.time()}
        self._span_cm = obs.span(f"node.{node}", kind="CHAIN")
        self._span: Any = None

    def __enter__(self) -> TraceEvent:
        self._span = self._span_cm.__enter__()
        return self.event

    def __exit__(self, exc_type, exc, tb) -> Literal[False]:
        self.event["ended_at"] = time.time()
        self.event["duration_s"] = round(self.event["ended_at"] - self.event["started_at"], 3)
        if exc is not None:
            self.event["error"] = f"{exc_type.__name__}: {exc}"
        if self._span is not None:
            obs.set_output(self._span, self.event.get("detail", {}))
            if self.event.get("error"):
                self._span.set_attribute("frus.node_error", self.event["error"][:500])
        self._span_cm.__exit__(exc_type, exc, tb)
        return False


def budget_left(state: AgentState, budgets) -> dict[str, int]:
    return {
        "llm_calls": budgets.max_llm_calls - state.get("llm_calls", 0),
        "retrieval_rounds": budgets.max_retrieval_rounds - state.get("retrieval_rounds", 0),
        "corrections": budgets.max_corrections - state.get("corrections", 0),
    }


def evidence_by_id(state: AgentState) -> dict[str, Evidence]:
    return {e.evidence_id: e for e in state.get("evidence", [])}
