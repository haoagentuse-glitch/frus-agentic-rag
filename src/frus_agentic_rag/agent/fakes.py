"""Injectable fakes for the walking skeleton and unit tests.

These exist so graph termination can be proven without LanceDB, BGE-M3 or
Ollama running — and so the four paths (simple, multi-hop, one correction,
unanswerable) can be forced deterministically rather than hoped for.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from frus_agentic_rag.agent.schemas import AgentAnswer, EvidenceGrade, HopGrade, QueryPlan, SubQuery
from frus_agentic_rag.models import Evidence, SearchFilters


def fake_evidence(n: int, prefix: str = "frus1969-76v17", hop: str = "main") -> list[Evidence]:
    return [
        Evidence(
            evidence_id=f"{prefix}:d{i + 1}:0",
            volume_id=prefix,
            document_id=f"d{i + 1}",
            doc_number=str(i + 1),
            subtype="historical-document",
            head=f"Fake document {i + 1}",
            date_from="1972-02-21",
            date_to="1972-02-21",
            text=f"Body of fake document {i + 1}.",
            score=1.0 - i * 0.01,
            rank_bm25=i + 1,
            hop=hop,
        )
        for i in range(n)
    ]


@dataclass
class FakeToolbox:
    """Returns evidence unless the query is marked unanswerable."""

    n_hits: int = 5
    empty_for: tuple[str, ...] = ("UNANSWERABLE",)
    calls: list[tuple[str, str]] = field(default_factory=list)
    # hop_id -> whether the FIRST attempt returns nothing (forces a correction)
    fail_first: set[str] = field(default_factory=set)
    _seen: set[str] = field(default_factory=set)

    def _empty(self, query: str, hop: str) -> bool:
        if any(m in query for m in self.empty_for):
            return True
        if hop in self.fail_first and hop not in self._seen:
            self._seen.add(hop)
            return True
        return False

    async def hybrid_search(
        self, query: str, filters: SearchFilters, top_k: int = 10, hop: str = ""
    ) -> list[Evidence]:
        self.calls.append(("hybrid_search", query))
        if self._empty(query, hop):
            return []
        return fake_evidence(min(self.n_hits, top_k), hop=hop or "main")

    async def lookup_document(self, volume_id: str, document_id: str) -> list[Evidence]:
        self.calls.append(("lookup_document", f"{volume_id}/{document_id}"))
        return fake_evidence(1, prefix=volume_id, hop="lookup")

    async def timeline_search(
        self, query: str, date_from, date_to, filters: SearchFilters, top_k: int = 10, hop: str = ""
    ) -> list[Evidence]:
        self.calls.append(("timeline_search", query))
        if self._empty(query, hop):
            return []
        return fake_evidence(min(self.n_hits, top_k), hop=hop or "timeline")

    async def get_adjacent_context(self, chunk_id: str) -> list[Evidence]:
        self.calls.append(("get_adjacent_context", chunk_id))
        return fake_evidence(2)

    async def get_series_status(self, volume_or_period: str) -> dict:
        self.calls.append(("get_series_status", volume_or_period))
        return {
            "query": volume_or_period,
            "matches": [
                {
                    "volume_id": "frus1969-76v17",
                    "title": "China, 1969-1972",
                    "status": "published",
                    "documents": 268,
                    "content_dates": ["1969-01-25", "1972-12-16"],
                    "url": "https://history.state.gov/historicaldocuments/frus1969-76v17",
                }
            ],
            "n_published_total": 552,
            "n_planned_total": 142,
        }


@dataclass
class FakeClient:
    """Stands in for OllamaClient. Scripted, never networked."""

    route: str = "simple"
    n_subqueries: int = 1
    grade_sequence: list[str] = field(default_factory=lambda: ["supported"])
    claim_ids: list[str] | None = None
    calls: int = 0
    _grade_i: int = 0
    seen: list[str] = field(default_factory=list)

    async def structured(self, system: str, user: str, schema, num_predict=None):
        self.calls += 1
        if schema is QueryPlan:
            self.seen.append("plan")
            return QueryPlan(
                route=self.route,  # type: ignore[arg-type]
                answer_language="en",
                needs_retrieval=True,
                subqueries=[
                    SubQuery(hop_id=f"h{i}", query=f"hop {i} query")
                    for i in range(self.n_subqueries)
                ],
                required_evidence=[f"fact {i}" for i in range(self.n_subqueries)],
            )
        if schema is EvidenceGrade:
            self.seen.append("grade")
            verdict = self.grade_sequence[min(self._grade_i, len(self.grade_sequence) - 1)]
            self._grade_i += 1
            has_evidence = "no evidence retrieved" not in user
            # Inverted grader: an unsupported hop rejects what it saw, a
            # supported one rejects nothing.
            ids = [] if (verdict == "supported" or not has_evidence) else ["frus1969-76v17:d1:0"]
            return EvidenceGrade(
                hops=[
                    HopGrade(
                        hop_id="h0",
                        verdict=verdict,  # type: ignore[arg-type]
                        rejected_evidence_ids=ids,
                        corrective_query="" if verdict == "supported" else "corrected query",
                    )
                ],
                overall=verdict,  # type: ignore[arg-type]
            )
        if schema is AgentAnswer:
            self.seen.append("synth")
            ids = self.claim_ids if self.claim_ids is not None else ["frus1969-76v17:d1:0"]
            return AgentAnswer(
                answer_text="A fake grounded answer.",
                claims=[{"text": "A fake claim.", "evidence_ids": ids}],  # type: ignore[list-item]
                limitations="This is a fake.",
            )
        raise AssertionError(f"unexpected schema {schema}")

    async def generate(self, system: str, user: str) -> str:
        self.calls += 1
        return "fake"
