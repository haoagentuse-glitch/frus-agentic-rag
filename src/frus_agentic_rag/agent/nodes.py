"""Graph nodes. Each returns a state delta and appends exactly one trace event."""

from __future__ import annotations

import asyncio
import re

from frus_agentic_rag.agent.prompts import (
    ABSTAIN_EN,
    ABSTAIN_ZH,
    GRADER_SYSTEM,
    PLANNER_SYSTEM,
    SYNTH_SYSTEM_EN,
    SYNTH_SYSTEM_ZH,
    grader_user,
    planner_user,
    synth_user,
)
from frus_agentic_rag.agent.schemas import (
    AgentAnswer,
    EvidenceGrade,
    HopGrade,
    QueryPlan,
    SubQuery,
)
from frus_agentic_rag.agent.state import AgentState, NodeTimer
from frus_agentic_rag.config import get_settings
from frus_agentic_rag.generation import citations as cite
from frus_agentic_rag.generation.ollama_client import StructuredOutputError, get_client
from frus_agentic_rag.retrieval.models import Claim, SearchFilters
from frus_agentic_rag.retrieval.tools import get_toolbox

_CJK = re.compile(r"[一-鿿]")


def detect_language(question: str) -> str:
    return "zh-TW" if _CJK.search(question) else "en"


# --- rule-first router ----------------------------------------------------

_STATUS_PAT = re.compile(
    r"(published|not yet published|planned|forthcoming|出版|已出版|尚未出版|規劃中|哪一卷)",
    re.I,
)
_TIMELINE_PAT = re.compile(
    r"(timeline|chronolog|sequence|順序|時間軸|經過|演變|什麼時候|何時|when did)", re.I
)
_LOOKUP_PAT = re.compile(r"(frus\d[\w\-]*|document\s+\d+|第\s*\d+\s*號文件)", re.I)
_COMPLEX_PAT = re.compile(
    r"(compare|difference|differ|versus|why did|cause|led to|比較|差異|為什麼|原因|導致|影響)",
    re.I,
)


def rule_route(question: str) -> str:
    """Deterministic route. Used as a prior and as the fallback when the LLM fails."""
    if _STATUS_PAT.search(question):
        return "status"
    if _LOOKUP_PAT.search(question):
        return "lookup"
    if _COMPLEX_PAT.search(question):
        return "complex"
    if _TIMELINE_PAT.search(question):
        return "timeline"
    return "simple"


# --- nodes ----------------------------------------------------------------


async def plan_query(state: AgentState) -> dict:
    """B1+: decompose. B0 skips this node entirely."""
    with NodeTimer("plan_query") as ev:
        question = state["question"]
        language = state.get("answer_language") or detect_language(question)
        fallback_route = rule_route(question)

        if state.get("system") == "B0":
            ev["detail"] = {"route": fallback_route, "planner": "skipped (B0)"}
            return {
                "route": fallback_route,
                "answer_language": language,
                "subqueries": [SubQuery(hop_id="main", query=question)],
                "required_evidence": [],
                "needs_retrieval": True,
                "trace": [ev],
            }

        client = get_client()
        try:
            plan = await client.structured(PLANNER_SYSTEM, planner_user(question), QueryPlan)
            used = client.calls
        except (StructuredOutputError, Exception) as exc:  # noqa: BLE001
            # Gate 2 of the kill-test: a broken planner must not break the run.
            ev["error"] = f"planner fallback: {type(exc).__name__}: {exc}"
            ev["detail"] = {"route": fallback_route, "planner": "rule-first fallback"}
            return {
                "route": fallback_route,
                "answer_language": language,
                "subqueries": [SubQuery(hop_id="main", query=question)],
                "needs_retrieval": True,
                "llm_calls": state.get("llm_calls", 0) + 1,
                "trace": [ev],
            }

        budgets = get_settings().budgets
        subs = plan.subqueries[: budgets.max_subqueries]
        if plan.needs_retrieval and not subs:
            subs = [SubQuery(hop_id="main", query=question)]
        # The model returns things like "/FRUS/1862/01" here; hop ids are used
        # as dict keys and trace labels, so normalise them to positional slugs.
        subs = [s.model_copy(update={"hop_id": f"h{i}"}) for i, s in enumerate(subs)]

        ev["route"] = plan.route
        ev["llm_calls"] = 1
        ev["detail"] = {
            "route": plan.route,
            "rule_route": fallback_route,
            "subqueries": [s.query for s in subs],
            "required_evidence": plan.required_evidence,
        }
        return {
            "route": plan.route,
            "answer_language": plan.answer_language or language,
            "needs_retrieval": plan.needs_retrieval,
            "subqueries": subs,
            "required_evidence": plan.required_evidence,
            "llm_calls": used,
            "trace": [ev],
        }


def _filters(sq: SubQuery) -> SearchFilters:
    return SearchFilters(
        volume_ids=sq.volume_ids,
        date_from=sq.date_from,
        date_to=sq.date_to,
        persons=sq.persons,
        subtypes=["historical-document"],
    )


async def dispatch_retrieval(state: AgentState) -> dict:
    """Route to a tool; complex plans fan their subqueries out concurrently."""
    with NodeTimer("dispatch_retrieval") as ev:
        tb = get_toolbox()
        settings = get_settings()
        route = state.get("route", "simple")
        subs = state.get("subqueries") or [SubQuery(hop_id="main", query=state["question"])]
        top_k = settings.top_k

        if route == "status":
            status = await tb.get_series_status(state["question"])
            ev["retrieval_calls"] = 1
            ev["detail"] = {"tool": "get_series_status", "matches": len(status["matches"])}
            return {
                "evidence": state.get("evidence", []),
                "retrieval_rounds": state.get("retrieval_rounds", 0) + 1,
                "series_status": status,
                "trace": [ev],
            }

        async def run(sq: SubQuery) -> list:
            if route == "lookup":
                m = re.search(r"(frus[\w\-]+)\D+(\d+)", sq.query, re.I)
                if m:
                    hits = await tb.lookup_document(m.group(1), f"d{m.group(2)}")
                    if hits:
                        return hits
            if route == "timeline":
                return await tb.timeline_search(
                    sq.query, sq.date_from, sq.date_to, _filters(sq), top_k, sq.hop_id
                )
            return await tb.hybrid_search(sq.query, _filters(sq), top_k, sq.hop_id)

        results = await asyncio.gather(*(run(s) for s in subs), return_exceptions=True)
        fresh = []
        errors = []
        for sq, r in zip(subs, results, strict=True):
            if isinstance(r, BaseException):
                errors.append(f"{sq.hop_id}: {type(r).__name__}: {r}")
            else:
                fresh.extend(r)

        ev["retrieval_calls"] = len(subs)
        ev["route"] = route
        ev["detail"] = {
            "tool": {"timeline": "timeline_search", "lookup": "lookup_document"}.get(
                route, "hybrid_search"
            ),
            "hops": [s.hop_id for s in subs],
            "hits": len(fresh),
            "errors": errors,
        }
        return {
            "evidence": _merge(state.get("evidence", []), fresh),
            "retrieval_rounds": state.get("retrieval_rounds", 0) + 1,
            "trace": [ev],
        }


def _merge(existing: list, fresh: list) -> list:
    """Dedupe by chunk id, then interleave hops round-robin.

    Sorting the pooled list by score would be wrong: RRF scores from different
    subqueries are not on a comparable scale, so one strong hop can take every
    slot the grader sees and a genuine multi-hop question then grades as
    unsupported on a hop that was actually retrieved.
    """
    by_id: dict[str, object] = {}
    for e in [*existing, *fresh]:
        prev = by_id.get(e.evidence_id)
        if prev is None:
            by_id[e.evidence_id] = e
        elif e.score > prev.score:  # type: ignore[attr-defined]
            by_id[e.evidence_id] = e.model_copy(update={"hop": prev.hop or e.hop})  # type: ignore[attr-defined]

    per_hop: dict[str, list] = {}
    for e in sorted(by_id.values(), key=lambda x: x.score, reverse=True):  # type: ignore[attr-defined]
        per_hop.setdefault(e.hop or "main", []).append(e)  # type: ignore[attr-defined]

    ordered: list = []
    queues = list(per_hop.values())
    i = 0
    while any(queues):
        q = queues[i % len(queues)]
        if q:
            ordered.append(q.pop(0))
        i += 1
        if i > 10_000:  # defensive: never spin on a malformed queue set
            break
    return ordered


async def grade_evidence(state: AgentState) -> dict:
    """B2+: does the accepted evidence cover the required hops?"""
    with NodeTimer("grade_evidence") as ev:
        evidence = state.get("evidence", [])
        if not evidence:
            ev["detail"] = {"overall": "unsupported", "reason": "no evidence"}
            return {
                "hop_grades": [],
                "accepted_evidence_ids": [],
                "missing_hops": [s.hop_id for s in state.get("subqueries", [])],
                "trace": [ev],
            }

        if state.get("system") in ("B0", "B1"):
            ev["detail"] = {"grader": "skipped", "accepted": len(evidence)}
            return {
                "accepted_evidence_ids": [e.evidence_id for e in evidence],
                "hop_grades": [],
                "missing_hops": [],
                "trace": [ev],
            }

        hops = state.get("required_evidence") or [s.query for s in state.get("subqueries", [])]
        client = get_client()
        try:
            grade = await client.structured(
                GRADER_SYSTEM,
                grader_user(state["question"], hops, evidence[: get_settings().top_k]),
                EvidenceGrade,
            )
        except Exception as exc:  # noqa: BLE001
            # A failed grader must not fabricate support; treat as partial and
            # let the retrieval-round budget end the run.
            ev["error"] = f"grader fallback: {type(exc).__name__}: {exc}"
            ev["detail"] = {"overall": "partial", "grader": "fallback accept-top-k"}
            return {
                "accepted_evidence_ids": [e.evidence_id for e in evidence[:5]],
                "hop_grades": [HopGrade(hop_id="all", verdict="partial")],
                "missing_hops": [],
                "llm_calls": state.get("llm_calls", 0) + 1,
                "trace": [ev],
            }

        valid = {e.evidence_id for e in evidence}
        accepted: list[str] = []
        missing: list[str] = []
        hallucinated: list[str] = []
        for hop in grade.hops:
            good = [i for i in hop.accepted_evidence_ids if i in valid]
            hallucinated.extend(i for i in hop.accepted_evidence_ids if i not in valid)
            accepted.extend(good)
            if hop.verdict != "supported":
                missing.append(hop.corrective_query or hop.hop_id)

        ev["llm_calls"] = 1
        ev["detail"] = {
            "overall": grade.overall,
            "hops": [h.model_dump() for h in grade.hops],
            "accepted": len(set(accepted)),
            "hallucinated_ids": hallucinated,
        }
        return {
            "hop_grades": grade.hops,
            "accepted_evidence_ids": sorted(set(accepted)),
            "missing_hops": missing,
            "llm_calls": client.calls,
            "trace": [ev],
        }


async def rewrite_missing(state: AgentState) -> dict:
    """Re-query only the uncovered hops. Runs at most once."""
    with NodeTimer("rewrite_missing") as ev:
        missing = state.get("missing_hops", [])[: get_settings().budgets.max_subqueries]
        subs = [
            SubQuery(hop_id=f"fix{i}", query=q)
            for i, q in enumerate(missing)
            if q and q.strip()
        ]
        ev["detail"] = {"rewritten": [s.query for s in subs]}
        return {
            "subqueries": subs or state.get("subqueries", []),
            "corrections": state.get("corrections", 0) + 1,
            "trace": [ev],
        }


async def synthesize(state: AgentState) -> dict:
    """Write the answer from accepted evidence only."""
    with NodeTimer("synthesize") as ev:
        evidence = state.get("evidence", [])
        accepted = set(state.get("accepted_evidence_ids") or [e.evidence_id for e in evidence])
        usable = [e for e in evidence if e.evidence_id in accepted][: get_settings().top_k]

        if not usable:
            ev["detail"] = {"reason": "no accepted evidence"}
            return {"draft_answer": None, "claims": [], "trace": [ev]}

        zh = state.get("answer_language") == "zh-TW"
        client = get_client()
        try:
            draft = await client.structured(
                SYNTH_SYSTEM_ZH if zh else SYNTH_SYSTEM_EN,
                synth_user(state["question"], usable),
                AgentAnswer,
            )
        except Exception as exc:  # noqa: BLE001
            ev["error"] = f"synthesis failed: {type(exc).__name__}: {exc}"
            return {"draft_answer": None, "claims": [], "llm_calls": client.calls, "trace": [ev]}

        ev["llm_calls"] = 1
        ev["detail"] = {"claims": len(draft.claims), "chars": len(draft.answer_text)}
        return {
            "draft_answer": cite.strip_urls(draft.answer_text),
            "claims": [c.model_dump() for c in draft.claims],
            "limitations": draft.limitations,
            "llm_calls": client.calls,
            "trace": [ev],
        }


async def validate_citations(state: AgentState) -> dict:
    """Deterministic gate. No LLM call, no second chance beyond a cheap repair."""
    with NodeTimer("validate_citations") as ev:
        draft = state.get("draft_answer")
        if not draft:
            ev["detail"] = {"errors": ["no draft answer"]}
            return {"citation_errors": ["no draft answer"], "outcome": "abstain", "trace": [ev]}

        claims = [Claim(**c) for c in state.get("claims", [])]
        evidence = state.get("evidence", [])
        accepted = state.get("accepted_evidence_ids") or None

        errors, cited, lines = cite.validate(claims, draft, evidence, accepted)

        repaired = False
        if errors and state.get("system") == "B3":
            claims = cite.repair_claims(claims, evidence)
            errors, cited, lines = cite.validate(claims, draft, evidence, accepted)
            repaired = True

        ev["detail"] = {
            "errors": errors,
            "cited_documents": len({(e.volume_id, e.document_id) for e in cited}),
            "repaired": repaired,
        }
        if errors:
            return {
                "citation_errors": errors,
                "outcome": "abstain",
                "claims": [c.model_dump() for c in claims],
                "trace": [ev],
            }
        return {
            "citation_errors": [],
            "citations": lines,
            "claims": [c.model_dump() for c in claims],
            "outcome": "answer",
            "trace": [ev],
        }


async def abstain(state: AgentState) -> dict:
    """Refuse rather than answer from outside FRUS. Costs no LLM call."""
    with NodeTimer("abstain") as ev:
        zh = state.get("answer_language") == "zh-TW"
        reasons = state.get("citation_errors") or state.get("missing_hops") or ["no evidence"]
        ev["detail"] = {"reasons": reasons[:5]}
        return {
            "outcome": "abstain",
            "draft_answer": ABSTAIN_ZH if zh else ABSTAIN_EN,
            "abstain_reason": "; ".join(str(r) for r in reasons[:5]),
            "citations": [],
            "trace": [ev],
        }
