"""Graph nodes. Each returns a state delta and appends exactly one trace event."""

from __future__ import annotations

import asyncio
import re

from frus_agentic_rag.agent import citations as cite
from frus_agentic_rag.agent.llm import StructuredOutputError, get_client
from frus_agentic_rag.agent.prompts import (
    ABSTAIN_EN,
    ABSTAIN_ZH,
    GRADER_MAX_EVIDENCE,
    GRADER_SYSTEM,
    PLANNER_SYSTEM,
    grader_user,
    planner_user,
    synth_system,
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
from frus_agentic_rag.models import Claim, SearchFilters
from frus_agentic_rag.retrieval.select import score_rows
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
        except (StructuredOutputError, Exception) as exc:
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


def _filters(sq: SubQuery, exclude: list[str] | None = None) -> tuple[SearchFilters, dict]:
    """Typed filters from a planned subquery, with the model's guesses checked.

    The planner has no way to know FRUS volume ids and invents them; an
    unresolvable id passed into the store matches nothing and empties the
    retrieval without raising. Everything it supplies is therefore resolved
    against the manifest or dropped, and what was dropped is returned so the
    trace can show it.
    """
    from frus_agentic_rag.corpus.manifest import resolve_volume_ids, valid_iso_date

    volumes, dropped_volumes = resolve_volume_ids(sq.volume_ids)
    date_from, date_to = valid_iso_date(sq.date_from), valid_iso_date(sq.date_to)
    dropped_dates = [
        d for d, kept in ((sq.date_from, date_from), (sq.date_to, date_to)) if d and not kept
    ]
    rejected = {}
    if dropped_volumes:
        rejected["volume_ids"] = dropped_volumes
    if dropped_dates:
        rejected["dates"] = dropped_dates
    return (
        SearchFilters(
            volume_ids=volumes,
            date_from=date_from,
            date_to=date_to,
            persons=sq.persons,
            subtypes=["historical-document"],
            exclude_documents=list(exclude or []),
        ),
        rejected,
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

        rejected_filters: list[dict] = []
        relaxed: list[str] = []

        async def run(sq: SubQuery) -> list:
            if route == "lookup":
                m = re.search(r"(frus[\w\-]+)\D+(\d+)", sq.query, re.I)
                if m:
                    hits = await tb.lookup_document(m.group(1), f"d{m.group(2)}")
                    # The direct lookup path does not go through SearchFilters,
                    # so it would hand back a document the run is withholding.
                    withheld = set(state.get("exclude_documents") or [])
                    hits = [h for h in hits if f"{h.volume_id}:{h.document_id}" not in withheld]
                    if hits:
                        return hits
            filters, rejected = _filters(sq, state.get("exclude_documents"))
            if rejected:
                rejected_filters.append({"hop": sq.hop_id, **rejected})

            # With several hops the cross-encoder runs once over the merged
            # union instead, against the question the user actually asked.
            per_hop_k = top_k if len(subs) == 1 else settings.rerank_candidates
            if route == "timeline":
                hits = await tb.timeline_search(
                    sq.query, filters.date_from, filters.date_to, filters, per_hop_k, sq.hop_id
                )
            else:
                hits = await tb.hybrid_search(sq.query, filters, per_hop_k, sq.hop_id)

            # A filter that removes every candidate is worse than no filter: the
            # question is still answerable, the constraint was just wrong.
            if not hits and (filters.volume_ids or filters.date_from or filters.date_to):
                relaxed.append(sq.hop_id)
                hits = await tb.hybrid_search(
                    sq.query,
                    SearchFilters(
                        subtypes=["historical-document"],
                        exclude_documents=list(state.get("exclude_documents") or []),
                    ),
                    top_k,
                    sq.hop_id,
                )
            return hits

        results = await asyncio.gather(*(run(s) for s in subs), return_exceptions=True)
        fresh = []
        errors = []
        for sq, r in zip(subs, results, strict=True):
            if isinstance(r, BaseException):
                errors.append(f"{sq.hop_id}: {type(r).__name__}: {r}")
            else:
                fresh.extend(r)

        merged = _merge(state.get("evidence", []), fresh)
        reranked = False
        if len(subs) > 1 and len(merged) > top_k:
            from frus_agentic_rag.retrieval import rerank as rr
            from frus_agentic_rag.retrieval.hybrid import rerank_union

            merged = await asyncio.to_thread(
                rerank_union, state["question"], merged, settings.evidence_after_rerank
            )
            # Truncation alone changes the length, so length cannot be the
            # signal: only a cross-encoder rank means the reordering happened.
            reranked = rr.available() and any(e.rank_rerank is not None for e in merged)

        if settings.context_expand_neighbours:
            merged = await _expand_neighbours(tb, merged, settings.context_expand_neighbours)

        # After ordering, before anything reads the text. The grader's 700-char
        # window and the synthesiser's 950 then spend their budget on the
        # sentences that matter rather than on whatever opened the passage.
        from frus_agentic_rag.retrieval.focus import focus_evidence

        merged, focus_stats = await asyncio.to_thread(focus_evidence, state["question"], merged)

        ev["retrieval_calls"] = len(subs)
        ev["route"] = route
        ev["detail"] = {
            "tool": {"timeline": "timeline_search", "lookup": "lookup_document"}.get(
                route, "hybrid_search"
            ),
            "hops": [s.hop_id for s in subs],
            "hits": len(fresh),
            "errors": errors,
            "rejected_filters": rejected_filters,
            "relaxed_hops": relaxed,
            "union_reranked": reranked,
            "evidence_after_merge": len(merged),
            "sentence_focus": focus_stats,
        }
        return {
            "evidence": merged,
            "retrieval_rounds": state.get("retrieval_rounds", 0) + 1,
            "trace": [ev],
        }


async def _expand_neighbours(tb, evidence: list, width: int) -> list:
    """Add each passage's neighbours from the same document, keeping order.

    The retrieval unit is a 512-token chunk and the citable unit is a document;
    a fact can sit one chunk away from the passage that matched. Neighbours
    inherit the score of the chunk that pulled them in so ranking is unchanged,
    and they are inserted next to it so the model reads the document in order.
    """
    out: list = []
    seen: set[str] = {e.evidence_id for e in evidence}
    for e in evidence:
        out.append(e)
        try:
            adj = await tb.get_adjacent_context(e.evidence_id)
        except Exception:
            continue
        for a in adj[: width * 2]:
            if a.evidence_id in seen:
                continue
            seen.add(a.evidence_id)
            # Just below its anchor: a neighbour is context, not a hit.
            out.append(a.model_copy(update={"score": e.score - 1e-6, "hop": e.hop}))
    return out


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

        if get_settings().grader_mode == "score":
            # No LLM call. The cross-encoder has already read every passage
            # against the query; thresholding that score is the same judgement
            # without a generation step that can misname an id or run out of
            # window. Costs nothing: the scores are a by-product of reranking.
            from frus_agentic_rag.retrieval.select import missing_hops, select_evidence

            accepted_ids, record = select_evidence(evidence)
            gaps = missing_hops(evidence, state.get("subqueries", []))
            ev["detail"] = {
                **record,
                "overall": "partial" if gaps else "supported",
                "scores": score_rows(evidence),
            }
            return {
                "hop_grades": [],
                "accepted_evidence_ids": accepted_ids,
                "missing_hops": gaps,
                "trace": [ev],
            }

        hops = state.get("required_evidence") or [s.query for s in state.get("subqueries", [])]
        # What the grader is shown, and therefore the only evidence its verdict
        # may remove. Everything past this window is unjudged, not rejected.
        shown = evidence[:GRADER_MAX_EVIDENCE]
        unseen = [e.evidence_id for e in evidence[GRADER_MAX_EVIDENCE:]]
        client = get_client()
        try:
            grade = await client.structured(
                GRADER_SYSTEM,
                grader_user(state["question"], hops, shown),
                EvidenceGrade,
            )
        except Exception as exc:
            # A failed grader judged nothing, so it may remove nothing. Keeping
            # the top five was a quiet narrowing that looked like a judgement:
            # the run then abstained on a citation the grader never rejected.
            ev["error"] = f"grader fallback: {type(exc).__name__}: {exc}"
            ev["detail"] = {"overall": "partial", "grader": "fallback keep-all"}
            return {
                "accepted_evidence_ids": [e.evidence_id for e in evidence],
                "hop_grades": [HopGrade(hop_id="all", verdict="partial")],
                "missing_hops": [],
                "llm_calls": state.get("llm_calls", 0) + 1,
                "trace": [ev],
            }

        valid = {e.evidence_id for e in evidence}
        shown_ids = [e.evidence_id for e in shown]
        missing: list[str] = []
        hallucinated: list[str] = []
        # A passage is dropped only when EVERY hop rejected it. The accept-list
        # version took the union of what the hops kept, so the dual is the
        # intersection of what they reject: on a multi-hop question a passage
        # that answers hop B is off-topic for hop A, and letting hop A alone
        # delete it is how the plan's own evidence disappeared.
        rejections: list[set[str]] = []
        for hop in grade.hops:
            hallucinated.extend(i for i in hop.rejected_evidence_ids if i not in valid)
            rejections.append({i for i in hop.rejected_evidence_ids if i in valid})
            if hop.verdict != "supported":
                missing.append(hop.corrective_query or hop.hop_id)
        rejected = set.intersection(*rejections) if rejections else set()

        # Unjudged evidence is kept. Treating "not named by the grader" as
        # "rejected" silently discarded whatever fell outside its window: the
        # agent held 17 documents while the grader saw 6, and the other 11 were
        # dropped without anything having judged them.
        accepted: list[str] = [i for i in shown_ids if i not in rejected] + list(unseen)

        ev["llm_calls"] = 1
        ev["detail"] = {
            "overall": grade.overall,
            "hops": [h.model_dump() for h in grade.hops],
            "shown_to_grader": len(shown),
            "unjudged_kept": len(unseen),
            "rejected_by_all_hops": sorted(rejected),
            "rejected_by_some_hop": sorted(set().union(*rejections) - rejected)
            if rejections
            else [],
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
    """Re-query only the uncovered hops, keeping the original constraints.

    The corrective query replaces the wording, not the scope. Building the retry
    from the query text alone dropped the plan's date and person filters: a
    question scoped to 1871-1873 came back on the second round with documents
    from 1874 and 1888, and the grader accepted them.
    """
    with NodeTimer("rewrite_missing") as ev:
        missing = state.get("missing_hops", [])[: get_settings().budgets.max_subqueries]
        original = state.get("subqueries") or []
        subs = []
        for i, q in enumerate(missing):
            if not q or not q.strip():
                continue
            # Inherit from the hop being corrected where possible, else the first.
            src = original[i] if i < len(original) else (original[0] if original else None)
            subs.append(
                SubQuery(
                    hop_id=f"fix{i}",
                    query=q,
                    date_from=src.date_from if src else None,
                    date_to=src.date_to if src else None,
                    persons=list(src.persons) if src else [],
                    volume_ids=list(src.volume_ids) if src else [],
                )
            )
        ev["detail"] = {
            "rewritten": [s.query for s in subs],
            "constraints_kept": [
                {"hop": s.hop_id, "dates": [s.date_from, s.date_to], "persons": s.persons}
                for s in subs
                if s.date_from or s.date_to or s.persons
            ],
        }
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
        # No top_k cut here: select_synthesis_evidence budgets the window by
        # document, and cutting to top_k chunks first is what starved it.
        usable = [e for e in evidence if e.evidence_id in accepted]

        if not usable:
            ev["detail"] = {"reason": "no accepted evidence"}
            return {"draft_answer": None, "claims": [], "trace": [ev]}

        client = get_client()
        try:
            draft = await client.structured(
                synth_system(state.get("answer_language") or "en"),
                synth_user(state["question"], usable),
                AgentAnswer,
                num_predict=get_settings().ollama_num_predict_synthesis,
            )
        except Exception as exc:
            ev["error"] = f"synthesis failed: {type(exc).__name__}: {exc}"
            return {"draft_answer": None, "claims": [], "llm_calls": client.calls, "trace": [ev]}

        claims = [Claim(text=c.text, evidence_ids=list(c.evidence_ids)) for c in draft.claims]
        attribution: dict = {"mode": "model"}
        if get_settings().citation_mode == "post_hoc":
            # The ids the model wrote are discarded here. It was never asked to
            # produce usable ones and, measured, could not.
            from frus_agentic_rag.agent.attribute import attribute

            claims, attribution = await asyncio.to_thread(attribute, claims, usable)

        ev["llm_calls"] = 1
        ev["detail"] = {
            "claims": len(claims),
            "chars": len(draft.answer_text),
            "attribution": attribution,
        }
        return {
            "draft_answer": cite.strip_urls(draft.answer_text),
            "claims": [c.model_dump() for c in claims],
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
        # Pass the list through as-is: grade_evidence always populates it (with
        # every id, for the ungraded systems), so an empty list here genuinely
        # means nothing was accepted.
        accepted = state.get("accepted_evidence_ids")

        errors, cited, lines, reasons = cite.validate(claims, draft, evidence, accepted)

        repaired = False
        if errors and state.get("system") == "B3":
            claims = cite.repair_claims(claims, evidence)
            errors, cited, lines, reasons = cite.validate(claims, draft, evidence, accepted)
            repaired = True

        # Uncited claims are removed from the answer rather than left in it:
        # a sentence with no citation would otherwise ship inside a cited
        # answer, which is precisely the thing the gate exists to prevent.
        surviving = cite.drop_uncited(claims, cited)
        ev["detail"] = {
            "errors": errors,
            "cited_documents": len({(e.volume_id, e.document_id) for e in cited}),
            "repaired": repaired,
            "claims_kept": len(surviving),
            "claims_dropped": len(claims) - len(surviving),
            "reasons": reasons[:8],
        }
        if errors:
            return {
                "citation_errors": errors,
                "outcome": "abstain",
                "claims": [c.model_dump() for c in claims],
                "trace": [ev],
            }
        claims = surviving
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
