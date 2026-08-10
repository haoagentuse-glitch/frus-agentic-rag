"""Entry points: the agentic graph and the 2-step baseline it must beat."""

from __future__ import annotations

import time
from typing import Any, cast

from frus_agentic_rag import observability as obs
from frus_agentic_rag.agent import citations as cite
from frus_agentic_rag.agent.graph import build_graph
from frus_agentic_rag.agent.llm import get_client
from frus_agentic_rag.agent.nodes import detect_language
from frus_agentic_rag.agent.schemas import Language
from frus_agentic_rag.agent.state import new_state
from frus_agentic_rag.config import get_settings
from frus_agentic_rag.models import Answer, Claim, Evidence


def _doc_ids(evidence) -> list[str]:
    """Chunk-level evidence collapsed to the citable unit, in stable order."""
    return sorted({f"{e.volume_id}:{e.document_id}" for e in evidence})


async def answer(
    question: str,
    language: str | None = None,
    system: str = "B3",
    thread_id: str | None = None,
    checkpointer: Any = None,
    exclude_documents: list[str] | None = None,
) -> Answer:
    language = cast(Language, language or detect_language(question))
    client = get_client()
    client.calls = 0
    obs.setup()

    t0 = time.perf_counter()
    graph = build_graph(system, checkpointer=checkpointer)
    config = {
        "recursion_limit": get_settings().budgets.recursion_limit,
        "configurable": {"thread_id": thread_id or f"{system}-{abs(hash(question))}"},
    }
    with obs.span(
        f"frus.answer.{system}", kind="AGENT", **{"frus.system": system, "frus.language": language}
    ) as root:
        obs.set_input(root, question)
        final = await graph.ainvoke(
            new_state(question, language, system, exclude_documents), config=config
        )
        obs.set_output(
            root,
            {
                "outcome": final.get("outcome"),
                "answer": (final.get("draft_answer") or "")[:2000],
                "citations": final.get("citations", []),
            },
        )
    latency = time.perf_counter() - t0

    evidence: list[Evidence] = final.get("evidence", [])
    accepted = set(final.get("accepted_evidence_ids") or [e.evidence_id for e in evidence])
    outcome = final.get("outcome") or ("answer" if final.get("draft_answer") else "abstain")

    claims = [Claim(**c) for c in final.get("claims", [])]
    by_id = {e.evidence_id: e for e in evidence}
    cited_ids = {i for c in claims for i in c.evidence_ids} if final.get("citations") else set()

    return Answer(
        question=question,
        language=language,
        outcome=outcome,  # type: ignore[arg-type]
        answer_text=final.get("draft_answer") or "",
        claims=claims,
        citations=final.get("citations", []),
        evidence=[e for e in evidence if e.evidence_id in accepted][: get_settings().top_k],
        retrieved_document_ids=_doc_ids(evidence),
        accepted_document_ids=_doc_ids(e for e in evidence if e.evidence_id in accepted),
        cited_document_ids=_doc_ids(by_id[i] for i in cited_ids if i in by_id),
        limitations=final.get("limitations", ""),
        abstain_reason=final.get("abstain_reason", ""),
        system=system,
        llm_calls=final.get("llm_calls", client.calls),
        retrieval_calls=sum(t.get("retrieval_calls", 0) for t in final.get("trace", [])),
        latency_s=round(latency, 3),
        trace=[dict(t) for t in final.get("trace", [])],
    )


async def answer_2step(question: str, language: str | None = None) -> Answer:
    """The fixed pipeline the agentic graph is measured against.

    query -> BM25 + BGE-M3 -> RRF -> top evidence -> Qwen -> citation gate.
    Deliberately kept as its own function: if the graph's B0 wiring ever drifts,
    this still shows what a non-agentic system does.
    """
    from frus_agentic_rag.agent.prompts import synth_system, synth_user
    from frus_agentic_rag.agent.schemas import AgentAnswer
    from frus_agentic_rag.models import SearchFilters
    from frus_agentic_rag.retrieval.tools import get_toolbox

    language = cast(Language, language or detect_language(question))
    settings = get_settings()
    client = get_client()
    client.calls = 0
    t0 = time.perf_counter()

    evidence = await get_toolbox().hybrid_search(
        question, SearchFilters(subtypes=["historical-document"]), settings.top_k, "main"
    )

    if not evidence:
        return Answer(
            question=question,
            language=language,
            outcome="abstain",
            abstain_reason="no evidence retrieved",
            system="B0-2step",
            latency_s=round(time.perf_counter() - t0, 3),
            retrieval_calls=1,
        )

    # The baseline gets the same evidence text the graph gets. Sentence
    # filtering is a retrieval-side stage, and letting it apply to only one arm
    # of the ablation would make the comparison about the retriever again.
    import asyncio

    from frus_agentic_rag.retrieval.focus import focus_evidence

    evidence, _ = await asyncio.to_thread(focus_evidence, question, evidence)

    try:
        draft = await client.structured(
            synth_system(language),
            synth_user(question, evidence),
            AgentAnswer,
            num_predict=settings.ollama_num_predict_synthesis,
        )
    except Exception as exc:
        return Answer(
            question=question,
            language=language,
            outcome="abstain",
            abstain_reason=f"generation failed: {exc}",
            system="B0-2step",
            llm_calls=client.calls,
            retrieval_calls=1,
            latency_s=round(time.perf_counter() - t0, 3),
        )

    claims = [Claim(text=c.text, evidence_ids=list(c.evidence_ids)) for c in draft.claims]
    # The baseline gets post-hoc attribution too. Leaving it on model-written
    # ids would make the ablation a comparison of citation schemes rather than
    # of the graph, which is the one thing it is supposed to isolate.
    if settings.citation_mode == "post_hoc":
        from frus_agentic_rag.agent.attribute import attribute

        claims, _ = await asyncio.to_thread(attribute, claims, evidence)
    text = cite.strip_urls(draft.answer_text)
    errors, cited, lines, _ = cite.validate(claims, text, evidence)

    return Answer(
        question=question,
        language=language,
        outcome="abstain" if errors else "answer",
        answer_text=text if not errors else "",
        claims=claims,
        citations=lines,
        evidence=evidence,
        retrieved_document_ids=_doc_ids(evidence),
        # No grader in the 2-step pipeline: everything retrieved is accepted.
        accepted_document_ids=_doc_ids(evidence),
        cited_document_ids=_doc_ids(cited) if not errors else [],
        limitations=draft.limitations,
        abstain_reason="; ".join(errors[:3]),
        system="B0-2step",
        llm_calls=client.calls,
        retrieval_calls=1,
        latency_s=round(time.perf_counter() - t0, 3),
    )
