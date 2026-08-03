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


async def answer(
    question: str,
    language: str | None = None,
    system: str = "B3",
    thread_id: str | None = None,
    checkpointer: Any = None,
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
        final = await graph.ainvoke(new_state(question, language, system), config=config)
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

    return Answer(
        question=question,
        language=language,
        outcome=outcome,  # type: ignore[arg-type]
        answer_text=final.get("draft_answer") or "",
        claims=[Claim(**c) for c in final.get("claims", [])],
        citations=final.get("citations", []),
        evidence=[e for e in evidence if e.evidence_id in accepted][: get_settings().top_k],
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
    from frus_agentic_rag.agent.prompts import SYNTH_SYSTEM_EN, SYNTH_SYSTEM_ZH, synth_user
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

    zh = language == "zh-TW"
    try:
        draft = await client.structured(
            SYNTH_SYSTEM_ZH if zh else SYNTH_SYSTEM_EN, synth_user(question, evidence), AgentAnswer
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

    claims = [Claim(text=c.text, evidence_ids=c.evidence_ids) for c in draft.claims]
    text = cite.strip_urls(draft.answer_text)
    errors, _cited, lines = cite.validate(claims, text, evidence)

    return Answer(
        question=question,
        language=language,
        outcome="abstain" if errors else "answer",
        answer_text=text if not errors else "",
        claims=claims,
        citations=lines,
        evidence=evidence,
        limitations=draft.limitations,
        abstain_reason="; ".join(errors[:3]),
        system="B0-2step",
        llm_calls=client.calls,
        retrieval_calls=1,
        latency_s=round(time.perf_counter() - t0, 3),
    )
