"""`frus graph-smoke`: prove the graph terminates in budget on all four paths.

Gate 2 of the build order. Nothing touches the real index until this passes.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from frus_agentic_rag.agent import fakes, run
from frus_agentic_rag.config import get_settings
from frus_agentic_rag.generation import ollama_client
from frus_agentic_rag.retrieval import tools

# (name, route, subqueries, grade sequence, question, toolbox kwargs, expected outcome)
PATHS = [
    ("simple", "simple", 1, ["supported"], "When did Nixon visit China?", {}, "answer"),
    (
        "complex_multihop",
        "complex",
        3,
        ["supported"],
        "Compare the 1969 and 1972 US positions on Taiwan.",
        {},
        "answer",
    ),
    # Correction triggered by an empty first retrieval. The grader is never
    # called on round 1, so the sequence starts at the post-correction grade.
    (
        "correction_empty_retrieval",
        "simple",
        1,
        ["supported"],
        "What was said about the Shanghai Communique?",
        {"fail_first": {"h0"}},
        "answer",
    ),
    # Correction triggered by the grader: evidence came back, but it does not
    # cover the hop, so one rewrite is spent before the answer stands.
    (
        "correction_partial_grade",
        "simple",
        1,
        ["partial", "supported"],
        "Which officials attended the February 1972 meetings?",
        {},
        "answer",
    ),
    (
        "unanswerable",
        "simple",
        1,
        ["unsupported"],
        "UNANSWERABLE topic never in FRUS",
        {},
        "abstain",
    ),
]


async def _run_one(
    name: str, route: str, n_sub: int, grades: list[str], question: str, tb_kwargs: dict, expect: str
) -> dict:
    tb = fakes.FakeToolbox(**tb_kwargs)
    client = fakes.FakeClient(route=route, n_subqueries=n_sub, grade_sequence=grades)

    tools.set_toolbox(tb)  # type: ignore[arg-type]
    ollama_client._CLIENT = client  # type: ignore[assignment]
    try:
        result = await run.answer(question, language="en", system="B3")
    finally:
        tools.set_toolbox(None)
        ollama_client._CLIENT = None

    budgets = get_settings().budgets
    retrieval_rounds = sum(1 for t in result.trace if t.get("node") == "dispatch_retrieval")
    return {
        "path": name,
        "outcome": result.outcome,
        "expected": expect,
        "ok": result.outcome == expect,
        "llm_calls": client.calls,
        "retrieval_rounds": retrieval_rounds,
        "tool_calls": len(tb.calls),
        "nodes": [t["node"] for t in result.trace],
        "within_llm_budget": client.calls <= budgets.max_llm_calls,
        "within_retrieval_budget": retrieval_rounds <= budgets.max_retrieval_rounds,
        "citations": len(result.citations),
        "abstain_reason": result.abstain_reason[:120],
    }


def run_smoke(out: Path | None = None) -> dict:
    results = [asyncio.run(_run_one(*p)) for p in PATHS]
    summary = {
        "paths": results,
        "all_terminated": True,
        "all_expected_outcome": all(r["ok"] for r in results),
        "all_within_budget": all(
            r["within_llm_budget"] and r["within_retrieval_budget"] for r in results
        ),
    }
    summary["pass"] = summary["all_expected_outcome"] and summary["all_within_budget"]

    out = out or get_settings().reports_dir / "graph_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary
