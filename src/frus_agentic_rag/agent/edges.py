"""Conditional edges. Every branch is bounded by an explicit budget check."""

from __future__ import annotations

from typing import Literal

from frus_agentic_rag.agent.state import AgentState
from frus_agentic_rag.config import get_settings


def after_plan(state: AgentState) -> Literal["dispatch_retrieval", "synthesize"]:
    # Only greetings and how-to-use questions may bypass the corpus.
    return "dispatch_retrieval" if state.get("needs_retrieval", True) else "synthesize"


def after_grade(
    state: AgentState,
) -> Literal["synthesize", "rewrite_missing", "abstain"]:
    budgets = get_settings().budgets
    grades = state.get("hop_grades", [])
    missing = [m for m in state.get("missing_hops", []) if m]
    evidence = state.get("evidence", [])

    supported = not grades or all(g.verdict == "supported" for g in grades)
    if supported and evidence:
        return "synthesize"

    exhausted = (
        state.get("retrieval_rounds", 0) >= budgets.max_retrieval_rounds
        or state.get("corrections", 0) >= budgets.max_corrections
        or state.get("llm_calls", 0) >= budgets.max_llm_calls - 1
    )

    if missing and not exhausted:
        return "rewrite_missing"

    # Out of budget: answer on partial evidence rather than throw it away, but
    # only if something was actually accepted. The citation gate still applies.
    if state.get("accepted_evidence_ids"):
        return "synthesize"
    return "abstain"


def after_citations(state: AgentState) -> Literal["__end__", "abstain"]:
    return "abstain" if state.get("outcome") == "abstain" else "__end__"
