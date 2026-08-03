"""The typed StateGraph.

Which nodes are live depends on the system under test, so B0..B3 run through
one code path and differ only in wiring — that is what makes the ablation a
statement about the nodes rather than about two different programs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from frus_agentic_rag.agent import edges, nodes
from frus_agentic_rag.agent.state import AgentState

SYSTEMS = ("B0", "B1", "B2", "B3")


def build_graph(system: str = "B3", checkpointer: Any = None):
    if system not in SYSTEMS:
        raise ValueError(f"unknown system {system!r}; expected one of {SYSTEMS}")

    g = StateGraph(AgentState)
    g.add_node("plan_query", nodes.plan_query)
    g.add_node("dispatch_retrieval", nodes.dispatch_retrieval)
    g.add_node("grade_evidence", nodes.grade_evidence)
    g.add_node("synthesize", nodes.synthesize)
    g.add_node("abstain", nodes.abstain)

    g.add_edge(START, "plan_query")
    g.add_conditional_edges("plan_query", edges.after_plan)

    if system in ("B0", "B1"):
        # No grading and no correction: retrieve once, then write.
        g.add_edge("dispatch_retrieval", "grade_evidence")
        g.add_edge("grade_evidence", "synthesize")
    else:
        g.add_node("rewrite_missing", nodes.rewrite_missing)
        g.add_edge("dispatch_retrieval", "grade_evidence")
        g.add_conditional_edges("grade_evidence", edges.after_grade)
        g.add_edge("rewrite_missing", "dispatch_retrieval")

    if system == "B3":
        g.add_node("validate_citations", nodes.validate_citations)
        g.add_edge("synthesize", "validate_citations")
        g.add_conditional_edges(
            "validate_citations", edges.after_citations, {"__end__": END, "abstain": "abstain"}
        )
    else:
        # B0-B2 still emit a citation block, but a bad one does not block.
        g.add_node("validate_citations", nodes.validate_citations)
        g.add_edge("synthesize", "validate_citations")
        g.add_edge("validate_citations", END)

    g.add_edge("abstain", END)
    return g.compile(checkpointer=checkpointer)


def sqlite_checkpointer(path: Path | None = None):
    """SQLite thread persistence. Caller is responsible for closing it."""
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    from frus_agentic_rag.config import get_settings

    path = path or get_settings().checkpoint_db
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    return SqliteSaver(conn)


def mermaid(system: str = "B3") -> str:
    return build_graph(system).get_graph().draw_mermaid()
