"""Termination and budget bounds on fake tools — no index, no Ollama."""

from __future__ import annotations

import pytest

from frus_agentic_rag.agent import fakes, run
from frus_agentic_rag.agent.smoke import PATHS, run_smoke
from frus_agentic_rag.config import get_settings
from frus_agentic_rag.generation import ollama_client
from frus_agentic_rag.retrieval import tools


@pytest.fixture
def wired():
    """Install fakes and guarantee they are removed even on failure."""

    def _install(tb: fakes.FakeToolbox, client: fakes.FakeClient) -> None:
        tools.set_toolbox(tb)  # type: ignore[arg-type]
        ollama_client._CLIENT = client  # type: ignore[assignment]

    yield _install
    tools.set_toolbox(None)
    ollama_client._CLIENT = None


@pytest.mark.parametrize(
    ("name", "route", "n_sub", "grades", "question", "tb_kwargs", "expect"),
    PATHS,
    ids=[p[0] for p in PATHS],
)
async def test_path_terminates_in_budget(
    wired, name, route, n_sub, grades, question, tb_kwargs, expect
):
    tb = fakes.FakeToolbox(**tb_kwargs)
    client = fakes.FakeClient(route=route, n_subqueries=n_sub, grade_sequence=grades)
    wired(tb, client)

    result = await run.answer(question, language="en", system="B3")
    budgets = get_settings().budgets

    assert result.outcome == expect
    assert client.calls <= budgets.max_llm_calls
    rounds = sum(1 for t in result.trace if t["node"] == "dispatch_retrieval")
    assert rounds <= budgets.max_retrieval_rounds


async def test_complex_plan_never_exceeds_three_subqueries(wired):
    tb = fakes.FakeToolbox()
    client = fakes.FakeClient(route="complex", n_subqueries=3)
    wired(tb, client)

    await run.answer("compare A and B", language="en", system="B3")
    searches = [c for c in tb.calls if c[0] == "hybrid_search"]
    assert len(searches) <= get_settings().budgets.max_subqueries


async def test_correction_runs_at_most_once(wired):
    tb = fakes.FakeToolbox(fail_first={"h0"})
    # Never satisfied: the budget, not the grader, has to stop this.
    client = fakes.FakeClient(grade_sequence=["partial"])
    wired(tb, client)

    result = await run.answer("a question", language="en", system="B3")
    rewrites = sum(1 for t in result.trace if t["node"] == "rewrite_missing")
    assert rewrites <= get_settings().budgets.max_corrections
    assert result.outcome in ("answer", "abstain")


async def test_invented_citation_forces_abstain(wired):
    tb = fakes.FakeToolbox()
    client = fakes.FakeClient(claim_ids=["frus1958-60v01:d999:0"])
    wired(tb, client)

    result = await run.answer("a question", language="en", system="B3")
    assert result.outcome == "abstain"
    assert not result.citations


async def test_b0_skips_planning_and_grading(wired):
    tb = fakes.FakeToolbox()
    client = fakes.FakeClient()
    wired(tb, client)

    result = await run.answer("a question", language="en", system="B0")
    nodes = [t["node"] for t in result.trace]
    assert "rewrite_missing" not in nodes
    assert "plan" not in client.seen  # B0 never calls the planner
    assert "grade" not in client.seen


def test_smoke_report_passes(tmp_path):
    summary = run_smoke(out=tmp_path / "graph_smoke.json")
    assert summary["pass"], summary
    assert summary["all_within_budget"]
