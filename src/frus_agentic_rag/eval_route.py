"""Route stability under paraphrase.

Gate 6 of the build spec: ten semantic intents, three wordings each. If the
same intent routes differently depending on phrasing, the router is not making
a decision, it is reacting to surface form — and the fix is a rule-first router,
not a larger recursion limit.

Only the planner runs here. No retrieval, no synthesis: this measures routing.
"""

from __future__ import annotations

import json
import statistics
import time
from collections import Counter
from pathlib import Path

from frus_agentic_rag.agent.nodes import rule_route
from frus_agentic_rag.agent.prompts import PLANNER_SYSTEM, planner_user
from frus_agentic_rag.agent.schemas import QueryPlan
from frus_agentic_rag.config import get_settings
from frus_agentic_rag.generation.ollama_client import get_client

# (intent id, expected route, three paraphrases of the same question)
INTENTS: list[tuple[str, str, list[str]]] = [
    (
        "single-doc-lookup",
        "lookup",
        [
            "What does FRUS document frus1969-76v17 d4 say?",
            "Show me document 4 in the volume frus1969-76v17.",
            "I want to read frus1969-76v17 document 4.",
        ],
    ),
    (
        "simple-fact",
        "simple",
        [
            "Who was the US Secretary of State during the Paris Peace Conference?",
            "Which official led US diplomacy at the Paris Peace Conference?",
            "Name the Secretary of State at the time of the Paris Peace Conference.",
        ],
    ),
    (
        "timeline",
        "timeline",
        [
            "What is the chronology of US-China contacts between 1969 and 1972?",
            "Give me a timeline of American approaches to China from 1969 to 1972.",
            "In what order did US-China contacts happen between 1969 and 1972?",
        ],
    ),
    (
        "comparison",
        "complex",
        [
            "Compare the US position on Taiwan in 1969 with the position in 1972.",
            "How did the American stance on Taiwan differ between 1969 and 1972?",
            "What changed in Washington's Taiwan policy from 1969 to 1972?",
        ],
    ),
    (
        "causal",
        "complex",
        [
            "Why did the United States open contacts with Beijing?",
            "What caused Washington to begin approaching the PRC?",
            "What reasons led the US to start talking to Beijing?",
        ],
    ),
    (
        "series-status",
        "status",
        [
            "Is the FRUS volume covering 2003 published yet?",
            "Has FRUS released the volume for 2003?",
            "Which FRUS volumes covering 2003 are still planned rather than published?",
        ],
    ),
    (
        "simple-fact-zh",
        "simple",
        [
            "尼克森是哪一年訪問中國的？",
            "尼克森訪華發生在哪一年？",
            "美國總統尼克森前往中國是什麼時候？",
        ],
    ),
    (
        "comparison-zh",
        "complex",
        [
            "比較 1969 年與 1972 年美方對台灣的立場。",
            "美國對台灣的態度在 1969 到 1972 年間有什麼差異？",
            "1969 年跟 1972 年，華府的台灣政策差在哪裡？",
        ],
    ),
    (
        "timeline-zh",
        "timeline",
        [
            "1969 到 1972 年美中接觸的時間軸是什麼？",
            "請按時間順序說明 1969 至 1972 年的美中往來。",
            "美中在 1969 到 1972 年間的接觸經過為何？",
        ],
    ),
    (
        "series-status-zh",
        "status",
        [
            "FRUS 涵蓋 2003 年的卷次已經出版了嗎？",
            "2003 年那一卷 FRUS 出版了沒有？",
            "哪些涵蓋 2003 年的 FRUS 卷次還在規劃中尚未出版？",
        ],
    ),
]


async def _route_once(question: str) -> dict:
    client = get_client()
    t0 = time.perf_counter()
    try:
        plan = await client.structured(PLANNER_SYSTEM, planner_user(question), QueryPlan)
        return {
            "question": question,
            "llm_route": plan.route,
            "rule_route": rule_route(question),
            "n_subqueries": len(plan.subqueries),
            "needs_retrieval": plan.needs_retrieval,
            "latency_s": round(time.perf_counter() - t0, 2),
            "schema_valid": True,
        }
    except Exception as exc:
        return {
            "question": question,
            "llm_route": None,
            "rule_route": rule_route(question),
            "n_subqueries": 0,
            "needs_retrieval": True,
            "latency_s": round(time.perf_counter() - t0, 2),
            "schema_valid": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


async def run_route_stability(out: Path | None = None) -> dict:
    settings = get_settings()
    intents: list[dict] = []

    for intent_id, expected, paraphrases in INTENTS:
        runs = [await _route_once(q) for q in paraphrases]
        llm_routes = [r["llm_route"] for r in runs]
        rule_routes = [r["rule_route"] for r in runs]
        majority, majority_n = Counter(llm_routes).most_common(1)[0]
        intents.append(
            {
                "intent": intent_id,
                "expected_route": expected,
                "llm_routes": llm_routes,
                "rule_routes": rule_routes,
                "llm_consistent": len(set(llm_routes)) == 1,
                "rule_consistent": len(set(rule_routes)) == 1,
                "llm_correct": sum(1 for r in llm_routes if r == expected),
                "rule_correct": sum(1 for r in rule_routes if r == expected),
                "majority_route": majority,
                "majority_share": round(majority_n / len(runs), 3),
                "subquery_counts": [r["n_subqueries"] for r in runs],
                "schema_valid": sum(1 for r in runs if r["schema_valid"]),
                "runs": runs,
            }
        )
        print(
            f"{intent_id:20} expected={expected:9} llm={llm_routes} "
            f"consistent={len(set(llm_routes)) == 1}",
            flush=True,
        )

    n_intents = len(intents)
    n_runs = n_intents * 3
    llm_consistency = sum(1 for i in intents if i["llm_consistent"]) / n_intents
    rule_consistency = sum(1 for i in intents if i["rule_consistent"]) / n_intents
    llm_accuracy = sum(i["llm_correct"] for i in intents) / n_runs
    rule_accuracy = sum(i["rule_correct"] for i in intents) / n_runs
    schema_valid_rate = sum(i["schema_valid"] for i in intents) / n_runs

    # Gate 2 of the spec: a "simple" intent that gets decomposed is over-planning.
    simple_runs = [r for i in intents if i["expected_route"] == "simple" for r in i["runs"]]
    over_decomposed = sum(1 for r in simple_runs if r["n_subqueries"] > 1)

    report = {
        "n_intents": n_intents,
        "paraphrases_per_intent": 3,
        "n_runs": n_runs,
        "model": settings.ollama_model,
        "llm_route_consistency": round(llm_consistency, 3),
        "rule_route_consistency": round(rule_consistency, 3),
        "llm_route_accuracy": round(llm_accuracy, 3),
        "rule_route_accuracy": round(rule_accuracy, 3),
        "structured_output_valid_rate": round(schema_valid_rate, 3),
        "simple_intents_over_decomposed": over_decomposed,
        "simple_intents_total": len(simple_runs),
        "median_latency_s": round(
            statistics.median([r["latency_s"] for i in intents for r in i["runs"]]), 2
        ),
        "gate_6": {
            "threshold": 0.80,
            "pass": llm_consistency >= 0.80,
            "consequence": (
                "route/tool path consistency is at or above 80%; the LLM router stands"
                if llm_consistency >= 0.80
                else "consistency below 80%: switch to a deterministic rule-first router "
                "and let the LLM produce only subqueries. Do not paper over this with a "
                "larger recursion limit."
            ),
        },
        "gate_11_structured_output": {
            "threshold": 0.95,
            "pass": schema_valid_rate >= 0.95,
            "consequence": (
                "structured output is reliable"
                if schema_valid_rate >= 0.95
                else "shorten the schema/prompt, then rule-first router, then try qwen3:8b"
            ),
        },
        "intents": intents,
    }

    out = out or settings.reports_dir / "route_stability.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return {k: v for k, v in report.items() if k != "intents"}
